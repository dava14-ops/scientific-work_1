"""
Locked analysis protocol v1.2 (CRITICAL FIX)
Applicability of an area-velocity draft resistance relation for cultivator
sweeps: 28 field runs (open dataset, Nagy & Tamas, Data in Brief, 2026).

CHANGES FROM v1.1:
- SCRS-CV rewritten to eliminate train/test data leakage. Training size is
  kept equal to (N - n_g); for each test observation in the held-out group a
  fresh random training set is drawn from all rows EXCEPT that observation.
- Data-consistency check: recompute q* = F/(w*d) from raw columns and compare
  with the published specific_resistance_kPa.
- Preprocessing hardened: NaN/Inf handling for all numeric fields and
  categorical columns BEFORE casting.
- Q1 partial F-test: survival-function p-value for numerical stability and an
  optional HC3 robust Wald test reported alongside the classic test.
- Paired bootstrap: relative RMSE change added; interpretation caveats noted.
- AICc convention explicitly documented (K = number of regressors including
  the constant, sigma^2 profiled out, matching statsmodels OLS).

FREEZE THIS FILE BEFORE inspecting final coefficient values.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="statsmodels")
warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.diagnostic import het_breuschpagan
except ImportError as err:
    sys.exit(f"Missing dependency: {err}")
try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

# ------------------------------------------------------------------ CONFIG --
CFG = {
    "data_file": "field_runs_28_summary.csv",
    "col_qstar": "specific_resistance_kPa",
    "col_speed_primary": "mean_speed_ms",
    "col_speed_sensitivity": "nominal_speed_ms",
    "col_tool": "tool_size",
    "col_campaign": "campaign",
    "col_force": "force_mean_N",
    "col_width": "width_mm",
    "col_depth": "depth_mm",
    "campaign_order": ["Dry", "Medium", "Wet"],
    "seed": 20261018,
    "n_boot": 5000,
    "n_scrs_reps": 200,
    "alpha": 0.05,
    "out_dir": "results",
}
LOG: list[str] = []
def log(msg: str) -> None:
    LOG.append(msg); print(msg)

# ------------------------------------------------------ DATA CONSISTENCY ---
def verify_qstar(df_raw: pd.DataFrame) -> dict:
    """
    Reconstruct q* from raw columns and compare with the published column.
    Assumes: F in newtons, w and d in millimetres, q* in kPa, so that
        q*_kPa = 1000 * F_N / (w_mm * d_mm).
    If any column is missing the check is skipped gracefully.
    """
    need = [CFG["col_qstar"], CFG["col_force"], CFG["col_width"], CFG["col_depth"]]
    absent = [c for c in need if c not in df_raw.columns]
    if absent:
        log(f"  q* consistency check skipped, columns absent: {absent}")
        return {"status": "skipped", "absent": absent}

    d = df_raw[need].copy()
    for c in need:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    d = d[(d[CFG["col_force"]] > 0) & (d[CFG["col_width"]] > 0) & (d[CFG["col_depth"]] > 0)]
    if len(d) < 3:
        log("  q* consistency check skipped: too few valid rows")
        return {"status": "insufficient_rows"}

    recomputed = 1000.0 * d[CFG["col_force"]] / (d[CFG["col_width"]] * d[CFG["col_depth"]])
    published = d[CFG["col_qstar"]]

    rel_err = np.abs(recomputed - published) / np.abs(published)
    corr = float(np.corrcoef(recomputed, published)[0, 1])
    out = {
        "status": "ok",
        "n": int(len(d)),
        "corr": corr,
        "max_rel_err": float(rel_err.max()),
        "mean_rel_err": float(rel_err.mean()),
        "convention": "q*_kPa = 1000 * F_N / (w_mm * d_mm)",
    }
    log(f"  q* consistency: corr={corr:.6f}, max_rel_err={out['max_rel_err']:.4e}")
    if out["max_rel_err"] > 1e-3:
        log("  WARNING: recomputed q* differs from published column by >0.1%.")
        log("           Confirm the dataset's own definition of specific resistance.")
    return out

# ------------------------------------------------------------- PREPROCESS --
def prepare(df_raw: pd.DataFrame, speed_col: str):
    need = [CFG["col_qstar"], speed_col, CFG["col_tool"], CFG["col_campaign"]]
    miss = [c for c in need if c not in df_raw.columns]
    if miss:
        raise KeyError(f"Columns absent from CSV: {miss}")

    d = df_raw.copy().reset_index(drop=True)

    # Numeric columns: coerce then drop rows with missing OR non-finite values.
    d["q_star"] = pd.to_numeric(d[CFG["col_qstar"]], errors="coerce")
    d["v"]      = pd.to_numeric(d[speed_col], errors="coerce")
    finite_mask = np.isfinite(d["q_star"]) & np.isfinite(d["v"])
    if (~finite_mask).any():
        log(f"  ! {int((~finite_mask).sum())} row(s) dropped: non-finite q* or speed")
        d = d.loc[finite_mask].reset_index(drop=True)
    d["v2"] = d["v"] ** 2

    # Categorical columns: inspect for missing values BEFORE string cast so a
    # missing value never silently becomes the literal category "nan".
    for col, label in ((CFG["col_tool"], "tool"), (CFG["col_campaign"], "camp")):
        n_missing = int(d[col].isna().sum())
        if n_missing:
            log(f"  ! {n_missing} row(s) dropped: missing {label}")
            d = d.loc[d[col].notna()].reset_index(drop=True)
        d[label] = d[col].astype(str).str.strip()
        empty = d[label] == ""
        if empty.any():
            log(f"  ! {int(empty.sum())} row(s) dropped: empty {label}")
            d = d.loc[~empty].reset_index(drop=True)

    tools = sorted(d["tool"].unique(), key=lambda s: (len(s), s))
    order = [c for c in CFG["campaign_order"] if c in set(d["camp"])]
    camps = order + sorted(set(d["camp"]) - set(order))
    d["tool"] = pd.Categorical(d["tool"], categories=tools)
    d["camp"] = pd.Categorical(d["camp"], categories=camps)
    return d, tools[0], camps[0]

# --------------------------------------------------------------- MODELS ----
def build_models(ref_tool: str, ref_camp: str) -> dict:
    T = f"C(tool, Treatment(reference='{ref_tool}'))"
    Cp = f"C(camp, Treatment(reference='{ref_camp}'))"
    return {
        "M0":  {"formula": "q_star ~ 1",               "groups": ()},
        "M1":  {"formula": "q_star ~ v2",               "groups": ()},
        "M1g": {"formula": f"q_star ~ v2 + {T}",        "groups": ("tool",)},
        "M1c": {"formula": f"q_star ~ v2 + {Cp}",       "groups": ("camp",)},
        "M2":  {"formula": f"q_star ~ v2 + {T} + {Cp}", "groups": ("tool", "camp")},
        "M2g": {"formula": f"q_star ~ {T} + {Cp}",      "groups": ("tool", "camp")},
    }

def fit_predict(formula, train, test, cat_cols, ctx=""):
    for c in cat_cols:
        tr = set(train[c].astype(str).unique())
        te = set(test[c].astype(str).unique())
        if not te <= tr:
            raise ValueError(
                f"{ctx}: '{formula}' carries a fixed effect for '{c}', but held-out "
                f"data contain unseen level(s) {sorted(te - tr)}. "
                f"Use a model without '{c}' for this validation."
            )
    model = smf.ols(formula, data=train).fit()
    return model, model.predict(test).to_numpy(dtype=float)

def metrics(errors) -> dict:
    err = np.asarray(errors, dtype=float)
    return {"n": int(err.size),
            "RMSE_kPa": float(np.sqrt(np.mean(err ** 2))),
            "MAE_kPa":  float(np.mean(np.abs(err))),
            "Bias_kPa": float(np.mean(err))}

# ------------------------------------------------------------ VALIDATION ---
def run_loocv(df, name, spec) -> pd.Series:
    err = np.empty(len(df))
    for i in range(len(df)):
        _, p = fit_predict(spec["formula"], df.drop(index=i), df.iloc[[i]],
                           spec["groups"], ctx=f"LOOCV/{name}/i={i}")
        err[i] = df["q_star"].iloc[i] - p[0]
    return pd.Series(err, index=df.index, name=name)

def run_group_out(df, name, spec, group_var):
    if group_var in spec["groups"]:
        log(f"  - {name}: ineligible for leave-one-{group_var}-out -> skipped")
        return None
    err = pd.Series(np.nan, index=df.index, name=name)
    for g, idx in df.groupby(group_var, observed=True).groups.items():
        idx = list(idx)
        _, p = fit_predict(spec["formula"], df.drop(index=idx), df.loc[idx],
                           spec["groups"], ctx=f"LO{group_var}/{name}/{g}")
        err.loc[idx] = df.loc[idx, "q_star"].to_numpy() - p
    return err

# -------------------------------------------------------- SCRS-CV (FIXED) --
def run_scrs_cv(df, name, spec, group_var, n_reps=200, seed=None):
    """
    Size-matched random control WITHOUT train/test overlap.

    For each held-out group g of size n_g the training size is kept equal to
    (N - n_g), i.e. identical to the full group-out training size. For every
    test observation t in g a fresh random training set is drawn from all rows
    EXCEPT t, so the test row can never appear in its own training set. The
    training set is required to contain all categorical levels the model uses.

    Each replicate fits one model per test observation, so a group of size m
    produces m fits per replicate. Compare the resulting RMSE distribution
    against the FULL group-out error for the SAME observations, not against
    the overall LOOCV RMSE across all 28 runs.
    """
    rng = np.random.default_rng(seed)
    groups = df.groupby(group_var, observed=True).groups
    n_total = len(df)
    summary = []

    for g, idx in groups.items():
        target_idx = list(idx)
        n_g = len(target_idx)
        n_train = n_total - n_g  # size-matched to full group-out

        rep_rmses = []
        successful = 0
        max_attempts = n_reps * 5

        for attempt in range(max_attempts):
            if successful >= n_reps:
                break

            errors = []
            rep_ok = True

            for t in target_idx:
                # Pool excludes the current test row to guarantee no overlap.
                pool = df.index.drop(t)
                if n_train > len(pool):
                    rep_ok = False
                    break

                sample = rng.choice(pool, size=n_train, replace=False)
                train = df.loc[sample]
                test = df.loc[[t]]

                # All categorical levels used by the model must be present.
                levels_ok = True
                for c in spec["groups"]:
                    if set(test[c].astype(str)) - set(train[c].astype(str)):
                        levels_ok = False
                        break
                if not levels_ok:
                    rep_ok = False
                    break

                try:
                    _, p = fit_predict(spec["formula"], train, test,
                                       spec["groups"],
                                       ctx=f"SCRS/{name}/{g}/rep{attempt}/t{t}")
                    errors.append(float(test["q_star"].iloc[0]) - p[0])
                except Exception:
                    rep_ok = False
                    break

            if rep_ok and len(errors) == n_g:
                rep_rmses.append(float(np.sqrt(np.mean(np.asarray(errors) ** 2))))
                successful += 1

        if successful < n_reps:
            log(f"  WARNING: SCRS for {name}/{g} achieved {successful}/{n_reps} reps")

        if rep_rmses:
            summary.append({
                "group": g,
                "n_reps": successful,
                "n_train_per_rep": n_train,
                "RMSE_mean": float(np.mean(rep_rmses)),
                "RMSE_std": float(np.std(rep_rmses, ddof=1)),
                "RMSE_median": float(np.median(rep_rmses)),
            })
    return summary

# ------------------------------------------------------------- BOOTSTRAP ---
def paired_boot_delta(err_a, err_b, n_boot, seed, alpha):
    """
    Paired bootstrap of the RMSE difference on the SAME test units.

    NOTE: this resamples already-computed CV errors; it does NOT refit models
    in each replicate. The interval therefore reflects uncertainty of the
    paired error comparison under this resampling scheme, not the full
    uncertainty of model training and selection.
    """
    a = np.asarray(err_a, float)
    b = np.asarray(err_b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    a, b = a[m], b[m]
    n = a.size
    if n < 4:
        return None

    rmse_a = float(np.sqrt(np.mean(a ** 2)))
    rmse_b = float(np.sqrt(np.mean(b ** 2)))
    obs = rmse_a - rmse_b
    rel_pct = 100.0 * obs / rmse_b if rmse_b > 0 else float("nan")

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    d = np.sqrt(np.mean(a[idx] ** 2, axis=1)) - np.sqrt(np.mean(b[idx] ** 2, axis=1))
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"n_paired": int(n), "delta_RMSE_A_minus_B": obs,
            "rel_change_pct": rel_pct,
            "ci_low": float(lo), "ci_high": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0),
            "favours": ("B" if lo > 0 else "A" if hi < 0 else "neither")}

# ------------------------------------------------------- FULL-DATA FITS ----
def model_table(df, name, spec, alpha):
    m = smf.ols(spec["formula"], data=df).fit()
    n = int(m.nobs)
    # AICc convention: K = number of regressors INCLUDING the constant but
    # excluding the profiled-out sigma^2. This matches statsmodels' OLS `aic`
    # definition (model.df_modelwc). Do not mix K definitions across models.
    K = int(getattr(m.model, "df_modelwc", len(m.params)))
    denom = n - K - 1
    aicc = float(m.aic) + (2 * K * (K + 1)) / denom if denom > 0 else float("nan")
    ci = m.conf_int(alpha=alpha)
    tab = pd.DataFrame({"term": m.params.index, "coef": m.params.values,
                        "std_err": m.bse.values, "ci_low": ci[0].values,
                        "ci_high": ci[1].values, "p_value": m.pvalues.values})
    tab.insert(0, "model", name)
    info = {"model": name, "n": n, "K": K, "R2": float(m.rsquared),
            "R2_adj": float(m.rsquared_adj), "AIC": float(m.aic), "AICc": aicc,
            "resid_SD_kPa": float(np.sqrt(m.mse_resid))}
    res = m.resid
    info["shapiro_p"] = float(sps.shapiro(res).pvalue) if (HAVE_SCIPY and n >= 8) else None
    try:
        bp = het_breuschpagan(res, m.model.exog)[1]
        info["bp_p"] = float(bp) if bp is not None and np.isfinite(bp) else None
    except Exception:
        info["bp_p"] = None
    fitted = m.fittedvalues.to_numpy(dtype=float)
    if np.std(fitted) > 0 and np.std(res) > 0:
        info["corr_resid_fitted"] = float(np.corrcoef(res, fitted)[0, 1])
    else:
        info["corr_resid_fitted"] = None
    return tab, info

# ------------------------------------------ Q1: PARTIAL F-TEST (HARDENED) --
def q1_diagnostic(df, alpha):
    """
    Partial F-test for whether tool effects remain after accounting for v2,
    i.e. whether the q* = F/(w*d) normalisation removes tool differences.

    The classic F-test assumes homoscedastic independent errors. An HC3 robust
    Wald test of the joint tool effects is reported alongside it; the two are
    presented separately, never silently substituted.
    """
    log("\n=== Q1 DIAGNOSTIC: Partial F-test (M1 vs M1+tool) ===")
    m1 = smf.ols("q_star ~ v2", data=df).fit()
    m1_tool = smf.ols("q_star ~ v2 + C(tool)", data=df).fit()

    df_diff = m1_tool.df_resid - m1.df_resid
    if df_diff >= 0:
        log("  ERROR: M1 not nested in M1+tool; aborting diagnostic")
        return None
    df_diff = -df_diff  # = 2 for three tools

    sse_diff = m1.ssr - m1_tool.ssr
    f_stat = (sse_diff / df_diff) / (m1_tool.ssr / m1_tool.df_resid)
    # Survival function is numerically preferable to 1 - cdf for tiny p-values.
    p_value = float(sps.f.sf(f_stat, df_diff, m1_tool.df_resid)) if HAVE_SCIPY else None

    p_str = f"{p_value:.4g}" if p_value is not None else "None"
    log(f"  classic F({df_diff},{int(m1_tool.df_resid)})={f_stat:.3f}, p={p_str}")

    # --- Robust joint Wald test of tool effects (HC3), reported separately ---
    robust = None
    try:
        m1_tool_rob = smf.ols("q_star ~ v2 + C(tool)", data=df).fit(cov_type="HC3")
        tool_terms = [t for t in m1_tool_rob.params.index if "tool" in t]
        if tool_terms and HAVE_SCIPY:
            # Build restriction: all tool coefficients equal zero.
            n_terms = len(m1_tool_rob.params)
            R = np.zeros((len(tool_terms), n_terms))
            for i, term in enumerate(tool_terms):
                R[i, list(m1_tool_rob.params.index).index(term)] = 1.0
            wald = m1_tool_rob.wald_test(R)
            robust = {
                "statistic": float(wald.statistic),
                "p_value": float(wald.pvalue),
                "df": int(wald.df_denom) if hasattr(wald, "df_denom") else len(tool_terms),
                "cov_type": "HC3",
            }
            log(f"  robust  Wald(HC3) joint tool test: chi2={robust['statistic']:.3f}, "
                f"p={robust['p_value']:.4g}")
    except Exception as exc:
        log(f"  robust Wald test unavailable: {exc}")

    return {"F": f_stat, "p": p_value, "df1": df_diff, "df2": int(m1_tool.df_resid),
            "robust": robust}

# ------------------------------------------------------------------ MAIN ---
def analyse(df_raw, speed_col, tag, out: Path):
    log(f"\n=== ANALYSIS PASS: {tag} (speed = {speed_col}) ===")
    df, ref_tool, ref_camp = prepare(df_raw, speed_col)
    log(f"  n={len(df)}; reference tool='{ref_tool}', reference campaign='{ref_camp}'")
    log(f"  q*: mean={df.q_star.mean():.2f} kPa, SD={df.q_star.std(ddof=1):.2f} kPa")
    specs = build_models(ref_tool, ref_camp)

    q1_result = q1_diagnostic(df, CFG["alpha"])

    rows_info, rows_coef = [], []
    for name, spec in specs.items():
        t, i = model_table(df, name, spec, CFG["alpha"])
        i["pass"] = tag
        rows_info.append(i)
        rows_coef.append(t)
    pd.DataFrame(rows_info).to_csv(out / f"model_info_{tag}.csv", index=False)
    pd.concat(rows_coef).to_csv(out / f"coefficients_{tag}.csv", index=False)

    # LOOCV
    loocv = {n: run_loocv(df, n, s) for n, s in specs.items()}
    lo_rows = [{"pass": tag, "model": n, "validation": "LOOCV", "group": "all", **metrics(e)}
               for n, e in loocv.items()]

    # Group-out CV
    gout = {}
    for gvar, label in (("camp", "LOCO"), ("tool", "LOTO")):
        for n, s in specs.items():
            e = run_group_out(df, n, s, gvar)
            if e is None:
                continue
            gout[(label, n)] = e
            for g in df[gvar].cat.categories:
                sub = e.loc[df[gvar] == g].dropna()
                if len(sub):
                    lo_rows.append({"pass": tag, "model": n, "validation": label,
                                    "group": g, **metrics(sub)})
    pd.DataFrame(lo_rows).to_csv(out / f"cv_errors_{tag}.csv", index=False)

    # SCRS-CV (leakage-free, size-matched)
    log("\n=== SCRS-CV (size-matched, leakage-free) ===")
    scrs_rows = []
    for gvar, label in (("camp", "LOCO"), ("tool", "LOTO")):
        for n, s in specs.items():
            if gvar in s["groups"]:
                continue
            scrs = run_scrs_cv(df, n, s, gvar,
                               n_reps=CFG["n_scrs_reps"], seed=CFG["seed"])
            for row in scrs:
                scrs_rows.append({"pass": tag, "model": n,
                                  "validation": f"SCRS-{label}", **row})
    pd.DataFrame(scrs_rows).to_csv(out / f"scrs_cv_{tag}.csv", index=False)

    # Bootstrap comparisons
    boot = []
    for a, b, q in (("M0", "M1", "Q2 speed effect, pooled"),
                    ("M2g", "M2", "Q2 speed effect, after group effects"),
                    ("M1", "M2", "Q3 tool+campaign effects")):
        r = paired_boot_delta(loocv[a], loocv[b], CFG["n_boot"], CFG["seed"], CFG["alpha"])
        if r:
            boot.append({"pass": tag, "comparison": f"{a} minus {b}", "question": q,
                         "type": "LOOCV paired", **r})

    # Transfer cost
    for label, gvar in (("LOCO", "camp"), ("LOTO", "tool")):
        for n in specs:
            e = gout.get((label, n))
            if e is None:
                continue
            for g in df[gvar].cat.categories:
                mask = (df[gvar] == g)
                r = paired_boot_delta(e.loc[mask], loocv[n].loc[mask],
                                      CFG["n_boot"], CFG["seed"], CFG["alpha"])
                if r:
                    boot.append({"pass": tag, "comparison": f"{n} {label}[{g}] minus LOOCV[{g}]",
                                 "question": "Q4 transfer cost", "type": f"{label} vs LOOCV", **r})
    pd.DataFrame(boot).to_csv(out / f"bootstrap_{tag}.csv", index=False)

    log(f"  wrote model_info/coefficients/cv_errors/scrs_cv/bootstrap for '{tag}'")
    return df, loocv, gout, q1_result

def main():
    out = Path(CFG["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    df_raw = pd.read_csv(CFG["data_file"])
    log(f"Loaded {CFG['data_file']}: {df_raw.shape[0]} rows x {df_raw.shape[1]} cols")
    log(f"Columns: {list(df_raw.columns)}")

    log("\n=== DATA CONSISTENCY: q* reconstruction ===")
    qstar_check = verify_qstar(df_raw)

    log("\n=== Q1: AREA PROPORTIONALITY ===")
    log("  Formal exponent test skipped: p confounded with tool identity (3 wd values).")
    log("  Using partial-F residual diagnostic instead (see report for safe wording).")

    q1_results = {}
    _, _, _, q1_results["primary"] = analyse(df_raw, CFG["col_speed_primary"], "primary", out)

    if CFG["col_speed_sensitivity"] in df_raw.columns:
        _, _, _, q1_results["sensitivity"] = analyse(df_raw, CFG["col_speed_sensitivity"], "sensitivity", out)
    else:
        log(f"\nSensitivity pass skipped: '{CFG['col_speed_sensitivity']}' not in CSV")

    manifest = {"protocol_version": "1.2", "config": CFG,
                "scipy": HAVE_SCIPY, "q1_method": "Partial F-test (classic + HC3 robust)",
                "qstar_consistency": qstar_check,
                "q1_results": q1_results, "log": LOG}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=str))
    log(f"\nDone. Outputs in '{out}/'")

if __name__ == "__main__":
    main()