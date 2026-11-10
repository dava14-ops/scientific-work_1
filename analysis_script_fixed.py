from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import statsmodels.formula.api as smf
    from statsmodels.stats.diagnostic import het_breuschpagan
except ImportError as err:  # pragma: no cover - environment dependency check
    sys.exit(f"Missing dependency: {err}. Install statsmodels and its dependencies.")

try:
    from scipy import stats as sps
    HAVE_SCIPY = True
except ImportError:  # pragma: no cover - statsmodels normally requires scipy
    sps = None
    HAVE_SCIPY = False


# ------------------------------------------------------------------ CONFIG --
# Place field_runs_28_summary.csv in the same directory as this script, or
# change data_file to the required path.
CFG: dict[str, Any] = {
    "data_file": "field_runs_28_summary.csv",
    "col_qstar": "specific_resistance_kPa",
    "col_speed_primary": "nominal_speed_ms",
    "col_speed_sensitivity": "mean_speed_ms",
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
    LOG.append(msg)
    print(msg)


def as_float(value: Any) -> float:
    """Convert scalar-like statsmodels output to a plain Python float."""
    return float(np.asarray(value).squeeze())


# ------------------------------------------------------ DATA CONSISTENCY ---
def verify_qstar(df_raw: pd.DataFrame) -> dict[str, Any]:
    """Reconstruct q* from F, w and d and compare it with the CSV column.

    Assumed units: F in N, w and d in mm, q* in kPa:
        q*_kPa = 1000 * F_N / (w_mm * d_mm).

    The check is skipped if the required columns are unavailable.
    """
    need = [CFG["col_qstar"], CFG["col_force"], CFG["col_width"], CFG["col_depth"]]
    absent = [col for col in need if col not in df_raw.columns]
    if absent:
        log(f"  q* consistency check skipped; missing columns: {absent}")
        return {"status": "skipped", "absent": absent}

    d = df_raw[need].copy()
    for col in need:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    finite = np.isfinite(d[need].to_numpy(dtype=float)).all(axis=1)
    positive = (d[need] > 0).all(axis=1)
    valid = finite & positive
    n_excluded = int((~valid).sum())
    d = d.loc[valid]

    if len(d) < 3:
        log("  q* consistency check skipped: fewer than 3 valid rows")
        return {
            "status": "insufficient_rows",
            "n_valid": int(len(d)),
            "n_excluded": n_excluded,
        }

    recomputed = (
        1000.0
        * d[CFG["col_force"]]
        / (d[CFG["col_width"]] * d[CFG["col_depth"]])
    )
    published = d[CFG["col_qstar"]]
    rel_err = np.abs(recomputed - published) / np.abs(published)

    if np.std(recomputed) > 0 and np.std(published) > 0:
        corr: float | None = float(np.corrcoef(recomputed, published)[0, 1])
    else:
        corr = None

    out = {
        "status": "ok",
        "n": int(len(d)),
        "n_excluded": n_excluded,
        "corr": corr,
        "max_rel_err": float(rel_err.max()),
        "mean_rel_err": float(rel_err.mean()),
        "convention": "q*_kPa = 1000 * F_N / (w_mm * d_mm)",
    }
    corr_text = "undefined" if corr is None else f"{corr:.6f}"
    log(
        f"  q* consistency: n={len(d)}, corr={corr_text}, "
        f"max_rel_err={out['max_rel_err']:.4e}"
    )
    if out["max_rel_err"] > 1e-3:
        log("  WARNING: reconstructed q* differs from the CSV column by more than 0.1%.")
        log("           Verify the dataset's units and definition of specific resistance.")
    return out


# ------------------------------------------------------------- PREPROCESS --
def prepare(df_raw: pd.DataFrame, speed_col: str):
    need = [CFG["col_qstar"], speed_col, CFG["col_tool"], CFG["col_campaign"]]
    missing = [col for col in need if col not in df_raw.columns]
    if missing:
        raise KeyError(f"Columns absent from CSV: {missing}")

    d = df_raw.copy().reset_index(drop=True)

    # Numeric columns: coerce strings and remove missing/non-finite observations.
    d["q_star"] = pd.to_numeric(d[CFG["col_qstar"]], errors="coerce")
    d["v"] = pd.to_numeric(d[speed_col], errors="coerce")
    finite_mask = np.isfinite(d["q_star"]) & np.isfinite(d["v"])
    if (~finite_mask).any():
        log(
            f"  ! {int((~finite_mask).sum())} row(s) dropped: "
            "q* or speed is missing/non-finite"
        )
        d = d.loc[finite_mask].reset_index(drop=True)

    if d.empty:
        raise ValueError(f"No valid observations remain for speed column '{speed_col}'.")

    d["v2"] = d["v"] ** 2
    if not np.isfinite(d["v2"]).all():
        raise ValueError("Squaring the speed produced non-finite values; check the speed column.")

    # Check categorical values before converting them to strings so NaN does
    # not silently become a literal category named 'nan'.
    for source_col, target_col in (
        (CFG["col_tool"], "tool"),
        (CFG["col_campaign"], "camp"),
    ):
        missing_mask = d[source_col].isna()
        if missing_mask.any():
            log(f"  ! {int(missing_mask.sum())} row(s) dropped: missing {target_col}")
            d = d.loc[~missing_mask].reset_index(drop=True)

        d[target_col] = d[source_col].astype(str).str.strip()
        empty_mask = d[target_col].eq("")
        if empty_mask.any():
            log(f"  ! {int(empty_mask.sum())} row(s) dropped: empty {target_col}")
            d = d.loc[~empty_mask].reset_index(drop=True)

    if d.empty:
        raise ValueError("No valid observations remain after removing missing categories.")

    tools = sorted(d["tool"].unique(), key=lambda value: (len(value), value))
    observed_campaigns = set(d["camp"].unique())
    ordered_campaigns = [
        campaign for campaign in CFG["campaign_order"] if campaign in observed_campaigns
    ]
    campaigns = ordered_campaigns + sorted(observed_campaigns - set(ordered_campaigns))

    if len(tools) < 2:
        raise ValueError("At least two distinct tool categories are required for tool comparisons.")
    if len(campaigns) < 2:
        raise ValueError("At least two campaign categories are required for campaign comparisons.")

    d["tool"] = pd.Categorical(d["tool"], categories=tools)
    d["camp"] = pd.Categorical(d["camp"], categories=campaigns)
    return d, tools[0], campaigns[0]


# --------------------------------------------------------------- MODELS ----
def build_models(ref_tool: str, ref_camp: str) -> dict[str, dict[str, Any]]:
    # repr() safely quotes category labels that contain apostrophes or spaces.
    tool_term = f"C(tool, Treatment(reference={ref_tool!r}))"
    camp_term = f"C(camp, Treatment(reference={ref_camp!r}))"
    return {
        "M0": {"formula": "q_star ~ 1", "groups": ()},
        "M1": {"formula": "q_star ~ v2", "groups": ()},
        "M1g": {"formula": f"q_star ~ v2 + {tool_term}", "groups": ("tool",)},
        "M1c": {"formula": f"q_star ~ v2 + {camp_term}", "groups": ("camp",)},
        "M2": {
            "formula": f"q_star ~ v2 + {tool_term} + {camp_term}",
            "groups": ("tool", "camp"),
        },
        "M2g": {
            "formula": f"q_star ~ {tool_term} + {camp_term}",
            "groups": ("tool", "camp"),
        },
    }


def fit_predict(formula: str, train: pd.DataFrame, test: pd.DataFrame, cat_cols, ctx: str = ""):
    """Fit on train and predict test, checking estimability of factor levels.

    A model containing a categorical fixed effect is fitted only when every
    category declared for that factor in the full prepared dataset is present
    in the training fold. This keeps treatment contrasts stable and avoids
    predictions from a fold with an unobserved reference category.
    """
    for col in cat_cols:
        if isinstance(train[col].dtype, pd.CategoricalDtype):
            expected = set(map(str, train[col].cat.categories))
        else:
            expected = set(map(str, train[col].dropna().unique()))

        observed = set(map(str, train[col].dropna().unique()))
        missing_levels = expected - observed
        if missing_levels:
            raise ValueError(
                f"{ctx}: training fold for '{formula}' lacks level(s) of '{col}': "
                f"{sorted(missing_levels)}. The fixed-effect model is not fully estimable."
            )

        test_levels = set(map(str, test[col].dropna().unique()))
        unseen_levels = test_levels - observed
        if unseen_levels:
            raise ValueError(
                f"{ctx}: test data contain unseen level(s) of '{col}': "
                f"{sorted(unseen_levels)}. Use a model without this fixed effect."
            )

    model = smf.ols(formula, data=train).fit()
    prediction = model.predict(test).to_numpy(dtype=float)
    if not np.isfinite(prediction).all():
        raise ValueError(f"{ctx}: model produced non-finite predictions.")
    return model, prediction


def metrics(errors) -> dict[str, Any]:
    err = np.asarray(errors, dtype=float)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return {"n": 0, "RMSE_kPa": None, "MAE_kPa": None, "Bias_kPa": None}
    return {
        "n": int(err.size),
        "RMSE_kPa": float(np.sqrt(np.mean(err**2))),
        "MAE_kPa": float(np.mean(np.abs(err))),
        "Bias_kPa": float(np.mean(err)),
    }


# ------------------------------------------------------------ VALIDATION ---
def run_loocv(df: pd.DataFrame, name: str, spec: dict[str, Any]) -> pd.Series:
    errors = np.empty(len(df), dtype=float)
    for i in range(len(df)):
        train = df.drop(index=i)
        test = df.iloc[[i]]
        _, prediction = fit_predict(
            spec["formula"], train, test, spec["groups"], ctx=f"LOOCV/{name}/i={i}"
        )
        errors[i] = float(df["q_star"].iloc[i] - prediction[0])
    return pd.Series(errors, index=df.index, name=name)


def run_group_out(df: pd.DataFrame, name: str, spec: dict[str, Any], group_var: str):
    if group_var in spec["groups"]:
        log(f"  - {name}: ineligible for leave-one-{group_var}-out; skipped")
        return None

    errors = pd.Series(np.nan, index=df.index, name=name, dtype=float)
    for group, indices in df.groupby(group_var, observed=True).groups.items():
        indices = list(indices)
        train = df.drop(index=indices)
        test = df.loc[indices]
        _, prediction = fit_predict(
            spec["formula"], train, test, spec["groups"],
            ctx=f"LO{group_var}/{name}/{group}",
        )
        errors.loc[indices] = test["q_star"].to_numpy(dtype=float) - prediction
    return errors


# -------------------------------------------------------- SCRS-CV (FIXED) --
def run_scrs_cv(
    df: pd.DataFrame,
    name: str,
    spec: dict[str, Any],
    group_var: str,
    n_reps: int = 200,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Size-matched random-control CV without test/train overlap.

    For a held-out group of size n_g, each target observation is predicted by
    a separately fitted model trained on N - n_g rows sampled from all rows
    except that target observation. Thus the training size matches group-out
    CV, but other observations belonging to the nominally held-out group may
    appear in the random-control training sample. This is a random-control
    comparison, not a test of transfer to a completely unseen group.
    """
    if n_reps < 1:
        raise ValueError("n_reps must be at least 1")

    rng = np.random.default_rng(seed)
    groups = df.groupby(group_var, observed=True).groups
    n_total = len(df)
    summary: list[dict[str, Any]] = []

    for group, indices in groups.items():
        target_indices = list(indices)
        group_size = len(target_indices)
        n_train = n_total - group_size
        rep_rmses: list[float] = []
        max_attempts = n_reps * 5

        for attempt in range(max_attempts):
            if len(rep_rmses) >= n_reps:
                break

            errors: list[float] = []
            replicate_ok = True

            for target_index in target_indices:
                # The current test row is never included in its training pool.
                pool = df.index.drop(target_index).to_numpy()
                if n_train > len(pool):
                    replicate_ok = False
                    break

                sampled_indices = rng.choice(pool, size=n_train, replace=False)
                train = df.loc[sampled_indices]
                test = df.loc[[target_index]]

                try:
                    _, prediction = fit_predict(
                        spec["formula"], train, test, spec["groups"],
                        ctx=f"SCRS/{name}/{group}/rep{attempt}/t{target_index}",
                    )
                    errors.append(float(test["q_star"].iloc[0] - prediction[0]))
                except (ValueError, np.linalg.LinAlgError):
                    replicate_ok = False
                    break

            if replicate_ok and len(errors) == group_size:
                rep_rmses.append(float(np.sqrt(np.mean(np.asarray(errors) ** 2))))

        if len(rep_rmses) < n_reps:
            log(
                f"  WARNING: SCRS for {name}/{group} achieved "
                f"{len(rep_rmses)}/{n_reps} requested repetitions"
            )

        if rep_rmses:
            summary.append(
                {
                    "group": group,
                    "n_reps": len(rep_rmses),
                    "n_train_per_target_fit": n_train,
                    "RMSE_mean": float(np.mean(rep_rmses)),
                    "RMSE_std": (
                        float(np.std(rep_rmses, ddof=1)) if len(rep_rmses) > 1 else None
                    ),
                    "RMSE_median": float(np.median(rep_rmses)),
                }
            )
    return summary


# ------------------------------------------------------------- BOOTSTRAP ---
def paired_boot_delta(err_a, err_b, n_boot: int, seed: int, alpha: float):
    """Paired bootstrap of the RMSE difference for the same test observations.

    This resamples already-computed cross-validation errors; it does not refit
    models in each bootstrap replicate. Its interval is therefore conditional
    on the calculated CV errors and is not a full model-training uncertainty
    interval. Rows are resampled independently (not by campaign/tool cluster).
    """
    a = np.asarray(err_a, dtype=float)
    b = np.asarray(err_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("Paired bootstrap requires error arrays of equal length.")

    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    n = a.size
    if n < 4:
        return None
    if n_boot < 1:
        raise ValueError("n_boot must be at least 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")

    rmse_a = float(np.sqrt(np.mean(a**2)))
    rmse_b = float(np.sqrt(np.mean(b**2)))
    observed_delta = rmse_a - rmse_b
    relative_change = 100.0 * observed_delta / rmse_b if rmse_b > 0 else float("nan")

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(n_boot, n))
    delta = (
        np.sqrt(np.mean(a[indices] ** 2, axis=1))
        - np.sqrt(np.mean(b[indices] ** 2, axis=1))
    )
    lower, upper = np.percentile(
        delta, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)]
    )

    favours = "B" if lower > 0 else "A" if upper < 0 else "neither"
    return {
        "n_paired": int(n),
        "RMSE_A": rmse_a,
        "RMSE_B": rmse_b,
        "delta_RMSE_A_minus_B": observed_delta,
        "rel_change_pct": relative_change,
        "ci_low": float(lower),
        "ci_high": float(upper),
        "excludes_zero": bool(lower > 0 or upper < 0),
        "favours": favours,
    }


# ------------------------------------------------------- FULL-DATA FITS ----
def model_table(df: pd.DataFrame, name: str, spec: dict[str, Any], alpha: float):
    model = smf.ols(spec["formula"], data=df).fit()
    n = int(model.nobs)

    # Standard Gaussian AICc parameter count:
    # K = regression coefficients (including intercept) + residual variance.
    k_regression = int(len(model.params))
    k_total = k_regression + 1
    aic_standard = float(-2.0 * model.llf + 2.0 * k_total)
    denominator = n - k_total - 1
    aicc = (
        float(aic_standard + 2.0 * k_total * (k_total + 1) / denominator)
        if denominator > 0
        else float("nan")
    )

    ci = model.conf_int(alpha=alpha)
    coefficient_table = pd.DataFrame(
        {
            "term": model.params.index,
            "coef": model.params.values,
            "std_err": model.bse.values,
            "ci_low": ci[0].values,
            "ci_high": ci[1].values,
            "p_value": model.pvalues.values,
        }
    )
    coefficient_table.insert(0, "model", name)

    residuals = model.resid.to_numpy(dtype=float)
    fitted = model.fittedvalues.to_numpy(dtype=float)

    if HAVE_SCIPY and n >= 8:
        shapiro_p = float(sps.shapiro(residuals).pvalue)
    else:
        shapiro_p = None

    try:
        bp_p = float(het_breuschpagan(residuals, model.model.exog)[1])
        if not np.isfinite(bp_p):
            bp_p = None
    except Exception:
        bp_p = None

    if np.std(fitted) > 0 and np.std(residuals) > 0:
        residual_fitted_corr = float(np.corrcoef(residuals, fitted)[0, 1])
    else:
        residual_fitted_corr = None

    info = {
        "model": name,
        "n": n,
        "K_regression_coefficients": k_regression,
        "K_including_variance": k_total,
        "R2": float(model.rsquared),
        "R2_adj": float(model.rsquared_adj),
        "AIC": aic_standard,
        "AIC_statsmodels_convention": float(model.aic),
        "AICc": aicc,
        "resid_SD_kPa": float(np.sqrt(model.mse_resid)),
        "shapiro_p": shapiro_p,
        "bp_p": bp_p,
        "corr_resid_fitted": residual_fitted_corr,
    }
    return coefficient_table, info


# -------------------------------- Q1: PARTIAL F-TEST (M1c versus M2) ------
def q1_diagnostic(df: pd.DataFrame, alpha: float) -> dict[str, Any] | None:
    """Test whether tool effects remain after adjustment for speed and campaign.

    Reduced model M1c: q* ~ v^2 + campaign.
    Full model M2:      q* ~ v^2 + tool + campaign.

    The classical nested-model partial F-test and the HC3 robust joint Wald
    F-test are reported separately. Neither is substituted for the other.
    """
    log("\n=== Q1 DIAGNOSTIC: Partial F-test (M1c versus M2) ===")

    tool_levels = list(df["tool"].cat.categories)
    campaign_levels = list(df["camp"].cat.categories)
    if len(tool_levels) < 2 or len(campaign_levels) < 2:
        log("  Q1 diagnostic skipped: at least two tool and campaign levels are required.")
        return None

    ref_tool = str(tool_levels[0])
    ref_camp = str(campaign_levels[0])
    tool_term = f"C(tool, Treatment(reference={ref_tool!r}))"
    camp_term = f"C(camp, Treatment(reference={ref_camp!r}))"

    reduced_formula = f"q_star ~ v2 + {camp_term}"
    full_formula = f"q_star ~ v2 + {tool_term} + {camp_term}"

    reduced = smf.ols(reduced_formula, data=df).fit()
    full = smf.ols(full_formula, data=df).fit()

    # statsmodels expects the unrestricted/full model as the caller and the
    # restricted/reduced model as the argument.
    try:
        f_stat, p_value, df_diff = full.compare_f_test(reduced)
    except Exception as exc:
        log(f"  Classical partial F-test unavailable: {exc}")
        f_stat = p_value = df_diff = float("nan")

    if not np.isfinite(f_stat) or not np.isfinite(df_diff) or df_diff <= 0:
        log("  ERROR: models are not nested as expected; partial F-test not reported.")
        return None

    df1 = int(round(float(df_diff)))
    df2 = int(round(float(full.df_resid)))
    log(f"  comparison: M1c versus M2")
    log(f"  classical partial F({df1},{df2})={float(f_stat):.3f}, p={float(p_value):.6g}")

    robust: dict[str, Any] | None = None
    try:
        # use_t=True plus use_f=True gives the finite-denominator-df Wald F test.
        full_hc3 = smf.ols(full_formula, data=df).fit(cov_type="HC3", use_t=True)
        tool_terms = [term for term in full_hc3.params.index if "tool" in term]
        if not tool_terms:
            raise RuntimeError("No tool coefficients were found in the full model.")

        restriction = np.zeros((len(tool_terms), len(full_hc3.params)), dtype=float)
        parameter_names = list(full_hc3.params.index)
        for row_index, term in enumerate(tool_terms):
            restriction[row_index, parameter_names.index(term)] = 1.0

        wald = full_hc3.wald_test(restriction, use_f=True, scalar=True)
        robust = {
            "statistic": as_float(wald.statistic),
            "p_value": as_float(wald.pvalue),
            "df1": as_float(wald.df_num),
            "df2": as_float(wald.df_denom),
            "cov_type": "HC3",
            "distribution": "F",
        }
        log(
            f"  HC3 robust Wald F({int(round(robust['df1']))},"
            f"{int(round(robust['df2']))})={robust['statistic']:.3f}, "
            f"p={robust['p_value']:.6g}"
        )
    except Exception as exc:
        log(f"  HC3 robust Wald F-test unavailable: {exc}")

    return {
        "comparison": "M1c versus M2",
        "reduced_formula": reduced_formula,
        "full_formula": full_formula,
        "F": float(f_stat),
        "p": float(p_value),
        "df1": df1,
        "df2": df2,
        "robust": robust,
    }


# ---------------------------------------------------------------- ANALYSIS --
def analyse(df_raw: pd.DataFrame, speed_col: str, tag: str, out: Path):
    log(f"\n=== ANALYSIS PASS: {tag} (speed = {speed_col}) ===")
    df, ref_tool, ref_camp = prepare(df_raw, speed_col)
    log(
        f"  n={len(df)}; reference tool='{ref_tool}', "
        f"reference campaign='{ref_camp}'"
    )
    log(f"  q*: mean={df.q_star.mean():.2f} kPa, SD={df.q_star.std(ddof=1):.2f} kPa")

    specs = build_models(ref_tool, ref_camp)
    q1_result = q1_diagnostic(df, CFG["alpha"])

    rows_info = []
    rows_coef = []
    for name, spec in specs.items():
        coefficient_table, info = model_table(df, name, spec, CFG["alpha"])
        info["pass"] = tag
        rows_info.append(info)
        rows_coef.append(coefficient_table)

    pd.DataFrame(rows_info).to_csv(out / f"model_info_{tag}.csv", index=False)
    pd.concat(rows_coef, ignore_index=True).to_csv(
        out / f"coefficients_{tag}.csv", index=False
    )

    # LOOCV: each observation is predicted from a model fitted without it.
    loocv = {name: run_loocv(df, name, spec) for name, spec in specs.items()}
    cv_rows = [
        {"pass": tag, "model": name, "validation": "LOOCV", "group": "all", **metrics(err)}
        for name, err in loocv.items()
    ]

    # Group-out CV: the complete campaign/tool category is withheld.
    group_out_errors = {}
    for group_var, label in (("camp", "LOCO"), ("tool", "LOTO")):
        for name, spec in specs.items():
            errors = run_group_out(df, name, spec, group_var)
            if errors is None:
                continue
            group_out_errors[(label, name)] = errors
            for group in df[group_var].cat.categories:
                subset = errors.loc[df[group_var] == group].dropna()
                if len(subset):
                    cv_rows.append(
                        {
                            "pass": tag,
                            "model": name,
                            "validation": label,
                            "group": group,
                            **metrics(subset),
                        }
                    )

    pd.DataFrame(cv_rows).to_csv(out / f"cv_errors_{tag}.csv", index=False)

    # SCRS-CV: random-control training sets matched in size to group-out sets.
    log("\n=== SCRS-CV (size-matched random control; no test/train overlap) ===")
    scrs_rows = []
    for group_var, label in (("camp", "LOCO"), ("tool", "LOTO")):
        for name, spec in specs.items():
            if group_var in spec["groups"]:
                continue
            scrs = run_scrs_cv(
                df,
                name,
                spec,
                group_var,
                n_reps=CFG["n_scrs_reps"],
                seed=CFG["seed"],
            )
            for row in scrs:
                scrs_rows.append(
                    {"pass": tag, "model": name, "validation": f"SCRS-{label}", **row}
                )
    pd.DataFrame(scrs_rows).to_csv(out / f"scrs_cv_{tag}.csv", index=False)

    # Paired bootstrap comparisons of the same LOOCV test observations.
    bootstrap_rows = []
    for model_a, model_b, question in (
        ("M0", "M1", "Q2 speed effect, pooled"),
        ("M2g", "M2", "Q2 speed effect, after group effects"),
        ("M1", "M2", "Q3 tool+campaign effects"),
    ):
        result = paired_boot_delta(
            loocv[model_a], loocv[model_b], CFG["n_boot"], CFG["seed"], CFG["alpha"]
        )
        if result is not None:
            bootstrap_rows.append(
                {
                    "pass": tag,
                    "comparison": f"{model_a} minus {model_b}",
                    "question": question,
                    "type": "LOOCV paired",
                    **result,
                }
            )

    # Transfer cost: compare group-out and LOOCV errors on identical test units.
    for label, group_var in (("LOCO", "camp"), ("LOTO", "tool")):
        for name in specs:
            errors = group_out_errors.get((label, name))
            if errors is None:
                continue
            for group in df[group_var].cat.categories:
                mask = df[group_var] == group
                result = paired_boot_delta(
                    errors.loc[mask],
                    loocv[name].loc[mask],
                    CFG["n_boot"],
                    CFG["seed"],
                    CFG["alpha"],
                )
                if result is not None:
                    bootstrap_rows.append(
                        {
                            "pass": tag,
                            "comparison": f"{name} {label}[{group}] minus LOOCV[{group}]",
                            "question": "Q4 transfer cost",
                            "type": f"{label} vs LOOCV",
                            **result,
                        }
                    )

    pd.DataFrame(bootstrap_rows).to_csv(out / f"bootstrap_{tag}.csv", index=False)
    log(f"  wrote model_info/coefficients/cv_errors/scrs_cv/bootstrap for '{tag}'")
    return df, loocv, group_out_errors, q1_result


# -------------------------------------------------------------------- MAIN --
def main() -> None:
    data_path = Path(CFG["data_file"])
    out = Path(CFG["out_dir"])
    out.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path.resolve()}\n"
            "Place the CSV beside this script or update CFG['data_file']."
        )

    if not 0 < CFG["alpha"] < 1:
        raise ValueError("CFG['alpha'] must be between 0 and 1.")
    if CFG["n_boot"] < 1 or CFG["n_scrs_reps"] < 1:
        raise ValueError("Bootstrap and SCRS repetition counts must be positive.")

    df_raw = pd.read_csv(data_path)
    log(f"Loaded {data_path}: {df_raw.shape[0]} rows x {df_raw.shape[1]} columns")
    log(f"Columns: {list(df_raw.columns)}")

    log("\n=== DATA CONSISTENCY: q* reconstruction ===")
    qstar_check = verify_qstar(df_raw)

    log("\n=== Q1: AREA PROPORTIONALITY ===")
    log("  Formal exponent test not performed: width is confounded with tool identity.")
    log("  Diagnostic used: partial F-test M1c versus M2, plus HC3 robust Wald F-test.")

    q1_results = {}
    _, _, _, q1_results["primary"] = analyse(
        df_raw, CFG["col_speed_primary"], "primary", out
    )

    sensitivity_col = CFG["col_speed_sensitivity"]
    if sensitivity_col in df_raw.columns:
        _, _, _, q1_results["sensitivity"] = analyse(
            df_raw, sensitivity_col, "sensitivity", out
        )
    else:
        log(f"\nSensitivity pass skipped: '{sensitivity_col}' is absent from the CSV.")

    log(f"\nDone. Outputs saved in: {out.resolve()}")
    manifest = {
        "protocol_version": "1.3-fixed",
        "config": CFG,
        "scipy_available": HAVE_SCIPY,
        "q1_method": "Partial F-test M1c versus M2 (classical + HC3 robust Wald F)",
        "aicc_definition": (
            "AICc with K equal to regression coefficients including intercept plus "
            "one residual-variance parameter"
        ),
        "qstar_consistency": qstar_check,
        "q1_results": q1_results,
        "log": LOG,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFATAL: {exc}", file=sys.stderr)
        raise
