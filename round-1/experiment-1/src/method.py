#!/usr/bin/env python3
"""
Welfare State as Recovery-Rate Governor: Full Four-Step Empirical Pipeline.

Steps:
  0. Monte Carlo validation of AR(1) estimators
  1. Per-country bias-corrected recovery-rate panel
  2. CSD anticipation test before gradual ERT onsets
  3. Primary panel regression of recovery rate on lagged realized redistribution
  4. Hazard model with inequality×redistribution interaction
"""

import json
import math
import resource
import sys
import time
from pathlib import Path

import psutil
from loguru import logger

# ── Logging ──────────────────────────────────────────────────────────────────
WS = Path(__file__).parent
LOGS = WS / "logs"
LOGS.mkdir(exist_ok=True)

GREEN, CYAN, END = "\033[92m", "\033[96m", "\033[0m"
logger.remove()
logger.add(
    sys.stdout,
    level="INFO",
    format=f"{GREEN}{{time:HH:mm:ss}}{END}|{{level:<7}}|{CYAN}{{function}}{END}| {{message}}",
)
logger.add(LOGS / "run.log", rotation="30 MB", level="DEBUG")

# ── Resource limits ───────────────────────────────────────────────────────────
_avail = psutil.virtual_memory().available
RAM_BUDGET = min(int(_avail * 0.75), 30 * 1024 ** 3)  # 75% of available, cap 30 GB
resource.setrlimit(resource.RLIMIT_AS, (RAM_BUDGET * 3, RAM_BUDGET * 3))

# ── Hardware ──────────────────────────────────────────────────────────────────
def _detect_cpus() -> int:
    import os
    try:
        parts = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if parts[0] != "max":
            return math.ceil(int(parts[0]) / int(parts[1]))
    except (FileNotFoundError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        pass
    return os.cpu_count() or 1

NUM_CPUS = _detect_cpus()
NUM_WORKERS = max(1, min(NUM_CPUS - 1, 5))
logger.info(f"Hardware: {NUM_CPUS} CPUs, workers={NUM_WORKERS}, RAM_BUDGET={RAM_BUDGET/1e9:.1f}GB")


@logger.catch(reraise=True)
def main() -> None:
    t_start = time.time()

    # ── Load data ────────────────────────────────────────────────────────────
    logger.info("=== Loading datasets ===")
    from data_loader import load_all, derive_onsets_from_vdem
    from utils import build_vdem_panel, build_gdp_panel, build_schooling_panel

    raw = load_all()
    vdem_panel = build_vdem_panel(raw["vdem_raw"])
    gdp_panel = build_gdp_panel(raw["gdp_raw"])
    schooling_panel = build_schooling_panel(raw["schooling_raw"])
    redistrib_df = raw["redistrib_raw"]
    ert_raw = raw["ert_raw"]

    # ERT or algorithmic fallback
    if ert_raw is not None:
        onset_df = ert_raw
        ert_source = "ERT"
    else:
        onset_df = derive_onsets_from_vdem(vdem_panel)
        ert_source = "algorithmic_from_vdem"

    logger.info(f"V-Dem: {vdem_panel['country'].nunique()} countries, "
                f"{vdem_panel['year'].min():.0f}-{vdem_panel['year'].max():.0f}")
    logger.info(f"Redistribution: {len(redistrib_df)} rows, {redistrib_df['country'].nunique() if 'country' in redistrib_df else 0} countries")
    logger.info(f"Onsets: {len(onset_df)} ({ert_source})")

    # ── Step 0: Monte Carlo ──────────────────────────────────────────────────
    logger.info("=== STEP 0: Monte Carlo ===")
    from step0_montecarlo import run_step0
    step0 = run_step0(n_sim=500, num_workers=min(NUM_WORKERS, 2))
    estimator_used = step0.get("recommended_estimator", "Andrews_MU")
    logger.info(f"Step 0 done. Recommended estimator: {estimator_used}")

    # ── Step 1: Recovery rates ───────────────────────────────────────────────
    logger.info("=== STEP 1: Recovery rates ===")
    from step1_recovery_rates import run_step1
    step1 = run_step1(vdem_panel, estimator_name=estimator_used, num_workers=NUM_WORKERS)
    recovery_panel = step1.pop("panel")
    logger.info(f"Step 1 done. {step1['n_country_years']} country-years")

    # ── Step 2: CSD anticipation ──────────────────────────────────────────────
    logger.info("=== STEP 2: CSD anticipation ===")
    from step2_csd_anticipation import run_step2
    step2 = run_step2(recovery_panel, onset_df)
    logger.info(f"Step 2 done. pred1={step2.get('prediction1_confirmed')}")

    # ── Step 3: Rate regression ───────────────────────────────────────────────
    logger.info("=== STEP 3: Rate regression ===")
    from step3_rate_regression import run_step3
    step3 = run_step3(recovery_panel, redistrib_df, gdp_panel, schooling_panel)
    logger.info(f"Step 3 done. pred2={step3.get('prediction2_confirmed')}")

    # ── Step 4: Hazard model ──────────────────────────────────────────────────
    logger.info("=== STEP 4: Hazard model ===")
    from step4_hazard import run_step4
    step4 = run_step4(vdem_panel, onset_df, redistrib_df, gdp_panel)
    logger.info(f"Step 4 done. pred3={step4.get('prediction3_confirmed')}")

    # ── Holm multiple-testing correction ─────────────────────────────────────
    from statsmodels.stats.multitest import multipletests

    p_pred1 = float(step2.get("p_wilcoxon", 1.0) or 1.0)
    p_pred2 = float(step3.get("p_redistribution_lag5", 1.0) or 1.0)
    p_pred3 = float(step4.get("p_interaction", 1.0) or 1.0)
    p_vals = [p_pred2, p_pred1, p_pred3]
    try:
        rejected, p_adj, _, _ = multipletests(p_vals, method="holm")
        holm = {
            "p_values_raw": [round(p, 4) for p in p_vals],
            "p_values_adjusted": [round(float(p), 4) for p in p_adj],
            "rejected": [bool(r) for r in rejected],
            "labels": ["Pred2_redistrib_coeff", "Pred1_csd_auc", "Pred3_interaction"],
        }
    except Exception as e:
        logger.error(f"Holm correction failed: {e}")
        holm = {"p_values_raw": p_vals, "p_values_adjusted": p_vals, "rejected": [False] * 3}

    t_total = round(time.time() - t_start, 1)
    logger.info(f"Total runtime: {t_total}s")

    # ── Build per-country examples ────────────────────────────────────────────
    # Each example = one country's recovery-rate panel entry with predictions
    import numpy as np

    # Build country-level summaries from recovery_panel
    panel_copy = recovery_panel.copy()
    panel_copy = panel_copy.rename(columns={"center_year": "year"}) if "center_year" in panel_copy.columns else panel_copy

    # Grand mean lambda_bc as baseline prediction
    grand_mean_lambda = float(panel_copy["lambda_bc"].mean()) if "lambda_bc" in panel_copy.columns else 0.0

    # Build regression-based prediction using step3 coefficient (if available)
    coeff_redist = step3.get("coeff_redistribution_lag5") or 0.0

    # Merge redistribution for per-country prediction
    redist_sub = raw["redistrib_raw"].copy()
    if not redist_sub.empty and "redistribution" in redist_sub.columns:
        country_redist = (
            redist_sub.groupby("country")["redistribution"]
            .mean()
            .reset_index()
            .rename(columns={"redistribution": "mean_redistribution"})
        )
    else:
        country_redist = None

    # Build per-country summary
    grp_cols = ["country", "lambda_bc"]
    if "ac1_rolling" in panel_copy.columns:
        grp_cols.append("ac1_rolling")
    country_summary = (
        panel_copy[grp_cols]
        .groupby("country")
        .agg(
            mean_lambda=("lambda_bc", "mean"),
            std_lambda=("lambda_bc", "std"),
            n_windows=("lambda_bc", "count"),
        )
        .reset_index()
    )
    if country_redist is not None:
        country_summary = country_summary.merge(country_redist, on="country", how="left")
    else:
        country_summary["mean_redistribution"] = float("nan")

    # Sort by name for reproducibility
    country_summary = country_summary.sort_values("country").reset_index(drop=True)

    examples = []
    for _, row in country_summary.iterrows():
        country = str(row["country"])
        mean_lam = float(row["mean_lambda"]) if not (isinstance(row["mean_lambda"], float) and math.isnan(row["mean_lambda"])) else 0.0
        n_win = int(row["n_windows"])
        mean_redist = row.get("mean_redistribution", float("nan"))
        has_redist = isinstance(mean_redist, float) and not math.isnan(mean_redist)

        # Welfare-governor prediction: grand_mean + coeff * (redistrib - grand_mean_redistrib)
        # If no redistrib data, fallback = grand mean
        if has_redist and coeff_redist != 0.0:
            grand_mean_r = float(country_redist["mean_redistribution"].mean()) if country_redist is not None else 0.0
            predict_welfare = round(grand_mean_lambda + coeff_redist * (float(mean_redist) - grand_mean_r), 5)
        else:
            predict_welfare = round(grand_mean_lambda, 5)

        # Baseline: grand mean (no covariates)
        predict_baseline = round(grand_mean_lambda, 5)

        examples.append({
            "input": json.dumps({
                "country": country,
                "n_recovery_windows": n_win,
                "mean_redistribution": round(float(mean_redist), 4) if has_redist else None,
                "grand_mean_lambda_bc": round(grand_mean_lambda, 5),
            }),
            "output": json.dumps({
                "mean_lambda_bc": round(mean_lam, 5),
                "std_lambda_bc": round(float(row["std_lambda"]), 5) if not (isinstance(row["std_lambda"], float) and math.isnan(row["std_lambda"])) else None,
            }),
            "metadata_country": country,
            "metadata_n_windows": n_win,
            "metadata_has_redistribution": has_redist,
            "predict_welfare_governor": str(predict_welfare),
            "predict_baseline_grand_mean": str(predict_baseline),
        })

    # Append pipeline-level summary examples (step summaries)
    examples.append({
        "input": "Pipeline summary: Monte Carlo estimator validation + CSD anticipation + panel regression + hazard model",
        "output": json.dumps({
            "recommended_estimator": step0.get("recommended_estimator"),
            "bias_ok": step0.get("bias_ok"),
            "prediction1_confirmed": step2.get("prediction1_confirmed"),
            "prediction2_confirmed": step3.get("prediction2_confirmed"),
            "prediction3_confirmed": step4.get("prediction3_confirmed"),
            "holm_rejected": holm.get("rejected"),
            "total_runtime_seconds": t_total,
        }),
        "metadata_step": "pipeline_summary",
        "predict_welfare_governor": str(step3.get("prediction2_confirmed", False)),
        "predict_baseline_grand_mean": "False",
    })

    method_out = {
        "metadata": {
            "method_name": "Welfare State as Recovery-Rate Governor",
            "description": "Four-step empirical pipeline: MC estimator validation, rolling AR(1) recovery rates, CSD anticipation test, panel regression, hazard model",
            "total_runtime_seconds": t_total,
            "data_sources": ["V-Dem (OWID)", "SWIID or OWID inequality", ert_source, "GDP (OWID)", "Schooling (OWID)"],
            "estimator_used": estimator_used,
            "detrending_method": "Hodrick-Prescott (lambda=6.25 annual)",
            "window_size": 20,
            "num_workers": NUM_WORKERS,
        },
        "datasets": [
            {
                "dataset": "welfare_state_democracy_pipeline",
                "examples": examples,
            }
        ],
    }

    # ── Also save detailed method_out.json ───────────────────────────────────
    detailed = {
        "step0_montecarlo": {k: v for k, v in step0.items() if k != "results"},
        "step0_mc_results_sample": step0.get("results", [])[:9],
        "step1_recovery_rates": step1,
        "step2_csd_anticipation": step2,
        "step3_rate_regression": {k: v for k, v in step3.items() if k != "all_models"},
        "step3_all_models": step3.get("all_models", {}),
        "step4_hazard": step4,
        "holm_correction": holm,
        "metadata": method_out["metadata"],
    }
    (WS / "method_out_detailed.json").write_text(json.dumps(detailed, indent=2, default=str))
    logger.info("Detailed output saved to method_out_detailed.json")

    # ── Save schema-compliant output ──────────────────────────────────────────
    (WS / "method_out.json").write_text(json.dumps(method_out, indent=2, default=str))
    logger.info("method_out.json saved")

    # Summary
    logger.info("=" * 60)
    logger.info(f"PIPELINE COMPLETE in {t_total}s")
    logger.info(f"  Pred1 (CSD AUC):        {'CONFIRMED' if step2.get('prediction1_confirmed') else 'NOT CONFIRMED'}")
    logger.info(f"  Pred2 (redistrib coeff): {'CONFIRMED' if step3.get('prediction2_confirmed') else 'NOT CONFIRMED'}")
    logger.info(f"  Pred3 (interaction):     {'CONFIRMED' if step4.get('prediction3_confirmed') else 'NOT CONFIRMED'}")
    logger.info(f"  Holm-adjusted:           {holm.get('rejected')}")


if __name__ == "__main__":
    main()
