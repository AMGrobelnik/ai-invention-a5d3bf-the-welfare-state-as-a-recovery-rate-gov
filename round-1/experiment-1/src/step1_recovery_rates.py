"""Step 1: Per-country bias-corrected recovery-rate panel."""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from loguru import logger

from estimators import ols_ar1, andrews_mu_ar1, ou_mle_ar1
from utils import detrend_hp, detrend_gp

WINDOW = 20
STEP = 1
N_BOOT_CI = 100


def _rolling_recovery_country(args: tuple) -> list[dict]:
    country, years, edi, estimator_name, window, n_boot = args
    try:
        if len(edi) < window + 3:
            return []

        # Detrend with HP (fast, well-validated for annual data)
        try:
            resid, trend = detrend_hp(edi)
        except Exception:
            resid = edi - np.mean(edi)

        results = []
        for i in range(0, len(resid) - window + 1, STEP):
            win = resid[i: i + window]
            if np.std(win) < 1e-6:
                continue
            if estimator_name == "Andrews_MU":
                phi_est = andrews_mu_ar1(win)
            elif estimator_name == "OU_MLE":
                phi_est, _ = ou_mle_ar1(win)
            else:
                phi_est = ols_ar1(win)

            phi_est = float(np.clip(phi_est, 0.01, 0.999))
            lambda_est = -np.log(phi_est)
            center_year = int(years[i + window // 2])

            # Rolling CSD indicators
            ac1 = float(np.corrcoef(win[:-1], win[1:])[0, 1]) if np.std(win) > 0 else 0.0
            var = float(np.var(win))

            # Delta-method CI for lambda (fast, no bootstrap)
            # SE(phi) ≈ (1-phi^2)/sqrt(T); SE(lambda) ≈ SE(phi)/(phi)
            se_phi = (1.0 - phi_est ** 2) / np.sqrt(window)
            se_lambda = se_phi / phi_est
            lambda_lo = max(0.0, lambda_est - 1.96 * se_lambda)
            lambda_hi = lambda_est + 1.96 * se_lambda

            results.append({
                "country": country,
                "center_year": center_year,
                "phi_bc": round(phi_est, 5),
                "lambda_bc": round(lambda_est, 5),
                "lambda_lo": round(lambda_lo, 5),
                "lambda_hi": round(lambda_hi, 5),
                "ac1_rolling": round(ac1, 5),
                "var_rolling": round(var, 8),
                "n_obs_window": window,
            })
        return results
    except Exception as e:
        logger.error(f"Country {country} failed: {e}")
        return []


def run_step1(vdem_panel: pd.DataFrame, estimator_name: str = "Andrews_MU",
              num_workers: int = 4) -> dict:
    logger.info(f"Step 1: Rolling recovery rates, estimator={estimator_name}, window={WINDOW}")

    countries = vdem_panel["country"].unique()
    tasks = []
    for country in countries:
        grp = vdem_panel[vdem_panel["country"] == country].sort_values("year")
        years = grp["year"].values.astype(float)
        edi = grp["edi"].values.astype(float)
        if len(edi) < WINDOW + 3:
            continue
        tasks.append((country, years, edi, estimator_name, WINDOW, N_BOOT_CI))

    logger.info(f"Processing {len(tasks)} countries with {num_workers} workers")

    all_rows = []
    with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(_rolling_recovery_country, t): t[0] for t in tasks}
        done = 0
        for fut in as_completed(futures):
            cname = futures[fut]
            try:
                rows = fut.result()
                all_rows.extend(rows)
                done += 1
                if done % 20 == 0:
                    logger.info(f"  Step1: {done}/{len(tasks)} countries done, {len(all_rows)} rows so far")
            except Exception as e:
                logger.error(f"Country {cname} exception: {e}")

    panel = pd.DataFrame(all_rows)
    if panel.empty:
        logger.warning("Step 1 produced no results!")
        return {"n_countries": 0, "n_country_years": 0, "panel": panel, "sample_rows": [],
                "signal_above_noise": False}

    n_countries = int(panel["country"].nunique())
    n_cy = len(panel)
    logger.info(f"Step 1 complete: {n_countries} countries, {n_cy} country-years")

    # Signal check: is variance of lambda_bc substantially above noise?
    lam_std = float(panel["lambda_bc"].std())
    lam_mean = float(panel["lambda_bc"].mean())
    signal_ok = lam_std > 0.02 and lam_mean > 0.0

    sample = panel.head(5).to_dict("records")

    return {
        "n_countries": n_countries,
        "n_country_years": n_cy,
        "panel": panel,
        "sample_rows": sample,
        "signal_above_noise": signal_ok,
        "lambda_mean": round(lam_mean, 4),
        "lambda_std": round(lam_std, 4),
    }
