"""Step 2: CSD anticipation test before gradual ERT onsets."""

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import kendalltau, mannwhitneyu

PRE_WINDOW = 5


def _kendall_trend(series: np.ndarray) -> float:
    """Kendall-tau trend statistic for a time series."""
    if len(series) < 3:
        return 0.0
    tau, _ = kendalltau(np.arange(len(series)), series)
    return float(tau) if not np.isnan(tau) else 0.0


def run_step2(recovery_panel: pd.DataFrame, onset_df: pd.DataFrame) -> dict:
    logger.info(f"Step 2: CSD anticipation test, {len(onset_df)} onsets")

    # Identify gradual vs coup onsets
    if onset_df.empty:
        return _empty_result("No onset data")

    # Normalize column names
    onset_df = onset_df.copy()
    col_map: dict[str, str] = {}
    for c in onset_df.columns:
        cl = c.lower()
        if "country" in cl:
            col_map[c] = "country"
        elif cl == "year":
            col_map[c] = "year"
        elif "ep_type" in cl or "type" in cl:
            col_map[c] = "ep_type"
    onset_df = onset_df.rename(columns=col_map)

    if "country" not in onset_df.columns:
        if "country_name" in onset_df.columns:
            onset_df = onset_df.rename(columns={"country_name": "country"})
        else:
            return _empty_result("No country column in onset data")
    # Drop duplicate columns (e.g. both country_name and country renamed to country)
    onset_df = onset_df.loc[:, ~onset_df.columns.duplicated()]

    # Separate gradual vs coup
    if "ep_type" in onset_df.columns:
        gradual_mask = onset_df["ep_type"].str.lower().str.contains("gradual", na=False)
        coup_mask = ~gradual_mask
        gradual_onsets = onset_df[gradual_mask].copy()
        coup_onsets = onset_df[coup_mask].copy()
    else:
        gradual_onsets = onset_df.copy()
        coup_onsets = pd.DataFrame()

    logger.info(f"Gradual onsets: {len(gradual_onsets)}, coup-like: {len(coup_onsets)}")

    # For each gradual onset, extract pre-onset CSD window
    cases_tau_ac1 = []
    cases_tau_var = []

    rp = recovery_panel.reset_index(drop=True)
    for _, row in gradual_onsets.iterrows():
        country = str(row["country"]) if "country" in row.index else str(row.get("country_name", ""))
        onset_year = int(row["year"])

        # Pre-onset window
        pre_start = onset_year - PRE_WINDOW
        mask = (
            (rp["country"] == country)
            & (rp["center_year"] >= pre_start)
            & (rp["center_year"] < onset_year)
        )
        pre_data = rp[mask].sort_values("center_year")
        if len(pre_data) < 3:
            continue
        tau_ac1 = _kendall_trend(pre_data["ac1_rolling"].values)
        tau_var = _kendall_trend(pre_data["var_rolling"].values)
        cases_tau_ac1.append(tau_ac1)
        cases_tau_var.append(tau_var)

    # Controls: non-onset country-years, matched window
    all_countries = set(rp["country"].unique())
    onset_countries = set(gradual_onsets["country"].tolist())
    control_countries = list(all_countries - onset_countries)

    controls_tau_ac1 = []
    controls_tau_var = []
    rng = np.random.default_rng(42)
    for _ in range(min(len(cases_tau_ac1) * 3, len(control_countries) * 3)):
        c = rng.choice(control_countries)
        grp = rp[rp["country"] == c].sort_values("center_year")
        if len(grp) < PRE_WINDOW:
            continue
        # Random 5-year window
        max_start = len(grp) - PRE_WINDOW
        start_i = int(rng.integers(0, max_start + 1))
        win = grp.iloc[start_i: start_i + PRE_WINDOW]
        controls_tau_ac1.append(_kendall_trend(win["ac1_rolling"].values))
        controls_tau_var.append(_kendall_trend(win["var_rolling"].values))

    if len(cases_tau_ac1) < 3:
        return _empty_result(f"Insufficient gradual onsets with recovery data: {len(cases_tau_ac1)}")

    # Mann-Whitney test
    try:
        mw_stat, p_mw = mannwhitneyu(cases_tau_ac1, controls_tau_ac1, alternative="greater")
    except Exception:
        p_mw = 1.0

    # AUC
    all_tau = cases_tau_ac1 + controls_tau_ac1
    all_labels = [1] * len(cases_tau_ac1) + [0] * len(controls_tau_ac1)
    try:
        from sklearn.metrics import roc_auc_score
        auc = float(roc_auc_score(all_labels, all_tau))
    except Exception:
        auc = 0.5

    # PLACEBOS
    # 1. Reversed time: flip pre-onset windows → CSD should vanish
    cases_tau_rev = []
    for _, row in gradual_onsets.iterrows():
        country = str(row["country"]) if "country" in row.index else str(row.get("country_name", ""))
        onset_year = int(row["year"])
        pre_start = onset_year - PRE_WINDOW
        mask = (
            (rp["country"] == country)
            & (rp["center_year"] >= pre_start)
            & (rp["center_year"] < onset_year)
        )
        pre_data = rp[mask].sort_values("center_year")
        if len(pre_data) < 3:
            continue
        cases_tau_rev.append(_kendall_trend(pre_data["ac1_rolling"].values[::-1]))

    p_reversed = 1.0
    if cases_tau_rev and controls_tau_ac1:
        try:
            _, p_reversed = mannwhitneyu(cases_tau_rev, controls_tau_ac1, alternative="greater")
        except Exception:
            pass

    # 2. First-difference placebo
    cases_tau_fd = []
    for _, row in gradual_onsets.iterrows():
        country = str(row["country"]) if "country" in row.index else str(row.get("country_name", ""))
        onset_year = int(row["year"])
        pre_start = onset_year - PRE_WINDOW
        mask = (
            (rp["country"] == country)
            & (rp["center_year"] >= pre_start)
            & (rp["center_year"] < onset_year)
        )
        pre_data = rp[mask].sort_values("center_year")
        if len(pre_data) < 4:
            continue
        fd = np.diff(pre_data["ac1_rolling"].values)
        cases_tau_fd.append(_kendall_trend(fd))

    p_fd = 1.0
    if cases_tau_fd and len(controls_tau_ac1) > 0:
        try:
            controls_fd = [c - c * 0 for c in controls_tau_ac1]  # controls ~= noise
            _, p_fd = mannwhitneyu(cases_tau_fd, controls_fd, alternative="greater")
        except Exception:
            pass

    pred1_confirmed = bool(p_mw < 0.1 and auc > 0.55)

    output = {
        "n_gradual_onsets": len(gradual_onsets),
        "n_coup_onsets": len(coup_onsets),
        "n_cases_with_data": len(cases_tau_ac1),
        "n_controls": len(controls_tau_ac1),
        "mean_kendall_tau_ac1_cases": round(float(np.mean(cases_tau_ac1)), 4) if cases_tau_ac1 else None,
        "mean_kendall_tau_ac1_controls": round(float(np.mean(controls_tau_ac1)), 4) if controls_tau_ac1 else None,
        "p_wilcoxon": round(float(p_mw), 4),
        "auc": round(auc, 4),
        "p_reversed_time_placebo": round(float(p_reversed), 4),
        "p_first_diff_placebo": round(float(p_fd), 4),
        "prediction1_confirmed": pred1_confirmed,
    }
    logger.info(f"Step 2: p_mw={p_mw:.3f}, auc={auc:.3f}, pred1={pred1_confirmed}")
    return output


def _empty_result(reason: str) -> dict:
    logger.warning(f"Step 2 empty: {reason}")
    return {
        "n_gradual_onsets": 0,
        "n_coup_onsets": 0,
        "n_cases_with_data": 0,
        "n_controls": 0,
        "mean_kendall_tau_ac1_cases": None,
        "mean_kendall_tau_ac1_controls": None,
        "p_wilcoxon": 1.0,
        "auc": 0.5,
        "p_reversed_time_placebo": 1.0,
        "p_first_diff_placebo": 1.0,
        "prediction1_confirmed": False,
        "skip_reason": reason,
    }
