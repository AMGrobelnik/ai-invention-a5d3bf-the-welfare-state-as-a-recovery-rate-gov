"""Step 4: Hazard model with inequality × redistribution interaction."""

import numpy as np
import pandas as pd
from loguru import logger


def run_step4(vdem_panel: pd.DataFrame, onset_df: pd.DataFrame,
              redistrib_df: pd.DataFrame, gdp_panel: pd.DataFrame) -> dict:
    logger.info("Step 4: Hazard model with interaction")

    if onset_df.empty or vdem_panel.empty:
        return _empty_result("Insufficient input data")

    # ── Build spell dataset ──
    onset_df = onset_df.copy()
    # Normalize country column
    for c in onset_df.columns:
        if "country" in c.lower() and c != "country":
            onset_df = onset_df.rename(columns={c: "country"})
            break

    # Get onset info
    if "country" not in onset_df.columns:
        return _empty_result("No country column in onset data")
    # Drop duplicate columns created by rename above
    onset_df = onset_df.loc[:, ~onset_df.columns.duplicated()]

    # Map country → first onset year (gradual only)
    if "ep_type" in onset_df.columns:
        gradual_mask = onset_df["ep_type"].str.lower().str.contains("gradual", na=False)
        grad_df = onset_df[gradual_mask]
    else:
        grad_df = onset_df

    onset_years: dict[str, int] = {}
    for _, row in grad_df.iterrows():
        country = str(row["country"]) if "country" in row.index else ""
        year = int(row["year"])
        if country not in onset_years or year < onset_years[country]:
            onset_years[country] = year

    # Build country-year democratic spells: all years before onset (or all if no onset)
    vdem = vdem_panel.copy()
    vdem = vdem.sort_values(["country", "year"])

    spell_rows = []
    for country, grp in vdem.groupby("country"):
        grp = grp.sort_values("year").reset_index(drop=True)
        onset_yr = onset_years.get(country)
        for _, row in grp.iterrows():
            yr = int(row["year"])
            # Only include years before or at onset (drop post-onset)
            if onset_yr is not None and yr > onset_yr:
                continue
            # Exclude very early (< 1960)
            if yr < 1960:
                continue
            spell_rows.append({
                "country": country,
                "year": yr,
                "edi": float(row["edi"]),
                "onset": 1 if (onset_yr is not None and yr == onset_yr) else 0,
            })

    spell_df = pd.DataFrame(spell_rows)
    if spell_df.empty:
        return _empty_result("Empty spell dataset")

    # Merge covariates
    if not redistrib_df.empty and "redistribution" in redistrib_df.columns:
        spell_df = spell_df.merge(
            redistrib_df[["country", "year"] + [c for c in ["redistribution", "gini_disp"]
                                                  if c in redistrib_df.columns]],
            on=["country", "year"], how="left"
        )
    if not gdp_panel.empty and "log_gdp" in gdp_panel.columns:
        spell_df = spell_df.merge(gdp_panel[["country", "year", "log_gdp"]], on=["country", "year"], how="left")

    # Create lags
    for col in ["redistribution", "gini_disp", "edi", "log_gdp"]:
        if col in spell_df.columns:
            spell_df = spell_df.sort_values(["country", "year"])
            spell_df[f"{col}_lag1"] = spell_df.groupby("country")[col].shift(1)

    # Decade
    spell_df["decade"] = (spell_df["year"] // 10 * 10).astype(int)

    # Democratic age: years in sample (proxy for dem_age)
    spell_df["dem_age"] = spell_df.groupby("country").cumcount()

    n_spells = len(spell_df)
    n_onsets = int(spell_df["onset"].sum())
    logger.info(f"Spell dataset: {n_spells} rows, {n_onsets} onsets")

    if n_onsets < 5:
        return _empty_result(f"Too few onset events: {n_onsets}")

    # ── Logistic regression ──
    has_redistrib = "redistribution_lag1" in spell_df.columns and spell_df["redistribution_lag1"].notna().sum() > 30
    has_gini = "gini_disp_lag1" in spell_df.columns and spell_df["gini_disp_lag1"].notna().sum() > 30
    has_gdp = "log_gdp_lag1" in spell_df.columns and spell_df["log_gdp_lag1"].notna().sum() > 30

    p_interaction = 1.0
    coeff_interaction = None

    if has_redistrib and has_gini:
        sub = spell_df.dropna(subset=["onset", "redistribution_lag1", "gini_disp_lag1"])
        if len(sub) > 50 and sub["onset"].sum() >= 5:
            formula = (
                "onset ~ gini_disp_lag1 * redistribution_lag1"
                + (" + log_gdp_lag1" if has_gdp else "")
                + " + dem_age + C(decade)"
            )
            try:
                import statsmodels.formula.api as smf
                fit = smf.logit(formula, data=sub).fit(maxiter=200, disp=0)
                # Extract interaction coefficient
                for name in fit.params.index:
                    if "gini_disp_lag1:redistribution_lag1" in name or "redistribution_lag1:gini_disp_lag1" in name:
                        coeff_interaction = round(float(fit.params[name]), 5)
                        p_interaction = round(float(fit.pvalues[name]), 4)
                        break

                # Marginal effects: d(onset_prob)/d(gini) at different redistribution levels
                redist_q25 = float(sub["redistribution_lag1"].quantile(0.25))
                redist_q75 = float(sub["redistribution_lag1"].quantile(0.75))
                marginal_effects = {
                    "at_redistrib_q25": _marginal_effect_gini(fit, sub, redist_q25),
                    "at_redistrib_q75": _marginal_effect_gini(fit, sub, redist_q75),
                }

                # Also try cloglog (GLM)
                cloglog_result = _fit_cloglog(sub, has_gdp)

                n_spells_model = len(sub)
                n_onsets_model = int(sub["onset"].sum())
            except Exception as e:
                logger.error(f"Logit failed: {e}")
                marginal_effects = {}
                cloglog_result = {}
                n_spells_model = 0
                n_onsets_model = 0
        else:
            logger.warning(f"Step 4: insufficient data after subset, N={len(sub)}, onsets={sub['onset'].sum() if len(sub) > 0 else 0}")
            marginal_effects = {}
            cloglog_result = {}
            n_spells_model = 0
            n_onsets_model = 0
    else:
        logger.warning("Step 4: missing redistribution or gini data")
        marginal_effects = {}
        cloglog_result = {}
        n_spells_model = n_spells
        n_onsets_model = n_onsets

        # Fallback: simple logit with edi_lag1 only
        if "edi_lag1" in spell_df.columns:
            sub_fb = spell_df.dropna(subset=["onset", "edi_lag1"])
            if len(sub_fb) > 50 and sub_fb["onset"].sum() >= 5:
                try:
                    import statsmodels.formula.api as smf
                    fit_fb = smf.logit("onset ~ edi_lag1 + dem_age + C(decade)", data=sub_fb).fit(maxiter=200, disp=0)
                    r_edi = {}
                    for name in fit_fb.params.index:
                        if "edi_lag1" in name:
                            r_edi = {"coeff": round(float(fit_fb.params[name]), 5),
                                     "p": round(float(fit_fb.pvalues[name]), 4)}
                    logger.info(f"Fallback logit edi_lag1: {r_edi}")
                except Exception as e:
                    logger.error(f"Fallback logit failed: {e}")

    pred3_confirmed = bool(coeff_interaction is not None and coeff_interaction < 0 and p_interaction < 0.1)

    return {
        "n_spells": n_spells,
        "n_onsets": n_onsets,
        "n_spells_model": n_spells_model if "n_spells_model" in dir() else n_spells,
        "n_onsets_model": n_onsets_model if "n_onsets_model" in dir() else n_onsets,
        "coeff_interaction": coeff_interaction,
        "p_interaction": round(float(p_interaction), 4),
        "direction": "High redistribution attenuates inequality hazard" if pred3_confirmed else "Not confirmed",
        "marginal_effects": marginal_effects if "marginal_effects" in dir() else {},
        "cloglog": cloglog_result if "cloglog_result" in dir() else {},
        "prediction3_confirmed": pred3_confirmed,
    }


def _marginal_effect_gini(fit, data: pd.DataFrame, redist_val: float) -> float | None:
    """Marginal effect of gini on onset probability at given redistribution level."""
    try:
        mean_row = data.mean(numeric_only=True).to_dict()
        mean_row["redistribution_lag1"] = redist_val
        # Linearized marginal effect from logit
        params = fit.params
        # Find gini and interaction coefficients
        beta_gini = 0.0
        beta_inter = 0.0
        for name, val in params.items():
            if name == "gini_disp_lag1":
                beta_gini = float(val)
            elif "gini_disp_lag1:redistribution_lag1" in name or "redistribution_lag1:gini_disp_lag1" in name:
                beta_inter = float(val)
        marginal = beta_gini + beta_inter * redist_val
        return round(marginal, 5)
    except Exception:
        return None


def _fit_cloglog(sub: pd.DataFrame, has_gdp: bool) -> dict:
    """Complementary log-log discrete hazard."""
    try:
        import statsmodels.api as sm
        y = sub["onset"].values
        X_cols = ["gini_disp_lag1", "redistribution_lag1", "dem_age"]
        if has_gdp:
            X_cols.append("log_gdp_lag1")
        X_sub = sub[X_cols].dropna()
        idx = X_sub.index
        y_sub = y[sub.index.get_indexer(idx)] if hasattr(sub.index, "get_indexer") else y[sub.index.isin(idx)]
        if len(X_sub) < 30:
            return {}
        X_sm = sm.add_constant(X_sub)
        # Add interaction manually
        X_sm["gini_x_redist"] = X_sm["gini_disp_lag1"] * X_sm["redistribution_lag1"]
        y_sm = sub.loc[X_sub.index, "onset"].values
        fit = sm.GLM(y_sm, X_sm, family=sm.families.Binomial(link=sm.families.links.CLogLog())).fit()
        inter_coeff = None
        inter_p = None
        if "gini_x_redist" in fit.params:
            inter_coeff = round(float(fit.params["gini_x_redist"]), 5)
            inter_p = round(float(fit.pvalues["gini_x_redist"]), 4)
        return {"coeff_interaction": inter_coeff, "p_interaction": inter_p, "n_obs": len(y_sm)}
    except Exception as e:
        logger.warning(f"Cloglog failed: {e}")
        return {}


def _empty_result(reason: str) -> dict:
    return {
        "n_spells": 0, "n_onsets": 0,
        "coeff_interaction": None, "p_interaction": None,
        "direction": "N/A", "marginal_effects": {}, "cloglog": {},
        "prediction3_confirmed": False,
        "skip_reason": reason,
    }
