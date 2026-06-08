"""Step 3: Primary panel regression of recovery rate on lagged realized redistribution."""

import numpy as np
import pandas as pd
from loguru import logger
from statsmodels.stats.multitest import multipletests


def _safe_ols(formula: str, data: pd.DataFrame, cov_type: str = "HC3",
              cov_kwds: dict | None = None) -> dict | None:
    try:
        import statsmodels.formula.api as smf
        mod = smf.ols(formula, data=data)
        if cov_kwds:
            fit = mod.fit(cov_type=cov_type, cov_kwds=cov_kwds)
        else:
            fit = mod.fit(cov_type=cov_type)
        return fit
    except Exception as e:
        logger.error(f"OLS failed: {e} | formula: {formula[:100]}")
        return None


def _extract_coeff(fit, param_name: str) -> dict:
    """Extract coefficient, SE, p-value for a parameter (partial match)."""
    if fit is None:
        return {"coeff": None, "se": None, "p": None}
    for name in fit.params.index:
        if param_name in name:
            return {
                "coeff": round(float(fit.params[name]), 5),
                "se": round(float(fit.bse[name]), 5),
                "p": round(float(fit.pvalues[name]), 4),
            }
    return {"coeff": None, "se": None, "p": None}


def _granger_fraction(panel: pd.DataFrame, y_col: str, x_col: str,
                      max_lag: int = 4) -> float:
    """Fraction of countries where x Granger-causes y at p<0.10."""
    from statsmodels.tsa.stattools import grangercausalitytests
    countries = panel["country"].unique()
    sig_count = 0
    valid_count = 0
    for country in countries:
        grp = panel[panel["country"] == country].sort_values("year")[[y_col, x_col]].dropna()
        if len(grp) < max_lag + 6:
            continue
        try:
            res = grangercausalitytests(grp[[y_col, x_col]].values, maxlag=max_lag, verbose=False)
            p_vals = [res[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)]
            if min(p_vals) < 0.10:
                sig_count += 1
            valid_count += 1
        except Exception:
            continue
    return round(sig_count / max(valid_count, 1), 3)


def run_step3(recovery_panel: pd.DataFrame, redistrib_df: pd.DataFrame,
              gdp_panel: pd.DataFrame, schooling_panel: pd.DataFrame) -> dict:
    logger.info("Step 3: Primary rate regression")

    if recovery_panel.empty:
        return _empty_result("Empty recovery panel")

    # ── Merge redistribution ──
    redist = redistrib_df.copy()
    if redist.empty or "redistribution" not in redist.columns:
        logger.warning("No redistribution data; Step 3 will have limited variables")
        redist = pd.DataFrame(columns=["country", "year", "redistribution", "gini_disp"])

    # Standardize country/year
    for df in [redist, gdp_panel, schooling_panel]:
        if "country" not in df.columns and "Entity" in df.columns:
            df.rename(columns={"Entity": "country"}, inplace=True)

    panel = recovery_panel.copy()
    panel = panel.rename(columns={"center_year": "year"})

    # Expand redistribution to annual via forward-fill (OECD IDD is sparse survey data)
    if not redist.empty and "redistribution" in redist.columns:
        redist_cols = ["country", "year"] + [c for c in ["redistribution", "gini_disp", "gini_mkt"] if c in redist.columns]
        redist_sub = redist[redist_cols].copy()
        # Build full country × year grid spanning redistrib coverage then ffill
        all_years = range(int(redist_sub["year"].min()), int(redist_sub["year"].max()) + 1)
        grids = []
        for country, grp in redist_sub.groupby("country"):
            grid = pd.DataFrame({"year": list(all_years), "country": country})
            grid = grid.merge(grp.drop(columns="country"), on="year", how="left")
            for col in redist_cols[2:]:
                grid[col] = grid[col].ffill(limit=5).bfill(limit=5)
            grids.append(grid)
        redist_filled = pd.concat(grids, ignore_index=True)

    # Merge covariates
    if not redist.empty and "redistribution" in redist.columns:
        panel = panel.merge(
            redist_filled[["country", "year"] + [c for c in ["redistribution", "gini_disp", "gini_mkt"]
                                                  if c in redist_filled.columns]],
            on=["country", "year"], how="left"
        )
    if not gdp_panel.empty and "log_gdp" in gdp_panel.columns:
        panel = panel.merge(gdp_panel[["country", "year", "log_gdp"]], on=["country", "year"], how="left")
    if not schooling_panel.empty and "schooling" in schooling_panel.columns:
        panel = panel.merge(schooling_panel[["country", "year", "schooling"]], on=["country", "year"], how="left")

    # Create lags (5-year primary, 3 and 7 for robustness)
    for col in ["redistribution", "log_gdp", "schooling"]:
        if col in panel.columns:
            for lag in [3, 5, 7]:
                panel = panel.sort_values(["country", "year"])
                panel[f"{col}_lag{lag}"] = panel.groupby("country")[col].shift(lag)

    # Add decade and region dummies (crude region from first letter of country)
    panel["decade"] = (panel["year"] // 10 * 10).astype(int)

    # Region assignment (rough heuristic from country name)
    region_map = _assign_regions(panel["country"].unique())
    panel["region"] = panel["country"].map(region_map).fillna("Other")

    logger.info(f"Panel shape after merge: {panel.shape}, countries: {panel['country'].nunique()}")

    # ── Primary regression ──
    results: dict = {}

    # Variables that exist in the panel
    has_redist = "redistribution_lag5" in panel.columns and panel["redistribution_lag5"].notna().sum() > 20
    has_gdp = "log_gdp_lag5" in panel.columns and panel["log_gdp_lag5"].notna().sum() > 50
    has_school = "schooling_lag5" in panel.columns and panel["schooling_lag5"].notna().sum() > 50
    has_gini = "gini_disp" in panel.columns and panel["gini_disp"].notna().sum() > 50

    base_controls = []
    if has_gdp:
        base_controls.append("log_gdp_lag5")
    if has_school:
        base_controls.append("schooling_lag5")
    if has_gini:
        base_controls.append("gini_disp")

    ctrl_str = " + ".join(base_controls) if base_controls else "1"

    p_redistrib = 1.0
    coeff_redistrib = None

    if has_redist:
        sub = panel.dropna(subset=["lambda_bc", "redistribution_lag5"])
        logger.info(f"Primary regression N={len(sub)}, countries={sub['country'].nunique()}")

        formula1 = f"lambda_bc ~ redistribution_lag5 + {ctrl_str} + C(region) + C(decade)"
        fit1 = _safe_ols(formula1, sub)
        if fit1 is not None:
            r1 = _extract_coeff(fit1, "redistribution_lag5")
            p_redistrib = r1["p"] if r1["p"] is not None else 1.0
            coeff_redistrib = r1["coeff"]
            results["mod1_primary"] = {
                "formula": formula1,
                "n_obs": len(sub),
                "n_countries": int(sub["country"].nunique()),
                "coeff_redistribution_lag5": r1["coeff"],
                "se_redistribution_lag5": r1["se"],
                "p_redistribution_lag5": r1["p"],
                "r2": round(float(fit1.rsquared), 4),
                "r2_adj": round(float(fit1.rsquared_adj), 4),
            }

        # Level regression: mean_edi ~ redistribution
        if "ac1_rolling" in panel.columns:
            formula3 = f"ac1_rolling ~ redistribution_lag5 + {ctrl_str} + C(region) + C(decade)"
            fit3 = _safe_ols(formula3, sub)
            if fit3 is not None:
                r3 = _extract_coeff(fit3, "redistribution_lag5")
                results["mod3_level"] = {
                    "formula": formula3,
                    "n_obs": len(sub),
                    "coeff_redistribution_lag5": r3["coeff"],
                    "p_redistribution_lag5": r3["p"],
                    "r2": round(float(fit3.rsquared), 4),
                }

        # Country FE
        sub2 = panel.dropna(subset=["lambda_bc", "redistribution_lag5"])
        if len(sub2) > 100 and sub2["country"].nunique() > 10:
            formula_fe = f"lambda_bc ~ redistribution_lag5 + {ctrl_str} + C(country)"
            fit_fe = _safe_ols(formula_fe, sub2, cov_type="cluster",
                               cov_kwds={"groups": sub2["country"]})
            if fit_fe is not None:
                r_fe = _extract_coeff(fit_fe, "redistribution_lag5")
                results["mod_fe"] = {
                    "formula": formula_fe,
                    "n_obs": len(sub2),
                    "coeff_redistribution_lag5": r_fe["coeff"],
                    "se_redistribution_lag5": r_fe["se"],
                    "p_redistribution_lag5": r_fe["p"],
                    "r2": round(float(fit_fe.rsquared), 4),
                }

        # Robustness: lag 3 and lag 7
        for lag in [3, 7]:
            col = f"redistribution_lag{lag}"
            if col in panel.columns:
                sub_r = panel.dropna(subset=["lambda_bc", col])
                if len(sub_r) > 50:
                    form_r = f"lambda_bc ~ {col} + {ctrl_str} + C(region) + C(decade)"
                    fit_r = _safe_ols(form_r, sub_r)
                    if fit_r is not None:
                        r_r = _extract_coeff(fit_r, col)
                        results[f"mod_robustness_lag{lag}"] = {
                            "coeff_redistribution": r_r["coeff"],
                            "p_redistribution": r_r["p"],
                            "n_obs": len(sub_r),
                        }

        # Granger causality
        granger_frac = _granger_fraction(
            panel[["country", "year", "lambda_bc", "redistribution"]].dropna(),
            "lambda_bc", "redistribution"
        )
    else:
        logger.warning("No redistribution_lag5 data; skipping primary regression")
        granger_frac = 0.0
        results["mod1_primary"] = {
            "formula": "N/A — redistribution data unavailable",
            "n_obs": 0,
            "n_countries": 0,
            "coeff_redistribution_lag5": None,
            "se_redistribution_lag5": None,
            "p_redistribution_lag5": None,
            "r2": None,
        }

    # Dissociation: level (ac1) ~ redistribution vs rate ~ redistribution
    p_level = None
    if "mod3_level" in results:
        p_level = results["mod3_level"].get("p_redistribution_lag5")
    dissociation_confirmed = bool(
        coeff_redistrib is not None and p_redistrib < 0.1
        and (p_level is None or p_level > 0.1)
    )

    pred2_confirmed = bool(
        coeff_redistrib is not None and coeff_redistrib > 0 and p_redistrib < 0.1
    )

    summary = {
        "n_obs": results.get("mod1_primary", {}).get("n_obs", 0),
        "n_countries": results.get("mod1_primary", {}).get("n_countries", 0),
        "coeff_redistribution_lag5": results.get("mod1_primary", {}).get("coeff_redistribution_lag5"),
        "se_redistribution_lag5": results.get("mod1_primary", {}).get("se_redistribution_lag5"),
        "p_redistribution_lag5": results.get("mod1_primary", {}).get("p_redistribution_lag5"),
        "r2": results.get("mod1_primary", {}).get("r2"),
        "granger_fraction_p10": granger_frac,
        "prediction2_confirmed": pred2_confirmed,
        "dissociation_confirmed": dissociation_confirmed,
        "all_models": results,
    }
    logger.info(f"Step 3: coeff={summary['coeff_redistribution_lag5']}, "
                f"p={summary['p_redistribution_lag5']}, pred2={pred2_confirmed}")
    return summary


def _assign_regions(countries: np.ndarray) -> dict:
    western = {"United States", "Canada", "Australia", "New Zealand", "Japan", "South Korea",
               "United Kingdom", "Germany", "France", "Italy", "Spain", "Portugal", "Netherlands",
               "Belgium", "Austria", "Switzerland", "Sweden", "Norway", "Denmark", "Finland",
               "Ireland", "Greece", "Luxembourg"}
    eastern_europe = {"Poland", "Hungary", "Czech Republic", "Slovakia", "Romania", "Bulgaria",
                      "Croatia", "Slovenia", "Serbia", "Albania", "North Macedonia", "Bosnia and Herzegovina",
                      "Montenegro", "Moldova", "Ukraine", "Belarus", "Lithuania", "Latvia", "Estonia"}
    latin_am = {"Brazil", "Argentina", "Chile", "Colombia", "Mexico", "Peru", "Venezuela",
                "Ecuador", "Bolivia", "Paraguay", "Uruguay", "Costa Rica", "Panama", "Cuba",
                "Dominican Republic", "Guatemala", "Honduras", "El Salvador", "Nicaragua", "Haiti"}
    africa = {"South Africa", "Nigeria", "Kenya", "Ethiopia", "Ghana", "Tanzania", "Senegal",
              "Cameroon", "Zambia", "Zimbabwe", "Uganda", "Mozambique", "Madagascar", "Mali",
              "Burkina Faso", "Niger", "Chad", "Sudan", "Egypt", "Morocco", "Tunisia", "Algeria"}
    asia = {"India", "China", "Indonesia", "Pakistan", "Bangladesh", "Philippines", "Thailand",
            "Vietnam", "Malaysia", "Myanmar", "Cambodia", "Sri Lanka", "Nepal", "Mongolia"}
    mideast = {"Turkey", "Iran", "Iraq", "Saudi Arabia", "Israel", "Jordan", "Lebanon",
               "Syria", "Yemen", "Kuwait", "UAE", "Qatar", "Bahrain", "Oman"}

    region_map: dict[str, str] = {}
    for c in countries:
        if c in western:
            region_map[c] = "Western"
        elif c in eastern_europe:
            region_map[c] = "Eastern_Europe"
        elif c in latin_am:
            region_map[c] = "Latin_America"
        elif c in africa:
            region_map[c] = "Africa"
        elif c in asia:
            region_map[c] = "Asia"
        elif c in mideast:
            region_map[c] = "Middle_East"
        else:
            region_map[c] = "Other"
    return region_map


def _empty_result(reason: str) -> dict:
    return {
        "n_obs": 0, "n_countries": 0,
        "coeff_redistribution_lag5": None,
        "se_redistribution_lag5": None,
        "p_redistribution_lag5": None,
        "r2": None,
        "granger_fraction_p10": None,
        "prediction2_confirmed": False,
        "dissociation_confirmed": False,
        "all_models": {},
        "skip_reason": reason,
    }
