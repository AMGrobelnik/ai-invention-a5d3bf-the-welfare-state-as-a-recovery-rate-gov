"""Load all required datasets from OWID local JSON files."""

import json
from pathlib import Path

import pandas as pd
from loguru import logger

OWID_TABLE_DIR = Path(
    "/home/adrian/projects/ai-inventor/.claude/skills/aii-owid-datasets/temp/tables"
)

# OWID table file paths
_VDEM_FILE = OWID_TABLE_DIR / "full_garden_democracy_2024-03-07_vdem_vdem.json"
_PWT_FILE = OWID_TABLE_DIR / "full_garden_ggdc_2025-07-31_penn_world_table_penn_world_table.json"
_OECD_FILE = OWID_TABLE_DIR / "full_garden_oecd_2025-04-16_income_distribution_database_income_distribution_database.json"


def _load_json_table(path: Path) -> pd.DataFrame:
    logger.info(f"Loading {path.name}")
    raw = json.loads(path.read_text())
    # OWID JSON is a list of dicts
    if isinstance(raw, list):
        return pd.DataFrame(raw)
    if isinstance(raw, dict) and "data" in raw:
        return pd.DataFrame(raw["data"])
    raise ValueError(f"Unexpected JSON structure in {path.name}")


def load_vdem_owid() -> pd.DataFrame:
    """V-Dem country-year table. Uses electoff_vdem (elected officials index) as EDI."""
    df = _load_json_table(_VDEM_FILE)
    logger.info(f"V-Dem raw: {df.shape}, cols: {list(df.columns)[:10]}")
    return df


def load_gdp() -> pd.DataFrame:
    """Penn World Table: rgdpe (real GDP, expenditure side) and pop (millions)."""
    df = _load_json_table(_PWT_FILE)
    logger.info(f"PWT raw: {df.shape}, cols: {list(df.columns)[:8]}")
    return df


def load_schooling() -> pd.DataFrame:
    """PWT human capital index as schooling proxy."""
    df = _load_json_table(_PWT_FILE)
    return df


def load_redistribution() -> pd.DataFrame:
    """OECD IDD: gini_market - gini_disposable = redistribution."""
    df = _load_json_table(_OECD_FILE)
    logger.info(f"OECD IDD raw: {df.shape}, cols: {list(df.columns)}")

    # Filter to "Total" age group for whole-population estimates
    if "age" in df.columns:
        df = df[df["age"] == "Total"].copy()

    col_map: dict[str, str] = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "country":
            col_map[c] = "country"
        elif cl == "year":
            col_map[c] = "year"
        elif cl == "gini_market":
            col_map[c] = "gini_mkt"
        elif cl == "gini_disposable":
            col_map[c] = "gini_disp"
        elif cl == "gini_reduction":
            col_map[c] = "gini_reduction"
    df = df.rename(columns=col_map)

    needed = [c for c in ["country", "year", "gini_mkt", "gini_disp"] if c in df.columns]
    df = df[needed].dropna(subset=["country", "year"])
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["gini_mkt"] = pd.to_numeric(df["gini_mkt"], errors="coerce")
    df["gini_disp"] = pd.to_numeric(df["gini_disp"], errors="coerce")
    df = df.dropna(subset=["gini_mkt", "gini_disp"])

    df["redistribution"] = df["gini_mkt"] - df["gini_disp"]
    logger.info(
        f"Redistribution panel: {df.shape}, {df['country'].nunique()} countries, "
        f"years {int(df['year'].min())}-{int(df['year'].max())}"
    )
    return df


def load_ert() -> pd.DataFrame | None:
    """ERT not available locally — return None, derive from V-Dem regime column."""
    logger.info("ERT: using algorithmic derivation from V-Dem regime_row_owid")
    return None


def derive_onsets_from_vdem(vdem_panel: pd.DataFrame) -> pd.DataFrame:
    """Derive backsliding onsets from sustained EDI decline > 0.05 over 3+ years."""
    records = []
    for country, grp in vdem_panel.groupby("country"):
        grp = grp.sort_values("year").reset_index(drop=True)
        if len(grp) < 6 or "edi" not in grp.columns:
            continue
        edi = grp["edi"].values
        years = grp["year"].values
        for i in range(2, len(edi) - 1):
            window = edi[max(0, i - 3): i + 1]
            if len(window) < 3:
                continue
            cum_drop = window[0] - window[-1]
            rate = cum_drop / len(window)
            if cum_drop > 0.05:
                ep_type = "gradual" if rate < 0.03 else "coup_like"
                records.append({
                    "country_name": country,
                    "country": country,
                    "year": int(years[i]),
                    "ep_type": ep_type,
                    "derived": True,
                })
    df = pd.DataFrame(records)
    if len(df):
        df = df.sort_values("year").groupby("country").first().reset_index()
    logger.info(f"Derived {len(df)} onsets algorithmically from V-Dem")
    return df


def load_all() -> dict:
    logger.info("=== Loading all datasets ===")
    data: dict = {}
    data["vdem_raw"] = load_vdem_owid()
    data["gdp_raw"] = load_gdp()
    data["schooling_raw"] = load_schooling()
    data["redistrib_raw"] = load_redistribution()
    data["ert_raw"] = load_ert()
    logger.info("=== Dataset loading complete ===")
    return data
