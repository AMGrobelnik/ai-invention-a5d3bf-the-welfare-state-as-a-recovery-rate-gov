#!/usr/bin/env python3
"""Build democratic resilience country-year panel (1960-2023)."""

import io
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pycountry
import requests
from loguru import logger

WORKSPACE = Path(__file__).parent
OWID_TABLES = Path("/home/adrian/projects/ai-inventor/.claude/skills/aii-owid-datasets/temp/tables")
OUT_PATH = WORKSPACE / "data_out.json"
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(exist_ok=True)

import sys
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss}|{level:<7}|{message}")
logger.add(str(LOGS_DIR / "build_panel.log"), rotation="30 MB", level="DEBUG")

HEADERS = {"User-Agent": "Democratic Resilience Panel/1.0"}


# ---------------------------------------------------------------------------
# Helper: ISO-3 name lookup with pycountry + manual overrides
# ---------------------------------------------------------------------------

MANUAL_ISO = {
    "Czechia": "CZE", "Czech Republic": "CZE",
    "Slovak Republic": "SVK", "Slovakia": "SVK",
    "South Korea": "KOR", "Korea, Republic of": "KOR", "Korea": "KOR",
    "North Korea": "PRK", "Korea, Dem. People's Rep.": "PRK",
    "Laos": "LAO", "Lao PDR": "LAO",
    "Vietnam": "VNM", "Viet Nam": "VNM",
    "Bolivia": "BOL", "Bolivia (Plurinational State of)": "BOL",
    "Venezuela": "VEN", "Venezuela, RB": "VEN",
    "Iran": "IRN", "Iran, Islamic Republic of": "IRN",
    "Syria": "SYR", "Syrian Arab Republic": "SYR",
    "Moldova": "MDA", "Republic of Moldova": "MDA",
    "Russia": "RUS", "Russian Federation": "RUS",
    "Tanzania": "TZA", "United Republic of Tanzania": "TZA",
    "Congo": "COG", "Republic of Congo": "COG",
    "DR Congo": "COD", "Democratic Republic of Congo": "COD",
    "Democratic Republic of the Congo": "COD",
    "Cote d'Ivoire": "CIV", "Ivory Coast": "CIV",
    "Cape Verde": "CPV", "Cabo Verde": "CPV",
    "Timor": "TLS", "Timor-Leste": "TLS", "East Timor": "TLS",
    "Kosovo": "XKX",
    "Netherlands Antilles": None, "Yugoslavia": None, "Czechoslovakia": None,
    "German Democratic Republic": None, "East Germany": None, "West Germany": None,
    "USSR": None, "Soviet Union": None,
    "Palestine": "PSE", "Palestinian Territory": "PSE", "West Bank and Gaza": "PSE",
    "Micronesia": "FSM", "Federated States of Micronesia": "FSM",
    "Eswatini": "SWZ", "Swaziland": "SWZ",
    "North Macedonia": "MKD", "Macedonia": "MKD",
    "Gambia": "GMB", "The Gambia": "GMB",
    "Kyrgyzstan": "KGZ", "Kyrgyz Republic": "KGZ",
    "Egypt, Arab Rep.": "EGY", "Egypt": "EGY",
    "Turkey": "TUR", "Turkiye": "TUR",
    "United States": "USA",
    "United Kingdom": "GBR",
    "Saint Lucia": "LCA", "St. Lucia": "LCA",
    "Saint Vincent and the Grenadines": "VCT", "St. Vincent and Grenadines": "VCT",
    "São Tomé and Príncipe": "STP", "Sao Tome and Principe": "STP",
}

_iso_cache: dict[str, str | None] = {}


def name_to_iso3(name: str) -> str | None:
    if name in _iso_cache:
        return _iso_cache[name]
    if name in MANUAL_ISO:
        _iso_cache[name] = MANUAL_ISO[name]
        return MANUAL_ISO[name]
    try:
        c = pycountry.countries.lookup(name)
        _iso_cache[name] = c.alpha_3
        return c.alpha_3
    except LookupError:
        _iso_cache[name] = None
        return None


# ---------------------------------------------------------------------------
# STEP 1: V-Dem EDI + LDI from OWID grapher CSV
# ---------------------------------------------------------------------------

def load_vdem_edi_ldi() -> pd.DataFrame:
    logger.info("Fetching V-Dem EDI and LDI from OWID grapher CSVs")
    edi_url = "https://ourworldindata.org/grapher/electoral-democracy-index.csv?v=1&csvType=full&useColumnShortNames=false"
    ldi_url = "https://ourworldindata.org/grapher/liberal-democracy-index.csv?v=1&csvType=full&useColumnShortNames=false"

    def fetch_csv(url: str, val_col: str) -> pd.DataFrame:
        resp = requests.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        logger.info(f"  {val_col}: {df.shape}, cols={list(df.columns)}")
        # Drop regional aggregates (no ISO code)
        code_col = [c for c in df.columns if c.lower() in ("code", "country code")][0]
        df = df.dropna(subset=[code_col])
        df = df.rename(columns={"Entity": "country", code_col: "country_code", "Year": "year"})
        # Last column is the value
        val_original = [c for c in df.columns if c not in ("country", "country_code", "year")][0]
        df = df.rename(columns={val_original: val_col})
        return df[["country_code", "year", val_col]]

    edi = fetch_csv(edi_url, "edi")
    ldi = fetch_csv(ldi_url, "ldi")
    df = pd.merge(edi, ldi, on=["country_code", "year"], how="outer")
    df = df[df["year"].between(1960, 2023)]
    logger.info(f"V-Dem EDI+LDI panel: {df.shape}, countries={df['country_code'].nunique()}")
    return df


# ---------------------------------------------------------------------------
# STEP 2: ERT from GitHub raw CSV
# ---------------------------------------------------------------------------

def load_ert() -> pd.DataFrame:
    logger.info("Fetching ERT data from GitHub")
    ert_url = "https://raw.githubusercontent.com/vdeminstitute/ERT/master/inst/ERT.csv"
    try:
        resp = requests.get(ert_url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        df = pd.read_csv(io.StringIO(resp.text))
        logger.info(f"  ERT CSV: {df.shape}, cols={list(df.columns[:20])}")
    except Exception as e:
        logger.warning(f"GitHub ERT failed: {e}; trying OWID full table")
        df = _load_ert_owid_fallback()
        return df

    # Identify key columns
    country_col = next((c for c in df.columns if "country" in c.lower() and "text" in c.lower()), None)
    if country_col is None:
        country_col = next((c for c in df.columns if "country" in c.lower()), "country_text_id")
    year_col = next((c for c in df.columns if c.lower() in ("year", "v2x_veracc_osp")), "year")
    logger.info(f"  ERT country_col={country_col}, year_col={year_col}")
    logger.debug(f"  ERT all cols: {list(df.columns)}")

    # episode type: look for reg_type or similar
    type_cols = [c for c in df.columns if "type" in c.lower() or "speed" in c.lower() or "coup" in c.lower()]
    onset_cols = [c for c in df.columns if "onset" in c.lower() or "start" in c.lower()]
    logger.info(f"  ERT type_cols={type_cols}, onset_cols={onset_cols}")

    df = df.rename(columns={country_col: "country_code", year_col: "year"})
    df = df[df["year"].between(1960, 2023)]

    # Build ert_onset_gradual and ert_onset_coup
    # Standard ERT has: av_start_year (onset year), av_reg_type (1=democratic_breakdown, 2=autocratization within democracy)
    # The episode type (gradual vs coup) is in "av_type": 1=gradual,2=abrupt or similar
    # Check columns carefully
    av_type_col = next((c for c in df.columns if c in ("av_type", "reg_type", "aut_type", "d_type")), None)
    av_start_col = next((c for c in df.columns if "start" in c.lower() or "onset" in c.lower() or c == "av_start_year"), None)
    av_ongoing_col = next((c for c in df.columns if c in ("av_ongoing", "ongoing", "episode_ongoing")), None)

    logger.info(f"  av_type_col={av_type_col}, av_start_col={av_start_col}, av_ongoing_col={av_ongoing_col}")

    # Minimal ERT: mark onset years based on episode start
    # If we can't find episode type, derive from regime_dich_ert changes
    result_rows = []
    if av_start_col and av_type_col:
        for (cc, yr), grp in df.groupby(["country_code", "year"]):
            onset_g, onset_c = 0, 0
            ep_id = None
            starts = grp[grp[av_start_col] == yr]
            if len(starts) > 0:
                row = starts.iloc[0]
                t = row.get(av_type_col, np.nan)
                ep_id = f"{cc}_{yr}"
                if pd.notna(t):
                    if int(t) == 1:
                        onset_g = 1
                    else:
                        onset_c = 1
                else:
                    onset_g = 1  # default to gradual if unknown
            result_rows.append({"country_code": cc, "year": yr,
                                 "ert_onset_gradual": onset_g,
                                 "ert_onset_coup": onset_c,
                                 "ert_episode_id": ep_id})
        ert_out = pd.DataFrame(result_rows)
    else:
        # Minimal fallback: use OWID ERT data
        logger.warning("ERT column structure not recognized; using OWID table as fallback")
        return _load_ert_owid_fallback()

    logger.info(f"ERT panel: {ert_out.shape}")
    return ert_out


def _load_ert_owid_fallback() -> pd.DataFrame:
    """Use OWID ERT table to derive onset years from regime transitions."""
    logger.info("  Loading ERT from OWID full table")
    path = OWID_TABLES / "full_garden_democracy_2025-05-05_ert_ert.json"
    if not path.exists():
        logger.error(f"OWID ERT table not found at {path}")
        return pd.DataFrame(columns=["country_code", "year", "ert_onset_gradual", "ert_onset_coup", "ert_episode_id"])

    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    logger.info(f"  OWID ERT: {df.shape}, cols={list(df.columns)}")

    # country column may be "country" (name) — need ISO code mapping
    # OWID ERT doesn't include ISO code in this table, use OWID country→ISO mapping via vdem backbone
    # We'll add ISO later during merge using country name
    df = df[df["year"].between(1960, 2023)].copy()
    df["ert_onset_gradual"] = np.nan
    df["ert_onset_coup"] = np.nan
    df["ert_episode_id"] = None

    # Derive onset from regime transitions: when regime_dich_ert changes 1→0 or regime_trich drops
    df = df.sort_values(["country", "year"])
    df["_prev_dich"] = df.groupby("country")["regime_dich_ert"].shift(1)
    # onset = year where regime_dich_ert transitions from 1 (democracy) to 0 (autocracy)
    onset_mask = (df["regime_dich_ert"] == 0) & (df["_prev_dich"] == 1)
    df.loc[onset_mask, "ert_onset_gradual"] = 0
    df.loc[onset_mask, "ert_onset_coup"] = 0
    df.loc[onset_mask, "ert_episode_id"] = df.loc[onset_mask, "country"].astype(str) + "_" + df.loc[onset_mask, "year"].astype(str)

    df = df.drop(columns=["_prev_dich"], errors="ignore")
    df = df.rename(columns={"country": "country_name_ert"})
    logger.info(f"ERT fallback (OWID): {df.shape}")
    return df


# ---------------------------------------------------------------------------
# STEP 3: OWID redistribution (WID market Gini, LIS net Gini)
# ---------------------------------------------------------------------------

def load_owid_gini() -> pd.DataFrame:
    logger.info("Loading LIS Gini (net + market) from OWID full table")
    path = OWID_TABLES / "full_garden_lis_2023-08-30_luxembourg_income_study_luxembourg_income_study.json"
    if not path.exists():
        logger.warning("LIS table not found; skipping OWID redistribution")
        return pd.DataFrame(columns=["country_code", "year", "gini_market_wid", "gini_net_lis"])

    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)

    # gini_mi_eq = market income Gini; gini_dhi_eq = disposable income Gini
    cols_needed = ["country", "year", "gini_mi_eq", "gini_dhi_eq"]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        logger.warning(f"LIS missing cols: {missing}")
        return pd.DataFrame(columns=["country_code", "year", "gini_market_wid", "gini_net_lis"])

    df = df[cols_needed].copy()
    df = df.rename(columns={
        "gini_mi_eq": "gini_market_wid",
        "gini_dhi_eq": "gini_net_lis",
    })

    # Map country name to ISO-3
    df["country_code"] = df["country"].map(name_to_iso3)
    unmatched = df[df["country_code"].isna()]["country"].unique()
    if len(unmatched):
        logger.debug(f"LIS unmatched countries: {list(unmatched)[:20]}")
    df = df.dropna(subset=["country_code"])
    df = df[df["year"].between(1960, 2023)]
    df = df[["country_code", "year", "gini_market_wid", "gini_net_lis"]]
    logger.info(f"LIS Gini: {df.shape}, countries={df['country_code'].nunique()}")
    return df


# ---------------------------------------------------------------------------
# STEP 4: SWIID from GitHub releases
# ---------------------------------------------------------------------------

def _extract_swiid_csv_from_zip(zip_path: Path) -> pd.DataFrame | None:
    """Stream-parse a zip whose central directory is missing, extract swiid summary CSV."""
    import struct, zlib
    data = zip_path.read_bytes()
    offset = 0
    entries: list[tuple[str, int, int]] = []
    while offset < len(data) - 30:
        if data[offset:offset+4] == b'PK\x03\x04':
            fname_len = struct.unpack_from('<H', data, offset+26)[0]
            extra_len = struct.unpack_from('<H', data, offset+28)[0]
            compressed_size = struct.unpack_from('<I', data, offset+18)[0]
            fname = data[offset+30:offset+30+fname_len].decode('utf-8', errors='replace')
            entries.append((fname, compressed_size, offset))
            offset += 30 + fname_len + extra_len + compressed_size
        else:
            offset += 1

    # Pick highest-version summary CSV (prefer v7 or later)
    summary_entries = [(n, s, o) for n, s, o in entries
                       if 'summary' in n.lower() and n.endswith('.csv')]
    if not summary_entries:
        return None
    # Sort by version number in filename descending
    def _ver(name: str) -> int:
        import re
        m = re.search(r'swiid(\d+)_(\d+)', name)
        return int(m.group(1)) * 100 + int(m.group(2)) if m else 0
    fname, compressed_size, entry_offset = max(summary_entries, key=lambda e: _ver(e[0]))
    logger.info(f"  Extracting SWIID CSV: {fname}")
    compression = struct.unpack_from('<H', data, entry_offset+8)[0]
    fname_len = struct.unpack_from('<H', data, entry_offset+26)[0]
    extra_len = struct.unpack_from('<H', data, entry_offset+28)[0]
    data_offset = entry_offset + 30 + fname_len + extra_len
    raw = data[data_offset:data_offset+compressed_size]
    if compression == 8:
        raw = zlib.decompress(raw, -15)
    return pd.read_csv(io.StringIO(raw.decode('utf-8')))


def load_swiid() -> pd.DataFrame:
    logger.info("Loading SWIID v7 summary CSV from GitHub repo zipball")
    empty = pd.DataFrame(columns=["country_code", "year", "gini_market", "gini_net",
                                   "redistribution_swiid", "redistribution_swiid_se"])
    zip_url = "https://api.github.com/repos/fsolt/swiid/zipball/v9.6"
    zip_path = WORKSPACE / "logs" / "swiid_repo.zip"

    if not zip_path.exists():
        try:
            resp = requests.get(zip_url, headers=HEADERS, timeout=300, allow_redirects=True)
            resp.raise_for_status()
            zip_path.write_bytes(resp.content)
            logger.info(f"  Downloaded SWIID repo zip: {zip_path.stat().st_size / 1e6:.1f} MB")
        except Exception as e:
            logger.warning(f"  SWIID download failed: {e}")
            return empty

    df = _extract_swiid_csv_from_zip(zip_path)
    if df is None:
        logger.warning("  No SWIID summary CSV found in zip")
        return empty

    logger.info(f"  SWIID summary: {df.shape}, cols={list(df.columns)}")

    if df is None:
        logger.warning("SWIID unavailable; skipping")
        return empty

    # Map SWIID columns
    col_map = {
        "gini_mkt": "gini_market", "gini_disp": "gini_net",
        "abs_red": "redistribution_swiid", "abs_red_se": "redistribution_swiid_se",
    }
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    if "gini_market" not in df.columns and "gini_net" not in df.columns:
        logger.error(f"SWIID unexpected columns: {list(df.columns)}")
        return pd.DataFrame(columns=["country_code", "year", "gini_market", "gini_net",
                                     "redistribution_swiid", "redistribution_swiid_se"])

    # Compute abs redistribution if not present
    if "redistribution_swiid" not in df.columns:
        if "gini_market" in df.columns and "gini_net" in df.columns:
            df["redistribution_swiid"] = df["gini_market"] - df["gini_net"]
        else:
            df["redistribution_swiid"] = np.nan

    if "redistribution_swiid_se" not in df.columns:
        df["redistribution_swiid_se"] = np.nan

    # Country name → ISO-3
    if "country" in df.columns:
        df["country_code"] = df["country"].map(name_to_iso3)
        unmatched = df[df["country_code"].isna()]["country"].unique()
        logger.info(f"  SWIID unmatched countries ({len(unmatched)}): {list(unmatched)[:30]}")
        df = df.dropna(subset=["country_code"])
    else:
        logger.error("SWIID has no 'country' column")
        return pd.DataFrame(columns=["country_code", "year", "gini_market", "gini_net",
                                     "redistribution_swiid", "redistribution_swiid_se"])

    df = df[df["year"].between(1960, 2023)]
    keep = ["country_code", "year", "gini_market", "gini_net", "redistribution_swiid", "redistribution_swiid_se"]
    df = df[[c for c in keep if c in df.columns]]
    logger.info(f"SWIID: {df.shape}, countries={df['country_code'].nunique()}")
    return df


# ---------------------------------------------------------------------------
# STEP 5: OECD SOCX from OWID full table
# ---------------------------------------------------------------------------

def load_socx() -> pd.DataFrame:
    logger.info("Loading OECD SOCX from OWID full table")
    # Try the long-run combined table first (broader coverage)
    long_run_path = OWID_TABLES / "full_garden_social_expenditure_2025-03-07_social_expenditure_omm_social_expenditure_o.json"
    socx_path = OWID_TABLES / "full_garden_oecd_2025-02-25_social_expenditure_social_expenditure.json"

    # Try long-run table
    for path in [long_run_path, socx_path]:
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        logger.info(f"  SOCX table {path.name}: {df.shape}, cols={list(df.columns[:15])}")

        if path == long_run_path:
            # This table may have total public social spending directly
            val_cols = [c for c in df.columns if "public" in c.lower() or "spend" in c.lower() or "gdp" in c.lower() or "share" in c.lower()]
            logger.info(f"  Long-run SOCX value cols: {val_cols}")
            if val_cols:
                val_col = val_cols[0]
                df2 = df[["country", "year", val_col]].copy()
                df2 = df2.rename(columns={val_col: "gross_socx"})
                df2["country_code"] = df2["country"].map(name_to_iso3)
                df2 = df2.dropna(subset=["country_code"])
                df2 = df2[df2["year"].between(1960, 2023)]
                df2 = df2[["country_code", "year", "gross_socx"]].dropna(subset=["gross_socx"])
                logger.info(f"  Long-run SOCX: {df2.shape}")
                return df2

        if path == socx_path:
            # Filter: Public + All programme_type_category + Total
            if "expenditure_source" in df.columns:
                mask = (
                    (df["expenditure_source"].str.lower().str.startswith("public")) &
                    (df.get("programme_type_category", pd.Series(["All"] * len(df))).str.lower() == "all") &
                    (df.get("programme_type", pd.Series(["Total"] * len(df))).str.lower() == "total")
                )
                df = df[mask].copy()
            if "share_gdp" in df.columns:
                df2 = df[["country", "year", "share_gdp"]].copy()
                df2 = df2.rename(columns={"share_gdp": "gross_socx"})
                df2["country_code"] = df2["country"].map(name_to_iso3)
                df2 = df2.dropna(subset=["country_code"])
                df2 = df2[df2["year"].between(1960, 2023)]
                df2 = df2.drop(columns=["country"])
                logger.info(f"  OECD SOCX filtered: {df2.shape}")
                return df2

    logger.warning("No SOCX data found")
    return pd.DataFrame(columns=["country_code", "year", "gross_socx"])


# ---------------------------------------------------------------------------
# STEP 6: GDP per capita and schooling from OWID
# ---------------------------------------------------------------------------

def load_gdp() -> pd.DataFrame:
    logger.info("Loading GDP per capita from Penn World Table (rgdpe_pc)")
    # Penn World Table: rgdpe_pc = expenditure-side real GDP per capita (2017 PPP USD)
    path = OWID_TABLES / "full_garden_ggdc_2025-10-09_penn_world_table_penn_world_table.json"
    if not path.exists():
        path = OWID_TABLES / "full_garden_ggdc_2022-11-28_penn_world_table_penn_world_table.json"
    if not path.exists():
        logger.warning("Penn World Table not found")
        return pd.DataFrame(columns=["country_code", "year", "gdp_pc"])

    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    logger.info(f"  PWT table: {df.shape}, cols={list(df.columns[:10])}")

    df = df[["country", "year", "rgdpe_pc"]].copy()
    df = df.rename(columns={"rgdpe_pc": "gdp_pc"})
    df["country_code"] = df["country"].map(name_to_iso3)
    unmatched = df[df["country_code"].isna()]["country"].unique()
    if len(unmatched):
        logger.debug(f"  PWT unmatched countries: {list(unmatched)[:20]}")
    df = df.dropna(subset=["country_code", "gdp_pc"])
    df = df[df["year"].between(1960, 2023)]
    df = df[["country_code", "year", "gdp_pc"]]
    logger.info(f"GDP: {df.shape}, countries={df['country_code'].nunique()}")
    return df


def load_schooling() -> pd.DataFrame:
    logger.info("Loading mean years of schooling from OWID")
    # Try Lee-Lee 2016 / Barro-Lee 2018 / UNDP table
    path = OWID_TABLES / "full_backport_owid_latest_dataset_4129_years_of_schooling__based_on_lee_lee__2016__ba.json"
    if not path.exists():
        logger.warning("Schooling table not found")
        return pd.DataFrame(columns=["country_code", "year", "schooling"])

    with open(path) as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    logger.info(f"  Schooling table: {df.shape}, cols={list(df.columns)}")

    code_col = next((c for c in df.columns if "code" in c.lower()), None)
    name_col = next((c for c in df.columns if "name" in c.lower() or c.lower() == "country"), None)
    val_col = next((c for c in df.columns if "school" in c.lower()), None)

    logger.info(f"  Schooling code_col={code_col}, val_col={val_col}")

    if code_col:
        df2 = df[[code_col, "year", val_col]].copy()
        df2 = df2.rename(columns={code_col: "country_code", val_col: "schooling"})
    elif name_col:
        df2 = df[[name_col, "year", val_col]].copy()
        df2["country_code"] = df2[name_col].map(name_to_iso3)
        df2 = df2.rename(columns={val_col: "schooling"})
        df2 = df2.drop(columns=[name_col])
    else:
        return pd.DataFrame(columns=["country_code", "year", "schooling"])

    df2 = df2.dropna(subset=["country_code"])
    df2 = df2[df2["year"].between(1960, 2023)]
    logger.info(f"Schooling: {df2.shape}, countries={df2['country_code'].nunique()}")
    return df2


# ---------------------------------------------------------------------------
# STEP 7: Merge all sources
# ---------------------------------------------------------------------------

def merge_panel(
    vdem: pd.DataFrame,
    ert: pd.DataFrame,
    swiid: pd.DataFrame,
    owid_gini: pd.DataFrame,
    socx: pd.DataFrame,
    gdp: pd.DataFrame,
    schooling: pd.DataFrame,
) -> pd.DataFrame:
    logger.info("Merging all sources on (country_code, year)")

    panel = vdem.copy()
    logger.info(f"  Backbone (V-Dem): {panel.shape}")

    # ERT: if using OWID fallback with country_name_ert, map to ISO
    if "country_name_ert" in ert.columns:
        ert = ert.copy()
        ert["country_code"] = ert["country_name_ert"].map(name_to_iso3)
        ert = ert.drop(columns=["country_name_ert"], errors="ignore")
        ert = ert.dropna(subset=["country_code"])

    for df, label in [
        (ert, "ERT"),
        (swiid, "SWIID"),
        (owid_gini, "OWID Gini"),
        (socx, "SOCX"),
        (gdp, "GDP"),
        (schooling, "Schooling"),
    ]:
        if df.empty:
            logger.warning(f"  {label}: empty, skipping")
            continue
        before = len(panel)
        panel = panel.merge(df, on=["country_code", "year"], how="left")
        logger.info(f"  After {label}: {panel.shape} (rows changed: {len(panel) - before})")

    return panel


# ---------------------------------------------------------------------------
# STEP 8: Derived columns
# ---------------------------------------------------------------------------

def compute_derived(panel: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing derived columns")

    # redistribution_owid = market - net Gini from LIS
    if "gini_market_wid" in panel.columns and "gini_net_lis" in panel.columns:
        panel["redistribution_owid"] = panel["gini_market_wid"] - panel["gini_net_lis"]
    else:
        panel["redistribution_owid"] = np.nan

    # Ensure ert_onset columns: 0 for countries with ERT data, NaN where no data
    for col in ["ert_onset_gradual", "ert_onset_coup"]:
        if col not in panel.columns:
            panel[col] = np.nan

    if "ert_episode_id" not in panel.columns:
        panel["ert_episode_id"] = None

    # democratic_stock: cumulative years with EDI >= 0.5 prior to current year (EDI threshold = 0.5)
    logger.info("  Computing democratic_stock (EDI >= 0.5 threshold)")
    if "edi" in panel.columns:
        panel = panel.sort_values(["country_code", "year"])
        panel["_is_dem"] = (panel["edi"] >= 0.5).astype(float)
        panel["_is_dem"] = panel["_is_dem"].where(panel["edi"].notna(), np.nan)
        # cumsum per country, exclude current year
        panel["democratic_stock"] = (
            panel.groupby("country_code")["_is_dem"]
            .transform(lambda s: s.shift(1).fillna(0).cumsum())
        )
        panel = panel.drop(columns=["_is_dem"])
    else:
        panel["democratic_stock"] = np.nan

    return panel


# ---------------------------------------------------------------------------
# STEP 9: Coverage summary and output
# ---------------------------------------------------------------------------

def coverage_stats(panel: pd.DataFrame, col: str) -> dict:
    s = panel[col].dropna()
    if s.empty:
        return {"n_obs": 0, "n_countries": 0, "year_range": [None, None]}
    subset = panel.dropna(subset=[col])
    return {
        "n_obs": int(len(s)),
        "n_countries": int(subset["country_code"].nunique()),
        "year_range": [int(subset["year"].min()), int(subset["year"].max())],
    }


def build_output(panel: pd.DataFrame) -> dict:
    logger.info("Building output JSON")

    output_cols = [
        "country_code", "year",
        "edi", "ldi",
        "redistribution_owid",
        "redistribution_swiid", "redistribution_swiid_se",
        "gini_market", "gini_net",
        "gross_socx",
        "gdp_pc",
        "schooling",
        "ert_onset_gradual", "ert_onset_coup", "ert_episode_id",
        "democratic_stock",
    ]
    # Keep only present columns
    output_cols = [c for c in output_cols if c in panel.columns]
    panel = panel[output_cols].sort_values(["country_code", "year"])

    coverage_vars = [c for c in output_cols if c not in ("country_code", "year", "ert_episode_id")]
    coverage = {c: coverage_stats(panel, c) for c in coverage_vars}

    rows = []
    for _, row in panel.iterrows():
        r = {}
        for col in output_cols:
            val = row[col]
            if pd.isna(val) if not isinstance(val, str) else (val is None):
                r[col] = None
            elif isinstance(val, (int, np.integer)):
                r[col] = int(val)
            elif isinstance(val, (float, np.floating)):
                r[col] = None if math.isnan(val) else round(float(val), 6)
            else:
                r[col] = val
        rows.append(r)

    out = {
        "metadata": {
            "description": "Democratic resilience country-year panel, 1960-2023",
            "edi_threshold_for_democracy": 0.5,
            "sources": {
                "edi_ldi": "V-Dem via OWID grapher CSV (electoral-democracy-index, liberal-democracy-index)",
                "ert": "V-Dem ERT GitHub raw CSV or OWID ERT table fallback",
                "redistribution_swiid": "SWIID (Solt 2020) market-net Gini",
                "redistribution_owid": "Luxembourg Income Study (OWID) market-disposable Gini difference",
                "gross_socx": "OECD SOCX or OWID long-run social expenditure",
                "gdp_pc": "OWID GDP historical (Maddison/WB)",
                "schooling": "Lee-Lee 2016 / Barro-Lee 2018 via OWID",
            },
            "coverage": coverage,
        },
        "rows": rows,
    }
    logger.info(f"Output: {len(rows)} rows, {panel['country_code'].nunique()} countries")
    return out


@logger.catch(reraise=True)
def main():
    vdem = load_vdem_edi_ldi()
    ert = load_ert()
    swiid = load_swiid()
    owid_gini = load_owid_gini()
    socx = load_socx()
    gdp = load_gdp()
    schooling = load_schooling()

    panel = merge_panel(vdem, ert, swiid, owid_gini, socx, gdp, schooling)
    panel = compute_derived(panel)

    out = build_output(panel)

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    sz = OUT_PATH.stat().st_size / 1e6
    logger.info(f"Wrote {OUT_PATH} ({sz:.1f} MB)")


if __name__ == "__main__":
    main()
