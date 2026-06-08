"""Shared utilities: V-Dem panel building, detrending, lag creation."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger


# ─── Hardware detection ──────────────────────────────────────────────────────

def detect_cpus() -> int:
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


NUM_CPUS = detect_cpus()


# ─── V-Dem panel normalization ───────────────────────────────────────────────

def build_vdem_panel(vdem_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize V-Dem OWID table into (country, year, edi) panel.

    OWID V-Dem table uses `electoff_vdem` (Elected Officials Index, 0–1) as
    the continuous democracy indicator closest to v2x_polyarchy.
    """
    df = vdem_raw.copy()

    # Rename columns
    col_map: dict[str, str] = {}
    for c in df.columns:
        cl = c.lower().strip()
        if cl == "entity":
            col_map[c] = "country"
        elif cl == "year":
            col_map[c] = "year"
        # OWID V-Dem table: elected officials index is the primary continuous EDI proxy
        elif cl == "electoff_vdem":
            col_map[c] = "edi"
        elif "electoral democracy" in cl or "v2x_polyarchy" in cl:
            col_map[c] = "edi"
        elif "liberal democracy" in cl or "v2x_libdem" in cl:
            col_map[c] = "ldi"
    df = df.rename(columns=col_map)

    # Pick best EDI column if not yet found
    if "edi" not in df.columns:
        # Try suffrage * elected officials proxy
        if "suffr_vdem" in df.columns and "electoff_vdem" in df.columns:
            df["edi"] = df["suffr_vdem"].astype(float) / 100.0 * df["electoff_vdem"].astype(float)
            logger.warning("Using suffr_vdem * electoff_vdem as EDI proxy")
        else:
            num_cols = [c for c in df.columns if c not in ("country", "year", "Code", "code")
                        and pd.api.types.is_numeric_dtype(df[c])]
            if num_cols:
                df = df.rename(columns={num_cols[0]: "edi"})
                logger.warning(f"Using column '{num_cols[0]}' as EDI proxy")
            else:
                raise ValueError(f"No EDI column found in V-Dem data. Columns: {list(df.columns)}")

    df = df[["country", "year", "edi"]].dropna()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["edi"] = pd.to_numeric(df["edi"], errors="coerce")
    df = df.dropna()
    df["country"] = df["country"].astype(str).str.strip()

    # Filter out aggregates
    exclude = {"World", "Europe", "Asia", "Africa", "Americas", "Oceania",
               "High income", "Low income", "Middle income"}
    df = df[~df["country"].isin(exclude)]

    logger.info(f"V-Dem panel: {df.shape}, {df['country'].nunique()} countries, "
                f"years {int(df['year'].min())}-{int(df['year'].max())}")
    return df.sort_values(["country", "year"]).reset_index(drop=True)


def build_gdp_panel(gdp_raw: pd.DataFrame) -> pd.DataFrame:
    """Build log GDP per capita panel.

    Handles PWT format (rgdpe in millions, pop in millions → gdp_pc = rgdpe/pop)
    and legacy OWID CSV format (Entity/Year/gdp_per_capita column).
    """
    df = gdp_raw.copy()

    # PWT format: rgdpe (real GDP, millions 2017 int-$) + pop (millions)
    if "rgdpe" in df.columns and "pop" in df.columns:
        df["country"] = df["country"].astype(str).str.strip()
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["rgdpe"] = pd.to_numeric(df["rgdpe"], errors="coerce")
        df["pop"] = pd.to_numeric(df["pop"], errors="coerce")
        df = df.dropna(subset=["country", "year", "rgdpe", "pop"])
        df = df[df["pop"] > 0]
        df["gdp_pc"] = df["rgdpe"] / df["pop"]  # both in millions → per-person int-$
        df["log_gdp"] = np.log(df["gdp_pc"].clip(lower=1))
        exclude = {"World", "Europe", "Asia", "Africa", "Americas", "Oceania"}
        df = df[~df["country"].isin(exclude)]
        logger.info(f"GDP panel (PWT): {df.shape}, {df['country'].nunique()} countries")
        return df[["country", "year", "log_gdp"]].dropna()

    # Legacy OWID CSV format
    col_map: dict[str, str] = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "entity":
            col_map[c] = "country"
        elif cl == "year":
            col_map[c] = "year"
        elif "gdp" in cl and ("per capita" in cl or "pc" in cl or "per_capita" in cl):
            col_map[c] = "gdp_pc"
    if "gdp_pc" not in [col_map.get(c, c) for c in df.columns]:
        num_cols = [c for c in df.columns if c not in ("Entity", "entity", "Year", "year", "Code", "code")
                    and pd.api.types.is_numeric_dtype(df[c])]
        if num_cols:
            col_map[num_cols[0]] = "gdp_pc"
    df = df.rename(columns=col_map)
    keep = [c for c in ["country", "year", "gdp_pc"] if c in df.columns]
    df = df[keep].dropna()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if "gdp_pc" in df.columns:
        df["log_gdp"] = np.log(pd.to_numeric(df["gdp_pc"], errors="coerce").clip(lower=1))
    return df[["country", "year", "log_gdp"]].dropna()


def build_schooling_panel(sch_raw: pd.DataFrame) -> pd.DataFrame:
    """Build schooling panel. Uses PWT human capital index (hc) as proxy if available."""
    df = sch_raw.copy()

    # PWT format: hc = human capital index (based on schooling + returns to education)
    if "hc" in df.columns and "country" in df.columns and "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df["schooling"] = pd.to_numeric(df["hc"], errors="coerce")
        exclude = {"World", "Europe", "Asia", "Africa", "Americas", "Oceania"}
        df = df[~df["country"].isin(exclude)]
        return df[["country", "year", "schooling"]].dropna()

    col_map: dict[str, str] = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "entity":
            col_map[c] = "country"
        elif cl == "year":
            col_map[c] = "year"
        elif "school" in cl or "education" in cl or "human capital" in cl:
            col_map[c] = "schooling"
    df = df.rename(columns=col_map)
    keep = [c for c in ["country", "year", "schooling"] if c in df.columns]
    if len(keep) < 3:
        return pd.DataFrame(columns=["country", "year", "schooling"])
    df = df[keep].dropna()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["schooling"] = pd.to_numeric(df["schooling"], errors="coerce")
    return df.dropna()


# ─── Detrending ──────────────────────────────────────────────────────────────

def detrend_hp(edi: np.ndarray, lamb: float = 6.25) -> tuple[np.ndarray, np.ndarray]:
    """Hodrick-Prescott filter (Ravn-Uhlig: lambda=6.25 for annual)."""
    from statsmodels.tsa.filters.hp_filter import hpfilter
    if len(edi) < 8:
        trend = np.full_like(edi, np.mean(edi))
        return edi - trend, trend
    cycle, trend = hpfilter(edi, lamb=lamb)
    return np.asarray(cycle), np.asarray(trend)


def detrend_gp(years: np.ndarray, edi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """GP detrending with RBF kernel (long length scale)."""
    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel
        kernel = RBF(length_scale=15, length_scale_bounds=(5, 50)) + WhiteKernel()
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        X = years.reshape(-1, 1).astype(float)
        gp.fit(X, edi)
        trend = gp.predict(X)
        return edi - trend, trend
    except Exception as e:
        logger.warning(f"GP detrending failed ({e}), falling back to HP")
        return detrend_hp(edi)


# ─── Lag creation ────────────────────────────────────────────────────────────

def add_lag(df: pd.DataFrame, col: str, lag: int, group_col: str = "country") -> pd.DataFrame:
    df = df.sort_values([group_col, "year"])
    df[f"{col}_lag{lag}"] = df.groupby(group_col)[col].shift(lag)
    return df


def add_rolling_mean(df: pd.DataFrame, col: str, window: int,
                     group_col: str = "country") -> pd.DataFrame:
    df = df.sort_values([group_col, "year"])
    df[f"{col}_roll{window}"] = (
        df.groupby(group_col)[col]
        .transform(lambda x: x.rolling(window, min_periods=window // 2).mean())
    )
    return df
