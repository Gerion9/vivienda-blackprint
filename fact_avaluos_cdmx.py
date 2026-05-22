#!/usr/bin/env python3
"""
fact_avaluos_cdmx.py
====================

Materializa una vista analítica para CDMX a partir de staging.stg_avaluos,
corre clustering K-Means sobre atributos físicos, calcula plusvalía nominal
por cluster (serie temporal 2019–2025), y emite un HTML editorial denso.

Uso:
    export BLACKPRINT_DB_PASSWORD='...'
    python3 fact_avaluos_cdmx.py                       # full pipeline → eda_cdmx.html
    python3 fact_avaluos_cdmx.py --skip-cluster        # solo descriptiva (sin K-Means)
    python3 fact_avaluos_cdmx.py --out reporte.html
    python3 fact_avaluos_cdmx.py --dump-ddl            # imprime el DDL sugerido (NO lo ejecuta)

READ-ONLY: el script SOLO lee staging.stg_avaluos. No crea, no modifica ni
escribe nada en la base de datos.

Decisiones clave (documentadas):
  - Filtro CDMX: cve_ent = '09'.
  - Filtro propósito: proposito = '1' (adquisición; 99.7% de los datos).
  - Winsorización a [p1, p99] por columna, calculada sobre toda CDMX (no por
    municipio — el rango intra-CDMX es lo que queremos preservar).
  - Clustering: K-Means con k = #clases observadas con n ≥ 200; features
    estandarizadas (sup_construida, sup_terreno, recamaras, banos,
    estacionamiento, edad_meses, niveles, m2_sv). random_state = 42.
  - Año 2025 SE MARCA como parcial (capture aún en curso).
  - Piso n ≥ 50 para reportar mediana por (cluster, año).
  - Plusvalía: % vs mediana m2_sv 2019, NOMINAL (sin deflactor INPC).
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.strtree import STRtree

# ---------------------------------------------------------------------------
# Conexión & constantes
# ---------------------------------------------------------------------------

DB = dict(
    host="35.211.253.113", port=5432,
    database="blackprint_db_prd", user="josue_user",
    password=os.environ.get("BLACKPRINT_DB_PASSWORD"),
)

CVE_ENT_CDMX = "09"

# Features base para clustering. m2_sv entra porque "depto chico caro" y
# "casa grande barata" son tipologías genuinamente distintas. lat/lon entran
# porque el "barrio" pesa fuerte en el mercado. NSE y POIs se suman al runtime.
PHYSICAL_FEATURES = [
    "sup_construida", "sup_terreno", "recamaras", "banos",
    "estacionamiento", "edad_meses", "niveles", "m2_sv",
]
GEO_FEATURES = ["latitud", "longitud"]
NSE_FEATURES = ["nse_score"]   # promedio ponderado del tier por alcaldía
# POI_FEATURES se elige dinámicamente (top-N business_category en CDMX).

# Buffer (m) para contar POIs alrededor de cada vivienda.
POI_BUFFER_M = 500
# Cuántas categorías de POI usar como features (top-N por volumen en CDMX).
POI_TOP_N = 6

# Punto de referencia para la proyección equirectangular (centro CDMX).
CDMX_REF_LAT = 19.43
CDMX_REF_LON = -99.13

# Piso de filas para mostrar mediana de serie temporal.
N_MIN_TIMESERIES = 50

# Año actual (capture parcial) — recalculado en runtime pero por defecto:
PARTIAL_YEAR = 2025

# ---------------------------------------------------------------------------
# DDL DE REFERENCIA — NO SE EJECUTA AUTOMÁTICAMENTE
# ---------------------------------------------------------------------------
# Este SQL queda aquí solo como documentación / sugerencia para cuando *tú*
# decidas (o no) materializar la vista por razones de performance.
# El script `--dump-ddl` lo imprime a stdout. Nunca se ejecuta.

DDL_FACT_AVALUOS_CDMX = """
DROP TABLE IF EXISTS presentation.fact_avaluos_cdmx;

CREATE TABLE presentation.fact_avaluos_cdmx AS
WITH parsed AS (
    SELECT
        id_avaluo,
        -- temporales
        CASE WHEN ano ~ '^\\d+$' THEN ano::int END                  AS ano,
        TO_DATE(NULLIF(fecha_avaluo, ''), 'DD/MM/YYYY')             AS fecha_avaluo,
        -- geografía
        cve_ent,
        LPAD(cve_mun, 3, '0')                                       AS cve_mun,
        nom_mun,
        cp,
        colonia,
        NULLIF(latitud,  '')::double precision                      AS latitud,
        NULLIF(longitud, '')::double precision                      AS longitud,
        -- tipología
        NULLIF(clase, '')::int                                      AS clase,
        NULLIF(tipo,  '')::int                                      AS tipo,
        uso_actual,
        NULLIF(conservacion, '')::int                               AS conservacion,
        -- atributos físicos
        NULLIF(sup_terreno,    '')::numeric                         AS sup_terreno,
        NULLIF(sup_construida, '')::numeric                         AS sup_construida,
        NULLIF(sup_vendible,   '')::numeric                         AS sup_vendible,
        NULLIF(edad_meses,     '')::int                             AS edad_meses,
        NULLIF(recamaras,      '')::int                             AS recamaras,
        NULLIF(banos,          '')::int                             AS banos,
        NULLIF(estacionamiento,'')::int                             AS estacionamiento,
        NULLIF(niveles,        '')::int                             AS niveles,
        -- valores monetarios (MXN nominales)
        NULLIF(valor_concluido,         '')::numeric                AS valor_concluido,
        NULLIF(valor_comparativo,       '')::numeric                AS valor_comparativo,
        NULLIF(valor_terreno_m2,        '')::numeric                AS valor_terreno_m2,
        NULLIF(m2_sv,                   '')::numeric                AS m2_sv,
        NULLIF(valor_fisico_construccion,'')::numeric               AS valor_fisico_construccion,
        NULLIF(valor_fisico_terreno,    '')::numeric                AS valor_fisico_terreno,
        -- otorgante (para perfilar mercado)
        siglas,
        grupo,
        -- propósito (filtro)
        proposito
    FROM staging.stg_avaluos
    WHERE cve_ent = '09'
)
SELECT *
FROM parsed
WHERE proposito = '1'
  AND clase IS NOT NULL
  AND valor_concluido IS NOT NULL AND valor_concluido > 0
  AND sup_construida  IS NOT NULL AND sup_construida  > 0
  AND m2_sv           IS NOT NULL AND m2_sv           > 0
  AND ano IS NOT NULL
  AND ano BETWEEN 2019 AND EXTRACT(YEAR FROM CURRENT_DATE)::int;

CREATE INDEX IF NOT EXISTS ix_fact_avaluos_cdmx_mun  ON presentation.fact_avaluos_cdmx (cve_mun);
CREATE INDEX IF NOT EXISTS ix_fact_avaluos_cdmx_ano  ON presentation.fact_avaluos_cdmx (ano);
CREATE INDEX IF NOT EXISTS ix_fact_avaluos_cdmx_cls  ON presentation.fact_avaluos_cdmx (clase);
"""

# Query para leer los datos a memoria (sea materializada o desde staging directo)
READ_QUERY = """
SELECT
    id_avaluo,
    CASE WHEN ano ~ '^\\d+$' THEN ano::int END                  AS ano,
    TO_DATE(NULLIF(fecha_avaluo, ''), 'DD/MM/YYYY')             AS fecha_avaluo,
    cve_mun, nom_mun, cp, colonia,
    NULLIF(latitud,  '')::double precision                      AS latitud,
    NULLIF(longitud, '')::double precision                      AS longitud,
    NULLIF(clase, '')::int                                      AS clase,
    NULLIF(tipo,  '')::int                                      AS tipo,
    NULLIF(sup_terreno,    '')::numeric                         AS sup_terreno,
    NULLIF(sup_construida, '')::numeric                         AS sup_construida,
    NULLIF(edad_meses,     '')::int                             AS edad_meses,
    NULLIF(recamaras,      '')::int                             AS recamaras,
    NULLIF(banos,          '')::int                             AS banos,
    NULLIF(estacionamiento,'')::int                             AS estacionamiento,
    NULLIF(niveles,        '')::int                             AS niveles,
    NULLIF(valor_concluido,            '')::numeric             AS valor_concluido,
    NULLIF(valor_terreno_m2,           '')::numeric             AS valor_terreno_m2,
    NULLIF(m2_sv,                      '')::numeric             AS m2_sv,
    NULLIF(valor_fisico_construccion,  '')::numeric             AS valor_fisico_construccion,
    NULLIF(constructor, '')                                     AS constructor,
    siglas, grupo
FROM staging.stg_avaluos
WHERE cve_ent = '09'
  AND proposito = '1'
  AND clase IS NOT NULL
  AND valor_concluido ~ '^\\d+(\\.\\d+)?$' AND valor_concluido::numeric > 0
  AND sup_construida  ~ '^\\d+(\\.\\d+)?$' AND sup_construida::numeric  > 0
  AND m2_sv           ~ '^\\d+(\\.\\d+)?$' AND m2_sv::numeric           > 0
  AND ano ~ '^\\d+$' AND ano::int BETWEEN 2019 AND EXTRACT(YEAR FROM CURRENT_DATE)::int
"""


# POIs en CDMX. Cargamos todos los puntos con lat/lon y main_category.
POI_QUERY = """
SELECT
    latitude::double precision  AS lat,
    longitude::double precision AS lon,
    business_category            AS main_category,
    cve_mun
FROM presentation.dim_pois_dataplor
WHERE cve_ent = '09'
  AND latitude  IS NOT NULL
  AND longitude IS NOT NULL
  AND business_category IS NOT NULL
"""


# NSE agregado al nivel de alcaldía (primero 5 chars del cvegeo AGEB =
# cve_ent + cve_mun). Sumamos hogares por tier.
NSE_QUERY = """
SELECT
    SUBSTRING(geo_code, 1, 5)         AS cve_full,
    SUBSTRING(geo_code, 3, 3)         AS cve_mun,
    SUM(COALESCE(ab,     0))::int     AS hh_ab,
    SUM(COALESCE(cplus,  0))::int     AS hh_cplus,
    SUM(COALESCE(c,      0))::int     AS hh_c,
    SUM(COALESCE(cminus, 0))::int     AS hh_cminus,
    SUM(COALESCE(dplus,  0))::int     AS hh_dplus,
    SUM(COALESCE(d,      0))::int     AS hh_d,
    SUM(COALESCE(e,      0))::int     AS hh_e
FROM presentation.dim_socioeconomic_level_ageb_locality
WHERE LEFT(geo_code, 2) = '09'
GROUP BY SUBSTRING(geo_code, 1, 5), SUBSTRING(geo_code, 3, 3)
"""


# Geometrías de alcaldías de CDMX.
MUNI_GEOM_QUERY = """
SELECT cvegeo, cve_mun, geometry_coords_json::text AS geom_raw
FROM presentation.dim_mexico_geometries
WHERE geometry_level = 'municipality'
  AND cve_ent = '09'
  AND geometry_coords_json IS NOT NULL
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def connect():
    conn = psycopg2.connect(connect_timeout=15, **DB)
    # Sesión read-only: cualquier intento de INSERT/UPDATE/DELETE/CREATE/DROP
    # falla con "cannot execute X in a read-only transaction". Garantía defensiva.
    conn.set_session(readonly=True, autocommit=False)
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = '600s'")
    conn.commit()
    return conn


def fmt_int(x):
    if x is None or (isinstance(x, float) and (np.isnan(x))): return "—"
    return f"{int(round(x)):,}"

def fmt_mxn(x):
    if x is None or (isinstance(x, float) and (np.isnan(x))): return "—"
    return f"${int(round(x)):,}"

def fmt_mxn_short(x):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    ax = abs(x)
    if ax >= 1e6: return f"${x/1e6:.2f}M"
    if ax >= 1e3: return f"${x/1e3:.0f}k"
    return f"${x:.0f}"

def fmt_num(x, decimals=1):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    return f"{x:,.{decimals}f}"

def fmt_pct(x, decimals=1):
    if x is None or (isinstance(x, float) and np.isnan(x)): return "—"
    return f"{x:+.{decimals}f}%"


def winsorize(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    s = s.copy()
    lo_v, hi_v = s.quantile(lo), s.quantile(hi)
    return s.clip(lower=lo_v, upper=hi_v)


def latlon_to_xy(lats: np.ndarray, lons: np.ndarray,
                  ref_lat: float = CDMX_REF_LAT,
                  ref_lon: float = CDMX_REF_LON) -> np.ndarray:
    """Proyección equirectangular simple alrededor de un punto de referencia.
    Error <1% para distancias <50km en CDMX — suficiente para conteos de POI
    en un buffer de 500m."""
    R = 6371000.0
    lat_rad = np.radians(np.asarray(lats, dtype=float))
    lon_rad = np.radians(np.asarray(lons, dtype=float))
    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)
    x = R * (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad)
    y = R * (lat_rad - ref_lat_rad)
    return np.column_stack([x, y])


# NSE: pesos por tier para colapsar a un score escalar (1=más bajo, 7=más alto).
NSE_TIER_WEIGHTS = {
    "hh_ab":      7.0,
    "hh_cplus":   6.0,
    "hh_c":       5.0,
    "hh_cminus":  4.0,
    "hh_dplus":   3.0,
    "hh_d":       2.0,
    "hh_e":       1.0,
}


def nse_score_from_row(row: pd.Series) -> float:
    """Score NSE promedio ponderado de hogares en una alcaldía.
    Devuelve NaN si no hay hogares."""
    total = sum(float(row.get(k, 0) or 0) for k in NSE_TIER_WEIGHTS)
    if total <= 0:
        return float("nan")
    num = sum(NSE_TIER_WEIGHTS[k] * float(row.get(k, 0) or 0) for k in NSE_TIER_WEIGHTS)
    return num / total


def nse_label_from_score(score: float) -> str:
    """Categoría legible a partir del score NSE."""
    if score is None or (isinstance(score, float) and np.isnan(score)): return "n/d"
    if score >= 6.0: return "AB"
    if score >= 5.2: return "C+"
    if score >= 4.4: return "C"
    if score >= 3.6: return "C-"
    if score >= 2.8: return "D+"
    if score >= 2.0: return "D"
    return "E"


# ---------------------------------------------------------------------------
# Análisis
# ---------------------------------------------------------------------------

def load_data(conn) -> pd.DataFrame:
    print("  → leyendo avalúos de CDMX desde staging.stg_avaluos…")
    df = pd.read_sql(READ_QUERY, conn)
    print(f"     · {len(df):,} filas")
    # Derivar trimestre desde fecha_avaluo (Q1..Q4); NaT → NaN.
    fecha = pd.to_datetime(df["fecha_avaluo"], errors="coerce")
    df["trimestre"] = fecha.dt.quarter.astype("Int64")
    return df


def load_pois(conn) -> pd.DataFrame:
    print("  → leyendo POIs de CDMX (dim_pois_dataplor)…")
    pois = pd.read_sql(POI_QUERY, conn)
    pois = pois.dropna(subset=["lat", "lon", "main_category"])
    pois = pois[
        pois["lat"].between(19.0, 19.7) &
        pois["lon"].between(-99.5, -98.85)
    ].reset_index(drop=True)
    print(f"     · {len(pois):,} POIs · {pois['main_category'].nunique()} main_categories")
    return pois


def load_nse_muni(conn) -> pd.DataFrame:
    print("  → leyendo NSE agregado a alcaldía (dim_socioeconomic_level_ageb_locality)…")
    nse = pd.read_sql(NSE_QUERY, conn)
    # nse_score por alcaldía
    nse["nse_score"] = nse.apply(nse_score_from_row, axis=1)
    nse["nse_total_hh"] = (nse[list(NSE_TIER_WEIGHTS.keys())].sum(axis=1)).astype(int)
    nse["nse_label"] = nse["nse_score"].apply(nse_label_from_score)
    print(f"     · {len(nse)} alcaldías con NSE.")
    return nse


def load_muni_geometries(conn) -> pd.DataFrame:
    print("  → leyendo geometrías de alcaldías (dim_mexico_geometries)…")
    g = pd.read_sql(MUNI_GEOM_QUERY, conn)
    print(f"     · {len(g)} polígonos.")
    return g


def coords_json_to_geojson(raw) -> dict | None:
    """Convierte la columna geometry_coords_json (nested arrays en 4326) a un
    dict GeoJSON geometry (Polygon | MultiPolygon). None si no se reconoce."""
    if raw is None: return None
    try:
        coords = raw if isinstance(raw, list) else json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not coords: return None
    try:
        if isinstance(coords[0][0], (int, float)):
            return {"type": "Polygon", "coordinates": [coords]}
        if isinstance(coords[0][0][0], (int, float)):
            return {"type": "Polygon", "coordinates": coords}
        if isinstance(coords[0][0][0][0], (int, float)):
            return {"type": "MultiPolygon", "coordinates": coords}
    except (IndexError, TypeError):
        return None
    return None


def coords_json_to_shapely(raw):
    """Para spatial joins. Devuelve shapely Polygon o MultiPolygon."""
    if raw is None: return None
    try:
        coords = raw if isinstance(raw, list) else json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not coords: return None
    try:
        if isinstance(coords[0][0], (int, float)):
            return Polygon(coords)
        if isinstance(coords[0][0][0], (int, float)):
            return Polygon(coords[0], coords[1:] if len(coords) > 1 else [])
        if isinstance(coords[0][0][0][0], (int, float)):
            return MultiPolygon([Polygon(r[0], r[1:] if len(r) > 1 else []) for r in coords])
    except (IndexError, TypeError, ValueError):
        return None
    return None


def attach_pois(df: pd.DataFrame, pois: pd.DataFrame,
                 top_n: int = POI_TOP_N, buffer_m: int = POI_BUFFER_M
                ) -> tuple[pd.DataFrame, list[str]]:
    """Para cada vivienda con lat/lon, cuenta POIs dentro de `buffer_m` por
    cada una de las top-`top_n` business_categories. Agrega columnas
    `poi_<cat>` al df. Devuelve (df_modificado, lista_categorías_usadas)."""
    top_cats = (pois["main_category"].value_counts()
                .head(top_n).index.tolist())
    pois_top = pois[pois["main_category"].isin(top_cats)].reset_index(drop=True)
    print(f"     · top-{top_n} categorías: {top_cats}")

    # KDTree de POIs en XY (metros)
    xy_pois = latlon_to_xy(pois_top["lat"].values, pois_top["lon"].values)
    tree = cKDTree(xy_pois)

    # Categoría → índice numérico (para bincount)
    cat_to_idx = {c: i for i, c in enumerate(top_cats)}
    cats_idx_arr = np.array([cat_to_idx[c] for c in pois_top["main_category"].values])

    # Para cada vivienda con lat/lon, query del KDTree
    has_geo = df["latitud"].notna() & df["longitud"].notna()
    counts = np.zeros((len(df), len(top_cats)), dtype=int)
    if has_geo.any():
        idx_geo = np.where(has_geo.values)[0]
        xy_viv = latlon_to_xy(df.loc[has_geo, "latitud"].values,
                               df.loc[has_geo, "longitud"].values)
        neighbors = tree.query_ball_point(xy_viv, r=buffer_m)
        for k, nbrs in enumerate(neighbors):
            if nbrs:
                bc = np.bincount(cats_idx_arr[nbrs], minlength=len(top_cats))
                counts[idx_geo[k]] = bc

    poi_cols = []
    for i, cat in enumerate(top_cats):
        # nombre seguro como columna
        safe = "poi_" + cat.replace(" ", "_").replace("/", "_").replace("-", "_")[:30]
        df[safe] = counts[:, i]
        poi_cols.append(safe)
    return df, poi_cols


def attach_nse(df: pd.DataFrame, nse: pd.DataFrame) -> pd.DataFrame:
    """Adjunta NSE de la alcaldía a cada vivienda por cve_mun."""
    # Asegurar cve_mun normalizado en ambos
    df["cve_mun_norm"] = df["cve_mun"].astype(str).str.zfill(3)
    nse_use = nse.assign(cve_mun_norm=nse["cve_mun"].astype(str).str.zfill(3))
    cols = ["cve_mun_norm", "nse_score", "nse_total_hh", "nse_label",
            "hh_ab", "hh_cplus", "hh_c", "hh_cminus", "hh_dplus", "hh_d", "hh_e"]
    nse_use = nse_use[cols]
    df = df.merge(nse_use, on="cve_mun_norm", how="left")
    df.drop(columns=["cve_mun_norm"], inplace=True)
    return df


def winsorize_block(df: pd.DataFrame) -> pd.DataFrame:
    """Recorta colas extremas en numéricas antes de cualquier agregación."""
    cols = ["valor_concluido", "valor_terreno_m2", "m2_sv",
            "sup_construida", "sup_terreno", "edad_meses",
            "valor_fisico_construccion"]
    for c in cols:
        if c in df.columns:
            df[c] = winsorize(df[c], 0.01, 0.99)
    return df


@dataclass
class DescriptiveStats:
    n_total: int
    n_municipios: int
    year_min: int
    year_max: int
    median_valor: float
    median_m2_sv: float
    median_terreno_m2: float
    p25_valor: float
    p75_valor: float
    by_clase: pd.DataFrame
    by_recamaras: pd.DataFrame
    by_banos: pd.DataFrame
    sup_construida_dist: list[tuple[float, float, int]]   # (lo, hi, count)
    sup_terreno_dist: list[tuple[float, float, int]]
    edad_meses_dist: list[tuple[float, float, int]]
    top_munis: pd.DataFrame
    # KPI por año / histogramas globales (alimenta el year-picker y el modal KPI)
    by_year_stats: dict = field(default_factory=dict)   # {"2019": {"valor":{median,p25,p75}, "m2_sv":{median}, "terreno":{median}}}
    by_year_hists: dict = field(default_factory=dict)   # {"2019": {"valor": hist_dict, ...}}
    global_hists:  dict = field(default_factory=dict)   # {"valor": hist_dict, "m2_sv": ..., "terreno": ...}


def describe(df: pd.DataFrame) -> DescriptiveStats:
    by_clase = (df.groupby("clase").agg(
        n=("id_avaluo", "count"),
        med_valor=("valor_concluido", "median"),
        med_m2=("m2_sv", "median"),
        med_sup=("sup_construida", "median"),
    ).reset_index().sort_values("clase"))

    by_rec = df["recamaras"].value_counts().sort_index().reset_index()
    by_rec.columns = ["recamaras", "n"]
    by_rec = by_rec[(by_rec["recamaras"] >= 0) & (by_rec["recamaras"] <= 8)]

    by_ban = df["banos"].value_counts().sort_index().reset_index()
    by_ban.columns = ["banos", "n"]
    by_ban = by_ban[(by_ban["banos"] >= 0) & (by_ban["banos"] <= 8)]

    def histogram(s: pd.Series, bins=20) -> list[tuple[float, float, int]]:
        s = s.dropna()
        if len(s) == 0: return []
        cuts = pd.cut(s, bins=bins, include_lowest=True)
        vc = cuts.value_counts().sort_index()
        return [(float(iv.left), float(iv.right), int(c)) for iv, c in vc.items()]

    top_munis = (df.groupby(["cve_mun", "nom_mun"])
                   .agg(n=("id_avaluo", "count"),
                        med_valor=("valor_concluido", "median"),
                        med_m2=("m2_sv", "median"))
                   .reset_index()
                   .sort_values("n", ascending=False)
                   .head(16))

    # ---- KPI por año + histogramas globales (para year-picker + modal KPI) ----
    metric_cols = {
        "valor":   "valor_concluido",
        "m2_sv":   "m2_sv",
        "terreno": "valor_terreno_m2",
    }
    global_hists = {}
    for mkey, col in metric_cols.items():
        if col in df.columns:
            h = _series_hist(df[col])
            if h is not None:
                global_hists[mkey] = h

    by_year_stats: dict = {}
    by_year_hists: dict = {}
    for y, sub in df.groupby("ano"):
        try:
            yk = str(int(y))
        except (TypeError, ValueError):
            continue
        ys = {}
        yh = {}
        for mkey, col in metric_cols.items():
            if col not in sub.columns:
                continue
            s = pd.to_numeric(sub[col], errors="coerce").dropna()
            if len(s) < 5:
                continue
            stat = {"median": float(s.median())}
            if mkey == "valor":
                stat["p25"] = float(s.quantile(0.25))
                stat["p75"] = float(s.quantile(0.75))
            ys[mkey] = stat
            h = _series_hist(sub[col])
            if h is not None:
                yh[mkey] = h
        if ys:
            by_year_stats[yk] = ys
            by_year_hists[yk] = yh

    return DescriptiveStats(
        n_total=len(df),
        n_municipios=df["cve_mun"].nunique(),
        year_min=int(df["ano"].min()),
        year_max=int(df["ano"].max()),
        median_valor=float(df["valor_concluido"].median()),
        median_m2_sv=float(df["m2_sv"].median()),
        median_terreno_m2=float(df["valor_terreno_m2"].median()),
        p25_valor=float(df["valor_concluido"].quantile(0.25)),
        p75_valor=float(df["valor_concluido"].quantile(0.75)),
        by_clase=by_clase,
        by_recamaras=by_rec,
        by_banos=by_ban,
        sup_construida_dist=histogram(df["sup_construida"], 22),
        sup_terreno_dist=histogram(df["sup_terreno"], 22),
        edad_meses_dist=histogram(df["edad_meses"], 22),
        top_munis=top_munis,
        by_year_stats=by_year_stats,
        by_year_hists=by_year_hists,
        global_hists=global_hists,
    )


# ---------------------------------------------------------------------------
# Clustering K-Means
# ---------------------------------------------------------------------------

@dataclass
class ClusterResult:
    k: int
    profile: pd.DataFrame              # un cluster por fila: n, medianas por feature
    crosstab: pd.DataFrame             # cluster × clase
    labels: pd.Series                  # cluster por id_avaluo (alineado con df)
    feature_means: pd.DataFrame        # means para narrar perfil
    cluster_names: dict[int, str]      # nombre human-readable por cluster


def cluster_kmeans(df: pd.DataFrame,
                    feature_cols: list[str]) -> ClusterResult | None:
    sub = df[feature_cols].copy()
    sub = sub.dropna()
    if len(sub) < 500:
        return None

    # k = número de clases observadas con n>=200, capeado [3, 6]
    clase_counts = df["clase"].value_counts()
    n_classes_relevant = int((clase_counts >= 200).sum())
    k = max(3, min(6, n_classes_relevant if n_classes_relevant > 0 else 4))

    scaler = StandardScaler()
    X = scaler.fit_transform(sub)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    sub = sub.assign(_cluster=labels)
    aligned = df.loc[sub.index].assign(_cluster=labels)

    # NSE label top-2 por cluster (si nse_label existe en df). Lista; el namer
    # imprime "NSE C+" o "NSE C+ y AB" según cuántos haya.
    nse_dominant: dict[int, list[str]] = {}
    if "nse_label" in aligned.columns:
        for cid, sub_c in aligned.groupby("_cluster"):
            modes = sub_c["nse_label"].dropna().value_counts()
            nse_dominant[int(cid)] = list(modes.head(2).index) if len(modes) else []

    # Municipio dominante por cluster
    muni_dominant = {}
    if "nom_mun" in aligned.columns:
        for cid, sub_c in aligned.groupby("_cluster"):
            modes = sub_c["nom_mun"].dropna().value_counts()
            muni_dominant[int(cid)] = modes.idxmax() if len(modes) else ""

    # Perfil mediano por cluster
    profile = (aligned.groupby("_cluster")
                      .agg(n=("id_avaluo", "count"),
                           med_sup_const=("sup_construida", "median"),
                           med_sup_terr=("sup_terreno", "median"),
                           med_recamaras=("recamaras", "median"),
                           med_banos=("banos", "median"),
                           med_estac=("estacionamiento", "median"),
                           med_edad_meses=("edad_meses", "median"),
                           med_niveles=("niveles", "median"),
                           med_m2_sv=("m2_sv", "median"),
                           med_valor=("valor_concluido", "median"))
                      .reset_index()
                      .sort_values("med_valor", ascending=True)
                      .reset_index(drop=True))

    # Reasignar IDs de cluster para que 0=más barato → k-1=más caro (UX: el orden importa)
    remap = {old: new for new, old in enumerate(profile["_cluster"].tolist())}
    aligned["_cluster"] = aligned["_cluster"].map(remap)
    profile["_cluster"] = profile["_cluster"].map(remap)
    profile = profile.sort_values("_cluster").reset_index(drop=True)
    nse_dominant = {remap[k_]: v for k_, v in nse_dominant.items()}
    muni_dominant = {remap[k_]: v for k_, v in muni_dominant.items()}

    # Tabla cruzada cluster × clase (computada pero ya no se renderiza)
    crosstab = pd.crosstab(aligned["_cluster"], aligned["clase"])

    # Percentiles de med_m2_sv entre clusters → etiquetas Segmento Alto/Medio/Bajo
    m2_vals = profile["med_m2_sv"].dropna()
    if len(m2_vals) >= 3:
        m2_cutoffs = (float(m2_vals.quantile(0.33)), float(m2_vals.quantile(0.66)))
    else:
        m2_cutoffs = None

    # Nombres descriptivos (heurística sobre medianas + NSE + municipio dominante)
    names = name_clusters(profile, nse_dominant, muni_dominant, m2_cutoffs)

    feature_means = aligned.groupby("_cluster")[feature_cols].mean()

    # Devolvemos labels alineados al df original (con NaN donde no aplicó)
    labels_full = pd.Series(index=df.index, dtype="Int64")
    labels_full.loc[aligned.index] = aligned["_cluster"].values

    return ClusterResult(
        k=k, profile=profile, crosstab=crosstab,
        labels=labels_full, feature_means=feature_means,
        cluster_names=names,
    )


def name_clusters(profile: pd.DataFrame,
                   nse_dominant: dict[int, list[str]] | None = None,
                   muni_dominant: dict[int, str] | None = None,
                   m2_cutoffs: tuple[float, float] | None = None) -> dict[int, str]:
    """Heurística humana para nombrar cada cluster: sólo los rasgos físicos
    (tamaño, tipología, antigüedad, segmento de precio). NSE y alcaldía se
    omiten — el NSE casi siempre cae en C / C- / C+ y la alcaldía dominante no
    agrega mucha separación entre clusters. El tier (Alto/Medio/Bajo) sale de
    los percentiles p33/p66 de med_m2_sv entre clusters."""
    nse_dominant = nse_dominant or {}
    muni_dominant = muni_dominant or {}
    p33, p66 = m2_cutoffs if m2_cutoffs else (None, None)
    names = {}
    for _, row in profile.iterrows():
        c = int(row["_cluster"])
        sc = row["med_sup_const"]
        st = row["med_sup_terr"]
        edad = row["med_edad_meses"]
        m2 = row["med_m2_sv"]
        niveles = row["med_niveles"]

        # categoría de tamaño
        if sc < 50: size = "Compacto"
        elif sc < 80: size = "Estándar"
        elif sc < 130: size = "Amplio"
        else: size = "Extenso"

        # tipología por relación terreno/construido (sin la etiqueta "Depto")
        if pd.notna(st) and st <= sc * 1.05:
            tipo = "Vertical"
        elif niveles and niveles >= 2 and pd.notna(st) and st < sc * 1.5:
            tipo = "Casa vertical"
        else:
            tipo = "Casa con terreno"

        # edad
        if edad < 24: age = "obra nueva"
        elif edad < 120: age = "reciente"
        elif edad < 240: age = "consolidado"
        else: age = "antiguo"

        # segmento por percentiles entre clusters (p33/p66 sobre med_m2_sv)
        if p33 is None or p66 is None or pd.isna(m2):
            tier = "Segmento Medio"
        elif m2 >= p66:
            tier = "Segmento Alto"
        elif m2 >= p33:
            tier = "Segmento Medio"
        else:
            tier = "Segmento Bajo"

        names[c] = f"{size} · {tipo} · {age} · {tier}"
    return names


# ---------------------------------------------------------------------------
# Logo, mapa interactivo (Leaflet) y preparación de datos
# ---------------------------------------------------------------------------

# Paleta compartida con la sección de plusvalía. El cluster id es el índice.
CLUSTER_PALETTE = ["#c9421f", "#1f6c5c", "#b88a1f", "#2a3d6e",
                   "#7b3b5e", "#4d6b1d", "#a44b1a"]


def _embed_png(path: str) -> str:
    """Lee un PNG del disco y devuelve un data: URI base64 listo para meter en
    un atributo src. Si el archivo no existe devuelve un data URI 1×1 transparente,
    así el HTML sigue válido aunque alguien lo regenere sin los assets."""
    p = Path(path)
    if not p.is_file():
        # 1×1 transparente
        return ("data:image/png;base64,"
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
    with p.open("rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


# Directorio del script — los logos viven junto a fact_avaluos_cdmx.py.
_ASSETS_DIR = Path(__file__).parent
LOGO_BLACKPRINT_DARK  = _embed_png(str(_ASSETS_DIR / "LogoSimple_Dark.png"))
LOGO_BLACKPRINT_LIGHT = _embed_png(str(_ASSETS_DIR / "LogoSimple_Light.png"))
LOGO_ORANGE_MARK      = _embed_png(str(_ASSETS_DIR / "Logo_Orange.png"))


def render_logo_svg() -> str:
    """Logo BlackPrint editorial. Mantenemos la versión "diamante en grilla"
    como fallback decorativo cuando no estamos en el hero (el wordmark vive
    inline en el header)."""
    return (
        '<svg class="bp-logo" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg" '
        'aria-label="Blackprint">'
        '<rect x="3" y="3" width="54" height="54" fill="none" '
        'stroke="currentColor" stroke-width="1.6"/>'
        '<line x1="3" y1="21" x2="57" y2="21" stroke="currentColor" '
        'stroke-width="0.5" stroke-dasharray="2 3" opacity="0.55"/>'
        '<line x1="3" y1="39" x2="57" y2="39" stroke="currentColor" '
        'stroke-width="0.5" stroke-dasharray="2 3" opacity="0.55"/>'
        '<line x1="21" y1="3" x2="21" y2="57" stroke="currentColor" '
        'stroke-width="0.5" stroke-dasharray="2 3" opacity="0.55"/>'
        '<line x1="39" y1="3" x2="39" y2="57" stroke="currentColor" '
        'stroke-width="0.5" stroke-dasharray="2 3" opacity="0.55"/>'
        '<rect x="23" y="23" width="14" height="14" fill="#c9421f" '
        'transform="rotate(45 30 30)"/>'
        '<circle cx="30" cy="30" r="2.2" fill="#f7f2e7"/>'
        '</svg>'
    )


def prepare_map_points(df: pd.DataFrame, cluster: "ClusterResult | None",
                        max_points: int = 250_000) -> dict:
    """Filtra puntos con lat/lon válidos dentro de CDMX, los etiqueta con
    el cluster K-Means y los muestrea de forma estratificada si superan
    `max_points`. Cada punto es un array compacto con todos los atributos
    necesarios para los filtros del explorador."""
    work = df.copy()
    if cluster is not None:
        work["_cluster"] = cluster.labels
    else:
        work["_cluster"] = pd.NA

    work = work.dropna(subset=["latitud", "longitud"])
    work = work[
        work["latitud"].between(19.00, 19.70) &
        work["longitud"].between(-99.50, -98.85)
    ]
    if cluster is not None:
        work = work.dropna(subset=["_cluster"])

    n_eligible = int(len(work))

    if len(work) > max_points and cluster is not None:
        target = max_points
        work = (work.groupby("_cluster", group_keys=False)
                    .apply(lambda g: g.sample(
                        n=max(40, int(round(target * len(g) / max(len(work), 1)))),
                        random_state=42, replace=False) if len(g) > 40 else g))
        if len(work) > max_points:
            work = work.sample(n=max_points, random_state=42)
    elif len(work) > max_points:
        work = work.sample(n=max_points, random_state=42)

    munis = list(pd.Series(work["nom_mun"].dropna().unique()).astype(str))
    muni_idx = {name: i for i, name in enumerate(munis)}

    def _i(v):
        return int(v) if pd.notna(v) else None
    def _ri(v):
        return int(round(float(v))) if pd.notna(v) else None

    points = []
    for _, r in work.iterrows():
        try:
            lat = round(float(r["latitud"]), 5)
            lon = round(float(r["longitud"]), 5)
        except (TypeError, ValueError):
            continue
        points.append([
            lat, lon,
            _i(r["_cluster"]),                              # 2: cluster
            _i(r["ano"]),                                   # 3: año
            _ri(r["m2_sv"]),                                # 4: $/m² SV
            _ri(r["valor_concluido"]),                      # 5: valor concluido
            _i(r["clase"]),                                 # 6: clase valuador
            muni_idx.get(str(r["nom_mun"])) if pd.notna(r["nom_mun"]) else None,  # 7: muni idx
            _i(r["trimestre"]),                             # 8: trimestre 1..4
            _i(r["recamaras"]),                             # 9: recamaras
            _i(r["banos"]),                                 # 10: baños
            _ri(r["sup_construida"]),                       # 11: m² construida
            _ri(r["sup_terreno"]),                          # 12: m² terreno
            _ri(r["valor_terreno_m2"]),                     # 13: $/m² terreno
            _i(r["edad_meses"]),                            # 14: edad meses
            _ri(r.get("valor_fisico_construccion")),        # 15: costo construcción
        ])

    return {"points": points, "munis": munis,
            "n_total_geo": n_eligible, "n_sampled": int(len(work))}


# ---------------------------------------------------------------------------
# Pre-cómputo de histogramas para los modales (cluster cards y top alcaldías)
# ---------------------------------------------------------------------------

# Atributos numéricos sobre los que precomputamos histogramas y stats.
HIST_ATTRS = [
    ("valor_concluido",   "Valor de la vivienda",    "$"),
    ("m2_sv",             "$ por m² construido",     "$"),
    ("sup_construida",    "Superficie construida",   "m²"),
    ("sup_terreno",       "Superficie terreno",      "m²"),
    ("valor_terreno_m2",  "$ por m² terreno",        "$"),
    ("edad_meses",        "Antigüedad (meses)",      "m"),
    ("recamaras",         "Recámaras",               ""),
    ("banos",             "Baños",                   ""),
    ("valor_fisico_construccion", "Costo construcción físico", "$"),
]


def _series_hist(s: pd.Series, bins: int = 20) -> dict | None:
    """Histograma compacto: bordes + conteos + mediana + promedio + n."""
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) < 5:
        return None
    lo, hi = float(s.min()), float(s.max())
    if hi <= lo:
        hi = lo + 1
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(s.values, bins=edges)
    return {
        "edges": [float(round(x, 2)) for x in edges.tolist()],
        "counts": counts.astype(int).tolist(),
        "median": float(s.median()),
        "mean":   float(s.mean()),
        "n":      int(len(s)),
    }


def _stats_block(s: pd.Series) -> dict:
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {"n": 0}
    return {
        "n": int(len(s)),
        "median": float(s.median()),
        "mean":   float(s.mean()),
        "p25":    float(s.quantile(0.25)),
        "p75":    float(s.quantile(0.75)),
        "min":    float(s.min()),
        "max":    float(s.max()),
    }


def compute_cluster_data(df: pd.DataFrame,
                          cluster: "ClusterResult | None") -> dict:
    """Por cluster, precomputa histogramas por año para cada atributo en
    HIST_ATTRS, y bloques de stats. Para que los modales en JS sean instantáneos."""
    if cluster is None:
        return {}
    work = df.copy()
    work["_cluster"] = cluster.labels
    work = work.dropna(subset=["_cluster"])
    work["_cluster"] = work["_cluster"].astype(int)

    out = {}
    years_all = sorted([int(y) for y in work["ano"].dropna().unique()])
    for cid, sub in work.groupby("_cluster"):
        attrs_block = {}
        for attr, label, unit in HIST_ATTRS:
            if attr not in sub.columns:
                continue
            hist_all = _series_hist(sub[attr])
            by_year = {}
            for y in years_all:
                ssub = sub[sub["ano"] == y][attr]
                h = _series_hist(ssub)
                if h is not None:
                    by_year[str(y)] = h
            if hist_all is None and not by_year:
                continue
            attrs_block[attr] = {
                "label": label, "unit": unit,
                "all": hist_all,
                "by_year": by_year,
            }
        # NSE summary del cluster
        nse_summary = {}
        if "nse_label" in sub.columns:
            nse_summary = (sub["nse_label"].dropna().value_counts(normalize=True)
                              .round(4).to_dict())
        # Distribución por alcaldía
        muni_dist = (sub["nom_mun"].dropna().value_counts().head(8).to_dict())

        out[int(cid)] = {
            "name": cluster.cluster_names.get(int(cid), f"Cluster {cid}"),
            "n_total": int(len(sub)),
            "stats": {
                "valor":     _stats_block(sub["valor_concluido"]),
                "m2_sv":     _stats_block(sub["m2_sv"]),
                "sup_const": _stats_block(sub["sup_construida"]),
                "edad":      _stats_block(sub["edad_meses"]),
            },
            "histograms": attrs_block,
            "nse_dist": {k: float(v) for k, v in nse_summary.items()},
            "muni_top": {k: int(v) for k, v in muni_dist.items()},
            "years": years_all,
        }
    return out


def compute_alcaldia_data(df: pd.DataFrame,
                           muni_geoms: pd.DataFrame,
                           cluster: "ClusterResult | None") -> dict:
    """Para cada alcaldía: geometría GeoJSON + histogramas + stats agregados.
    El histograma viene por atributo (HIST_ATTRS) y per-año."""
    work = df.copy()
    if cluster is not None:
        work["_cluster"] = cluster.labels

    # cve_mun normalizado en ambos lados
    work["_cve_mun"] = work["cve_mun"].astype(str).str.zfill(3)
    muni_geoms = muni_geoms.copy()
    muni_geoms["_cve_mun"] = muni_geoms["cve_mun"].astype(str).str.zfill(3)
    # nombre humano: tomamos el más frecuente de nom_mun por cve_mun
    name_by_muni = (work.dropna(subset=["nom_mun"])
                        .groupby("_cve_mun")["nom_mun"]
                        .agg(lambda s: s.mode().iloc[0] if len(s) else "")
                        .to_dict())

    features = []
    out_meta = {}
    for _, mg in muni_geoms.iterrows():
        cve = mg["_cve_mun"]
        geom = coords_json_to_geojson(mg["geom_raw"])
        if geom is None:
            continue
        sub = work[work["_cve_mun"] == cve]
        nom = name_by_muni.get(cve, "")
        if not nom and len(sub):
            nom = str(sub["nom_mun"].mode().iloc[0]) if len(sub["nom_mun"].mode()) else cve
        # histogramas por atributo
        attrs_block = {}
        years_all = sorted([int(y) for y in sub["ano"].dropna().unique()]) if len(sub) else []
        for attr, label, unit in HIST_ATTRS:
            if attr not in sub.columns:
                continue
            h_all = _series_hist(sub[attr])
            by_year = {}
            for y in years_all:
                hh = _series_hist(sub[sub["ano"] == y][attr])
                if hh is not None: by_year[str(y)] = hh
            if h_all is None and not by_year:
                continue
            attrs_block[attr] = {
                "label": label, "unit": unit,
                "all": h_all, "by_year": by_year,
            }
        # cluster predominante
        dom_cluster = None
        if cluster is not None and "_cluster" in sub.columns:
            modes = sub["_cluster"].dropna().value_counts()
            if len(modes):
                dom_cluster = int(modes.idxmax())
        # NSE
        nse_dist = {}
        if "nse_label" in sub.columns:
            nse_dist = (sub["nse_label"].dropna().value_counts(normalize=True)
                            .round(4).to_dict())
        props = {
            "cve_mun": cve,
            "nom_mun": nom,
            "n":       int(len(sub)),
            "med_valor":  float(sub["valor_concluido"].median()) if len(sub) else None,
            "med_m2_sv":  float(sub["m2_sv"].median())            if len(sub) else None,
            "med_terreno":float(sub["valor_terreno_m2"].median()) if len(sub) else None,
            "dom_cluster": dom_cluster,
            "nse_dist":   {k: float(v) for k, v in nse_dist.items()},
            "years":      years_all,
        }
        features.append({
            "type": "Feature",
            "geometry": geom,
            "properties": props,
        })
        out_meta[cve] = {
            "name": nom,
            "props": props,
            "histograms": attrs_block,
        }

    geojson = {"type": "FeatureCollection", "features": features}
    return {"geojson": geojson, "by_cve": out_meta}


# JS del explorador territorial. NO se .format()-ea: usamos .replace() para
# inyectar el JSON con los puntos y evitar tener que doblar todas las llaves.
MAP_JS_TEMPLATE = r"""
(function() {
  const DATA = __MAP_DATA__;
  if (typeof L === 'undefined') {
    const c = document.getElementById('map-canvas');
    if (c) c.innerHTML = '<div class="map-err">Leaflet no pudo cargarse (sin conexión a unpkg). El resto del reporte funciona.</div>';
    return;
  }

  // Índices de campos en cada punto
  const F_LAT=0,F_LON=1,F_C=2,F_Y=3,F_M=4,F_V=5,F_CL=6,F_MI=7,F_Q=8,F_REC=9,F_BAN=10,F_SUP=11,F_TER=12,F_TM=13,F_EDAD=14,F_FIS=15;

  const points       = DATA.points || [];
  const names        = DATA.names  || {};
  const palette      = DATA.palette || [];
  const munis        = DATA.munis || [];
  const partialYear  = DATA.partialYear || 2025;
  const clusterData  = DATA.clusterData || {};
  const alcaldiaData = (DATA.alcaldiaData && DATA.alcaldiaData.by_cve) || {};
  const geojson      = (DATA.alcaldiaData && DATA.alcaldiaData.geojson) || null;
  const yearsList    = DATA.years || [];
  const claseList    = DATA.clases || [];

  const state = {
    year: 'all',
    quarters: new Set([1,2,3,4]),
    priceMin: null, priceMax: null,
    recamaras: new Set(), banos: new Set(),
    supMin: null, supMax: null,
    terMin: null, terMax: null,
    tmMin: null,  tmMax: null,
    edadMin: null, edadMax: null,
    fisMin: null, fisMax: null,
    tipoMode: 'none', tipoValue: null,
    centerLat: 19.4326, centerLon: -99.1332, buffer: 1000,
  };

  // ============ Helpers ============
  function haversine(lat1, lon1, lat2, lon2) {
    const R = 6371000;
    const dLat = (lat2-lat1)*Math.PI/180;
    const dLon = (lon2-lon1)*Math.PI/180;
    const a = Math.sin(dLat/2)**2 + Math.cos(lat1*Math.PI/180)*Math.cos(lat2*Math.PI/180)*Math.sin(dLon/2)**2;
    return 2*R*Math.asin(Math.sqrt(a));
  }
  function median(arr) {
    if (!arr.length) return null;
    const s = arr.slice().sort((a,b)=>a-b);
    const mid = Math.floor(s.length/2);
    return s.length%2 ? s[mid] : (s[mid-1]+s[mid])/2;
  }
  function fmtInt(x) {
    if (x===null||x===undefined||isNaN(x)) return '—';
    return Math.round(x).toLocaleString('en-US');
  }
  function fmtMxn(x) {
    if (x===null||x===undefined||isNaN(x)) return '—';
    return '$'+Math.round(x).toLocaleString('en-US');
  }
  function fmtMxnShort(x) {
    if (x===null||x===undefined||isNaN(x)) return '—';
    const ax = Math.abs(x);
    if (ax>=1e6) return '$'+(x/1e6).toFixed(2)+'M';
    if (ax>=1e3) return '$'+Math.round(x/1e3)+'k';
    return '$'+Math.round(x);
  }
  function esc(s) {
    if (s===null||s===undefined) return '';
    return String(s).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  }

  // ============ Map init + base layers ============
  const map = L.map('map-canvas', {
    center: [19.4326, -99.1332], zoom: 11, preferCanvas: true,
    zoomControl: true, scrollWheelZoom: true,
  });
  const lightLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
    attribution: '© OpenStreetMap · © CartoDB',
    subdomains: 'abcd', maxZoom: 19,
  });
  const satLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
    attribution: 'Imagery © Esri · World Imagery', maxZoom: 19,
  });
  lightLayer.addTo(map);

  // Control de Recentrar como botón dentro del mapa (topright)
  const RecenterCtrl = L.Control.extend({
    options: { position: 'topright' },
    onAdd: function() {
      const container = L.DomUtil.create('div', 'leaflet-bar leaflet-control recentrar-ctrl');
      container.innerHTML = '<a href="#" role="button" title="Recentrar mapa en el centro del buffer">&#x2316; Recentrar</a>';
      L.DomEvent.disableClickPropagation(container);
      container.querySelector('a').addEventListener('click', function(e) {
        e.preventDefault();
        const z = state.buffer<=500?15:state.buffer<=1500?14:state.buffer<=3000?13:12;
        map.setView([state.centerLat, state.centerLon], z);
      });
      return container;
    }
  });
  new RecenterCtrl().addTo(map);

  // ============ Capa de alcaldías ============
  let alcLayer = null;
  if (geojson && geojson.features && geojson.features.length) {
    alcLayer = L.geoJSON(geojson, {
      style: () => ({color:'#1c1a17', weight:1.2, opacity:0.7, fillColor:'#c9421f', fillOpacity:0, dashArray:'3 3'}),
      onEachFeature: (feat, layer) => {
        const p = feat.properties || {};
        layer.on('mouseover', () => {
          layer.setStyle({fillOpacity:0.18, weight:2.2, opacity:1});
          const nseTop = Object.keys(p.nse_dist||{})[0] || 'n/d';
          const tt = L.tooltip({permanent:false, sticky:true, direction:'top', offset:[0,-8], className:'alc-tip'})
            .setContent('<b>'+esc(p.nom_mun)+'</b><br/>'
                      + 'n: '+fmtInt(p.n)+' · vivienda '+fmtMxnShort(p.med_valor)+'<br/>'
                      + '$/m² '+fmtMxnShort(p.med_m2_sv)+' · NSE '+esc(nseTop)+'<br/>'
                      + '<i>click para detalle</i>');
          layer.bindTooltip(tt).openTooltip();
        });
        layer.on('mouseout', () => {
          layer.setStyle({fillOpacity:0, weight:1.2, opacity:0.7});
          layer.closeTooltip();
        });
        layer.on('click', () => openAlcaldiaModal(p.cve_mun));
      },
    }).addTo(map);
  }

  // ============ Capa de marcadores (canvas) ============
  const canvasRenderer = L.canvas({padding:0.5});
  let markerLayer = L.layerGroup().addTo(map);
  let markersOn = true;
  function renderMarkers(pts) {
    markerLayer.clearLayers();
    if (!markersOn) return;
    for (let i=0; i<pts.length; i++) {
      const p = pts[i];
      const c = p[F_C];
      if (c===null||c===undefined) continue;
      L.circleMarker([p[F_LAT], p[F_LON]], {
        radius: 2.6,
        color: palette[c], fillColor: palette[c],
        fillOpacity: 0.72, weight: 0,
        renderer: canvasRenderer,
      }).addTo(markerLayer);
    }
  }

  // ============ Capa de heatmap por precio ============
  let heatLayer = null, heatmapOn = false;
  function renderHeat(pts) {
    if (heatLayer) { map.removeLayer(heatLayer); heatLayer = null; }
    if (!heatmapOn) return;
    const valid = [];
    for (const p of pts) if (p[F_V] !== null && p[F_V] !== undefined) valid.push(p[F_V]);
    if (!valid.length) return;
    valid.sort((a,b)=>a-b);
    const p10 = valid[Math.floor(valid.length*0.10)] || 0;
    const p90 = valid[Math.floor(valid.length*0.90)] || 1;
    const span = Math.max(p90 - p10, 1);
    const data = [];
    for (const p of pts) {
      if (p[F_V] === null) continue;
      const w = Math.max(0.05, Math.min(1.0, (p[F_V] - p10) / span));
      data.push([p[F_LAT], p[F_LON], w]);
    }
    heatLayer = L.heatLayer(data, {
      radius: 22, blur: 18, minOpacity: 0.35, maxZoom: 17,
      gradient: { 0.0:'#2a3d6e', 0.3:'#1f6c5c', 0.55:'#b88a1f', 0.78:'#c9421f', 1.0:'#7b3b5e' },
    });
    heatLayer.addTo(map);
  }

  // ============ Buffer + marker central ============
  let bufferLayer = null, centerLayer = null;
  function renderBuffer() {
    if (bufferLayer) map.removeLayer(bufferLayer);
    if (centerLayer) map.removeLayer(centerLayer);
    bufferLayer = L.circle([state.centerLat, state.centerLon], {
      radius: state.buffer,
      color: '#c9421f', weight: 1.8,
      fillColor: '#c9421f', fillOpacity: 0.05,
      dashArray: '5 4',
    }).addTo(map);
    centerLayer = L.marker([state.centerLat, state.centerLon], {
      icon: L.divIcon({className:'ctr-icon', html:'<div class="ctr-cross"><i></i><i></i></div>', iconSize:[22,22], iconAnchor:[11,11]}),
      interactive: false,
    }).addTo(map);
  }

  // ============ Filtros ============
  function pointPasses(p) {
    const y = p[F_Y];
    if (state.year !== 'all' && y !== state.year) return false;
    if (p[F_Q] !== null && !state.quarters.has(p[F_Q])) return false;
    if (state.priceMin !== null && p[F_V] !== null && p[F_V] < state.priceMin) return false;
    if (state.priceMax !== null && p[F_V] !== null && p[F_V] > state.priceMax) return false;
    if (state.recamaras.size > 0 && p[F_REC] !== null && !state.recamaras.has(p[F_REC])) return false;
    if (state.banos.size > 0 && p[F_BAN] !== null && !state.banos.has(p[F_BAN])) return false;
    if (state.supMin !== null && p[F_SUP] !== null && p[F_SUP] < state.supMin) return false;
    if (state.supMax !== null && p[F_SUP] !== null && p[F_SUP] > state.supMax) return false;
    if (state.terMin !== null && p[F_TER] !== null && p[F_TER] < state.terMin) return false;
    if (state.terMax !== null && p[F_TER] !== null && p[F_TER] > state.terMax) return false;
    if (state.tmMin !== null && p[F_TM] !== null && p[F_TM] < state.tmMin) return false;
    if (state.tmMax !== null && p[F_TM] !== null && p[F_TM] > state.tmMax) return false;
    if (state.edadMin !== null && p[F_EDAD] !== null && p[F_EDAD] < state.edadMin) return false;
    if (state.edadMax !== null && p[F_EDAD] !== null && p[F_EDAD] > state.edadMax) return false;
    if (state.fisMin !== null && p[F_FIS] !== null && p[F_FIS] < state.fisMin) return false;
    if (state.fisMax !== null && p[F_FIS] !== null && p[F_FIS] > state.fisMax) return false;
    if (state.tipoMode === 'cluster' && state.tipoValue !== null && p[F_C] !== state.tipoValue) return false;
    if (state.tipoMode === 'clase'   && state.tipoValue !== null && p[F_CL] !== state.tipoValue) return false;
    return true;
  }
  function pointsAfterFilters() {
    const out = [];
    for (let i=0; i<points.length; i++) if (pointPasses(points[i])) out.push(points[i]);
    return out;
  }
  function pointsInBuffer(arr) {
    const out = [];
    for (let i=0; i<arr.length; i++) {
      const p = arr[i];
      if (haversine(state.centerLat, state.centerLon, p[F_LAT], p[F_LON]) <= state.buffer) out.push(p);
    }
    return out;
  }

  // ============ Panel resumen ============
  function updateSummary(filtered, inBuf) {
    document.getElementById('m-n-all').textContent = fmtInt(filtered.length);
    document.getElementById('m-n').textContent     = fmtInt(inBuf.length);

    // Rango observado de valor_fisico_construccion en el buffer (año filtrado).
    // El filtro de año ya viene aplicado en `filtered` (e inBuf es subset), por
    // lo tanto los puntos del buffer reflejan el año seleccionado.
    var fisVals = [];
    for (var k = 0; k < inBuf.length; k++) {
      var fv = inBuf[k][F_FIS];
      if (fv !== null && fv !== undefined) fisVals.push(fv);
    }
    var minEl = document.getElementById('m-valcon-min-obs');
    var maxEl = document.getElementById('m-valcon-max-obs');
    var nEl   = document.getElementById('m-valcon-n-obs');
    if (minEl && maxEl && nEl) {
      if (fisVals.length) {
        var mn = Math.min.apply(null, fisVals), mx = Math.max.apply(null, fisVals);
        minEl.textContent = fmtMxn(mn);
        maxEl.textContent = fmtMxn(mx);
        nEl.textContent   = fisVals.length.toLocaleString('en-US');
      } else {
        minEl.textContent = '—';
        maxEl.textContent = '—';
        nEl.textContent   = '0';
      }
    }

    if (!inBuf.length) {
      document.getElementById('m-med-v').textContent = '—';
      document.getElementById('m-med-m').textContent = '—';
      document.getElementById('m-dom-tag').textContent = '—';
      document.getElementById('m-dom-tag').style.background = '#999';
      document.getElementById('m-dom-name').textContent = 'sin avalúos en zona';
      document.getElementById('m-dom-share').textContent = '';
      var mp = document.getElementById('m-margen-pct'); if (mp) mp.textContent = '—';
      var md = document.getElementById('m-margen-detail'); if (md) md.textContent = '— vs —';
      return;
    }
    const vs=[], ms=[], cCount={};
    // landVals = sup_terreno (m²) * valor_terreno_m2 ($/m²) por vivienda. Sin
    // tener ambos, no podemos imputar el valor de terreno y la fila se omite
    // del cálculo de margen (no se rellena con ceros para no sesgar a alza).
    var landVals = [];
    for (const p of inBuf) {
      if (p[F_V] !== null) vs.push(p[F_V]);
      if (p[F_M] !== null) ms.push(p[F_M]);
      if (p[F_C] !== null) cCount[p[F_C]] = (cCount[p[F_C]]||0)+1;
      if (p[F_TER] !== null && p[F_TER] !== undefined &&
          p[F_TM]  !== null && p[F_TM]  !== undefined) {
        landVals.push(p[F_TER] * p[F_TM]);
      }
    }
    document.getElementById('m-med-v').textContent = fmtMxn(median(vs));
    document.getElementById('m-med-m').textContent = fmtMxn(median(ms));
    const sortedC = Object.entries(cCount).sort((a,b)=>b[1]-a[1]);
    if (sortedC.length) {
      const cid = parseInt(sortedC[0][0]);
      const dn  = sortedC[0][1];
      const tag = document.getElementById('m-dom-tag');
      tag.textContent = 'C'+cid;
      tag.style.background = palette[cid];
      document.getElementById('m-dom-name').textContent = names[cid] || '';
      document.getElementById('m-dom-share').textContent =
        ((dn/inBuf.length)*100).toFixed(0)+'% · '+dn.toLocaleString('en-US')+' / '+inBuf.length.toLocaleString('en-US');
    }

    // Margen del Desarrollador con la fórmula corregida:
    //   margen = (vivienda − terreno − construcción) / vivienda
    // donde
    //   vivienda     = mediana(valor_concluido) en buffer
    //   terreno      = mediana(sup_terreno · valor_terreno_m2) en buffer
    //   construcción = mediana(valor_fisico_construccion) en buffer
    // El valor_fisico_construccion ya excluye el terreno (es costo físico de
    // la obra), pero el valor_concluido incluye terreno + edificación + plus
    // de mercado — sin restar el terreno el margen quedaba inflado.
    var medVal  = median(vs);
    var medFis  = (fisVals.length  >= 30) ? median(fisVals)  : null;
    var medLand = (landVals.length >= 30) ? median(landVals) : null;
    var mpEl = document.getElementById('m-margen-pct');
    var mdEl = document.getElementById('m-margen-detail');
    if (mpEl && mdEl) {
      if (medVal && medFis !== null && medLand !== null) {
        var pct = (medVal - medLand - medFis) / medVal * 100;
        mpEl.textContent = (pct >= 0 ? '+' : '') + pct.toFixed(1) + '%';
        mpEl.className = 'val ' + (pct >= 0 ? 'pos' : 'neg');
        mdEl.textContent =
          fmtMxnShort(medVal)  + ' − ' +
          fmtMxnShort(medLand) + ' terreno − ' +
          fmtMxnShort(medFis)  + ' construcción';
      } else {
        mpEl.textContent = '—';
        mpEl.className = 'val';
        if (medFis === null && medLand === null)      mdEl.textContent = 'n insuficiente (<30 con costo físico y con terreno)';
        else if (medLand === null)                    mdEl.textContent = 'n insuficiente (<30 con valor de terreno)';
        else if (medFis === null)                     mdEl.textContent = 'n insuficiente (<30 con costo físico)';
        else                                          mdEl.textContent = '— vs —';
      }
    }
  }

  // ============ Plusvalía — TODOS los clusters ============
  function renderPlusvaliaAll(inBuf) {
    const svg = document.getElementById('m-plus-svg');
    const tbody = document.getElementById('m-plus-tbody');
    const legend = document.getElementById('m-plus-legend');
    if (!svg) return;
    const agg = {};
    for (const p of inBuf) {
      if (p[F_C]===null || p[F_Y]===null || p[F_M]===null) continue;
      if (!agg[p[F_C]]) agg[p[F_C]] = {};
      if (!agg[p[F_C]][p[F_Y]]) agg[p[F_C]][p[F_Y]] = [];
      agg[p[F_C]][p[F_Y]].push(p[F_M]);
    }
    const series = {};
    const yearsSet = new Set();
    for (const c in agg) {
      const arr = [];
      for (const y in agg[c]) {
        const ms = agg[c][y];
        if (ms.length < 3) continue;
        arr.push([parseInt(y), median(ms), ms.length]);
        yearsSet.add(parseInt(y));
      }
      if (arr.length) { arr.sort((a,b)=>a[0]-b[0]); series[c] = arr; }
    }
    const yArr = Array.from(yearsSet).sort((a,b)=>a-b);
    if (!yArr.length || !Object.keys(series).length) {
      svg.innerHTML = '<text x="50%" y="50%" text-anchor="middle" style="font-family:Fraunces,serif;font-style:italic;font-size:13px;fill:#6a635a;">datos insuficientes con los filtros activos · ajusta la zona o los filtros</text>';
      tbody.innerHTML = '';
      legend.innerHTML = '';
      return;
    }
    const idx = {};
    let vMin = Infinity, vMax = -Infinity;
    for (const c in series) {
      const arr = series[c];
      const baseYear = arr[0][0];
      const baseVal  = arr[0][1];
      const seq = arr.map(([y,m,n]) => {
        const pct = baseVal>0 ? (m/baseVal - 1)*100 : 0;
        if (pct < vMin) vMin = pct;
        if (pct > vMax) vMax = pct;
        return [y, pct, n, y===partialYear, baseYear, m];
      });
      idx[c] = seq;
    }
    vMin = Math.min(vMin, 0) - 5;
    vMax = Math.max(vMax, 0) + 5;
    const yMin = yArr[0], yMax = yArr[yArr.length-1];
    const W = 880, H = 240, PL = 60, PR = 24, PT = 14, PB = 34;
    const xOf = y => PL + (yMax===yMin?0.5:(y-yMin)/(yMax-yMin))*(W-PL-PR);
    const yOf = v => PT + (1-(v-vMin)/(vMax-vMin))*(H-PT-PB);
    let html = '';
    const step = (vMax-vMin) <= 80 ? 10 : 20;
    let g = Math.ceil(vMin/step)*step;
    while (g <= vMax) {
      html += '<line x1="'+PL+'" y1="'+yOf(g).toFixed(1)+'" x2="'+(W-PR)+'" y2="'+yOf(g).toFixed(1)+'" class="grid"/>';
      html += '<text x="'+(PL-6)+'" y="'+(yOf(g)+3).toFixed(1)+'" class="axlbl" text-anchor="end">'+(g>0?'+':'')+g+'%</text>';
      g += step;
    }
    html += '<line x1="'+PL+'" y1="'+yOf(0).toFixed(1)+'" x2="'+(W-PR)+'" y2="'+yOf(0).toFixed(1)+'" class="grid base"/>';
    for (const y of yArr) {
      const isP = y === partialYear;
      html += '<text x="'+xOf(y).toFixed(1)+'" y="'+(H-PB+18)+'" class="axlbl'+(isP?' partial':'')+'" text-anchor="middle">'+y+(isP?'*':'')+'</text>';
    }
    const sortedCids = Object.keys(idx).map(k=>parseInt(k)).sort((a,b)=>a-b);
    for (const cid of sortedCids) {
      const seq = idx[cid];
      const color = palette[cid];
      const pts = seq.map(s => xOf(s[0]).toFixed(1)+','+yOf(s[1]).toFixed(1)).join(' ');
      html += '<polyline points="'+pts+'" class="ts-line" style="stroke:'+color+'"/>';
      seq.forEach(s => {
        const r = s[3] ? 3.4 : 4.4;
        html += '<circle cx="'+xOf(s[0]).toFixed(1)+'" cy="'+yOf(s[1]).toFixed(1)+'" r="'+r+'" fill="'+color+'" stroke="#fbf7eb" stroke-width="1.6"'+(s[3]?' opacity="0.75"':'')+'><title>C'+cid+' · '+s[0]+(s[3]?' (parcial)':'')+' · n='+s[2]+' · idx '+(s[1]>0?'+':'')+s[1].toFixed(1)+'% vs '+s[4]+'</title></circle>';
      });
    }
    svg.innerHTML = html;
    // Legenda
    let lhtml = '';
    sortedCids.forEach(cid => {
      const last = idx[cid][idx[cid].length-1];
      lhtml += '<li data-cluster="'+cid+'"><span class="leg-swatch" style="background:'+palette[cid]+'"></span>'
            +  '<span class="leg-name"><b>C'+cid+'</b> · '+esc(names[cid]||'')+'</span>'
            +  '<span class="leg-val">'+(last[1]>0?'+':'')+last[1].toFixed(1)+'%</span></li>';
    });
    legend.innerHTML = lhtml;
    legend.querySelectorAll('li[data-cluster]').forEach(li => {
      li.addEventListener('click', () => openClusterModal(parseInt(li.dataset.cluster)));
    });
    // Tabla
    let thtml = '<tr><th>Cluster</th>';
    yArr.forEach(y => { const isP = y === partialYear; thtml += '<th class="num'+(isP?' partial':'')+'">'+y+(isP?'*':'')+'</th>'; });
    thtml += '</tr>';
    sortedCids.forEach(cid => {
      thtml += '<tr><th>C'+cid+'</th>';
      const byY = {}; idx[cid].forEach(s => { byY[s[0]] = s; });
      yArr.forEach(y => {
        const s = byY[y];
        if (!s) { thtml += '<td class="num dim">—</td>'; }
        else {
          let cls = 'num';
          if (s[3]) cls += ' partial';
          if (s[1]>0) cls += ' pos'; else if (s[1]<0) cls += ' neg';
          thtml += '<td class="'+cls+'">'+(s[1]>0?'+':'')+s[1].toFixed(1)+'%</td>';
        }
      });
      thtml += '</tr>';
    });
    tbody.innerHTML = thtml;
  }

  // ============ Modal ============
  function openModal(title, body) {
    document.getElementById('modal-title').innerHTML = title;
    document.getElementById('modal-body').innerHTML = body;
    document.getElementById('modal-overlay').classList.add('open');
  }
  function closeModal() {
    document.getElementById('modal-overlay').classList.remove('open');
  }
  const modalCloseBtn = document.getElementById('modal-close');
  if (modalCloseBtn) modalCloseBtn.addEventListener('click', closeModal);
  const modalOv = document.getElementById('modal-overlay');
  if (modalOv) modalOv.addEventListener('click', e => { if (e.target.id === 'modal-overlay') closeModal(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

  function renderHistogramSvg(h, w, hHeight, fmt) {
    if (!h || !h.counts || !h.counts.length) {
      return '<svg viewBox="0 0 '+w+' '+hHeight+'" class="hist-svg"><text x="50%" y="50%" text-anchor="middle" style="font-family:Fraunces,serif;font-style:italic;font-size:11.5px;fill:#6a635a;">datos insuficientes</text></svg>';
    }
    const PL = 38, PR = 14, PT = 14, PB = 24;
    const maxC = Math.max.apply(null, h.counts) || 1;
    const bw = (w-PL-PR) / h.counts.length;
    let svg = '';
    h.counts.forEach((c, i) => {
      const bh = (hHeight-PT-PB) * c/maxC;
      const x = PL + i*bw;
      const y = hHeight-PB - bh;
      svg += '<rect x="'+x.toFixed(2)+'" y="'+y.toFixed(2)+'" width="'+(bw-0.6).toFixed(2)+'" height="'+bh.toFixed(2)+'" class="hb"><title>['+fmt(h.edges[i])+', '+fmt(h.edges[i+1])+'] · n='+c+'</title></rect>';
    });
    const lo = h.edges[0], hi = h.edges[h.edges.length-1];
    const xOfVal = v => PL + (v-lo)/Math.max(hi-lo,1e-9)*(w-PL-PR);
    if (h.median !== undefined && h.median !== null) {
      const x = xOfVal(h.median);
      svg += '<line x1="'+x+'" y1="'+(PT-3)+'" x2="'+x+'" y2="'+(hHeight-PB)+'" class="med-line"/>';
      svg += '<text x="'+(x+4)+'" y="'+(PT+8)+'" class="med-lbl">med '+fmt(h.median)+'</text>';
    }
    if (h.mean !== undefined && h.mean !== null) {
      const x = xOfVal(h.mean);
      svg += '<line x1="'+x+'" y1="'+(PT-3)+'" x2="'+x+'" y2="'+(hHeight-PB)+'" class="mean-line"/>';
      svg += '<text x="'+(x+4)+'" y="'+(PT+22)+'" class="mean-lbl">prom '+fmt(h.mean)+'</text>';
    }
    svg += '<line x1="'+PL+'" y1="'+(hHeight-PB)+'" x2="'+(w-PR)+'" y2="'+(hHeight-PB)+'" class="hax"/>';
    svg += '<text x="'+PL+'" y="'+(hHeight-PB+14)+'" class="hlbl">'+fmt(lo)+'</text>';
    svg += '<text x="'+(w-PR)+'" y="'+(hHeight-PB+14)+'" class="hlbl" text-anchor="end">'+fmt(hi)+'</text>';
    return '<svg viewBox="0 0 '+w+' '+hHeight+'" class="hist-svg">'+svg+'</svg>';
  }
  function _fmtFor(unit) {
    if (unit === '$') return v => '$'+(Math.abs(v)>=1e6?(v/1e6).toFixed(2)+'M':Math.abs(v)>=1e3?Math.round(v/1e3)+'k':Math.round(v));
    if (unit === 'm²') return v => Math.round(v)+' m²';
    if (unit === 'm')  return v => (v/12).toFixed(0)+' años';
    return v => Math.round(v).toLocaleString('en-US');
  }
  // Overlay translúcido sobre bin edges compartidos (mismo ancho, diferente histograma).
  function buildOverlayHistSvg(series, w, hPx, fmt) {
    var valid = series.filter(function(s){ return s.hist && s.hist.counts && s.hist.counts.length; });
    if (!valid.length) {
      return '<svg viewBox="0 0 '+w+' '+hPx+'" class="hist-svg"><text x="50%" y="50%" text-anchor="middle" style="font-family:Fraunces,serif;font-style:italic;font-size:11.5px;fill:#6a635a;">datos insuficientes</text></svg>';
    }
    var PL = 38, PR = 14, PT = 14, PB = 24;
    var lo = Infinity, hi = -Infinity;
    valid.forEach(function(s){
      lo = Math.min(lo, s.hist.edges[0]);
      hi = Math.max(hi, s.hist.edges[s.hist.edges.length-1]);
    });
    if (hi <= lo) hi = lo + 1;
    var NBINS = 20;
    var step = (hi - lo) / NBINS;
    var edges = [];
    for (var i = 0; i <= NBINS; i++) edges.push(lo + step*i);
    function rebin(h) {
      var counts = new Array(NBINS).fill(0);
      for (var i = 0; i < h.counts.length; i++) {
        var c = h.counts[i];
        if (!c) continue;
        var mid = (h.edges[i] + h.edges[i+1]) / 2;
        var idx = Math.floor((mid - lo) / step);
        if (idx < 0) idx = 0;
        if (idx >= NBINS) idx = NBINS - 1;
        counts[idx] += c;
      }
      return counts;
    }
    var rebinned = valid.map(function(s){ return rebin(s.hist); });
    var maxC = 1;
    rebinned.forEach(function(arr){ arr.forEach(function(c){ if (c > maxC) maxC = c; }); });
    var bw = (w - PL - PR) / NBINS;
    var layersSvg = '';
    rebinned.forEach(function(counts, sidx) {
      var color = valid[sidx].color, label = valid[sidx].label;
      var bars = '';
      for (var i = 0; i < NBINS; i++) {
        var c = counts[i]; if (!c) continue;
        var bh = (hPx-PT-PB) * c / maxC;
        var x = PL + i*bw, y = hPx-PB - bh;
        bars += '<rect x="'+x.toFixed(2)+'" y="'+y.toFixed(2)+'" width="'+(bw-0.6).toFixed(2)+'" height="'+bh.toFixed(2)+'" fill="'+color+'" fill-opacity="0.45" class="cmp-layer"><title>'+esc(label)+' · ['+fmt(edges[i])+', '+fmt(edges[i+1])+'] · n='+c+'</title></rect>';
      }
      layersSvg += '<g data-label="'+esc(label)+'">' + bars + '</g>';
    });
    var axis = '<line x1="'+PL+'" y1="'+(hPx-PB)+'" x2="'+(w-PR)+'" y2="'+(hPx-PB)+'" class="hax"/>'
             + '<text x="'+PL+'" y="'+(hPx-PB+14)+'" class="hlbl">'+fmt(lo)+'</text>'
             + '<text x="'+(w-PR)+'" y="'+(hPx-PB+14)+'" class="hlbl" text-anchor="end">'+fmt(hi)+'</text>';
    return '<svg viewBox="0 0 '+w+' '+hPx+'" class="hist-svg">'+layersSvg+axis+'</svg>';
  }

  function buildHistogramsBlock(histObj, year, ctx) {
    if (!histObj) return '<p class="empty">Sin histogramas disponibles.</p>';
    ctx = ctx || {};
    const W = 380, H = 170;
    let html = '<div class="hist-grid" data-year="'+(year||'all')+'">';
    const order = ['valor_concluido','m2_sv','sup_construida','sup_terreno','valor_terreno_m2','edad_meses','recamaras','banos','valor_fisico_construccion'];
    for (const attr of order) {
      const block = histObj[attr];
      if (!block) continue;
      let h = (year && year !== 'all' && block.by_year && block.by_year[year]) ? block.by_year[year] : block.all;
      if (!h) continue;
      const fmt = _fmtFor(block.unit);
      var cmpBtn = '';
      if (ctx.sourceType === 'cluster') {
        cmpBtn = '<button type="button" class="hist-cmp-btn" data-attr="'+attr+'" data-cid="'+ctx.sourceId+'">+ Comparar</button>';
      }
      html += '<div class="hist-card" data-attr="'+attr+'">'
           +    '<div class="hist-head">'
           +      '<span class="hist-title">'+esc(block.label)+'</span>'
           +      '<span class="hist-meta">n='+h.n.toLocaleString('en-US')+' · med '+fmt(h.median)+' · prom '+fmt(h.mean)+'</span>'
           +      cmpBtn
           +    '</div>'
           +    '<div class="hist-body">'+renderHistogramSvg(h, W, H, fmt)+'</div>'
           +    '<div class="hist-cmp-panel" style="display:none"></div>'
           +  '</div>';
    }
    html += '</div>';
    return html;
  }

  // Click delegado para "+ Comparar" en cualquier hist-card de un modal de cluster.
  document.addEventListener('click', function(ev) {
    var btn = ev.target.closest && ev.target.closest('.hist-cmp-btn');
    if (!btn) return;
    var card = btn.closest('.hist-card');
    var panel = card.querySelector('.hist-cmp-panel');
    var attr = btn.dataset.attr;
    var thisCid = parseInt(btn.dataset.cid, 10);
    var yearSel = document.getElementById('modal-year');
    var year = yearSel ? yearSel.value : 'all';
    if (panel.style.display !== 'none') {
      panel.style.display = 'none';
      panel.innerHTML = '';
      btn.classList.remove('on');
      btn.textContent = '+ Comparar';
      card.querySelector('.hist-body').style.display = '';
      return;
    }
    btn.classList.add('on');
    btn.textContent = '✕ Salir';
    // Picker de clusters
    var picks = Object.keys(clusterData).filter(function(k){ return parseInt(k,10) !== thisCid; }).map(function(k){ return parseInt(k,10); });
    var checkboxes = picks.map(function(c) {
      return '<label class="cmp-pick"><input type="checkbox" class="cmp-pick-chk" data-cid="'+c+'"/>'
           + '<span class="cmp-pick-sw" style="background:'+palette[c]+'"></span>'
           + '<span class="cmp-pick-name">C'+c+' · '+(esc((clusterData[c]||{}).name||''))+'</span></label>';
    }).join('');
    panel.innerHTML = '<div class="cmp-picker"><span class="cmp-lbl">Comparar con:</span>'+checkboxes+'</div>'
                    + '<div class="cmp-alpha"><label>Opacidad <input type="range" class="cmp-alpha-slider" min="0.10" max="0.90" step="0.05" value="0.45"/> <span class="cmp-alpha-val">0.45</span></label></div>'
                    + '<div class="cmp-overlay-host"></div>';
    panel.style.display = '';
    // Activos: el cluster del modal + todos los chequeados
    function rebuild() {
      var selected = [thisCid];
      panel.querySelectorAll('.cmp-pick-chk:checked').forEach(function(chk){ selected.push(parseInt(chk.dataset.cid, 10)); });
      var fmt = _fmtFor((clusterData[thisCid] && clusterData[thisCid].histograms && clusterData[thisCid].histograms[attr] && clusterData[thisCid].histograms[attr].unit) || '');
      var series = selected.map(function(c) {
        var cd = clusterData[c] || {};
        var block = (cd.histograms || {})[attr] || {};
        var h = (year && year !== 'all' && block.by_year && block.by_year[year]) ? block.by_year[year] : block.all;
        return { label: 'C'+c+' · '+((cd.name||'').slice(0, 30)), color: palette[c], hist: h };
      });
      panel.querySelector('.cmp-overlay-host').innerHTML = buildOverlayHistSvg(series, 380, 170, fmt);
      // Aplica alpha actual
      var a = parseFloat(panel.querySelector('.cmp-alpha-slider').value);
      panel.querySelectorAll('.cmp-overlay-host .cmp-layer').forEach(function(el){ el.setAttribute('fill-opacity', a.toFixed(2)); });
      panel.querySelector('.cmp-alpha-val').textContent = a.toFixed(2);
      // Esconde el histograma original (mostramos el overlay que ya incluye al cluster activo)
      card.querySelector('.hist-body').style.display = 'none';
    }
    panel.querySelectorAll('.cmp-pick-chk').forEach(function(chk){ chk.addEventListener('change', rebuild); });
    panel.querySelector('.cmp-alpha-slider').addEventListener('input', rebuild);
    rebuild();
  });

  function openClusterModal(cid) {
    const cd = clusterData[cid];
    if (!cd) { openModal('Cluster C'+cid, '<p class="empty">Sin datos para este cluster.</p>'); return; }
    const yearOpts = ['<option value="all">Todos los años</option>'].concat(
      (cd.years||[]).map(y => '<option value="'+y+'">'+y+'</option>')
    ).join('');
    const munis = cd.muni_top || {};
    const muniHtml = Object.entries(munis).map(([n,c]) =>
      '<li><span class="cl-muni-name">'+esc(n)+'</span><span class="cl-muni-n">'+c.toLocaleString('en-US')+'</span></li>'
    ).join('');
    const nse = cd.nse_dist || {};
    const nseHtml = Object.entries(nse).map(([k,v]) =>
      '<li><span class="cl-nse-name">NSE '+esc(k)+'</span><span class="cl-nse-pct">'+(v*100).toFixed(1)+'%</span></li>'
    ).join('');
    const valor = (cd.stats && cd.stats.valor) || {};
    const m2sv  = (cd.stats && cd.stats.m2_sv) || {};
    const supc  = (cd.stats && cd.stats.sup_const) || {};
    const edad  = (cd.stats && cd.stats.edad) || {};
    const title = '<span class="modal-tag" style="background:'+palette[cid]+'">C'+cid+'</span> '+esc(cd.name);
    const body = '<div class="modal-toolbar">'
              +    '<label>Año <select id="modal-year">'+yearOpts+'</select></label>'
              +  '</div>'
              +  '<div class="modal-stats">'
              +    '<div class="ms"><span class="lbl">mediana valor concluido</span><span class="val">'+fmtMxnShort(valor.median)+'</span><span class="sub">prom '+fmtMxnShort(valor.mean)+'</span></div>'
              +    '<div class="ms"><span class="lbl">mediana $/m²</span><span class="val">'+fmtMxnShort(m2sv.median)+'</span><span class="sub">prom '+fmtMxnShort(m2sv.mean)+'</span></div>'
              +    '<div class="ms"><span class="lbl">mediana sup const</span><span class="val">'+(supc.median?Math.round(supc.median)+' m²':'—')+'</span></div>'
              +    '<div class="ms"><span class="lbl">mediana edad</span><span class="val">'+(edad.median?(edad.median/12).toFixed(0)+' a':'—')+'</span></div>'
              +  '</div>'
              +  '<div class="modal-aux">'
              +    '<div class="aux-block"><h4>Distribución NSE</h4><ul class="cl-nse-list">'+(nseHtml||'<li class="empty">n/d</li>')+'</ul></div>'
              +    '<div class="aux-block"><h4>Top municipios del cluster</h4><ul class="cl-muni-list">'+(muniHtml||'<li class="empty">n/d</li>')+'</ul></div>'
              +  '</div>'
              +  '<h3 class="modal-sec">Histogramas por atributo (mediana y promedio marcados) · <span class="hint">click en <b>+ Comparar</b> para superponer otros clusters</span></h3>'
              +  '<div id="modal-hists" data-source="cluster" data-cid="'+cid+'">'+buildHistogramsBlock(cd.histograms, 'all', {sourceType:'cluster', sourceId:cid})+'</div>';
    openModal(title, body);
    const yearSel = document.getElementById('modal-year');
    if (yearSel) yearSel.addEventListener('change', e => {
      document.getElementById('modal-hists').innerHTML = buildHistogramsBlock(cd.histograms, e.target.value, {sourceType:'cluster', sourceId:cid});
    });
  }

  function openAlcaldiaModal(cve) {
    const ad = alcaldiaData[cve];
    if (!ad) { openModal('Municipio '+cve, '<p class="empty">Sin datos para este municipio.</p>'); return; }
    const props = ad.props || {};
    const yearOpts = ['<option value="all">Todos los años</option>'].concat(
      (props.years||[]).map(y => '<option value="'+y+'">'+y+'</option>')
    ).join('');
    const nse = props.nse_dist || {};
    const nseHtml = Object.entries(nse).map(([k,v]) =>
      '<li><span class="cl-nse-name">NSE '+esc(k)+'</span><span class="cl-nse-pct">'+(v*100).toFixed(1)+'%</span></li>'
    ).join('');
    const dom = props.dom_cluster;
    const title = '<span class="modal-tag" style="background:#1c1a17">Alc</span> '+esc(ad.name);
    const body = '<div class="modal-toolbar">'
              +    '<label>Año <select id="modal-year">'+yearOpts+'</select></label>'
              +  '</div>'
              +  '<div class="modal-stats">'
              +    '<div class="ms"><span class="lbl">mediana valor concluido</span><span class="val">'+fmtMxnShort(props.med_valor)+'</span></div>'
              +    '<div class="ms"><span class="lbl">mediana $/m² constr</span><span class="val">'+fmtMxnShort(props.med_m2_sv)+'</span></div>'
              +    '<div class="ms"><span class="lbl">mediana $/m² terreno</span><span class="val">'+fmtMxnShort(props.med_terreno)+'</span></div>'
              +    '<div class="ms"><span class="lbl">cluster predominante</span><span class="val">'+((dom!==null&&dom!==undefined)?'<span class="modal-tag-sm" style="background:'+palette[dom]+'">C'+dom+'</span>':'—')+'</span></div>'
              +  '</div>'
              +  '<div class="modal-aux">'
              +    '<div class="aux-block full"><h4>Distribución NSE del municipio</h4><ul class="cl-nse-list">'+(nseHtml||'<li class="empty">n/d</li>')+'</ul></div>'
              +  '</div>'
              +  '<h3 class="modal-sec">Histogramas (mediana y promedio marcados — para QA visual)</h3>'
              +  '<div id="modal-hists">'+buildHistogramsBlock(ad.histograms, 'all', {sourceType:'muni', sourceId:cve})+'</div>';
    openModal(title, body);
    const yearSel = document.getElementById('modal-year');
    if (yearSel) yearSel.addEventListener('change', e => {
      document.getElementById('modal-hists').innerHTML = buildHistogramsBlock(ad.histograms, e.target.value);
    });
  }

  // Exponer para handlers fuera de este IIFE (cards de §03, filas de §01,
  // KPIs, comparativa de municipios, etc.)
  window.__openClusterModal  = openClusterModal;
  window.__openAlcaldiaModal = openAlcaldiaModal;
  window.__openModal         = openModal;
  window.__closeModal        = closeModal;
  window.__renderHistSvg     = renderHistogramSvg;
  window.__fmtMxnShort       = fmtMxnShort;
  window.__fmtFor            = _fmtFor;
  window.__esc               = esc;
  window.__palette           = palette;
  window.__clusterData       = clusterData;
  window.__alcaldiaData      = alcaldiaData;

  // ============ Master update ============
  function applyFilters() {
    renderBuffer();
    const filtered = pointsAfterFilters();
    renderMarkers(filtered);
    renderHeat(filtered);
    const inBuf = pointsInBuffer(filtered);
    updateSummary(filtered, inBuf);
    renderPlusvaliaAll(inBuf);
  }

  // ============ Wire up inputs ============
  let _dbApply = null;
  function debApply() { clearTimeout(_dbApply); _dbApply = setTimeout(applyFilters, 100); }
  function setupNumInput(id, key) {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', () => {
      const v = el.value.trim();
      state[key] = v === '' ? null : parseFloat(v);
      debApply();
    });
  }
  function setupChipGroup(containerId, key, parser) {
    const cont = document.getElementById(containerId);
    if (!cont) return;
    cont.querySelectorAll('button.chip').forEach(btn => {
      btn.addEventListener('click', () => {
        const v = parser(btn.dataset.val);
        if (state[key].has(v)) { state[key].delete(v); btn.classList.remove('on'); }
        else { state[key].add(v); btn.classList.add('on'); }
        debApply();
      });
    });
  }

  document.getElementById('map-lat').addEventListener('input', e => { state.centerLat = parseFloat(e.target.value)||state.centerLat; debApply(); });
  document.getElementById('map-lon').addEventListener('input', e => { state.centerLon = parseFloat(e.target.value)||state.centerLon; debApply(); });
  document.getElementById('map-buf').addEventListener('input', e => {
    state.buffer = parseInt(e.target.value)||1000;
    document.getElementById('map-buf-val').textContent = state.buffer.toLocaleString('en-US');
    debApply();
  });
  // (Recentrar vive como L.Control adentro del mapa — ver más abajo)
  map.on('click', e => {
    state.centerLat = e.latlng.lat;
    state.centerLon = e.latlng.lng;
    document.getElementById('map-lat').value = e.latlng.lat.toFixed(4);
    document.getElementById('map-lon').value = e.latlng.lng.toFixed(4);
    // Sincronizar inputs y mini-mapa del predictor (§06). El handler del
    // mini-mapa expone __syncPredMiniMarker; lo llamamos si ya está montado.
    const pl = document.getElementById('pred-lat');
    const pn = document.getElementById('pred-lon');
    if (pl) pl.value = e.latlng.lat.toFixed(4);
    if (pn) pn.value = e.latlng.lng.toFixed(4);
    if (typeof window.__syncPredMiniMarker === 'function') {
      window.__syncPredMiniMarker(e.latlng.lat, e.latlng.lng);
    }
    debApply();
  });

  document.getElementById('f-year').addEventListener('change', e => {
    state.year = e.target.value === 'all' ? 'all' : parseInt(e.target.value);
    debApply();
  });
  document.querySelectorAll('#f-quarters input').forEach(cb => {
    cb.addEventListener('change', () => {
      const q = parseInt(cb.value);
      if (cb.checked) state.quarters.add(q); else state.quarters.delete(q);
      debApply();
    });
  });
  setupNumInput('f-price-min', 'priceMin');
  setupNumInput('f-price-max', 'priceMax');
  setupNumInput('f-sup-min',   'supMin');
  setupNumInput('f-sup-max',   'supMax');
  setupNumInput('f-ter-min',   'terMin');
  setupNumInput('f-ter-max',   'terMax');
  setupNumInput('f-tm-min',    'tmMin');
  setupNumInput('f-tm-max',    'tmMax');
  setupNumInput('f-edad-min',  'edadMin');
  setupNumInput('f-edad-max',  'edadMax');
  // El filtro de "Valor de la Construcción" vive ahora en el panel derecho
  // como m-valcon-min/max (reemplaza al viejo f-fis-min/max del sidebar).
  setupNumInput('m-valcon-min', 'fisMin');
  setupNumInput('m-valcon-max', 'fisMax');
  setupChipGroup('f-recamaras', 'recamaras', s => parseInt(s));
  setupChipGroup('f-banos',     'banos',     s => parseInt(s));

  document.querySelectorAll('input[name="f-tipo-mode"]').forEach(rb => {
    rb.addEventListener('change', () => {
      state.tipoMode = rb.value;
      document.getElementById('f-cluster-wrap').style.display = (rb.value==='cluster')?'flex':'none';
      document.getElementById('f-clase-wrap').style.display   = (rb.value==='clase')  ?'flex':'none';
      if (rb.value === 'none') state.tipoValue = null;
      debApply();
    });
  });
  document.getElementById('f-cluster').addEventListener('change', e => {
    state.tipoValue = e.target.value === '' ? null : parseInt(e.target.value);
    debApply();
  });
  document.getElementById('f-clase').addEventListener('change', e => {
    state.tipoValue = e.target.value === '' ? null : parseInt(e.target.value);
    debApply();
  });

  document.getElementById('f-reset').addEventListener('click', () => {
    state.year = 'all'; state.quarters = new Set([1,2,3,4]);
    state.priceMin=state.priceMax=null;
    state.recamaras = new Set(); state.banos = new Set();
    state.supMin=state.supMax=null;
    state.terMin=state.terMax=null;
    state.tmMin=state.tmMax=null;
    state.edadMin=state.edadMax=null;
    state.fisMin=state.fisMax=null;
    state.tipoMode = 'none'; state.tipoValue = null;
    document.getElementById('f-year').value = 'all';
    document.querySelectorAll('#f-quarters input').forEach(cb => cb.checked = true);
    ['f-price-min','f-price-max','f-sup-min','f-sup-max','f-ter-min','f-ter-max','f-tm-min','f-tm-max','f-edad-min','f-edad-max','m-valcon-min','m-valcon-max']
      .forEach(id => { const el = document.getElementById(id); if (el) el.value = ''; });
    document.querySelectorAll('#f-recamaras .chip, #f-banos .chip').forEach(c => c.classList.remove('on'));
    document.querySelector('input[name="f-tipo-mode"][value="none"]').checked = true;
    document.getElementById('f-cluster-wrap').style.display = 'none';
    document.getElementById('f-clase-wrap').style.display = 'none';
    applyFilters();
  });

  // Toggles de capas
  document.getElementById('f-sat').addEventListener('change', e => {
    if (e.target.checked) { map.removeLayer(lightLayer); satLayer.addTo(map); }
    else                  { map.removeLayer(satLayer); lightLayer.addTo(map); }
  });
  document.getElementById('f-alc').addEventListener('change', e => {
    if (!alcLayer) return;
    if (e.target.checked) alcLayer.addTo(map); else map.removeLayer(alcLayer);
  });
  document.getElementById('f-heat').addEventListener('change', e => {
    heatmapOn = e.target.checked;
    applyFilters();
  });
  document.getElementById('f-markers').addEventListener('change', e => {
    markersOn = e.target.checked;
    applyFilters();
  });

  // Year selector dentro del panel: sincronizado con el filtro de la sidebar.
  (function setupPanelYear() {
    const panelSel = document.getElementById('m-panel-year');
    const sideSel  = document.getElementById('f-year');
    if (!panelSel || !sideSel) return;
    // Replicar opciones
    panelSel.innerHTML = sideSel.innerHTML;
    panelSel.value = sideSel.value;
    panelSel.addEventListener('change', e => {
      sideSel.value = e.target.value;
      state.year = e.target.value === 'all' ? 'all' : parseInt(e.target.value);
      debApply();
    });
    // Cuando se cambia desde la sidebar, reflejar en el panel
    sideSel.addEventListener('change', () => { panelSel.value = sideSel.value; });
  })();

  // Cluster predominante (top del panel) click → modal
  document.getElementById('m-dom-tag').addEventListener('click', () => {
    const tag = document.getElementById('m-dom-tag').textContent.trim();
    if (tag && tag.startsWith('C')) openClusterModal(parseInt(tag.slice(1)));
  });

  // Wire up rows clickables fuera de este IIFE
  document.querySelectorAll('tr.muni-row').forEach(tr => {
    tr.addEventListener('click', () => openAlcaldiaModal(tr.dataset.cve));
    tr.addEventListener('keydown', e => { if (e.key === 'Enter') openAlcaldiaModal(tr.dataset.cve); });
  });
  document.querySelectorAll('article.cluster-card.clickable').forEach(card => {
    card.addEventListener('click', () => openClusterModal(parseInt(card.dataset.cluster)));
    card.addEventListener('keydown', e => { if (e.key === 'Enter') openClusterModal(parseInt(card.dataset.cluster)); });
  });

  // Build static legend (cluster pills al pie del map-block)
  const totals = {};
  for (const p of points) if (p[F_C] !== null && p[F_C] !== undefined) totals[p[F_C]] = (totals[p[F_C]]||0)+1;
  const lgEl = document.getElementById('m-legend');
  Object.keys(names).map(k=>parseInt(k)).sort((a,b)=>a-b).forEach(cid => {
    const li = document.createElement('li');
    li.setAttribute('data-cluster', cid);
    li.tabIndex = 0;
    li.innerHTML = '<span class="leg-swatch" style="background:'+palette[cid]+'"></span>'
                +  '<span class="leg-name"><b>C'+cid+'</b> · '+esc(names[cid]||'')+'</span>'
                +  '<span class="leg-count">'+(totals[cid]||0).toLocaleString('en-US')+'</span>';
    li.addEventListener('click', () => openClusterModal(cid));
    li.addEventListener('keydown', e => { if (e.key === 'Enter') openClusterModal(cid); });
    lgEl.appendChild(li);
  });

  // ============ Predictor: Regresión lineal en log-precio ============
  const regression = DATA.regression && DATA.regression.features ? DATA.regression : null;

  function predictPriceLR(lat, lon, supC, rec, ban, edadAnos, cid) {
    if (!regression) return null;
    const edadM = (edadAnos || 0) * 12;
    const sup = Math.max(supC, 1);
    const featMap = {
      lat: lat, lon: lon,
      sup_const: sup,
      log_sup_const: Math.log(sup),
      recamaras: rec,
      banos: ban,
      edad_meses: edadM,
      sqrt_edad: Math.sqrt(Math.max(edadM, 0)),
    };
    for (let k = 1; k < regression.n_clusters; k++) {
      featMap['clu_' + k] = (cid === k) ? 1 : 0;
    }
    let logPred = regression.intercept;
    for (let i = 0; i < regression.features.length; i++) {
      const v = featMap[regression.features[i]];
      if (v !== undefined) logPred += regression.coefs[i] * v;
    }
    const sigma = regression.sigma;
    return {
      predicted: Math.exp(logPred),
      ciLow:     Math.exp(logPred - 1.96 * sigma),
      ciHigh:    Math.exp(logPred + 1.96 * sigma),
      sigmaLog:  sigma,
      r2:        regression.r2,
      nTrain:    regression.n_train,
    };
  }

  // Encuentra los Top-N comparables: mismo cluster, dentro de un radio (m),
  // ordenados por similitud (mayor = más parecido).
  function findComparables(lat, lon, supC, rec, ban, edadAnos, cid, radiusM, topN) {
    radiusM = radiusM || 3000;
    topN = topN || 10;
    const edadM = (edadAnos || 0) * 12;
    const matches = [];
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      if (p[F_C] !== cid) continue;
      if (p[F_V] === null || p[F_V] === undefined) continue;
      const dist = haversine(lat, lon, p[F_LAT], p[F_LON]);
      if (dist > radiusM) continue;
      // Distancia normalizada en el espacio de features (no cuenta lat/lon —
      // el filtro de 3 km ya hace el corte geográfico).
      const dSup  = (p[F_SUP]  !== null) ? (p[F_SUP]  - supC) / 50.0  : 1;
      const dRec  = (p[F_REC]  !== null) ? (p[F_REC]  - rec)          : 1;
      const dBan  = (p[F_BAN]  !== null) ? (p[F_BAN]  - ban)          : 1;
      const dEdad = (p[F_EDAD] !== null) ? (p[F_EDAD] - edadM) / 60.0 : 1;
      const fd = Math.sqrt(dSup*dSup + dRec*dRec + dBan*dBan + dEdad*dEdad);
      const sim = Math.exp(-fd) * 100;   // 0..100
      matches.push({ idx: i, p: p, dist_m: dist, sim: sim });
    }
    matches.sort((a, b) => b.sim - a.sim);
    return matches.slice(0, topN);
  }

  function buildLoadingLogo() {
    return '<svg class="loading-logo" viewBox="0 0 60 60" xmlns="http://www.w3.org/2000/svg">'
      + '<rect class="lframe" x="3" y="3" width="54" height="54" fill="none" stroke="currentColor" stroke-width="1.6"/>'
      + '<line x1="3" y1="21" x2="57" y2="21" stroke="currentColor" stroke-width="0.5" stroke-dasharray="2 3" opacity="0.55"/>'
      + '<line x1="3" y1="39" x2="57" y2="39" stroke="currentColor" stroke-width="0.5" stroke-dasharray="2 3" opacity="0.55"/>'
      + '<line x1="21" y1="3" x2="21" y2="57" stroke="currentColor" stroke-width="0.5" stroke-dasharray="2 3" opacity="0.55"/>'
      + '<line x1="39" y1="3" x2="39" y2="57" stroke="currentColor" stroke-width="0.5" stroke-dasharray="2 3" opacity="0.55"/>'
      + '<g class="lrotor" style="transform-origin:30px 30px"><rect x="22" y="22" width="14" height="14" fill="#c9421f"/><circle cx="29" cy="29" r="2.2" fill="#f7f2e7"/></g>'
      + '</svg>';
  }

  function renderComparablesTable(comps, inputs) {
    if (!comps.length) {
      return '<p class="empty">No hay comparables del cluster <b>C' + inputs.cid + '</b> dentro de 3 km de tu punto. Prueba con otro submercado o moviendo la zona.</p>';
    }
    let html = '<table class="pred-table comp-table">'
      + '<thead><tr>'
      +   '<th>#</th>'
      +   '<th>Similitud</th>'
      +   '<th class="num">Dist</th>'
      +   '<th>Lat / Lon · municipio</th>'
      +   '<th class="num">Sup m²</th>'
      +   '<th class="num">Rec</th>'
      +   '<th class="num">Baños</th>'
      +   '<th class="num">Edad</th>'
      +   '<th class="num">Valor</th>'
      +   '<th class="num">$/m²</th>'
      +   '<th>Cluster</th>'
      + '</tr></thead><tbody>';
    comps.forEach((c, i) => {
      const p = c.p;
      const ed = (p[F_EDAD] !== null) ? (p[F_EDAD]/12).toFixed(0)+' a' : '—';
      const mn = (p[F_MI] !== null) ? esc(munis[p[F_MI]] || '') : '';
      const cluCol = (p[F_C] !== null)
        ? '<span class="pred-tag" style="background:'+palette[p[F_C]]+'">C'+p[F_C]+'</span>'
        : '—';
      const simPct = Math.max(0, Math.min(100, c.sim));
      const simBar = '<div class="sim-cell">'
                   +   '<div class="sim-bar"><span style="width:'+simPct.toFixed(1)+'%"></span></div>'
                   +   '<span class="sim-val">'+simPct.toFixed(0)+'%</span>'
                   + '</div>';
      html += '<tr>'
        + '<td>'+(i+1)+'</td>'
        + '<td>'+simBar+'</td>'
        + '<td class="num">'+(c.dist_m/1000).toFixed(2)+' km</td>'
        + '<td>'+p[F_LAT].toFixed(3)+', '+p[F_LON].toFixed(3)+'<br/><small>'+mn+'</small></td>'
        + '<td class="num">'+(p[F_SUP]!==null?p[F_SUP]:'—')+'</td>'
        + '<td class="num">'+(p[F_REC]!==null?p[F_REC]:'—')+'</td>'
        + '<td class="num">'+(p[F_BAN]!==null?p[F_BAN]:'—')+'</td>'
        + '<td class="num">'+ed+'</td>'
        + '<td class="num">'+fmtMxnShort(p[F_V])+'</td>'
        + '<td class="num">'+(p[F_M]!==null?fmtMxnShort(p[F_M]):'—')+'</td>'
        + '<td>'+cluCol+'</td>'
        + '</tr>';
    });
    html += '</tbody></table>';
    return html;
  }

  function renderPredResult(r, comps, inputs) {
    const result = document.getElementById('pred-result');
    if (!r || isNaN(r.predicted)) {
      result.innerHTML = '<p class="empty">No se pudo calcular la predicción (regresión no disponible).</p>';
      return;
    }
    const cid = inputs.cid;
    const cidName = (cid !== null) ? (names[cid] || '') : '';
    const m2 = inputs.sup > 0 ? (r.predicted / inputs.sup) : null;
    const ciM2Low  = inputs.sup > 0 ? (r.ciLow  / inputs.sup) : null;
    const ciM2High = inputs.sup > 0 ? (r.ciHigh / inputs.sup) : null;
    const stats = (() => {
      // Estadísticas rápidas de los comparables (independiente de la regresión)
      if (!comps.length) return null;
      const vs = comps.map(c => c.p[F_V]).filter(v => v !== null);
      vs.sort((a,b)=>a-b);
      return {
        n: vs.length,
        med: vs[Math.floor(vs.length/2)],
        min: vs[0],
        max: vs[vs.length-1],
      };
    })();

    let html = '<div class="pred-output">';
    // Bloque principal
    html += '<div class="pred-main">'
         +   '<span class="pred-lbl">Precio estimado · regresión log-lineal</span>'
         +   '<span class="pred-big">'+fmtMxn(r.predicted)+'</span>'
         +   '<span class="pred-band">'
         +     'IC 95%: <b>'+fmtMxnShort(r.ciLow)+' — '+fmtMxnShort(r.ciHigh)+'</b>'
         +     (m2 ? ' · <b>'+fmtMxnShort(m2)+'/m²</b>' : '')
         +     (ciM2Low && ciM2High ? ' (IC '+fmtMxnShort(ciM2Low)+' — '+fmtMxnShort(ciM2High)+'/m²)' : '')
         +   '</span>'
         + '</div>';
    // Meta cards — métricas de diagnóstico técnico (R² y σ) se omitieron por
    // pedido: no aportan a la decisión del consultor.
    html += '<div class="pred-meta">'
         +   '<div><span class="lbl">Submercado seleccionado</span><span class="val">'+(cid!==null?'<span class="pred-tag" style="background:'+palette[cid]+'">C'+cid+'</span> '+esc(cidName):'—')+'</span></div>'
         +   '<div><span class="lbl">Comparables encontrados</span><span class="val">'+comps.length+'</span><span class="lbl-sub">cluster C'+cid+' · radio 3 km</span></div>'
         + '</div>';

    // ICM — Análisis Comparativo de Mercado
    html += '<div class="acm-block">'
         +   '<div class="acm-head">'
         +     '<h4 class="pred-sec">Análisis Comparativo de Mercado <span class="acm-sub">· Top 10 dentro de 3 km · cluster C'+cid+'</span></h4>'
         +     (stats ? '<span class="acm-stat">mediana comparables: <b>'+fmtMxnShort(stats.med)+'</b> · rango '+fmtMxnShort(stats.min)+'—'+fmtMxnShort(stats.max)+'</span>' : '')
         +   '</div>'
         +   renderComparablesTable(comps, inputs)
         + '</div>';

    html += '</div>';
    result.innerHTML = html;
  }

  // Setup del predictor: el HTML de §06 se inserta DESPUÉS de este <script>
  // en el body, así que en este punto del parser los inputs (pred-go, pred-lat,
  // etc.) todavía no existen. Esperamos a DOMContentLoaded — y, por seguridad,
  // a un microtask si la página ya terminó de parsear — para attachar listeners
  // y construir el mini-mapa.
  function setupPredictor() {
    const btn = document.getElementById('pred-go');
    if (!btn) return;
    btn.addEventListener('click', () => {
      const lat  = parseFloat(document.getElementById('pred-lat').value);
      const lon  = parseFloat(document.getElementById('pred-lon').value);
      const sup  = parseFloat(document.getElementById('pred-sup').value);
      const rec  = parseInt(document.getElementById('pred-rec').value);
      const ban  = parseInt(document.getElementById('pred-ban').value);
      const edad = parseFloat(document.getElementById('pred-edad').value);
      const cidStr = document.getElementById('pred-cluster').value;
      const cid = cidStr === '' ? 0 : parseInt(cidStr);
      const result = document.getElementById('pred-result');
      if ([lat, lon, sup].some(v => isNaN(v))) {
        result.innerHTML = '<p class="empty">Llena al menos latitud, longitud y superficie construida.</p>';
        return;
      }
      // Loading
      result.innerHTML =
        '<div class="pred-loading">'
        +   buildLoadingLogo()
        +   '<div class="pred-status-wrap">'
        +     '<span class="pred-status" id="pred-status">Cargando coeficientes…</span>'
        +     '<span class="pred-tick" id="pred-tick">↻ leyendo modelo entrenado…</span>'
        +   '</div>'
        + '</div>';
      let stage = 0;
      const stages = [
        ['Cargando modelo…',              '↻ leyendo coeficientes entrenados (n='+(regression?regression.n_train.toLocaleString("en-US"):'?')+')…'],
        ['Procesando atributos…',         '↻ ubicación, superficie, recámaras, baños, edad…'],
        ['Aplicando submercado…',         '↻ activando C'+cid+' del clustering…'],
        ['Calculando rango esperado…',    '↻ intervalo de confianza al 95%…'],
        ['Buscando comparables…',         '↻ mismo cluster + 3 km, ordenados por similitud…'],
      ];
      const ticker = setInterval(() => {
        stage = (stage + 1) % stages.length;
        const ps = document.getElementById('pred-status');
        const pt = document.getElementById('pred-tick');
        if (ps) ps.textContent = stages[stage][0];
        if (pt) pt.textContent = stages[stage][1];
      }, 480);
      setTimeout(() => {
        const r     = predictPriceLR(lat, lon, sup, isNaN(rec)?2:rec, isNaN(ban)?2:ban, isNaN(edad)?10:edad, cid);
        const comps = findComparables(lat, lon, sup, isNaN(rec)?2:rec, isNaN(ban)?2:ban, isNaN(edad)?10:edad, cid, 3000, 10);
        clearInterval(ticker);
        renderPredResult(r, comps, {lat, lon, sup, rec, ban, edad, cid});
      }, 1300);
    });

    // Mini-mapa propio de §06 para elegir zona sin tener que volver a §05.
    // Renderiza un sample visual reducido (~6k puntos) por performance — el
    // foco acá es picking de zona, no exploración densa.
    const miniHost = document.getElementById('pred-mini-map');
    let miniMap = null, miniMarker = null;
    function syncInputsFromLatLng(ll) {
      document.getElementById('pred-lat').value = ll.lat.toFixed(4);
      document.getElementById('pred-lon').value = ll.lng.toFixed(4);
    }
    if (miniHost && typeof L !== 'undefined') {
      miniMap = L.map(miniHost, {
        center: [19.4326, -99.1332], zoom: 11,
        zoomControl: true, scrollWheelZoom: true, preferCanvas: true,
        attributionControl: false,
      });
      L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png', {
        subdomains: 'abcd', maxZoom: 19,
      }).addTo(miniMap);
      // Sample visual: ~6k puntos coloreados por cluster, sin tooltips ni
      // eventos para mantenerlo liviano.
      const miniRenderer = L.canvas({padding: 0.5});
      const stride = Math.max(1, Math.ceil(points.length / 6000));
      for (let i = 0; i < points.length; i += stride) {
        const p = points[i];
        if (p[F_C] === null || p[F_C] === undefined) continue;
        L.circleMarker([p[F_LAT], p[F_LON]], {
          radius: 1.8, color: palette[p[F_C]], fillColor: palette[p[F_C]],
          fillOpacity: 0.55, weight: 0, renderer: miniRenderer,
        }).addTo(miniMap);
      }
      miniMarker = L.marker([19.4326, -99.1332], { draggable: true }).addTo(miniMap);
      miniMarker.on('drag', (e) => syncInputsFromLatLng(e.target.getLatLng()));
      miniMap.on('click', (e) => {
        miniMarker.setLatLng(e.latlng);
        syncInputsFromLatLng(e.latlng);
      });
      // Recalcula tiles después de que el contenedor esté layouted (el div
      // puede ser invisible al primer init si la sección entra en viewport
      // después, lo cual deja tiles grises).
      setTimeout(() => { miniMap.invalidateSize(); }, 60);

      // Expone función para que el handler del mapa §05 mueva este marker
      // cuando el usuario clickea allá.
      window.__syncPredMiniMarker = function(lat, lon) {
        miniMarker.setLatLng([lat, lon]);
        miniMap.panTo([lat, lon]);
      };
    }

    // Mantener el marker en sync con escrituras directas a los inputs.
    function syncMarkerFromInputs() {
      if (!miniMarker) return;
      const lat = parseFloat(document.getElementById('pred-lat').value);
      const lon = parseFloat(document.getElementById('pred-lon').value);
      if (isNaN(lat) || isNaN(lon)) return;
      miniMarker.setLatLng([lat, lon]);
      if (miniMap) miniMap.panTo([lat, lon]);
    }
    ['pred-lat','pred-lon'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.addEventListener('change', syncMarkerFromInputs);
    });

    // Botón: copiar el centro del mapa de §05.
    const useMapBtn = document.getElementById('pred-from-map');
    if (useMapBtn) useMapBtn.addEventListener('click', () => {
      const ll = L.latLng(state.centerLat, state.centerLon);
      syncInputsFromLatLng(ll);
      if (miniMarker) { miniMarker.setLatLng(ll); if (miniMap) miniMap.panTo(ll); }
    });

    // Botón: reset al centro de CDMX.
    const resetMapBtn = document.getElementById('pred-reset-map');
    if (resetMapBtn) resetMapBtn.addEventListener('click', () => {
      const ll = L.latLng(19.4326, -99.1332);
      syncInputsFromLatLng(ll);
      if (miniMarker) miniMarker.setLatLng(ll);
      if (miniMap) miniMap.setView(ll, 11);
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupPredictor);
  } else {
    setupPredictor();
  }

  // Exportar Vista PNG (html2canvas)
  var mExport = document.getElementById('m-export-png');
  if (mExport) mExport.addEventListener('click', function() {
    if (typeof html2canvas === 'undefined') {
      alert('html2canvas no se cargó (sin conexión a CDN). No se puede exportar.');
      return;
    }
    var node = document.querySelector('.map-block .map-layout');
    html2canvas(node, { useCORS: true, backgroundColor: '#f7f2e7' }).then(function(canvas) {
      var a = document.createElement('a');
      a.download = 'explorador-territorial.png';
      a.href = canvas.toDataURL('image/png');
      a.click();
    });
  });

  // Initial render
  applyFilters();
})();
"""


# ---------------------------------------------------------------------------
# Comparativa de municipios: extrae histogramas compactos del compute_alcaldia
# ---------------------------------------------------------------------------

# Atributos visibles en el modal de comparación (subset curado de HIST_ATTRS).
MUNI_CMP_ATTRS = [
    ("valor_concluido",   "Valor concluido",         "$"),
    ("m2_sv",             "$ por m² construido",     "$"),
    ("valor_terreno_m2",  "$ por m² terreno",        "$"),
    ("sup_construida",    "Superficie construida",   "m²"),
    ("sup_terreno",       "Superficie terreno",      "m²"),
    ("edad_meses",        "Antigüedad (meses)",      "m"),
]


def build_muni_compare_data(alcaldia_data: dict) -> dict:
    """Reorganiza los histogramas (all years) por cve_mun en formato compacto
    para el modal de comparación. Cada muni: {name, n, hists:{attr:hist}}."""
    by_cve_in = (alcaldia_data or {}).get("by_cve", {}) or {}
    by_cve_out: dict = {}
    for cve, meta in by_cve_in.items():
        hists_in = (meta or {}).get("histograms", {}) or {}
        hists_out = {}
        for attr, _, _ in MUNI_CMP_ATTRS:
            block = hists_in.get(attr)
            if not block:
                continue
            h_all = block.get("all")
            if h_all:
                hists_out[attr] = h_all
        by_cve_out[cve] = {
            "name": (meta.get("name") or cve),
            "n":    (meta.get("props", {}) or {}).get("n", 0),
            "hists": hists_out,
        }
    return {
        "by_cve": by_cve_out,
        "attrs":  [{"attr": a, "label": l, "unit": u} for a, l, u in MUNI_CMP_ATTRS],
    }


# ---------------------------------------------------------------------------
# §02 Mapa de Atributos Físicos
# ---------------------------------------------------------------------------

PHYS_ATTR_HIST_ATTRS = [
    ("clase",          "Clase del valuador",      ""),
    ("recamaras",      "Recámaras",               ""),
    ("banos",          "Baños",                   ""),
    ("sup_construida", "Superficie construida",   "m²"),
    ("sup_terreno",    "Superficie terreno",      "m²"),
    ("edad_meses",     "Antigüedad (meses)",      "m"),
]


def prepare_phys_attr_points(df: pd.DataFrame, max_points: int = 220_000) -> dict:
    """Subset compacto para el mapa de atributos físicos: lat, lon y atributos
    físicos. No requiere clustering. Sample si supera max_points (los stats en
    buffer se calculan sobre lo que ve el JS, lo aceptamos como aproximación)."""
    work = df.dropna(subset=["latitud", "longitud"]).copy()
    work = work[
        work["latitud"].between(19.00, 19.70) &
        work["longitud"].between(-99.50, -98.85)
    ]
    if len(work) > max_points:
        work = work.sample(n=max_points, random_state=42)

    def _i(v):
        try: return int(v) if pd.notna(v) else None
        except (TypeError, ValueError): return None
    def _ri(v):
        try: return int(round(float(v))) if pd.notna(v) else None
        except (TypeError, ValueError): return None

    points = []
    for _, r in work.iterrows():
        try:
            lat = round(float(r["latitud"]), 5)
            lon = round(float(r["longitud"]), 5)
        except (TypeError, ValueError):
            continue
        points.append([
            lat, lon,
            _i(r.get("clase")),                            # 2 clase
            _i(r.get("recamaras")),                        # 3 recamaras
            _i(r.get("banos")),                            # 4 banos
            _ri(r.get("sup_construida")),                  # 5 sup_construida
            _ri(r.get("sup_terreno")),                     # 6 sup_terreno
            _i(r.get("edad_meses")),                       # 7 edad_meses
            _i(r.get("ano")),                              # 8 año
            _ri(r.get("valor_concluido")),                 # 9 valor_concluido
        ])
    return {"points": points, "n_total": int(len(work))}


PHYS_MAP_JS_TEMPLATE = r"""
(function() {
  var DATA = __PHYS_MAP_DATA__;
  var pts = DATA.points || [];
  var canvas = document.getElementById('phys-map-canvas');
  if (!canvas) return;
  if (typeof L === 'undefined') {
    canvas.innerHTML = '<div class="map-err">Leaflet no pudo cargarse. Mapa de atributos físicos deshabilitado.</div>';
    return;
  }

  // Índices
  var P_LAT=0,P_LON=1,P_CL=2,P_REC=3,P_BAN=4,P_SC=5,P_ST=6,P_ED=7,P_Y=8,P_VC=9;

  // Mapa: drag habilitado para que el usuario pueda navegar la ciudad y
  // ubicar zonas específicas; el click recentra el buffer (no auto-zoom).
  var map = L.map(canvas, {
    center: [19.4326, -99.1332],
    zoom: 12,
    dragging: true,
    scrollWheelZoom: true,
    doubleClickZoom: true,
    boxZoom: false,
    keyboard: false,
    zoomControl: true,
    attributionControl: false
  });

  var lightLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    { subdomains: 'abcd', maxZoom: 20 }).addTo(map);
  var satLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    { maxZoom: 19 });

  // Marcadores compactos (canvas para performance). El sample visual es
  // intencionalmente bajo (~10k) y con opacidad reducida para que las calles
  // y colonias del basemap sean legibles debajo.
  var renderer = L.canvas({padding: 0.5});
  var visualSample = pts;
  if (visualSample.length > 10000) {
    var step = Math.ceil(visualSample.length / 10000);
    visualSample = visualSample.filter(function(_, i){ return i % step === 0; });
  }
  visualSample.forEach(function(p) {
    L.circleMarker([p[P_LAT], p[P_LON]], {
      radius: 1.7, color: '#c9421f', fillColor: '#c9421f', fillOpacity: 0.42, weight: 0, renderer: renderer
    }).addTo(map);
  });

  // Buffer
  var center = L.latLng(19.4326, -99.1332);
  var bufRadius = 1000;
  var bufCircle = L.circle(center, { radius: bufRadius, color: '#c9421f', weight: 1.8, fillColor: '#c9421f', fillOpacity: 0.08 }).addTo(map);
  var bufPin = L.circleMarker(center, { radius: 5, color: '#c9421f', fillColor: '#c9421f', fillOpacity: 1, weight: 1 }).addTo(map);

  // Helpers
  function haversineM(a, b) {
    var R = 6371000, dLat = (b.lat-a.lat)*Math.PI/180, dLon = (b.lng-a.lng)*Math.PI/180;
    var lat1 = a.lat*Math.PI/180, lat2 = b.lat*Math.PI/180;
    var h = Math.sin(dLat/2)*Math.sin(dLat/2) + Math.sin(dLon/2)*Math.sin(dLon/2)*Math.cos(lat1)*Math.cos(lat2);
    return 2*R*Math.asin(Math.sqrt(h));
  }
  function inBuffer() {
    var out = [];
    for (var i = 0; i < pts.length; i++) {
      var p = pts[i];
      if (haversineM({lat:p[P_LAT],lng:p[P_LON]}, center) <= bufRadius) out.push(p);
    }
    return out;
  }
  function median(arr) {
    arr = arr.filter(function(v){ return v !== null && v !== undefined && !isNaN(v); }).sort(function(a,b){return a-b;});
    if (!arr.length) return null;
    var m = Math.floor(arr.length/2);
    return arr.length % 2 ? arr[m] : (arr[m-1]+arr[m])/2;
  }
  function mean(arr) {
    var clean = arr.filter(function(v){ return v !== null && v !== undefined && !isNaN(v); });
    if (!clean.length) return null;
    var s = 0;
    for (var i = 0; i < clean.length; i++) s += clean[i];
    return s / clean.length;
  }
  function mode(arr) {
    var c = {}, best = null, bestN = 0;
    arr.forEach(function(v){ if (v === null || v === undefined) return; c[v] = (c[v]||0)+1; if (c[v] > bestN) { bestN = c[v]; best = v; } });
    return best;
  }

  // Histograma para click en métrica
  function buildHistFromValues(values, nbins) {
    var v = values.filter(function(x){ return x !== null && x !== undefined && !isNaN(x); });
    if (v.length < 5) return null;
    nbins = nbins || 20;
    var lo = Math.min.apply(null, v), hi = Math.max.apply(null, v);
    if (hi <= lo) hi = lo + 1;
    var step = (hi - lo) / nbins;
    var edges = [], counts = new Array(nbins).fill(0);
    for (var i = 0; i <= nbins; i++) edges.push(lo + step*i);
    v.forEach(function(x){
      var idx = Math.floor((x - lo) / step);
      if (idx < 0) idx = 0; if (idx >= nbins) idx = nbins - 1;
      counts[idx]++;
    });
    var sum = v.reduce(function(a,b){return a+b;}, 0);
    var sorted = v.slice().sort(function(a,b){return a-b;});
    var med = sorted.length%2 ? sorted[Math.floor(sorted.length/2)] : (sorted[sorted.length/2-1]+sorted[sorted.length/2])/2;
    return { edges: edges, counts: counts, median: med, mean: sum/v.length, n: v.length };
  }
  function fmtCount(v) { return Math.round(v).toLocaleString('en-US'); }
  function fmtM2(v)    { return Math.round(v) + ' m²'; }
  function fmtMeses(v) { return (v/12).toFixed(0) + ' a'; }
  var ATTR_META = {
    clase:        { label: 'Clase del valuador',     idx: P_CL, fmt: fmtCount },
    recamaras:    { label: 'Recámaras',              idx: P_REC, fmt: fmtCount },
    banos:        { label: 'Baños',                  idx: P_BAN, fmt: fmtCount },
    sup_const:    { label: 'Superficie construida',  idx: P_SC, fmt: fmtM2 },
    sup_terr:     { label: 'Superficie terreno',     idx: P_ST, fmt: fmtM2 },
    edad:         { label: 'Antigüedad',             idx: P_ED, fmt: fmtMeses }
  };

  // Formatos para la lista de atributos. Las funciones del panel usan
  // separador de miles y distinguen visualmente la mediana (entera) del
  // promedio (1 decimal en conteos y años).
  var paFmtCntMed   = function(v) { return Math.round(v).toLocaleString('en-US'); };
  var paFmtCntMean  = function(v) { return v.toFixed(1); };
  var paFmtM2       = function(v) { return Math.round(v).toLocaleString('en-US') + ' m²'; };
  var paFmtAnosMed  = function(v) { return (v/12).toFixed(0) + ' a'; };
  var paFmtAnosMean = function(v) { return (v/12).toFixed(1) + ' a'; };

  function renderPair(elId, vals, fmtMed, fmtMn) {
    var med = median(vals);
    var mn  = mean(vals);
    var a = (med === null) ? '—' : fmtMed(med);
    var b = (mn  === null) ? '—' : (fmtMn || fmtMed)(mn);
    document.getElementById(elId).innerHTML = a + ' <span class="pa-sep">|</span> ' + b;
  }

  function refresh() {
    var buf = inBuffer();
    var n = buf.length;
    document.getElementById('phys-n').textContent = n.toLocaleString('en-US');
    document.getElementById('phys-center').textContent = center.lat.toFixed(4) + ', ' + center.lng.toFixed(4);
    document.getElementById('phys-buf-val').textContent = (bufRadius).toLocaleString('en-US');
    if (!n) {
      document.getElementById('pa-clase').textContent = '—';
      ['pa-rec','pa-ban','pa-sc','pa-st','pa-edad'].forEach(function(id){
        document.getElementById(id).innerHTML = '—';
      });
      return;
    }
    var cl = mode(buf.map(function(p){return p[P_CL];}));
    document.getElementById('pa-clase').textContent = (cl === null ? '—' : ('Clase '+cl));
    renderPair('pa-rec',  buf.map(function(p){return p[P_REC];}), paFmtCntMed,  paFmtCntMean);
    renderPair('pa-ban',  buf.map(function(p){return p[P_BAN];}), paFmtCntMed,  paFmtCntMean);
    renderPair('pa-sc',   buf.map(function(p){return p[P_SC];}),  paFmtM2);
    renderPair('pa-st',   buf.map(function(p){return p[P_ST];}),  paFmtM2);
    renderPair('pa-edad', buf.map(function(p){return p[P_ED];}),  paFmtAnosMed, paFmtAnosMean);
  }

  // Click en el mapa: recentra el buffer sin tocar el nivel de zoom (el usuario
  // pidió no auto-zoom; el panel de atributos se actualiza dentro del buffer).
  map.on('click', function(e) {
    center = e.latlng;
    bufCircle.setLatLng(center).setRadius(bufRadius);
    bufPin.setLatLng(center);
    refresh();
  });

  // Slider de buffer
  var slider = document.getElementById('phys-buf');
  if (slider) slider.addEventListener('input', function() {
    bufRadius = parseInt(slider.value, 10) || 1000;
    bufCircle.setRadius(bufRadius);
    refresh();
  });

  // Toggle satélite
  var sat = document.getElementById('phys-sat');
  if (sat) sat.addEventListener('change', function() {
    if (sat.checked) { map.removeLayer(lightLayer); satLayer.addTo(map); }
    else { map.removeLayer(satLayer); lightLayer.addTo(map); }
  });

  // Click en métrica del popup → histograma del buffer
  document.querySelectorAll('.phys-attr.clickable').forEach(function(li) {
    li.addEventListener('click', function() {
      var attr = li.dataset.attr;
      var meta = ATTR_META[attr];
      if (!meta || !window.__openModal || !window.__renderHistSvg) return;
      var buf = inBuffer();
      var values = buf.map(function(p){ return p[meta.idx]; });
      var hist = buildHistFromValues(values);
      var title = '<span class="modal-tag" style="background:#1f6c5c">Buf</span> '+meta.label+' · buffer ('+buf.length.toLocaleString('en-US')+' avalúos)';
      var body;
      if (!hist) {
        body = '<p class="empty">Datos insuficientes en el buffer (n &lt; 5).</p>';
      } else {
        body = '<div class="modal-stats">'
             + '<div class="ms"><span class="lbl">mediana</span><span class="val">'+meta.fmt(hist.median)+'</span></div>'
             + '<div class="ms"><span class="lbl">promedio</span><span class="val">'+meta.fmt(hist.mean)+'</span></div>'
             + '<div class="ms"><span class="lbl">n</span><span class="val">'+hist.n.toLocaleString('en-US')+'</span></div>'
             + '</div>'
             + '<div class="hist-card"><div class="hist-head"><span class="hist-title">'+meta.label+' · '+buf.length.toLocaleString('en-US')+' avalúos</span></div>'
             + window.__renderHistSvg(hist, 720, 240, meta.fmt) + '</div>';
      }
      window.__openModal(title, body);
    });
  });

  // Exportar PNG
  var exp = document.getElementById('phys-export-pdf');
  if (exp) exp.addEventListener('click', function() {
    if (typeof html2canvas === 'undefined') {
      alert('html2canvas no se cargó (sin conexión a CDN). No se puede exportar.');
      return;
    }
    var node = document.querySelector('.map-layout.phys') || document.querySelector('.phys-map-wrap');
    html2canvas(node, { useCORS: true, backgroundColor: '#f7f2e7' }).then(function(canvas) {
      var a = document.createElement('a');
      a.download = 'atributos-fisicos.png';
      a.href = canvas.toDataURL('image/png');
      a.click();
    });
  });

  refresh();
})();
"""


def render_phys_attr_map_section(phys_data: dict,
                                  years: list[int],
                                  clases: list[int]) -> str:
    if not phys_data.get("points"):
        return (
            '<h2 class="sec"><span class="num-tag">§02</span>Atributos Físicos <em>de la Vivienda</em></h2>'
            '<div class="sec-rule"></div>'
            '<p class="empty">No hay puntos con lat/lon válidos para este mapa.</p>'
        )

    payload = json.dumps({"points": phys_data["points"]},
                         separators=(",", ":"), ensure_ascii=False, default=str).replace("</", "<\\/")
    js = PHYS_MAP_JS_TEMPLATE.replace("__PHYS_MAP_DATA__", payload)

    return f"""
    <h2 class="sec"><span class="num-tag">§02</span>Atributos Físicos <em>de la Vivienda</em></h2>
    <div class="sec-rule"></div>
    <p class="sec-purpose">Para diseñar producto e iterar hipótesis: ¿qué tipo de inmueble se está transaccionando en una zona específica y con qué dimensiones? Define un radio alrededor del punto que te interesa y mira el perfil físico predominante de las viviendas vendidas allí.</p>
    <ul class="sec-bullets">
      <li><b>Arrastra el mapa</b> para navegar la ciudad; <b>rueda</b> o botones <b>+/−</b> para zoom.</li>
      <li><b>Click</b> en cualquier punto para fijar el centro del buffer.</li>
      <li>Ajusta el <b>tamaño del buffer</b> con el slider del panel derecho.</li>
      <li>El panel resume <b>mediana y promedio</b> de cada atributo dentro del buffer.</li>
      <li><b>Click en una métrica</b> abre su histograma restringido al buffer.</li>
    </ul>

    <div class="map-block phys-map-wrap">
      <div class="map-layout phys">
        <div class="map-area">
          <div id="phys-map-canvas"></div>
          <div class="phys-map-tools">
            <label class="toggle-row"><input type="checkbox" id="phys-sat"/> <span>Imagen satelital</span></label>
            <button id="phys-export-pdf" type="button" class="export-png">Exportar Vista PNG</button>
          </div>
        </div>
        <aside class="map-panel phys-panel">
          <h4>Atributos en el buffer</h4>
          <p class="phys-loc">Centro: <span id="phys-center">19.4326, -99.1332</span><br/>
             Buffer: <b id="phys-buf-val">1,000</b> m · n=<span id="phys-n">—</span></p>
          <ul class="phys-attr-list">
            <li class="phys-attr clickable" data-attr="clase">    <span class="lbl">Clase dominante</span><span class="val" id="pa-clase">—</span></li>
            <li class="phys-attr clickable" data-attr="recamaras"><span class="lbl">Recámaras (Mediana <span class="pa-sep">|</span> Promedio)</span><span class="val" id="pa-rec">—</span></li>
            <li class="phys-attr clickable" data-attr="banos">    <span class="lbl">Baños (Mediana <span class="pa-sep">|</span> Promedio)</span><span class="val" id="pa-ban">—</span></li>
            <li class="phys-attr clickable" data-attr="sup_const"><span class="lbl">Sup. construida (Mediana <span class="pa-sep">|</span> Promedio)</span><span class="val" id="pa-sc">—</span></li>
            <li class="phys-attr clickable" data-attr="sup_terr"> <span class="lbl">Sup. terreno (Mediana <span class="pa-sep">|</span> Promedio)</span><span class="val" id="pa-st">—</span></li>
            <li class="phys-attr clickable" data-attr="edad">     <span class="lbl">Edad (Mediana <span class="pa-sep">|</span> Promedio)</span><span class="val" id="pa-edad">—</span></li>
          </ul>
          <label class="phys-buf-label">Tamaño del buffer (m)</label>
          <input type="range" id="phys-buf" min="200" max="5000" step="100" value="1000"/>
          <p class="hint">Click en una métrica → histograma de esa métrica dentro del buffer.</p>
        </aside>
      </div>
    </div>

    <script>{js}</script>
    """


# ---------------------------------------------------------------------------
# §07 Absorción por desarrollador
# ---------------------------------------------------------------------------

def compute_absorcion_data(df: pd.DataFrame, top_n: int = 20) -> dict:
    """Top-N desarrolladores por id_avaluo único. Devuelve filas compactas
    para que el JS filtre por atributos físicos y reagrupe por año/trimestre."""
    if "constructor" not in df.columns:
        return {"developers": [], "rows": [], "years": []}
    work = df.dropna(subset=["constructor"]).copy()
    work["constructor"] = work["constructor"].astype(str).str.strip()
    work = work[work["constructor"].str.len() > 0]

    # id_avaluo único por desarrollador
    if "id_avaluo" in work.columns:
        counts = (work.drop_duplicates(subset=["id_avaluo", "constructor"])
                      .groupby("constructor").size()
                      .sort_values(ascending=False))
    else:
        counts = work["constructor"].value_counts()
    top_devs = counts.head(top_n).index.tolist()
    cons_idx = {name: i for i, name in enumerate(top_devs)}

    def _i(v):
        try: return int(v) if pd.notna(v) else None
        except (TypeError, ValueError): return None
    def _ri(v):
        try: return int(round(float(v))) if pd.notna(v) else None
        except (TypeError, ValueError): return None

    rows = []
    seen = set()
    for _, r in work.iterrows():
        ci = cons_idx.get(r["constructor"])
        if ci is None:  # esconde "Otros"
            continue
        ida = r.get("id_avaluo")
        if ida is not None and (ida, ci) in seen:
            continue
        if ida is not None:
            seen.add((ida, ci))
        rows.append([
            ci,
            _i(r.get("ano")),
            _i(r.get("trimestre")),
            _i(r.get("clase")),
            _i(r.get("recamaras")),
            _i(r.get("banos")),
            _ri(r.get("sup_construida")),
            _ri(r.get("sup_terreno")),
            _i(r.get("edad_meses")),
            _ri(r.get("valor_concluido")),
        ])

    years = sorted({rw[1] for rw in rows if rw[1] is not None})

    return {
        "developers": top_devs,
        "totals":     {d: int(counts.get(d, 0)) for d in top_devs},
        "rows":       rows,
        "years":      years,
    }


ABS_JS_TEMPLATE = r"""
(function() {
  var DATA = __ABS_DATA__;
  var devs = DATA.developers || [];
  var rows = DATA.rows || [];
  var years = DATA.years || [];

  // Índices de columna en cada row
  var I_DEV=0, I_Y=1, I_Q=2, I_CL=3, I_REC=4, I_BAN=5, I_SC=6, I_ST=7, I_ED=8;

  // Estado de filtros
  var state = {
    gran: 'quarter',
    chart: 'line',
    clase: null,
    rec:  new Set(),
    ban:  new Set(),
    scMin: null, scMax: null,
    stMin: null, stMax: null,
    edMin: null, edMax: null,
    hidden: new Set()  // dev indices con la serie oculta
  };

  function setFromInput(id, key) {
    var el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', function() {
      var v = el.value.trim();
      state[key] = v === '' ? null : parseFloat(v);
      apply();
    });
  }
  function setupChips(containerId, key) {
    var c = document.getElementById(containerId);
    if (!c) return;
    c.querySelectorAll('button.chip').forEach(function(btn) {
      btn.addEventListener('click', function() {
        var v = parseInt(btn.dataset.val, 10);
        if (state[key].has(v)) { state[key].delete(v); btn.classList.remove('on'); }
        else { state[key].add(v); btn.classList.add('on'); }
        apply();
      });
    });
  }

  function filtered() {
    var out = [];
    for (var i = 0; i < rows.length; i++) {
      var r = rows[i];
      if (state.clase !== null && r[I_CL] !== state.clase) continue;
      if (state.rec.size > 0 && !state.rec.has(r[I_REC])) continue;
      if (state.ban.size > 0 && !state.ban.has(r[I_BAN])) continue;
      if (state.scMin !== null && (r[I_SC] === null || r[I_SC] < state.scMin)) continue;
      if (state.scMax !== null && (r[I_SC] === null || r[I_SC] > state.scMax)) continue;
      if (state.stMin !== null && (r[I_ST] === null || r[I_ST] < state.stMin)) continue;
      if (state.stMax !== null && (r[I_ST] === null || r[I_ST] > state.stMax)) continue;
      if (state.edMin !== null && (r[I_ED] === null || r[I_ED] < state.edMin)) continue;
      if (state.edMax !== null && (r[I_ED] === null || r[I_ED] > state.edMax)) continue;
      out.push(r);
    }
    return out;
  }

  function periodKey(r) {
    if (state.gran === 'year') return String(r[I_Y]);
    return r[I_Y] + '-Q' + (r[I_Q] || '?');
  }
  function allPeriods() {
    if (state.gran === 'year') return years.map(String);
    var out = [];
    years.forEach(function(y) {
      [1,2,3,4].forEach(function(q){ out.push(y + '-Q' + q); });
    });
    return out;
  }

  function groupForChart(rs) {
    // grouped[devIdx][period] = count
    var grouped = {};
    rs.forEach(function(r) {
      var di = r[I_DEV];
      if (state.hidden.has(di)) return;
      if (r[I_Y] === null || r[I_Y] === undefined) return;
      if (state.gran !== 'year' && (r[I_Q] === null || r[I_Q] === undefined)) return;
      var pk = periodKey(r);
      if (!grouped[di]) grouped[di] = {};
      grouped[di][pk] = (grouped[di][pk] || 0) + 1;
    });
    return grouped;
  }

  function topRanking(rs) {
    // El TOP debe variar con la selección — sólo cuenta desarrolladores
    // incluidos en el gráfico (no los excluidos via click en la lista).
    var counts = {};
    rs.forEach(function(r) {
      var di = r[I_DEV];
      if (state.hidden.has(di)) return;
      counts[di] = (counts[di]||0) + 1;
    });
    var arr = Object.keys(counts).map(function(k){ return {idx:parseInt(k,10), n:counts[k]}; });
    arr.sort(function(a,b){ return b.n - a.n; });
    return arr.slice(0, 10);
  }

  // Paleta 10 colores
  var COLORS = ['#c9421f','#1f6c5c','#b88a1f','#2a3d6e','#7b3b5e','#4d6b1d','#a44b1a','#0f4c75','#bc4749','#6a994e'];

  function renderLine(grouped, periods) {
    var W = 980, H = 460, PL = 60, PR = 24, PT = 24, PB = 60;
    var maxY = 0;
    Object.values(grouped).forEach(function(byP) { Object.values(byP).forEach(function(v){ if (v > maxY) maxY = v; }); });
    if (!maxY) maxY = 1;
    var xStep = (W - PL - PR) / Math.max(periods.length - 1, 1);
    var yOf = function(v) { return H - PB - (v / maxY) * (H - PT - PB); };
    var xOf = function(i) { return PL + i * xStep; };
    var svg = '';
    // grid + ejes
    for (var g = 1; g <= 4; g++) {
      var yv = Math.round(maxY * g / 4);
      var y = yOf(yv);
      svg += '<line x1="'+PL+'" y1="'+y+'" x2="'+(W-PR)+'" y2="'+y+'" stroke="#dcd4c2" stroke-width="0.6"/>';
      svg += '<text x="'+(PL-6)+'" y="'+(y+3)+'" text-anchor="end" font-size="10" fill="#6a635a">'+yv+'</text>';
    }
    // x labels (rotadas si es trimestre)
    periods.forEach(function(p, i) {
      if (state.gran === 'year' || i % 2 === 0) {
        svg += '<text x="'+xOf(i)+'" y="'+(H-PB+14)+'" text-anchor="middle" font-size="10" fill="#6a635a">'+p+'</text>';
      }
    });
    svg += '<line x1="'+PL+'" y1="'+(H-PB)+'" x2="'+(W-PR)+'" y2="'+(H-PB)+'" stroke="#1c1a17" stroke-width="0.8"/>';
    // lines
    Object.keys(grouped).forEach(function(di) {
      var byP = grouped[di];
      var pts = [];
      periods.forEach(function(p, i) {
        var v = byP[p] || 0;
        pts.push(xOf(i).toFixed(1) + ',' + yOf(v).toFixed(1));
      });
      var color = COLORS[parseInt(di,10) % COLORS.length];
      svg += '<polyline points="'+pts.join(' ')+'" fill="none" stroke="'+color+'" stroke-width="1.8" opacity="0.92"/>';
      // dots
      periods.forEach(function(p, i) {
        var v = byP[p];
        if (!v) return;
        svg += '<circle cx="'+xOf(i).toFixed(1)+'" cy="'+yOf(v).toFixed(1)+'" r="3" fill="'+color+'"><title>'+devs[di]+' · '+p+' · '+v+'</title></circle>';
      });
    });
    return svg;
  }

  function renderBars(grouped, periods) {
    var W = 980, H = 460, PL = 60, PR = 24, PT = 24, PB = 60;
    var keys = Object.keys(grouped);
    var maxY = 0;
    periods.forEach(function(p) {
      keys.forEach(function(di) { var v = grouped[di][p] || 0; if (v > maxY) maxY = v; });
    });
    if (!maxY) maxY = 1;
    var groupW = (W - PL - PR) / periods.length;
    var bw = Math.max(2, (groupW - 2) / Math.max(keys.length, 1));
    var yOf = function(v) { return H - PB - (v / maxY) * (H - PT - PB); };
    var svg = '';
    for (var g = 1; g <= 4; g++) {
      var yv = Math.round(maxY * g / 4);
      var y = yOf(yv);
      svg += '<line x1="'+PL+'" y1="'+y+'" x2="'+(W-PR)+'" y2="'+y+'" stroke="#dcd4c2" stroke-width="0.6"/>';
      svg += '<text x="'+(PL-6)+'" y="'+(y+3)+'" text-anchor="end" font-size="10" fill="#6a635a">'+yv+'</text>';
    }
    periods.forEach(function(p, i) {
      var x0 = PL + i * groupW + 1;
      if (state.gran === 'year' || i % 2 === 0) {
        svg += '<text x="'+(x0 + groupW/2)+'" y="'+(H-PB+14)+'" text-anchor="middle" font-size="10" fill="#6a635a">'+p+'</text>';
      }
      keys.forEach(function(di, k) {
        var v = grouped[di][p] || 0;
        if (!v) return;
        var color = COLORS[parseInt(di,10) % COLORS.length];
        var y = yOf(v);
        var x = x0 + k * bw;
        svg += '<rect x="'+x.toFixed(1)+'" y="'+y.toFixed(1)+'" width="'+(bw-0.6).toFixed(1)+'" height="'+(H-PB-y).toFixed(1)+'" fill="'+color+'" opacity="0.9"><title>'+devs[di]+' · '+p+' · '+v+'</title></rect>';
      });
    });
    svg += '<line x1="'+PL+'" y1="'+(H-PB)+'" x2="'+(W-PR)+'" y2="'+(H-PB)+'" stroke="#1c1a17" stroke-width="0.8"/>';
    return svg;
  }

  function renderArea(grouped, periods) {
    // Stacked area: por cada periodo, suma desde abajo cada serie.
    var W = 980, H = 460, PL = 60, PR = 24, PT = 24, PB = 60;
    var keys = Object.keys(grouped);
    var totals = periods.map(function(p) {
      var t = 0;
      keys.forEach(function(di){ t += (grouped[di][p]||0); });
      return t;
    });
    var maxY = Math.max.apply(null, totals) || 1;
    var xStep = (W - PL - PR) / Math.max(periods.length - 1, 1);
    var yOf = function(v) { return H - PB - (v / maxY) * (H - PT - PB); };
    var xOf = function(i) { return PL + i * xStep; };
    var svg = '';
    // grid
    for (var g = 1; g <= 4; g++) {
      var yv = Math.round(maxY * g / 4);
      var y = yOf(yv);
      svg += '<line x1="'+PL+'" y1="'+y+'" x2="'+(W-PR)+'" y2="'+y+'" stroke="#dcd4c2" stroke-width="0.6"/>';
      svg += '<text x="'+(PL-6)+'" y="'+(y+3)+'" text-anchor="end" font-size="10" fill="#6a635a">'+yv+'</text>';
    }
    var bottom = periods.map(function(){ return 0; });
    keys.forEach(function(di) {
      var top = periods.map(function(p, i){ return bottom[i] + (grouped[di][p]||0); });
      var fwd = top.map(function(v, i){ return xOf(i).toFixed(1) + ',' + yOf(v).toFixed(1); });
      var bwd = [];
      for (var i = bottom.length - 1; i >= 0; i--) bwd.push(xOf(i).toFixed(1) + ',' + yOf(bottom[i]).toFixed(1));
      var color = COLORS[parseInt(di,10) % COLORS.length];
      svg += '<polygon points="'+fwd.concat(bwd).join(' ')+'" fill="'+color+'" fill-opacity="0.55" stroke="'+color+'" stroke-width="1"><title>'+devs[di]+'</title></polygon>';
      bottom = top;
    });
    periods.forEach(function(p, i) {
      if (state.gran === 'year' || i % 2 === 0) {
        svg += '<text x="'+xOf(i)+'" y="'+(H-PB+14)+'" text-anchor="middle" font-size="10" fill="#6a635a">'+p+'</text>';
      }
    });
    svg += '<line x1="'+PL+'" y1="'+(H-PB)+'" x2="'+(W-PR)+'" y2="'+(H-PB)+'" stroke="#1c1a17" stroke-width="0.8"/>';
    return svg;
  }

  function apply() {
    var rs = filtered();
    var periods = allPeriods();
    var grouped = groupForChart(rs);
    var svgInner = '';
    if (state.chart === 'line') svgInner = renderLine(grouped, periods);
    else if (state.chart === 'bars') svgInner = renderBars(grouped, periods);
    else svgInner = renderArea(grouped, periods);
    document.getElementById('abs-svg').innerHTML = svgInner;

    // Determina el universo: todos los desarrolladores presentes en los datos
    // filtrados, separados en incluidos y excluidos.
    var presentSet = {};
    rs.forEach(function(r) { presentSet[r[I_DEV]] = true; });
    var presentDevs = Object.keys(presentSet).map(function(k){ return parseInt(k,10); });
    presentDevs.sort(function(a,b){
      var na = (devs[a]||'').toLowerCase(), nb = (devs[b]||'').toLowerCase();
      return na < nb ? -1 : na > nb ? 1 : 0;
    });
    var incl = [], excl = [];
    presentDevs.forEach(function(di) {
      if (state.hidden.has(di)) excl.push(di);
      else                       incl.push(di);
    });
    function chip(di, cls) {
      var color = COLORS[di % COLORS.length];
      return '<li class="abs-leg-item '+cls+'" data-di="'+di+'"><span class="abs-leg-sw" style="background:'+color+'"></span>'+devs[di]+'</li>';
    }
    document.getElementById('abs-incl-list').innerHTML = incl.map(function(di){return chip(di,'in');}).join('')
      || '<li class="empty">Ninguno — el gráfico está vacío.</li>';
    document.getElementById('abs-excl-list').innerHTML = excl.map(function(di){return chip(di,'out');}).join('')
      || '<li class="empty">Ninguno — todos los desarrolladores están en el gráfico.</li>';
    document.getElementById('abs-incl-n').textContent = incl.length;
    document.getElementById('abs-excl-n').textContent = excl.length;
    var metaEl = document.getElementById('abs-chart-meta');
    if (metaEl) metaEl.textContent = incl.length + ' series activas · ' + excl.length + ' excluidas';

    // Wire-up de click en cualquier chip — toggle hidden.
    document.querySelectorAll('.abs-dev-lists .abs-leg-item').forEach(function(li) {
      li.addEventListener('click', function() {
        var di = parseInt(li.dataset.di, 10);
        if (state.hidden.has(di)) state.hidden.delete(di);
        else state.hidden.add(di);
        apply();
      });
    });

    // Top 10 — ahora filtra por state.hidden (ver topRanking).
    var topR = topRanking(rs);
    var maxN = topR.length ? topR[0].n : 1;
    var rank = topR.map(function(item, i) {
      var color = COLORS[item.idx % COLORS.length];
      var pct = (100 * item.n / maxN).toFixed(0);
      return '<li><span class="abs-rk-pos">'+(i+1)+'</span><span class="abs-rk-sw" style="background:'+color+'"></span>'
           + '<span class="abs-rk-name">'+devs[item.idx]+'</span>'
           + '<span class="abs-rk-bar"><span class="abs-rk-fill" style="width:'+pct+'%;background:'+color+'"></span></span>'
           + '<span class="abs-rk-n">'+item.n.toLocaleString('en-US')+'</span></li>';
    }).join('');
    document.getElementById('abs-rank').innerHTML = rank || '<li class="empty">Sin desarrolladores incluidos · ajusta filtros o reincluye alguno.</li>';
  }

  // Wire-up controles
  document.querySelectorAll('#abs-gran .seg-btn').forEach(function(b) {
    b.addEventListener('click', function() {
      document.querySelectorAll('#abs-gran .seg-btn').forEach(function(x){ x.classList.remove('on'); });
      b.classList.add('on');
      state.gran = b.dataset.gran;
      apply();
    });
  });
  document.querySelectorAll('#abs-chart .seg-btn').forEach(function(b) {
    b.addEventListener('click', function() {
      document.querySelectorAll('#abs-chart .seg-btn').forEach(function(x){ x.classList.remove('on'); });
      b.classList.add('on');
      state.chart = b.dataset.chart;
      apply();
    });
  });

  var claseSel = document.getElementById('abs-clase');
  if (claseSel) claseSel.addEventListener('change', function() {
    var v = claseSel.value;
    state.clase = (v === '') ? null : parseInt(v, 10);
    apply();
  });
  setupChips('abs-rec', 'rec');
  setupChips('abs-ban', 'ban');
  setFromInput('abs-sc-min', 'scMin'); setFromInput('abs-sc-max', 'scMax');
  setFromInput('abs-st-min', 'stMin'); setFromInput('abs-st-max', 'stMax');
  setFromInput('abs-edad-min', 'edMin'); setFromInput('abs-edad-max', 'edMax');

  var reset = document.getElementById('abs-reset');
  if (reset) reset.addEventListener('click', function() {
    state.clase = null;
    state.rec.clear(); state.ban.clear();
    state.scMin = state.scMax = state.stMin = state.stMax = state.edMin = state.edMax = null;
    state.hidden.clear();
    if (claseSel) claseSel.value = '';
    document.querySelectorAll('#abs-rec .chip.on, #abs-ban .chip.on').forEach(function(c){ c.classList.remove('on'); });
    document.querySelectorAll('.abs-filters input[type="number"]').forEach(function(i){ i.value = ''; });
    apply();
  });

  apply();
})();
"""


def render_absorcion_section(absorcion_data: dict, clases: list[int]) -> str:
    if not absorcion_data.get("developers"):
        return (
            '<h2 class="sec"><span class="num-tag">§07</span>Absorción <em>por desarrollador</em></h2>'
            '<div class="sec-rule"></div>'
            '<p class="empty">La columna <code>constructor</code> está vacía o no se pudo derivar.</p>'
        )

    payload = json.dumps(absorcion_data, separators=(",", ":"),
                        ensure_ascii=False, default=str).replace("</", "<\\/")
    js = ABS_JS_TEMPLATE.replace("__ABS_DATA__", payload)

    clase_opts = '<option value="">Todas</option>' + "".join(
        f'<option value="{c}">Clase {c}</option>' for c in clases
    )
    rec_chips = "".join(
        f'<button type="button" class="chip" data-val="{i}">{i}{"+" if i==5 else ""}</button>'
        for i in range(0, 6)
    )
    ban_chips = "".join(
        f'<button type="button" class="chip" data-val="{i}">{i}{"+" if i==5 else ""}</button>'
        for i in range(0, 6)
    )

    return f"""
    <h2 class="sec"><span class="num-tag">§07</span>Absorción <em>por desarrollador</em></h2>
    <div class="sec-rule"></div>
    <p class="sec-purpose">Para ubicar a quién le está funcionando vender en qué tipología y ritmo: identifica desarrolladores activos, su volumen y su evolución trimestral. Útil para benchmark competitivo y para detectar players que están absorbiendo segmentos comparables al producto que estás diseñando.</p>
    <p class="sec-intro">Cantidad de viviendas hipotecadas vendidas por desarrollador (columna <em>constructor</em> de stg_avaluos) a lo largo del tiempo. <b>Filtra por atributos físicos</b> para acotar el benchmark a tu tipología de interés. Click en cualquier desarrollador para esconder/mostrar su serie. ¿Te interesa un benchmark profundo de alguno? <b>Avísanos</b>, lo armamos.</p>

    <div class="abs-block">
      <aside class="abs-filters">
        <div class="abs-controls">
          <div class="seg" id="abs-gran">
            <button type="button" class="seg-btn on" data-gran="quarter">Trimestre</button>
            <button type="button" class="seg-btn" data-gran="year">Año</button>
          </div>
          <div class="seg" id="abs-chart">
            <button type="button" class="seg-btn on" data-chart="line">Líneas</button>
            <button type="button" class="seg-btn" data-chart="area">Área apilada</button>
            <button type="button" class="seg-btn" data-chart="bars">Barras</button>
          </div>
        </div>

        <h5>Filtros físicos</h5>
        <div class="f-row">
          <label class="fp-label">Clase del valuador</label>
          <select id="abs-clase">{clase_opts}</select>
        </div>
        <div class="f-row">
          <label class="fp-label">Recámaras</label>
          <div class="chip-row" id="abs-rec">{rec_chips}</div>
        </div>
        <div class="f-row">
          <label class="fp-label">Baños</label>
          <div class="chip-row" id="abs-ban">{ban_chips}</div>
        </div>
        <div class="f-row">
          <label class="fp-label">Sup. construida (m²)</label>
          <div class="dual-input">
            <input type="number" id="abs-sc-min" placeholder="mín"/>
            <input type="number" id="abs-sc-max" placeholder="máx"/>
          </div>
        </div>
        <div class="f-row">
          <label class="fp-label">Sup. terreno (m²)</label>
          <div class="dual-input">
            <input type="number" id="abs-st-min" placeholder="mín"/>
            <input type="number" id="abs-st-max" placeholder="máx"/>
          </div>
        </div>
        <div class="f-row">
          <label class="fp-label">Antigüedad (meses)</label>
          <div class="dual-input">
            <input type="number" id="abs-edad-min" placeholder="mín"/>
            <input type="number" id="abs-edad-max" placeholder="máx"/>
          </div>
        </div>
        <button id="abs-reset" type="button" class="abs-reset-btn">Reset</button>
      </aside>

      <div class="abs-chart-wrap">
        <div class="abs-chart-head">
          <h4 class="abs-chart-title">Click en desarrollador para <b>removerlo</b> / <b>ponerlo</b></h4>
          <span class="abs-chart-meta" id="abs-chart-meta">— series activas</span>
        </div>
        <svg id="abs-svg" viewBox="0 0 980 460" preserveAspectRatio="xMidYMid meet"></svg>

        <div class="abs-dev-lists">
          <section class="abs-dev-section abs-dev-incl">
            <header><span class="abs-dot in"></span><h5>Incluidos en el gráfico</h5><span class="abs-dev-n" id="abs-incl-n">0</span></header>
            <ul class="abs-legend abs-incl-list" id="abs-incl-list"></ul>
            <p class="abs-dev-hint">Click para removerlo del gráfico.</p>
          </section>
          <section class="abs-dev-section abs-dev-excl">
            <header><span class="abs-dot out"></span><h5>Excluidos (click para incluir)</h5><span class="abs-dev-n" id="abs-excl-n">0</span></header>
            <ul class="abs-legend abs-excl-list" id="abs-excl-list"></ul>
            <p class="abs-dev-hint">Click para volverlo a poner en el gráfico.</p>
          </section>
        </div>
      </div>

      <aside class="abs-top">
        <h4>Top 10 desarrolladores</h4>
        <p class="hint">según los filtros activos y los incluidos en el gráfico</p>
        <ol class="abs-rank" id="abs-rank"></ol>
        <p class="abs-cta">¿Quieres un benchmark profundo de alguno? <b>Avísanos</b>, lo armamos a medida.</p>
      </aside>
    </div>

    <script>{js}</script>
    """


def train_price_regression(df: pd.DataFrame,
                            cluster: "ClusterResult | None") -> dict | None:
    """Regresión lineal en log-precio sobre TODO el dataset (sin train/test
    split — la idea es coeficientes estables, no diagnosticar overfit).

    Features:
      lat, lon, sup_construida, log(sup_construida), recámaras, baños,
      edad_meses, sqrt(edad_meses), y dummies de cluster (C0 = baseline).

    Devuelve dict listo para inyectarse en el JS:
      {intercept, coefs[], features[], sigma, r2, n_train, n_clusters}

    El intervalo de confianza al 95% en JS se aproxima como
      exp(pred_log ± 1.96 · sigma)
    ignorando leverage (válido para N grande).
    """
    work = df.copy()
    if cluster is not None:
        work["_cluster"] = cluster.labels
    req = ["valor_concluido", "latitud", "longitud", "sup_construida",
           "recamaras", "banos", "edad_meses"]
    work = work.dropna(subset=req)
    if cluster is not None:
        work = work.dropna(subset=["_cluster"])
        work["_cluster"] = work["_cluster"].astype(int)
    if len(work) < 1000:
        return None

    sup  = work["sup_construida"].astype(float).clip(lower=1.0).values
    edad = work["edad_meses"].astype(float).clip(lower=0.0).values
    feats = {
        "lat":           work["latitud"].astype(float).values,
        "lon":           work["longitud"].astype(float).values,
        "sup_const":     sup,
        "log_sup_const": np.log(sup),
        "recamaras":     work["recamaras"].astype(float).values,
        "banos":         work["banos"].astype(float).values,
        "edad_meses":    edad,
        "sqrt_edad":     np.sqrt(edad),
    }
    if cluster is not None:
        for k in range(1, cluster.k):
            feats[f"clu_{k}"] = (work["_cluster"] == k).astype(float).values
    feature_names = list(feats.keys())
    X = np.column_stack([feats[n] for n in feature_names])
    y = np.log(work["valor_concluido"].astype(float).clip(lower=1.0).values)

    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(X, y)
    y_pred = lr.predict(X)
    residuals = y - y_pred
    n = len(y)
    p = X.shape[1] + 1
    sigma = float(np.sqrt((residuals ** 2).sum() / max(n - p, 1)))
    r2 = float(lr.score(X, y))

    return {
        "intercept":  float(lr.intercept_),
        "coefs":      [float(c) for c in lr.coef_],
        "features":   feature_names,
        "sigma":      sigma,
        "r2":         r2,
        "n_train":    int(n),
        "n_clusters": (cluster.k if cluster else 0),
    }


def render_predictor_section(cluster: "ClusterResult | None") -> str:
    """Sección §06 — estimador de precio (regresión lineal en log-precio
    entrenada sobre el dataset completo) + Análisis Comparativo de Mercado
    (comparables del cluster seleccionado dentro de 3 km, con score de
    similitud). La lógica vive en MAP_JS_TEMPLATE (comparte el dataset de
    puntos + la regresión)."""
    if cluster is None:
        return ""
    cluster_opts = "\n".join(
        f'<option value="{cid}">C{cid} · {html.escape(name)}</option>'
        for cid, name in sorted(cluster.cluster_names.items())
    )
    return f"""
    <h2 class="sec"><span class="num-tag">§06</span>Estima el <em>precio</em> de un inmueble</h2>
    <div class="sec-rule"></div>
    <p class="sec-purpose">Para sanity-check de pricing y diseño de producto: ingresa una zona y los atributos del inmueble que quieres lanzar, y obtén el precio esperado más los 10 comparables más similares dentro de 3 km. Útil para validar que un producto entra en el rango del submercado antes de cerrar el diseño.</p>
    <p class="sec-intro">Precio estimado con regresión lineal sobre los 189k avalúos completos. Toma en cuenta latitud, longitud, superficie construida, recámaras, baños, antigüedad y submercado (cluster). Debajo, los <b>10 inmuebles más comparables</b> dentro de 3 km, ordenados por similitud.</p>

    <div class="predictor-block">
      <div class="pred-steps">
        <article class="pred-step pred-step-zone">
          <header><span class="pred-num">1</span><h3>Zona — click en el mapa o ingresá las coordenadas</h3></header>
          <div class="pred-zone-grid">
            <div class="pred-mini-map-wrap">
              <div id="pred-mini-map"></div>
              <p class="pred-mini-hint">Click en cualquier punto del mapa para fijar la zona del inmueble.</p>
            </div>
            <div class="pred-inputs">
              <label>Latitud<input type="number" id="pred-lat" step="0.0001" value="19.4326"/></label>
              <label>Longitud<input type="number" id="pred-lon" step="0.0001" value="-99.1332"/></label>
              <div class="pred-btn-row">
                <button type="button" id="pred-from-map" class="pred-secondary">↑ Usar centro de §05</button>
                <button type="button" id="pred-reset-map" class="pred-secondary">Centro de CDMX</button>
              </div>
            </div>
          </div>
        </article>

        <article class="pred-step">
          <header><span class="pred-num">2</span><h3>Diseño + submercado</h3></header>
          <div class="pred-inputs grid">
            <label>Sup. construida (m²)<input type="number" id="pred-sup" placeholder="ej. 85" min="20" max="600" value="85"/></label>
            <label>Recámaras<input type="number" id="pred-rec" placeholder="ej. 2" min="0" max="8" value="2"/></label>
            <label>Baños<input type="number" id="pred-ban" placeholder="ej. 2" min="0" max="8" value="2"/></label>
            <label>Antigüedad (años)<input type="number" id="pred-edad" placeholder="ej. 10" min="0" max="120" value="10"/></label>
          </div>
          <label class="cluster-select-lbl">Cluster / submercado del inmueble
            <select id="pred-cluster">{cluster_opts}</select>
          </label>
          <p class="pred-hint">Por ahora se elige a mano; en una iteración futura lo inferimos automáticamente o lo sacamos del set de features.</p>
        </article>

        <article class="pred-step">
          <header><span class="pred-num">3</span><h3>Predecir</h3></header>
          <div class="pred-inputs">
            <button type="button" id="pred-go" class="pred-primary">Estimar precio</button>
            <p class="pred-hint">Regresión local en tu navegador — los coeficientes ya vienen entrenados.</p>
          </div>
        </article>
      </div>

      <div id="pred-result" class="pred-result">
        <p class="empty">Llena los campos, elige el submercado y dale <b>Estimar precio</b>.</p>
      </div>
    </div>
    """


def render_map_section(map_data: dict, cluster: "ClusterResult | None",
                        partial_year: int,
                        cluster_data: dict,
                        alcaldia_data: dict,
                        years: list[int],
                        clases: list[int],
                        regression: dict | None = None) -> str:
    """HTML del bloque §05 explorador. Controles + filtros + mapa + panel +
    plusvalía-all-clusters + legenda. El JS hace toda la lógica dinámica."""
    if cluster is None or not map_data["points"]:
        return (
            '<h2 class="sec"><span class="num-tag">§05</span>Explorador <em>territorial</em></h2>'
            '<div class="sec-rule"></div>'
            '<p class="empty">El explorador requiere clustering activo y puntos con lat/lon válidos.</p>'
        )

    payload = json.dumps({
        "points": map_data["points"],
        "munis":  map_data["munis"],
        "names":  {str(k): v for k, v in cluster.cluster_names.items()},
        "palette": CLUSTER_PALETTE,
        "partialYear": partial_year,
        "clusterData": {str(k): v for k, v in (cluster_data or {}).items()},
        "alcaldiaData": alcaldia_data or {},
        "years":  years,
        "clases": clases,
        "regression": regression or {},
    }, separators=(",", ":"), ensure_ascii=False, default=str).replace("</", "<\\/")

    js = MAP_JS_TEMPLATE.replace("__MAP_DATA__", payload)

    # Filtros UI
    year_opts = '<option value="all">Todos</option>' + "".join(
        f'<option value="{y}">{y}</option>' for y in years
    )
    cluster_opts = '<option value="">— selecciona —</option>' + "".join(
        f'<option value="{cid}">C{cid} · {html.escape(name)}</option>'
        for cid, name in sorted(cluster.cluster_names.items())
    )
    clase_opts = '<option value="">— selecciona —</option>' + "".join(
        f'<option value="{c}">Clase {c}</option>' for c in clases
    )
    rec_chips = "".join(
        f'<button type="button" class="chip" data-val="{i}">{i}{"+" if i==5 else ""}</button>'
        for i in range(0, 6)
    )
    ban_chips = "".join(
        f'<button type="button" class="chip" data-val="{i}">{i}{"+" if i==5 else ""}</button>'
        for i in range(0, 6)
    )

    body = f"""
    <h2 class="sec"><span class="num-tag">§05</span>Explorador <em>territorial</em></h2>
    <div class="sec-rule"></div>
    <p class="sec-purpose">Para validar hipótesis de zona y producto: ¿qué tipologías se transaccionan dónde, a qué precio, y cuál es el margen estimado del desarrollador después de descontar terreno y costo físico? Combina filtros (tamaño, recámaras, año, trimestre, clase) con el buffer geográfico para aislar comparables relevantes a tu lote.</p>

    <div class="map-block">
      <div class="map-layout">

        <!-- Sidebar de filtros -->
        <aside class="filter-panel">
          <div class="fp-head">
            <h4>Filtros</h4>
            <button id="f-reset" type="button">Reset</button>
          </div>

          <div class="f-row tipo-mode">
            <label class="fp-label">Tipología</label>
            <div class="radio-row">
              <label><input type="radio" name="f-tipo-mode" value="none" checked/> Sin filtro</label>
              <label><input type="radio" name="f-tipo-mode" value="cluster"/> Cluster</label>
              <label><input type="radio" name="f-tipo-mode" value="clase"/> Clase</label>
            </div>
            <div class="f-sub-row" id="f-cluster-wrap" style="display:none">
              <select id="f-cluster">{cluster_opts}</select>
            </div>
            <div class="f-sub-row" id="f-clase-wrap" style="display:none">
              <select id="f-clase">{clase_opts}</select>
            </div>
          </div>

          <div class="f-row">
            <label class="fp-label">Año</label>
            <select id="f-year">{year_opts}</select>
          </div>

          <div class="f-row">
            <label class="fp-label">Trimestres</label>
            <div class="check-row" id="f-quarters">
              <label><input type="checkbox" value="1" checked/> Q1</label>
              <label><input type="checkbox" value="2" checked/> Q2</label>
              <label><input type="checkbox" value="3" checked/> Q3</label>
              <label><input type="checkbox" value="4" checked/> Q4</label>
            </div>
          </div>

          <div class="f-row">
            <label class="fp-label">Precio · valor concluido (MXN)</label>
            <div class="dual-input">
              <input type="number" id="f-price-min" placeholder="mín"/>
              <input type="number" id="f-price-max" placeholder="máx"/>
            </div>
          </div>

          <div class="f-row">
            <label class="fp-label">Recámaras</label>
            <div class="chip-row" id="f-recamaras">{rec_chips}</div>
          </div>
          <div class="f-row">
            <label class="fp-label">Baños</label>
            <div class="chip-row" id="f-banos">{ban_chips}</div>
          </div>

          <div class="f-row">
            <label class="fp-label">Sup. construida (m²)</label>
            <div class="dual-input">
              <input type="number" id="f-sup-min" placeholder="mín"/>
              <input type="number" id="f-sup-max" placeholder="máx"/>
            </div>
          </div>
          <div class="f-row">
            <label class="fp-label">Sup. terreno (m²)</label>
            <div class="dual-input">
              <input type="number" id="f-ter-min" placeholder="mín"/>
              <input type="number" id="f-ter-max" placeholder="máx"/>
            </div>
          </div>
          <div class="f-row">
            <label class="fp-label">$/m² terreno (MXN)</label>
            <div class="dual-input">
              <input type="number" id="f-tm-min" placeholder="mín"/>
              <input type="number" id="f-tm-max" placeholder="máx"/>
            </div>
          </div>
          <div class="f-row">
            <label class="fp-label">Antigüedad (meses)</label>
            <div class="dual-input">
              <input type="number" id="f-edad-min" placeholder="mín"/>
              <input type="number" id="f-edad-max" placeholder="máx"/>
            </div>
          </div>
          <div class="f-row latlon-row">
            <label class="fp-label">Centro del buffer</label>
            <div class="dual-input small">
              <input type="number" id="map-lat" step="0.0001" value="19.4326" title="latitud"/>
              <input type="number" id="map-lon" step="0.0001" value="-99.1332" title="longitud"/>
            </div>
            <div class="f-sub-row buffer-row">
              <span class="buf-lbl">Buffer · <b id="map-buf-val">1,000</b> m</span>
              <input type="range" id="map-buf" min="200" max="5000" step="100" value="1000"/>
            </div>
          </div>

          <div class="f-row">
            <label class="fp-label">Capas del mapa</label>
            <label class="toggle-row"><input type="checkbox" id="f-sat"/> <span>Imagen satelital</span></label>
            <label class="toggle-row"><input type="checkbox" id="f-alc" checked/> <span>Mostrar municipios</span></label>
            <label class="toggle-row"><input type="checkbox" id="f-heat"/> <span>Heatmap por precio</span></label>
            <label class="toggle-row"><input type="checkbox" id="f-markers" checked/> <span>Puntos por cluster</span></label>
          </div>
        </aside>

        <!-- Mapa -->
        <div class="map-area">
          <div id="map-canvas"></div>
        </div>

        <!-- Panel de resumen -->
        <aside class="map-panel">
          <div class="map-panel-head">
            <h4>Resumen · zona activa</h4>
            <div class="panel-year">
              <span class="py-lbl">Año</span>
              <select id="m-panel-year"></select>
            </div>
          </div>
          <div class="map-kpi">
            <span class="lbl">avalúos (filtros)</span>
            <span class="val" id="m-n-all">—</span>
          </div>
          <div class="map-kpi">
            <span class="lbl">avalúos en buffer</span>
            <span class="val" id="m-n">—</span>
          </div>
          <div class="map-kpi">
            <span class="lbl">mediana valor concluido</span>
            <span class="val" id="m-med-v">—</span>
          </div>
          <div class="map-kpi">
            <span class="lbl">mediana $/m² construido</span>
            <span class="val" id="m-med-m">—</span>
          </div>
          <div class="map-dominant">
            <span class="lbl">tipología predominante (click → modal)</span>
            <div class="dom-row">
              <span class="cluster-tag clickable" id="m-dom-tag" style="background:#000">—</span>
            </div>
            <span class="val small" id="m-dom-name">—</span>
            <span class="sub" id="m-dom-share"></span>
          </div>

          <h5>Valor de la Construcción</h5>
          <p class="hint valcon-hint">año seleccionado · buffer</p>
          <div class="dual-input">
            <input type="number" id="m-valcon-min" placeholder="mín MXN"/>
            <input type="number" id="m-valcon-max" placeholder="máx MXN"/>
          </div>
          <div class="m-valcon-range">
            <span class="lbl">rango observado</span>
            <span class="vrow"><span class="vtag">mín</span><span class="val" id="m-valcon-min-obs">—</span></span>
            <span class="vrow"><span class="vtag">máx</span><span class="val" id="m-valcon-max-obs">—</span></span>
            <span class="vrow nbar"><span class="vtag">n</span><span class="val" id="m-valcon-n-obs">—</span></span>
          </div>

          <div class="map-kpi accent">
            <span class="lbl">Margen del Desarrollador</span>
            <span class="val" id="m-margen-pct">—</span>
            <span class="sub" id="m-margen-detail">— vs —</span>
            <span class="sub margin-formula">vivienda − terreno − construcción</span>
          </div>

          <div class="map-export-row">
            <button id="m-export-png" type="button" class="export-png">Exportar Vista PNG</button>
          </div>
        </aside>

      </div>

      <!-- Plusvalía — TODOS los clusters de la zona -->
      <section class="plus-block">
        <header>
          <h4>Plusvalía nominal m²/SV · TODOS los clusters · zona activa</h4>
          <p>Mediana de m² vendible por cluster y año, referida al primer año observado en la zona. Recalculado en vivo según los filtros y el buffer. <span class="partial-tag">*</span> año parcial.</p>
        </header>
        <div class="plus-chart-wrap">
          <svg id="m-plus-svg" viewBox="0 0 880 240" preserveAspectRatio="xMidYMid meet"></svg>
        </div>
        <ul class="ts-legend" id="m-plus-legend"></ul>
        <div class="ts-table-wrap">
          <table class="ts-table"><tbody id="m-plus-tbody"></tbody></table>
        </div>
      </section>

      <!-- Legenda global de clusters (click para abrir modal) -->
      <div class="map-legend">
        <h5>Tipologías K-Means · click para ver histogramas y stats</h5>
        <ul id="m-legend"></ul>
      </div>
    </div>

    <script>{js}</script>
    """
    return body


# ---------------------------------------------------------------------------
# Serie temporal de plusvalía
# ---------------------------------------------------------------------------

def timeseries_plusvalia(df: pd.DataFrame, cluster_labels: pd.Series | None
                        ) -> dict[int | str, list[dict]]:
    """
    Para cada cluster (o 'all' si no hay clusters), devuelve una lista de
    {ano, n, mediana_m2_sv, idx_base_2019, yoy_pct, partial}.
    """
    work = df.copy()
    if cluster_labels is not None:
        work["_cluster"] = cluster_labels
        work = work.dropna(subset=["_cluster"])
        work["_cluster"] = work["_cluster"].astype(int)
        groups = work.groupby("_cluster")
    else:
        work["_cluster"] = "all"
        groups = work.groupby("_cluster")

    out: dict[int | str, list[dict]] = {}
    for cluster_id, sub in groups:
        yearly = (sub.groupby("ano")
                     .agg(n=("id_avaluo", "count"),
                          med_m2=("m2_sv", "median"))
                     .reset_index()
                     .sort_values("ano"))
        # Base: mediana 2019 (si no hay 2019 con n suficiente, primer año disponible)
        valid = yearly[yearly["n"] >= N_MIN_TIMESERIES]
        if len(valid) == 0:
            out[cluster_id] = []
            continue
        base_row = valid.iloc[0]
        base_year = int(base_row["ano"])
        base_val = float(base_row["med_m2"])

        records = []
        prev_val = None
        for _, r in yearly.iterrows():
            ano = int(r["ano"])
            n = int(r["n"])
            med = float(r["med_m2"]) if pd.notna(r["med_m2"]) else None
            insufficient = n < N_MIN_TIMESERIES
            idx = ((med / base_val) - 1) * 100 if (med and not insufficient) else None
            yoy = ((med / prev_val) - 1) * 100 if (med and prev_val and not insufficient) else None
            records.append(dict(
                ano=ano, n=n, mediana_m2_sv=med, idx_base=idx, yoy=yoy,
                base_year=base_year, partial=(ano == PARTIAL_YEAR),
                insufficient=insufficient,
            ))
            if med and not insufficient:
                prev_val = med
        out[cluster_id] = records
    return out


# ---------------------------------------------------------------------------
# Render HTML
# ---------------------------------------------------------------------------

def render_html(
    stats: DescriptiveStats,
    cluster: ClusterResult | None,
    plusvalia: dict[Any, list[dict]],
    out_path: str,
    generated: str,
    logo_svg: str,
    map_section: str,
    predictor_section: str,
    phys_map_section: str = "",
    absorcion_section: str = "",
    muni_compare_data: dict | None = None,
) -> None:

    # ---- Year picker (años disponibles, default "Todos") ----
    years_avail = sorted(stats.by_year_stats.keys()) if getattr(stats, "by_year_stats", None) else []
    year_btns = ['<button class="yp on" data-year="all" type="button">Todos</button>']
    for y in years_avail:
        partial = (int(y) == PARTIAL_YEAR)
        year_btns.append(
            f'<button class="yp{" partial" if partial else ""}" data-year="{y}" type="button">{y}{"*" if partial else ""}</button>'
        )
    year_picker_block = (
        '<div class="year-picker" id="year-picker">'
        + '<span class="yp-lbl">Año</span>'
        + "".join(year_btns)
        + '<span class="yp-hint">Click en un KPI para abrir su histograma del año seleccionado.</span>'
        + '</div>'
    )

    # ---- KPI block (3 cajas, clickeables) ----
    kpi_block = f"""
    <section class="kpis kpis-3">
      <button type="button" class="kpi accent clickable" data-metric="valor">
        <span class="label">Valor mediano · vivienda</span>
        <span class="value" id="kpi-valor">{fmt_mxn(stats.median_valor)}</span>
        <span class="sub" id="kpi-valor-sub">p25 {fmt_mxn_short(stats.p25_valor)} · p75 {fmt_mxn_short(stats.p75_valor)}</span>
      </button>
      <button type="button" class="kpi clickable" data-metric="m2_sv">
        <span class="label">Costo por m² · construido</span>
        <span class="value" id="kpi-m2sv">{fmt_mxn(stats.median_m2_sv)}</span>
        <span class="sub" id="kpi-m2sv-sub">mediana m² vendible</span>
      </button>
      <button type="button" class="kpi clickable" data-metric="terreno">
        <span class="label">Costo por m² · terreno</span>
        <span class="value" id="kpi-terr">{fmt_mxn(stats.median_terreno_m2)}</span>
        <span class="sub" id="kpi-terr-sub">mediana suelo</span>
      </button>
    </section>
    """

    top_munis_html = render_top_munis(stats.top_munis)

    # Bloque clustering
    if cluster is not None:
        cluster_html = render_cluster_section(cluster)
        plusvalia_html = render_plusvalia_section(plusvalia, cluster)
    else:
        cluster_html = '<p class="empty">Clustering omitido por flag --skip-cluster.</p>'
        plusvalia_html = render_plusvalia_section(plusvalia, None)

    # ---- Payload global para JS: stats por año + histogramas + datos de comparación de municipios ----
    cdmx_payload = {
        "byYearStats": getattr(stats, "by_year_stats", {}) or {},
        "byYearHists": getattr(stats, "by_year_hists", {}) or {},
        "globalStats": {
            "valor":    {"median": stats.median_valor,   "p25": stats.p25_valor, "p75": stats.p75_valor},
            "m2_sv":    {"median": stats.median_m2_sv},
            "terreno":  {"median": stats.median_terreno_m2},
        },
        "globalHists": getattr(stats, "global_hists", {}) or {},
        "muniCompare": muni_compare_data or {},
    }
    cdmx_payload_json = json.dumps(cdmx_payload, separators=(",", ":"),
                                    ensure_ascii=False, default=str).replace("</", "<\\/")
    cdmx_payload_script = f'<script>window.__CDMX = {cdmx_payload_json};</script>'

    doc = HTML_TEMPLATE.format(
        generated=generated,
        logo_svg=logo_svg,
        logo_blackprint_dark=LOGO_BLACKPRINT_DARK,
        logo_blackprint_light=LOGO_BLACKPRINT_LIGHT,
        logo_orange=LOGO_ORANGE_MARK,
        year_picker_block=year_picker_block,
        kpi_block=kpi_block,
        top_munis_html=top_munis_html,
        cluster_html=cluster_html,
        plusvalia_html=plusvalia_html,
        phys_map_section=phys_map_section,
        map_section=map_section,
        predictor_section=predictor_section,
        absorcion_section=absorcion_section,
        cdmx_payload_script=cdmx_payload_script,
        global_js=GLOBAL_JS,
    )
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(doc)
    print(f"  ✓ HTML escrito en {out_path}")


def bars_horizontal(rows: list[tuple[str, int, str]], unit="") -> str:
    if not rows: return ""
    max_v = max(v for _, v, _ in rows) or 1
    out = ['<ul class="bars-h">']
    for label, v, extra in rows:
        pct = 100 * v / max_v
        out.append(f"""
          <li>
            <div class="bar-h-label">{html.escape(label)}</div>
            <div class="bar-h-track">
              <div class="bar-h-fill" style="width:{pct:.2f}%"></div>
            </div>
            <div class="bar-h-count">{fmt_int(v)}</div>
            <div class="bar-h-extra">{extra}</div>
          </li>
        """)
    out.append("</ul>")
    return "".join(out)


def bars_vertical(rows: list[tuple[str, int]]) -> str:
    if not rows: return ""
    max_v = max(v for _, v in rows) or 1
    out = ['<div class="bars-v">']
    for label, v in rows:
        pct = 100 * v / max_v
        out.append(f"""
          <div class="bar-v-col">
            <div class="bar-v-track">
              <div class="bar-v-fill" style="height:{pct:.2f}%"
                   title="{html.escape(label)}: {fmt_int(v)}"></div>
            </div>
            <div class="bar-v-label">{html.escape(label)}</div>
            <div class="bar-v-count">{fmt_int(v)}</div>
          </div>
        """)
    out.append("</div>")
    return "".join(out)


def histogram_svg(bins: list[tuple[float, float, int]], label_fmt=str) -> str:
    if not bins: return ""
    W, H, PAD = 580, 140, 8
    max_c = max(c for _, _, c in bins) or 1
    bw = (W - 2 * PAD) / len(bins)
    bars = []
    for i, (lo, hi, c) in enumerate(bins):
        h = (H - 28) * c / max_c
        x = PAD + i * bw
        y = H - 18 - h
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bw - 0.8:.2f}" '
            f'height="{h:.2f}" class="hbar">'
            f'<title>[{label_fmt(lo)}, {label_fmt(hi)}]: {fmt_int(c)}</title></rect>'
        )
    lo0 = bins[0][0]; hi_last = bins[-1][1]
    return f"""
    <svg class="hist" viewBox="0 0 {W} {H}" preserveAspectRatio="none">
      {''.join(bars)}
      <line x1="{PAD}" y1="{H-18}" x2="{W-PAD}" y2="{H-18}" class="hax"/>
      <text x="{PAD}" y="{H-4}" class="hlbl">{label_fmt(lo0)}</text>
      <text x="{W-PAD}" y="{H-4}" class="hlbl" text-anchor="end">{label_fmt(hi_last)}</text>
    </svg>
    """


def render_top_munis(top: pd.DataFrame) -> str:
    if top.empty: return ""
    rows = []
    for _, r in top.iterrows():
        cve = str(r.cve_mun).zfill(3)
        rows.append(f"""
          <tr class="muni-row" data-cve="{cve}" data-name="{html.escape(str(r.nom_mun))}" tabindex="0">
            <td class="muni">{html.escape(str(r.nom_mun))}</td>
            <td class="num">{fmt_int(r.n)}</td>
            <td class="num strong">{fmt_mxn_short(r.med_valor)}</td>
            <td class="num">{fmt_mxn_short(r.med_m2)}</td>
            <td class="num arrow">›</td>
          </tr>
        """)
    return f"""
    <table class="muni-table clickable">
      <thead>
        <tr>
          <th>Municipio</th><th class="num">n avalúos</th>
          <th class="num">Mediana vivienda</th><th class="num">Mediana $/m²</th>
          <th class="num"></th>
        </tr>
      </thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    <p class="table-hint">Click en una fila para abrir el histograma del municipio con mediana y promedio marcados.</p>
    """


def render_cluster_section(c: ClusterResult) -> str:
    # Cards de cluster (uno por cluster, ordenado por precio)
    cards = []
    for _, r in c.profile.iterrows():
        cid = int(r["_cluster"])
        name = c.cluster_names.get(cid, f"Cluster {cid}")
        cards.append(f"""
          <article class="cluster-card clickable" data-cluster="{cid}" tabindex="0">
            <header>
              <span class="cluster-tag">C{cid}</span>
              <h3>{html.escape(name)}</h3>
              <span class="arrow">›</span>
            </header>
            <div class="cluster-stats">
              <div><span class="cl-lbl">mediana valor concluido</span><span class="cl-val">{fmt_mxn_short(r.med_valor)}</span></div>
              <div><span class="cl-lbl">$/m² construido</span><span class="cl-val">{fmt_mxn_short(r.med_m2_sv)}</span></div>
              <div><span class="cl-lbl">sup. construida</span><span class="cl-val">{fmt_num(r.med_sup_const, 0)} m²</span></div>
              <div><span class="cl-lbl">sup. terreno</span><span class="cl-val">{fmt_num(r.med_sup_terr, 0)} m²</span></div>
              <div><span class="cl-lbl">recámaras / baños</span><span class="cl-val">{int(r.med_recamaras)} / {int(r.med_banos)}</span></div>
              <div><span class="cl-lbl">edad mediana</span><span class="cl-val">{int(r.med_edad_meses/12)} años</span></div>
              <div><span class="cl-lbl">estacionamientos</span><span class="cl-val">{int(r.med_estac)}</span></div>
            </div>
          </article>
        """)

    return f"""
    <div class="cluster-cards">{''.join(cards)}</div>
    <p class="table-hint">Click en una tarjeta para ver histogramas detallados con filtro por año, stats y composición por municipio / NSE. Dentro del modal, usa <b>+ Comparar</b> en cualquier histograma para superponer otros clusters con control de opacidad.</p>
    """


def render_plusvalia_section(plusvalia: dict[Any, list[dict]],
                              cluster: ClusterResult | None) -> str:
    if not plusvalia:
        return '<p class="empty">Serie temporal no disponible.</p>'

    # Construye una serie por cluster: cada serie es una línea SVG.
    W, H, PADL, PADR, PADT, PADB = 880, 320, 56, 24, 16, 38

    # universo de años
    all_years = sorted({rec["ano"] for recs in plusvalia.values() for rec in recs})
    if not all_years:
        return '<p class="empty">Sin años suficientes.</p>'
    y_min, y_max = min(all_years), max(all_years)
    if y_max == y_min: y_max = y_min + 1

    # rango y de plusvalía (idx_base) — agregamos un colchón
    all_idx = [r["idx_base"] for recs in plusvalia.values()
                for r in recs if r["idx_base"] is not None]
    if not all_idx:
        return '<p class="empty">No hay suficientes datos para series temporales.</p>'
    v_min = min(all_idx + [0]) - 5
    v_max = max(all_idx) + 5

    def x_of(year): return PADL + (year - y_min) / (y_max - y_min) * (W - PADL - PADR)
    def y_of(v):    return PADT + (1 - (v - v_min) / (v_max - v_min)) * (H - PADT - PADB)

    palette = ["#c9421f", "#1f6c5c", "#b88a1f", "#2a3d6e", "#7b3b5e", "#4d6b1d", "#a44b1a"]

    # ejes
    axis_lines = []
    # eje y: gridlines en múltiplos de 10
    step = 10 if v_max - v_min <= 80 else 20
    g = int(np.floor(v_min/step)*step)
    while g <= v_max:
        y = y_of(g)
        axis_lines.append(
            f'<line x1="{PADL}" y1="{y:.2f}" x2="{W-PADR}" y2="{y:.2f}" class="grid"/>'
            f'<text x="{PADL-6}" y="{y+3:.2f}" class="axlbl" text-anchor="end">'
            f'{g:+d}%</text>'
        )
        g += step
    # baseline en 0
    y0 = y_of(0)
    axis_lines.append(f'<line x1="{PADL}" y1="{y0:.2f}" x2="{W-PADR}" y2="{y0:.2f}" class="grid base"/>')

    # eje x: años
    for year in all_years:
        x = x_of(year)
        partial = (year == PARTIAL_YEAR)
        cls = "axlbl partial" if partial else "axlbl"
        axis_lines.append(
            f'<text x="{x:.2f}" y="{H-PADB+18}" class="{cls}" text-anchor="middle">{year}{"*" if partial else ""}</text>'
        )

    # líneas por cluster
    lines_svg = []
    legend = []
    sorted_clusters = sorted(plusvalia.keys(), key=lambda x: (isinstance(x, str), x))

    for i, cid in enumerate(sorted_clusters):
        recs = plusvalia[cid]
        if not recs: continue
        color = palette[i % len(palette)]
        # nombre
        if cluster and isinstance(cid, (int, np.integer)):
            name = f"C{cid} · {cluster.cluster_names.get(int(cid), '')}"
        else:
            name = "Toda la muestra"
        # polyline
        pts = []
        circles = []
        for r in recs:
            if r["idx_base"] is None:  # n insuficiente o base
                continue
            x = x_of(r["ano"])
            y = y_of(r["idx_base"])
            pts.append(f"{x:.2f},{y:.2f}")
            partial = r["partial"]
            dot_class = "dot partial" if partial else "dot"
            tooltip = (f'{name}\\nAño {r["ano"]}{" (parcial)" if partial else ""}\\n'
                       f'n={r["n"]:,}\\nmediana m² ${r["mediana_m2_sv"]:,.0f}\\n'
                       f'vs {r["base_year"]}: {r["idx_base"]:+.1f}%')
            circles.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4.5" class="{dot_class}" '
                f'style="fill:{color}"><title>{tooltip}</title></circle>'
            )
        if pts:
            lines_svg.append(
                f'<polyline points="{" ".join(pts)}" class="series" '
                f'style="stroke:{color}"/>'
            )
            lines_svg.append("".join(circles))
            # legend item
            last = recs[-1]
            last_val = next((r["idx_base"] for r in reversed(recs) if r["idx_base"] is not None), None)
            base_year = recs[0].get("base_year", y_min)
            legend.append(f"""
              <li>
                <span class="leg-swatch" style="background:{color}"></span>
                <span class="leg-name">{html.escape(name)}</span>
                <span class="leg-val">{fmt_pct(last_val) if last_val is not None else "—"}
                  <span class="leg-sub">vs {base_year}</span>
                </span>
              </li>
            """)

    # tabla compacta YoY
    table_rows = []
    header_years = "".join(
        f'<th class="num{" partial" if y == PARTIAL_YEAR else ""}">{y}{"*" if y == PARTIAL_YEAR else ""}</th>'
        for y in all_years
    )
    for cid in sorted_clusters:
        recs_by_year = {r["ano"]: r for r in plusvalia[cid]}
        if cluster and isinstance(cid, (int, np.integer)):
            name = f"C{cid}"
        else:
            name = "Total"
        cells = []
        for y in all_years:
            r = recs_by_year.get(y)
            if r is None or r["idx_base"] is None:
                cells.append('<td class="num dim">—</td>')
            else:
                cls = "num"
                if r["partial"]: cls += " partial"
                if r["idx_base"] > 0: cls += " pos"
                elif r["idx_base"] < 0: cls += " neg"
                cells.append(f'<td class="{cls}">{r["idx_base"]:+.1f}%</td>')
        table_rows.append(f'<tr><th>{name}</th>{"".join(cells)}</tr>')

    chart_svg = f"""
    <svg class="ts-chart" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet">
      {''.join(axis_lines)}
      {''.join(lines_svg)}
    </svg>
    """

    return f"""
    <div class="ts-block">
      <div class="ts-chart-wrap">{chart_svg}</div>
      <ul class="ts-legend">{''.join(legend)}</ul>
      <div class="ts-table-wrap">
        <table class="ts-table">
          <thead><tr><th>Cluster</th>{header_years}</tr></thead>
          <tbody>{''.join(table_rows)}</tbody>
        </table>
      </div>
      <p class="ts-note">
        Plusvalía nominal en MXN/m² vendible, referida a la mediana del primer año con n≥{N_MIN_TIMESERIES}.
        <b>Sin deflactar por INPC</b> — incluye inflación general. <span class="partial-tag">*</span> año parcial (captura en curso).
      </p>
    </div>
    """


# ---------------------------------------------------------------------------
# HTML template (self-contained CSS, Google Fonts CDN, sin JS)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<title>Módulo de Vivienda · CDMX</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,400;9..144,600;9..144,800;9..144,900&family=JetBrains+Mono:wght@400;500;600&family=Inter+Tight:wght@400;500;600&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script src="https://unpkg.com/leaflet.heat@0.2.0/dist/leaflet-heat.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
<style>
  :root {{
    --bg: #ecebe7;             /* gris cálido — el "papel" de la página */
    --paper: #f5f3ee;          /* gris muy claro para tarjetas */
    --paper-deep: #e3e1db;     /* gris medio claro para hovers */
    --ink: #1a1a1a;            /* negro BlackPrint */
    --ink-soft: #6b6b6b;       /* gris medio neutro */
    --line: #1a1a1a;
    --accent: #ed6b1f;         /* orange BlackPrint (Logo_Orange.png) */
    --accent-deep: #c4501a;
    --teal: #1f6c5c;
    --gold: #b88a1f;
    --neg: #c4501a;
    --pos: #1f6c5c;
    --rose: #d49ba2;           /* rosa polvo — usar con cuentagotas */
    --bp-dark: #111111;        /* fondo oscuro del hero/footer */
    --bp-dark-soft: #1f1f1f;
    --bp-gray: #8a8985;        /* gris secundario */
    --on-dark: #f4f2ec;        /* texto sobre fondo oscuro */
    --on-dark-soft: rgba(244, 242, 236, 0.62);
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: var(--bg); color: var(--ink);
    font-family: 'Inter Tight', system-ui, sans-serif; font-size: 14.5px; line-height: 1.55;
    -webkit-font-smoothing: antialiased; }}
  body {{
    background-image:
      radial-gradient(rgba(26,26,26,0.035) 1px, transparent 1px),
      radial-gradient(rgba(237,107,31,0.05) 1px, transparent 1px);
    background-size: 6px 6px, 19px 19px;
    background-position: 0 0, 3px 3px;
  }}
  .mono {{ font-family: 'JetBrains Mono', ui-monospace, monospace; }}
  .serif {{ font-family: 'Fraunces', Georgia, serif; }}
  .container {{ max-width: 1240px; margin: 0 auto; padding: 56px 32px 96px; }}

  /* HERO BlackPrint — fondo negro, wordmark blanco, mark naranja */
  header.hero.hero-dark {{
    background: var(--bp-dark); color: var(--on-dark);
    padding: 32px 36px 36px; border: 1px solid var(--ink);
    margin-bottom: 48px;
    box-shadow: 8px 8px 0 var(--bp-gray);
    position: relative; overflow: hidden;
  }}
  .hero-bp-mark {{
    display: flex; align-items: center; justify-content: space-between;
    padding-bottom: 22px; border-bottom: 1px solid rgba(244, 242, 236, 0.18);
    margin-bottom: 24px;
  }}
  .hero-wordmark {{
    height: 28px; width: auto; display: block;
    filter: brightness(0) invert(1);   /* asegura blanco aunque el PNG no lo sea */
  }}
  .hero-orange-mark {{ height: 40px; width: auto; display: block; }}
  .hero-main {{
    display: grid; grid-template-columns: 1.4fr 1fr; gap: 48px; align-items: end;
  }}
  header.hero .marca {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    letter-spacing: 0.24em; text-transform: uppercase; color: var(--on-dark-soft);
    margin-bottom: 16px;
    display: flex; gap: 12px; align-items: center;
  }}
  header.hero .marca::before {{
    content: ""; width: 10px; height: 10px; background: var(--accent);
    border-radius: 50%; display: inline-block;
  }}
  header.hero h1 {{
    font-family: 'Fraunces', serif; font-weight: 900; font-style: normal;
    font-size: clamp(44px, 6vw, 84px); line-height: 0.94; letter-spacing: -0.03em;
    margin: 0 0 14px; color: var(--on-dark);
    font-variation-settings: "opsz" 144;
  }}
  header.hero h1 em {{
    font-style: italic; color: var(--accent); font-weight: 400;
    font-variation-settings: "opsz" 144;
  }}
  header.hero .sub {{
    font-family: 'Fraunces', serif; font-size: 16.5px; color: var(--on-dark-soft);
    font-style: italic; max-width: 44ch; font-weight: 300; margin: 0;
    line-height: 1.5;
  }}
  header.hero .meta-block {{
    border-left: 1px solid rgba(244, 242, 236, 0.22); padding-left: 24px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: var(--on-dark-soft); line-height: 1.85;
  }}
  header.hero .meta-block dl {{ margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 4px 14px;}}
  header.hero .meta-block dt {{ text-transform: uppercase; letter-spacing: 0.14em; color: var(--on-dark-soft);}}
  header.hero .meta-block dd {{ margin: 0; color: var(--on-dark); }}

  /* KPI grid */
  section.kpis {{
    display: grid; grid-template-columns: 1.3fr 1fr 1fr 1.1fr; gap: 0;
    border: 1px solid var(--line); margin-bottom: 56px;
    background: var(--paper);
  }}
  .kpi {{
    padding: 22px 24px; border-right: 1px solid var(--line);
    display: flex; flex-direction: column; gap: 6px;
    position: relative;
  }}
  .kpi:last-child {{ border-right: none; }}
  .kpi.accent {{ background: var(--bp-dark); color: var(--on-dark); }}
  .kpi.accent .label, .kpi.accent .sub {{ color: var(--on-dark-soft);}}
  .kpi.accent .value {{ color: var(--on-dark); }}
  .kpi.accent::before {{
    content: ""; position: absolute; top: 12px; right: 12px;
    width: 8px; height: 8px; background: var(--accent); border-radius: 50%;
  }}
  .kpi .label {{
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    letter-spacing: 0.16em; text-transform: uppercase; color: var(--ink-soft);
  }}
  .kpi .value {{
    font-family: 'Fraunces', serif; font-weight: 700; font-size: 38px;
    letter-spacing: -0.02em; line-height: 1; margin-top: 2px;
    font-variation-settings: "opsz" 96;
  }}
  .kpi .sub {{ font-family: 'JetBrains Mono', monospace; font-size: 11.5px; color: var(--ink-soft); }}

  /* Section titles */
  h2.sec {{
    font-family: 'Fraunces', serif; font-weight: 600; font-style: italic;
    font-size: 32px; letter-spacing: -0.015em; margin: 64px 0 4px;
    font-variation-settings: "opsz" 72;
  }}
  h2.sec .num-tag {{
    font-family: 'JetBrains Mono', monospace; font-style: normal;
    font-size: 13px; letter-spacing: 0.16em; color: var(--accent);
    margin-right: 12px; vertical-align: middle; font-weight: 500;
  }}
  .sec-rule {{
    height: 1px; background: var(--line); margin-bottom: 24px;
    position: relative;
  }}
  .sec-rule::after {{
    content: ""; position: absolute; left: 0; top: -1px;
    width: 60px; height: 3px; background: var(--accent);
  }}
  .sec-intro {{
    font-family: 'Fraunces', serif; font-style: italic; font-size: 16.5px;
    color: var(--ink-soft); max-width: 70ch; margin: 0 0 28px; font-weight: 300;
  }}
  .sec-purpose {{
    font-family: 'Inter Tight', sans-serif; font-size: 14px;
    color: var(--ink); max-width: 78ch; margin: 0 0 12px; line-height: 1.55;
    padding: 10px 14px; background: rgba(201, 66, 31, 0.05);
    border-left: 3px solid var(--accent); border-radius: 0 4px 4px 0;
  }}
  .sec-purpose b {{ color: var(--ink); }}
  .sec-bullets {{
    list-style: none; padding: 0; margin: 0 0 24px;
    display: flex; flex-direction: column; gap: 6px;
    max-width: 78ch;
  }}
  .sec-bullets li {{
    font-family: 'Inter Tight', sans-serif; font-size: 13px;
    color: var(--ink-soft); line-height: 1.5; padding-left: 18px;
    position: relative;
  }}
  .sec-bullets li::before {{
    content: "▸"; position: absolute; left: 0; top: 0;
    color: var(--accent); font-weight: 700;
  }}
  .sec-bullets li b {{ color: var(--ink); font-weight: 600; }}

  /* Grids de bloques */
  .row {{ display: grid; gap: 24px; margin-bottom: 28px;}}
  .row-2 {{ grid-template-columns: 1fr 1fr; }}
  .row-3 {{ grid-template-columns: 1.4fr 1fr 1fr; }}
  .row-3b {{ grid-template-columns: 1fr 1fr 1fr; }}

  .panel {{
    background: var(--paper); border: 1px solid var(--line);
    padding: 20px 22px;
    box-shadow: 5px 5px 0 var(--line);
  }}
  .panel h3 {{
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--ink-soft); margin: 0 0 16px;
    display: flex; align-items: center; gap: 8px;
  }}
  .panel h3::before {{
    content: ""; display: inline-block; width: 6px; height: 6px;
    background: var(--accent); transform: rotate(45deg);
  }}
  .panel .strong-stat {{
    font-family: 'Fraunces', serif; font-size: 28px; font-weight: 700;
    letter-spacing: -0.02em; line-height: 1; margin-bottom: 4px;
  }}

  /* bars horizontal */
  ul.bars-h {{ list-style: none; padding: 0; margin: 0; }}
  ul.bars-h li {{
    display: grid; grid-template-columns: 70px 1fr 70px 80px;
    align-items: center; gap: 12px; padding: 6px 0;
    border-bottom: 1px dotted #cdc6b3;
  }}
  ul.bars-h li:last-child {{ border-bottom: none; }}
  .bar-h-label {{ font-family: 'JetBrains Mono', monospace; font-size: 11.5px; font-weight: 500; }}
  .bar-h-track {{
    position: relative; background: #e0d8c3; height: 14px; border: 1px solid var(--line);
  }}
  .bar-h-fill {{
    position: absolute; left: 0; top: 0; bottom: 0; background: var(--accent);
    background-image: repeating-linear-gradient(45deg,
      transparent, transparent 4px,
      rgba(0,0,0,0.08) 4px, rgba(0,0,0,0.08) 5px);
  }}
  .bar-h-count {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; text-align: right; font-variant-numeric: tabular-nums; }}
  .bar-h-extra {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--ink-soft); text-align: right; }}

  /* bars vertical */
  .bars-v {{ display: flex; gap: 8px; align-items: flex-end; height: 180px; padding-top: 16px;}}
  .bar-v-col {{ flex: 1; display: flex; flex-direction: column; align-items: center; }}
  .bar-v-track {{ width: 100%; height: 130px; position: relative;
    border-bottom: 1px solid var(--line); display: flex; align-items: flex-end; }}
  .bar-v-fill {{ width: 100%; background: var(--teal);
    background-image: linear-gradient(0deg, var(--teal) 0%, #2b8a76 100%);
    transition: height 0.4s ease; min-height: 1px;}}
  .bar-v-label {{ font-family: 'JetBrains Mono', monospace; font-size: 11px; margin-top: 6px; font-weight: 500;}}
  .bar-v-count {{ font-family: 'JetBrains Mono', monospace; font-size: 9.5px; color: var(--ink-soft);}}

  /* histograms */
  svg.hist {{ width: 100%; height: 140px; display: block; }}
  svg.hist .hbar {{ fill: var(--accent); fill-opacity: 0.78; }}
  svg.hist .hbar:hover {{ fill-opacity: 1; fill: var(--accent-deep); }}
  svg.hist .hax {{ stroke: var(--line); stroke-width: 1;}}
  svg.hist .hlbl {{ font-family: 'JetBrains Mono', monospace; font-size: 9.5px; fill: var(--ink-soft);}}

  /* alcaldías */
  .muni-table {{ width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 12.5px;}}
  .muni-table th {{
    text-align: left; padding: 6px 10px;
    background: var(--paper-deep); border-bottom: 1px solid var(--line);
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.14em; font-weight: 600;
    color: var(--ink-soft);
  }}
  .muni-table td {{ padding: 6px 10px; border-bottom: 1px dotted #cdc6b3; }}
  .muni-table tr:hover td {{ background: #f0eadb; }}
  td.muni {{ font-family: 'Inter Tight', sans-serif; font-weight: 500;}}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  td.num.strong {{ color: var(--accent); font-weight: 600;}}

  /* CLUSTERS */
  .cluster-cards {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px; margin-bottom: 32px;
  }}
  .cluster-card {{
    background: var(--paper); border: 1px solid var(--line);
    padding: 16px 18px; position: relative;
    box-shadow: 4px 4px 0 var(--line);
  }}
  .cluster-card header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 12px;
    border-bottom: 1px dotted #cdc6b3; padding-bottom: 10px;}}
  .cluster-tag {{
    display: inline-block; background: var(--ink); color: var(--paper);
    font-family: 'JetBrains Mono', monospace; font-weight: 600; font-size: 11px;
    padding: 2px 7px; letter-spacing: 0.05em;
  }}
  .cluster-card h3 {{
    font-family: 'Fraunces', serif; font-style: italic; font-size: 16px; font-weight: 500;
    margin: 0; flex: 1; letter-spacing: -0.005em;
  }}
  .cluster-card h3::before {{ display: none; }}
  .cluster-stats {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 6px 14px;
    font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
  }}
  .cluster-stats > div {{ display: flex; flex-direction: column; }}
  .cl-lbl {{ font-size: 9.5px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--ink-soft); }}
  .cl-val {{ font-weight: 600; font-size: 14px; color: var(--ink); }}
  .cl-sub {{ font-size: 10px; color: var(--ink-soft); }}

  /* crosstab */
  .crosstab-wrap h4 {{
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--ink-soft); margin: 24px 0 6px;
  }}
  .crosstab-wrap .cap {{
    font-family: 'Fraunces', serif; font-style: italic;
    font-size: 13.5px; color: var(--ink-soft); margin: 0 0 12px; max-width: 70ch;
  }}
  table.crosstab {{ border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 12px;}}
  table.crosstab th {{
    background: var(--paper-deep); padding: 6px 12px; border: 1px solid var(--line);
    font-size: 10.5px; letter-spacing: 0.12em; text-transform: uppercase;
  }}
  table.crosstab td {{ padding: 8px 12px; border: 1px solid var(--line); text-align: right;}}
  .heatcell {{
    background: rgba(201, 66, 31, calc(var(--int) * 0.55));
    color: var(--ink); font-weight: 600;
    position: relative;
  }}
  .heat-pct {{ display: block; font-size: 9.5px; font-weight: 400; color: var(--ink-soft); margin-top: 2px;}}

  /* TIMESERIES */
  .ts-block {{
    background: var(--paper); border: 1px solid var(--line);
    padding: 24px; box-shadow: 6px 6px 0 var(--line);
  }}
  .ts-chart-wrap {{ background: #fbf7eb; border: 1px solid #d3cbb5; padding: 12px; margin-bottom: 18px;}}
  svg.ts-chart {{ width: 100%; height: auto; display: block; }}
  svg.ts-chart .series {{
    fill: none; stroke-width: 2.2;
    stroke-linejoin: round; stroke-linecap: round;
  }}
  svg.ts-chart .dot {{ stroke: var(--paper); stroke-width: 1.5; }}
  svg.ts-chart .dot.partial {{ stroke-dasharray: 2 2; opacity: 0.7;}}
  svg.ts-chart .grid {{ stroke: #d3cbb5; stroke-width: 0.6; }}
  svg.ts-chart .grid.base {{ stroke: var(--line); stroke-width: 1.2; }}
  svg.ts-chart .axlbl {{ font-family: 'JetBrains Mono', monospace; font-size: 10.5px; fill: var(--ink-soft); }}
  svg.ts-chart .axlbl.partial {{ fill: var(--accent); font-style: italic;}}

  .ts-legend {{
    list-style: none; padding: 0; margin: 0 0 18px;
    display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 10px 24px;
  }}
  .ts-legend li {{
    display: grid; grid-template-columns: auto 1fr auto; gap: 10px; align-items: baseline;
    padding: 6px 0; border-bottom: 1px dotted #cdc6b3;
  }}
  .leg-swatch {{ width: 14px; height: 4px; display: block; }}
  .leg-name {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px; color: var(--ink);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }}
  .leg-val {{
    font-family: 'Fraunces', serif; font-weight: 700; font-size: 17px;
    font-variant-numeric: tabular-nums;
  }}
  .leg-sub {{ font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
    color: var(--ink-soft); font-weight: 400; display: block; text-align: right; }}

  table.ts-table {{
    width: 100%; border-collapse: collapse; font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
  }}
  table.ts-table th {{
    background: var(--paper-deep); padding: 6px 8px; border-bottom: 1px solid var(--line);
    font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase; text-align: right;
  }}
  table.ts-table th:first-child {{ text-align: left;}}
  table.ts-table td, table.ts-table th {{ padding: 5px 8px; }}
  table.ts-table td.num.pos {{ color: var(--pos); font-weight: 600;}}
  table.ts-table td.num.neg {{ color: var(--neg); font-weight: 600;}}
  table.ts-table td.num.partial {{ font-style: italic; opacity: 0.75;}}
  table.ts-table td.num.partial.pos {{ color: var(--pos);}}
  table.ts-table td.dim {{ color: #b9b1a0;}}

  .ts-note {{ font-family: 'Fraunces', serif; font-style: italic; font-size: 13.5px;
    color: var(--ink-soft); margin: 16px 0 0; }}
  .partial-tag {{ color: var(--accent); font-style: normal; font-weight: 700;}}

  /* Logo */
  .bp-logo {{
    width: 64px; height: 64px; display: block; margin-bottom: 22px;
    color: var(--ink);
  }}

  /* §06 Explorador territorial */
  .map-block {{
    background: var(--paper); border: 1px solid var(--line);
    padding: 24px; box-shadow: 6px 6px 0 var(--line);
  }}
  h2.sec em {{
    font-style: italic; color: var(--accent);
    font-variation-settings: "opsz" 72;
  }}
  .map-controls {{
    display: grid; grid-template-columns: 1fr 1fr 1.8fr auto;
    gap: 16px; margin-bottom: 18px;
  }}
  .map-input {{ display: flex; flex-direction: column; gap: 6px; }}
  .map-input label {{
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    letter-spacing: 0.18em; text-transform: uppercase; color: var(--ink-soft);
  }}
  .map-input input[type="number"] {{
    font-family: 'JetBrains Mono', monospace; font-size: 13px;
    padding: 8px 10px; background: #fbf7eb;
    border: 1px solid var(--line); color: var(--ink); outline: none;
  }}
  .map-input input[type="number"]:focus {{ border-color: var(--accent); }}
  .map-input input[type="range"] {{
    width: 100%; accent-color: var(--accent); height: 28px;
  }}
  .map-input button {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    padding: 9px 16px; background: var(--ink); color: var(--paper);
    border: 1px solid var(--ink); cursor: pointer;
    letter-spacing: 0.1em; text-transform: uppercase; font-weight: 600;
    transition: background 0.15s;
  }}
  .map-input button:hover {{ background: var(--accent); border-color: var(--accent); }}

  .map-grid {{
    display: grid; grid-template-columns: 2fr 1fr; gap: 16px; margin-bottom: 16px;
  }}
  #map-canvas {{
    height: 520px; border: 1px solid var(--line);
    background: #f0eadb;
  }}
  /* Leaflet overrides para encajar con la estética */
  .leaflet-container {{ background: #ebe4d3; font-family: 'Inter Tight', sans-serif;}}
  .leaflet-control-attribution {{
    font-family: 'JetBrains Mono', monospace; font-size: 9px !important;
    background: rgba(247, 242, 231, 0.85) !important;
  }}
  .leaflet-bar a {{
    background: var(--paper) !important; color: var(--ink) !important;
    border-bottom: 1px solid var(--line) !important;
  }}
  .leaflet-bar a:hover {{ background: var(--accent) !important; color: #fff !important;}}

  .ctr-icon {{ background: transparent; border: none; }}
  .ctr-cross {{
    position: relative; width: 22px; height: 22px;
  }}
  .ctr-cross i {{
    position: absolute; background: var(--accent); display: block;
  }}
  .ctr-cross i:nth-child(1) {{ top: 10px; left: 0; width: 22px; height: 2px;}}
  .ctr-cross i:nth-child(2) {{ left: 10px; top: 0; height: 22px; width: 2px;}}
  .ctr-cross::after {{
    content: ""; position: absolute; top: 6px; left: 6px;
    width: 10px; height: 10px; border: 1.5px solid var(--accent);
    background: rgba(247, 242, 231, 0.85); border-radius: 50%;
  }}

  .map-panel {{
    background: #fbf7eb; border: 1px solid var(--line);
    padding: 18px 18px 20px;
  }}
  .map-panel h4 {{
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    letter-spacing: 0.18em; text-transform: uppercase; color: var(--ink-soft);
    margin: 0 0 14px; padding-bottom: 8px; border-bottom: 1px solid var(--line);
  }}
  .map-panel h5 {{
    font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
    letter-spacing: 0.16em; text-transform: uppercase; color: var(--ink-soft);
    margin: 18px 0 8px;
  }}
  .map-kpi {{
    display: flex; flex-direction: column; gap: 1px;
    padding: 8px 0; border-bottom: 1px dotted #cdc6b3;
  }}
  .map-kpi .lbl {{
    font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
    color: var(--ink-soft); text-transform: uppercase; letter-spacing: 0.12em;
  }}
  .map-kpi .val {{
    font-family: 'Fraunces', serif; font-size: 24px; font-weight: 700;
    letter-spacing: -0.02em; line-height: 1.1;
    font-variation-settings: "opsz" 96;
  }}
  .map-dominant {{
    padding: 14px 0 4px; border-bottom: 1px dotted #cdc6b3;
  }}
  .map-dominant .lbl {{
    font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
    color: var(--ink-soft); text-transform: uppercase; letter-spacing: 0.12em;
    display: block; margin-bottom: 6px;
  }}
  .map-dominant .dom-row {{ margin-bottom: 6px; }}
  .map-dominant .cluster-tag {{
    display: inline-block; padding: 4px 10px; color: white;
    font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
    font-weight: 700; letter-spacing: 0.06em;
  }}
  .map-dominant .val.small {{
    font-family: 'Fraunces', serif; font-style: italic;
    font-size: 14.5px; font-weight: 500; display: block;
    line-height: 1.3; color: var(--ink);
  }}
  .map-dominant .sub {{
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    color: var(--ink-soft); display: block; margin-top: 4px;
  }}

  #m-ts-svg {{
    width: 100%; height: 130px; background: #f4ecda;
    border: 1px solid #d3cbb5; display: block;
  }}

  .clase-bars {{ list-style: none; padding: 0; margin: 6px 0 0; }}
  .clase-bars li {{
    display: grid; grid-template-columns: 48px 1fr 50px;
    gap: 8px; align-items: center; padding: 3px 0;
  }}
  .cl-tag {{
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    font-weight: 600; color: var(--ink);
  }}
  .cl-bar {{
    background: #e0d8c3; height: 8px; border: 1px solid var(--line);
    position: relative; overflow: hidden;
  }}
  .cl-bar > span {{
    display: block; height: 100%; background: var(--teal);
    background-image: linear-gradient(90deg, var(--teal), #2b8a76);
  }}
  .cl-n {{
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    text-align: right; color: var(--ink-soft);
    font-variant-numeric: tabular-nums;
  }}

  .map-legend {{ padding-top: 16px; border-top: 1px solid var(--line); }}
  .map-legend h5 {{
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    letter-spacing: 0.16em; text-transform: uppercase; color: var(--ink-soft);
    margin: 0 0 10px;
  }}
  .map-legend ul {{
    list-style: none; padding: 0; margin: 0;
    display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
    gap: 6px 18px;
  }}
  .map-legend li {{
    display: grid; grid-template-columns: 14px 1fr auto;
    gap: 10px; align-items: center;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    padding: 4px 0; border-bottom: 1px dotted #cdc6b3;
  }}
  .map-legend .leg-swatch {{
    width: 12px; height: 12px; border-radius: 50%;
    border: 1.5px solid var(--paper);
    box-shadow: 0 0 0 1px rgba(28,26,23,0.25);
  }}
  .map-legend .leg-name b {{ color: var(--accent); font-weight: 700;}}
  .map-legend .leg-count {{
    color: var(--ink-soft); font-variant-numeric: tabular-nums;
  }}

  /* Hint helper bajo tablas y grids interactivas */
  .table-hint {{
    font-family: 'Fraunces', serif; font-style: italic; font-size: 12.5px;
    color: var(--ink-soft); margin: 8px 0 0;
  }}

  /* Top alcaldías clickable */
  .muni-table.clickable tr.muni-row {{ cursor: pointer; }}
  .muni-table.clickable tr.muni-row:hover td {{ background: #f0eadb; }}
  .muni-table.clickable td.arrow {{
    color: var(--accent); font-weight: 700; font-size: 14px;
    width: 24px;
  }}
  .muni-table.clickable tr.muni-row:focus {{ outline: 2px solid var(--accent); outline-offset: -2px; }}

  /* Cluster cards clickable */
  .cluster-card.clickable {{ cursor: pointer; transition: transform 0.12s, box-shadow 0.12s; }}
  .cluster-card.clickable:hover {{
    transform: translate(-2px, -2px);
    box-shadow: 7px 7px 0 var(--line);
  }}
  .cluster-card.clickable:focus {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
  .cluster-card header .arrow {{
    color: var(--accent); font-weight: 700; font-size: 16px; margin-left: auto;
  }}

  /* === LAYOUT NUEVO DEL EXPLORADOR === */
  .map-layout {{
    display: grid;
    grid-template-columns: 248px 1fr 288px;
    gap: 16px; margin-bottom: 18px;
  }}
  @media (max-width: 1200px) {{
    .map-layout {{ grid-template-columns: 220px 1fr 240px; }}
  }}
  .map-area {{ min-width: 0; }}
  #map-canvas {{
    height: 560px; width: 100%; border: 1px solid var(--line);
    background: #f0eadb;
  }}

  /* === FILTER PANEL === */
  .filter-panel {{
    background: #fbf7eb; border: 1px solid var(--line);
    padding: 12px 14px 16px;
    overflow-y: auto; max-height: 720px;
  }}
  .fp-head {{
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid var(--line); padding-bottom: 8px; margin-bottom: 12px;
  }}
  .fp-head h4 {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--ink-soft); margin: 0;
  }}
  .fp-head button {{
    font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
    letter-spacing: 0.12em; text-transform: uppercase;
    padding: 4px 10px; background: var(--ink); color: var(--paper);
    border: 1px solid var(--ink); cursor: pointer;
  }}
  .fp-head button:hover {{ background: var(--accent); border-color: var(--accent); }}

  .f-row {{ margin-bottom: 12px; }}
  .fp-label {{
    font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-soft);
    display: block; margin-bottom: 4px;
  }}
  .f-row select, .f-row input[type="number"] {{
    font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
    padding: 5px 7px; background: #fff; border: 1px solid #c9c2af;
    color: var(--ink); width: 100%; outline: none;
  }}
  .f-row select:focus, .f-row input[type="number"]:focus {{ border-color: var(--accent); }}

  .dual-input {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }}
  .dual-input.small input {{ padding: 4px 6px; font-size: 11px;}}

  .check-row, .radio-row {{
    display: flex; flex-wrap: wrap; gap: 8px;
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
  }}
  .check-row label, .radio-row label {{
    display: inline-flex; align-items: center; gap: 4px; cursor: pointer;
    padding: 3px 6px; border: 1px solid #d3cbb5; background: #f7f2e7;
  }}
  .check-row input, .radio-row input {{ accent-color: var(--accent); }}

  .chip-row {{ display: flex; flex-wrap: wrap; gap: 4px; }}
  .chip-row .chip {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    padding: 4px 9px; background: #f7f2e7; color: var(--ink);
    border: 1px solid #c9c2af; cursor: pointer;
    transition: background 0.1s;
  }}
  .chip-row .chip:hover {{ background: #e9e1c9; }}
  .chip-row .chip.on {{
    background: var(--accent); color: #fff; border-color: var(--accent);
  }}

  .f-sub-row {{ margin-top: 6px; }}
  .buffer-row {{ display: flex; flex-direction: column; gap: 4px; margin-top: 8px; }}
  .buf-lbl {{ font-family: 'JetBrains Mono', monospace; font-size: 10px; color: var(--ink-soft);}}
  .buf-lbl b {{ color: var(--accent); }}
  .latlon-row .full-btn {{
    width: 100%; margin-top: 8px;
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    padding: 6px 10px; background: var(--ink); color: var(--paper);
    border: 1px solid var(--ink); cursor: pointer;
    text-transform: uppercase; letter-spacing: 0.08em;
  }}
  .latlon-row .full-btn:hover {{ background: var(--accent); border-color: var(--accent);}}

  .toggle-row {{
    display: flex; align-items: center; gap: 6px;
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    padding: 4px 0;
  }}
  .toggle-row input {{ accent-color: var(--accent); }}

  /* Cluster predominante clickable */
  .cluster-tag.clickable {{ cursor: pointer; }}
  .cluster-tag.clickable:hover {{ filter: brightness(1.15); outline: 2px solid var(--ink); outline-offset: -2px; }}

  /* === BLOQUE PLUSVALIA TODOS LOS CLUSTERS === */
  .plus-block {{
    background: #fbf7eb; border: 1px solid var(--line); padding: 18px 20px;
    margin-bottom: 18px;
  }}
  .plus-block > header {{
    border-bottom: 1px solid var(--line); padding-bottom: 8px; margin-bottom: 12px;
  }}
  .plus-block > header h4 {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    letter-spacing: 0.18em; text-transform: uppercase;
    color: var(--ink-soft); margin: 0 0 4px;
  }}
  .plus-block > header p {{
    font-family: 'Fraunces', serif; font-style: italic; font-size: 13px;
    color: var(--ink-soft); margin: 0;
  }}
  .plus-chart-wrap {{
    background: #f4ecda; border: 1px solid #d3cbb5; padding: 6px 8px; margin-bottom: 10px;
  }}
  #m-plus-svg {{ width: 100%; height: auto; display: block; }}
  #m-plus-svg .ts-line {{
    fill: none; stroke-width: 2.2; stroke-linejoin: round; stroke-linecap: round;
  }}
  #m-plus-svg .grid {{ stroke: #d3cbb5; stroke-width: 0.6; }}
  #m-plus-svg .grid.base {{ stroke: var(--line); stroke-width: 1.2; }}
  #m-plus-svg .axlbl {{ font-family: 'JetBrains Mono', monospace; font-size: 10.5px; fill: var(--ink-soft); }}
  #m-plus-svg .axlbl.partial {{ fill: var(--accent); font-style: italic; }}

  /* === MODAL === */
  .modal-overlay {{
    position: fixed; inset: 0; background: rgba(28, 26, 23, 0.55);
    display: none; justify-content: center; align-items: flex-start;
    padding: 40px 16px; z-index: 1000; overflow-y: auto;
    backdrop-filter: blur(2px);
  }}
  .modal-overlay.open {{ display: flex; }}
  .modal {{
    background: var(--paper); border: 1px solid var(--line);
    box-shadow: 12px 12px 0 var(--line);
    max-width: 1140px; width: 100%; padding: 0;
    animation: fade-up 0.25s ease both;
  }}
  .modal-header {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 16px 24px; border-bottom: 1px solid var(--line);
    background: var(--paper-deep);
  }}
  .modal-header h2 {{
    font-family: 'Fraunces', serif; font-weight: 700;
    font-size: 22px; letter-spacing: -0.01em; margin: 0; line-height: 1.2;
  }}
  .modal-tag {{
    display: inline-block; padding: 3px 9px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    font-weight: 700; color: #fff; letter-spacing: 0.06em;
    vertical-align: middle; margin-right: 6px;
  }}
  .modal-tag-sm {{
    display: inline-block; padding: 2px 7px;
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    font-weight: 700; color: #fff;
  }}
  #modal-close {{
    flex-shrink: 0;
    width: 38px; height: 38px;
    background: transparent; border: 1px solid var(--line); cursor: pointer;
    font-size: 22px; color: var(--ink); padding: 0; line-height: 1;
    display: flex; align-items: center; justify-content: center;
    font-family: 'JetBrains Mono', monospace;
  }}
  #modal-close:hover {{ background: var(--accent); color: #fff; border-color: var(--accent); }}
  .modal-header h2 {{ flex: 1; min-width: 0; }}
  .modal-body {{ padding: 18px 24px 24px; }}

  .modal-toolbar {{
    display: flex; gap: 18px; align-items: center; justify-content: space-between;
    padding-bottom: 12px; border-bottom: 1px dotted #cdc6b3; margin-bottom: 12px;
  }}
  .modal-toolbar label {{
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-soft);
    display: flex; gap: 8px; align-items: center;
  }}
  .modal-toolbar select {{
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    padding: 5px 8px; background: #fff; border: 1px solid var(--line);
  }}
  .modal-n {{
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    color: var(--ink-soft);
  }}
  .modal-n b {{ color: var(--ink); font-weight: 600;}}

  .modal-stats {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin-bottom: 14px;
  }}
  .modal-stats .ms {{
    background: #fbf7eb; border: 1px solid var(--line);
    padding: 10px 14px;
  }}
  .modal-stats .ms .lbl {{
    font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
    letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-soft);
    display: block; margin-bottom: 2px;
  }}
  .modal-stats .ms .val {{
    font-family: 'Fraunces', serif; font-weight: 700; font-size: 22px;
    letter-spacing: -0.02em; display: block; line-height: 1;
    font-variation-settings: "opsz" 96;
  }}
  .modal-stats .ms .sub {{
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    color: var(--ink-soft); display: block; margin-top: 4px;
  }}

  .modal-aux {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 18px; }}
  .modal-aux .aux-block {{ background: #fbf7eb; border: 1px solid var(--line); padding: 12px 16px;}}
  .modal-aux .aux-block.full {{ grid-column: 1 / -1; }}
  .modal-aux h4 {{
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    letter-spacing: 0.16em; text-transform: uppercase; color: var(--ink-soft);
    margin: 0 0 8px;
  }}
  .cl-nse-list, .cl-muni-list {{ list-style: none; padding: 0; margin: 0; }}
  .cl-nse-list li, .cl-muni-list li {{
    display: flex; justify-content: space-between;
    padding: 3px 0; border-bottom: 1px dotted #cdc6b3;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
  }}
  .cl-nse-list li:last-child, .cl-muni-list li:last-child {{ border-bottom: none; }}
  .cl-nse-pct, .cl-muni-n {{ color: var(--accent); font-weight: 600;}}
  .cl-nse-list li.empty, .cl-muni-list li.empty {{ font-style: italic; color: var(--ink-soft); border: none; }}

  .modal-sec {{
    font-family: 'Fraunces', serif; font-style: italic; font-size: 17px;
    color: var(--ink); margin: 16px 0 10px; font-weight: 600;
    letter-spacing: -0.005em;
  }}
  .hist-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    gap: 12px;
  }}
  .hist-card {{
    background: #fbf7eb; border: 1px solid var(--line); padding: 10px 12px;
  }}
  .hist-head {{
    display: flex; justify-content: space-between; align-items: baseline;
    border-bottom: 1px dotted #cdc6b3; padding-bottom: 4px; margin-bottom: 4px;
  }}
  .hist-title {{
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    letter-spacing: 0.12em; text-transform: uppercase; font-weight: 600;
  }}
  .hist-meta {{
    font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
    color: var(--ink-soft);
  }}
  svg.hist-svg {{ width: 100%; height: auto; display: block; }}
  svg.hist-svg .hb {{ fill: var(--accent); fill-opacity: 0.7; }}
  svg.hist-svg .hb:hover {{ fill-opacity: 1; }}
  svg.hist-svg .hax {{ stroke: var(--line); stroke-width: 0.8;}}
  svg.hist-svg .hlbl {{ font-family: 'JetBrains Mono', monospace; font-size: 9.5px; fill: var(--ink-soft);}}
  svg.hist-svg .med-line {{ stroke: var(--teal); stroke-width: 2; stroke-dasharray: 4 3; }}
  svg.hist-svg .mean-line {{ stroke: var(--ink); stroke-width: 1.5; stroke-dasharray: 1 2;}}
  svg.hist-svg .med-lbl  {{ font-family: 'JetBrains Mono', monospace; font-size: 9px; fill: var(--teal); font-weight: 700;}}
  svg.hist-svg .mean-lbl {{ font-family: 'JetBrains Mono', monospace; font-size: 9px; fill: var(--ink);  font-weight: 700;}}

  /* Tooltip de alcaldías sobre el mapa */
  .leaflet-tooltip.alc-tip {{
    background: rgba(247, 242, 231, 0.97); border: 1px solid var(--line);
    color: var(--ink); font-family: 'Inter Tight', sans-serif;
    font-size: 12px; padding: 6px 10px; box-shadow: 3px 3px 0 rgba(28,26,23,0.18);
  }}
  .leaflet-tooltip.alc-tip i {{ color: var(--accent); font-size: 10px; }}

  .map-err {{ padding: 24px; font-family: 'Fraunces', serif; font-style: italic;
              color: var(--ink-soft); text-align: center; }}

  /* === Recentrar control dentro del mapa === */
  .recentrar-ctrl {{
    border: 1px solid var(--line) !important;
    box-shadow: 3px 3px 0 rgba(28,26,23,0.15) !important;
  }}
  .recentrar-ctrl a {{
    display: inline-block;
    background: var(--ink); color: var(--paper) !important;
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    padding: 7px 12px; cursor: pointer;
    text-decoration: none !important; text-transform: uppercase; letter-spacing: 0.08em;
    width: auto !important; height: auto !important; line-height: 1.2 !important;
    font-weight: 600;
  }}
  .recentrar-ctrl a:hover {{ background: var(--accent) !important; }}

  /* === Year selector en el panel === */
  .map-panel-head {{
    display: flex; justify-content: space-between; align-items: center;
    border-bottom: 1px solid var(--line); padding-bottom: 8px; margin-bottom: 14px;
  }}
  .map-panel-head h4 {{
    margin: 0 !important; padding: 0 !important; border: 0 !important;
  }}
  .panel-year {{ display: flex; align-items: center; gap: 6px; }}
  .panel-year .py-lbl {{
    font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-soft);
  }}
  .panel-year select {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    padding: 3px 6px; background: #fff; border: 1px solid var(--line);
    color: var(--ink);
  }}

  /* === §06 Estimador de precio === */
  .predictor-block {{
    background: var(--paper); border: 1px solid var(--line);
    padding: 22px; box-shadow: 6px 6px 0 var(--line);
  }}
  .pred-steps {{
    display: grid; grid-template-columns: 1fr 1.6fr 1fr;
    gap: 18px; margin-bottom: 24px;
  }}
  .pred-step-zone {{ grid-column: 1 / -1; }}
  .pred-zone-grid {{
    display: grid; grid-template-columns: 1.4fr 1fr; gap: 14px; align-items: stretch;
  }}
  .pred-mini-map-wrap {{ display: flex; flex-direction: column; gap: 6px; }}
  #pred-mini-map {{
    width: 100%; height: 260px; border: 1px solid var(--line);
    background: #eee; border-radius: 3px;
  }}
  .pred-mini-hint {{
    font-family: 'Fraunces', serif; font-style: italic; font-size: 11.5px;
    color: var(--ink-soft); margin: 0;
  }}
  @media (max-width: 1000px) {{
    .pred-steps {{ grid-template-columns: 1fr; }}
    .pred-zone-grid {{ grid-template-columns: 1fr; }}
    #pred-mini-map {{ height: 220px; }}
  }}
  .pred-step {{
    background: #fbf7eb; border: 1px solid var(--line);
    padding: 14px 16px 16px;
  }}
  .pred-step > header {{
    display: flex; align-items: center; gap: 10px;
    border-bottom: 1px dotted #cdc6b3; padding-bottom: 8px; margin-bottom: 12px;
  }}
  .pred-num {{
    display: inline-flex; align-items: center; justify-content: center;
    width: 26px; height: 26px; border-radius: 50%;
    background: var(--accent); color: #fff;
    font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 13px;
  }}
  .pred-step h3 {{
    font-family: 'Fraunces', serif; font-style: italic; font-size: 16.5px;
    margin: 0; font-weight: 500; letter-spacing: -0.005em;
  }}
  .pred-inputs label {{
    display: flex; flex-direction: column; gap: 4px; margin-bottom: 10px;
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-soft);
  }}
  .pred-inputs.grid {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px 14px;
  }}
  .pred-inputs input[type="number"] {{
    font-family: 'JetBrains Mono', monospace; font-size: 14px;
    padding: 8px 10px; background: #fff; border: 1px solid #c9c2af;
    color: var(--ink); outline: none; text-transform: none;
  }}
  .pred-inputs input[type="number"]:focus {{ border-color: var(--accent); }}
  .pred-secondary {{
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    padding: 6px 10px; background: var(--paper-deep); color: var(--ink);
    border: 1px solid var(--line); cursor: pointer;
    letter-spacing: 0.06em;
  }}
  .pred-secondary:hover {{ background: var(--ink); color: var(--paper); }}
  .pred-primary {{
    font-family: 'JetBrains Mono', monospace; font-size: 13px;
    font-weight: 700; padding: 14px 18px;
    background: var(--accent); color: #fff; border: 1px solid var(--accent);
    cursor: pointer; letter-spacing: 0.1em; text-transform: uppercase;
    width: 100%;
    transition: background 0.15s, transform 0.1s;
  }}
  .pred-primary:hover {{ background: var(--accent-deep); transform: translate(-1px,-1px); box-shadow: 3px 3px 0 var(--line);}}
  .pred-hint {{
    font-family: 'Fraunces', serif; font-style: italic; font-size: 11.5px;
    color: var(--ink-soft); margin: 6px 0 0;
  }}

  .cluster-select-lbl {{
    display: flex; flex-direction: column; gap: 4px; margin-top: 12px;
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-soft);
  }}
  .cluster-select-lbl select {{
    font-family: 'JetBrains Mono', monospace; font-size: 12.5px;
    padding: 8px 10px; background: #fff; border: 1px solid #c9c2af;
    color: var(--ink); outline: none; text-transform: none; letter-spacing: 0;
  }}
  .cluster-select-lbl select:focus {{ border-color: var(--accent); }}

  .pred-btn-row {{ display: flex; gap: 6px; flex-wrap: wrap; }}
  .pred-primary-sm {{
    flex: 1; min-width: 120px;
    font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700;
    padding: 9px 12px; background: var(--accent); color: #fff;
    border: 1px solid var(--accent); cursor: pointer;
    letter-spacing: 0.08em; text-transform: uppercase;
  }}
  .pred-primary-sm:hover {{ background: var(--accent-deep); }}

  /* Banner efímero "click en mapa" */
  .pick-banner {{
    position: fixed; top: 16px; left: 50%; transform: translate(-50%, -100%);
    background: var(--ink); color: var(--paper);
    border: 1px solid var(--accent); box-shadow: 4px 4px 0 rgba(28,26,23,0.25);
    padding: 10px 20px; z-index: 1500;
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    letter-spacing: 0.06em;
    transition: transform 0.25s ease;
  }}
  .pick-banner.show {{ transform: translate(-50%, 0); }}
  .pick-banner b {{ color: var(--accent); }}
  .pick-banner small {{ color: rgba(244, 234, 215, 0.7); margin-left: 6px;}}

  /* Análisis Comparativo de Mercado */
  .acm-block {{
    margin-top: 24px; padding-top: 18px;
    border-top: 1px dashed var(--line);
  }}
  .acm-head {{
    display: flex; justify-content: space-between; align-items: baseline;
    gap: 18px; margin-bottom: 12px; flex-wrap: wrap;
  }}
  .acm-head h4 {{ margin: 0; flex: 1; }}
  .acm-sub {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    font-style: normal; color: var(--ink-soft); letter-spacing: 0.08em;
  }}
  .acm-stat {{
    font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
    color: var(--ink-soft);
  }}
  .acm-stat b {{ color: var(--ink); font-weight: 700;}}

  /* Comparables table: similarity bar column */
  .pred-table.comp-table {{ font-size: 11px; }}
  .pred-table.comp-table td {{ padding: 7px 8px; }}
  .sim-cell {{ display: flex; align-items: center; gap: 8px; min-width: 110px; }}
  .sim-bar {{
    flex: 1; height: 10px;
    background: #e0d8c3; border: 1px solid var(--line); position: relative;
    overflow: hidden;
  }}
  .sim-bar > span {{
    display: block; height: 100%;
    background: var(--accent);
    background-image: repeating-linear-gradient(45deg,
      transparent, transparent 4px,
      rgba(0,0,0,0.10) 4px, rgba(0,0,0,0.10) 5px);
  }}
  .sim-val {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: var(--accent); font-weight: 700; min-width: 38px; text-align: right;
  }}

  .pred-meta .lbl-sub {{
    font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
    color: var(--ink-soft); margin-top: 2px;
  }}

  /* Estado: empty / loading / output */
  .pred-result {{
    background: #fbf7eb; border: 1px solid var(--line); padding: 18px 20px;
    min-height: 220px;
  }}
  .pred-result .empty {{ padding: 0; }}

  .pred-loading {{
    display: flex; gap: 18px; align-items: center;
    padding: 20px; min-height: 160px;
  }}
  .loading-logo {{ width: 64px; height: 64px; color: var(--ink); flex-shrink: 0;}}
  @keyframes lspin {{ from {{ transform: rotate(0deg);}} to {{ transform: rotate(360deg);}} }}
  @keyframes lpulse {{ 0%,100% {{ opacity: 1;}} 50% {{ opacity: 0.4;}} }}
  .loading-logo .lrotor {{ animation: lspin 1.4s linear infinite; transform-origin: 30px 30px; }}
  .loading-logo .lframe {{ animation: lpulse 1.4s ease-in-out infinite; }}
  .pred-status-wrap {{ display: flex; flex-direction: column; gap: 6px; }}
  .pred-status {{
    font-family: 'Fraunces', serif; font-style: italic; font-size: 22px;
    color: var(--ink); font-weight: 500;
  }}
  .pred-tick {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px;
    color: var(--ink-soft); letter-spacing: 0.06em;
  }}

  .pred-output {{ display: block; }}
  .pred-main {{
    display: flex; flex-direction: column; align-items: flex-start; gap: 4px;
    padding-bottom: 14px; margin-bottom: 14px;
    border-bottom: 1px solid var(--line);
  }}
  .pred-main .pred-lbl {{
    font-family: 'JetBrains Mono', monospace; font-size: 10px;
    letter-spacing: 0.18em; text-transform: uppercase; color: var(--ink-soft);
  }}
  .pred-main .pred-big {{
    font-family: 'Fraunces', serif; font-weight: 800; font-size: 76px;
    letter-spacing: -0.03em; color: var(--accent); line-height: 0.95;
    font-variation-settings: "opsz" 144;
    margin: 4px 0;
  }}
  .pred-main .pred-band {{
    font-family: 'JetBrains Mono', monospace; font-size: 12px;
    color: var(--ink-soft);
  }}
  .pred-meta {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 12px; margin-bottom: 18px;
  }}
  .pred-meta > div {{
    background: var(--paper); border: 1px solid var(--line); padding: 10px 12px;
    display: flex; flex-direction: column; gap: 4px;
  }}
  .pred-meta .lbl {{
    font-family: 'JetBrains Mono', monospace; font-size: 9.5px;
    letter-spacing: 0.14em; text-transform: uppercase; color: var(--ink-soft);
  }}
  .pred-meta .val {{
    font-family: 'Fraunces', serif; font-weight: 600; font-size: 16px;
    color: var(--ink); line-height: 1.3;
  }}
  .pred-meta .val small {{
    display: block; font-family: 'JetBrains Mono', monospace; font-size: 10px;
    color: var(--ink-soft); font-weight: 400; margin-top: 2px;
  }}
  .pred-tag {{
    display: inline-block; padding: 2px 6px; color: #fff;
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    font-weight: 700; letter-spacing: 0.05em;
  }}
  .pred-sec {{
    font-family: 'Fraunces', serif; font-style: italic; font-size: 15px;
    margin: 16px 0 8px; color: var(--ink-soft);
  }}
  table.pred-table {{ width: 100%; border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace; font-size: 11.5px;
  }}
  table.pred-table th {{
    background: var(--paper-deep); padding: 6px 8px; border-bottom: 1px solid var(--line);
    font-size: 9.5px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--ink-soft); text-align: left;
  }}
  table.pred-table th.num {{ text-align: right; }}
  table.pred-table td {{ padding: 6px 8px; border-bottom: 1px dotted #cdc6b3; vertical-align: top;}}
  table.pred-table td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  table.pred-table tr:hover td {{ background: #f0eadb; }}
  table.pred-table small {{ color: var(--ink-soft); font-size: 9.5px;}}

  .empty {{ font-style: italic; color: var(--ink-soft); padding: 24px; text-align: center;}}

  footer.bp-footer {{
    margin-top: 96px; padding: 28px 32px; background: var(--bp-dark);
    color: var(--on-dark); display: flex; justify-content: space-between;
    align-items: center; border: 1px solid var(--ink);
    box-shadow: 6px 6px 0 var(--bp-gray);
  }}
  .bp-footer-left {{ display: flex; align-items: center; gap: 16px; }}
  .bp-footer-wordmark {{
    height: 22px; width: auto; display: block;
    filter: brightness(0) invert(1);
  }}
  .bp-footer-orange {{ height: 28px; width: auto; display: block; }}
  .bp-footer-right {{
    display: flex; flex-direction: column; align-items: flex-end; gap: 4px;
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    letter-spacing: 0.12em; text-transform: uppercase;
  }}
  .bp-footer-tagline {{ color: var(--on-dark); }}
  .bp-footer-tagline b {{ color: var(--accent); }}
  .bp-footer-meta {{ color: var(--on-dark-soft); font-size: 9.5px; }}

  /* staggered reveal */
  @keyframes fade-up {{
    from {{ opacity: 0; transform: translateY(12px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}
  header.hero, section.kpis, h2.sec, .sec-rule, .sec-intro, .panel, .ts-block, .cluster-cards, .crosstab-wrap {{
    animation: fade-up 0.55s ease both;
  }}
  section.kpis {{ animation-delay: 0.08s;}}
  .panel:nth-child(1) {{ animation-delay: 0.12s;}}
  .panel:nth-child(2) {{ animation-delay: 0.18s;}}
  .panel:nth-child(3) {{ animation-delay: 0.24s;}}

  /* === Year-picker (pills) === */
  .year-picker {{
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    margin: 24px 0 14px; padding: 10px 14px;
    background: rgba(28, 26, 23, 0.04); border-radius: 8px;
  }}
  .year-picker .yp-lbl {{
    text-transform: uppercase; letter-spacing: 0.14em; font-size: 11px;
    color: var(--ink-soft); font-weight: 600; margin-right: 6px;
  }}
  .year-picker .yp {{
    padding: 5px 12px; border: 1px solid #d3ccba; border-radius: 999px;
    background: #fbf7eb; color: var(--ink); font: 500 12px 'Inter Tight', sans-serif;
    cursor: pointer; transition: all 0.15s;
  }}
  .year-picker .yp:hover {{ border-color: var(--accent); }}
  .year-picker .yp.on {{ background: var(--ink); color: #fbf7eb; border-color: var(--ink); }}
  .year-picker .yp.partial {{ font-style: italic; }}
  .year-picker .yp-hint {{
    margin-left: auto; font-size: 11px; color: var(--ink-soft); font-style: italic;
  }}

  /* === KPIs clickeables (3 cajas) === */
  section.kpis-3 {{ grid-template-columns: repeat(3, 1fr); }}
  .kpi.clickable {{
    cursor: pointer; border: 1px solid transparent; transition: all 0.15s;
    text-align: left; font: inherit;
  }}
  .kpi.clickable:hover {{
    border-color: var(--accent);
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(28,26,23,0.08);
  }}
  .kpi.clickable:focus {{ outline: 2px solid var(--accent); outline-offset: 2px; }}

  /* === Mercado por municipio: header + Comparar === */
  .muni-header {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 12px;
  }}
  .muni-cmp-btn {{
    padding: 6px 14px; border: 1px solid var(--ink); background: transparent;
    color: var(--ink); border-radius: 999px; font: 500 12px 'Inter Tight', sans-serif;
    cursor: pointer; transition: all 0.15s;
  }}
  .muni-cmp-btn:hover {{ background: var(--ink); color: #fbf7eb; }}
  .muni-cmp-btn.on {{ background: var(--accent); border-color: var(--accent); color: #fff; }}

  .muni-cmp-bar {{
    display: none; align-items: center; gap: 14px;
    margin-top: 10px; padding: 8px 14px; border-radius: 8px;
    background: rgba(201, 66, 31, 0.08);
  }}
  .muni-cmp-bar.open {{ display: flex; }}
  .muni-cmp-bar .muni-cmp-count {{ font-weight: 600; color: var(--accent); }}
  .muni-cmp-bar button {{
    padding: 5px 12px; border: 1px solid var(--ink); background: #fbf7eb;
    border-radius: 6px; cursor: pointer; font: 500 12px 'Inter Tight', sans-serif;
  }}
  .muni-cmp-bar button.muni-cmp-go {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
  .muni-cmp-bar button.muni-cmp-go:disabled {{ background: #d3ccba; border-color: #d3ccba; cursor: not-allowed; }}
  .muni-table.cmp-mode tr.muni-row {{ position: relative; }}
  .muni-table.cmp-mode tr.muni-row::before {{
    content: '○'; position: absolute; left: 6px; top: 50%; transform: translateY(-50%);
    color: #b8b1a0; font-size: 14px;
  }}
  .muni-table.cmp-mode tr.muni-row.cmp-on::before {{ content: '●'; color: var(--accent); }}
  .muni-table.cmp-mode tr.muni-row td:first-child {{ padding-left: 26px; }}

  /* === Comparación (modal) — chips, alpha slider, leyenda === */
  .cmp-chips {{ display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }}
  .cmp-chip {{
    display: inline-block; padding: 4px 10px; border-radius: 999px;
    color: #fff; font-size: 11px; font-weight: 500;
  }}
  .cmp-alpha {{
    display: flex; align-items: center; gap: 10px; margin: 12px 0;
    padding: 8px 12px; background: #f4eed9; border-radius: 6px;
  }}
  .cmp-alpha label {{ display: flex; align-items: center; gap: 10px; font-size: 12px; color: var(--ink-soft); }}
  .cmp-alpha input[type="range"] {{ width: 180px; }}
  .cmp-hists {{ display: flex; flex-direction: column; gap: 16px; }}
  .cmp-leg {{ list-style: none; padding: 6px 0 0; margin: 0; display: flex; flex-wrap: wrap; gap: 12px; font-size: 11px; }}
  .cmp-leg li {{ display: flex; align-items: center; gap: 6px; }}
  .cmp-leg-sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; }}
  .cmp-leg-name {{ color: var(--ink); font-weight: 500; }}
  .cmp-leg-val {{ color: var(--ink-soft); }}

  /* === Selector de años dentro del modal de KPI === */
  .kpi-year-picker {{
    display: flex; flex-wrap: wrap; align-items: center; gap: 6px;
    padding: 10px 12px; background: #f4eed9; border-radius: 6px;
    margin-bottom: 14px;
  }}
  .kpi-year-lbl {{
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    text-transform: uppercase; letter-spacing: 0.1em; color: var(--ink-soft);
    margin-right: 6px;
  }}
  .kpi-year-chip {{
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 9px; border: 1px solid #d3ccba; background: #fbf7eb;
    border-radius: 999px; font-size: 11px; cursor: pointer; user-select: none;
    font-family: 'JetBrains Mono', monospace;
  }}
  .kpi-year-chip:has(input:checked) {{
    background: var(--ink); color: #fbf7eb; border-color: var(--ink);
  }}
  .kpi-year-chip input {{ accent-color: var(--accent); }}

  /* === Comparar dentro de un histograma del modal de cluster === */
  .hist-cmp-btn {{
    margin-left: auto; padding: 3px 8px; border: 1px solid var(--ink-soft);
    background: transparent; border-radius: 999px; font-size: 10px;
    cursor: pointer; color: var(--ink); transition: all 0.15s;
  }}
  .hist-cmp-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
  .hist-cmp-btn.on {{ background: var(--accent); border-color: var(--accent); color: #fff; }}
  .hist-head {{ display: flex; align-items: center; gap: 10px; }}
  .hist-cmp-panel {{
    margin-top: 8px; padding: 10px; border-top: 1px dashed #d3ccba; background: #fbf7eb; border-radius: 4px;
  }}
  .cmp-picker {{ display: flex; flex-wrap: wrap; gap: 10px; align-items: center; margin-bottom: 8px; }}
  .cmp-lbl {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--ink-soft); }}
  .cmp-pick {{ display: inline-flex; align-items: center; gap: 5px; font-size: 11px; cursor: pointer; }}
  .cmp-pick-sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; }}
  .cmp-pick-name {{ color: var(--ink); }}
  .cmp-overlay-host {{ margin-top: 6px; }}

  /* === Mapa Atributos Físicos (§02) === */
  .map-layout.phys {{
    display: grid; grid-template-columns: 1fr 320px; gap: 18px;
    background: var(--paper); border: 1px solid #e6dfca; border-radius: 6px;
    padding: 14px; margin-top: 14px;
  }}
  .map-layout.phys .map-area {{ position: relative; }}
  .map-layout.phys #phys-map-canvas {{
    height: 560px; width: 100%; border-radius: 4px; overflow: hidden;
  }}
  .phys-map-tools {{
    display: flex; align-items: center; gap: 14px;
    margin-top: 10px; padding: 8px 10px;
    background: #fbf7eb; border-radius: 4px;
  }}
  .phys-map-tools .toggle-row {{ display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--ink-soft); }}
  .phys-panel {{
    background: #fbf7eb; border-radius: 4px; padding: 14px;
    display: flex; flex-direction: column; gap: 10px;
  }}
  .phys-panel h4 {{ margin: 0 0 6px; font-size: 13px; text-transform: uppercase; letter-spacing: 0.12em; color: var(--ink-soft);}}
  .phys-loc {{ font-size: 11px; color: var(--ink-soft); margin: 0; line-height: 1.45; }}
  .phys-attr-list {{ list-style: none; padding: 0; margin: 6px 0; }}
  .phys-attr {{
    display: flex; justify-content: space-between; padding: 8px 10px;
    border-bottom: 1px dashed #e0d8c4; cursor: pointer; border-radius: 3px;
    transition: background 0.12s;
  }}
  .phys-attr:hover {{ background: rgba(31, 108, 92, 0.08); }}
  .phys-attr .lbl {{ font-size: 11px; color: var(--ink-soft); text-transform: uppercase; letter-spacing: 0.08em;}}
  .phys-attr .val {{ font-size: 13px; color: var(--ink); font-weight: 600; }}
  .pa-sep {{ color: var(--ink-soft); font-weight: 400; opacity: 0.7; padding: 0 2px; }}
  .phys-buf-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--ink-soft); margin-top: 8px; }}
  .phys-panel .hint {{ font-size: 11px; font-style: italic; color: var(--ink-soft); margin: 4px 0 0; }}
  .export-png {{
    padding: 6px 14px; background: var(--ink); color: #fbf7eb;
    border: none; border-radius: 999px; font: 500 12px 'Inter Tight', sans-serif;
    cursor: pointer; transition: all 0.15s; margin-left: auto;
  }}
  .export-png:hover {{ background: var(--accent); }}

  /* === Margen del Desarrollador (en el panel §05) === */
  .map-kpi.accent {{
    background: rgba(201, 66, 31, 0.06); border-radius: 4px;
    padding: 10px; margin-top: 10px; border-left: 3px solid var(--accent);
  }}
  .map-kpi.accent .val {{ font-size: 22px; font-weight: 700; }}
  .map-kpi.accent .val.pos {{ color: #1f6c5c; }}
  .map-kpi.accent .val.neg {{ color: var(--accent); }}
  .m-valcon-range {{
    font-size: 11px; color: var(--ink-soft); margin: 6px 0 10px;
    display: flex; flex-direction: column; gap: 2px;
  }}
  .m-valcon-range .lbl {{ text-transform: uppercase; letter-spacing: 0.08em; font-size: 10px; opacity: 0.85; }}
  .m-valcon-range .vrow {{ display: flex; align-items: baseline; gap: 6px; }}
  .m-valcon-range .vtag {{
    display: inline-block; min-width: 26px; font-size: 9.5px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.08em; color: var(--ink-soft);
  }}
  .m-valcon-range .vrow.nbar .vtag {{ opacity: 0.55; }}
  .m-valcon-range .val {{ color: var(--ink); font-weight: 600; font-size: 12px; }}
  .valcon-hint {{ margin: 0 0 4px; font-size: 10.5px; }}
  .margin-formula {{ font-size: 10px; opacity: 0.65; margin-top: 2px; display: block; }}
  .map-export-row {{ margin-top: 14px; display: flex; }}
  .hint {{ font-style: italic; color: var(--ink-soft); font-size: 10.5px; }}

  /* === §07 Absorción === */
  .abs-block {{
    display: grid; grid-template-columns: 260px 1fr 260px; gap: 14px;
    background: var(--paper); border: 1px solid #e6dfca; border-radius: 6px;
    padding: 14px; margin-top: 14px;
  }}
  .abs-filters {{
    background: #fbf7eb; border-radius: 4px; padding: 14px;
    display: flex; flex-direction: column; gap: 10px;
  }}
  .abs-filters h5 {{ margin: 8px 0 4px; font-size: 12px; text-transform: uppercase; letter-spacing: 0.1em; color: var(--ink-soft); }}
  .abs-filters .f-row {{ display: flex; flex-direction: column; gap: 4px; }}
  .abs-filters .fp-label {{ font-size: 11px; color: var(--ink-soft); text-transform: uppercase; letter-spacing: 0.08em; }}
  .abs-filters select, .abs-filters input[type="number"] {{
    padding: 4px 8px; border: 1px solid #d3ccba; border-radius: 3px;
    background: #f7f2e7; font-size: 12px;
  }}
  .abs-filters .chip-row {{ display: flex; flex-wrap: wrap; gap: 4px; }}
  .abs-filters .chip {{
    padding: 3px 9px; border: 1px solid #d3ccba; background: #f7f2e7;
    border-radius: 999px; font-size: 11px; cursor: pointer;
  }}
  .abs-filters .chip.on {{ background: var(--ink); color: #fbf7eb; border-color: var(--ink); }}
  .abs-controls {{ display: flex; flex-direction: column; gap: 6px; padding-bottom: 8px; border-bottom: 1px dashed #e0d8c4; }}
  .seg {{ display: inline-flex; border: 1px solid #d3ccba; border-radius: 4px; overflow: hidden; }}
  .seg .seg-btn {{
    padding: 5px 10px; background: #f7f2e7; border: none;
    font: 500 11px 'Inter Tight', sans-serif; cursor: pointer; color: var(--ink);
  }}
  .seg .seg-btn.on {{ background: var(--ink); color: #fbf7eb; }}
  .seg .seg-btn + .seg-btn {{ border-left: 1px solid #d3ccba; }}
  .abs-reset-btn {{
    padding: 5px 12px; background: transparent; border: 1px solid var(--ink-soft);
    border-radius: 4px; cursor: pointer; font: 500 11px 'Inter Tight', sans-serif;
    margin-top: 8px; color: var(--ink-soft);
  }}
  .abs-reset-btn:hover {{ border-color: var(--accent); color: var(--accent); }}
  .abs-chart-wrap {{
    background: #fbf7eb; border-radius: 4px; padding: 14px;
    display: flex; flex-direction: column; gap: 8px;
  }}
  .abs-chart-head {{
    display: flex; justify-content: space-between; align-items: baseline; gap: 12px;
    padding-bottom: 8px; border-bottom: 1px dashed #d8d0bb;
  }}
  .abs-chart-title {{
    margin: 0; font-family: 'JetBrains Mono', monospace; font-size: 12px;
    font-weight: 500; letter-spacing: 0.04em; color: var(--ink);
  }}
  .abs-chart-title b {{ color: var(--accent); }}
  .abs-chart-meta {{
    font-family: 'JetBrains Mono', monospace; font-size: 10.5px;
    color: var(--ink-soft); white-space: nowrap;
  }}
  #abs-svg {{ width: 100%; height: 460px; background: #fffdf6; border-radius: 3px; }}
  .abs-legend {{
    list-style: none; padding: 0; margin: 0;
    display: flex; flex-wrap: wrap; gap: 8px;
  }}
  .abs-leg-item {{
    display: flex; align-items: center; gap: 5px; padding: 3px 8px;
    border-radius: 999px; background: #f4eed9; font-size: 11px; cursor: pointer;
    transition: opacity 0.15s, background 0.15s;
    border: 1px solid transparent;
  }}
  .abs-leg-item.off {{ opacity: 0.35; text-decoration: line-through; }}
  .abs-leg-item.out {{
    opacity: 0.55; background: transparent; border: 1px dashed #c2bba6;
    color: var(--ink-soft);
  }}
  .abs-leg-item.in:hover {{ background: rgba(201, 66, 31, 0.12); }}
  .abs-leg-item.out:hover {{ background: rgba(31, 108, 92, 0.10); opacity: 0.9; }}
  .abs-leg-sw {{ display: inline-block; width: 10px; height: 10px; border-radius: 2px; }}

  /* Listas Incluidos / Excluidos debajo del gráfico */
  .abs-dev-lists {{
    display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 6px;
  }}
  .abs-dev-section {{
    background: #fffdf6; border: 1px solid #e6dfca; border-radius: 4px;
    padding: 10px 12px; display: flex; flex-direction: column; gap: 6px;
  }}
  .abs-dev-section header {{
    display: flex; align-items: center; gap: 8px;
    padding-bottom: 6px; border-bottom: 1px dotted #d8d0bb;
  }}
  .abs-dev-section h5 {{
    margin: 0; flex: 1; font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.1em; color: var(--ink); font-weight: 600;
  }}
  .abs-dev-n {{
    font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 700;
    color: var(--accent); background: rgba(201, 66, 31, 0.10);
    border-radius: 999px; padding: 1px 8px;
  }}
  .abs-dev-excl .abs-dev-n {{ color: var(--ink-soft); background: #f0e8d3; }}
  .abs-dev-hint {{
    font-family: 'Fraunces', serif; font-style: italic; font-size: 10.5px;
    color: var(--ink-soft); margin: 0;
  }}
  .abs-dot {{
    width: 8px; height: 8px; border-radius: 50%;
  }}
  .abs-dot.in {{ background: var(--accent); }}
  .abs-dot.out {{ background: transparent; border: 1.5px dashed #b6ad94; }}
  .abs-legend li.empty {{
    list-style: none; font-style: italic; color: var(--ink-soft);
    font-size: 10.5px; padding: 4px 0; cursor: default;
    background: transparent !important; border: none !important;
  }}
  @media (max-width: 900px) {{
    .abs-dev-lists {{ grid-template-columns: 1fr; }}
  }}
  .abs-top {{
    background: #fbf7eb; border-radius: 4px; padding: 14px;
    display: flex; flex-direction: column; gap: 6px;
  }}
  .abs-top h4 {{ margin: 0; font-size: 13px; text-transform: uppercase; letter-spacing: 0.12em; color: var(--ink-soft); }}
  .abs-top .hint {{ margin: 0 0 6px; }}
  .abs-rank {{ list-style: none; padding: 0; margin: 0; counter-reset: rk; display: flex; flex-direction: column; gap: 5px; }}
  .abs-rank li {{
    display: grid; grid-template-columns: 18px 10px 1fr 50px 38px; align-items: center; gap: 6px;
    padding: 4px 6px; border-radius: 3px;
  }}
  .abs-rank li:hover {{ background: #f4eed9; }}
  .abs-rk-pos {{ font-weight: 700; color: var(--ink-soft); font-size: 11px; }}
  .abs-rk-sw {{ width: 8px; height: 8px; border-radius: 50%; }}
  .abs-rk-name {{ font-size: 11px; color: var(--ink); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .abs-rk-bar {{ position: relative; height: 6px; background: #ede5d0; border-radius: 3px; overflow: hidden; }}
  .abs-rk-fill {{ position: absolute; top: 0; left: 0; height: 100%; }}
  .abs-rk-n {{ font-size: 11px; color: var(--ink); font-weight: 600; text-align: right; }}
  .abs-cta {{
    margin-top: 10px; padding: 8px 10px; background: rgba(212, 155, 162, 0.20);
    border-left: 2px solid var(--rose); border-radius: 0 4px 4px 0;
    font-size: 11px; color: var(--ink); line-height: 1.45;
  }}
  .abs-cta b {{ color: var(--accent); }}

  /* === Responsive guard rails para layouts grid grandes === */
  @media (max-width: 1100px) {{
    .abs-block {{ grid-template-columns: 1fr; }}
    .map-layout.phys {{ grid-template-columns: 1fr; }}
  }}
</style>
</head>
<body>

<div class="container">

  <header class="hero hero-dark">
    <div class="hero-bp-mark">
      <img src="{logo_blackprint_light}" alt="BlackPrint" class="hero-wordmark"/>
      <img src="{logo_orange}" alt="" class="hero-orange-mark" aria-hidden="true"/>
    </div>
    <div class="hero-main">
      <div class="hero-left">
        <div class="marca">Módulo de Vivienda · CDMX</div>
        <h1>Módulo de<br/><em>Vivienda</em></h1>
        <p class="sub">Inteligencia de mercado para planear desarrollos inmobiliarios: identifica producto, valida hipótesis de diseño y conoce el precio al que se transacciona cada tipología en cada zona.</p>
      </div>
      <div class="meta-block">
        <dl>
          <dt>fuente</dt><dd>staging.stg_avaluos</dd>
          <dt>vista</dt><dd>fact_avaluos_cdmx</dd>
          <dt>filtro</dt><dd>cve_ent='09', proposito='1'</dd>
          <dt>moneda</dt><dd>MXN nominales</dd>
          <dt>generado</dt><dd>{generated}</dd>
        </dl>
      </div>
    </div>
  </header>

  <h2 class="sec"><span class="num-tag">§00</span>Estadísticas Generales <em>por Año</em></h2>
  <div class="sec-rule"></div>
  <p class="sec-purpose">Vista rápida del mercado de vivienda hipotecada en CDMX: valor mediano del inmueble concluido, costo por m² de construcción y de terreno. Selecciona un año para enfocar los KPIs y haz click en cualquier recuadro para abrir su histograma (y compararlo con otros años desde el modal).</p>

  {year_picker_block}

  {kpi_block}

  <h2 class="sec"><span class="num-tag">§01</span>Mercado <em>por municipio</em></h2>
  <div class="sec-rule"></div>
  <p class="sec-intro">Volumen de avalúos, mediana de valor de vivienda y mediana de precio por m² construido. <b>Click en una fila</b> para explorar submercados.</p>

  <div class="panel">
    <div class="muni-header">
      <h3>Top municipios por volumen de avalúo</h3>
      <button id="muni-compare-toggle" type="button" class="muni-cmp-btn">+ Comparar municipios</button>
    </div>
    {top_munis_html}
  </div>

  {phys_map_section}

  <h2 class="sec"><span class="num-tag">§03</span>Submercados <em>de Vivienda en Ciudad de México</em></h2>
  <div class="sec-rule"></div>
  <p class="sec-intro">Clusters no supervisados sobre atributos físicos, geográficos (lat/lon), socioeconómicos (NSE del municipio) y amenidades (conteo de POIs por <em>main_category</em> en buffer de 500 m). <b>Click en una tarjeta</b> para abrir el modal con histogramas, distribución NSE y top municipios del submercado. Dentro del modal, cada histograma tiene <b>+ Comparar</b> para superponer otros clusters con control de opacidad.</p>

  {cluster_html}

  <h2 class="sec"><span class="num-tag">§04</span>Plusvalía nominal · <em>2019 → presente</em></h2>
  <div class="sec-rule"></div>
  <p class="sec-intro">Variación porcentual de la mediana de m² construido respecto al primer año observado, por submercado (cluster K-Means). Asterisco = año con captura parcial. Series sin deflactar: una porción de la pendiente es inflación general, no apreciación real.</p>

  {plusvalia_html}

  <!-- Modal overlay compartido por cluster cards, top municipios y legendas.
       Va ANTES del bloque del explorador para que el script encuentre el botón
       de cerrar en el DOM al adjuntar el listener. -->
  <div id="modal-overlay" class="modal-overlay" role="dialog" aria-modal="true">
    <div class="modal" role="document">
      <header class="modal-header">
        <h2 id="modal-title">—</h2>
        <button id="modal-close" type="button" aria-label="Cerrar">×</button>
      </header>
      <div id="modal-body" class="modal-body"></div>
    </div>
  </div>

  {map_section}

  {predictor_section}

  {absorcion_section}

  <footer class="bp-footer">
    <div class="bp-footer-left">
      <img src="{logo_blackprint_light}" alt="BlackPrint" class="bp-footer-wordmark"/>
      <img src="{logo_orange}" alt="" class="bp-footer-orange" aria-hidden="true"/>
    </div>
    <div class="bp-footer-right">
      <span class="bp-footer-tagline">CDMX · <b>Módulo de Vivienda</b></span>
      <span class="bp-footer-meta">Generado {generated}</span>
    </div>
  </footer>

</div>

{cdmx_payload_script}
<script>{global_js}</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Script global: corre después del IIFE del mapa, reusa los helpers expuestos
# en window.__* para year-picker, modal de KPIs y comparativa de municipios.
# ---------------------------------------------------------------------------

GLOBAL_JS = r"""
(function() {
  var CDMX = (window.__CDMX || {});
  var openModal = window.__openModal;
  var renderHistSvg = window.__renderHistSvg;
  var fmtMxnShort = window.__fmtMxnShort;
  var fmtFor = window.__fmtFor;
  var esc = window.__esc || function(s){return String(s||'').replace(/[<>&]/g, function(c){return c==='<'?'&lt;':c==='>'?'&gt;':'&amp;';});};

  function fmtMxnFull(v) {
    if (v === null || v === undefined || isNaN(v)) return '—';
    return '$' + Math.round(v).toLocaleString('en-US');
  }
  function fmtPctSigned(v) {
    if (v === null || v === undefined || isNaN(v)) return '—';
    return (v >= 0 ? '+' : '') + v.toFixed(1) + '%';
  }

  // ============ Year picker → actualiza KPIs ============
  var currentYear = 'all';
  function applyYearToKpis(year) {
    var stats;
    if (year === 'all' || !CDMX.byYearStats || !CDMX.byYearStats[year]) {
      stats = {
        valor:   { median: CDMX.globalStats && CDMX.globalStats.valor   && CDMX.globalStats.valor.median,
                   p25:    CDMX.globalStats && CDMX.globalStats.valor   && CDMX.globalStats.valor.p25,
                   p75:    CDMX.globalStats && CDMX.globalStats.valor   && CDMX.globalStats.valor.p75 },
        m2_sv:   { median: CDMX.globalStats && CDMX.globalStats.m2_sv   && CDMX.globalStats.m2_sv.median },
        terreno: { median: CDMX.globalStats && CDMX.globalStats.terreno && CDMX.globalStats.terreno.median }
      };
    } else {
      stats = CDMX.byYearStats[year];
    }
    var v = (stats.valor || {});
    var m = (stats.m2_sv || {});
    var t = (stats.terreno || {});
    var elV = document.getElementById('kpi-valor');
    var elVs = document.getElementById('kpi-valor-sub');
    var elM = document.getElementById('kpi-m2sv');
    var elMs = document.getElementById('kpi-m2sv-sub');
    var elT = document.getElementById('kpi-terr');
    var elTs = document.getElementById('kpi-terr-sub');
    if (elV) elV.textContent = fmtMxnFull(v.median);
    if (elVs) elVs.textContent = (v.p25 !== undefined && v.p75 !== undefined && v.p25 !== null && v.p75 !== null)
      ? 'p25 ' + (fmtMxnShort?fmtMxnShort(v.p25):fmtMxnFull(v.p25)) + ' · p75 ' + (fmtMxnShort?fmtMxnShort(v.p75):fmtMxnFull(v.p75))
      : (year === 'all' ? 'mediana global' : 'mediana ' + year);
    if (elM) elM.textContent = fmtMxnFull(m.median);
    if (elMs) elMs.textContent = year === 'all' ? 'mediana m² vendible' : 'mediana m² vendible · ' + year;
    if (elT) elT.textContent = fmtMxnFull(t.median);
    if (elTs) elTs.textContent = year === 'all' ? 'mediana suelo' : 'mediana suelo · ' + year;
  }

  document.querySelectorAll('#year-picker .yp').forEach(function(btn) {
    btn.addEventListener('click', function() {
      document.querySelectorAll('#year-picker .yp').forEach(function(b){ b.classList.remove('on'); });
      btn.classList.add('on');
      currentYear = btn.dataset.year;
      applyYearToKpis(currentYear);
    });
  });

  // ============ KPI click → modal con histograma ============
  var METRIC_META = {
    valor:   { label: 'Valor de la vivienda (concluido)', unit: '$' },
    m2_sv:   { label: 'Costo por m² · construido',       unit: '$' },
    terreno: { label: 'Costo por m² · terreno',          unit: '$' }
  };
  // Paleta por año para overlays dentro del modal KPI.
  var YEAR_PALETTE = ['#c9421f','#1f6c5c','#b88a1f','#2a3d6e','#7b3b5e','#4d6b1d','#a44b1a','#0f4c75'];

  function openKpiModal(metric) {
    if (!openModal || !renderHistSvg) {
      console.warn('Modal helpers no disponibles (¿se renderizó el explorador?).');
      alert('El histograma necesita el explorador territorial activo (clustering).');
      return;
    }
    var meta = METRIC_META[metric] || { label: metric, unit: '' };
    var fmt  = fmtFor ? fmtFor(meta.unit) : function(v){ return Math.round(v).toLocaleString('en-US'); };
    var byYearHists = CDMX.byYearHists || {};
    var allYears = Object.keys(byYearHists)
                     .map(function(y){ return parseInt(y, 10); })
                     .filter(function(y){ return !isNaN(y); })
                     .sort(function(a,b){ return a-b; });

    // Estado de selección — usamos números para años y el string 'all' para
    // "todos los años". Inicializa con el año seleccionado en el year-picker.
    var selected = new Set();
    selected.add(currentYear === 'all' ? 'all' : parseInt(currentYear, 10));

    function getHistFor(yKey) {
      if (yKey === 'all') return (CDMX.globalHists || {})[metric];
      return ((CDMX.byYearHists || {})[yKey] || {})[metric];
    }
    function labelFor(yKey) { return yKey === 'all' ? 'todos los años' : String(yKey); }

    function buildBody() {
      var sortedSel = Array.from(selected).sort(function(a, b) {
        if (a === 'all') return -1; if (b === 'all') return 1;
        return a - b;
      });
      var series = [];
      sortedSel.forEach(function(yKey, idx) {
        var h = getHistFor(yKey);
        if (h) series.push({
          key:    yKey,
          label:  labelFor(yKey),
          color:  YEAR_PALETTE[idx % YEAR_PALETTE.length],
          hist:   h,
        });
      });

      // Chips para tildar/destildar años. El check inicial refleja `selected`.
      var chips = ['<label class="kpi-year-chip"><input type="checkbox" data-year="all"'
                 + (selected.has('all')?' checked':'')+'/> Todos</label>'];
      allYears.forEach(function(y) {
        chips.push('<label class="kpi-year-chip"><input type="checkbox" data-year="'+y+'"'
                 + (selected.has(y)?' checked':'')+'/> '+y+'</label>');
      });
      var picker = '<div class="kpi-year-picker">'
                 +   '<span class="kpi-year-lbl">Comparar años</span>'
                 +   chips.join('')
                 + '</div>';

      // Stats: si hay 1 sola serie usa el resumen single; si hay >1, leyenda
      // de medianas por serie debajo del overlay.
      var statsHtml = '';
      var histHtml  = '';
      if (!series.length) {
        histHtml = '<p class="empty">Sin años seleccionados.</p>';
      } else if (series.length === 1) {
        var s = series[0];
        statsHtml = '<div class="modal-stats">'
                  +   '<div class="ms"><span class="lbl">mediana</span><span class="val">' + fmt(s.hist.median) + '</span></div>'
                  +   '<div class="ms"><span class="lbl">promedio</span><span class="val">' + fmt(s.hist.mean)   + '</span></div>'
                  +   '<div class="ms"><span class="lbl">n</span><span class="val">' + (s.hist.n||0).toLocaleString('en-US') + '</span></div>'
                  + '</div>';
        histHtml = '<div class="hist-card">'
                 +   '<div class="hist-head"><span class="hist-title">' + esc(meta.label) + ' · ' + esc(s.label) + '</span></div>'
                 +   renderHistSvg(s.hist, 720, 260, fmt)
                 + '</div>';
      } else {
        var legendLis = series.map(function(s) {
          return '<li><span class="cmp-leg-sw" style="background:'+s.color+'"></span>'
               + '<span class="cmp-leg-name">'+esc(s.label)+'</span>'
               + '<span class="cmp-leg-val">med '+fmt(s.hist.median)+' · prom '+fmt(s.hist.mean)+' · n='+(s.hist.n||0).toLocaleString('en-US')+'</span></li>';
        }).join('');
        histHtml = '<div class="hist-card">'
                 +   '<div class="hist-head"><span class="hist-title">'+esc(meta.label)+' · superposición de años</span></div>'
                 +   buildOverlayHistogram(series, 720, 260, fmt)
                 +   '<div class="cmp-alpha"><label>Opacidad de capas <input type="range" id="kpi-alpha" min="0.10" max="0.90" step="0.05" value="0.45"/> <span id="kpi-alpha-val">0.45</span></label></div>'
                 +   '<ul class="cmp-leg">'+legendLis+'</ul>'
                 + '</div>';
      }
      return picker + statsHtml + histHtml;
    }

    function wireChips() {
      document.querySelectorAll('#modal-body .kpi-year-chip input').forEach(function(cb) {
        cb.addEventListener('change', function() {
          var raw = cb.dataset.year;
          var key = (raw === 'all') ? 'all' : parseInt(raw, 10);
          if (cb.checked) selected.add(key);
          else            selected.delete(key);
          // Garantizar al menos una selección — si todo se deseleccionó,
          // re-elige el año original.
          if (selected.size === 0) {
            cb.checked = true;
            selected.add(key);
            return;
          }
          refresh();
        });
      });
    }
    function wireAlpha() {
      var slider = document.getElementById('kpi-alpha');
      if (!slider) return;
      var apply = function() {
        var a = parseFloat(slider.value);
        document.querySelectorAll('#modal-body .cmp-layer').forEach(function(el) {
          el.setAttribute('fill-opacity', a.toFixed(2));
        });
        var lbl = document.getElementById('kpi-alpha-val');
        if (lbl) lbl.textContent = a.toFixed(2);
      };
      slider.addEventListener('input', apply);
      apply();
    }
    function refresh() {
      document.getElementById('modal-body').innerHTML = buildBody();
      wireChips();
      wireAlpha();
    }

    var title = '<span class="modal-tag" style="background:#1c1a17">$</span> ' + esc(meta.label);
    openModal(title, buildBody());
    wireChips();
    wireAlpha();
  }
  document.querySelectorAll('.kpi.clickable').forEach(function(btn) {
    btn.addEventListener('click', function() {
      openKpiModal(btn.dataset.metric);
    });
  });

  // ============ Comparativa de municipios (botón en §01) ============
  var muniBtn = document.getElementById('muni-compare-toggle');
  var muniRows = function() { return document.querySelectorAll('tr.muni-row'); };
  var muniMode = false;
  var muniSelected = new Set();
  var MAX_MUNI_CMP = 5;

  function refreshMuniBar() {
    var bar = document.getElementById('muni-cmp-bar');
    if (!bar) return;
    var n = muniSelected.size;
    bar.querySelector('.muni-cmp-count').textContent = n + ' / ' + MAX_MUNI_CMP;
    var go = bar.querySelector('.muni-cmp-go');
    go.disabled = n < 2;
  }
  function ensureMuniBar() {
    var bar = document.getElementById('muni-cmp-bar');
    if (bar) return bar;
    bar = document.createElement('div');
    bar.id = 'muni-cmp-bar';
    bar.className = 'muni-cmp-bar';
    bar.innerHTML = '<span>Comparación de municipios</span>'
                  + '<span class="muni-cmp-count">0 / ' + MAX_MUNI_CMP + '</span>'
                  + '<button type="button" class="muni-cmp-go" disabled>Abrir comparación</button>'
                  + '<button type="button" class="muni-cmp-clear">Limpiar</button>';
    var table = document.querySelector('.muni-table');
    if (table && table.parentNode) table.parentNode.insertBefore(bar, table.nextSibling);
    bar.querySelector('.muni-cmp-go').addEventListener('click', function() {
      openMuniCompareModal(Array.from(muniSelected));
    });
    bar.querySelector('.muni-cmp-clear').addEventListener('click', function() {
      muniSelected.clear();
      document.querySelectorAll('tr.muni-row.cmp-on').forEach(function(tr){ tr.classList.remove('cmp-on'); });
      refreshMuniBar();
    });
    return bar;
  }
  function setMuniMode(on) {
    muniMode = on;
    if (muniBtn) {
      muniBtn.classList.toggle('on', on);
      muniBtn.textContent = on ? '✕ Salir de comparación' : '+ Comparar municipios';
    }
    var table = document.querySelector('.muni-table');
    if (table) table.classList.toggle('cmp-mode', on);
    if (on) {
      ensureMuniBar().classList.add('open');
    } else {
      var bar = document.getElementById('muni-cmp-bar');
      if (bar) bar.classList.remove('open');
      muniSelected.clear();
      document.querySelectorAll('tr.muni-row.cmp-on').forEach(function(tr){ tr.classList.remove('cmp-on'); });
    }
  }
  if (muniBtn) muniBtn.addEventListener('click', function() { setMuniMode(!muniMode); });

  muniRows().forEach(function(tr) {
    tr.addEventListener('click', function(ev) {
      if (!muniMode) return;
      ev.stopPropagation();
      ev.preventDefault();
      var cve = tr.dataset.cve;
      if (muniSelected.has(cve)) {
        muniSelected.delete(cve);
        tr.classList.remove('cmp-on');
      } else {
        if (muniSelected.size >= MAX_MUNI_CMP) {
          tr.animate([{boxShadow:'0 0 0 0 rgba(201,66,31,0)'},{boxShadow:'0 0 0 6px rgba(201,66,31,0.45)'},{boxShadow:'0 0 0 0 rgba(201,66,31,0)'}], {duration:600});
          return;
        }
        muniSelected.add(cve);
        tr.classList.add('cmp-on');
      }
      refreshMuniBar();
    }, true);
  });

  // ============ Modal de comparación de municipios ============
  var MUNI_PALETTE = ['#c9421f', '#1f6c5c', '#b88a1f', '#2a3d6e', '#7b3b5e'];

  function alphaApply(modalRoot) {
    var slider = modalRoot.querySelector('#muni-alpha');
    if (!slider) return;
    var a = parseFloat(slider.value);
    modalRoot.querySelectorAll('.cmp-layer').forEach(function(el) {
      el.setAttribute('fill-opacity', a.toFixed(2));
    });
    var lbl = modalRoot.querySelector('#muni-alpha-val');
    if (lbl) lbl.textContent = a.toFixed(2);
  }

  function buildOverlayHistogram(series, w, hPx, fmt) {
    // series = [{label, color, hist}]. Bin width compartido (mismo edges en todos).
    var valid = series.filter(function(s){ return s.hist && s.hist.counts && s.hist.counts.length; });
    if (!valid.length) {
      return '<svg viewBox="0 0 '+w+' '+hPx+'" class="hist-svg"><text x="50%" y="50%" text-anchor="middle" style="font-family:Fraunces,serif;font-style:italic;font-size:11.5px;fill:#6a635a;">datos insuficientes</text></svg>';
    }
    var PL = 38, PR = 14, PT = 14, PB = 24;
    // rango común: min de todos los edges[0], max de todos los edges[last]
    var lo = Infinity, hi = -Infinity;
    valid.forEach(function(s){
      lo = Math.min(lo, s.hist.edges[0]);
      hi = Math.max(hi, s.hist.edges[s.hist.edges.length-1]);
    });
    if (hi <= lo) hi = lo + 1;
    var NBINS = 20;
    var step = (hi - lo) / NBINS;
    var edges = [];
    for (var i = 0; i <= NBINS; i++) edges.push(lo + step*i);
    // rebin cada serie sobre los edges comunes
    function rebin(h) {
      var counts = new Array(NBINS).fill(0);
      for (var i = 0; i < h.counts.length; i++) {
        var c = h.counts[i];
        if (!c) continue;
        var mid = (h.edges[i] + h.edges[i+1]) / 2;
        var idx = Math.floor((mid - lo) / step);
        if (idx < 0) idx = 0;
        if (idx >= NBINS) idx = NBINS - 1;
        counts[idx] += c;
      }
      return counts;
    }
    var rebinned = valid.map(function(s){ return rebin(s.hist); });
    var maxC = 1;
    rebinned.forEach(function(arr){ arr.forEach(function(c){ if (c > maxC) maxC = c; }); });
    var bw = (w - PL - PR) / NBINS;
    var layersSvg = '';
    rebinned.forEach(function(counts, sidx) {
      var color = valid[sidx].color;
      var label = valid[sidx].label;
      var bars = '';
      for (var i = 0; i < NBINS; i++) {
        var c = counts[i];
        if (!c) continue;
        var bh = (hPx - PT - PB) * c / maxC;
        var x = PL + i * bw;
        var y = hPx - PB - bh;
        bars += '<rect x="'+x.toFixed(2)+'" y="'+y.toFixed(2)+'" width="'+(bw-0.6).toFixed(2)+'" height="'+bh.toFixed(2)+'" fill="'+color+'" fill-opacity="0.45" class="cmp-layer"><title>'+esc(label)+' · ['+fmt(edges[i])+', '+fmt(edges[i+1])+'] · n='+c+'</title></rect>';
      }
      layersSvg += '<g data-label="'+esc(label)+'">' + bars + '</g>';
    });
    var axis = '<line x1="'+PL+'" y1="'+(hPx-PB)+'" x2="'+(w-PR)+'" y2="'+(hPx-PB)+'" class="hax"/>'
             + '<text x="'+PL+'" y="'+(hPx-PB+14)+'" class="hlbl">'+fmt(lo)+'</text>'
             + '<text x="'+(w-PR)+'" y="'+(hPx-PB+14)+'" class="hlbl" text-anchor="end">'+fmt(hi)+'</text>';
    return '<svg viewBox="0 0 '+w+' '+hPx+'" class="hist-svg">'+layersSvg+axis+'</svg>';
  }

  function openMuniCompareModal(cveList) {
    var data = (CDMX.muniCompare || {}).by_cve || {};
    var meta = (CDMX.muniCompare || {}).attrs || [];
    var picked = cveList.filter(function(c){ return !!data[c]; });
    if (picked.length < 2) {
      alert('Necesitas al menos 2 municipios con datos para comparar.');
      return;
    }
    var series = picked.slice(0, MAX_MUNI_CMP).map(function(cve, i) {
      return {
        cve: cve,
        name: (data[cve] && data[cve].name) || cve,
        color: MUNI_PALETTE[i % MUNI_PALETTE.length],
        n: (data[cve] && data[cve].n) || 0,
        hists: (data[cve] && data[cve].hists) || {}
      };
    });

    var chips = series.map(function(s){
      return '<span class="cmp-chip" style="background:'+s.color+'">'+esc(s.name)+' · n='+s.n.toLocaleString('en-US')+'</span>';
    }).join('');

    var title = '<span class="modal-tag" style="background:#1c1a17">Cmp</span> Comparación · ' + picked.length + ' municipios';

    var hists = '';
    meta.forEach(function(m) {
      var fmt = fmtFor ? fmtFor(m.unit) : function(v){ return Math.round(v).toLocaleString('en-US'); };
      var serForMetric = series.map(function(s){ return {label: s.name, color: s.color, hist: s.hists[m.attr]}; });
      var statsRow = series.map(function(s){
        var h = s.hists[m.attr];
        return '<li><span class="cmp-leg-sw" style="background:'+s.color+'"></span><span class="cmp-leg-name">'+esc(s.name)+'</span><span class="cmp-leg-val">'+(h?'med '+fmt(h.median):'—')+'</span></li>';
      }).join('');
      hists += '<div class="hist-card">'
            +    '<div class="hist-head"><span class="hist-title">'+esc(m.label)+'</span></div>'
            +    buildOverlayHistogram(serForMetric, 720, 220, fmt)
            +    '<ul class="cmp-leg">'+statsRow+'</ul>'
            + '</div>';
    });

    var body = '<div class="cmp-chips">'+chips+'</div>'
             + '<div class="cmp-alpha"><label>Opacidad de capas <input type="range" id="muni-alpha" min="0.10" max="0.90" step="0.05" value="0.45"/> <span id="muni-alpha-val">0.45</span></label></div>'
             + '<div class="cmp-hists">'+hists+'</div>';

    openModal(title, body);
    var modalRoot = document.getElementById('modal-overlay');
    var slider = modalRoot.querySelector('#muni-alpha');
    if (slider) slider.addEventListener('input', function(){ alphaApply(modalRoot); });
    alphaApply(modalRoot);
  }

  // expone por si otros scripts necesitan abrirlo
  window.__openMuniCompareModal = openMuniCompareModal;

  // Defaults
  applyYearToKpis('all');
})();
"""



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cluster", action="store_true",
                    help="Omite K-Means (solo descriptiva + serie de toda la muestra).")
    ap.add_argument("--out", default="eda_cdmx.html")
    ap.add_argument("--dump-ddl", action="store_true",
                    help="Imprime el DDL sugerido a stdout y termina. No toca la DB.")
    args = ap.parse_args()

    if args.dump_ddl:
        print("-- DDL sugerido. NO se ejecuta automáticamente.")
        print("-- Copia y pega en tu cliente SQL si decides materializar.")
        print(DDL_FACT_AVALUOS_CDMX)
        return

    print("[1/8] Conectando (sesión read-only)…")
    conn = connect()

    try:
        print("[2/8] Cargando datos…")
        df = load_data(conn)
        df = winsorize_block(df)

        print("[3/8] Cargando POIs / NSE / geometrías…")
        pois = load_pois(conn)
        nse  = load_nse_muni(conn)
        muni_geoms = load_muni_geometries(conn)

        print("[4/8] Adjuntando POIs (buffer 500m) y NSE a cada vivienda…")
        df, poi_cols = attach_pois(df, pois, top_n=POI_TOP_N, buffer_m=POI_BUFFER_M)
        df = attach_nse(df, nse)
        # Estandarizar nse_score (algunas alcaldías pueden tener n/d → NaN); NaN
        # se dropea en cluster_kmeans junto con cualquier otra feature ausente.

        print("[5/8] Descriptiva…")
        stats = describe(df)
        print(f"      · mediana valor: {fmt_mxn(stats.median_valor)}")
        print(f"      · mediana $/m²:  {fmt_mxn(stats.median_m2_sv)}")

        cluster: ClusterResult | None = None
        if not args.skip_cluster:
            print("[6/8] K-Means (físicas + geo + POIs + NSE)…")
            feature_cols = PHYSICAL_FEATURES + GEO_FEATURES + NSE_FEATURES + poi_cols
            print(f"      · features ({len(feature_cols)}): {feature_cols}")
            cluster = cluster_kmeans(df, feature_cols)
            if cluster is None:
                print("      ⚠ no se pudo correr clustering (datos insuficientes).")
            else:
                print(f"      · k = {cluster.k}")
                for cid, name in cluster.cluster_names.items():
                    n = int(cluster.profile.loc[cluster.profile["_cluster"] == cid, "n"].iloc[0])
                    print(f"        C{cid} (n={n:,}): {name}")

        print("[7/8] Serie temporal de plusvalía + precómputos para modales…")
        plusvalia = timeseries_plusvalia(df, cluster.labels if cluster else None)
        cluster_data  = compute_cluster_data(df, cluster)
        alcaldia_data = compute_alcaldia_data(df, muni_geoms, cluster)
        regression    = train_price_regression(df, cluster)
        if regression:
            print(f"      · regresión log-precio entrenada: "
                  f"R²={regression['r2']:.3f}, "
                  f"σ={regression['sigma']:.3f} (log MXN), "
                  f"n={regression['n_train']:,}, "
                  f"{len(regression['features'])} features.")
        else:
            print("      ⚠ no se pudo entrenar la regresión.")
        print(f"      · {len(cluster_data)} clusters con histogramas, "
              f"{len(alcaldia_data.get('by_cve', {}))} alcaldías con geometría.")

        print("[8/8] Preparando explorador territorial + mapa de atributos físicos + absorción…")
        map_data = prepare_map_points(df, cluster)
        print(f"      · {len(map_data['points']):,} puntos en el explorador "
              f"(de {map_data['n_total_geo']:,} elegibles).")
        years_sorted = sorted([int(y) for y in df["ano"].dropna().unique().tolist()])
        clases_sorted = sorted([int(c) for c in df["clase"].dropna().unique().tolist()])
        map_section = render_map_section(
            map_data, cluster, PARTIAL_YEAR,
            cluster_data=cluster_data,
            alcaldia_data=alcaldia_data,
            years=years_sorted,
            clases=clases_sorted,
            regression=regression,
        )
        predictor_section = render_predictor_section(cluster)

        # §02: nuevo mapa de Atributos Físicos
        phys_data = prepare_phys_attr_points(df)
        print(f"      · {len(phys_data['points']):,} puntos con lat/lon para mapa de atributos físicos.")
        phys_map_section = render_phys_attr_map_section(phys_data, years_sorted, clases_sorted)

        # §07: absorción por desarrollador
        absorcion_data = compute_absorcion_data(df)
        print(f"      · {len(absorcion_data['developers']):,} desarrolladores top, "
              f"{len(absorcion_data['rows']):,} avalúos con constructor.")
        absorcion_section = render_absorcion_section(absorcion_data, clases_sorted)

        # Datos para comparación de municipios (histogramas compactos por métrica)
        muni_compare_data = build_muni_compare_data(alcaldia_data)

        logo_svg = render_logo_svg()

        print("Renderizando HTML…")
        render_html(
            stats=stats,
            cluster=cluster,
            plusvalia=plusvalia,
            out_path=args.out,
            generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
            logo_svg=logo_svg,
            map_section=map_section,
            predictor_section=predictor_section,
            phys_map_section=phys_map_section,
            absorcion_section=absorcion_section,
            muni_compare_data=muni_compare_data,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
