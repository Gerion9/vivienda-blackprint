#!/usr/bin/env python3
"""
EDA standalone para blackprint_db_prd.staging.stg_avaluos
=========================================================

Uso:
    export BLACKPRINT_DB_PASSWORD='...'
    python eda_avaluos.py                       # muestra 50k, sin cruces
    python eda_avaluos.py --full                # tabla completa
    python eda_avaluos.py --sample 200000       # muestra custom
    python eda_avaluos.py --crosses             # cruzar con dim_mexico_geometries / NSE si hay geo
    python eda_avaluos.py --out reporte.html    # ruta de salida

Estrategia:
    Fase 1: inventario (information_schema + COUNT + 5 filas de muestra)
    Fase 2: perfil por columna (cardinalidad, nulos, stats numéricas, top categóricos, rango temporal)
    Fase 3: cruces opcionales contra presentation.* si detecta lat/lon o cvegeo
    Fase 4: render HTML self-contained (sin CDNs, SVGs inline para histogramas)
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd
import psycopg2
from psycopg2 import sql

# ---------------------------------------------------------------------------
# Conexión
# ---------------------------------------------------------------------------

DB_HOST = "35.211.253.113"
DB_PORT = 5432
DB_NAME = "blackprint_db_prd"
DB_USER = "josue_user"
SCHEMA = "staging"
TABLE = "stg_avaluos"
FQ_TABLE = f"{SCHEMA}.{TABLE}"

# Categorización de tipos pg → bucket de análisis
NUMERIC_PG = {
    "smallint", "integer", "bigint", "decimal", "numeric", "real",
    "double precision", "money",
}
TEXT_PG = {"text", "character varying", "character", "varchar", "char", "citext", "uuid"}
DATE_PG = {"date", "timestamp without time zone", "timestamp with time zone", "timestamptz"}
BOOL_PG = {"boolean"}
JSON_PG = {"json", "jsonb"}
GEO_PG = {"geometry", "geography"}

# Heurística para detectar columnas geográficas por nombre cuando el tipo es numérico
LAT_HINTS = {"lat", "latitude", "latitud", "y_coord", "y"}
LON_HINTS = {"lon", "lng", "long", "longitude", "longitud", "x_coord", "x"}
CVEGEO_HINTS = {"cvegeo", "cve_geo", "geo_code", "geocode"}


def connect():
    pwd = os.environ.get("BLACKPRINT_DB_PASSWORD")
    if not pwd:
        sys.exit("ERROR: BLACKPRINT_DB_PASSWORD no está en el ambiente.")
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, dbname=DB_NAME,
        user=DB_USER, password=pwd, connect_timeout=10,
    )
    with conn.cursor() as cur:
        cur.execute("SET statement_timeout = '300s'")
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Dataclasses para llevar el resultado por columna
# ---------------------------------------------------------------------------

@dataclass
class ColumnProfile:
    name: str
    pg_type: str
    bucket: str                       # 'numeric' | 'text' | 'date' | 'bool' | 'json' | 'geo' | 'other'
    nullable: bool
    n_total: int = 0
    n_null: int = 0
    n_distinct: int | None = None
    # numéricos
    nmin: float | None = None
    nmax: float | None = None
    nmean: float | None = None
    nstd: float | None = None
    nmedian: float | None = None
    np95: float | None = None
    nzero: int | None = None
    histogram: list[tuple[float, float, int]] = field(default_factory=list)
    # categóricos / texto
    top: list[tuple[Any, int]] = field(default_factory=list)
    avg_len: float | None = None
    # fechas
    dmin: str | None = None
    dmax: str | None = None
    monthly: list[tuple[str, int]] = field(default_factory=list)
    # flags
    looks_like_lat: bool = False
    looks_like_lon: bool = False
    looks_like_cvegeo: bool = False
    cvegeo_lengths: dict[int, int] = field(default_factory=dict)

    @property
    def pct_null(self) -> float:
        return 0.0 if self.n_total == 0 else 100 * self.n_null / self.n_total


# ---------------------------------------------------------------------------
# Fase 1: inventario estructural
# ---------------------------------------------------------------------------

def fetch_schema(conn) -> list[ColumnProfile]:
    q = """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = %s AND table_name = %s
        ORDER BY ordinal_position
    """
    with conn.cursor() as cur:
        cur.execute(q, (SCHEMA, TABLE))
        rows = cur.fetchall()
    if not rows:
        sys.exit(f"ERROR: tabla {FQ_TABLE} no existe o sin permisos.")
    cols: list[ColumnProfile] = []
    for name, dtype, nullable in rows:
        bucket = classify(dtype, name)
        cp = ColumnProfile(
            name=name, pg_type=dtype, bucket=bucket,
            nullable=(nullable == "YES"),
        )
        lname = name.lower()
        if lname in LAT_HINTS:
            cp.looks_like_lat = True
        if lname in LON_HINTS:
            cp.looks_like_lon = True
        if lname in CVEGEO_HINTS or "cvegeo" in lname:
            cp.looks_like_cvegeo = True
        cols.append(cp)
    return cols


def classify(pg_type: str, name: str) -> str:
    if pg_type in NUMERIC_PG: return "numeric"
    if pg_type in TEXT_PG: return "text"
    if pg_type in DATE_PG: return "date"
    if pg_type in BOOL_PG: return "bool"
    if pg_type in JSON_PG: return "json"
    if pg_type in GEO_PG: return "geo"
    return "other"


def fetch_row_count(conn) -> int:
    with conn.cursor() as cur:
        cur.execute(f"SELECT COUNT(*) FROM {FQ_TABLE}")
        return cur.fetchone()[0]


def fetch_table_size(conn) -> str:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_size_pretty(pg_total_relation_size(%s))", (FQ_TABLE,))
        return cur.fetchone()[0]


def fetch_sample_rows(conn, n: int = 5) -> pd.DataFrame:
    return pd.read_sql(f"SELECT * FROM {FQ_TABLE} LIMIT {n}", conn)


# ---------------------------------------------------------------------------
# Fase 2: muestra y perfilado
# ---------------------------------------------------------------------------

def fetch_sample(conn, n_sample: int | None) -> pd.DataFrame:
    if n_sample is None:
        print(f"  → leyendo tabla completa de {FQ_TABLE}…")
        return pd.read_sql(f"SELECT * FROM {FQ_TABLE}", conn)
    # TABLESAMPLE no siempre disponible en views; uso ORDER BY random() acotado.
    # Para >100k filas, ORDER BY random() es caro; uso bernoulli si la tabla es base.
    print(f"  → muestra aleatoria de {n_sample:,} filas…")
    q = f"SELECT * FROM {FQ_TABLE} ORDER BY random() LIMIT {n_sample}"
    return pd.read_sql(q, conn)


def profile_column(df: pd.DataFrame, cp: ColumnProfile) -> None:
    s = df[cp.name]
    cp.n_total = len(s)
    cp.n_null = int(s.isna().sum())

    if cp.bucket == "numeric":
        profile_numeric(s, cp)
    elif cp.bucket == "text":
        profile_text(s, cp)
    elif cp.bucket == "date":
        profile_date(s, cp)
    elif cp.bucket == "bool":
        profile_bool(s, cp)
    elif cp.bucket == "json":
        cp.n_distinct = s.astype(str).nunique(dropna=True)
    else:
        cp.n_distinct = s.astype(str).nunique(dropna=True)

    # post: detectar cvegeo por contenido aunque no se llamara así
    if cp.bucket == "text" and not cp.looks_like_cvegeo:
        non_null = s.dropna().astype(str)
        if len(non_null) > 0:
            lens = non_null.str.len()
            digits = non_null.str.match(r"^\d+$").mean()
            common = set(lens.value_counts().head(3).index.tolist())
            inegi_lengths = {2, 5, 9, 13, 14, 17}
            if digits > 0.9 and (common & inegi_lengths):
                cp.looks_like_cvegeo = True
                cp.cvegeo_lengths = lens.value_counts().head(5).to_dict()


def profile_numeric(s: pd.Series, cp: ColumnProfile) -> None:
    s = pd.to_numeric(s, errors="coerce")
    nn = s.dropna()
    cp.n_distinct = int(nn.nunique())
    if len(nn) == 0:
        return
    cp.nmin = float(nn.min())
    cp.nmax = float(nn.max())
    cp.nmean = float(nn.mean())
    cp.nstd = float(nn.std()) if len(nn) > 1 else 0.0
    cp.nmedian = float(nn.median())
    cp.np95 = float(nn.quantile(0.95))
    cp.nzero = int((nn == 0).sum())
    # histograma sin matplotlib: 24 bins entre p1 y p99 para evitar outliers brutales
    lo, hi = float(nn.quantile(0.01)), float(nn.quantile(0.99))
    if hi <= lo:
        lo, hi = cp.nmin, cp.nmax if cp.nmax > cp.nmin else cp.nmin + 1
    if hi > lo:
        try:
            cuts = pd.cut(nn.clip(lo, hi), bins=24, include_lowest=True)
            counts = cuts.value_counts().sort_index()
            cp.histogram = [
                (float(iv.left), float(iv.right), int(c))
                for iv, c in counts.items()
            ]
        except Exception:
            cp.histogram = []
    # detectar lat/lon por rango si nombre no fue evidente
    if not cp.looks_like_lat and not cp.looks_like_lon:
        if -90 <= cp.nmin and cp.nmax <= 90 and cp.np95 < 35 and cp.nmedian > 14:
            cp.looks_like_lat = True   # México lat range
        elif -180 <= cp.nmin and cp.nmax <= 0 and cp.nmedian < -85 and cp.nmedian > -120:
            cp.looks_like_lon = True


def profile_text(s: pd.Series, cp: ColumnProfile) -> None:
    nn = s.dropna().astype(str)
    cp.n_distinct = int(nn.nunique())
    if len(nn) == 0:
        return
    cp.avg_len = float(nn.str.len().mean())
    vc = nn.value_counts().head(15)
    cp.top = [(k, int(v)) for k, v in vc.items()]


def profile_date(s: pd.Series, cp: ColumnProfile) -> None:
    d = pd.to_datetime(s, errors="coerce", utc=False)
    nn = d.dropna()
    cp.n_distinct = int(nn.nunique())
    if len(nn) == 0:
        return
    cp.dmin = nn.min().isoformat()
    cp.dmax = nn.max().isoformat()
    monthly = (
        nn.dt.to_period("M").value_counts().sort_index().tail(36)
    )
    cp.monthly = [(str(p), int(c)) for p, c in monthly.items()]


def profile_bool(s: pd.Series, cp: ColumnProfile) -> None:
    nn = s.dropna()
    cp.n_distinct = int(nn.nunique())
    vc = nn.value_counts()
    cp.top = [(str(k), int(v)) for k, v in vc.items()]


# ---------------------------------------------------------------------------
# Fase 3: cruces opcionales con presentation.*
# ---------------------------------------------------------------------------

def run_crosses(conn, df: pd.DataFrame, cols: list[ColumnProfile]) -> dict[str, Any]:
    """Si encontramos lat/lon o cvegeo, intentamos un crosswalk barato."""
    crosses: dict[str, Any] = {}
    lat_col = next((c.name for c in cols if c.looks_like_lat), None)
    lon_col = next((c.name for c in cols if c.looks_like_lon), None)
    cvegeo_cols = [c for c in cols if c.looks_like_cvegeo]

    if lat_col and lon_col:
        # Cobertura por estado: contamos filas dentro de bbox de cada cve_ent
        # Para evitar reverse-geocode caro, agrupamos por bbox aproximado vía dim_mexico_geometries.
        # Atajo barato: bucketizar lat/lon a 0.5° y contar.
        print("  → coverage grid (lat/lon → grid 0.5°)…")
        sub = df[[lat_col, lon_col]].dropna().copy()
        sub[lat_col] = pd.to_numeric(sub[lat_col], errors="coerce")
        sub[lon_col] = pd.to_numeric(sub[lon_col], errors="coerce")
        sub = sub.dropna()
        if len(sub):
            sub["lat_b"] = (sub[lat_col] * 2).round() / 2
            sub["lon_b"] = (sub[lon_col] * 2).round() / 2
            grid = (
                sub.groupby(["lat_b", "lon_b"])
                .size().reset_index(name="n")
                .sort_values("n", ascending=False)
                .head(50)
            )
            crosses["lat_col"] = lat_col
            crosses["lon_col"] = lon_col
            crosses["bbox"] = {
                "lat_min": float(sub[lat_col].min()),
                "lat_max": float(sub[lat_col].max()),
                "lon_min": float(sub[lon_col].min()),
                "lon_max": float(sub[lon_col].max()),
            }
            crosses["grid"] = grid.to_dict(orient="records")

    for cp in cvegeo_cols:
        s = df[cp.name].dropna().astype(str)
        if not len(s):
            continue
        lens = s.str.len().value_counts().head(5).to_dict()
        # validez: % de las longitudes que matchean INEGI {2,5,9,13,14,17}
        valid = s.str.len().isin([2, 5, 9, 13, 14, 17]).mean()
        crosses[f"cvegeo_{cp.name}"] = {
            "lengths": {int(k): int(v) for k, v in lens.items()},
            "valid_pct": round(100 * float(valid), 2),
        }

    return crosses


# ---------------------------------------------------------------------------
# Fase 4: render HTML
# ---------------------------------------------------------------------------

def render_html(
    cols: list[ColumnProfile],
    n_rows: int,
    table_size: str,
    sample_size: int,
    sample_df: pd.DataFrame,
    crosses: dict[str, Any],
    out_path: str,
) -> None:
    bucket_counts: dict[str, int] = {}
    for c in cols:
        bucket_counts[c.bucket] = bucket_counts.get(c.bucket, 0) + 1

    bucket_chips = " ".join(
        f'<span class="chip chip-{b}">{b}: {n}</span>'
        for b, n in sorted(bucket_counts.items(), key=lambda kv: -kv[1])
    )

    # --- tabla de columnas ---
    rows_html = []
    for c in cols:
        flags = []
        if c.looks_like_lat: flags.append("LAT")
        if c.looks_like_lon: flags.append("LON")
        if c.looks_like_cvegeo: flags.append("CVEGEO?")
        flag_html = " ".join(f'<span class="flag">{f}</span>' for f in flags)
        rows_html.append(f"""
            <tr>
              <td class="mono">{html.escape(c.name)}</td>
              <td class="mono dim">{html.escape(c.pg_type)}</td>
              <td><span class="chip chip-{c.bucket}">{c.bucket}</span> {flag_html}</td>
              <td class="num">{c.n_total:,}</td>
              <td class="num">{c.n_null:,} <span class="dim">({c.pct_null:.1f}%)</span></td>
              <td class="num">{c.n_distinct if c.n_distinct is not None else '—'}</td>
            </tr>
        """)
    columns_table = "\n".join(rows_html)

    # --- detalle por columna ---
    detail_blocks = []
    for c in cols:
        detail_blocks.append(render_detail(c))
    details_html = "\n".join(detail_blocks)

    # --- muestra de filas ---
    sample_html = sample_df.head(5).to_html(
        classes="sample-table", border=0, index=False, na_rep="—"
    )

    # --- cruces ---
    crosses_html = render_crosses(crosses) if crosses else ""

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html_doc = HTML_TEMPLATE.format(
        table=FQ_TABLE,
        generated=generated,
        n_rows=f"{n_rows:,}",
        table_size=table_size,
        n_cols=len(cols),
        sample_size=f"{sample_size:,}",
        bucket_chips=bucket_chips,
        columns_table=columns_table,
        details_html=details_html,
        sample_html=sample_html,
        crosses_html=crosses_html,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html_doc)
    print(f"\n✓ Reporte escrito en {out_path}")


def render_detail(c: ColumnProfile) -> str:
    body = ""
    if c.bucket == "numeric" and c.nmin is not None:
        body += f"""
          <div class="stats-grid">
            <div><span class="lbl">min</span><span class="val">{fmt_num(c.nmin)}</span></div>
            <div><span class="lbl">p50</span><span class="val">{fmt_num(c.nmedian)}</span></div>
            <div><span class="lbl">mean</span><span class="val">{fmt_num(c.nmean)}</span></div>
            <div><span class="lbl">p95</span><span class="val">{fmt_num(c.np95)}</span></div>
            <div><span class="lbl">max</span><span class="val">{fmt_num(c.nmax)}</span></div>
            <div><span class="lbl">σ</span><span class="val">{fmt_num(c.nstd)}</span></div>
            <div><span class="lbl">zeros</span><span class="val">{c.nzero:,}</span></div>
            <div><span class="lbl">nulls</span><span class="val">{c.n_null:,} ({c.pct_null:.1f}%)</span></div>
          </div>
        """
        body += svg_histogram(c.histogram)
    elif c.bucket == "text" and c.top:
        body += f'<div class="meta">distinct: {c.n_distinct:,} · avg len: {c.avg_len:.1f} · nulls: {c.pct_null:.1f}%</div>'
        body += render_top(c.top)
        if c.looks_like_cvegeo and c.cvegeo_lengths:
            lens_html = " ".join(
                f'<span class="chip">{int(k)}c: {v:,}</span>'
                for k, v in c.cvegeo_lengths.items()
            )
            body += f'<div class="cvegeo-lens">cvegeo lengths: {lens_html}</div>'
    elif c.bucket == "date" and c.dmin:
        body += f"""
          <div class="meta">range: <b>{c.dmin}</b> → <b>{c.dmax}</b> · distinct: {c.n_distinct:,} · nulls: {c.pct_null:.1f}%</div>
        """
        body += svg_timeseries(c.monthly)
    elif c.bucket == "bool" and c.top:
        body += render_top(c.top)
    else:
        body += f'<div class="meta dim">distinct: {c.n_distinct if c.n_distinct is not None else "—"} · nulls: {c.pct_null:.1f}%</div>'

    return f"""
      <article class="col-detail" id="col-{html.escape(c.name)}">
        <header>
          <h3 class="mono">{html.escape(c.name)}</h3>
          <span class="chip chip-{c.bucket}">{c.bucket}</span>
          <span class="mono dim">{html.escape(c.pg_type)}</span>
        </header>
        {body}
      </article>
    """


def render_top(top: list[tuple[Any, int]]) -> str:
    if not top:
        return ""
    total = sum(v for _, v in top) or 1
    rows = []
    for k, v in top:
        pct = 100 * v / total
        label = html.escape(str(k))[:80]
        rows.append(f"""
          <li>
            <span class="bar" style="--w:{pct:.1f}%"></span>
            <span class="bar-label mono">{label}</span>
            <span class="bar-count">{v:,}</span>
          </li>
        """)
    return f'<ul class="bars">{"".join(rows)}</ul>'


def svg_histogram(bins: list[tuple[float, float, int]]) -> str:
    if not bins:
        return ""
    W, H, PAD = 560, 110, 4
    max_c = max(c for _, _, c in bins) or 1
    bw = (W - 2 * PAD) / len(bins)
    bars = []
    for i, (lo, hi, c) in enumerate(bins):
        h = (H - 20) * c / max_c
        x = PAD + i * bw
        y = H - 10 - h
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bw - 1:.2f}" height="{h:.2f}" '
            f'class="hbar"><title>[{fmt_num(lo)}, {fmt_num(hi)}]: {c:,}</title></rect>'
        )
    lo0 = bins[0][0]
    hi_last = bins[-1][1]
    return f"""
    <svg class="hist" viewBox="0 0 {W} {H}" preserveAspectRatio="none">
      {''.join(bars)}
      <line x1="{PAD}" y1="{H-10}" x2="{W-PAD}" y2="{H-10}" class="axis"/>
      <text x="{PAD}" y="{H-1}" class="axlbl">{fmt_num(lo0)}</text>
      <text x="{W-PAD}" y="{H-1}" class="axlbl" text-anchor="end">{fmt_num(hi_last)}</text>
    </svg>
    """


def svg_timeseries(monthly: list[tuple[str, int]]) -> str:
    if not monthly:
        return ""
    W, H, PAD = 560, 110, 6
    max_c = max(c for _, c in monthly) or 1
    bw = (W - 2 * PAD) / len(monthly)
    bars = []
    for i, (p, c) in enumerate(monthly):
        h = (H - 22) * c / max_c
        x = PAD + i * bw
        y = H - 12 - h
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bw - 1:.2f}" height="{h:.2f}" '
            f'class="hbar"><title>{p}: {c:,}</title></rect>'
        )
    return f"""
    <svg class="hist" viewBox="0 0 {W} {H}" preserveAspectRatio="none">
      {''.join(bars)}
      <line x1="{PAD}" y1="{H-12}" x2="{W-PAD}" y2="{H-12}" class="axis"/>
      <text x="{PAD}" y="{H-2}" class="axlbl">{monthly[0][0]}</text>
      <text x="{W-PAD}" y="{H-2}" class="axlbl" text-anchor="end">{monthly[-1][0]}</text>
    </svg>
    """


def render_crosses(crosses: dict[str, Any]) -> str:
    parts = ['<section class="crosses"><h2>Cruces detectados</h2>']
    if "bbox" in crosses:
        b = crosses["bbox"]
        parts.append(f"""
        <div class="cross-card">
          <h3>Geo · ({crosses['lat_col']}, {crosses['lon_col']})</h3>
          <div class="meta">
            bbox: lat [{b['lat_min']:.4f}, {b['lat_max']:.4f}]
                · lon [{b['lon_min']:.4f}, {b['lon_max']:.4f}]
          </div>
        """)
        # tabla del grid top-50
        rows = "".join(
            f"<tr><td class='num'>{r['lat_b']:.2f}</td>"
            f"<td class='num'>{r['lon_b']:.2f}</td>"
            f"<td class='num'>{r['n']:,}</td></tr>"
            for r in crosses["grid"][:25]
        )
        parts.append(f"""
          <table class="sample-table">
            <thead><tr><th>lat (0.5°)</th><th>lon (0.5°)</th><th>n filas</th></tr></thead>
            <tbody>{rows}</tbody>
          </table>
        </div>
        """)
    for k, v in crosses.items():
        if not k.startswith("cvegeo_"):
            continue
        col = k.replace("cvegeo_", "")
        lens = " ".join(f'<span class="chip">{ln}c: {n:,}</span>' for ln, n in v["lengths"].items())
        parts.append(f"""
        <div class="cross-card">
          <h3 class="mono">{col}</h3>
          <div class="meta">
            INEGI-valid length: <b>{v['valid_pct']}%</b><br>
            distribución: {lens}
          </div>
        </div>
        """)
    parts.append("</section>")
    return "\n".join(parts)


def fmt_num(x: float | None) -> str:
    if x is None: return "—"
    ax = abs(x)
    if ax == 0: return "0"
    if ax >= 1e9: return f"{x/1e9:.2f}B"
    if ax >= 1e6: return f"{x/1e6:.2f}M"
    if ax >= 1e3: return f"{x/1e3:.2f}k"
    if ax >= 1:   return f"{x:,.2f}"
    return f"{x:.4g}"


# ---------------------------------------------------------------------------
# HTML template (self-contained, no CDNs)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8"/>
<title>EDA · {table}</title>
<style>
  :root {{
    --bg: #f4f1ea;
    --paper: #fbf8f2;
    --ink: #1a1a1a;
    --ink-soft: #5a564f;
    --line: #2a2722;
    --accent: #c9421f;
    --accent-2: #2f5d3a;
    --gold: #b88a1f;
    --num: #2a3d6e;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; background: var(--bg); color: var(--ink); }}
  body {{
    font-family: 'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif;
    font-size: 15px; line-height: 1.55;
    background-image:
      radial-gradient(rgba(0,0,0,0.025) 1px, transparent 1px);
    background-size: 4px 4px;
  }}
  .mono {{ font-family: 'IBM Plex Mono', 'Courier New', ui-monospace, monospace; font-size: 0.92em; }}
  .container {{ max-width: 1180px; margin: 0 auto; padding: 48px 32px 96px; }}
  header.hero {{
    border-bottom: 3px double var(--line);
    padding-bottom: 24px; margin-bottom: 36px;
    display: grid; grid-template-columns: 1fr auto; gap: 24px; align-items: end;
  }}
  header.hero .titles h1 {{
    margin: 0; font-size: 44px; line-height: 1; letter-spacing: -0.02em;
    font-weight: 900;
  }}
  header.hero .titles h1::before {{
    content: "§"; color: var(--accent); margin-right: 8px; font-weight: 400;
  }}
  header.hero .titles p {{ margin: 6px 0 0; color: var(--ink-soft); }}
  header.hero .meta-block {{
    text-align: right; font-size: 12px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--ink-soft);
  }}
  .kpis {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 0;
    border: 1px solid var(--line); background: var(--paper);
    margin-bottom: 36px;
  }}
  .kpi {{
    padding: 18px 20px; border-right: 1px solid var(--line);
  }}
  .kpi:last-child {{ border-right: none; }}
  .kpi .k {{ font-size: 10.5px; letter-spacing: 0.18em; text-transform: uppercase; color: var(--ink-soft); }}
  .kpi .v {{ font-size: 28px; font-weight: 800; margin-top: 4px; color: var(--ink); }}
  .kpi .v .unit {{ font-size: 13px; color: var(--ink-soft); font-weight: 400; margin-left: 4px;}}

  h2 {{
    font-size: 13px; letter-spacing: 0.22em; text-transform: uppercase;
    border-bottom: 1px solid var(--line); padding-bottom: 6px;
    margin: 48px 0 16px;
    display: flex; align-items: center; gap: 12px;
  }}
  h2::before {{ content: "❖"; color: var(--accent); }}

  /* chips por bucket de tipo */
  .chip {{
    display: inline-block; padding: 1px 8px; border-radius: 2px;
    font-size: 10.5px; letter-spacing: 0.06em; text-transform: uppercase;
    background: #e9e3d4; color: var(--ink); font-family: 'IBM Plex Mono', monospace;
    border: 1px solid #d6cfb9;
  }}
  .chip-numeric {{ background: #dde6f3; border-color: #b9c5dd; color: var(--num); }}
  .chip-text    {{ background: #ede0d4; border-color: #d6c3a8; color: #6a3d1c; }}
  .chip-date    {{ background: #dceadd; border-color: #b8d2bc; color: var(--accent-2); }}
  .chip-bool    {{ background: #f3e3d2; border-color: #e0c39a; color: #8a4a14; }}
  .chip-json    {{ background: #e4dde7; border-color: #c8bccc; color: #5a3b6e; }}
  .chip-geo     {{ background: #f5d8d0; border-color: #d8a392; color: var(--accent); }}
  .flag {{
    display:inline-block; padding: 1px 6px; border:1px dashed var(--accent);
    color: var(--accent); font-size: 10px; letter-spacing: 0.1em;
    text-transform: uppercase; margin-left: 4px;
  }}

  /* tabla principal de columnas */
  table.cols {{
    width: 100%; border-collapse: collapse; background: var(--paper);
    border: 1px solid var(--line); font-size: 13.5px;
  }}
  table.cols th, table.cols td {{
    text-align: left; padding: 8px 12px; border-bottom: 1px dotted #c8c2b1;
  }}
  table.cols th {{
    background: #ece6d6; font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.14em;
    border-bottom: 1px solid var(--line);
  }}
  table.cols tr:hover td {{ background: #f0eadb; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; font-family: 'IBM Plex Mono', monospace; }}
  .dim {{ color: var(--ink-soft); }}

  /* detalle por columna */
  .details-grid {{
    display: grid; grid-template-columns: repeat(2, 1fr); gap: 18px;
  }}
  @media (max-width: 800px) {{ .details-grid {{ grid-template-columns: 1fr; }} }}
  .col-detail {{
    background: var(--paper); border: 1px solid var(--line);
    padding: 16px 18px;
    box-shadow: 4px 4px 0 var(--line);
  }}
  .col-detail header {{
    display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap;
    border-bottom: 1px solid #d6cfb9; padding-bottom: 8px; margin-bottom: 12px;
  }}
  .col-detail h3 {{ margin: 0; font-size: 16px; font-weight: 700; }}
  .meta {{ font-size: 12.5px; color: var(--ink-soft); margin-bottom: 8px; }}
  .meta b {{ color: var(--ink); }}
  .stats-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 4px 12px;
    font-size: 12.5px; margin-bottom: 8px;
  }}
  .stats-grid > div {{ display:flex; flex-direction:column; }}
  .stats-grid .lbl {{ font-size: 10px; letter-spacing: 0.12em; text-transform: uppercase; color: var(--ink-soft); }}
  .stats-grid .val {{ font-family: 'IBM Plex Mono', monospace; font-weight: 600; }}

  svg.hist {{ width: 100%; height: 110px; display: block; margin-top: 4px; }}
  svg.hist .hbar {{ fill: var(--accent); fill-opacity: 0.78; }}
  svg.hist .hbar:hover {{ fill-opacity: 1; }}
  svg.hist .axis {{ stroke: var(--line); stroke-width: 1; }}
  svg.hist .axlbl {{ font-family: 'IBM Plex Mono', monospace; font-size: 9px; fill: var(--ink-soft); }}

  ul.bars {{ list-style: none; padding: 0; margin: 6px 0 0; font-size: 12.5px; }}
  ul.bars li {{
    position: relative; display: grid;
    grid-template-columns: 1fr auto; column-gap: 8px;
    padding: 2px 8px; margin-bottom: 2px;
    border: 1px solid #d6cfb9; overflow: hidden; isolation: isolate;
  }}
  ul.bars li .bar {{
    position: absolute; left: 0; top: 0; bottom: 0;
    width: var(--w); background: #e3dbc4; z-index: -1;
  }}
  ul.bars li .bar-label {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  ul.bars li .bar-count {{ font-family: 'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; }}

  .cvegeo-lens {{ margin-top: 8px; font-size: 12px; }}
  .cvegeo-lens .chip {{ margin-right: 4px; }}

  /* muestra */
  .sample-table {{
    font-size: 11.5px; border-collapse: collapse; width: 100%;
    background: var(--paper); border: 1px solid var(--line);
    font-family: 'IBM Plex Mono', monospace;
  }}
  .sample-table th {{ background: #ece6d6; padding: 6px 8px; text-align: left; border-bottom: 1px solid var(--line); }}
  .sample-table td {{ padding: 4px 8px; border-bottom: 1px dotted #d6cfb9; max-width: 240px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
  .sample-wrap {{ overflow-x: auto; border: 1px solid var(--line); }}

  /* cruces */
  section.crosses .cross-card {{
    background: var(--paper); border: 1px solid var(--line);
    border-left: 4px solid var(--accent-2);
    padding: 14px 18px; margin-bottom: 14px;
  }}
  section.crosses .cross-card h3 {{ margin: 0 0 6px; font-size: 14px; }}

  footer {{
    margin-top: 64px; padding-top: 16px; border-top: 1px solid var(--line);
    font-size: 11px; letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--ink-soft); display: flex; justify-content: space-between;
  }}
</style>
</head>
<body>
<div class="container">

  <header class="hero">
    <div class="titles">
      <h1>EDA / {table}</h1>
      <p>Profiling exploratorio — perfil estructural, distribuciones y cruces geográficos.</p>
    </div>
    <div class="meta-block">
      <div>Generado · {generated}</div>
      <div>blackprint_db_prd</div>
    </div>
  </header>

  <section class="kpis">
    <div class="kpi"><div class="k">filas (total)</div><div class="v">{n_rows}</div></div>
    <div class="kpi"><div class="k">columnas</div><div class="v">{n_cols}</div></div>
    <div class="kpi"><div class="k">tamaño en disco</div><div class="v">{table_size}</div></div>
    <div class="kpi"><div class="k">muestra analizada</div><div class="v">{sample_size}<span class="unit">filas</span></div></div>
  </section>

  <div style="margin-bottom: 24px;">{bucket_chips}</div>

  <h2>Inventario de columnas</h2>
  <table class="cols">
    <thead>
      <tr>
        <th>columna</th>
        <th>tipo pg</th>
        <th>bucket / flags</th>
        <th class="num">n (muestra)</th>
        <th class="num">nulls</th>
        <th class="num">distinct</th>
      </tr>
    </thead>
    <tbody>{columns_table}</tbody>
  </table>

  <h2>Muestra · primeras 5 filas</h2>
  <div class="sample-wrap">{sample_html}</div>

  {crosses_html}

  <h2>Perfil por columna</h2>
  <div class="details-grid">{details_html}</div>

  <footer>
    <span>{table}</span>
    <span>blackprint · exploratory data analysis</span>
  </footer>
</div>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=50_000,
                    help="Tamaño de muestra. Default 50000. Usa --full para tabla completa.")
    ap.add_argument("--full", action="store_true", help="Leer toda la tabla.")
    ap.add_argument("--crosses", action="store_true",
                    help="Si encuentra lat/lon o cvegeo, intentar cruces.")
    ap.add_argument("--out", default="eda_stg_avaluos.html")
    args = ap.parse_args()

    sample_n = None if args.full else args.sample

    print(f"[1/4] Conectando a {DB_NAME}…")
    conn = connect()

    try:
        print("[2/4] Fase 1 · inventario estructural")
        cols = fetch_schema(conn)
        n_rows = fetch_row_count(conn)
        table_size = fetch_table_size(conn)
        print(f"      · {len(cols)} columnas · {n_rows:,} filas · {table_size}")

        print("[3/4] Fase 2 · muestra + perfilado por columna")
        df = fetch_sample(conn, sample_n)
        print(f"      · muestra obtenida: {len(df):,} filas")
        for c in cols:
            try:
                profile_column(df, c)
            except Exception as e:
                print(f"      ⚠ error perfilando {c.name}: {e}")

        crosses: dict[str, Any] = {}
        if args.crosses:
            print("[3b]  Fase 3 · cruces opcionales")
            crosses = run_crosses(conn, df, cols)

        print("[4/4] Fase 4 · render HTML")
        render_html(
            cols=cols,
            n_rows=n_rows,
            table_size=table_size,
            sample_size=len(df),
            sample_df=df,
            crosses=crosses,
            out_path=args.out,
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()