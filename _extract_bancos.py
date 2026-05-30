# -*- coding: utf-8 -*-
"""EXTRACTOR READ-ONLY para el dataset de "Bancos hipotecarios" (col. `nombre`)
de la seccion §07 del dashboard. Reproduce EXACTAMENTE la logica de
compute_absorcion_data() del fact, pero sobre la columna `nombre`.

- Un solo SELECT (solo lectura) contra staging.stg_avaluos. NO escribe en la BD.
- Mismos filtros que READ_QUERY (CDMX, proposito=1, valores validos, ano 2019..hoy).
- Mismo esquema de fila: [idx, ano, trim, clase, rec, ban, sup_c, sup_t, edad, valor]
- Salida: public/_abs_bancos.json  (lo consume el dashboard).

Uso:  python _extract_bancos.py
"""
import os, sys, re, json

HERE = os.path.dirname(os.path.abspath(__file__))

# --- cargar .env manualmente (sin dependencias) ---
envp = os.path.join(HERE, ".env")
if os.path.exists(envp):
    for line in open(envp, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import psycopg2

DB = dict(host="35.211.253.113", port=5432, database="blackprint_db_prd",
          user="josue_user", password=os.environ.get("BLACKPRINT_DB_PASSWORD"))

TOP_N = 20

# Mismo universo que READ_QUERY del fact + columnas `nombre` y `constructor`,
# con `ano`/`trimestre`/atributos ya tipados desde SQL.
QUERY = r"""
SELECT
    id_avaluo,
    ano::int                                                         AS ano,
    EXTRACT(QUARTER FROM TO_DATE(NULLIF(fecha_avaluo,''),'DD/MM/YYYY'))::int AS trimestre,
    NULLIF(clase, '')::int                                           AS clase,
    NULLIF(recamaras, '')::int                                       AS recamaras,
    NULLIF(banos, '')::int                                           AS banos,
    NULLIF(sup_construida, '')::numeric                              AS sup_construida,
    NULLIF(sup_terreno, '')::numeric                                 AS sup_terreno,
    NULLIF(edad_meses, '')::int                                      AS edad_meses,
    NULLIF(valor_concluido, '')::numeric                             AS valor_concluido,
    NULLIF(TRIM(constructor), '')                                    AS constructor,
    NULLIF(TRIM(nombre), '')                                         AS nombre
FROM staging.stg_avaluos
WHERE cve_ent = '09'
  AND proposito = '1'
  AND clase IS NOT NULL
  AND valor_concluido ~ '^\d+(\.\d+)?$' AND valor_concluido::numeric > 0
  AND sup_construida  ~ '^\d+(\.\d+)?$' AND sup_construida::numeric  > 0
  AND m2_sv           ~ '^\d+(\.\d+)?$' AND m2_sv::numeric           > 0
  AND ano ~ '^\d+$' AND ano::int BETWEEN 2019 AND EXTRACT(YEAR FROM CURRENT_DATE)::int
"""

# indices de columna en cada tupla devuelta
C_ID, C_ANO, C_TRI, C_CLA, C_REC, C_BAN, C_SC, C_ST, C_ED, C_VAL, C_CONS, C_NOM = range(12)


def _i(v):
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _ri(v):
    try:
        return int(round(float(v))) if v is not None else None
    except (TypeError, ValueError):
        return None


def build_dataset(records, ent_col):
    """Replica compute_absorcion_data() sobre la columna `ent_col` (indice)."""
    # 1) conteo de id_avaluo unico por entidad
    per_entity = {}            # name -> set(id_avaluo)
    for r in records:
        name = r[ent_col]
        if not name:
            continue
        ida = r[C_ID]
        per_entity.setdefault(name, set()).add(ida if ida is not None else id(r))
    counts = {k: len(v) for k, v in per_entity.items()}
    top = sorted(counts, key=lambda k: counts[k], reverse=True)[:TOP_N]
    idx = {name: i for i, name in enumerate(top)}

    rows, seen = [], set()
    for r in records:
        name = r[ent_col]
        ci = idx.get(name) if name else None
        if ci is None:
            continue
        ida = r[C_ID]
        if ida is not None and (ida, ci) in seen:
            continue
        if ida is not None:
            seen.add((ida, ci))
        rows.append([
            ci,
            _i(r[C_ANO]), _i(r[C_TRI]), _i(r[C_CLA]), _i(r[C_REC]), _i(r[C_BAN]),
            _ri(r[C_SC]), _ri(r[C_ST]), _i(r[C_ED]), _ri(r[C_VAL]),
        ])
    years = sorted({rw[1] for rw in rows if rw[1] is not None})
    return {
        "developers": top,
        "totals": {d: counts.get(d, 0) for d in top},
        "rows": rows,
        "years": years,
    }


def main():
    try:
        conn = psycopg2.connect(connect_timeout=12, **DB)
    except Exception as e:
        print("CONNECT_FAIL:", repr(e)[:300]); sys.exit(2)
    print("CONNECT_OK")
    cur = conn.cursor()

    cur.execute("""SELECT column_name FROM information_schema.columns
                   WHERE table_schema='staging' AND table_name='stg_avaluos'""")
    cols = {r[0] for r in cur.fetchall()}
    print("HAS_nombre:", "nombre" in cols, "| HAS_constructor:", "constructor" in cols)
    if "nombre" not in cols:
        print("ABORT: la columna `nombre` no existe en staging.stg_avaluos"); sys.exit(3)

    cur.execute(QUERY)
    records = cur.fetchall()
    print(f"ROWS_FETCHED={len(records):,}")

    bank = build_dataset(records, C_NOM)
    dev  = build_dataset(records, C_CONS)   # solo para validacion/diagnostico

    # --- validacion legible ---
    BANK_RX = re.compile(r"^(BANC[OA]|HSBC|SCOTIA|SANTANDER|CITIBANAMEX|CITIBA|BBVA|BANORTE|BANREGIO|INBURSA|CIBANCO|HIPOTECARIA?|MULTIVA|INTERCAM|VE POR MAS|MIFEL|ACTINVER|INVEX|MONEX|AFIRME|AUTOFIN|INFONAVIT|FOVISSSTE)", re.I)
    print(f"\n--- TOP {TOP_N} `nombre` (BANCOS reales -> seccion Bancos hipotecarios) ---")
    for n in bank["developers"]:
        print(f"  {bank['totals'][n]:>7,}  {n}")
    print(f"\n--- TOP {TOP_N} `constructor` (DESARROLLADORES). Marca = hoy se clasifica mal como banco ---")
    for n in dev["developers"]:
        flag = "   [hoy=BANCO por regex]" if BANK_RX.match(n or "") else ""
        print(f"  {dev['totals'][n]:>7,}  {n}{flag}")

    out = {
        "_meta": {
            "source": "staging.stg_avaluos col=nombre",
            "filters": "cve_ent=09 AND proposito=1 AND valores validos AND ano 2019..now",
            "top_n": TOP_N,
            "row_schema": ["idx", "ano", "trimestre", "clase", "recamaras", "banos",
                           "sup_construida", "sup_terreno", "edad_meses", "valor_concluido"],
            "rows_fetched": len(records),
        },
        "bank": bank,
        "dev_reference": dev,   # referencia; el dashboard sigue usando su `dev` embebido
    }
    outp = os.path.join(HERE, "public", "_abs_bancos.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    conn.close()
    print(f"\nWROTE {outp}")
    print(f"BANK_SERIES={len(bank['developers'])} BANK_ROWS={len(bank['rows'])} "
          f"DEV_SERIES={len(dev['developers'])} DEV_ROWS={len(dev['rows'])}")
    print("EXTRACT_DONE")


if __name__ == "__main__":
    main()
