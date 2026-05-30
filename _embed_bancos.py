# -*- coding: utf-8 -*-
"""Embebe el dataset de Bancos hipotecarios (col. `nombre`) en public/dashboard.html.
- Acorta nombres largos para la leyenda (solo etiqueta; conteos intactos).
- Descarta entradas que NO son prestamistas (p. ej. "AVALUOS ... PRESTACION LABORAL").
- Re-indexa filas tras el descarte.
- Inyecta `var DATA_BANK = {...}` + registro de datasets (dev/bank/all) + hook de
  intercambio, justo despues de `var years = DATA.years || [];` en la IIFE de §07.
Operacion LOCAL (no toca la BD). Hace backup .bak antes de escribir.
"""
import json, os, re, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC  = os.path.join(HERE, "public", "_abs_bancos.json")
HTML = os.path.join(HERE, "public", "dashboard.html")

ANCHOR = "  var years = DATA.years || [];\n"

# --- 1) cargar dataset extraido (UTF-8) ---
data = json.load(open(SRC, encoding="utf-8"))
bank = data["bank"]
devs_raw = bank["developers"]
rows_raw = bank["rows"]
years    = bank["years"]


# --- 2) acortar nombres + marcar descartes ---
def shorten(raw):
    u = (raw or "").upper()
    if "PRESTACI" in u or "AVALUOS REALIZADOS" in u or "AVALÚOS REALIZADOS" in u:
        return None  # no es un banco -> descartar
    rules = [
        ("INFONAVITFOVISSSTE", "Infonavit-Fovissste"),
        ("BBVA", "BBVA"),
        ("SCOTIABANK", "Scotiabank"),
        ("BANORTE", "Banorte"),
        ("MERCANTIL DEL NORTE", "Banorte"),
        ("SANTANDER", "Santander"),
        ("HSBC", "HSBC"),
        ("BANAMEX", "Banamex"),
        ("NACIONAL DE M", "Banamex"),
        ("ARMADA", "Banjército"),
        ("VE POR M", "Ve por Más"),
        ("MIFEL", "Mifel"),
        ("AFIRME", "Afirme"),
        ("INBURSA", "Inbursa"),
        ("DEL BAJ", "BanBajío"),
        ("ION FINANCIERA", "ION Financiera"),
        ("INFONAVIT", "Infonavit"),
        ("FOVISSSTE", "Fovissste"),
        ("COFINAVIT", "Cofinavit"),
        ("COFISSSTE", "Cofissste"),
    ]
    for key, disp in rules:
        if key in u:
            return disp
    if u == "HIR":
        return "HIR Casa"
    # fallback: corta el boilerplate corporativo y title-case
    s = re.split(r"\s+S\s*A\b", raw, maxsplit=1)[0]
    s = re.sub(r"\s+", " ", s).strip()
    return s.title() if s else raw


mapping = []      # (raw, display|DROP)
keep_old_idx = [] # indices originales que se conservan
new_devs = []
for i, raw in enumerate(devs_raw):
    disp = shorten(raw)
    if disp is None:
        mapping.append((raw, "✗ DESCARTADO (no es banco)"))
        continue
    mapping.append((raw, disp))
    keep_old_idx.append(i)
    new_devs.append(disp)

remap = {old: new for new, old in enumerate(keep_old_idx)}

new_rows = []
for r in rows_raw:
    oi = r[0]
    if oi not in remap:
        continue
    r2 = list(r)
    r2[0] = remap[oi]
    new_rows.append(r2)

bank_clean = {"developers": new_devs, "rows": new_rows, "years": years}
blob = json.dumps(bank_clean, ensure_ascii=False, separators=(",", ":"))

# --- 3) JS: registro de datasets por entidad + hook de intercambio ---
REGISTRY_JS = """\
  /* === §07 · datasets por ENTIDAD ===========================================
     dev  = `constructor` (quien construyo/registro el avaluo)
     bank = `nombre`      (institucion que OTORGO el credito = banco real)
     all  = union de ambas (indices de bank desplazados tras los de dev).
     El toggle Entidad llama window.__absSetDataset(name) y re-renderiza. */
  window.__absDatasets = (function(){
    var dev  = { developers: (DATA.developers||[]).slice(),
                 rows:       (DATA.rows||[]),
                 years:      (DATA.years||[]).slice() };
    var bk   = { developers: (DATA_BANK.developers||[]),
                 rows:       (DATA_BANK.rows||[]),
                 years:      (DATA_BANK.years||[]) };
    var off  = dev.developers.length;
    var allDevs = dev.developers.concat(bk.developers);
    var allRows = dev.rows.concat(bk.rows.map(function(r){ var c=r.slice(); c[0]=r[0]+off; return c; }));
    var yset = {}; dev.years.concat(bk.years).forEach(function(y){ yset[y]=1; });
    var allYears = Object.keys(yset).map(Number).sort(function(a,b){ return a-b; });
    return { dev: dev, bank: bk, all: { developers: allDevs, rows: allRows, years: allYears } };
  })();
  window.__absSetDataset = function(name){
    var ds = window.__absDatasets[name] || window.__absDatasets.dev;
    devs = ds.developers || []; rows = ds.rows || []; years = ds.years || [];
    if (typeof state !== 'undefined' && state) state.hidden = new Set();
    apply();
  };
"""

# --- 4) inyectar en dashboard.html (UTF-8) ---
html = open(HTML, encoding="utf-8").read()
if "var DATA_BANK =" in html:
    print("ABORT: ya parece inyectado (`var DATA_BANK =` presente). "
          "Restaura desde dashboard.html.bak si quieres re-inyectar.")
    sys.exit(1)
n = html.count(ANCHOR)
if n != 1:
    print(f"ABORT: ancla encontrada {n} veces (esperaba 1). No se modifico nada.")
    sys.exit(1)

injection = ANCHOR + "  var DATA_BANK = " + blob + ";\n" + REGISTRY_JS
shutil.copyfile(HTML, HTML + ".bak")
html2 = html.replace(ANCHOR, injection, 1)
open(HTML, "w", encoding="utf-8").write(html2)

# --- 5) reporte ---
print("MAPEO nombre -> etiqueta (conteos intactos):")
for raw, disp in mapping:
    short = (raw[:62] + "...") if len(raw) > 65 else raw
    print(f"  {short:<66} ->  {disp}")
print()
print(f"SERIES_BANCO_FINALES = {len(new_devs)}  (descartadas: {len(devs_raw)-len(new_devs)})")
print(f"FILAS_BANCO = {len(new_rows):,}")
print(f"TAMANO dashboard.html: {os.path.getsize(HTML+'.bak'):,} -> {os.path.getsize(HTML):,} bytes "
      f"(+{os.path.getsize(HTML)-os.path.getsize(HTML+'.bak'):,})")
print("EMBED_DONE")
