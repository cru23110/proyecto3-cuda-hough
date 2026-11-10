# Uso: python scripts/analyze.py
import re
from pathlib import Path

import pandas as pd
from scipy import stats

RES = Path("results")
TEX = Path("docs/informe/tablas")
VERSIONS = {"global": "Global", "const": "Global + Constante", "shared": "Global + Constante + Compartida"}


def describe(ms):
    n = len(ms)
    ci = stats.t.ppf(0.975, n - 1) * ms.std() / n ** 0.5
    return {"n": n, "media": ms.mean(), "sd": ms.std(), "mediana": ms.median(), "min": ms.min(),
            "max": ms.max(), "cv_pct": 100 * ms.std() / ms.mean(), "ic95": ci}


def tex_table(df, cols, fmt, path, header):
    cell = lambda v: str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)
    rows = [" & ".join(fmt[c](r[c]) if c in fmt else cell(r[c]) for c in cols) + r" \\" for _, r in df.iterrows()]
    body = "\n".join([r"\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}", r"\toprule",
                      header + r" \\", r"\midrule", *rows, r"\bottomrule", r"\end{tabular}"])
    path.write_text(body + "\n", encoding="utf-8")


f3 = lambda v: f"{v:.3f}"
TEX.mkdir(parents=True, exist_ok=True)

# Bitácoras oficiales
data = {v: pd.read_csv(RES / f"bitacora_{v}.csv") for v in VERSIONS}
cpu = re.search(r"CPU: ([\d.]+) ms", (RES / "system_info.txt").read_text())
cpu_ms = float(cpu.group(1)) if cpu else float("nan")

summary = pd.DataFrame([{"version": v, **describe(d.kernel_ms)} for v, d in data.items()])
base = summary.loc[summary.version == "global", "mediana"].item()
summary["speedup_global"] = base / summary.mediana
summary["speedup_cpu"] = cpu_ms / summary.mediana
summary.to_csv(RES / "summary.csv", index=False, float_format="%.6f")

# Welch: ¿diferencia real?
NOMBRE = {"global": "Global", "const": "Constante", "shared": "Compartida"}
welch = []
for a, b in [("global", "const"), ("global", "shared"), ("const", "shared")]:
    t, p = stats.ttest_ind(data[a].kernel_ms, data[b].kernel_ms, equal_var=False)
    welch.append({"comparacion": f"{NOMBRE[a]} y {NOMBRE[b].lower()}", "t": t, "p": p, "significativa": p < 0.05})
welch = pd.DataFrame(welch)

md = ["| Versión | n | Media ± sd, ms | Mediana, ms | Mínimo, ms | IC 95 % | Aceleración contra global | Aceleración contra CPU |",
      "|---|---|---|---|---|---|---|---|"]
for _, r in summary.iterrows():
    md.append(f"| {VERSIONS[r.version]} | {r.n} | {r.media:.3f} ± {r.sd:.3f} | {r.mediana:.3f} | {r['min']:.3f} | "
              f"± {r.ic95:.3f} | {r.speedup_global:.2f}× | {r.speedup_cpu:.0f}× |")
md += ["", f"CPU secuencial: {cpu_ms:.2f} ms. Aceleración calculada con la mediana.", "",
       "| Prueba t de Welch | t | p | ¿Significativa (α = 0.05)? |", "|---|---|---|---|"]
md += [f"| {r.comparacion} | {r.t:.2f} | {r.p:.2e} | {'Sí' if r.significativa else 'No'} |" for _, r in welch.iterrows()]
(RES / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

s = summary.assign(nombre=summary.version.map(VERSIONS))
tex_table(s, ["nombre", "media", "sd", "mediana", "min", "max", "ic95", "speedup_global"],
          {"media": f3, "sd": f3, "mediana": f3, "min": f3, "max": f3, "ic95": lambda v: rf"$\pm${v:.3f}",
           "speedup_global": lambda v: rf"{v:.2f}$\times$"},
          TEX / "resumen.tex", r"Versión & Media & Desviación & Mediana & Mínimo & Máximo & IC 95\,\% & Aceleración")
tex_table(welch, ["comparacion", "t", "p", "significativa"],
          {"t": lambda v: f"{v:.2f}", "p": lambda v: "$%s \\times 10^{%d}$" % tuple(f"{v:.1e}".replace("e-0", "e-").split("e")[:1] + [int(f"{v:.1e}".split("e")[1])]),
           "significativa": lambda v: "Sí" if v else "No"},
          TEX / "welch.tex", r"Comparación & $t$ & $p$ & Significativa")

# Bitácora: primeras 10
first = pd.DataFrame({"run": range(1, 11), **{v: d.kernel_ms.values[:10] for v, d in data.items()}})
tex_table(first, ["run", *VERSIONS], {v: f3 for v in VERSIONS}, TEX / "bitacora10.tex",
          r"Corrida & Global & Global + Constante & Global + Constante + Compartida")
full = pd.DataFrame({"run": range(1, len(data["global"]) + 1), **{v: d.kernel_ms.values for v, d in data.items()}})
tex_table(full, ["run", *VERSIONS], {v: f3 for v in VERSIONS}, TEX / "bitacora_completa.tex",
          r"Corrida & Global & Global + Constante & Global + Constante + Compartida")

# Barrido de bloques
sweep_path = RES / "bitacora_block_sweep.csv"
if sweep_path.exists():
    # Mínimo: inmune al ruido
    sw = pd.read_csv(sweep_path).groupby(["version", "block"]).kernel_ms.min().unstack(0)[list(VERSIONS)]
    occ = pd.read_csv(RES / "ocupancia.txt", header=None, names=["version", "block", "occ", "pct"])
    occ["bpsm"] = occ.occ.str.extract(r"(\d+) bloques").astype(int)
    occ = occ.pivot(index="block", columns="version", values="bpsm")
    sw.to_csv(RES / "block_sweep_summary.csv", float_format="%.6f")
    t = sw.reset_index().assign(occ_g=occ["global"].values, occ_s=occ["shared"].values)
    tex_table(t, ["block", *VERSIONS, "occ_g", "occ_s"], {v: f3 for v in VERSIONS},
              TEX / "barrido.tex", r"Hilos por bloque & Global & Constante & Compartida & Bloques por SM, global & Bloques por SM, compartida")
    print("\nMínimo por bloque (ms):\n", sw.round(4))

# Control: ¿el ruido es externo?
ctrl_path = RES / "bitacora_control_ruido.csv"
if ctrl_path.exists():
    g = pd.read_csv(ctrl_path).groupby(["version", "block"], sort=False).kernel_ms
    ct = pd.DataFrame({"n": g.size(), "min": g.min(), "mediana": g.median(), "max": g.max(),
                       "lentas": g.apply(lambda s: 100 * (s > 1.2 * s.min()).mean())}).reset_index()
    ct["kernel"] = ct.version.map({"global": "Global", "shared": "Compartida"}) + ", bloques de " + ct.block.astype(str)
    tex_table(ct, ["kernel", "n", "min", "mediana", "max", "lentas"],
              {"min": f3, "mediana": f3, "max": f3, "lentas": lambda v: rf"{v:.0f}\,\%"},
              TEX / "control_ruido.tex", r"Kernel & Corridas & Mínimo & Mediana & Máximo & Corridas lentas")
    print("\nControl de ruido:\n", ct.round(3).to_string(index=False))

# Variantes (7a y volcado)
var_path = RES / "bitacora_variantes.csv"
if var_path.exists():
    va = pd.read_csv(var_path)
    vs = va.groupby(["group", "variant"], sort=False).agg(mediana=("kernel_ms", "median"), min=("kernel_ms", "min"),
                                                          mismatches=("mismatches", "first")).reset_index()
    vs.to_csv(RES / "variantes_summary.csv", index=False, float_format="%.6f")
    for g in ["trig", "shared"]:
        tex_table(vs[vs.group == g], ["variant", "mediana", "min", "mismatches"],
                  {"variant": lambda v: v.replace("_", r"\_"), "mediana": f3, "min": f3},
                  TEX / f"variantes_{g}.tex", r"Variante & Mediana & Mínimo & Diferencias")
    print("\nVariantes:\n", vs.round(4).to_string(index=False))

# Optimizaciones de la compartida
OPT = {"base_256": "Versión final, 256 hilos", "base_512": "Versión final, 512 hilos",
       "ldg_512": r"Tabla global con \texttt{\_\_ldg}, 512", "div_const_512": r"$\theta$ por hilo, tabla constante",
       "div_ldg_512": r"$\theta$ por hilo, tabla global \texttt{\_\_ldg}", "multi_256": "Varios píxeles por hilo, 256",
       "multi_512": "Varios píxeles por hilo, 512", "fijo_256": "Solo inicializar y volcar, 256",
       "fijo_512": "Solo inicializar y volcar, 512", "fijo_multi_512": "Solo inicializar y volcar, malla fija"}
opt_path = RES / "bitacora_optimizaciones.csv"
if opt_path.exists():
    op = pd.read_csv(opt_path).groupby(["variant", "block", "grid"], sort=False).agg(
        mediana=("kernel_ms", "median"), min=("kernel_ms", "min"), mismatches=("mismatches", "first")).reset_index()
    op.to_csv(RES / "optimizaciones_summary.csv", index=False, float_format="%.6f")
    op["nombre"] = op.variant.map(OPT)
    tex_table(op, ["nombre", "grid", "mediana", "min", "mismatches"], {"mediana": f3, "min": f3},
              TEX / "optimizaciones.tex", r"Variante & Bloques & Mediana & Mínimo & Diferencias")
    print("\nOptimizaciones:\n", op.drop(columns="nombre").round(4).to_string(index=False))

print(summary.round(4).to_string(index=False))
print(welch.to_string(index=False))
