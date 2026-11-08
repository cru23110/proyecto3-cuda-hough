# Uso: python scripts/plots.py
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

RES, OUT = Path("results"), Path("results/figures")
VERSIONS = {"global": "Global", "const": "Global + Constante", "shared": "Global + Constante\n+ Compartida"}
COLORS = {"global": "#2a78d6", "const": "#eb6834", "shared": "#1baf7a"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 100,
    "savefig.dpi": 200, "savefig.bbox": "tight", "figure.facecolor": "white",
})
OUT.mkdir(parents=True, exist_ok=True)
data = {v: pd.read_csv(RES / f"bitacora_{v}.csv").kernel_ms for v in VERSIONS}


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)
    print("Figura:", OUT / name)


# 1. Dispersión por versión
fig, ax = plt.subplots(figsize=(7, 4))
rng = np.random.default_rng(0)
for i, (v, ms) in enumerate(data.items()):
    ax.scatter(i + rng.uniform(-0.12, 0.12, len(ms)), ms, s=36, color=COLORS[v], alpha=0.8,
               edgecolor="white", linewidth=1, zorder=3)
    med = ms.median()
    ax.hlines(med, i - 0.25, i + 0.25, color=INK, linewidth=2, zorder=4)
    ax.annotate(f"mediana {med:.3f} ms", (i, med), xytext=(0, 9), textcoords="offset points", ha="center",
                fontsize=10, color=INK, bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
ax.set_xticks(range(3), VERSIONS.values())
ax.set_xlim(-0.5, 2.5)
ax.set_ylim(0, None)
ax.set_ylabel("Tiempo de kernel (ms)")
ax.grid(axis="x", visible=False)
ax.set_title("30 corridas por versión (bloques de 256 hilos)", loc="left", color=INK)
save(fig, "tiempos_por_version.png")

# 2. Speedup vs global
med = {v: ms.median() for v, ms in data.items()}
fig, ax = plt.subplots(figsize=(7, 3.6))
for i, v in enumerate(VERSIONS):
    sp = med["global"] / med[v]
    ax.bar(i, sp, width=0.55, color=COLORS[v])
    ax.annotate(f"{sp:.2f}×", (i, sp), xytext=(0, 4), textcoords="offset points", ha="center", color=INK,
                fontweight="bold")
ax.axhline(1, color=MUTED, linewidth=1, linestyle="--")
ax.set_xticks(range(3), VERSIONS.values())
ax.set_ylabel("Speedup vs. global (mediana)")
ax.grid(axis="x", visible=False)
ax.set_title("Aceleración respecto de la versión global", loc="left", color=INK)
save(fig, "speedup.png")

# 3. Barrido de bloques
sweep = pd.read_csv(RES / "bitacora_block_sweep.csv").groupby(["version", "block"]).kernel_ms.min()
fig, ax = plt.subplots(figsize=(7, 4))
for v in VERSIONS:
    s = sweep[v]
    # Global y constante se superponen
    ax.plot(s.index, s.values, color=COLORS[v], linewidth=6 if v == "global" else 2, marker="o",
            markersize=13 if v == "global" else 8, markeredgecolor="white", markeredgewidth=2)
    ax.annotate(VERSIONS[v].replace("\n", " "), (s.index[-1], s.values[-1]),
                xytext=(10, {"global": 9, "const": -7}.get(v, 0)),
                textcoords="offset points", va="center", fontsize=10, color=INK)
best = sweep["shared"].idxmin()
ax.annotate(f"{sweep['shared'][best]:.3f} ms", (best, sweep["shared"][best]), xytext=(0, -16),
            textcoords="offset points", ha="center", fontsize=10, color=INK)
ax.set_xscale("log", base=2)
ax.set_xticks([128, 256, 512, 1024], ["128", "256", "512", "1024"])
ax.minorticks_off()
ax.set_xlim(100, 1900)
ax.set_ylim(0, None)
ax.set_xlabel("Hilos por bloque")
ax.set_ylabel("Mínimo del kernel (ms)")
ax.set_title("Barrido de tamaño de bloque (mínimo de 30 corridas)", loc="left", color=INK)
save(fig, "barrido_bloques.png")

# 4. Variantes trigonométricas
va = pd.read_csv(RES / "bitacora_variantes.csv")
tr = va[va.group == "trig"].groupby("variant", sort=False).agg(m=("kernel_ms", "median"), bad=("mismatches", "first"))
labels = {"cosf_kernel": "cosf/sinf en kernel", "sincosf_intrin": "__sincosf (intrínseca)",
          "global": "tabla global", "global_ldg": "tabla global + __ldg", "constant": "tabla constante",
          "constant_div": "constante, índice por hilo", "global_ldg_div": "global + __ldg, índice por hilo"}
fig, ax = plt.subplots(figsize=(7, 4))
y = np.arange(len(tr))[::-1]
for yi, (name, r) in zip(y, tr.iterrows()):
    div = name.endswith("_div")
    ax.barh(yi, r.m, height=0.6, color="#6250d6" if div else COLORS["global"],
            label="θ distinto por hilo del warp" if div else "θ igual en todo el warp")
    note = f"{r.m:.3f} ms" + (f"  ({int(r.bad)} mismatches)" if r.bad else "")
    ax.annotate(note, (r.m, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=10, color=INK)
ax.set_yticks(y, [labels[n] for n in tr.index])
ax.set_xlim(0, 0.8)
ax.set_xlabel("Mediana del kernel (ms)")
ax.grid(axis="y", visible=False)
h, l = ax.get_legend_handles_labels()
ax.legend([h[0], h[-1]], [l[0], l[-1]], loc="lower right", frameon=False, fontsize=10)
ax.set_title("Pregunta 7a: de dónde salen cos y sin", loc="left", color=INK)
save(fig, "variantes_trig.png")

# 5. Entrada, Hough y líneas
pic = Image.open("runway.pgm")
panels = [(pic, "Entrada (bordes Sobel)", "gray"), (Image.open("output/runway_shared_hough.png"),
          "Espacio de Hough (θ × r)", None), (Image.open("output/runway_shared_lines.png"), "Líneas detectadas", None)]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.6, 0.9, 1.6]})
for ax, (img, title, cmap) in zip(axes, panels):
    ax.imshow(img, cmap=cmap)
    ax.set_title(title, color=INK)
    ax.axis("off")
save(fig, "pipeline.png")

# 6. Sin NMS vs con NMS
nms = Path("results/figures/sin_nms_global_lines.png")
if nms.exists():
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, (p, t) in zip(axes, [(nms, "Solo umbral: 229 líneas"), (Path("output/runway_global_lines.png"),
                                                                    "Umbral + máximo local 5×5: 11 líneas")]):
        ax.imshow(Image.open(p))
        ax.set_title(t, color=INK)
        ax.axis("off")
    save(fig, "nms_comparacion.png")
