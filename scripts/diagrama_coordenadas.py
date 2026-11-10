# Genera docs/coordenadas.png
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W, H = 800, 600
fig, ax = plt.subplots(figsize=(8, 6.4))
ax.add_patch(plt.Rectangle((0, 0), W, H, fill=False, lw=2))

# Ejes de imagen
ax.annotate("", (W + 40, H), (0, H), arrowprops=dict(arrowstyle="->", color="gray"))
ax.annotate("", (0, -40), (0, H), arrowprops=dict(arrowstyle="->", color="gray"))
ax.text(W + 45, H, "i (columna)", va="center", color="gray")
ax.text(5, -55, "j (fila)", color="gray")
ax.text(5, H + 10, "(0,0) = gloID 0", color="gray")

# Ejes cartesianos
cx, cy = W / 2, H / 2
ax.annotate("", (W - 20, cy), (20, cy), arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2))
ax.annotate("", (cx, H - 20), (cx, 20), arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2))
ax.text(W - 15, cy + 10, "x", color="tab:blue", fontsize=14)
ax.text(cx + 8, H - 25, "y", color="tab:blue", fontsize=14)
ax.text(cx + 8, cy - 25, "(0,0)\nxCent=400, yCent=300", color="tab:blue")

# rMax
ax.plot([cx, W], [cy, 0], "--", color="tab:red")
ax.text(cx + 220, cy - 120, "rMax = 500", color="tab:red", rotation=-37)

# Píxel ejemplo
i, j = 200, 1
ax.plot(i, H - j, "o", color="tab:orange", ms=9)
ax.annotate("gloID = 1000\ni = 1000 % 800 = 200\nj = 1000 / 800 = 1\n"
            "xCoord = 200 - 400 = -200\nyCoord = 300 - 1 = 299",
            (i, H - j), (25, H - 70), va="top", color="tab:orange",
            arrowprops=dict(arrowstyle="->", color="tab:orange"))

ax.set_xlim(-30, W + 140)
ax.set_ylim(-80, H + 40)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("Imagen 800x600: indices (i, j) a coordenadas (x, y)", pad=25)
os.makedirs("docs", exist_ok=True)
plt.savefig("docs/coordenadas.png", dpi=150, bbox_inches="tight")
