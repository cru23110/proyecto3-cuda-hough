# Uso: python scripts/compare_outputs.py [carpeta]
import sys
from pathlib import Path

import numpy as np

folder = Path(sys.argv[1] if len(sys.argv) > 1 else "output")
accs = sorted(folder.glob("*_acc.bin"))
if len(accs) < 2:
    sys.exit(f"Se necesitan al menos 2 acumuladores en {folder}")

ref_path = next((p for p in accs if "_global_" in p.name), accs[0])
ref = np.fromfile(ref_path, dtype=np.int32)
ref_png = ref_path.with_name(ref_path.name.replace("_acc.bin", "_lines.png")).read_bytes()
print(f"Referencia: {ref_path.name} ({ref.sum()} votos)")

ok = True
for p in accs:
    if p == ref_path:
        continue
    acc = np.fromfile(p, dtype=np.int32)
    png = p.with_name(p.name.replace("_acc.bin", "_lines.png"))
    diff = int(np.count_nonzero(acc != ref)) if acc.shape == ref.shape else -1
    same_png = png.exists() and png.read_bytes() == ref_png
    ok &= diff == 0 and same_png
    print(f"{p.name}: bins distintos = {diff}, imagen identica = {same_png}")

print("Resultado:", "TODAS IGUALES" if ok else "HAY DIFERENCIAS")
sys.exit(0 if ok else 1)
