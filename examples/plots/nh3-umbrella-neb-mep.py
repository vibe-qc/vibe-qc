"""Plot the minimum-energy path from the NH3 umbrella inversion NEB.

Reads ``examples/workflows/output-nh3-umbrella-neb.traj`` (produced by
``examples/workflows/input-nh3-umbrella-neb.py``) and writes
``docs/_static/plots/nh3-umbrella-neb-mep.png`` — the figure embedded
in tutorial 19.

Run:
    .venv/bin/python examples/plots/nh3-umbrella-neb-mep.py

If the .traj file is missing, run the parent example first:
    .venv/bin/python examples/workflows/input-nh3-umbrella-neb.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
TRAJ_PATH = HERE.parent / "workflows" / "output-nh3-umbrella-neb.traj"
PLOT_OUT = HERE.parent.parent / "docs" / "_static" / "plots" / "nh3-umbrella-neb-mep.png"

KCAL_PER_EV = 23.0605

if not TRAJ_PATH.exists():
    raise SystemExit(
        f"{TRAJ_PATH} not found.\n"
        f"Run examples/workflows/input-nh3-umbrella-neb.py first to produce the trajectory."
    )

images = read(str(TRAJ_PATH), index=":")
e0 = images[0].get_potential_energy()
mep_kcal = np.array([(img.get_potential_energy() - e0) * KCAL_PER_EV for img in images])
indices = np.arange(len(images))
ts_index = int(np.argmax(mep_kcal))
barrier = mep_kcal[ts_index]

PLOT_OUT.parent.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=150)

# MEP curve and image markers
ax.plot(indices, mep_kcal, color="#1f77b4", linewidth=2.0, zorder=2)
ax.scatter(indices, mep_kcal, color="#1f77b4", s=42, zorder=3,
           edgecolor="white", linewidth=1.0)

# Highlight the TS image
ax.scatter([ts_index], [barrier], color="#d62728", s=120, marker="*",
           zorder=4, edgecolor="white", linewidth=1.0,
           label=f"TS (image {ts_index})  ΔE‡ = {barrier:.2f} kcal/mol")

# Experimental barrier reference line
ax.axhline(5.8, color="#7f7f7f", linestyle="--", linewidth=1.0,
           label="Experimental barrier (~5.8 kcal/mol)")

ax.set_xlabel("Image index along the band")
ax.set_ylabel(r"$\Delta E$ (kcal mol$^{-1}$)")
ax.set_title("NH$_3$ umbrella inversion MEP — CI-NEB at HF/STO-3G")
ax.set_xticks(indices)
ax.set_ylim(-1.0, max(barrier * 1.15, 13.0))
ax.grid(alpha=0.3, linestyle=":")
ax.legend(loc="upper right", frameon=True, fontsize=9)

fig.tight_layout()
fig.savefig(PLOT_OUT, dpi=300, bbox_inches="tight")

print(f"Wrote {PLOT_OUT.relative_to(HERE.parent.parent)}")
print(f"  Barrier (TS image {ts_index}): {barrier:.3f} kcal/mol")
print(f"  Path symmetric: {np.allclose(mep_kcal, mep_kcal[::-1], atol=0.01)}")
