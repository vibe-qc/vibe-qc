"""vibe-view showcase: basis-function visualisation (H2O / pob-TZVP).

Evaluates individual atomic-orbital basis functions on a 3-D grid
and packs them as ``basis.ao`` sections in a QVF.  This is the
producer-side counterpart to the basis-function picker in vibe-view:
every AO that is written can be selected and rendered as an isosurface.

The script uses the ``pob-TZVP`` basis set (Peintinger et al. 2013),
which is representative of the basis-set optimisation workflows that
this feature serves.  For a small 3-atom molecule the full grid
evaluation of all primitives + contracted shells runs in a few
seconds on a laptop.

Run:

    ~/path/to/vibeqc/.venv/bin/python \\
        examples/vibe_view/showcase_basis_functions.py

Output:

    output-h2o-ao.qvf

Then:

    vibe-view open output-h2o-ao.qvf

What you see
------------

The left sidebar has a "Basis Functions (N)" entry where N is the
number of sections in the file.  Click it and the first AO renders
in the 3-D viewport with the default isosurface controls.

The right panel shows:

  * **Basis Functions** — a dropdown that lets you switch between
    individual AOs.  Each entry is labelled by atom symbol, shell
    type, and shell/primitive index.
  * **Isosurface Controls** — isovalue, colormap, opacity, and clip
    planes (same shared controls used by all volume sections).

Periodic replication controls appear if the file has a lattice.

Walking the panels
------------------

    1.  Pick an H 1s primitive.  At isovalue 0.02 you see a smooth
        sphere around the H atom — the contracted 1s function.

    2.  Pick an O 2sp primitive (diffuse, α ≈ 0.2).  Dial the isovalue
        down to 0.005 — you see how far the diffuse lobe extends.

    3.  Switch to RdBu colormap, pick an O 3d contracted shell.
        The sign alternating (+/−) lobes of the d functions become
        visible — red positive, blue negative.

    4.  Compare the most diffuse vs most contracted primitive of the
        same shell by switching between them.

    5.  Increase replication (if the QVF was produced from a periodic
        calculation) to see how the AO bleeds into neighbouring cells.

Filtering
---------

The default mode exports **both** contracted shells and individual
primitives.  The ``mode`` parameter controls selection:

  * ``mode="all"`` — every contracted AO + every primitive (default).
  * ``mode="primitives"`` — individual primitives only.
  * ``mode="contracted"`` — contracted shells only.

You can also pass explicit ``ao_indices`` or ``primitive_indices``
lists to narrow the export to exactly the functions you care about.
"""

from __future__ import annotations

from pathlib import Path

from vibeqc import Atom, Molecule
from vibeqc._vibeqc_core import BasisSet
from vibeqc.output.formats.qvf import (
    QVF_FORMAT_VERSION,
    qvf_ao_data,
    write_qvf,
)
from vibeqc.output.plan import OutputPlan

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Build the molecule + basis
# ---------------------------------------------------------------------------
# Water in its equilibrium geometry (bohr). O at origin, H atoms in the
# xz plane at ~104.5 degrees.
mol = Molecule(
    [
        Atom(8, [0.0000, 0.0000, 0.0000]),
        Atom(1, [1.4320, 0.0000, 1.1160]),
        Atom(1, [-1.4320, 0.0000, 1.1160]),
    ],
)

# pob-TZVP is a triple-zeta basis with polarisation (Peintinger 2013).
# If the basis is not bundled, fall back to 6-31g* which is always
# available.
try:
    basis = BasisSet(mol, "pob-TZVP")
    basis_label = "pob-TZVP"
except RuntimeError:
    basis = BasisSet(mol, "6-31g*")
    basis_label = "6-31g*"

# ---------------------------------------------------------------------------
# Inspect the basis
# ---------------------------------------------------------------------------
shells = basis.shells()
print("=" * 72)
print(f" Basis: {basis_label}")
print(f" Atoms:  {len(mol.atoms)}")
print(f" Shells: {len(shells)}")
print(f" nbf:    {basis.nbasis}")
print("=" * 72)
print()

for si, s in enumerate(shells):
    sym = (
        ["?", "H", "He", "Li", "Be", "B", "C", "N", "O", "F"][s.atom_index + 1]
        if s.atom_index < 9
        else "?"
    )
    lt = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g"}[s.l]
    nprim = len(s.exponents)
    exps = ", ".join(f"{x:.4f}" for x in s.exponents)
    print(f"  Shell {si:2d}: {sym} {lt}  {nprim} primitives  exps=[{exps}]")
print()

# ---------------------------------------------------------------------------
# Evaluate the AOs on a grid
# ---------------------------------------------------------------------------
# Default grid: 0.15 bohr spacing (finer than density) to resolve
# compact primitives; 8.0 bohr padding to capture diffuse tails.
print("Evaluating AOs on grid (this may take a minute) ...")
ao_data = qvf_ao_data(
    basis,
    mol,
    basis_label=basis_label,
    # Limit to a subset for the showcase — evaluating ALL primitives
    # of pob-TZVP on water (~120 AOs) at default resolution produces
    # a ~200 MB QVF.  For a quick demo we export every contracted AO
    # plus only the first primitive of each shell.
    include_contracted=True,
    include_primitives=False,
)
print(f"  → {len(ao_data)} AO sections evaluated")
print()

# ---------------------------------------------------------------------------
# Build viewer_defaults with per-section hints
# ---------------------------------------------------------------------------
viewer_defaults: dict = {
    "auto_open": [ao_data[0]["section_id"]] if ao_data else [],
}
# Per-AO isovalue hints based on exponent.
for ao in ao_data:
    exp = ao["ao_metadata"]["exponent"]
    if exp < 0.1:
        iso = 0.005
    elif exp < 1.0:
        iso = 0.02
    elif exp < 10.0:
        iso = 0.05
    else:
        iso = 0.10
    viewer_defaults[ao["section_id"]] = {
        "isovalue": iso,
        "colormap": "RdBu",
        "opacity": 0.5,
    }

# ---------------------------------------------------------------------------
# Write the QVF
# ---------------------------------------------------------------------------
print("Writing QVF ...")
plan = OutputPlan()
path = write_qvf(
    HERE / "output-h2o-ao",
    plan,
    molecule=mol,
    basis=basis_label,
    method="RHF",
    ao_data=ao_data,
    viewer_defaults=viewer_defaults,
)
print(f"  → {path}")
print()

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("=" * 72)
print(" Done. The QVF carries these sections:")
print(f"   structure          — {len(mol.atoms)}-atom water")
print(f"   basis.ao  × {len(ao_data):3d}   — per-AO isosurfaces")
print()
print(" Open it:")
print(f"   vibe-view open {path}")
print()
print(" In vibe-view, click 'Basis Functions' in the left sidebar,")
print(" then use the AO dropdown in the right panel to switch between")
print(" individual basis functions. Adjust isovalue with the slider.")
print("=" * 72)
