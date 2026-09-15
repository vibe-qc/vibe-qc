"""Generate input scripts for the canonical periodic test grid.

Reads the system registry from ``_systems.py`` and emits, for each
(system, method, basis) combination, two scripts:

  examples/periodic/<slug>/<slug>-<method>-<basis>.py        (vibe-qc)
  examples/periodic_pyscf/<slug>/<slug>-<method>-<basis>.py  (PySCF reference)

Method tags:
  RHF              — restricted Hartree-Fock
  RKS-PBE          — restricted KS, PBE GGA
  RKS-B3LYP        — restricted KS, B3LYP hybrid (hf_x = 0.20)

Basis tags (script suffix):
  pobtzvp          — pob-TZVP (loaded from vibe-qc's bundled .g94 in
                     both codes via vibeqc._basis_g94)

Run::
    .venv/bin/python examples/periodic/_gen.py

(idempotent — safe to re-run after editing this generator)
"""
from __future__ import annotations

from pathlib import Path
from pprint import pformat

from _systems import SYSTEMS

ROOT = Path(__file__).resolve().parents[1]      # examples/
PERIODIC = ROOT / "periodic"
PYSCF_REF = ROOT / "periodic_pyscf"


METHODS = [
    # (method_tag, vq_method, vq_functional, pyscf_xc)
    ("RHF",       "RHF", None,    None),
    ("RKS-PBE",   "RKS", "pbe",   "pbe"),
    # FLAVOR NOTE: vibe-qc's bare "b3lyp" is the VWN5 flavor (ORCA
    # definition) == CRYSTAL14's B3LYP keyword, so the vibe-qc side
    # uses the bare name. PySCF's bare "b3lyp" is the Gaussian/VWN-RPA
    # flavor, so the PySCF reference twin MUST spell "b3lyp5" — the
    # old b3lyp/b3lyp pairing silently crossed the ~10-15 mHa/heavy-
    # atom flavor gap on every parity pair.
    ("RKS-B3LYP", "RKS", "b3lyp", "b3lyp5"),
]
BASIS_TAG = "pobtzvp"
BASIS_NAME = "pob-tzvp"


def geometry_literals(atoms_ase) -> tuple[str, str]:
    """Return Python literals for cell + atom data.

    Generated examples must be copyable to a scratch directory and run
    without sibling helper modules. ``_systems.py`` remains the registry
    used by this generator only; its ASE-built geometry is inlined into
    each emitted input file.
    """
    cell_ang = atoms_ase.cell.array.tolist()
    atom_data = [
        (str(sym), [float(x) for x in pos])
        for sym, pos in zip(
            atoms_ase.get_chemical_symbols(), atoms_ase.get_positions()
        )
    ]
    return pformat(cell_ang, width=78), pformat(atom_data, width=78)


def vibeqc_script(
    slug: str,
    method_tag: str,
    vq_method: str,
    vq_functional: str | None,
    cell_literal: str,
    atom_literal: str,
) -> str:
    """Render the vibe-qc input.py for one (system, method, basis) cell."""
    fn_args = ""
    if vq_method == "RKS":
        fn_args = f'\n    functional="{vq_functional}",'
    return f'''"""{slug} — {method_tag} / {BASIS_NAME} / Γ via the native GDF driver.

Standalone input: geometry is embedded below, so this file can be copied
to a scratch calculation directory and run without sibling helper modules.

Periodic SCF runs through ``vibeqc.run_periodic_job`` -> the native
``run_rhf_periodic_gamma_gdf`` driver (Lpq, S, T, V_ne, V_xc, SCF loop,
J/K einsums, and DIIS are vibe-qc-owned; PySCF is only an external
reference). ``fmixing_percent=30`` mirrors CRYSTAL's default FMIXING
Fock/KS matrix mixing.

Run::
    .venv/bin/python examples/periodic/{slug}/{slug}-{method_tag}-{BASIS_TAG}.py

Produces (sibling files, sharing the script's stem):
    {{stem}}.out, {{stem}}.system, {{stem}}.molden
"""
from pathlib import Path

import numpy as np

from vibeqc import (
    Atom, BasisSet, PeriodicSystem,
    run_periodic_job,
)

ANG2BOHR = 1.0 / 0.529177210903
OUTPUT_STEM = Path(__file__).resolve().with_suffix("")

CELL_ANG = {cell_literal}
ATOM_DATA = {atom_literal}

cell_bohr = np.array(CELL_ANG, dtype=float) * ANG2BOHR
Z_BY_SYM = {{
    "H": 1, "Li": 3, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17,
    "Ti": 22, "Zn": 30,
}}
atoms = [
    Atom(Z_BY_SYM[sym], [float(x) * ANG2BOHR for x in pos_ang])
    for sym, pos_ang in ATOM_DATA
]
system = PeriodicSystem(3, cell_bohr, atoms)
basis = BasisSet(system.unit_cell_molecule(), "{BASIS_NAME}")

run_periodic_job(
    system, basis,
    method="{vq_method}",{fn_args}
    output=OUTPUT_STEM,
    use_diis=True,
    damping=0.0,
    fmixing_percent=30.0,
    max_iter=80,
    conv_tol_energy=1e-7,
    write_density=False,   # keep the generated parity grid lightweight.
)
'''


def pyscf_script(
    slug: str,
    method_tag: str,
    pyscf_xc: str | None,
    cell_literal: str,
    atom_literal: str,
) -> str:
    """Render the PySCF mirror input.py."""
    if pyscf_xc is None:
        scf_block = (
            "from pyscf.pbc import scf as pbc_scf\n"
            "mf = pbc_scf.RHF(cell)"
        )
    else:
        scf_block = (
            "from pyscf.pbc import dft as pbc_dft\n"
            f'mf = pbc_dft.RKS(cell)\nmf.xc = "{pyscf_xc}"'
        )
    return f'''"""PySCF mirror — {slug} {method_tag} / {BASIS_NAME} / Γ via GDF
(exxdiv=ewald). Same geometry / basis as
``../../periodic/{slug}/{slug}-{method_tag}-{BASIS_TAG}.py``.

Run::
    .venv/bin/python examples/periodic_pyscf/{slug}/{slug}-{method_tag}-{BASIS_TAG}.py
Produces:
    {{stem}}.out
"""
from pathlib import Path

import numpy as np

from pyscf.pbc import gto as pbc_gto
from pyscf.pbc import df as pbc_df
from vibeqc._basis_g94 import load_g94_for_pyscf

OUTPUT_STEM = Path(__file__).resolve().with_suffix("")
out_path = OUTPUT_STEM.with_suffix(".out")

CELL_ANG = {cell_literal}
ATOM_DATA = {atom_literal}

atom_str = "; ".join(
    f"{{sym}} {{x:.6f}} {{y:.6f}} {{z:.6f}}"
    for sym, (x, y, z) in ATOM_DATA
)
unique_syms = sorted({{sym for sym, _pos in ATOM_DATA}})
basis_dict = load_g94_for_pyscf("{BASIS_NAME}", unique_syms)

# Force the GDF FFT mesh to a sane fixed value matching what the
# vibe-qc driver uses internally. Auto-mesh from PySCF + a tight pob-TZVP
# valence exponent picks {1961}^3 (~60 GB intermediates) — fills disk.
cell = pbc_gto.M(
    atom=atom_str,
    a=np.array(CELL_ANG, dtype=float),
    basis=basis_dict,
    unit="A",
    verbose=4,
    output=str(out_path),
    mesh=[31, 31, 31],
    precision=1e-8,
)
{scf_block}
mf.exxdiv = "ewald"
mf.with_df = pbc_df.GDF(cell)
mf.with_df.build()
mf.max_cycle = 80
mf.conv_tol = 1e-7
e = mf.kernel()
print(f"\\nFinal: E = {{e:.10f}} Ha  converged={{mf.converged}}")
'''


def main():
    n_written = 0
    for slug, builder in SYSTEMS:
        atoms_ase = builder()
        cell_literal, atom_literal = geometry_literals(atoms_ase)
        sys_dir_vq = PERIODIC / slug
        sys_dir_py = PYSCF_REF / slug
        sys_dir_vq.mkdir(parents=True, exist_ok=True)
        sys_dir_py.mkdir(parents=True, exist_ok=True)
        for method_tag, vq_method, vq_func, pyscf_xc in METHODS:
            stem = f"{slug}-{method_tag}-{BASIS_TAG}"
            (sys_dir_vq / f"{stem}.py").write_text(
                vibeqc_script(
                    slug, method_tag, vq_method, vq_func,
                    cell_literal, atom_literal,
                )
            )
            (sys_dir_py / f"{stem}.py").write_text(
                pyscf_script(
                    slug, method_tag, pyscf_xc,
                    cell_literal, atom_literal,
                )
            )
            n_written += 2
    print(f"Wrote {n_written} input scripts under "
          f"{PERIODIC.relative_to(ROOT.parent)}/* and "
          f"{PYSCF_REF.relative_to(ROOT.parent)}/*")


if __name__ == "__main__":
    main()
