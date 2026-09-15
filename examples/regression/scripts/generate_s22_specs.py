"""One-shot generator: read S22 xyz files from a cached GMTKN55
clone, emit one MoleculeSpec module per system. Run once; generated
files are committed to the repo. Re-run if GMTKN55 updates upstream
geometries.

Prerequisites
=============

    git clone --depth=1 https://github.com/grimme-lab/GMTKN55 \
        $VIBEQC_TESTSETS_DIR/GMTKN55

then export ``VIBEQC_TESTSETS_DIR`` (or edit ``SRC`` below). The
script reads ``${VIBEQC_TESTSETS_DIR}/GMTKN55/S22/<NN>/struc.xyz``
for each S22 system in the ``SYSTEMS`` dict and writes one
``s22_<slug>.py`` module to
``examples/regression/systems/molecules/``.

Companion pattern for S66 / ACONF / etc. — copy this file, swap
the ``SRC`` path, swap the ``SYSTEMS`` dict; the rest of the
template is generic.
"""
from __future__ import annotations

import os
from pathlib import Path

SRC = Path(
    os.environ.get("VIBEQC_TESTSETS_DIR", "/tmp/vibeqc_testsets")
) / "GMTKN55" / "S22"

# Default destination is ``../systems/molecules/`` relative to this
# script's location — works whether invoked from the repo root or
# anywhere else on disk.
_HERE = Path(__file__).resolve().parent
DST = _HERE.parent / "systems" / "molecules"

PT = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5,
    "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
    "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
}

# (S22 number) → (file slug, family, brief description)
# Standard S22 system names from Jurečka et al. 2006 PCCP 8, 1985.
SYSTEMS = {
    "03": ("s22_formic_acid_dimer",     "molecule_noncovalent_hb",
           "formic acid dimer (HCOOH)₂ — doubly H-bonded carboxylic-acid pair"),
    "05": ("s22_uracil_dimer_hb",       "molecule_noncovalent_hb",
           "uracil dimer, hydrogen-bonded — two cooperative N-H···O=C bonds"),
    "06": ("s22_2py_2ampy",             "molecule_noncovalent_hb",
           "2-pyridoxine ⋯ 2-aminopyridine — hydrogen-bonded heterocyclic pair"),
    "07": ("s22_adenine_thymine_wc",    "molecule_noncovalent_hb",
           "adenine⋯thymine Watson-Crick pair — biological H-bond reference"),
    "09": ("s22_ethene_dimer",          "molecule_noncovalent_dispersion",
           "ethene dimer (C₂H₄)₂ — π-dispersion model"),
    "10": ("s22_methane_benzene",       "molecule_noncovalent_dispersion",
           "methane⋯benzene — CH⋯π dispersion test"),
    "11": ("s22_benzene_dimer_pd",      "molecule_noncovalent_dispersion",
           "benzene dimer, parallel-displaced — π-stacking minimum"),
    "12": ("s22_pyrazine_dimer",        "molecule_noncovalent_dispersion",
           "pyrazine dimer — heteroaromatic π-stacking"),
    "13": ("s22_uracil_dimer_stacked",  "molecule_noncovalent_dispersion",
           "uracil dimer, stacked — base-stacking analog"),
    "14": ("s22_indole_benzene_stack",  "molecule_noncovalent_dispersion",
           "indole⋯benzene, stacked — Trp-Phe model"),
    "15": ("s22_adenine_thymine_stack", "molecule_noncovalent_dispersion",
           "adenine⋯thymine, stacked — DNA base-stacking reference"),
    "16": ("s22_ethene_ethyne",         "molecule_noncovalent_mixed",
           "ethene⋯ethyne — mixed π/H-bond character"),
    "17": ("s22_benzene_water",         "molecule_noncovalent_mixed",
           "benzene⋯water — OH⋯π / mixed dispersion + electrostatic"),
    "18": ("s22_benzene_ammonia",       "molecule_noncovalent_mixed",
           "benzene⋯ammonia — NH⋯π / mixed"),
    "19": ("s22_benzene_hcn",           "molecule_noncovalent_mixed",
           "benzene⋯HCN — CH⋯π / mixed"),
    "21": ("s22_indole_benzene_t",      "molecule_noncovalent_mixed",
           "indole⋯benzene, T-shaped — protein-aromatic mixed"),
    "22": ("s22_phenol_dimer",          "molecule_noncovalent_mixed",
           "phenol dimer — OH⋯π plus dispersion"),
}

TEMPLATE = '''"""S22 #{num} — {desc}.

Geometry from the GMTKN55 distribution of the S22 test set (Jurečka,
Šponer, Černý, Hobza 2006), in Ångström. Revised S22A reference
interaction energies in Marshall, Burns, Sherrill 2011 (the canonical
post-2011 CCSD(T)/CBS reference).

For the cross-code parity suite this is a single-determinant SCF on
the dimer geometry — interaction-energy benchmarking (E_AB − E_A −
E_B + counterpoise) is wave-2 multi-component-machinery work.
"""
from __future__ import annotations

from ...core.spec import AtomCart, MoleculeSpec

SPEC = MoleculeSpec(
    id="{id_}",
    family="{family}",
    atoms=(
{atoms_block}
    ),
    charge=0,
    multiplicity=1,
    default_conv_tol_energy=1e-9,
    default_max_iter=80,
    notes=(
        "S22 system #{num} — {desc}. Geometry from grimme-lab/GMTKN55 "
        "S22/{num}/struc.xyz."
    ),
    citation=(
        "Jurečka, Šponer, Černý, Hobza, *Phys. Chem. Chem. Phys.* "
        "**2006**, 8, 1985 (S22 origin, DOI 10.1039/B600027D); "
        "Marshall, Burns, Sherrill, *J. Chem. Phys.* **2011**, 135, "
        "194102 (S22A revised CCSD(T)/CBS, DOI 10.1063/1.3659142). "
        "Geometry: grimme-lab/GMTKN55 S22/{num}/struc.xyz."
    ),
)
'''


def main() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    for num, (id_, family, desc) in SYSTEMS.items():
        xyz_path = SRC / num / "struc.xyz"
        if not xyz_path.exists():
            print(f"  ! missing {xyz_path}")
            continue
        lines = xyz_path.read_text().splitlines()
        n_atoms = int(lines[0].strip())
        # GMTKN55 struc.xyz has: <natoms>\n<blank or comment>\n<atom lines>
        atoms = []
        for line in lines[2 : 2 + n_atoms]:
            parts = line.split()
            sym = parts[0]
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            z_atomic = PT[sym]
            atoms.append(
                f'        AtomCart(symbol="{sym}", z={z_atomic}, '
                f"xyz_ang=({x:11.8f}, {y:11.8f}, {z:11.8f})),"
            )
        atoms_block = "\n".join(atoms)
        body = TEMPLATE.format(
            num=num, id_=id_, family=family, desc=desc,
            atoms_block=atoms_block,
        )
        out_path = DST / f"{id_}.py"
        out_path.write_text(body)
        print(f"  ✓ wrote {out_path.name}  ({n_atoms} atoms)")


if __name__ == "__main__":
    main()
