"""Generate the independent Ag V_ECP reference pinned by tests/test_basis_ecp_sidecars.py.

Run this script with an external PySCF interpreter, never by importing PySCF
into vibe-qc (#515). It reads the Bredow CRYSTAL source records under
``python/vibeqc/basis_library/sources/`` directly, with its own parser and the
CRYSTAL-to-Gaussian ECP power conversion (n = NKL + 2) written here by hand,
so it shares no code with vibe-qc's basis bridge. For each pob basis it
builds the AgCl pair of the #52 cell (Ag at the origin, Cl at half the cell
diagonal) and records the trace of the Ag V_ECP over the spherical AO basis
and its sorted generalized eigenvalues against the AO overlap. Both are
invariant to AO ordering, signs and contraction conventions, so the test can
compare them with vibe-qc's matrix directly.

Example:
    "$VIBEQC_PYSCF_PYTHON" examples/regression/pob_ag_ecp_reference.py \
        --output tests/data/pob_ag_ecp_reference.json

References (the records read here): Ag of pob-TZVP, Laun, Vilela Oliveira
and Bredow, J. Comput. Chem. 39, 1285 (2018), DOI 10.1002/jcc.25195; Ag of
pob-TZVP-rev2, Laun and Bredow, J. Comput. Chem. 43, 839 (2022), DOI
10.1002/jcc.26839; Cl, Peintinger, Vilela Oliveira and Bredow, J. Comput.
Chem. 34, 451 (2013), DOI 10.1002/jcc.23153, and Vilela Oliveira, Laun,
Peintinger and Bredow, J. Comput. Chem. 40, 2364 (2019), DOI
10.1002/jcc.26013.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

BASES = {"pob-tzvp": "pob-TZVP", "pob-tzvp-rev2": "pob-TZVP-rev2"}
BOHR_PER_ANGSTROM = 1.8897261246257702
HALF_DIAGONAL = 5.55 * BOHR_PER_ANGSTROM / 2     # the #52 AgCl cell, two centres
CRYSTAL_L = {0: [0], 1: [0, 1], 2: [1], 3: [2], 4: [3]}   # 1 is an sp shell
ECP_CHANNELS = ("ul", "S", "P", "D", "F", "G")            # M, M0 .. M4


def parse_crystal_record(path: Path, symbol: str):
    """PySCF basis and NWChem ECP text from one CRYSTAL per-element file."""
    tokens = [line.split() for line in path.read_text().splitlines() if line.strip()]
    header, n_shells = int(tokens[0][0]), int(tokens[0][1])
    z = header % 100 if header > 200 else header
    row = 1
    ecp_text = None
    if header > 200:
        if tokens[row] != ["INPUT"]:
            raise ValueError(f"{path}: expected an INPUT ECP block")
        z_eff = float(tokens[row + 1][0])
        counts = [int(t) for t in tokens[row + 1][1:]]
        row += 2
        lines = [f"{symbol} nelec {round(z - z_eff)}"]
        for channel, count in zip(ECP_CHANNELS, counts):
            terms = tokens[row:row + count]
            row += count
            if channel == "ul" or count:
                lines.append(f"{symbol} {channel}")
            if channel == "ul" and not count:
                lines.append("2 1.0 0.0")      # no local terms: a zero local potential
            for exponent, coefficient, nkl in terms:
                lines.append(f"{int(nkl) + 2} {exponent} {coefficient}")
        ecp_text = "\n".join(lines)
    basis = []
    for _ in range(n_shells):
        _ityp, shell_l, n_prim, _charge, scale = tokens[row]
        row += 1
        primitives = tokens[row:row + int(n_prim)]
        row += int(n_prim)
        factor = float(scale) ** 2
        for column, l in enumerate(CRYSTAL_L[int(shell_l)]):
            basis.append([l] + [[float(p[0]) * factor, float(p[1 + column])] for p in primitives])
    return z, basis, ecp_text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # These imports belong exclusively to the external reference process.
    import numpy as np
    import pyscf
    import scipy.linalg

    root = Path(__file__).resolve().parents[2] / "python/vibeqc/basis_library/sources"
    records = {"description": "independent PySCF reference for the Ag V_ECP of AgCl (#515)",
               "program": "PySCF", "version": pyscf.__version__,
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "geometry_bohr": {"Ag": [0.0, 0.0, 0.0], "Cl": [HALF_DIAGONAL] * 3},
               "bases": {}}
    for name, folder in BASES.items():
        paths = {s: root / folder / f"{z}_{s}" for s, z in (("Ag", 47), ("Cl", 17))}
        _, ag_basis, ag_ecp = parse_crystal_record(paths["Ag"], "Ag")
        _, cl_basis, cl_ecp = parse_crystal_record(paths["Cl"], "Cl")
        if ag_ecp is None or cl_ecp is not None:
            raise ValueError(f"{name}: expected an ECP on Ag and none on Cl")
        molecule = pyscf.gto.M(
            atom=[("Ag", [0.0, 0.0, 0.0]), ("Cl", [HALF_DIAGONAL] * 3)],
            basis={"Ag": ag_basis, "Cl": cl_basis},
            ecp={"Ag": pyscf.gto.basis.parse_ecp(ag_ecp)},
            unit="Bohr", cart=False, verbose=0,
        )
        v_ecp = molecule.intor("ECPscalar_sph")
        overlap = molecule.intor("int1e_ovlp_sph")
        eigenvalues = np.sort(scipy.linalg.eigh(v_ecp, overlap, eigvals_only=True))
        records["bases"][name] = {
            "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in paths.values()},
            "ecp_nwchem": ag_ecp,
            "n_ao": int(molecule.nao_nr()),
            "trace": float(np.trace(v_ecp)),
            "generalized_eigenvalues": [float(x) for x in eigenvalues],
        }
    args.output.write_text(json.dumps(records, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
