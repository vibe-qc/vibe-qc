"""Generate the PySCF oscillator-strength reference pinned by tests/test_tddft.py (#545).

Run this script with an external PySCF interpreter, never by importing PySCF
into vibe-qc. H2O and the H2O+ doublet in cc-pVDZ (spherical, Bohr): TDA from
``tdscf.TDA``; TDHF from a full diagonalisation of PySCF's own A and B
matrices, because its Davidson solver returned spurious roots for this
system. The TDHF vectors are put on PySCF's own (X, Y) normalisation and
passed to its own ``transition_dipole`` and ``oscillator_strength``, so every
dipole contraction is PySCF's.

Example:
    "$VIBEQC_PYSCF_PYTHON" examples/regression/tddft_oscillator_reference.py \
        --output /tmp/tddft-oscillator-reference.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ATOMS_BOHR = [("O", (0.0, 0.0, 0.2217)), ("H", (0.0, 1.4305, -0.8867)), ("H", (0.0, -1.4305, -0.8867))]
N_STATES = 6


def _tda(td):
    import numpy as np

    td.nstates = N_STATES
    td.conv_tol = 1e-10
    td.kernel()
    return _record(td, np)


def _full_tdhf(td, unrestricted):
    import numpy as np

    if unrestricted:
        (Aaa, Aab, Abb), (Baa, Bab, Bbb) = td.get_ab()
        noa, nva = Aaa.shape[:2]
        nob, nvb = Abb.shape[:2]
        na, nb = noa * nva, nob * nvb
        A = np.block([[Aaa.reshape(na, na), Aab.reshape(na, nb)],
                      [Aab.reshape(na, nb).T, Abb.reshape(nb, nb)]])
        B = np.block([[Baa.reshape(na, na), Bab.reshape(na, nb)],
                      [Bab.reshape(na, nb).T, Bbb.reshape(nb, nb)]])
    else:
        A, B = td.get_ab()
        no, nv = A.shape[:2]
        A, B = A.reshape(no * nv, no * nv), B.reshape(no * nv, no * nv)
    n = A.shape[0]
    w, v = np.linalg.eig(np.block([[A, B], [-B, -A]]))
    roots = [i for i in np.argsort(w.real) if w[i].real > 1e-8][:N_STATES]
    energies, xy = [], []
    for i in roots:
        x, y = v[:n, i].real, v[n:, i].real
        norm = x @ x - y @ y
        if unrestricted:            # PySCF UHF: |X|^2 - |Y|^2 = 1
            x, y = x / np.sqrt(norm), y / np.sqrt(norm)
            xy.append(((x[:na].reshape(noa, nva), x[na:].reshape(nob, nvb)),
                       (y[:na].reshape(noa, nva), y[na:].reshape(nob, nvb))))
        else:                       # PySCF RHF singlet: |X|^2 - |Y|^2 = 1/2
            x, y = x * np.sqrt(0.5 / norm), y * np.sqrt(0.5 / norm)
            xy.append((x.reshape(no, nv), y.reshape(no, nv)))
        energies.append(w[i].real)
    td.e, td.xy, td.converged = np.asarray(energies), xy, [True] * len(energies)
    return _record(td, np)


def _record(td, np):
    mu = np.asarray(td.transition_dipole())
    f = np.asarray(td.oscillator_strength(gauge="length"))
    return [[float(e), float(np.linalg.norm(m)), float(x)] for e, m, x in zip(td.e, mu, f)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    # These imports belong exclusively to the external reference process.
    import pyscf
    from pyscf import gto, scf, tdscf

    neutral = gto.M(atom=ATOMS_BOHR, basis="cc-pvdz", unit="Bohr", cart=False, verbose=0)
    rhf = scf.RHF(neutral).run(conv_tol=1e-12)
    cation = gto.M(atom=ATOMS_BOHR, basis="cc-pvdz", unit="Bohr", cart=False,
                   charge=1, spin=1, verbose=0)
    uhf = scf.UHF(cation).run(conv_tol=1e-12)
    record = {
        "program": "PySCF", "version": pyscf.__version__,
        "columns": ["omega_hartree", "transition_dipole_norm_au", "oscillator_strength"],
        "rhf_tda": _tda(tdscf.TDA(rhf)),
        "rhf_tdhf": _full_tdhf(tdscf.TDHF(rhf), unrestricted=False),
        "uhf_tda": _tda(tdscf.TDA(uhf)),
        "uhf_tdhf": _full_tdhf(tdscf.TDHF(uhf), unrestricted=True),
    }
    args.output.write_text(json.dumps(record, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
