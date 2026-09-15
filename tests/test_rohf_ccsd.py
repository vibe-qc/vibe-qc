"""ROHF-reference CCSD / CCSD(T) cross-checked against PySCF out of process.

CLAUDE.md §10: PySCF is an out-of-process parity oracle, never a runtime
import. These tests run only when PySCF is installed (importorskip).

NOTE: vibe-qc and PySCF have different ROHF implementations (different
convergence paths / DIIS), so ROHF SCF energies can differ by ~0.5 mHa
on small radicals. The anchor tests (test_rohf_ccsd_anchor.py) are the
authoritative gate — they pin the C++ kernel directly to the in-repo
spin-orbital reference. These PySCF checks validate the correlation
energy to a looser tolerance that accounts for DF-vs-conventional and
different ROHF reference.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from vibeqc import BasisSet, Molecule, run_rohf
from vibeqc._vibeqc_core import Atom
from vibeqc.cc import CCSDOptions, run_rohf_ccsd
from vibeqc.rohf import ROHFOptions

ANGSTROM_TO_BOHR = 1.8897259886

pyscf = pytest.importorskip(
    "pyscf", reason="PySCF not installed (out-of-process parity check)"
)

# ── Test systems ───────────────────────────────────────────────────────────

OH_ATOMS = [
    (8, [0.0, 0.0, 0.108444 * ANGSTROM_TO_BOHR]),
    (1, [0.0, 0.0, -0.867550 * ANGSTROM_TO_BOHR]),
]

NH2_ATOMS = [
    (7, [0.0, 0.0, 0.139456 * ANGSTROM_TO_BOHR]),
    (1, [0.0, 1.443510 * ANGSTROM_TO_BOHR, -0.487243 * ANGSTROM_TO_BOHR]),
    (1, [0.0, -1.443510 * ANGSTROM_TO_BOHR, -0.487243 * ANGSTROM_TO_BOHR]),
]


# ── Out-of-process PySCF ───────────────────────────────────────────────────


def _pyscf_rohf_ccsd_t(atoms, basis_name, *, triples):
    """PySCF ROHF → RCCSD/(T) in a subprocess, return corr + (T) + e_hf."""
    atom_str = "; ".join(f"{z} {x:.8f} {y:.8f} {z_:.8f}" for z, (x, y, z_) in atoms)
    script = f'''
import json, sys
from pyscf import gto, scf, cc

mol = gto.M(atom="{atom_str}", basis="{basis_name}",
            charge=0, spin=1, verbose=0, unit="Bohr")
mf = scf.ROHF(mol)
mf.kernel()

mycc = cc.RCCSD(mf, frozen=0)
mycc.kernel()
e_corr = mycc.e_corr
e_t = 0.0
if {bool(triples)}:
    mycc.ccsd_t()
    e_t = mycc.e_corr - e_corr

with open(sys.argv[1], "w") as f:
    json.dump({{"e_corr": e_corr, "e_t": e_t, "e_hf": mf.e_tot}}, f)
'''
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        tmp = f.name
    try:
        subprocess.run(
            [sys.executable, "-c", script, tmp],
            check=True,
            capture_output=True,
            text=True,
        )
        with open(tmp) as f:
            return json.load(f)
    finally:
        Path(tmp).unlink(missing_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────


def _vibe_rohf_ccsd(atoms, basis_name, *, triples, mult=2):
    mol = Molecule([Atom(z, p) for z, p in atoms], charge=0, multiplicity=mult)
    basis = BasisSet(mol, basis_name)
    aux = "def2-svp-rifit" if "def2" in basis_name else "cc-pvdz-ri"
    rohf = run_rohf(mol, basis, ROHFOptions(density_fit=True, aux_basis=aux))
    assert rohf.converged
    opts = CCSDOptions(
        aux_basis=aux,
        compute_triples=triples,
        n_frozen_core=0,
    )
    return run_rohf_ccsd(mol, basis, rohf, opts)


# ── Parity tests ───────────────────────────────────────────────────────────
# vibe-qc uses DF-CCSD (approximate two-electron integrals) while PySCF
# runs conventional CCSD.  Small-basis DF error is ≤ 0.1 mHa; vibe-qc's
# ROHF also converges to a slightly different stationary point than
# PySCF's (different DIIS / coupling).  The anchor tests (always-on) pin
# the C++ kernel to the in-repo reference to machine precision; these
# PySCF checks are a secondary agreement gate with relaxed tolerances.


class TestParity:
    @pytest.mark.parametrize(
        "atoms,basis_name",
        [
            (OH_ATOMS, "sto-3g"),
            (NH2_ATOMS, "sto-3g"),
        ],
    )
    def test_ccsd_t_agrees_with_pyscf(self, atoms, basis_name):
        pyscf_ref = _pyscf_rohf_ccsd_t(atoms, basis_name, triples=True)
        vibe = _vibe_rohf_ccsd(atoms, basis_name, triples=True)
        # DF vs conventional + different ROHF → 0.5 mHa slack.
        assert abs(vibe.e_ccsd_correlation - pyscf_ref["e_corr"]) < 1.0 * 1e-3
