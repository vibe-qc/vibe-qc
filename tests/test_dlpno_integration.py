"""End-to-end integration test: DLPNO on H2O/STO-3G with real DF integrals.

Geometry/reference-energy guards, pair-classification and domain
structure checks, and the M2-gate parity of `run_dlpno_mp2` against
canonical DF-MP2 with the same fitting basis. The pre-M2 placeholder
pipeline this file once exercised was retired in M3b; the validated
physics is covered by tests/test_dlpno_mp2.py and tests/test_dlpno_ccsd.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import BasisSet, MP2Options, RHFOptions, run_mp2, run_rhf
from vibeqc._vibeqc_core import Atom, Molecule, compute_overlap
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno.pairs import classify_pairs
from vibeqc.dlpno.pao import build_all_domains

ANGSTROM_TO_BOHR = 1.8897259886

# Reference RHF energy for this geometry/basis (canonical, also asserted by
# tests/test_mp2.py-style parity runs): catches geometry/unit regressions.
E_HF_REF = -74.964454

AUX_BASIS = "def2-svp-rifit"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_ao_coords(basis) -> np.ndarray:
    """Per-AO coordinates from basis shell origins (spherical shells)."""
    nbf = basis.nbasis
    coords = np.zeros((nbf, 3), dtype=float)
    bf = 0
    for sh in basis.shells():
        n_func = 2 * int(sh.l) + 1
        origin = np.array(sh.origin)
        coords[bf : bf + n_func] = origin
        bf += n_func
    return coords


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


class TestDLPNOEndToEnd:
    """Full DLPNO pipeline on H2O/STO-3G against canonical DF-MP2."""

    @pytest.fixture(scope="class")
    @classmethod
    def h2o_system(cls):
        """Build H2O/STO-3G, run RHF, DF with a real RI fitting basis."""
        mol = Molecule(
            [
                Atom(8, [0.0, 0.0, 0.0]),
                Atom(1, [0.0, 0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
                Atom(1, [0.0, -0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
            ],
            charge=0,
            multiplicity=1,
        )

        basis = BasisSet(mol, "sto-3g")

        opts = RHFOptions()
        opts.max_iter = 100
        rhf = run_rhf(mol, basis, opts)
        assert rhf.converged, "RHF did not converge"

        # Real RI fitting basis — NOT the orbital basis.
        aux = BasisSet(mol, AUX_BASIS)
        df = DensityFitting(basis, aux, aux_basis_name=AUX_BASIS)

        C = rhf.mo_coeffs.copy()
        eps = rhf.mo_energies.copy()
        n_occ = mol.n_electrons() // 2

        # Canonical DF-MP2 with the same fitting basis: the parity anchor.
        mp2_opts = MP2Options()
        mp2_opts.n_frozen_core = 0
        mp2_opts.density_fit = True
        mp2_opts.aux_basis = AUX_BASIS
        mp2 = run_mp2(mol, basis, rhf, mp2_opts)

        return {
            "mol": mol,
            "basis": basis,
            "rhf": rhf,
            "df": df,
            "C": C,
            "eps": eps,
            "C_occ_canon": C[:, :n_occ],
            "eps_occ": eps[:n_occ],
            "n_occ": n_occ,
            "e_mp2_ref": mp2.e_correlation,
        }

    def test_reference_energies_sane(self, h2o_system):
        """Geometry/basis guard: the canonical anchors are textbook values."""
        d = h2o_system
        assert d["rhf"].energy == pytest.approx(E_HF_REF, abs=2e-4)
        # Canonical MP2/STO-3G correlation energy for H2O is ~-0.039 Ha.
        assert -0.055 < d["e_mp2_ref"] < -0.030

    def test_parity_with_canonical_df_mp2(self, h2o_system):
        """M2 gate (ratchet promoted 2026-06-10): DLPNO-MP2 vs canonical DF-MP2.

        The M1 strict-xfail ratchet pointed at the legacy
        ``solve_dlpno_ccsd`` scaffold; M2 closed the gap with the real
        driver (``vibeqc.dlpno.mp2.run_dlpno_mp2``), so the parity test
        now asserts hard against it. The legacy scaffold itself was
        retired in M3b.
        """
        from vibeqc.dlpno.mp2 import DLPNOMP2Options, run_dlpno_mp2

        d = h2o_system
        r = run_dlpno_mp2(
            d["mol"],
            d["basis"],
            d["rhf"],
            d["df"],
            DLPNOMP2Options(
                n_frozen=0,
                tcut_pno=1e-8,
                tcut_pno_weak=1e-7,
                tcut_mkn=1e-3,
                tcut_pairs=1e-6,
                tcut_pairs_weak=1e-4,
            ),
        )
        assert r.converged
        gap = abs(r.e_corr - d["e_mp2_ref"]) / abs(d["e_mp2_ref"])
        assert gap < 0.002, f"DLPNO-MP2 gap vs canonical DF-MP2: {gap:.4%}"

    def test_pair_classification_counts(self, h2o_system):
        """H2O/STO-3G: all i<j pairs enumerated (diagonal pairs are an M2 gap)."""
        d = h2o_system
        coords = _build_ao_coords(d["basis"])
        S = np.asarray(compute_overlap(d["basis"])).copy()

        pairs = classify_pairs(
            d["C_occ_canon"], S, coords, tcut_pairs=1e-4, tcut_pairs_weak=1e-4
        )
        n_occ = d["n_occ"]
        n_expected = n_occ * (n_occ - 1) // 2
        assert pairs.n_total == n_expected
        print(f"\nPairs: S={pairs.n_strong} W={pairs.n_weak} D={pairs.n_distant}")

    def test_domain_construction(self, h2o_system):
        """H2O/STO-3G: domains built for all strong pairs, PAOs well-formed."""
        d = h2o_system
        coords = _build_ao_coords(d["basis"])
        S = np.asarray(compute_overlap(d["basis"])).copy()

        pairs = classify_pairs(d["C_occ_canon"], S, coords, tcut_pairs=1e-2)
        domains_result = build_all_domains(
            d["mol"], d["basis"], d["C_occ_canon"], S, pairs, tcut_mkn=1e-3
        )
        n_domains = len(domains_result.domains)
        print(f"\nDomains: {n_domains} built for {pairs.n_strong} strong pairs")

        for dom in domains_result.domains.values():
            assert dom.n_pao > 0
            assert len(dom.atom_indices) > 0
            assert dom.C_pao.shape[1] == dom.n_pao
