"""C++ ROHF-reference UCCSD / UCCSD(T) pinned to the spin-orbital anchor.

The C++ kernel (cpp/src/uccsd.cpp via run_uccsd_from_mos) is
reference-agnostic — it accepts MO coefficients and per-spin Fock
matrices directly. ROHF supplies a single set of spatial orbitals
(C_alpha == C_beta) with per-spin Fock matrices (F_alpha != F_beta);
the occupied/virtual partition differs per spin via multiplicity.

The spin-orbital original lives in
``vibeqc.dlpno._ccsd_ref.run_ref_rohf_ccsd`` (a thin wrapper around the
reference-agnostic ``run_ref_uccsd``, which is itself FCI-anchored and
validated against PySCF). These tests pin the C++ correlation and (T)
energies to that anchor on small open-shell radicals with ROHF references,
so the C++ port cannot drift from the validated equation set.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import BasisSet, Molecule, run_rohf
from vibeqc._vibeqc_core import Atom
from vibeqc.cc import CCSDOptions, run_rohf_ccsd
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno._ccsd_ref import run_ref_rohf_ccsd
from vibeqc.rohf import ROHFOptions

ANGSTROM_TO_BOHR = 1.8897259886
MICRO_HA = 1.0e-6

# ── Test systems (atoms in Bohr, multiplicity) ─────────────────────────────

OH_ATOMS = [
    (8, [0.0, 0.0, 0.108444 * ANGSTROM_TO_BOHR]),
    (1, [0.0, 0.0, -0.867550 * ANGSTROM_TO_BOHR]),
]

NH2_ATOMS = [
    (7, [0.0, 0.0, 0.139456 * ANGSTROM_TO_BOHR]),
    (1, [0.0, 1.443510 * ANGSTROM_TO_BOHR, -0.487243 * ANGSTROM_TO_BOHR]),
    (1, [0.0, -1.443510 * ANGSTROM_TO_BOHR, -0.487243 * ANGSTROM_TO_BOHR]),
]

O2_ATOMS = [
    (8, [0.0, 0.0, 0.603555 * ANGSTROM_TO_BOHR]),
    (8, [0.0, 0.0, -0.603555 * ANGSTROM_TO_BOHR]),
]

AUX = "def2-svp-rifit"


# ── Helpers ────────────────────────────────────────────────────────────────


def _rohf(atoms, basis_name, mult):
    mol = Molecule([Atom(z, p) for z, p in atoms], charge=0, multiplicity=mult)
    basis = BasisSet(mol, basis_name)
    opts = ROHFOptions(density_fit=True, aux_basis=AUX, max_iter=200)
    result = run_rohf(mol, basis, opts)
    assert result.converged, f"ROHF not converged: {basis_name}"
    return mol, basis, result


def _anchor(mol, basis, rohf, *, triples, n_frozen=0):
    """Spin-orbital reference correlation / (T) on the same ROHF + DF."""
    df = DensityFitting(basis, BasisSet(mol, AUX), aux_basis_name=AUX)
    Ca = np.asarray(rohf.mo_coeffs_alpha)
    Cb = np.asarray(rohf.mo_coeffs_beta)
    Fa = np.asarray(rohf.fock_alpha)
    Fb = np.asarray(rohf.fock_beta)
    n_elec = mol.n_electrons()
    two_s = mol.multiplicity - 1
    n_a = (n_elec + two_s) // 2
    n_b = n_elec - n_a
    n_a_active = n_a - n_frozen
    n_b_active = n_b - n_frozen
    # Slice active orbitals (skip frozen core).
    n_orb = Ca.shape[1]
    Ca_act = Ca[:, np.r_[n_frozen:n_a, n_a:n_orb]]
    Cb_act = Cb[:, np.r_[n_frozen:n_b, n_b:n_orb]]
    ref = run_ref_rohf_ccsd(
        Ca_act.T @ Fa @ Ca_act,
        Cb_act.T @ Fb @ Cb_act,
        df.mo_transform(Ca_act, Ca_act),
        df.mo_transform(Cb_act, Cb_act),
        n_a_active,
        n_b_active,
        e_hf=rohf.energy,
        compute_triples=triples,
    )
    assert ref.converged
    return ref


def _cpp(mol, basis, rohf, *, triples, n_frozen=0):
    opts = CCSDOptions(aux_basis=AUX, compute_triples=triples, n_frozen_core=n_frozen)
    return run_rohf_ccsd(mol, basis, rohf, opts)


# ── Anchor gate (always-on) ────────────────────────────────────────────────


class TestCppAnchorOH:
    @pytest.fixture(scope="class")
    @classmethod
    def oh(cls):
        return _rohf(OH_ATOMS, "sto-3g", 2)

    def test_uccsd_t_matches_anchor(self, oh):
        mol, basis, rohf = oh
        ref = _anchor(mol, basis, rohf, triples=True)
        res = _cpp(mol, basis, rohf, triples=True)
        assert res.converged
        assert abs(res.e_ccsd_correlation - ref.e_corr) < 5.0 * MICRO_HA
        assert abs(res.e_t - ref.e_t) < 5.0 * MICRO_HA
        assert abs(res.e_ccsd_t - (ref.e_hf + ref.e_corr + ref.e_t)) < 1e-9

    def test_result_identities(self, oh):
        mol, basis, rohf = oh
        res = _cpp(mol, basis, rohf, triples=True)
        assert abs(res.e_hf - rohf.energy) < 1e-10
        assert abs(res.e_ccsd - (res.e_hf + res.e_ccsd_correlation)) < 1e-10
        assert abs(res.e_ccsd_t - (res.e_ccsd + res.e_t)) < 1e-10
        assert abs(res.e_total - res.e_ccsd_t) < 1e-10


class TestCppAnchorNH2:
    @pytest.fixture(scope="class")
    @classmethod
    def nh2(cls):
        return _rohf(NH2_ATOMS, "sto-3g", 2)

    def test_uccsd_t_matches_anchor(self, nh2):
        mol, basis, rohf = nh2
        ref = _anchor(mol, basis, rohf, triples=True)
        res = _cpp(mol, basis, rohf, triples=True)
        assert res.converged
        assert abs(res.e_ccsd_correlation - ref.e_corr) < 5.0 * MICRO_HA
        assert abs(res.e_t - ref.e_t) < 5.0 * MICRO_HA


class TestCppAnchorO2:
    @pytest.fixture(scope="class")
    @classmethod
    def o2(cls):
        return _rohf(O2_ATOMS, "sto-3g", 3)

    def test_uccsd_t_matches_anchor(self, o2):
        mol, basis, rohf = o2
        ref = _anchor(mol, basis, rohf, triples=True)
        res = _cpp(mol, basis, rohf, triples=True)
        assert res.converged
        assert abs(res.e_ccsd_correlation - ref.e_corr) < 5.0 * MICRO_HA
        assert abs(res.e_t - ref.e_t) < 5.0 * MICRO_HA


class TestFrozenCore:
    def test_frozen_core_window_is_consistent(self):
        """ROHF CCSD(T) with frozen core: OH/sto-3g, n_frozen=1."""
        mol, basis, rohf = _rohf(OH_ATOMS, "sto-3g", 2)
        res = _cpp(mol, basis, rohf, triples=True, n_frozen=1)
        assert res.converged
        assert res.n_iter < 100
        full = _cpp(mol, basis, rohf, triples=True, n_frozen=0)
        assert abs(res.e_ccsd_correlation) < abs(full.e_ccsd_correlation)


def test_rohf_ccsd_rejects_unconverged_rohf():
    mol, basis, _ = _rohf(OH_ATOMS, "sto-3g", 2)
    from vibeqc.rohf import ROHFResult

    fake = ROHFResult(
        energy=0.0,
        e_electronic=0.0,
        n_iter=0,
        converged=False,
        mo_energies=np.zeros((1,)),
        mo_coeffs=np.zeros((1, 1)),
        mo_occupations=np.zeros((1,)),
        density=np.zeros((1, 1)),
        density_alpha=np.zeros((1, 1)),
        density_beta=np.zeros((1, 1)),
        fock=np.zeros((1, 1)),
        fock_alpha=np.zeros((1, 1)),
        fock_beta=np.zeros((1, 1)),
        s_squared=0.0,
        s_squared_ideal=0.0,
        n_alpha=0,
        n_beta=0,
    )
    with pytest.raises(RuntimeError, match="not converged"):
        run_rohf_ccsd(mol, basis, fake)
