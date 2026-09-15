"""Open-shell DLPNO-UCCSD(T) pilot correctness gates (M2).

The open-shell sibling of ``test_dlpno_ccsd.py``. The pilot
(`dlpno.uccsd.run_dlpno_uccsd_pilot`, spin-orbital subspace-projected
UCCSD on a UHF reference) is validated against the FCI-anchored spin-orbital
anchor `dlpno._ccsd_ref.run_ref_uccsd` (the same kernel that anchors
``cpp/src/uccsd.cpp``):

* **Exactness / projection ratchet** -- untruncated (tcut_pno=0) the pilot
  reproduces the anchor to ~µHa, for canonical *and* Boys-localised
  occupieds (full-rank PNO rotations exercise expand/project), and with a
  frozen core. A sign/transpose/projection bug breaks this.
* **(T)** -- at full domains the perturbative triples match the anchor's
  ``run_ref_uccsd(compute_triples=True)`` (``localise="none"`` exact in the
  spin-orbital basis; ``"boys"`` via the canonical-rotation path).
* **Truncation** -- ``tcut_pno=1e-7`` recovers ~99.x% with real PNO
  compression on a DZ basis (slow lane).
* **Native local-residual prerequisite** -- the caller-supplied spin-orbital
  per-pair kernel is pinned element-wise to the same Python residual oracle.
  This is a production-engine building block, not a public open-shell route.
* **Native local-triples prerequisite** -- one caller-supplied occupied
  triple and arbitrary virtual domain are pinned to the spin-orbital (T)
  oracle before any local TNO assembly is enabled.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from vibeqc import BasisSet, UHFOptions, run_uhf
from vibeqc._vibeqc_core import Atom, Molecule
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno._ccsd_ref import run_ref_uccsd
from vibeqc.dlpno.uccsd import (
    DLPNOUCCSDPilotOptions as _DLPNOUCCSDPilotOptions,
    run_dlpno_uccsd_pilot,
)

A = 1.8897259886
AUX = "def2-svp-rifit"
MICRO_HA = 1e-6

# Preserve the all-electron, pre-#448 pilot convention behind the recorded
# UCCSD exactness and truncation evidence.
DLPNOUCCSDPilotOptions = partial(
    _DLPNOUCCSDPilotOptions,
    n_frozen=0,
    tcut_pno=1e-7,
)

OH = [(8, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 0.97 * A])]
CH3 = [
    (6, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.0, 1.079 * A]),
    (1, [1.018 * A, 0.0, -0.36 * A]),
    (1, [-0.509 * A, 0.881 * A, -0.36 * A]),
]


def _setup(atoms, basis_name):
    mol = Molecule([Atom(z, p) for z, p in atoms], charge=0, multiplicity=2)
    basis = BasisSet(mol, basis_name)
    opts = UHFOptions()
    opts.max_iter = 300
    uhf = run_uhf(mol, basis, opts)
    assert uhf.converged
    df = DensityFitting(basis, BasisSet(mol, AUX), aux_basis_name=AUX)
    return mol, basis, uhf, df


def _ref(mol, uhf, df, nf=0, triples=False):
    """The spin-orbital UCCSD(T) anchor on the same DF integrals.

    ``run_ref_uccsd`` has no frozen-core argument, so a frozen core is
    realised by restricting the MO set to the active orbitals (drop the first
    ``nf`` columns of each spin) -- exactly what the pilot correlates.
    """
    Ca = np.asarray(uhf.mo_coeffs_alpha)[:, nf:]
    Cb = np.asarray(uhf.mo_coeffs_beta)[:, nf:]
    Fa = np.asarray(uhf.fock_alpha)
    Fb = np.asarray(uhf.fock_beta)
    ne, twos = mol.n_electrons(), mol.multiplicity - 1
    na, nb = (ne + twos) // 2, (ne - twos) // 2
    return run_ref_uccsd(
        Ca.T @ Fa @ Ca,
        Cb.T @ Fb @ Cb,
        df.mo_transform(Ca, Ca),
        df.mo_transform(Cb, Cb),
        na - nf,
        nb - nf,
        e_hf=uhf.energy,
        compute_triples=triples,
    )


NO_TRUNCATION = dict(tcut_pno=0.0)


@pytest.fixture(scope="module")
def oh_sto3g():
    return _setup(OH, "sto-3g")


class TestAnchorReproduced:
    """run_ref_uccsd on OH/STO-3G is the documented anchor (e_corr ≈ -0.024515)."""

    def test_anchor_value(self, oh_sto3g):
        mol, _, uhf, df = oh_sto3g
        ref = _ref(mol, uhf, df, triples=True)
        assert ref.converged
        # OH/STO-3G spin-orbital UCCSD anchor for the open-shell pilot.
        assert ref.e_corr == pytest.approx(-0.02451473, abs=5e-7)
        assert ref.e_t < 0.0  # (T) lowers the energy


class TestPilotExactnessLimit:
    """Untruncated DLPNO-UCCSD pilot ≡ the spin-orbital anchor."""

    @pytest.mark.parametrize("localise", ["none", "boys"])
    def test_full_equals_anchor(self, oh_sto3g, localise):
        mol, basis, uhf, df = oh_sto3g
        ref = _ref(mol, uhf, df)
        r = run_dlpno_uccsd_pilot(
            mol, basis, uhf, df,
            DLPNOUCCSDPilotOptions(localise=localise, **NO_TRUNCATION),
        )
        assert r.converged
        assert abs(r.e_corr - ref.e_corr) < MICRO_HA

    def test_frozen_core_exact(self, oh_sto3g):
        mol, basis, uhf, df = oh_sto3g
        ref = _ref(mol, uhf, df, nf=1)
        r = run_dlpno_uccsd_pilot(
            mol, basis, uhf, df,
            DLPNOUCCSDPilotOptions(localise="none", n_frozen=1, **NO_TRUNCATION),
        )
        assert r.n_frozen == 1
        assert r.converged
        assert abs(r.e_corr - ref.e_corr) < MICRO_HA

    def test_diagonal_pairs_excluded(self, oh_sto3g):
        # Diagonal spin-orbital pairs (P,P) vanish by Pauli antisymmetry, so
        # the pair count is strictly P<Q over the no occupied spin orbitals.
        mol, basis, uhf, df = oh_sto3g
        r = run_dlpno_uccsd_pilot(
            mol, basis, uhf, df,
            DLPNOUCCSDPilotOptions(localise="none", **NO_TRUNCATION),
        )
        ne, twos = mol.n_electrons(), mol.multiplicity - 1
        no = (ne + twos) // 2 + (ne - twos) // 2  # n_occ_a + n_occ_b
        assert r.n_pairs == no * (no - 1) // 2


class TestTriples:
    """(T) correction matches the anchor at full domains (both localise modes)."""

    @pytest.mark.parametrize("localise", ["none", "boys"])
    def test_full_triples_equal_anchor(self, oh_sto3g, localise):
        mol, basis, uhf, df = oh_sto3g
        ref = _ref(mol, uhf, df, triples=True)
        r = run_dlpno_uccsd_pilot(
            mol, basis, uhf, df,
            DLPNOUCCSDPilotOptions(
                localise=localise, compute_triples=True, **NO_TRUNCATION
            ),
        )
        assert r.converged
        assert abs(r.e_corr - ref.e_corr) < MICRO_HA
        assert abs(r.e_t - ref.e_t) < 0.05 * MICRO_HA
        assert r.e_total == pytest.approx(
            r.e_hf + r.e_corr + r.e_t, abs=1e-12
        )


class TestStructureAndGuards:
    def test_unknown_localise_raises(self, oh_sto3g):
        mol, basis, uhf, df = oh_sto3g
        with pytest.raises(ValueError):
            run_dlpno_uccsd_pilot(
                mol, basis, uhf, df, DLPNOUCCSDPilotOptions(localise="pipek")
            )

    def test_max_nbf_guard(self, oh_sto3g):
        mol, basis, uhf, df = oh_sto3g
        with pytest.raises(ValueError, match="max_nbf"):
            run_dlpno_uccsd_pilot(
                mol, basis, uhf, df, DLPNOUCCSDPilotOptions(max_nbf=3)
            )

    def test_total_energy_composition(self, oh_sto3g):
        mol, basis, uhf, df = oh_sto3g
        r = run_dlpno_uccsd_pilot(
            mol, basis, uhf, df,
            DLPNOUCCSDPilotOptions(localise="none", **NO_TRUNCATION),
        )
        assert r.e_total == pytest.approx(r.e_hf + r.e_corr, abs=1e-12)


class TestNativeLocalResidualKernel:
    """Native spin-orbital residual prerequisite for local UCCSD."""

    @staticmethod
    def _case():
        rng = np.random.default_rng(20260714)
        no, nv, naux = 3, 4, 6
        n = no + nv

        # A generic spin-orbital DF tensor. Symmetric charge-density blocks
        # retain the physical B[p,q] = B[q,p] contract while the dense random
        # mixing exercises a local PNO gauge that is not alpha/beta block
        # diagonal.
        b_so = rng.normal(scale=0.15, size=(naux, n, n))
        b_so = 0.5 * (b_so + b_so.transpose(0, 2, 1))

        f_so = rng.normal(scale=0.03, size=(n, n))
        f_so = 0.5 * (f_so + f_so.T)
        np.fill_diagonal(
            f_so,
            np.array([-1.40, -1.05, -0.72, 0.25, 0.54, 0.83, 1.16]),
        )
        t1 = rng.normal(scale=0.02, size=(no, nv))
        raw_t2 = rng.normal(scale=0.01, size=(no, no, nv, nv))
        t2 = 0.25 * (
            raw_t2
            - raw_t2.swapaxes(0, 1)
            - raw_t2.swapaxes(2, 3)
            + raw_t2.transpose(1, 0, 3, 2)
        )
        return no, nv, b_so, f_so, t1, t2

    def test_matches_spin_orbital_oracle_elementwise(self):
        from vibeqc._vibeqc_core import dlpno_uccsd_pair_residual
        from vibeqc.dlpno._ccsd_ref import so_residuals

        no, nv, b_so, f_so, t1, t2 = self._case()
        n = no + nv
        chem = np.einsum("Ppr,Pqs->pqrs", b_so, b_so, optimize=True)
        eri = chem - chem.transpose(0, 1, 3, 2)
        fock_od = f_so.copy()
        np.fill_diagonal(fock_od, 0.0)
        eps = np.diag(f_so)
        d1 = eps[:no, None] - eps[None, no:]
        d2 = (
            eps[:no, None, None, None]
            + eps[None, :no, None, None]
            - eps[None, None, no:, None]
            - eps[None, None, None, no:]
        )
        ref_r1, ref_r2 = so_residuals(
            f_so,
            fock_od,
            eri,
            t1,
            t2,
            slice(0, no),
            slice(no, n),
            d1,
            d2,
        )

        native_r1, native_r2_flat = dlpno_uccsd_pair_residual(
            np.ascontiguousarray(t1),
            np.ascontiguousarray(t2.reshape(no * no, nv * nv)),
            np.ascontiguousarray(b_so.reshape(b_so.shape[0], n * n)),
            np.ascontiguousarray(f_so),
        )

        np.testing.assert_allclose(native_r1, ref_r1, atol=2e-12, rtol=2e-12)
        np.testing.assert_allclose(
            np.asarray(native_r2_flat).reshape(no, no, nv, nv),
            ref_r2,
            atol=2e-12,
            rtol=2e-12,
        )

    @pytest.mark.parametrize("bad_input", ["t2", "b_so", "f_so"])
    def test_rejects_inconsistent_shapes(self, bad_input):
        from vibeqc._vibeqc_core import dlpno_uccsd_pair_residual

        no, nv, b_so, f_so, t1, t2 = self._case()
        n = no + nv
        t2_flat = np.ascontiguousarray(t2.reshape(no * no, nv * nv))
        b_flat = np.ascontiguousarray(b_so.reshape(b_so.shape[0], n * n))
        if bad_input == "t2":
            t2_flat = t2_flat[:-1]
        elif bad_input == "b_so":
            b_flat = b_flat[:, :-1]
        else:
            f_so = f_so[:-1]

        with pytest.raises(ValueError, match=rf"(?i){bad_input}"):
            dlpno_uccsd_pair_residual(t1, t2_flat, b_flat, f_so)


class TestNativeLocalTriplesKernel:
    """Native spin-orbital per-triple prerequisite for local UCCSD(T)."""

    @staticmethod
    def _case():
        rng = np.random.default_rng(20260720)
        no, nv, naux = 4, 3, 7
        n = no + nv

        b_so = rng.normal(scale=0.2, size=(naux, n, n))
        b_so = 0.5 * (b_so + b_so.transpose(0, 2, 1))
        chem = np.einsum("Ppr,Pqs->pqrs", b_so, b_so, optimize=True)
        eri = chem - chem.transpose(0, 1, 3, 2)

        t1 = rng.normal(scale=0.02, size=(no, nv))
        raw_t2 = rng.normal(scale=0.01, size=(no, no, nv, nv))
        t2 = 0.25 * (
            raw_t2
            - raw_t2.swapaxes(0, 1)
            - raw_t2.swapaxes(2, 3)
            + raw_t2.transpose(1, 0, 3, 2)
        )
        eps_o = np.array([-1.4, -1.1, -0.8, -0.55])
        eps_v = np.array([0.25, 0.6, 1.05])
        return (
            t1,
            t2,
            eri[no:, :no, no:, no:],
            eri[:no, no:, :no, :no],
            eri[:no, :no, no:, no:],
            eps_o,
            eps_v,
        )

    def test_matches_spin_orbital_oracle(self):
        from vibeqc._vibeqc_core import dlpno_spin_orbital_triple_energy
        from vibeqc.dlpno._ccsd_ref import so_triple_energy

        t1, t2, vovv, ovoo, oovv, eps_o, eps_v = self._case()
        no, nv = t1.shape
        reference = so_triple_energy(
            0, 1, 3, t1, t2, vovv, ovoo, oovv, eps_o, eps_v
        )
        native = dlpno_spin_orbital_triple_energy(
            0,
            1,
            3,
            np.ascontiguousarray(t1),
            np.ascontiguousarray(t2.reshape(no * no, nv * nv)),
            np.ascontiguousarray(vovv.reshape(nv * no, nv * nv)),
            np.ascontiguousarray(ovoo.reshape(no * nv, no * no)),
            np.ascontiguousarray(oovv.reshape(no * no, nv * nv)),
            eps_o,
            eps_v,
        )
        assert native == pytest.approx(reference, abs=2e-12, rel=2e-12)

    def test_zero_amplitudes_give_zero(self):
        from vibeqc._vibeqc_core import dlpno_spin_orbital_triple_energy

        t1, t2, vovv, ovoo, oovv, eps_o, eps_v = self._case()
        no, nv = t1.shape
        energy = dlpno_spin_orbital_triple_energy(
            0,
            1,
            2,
            np.zeros_like(t1),
            np.zeros((no * no, nv * nv)),
            np.ascontiguousarray(vovv.reshape(nv * no, nv * nv)),
            np.ascontiguousarray(ovoo.reshape(no * nv, no * no)),
            np.ascontiguousarray(oovv.reshape(no * no, nv * nv)),
            eps_o,
            eps_v,
        )
        assert energy == 0.0

    @pytest.mark.parametrize(
        ("bad_input", "match"),
        [
            ("indices", "occupied indices"),
            ("t2", "T2_flat"),
            ("vovv", "eri_vovv"),
            ("ovoo", "eri_ovoo"),
            ("oovv", "eri_oovv"),
            ("eps", "orbital-energy"),
        ],
    )
    def test_rejects_inconsistent_inputs(self, bad_input, match):
        from vibeqc._vibeqc_core import dlpno_spin_orbital_triple_energy

        t1, t2, vovv, ovoo, oovv, eps_o, eps_v = self._case()
        no, nv = t1.shape
        args = [
            0,
            1,
            2,
            np.ascontiguousarray(t1),
            np.ascontiguousarray(t2.reshape(no * no, nv * nv)),
            np.ascontiguousarray(vovv.reshape(nv * no, nv * nv)),
            np.ascontiguousarray(ovoo.reshape(no * nv, no * no)),
            np.ascontiguousarray(oovv.reshape(no * no, nv * nv)),
            eps_o,
            eps_v,
        ]
        if bad_input == "indices":
            args[1] = 0
        elif bad_input == "t2":
            args[4] = args[4][:-1]
        elif bad_input == "vovv":
            args[5] = args[5][:-1]
        elif bad_input == "ovoo":
            args[6] = args[6][:-1]
        elif bad_input == "oovv":
            args[7] = args[7][:-1]
        else:
            args[8] = args[8][:-1]

        with pytest.raises(ValueError, match=match):
            dlpno_spin_orbital_triple_energy(*args)


@pytest.mark.slow
class TestPilotTruncationDZ:
    """Recovery tiers on a DZ basis (O(N^6) pilot -- slow lane)."""

    @pytest.fixture(scope="class")
    @classmethod
    def oh_dz(cls):
        return _setup(OH, "def2-svp")

    @pytest.mark.parametrize("localise", ["none", "boys"])
    def test_exactness_limit_dz(self, oh_dz, localise):
        mol, basis, uhf, df = oh_dz
        ref = _ref(mol, uhf, df)
        r = run_dlpno_uccsd_pilot(
            mol, basis, uhf, df,
            DLPNOUCCSDPilotOptions(localise=localise, **NO_TRUNCATION),
        )
        assert r.converged
        assert abs(r.e_corr - ref.e_corr) < MICRO_HA

    def test_truncation_recovers_and_compresses(self, oh_dz):
        mol, basis, uhf, df = oh_dz
        ref = _ref(mol, uhf, df)
        r = run_dlpno_uccsd_pilot(
            mol, basis, uhf, df,
            DLPNOUCCSDPilotOptions(localise="none", tcut_pno=1e-7),
        )
        assert r.converged
        recovery = r.e_corr / ref.e_corr
        # Measured 2026-06-21: 99.84% -- assert the tier with margin.
        assert recovery > 0.997, f"recovery {recovery:.5%}"
        assert recovery < 1.003, f"recovery {recovery:.5%} (overshoot)"
        n_vir = 2 * (basis.nbasis - 5)  # spin-orbital virtuals (na=5, nb=4)
        assert r.avg_pno < 0.9 * n_vir  # real PNO compression

    def test_full_triples_dz_equal_anchor(self, oh_dz):
        mol, basis, uhf, df = oh_dz
        ref = _ref(mol, uhf, df, triples=True)
        r = run_dlpno_uccsd_pilot(
            mol, basis, uhf, df,
            DLPNOUCCSDPilotOptions(
                localise="boys", compute_triples=True, **NO_TRUNCATION
            ),
        )
        assert r.converged
        assert abs(r.e_t - ref.e_t) < MICRO_HA
