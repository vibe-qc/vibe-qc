"""Open-shell DLPNO-UMP2 (M1) -- the exactness ratchet vs canonical UMP2.

`dlpno.ump2.run_dlpno_ump2` resolves the UHF-reference MP2 correlation into
the three spin channels (αα/ββ same-spin antisymmetrised + αβ opposite-spin)
and builds per-spin-pair PNOs from the canonical-occupied pair densities.

The M1 deliverable: with full domains and no PNO truncation (tcut_pno=0) it
reproduces canonical UMP2 -- channel by channel -- to machine precision; with
truncation it recovers ~99.9% with real PNO compression.

Anchor: the in-test canonical UMP2 is `correlation.ump2_energy` (Szabo &
Ostlund §6.7, validated against `cpp/src/ump2.cpp`) fed the *same* DF (ov|ov)
integrals, so the exactness limit is a like-for-like comparison.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from vibeqc import BasisSet, UHFOptions, run_uhf
from vibeqc._vibeqc_core import Atom, Molecule
from vibeqc.correlation import ump2_energy
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno.ump2 import DLPNOUMP2Options as _DLPNOUMP2Options, run_dlpno_ump2

A = 1.8897259886
AUX = "def2-svp-rifit"
MICRO_HA = 1e-6

# Retain the all-electron, pre-#448 UMP2 threshold convention used when these
# exactness and compression references were established.
DLPNOUMP2Options = partial(
    _DLPNOUMP2Options,
    n_frozen=0,
    tcut_pno=1e-8,
    tcut_pairs=0.0,
)

OH = [(8, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 0.97 * A])]
CH3 = [
    (6, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.0, 1.079 * A]),
    (1, [1.018 * A, 0.0, -0.36 * A]),
    (1, [-0.509 * A, 0.881 * A, -0.36 * A]),
]


def _setup(atoms, mult, basis_name):
    mol = Molecule([Atom(z, p) for z, p in atoms], charge=0, multiplicity=mult)
    basis = BasisSet(mol, basis_name)
    opts = UHFOptions()
    opts.max_iter = 300
    uhf = run_uhf(mol, basis, opts)
    assert uhf.converged
    df = DensityFitting(basis, BasisSet(mol, AUX), aux_basis_name=AUX)
    return mol, basis, uhf, df


def _canonical_ump2(mol, uhf, df, nf=0):
    """Canonical UMP2 from the same DF (ov|ov) integrals -- the anchor."""
    Ca, Cb = np.asarray(uhf.mo_coeffs_alpha), np.asarray(uhf.mo_coeffs_beta)
    ea, eb = np.asarray(uhf.mo_energies_alpha), np.asarray(uhf.mo_energies_beta)
    ne, two_s = mol.n_electrons(), mol.multiplicity - 1
    na, nb = (ne + two_s) // 2, (ne - two_s) // 2
    Ba = np.asarray(df.mo_transform(Ca[:, nf:na], Ca[:, na:]))
    Bb = np.asarray(df.mo_transform(Cb[:, nf:nb], Cb[:, nb:]))
    ovaa = np.einsum("Pia,Pjb->iajb", Ba, Ba, optimize=True)
    ovbb = np.einsum("Pia,Pjb->iajb", Bb, Bb, optimize=True)
    ovab = np.einsum("Pia,Pjb->iajb", Ba, Bb, optimize=True)
    return ump2_energy(ea[nf:na], ea[na:], eb[nf:nb], eb[nb:], ovaa, ovbb, ovab)


class TestExactnessRatchet:
    """Full-domain DLPNO-UMP2 == canonical UMP2, channel by channel."""

    @pytest.mark.parametrize(
        "atoms,basis", [(OH, "sto-3g"), (OH, "def2-svp"), (CH3, "def2-svp")]
    )
    def test_full_equals_canonical_ump2(self, atoms, basis):
        mol, b, uhf, df = _setup(atoms, 2, basis)
        ref = _canonical_ump2(mol, uhf, df)
        r = run_dlpno_ump2(mol, b, uhf, df, DLPNOUMP2Options(tcut_pno=0.0))
        assert abs(r.e_aa - ref.e_aa) < MICRO_HA
        assert abs(r.e_bb - ref.e_bb) < MICRO_HA
        assert abs(r.e_ab - ref.e_ab) < MICRO_HA
        assert abs(r.e_corr - ref.e_corr) < MICRO_HA

    def test_frozen_core_exact(self):
        mol, b, uhf, df = _setup(OH, 2, "def2-svp")
        ref = _canonical_ump2(mol, uhf, df, nf=1)
        r = run_dlpno_ump2(
            mol, b, uhf, df, DLPNOUMP2Options(tcut_pno=0.0, n_frozen=1)
        )
        assert r.n_frozen == 1
        assert abs(r.e_corr - ref.e_corr) < MICRO_HA

    def test_all_three_channels_nonzero_on_dz(self):
        # def2-SVP has enough virtuals that every spin channel contributes
        # (the same-spin channels vanish in a minimal basis).
        mol, b, uhf, df = _setup(OH, 2, "def2-svp")
        r = run_dlpno_ump2(mol, b, uhf, df, DLPNOUMP2Options(tcut_pno=0.0))
        assert r.e_aa < -1e-3 and r.e_bb < -1e-3 and r.e_ab < -1e-2


class TestTruncation:
    def test_truncated_recovers_and_compresses(self):
        mol, b, uhf, df = _setup(CH3, 2, "def2-svp")
        ref = _canonical_ump2(mol, uhf, df)
        r = run_dlpno_ump2(mol, b, uhf, df, DLPNOUMP2Options(tcut_pno=1e-8))
        recovery = r.e_corr / ref.e_corr
        assert 0.998 < recovery < 1.001, f"recovery {recovery:.5%}"
        n_vir_a = np.asarray(uhf.mo_coeffs_alpha).shape[0] - (
            (mol.n_electrons() + 1) // 2
        )
        assert r.avg_pno_ab < 0.9 * n_vir_a  # real PNO compression

    def test_truncation_is_monotone(self):
        mol, b, uhf, df = _setup(OH, 2, "def2-svp")
        e0 = run_dlpno_ump2(mol, b, uhf, df, DLPNOUMP2Options(tcut_pno=0.0)).e_corr
        loose = run_dlpno_ump2(
            mol, b, uhf, df, DLPNOUMP2Options(tcut_pno=1e-5)
        ).e_corr
        tight = run_dlpno_ump2(
            mol, b, uhf, df, DLPNOUMP2Options(tcut_pno=1e-9)
        ).e_corr
        # tighter cut -> closer to (more negative, full) correlation
        assert abs(tight - e0) < abs(loose - e0)


class TestStructureAndGuards:
    def test_scs_scaling(self):
        mol, b, uhf, df = _setup(OH, 2, "def2-svp")
        plain = run_dlpno_ump2(mol, b, uhf, df, DLPNOUMP2Options(tcut_pno=0.0))
        scs = run_dlpno_ump2(
            mol, b, uhf, df,
            DLPNOUMP2Options(tcut_pno=0.0, ss_scale=1.0 / 3.0, os_scale=6.0 / 5.0),
        )
        expected = 6.0 / 5.0 * plain.e_ab + 1.0 / 3.0 * (plain.e_aa + plain.e_bb)
        assert abs(scs.e_corr - expected) < 1e-12

    def test_unknown_localise_raises(self):
        mol, b, uhf, df = _setup(OH, 2, "sto-3g")
        with pytest.raises(ValueError):
            run_dlpno_ump2(mol, b, uhf, df, DLPNOUMP2Options(localise="pipek"))


class TestBoysLocalizedRatchet:
    """M1b: Boys-localised coupled LMP2 (full virtual) == canonical UMP2."""

    @pytest.mark.parametrize("atoms,basis", [(OH, "def2-svp"), (CH3, "def2-svp")])
    def test_boys_full_equals_canonical_ump2(self, atoms, basis):
        mol, b, uhf, df = _setup(atoms, 2, basis)
        ref = _canonical_ump2(mol, uhf, df)
        r = run_dlpno_ump2(
            mol, b, uhf, df, DLPNOUMP2Options(localise="boys", tcut_pno=0.0)
        )
        assert r.localise == "boys" and r.converged
        # LMP2 with the full virtual space == canonical MP2, channel by channel.
        assert abs(r.e_aa - ref.e_aa) < MICRO_HA
        assert abs(r.e_bb - ref.e_bb) < MICRO_HA
        assert abs(r.e_ab - ref.e_ab) < MICRO_HA
        assert abs(r.e_corr - ref.e_corr) < MICRO_HA

    def test_boys_frozen_core_exact(self):
        mol, b, uhf, df = _setup(OH, 2, "def2-svp")
        ref = _canonical_ump2(mol, uhf, df, nf=1)
        r = run_dlpno_ump2(
            mol, b, uhf, df,
            DLPNOUMP2Options(localise="boys", tcut_pno=0.0, n_frozen=1),
        )
        assert r.converged
        assert abs(r.e_corr - ref.e_corr) < MICRO_HA

    def test_boys_matches_canonical_path(self):
        # Boys and canonical-occupied paths give the same correlation energy
        # (both reproduce canonical UMP2 at full virtual / no truncation).
        mol, b, uhf, df = _setup(OH, 2, "def2-svp")
        none = run_dlpno_ump2(mol, b, uhf, df, DLPNOUMP2Options(tcut_pno=0.0))
        boys = run_dlpno_ump2(
            mol, b, uhf, df, DLPNOUMP2Options(localise="boys", tcut_pno=0.0)
        )
        assert abs(boys.e_corr - none.e_corr) < MICRO_HA

    def test_total_energy_composition(self):
        mol, b, uhf, df = _setup(OH, 2, "sto-3g")
        r = run_dlpno_ump2(mol, b, uhf, df, DLPNOUMP2Options(tcut_pno=0.0))
        assert r.e_total == pytest.approx(r.e_hf + r.e_corr, abs=1e-12)


class TestBoysPNOTruncation:
    """M1c: Boys + per-pair PNO truncation with cross-pair projections."""

    @staticmethod
    def _boys_channels(mol, b, uhf, df, tcut):
        """Set up Boys-localised B-tensors and run the M1c per-pair PNO
        solvers + the independent M1b full-virtual solvers for cross-check."""
        from vibeqc.dlpno.ump2 import (
            _boys_localise,
            _opp_spin_localized,
            _opp_spin_localized_pno,
            _same_spin_localized,
            _same_spin_localized_pno,
        )

        Ca, Cb = np.asarray(uhf.mo_coeffs_alpha), np.asarray(uhf.mo_coeffs_beta)
        ea, eb = np.asarray(uhf.mo_energies_alpha), np.asarray(uhf.mo_energies_beta)
        Fa, Fb = np.asarray(uhf.fock_alpha), np.asarray(uhf.fock_beta)
        ne, twos = mol.n_electrons(), mol.multiplicity - 1
        na, nb = (ne + twos) // 2, (ne - twos) // 2
        Cao = _boys_localise(Ca[:, :na], b)
        Cbo = _boys_localise(Cb[:, :nb], b)
        Fa_oo, Fb_oo = Cao.T @ Fa @ Cao, Cbo.T @ Fb @ Cbo
        eva, evb = ea[na:], eb[nb:]
        Ba = np.asarray(df.mo_transform(Cao, Ca[:, na:]))
        Bb = np.asarray(df.mo_transform(Cbo, Cb[:, nb:]))
        pno = (
            _same_spin_localized_pno(Ba, Fa_oo, eva, tcut, 300, 1e-10)[0]
            + _same_spin_localized_pno(Bb, Fb_oo, evb, tcut, 300, 1e-10)[0]
            + _opp_spin_localized_pno(Ba, Fa_oo, eva, Bb, Fb_oo, evb, tcut, 300, 1e-10)[0]
        )
        full = (
            _same_spin_localized(Ba, Fa_oo, eva, 300, 1e-10)[0]
            + _same_spin_localized(Bb, Fb_oo, evb, 300, 1e-10)[0]
            + _opp_spin_localized(Ba, Fa_oo, eva, Bb, Fb_oo, evb, 300, 1e-10)[0]
        )
        return pno, full

    def test_projection_ratchet_exact_at_full_domains(self):
        # At tcut_pno=0 each pair's PNOs are full-rank rotations, so the
        # cross-pair projections are exact unitaries: the M1c per-pair-PNO
        # solver must equal both the independent M1b full-virtual solver and
        # canonical UMP2. A sign/transpose/dual-side projection bug breaks this.
        mol, b, uhf, df = _setup(OH, 2, "def2-svp")
        ref = _canonical_ump2(mol, uhf, df)
        pno, full = self._boys_channels(mol, b, uhf, df, tcut=0.0)
        assert abs(pno - full) < MICRO_HA          # two implementations agree
        assert abs(pno - ref.e_corr) < MICRO_HA    # and equal canonical UMP2

    def test_truncation_recovers_and_compresses(self):
        mol, b, uhf, df = _setup(CH3, 2, "def2-svp")
        ref = _canonical_ump2(mol, uhf, df)
        r = run_dlpno_ump2(
            mol, b, uhf, df, DLPNOUMP2Options(localise="boys", tcut_pno=1e-8)
        )
        assert r.localise == "boys" and r.converged
        recovery = r.e_corr / ref.e_corr
        assert 0.99 < recovery < 1.005, f"recovery {recovery:.4%}"
        n_vir_a = np.asarray(uhf.mo_coeffs_alpha).shape[0] - (
            (mol.n_electrons() + 1) // 2
        )
        assert r.avg_pno_ab < 0.9 * n_vir_a  # real PNO compression

    def test_truncation_monotone(self):
        mol, b, uhf, df = _setup(OH, 2, "def2-svp")
        e0 = run_dlpno_ump2(
            mol, b, uhf, df, DLPNOUMP2Options(localise="boys", tcut_pno=0.0)
        ).e_corr
        loose = run_dlpno_ump2(
            mol, b, uhf, df, DLPNOUMP2Options(localise="boys", tcut_pno=1e-5)
        ).e_corr
        tight = run_dlpno_ump2(
            mol, b, uhf, df, DLPNOUMP2Options(localise="boys", tcut_pno=1e-9)
        ).e_corr
        assert abs(tight - e0) < abs(loose - e0)


# OH radical + a water 8 A away (doublet): the inter-fragment pairs are weak,
# so a small tcut_pairs screens them losslessly -- the O(N) locality lever.
OH_H2O = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.0, 0.97 * A]),
    (8, [0.0, 0.0, 8.0 * A]),
    (1, [0.0, 0.757 * A, (8.0 - 0.587) * A]),
    (1, [0.0, -0.757 * A, (8.0 - 0.587) * A]),
]


class TestWeakPairScreening:
    """tcut_pairs: distant localised pairs treated at the semicanonical
    estimate -> O(N) pair list. No-op on compact molecules; lossless on
    separated fragments."""

    def test_compact_noop(self):
        # Every pair in a compact radical is strong, so a small tcut_pairs
        # screens nothing and the energy is bit-identical to no screening.
        mol, b, uhf, df = _setup(OH, 2, "def2-svp")
        base = dict(localise="boys", tcut_pno=1e-8)
        ref = run_dlpno_ump2(mol, b, uhf, df, DLPNOUMP2Options(tcut_pairs=0.0, **base))
        scr = run_dlpno_ump2(mol, b, uhf, df, DLPNOUMP2Options(tcut_pairs=1e-5, **base))
        assert scr.n_screened == 0
        assert abs(scr.e_corr - ref.e_corr) < 1e-12

    def test_monotone_screening_count(self):
        mol, b, uhf, df = _setup(OH, 2, "def2-svp")
        base = dict(localise="boys", tcut_pno=1e-8)
        counts = [
            run_dlpno_ump2(
                mol, b, uhf, df, DLPNOUMP2Options(tcut_pairs=t, **base)
            ).n_screened
            for t in (0.0, 1e-5, 1e-4, 1e-3)
        ]
        assert counts[0] == 0
        assert counts == sorted(counts)
        assert counts[-1] > 0

    @pytest.mark.slow  # 43-bf OH+H2O, several PNO solves
    def test_separated_fragments_lossless(self):
        mol, b, uhf, df = _setup(OH_H2O, 2, "def2-svp")
        base = dict(localise="boys", tcut_pno=1e-8)
        ref = run_dlpno_ump2(mol, b, uhf, df, DLPNOUMP2Options(tcut_pairs=0.0, **base))
        scr = run_dlpno_ump2(mol, b, uhf, df, DLPNOUMP2Options(tcut_pairs=1e-5, **base))
        # the inter-fragment pairs (OH-occ x H2O-occ) all screen, losslessly
        assert scr.n_screened > 50
        assert abs(scr.e_corr - ref.e_corr) < 1e-5  # < 0.01 kcal/mol

    @pytest.mark.slow
    def test_screening_on_full_virtual_path(self):
        # Screening also works on the tcut_pno=0 (M1b full-virtual) path.
        mol, b, uhf, df = _setup(OH_H2O, 2, "def2-svp")
        base = dict(localise="boys", tcut_pno=0.0)
        ref = run_dlpno_ump2(mol, b, uhf, df, DLPNOUMP2Options(tcut_pairs=0.0, **base))
        scr = run_dlpno_ump2(mol, b, uhf, df, DLPNOUMP2Options(tcut_pairs=1e-5, **base))
        assert scr.n_screened > 50
        assert abs(scr.e_corr - ref.e_corr) < 1e-5


def test_fully_spin_polarised_references_are_accepted():
    """``n_beta = 0`` is a valid reference, not a frozen-core error.

    The guard used ``nf >= nbeta``, which rejects the legitimate ``nbeta == 0``
    boundary with a message about ``n_frozen`` that the caller never set. Both
    regimes are covered: a one-electron system has no pairs at all and must
    return exactly zero, while a fully spin-polarised many-electron system has
    genuine same-spin pairs and must return a finite number.
    """
    from vibeqc import BasisSet, UHFOptions, run_uhf
    from vibeqc._vibeqc_core import Atom, Molecule
    from vibeqc.density_fitting import DensityFitting
    from vibeqc.dlpno.ump2 import DLPNOUMP2Options, run_dlpno_ump2

    bohr = 1.8897259886
    aux = "def2-svp-rifit"

    def _run(atoms, multiplicity):
        molecule = Molecule(
            [Atom(z, p) for z, p in atoms], charge=0, multiplicity=multiplicity
        )
        basis = BasisSet(molecule, "def2-svp")
        opts = UHFOptions()
        opts.max_iter = 500
        uhf = run_uhf(molecule, basis, opts)
        assert uhf.converged
        df = DensityFitting(
            basis, BasisSet(molecule, aux), aux_basis_name=aux
        )
        return run_dlpno_ump2(
            molecule,
            basis,
            uhf,
            df,
            DLPNOUMP2Options(tcut_pno=0.0, tcut_pairs=0.0),
        )

    # One electron: no pairs exist, so zero is the only correct answer.
    assert _run([(1, [0.0, 0.0, 0.0])], 2).e_corr == 0.0

    # Linear H3 quartet: three alpha electrons, three alpha-alpha pairs.
    h3 = [
        (1, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.0, 1.6 * bohr]),
        (1, [0.0, 0.0, 3.2 * bohr]),
    ]
    high_spin = _run(h3, 4)
    assert high_spin.e_corr < -1e-5  # genuinely correlated, not silently zero
