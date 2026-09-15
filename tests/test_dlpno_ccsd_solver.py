"""Reduced-scaling local DLPNO-CCSD solver — the M3c parity ratchet.

`dlpno.ccsd_local_solver.run_local_dlpno_ccsd` stores amplitudes in pair PNO
spaces and, by default, contracts each residual in the paper's atom-based
extended PAO domain before projecting it back. The M3c deliverable: in the
full-domain limit it reproduces canonical closed-shell CCSD exactly (the
analogue of the M1/M2 DLPNO-MP2 ratchet, here flipped to a hard gate). The
retained ≈99.9 % recovery figure belongs to the explicitly pinned legacy
recipe below, not the current default.

Anchored: canonical CCSD here is `_ccsd_cs.run_cs_ccsd`, which is
FCI-validated through `_ccsd_ref` (CCSD ≡ FCI for two electrons).
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from vibeqc import BasisSet, RHFOptions, run_rhf
from vibeqc._vibeqc_core import Atom, Molecule
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno._ccsd_cs import run_cs_ccsd
from vibeqc.dlpno.ccsd import (
    DLPNOCCSDPilotOptions as _DLPNOCCSDPilotOptions,
    run_dlpno_ccsd_pilot,
)
from vibeqc.dlpno.ccsd_local_solver import (
    LocalCCSDOptions as _LocalCCSDOptions,
    run_local_dlpno_ccsd,
)
from vibeqc.dlpno.thresholds import options_from_dlpno_thresholds

ANGSTROM_TO_BOHR = 1.8897259886
AUX = "def2-svp-rifit"
MICRO_HA = 1e-6

# These parity and calibration ratchets are pre-unification numerical
# evidence. Explicit factories prevent #140/#448 defaults from silently
# reinterpreting the old reference values.
DLPNOCCSDPilotOptions = partial(
    _DLPNOCCSDPilotOptions,
    n_frozen=0,
    tcut_pno=1e-8,
    tcut_mkn=1e-3,
)
LocalCCSDOptions = partial(
    _LocalCCSDOptions,
    n_frozen=0,
    tcut_pno=1e-7,
    tcut_mkn=0.0,
    tcut_pairs=1e-4,
    residual_domain="pair",
)

H2_ATOMS = [(1, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 1.4])]
H2O_ATOMS = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
    (1, [0.0, -0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
]
FULL = dict(
    tcut_pno=0.0, tcut_mkn=0.0, coupling_radius=0.0, tcut_pairs=0.0
)  # exact reference (no PNO truncation, no coupling/pair screening)


def _setup(atoms, basis_name):
    mol = Molecule([Atom(z, p) for z, p in atoms], charge=0, multiplicity=1)
    basis = BasisSet(mol, basis_name)
    opts = RHFOptions()
    opts.max_iter = 100
    rhf = run_rhf(mol, basis, opts)
    assert rhf.converged
    df = DensityFitting(basis, BasisSet(mol, AUX), aux_basis_name=AUX)
    C = np.asarray(rhf.mo_coeffs)
    F = np.asarray(rhf.fock)
    n_occ = mol.n_electrons() // 2
    f_mo = C.T @ F @ C
    Co, Cv = C[:, :n_occ], C[:, n_occ:]
    canon = run_cs_ccsd(
        f_mo, np.asarray(df.mo_transform(Co, Cv)), np.asarray(df.mo_transform(Cv, Cv)),
        np.asarray(df.mo_transform(Co, Co)), n_occ, e_hf=rhf.energy,
    )
    return mol, basis, rhf, df, canon.e_corr


class TestParityRatchet:
    """Full-domain local DLPNO-CCSD == canonical CCSD (the flipped ratchet)."""

    @pytest.mark.parametrize("atoms,basis", [(H2_ATOMS, "sto-3g"), (H2O_ATOMS, "sto-3g")])
    def test_full_domain_equals_canonical_ccsd(self, atoms, basis):
        mol, b, rhf, df, e_ref = _setup(atoms, basis)
        r = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(localise="boys", **FULL)
        )
        assert r.converged
        assert abs(r.e_corr - e_ref) < 1.0 * MICRO_HA

    def test_full_domain_canonical_occupieds(self):
        mol, b, rhf, df, e_ref = _setup(H2O_ATOMS, "sto-3g")
        r = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(localise="none", **FULL)
        )
        assert r.converged
        assert abs(r.e_corr - e_ref) < 1.0 * MICRO_HA

    def test_h2_equals_fci(self):
        """CCSD ≡ FCI for two electrons — the FCI anchor reaches the solver."""
        mol, b, rhf, df, e_ref = _setup(H2_ATOMS, "sto-3g")
        r = run_local_dlpno_ccsd(mol, b, rhf, df, LocalCCSDOptions(localise="boys", **FULL))
        # e_ref is canonical CCSD = FCI for H2; solver must hit it.
        assert abs(r.e_corr - e_ref) < 1.0 * MICRO_HA


@pytest.mark.slow
class TestRecoveryDZ:
    def test_full_domain_exact_dz(self):
        mol, b, rhf, df, e_ref = _setup(H2O_ATOMS, "def2-svp")
        r = run_local_dlpno_ccsd(mol, b, rhf, df, LocalCCSDOptions(localise="boys", **FULL))
        assert r.converged
        assert abs(r.e_corr - e_ref) < 1.0 * MICRO_HA

    def test_truncated_recovers_999(self):
        mol, b, rhf, df, e_ref = _setup(H2O_ATOMS, "def2-svp")
        r = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(localise="boys", tcut_pno=1e-7, tcut_mkn=0.0)
        )
        assert r.converged
        recovery = r.e_corr / e_ref
        assert recovery > 0.998, f"recovery {recovery:.5%}"
        assert recovery < 1.003
        n_vir = b.nbasis - mol.n_electrons() // 2
        avg = sum(r.pno_per_pair.values()) / len(r.pno_per_pair)
        assert avg < 0.9 * n_vir  # real PNO compression


class TestPnoTailDiagnostic:
    """G-CORR-005: the CCSD tail residual is diagnostic, not a correction."""

    def test_tail_estimate_not_added_to_energy(self):
        mol, b, rhf, df, _ = _setup(H2O_ATOMS, "sto-3g")
        dom = dict(localise="boys", tcut_mkn=0.0, tcut_pairs=0.0, coupling_radius=0.0)
        full = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(tcut_pno=0.0, estimate_pno_tail=True, **dom)
        )
        raw = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(tcut_pno=1e-4, **dom)
        )
        diag = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(tcut_pno=1e-4, estimate_pno_tail=True, **dom)
        )

        assert full.e_pno_tail_estimate == pytest.approx(0.0, abs=1e-14)
        assert diag.e_corr == pytest.approx(raw.e_corr, abs=1e-14)
        assert diag.e_pno_tail_estimate < 0.0

        # Direction of the truncation error, and what the tail estimate does
        # to it, both depend on the pair-density convention (#65):
        #
        #   legacy  raw_err = -4.0803e-04, tail = -7.349e-04 -> compounds it
        #   mp2     raw_err = +3.1901e-04, tail = -2.852e-04 -> cancels part
        #
        # `mp2` retains more PNOs at the same `tcut_pno`, so this truncated
        # space is larger and stops over-correlating; the negative tail
        # estimate then partially cancels a positive error instead of
        # deepening a negative one. Neither is added to the energy, which is
        # what the assertions above actually pin.
        raw_err = raw.e_corr - full.e_corr
        assert raw_err > 0.0  # the retained space under-correlates under mp2
        assert abs(raw_err + diag.e_pno_tail_estimate) < abs(raw_err)


class TestSparseCoupling:
    """Per-pair local occupied set: exact at large radius, prunes far pairs.

    The coupling_radius knob restricts the na² occupied coupling sums to a
    bounded local set — the linear-scaling lever. At a radius wider than the
    molecule every pair still couples to every occupied (bit-for-bit identical
    to full coupling); on spatially separated fragments a finite radius drops
    only genuinely-decoupled cross terms, so the energy is unchanged.
    """

    def test_large_radius_equals_full_coupling(self):
        mol, b, rhf, df, _ = _setup(H2O_ATOMS, "sto-3g")
        dom = dict(localise="boys", tcut_pno=0.0, tcut_mkn=0.0, tcut_pairs=0.0)
        full = run_local_dlpno_ccsd(mol, b, rhf, df, LocalCCSDOptions(coupling_radius=0.0, **dom))
        wide = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(coupling_radius=1e3, **dom)
        )
        assert wide.avg_coupled_occ == pytest.approx(full.avg_coupled_occ)
        assert abs(wide.e_corr - full.e_corr) < 1e-12  # identical local sets

    def test_distant_fragments_decouple_losslessly(self):
        # Three H2 along z at 0 / 6 / 12 bohr. An 8-bohr coupling radius keeps
        # nearest-neighbour coupling (6 bohr apart) but drops the next-nearest
        # (12 bohr) from the terminal pairs' occupied sets — shrinking the
        # average set below na while the dropped coupling is sub-µHa.
        atoms = []
        for k in range(3):
            atoms += [(1, [0.0, 0.0, k * 6.0]), (1, [0.0, 0.0, k * 6.0 + 1.4])]
        mol, b, rhf, df, _ = _setup(atoms, "sto-3g")
        dom = dict(localise="boys", tcut_pno=0.0, tcut_mkn=0.0, tcut_pairs=0.0)
        full = run_local_dlpno_ccsd(mol, b, rhf, df, LocalCCSDOptions(coupling_radius=0.0, **dom))
        sparse = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(coupling_radius=8.0, **dom)
        )
        n_occ = mol.n_electrons() // 2
        assert sparse.avg_coupled_occ < n_occ          # next-nearest pruned
        assert abs(sparse.e_corr - full.e_corr) < 1.0 * MICRO_HA  # lossless to 1 µHa


class TestPairScreening:
    """tcut_pairs: weak pairs treated at full-virtual MP2, not truncated CCSD.

    A pair whose full-virtual MP2 estimate is below tcut_pairs is dropped from
    the CCSD iteration and gets its exact MP2 pair energy instead. For a
    genuinely weak pair the CCSD correction is negligible and full-virtual MP2
    carries no PNO-truncation error, so screening is near-lossless while
    removing the pair's CCSD cost — the linear-scaling lever (ORCA's TCutPairs,
    default 1e-4). Calibrated default-on; the exact reference pins tcut_pairs=0.
    """

    def test_separated_fragments_screen_losslessly(self):
        # Two H2 molecules ~8 bohr apart (one occupied each). The single
        # inter-fragment pair is weak: tcut_pairs screens it to MP2 while the
        # two intra-H2 diagonal pairs stay at CCSD, and the energy is unchanged
        # to ~µHa (the cross-pair CCSD-vs-MP2 difference at this separation).
        atoms = [
            (1, [0.0, 0.0, 0.0]),
            (1, [0.0, 0.0, 1.4]),
            (1, [0.0, 0.0, 9.4]),
            (1, [0.0, 0.0, 10.8]),
        ]
        mol, b, rhf, df, _ = _setup(atoms, "sto-3g")
        dom = dict(localise="boys", tcut_pno=0.0, tcut_mkn=0.0, coupling_radius=0.0)
        unscreened = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(tcut_pairs=0.0, **dom)
        )
        screened = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(tcut_pairs=1e-4, **dom)
        )
        assert unscreened.n_screened == 0
        assert screened.n_screened == 1  # the inter-fragment pair only
        assert screened.n_pairs == unscreened.n_pairs - 1  # one fewer CCSD pair
        assert abs(screened.e_corr - unscreened.e_corr) < 5.0 * MICRO_HA

    def test_monotonic_screening_count(self):
        # n_screened is non-decreasing in tcut_pairs (the threshold is a cut on
        # the same pair-energy spectrum). A 3-H2 chain has several weak
        # inter-fragment pairs that screen progressively.
        atoms = []
        for k in range(3):
            atoms += [(1, [0.0, 0.0, k * 8.0]), (1, [0.0, 0.0, k * 8.0 + 1.4])]
        mol, b, rhf, df, _ = _setup(atoms, "sto-3g")
        dom = dict(localise="boys", tcut_pno=0.0, tcut_mkn=0.0, coupling_radius=0.0)
        counts = [
            run_local_dlpno_ccsd(
                mol, b, rhf, df, LocalCCSDOptions(tcut_pairs=t, **dom)
            ).n_screened
            for t in (0.0, 1e-6, 1e-4, 1e-2)
        ]
        assert counts[0] == 0
        assert counts == sorted(counts)  # non-decreasing
        assert counts[-1] > counts[0]  # screening is live

    def test_all_screened_returns_mp2_correlation_and_total(self):
        """The valid no-iterated-pair boundary must not return zero energy."""
        mol, b, rhf, df, _ = _setup(H2O_ATOMS, "sto-3g")
        result = run_local_dlpno_ccsd(
            mol,
            b,
            rhf,
            df,
            LocalCCSDOptions(
                localise="boys",
                tcut_pno=0.0,
                tcut_mkn=0.0,
                tcut_pairs=1.0,
                coupling_radius=0.0,
                compute_triples=True,
            ),
        )

        assert result.converged
        assert result.n_pairs == 0
        assert result.n_screened > 0
        assert not result.triples_executed
        assert result.e_t == 0.0
        assert result.e_corr < 0.0
        assert result.e_total == pytest.approx(
            result.e_hf + result.e_corr,
            abs=1e-12,
        )


@pytest.mark.slow
class TestPairScreeningAccuracy:
    """The calibrated default (1e-4) is accuracy-neutral on a real molecule."""

    def test_n2_default_accuracy_neutral(self):
        # N2/def2-SVP has 7 occupieds: the default tcut_pairs=1e-4 screens a
        # few weak core/cross pairs (10.7 % in calibration) while the total
        # correlation+(T) is unchanged to well within chemical accuracy vs the
        # no-screening reference (calibration: 0.001 kcal/mol, ~1.6 µHa).
        atoms = [(7, [0, 0, 0]), (7, [0, 0, 1.098 * ANGSTROM_TO_BOHR])]
        mol, b, rhf, df, _ = _setup(atoms, "def2-svp")
        base = dict(localise="boys", tcut_pno=1e-7, compute_triples=True, max_nbf=300)
        ref = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(tcut_pairs=0.0, **base)
        )
        deflt = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(tcut_pairs=1e-4, **base)
        )
        assert ref.n_screened == 0
        assert 0 < deflt.n_screened <= 6  # weak tail only (≤ ~20 % of 28 pairs)
        # Same three pairs are screened under either convention; what moves
        # is how much correlation the screened-away pairs were carrying.
        # Measured (#65): legacy 1.1464e-06 Ha (0.0007 kcal/mol), mp2
        # 6.1755e-05 Ha (0.0388 kcal/mol). `mp2`'s wider PNO spaces recover
        # more from every retained pair, so the weak tail dropped at
        # MP2 level is a larger share of the total. Still two orders inside
        # chemical accuracy, which is the property this pins.
        d_corr_t = abs((deflt.e_corr + deflt.e_t) - (ref.e_corr + ref.e_t))
        assert d_corr_t < 1e-4  # < 0.06 kcal/mol — accuracy-neutral


class TestStructure:
    def test_diagonal_pairs_and_singles(self):
        mol, b, rhf, df, _ = _setup(H2O_ATOMS, "sto-3g")
        r = run_local_dlpno_ccsd(mol, b, rhf, df, LocalCCSDOptions(localise="boys", **FULL))
        n_occ = mol.n_electrons() // 2
        assert r.n_pairs == n_occ * (n_occ + 1) // 2  # includes diagonal
        assert r.e_total == pytest.approx(r.e_hf + r.e_corr, abs=1e-12)


class TestCppKernel:
    """The legacy pair-domain C++ residual matches its NumPy path exactly."""

    def test_target_kernel_matches_full_residual(self):
        from vibeqc._vibeqc_core import (
            dlpno_pair_residual,
            dlpno_target_pair_residual,
        )

        rng = np.random.default_rng(87)
        no, nv, naux = 3, 4, 7
        t1 = np.ascontiguousarray(0.02 * rng.standard_normal((no, nv)))
        t2 = np.ascontiguousarray(
            0.02 * rng.standard_normal((no * no, nv * nv))
        )
        b_ov = np.ascontiguousarray(rng.standard_normal((naux, no * nv)))
        b_oo = np.ascontiguousarray(rng.standard_normal((naux, no * no)))
        b_vv = np.ascontiguousarray(rng.standard_normal((naux, nv * nv)))
        f_oo = np.ascontiguousarray(rng.standard_normal((no, no)))
        f_vv = np.ascontiguousarray(rng.standard_normal((nv, nv)))
        f_ov = np.ascontiguousarray(rng.standard_normal((no, nv)))

        r1, r2 = dlpno_pair_residual(
            t1, t2, b_ov, b_oo, b_vv, f_oo, f_vv, f_ov
        )
        for i, j in ((0, 0), (0, 2), (2, 1)):
            r1_target, r2_target = dlpno_target_pair_residual(
                t1, t2, b_ov, b_oo, b_vv, f_oo, f_vv, f_ov, i, j
            )
            assert np.allclose(np.asarray(r1_target)[0], np.asarray(r1)[i])
            assert np.allclose(
                np.asarray(r2_target),
                np.asarray(r2).reshape(no, no, nv, nv)[i, j],
            )

    def test_cpp_matches_numpy(self):
        from vibeqc.dlpno.ccsd_local_solver import _HAVE_CPP_RESIDUAL

        if not _HAVE_CPP_RESIDUAL:
            pytest.skip("core built without dlpno_pair_residual")
        mol, b, rhf, df, _ = _setup(H2O_ATOMS, "sto-3g")
        opts = dict(localise="boys", tcut_pno=1e-7)
        cpp = run_local_dlpno_ccsd(mol, b, rhf, df, LocalCCSDOptions(use_cpp_kernel=True, **opts))
        npy = run_local_dlpno_ccsd(mol, b, rhf, df, LocalCCSDOptions(use_cpp_kernel=False, **opts))
        assert cpp.converged and npy.converged
        assert abs(cpp.e_corr - npy.e_corr) < 1e-11  # bit-for-bit kernel parity


class TestTNODomainDiagnostics:
    """The public closed-shell `(T1)` route reports how far it truncated.

    ``LocalCCSDOptions.tcut_tno`` defaults to 0.0, so a default
    ``dlpno-ccsd(t)`` job truncates nothing.  A user who opts into a positive
    value used to get no feedback at all: the result carried the energy and
    nothing about the TNO domains behind it.  A spatial triple excitation
    promotes three electrons into three *distinct* virtuals, so a domain
    holding fewer than three contributes exactly zero rather than
    approximately, and the energy alone cannot distinguish that from a small
    correction.
    """

    @staticmethod
    def _run(tcut_tno):
        mol, basis, rhf, df, _ = _setup(H2O_ATOMS, "def2-svp")
        return run_local_dlpno_ccsd(
            mol,
            basis,
            rhf,
            df,
            LocalCCSDOptions(
                tcut_pno=0.0,
                tcut_pairs=0.0,
                compute_triples=True,
                triples_mode="t1",
                tcut_tno=tcut_tno,
            ),
        )

    def test_full_domain_reports_the_whole_virtual_space(self):
        result = self._run(0.0)
        assert result.converged
        n_vir = 24 - 5  # def2-SVP H2O: 24 basis functions, 5 occupied
        assert result.avg_tno == pytest.approx(float(n_vir))
        assert result.n_degenerate_tno_triples == 0
        assert min(result.tno_per_triple.values()) == n_vir

    def test_collapse_onset_tracks_the_error_break(self):
        """The first degenerate triple marks where the error curve breaks.

        Measured on H2O/def2-SVP (full domain
        ``e_t = -0.003009059087680666`` Ha, 19.0 TNOs per triple): the error
        is 0.086 uHa at ``1e-6`` with every domain still at least four
        virtuals, then jumps 400-fold to 33 uHa at ``1e-5``, which is exactly
        where the first triple collapses to a single virtual.  The
        diagnostic, not the energy, is what identifies that boundary.
        """
        full = self._run(0.0)
        fine = self._run(1e-6)
        coarse = self._run(1e-5)
        assert full.converged and fine.converged and coarse.converged

        # 1e-6: real truncation, no collapse, sub-uHa error.
        assert fine.avg_tno < full.avg_tno
        assert fine.n_degenerate_tno_triples == 0
        assert min(fine.tno_per_triple.values()) >= 3
        assert abs(fine.e_t - full.e_t) < 1.0 * MICRO_HA

        # 1e-5: a triple has collapsed below three virtuals, and the error
        # leaves the sub-uHa regime with it.
        assert coarse.n_degenerate_tno_triples >= 1
        assert min(coarse.tno_per_triple.values()) < 3
        assert abs(coarse.e_t - full.e_t) > 10.0 * MICRO_HA

    def test_exact_mode_builds_no_tno_domain(self):
        """``triples_mode="exact"`` expands to the full space, not a domain."""
        mol, basis, rhf, df, _ = _setup(H2O_ATOMS, "def2-svp")
        result = run_local_dlpno_ccsd(
            mol,
            basis,
            rhf,
            df,
            LocalCCSDOptions(
                tcut_pno=0.0,
                tcut_pairs=0.0,
                compute_triples=True,
                triples_mode="exact",
            ),
        )
        assert result.converged
        assert result.tno_per_triple == {}
        assert result.n_degenerate_tno_triples == 0

    def test_coupling_radius_screening_is_counted(self):
        """`coupling_radius` drops occupied triple keys; the count is reported.

        Screening is the occupied-side analogue of TNO truncation, and it can
        be far more aggressive than it looks. On an H10 chain at 1.8 bohr
        spacing (16.2 bohr long, strongly coupled occupieds) a
        ``coupling_radius`` of 5 bohr drops 16 of the 25 distinct triple keys
        and takes 90% of the triples energy with them. The energy alone does
        not say so; the counts do.
        """
        mol = Molecule(
            [Atom(1, [0.0, 0.0, i * 1.8]) for i in range(10)],
            charge=0,
            multiplicity=1,
        )
        basis = BasisSet(mol, "sto-3g")
        opts = RHFOptions()
        opts.max_iter = 400
        rhf = run_rhf(mol, basis, opts)
        assert rhf.converged
        df = DensityFitting(
            basis, BasisSet(mol, AUX), aux_basis_name=AUX
        )
        common = dict(
            tcut_pno=0.0,
            tcut_pairs=0.0,
            tcut_tno=0.0,
            compute_triples=True,
            triples_mode="t1",
            localise="boys",
        )
        full = run_local_dlpno_ccsd(
            mol, basis, rhf, df, LocalCCSDOptions(coupling_radius=0.0, **common)
        )
        screened = run_local_dlpno_ccsd(
            mol, basis, rhf, df, LocalCCSDOptions(coupling_radius=5.0, **common)
        )
        assert full.converged and screened.converged

        assert full.n_triple_keys == 25
        assert full.n_triple_keys_screened == 0
        assert screened.n_triple_keys == 9
        assert screened.n_triple_keys_screened == 16
        # Kept + screened is the same total either way.
        assert (
            screened.n_triple_keys + screened.n_triple_keys_screened
            == full.n_triple_keys
        )
        # And the loss is severe enough that reporting it matters.
        assert abs(screened.e_t) < 0.2 * abs(full.e_t)


class TestTruncatedSubspaceOracle:
    """The local engine must match its own oracle in the truncated regime.

    `run_dlpno_ccsd_pilot` targets the same DLPNO ansatz, per-pair PNO spaces,
    but evaluates the exact residual in the full space and projects back,
    which is the stationarity condition of subspace-constrained CCSD. It is
    therefore the oracle under truncation, not only at full domain, and its
    own docstring says the reduced-scaling engine is what it exists to
    validate.

    The historical pair-domain contraction disagrees under truncation and is
    retained below as a strict negative control. The current default uses the
    atom-based extended domain and has its own operative preset ratchet.
    """

    @staticmethod
    def _both(tcut_pno, residual_domain="pair"):
        mol, basis, rhf, df, _ = _setup(H2O_ATOMS, "def2-svp")
        local = run_local_dlpno_ccsd(
            mol,
            basis,
            rhf,
            df,
            LocalCCSDOptions(
                tcut_pno=tcut_pno,
                tcut_pairs=0.0,
                compute_triples=False,
                residual_domain=residual_domain,
            ),
        )
        pilot = run_dlpno_ccsd_pilot(
            mol,
            basis,
            rhf,
            df,
            DLPNOCCSDPilotOptions(
                tcut_pno=tcut_pno, tcut_mkn=0.0, localise="boys"
            ),
        )
        assert local.converged
        return local.e_corr, pilot.e_corr

    @pytest.mark.slow
    def test_current_default_is_extended_and_tightens_toward_oracle(self):
        """#98: the public paper-domain presets improve in tightening order."""
        mol, basis, rhf, df, _ = _setup(H2O_ATOMS, "def2-svp")

        default = _LocalCCSDOptions()
        normal = options_from_dlpno_thresholds(_LocalCCSDOptions, "normal")
        assert default.residual_domain == "extended"
        for field in ("tcut_pairs", "tcut_pno", "tcut_mkn"):
            assert getattr(default, field) == getattr(normal, field)

        full = run_local_dlpno_ccsd(
            mol,
            basis,
            rhf,
            df,
            _LocalCCSDOptions(
                n_frozen=None,
                tcut_pno=0.0,
                tcut_mkn=0.0,
                tcut_pairs=0.0,
                coupling_radius=0.0,
                residual_domain="full",
                compute_triples=False,
                localise="boys",
            ),
        )
        assert full.converged and full.n_screened == 0
        errors = []
        avg_pnos = []
        locals_by_preset = {}
        for preset in ("loose", "normal", "tight"):
            local = run_local_dlpno_ccsd(
                mol,
                basis,
                rhf,
                df,
                options_from_dlpno_thresholds(
                    _LocalCCSDOptions,
                    preset,
                    compute_triples=False,
                ),
            )
            assert local.converged
            assert local.n_screened == 0
            locals_by_preset[preset] = local
            errors.append(abs(local.e_corr - full.e_corr))
            avg_pnos.append(
                sum(local.pno_per_pair.values()) / len(local.pno_per_pair)
            )

        normal_pilot = run_dlpno_ccsd_pilot(
            mol,
            basis,
            rhf,
            df,
            options_from_dlpno_thresholds(
                _DLPNOCCSDPilotOptions,
                "normal",
            ),
        )
        assert normal_pilot.converged
        assert locals_by_preset["normal"].e_corr == pytest.approx(
            normal_pilot.e_corr, abs=50e-6
        )

        slack = 5e-6
        assert errors[1] <= errors[0] + slack
        assert errors[2] <= errors[1] + slack
        assert errors[2] < errors[0]
        assert avg_pnos[0] <= avg_pnos[1] <= avg_pnos[2]
        assert avg_pnos[2] > avg_pnos[0]

    @pytest.mark.slow
    def test_local_matches_pilot_at_full_domain(self):
        """The agreement that does hold, and that the suite already relied on."""
        local, pilot = self._both(0.0)
        assert local == pytest.approx(pilot, abs=1e-9)

    @pytest.mark.slow
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Legacy residual_domain='pair' projects before contraction and "
            "therefore over-correlates under PNO truncation. H2O/def2-SVP "
            "deviates by -2366 uHa at 1e-6 and -227 uHa at 1e-7. Strict, so "
            "the retained negative control fails loudly if that legacy "
            "algorithm changes."
        ),
    )
    def test_local_matches_pilot_under_truncation(self):
        """The agreement that should hold and does not."""
        local, pilot = self._both(1e-6)
        # 50 uHa is generous: the measured gap is 2366 uHa, and a genuine fix
        # should land far inside this.
        assert local == pytest.approx(pilot, abs=50e-6)

    @pytest.mark.slow
    def test_extended_residual_domain_restores_agreement(self):
        """Contracting before projecting removes the deviation.

        The legacy `"pair"` path projects the coupled pairs' amplitudes into
        pair `ij`'s PNO space and contracts there, which discards the parts of
        those amplitudes that lie outside `ij`'s space but still feed back into
        it through the integrals. The default `"extended"` path contracts in
        the atom-based extended PAO domain and projects the residual back.

        Riplinger and Neese, J. Chem. Phys. 138, 034106 (2013), Sec. II B 1
        require an *extended* domain for precisely this reason: it is
        "necessary in order to create a sufficiently large buffer for the
        accurate calculation of the pair-pair interaction terms".

        Measured on H2O/def2-SVP, deviation from the pilot in uHa:

            tcut_pno     "pair"  "extended"
            1e-5       -10060.8        +1.5
            1e-6        -2365.9        -0.1
            1e-7         -226.6        +0.0
        """
        for tcut in (1e-6, 1e-7):
            local, pilot = self._both(tcut, residual_domain="extended")
            assert local == pytest.approx(pilot, abs=5e-6), (
                f"extended residual domain still deviates at tcut_pno={tcut}"
            )

    def test_residual_domain_is_validated(self):
        mol, basis, rhf, df, _ = _setup(H2_ATOMS, "sto-3g")
        with pytest.raises(ValueError, match="unknown residual_domain"):
            run_local_dlpno_ccsd(
                mol,
                basis,
                rhf,
                df,
                LocalCCSDOptions(residual_domain="union-of-everything"),
            )

    @pytest.mark.slow
    def test_extended_domain_is_exact_at_full_pno_space_too(self):
        """The new path must not perturb the case that already worked."""
        local, pilot = self._both(0.0, residual_domain="full")
        assert local == pytest.approx(pilot, abs=1e-9)

    @pytest.mark.slow
    def test_extended_domain_spans_the_full_space_at_default_atom_domains(self):
        """The paper's extended domain is defined over ATOMS, not PNO spaces.

        Riplinger and Neese build it from "the union of the domains of orbitals
        i and j and furthermore ... all atoms of the domains of all orbitals k",
        spanned by the PAOs on those atoms. That construction does not depend
        on ``tcut_pno``, which is the whole point: a buffer assembled out of the
        truncated PNO spaces would be made of the very deficiency it exists to
        absorb.

        A consequence worth pinning, because it is the cheapest check that the
        construction is the right one: the explicit ``_both`` helper sets
        ``tcut_mkn=0``, so every orbital domain is *all* atoms. The extended
        domain is then the entire virtual space and must coincide with
        ``residual_domain="full"`` exactly, not merely closely.

        An earlier revision of this test asserted the opposite, that extended
        could not reach the pilot. That was pinning a weaker construction
        (the union of coupled pairs' PNO spaces) which is not what the paper
        prescribes.
        """
        pair, pilot = self._both(1e-6)
        ext, _ = self._both(1e-6, residual_domain="extended")
        full, _ = self._both(1e-6, residual_domain="full")

        # At tcut_mkn = 0 the atom union is everything, so the two coincide.
        assert ext == pytest.approx(full, abs=1e-12)
        # And both are far better than contracting inside the pair domain.
        assert abs(ext - pilot) < abs(pair - pilot) / 100.0
        assert abs(ext - pilot) < 5e-6

    @pytest.mark.slow
    def test_residual_domain_dimensions_are_ordered(self):
        """pair < extended <= full, reported so a user can see the cost."""
        mol, basis, rhf, df, _ = _setup(H2O_ATOMS, "def2-svp")

        def _domain(residual_domain):
            return run_local_dlpno_ccsd(
                mol,
                basis,
                rhf,
                df,
                LocalCCSDOptions(
                    tcut_pno=1e-6,
                    tcut_pairs=0.0,
                    compute_triples=False,
                    residual_domain=residual_domain,
                ),
            ).avg_residual_domain

        pair, ext, full = (
            _domain("pair"), _domain("extended"), _domain("full")
        )
        assert pair < ext <= full
        assert full == pytest.approx(19.0)  # def2-SVP H2O: 24 - 5 virtuals


# S22-01 ammonia dimer (Jurecka et al. 2006 geometry, bohr): the #98 / #689
# witness. cc-pVDZ / cc-pVDZ-RI, published NormalPNO frozen core.
S22_01_A = [
    (7, [-2.98334443, -0.08808202, 0.0]),
    (1, [-4.07920220, 0.25775107, -1.52985602]),
    (1, [-4.07920220, 0.25775107, 1.52985602]),
    (1, [-1.60526743, 1.24380442, 0.0]),
]
S22_01_B = [
    (7, [2.98334443, 0.08808202, 0.0]),
    (1, [4.07920220, -0.25775107, -1.52985602]),
    (1, [1.60526743, -1.24380442, 0.0]),
    (1, [4.07920220, -0.25775107, 1.52985602]),
]
_PUBLISHED_PRESETS = ("loose", "normal", "tight")


def _setup_ccpvdz(atoms):
    mol = Molecule([Atom(z, p) for z, p in atoms], charge=0, multiplicity=1)
    basis = BasisSet(mol, "cc-pvdz")
    opts = RHFOptions()
    opts.max_iter = 200
    opts.conv_tol_energy = 1e-10
    opts.conv_tol_grad = 1e-7
    rhf = run_rhf(mol, basis, opts)
    assert rhf.converged
    df = DensityFitting(
        basis, BasisSet(mol, "cc-pvdz-ri"), aux_basis_name="cc-pvdz-ri"
    )
    return mol, basis, rhf, df


def _preset_ladder(mol, basis, rhf, df, pno_norm="mp2"):
    """Full-domain reference plus the three published presets (default
    residual domain, default frozen core). Returns (full, {preset: result}).

    `pno_norm` is explicit because the retained #98 reference values were
    measured under ``"legacy"``, which was the default before the #65 ruling.
    A `tcut_pno` preset only means what the density it was calibrated against
    means, so a ladder is only comparable within one convention.
    """
    full = run_local_dlpno_ccsd(
        mol,
        basis,
        rhf,
        df,
        _LocalCCSDOptions(
            tcut_pno=0.0,
            tcut_mkn=0.0,
            tcut_pairs=0.0,
            coupling_radius=0.0,
            residual_domain="full",
            pno_norm=pno_norm,
            compute_triples=False,
        ),
    )
    assert full.converged
    ladder = {}
    for preset in _PUBLISHED_PRESETS:
        r = run_local_dlpno_ccsd(
            mol,
            basis,
            rhf,
            df,
            options_from_dlpno_thresholds(
                _LocalCCSDOptions, preset, compute_triples=False,
                pno_norm=pno_norm,
            ),
        )
        assert r.converged
        ladder[preset] = r
    return full, ladder


class TestExtendedDomainGrouping:
    """#689: the extended-domain contraction is shared, compiled, and exact.

    The extended residual domain of a pair depends only on its occupied
    coupling set and the union of the coupled orbitals' atom domains. Pairs
    sharing both see identical coupled amplitudes in an identical basis, so
    the solver contracts the residual once per such group and lets every
    member read its own block. On a compact molecule every pair is in one
    group, so the per-iteration cost is one canonical-shaped residual instead
    of one per pair: on S22-01 that took the ammonia dimer from 406 s to
    5.5 s and the monomer from 5.2 s to 0.4 s at NormalPNO with the energies
    unchanged to the last digit.
    """

    def test_compact_molecule_contracts_once_per_iteration(self, monkeypatch):
        from vibeqc.dlpno import ccsd_local_solver as solver_mod

        if not solver_mod._HAVE_CPP_RESIDUAL:
            pytest.skip("core built without dlpno_pair_residual")
        mol, basis, rhf, df, _ = _setup(H2O_ATOMS, "def2-svp")
        calls = []
        kernel = solver_mod._cpp_pair_residual
        monkeypatch.setattr(
            solver_mod,
            "_cpp_pair_residual",
            lambda *a, **k: (calls.append(a[0].shape), kernel(*a, **k))[1],
        )
        r = run_local_dlpno_ccsd(
            mol,
            basis,
            rhf,
            df,
            options_from_dlpno_thresholds(
                _LocalCCSDOptions, "normal", compute_triples=False
            ),
        )
        assert r.converged
        assert r.n_pairs > 1
        # One group, one contraction per iteration, in the whole virtual
        # space (def2-SVP H2O: 24 - 5 = 19 virtuals, every atom domain being
        # the whole molecule).
        assert r.n_residual_domains == 1
        assert r.avg_residual_domain == pytest.approx(19.0)
        assert len(calls) == r.n_iter
        assert calls[0] == (r.n_pairs and 4, 19)  # (n_active_occ, n_ext)

    def test_extended_cpp_kernel_matches_numpy_fallback(self):
        from vibeqc.dlpno.ccsd_local_solver import _HAVE_CPP_RESIDUAL

        if not _HAVE_CPP_RESIDUAL:
            pytest.skip("core built without dlpno_pair_residual")
        mol, basis, rhf, df, _ = _setup(H2O_ATOMS, "def2-svp")
        opts = dict(tcut_pno=1e-6, tcut_pairs=0.0, compute_triples=False)
        cpp = run_local_dlpno_ccsd(
            mol, basis, rhf, df, _LocalCCSDOptions(use_cpp_kernel=True, **opts)
        )
        npy = run_local_dlpno_ccsd(
            mol, basis, rhf, df, _LocalCCSDOptions(use_cpp_kernel=False, **opts)
        )
        assert cpp.converged and npy.converged
        assert cpp.n_residual_domains == npy.n_residual_domains == 1
        assert abs(cpp.e_corr - npy.e_corr) < 1e-10  # kernel parity

    def test_extended_equals_full_when_domains_are_proper_subsets(self):
        """The extended contraction is exact against ``full`` at the same
        coupling set, and that is what lets it be cheaper.

        Two waters 10 A apart: Boys orbitals localise on one water each, the
        12-bohr coupling radius keeps each pair's occupied set on its own
        water, and ``tcut_mkn`` keeps each orbital domain on its own water,
        so every pair's extended domain is one water's PAOs (19 of the 38
        virtuals) and there are two groups. Every coupled pair's PNO space is
        spanned by those PAOs, so contracting there loses nothing relative to
        the whole virtual space.
        """
        shift = 10.0 * ANGSTROM_TO_BOHR
        atoms = list(H2O_ATOMS) + [
            (z, [x + shift, y, zc]) for z, (x, y, zc) in H2O_ATOMS
        ]
        mol, basis, rhf, df, _ = _setup(atoms, "def2-svp")
        common = dict(
            tcut_pno=1e-6,
            tcut_mkn=1e-3,
            tcut_pairs=1e-4,
            coupling_radius=12.0,
            compute_triples=False,
        )
        ext = run_local_dlpno_ccsd(
            mol, basis, rhf, df, _LocalCCSDOptions(residual_domain="extended", **common)
        )
        full = run_local_dlpno_ccsd(
            mol, basis, rhf, df, _LocalCCSDOptions(residual_domain="full", **common)
        )
        assert ext.converged and full.converged
        assert ext.n_screened == full.n_screened > 0  # inter-water pairs
        assert ext.n_pairs == full.n_pairs
        n_vir = basis.nbasis - mol.n_electrons() // 2
        assert full.avg_residual_domain == pytest.approx(float(n_vir))
        assert ext.avg_residual_domain == pytest.approx(19.0)
        assert ext.n_residual_domains == 2
        # Four active occupieds per water under the published frozen core.
        assert ext.avg_coupled_occ == pytest.approx(4.0)
        assert abs(ext.e_corr - full.e_corr) < 1e-8

    def test_s22_01_monomer_ladder_tightens_monotonically(self):
        """The #98 witness at the current defaults, kept fast by #689.

        Reference values: #98 note 15488 (fixer and two independent
        verifiers agree to all printed digits), NH3 monomer A of S22-01,
        cc-pVDZ / cc-pVDZ-RI, published frozen core, Loose/Normal/Tight
        against the full-domain oracle: |err| 7.3716e-4, 6.0115e-4,
        2.5580e-5 Ha.

        Pinned to ``pno_norm="legacy"`` because that is the convention those
        verified numbers were measured under; the default moved to ``"mp2"``
        with the #65 ruling. They are kept rather than re-measured: a
        re-measured ladder would be this lane's own unverified numbers
        replacing two independent verifications, which the correlation
        bugsweep handover explicitly warns against. The default's ladder is
        covered by its monotonicity below.
        """
        mol, basis, rhf, df = _setup_ccpvdz(S22_01_A)
        full, ladder = _preset_ladder(mol, basis, rhf, df, pno_norm="legacy")
        errors = [
            abs(ladder[p].e_corr - full.e_corr) for p in _PUBLISHED_PRESETS
        ]
        assert errors[0] > errors[1] > errors[2]
        assert errors == pytest.approx([7.3716e-4, 6.0115e-4, 2.5580e-5], abs=5e-7)
        for r in ladder.values():
            # 4 atoms: every domain is the whole molecule (24 virtuals) and
            # the ten surviving pairs share one contraction.
            assert r.n_residual_domains == 1
            assert r.avg_residual_domain == pytest.approx(24.0)
        assert full.e_corr == pytest.approx(-0.2027554234, abs=2e-9)

    def test_default_norm_ladder_also_tightens_monotonically(self):
        """The shipped default's ladder, as a property rather than a pin.

        The two tests around this one keep the independently verified #98
        numbers, which belong to ``"legacy"``. The default is now ``"mp2"``
        (#65), and its absolute ladder has been measured only here -- one
        lane, no independent verification -- so pinning those digits would
        dress up an unverified calculation as a reference. What is actually
        being claimed is the property the #98 witness established: truncation
        error falls monotonically as the preset tightens. That holds under
        either convention and is what a preset ladder has to do to mean
        anything.
        """
        mol, basis, rhf, df = _setup_ccpvdz(S22_01_A)
        full, ladder = _preset_ladder(mol, basis, rhf, df)  # default: mp2
        assert all(r.pno_norm == "mp2" for r in ladder.values())
        errors = [
            abs(ladder[p].e_corr - full.e_corr) for p in _PUBLISHED_PRESETS
        ]
        assert errors[0] > errors[1] > errors[2], errors
        # The oracle is convention-independent: tcut_pno=0 retains every PNO,
        # so no density can change which space is kept.
        assert full.e_corr == pytest.approx(-0.2027554234, abs=2e-9)

    @pytest.mark.slow
    def test_s22_01_dimer_interaction_error_tightens_monotonically(self):
        """The S22-statistic property from #98 note 15577, now affordable.

        Interaction correlation energy (dimer minus monomers) truncation
        error against the full-domain oracle, Loose/Normal/Tight:
        |err| 0.342, 0.311, 0.143 mHa, converging with tightening; the legacy
        pair recipe stays near 0.7 mHa across the ladder.

        Pinned to ``pno_norm="legacy"`` for the same reason as the monomer
        ladder above: these are retained #98 values, measured under the
        pre-#65 default.
        """
        systems = {
            name: _setup_ccpvdz(atoms)
            for name, atoms in (
                ("dimer", S22_01_A + S22_01_B),
                ("monoA", S22_01_A),
                ("monoB", S22_01_B),
            )
        }
        runs = {
            name: _preset_ladder(*sysm, pno_norm="legacy")
            for name, sysm in systems.items()
        }
        full_int = (
            runs["dimer"][0].e_corr
            - runs["monoA"][0].e_corr
            - runs["monoB"][0].e_corr
        )
        errors = []
        for preset in _PUBLISHED_PRESETS:
            e_int = (
                runs["dimer"][1][preset].e_corr
                - runs["monoA"][1][preset].e_corr
                - runs["monoB"][1][preset].e_corr
            )
            errors.append(abs(e_int - full_int))
        assert errors[0] > errors[1] > errors[2]
        assert errors == pytest.approx([3.42e-4, 3.11e-4, 1.43e-4], abs=5e-6)
        assert full_int == pytest.approx(-2.05953e-3, abs=5e-8)
        dimer = runs["dimer"][1]["normal"]
        assert dimer.n_residual_domains == 1
        assert dimer.avg_residual_domain == pytest.approx(48.0)

    @pytest.mark.slow
    def test_extended_equals_full_with_nested_domains(self):
        """The proper-subset leg of the exactness claim.

        Three waters 6 A apart along x: each pair's own PAO domain is one
        water, its extended domain (union over the 12-bohr coupling set) is
        two or three waters, and the whole molecule is 57 virtuals. A pair
        domain is therefore a proper subset of its extended domain, which is
        itself a proper subset of the full space, and several groups form.
        Extended and full agree to machine precision here (2e-16 Ha
        measured); the claim in the module docstring is "to ``lindep``
        precision" because each domain's PAO metric is orthonormalised on its
        own.
        """
        shift = 6.0 * ANGSTROM_TO_BOHR
        atoms = []
        for k in range(3):
            atoms += [(z, [x + k * shift, y, zc]) for z, (x, y, zc) in H2O_ATOMS]
        mol, basis, rhf, df, _ = _setup(atoms, "def2-svp")
        common = dict(
            tcut_pno=1e-6,
            tcut_mkn=1e-3,
            tcut_pairs=1e-4,
            coupling_radius=12.0,
            compute_triples=False,
        )
        ext = run_local_dlpno_ccsd(
            mol, basis, rhf, df, _LocalCCSDOptions(residual_domain="extended", **common)
        )
        full = run_local_dlpno_ccsd(
            mol, basis, rhf, df, _LocalCCSDOptions(residual_domain="full", **common)
        )
        assert ext.converged and full.converged
        assert ext.n_pairs == full.n_pairs
        n_vir = basis.nbasis - mol.n_electrons() // 2
        assert full.avg_residual_domain == pytest.approx(float(n_vir))
        # Extended domains are proper subsets of the molecule, and the
        # coupling sets are proper subsets of the 12 active occupieds.
        assert 19.0 < ext.avg_residual_domain < float(n_vir)
        assert 4.0 < ext.avg_coupled_occ < 12.0
        assert ext.n_residual_domains > 1
        assert abs(ext.e_corr - full.e_corr) < 1e-9


class TestPNONormOnRealAmplitudes:
    """#701: what `tcut_pno` is compared against, measured on a molecule.

    `tcut_pno` cuts pair-density occupation numbers, so the published
    Loose/Normal/Tight values only mean what Liakos et al. calibrated them to
    mean under the density they were calibrated against: Riplinger and Neese's
    Eq. 23 (`pno_norm="mp2"`, ORCA's `PNONorm MP2Norm` default). vibe-qc's
    historical density omits the spin-adapted contravariant amplitude and is
    the `"legacy"` default here, so nothing moves until #701 is ruled on.
    """

    @staticmethod
    def _run(mol, basis, rhf, df, **kw):
        return run_local_dlpno_ccsd(
            mol,
            basis,
            rhf,
            df,
            options_from_dlpno_thresholds(
                _LocalCCSDOptions, "normal", compute_triples=False, **kw
            ),
        )

    def test_default_is_mp2_and_is_bit_identical_to_requesting_it(self):
        """The #65 (old #701) ruling: the shipped density is Riplinger and
        Neese's Eq. 23, so a preset name means what the literature means."""
        mol, basis, rhf, df, _ = _setup(H2O_ATOMS, "def2-svp")
        assert _LocalCCSDOptions().pno_norm == "mp2"
        default = self._run(mol, basis, rhf, df)
        explicit = self._run(mol, basis, rhf, df, pno_norm="mp2")
        assert default.converged and explicit.converged
        assert default.e_corr == explicit.e_corr
        assert default.pno_per_pair == explicit.pno_per_pair
        assert default.pno_norm == "mp2"

    def test_mp2_norm_keeps_more_pnos_at_the_same_threshold(self):
        """The measured direction and size of the effect.

        On H2O/def2-SVP at NormalPNO the published density retains more PNOs
        per pair than the legacy one at the identical `tcut_pno`, i.e. the
        default is effectively looser than the preset name implies. Across
        H2O and NH3 (cc-pVDZ and cc-pVTZ) and the S22-01/02/08 dimers the
        deficit ran 4.6-18.5 %.
        """
        mol, basis, rhf, df, _ = _setup(H2O_ATOMS, "def2-svp")
        legacy = self._run(mol, basis, rhf, df, pno_norm="legacy")
        mp2 = self._run(mol, basis, rhf, df, pno_norm="mp2")
        assert legacy.converged and mp2.converged
        assert legacy.pno_per_pair.keys() == mp2.pno_per_pair.keys()
        for key, n_legacy in legacy.pno_per_pair.items():
            assert mp2.pno_per_pair[key] >= n_legacy, key
        avg_legacy = sum(legacy.pno_per_pair.values()) / len(legacy.pno_per_pair)
        avg_mp2 = sum(mp2.pno_per_pair.values()) / len(mp2.pno_per_pair)
        assert avg_mp2 > avg_legacy
        assert mp2.e_corr != legacy.e_corr  # a different retained space

    def test_the_two_published_norms_nearly_coincide(self):
        """Riplinger and Neese's Fig. 2, reproduced.

        The IEPA norm is the MP2 norm times the per-pair scalar
        (1 + delta_ij) / N_ij with N_ij = 1 + <Tt+ T>. On physical amplitudes
        N_ij is within a percent of 1, so the two agree closely; that is why
        the 2013 paper could switch conventions and why the ORCA manual calls
        them "near identical".
        """
        mol, basis, rhf, df, _ = _setup(H2O_ATOMS, "def2-svp")
        mp2 = self._run(mol, basis, rhf, df, pno_norm="mp2")
        iepa = self._run(mol, basis, rhf, df, pno_norm="iepa")
        assert mp2.converged and iepa.converged
        avg_mp2 = sum(mp2.pno_per_pair.values()) / len(mp2.pno_per_pair)
        avg_iepa = sum(iepa.pno_per_pair.values()) / len(iepa.pno_per_pair)
        assert abs(avg_iepa - avg_mp2) <= 1.0
        assert abs(iepa.e_corr - mp2.e_corr) < 1e-4

    def test_full_pno_space_is_norm_independent(self):
        """At `tcut_pno=0` nothing is truncated, so the density cannot matter.

        Every convention is positive semidefinite with the same null space, so
        the retained space is the whole domain and the energy is identical.
        """
        mol, basis, rhf, df, _ = _setup(H2O_ATOMS, "def2-svp")
        runs = {
            norm: run_local_dlpno_ccsd(
                mol,
                basis,
                rhf,
                df,
                _LocalCCSDOptions(
                    tcut_pno=0.0,
                    tcut_mkn=0.0,
                    tcut_pairs=0.0,
                    coupling_radius=0.0,
                    compute_triples=False,
                    pno_norm=norm,
                ),
            )
            for norm in ("legacy", "mp2", "iepa")
        }
        assert all(r.converged for r in runs.values())
        assert runs["mp2"].e_corr == pytest.approx(runs["legacy"].e_corr, abs=1e-10)
        assert runs["iepa"].e_corr == pytest.approx(runs["legacy"].e_corr, abs=1e-10)

    def test_unknown_norm_is_refused(self):
        mol, basis, rhf, df, _ = _setup(H2_ATOMS, "sto-3g")
        with pytest.raises(ValueError, match="unknown pno_norm"):
            run_local_dlpno_ccsd(
                mol, basis, rhf, df, _LocalCCSDOptions(pno_norm="meyer")
            )


class TestPairSpaceLadder:
    """#700: the particle-particle ladder is contracted in each pair's own
    PNO space, exactly.

    After #689 the extended-domain residual costs one canonical-shaped CCSD
    contraction per (coupling set, extended domain) group. On any molecule
    whose domains span it there is one group with ``n_ext = n_vir``, so the
    per-iteration cost equals canonical DF-CCSD and the local method saves
    nothing -- measured on an n-alkane ladder, ``n_ext/n_vir`` is still 1.00
    at C6. The O(n_ext^4) part of that is the ladder.

    Riplinger and Neese, J. Chem. Phys. 138, 034106 (2013), Sec. II C contract
    every residual term in the SOURCE pair's PNO space. For the ladder the
    source is the target pair itself: the contracted indices e,f are carried
    by ``tau_ij``. Moving it there is exact, not an approximation -- which is
    what these tests pin, because an "optimisation" that quietly changed the
    energy would look like a speedup.
    """

    def _run(self, mol, basis, rhf, df, ladder, **kw):
        return run_local_dlpno_ccsd(
            mol, basis, rhf, df,
            LocalCCSDOptions(
                residual_domain="extended",
                use_cpp_kernel=False,
                ladder_in_pno_space=ladder,
                max_iter=60,
                conv_tol_energy=1e-11,
                conv_tol_residual=1e-9,
                **kw,
            ),
        )

    def test_energy_is_unchanged_to_machine_precision(self):
        """The whole claim: same fixed point, not merely a close one."""
        mol, basis, rhf, df, _canon = _setup(H2O_ATOMS, "def2-svp")
        old = self._run(mol, basis, rhf, df, False)
        new = self._run(mol, basis, rhf, df, True)
        assert old.converged and new.converged
        assert new.n_iter == old.n_iter
        assert abs(new.e_corr - old.e_corr) < 1e-12 * abs(old.e_corr)

    def test_exact_with_singles_outside_the_pair_space(self):
        """tau_ij = t2_ij + t1_i x t1_j, and the singles are expanded in the
        DIAGONAL pairs' PNOs, so the t1 part is NOT inside pair (i,j)'s space.
        Projecting it there first is wrong by ~1e-7 Ha on an S22 dimer --
        above this tolerance, so this test is what distinguishes the exact
        mixed-basis treatment from the plausible-looking shortcut.
        """
        mol, basis, rhf, df = _setup_ccpvdz(S22_01_A + S22_01_B)
        old = self._run(mol, basis, rhf, df, False, tcut_pno=3.33e-7,
                        tcut_mkn=1e-3, tcut_pairs=1e-4)
        new = self._run(mol, basis, rhf, df, True, tcut_pno=3.33e-7,
                        tcut_mkn=1e-3, tcut_pairs=1e-4)
        assert old.converged and new.converged
        assert old.t1_norm > 1e-6, "a system with no singles cannot test this"
        assert abs(new.e_corr - old.e_corr) < 1e-12 * abs(old.e_corr)

    def test_ladder_block_matches_term_by_term(self):
        """Block by block, not just in the contracted energy: the pair-space
        ladder reproduces the residual the extended basis produces."""
        import numpy as np
        from vibeqc.dlpno._ccsd_cs import (
            _blocks, cs_ccsd_pair_ladder, cs_ccsd_residual,
        )

        rng = np.random.default_rng(11)
        nL, n, naux = 4, 9, 17
        B_ov = rng.standard_normal((naux, nL, n)) * 0.1
        B_vv = rng.standard_normal((naux, n, n)) * 0.1
        B_vv = 0.5 * (B_vv + B_vv.transpose(0, 2, 1))
        B_oo = rng.standard_normal((naux, nL, nL)) * 0.1
        B_oo = 0.5 * (B_oo + B_oo.transpose(0, 2, 1))
        V = _blocks(B_ov, B_vv, B_oo)
        t1 = rng.standard_normal((nL, n)) * 0.05
        t2 = rng.standard_normal((nL, nL, n, n)) * 0.05
        t2 = 0.5 * (t2 + t2.transpose(1, 0, 3, 2))
        f_oo = rng.standard_normal((nL, nL)); f_oo = 0.5 * (f_oo + f_oo.T)
        f_vv = rng.standard_normal((n, n)); f_vv = 0.5 * (f_vv + f_vv.T)
        f_ov = rng.standard_normal((nL, n)) * 0.1

        _, R2_full = cs_ccsd_residual(t1, t2, f_oo, f_vv, f_ov, V)
        V_no_vvvv = {k: v for k, v in V.items() if k != "vvvv"}
        _, R2_no_ladder = cs_ccsd_residual(
            t1, t2, f_oo, f_vv, f_ov, V_no_vvvv, include_ladder=False
        )
        tau = t2 + np.einsum("ia,jb->ijab", t1, t1)
        # Here every PNO space is the same space, so the mixed-basis singles
        # vectors reduce to the plain ones; what this pins is the term algebra.
        for li in range(nL):
            for lj in range(nL):
                u_i = np.einsum("Pae,e->Pa", B_vv, t1[li])
                v_j = np.einsum("Pae,e->Pa", B_vv, t1[lj])
                w_i = np.einsum("Pme,e->Pm", B_ov, t1[li])
                w_j = np.einsum("Pme,e->Pm", B_ov, t1[lj])
                lad = cs_ccsd_pair_ladder(
                    t2[li, lj], t1, tau, B_ov, B_vv, u_i, v_j, w_i, w_j
                )
                got = R2_no_ladder[li, lj] + lad
                ref = R2_full[li, lj]
                assert np.allclose(got, ref, rtol=0, atol=1e-12 * np.abs(ref).max())

    def test_refuses_when_the_compiled_core_predates_the_flag(self, monkeypatch):
        """A core built before `include_ladder` computes the ladder anyway, so
        adding the pair-space term would DOUBLE-COUNT it. Refuse instead of
        silently returning a wrong energy."""
        import vibeqc.dlpno.ccsd_local_solver as solver

        monkeypatch.setattr(solver, "_HAVE_CPP_LADDER_FLAG", False)
        mol, basis, rhf, df, _canon = _setup(H2O_ATOMS, "def2-svp")
        with pytest.raises(ValueError, match="include_ladder"):
            run_local_dlpno_ccsd(
                mol, basis, rhf, df,
                LocalCCSDOptions(
                    residual_domain="extended",
                    use_cpp_kernel=True,
                    ladder_in_pno_space=True,
                ),
            )

    def test_compiled_path_is_exact_too(self):
        """The whole point of the C++ include_ladder flag: the compiled route
        can omit the term so the pair-space one can replace it."""
        import vibeqc.dlpno.ccsd_local_solver as solver

        if not solver._HAVE_CPP_LADDER_FLAG:
            pytest.skip("compiled core predates the include_ladder flag")
        mol, basis, rhf, df = _setup_ccpvdz(S22_01_A + S22_01_B)
        common = dict(
            residual_domain="extended", use_cpp_kernel=True, max_iter=60,
            conv_tol_energy=1e-11, conv_tol_residual=1e-9,
            tcut_pno=3.33e-7, tcut_mkn=1e-3, tcut_pairs=1e-4,
        )
        off = run_local_dlpno_ccsd(
            mol, basis, rhf, df,
            LocalCCSDOptions(ladder_in_pno_space=False, **common))
        on = run_local_dlpno_ccsd(
            mol, basis, rhf, df,
            LocalCCSDOptions(ladder_in_pno_space=True, **common))
        assert off.converged and on.converged
        assert on.n_ladder_pno_groups > 0
        assert abs(on.e_corr - off.e_corr) < 1e-12 * abs(off.e_corr)

    def test_the_locality_gate_leaves_compact_molecules_bit_identical(self):
        """Moving the ladder is a REGRESSION when the extended basis is barely
        wider than the pair space -- 3x slower on water/def2-SVP, where the
        ratio is 1.2. Auto must therefore leave such systems alone, and leave
        them BIT-identical, since that is where the retained reference values
        live.
        """
        mol, basis, rhf, df, _canon = _setup(H2O_ATOMS, "def2-svp")
        common = dict(
            residual_domain="extended", use_cpp_kernel=True, max_iter=60,
            conv_tol_energy=1e-11, conv_tol_residual=1e-9,
        )
        off = run_local_dlpno_ccsd(
            mol, basis, rhf, df,
            LocalCCSDOptions(ladder_in_pno_space=False, **common))
        auto = run_local_dlpno_ccsd(
            mol, basis, rhf, df, LocalCCSDOptions(**common))
        assert auto.n_ladder_pno_groups == 0, "gate should not fire here"
        assert auto.e_corr == off.e_corr, "auto must not perturb a gated-off run"


class TestPairSpaceRing:
    """#700 (b): the ring terms are contracted in each SOURCE pair's PNO
    space, exactly.

    Once the particle-particle ladder has moved out (#700 (a)), the ring terms
    are what is left: measured with direct timers inside the compiled kernel
    they are 87% (C3H8) to 94% (C4H10) of the extended-domain residual, split
    between building W1/W2/WX and contracting them.

    Riplinger and Neese, J. Chem. Phys. 138, 034106 (2013), Eqs. (26)-(28).
    The ladder moved into the TARGET pair's space because its contracted
    indices e,f are carried by tau_ij. These do not: the contracted index
    belongs to the SOURCE pair, so the amplitude stays in its own space and
    only the free virtual index crosses over, through the Eq. (30) overlap
    S^{ij,kl} = U_ij^T U_kl. Projecting the source amplitude into the target
    pair first is the #98 truncation -- it looks like the same optimisation
    and silently changes the energy, which is what these tests exist to catch.
    """

    def _run(self, mol, basis, rhf, df, ring, **kw):
        # numpy route: the equivalence is a statement about the algebra, and
        # the numpy residual is where it is easiest to read. The compiled
        # route is covered separately below.
        return run_local_dlpno_ccsd(
            mol, basis, rhf, df,
            LocalCCSDOptions(
                residual_domain="extended",
                use_cpp_kernel=False,
                ring_in_pno_space=ring,
                max_iter=60,
                conv_tol_energy=1e-11,
                conv_tol_residual=1e-9,
                **kw,
            ),
        )

    def test_energy_is_unchanged_to_machine_precision(self):
        """Same fixed point, not merely a close one."""
        mol, basis, rhf, df, _canon = _setup(H2O_ATOMS, "def2-svp")
        old = self._run(mol, basis, rhf, df, False)
        new = self._run(mol, basis, rhf, df, True)
        assert old.converged and new.converged
        assert new.n_iter == old.n_iter
        assert abs(new.e_corr - old.e_corr) < 1e-12 * abs(old.e_corr)

    def test_exact_where_the_spaces_actually_differ(self):
        """H2O/def2-SVP has locality ratio ~1.2: its PNO and extended spaces
        nearly coincide, so agreement there is close to vacuous. This runs
        with real PNO truncation and a genuinely wider extended basis, which
        is where a truncating implementation would part company.
        """
        mol, basis, rhf, df = _setup_ccpvdz(S22_01_A + S22_01_B)
        common = dict(tcut_pno=3.33e-7, tcut_mkn=1e-3, tcut_pairs=1e-4)
        old = self._run(mol, basis, rhf, df, False, **common)
        new = self._run(mol, basis, rhf, df, True, **common)
        assert old.converged and new.converged
        assert abs(new.e_corr - old.e_corr) < 1e-12 * abs(old.e_corr)

    def test_the_truncating_route_really_does_differ(self):
        """Teeth for the two tests above.

        ``residual_domain="pair"`` projects the coupled amplitudes into the
        target pair BEFORE contracting -- the #98 truncation, and exactly the
        shortcut the source-space formulation avoids. If it agreed to machine
        precision on this system then matching `extended` would prove nothing.
        """
        mol, basis, rhf, df = _setup_ccpvdz(S22_01_A + S22_01_B)
        common = dict(tcut_pno=3.33e-7, tcut_mkn=1e-3, tcut_pairs=1e-4)
        ext = self._run(mol, basis, rhf, df, True, **common)
        trunc = run_local_dlpno_ccsd(
            mol, basis, rhf, df,
            LocalCCSDOptions(
                residual_domain="pair", use_cpp_kernel=False, max_iter=60,
                conv_tol_energy=1e-11, conv_tol_residual=1e-9, **common),
        )
        assert ext.converged and trunc.converged
        assert abs(trunc.e_corr - ext.e_corr) > 1e-9 * abs(ext.e_corr), (
            "the truncating route agrees to machine precision here, so this "
            "system cannot distinguish the two formulations"
        )

    def test_include_ring_false_removes_exactly_the_ring_terms(self):
        """Pins the kernel gate itself: dropping the ring terms must remove
        those three contractions and nothing else."""
        import numpy as np
        from vibeqc.dlpno._ccsd_cs import _blocks, cs_ccsd_residual

        rng = np.random.default_rng(23)
        nL, n, naux = 4, 9, 17
        B_ov = rng.standard_normal((naux, nL, n)) * 0.1
        B_vv = rng.standard_normal((naux, n, n)) * 0.1
        B_vv = 0.5 * (B_vv + B_vv.transpose(0, 2, 1))
        B_oo = rng.standard_normal((naux, nL, nL)) * 0.1
        B_oo = 0.5 * (B_oo + B_oo.transpose(0, 2, 1))
        V = _blocks(B_ov, B_vv, B_oo)
        t1 = rng.standard_normal((nL, n)) * 0.05
        t2 = rng.standard_normal((nL, nL, n, n)) * 0.05
        t2 = 0.5 * (t2 + t2.transpose(1, 0, 3, 2))
        f_oo = rng.standard_normal((nL, nL)); f_oo = 0.5 * (f_oo + f_oo.T)
        f_vv = rng.standard_normal((n, n)); f_vv = 0.5 * (f_vv + f_vv.T)
        f_ov = rng.standard_normal((nL, n)) * 0.1

        _, R2_full = cs_ccsd_residual(t1, t2, f_oo, f_vv, f_ov, V)
        _, R2_bare, (W1, W2, WX) = cs_ccsd_residual(
            t1, t2, f_oo, f_vv, f_ov, V, include_ring=False, return_ring=True
        )
        ring_half = (
            np.einsum("imae,mejb->ijab", t2 - t2.transpose(1, 0, 2, 3), W1)
            + np.einsum("imae,mejb->ijab", t2, W2)
            + np.einsum("mjae,meib->ijab", t2, WX)
        )
        got = R2_bare + ring_half + ring_half.transpose(1, 0, 3, 2)
        assert np.allclose(got, R2_full, rtol=0,
                           atol=1e-12 * np.abs(R2_full).max())

    def test_refuses_when_the_compiled_core_predates_the_flag(self, monkeypatch):
        """A core built before `include_ring` contracts the ring terms in the
        extended basis regardless, so adding the source-space term would
        DOUBLE-COUNT them. Refuse rather than return a wrong energy."""
        import vibeqc.dlpno.ccsd_local_solver as solver

        if not solver._HAVE_CPP_RESIDUAL:
            pytest.skip("compiled residual kernel not built")
        monkeypatch.setattr(solver, "_HAVE_CPP_RING_FLAG", False)
        mol, basis, rhf, df, _canon = _setup(H2O_ATOMS, "def2-svp")
        with pytest.raises(ValueError, match="include_ring"):
            run_local_dlpno_ccsd(
                mol, basis, rhf, df,
                LocalCCSDOptions(residual_domain="extended",
                                 use_cpp_kernel=True, ring_in_pno_space=True),
            )

    def test_compiled_route_agrees_with_the_extended_basis(self):
        """The same equivalence on the compiled kernel, which is the route
        that actually runs: `include_ring=False` there omits the three ring
        contractions and hands back W1/W2/WX for the source-space
        contraction."""
        import vibeqc.dlpno.ccsd_local_solver as solver

        if not (solver._HAVE_CPP_RESIDUAL and solver._HAVE_CPP_RING_FLAG):
            pytest.skip("compiled core without the include_ring flag")
        mol, basis, rhf, df, _canon = _setup(H2O_ATOMS, "def2-svp")
        common = dict(
            residual_domain="extended", use_cpp_kernel=True, max_iter=60,
            conv_tol_energy=1e-11, conv_tol_residual=1e-9,
        )
        old = run_local_dlpno_ccsd(
            mol, basis, rhf, df,
            LocalCCSDOptions(ring_in_pno_space=False, **common))
        new = run_local_dlpno_ccsd(
            mol, basis, rhf, df,
            LocalCCSDOptions(ring_in_pno_space=True, **common))
        assert old.converged and new.converged
        assert new.n_iter == old.n_iter
        assert abs(new.e_corr - old.e_corr) < 1e-12 * abs(old.e_corr)

    def test_refuses_in_legacy_pair_mode(self):
        """"pair" mode already truncates the coupled amplitudes; layering the
        source-space ring on top would be meaningless."""
        mol, basis, rhf, df, _canon = _setup(H2O_ATOMS, "def2-svp")
        with pytest.raises(ValueError, match="residual_domain"):
            run_local_dlpno_ccsd(
                mol, basis, rhf, df,
                LocalCCSDOptions(residual_domain="pair",
                                 use_cpp_kernel=False, ring_in_pno_space=True),
            )

