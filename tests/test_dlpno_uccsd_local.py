"""Local-domain open-shell DLPNO-UCCSD correctness ratchets.

The dense O(N^6) pilot is the independent oracle.  At full PNO and occupied
domains the local solver must reproduce it while obtaining every residual
from the caller-supplied native local-domain kernel.
"""

from __future__ import annotations

import math
from functools import partial

import pytest
from vibeqc import BasisSet, UHFOptions, run_uhf
from vibeqc._vibeqc_core import Atom, Molecule
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno.uccsd import (
    DLPNOUCCSDPilotOptions as _DLPNOUCCSDPilotOptions,
    run_dlpno_uccsd_pilot,
)
from vibeqc.dlpno.uccsd_local_solver import (
    LocalUCCSDOptions as _LocalUCCSDOptions,
    run_local_dlpno_uccsd,
)

ANGSTROM_TO_BOHR = 1.8897259886
AUX = "def2-svp-rifit"

# The broad local-solver ratchet predates #140/#448. Keep its all-electron,
# full-domain defaults explicit; the unified public defaults have dedicated
# convention tests elsewhere.
DLPNOUCCSDPilotOptions = partial(
    _DLPNOUCCSDPilotOptions,
    n_frozen=0,
    tcut_pno=1e-7,
)
LocalUCCSDOptions = partial(
    _LocalUCCSDOptions,
    n_frozen=0,
    tcut_pno=1e-7,
    tcut_pairs=0.0,
    tcut_mkn=0.0,
)
OH = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.0, 0.97 * ANGSTROM_TO_BOHR]),
]
SEPARATED_H_H2 = [
    (1, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.0, 8.0 * ANGSTROM_TO_BOHR]),
    (1, [0.0, 0.0, 8.74 * ANGSTROM_TO_BOHR]),
]
SEPARATED_OH_H2 = OH + [
    (1, [0.0, 0.0, 8.0 * ANGSTROM_TO_BOHR]),
    (1, [0.0, 0.0, 8.74 * ANGSTROM_TO_BOHR]),
]
# Triplet methylene, HCH = 133.9 deg, r(CH) = 1.075 A.  The only high-spin
# system in this module: every other fixture is a doublet, so this is what
# exercises unequal alpha/beta occupied block sizes.
_CH2_HALF_ANGLE = math.radians(133.9 / 2.0)
_CH2_R = 1.075 * ANGSTROM_TO_BOHR
CH2_TRIPLET = [
    (6, [0.0, 0.0, 0.0]),
    (1, [_CH2_R * math.sin(_CH2_HALF_ANGLE), 0.0, _CH2_R * math.cos(_CH2_HALF_ANGLE)]),
    (1, [-_CH2_R * math.sin(_CH2_HALF_ANGLE), 0.0, _CH2_R * math.cos(_CH2_HALF_ANGLE)]),
]


@pytest.fixture(scope="module")
def oh_sto3g():
    molecule = Molecule(
        [Atom(z, position) for z, position in OH],
        charge=0,
        multiplicity=2,
    )
    basis = BasisSet(molecule, "sto-3g")
    scf_options = UHFOptions()
    scf_options.max_iter = 300
    uhf = run_uhf(molecule, basis, scf_options)
    assert uhf.converged
    df = DensityFitting(
        basis,
        BasisSet(molecule, AUX),
        aux_basis_name=AUX,
    )
    return molecule, basis, uhf, df


@pytest.fixture(scope="module")
def ch2_triplet_sto3g():
    molecule = Molecule(
        [Atom(z, position) for z, position in CH2_TRIPLET],
        charge=0,
        multiplicity=3,
    )
    basis = BasisSet(molecule, "sto-3g")
    scf_options = UHFOptions()
    scf_options.max_iter = 400
    uhf = run_uhf(molecule, basis, scf_options)
    assert uhf.converged
    df = DensityFitting(
        basis,
        BasisSet(molecule, AUX),
        aux_basis_name=AUX,
    )
    return molecule, basis, uhf, df


@pytest.fixture(scope="module")
def separated_h_h2_sto3g():
    molecule = Molecule(
        [Atom(z, position) for z, position in SEPARATED_H_H2],
        charge=0,
        multiplicity=2,
    )
    basis = BasisSet(molecule, "sto-3g")
    scf_options = UHFOptions()
    scf_options.max_iter = 300
    uhf = run_uhf(molecule, basis, scf_options)
    assert uhf.converged
    df = DensityFitting(
        basis,
        BasisSet(molecule, AUX),
        aux_basis_name=AUX,
    )
    return molecule, basis, uhf, df


@pytest.fixture(scope="module")
def separated_oh_h2_sto3g():
    molecule = Molecule(
        [Atom(z, position) for z, position in SEPARATED_OH_H2],
        charge=0,
        multiplicity=2,
    )
    basis = BasisSet(molecule, "sto-3g")
    scf_options = UHFOptions()
    scf_options.max_iter = 300
    uhf = run_uhf(molecule, basis, scf_options)
    assert uhf.converged
    df = DensityFitting(
        basis,
        BasisSet(molecule, AUX),
        aux_basis_name=AUX,
    )
    return molecule, basis, uhf, df


@pytest.mark.parametrize("localise", ["none", "boys"])
def test_full_domain_matches_dense_pilot(oh_sto3g, localise):
    molecule, basis, uhf, df = oh_sto3g
    pilot = run_dlpno_uccsd_pilot(
        molecule,
        basis,
        uhf,
        df,
        DLPNOUCCSDPilotOptions(localise=localise, tcut_pno=0.0),
    )
    local = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(
            localise=localise,
            tcut_pno=0.0,
            coupling_radius=0.0,
        ),
    )

    assert pilot.converged
    assert local.converged
    assert local.e_corr == pytest.approx(pilot.e_corr, abs=1e-8)
    assert local.e_t == 0.0
    assert local.e_total == pytest.approx(local.e_hf + local.e_corr)
    assert local.n_pairs == pilot.n_pairs
    assert local.avg_pno == pytest.approx(
        2 * basis.nbasis - molecule.n_electrons()
    )
    assert local.avg_pao == pytest.approx(
        2 * basis.nbasis - molecule.n_electrons()
    )
    assert local.avg_singles_domain == pytest.approx(
        2 * basis.nbasis - molecule.n_electrons()
    )
    assert set(local.singles_per_occ) == set(range(molecule.n_electrons()))
    assert local.avg_coupled_occ == pytest.approx(molecule.n_electrons())


@pytest.mark.slow
def test_truncated_dz_matches_dense_pno_pilot():
    molecule = Molecule(
        [Atom(z, position) for z, position in OH],
        charge=0,
        multiplicity=2,
    )
    basis = BasisSet(molecule, "def2-svp")
    scf_options = UHFOptions()
    scf_options.max_iter = 300
    uhf = run_uhf(molecule, basis, scf_options)
    assert uhf.converged
    df = DensityFitting(
        basis,
        BasisSet(molecule, AUX),
        aux_basis_name=AUX,
    )
    threshold = 1e-7
    pilot = run_dlpno_uccsd_pilot(
        molecule,
        basis,
        uhf,
        df,
        DLPNOUCCSDPilotOptions(localise="boys", tcut_pno=threshold),
    )
    local = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(
            localise="boys",
            tcut_pno=threshold,
            coupling_radius=0.0,
            max_iter=120,
        ),
    )

    assert pilot.converged
    assert local.converged
    assert local.e_corr == pytest.approx(pilot.e_corr, abs=2e-6)
    full_virtuals = 2 * basis.nbasis - molecule.n_electrons()
    assert local.avg_pno < 0.6 * full_virtuals
    assert local.avg_singles_domain == pytest.approx(full_virtuals)


def test_rejects_negative_pno_threshold(oh_sto3g):
    molecule, basis, uhf, df = oh_sto3g
    with pytest.raises(ValueError, match="tcut_pno"):
        run_local_dlpno_uccsd(
            molecule,
            basis,
            uhf,
            df,
            LocalUCCSDOptions(tcut_pno=-1.0),
        )


def test_rejects_negative_pair_threshold(oh_sto3g):
    molecule, basis, uhf, df = oh_sto3g
    with pytest.raises(ValueError, match="tcut_pairs"):
        run_local_dlpno_uccsd(
            molecule,
            basis,
            uhf,
            df,
            LocalUCCSDOptions(tcut_pairs=-1.0),
        )


@pytest.mark.parametrize("field", ["tcut_mkn", "lindep"])
def test_rejects_negative_pao_threshold(oh_sto3g, field):
    molecule, basis, uhf, df = oh_sto3g
    with pytest.raises(ValueError, match=field):
        run_local_dlpno_uccsd(
            molecule,
            basis,
            uhf,
            df,
            LocalUCCSDOptions(**{field: -1.0}),
        )


def test_rejects_negative_singles_pno_threshold(oh_sto3g):
    molecule, basis, uhf, df = oh_sto3g
    with pytest.raises(ValueError, match="tcut_pno_singles"):
        run_local_dlpno_uccsd(
            molecule,
            basis,
            uhf,
            df,
            LocalUCCSDOptions(tcut_pno_singles=-1.0),
        )


def test_rejects_negative_tno_threshold(oh_sto3g):
    molecule, basis, uhf, df = oh_sto3g
    with pytest.raises(ValueError, match="tcut_tno"):
        run_local_dlpno_uccsd(
            molecule,
            basis,
            uhf,
            df,
            LocalUCCSDOptions(tcut_tno=-1.0),
        )


def test_rejects_unknown_triples_mode(oh_sto3g):
    molecule, basis, uhf, df = oh_sto3g
    with pytest.raises(ValueError, match="triples_mode"):
        run_local_dlpno_uccsd(
            molecule,
            basis,
            uhf,
            df,
            LocalUCCSDOptions(triples_mode="unknown"),
        )


def test_full_tno_domain_matches_dense_pilot_triples(oh_sto3g):
    molecule, basis, uhf, df = oh_sto3g
    pilot = run_dlpno_uccsd_pilot(
        molecule,
        basis,
        uhf,
        df,
        DLPNOUCCSDPilotOptions(
            localise="none",
            tcut_pno=0.0,
            compute_triples=True,
        ),
    )
    local = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(
            localise="none",
            tcut_pno=0.0,
            tcut_pairs=0.0,
            tcut_mkn=0.0,
            tcut_pno_singles=0.0,
            coupling_radius=0.0,
            compute_triples=True,
            tcut_tno=0.0,
        ),
    )

    no = molecule.n_electrons()
    nv = 2 * basis.nbasis - no
    assert pilot.converged
    assert local.converged
    assert local.e_corr == pytest.approx(pilot.e_corr, abs=1e-8)
    assert local.e_t == pytest.approx(pilot.e_t, abs=1e-11)
    assert local.e_total == pytest.approx(
        local.e_hf + local.e_corr + local.e_t,
        abs=1e-12,
    )
    assert local.n_triples == math.comb(no, 3)
    assert local.n_screened_triples == 0
    assert local.avg_tno == pytest.approx(nv)
    assert set(local.tno_per_triple) == {
        (i, j, k)
        for i in range(no)
        for j in range(i + 1, no)
        for k in range(j + 1, no)
    }


@pytest.mark.parametrize("triples_mode", ["t0", "t1"])
def test_high_spin_triplet_matches_dense_pilot(ch2_triplet_sto3g, triples_mode):
    """First multiplicity=3 coverage for the open-shell local engine.

    Every other fixture here, and every case in ``test_dlpno_uccsd.py``, is a
    doublet, so the alpha and beta occupied blocks have always been within one
    orbital of each other.  Triplet CH2 has two more alpha than beta occupieds,
    which is what exercises the per-spin partitioning: the ``(T1)`` rotation
    diagonalises the occupied Fock in each spin block separately, and the
    pair/singles domains are built per spin.  At full domain both triples modes
    must still reproduce the dense pilot exactly.
    """
    molecule, basis, uhf, df = ch2_triplet_sto3g
    pilot = run_dlpno_uccsd_pilot(
        molecule,
        basis,
        uhf,
        df,
        DLPNOUCCSDPilotOptions(
            localise="none", tcut_pno=0.0, compute_triples=True
        ),
    )
    local = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(
            localise="none",
            tcut_pno=0.0,
            tcut_pairs=0.0,
            tcut_mkn=0.0,
            tcut_pno_singles=0.0,
            coupling_radius=0.0,
            compute_triples=True,
            tcut_tno=0.0,
            triples_mode=triples_mode,
        ),
    )

    ne = molecule.n_electrons()
    two_s = molecule.multiplicity - 1
    n_alpha = (ne + two_s) // 2
    n_beta = (ne - two_s) // 2
    assert n_alpha - n_beta == 2, "fixture must be genuinely high-spin"

    assert pilot.converged
    assert local.converged
    assert local.e_corr == pytest.approx(pilot.e_corr, abs=1e-8)
    assert local.e_t == pytest.approx(pilot.e_t, abs=1e-11)
    assert local.n_triples == math.comb(ne, 3)
    assert local.n_degenerate_tno_triples == 0
    assert local.avg_tno == pytest.approx(2 * basis.nbasis - ne)


@pytest.mark.parametrize("localise", ["none", "boys"])
def test_t1_full_domain_matches_dense_pilot_triples(oh_sto3g, localise):
    molecule, basis, uhf, df = oh_sto3g
    pilot = run_dlpno_uccsd_pilot(
        molecule,
        basis,
        uhf,
        df,
        DLPNOUCCSDPilotOptions(
            localise=localise,
            tcut_pno=0.0,
            compute_triples=True,
        ),
    )
    local = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(
            localise=localise,
            tcut_pno=0.0,
            tcut_pairs=0.0,
            tcut_mkn=0.0,
            tcut_pno_singles=0.0,
            coupling_radius=0.0,
            compute_triples=True,
            tcut_tno=0.0,
            triples_mode="t1",
        ),
    )

    no = molecule.n_electrons()
    nv = 2 * basis.nbasis - no
    assert pilot.converged
    assert local.converged
    assert local.triples_mode == "t1"
    assert local.e_corr == pytest.approx(pilot.e_corr, abs=1e-8)
    assert local.e_t == pytest.approx(pilot.e_t, abs=1e-11)
    assert local.n_triples == math.comb(no, 3)
    assert local.avg_tno == pytest.approx(nv)


def test_t1_tno_truncation_reduces_domain(oh_sto3g):
    molecule, basis, uhf, df = oh_sto3g
    common = dict(
        localise="boys",
        tcut_pno=0.0,
        tcut_pairs=0.0,
        tcut_mkn=0.0,
        tcut_pno_singles=0.0,
        coupling_radius=0.0,
        compute_triples=True,
        triples_mode="t1",
    )
    reference = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(tcut_tno=0.0, **common),
    )
    local = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(tcut_tno=1e-18, **common),
    )

    assert reference.converged
    assert local.converged
    assert local.avg_tno < reference.avg_tno
    assert local.e_t == pytest.approx(reference.e_t, abs=3e-8)
    assert reference.n_degenerate_tno_triples == 0


def test_t1_tno_truncation_collapses_on_a_minimal_basis(oh_sto3g):
    """A minimal basis cannot calibrate ``tcut_tno`` -- it is all-or-nothing.

    A triple excitation needs three *distinct* spin virtuals.  OH/STO-3G
    retains only 3.0 TNOs per triple at full domain, so any truncation drops
    75 of the 84 triples below three and their energy vanishes identically
    rather than degrading: `e_t` falls from -2.29e-07 Ha to order 1e-46.  This
    is pinned so that a later calibration is not attempted on a minimal basis
    and read as a threshold-accuracy curve -- see
    ``test_t1_tno_threshold_calibration_is_monotone_on_dz`` for the real one.
    """
    molecule, basis, uhf, df = oh_sto3g
    common = dict(
        localise="boys",
        tcut_pno=0.0,
        tcut_pairs=0.0,
        coupling_radius=0.0,
        compute_triples=True,
        triples_mode="t1",
    )
    full = run_local_dlpno_uccsd(
        molecule, basis, uhf, df, LocalUCCSDOptions(tcut_tno=0.0, **common)
    )
    truncated = run_local_dlpno_uccsd(
        molecule, basis, uhf, df, LocalUCCSDOptions(tcut_tno=1e-12, **common)
    )

    assert full.converged and truncated.converged
    assert full.avg_tno == pytest.approx(3.0)
    assert full.n_degenerate_tno_triples == 0
    assert full.e_t == pytest.approx(-2.290917743938744e-07, rel=1e-9)

    # Not "smaller" -- gone. 75 of the 84 triples lose their third virtual and
    # are identically zero; the 9 that survive contribute ~1e-46 Ha, so the
    # correction is destroyed either way.
    assert truncated.avg_tno < 2.0
    assert truncated.n_triples == 84
    assert truncated.n_degenerate_tno_triples == 75
    assert abs(truncated.e_t) < 1e-40


@pytest.mark.slow
def test_t1_tno_threshold_calibration_is_monotone_on_dz():
    """(T1) ``tcut_tno`` accuracy/domain ratchet on OH/def2-SVP.

    The calibration measured on 2026-07-29 (full-domain
    ``e_t = -1.778474224055274e-03`` Ha, 29.0 spin virtuals per triple):

        tcut_tno    e_t error vs full domain    avg TNO
        1e-12              +11.27 uHa           26.071
        1e-09              +12.39 uHa           24.726
        1e-06              +63.37 uHa           19.917
        1e-04             +759.06 uHa           10.548

    Two properties are pinned, both of which a truncation must satisfy and
    neither of which the previous single-point 1e-18 test could see: the error
    grows monotonically with the threshold, and it is signed -- truncation
    *loses* correlation, so every truncated `e_t` is less negative than the
    full-domain one.

    The third pin is where the curve stops being a truncation curve.  Through
    ``1e-6`` every triple keeps at least three spin virtuals, so the error is
    pure truncation; by ``1e-4`` the smallest domain is one virtual and 13 of
    the 84 triples have collapsed to identically zero, mixing collapse into the
    reported error.  ``1e-6`` is therefore the coarsest calibrated threshold on
    this system.
    """
    molecule = Molecule(
        [Atom(z, position) for z, position in OH],
        charge=0,
        multiplicity=2,
    )
    basis = BasisSet(molecule, "def2-svp")
    scf_options = UHFOptions()
    scf_options.max_iter = 300
    uhf = run_uhf(molecule, basis, scf_options)
    assert uhf.converged
    df = DensityFitting(
        basis, BasisSet(molecule, AUX), aux_basis_name=AUX
    )
    common = dict(
        localise="boys",
        tcut_pno=0.0,
        tcut_pairs=0.0,
        coupling_radius=0.0,
        compute_triples=True,
        triples_mode="t1",
    )

    runs = {
        tcut: run_local_dlpno_uccsd(
            molecule, basis, uhf, df, LocalUCCSDOptions(tcut_tno=tcut, **common)
        )
        for tcut in (0.0, 1e-9, 1e-6, 1e-4)
    }
    assert all(run.converged for run in runs.values())

    full = runs[0.0]
    assert full.e_t == pytest.approx(-1.778474224055274e-03, rel=1e-8)
    assert full.avg_tno == pytest.approx(29.0)

    thresholds = [1e-9, 1e-6, 1e-4]
    errors = [runs[t].e_t - full.e_t for t in thresholds]
    domains = [runs[t].avg_tno for t in thresholds]

    # Truncation only ever loses correlation energy.
    assert all(error > 0.0 for error in errors)
    # Coarser threshold: strictly worse energy, strictly smaller domain.
    assert errors == sorted(errors)
    assert domains == sorted(domains, reverse=True)

    # Where the curve stops being a truncation curve: through 1e-6 every
    # triple keeps at least three virtuals (min 8 at 1e-6), so the error is
    # pure truncation. At 1e-4 the smallest domain is a single virtual and 13
    # of the 84 triples are identically zero, so its +759 uHa mixes truncation
    # with outright collapse and must not be read as a truncation error.
    assert runs[0.0].n_degenerate_tno_triples == 0
    assert runs[1e-9].n_degenerate_tno_triples == 0
    assert runs[1e-6].n_degenerate_tno_triples == 0
    assert runs[1e-4].n_degenerate_tno_triples == 13

    # The literature default (T_CutTNO = 1e-9, Guo et al. 2018) stays well
    # inside 0.1 kcal/mol of the full domain on this system.
    assert errors[0] < 40e-6


@pytest.mark.parametrize(
    "screening",
    [{"coupling_radius": 1.0}, {"tcut_pairs": 1e-4}],
)
def test_t1_triples_reject_local_screening(oh_sto3g, screening):
    """Screening that really removes a triple still fails closed under (T1)."""
    molecule, basis, uhf, df = oh_sto3g
    with pytest.raises(NotImplementedError, match="triple screening") as excinfo:
        run_local_dlpno_uccsd(
            molecule,
            basis,
            uhf,
            df,
            LocalUCCSDOptions(
                localise="boys",
                tcut_pno=0.0,
                compute_triples=True,
                triples_mode="t1",
                **screening,
            ),
        )
    # The refusal names the realised screened count, not just the threshold.
    assert "screen" in str(excinfo.value)
    assert "occupied triples" in str(excinfo.value)


@pytest.mark.parametrize(
    "screening",
    [{"coupling_radius": 50.0}, {"coupling_radius": 1e6}, {"tcut_pairs": 0.0}],
)
def test_t1_triples_run_under_noop_screening(oh_sto3g, screening):
    """Thresholds that screen nothing leave (T1) reachable and unchanged.

    A ``coupling_radius`` well beyond the molecule removes no triple.  The
    guard is decided on that realised set, so (T1) runs the full triple list
    and must reproduce the unscreened (T1) energy exactly -- same code path,
    same amplitudes.  Note ``tcut_pairs`` is deliberately absent here except
    at its zero default: see
    ``test_any_positive_tcut_pairs_screens_this_system``.
    """
    molecule, basis, uhf, df = oh_sto3g
    common = dict(
        localise="boys",
        tcut_pno=0.0,
        compute_triples=True,
        triples_mode="t1",
    )
    reference = run_local_dlpno_uccsd(
        molecule, basis, uhf, df, LocalUCCSDOptions(**common)
    )
    relaxed = run_local_dlpno_uccsd(
        molecule, basis, uhf, df, LocalUCCSDOptions(**screening, **common)
    )

    assert reference.converged
    assert relaxed.converged
    assert relaxed.n_screened_triples == 0
    assert relaxed.n_triples == reference.n_triples
    assert relaxed.n_screened == 0
    assert relaxed.e_t == pytest.approx(reference.e_t, abs=1e-12)
    assert relaxed.e_corr == pytest.approx(reference.e_corr, abs=1e-12)


def test_any_positive_tcut_pairs_screens_this_system(oh_sto3g):
    """Why the (T1) relaxation buys nothing for ``tcut_pairs`` on OH/STO-3G.

    16 of the 36 spin-orbital pairs carry an identically zero MP2 energy (the
    same-spin pairs that vanish for this small doublet), so a ``tcut_pairs``
    as tight as 1e-14 already screens them, and every one of the 84 occupied
    triples then contains a screened pair.  The realised-set guard is
    therefore genuinely protective here rather than cosmetic: only
    ``coupling_radius`` reaches the no-op branch on this system.
    """
    molecule, basis, uhf, df = oh_sto3g
    t0 = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(
            localise="boys",
            tcut_pno=0.0,
            tcut_pairs=1e-14,
            compute_triples=True,
            triples_mode="t0",
        ),
    )
    assert t0.n_screened == 16
    assert t0.n_triples == 0
    assert t0.n_screened_triples == 84
    assert not t0.triples_executed
    assert all(
        abs(energy) < 1e-30
        for energy in t0.screened_pair_energies.values()
    )

    iterative = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(
            localise="boys",
            tcut_pno=0.0,
            tcut_pairs=1e-14,
            compute_triples=True,
            triples_mode="t1-iterative",
        ),
    )
    assert iterative.n_triples == 0
    assert iterative.n_screened_triples == 84
    assert not iterative.triples_executed
    assert iterative.e_t == 0.0

    with pytest.raises(NotImplementedError, match="84 of 84"):
        run_local_dlpno_uccsd(
            molecule,
            basis,
            uhf,
            df,
            LocalUCCSDOptions(
                localise="boys",
                tcut_pno=0.0,
                tcut_pairs=1e-14,
                compute_triples=True,
                triples_mode="t1",
            ),
        )


def test_t1_noop_screening_guard_matches_t0_triple_list(separated_h_h2_sto3g):
    """The guard and the (T0) triple loop share one screening predicate.

    On separated H + H2 a 5-bohr radius genuinely screens, so (T0) drops the
    triple and (T1) must refuse rather than silently evaluating a list its
    canonical occupied rotation cannot represent.
    """
    molecule, basis, uhf, df = separated_h_h2_sto3g
    common = dict(
        localise="boys",
        tcut_pno=0.0,
        tcut_pairs=0.0,
        coupling_radius=5.0,
        compute_triples=True,
    )
    t0 = run_local_dlpno_uccsd(
        molecule, basis, uhf, df, LocalUCCSDOptions(triples_mode="t0", **common)
    )
    assert t0.n_screened_triples > 0

    with pytest.raises(NotImplementedError, match="triple screening"):
        run_local_dlpno_uccsd(
            molecule,
            basis,
            uhf,
            df,
            LocalUCCSDOptions(triples_mode="t1", **common),
        )


def test_tno_truncation_preserves_separated_radical(
    separated_oh_h2_sto3g,
):
    molecule, basis, uhf, df = separated_oh_h2_sto3g
    common = dict(
        localise="boys",
        tcut_pno=0.0,
        tcut_pairs=0.0,
        tcut_mkn=1e-3,
        tcut_pno_singles=0.0,
        coupling_radius=0.0,
        compute_triples=True,
    )
    reference = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(tcut_tno=0.0, **common),
    )
    local = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(tcut_tno=1e-8, **common),
    )

    assert reference.converged
    assert local.converged
    assert local.n_triples == reference.n_triples
    assert local.avg_tno < 0.75 * reference.avg_tno
    assert local.e_t == pytest.approx(reference.e_t, abs=1e-9)


def test_weak_pair_screening_preserves_compact_radical(oh_sto3g):
    molecule, basis, uhf, df = oh_sto3g
    common = dict(localise="boys", tcut_pno=0.0, coupling_radius=0.0)
    reference = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(tcut_pairs=0.0, **common),
    )
    screened = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(tcut_pairs=1e-8, **common),
    )

    assert reference.converged
    assert screened.converged
    assert screened.n_screened > 0
    assert screened.n_pairs + screened.n_screened == reference.n_pairs
    assert screened.e_corr == pytest.approx(reference.e_corr, abs=2e-10)
    assert screened.e_screened_mp2 == pytest.approx(
        sum(screened.screened_pair_energies.values()),
        abs=1e-30,
    )
    assert set(screened.pno_per_pair).isdisjoint(
        screened.screened_pair_energies
    )


def test_weak_pair_screening_separated_radical_is_lossless(
    separated_h_h2_sto3g,
):
    molecule, basis, uhf, df = separated_h_h2_sto3g
    common = dict(localise="boys", tcut_pno=0.0, coupling_radius=0.0)
    reference = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(tcut_pairs=0.0, **common),
    )
    screened = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(
            tcut_pairs=1e-4,
            compute_triples=True,
            **common,
        ),
    )
    all_screened = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(tcut_pairs=1.0, **common),
    )

    assert reference.converged
    assert screened.converged
    assert screened.n_pairs == 1
    assert screened.n_screened == 2
    assert screened.n_triples == 0
    assert screened.n_screened_triples == 1
    assert screened.e_t == 0.0
    assert not screened.triples_executed
    assert screened.e_corr == pytest.approx(reference.e_corr, abs=2e-9)
    assert all(
        abs(energy) < 1e-4
        for energy in screened.screened_pair_energies.values()
    )

    assert all_screened.converged
    assert all_screened.n_pairs == 0
    assert all_screened.n_screened == reference.n_pairs
    assert all_screened.avg_pno == 0.0
    assert all_screened.e_corr == pytest.approx(all_screened.e_screened_mp2)


def test_distance_screening_propagates_to_triples(separated_h_h2_sto3g):
    molecule, basis, uhf, df = separated_h_h2_sto3g
    local = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(
            localise="boys",
            tcut_pno=0.0,
            tcut_pairs=0.0,
            coupling_radius=5.0,
            compute_triples=True,
        ),
    )

    assert local.converged
    assert local.n_triples == 0
    assert local.n_screened_triples == 1
    assert local.e_t == 0.0
    assert not local.triples_executed


def test_pao_pair_domains_preserve_separated_radical(separated_oh_h2_sto3g):
    molecule, basis, uhf, df = separated_oh_h2_sto3g
    common = dict(
        localise="boys",
        tcut_pno=0.0,
        tcut_pairs=0.0,
        coupling_radius=0.0,
    )
    reference = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(tcut_mkn=0.0, **common),
    )
    local = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(tcut_mkn=1e-3, **common),
    )

    assert reference.converged
    assert local.converged
    assert local.n_pairs == reference.n_pairs
    assert local.e_corr == pytest.approx(reference.e_corr, abs=2e-9)
    full_virtuals = 2 * basis.nbasis - molecule.n_electrons()
    assert local.avg_pao < 0.8 * full_virtuals
    assert min(local.pao_per_pair.values()) < full_virtuals
    assert set(local.pao_per_pair) == set(local.pno_per_pair)
    assert all(
        local.pno_per_pair[pair] <= n_pao
        for pair, n_pao in local.pao_per_pair.items()
    )


def test_singles_pnos_preserve_separated_radical(separated_oh_h2_sto3g):
    molecule, basis, uhf, df = separated_oh_h2_sto3g
    common = dict(
        localise="boys",
        tcut_pno=0.0,
        tcut_pairs=0.0,
        tcut_mkn=1e-3,
        coupling_radius=0.0,
    )
    reference = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(tcut_pno_singles=0.0, **common),
    )
    local = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(tcut_pno_singles=1e-6, **common),
    )

    assert reference.converged
    assert local.converged
    assert local.e_corr == pytest.approx(reference.e_corr, abs=2e-9)
    full_virtuals = 2 * basis.nbasis - molecule.n_electrons()
    assert reference.avg_singles_domain == pytest.approx(full_virtuals)
    assert local.avg_singles_domain < 0.4 * full_virtuals
    assert set(local.singles_per_occ) == set(reference.singles_per_occ)

    ne = molecule.n_electrons()
    nalpha = (ne + molecule.multiplicity - 1) // 2
    nbeta = ne - nalpha
    nva = basis.nbasis - nalpha
    nvb = basis.nbasis - nbeta
    assert all(
        count <= (nva if occupied < nalpha else nvb)
        for occupied, count in local.singles_per_occ.items()
    )


def test_empty_pao_atom_domain_fails_closed(separated_h_h2_sto3g):
    molecule, basis, uhf, df = separated_h_h2_sto3g
    with pytest.raises(ValueError, match="empty PAO atom domain"):
        run_local_dlpno_uccsd(
            molecule,
            basis,
            uhf,
            df,
            LocalUCCSDOptions(tcut_mkn=2.0),
        )


# --- Fully spin-polarised references (n_beta = 0) ---------------------------


def _spin_polarised(atoms, multiplicity, basis_name="def2-svp"):
    molecule = Molecule(
        [Atom(z, position) for z, position in atoms],
        charge=0,
        multiplicity=multiplicity,
    )
    basis = BasisSet(molecule, basis_name)
    scf_options = UHFOptions()
    scf_options.max_iter = 500
    uhf = run_uhf(molecule, basis, scf_options)
    assert uhf.converged
    df = DensityFitting(
        basis, BasisSet(molecule, AUX), aux_basis_name=AUX
    )
    return molecule, basis, uhf, df


def test_one_electron_reference_has_zero_correlation():
    """The zero-pair limit is a valid input, not an error.

    A one-electron system has no electron pairs, so the only correct
    correlation energy is exactly zero. This used to raise
    ``n_frozen=0 out of range for n_beta=0``: the frozen-core guard used
    ``nf >= nbeta``, which rejects the legitimate ``nbeta == 0`` boundary and
    blamed a parameter the caller never set.
    """
    molecule, basis, uhf, df = _spin_polarised([(1, [0.0, 0.0, 0.0])], 2)
    assert molecule.n_electrons() == 1
    result = run_local_dlpno_uccsd(
        molecule, basis, uhf, df, LocalUCCSDOptions()
    )
    assert result.e_corr == 0.0
    assert result.n_pairs == 0

    pilot = run_dlpno_uccsd_pilot(
        molecule,
        basis,
        uhf,
        df,
        DLPNOUCCSDPilotOptions(compute_triples=True),
    )
    assert pilot.converged
    assert pilot.n_pairs == 0
    assert not pilot.triples_executed
    assert pilot.e_total == pytest.approx(pilot.e_hf)


def test_high_spin_zero_beta_reference_is_exact_against_canonical_uccsd():
    """``n_beta = 0`` is not only the trivial case, and must be right.

    Linear H3 in its quartet state has three electrons, all alpha, and three
    genuine alpha-alpha pairs, so its correlation energy is nonzero. It shares
    the ``n_beta = 0`` boundary with the one-electron case and was rejected by
    the same guard.

    Triplet He deliberately is *not* used here: its occupieds are both ``s``
    and its def2-SVP virtuals are a single ``p`` shell, so every
    antisymmetrised alpha-alpha integral vanishes by symmetry and it returns
    zero for reasons that have nothing to do with this code path.
    """
    from vibeqc.cc import run_uccsd

    bohr = ANGSTROM_TO_BOHR
    atoms = [
        (1, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.0, 1.6 * bohr]),
        (1, [0.0, 0.0, 3.2 * bohr]),
    ]
    molecule, basis, uhf, df = _spin_polarised(atoms, 4)
    assert molecule.n_electrons() == 3

    canonical = run_uccsd(molecule, basis, uhf)
    reference = canonical.e_ccsd_correlation
    assert reference < -1e-4  # the system really does correlate

    local = run_local_dlpno_uccsd(
        molecule,
        basis,
        uhf,
        df,
        LocalUCCSDOptions(
            localise="none", tcut_pno=0.0, tcut_pairs=0.0, tcut_mkn=0.0
        ),
    )
    assert local.converged
    assert local.e_corr == pytest.approx(reference, abs=1e-9)


# --- The screening guard, and the route that retires it --------------------


def test_iterative_t1_matches_the_rotation_when_nothing_is_screened(oh_sto3g):
    """Both `(T1)` routes solve the same equations; only the basis differs.

    ``triples_mode="t1"`` rotates the occupied indices into the per-spin
    canonical basis. ``"t1-iterative"`` keeps them localised and reintroduces
    the off-diagonal occupied Fock coupling by iterating the triples
    amplitudes through a spin-orbital generalization of Guo et al. 2018 Eq.
    (2). Guo et al. 2020 is the dedicated open-shell method-family source.
    With no screening they must agree, and both must differ from the
    semicanonical `(T0)`, or the iteration is not capturing the coupling.
    """
    molecule, basis, uhf, df = oh_sto3g
    common = dict(
        localise="boys", tcut_pno=0.0, tcut_pairs=0.0, compute_triples=True
    )
    rotation = run_local_dlpno_uccsd(
        molecule, basis, uhf, df,
        LocalUCCSDOptions(triples_mode="t1", **common),
    )
    iterative = run_local_dlpno_uccsd(
        molecule, basis, uhf, df,
        LocalUCCSDOptions(triples_mode="t1-iterative", **common),
    )
    semicanonical = run_local_dlpno_uccsd(
        molecule, basis, uhf, df,
        LocalUCCSDOptions(triples_mode="t0", **common),
    )

    assert iterative.e_t == pytest.approx(rotation.e_t, abs=1e-14)
    assert iterative.n_triples == rotation.n_triples
    assert iterative.triples_executed
    # The coupling really is being captured, so this is not a trivial pass.
    assert abs(rotation.e_t - semicanonical.e_t) > 1e-9


def test_iterative_t1_carries_a_screened_triple_list(separated_oh_h2_sto3g):
    """The capability the fail-closed guard was standing in for.

    A distance cutoff on separated OH + H2 screens the cross-fragment triples
    and keeps the intra-fragment ones. The canonical-rotation `(T1)` must still
    refuse, because after the rotation an LMO distance no longer refers to the
    indices being dropped. The localised-basis iteration carries the same list
    the `(T0)` route does, and must produce a real screened `(T1)` energy.
    """
    molecule, basis, uhf, df = separated_oh_h2_sto3g
    common = dict(
        localise="boys",
        tcut_pno=0.0,
        tcut_pairs=0.0,
        compute_triples=True,
        coupling_radius=5.0,
    )

    semicanonical = run_local_dlpno_uccsd(
        molecule, basis, uhf, df,
        LocalUCCSDOptions(triples_mode="t0", **common),
    )
    assert semicanonical.n_screened_triples > 0  # screening really bites
    assert semicanonical.n_triples > 0           # but not everything

    with pytest.raises(NotImplementedError, match="triple screening"):
        run_local_dlpno_uccsd(
            molecule, basis, uhf, df,
            LocalUCCSDOptions(triples_mode="t1", **common),
        )

    iterative = run_local_dlpno_uccsd(
        molecule, basis, uhf, df,
        LocalUCCSDOptions(triples_mode="t1-iterative", **common),
    )
    # Same list as (T0), and a genuine nonzero screened (T1) energy.
    assert iterative.n_triples == semicanonical.n_triples
    assert iterative.n_screened_triples == semicanonical.n_screened_triples
    assert iterative.e_t != 0.0

    # It removes the semicanonical error rather than inheriting it: the
    # screened (T1) sits nearer the unscreened (T1) than the screened (T0) does.
    unscreened_t1 = run_local_dlpno_uccsd(
        molecule, basis, uhf, df,
        LocalUCCSDOptions(
            triples_mode="t1",
            localise="boys",
            tcut_pno=0.0,
            tcut_pairs=0.0,
            compute_triples=True,
            coupling_radius=0.0,
        ),
    )
    assert abs(iterative.e_t - unscreened_t1.e_t) < abs(
        semicanonical.e_t - unscreened_t1.e_t
    )


def test_unknown_triples_mode_still_rejected(oh_sto3g):
    molecule, basis, uhf, df = oh_sto3g
    with pytest.raises(ValueError, match="unknown triples_mode"):
        run_local_dlpno_uccsd(
            molecule, basis, uhf, df,
            LocalUCCSDOptions(triples_mode="t2-iterative"),
        )
