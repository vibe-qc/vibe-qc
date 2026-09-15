"""Regression coverage for the molecular SCF oscillation detector."""

from __future__ import annotations

import pytest

from vibeqc import Atom, BasisSet, Molecule, RKSOptions, run_rks
from vibeqc import _vibeqc_core as core


ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def _trace(
    rows: list[tuple[float, float]],
    diis_subspaces: list[int] | None = None,
) -> list[core.SCFIteration]:
    subspaces = diis_subspaces or [0] * len(rows)
    return [
        core.SCFIteration(
            iter=index,
            delta_e=delta_e,
            grad_norm=grad_norm,
            diis_subspace=subspaces[index - 1],
        )
        for index, (delta_e, grad_norm) in enumerate(rows, start=1)
    ]


def _detects(
    rows: list[tuple[float, float]],
    *,
    conv_tol_energy: float = 1.0e-8,
    diis_subspaces: list[int] | None = None,
) -> bool:
    return core._detect_scf_oscillation_for_testing(
        _trace(rows, diis_subspaces),
        window=10,
        min_sign_flips=5,
        conv_tol_energy=conv_tol_energy,
    )


def test_archived_pyridazine_contracting_tail_is_not_oscillation() -> None:
    """The exact v0.15.47 qcil_03101 prefix must retain DIIS."""
    rows = [
        (0.0, 1.453422901716761),
        (-0.06610488918352075, 0.8244199208938977),
        (0.001294998749642673, 0.8702527717465095),
        (-0.01607399694620426, 0.11300364488670293),
        (0.00036033462697560026, 0.15096572730114485),
        (-0.0008106837193508909, 0.004633999013158597),
        (-7.115374955901643e-7, 0.0020591959777994085),
        (-1.608622142157401e-7, 0.0002865952413802657),
        (2.0360007511044387e-6, 0.00011608373728338919),
        (-5.5161422096716706e-9, 0.0000638108700146208),
    ]

    assert not _detects(rows)


def test_archived_thymine_contracting_oscillations_do_not_trigger() -> None:
    """Every tested prefix of qcil_04032 still makes same-phase progress.

    Its order-one early energy alternation proves that an absolute amplitude
    gate alone cannot distinguish a healthy trajectory. The later prefix also
    defeats endpoint and half-window trend checks while its period-two
    commutator envelopes are still contracting.
    """
    rows = [
        (0.0, 3.4915050908985066),
        (16.709922238607305, 14.57203960761494),
        (-16.851635983034612, 3.5699740217561793),
        (0.9799441726472651, 4.8031536552900755),
        (-1.1201230051080984, 2.2117030140066958),
        (-0.3354409967146239, 1.3246046646667482),
        (-0.07131685736203508, 0.9472480260332491),
        (-0.025052447361758823, 0.7294252828951213),
        (0.3674381857508706, 2.007556188565799),
        (-0.39625360653940334, 0.5052052690274157),
        (0.40957779226528146, 2.067034150643915),
        (-0.42223201796923604, 0.2785011628734186),
        (0.4748995769427893, 2.231269449519239),
        (-0.0010234729820695065, 2.236514259690979),
        (0.00423815871727129, 2.2325542162595484),
        (-0.002213170684626675, 2.248013221350225),
        (-0.021450176064490734, 2.18215522613067),
        (-0.005835456759996305, 2.161163558125005),
        (-0.005038781486518928, 2.147788882478648),
        (0.010004079866803295, 2.1820775984981076),
        (0.38305683226008114, 3.320814873264365),
        (-0.5054406709271007, 2.094276242074813),
        (-0.12401753887547784, 1.651968792292829),
        (-0.05749573886646431, 1.3841936308582863),
        (-0.08577160556956187, 0.9046741203159964),
        (0.3340477604451735, 1.958693059816022),
        (-0.36468098020259276, 0.6592867568706001),
        (0.3771380147622949, 2.0526457255050303),
        (0.020927728544847923, 2.09452849199985),
        (-0.004715908209391273, 2.113056652717442),
        (0.010806902237391114, 2.132637741009359),
        (-0.01064788270241479, 2.1096087336530482),
        (-0.3889824741922894, 0.7165401823539027),
        (0.003698469741948429, 0.764501653217012),
        (-0.055627230769914604, 0.10673423339098662),
        (0.0033022534441897733, 0.27630660014700303),
        (0.0037081722623497626, 0.35537430131062564),
        (0.0009200681249694753, 0.37157499364793534),
        (0.0004632043846868328, 0.37965577786492677),
        (0.0015028006268948957, 0.3955880649113951),
        (-0.004416732857407624, 0.32322734411188075),
        (-0.0008119806232116389, 0.3029996089678805),
        (-0.003989260039588771, 0.19523845531081518),
        (-0.002629194766313958, 0.02669519116222121),
        (-6.024453978170641e-05, 0.01011487270667862),
        (-1.548160571474e-05, 0.0068232519690253415),
        (-1.2850098300987156e-05, 0.007603574720325138),
        (-8.112541308946675e-06, 0.004801683153679052),
        (-1.3787578836854664e-05, 0.0066474372171371615),
        (-5.19246577823651e-06, 0.002410526338770225),
        (-3.320196810818743e-07, 0.001956445265033754),
        (-1.403523128828965e-07, 0.001410258105221891),
        (2.0337301975814626e-07, 0.0014795097748091584),
        (-3.512973307806533e-07, 0.0005010809698266937),
        (-1.127000359701924e-08, 0.00025829653878893635),
        (-6.460140866693109e-09, 0.00012669446429487647),
        (-1.375610736431554e-09, 4.676974057358747e-05),
        (-1.8189894035458565e-10, 1.970660412771606e-05),
        (-5.184119800105691e-11, 1.1665541553772293e-05),
        (-2.3419488570652902e-11, 5.726645329491182e-06),
        (-1.5916157281026244e-12, 1.9385109118212468e-06),
        (9.549694368615746e-12, 1.0164292047700031e-06),
        (-2.5011104298755527e-12, 2.830114743378663e-07),
    ]

    assert all(not _detects(rows[:end]) for end in range(10, len(rows) + 1))


def test_tolerance_scaled_microcycle_is_not_oscillation() -> None:
    """Alternation inside two orders of the energy tolerance is noise."""
    rows = [
        ((-1.0 if index % 2 else 1.0) * 5.0e-7, 0.1 + 0.01 * index)
        for index in range(10)
    ]

    assert not _detects(rows, conv_tol_energy=1.0e-8)


def test_sustained_period_two_limit_cycle_still_triggers() -> None:
    """The measured H2O/LDA no-aids oscillator remains a positive control."""
    rows = [
        (-0.0007248967027635445, 0.3888328279092704),
        (-0.00045414765546070157, 0.38132299437385764),
        (0.002420224970094864, 0.40389798356779355),
        (-0.0004912576604851893, 0.3911577144510457),
        (0.002520980798735195, 0.41400880383287625),
        (-0.0006076440428159913, 0.4001921543275079),
        (0.002619940555277367, 0.42370532513222536),
        (-0.0007307207493312262, 0.40888776343837874),
        (0.0027147011399222265, 0.43304432673737014),
        (-0.0008591438382126171, 0.4172321075326548),
    ]

    assert _detects(rows, conv_tol_energy=1.0e-7)


def test_diis_restart_splits_the_detection_window() -> None:
    """The qcil_02626 iteration-27 restart starts a new detector epoch."""
    rows = [
        ((-1.0 if index % 2 else 1.0) * 1.0e-3, 0.1 + 0.01 * index)
        for index in range(10)
    ]
    diis_subspaces = [4, 5, 6, 7, 8, 1, 2, 3, 4, 5]

    assert not _detects(rows, diis_subspaces=diis_subspaces)


def _run_archived_b3lyp_case(
    geometry: list[tuple[int, tuple[float, float, float]]],
    basis_name: str,
):
    molecule = Molecule(
        [
            Atom(
                atomic_number,
                [coordinate * ANGSTROM_TO_BOHR for coordinate in xyz],
            )
            for atomic_number, xyz in geometry
        ],
        multiplicity=1,
    )
    basis = BasisSet(molecule, basis_name)
    options = RKSOptions()
    options.functional = "b3lyp"
    options.use_diis = True
    options.max_iter = 120
    options.conv_tol_energy = 1.0e-8
    return run_rks(molecule, basis, options)


@pytest.mark.slow
@pytest.mark.timeout(1200)
def test_qcil_02626_norbornadiene_retains_archived_convergence() -> None:
    """The restart-aware detector restores bounded qcil_02626 convergence."""
    geometry = [
        (6, (0.000, 0.000, 0.752)),
        (6, (0.000, 1.176, -0.116)),
        (6, (0.000, -1.176, -0.116)),
        (6, (1.247, 0.000, -0.569)),
        (6, (-1.247, 0.000, -0.569)),
        (6, (1.248, 0.000, 1.468)),
        (6, (-1.248, 0.000, 1.468)),
        (1, (0.000, -2.133, 0.353)),
        (1, (1.852, 0.906, -0.550)),
        (1, (-1.852, -0.906, -0.550)),
        (1, (2.098, 0.000, 2.128)),
        (1, (-2.098, 0.000, 2.128)),
        (1, (0.000, 0.931, 2.424)),
        (1, (0.000, -0.931, 2.424)),
        (1, (0.000, 2.133, 0.353)),
        (1, (1.852, -0.906, -0.550)),
        (1, (-1.852, 0.906, -0.550)),
    ]

    result = _run_archived_b3lyp_case(geometry, "cc-pvtz")

    assert result.converged
    # v0.15.47 converged in 23 before incremental Fock became default-on.
    # Current main's IID 129 coarse/fine transition rebuilds DIIS and makes
    # this 37 iterations; keep it bounded well below the broken >=120 path.
    assert result.n_iter <= 45
    assert result.energy == pytest.approx(-271.6200511474, abs=2.0e-7)


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_qcil_04032_thymine_retains_archived_convergence_and_basin() -> None:
    """Large contracting swings keep DIIS and reach qcil_04032's basin."""
    geometry = [
        (7, (0.000, 1.227, 0.000)),
        (6, (1.089, 0.484, 0.000)),
        (6, (-1.089, 0.484, 0.000)),
        (7, (0.694, -0.831, 0.000)),
        (6, (-0.694, -0.831, 0.000)),
        (6, (1.709, -1.793, 0.000)),
        (8, (1.430, -2.994, 0.000)),
        (6, (-1.250, -1.946, 0.000)),
        (8, (-2.441, -1.661, 0.000)),
        (6, (-1.495, 0.964, 0.000)),
        (1, (2.089, 0.915, 0.000)),
        (1, (2.748, -1.461, 0.000)),
        (1, (-2.533, 0.555, 0.000)),
        (1, (-1.440, 2.049, 0.000)),
        (1, (-1.518, 0.518, 0.000)),
        (1, (-0.019, 2.229, 0.000)),
    ]

    result = _run_archived_b3lyp_case(geometry, "cc-pvdz")

    assert result.converged
    assert result.n_iter <= 75
    # Current main's grid/direct-SCF stack differs from v0.15.47 by 2.25 uHa
    # on this deck. This bound is still over three orders of magnitude tighter
    # than the 9.49 mHa wrong basin reached by the broken shifted replay.
    assert result.energy == pytest.approx(-485.9085228454, abs=5.0e-6)
