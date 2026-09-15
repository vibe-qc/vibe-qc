"""Regression: RKS/UKS BIPOLE SCF convergence robustness.

Gap B from handovers/HANDOVER_BIPOLE_PRODUCTION.md: RKS BIPOLE
historically oscillated ~0.5 Ha on MgO primitive. The 2026-07-13
Gap-B validation (Sec. 0a of that handover) closed the question:

* The oscillation does not exist in the current corrected-gauge code.
  Plain DIIS converges the MgO fixture monotonically in ~10 iterations
  with NO aids (pinned below); PySCF KRKS converges the same fixture
  in 7 cycles, so the fixture itself is benign.
* CRYSTAL-style FMIXING 30% is validated as a *Roothaan* damper (the
  DIIS-off iteration wobbles at ~1 mHa and stalls; FMIXING-30
  converges it to the same fixed point). Under DIIS it is redundant
  and only slows convergence, so the auto default now fires only when
  ``use_diis=False`` (``resolve_auto_fock_mixing``).
* Smearing is NOT needed for convergence here; it remains a
  near-degeneracy tool (see smearing_basin_warning).
"""

from __future__ import annotations

import numpy as np
import vibeqc as vq
from vibeqc._vibeqc_core import (
    InitialGuess,
    PeriodicKSOptions,
    PeriodicRHFOptions,
    SpinlockMode,
    monkhorst_pack,
)


def _h2_3d(box: float = 8.0):
    """H₂ in a cubic box."""
    lattice = np.eye(3) * box
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _ks_opts(
    functional: str = "svwn",
    max_iter: int = 20,
    smearing_T: float = 0.01,
) -> PeriodicKSOptions:
    opts = PeriodicKSOptions()
    opts.functional = functional
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.max_iter = max_iter
    opts.initial_guess = InitialGuess.HCORE
    opts.use_diis = True
    opts.diis_start_iter = 1
    opts.smearing_temperature = smearing_T
    return opts


def test_rks_bipole_converges_with_smearing():
    """RKS BIPOLE converges with smearing (plain DIIS, no auto FMIXING)."""
    system, basis = _h2_3d()
    kmesh = monkhorst_pack(system, [2, 1, 1])

    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    result = run_pbc_bipole_rks(
        system,
        basis,
        kmesh,
        _ks_opts(),
        progress=False,
        ewald_precision=1e-6,
    )
    assert result.converged, f"RKS BIPOLE did not converge (n_iter={result.n_iter})"
    assert result.n_iter >= 2, "Expected ≥2 SCF iterations"
    # H₂ ground state ~ -1.1 Ha.
    assert -1.5 < result.energy < -0.5, f"Unphysical energy: {result.energy}"
    # Smearing diagnostics must be non-zero.
    assert result.smearing_temperature > 0.0
    assert abs(result.entropy) > 0.0
    assert result.fock_mixing == 0.0


def test_uks_bipole_converges_with_smearing():
    """UKS BIPOLE converges with smearing (plain DIIS, no auto FMIXING)."""
    lattice = np.eye(3) * 8.0
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [4.0, 4.0, 4.0])],
        charge=0,
        multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])

    from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

    result = run_pbc_bipole_uks(
        system,
        basis,
        kmesh,
        _ks_opts(max_iter=10),
        progress=False,
        ewald_precision=1e-6,
    )
    assert result.converged, f"UKS BIPOLE did not converge (n_iter={result.n_iter})"
    assert result.smearing_temperature > 0.0
    assert result.free_energy == pytest.approx(
        result.energy - result.smearing_temperature * result.entropy,
        abs=1e-12,
    )
    assert result.fock_mixing == 0.0


def test_rks_bipole_converges_without_smearing():
    """RKS BIPOLE converges without smearing (plain DIIS, no aids)."""
    system, basis = _h2_3d()
    kmesh = monkhorst_pack(system, [1, 1, 1])

    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    result = run_pbc_bipole_rks(
        system,
        basis,
        kmesh,
        _ks_opts(smearing_T=0.0, max_iter=30),
        progress=False,
        ewald_precision=1e-6,
    )
    assert result.converged, (
        f"RKS BIPOLE (no smearing) did not converge (n_iter={result.n_iter})"
    )
    assert -1.5 < result.energy < -0.5


def test_rks_bipole_fermi_level_physically_reasonable():
    """Fermi level with smearing is between HOMO and LUMO."""
    system, basis = _h2_3d()
    kmesh = monkhorst_pack(system, [1, 1, 1])

    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    result = run_pbc_bipole_rks(
        system,
        basis,
        kmesh,
        _ks_opts(max_iter=20),
        progress=False,
        ewald_precision=1e-6,
    )
    assert result.converged
    # Fermi level must be within the HOMO-LUMO gap band.
    mo_e = np.sort(np.concatenate([np.asarray(e) for e in result.mo_energies]))
    n_occ = system.n_electrons() // 2
    e_homo = mo_e[n_occ - 1]
    e_lumo = mo_e[n_occ]
    fermi = result.fermi_level
    assert e_homo <= fermi <= e_lumo + 0.1, (
        f"Fermi level {fermi:.4f} outside [{e_homo:.4f}, {e_lumo:.4f}]"
    )


# ---------------------------------------------------------------------------
# Auto-FMIXING default resolution (Gap-B validated, 2026-07-13): fires
# only for DFT functionals when DIIS is off; explicit values honored.
# ---------------------------------------------------------------------------
import pytest  # noqa: E402

from vibeqc.pbc_bipole_common import (  # noqa: E402
    resolve_auto_fock_mixing,
    resolve_fock_mixing,
)


def test_fock_mixing_keyword_override_precedence_does_not_rewrite_options():
    """The optional direct-driver keyword wins without mutating options.

    ``None`` remains the sentinel for "read the options object".  An explicit
    zero is therefore a real override, not another spelling of the sentinel.
    """

    opts = PeriodicKSOptions()
    opts.fock_mixing = 0.10

    assert resolve_fock_mixing(opts, None, where="test") == pytest.approx(0.10)
    assert resolve_fock_mixing(opts, 0.25, where="test") == pytest.approx(0.25)
    assert resolve_fock_mixing(opts, 0.0, where="test") == 0.0
    assert opts.fock_mixing == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("options_value", "override"),
    [
        (-0.10, None),
        (1.0, None),
        (float("nan"), None),
        (0.10, -0.10),
        (0.10, 1.0),
        (0.10, float("nan")),
    ],
)
def test_fock_mixing_resolution_rejects_invalid_selected_source(
    options_value,
    override,
):
    """Range validation follows the selected source, including NaN."""

    opts = PeriodicKSOptions()
    opts.fock_mixing = options_value
    with pytest.raises(ValueError, match="fock_mixing must be in"):
        resolve_fock_mixing(opts, override, where="test")


def test_auto_fock_mixing_silent_under_diis():
    """Pure DFT + DIIS: no auto FMIXING (redundant damping, measured
    +4 iterations on MgO RKS [2,2,2]/c6 for an identical fixed point)."""
    assert resolve_auto_fock_mixing(
        0.0, alpha_hf=0.0, use_diis=True, where="t"
    ) == 0.0


def test_auto_fock_mixing_fires_without_diis():
    """Pure DFT without DIIS: CRYSTAL-style 30% damps the bare Roothaan
    iteration (validated: MgO converges in 18 iterations with it, wobbles
    unconverged without)."""
    assert resolve_auto_fock_mixing(
        0.0, alpha_hf=0.0, use_diis=False, where="t"
    ) == pytest.approx(0.30)


def test_auto_fock_mixing_never_for_hf():
    assert resolve_auto_fock_mixing(
        0.0, alpha_hf=1.0, use_diis=False, where="t"
    ) == 0.0


def test_auto_fock_mixing_explicit_honored():
    assert resolve_auto_fock_mixing(
        0.25, alpha_hf=0.0, use_diis=True, where="t"
    ) == pytest.approx(0.25)
    # The epsilon sentinel (used by the FD-gradient tests to force
    # mixing off before this default existed) stays honored.
    assert resolve_auto_fock_mixing(
        1e-12, alpha_hf=0.0, use_diis=False, where="t"
    ) == pytest.approx(1e-12)


def test_auto_fock_mixing_range_validation():
    with pytest.raises(ValueError, match="fock_mixing must be in"):
        resolve_auto_fock_mixing(1.0, alpha_hf=0.0, use_diis=True, where="t")
    with pytest.raises(ValueError, match="fock_mixing must be in"):
        resolve_auto_fock_mixing(-0.1, alpha_hf=0.0, use_diis=True, where="t")


@pytest.mark.parametrize("spin", ("restricted", "unrestricted"))
def test_bipole_ks_result_records_auto_fock_mixing_without_diis(spin):
    """The public result preserves the value that drove the SCF loop."""

    if spin == "restricted":
        system, basis = _h2_3d()
        driver = vq.run_pbc_bipole_rks
    else:
        system = vq.PeriodicSystem(
            3,
            np.eye(3) * 8.0,
            [vq.Atom(1, [4.0, 4.0, 4.0])],
            charge=0,
            multiplicity=2,
        )
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        driver = vq.run_pbc_bipole_uks
    kmesh = monkhorst_pack(system, [1, 1, 1])
    opts = _ks_opts(max_iter=2, smearing_T=0.0)
    opts.use_diis = False
    opts.conv_tol_energy = 0.0
    opts.conv_tol_grad = 0.0

    result = driver(
        system,
        basis,
        kmesh,
        opts,
        progress=False,
        ewald_precision=1e-6,
    )

    assert opts.fock_mixing == 0.0
    assert result.n_iter == 2
    assert not result.converged
    assert result.fock_mixing == pytest.approx(0.30)


@pytest.mark.parametrize("spin", ("restricted", "unrestricted"))
def test_bipole_ks_keyword_fock_mixing_overrides_options_in_real_loop(spin):
    """A direct keyword drives the loop and result without rewriting options."""

    if spin == "restricted":
        system, basis = _h2_3d()
        driver = vq.run_pbc_bipole_rks
    else:
        system = vq.PeriodicSystem(
            3,
            np.eye(3) * 8.0,
            [vq.Atom(1, [4.0, 4.0, 4.0])],
            charge=0,
            multiplicity=2,
        )
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        driver = vq.run_pbc_bipole_uks
    kmesh = monkhorst_pack(system, [1, 1, 1])
    opts = _ks_opts(max_iter=2, smearing_T=0.0)
    opts.use_diis = False
    opts.conv_tol_energy = 0.0
    opts.conv_tol_grad = 0.0
    opts.fock_mixing = 0.10

    result = driver(
        system,
        basis,
        kmesh,
        opts,
        fock_mixing=0.25,
        progress=False,
        ewald_precision=1e-6,
    )

    assert opts.fock_mixing == pytest.approx(0.10)
    assert result.n_iter == 2
    assert not result.converged
    assert result.fock_mixing == pytest.approx(0.25)


@pytest.mark.parametrize("spin", ("restricted", "unrestricted"))
def test_bipole_hf_keyword_fock_mixing_overrides_options_in_real_loop(spin):
    """The RHF/UHF loops share the same non-mutating override contract."""

    if spin == "restricted":
        system, basis = _h2_3d()
        driver = vq.run_pbc_bipole_rhf
    else:
        system = vq.PeriodicSystem(
            3,
            np.eye(3) * 8.0,
            [vq.Atom(1, [4.0, 4.0, 4.0])],
            charge=0,
            multiplicity=2,
        )
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        driver = vq.run_pbc_bipole_uhf
    kmesh = monkhorst_pack(system, [1, 1, 1])
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.max_iter = 2
    opts.initial_guess = InitialGuess.HCORE
    opts.use_diis = False
    opts.conv_tol_energy = 0.0
    opts.conv_tol_grad = 0.0
    opts.fock_mixing = 0.10

    result = driver(
        system,
        basis,
        kmesh,
        opts,
        fock_mixing=0.25,
        progress=False,
        ewald_precision=1e-6,
    )

    assert opts.fock_mixing == pytest.approx(0.10)
    assert result.n_iter == 2
    assert not result.converged
    assert result.fock_mixing == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("module_name", "driver_name", "options_cls", "functional"),
    [
        (
            "vibeqc.pbc_bipole_uhf",
            "run_pbc_bipole_uhf",
            PeriodicRHFOptions,
            None,
        ),
        (
            "vibeqc.pbc_bipole_uks",
            "run_pbc_bipole_uks",
            PeriodicKSOptions,
            "svwn",
        ),
    ],
)
def test_bipole_spin_schedule_forwards_fock_mixing_override_to_both_phases(
    monkeypatch,
    module_name,
    driver_name,
    options_cls,
    functional,
):
    """The unrestricted two-phase closure must not drop the keyword."""

    import importlib

    module = importlib.import_module(module_name)
    driver = getattr(module, driver_name)
    spinlock = importlib.import_module("vibeqc.spinlock_periodic")
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [vq.Atom(1, [4.0, 4.0, 4.0])],
        charge=0,
        multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])
    opts = options_cls()
    opts.spinlock_mode = SpinlockMode.SPIN_SCHEDULE
    opts.spinlock_iterations = 1
    opts.fock_mixing = 0.10
    seen = []
    sentinel = object()

    def fake_recursive(*_args, **kwargs):
        seen.append(kwargs.get("fock_mixing"))
        return sentinel

    def fake_schedule(phase_runner, scheduled_system, scheduled_opts):
        monkeypatch.setattr(module, driver_name, fake_recursive)
        phase_runner(scheduled_system, scheduled_opts)
        return phase_runner(scheduled_system, scheduled_opts)

    monkeypatch.setattr(spinlock, "run_spin_schedule", fake_schedule)
    kwargs = {"fock_mixing": 0.25}
    if functional is not None:
        kwargs["functional"] = functional

    result = driver(system, basis, kmesh, opts, **kwargs)

    assert result is sentinel
    assert seen == pytest.approx([0.25, 0.25])
    assert opts.fock_mixing == pytest.approx(0.10)


@pytest.mark.parametrize(
    ("module_name", "driver_name", "options_cls"),
    [
        (
            "vibeqc.pbc_bipole_uhf",
            "run_pbc_bipole_uhf",
            PeriodicRHFOptions,
        ),
        (
            "vibeqc.pbc_bipole_uks",
            "run_pbc_bipole_uks",
            PeriodicKSOptions,
        ),
    ],
)
def test_bipole_spin_schedule_forwards_exact_zone_to_both_phases(
    monkeypatch,
    module_name,
    driver_name,
    options_cls,
):
    """The exact-zone cutoff must survive both recursive phase calls."""

    import importlib

    module = importlib.import_module(module_name)
    driver = getattr(module, driver_name)
    spinlock = importlib.import_module("vibeqc.spinlock_periodic")
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [vq.Atom(1, [4.0, 4.0, 4.0])],
        charge=0,
        multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])
    opts = options_cls()
    opts.spinlock_mode = SpinlockMode.SPIN_SCHEDULE
    opts.spinlock_iterations = 1
    seen = []
    sentinel = object()

    def fake_recursive(*_args, **kwargs):
        seen.append(kwargs.get("exact_zone_bohr"))
        return sentinel

    def fake_schedule(phase_runner, scheduled_system, scheduled_opts):
        monkeypatch.setattr(module, driver_name, fake_recursive)
        phase_runner(scheduled_system, scheduled_opts)
        return phase_runner(scheduled_system, scheduled_opts)

    monkeypatch.setattr(spinlock, "run_spin_schedule", fake_schedule)

    result = driver(
        system,
        basis,
        kmesh,
        opts,
        exact_zone_bohr=9.25,
    )

    assert result is sentinel
    assert seen == pytest.approx([9.25, 9.25])


# ---------------------------------------------------------------------------
# The actual Gap B case: MgO primitive (tight ionic cell), multi-k RKS.
# The four tests above use H2 boxes — soft covalent cells that converge
# easily; they never exercised the system the Gap B handover is about.
# ---------------------------------------------------------------------------


def _mgo_fixture():
    """MgO primitive rocksalt, a = 4.21 Å, STO-3G — the Gap-B cell."""
    ANG2BOHR = 1.0 / 0.529177210903
    a = 4.21 * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(12, [0.0, 0.0, 0.0]), vq.Atom(8, [a / 2.0, a / 2.0, a / 2.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


@pytest.mark.slow
def test_rks_bipole_mgo_no_aids_converges_to_stationary_state():
    """Gap-B symptom regression (CLAUDE.md §7): the historical ~0.5 Ha
    MgO RKS oscillation must stay gone WITHOUT any convergence aid.

    Plain DIIS, no FMIXING (the auto default is silent under DIIS), no
    smearing, no level shift. Originally validated 2026-07-13 on the
    exact-FT route (strictly monotone −268.33 → −270.9199 over 10
    iterations); re-validated 2026-07-18 on the production padded SR+LR
    route (12 iterations to −269.3261, integer occupations 2×10).
    PySCF KRKS converges the same fixture in 7 cycles — the fixture is
    benign; the old oscillation belonged to the legacy gauge +
    pre-v0.14 cross-cell XC bug era. A reappearing SUSTAINED
    oscillation is a BUG per CLAUDE.md §7 — do not fix this test by
    adding damping.

    The maintained fixture uses the measured 12-bohr support (S(k)-fold
    drift 5.0e-3; cutoff 6 is refused at 4.1e-2).  This is the same
    production cutoff used by the parity driver and keeps this test on a
    supported SCF basin rather than pinning the old c6 truncated fixed point.

    The assertions below pin the exact terminal stationarity contract,
    iteration count, supported energy window, and integer occupations.
    DIIS can transiently raise the energy: the current ten-iteration trace
    rises by 0.06172 Ha at iteration 3 before settling to the stationary
    state. A sustained oscillation cannot pass the terminal energy and
    commutator tolerances within the iteration budget.
    """
    system, basis = _mgo_fixture()
    kmesh = monkhorst_pack(system, [2, 2, 2])

    opts = PeriodicKSOptions()
    opts.functional = "svwn"
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 12.0
    opts.max_iter = 16
    opts.use_diis = True
    opts.conv_tol_energy = 1e-8
    opts.initial_guess = InitialGuess.SAD
    opts.smearing_temperature = 0.0
    # fock_mixing left at 0.0 -> auto default resolves to OFF (DIIS on)

    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    result = run_pbc_bipole_rks(
        system,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
        progress=False,
    )
    assert result.converged, (
        f"MgO RKS (no aids) did not converge (n_iter={result.n_iter})"
    )
    assert result.n_iter <= 14, (
        f"MgO RKS (no aids) took {result.n_iter} iterations (expected ~12)"
    )
    assert -270.51 < result.energy < -270.49, (
        f"MgO RKS energy {result.energy:.6f} outside the cutoff-12 "
        f"fixed-point window (measured -270.504403, 2026-08-25)"
    )
    assert result.fock_mixing == 0.0
    terminal = result.scf_trace[-1]
    assert abs(terminal.delta_e) < opts.conv_tol_energy
    assert terminal.grad_norm < opts.conv_tol_grad
    assert terminal.energy == pytest.approx(result.energy, rel=0.0, abs=1e-12)
    # Integer occupations: T=0 Aufbau on a genuine insulator.
    occ = np.concatenate([np.asarray(o) for o in result.occupations]) \
        if result.occupations else np.array([])
    if occ.size:
        assert np.allclose(occ, np.round(occ), atol=1e-12)


@pytest.mark.slow
def test_rks_bipole_mgo_tight_ionic_converges_with_smearing():
    """The corrected route refuses the historically under-converged c6
    MgO smearing fixture before entering SCF.

    The old convergence pin used a cutoff whose overlap-fold drift is
    4.1e-2.  That numerical-support domain is now explicitly outside the
    production envelope; preserving its historical fixed point would weaken
    the fail-closed guard.  The converged finite-temperature route is covered
    on compact fixtures above, while this test pins the MgO safety boundary.
    """
    system, basis = _mgo_fixture()
    kmesh = monkhorst_pack(system, [2, 2, 2])

    opts = PeriodicKSOptions()
    opts.functional = "svwn"
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.lattice_opts.nuclear_cutoff_bohr = 6.0
    opts.max_iter = 40
    opts.use_diis = True
    opts.conv_tol_energy = 1e-8
    opts.initial_guess = InitialGuess.SAD
    opts.smearing_temperature = 0.01
    # fock_mixing left at 0.0 -> auto default resolves to OFF (DIIS on)

    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    from vibeqc.pbc_bipole_common import BipoleFoldUnreliableError

    with pytest.raises(BipoleFoldUnreliableError, match="fold drift"):
        run_pbc_bipole_rks(
            system,
            basis,
            kmesh,
            opts,
            use_ewald_j_split=True,
            ewald_precision=1e-8,
            progress=False,
        )


def test_band_overlap_t0_mesh_uses_the_occupation_driven_density():
    """#509: the BIPOLE RKS route used to fail closed on a band-overlap
    T = 0 mesh (the fixed C[:, :n_occ] slice cannot represent the global
    fill's per-k varying counts). It must now run with the occupation-driven
    builder: no fixed-subspace refusal, and the carried occupations are the
    global fill, not the legacy per-k pattern. The legacy gauge still fails
    closed at runtime on the same mesh."""
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks
    from vibeqc.smearing import (
        occupations_are_per_k_integer_aufbau as _occupations_are_per_k_integer_aufbau,
    )

    system = vq.PeriodicSystem(3, 5.0 * np.eye(3), [vq.Atom(12, [0, 0, 0])], 0, 1)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = vq.monkhorst_pack(system, [2, 2, 2])
    opts = vq.PeriodicKSOptions()
    opts.functional = "lda"
    opts.max_iter = 1  # the builder contract is exercised at the first fold
    opts.conv_tol_energy = 1e-8
    opts.conv_tol_grad = 1e-5
    opts.use_diis = True
    r = run_pbc_bipole_rks(
        system, basis, kmesh, opts, functional="lda",
        use_ewald_j_split=True, ewald_precision=1e-8, progress=False,
    )
    assert r.n_iter >= 1
    assert np.isfinite(r.energy)
    assert not _occupations_are_per_k_integer_aufbau(r.occupations, 6), (
        "the carried occupations collapsed to the per-k integer pattern; "
        "the band-overlap fill was not exercised"
    )

    # Runtime fail-closed: the legacy gauge cannot consume fractional
    # occupations, and the same mesh discovers them at the first fold.
    with pytest.raises(
        NotImplementedError, match="fractional occupations require"
    ):
        run_pbc_bipole_rks(
            system, basis, kmesh, opts, functional="lda",
            use_ewald_j_split=True, use_exchange_ewald_split=False,
            ewald_precision=1e-8, progress=False,
        )
