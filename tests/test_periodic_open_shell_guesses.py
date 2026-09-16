"""Open-shell periodic SAP/HUECKEL/MINAO initial-guess wiring."""
from __future__ import annotations

from types import SimpleNamespace

import importlib

import numpy as np
import pytest
from vibeqc.pbc_bipole_common import bvk_torus_density_matrices

import vibeqc as vq
import vibeqc.guess as periodic_guess
import vibeqc.pbc_gdf as pbc_gdf
import vibeqc.pbc_bipole_uhf as pbc_bipole_uhf
import vibeqc.pbc_bipole_uks as pbc_bipole_uks
import vibeqc.periodic_k_gdf as periodic_k_gdf
import vibeqc.periodic_rijcosx as periodic_rijcosx
import vibeqc.periodic_rohf_ewald as periodic_rohf_ewald
import vibeqc.periodic_rohf_multi_k_ewald as periodic_rohf_multi_k_ewald
import vibeqc.periodic_roks_multi_k_ewald as periodic_roks_multi_k_ewald
import vibeqc.periodic_uhf_multi_k_ewald as periodic_uhf_multi_k_ewald
import vibeqc.periodic_uks_multi_k_ewald as periodic_uks_multi_k_ewald
from vibeqc._vibeqc_core import compute_overlap_lattice
from vibeqc.guess import initial_densities_open_shell


_OPEN_SHELL_PERIODIC_HELPER_GUESSES = [
    vq.InitialGuess.SAP,
    vq.InitialGuess.HUECKEL,
    vq.InitialGuess.MINAO,
]

_OPEN_SHELL_PERIODIC_DRIVER_GUESSES = [
    *_OPEN_SHELL_PERIODIC_HELPER_GUESSES,
    vq.InitialGuess.PATOM,
]


def _h_box():
    system = vq.PeriodicSystem(
        3,
        10.0 * np.eye(3),
        [vq.Atom(1, [0.0, 0.0, 0.0])],
        charge=0,
        multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _gamma(lat_set) -> np.ndarray:
    mat = np.zeros_like(np.asarray(lat_set.blocks[0], dtype=float))
    for block in lat_set.blocks:
        mat += np.asarray(block, dtype=float)
    return 0.5 * (mat + mat.T)


def _rhf_opts(guess):
    opts = vq.PeriodicRHFOptions()
    opts.initial_guess = guess
    opts.conv_tol_energy = 1e-8
    opts.max_iter = 60
    return opts


def _uks_opts(guess):
    opts = vq.PeriodicKSOptions()
    opts.initial_guess = guess
    opts.functional = "lda"
    opts.conv_tol_energy = 1e-8
    opts.max_iter = 60
    return opts


def _assert_guess_matches_sad(label, energy):
    e_sad = energy(vq.InitialGuess.SAD)
    for guess in _OPEN_SHELL_PERIODIC_DRIVER_GUESSES:
        e_guess = energy(guess)
        assert e_guess == pytest.approx(e_sad, abs=1e-7), (
            f"{label}/{guess.name}: {e_guess} vs SAD {e_sad}"
        )


@pytest.mark.parametrize(
    ("driver", "functional"),
    [
        (periodic_k_gdf.run_kuhf_periodic_gdf, None),
        (periodic_k_gdf.run_kuks_periodic_gdf, "lda"),
    ],
    ids=["kuhf", "kuks"],
)
def test_direct_multik_gdf_sap_rejects_atomic_spins_before_setup(
    monkeypatch,
    driver,
    functional,
):
    """Direct GDF APIs must not silently discard SAP atomic-spin tags."""
    monkeypatch.setattr(
        periodic_k_gdf,
        "reject_periodic_gdf_unsupported_functional",
        lambda *_args, **_kwargs: pytest.fail(
            "atomic_spins + SAP reached GDF backend setup"
        ),
    )
    system, basis = _h_box()
    opts = (
        _uks_opts(vq.InitialGuess.SAP)
        if functional
        else _rhf_opts(vq.InitialGuess.SAP)
    )
    opts.atomic_spins = [1]
    kwargs = {"functional": functional} if functional else {}

    with pytest.raises(
        RuntimeError,
        match="atomic_spins.*requires the SAD guess.*SAP",
    ):
        driver(system, basis, (2, 1, 1), opts, progress=False, **kwargs)


@pytest.mark.parametrize(
    "guess",
    [vq.InitialGuess.FRAGMO],
)
def test_direct_multik_open_gdf_unsupported_guess_fails_before_setup(
    monkeypatch, guess
):
    """Unwired density modes fail instead of becoming Hcore."""
    monkeypatch.setattr(
        periodic_k_gdf,
        "_oneel_lattice_opts",
        lambda *_args, **_kwargs: pytest.fail(
            "unsupported guess reached GDF backend setup"
        ),
    )
    system, basis = _h_box()
    opts = _rhf_opts(guess)

    with pytest.raises(NotImplementedError, match=f"initial_guess={guess.name}"):
        periodic_k_gdf.run_kuhf_periodic_gdf(
            system, basis, (2, 1, 1), opts, progress=False
        )


def test_gamma_gdf_open_shell_auto_executes_shared_sad_density(monkeypatch):
    """Gamma GDF AUTO executes SAD rather than retaining its Hcore seed."""
    system, basis = _h_box()
    opts = _rhf_opts(vq.InitialGuess.AUTO)
    opts.max_iter = 0
    setup = pbc_gdf._PbcGdfGammaSetup(
        S=np.eye(basis.nbasis),
        Hcore=np.eye(basis.nbasis),
        X=np.eye(basis.nbasis),
        n_kept=basis.nbasis,
        Lpq=np.zeros((1, basis.nbasis, basis.nbasis)),
        aux=SimpleNamespace(name="stub-aux", nbasis=1),
        madelung=0.0,
        e_nuc=0.0,
        gauge_lat_opts=opts.lattice_opts,
        compcell_fit_state=None,
        fit_cutoff_2c=float("nan"),
        fit_cutoff_3c=float("nan"),
    )
    monkeypatch.setattr(
        pbc_gdf, "_pbc_gdf_gamma_setup", lambda *_args, **_kwargs: setup
    )
    seen = []
    expected_alpha = np.full((basis.nbasis, basis.nbasis), 0.75)
    expected_beta = np.zeros((basis.nbasis, basis.nbasis))

    def fake_density(
        _mol, _basis, _n_alpha, _n_beta, initial_guess, **kwargs
    ):
        seen.append((initial_guess, kwargs))
        return expected_alpha.copy(), expected_beta.copy()

    monkeypatch.setattr(pbc_gdf, "initial_densities_open_shell", fake_density)
    result = pbc_gdf.run_pbc_gdf_uhf(
        system,
        basis,
        opts,
        check_energy_sanity=False,
        progress=False,
        verbose=0,
    )

    assert seen[0][0] == vq.InitialGuess.SAD
    np.testing.assert_allclose(result.density_alpha, expected_alpha)
    np.testing.assert_allclose(result.density_beta, expected_beta)


@pytest.mark.parametrize(
    ("driver", "functional"),
    [
        (periodic_rijcosx.run_periodic_rijcosx_uhf, None),
        (periodic_rijcosx.run_periodic_rijcosx_uks, "lda"),
    ],
    ids=["uhf", "uks"],
)
def test_direct_gamma_rijcosx_sap_rejects_atomic_spins_before_setup(
    monkeypatch,
    driver,
    functional,
):
    """Direct RIJCOSX APIs must not silently discard SAP atomic-spin tags."""
    monkeypatch.setattr(
        periodic_rijcosx,
        "_vacuum_envelope_setup",
        lambda *_args, **_kwargs: pytest.fail(
            "atomic_spins + SAP reached RIJCOSX backend setup"
        ),
    )
    system, basis = _h_box()
    opts = (
        _uks_opts(vq.InitialGuess.SAP)
        if functional
        else _rhf_opts(vq.InitialGuess.SAP)
    )
    opts.atomic_spins = [1]
    kwargs = {"functional": functional} if functional else {}

    with pytest.raises(
        ValueError,
        match="atomic_spins requires SAD",
    ):
        driver(system, basis, opts, progress=False, **kwargs)


@pytest.mark.parametrize("guess", _OPEN_SHELL_PERIODIC_HELPER_GUESSES)
def test_periodic_open_shell_guess_helper_uses_context(guess):
    """The helper builds trace-correct per-spin densities with context."""
    system, basis = _h_box()
    opts = vq.LatticeSumOptions()
    Da, Db = initial_densities_open_shell(
        system.unit_cell_molecule(),
        basis,
        1,
        0,
        guess,
        is_periodic=True,
        periodic_system=system,
        lattice_opts=opts,
    )
    S = _gamma(compute_overlap_lattice(basis, system, opts))
    assert Da.shape == (basis.nbasis, basis.nbasis)
    assert Db.shape == (basis.nbasis, basis.nbasis)
    assert np.max(np.abs(Da - Da.T)) < 1e-10
    assert np.max(np.abs(Db - Db.T)) < 1e-10
    assert float(np.trace(Da @ S)) == pytest.approx(1.0, abs=1e-8)
    assert float(np.trace(Db @ S)) == pytest.approx(0.0, abs=1e-10)


def test_periodic_open_shell_gamma_ewald_guesses_reach_same_basin_as_sad():
    """Gamma-Ewald UHF/UKS pass the periodic guess context."""
    system, basis = _h_box()

    def uhf_energy(guess):
        result = vq.run_uhf_periodic_gamma_ewald3d(
            system,
            basis,
            _rhf_opts(guess),
            auto_optimize_truncation=False,
            verbose=0,
        )
        assert result.converged, f"Gamma UHF/{guess.name} did not converge"
        return result.energy

    def uks_energy(guess):
        result = vq.run_uks_periodic_gamma_ewald3d(
            system,
            basis,
            _uks_opts(guess),
            omega=0.5,
            verbose=0,
        )
        assert result.converged, f"Gamma UKS/{guess.name} did not converge"
        return result.energy

    _assert_guess_matches_sad("Gamma UHF", uhf_energy)
    _assert_guess_matches_sad("Gamma UKS", uks_energy)


def test_periodic_sap_gamma_rohf_forwards_periodic_context(monkeypatch):
    """The exported Gamma ROHF route must not send SAP to a contextless helper."""
    seen = []

    class ContextObserved(RuntimeError):
        pass

    def observe_context(*args, **kwargs):
        seen.append((args, kwargs))
        raise ContextObserved

    monkeypatch.setattr(
        periodic_rohf_ewald,
        "initial_densities_open_shell",
        observe_context,
    )
    system, basis = _h_box()
    opts = _rhf_opts(vq.InitialGuess.SAP)
    opts.max_iter = 1
    opts.lattice_opts.cutoff_bohr = 4.0
    opts.lattice_opts.nuclear_cutoff_bohr = 4.0

    with pytest.raises(ContextObserved):
        periodic_rohf_ewald.run_rohf_periodic_gamma_ewald3d(
            system,
            basis,
            opts,
            auto_optimize_truncation=False,
            progress=False,
            verbose=0,
        )

    assert len(seen) == 1
    args, kwargs = seen[0]
    assert args[4] == vq.InitialGuess.SAP
    assert kwargs["is_periodic"] is True
    assert kwargs["periodic_system"] is system
    assert kwargs["lattice_opts"] is opts.lattice_opts


def test_periodic_open_shell_multik_ewald_guesses_reach_same_basin_as_sad():
    """Multi-k Ewald UHF/UKS pass the periodic guess context."""
    system, basis = _h_box()
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])

    def uhf_energy(guess):
        result = vq.run_uhf_periodic_multi_k_ewald3d(
            system, basis, kmesh, _rhf_opts(guess), verbose=0)
        assert result.converged, f"multi-k UHF/{guess.name} did not converge"
        return result.energy

    def uks_energy(guess):
        result = vq.run_uks_periodic_multi_k_ewald3d(
            system, basis, kmesh, _uks_opts(guess), verbose=0)
        assert result.converged, f"multi-k UKS/{guess.name} did not converge"
        return result.energy

    _assert_guess_matches_sad("multi-k UHF", uhf_energy)
    _assert_guess_matches_sad("multi-k UKS", uks_energy)


def test_periodic_open_shell_gdf_guesses_reach_same_basin_as_sad():
    """GDF UHF/UKS pass the periodic guess context."""
    system, basis = _h_box()

    def uhf_energy(guess):
        result = vq.run_pbc_gdf_uhf(
            system, basis, _rhf_opts(guess), progress=False, verbose=0)
        assert result.converged, f"GDF UHF/{guess.name} did not converge"
        return result.energy

    def uks_energy(guess):
        result = vq.run_pbc_gdf_uks(
            system,
            basis,
            _uks_opts(guess),
            functional="lda",
            progress=False,
            verbose=0,
        )
        assert result.converged, f"GDF UKS/{guess.name} did not converge"
        return result.energy

    _assert_guess_matches_sad("GDF UHF", uhf_energy)
    _assert_guess_matches_sad("GDF UKS", uks_energy)


def test_periodic_open_shell_bipole_guesses_reach_same_basin_as_sad():
    """BIPOLE UHF/UKS pass the periodic guess context."""
    system, basis = _h_box()
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])

    def uhf_energy(guess):
        result = vq.run_pbc_bipole_uhf(
            system,
            basis,
            kmesh,
            _rhf_opts(guess),
            progress=False,
            verbose=0,
        )
        assert result.converged, f"BIPOLE UHF/{guess.name} did not converge"
        return result.energy

    def uks_energy(guess):
        result = vq.run_pbc_bipole_uks(
            system,
            basis,
            kmesh,
            _uks_opts(guess),
            progress=False,
            verbose=0,
        )
        assert result.converged, f"BIPOLE UKS/{guess.name} did not converge"
        return result.energy

    _assert_guess_matches_sad("BIPOLE UHF", uhf_energy)
    _assert_guess_matches_sad("BIPOLE UKS", uks_energy)


@pytest.mark.parametrize(
    ("module", "driver_name", "is_ks", "is_bipole"),
    [
        (pbc_bipole_uhf, "run_pbc_bipole_uhf", False, True),
        (pbc_bipole_uks, "run_pbc_bipole_uks", True, True),
        (
            periodic_rohf_multi_k_ewald,
            "run_rohf_periodic_multi_k_ewald3d",
            False,
            False,
        ),
        (
            periodic_roks_multi_k_ewald,
            "run_roks_periodic_multi_k_ewald3d",
            True,
            False,
        ),
    ],
    ids=["uhf", "uks", "rohf", "roks"],
)
@pytest.mark.parametrize(
    "single_gamma",
    [False, True],
    ids=["two-point", "single-gamma"],
)
@pytest.mark.parametrize(
    "guess",
    [vq.InitialGuess.SAP, vq.InitialGuess.HUECKEL],
    ids=["sap", "hueckel"],
)
def test_periodic_fock_guess_open_shell_dispatches_bloch_fock(
    monkeypatch,
    module,
    driver_name,
    is_ks,
    is_bipole,
    single_gamma,
    guess,
):
    """Every open-shell corrected-Ewald route must build F_guess(k)."""
    seen = []

    class SapPerKConsumed(RuntimeError):
        pass

    def fake_fock_guess(system, basis, kpoints_cart, initial_guess, **kwargs):
        assert initial_guess == guess
        kpoints = list(kpoints_cart)
        seen.append((system, basis, kpoints, kwargs))
        return tuple(
            np.eye(basis.nbasis, dtype=complex) * (10.0 + index)
            for index, _ in enumerate(kpoints)
        )

    n_test_k = 1 if single_gamma else 2
    expected_diagonalisations = n_test_k * (2 if is_bipole else 1)
    diagonalised = []

    def stop_after_guess(fock, orthogonalizer):
        diagonalised.append(np.asarray(fock).copy())
        if len(diagonalised) == expected_diagonalisations:
            raise SapPerKConsumed
        X = np.asarray(orthogonalizer, dtype=complex)
        return X, np.arange(X.shape[1], dtype=float)

    monkeypatch.setattr(module, "periodic_fock_guess_k", fake_fock_guess)
    monkeypatch.setattr(module, "_diag_in_orth_basis", stop_after_guess)
    system, basis = _h_box()
    kmesh = (
        vq.monkhorst_pack(system, [1, 1, 1])
        if single_gamma
        else vq.monkhorst_pack(system, [2, 1, 1])
    )
    opts = _uks_opts(guess) if is_ks else _rhf_opts(guess)
    opts.max_iter = 1
    opts.lattice_opts.cutoff_bohr = 4.0
    opts.lattice_opts.nuclear_cutoff_bohr = 4.0
    kwargs = {"progress": False, "verbose": 0}
    if is_bipole:
        kwargs["use_ewald_j_split"] = True
    else:
        kwargs["auto_optimize_truncation"] = False
        kwargs["sr_image_precision"] = None

    with pytest.raises(SapPerKConsumed):
        getattr(module, driver_name)(system, basis, kmesh, opts, **kwargs)

    assert len(seen) == 1
    _, _, kpoints_cart, guess_kwargs = seen[0]
    assert len(kpoints_cart) == n_test_k
    if single_gamma:
        np.testing.assert_allclose(kpoints_cart[0], 0.0, atol=1e-12)
    else:
        assert not np.array_equal(kpoints_cart[0], kpoints_cart[1])
    assert guess_kwargs["kinetic_lattice"] is not None
    assert guess_kwargs["overlap_lattice"] is not None
    assert guess_kwargs["lattice_opts"] is not None
    expected_values = (
        [10.0, 10.0]
        if single_gamma and is_bipole
        else [10.0]
        if single_gamma
        else [10.0, 10.0, 11.0, 11.0]
        if is_bipole
        else [10.0, 11.0]
    )
    assert [float(block[0, 0].real) for block in diagonalised] == expected_values


@pytest.mark.parametrize(
    ("module", "driver_name", "is_ks"),
    [
        (
            periodic_rohf_multi_k_ewald,
            "run_rohf_periodic_multi_k_ewald3d",
            False,
        ),
        (
            periodic_roks_multi_k_ewald,
            "run_roks_periodic_multi_k_ewald3d",
            True,
        ),
    ],
    ids=["rohf", "roks"],
)
def test_multik_restricted_open_shell_sad_consumes_shared_density(
    monkeypatch,
    module,
    driver_name,
    is_ks,
):
    """A true multi-k SAD request must not retain the Hcore density."""

    class DensityConsumed(RuntimeError):
        pass

    class DensitySentinel:
        def __array__(self, dtype=None, copy=None):
            raise DensityConsumed

    def fake_densities(*_args, **_kwargs):
        return DensitySentinel(), DensitySentinel()

    monkeypatch.setattr(
        periodic_guess,
        "initial_densities_open_shell",
        fake_densities,
    )
    system, basis = _h_box()
    kmesh = vq.monkhorst_pack(system, [2, 1, 1])
    opts = (
        _uks_opts(vq.InitialGuess.SAD)
        if is_ks
        else _rhf_opts(vq.InitialGuess.SAD)
    )
    opts.max_iter = 1
    opts.lattice_opts.cutoff_bohr = 4.0
    opts.lattice_opts.nuclear_cutoff_bohr = 4.0

    with pytest.raises(DensityConsumed):
        getattr(module, driver_name)(
            system,
            basis,
            kmesh,
            opts,
            auto_optimize_truncation=False,
            sr_image_precision=None,
            progress=False,
            verbose=0,
        )


@pytest.mark.parametrize(
    ("module", "driver_name", "is_ks"),
    [
        (
            periodic_uhf_multi_k_ewald,
            "run_uhf_periodic_multi_k_ewald3d",
            False,
        ),
        (
            periodic_uks_multi_k_ewald,
            "run_uks_periodic_multi_k_ewald3d",
            True,
        ),
    ],
    ids=["uhf", "uks"],
)
@pytest.mark.parametrize(
    "guess",
    [vq.InitialGuess.SAP, vq.InitialGuess.HUECKEL],
    ids=["sap", "hueckel"],
)
def test_periodic_fock_guess_legacy_ewald_open_shell_dispatches_bloch_fock(
    monkeypatch,
    module,
    driver_name,
    is_ks,
    guess,
):
    """Legacy multi-k Ewald UHF/UKS fill both spins from F_guess(k)."""
    seen = []

    class SapPerKConsumed(RuntimeError):
        pass

    def fake_fock_guess(system, basis, kpoints_cart, initial_guess, **kwargs):
        assert initial_guess == guess
        kpoints = list(kpoints_cart)
        seen.append((system, basis, kpoints, kwargs))
        return tuple(
            np.eye(basis.nbasis, dtype=complex) * (10.0 + index)
            for index, _ in enumerate(kpoints)
        )

    diagonalised = []

    def stop_after_all_spins_and_k(fock, orthogonalizer):
        diagonalised.append(np.asarray(fock).copy())
        if len(diagonalised) == 4:
            raise SapPerKConsumed
        X = np.asarray(orthogonalizer, dtype=complex)
        return X, np.arange(X.shape[1], dtype=float)

    monkeypatch.setattr(module, "periodic_fock_guess_k", fake_fock_guess)
    monkeypatch.setattr(
        module, "_diag_in_orth_basis", stop_after_all_spins_and_k
    )
    system, basis = _h_box()
    kmesh = vq.monkhorst_pack(system, [2, 1, 1])
    opts = _uks_opts(guess) if is_ks else _rhf_opts(guess)
    opts.max_iter = 1
    opts.lattice_opts.cutoff_bohr = 4.0
    opts.lattice_opts.nuclear_cutoff_bohr = 4.0

    with pytest.raises(SapPerKConsumed):
        getattr(module, driver_name)(
            system,
            basis,
            kmesh,
            opts,
            auto_optimize_truncation=False,
            progress=False,
            verbose=0,
        )

    assert len(seen) == 1
    _, _, kpoints_cart, kwargs = seen[0]
    assert len(kpoints_cart) == 2
    assert not np.array_equal(kpoints_cart[0], kpoints_cart[1])
    assert kwargs["kinetic_lattice"] is not None
    assert kwargs["overlap_lattice"] is not None
    assert kwargs["lattice_opts"] is not None
    assert [float(block[0, 0].real) for block in diagonalised] == [
        10.0,
        10.0,
        11.0,
        11.0,
    ]


@pytest.mark.parametrize("multik", [False, True])
@pytest.mark.parametrize("kind", [vq.InitialGuess.SAD, vq.InitialGuess.MINAO])
def test_lih_density_guess_counts_actual_gamma_and_weighted_bloch_metric(multik, kind):
    from vibeqc import _vibeqc_core as core
    from vibeqc.guess import initial_density_closed_shell

    system = vq.PeriodicSystem(
        3, np.eye(3) * 8.0,
        [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [0, 0, 3.0])],
    )
    mol = system.unit_cell_molecule()
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 14.0
    lattice_overlap = core.compute_overlap_lattice(basis, system, opts)
    points = [np.array([0.13, 0.07, 0.02]), np.array([-0.13, -0.07, -0.02])]
    overlaps = [np.asarray(core.bloch_sum(lattice_overlap, k)) for k in points]
    weights = np.array([0.37, 0.63])
    if not multik:
        overlaps = [_gamma(lattice_overlap)]
        weights = np.ones(1)
    density = initial_density_closed_shell(
        mol, basis, 2, kind, periodic_system=system, lattice_opts=opts,
        overlap=overlaps, weights=weights,
    )
    assert sum(w * np.trace(density @ sk) for w, sk in zip(weights, overlaps)) == pytest.approx(4, abs=2e-11)
    da, db = initial_densities_open_shell(
        mol, basis, 3, 1, kind, periodic_system=system, lattice_opts=opts,
        overlap=overlaps, weights=weights,
    )
    for matrix, target in ((da, 3), (db, 1)):
        assert sum(w * np.trace(matrix @ sk) for w, sk in zip(weights, overlaps)) == pytest.approx(target, abs=2e-11)


@pytest.mark.parametrize("method", ["rhf", "rks"])
def test_gamma_ewald_patom_reaches_full_hf_in_field_builder(monkeypatch, method):
    import importlib
    import vibeqc as vq

    module = importlib.import_module(f"vibeqc.periodic_{method}_ewald")
    system = vq.PeriodicSystem(
        3, np.eye(3) * 16, [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [0, 0, 3])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicKSOptions() if method == "rks" else vq.PeriodicRHFOptions()
    opts.initial_guess = vq.InitialGuess.PATOM
    opts.max_iter = 0
    opts.lattice_opts.cutoff_bohr = 12
    opts.lattice_opts.nuclear_cutoff_bohr = 12
    calls = []
    original = module.build_jk_gamma_molecular_limit

    def capture(basis_arg, system_arg, lattice_options, density, omega):
        calls.append((np.asarray(density).copy(), omega))
        return original(basis_arg, system_arg, lattice_options, density, omega)

    monkeypatch.setattr(module, "build_jk_gamma_molecular_limit", capture)
    result = getattr(module, f"run_{method}_periodic_gamma_ewald3d")(
        system, basis, opts, progress=False)
    assert len(calls) == 1  # The in-field builder, before any SCF iteration.
    assert calls[0][1] == 0  # Full HF exchange even for a pure-DFT target.
    assert np.trace(calls[0][0] @ result.overlap) == pytest.approx(4, abs=1e-10)
    assert np.trace(result.density @ result.overlap) == pytest.approx(4, abs=1e-10)
    assert np.linalg.norm(result.density - calls[0][0]) > 1e-3
    assert result.guess_selection.effective == vq.InitialGuess.PATOM


def _patom_case(*, spin=False, ks=True):
    atoms = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])]
    if spin:
        atoms.append(vq.Atom(1, [0, 1.4, 0]))
    system = vq.PeriodicSystem(3, np.eye(3) * 6, atoms, multiplicity=2 if spin else 1)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    mesh = vq.monkhorst_pack(system, [3, 1, 1])
    opts = vq.PeriodicKSOptions() if ks else vq.PeriodicRHFOptions()
    opts.initial_guess = vq.InitialGuess.PATOM
    opts.max_iter = 1
    opts.use_diis = False
    opts.damping = 0
    opts.lattice_opts.cutoff_bohr = 7
    opts.lattice_opts.nuclear_cutoff_bohr = 10
    if ks:
        opts.functional = "pbe"
    return system, basis, mesh, opts


def _patom_populations(result, mesh, targets):
    for name, target in targets.items():
        density = getattr(result, name)
        matrices = (density if isinstance(density, (list, tuple)) else
                    bvk_torus_density_matrices(density, list(mesh.kpoints), (3, 1, 1)))
        for matrix in matrices:
            assert np.iscomplexobj(matrix)
            np.testing.assert_allclose(matrix, matrix.conj().T, atol=2e-12)
        count = sum(w * np.trace(d @ s) for w, d, s in
                    zip(mesh.weights, matrices, result.overlap))
        assert count == pytest.approx(target, abs=2e-11)
    for field in ("requested", "effective", "transport"):
        assert getattr(result.guess_selection, field) == vq.InitialGuess.PATOM


@pytest.mark.parametrize("route", ["ewald", "bipole", "gdf"])
def test_pure_rks_patom_reaches_full_hf_builder(monkeypatch, route):
    modules = {"ewald": "periodic_rks_multi_k_ewald", "bipole": "pbc_bipole_rks", "gdf": "periodic_k_gdf"}
    hooks = {"ewald": "build_periodic_fock_ewald3d_k", "bipole": "build_bipole_restricted_fock", "gdf": "apply_exxdiv_ewald_to_K"}
    module = importlib.import_module("vibeqc." + modules[route])
    system, basis, mesh, opts = _patom_case()
    seen = []
    original = getattr(module, hooks[route])
    def capture(*args, **kwargs):
        if route == "ewald":
            exx = kwargs.get("exchange_assembly")
            seen.append(0 if exx is None else exx.c_full)
        elif route == "bipole":
            seen.append(kwargs["alpha_hf"])
        else:
            seen.append(sum(w * np.trace(d @ s) for w, d, s in
                            zip(mesh.weights, args[2], args[1])))
        return original(*args, **kwargs)
    monkeypatch.setattr(module, hooks[route], capture)
    if route == "ewald":
        result = module.run_rks_periodic_multi_k_ewald3d(system, basis, mesh, opts, omega=.5, auto_optimize_truncation=False, progress=False)
    elif route == "bipole":
        result = module.run_pbc_bipole_rks(system, basis, mesh, opts, ewald_precision=1e-6, sr_image_precision=None, progress=False)
    else:
        result = module.run_krhf_periodic_gdf(system, basis, mesh, opts, functional="pbe", use_compcell=True, aux_basis="def2-svp-jk", rsgdf_ke_cutoff=40, progress=False)
    if route == "gdf":
        assert seen == pytest.approx([2], abs=2e-11)
    else:
        assert seen[0] == 1
        assert all(value == 0 for value in seen[1:])
    _patom_populations(result, mesh, {"density": 2})


@pytest.mark.parametrize("ks", [False, True], ids=["uhf", "uks"])
def test_multik_gdf_patom_spin_seed_exchange_and_populations(monkeypatch, ks):
    import vibeqc.periodic_k_gdf as module
    system, basis, mesh, opts = _patom_case(spin=True, ks=ks)
    seen = []
    original = module.apply_exxdiv_ewald_to_K
    def capture(ks, overlaps, densities, *args, **kwargs):
        seen.append(sum(w * np.trace(d @ s) for w, d, s in
                        zip(mesh.weights, densities, overlaps)))
        return original(ks, overlaps, densities, *args, **kwargs)
    monkeypatch.setattr(module, "apply_exxdiv_ewald_to_K", capture)
    result = module.run_kuhf_periodic_gdf(system, basis, mesh, opts, functional="pbe" if ks else None, aux_basis="def2-svp-jk", rsgdf_ke_cutoff=40, progress=False)
    assert seen[:2] == pytest.approx([2, 1], abs=2e-11)
    if ks:
        assert len(seen) == 2  # Pure PBE has no SCF exchange; both calls are PATOM.
    _patom_populations(result, mesh, {"density_alpha": 2, "density_beta": 1})


def test_patom_noncompcell_rks_matches_direct_ewald_first_cycle():
    """One PATOM seed, one cycle, two Coulomb routes (#275).

    The lattice cutoff is raised to 20 bohr here, away from the 7 bohr the
    shared ``_patom_case`` uses. At 7 bohr neither route is converged: the
    Ewald leg alone moves 3.49 mHa between cutoff 7 and 12, so the routes'
    2.21 mHa "disagreement" there was mostly each one's own truncation
    error, not a route difference. Both legs are stable to ~1e-15 between
    20 and 30 bohr.

    Converged, the two routes agree exactly where they must and differ by a
    small, reproducible amount in the Coulomb term:

        e_xc       identical to all digits
        e_nuclear  identical to 1.7e-14
        e_electronic / total   gdf - ewald = -4.738114e-05 Ha

    So the XC grid, the functional and the nuclear lattice sum agree
    exactly, and the whole residual lives in the Coulomb treatment (Ewald
    versus the fitted GDF Coulomb). That residual is NOT auxiliary-basis
    incompleteness in any way this fixture can show: ``rsgdf_ke_cutoff`` 40
    to 320 leaves the GDF energy bit-identical, and the def2 JK-fitting sets
    are all nbasis=36 on hydrogen, so they cannot separate the two
    explanations here. Whether 4.7e-05 Ha is the expected fitting error or a
    defect is an open question on #275, deliberately not settled by widening
    a tolerance.

    The previous revision asserted the totals equal to 2e-10 at the
    unconverged 7 bohr cutoff. That pin was never achievable for two
    independently truncated routes with different Coulomb treatments.
    """
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf
    from vibeqc.periodic_rks_multi_k_ewald import run_rks_periodic_multi_k_ewald3d
    system, basis, mesh, opts = _patom_case()
    opts.lattice_opts.cutoff_bohr = 20
    ewald = run_rks_periodic_multi_k_ewald3d(system, basis, mesh, opts, omega=.5, auto_optimize_truncation=False, progress=False)
    gdf = run_krhf_periodic_gdf(system, basis, mesh, opts, functional="pbe", use_compcell=False, aux_basis="def2-svp-jk", rsgdf_ke_cutoff=40, progress=False)
    # What one seed and one cycle must reproduce exactly across the routes.
    assert gdf.e_xc == pytest.approx(ewald.e_xc, abs=1e-12)
    assert gdf.e_nuclear == pytest.approx(ewald.e_nuclear, abs=1e-12)
    # The Coulomb-treatment residual, pinned so a regression moves it.
    assert gdf.energy == pytest.approx(ewald.energy, abs=1e-4)
    assert gdf.energy - ewald.energy == pytest.approx(-4.738114e-05, abs=1e-9)
    _patom_populations(gdf, mesh, {"density": 2})


@pytest.mark.parametrize("ks", [False, True])
def test_open_multik_gdf_read_preserves_spin_density_and_validates_request(ks):
    from vibeqc.guess_read import resolve_periodic_read_densities_k_open
    from vibeqc.periodic_k_gdf import run_kuhf_periodic_gdf

    system, basis, mesh, opts = _patom_case(spin=True, ks=ks)
    kwargs = {"functional": "pbe"} if ks else {}
    common = dict(gdf_method="rsgdf", rsgdf_ke_cutoff=20, progress=False, check_energy_sanity=False)
    source = run_kuhf_periodic_gdf(system, basis, mesh, opts, **kwargs, **common)
    densities = resolve_periodic_read_densities_k_open(
        read_from=source, basis=basis, system=system, kmesh=mesh,
    )
    for actual, expected in zip(densities, (source.density_alpha, source.density_beta)):
        np.testing.assert_allclose(actual, expected, atol=1e-13)
    opts.initial_guess = vq.InitialGuess.READ
    result = run_kuhf_periodic_gdf(
        system, basis, mesh, opts, initial_density_k=densities, **kwargs, **common,
    )
    for blocks, count in ((result.density_alpha, 2), (result.density_beta, 1)):
        assert sum(w * np.trace(d @ s).real for w, d, s in
                   zip(mesh.weights, blocks, result.overlap)) == pytest.approx(count, abs=1e-11)
    assert result.guess_selection.effective == vq.InitialGuess.READ
    opts.initial_guess = vq.InitialGuess.FRAGMO
    with pytest.raises(NotImplementedError, match="FRAGMO"):
        run_kuhf_periodic_gdf(
            system, basis, mesh, opts, initial_density_k=densities, **kwargs, **common,
        )


@pytest.mark.parametrize("method,route", [
    ("RHF", "gdf"), ("UHF", "gdf"), ("ROHF", "bipole"),
    ("RKS", "bipole"), ("UKS", "bipole"), ("ROKS", "bipole"),
])
@pytest.mark.parametrize("nk", [1, 3], ids=["gamma", "non_special_multik"])
def test_public_periodic_guess_converged_basins_and_counts(tmp_path, method, route, nk):
    from vibeqc.pbc_bipole_common import bvk_torus_density_matrices
    opened = method in ("UHF", "UKS", "ROHF", "ROKS")
    atoms = [vq.Atom(1, [0, 0, 0])]
    if not opened:
        atoms.append(vq.Atom(1, [0, 0, 1.4]))
    system = vq.PeriodicSystem(3, np.eye(3) * 10, atoms, multiplicity=2 if opened else 1)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    mesh = vq.monkhorst_pack(system, [nk, 1, 1])
    kwargs = dict(
        method=method, jk_method=route, kpoints=mesh,
        max_iter=35, conv_tol_energy=1e-9, convergence="off", damping=0.,
        citations=False, verbose=0, progress=False, output_qvf=False,
        write_xyz_file=False, write_cif_file=False, write_xsf_structure_file=False,
        write_molden_file=False, write_population_file=False,
    )
    if method.endswith("KS"):
        kwargs["functional"] = "pbe"
    if route == "gdf":
        kwargs.update(aux_basis="def2-svp-jk", rsgdf_ke_cutoff=40.)
    elif method not in ("ROHF", "ROKS"):
        kwargs.update(bipole_cutoff_bohr=12., bipole_nuclear_cutoff_bohr=12.,
                      sr_image_precision=None, ewald_precision=1e-6)
    energies = []
    for requested in (vq.InitialGuess.AUTO, vq.InitialGuess.HCORE, vq.InitialGuess.SAD):
        result = vq.run_periodic_job(system, basis, initial_guess=requested,
                                     output=tmp_path / requested.name, **kwargs)
        assert result.converged
        energies.append(result.energy)
        effective = vq.InitialGuess.SAD if requested == vq.InitialGuess.AUTO else requested
        assert result.guess_selection.requested == requested
        assert result.guess_selection.effective == effective
        assert result.guess_selection.transport == effective
        overlaps = np.asarray(result.overlap)
        if overlaps.ndim == 2:
            overlaps = overlaps[None]
        populations = {"density_alpha": 1, "density_beta": 0} if opened else {"density": 2}
        for name, target in populations.items():
            density = getattr(result, name)
            if hasattr(density, "cells"):
                matrices = bvk_torus_density_matrices(density, list(mesh.kpoints), (nk, 1, 1))
            else:
                matrices = np.asarray(density)
                if matrices.ndim == 2:
                    matrices = matrices[None]
            assert len(matrices) == len(overlaps) == nk
            for matrix in matrices:
                np.testing.assert_allclose(matrix, matrix.conj().T, atol=2e-11)
            count = sum(w * np.trace(d @ s) for w, d, s in zip(mesh.weights, matrices, overlaps))
            assert count == pytest.approx(target, abs=2e-10)
    np.testing.assert_allclose(energies, energies[0], rtol=0, atol=2e-8)


@pytest.mark.parametrize('family', ['uhf', 'uks'])
def test_multik_ewald_seed_damping_energy_error_exchange_and_shift(monkeypatch, family):
    import importlib
    import inspect
    """Actual SAD and damped densities must reach every first-cycle consumer."""
    from vibeqc.periodic_k_density import real_space_density_from_per_k_density
    from vibeqc.periodic_corrected_exchange import CorrectedEwaldExchange

    module = importlib.import_module(f'vibeqc.periodic_{family}_multi_k_ewald')
    system = vq.PeriodicSystem(
        3, 12.0 * np.eye(3),
        [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [1.5, 0, 0]),
         vq.Atom(1, [3.2, 0.3, 0])], multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), 'sto-3g')
    kmesh = vq.monkhorst_pack(system, [3, 1, 1])
    options = vq.PeriodicRHFOptions() if family == 'uhf' else vq.PeriodicKSOptions()
    options.initial_guess = vq.InitialGuess.SAD
    options.max_iter = 2
    options.damping = 0.3
    options.use_diis = False
    options.level_shift = 0.1
    options.lattice_opts.cutoff_bohr = 4.0
    options.lattice_opts.nuclear_cutoff_bohr = 4.0
    if family == 'uks':
        options.functional = 'pbe0'
        options.grid.n_radial = 10

    seeds, builds, iterations, shifts, reciprocal_exchange = [], [], [], [], []
    initial_builder = module.initial_densities_open_shell
    def seed(*args, **kwargs):
        result = initial_builder(*args, **kwargs)
        seeds.append(tuple(np.asarray(d).copy() for d in result))
        return result
    monkeypatch.setattr(module, 'initial_densities_open_shell', seed)

    build_name = ('_build_uhf_fock_blocks_ewald3d' if family == 'uhf'
                  else '_build_uks_fock_2e_blocks_ewald3d')
    original_build = getattr(module, build_name)
    def fock(*args, **kwargs):
        builds.append(tuple([np.asarray(b).copy() for b in d.blocks] for d in args[2:4]))
        return original_build(*args, **kwargs)
    monkeypatch.setattr(module, build_name, fock)

    original_exchange = CorrectedEwaldExchange.k_space_terms_all_k
    def exchange(self, overlap_k, density_k):
        reciprocal_exchange.append([np.asarray(d).copy() for d in density_k])
        return original_exchange(self, overlap_k, density_k)
    monkeypatch.setattr(CorrectedEwaldExchange, 'k_space_terms_all_k', exchange)

    original_iteration = module.SCFIteration
    def iteration(**fields):
        # The public iteration record is a stable hook, independent of line
        # numbers. Capture the physical SCF state before DIIS/level shift.
        state = inspect.currentframe().f_back.f_locals
        ia = state['D_a_used_per_k']
        ib = state['D_b_used_per_k']
        pair = (ia, ib)
        cycle = len(iterations)
        expected_energy = 0.0
        expected_grad = 0.0
        for spin, density_k in enumerate(pair):
            if cycle == 0:
                for density in density_k:
                    np.testing.assert_allclose(density, seeds[0][spin], atol=1e-13)
            else:
                current = state['D_a_cur_per_k' if spin == 0 else 'D_b_cur_per_k']
                for actual, cur, previous in zip(density_k, current, iterations[0]['density'][spin]):
                    np.testing.assert_allclose(actual, 0.7 * cur + 0.3 * previous, atol=1e-13)
            count = sum(w * np.trace(d @ s).real
                        for w, d, s in zip(kmesh.weights, density_k, state['S_k_list']))
            assert count == pytest.approx(2 - spin, abs=1e-12)
            folded = real_space_density_from_per_k_density(density_k, kmesh, state['cells'])
            for actual, expected in zip(builds[cycle][spin], folded.blocks):
                np.testing.assert_allclose(actual, expected, atol=1e-13)
        for index, weight in enumerate(kmesh.weights):
            da, db = ia[index], ib[index]
            s, h = state['S_k_list'][index], state['Hcore_k_list'][index]
            fa, fb = state['F_alpha_k_list'][index], state['F_beta_k_list'][index]
            if family == 'uhf':
                expected_energy += 0.5 * weight * np.trace((da + db) @ h + da @ fa + db @ fb).real
            else:
                expected_energy += weight * np.trace(
                    (da + db) @ h + 0.5 * da @ state['F_HF_alpha_k_list'][index]
                    + 0.5 * db @ state['F_HF_beta_k_list'][index]).real
            ea, eb = fa @ da @ s, fb @ db @ s
            ea, eb = ea - ea.conj().T, eb - eb.conj().T
            expected_grad += weight * np.sqrt(np.linalg.norm(ea)**2 + np.linalg.norm(eb)**2)
        expected_energy += state['e_nuc'] + state['E_madelung_fix']
        if family == 'uks':
            expected_energy += state['E_xc']
            for actual, expected in zip(reciprocal_exchange[2*cycle:2*cycle+2], pair):
                np.testing.assert_allclose(actual, expected, atol=1e-13)
        assert fields['energy'] == pytest.approx(expected_energy, abs=1e-12)
        assert fields['grad_norm'] == pytest.approx(expected_grad, abs=1e-12)
        iterations.append(dict(
            density=tuple([d.copy() for d in ds] for ds in pair),
            fock=([f.copy() for f in state['F_alpha_k_list']],
                  [f.copy() for f in state['F_beta_k_list']]),
            overlap=[s.copy() for s in state['S_k_list']],
        ))
        return original_iteration(**fields)
    monkeypatch.setattr(module, 'SCFIteration', iteration)

    original_diag = module._diag_in_orth_basis
    def diagonalize(fock_k, orthogonalizer):
        if iterations:
            state = iterations[-1]
            index, spin = (len(shifts) % 6) // 2, len(shifts) % 2
            density, overlap = state['density'][spin][index], state['overlap'][index]
            expected = state['fock'][spin][index] + 0.1 * (overlap - overlap @ density @ overlap)
            np.testing.assert_allclose(fock_k, expected, atol=1e-12)
            shifts.append(True)
        return original_diag(fock_k, orthogonalizer)
    monkeypatch.setattr(module, '_diag_in_orth_basis', diagonalize)

    result = getattr(module, f'run_{family}_periodic_multi_k_ewald3d')(
        system, basis, kmesh, options, grid_shape=8,
        auto_optimize_truncation=False, progress=False, verbose=0,
    )
    assert len(seeds) == 1
    assert len(iterations) == 2
    assert len(shifts) == 12
    if family == 'uks':
        assert len(reciprocal_exchange) == 4
    assert np.isfinite(result.energy)


@pytest.mark.parametrize('method',['rhf','rks','uhf','uks','rohf','roks'])
def test_ewald_complete_multik_read(method):
    import importlib
    from vibeqc.guess_read import resolve_periodic_read_density_k_closed, resolve_periodic_read_densities_k_open
    opened=method not in ('rhf','rks');ks=method.endswith('ks')
    atoms=[vq.Atom(1,[0,0,0])]
    if not opened: atoms.append(vq.Atom(1,[0,0,1.4]))
    system=vq.PeriodicSystem(3,np.eye(3)*10.,atoms,multiplicity=2 if opened else 1)
    basis=vq.BasisSet(system.unit_cell_molecule(),'sto-3g');mesh=vq.monkhorst_pack(system,[3,1,1])
    opts=vq.PeriodicKSOptions() if ks else vq.PeriodicRHFOptions()
    opts.initial_guess=vq.InitialGuess.SAD;opts.max_iter=6;opts.use_diis=False;opts.damping=0
    opts.lattice_opts.cutoff_bohr=12.;opts.lattice_opts.nuclear_cutoff_bohr=12.
    if ks:opts.functional='pbe'
    mod=importlib.import_module('vibeqc.periodic_'+method+'_multi_k_ewald')
    driver=getattr(mod,'run_'+method+'_periodic_multi_k_ewald3d')
    kw=dict(progress=False,auto_optimize_truncation=False)
    source=driver(system,basis,mesh,opts,**kw)
    loader=resolve_periodic_read_densities_k_open if opened else resolve_periodic_read_density_k_closed
    ds=loader(read_from=source,basis=basis,system=system,kmesh=mesh)
    opts.initial_guess=vq.InitialGuess.READ
    result=driver(system,basis,mesh,opts,initial_density_k=ds,**kw)
    assert result.converged
    assert result.guess_selection.effective==vq.InitialGuess.READ
    assert result.energy==pytest.approx(source.energy,abs=1e-8)


@pytest.mark.parametrize("method,route", [("ROHF", "gdf"), ("ROHF", "bipole"),
                                          ("ROKS", "bipole"), ("UHF", "rijcosx"),
                                          ("UKS", "gpw")])
@pytest.mark.parametrize("nk", [1, 3])
def test_public_open_shell_read_preserves_spin_state(tmp_path, method, route, nk):
    system = vq.PeriodicSystem(3, 12 * np.eye(3), [vq.Atom(1, [6., 6., 6.])], multiplicity=2)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kwargs = dict(method=method, jk_method=route, kpoints=[nk, 1, 1],
                  max_iter=40, convergence="off", output_qvf=False,
                  write_molden_file=False, write_density=False, write_xyz_file=False,
                  write_cif_file=False, write_xsf_structure_file=False,
                  write_population_file=False, citations=False, progress=False)
    if method.endswith("KS"):
        kwargs["functional"] = "lda"
    if route == "gpw":
        kwargs["cutoff_ha"] = 8.
    elif route in ("gdf", "rijcosx"):
        kwargs["rsgdf_ke_cutoff"] = 12.
    source = vq.run_periodic_job(system, basis, initial_guess="HCORE", output=tmp_path / "source", **kwargs)
    result = vq.run_periodic_job(system, basis, initial_guess="READ", read_from=source,
                                 output=tmp_path / "restart", **kwargs)
    assert source.converged and result.converged
    assert result.energy == pytest.approx(source.energy, abs=1e-8)
    assert result.guess_selection.requested == vq.InitialGuess.READ
    assert result.guess_selection.effective == vq.InitialGuess.READ
    assert result.guess_selection.transport == vq.InitialGuess.READ
    from vibeqc.guess_read import resolve_periodic_read_densities_k_open
    mesh = vq.monkhorst_pack(system, [nk, 1, 1])
    pair = resolve_periodic_read_densities_k_open(read_from=result, basis=basis, system=system, kmesh=mesh)
    metrics = np.asarray(result.overlap)
    if metrics.ndim == 2:
        metrics = metrics[None]
    for blocks, population in zip(pair, (1, 0)):
        count = sum(w * np.trace(d @ metric) for w, d, metric in zip(mesh.weights, blocks, metrics))
        assert count == pytest.approx(population, abs=1e-10)


@pytest.mark.parametrize("route", ["gpw", "gapw"])
def test_public_closed_grid_multik_auto_and_read(tmp_path, route):
    from vibeqc.guess_read import resolve_periodic_read_density_k_closed
    system = vq.PeriodicSystem(3, 10 * np.eye(3),
                              [vq.Atom(1, [5., 5., 4.3]), vq.Atom(1, [5., 5., 5.7])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    mesh = vq.monkhorst_pack(system, [1, 1, 3])
    kwargs = dict(method="RKS", functional="lda", jk_method=route, kpoints=[1, 1, 3],
                  cutoff_ha=8., max_iter=40, convergence="off", output_qvf=False,
                  write_molden_file=False, write_density=False, write_xyz_file=False,
                  write_cif_file=False, write_xsf_structure_file=False,
                  write_population_file=False, citations=False, progress=False)
    source = vq.run_periodic_job(system, basis, output=tmp_path / "source", **kwargs)
    assert source.guess_selection.requested == vq.InitialGuess.AUTO
    assert source.guess_selection.effective == vq.InitialGuess.SAD
    blocks = resolve_periodic_read_density_k_closed(
        read_from=source, basis=basis, system=system, kmesh=mesh)
    np.testing.assert_allclose(blocks, source.density_k, atol=1e-13)
    assert sum(w * np.trace(d @ s).real for w, d, s in
               zip(mesh.weights, blocks, source.overlap_k)) == pytest.approx(2., abs=1e-10)
    result = vq.run_periodic_job(system, basis, initial_guess="READ", read_from=source,
                                 output=tmp_path / "restart", **kwargs)
    assert source.converged and result.converged
    assert result.guess_selection.effective == vq.InitialGuess.READ
    assert result.energy == pytest.approx(source.energy, abs=1e-8)


@pytest.mark.parametrize("populations", [(2., 0.), (0., 2.), (1.5, .5), (1., 1.)])
@pytest.mark.parametrize("target", [(1, 1), (2, 0), (2, 1)])
def test_multik_spin_restart_normalization_matches_native_global_metric(populations, target):
    from scipy.linalg import block_diag
    from vibeqc.guess import normalize_spin_density_k_guess, normalize_spin_density_guess, normalize_density_guess

    weights = np.array([.2, .3, .5])
    metrics = [np.array([[1. + .1 * k, .05j], [-.05j, 1.4]]) for k in range(3)]
    # Unequal populations at different k points, with non-real AO phases.
    vectors = [np.array([1., (.2 + .1 * k) * 1j]) for k in range(3)]
    a = [populations[0] * (k + 1) * np.outer(v, v.conj()) for k, v in enumerate(vectors)]
    b = [populations[1] * (3 - k) * np.outer(v.conj(), v) for k, v in enumerate(vectors)]
    total = sum(w * np.trace((da + db) @ s).real for w, da, db, s in zip(weights, a, b, metrics))
    a, b = [[d * 2 / total for d in blocks] for blocks in (a, b)]
    da, db = normalize_spin_density_k_guess(a, b, metrics, weights, *target)
    # A small explicit block metric is an independent native oracle. The
    # production adapter must never allocate this O((nk*nao)^2) matrix.
    global_metric = block_diag(*(w * s for w, s in zip(weights, metrics)))
    if all(populations):
        # Populated READ channels retain their separate spatial patterns.
        reference = [normalize_density_guess(block_diag(*blocks), global_metric, count)
                     for blocks, count in zip((a, b), target)]
    else:
        # With one empty channel, native spin transfer gives the same seed.
        reference = normalize_spin_density_guess(block_diag(*a), block_diag(*b), global_metric, *target)
    for actual, expected, count in zip((da, db), reference, target):
        np.testing.assert_allclose(block_diag(*actual), expected, atol=1e-12)
        assert sum(w * np.trace(d @ s).real for w, d, s in zip(weights, actual, metrics)) == pytest.approx(count, abs=1e-12)
        for density in actual:
            np.testing.assert_allclose(density, density.conj().T, atol=1e-13)
            assert np.linalg.eigvalsh(density).min() > -1e-12
    if not all(populations):
        for old_a, old_b, new_a, new_b in zip(a, b, da, db):
            np.testing.assert_allclose(new_a + new_b, (old_a + old_b) * sum(target) / 2, atol=1e-12)


@pytest.mark.parametrize("method,route", [
    ("UHF", "gdf"), ("UKS", "gdf"), ("ROHF", "gdf"),
    ("UHF", "bipole"), ("UKS", "bipole"), ("ROKS", "bipole"),
    ("UKS", "gpw"), ("ROKS", "gpw"),
])
@pytest.mark.parametrize("nk", [1, 3])
def test_public_periodic_read_from_fully_polarized_source(tmp_path, method, route, nk):
    """Changing multiplicity must populate an empty spin channel before SCF."""
    atoms = [vq.Atom(1, [6., 6., 5.3]), vq.Atom(1, [6., 6., 6.7])]
    source_system = vq.PeriodicSystem(3, 12 * np.eye(3), atoms, multiplicity=3)
    target_system = vq.PeriodicSystem(3, 12 * np.eye(3), atoms)
    basis = vq.BasisSet(source_system.unit_cell_molecule(), "sto-3g")
    kwargs = dict(method=method, jk_method=route, kpoints=[nk, 1, 1],
                  max_iter=40, convergence="off", output_qvf=False,
                  write_molden_file=False, write_density=False, write_xyz_file=False,
                  write_cif_file=False, write_xsf_structure_file=False,
                  write_population_file=False, citations=False, progress=False)
    if method.endswith("KS"):
        kwargs["functional"] = "lda"
    if route == "gpw":
        kwargs["cutoff_ha"] = 8.
    elif route == "gdf":
        kwargs.update(rsgdf_ke_cutoff=12., rsgdf_tail_ke_cutoff=0.)
    source = vq.run_periodic_job(source_system, basis, initial_guess="HCORE",
                                 output=tmp_path / "source", **kwargs)
    target_basis = vq.BasisSet(target_system.unit_cell_molecule(), "sto-3g")
    result = vq.run_periodic_job(target_system, target_basis, initial_guess="READ", read_from=source,
                                 output=tmp_path / "restart", **kwargs)
    reference = vq.run_periodic_job(target_system, target_basis, initial_guess="HCORE",
                                    output=tmp_path / "reference", **kwargs)
    assert source.converged and result.converged and reference.converged
    assert result.energy == pytest.approx(reference.energy, abs=1e-8)
    assert result.guess_selection.effective == vq.InitialGuess.READ
    from vibeqc.guess_read import resolve_periodic_read_densities_k_open
    mesh = vq.monkhorst_pack(target_system, [nk, 1, 1])
    pair = resolve_periodic_read_densities_k_open(
        read_from=result, basis=target_basis, system=target_system, kmesh=mesh,
    )
    metrics = np.asarray(result.overlap)
    if metrics.ndim == 2:
        metrics = metrics[None]
    for blocks in pair:
        count = sum(w * np.trace(d @ s).real for w, d, s in zip(mesh.weights, blocks, metrics))
        assert count == pytest.approx(1., abs=1e-10)


def test_multik_spin_restart_preserves_valid_broken_symmetry():
    from vibeqc.guess import normalize_spin_density_k_guess

    a = [np.diag([.4, 0.]), np.diag([1.2, 0.])]
    b = [np.diag([0., 1.6]), np.diag([0., .8])]
    da, db = normalize_spin_density_k_guess(a, b, [np.eye(2)] * 2, [.25, .75], 1, 1)
    np.testing.assert_array_equal(da, a)
    np.testing.assert_array_equal(db, b)


def test_gamma_read_normalization_retains_populated_spin_patterns():
    system = vq.PeriodicSystem(3, 12 * np.eye(3), [
        vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 3.]),
    ])
    mol = system.unit_cell_molecule()
    basis = vq.BasisSet(mol, "sto-3g")
    overlap = np.asarray(vq.compute_overlap(basis))
    alpha, beta = np.diag([1.5, 0.]), np.diag([0., .5])
    da, db = initial_densities_open_shell(
        mol, basis, 1, 1, vq.InitialGuess.READ, is_periodic=True,
        overlap=overlap, read_density_alpha=alpha, read_density_beta=beta,
    )
    np.testing.assert_allclose(da, alpha / np.trace(alpha @ overlap), atol=1e-12)
    np.testing.assert_allclose(db, beta / np.trace(beta @ overlap), atol=1e-12)


@pytest.mark.parametrize("nk", [1, 3])
def test_lattice_spin_restart_populates_empty_channel(nk):
    from vibeqc.guess import normalize_periodic_lattice_spin_restart
    from vibeqc.periodic_k_density import real_space_density_from_per_k_density
    from vibeqc.pbc_bipole_common import home_cell_block

    system = vq.PeriodicSystem(3, np.eye(3) * 10., [vq.Atom(2, [0., 0., 0.])])
    mesh = vq.monkhorst_pack(system, [nk, 1, 1])
    cells = vq.direct_lattice_cells(system, .1 if nk == 1 else 10.1)
    phase = np.array([[0., .1j], [-.1j, 0.]])
    blocks = [np.eye(2) + np.sin(k[0] * 10.) * phase for k in mesh.kpoints]
    alpha = real_space_density_from_per_k_density(blocks, mesh, cells)
    beta = real_space_density_from_per_k_density([np.zeros((2, 2))] * nk, mesh, cells)
    pair = normalize_periodic_lattice_spin_restart(
        alpha, beta, [np.eye(2)] * nk, mesh.weights, 1, 1, mesh,
    )
    for density in pair:
        actual = ([home_cell_block(density)] if nk == 1 else
                  bvk_torus_density_matrices(density, list(mesh.kpoints), mesh.mesh))
        np.testing.assert_allclose(actual, .5 * np.asarray(blocks), atol=1e-12)


@pytest.mark.parametrize("fault", ["missing_spin_block", "nonfinite", "negative", "nonhermitian", "weights"])
def test_multik_spin_restart_rejects_invalid_empty_target(fault):
    from vibeqc.guess import normalize_spin_density_k_guess

    a, b = [np.eye(2, dtype=complex)], [np.eye(2, dtype=complex)]
    weights = [1.]
    if fault == "missing_spin_block":
        b = []
    elif fault == "nonfinite":
        b[0][0, 0] = np.nan
    elif fault == "negative":
        b[0][0, 0] = -1.
    elif fault == "nonhermitian":
        b[0][0, 1] = 1j
    else:
        weights = [-1.]
    with pytest.raises(ValueError):
        normalize_spin_density_k_guess(a, b, [np.eye(2)], weights, 0, 0)


def test_closed_multik_read_prefers_physical_density_to_orbitals():
    from types import SimpleNamespace
    from vibeqc.guess_read import resolve_periodic_read_density_k_closed
    physical = [np.diag([0.7, 0.3]).astype(complex)] * 3
    source = SimpleNamespace(density_k=physical, mo_coeffs_k=[np.eye(2)] * 3,
                             occupations_k=[np.array([2., 0.])] * 3)
    actual = resolve_periodic_read_density_k_closed(read_from=source, expected_n_k=3, n_basis=2)
    np.testing.assert_array_equal(actual, physical)


@pytest.mark.parametrize("mesh", [(1, 1, 1), (3, 1, 1)])
@pytest.mark.parametrize("method", ["RHF", "UHF", "RKS", "UKS"])
def test_public_periodic_ecp_reaches_valence_sad_builder(tmp_path, monkeypatch, mesh, method):
    """Actual public SCF must transport ECP data into the selected builder."""
    system = vq.PeriodicSystem(3, np.eye(3) * 16., [
        vq.Atom(11, [0., 0., 0.]), vq.Atom(1, [0., 0., 3.5]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "lanl2dz")
    spin = method in ("UHF", "UKS")
    name = "initial_densities_open_shell" if spin else "initial_density_closed_shell"
    builder = getattr(periodic_k_gdf, name)
    reached = []
    def probe(*args, **kwargs):
        context = kwargs["ecp_context"]
        assert context.active and context.total_ncore == 10
        assert list(context.effective_charges) == [1., 1.]
        value = builder(*args, **kwargs)
        metric = sum(w * np.asarray(s) for w, s in zip(kwargs["weights"], kwargs["overlap"]))
        densities = value if spin else (value,)
        for density in densities:
            assert np.trace(density @ metric).real == pytest.approx(1 if spin else 2, abs=1e-11)
            np.testing.assert_allclose(density, density.conj().T, atol=1e-13)
        reached.append(context)
        return value
    monkeypatch.setattr(periodic_k_gdf, name, probe)
    result = vq.run_periodic_job(
        system, basis, method=method, functional="lda" if method.endswith("KS") else None,
        jk_method="gdf", aux_basis="def2-svp-jk", kpoints=mesh, initial_guess="AUTO", max_iter=80,
        rsgdf_ke_cutoff=12., rsgdf_tail_ke_cutoff=0.,
        output=str(tmp_path / f"{method}-{mesh[0]}"), output_qvf=False,
    )
    assert reached and result.converged
    assert result.guess_selection.requested == vq.InitialGuess.AUTO
    assert result.guess_selection.effective == vq.InitialGuess.SAD


@pytest.mark.parametrize("guess", ["SAP", "MINAO"])
@pytest.mark.parametrize("mesh", [(1,1,1), (3,1,1)])
def test_public_periodic_ecp_sap_minao(tmp_path, guess, mesh):
    system = vq.PeriodicSystem(3, np.eye(3) * 20., [vq.Atom(11, [0.,0.,0.]), vq.Atom(1,[0.,0.,3.5])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "lanl2dz")
    result = vq.run_periodic_job(system, basis, method="RHF", jk_method="gdf",
        kpoints=mesh, initial_guess=guess, aux_basis="def2-svp-jk", output=str(tmp_path/'ecp'),
        output_qvf=False, max_iter=80, rsgdf_ke_cutoff=12., rsgdf_tail_ke_cutoff=0.)
    assert result.converged
    assert result.guess_selection.effective == getattr(vq.InitialGuess,guess)


@pytest.mark.parametrize("method", ["gamma_rhf", "krhf", "krks", "dispatch_gamma_rhf", "dispatch_krhf", "dispatch_krks"])
def test_direct_native_periodic_ecp_sad_and_hcore_share_energy_basin(method):
    dispatch = method.startswith("dispatch_")
    method = method.removeprefix("dispatch_")
    from vibeqc.periodic_runner import _resolve_ecp_data

    system = vq.PeriodicSystem(3, np.eye(3) * 20., [
        vq.Atom(11, [0., 0., 0.]), vq.Atom(1, [0., 0., 3.5]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "lanl2dz")
    blocks, centers, charges, ncore = _resolve_ecp_data(system, basis)
    options_cls = {"gamma_rhf": vq.PeriodicRHFOptions,
                   "krhf": vq.PeriodicSCFOptions, "krks": vq.PeriodicKSOptions}[method]
    mesh = vq.monkhorst_pack(system, [1, 1, 1])
    results = []
    for guess in (vq.InitialGuess.SAD, vq.InitialGuess.HCORE, vq.InitialGuess.SAP, vq.InitialGuess.MINAO):
        opts = options_cls()
        opts.ecp_primitive_blocks = blocks
        opts.ecp_home_centers = centers
        opts.ecp_effective_charges = charges
        opts.ecp_total_ncore = ncore
        opts.initial_guess = guess
        opts.max_iter = 100
        opts.conv_tol_energy = 1e-10
        opts.lattice_opts.cutoff_bohr = 5.
        opts.lattice_opts.nuclear_cutoff_bohr = 5.
        if method == "krks":
            opts.functional = "lda"
        if method == "gamma_rhf":
            driver = vq.run_rhf_periodic_gamma_scf if dispatch else vq._vibeqc_core.run_rhf_periodic_gamma
            result = driver(system, basis, opts)
        elif method == "krhf":
            driver = vq.run_rhf_periodic_scf if dispatch else vq._vibeqc_core.run_rhf_periodic
            result = driver(system, basis, mesh, opts)
        else:
            driver = vq.run_rks_periodic_scf if dispatch else vq._vibeqc_core.run_rks_periodic
            result = driver(system, basis, mesh, opts)
        assert result.converged
        assert result.guess_selection.effective == guess
        results.append(result)
    assert all(r.energy == pytest.approx(results[0].energy, abs=1e-8) for r in results)


@pytest.mark.parametrize("family,multi", [
    ("rhf", False), ("rks", False), ("uhf", False), ("uks", False),
    ("rohf", False), ("rhf", True), ("rks", True), ("uhf", True),
    ("uks", True), ("rohf", True), ("roks", True),
])
def test_direct_ewald_ecp_guess_fails_before_integrals(family, multi):
    from vibeqc.periodic_runner import _resolve_ecp_data
    system = vq.PeriodicSystem(3, np.eye(3) * 16., [vq.Atom(30, [0., 0., 0.])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "lanl2dz")
    blocks, centers, charges, ncore = _resolve_ecp_data(system, basis)
    opts = vq.PeriodicKSOptions() if family.endswith("ks") else vq.PeriodicRHFOptions()
    opts.ecp_primitive_blocks = blocks
    opts.ecp_home_centers = centers
    opts.ecp_effective_charges = charges
    opts.ecp_total_ncore = ncore
    opts.initial_guess = vq.InitialGuess.SAD
    module = importlib.import_module(f"vibeqc.periodic_{family}_{'multi_k_' if multi else ''}ewald")
    driver = getattr(module, f"run_{family}_periodic_{'multi_k' if multi else 'gamma'}_ewald3d")
    args = (system, None, vq.monkhorst_pack(system, [3, 1, 1]), opts) if multi else (system, None, opts)
    with pytest.raises(NotImplementedError, match="ECP Hamiltonians.*Python Ewald"):
        driver(*args)


def test_retained_k_restart_checks_complex_time_reversal_on_short_cell_list():
    from vibeqc.guess import periodic_restart_lattice_density
    from vibeqc.pbc_bipole_common import home_cell_block
    system = vq.PeriodicSystem(3, np.eye(3) * 10., [
        vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 1.4]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 4.
    cells = compute_overlap_lattice(basis, system, opts).cells
    assert len(cells) == 1
    mesh = vq.monkhorst_pack(system, [3, 1, 1])
    d = np.array([[1., .1j], [-.1j, 1.]])
    # Generate the conjugate pair from coordinates, without assuming the
    # mesh storage order or the position of Gamma.
    blocks = [np.eye(2, dtype=complex)
              + np.sin(k[0] * 10.) * (d - np.eye(2)) for k in mesh.kpoints]
    result = periodic_restart_lattice_density(
        blocks, [np.eye(2)] * 3, mesh.weights, 2, mesh, cells,
        retained_k_system=system,
    )
    np.testing.assert_allclose(home_cell_block(result), np.eye(2), atol=1e-13)
    with pytest.raises(NotImplementedError, match="time-reversal-compatible"):
        periodic_restart_lattice_density(
            [d, d, d], [np.eye(2)] * 3, mesh.weights, 2, mesh, cells,
            retained_k_system=system,
        )


@pytest.mark.parametrize("method", ["ROHF", "ROKS"])
def test_restricted_open_ewald_refuses_unimplemented_spinlock(method):
    from vibeqc.guess import select_periodic_driver_guess
    system, _ = _h_box()
    options = vq.PeriodicRHFOptions() if method == "ROHF" else vq.PeriodicKSOptions()
    options.spinlock_mode = vq.SpinlockMode.SPIN_SCHEDULE
    options.spinlock_iterations = 3
    with pytest.raises(NotImplementedError, match="SPINLOCK"):
        select_periodic_driver_guess(
            system, options, route="ewald", method=method, driver="Ewald", multi_k=True)
