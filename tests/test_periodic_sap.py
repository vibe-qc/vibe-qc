"""Periodic SAP guess potential — lattice-summed Ewald-split SAP.

``compute_vsap_lattice`` builds the periodic SAP (superposition-of-atomic-
potentials) guess potential V^SAP_μν(g). Each molecular SAP term
c_i·erf(ω_i r)/r is Ewald-split with a moderate parameter η into an
analytical short-range part (two erfc pieces, libint ``erfc_nuclear``,
real-space lattice-summed — this resolves the sharp near-nucleus structure)
plus a smooth long-range part (erf(η r)/r with the net effective charges
Q_A = Σ_i c_i, grid-quadratured). The three pieces reconstruct the molecular
+Σ c_i erf(ω_i)/r exactly; the G=0 background is a constant shift that leaves
the guess density invariant.

Validated here at the integral level against the analytical *molecular* SAP
matrix (``compute_sap_potential_molecular``): the Γ-folded periodic V^SAP is
real-symmetric, attractive, and reproduces the molecular potential in the
isolated (large-cell) limit, and the one-shot guess Fock F_SAP = T + V_SAP is
well-formed. End-to-end (SAP seeds a periodic SCF into the same basin as SAD)
lands with the periodic-driver wiring, including the closed-shell Python
periodic Ewald/GDF/BIPOLE routes.
"""
from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pytest

import vibeqc as vq
import vibeqc.guess as periodic_guess
import vibeqc.pbc_bipole as pbc_bipole
import vibeqc.pbc_bipole_rks as pbc_bipole_rks
import vibeqc.periodic_cosx_k as periodic_cosx_k
import vibeqc.periodic_grid as periodic_grid
import vibeqc.periodic_k_gdf as periodic_k_gdf
import vibeqc.periodic_rhf_multi_k_ewald as periodic_rhf_multi_k_ewald
import vibeqc.periodic_rks_multi_k_ewald as periodic_rks_multi_k_ewald
from vibeqc._vibeqc_core import (
    bloch_sum,
    compute_kinetic_lattice,
    compute_overlap_lattice,
    compute_sap_potential_molecular,
    compute_vsap_lattice,
)
from vibeqc.periodic_grid import build_periodic_becke_grid
from vibeqc.progress import ProgressLogger

sla = pytest.importorskip("scipy.linalg")


def _gamma(lat_set):
    return np.real(bloch_sum(lat_set, np.zeros(3)))


def _lat_opts(cutoff: float = 18.0, nuc: float = 22.0):
    lo = vq.LatticeSumOptions()
    lo.cutoff_bohr = cutoff
    lo.nuclear_cutoff_bohr = nuc
    return lo


def _lih(a_bohr: float):
    lat = a_bohr * np.eye(3)
    atoms = [vq.Atom(3, [0.0, 0.0, 0.0]),
             vq.Atom(1, [a_bohr / 2, a_bohr / 2, a_bohr / 2])]
    system = vq.PeriodicSystem(3, lat, atoms, charge=0, multiplicity=1)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _he_box():
    system = vq.PeriodicSystem(
        3,
        10.0 * np.eye(3),
        [vq.Atom(2, [0.0, 0.0, 0.0])],
        charge=0,
        multiplicity=1,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _stub_multik_gdf_setup(monkeypatch):
    """Keep max_iter=0 routing probes independent of the expensive J/K fit."""
    def zero_lpq_cache(
        _system,
        basis,
        _aux,
        kpoints_cart,
        need_k_pairs,
        **_kwargs,
    ):
        n_k = len(kpoints_cart)
        pairs = (
            ((i, j) for i in range(n_k) for j in range(n_k))
            if need_k_pairs
            else ((i, i) for i in range(n_k))
        )
        return {
            pair: np.zeros((1, basis.nbasis, basis.nbasis), dtype=complex)
            for pair in pairs
        }

    class ZeroIterationCosx:
        def __init__(self, *_args, **_kwargs):
            self.cells = ()
            self.caches = SimpleNamespace(n_deltas=0)

    monkeypatch.setattr(
        periodic_k_gdf,
        "_build_rsgdf_lpq_cache_shared_q",
        zero_lpq_cache,
    )
    monkeypatch.setattr(periodic_cosx_k, "KPointCosxK", ZeroIterationCosx)


class _GuessOptionsProxy:
    """Delegate every SCF option except a string-valued guess selector."""

    def __init__(self, base, initial_guess):
        object.__setattr__(self, "_base", base)
        object.__setattr__(self, "initial_guess", initial_guess)

    def __getattr__(self, name):
        return getattr(self._base, name)

    def __setattr__(self, name, value):
        if name in {"_base", "initial_guess"}:
            object.__setattr__(self, name, value)
        else:
            setattr(self._base, name, value)


def test_vsap_gamma_symmetric_and_attractive():
    # On a normal-sized cell the Γ-folded V^SAP is symmetric to the grid floor.
    # (For very tight cells the smooth long-range grid quadrature limits the
    # symmetry, exactly as the production compute_nuclear_lattice_ewald — see
    # test_vsap_symmetry_no_worse_than_nuclear_ewald — but it vanishes as the
    # cell opens up; here ~1e-4 at a=12 bohr.)
    system, basis = _lih(12.0)
    grid = build_periodic_becke_grid(system)
    V = _gamma(compute_vsap_lattice(basis, system, grid,
                                    "sap_helfem_large", _lat_opts(cutoff=20.0,
                                                                  nuc=25.0)))
    assert np.all(np.isfinite(V))
    assert np.max(np.abs(V - V.T)) < 1e-3
    # The core (Li 1s) on-site element is strongly attractive. Full
    # attractiveness in the isolated limit is checked by the molecular-parity
    # test; the near-neutral H site (SAP net charge ≈ −0.1) can ride slightly
    # above zero under the periodic Madelung field of the net effective charges.
    assert np.diag(V)[0] < -0.5


def test_vsap_symmetry_no_worse_than_nuclear_ewald():
    """V^SAP and the production grid-Ewald nuclear attraction share the same
    Becke-grid long-range quadrature, so on a tight cell their Γ-fold symmetry
    is limited by the same floor; V^SAP must be no worse than that reference."""
    from vibeqc._vibeqc_core import compute_nuclear_lattice_ewald
    system, basis = _lih(7.6)
    grid = build_periodic_becke_grid(system)
    lo = _lat_opts()
    asym_sap = np.max(np.abs(
        (V := _gamma(compute_vsap_lattice(basis, system, grid,
                                          "sap_helfem_large", lo))) - V.T))
    asym_ne = np.max(np.abs(
        (Vne := _gamma(compute_nuclear_lattice_ewald(basis, system, grid, lo)))
        - Vne.T))
    assert asym_sap <= asym_ne + 1e-9


def test_vsap_matches_molecular_in_large_cell():
    """In a large isolated cell the periodic V^SAP reproduces the analytical
    molecular SAP matrix (the remaining difference is the small periodic
    Madelung field of the net effective charges)."""
    a = 30.0
    lat = a * np.eye(3)
    system = vq.PeriodicSystem(
        3, lat, [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [3.0, 0, 0])],
        charge=0, multiplicity=1)
    mol = system.unit_cell_molecule()
    basis = vq.BasisSet(mol, "sto-3g")
    grid = build_periodic_becke_grid(system)
    lo = _lat_opts(cutoff=20.0, nuc=25.0)

    V_mol = np.asarray(compute_sap_potential_molecular(
        basis, mol, "sap_helfem_large"))
    V_per = _gamma(compute_vsap_lattice(basis, system, grid,
                                        "sap_helfem_large", lo))
    # Same (attractive) sign on every diagonal, and quantitative agreement to
    # the periodic-Madelung floor.
    assert np.all(np.diag(V_mol) < 0.0)
    assert np.all(np.diag(V_per) < 0.0)
    assert np.max(np.abs(V_per - V_mol)) < 0.15


def test_vsap_guess_fock_is_well_formed():
    """F_SAP = T + V_SAP diagonalised against the cell overlap has a real
    spectrum with a bound lowest state and yields a normalised guess density."""
    system, basis = _lih(7.6)
    grid = build_periodic_becke_grid(system)
    lo = _lat_opts()
    S = _gamma(compute_overlap_lattice(basis, system, lo))
    T = _gamma(compute_kinetic_lattice(basis, system, lo))
    V = _gamma(compute_vsap_lattice(basis, system, grid, "sap_helfem_large", lo))
    eps, C = sla.eigh(T + V, S)
    assert np.all(np.isfinite(eps))
    assert eps[0] < 0.0  # attractive SAP → bound lowest orbital
    n_elec = system.unit_cell_molecule().n_electrons()
    nocc = n_elec // 2
    D = 2.0 * C[:, :nocc] @ C[:, :nocc].T
    assert abs(np.trace(D @ S) - n_elec) < 1e-6
    assert np.max(np.abs(D - D.T)) < 1e-10


def test_periodic_sap_multik_reaches_same_basin_as_sad():
    """SAP is wired into the multi-k C++ RHF and RKS periodic drivers
    (F_SAP(k) = T(k) + V_SAP(k)). The guess does not change the converged
    minimum, so SAP and SAD reach the same energy. He (2x2x2 k-mesh) converges
    cleanly; the metal case (smearing) is a separate convergence matter."""
    def mk(driver, guess, opts):
        s = vq.PeriodicSystem(3, 10.0 * np.eye(3), [vq.Atom(2, [0, 0, 0])],
                              charge=0, multiplicity=1)
        b = vq.BasisSet(s.unit_cell_molecule(), "sto-3g")
        km = vq.monkhorst_pack(s, [2, 2, 2])
        opts.conv_tol_energy = 1e-9
        opts.initial_guess = guess
        r = driver(s, b, km, opts)
        assert r.converged, f"{driver.__name__}/{guess.name} did not converge"
        return r.energy

    # multi-k RHF (PeriodicSCFOptions)
    e_rhf_sap = mk(vq.run_rhf_periodic, vq.InitialGuess.SAP, vq.PeriodicSCFOptions())
    e_rhf_sad = mk(vq.run_rhf_periodic, vq.InitialGuess.SAD, vq.PeriodicSCFOptions())
    assert abs(e_rhf_sap - e_rhf_sad) < 1e-7

    # multi-k RKS/LDA (PeriodicKSOptions)
    def _ks():
        o = vq.PeriodicKSOptions()
        o.functional = "lda"
        return o
    e_rks_sap = mk(vq.run_rks_periodic, vq.InitialGuess.SAP, _ks())
    e_rks_sad = mk(vq.run_rks_periodic, vq.InitialGuess.SAD, _ks())
    assert abs(e_rks_sap - e_rks_sad) < 1e-7


def test_periodic_sap_python_gamma_ewald_reaches_same_basin_as_sad():
    """The Python Gamma-Ewald RHF/RKS drivers pass the periodic context needed
    by the SAP Fock-mode seam."""
    def rhf_energy(guess):
        system, basis = _he_box()
        opts = vq.PeriodicRHFOptions()
        opts.conv_tol_energy = 1e-9
        opts.initial_guess = guess
        result = vq.run_rhf_periodic_gamma_ewald3d(
            system, basis, opts, auto_optimize_truncation=False, verbose=0)
        assert result.converged, f"RHF/{guess.name} did not converge"
        return result.energy

    def rks_energy(guess):
        system, basis = _he_box()
        opts = vq.PeriodicKSOptions()
        opts.functional = "lda"
        opts.conv_tol_energy = 1e-9
        opts.initial_guess = guess
        result = vq.run_rks_periodic_gamma_ewald3d(
            system, basis, opts, auto_optimize_truncation=False, verbose=0)
        assert result.converged, f"RKS/{guess.name} did not converge"
        return result.energy

    assert abs(rhf_energy(vq.InitialGuess.SAP)
               - rhf_energy(vq.InitialGuess.SAD)) < 1e-7
    assert abs(rks_energy(vq.InitialGuess.SAP)
               - rks_energy(vq.InitialGuess.SAD)) < 1e-7


def test_periodic_sap_python_gdf_rhf_reaches_same_basin_as_sad():
    """The Python GDF RHF driver passes the periodic SAP context."""
    def energy(guess):
        system, basis = _he_box()
        opts = vq.PeriodicRHFOptions()
        opts.conv_tol_energy = 1e-9
        opts.initial_guess = guess
        result = vq.run_pbc_gdf_rhf(
            system, basis, opts, progress=False, verbose=0)
        assert result.converged, f"GDF RHF/{guess.name} did not converge"
        return result.energy

    assert abs(energy(vq.InitialGuess.SAP)
               - energy(vq.InitialGuess.SAD)) < 1e-7


def test_periodic_sap_python_bipole_reaches_same_basin_as_sad():
    """The Python BIPOLE RHF/RKS drivers pass the periodic SAP context."""
    def rhf_energy(guess):
        system, basis = _he_box()
        kmesh = vq.monkhorst_pack(system, [1, 1, 1])
        opts = vq.PeriodicRHFOptions()
        opts.conv_tol_energy = 1e-9
        opts.initial_guess = guess
        result = vq.run_pbc_bipole_rhf(
            system, basis, kmesh, opts, progress=False, verbose=0)
        assert result.converged, f"BIPOLE RHF/{guess.name} did not converge"
        return result.energy

    def rks_energy(guess):
        system, basis = _he_box()
        kmesh = vq.monkhorst_pack(system, [1, 1, 1])
        opts = vq.PeriodicKSOptions()
        opts.functional = "lda"
        opts.conv_tol_energy = 1e-9
        opts.initial_guess = guess
        result = vq.run_pbc_bipole_rks(
            system, basis, kmesh, opts, progress=False, verbose=0)
        assert result.converged, f"BIPOLE RKS/{guess.name} did not converge"
        return result.energy

    assert abs(rhf_energy(vq.InitialGuess.SAP)
               - rhf_energy(vq.InitialGuess.SAD)) < 1e-7
    assert abs(rks_energy(vq.InitialGuess.SAP)
               - rks_energy(vq.InitialGuess.SAD)) < 1e-7


@pytest.mark.parametrize("k_exchange", ["gdf", "cosx"])
@pytest.mark.parametrize(
    ("driver", "functional"),
    [
        (periodic_k_gdf.run_krhf_periodic_gdf, None),
        (periodic_k_gdf.run_krks_periodic_gdf, "pbe0"),
    ],
    ids=["krhf", "krks"],
)
def test_periodic_sap_python_multik_gdf_closed_shell_is_executed(
    monkeypatch,
    k_exchange,
    driver,
    functional,
):
    """True multi-k GDF/RIJCOSX must build and report the SAP seed."""
    _stub_multik_gdf_setup(monkeypatch)
    sap_calls = []
    oneel_opts = vq.LatticeSumOptions()
    monkeypatch.setattr(
        periodic_k_gdf,
        "_oneel_lattice_opts",
        lambda *_args, **_kwargs: oneel_opts,
    )

    def tracked_vsap(basis, system, _grid, sap_basis_name, lattice_opts):
        sap_calls.append((sap_basis_name, lattice_opts))
        # Integral-level SAP tests above cover the real builder. Returning a
        # lattice matrix here keeps this public-route execution probe fast.
        return compute_kinetic_lattice(basis, system, lattice_opts)

    monkeypatch.setattr(periodic_guess, "compute_vsap_lattice", tracked_vsap)
    system, basis = _he_box()
    options = (
        vq.PeriodicKSOptions() if functional else vq.PeriodicRHFOptions()
    )
    options.initial_guess = vq.InitialGuess.SAP
    options.max_iter = 0
    stream = io.StringIO()
    kwargs = {"functional": functional} if functional else {}
    result = driver(
        system,
        basis,
        (2, 1, 1),
        options,
        use_compcell=True,
        k_exchange=k_exchange,
        rcut_strategy="flat",
        check_energy_sanity=False,
        progress=ProgressLogger(stream=stream, verbose=2),
        **kwargs,
    )

    assert len(sap_calls) == 1
    assert sap_calls[0][0] == "sap_helfem_large"
    assert sap_calls[0][1] is oneel_opts
    assert "initial guess: SAP" in stream.getvalue()
    for density in result.density:
        np.testing.assert_allclose(
            density, density.conj().T, rtol=0.0, atol=1e-10
        )
    electron_count = sum(
        float(weight)
        * float(np.trace(density @ overlap).real)
        for weight, density, overlap in zip(
            result.kpoint_weights,
            result.density,
            result.overlap,
        )
    )
    assert electron_count == pytest.approx(2.0, abs=1e-8)


@pytest.mark.parametrize("k_exchange", ["gdf", "cosx"])
@pytest.mark.parametrize(
    ("driver", "functional"),
    [
        (periodic_k_gdf.run_kuhf_periodic_gdf, None),
        (periodic_k_gdf.run_kuks_periodic_gdf, "pbe0"),
    ],
    ids=["kuhf", "kuks"],
)
def test_periodic_sap_python_multik_gdf_open_shell_is_executed(
    monkeypatch,
    k_exchange,
    driver,
    functional,
):
    """Open-shell GDF/RIJCOSX fills the common SAP bands per spin."""
    _stub_multik_gdf_setup(monkeypatch)
    sap_calls = []
    oneel_opts = vq.LatticeSumOptions()
    monkeypatch.setattr(
        periodic_k_gdf,
        "_oneel_lattice_opts",
        lambda *_args, **_kwargs: oneel_opts,
    )

    def tracked_vsap(basis, system, _grid, sap_basis_name, lattice_opts):
        sap_calls.append((sap_basis_name, lattice_opts))
        return compute_kinetic_lattice(basis, system, lattice_opts)

    monkeypatch.setattr(periodic_guess, "compute_vsap_lattice", tracked_vsap)
    system = vq.PeriodicSystem(
        3,
        10.0 * np.eye(3),
        [vq.Atom(1, [0.0, 0.0, 0.0])],
        charge=0,
        multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = (
        vq.PeriodicKSOptions() if functional else vq.PeriodicRHFOptions()
    )
    options.initial_guess = vq.InitialGuess.SAP
    options.max_iter = 0
    stream = io.StringIO()
    kwargs = {"functional": functional} if functional else {}
    result = driver(
        system,
        basis,
        (2, 1, 1),
        options,
        k_exchange=k_exchange,
        rcut_strategy="flat",
        check_energy_sanity=False,
        progress=ProgressLogger(stream=stream, verbose=2),
        **kwargs,
    )

    assert len(sap_calls) == 1
    assert sap_calls[0][0] == "sap_helfem_large"
    assert sap_calls[0][1] is oneel_opts
    assert "initial guess: SAP" in stream.getvalue()
    for density in (*result.density_alpha, *result.density_beta):
        np.testing.assert_allclose(
            density, density.conj().T, rtol=0.0, atol=1e-10
        )

    def electron_count(densities):
        return sum(
            float(weight)
            * float(np.trace(density @ overlap).real)
            for weight, density, overlap in zip(
                result.kpoint_weights,
                densities,
                result.overlap,
            )
        )

    assert electron_count(result.density_alpha) == pytest.approx(1.0, abs=1e-8)
    assert electron_count(result.density_beta) == pytest.approx(0.0, abs=1e-8)


def test_periodic_sap_multik_gdf_caller_density_takes_precedence(monkeypatch):
    """An explicit per-k restart remains authoritative over options.SAP."""
    _stub_multik_gdf_setup(monkeypatch)

    def unexpected_vsap(*_args, **_kwargs):
        pytest.fail("SAP builder must not run when initial_density_k is supplied")

    monkeypatch.setattr(
        periodic_guess,
        "compute_vsap_lattice",
        unexpected_vsap,
    )
    system, basis = _he_box()
    options = vq.PeriodicRHFOptions()
    options.initial_guess = vq.InitialGuess.SAP
    options.max_iter = 0
    stream = io.StringIO()
    density_k = [
        2 * np.eye(basis.nbasis, dtype=complex)
        for _ in range(2)
    ]
    result = periodic_k_gdf.run_krhf_periodic_gdf(
        system,
        basis,
        (2, 1, 1),
        options,
        use_compcell=True,
        rcut_strategy="flat",
        initial_density_k=density_k,
        check_energy_sanity=False,
        progress=ProgressLogger(stream=stream, verbose=2),
    )

    assert len(result.density) == 2
    assert result.guess_selection.transport == vq.InitialGuess.READ


@pytest.mark.parametrize(
    "selector",
    [vq.InitialGuess.HUECKEL, " huckel "],
    ids=["enum", "string-alias"],
)
def test_multik_gdf_hueckel_uses_shared_per_k_fock(monkeypatch, selector):
    """HUECKEL is normalized and executed, never relabelled Hcore."""
    _stub_multik_gdf_setup(monkeypatch)
    oneel_opts = vq.LatticeSumOptions()
    monkeypatch.setattr(
        periodic_k_gdf,
        "_oneel_lattice_opts",
        lambda *_args, **_kwargs: oneel_opts,
    )
    seen = []

    class FockGuessReached(RuntimeError):
        pass

    def stop_at_fock(
        system,
        basis,
        kpoints_cart,
        initial_guess,
        **kwargs,
    ):
        seen.append((system, basis, list(kpoints_cart), initial_guess, kwargs))
        raise FockGuessReached

    monkeypatch.setattr(
        periodic_k_gdf, "periodic_fock_guess_k", stop_at_fock
    )
    system, basis = _he_box()
    base = vq.PeriodicRHFOptions()
    base.max_iter = 0
    options = _GuessOptionsProxy(base, selector)

    with pytest.raises(FockGuessReached):
        periodic_k_gdf.run_krhf_periodic_gdf(
            system,
            basis,
            (2, 1, 1),
            options,
            use_compcell=True,
            rcut_strategy="flat",
            check_energy_sanity=False,
            progress=False,
        )

    assert len(seen) == 1
    assert seen[0][3] == vq.InitialGuess.HUECKEL
    assert seen[0][4]["kinetic_lattice"] is not None
    assert seen[0][4]["overlap_lattice"] is not None


@pytest.mark.parametrize(
    ("selector", "effective"),
    [
        (vq.InitialGuess.AUTO, vq.InitialGuess.SAD),
        (" minao ", vq.InitialGuess.MINAO),
    ],
    ids=["auto-to-sad-enum", "minao-string"],
)
def test_multik_gdf_density_guess_uses_shared_adapter(
    monkeypatch, selector, effective
):
    """Density-mode guesses reach the adapter instead of Hcore Aufbau."""
    _stub_multik_gdf_setup(monkeypatch)
    oneel_opts = vq.LatticeSumOptions()
    monkeypatch.setattr(
        periodic_k_gdf,
        "_oneel_lattice_opts",
        lambda *_args, **_kwargs: oneel_opts,
    )
    seen = []

    class DensityGuessReached(RuntimeError):
        pass

    def stop_at_density(_mol, _basis, _n_occ, initial_guess, **kwargs):
        seen.append((initial_guess, kwargs))
        raise DensityGuessReached

    monkeypatch.setattr(
        periodic_k_gdf, "initial_density_closed_shell", stop_at_density
    )
    system, basis = _he_box()
    base = vq.PeriodicRHFOptions()
    base.max_iter = 0
    options = _GuessOptionsProxy(base, selector)

    with pytest.raises(DensityGuessReached):
        periodic_k_gdf.run_krhf_periodic_gdf(
            system,
            basis,
            (2, 1, 1),
            options,
            use_compcell=True,
            rcut_strategy="flat",
            check_energy_sanity=False,
            progress=False,
        )

    assert len(seen) == 1
    assert seen[0][0] == effective
    kwargs = seen[0][1]
    assert kwargs["is_periodic"] is True
    assert kwargs["periodic_system"] is system
    assert kwargs["lattice_opts"] is oneel_opts
    assert len(kwargs["overlap"]) == 2
    np.testing.assert_allclose(kwargs["weights"], [0.5, 0.5])


def test_multik_open_gdf_auto_uses_shared_spin_density(monkeypatch):
    """The periodic AUTO policy resolves to an executed SAD spin seed."""
    _stub_multik_gdf_setup(monkeypatch)
    oneel_opts = vq.LatticeSumOptions()
    monkeypatch.setattr(
        periodic_k_gdf,
        "_oneel_lattice_opts",
        lambda *_args, **_kwargs: oneel_opts,
    )
    seen = []

    class SpinDensityGuessReached(RuntimeError):
        pass

    def stop_at_density(
        _mol, _basis, _n_alpha, _n_beta, initial_guess, **kwargs
    ):
        seen.append((initial_guess, kwargs))
        raise SpinDensityGuessReached

    monkeypatch.setattr(
        periodic_k_gdf, "initial_densities_open_shell", stop_at_density
    )
    system = vq.PeriodicSystem(
        3,
        10.0 * np.eye(3),
        [vq.Atom(1, [0.0, 0.0, 0.0])],
        charge=0,
        multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    base = vq.PeriodicRHFOptions()
    base.max_iter = 0
    options = _GuessOptionsProxy(base, " auto ")

    with pytest.raises(SpinDensityGuessReached):
        periodic_k_gdf.run_kuhf_periodic_gdf(
            system,
            basis,
            (2, 1, 1),
            options,
            rcut_strategy="flat",
            check_energy_sanity=False,
            progress=False,
        )

    assert seen[0][0] == vq.InitialGuess.SAD
    assert seen[0][1]["periodic_system"] is system
    assert seen[0][1]["lattice_opts"] is oneel_opts


def test_periodic_sap_fock_helper_bloch_sums_each_k(monkeypatch):
    """The shared helper must build F_SAP(k), not repeat its Gamma block."""
    system, basis = _he_box()
    basis = vq.BasisSet(system.unit_cell_molecule(), "6-31g")
    kinetic_lattice = object()
    sap_lattice = object()
    lattice_opts = object()
    calls = []

    monkeypatch.setattr(
        periodic_grid,
        "build_periodic_becke_grid",
        lambda system: ("grid", system),
    )

    def fake_vsap(got_basis, got_system, grid, table, options):
        assert got_basis is basis
        assert got_system is system
        assert grid == ("grid", system)
        assert table == "sap_helfem_large"
        assert options is lattice_opts
        return sap_lattice

    def fake_bloch_sum(lattice, k_cart):
        k_x = float(np.asarray(k_cart)[0])
        calls.append((lattice, k_x))
        if lattice is kinetic_lattice:
            return np.diag([1.0 + k_x, 2.0])
        assert lattice is sap_lattice
        return np.diag([-2.0 + 3.0 * k_x, -1.0])

    monkeypatch.setattr(periodic_guess, "compute_vsap_lattice", fake_vsap)
    monkeypatch.setattr(periodic_guess, "bloch_sum", fake_bloch_sum)

    fock_k = periodic_guess.periodic_sap_fock_k(
        system,
        basis,
        ([0.0, 0.0, 0.0], [0.25, 0.0, 0.0]),
        lattice_opts=lattice_opts,
        kinetic_lattice=kinetic_lattice,
    )

    np.testing.assert_allclose(fock_k[0], np.diag([-1.0, 1.0]))
    np.testing.assert_allclose(fock_k[1], np.diag([0.0, 1.0]))
    assert not np.array_equal(fock_k[0], fock_k[1])
    assert calls == [
        (kinetic_lattice, 0.0),
        (sap_lattice, 0.0),
        (kinetic_lattice, 0.25),
        (sap_lattice, 0.25),
    ]


@pytest.mark.parametrize(
    ("module", "driver_name", "is_ks"),
    [
        (pbc_bipole, "run_pbc_bipole_rhf", False),
        (pbc_bipole_rks, "run_pbc_bipole_rks", True),
    ],
    ids=["rhf", "rks"],
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
def test_periodic_fock_guess_bipole_closed_shell_dispatches_bloch_fock(
    monkeypatch,
    module,
    driver_name,
    is_ks,
    single_gamma,
    guess,
):
    """BIPOLE RHF/RKS execute each Fock-mode guess on every k point."""
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
    expected_diagonalisations = 1 if single_gamma else 2

    def stop_after_all_k(fock, orthogonalizer):
        diagonalised.append(np.asarray(fock).copy())
        if len(diagonalised) == expected_diagonalisations:
            raise SapPerKConsumed
        X = np.asarray(orthogonalizer, dtype=complex)
        return X, np.arange(X.shape[1], dtype=float)

    monkeypatch.setattr(module, "periodic_fock_guess_k", fake_fock_guess)
    monkeypatch.setattr(module, "_diag_in_orth_basis", stop_after_all_k)
    system, basis = _he_box()
    kmesh = (
        vq.monkhorst_pack(system, [1, 1, 1])
        if single_gamma
        else vq.monkhorst_pack(system, [2, 1, 1])
    )
    opts = vq.PeriodicKSOptions() if is_ks else vq.PeriodicRHFOptions()
    if is_ks:
        opts.functional = "lda"
    opts.initial_guess = guess
    opts.max_iter = 1
    opts.lattice_opts.cutoff_bohr = 4.0
    opts.lattice_opts.nuclear_cutoff_bohr = 4.0

    with pytest.raises(SapPerKConsumed):
        getattr(module, driver_name)(
            system,
            basis,
            kmesh,
            opts,
            use_ewald_j_split=True,
            progress=False,
            verbose=0,
        )

    assert len(seen) == 1
    _, _, kpoints_cart, kwargs = seen[0]
    assert len(kpoints_cart) == expected_diagonalisations
    if single_gamma:
        np.testing.assert_allclose(kpoints_cart[0], 0.0, atol=1e-12)
    else:
        assert not np.array_equal(kpoints_cart[0], kpoints_cart[1])
    assert kwargs["kinetic_lattice"] is not None
    assert kwargs["overlap_lattice"] is not None
    assert kwargs["lattice_opts"] is not None
    np.testing.assert_allclose(diagonalised[0], np.eye(basis.nbasis) * 10.0)
    if not single_gamma:
        np.testing.assert_allclose(diagonalised[1], np.eye(basis.nbasis) * 11.0)


@pytest.mark.parametrize(
    ("module", "driver_name", "is_ks"),
    [
        (
            periodic_rhf_multi_k_ewald,
            "run_rhf_periodic_multi_k_ewald3d",
            False,
        ),
        (
            periodic_rks_multi_k_ewald,
            "run_rks_periodic_multi_k_ewald3d",
            True,
        ),
    ],
    ids=["rhf", "rks"],
)
@pytest.mark.parametrize(
    "guess",
    [vq.InitialGuess.SAP, vq.InitialGuess.HUECKEL],
    ids=["sap", "hueckel"],
)
def test_periodic_fock_guess_legacy_ewald_closed_shell_dispatches_bloch_fock(
    monkeypatch,
    module,
    driver_name,
    is_ks,
    guess,
):
    """Legacy multi-k Ewald RHF/RKS diagonalise each F_guess(k)."""
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

    def stop_after_all_k(fock, orthogonalizer):
        diagonalised.append(np.asarray(fock).copy())
        if len(diagonalised) == 2:
            raise SapPerKConsumed
        X = np.asarray(orthogonalizer, dtype=complex)
        return X, np.arange(X.shape[1], dtype=float)

    monkeypatch.setattr(module, "periodic_fock_guess_k", fake_fock_guess)
    monkeypatch.setattr(module, "_diag_in_orth_basis", stop_after_all_k)
    system, basis = _he_box()
    kmesh = vq.monkhorst_pack(system, [2, 1, 1])
    opts = vq.PeriodicKSOptions() if is_ks else vq.PeriodicRHFOptions()
    if is_ks:
        opts.functional = "lda"
    opts.initial_guess = guess
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
    np.testing.assert_allclose(diagonalised[0], np.eye(basis.nbasis) * 10.0)
    np.testing.assert_allclose(diagonalised[1], np.eye(basis.nbasis) * 11.0)
