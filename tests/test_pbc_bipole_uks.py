"""Tests for the CRYSTAL-gauge BIPOLE UKS driver.

Regression coverage for the non-converged spheropole-reporting bug: a
``max_iter=1`` (non-converged) UKS run on a 3D system must report
``result.e_ext_el_spheropole`` as the nonzero float that was actually
folded into ``E_total`` in-loop — not ``None``. Before the fix the UKS
driver (a) never set ``e_ext_el_spheropole`` on its per-iteration
``PBCBipoleEnergyComponents`` and (b) hard-coded ``E_sphero_final = None``
in its non-converged post-loop branch, so the reported breakdown field
was dropped even though ``E_total`` already included the term. The sibling
RKS / UHF / RHF drivers always reported it; this pins UKS to that contract.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import InitialGuess, monkhorst_pack
from vibeqc._vibeqc_core import PeriodicKSOptions, PeriodicRHFOptions
from vibeqc.pbc_bipole import run_pbc_bipole_rhf
from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks
from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks


def _ks_opts(max_iter: int = 1) -> PeriodicKSOptions:
    opts = PeriodicKSOptions()
    opts.lattice_opts.cutoff_bohr = 5.0
    opts.lattice_opts.nuclear_cutoff_bohr = 10.0
    opts.max_iter = max_iter
    opts.initial_guess = InitialGuess.HCORE
    opts.use_diis = False
    opts.functional = "pbe"
    return opts


def _rhf_opts(max_iter: int = 1) -> PeriodicRHFOptions:
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 5.0
    opts.lattice_opts.nuclear_cutoff_bohr = 10.0
    opts.max_iter = max_iter
    opts.initial_guess = InitialGuess.HCORE
    opts.use_diis = False
    return opts


def _h2_3d():
    lattice = np.eye(3) * 8.0
    system = vq.PeriodicSystem(
        3, lattice,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])
    return system, basis, kmesh


def test_pbc_bipole_uks_is_exported():
    assert vq.run_pbc_bipole_uks is run_pbc_bipole_uks


def test_default_odd_electron_uks_reports_executed_doublet():
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [vq.Atom(1, [0.0, 0.0, 0.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])

    result = run_pbc_bipole_uks(
        system,
        basis,
        kmesh,
        _ks_opts(),
        ewald_precision=1e-6,
        progress=False,
    )

    assert math.isclose(result.s_squared, 0.75, abs_tol=1e-12)
    assert math.isclose(result.s_squared_ideal, 0.75, abs_tol=1e-12)
    assert system.multiplicity == 1


@pytest.mark.parametrize("functional", ["vv10", "b2plyp"])
@pytest.mark.parametrize(
    "driver",
    [
        pytest.param(run_pbc_bipole_rks, id="rks"),
        pytest.param(run_pbc_bipole_uks, id="uks"),
    ],
)
def test_bipole_ks_rejects_incomplete_functional(driver, functional):
    system, basis, kmesh = _h2_3d()
    with pytest.raises(NotImplementedError):
        driver(
            system,
            basis,
            kmesh,
            _ks_opts(),
            functional=functional,
            progress=False,
        )


def test_nonconverged_uks_reports_spheropole_3d():
    """max_iter=1 (non-converged) 3D UKS must report the spheropole float.

    The spheropole is a 3D-Ewald-gauge term; on a 3D box it is a nonzero
    float that is folded into ``E_total`` in-loop. The reported breakdown
    field must agree with it and must NOT be ``None`` for a non-converged
    run (the original bug).
    """
    system, basis, kmesh = _h2_3d()

    result = run_pbc_bipole_uks(
        system, basis, kmesh, _ks_opts(),
        # The spheropole reporting path is a LEGACY-gauge feature: the
        # corrected gauge (the Γ default since option (b) Phase 4b)
        # omits the term entirely (e_ext_el_spheropole is None).
        use_exchange_ewald_split=False,
        ewald_precision=1e-6,
        progress=False,
    )

    assert result.n_iter == 1
    assert result.converged is False
    # Regression: must be a real float, not None.
    assert result.e_ext_el_spheropole is not None
    assert isinstance(result.e_ext_el_spheropole, float)
    assert result.e_ext_el_spheropole > 0.0

    comp = result.energy_components[-1]
    # The result-level field must equal the last per-iteration component...
    assert comp.e_ext_el_spheropole is not None
    assert math.isclose(
        result.e_ext_el_spheropole,
        comp.e_ext_el_spheropole,
        abs_tol=1e-12,
    )
    # ...and the per-iteration energy breakdown must close, spheropole
    # included (this is the closure the UHF driver's smoke test asserts).
    assert math.isclose(
        comp.e_electronic + comp.e_nuclear_repulsion
        + (comp.e_ext_el_spheropole or 0.0),
        comp.e_total,
        abs_tol=1e-10,
    )


def test_nonconverged_uks_spheropole_matches_rhf():
    """Closed-shell singlet at the HCORE guess: UKS total density == RHF's,
    so the spheropole (a functional of the total density only) is identical.
    Cross-checks UKS against the always-correct RHF reporting path.
    """
    system, basis, kmesh = _h2_3d()

    # Legacy gauge on the RHF side: under the Ewald exchange split (the
    # Γ RHF default) the spheropole is omitted from RHF results
    # (e_ext_el_spheropole is None); UKS still reports it (option (b)
    # Phase 4 propagation pending), so compare in the legacy gauge.
    rhf = run_pbc_bipole_rhf(
        system, basis, kmesh, _rhf_opts(),
        use_exchange_ewald_split=False,
        ewald_precision=1e-6,
        progress=False,
    )
    uks = run_pbc_bipole_uks(
        system, basis, kmesh, _ks_opts(),
        use_exchange_ewald_split=False,  # legacy gauge, like the RHF side
        ewald_precision=1e-6,
        progress=False,
    )

    assert rhf.e_ext_el_spheropole is not None
    assert uks.e_ext_el_spheropole is not None
    assert math.isclose(
        uks.e_ext_el_spheropole,
        rhf.e_ext_el_spheropole,
        abs_tol=1e-10,
    )


def test_multik_sad_executes_its_own_consistent_bloch_density(monkeypatch):
    """SAD must supply the first physical Fock instead of being discarded."""
    lattice = np.eye(3) * 8.0
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [2, 1, 1])

    import vibeqc.pbc_bipole_uks as driver_module

    first_cycle = {}
    builder = driver_module.build_bipole_unrestricted_fock

    def capture(context, alpha, beta, **kwargs):
        if guess not in first_cycle:
            home = next(i for i, cell in enumerate(alpha.cells) if not np.any(cell.index))
            first_cycle[guess] = (
                np.asarray(alpha.blocks[home]).copy(), np.asarray(beta.blocks[home]).copy()
            )
        return builder(context, alpha, beta, **kwargs)

    monkeypatch.setattr(driver_module, "build_bipole_unrestricted_fock", capture)
    for guess in (InitialGuess.HCORE, InitialGuess.SAD):
        opts = _ks_opts(max_iter=1)
        opts.initial_guess = guess
        result = run_pbc_bipole_uks(
            system,
            basis,
            kmesh,
            opts,
            ewald_precision=1.0e-6,
            sr_image_precision=None,
            progress=False,
        )
        assert result.n_iter == 1

    for sad, hcore in zip(first_cycle[InitialGuess.SAD], first_cycle[InitialGuess.HCORE]):
        assert np.linalg.norm(sad - hcore) > 1e-3
        assert np.max(np.abs(sad - np.diag(np.diag(sad)))) < 1e-12


def test_multik_patom_feeds_its_sad_seed_to_the_in_field_step(monkeypatch):
    """PATOM's deliberate local seed must not be replaced by Hcore."""
    import vibeqc.pbc_bipole_uks as uks_module

    lattice = np.eye(3) * 8.0
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [2, 1, 1])
    seed_alpha = np.eye(int(basis.nbasis)) * 0.37
    seed_beta = np.eye(int(basis.nbasis)) * 0.11

    def fake_open_shell_guess(*args, **kwargs):
        return seed_alpha.copy(), seed_beta.copy()

    captured = {}

    class _StopAfterPatomSeed(RuntimeError):
        pass

    def capture_first_fock(context, density_alpha, density_beta, **kwargs):
        captured["cells"] = [
            tuple(np.asarray(cell.index, dtype=int))
            for cell in density_alpha.cells
        ]
        captured["alpha"] = [np.asarray(block).copy() for block in density_alpha.blocks]
        captured["beta"] = [np.asarray(block).copy() for block in density_beta.blocks]
        raise _StopAfterPatomSeed

    monkeypatch.setattr(
        uks_module,
        "initial_densities_open_shell",
        fake_open_shell_guess,
    )
    monkeypatch.setattr(
        uks_module,
        "build_bipole_unrestricted_fock",
        capture_first_fock,
    )

    opts = _ks_opts(max_iter=1)
    opts.initial_guess = InitialGuess.PATOM
    with pytest.raises(_StopAfterPatomSeed):
        run_pbc_bipole_uks(
            system,
            basis,
            kmesh,
            opts,
            ewald_precision=1.0e-6,
            sr_image_precision=None,
            progress=False,
        )

    home = captured["cells"].index((0, 0, 0))
    np.testing.assert_allclose(captured["alpha"][home], seed_alpha)
    np.testing.assert_allclose(captured["beta"][home], seed_beta)
    for idx, cell in enumerate(captured["cells"]):
        translation = np.asarray(cell) @ lattice
        phase = sum(w * np.exp(-1j * np.dot(k, translation)) for w, k in zip(kmesh.weights, kmesh.kpoints))
        np.testing.assert_allclose(captured["alpha"][idx], phase.real * seed_alpha, atol=1e-12)
        np.testing.assert_allclose(captured["beta"][idx], phase.real * seed_beta, atol=1e-12)


def test_uks_pure_functional_j_contracts_full_wide_density():
    """UKS pure-functional J must contract the FULL wide density support.

    Under the Ewald exchange split (the production default) the per-spin
    densities live on the 2x-cutoff cell list. The unrestricted builder
    used to form ``density_total`` on the 1x radial overlap template
    (``_combine_density_sets``), silently zeroing P(h) for |h| in
    (cutoff, 2x cutoff] in the pure-functional (``alpha_hf = 0``)
    J-only build; the C++ builder skips missing P(h) lookups without
    complaint. In the production default the damage is LATENT: the
    cell-level Schwarz guard requires the ket displacement h to sit in
    the internal (1x) cell list, so P(h) beyond 1x is never read. It
    goes LIVE exactly when the internal ball is widened -- the M4a
    ``sr_image_extent_bohr`` padded build (escalated to become the
    default) -- or when Schwarz screening is off. The hybrid/UHF path
    was never affected: it builds J per spin, directly on the wide
    spin densities. Pair-resolved SYM3b fixed this for its own mode
    only (M3); this pins the plain padded path.

    The pin: at iteration 1 from the HCORE guess every BIPOLE driver
    holds the SAME closed-shell total density, and ``e_j_short_range``
    / ``e_j_long_range`` are functionals of that density alone (no XC,
    no exchange in those components). RHF builds its padded split J on
    the full wide density, so UKS must reproduce RHF's J components
    exactly. (RKS is NOT the reference here: pure-functional RKS
    defaults to the exact analytic-FT Ewald J, a different route.)
    """
    lattice = np.eye(3) * 4.0
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(2, [0.0, 0.0, 0.0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1])

    ks = PeriodicKSOptions()
    # Keep the AO fold converged (drift 7.0e-5) while retaining an 8-bohr
    # short-range image extent that is wider than the 7-bohr density cutoff.
    ks.lattice_opts.cutoff_bohr = 7.0
    ks.lattice_opts.nuclear_cutoff_bohr = 8.0
    ks.max_iter = 1
    ks.initial_guess = InitialGuess.HCORE
    ks.use_diis = False
    ks.functional = "svwn"

    hf = PeriodicRHFOptions()
    hf.lattice_opts.cutoff_bohr = 7.0
    hf.lattice_opts.nuclear_cutoff_bohr = 8.0
    hf.max_iter = 1
    hf.initial_guess = InitialGuess.HCORE
    hf.use_diis = False

    rhf = run_pbc_bipole_rhf(
        system, basis, kmesh, hf,
        ewald_precision=1e-6, sr_image_extent_bohr=8.0, progress=False,
    )
    uks = run_pbc_bipole_uks(
        system, basis, kmesh, ks,
        ewald_precision=1e-6, sr_image_extent_bohr=8.0, progress=False,
    )

    c_rhf = rhf.energy_components[-1]
    c_uks = uks.energy_components[-1]
    assert c_rhf.e_j_short_range is not None
    assert c_uks.e_j_short_range is not None
    assert math.isclose(
        c_uks.e_j_short_range, c_rhf.e_j_short_range, abs_tol=1e-10
    )
    assert math.isclose(
        c_uks.e_j_long_range, c_rhf.e_j_long_range, abs_tol=1e-10
    )


@pytest.mark.parametrize(
    "driver",
    [
        pytest.param(run_pbc_bipole_rks, id="rks"),
        pytest.param(run_pbc_bipole_uks, id="uks"),
    ],
)
def test_pbe0_result_reports_disjoint_coulomb_and_exchange(driver):
    """Public hybrid J and K fields must sum to the physical two-electron term."""
    system, basis, kmesh = _h2_3d()
    opts = _ks_opts(max_iter=1)
    opts.functional = "pbe0"
    result = driver(
        system,
        basis,
        kmesh,
        opts,
        functional="pbe0",
        ewald_precision=1e-6,
        progress=False,
    )

    comp = result.energy_components[-1]
    e_j = (
        (comp.e_j_short_range or 0.0)
        + (comp.e_j_long_range or 0.0)
        + (comp.e_j_multipole or 0.0)
    )
    assert comp.e_exchange is not None
    assert result.e_coulomb > 0.0
    assert result.e_hf_exchange < 0.0
    assert math.isclose(result.e_coulomb, e_j, abs_tol=1e-10)
    assert math.isclose(
        result.e_hf_exchange,
        comp.e_exchange,
        abs_tol=1e-10,
    )
    assert math.isclose(
        result.e_coulomb + result.e_hf_exchange,
        comp.e_two_electron,
        abs_tol=1e-10,
    )


@pytest.mark.parametrize("spin", [False, True], ids=["rks", "uks"])
def test_multik_constructed_sad_reaches_hubbard_and_accelerator(monkeypatch, spin):
    """A repeated BvK cell list must not multiply the first-cycle density."""
    from vibeqc import _vibeqc_core as core
    from vibeqc.periodic_scf_accelerators import (
        MultiKPeriodicSCFAccelerator, MultiKPeriodicUHFAccelerator,
    )
    import importlib

    module = importlib.import_module(
        "vibeqc.pbc_bipole_uks" if spin else "vibeqc.pbc_bipole_rks"
    )
    system = vq.PeriodicSystem(3, np.eye(3) * 3, [vq.Atom(2, [0, 0, 0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    mesh = monkhorst_pack(system, [3, 1, 1])
    options = _ks_opts(max_iter=1)
    options.initial_guess = InitialGuess.SAD
    options.lattice_opts.cutoff_bohr = 10
    options.use_diis = True
    calls = {"builder": 0, "hubbard": 0, "accelerator": 0}
    builder_name = "initial_densities_open_shell" if spin else "initial_density_closed_shell"
    builder = getattr(module, builder_name)

    def build(*args, **kwargs):
        calls["builder"] += 1
        return builder(*args, **kwargs)

    monkeypatch.setattr(module, builder_name, build)
    hubbard = core._compute_dft_plus_u_multi_k_per_spin_cxx
    overlaps = []

    def apply_u(sites, groups, sk, densities, weights):
        calls["hubbard"] += 1
        overlaps[:] = sk
        count = sum(w * np.trace(d @ s) for w, d, s in zip(weights, densities, sk))
        assert count == pytest.approx(1, abs=1e-11)
        return hubbard(sites, groups, sk, densities, weights)

    monkeypatch.setattr(core, "_compute_dft_plus_u_multi_k_per_spin_cxx", apply_u)

    class StopAfterFirstFock(Exception):
        pass

    def accelerate(self, *args, **kwargs):
        calls["accelerator"] += 1
        names = ("density_alpha_k_list", "density_beta_k_list") if spin else ("density_k_list",)
        for name in names:
            count = sum(w * np.trace(d @ s) for w, d, s in zip(
                kwargs["weights"], kwargs[name], overlaps
            ))
            assert count == pytest.approx(1 if spin else 2, abs=1e-11)
        raise StopAfterFirstFock

    monkeypatch.setattr(
        MultiKPeriodicUHFAccelerator if spin else MultiKPeriodicSCFAccelerator,
        "extrapolate_uhf" if spin else "extrapolate_rhf", accelerate
    )
    with pytest.raises(StopAfterFirstFock):
        driver = module.run_pbc_bipole_uks if spin else module.run_pbc_bipole_rks
        driver(
            system, basis, mesh, options, progress=False,
            use_exchange_ewald_split=False, ewald_precision=1e-6,
            dft_plus_u=[vq.HubbardSite(0, 0, U_ev=1.0)],
        )
    assert calls == {"builder": 1, "hubbard": 2 if spin else 1, "accelerator": 1}


def test_multik_patom_refinement_reaches_all_first_cycle_consumers(monkeypatch):
    """PATOM's refined two-AO matrices replace SAD in +U and DIIS too."""
    import vibeqc.pbc_bipole_uks as module
    from vibeqc import _vibeqc_core as core
    from vibeqc.periodic_scf_accelerators import MultiKPeriodicUHFAccelerator

    system = vq.PeriodicSystem(3, np.eye(3) * 5, [
        vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    mesh = monkhorst_pack(system, [3, 1, 1])
    options = _ks_opts(max_iter=1)
    options.initial_guess = InitialGuess.PATOM
    options.lattice_opts.cutoff_bohr = 10
    options.use_diis = True
    expected = []
    seeds = []
    native_build = module.build_bipole_unrestricted_fock

    def build(*args, **kwargs):
        if kwargs["patom_hf_like"]:
            seeds.append(np.asarray(args[1].blocks[0]).copy())
        else:
            for name in ("coeffs_alpha_for_rho", "coeffs_beta_for_rho"):
                coeffs = kwargs[name]
                assert coeffs is not None
                expected.append([np.asarray(c)[:, :1] @ np.asarray(c)[:, :1].conj().T for c in coeffs])
        return native_build(*args, **kwargs)

    monkeypatch.setattr(module, "build_bipole_unrestricted_fock", build)
    native_u = core._compute_dft_plus_u_multi_k_per_spin_cxx
    u_calls = []

    def apply_u(sites, groups, overlaps, densities, weights):
        np.testing.assert_allclose(densities, expected[len(u_calls)], atol=1e-11)
        u_calls.append(True)
        return native_u(sites, groups, overlaps, densities, weights)

    monkeypatch.setattr(core, "_compute_dft_plus_u_multi_k_per_spin_cxx", apply_u)

    class StopAfterFirstFock(Exception):
        pass

    def accelerate(self, *args, **kwargs):
        for index, name in enumerate(("density_alpha_k_list", "density_beta_k_list")):
            np.testing.assert_allclose(kwargs[name], expected[index], atol=1e-11)
        assert np.linalg.norm(expected[0][0] - seeds[0]) > 1e-3
        raise StopAfterFirstFock

    monkeypatch.setattr(MultiKPeriodicUHFAccelerator, "extrapolate_uhf", accelerate)
    with pytest.raises(StopAfterFirstFock):
        module.run_pbc_bipole_uks(
            system, basis, mesh, options, progress=False,
            use_exchange_ewald_split=False, ewald_precision=1e-6,
            dft_plus_u=[vq.HubbardSite(0, 0, U_ev=1.0)],
        )
    assert len(u_calls) == 2


@pytest.mark.parametrize('method', ['rhf','rks','uhf','uks'])
def test_bipole_per_k_read_reaches_fock_with_exact_population(method):
    import importlib
    from vibeqc.guess_read import resolve_periodic_read_density_k_closed, resolve_periodic_read_densities_k_open
    from vibeqc.pbc_bipole_common import bvk_torus_density_matrices
    opened=method in ('uhf','uks'); ks=method.endswith('ks')
    system=vq.PeriodicSystem(3,np.eye(3)*8.,[vq.Atom(1,[0,0,0]),vq.Atom(1,[0,0,1.4])])
    basis=vq.BasisSet(system.unit_cell_molecule(),'sto-3g');mesh=vq.monkhorst_pack(system,[3,1,1])
    opts=vq.PeriodicKSOptions() if ks else vq.PeriodicRHFOptions()
    opts.initial_guess=vq.InitialGuess.SAD;opts.max_iter=1;opts.use_diis=False
    opts.lattice_opts.cutoff_bohr=5.;opts.lattice_opts.nuclear_cutoff_bohr=10.
    if ks: opts.functional='pbe'
    module=importlib.import_module('vibeqc.pbc_bipole'+('' if method=='rhf' else '_'+method))
    driver=getattr(module,'run_pbc_bipole_'+method)
    kw=dict(progress=False,ewald_precision=1e-6,sr_image_precision=None)
    source=driver(system,basis,mesh,opts,**kw)
    assert source.restart_mesh == tuple(mesh.mesh)
    loader=resolve_periodic_read_densities_k_open if opened else resolve_periodic_read_density_k_closed
    density=loader(read_from=source,basis=basis,system=system,kmesh=mesh)
    opts.initial_guess=vq.InitialGuess.READ
    result=driver(system,basis,mesh,opts,initial_density_k=density,**kw)
    assert result.guess_selection.effective==vq.InitialGuess.READ
    assert result.n_iter==1
    for channel,n in ([('density_alpha',1),('density_beta',1)] if opened else [('density',2)]):
        ds=bvk_torus_density_matrices(getattr(result,channel),list(mesh.kpoints),mesh.mesh)
        count=sum(w*np.trace(d@s).real for w,d,s in zip(mesh.weights,ds,result.overlap))
        assert count==pytest.approx(n,abs=1e-10)
    opts.initial_guess=vq.InitialGuess.FRAGMO
    with pytest.raises(NotImplementedError,match='FRAGMO'):
        driver(system,basis,mesh,opts,initial_density_k=density,**kw)
