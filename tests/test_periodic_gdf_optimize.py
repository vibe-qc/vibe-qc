"""GDF-objective optimizer wiring in ``run_periodic_job`` (G-PBC-002).

Pre-fix, ``run_periodic_job(optimize=True)`` relaxed every route on the
BIPOLE force objective (``bipole_optimize.relax_atoms``) -- including
runs whose SCF energy came from the GDF drivers, so the relaxation
converged to a stationary point of a surface the job never reported.
These tests pin the fix: a GDF-routed SCF relaxes on the same GDF
analytic-gradient objective (the dispatched driver re-run with
``compute_gradient=True`` per candidate geometry), the ``.out`` records
the active force objective, and every envelope the GDF gradient rejects
fails closed instead of silently swapping objectives.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


def _h2_12bohr(d: float = 1.5):
    """H2 in a 12-bohr cubic box (the GDF gradient-test fixture cell)."""
    box = 12.0
    c = box / 2.0
    atoms = [
        vq.Atom(1, [c, c, c - d / 2.0]),
        vq.Atom(1, [c, c, c + d / 2.0]),
    ]
    system = vq.PeriodicSystem(3, np.eye(3) * box, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _h2_slab(d: float = 1.5):
    """H2 in the bounded vacuum-padded dim=2 slab-GDF envelope."""
    box = 12.0
    c = box / 2.0
    system = vq.PeriodicSystem(
        2,
        np.diag([box, box, 20.0]),
        [
            vq.Atom(1, [c, c, 10.0 - d / 2.0]),
            vq.Atom(1, [c, c, 10.0 + d / 2.0]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


_NO_SIDECARS = dict(
    write_molden_file=False,
    write_density=False,
    write_xyz_file=False,
    write_poscar_file=False,
    write_xsf_structure_file=False,
    write_cif_file=False,
    write_population_file=False,
    citations=False,
    progress=False,
)


def _bond_length(system) -> float:
    a, b = system.unit_cell
    return float(
        np.linalg.norm(np.asarray(a.xyz, dtype=float) - np.asarray(b.xyz, dtype=float))
    )


def test_gamma_gdf_optimize_relaxes_on_gdf_objective(tmp_path):
    """Γ RHF/GDF + optimize=True relaxes on the GDF analytic gradient.

    H2 started stretched (1.5 bohr); the rsgdf surface at ke=60 must pull
    it back to a sane STO-3G bond length, and the ``.out`` must state the
    GDF force objective (the provenance line that distinguishes this from
    the pre-fix silent BIPOLE relaxation).
    """
    sysp, basis = _h2_12bohr(d=1.5)
    out_stem = tmp_path / "h2-gamma-gdf-opt"
    r = vq.run_periodic_job(
        sysp,
        basis,
        method="RHF",
        jk_method="gdf",
        rsgdf_ke_cutoff=60.0,
        output=out_stem,
        optimize=True,
        optimize_max_iter=30,
        optimize_conv_tol_grad=1e-3,
        **_NO_SIDECARS,
    )

    assert r.converged
    d_final = _bond_length(r.system)
    # STO-3G periodic H2 equilibrium is ~1.35 bohr; anything inside this
    # window proves a real relaxation happened on a sensible surface.
    assert 1.2 < d_final < 1.45
    assert r.energy < -1.117  # below the stretched-geometry SCF energy

    out_text = (tmp_path / "h2-gamma-gdf-opt.out").read_text()
    assert "Geometry optimization" in out_text
    assert "force objective     = GDF analytic gradient (rsgdf)" in out_text


def test_multik_gdf_optimize_dispatches_to_multik_gradient_driver(
    monkeypatch, tmp_path
):
    """(2,1,1) GDF optimize re-runs the SAME multi-k driver with
    ``compute_gradient=True`` -- the SCF-objective-identity pin."""
    from vibeqc import periodic_runner
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf as real_krhf

    calls: list[dict] = []

    def counting_krhf(system, basis, kmesh, opts, **kwargs):
        calls.append(
            {
                "compute_gradient": bool(kwargs.get("compute_gradient", False)),
                "gdf_method": kwargs.get("gdf_method"),
                "kmesh": tuple(kmesh),
            }
        )
        return real_krhf(system, basis, kmesh, opts, **kwargs)

    # The runner binds the driver at module scope; patching that name
    # covers both the SCF dispatch and the captured optimizer objective.
    monkeypatch.setattr(periodic_runner, "run_krhf_periodic_gdf", counting_krhf)

    sysp, basis = _h2_12bohr(d=1.5)
    r = vq.run_periodic_job(
        sysp,
        basis,
        method="RHF",
        jk_method="gdf",
        kpoints=(2, 1, 1),
        rsgdf_ke_cutoff=60.0,
        output=tmp_path / "h2-multik-gdf-opt",
        optimize=True,
        optimize_max_iter=30,
        optimize_conv_tol_grad=1e-3,
        **_NO_SIDECARS,
    )

    assert r.converged
    assert 1.2 < _bond_length(r.system) < 1.45

    # First call is the plain SCF; every optimizer evaluation re-runs the
    # identical multi-k driver with the analytic gradient on.
    assert calls[0]["compute_gradient"] is False
    opt_calls = calls[1:]
    assert opt_calls, "optimizer never re-ran the captured GDF driver"
    assert all(c["compute_gradient"] for c in opt_calls)
    assert all(c["kmesh"] == (2, 1, 1) for c in opt_calls)
    assert all(c["gdf_method"] == "rsgdf" for c in opt_calls)

    out_text = (tmp_path / "h2-multik-gdf-opt.out").read_text()
    assert "force objective     = GDF analytic gradient (rsgdf)" in out_text


def test_gdf_optimize_cell_fails_closed(tmp_path):
    """optimize_cell on the GDF route raises pre-SCF: no GDF stress."""
    sysp, basis = _h2_12bohr()
    with pytest.raises(NotImplementedError, match="optimize_cell.*GDF route"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RHF",
            jk_method="gdf",
            output=tmp_path / "h2-gdf-cell-opt",
            optimize=True,
            optimize_cell=True,
            **_NO_SIDECARS,
        )


def test_gdf_optimize_mdf_envelope_fails_in_preflight(tmp_path):
    """An unsupported fit fails before SCF/dry-run, never by route swap."""
    sysp, basis = _h2_12bohr()
    stem = tmp_path / "h2-gdf-mdf-opt"
    with pytest.raises(
        NotImplementedError,
        match="GDF optimization currently supports.*rsgdf",
    ):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RHF",
            jk_method="gdf",
            gdf_method="mdf",
            output=stem,
            optimize=True,
            optimize_max_iter=5,
            dry_run=True,
            **_NO_SIDECARS,
        )
    assert not stem.with_suffix(".system").exists()


def test_gdf_optimize_dispersion_fails_before_dry_run(tmp_path):
    sysp, basis = _h2_12bohr()
    stem = tmp_path / "h2-gdf-d3-opt"
    with pytest.raises(NotImplementedError, match="GDF optimization.*dispersion"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RHF",
            jk_method="gdf",
            dispersion="d3bj",
            optimize=True,
            dry_run=True,
            output=stem,
            **_NO_SIDECARS,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    "temperature_source",
    ["explicit", "kpoints", "auto"],
)
def test_gdf_optimize_finite_temperature_fails_before_dry_run(
    tmp_path,
    temperature_source,
):
    if temperature_source == "auto":
        sysp = vq.PeriodicSystem(
            3,
            np.eye(3) * 6.0,
            [vq.Atom(4, [3.0, 3.0, 3.0])],
        )
        basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    else:
        sysp, basis = _h2_12bohr()
    stem = tmp_path / f"gdf-smeared-opt-{temperature_source}"
    kwargs = {}
    if temperature_source == "explicit":
        kwargs["smearing_temperature"] = 0.005
    elif temperature_source == "kpoints":
        kpoints = vq.KPoints.gamma(sysp)
        kpoints.smearing = vq.SmearingOptions(
            temperature=0.005,
            flavor="fermi-dirac",
        )
        kwargs["kpoints"] = kpoints
    else:
        kwargs["convergence"] = "auto"

    with pytest.raises(
        NotImplementedError,
        match="finite-temperature GDF optimization",
    ):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RKS",
            functional="lda",
            jk_method="gdf",
            optimize=True,
            dry_run=True,
            output=stem,
            **_NO_SIDECARS,
            **kwargs,
        )
    assert not stem.with_suffix(".system").exists()


def test_gdf_optimize_zero_temperature_dry_run_stays_supported(tmp_path):
    sysp, basis = _h2_12bohr()
    stem = tmp_path / "h2-gdf-zero-t-opt"
    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RHF",
        jk_method="gdf",
        smearing_temperature=0.0,
        optimize=True,
        dry_run=True,
        output=stem,
        **_NO_SIDECARS,
    )
    assert result is None
    assert stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
def test_gdf_optimize_legacy_gamma_fallback_fails_closed(tmp_path, dry_run):
    """A Γ run that falls back to the legacy molecular-limit GDF driver
    (here: explicit Fock mixing) has no analytic gradient and must fail
    closed with the actionable runner message."""
    sysp, basis = _h2_12bohr()
    stem = tmp_path / f"h2-gdf-legacy-opt-{dry_run}"
    with pytest.raises(
        NotImplementedError,
        match="legacy Gamma molecular-limit GDF driver",
    ):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RHF",
            jk_method="gdf",
            fmixing_percent=30.0,
            output=stem,
            optimize=True,
            optimize_max_iter=5,
            dry_run=dry_run,
            **_NO_SIDECARS,
        )
    assert not stem.with_suffix(".system").exists()


def test_gdf_optimize_nonuniform_weights_fail_before_dry_run(tmp_path):
    sysp, basis = _h2_12bohr()
    kpoints = vq.KPoints.from_list(
        sysp,
        [[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]],
        weights=[0.75, 0.25],
    )
    stem = tmp_path / "h2-gdf-weighted-opt"
    with pytest.raises(NotImplementedError, match="uniform full-BZ"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RHF",
            jk_method="gdf",
            kpoints=kpoints,
            optimize=True,
            dry_run=True,
            output=stem,
            **_NO_SIDECARS,
        )
    assert not stem.with_suffix(".system").exists()


def test_gdf_optimize_pure_rks_multik_fails_before_dry_run(tmp_path):
    sysp, basis = _h2_12bohr()
    stem = tmp_path / "h2-gdf-pbe-multik-opt"
    with pytest.raises(NotImplementedError, match="pure-functional multi-k"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RKS",
            functional="pbe",
            jk_method="gdf",
            kpoints=(2, 1, 1),
            optimize=True,
            dry_run=True,
            output=stem,
            **_NO_SIDECARS,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
def test_gdf_optimize_pure_rks_explicit_gamma_mixer_fails_preflight(
    tmp_path,
    dry_run,
):
    """A non-None mixer sends explicit Gamma RKS into the full KRKS loop.

    Pure RKS leaves that loop on use_compcell=False, so its optimizer rerun
    has no fitted Lpq cache to differentiate. Reject both dry and live jobs
    before either can create a manifest.
    """
    sysp, basis = _h2_12bohr()
    stem = tmp_path / f"h2-gdf-pbe-gamma-mixed-opt-{dry_run}"
    with pytest.raises(
        NotImplementedError,
        match="pure-functional explicit-Gamma density-mixed RKS/GDF",
    ):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RKS",
            functional="pbe",
            jk_method="gdf",
            kpoints=(1, 1, 1),
            density_mixer="anderson",
            optimize=True,
            dry_run=dry_run,
            output=stem,
            **_NO_SIDECARS,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    "density_mixer",
    [None, "", "none", "diis"],
    ids=["none-object", "empty", "none-string", "diis"],
)
def test_gdf_optimize_pure_rks_gamma_default_diis_alias_stays_supported(
    tmp_path,
    density_mixer,
):
    """All default-DIIS aliases retain the differentiable Gamma fast path."""
    sysp, basis = _h2_12bohr()
    alias = "none-object" if density_mixer is None else density_mixer or "empty"
    stem = tmp_path / f"h2-gdf-pbe-gamma-opt-{alias}"
    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RKS",
        functional="pbe",
        jk_method="gdf",
        kpoints=(1, 1, 1),
        density_mixer=density_mixer,
        optimize=True,
        dry_run=True,
        output=stem,
        **_NO_SIDECARS,
    )
    assert result is None
    assert stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
def test_gdf_optimize_pure_rks_single_non_gamma_fails_preflight(
    tmp_path,
    dry_run,
):
    """A one-point non-Gamma list still dispatches through full KRKS."""
    sysp, basis = _h2_12bohr()
    kpoints = vq.KPoints.from_list(
        sysp,
        [[0.25, 0.0, 0.0]],
        weights=[1.0],
    )
    stem = tmp_path / f"h2-gdf-pbe-single-nongamma-opt-{dry_run}"
    with pytest.raises(
        NotImplementedError,
        match="pure-functional explicit non-Gamma RKS/GDF",
    ):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RKS",
            functional="pbe",
            jk_method="gdf",
            kpoints=kpoints,
            optimize=True,
            dry_run=dry_run,
            output=stem,
            **_NO_SIDECARS,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
@pytest.mark.parametrize(
    ("mixer_kwargs", "message"),
    [
        pytest.param(
            {"density_mixer": "bogus"},
            "density_mixer must be",
            id="unknown-mixer",
        ),
        pytest.param(
            {"density_mixer": "diis", "density_mixer_depth": 7},
            "default Fock-DIIS route",
            id="diis-parameters",
        ),
    ],
)
def test_gdf_optimize_density_mixer_request_fails_before_manifest(
    tmp_path,
    dry_run,
    mixer_kwargs,
    message,
):
    """Explicit mixer validation precedes both dry and live execution."""
    sysp, basis = _h2_12bohr()
    case = "-".join(str(value) for value in mixer_kwargs.values())
    stem = tmp_path / f"h2-gdf-pbe0-gamma-mixer-{case}-{dry_run}"
    with pytest.raises(ValueError, match=message):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RKS",
            functional="pbe0",
            jk_method="gdf",
            kpoints=(1, 1, 1),
            optimize=True,
            dry_run=dry_run,
            output=stem,
            **mixer_kwargs,
            **_NO_SIDECARS,
        )
    assert not stem.with_suffix(".system").exists()


def test_gdf_optimize_hybrid_rks_multik_dry_run_stays_supported(tmp_path):
    sysp, basis = _h2_12bohr()
    stem = tmp_path / "h2-gdf-pbe0-multik-opt"
    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RKS",
        functional="pbe0",
        jk_method="gdf",
        kpoints=(2, 1, 1),
        optimize=True,
        dry_run=True,
        output=stem,
        **_NO_SIDECARS,
    )
    assert result is None
    assert stem.with_suffix(".system").exists()


def test_gdf_optimize_pure_rks_slab_dry_run_stays_supported(tmp_path):
    system, basis = _h2_slab()
    stem = tmp_path / "h2-gdf-pbe-slab-opt"
    result = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional="pbe",
        jk_method="gdf",
        kpoints=(2, 2, 1),
        optimize=True,
        dry_run=True,
        output=stem,
        **_NO_SIDECARS,
    )
    assert result is None
    assert stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
@pytest.mark.parametrize(
    ("control", "value", "message"),
    [
        ("fock_mixing", 0.25, "slab GDF optimization.*fock_mixing"),
        ("level_shift", 0.2, "slab GDF optimization.*level_shift"),
        (
            "density_mixer",
            "anderson",
            "slab GDF optimization.*supports Fock DIIS only",
        ),
    ],
    ids=["fock-mixing", "level-shift", "density-mixer"],
)
def test_gdf_optimize_slab_controls_fail_before_manifest(
    tmp_path,
    dry_run,
    control,
    value,
    message,
):
    system, basis = _h2_slab()
    stem = tmp_path / f"h2-gdf-slab-{control}-{dry_run}"
    with pytest.raises(NotImplementedError, match=message):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional="pbe",
            jk_method="gdf",
            kpoints=(2, 2, 1),
            optimize=True,
            dry_run=dry_run,
            output=stem,
            **{control: value},
            **_NO_SIDECARS,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
@pytest.mark.parametrize(
    ("control", "message"),
    [
        ("fock_mixing", "slab GDF optimization.*fock_mixing"),
        ("level_shift", "slab GDF optimization.*level_shift"),
    ],
    ids=["auto-fock-mixing", "auto-level-shift"],
)
def test_gdf_optimize_slab_auto_resolved_controls_fail_preflight(
    monkeypatch,
    tmp_path,
    dry_run,
    control,
    message,
):
    """Preflight validates finalized AUTO values, not only raw kwargs."""
    from vibeqc import periodic_runner

    real_resolve = periodic_runner.resolve_convergence_strategy

    def resolve_with_unsupported_auto(*args, **kwargs):
        strategy = real_resolve(*args, **kwargs)
        knobs = dict(strategy.knobs)
        resolution = knobs[control]
        knobs[control] = type(resolution)(
            0.25,
            "auto",
            "test profile selects a slab-unsupported control",
        )
        return type(strategy)(
            mode=strategy.mode,
            classification=strategy.classification,
            knobs=knobs,
        )

    monkeypatch.setattr(
        periodic_runner,
        "resolve_convergence_strategy",
        resolve_with_unsupported_auto,
    )
    system, basis = _h2_slab()
    stem = tmp_path / f"h2-gdf-slab-{control}-auto-{dry_run}"
    with pytest.raises(NotImplementedError, match=message):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional="pbe",
            jk_method="gdf",
            kpoints=(2, 2, 1),
            convergence="auto",
            optimize=True,
            dry_run=dry_run,
            output=stem,
            **_NO_SIDECARS,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
@pytest.mark.parametrize("mesh_kind", ["kpoints", "bloch"], ids=str)
def test_gdf_optimize_slab_object_mesh_fails_before_manifest(
    tmp_path,
    dry_run,
    mesh_kind,
):
    system, basis = _h2_slab()
    kpoints = vq.KPoints.gamma(system)
    if mesh_kind == "bloch":
        kpoints = kpoints.to_bloch_kmesh()
    stem = tmp_path / f"h2-gdf-slab-{mesh_kind}-{dry_run}"
    with pytest.raises(
        NotImplementedError,
        match="slab GDF optimization.*tuple mesh",
    ):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional="pbe",
            jk_method="gdf",
            kpoints=kpoints,
            optimize=True,
            dry_run=dry_run,
            output=stem,
            **_NO_SIDECARS,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
@pytest.mark.parametrize(
    ("control", "value", "message"),
    [
        (
            "rsgdf_tail_ke_cutoff",
            100.0,
            "slab GDF optimization.*high-\\|G\\| tail",
        ),
        (
            "bz_integration",
            "smearing",
            "slab GDF optimization.*integer zero-temperature",
        ),
    ],
    ids=["tail-cutoff", "bz-integration"],
)
def test_gdf_optimize_slab_adapter_options_fail_before_manifest(
    tmp_path,
    dry_run,
    control,
    value,
    message,
):
    system, basis = _h2_slab()
    stem = tmp_path / f"h2-gdf-slab-{control}-{dry_run}"
    with pytest.raises(NotImplementedError, match=message):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional="pbe",
            jk_method="gdf",
            kpoints=(2, 2, 1),
            optimize=True,
            dry_run=dry_run,
            output=stem,
            **{control: value},
            **_NO_SIDECARS,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("method", ["RKS", "UKS"])
@pytest.mark.parametrize("functional", ["vv10", "b2plyp"])
def test_gdf_unsupported_functionals_fail_before_dry_run(
    tmp_path,
    method,
    functional,
):
    sysp, basis = _h2_12bohr()
    stem = tmp_path / f"gdf-{method.lower()}-{functional}"
    with pytest.raises(NotImplementedError, match="VV10|double-hybrid"):
        vq.run_periodic_job(
            sysp,
            basis,
            method=method,
            functional=functional,
            jk_method="gdf",
            dry_run=True,
            output=stem,
            **_NO_SIDECARS,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("functional", ["vv10", "b2plyp"])
def test_direct_gdf_drivers_reject_unsupported_functionals(functional):
    from vibeqc.periodic_k_gdf import (
        run_krks_periodic_gdf,
        run_kuks_periodic_gdf,
    )

    sysp, basis = _h2_12bohr()
    with pytest.raises(NotImplementedError, match="VV10|double-hybrid"):
        vq.run_pbc_gdf_rhf(sysp, basis, functional=functional)
    with pytest.raises(NotImplementedError, match="VV10|double-hybrid"):
        vq.run_pbc_gdf_uks(sysp, basis, functional=functional)
    with pytest.raises(NotImplementedError, match="VV10|double-hybrid"):
        run_krks_periodic_gdf(
            sysp,
            basis,
            (2, 1, 1),
            functional=functional,
        )
    with pytest.raises(NotImplementedError, match="VV10|double-hybrid"):
        run_kuks_periodic_gdf(
            sysp,
            basis,
            (2, 1, 1),
            functional=functional,
        )


def test_gdf_relax_objective_is_free_energy_when_smeared():
    """Objective/gradient consistency for smeared relaxations: the GDF
    analytic gradient of a Fermi-Dirac smeared SCF is dA/dR of the
    Mermin free energy A = E - T.S (2026-07-30 smearing gradient
    envelope), so ``_relax_periodic_gdf_atoms`` must feed L-BFGS-B
    ``result.free_energy`` as the scalar objective, not
    ``result.energy``. Stub driver: A and E are quadratics with
    DIFFERENT minima along atom-0 z and the gradient is dA/dR; the
    relaxation must land on A's minimum (pairing the E surface with
    the A gradient would stall the line search away from either)."""
    from vibeqc.periodic_runner import _relax_periodic_gdf_atoms

    sysp, _ = _h2_12bohr(d=1.5)
    box = 12.0
    z_a_min = box / 2.0 - 0.9   # minimum of the free energy A
    z_e_min = box / 2.0 - 0.3   # minimum of the bare energy E

    class _Stub:
        pass

    def rerun(sys2, basis2):
        z = float(np.asarray(sys2.unit_cell[0].xyz, dtype=float)[2])
        r = _Stub()
        r.converged = True
        r.smearing_temperature = 0.05
        r.energy = (z - z_e_min) ** 2
        r.free_energy = (z - z_a_min) ** 2
        g = np.zeros((2, 3))
        g[0, 2] = 2.0 * (z - z_a_min)  # dA/dR
        r.gradient = g
        return r

    opt = _relax_periodic_gdf_atoms(
        sysp, "sto-3g", rerun, max_iter=60, conv_tol_grad=1e-8
    )
    z_final = float(
        np.asarray(opt.system.unit_cell[0].xyz, dtype=float)[2]
    )
    assert opt.converged
    assert abs(z_final - z_a_min) < 1e-6, (
        f"relaxed to z={z_final:.6f}; expected the free-energy minimum "
        f"{z_a_min:.6f} (energy minimum is {z_e_min:.6f})"
    )


def test_gdf_relax_preserves_programmatic_basis(monkeypatch):
    from types import SimpleNamespace
    from vibeqc.periodic_runner import _relax_periodic_gdf_atoms

    system, _ = _h2_12bohr()
    shells = [vq.ShellInfo(i, 1, True, [0.71], [1.], list(atom.xyz))
              for i, atom in enumerate(system.unit_cell)]
    template = vq.BasisSet(system.unit_cell_molecule(), shells, 'in-memory-p')
    evaluated = []

    def rerun(displaced, basis):
        assert basis.name == template.name
        for old, new in zip(template.shells(), basis.shells()):
            assert new.l == old.l
            np.testing.assert_array_equal(new.exponents, old.exponents)
            np.testing.assert_array_equal(new.coefficients, old.coefficients)
            np.testing.assert_array_equal(new.origin, displaced.unit_cell[new.atom_index].xyz)
        evaluated.append(displaced)
        return SimpleNamespace(converged=True, energy=0., gradient=np.zeros((2, 3)))

    def minimize(fun, x, **kwargs):
        trial = x.copy()
        trial[0] += 0.01
        value, gradient = fun(trial)
        return SimpleNamespace(x=trial, fun=value, jac=gradient, nit=1, success=True)

    monkeypatch.setattr('scipy.optimize.minimize', minimize)
    result = _relax_periodic_gdf_atoms(system, template, rerun, max_iter=2, conv_tol_grad=1e-8)
    assert result.converged and len(evaluated) == 1
    assert not np.array_equal(evaluated[0].unit_cell[0].xyz, system.unit_cell[0].xyz)


def test_gdf_optimizer_moves_ecp_centers_and_restores_on_failure():
    from types import SimpleNamespace
    from vibeqc.periodic_runner import _gdf_displaced_ecp_centers

    system, _ = _h2_12bohr(d=1.5)
    displaced, _ = _h2_12bohr(d=1.8)
    centers = [list(system.unit_cell[1].xyz)]
    opts = SimpleNamespace(ecp_home_centers=centers)
    with pytest.raises(RuntimeError, match='SCF failed'):
        with _gdf_displaced_ecp_centers(system, displaced, [opts, opts]):
            np.testing.assert_array_equal(opts.ecp_home_centers, [displaced.unit_cell[1].xyz])
            raise RuntimeError('SCF failed')
    assert opts.ecp_home_centers == centers


def test_gdf_optimizer_rejects_unowned_ecp_before_mutation():
    from types import SimpleNamespace
    from vibeqc.periodic_runner import _gdf_displaced_ecp_centers

    system, _ = _h2_12bohr()
    displaced, _ = _h2_12bohr(d=1.8)
    centers = [list(system.unit_cell[1].xyz)]
    valid = SimpleNamespace(ecp_home_centers=centers)
    invalid = SimpleNamespace(ecp_home_centers=[[100., 0., 0.]])
    with pytest.raises(ValueError, match='exactly one home atom'):
        with _gdf_displaced_ecp_centers(system, displaced, [valid, invalid]):
            pytest.fail('invalid projector was admitted')
    assert valid.ecp_home_centers == centers
