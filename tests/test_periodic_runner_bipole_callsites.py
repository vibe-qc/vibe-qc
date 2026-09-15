"""Regression + contract coverage for the BIPOLE dispatch callsites in
``run_periodic_job``.

Commit ``891acead`` ("feat(pbc): wire gilat through gdf and bipole rks") put
the new ``bz_integration`` keyword on the wrong callsite in
``periodic_runner.py`` and shipped two defects through eight consecutive
tags (v0.15.23..v0.15.30, fixed by ``38c4afed`` in v0.15.31):

1. Hard failure: the ``run_pbc_bipole_rhf`` callsite passed
   ``bz_integration=`` unconditionally while the callee did not accept it,
   so *every* RHF + ``jk_method="bipole"`` + multi-k run raised
   ``TypeError`` before any integral work — the user never had to pass
   ``bz_integration`` to trigger it.
2. Silent failure: ``run_pbc_bipole_rks`` accepted ``bz_integration`` but
   its callsite never passed it, so a user-requested ``"gilat"`` was
   silently dropped (that forwarding assertion lives in
   ``test_bz_integration_gilat_scf.py``).

The suite missed both for eight releases because no test drove RHF +
bipole at a real multi-k mesh on a 3-D cell through the public
``run_periodic_job`` route. This file closes that gap twice over: a live
regression test for the RHF route, and a static signature contract that
catches a swapped callsite for *any* ``run_pbc_bipole_*`` driver without
running an SCF (BUG-PER-002 in handovers/HANDOVER_OPEN_BUGS_V015.md).
"""

from __future__ import annotations

import ast
import importlib
import inspect
import tomllib
from types import SimpleNamespace

import numpy as np
import pytest
import vibeqc as vq

# The four BIPOLE driver homes; a run_pbc_bipole_* callsite in
# periodic_runner.py must resolve to a function defined in one of these.
_DRIVER_MODULES = (
    "vibeqc.pbc_bipole",
    "vibeqc.pbc_bipole_rks",
    "vibeqc.pbc_bipole_uhf",
    "vibeqc.pbc_bipole_uks",
)


def _h2_3d(box: float = 18.0):
    c = box / 2
    sysp = vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(1, [c, c, c - 0.7]), vq.Atom(1, [c, c, c + 0.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


@pytest.mark.parametrize("name", ["bipole-runner-custom", "sto-3g"])
def test_bipole_optimizer_receives_explicit_basis(monkeypatch, tmp_path, name):
    """The real SCF and the optimizer must retain one basis definition."""
    import vibeqc.bipole_optimize as optimizer

    system = vq.PeriodicSystem(3, 12*np.eye(3), [vq.Atom(2, [0, 0, 0])])
    basis = vq.BasisSet(system.unit_cell_molecule(), [
        vq.ShellInfo(0, 0, True, [1.2], [1.], [0, 0, 0]),
    ], name, False)

    class ReachedOptimizer(Exception):
        pass

    def capture(system_arg, basis_arg, *args, **kwargs):
        assert basis_arg is basis
        raise ReachedOptimizer

    monkeypatch.setattr(optimizer, "relax_atoms", capture)
    with pytest.raises(ReachedOptimizer):
        vq.run_periodic_job(
            system, basis, method="RHF", jk_method="bipole", kpoints=(1, 1, 1),
            optimize=True, bipole_cutoff_bohr=5., bipole_nuclear_cutoff_bohr=8.,
            ewald_omega=.6, sr_image_precision=None, initial_guess="HCORE",
            output=tmp_path / "custom-opt", output_qvf=False,
            write_population_file=False, citations=False, progress=False,
        )


def _forbid_gaussian_gamma_downstream(monkeypatch):
    """Make any output or BIPOLE SCF dispatch an immediate test failure."""

    def unexpected_downstream(*args, **kwargs):
        raise AssertionError(
            "origin-only Gaussian Gamma input reached output or SCF dispatch"
        )

    runner = importlib.import_module("vibeqc.periodic_runner")
    monkeypatch.setattr(runner, "ManifestUpdater", unexpected_downstream)
    monkeypatch.setattr(runner, "OutputWriter", unexpected_downstream)
    for module_name, method in zip(
        _DRIVER_MODULES,
        ("rhf", "rks", "uhf", "uks"),
        strict=True,
    ):
        monkeypatch.setattr(
            importlib.import_module(module_name),
            f"run_pbc_bipole_{method}",
            unexpected_downstream,
        )


@pytest.mark.parametrize(
    ("method", "functional"),
    [
        ("RHF", None),
        ("RKS", "pbe"),
        ("UHF", None),
        ("UKS", "pbe"),
    ],
)
@pytest.mark.parametrize(
    (
        "dry_run",
        "kpoints",
        "interaction",
        "nuclear",
        "ewald_omega",
        "home_only_role",
    ),
    [
        (False, None, 15.0, 20.0, None, "interaction"),
        (True, (1, 1, 1), 20.0, 15.0, 1.0, "nuclear"),
    ],
)
def test_wide_gaussian_gamma_ewald_reaches_output(
    monkeypatch,
    tmp_path,
    method,
    functional,
    dry_run,
    kpoints,
    interaction,
    nuclear,
    ewald_omega,
    home_only_role,
):
    """A home-only short-range list is valid with reciprocal Ewald coupling."""
    system, basis = _h2_3d(box=18.0)
    assert len(vq.direct_lattice_cells(system, 15.0)) == 1
    assert len(vq.direct_lattice_cells(system, 20.0)) > 1

    class ReachedOutput(Exception):
        pass

    def reached_output(*args, **kwargs):
        raise ReachedOutput

    runner = importlib.import_module("vibeqc.periodic_runner")
    monkeypatch.setattr(runner, "ManifestUpdater", reached_output)
    monkeypatch.setattr(runner, "OutputWriter", reached_output)
    stem = tmp_path / "must-not-exist" / f"wide-{method.lower()}-{home_only_role}"
    method_kwargs = {"functional": functional} if functional is not None else {}

    with pytest.raises(ReachedOutput):
        vq.run_periodic_job(
            system,
            basis,
            method=method,
            jk_method="bipole",
            kpoints=kpoints,
            bipole_cutoff_bohr=interaction,
            bipole_nuclear_cutoff_bohr=nuclear,
            ewald_omega=ewald_omega,
            dry_run=dry_run,
            output=stem,
            output_qvf=False,
            citations=False,
            progress=False,
            **method_kwargs,
        )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        (
            "bipole_cutoff_bohr",
            np.nan,
            r"bipole_cutoff_bohr must be finite and positive",
        ),
        (
            "bipole_nuclear_cutoff_bohr",
            np.inf,
            r"bipole_nuclear_cutoff_bohr must be finite and positive",
        ),
        ("ewald_omega", np.inf, r"ewald_omega must be finite and positive"),
        (
            "ewald_precision",
            np.nan,
            r"ewald_precision must be finite and in \(0, 1\)",
        ),
    ],
)
def test_nonfinite_bipole_preflight_control_rejects_before_output(
    monkeypatch,
    tmp_path,
    keyword,
    value,
    message,
):
    system, basis = _h2_3d(box=12.0)
    _forbid_gaussian_gamma_downstream(monkeypatch)
    stem = tmp_path / "must-not-exist" / f"nonfinite-{keyword}"

    with pytest.raises(
        ValueError,
        match=message,
    ):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="bipole",
            dry_run=True,
            output=stem,
            output_qvf=False,
            citations=False,
            progress=False,
            **{keyword: value},
        )

    assert not stem.parent.exists()


@pytest.mark.parametrize("method", ["RHF", "UHF", "RKS", "UKS"])
def test_home_only_erfc_public_scf_is_split_consistent(tmp_path, method):
    system, basis = _h2_3d(box=30.0)
    assert len(vq.direct_lattice_cells(system, 12.0)) == 1
    values = []
    for alpha in (0.4, 0.6):
        kwargs = {"functional": "lda"} if method.endswith("KS") else {}
        result = vq.run_periodic_job(
            system, basis, method=method, jk_method="bipole",
            kpoints=(1, 1, 1), bipole_cutoff_bohr=12.0,
            bipole_nuclear_cutoff_bohr=15.0, ewald_omega=alpha,
            conv_tol_energy=1e-10, max_iter=60,
            output=tmp_path / f"home-only-{method}-{alpha}",
            output_qvf=False, citations=False, progress=False,
            **kwargs,
        )
        assert result.converged
        values.append(result.energy)
    assert values[0] == pytest.approx(values[1], abs=1e-7)


def test_compact_gaussian_gamma_with_both_image_sets_remains_supported(tmp_path):
    system, basis = _h2_3d(box=12.0)
    stem = tmp_path / "compact-gamma"

    result = vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        jk_method="bipole",
        bipole_cutoff_bohr=15.0,
        bipole_nuclear_cutoff_bohr=15.0,
        dry_run=True,
        output=stem,
        output_qvf=False,
        citations=False,
        progress=False,
    )

    assert result is None
    assert stem.with_suffix(".system").exists()


def test_wide_gamma_uses_resolved_ewald_nuclear_cutoff(tmp_path):
    """Do not reject the raw nuclear radius when Ewald grows the executed one."""
    system, basis = _h2_3d(box=18.0)
    assert len(vq.direct_lattice_cells(system, 15.0)) == 1
    assert len(vq.direct_lattice_cells(system, 20.0)) > 1
    stem = tmp_path / "resolved-ewald-nuclear-cutoff"

    result = vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        jk_method="bipole",
        bipole_cutoff_bohr=20.0,
        bipole_nuclear_cutoff_bohr=15.0,
        dry_run=True,
        output=stem,
        output_qvf=False,
        citations=False,
        progress=False,
    )

    assert result is None
    assert stem.with_suffix(".system").exists()


def test_explicit_gapw_molecular_limit_remains_supported(tmp_path):
    """The existing explicit molecular declaration is not inferred away."""
    system, basis = _h2_3d(box=50.0)
    assert len(vq.direct_lattice_cells(system, 25.0)) == 1
    stem = tmp_path / "declared-gapw-molecular-limit"

    result = vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        jk_method="gapw",
        gapw_molecular_limit=True,
        dry_run=True,
        output=stem,
        output_qvf=False,
        citations=False,
        progress=False,
    )

    assert result is None
    assert stem.with_suffix(".system").exists()


def test_rhf_bipole_multik_reaches_scf(tmp_path):
    """RHF + bipole + multi-k through run_periodic_job must reach the SCF.

    Fails with ``TypeError: run_pbc_bipole_rhf() got an unexpected keyword
    argument 'bz_integration'`` on v0.15.23..v0.15.30 — note that
    ``bz_integration`` is deliberately NOT passed here; the broken callsite
    forwarded it unconditionally.

    Runs a converged SCF: ``run_periodic_job`` fails closed on
    non-convergence (the 2026-08-13 periodic SCF gate), so the
    callsite check needs the returned converged result.
    """
    sysp, basis = _h2_3d(box=18.0)
    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RHF",
        jk_method="bipole",
        kpoints=(2, 1, 1),
        max_iter=60,
        output=tmp_path / "rhf_bipole_multik",
        output_qvf=False,
        write_density=False,
        citations=False,
        progress=False,
    )
    assert result is not None
    assert np.isfinite(float(result.energy))


def _resolve_driver(name: str):
    for mod_name in _DRIVER_MODULES:
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, name, None)
        if fn is not None and getattr(fn, "__module__", None) == mod_name:
            return fn
    raise AssertionError(
        f"{name} is called from periodic_runner.py but is not defined in any "
        f"known BIPOLE driver module {_DRIVER_MODULES}; if a new driver was "
        "added, extend _DRIVER_MODULES so its callsites stay under contract."
    )


def test_bipole_callsites_bind_to_driver_signatures():
    """Every run_pbc_bipole_* callsite passes only keywords the callee accepts.

    Static contract: parses periodic_runner.py, collects the keyword names at
    each ``run_pbc_bipole_*`` call, and binds them against the driver's real
    ``inspect.signature``. A keyword wired to the wrong callsite (the
    v0.15.23..v0.15.30 defect class) fails here without running any SCF.
    """
    pr = importlib.import_module("vibeqc.periodic_runner")
    tree = ast.parse(inspect.getsource(pr))

    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            continue
        if name.startswith("run_pbc_bipole_"):
            sites.append((name, node))

    # The four live method dispatches must remain visible. Historical
    # fail-closed compatibility code may add more sites, but tests must not
    # require those retired backend-switching branches to exist.
    assert len(sites) >= 4, (
        f"expected >= 4 run_pbc_bipole_* callsites in periodic_runner.py, "
        f"found {len(sites)} — the AST scan no longer sees the dispatch"
    )
    assert {
        "run_pbc_bipole_rhf",
        "run_pbc_bipole_rks",
        "run_pbc_bipole_uhf",
        "run_pbc_bipole_uks",
    } <= {name for name, _ in sites}

    problems = []
    for name, node in sites:
        sig = inspect.signature(_resolve_driver(name))
        keywords = [k.arg for k in node.keywords if k.arg is not None]
        try:
            sig.bind_partial(
                *([None] * len(node.args)), **{k: None for k in keywords}
            )
        except TypeError as exc:
            problems.append(
                f"periodic_runner.py:{node.lineno}: {name}(...) passes "
                f"keyword(s) the driver signature rejects: {exc}"
            )
    assert not problems, "\n".join(problems)


def test_bipole_callsites_forward_public_symmetry_controls():
    """Every BIPOLE call must preserve the public symmetry decision."""
    pr = importlib.import_module("vibeqc.periodic_runner")
    tree = ast.parse(inspect.getsource(pr))

    problems = []
    n_sites = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (
            func.id
            if isinstance(func, ast.Name)
            else func.attr
            if isinstance(func, ast.Attribute)
            else ""
        )
        if not name.startswith("run_pbc_bipole_"):
            continue
        n_sites += 1
        values = {kw.arg: kw.value for kw in node.keywords if kw.arg is not None}
        expected = {
            "use_fock_symmetry": "symmetry_stabilize",
            "use_fock_symmetry_reduce": "symmetry_reduce_fock",
        }
        for keyword, public_name in expected.items():
            value = values.get(keyword)
            if not isinstance(value, ast.Name) or value.id != public_name:
                problems.append(
                    f"periodic_runner.py:{node.lineno}: {name}(...) must pass "
                    f"{keyword}={public_name}"
                )

    assert n_sites >= 6
    assert not problems, "\n".join(problems)


def test_bipole_public_symmetry_reduction_default_preserves_driver_auto():
    pr = importlib.import_module("vibeqc.periodic_runner")
    assert inspect.signature(pr.run_periodic_job).parameters[
        "symmetry_reduce_fock"
    ].default is None


def test_explicit_bipole_non_dense_solver_fails_before_dry_run_manifest(
    tmp_path,
):
    system, basis = _h2_3d(box=12.0)
    stem = tmp_path / "bipole-davidson"

    with pytest.raises(NotImplementedError, match="supports only.*dense"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="bipole",
            solver="davidson",
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    ("module_name", "driver_name", "options_cls"),
    [
        ("vibeqc.pbc_bipole", "run_pbc_bipole_rhf", vq.PeriodicRHFOptions),
        (
            "vibeqc.pbc_bipole_rks",
            "run_pbc_bipole_rks",
            vq.PeriodicKSOptions,
        ),
        (
            "vibeqc.pbc_bipole_uhf",
            "run_pbc_bipole_uhf",
            vq.PeriodicRHFOptions,
        ),
        (
            "vibeqc.pbc_bipole_uks",
            "run_pbc_bipole_uks",
            vq.PeriodicKSOptions,
        ),
    ],
)
def test_direct_bipole_drivers_reject_iterative_solver_options(
    module_name,
    driver_name,
    options_cls,
):
    system, basis = _h2_3d()
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])
    options = options_cls()
    options.use_davidson = True
    driver = getattr(importlib.import_module(module_name), driver_name)

    with pytest.raises(NotImplementedError, match="iterative diagonalization"):
        driver(system, basis, kmesh, options)


@pytest.mark.parametrize(
    ("module_name", "driver_name"),
    [
        ("vibeqc.pbc_bipole_rks", "run_pbc_bipole_rks"),
        ("vibeqc.pbc_bipole_uks", "run_pbc_bipole_uks"),
    ],
)
@pytest.mark.parametrize("damping", [-0.001, 1.0])
def test_direct_bipole_ks_drivers_reject_out_of_range_damping(
    module_name,
    driver_name,
    damping,
):
    multiplicity = 3 if driver_name.endswith("uks") else 1
    system, _ = _h2_3d(box=8.0)
    system.multiplicity = multiplicity
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])
    options = vq.PeriodicKSOptions()
    options.functional = "lda"
    options.damping = damping
    options.lattice_opts.cutoff_bohr = 2.0
    driver = getattr(importlib.import_module(module_name), driver_name)

    with pytest.raises(ValueError, match=r"damping must be in \[0,1\)"):
        driver(
            system,
            basis,
            kmesh,
            options,
            progress=False,
            sr_image_precision=None,
        )


@pytest.mark.parametrize(
    "kpoints",
    [
        SimpleNamespace(
            kpoints=[np.zeros(3), np.array([0.5, 0.0, 0.0])],
            mesh=(1, 1, 1),
            ir_mapping=[],
        ),
        SimpleNamespace(
            kpoints=[np.zeros(3)],
            mesh=(2, 2, 2),
            ir_mapping=np.zeros(8, dtype=int),
        ),
    ],
    ids=["explicit-list-placeholder-mesh", "one-point-reduced-mesh"],
)
def test_runner_helpers_classify_true_multik_requests(kpoints):
    pr = importlib.import_module("vibeqc.periodic_runner")

    assert pr._bloch_kmesh_full_size(kpoints) > 1
    assert pr._is_multik_kpoints(kpoints) is True


@pytest.mark.parametrize(
    ("task_kwargs", "label"),
    [
        ({"hessian": True}, "hessian"),
        ({"tddft": True}, "tddft"),
        ({"coop_cohp": True}, "coop_cohp"),
    ],
)
def test_bipole_surrogate_post_scf_tasks_fail_before_scf(
    tmp_path,
    task_kwargs,
    label,
):
    system, basis = _h2_3d(box=12.0)
    with pytest.raises(NotImplementedError, match=label):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="bipole",
            output=tmp_path / label,
            output_qvf=False,
            progress=False,
            **task_kwargs,
        )


def test_bipole_ecp_metadata_fails_before_scf(monkeypatch, tmp_path):
    pr = importlib.import_module("vibeqc.periodic_runner")
    system = vq.PeriodicSystem(
        3, 12.0 * np.eye(3),
        [vq.Atom(11, [0.0, 0.0, 0.0]), vq.Atom(1, [1.4, 0.0, 0.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    block = vq.ECPPrimitiveBlock()
    block.n_primitive = 1
    block.exponents = [1.0]
    block.coefficients = [1.0]
    block.ams = [0]
    block.ns = [2]
    monkeypatch.setattr(
        pr,
        "_resolve_ecp_data",
        lambda *_args: ([block], [[0.0, 0.0, 0.0]], [1.0, 1.0], 10),
    )

    with pytest.raises(NotImplementedError, match="ECP-bearing bases"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="bipole",
            output=tmp_path / "ecp",
            output_qvf=False,
            progress=False,
        )


@pytest.mark.parametrize(
    "flavor",
    ["methfessel-paxton", "marzari-vanderbilt"],
)
def test_bipole_non_fd_smearing_fails_before_scf(tmp_path, flavor):
    system, basis = _h2_3d(box=12.0)
    with pytest.raises(NotImplementedError, match="Fermi-Dirac occupations"):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional="lda",
            jk_method="bipole",
            smearing=vq.SmearingOptions(temperature=0.005, flavor=flavor),
            output=tmp_path / flavor,
            output_qvf=False,
            progress=False,
        )


def test_bipole_legacy_non_fd_smearing_fails_before_scf(tmp_path):
    system, basis = _h2_3d(box=12.0)
    with pytest.raises(NotImplementedError, match="Fermi-Dirac occupations"):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional="lda",
            jk_method="bipole",
            smearing_temperature=0.005,
            smearing_method="methfessel-paxton",
            output=tmp_path / "legacy-mp",
            output_qvf=False,
            progress=False,
        )


@pytest.mark.parametrize("method", ["RHF", "ROHF", "ROKS"])
def test_bipole_integer_occupation_smearing_fails_dry_run_preflight(
    tmp_path,
    method,
):
    system, basis = _h2_3d(box=12.0 if method == "RHF" else 18.0)
    with pytest.raises(NotImplementedError, match=r"integer.*occupations"):
        vq.run_periodic_job(
            system,
            basis,
            method=method,
            functional="lda" if method == "ROKS" else None,
            jk_method="bipole",
            smearing_temperature=0.005,
            dry_run=True,
            output=tmp_path / f"{method.lower()}-smearing-preflight",
            output_qvf=False,
            progress=False,
        )


def test_bipole_far_field_fails_before_dry_run_manifest(tmp_path):
    system, basis = _h2_3d(box=12.0)
    stem = tmp_path / "far-field-preflight"

    with pytest.raises(NotImplementedError, match="three-translation"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="bipole",
            use_multipole_far_field=True,
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


# --- #511: an explicit use_multipole_far_field is never silently dropped ----
#
# The flag reaches a consumer on exactly one route: the four direct BIPOLE
# drivers read it and nothing else in run_periodic_job does. Until #511 its
# only validation was the reject_bipole_quartet_far_field call above, which
# sat *inside* the BIPOLE preflight branch -- so an explicit True on any
# other jk_method reached neither a consumer nor a guard, and the job ran
# the exact traversal to completion reporting success.
#
# Measured on 0.15.150 before the fix, five routes dropped it in silence --
# auto, gdf, rijcosx, direct, aiccm2026dev-b -- and the dry-run options
# digest was byte-identical (75c9c04eb14d) between True and False, so not
# even the reproducibility record distinguished the two requests.
#
# fft_poisson (retired, v0.13.0) and aiccm2026dev-a (gated) are deliberately
# absent: they fail earlier under a different rule, so including them would
# make this test pass for the wrong reason.
_NON_BIPOLE_FAR_FIELD_ROUTES = [
    "auto",  # resolves to GDF for closed-shell RHF on dim=3
    "gdf",
    "rijcosx",
    "direct",
    "aiccm2026dev-b",
]


@pytest.mark.parametrize("jk_method", _NON_BIPOLE_FAR_FIELD_ROUTES)
def test_far_field_on_non_bipole_route_fails_closed(tmp_path, jk_method):
    """A route that cannot honour the flag must refuse it, not ignore it."""
    system, basis = _h2_3d()
    stem = tmp_path / f"far-field-{jk_method}"

    with pytest.raises(NotImplementedError, match="use_multipole_far_field"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method=jk_method,
            use_multipole_far_field=True,
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    # The refusal must precede the dry-run manifest, so queue preflight
    # cannot certify a request live dispatch would silently reinterpret.
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("jk_method", _NON_BIPOLE_FAR_FIELD_ROUTES)
def test_far_field_default_still_runs_on_non_bipole_route(tmp_path, jk_method):
    """Only an explicit True is refused; the default stays silent.

    The guard would be worthless if it also rejected the supported value --
    False selects the exact four-centre traversal on every route.
    """
    system, basis = _h2_3d()
    stem = tmp_path / f"far-field-default-{jk_method}"

    vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        jk_method=jk_method,
        use_multipole_far_field=False,
        dry_run=True,
        output=stem,
        output_qvf=False,
        progress=False,
    )
    assert stem.with_suffix(".system").exists()


def test_far_field_refusal_names_the_route_the_caller_asked_for(tmp_path):
    """The refusal must name the resolved route, not a route they never used.

    A fail-closed message is the caller's only pointer to which argument to
    change; #498 is the standing example of what a misattributed one costs.
    """
    system, basis = _h2_3d()

    with pytest.raises(NotImplementedError) as excinfo:
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="gdf",
            use_multipole_far_field=True,
            dry_run=True,
            output=tmp_path / "far-field-attribution",
            output_qvf=False,
            progress=False,
        )
    message = str(excinfo.value)
    assert "'gdf'" in message, message
    assert "use_multipole_far_field" in message, message


def test_far_field_type_contract_now_applies_off_bipole(tmp_path):
    """bool-or-None is enforced on every route, not on BIPOLE alone.

    Hoisting the guard out of the BIPOLE branch made
    ``reject_bipole_quartet_far_field`` the single owner of the type
    contract for all routes; before #511 a non-bool reached the GDF route
    unchecked.
    """
    system, basis = _h2_3d()

    with pytest.raises(TypeError, match="must be bool or None"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="gdf",
            use_multipole_far_field=0,  # int, not bool
            dry_run=True,
            output=tmp_path / "far-field-type",
            output_qvf=False,
            progress=False,
        )


@pytest.mark.parametrize(
    ("method", "functional", "kwargs"),
    [
        ("RKS", "lda", {"smearing_temperature": 0.005}),
        ("UKS", "lda", {"bz_integration": "gilat"}),
        ("UHF", None, {"bz_integration": "gilat"}),
    ],
)
def test_fractional_legacy_gauge_fails_before_dry_run_manifest(
    tmp_path,
    method,
    functional,
    kwargs,
):
    system, basis = _h2_3d(box=12.0)
    stem = tmp_path / f"{method.lower()}-fractional-legacy"
    with pytest.raises(NotImplementedError, match="fractional occupations"):
        vq.run_periodic_job(
            system,
            basis,
            method=method,
            functional=functional,
            jk_method="bipole",
            use_exchange_ewald_split=False,
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
            **kwargs,
        )
    assert not stem.with_suffix(".system").exists()


def test_auto_smearing_legacy_gauge_fails_before_dry_run_manifest(tmp_path):
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 6.0,
        [vq.Atom(4, [3.0, 3.0, 3.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "auto-smearing-legacy"
    with pytest.raises(NotImplementedError, match="fractional occupations"):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional="lda",
            jk_method="bipole",
            convergence="auto",
            use_exchange_ewald_split=False,
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    ("method", "bz_integration", "smearing_temperature", "match"),
    [
        ("RKS", "bogus", 0.0, "bz_integration must be"),
        ("RHF", "gilat", 0.0, "does not implement"),
        ("UKS", "gilat", 0.005, "T=0 integrator"),
    ],
)
def test_bipole_bz_integration_fails_before_dry_run_manifest(
    tmp_path,
    method,
    bz_integration,
    smearing_temperature,
    match,
):
    system, basis = _h2_3d(box=12.0)
    if method == "UKS":
        system = vq.PeriodicSystem(
            3,
            np.asarray(system.lattice),
            list(system.unit_cell),
            multiplicity=3,
        )
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / f"bz-preflight-{method.lower()}"

    with pytest.raises((ValueError, NotImplementedError), match=match):
        vq.run_periodic_job(
            system,
            basis,
            method=method,
            functional="lda" if method in ("RKS", "UKS") else None,
            jk_method="bipole",
            bz_integration=bz_integration,
            smearing_temperature=smearing_temperature,
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"exchange_exxdiv": "bogus"}, "exchange_exxdiv must be"),
        ({"sr_image_precision": 0.0}, "sr_image_precision must be"),
        (
            {"use_oda": True, "use_diis": True},
            "use_oda and use_diis are mutually exclusive",
        ),
        (
            {"use_oda": True, "use_diis": False, "oda_trust_lambda_max": 1.1},
            "oda_trust_lambda_max must be",
        ),
        (
            {
                "bipole_exact_zone_bohr": 5.0,
                "use_exchange_ewald_split": False,
            },
            "requires the corrected Ewald exchange split",
        ),
        (
            {"bipole_exact_zone_bohr": 5.0, "sr_image_precision": None},
            "requires the padded short-range image path",
        ),
        (
            {
                "bipole_exact_zone_bohr": 5.0,
                "symmetry": "attach",
                "symmetry_reduce_fock": True,
            },
            "not wired for explicit Fock symmetry",
        ),
        (
            {
                "bipole_exact_zone_bohr": 15.0,
                "bipole_cutoff_bohr": 15.0,
            },
            "strictly below the BIPOLE operator cutoff",
        ),
    ],
)
def test_bipole_raw_controls_fail_before_dry_run_manifest(
    tmp_path,
    kwargs,
    match,
):
    system, basis = _h2_3d(box=12.0)
    stem = tmp_path / ("raw-preflight-" + match.split()[0])

    with pytest.raises((ValueError, NotImplementedError), match=match):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="bipole",
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
            **kwargs,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("method", ["RHF", "RKS", "UHF", "UKS"])
def test_auto_gamma_dft_plus_u_plans_bipole_before_dry_run(tmp_path, method):
    system, basis = _h2_3d(box=12.0)
    if method in ("UHF", "UKS"):
        system = vq.PeriodicSystem(
            3,
            np.asarray(system.lattice),
            list(system.unit_cell),
            multiplicity=3,
        )
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / f"auto-plus-u-{method.lower()}"

    result = vq.run_periodic_job(
        system,
        basis,
        method=method,
        functional="lda" if method in ("RKS", "UKS") else None,
        dft_plus_u=[vq.HubbardSite(0, 0, U_ev=2.0)],
        dry_run=True,
        output=stem,
        output_qvf=False,
        progress=False,
    )

    assert result is None
    manifest = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))
    run = manifest["run"]
    assert run["jk_method_requested"] == "auto"
    assert run["jk_method_resolved"] == "bipole"
    assert run["jk_method_executed"] == "bipole"
    assert run["dft_plus_u_route"] == f"bipole_{method.lower()}_gamma"


@pytest.mark.parametrize("method", ["RHF", "RKS"])
def test_auto_multik_dft_plus_u_plans_native_gdf_before_dry_run(
    tmp_path,
    method,
):
    system, basis = _h2_3d()
    stem = tmp_path / f"auto-plus-u-gdf-{method.lower()}"

    result = vq.run_periodic_job(
        system,
        basis,
        method=method,
        functional="lda" if method == "RKS" else None,
        kpoints=(2, 1, 1),
        dft_plus_u=[vq.HubbardSite(0, 0, U_ev=2.0)],
        dry_run=True,
        output=stem,
        output_qvf=False,
        progress=False,
    )

    assert result is None
    manifest = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))
    run = manifest["run"]
    assert run["jk_method_resolved"] == "gdf"
    assert run["jk_method_executed"] == "gdf"
    assert run["dft_plus_u_route"] == f"gdf_{method.lower()}_multi_k"


@pytest.mark.parametrize(
    ("jk_method", "method"),
    [
        ("gpw", "RHF"),
        ("gpw", "UHF"),
        ("gpw", "UKS"),
        ("gapw", "RHF"),
        ("gapw", "UHF"),
        ("gapw", "UKS"),
    ],
)
def test_invalid_multik_gpw_gapw_plus_u_fails_before_dry_run(
    tmp_path,
    jk_method,
    method,
):
    system, basis = _h2_3d()
    if method in ("UHF", "UKS"):
        system = vq.PeriodicSystem(
            3,
            np.asarray(system.lattice),
            list(system.unit_cell),
            multiplicity=3,
        )
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / f"invalid-{jk_method}-{method.lower()}-plus-u"

    with pytest.raises(NotImplementedError, match=r"multi-k .* DFT\+U"):
        vq.run_periodic_job(
            system,
            basis,
            method=method,
            functional="lda" if method == "UKS" else None,
            jk_method=jk_method,
            gapw_molecular_limit=(jk_method == "gapw" and method != "UKS"),
            kpoints=(2, 1, 1),
            dft_plus_u=[vq.HubbardSite(0, 0, U_ev=2.0)],
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("jk_method", ["gpw", "gapw"])
def test_explicit_unit_rks_plus_u_provenance_matches_multik_dispatch(
    tmp_path,
    jk_method,
):
    system, basis = _h2_3d()
    stem = tmp_path / f"unit-rks-{jk_method}-plus-u"

    result = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional="lda",
        jk_method=jk_method,
        kpoints=(1, 1, 1),
        dft_plus_u=[vq.HubbardSite(0, 0, U_ev=2.0)],
        dry_run=True,
        output=stem,
        output_qvf=False,
        progress=False,
    )

    assert result is None
    manifest = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))
    assert manifest["run"]["dft_plus_u_route"] == f"{jk_method}_rks_multi_k"


@pytest.mark.parametrize("jk_method", ["gpw", "gapw"])
@pytest.mark.parametrize("kpoints", [(1, 1, 1), (2, 1, 1)])
def test_multik_gpw_gapw_hybrid_rks_plus_u_fails_before_dry_run(
    tmp_path,
    jk_method,
    kpoints,
):
    system, basis = _h2_3d()
    stem = tmp_path / f"hybrid-{jk_method}-plus-u-{'x'.join(map(str, kpoints))}"

    with pytest.raises(NotImplementedError, match="hybrid or range-separated"):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional="pbe0",
            jk_method=jk_method,
            kpoints=kpoints,
            dft_plus_u=[vq.HubbardSite(0, 0, U_ev=2.0)],
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    ("dim", "method"),
    [(1, "RHF"), (1, "UHF"), (1, "UKS"), (2, "UKS")],
)
def test_auto_low_dimensional_dft_plus_u_fails_before_dry_run(
    tmp_path,
    dim,
    method,
):
    atoms = [vq.Atom(1, [9.0, 9.0, 8.3]), vq.Atom(1, [9.0, 9.0, 9.7])]
    if dim == 1:
        system = vq.PeriodicSystem(
            1,
            np.diag([18.0, 18.0, 18.0]),
            atoms,
            multiplicity=3 if method in ("UHF", "UKS") else 1,
        )
    else:
        system = vq.slab_2d(
            [18.0, 0.0, 0.0],
            [0.0, 18.0, 0.0],
            atoms,
            multiplicity=3,
        )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / f"low-d-plus-u-{dim}-{method.lower()}"

    with pytest.raises(NotImplementedError, match="low-dimensional"):
        vq.run_periodic_job(
            system,
            basis,
            method=method,
            functional="lda" if method == "UKS" else None,
            dft_plus_u=[vq.HubbardSite(0, 0, U_ev=2.0)],
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


def test_auto_gamma_dft_plus_u_inherits_bipole_post_scf_guards(tmp_path):
    system, basis = _h2_3d(box=12.0)

    with pytest.raises(NotImplementedError, match="hessian"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            dft_plus_u=[vq.HubbardSite(0, 0, U_ev=2.0)],
            hessian=True,
            output=tmp_path / "auto-plus-u-hessian",
            output_qvf=False,
            progress=False,
        )


def test_bipole_qvf_skips_generic_surrogate_properties():
    pr = importlib.import_module("vibeqc.periodic_runner")
    assert not pr._qvf_periodic_property_payload_supported(
        vq.PeriodicJKMethod.BIPOLE
    )
    assert pr._qvf_periodic_property_payload_supported(vq.PeriodicJKMethod.GDF)
    assert not pr._qvf_periodic_property_payload_supported(
        vq.PeriodicJKMethod.GDF,
        uses_external_xc=True,
    )


def test_periodic_qvf_wavefunction_gate_accepts_open_shell_mos():
    pr = importlib.import_module("vibeqc.periodic_runner")
    result = SimpleNamespace(
        mo_coeffs_alpha=[np.eye(2)],
        mo_coeffs_beta=[np.eye(2)],
    )
    assert not pr._has_valid_mo_coeffs(result)
    assert pr._should_prepare_periodic_qvf_wavefunction(True, result)
    assert not pr._should_prepare_periodic_qvf_wavefunction(False, result)


def test_periodic_terminal_checkpoint_uses_result_convergence():
    pr = importlib.import_module("vibeqc.periodic_runner")
    tree = ast.parse(inspect.getsource(pr))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "finalize"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "_checkpointer"
    ]
    assert len(calls) == 1
    status = calls[0].args[0]
    assert isinstance(status, ast.Call)
    assert isinstance(status.func, ast.Name)
    assert status.func.id == "_checkpoint_terminal_run_status"
    assert len(status.args) == 1
    assert isinstance(status.args[0], ast.Name)
    assert status.args[0].id == "result"


@pytest.mark.parametrize(
    ("task_kwargs", "match"),
    [
        ({"optimize": True, "dispersion": "d3bj"}, "dispersion"),
        ({"optimize": True, "symmetry_stabilize": True}, "Fock symmetry"),
        (
            {
                "optimize": True,
                "optimize_cell": True,
            },
            "variable-cell",
        ),
    ],
)
def test_bipole_unsafe_optimizer_objectives_fail_before_scf(
    tmp_path,
    task_kwargs,
    match,
):
    system, basis = _h2_3d(box=12.0)
    with pytest.raises(NotImplementedError, match=match):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="bipole",
            output=tmp_path / "unsafe-opt",
            output_qvf=False,
            progress=False,
            **task_kwargs,
        )


def test_optimize_cell_requires_optimize(tmp_path):
    system, basis = _h2_3d(box=12.0)
    with pytest.raises(ValueError, match="requires optimize=True"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="bipole",
            optimize_cell=True,
            output=tmp_path / "cell-only",
            output_qvf=False,
            progress=False,
        )


def test_explicit_bipole_symmetry_requires_attached_model(tmp_path):
    system, basis = _h2_3d(box=12.0)
    with pytest.raises(ValueError, match="requires an attached symmetry model"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="bipole",
            symmetry_reduce_fock=True,
            output=tmp_path / "symmetry-no-model",
            output_qvf=False,
            progress=False,
        )


@pytest.mark.parametrize("method", ["UHF", "UKS"])
def test_broken_spin_state_rejects_explicit_fock_symmetry_before_dry_run(
    tmp_path,
    method,
):
    system, basis = _h2_3d(box=12.0)
    stem = tmp_path / f"{method.lower()}-broken-symmetry"
    with pytest.raises(NotImplementedError, match="atomic_spins"):
        vq.run_periodic_job(
            system,
            basis,
            method=method,
            functional="lda" if method == "UKS" else None,
            jk_method="bipole",
            symmetry="attach",
            symmetry_reduce_fock=True,
            atomic_spins=[1, -1],
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    ("module_name", "driver_name"),
    [
        ("vibeqc.pbc_bipole", "run_pbc_bipole_rhf"),
        ("vibeqc.pbc_bipole_rks", "run_pbc_bipole_rks"),
        ("vibeqc.pbc_bipole_uhf", "run_pbc_bipole_uhf"),
        ("vibeqc.pbc_bipole_uks", "run_pbc_bipole_uks"),
    ],
)
def test_direct_bipole_drivers_reject_ecp_options(module_name, driver_name):
    system, basis = _h2_3d()
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])
    driver = getattr(importlib.import_module(module_name), driver_name)

    with pytest.raises(NotImplementedError, match="ECP metadata"):
        driver(
            system,
            basis,
            kmesh,
            SimpleNamespace(ecp_total_ncore=2),
        )


@pytest.mark.parametrize(
    ("module_name", "driver_name", "options_cls"),
    [
        ("vibeqc.pbc_bipole", "run_pbc_bipole_rhf", vq.PeriodicRHFOptions),
        (
            "vibeqc.pbc_bipole_rks",
            "run_pbc_bipole_rks",
            vq.PeriodicKSOptions,
        ),
        (
            "vibeqc.pbc_bipole_uhf",
            "run_pbc_bipole_uhf",
            vq.PeriodicRHFOptions,
        ),
        (
            "vibeqc.pbc_bipole_uks",
            "run_pbc_bipole_uks",
            vq.PeriodicKSOptions,
        ),
    ],
)
def test_direct_bipole_drivers_reject_ecp_paired_basis_names(
    module_name,
    driver_name,
    options_cls,
):
    # LANL2DZ replaces 60 core electrons on platinum. The guard decides per
    # element through the registry, so the cell has to contain an atom the
    # sidecar actually covers; H2/LANL2DZ is an all-electron request.
    a = 5.0 * 1.8897261246257702  # angstrom -> bohr
    system = vq.PeriodicSystem(3, np.eye(3) * a, [vq.Atom(78, [0.0, 0.0, 0.0])])
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])
    driver = getattr(importlib.import_module(module_name), driver_name)

    with pytest.raises(NotImplementedError, match="ECP metadata"):
        driver(
            system,
            SimpleNamespace(name="lanl2dz"),
            kmesh,
            options_cls(),
        )


def test_direct_bipole_guard_admits_light_cell_in_ecp_family_basis():
    """LANL2DZ carries no ECP for hydrogen: the request is all-electron."""
    from vibeqc.pbc_bipole_common import reject_bipole_ecp_options

    reject_bipole_ecp_options(
        SimpleNamespace(),
        driver="direct-test",
        basis=SimpleNamespace(name="lanl2dz"),
        system=SimpleNamespace(unit_cell=[SimpleNamespace(Z=1)]),
    )


@pytest.mark.parametrize("basis_name", ["def2-tzvp", "pob-tzvp-rev2"])
def test_direct_bipole_guard_rejects_element_dependent_ecp_bases(basis_name):
    from vibeqc.pbc_bipole_common import reject_bipole_ecp_options

    heavy_system = SimpleNamespace(unit_cell=[SimpleNamespace(Z=79)])
    with pytest.raises(NotImplementedError, match="ECP metadata"):
        reject_bipole_ecp_options(
            SimpleNamespace(),
            driver="direct-test",
            basis=SimpleNamespace(name=basis_name),
            system=heavy_system,
        )


def test_direct_bipole_guard_allows_bundled_all_electron_pob_record():
    from vibeqc.pbc_bipole_common import reject_bipole_ecp_options

    reject_bipole_ecp_options(
        SimpleNamespace(),
        driver="direct-test",
        basis=SimpleNamespace(name="pob-tzvp-rev2"),
        system=SimpleNamespace(unit_cell=[SimpleNamespace(Z=8)]),
    )


@pytest.mark.parametrize(
    ("kwargs", "label"),
    [
        ({"symmetry_stabilize": True}, "symmetry_stabilize"),
        ({"symmetry_reduce_fock": True}, "symmetry_reduce_fock"),
        ({"ewald_precision": 1.0e-10}, "ewald_precision"),
        ({"use_oda": True, "use_diis": False}, "use_oda"),
        ({"use_mom": True}, "use_mom"),
        ({"ewald_omega": 0.4}, "ewald_omega"),
        ({"use_exchange_ewald_split": False}, "use_exchange_ewald_split"),
        ({"exchange_exxdiv": "none"}, "exchange_exxdiv"),
        ({"fmixing_percent": 10.0}, "fock_mixing/fmixing_percent"),
    ],
)
def test_restricted_open_shell_bipole_rejects_ignored_explicit_knobs(
    tmp_path,
    kwargs,
    label,
):
    system, basis = _h2_3d()
    stem = tmp_path / f"rohf-{label.replace('/', '-')}"
    with pytest.raises(NotImplementedError, match=label):
        vq.run_periodic_job(
            system,
            basis,
            method="ROHF",
            jk_method="bipole",
            symmetry="attach",
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
            **kwargs,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    ("module_name", "driver_name", "options_cls"),
    [
        (
            "vibeqc.pbc_bipole",
            "run_pbc_bipole_rhf",
            vq.PeriodicRHFOptions,
        ),
        (
            "vibeqc.pbc_bipole_rks",
            "run_pbc_bipole_rks",
            vq.PeriodicKSOptions,
        ),
    ],
)
def test_direct_closed_shell_bipole_multik_read_requires_lattice_density(
    module_name,
    driver_name,
    options_cls,
):
    system, basis = _h2_3d()
    kmesh = vq.monkhorst_pack(system, [2, 1, 1])
    options = options_cls()
    options.initial_guess = vq.InitialGuess.READ
    driver = getattr(importlib.import_module(module_name), driver_name)

    with pytest.raises(NotImplementedError, match="real-space initial_density"):
        driver(system, basis, kmesh, options)


@pytest.mark.parametrize(
    ("module_name", "driver_name", "options_cls"),
    [
        (
            "vibeqc.pbc_bipole_uhf",
            "run_pbc_bipole_uhf",
            vq.PeriodicRHFOptions,
        ),
        (
            "vibeqc.pbc_bipole_uks",
            "run_pbc_bipole_uks",
            vq.PeriodicKSOptions,
        ),
    ],
)
def test_direct_open_shell_bipole_multik_read_requires_lattice_densities(
    module_name,
    driver_name,
    options_cls,
):
    system, basis = _h2_3d()
    system.multiplicity = 3
    kmesh = vq.monkhorst_pack(system, [2, 1, 1])
    options = options_cls()
    options.initial_guess = vq.InitialGuess.READ
    driver = getattr(importlib.import_module(module_name), driver_name)

    with pytest.raises(NotImplementedError, match="init_alpha/init_beta"):
        driver(system, basis, kmesh, options)


@pytest.mark.parametrize("method", ["UHF", "UKS"])
def test_public_bipole_rejects_unrestricted_oda_before_dry_run(
    tmp_path,
    method,
):
    system, basis = _h2_3d(box=12.0)
    system.multiplicity = 3
    stem = tmp_path / f"{method.lower()}-oda"

    with pytest.raises(NotImplementedError, match="BIPOLE ODA"):
        vq.run_periodic_job(
            system,
            basis,
            method=method,
            functional="lda" if method == "UKS" else None,
            jk_method="bipole",
            use_oda=True,
            use_diis=False,
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    "occupation_kwargs",
    [
        {"smearing_temperature": 0.01},
        {"bz_integration": "gilat", "kpoints": (2, 1, 1)},
    ],
)
def test_public_bipole_rejects_fractional_oda_before_dry_run(
    tmp_path,
    occupation_kwargs,
):
    system, basis = _h2_3d(
        box=12.0 if "kpoints" not in occupation_kwargs else 18.0
    )
    stem = tmp_path / "rks-fractional-oda"

    with pytest.raises(NotImplementedError, match="BIPOLE ODA"):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional="lda",
            jk_method="bipole",
            use_oda=True,
            use_diis=False,
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
            **occupation_kwargs,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    ("module_name", "driver_name", "options_cls", "functional"),
    [
        (
            "vibeqc.pbc_bipole",
            "run_pbc_bipole_rhf",
            vq.PeriodicRHFOptions,
            None,
        ),
        (
            "vibeqc.pbc_bipole_rks",
            "run_pbc_bipole_rks",
            vq.PeriodicKSOptions,
            "lda",
        ),
        (
            "vibeqc.pbc_bipole_uhf",
            "run_pbc_bipole_uhf",
            vq.PeriodicRHFOptions,
            None,
        ),
        (
            "vibeqc.pbc_bipole_uks",
            "run_pbc_bipole_uks",
            vq.PeriodicKSOptions,
            "lda",
        ),
    ],
)
def test_direct_bipole_rejects_oda(
    module_name,
    driver_name,
    options_cls,
    functional,
):
    system, basis = _h2_3d()
    if driver_name.endswith(("uhf", "uks")):
        system.multiplicity = 3
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])
    options = options_cls()
    options.use_diis = False
    if functional is not None:
        options.functional = functional
    driver = getattr(importlib.import_module(module_name), driver_name)

    with pytest.raises(NotImplementedError, match="ODA"):
        driver(
            system,
            basis,
            kmesh,
            options,
            use_oda=True,
        )


def test_auto_plus_u_bipole_fallback_rechecks_dense_solver(tmp_path):
    system, basis = _h2_3d(box=12.0)
    stem = tmp_path / "auto-plus-u-davidson"

    with pytest.raises(NotImplementedError, match="solver='dense'"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            dft_plus_u=[vq.HubbardSite(0, 0, U_ev=2.0)],
            solver="davidson",
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


def test_kpoints_smearing_rejects_bipole_optimization_before_dry_run(
    tmp_path,
):
    """KPoints-carried finite T has the same preflight as explicit smearing."""
    system, basis = _h2_3d(box=12.0)
    kpoints = vq.KPoints.monkhorst_pack(system, (1, 1, 1))
    kpoints.smearing = vq.SmearingOptions(
        temperature=0.005,
        flavor="fermi-dirac",
    )
    stem = tmp_path / "kpoints-smearing-optimize"

    with pytest.raises(NotImplementedError, match="finite-temperature BIPOLE"):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional="lda",
            jk_method="bipole",
            kpoints=kpoints,
            optimize=True,
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


def test_bipole_optimizer_rejects_ad_hoc_multik_before_dry_run(tmp_path):
    system, basis = _h2_3d()
    kpoints = vq.KPoints.from_list(
        system,
        [[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0]],
    )
    stem = tmp_path / "bipole-optimize-ad-hoc-kmesh"

    with pytest.raises(NotImplementedError, match="complete Monkhorst-Pack"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="bipole",
            kpoints=kpoints,
            optimize=True,
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


def test_bipole_optimizer_rejects_legacy_multik_before_dry_run(tmp_path):
    system, basis = _h2_3d()
    stem = tmp_path / "bipole-optimize-legacy-multik"

    with pytest.raises(NotImplementedError, match="corrected Ewald"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="bipole",
            kpoints=(2, 1, 1),
            use_exchange_ewald_split=False,
            optimize=True,
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("functional", ["vv10", "b2plyp"])
@pytest.mark.parametrize("method", ["RKS", "UKS"])
def test_bipole_rejects_incomplete_ks_functional_before_dry_run(
    tmp_path,
    functional,
    method,
):
    system, basis = _h2_3d(box=12.0)
    stem = tmp_path / f"bipole-{method.lower()}-{functional}"
    with pytest.raises(NotImplementedError):
        vq.run_periodic_job(
            system,
            basis,
            method=method,
            functional=functional,
            jk_method="bipole",
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("method", ["UHF", "UKS"])
def test_public_bipole_multik_spin_schedule_fails_before_dry_run(
    tmp_path,
    method,
):
    system, basis = _h2_3d()
    system.multiplicity = 3
    stem = tmp_path / f"{method.lower()}-multik-spin-schedule"

    with pytest.raises(NotImplementedError, match="SPIN_SCHEDULE is Gamma-only"):
        vq.run_periodic_job(
            system,
            basis,
            method=method,
            functional="lda" if method == "UKS" else None,
            jk_method="bipole",
            kpoints=(2, 1, 1),
            spinlock="spin_schedule",
            spinlock_value=2,
            spinlock_iterations=2,
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


def test_atomic_spin_length_fails_before_dry_run(tmp_path):
    system, basis = _h2_3d(box=12.0)
    system.multiplicity = 3
    stem = tmp_path / "bad-atomic-spin-length"

    with pytest.raises(ValueError, match="atomic_spins length.*atom count"):
        vq.run_periodic_job(
            system,
            basis,
            method="UHF",
            jk_method="bipole",
            atomic_spins=[1],
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    ("method", "functional"),
    [("RHF", None), ("RKS", "lda"), ("UHF", None), ("UKS", "lda")],
)
def test_public_bipole_rejects_lone_non_gamma_twist_before_dry_run(
    tmp_path,
    method,
    functional,
):
    system, basis = _h2_3d()
    if method in ("UHF", "UKS"):
        system.multiplicity = 3
    kpoints = vq.KPoints.from_list(system, [[0.25, 0.0, 0.0]])
    stem = tmp_path / f"{method.lower()}-lone-twist"

    with pytest.raises(NotImplementedError, match="lone non-Gamma"):
        vq.run_periodic_job(
            system,
            basis,
            method=method,
            functional=functional,
            jk_method="bipole",
            kpoints=kpoints,
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


def test_auto_plus_u_bipole_fallback_rechecks_lone_twist_after_planning(
    tmp_path,
):
    """AUTO capability checks must apply to the finalized +U route."""
    system, basis = _h2_3d()
    kpoints = vq.KPoints.from_list(system, [[0.25, 0.0, 0.0]])
    stem = tmp_path / "auto-plus-u-lone-twist"

    with pytest.raises(NotImplementedError, match="lone non-Gamma"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            kpoints=kpoints,
            dft_plus_u=[vq.HubbardSite(0, 0, U_ev=2.0)],
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    ("module_name", "driver_name", "options_cls", "functional"),
    [
        ("vibeqc.pbc_bipole", "run_pbc_bipole_rhf", vq.PeriodicRHFOptions, None),
        ("vibeqc.pbc_bipole_rks", "run_pbc_bipole_rks", vq.PeriodicKSOptions, "lda"),
        ("vibeqc.pbc_bipole_uhf", "run_pbc_bipole_uhf", vq.PeriodicRHFOptions, None),
        ("vibeqc.pbc_bipole_uks", "run_pbc_bipole_uks", vq.PeriodicKSOptions, "lda"),
    ],
)
def test_direct_bipole_rejects_lone_non_gamma_twist(
    module_name,
    driver_name,
    options_cls,
    functional,
):
    system, basis = _h2_3d()
    if driver_name.endswith(("uhf", "uks")):
        system.multiplicity = 3
    kmesh = vq.KPoints.from_list(
        system, [[0.25, 0.0, 0.0]]
    ).to_bloch_kmesh()
    options = options_cls()
    if functional is not None:
        options.functional = functional
    driver = getattr(importlib.import_module(module_name), driver_name)

    with pytest.raises(NotImplementedError, match="lone non-Gamma"):
        driver(system, basis, kmesh, options)


def test_explicit_exchange_split_requires_complete_mp_mesh_before_dry_run(
    tmp_path,
):
    system, basis = _h2_3d()
    kpoints = vq.KPoints.from_list(
        system,
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
    )
    stem = tmp_path / "explicit-list-corrected-exchange"

    with pytest.raises(NotImplementedError, match="complete Monkhorst-Pack"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="bipole",
            kpoints=kpoints,
            use_exchange_ewald_split=True,
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


def test_gilat_requires_complete_mp_mesh_before_dry_run(tmp_path):
    system, basis = _h2_3d()
    kpoints = vq.KPoints.from_list(
        system,
        [[0.0, 0.0, 0.0], [0.25, 0.0, 0.0]],
    )
    stem = tmp_path / "explicit-list-gilat"

    with pytest.raises(NotImplementedError, match="complete Monkhorst-Pack"):
        vq.run_periodic_job(
            system,
            basis,
            method="RKS",
            functional="lda",
            jk_method="bipole",
            kpoints=kpoints,
            bz_integration="gilat",
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    ("module_name", "driver_name", "options_cls", "functional"),
    [
        (
            "vibeqc.pbc_bipole_uhf",
            "run_pbc_bipole_uhf",
            vq.PeriodicRHFOptions,
            None,
        ),
        (
            "vibeqc.pbc_bipole_uks",
            "run_pbc_bipole_uks",
            vq.PeriodicKSOptions,
            "lda",
        ),
    ],
)
def test_direct_bipole_multik_spin_schedule_fails_before_phase_one(
    module_name,
    driver_name,
    options_cls,
    functional,
):
    system, basis = _h2_3d()
    system.multiplicity = 3
    kmesh = vq.monkhorst_pack(system, [2, 1, 1])
    options = options_cls()
    options.spinlock_mode = vq.SpinlockMode.SPIN_SCHEDULE
    options.spinlock_value = 2
    options.spinlock_iterations = 2
    if functional is not None:
        options.functional = functional
    driver = getattr(importlib.import_module(module_name), driver_name)

    with pytest.raises(NotImplementedError, match="SPIN_SCHEDULE is Gamma-only"):
        driver(system, basis, kmesh, options)


@pytest.mark.parametrize(
    ("module_name", "driver_name", "options_cls", "functional"),
    [
        (
            "vibeqc.pbc_bipole",
            "run_pbc_bipole_rhf",
            vq.PeriodicRHFOptions,
            None,
        ),
        (
            "vibeqc.pbc_bipole_rks",
            "run_pbc_bipole_rks",
            vq.PeriodicKSOptions,
            "lda",
        ),
        (
            "vibeqc.pbc_bipole_uhf",
            "run_pbc_bipole_uhf",
            vq.PeriodicRHFOptions,
            None,
        ),
        (
            "vibeqc.pbc_bipole_uks",
            "run_pbc_bipole_uks",
            vq.PeriodicKSOptions,
            "lda",
        ),
    ],
)
def test_direct_bipole_rejects_nonpositive_max_iter(
    module_name,
    driver_name,
    options_cls,
    functional,
):
    system, basis = _h2_3d()
    if driver_name.endswith(("uhf", "uks")):
        system.multiplicity = 3
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])
    options = options_cls()
    options.max_iter = 0
    if functional is not None:
        options.functional = functional
    driver = getattr(importlib.import_module(module_name), driver_name)

    with pytest.raises(ValueError, match="max_iter must be at least 1"):
        driver(system, basis, kmesh, options)


def test_public_bipole_rejects_nonpositive_max_iter_before_dry_run(tmp_path):
    system, basis = _h2_3d(box=12.0)
    stem = tmp_path / "zero-max-iter"

    with pytest.raises(ValueError, match="max_iter must be at least 1"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="bipole",
            max_iter=0,
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


def test_bipole_nonconvergence_refusal_names_the_terminal_check(tmp_path):
    """GitLab #116: a refused SCF says what the terminal check measured.

    Before, the message was only "did not converge after N iterations",
    which points at the SCF and hides that the density returned to the
    caller was judged on the exact operator; the numbers that failed were
    printed nowhere. They now travel in the refusal, read off the terminal
    trace row that the post-loop refresh writes.
    """
    system, basis = _h2_3d(box=12.0)
    with pytest.raises(RuntimeError) as excinfo:
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="bipole",
            max_iter=1,
            output=tmp_path / "refused",
            output_qvf=False,
            progress=False,
        )
    message = str(excinfo.value)
    assert "did not converge after 1 iterations" in message
    assert "Terminal check on the density returned" in message
    assert "|dE| = " in message and "conv_tol_energy" in message
    assert "||[F,DS]|| = " in message and "conv_tol_grad" in message
