"""Regression tests for BUG 80 — DFT integration grid defaults.

vibe-qc's legacy default grid (ProductGaussLegendre 17×36, no pruning,
Becke partition) produces systematic energy differences of 0.1–2.8 mHa
relative to ORCA's DefGrid3.  The ``grid_level`` parameter selects the
grid resolution; the default (``"orca-defgrid3"``) matches ORCA.

These tests verify:
1. The four grid levels produce different energies.
2. ``grid_level="orca-defgrid3"`` is the default for DFT.
3. The coarser grid produces a larger ORCA mismatch than the fine grid.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from vibeqc import (
    Atom,
    BasisSet,
    GridOptions,
    Molecule,
    RKSOptions,
    ROKSOptions,
    UKSOptions,
    run_job,
    run_rks,
)
from vibeqc.runner import (
    _GRID_LEVEL_PRESETS,
    _GRID_OPTION_FIELDS,
    _apply_grid_level,
    grid_is_untouched,
    ks_options_need_grid_default,
)


# ---- shared fixtures -------------------------------------------------------

_H2O = Molecule(
    [
        Atom(8, [0.000000, 0.000000, 0.117790]),
        Atom(1, [0.000000, 0.755453, -0.471161]),
        Atom(1, [0.000000, -0.755453, -0.471161]),
    ]
)


def _pbe_energy(mol: object, basis: str, grid_level: str) -> float:
    return float(run_rks(mol, basis, grid_level=grid_level).energy)


def _assert_grid_matches_preset(grid: GridOptions, level: str) -> None:
    expected_grid = GridOptions()
    _apply_grid_level(expected_grid, level)
    for field in _GRID_LEVEL_PRESETS[level]:
        assert getattr(grid, field) == getattr(expected_grid, field)


# ---- grid level presets ----------------------------------------------------


def test_grid_level_presets_are_known():
    """Every supported generic or functional-specific grid preset is known."""
    assert set(_GRID_LEVEL_PRESETS.keys()) == {
        "skala",
        "orca-defgrid3",
        "fine",
        "coarse",
        "legacy",
    }


@pytest.mark.parametrize(
    "level", ["skala", "orca-defgrid3", "fine", "coarse", "legacy"]
)
def test_apply_grid_level_sets_expected_fields(level):
    """Applying each grid level must produce a GridOptions that differs from
    the unmodified C++ default — except "legacy" which intentionally matches
    the old C++ defaults for backward compatibility."""
    g = GridOptions()
    original_n_radial = g.n_radial
    original_angular = g.angular
    original_partition = g.partition
    original_profile = g.atomic_grid_profile
    _apply_grid_level(g, level)
    if level == "legacy":
        # Legacy must preserve the original C++ defaults.
        assert g.n_radial == original_n_radial
        assert g.angular == original_angular
        assert g.partition == original_partition
    else:
        # Every other preset changes at least n_radial or angular.
        assert (
            g.n_radial != original_n_radial
            or g.angular != original_angular
            or g.atomic_grid_profile != original_profile
        ), f"{level}: grid was not modified"


def test_unknown_grid_level_raises():
    g = GridOptions()
    with pytest.raises(ValueError, match="Unknown grid_level"):
        _apply_grid_level(g, "bogus")


# ---- energy changes across grid levels -------------------------------------


def test_grid_levels_produce_different_energies_pbe_sto3g():
    """PBE/STO-3G on water: the four grid levels must produce four distinct
    total energies, demonstrating the grid is the source of the BUG 80 shift."""
    energies = {}
    for level in ["orca-defgrid3", "fine", "coarse", "legacy"]:
        energies[level] = _pbe_energy(_H2O, "sto-3g", level)

    # All four must be distinct at sub-µHa tolerance.
    values = list(energies.values())
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            assert abs(values[i] - values[j]) > 1e-9, (
                f"Grids {list(energies.keys())[i]} and "
                f"{list(energies.keys())[j]} produce identical energy "
                f"{values[i]:.12f}"
            )


def test_coarser_grid_produces_larger_orca_mismatch_pbe_sto3g():
    """Coarser grid must produce a larger displacement from the orca-defgrid3
    energy than the fine grid does."""
    e_orca = _pbe_energy(_H2O, "sto-3g", "orca-defgrid3")
    e_fine = _pbe_energy(_H2O, "sto-3g", "fine")
    e_coarse = _pbe_energy(_H2O, "sto-3g", "coarse")

    delta_fine = abs(e_fine - e_orca)
    delta_coarse = abs(e_coarse - e_orca)
    assert delta_coarse > delta_fine, (
        f"Expected coarse grid ({delta_coarse:.6e} mHa) to deviate more from "
        f"orca-defgrid3 than fine grid ({delta_fine:.6e} mHa)"
    )


def test_legacy_grid_differs_from_orca_default():
    """The legacy grid must produce a measurable energy difference from the
    new default, confirming BUG 80's systematic shift is real."""
    e_orca = _pbe_energy(_H2O, "sto-3g", "orca-defgrid3")
    e_legacy = _pbe_energy(_H2O, "sto-3g", "legacy")
    delta_mHa = abs(e_orca - e_legacy) * 1000
    assert delta_mHa > 0.0001, (
        f"Legacy vs orca-defgrid3 delta = {delta_mHa:.6f} mHa; "
        f"expected > 0.0001 mHa"
    )


# ---- default behaviour -----------------------------------------------------


def test_default_grid_is_orca_defgrid3():
    """When grid_level is not passed to run_rks, the grid must be the
    orca-defgrid3 preset (the new BUG 80 fix default), not the legacy grid."""
    # Explicit orca-defgrid3
    e_explicit = _pbe_energy(_H2O, "sto-3g", "orca-defgrid3")
    # Default (no grid_level kwarg)
    e_default = float(run_rks(_H2O, "sto-3g").energy)
    assert e_default == pytest.approx(e_explicit, abs=1e-12), (
        f"Default grid energy {e_default:.12f} does not match "
        f"orca-defgrid3 {e_explicit:.12f}; default grid may still be legacy."
    )


def test_default_grid_differs_from_legacy():
    """The default grid must NOT equal the legacy grid."""
    e_default = float(run_rks(_H2O, "sto-3g").energy)
    e_legacy = _pbe_energy(_H2O, "sto-3g", "legacy")
    assert abs(e_default - e_legacy) > 1e-9, (
        f"Default grid energy {e_default:.12f} matches legacy "
        f"{e_legacy:.12f}; default grid was not changed."
    )


@pytest.mark.parametrize("options_attr", ["_rks_options", "_uks_options"])
def test_ase_default_ks_options_apply_selected_grid(options_attr):
    """ASE-created KS options must not bypass the package grid default."""
    from vibeqc.ase import VibeQC

    calculator = VibeQC(functional="pbe", grid_level="fine")
    _assert_grid_matches_preset(getattr(calculator, options_attr).grid, "fine")


def test_ase_rks_hessian_uses_the_scf_grid(monkeypatch):
    """ASE response properties must stay on the selected RKS surface."""
    import vibeqc.ase as ase_module

    captured = {}

    def fake_hessian(*args, **kwargs):
        captured["grid"] = kwargs["grid_options"]
        return SimpleNamespace(hessian=[[0.0]])

    monkeypatch.setattr(
        ase_module, "compute_hessian_rks_analytic", fake_hessian
    )
    calculator = ase_module.VibeQC(functional="pbe", grid_level="fine")
    calculator._compute_hessian("RKS", _H2O, object(), object())

    assert captured["grid"] is calculator._rks_options.grid


@pytest.mark.parametrize(
    ("method", "options_attr"),
    [("rks", "_rks_options"), ("uks", "_uks_options")],
)
def test_geomopt_default_ks_options_share_scf_and_gradient_grid(
    method, options_attr
):
    """The provider's implicit SCF and analytic gradient use one KS grid."""
    from vibeqc.geomopt import MolecularSCFProvider

    provider = MolecularSCFProvider(
        "sto-3g", method=method, functional="pbe", grid_level="fine"
    )
    options = provider._mean_field_options()

    assert options is getattr(provider, options_attr)
    _assert_grid_matches_preset(options.grid, "fine")
    _assert_grid_matches_preset(provider._grid_options, "fine")


@pytest.mark.parametrize(
    ("method", "options_cls", "options_kw", "options_attr"),
    [
        ("rks", RKSOptions, "rks_options", "_rks_options"),
        ("uks", UKSOptions, "uks_options", "_uks_options"),
    ],
)
def test_explicit_ks_grid_wins_across_ase_and_geomopt(
    method, options_cls, options_kw, options_attr
):
    """A preset never overwrites a caller-provided KS options grid."""
    from vibeqc.ase import VibeQC
    from vibeqc.geomopt import MolecularSCFProvider

    custom = options_cls()
    custom.grid.n_radial = 53
    custom.grid.angular = "product"
    custom.grid.n_theta = 11
    custom.grid.n_phi = 22

    calculator = VibeQC(
        functional="pbe", grid_level="fine", **{options_kw: custom}
    )
    assert getattr(calculator, options_attr).grid.n_radial == 53

    provider = MolecularSCFProvider(
        "sto-3g",
        method=method,
        functional="pbe",
        grid_level="fine",
        **{options_kw: custom},
    )
    selected = provider._mean_field_options()
    assert selected.grid.n_radial == 53
    assert provider._grid_options.n_radial == 53


def test_run_job_internal_options_keep_selected_grid(tmp_path):
    """A high-level guess must not turn the default KS grid back to legacy."""
    common = {
        "basis": "sto-3g",
        "method": "rks",
        "functional": "pbe",
        "name_molecule": False,
        "output_qvf": False,
        "write_molden_file": False,
        "write_xyz_file": False,
        "write_population_file": False,
        "citations": False,
        "progress": False,
        "crash_dump": False,
    }

    default = run_job(_H2O, output=tmp_path / "default", **common)
    materialized = run_job(
        _H2O,
        output=tmp_path / "materialized",
        initial_guess="patom",
        **common,
    )

    assert float(materialized.energy) == pytest.approx(
        float(default.energy), abs=1e-12
    )


@pytest.mark.parametrize(
    ("extra", "level"),
    [
        ({"density_fit": True}, "fine"),
        ({"optimize": True}, "coarse"),
    ],
)
def test_run_job_feature_options_keep_selected_grid(
    tmp_path, monkeypatch, extra, level
):
    """Feature-created KS options inherit grid_level before preflight."""
    import vibeqc.runner as runner_module

    captured = {}

    def fake_estimate(*args, **kwargs):
        captured["options"] = kwargs["options"]
        return SimpleNamespace(total_bytes=1)

    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")
    monkeypatch.setattr(runner_module, "estimate_memory", fake_estimate)
    runner_module.run_job(
        _H2O,
        basis="sto-3g",
        method="rks",
        functional="pbe",
        grid_level=level,
        dry_run=True,
        output=tmp_path / f"feature_{level}",
        **extra,
    )

    scf_options = captured["options"]["scf_options"]
    _assert_grid_matches_preset(scf_options.grid, level)
    if extra.get("density_fit"):
        assert scf_options.density_fit is True


@pytest.mark.parametrize(
    ("method", "options_cls", "options_kw"),
    [
        ("rks", RKSOptions, "rks_options"),
        ("uks", UKSOptions, "uks_options"),
    ],
)
def test_native_optimizer_gradient_inherits_scf_grid(
    method, options_cls, options_kw
):
    """Native and Brent optimizers differentiate their selected SCF grid."""
    from vibeqc.molecular_optimize import _resolve_molecular_ks_gradient_grid

    options = options_cls()
    _apply_grid_level(options.grid, "fine")
    kwargs = {"rks_options": None, "uks_options": None}
    kwargs[options_kw] = options

    selected = _resolve_molecular_ks_gradient_grid(
        method, grid_options=None, **kwargs
    )
    assert selected is options.grid

    explicit = GridOptions()
    overridden = _resolve_molecular_ks_gradient_grid(
        method, grid_options=explicit, **kwargs
    )
    assert overridden is explicit


# ---- grid_level with explicit options -------------------------------------


def test_explicit_options_preserve_custom_grid():
    """When the caller passes explicit RKSOptions with a custom grid,
    their settings win over grid_level."""
    from vibeqc import RKSOptions

    opts = RKSOptions()
    # Set a custom grid that differs from all presets.
    opts.grid.n_radial = 53
    opts.grid.angular = "product"
    opts.grid.n_theta = 11
    opts.grid.n_phi = 22
    result = run_rks(_H2O, "sto-3g", options=opts)
    e_custom = float(result.energy)

    # This energy must differ from all preset energies.
    for level in ["orca-defgrid3", "fine", "coarse", "legacy"]:
        e_preset = _pbe_energy(_H2O, "sto-3g", level)
        assert abs(e_custom - e_preset) > 1e-9, (
            f"Custom grid produced energy matching {level} preset"
        )


# ---- basis-set independence ------------------------------------------------


def test_grid_delta_persists_across_basis_sizes():
    """The legacy-vs-orca energy gap must be measurable at the minimal-basis
    level.  At larger basis sets the grid sensitivity drops, which is itself
    the BUG 80 observation: the shift is systematic but shrinks with basis
    quality (0.01 mHa at def2-TZVP per the report).  The test only requires
    the gap to be non-zero at STO-3G where the effect is largest; at def2-SVP
    PBE/H2O the gap can be sub-µHa (grid-converged at that basis)."""
    # Minimal basis — large gap expected.
    e_orca_sto = _pbe_energy(_H2O, "sto-3g", "orca-defgrid3")
    e_legacy_sto = _pbe_energy(_H2O, "sto-3g", "legacy")
    delta_sto = abs(e_orca_sto - e_legacy_sto) * 1000
    assert delta_sto > 0.0001, (
        f"STO-3G: legacy vs orca delta = {delta_sto:.6f} mHa"
    )

    # Larger basis — gap shrinks (expected, per BUG 80).
    e_orca_svp = _pbe_energy(_H2O, "def2-svp", "orca-defgrid3")
    e_legacy_svp = _pbe_energy(_H2O, "def2-svp", "legacy")
    delta_svp = abs(e_orca_svp - e_legacy_svp) * 1000
    # Must be smaller than the STO-3G gap.
    assert delta_svp < delta_sto, (
        f"def2-SVP delta {delta_svp:.6f} mHa should be smaller than "
        f"STO-3G delta {delta_sto:.6f} mHa"
    )


# ---- GitLab #663: a passed-but-untouched grid still receives grid_level ----

_H2O_CATION = Molecule(
    [
        Atom(8, [0.000000, 0.000000, 0.117790]),
        Atom(1, [0.000000, 0.755453, -0.471161]),
        Atom(1, [0.000000, -0.755453, -0.471161]),
    ],
    charge=1,
    multiplicity=2,
)

_RUN_JOB_QUIET = {
    "basis": "sto-3g",
    "method": "rks",
    "functional": "pbe",
    "name_molecule": False,
    "output_qvf": False,
    "write_molden_file": False,
    "write_xyz_file": False,
    "write_population_file": False,
    "citations": False,
    "progress": False,
    "crash_dump": False,
}


def test_grid_is_untouched_tracks_every_public_field():
    """The helper answers "did the caller choose a grid" from the grid's
    fields, never from the presence of an options object."""
    assert grid_is_untouched(None)
    assert grid_is_untouched(GridOptions())
    for name, value in (
        ("n_radial", 53),
        ("n_theta", 11),
        ("n_phi", 22),
        ("lebedev_order", 35),
        ("becke_k", 4),
        ("vv10_grid_factor", 2.0),
        ("angular", "lebedev"),
        ("orca_angular_points", [302]),
        ("atomic_grid_profile", "pyscf-level3"),
        ("angular_pruning", "nwchem"),
        ("partition", "stratmann"),
    ):
        grid = GridOptions()
        setattr(grid, name, value)
        assert not grid_is_untouched(grid), name
    # Every bound public field is in the comparison, so a field added to the
    # C++ struct fails here instead of being treated as never customised.
    assert set(_GRID_OPTION_FIELDS) == {
        name for name in dir(GridOptions()) if not name.startswith("_")
    }
    for level in _GRID_LEVEL_PRESETS:
        grid = GridOptions()
        _apply_grid_level(grid, level)
        # The legacy preset IS the construction default, which is why the
        # documented way to request it is grid_level="legacy".
        assert grid_is_untouched(grid) is (level == "legacy"), level
    assert ks_options_need_grid_default(None)
    assert ks_options_need_grid_default(RKSOptions())
    assert ks_options_need_grid_default(UKSOptions())
    assert ks_options_need_grid_default(ROKSOptions())  # grid defaults to None
    chosen = RKSOptions()
    chosen.grid.n_radial = 53
    assert not ks_options_need_grid_default(chosen)


@pytest.mark.parametrize(
    ("method", "molecule", "options_kw", "options_cls"),
    [
        ("rks", _H2O, "rks_options", RKSOptions),
        ("uks", _H2O_CATION, "uks_options", UKSOptions),
        ("roks", _H2O_CATION, "roks_options", ROKSOptions),
    ],
)
def test_run_job_empty_options_receive_grid_level(
    tmp_path, monkeypatch, method, molecule, options_kw, options_cls
):
    """#663: ``RKSOptions()`` passed to configure nothing must not opt the run
    out of ``grid_level``.  Pre-fix the captured grid was the untouched legacy
    default for all three methods (dry-run capture, no SCF)."""
    import vibeqc.runner as runner_module

    captured = {}

    def fake_estimate(*args, **kwargs):
        captured["options"] = kwargs["options"]
        return SimpleNamespace(total_bytes=1)

    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")
    monkeypatch.setattr(runner_module, "estimate_memory", fake_estimate)
    empty = options_cls()
    runner_module.run_job(
        molecule,
        basis="sto-3g",
        method=method,
        functional="pbe",
        grid_level="fine",
        dry_run=True,
        output=tmp_path / f"empty_{method}",
        **{options_kw: empty},
    )
    scf_options = captured["options"]["scf_options"]
    # The caller's own object ran, and its grid records the preset it used.
    assert scf_options is empty
    _assert_grid_matches_preset(empty.grid, "fine")


def test_run_job_empty_options_match_no_options_energy(tmp_path):
    """#663, the issue's repro: no options object and an empty one give the
    same grid and energy; both differ from the legacy grid."""
    default = run_job(_H2O, output=tmp_path / "default", **_RUN_JOB_QUIET)
    empty = run_job(
        _H2O, output=tmp_path / "empty", rks_options=RKSOptions(), **_RUN_JOB_QUIET
    )
    legacy = run_job(
        _H2O, output=tmp_path / "legacy", grid_level="legacy", **_RUN_JOB_QUIET
    )
    assert float(empty.energy) == pytest.approx(float(default.energy), abs=1e-12)
    assert abs(float(empty.energy) - float(legacy.energy)) > 1e-9


@pytest.mark.parametrize(
    ("method", "options_cls", "options_kw", "options_attr"),
    [
        ("rks", RKSOptions, "rks_options", "_rks_options"),
        ("uks", UKSOptions, "uks_options", "_uks_options"),
    ],
)
def test_empty_ks_options_receive_grid_level_across_ase_and_geomopt(
    method, options_cls, options_kw, options_attr
):
    """#663 on the ASE calculator and the geometry-optimization provider: an
    untouched grid on a supplied options object receives the preset, and the
    provider's gradient grid follows it."""
    from vibeqc.ase import VibeQC
    from vibeqc.geomopt import MolecularSCFProvider

    calculator = VibeQC(
        functional="pbe", grid_level="fine", **{options_kw: options_cls()}
    )
    _assert_grid_matches_preset(getattr(calculator, options_attr).grid, "fine")

    provider = MolecularSCFProvider(
        "sto-3g",
        method=method,
        functional="pbe",
        grid_level="fine",
        **{options_kw: options_cls()},
    )
    selected = provider._mean_field_options()
    _assert_grid_matches_preset(selected.grid, "fine")
    _assert_grid_matches_preset(provider._grid_options, "fine")


def test_low_level_run_rks_uses_a_supplied_options_grid_as_given():
    """The layering #663 leaves in place: the low-level wrappers run the
    options they are handed (an empty ``RKSOptions()`` is the construction
    default, i.e. the legacy grid) and leave the grid untouched; the entry
    points with a ``grid_level`` are the ones that apply it."""
    from vibeqc import run_uks

    opts = RKSOptions()
    as_given = float(run_rks(_H2O, "sto-3g", options=opts).energy)
    assert grid_is_untouched(opts.grid)
    legacy = _pbe_energy(_H2O, "sto-3g", "legacy")
    default = _pbe_energy(_H2O, "sto-3g", "orca-defgrid3")
    assert as_given == pytest.approx(legacy, abs=1e-12)
    assert abs(as_given - default) > 1e-9
    uks_opts = UKSOptions()
    uks_opts.functional = "pbe"
    run_uks(_H2O_CATION, "sto-3g", options=uks_opts)
    assert grid_is_untouched(uks_opts.grid)


@pytest.mark.parametrize(
    ("method", "molecule", "options_kw", "options_cls"),
    [
        ("rks", _H2O, "rks_options", RKSOptions),
        ("uks", _H2O_CATION, "uks_options", UKSOptions),
    ],
)
def test_run_single_point_applies_grid_level_to_an_untouched_grid(
    method, molecule, options_kw, options_cls
):
    """The dispatcher's own gate, pinned without run_job in front of it: it is
    the live gate for the finite-difference optimizer and the scan driver."""
    import vibeqc.runner as runner_module

    empty = options_cls()
    out: dict = {}
    runner_module._run_single_point(
        method,
        molecule,
        BasisSet(molecule, "sto-3g"),
        functional="pbe",
        grid_level="fine",
        used_scf_options_out=out,
        **{options_kw: empty},
    )
    assert out["scf_options"] is empty
    _assert_grid_matches_preset(empty.grid, "fine")


def test_run_job_legacy_grid_survives_the_geomopt_provider(tmp_path, monkeypatch):
    """#663 regression guard: ``grid_level="legacy"`` leaves the grid equal to
    the construction defaults, so the provider run_job builds must receive the
    level too, or it re-defaults the untouched grid to orca-defgrid3."""
    import vibeqc.geomopt as geomopt_module

    seen: dict = {}

    class _Stop(Exception):
        pass

    def fake_run_geomopt(molecule, provider, **kwargs):
        seen["provider"] = provider
        raise _Stop()

    monkeypatch.setattr(geomopt_module, "run_geomopt", fake_run_geomopt)
    with pytest.raises(_Stop):
        run_job(
            _H2O,
            output=tmp_path / "legacy_geomopt",
            grid_level="legacy",
            optimize=True,
            geom_opt="bfgs",
            **_RUN_JOB_QUIET,
        )
    provider = seen["provider"]
    assert provider._grid_level == "legacy"
    assert grid_is_untouched(provider._mean_field_options().grid)


def test_run_job_legacy_grid_survives_the_ase_optimizer_backend(
    tmp_path, monkeypatch
):
    """Same guard for the ASE backend: the calculator run_job builds must be
    told the level, or its own orca-defgrid3 default overwrites the untouched
    legacy grid."""
    import vibeqc.ase as ase_module

    seen: dict = {}

    class _Stop(Exception):
        pass

    class _FakeVibeQC:
        def __init__(self, **kwargs):
            seen.update(kwargs)
            raise _Stop()

    monkeypatch.setattr(ase_module, "VibeQC", _FakeVibeQC)
    with pytest.raises(_Stop):
        run_job(
            _H2O,
            output=tmp_path / "legacy_ase",
            grid_level="legacy",
            optimize=True,
            optimizer_backend="ase",
            max_opt_steps=1,
            **_RUN_JOB_QUIET,
        )
    assert seen.get("grid_level") == "legacy"
    assert grid_is_untouched(seen["rks_options"].grid)


def test_reused_options_object_keeps_the_first_preset(tmp_path, monkeypatch):
    """Documented in ``run_job``: the preset is recorded on the caller's
    object, so a reused object counts as customised on the next call and a
    different ``grid_level`` there is ignored."""
    import vibeqc.runner as runner_module

    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")
    monkeypatch.setattr(
        runner_module,
        "estimate_memory",
        lambda *args, **kwargs: SimpleNamespace(total_bytes=1),
    )
    def _dry_run(options, level, tag):
        runner_module.run_job(
            _H2O,
            basis="sto-3g",
            method="rks",
            functional="pbe",
            grid_level=level,
            dry_run=True,
            output=tmp_path / f"reuse_{tag}",
            rks_options=options,
        )

    opts = RKSOptions()
    for i, level in enumerate(("fine", "coarse")):
        _dry_run(opts, level, f"a{i}")
        _assert_grid_matches_preset(opts.grid, "fine")
    # The one exception, documented too: a recorded "legacy" preset equals an
    # untouched grid, so a later level still applies.
    legacy_first = RKSOptions()
    _dry_run(legacy_first, "legacy", "b0")
    assert grid_is_untouched(legacy_first.grid)
    _dry_run(legacy_first, "fine", "b1")
    _assert_grid_matches_preset(legacy_first.grid, "fine")


def test_copy_grid_options_copies_every_public_field():
    """The TRAH retry clone dropped ``atomic_grid_profile`` and
    ``vv10_grid_factor``; it now iterates the canonical field list."""
    from vibeqc.runner import _copy_grid_options

    source = GridOptions()
    for name, value in (
        ("n_radial", 53),
        ("n_theta", 11),
        ("n_phi", 22),
        ("lebedev_order", 35),
        ("becke_k", 4),
        ("vv10_grid_factor", 2.0),
        ("angular", "lebedev"),
        ("orca_angular_points", [302]),
        ("atomic_grid_profile", "pyscf-level3"),
        ("angular_pruning", "nwchem"),
        ("partition", "stratmann"),
    ):
        setattr(source, name, value)
    copy = _copy_grid_options(source)
    assert copy is not source
    for name in _GRID_OPTION_FIELDS:
        assert getattr(copy, name) == getattr(source, name), name
