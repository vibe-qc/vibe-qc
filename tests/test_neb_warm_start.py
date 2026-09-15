"""Density warm-start tests for ``vibeqc.run_neb``.

``warm_start=True`` (default) makes the SCF at outer iteration N+1
start from outer iter N's converged density for the same image —
the canonical within-image warm-start across NEB iterations from
the periodic NEB warm-start milestone.

These tests pin:

* Identical results vs. cold-start. The energy / gradient / final
  TS index must match the cold-start path to numerical precision
  — warm-start changes only the initial guess, not the converged
  state.
* Default is on. ``run_neb(...)`` with no kwarg threads densities
  through. Pass ``warm_start=False`` to force cold-start.
* RHF / UHF / RKS / UKS use the same within-image density cache. UKS keeps
  stability analysis check-only so a generic corrective follow cannot switch
  one image independently to another electronic basin.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import Atom, Molecule, run_neb


def _h2(z2: float) -> Molecule:
    """Closed-shell H₂ along z; ``z2`` is the second-atom z-coord."""
    return Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, float(z2)])])


@pytest.fixture
def h2_endpoints():
    return _h2(1.4), _h2(2.0)


def _run(reactant, product, *, warm_start: bool):
    return run_neb(
        reactant,
        product,
        basis="sto-3g",
        n_images=3,
        method="RHF",
        interpolation="linear",
        max_iter=20,
        conv_tol_force=5e-3,
        n_jobs=1,
        initial_step=0.05,
        warm_start=warm_start,
    )


@pytest.mark.parametrize(
    ("helper_name", "options_cls", "core_driver_name"),
    [
        ("_run_rks_warm_start", vq.RKSOptions, "run_rks_scf_with_jk"),
        ("_run_uks_warm_start", vq.UKSOptions, "run_uks_scf_with_jk"),
    ],
)
def test_explicit_functional_overrides_native_lda_default(
    monkeypatch,
    helper_name,
    options_cls,
    core_driver_name,
):
    """The advertised top-level functional must reach molecular NEB SCF."""
    import vibeqc._vibeqc_core as core
    import vibeqc.neb as neb_module

    molecule = _h2(1.4)
    options = options_cls()
    assert options.functional.lower() == "lda"
    seen = []

    monkeypatch.setattr(
        neb_module,
        "_build_scf_common_pieces",
        lambda mol, basis: (np.eye(2), np.eye(2), 0.0, object()),
    )
    monkeypatch.setattr(
        neb_module,
        "_resolve_xc_grid",
        lambda mol, grid_options: object(),
    )
    monkeypatch.setattr(
        neb_module,
        "_molecular_cold_start_closed",
        lambda mol, basis, options, S, Hcore, jk: np.eye(2),
    )
    monkeypatch.setattr(
        neb_module,
        "_molecular_cold_start_open",
        lambda mol, basis, options, n_alpha, n_beta, S, Hcore, jk, **kwargs: (
            np.eye(2),
            np.eye(2),
        ),
    )

    def capture_driver(*args, **kwargs):
        opts = next(arg for arg in args if isinstance(arg, options_cls))
        seen.append(opts.functional)
        return object()

    monkeypatch.setattr(core, core_driver_name, capture_driver)
    helper = getattr(neb_module, helper_name)
    helper(
        molecule,
        object(),
        options,
        "pbe",
        None,
        None,
    )

    assert seen == ["pbe"]


def test_uks_warm_start_analyzes_without_following_or_mutating_options(
    monkeypatch,
):
    """NEB images retain a verdict without independent basin switching."""
    import vibeqc._vibeqc_core as core
    import vibeqc.neb as neb_module

    molecule = _h2(1.4)
    user_options = vq.UKSOptions()
    user_options.functional = "LDA"
    user_options.stability_max_retries = 7
    assert user_options.stability_check
    assert not user_options._stability_check_explicit

    monkeypatch.setattr(
        neb_module,
        "_build_scf_common_pieces",
        lambda mol, basis: (np.eye(2), np.eye(2), 0.0, object()),
    )
    monkeypatch.setattr(
        neb_module,
        "_resolve_xc_grid",
        lambda mol, grid_options: object(),
    )
    monkeypatch.setattr(
        neb_module,
        "_molecular_cold_start_open",
        lambda *args, **kwargs: (np.eye(2), np.eye(2)),
    )
    seen = []

    def capture_driver(*args, **kwargs):
        captured = kwargs["options"] if "options" in kwargs else args[8]
        seen.append(captured)
        return object()

    monkeypatch.setattr(core, "run_uks_scf_with_jk", capture_driver)
    neb_module._run_uks_warm_start(
        molecule,
        object(),
        user_options,
        "PBE",
        None,
        None,
    )

    assert len(seen) == 1
    internal = seen[0]
    assert internal is not user_options
    assert internal.functional == "PBE"
    assert internal.stability_check
    assert not internal._stability_check_explicit
    assert internal.stability_max_retries == 0
    assert user_options.functional == "LDA"
    assert user_options.stability_max_retries == 7
    assert user_options.stability_check
    assert not user_options._stability_check_explicit


def test_uks_neb_image_fails_loud_when_check_only_verdict_is_unstable(
    monkeypatch,
):
    """An unstable image cannot disappear into an ordinary NEB gradient."""
    import vibeqc.neb as neb_module

    molecule = _h2(1.4)
    positions = np.asarray(
        [atom.xyz for atom in molecule.atoms], dtype=float
    )
    unstable = SimpleNamespace(
        converged=True,
        energy=-1.0,
        density_alpha=np.eye(2),
        density_beta=np.eye(2),
        stability_checked=True,
        stability_analysis_converged=True,
        stability_eigenvalue=-0.125,
        internal_instability=True,
    )
    monkeypatch.setattr(
        neb_module,
        "_run_uks_warm_start",
        lambda *args, **kwargs: unstable,
    )

    with pytest.raises(
        neb_module.NEBImageSCFError,
        match=r"image 2.*internally unstable.*continuity",
    ):
        neb_module._evaluate_image(
            positions,
            molecule,
            "sto-3g",
            "UKS",
            functional="PBE",
            rhf_options=None,
            uhf_options=None,
            rks_options=None,
            uks_options=vq.UKSOptions(),
            gradient_options=None,
            grid_options=None,
            dispersion_params=None,
            image_index=2,
        )


def test_uks_neb_image_fails_loud_when_stability_is_unverified(monkeypatch):
    """A failed stability eigensolve must not be discarded by NEB."""
    import vibeqc.neb as neb_module

    molecule = _h2(1.4)
    positions = np.asarray(
        [atom.xyz for atom in molecule.atoms], dtype=float
    )
    unverified = SimpleNamespace(
        converged=True,
        energy=-1.0,
        density_alpha=np.eye(2),
        density_beta=np.eye(2),
        stability_checked=True,
        stability_analysis_converged=False,
        stability_eigenvalue=0.0,
        internal_instability=False,
    )
    monkeypatch.setattr(
        neb_module,
        "_run_uks_warm_start",
        lambda *args, **kwargs: unverified,
    )

    with pytest.raises(
        neb_module.NEBImageSCFError,
        match=r"image 3.*eigensolver did not converge.*UNVERIFIED",
    ):
        neb_module._evaluate_image(
            positions,
            molecule,
            "sto-3g",
            "UKS",
            functional="PBE",
            rhf_options=None,
            uhf_options=None,
            rks_options=None,
            uks_options=vq.UKSOptions(),
            gradient_options=None,
            grid_options=None,
            dispersion_params=None,
            image_index=3,
        )

@pytest.mark.parametrize(
    ("method", "options_cls", "helper_name"),
    [
        ("RKS", vq.RKSOptions, "_run_rks_warm_start"),
        ("UKS", vq.UKSOptions, "_run_uks_warm_start"),
    ],
)
@pytest.mark.parametrize("top_level_override", [False, True])
def test_ks_grid_options_shared_by_warm_start_scf_and_gradient(
    monkeypatch,
    method,
    options_cls,
    helper_name,
    top_level_override,
):
    """Options-grid fallback and explicit override select one KS surface."""
    import vibeqc.molecular_optimize as optimize_module
    import vibeqc.neb as neb_module

    options = options_cls()
    options.grid.n_radial = 17
    top_level_grid = None
    if top_level_override:
        top_level_grid = vq.GridOptions()
        top_level_grid.n_radial = 31
    expected_grid = top_level_grid if top_level_grid is not None else options.grid
    seen = {}

    def capture_scf(
        molecule,
        basis,
        selected_options,
        functional,
        selected_grid,
        initial_density,
    ):
        seen["scf_grid"] = selected_grid
        common = {"converged": True, "energy": -1.0}
        if method == "RKS":
            return SimpleNamespace(density=np.eye(2), **common)
        return SimpleNamespace(
            density_alpha=np.eye(2),
            density_beta=np.eye(2),
            **common,
        )

    def capture_gradient(
        molecule,
        basis,
        scf_result,
        selected_method,
        *,
        gradient_options,
        grid_options,
        dispersion_params,
        dft_plus_u,
    ):
        seen["gradient_grid"] = grid_options
        return np.zeros((len(molecule.atoms), 3))

    monkeypatch.setattr(neb_module, helper_name, capture_scf)
    monkeypatch.setattr(
        optimize_module,
        "_compute_molecular_gradient",
        capture_gradient,
    )

    options_kwargs = {
        "rks_options": options if method == "RKS" else None,
        "uks_options": options if method == "UKS" else None,
    }
    molecule = _h2(1.4)
    positions = np.asarray([atom.xyz for atom in molecule.atoms], dtype=float)
    neb_module._evaluate_image(
        positions,
        molecule,
        "sto-3g",
        method,
        functional="pbe",
        rhf_options=None,
        uhf_options=None,
        gradient_options=None,
        grid_options=top_level_grid,
        dispersion_params=None,
        **options_kwargs,
    )

    assert seen["scf_grid"] is expected_grid
    assert seen["gradient_grid"] is expected_grid


def test_molecular_cold_start_honors_read_density():
    """The low-level warm loop must not replace READ with SAD/PATOM."""
    import vibeqc.neb as neb_module

    molecule = _h2(1.4)
    basis = vq.BasisSet(molecule, "sto-3g")
    overlap, hcore, _e_nuc, jk = neb_module._build_scf_common_pieces(
        molecule, basis
    )
    requested = np.array([[1.25, -0.1], [-0.1, 0.75]])
    options = vq.RHFOptions()
    options.initial_guess = vq.InitialGuess.READ
    options.read_density = requested

    actual = neb_module._molecular_cold_start_closed(
        molecule,
        basis,
        options,
        overlap,
        hcore,
        jk,
    )

    np.testing.assert_allclose(actual, requested)


@pytest.mark.parametrize(
    ("method", "options_cls", "open_shell"),
    [
        ("RHF", vq.RHFOptions, False),
        ("UHF", vq.UHFOptions, True),
    ],
)
def test_molecular_neb_resolves_read_path_before_dry_run(
    tmp_path,
    monkeypatch,
    method,
    options_cls,
    open_shell,
):
    """Dry-run validates the same path-backed READ state as live NEB."""
    from vibeqc import guess_read

    if open_shell:
        reactant = Molecule(
            [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
            1,
            2,
        )
        product = Molecule(
            [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.8])],
            1,
            2,
        )
    else:
        reactant, product = _h2(1.4), _h2(1.8)

    options = options_cls()
    options.initial_guess = vq.InitialGuess.READ
    options.read_path = str(tmp_path / "prior.qvf")
    expected = np.eye(2)
    if open_shell:
        options.read_density_alpha = 0.10 * expected
        options.read_density_beta = 0.05 * expected
        monkeypatch.setattr(
            guess_read,
            "resolve_read_densities_open",
            lambda *args, **kwargs: (0.75 * expected, 0.25 * expected),
        )
        option_kwargs = {"uhf_options": options}
    else:
        options.read_density = 0.20 * expected
        monkeypatch.setattr(
            guess_read,
            "resolve_read_density_closed",
            lambda *args, **kwargs: expected,
        )
        option_kwargs = {"rhf_options": options}

    stem = tmp_path / f"{method.lower()}-read-dry-run"
    result = run_neb(
        reactant,
        product,
        basis="sto-3g",
        n_images=1,
        method=method,
        dry_run=True,
        output=stem,
        **option_kwargs,
    )

    assert result is None
    assert stem.with_suffix(".system").exists()
    if open_shell:
        np.testing.assert_allclose(options.read_density_alpha, 0.75 * expected)
        np.testing.assert_allclose(options.read_density_beta, 0.25 * expected)
    else:
        np.testing.assert_allclose(options.read_density, expected)


@pytest.mark.parametrize(
    ("method", "options_cls", "open_shell"),
    [
        ("RHF", vq.RHFOptions, False),
        ("UHF", vq.UHFOptions, True),
    ],
)
def test_molecular_neb_missing_read_source_fails_before_manifest(
    tmp_path,
    method,
    options_cls,
    open_shell,
):
    """An invalid READ source cannot be certified by NEB dry-run."""
    if open_shell:
        reactant = Molecule(
            [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
            1,
            2,
        )
        product = Molecule(
            [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.8])],
            1,
            2,
        )
    else:
        reactant, product = _h2(1.4), _h2(1.8)

    options = options_cls()
    options.initial_guess = vq.InitialGuess.READ
    options.read_path = str(tmp_path / "missing.qvf")
    option_kwargs = (
        {"uhf_options": options}
        if open_shell
        else {"rhf_options": options}
    )
    stem = tmp_path / f"{method.lower()}-missing-read"

    with pytest.raises(FileNotFoundError):
        run_neb(
            reactant,
            product,
            basis="sto-3g",
            n_images=1,
            method=method,
            dry_run=True,
            output=stem,
            **option_kwargs,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    ("method", "options_cls", "open_shell"),
    [
        ("RHF", vq.RHFOptions, False),
        ("UHF", vq.UHFOptions, True),
    ],
)
def test_molecular_neb_fragmo_requires_precomputed_density(
    tmp_path,
    method,
    options_cls,
    open_shell,
):
    """NEB has no fragments seam, so incomplete FRAGMO fails preflight."""
    if open_shell:
        reactant = Molecule(
            [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
            1,
            2,
        )
        product = Molecule(
            [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.8])],
            1,
            2,
        )
    else:
        reactant, product = _h2(1.4), _h2(1.8)

    options = options_cls()
    options.initial_guess = vq.InitialGuess.FRAGMO
    option_kwargs = (
        {"uhf_options": options}
        if open_shell
        else {"rhf_options": options}
    )
    stem = tmp_path / f"{method.lower()}-incomplete-fragmo"

    with pytest.raises(ValueError, match="FRAGMO requires"):
        run_neb(
            reactant,
            product,
            basis="sto-3g",
            n_images=1,
            method=method,
            dry_run=True,
            output=stem,
            **option_kwargs,
        )

    assert not stem.with_suffix(".system").exists()


def test_molecular_open_cold_start_honors_atomspin():
    """ATOMSPIN tags reach the GuessEngine rather than a fixed SAD split."""
    import vibeqc.neb as neb_module

    molecule = Molecule(
        [
            Atom(1, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, 1.4]),
            Atom(1, [0.0, 0.0, 4.4]),
        ],
        0,
        2,
    )
    basis = vq.BasisSet(molecule, "sto-3g")
    overlap, hcore, _e_nuc, jk = neb_module._build_scf_common_pieces(
        molecule, basis
    )
    options = vq.UHFOptions()
    options.initial_guess = vq.InitialGuess.SAD
    options.atomic_spins = [1, -1, 1]
    tagged_a, tagged_b = neb_module._molecular_cold_start_open(
        molecule,
        basis,
        options,
        2,
        1,
        overlap,
        hcore,
        jk,
        use_jk_for_density_mode=False,
    )
    options.atomic_spins = []
    plain_a, plain_b = neb_module._molecular_cold_start_open(
        molecule,
        basis,
        options,
        2,
        1,
        overlap,
        hcore,
        jk,
        use_jk_for_density_mode=False,
    )

    assert np.linalg.norm(tagged_a - plain_a) > 1.0e-6
    assert np.linalg.norm(tagged_b - plain_b) > 1.0e-6


def test_default_molecular_ks_neb_uses_defgrid3(monkeypatch):
    """Options=None retains the public molecular DFT grid default."""
    import vibeqc.molecular_optimize as optimize_module
    import vibeqc.neb as neb_module
    from vibeqc.runner import _apply_grid_level

    expected = vq.GridOptions()
    _apply_grid_level(expected, "orca-defgrid3")
    seen = {}

    def capture_scf(mol, basis, options, functional, grid, initial_density):
        seen["scf"] = grid
        return SimpleNamespace(
            converged=True,
            energy=-1.0,
            density=np.eye(2),
        )

    def capture_gradient(*args, grid_options, **kwargs):
        seen["gradient"] = grid_options
        return np.zeros((2, 3))

    monkeypatch.setattr(neb_module, "_run_rks_warm_start", capture_scf)
    monkeypatch.setattr(
        optimize_module,
        "_compute_molecular_gradient",
        capture_gradient,
    )
    molecule = _h2(1.4)
    positions = np.asarray([atom.xyz for atom in molecule.atoms], dtype=float)
    neb_module._evaluate_image(
        positions,
        molecule,
        "sto-3g",
        "RKS",
        functional="pbe",
        rhf_options=None,
        uhf_options=None,
        rks_options=None,
        uks_options=None,
        gradient_options=None,
        grid_options=None,
        dispersion_params=None,
    )

    for grid in (seen["scf"], seen["gradient"]):
        assert grid.n_radial == expected.n_radial
        assert grid.lebedev_order == expected.lebedev_order


class TestWarmStartEquivalence:
    def test_warm_start_matches_cold_start_bitexact(self, h2_endpoints):
        """The converged state of the SCF doesn't depend on the
        initial guess; warm-start must produce identical NEB
        trajectories to cold-start at the same outer-iteration
        budget."""
        r, p = h2_endpoints
        cold = _run(r, p, warm_start=False)
        warm = _run(r, p, warm_start=True)
        # Same number of outer iterations (the quick-min step rule
        # is deterministic, and identical SCF energies at each step
        # → identical NEB forces → identical step directions).
        assert warm.n_iter == cold.n_iter
        # Energies bit-exact.
        np.testing.assert_allclose(warm.energies, cold.energies, atol=1e-10)
        # Force convergence metric also bit-exact.
        assert abs(warm.max_force - cold.max_force) < 1e-10
        # Same TS image.
        assert warm.transition_state_index == cold.transition_state_index
        # Same final geometries per image.
        for w_img, c_img in zip(warm.path.images, cold.path.images):
            w_pos = np.array([a.xyz for a in w_img.system.atoms])
            c_pos = np.array([a.xyz for a in c_img.system.atoms])
            np.testing.assert_allclose(w_pos, c_pos, atol=1e-10)


class TestWarmStartDefault:
    def test_default_kwarg_is_true(self):
        """``warm_start=True`` is the default — calling run_neb
        without the kwarg engages the warm-start path."""
        import inspect

        sig = inspect.signature(run_neb)
        warm_param = sig.parameters.get("warm_start")
        assert warm_param is not None
        assert warm_param.default is True


class TestWarmStartOpenShellEquivalence:
    """UHF / UKS warm-start carry the ``(alpha, beta)`` density tuple
    across outer iterations. Bit-exactness against cold-start is the
    same invariant the closed-shell test covers."""

    def _h3_doublet(self, z2: float) -> Molecule:
        return Molecule(
            [
                Atom(1, [0.0, 0.0, 0.0]),
                Atom(1, [0.0, 0.0, float(z2)]),
                Atom(1, [0.0, 0.0, 4.4]),
            ],
            0,
            2,
        )

    def _run_uhf(self, *, warm_start: bool):
        from vibeqc import UHFOptions

        uhf = UHFOptions()
        uhf.max_iter = 200
        return run_neb(
            self._h3_doublet(1.4),
            self._h3_doublet(3.0),
            basis="sto-3g",
            n_images=3,
            method="UHF",
            uhf_options=uhf,
            interpolation="linear",
            max_iter=10,
            conv_tol_force=1e-2,
            n_jobs=1,
            initial_step=0.05,
            warm_start=warm_start,
        )

    def test_uhf_warm_matches_cold(self):
        cold = self._run_uhf(warm_start=False)
        warm = self._run_uhf(warm_start=True)
        assert warm.n_iter == cold.n_iter
        # UHF SCF convergence has slightly more numerical noise than
        # RHF (multiple symmetry-equivalent broken-symmetry guesses
        # land on different convergence trajectories at machine
        # precision). 1e-6 is comfortably tight for any practical
        # NEB use.
        np.testing.assert_allclose(warm.energies, cold.energies, atol=1e-6)
        for w_img, c_img in zip(warm.path.images, cold.path.images):
            w_pos = np.array([a.xyz for a in w_img.system.atoms])
            c_pos = np.array([a.xyz for a in c_img.system.atoms])
            np.testing.assert_allclose(w_pos, c_pos, atol=1e-6)


class TestWarmStartKSEquivalence:
    """RKS / UKS warm-start additionally needs the XC integration
    grid — the low-level ``run_*ks_scf_with_jk`` entry takes a
    ``xc_grid`` argument the warm-start helper builds via
    ``build_grid``."""

    def _h2(self, z2: float) -> Molecule:
        return Molecule(
            [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, float(z2)])]
        )

    def _h3_doublet(self, z2: float) -> Molecule:
        return Molecule(
            [
                Atom(1, [0.0, 0.0, 0.0]),
                Atom(1, [0.0, 0.0, float(z2)]),
                Atom(1, [0.0, 0.0, 4.4]),
            ],
            0,
            2,
        )

    def _run_rks(self, *, warm_start: bool):
        from vibeqc import RKSOptions

        opts = RKSOptions()
        opts.functional = "lda"
        return run_neb(
            self._h2(1.4),
            self._h2(2.0),
            basis="sto-3g",
            n_images=3,
            method="RKS",
            functional="lda",
            rks_options=opts,
            interpolation="linear",
            max_iter=10,
            conv_tol_force=1e-2,
            n_jobs=1,
            initial_step=0.05,
            warm_start=warm_start,
        )

    def _run_uks(self, *, warm_start: bool):
        from vibeqc import UKSOptions

        opts = UKSOptions()
        opts.functional = "lda"
        opts.max_iter = 200
        return run_neb(
            self._h3_doublet(1.4),
            self._h3_doublet(3.0),
            basis="sto-3g",
            n_images=3,
            method="UKS",
            functional="lda",
            uks_options=opts,
            interpolation="linear",
            max_iter=8,
            conv_tol_force=1e-2,
            n_jobs=1,
            initial_step=0.05,
            warm_start=warm_start,
        )

    def test_rks_warm_matches_cold(self):
        cold = self._run_rks(warm_start=False)
        warm = self._run_rks(warm_start=True)
        assert warm.n_iter == cold.n_iter
        np.testing.assert_allclose(warm.energies, cold.energies, atol=1e-7)

    def test_uks_warm_matches_cold(self):
        cold = self._run_uks(warm_start=False)
        warm = self._run_uks(warm_start=True)
        assert warm.n_iter == cold.n_iter
        # UKS has the same numerical-noise budget as UHF + an XC-grid
        # quadrature contribution that's deterministic per-geometry
        # but accrues across many SCF iterations.
        np.testing.assert_allclose(warm.energies, cold.energies, atol=1e-5)


class TestWarmStartPeriodic:
    """Periodic NEB warm-start: density blocks of the converged
    LatticeMatrixSet are cached per image and fed to
    ``run_pbc_bipole_rhf(..., initial_density=blocks)`` on the next
    outer iter. Additionally, each FD-displaced SCF within an
    outer iter warm-starts from the reference SCF's converged
    density (within-image FD speedup). These dispatch/identity tests select
    ``sr_image_precision=None`` explicitly. Their 8-bohr cells keep the
    four-bohr operator cutoff fold-converged while remaining bounded-cost
    molecular-limit diagnostics; production NEB keeps M5 padding."""

    def _h2_in_box(self, z2: float):
        from vibeqc import PeriodicSystem

        L = np.diag([8.0, 8.0, 8.0])
        return PeriodicSystem(
            3,
            L,
            [
                Atom(1, [0.0, 0.0, 0.0]),
                Atom(1, [0.0, 0.0, float(z2)]),
            ],
        )

    def _run(self, *, warm_start: bool):
        from vibeqc import LatticeSumOptions, PeriodicRHFOptions

        lat = LatticeSumOptions()
        lat.cutoff_bohr = 4.0
        opts = PeriodicRHFOptions()
        opts.lattice_opts = lat
        opts.max_iter = 30
        return run_neb(
            self._h2_in_box(1.4),
            self._h2_in_box(1.8),
            basis="sto-3g",
            n_images=2,
            method="RHF",
            rhf_options=opts,
            interpolation="linear",
            max_iter=2,
            conv_tol_force=1e-1,
            n_jobs=1,
            kpoints=(1, 1, 1),
            fd_step_bohr=5e-3,
            sr_image_precision=None,
            warm_start=warm_start,
        )

    def test_periodic_warm_matches_cold(self):
        """Periodic RHF NEB: bit-exactness of warm vs. cold path
        — the SCF converges to the same density regardless of
        initial guess, so the entire NEB trajectory (energies +
        forces + per-image positions) is identical at any fixed
        ``max_iter`` budget."""
        cold = self._run(warm_start=False)
        warm = self._run(warm_start=True)
        assert warm.n_iter == cold.n_iter
        np.testing.assert_allclose(warm.energies, cold.energies, atol=1e-8)
        assert abs(warm.max_force - cold.max_force) < 1e-8
        # Per-image final geometries identical (atoms in same cell).
        for w_img, c_img in zip(warm.path.images, cold.path.images):
            w = np.array([a.xyz for a in w_img.system.unit_cell])
            c = np.array([a.xyz for a in c_img.system.unit_cell])
            np.testing.assert_allclose(w, c, atol=1e-8)

    def test_periodic_initial_density_kwarg_on_run_pbc_bipole_rhf(self):
        """Independent of NEB: ``run_pbc_bipole_rhf(...,
        initial_density=blocks)`` reproduces the converged-from-cold
        energy when fed the converged density back in. Pins the
        low-level API contract that the NEB driver relies on."""
        from vibeqc import (
            BasisSet,
            LatticeSumOptions,
            PeriodicRHFOptions,
            monkhorst_pack,
        )
        from vibeqc.pbc_bipole import run_pbc_bipole_rhf

        sys0 = self._h2_in_box(1.4)
        km = monkhorst_pack(sys0, (1, 1, 1))
        basis = BasisSet(sys0.unit_cell_molecule(), "sto-3g")
        lat = LatticeSumOptions()
        lat.cutoff_bohr = 4.0
        opts = PeriodicRHFOptions()
        opts.lattice_opts = lat
        opts.max_iter = 30

        cold = run_pbc_bipole_rhf(
            sys0, basis, km, opts, sr_image_precision=None
        )
        blocks = [np.asarray(b).copy() for b in cold.density.blocks]
        warm = run_pbc_bipole_rhf(
            sys0,
            basis,
            km,
            opts,
            initial_density=blocks,
            sr_image_precision=None,
        )
        assert float(warm.energy) == pytest.approx(float(cold.energy), abs=1e-10)
        # Warm SCF should converge in ≤ cold's iteration count — the
        # cold path starts from SAD, the warm path starts from the
        # converged density.
        assert warm.n_iter <= cold.n_iter


class TestPeriodicUHFInitialDensity:
    """Periodic UHF warm-start kwargs (``init_alpha`` + ``init_beta``).
    Same convention as RHF but with the open-shell (α, β) split.
    The NEB driver's ``_evaluate_image_periodic`` returns the
    converged (α, β) blocks tuple and threads it back on the next
    outer iter; this test pins the low-level API contract that the
    NEB dispatch relies on."""

    def _h3_doublet_in_box(self) -> "PeriodicSystem":  # noqa: F821
        from vibeqc import PeriodicSystem

        L = np.diag([8.0, 8.0, 8.0])
        return PeriodicSystem(
            3,
            L,
            [
                Atom(1, [0.0, 0.0, 0.0]),
                Atom(1, [0.0, 0.0, 1.4]),
                Atom(1, [0.0, 0.0, 2.8]),
            ],
            charge=0,
            multiplicity=2,
        )

    def test_uhf_init_alpha_beta_kwargs_warm_matches_cold(self):
        from vibeqc import (
            BasisSet,
            LatticeSumOptions,
            PeriodicRHFOptions,
            monkhorst_pack,
        )
        from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf

        sys0 = self._h3_doublet_in_box()
        km = monkhorst_pack(sys0, (1, 1, 1))
        basis = BasisSet(sys0.unit_cell_molecule(), "sto-3g")
        lat = LatticeSumOptions()
        lat.cutoff_bohr = 4.0
        opts = PeriodicRHFOptions()
        opts.lattice_opts = lat
        opts.max_iter = 100

        cold = run_pbc_bipole_uhf(
            sys0, basis, km, opts, sr_image_precision=None
        )
        assert cold.converged
        ba = [np.asarray(b).copy() for b in cold.density_alpha.blocks]
        bb = [np.asarray(b).copy() for b in cold.density_beta.blocks]
        warm = run_pbc_bipole_uhf(
            sys0,
            basis,
            km,
            opts,
            init_alpha=ba,
            init_beta=bb,
            sr_image_precision=None,
        )
        assert warm.converged
        # Open-shell UHF can land on slightly different broken-symmetry
        # solutions at machine precision; 1e-5 Ha is the practical
        # noise budget.
        assert float(warm.energy) == pytest.approx(
            float(cold.energy), abs=1e-5
        )
        assert warm.n_iter <= cold.n_iter

    def test_uhf_periodic_neb_end_to_end_warm_matches_cold(self):
        """End-to-end periodic UHF NEB warm vs cold: exercises the
        full ``run_neb → _evaluate_image_periodic →
        run_pbc_bipole_uhf(init_alpha, init_beta) → outer-loop cache``
        dispatch path. Pins bit-exactness on energies + max_force +
        per-image positions across both modes."""
        from vibeqc import (
            LatticeSumOptions,
            PeriodicRHFOptions,
            PeriodicSystem,
            run_neb,
        )

        L = np.diag([8.0, 8.0, 8.0])
        # H₂⁺ in an 8-bohr box: 1 electron, doublet. Open-shell, small,
        # converges in a few iters per SCF.
        reactant = PeriodicSystem(
            3,
            L,
            [
                Atom(1, [0.0, 0.0, 0.0]),
                Atom(1, [0.0, 0.0, 1.4]),
            ],
            charge=1,
            multiplicity=2,
        )
        product = PeriodicSystem(
            3,
            L,
            [
                Atom(1, [0.0, 0.0, 0.0]),
                Atom(1, [0.0, 0.0, 1.8]),
            ],
            charge=1,
            multiplicity=2,
        )
        lat = LatticeSumOptions()
        lat.cutoff_bohr = 4.0
        opts = PeriodicRHFOptions()
        opts.lattice_opts = lat
        opts.max_iter = 100

        common = dict(
            basis="sto-3g",
            n_images=2,
            method="UHF",
            uhf_options=opts,
            interpolation="linear",
            max_iter=2,
            conv_tol_force=1e-1,
            n_jobs=1,
            kpoints=(1, 1, 1),
            fd_step_bohr=5e-3,
            sr_image_precision=None,
        )
        cold = run_neb(reactant, product, warm_start=False, **common)
        warm = run_neb(reactant, product, warm_start=True, **common)

        # Both modes must drive the same NEB trajectory at any fixed
        # max_iter budget. Periodic UHF has the same numerical-noise
        # budget as molecular UHF — 1e-6 Ha is comfortably tight.
        assert warm.n_iter == cold.n_iter
        np.testing.assert_allclose(warm.energies, cold.energies, atol=1e-6)
        assert abs(warm.max_force - cold.max_force) < 1e-6
        for w_img, c_img in zip(warm.path.images, cold.path.images):
            w = np.array([a.xyz for a in w_img.system.unit_cell])
            c = np.array([a.xyz for a in c_img.system.unit_cell])
            np.testing.assert_allclose(w, c, atol=1e-6)

    def test_uhf_mismatched_alpha_beta_raises(self):
        from vibeqc import (
            BasisSet,
            LatticeSumOptions,
            PeriodicRHFOptions,
            monkhorst_pack,
        )
        from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf

        sys0 = self._h3_doublet_in_box()
        km = monkhorst_pack(sys0, (1, 1, 1))
        basis = BasisSet(sys0.unit_cell_molecule(), "sto-3g")
        lat = LatticeSumOptions()
        lat.cutoff_bohr = 4.0
        opts = PeriodicRHFOptions()
        opts.lattice_opts = lat
        opts.max_iter = 30
        cold = run_pbc_bipole_uhf(
            sys0, basis, km, opts, sr_image_precision=None
        )
        ba = [np.asarray(b).copy() for b in cold.density_alpha.blocks]
        # init_alpha without init_beta (or vice versa) must raise.
        with pytest.raises(ValueError, match="provided together"):
            run_pbc_bipole_uhf(
                sys0,
                basis,
                km,
                opts,
                init_alpha=ba,
                sr_image_precision=None,
            )


class TestPeriodicRKSInitialDensity:
    """Periodic RKS warm-start kwarg (``initial_density``).

    The full ``run_neb → _evaluate_image_periodic → driver`` dispatch
    is the same shared code path covered by RHF
    (``TestWarmStartPeriodic.test_periodic_warm_matches_cold``); the
    NEB-driver layer just hands off the cached density via
    ``_warm_kwargs`` (closed-shell ``initial_density=`` branch). The
    unique surface for RKS is the BIPOLE driver kwarg + the XC-grid
    rebuild on warm vs. cold paths.

    This test pins that contract at the API level — much cheaper
    than a full periodic NEB run (~few SCFs vs. ~hundreds for the
    NEB outer loop's FD gradient × max_iter × warm/cold sweep).

    Unblocked by the ``build_xc_periodic`` arg-order fix that landed
    alongside Increment 4d-bipole UKS DFT+U (``7dabcc28``)."""

    def test_rks_initial_density_kwarg_warm_matches_cold(self):
        from vibeqc import (
            BasisSet,
            LatticeSumOptions,
            PeriodicKSOptions,
            PeriodicSystem,
            monkhorst_pack,
        )
        from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

        L = np.diag([8.0, 8.0, 8.0])
        sys0 = PeriodicSystem(
            3, L,
            [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
        )
        km = monkhorst_pack(sys0, (1, 1, 1))
        basis = BasisSet(sys0.unit_cell_molecule(), "sto-3g")
        lat = LatticeSumOptions()
        lat.cutoff_bohr = 4.0
        opts = PeriodicKSOptions()
        opts.lattice_opts = lat
        opts.max_iter = 100
        opts.functional = "lda"

        cold = run_pbc_bipole_rks(
            sys0, basis, km, opts, sr_image_precision=None
        )
        assert cold.converged
        blocks = [np.asarray(b).copy() for b in cold.density.blocks]
        warm = run_pbc_bipole_rks(
            sys0,
            basis,
            km,
            opts,
            initial_density=blocks,
            sr_image_precision=None,
        )
        assert warm.converged
        # RKS BIPOLE converges bit-exact (1e-9 Ha) on H₂/STO-3G/LDA
        # — the XC grid + functional eval are deterministic.
        assert float(warm.energy) == pytest.approx(
            float(cold.energy), abs=1e-9
        )
        assert warm.n_iter <= cold.n_iter


class TestPeriodicUKSInitialDensity:
    """Periodic UKS warm-start kwargs (``init_alpha`` + ``init_beta``).

    Open-shell sibling of ``TestPeriodicRKSInitialDensity``. Same NEB-
    driver dispatch reasoning: the full E2E path is already covered
    by ``test_uhf_periodic_neb_end_to_end_warm_matches_cold`` (the
    ``_warm_kwargs`` open-shell branch is shared between UHF and
    UKS). What's unique to UKS is the driver kwarg + the open-shell
    XC functional path under warm-start."""

    def test_uks_init_alpha_beta_kwargs_warm_matches_cold(self):
        from vibeqc import (
            BasisSet,
            LatticeSumOptions,
            PeriodicKSOptions,
            PeriodicSystem,
            monkhorst_pack,
        )
        from vibeqc.pbc_bipole_uks import run_pbc_bipole_uks

        L = np.diag([8.0, 8.0, 8.0])
        # H₂⁺ doublet: 1 electron, open-shell, minimal cost.
        sys0 = PeriodicSystem(
            3, L,
            [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
            charge=1,
            multiplicity=2,
        )
        km = monkhorst_pack(sys0, (1, 1, 1))
        basis = BasisSet(sys0.unit_cell_molecule(), "sto-3g")
        lat = LatticeSumOptions()
        lat.cutoff_bohr = 4.0
        opts = PeriodicKSOptions()
        opts.lattice_opts = lat
        opts.max_iter = 100
        opts.functional = "lda"

        cold = run_pbc_bipole_uks(
            sys0, basis, km, opts, sr_image_precision=None
        )
        assert cold.converged
        ba = [np.asarray(b).copy() for b in cold.density_alpha.blocks]
        bb = [np.asarray(b).copy() for b in cold.density_beta.blocks]
        warm = run_pbc_bipole_uks(
            sys0,
            basis,
            km,
            opts,
            init_alpha=ba,
            init_beta=bb,
            sr_image_precision=None,
        )
        assert warm.converged
        # Open-shell UKS BIPOLE: ~1e-5 Ha broken-symmetry noise budget
        # (matches the UHF / UKS molecular convention).
        assert float(warm.energy) == pytest.approx(
            float(cold.energy), abs=1e-5
        )
        assert warm.n_iter <= cold.n_iter


@pytest.mark.parametrize("method", ["rohf", "roks"])
def test_restricted_open_neb_projects_warm_seed_and_keeps_physical_guess(method):
    from vibeqc.neb import _evaluate_image

    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])], multiplicity=3)
    opts = getattr(vq, method.upper() + "Options")()
    opts.initial_guess = vq.InitialGuess.SAD
    opts.atomic_spins = [1, 1]
    opts.max_iter = 50
    grid = vq.GridOptions()
    grid.n_radial = 20
    grid.lebedev_order = 11
    kwargs = dict(functional="lda" if method == "roks" else None,
                  rhf_options=None, uhf_options=None, rks_options=None,
                  uks_options=None, gradient_options=None, grid_options=grid,
                  dispersion_params=None, **{method + "_options": opts})
    positions = np.array([a.xyz for a in mol.atoms])
    _, _, source = _evaluate_image(positions, mol, "sto-3g", method, **kwargs)
    positions[1, 2] = 1.6
    energy, gradient, warm = _evaluate_image(
        positions, mol, "sto-3g", method, initial_density=source, **kwargs,
    )
    assert np.isfinite(energy) and np.all(np.isfinite(gradient))
    assert source.guess_selection.effective == vq.InitialGuess.SAD
    assert warm.guess_selection.requested == vq.InitialGuess.SAD
    assert warm.guess_selection.effective == vq.InitialGuess.SAD
    assert warm.guess_selection.transport == vq.InitialGuess.READ
    assert opts.atomic_spins == [1, 1]
    assert opts.initial_guess == vq.InitialGuess.SAD
    overlap = np.asarray(vq.compute_overlap(warm.restart_basis))
    assert np.trace(np.asarray(warm.density_alpha) @ overlap) == pytest.approx(2, abs=1e-11)
    np.testing.assert_allclose(warm.density_beta, 0, atol=1e-14)
    assert np.linalg.norm(np.asarray(warm.density_alpha) - source.density_alpha) > 1e-3
