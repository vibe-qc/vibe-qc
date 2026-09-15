"""Periodic NEB acceptance test (Increment 4).

The Inc 4 spec: "tiny slab / periodic system, a few iters, runs
without raising, produces a sane path." Convergence quality is
not the focus — until the J^LR reciprocal-Ewald contribution
lands in the BIPOLE analytic gradient, the
periodic NEB driver uses a finite-difference fallback per image
(6N + 1 BIPOLE SCFs per image per outer iteration). The test
deliberately runs at a very tight lattice-sum cutoff so the
BIPOLE SCFs are cheap enough for CI — physical accuracy of the
gradient at that cutoff is not the point; the point is exercising
the dispatch + result-shape contract end to end.

Tests:

* ``run_neb`` accepts ``PeriodicSystem`` endpoints without
  raising and returns a ``NEBResult`` with the right shape.
* The lattice of every image (including intermediates) matches
  the endpoint lattice — variable-cell NEB is out of scope.
* Per-image energies are finite (no NaN / Inf leaking through
  the FD difference).
* ``transition_state_index`` is set.
* Mixing ``PeriodicSystem`` and ``Molecule`` endpoints raises
  ``ValueError`` (the dispatch matches the system-type guard).

The full driver test class for the molecular case
(``tests/test_neb_driver.py``) covers convergence quality + the
improved-tangent / CI-NEB algorithmics; this file covers only the
periodic-dispatch surface.
"""

from __future__ import annotations

import tomllib
from types import SimpleNamespace

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import (
    Atom,
    BasisSet,
    LatticeSumOptions,
    Molecule,
    NEBResult,
    PeriodicRHFOptions,
    PeriodicSystem,
    run_neb,
)


def test_periodic_neb_keeps_production_padding_default() -> None:
    import inspect

    param = inspect.signature(run_neb).parameters["sr_image_precision"]
    assert param.default == pytest.approx(1e-6)


def test_periodic_fd_rejects_nonconverged_displaced_scf(monkeypatch):
    """Every energy entering a finite difference must be converged."""
    import vibeqc.bipole_optimize as optimize_mod
    import vibeqc.neb as neb_mod

    system = _h2_in_cubic_box(1.4)
    positions = np.asarray(
        [atom.xyz for atom in system.unit_cell], dtype=float
    )
    calls = []

    def fake_run_scf(*args, **kwargs):
        calls.append(1)
        converged = len(calls) == 1
        result = SimpleNamespace(
            converged=converged,
            n_iter=3,
            energy=-1.0,
            density=SimpleNamespace(blocks=[np.eye(2)]),
        )
        return -1.0, result

    monkeypatch.setattr(optimize_mod, "_run_scf", fake_run_scf)
    with pytest.raises(neb_mod.NEBImageSCFError, match="did not converge"):
        neb_mod._evaluate_image_periodic(
            positions,
            system,
            "sto-3g",
            "RHF",
            kmesh=vq.monkhorst_pack(system, (1, 1, 1)),
            functional=None,
            rhf_options=PeriodicRHFOptions(),
            uhf_options=None,
            rks_options=None,
            uks_options=None,
            fd_step_bohr=1.0e-3,
            sr_image_precision=None,
        )
    assert len(calls) == 2


@pytest.fixture
def tight_lattice_opts() -> PeriodicRHFOptions:
    """BIPOLE SCF options with a deliberately tight lattice-sum
    cutoff so the test fits in CI time. The paired 8-bohr cubic cell keeps the
    measured overlap fold converged at this 4-bohr operator cutoff while
    retaining a central-cell, molecular-limit dispatch fixture. The end-to-end
    call sites pair this with the explicit ``sr_image_precision=None``
    historical-domain diagnostic; production padding remains the driver
    default."""
    lat = LatticeSumOptions()
    lat.cutoff_bohr = 4.0
    opts = PeriodicRHFOptions()
    opts.lattice_opts = lat
    opts.max_iter = 30
    return opts


def _h2_in_cubic_box(z2: float, *, dim: int = 3) -> PeriodicSystem:
    L = np.diag([8.0, 8.0, 8.0])
    return PeriodicSystem(
        dim,
        L,
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, float(z2)])],
    )


def _nacl_in_cubic_box(z2: float) -> PeriodicSystem:
    """Na and Cl both carry LANL2DZ ECPs. The ECP decision is per element,
    so an H2 box in LANL2DZ is a correct all-electron calculation; the
    ECP-paired refusal under test needs atoms that replace core electrons."""
    return PeriodicSystem(
        3,
        np.diag([8.0, 8.0, 8.0]),
        [Atom(11, [0.0, 0.0, 0.0]), Atom(17, [0.0, 0.0, float(z2)])],
    )


def _au_in_cubic_box(x: float) -> PeriodicSystem:
    return PeriodicSystem(
        3,
        np.diag([8.0, 8.0, 8.0]),
        [Atom(79, [float(x), 0.0, 0.0])],
        charge=0,
        multiplicity=2,
    )


def test_periodic_neb_dry_run_estimate_charges_fd_fanout(
    tmp_path,
    monkeypatch,
) -> None:
    from vibeqc.memory import estimate_memory, estimate_neb_memory

    reactant = _h2_in_cubic_box(1.4)
    product = _h2_in_cubic_box(1.8)
    stem = tmp_path / "periodic_neb_est"
    monkeypatch.setenv("VIBEQC_DRY_RUN", "1")
    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")

    result = run_neb(
        reactant,
        product,
        basis="sto-3g",
        n_images=2,
        method="RHF",
        output=stem,
        n_jobs=0,
    )

    assert result is None
    with stem.with_suffix(".system").open("rb") as fh:
        body = tomllib.load(fh)
    basis_obj = BasisSet(reactant.unit_cell_molecule(), "sto-3g")
    per_image = estimate_memory(
        reactant.unit_cell_molecule(),
        basis_obj,
        method="rhf",
        options=None,
    )
    expected = estimate_neb_memory(
        per_image,
        n_images=2,
        n_jobs=1,
        n_atoms=2,
        n_basis=basis_obj.nbasis,
        open_shell=False,
        finite_difference_evaluations=13,
        warm_start=True,
    )
    no_fd = estimate_neb_memory(
        per_image,
        n_images=2,
        n_jobs=1,
        n_atoms=2,
        n_basis=basis_obj.nbasis,
        open_shell=False,
        finite_difference_evaluations=1,
        warm_start=True,
    )

    assert body["outputs"]["status"] == "dry_run"
    assert body["plan"]["job_kind"] == "neb"
    assert body["memory"]["estimate_bytes"] == expected.total_bytes
    assert "NEB finite-difference gradient scratch" in expected.by_category
    assert expected.total_bytes > no_fd.total_bytes


def _forbid_periodic_image_evaluation(*args, **kwargs):
    pytest.fail("periodic image evaluation started before NEB preflight")


def _forbid_neb_manifest(*args, **kwargs):
    pytest.fail("NEB dry-run manifest was written before preflight")


@pytest.mark.parametrize(
    "invalid_precision",
    [0.0, 1.0, -1.0e-6, np.nan, np.inf],
    ids=["zero", "one", "negative", "nan", "infinity"],
)
def test_periodic_neb_rejects_invalid_sr_precision_before_dry_run_manifest(
    tmp_path,
    monkeypatch,
    invalid_precision,
) -> None:
    import vibeqc.neb as neb_mod

    monkeypatch.delenv("VIBEQC_DRY_RUN", raising=False)
    monkeypatch.setattr(
        neb_mod,
        "_evaluate_image_periodic",
        _forbid_periodic_image_evaluation,
    )
    monkeypatch.setattr(
        neb_mod,
        "_write_neb_dry_run_manifest",
        _forbid_neb_manifest,
    )
    stem = tmp_path / "invalid_sr_precision_neb"

    with pytest.raises(ValueError, match="sr_image_precision"):
        run_neb(
            _h2_in_cubic_box(1.4),
            _h2_in_cubic_box(1.8),
            basis="sto-3g",
            n_images=1,
            method="RHF",
            interpolation="linear",
            max_iter=1,
            n_jobs=1,
            sr_image_precision=invalid_precision,
            output=stem,
            dry_run=True,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
def test_periodic_neb_rejects_endpoint_dim_mismatch_before_dispatch(
    tmp_path,
    monkeypatch,
    dry_run: bool,
) -> None:
    import vibeqc.neb as neb_mod

    monkeypatch.delenv("VIBEQC_DRY_RUN", raising=False)
    monkeypatch.setattr(
        neb_mod,
        "_evaluate_image_periodic",
        _forbid_periodic_image_evaluation,
    )
    monkeypatch.setattr(
        neb_mod,
        "_write_neb_dry_run_manifest",
        _forbid_neb_manifest,
    )
    stem = tmp_path / "mixed_dimensionality_neb"

    with pytest.raises(ValueError, match="same periodic dimension"):
        run_neb(
            _h2_in_cubic_box(1.4, dim=3),
            _h2_in_cubic_box(1.8, dim=2),
            basis="sto-3g",
            n_images=1,
            method="RHF",
            interpolation="linear",
            max_iter=1,
            n_jobs=1,
            output=stem,
            dry_run=dry_run,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
@pytest.mark.parametrize("dim", [1, 2], ids=["dim-1", "dim-2"])
def test_periodic_gaussian_neb_rejects_low_dimension_before_dispatch(
    tmp_path,
    monkeypatch,
    dry_run: bool,
    dim: int,
) -> None:
    import vibeqc.neb as neb_mod

    monkeypatch.delenv("VIBEQC_DRY_RUN", raising=False)
    monkeypatch.setattr(
        neb_mod,
        "_evaluate_image_periodic",
        _forbid_periodic_image_evaluation,
    )
    stem = tmp_path / "low_dimensional_neb"

    with pytest.raises(NotImplementedError, match="requires a 3-D periodic"):
        run_neb(
            _h2_in_cubic_box(1.4, dim=dim),
            _h2_in_cubic_box(1.8, dim=dim),
            basis="sto-3g",
            n_images=1,
            method="RHF",
            interpolation="linear",
            max_iter=1,
            n_jobs=1,
            output=stem,
            dry_run=dry_run,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
def test_periodic_gaussian_neb_rejects_dispersion_before_dispatch(
    tmp_path,
    monkeypatch,
    dry_run: bool,
) -> None:
    import vibeqc.neb as neb_mod

    monkeypatch.delenv("VIBEQC_DRY_RUN", raising=False)
    monkeypatch.setattr(
        neb_mod,
        "_evaluate_image_periodic",
        _forbid_periodic_image_evaluation,
    )
    stem = tmp_path / "dispersion_neb"

    with pytest.raises(NotImplementedError, match="dispersion_params"):
        run_neb(
            _h2_in_cubic_box(1.4),
            _h2_in_cubic_box(1.8),
            basis="sto-3g",
            n_images=1,
            method="RHF",
            dispersion_params=object(),
            interpolation="linear",
            max_iter=1,
            n_jobs=1,
            output=stem,
            dry_run=dry_run,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
def test_periodic_gaussian_neb_rejects_finite_temperature_before_dispatch(
    tmp_path,
    monkeypatch,
    dry_run: bool,
) -> None:
    import vibeqc.neb as neb_mod

    monkeypatch.delenv("VIBEQC_DRY_RUN", raising=False)
    monkeypatch.setattr(
        neb_mod,
        "_evaluate_image_periodic",
        _forbid_periodic_image_evaluation,
    )
    monkeypatch.setattr(
        neb_mod,
        "_write_neb_dry_run_manifest",
        _forbid_neb_manifest,
    )
    stem = tmp_path / "finite_temperature_neb"
    options = SimpleNamespace(smearing_temperature=0.005)

    with pytest.raises(NotImplementedError, match="finite-temperature"):
        run_neb(
            _h2_in_cubic_box(1.4),
            _h2_in_cubic_box(1.8),
            basis="sto-3g",
            n_images=1,
            method="RKS",
            functional="lda",
            rks_options=options,
            interpolation="linear",
            max_iter=1,
            n_jobs=1,
            output=stem,
            dry_run=dry_run,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
def test_periodic_gaussian_neb_rejects_read_before_dispatch(
    tmp_path,
    monkeypatch,
    dry_run: bool,
) -> None:
    import vibeqc.neb as neb_mod

    monkeypatch.delenv("VIBEQC_DRY_RUN", raising=False)
    monkeypatch.setattr(
        neb_mod,
        "_evaluate_image_periodic",
        _forbid_periodic_image_evaluation,
    )
    monkeypatch.setattr(
        neb_mod,
        "_write_neb_dry_run_manifest",
        _forbid_neb_manifest,
    )
    options = PeriodicRHFOptions()
    options.initial_guess = vq.InitialGuess.READ
    options.read_path = "prior.qvf"
    stem = tmp_path / "read_neb"

    with pytest.raises(NotImplementedError, match=r"does not support.*READ"):
        run_neb(
            _h2_in_cubic_box(1.4),
            _h2_in_cubic_box(1.8),
            basis="sto-3g",
            n_images=1,
            method="RHF",
            rhf_options=options,
            interpolation="linear",
            max_iter=1,
            n_jobs=1,
            output=stem,
            dry_run=dry_run,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
@pytest.mark.parametrize(
    ("method", "options_name", "options_factory", "guess"),
    [
        ("RHF", "rhf_options", vq.PeriodicRHFOptions, vq.InitialGuess.FRAGMO),
        ("UHF", "uhf_options", vq.PeriodicRHFOptions, vq.InitialGuess.FRAGMO),
        ("RKS", "rks_options", vq.PeriodicKSOptions, vq.InitialGuess.FRAGMO),
        ("UKS", "uks_options", vq.PeriodicKSOptions, vq.InitialGuess.FRAGMO),
    ],
    ids=[
        "rhf-fragmo",
        "uhf-fragmo",
        "rks-fragmo",
        "uks-fragmo",
    ],
)
def test_periodic_gaussian_neb_rejects_unsupported_guesses_before_dispatch(
    tmp_path,
    monkeypatch,
    dry_run: bool,
    method: str,
    options_name: str,
    options_factory,
    guess,
) -> None:
    import vibeqc.neb as neb_mod

    monkeypatch.delenv("VIBEQC_DRY_RUN", raising=False)
    monkeypatch.setattr(
        neb_mod,
        "_evaluate_image_periodic",
        _forbid_periodic_image_evaluation,
    )
    monkeypatch.setattr(
        neb_mod,
        "_write_neb_dry_run_manifest",
        _forbid_neb_manifest,
    )
    options = options_factory()
    options.initial_guess = guess
    stem = tmp_path / f"{method.lower()}-{guess.name.lower()}-neb"
    kwargs = {options_name: options}
    if method in ("RKS", "UKS"):
        kwargs["functional"] = "lda"

    with pytest.raises(NotImplementedError, match=guess.name):
        run_neb(
            _h2_in_cubic_box(1.4),
            _h2_in_cubic_box(1.8),
            basis="sto-3g",
            n_images=1,
            method=method,
            interpolation="linear",
            max_iter=1,
            n_jobs=1,
            output=stem,
            dry_run=dry_run,
            **kwargs,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
@pytest.mark.parametrize(
    "metadata_field",
    ["smearing", "bz_integration"],
    ids=["smearing", "bz-integration"],
)
def test_periodic_gaussian_neb_rejects_kpoints_occupation_metadata(
    tmp_path,
    monkeypatch,
    dry_run: bool,
    metadata_field: str,
) -> None:
    import vibeqc.neb as neb_mod

    monkeypatch.delenv("VIBEQC_DRY_RUN", raising=False)
    monkeypatch.setattr(
        neb_mod,
        "_evaluate_image_periodic",
        _forbid_periodic_image_evaluation,
    )
    monkeypatch.setattr(
        neb_mod,
        "_write_neb_dry_run_manifest",
        _forbid_neb_manifest,
    )
    reactant = _h2_in_cubic_box(1.4)
    user_kpoints = vq.KPoints.gamma(reactant)
    if metadata_field == "smearing":
        user_kpoints.smearing = vq.SmearingOptions(temperature=0.005)
    else:
        user_kpoints.bz_integration = "gilat"
    stem = tmp_path / f"kpoints_{metadata_field}_neb"

    with pytest.raises(
        NotImplementedError,
        match=rf"KPoints\.{metadata_field}",
    ):
        run_neb(
            reactant,
            _h2_in_cubic_box(1.8),
            basis="sto-3g",
            n_images=1,
            method="RKS",
            functional="lda",
            interpolation="linear",
            max_iter=1,
            n_jobs=1,
            kpoints=user_kpoints,
            output=stem,
            dry_run=dry_run,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
@pytest.mark.parametrize(
    "invalid_mesh",
    [
        (2.9, 1, 1),
        [True, 1, 1],
        (1, 1),
        [0, 1, 1],
    ],
    ids=["fractional", "boolean", "wrong-length", "non-positive"],
)
def test_periodic_gaussian_neb_rejects_invalid_mesh_before_dispatch(
    tmp_path,
    monkeypatch,
    dry_run: bool,
    invalid_mesh,
) -> None:
    import vibeqc.neb as neb_mod

    monkeypatch.delenv("VIBEQC_DRY_RUN", raising=False)
    monkeypatch.setattr(
        neb_mod,
        "_evaluate_image_periodic",
        _forbid_periodic_image_evaluation,
    )
    monkeypatch.setattr(
        neb_mod,
        "_write_neb_dry_run_manifest",
        _forbid_neb_manifest,
    )
    stem = tmp_path / "invalid_kmesh_neb"

    with pytest.raises(
        ValueError,
        match="exactly three positive integers",
    ):
        run_neb(
            _h2_in_cubic_box(1.4),
            _h2_in_cubic_box(1.8),
            basis="sto-3g",
            n_images=1,
            method="RHF",
            interpolation="linear",
            max_iter=1,
            n_jobs=1,
            kpoints=invalid_mesh,
            output=stem,
            dry_run=dry_run,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
@pytest.mark.parametrize(
    ("unsupported", "error_match"),
    [
        ("iterative-solver", "iterative diagonalization"),
        ("lone-twist", "lone non-Gamma"),
        ("spin-schedule-multik", "SPIN_SCHEDULE is Gamma-only"),
        ("ad-hoc-multik", "complete Monkhorst-Pack mesh"),
    ],
    ids=["solver", "lone-twist", "spin-schedule", "ad-hoc-multik"],
)
def test_periodic_gaussian_neb_rejects_unsupported_bipole_kmesh_state(
    tmp_path,
    monkeypatch,
    dry_run: bool,
    unsupported: str,
    error_match: str,
) -> None:
    import vibeqc.neb as neb_mod

    monkeypatch.delenv("VIBEQC_DRY_RUN", raising=False)
    monkeypatch.setattr(
        neb_mod,
        "_evaluate_image_periodic",
        _forbid_periodic_image_evaluation,
    )
    monkeypatch.setattr(
        neb_mod,
        "_write_neb_dry_run_manifest",
        _forbid_neb_manifest,
    )
    reactant = _h2_in_cubic_box(1.4)
    options = PeriodicRHFOptions()
    user_kpoints = None
    if unsupported == "iterative-solver":
        options.use_davidson = True
    elif unsupported == "lone-twist":
        user_kpoints = vq.KPoints.from_list(reactant, [[0.25, 0.0, 0.0]])
    elif unsupported == "spin-schedule-multik":
        options.spinlock_mode = vq.SpinlockMode.SPIN_SCHEDULE
        options.spinlock_iterations = 2
        user_kpoints = vq.KPoints.gamma_centred(reactant, (2, 1, 1))
    else:
        user_kpoints = vq.KPoints.from_list(
            reactant,
            [[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0]],
        )
    stem = tmp_path / f"unsupported_{unsupported}_neb"

    with pytest.raises(NotImplementedError, match=error_match):
        run_neb(
            reactant,
            _h2_in_cubic_box(1.8),
            basis="sto-3g",
            n_images=1,
            method="UHF",
            uhf_options=options,
            interpolation="linear",
            max_iter=1,
            n_jobs=1,
            kpoints=user_kpoints,
            output=stem,
            dry_run=dry_run,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
def test_periodic_gaussian_neb_rejects_ecp_metadata_before_dispatch(
    tmp_path,
    monkeypatch,
    dry_run: bool,
) -> None:
    import vibeqc.neb as neb_mod

    monkeypatch.delenv("VIBEQC_DRY_RUN", raising=False)
    monkeypatch.setattr(
        neb_mod,
        "_evaluate_image_periodic",
        _forbid_periodic_image_evaluation,
    )
    options = SimpleNamespace(ecp_total_ncore=2)
    stem = tmp_path / "ecp_metadata_neb"

    with pytest.raises(NotImplementedError, match="ECP metadata"):
        run_neb(
            _h2_in_cubic_box(1.4),
            _h2_in_cubic_box(1.8),
            basis="sto-3g",
            n_images=1,
            method="RHF",
            rhf_options=options,
            interpolation="linear",
            max_iter=1,
            n_jobs=1,
            output=stem,
            dry_run=dry_run,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
def test_periodic_gaussian_neb_rejects_ecp_paired_basis_before_dispatch(
    tmp_path,
    monkeypatch,
    dry_run: bool,
) -> None:
    import vibeqc.neb as neb_mod

    monkeypatch.delenv("VIBEQC_DRY_RUN", raising=False)
    monkeypatch.setattr(
        neb_mod,
        "_evaluate_image_periodic",
        _forbid_periodic_image_evaluation,
    )
    stem = tmp_path / "ecp_paired_basis_neb"

    with pytest.raises(NotImplementedError, match="ECP metadata"):
        run_neb(
            _nacl_in_cubic_box(4.4),
            _nacl_in_cubic_box(4.8),
            basis="lanl2dz",
            n_images=1,
            method="RHF",
            interpolation="linear",
            max_iter=1,
            n_jobs=1,
            output=stem,
            dry_run=dry_run,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
def test_periodic_gaussian_neb_rejects_heavy_def2_basis_before_dispatch(
    tmp_path,
    monkeypatch,
    dry_run: bool,
) -> None:
    import vibeqc.neb as neb_mod

    monkeypatch.delenv("VIBEQC_DRY_RUN", raising=False)
    monkeypatch.setattr(
        neb_mod,
        "_evaluate_image_periodic",
        _forbid_periodic_image_evaluation,
    )
    stem = tmp_path / "au_def2_neb"

    with pytest.raises(NotImplementedError, match="ECP metadata"):
        run_neb(
            _au_in_cubic_box(0.0),
            _au_in_cubic_box(0.2),
            basis="def2-tzvp",
            n_images=1,
            method="UHF",
            interpolation="linear",
            max_iter=1,
            n_jobs=1,
            output=stem,
            dry_run=dry_run,
        )

    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
def test_periodic_gaussian_neb_rejects_pob_basis_before_dispatch(
    tmp_path,
    monkeypatch,
    dry_run: bool,
) -> None:
    import vibeqc.neb as neb_mod

    monkeypatch.delenv("VIBEQC_DRY_RUN", raising=False)
    monkeypatch.setattr(
        neb_mod,
        "_evaluate_image_periodic",
        _forbid_periodic_image_evaluation,
    )
    stem = tmp_path / "pob_basis_neb"

    with pytest.raises(NotImplementedError, match="POB basis families"):
        run_neb(
            _h2_in_cubic_box(1.4),
            _h2_in_cubic_box(1.8),
            basis="pob-tzvp-rev2",
            n_images=1,
            method="RHF",
            interpolation="linear",
            max_iter=1,
            n_jobs=1,
            output=stem,
            dry_run=dry_run,
        )

    assert not stem.with_suffix(".system").exists()


class TestPeriodicNEBDispatch:
    """Run a single-outer-iteration periodic NEB and verify the
    API surface holds together — does NOT assert convergence."""

    def test_runs_without_raising_and_returns_periodic_path(
        self, tight_lattice_opts: PeriodicRHFOptions
    ) -> None:
        reactant = _h2_in_cubic_box(1.4)
        product = _h2_in_cubic_box(1.8)
        result = run_neb(
            reactant, product,
            basis="sto-3g",
            n_images=2,
            method="RHF",
            rhf_options=tight_lattice_opts,
            interpolation="linear",
            max_iter=1,
            conv_tol_force=1e-1,
            n_jobs=1,
            initial_step=0.05,
            kpoints=(1, 1, 1),
            fd_step_bohr=5e-3,
            sr_image_precision=None,
        )
        # NEBResult shape.
        assert isinstance(result, NEBResult)
        assert len(result.path.images) == 4  # 2 intermediate + 2 endpoints
        # Every image must be a PeriodicSystem (no silent
        # conversion to Molecule along the way).
        for img in result.path.images:
            assert isinstance(img.system, PeriodicSystem)
        # Energies are finite (no NaN / Inf leaking through FD).
        assert np.all(np.isfinite(result.energies))
        # TS index is set.
        assert result.transition_state_index is not None
        assert 1 <= result.transition_state_index <= 2
        # max_force is finite — the FD-gradient path can blow up
        # when an SCF doesn't converge at a displaced geometry, so
        # we don't assert a magnitude bound here. NaN / Inf would
        # be a hard failure though.
        assert np.isfinite(result.max_force)

    def test_lattice_preserved_across_every_image(
        self, tight_lattice_opts: PeriodicRHFOptions
    ) -> None:
        """Variable-cell NEB is out of scope — the lattice of every
        intermediate image must match the endpoint lattice exactly."""
        reactant = _h2_in_cubic_box(1.4)
        product = _h2_in_cubic_box(1.8)
        result = run_neb(
            reactant, product,
            basis="sto-3g",
            n_images=2, method="RHF",
            rhf_options=tight_lattice_opts,
            interpolation="linear",
            max_iter=1, conv_tol_force=1e-1, n_jobs=1,
            kpoints=(1, 1, 1), fd_step_bohr=5e-3,
            sr_image_precision=None,
        )
        ref = np.asarray(reactant.lattice, dtype=float)
        for img in result.path.images:
            L = np.asarray(img.system.lattice, dtype=float)
            np.testing.assert_allclose(L, ref, atol=1e-12)

    def test_kpoints_accepts_tuple_and_resolves_via_monkhorst_pack(
        self, tight_lattice_opts: PeriodicRHFOptions
    ) -> None:
        """Passing ``kpoints=(1, 1, 1)`` must produce the same result
        as passing ``kpoints=None`` (both default to Γ-only)."""
        reactant = _h2_in_cubic_box(1.4)
        product = _h2_in_cubic_box(1.8)
        result_explicit = run_neb(
            reactant, product,
            basis="sto-3g", n_images=2, method="RHF",
            rhf_options=tight_lattice_opts,
            interpolation="linear",
            max_iter=1, conv_tol_force=1e-1, n_jobs=1,
            kpoints=(1, 1, 1), fd_step_bohr=5e-3,
            sr_image_precision=None,
        )
        result_default = run_neb(
            reactant, product,
            basis="sto-3g", n_images=2, method="RHF",
            rhf_options=tight_lattice_opts,
            interpolation="linear",
            max_iter=1, conv_tol_force=1e-1, n_jobs=1,
            kpoints=None, fd_step_bohr=5e-3,
            sr_image_precision=None,
        )
        np.testing.assert_allclose(
            result_explicit.energies,
            result_default.energies,
            atol=1e-10,
        )

    @pytest.mark.parametrize(
        "kpoints_factory",
        [
            lambda sysp: vq.monkhorst_pack(sysp, [1, 1, 1]),
            lambda sysp: vq.KPoints.gamma(sysp),
        ],
        ids=["native-bloch-kmesh", "kpoints-object"],
    )
    def test_kpoints_accepts_materialized_mesh_objects(
        self,
        monkeypatch,
        tight_lattice_opts: PeriodicRHFOptions,
        kpoints_factory,
    ) -> None:
        """Periodic NEB must pass materialized k-mesh objects to BIPOLE."""
        import vibeqc.neb as neb_mod

        reactant = _h2_in_cubic_box(1.4)
        product = _h2_in_cubic_box(1.8)
        user_kpoints = kpoints_factory(reactant)
        expected = vq.as_bloch_kmesh(user_kpoints)
        captured = []
        captured_sr_precision = []

        def fake_evaluate_image_periodic(
            positions,
            template,
            basis_name,
            method,
            *,
            kmesh,
            sr_image_precision,
            **kwargs,
        ):
            captured.append(kmesh)
            captured_sr_precision.append(sr_image_precision)
            return 0.0, np.zeros_like(positions), None

        monkeypatch.setattr(
            neb_mod,
            "_evaluate_image_periodic",
            fake_evaluate_image_periodic,
        )

        result = run_neb(
            reactant,
            product,
            basis="sto-3g",
            n_images=1,
            method="RHF",
            rhf_options=tight_lattice_opts,
            interpolation="linear",
            max_iter=1,
            conv_tol_force=1e-1,
            n_jobs=1,
            kpoints=user_kpoints,
            fd_step_bohr=5e-3,
            sr_image_precision=None,
        )

        assert isinstance(result, NEBResult)
        assert captured
        assert captured_sr_precision
        assert all(value is None for value in captured_sr_precision)
        for kmesh in captured:
            np.testing.assert_allclose(
                np.asarray(kmesh.kpoints, dtype=float),
                np.asarray(expected.kpoints, dtype=float),
            )
            np.testing.assert_allclose(
                np.asarray(kmesh.weights, dtype=float),
                np.asarray(expected.weights, dtype=float),
            )
        if isinstance(user_kpoints, vq.BlochKMesh):
            assert all(kmesh is user_kpoints for kmesh in captured)


class TestMixedEndpointsRejected:
    def test_periodic_reactant_molecular_product_raises(self) -> None:
        reactant = _h2_in_cubic_box(1.4)
        product = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.8])])
        with pytest.raises(ValueError, match="same system type"):
            run_neb(reactant, product, basis="sto-3g", n_images=2)

    def test_molecular_reactant_periodic_product_raises(self) -> None:
        reactant = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
        product = _h2_in_cubic_box(1.8)
        with pytest.raises(ValueError, match="same system type"):
            run_neb(reactant, product, basis="sto-3g", n_images=2)
