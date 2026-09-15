"""Semiempirical energy/gradient routes through the public NEB driver."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.semiempirical.methods.msindo_ccm import CCMOptions
from vibeqc.semiempirical.periodic import (
    evaluate_periodic_energy_gradient,
    finite_difference_gradient,
)
from vibeqc.semiempirical.routes import (
    EXECUTION_NATIVE,
    EXECUTION_NATIVE_BATCHED_FD,
    EXECUTION_PYTHON_ORCHESTRATION,
    SemiempiricalRoutePlan,
    plan_periodic_semiempirical_route,
)
from vibeqc.semiempirical.seccm import (
    bind_finite_group,
    build_seccm_topology,
)


def _h2(distance: float = 1.4, *, multiplicity: int = 1) -> vq.Molecule:
    return vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, distance])],
        0,
        multiplicity,
    )


def _h2_slab(distance: float = 1.4) -> vq.PeriodicSystem:
    return vq.PeriodicSystem(
        2,
        np.diag([8.0, 8.0, 30.0]),
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, distance])],
    )


def _hli_chain(distance: float = 1.39) -> vq.PeriodicSystem:
    return vq.PeriodicSystem(
        1,
        np.diag([4.1, 30.0, 30.0]),
        [
            vq.Atom(1, [0.17, 0.31, 0.0]),
            vq.Atom(3, [distance, -0.22, 0.0]),
        ],
    )


def _dftb0_seccm_chain(
    distance: float = 1.4,
    *,
    replicas: int = 1,
) -> tuple[vq.PeriodicSystem, object]:
    primitive = np.array([8.0, 0.0, 0.0])
    coords = np.vstack(
        [
            np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]])
            + cell * primitive
            for cell in range(replicas)
        ]
    )
    cyclic = replicas * primitive
    topology = bind_finite_group(
        build_seccm_topology(
            coords,
            [cyclic],
            length_unit="bohr",
            geometry_quantum=1.0e-10,
        ),
        primitive_vectors=[primitive],
        replicas=(replicas, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    system = vq.PeriodicSystem(
        1,
        np.diag([cyclic[0], 20.0, 20.0]),
        [vq.Atom(1, coordinate.tolist()) for coordinate in coords],
    )
    return system, topology


def _dftb0_seccm_skew_chain(
    distance: float = 1.4,
) -> tuple[vq.PeriodicSystem, object]:
    primitive = np.array([8.0, 1.25, 0.5])
    direction = primitive / np.linalg.norm(primitive)
    coords = np.vstack((np.zeros(3), distance * direction))
    topology = bind_finite_group(
        build_seccm_topology(
            coords,
            [primitive],
            length_unit="bohr",
            geometry_quantum=1.0e-10,
        ),
        primitive_vectors=[primitive],
        replicas=(1, 1, 1),
        geometry_tolerance=1.0e-9,
        length_unit="bohr",
    )
    lattice = np.column_stack(
        (primitive, np.array([0.0, 20.0, 0.0]), np.array([0.0, 0.0, 20.0]))
    )
    system = vq.PeriodicSystem(
        1,
        lattice,
        [vq.Atom(1, coordinate.tolist()) for coordinate in coords],
    )
    return system, topology


def _fake_image_evaluator(
    positions: np.ndarray,
    **_kwargs,
) -> tuple[float, np.ndarray, None]:
    coords = np.asarray(positions, dtype=float)
    return float(np.sum(coords * coords)), 2.0 * coords, None


@pytest.mark.parametrize(
    "method, expected_method",
    [
        ("dftb", "dftb0"),
        ("scc-dftb", "scc_dftb"),
        ("gfn2-xtb", "gfn2_xtb"),
        ("pm6", "pm6"),
        ("upm6", "upm6"),
        ("om1", "om1"),
        ("om2", "om2"),
        ("om3", "om3"),
        ("msindo", "msindo"),
    ],
)
def test_molecular_semiempirical_neb_dispatches_canonical_route(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    expected_method: str,
) -> None:
    import vibeqc.neb as neb

    monkeypatch.setattr(neb, "_evaluate_image_semiempirical", _fake_image_evaluator)
    monkeypatch.setattr(neb, "_evaluate_image_msindo", _fake_image_evaluator)

    result = vq.run_neb(
        _h2(1.4),
        _h2(1.6),
        method=method,
        n_images=1,
        interpolation="linear",
        max_iter=1,
        conv_tol_force=1.0e6,
        n_jobs=1,
    )

    assert result is not None
    assert result.method == expected_method
    assert result.basis is None
    assert result.functional is None


def test_semiempirical_neb_progress_uses_output_channel(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import vibeqc.neb as neb
    from vibeqc.output import OutputChannel

    monkeypatch.setattr(neb, "_evaluate_image_semiempirical", _fake_image_evaluator)
    stream = io.StringIO()
    with OutputChannel.to_stream(stream):
        vq.run_neb(
            _h2(1.4),
            _h2(1.6),
            method="dftb0",
            n_images=1,
            interpolation="linear",
            max_iter=1,
            conv_tol_force=1.0e6,
            n_jobs=1,
            progress=True,
        )

    assert "neb iter" in stream.getvalue()
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "method, expected_method",
    [
        ("dftb-0", "dftb0"),
        ("sccdftb", "scc_dftb"),
        ("gfn2", "gfn2_xtb"),
        ("pm6", "pm6"),
    ],
)
def test_gamma_periodic_semiempirical_neb_dispatches_canonical_route(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    expected_method: str,
) -> None:
    import vibeqc.neb as neb

    monkeypatch.setattr(
        neb,
        "_evaluate_image_periodic_semiempirical",
        _fake_image_evaluator,
    )
    result = vq.run_neb(
        _h2_slab(1.4),
        _h2_slab(1.6),
        method=method,
        kpoints=(1, 1, 1),
        n_images=1,
        interpolation="linear",
        max_iter=1,
        conv_tol_force=1.0e6,
        n_jobs=1,
    )

    assert result is not None
    assert result.method == expected_method
    assert result.is_periodic
    assert result.basis is None
    assert result.functional is None


@pytest.mark.parametrize(
    "method, status_route, execution",
    [
        ("dftb0", "periodic-dftb0-gradient-analytic", EXECUTION_NATIVE),
        (
            "scc-dftb",
            "periodic-scc-dftb-gradient-fd",
            EXECUTION_PYTHON_ORCHESTRATION,
        ),
        (
            "gfn2-xtb",
            "periodic-gfn2-gradient",
            EXECUTION_PYTHON_ORCHESTRATION,
        ),
        ("pm6", "periodic-pm6", EXECUTION_NATIVE_BATCHED_FD),
    ],
)
def test_periodic_neb_gradient_route_status_is_explicit(
    method: str,
    status_route: str,
    execution: str,
) -> None:
    plan = plan_periodic_semiempirical_route(
        method,
        _h2_slab(),
        properties=("energy", "gradient"),
    )

    assert plan.status_route == status_route
    assert plan.execution == execution


@pytest.mark.parametrize("method", ["pm7", "om1", "om2", "om3"])
def test_incomplete_periodic_nddo_neb_routes_are_gated(method: str) -> None:
    with pytest.raises(NotImplementedError):
        plan_periodic_semiempirical_route(
            method,
            _h2_slab(),
            properties=("energy", "gradient"),
        )
    with pytest.raises(NotImplementedError):
        evaluate_periodic_energy_gradient(method, _h2_slab())
    with pytest.raises(NotImplementedError):
        vq.run_neb(
            _h2_slab(1.4),
            _h2_slab(1.6),
            method=method,
            kpoints=(1, 1, 1),
            n_images=1,
            max_iter=1,
        )


def test_seccm_neb_route_selects_analytic_gradient_status() -> None:
    plan = SemiempiricalRoutePlan.from_request(
        "seccm",
        properties=("energy", "gradient"),
        ccm_options=CCMOptions(
            translations=[[3.2, 0.0, 0.0]],
            madelung=True,
        ),
    )

    assert plan.status_route == "msindo-ccm-gradient-analytic"
    assert plan.execution == EXECUTION_NATIVE


def test_periodic_dftb0_analytic_gradient_matches_mgo_slab_fd() -> None:
    from vibeqc._vibeqc_core import semiempirical as _se
    from vibeqc.semiempirical.parameters import default_parameters

    slab = vq.PeriodicSystem(
        2,
        np.diag([8.0, 8.0, 30.0]),
        [
            vq.Atom(12, [0.2, 0.1, 0.0]),
            vq.Atom(8, [3.5, 4.0, 0.4]),
        ],
    )
    energy, analytic = evaluate_periodic_energy_gradient("dftb0", slab)

    params = default_parameters()
    options = _se.PeriodicDFTB0Options()
    options.cutoff_bohr = 15.0
    finite_difference = finite_difference_gradient(
        slab,
        lambda candidate: float(
            _se.run_dftb0_gamma(candidate, params, options).energy
        ),
        h=1.0e-4,
    )

    assert energy == pytest.approx(-3.768068706362752, abs=1.0e-12)
    assert np.max(np.abs(analytic)) > 1.0e-5
    assert np.max(np.abs(analytic - finite_difference)) < 2.0e-8


def test_periodic_dftb0_gradient_drives_public_neb() -> None:
    result = vq.run_neb(
        _h2_slab(1.3),
        _h2_slab(1.5),
        method="dftb0",
        n_images=1,
        interpolation="linear",
        max_iter=1,
        conv_tol_force=1.0e6,
        n_jobs=1,
    )
    assert result is not None
    assert result.method == "dftb0"
    assert np.all(np.isfinite(result.energies))
    assert all(
        image.gradient is not None
        and np.all(np.isfinite(image.gradient))
        for image in result.path.images
    )


def test_dftb0_seccm_analytic_gradient_drives_public_neb() -> None:
    reactant, topology = _dftb0_seccm_chain(1.3)
    product, _ = _dftb0_seccm_chain(1.5)

    result = vq.run_neb(
        reactant,
        product,
        method="dftb0",
        seccm_topology=topology,
        n_images=1,
        interpolation="linear",
        max_iter=1,
        conv_tol_force=1.0e6,
        n_jobs=1,
    )

    assert result is not None
    assert result.method == "dftb0"
    assert result.semiempirical_route_method == "dftb0_seccm"
    assert result.is_periodic
    assert np.all(np.isfinite(result.energies))
    assert all(
        image.gradient is not None
        and np.all(np.isfinite(image.gradient))
        for image in result.path.images
    )


def test_dftb0_seccm_neb_accepts_skew_active_lattice_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibeqc.neb as neb

    monkeypatch.setattr(neb, "_evaluate_image_semiempirical", _fake_image_evaluator)
    reactant, topology = _dftb0_seccm_skew_chain(1.3)
    product, _ = _dftb0_seccm_skew_chain(1.5)

    result = vq.run_neb(
        reactant,
        product,
        method="dftb0",
        seccm_topology=topology,
        n_images=1,
        interpolation="linear",
        max_iter=1,
        conv_tol_force=1.0e6,
        n_jobs=1,
    )

    assert result is not None
    assert result.semiempirical_route_method == "dftb0_seccm"


def test_dftb0_seccm_neb_fails_before_image_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibeqc.neb as neb

    reactant, topology = _dftb0_seccm_chain(1.3)
    product, _ = _dftb0_seccm_chain(1.5)
    called = False

    def evaluator(*_args, **_kwargs):
        nonlocal called
        called = True
        return _fake_image_evaluator(*_args, **_kwargs)

    monkeypatch.setattr(neb, "_evaluate_image_semiempirical", evaluator)
    bad_topology = topology.rebuild_displacements(
        np.array([list(atom.xyz) for atom in reactant.unit_cell]),
        translations=[np.array([9.0, 0.0, 0.0])],
    )
    with pytest.raises(ValueError, match="must match"):
        vq.run_neb(
            reactant,
            product,
            method="dftb0",
            seccm_topology=bad_topology,
            n_images=1,
        )
    assert not called


def test_dftb0_seccm_neb_ties_require_explicit_trust_bound() -> None:
    reactant, topology = _dftb0_seccm_chain(1.3, replicas=2)
    product, _ = _dftb0_seccm_chain(1.5, replicas=2)

    assert topology.has_reference_ties
    with pytest.raises(ValueError, match="seccm_max_tie_score_excursion"):
        vq.run_neb(
            reactant,
            product,
            method="dftb0",
            seccm_topology=topology,
            n_images=1,
        )

    result = vq.run_neb(
        reactant,
        product,
        method="dftb0",
        seccm_topology=topology,
        seccm_max_tie_score_excursion=1.0e-2,
        n_images=1,
        interpolation="linear",
        max_iter=1,
        conv_tol_force=1.0e6,
        n_jobs=1,
    )
    assert result is not None


@pytest.mark.parametrize(
    "method",
    ["scc-dftb", "gfn2-xtb", "pm6"],
)
def test_periodic_semiempirical_energy_gradient_evaluators_are_finite(
    method: str,
) -> None:
    energy, gradient = evaluate_periodic_energy_gradient(
        method,
        _h2_slab(1.4),
    )

    assert np.isfinite(energy)
    assert gradient.shape == (2, 3)
    assert np.all(np.isfinite(gradient))
    assert np.max(np.abs(gradient)) > 1.0e-4


@pytest.mark.parametrize(
    "method, expected_method, expected_status",
    [
        ("dftb-0", "dftb0", "periodic-dftb0-kpoint-gradient-fd"),
        (
            "sccdftb",
            "scc_dftb",
            "periodic-scc-dftb-kpoint-gradient-fd",
        ),
    ],
)
def test_periodic_dftb_neb_dispatches_full_k_route(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    expected_method: str,
    expected_status: str,
) -> None:
    import vibeqc.neb as neb

    seen = []

    def evaluator(positions, *, route_plan, kpoints, **kwargs):
        seen.append((route_plan, kpoints))
        return _fake_image_evaluator(positions, **kwargs)

    monkeypatch.setattr(neb, "_evaluate_image_periodic_semiempirical", evaluator)
    result = vq.run_neb(
        _h2_slab(1.4),
        _h2_slab(1.6),
        method=method,
        kpoints=(2, 1, 1),
        n_images=1,
        interpolation="linear",
        max_iter=1,
        conv_tol_force=1.0e6,
        n_jobs=1,
    )

    assert result is not None
    assert result.method == expected_method
    assert seen
    assert all(plan.boundary == "periodic_k" for plan, _ in seen)
    assert all(plan.status_route == expected_status for plan, _ in seen)
    assert all(mesh == (2, 1, 1) for _, mesh in seen)


def test_periodic_non_dftb_neb_rejects_full_k_before_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibeqc.neb as neb

    called = False

    def evaluator(*_args, **_kwargs):
        nonlocal called
        called = True
        return _fake_image_evaluator(*_args, **_kwargs)

    monkeypatch.setattr(neb, "_evaluate_image_periodic_semiempirical", evaluator)
    with pytest.raises(NotImplementedError, match="only for DFTB0 and SCC-DFTB"):
        vq.run_neb(
            _h2_slab(1.4),
            _h2_slab(1.6),
            method="gfn2-xtb",
            kpoints=(2, 1, 1),
            n_images=1,
        )
    assert not called


@pytest.mark.parametrize("kpoints", [(1.2, 1, 1), (1, 1, 0)])
def test_periodic_semiempirical_neb_rejects_invalid_mesh_tuples(kpoints) -> None:
    with pytest.raises(ValueError, match="three finite positive integers"):
        vq.run_neb(
            _h2_slab(1.4),
            _h2_slab(1.6),
            method="dftb0",
            kpoints=kpoints,
            n_images=1,
        )


def test_dftb0_full_k_native_gradient_drives_public_neb_qvf(tmp_path) -> None:
    result = vq.run_neb(
        _hli_chain(1.39),
        _hli_chain(1.45),
        method="dftb0",
        kpoints=(3, 1, 1),
        semiempirical_cutoff_bohr=12.0,
        n_images=1,
        interpolation="linear",
        max_iter=1,
        conv_tol_force=1.0e6,
        n_jobs=1,
    )

    assert result is not None
    assert result.method == "dftb0"
    assert result.is_periodic
    assert np.all(np.isfinite(result.energies))
    for image in result.path.images:
        assert image.gradient is not None
        assert np.all(np.isfinite(image.gradient))

    blob = _archive_text(result.write_qvf(tmp_path / "dftb0_full_k_neb"))
    assert "porezag_dftb0_1995" in blob


@pytest.mark.parametrize(
    "product, match",
    [(_h2(1.6), "same charge"), (_h2(1.6, multiplicity=3), "multiplicity")],
)
def test_semiempirical_neb_rejects_incompatible_electronic_state(
    product: vq.Molecule,
    match: str,
) -> None:
    reactant = _h2(1.4)
    if "charge" in match:
        product = vq.Molecule(list(product.atoms), 2, product.multiplicity)
    with pytest.raises(ValueError, match=match):
        vq.run_neb(reactant, product, method="dftb0", n_images=1)


def test_periodic_upm6_neb_fails_closed() -> None:
    with pytest.raises(NotImplementedError, match="closed-shell only"):
        vq.run_neb(
            _h2_slab(1.4),
            _h2_slab(1.6),
            method="upm6",
            n_images=1,
        )


def _periodic_hf_chain(distance_ang: float = 1.6) -> vq.PeriodicSystem:
    from vibeqc.molecule import ANGSTROM_TO_BOHR

    return vq.PeriodicSystem(
        1,
        np.diag([3.2, 20.0, 20.0]) * ANGSTROM_TO_BOHR,
        [
            vq.Atom(1, [0.0, 0.0, 0.0]),
            vq.Atom(9, [distance_ang * ANGSTROM_TO_BOHR, 0.0, 0.0]),
        ],
    )


def _periodic_hf_skew_chain(distance_ang: float = 1.6) -> vq.PeriodicSystem:
    from vibeqc.molecule import ANGSTROM_TO_BOHR

    active_angstrom = np.array([3.2, 0.55, 0.25])
    direction = active_angstrom / np.linalg.norm(active_angstrom)
    lattice_angstrom = np.column_stack(
        (
            active_angstrom,
            np.array([0.0, 20.0, 0.0]),
            np.array([0.0, 0.0, 20.0]),
        )
    )
    return vq.PeriodicSystem(
        1,
        lattice_angstrom * ANGSTROM_TO_BOHR,
        [
            vq.Atom(1, [0.0, 0.0, 0.0]),
            vq.Atom(
                9,
                (distance_ang * ANGSTROM_TO_BOHR * direction).tolist(),
            ),
        ],
    )


def test_periodic_seccm_neb_accepts_skew_active_lattice_column(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import vibeqc.neb as neb

    monkeypatch.setattr(neb, "_evaluate_image_semiempirical", _fake_image_evaluator)
    options = CCMOptions(translations=[[3.2, 0.55, 0.25]], madelung=True)
    result = vq.run_neb(
        _periodic_hf_skew_chain(1.55),
        _periodic_hf_skew_chain(1.65),
        method="seccm",
        ccm_options=options,
        n_images=1,
        interpolation="linear",
        max_iter=1,
        conv_tol_force=1.0e6,
        n_jobs=1,
    )

    assert result is not None
    assert result.method == "seccm"


def test_periodic_seccm_neb_requires_matching_hamiltonian_cell() -> None:
    options = CCMOptions(translations=[[3.2, 0.0, 0.0]], madelung=True)
    result = vq.run_neb(
        _periodic_hf_chain(1.55),
        _periodic_hf_chain(1.65),
        method="seccm",
        ccm_options=options,
        n_images=1,
        interpolation="linear",
        max_iter=1,
        conv_tol_force=1.0e6,
        n_jobs=1,
    )
    assert result is not None
    assert result.method == "seccm"
    assert result.is_periodic
    assert np.all(np.isfinite(result.energies))
    assert all(
        image.gradient is not None
        and np.all(np.isfinite(image.gradient))
        for image in result.path.images
    )

    bad_options = CCMOptions(translations=[[3.3, 0.0, 0.0]], madelung=True)
    with pytest.raises(ValueError, match="must match"):
        vq.run_neb(
            _periodic_hf_chain(1.55),
            _periodic_hf_chain(1.65),
            method="seccm",
            ccm_options=bad_options,
            n_images=1,
        )


def test_run_seccm_exposes_analytic_gradient() -> None:
    from vibeqc.molecule import ANGSTROM_TO_BOHR
    from vibeqc.semiempirical.methods.msindo_ccm_gradient_analytic import (
        ccm_gradient_analytic,
    )
    from vibeqc.semiempirical.runner import run_seccm

    molecule = vq.Molecule(
        [
            vq.Atom(1, [0.0, 0.0, 0.0]),
            vq.Atom(9, [1.6 * ANGSTROM_TO_BOHR, 0.0, 0.0]),
        ]
    )
    options = CCMOptions(translations=[[3.2, 0.0, 0.0]], madelung=True)
    result = run_seccm(molecule, options)
    expected = ccm_gradient_analytic(
        [1, 9],
        [[0.0, 0.0, 0.0], [1.6, 0.0, 0.0]],
        [[3.2, 0.0, 0.0]],
        madelung=True,
    )

    assert result.converged
    assert np.asarray(result.gradient()) == pytest.approx(expected, abs=1.0e-12)


def _archive_text(path) -> str:
    with zipfile.ZipFile(path) as archive:
        return "".join(
            archive.read(name).decode("utf-8", "replace")
            for name in archive.namelist()
        )


def test_semiempirical_neb_qvf_citations_are_route_specific(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import vibeqc.neb as neb

    monkeypatch.setattr(
        neb,
        "_evaluate_image_periodic_semiempirical",
        _fake_image_evaluator,
    )
    result = vq.run_neb(
        _h2_slab(1.4),
        _h2_slab(1.6),
        method="dftb0",
        n_images=1,
        interpolation="linear",
        max_iter=1,
        conv_tol_force=1.0e6,
        n_jobs=1,
    )
    assert result is not None
    blob = _archive_text(result.write_qvf(tmp_path / "dftb0_neb"))
    lower = blob.lower()
    assert "porezag_dftb0_1995" in blob
    assert "henkelman" in lower
    assert "perdew_pbe_1996" not in blob


def test_seccm_neb_qvf_cites_boundary_and_active_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import vibeqc.neb as neb

    monkeypatch.setattr(neb, "_evaluate_image_semiempirical", _fake_image_evaluator)
    result = vq.run_neb(
        _periodic_hf_chain(1.55),
        _periodic_hf_chain(1.65),
        method="seccm",
        ccm_options=CCMOptions(
            translations=[[3.2, 0.0, 0.0]],
            madelung=True,
        ),
        n_images=1,
        interpolation="linear",
        max_iter=1,
        conv_tol_force=1.0e6,
        n_jobs=1,
    )
    assert result is not None
    blob = _archive_text(result.write_qvf(tmp_path / "seccm_neb"))
    lower = blob.lower()
    assert "bredow_geudtner_jug_ccm_2001" in blob
    assert "peintinger_bredow_ccm_2014" in blob
    assert "ahlswede_jug_msindo_1_1999" in blob
    assert "ahlswede_jug_msindo_2_1999" in blob
    assert "libint" not in lower
    assert "hehre_sto_ng_1969" not in blob
    assert "pisani_crystal_1988" not in blob
    assert "perdew_pbe_1996" not in blob


def test_dftb0_seccm_neb_qvf_cites_boundary_and_adapter(tmp_path) -> None:
    reactant, topology = _dftb0_seccm_chain(1.3)
    product, _ = _dftb0_seccm_chain(1.5)
    result = vq.run_neb(
        reactant,
        product,
        method="dftb0",
        seccm_topology=topology,
        n_images=1,
        interpolation="linear",
        max_iter=1,
        conv_tol_force=1.0e6,
        n_jobs=1,
    )
    assert result is not None

    blob = _archive_text(result.write_qvf(tmp_path / "dftb0_seccm_neb"))
    assert "porezag_dftb0_1995" in blob
    assert "bredow_geudtner_jug_ccm_2001" in blob
    assert "peintinger_bredow_ccm_2014" in blob
    assert "ahlswede_jug_msindo_1_1999" not in blob
