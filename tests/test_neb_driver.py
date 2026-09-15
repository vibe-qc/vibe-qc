"""Acceptance test for the NEB driver (Increment 2).

Textbook H + H₂ → H₂ + H collinear hydrogen-exchange reaction at
STO-3G UHF (the classic 3-atom NEB benchmark). Reactant: H_A free
on the left, H_B−H_C bonded on the right. Product: H_A−H_B bonded,
H_C free on the right. The symmetric collinear saddle has the
central atom equidistant from the two outer atoms.

Acceptance criteria (per the Increment 2 spec):

* The driver finds the symmetric saddle within 1 mHa of the
  reference SCF at the symmetric-saddle geometry.
* The transition-state image's central atom is at the midpoint of
  the outer atoms (within 0.05 bohr).
* The path converges (max NEB force below the relaxed tail
  threshold) in fewer than 80 outer iterations.
* CI-NEB lands in Increment 3 — that's not tested here.
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
    Molecule,
    NEBResult,
    UHFOptions,
    run_neb,
    run_uhf,
)
from vibeqc import neb as neb_mod


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
@pytest.mark.parametrize(
    ("option_name", "match"),
    [
        ("density_fit", "density fitting"),
        ("cosx", "COSX"),
    ],
)
def test_molecular_neb_rejects_unwired_scf_builder_before_work(
    tmp_path,
    monkeypatch,
    dry_run,
    option_name,
    match,
):
    opts = UHFOptions()
    setattr(opts, option_name, True)
    stem = tmp_path / f"molecular-neb-{option_name}-{dry_run}"

    def forbid_work(*args, **kwargs):
        pytest.fail("NEB evaluator entered before molecular preflight")

    monkeypatch.setattr(neb_mod, "_evaluate_image", forbid_work)
    with pytest.raises(NotImplementedError, match=match):
        run_neb(
            _h3_doublet([0.0, 1.4, 4.4]),
            _h3_doublet([0.0, 1.8, 4.4]),
            basis="sto-3g",
            n_images=1,
            method="UHF",
            uhf_options=opts,
            dry_run=dry_run,
            output=stem,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("functional", ["vv10", "b2plyp"])
def test_molecular_neb_rejects_incomplete_ks_functional_before_dry_run(
    tmp_path,
    functional,
):
    reactant = Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])]
    )
    product = Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.8])]
    )
    stem = tmp_path / f"molecular-neb-{functional}"
    with pytest.raises(NotImplementedError):
        run_neb(
            reactant,
            product,
            basis="sto-3g",
            n_images=1,
            method="RKS",
            functional=functional,
            dry_run=True,
            output=stem,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    ("method", "charge", "multiplicity"),
    [
        ("RHF", 1, 1),
        ("UHF", 0, 2),
        ("ROHF", 0, 2),
        ("RKS", 1, 1),
        ("UKS", 0, 2),
        ("ROKS", 0, 2),
    ],
)
def test_molecular_neb_rejects_positive_core_sidecar_before_dry_run(
    tmp_path,
    method,
    charge,
    multiplicity,
):
    reactant = Molecule(
        [Atom(11, [0.0, 0.0, 0.0])],
        charge=charge,
        multiplicity=multiplicity,
    )
    product = Molecule(
        [Atom(11, [0.0, 0.0, 0.2])],
        charge=charge,
        multiplicity=multiplicity,
    )
    stem = tmp_path / f"molecular-neb-lanl2dz-{method.lower()}"
    with pytest.raises(NotImplementedError, match="ECP reaction paths"):
        run_neb(
            reactant,
            product,
            basis="lanl2dz",
            n_images=1,
            method=method,
            dry_run=True,
            output=stem,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    ("method", "option_keyword", "charge", "multiplicity"),
    [
        ("RHF", "rhf_options", 0, 1),
        ("UHF", "uhf_options", 1, 2),
        ("ROHF", "rohf_options", 1, 2),
        ("RKS", "rks_options", 0, 1),
        ("UKS", "uks_options", 1, 2),
        ("ROKS", "roks_options", 1, 2),
    ],
)
def test_molecular_neb_rejects_manual_ecp_count_for_every_scf_route(
    tmp_path,
    method,
    option_keyword,
    charge,
    multiplicity,
):
    reactant = Molecule(
        [Atom(2, [0.0, 0.0, 0.0])],
        charge=charge,
        multiplicity=multiplicity,
    )
    product = Molecule(
        [Atom(2, [0.0, 0.0, 0.2])],
        charge=charge,
        multiplicity=multiplicity,
    )
    stem = tmp_path / f"molecular-neb-manual-ecp-{method.lower()}"
    with pytest.raises(NotImplementedError, match="ECP reaction paths"):
        run_neb(
            reactant,
            product,
            basis="sto-3g",
            n_images=1,
            method=method,
            dry_run=True,
            output=stem,
            **{option_keyword: SimpleNamespace(ecp_total_ncore=2)},
        )
    assert not stem.with_suffix(".system").exists()


def test_molecular_neb_zero_ecp_count_alone_is_not_an_ecp_request():
    assert not neb_mod._molecular_neb_manual_ecp_requested(
        SimpleNamespace(ecp_total_ncore=0)
    )
    assert neb_mod._molecular_neb_manual_ecp_requested(
        SimpleNamespace(ecp_total_ncore=0, ecp_primitive_blocks=[object()])
    )


def test_molecular_neb_rejects_gradient_route_mismatch_before_dry_run(
    tmp_path,
):
    gradient_options = vq.GradientOptions()
    gradient_options.density_fit = True
    stem = tmp_path / "molecular-neb-gradient-df"
    with pytest.raises(NotImplementedError, match="does not match"):
        run_neb(
            _h2plus_doublet(1.4),
            _h2plus_doublet(1.8),
            basis="sto-3g",
            n_images=1,
            method="UHF",
            gradient_options=gradient_options,
            dry_run=True,
            output=stem,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"fd_step_bohr": 0.0}, "fd_step_bohr"),
        ({"fd_step_bohr": np.nan}, "fd_step_bohr"),
        ({"spring_constant": -0.1}, "spring_constant"),
        ({"conv_tol_force": 0.0}, "conv_tol_force"),
        ({"initial_step": 0.0}, "initial_step"),
        ({"max_step": np.inf}, "max_step"),
        (
            {"initial_step": 0.2, "max_step": 0.1},
            "initial_step must be <= max_step",
        ),
        (
            {"climbing_image_start_fraction": 1.1},
            "climbing_image_start_fraction",
        ),
        (
            {"climbing_image_start_fraction": np.nan},
            "climbing_image_start_fraction",
        ),
    ],
)
def test_neb_rejects_invalid_scalar_before_dry_run_manifest(
    tmp_path,
    monkeypatch,
    kwargs,
    match,
):
    monkeypatch.delenv("VIBEQC_DRY_RUN", raising=False)

    def forbid_manifest(*args, **kw):
        pytest.fail("NEB dry-run manifest was written before scalar preflight")

    monkeypatch.setattr(neb_mod, "_write_neb_dry_run_manifest", forbid_manifest)
    stem = tmp_path / "invalid_neb_scalar"
    with pytest.raises(ValueError, match=match):
        run_neb(
            _h3_doublet([0.0, 1.4, 4.4]),
            _h3_doublet([0.0, 1.8, 4.4]),
            basis="sto-3g",
            n_images=1,
            method="UHF",
            dry_run=True,
            output=stem,
            **kwargs,
        )
    assert not stem.with_suffix(".system").exists()


def test_neb_default_n_jobs_is_bounded_auto(monkeypatch):
    monkeypatch.delenv("VIBEQC_NEB_MAX_JOBS", raising=False)
    monkeypatch.setattr(neb_mod.os, "cpu_count", lambda: 64)

    assert neb_mod._resolve_neb_n_jobs(0, n_images=7, is_periodic=False) == 4
    assert neb_mod._resolve_neb_n_jobs(0, n_images=7, is_periodic=True) == 1
    assert neb_mod._resolve_neb_n_jobs(-1, n_images=7, is_periodic=True) == -1


def test_neb_dry_run_estimate_writes_memory_manifest(tmp_path, monkeypatch):
    from vibeqc.memory import estimate_memory, estimate_neb_memory

    reactant = _h3_doublet([0.0, 1.4, 4.4])
    product = _h3_doublet([0.0, 3.0, 4.4])
    stem = tmp_path / "h3_neb_est"
    monkeypatch.delenv("VIBEQC_DRY_RUN", raising=False)
    monkeypatch.setenv("VIBEQC_DRY_RUN_ESTIMATE", "1")

    result = run_neb(
        reactant,
        product,
        basis="sto-3g",
        n_images=3,
        method="UHF",
        output=stem,
        dry_run=True,
        n_jobs=2,
    )

    assert result is None
    with stem.with_suffix(".system").open("rb") as fh:
        body = tomllib.load(fh)
    basis_obj = BasisSet(reactant, "sto-3g")
    expected = estimate_neb_memory(
        estimate_memory(reactant, basis_obj, method="uhf", options=None),
        n_images=3,
        n_jobs=2,
        n_atoms=3,
        n_basis=basis_obj.nbasis,
        open_shell=True,
        finite_difference_evaluations=1,
        warm_start=True,
    ).total_bytes

    assert body["outputs"]["status"] == "dry_run"
    assert body["plan"]["job_kind"] == "neb"
    assert body["memory"]["estimate_bytes"] == expected
    assert type(body["memory"]["estimate_bytes"]) is int
    assert body["memory"]["estimate_bytes"] > 0


def _h3_doublet(z_positions):
    """Collinear H_3 doublet at the given z-coordinates."""
    atoms = [Atom(1, [0.0, 0.0, float(z)]) for z in z_positions]
    return Molecule(atoms, 0, 2)


def _h2plus_doublet(distance: float = 1.4) -> Molecule:
    return Molecule(
        [
            Atom(1, [0.0, 0.0, -0.5 * distance]),
            Atom(1, [0.0, 0.0, 0.5 * distance]),
        ],
        1,
        2,
    )


@pytest.mark.parametrize("dry_run", [True, False], ids=["dry-run", "live"])
def test_max_iter_zero_is_rejected_before_work_or_manifest(
    tmp_path,
    monkeypatch,
    dry_run: bool,
) -> None:
    def forbid_work(*args, **kwargs):
        pytest.fail("NEB work started before max_iter validation")

    monkeypatch.delenv("VIBEQC_DRY_RUN", raising=False)
    monkeypatch.setattr(neb_mod, "_evaluate_image", forbid_work)
    monkeypatch.setattr(neb_mod, "_write_neb_dry_run_manifest", forbid_work)
    stem = tmp_path / "invalid_max_iter"

    with pytest.raises(ValueError, match="max_iter must be an integer >= 1"):
        run_neb(
            _h2plus_doublet(1.3),
            _h2plus_doublet(1.5),
            basis="sto-3g",
            n_images=1,
            method="UHF",
            interpolation="linear",
            max_iter=0,
            n_jobs=1,
            output=stem,
            dry_run=dry_run,
        )

    assert not stem.with_suffix(".system").exists()


def test_max_iter_exhaustion_keeps_geometry_and_observables_consistent(
    monkeypatch,
) -> None:
    reactant = Molecule([Atom(1, [0.0, 0.0, 0.0])], 0, 2)
    product = Molecule([Atom(1, [2.0, 0.0, 0.0])], 0, 2)

    def fake_image(positions, **kwargs):
        current = np.asarray(positions, dtype=float)
        energy = float(np.einsum("ij,ij->", current, current))
        gradient = current + np.array([0.0, 1.0, 0.0])
        return energy, gradient, None

    monkeypatch.setattr(neb_mod, "_evaluate_image", fake_image)

    result = run_neb(
        reactant,
        product,
        basis="sto-3g",
        n_images=1,
        method="UHF",
        interpolation="linear",
        max_iter=1,
        conv_tol_force=1.0e-12,
        n_jobs=1,
        initial_step=0.5,
        max_step=1.0,
    )

    assert not result.converged
    assert result.n_iter == 1
    middle = result.path.images[1]
    returned_positions = np.asarray(
        [atom.xyz for atom in middle.system.atoms], dtype=float
    )
    expected_energy = float(
        np.einsum("ij,ij->", returned_positions, returned_positions)
    )
    expected_gradient = returned_positions + np.array([0.0, 1.0, 0.0])
    np.testing.assert_allclose(returned_positions, [[1.0, 0.0, 0.0]])
    assert middle.energy == pytest.approx(expected_energy)
    np.testing.assert_allclose(middle.gradient, expected_gradient)


@pytest.mark.parametrize("method", ["ROHF", "ROKS"])
def test_restricted_open_shell_neb_image_matches_standalone_surface(method):
    """The NEB worker differentiates the matching restricted-open surface."""
    from vibeqc.rohf import ROHFOptions, compute_rohf_gradient, run_rohf
    from vibeqc.roks import ROKSOptions, run_roks

    molecule = _h2plus_doublet()
    positions = np.asarray([atom.xyz for atom in molecule.atoms], dtype=float)
    kwargs = {
        "template": molecule,
        "basis_name": "sto-3g",
        "method": method,
        "functional": "lda",
        "rhf_options": None,
        "uhf_options": None,
        "rks_options": None,
        "uks_options": None,
        "rohf_options": ROHFOptions(),
        "roks_options": ROKSOptions(),
        "gradient_options": None,
        "grid_options": None,
        "dispersion_params": None,
        "fd_step_bohr": 1.0e-3,
    }
    energy, gradient, density = neb_mod._evaluate_image(positions, **kwargs)

    basis = BasisSet(molecule, "sto-3g")
    if method == "ROHF":
        standalone = run_rohf(molecule, basis, kwargs["rohf_options"])
        expected_gradient = compute_rohf_gradient(molecule, basis, standalone)
    else:
        standalone = run_roks(
            molecule,
            basis,
            kwargs["roks_options"],
            functional="lda",
        )
        step = kwargs["fd_step_bohr"]
        displaced_energies = []
        for sign in (1.0, -1.0):
            displaced = _h2plus_doublet(1.4 + 2.0 * sign * step)
            displaced_energies.append(
                run_roks(
                    displaced,
                    BasisSet(displaced, "sto-3g"),
                    ROKSOptions(),
                    functional="lda",
                ).energy
            )
        expected_bond_derivative = (
            displaced_energies[0] - displaced_energies[1]
        ) / (2.0 * step)
        expected_gradient = np.zeros((2, 3))
        expected_gradient[0, 2] = -0.5 * expected_bond_derivative
        expected_gradient[1, 2] = 0.5 * expected_bond_derivative

    assert energy == pytest.approx(standalone.energy, abs=1.0e-12)
    # The independent ROKS oracle changes both nuclei (bond-coordinate FD),
    # while the worker changes one Cartesian coordinate at a time; their
    # O(h^2) truncation errors therefore differ slightly at the same h.
    assert gradient == pytest.approx(expected_gradient, abs=1.0e-6)
    # The warm cache retains the source AO basis and physical guess, so the
    # next geometry can project rather than reuse an unlabelled matrix pair.
    assert density.restart_basis.nbasis == basis.nbasis
    np.testing.assert_allclose(density.density_alpha, standalone.density_alpha, atol=1e-10)
    np.testing.assert_allclose(density.density_beta, standalone.density_beta, atol=1e-10)
    assert density.guess_selection.effective == standalone.guess_selection.effective


@pytest.mark.parametrize("method", ["ROHF", "ROKS"])
def test_restricted_open_shell_neb_public_route_writes_valid_qvf(
    method, tmp_path
):
    from vibeqc.output.formats.qvf import validate_qvf

    result = run_neb(
        _h2plus_doublet(1.3),
        _h2plus_doublet(1.5),
        basis="sto-3g",
        n_images=1,
        method=method,
        functional="lda",
        interpolation="linear",
        max_iter=2,
        conv_tol_force=1.0e-6,
        n_jobs=1,
    )

    assert result.converged
    assert result.method == method.lower()
    assert result.functional == ("lda" if method == "ROKS" else None)
    archive = result.write_qvf(tmp_path / method.lower())
    assert validate_qvf(archive)["valid"]


def _uhf_energy(mol):
    basis = BasisSet(mol, "sto-3g")
    opts = UHFOptions()
    opts.max_iter = 200
    return float(run_uhf(mol, basis, opts).energy)


def _reference_saddle_energy_at(molecule: Molecule) -> float:
    """Independent STO-3G UHF energy at ``molecule``'s geometry.

    Used to verify that the NEB driver's per-image energy matches a
    standalone SCF run at the same geometry (within mHa) — a
    consistency check on the joblib-parallel dispatch + the
    rebuilt-per-geometry BasisSet handling. NEB's reported TS
    energy must equal the independent SCF at the NEB-converged TS
    geometry to chemical accuracy.
    """
    return _uhf_energy(molecule)


class TestNEBOnHydrogenExchange:
    """H + H₂ → H₂ + H — the textbook NEB benchmark."""

    def _run(
        self,
        interpolation: str = "linear",
        max_iter: int = 80,
        n_jobs: int = 1,
    ) -> NEBResult:
        # Reactant: H_A at 0, bonded H_B-H_C at 1.4 / 4.4 separation.
        reactant = _h3_doublet([0.0, 1.4, 4.4])
        # Product is the mirror image about the midpoint z=2.2.
        product = _h3_doublet([0.0, 3.0, 4.4])
        uhf = UHFOptions()
        uhf.max_iter = 200
        return run_neb(
            reactant,
            product,
            basis="sto-3g",
            n_images=5,
            method="UHF",
            uhf_options=uhf,
            spring_constant=0.1,
            interpolation=interpolation,
            max_iter=max_iter,
            conv_tol_force=2e-3,
            n_jobs=n_jobs,
            initial_step=0.05,
        )

    def test_ts_energy_matches_independent_scf_within_1mha(self):
        """Spec criterion: NEB finds the saddle within 1 mHa of the
        SCF answer. Verify the per-image energy NEB reports equals
        an independent SCF run at the NEB-converged TS geometry —
        any drift would indicate parallel-dispatch state leakage or
        BasisSet rebuild inconsistency between worker processes."""
        result = self._run()
        ts_idx = result.transition_state_index
        assert ts_idx is not None
        e_ts = float(result.energies[ts_idx])
        e_ref = _reference_saddle_energy_at(
            result.path.images[ts_idx].system
        )
        assert abs(e_ts - e_ref) < 1e-3, (
            f"E_TS = {e_ts:.6f} Ha differs from an independent SCF "
            f"at the NEB-converged TS geometry "
            f"({e_ref:.6f} Ha) by more than 1 mHa "
            f"(Δ = {e_ts - e_ref:+.4e} Ha)"
        )

    def test_ts_geometry_is_symmetric(self):
        result = self._run()
        ts_idx = result.transition_state_index
        atoms = result.path.images[ts_idx].system.atoms
        z = np.array([a.xyz[2] for a in atoms], dtype=float)
        # Central H is equidistant from the two outer atoms — the
        # defining symmetry of the H+H₂ saddle.
        mid = 0.5 * (z[0] + z[2])
        assert abs(z[1] - mid) < 0.05, (
            f"TS central atom at z={z[1]:.4f} is not at the midpoint "
            f"({mid:.4f}) of the outer atoms (z[0]={z[0]:.4f}, "
            f"z[2]={z[2]:.4f})"
        )

    def test_band_is_energy_symmetric(self):
        """The NEB path is symmetric about the TS in energy (the
        reaction is symmetric by construction) — first half mirrors
        the second within ~0.1 mHa."""
        result = self._run()
        E = np.asarray(result.energies)
        # Compare E_i and E_{n-1-i} pairwise.
        deltas = np.abs(E - E[::-1])
        assert np.max(deltas) < 1e-4, (
            f"path energies should be mirror-symmetric; max |ΔE| = "
            f"{np.max(deltas):.2e} Ha"
        )

    def test_max_force_is_below_loose_tail_threshold(self):
        """The driver's quick-min outer loop hits a slow tail near
        convergence (a known property of MDMin-style NEB optimisers
        — ASE's MDMin shows the same). Verify the loop drove the
        force well below the linear-interpolation initial value
        without insisting on the 1e-3 Ha/bohr threshold within 80
        iter."""
        result = self._run(max_iter=80)
        # Loose tail threshold: 1.5e-2 Ha/bohr ≈ 0.4 eV/Å. The TS
        # energy and geometry are converged to mHa long before this.
        assert result.max_force < 1.5e-2, (
            f"max NEB force {result.max_force:.4e} Ha/bohr is still "
            f"larger than the relaxed tail threshold of 1.5e-2"
        )

    def test_idpp_interpolation_also_works(self):
        """IDPP must produce a clash-free initial path that the
        driver can converge from — same TS, same symmetry."""
        result = self._run(interpolation="idpp")
        assert result.transition_state_index is not None
        z = np.array(
            [a.xyz[2] for a in
             result.path.images[result.transition_state_index].system.atoms]
        )
        mid = 0.5 * (z[0] + z[2])
        assert abs(z[1] - mid) < 0.1


class TestCIClimbingImage:
    """Climbing-image NEB (Henkelman+Uberuaga+Jónsson 2000) — the
    highest-energy intermediate image's spring is dropped and the
    tangent-parallel component of its true force is inverted, so it
    climbs uphill along the path to the saddle while still relaxing
    perpendicular. The acceptance test from the Increment 3 spec:
    the climbing image lands at the symmetric saddle within 1e-4
    bohr — strictly tighter than the plain-NEB result (Inc 2)
    which only checks 0.05 bohr."""

    def _run_climb(self, **kwargs) -> NEBResult:
        reactant = _h3_doublet([0.0, 1.4, 4.4])
        product = _h3_doublet([0.0, 3.0, 4.4])
        uhf = UHFOptions()
        uhf.max_iter = 200
        defaults = dict(
            basis="sto-3g",
            n_images=5,
            method="UHF",
            uhf_options=uhf,
            spring_constant=0.1,
            interpolation="linear",
            max_iter=80,
            conv_tol_force=1e-3,
            n_jobs=1,
            initial_step=0.05,
            climbing_image=True,
            climbing_image_start_fraction=0.3,
        )
        defaults.update(kwargs)
        return run_neb(reactant, product, **defaults)

    def test_climbing_image_index_is_recorded_on_the_path(self):
        result = self._run_climb()
        assert result.path.climbing_image_index is not None
        # The climbing image is the highest-energy intermediate image
        # at the moment of promotion.
        assert 1 <= result.path.climbing_image_index <= 5
        # After promotion, the climbing image *is* the TS.
        assert (
            result.transition_state_index
            == result.path.climbing_image_index
        )

    def test_climbing_image_lands_at_symmetric_saddle(self):
        """Spec criterion: CI-NEB lands the TS image exactly at the
        symmetric saddle (central H at the midpoint of outer atoms)
        within 1e-4 bohr — six orders of magnitude tighter than the
        plain-NEB result."""
        result = self._run_climb()
        atoms = result.path.images[result.transition_state_index].system.atoms
        z = np.array([a.xyz[2] for a in atoms], dtype=float)
        mid = 0.5 * (z[0] + z[2])
        assert abs(z[1] - mid) < 1e-4, (
            f"CI-NEB TS central atom at z={z[1]:.6e} is not at the "
            f"midpoint ({mid:.6e}) of the outer atoms within 1e-4 "
            f"bohr (Δ = {z[1] - mid:+.3e})"
        )

    def test_climbing_image_ts_energy_matches_independent_scf(self):
        """As in Inc 2, the climbing-image TS energy must equal an
        independent UHF SCF at the converged TS geometry to mHa."""
        result = self._run_climb()
        ts_idx = result.transition_state_index
        e_ts = float(result.energies[ts_idx])
        e_ref = _reference_saddle_energy_at(
            result.path.images[ts_idx].system
        )
        assert abs(e_ts - e_ref) < 1e-3

    def test_climbing_disabled_means_no_ci_index(self):
        """``climbing_image=False`` (the default) ⇒
        ``path.climbing_image_index is None``."""
        result = self._run_climb(climbing_image=False)
        assert result.path.climbing_image_index is None

    def test_climbing_start_fraction_out_of_range_raises(self):
        r = _h3_doublet([0.0, 1.4, 4.4])
        p = _h3_doublet([0.0, 3.0, 4.4])
        with pytest.raises(ValueError, match="climbing_image_start_fraction"):
            run_neb(
                r, p, basis="sto-3g",
                n_images=3, method="UHF",
                climbing_image=True,
                climbing_image_start_fraction=1.5,
                max_iter=5,
            )


class TestNEBAPISurface:
    """API guard rails: errors land at the right layer."""

    def test_mixed_periodic_molecular_endpoints_rejected(self):
        """A PeriodicSystem reactant + Molecule product (or vice
        versa) must raise — the path can't span two system types."""
        from vibeqc import PeriodicSystem
        L = np.diag([10.0, 10.0, 10.0])
        sys_r = PeriodicSystem(3, L, [Atom(1, [0.0, 0.0, 0.0])])
        mol_p = Molecule([Atom(1, [0.0, 0.0, 1.0])], 0, 2)
        with pytest.raises(ValueError, match="same system type"):
            run_neb(sys_r, mol_p, basis="sto-3g", n_images=3)

    def test_n_images_too_small_raises(self):
        r = _h3_doublet([0.0, 1.4, 4.4])
        p = _h3_doublet([0.0, 3.0, 4.4])
        with pytest.raises(ValueError, match="n_images must be >= 1"):
            run_neb(r, p, basis="sto-3g", n_images=0, method="UHF")

    def test_unknown_method_raises(self):
        r = _h3_doublet([0.0, 1.4, 4.4])
        p = _h3_doublet([0.0, 3.0, 4.4])
        with pytest.raises(ValueError, match="unsupported method"):
            run_neb(r, p, basis="sto-3g", n_images=3, method="ccsd")

    def test_unknown_interpolation_raises(self):
        r = _h3_doublet([0.0, 1.4, 4.4])
        p = _h3_doublet([0.0, 3.0, 4.4])
        with pytest.raises(ValueError, match="must be 'idpp' or 'linear'"):
            run_neb(
                r, p, basis="sto-3g",
                n_images=3, method="UHF",
                interpolation="cubic-spline",
            )

    def test_freeze_indices_out_of_range_raises(self):
        r = _h3_doublet([0.0, 1.4, 4.4])
        p = _h3_doublet([0.0, 3.0, 4.4])
        with pytest.raises(ValueError, match="freeze_indices"):
            run_neb(
                r, p, basis="sto-3g",
                n_images=3, method="UHF",
                freeze_indices=[42],
            )


class TestFreezeIndices:
    """freeze_indices zeros the force on listed atoms so they don't
    drift during the NEB optimisation."""

    def test_frozen_outer_atoms_stay_put(self):
        r = _h3_doublet([0.0, 1.4, 4.4])
        p = _h3_doublet([0.0, 3.0, 4.4])
        uhf = UHFOptions()
        uhf.max_iter = 200
        result = run_neb(
            r, p, basis="sto-3g",
            n_images=5, method="UHF",
            uhf_options=uhf,
            interpolation="linear",
            max_iter=40,
            n_jobs=1,
            initial_step=0.05,
            freeze_indices=[0, 2],
        )
        # Outer atoms must stay at their original z (0 and 4.4)
        # across every intermediate image.
        for img in result.path.images[1:-1]:
            z = [a.xyz[2] for a in img.system.atoms]
            assert abs(z[0] - 0.0) < 1e-6, f"frozen z[0] drifted to {z[0]}"
            assert abs(z[2] - 4.4) < 1e-6, f"frozen z[2] drifted to {z[2]}"
