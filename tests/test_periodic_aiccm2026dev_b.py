"""Independent finite-torus AICCM2026DEV_B invariants and dispatch tests."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import tomllib
import zipfile

import numpy as np
import pytest
import vibeqc as vq
from vibeqc._vibeqc_core import (
    aiccm2026dev_b_mayer_pair_matrix,
    aiccm2026dev_b_mayer_pair_tensor,
    aiccm2026dev_b_mp2_energy_from_lov,
)
from vibeqc.level_shift_schedule import LevelShiftSchedule
from vibeqc.pbc_bipole_fock import BipoleScreenedExchangeExecution
from vibeqc.periodic.chi.scf import (
    AICCM2026DevBBackend,
    AICCM2026DevBExactExchangeAssembly,
    AICCM2026DevBExperimentalWarning,
    AICCM2026DevBFiniteTorusConvention,
    _distributed_residue_inverse_bloch_transform,
    aiccm2026dev_b_charge_bookkeeping,
    cyclic_gamma_mesh,
    cyclic_lattice_extension,
    inverse_bloch_transform,
    pair_wigner_seitz_representatives,
    rhf_idempotency_error,
    run_aiccm2026dev_b_rhf,
    run_aiccm2026dev_b_rks,
    run_aiccm2026dev_b_uhf,
    run_aiccm2026dev_b_uks,
    wigner_seitz_representatives,
)
from vibeqc.periodic_jk_method import (
    PeriodicJKMethod,
    pick_jk_method,
    validate_jk_method,
)
from vibeqc.periodic.chi.properties import (
    _cell_residues,
    _finite_torus_full_matrix,
    _fold_mayer_bond_tensor,
    _fold_mayer_bonds,
    _mayer_atom_pair_matrix,
)
from vibeqc.periodic_k_gdf import PeriodicKRHFGDFResult
from vibeqc.periodic_runner import (
    _aiccm_b_qvf_supercell_system,
    _qvf_periodic_property_payload_supported,
)


def _h2_system(*, dim: int = 3, box: float = 12.0):
    lattice = np.diag([4.0 if dim == 1 else box, box, box])
    system = vq.PeriodicSystem(
        dim,
        lattice,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def test_exact_exchange_assembly_is_a_public_immutable_value() -> None:
    assembly = AICCM2026DevBExactExchangeAssembly(0.25, 0.0, 0.0)

    assert (
        vq.AICCM2026DevBExactExchangeAssembly
        is AICCM2026DevBExactExchangeAssembly
    )
    assert assembly.resolver == (
        "vibeqc.periodic_screened_exchange.resolve_periodic_exchange"
    )
    assert assembly.schema == (
        "vibeqc.aiccm2026dev-b.exact-exchange-assembly/v1"
    )
    assert assembly.screened_exchange_applicability == "inactive"
    assert assembly.screened_exchange_assembly == "not-applicable"
    with pytest.raises(AttributeError):
        assembly.c_full = 0.5


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ((float("nan"), 0.0, 0.0), "c_full must be finite"),
        ((0.0, float("inf"), 1.0), "c_sr must be finite"),
        ((0.0, 0.0, float("inf")), "omega_screen_bohr_inv must be finite"),
        ((0.0, 0.0, 0.11), "must be zero when c_sr is zero"),
        ((0.0, 0.25, 0.0), "must be positive when c_sr is nonzero"),
        ((0.0, 0.25, -0.11), "must be positive when c_sr is nonzero"),
        ((0.0, -0.25, 0.11), "c_sr must be nonnegative"),
        ((0.25, 0.25, 0.11), "does not support simultaneous"),
        ((10**1000, 0.0, 0.0), "c_full must be a finite float"),
        ((True, 0.0, 0.0), "c_full must be a finite float"),
    ],
)
def test_exact_exchange_assembly_rejects_inconsistent_values(
    values: tuple[float, float, float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        AICCM2026DevBExactExchangeAssembly(*values)


@pytest.mark.parametrize(
    ("applicability", "assembly", "message"),
    [
        ("active", "not-applicable", "active screened exchange must use"),
        ("inactive", "short-range-direct", "must use.*not-applicable"),
        ("sometimes", "not-applicable", "must be 'active' or 'inactive'"),
        ("active", "unknown", "active screened exchange must use"),
        ([], "not-applicable", "must be 'active' or 'inactive'"),
        ("active", [], "screened_exchange_assembly must be text"),
    ],
)
def test_exact_exchange_assembly_rejects_screened_branch_contradictions(
    applicability: object,
    assembly: object,
    message: str,
) -> None:
    c_sr = 0.25 if applicability == "active" else 0.0
    omega = 0.11 if c_sr else 0.0
    with pytest.raises(ValueError, match=message):
        AICCM2026DevBExactExchangeAssembly(
            c_full=0.0,
            c_sr=c_sr,
            omega_screen_bohr_inv=omega,
            screened_exchange_applicability=applicability,
            screened_exchange_assembly=assembly,
        )


def test_finite_torus_convention_rejects_unknown_q0_applicability() -> None:
    with pytest.raises(ValueError, match="must be 'active' or 'inactive'"):
        AICCM2026DevBFiniteTorusConvention(
            coulomb_kernel="3d-periodic-g0",
            exchange_q0="bvk-ewald",
            exchange_q0_applicability="sometimes",
            boundary_model="3d-periodic",
            periodic_dimension=3,
            character_mesh_shape=(1, 1, 1),
            bvk_madelung_supercell_repetitions=(1, 1, 1),
            bvk_madelung_supercell_lattice_bohr=(
                (8.0, 0.0, 0.0),
                (0.0, 8.0, 0.0),
                (0.0, 0.0, 8.0),
            ),
        )


def test_finite_torus_convention_requires_column_lattice_vectors() -> None:
    with pytest.raises(ValueError, match="must be 'columns'"):
        AICCM2026DevBFiniteTorusConvention(
            coulomb_kernel="3d-periodic-g0",
            exchange_q0="bvk-ewald",
            exchange_q0_applicability="active",
            boundary_model="3d-periodic",
            periodic_dimension=3,
            character_mesh_shape=(1, 1, 1),
            bvk_madelung_supercell_repetitions=(1, 1, 1),
            bvk_madelung_supercell_lattice_bohr=(
                (8.0, 0.0, 0.0),
                (0.0, 8.0, 0.0),
                (0.0, 0.0, 8.0),
            ),
            lattice_vector_convention="rows",
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("ccm_approach", "representation-label"),
        ("ccm_construction", "union-and-weight"),
        ("evaluation_representation", "real-gamma-supercell"),
    ),
)
def test_finite_torus_convention_rejects_wrong_construction_identity(
    field_name: str,
    invalid_value: str,
) -> None:
    fields = {
        "coulomb_kernel": "3d-periodic-g0",
        "exchange_q0": "bvk-ewald",
        "exchange_q0_applicability": "active",
        "boundary_model": "3d-periodic",
        "periodic_dimension": 3,
        "character_mesh_shape": (1, 1, 1),
        "bvk_madelung_supercell_repetitions": (1, 1, 1),
        "bvk_madelung_supercell_lattice_bohr": (
            (8.0, 0.0, 0.0),
            (0.0, 8.0, 0.0),
            (0.0, 0.0, 8.0),
        ),
        field_name: invalid_value,
    }
    with pytest.raises(ValueError, match=field_name):
        AICCM2026DevBFiniteTorusConvention(**fields)


def _stack_residue_blocks_for_test(
    blocks: dict[tuple[int, int, int], np.ndarray],
    mesh: tuple[int, int, int],
) -> np.ndarray:
    return np.ascontiguousarray(
        np.stack([blocks[residue] for residue in _cell_residues(mesh)]),
        dtype=float,
    )


def _expand_mayer_tensor_for_test(
    tensor: np.ndarray,
    mesh: tuple[int, int, int],
) -> np.ndarray:
    residues = _cell_residues(mesh)
    residue_index = {residue: index for index, residue in enumerate(residues)}
    n_cells, n_atoms, _ = tensor.shape
    out = np.zeros((n_cells * n_atoms, n_cells * n_atoms), dtype=float)
    for origin_index, origin in enumerate(residues):
        for target_index, target in enumerate(residues):
            delta = tuple(
                (int(target[axis]) - int(origin[axis])) % int(mesh[axis])
                for axis in range(3)
            )
            block = tensor[residue_index[delta]]
            row = slice(origin_index * n_atoms, (origin_index + 1) * n_atoms)
            col = slice(target_index * n_atoms, (target_index + 1) * n_atoms)
            out[row, col] = block
    return out


def test_selector_is_distinct_and_opt_in() -> None:
    method = pick_jk_method(
        "aiccm2026dev-b",
        lattice=np.eye(3) * 10.0,
        basis_name="sto-3g",
        n_atoms=2,
        scf_method="RHF",
    )
    assert method is PeriodicJKMethod.AICCM2026DEV_B
    assert method is not PeriodicJKMethod.GDF
    validate_jk_method(method, lattice=np.eye(3), basis_name="sto-3g")
    auto = pick_jk_method(
        "auto",
        lattice=np.eye(3) * 10.0,
        basis_name="sto-3g",
        n_atoms=2,
        scf_method="RHF",
    )
    assert auto is PeriodicJKMethod.GDF


@pytest.mark.parametrize(
    "exchange_exxdiv",
    (None, "none", "strict-zero-mode", "free"),
)
def test_runner_rejects_unimplemented_b_exchange_q0_convention(
    exchange_exxdiv: str | None,
    tmp_path: Path,
) -> None:
    system, basis = _h2_system()
    with pytest.raises(
        ValueError,
        match="fixes the finite-torus exchange_q0 convention to 'bvk-ewald'",
    ):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="aiccm2026dev-b",
            kpoints=(1, 1, 1),
            exchange_exxdiv=exchange_exxdiv,
            output=tmp_path / "b-q0-reject",
            output_qvf=False,
            progress=False,
        )


@pytest.mark.parametrize("write_population_file", [None, True])
def test_runner_multicell_population_sidecars_use_finite_torus_plan(
    write_population_file: bool | None,
    tmp_path: Path,
) -> None:
    """χ population output is a full-torus analysis, not a Gamma proxy."""

    system, basis = _h2_system()
    stem = tmp_path / f"b-population-{write_population_file}"

    vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        jk_method="aiccm2026dev-b",
        aiccm_backend="four_center",
        kpoints=(2, 2, 2),
        output=stem,
        write_molden_file=False,
        write_density=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=write_population_file,
        citations=False,
        output_qvf=False,
        dry_run=True,
        progress=False,
    )

    manifest = tomllib.loads(stem.with_suffix(".system").read_text())
    population_files = {
        (Path(row["path"]).name, row["format"])
        for row in manifest["plan"]["files"]
        if row["role"] == "population"
    }
    assert population_files == {
        (f"{stem.name}.population.txt", "aiccm2026dev-b-text"),
        (f"{stem.name}.population.json", "aiccm2026dev-b-json"),
    }


def test_runner_multik_population_rejection_names_periodic_routes(
    tmp_path: Path,
) -> None:
    """A route without a crystal population convention still fails closed."""

    system, basis = _h2_system()
    with pytest.raises(
        NotImplementedError,
        match=(
            "Full-k population output is implemented for BIPOLE.*"
            "aiccm2026dev-b"
        ),
    ):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="gdf",
            kpoints=(2, 2, 2),
            output=tmp_path / "gdf-population-reject",
            write_molden_file=False,
            write_xyz_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=True,
            citations=False,
            output_qvf=False,
            dry_run=True,
            progress=False,
        )


def test_qvf_generic_property_rebuild_excludes_chi_ccm() -> None:
    assert not _qvf_periodic_property_payload_supported(
        PeriodicJKMethod.AICCM2026DEV_B
    )
    assert _qvf_periodic_property_payload_supported(PeriodicJKMethod.GDF)
    assert not _qvf_periodic_property_payload_supported(
        PeriodicJKMethod.GDF,
        uses_external_xc=True,
    )


def test_runner_rejects_chi_coop_cohp_before_scf(
    monkeypatch,
    tmp_path: Path,
) -> None:
    system, basis = _h2_system()

    def unexpected_driver(*_args, **_kwargs):
        raise AssertionError("entered χ SCF for unsupported COOP/COHP")

    monkeypatch.setattr(
        "vibeqc.periodic_runner.run_aiccm2026dev_b_rhf",
        unexpected_driver,
    )

    with pytest.raises(
        NotImplementedError,
        match="COOP/COHP.*converged finite-character Hamiltonian",
    ):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="aiccm2026dev-b",
            coop_cohp=True,
            output_qvf=True,
            output=tmp_path / "b-coop-cohp-reject",
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

    assert not (tmp_path / "b-coop-cohp-reject.out").exists()


@pytest.mark.parametrize("temperature", [0.01, -0.01, float("nan")])
def test_gamma_only_ri_rhf_rejects_nonzero_smearing_before_fitted_scf(
    temperature: float,
    monkeypatch,
) -> None:
    """Gamma-only RI RHF must fail closed before fitted SCF for every
    nonzero smearing request (issue #206). A positive temperature would
    otherwise fall through to the legacy cutoff-selected Gamma GDF driver
    while the result still carried finite-torus metadata; negative and
    nonfinite temperatures must keep an explicit validation error."""
    from vibeqc.periodic_rhf_gdf import PeriodicRHFOptions

    system, basis = _h2_system()

    def unexpected_driver(*_args, **_kwargs):
        raise AssertionError(
            "entered fitted backend SCF for a nonzero-smearing Gamma-only "
            "RI request"
        )

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_krhf_periodic_gdf",
        unexpected_driver,
    )
    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_pbc_bipole_rhf",
        unexpected_driver,
    )

    options = PeriodicRHFOptions()
    options.smearing_temperature = temperature
    with pytest.raises(ValueError, match="smearing"):
        run_aiccm2026dev_b_rhf(
            system,
            basis,
            lattice_extension=(1, 1, 1),
            options=options,
            backend="ri",
        )


def test_zero_smearing_gamma_only_ri_rhf_still_reaches_fitted_dispatch(
    monkeypatch,
) -> None:
    """The zero-temperature Gamma-only RI selector contract is unchanged:
    smearing_temperature = 0 still dispatches to the fitted backend."""
    from vibeqc.periodic_rhf_gdf import PeriodicRHFOptions

    system, basis = _h2_system()

    def dispatch_sentinel(*_args, **_kwargs):
        raise RuntimeError("dispatch-reached")

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_krhf_periodic_gdf",
        dispatch_sentinel,
    )
    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_pbc_bipole_rhf",
        dispatch_sentinel,
    )

    options = PeriodicRHFOptions()
    options.smearing_temperature = 0.0
    with pytest.raises(RuntimeError, match="dispatch-reached"):
        run_aiccm2026dev_b_rhf(
            system,
            basis,
            lattice_extension=(1, 1, 1),
            options=options,
            backend="ri",
        )


def test_runner_qvf_preserves_native_chi_mayer_bond_orders(
    tmp_path: Path,
) -> None:
    system, basis = _h2_system()
    qvf_path = tmp_path / "b-native-mayer.qvf"

    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="aiccm2026dev-b",
            aiccm_backend="four_center",
            aiccm_lattice_extension=(1, 1, 1),
            output=tmp_path / "b-native-mayer",
            output_qvf=True,
            write_molden_file=False,
            write_density=False,
            density_spacing_bohr=2.0,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=True,
            citations=False,
            progress=False,
        )

    assert result.converged
    with zipfile.ZipFile(qvf_path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        sections = [
            section
            for section in manifest["sections"]
            if section["kind"] == "bond_orders"
        ]
        assert len(sections) == 1
        member = sections[0]["members"]["bond_orders"]
        payload = json.loads(zf.read(member["path"]))

    assert payload["method"] == "mayer"
    assert len(payload["pairs"]) == 1
    pair = payload["pairs"][0]
    assert {pair["i"], pair["j"]} == {0, 1}
    assert (pair["symbol_i"], pair["symbol_j"]) == ("H", "H")
    assert pair["order"] == pytest.approx(1.0, abs=1.0e-6)


def test_runner_qvf_mesh_one_density_grid_spans_cell(tmp_path: Path) -> None:
    """χ-CCM-B QVF emits a periodic cell grid and a Gamma GTO payload."""
    from vibeqc.output.formats.qvf import _BOHR_TO_ANGSTROM, validate_qvf

    L = 12.0
    system = vq.PeriodicSystem(
        3,
        np.diag([L, L, L]),
        [
            vq.Atom(1, [L / 2, L / 2, L / 2 - 0.7]),
            vq.Atom(1, [L / 2, L / 2, L / 2 + 0.7]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.warns(
        AICCM2026DevBExperimentalWarning
    ) as warning_records:
        result = vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="aiccm2026dev-b",
            aiccm_backend="four_center",
            aiccm_lattice_extension=(1, 1, 1),
            output=tmp_path / "b-qvf",
            output_qvf=True,
            write_molden_file=False,
            write_density=False,
            density_spacing_bohr=0.5,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            progress=False,
        )

    assert result.converged
    assert not any(
        "dos_computation" in str(record.message)
        for record in warning_records
    )
    out_text = (tmp_path / "b-qvf.out").read_text(encoding="utf-8")
    assert "CCM approach        = chi-ccm" in out_text
    assert (
        "CCM construction    = finite-translation-group-character"
        in out_text
    )
    assert "evaluation repr.    = gamma-centred-character-mesh" in out_text
    assert "q=0 applicability   = active" in out_text
    assert (
        "EXX assembly schema = "
        "vibeqc.aiccm2026dev-b.exact-exchange-assembly/v1"
    ) in out_text
    assert "screened EXX live   = inactive" in out_text
    assert "screened EXX build  = not-applicable" in out_text
    assert 'exchange_q0_applicability = "active"' in (
        tmp_path / "b-qvf.system"
    ).read_text(encoding="utf-8")
    system_manifest = tomllib.loads(
        (tmp_path / "b-qvf.system").read_text(encoding="utf-8")
    )
    assert system_manifest["run"]["exchange_q0"] == "BvK-ewald"
    assert system_manifest["run"]["method_status"] == "experimental"
    assert system_manifest["run"]["ccm_approach"] == "chi-ccm"
    assert (
        system_manifest["run"]["ccm_construction"]
        == "finite-translation-group-character"
    )
    assert (
        system_manifest["run"]["evaluation_representation"]
        == "gamma-centred-character-mesh"
    )
    assert system_manifest["run"]["exact_exchange_assembly_schema"] == (
        "vibeqc.aiccm2026dev-b.exact-exchange-assembly/v1"
    )
    assert system_manifest["run"]["exact_exchange_resolver"] == (
        "vibeqc.periodic_screened_exchange.resolve_periodic_exchange"
    )
    assert system_manifest["run"]["exact_exchange_c_full"] == pytest.approx(
        1.0
    )
    assert system_manifest["run"]["exact_exchange_c_sr"] == pytest.approx(0.0)
    assert system_manifest["run"][
        "exact_exchange_omega_screen_bohr_inv"
    ] == pytest.approx(0.0)
    assert (
        system_manifest["run"]["screened_exchange_applicability"]
        == "inactive"
    )
    assert (
        system_manifest["run"]["screened_exchange_assembly"]
        == "not-applicable"
    )
    qvf_path = tmp_path / "b-qvf.qvf"
    assert validate_qvf(qvf_path)["valid"] is True
    with zipfile.ZipFile(qvf_path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["provenance"]["charge"] == 0
        assert manifest["provenance"]["n_electrons"] == 2
        assert manifest["provenance"]["multiplicity"] == 1
        job_spec = json.loads(zf.read("job_spec/spec.json"))
        assert job_spec["charge"] == 0
        assert job_spec["multiplicity"] == 1
        kinds = [section["kind"] for section in manifest["sections"]]
        assert "volume.density" in kinds
        assert kinds.count("volume.orbital") >= 2
        assert "x_vibeqc.aiccm2026dev_b_convention" in kinds
        assert "wavefunction.gto" in kinds
        assert "x_vibeqc.bloch_wavefunction" not in kinds
        assert {
            "dos.total",
            "dos.projected",
            "dos.coop",
            "dos.cohp",
            "bond_orders",
        }.isdisjoint(kinds)
        assert manifest["extensions"]["x_vibeqc"]["version"] == "1.0"

        struct_sec = next(
            section for section in manifest["sections"]
            if section["kind"] == "structure"
        )
        struct = json.loads(
            zf.read(struct_sec["members"]["structure"]["path"])
        )
        assert struct["pbc"] == [True, True, True]
        assert struct["dimensionality"] == 3
        np.testing.assert_allclose(
            struct["lattice_vectors"],
            (np.asarray(system.lattice) * _BOHR_TO_ANGSTROM).T,
        )

        dens_sec = next(
            section for section in manifest["sections"]
            if section["kind"] == "volume.density"
        )
        grid = json.loads(zf.read(dens_sec["members"]["grid"]["path"]))
        voxel_vectors = np.asarray(grid["voxel_vectors"], dtype=float)
        shape = np.asarray(grid["shape"], dtype=float)
        np.testing.assert_allclose(
            voxel_vectors * shape[:, None],
            np.asarray(system.lattice, dtype=float).T,
        )
        data_member = dens_sec["members"]["data"]
        density = np.frombuffer(
            zf.read(data_member["path"]),
            dtype=np.dtype(data_member["dtype"]),
        ).reshape(tuple(data_member["shape"]))
        dvol = abs(float(np.linalg.det(voxel_vectors)))
        assert float(density.sum() * dvol) == pytest.approx(2.0, abs=1.0e-2)

        orbital_sections = [
            section for section in manifest["sections"]
            if section["kind"] == "volume.orbital"
        ]
        for orbital_sec in orbital_sections:
            assert orbital_sec["component"] == "real"
            assert "Gamma" in orbital_sec["label"]
            grid = json.loads(
                zf.read(orbital_sec["members"]["grid"]["path"])
            )
            orbital_voxels = np.asarray(grid["voxel_vectors"], dtype=float)
            orbital_shape = np.asarray(grid["shape"], dtype=float)
            np.testing.assert_allclose(
                orbital_voxels * orbital_shape[:, None],
                np.asarray(system.lattice, dtype=float).T,
            )
            member = orbital_sec["members"]["data"]
            orbital = np.frombuffer(
                zf.read(member["path"]),
                dtype=np.dtype(member["dtype"]),
            )
            assert np.all(np.isfinite(orbital))
            assert float(np.max(np.abs(orbital))) > 1.0e-8

        convention_sec = next(
            section for section in manifest["sections"]
            if section["kind"] == "x_vibeqc.aiccm2026dev_b_convention"
        )
        convention = json.loads(
            zf.read(convention_sec["members"]["convention"]["path"])
        )
        assert convention["method_selector"] == "aiccm2026dev-b"
        assert convention["schema_version"] == 2
        assert convention["ccm_approach"] == "chi-ccm"
        assert (
            convention["ccm_construction"]
            == "finite-translation-group-character"
        )
        assert (
            convention["evaluation_representation"]
            == "gamma-centred-character-mesh"
        )
        assert (
            convention["representation"]
            == "finite-character Gamma-centred character mesh"
        )
        assert convention["mesh"] == [1, 1, 1]
        finite = convention["finite_torus_convention"]
        assert finite["ccm_approach"] == "chi-ccm"
        assert finite["ccm_construction"] == "finite-translation-group-character"
        assert finite["evaluation_representation"] == "gamma-centred-character-mesh"
        assert finite["coulomb_kernel"] == "3d-periodic-g0"
        assert finite["exchange_q0"] == "bvk-ewald"
        assert finite["exchange_q0_applicability"] == "active"
        exact_exchange = convention["exact_exchange_assembly"]
        assert exact_exchange == {
            "c_full": 1.0,
            "c_sr": 0.0,
            "omega_screen_bohr_inv": 0.0,
            "screened_exchange_applicability": "inactive",
            "screened_exchange_assembly": "not-applicable",
            "schema": (
                "vibeqc.aiccm2026dev-b.exact-exchange-assembly/v1"
            ),
            "resolver": (
                "vibeqc.periodic_screened_exchange.resolve_periodic_exchange"
            ),
        }


def test_runner_qvf_mesh_two_density_grid_spans_bvk_supercell(
    tmp_path: Path,
) -> None:
    """χ-CCM-B mesh>1 QVF emits the full BvK torus plus Gamma GTO data."""
    from vibeqc.output.formats.qvf import _BOHR_TO_ANGSTROM, validate_qvf

    L = 12.0
    mesh = (2, 1, 1)
    primitive_lattice = np.diag([L, L, L])
    system = vq.PeriodicSystem(
        3,
        primitive_lattice,
        [
            vq.Atom(1, [L / 2, L / 2, L / 2 - 0.7]),
            vq.Atom(1, [L / 2, L / 2, L / 2 + 0.7]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="aiccm2026dev-b",
            aiccm_backend="four_center",
            aiccm_lattice_extension=mesh,
            output=tmp_path / "b-qvf-mesh2",
            output_qvf=True,
            write_molden_file=False,
            write_density=False,
            density_spacing_bohr=0.5,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            progress=False,
        )

    assert result.converged
    qvf_path = tmp_path / "b-qvf-mesh2.qvf"
    assert validate_qvf(qvf_path)["valid"] is True
    super_lattice = primitive_lattice * np.asarray(mesh, dtype=float)[None, :]
    with zipfile.ZipFile(qvf_path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["provenance"]["charge"] == 0
        assert manifest["provenance"]["n_electrons"] == 4
        assert manifest["provenance"]["multiplicity"] == 1
        job_spec = json.loads(zf.read("job_spec/spec.json"))
        assert job_spec["charge"] == 0
        assert job_spec["multiplicity"] == 1
        kinds = [section["kind"] for section in manifest["sections"]]
        assert "volume.density" in kinds
        assert kinds.count("volume.orbital") >= 2
        assert "x_vibeqc.aiccm2026dev_b_convention" in kinds
        assert "wavefunction.gto" in kinds
        assert "x_vibeqc.bloch_wavefunction" not in kinds
        assert manifest["extensions"]["x_vibeqc"]["version"] == "1.0"

        struct_sec = next(
            section for section in manifest["sections"]
            if section["kind"] == "structure"
        )
        struct = json.loads(
            zf.read(struct_sec["members"]["structure"]["path"])
        )
        assert len(struct["atoms"]) == 4
        assert struct["pbc"] == [True, True, True]
        assert struct["dimensionality"] == 3
        np.testing.assert_allclose(
            struct["lattice_vectors"],
            (super_lattice * _BOHR_TO_ANGSTROM).T,
        )

        dens_sec = next(
            section for section in manifest["sections"]
            if section["kind"] == "volume.density"
        )
        grid = json.loads(zf.read(dens_sec["members"]["grid"]["path"]))
        voxel_vectors = np.asarray(grid["voxel_vectors"], dtype=float)
        shape = np.asarray(grid["shape"], dtype=float)
        np.testing.assert_allclose(
            voxel_vectors * shape[:, None],
            super_lattice.T,
        )
        data_member = dens_sec["members"]["data"]
        density = np.frombuffer(
            zf.read(data_member["path"]),
            dtype=np.dtype(data_member["dtype"]),
        ).reshape(tuple(data_member["shape"]))
        dvol = abs(float(np.linalg.det(voxel_vectors)))
        assert float(density.sum() * dvol) == pytest.approx(4.0, abs=1.0e-2)

        orbital_sections = [
            section for section in manifest["sections"]
            if section["kind"] == "volume.orbital"
        ]
        for orbital_sec in orbital_sections:
            assert orbital_sec["component"] == "real"
            assert "Gamma" in orbital_sec["label"]
            grid = json.loads(
                zf.read(orbital_sec["members"]["grid"]["path"])
            )
            orbital_voxels = np.asarray(grid["voxel_vectors"], dtype=float)
            orbital_shape = np.asarray(grid["shape"], dtype=float)
            np.testing.assert_allclose(
                orbital_voxels * orbital_shape[:, None],
                super_lattice.T,
            )
            member = orbital_sec["members"]["data"]
            orbital = np.frombuffer(
                zf.read(member["path"]),
                dtype=np.dtype(member["dtype"]),
            )
            assert np.all(np.isfinite(orbital))
            assert float(np.max(np.abs(orbital))) > 1.0e-8

        convention_sec = next(
            section for section in manifest["sections"]
            if section["kind"] == "x_vibeqc.aiccm2026dev_b_convention"
        )
        convention = json.loads(
            zf.read(convention_sec["members"]["convention"]["path"])
        )
        assert convention["method_selector"] == "aiccm2026dev-b"
        assert convention["mesh"] == [2, 1, 1]
        finite = convention["finite_torus_convention"]
        assert finite["coulomb_kernel"] == "3d-periodic-g0"
        assert finite["exchange_q0"] == "bvk-ewald"
        assert finite["exchange_q0_applicability"] == "active"


@pytest.mark.slow
def test_runner_qvf_wannier_centers_overlay_for_vacuum_padded_chain(
    tmp_path: Path,
) -> None:
    """χ-CCM-B QVF can carry Wannier centres for a 3D chain-like torus."""
    from vibeqc.output.formats.qvf import _BOHR_TO_ANGSTROM, validate_qvf

    mesh = (4, 1, 1)
    system = vq.PeriodicSystem(
        3,
        np.diag([5.0, 20.0, 20.0]),
        [vq.Atom(1, [1.8, 10.0, 10.0]), vq.Atom(1, [3.2, 10.0, 10.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="aiccm2026dev-b",
            aiccm_backend="ri",
            aiccm_lattice_extension=mesh,
            output=tmp_path / "b-qvf-wannier",
            output_qvf=True,
            qvf_wannier_centers=True,
            write_molden_file=False,
            write_density=False,
            density_spacing_bohr=0.5,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            citations=False,
            max_iter=80,
            progress=False,
        )

    assert result.converged
    qvf_path = tmp_path / "b-qvf-wannier.qvf"
    assert validate_qvf(qvf_path)["valid"] is True
    with zipfile.ZipFile(qvf_path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        kinds = [section["kind"] for section in manifest["sections"]]
        assert "volume.density" in kinds
        assert "x_ccm.wannier_centers" in kinds
        assert manifest["extensions"]["x_ccm"]["version"] == "1.0"
        super_lattice = np.asarray(system.lattice, dtype=float) * np.asarray(
            mesh,
            dtype=float,
        )[None, :]

        struct_sec = next(
            section for section in manifest["sections"]
            if section["kind"] == "structure"
        )
        struct = json.loads(
            zf.read(struct_sec["members"]["structure"]["path"])
        )
        assert struct["pbc"] == [True, True, True]
        assert struct["dimensionality"] == 3
        np.testing.assert_allclose(
            struct["lattice_vectors"],
            (super_lattice * _BOHR_TO_ANGSTROM).T,
        )

        dens_sec = next(
            section for section in manifest["sections"]
            if section["kind"] == "volume.density"
        )
        grid = json.loads(zf.read(dens_sec["members"]["grid"]["path"]))
        voxel_vectors = np.asarray(grid["voxel_vectors"], dtype=float)
        shape = np.asarray(grid["shape"], dtype=float)
        np.testing.assert_allclose(
            voxel_vectors * shape[:, None],
            super_lattice.T,
        )
        data_member = dens_sec["members"]["data"]
        density = np.frombuffer(
            zf.read(data_member["path"]),
            dtype=np.dtype(data_member["dtype"]),
        ).reshape(tuple(data_member["shape"]))
        dvol = abs(float(np.linalg.det(voxel_vectors)))
        assert float(density.sum() * dvol) == pytest.approx(8.0, abs=0.1)

        wannier_sec = next(
            section for section in manifest["sections"]
            if section["kind"] == "x_ccm.wannier_centers"
        )
        centers = json.loads(
            zf.read(wannier_sec["members"]["centers"]["path"])
        )["centers"]
        assert len(centers) == 4
        np.testing.assert_allclose(
            np.sort([entry["center"][0] for entry in centers]),
            np.asarray([2.5, 7.5, 12.5, 17.5]) * _BOHR_TO_ANGSTROM,
            atol=1.0e-5,
        )
        for entry in centers:
            assert len(entry["center"]) == 3
            assert np.all(np.isfinite(entry["center"]))
            assert entry["spread"] > 0.0
            assert entry["label"].startswith("χ-CCM-B Wannier")


def test_runner_qvf_wannier_centers_overlay_for_3d_h2_pair(
    tmp_path: Path,
) -> None:
    """χ-CCM-B QVF overlay also works for the 3D periodic H2-pair control."""
    from vibeqc.output.formats.qvf import _BOHR_TO_ANGSTROM, validate_qvf

    mesh = (2, 1, 1)
    L = 12.0
    system = vq.PeriodicSystem(
        3,
        np.diag([L, L, L]),
        [
            vq.Atom(1, [L / 2, L / 2, L / 2 - 0.7]),
            vq.Atom(1, [L / 2, L / 2, L / 2 + 0.7]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="aiccm2026dev-b",
            aiccm_backend="ri",
            aiccm_lattice_extension=mesh,
            output=tmp_path / "b-qvf-h2pair-3d",
            output_qvf=True,
            qvf_wannier_centers=True,
            write_molden_file=False,
            write_density=False,
            density_spacing_bohr=0.5,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            citations=False,
            max_iter=80,
            progress=False,
        )

    assert result.converged
    qvf_path = tmp_path / "b-qvf-h2pair-3d.qvf"
    assert validate_qvf(qvf_path)["valid"] is True
    super_lattice = np.asarray(system.lattice, dtype=float) * np.asarray(
        mesh,
        dtype=float,
    )[None, :]
    with zipfile.ZipFile(qvf_path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        kinds = [section["kind"] for section in manifest["sections"]]
        assert "volume.density" in kinds
        assert kinds.count("volume.orbital") >= 2
        assert "x_ccm.wannier_centers" in kinds
        assert manifest["extensions"]["x_ccm"]["version"] == "1.0"

        struct_sec = next(
            section for section in manifest["sections"]
            if section["kind"] == "structure"
        )
        struct = json.loads(
            zf.read(struct_sec["members"]["structure"]["path"])
        )
        assert struct["pbc"] == [True, True, True]
        assert struct["dimensionality"] == 3
        np.testing.assert_allclose(
            struct["lattice_vectors"],
            (super_lattice * _BOHR_TO_ANGSTROM).T,
        )

        dens_sec = next(
            section for section in manifest["sections"]
            if section["kind"] == "volume.density"
        )
        grid = json.loads(zf.read(dens_sec["members"]["grid"]["path"]))
        voxel_vectors = np.asarray(grid["voxel_vectors"], dtype=float)
        shape = np.asarray(grid["shape"], dtype=float)
        np.testing.assert_allclose(
            voxel_vectors * shape[:, None],
            super_lattice.T,
        )
        data_member = dens_sec["members"]["data"]
        density = np.frombuffer(
            zf.read(data_member["path"]),
            dtype=np.dtype(data_member["dtype"]),
        ).reshape(tuple(data_member["shape"]))
        dvol = abs(float(np.linalg.det(voxel_vectors)))
        assert float(density.sum() * dvol) == pytest.approx(4.0, abs=1.0e-2)

        wannier_sec = next(
            section for section in manifest["sections"]
            if section["kind"] == "x_ccm.wannier_centers"
        )
        centers = json.loads(
            zf.read(wannier_sec["members"]["centers"]["path"])
        )["centers"]
        assert len(centers) == 2
        centers_arr = np.asarray([entry["center"] for entry in centers])
        centers_arr = centers_arr[np.argsort(centers_arr[:, 0])]
        np.testing.assert_allclose(
            centers_arr,
            np.asarray([[6.0, 6.0, 6.0], [18.0, 6.0, 6.0]])
            * _BOHR_TO_ANGSTROM,
            atol=1.0e-5,
        )


@pytest.mark.parametrize(
    ("n_hydrogen", "charge", "multiplicity", "expected_multiplicity"),
    [(2, 0, 1, 1), (2, 1, 2, 5), (3, 0, 2, 5)],
)
def test_qvf_supercell_scales_charge_and_spin_multiplicity(
    n_hydrogen: int,
    charge: int,
    multiplicity: int,
    expected_multiplicity: int,
) -> None:
    """The visual BvK system repeats charge and spin, not a singlet label.

    The charged case is a helper-level field-scaling control; it does not
    broaden the neutral χ SCF production envelope.
    """

    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 12.0,
        [
            vq.Atom(1, [4.6 + 1.4 * index, 6.0, 6.0])
            for index in range(n_hydrogen)
        ],
        charge=charge,
        multiplicity=multiplicity,
    )

    super_system = _aiccm_b_qvf_supercell_system(system, (2, 2, 1))

    assert len(super_system.unit_cell) == 4 * n_hydrogen
    assert super_system.charge == 4 * charge
    assert super_system.multiplicity == expected_multiplicity
    assert super_system.n_electrons() == 4 * system.n_electrons()


def test_runner_qvf_unrestricted_wannier_overlay_keeps_spin_density(
    tmp_path: Path,
) -> None:
    """Unrestricted QVF keeps spin density/centers plus additive GTO data."""
    from vibeqc.output.formats.qvf import _BOHR_TO_ANGSTROM, validate_qvf

    L = 15.0
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * L,
        [
            vq.Atom(1, [L / 2 - 1.4, L / 2, L / 2]),
            vq.Atom(1, [L / 2, L / 2, L / 2]),
            vq.Atom(1, [L / 2 + 1.4, L / 2, L / 2]),
        ],
        multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = vq.run_periodic_job(
            system,
            basis,
            method="UHF",
            jk_method="aiccm2026dev-b",
            aiccm_backend="ri",
            aiccm_lattice_extension=(1, 1, 1),
            output=tmp_path / "b-qvf-h3-uhf",
            output_qvf=True,
            qvf_wannier_centers=True,
            write_molden_file=False,
            write_density=False,
            density_spacing_bohr=0.75,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            citations=False,
            max_iter=80,
            progress=False,
        )

    assert result.converged
    diagnostics = result.aiccm2026dev_b
    assert diagnostics.n_alpha_error < 1.0e-10
    assert diagnostics.n_beta_error < 1.0e-10
    qvf_path = tmp_path / "b-qvf-h3-uhf.qvf"
    assert validate_qvf(qvf_path)["valid"] is True
    with zipfile.ZipFile(qvf_path) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["provenance"]["method"].lower() == "uhf"
        assert manifest["provenance"]["n_electrons"] == 3
        assert manifest["provenance"]["multiplicity"] == 2
        job_spec = json.loads(zf.read("job_spec/spec.json"))
        assert job_spec["charge"] == 0
        assert job_spec["multiplicity"] == 2
        kinds = [section["kind"] for section in manifest["sections"]]
        assert "volume.density" in kinds
        assert "volume.orbital" not in kinds
        assert "x_ccm.wannier_centers" in kinds
        assert "x_vibeqc.aiccm2026dev_b_convention" in kinds
        assert "wavefunction.gto" in kinds
        assert manifest["extensions"]["x_ccm"]["version"] == "1.0"

        struct_sec = next(
            section for section in manifest["sections"]
            if section["kind"] == "structure"
        )
        struct = json.loads(
            zf.read(struct_sec["members"]["structure"]["path"])
        )
        assert len(struct["atoms"]) == 3
        assert struct["pbc"] == [True, True, True]
        assert struct["dimensionality"] == 3
        np.testing.assert_allclose(
            struct["lattice_vectors"],
            (np.asarray(system.lattice) * _BOHR_TO_ANGSTROM).T,
        )

        dens_sec = next(
            section for section in manifest["sections"]
            if section["kind"] == "volume.density"
        )
        grid = json.loads(zf.read(dens_sec["members"]["grid"]["path"]))
        voxel_vectors = np.asarray(grid["voxel_vectors"], dtype=float)
        shape = np.asarray(grid["shape"], dtype=float)
        np.testing.assert_allclose(
            voxel_vectors * shape[:, None],
            np.asarray(system.lattice, dtype=float).T,
        )
        data_member = dens_sec["members"]["data"]
        density = np.frombuffer(
            zf.read(data_member["path"]),
            dtype=np.dtype(data_member["dtype"]),
        ).reshape(tuple(data_member["shape"]))
        dvol = abs(float(np.linalg.det(voxel_vectors)))
        assert float(density.sum() * dvol) == pytest.approx(3.0, abs=0.1)

        wannier_sec = next(
            section for section in manifest["sections"]
            if section["kind"] == "x_ccm.wannier_centers"
        )
        centers = json.loads(
            zf.read(wannier_sec["members"]["centers"]["path"])
        )["centers"]
        assert len(centers) == 3
        labels = [entry["label"] for entry in centers]
        assert sum(
            label.startswith("χ-CCM-B alpha Wannier")
            for label in labels
        ) == 2
        assert sum(
            label.startswith("χ-CCM-B beta Wannier")
            for label in labels
        ) == 1
        for entry in centers:
            assert len(entry["center"]) == 3
            assert np.all(np.isfinite(entry["center"]))
            assert entry["spread"] >= 0.0

        convention_sec = next(
            section for section in manifest["sections"]
            if section["kind"] == "x_vibeqc.aiccm2026dev_b_convention"
        )
        convention = json.loads(
            zf.read(convention_sec["members"]["convention"]["path"])
        )
        assert convention["method_selector"] == "aiccm2026dev-b"
        assert convention["electronic_method"].lower() == "uhf"
        assert convention["mesh"] == [1, 1, 1]


def test_qvf_wannier_centers_requires_qvf_output(tmp_path: Path) -> None:
    system, basis = _h2_system()
    with pytest.raises(ValueError, match="requires output_qvf=True"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="aiccm2026dev-b",
            aiccm_backend="ri",
            output=tmp_path / "no-qvf-centers",
            output_qvf=False,
            qvf_wannier_centers=True,
            write_molden_file=False,
            write_density=False,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            progress=False,
        )


def test_native_mayer_kernel_matches_full_supercell_reference() -> None:
    rng = np.random.default_rng(20260625)
    mesh = (2, 2, 1)
    nbf = 4
    ao_atoms = np.asarray([0, 0, 1, 2], dtype=np.int64)
    density_blocks = {
        residue: rng.normal(size=(nbf, nbf)) for residue in _cell_residues(mesh)
    }
    overlap_blocks = {
        residue: rng.normal(size=(nbf, nbf)) for residue in _cell_residues(mesh)
    }
    n_cells = int(np.prod(mesh))
    n_atoms = int(np.max(ao_atoms)) + 1
    ao_atoms_full = np.concatenate(
        [ao_atoms + cell * n_atoms for cell in range(n_cells)]
    )

    expected = _mayer_atom_pair_matrix(
        _finite_torus_full_matrix(density_blocks, mesh),
        _finite_torus_full_matrix(overlap_blocks, mesh),
        ao_atoms_full,
    )
    tensor = aiccm2026dev_b_mayer_pair_tensor(
        _stack_residue_blocks_for_test(density_blocks, mesh),
        _stack_residue_blocks_for_test(overlap_blocks, mesh),
        ao_atoms,
        np.asarray(mesh, dtype=np.int64),
    )
    got = aiccm2026dev_b_mayer_pair_matrix(
        _stack_residue_blocks_for_test(density_blocks, mesh),
        _stack_residue_blocks_for_test(overlap_blocks, mesh),
        ao_atoms,
        np.asarray(mesh, dtype=np.int64),
    )

    assert tensor.shape == (n_cells, n_atoms, n_atoms)
    np.testing.assert_allclose(
        _expand_mayer_tensor_for_test(tensor, mesh),
        expected,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(got, expected, atol=1.0e-12, rtol=1.0e-12)


def test_native_unrestricted_mayer_kernel_matches_full_supercell_reference() -> None:
    rng = np.random.default_rng(20260626)
    mesh = (3, 1, 1)
    nbf = 3
    ao_atoms = np.asarray([0, 1, 1], dtype=np.int64)
    alpha_blocks = {
        residue: rng.normal(size=(nbf, nbf)) for residue in _cell_residues(mesh)
    }
    beta_blocks = {
        residue: rng.normal(size=(nbf, nbf)) for residue in _cell_residues(mesh)
    }
    overlap_blocks = {
        residue: rng.normal(size=(nbf, nbf)) for residue in _cell_residues(mesh)
    }
    n_cells = int(np.prod(mesh))
    n_atoms = int(np.max(ao_atoms)) + 1
    ao_atoms_full = np.concatenate(
        [ao_atoms + cell * n_atoms for cell in range(n_cells)]
    )

    expected = _mayer_atom_pair_matrix(
        _finite_torus_full_matrix(alpha_blocks, mesh),
        _finite_torus_full_matrix(overlap_blocks, mesh),
        ao_atoms_full,
        density_beta=_finite_torus_full_matrix(beta_blocks, mesh),
    )
    tensor = aiccm2026dev_b_mayer_pair_tensor(
        _stack_residue_blocks_for_test(alpha_blocks, mesh),
        _stack_residue_blocks_for_test(overlap_blocks, mesh),
        ao_atoms,
        np.asarray(mesh, dtype=np.int64),
        density_beta_blocks=_stack_residue_blocks_for_test(beta_blocks, mesh),
    )
    got = aiccm2026dev_b_mayer_pair_matrix(
        _stack_residue_blocks_for_test(alpha_blocks, mesh),
        _stack_residue_blocks_for_test(overlap_blocks, mesh),
        ao_atoms,
        np.asarray(mesh, dtype=np.int64),
        density_beta_blocks=_stack_residue_blocks_for_test(beta_blocks, mesh),
    )

    assert tensor.shape == (n_cells, n_atoms, n_atoms)
    np.testing.assert_allclose(
        _expand_mayer_tensor_for_test(tensor, mesh),
        expected,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    np.testing.assert_allclose(got, expected, atol=1.0e-12, rtol=1.0e-12)


def test_compact_mayer_fold_matches_full_atom_matrix_fold() -> None:
    rng = np.random.default_rng(20260627)
    mesh = (2, 1, 1)
    system = vq.PeriodicSystem(
        3,
        np.diag([8.0, 9.0, 10.0]),
        [
            vq.Atom(1, [0.0, 0.0, 0.0]),
            vq.Atom(6, [1.5, 0.0, 0.0]),
            vq.Atom(8, [0.0, 1.3, 0.0]),
        ],
    )
    tensor = rng.normal(size=(int(np.prod(mesh)), len(system.unit_cell), 3))
    matrix = _expand_mayer_tensor_for_test(tensor, mesh)
    convention = SimpleNamespace(exchange_q0="bvk-ewald")

    compact = _fold_mayer_bond_tensor(
        tensor,
        system,
        mesh,
        finite_torus_convention=convention,
        threshold=-np.inf,
        max_bonds=None,
    )
    full = _fold_mayer_bonds(
        matrix,
        system,
        mesh,
        finite_torus_convention=convention,
        threshold=-np.inf,
        max_bonds=None,
    )

    assert len(compact.bonds) == len(full.bonds)
    for got_bond, expected_bond in zip(compact.bonds, full.bonds):
        assert got_bond.atom_i == expected_bond.atom_i
        assert got_bond.atom_j == expected_bond.atom_j
        assert got_bond.translation == expected_bond.translation
        assert got_bond.distance_bohr == pytest.approx(expected_bond.distance_bohr)
        assert got_bond.order == pytest.approx(expected_bond.order)
    assert compact.translational_spread == pytest.approx(full.translational_spread)


def test_direct_entry_point_warns_that_b_stream_is_experimental() -> None:
    system, basis = _h2_system()
    options = vq.PeriodicRHFOptions()
    options.max_iter = 1
    options.damping = 0.25
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_rhf(
            system,
            basis,
            (1, 1, 1),
            options,
            progress=False,
        )
    diagnostics = result.aiccm2026dev_b
    assert diagnostics.scf_trace_length == len(result.scf_trace)
    assert diagnostics.final_scf_grad_norm is not None
    assert diagnostics.scf_accelerator == options.scf_accelerator.name
    assert diagnostics.damping == pytest.approx(0.25)
    assert diagnostics.ccm_approach == "chi-ccm"
    assert diagnostics.ccm_construction == "finite-translation-group-character"
    assert diagnostics.evaluation_representation == "gamma-centred-character-mesh"
    assert diagnostics.coulomb_kernel == "3d-periodic-g0"
    assert diagnostics.exchange_q0 == "bvk-ewald"
    assert diagnostics.exchange_q0_applicability == "active"
    assert diagnostics.boundary_model == "3d-periodic"
    assert result.ccm_approach == "chi-ccm"
    assert result.ccm_construction == "finite-translation-group-character"
    assert result.evaluation_representation == "gamma-centred-character-mesh"
    assert result.exchange_q0 == "bvk-ewald"
    assert result.exchange_q0_applicability == "active"
    farming = result.output_cell_farming_execution
    assert farming.task_kind == "chi-direct-output-cell"
    assert farming.strategy == "cyclic"
    assert farming.world_size == 1
    assert farming.active is False
    assert farming.complete_internal_translation_sum is True
    assert diagnostics.direct_output_cell_farming == farming
    assert "include the declared exchange_q0 seam" in (
        result.finite_torus_convention.orbital_energy_convention
    )
    assert result.coulomb_kernel == "3d-periodic-g0"
    bonds = vq.aiccm2026dev_b_mayer_bond_orders(
        result,
        system,
        basis,
        threshold=0.0,
        max_bonds=1,
    )
    assert bonds.finite_torus_convention == diagnostics.finite_torus_convention
    assert bonds.finite_torus_convention.exchange_q0 == "bvk-ewald"


@pytest.mark.parametrize("functional", ["pbe", "pbe0"])
def test_rks_output_cell_farming_serial_matches_replicated_driver(
    functional: str,
) -> None:
    """The restricted KS scheduling seam changes no serial arithmetic."""

    system, basis = _h2_system(dim=3)
    mesh = (2, 1, 1)
    kmesh = cyclic_gamma_mesh(system, mesh).to_bloch_kmesh()

    reference_options = vq.PeriodicKSOptions()
    reference_options.functional = functional
    reference_options.max_iter = 1
    reference_options.lattice_opts.cutoff_bohr = 8.0
    reference_options.lattice_opts.nuclear_cutoff_bohr = 8.0
    reference = vq.run_pbc_bipole_rks(
        system,
        basis,
        kmesh,
        reference_options,
        functional=functional,
        fock_mixing=0.0,
        use_ewald_j_split=True,
        use_exchange_ewald_split=True,
        exchange_exxdiv="ewald",
        use_multipole_far_field=False,
        sr_image_precision=1.0e-6,
        farm_output_cells=False,
        progress=False,
    )

    chi_options = vq.PeriodicKSOptions()
    chi_options.functional = functional
    chi_options.max_iter = 1
    chi_options.lattice_opts.cutoff_bohr = 8.0
    chi_options.lattice_opts.nuclear_cutoff_bohr = 8.0
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        farmed = run_aiccm2026dev_b_rks(
            system,
            basis,
            functional,
            mesh,
            chi_options,
            backend="four_center",
            fock_mixing=0.0,
            progress=False,
        )

    assert farmed.energy == reference.energy
    assert farmed.e_electronic == reference.e_electronic
    assert farmed.e_xc == reference.e_xc
    for got, want in zip(farmed.fock, reference.fock):
        np.testing.assert_array_equal(got, want)
    for got, want in zip(farmed.density.blocks, reference.density.blocks):
        np.testing.assert_array_equal(got, want)
    executed = farmed.output_cell_farming_execution
    assert executed.task_kind == "chi-direct-output-cell"
    assert executed.active is False
    assert executed.complete_internal_translation_sum is True
    assert farmed.aiccm2026dev_b.direct_output_cell_farming == executed


def test_ecp_bookkeeping_records_effective_neutrality() -> None:
    system, _ = _h2_system()
    options = SimpleNamespace(
        ecp_total_ncore=1,
        ecp_effective_charges=[0.5, 0.5],
    )

    bookkeeping = aiccm2026dev_b_charge_bookkeeping(system, options)

    assert bookkeeping.physical_electrons == 2
    assert bookkeeping.effective_electrons == 1
    assert bookkeeping.ecp_total_ncore == 1
    assert bookkeeping.effective_nuclear_charges == pytest.approx((0.5, 0.5))
    assert bookkeeping.effective_nuclear_charge == pytest.approx(1.0)
    assert bookkeeping.effective_net_charge == pytest.approx(0.0)
    assert bookkeeping.has_ecp


def test_ecp_effective_charge_mismatch_fails_before_scf() -> None:
    system, basis = _h2_system()
    options = vq.PeriodicKSOptions()
    options.ecp_total_ncore = 1
    options.ecp_effective_charges = [0.75, 0.75]

    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(ValueError, match="effective nuclear charges"):
            run_aiccm2026dev_b_rks(
                system,
                basis,
                "pbe",
                (1, 1, 1),
                options,
                progress=False,
            )


def test_ecp_odd_effective_closed_shell_count_fails_before_scf() -> None:
    system, basis = _h2_system()
    options = vq.PeriodicKSOptions()
    options.ecp_total_ncore = 1
    options.ecp_effective_charges = [0.5, 0.5]

    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(ValueError, match="even effective electron count"):
            run_aiccm2026dev_b_rks(
                system,
                basis,
                "pbe",
                (1, 1, 1),
                options,
                progress=False,
            )


def test_even_one_dimensional_boundary_has_half_weights() -> None:
    system, _ = _h2_system(dim=1)
    representatives = wigner_seitz_representatives(system, (4,))
    boundary = [rep for rep in representatives if rep.residue == (2, 0, 0)]
    assert {rep.translation for rep in boundary} == {(-2, 0, 0), (2, 0, 0)}
    assert [rep.weight for rep in boundary] == pytest.approx([0.5, 0.5])

    totals: dict[tuple[int, int, int], float] = {}
    for representative in representatives:
        totals[representative.residue] = (
            totals.get(representative.residue, 0.0) + representative.weight
        )
    assert len(totals) == 4
    assert all(total == pytest.approx(1.0) for total in totals.values())


def test_wigner_seitz_uses_column_lattice_convention_for_skew_cell() -> None:
    lattice = np.array(
        [
            [4.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 20.0],
        ]
    )
    system = vq.PeriodicSystem(
        2,
        lattice,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    boundary = [
        representative
        for representative in wigner_seitz_representatives(system, (1, 2))
        if representative.residue == (0, 1, 0)
    ]
    assert {representative.translation for representative in boundary} == {
        (-1, 1, 0),
        (0, -1, 0),
        (0, 1, 0),
        (1, -1, 0),
    }
    assert {
        representative.displacement_bohr for representative in boundary
    } == {
        (-2.0, -3.0, 0.0),
        (-2.0, 3.0, 0.0),
        (2.0, -3.0, 0.0),
        (2.0, 3.0, 0.0),
    }
    assert [representative.weight for representative in boundary] == pytest.approx(
        [0.25, 0.25, 0.25, 0.25]
    )


def test_pair_offset_changes_the_minimum_image_without_changing_class_weight() -> None:
    system, _ = _h2_system(dim=1)
    offset = 0.75 * float(system.lattice[0, 0])
    representatives = pair_wigner_seitz_representatives(
        system,
        (4,),
        (0.0, 0.0, 0.0),
        (offset, 0.0, 0.0),
    )
    residue_two = [rep for rep in representatives if rep.residue == (2, 0, 0)]
    assert len(residue_two) == 1
    assert residue_two[0].translation == (-2, 0, 0)
    assert residue_two[0].weight == pytest.approx(1.0)


def _cyclic_count_result(system, control, counts):
    if control == "gamma_mesh":
        return cyclic_gamma_mesh(system, counts).mesh
    keyword = {"mesh": "mesh", "extension": "lattice_extension",
               "shells": "wigner_seitz_shells"}[control]
    return cyclic_lattice_extension(system, **{keyword: counts}).repetitions


@pytest.mark.parametrize("control", ["gamma_mesh", "mesh", "extension", "shells"])
@pytest.mark.parametrize("counts", [
    True, False, np.bool_(True), 2.9, 2.0, np.float64(2.0),
    "211", b"2", 2+0j, (2.9,), (2.0,), (-0.5,), (True,), (np.bool_(True),),
    ("2",), ((2,),), np.array([2.0]), np.array([True]),
    (float("nan"),), (float("inf"),), (2, 1.5, 1),
])
def test_cyclic_counts_reject_coercion_before_geometry(control, counts, monkeypatch):
    from vibeqc.periodic.chi import scf as chi
    def forbidden(*args, **kwargs):
        pytest.fail("invalid cyclic counts reached reciprocal mesh construction")
    monkeypatch.setattr(chi.KPoints, "gamma_centred", forbidden)
    # No lattice: invalid input must be rejected before supercell construction.
    with pytest.raises(ValueError, match="must contain integers"):
        _cyclic_count_result(SimpleNamespace(dim=1), control, counts)


@pytest.mark.parametrize("control", ["gamma_mesh", "mesh", "extension", "shells"])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("form", ["scalar", "numpy_scalar", "active", "full", "array"])
def test_cyclic_counts_preserve_exact_integer_conventions(control, dim, form):
    system = vq.PeriodicSystem(dim, np.eye(3)*8.0, [vq.Atom(2, [0., 0., 0.])])
    active = (2, 3, 2)[:dim]
    inactive = 0 if control == "shells" else 1
    if form in ("scalar", "numpy_scalar"):
        counts = 2 if form == "scalar" else np.int64(2)
        active = (2,)*dim
    elif form == "active":
        counts = tuple(np.int64(x) for x in active)
    elif form == "full":
        counts = active + (inactive,)*(3-dim)
    else:
        counts = np.array(active + (inactive,)*(3-dim), dtype=np.uint32)
    expected = tuple(2*x+1 if control == "shells" else x for x in active)
    expected += (1,)*(3-dim)
    assert _cyclic_count_result(system, control, counts) == expected


@pytest.mark.parametrize("control", ["gamma_mesh", "mesh", "extension", "shells"])
@pytest.mark.parametrize("counts", [(-1,), (), (1, 1, 1, 1), (2, 2, 1)])
def test_cyclic_counts_keep_range_length_and_inactive_axis_guards(control, counts):
    with pytest.raises(ValueError):
        _cyclic_count_result(SimpleNamespace(dim=1), control, counts)


def test_cyclic_counts_keep_default_zero_shells_and_exclusivity():
    system = vq.PeriodicSystem(2, np.eye(3)*8.0, [vq.Atom(2, [0., 0., 0.])])
    assert cyclic_lattice_extension(system).repetitions == (1, 1, 1)
    assert cyclic_lattice_extension(system, wigner_seitz_shells=0).repetitions == (1, 1, 1)
    assert cyclic_lattice_extension(system, wigner_seitz_shells=(0, 1)).repetitions == (1, 3, 1)
    for left, right in [("mesh", "lattice_extension"), ("mesh", "wigner_seitz_shells"),
                        ("lattice_extension", "wigner_seitz_shells")]:
        with pytest.raises(ValueError, match="choose exactly one"):
            cyclic_lattice_extension(system, **{left: 2, right: 2})


@pytest.mark.parametrize("method", ["rhf", "rks", "uhf", "uks"])
@pytest.mark.parametrize("control", ["mesh", "lattice_extension", "wigner_seitz_shells"])
@pytest.mark.parametrize("counts", [(2.9, 1, 1), True, "211"])
def test_cyclic_counts_fail_before_scf_backend(method, control, counts, monkeypatch):
    from vibeqc.periodic.chi import scf as chi
    def forbidden(*args, **kwargs):
        pytest.fail("invalid cyclic counts reached SCF backend selection")
    monkeypatch.setattr(chi, "_resolve_backend", forbidden)
    monkeypatch.setattr(chi, "_warn_experimental", lambda: None)
    kwargs = {control: counts}
    if method.endswith("ks"):
        kwargs["functional"] = "PBE"
    with pytest.raises(ValueError, match="must contain integers"):
        getattr(chi, f"run_aiccm2026dev_b_{method}")(
            SimpleNamespace(dim=3), None, **kwargs
        )


def test_cyclic_mesh_is_unreduced_and_gamma_centred() -> None:
    system, _ = _h2_system(dim=1)
    kpoints = cyclic_gamma_mesh(system, (4,))
    assert kpoints.mesh == (4, 1, 1)
    assert kpoints.shift == (0, 0, 0)
    assert len(kpoints.weights) == 4
    assert np.allclose(kpoints.weights, 0.25)
    assert np.any(np.all(np.isclose(kpoints.kpoints_frac, 0.0), axis=1))


def test_real_space_extension_is_primary_and_shell_radius_is_unambiguous() -> None:
    system, _ = _h2_system(dim=1)
    extension = cyclic_lattice_extension(system, wigner_seitz_shells=2)
    assert extension.repetitions == (5, 1, 1)
    assert extension.wigner_seitz_half_extent == (2.5, 0.5, 0.5)
    assert extension.n_cells == 5
    assert extension.supercell_lattice_bohr[0] == pytest.approx([20.0, 0.0, 0.0])
    assert cyclic_gamma_mesh(system, extension.repetitions).mesh == (5, 1, 1)

    with pytest.raises(ValueError, match="choose exactly one"):
        cyclic_lattice_extension(
            system,
            (5, 1, 1),
            wigner_seitz_shells=2,
        )


def test_real_space_extension_scales_skew_lattice_columns() -> None:
    lattice = np.asarray(
        [
            [7.0, 0.4, 0.2],
            [0.3, 8.0, 0.5],
            [0.1, 0.6, 9.0],
        ]
    )
    mesh = (2, 3, 1)
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )

    extension = cyclic_lattice_extension(system, mesh)

    expected = lattice @ np.diag(mesh)
    np.testing.assert_allclose(extension.supercell_lattice_bohr, expected)
    assert not np.allclose(expected, np.diag(mesh) @ lattice)


@pytest.mark.slow
def test_unrestricted_b_stream_closes_spin_and_projector_invariants(
    tmp_path: Path,
) -> None:
    lattice = np.eye(3) * 15.0
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(3, [0.0, 0.0, 0.0])],
        multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = vq.run_periodic_job(
            system,
            basis,
            method="UHF",
            jk_method="aiccm2026dev-b",
            aiccm_backend="ri",
            aiccm_lattice_extension=(1, 1, 1),
            output=tmp_path / "open-b",
            write_molden_file=False,
            write_density=False,
            write_xyz_file=False,
            progress=False,
        )
    diagnostics = result.aiccm2026dev_b
    assert result.converged
    assert diagnostics.variational_space.startswith("translation-invariant unrestricted")
    assert diagnostics.density_idempotency_error < 1e-10
    assert diagnostics.electron_count_error < 1e-10
    assert diagnostics.n_alpha_error < 1e-10
    assert diagnostics.n_beta_error < 1e-10
    assert diagnostics.s_squared == pytest.approx(0.75, abs=1e-10)


@pytest.mark.slow
# Only 3D four-center converges: the 1D/2D path is guarded with a
# NotImplementedError (missing neutral wire/slab Coulomb gauge) and its
# fail-closed behaviour is covered by
# test_open_shell_lower_dimensional_four_center_fails_closed.
@pytest.mark.parametrize("dimension", [3])
def test_unrestricted_four_center_dimensions_are_finite_and_idempotent(
    dimension: int,
) -> None:
    system = vq.PeriodicSystem(
        dimension,
        np.eye(3) * 15.0,
        [vq.Atom(3, [0.0, 0.0, 0.0])],
        multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_uhf(
            system,
            basis,
            (1, 1, 1),
            backend="four-center",
            progress=False,
        )
    assert result.converged
    assert np.isfinite(result.energy)
    assert result.aiccm2026dev_b.density_idempotency_error < 1e-10
    assert result.exchange_q0_applicability == "active"


@pytest.mark.slow
def test_unrestricted_ks_and_rijcosx_backends_are_wired() -> None:
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 15.0,
        [vq.Atom(3, [0.0, 0.0, 0.0])],
        multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        uks = vq.run_aiccm2026dev_b_uks(
            system,
            basis,
            "pbe0",
            (2, 1, 1),
            backend="rijcosx",
            progress=False,
        )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        rijcosx = run_aiccm2026dev_b_uhf(
            system,
            basis,
            (2, 1, 1),
            backend="rijcosx",
            progress=False,
        )
    assert uks.converged
    assert uks.aiccm2026dev_b.electronic_method.lower() == "uks/pbe0"
    assert uks.exchange_q0_applicability == "active"
    assert uks.backend == "aiccm2026dev-b-rijcosx"
    assert uks.aiccm2026dev_b.density_idempotency_error < 1e-10
    assert rijcosx.converged
    assert rijcosx.backend == "aiccm2026dev-b-rijcosx"
    assert rijcosx.aiccm2026dev_b.density_idempotency_error < 1e-10


def test_inverse_bloch_transform_two_cell_group() -> None:
    matrices = [np.array([[3.0]]), np.array([[1.0]])]
    kpoints = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    blocks = inverse_bloch_transform(
        matrices,
        kpoints,
        [(0, 0, 0), (1, 0, 0)],
    )
    assert blocks[:, 0, 0].real == pytest.approx([2.0, 1.0])
    assert np.max(np.abs(blocks.imag)) < 1e-15


def test_native_inverse_bloch_transform_matches_direct_formula() -> None:
    core = pytest.importorskip("vibeqc._vibeqc_core")
    if not hasattr(core, "aiccm2026dev_b_inverse_bloch_transform"):
        pytest.skip("editable extension lacks the native χ-CCM Bloch transform")

    rng = np.random.default_rng(20260626)
    matrices = rng.normal(size=(3, 2, 2)) + 1j * rng.normal(size=(3, 2, 2))
    kpoints = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 0.0, 0.0],
            [2.0 / 3.0, 0.0, 0.0],
        ]
    )
    translations = np.array([(0, 0, 0), (1, 0, 0), (2, 0, 0)], dtype=int)
    weights = np.full(3, 1.0 / 3.0)
    got = inverse_bloch_transform(matrices, kpoints, translations, weights)
    phases = np.exp(-2j * np.pi * (translations @ kpoints.T))
    expected = np.einsum("rk,k,kij->rij", phases, weights, matrices, optimize=True)
    assert np.allclose(got, expected, atol=1e-14)


@pytest.mark.parametrize(
    ("mesh", "expected_representatives", "expected_multiplicities"),
    [
        ((3,), 3, (1, 1, 1)),
        ((4,), 5, (1, 1, 2, 1)),
    ],
)
def test_residue_inverse_transform_groups_aliases_before_scheduling(
    mesh: tuple[int, ...],
    expected_representatives: int,
    expected_multiplicities: tuple[int, ...],
) -> None:
    system, _ = _h2_system(dim=1)
    kpoints = cyclic_gamma_mesh(system, mesh)
    matrices = [
        np.asarray([[2.0 + 0.25 * index]], dtype=np.complex128)
        for index in range(len(kpoints.weights))
    ]
    representatives = wigner_seitz_representatives(system, mesh)
    mesh_3d = tuple(mesh) + (1,) * (3 - len(mesh))

    blocks, execution = _distributed_residue_inverse_bloch_transform(
        matrices,
        kpoints.kpoints_frac,
        representatives,
        kpoints.weights,
        mesh=mesh_3d,
        representative_scope="zero-offset-cell",
    )
    unique_residues = tuple(sorted({rep.residue for rep in representatives}))
    expected = inverse_bloch_transform(
        matrices,
        kpoints.kpoints_frac,
        unique_residues,
        kpoints.weights,
    )

    np.testing.assert_array_equal(blocks, expected)
    assert execution.schema == (
        "vibeqc.aiccm2026dev-b.residue-inverse-transform-execution/v1"
    )
    assert execution.task_kind == "chi-residue-inverse-transform"
    assert execution.result_distribution == "all-ranks-canonical-residue-order"
    assert execution.active is False
    assert execution.world_size == 1
    assert execution.rank == 0
    assert execution.n_residues == len(kpoints.weights)
    assert execution.n_representatives == expected_representatives
    assert execution.character_mesh == mesh_3d
    assert execution.character_labels == unique_residues
    assert execution.residue_keys == unique_residues
    assert execution.local_residue_keys == unique_residues
    assert len(execution.ordered_residue_fingerprint) == 64
    assert len(execution.construction_fingerprint) == 64
    assert execution.representative_scope == "zero-offset-cell"
    assert execution.representative_multiplicities == expected_multiplicities
    assert execution.representative_weight_sums == pytest.approx(
        (1.0,) * len(unique_residues)
    )
    assert execution.complete_character_sum is True
    assert execution.supplied_representatives_grouped is True
    assert execution.extra_wigner_weight_applied is False
    with pytest.raises(ValueError, match="rank/world size"):
        replace(execution, world_size=0)


def test_residue_inverse_transform_keeps_four_way_skew_tie_on_one_task() -> None:
    lattice = np.array(
        [
            [4.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
            [0.0, 0.0, 20.0],
        ]
    )
    system = vq.PeriodicSystem(
        2,
        lattice,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    mesh = (1, 2)
    kpoints = cyclic_gamma_mesh(system, mesh)
    representatives = wigner_seitz_representatives(system, mesh)
    blocks, execution = _distributed_residue_inverse_bloch_transform(
        [np.asarray([[1.0]]), np.asarray([[0.5]])],
        kpoints.kpoints_frac,
        representatives,
        kpoints.weights,
        mesh=(1, 2, 1),
        representative_scope="zero-offset-cell",
    )

    assert blocks.shape == (2, 1, 1)
    assert execution.residue_keys == ((0, 0, 0), (0, 1, 0))
    assert execution.representative_multiplicities == (1, 4)
    assert execution.representative_weight_sums == pytest.approx((1.0, 1.0))
    assert execution.n_representatives == 5


def test_residue_inverse_transform_rejects_nonunit_alias_weights() -> None:
    system, _ = _h2_system(dim=1)
    mesh = (4,)
    kpoints = cyclic_gamma_mesh(system, mesh)
    representatives = list(wigner_seitz_representatives(system, mesh))
    boundary_index = next(
        index
        for index, representative in enumerate(representatives)
        if representative.residue == (2, 0, 0)
    )
    representatives[boundary_index] = replace(
        representatives[boundary_index],
        weight=0.4,
    )

    with pytest.raises(ValueError, match="weights to sum to one"):
        _distributed_residue_inverse_bloch_transform(
            [np.asarray([[float(index)]]) for index in range(4)],
            kpoints.kpoints_frac,
            representatives,
            kpoints.weights,
            mesh=(4, 1, 1),
            representative_scope="zero-offset-cell",
        )


def test_residue_inverse_transform_rejects_nondual_character_net() -> None:
    system, _ = _h2_system(dim=1)
    mesh = (4, 1, 1)
    kpoints = cyclic_gamma_mesh(system, mesh)
    kfrac = np.asarray(kpoints.kpoints_frac).copy()
    kfrac[1, 0] += 0.01

    with pytest.raises(ValueError, match="not on the declared dual"):
        _distributed_residue_inverse_bloch_transform(
            [np.asarray([[float(index)]]) for index in range(4)],
            kfrac,
            wigner_seitz_representatives(system, mesh),
            kpoints.weights,
            mesh=mesh,
            representative_scope="zero-offset-cell",
        )


def test_residue_inverse_transform_rejects_noncongruent_representative() -> None:
    system, _ = _h2_system(dim=1)
    mesh = (3, 1, 1)
    kpoints = cyclic_gamma_mesh(system, mesh)
    representatives = list(wigner_seitz_representatives(system, mesh))
    representatives[0] = replace(representatives[0], translation=(1, 0, 0))

    with pytest.raises(ValueError, match="not congruent"):
        _distributed_residue_inverse_bloch_transform(
            [np.asarray([[float(index)]]) for index in range(3)],
            kpoints.kpoints_frac,
            representatives,
            kpoints.weights,
            mesh=mesh,
            representative_scope="zero-offset-cell",
        )


def test_native_mp2_lov_kernel_matches_python_reference() -> None:
    rng = np.random.default_rng(20260624)
    mesh = (2, 3, 1)
    n_k = int(np.prod(mesh))
    n_aux = 3
    n_occ = 2
    n_vir = 2
    lov = {}
    for ki in range(n_k):
        for ka in range(n_k):
            real = rng.normal(scale=0.1, size=(n_aux, n_occ, n_vir))
            imag = rng.normal(scale=0.03, size=(n_aux, n_occ, n_vir))
            lov[(ki, ka)] = np.ascontiguousarray(real + 1j * imag)
    energies = np.asarray([
        [-0.70 + 0.02 * k, -0.45 + 0.01 * k,
         0.25 + 0.03 * k, 0.60 + 0.04 * k]
        for k in range(n_k)
    ])

    def _mesh_label(index: int) -> tuple[int, int, int]:
        m2 = index % mesh[2]
        rest = index // mesh[2]
        m1 = rest % mesh[1]
        return rest // mesh[1], m1, m2

    def _mesh_index(label: tuple[int, int, int]) -> int:
        return (label[0] * mesh[1] + label[1]) * mesh[2] + label[2]

    kconserv = np.empty((n_k, n_k, n_k), dtype=np.int64)
    for ki in range(n_k):
        for ka in range(n_k):
            for kj in range(n_k):
                mi = _mesh_label(ki)
                ma = _mesh_label(ka)
                mj = _mesh_label(kj)
                kconserv[ki, ka, kj] = _mesh_index(tuple(
                    (mi[axis] - ma[axis] + mj[axis]) % mesh[axis]
                    for axis in range(3)
                ))

    e_ss_ref = 0.0
    e_os_ref = 0.0
    max_imag_ref = 0.0
    oovv = [
        np.empty((n_occ, n_occ, n_vir, n_vir), dtype=complex)
        for _ in range(n_k)
    ]
    for ki in range(n_k):
        eps_i = energies[ki, :n_occ]
        for kj in range(n_k):
            eps_j = energies[kj, :n_occ]
            for ka in range(n_k):
                kb = int(kconserv[ki, ka, kj])
                oovv[ka] = (1.0 / n_k) * np.einsum(
                    "Pia,Pjb->ijab",
                    lov[(ki, ka)],
                    lov[(kj, kb)],
                    optimize=True,
                )
            for ka in range(n_k):
                kb = int(kconserv[ki, ka, kj])
                denominator = (
                    eps_i[:, None, None, None]
                    + eps_j[None, :, None, None]
                    - energies[ka, None, None, n_occ:, None]
                    - energies[kb, None, None, None, n_occ:]
                )
                amplitudes = np.conj(oovv[ka] / denominator)
                direct_complex = 2.0 * np.einsum(
                    "ijab,ijab->", amplitudes, oovv[ka], optimize=True
                )
                exchange_complex = -np.einsum(
                    "ijab,ijba->", amplitudes, oovv[kb], optimize=True
                )
                max_imag_ref = max(
                    max_imag_ref,
                    abs(float(np.imag(direct_complex))),
                    abs(float(np.imag(exchange_complex))),
                )
                direct = float(np.real(direct_complex))
                exchange = float(np.real(exchange_complex))
                e_ss_ref += 0.5 * direct + exchange
                e_os_ref += 0.5 * direct
    e_ss_ref /= n_k
    e_os_ref /= n_k

    for is_shift in ((0, 0, 0), (1, 1, 0)):
        e_ss, e_os, max_imag = aiccm2026dev_b_mp2_energy_from_lov(
            lov,
            np.ascontiguousarray(energies),
            mesh,
            is_shift,
        )
        assert float(e_ss) == pytest.approx(e_ss_ref, abs=1e-13)
        assert float(e_os) == pytest.approx(e_os_ref, abs=1e-13)
        assert float(max_imag) == pytest.approx(max_imag_ref, abs=1e-13)


def test_native_mp2_lov_kernel_rejects_mesh_before_factor_access() -> None:
    energies = np.asarray([[-0.5, 0.5]], dtype=float)

    # The project mesh is represented by three integers only. A mismatched
    # one-row energy input must fail before the empty LOV mapping is touched;
    # in particular, this path must not allocate the former 512^3 table.
    with pytest.raises(RuntimeError, match="contains 512 points.*energies has 1"):
        aiccm2026dev_b_mp2_energy_from_lov(
            {},
            energies,
            (8, 8, 8),
            (0, 0, 0),
        )

    with pytest.raises(RuntimeError, match="must be 0 or 1"):
        aiccm2026dev_b_mp2_energy_from_lov(
            {},
            energies,
            (1, 1, 1),
            (2, 0, 0),
        )


def test_rhf_idempotency_uses_nonorthogonal_metric() -> None:
    overlap = np.diag([2.0, 1.0])
    density = np.diag([1.0, 0.0])
    assert rhf_idempotency_error([density], [overlap]) < 1e-15


def _fake_result(
    kpoints,
    *,
    runtime_backend: str | None = None,
    screened_exchange_execution: BipoleScreenedExchangeExecution | None = None,
    exchange_ewald_split: bool = False,
    exchange_exxdiv: str | None = None,
) -> PeriodicKRHFGDFResult:
    n_k = len(kpoints.weights)
    overlap = [np.eye(2, dtype=complex) for _ in range(n_k)]
    density = [np.diag([2.0, 0.0]).astype(complex) for _ in range(n_k)]
    zeros = [np.zeros((2, 2), dtype=complex) for _ in range(n_k)]
    result = PeriodicKRHFGDFResult(
        energy=-1.0,
        e_electronic=-1.5,
        e_nuclear=0.5,
        n_iter=2,
        converged=True,
        mo_energies=[np.array([-0.5, 0.5]) for _ in range(n_k)],
        mo_coeffs=[np.eye(2, dtype=complex) for _ in range(n_k)],
        fock=zeros,
        overlap=overlap,
        hcore=zeros,
        density=density,
        kpoints_cart=np.asarray(kpoints.kpoints_cart),
        kpoint_weights=np.asarray(kpoints.weights),
    )
    result.backend = "native-multi-k-gdf-gdf-rhf"
    result.effective_n_electrons = 2
    result.kpoints_frac = np.asarray(kpoints.kpoints_frac)
    if runtime_backend is not None:
        result.runtime_backend = runtime_backend
    result.screened_exchange_execution = screened_exchange_execution
    result.exchange_ewald_split = exchange_ewald_split
    result.exchange_exxdiv = exchange_exxdiv
    return result


def _fake_unrestricted_result(
    kpoints,
    *,
    runtime_backend: str | None = None,
    screened_exchange_execution: BipoleScreenedExchangeExecution | None = None,
    exchange_ewald_split: bool = False,
    exchange_exxdiv: str | None = None,
) -> SimpleNamespace:
    n_k = len(kpoints.weights)
    overlap = [np.eye(2, dtype=complex) for _ in range(n_k)]
    density_alpha = [
        np.diag([1.0, 0.0]).astype(complex) for _ in range(n_k)
    ]
    density_beta = [np.zeros((2, 2), dtype=complex) for _ in range(n_k)]
    result = SimpleNamespace(
        energy=-0.5,
        e_electronic=-1.0,
        e_nuclear=0.5,
        n_iter=2,
        converged=True,
        s_squared=0.75,
        s_squared_ideal=0.75,
        overlap=overlap,
        density_alpha=density_alpha,
        density_beta=density_beta,
        scf_trace=[],
        kpoints_cart=np.asarray(kpoints.kpoints_cart),
        kpoint_weights=np.asarray(kpoints.weights),
        backend="native-multi-k-gdf-gdf-uhf",
        screened_exchange_execution=screened_exchange_execution,
        exchange_ewald_split=exchange_ewald_split,
        exchange_exxdiv=exchange_exxdiv,
    )
    if runtime_backend is not None:
        result.runtime_backend = runtime_backend
    return result


_FOCK_MIXING_UNSET = object()


def _open_shell_h_system():
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 12.0,
        [vq.Atom(1, [6.0, 6.0, 6.0])],
        charge=0,
        multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _run_fock_mixing_route(
    method: str,
    system,
    basis,
    options,
    backend: str,
    *,
    fock_mixing=_FOCK_MIXING_UNSET,
    **route_kwargs,
):
    kwargs = {
        "backend": backend,
        "progress": False,
        **route_kwargs,
    }
    if fock_mixing is not _FOCK_MIXING_UNSET:
        kwargs["fock_mixing"] = fock_mixing
    if method == "RHF":
        return run_aiccm2026dev_b_rhf(
            system,
            basis,
            (2, 1, 1),
            options,
            **kwargs,
        )
    if method == "RKS":
        return run_aiccm2026dev_b_rks(
            system,
            basis,
            "pbe",
            (2, 1, 1),
            options,
            **kwargs,
        )
    if method == "UHF":
        return run_aiccm2026dev_b_uhf(
            system,
            basis,
            (2, 1, 1),
            options,
            **kwargs,
        )
    if method == "UKS":
        return run_aiccm2026dev_b_uks(
            system,
            basis,
            "pbe",
            (2, 1, 1),
            options,
            **kwargs,
        )
    raise AssertionError(f"unexpected test method {method!r}")


@pytest.mark.parametrize(
    ("method", "backend_symbol"),
    [
        ("RHF", "run_pbc_bipole_rhf"),
        ("RKS", "run_pbc_bipole_rks"),
        ("UHF", "run_pbc_bipole_uhf"),
        ("UKS", "run_pbc_bipole_uks"),
    ],
)
def test_four_center_rejects_explicit_aux_basis_before_backend_dispatch(
    monkeypatch,
    method: str,
    backend_symbol: str,
) -> None:
    """Issue 385: a four-center Hamiltonian has no auxiliary basis.

    The selector used to accept ``aux_basis`` and silently drop it from every
    BIPOLE call.  An explicit control must fail before backend work rather than
    describe a calculation that was not executed.
    """
    if method in ("RHF", "RKS"):
        system, basis = _h2_system(dim=3)
    else:
        system, basis = _open_shell_h_system()
    options = (
        vq.PeriodicKSOptions()
        if method in ("RKS", "UKS")
        else vq.PeriodicRHFOptions()
    )
    called = False

    def fail_backend(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("inert four-center aux_basis reached BIPOLE")

    monkeypatch.setattr(
        f"vibeqc.periodic.chi.scf.{backend_symbol}",
        fail_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(
            ValueError,
            match="aux_basis cannot be used with backend='four_center'",
        ):
            _run_fock_mixing_route(
                method,
                system,
                basis,
                options,
                "four_center",
                aux_basis="def2-svp-jk",
            )

    assert called is False


@pytest.mark.parametrize(
    ("method", "backend_symbol"),
    [
        ("RHF", "run_krhf_periodic_gdf"),
        ("RKS", "run_krks_periodic_gdf"),
        ("UHF", "run_kuhf_periodic_gdf"),
        ("UKS", "run_kuks_periodic_gdf"),
    ],
)
@pytest.mark.parametrize(
    ("backend", "exchange_backend"),
    [("ri", "gdf"), ("rijcosx", "cosx")],
)
def test_fitted_routes_forward_explicit_aux_basis(
    monkeypatch,
    method: str,
    backend_symbol: str,
    backend: str,
    exchange_backend: str,
) -> None:
    """Issue 385 does not narrow either fitted auxiliary-basis contract."""
    if method in ("RHF", "RKS"):
        system, basis = _h2_system(dim=3)
    else:
        system, basis = _open_shell_h_system()
    options = (
        vq.PeriodicKSOptions()
        if method in ("RKS", "UKS")
        else vq.PeriodicRHFOptions()
    )
    captured: dict[str, object] = {}

    def fake_backend(_system, _basis, kpoints, _options, **kwargs):
        captured.update(kwargs)
        if method in ("RHF", "RKS"):
            result = _fake_result(kpoints)
        else:
            result = _fake_unrestricted_result(kpoints)
        result.backend = (
            f"native-multi-k-gdf-{exchange_backend}-{method.lower()}"
        )
        return result

    monkeypatch.setattr(
        f"vibeqc.periodic.chi.scf.{backend_symbol}",
        fake_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        _run_fock_mixing_route(
            method,
            system,
            basis,
            options,
            backend,
            aux_basis="def2-svp-jk",
        )

    assert captured["aux_basis"] == "def2-svp-jk"


def test_runner_four_center_rejects_explicit_aux_basis(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The unified selector inherits issue 385's pre-dispatch refusal."""
    system, basis = _h2_system(dim=3)
    called = False

    def fail_backend(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("public inert aux_basis request reached BIPOLE")

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_pbc_bipole_rhf",
        fail_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(
            ValueError,
            match="aux_basis cannot be used with backend='four_center'",
        ):
            vq.run_periodic_job(
                system,
                basis,
                method="RHF",
                jk_method="aiccm2026dev-b",
                aiccm_backend="four_center",
                aiccm_lattice_extension=(2, 1, 1),
                aux_basis="def2-svp-jk",
                output=tmp_path / "b-rhf-four-center-aux-reject",
                output_qvf=False,
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

    assert called is False


@pytest.mark.parametrize(
    ("method", "backend", "backend_symbol"),
    [
        ("RHF", "four_center", "run_pbc_bipole_rhf"),
        ("RHF", "ri", "run_krhf_periodic_gdf"),
        ("RHF", "rijcosx", "run_krhf_periodic_gdf"),
        ("RKS", "four_center", "run_pbc_bipole_rks"),
        ("RKS", "ri", "run_krks_periodic_gdf"),
        ("RKS", "rijcosx", "run_krks_periodic_gdf"),
        ("UHF", "four_center", "run_pbc_bipole_uhf"),
        ("UHF", "ri", "run_kuhf_periodic_gdf"),
        ("UHF", "rijcosx", "run_kuhf_periodic_gdf"),
        ("UKS", "four_center", "run_pbc_bipole_uks"),
        ("UKS", "ri", "run_kuks_periodic_gdf"),
        ("UKS", "rijcosx", "run_kuks_periodic_gdf"),
    ],
)
def test_quadratic_fallback_fails_closed_before_backend_dispatch(
    monkeypatch,
    method: str,
    backend: str,
    backend_symbol: str,
) -> None:
    if method in ("RHF", "RKS"):
        system, basis = _h2_system(dim=3)
    else:
        system, basis = _open_shell_h_system()
    options = (
        vq.PeriodicKSOptions()
        if method in ("RKS", "UKS")
        else vq.PeriodicRHFOptions()
    )
    options.quadratic_fallback_iter = 3
    options.quadratic_fallback_shift = 0.2
    options.quadratic_fallback_max_step = 0.05
    called = False

    def fail_backend(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("unsupported quadratic fallback reached backend")

    monkeypatch.setattr(
        f"vibeqc.periodic.chi.scf.{backend_symbol}",
        fail_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(
            NotImplementedError,
            match="quadratic SCF fallback is not implemented",
        ):
            _run_fock_mixing_route(
                method,
                system,
                basis,
                options,
                backend,
                fock_mixing=0.0,
            )

    assert called is False


def test_quadratic_fallback_iteration_requires_nonnegative_integer() -> None:
    system, basis = _h2_system(dim=3)
    options = vq.PeriodicRHFOptions()
    options.quadratic_fallback_iter = -1

    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(
            ValueError,
            match="quadratic_fallback_iter must be a non-negative integer",
        ):
            run_aiccm2026dev_b_rhf(
                system,
                basis,
                (2, 1, 1),
                options,
                backend="ri",
                progress=False,
            )


def test_inactive_quadratic_fallback_reaches_backend(monkeypatch) -> None:
    system, basis = _h2_system(dim=3)
    options = vq.PeriodicRHFOptions()
    options.quadratic_fallback_iter = 0
    options.quadratic_fallback_shift = 0.2
    options.quadratic_fallback_max_step = 0.05
    kpoints = cyclic_gamma_mesh(system, (2, 1, 1))
    called = False

    def fake_backend(_system, _basis, _kpoints, passed_options, **_kwargs):
        nonlocal called
        called = True
        assert passed_options is options
        assert passed_options.quadratic_fallback_iter == 0
        return _fake_result(kpoints)

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_krhf_periodic_gdf",
        fake_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_rhf(
            system,
            basis,
            (2, 1, 1),
            options,
            backend="ri",
            progress=False,
        )

    assert called is True
    assert result.backend == "aiccm2026dev-b-ri"


@pytest.mark.parametrize(
    ("method", "backend_symbol"),
    [
        ("RHF", "run_pbc_bipole_rhf"),
        ("RKS", "run_pbc_bipole_rks"),
        ("UHF", "run_pbc_bipole_uhf"),
        ("UKS", "run_pbc_bipole_uks"),
    ],
)
@pytest.mark.parametrize("physical_pairs", [False, True])
def test_four_center_records_executed_m5_domain_provenance(
    monkeypatch,
    method: str,
    backend_symbol: str,
    physical_pairs: bool,
) -> None:
    if method in ("RHF", "RKS"):
        system, basis = _h2_system(dim=3)
        options = (
            vq.PeriodicRHFOptions()
            if method == "RHF"
            else vq.PeriodicKSOptions()
        )
    else:
        system, basis = _open_shell_h_system()
        options = (
            vq.PeriodicRHFOptions()
            if method == "UHF"
            else vq.PeriodicKSOptions()
        )
    options.lattice_opts.pair_complete_1e = physical_pairs
    kpoints = cyclic_gamma_mesh(system, (2, 1, 1))
    captured = {}

    def fake_backend(*_args, **kwargs):
        assert _args[3] is options
        assert _args[3].lattice_opts.pair_complete_1e is physical_pairs
        captured["sr_image_precision"] = kwargs["sr_image_precision"]
        captured["use_multipole_far_field"] = kwargs[
            "use_multipole_far_field"
        ]
        captured["farm_output_cells"] = kwargs.get("farm_output_cells")
        captured["output_cell_farming_strategy"] = kwargs.get(
            "output_cell_farming_strategy"
        )
        captured["output_cell_farming_task_kind"] = kwargs.get(
            "output_cell_farming_task_kind"
        )
        result = (
            _fake_result(kpoints, runtime_backend="pbc-bipole")
            if method in ("RHF", "RKS")
            else _fake_unrestricted_result(
                kpoints,
                runtime_backend="pbc-bipole",
            )
        )
        result.sr_image_extent_bohr = 27.5
        return result

    monkeypatch.setattr(
        f"vibeqc.periodic.chi.scf.{backend_symbol}",
        fake_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = _run_fock_mixing_route(
            method,
            system,
            basis,
            options,
            "four_center",
        )

    assert captured["sr_image_precision"] == pytest.approx(1.0e-6)
    assert captured["use_multipole_far_field"] is False
    if method in ("RHF", "RKS"):
        assert captured["farm_output_cells"] is True
        assert captured["output_cell_farming_strategy"] == "cyclic"
        assert (
            captured["output_cell_farming_task_kind"]
            == "chi-direct-output-cell"
        )
    else:
        assert captured["farm_output_cells"] is None
        assert captured["output_cell_farming_strategy"] is None
        assert captured["output_cell_farming_task_kind"] is None
    assert result.backend == "aiccm2026dev-b-four_center"
    assert result.runtime_backend == "pbc-bipole"
    expected_policy = (
        "m5-physical-pair-midpoint-erfc/v1"
        if physical_pairs else "m5-qqr-padded-erfc/v1"
    )
    assert result.sr_image_domain_policy == expected_policy
    assert result.sr_image_precision == pytest.approx(1.0e-6)
    assembly = result.exact_exchange_assembly
    assert result.aiccm2026dev_b.exact_exchange_assembly is assembly
    expected_c_full = 1.0 if method in ("RHF", "UHF") else 0.0
    assert assembly.c_full == pytest.approx(expected_c_full)
    assert assembly.c_sr == pytest.approx(0.0)
    assert assembly.omega_screen_bohr_inv == pytest.approx(0.0)


def test_four_center_inherits_critical_fold_fail_close(monkeypatch) -> None:
    """B must not bypass the shared corrected-gauge support stop."""

    system, basis = _h2_system(dim=3)

    def entered_expensive_fock(*_args, **_kwargs):
        raise AssertionError("entered the expensive BIPOLE Fock build")

    monkeypatch.setattr(
        "vibeqc.pbc_bipole_common.s_fold_truncation_drift",
        lambda *_args, **_kwargs: 0.1,
    )
    monkeypatch.setattr(
        "vibeqc.pbc_bipole.build_bipole_restricted_fock",
        entered_expensive_fock,
    )

    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(
            RuntimeError,
            match=r"S\(k\).*1\.0e-01.*refusing to enter SCF",
        ):
            run_aiccm2026dev_b_rhf(
                system,
                basis,
                (2, 1, 1),
                backend="four_center",
                progress=False,
            )


def test_four_center_omits_domain_attestation_without_resolved_extent(
    monkeypatch,
) -> None:
    system, basis = _h2_system(dim=3)
    kpoints = cyclic_gamma_mesh(system, (2, 1, 1))

    def fake_backend(*_args, **_kwargs):
        return _fake_result(kpoints, runtime_backend="pbc-bipole")

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_pbc_bipole_rhf",
        fake_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_rhf(
            system,
            basis,
            (2, 1, 1),
            backend="four_center",
            progress=False,
        )

    assert result.runtime_backend == "pbc-bipole"
    assert not hasattr(result, "sr_image_domain_policy")
    assert not hasattr(result, "sr_image_precision")
    assert not hasattr(result, "aiccm_resolved_gdf_method")
    assert not hasattr(result, "aiccm_resolved_rsgdf_ke_cutoff")
    assert not hasattr(result, "aiccm_resolved_rsgdf_tail_ke_cutoff")
    assert not hasattr(result, "aiccm_resolved_mdf_ke_cutoff")


@pytest.mark.parametrize(
    ("backend", "low_level_backend", "k_exchange"),
    [
        ("ri", "native-multi-k-gdf-gdf-rhf", "gdf"),
        ("rijcosx", "native-multi-k-gdf-cosx-rhf", "cosx"),
    ],
)
def test_fitted_routes_preserve_distinct_low_level_runtime_backend(
    monkeypatch,
    backend: str,
    low_level_backend: str,
    k_exchange: str,
) -> None:
    system, basis = _h2_system(dim=3)

    def fake_gdf(_system, _basis, kpoints, _options, **kwargs):
        assert kwargs["k_exchange"] == k_exchange
        result = _fake_result(kpoints)
        result.backend = low_level_backend
        return result

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_krhf_periodic_gdf",
        fake_gdf,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_rhf(
            system,
            basis,
            (2, 1, 1),
            backend=backend,
            progress=False,
        )

    assert result.backend == f"aiccm2026dev-b-{backend}"
    assert result.runtime_backend == low_level_backend
    # IID 344: an unheld run POSITIVELY reports held=False as a structured
    # field -- absence of the flag is not a verdict.
    assert result.parity_held is False
    assert result.aiccm_resolved_gdf_method == "rsgdf"
    assert result.aiccm_resolved_rsgdf_ke_cutoff == pytest.approx(200.0)
    assert result.aiccm_resolved_rsgdf_tail_ke_cutoff is None
    assert result.aiccm_resolved_mdf_ke_cutoff == pytest.approx(40.0)


def test_fitted_route_reports_structured_parity_hold(monkeypatch) -> None:
    """IID 344: when the inner multi-k driver holds the absolute energy
    (``+PARITY_HELD`` on its backend), the aiccm2026dev-b result must carry
    the executed backend verbatim AND expose the structured ``parity_held``
    boolean -- a record consumer separates held rows on the field, not by
    substring conventions or by grepping the .err sidecar."""
    system, basis = _h2_system(dim=3)
    held_backend = "native-multi-k-gdf-gdf-rhf+PARITY_HELD"

    def fake_gdf(_system, _basis, kpoints, _options, **kwargs):
        result = _fake_result(kpoints)
        result.backend = held_backend
        return result

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_krhf_periodic_gdf",
        fake_gdf,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_rhf(
            system,
            basis,
            (2, 1, 1),
            backend="ri",
            progress=False,
        )

    assert result.runtime_backend == held_backend
    assert result.parity_held is True


def test_fitted_result_records_nondefault_forwarded_settings(monkeypatch) -> None:
    system, basis = _h2_system(dim=3)
    captured = {}

    def fake_gdf(_system, _basis, kpoints, _options, **kwargs):
        captured.update(kwargs)
        result = _fake_result(kpoints)
        result.backend = "native-multi-k-gdf-gdf-rhf"
        return result

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_krhf_periodic_gdf",
        fake_gdf,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_rhf(
            system,
            basis,
            (2, 1, 1),
            backend="ri",
            gdf_method="mdf",
            rsgdf_ke_cutoff=137.0,
            mdf_ke_cutoff=53.0,
            progress=False,
        )

    assert captured["gdf_method"] == "mdf"
    assert captured["rsgdf_ke_cutoff"] == pytest.approx(137.0)
    assert captured["rsgdf_tail_ke_cutoff"] is None
    assert captured["mdf_ke_cutoff"] == pytest.approx(53.0)
    assert result.aiccm_resolved_gdf_method == "mdf"
    assert result.aiccm_resolved_rsgdf_ke_cutoff == pytest.approx(137.0)
    assert result.aiccm_resolved_rsgdf_tail_ke_cutoff is None
    assert result.aiccm_resolved_mdf_ke_cutoff == pytest.approx(53.0)


@pytest.mark.parametrize("backend", ["ri", "rijcosx"])
@pytest.mark.parametrize("tail_ke_cutoff", [None, 420.0])
@pytest.mark.parametrize(
    ("method", "backend_symbol"),
    [
        ("RHF", "run_krhf_periodic_gdf"),
        ("RKS", "run_krks_periodic_gdf"),
        ("UHF", "run_kuhf_periodic_gdf"),
        ("UKS", "run_kuks_periodic_gdf"),
    ],
)
def test_fitted_routes_pin_character_mesh_gradient_and_tail_transport(
    monkeypatch,
    method: str,
    backend_symbol: str,
    backend: str,
    tail_ke_cutoff: float | None,
) -> None:
    """Pin character mesh, gradient, and exact fitted-tail transport."""

    if method in ("RHF", "RKS"):
        system, basis = _h2_system(dim=3)
        options = (
            vq.PeriodicRHFOptions()
            if method == "RHF"
            else vq.PeriodicKSOptions()
        )
    else:
        system, basis = _open_shell_h_system()
        options = (
            vq.PeriodicRHFOptions()
            if method == "UHF"
            else vq.PeriodicKSOptions()
        )
    kpoints = cyclic_gamma_mesh(system, (2, 1, 1))
    captured: dict[str, object] = {}

    def fake_backend(*_args, **kwargs):
        captured.update(kwargs)
        exchange_backend = "cosx" if backend == "rijcosx" else "gdf"
        low_level_backend = (
            f"native-multi-k-gdf-{exchange_backend}-{method.lower()}"
        )
        if method in ("RHF", "RKS"):
            result = _fake_result(kpoints)
        else:
            result = _fake_unrestricted_result(kpoints)
        result.backend = low_level_backend
        return result

    monkeypatch.setattr(
        f"vibeqc.periodic.chi.scf.{backend_symbol}",
        fake_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = _run_fock_mixing_route(
            method,
            system,
            basis,
            options,
            backend,
            fock_mixing=0.0,
            rsgdf_tail_ke_cutoff=tail_ke_cutoff,
        )

    assert captured["ibz_native"] is False
    assert captured["compute_gradient"] is False
    assert captured["rsgdf_tail_ke_cutoff"] == tail_ke_cutoff
    assert result.backend == f"aiccm2026dev-b-{backend}"
    assert result.aiccm_resolved_gdf_method == "rsgdf"
    assert result.aiccm_resolved_rsgdf_ke_cutoff == pytest.approx(200.0)
    assert result.aiccm_resolved_rsgdf_tail_ke_cutoff == tail_ke_cutoff
    assert result.aiccm_resolved_mdf_ke_cutoff == pytest.approx(40.0)


@pytest.mark.parametrize(
    ("backend", "gdf_method", "match"),
    [
        ("four_center", "rsgdf", "fitted RI and RIJCOSX"),
        ("ri", "mdf", "requires gdf_method='rsgdf'"),
    ],
)
def test_rsgdf_tail_rejects_non_rsgdf_b_routes_before_scf(
    monkeypatch,
    backend: str,
    gdf_method: str,
    match: str,
) -> None:
    system, basis = _h2_system(dim=3)

    def unexpected_backend(*_args, **_kwargs):
        raise AssertionError("entered SCF despite an unsupported χ tail")

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_pbc_bipole_rhf",
        unexpected_backend,
    )
    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_krhf_periodic_gdf",
        unexpected_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(NotImplementedError, match=match):
            run_aiccm2026dev_b_rhf(
                system,
                basis,
                (2, 1, 1),
                backend=backend,
                gdf_method=gdf_method,
                rsgdf_tail_ke_cutoff=420.0,
                progress=False,
            )


@pytest.mark.parametrize(
    ("base_ke_cutoff", "tail_ke_cutoff"),
    [
        (200.0, True),
        (200.0, np.nan),
        (200.0, 200.0),
        (200.0, 120.0),
        (-2.0, -1.0),
        (True, 2.0),
    ],
)
def test_rsgdf_tail_requires_finite_shell_above_base(
    monkeypatch,
    base_ke_cutoff,
    tail_ke_cutoff,
) -> None:
    system, basis = _h2_system(dim=3)
    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_krhf_periodic_gdf",
        lambda *_args, **_kwargs: pytest.fail("entered SCF with invalid tail"),
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(ValueError, match="strictly greater"):
            run_aiccm2026dev_b_rhf(
                system,
                basis,
                (2, 1, 1),
                backend="ri",
                rsgdf_ke_cutoff=base_ke_cutoff,
                rsgdf_tail_ke_cutoff=tail_ke_cutoff,
                progress=False,
            )


def test_gamma_dense_core_rsgdf_tail_must_be_explicit_before_scf(
    monkeypatch,
) -> None:
    conventional_a = 4.211 * 1.8897259886
    primitive = 0.5 * conventional_a * np.array(
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
        dtype=float,
    )
    oxygen = 0.5 * (primitive[0] + primitive[1] + primitive[2])
    system = vq.PeriodicSystem(
        3,
        primitive,
        [
            vq.Atom(12, [0.0, 0.0, 0.0]),
            vq.Atom(8, oxygen.tolist()),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    captured: dict[str, object] = {}

    def fake_gdf(_system, _basis, kpoints, _options, **kwargs):
        captured.update(kwargs)
        result = _fake_result(kpoints)
        result.backend = "native-gamma-gdf-via-k-gdf"
        return result

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_krhf_periodic_gdf",
        fake_gdf,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(NotImplementedError, match="auto-resolve.*explicit"):
            run_aiccm2026dev_b_rhf(
                system,
                basis,
                (1, 1, 1),
                backend="ri",
                progress=False,
            )
    assert captured == {}

    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_rhf(
            system,
            basis,
            (1, 1, 1),
            backend="ri",
            rsgdf_tail_ke_cutoff=4000.0,
            progress=False,
        )
    assert captured["rsgdf_tail_ke_cutoff"] == pytest.approx(4000.0)
    assert result.aiccm_resolved_rsgdf_tail_ke_cutoff == pytest.approx(4000.0)

    captured.clear()
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_rhf(
            system,
            basis,
            (1, 1, 1),
            backend="ri",
            rsgdf_ke_cutoff=4000.0,
            progress=False,
        )
    assert captured["rsgdf_tail_ke_cutoff"] is None
    assert result.aiccm_resolved_rsgdf_tail_ke_cutoff is None


def test_gamma_ri_rhf_uses_converged_3d_ewald_nuclear_repulsion(
    monkeypatch,
) -> None:
    """The fitted Gamma limit must not revive the truncated Ewald sum."""

    class ExpectedConvergedEwald(Exception):
        pass

    def converged_ewald(*_args, **_kwargs):
        raise ExpectedConvergedEwald

    def truncated_ewald(*_args, **_kwargs):
        raise AssertionError("Gamma RI used nuclear_repulsion_per_cell")

    monkeypatch.setattr(
        "vibeqc.pbc_gdf.ewald_nuclear_repulsion",
        converged_ewald,
    )
    monkeypatch.setattr(
        "vibeqc.pbc_gdf.nuclear_repulsion_per_cell",
        truncated_ewald,
    )
    system, basis = _h2_system(dim=3)

    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(ExpectedConvergedEwald):
            run_aiccm2026dev_b_rhf(
                system,
                basis,
                (1, 1, 1),
                backend="ri",
                fock_mixing=0.0,
                progress=False,
            )


@pytest.mark.parametrize("backend", ["ri", "rijcosx"])
@pytest.mark.parametrize("method", ["RHF", "RKS", "UHF", "UKS"])
def test_fitted_multik_routes_use_converged_3d_ewald_nuclear_repulsion(
    monkeypatch,
    method: str,
    backend: str,
) -> None:
    """Restricted and open-shell fitted routes share one 3D Ewald helper."""

    class ExpectedConvergedEwald(Exception):
        pass

    def converged_ewald(*_args, **_kwargs):
        raise ExpectedConvergedEwald

    def truncated_ewald(*_args, **_kwargs):
        raise AssertionError("multi-k fitted route used truncated Ewald e_nuc")

    monkeypatch.setattr(
        "vibeqc.periodic_k_gdf.ewald_nuclear_repulsion",
        converged_ewald,
    )
    monkeypatch.setattr(
        "vibeqc.periodic_k_gdf.nuclear_repulsion_per_cell",
        truncated_ewald,
    )
    if method in ("RHF", "RKS"):
        system, basis = _h2_system(dim=3)
        options = (
            vq.PeriodicRHFOptions()
            if method == "RHF"
            else vq.PeriodicKSOptions()
        )
    else:
        system, basis = _open_shell_h_system()
        options = (
            vq.PeriodicRHFOptions()
            if method == "UHF"
            else vq.PeriodicKSOptions()
        )

    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(ExpectedConvergedEwald):
            _run_fock_mixing_route(
                method,
                system,
                basis,
                options,
                backend,
                fock_mixing=0.0,
            )


@pytest.mark.parametrize("method", ["RHF", "UHF"])
def test_compact_dense_core_mdf_fails_closed_through_chi_selector(
    method: str,
) -> None:
    """χ-B propagates the shared compact-cell MDF safety boundary."""

    angstrom_to_bohr = 1.8897259886
    conventional_a = 4.211 * angstrom_to_bohr
    primitive = 0.5 * conventional_a * np.array(
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]],
        dtype=float,
    )
    oxygen = 0.5 * (primitive[0] + primitive[1] + primitive[2])
    system = vq.PeriodicSystem(
        3,
        primitive,
        [
            vq.Atom(12, [0.0, 0.0, 0.0]),
            vq.Atom(8, oxygen.tolist()),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = vq.PeriodicRHFOptions()
    options.max_iter = 2
    options.lattice_opts.cutoff_bohr = 18.0
    options.lattice_opts.nuclear_cutoff_bohr = 18.0

    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(NotImplementedError, match="dense-core"):
            if method == "RHF":
                run_aiccm2026dev_b_rhf(
                    system,
                    basis,
                    (2, 1, 1),
                    options,
                    backend="ri",
                    gdf_method="mdf",
                    aux_basis="def2-svp-jk",
                    progress=False,
                )
            else:
                run_aiccm2026dev_b_uhf(
                    system,
                    basis,
                    (2, 1, 1),
                    options,
                    backend="ri",
                    gdf_method="mdf",
                    aux_basis="def2-svp-jk",
                    progress=False,
                )


@pytest.mark.parametrize(
    ("backend", "low_level_backend"),
    [
        ("ri", "native-multi-k-gdf-cosx-rhf"),
        ("rijcosx", "native-multi-k-gdf-gdf-rhf"),
        ("ri", "native-multi-k-gdf-gdf-rks"),
    ],
)
def test_fitted_routes_reject_mismatched_runtime_backend(
    monkeypatch,
    backend: str,
    low_level_backend: str,
) -> None:
    system, basis = _h2_system(dim=3)

    def fake_gdf(_system, _basis, kpoints, _options, **_kwargs):
        result = _fake_result(kpoints)
        result.backend = low_level_backend
        return result

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_krhf_periodic_gdf",
        fake_gdf,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(RuntimeError, match="inconsistent runtime backend"):
            run_aiccm2026dev_b_rhf(
                system,
                basis,
                (2, 1, 1),
                backend=backend,
                progress=False,
            )


@pytest.mark.parametrize(
    ("method", "backend", "backend_symbol"),
    [
        ("RHF", "four_center", "run_pbc_bipole_rhf"),
        ("RHF", "ri", "run_krhf_periodic_gdf"),
        ("RHF", "rijcosx", "run_krhf_periodic_gdf"),
        ("RKS", "four_center", "run_pbc_bipole_rks"),
        ("RKS", "ri", "run_krks_periodic_gdf"),
        ("RKS", "rijcosx", "run_krks_periodic_gdf"),
    ],
)
def test_fock_mixing_keyword_overrides_options_on_restricted_routes(
    monkeypatch,
    method: str,
    backend: str,
    backend_symbol: str,
) -> None:
    system, basis = _h2_system(dim=3)
    kpoints = cyclic_gamma_mesh(system, (2, 1, 1))
    options = vq.PeriodicRHFOptions() if method == "RHF" else vq.PeriodicKSOptions()
    options.fock_mixing = 0.10
    captured = {}

    def fake_backend(*_args, **kwargs):
        captured["fock_mixing"] = kwargs["fock_mixing"]
        result = _fake_result(kpoints)
        if backend == "four_center":
            result.runtime_backend = "pbc-bipole"
        else:
            exchange_backend = "cosx" if backend == "rijcosx" else "gdf"
            result.backend = (
                f"native-multi-k-gdf-{exchange_backend}-{method.lower()}"
            )
        result.fock_mixing = 0.25
        return result

    monkeypatch.setattr(
        f"vibeqc.periodic.chi.scf.{backend_symbol}",
        fake_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = _run_fock_mixing_route(
            method,
            system,
            basis,
            options,
            backend,
            fock_mixing=0.25,
        )

    assert options.fock_mixing == pytest.approx(0.10)
    assert captured["fock_mixing"] == pytest.approx(0.25)
    assert result.aiccm2026dev_b.fock_mixing == pytest.approx(0.25)


def test_gamma_ri_rhf_zero_keyword_controls_gdf_route_gate(monkeypatch) -> None:
    """The resolved override, not stale options, selects the Γ RI operator."""
    system, basis = _h2_system(dim=3)
    options = vq.PeriodicRHFOptions()
    options.fock_mixing = 0.10
    options.level_shift_schedule = [0.0, 0.0]
    calls: list[str] = []

    def fake_pure_gamma(_system, _basis, passed_options, **kwargs):
        calls.append("pure-gdf-ewald")
        assert passed_options is options
        assert kwargs["exxdiv"] == "ewald"
        zeros = np.zeros((2, 2), dtype=complex)
        return SimpleNamespace(
            energy=-1.0,
            e_electronic=-1.5,
            e_nuclear=0.5,
            n_iter=2,
            converged=True,
            mo_energies=np.array([-0.5, 0.5]),
            mo_coeffs=np.eye(2, dtype=complex),
            fock=zeros,
            overlap=np.eye(2, dtype=complex),
            hcore=zeros,
            density=np.diag([2.0, 0.0]).astype(complex),
            scf_trace=[],
            fock_mixing=0.0,
        )

    def fail_legacy_gamma(*_args, **_kwargs):
        calls.append("legacy-gamma")
        raise AssertionError("explicit zero reached the legacy Γ GDF route")

    monkeypatch.setattr(
        "vibeqc.pbc_gdf.run_pbc_gdf_rhf",
        fake_pure_gamma,
    )
    monkeypatch.setattr(
        "vibeqc.periodic_k_gdf.run_rhf_periodic_gamma_gdf",
        fail_legacy_gamma,
    )

    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_rhf(
            system,
            basis,
            (1, 1, 1),
            options,
            backend="ri",
            fock_mixing=0.0,
            progress=False,
        )

    assert calls == ["pure-gdf-ewald"]
    assert options.fock_mixing == pytest.approx(0.10)
    assert result.aiccm2026dev_b.fock_mixing == pytest.approx(0.0)
    assert result.finite_torus_convention.exchange_q0 == "bvk-ewald"


@pytest.mark.parametrize(
    ("options_mixing", "call_kwargs"),
    [
        (0.0, {"fock_mixing": 0.25}),
        (0.10, {}),
    ],
    ids=("keyword", "options-only"),
)
def test_gamma_ri_rhf_rejects_nonzero_resolved_fock_mixing(
    monkeypatch,
    options_mixing: float,
    call_kwargs: dict[str, float],
) -> None:
    system, basis = _h2_system(dim=3)
    options = vq.PeriodicRHFOptions()
    options.fock_mixing = options_mixing
    options.level_shift = 0.3
    options.quadratic_fallback_iter = 3
    called = False

    def fail_backend(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("unsupported Gamma-only RI mixing reached GDF")

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_krhf_periodic_gdf",
        fail_backend,
    )
    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf._auto_rsgdf_tail_ke_cutoff",
        lambda *_args, **_kwargs: pytest.fail(
            "implicit-tail guard ran before the Gamma mixing guard"
        ),
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(
            NotImplementedError,
            match="Gamma-only RI RHF cannot execute previous-Fock mixing",
        ):
            run_aiccm2026dev_b_rhf(
                system,
                basis,
                (1, 1, 1),
                options,
                backend="ri",
                progress=False,
                **call_kwargs,
            )

    assert called is False
    assert options.fock_mixing == pytest.approx(options_mixing)


@pytest.mark.parametrize(
    ("static_shift", "schedule"),
    [
        (0.3, []),
        (0.0, [0.3, 0.0]),
    ],
    ids=("static", "schedule"),
)
def test_gamma_ri_rhf_level_shift_fails_closed_before_backend_dispatch(
    monkeypatch,
    static_shift: float,
    schedule: list[float],
) -> None:
    system, basis = _h2_system(dim=3)
    options = vq.PeriodicRHFOptions()
    options.level_shift = static_shift
    options.level_shift_schedule = schedule
    options.quadratic_fallback_iter = 3
    called = False

    def fail_backend(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("unsupported Gamma-only RI shift reached GDF")

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_krhf_periodic_gdf",
        fail_backend,
    )
    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf._auto_rsgdf_tail_ke_cutoff",
        lambda *_args, **_kwargs: pytest.fail(
            "implicit-tail guard ran before the Gamma level-shift guard"
        ),
    )

    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(
            NotImplementedError,
            match="Gamma-only RI RHF cannot execute level shifting",
        ):
            run_aiccm2026dev_b_rhf(
                system,
                basis,
                (1, 1, 1),
                options,
                backend="ri",
                fock_mixing=0.0,
                progress=False,
            )

    assert called is False


def test_runner_gamma_ri_rhf_level_shift_fails_closed_before_backend_dispatch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    system, basis = _h2_system(dim=3)
    called = False

    def fail_backend(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("public Gamma-only RI shift reached GDF")

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_krhf_periodic_gdf",
        fail_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(
            NotImplementedError,
            match="Gamma-only RI RHF cannot execute level shifting",
        ):
            vq.run_periodic_job(
                system,
                basis,
                method="RHF",
                jk_method="aiccm2026dev-b",
                aiccm_backend="ri",
                aiccm_lattice_extension=(1, 1, 1),
                level_shift=0.3,
                output=tmp_path / "b-rhf-ri-gamma-shift",
                output_qvf=False,
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

    assert called is False


def test_multi_character_ri_restricted_level_shift_schedule_reaches_backend(
    monkeypatch,
) -> None:
    system, basis = _h2_system(dim=3)
    options = vq.PeriodicRHFOptions()
    options.level_shift = 0.3
    options.level_shift_warmup_cycles = 4
    options.level_shift_schedule = [0.2, 0.0]
    mesh = (2, 1, 1)
    kpoints = cyclic_gamma_mesh(system, mesh)
    called = False

    def fake_backend(_system, _basis, _kpoints, passed_options, **_kwargs):
        nonlocal called
        called = True
        assert passed_options is options
        assert passed_options.level_shift == pytest.approx(0.3)
        assert passed_options.level_shift_warmup_cycles == 4
        assert passed_options.level_shift_schedule == [0.2, 0.0]
        return _fake_result(kpoints)

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_krhf_periodic_gdf",
        fake_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_rhf(
            system,
            basis,
            mesh,
            options,
            backend="ri",
            fock_mixing=0.0,
            progress=False,
        )

    assert called is True
    assert result.aiccm2026dev_b.level_shift == pytest.approx(0.3)
    assert result.aiccm2026dev_b.level_shift_warmup_cycles == 0


@pytest.mark.parametrize(
    ("method", "backend_symbol"),
    [
        ("RHF", "run_pbc_bipole_rhf"),
        ("RKS", "run_pbc_bipole_rks"),
    ],
)
def test_restricted_four_center_static_level_shift_reaches_backend(
    monkeypatch,
    method: str,
    backend_symbol: str,
) -> None:
    system, basis = _h2_system(dim=3)
    options = (
        vq.PeriodicRHFOptions()
        if method == "RHF"
        else vq.PeriodicKSOptions()
    )
    options.level_shift = 0.3
    options.level_shift_warmup_cycles = 0
    options.level_shift_schedule = []
    options.max_iter = 1
    mesh = (1, 1, 1)
    kpoints = cyclic_gamma_mesh(system, mesh)
    called = False

    def fake_backend(_system, _basis, _kpoints, passed_options, **kwargs):
        nonlocal called
        called = True
        assert passed_options is options
        assert passed_options.level_shift == pytest.approx(0.3)
        assert kwargs["level_shift_schedule"] is None
        return _fake_result(kpoints, runtime_backend="pbc-bipole")

    monkeypatch.setattr(
        f"vibeqc.periodic.chi.scf.{backend_symbol}",
        fake_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        if method == "RHF":
            result = run_aiccm2026dev_b_rhf(
                system,
                basis,
                mesh,
                options,
                backend="four_center",
                fock_mixing=0.0,
                progress=False,
            )
        else:
            result = run_aiccm2026dev_b_rks(
                system,
                basis,
                "pbe",
                mesh,
                options,
                backend="four_center",
                fock_mixing=0.0,
                progress=False,
            )

    assert called is True
    assert result.aiccm2026dev_b.level_shift == pytest.approx(0.3)
    assert result.aiccm2026dev_b.level_shift_warmup_cycles == 0


@pytest.mark.parametrize(
    ("method", "backend_symbol"),
    [
        ("RHF", "run_pbc_bipole_rhf"),
        ("RKS", "run_pbc_bipole_rks"),
    ],
)
@pytest.mark.parametrize(
    ("requested_warmup", "max_iter", "resolved_warmup"),
    [
        (-1, 8, 5),
        (2, 8, 2),
        (10, 3, 2),
    ],
    ids=("auto", "explicit", "capped"),
)
def test_restricted_four_center_level_shift_warmup_reaches_backend(
    monkeypatch,
    method: str,
    backend_symbol: str,
    requested_warmup: int,
    max_iter: int,
    resolved_warmup: int,
) -> None:
    system, basis = _h2_system(dim=3)
    options = (
        vq.PeriodicRHFOptions()
        if method == "RHF"
        else vq.PeriodicKSOptions()
    )
    options.level_shift = 0.3
    options.level_shift_warmup_cycles = requested_warmup
    options.level_shift_schedule = []
    options.max_iter = max_iter
    mesh = (1, 1, 1)
    kpoints = cyclic_gamma_mesh(system, mesh)
    executed_trajectory = None

    def fake_backend(_system, _basis, _kpoints, passed_options, **kwargs):
        nonlocal executed_trajectory
        assert passed_options is options
        schedule = kwargs["level_shift_schedule"]
        assert schedule is not None
        executed_trajectory = [
            schedule.at(iteration) for iteration in range(1, max_iter + 1)
        ]
        return _fake_result(kpoints, runtime_backend="pbc-bipole")

    monkeypatch.setattr(
        f"vibeqc.periodic.chi.scf.{backend_symbol}",
        fake_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        if method == "RHF":
            result = run_aiccm2026dev_b_rhf(
                system,
                basis,
                mesh,
                options,
                backend="four_center",
                fock_mixing=0.0,
                progress=False,
            )
        else:
            result = run_aiccm2026dev_b_rks(
                system,
                basis,
                "pbe",
                mesh,
                options,
                backend="four_center",
                fock_mixing=0.0,
                progress=False,
            )

    assert executed_trajectory == [0.3] * resolved_warmup + [0.0] * (
        max_iter - resolved_warmup
    )
    assert options.level_shift_warmup_cycles == requested_warmup
    assert result.aiccm2026dev_b.level_shift_warmup_cycles == resolved_warmup


@pytest.mark.parametrize(
    ("method", "backend_symbol"),
    [
        ("RHF", "run_pbc_bipole_rhf"),
        ("RKS", "run_pbc_bipole_rks"),
    ],
)
@pytest.mark.parametrize("requested_warmup", [-1, 1])
def test_restricted_four_center_active_warmup_requires_unshifted_tail(
    monkeypatch,
    method: str,
    backend_symbol: str,
    requested_warmup: int,
) -> None:
    system, basis = _h2_system(dim=3)
    options = (
        vq.PeriodicRHFOptions()
        if method == "RHF"
        else vq.PeriodicKSOptions()
    )
    options.level_shift = 0.3
    options.level_shift_warmup_cycles = requested_warmup
    options.level_shift_schedule = []
    options.max_iter = 1
    options.quadratic_fallback_iter = 3
    called = False

    def fail_backend(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("tail-less warm-up request reached BIPOLE")

    monkeypatch.setattr(
        f"vibeqc.periodic.chi.scf.{backend_symbol}",
        fail_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(ValueError, match="requires max_iter >= 2"):
            if method == "RHF":
                run_aiccm2026dev_b_rhf(
                    system,
                    basis,
                    (1, 1, 1),
                    options,
                    backend="four_center",
                    progress=False,
                )
            else:
                run_aiccm2026dev_b_rks(
                    system,
                    basis,
                    "pbe",
                    (1, 1, 1),
                    options,
                    backend="four_center",
                    progress=False,
                )

    assert called is False


def test_restricted_four_center_rhf_executes_resolved_warmup_trajectory(
    monkeypatch,
) -> None:
    system, basis = _h2_system(dim=3)
    options = vq.PeriodicRHFOptions()
    options.level_shift = 0.3
    options.level_shift_warmup_cycles = 2
    options.level_shift_schedule = []
    options.max_iter = 3
    observed: list[tuple[int, float]] = []
    original_at = LevelShiftSchedule.at

    def recording_at(self, iteration: int) -> float:
        value = original_at(self, iteration)
        observed.append((iteration, value))
        return value

    monkeypatch.setattr(LevelShiftSchedule, "at", recording_at)
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_rhf(
            system,
            basis,
            (1, 1, 1),
            options,
            backend="four_center",
            fock_mixing=0.0,
            progress=False,
        )

    assert result.n_iter == 3
    assert observed == [(1, 0.3), (2, 0.3), (3, 0.0)]
    assert result.aiccm2026dev_b.level_shift_warmup_cycles == 2


@pytest.mark.parametrize(
    ("method", "backend_symbol"),
    [
        ("RHF", "run_pbc_bipole_rhf"),
        ("RKS", "run_pbc_bipole_rks"),
    ],
)
@pytest.mark.parametrize("requested_warmup", [-2, -1, 4])
def test_restricted_four_center_zero_shift_keeps_warmup_dormant(
    monkeypatch,
    method: str,
    backend_symbol: str,
    requested_warmup: int,
) -> None:
    system, basis = _h2_system(dim=3)
    options = (
        vq.PeriodicRHFOptions()
        if method == "RHF"
        else vq.PeriodicKSOptions()
    )
    options.level_shift = 0.0
    options.level_shift_warmup_cycles = requested_warmup
    options.level_shift_schedule = []
    kpoints = cyclic_gamma_mesh(system, (1, 1, 1))

    def fake_backend(_system, _basis, _kpoints, passed_options, **kwargs):
        assert passed_options is options
        assert kwargs["level_shift_schedule"] is None
        return _fake_result(kpoints, runtime_backend="pbc-bipole")

    monkeypatch.setattr(
        f"vibeqc.periodic.chi.scf.{backend_symbol}",
        fake_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        if method == "RHF":
            result = run_aiccm2026dev_b_rhf(
                system,
                basis,
                (1, 1, 1),
                options,
                backend="four_center",
                fock_mixing=0.0,
                progress=False,
            )
        else:
            result = run_aiccm2026dev_b_rks(
                system,
                basis,
                "pbe",
                (1, 1, 1),
                options,
                backend="four_center",
                fock_mixing=0.0,
                progress=False,
            )

    assert result.aiccm2026dev_b.level_shift_warmup_cycles == 0


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "message"),
    [
        ("level_shift", -0.1, "level_shift must be non-negative"),
        ("level_shift", np.inf, "level_shift must be a finite scalar"),
        (
            "level_shift_warmup_cycles",
            -2,
            "level_shift_warmup_cycles must be an integer",
        ),
    ],
)
def test_restricted_four_center_level_shift_warmup_validation(
    monkeypatch,
    field_name: str,
    invalid_value: float | int,
    message: str,
) -> None:
    system, basis = _h2_system(dim=3)
    options = vq.PeriodicRHFOptions()
    options.level_shift = 0.3
    options.level_shift_schedule = []
    setattr(options, field_name, invalid_value)
    options.quadratic_fallback_iter = 3
    called = False

    def fail_backend(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("invalid warm-up request reached BIPOLE")

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_pbc_bipole_rhf",
        fail_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(ValueError, match=message):
            run_aiccm2026dev_b_rhf(
                system,
                basis,
                (1, 1, 1),
                options,
                backend="four_center",
                progress=False,
            )

    assert called is False


@pytest.mark.parametrize(
    ("method", "backend_symbol"),
    [
        ("RHF", "run_pbc_bipole_rhf"),
        ("RKS", "run_pbc_bipole_rks"),
    ],
)
@pytest.mark.parametrize(
    ("static_shift", "schedule"),
    [
        (0.0, [0.2, 0.0]),
        (0.3, [0.0, 0.0]),
    ],
    ids=("nonzero-schedule", "all-zero-schedule-supersedes-static"),
)
def test_restricted_four_center_level_shift_schedule_fails_closed(
    monkeypatch,
    method: str,
    backend_symbol: str,
    static_shift: float,
    schedule: list[float],
) -> None:
    system, basis = _h2_system(dim=3)
    options = vq.PeriodicRHFOptions() if method == "RHF" else vq.PeriodicKSOptions()
    options.level_shift = static_shift
    options.level_shift_schedule = schedule
    options.quadratic_fallback_iter = 3
    called = False

    def fail_backend(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("unsupported four-center schedule reached BIPOLE")

    monkeypatch.setattr(
        f"vibeqc.periodic.chi.scf.{backend_symbol}",
        fail_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(
            NotImplementedError,
            match="restricted four_center does not transport",
        ):
            if method == "RHF":
                run_aiccm2026dev_b_rhf(
                    system,
                    basis,
                    (1, 1, 1),
                    options,
                    backend="four_center",
                    progress=False,
                )
            else:
                run_aiccm2026dev_b_rks(
                    system,
                    basis,
                    "pbe",
                    (1, 1, 1),
                    options,
                    backend="four_center",
                    progress=False,
                )

    assert called is False


@pytest.mark.parametrize(
    ("schedule", "message"),
    [
        ([np.inf], "finite scalars"),
        ([-0.1], "non-negative"),
    ],
)
def test_restricted_four_center_level_shift_schedule_validation(
    schedule: list[float],
    message: str,
) -> None:
    system, basis = _h2_system(dim=3)
    options = vq.PeriodicRHFOptions()
    options.level_shift_schedule = schedule

    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(ValueError, match=message):
            run_aiccm2026dev_b_rhf(
                system,
                basis,
                (1, 1, 1),
                options,
                backend="four_center",
                progress=False,
            )


@pytest.mark.parametrize(
    ("method", "backend_symbol"),
    [
        ("UHF", "run_pbc_bipole_uhf"),
        ("UKS", "run_pbc_bipole_uks"),
    ],
)
def test_fock_mixing_keyword_overrides_options_on_unrestricted_four_center(
    monkeypatch,
    method: str,
    backend_symbol: str,
) -> None:
    system, basis = _open_shell_h_system()
    kpoints = cyclic_gamma_mesh(system, (2, 1, 1))
    options = vq.PeriodicRHFOptions() if method == "UHF" else vq.PeriodicKSOptions()
    options.fock_mixing = 0.10
    captured = {}

    def fake_backend(*_args, **kwargs):
        captured["fock_mixing"] = kwargs["fock_mixing"]
        result = _fake_unrestricted_result(
            kpoints,
            runtime_backend="pbc-bipole",
        )
        result.fock_mixing = 0.25
        return result

    monkeypatch.setattr(
        f"vibeqc.periodic.chi.scf.{backend_symbol}",
        fake_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = _run_fock_mixing_route(
            method,
            system,
            basis,
            options,
            "four_center",
            fock_mixing=0.25,
        )

    assert options.fock_mixing == pytest.approx(0.10)
    assert captured["fock_mixing"] == pytest.approx(0.25)
    assert result.aiccm2026dev_b.fock_mixing == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("method", "backend", "backend_symbol"),
    [
        ("UHF", "ri", "run_kuhf_periodic_gdf"),
        ("UHF", "rijcosx", "run_kuhf_periodic_gdf"),
        ("UKS", "ri", "run_kuks_periodic_gdf"),
        ("UKS", "rijcosx", "run_kuks_periodic_gdf"),
    ],
)
def test_fitted_open_shell_rejects_fock_mixing_keyword_override(
    monkeypatch,
    method: str,
    backend: str,
    backend_symbol: str,
) -> None:
    system, basis = _open_shell_h_system()
    options = vq.PeriodicRHFOptions() if method == "UHF" else vq.PeriodicKSOptions()
    options.fock_mixing = 0.0
    called = False

    def fake_backend(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("unsupported fitted-open route reached its backend")

    monkeypatch.setattr(
        f"vibeqc.periodic.chi.scf.{backend_symbol}",
        fake_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(NotImplementedError, match="(?i)fock.mixing"):
            _run_fock_mixing_route(
                method,
                system,
                basis,
                options,
                backend,
                fock_mixing=0.25,
            )

    assert called is False


@pytest.mark.parametrize(
    ("method", "backend", "backend_symbol"),
    [
        ("UHF", "ri", "run_kuhf_periodic_gdf"),
        ("UHF", "rijcosx", "run_kuhf_periodic_gdf"),
        ("UKS", "ri", "run_kuks_periodic_gdf"),
        ("UKS", "rijcosx", "run_kuks_periodic_gdf"),
    ],
)
def test_fitted_open_shell_rejects_options_only_fock_mixing(
    monkeypatch,
    method: str,
    backend: str,
    backend_symbol: str,
) -> None:
    system, basis = _open_shell_h_system()
    options = vq.PeriodicRHFOptions() if method == "UHF" else vq.PeriodicKSOptions()
    options.fock_mixing = 0.10
    called = False

    def fake_backend(*_args, **_kwargs):
        nonlocal called
        called = True
        return _fake_unrestricted_result(cyclic_gamma_mesh(system, (2, 1, 1)))

    monkeypatch.setattr(
        f"vibeqc.periodic.chi.scf.{backend_symbol}",
        fake_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(NotImplementedError, match="(?i)fock.mixing"):
            _run_fock_mixing_route(
                method,
                system,
                basis,
                options,
                backend,
            )

    assert called is False


@pytest.mark.parametrize(
    ("method", "backend", "backend_symbol"),
    [
        ("UHF", "ri", "run_kuhf_periodic_gdf"),
        ("UHF", "rijcosx", "run_kuhf_periodic_gdf"),
        ("UKS", "ri", "run_kuks_periodic_gdf"),
        ("UKS", "rijcosx", "run_kuks_periodic_gdf"),
    ],
)
def test_zero_keyword_overrides_nonzero_options_on_fitted_open_shell(
    monkeypatch,
    method: str,
    backend: str,
    backend_symbol: str,
) -> None:
    system, basis = _open_shell_h_system()
    kpoints = cyclic_gamma_mesh(system, (2, 1, 1))
    options = vq.PeriodicRHFOptions() if method == "UHF" else vq.PeriodicKSOptions()
    options.fock_mixing = 0.10
    called = False

    def fake_backend(*_args, **_kwargs):
        nonlocal called
        called = True
        result = _fake_unrestricted_result(kpoints)
        exchange_backend = "cosx" if backend == "rijcosx" else "gdf"
        result.backend = (
            f"native-multi-k-gdf-{exchange_backend}-{method.lower()}"
        )
        result.fock_mixing = 0.0
        return result

    monkeypatch.setattr(
        f"vibeqc.periodic.chi.scf.{backend_symbol}",
        fake_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = _run_fock_mixing_route(
            method,
            system,
            basis,
            options,
            backend,
            fock_mixing=0.0,
        )

    assert called is True
    assert options.fock_mixing == pytest.approx(0.10)
    assert result.aiccm2026dev_b.fock_mixing == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("method", "backend", "backend_symbol"),
    [
        ("UHF", "four_center", "run_pbc_bipole_uhf"),
        ("UHF", "ri", "run_kuhf_periodic_gdf"),
        ("UHF", "rijcosx", "run_kuhf_periodic_gdf"),
        ("UKS", "four_center", "run_pbc_bipole_uks"),
        ("UKS", "ri", "run_kuks_periodic_gdf"),
        ("UKS", "rijcosx", "run_kuks_periodic_gdf"),
    ],
)
@pytest.mark.parametrize(
    ("static_shift", "schedule"),
    [
        (0.2, []),
        (0.0, [0.2, 0.0]),
    ],
    ids=("static", "schedule"),
)
def test_unrestricted_level_shift_fails_closed_before_backend_dispatch(
    monkeypatch,
    method: str,
    backend: str,
    backend_symbol: str,
    static_shift: float,
    schedule: list[float],
) -> None:
    system, basis = _open_shell_h_system()
    options = vq.PeriodicRHFOptions() if method == "UHF" else vq.PeriodicKSOptions()
    options.level_shift = static_shift
    options.level_shift_schedule = schedule
    options.quadratic_fallback_iter = 3
    called = False

    def fail_backend(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("unsupported unrestricted level shift reached backend")

    monkeypatch.setattr(
        f"vibeqc.periodic.chi.scf.{backend_symbol}",
        fail_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(
            NotImplementedError,
            match="unrestricted level shifting is not validated",
        ):
            _run_fock_mixing_route(
                method,
                system,
                basis,
                options,
                backend,
            )

    assert called is False


@pytest.mark.parametrize(
    ("method", "backend", "backend_symbol"),
    [
        ("UHF", "four_center", "run_pbc_bipole_uhf"),
        ("UHF", "ri", "run_kuhf_periodic_gdf"),
        ("UHF", "rijcosx", "run_kuhf_periodic_gdf"),
        ("UKS", "four_center", "run_pbc_bipole_uks"),
        ("UKS", "ri", "run_kuks_periodic_gdf"),
        ("UKS", "rijcosx", "run_kuks_periodic_gdf"),
    ],
)
def test_unrestricted_all_zero_level_shift_schedule_reaches_backend(
    monkeypatch,
    method: str,
    backend: str,
    backend_symbol: str,
) -> None:
    system, basis = _open_shell_h_system()
    kpoints = cyclic_gamma_mesh(system, (2, 1, 1))
    options = vq.PeriodicRHFOptions() if method == "UHF" else vq.PeriodicKSOptions()
    options.level_shift = 0.0
    options.level_shift_schedule = [0.0, 0.0]
    called = False

    def fake_backend(*_args, **_kwargs):
        nonlocal called
        called = True
        result = _fake_unrestricted_result(kpoints)
        if backend == "four_center":
            result.runtime_backend = "pbc-bipole"
        else:
            exchange_backend = "cosx" if backend == "rijcosx" else "gdf"
            result.backend = (
                f"native-multi-k-gdf-{exchange_backend}-{method.lower()}"
            )
        return result

    monkeypatch.setattr(
        f"vibeqc.periodic.chi.scf.{backend_symbol}",
        fake_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = _run_fock_mixing_route(
            method,
            system,
            basis,
            options,
            backend,
        )

    assert called is True
    assert result.aiccm2026dev_b.level_shift == pytest.approx(0.0)
    assert result.aiccm2026dev_b.level_shift_warmup_cycles == 0


@pytest.mark.parametrize(
    ("method", "backend", "backend_symbol"),
    [
        ("UHF", "four_center", "run_pbc_bipole_uhf"),
        ("UHF", "ri", "run_kuhf_periodic_gdf"),
        ("UHF", "rijcosx", "run_kuhf_periodic_gdf"),
        ("UKS", "four_center", "run_pbc_bipole_uks"),
        ("UKS", "ri", "run_kuks_periodic_gdf"),
        ("UKS", "rijcosx", "run_kuks_periodic_gdf"),
    ],
)
def test_runner_unrestricted_level_shift_fails_closed(
    monkeypatch,
    tmp_path: Path,
    method: str,
    backend: str,
    backend_symbol: str,
) -> None:
    system, basis = _open_shell_h_system()
    called = False

    def fail_backend(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("public level-shift request reached backend")

    monkeypatch.setattr(
        f"vibeqc.periodic.chi.scf.{backend_symbol}",
        fail_backend,
    )
    functional = {"functional": "pbe"} if method == "UKS" else {}
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(
            NotImplementedError,
            match="unrestricted level shifting is not validated",
        ):
            vq.run_periodic_job(
                system,
                basis,
                method=method,
                jk_method="aiccm2026dev-b",
                aiccm_backend=backend,
                aiccm_lattice_extension=(2, 1, 1),
                level_shift=0.2,
                output=tmp_path / f"b-{method.lower()}-{backend}-shift",
                output_qvf=False,
                write_molden_file=False,
                write_density=False,
                write_xyz_file=False,
                write_poscar_file=False,
                write_xsf_structure_file=False,
                write_cif_file=False,
                write_population_file=False,
                citations=False,
                progress=False,
                **functional,
            )

    assert called is False


def test_driver_passes_exact_cyclic_mesh_and_attaches_invariants(monkeypatch) -> None:
    system, basis = _h2_system(dim=3)
    captured = {}

    def fake_gdf(_system, _basis, kpoints, _options, **kwargs):
        captured["kpoints"] = kpoints
        captured["kwargs"] = kwargs
        return _fake_result(kpoints)

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_krhf_periodic_gdf",
        fake_gdf,
    )
    result = run_aiccm2026dev_b_rhf(
        system,
        basis,
        (4, 1, 1),
        backend="ri",
        progress=False,
    )
    kpoints = captured["kpoints"]
    assert kpoints.mesh == (4, 1, 1)
    assert kpoints.shift == (0, 0, 0)
    assert result.backend == "aiccm2026dev-b-ri"
    assert result.aiccm2026dev_b.backend == "ri"
    assert result.aiccm2026dev_b.n_cyclic_cells == 4
    assert result.aiccm2026dev_b.wigner_seitz_partition_error < 1e-15
    assert result.aiccm2026dev_b.density_idempotency_error < 1e-15
    assert result.aiccm2026dev_b.electron_count_error < 1e-15
    residue_execution = result.residue_inverse_transform_execution
    assert result.aiccm2026dev_b.residue_inverse_transform == residue_execution
    assert residue_execution.task_kind == "chi-residue-inverse-transform"
    assert residue_execution.active is False
    assert residue_execution.world_size == 1
    assert residue_execution.residue_keys == (
        (0, 0, 0),
        (1, 0, 0),
        (2, 0, 0),
        (3, 0, 0),
    )
    assert residue_execution.representative_multiplicities == (1, 1, 2, 1)
    assert residue_execution.extra_wigner_weight_applied is False
    assert result.exchange_q0 == "bvk-ewald"
    assert result.coulomb_kernel == "3d-periodic-g0"
    convention = result.aiccm2026dev_b.finite_torus_convention
    assert convention is not None
    assert convention.ccm_approach == "chi-ccm"
    assert convention.ccm_construction == "finite-translation-group-character"
    assert convention.evaluation_representation == "gamma-centred-character-mesh"
    assert convention.coulomb_kernel == "3d-periodic-g0"
    assert convention.exchange_q0 == "bvk-ewald"
    assert convention.exchange_q0_applicability == "active"
    assert result.exchange_q0_applicability == "active"
    assert convention.boundary_model == "3d-periodic"
    assert convention.lattice_vector_convention == "columns"
    assert convention.character_mesh_shape == (4, 1, 1)
    assert convention.bvk_madelung_supercell_repetitions == (4, 1, 1)
    assert convention.bvk_madelung_supercell_lattice_bohr[0] == pytest.approx(
        (48.0, 0.0, 0.0)
    )
    kpath = vq.KPoints.band_path(
        system,
        scheme="manual",
        segments=[([0.0, 0.0, 0.0], "G", [0.5, 0.0, 0.0], "X")],
        points_per_segment=2,
    ).to_kpath()
    bands = vq.aiccm2026dev_b_band_structure(result, system, kpath)
    assert bands.metadata["finite_torus_convention"] == convention
    assert bands.metadata["exchange_q0"] == "bvk-ewald"
    assert bands.metadata["exchange_q0_applicability"] == "active"
    assert (
        bands.metadata["exact_exchange_assembly"]
        is result.exact_exchange_assembly
    )
    delattr(result, "exact_exchange_assembly")
    with pytest.raises(RuntimeError, match="versioned exact-exchange assembly"):
        vq.aiccm2026dev_b_band_structure(result, system, kpath)


@pytest.mark.parametrize(
    (
        "functional",
        "expected_applicability",
        "expected_c_full",
        "expected_c_sr",
        "expected_omega",
    ),
    [
        ("pbe", "inactive", 0.0, 0.0, 0.0),
        ("hse06", "inactive", 0.0, 0.25, 0.11),
        ("pbe0", "active", 0.25, 0.0, 0.0),
    ],
)
def test_rks_convention_records_live_full_range_exchange_q0_applicability(
    monkeypatch,
    functional: str,
    expected_applicability: str,
    expected_c_full: float,
    expected_c_sr: float,
    expected_omega: float,
) -> None:
    system, basis = _h2_system(dim=3)
    character_kpoints = cyclic_gamma_mesh(system, (2, 1, 1))

    def fake_bipole(_system, _basis, _kpoints, _options, **_kwargs):
        execution = None
        if functional == "hse06":
            execution = BipoleScreenedExchangeExecution(
                c_sr=0.25,
                omega_screen_bohr_inv=0.11,
            )
        return _fake_result(
            character_kpoints,
            runtime_backend="pbc-bipole",
            screened_exchange_execution=execution,
            exchange_ewald_split=functional == "hse06",
            exchange_exxdiv=("ewald" if functional == "hse06" else None),
        )

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_pbc_bipole_rks",
        fake_bipole,
    )
    result = run_aiccm2026dev_b_rks(
        system,
        basis,
        functional,
        (2, 1, 1),
        backend="four_center",
        progress=False,
    )

    convention = result.finite_torus_convention
    assembly = result.exact_exchange_assembly
    assert result.aiccm2026dev_b.exact_exchange_assembly is assembly
    assert assembly.c_full == pytest.approx(expected_c_full)
    assert assembly.c_sr == pytest.approx(expected_c_sr)
    assert assembly.omega_screen_bohr_inv == pytest.approx(expected_omega)
    assert assembly.schema == (
        "vibeqc.aiccm2026dev-b.exact-exchange-assembly/v1"
    )
    assert assembly.screened_exchange_applicability == (
        "active" if expected_c_sr else "inactive"
    )
    assert assembly.screened_exchange_assembly == (
        "short-range-direct" if expected_c_sr else "not-applicable"
    )
    if functional == "hse06":
        # The shared corrected-gauge selector remains recorded even though
        # c_full=0 makes its exchange seam inactive. It must not override the
        # branch-emitted direct K_erfc evidence.
        assert result.exchange_ewald_split is True
        assert result.exchange_exxdiv == "ewald"
    assert convention.exchange_q0 == "bvk-ewald"
    assert convention.exchange_q0_applicability == expected_applicability
    assert result.exchange_q0_applicability == expected_applicability
    if expected_applicability == "active":
        assert "include the declared exchange_q0 seam" in (
            convention.orbital_energy_convention
        )
    else:
        assert "full-range exact-exchange coefficient is zero" in (
            convention.orbital_energy_convention
        )


def test_exact_exchange_metadata_comes_from_live_shared_resolver(monkeypatch) -> None:
    system, basis = _h2_system(dim=3)
    character_kpoints = cyclic_gamma_mesh(system, (2, 1, 1))
    calls: list[tuple[str, str]] = []

    def fake_resolver(func, *, where: str):
        calls.append((str(func.name), where))
        return SimpleNamespace(c_full=0.0, c_sr=0.375, omega_screen=0.2)

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.resolve_periodic_exchange",
        fake_resolver,
    )
    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_pbc_bipole_rks",
        lambda *_args, **_kwargs: _fake_result(
            character_kpoints,
            runtime_backend="pbc-bipole",
            screened_exchange_execution=BipoleScreenedExchangeExecution(
                c_sr=0.375,
                omega_screen_bohr_inv=0.2,
            ),
            exchange_ewald_split=True,
            exchange_exxdiv="ewald",
        ),
    )

    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_rks(
            system,
            basis,
            "pbe0",
            (2, 1, 1),
            backend="four_center",
            progress=False,
        )

    assert calls == [("pbe0", "run_aiccm2026dev_b_rks")]
    assert result.exact_exchange_assembly == AICCM2026DevBExactExchangeAssembly(
        c_full=0.0,
        c_sr=0.375,
        omega_screen_bohr_inv=0.2,
        screened_exchange_applicability="active",
        screened_exchange_assembly="short-range-direct",
    )
    assert result.exchange_q0_applicability == "inactive"


def test_uks_hse_records_spin_resolved_screened_exchange_assembly(
    monkeypatch,
) -> None:
    system, basis = _open_shell_h_system()
    character_kpoints = cyclic_gamma_mesh(system, (2, 1, 1))

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_pbc_bipole_uks",
        lambda *_args, **_kwargs: _fake_unrestricted_result(
            character_kpoints,
            runtime_backend="pbc-bipole",
            screened_exchange_execution=BipoleScreenedExchangeExecution(
                c_sr=0.25,
                omega_screen_bohr_inv=0.11,
            ),
            exchange_ewald_split=True,
            exchange_exxdiv="ewald",
        ),
    )

    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_uks(
            system,
            basis,
            "hse06",
            (2, 1, 1),
            backend="four_center",
            progress=False,
        )

    assembly = result.exact_exchange_assembly
    assert result.aiccm2026dev_b.exact_exchange_assembly is assembly
    assert assembly.c_full == pytest.approx(0.0)
    assert assembly.c_sr == pytest.approx(0.25)
    assert assembly.omega_screen_bohr_inv == pytest.approx(0.11)
    assert assembly.screened_exchange_applicability == "active"
    assert assembly.screened_exchange_assembly == "short-range-direct"
    assert result.exchange_q0_applicability == "inactive"


def test_rks_hse_live_direct_branch_attestation_survives_gauge_selectors() -> None:
    system, basis = _h2_system(dim=3)

    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_rks(
            system,
            basis,
            "hse06",
            (1, 1, 1),
            backend="four_center",
            progress=False,
        )

    assert result.converged
    assert result.exchange_ewald_split is True
    assert result.exchange_exxdiv == "ewald"
    assert result.exchange_q0_applicability == "inactive"
    assembly = result.exact_exchange_assembly
    assert assembly.screened_exchange_applicability == "active"
    assert assembly.screened_exchange_assembly == "short-range-direct"
    assert assembly.c_sr == pytest.approx(0.25)
    assert assembly.omega_screen_bohr_inv == pytest.approx(0.11)
    assert result.output_cell_farming_execution is None
    assert result.aiccm2026dev_b.direct_output_cell_farming is None


def test_generic_rks_output_farming_rejects_screened_multiphase_evidence() -> None:
    system, basis = _h2_system(dim=3)
    options = vq.PeriodicKSOptions()
    options.functional = "hse06"
    options.max_iter = 1

    with pytest.raises(NotImplementedError, match="single-phase execution record"):
        vq.run_pbc_bipole_rks(
            system,
            basis,
            cyclic_gamma_mesh(system, (1, 1, 1)).to_bloch_kmesh(),
            options,
            functional="hse06",
            farm_output_cells=True,
            progress=False,
        )


def test_rks_hse_selector_explicitly_keeps_output_farming_serial(
    monkeypatch,
) -> None:
    system, basis = _h2_system(dim=3)
    character_kpoints = cyclic_gamma_mesh(system, (2, 1, 1))
    captured: dict[str, object] = {}

    def fake_backend(*_args, **kwargs):
        captured.update(kwargs)
        return _fake_result(
            character_kpoints,
            runtime_backend="pbc-bipole",
            screened_exchange_execution=BipoleScreenedExchangeExecution(
                c_sr=0.25,
                omega_screen_bohr_inv=0.11,
            ),
            exchange_ewald_split=True,
            exchange_exxdiv="ewald",
        )

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_pbc_bipole_rks",
        fake_backend,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_rks(
            system,
            basis,
            "hse06",
            (2, 1, 1),
            backend="four_center",
            progress=False,
        )

    assert captured["farm_output_cells"] is False
    assert captured["output_cell_farming_task_kind"] == "chi-direct-output-cell"
    assert result.aiccm2026dev_b.direct_output_cell_farming is None


@pytest.mark.parametrize(
    ("execution", "message"),
    [
        (None, "lacks BIPOLE branch-level"),
        (
            BipoleScreenedExchangeExecution(
                c_sr=0.20,
                omega_screen_bohr_inv=0.11,
            ),
            "contradicts the live resolver",
        ),
    ],
)
def test_rks_hse_fails_closed_on_missing_or_contradictory_execution_evidence(
    monkeypatch,
    execution: BipoleScreenedExchangeExecution | None,
    message: str,
) -> None:
    system, basis = _h2_system(dim=3)
    character_kpoints = cyclic_gamma_mesh(system, (2, 1, 1))
    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_pbc_bipole_rks",
        lambda *_args, **_kwargs: _fake_result(
            character_kpoints,
            runtime_backend="pbc-bipole",
            screened_exchange_execution=execution,
            exchange_ewald_split=True,
            exchange_exxdiv="ewald",
        ),
    )

    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(RuntimeError, match=message):
            run_aiccm2026dev_b_rks(
                system,
                basis,
                "hse06",
                (2, 1, 1),
                backend="four_center",
                progress=False,
            )


def test_rks_nonscreened_route_rejects_stray_screened_execution_evidence(
    monkeypatch,
) -> None:
    system, basis = _h2_system(dim=3)
    character_kpoints = cyclic_gamma_mesh(system, (2, 1, 1))
    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_pbc_bipole_rks",
        lambda *_args, **_kwargs: _fake_result(
            character_kpoints,
            runtime_backend="pbc-bipole",
            screened_exchange_execution=BipoleScreenedExchangeExecution(
                c_sr=0.25,
                omega_screen_bohr_inv=0.11,
            ),
        ),
    )

    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(RuntimeError, match="route with c_sr=0"):
            run_aiccm2026dev_b_rks(
                system,
                basis,
                "pbe0",
                (2, 1, 1),
                backend="four_center",
                progress=False,
            )


@pytest.mark.parametrize("backend", ["ri", "rijcosx"])
def test_rks_fitted_backends_keep_range_separated_fail_close(
    monkeypatch,
    backend: str,
) -> None:
    """Shared COSX support must not silently widen the χ backend contract."""

    system, basis = _h2_system(dim=3)

    def unexpected_shared_dispatch(*_args, **_kwargs):
        pytest.fail("range-separated χ route reached the shared GDF driver")

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_krks_periodic_gdf",
        unexpected_shared_dispatch,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(
            NotImplementedError,
            match="RI and RIJCOSX backends remain fail-closed",
        ):
            run_aiccm2026dev_b_rks(
                system,
                basis,
                "hse06",
                (2, 1, 1),
                backend=backend,
                progress=False,
            )


@pytest.mark.parametrize("backend", ["ri", "rijcosx"])
def test_uks_fitted_backends_keep_range_separated_fail_close(
    monkeypatch,
    backend: str,
) -> None:
    """Open-shell shared COSX support must not widen the χ contract."""

    system, basis = _open_shell_h_system()

    def unexpected_shared_dispatch(*_args, **_kwargs):
        pytest.fail("range-separated χ route reached the shared GDF driver")

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_kuks_periodic_gdf",
        unexpected_shared_dispatch,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        with pytest.raises(
            NotImplementedError,
            match="RI and RIJCOSX backends remain fail-closed",
        ):
            run_aiccm2026dev_b_uks(
                system,
                basis,
                "hse06",
                (2, 1, 1),
                backend=backend,
                progress=False,
            )


def test_slab_selector_does_not_offer_generic_route_as_chi_substitute() -> None:
    with pytest.raises(
        NotImplementedError,
        match="Generic slab_ewald_2d or slab GDF routes are not substitutes",
    ):
        vq.pick_jk_method(
            "aiccm2026dev-b",
            lattice=np.diag([8.0, 8.0, 20.0]),
            basis_name="sto-3g",
            n_atoms=2,
            dim=2,
            scf_method="RHF",
        )


def test_rks_diagnostics_prefer_backend_resolved_fock_mixing(monkeypatch) -> None:
    system, basis = _h2_system(dim=3)
    character_kpoints = cyclic_gamma_mesh(system, (2, 1, 1))
    options = vq.PeriodicKSOptions()
    options.use_diis = False
    options.fock_mixing = 0.0

    def fake_bipole(_system, _basis, _kpoints, _options, **_kwargs):
        result = _fake_result(
            character_kpoints,
            runtime_backend="pbc-bipole",
        )
        result.fock_mixing = 0.30
        return result

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_pbc_bipole_rks",
        fake_bipole,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_rks(
            system,
            basis,
            "pbe",
            (2, 1, 1),
            options,
            backend="four_center",
            progress=False,
        )

    assert options.fock_mixing == 0.0
    assert result.aiccm2026dev_b.fock_mixing == pytest.approx(0.30)


@pytest.mark.parametrize(
    ("requested", "executed"),
    [(0.0, 0.0), (0.25, 0.25)],
)
def test_rks_diagnostics_preserve_fitted_backend_fock_mixing(
    monkeypatch,
    requested: float,
    executed: float,
) -> None:
    system, basis = _h2_system(dim=3)
    options = vq.PeriodicKSOptions()
    options.use_diis = False
    options.fock_mixing = 0.0
    captured = {}

    def fake_gdf(_system, _basis, kpoints, _options, **kwargs):
        captured["fock_mixing"] = kwargs["fock_mixing"]
        result = _fake_result(kpoints)
        result.backend = "native-multi-k-gdf-gdf-rks"
        result.fock_mixing = executed
        return result

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_krks_periodic_gdf",
        fake_gdf,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_rks(
            system,
            basis,
            "pbe",
            (2, 1, 1),
            options,
            backend="ri",
            fock_mixing=requested,
            progress=False,
        )

    assert captured["fock_mixing"] == requested
    assert result.aiccm2026dev_b.fock_mixing == pytest.approx(executed)


def test_uks_diagnostics_prefer_backend_resolved_fock_mixing(monkeypatch) -> None:
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 12.0,
        [vq.Atom(1, [6.0, 6.0, 6.0])],
        charge=0,
        multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    character_kpoints = cyclic_gamma_mesh(system, (2, 1, 1))
    options = vq.PeriodicKSOptions()
    options.use_diis = False
    options.fock_mixing = 0.0

    def fake_bipole(_system, _basis, _kpoints, _options, **_kwargs):
        result = _fake_unrestricted_result(
            character_kpoints,
            runtime_backend="pbc-bipole",
        )
        result.fock_mixing = 0.30
        return result

    monkeypatch.setattr(
        "vibeqc.periodic.chi.scf.run_pbc_bipole_uks",
        fake_bipole,
    )
    with pytest.warns(AICCM2026DevBExperimentalWarning):
        result = run_aiccm2026dev_b_uks(
            system,
            basis,
            "pbe",
            (2, 1, 1),
            options,
            backend="four_center",
            progress=False,
        )

    assert options.fock_mixing == 0.0
    assert result.aiccm2026dev_b.fock_mixing == pytest.approx(0.30)


def test_runner_dispatches_independent_keyword(monkeypatch, tmp_path: Path) -> None:
    system, basis = _h2_system()
    captured = {}

    def fake_driver(_system, _basis, mesh, _options, **kwargs):
        captured["mesh"] = mesh
        captured["backend"] = kwargs["backend"]
        captured["gdf_method"] = kwargs["gdf_method"]
        captured["rsgdf_ke_cutoff"] = kwargs["rsgdf_ke_cutoff"]
        captured["rsgdf_tail_ke_cutoff"] = kwargs[
            "rsgdf_tail_ke_cutoff"
        ]
        captured["mdf_ke_cutoff"] = kwargs["mdf_ke_cutoff"]
        captured["level_shift"] = _options.level_shift
        captured["dynamic_damping"] = _options.dynamic_damping
        kpoints = cyclic_gamma_mesh(_system, mesh)
        result = _fake_result(kpoints)
        result.aiccm2026dev_b = type(
            "Diagnostics",
            (),
            {
                "mesh": (2, 1, 1),
                "n_cyclic_cells": 2,
                "wigner_seitz_partition_error": 0.0,
                "density_idempotency_error": 0.0,
                "electron_count_error": 0.0,
                "inverse_bloch_imaginary_residual": 0.0,
                "backend": "ri",
                "electronic_method": "RHF",
                "ccm_approach": "chi-ccm",
                "ccm_construction": "finite-translation-group-character",
                "evaluation_representation": "gamma-centred-character-mesh",
            },
        )()
        return result

    monkeypatch.setattr("vibeqc.periodic_runner.run_aiccm2026dev_b_rhf", fake_driver)
    result = vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        jk_method="aiccm2026dev-b",
        aiccm_backend="ri",
        rsgdf_ke_cutoff=80.0,
        rsgdf_tail_ke_cutoff=400.0,
        mdf_ke_cutoff=12.0,
        level_shift=0.3,
        dynamic_damping=False,
        kpoints=(2, 1, 1),
        output=tmp_path / "aiccm2026dev-b",
        output_qvf=False,
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
    assert captured["mesh"] == (2, 1, 1)
    assert captured["backend"] == "ri"
    assert captured["gdf_method"] == "rsgdf"
    assert captured["rsgdf_ke_cutoff"] == pytest.approx(80.0)
    assert captured["rsgdf_tail_ke_cutoff"] == pytest.approx(400.0)
    assert captured["mdf_ke_cutoff"] == pytest.approx(12.0)
    assert captured["level_shift"] == pytest.approx(0.3)
    assert captured["dynamic_damping"] is False
    log_text = (tmp_path / "aiccm2026dev-b.out").read_text(encoding="utf-8")
    assert "gdf_method          = rsgdf" in log_text
    assert "rsgdf_ke_cutoff     = 80.0" in log_text
    assert "rsgdf_tail_ke_cutoff = 400.0" in log_text
    assert "mdf_ke_cutoff       = 12.0" in log_text
    assert "dynamic_damping     = False" in log_text
    assert result.energy == pytest.approx(-1.0)

    vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        jk_method="aiccm2026dev-b",
        aiccm_backend="ri",
        kpoints=(2, 1, 1),
        output=tmp_path / "aiccm2026dev-b-default-dynamic",
        output_qvf=False,
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
    assert captured["dynamic_damping"] is True


@pytest.mark.parametrize(
    ("backend", "gdf_method", "base_ke_cutoff", "tail_ke_cutoff", "match"),
    [
        (
            "four_center",
            "rsgdf",
            200.0,
            420.0,
            "fitted RI and RIJCOSX",
        ),
        ("ri", "mdf", 200.0, 420.0, "requires gdf_method='rsgdf'"),
        ("ri", "rsgdf", -2.0, -1.0, "positive and finite"),
    ],
)
def test_runner_rejects_tail_outside_fitted_rsgdf_envelope_before_dispatch(
    monkeypatch,
    tmp_path: Path,
    backend: str,
    gdf_method: str,
    base_ke_cutoff,
    tail_ke_cutoff,
    match: str,
) -> None:
    system, basis = _h2_system()
    monkeypatch.setattr(
        "vibeqc.periodic_runner.run_aiccm2026dev_b_rhf",
        lambda *_args, **_kwargs: pytest.fail(
            "dispatched χ SCF outside the fitted RSGDF tail envelope"
        ),
    )

    with pytest.raises((NotImplementedError, ValueError), match=match):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="aiccm2026dev-b",
            aiccm_backend=backend,
            gdf_method=gdf_method,
            rsgdf_ke_cutoff=base_ke_cutoff,
            rsgdf_tail_ke_cutoff=tail_ke_cutoff,
            kpoints=(2, 1, 1),
            output=tmp_path / f"tail-preflight-{backend}-{gdf_method}",
            output_qvf=False,
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


def test_runner_rejects_explicit_dynamic_damping_outside_b(tmp_path: Path) -> None:
    system, basis = _h2_system()

    with pytest.raises(NotImplementedError, match="dynamic_damping"):
        vq.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="gdf",
            dynamic_damping=False,
            output=tmp_path / "non-b-dynamic-damping",
            output_qvf=False,
            citations=False,
            progress=False,
        )


def test_runner_dispatches_rks_and_backend(monkeypatch, tmp_path: Path) -> None:
    system, basis = _h2_system()
    captured = {}

    def fake_driver(_system, _basis, functional, mesh, _options, **kwargs):
        captured["functional"] = functional
        captured["mesh"] = mesh
        captured["backend"] = kwargs["backend"]
        kpoints = cyclic_gamma_mesh(_system, mesh)
        result = _fake_result(kpoints)
        result.aiccm2026dev_b = type(
            "Diagnostics",
            (),
            {
                "mesh": (2, 1, 1),
                "n_cyclic_cells": 2,
                "wigner_seitz_partition_error": 0.0,
                "density_idempotency_error": 0.0,
                "electron_count_error": 0.0,
                "inverse_bloch_imaginary_residual": 0.0,
                "backend": "rijcosx",
                "electronic_method": "RKS/PBE0",
                "ccm_approach": "chi-ccm",
                "ccm_construction": "finite-translation-group-character",
                "evaluation_representation": "gamma-centred-character-mesh",
            },
        )()
        return result

    monkeypatch.setattr("vibeqc.periodic_runner.run_aiccm2026dev_b_rks", fake_driver)
    result = vq.run_periodic_job(
        system,
        basis,
        method="RKS",
        functional="pbe0",
        jk_method="aiccm2026dev-b",
        aiccm_backend="rijcosx",
        kpoints=(2, 1, 1),
        output=tmp_path / "aiccm2026dev-b-rks",
        output_qvf=False,
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
    assert captured == {
        "functional": "pbe0",
        "mesh": (2, 1, 1),
        "backend": "rijcosx",
    }
    assert result.energy == pytest.approx(-1.0)


def test_charged_and_open_shell_cells_fail_closed() -> None:
    system, basis = _h2_system()
    system.charge = 1
    with pytest.raises(ValueError, match="neutral effective primitive cell"):
        run_aiccm2026dev_b_rhf(system, basis, (1, 1, 1), progress=False)
    system.charge = 0
    system.multiplicity = 3
    with pytest.raises(ValueError, match="closed-shell RHF/RKS only"):
        run_aiccm2026dev_b_rhf(system, basis, (1, 1, 1), progress=False)


def test_inconsistent_q_only_density_fit_fails_closed() -> None:
    system, basis = _h2_system()
    with pytest.raises(ValueError, match="q-only compcell fit"):
        run_aiccm2026dev_b_rhf(
            system,
            basis,
            (2, 1, 1),
            backend="ri",
            gdf_method="compcell",
            progress=False,
        )


def test_backend_enum_and_gamma_approximation_guards() -> None:
    system, basis = _h2_system()
    assert AICCM2026DevBBackend.FOUR_CENTER.value == "four_center"
    with pytest.raises(NotImplementedError, match="RIJCOSX requires"):
        run_aiccm2026dev_b_rhf(
            system,
            basis,
            (1, 1, 1),
            backend="rijcosx",
            progress=False,
        )
    with pytest.raises(NotImplementedError, match="RI and RIJCOSX RKS"):
        run_aiccm2026dev_b_rks(
            system,
            basis,
            "pbe",
            (1, 1, 1),
            backend="ri",
            progress=False,
        )


@pytest.mark.slow
def test_all_integral_backends_run_rhf_and_hybrid_rks() -> None:
    system = vq.PeriodicSystem(
        3,
        np.diag([8.0, 12.0, 12.0]),
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    rhf_energies = {}
    rks_energies = {}
    for backend in ("four_center", "ri", "rijcosx"):
        rhf_options = vq.PeriodicRHFOptions()
        rhf_options.max_iter = 25
        rhf_options.conv_tol_energy = 1e-7
        rhf_options.damping = 0.0
        rhf = run_aiccm2026dev_b_rhf(
            system,
            basis,
            (2, 1, 1),
            rhf_options,
            backend=backend,
            progress=False,
        )
        rks_options = vq.PeriodicKSOptions()
        rks_options.max_iter = 25
        rks_options.conv_tol_energy = 1e-6
        rks_options.damping = 0.0
        rks = run_aiccm2026dev_b_rks(
            system,
            basis,
            "pbe0",
            (2, 1, 1),
            rks_options,
            backend=backend,
            progress=False,
        )
        assert rhf.converged and rks.converged
        assert rhf.aiccm2026dev_b.density_idempotency_error < 1e-7
        assert rks.aiccm2026dev_b.density_idempotency_error < 1e-7
        rhf_energies[backend] = rhf.energy
        rks_energies[backend] = rks.energy

    assert max(rhf_energies.values()) - min(rhf_energies.values()) < 7e-5
    assert max(rks_energies.values()) - min(rks_energies.values()) < 7e-5


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("backend", ["ri", "rijcosx"])
@pytest.mark.parametrize("driver", [run_aiccm2026dev_b_rhf, run_aiccm2026dev_b_rks])
def test_lower_dimensional_ri_and_rijcosx_routes_fail_closed(
    dim,
    backend,
    driver,
) -> None:
    """Shared neutral-RI/GDF low-D Coulomb support is not a Coulomb kernel."""

    system, basis = _h2_system(dim=dim, box=12.0)
    mesh = (2, 1, 1)
    options = (
        vq.PeriodicKSOptions()
        if driver is run_aiccm2026dev_b_rks
        else vq.PeriodicRHFOptions()
    )
    options.max_iter = 2

    with pytest.raises(
        NotImplementedError,
        match="neutral wire/slab Hamiltonian",
    ):
        with pytest.warns(AICCM2026DevBExperimentalWarning):
            if driver is run_aiccm2026dev_b_rks:
                driver(
                    system,
                    basis,
                    "pbe",
                    mesh,
                    options,
                    backend=backend,
                    progress=False,
                )
            else:
                driver(
                    system,
                    basis,
                    mesh,
                    options,
                    backend=backend,
                    progress=False,
                )


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("method", ["RHF", "RKS", "UHF", "UKS"])
@pytest.mark.parametrize("mixing", [0.0, 0.10])
def test_one_cell_lower_dimensional_ri_keeps_d72_fail_close(
    dim: int,
    method: str,
    mixing: float,
) -> None:
    """Gamma and mixing guards must not expose a wire/slab RI energy."""

    if method in ("RHF", "RKS"):
        system, basis = _h2_system(dim=dim, box=12.0)
    else:
        lattice = np.diag([4.0 if dim == 1 else 12.0, 12.0, 12.0])
        system = vq.PeriodicSystem(
            dim,
            lattice,
            [vq.Atom(1, [0.0, 0.0, 0.0])],
            charge=0,
            multiplicity=2,
        )
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = (
        vq.PeriodicKSOptions()
        if method in ("RKS", "UKS")
        else vq.PeriodicRHFOptions()
    )
    options.fock_mixing = mixing

    with pytest.raises(
        NotImplementedError,
        match="neutral wire/slab Hamiltonian",
    ):
        with pytest.warns(AICCM2026DevBExperimentalWarning):
            if method == "RHF":
                run_aiccm2026dev_b_rhf(
                    system,
                    basis,
                    (1, 1, 1),
                    options,
                    backend="ri",
                    progress=False,
                )
            elif method == "RKS":
                run_aiccm2026dev_b_rks(
                    system,
                    basis,
                    "pbe",
                    (1, 1, 1),
                    options,
                    backend="ri",
                    progress=False,
                )
            elif method == "UHF":
                run_aiccm2026dev_b_uhf(
                    system,
                    basis,
                    (1, 1, 1),
                    options,
                    backend="ri",
                    progress=False,
                )
            else:
                run_aiccm2026dev_b_uks(
                    system,
                    basis,
                    "pbe",
                    (1, 1, 1),
                    options,
                    backend="ri",
                    progress=False,
                )


def test_lower_dimensional_guard_precedes_quadratic_fallback_guard() -> None:
    system, basis = _h2_system(dim=1, box=12.0)
    options = vq.PeriodicRHFOptions()
    options.quadratic_fallback_iter = 3

    with pytest.raises(
        NotImplementedError,
        match="neutral wire/slab Hamiltonian",
    ):
        with pytest.warns(AICCM2026DevBExperimentalWarning):
            run_aiccm2026dev_b_rhf(
                system,
                basis,
                (2, 1, 1),
                options,
                backend="ri",
                progress=False,
            )


@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("driver", [run_aiccm2026dev_b_rhf, run_aiccm2026dev_b_rks])
@pytest.mark.parametrize(
    ("static_shift", "warmup", "schedule"),
    [
        (0.0, -1, [0.2, 0.0]),
        (0.3, 2, []),
    ],
    ids=("explicit-schedule", "static-warmup"),
)
def test_lower_dimensional_four_center_fails_closed(
    dim,
    driver,
    static_shift,
    warmup,
    schedule,
) -> None:
    """D72 precedes four-center schedule and warm-up checks in 1D and 2D."""

    system, basis = _h2_system(dim=dim, box=15.0)
    options = (
        vq.PeriodicKSOptions()
        if driver is run_aiccm2026dev_b_rks
        else vq.PeriodicRHFOptions()
    )
    options.max_iter = 2
    options.level_shift = static_shift
    options.level_shift_warmup_cycles = warmup
    options.level_shift_schedule = schedule
    with pytest.raises(NotImplementedError, match="neutral wire/slab Coulomb"):
        with pytest.warns(AICCM2026DevBExperimentalWarning):
            if driver is run_aiccm2026dev_b_rks:
                driver(
                    system,
                    basis,
                    "pbe",
                    (2, 1, 1),
                    options,
                    backend="four_center",
                    progress=False,
                )
            else:
                driver(
                    system,
                    basis,
                    (2, 1, 1),
                    options,
                    backend="four_center",
                    progress=False,
                )


def test_open_shell_lower_dimensional_four_center_fails_closed() -> None:
    system = vq.PeriodicSystem(
        1,
        np.diag([15.0, 40.0, 40.0]),
        [vq.Atom(1, [0.0, 0.0, 0.0])],
    )
    system.multiplicity = 2
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = vq.PeriodicRHFOptions()
    options.max_iter = 2
    with pytest.raises(NotImplementedError, match="neutral wire/slab Coulomb"):
        with pytest.warns(AICCM2026DevBExperimentalWarning):
            run_aiccm2026dev_b_uhf(
                system,
                basis,
                (2, 1, 1),
                options,
                backend="four_center",
                progress=False,
            )


def test_open_shell_lower_dimensional_guard_precedes_level_shift_guard() -> None:
    system = vq.PeriodicSystem(
        1,
        np.diag([15.0, 40.0, 40.0]),
        [vq.Atom(1, [0.0, 0.0, 0.0])],
        multiplicity=2,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = vq.PeriodicRHFOptions()
    options.level_shift = 0.2

    with pytest.raises(
        NotImplementedError,
        match="neutral wire/slab Coulomb",
    ):
        with pytest.warns(AICCM2026DevBExperimentalWarning):
            run_aiccm2026dev_b_uhf(
                system,
                basis,
                (2, 1, 1),
                options,
                backend="four_center",
                progress=False,
            )


@pytest.mark.slow
def test_small_end_to_end_rhf_is_idempotent() -> None:
    system, basis = _h2_system(box=12.0)
    options = vq.PeriodicRHFOptions()
    options.max_iter = 30
    options.conv_tol_energy = 1e-8
    options.damping = 0.0
    result = run_aiccm2026dev_b_rhf(
        system,
        basis,
        (1, 1, 1),
        options,
        progress=False,
    )
    assert result.converged
    assert result.aiccm2026dev_b.density_idempotency_error < 1e-7
    assert result.aiccm2026dev_b.electron_count_error < 1e-7
    assert result.kpoints_frac.shape == (1, 3)
    bonds = vq.aiccm2026dev_b_mayer_bond_orders(
        result,
        system,
        basis,
        threshold=1.0e-3,
    )
    assert bonds.bonds
    assert bonds.finite_torus_convention == result.finite_torus_convention
    assert bonds.translational_spread < 1.0e-8
    props = vq.derive_aiccm2026dev_b_scf_properties(result, system, basis)
    assert props.finite_torus_convention == result.finite_torus_convention
    kpath = vq.KPoints.band_path(
        system,
        scheme="manual",
        segments=[([0.0, 0.0, 0.0], "G", [0.5, 0.0, 0.0], "X")],
        points_per_segment=3,
    ).to_kpath()
    bands = vq.aiccm2026dev_b_band_structure(result, system, kpath)
    assert bands.energies.shape == (kpath.n_points, basis.nbasis)
    assert np.all(np.isfinite(bands.energies))


@pytest.mark.slow
def test_alternating_h4_two_cell_benchmark_against_existing_ccm() -> None:
    from vibeqc.periodic.ccm import CCMSystem
    from vibeqc.periodic.ccm.scf import run_ccm_rhf

    angstrom_to_bohr = 1.0 / 0.529177210903
    positions = [
        [x * angstrom_to_bohr, 0.0, 0.0] for x in (0.0, 0.8, 2.0, 2.8)
    ]
    system = vq.PeriodicSystem(
        3,
        np.diag([4.0 * angstrom_to_bohr, 40.0, 40.0]),
        [vq.Atom(1, position) for position in positions],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    options = vq.PeriodicRHFOptions()
    options.max_iter = 40
    options.conv_tol_energy = 1e-8
    options.damping = 0.0
    independent = run_aiccm2026dev_b_rhf(
        system,
        basis,
        (2, 1, 1),
        options,
        backend="ri",
        progress=False,
    )
    historical = run_ccm_rhf(
        CCMSystem(system, (2, 1, 1), "sto-3g"),
        method="union12",
        max_iter=40,
        conv_tol=1e-8,
    )
    peer = run_ccm_rhf(
        CCMSystem(system, (2, 1, 1), "sto-3g"),
        method="aiccm2026dev-a",
        max_iter=40,
        conv_tol=1e-8,
    )
    historical_energy_per_cell = historical.energy_per_atom * len(system.unit_cell)
    peer_energy_per_cell = peer.energy_per_atom * len(system.unit_cell)

    assert independent.converged
    assert historical.converged
    assert peer.converged
    assert independent.energy == pytest.approx(-2.1648043358, abs=5e-8)
    assert independent.aiccm2026dev_b.density_idempotency_error < 1e-7
    # Agreement is intentionally a weak comparison signal, not the theory
    # gate. It is nevertheless useful for catching a gross head-to-head drift.
    assert abs(independent.energy - historical_energy_per_cell) < 5e-3
    assert abs(independent.energy - peer_energy_per_cell) < 5e-3
