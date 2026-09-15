"""M1 tests: primitive-cell reduction via spglib.

Validates:
  - MgO rocksalt conventional (8 atoms) → primitive (2 atoms)
  - Si diamond conventional (8 atoms) → primitive (2 atoms)
  - Odd-Z FCC metals retain valid SCF multiplicity after reduction
  - Already-primitive cell raises ValueError
  - The reduced cell runs through the periodic runner without error
  - Finite-Gamma conventional / primitive energies stay close on a per-atom
    basis without asserting exact extensivity at different folded k meshes
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
import vibeqc.periodic_runner as periodic_runner_module
from vibeqc.periodic_runner import (
    _basis_summary,
    _build_primitive_summary,
    _remap_kpoints_after_primitive_reduction,
    _reduce_system_to_primitive,
    _system_with_valid_default_multiplicity,
    run_periodic_job,
)

# ---------------------------------------------------------------------------
# Helper: build MgO conventional rocksalt cell
# ---------------------------------------------------------------------------


def _build_mgo_conventional(a_bohr: float = 4.0) -> vq.PeriodicSystem:
    """MgO rocksalt conventional cubic cell: 8 atoms (4 Mg + 4 O)."""
    a = float(a_bohr)
    atoms = [
        vq.Atom(12, [0, 0, 0]),
        vq.Atom(12, [a / 2, a / 2, 0]),
        vq.Atom(12, [a / 2, 0, a / 2]),
        vq.Atom(12, [0, a / 2, a / 2]),
        vq.Atom(8, [a / 2, 0, 0]),
        vq.Atom(8, [0, a / 2, 0]),
        vq.Atom(8, [0, 0, a / 2]),
        vq.Atom(8, [a / 2, a / 2, a / 2]),
    ]
    return vq.PeriodicSystem(3, np.eye(3) * a, atoms)


def _build_si_conventional(a_bohr: float = 10.263) -> vq.PeriodicSystem:
    """Si diamond conventional cubic cell: 8 Si atoms, Fd-3m."""
    a = float(a_bohr)
    frac = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
            [0.25, 0.25, 0.25],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
        ]
    )
    L = np.eye(3) * a
    atoms = [vq.Atom(14, (L @ f).tolist()) for f in frac]
    return vq.PeriodicSystem(3, L, atoms)


def _build_fcc_metal_conventional(
    atomic_number: int,
    a_bohr: float = 8.0,
) -> vq.PeriodicSystem:
    """Conventional FCC cell with four equivalent metal atoms."""
    a = float(a_bohr)
    atoms = [
        vq.Atom(atomic_number, [0, 0, 0]),
        vq.Atom(atomic_number, [0, a / 2, a / 2]),
        vq.Atom(atomic_number, [a / 2, 0, a / 2]),
        vq.Atom(atomic_number, [a / 2, a / 2, 0]),
    ]
    return vq.PeriodicSystem(3, np.eye(3) * a, atoms)


def _build_mgo_primitive(a_bohr: float = 4.0) -> vq.PeriodicSystem:
    """Simple-cubic Pm-3m MgO cell used by already-primitive tests.

    This is not the Fm-3m rocksalt primitive produced by reducing
    _build_mgo_conventional().
    """
    a = float(a_bohr)
    atoms = [
        vq.Atom(12, [0, 0, 0]),
        vq.Atom(8, [a / 2, a / 2, a / 2]),
    ]
    return vq.PeriodicSystem(3, np.eye(3) * a, atoms)


# ---------------------------------------------------------------------------
# Reduction tests
# ---------------------------------------------------------------------------


def test_mgo_conventional_to_primitive():
    """MgO: 8 atoms → 2 atoms, Fm-3m, 192 ops."""
    sys = _build_mgo_conventional()
    prim, sg = _reduce_system_to_primitive(sys, symprec=1e-4)
    assert len(prim.unit_cell) == 2
    assert len(sys.unit_cell) == 8
    assert sg.international_symbol == "Fm-3m"
    assert sg.number == 225
    assert sg.order == 192
    # Two unique atom types
    assert len(set(sg.equivalent_atoms)) == 2
    # Primitive cell also has symmetry attached
    assert prim.symmetry is not None
    assert prim.symmetry.international_symbol == "Fm-3m"


def test_si_conventional_to_primitive():
    """Si: 8 atoms → 2 atoms, Fd-3m."""
    sys = _build_si_conventional()
    prim, sg = _reduce_system_to_primitive(sys, symprec=1e-4)
    assert len(prim.unit_cell) == 2
    assert sg.international_symbol == "Fd-3m"
    assert sg.number == 227
    # All 8 Si atoms are equivalent
    assert len(set(sg.equivalent_atoms)) == 1


@pytest.mark.parametrize("atomic_number", [29, 47, 79], ids=["Cu", "Ag", "Au"])
def test_odd_z_metal_reduction_keeps_scf_multiplicity_valid(atomic_number):
    """Reduction must update the SCF system, not only its basis molecule."""
    conventional = _build_fcc_metal_conventional(atomic_number)
    assert conventional.n_electrons() == 4 * atomic_number
    assert conventional.multiplicity == 1

    primitive, _ = _reduce_system_to_primitive(conventional, symprec=1e-4)
    n_electrons = primitive.n_electrons()

    assert len(primitive.unit_cell) == 1
    assert n_electrons == atomic_number
    assert primitive.multiplicity == 2
    assert (n_electrons - primitive.multiplicity + 1) % 2 == 0
    assert primitive.unit_cell_molecule().multiplicity == primitive.multiplicity
    assert conventional.multiplicity == 1


@pytest.mark.parametrize("atomic_number", [29, 47, 79], ids=["Cu", "Ag", "Au"])
def test_odd_z_primitive_cell_gets_scf_valid_default_multiplicity(atomic_number):
    """Already-primitive odd-electron cells need the same SCF correction."""
    original = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [vq.Atom(atomic_number, [0, 0, 0])],
    )

    normalized = _system_with_valid_default_multiplicity(original)

    assert normalized is not original
    assert original.multiplicity == 1
    assert normalized.multiplicity == 2
    assert (
        normalized.n_electrons() - normalized.multiplicity + 1
    ) % 2 == 0


def test_primitive_reduction_preserves_explicit_multiplicity():
    """A valid user-selected magnetic multiplicity is not auto-replaced."""
    conventional = _build_fcc_metal_conventional(12)
    conventional.multiplicity = 3

    primitive, _ = _reduce_system_to_primitive(conventional, symprec=1e-4)

    assert primitive.multiplicity == 3
    assert (primitive.n_electrons() - primitive.multiplicity + 1) % 2 == 0


def test_runner_normalizes_primitive_odd_z_before_scf(monkeypatch, tmp_path):
    """The high-level runner must pass the corrected system to GDF SCF."""
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [vq.Atom(29, [0, 0, 0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    dispatched_systems = []

    class ReachedGDFDriver(Exception):
        pass

    def stop_at_gdf_driver(candidate, *args, **kwargs):
        dispatched_systems.append(candidate)
        raise ReachedGDFDriver

    monkeypatch.setattr(
        periodic_runner_module,
        "run_pbc_gdf_uhf",
        stop_at_gdf_driver,
    )

    with pytest.raises(ReachedGDFDriver):
        periodic_runner_module.run_periodic_job(
            system,
            basis,
            method="UHF",
            jk_method="gdf",
            output=tmp_path / "cu_primitive",
            output_qvf=False,
            write_molden_file=False,
            write_xyz_file=False,
            write_poscar_file=False,
            write_xsf_structure_file=False,
            write_cif_file=False,
            write_population_file=False,
            citations=False,
            progress=False,
        )

    assert len(dispatched_systems) == 1
    scf_system = dispatched_systems[0]
    assert scf_system.multiplicity == 2
    assert (scf_system.n_electrons() - scf_system.multiplicity + 1) % 2 == 0
    assert system.multiplicity == 1


def test_runner_rechecks_restricted_parity_after_reduction(tmp_path):
    """An even conventional RHF cell cannot hide an odd primitive cell."""
    system = _build_fcc_metal_conventional(29)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")

    with pytest.raises(ValueError, match="odd electron count.*UHF"):
        periodic_runner_module.run_periodic_job(
            system,
            basis,
            method="RHF",
            jk_method="gdf",
            reduce_to_primitive=True,
            dry_run=True,
            output=tmp_path / "cu_reduced_rhf",
        )


def test_already_primitive_raises():
    """Primitive cell raises ValueError with a helpful message."""
    sys = _build_mgo_primitive()
    with pytest.raises(ValueError, match="already primitive"):
        _reduce_system_to_primitive(sys, symprec=1e-4)


def test_already_primitive_mentions_space_group():
    """Error message includes the detected space group."""
    sys = _build_mgo_primitive()
    with pytest.raises(ValueError, match="Pm-3m"):
        _reduce_system_to_primitive(sys, symprec=1e-4)


def test_primitive_volume_ratio():
    """Primitive cell has 1/4 the volume of conventional FCC."""
    sys = _build_mgo_conventional()
    prim, _ = _reduce_system_to_primitive(sys, symprec=1e-4)
    V_in = float(abs(np.linalg.det(np.asarray(sys.lattice, dtype=float))))
    V_out = float(abs(np.linalg.det(np.asarray(prim.lattice, dtype=float))))
    assert V_out == pytest.approx(V_in / 4.0, rel=1e-10)


def test_primitive_reduction_remaps_object_monkhorst_pack_mesh():
    """An MP object must be rebuilt on the primitive reciprocal lattice."""
    conventional = _build_mgo_conventional()
    primitive, _ = _reduce_system_to_primitive(conventional, symprec=1e-4)
    requested = vq.KPoints.monkhorst_pack(
        conventional,
        (2, 1, 1),
        shift=(0, 0, 0),
    )
    requested.smearing = vq.SmearingOptions(temperature=0.005)

    remapped = _remap_kpoints_after_primitive_reduction(
        conventional,
        primitive,
        requested,
    )
    expected = vq.KPoints.monkhorst_pack(
        primitive,
        (2, 1, 1),
        shift=(0, 0, 0),
    )

    assert remapped._system is primitive
    assert remapped.smearing is requested.smearing
    assert np.allclose(remapped.kpoints_frac, expected.kpoints_frac)
    assert np.allclose(remapped.kpoints_cart, expected.kpoints_cart)
    assert not np.allclose(remapped.kpoints_cart, requested.kpoints_cart)


def test_primitive_reduction_rejects_explicit_kpoint_object():
    """An arbitrary explicit list has no unique primitive-cell remapping."""
    conventional = _build_mgo_conventional()
    primitive, _ = _reduce_system_to_primitive(conventional, symprec=1e-4)
    requested = vq.KPoints.from_list(conventional, [[0.25, 0.0, 0.0]])

    with pytest.raises(NotImplementedError, match="Monkhorst-Pack"):
        _remap_kpoints_after_primitive_reduction(
            conventional,
            primitive,
            requested,
        )


def test_runner_rejects_dft_plus_u_before_primitive_site_remapping(tmp_path):
    """Input-cell Hubbard indices cannot silently target primitive atoms."""
    conventional = _build_mgo_conventional()
    basis = vq.BasisSet(conventional.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "mgo-primitive-plus-u"

    with pytest.raises(NotImplementedError, match="Hubbard atom indices"):
        vq.run_periodic_job(
            conventional,
            basis,
            method="RKS",
            functional="lda",
            jk_method="bipole",
            dft_plus_u=[vq.HubbardSite(0, 1, U_ev=2.0)],
            reduce_to_primitive=True,
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


@pytest.mark.parametrize("method", ["UHF", "UKS"])
def test_runner_rejects_atomic_spin_tags_before_primitive_remapping(
    tmp_path,
    method,
):
    conventional = _build_mgo_conventional()
    conventional.multiplicity = 3
    basis = vq.BasisSet(conventional.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / f"mgo-primitive-{method.lower()}-atomspin"

    with pytest.raises(NotImplementedError, match="atomic_spins.*primitive"):
        vq.run_periodic_job(
            conventional,
            basis,
            method=method,
            functional="lda" if method == "UKS" else None,
            jk_method="bipole",
            atomic_spins=[1, 1, 1, 1, -1, -1, -1, -1],
            reduce_to_primitive=True,
            dry_run=True,
            output=stem,
            output_qvf=False,
            progress=False,
        )
    assert not stem.with_suffix(".system").exists()


def test_basis_function_count():
    """Basis size scales with atom count."""
    sys = _build_mgo_conventional()
    prim, _ = _reduce_system_to_primitive(sys, symprec=1e-4)
    basis_in = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    basis_out = vq.BasisSet(prim.unit_cell_molecule(), "sto-3g")
    # 8 atoms vs 2 atoms: 7 bf per atom (Mg: 5, O: 5 for STO-3G) → 56 vs 14
    assert basis_in.nbasis == 56
    assert basis_out.nbasis == 14


def test_primitive_summary_contains_space_group():
    sys_in = _build_mgo_conventional()
    prim, sg = _reduce_system_to_primitive(sys_in, symprec=1e-4)
    text = _build_primitive_summary(sys_in, prim, sg)
    assert "Fm-3m" in text
    assert "No. 225" in text
    assert "m-3m" in text
    assert ("4.0x fewer atoms" in text) or ("4.0× fewer atoms" in text)
    assert "inequivalent" in text
    lines = text.splitlines()
    assert len(lines[1]) == max(len(line) for line in lines[2:])


def test_periodic_basis_summary_rule_sizes_to_rows():
    basis = type(
        "Basis",
        (),
        {"name": "a-long-basis-name", "nbasis": 12, "nshells": 7},
    )()
    lines = _basis_summary(basis).splitlines()
    assert lines[0] == "  Basis"
    assert lines[2] == "    name    = a-long-basis-name"
    assert len(lines[1]) == max(len(line) for line in lines[2:])


# ---------------------------------------------------------------------------
# Integration with run_periodic_job (GDF path for speed)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_reduce_to_primitive_via_runner_mgo_gdf(tmp_path):
    """run_periodic_job with reduce_to_primitive=True dispatches
    correctly and produces a convergent energy on the primitive cell."""
    sys = _build_mgo_conventional(a_bohr=7.9)
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    result = run_periodic_job(
        sys,
        basis,
        method="RHF",
        jk_method="gdf",
        output=str(tmp_path / "test_mgo_prim_gdf"),
        max_iter=60,
        conv_tol_energy=1e-6,
        reduce_to_primitive=True,
        symmetry_precision=1e-4,
        write_molden_file=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        citations=False,
    )
    assert result.converged
    assert result.n_iter >= 1
    # Energy should be physically reasonable for the physical MgO lattice.
    assert -280.0 < result.energy < -260.0


@pytest.mark.slow
def test_reduce_to_primitive_same_as_no_reduce_ratio(tmp_path):
    """Gamma-only conventional and primitive cells are close per atom.

    The conventional Γ cell folds four primitive k points, so exact
    E_conv == 4 * E_prim(Gamma) is not a valid finite-size assertion.
    """
    sys = _build_mgo_conventional(a_bohr=5.0)  # larger cell = faster GDF
    prim, _ = _reduce_system_to_primitive(sys, symprec=1e-4)
    atom_ratio = float(len(sys.unit_cell) / len(prim.unit_cell))
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")

    # Without reduction (conventional, 8 atoms)
    r_conv = run_periodic_job(
        sys,
        basis,
        method="RHF",
        jk_method="gdf",
        output=str(tmp_path / "test_mgo_ratio_conv"),
        max_iter=30,
        conv_tol_energy=1e-6,
        reduce_to_primitive=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        citations=False,
    )

    # With reduction (primitive, 2 atoms)
    r_prim = run_periodic_job(
        sys,
        basis,
        method="RHF",
        jk_method="gdf",
        output=str(tmp_path / "test_mgo_ratio_prim"),
        max_iter=30,
        conv_tol_energy=1e-6,
        reduce_to_primitive=True,
        write_molden_file=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        citations=False,
    )

    assert r_conv.converged
    assert r_prim.converged
    assert (r_conv.energy / atom_ratio) == pytest.approx(
        r_prim.energy,
        rel=5e-3,
    )


@pytest.mark.slow
def test_reduce_to_primitive_false_is_noop(tmp_path):
    """With reduce_to_primitive=False, the standard path is unchanged."""
    sys = _build_mgo_primitive()
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    result = run_periodic_job(
        sys,
        basis,
        method="RHF",
        jk_method="gdf",
        output=str(tmp_path / "test_mgo_noop"),
        max_iter=5,
        conv_tol_energy=1e-3,
        reduce_to_primitive=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        citations=False,
    )
    # Should run without error; energy doesn't matter for this test
    assert result.n_iter >= 1


@pytest.mark.slow
def test_m1_plus_m2a_integration(tmp_path):
    """M1 (primitive reduction) + M2a (auto reduced S/T) work together.

    Creates a conventional MgO cell, runs via GDF with
    reduce_to_primitive=True, and verifies the result matches a
    manually-built primitive cell computation.
    """
    sys_conv = _build_mgo_conventional(a_bohr=5.0)
    basis_conv = vq.BasisSet(sys_conv.unit_cell_molecule(), "sto-3g")

    # Run with M1+M2a: reduce to primitive, auto-use reduced S/T
    r1 = run_periodic_job(
        sys_conv,
        basis_conv,
        method="RHF",
        jk_method="gdf",
        output=str(tmp_path / "test_m1m2a_integration"),
        max_iter=30,
        conv_tol_energy=1e-6,
        reduce_to_primitive=True,
        symmetry_precision=1e-4,
        write_molden_file=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        citations=False,
    )

    # Run the same manually: use the exact primitive that the reducer returns.
    sys_prim, _ = _reduce_system_to_primitive(sys_conv, symprec=1e-4)
    basis_prim = vq.BasisSet(sys_prim.unit_cell_molecule(), "sto-3g")
    r2 = run_periodic_job(
        sys_prim,
        basis_prim,
        method="RHF",
        jk_method="gdf",
        output=str(tmp_path / "test_m1m2a_manual"),
        max_iter=30,
        conv_tol_energy=1e-6,
        reduce_to_primitive=False,
        write_molden_file=False,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=False,
        citations=False,
    )

    assert r1.converged
    assert r2.converged
    # Both should produce the same energy (both run on primitive cell)
    assert abs(r1.energy - r2.energy) < 1e-8, (
        f"M1+M2a energy {r1.energy:.10f} differs from manual "
        f"{r2.energy:.10f} by {abs(r1.energy - r2.energy):.2e}"
    )
