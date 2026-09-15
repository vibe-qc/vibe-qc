"""COOP/COHP compute + QVF writer smoke tests.

Pins the physics (bonding/antibonding peak positions, sign conventions,
E_F integration) and the plumbing (serialization round-trip, schema
validation).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.output.formats.qvf import validate_qvf, write_qvf
from vibeqc.output.plan import OutputPlan

# Load the canonical schema for schema-drift checks
_SCHEMA_PATH = (
    Path(__file__).resolve().parent.parent
    / "python"
    / "vibeqc"
    / "output"
    / "formats"
    / "qvf_manifest_v2.schema.json"
)
with open(_SCHEMA_PATH, encoding="utf-8") as _f:
    _SCHEMA = json.load(_f)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def test_periodic_mayer_singlet_and_nonorthogonal_ao_rescaling():
    from vibeqc.coop_cohp import _periodic_mayer_from_density

    overlap = np.array([[1., .3], [.3, 1.]])
    c = np.array([1., .4])
    c /= np.sqrt(c @ overlap @ c)
    density = 2. * np.outer(c, c)
    ps = density @ overlap
    assert not np.allclose(ps, ps.T)
    expected = ps[0, 1] * ps[1, 0]
    restricted = _periodic_mayer_from_density([[density]], [overlap], [1.], [0, 1], 2)
    unrestricted = _periodic_mayer_from_density(
        [[density / 2], [density / 2]], [overlap], [1.], [0, 1], 2,
    )
    assert restricted[0, 1] == pytest.approx(expected, abs=1e-12)
    np.testing.assert_allclose(unrestricted, restricted, atol=1e-12, rtol=0)
    scaling = np.diag([.7, 1.8])
    inverse = np.linalg.inv(scaling)
    rescaled = _periodic_mayer_from_density(
        [[inverse @ density @ inverse]], [scaling @ overlap @ scaling], [1.], [0, 1], 2,
    )
    np.testing.assert_allclose(rescaled, restricted, atol=1e-12, rtol=0)
    fractional = _periodic_mayer_from_density(
        [[.6 * density]], [overlap], [1.], [0, 1], 2,
    )
    np.testing.assert_allclose(fractional, .36 * restricted, atol=1e-12, rtol=0)


def _h2_chain_1d():
    """H₂-per-cell 1D chain with 6-bohr cell, 1.4 bohr bond."""
    system = vq.PeriodicSystem(
        1,
        [[6.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]],
        [vq.Atom(1, [0.0, 15.0, 15.0]), vq.Atom(1, [1.4, 15.0, 15.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


# ---------------------------------------------------------------------------
# Grouping helper
# ---------------------------------------------------------------------------


def test_ao_pairs_per_atom_pair_h2():
    system, basis = _h2_chain_1d()
    pairs = vq.ao_pairs_per_atom_pair(system, basis)
    # STO-3G H = 1 s-shell per atom → (0, 1) pair with 1 AO each
    assert len(pairs) == 1
    assert (0, 1) in pairs
    ao_a, ao_b = pairs[(0, 1)]
    assert ao_a == [0]
    assert ao_b == [1]


def test_ao_pairs_per_atom_pair_cutoff():
    """Distant atoms are excluded by cutoff."""
    system = vq.PeriodicSystem(
        1,
        [[20.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]],
        [vq.Atom(1, [0.0, 15.0, 15.0]), vq.Atom(1, [15.0, 15.0, 15.0])],  # far apart
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    pairs_default = vq.ao_pairs_per_atom_pair(system, basis)
    assert len(pairs_default) == 0
    # With a large cutoff, they now appear
    pairs_large = vq.ao_pairs_per_atom_pair(system, basis, pair_distance_cutoff=20.0)
    assert len(pairs_large) == 1


# ---------------------------------------------------------------------------
# COOP/COHP compute tests
# ---------------------------------------------------------------------------


def test_coop_bonding_antibonding_h2():
    """1D H2 chain: COOP should show bonding below E_F and antibonding above."""
    system, basis = _h2_chain_1d()
    # Use Hcore (non-interacting) Fock so we get clean bonding/antibonding
    from vibeqc._vibeqc_core import (
        LatticeSumOptions,
        compute_kinetic_lattice,
        compute_nuclear_lattice,
        compute_overlap_lattice,
        monkhorst_pack,
    )

    opts = LatticeSumOptions()
    S = compute_overlap_lattice(basis, system, opts)
    T = compute_kinetic_lattice(basis, system, opts)
    V = compute_nuclear_lattice(basis, system, opts)
    km = monkhorst_pack(system, [8, 1, 1])

    result = vq.compute_coop_cohp(
        [T, V],  # Hcore = T + V
        S,
        system,
        basis,
        km,
        H_terms=[T, V],  # full Hcore for COHP
        sigma=0.05,
        n_grid=201,
        n_electrons_per_cell=2,
    )

    assert result.energies.shape == (201,)
    assert result.coop.shape == (1, 201)  # one H-H pair
    assert result.cohp is not None
    assert result.cohp.shape == (1, 201)
    assert result.integrated_coop.shape == (1,)
    assert result.integrated_cohp is not None
    assert result.integrated_cohp.shape == (1,)
    assert len(result.pairs) == 1
    assert result.pairs[0]["symbol_i"] == "H"
    assert result.pairs[0]["symbol_j"] == "H"

    # COOP should have both positive (bonding) and negative (antibonding) regions
    coop_max = float(result.coop[0].max())
    coop_min = float(result.coop[0].min())
    assert coop_max > 0, f"COOP should have positive values, max={coop_max}"
    assert coop_min < 0, f"COOP should have negative values, min={coop_min}"

    # -COHP should also have both signs
    cohp_max = float(result.cohp[0].max())
    cohp_min = float(result.cohp[0].min())
    assert cohp_max > 0, f"-COHP should have positive values, max={cohp_max}"
    assert cohp_min < 0, f"-COHP should have negative values, min={cohp_min}"

    # ICOHP should be positive (net bonding)
    assert float(result.integrated_cohp[0]) > 0, (
        f"Expected positive integrated -COHP, got {result.integrated_cohp[0]}"
    )


def test_coop_no_h_real_skips_cohp():
    """When H_real is None, cohp is None."""
    system, basis = _h2_chain_1d()
    from vibeqc._vibeqc_core import (
        LatticeSumOptions,
        compute_kinetic_lattice,
        compute_overlap_lattice,
        monkhorst_pack,
    )

    opts = LatticeSumOptions()
    S = compute_overlap_lattice(basis, system, opts)
    T = compute_kinetic_lattice(basis, system, opts)
    km = monkhorst_pack(system, [4, 1, 1])

    result = vq.compute_coop_cohp(
        T,
        S,
        system,
        basis,
        km,
        sigma=0.1,
        n_grid=101,
    )
    assert result.coop is not None
    assert result.cohp is None
    assert result.integrated_cohp is None


def test_coop_no_pairs_raises():
    """A system with no pairs within cutoff raises ValueError."""
    system = vq.PeriodicSystem(
        1,
        [[30.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]],
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [25.0, 0.0, 0.0])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    from vibeqc._vibeqc_core import (
        LatticeSumOptions,
        compute_kinetic_lattice,
        compute_overlap_lattice,
        monkhorst_pack,
    )

    opts = LatticeSumOptions()
    S = compute_overlap_lattice(basis, system, opts)
    T = compute_kinetic_lattice(basis, system, opts)
    km = monkhorst_pack(system, [2, 1, 1])

    with pytest.raises(ValueError, match="no atom pairs"):
        vq.compute_coop_cohp(T, S, system, basis, km)


def test_coop_spin_polarized_shapes():
    """When F_terms_beta is given, result arrays have spin dimension."""
    system, basis = _h2_chain_1d()
    from vibeqc._vibeqc_core import (
        LatticeSumOptions,
        compute_kinetic_lattice,
        compute_overlap_lattice,
        monkhorst_pack,
    )

    opts = LatticeSumOptions()
    S = compute_overlap_lattice(basis, system, opts)
    T = compute_kinetic_lattice(basis, system, opts)
    km = monkhorst_pack(system, [4, 1, 1])

    # Pass same T for both spins — unrealistic but tests the shape contract
    result = vq.compute_coop_cohp(
        T,
        S,
        system,
        basis,
        km,
        F_terms_beta=T,
        sigma=0.1,
        n_grid=51,
    )
    assert result.coop.shape == (2, 1, 51)  # (n_spin, n_pairs, n_e)
    assert result.integrated_coop.shape == (2, 1)
    assert result.cohp is None  # H_terms not given

    # With H_terms
    result2 = vq.compute_coop_cohp(
        T,
        S,
        system,
        basis,
        km,
        F_terms_beta=T,
        H_terms=T,
        sigma=0.1,
        n_grid=51,
    )
    assert result2.cohp.shape == (2, 1, 51)
    assert result2.integrated_cohp.shape == (2, 1)


# ---------------------------------------------------------------------------
# QVF writer: dos.coop
# ---------------------------------------------------------------------------


def _make_plan(tmp_path: Path) -> OutputPlan:
    """Minimal output plan for writer tests."""
    return OutputPlan.from_run_job_kwargs(
        output=tmp_path / "test",
        method="rhf",
        basis="sto-3g",
        functional=None,
        job_kind="periodic_scf",
    )


def _make_synthetic_coop_data():
    """Create synthetic COOP data matching the writer contract."""
    n_pairs, n_points = 3, 101
    energies = np.linspace(-10.0, 5.0, n_points, dtype=np.float64)  # eV
    # COOP: bonding peak at -5 eV, antibonding at +3 eV
    proj = np.zeros((n_pairs, n_points), dtype=np.float64)
    for p in range(n_pairs):
        # Bonding (positive Gaussian)
        proj[p] += 3.0 * np.exp(-0.5 * ((energies + 5.0) / 0.5) ** 2)
        # Antibonding (negative Gaussian)
        proj[p] -= 2.0 * np.exp(-0.5 * ((energies - 3.0) / 0.5) ** 2)

    integrated = np.trapezoid(
        proj[:, energies <= 0.0], energies[: np.sum(energies <= 0.0)], axis=1
    )

    pairs = [
        {"i": 0, "j": 1, "symbol_i": "Si", "symbol_j": "Si", "distance_ang": 2.35},
        {"i": 0, "j": 2, "symbol_i": "Si", "symbol_j": "O", "distance_ang": 1.62},
        {"i": 1, "j": 2, "symbol_i": "Si", "symbol_j": "O", "distance_ang": 1.62},
    ]

    return {
        "energies": energies,
        "projections": proj,
        "integrated": integrated,
        "energies_units": "eV",
        "n_spin": 1,
        "fermi_energy_ev": 0.0,
        "sigma_ev": 0.27,
        "pairs": pairs,
    }


def test_write_coop_roundtrip(tmp_path):
    """dos.coop writes, validates, and round-trips."""
    plan = _make_plan(tmp_path)
    coop_data = _make_synthetic_coop_data()
    path = write_qvf(tmp_path / "test", plan, coop_data=coop_data)
    assert path.suffix == ".qvf"
    assert path.is_file()

    report = validate_qvf(path)
    assert report["valid"], f"Validation errors: {report.get('errors', [])}"

    # Load manifest from zip for section inspection
    import zipfile as _zf

    with _zf.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
    sections = {s["kind"]: s for s in manifest["sections"]}
    assert "dos.coop" in sections
    sec = sections["dos.coop"]
    assert sec["id"] == "coop0"
    assert sec["n_spin"] == 1
    assert sec["fermi_energy_ev"] == pytest.approx(0.0)
    assert sec["sigma_ev"] == pytest.approx(0.27)
    assert len(sec["pairs"]) == 3
    assert sec["pairs"][0]["symbol_i"] == "Si"


# ---------------------------------------------------------------------------
# QVF writer: dos.cohp
# ---------------------------------------------------------------------------


def _make_synthetic_cohp_data():
    """Create synthetic -COHP data matching the writer contract."""
    n_pairs, n_points = 2, 101
    energies = np.linspace(-10.0, 5.0, n_points, dtype=np.float64)
    proj = np.zeros((n_pairs, n_points), dtype=np.float64)
    for p in range(n_pairs):
        proj[p] += 4.0 * np.exp(-0.5 * ((energies + 4.0) / 0.4) ** 2)
        proj[p] -= 3.0 * np.exp(-0.5 * ((energies - 2.0) / 0.4) ** 2)

    integrated = np.trapezoid(
        proj[:, energies <= 0.0], energies[: np.sum(energies <= 0.0)], axis=1
    )

    pairs = [
        {"i": 0, "j": 1, "symbol_i": "C", "symbol_j": "C", "distance_ang": 1.42},
        {"i": 0, "j": 2, "symbol_i": "C", "symbol_j": "H", "distance_ang": 1.09},
    ]

    return {
        "energies": energies,
        "projections": proj,
        "integrated": integrated,
        "energies_units": "eV",
        "n_spin": 1,
        "fermi_energy_ev": 0.0,
        "sigma_ev": 0.27,
        "pairs": pairs,
    }


def test_write_cohp_roundtrip(tmp_path):
    """dos.cohp writes, validates, and round-trips."""
    plan = _make_plan(tmp_path)
    cohp_data = _make_synthetic_cohp_data()
    path = write_qvf(tmp_path / "test", plan, cohp_data=cohp_data)
    assert path.suffix == ".qvf"

    report = validate_qvf(path)
    assert report["valid"], f"Validation errors: {report.get('errors', [])}"

    import zipfile as _zf

    with _zf.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
    sections = {s["kind"]: s for s in manifest["sections"]}
    assert "dos.cohp" in sections
    sec = sections["dos.cohp"]
    assert sec["id"] == "cohp0"
    assert sec["n_spin"] == 1
    assert len(sec["pairs"]) == 2


def test_write_coop_and_cohp_together(tmp_path):
    """Both sections coexist in the same archive."""
    plan = _make_plan(tmp_path)
    coop_data = _make_synthetic_coop_data()
    cohp_data = _make_synthetic_cohp_data()
    path = write_qvf(
        tmp_path / "test",
        plan,
        coop_data=coop_data,
        cohp_data=cohp_data,
    )
    report = validate_qvf(path)
    assert report["valid"], f"Validation errors: {report.get('errors', [])}"
    import zipfile as _zf

    with _zf.ZipFile(path, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
    sections = {s["kind"]: s for s in manifest["sections"]}
    assert "dos.coop" in sections
    assert "dos.cohp" in sections


def test_coop_writer_rejects_wrong_shape(tmp_path):
    """Shape validation in the writer."""
    plan = _make_plan(tmp_path)
    bad_data = {
        "energies": np.linspace(-10, 5, 101, dtype=np.float64),
        "projections": np.zeros((101,), dtype=np.float64),  # should be (n_pairs, n_e)
        "integrated": np.zeros(3, dtype=np.float64),
        "pairs": [
            {"i": 0, "j": 1, "symbol_i": "A", "symbol_j": "B", "distance_ang": 1.5}
        ],
    }
    with pytest.raises(ValueError, match="dos.coop"):
        write_qvf(tmp_path / "test", plan, coop_data=bad_data)


def test_coop_writer_spin_polarized_shape(tmp_path):
    """Spin-polarized projections are accepted."""
    plan = _make_plan(tmp_path)
    n_pairs, n_points = 2, 51
    data = {
        "energies": np.linspace(-10, 5, n_points, dtype=np.float64),
        "projections": np.zeros((2, n_pairs, n_points), dtype=np.float64),
        "integrated": np.zeros((2, n_pairs), dtype=np.float64),
        "energies_units": "eV",
        "n_spin": 2,
        "fermi_energy_ev": 0.0,
        "sigma_ev": 0.27,
        "pairs": [
            {"i": 0, "j": 1, "symbol_i": "Fe", "symbol_j": "Fe", "distance_ang": 2.5},
            {"i": 0, "j": 2, "symbol_i": "Fe", "symbol_j": "O", "distance_ang": 2.0},
        ],
    }
    path = write_qvf(tmp_path / "test", plan, coop_data=data)
    report = validate_qvf(path)
    assert report["valid"], f"Validation errors: {report.get('errors', [])}"


def test_spin_polarized_bad_shape_raises(tmp_path):
    """Spin-polarized with wrong shape raises."""
    plan = _make_plan(tmp_path)
    bad = {
        "energies": np.linspace(-10, 5, 51, dtype=np.float64),
        "projections": np.zeros((3, 2, 51), dtype=np.float64),  # 3 spins!
        "integrated": np.zeros(3, dtype=np.float64),
        "n_spin": 2,
        "pairs": [],
    }
    with pytest.raises(ValueError, match="n_spin=2"):
        write_qvf(tmp_path / "test", plan, coop_data=bad)

    # Bad integrated shape for n_spin=2
    bad_int = {
        "energies": np.linspace(-10, 5, 51, dtype=np.float64),
        "projections": np.zeros((2, 2, 51), dtype=np.float64),
        "integrated": np.zeros(2, dtype=np.float64),  # should be (2, 2)
        "n_spin": 2,
        "pairs": [
            {"i": 0, "j": 1, "symbol_i": "A", "symbol_j": "B", "distance_ang": 1.5},
            {"i": 0, "j": 2, "symbol_i": "A", "symbol_j": "C", "distance_ang": 2.0},
        ],
    }
    with pytest.raises(ValueError, match="integrated must be"):
        write_qvf(tmp_path / "test", plan, coop_data=bad_int)


# ---------------------------------------------------------------------------
# Schema drift guard
# ---------------------------------------------------------------------------


def test_kinds_in_schema_and_implemented():
    """dos.coop and dos.cohp must be in both _IMPLEMENTED_KINDS and schema."""
    from vibeqc.output.formats.qvf import _IMPLEMENTED_KINDS

    assert "dos.coop" in _IMPLEMENTED_KINDS, "dos.coop must be in _IMPLEMENTED_KINDS"
    assert "dos.cohp" in _IMPLEMENTED_KINDS, "dos.cohp must be in _IMPLEMENTED_KINDS"

    # Check schema has the const entries
    section_kinds = []
    for defn in _SCHEMA["$defs"].values():
        if isinstance(defn, dict) and "properties" in defn:
            kind_prop = defn["properties"].get("kind", {})
            if "const" in kind_prop:
                section_kinds.append(kind_prop["const"])

    assert "dos.coop" in section_kinds, (
        "dos.coop must be in schema $defs as SectionDOSCOOP"
    )
    assert "dos.cohp" in section_kinds, (
        "dos.cohp must be in schema $defs as SectionDOSCOHP"
    )

    # Check oneOf
    one_of_refs = [
        ref["$ref"].split("/")[-1] for ref in _SCHEMA["$defs"]["Section"]["oneOf"]
    ]
    assert "SectionDOSCOOP" in one_of_refs
    assert "SectionDOSCOHP" in one_of_refs


# ---------------------------------------------------------------------------
# Plot smoke tests
# ---------------------------------------------------------------------------


def test_coop_figure_smoke():
    """coop_figure runs without error on synthetic data."""
    from vibeqc.plot import coop_figure

    result = _synthetic_coop_result()
    fig = coop_figure(result)
    assert fig is not None


def test_cohp_figure_smoke():
    """cohp_figure runs without error on synthetic data."""
    from vibeqc.plot import cohp_figure

    result = _synthetic_cohp_result()
    fig = cohp_figure(result)
    assert fig is not None


def test_cohp_figure_raises_when_cohp_is_none():
    """cohp_figure raises ValueError when result.cohp is None."""
    from vibeqc.plot import cohp_figure

    result = _synthetic_coop_result()  # cohp is None
    with pytest.raises(ValueError, match="result.cohp is None"):
        cohp_figure(result)


def _synthetic_coop_result():
    """Minimal COOPCOHPResult for plot smoke testing."""
    from vibeqc.coop_cohp import COOPCOHPResult

    n_pairs, n_points = 2, 101
    energies = np.linspace(-10.0, 5.0, n_points)
    coop = np.zeros((n_pairs, n_points))
    coop[0] = np.exp(-0.5 * ((energies + 4.0) / 0.5) ** 2) - 0.5 * np.exp(
        -0.5 * ((energies - 2.0) / 0.5) ** 2
    )
    coop[1] = 0.7 * np.exp(-0.5 * ((energies + 3.0) / 0.5) ** 2) - 0.3 * np.exp(
        -0.5 * ((energies - 1.0) / 0.5) ** 2
    )
    integrated = np.array([0.5, 0.3])
    pairs = [
        {"i": 0, "j": 1, "symbol_i": "Si", "symbol_j": "Si", "distance_ang": 2.35},
        {"i": 0, "j": 2, "symbol_i": "Si", "symbol_j": "O", "distance_ang": 1.62},
    ]
    return COOPCOHPResult(
        energies=energies,
        coop=coop,
        cohp=None,
        integrated_coop=integrated,
        integrated_cohp=None,
        pairs=pairs,
        fermi_energy=-0.2,
        sigma=0.01,
    )


def _synthetic_cohp_result():
    """Minimal COOPCOHPResult with COHP for plot smoke testing."""
    from vibeqc.coop_cohp import COOPCOHPResult

    n_pairs, n_points = 2, 101
    energies = np.linspace(-10.0, 5.0, n_points)
    cohp = np.zeros((n_pairs, n_points))
    cohp[0] = 2.0 * np.exp(-0.5 * ((energies + 4.0) / 0.4) ** 2) - 1.5 * np.exp(
        -0.5 * ((energies - 2.0) / 0.4) ** 2
    )
    cohp[1] = 1.5 * np.exp(-0.5 * ((energies + 3.5) / 0.4) ** 2) - 1.0 * np.exp(
        -0.5 * ((energies - 1.5) / 0.4) ** 2
    )
    integrated = np.array([0.8, 0.6])
    pairs = [
        {"i": 0, "j": 1, "symbol_i": "C", "symbol_j": "C", "distance_ang": 1.42},
        {"i": 0, "j": 2, "symbol_i": "C", "symbol_j": "H", "distance_ang": 1.09},
    ]
    return COOPCOHPResult(
        energies=energies,
        coop=np.zeros_like(cohp),
        cohp=cohp,
        integrated_coop=np.zeros(2),
        integrated_cohp=integrated,
        pairs=pairs,
        fermi_energy=-0.2,
        sigma=0.01,
    )


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def test_cli_coop_prints_table(tmp_path):
    """vibeqc coop <qvf> prints a pair table."""
    import io
    from contextlib import redirect_stderr, redirect_stdout

    coop_data = _make_synthetic_coop_data()
    cohp_data = _make_synthetic_cohp_data()
    plan = _make_plan(tmp_path)
    path = write_qvf(tmp_path / "test", plan, coop_data=coop_data, cohp_data=cohp_data)

    from vibeqc._cli import main as cli_main

    out_buf = io.StringIO()
    with redirect_stdout(out_buf), redirect_stderr(io.StringIO()):
        try:
            rc = cli_main(["coop", str(path)])
        except SystemExit:
            rc = 0
    assert rc == 0
    out = out_buf.getvalue()
    assert "COOP" in out
    assert "-COHP" in out
    assert "Si" in out
    assert "C" in out


def test_cli_coop_json_flag(tmp_path):
    """vibeqc coop --json <qvf> prints valid JSON."""
    import io
    import json
    from contextlib import redirect_stderr, redirect_stdout

    coop_data = _make_synthetic_coop_data()
    cohp_data = _make_synthetic_cohp_data()
    plan = _make_plan(tmp_path)
    path = write_qvf(tmp_path / "test", plan, coop_data=coop_data, cohp_data=cohp_data)

    from vibeqc._cli import main as cli_main

    out_buf = io.StringIO()
    with redirect_stdout(out_buf), redirect_stderr(io.StringIO()):
        try:
            rc = cli_main(["coop", "--json", str(path)])
        except SystemExit:
            rc = 0
    assert rc == 0
    data = json.loads(out_buf.getvalue())
    assert "coop" in data
    assert "cohp" in data
    assert len(data["coop"]["pairs"]) == 3
    assert data["coop"]["pairs"][0]["symbol_i"] == "Si"
    assert isinstance(data["coop"]["pairs"][0]["integrated"], float)


def test_cli_mayer_prints_table(tmp_path):
    """vibeqc mayer <qvf> prints bond order table."""
    import io
    import json
    from contextlib import redirect_stderr, redirect_stdout

    # Build a QVF with bond_orders
    bo_pairs = [
        {
            "i": 0,
            "j": 1,
            "symbol_i": "Si",
            "symbol_j": "Si",
            "order": 1.05,
            "distance_ang": 2.35,
        },
        {
            "i": 0,
            "j": 2,
            "symbol_i": "Si",
            "symbol_j": "O",
            "order": 0.82,
            "distance_ang": 1.62,
        },
    ]
    from vibeqc.output.formats.qvf import write_qvf

    plan = _make_plan(tmp_path)
    path = write_qvf(
        tmp_path / "test",
        plan,
        bond_orders_data={"method": "mayer", "pairs": bo_pairs},
    )

    from vibeqc._cli import main as cli_main

    out_buf = io.StringIO()
    with redirect_stdout(out_buf), redirect_stderr(io.StringIO()):
        try:
            rc = cli_main(["mayer", str(path)])
        except SystemExit:
            rc = 0
    assert rc == 0
    out = out_buf.getvalue()
    assert "Mayer" in out
    assert "Si" in out
    assert "1.0500" in out

    # --json flag
    out_buf2 = io.StringIO()
    with redirect_stdout(out_buf2), redirect_stderr(io.StringIO()):
        try:
            rc2 = cli_main(["mayer", "--json", str(path)])
        except SystemExit:
            rc2 = 0
    assert rc2 == 0
    data = json.loads(out_buf2.getvalue())
    assert data["method"] == "mayer"
    assert len(data["pairs"]) == 2
    # JSON mode always shows all pairs (threshold only applies to table)
    assert data["pairs"][0]["order"] == 1.05
    assert data["pairs"][1]["order"] == 0.82


# ---------------------------------------------------------------------------
# Periodic Mayer bond orders
# ---------------------------------------------------------------------------


def test_periodic_mayer_h2():
    """1D H2 chain: Mayer bond order ≈ 1.0 (single bond)."""
    system, basis = _h2_chain_1d()
    from vibeqc._vibeqc_core import (
        LatticeSumOptions,
        compute_kinetic_lattice,
        compute_nuclear_lattice,
        compute_overlap_lattice,
        monkhorst_pack,
    )

    opts = LatticeSumOptions()
    S = compute_overlap_lattice(basis, system, opts)
    T = compute_kinetic_lattice(basis, system, opts)
    V = compute_nuclear_lattice(basis, system, opts)
    km = monkhorst_pack(system, [8, 1, 1])

    M = vq.periodic_mayer_bond_orders(
        [T, V],
        S,
        system,
        basis,
        km,
        n_electrons_per_cell=2,
    )
    assert M.shape == (2, 2)
    # H-H bond order should be close to 1.0
    assert 0.5 < M[0, 1] < 1.1, f"Expected H-H bond order near 1.0, got {M[0, 1]}"
    assert M[0, 1] == pytest.approx(M[1, 0])
    # No self-bond order
    assert M[0, 0] == pytest.approx(0.0, abs=1e-10)
    assert M[1, 1] == pytest.approx(0.0, abs=1e-10)


def test_periodic_mayer_unrestricted_shape():
    """Unrestricted call with F_terms_beta produces correct shape."""
    system, basis = _h2_chain_1d()
    from vibeqc._vibeqc_core import (
        LatticeSumOptions,
        compute_kinetic_lattice,
        compute_overlap_lattice,
        monkhorst_pack,
    )

    opts = LatticeSumOptions()
    S = compute_overlap_lattice(basis, system, opts)
    T = compute_kinetic_lattice(basis, system, opts)
    km = monkhorst_pack(system, [4, 1, 1])

    M = vq.periodic_mayer_bond_orders(
        T,
        S,
        system,
        basis,
        km,
        F_terms_beta=T,
    )
    assert M.shape == (2, 2)


def test_coop_spin_polarized_fermi():
    """Unrestricted COOP/COHP uses per-spin Fermi levels."""
    system, basis = _h2_chain_1d()
    from vibeqc._vibeqc_core import (
        LatticeSumOptions,
        compute_kinetic_lattice,
        compute_nuclear_lattice,
        compute_overlap_lattice,
        monkhorst_pack,
    )

    opts = LatticeSumOptions()
    S = compute_overlap_lattice(basis, system, opts)
    T = compute_kinetic_lattice(basis, system, opts)
    V = compute_nuclear_lattice(basis, system, opts)
    km = monkhorst_pack(system, [4, 1, 1])

    result = vq.compute_coop_cohp(
        [T, V],
        S,
        system,
        basis,
        km,
        H_terms=[T, V],
        F_terms_beta=[T, V],
        sigma=0.1,
        n_grid=51,
        n_electrons_per_cell=3,
    )
    assert result.coop.shape == (2, 1, 51)
    assert result.integrated_coop.shape == (2, 1)
    diff = abs(
        float(result.integrated_coop[0, 0]) - float(result.integrated_coop[1, 0])
    )
    assert diff > 1e-8, (
        f"Alpha/beta ICOOP should differ for n_elec=3, "
        f"got a={result.integrated_coop[0, 0]:.6f} b={result.integrated_coop[1, 0]:.6f}"
    )


# ---------------------------------------------------------------------------
# Runner gate: the analysis needs the QVF archive as its container (IID 195)
# ---------------------------------------------------------------------------


def _h2_cube_3d():
    """Cheapest 3-D cell with a bond: H2 in a 6-bohr cube, STO-3G."""
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 6.0,
        unit_cell=[vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def test_runner_coop_cohp_requires_output_qvf(tmp_path):
    """``coop_cohp=True`` with ``output_qvf=False`` fails closed before SCF.

    Until IID 195 the request was dropped silently: the analysis lives inside
    the DOS/QVF block of ``run_periodic_job`` and has no ``.out`` or sidecar
    rendering, so with QVF off the run converged, wrote no ``.qvf``, and said
    nothing about COOP/COHP anywhere. The gate mirrors ``qvf_wannier_centers``
    (the sibling whose only sink is the archive) and names both knobs.
    """
    from vibeqc.periodic_runner import run_periodic_job

    system, basis = _h2_cube_3d()
    stem = tmp_path / "h2_coop_no_qvf"
    with pytest.raises(ValueError, match="coop_cohp=True requires output_qvf=True"):
        run_periodic_job(
            system,
            basis,
            method="RHF",
            output=stem,
            output_qvf=False,
            coop_cohp=True,
            verbose=0,
        )
    # Pre-SCF refusal: nothing was produced under the stem.
    assert not list(tmp_path.glob("h2_coop_no_qvf*")), (
        "the refusal must happen before any artefact is written"
    )


def test_cohp_uses_each_analyzed_spin_hamiltonian(monkeypatch):
    import vibeqc.coop_cohp as module
    from vibeqc import _vibeqc_core as core

    system = vq.PeriodicSystem(3, np.eye(3)*12., [
        vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [1.4, 0., 0.]),
    ])
    basis = vq.BasisSet(system.unit_cell_molecule(), 'sto-3g')
    cells = core.direct_lattice_cells(system, .1)
    lattice = lambda matrix: core.make_lattice_matrix_set(2, cells, [matrix])
    fa = np.array([[-1., -.2], [-.2, -1.]])
    fb = np.array([[-.7, .3], [.3, -.7]])
    mesh = core.monkhorst_pack(system, [1, 1, 1])
    observed = []
    kernel = module._cohp_weights_k

    def record(coefficients, operator, pairs):
        observed.append(np.asarray(operator).copy())
        all_pairs = [([a], [b]) for a in range(2) for b in range(2)]
        partition = np.asarray(kernel(coefficients, operator, all_pairs)).sum(axis=0)
        np.testing.assert_allclose(partition, np.linalg.eigvalsh(operator), atol=1e-12, rtol=0)
        return kernel(coefficients, operator, pairs)

    monkeypatch.setattr(module, '_cohp_weights_k', record)
    result = module.compute_coop_cohp(
        lattice(fa), lattice(np.eye(2)), system, basis, mesh,
        F_terms_beta=lattice(fb), include_cohp=True, n_electrons_per_cell=2,
    )
    assert result.cohp.shape[0] == 2
    np.testing.assert_array_equal(observed, [fa, fb])
    beta_energies, beta_coefficients = np.linalg.eigh(fb)
    assert result.energies[-1] > beta_energies[-1]
    beta_weights = np.asarray(kernel(beta_coefficients, fb, [([0], [1])]))[0]
    gaussian = np.exp(-.5 * ((result.energies[None, :] - beta_energies[:, None])
                            / result.sigma)**2) / (np.sqrt(2*np.pi) * result.sigma)
    np.testing.assert_allclose(result.cohp[1, 0], -beta_weights @ gaussian, atol=1e-12, rtol=1e-12)
    with pytest.raises(ValueError, match='different operator'):
        module.compute_coop_cohp(
            lattice(fa), lattice(np.eye(2)), system, basis, mesh,
            H_terms=lattice(-np.eye(2)),
        )
