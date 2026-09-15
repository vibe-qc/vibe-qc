"""Population / properties dump — ``{stem}.population.{txt,json}``.

Pins the contract for the Phase O6 writer:

  1. ``write_population(stem, result, basis, molecule)`` emits both
     ``{stem}.population.txt`` and ``{stem}.population.json``.
  2. The TXT form is human-readable with ``#``-prefixed section
     headers and tab-separated bodies — loadable into pandas /
     awk / spreadsheet importers with ``#`` as the comment marker.
  3. The JSON form has top-level keys ``mulliken`` / ``loewdin``
     / ``hirshfeld`` / ``mayer`` / ``dipole`` / ``errors``, each shaped
     as a list of objects (or single object for ``dipole`` / ``errors``).
  4. A property-computation failure on one section does NOT
     suppress the other sections — partial success is preserved.
  5. ``compute_population_summary`` returns a
     :class:`PopulationSummary` with the four fields populated
     independently.
  6. ``OutputPlan.from_run_job_kwargs(write_population=True)``
     declares the ``.population.txt`` + ``.population.json`` rows.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from vibeqc.output import OutputPlan, write_population
from vibeqc.output.formats.population import (
    PopulationSummary,
    compute_aiccm2026dev_b_population_summary,
    compute_bipole_population_summary,
    compute_native_mulliken_population_summary,
    compute_population_summary,
    format_population_json,
    format_population_txt,
    unsupported_population_summary,
    write_population_summary,
)


# Duck-typed stand-ins so the test suite doesn't need the C++ core
# importable. We monkeypatch the property-computing helpers to
# return deterministic values; the writer is what we're testing.
@dataclass
class _Atom:
    Z: int
    xyz: tuple


@dataclass
class _Mol:
    atoms: list


def _h2o() -> _Mol:
    return _Mol(
        [
            _Atom(8, (0.0, 0.0, 0.0)),
            _Atom(1, (0.0, 1.4, 0.0)),
            _Atom(1, (0.0, -1.4, 0.0)),
        ]
    )


# ---------------------------------------------------------------------- #
# format_population_txt / json
# ---------------------------------------------------------------------- #


def _seeded_summary() -> PopulationSummary:
    s = PopulationSummary()
    s.mulliken_atoms = [
        (0, "O", 8.0, -0.5),
        (1, "H", 1.0, +0.25),
        (2, "H", 1.0, +0.25),
    ]
    s.loewdin_atoms = [
        (0, "O", 8.0, -0.42),
        (1, "H", 1.0, +0.21),
        (2, "H", 1.0, +0.21),
    ]
    s.hirshfeld_atoms = [
        (0, "O", 8.0, -0.55),
        (1, "H", 1.0, +0.275),
        (2, "H", 1.0, +0.275),
    ]
    s.mayer_bonds = [
        (0, 1, "O", "H", 0.95),
        (0, 2, "O", "H", 0.95),
    ]
    s.dipole = {
        "x_ebohr": 0.0,
        "y_ebohr": 0.0,
        "z_ebohr": -0.78,
        "total_ebohr": 0.78,
        "total_debye": 1.98,
        "origin_bohr": [0.0, 0.0, 0.116],
    }
    return s


def test_format_txt_has_section_headers() -> None:
    text = format_population_txt(_seeded_summary())
    assert "# === Mulliken atomic charges ===" in text
    assert "# === Löwdin atomic charges ===" in text
    assert "# === Hirshfeld atomic charges ===" in text
    assert "# === Mayer bond orders (top entries) ===" in text
    assert "# === Wiberg bond indices (top entries) ===" in text
    assert "# === NPA atomic charges ===" in text
    assert "# === Dipole moment ===" in text


def test_format_txt_atoms_are_tab_separated() -> None:
    text = format_population_txt(_seeded_summary())
    # Find the Mulliken body — first non-comment, non-blank line
    # after the header.
    in_mul = False
    for line in text.splitlines():
        if "=== Mulliken" in line:
            in_mul = True
            continue
        if in_mul and line and not line.startswith("#") and not line.strip() == "":
            assert "\t" in line, f"expected tab-separated row: {line!r}"
            break


def test_format_json_has_documented_top_level_keys() -> None:
    payload = format_population_json(_seeded_summary())
    data = json.loads(payload)
    assert set(data.keys()) == {
        "mulliken",
        "loewdin",
        "hirshfeld",
        "mayer",
        "wiberg",
        "npa",
        "dipole",
        "errors",
    }


def test_format_json_atoms_carry_symbol_and_charge() -> None:
    payload = format_population_json(_seeded_summary())
    data = json.loads(payload)
    mul = data["mulliken"]
    assert mul[0] == {"idx": 0, "symbol": "O", "Z": 8.0, "charge": -0.5}


def test_format_json_dipole_has_debye_and_origin() -> None:
    payload = format_population_json(_seeded_summary())
    data = json.loads(payload)
    d = data["dipole"]
    assert d["total_debye"] == pytest.approx(1.98)
    assert d["origin_bohr"] == [0.0, 0.0, 0.116]


# ---------------------------------------------------------------------- #
# Partial-success: errored sections don't suppress others
# ---------------------------------------------------------------------- #


def test_format_txt_marks_errored_sections() -> None:
    s = PopulationSummary()
    s.mulliken_atoms = [(0, "C", 6.0, 0.0)]
    s.errors["loewdin"] = "ValueError: bad overlap"
    s.errors["mayer"] = "RuntimeError: nan in BO matrix"
    text = format_population_txt(s)
    assert "ValueError: bad overlap" in text
    assert "RuntimeError: nan in BO matrix" in text
    # Mulliken still rendered.
    assert "C\t6.0\t+0.000000" in text


def test_format_json_carries_errors_dict() -> None:
    s = PopulationSummary()
    s.errors["dipole"] = "DimensionError: not invertible"
    payload = format_population_json(s)
    data = json.loads(payload)
    assert data["errors"]["dipole"] == "DimensionError: not invertible"


def test_unsupported_population_sidecars_do_not_emit_raw_value_errors(
    tmp_path: Path,
) -> None:
    """BIPOLE periodic densities are lattice matrix sets, not molecular
    density matrices. Unsupported sidecars must carry a stable route marker,
    never raw NumPy shape exceptions from molecular property helpers.
    """
    summary = unsupported_population_summary(
        "periodic BIPOLE population properties require lattice-summed "
        "property integrals"
    )
    txt_path, json_path = write_population_summary(tmp_path / "bipole", summary)

    txt = txt_path.read_text(encoding="utf-8")
    raw_json = json_path.read_text(encoding="utf-8")
    payload = json.loads(raw_json)

    for artifact in (txt, raw_json):
        assert "unsupported:" in artifact
        assert "ValueError" not in artifact
        assert "einstein sum" not in artifact
        assert "matmul:" not in artifact
        assert "Traceback" not in artifact

    assert set(payload["errors"]) == {
        "mulliken",
        "loewdin",
        "hirshfeld",
        "mayer",
        "dipole",
        "mulliken_spin",
    }
    assert all(
        message.startswith("unsupported:")
        for message in payload["errors"].values()
    )


def test_native_mulliken_summary_preserves_charges_and_marks_other_analyses() -> None:
    summary = compute_native_mulliken_population_summary(
        [-0.6, 0.3, 0.3],
        _h2o(),
        "scc_dftb",
    )

    assert [row[3] for row in summary.mulliken_atoms] == [-0.6, 0.3, 0.3]
    assert set(summary.errors) == {
        "loewdin",
        "hirshfeld",
        "mayer",
        "wiberg",
        "dipole",
    }
    assert "Natural Atomic Orbital" in summary.unavailable["npa"]
    assert all(
        message.startswith("unsupported: scc_dftb native population")
        for message in summary.errors.values()
    )


@pytest.mark.parametrize(
    ("charges", "message"),
    (([-0.5, 0.5], "charge count"), ([-0.5, 0.2, 0.2], "sum to")),
)
def test_native_mulliken_summary_rejects_invalid_producer_payload(
    charges,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        compute_native_mulliken_population_summary(charges, _h2o(), "scc_dftb")


def _tiny_h2_bipole_case(lattice_bohr: float = 12.0):
    import vibeqc as vq

    system = vq.PeriodicSystem(
        3,
        np.eye(3) * lattice_bohr,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _test_cell_key(cell: Any) -> tuple[int, int, int]:
    return tuple(int(x) for x in np.asarray(cell.index, dtype=int).reshape(3))


def _test_ao_to_atom_map(basis: Any) -> np.ndarray:
    per_ao: list[int] = []
    for shell in basis.shells():
        per_ao.extend([int(shell.atom_index)] * (2 * int(shell.l) + 1))
    return np.asarray(per_ao, dtype=np.int64)


def _independent_lattice_mulliken_populations(
    vq: Any,
    result: Any,
    basis: Any,
    system: Any,
    lattice_options: Any,
) -> tuple[np.ndarray, bool]:
    """Independent BIPOLE Mulliken oracle over overlap/operator cells."""
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = float(lattice_options.cutoff_bohr)
    opts.nuclear_cutoff_bohr = float(lattice_options.nuclear_cutoff_bohr)
    overlap = vq.compute_overlap_lattice(basis, system, opts)
    density_by_key = {
        _test_cell_key(cell): np.asarray(block)
        for cell, block in zip(result.density.cells, result.density.blocks)
    }
    ao_to_atom = _test_ao_to_atom_map(basis)
    populations = np.zeros(len(system.unit_cell_molecule().atoms), dtype=float)
    saw_image_block = False
    for cell, s_block_raw in zip(overlap.cells, overlap.blocks):
        key = _test_cell_key(cell)
        d_block = density_by_key[key]
        s_block = np.asarray(s_block_raw)
        if key != (0, 0, 0) and np.max(np.abs(d_block * s_block)) > 1e-14:
            saw_image_block = True
        pair_pop = np.real(d_block * s_block)
        for mu in range(pair_pop.shape[0]):
            atom_mu = int(ao_to_atom[mu])
            for nu in range(pair_pop.shape[1]):
                value = float(pair_pop[mu, nu])
                atom_nu = int(ao_to_atom[nu])
                if atom_mu == atom_nu:
                    populations[atom_mu] += value
                else:
                    populations[atom_mu] += 0.5 * value
                    populations[atom_nu] += 0.5 * value
    return populations, saw_image_block


def test_bipole_population_summary_mulliken_lattice_sum_invariant() -> None:
    import vibeqc as vq

    system, basis = _tiny_h2_bipole_case(lattice_bohr=8.0)
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 8.0
    opts.lattice_opts.nuclear_cutoff_bohr = 8.0
    opts.max_iter = 1
    opts.initial_guess = vq.InitialGuess.HCORE
    opts.use_diis = False
    opts.damping = 0.0
    result = vq.run_pbc_bipole_rhf(
        system,
        basis,
        vq.monkhorst_pack(system, [1, 1, 1]),
        opts,
        progress=False,
    )

    mol = system.unit_cell_molecule()
    summary = compute_bipole_population_summary(
        result,
        basis,
        mol,
        system,
        lattice_options=opts.lattice_opts,
    )

    assert "mulliken" not in summary.errors
    assert len(summary.mulliken_atoms) == len(mol.atoms)
    charges = np.asarray([row[3] for row in summary.mulliken_atoms], dtype=float)
    assert np.all(np.isfinite(charges))
    assert charges.sum() == pytest.approx(0.0, abs=1e-8)

    electron_pop = np.asarray(
        [row[2] - row[3] for row in summary.mulliken_atoms],
        dtype=float,
    )
    oracle_pop, saw_image_block = _independent_lattice_mulliken_populations(
        vq,
        result,
        basis,
        system,
        opts.lattice_opts,
    )
    assert saw_image_block
    np.testing.assert_allclose(electron_pop, oracle_pop, atol=1e-10)
    assert float(electron_pop.sum()) == pytest.approx(
        float(system.n_electrons()),
        abs=1e-8,
    )

    assert "mayer" not in summary.errors
    assert summary.mayer_bonds
    h_h = [bond for bond in summary.mayer_bonds if {bond[2], bond[3]} == {"H"}]
    assert len(h_h) == 1
    assert 0.1 < h_h[0][4] < 2.0

    assert "loewdin" not in summary.errors
    assert len(summary.loewdin_atoms) == len(mol.atoms)
    lowdin_charges = np.asarray([row[3] for row in summary.loewdin_atoms], dtype=float)
    assert np.all(np.isfinite(lowdin_charges))
    assert lowdin_charges.sum() == pytest.approx(0.0, abs=1e-8)
    assert summary.errors["hirshfeld"].startswith("unsupported:")
    assert summary.errors["dipole"].startswith("unsupported:")


def test_bipole_population_summary_mayer_uses_multik_orbitals() -> None:
    """BIPOLE Mayer sidecars use the SCF k-mesh, not a Gamma proxy."""
    import vibeqc as vq

    system, basis = _tiny_h2_bipole_case(lattice_bohr=12.0)
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 8.0
    opts.lattice_opts.nuclear_cutoff_bohr = 8.0
    opts.max_iter = 1
    opts.initial_guess = vq.InitialGuess.HCORE
    opts.use_diis = False
    opts.damping = 0.0
    kmesh = vq.monkhorst_pack(system, [2, 1, 1])
    result = vq.run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        opts,
        progress=False,
    )

    summary = compute_bipole_population_summary(
        result,
        basis,
        system.unit_cell_molecule(),
        system,
        lattice_options=opts.lattice_opts,
        kmesh=kmesh,
    )

    assert "mayer" not in summary.errors
    assert len(summary.mayer_bonds) == 1
    i, j, si, sj, order = summary.mayer_bonds[0]
    assert (i, j, si, sj) == (0, 1, "H", "H")
    assert order == pytest.approx(1.0, abs=1e-10)
    assert "loewdin" not in summary.errors
    lowdin_charges = np.asarray([row[3] for row in summary.loewdin_atoms], dtype=float)
    assert lowdin_charges.sum() == pytest.approx(0.0, abs=1e-8)
    assert summary.errors["hirshfeld"].startswith("unsupported:")


def test_aiccm2026dev_b_population_summary_uses_finite_torus_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AICCM-B sidecars must not pass block/k-point densities to molecular
    population helpers; large basis block containers used to trigger NumPy's
    high-rank einsum limit there.
    """

    class _Shell:
        l = 0
        pure = True

        def __init__(self, atom_index: int):
            self.atom_index = atom_index

    class _Basis:
        nbasis = 2

        def shells(self) -> list[_Shell]:
            return [_Shell(0), _Shell(1)]

    basis = _Basis()
    system = SimpleNamespace(
        unit_cell=[
            _Atom(1, (0.0, 0.0, 0.0)),
            _Atom(1, (0.0, 0.0, 1.4)),
        ],
    )
    molecule = _Mol(list(system.unit_cell))
    density = np.array([[1.0, 0.25], [0.25, 1.0]])
    overlap = np.eye(2)
    result = SimpleNamespace(
        converged=True,
        kpoint_weights=np.array([1.0]),
        density=SimpleNamespace(blocks=[density]),
        overlap=SimpleNamespace(blocks=[overlap]),
        aiccm2026dev_b=SimpleNamespace(
            effective_nuclear_charges=np.array([1.0, 1.0]),
            effective_electron_count=2,
        ),
    )

    def _fake_props(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(mulliken_charges=np.array([0.0, 0.0]))

    def _fake_bonds(*_args: Any, **_kwargs: Any) -> Any:
        return SimpleNamespace(
            bonds=(
                SimpleNamespace(
                    atom_i=0,
                    atom_j=1,
                    symbol_i="H",
                    symbol_j="H",
                    order=0.8,
                ),
            )
        )

    monkeypatch.setattr(
        "vibeqc.periodic.chi.properties."
        "derive_aiccm2026dev_b_scf_properties",
        _fake_props,
    )
    monkeypatch.setattr(
        "vibeqc.periodic.chi.properties."
        "aiccm2026dev_b_mayer_bond_orders",
        _fake_bonds,
    )

    summary = compute_aiccm2026dev_b_population_summary(
        result,
        basis,
        molecule,
        system,
    )

    assert summary.mulliken_atoms == [(0, "H", 1.0, 0.0), (1, "H", 1.0, 0.0)]
    assert [row[3] for row in summary.loewdin_atoms] == pytest.approx([0.0, 0.0])
    assert summary.mayer_bonds == [(0, 1, "H", "H", 0.8)]
    assert summary.errors["hirshfeld"].startswith("unsupported:")
    assert summary.errors["dipole"].startswith("unsupported:")


def test_run_periodic_job_bipole_population_sidecar_and_qvf_are_clean(
    tmp_path: Path,
) -> None:
    import vibeqc as vq

    system, basis = _tiny_h2_bipole_case()
    stem = tmp_path / "h2-bipole-pop"
    # A converging SCF is required: since the fail-closed non-convergence
    # gate (3fc4b068f) the runner raises before writing any population
    # sidecar when the cap is hit (IID 190; this H2 cell converges in 2
    # iterations). The 0.5 bohr QVF density grid keeps the BvK density
    # artifact inside its integral tolerance; 3.0 bohr tripped the
    # compatibility fallback once the run actually reached the QVF writer.
    result = vq.run_periodic_job(
        system,
        basis,
        method="RHF",
        jk_method="bipole",
        output=stem,
        max_iter=10,
        conv_tol_energy=1e-6,
        initial_guess="HCORE",
        use_diis=False,
        damping=0.0,
        write_molden_file=False,
        write_density=False,
        density_spacing_bohr=0.5,
        write_xyz_file=False,
        write_poscar_file=False,
        write_xsf_structure_file=False,
        write_cif_file=False,
        write_population_file=True,
        citations=False,
        output_qvf=True,
        progress=False,
    )
    assert result.converged

    txt = stem.with_suffix(".population.txt").read_text(encoding="utf-8")
    raw_json = stem.with_suffix(".population.json").read_text(encoding="utf-8")
    payload = json.loads(raw_json)
    for artifact in (txt, raw_json):
        assert "ValueError" not in artifact
        assert "einstein sum" not in artifact
        assert "matmul:" not in artifact
        assert "Traceback" not in artifact

    mulliken = payload["mulliken"]
    assert len(mulliken) == 2
    charges = np.asarray([row["charge"] for row in mulliken], dtype=float)
    assert np.all(np.isfinite(charges))
    assert charges.sum() == pytest.approx(0.0, abs=1e-8)
    assert "mulliken" not in payload["errors"]
    assert payload["mayer"]
    assert "mayer" not in payload["errors"]
    loewdin = payload["loewdin"]
    assert len(loewdin) == 2
    loewdin_charges = np.asarray([row["charge"] for row in loewdin], dtype=float)
    assert np.all(np.isfinite(loewdin_charges))
    assert loewdin_charges.sum() == pytest.approx(0.0, abs=1e-8)
    assert "loewdin" not in payload["errors"]
    assert payload["errors"]["hirshfeld"].startswith("unsupported:")
    assert payload["errors"]["dipole"].startswith("unsupported:")

    with zipfile.ZipFile(stem.with_suffix(".qvf")) as zf:
        manifest = json.loads(zf.read("manifest.json"))
        props = [
            section
            for section in manifest["sections"]
            if section["kind"] == "atom_properties"
        ]
        assert len(props) == 1
        members = props[0]["members"]
        assert set(members) == {"mulliken_charge", "loewdin_charge"}
        qvf_charges = np.frombuffer(
            zf.read(members["mulliken_charge"]["path"]),
            dtype=np.float64,
        )
        qvf_lowdin_charges = np.frombuffer(
            zf.read(members["loewdin_charge"]["path"]),
            dtype=np.float64,
        )
        bond_sections = [
            section
            for section in manifest["sections"]
            if section["kind"] == "bond_orders"
        ]
        assert len(bond_sections) == 1
    np.testing.assert_allclose(qvf_charges, charges, atol=1e-12)
    np.testing.assert_allclose(qvf_lowdin_charges, loewdin_charges, atol=1e-12)


# ---------------------------------------------------------------------- #
# Disk writers
# ---------------------------------------------------------------------- #


def test_write_population_summary_emits_both_files(
    tmp_path: Path,
) -> None:
    s = _seeded_summary()
    txt_path, json_path = write_population_summary(
        tmp_path / "h2o",
        s,
    )
    assert txt_path == tmp_path / "h2o.population.txt"
    assert json_path == tmp_path / "h2o.population.json"
    assert txt_path.is_file() and json_path.is_file()
    # JSON parses cleanly.
    json.loads(json_path.read_text(encoding="utf-8"))


def test_write_population_compute_path_handles_all_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every property helper raises, ``compute_population_summary``
    should return a summary with the errors dict populated and empty
    body fields — never raise itself, never crash the caller."""
    import vibeqc.output.formats.population as pop_mod

    def _boom(*a: Any, **k: Any) -> Any:
        raise RuntimeError("simulated property failure")

    # Patch only the loader paths the writer uses; the real
    # vibeqc.properties module is left intact for other tests.
    monkeypatch.setattr(
        "vibeqc.properties.mulliken_charges",
        _boom,
        raising=False,
    )
    monkeypatch.setattr(
        "vibeqc.properties.loewdin_charges",
        _boom,
        raising=False,
    )
    monkeypatch.setattr(
        "vibeqc.properties.hirshfeld_charges",
        _boom,
        raising=False,
    )
    monkeypatch.setattr(
        "vibeqc.properties.mayer_bond_orders",
        _boom,
        raising=False,
    )
    monkeypatch.setattr(
        "vibeqc.properties.dipole_moment",
        _boom,
        raising=False,
    )

    # Dummy result + basis (the property helpers never get called
    # because of the monkeypatches above).
    summary = compute_population_summary(
        object(),
        object(),
        _h2o(),
    )
    assert summary.errors.keys() >= {
        "mulliken",
        "loewdin",
        "hirshfeld",
        "mayer",
        "dipole",
    }
    assert summary.mulliken_atoms == []
    assert summary.hirshfeld_atoms == []
    assert summary.dipole is None


def test_unimplemented_npa_is_not_attempted_or_reported_as_failure(monkeypatch) -> None:
    calls = []

    def npa_spy(*args, **kwargs):
        calls.append(True)
        raise NotImplementedError("NPA must not be attempted")

    monkeypatch.setattr("vibeqc.nbo.npa_charges", npa_spy)
    monkeypatch.setattr("vibeqc._vibeqc_core.compute_overlap", lambda *a: np.eye(3))
    monkeypatch.setattr(
        "vibeqc.properties.mulliken_charges", lambda *a, **k: [-0.6, 0.3, 0.3]
    )

    def failed_dipole(*args, **kwargs):
        raise RuntimeError("dipole failed independently")

    monkeypatch.setattr("vibeqc.properties.dipole_moment", failed_dipole)
    summary = compute_population_summary(
        SimpleNamespace(density=np.eye(3)), object(), _h2o()
    )
    assert calls == []
    assert summary.npa_atoms == []
    assert "npa" not in summary.errors
    assert "Natural Atomic Orbital" in summary.unavailable["npa"]
    payload = json.loads(format_population_json(summary))
    assert payload["npa"] == []
    assert "npa" not in payload["errors"]
    assert payload["unavailable"]["npa"] == summary.unavailable["npa"]
    assert payload["mulliken"][0]["charge"] == -0.6
    assert payload["errors"]["dipole"] == "RuntimeError: dipole failed independently"
    text = format_population_txt(summary)
    assert "# npa: not implemented --" in text
    assert "# npa: N/A" not in text
    assert "# dipole: N/A -- RuntimeError: dipole failed independently" in text


def test_compute_summary_stores_charge_not_population(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``mulliken_charges`` / ``loewdin_charges`` already return partial
    charges (q = Z − population). ``compute_population_summary`` must store
    those charges verbatim in the row's ``charge`` slot — NOT ``Z − q``,
    which would re-derive the population and report e.g. +5.85 as the
    "charge" on carbon. Regression for that double-subtraction bug."""
    import numpy as np

    # Known partial charges for water: O negative, H positive, sum ≈ 0.
    mul = np.array([-0.70, 0.35, 0.35])
    low = np.array([-0.62, 0.31, 0.31])
    hir = np.array([-0.74, 0.37, 0.37])
    monkeypatch.setattr(
        "vibeqc.properties.mulliken_charges", lambda *a, **k: mul, raising=False
    )
    monkeypatch.setattr(
        "vibeqc.properties.loewdin_charges", lambda *a, **k: low, raising=False
    )
    monkeypatch.setattr(
        "vibeqc.properties.hirshfeld_charges",
        lambda *a, **k: SimpleNamespace(charges=hir),
        raising=False,
    )
    monkeypatch.setattr(
        "vibeqc.properties.mayer_bond_orders", lambda *a, **k: [], raising=False
    )
    monkeypatch.setattr(
        "vibeqc.properties.dipole_moment", lambda *a, **k: None, raising=False
    )

    summary = compute_population_summary(object(), object(), _h2o())

    # row = (idx, symbol, Z, charge); charge column must equal the helper output.
    assert [round(r[3], 6) for r in summary.mulliken_atoms] == [-0.70, 0.35, 0.35]
    assert [round(r[3], 6) for r in summary.loewdin_atoms] == [-0.62, 0.31, 0.31]
    assert [round(r[3], 6) for r in summary.hirshfeld_atoms] == [
        -0.74,
        0.37,
        0.37,
    ]
    # Charges must sum to ~the molecular charge (0 here), never to N_electrons.
    assert abs(sum(r[3] for r in summary.mulliken_atoms)) < 1e-9
    # Z column is still the atomic number.
    assert [r[2] for r in summary.mulliken_atoms] == [8.0, 1.0, 1.0]


# ----------------------------------------------------------------------
# Integration: compute_population_summary with a real SCF result
# ----------------------------------------------------------------------


def test_compute_population_summary_real_scf() -> None:
    """ "compute_population_summary" must return real numerical data
    from :mod:`vibeqc.properties` when fed a genuine RHF result.

    Regression for the import-path bug where ``from ..properties``
    silently failed (ModuleNotFoundError swallowed by try/except) and
    every section landed in the errors dict with zero-row bodies.
    """
    import vibeqc as vq

    mol = vq.Molecule(
        [
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 1.43, -0.98]),
            vq.Atom(1, [0.0, -1.43, -0.98]),
        ]
    )
    basis = vq.BasisSet(mol, "sto-3g")
    result = vq.run_rhf(mol, basis)

    summary = compute_population_summary(result, basis, mol)

    # No errors should be recorded for a well-behaved neutral molecule.
    assert "mulliken" not in summary.errors, (
        f"mulliken error: {summary.errors.get('mulliken')}"
    )
    assert "loewdin" not in summary.errors, (
        f"loewdin error: {summary.errors.get('loewdin')}"
    )
    assert "hirshfeld" not in summary.errors, (
        f"hirshfeld error: {summary.errors.get('hirshfeld')}"
    )
    assert "mayer" not in summary.errors, f"mayer error: {summary.errors.get('mayer')}"
    assert "dipole" not in summary.errors, (
        f"dipole error: {summary.errors.get('dipole')}"
    )
    assert summary.npa_atoms == []
    assert "Natural Atomic Orbital" in summary.unavailable["npa"]

    # Mulliken charges: one entry per atom (3 for H2O).
    assert len(summary.mulliken_atoms) == 3
    for i, sym, z, charge in summary.mulliken_atoms:
        assert isinstance(sym, str) and len(sym) in (1, 2)
        assert z > 0  # atomic number
        assert np.isfinite(charge), f"atom {i}: charge={charge}"

    # Löwdin charges likewise.
    assert len(summary.loewdin_atoms) == 3
    for i, sym, z, charge in summary.loewdin_atoms:
        assert np.isfinite(charge), f"atom {i}: charge={charge}"

    assert len(summary.hirshfeld_atoms) == 3
    for i, sym, z, charge in summary.hirshfeld_atoms:
        assert np.isfinite(charge), f"atom {i}: charge={charge}"

    # Mayer bond orders: at least the two O-H bonds.
    assert len(summary.mayer_bonds) >= 2
    oh_bonds = [b for b in summary.mayer_bonds if {b[2], b[3]} == {"O", "H"}]
    assert len(oh_bonds) == 2
    for bond in oh_bonds:
        assert 0.0 < bond[4] < 1.5, f"OH bond order = {bond[4]}"

    # Dipole moment: H2O has a non-zero dipole.
    assert summary.dipole is not None
    assert summary.dipole["total_debye"] > 0.1
    assert np.isfinite(summary.dipole["total_debye"])


def test_compute_population_summary_open_shell_spin_populations() -> None:
    """Unrestricted results carry per-atom Mulliken spin populations.

    Regression: the spin block sized its AO-to-atom map with
    ``shell.nfunctions()``, which ``ShellInfo`` does not have, so every
    UHF/UKS summary recorded an ``AttributeError`` under ``mulliken_spin``
    and no spin rows reached the QVF writer.
    """
    import vibeqc as vq

    # OH radical (doublet): the unpaired electron sits in an O 2p pi orbital.
    mol = vq.Molecule(
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.83])],
        multiplicity=2,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    result = vq.run_uhf(mol, basis)
    assert result.converged

    summary = compute_population_summary(result, basis, mol)

    assert "mulliken_spin" not in summary.errors, summary.errors["mulliken_spin"]
    assert [row[:3] for row in summary.mulliken_spin_atoms] == [
        (0, "O", 8.0),
        (1, "H", 1.0),
    ]
    spin = np.asarray([row[3] for row in summary.mulliken_spin_atoms], dtype=float)
    n_electrons = int(mol.n_electrons())
    n_alpha = (n_electrons + int(mol.multiplicity) - 1) // 2
    n_beta = n_electrons - n_alpha
    assert spin.sum() == pytest.approx(n_alpha - n_beta, abs=1e-8)
    # The sum holds for any AO-to-atom map of the right length; the unpaired
    # electron landing on O is what pins the per-atom assignment.
    assert spin[0] > 0.9


def test_compute_population_summary_complex_hermitian_density_is_quiet() -> None:
    import warnings

    import vibeqc as vq

    mol = vq.Molecule(
        [
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 1.43, -0.98]),
            vq.Atom(1, [0.0, -1.43, -0.98]),
        ]
    )
    basis = vq.BasisSet(mol, "sto-3g")
    result = vq.run_rhf(mol, basis)
    P_real = np.asarray(result.density)
    n = P_real.shape[0]
    anti = np.triu(np.ones((n, n)), 1)
    anti = anti - anti.T
    complex_density = P_real.astype(np.complex128) + 1.0e-8j * anti

    class _Result:
        density = complex_density

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        summary = compute_population_summary(_Result(), basis, mol)

    assert not any(
        "ComplexWarning" in warning.category.__name__
        or "imaginary" in str(warning.message)
        for warning in caught
    )
    assert summary.errors == {}
    assert "Natural Atomic Orbital" in summary.unavailable["npa"]
    assert len(summary.mulliken_atoms) == len(mol.atoms)
    assert summary.dipole is not None


def test_population_json_from_real_scf_has_charges(
    tmp_path: Path,
) -> None:
    """End-to-end: the .population.json emitted after a real SCF must
    contain numerical Mulliken charge entries (not an empty list).
    """
    import vibeqc as vq

    mol = vq.Molecule(
        [
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 1.43, -0.98]),
            vq.Atom(1, [0.0, -1.43, -0.98]),
        ]
    )
    basis = vq.BasisSet(mol, "sto-3g")
    result = vq.run_rhf(mol, basis)

    stem = tmp_path / "h2o"
    txt_path, json_path = write_population(stem, result, basis, mol)

    assert json_path.is_file()
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    mul = payload["mulliken"]
    assert len(mul) == 3, f"expected 3 mulliken entries, got {len(mul)}"
    for entry in mul:
        assert "charge" in entry
        assert isinstance(entry["charge"], (int, float))
        assert np.isfinite(entry["charge"])

    # The errors dict should be empty — every section succeeded.
    assert len(payload["hirshfeld"]) == 3
    for section in ("mulliken", "loewdin", "hirshfeld", "mayer", "dipole"):
        assert section not in payload.get("errors", {}), (
            f"{section}: {payload['errors'].get(section)}"
        )
    assert payload["npa"] == []
    assert "Natural Atomic Orbital" in payload["unavailable"]["npa"]


# ----------------------------------------------------------------------
# OutputPlan declares the two siblings
# ---------------------------------------------------------------------- #


def test_plan_declares_population_pair(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf",
        basis="sto-3g",
        functional=None,
        write_population=True,
    )
    paths = {str(f.path) for f in plan.files}
    assert str(tmp_path / "h2o.population.txt") in paths
    assert str(tmp_path / "h2o.population.json") in paths


def test_plan_write_population_false_drops_pair(tmp_path: Path) -> None:
    plan = OutputPlan.from_run_job_kwargs(
        output=tmp_path / "h2o",
        method="rhf",
        basis="sto-3g",
        functional=None,
        write_population=False,
    )
    pop_files = plan.files_by_role("population")
    assert pop_files == ()

@pytest.mark.parametrize("container", ["matrix", "list", "stack"])
@pytest.mark.parametrize("open_shell", [False, True])
def test_gdf_gamma_population_uses_accepted_metric_and_spin_density(
    monkeypatch, container, open_shell,
):
    from vibeqc.output.formats import population as pop

    mol = _Mol([_Atom(3, (0., 0., 0.)), _Atom(1, (1., 0., 0.))])
    basis = SimpleNamespace(nbasis=2)
    monkeypatch.setattr(pop, "_basis_ao_to_atom", lambda _: np.array([0, 1]))
    # S = L L^H, C = L^-H: C^H S C = I. Complex/nonorthogonal input
    # exposes both a home-cell-metric substitution and a transposed contraction.
    L = np.array([[1.4, 0.], [0.2 + 0.3j, 0.9]])
    S = L @ L.conj().T
    C = np.linalg.inv(L.conj().T) @ (np.array([[1., 1.], [-1., 1.]]) / np.sqrt(2.))
    occ_a, occ_b = np.array([0.8, 0.6]), np.array([0.3, 0.3])
    Da, Db = (C * occ_a) @ C.conj().T, (C * occ_b) @ C.conj().T

    def pack(a):
        return a if container == "matrix" else [a] if container == "list" else a[None]

    result = SimpleNamespace(
        overlap=pack(S), density=pack(Da + Db),
        density_alpha=pack(Da) if open_shell else None,
        density_beta=pack(Db) if open_shell else None,
        kpoints_cart=np.zeros((1, 3)), kpoint_weights=[1.],
        # Deliberately unrelated occupations: returned density is authoritative.
        mo_coeffs=pack(C), occupations=pack(np.zeros(2)),
    )
    summary = pop._compute_gdf_gamma_population_summary(
        result, basis, mol, nuclear_charges=[1., 1.], bond_threshold=-np.inf,
    )
    expected_pop = np.array([
        sum((Da + Db)[i, j] * S[j, i] for j in range(2)).real
        for i in range(2)
    ])
    np.testing.assert_allclose([r[3] for r in summary.mulliken_atoms], 1. - expected_pop, atol=1e-12)
    assert sum(r[3] for r in summary.mulliken_atoms) == pytest.approx(0., abs=1e-12)
    # Closed-form positive square root of a 2x2 matrix, no eigensolver oracle.
    determinant_root = np.sqrt(np.linalg.det(S).real)
    root = (S + determinant_root * np.eye(2)) / np.sqrt(np.trace(S).real + 2 * determinant_root)
    expected_low = 1. - np.diag(root @ (Da + Db) @ root).real
    np.testing.assert_allclose([r[3] for r in summary.loewdin_atoms], expected_low, atol=1e-12)
    assert sum(r[3] for r in summary.loewdin_atoms) == pytest.approx(0., abs=1e-12)
    assert abs(np.trace(Da + Db).real - 2.) > 0.1  # home-cell I would be wrong
    channels = [Da, Db] if open_shell else [Da + Db]
    factor = 2. if open_shell else 1.
    # Explicit four-index contraction, independent of the production helper.
    expected_bond = factor * sum(
        (D[0, u] * S[u, 1] * D[1, v] * S[v, 0]).real
        for D in channels for u in range(2) for v in range(2)
    )
    assert summary.mayer_bonds[0][4] == pytest.approx(expected_bond, abs=1e-12)
    if open_shell:
        expected_spin = [sum((Da - Db)[i, j] * S[j, i] for j in range(2)).real
                         for i in range(2)]
        np.testing.assert_allclose([r[3] for r in summary.mulliken_spin_atoms], expected_spin, atol=1e-12)
        assert sum(expected_spin) == pytest.approx(0.8, abs=1e-12)
    else:
        assert not summary.mulliken_spin_atoms
    assert set(summary.errors) == {"hirshfeld", "dipole", "wiberg"}
    assert all(v.startswith("unsupported:") for v in summary.errors.values())
    assert "npa" in summary.unavailable


@pytest.mark.parametrize("invalid", ["multi-k", "shifted", "weight", "half-spin", "shape", "nonfinite"])
def test_gdf_gamma_population_rejects_misaligned_state(monkeypatch, invalid):
    from vibeqc.output.formats import population as pop

    monkeypatch.setattr(pop, "_basis_ao_to_atom", lambda _: np.array([0, 1]))
    result = SimpleNamespace(overlap=[np.eye(2)], density=[np.eye(2)],
                             kpoints_cart=np.zeros((1, 3)), kpoint_weights=[1.])
    if invalid == "multi-k":
        result.overlap = [np.eye(2), np.eye(2)]
        result.density = [np.eye(2), np.eye(2)]
    elif invalid == "shifted":
        result.kpoints_cart[0, 0] = 0.1
    elif invalid == "weight":
        result.kpoint_weights = [0.5]
    elif invalid == "half-spin":
        result.density_alpha = [np.eye(2)]
    elif invalid == "shape":
        result.density = [np.eye(3)]
    elif invalid == "nonfinite":
        result.overlap[0][0, 0] = np.nan
    with pytest.raises(ValueError, match="GDF"):
        pop._compute_gdf_gamma_population_summary(
            result, SimpleNamespace(nbasis=2), _Mol([_Atom(1, (0, 0, 0)), _Atom(1, (1, 0, 0))]))


@pytest.mark.parametrize("mesh", [None, (1, 1, 1)], ids=["default-gamma", "explicit-gamma"])
@pytest.mark.parametrize("method", ["RHF", "RKS", "UHF", "UKS"])
def test_gdf_gamma_population_sidecar_uses_scf_overlap(tmp_path, mesh, method):
    import vibeqc as vq

    open_shell = method in ("UHF", "UKS")
    box = 12. if open_shell else 4.
    system = vq.PeriodicSystem(3, np.eye(3) * box,
                              [vq.Atom(1, [0., 0., -.7]), vq.Atom(1, [0., 0., .7])])
    system.multiplicity = 3 if open_shell else 1
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "gdf-pop"
    result = vq.run_periodic_job(
        system, basis, method=method, functional="LDA" if method.endswith("KS") else None,
        jk_method="gdf", kpoints=mesh, aux_basis="def2-svp-jk", output=stem,
        initial_guess="HCORE", max_iter=80, conv_tol_energy=1e-9,
        write_molden_file=False, write_density=False, write_population_file=True,
        output_qvf=True, density_spacing_bohr=0.5, progress=False,
    )
    assert result.converged
    def one(value):
        a = np.asarray(value)
        return a[0] if a.ndim == 3 else a
    # Default-Gamma RKS uses the UKS/GDF engine in its singlet limit, so
    # the returned density layout need not match the requested method name.
    from vibeqc.spin_channels import spin_densities
    density_alpha, density_beta = spin_densities(result)
    D = (one(density_alpha) + one(density_beta) if density_alpha is not None
         else one(result.density))
    S = one(result.overlap)
    expected = 1. - np.diag(D @ S).real  # exactly one s AO on each H
    payload = json.loads(stem.with_suffix(".population.json").read_text())
    np.testing.assert_allclose([r["charge"] for r in payload["mulliken"]], expected, atol=1e-10, rtol=0)
    assert sum(expected) == pytest.approx(0., abs=1e-8)
    assert sum(r["charge"] for r in payload["loewdin"]) == pytest.approx(0., abs=1e-8)
    assert all(key not in payload["errors"] for key in ("mulliken", "loewdin", "mayer"))
    if open_shell:
        assert "mulliken_spin" not in payload["errors"]
    with zipfile.ZipFile(stem.with_suffix(".qvf")) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        section, = [s for s in manifest["sections"] if s["kind"] == "atom_properties"]
        charges = np.frombuffer(archive.read(section["members"]["mulliken_charge"]["path"]), dtype=np.float64)
        if open_shell:
            spin = np.frombuffer(archive.read(section["members"]["spin_population"]["path"]), dtype=np.float64)
            expected_spin = np.diag((one(result.density_alpha) - one(result.density_beta)) @ S).real
            np.testing.assert_allclose(spin, expected_spin, atol=1e-10, rtol=0)
            assert sum(spin) == pytest.approx(2., abs=1e-8)
    np.testing.assert_allclose(charges, expected, atol=1e-10, rtol=0)
