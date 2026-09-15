"""NEB Increment 5 — NEBResult.write_qvf round-trip tests.

``NEBResult.write_qvf(stem)`` assembles an OutputPlan + reaction.path
context and delegates to ``vibeqc.output.formats.qvf.write_qvf``.
The archive carries:

* a ``structure`` section (reactant geometry),
* a ``reaction.path`` section (per-image frames + waypoints +
  energies + normalised reaction coordinate; for periodic NEB
  also the per-frame lattice + dim per QVF v2),
* a ``citations`` section (BibTeX assembled with
  ``uses_neb=True`` plus ``uses_ci_neb=True`` when the run was
  climbing-image).

These tests exercise:

* Molecular NEB → v1 archive, structure + reaction.path +
  citations sections, waypoints, energies, reaction coordinate.
* Citations include Henkelman+Jónsson 2000 (improved tangent) +
  Smidstrup 2014 (IDPP); add Henkelman+Uberuaga+Jónsson 2000
  (CI-NEB) iff the run was climbing-image.
* Vibe-view's ``QVFReader.read_reaction_path`` opens the archive
  and surfaces the path + waypoints intact.
* Periodic NEB → v2 archive with lattice + dim populated.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from vibeqc import (
    Atom,
    LatticeSumOptions,
    Molecule,
    PeriodicRHFOptions,
    PeriodicSystem,
    UHFOptions,
    run_neb,
)
from vibeqc.output.formats.qvf import validate_qvf


# ---- Helpers --------------------------------------------------------------


def _h3_doublet(z_positions):
    return Molecule(
        [Atom(1, [0.0, 0.0, float(z)]) for z in z_positions], 0, 2
    )


def _h2_in_box(z2: float) -> PeriodicSystem:
    # Keep this archive-format smoke in the molecular-limit regime so the
    # BIPOLE fold-convergence guard does not confuse QVF coverage with an
    # intentionally underconverged compact-cell calculation.
    L = np.diag([18.0, 18.0, 18.0])
    return PeriodicSystem(
        3, L, [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, float(z2)])]
    )


@pytest.fixture
def tight_periodic_opts() -> PeriodicRHFOptions:
    lat = LatticeSumOptions()
    lat.cutoff_bohr = 8.0
    opts = PeriodicRHFOptions()
    opts.lattice_opts = lat
    opts.max_iter = 30
    return opts


@pytest.fixture(scope="module")
def molecular_neb_result():
    """A cheap molecular NEB run — H + H₂ → H₂ + H, 3 images, no
    climbing image. Shared across the molecular-archive tests."""
    reactant = _h3_doublet([0.0, 1.4, 4.4])
    product = _h3_doublet([0.0, 3.0, 4.4])
    uhf = UHFOptions()
    uhf.max_iter = 200
    return run_neb(
        reactant, product,
        basis="sto-3g",
        n_images=3,
        method="UHF",
        uhf_options=uhf,
        interpolation="linear",
        max_iter=10,
        conv_tol_force=2e-2,
        n_jobs=1,
        initial_step=0.05,
    )


@pytest.fixture(scope="module")
def ci_neb_result():
    """A climbing-image NEB run — same system, climbing on."""
    reactant = _h3_doublet([0.0, 1.4, 4.4])
    product = _h3_doublet([0.0, 3.0, 4.4])
    uhf = UHFOptions()
    uhf.max_iter = 200
    return run_neb(
        reactant, product,
        basis="sto-3g",
        n_images=3,
        method="UHF",
        uhf_options=uhf,
        interpolation="linear",
        max_iter=15,
        conv_tol_force=2e-2,
        n_jobs=1,
        initial_step=0.05,
        climbing_image=True,
        climbing_image_start_fraction=0.3,
    )


# ---- Molecular archive (v1) ----------------------------------------------


class TestMolecularNEBQVF:
    def test_archive_validates_and_has_required_sections(
        self, molecular_neb_result, tmp_path: Path
    ) -> None:
        path = molecular_neb_result.write_qvf(tmp_path / "mol_neb")
        assert path.suffix == ".qvf"
        report = validate_qvf(path)
        assert report["valid"], report["errors"]
        with zipfile.ZipFile(path) as zf:
            mf = json.loads(zf.read("manifest.json"))
            kinds = {s["kind"] for s in mf["sections"]}
            assert kinds == {"structure", "reaction.path", "citations"}
            # Molecular NEB → v1 archive (no lattice on the
            # reaction.path section).
            assert mf["qvf_version"] == 1
            rxn = next(
                s for s in mf["sections"] if s["kind"] == "reaction.path"
            )
            assert "lattice" not in rxn["members"]

    def test_waypoints_carry_reactant_product_and_ts(
        self, molecular_neb_result, tmp_path: Path
    ) -> None:
        path = molecular_neb_result.write_qvf(tmp_path / "mol_wp")
        with zipfile.ZipFile(path) as zf:
            rxn = next(
                s for s in json.loads(zf.read("manifest.json"))["sections"]
                if s["kind"] == "reaction.path"
            )
            meta = json.loads(zf.read(rxn["members"]["metadata"]["path"]))
            kinds = {w["kind"] for w in meta["waypoints"]}
            assert kinds == {"reactant", "transition_state", "product"}
            # reaction_coordinate spans [0, 1] and is monotonically
            # non-decreasing (it's cumulative arc length normalised).
            rc = meta["reaction_coordinate"]
            assert rc[0] == pytest.approx(0.0)
            assert rc[-1] == pytest.approx(1.0)
            assert all(rc[i + 1] >= rc[i] for i in range(len(rc) - 1))

    def test_energies_match_result_per_image(
        self, molecular_neb_result, tmp_path: Path
    ) -> None:
        path = molecular_neb_result.write_qvf(tmp_path / "mol_e")
        with zipfile.ZipFile(path) as zf:
            rxn = next(
                s for s in json.loads(zf.read("manifest.json"))["sections"]
                if s["kind"] == "reaction.path"
            )
            meta = json.loads(zf.read(rxn["members"]["metadata"]["path"]))
            for e_qvf, e_res in zip(
                meta["energies"], molecular_neb_result.energies
            ):
                assert float(e_qvf) == pytest.approx(float(e_res), abs=1e-12)

    def test_citations_section_carries_henkelman_and_smidstrup(
        self, molecular_neb_result, tmp_path: Path
    ) -> None:
        path = molecular_neb_result.write_qvf(tmp_path / "mol_cite")
        with zipfile.ZipFile(path) as zf:
            bib = zf.read("citations/references.bib").decode("utf-8")
        assert "henkelman_jonsson_neb_2000" in bib
        assert "smidstrup_idpp_2014" in bib
        # Plain (non-climbing) NEB → CI-NEB paper must NOT appear.
        assert "henkelman_uberuaga_jonsson_ci_neb_2000" not in bib

    def test_vibe_view_reader_round_trips_the_path(
        self, molecular_neb_result, tmp_path: Path
    ) -> None:
        vibeview_qvf = pytest.importorskip(
            "vibeview.qvf",
            reason="vibe-view QVF reader is not installed in this environment",
        )
        QVFReader = vibeview_qvf.QVFReader

        path = molecular_neb_result.write_qvf(tmp_path / "mol_reader")
        reader = QVFReader(path)
        try:
            assert reader.manifest.qvf_version == 1
            section_id = next(
                s.id
                for s in reader.manifest.sections
                if s.kind == "reaction.path"
            )
            data = reader.read_reaction_path(section_id)
            assert data.lattice is None
            assert data.dim is None
            assert data.coords.shape == (
                len(molecular_neb_result.path.images),
                len(molecular_neb_result.path.images[0].system.atoms),
                3,
            )
            wp_kinds = {w.kind for w in data.waypoints}
            assert wp_kinds == {"reactant", "transition_state", "product"}
        finally:
            reader.close()


# ---- Climbing-image archive ----------------------------------------------


class TestCINEBQVF:
    def test_ci_neb_citations_include_climbing_paper(
        self, ci_neb_result, tmp_path: Path
    ) -> None:
        path = ci_neb_result.write_qvf(tmp_path / "ci_neb")
        with zipfile.ZipFile(path) as zf:
            bib = zf.read("citations/references.bib").decode("utf-8")
        assert "henkelman_uberuaga_jonsson_ci_neb_2000" in bib
        assert "henkelman_jonsson_neb_2000" in bib
        assert "smidstrup_idpp_2014" in bib

    def test_ts_waypoint_points_at_climbing_image(
        self, ci_neb_result, tmp_path: Path
    ) -> None:
        path = ci_neb_result.write_qvf(tmp_path / "ci_ts")
        with zipfile.ZipFile(path) as zf:
            rxn = next(
                s for s in json.loads(zf.read("manifest.json"))["sections"]
                if s["kind"] == "reaction.path"
            )
            meta = json.loads(zf.read(rxn["members"]["metadata"]["path"]))
        ts_wp = next(
            w for w in meta["waypoints"] if w["kind"] == "transition_state"
        )
        assert ts_wp["frame_index"] == ci_neb_result.path.climbing_image_index


# ---- Periodic archive (v2) -----------------------------------------------


class TestPeriodicNEBQVF:
    def test_periodic_archive_is_v1_with_lattice_member(
        self, tight_periodic_opts, tmp_path: Path
    ) -> None:
        reactant = _h2_in_box(1.4)
        product = _h2_in_box(1.8)
        result = run_neb(
            reactant, product,
            basis="sto-3g",
            n_images=2,
            method="RHF",
            rhf_options=tight_periodic_opts,
            interpolation="linear",
            max_iter=1,
            conv_tol_force=1e-1,
            n_jobs=1,
            kpoints=(1, 1, 1),
            fd_step_bohr=5e-3,
        )
        path = result.write_qvf(tmp_path / "periodic_neb")
        report = validate_qvf(path)
        assert report["valid"], report["errors"]
        with zipfile.ZipFile(path) as zf:
            mf = json.loads(zf.read("manifest.json"))
            # Ruling 2026-07-10: periodic paths stay qvf_version 1; the
            # `lattice` member (asserted below) is what marks them periodic.
            assert mf["qvf_version"] == 1
            rxn = next(
                s for s in mf["sections"] if s["kind"] == "reaction.path"
            )
            assert "lattice" in rxn["members"]
            lat_member = rxn["members"]["lattice"]
            assert lat_member["shape"] == [3, 3]
            meta = json.loads(zf.read(rxn["members"]["metadata"]["path"]))
            assert meta["dim"] == 3
