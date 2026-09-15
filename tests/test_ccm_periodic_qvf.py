"""Vibe-view-ready periodic QVF emission for the Γ-CCM (``write_ccm_periodic_qvf``).

Pins the vibe-view contract that the periodic renderer keys off:

* ``structure.pbc`` reflects the true dimensionality; ``lattice_vectors`` are
  ANGSTROM row vectors matching the BvK supercell period (not 0.53×/transposed);
* the torus density/orbital grids span exactly one cell (``n_i·step_i == L_i``)
  and the density integrates to N electrons (the wrap is in the data);
* the ``x_ccm.wannier_centers`` overlay has one centre per Wannier function with a
  **positive** spread (no position-operator aliasing) sitting where expected;
* ``validate_qvf`` accepts the archive end-to-end.
"""

from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.output.formats.qvf import validate_qvf
from vibeqc.periodic.ccm import CCMSystem, write_ccm_periodic_qvf
from vibeqc.periodic.ccm.neutral import ccm_eri_neutral
from vibeqc.periodic.ccm.scf import run_ccm_rhf

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane

_BOHR = 0.529177210903


@pytest.fixture(scope="module")
def h_chain_qvf(tmp_path_factory):
    """A 1-D H₂-dimerized chain (period 16 bohr), atom on the x=0 boundary so a
    Wannier function straddles the cell face -- the clearest wrap case."""
    W = 12.0
    unit = PeriodicSystem(3, np.diag([4.0, W, W]),
                          [Atom(1, [0.0, W / 2, W / 2]),
                           Atom(1, [1.4, W / 2, W / 2])], 0, 1)
    ccm = CCMSystem(unit, (4, 1, 1), "sto-3g")
    res = run_ccm_rhf(ccm, eri=ccm_eri_neutral(ccm, ke_cutoff=40.0))
    stem = tmp_path_factory.mktemp("qvf") / "h_chain"
    path = write_ccm_periodic_qvf(ccm, res, stem, dim=1, basis="sto-3g",
                                  grid_spacing_bohr=0.25)
    return ccm, path


def test_qvf_validates_and_has_periodic_sections(h_chain_qvf):
    ccm, path = h_chain_qvf
    assert path.exists() and path.suffix == ".qvf"
    assert validate_qvf(path)["valid"] is True
    with zipfile.ZipFile(path) as zf:
        kinds = [s["kind"] for s in json.loads(zf.read("manifest.json"))["sections"]]
    assert "structure" in kinds
    assert "volume.density" in kinds
    assert kinds.count("volume.orbital") >= 1
    assert "x_ccm.wannier_centers" in kinds


def test_structure_pbc_and_lattice(h_chain_qvf):
    ccm, path = h_chain_qvf
    with zipfile.ZipFile(path) as zf:
        struct = json.loads(zf.read("structure/structure.json"))
    assert struct["pbc"] == [True, False, False]              # 1-D chain
    lv = np.asarray(struct["lattice_vectors"], float)         # ANGSTROM, row vectors
    period_ang = float(ccm.cluster_vectors[0, 0]) * _BOHR
    assert lv[0, 0] == pytest.approx(period_ang, rel=1e-6)    # |a| not 0.53× / transposed
    assert lv[1, 0] == 0.0 and lv[0, 1] == 0.0                # diagonal box


def test_torus_grid_spans_cell_and_density_integrates(h_chain_qvf):
    ccm, path = h_chain_qvf
    with zipfile.ZipFile(path) as zf:
        man = json.loads(zf.read("manifest.json"))
        dens = next(s for s in man["sections"] if s["kind"] == "volume.density")
        grid = json.loads(zf.read(dens["members"]["grid"]["path"]))
        nx = grid["shape"][0]
        step_x = grid["voxel_vectors"][0][0]                  # bohr
        data = np.frombuffer(zf.read(dens["members"]["data"]["path"]), dtype=np.float32)
    # grid spans exactly one cell along the periodic axis
    assert nx * step_x == pytest.approx(float(ccm.cluster_vectors[0, 0]), rel=1e-6)
    # density integrates to N electrons
    lat = np.asarray(ccm.cluster_lattice, float)
    dV = abs(np.linalg.det(lat)) / data.size
    assert float(data.sum()) * dV == pytest.approx(ccm.supercell.n_electrons(), abs=5e-2)


def test_wannier_centers_positive_spread_and_placed(h_chain_qvf):
    ccm, path = h_chain_qvf
    with zipfile.ZipFile(path) as zf:
        wc = json.loads(zf.read("x_ccm_wannier_centers/data.json"))["centers"]
    n_occ = ccm.supercell.n_electrons() // 2
    assert len(wc) == n_occ
    for c in wc:
        assert c["spread"] > 0.0                              # no position-operator aliasing
        assert len(c["center"]) == 3
    # centres sit near the H₂ bond midpoints (~0.7 + k·4 bohr → ang), incl. the
    # boundary-straddling one (wrap-aware circular mean, not aliased)
    xs = sorted((c["center"][0] / _BOHR) % float(ccm.cluster_vectors[0, 0]) for c in wc)
    for x, expect in zip(xs, [0.7, 4.7, 8.7, 12.7]):
        assert abs(x - expect) < 0.6
