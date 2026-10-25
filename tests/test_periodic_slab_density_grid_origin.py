"""Artifact grids must contain the atoms along a non-periodic axis (#45).

A 2-D UKS multi-k job on an H2 slab refused its QVF density artifact with a
primitive-cell grid integral of 1.07 electrons against an exact 2.0, and the
refusal advised decreasing ``density_spacing_bohr``.  Following that advice
made the number worse (1.14 at 0.4 bohr, 1.07 at 0.2), which is the tell that
the mesh was never the problem.

Every periodic density evaluator here image-sums AOs along the ``dim``
periodic axes only::

    range(-radius, radius + 1) if axis < dim else (0,)

That is correct physics: a slab is not periodic along a3, so there is no image
to sum.  The consequence is that along a synthesized axis the grid box has to
physically contain the atoms, because nothing translates them back into it.
The box was pinned at the Cartesian origin and spanned ``frac in [0, 1)^3``,
while slabs and chains are conventionally written straddling ``z = 0`` -- the
repository's own H2 slab fixture puts an atom at the origin, with half its
charge cloud below it.  That half simply had nowhere to be evaluated.

The fix centres the box on the atoms along each non-periodic axis and leaves a
fully periodic cell exactly where it was.  These tests pin both halves, and
the honesty of the remaining failure: a slab whose vacuum really is too thin
must still be refused, and must now be told that widening the cell, not
refining the mesh, is the remedy.

The number that #308 reports came from the *Gamma* route of the public runner,
not from the multi-k job above; the last section pins that route too.
"""

from __future__ import annotations

import json
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.periodic_runner import _qvf_grid_origin_bohr


def _h2_slab(z_lower: float, z_upper: float, c_bohr: float = 30.0):
    return vq.PeriodicSystem(
        2,
        np.diag([8.0, 8.0, c_bohr]),
        [vq.Atom(1, [0.0, 0.0, z_lower]), vq.Atom(1, [0.0, 0.0, z_upper])],
        0,
        1,
    )


def _run_density_artifact(system, tmp_path, spacing_bohr: float):
    """Run the 2-D multi-k UKS job and return its density-grid complaint."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vq.run_periodic_job(
            system=system,
            basis=vq.BasisSet(system.unit_cell_molecule(), "sto-3g"),
            method="uks",
            functional="pbe",
            kpoints=(3, 3, 1),
            output=str(tmp_path / "slab"),
            output_qvf=True,
            density_spacing_bohr=spacing_bohr,
        )
    for record in caught:
        message = str(record.message)
        if "grid integral" in message:
            return message
    return None


def test_bulk_cell_keeps_the_origin_it_always_had():
    """A fully periodic cell must not move: dim=3 stays bit-identical."""
    bulk = vq.PeriodicSystem(
        3,
        np.diag([6.0, 6.0, 6.0]),
        [vq.Atom(1, [0.0, 0.0, -0.7]), vq.Atom(1, [0.0, 0.0, 0.7])],
        0,
        1,
    )
    assert _qvf_grid_origin_bohr(bulk).tolist() == [0.0, 0.0, 0.0]


def test_slab_origin_centres_only_the_non_periodic_axis():
    slab = _h2_slab(-0.7, 0.7)
    origin = _qvf_grid_origin_bohr(slab)

    # The two periodic axes are untouched, so in-plane behaviour is unchanged.
    assert origin[0] == 0.0 and origin[1] == 0.0

    # Along a3 the atoms land in the middle of the box, not on its face.
    lattice = np.asarray(slab.lattice, dtype=float)
    coords = np.asarray([atom.xyz for atom in slab.unit_cell], dtype=float)
    frac_c = np.linalg.solve(lattice, (coords - origin).T)[2]
    assert frac_c.min() > 0.0 and frac_c.max() < 1.0
    assert 0.5 * (frac_c.min() + frac_c.max()) == pytest.approx(0.5, abs=1e-12)


def test_chain_origin_centres_both_non_periodic_axes():
    """dim=1 has two synthesized axes; both need containing, a1 does not."""
    chain = vq.PeriodicSystem(
        1,
        np.diag([4.1, 30.0, 30.0]),
        [vq.Atom(1, [0.17, 0.31, 0.0]), vq.Atom(3, [1.39, -0.22, 0.0])],
        0,
        2,
    )
    origin = _qvf_grid_origin_bohr(chain)
    assert origin[0] == 0.0
    lattice = np.asarray(chain.lattice, dtype=float)
    coords = np.asarray([atom.xyz for atom in chain.unit_cell], dtype=float)
    frac = np.linalg.solve(lattice, (coords - origin).T)
    for axis in (1, 2):
        assert frac[axis].min() > 0.0 and frac[axis].max() < 1.0
        assert 0.5 * (frac[axis].min() + frac[axis].max()) == pytest.approx(
            0.5, abs=1e-12
        )


@pytest.mark.parametrize(
    "z_lower, z_upper",
    [
        # An atom sitting exactly on the box face loses the half of its cloud
        # that falls below it: this is the repository's own slab convention.
        (0.0, 1.4),
        # The fixture that measured 1.0716 electrons of an exact 2.0.
        (-0.7, 0.7),
    ],
)
@pytest.mark.parametrize("spacing_bohr", [0.4, 0.2])
def test_slab_density_artifact_is_certified(
    tmp_path, z_lower, z_upper, spacing_bohr
):
    """The grid the artifact is certified on must hold all the charge."""
    complaint = _run_density_artifact(
        _h2_slab(z_lower, z_upper), tmp_path, spacing_bohr
    )
    assert complaint is None, complaint


def test_a_genuinely_thin_slab_is_still_refused_and_says_why(tmp_path):
    """Centring is not a licence to certify a box the density overflows.

    4 bohr of c leaves no room for the STO-3G tails even centred, so the
    artifact must still be refused -- and the refusal must point at the cell,
    not at the mesh, because refining the mesh cannot recover the charge.
    """
    complaint = _run_density_artifact(
        _h2_slab(-0.7, 0.7, c_bohr=4.0), tmp_path, 0.3
    )
    assert complaint is not None
    assert "periodic in 2 of 3 directions" in complaint
    assert "widen that lattice vector" in complaint
    assert "Refining density_spacing_bohr cannot recover it" in complaint
    assert "decrease density_spacing_bohr" not in complaint


# ---------------------------------------------------------------------------
# The Gamma route (#308)
# ---------------------------------------------------------------------------
#
# #308 reported the 1.0716 electrons on the Gamma route of the public runner,
# on ``tests/test_slab_2d_routing.py``'s H2 layer. ``_converged_gamma_
# density_grid_for_artifact`` takes the same centred origin as the lattice
# evaluator, so the #45 fix covers it, but nothing measured that route after
# the fix. Measured on main after it, as the integral of the written
# ``volume.density`` against an exact 2.0, for RKS and UKS alike:
# 1.9999846330 at 0.4 bohr, 2.0000001221 at 0.2 and 2.0000000000 at 0.1.
# Refinement converges instead of relocating the deficit, which is the
# closure criterion the issue asks for. With the origin pinned back at the
# Cartesian origin the same job is refused at 1.07 electrons again.


def _routing_h2_slab():
    """The dim=2 H2 layer of ``test_slab_2d_routing.py``: real z, a3 synthesized."""
    atoms = [vq.Atom(1, [9.0, 9.0, -0.7]), vq.Atom(1, [9.0, 9.0, 0.7])]
    return vq.slab_2d([18.0, 0.0, 0.0], [0.0, 18.0, 0.0], atoms)


def _written_density_integral(qvf_path: Path) -> tuple[float, tuple[int, ...]]:
    """Integrate the ``volume.density`` a QVF archive actually stores."""
    with zipfile.ZipFile(qvf_path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        section = next(
            item for item in manifest["sections"] if item["kind"] == "volume.density"
        )
        data = section["members"]["data"]
        rho = np.frombuffer(
            archive.read(data["path"]), dtype=np.dtype(data["dtype"])
        ).reshape(tuple(data["shape"]))
        grid = json.loads(archive.read(section["members"]["grid"]["path"]))
    voxels = np.asarray(grid["voxel_vectors"], dtype=float)
    return float(rho.sum() * abs(np.linalg.det(voxels))), tuple(data["shape"])


def _run_gamma_density_artifact(system, stem: Path, method: str, spacing_bohr: float):
    """Run the Gamma job through the public runner.

    Returns the density-grid complaint (``None`` when the artifact was
    certified) and the integral and shape of the volume the archive stores.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = vq.run_periodic_job(
            system=system,
            basis=vq.BasisSet(system.unit_cell_molecule(), "sto-3g"),
            method=method,
            functional="pbe",
            jk_method="auto",
            kpoints=(1, 1, 1),
            output=str(stem),
            output_qvf=True,
            density_spacing_bohr=spacing_bohr,
        )
    assert result.converged
    complaint = next(
        (str(record.message) for record in caught if "grid integral" in str(record.message)),
        None,
    )
    return complaint, _written_density_integral(Path(f"{stem}.qvf"))


def test_gamma_slab_origin_is_centred_on_the_synthesized_axis():
    """``slab_2d`` pads a3 to 31.4 bohr around z = 0; the box must straddle it."""
    assert _qvf_grid_origin_bohr(_routing_h2_slab()).tolist() == pytest.approx(
        [0.0, 0.0, -15.7], abs=1e-12
    )


@pytest.mark.parametrize("method", ["rks", "uks"])
@pytest.mark.parametrize("spacing_bohr", [0.4, 0.2])
def test_gamma_slab_density_artifact_holds_all_the_charge(tmp_path, method, spacing_bohr):
    """The Gamma route certifies and writes a volume that integrates to 2 electrons."""
    complaint, (integral, shape) = _run_gamma_density_artifact(
        _routing_h2_slab(), tmp_path / "gamma", method, spacing_bohr
    )
    assert complaint is None, complaint
    # The artifact's own 5e-3 gate, applied to the stored float32 volume
    # rather than to the certification's working array.
    assert integral == pytest.approx(2.0, abs=5.0e-3)
    # The grid spans the whole synthesized a3, which is where the lost half
    # of the charge used to fall outside.
    assert shape[2] * spacing_bohr >= 31.0


def test_gamma_slab_artifact_error_shrinks_under_refinement(tmp_path):
    """Refining the mesh must reduce the residual, not move it (#308)."""
    errors = []
    for spacing_bohr in (0.4, 0.2):
        complaint, (integral, _) = _run_gamma_density_artifact(
            _routing_h2_slab(), tmp_path / f"spacing{spacing_bohr}", "rks", spacing_bohr
        )
        assert complaint is None, complaint
        errors.append(abs(integral - 2.0))
    assert errors[0] < 1.0e-4, errors
    assert errors[1] < errors[0], errors
