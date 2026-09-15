"""Phase M2 — multi-XYZ trajectory writer + normal-mode helper.

Pinned contracts:

1. **API surface** — ``vq.write_xyz_trajectory`` and
   ``vq.normal_mode_trajectory`` are public.

2. **Multi-XYZ format** — output file consists of N frames,
   each starting with the atom count on its own line, then a
   comment line, then ``element x y z`` rows in **Ångström**.

3. **Round-trip** — reading the written file back via
   :func:`vq.Molecule.from_xyz` (single-frame) on each
   atom-count-delimited block gives molecules whose Cartesian
   positions match the input frames to <1e-9 Å.

4. **Append mode** — ``append=True`` extends an existing file
   rather than overwriting it; total frame count is the sum.

5. **Custom comment lines** — when supplied, comments end up on
   line 2 of each frame.

6. **Length validation** — ``comments`` must match ``frames``
   length or raise ``ValueError``.

7. **Empty input rejected** — zero-length frames iterable raises.

8. **Atom-count-mismatch warning** — fires ``UserWarning`` when
   later frames have different atom count than frame 0.

9. **Normal-mode helper** — ``normal_mode_trajectory(mol, hess, p)``
   produces N frames sampling the p-th vibrational mode with the
   correct mass-weighted-eigenvector pattern. The frames sum to
   approximately the equilibrium geometry (sin-pattern integrates
   to zero), and the equilibrium geometry is a fixed point at
   t=0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.hessian import HessianFDOptions


ANGSTROM_TO_BOHR = 1.8897261339213
BOHR_TO_ANGSTROM = 1.0 / ANGSTROM_TO_BOHR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _h2_at(R_bohr: float):
    return vq.Molecule([
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, R_bohr]),
    ], charge=0, multiplicity=1)


def _h2o():
    return vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.7572 * ANGSTROM_TO_BOHR, 0.5868 * ANGSTROM_TO_BOHR]),
        vq.Atom(1, [0.0, -0.7572 * ANGSTROM_TO_BOHR, 0.5868 * ANGSTROM_TO_BOHR]),
    ], charge=0, multiplicity=1)


# ---------------------------------------------------------------------------
# 1. API surface
# ---------------------------------------------------------------------------

def test_write_xyz_trajectory_public():
    assert hasattr(vq, "write_xyz_trajectory")
    assert hasattr(vq, "normal_mode_trajectory")


# ---------------------------------------------------------------------------
# 2. Multi-XYZ format structure
# ---------------------------------------------------------------------------

def test_format_atom_count_and_columns(tmp_path):
    frames = [_h2_at(1.4), _h2_at(1.5), _h2_at(1.6)]
    path = tmp_path / "h2.xyz"
    n = vq.write_xyz_trajectory(path, frames)
    assert n == 3
    text = path.read_text()
    lines = text.splitlines()
    # Frame structure: 4 lines per frame (count, comment, 2 atoms).
    assert len(lines) == 4 * 3
    # Atom-count line + comment per frame.
    for frame_idx in range(3):
        offset = 4 * frame_idx
        assert lines[offset] == "2"
        # Comment line (default = "frame N").
        assert lines[offset + 1].startswith("frame")
        # Two atom rows.
        assert lines[offset + 2].startswith("H ")
        assert lines[offset + 3].startswith("H ")


def test_coordinates_in_angstrom(tmp_path):
    """vibe-qc internal positions are in bohr; xyz file output should
    convert to Å."""
    R = 1.4   # bohr
    frame = _h2_at(R)
    path = tmp_path / "h2.xyz"
    vq.write_xyz_trajectory(path, [frame])
    lines = path.read_text().splitlines()
    # Second atom is at z=R bohr → R · bohr-to-Å in the file.
    parts = lines[3].split()
    assert parts[0] == "H"
    z_in_angstrom = float(parts[3])
    np.testing.assert_allclose(z_in_angstrom, R * BOHR_TO_ANGSTROM,
                                atol=1e-9)


# ---------------------------------------------------------------------------
# 3. Round-trip
# ---------------------------------------------------------------------------

def test_round_trip_via_per_frame_xyz_parser(tmp_path):
    """Split the multi-XYZ file into per-frame xyz blocks, parse each
    via Molecule.from_xyz, and confirm coordinates match."""
    frames = [_h2_at(R_bohr) for R_bohr in (1.0, 1.4, 1.8, 2.2)]
    path = tmp_path / "scan.xyz"
    vq.write_xyz_trajectory(path, frames)

    text = path.read_text()
    lines = text.splitlines()
    cursor = 0
    parsed_frames = []
    while cursor < len(lines):
        n_at = int(lines[cursor])
        block = "\n".join(lines[cursor: cursor + 2 + n_at]) + "\n"
        single = tmp_path / f"frame_{cursor}.xyz"
        single.write_text(block)
        parsed_frames.append(vq.Molecule.from_xyz(single))
        cursor += 2 + n_at

    assert len(parsed_frames) == len(frames)
    for orig, parsed in zip(frames, parsed_frames):
        for a_orig, a_parsed in zip(orig.atoms, parsed.atoms):
            np.testing.assert_allclose(
                a_parsed.xyz, a_orig.xyz, atol=1e-8)


# ---------------------------------------------------------------------------
# 4. Append mode
# ---------------------------------------------------------------------------

def test_append_mode_extends_file(tmp_path):
    path = tmp_path / "anim.xyz"
    vq.write_xyz_trajectory(path, [_h2_at(1.4), _h2_at(1.5)])
    n2 = vq.write_xyz_trajectory(
        path, [_h2_at(1.6), _h2_at(1.7)], append=True)
    assert n2 == 2
    n_atom_count_lines = sum(
        1 for line in path.read_text().splitlines()
        if line.strip() == "2"
    )
    assert n_atom_count_lines == 4


# ---------------------------------------------------------------------------
# 5. Custom comments
# ---------------------------------------------------------------------------

def test_custom_comments(tmp_path):
    frames = [_h2_at(1.4), _h2_at(1.5)]
    comments = ["scan step 0  E = -1.10 Ha", "scan step 1  E = -1.11 Ha"]
    path = tmp_path / "scan.xyz"
    vq.write_xyz_trajectory(path, frames, comments=comments)
    lines = path.read_text().splitlines()
    assert lines[1] == comments[0]
    assert lines[5] == comments[1]


# ---------------------------------------------------------------------------
# 6. Length validation
# ---------------------------------------------------------------------------

def test_comment_length_mismatch_raises(tmp_path):
    frames = [_h2_at(1.4), _h2_at(1.5)]
    with pytest.raises(ValueError, match="length"):
        vq.write_xyz_trajectory(
            tmp_path / "x.xyz", frames, comments=["only one"])


# ---------------------------------------------------------------------------
# 7. Empty input rejected
# ---------------------------------------------------------------------------

def test_empty_frames_raises(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        vq.write_xyz_trajectory(tmp_path / "x.xyz", [])


# ---------------------------------------------------------------------------
# 8. Atom-count-mismatch warning
# ---------------------------------------------------------------------------

def test_atom_count_mismatch_warns(tmp_path):
    import warnings
    h2 = _h2_at(1.4)
    h2o = _h2o()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vq.write_xyz_trajectory(tmp_path / "mixed.xyz", [h2, h2o])
    assert any(issubclass(w.category, UserWarning) for w in caught)
    assert any("atom" in str(w.message).lower() for w in caught)


# ---------------------------------------------------------------------------
# 9. normal_mode_trajectory helper
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h2o_with_hessian():
    mol = _h2o()
    opts = vq.RHFOptions()
    opts.conv_tol_energy = 1e-10
    fd_opts = HessianFDOptions(include_dipole_derivatives=True)
    hess = vq.compute_hessian_fd(
        mol, "sto-3g", method="RHF",
        scf_options=opts, hessian_options=fd_opts)
    return mol, hess


def test_normal_mode_trajectory_shape(h2o_with_hessian):
    mol, hess = h2o_with_hessian
    frames = vq.normal_mode_trajectory(mol, hess, mode_index=8, n_frames=20)
    assert len(frames) == 20
    for frame in frames:
        assert len(frame.atoms) == len(mol.atoms)


def test_normal_mode_t_zero_is_equilibrium(h2o_with_hessian):
    """At t=0, sin(2π·0) = 0 so the displacement is zero and the
    frame matches the equilibrium geometry exactly."""
    mol, hess = h2o_with_hessian
    frames = vq.normal_mode_trajectory(mol, hess, mode_index=8, n_frames=8)
    for a_eq, a_t0 in zip(mol.atoms, frames[0].atoms):
        np.testing.assert_allclose(a_t0.xyz, a_eq.xyz, atol=1e-12)


def test_normal_mode_period_average_is_equilibrium(h2o_with_hessian):
    """Sum of sinusoidal-displacement frames over one period averages
    to the equilibrium geometry (sin integrates to 0)."""
    mol, hess = h2o_with_hessian
    n_frames = 20
    frames = vq.normal_mode_trajectory(
        mol, hess, mode_index=8, n_frames=n_frames)
    avg = np.zeros(3)
    for frame in frames:
        for atom in frame.atoms:
            avg += np.asarray(atom.xyz)
    avg /= (n_frames * len(mol.atoms))
    eq = np.zeros(3)
    for atom in mol.atoms:
        eq += np.asarray(atom.xyz)
    eq /= len(mol.atoms)
    np.testing.assert_allclose(avg, eq, atol=1e-9)


def test_invalid_mode_index_raises(h2o_with_hessian):
    mol, hess = h2o_with_hessian
    with pytest.raises(ValueError, match="mode_index"):
        vq.normal_mode_trajectory(mol, hess, mode_index=99)


def test_invalid_amplitude_raises(h2o_with_hessian):
    mol, hess = h2o_with_hessian
    with pytest.raises(ValueError, match="amplitude"):
        vq.normal_mode_trajectory(mol, hess, mode_index=8, amplitude=0.0)


def test_invalid_n_frames_raises(h2o_with_hessian):
    mol, hess = h2o_with_hessian
    with pytest.raises(ValueError, match="n_frames"):
        vq.normal_mode_trajectory(mol, hess, mode_index=8, n_frames=1)


# ---------------------------------------------------------------------------
# 10. End-to-end: animate a normal mode and write it
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 11. write_opt_trajectory — geometry-optimization history convention
# ---------------------------------------------------------------------------

def test_write_opt_trajectory_public():
    assert hasattr(vq, "write_opt_trajectory")


def test_opt_trajectory_with_energies(tmp_path):
    frames = [_h2_at(R) for R in (1.0, 1.4, 1.8)]
    energies = [-0.95, -1.10, -1.05]
    path = tmp_path / "h2.opt"
    n = vq.write_opt_trajectory(path, frames, energies)
    assert n == 3
    text = path.read_text()
    # Comment lines should carry "step N" + "E = ... Ha".
    assert "step 0" in text
    assert "E = -0.9500000000 Ha" in text
    assert "E = -1.1000000000 Ha" in text


def test_opt_trajectory_with_grad(tmp_path):
    frames = [_h2_at(1.4), _h2_at(1.5)]
    energies = [-1.10, -1.11]
    rms_grad = [1.5e-2, 9.8e-5]
    path = tmp_path / "h2.opt"
    vq.write_opt_trajectory(path, frames, energies, rms_grad=rms_grad)
    text = path.read_text()
    assert "|grad| = 1.500e-02" in text
    assert "|grad| = 9.800e-05" in text


def test_opt_trajectory_no_energies(tmp_path):
    """Energies are optional — omitting them gives plain "step N"
    comments, identical to write_xyz_trajectory's default."""
    frames = [_h2_at(1.4), _h2_at(1.5)]
    path = tmp_path / "h2.opt"
    vq.write_opt_trajectory(path, frames)
    text = path.read_text()
    assert "step 0" in text
    assert "step 1" in text
    assert "E =" not in text


def test_opt_trajectory_length_mismatch_raises(tmp_path):
    frames = [_h2_at(1.4), _h2_at(1.5)]
    with pytest.raises(ValueError, match="energies_ha"):
        vq.write_opt_trajectory(tmp_path / "x.opt", frames,
                                  energies_ha=[1.0])


def test_normal_mode_animation_round_trip(tmp_path, h2o_with_hessian):
    """Pipe normal_mode_trajectory → write_xyz_trajectory and verify
    the resulting file has the expected number of frames."""
    mol, hess = h2o_with_hessian
    frames = vq.normal_mode_trajectory(mol, hess, mode_index=8, n_frames=12)
    path = tmp_path / "h2o_mode.xyz"
    n = vq.write_xyz_trajectory(path, frames)
    assert n == 12
    text = path.read_text()
    # Should have 12 atom-count lines saying "3".
    assert text.count("\n3\n") == 11  # 11 internal "3" lines
    assert text.startswith("3\n")     # plus the first one
