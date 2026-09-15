"""Multi-XYZ trajectory writer.

Phase M2 -- companion to ``write_orca_hess`` (M1, Phase 17 visualization
interop). Where M1 ships vibrational modes into the moltui /
chemcraft / VMD ecosystem, M2 ships **geometry sequences**: the same
ecosystem (plus ASE, OVITO, Avogadro, PyMOL) reads multi-XYZ
trajectories and animates them frame-by-frame.

The multi-XYZ format is the de-facto interchange format for trajectory
data -- geometry-optimization paths, NEB images, MD trajectories,
normal-mode-displacement movies. Each frame is just a standard XYZ
file (atom count + comment line + ``element x y z`` rows) and frames
are concatenated with no separator. The format is fully ASCII,
documented, and version-stable.

Use cases vibe-qc users hit today:

  - **Geometry optimization animation**: write each ``run_job(...)``
    iterate as a frame, then ``moltui opt.xyz`` shows the optimization
    path animated.
  - **Normal-mode movie**: discretise a normal mode into N frames at
    ±A.L_p amplitudes and write them as a multi-XYZ -- moltui then
    animates the vibration.
  - **NEB path**: each NEB image is a frame; visualize the reaction
    path.
  - **MD snapshots**: dump every Nth MD step; replay later.

Public API:

    vq.write_xyz_trajectory(path, frames, comments=None, *, append=False)
    vq.normal_mode_trajectory(mol, hessian_result, mode_index, *,
                                amplitude=0.5, n_frames=20) -> list of
                                Molecules; pair with
                                ``write_xyz_trajectory`` to drop into
                                moltui.

Coordinate units in the file are **Ångström** (matching the standard
XYZ convention; vibe-qc internally uses bohr but converts on write).
"""

from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np

from ..._vibeqc_core import Atom, Molecule

from .._text_safety import scrub_output_text


__all__ = [
    "write_xyz_trajectory",
    "write_opt_trajectory",
    "normal_mode_trajectory",
]


# 1 bohr = 0.529177210903 Å (CODATA 2018).
_BOHR_TO_ANGSTROM = 0.529177210903


# ----------------------------------------------------------------------
# Element symbol table (1..118), shared with ORCA .hess writer.
# ----------------------------------------------------------------------

_ELEMENT_SYMBOLS = (
    "H",  "He", "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar", "K",  "Ca",
    "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",  "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I",  "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U",  "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
    "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
)


def _element_symbol(z: int) -> str:
    if 1 <= z <= len(_ELEMENT_SYMBOLS):
        return _ELEMENT_SYMBOLS[z - 1]
    return f"Z{z}"


# ----------------------------------------------------------------------
# Multi-XYZ writer
# ----------------------------------------------------------------------

def _write_one_xyz_frame(fh, mol: Molecule, comment: str = "") -> None:
    """Write one frame: atom count, comment line, then
    ``symbol x y z`` rows in Å."""
    n = len(mol.atoms)
    fh.write(f"{n}\n")
    # Per-frame comment line is user-controlled (write_xyz_trajectory
    # comments=); neutralise bidi / control chars. See _text_safety.
    fh.write(f"{scrub_output_text(comment)}\n")
    for atom in mol.atoms:
        sym = _element_symbol(int(atom.Z))
        x = atom.xyz[0] * _BOHR_TO_ANGSTROM
        y = atom.xyz[1] * _BOHR_TO_ANGSTROM
        z = atom.xyz[2] * _BOHR_TO_ANGSTROM
        fh.write(f"{sym:<3s} {x: .10f} {y: .10f} {z: .10f}\n")


def write_xyz_trajectory(
    path: Union[str, os.PathLike],
    frames: Iterable[Molecule],
    comments: Optional[Sequence[str]] = None,
    *,
    append: bool = False,
) -> int:
    """Write a sequence of :class:`Molecule` frames as a multi-XYZ
    trajectory file.

    Parameters
    ----------
    path:
        Output file path.
    frames:
        Iterable of :class:`Molecule` objects, one per trajectory
        frame. All frames should share the same atom count and
        ordering -- most viewers (moltui, ASE, OVITO) assume this so
        atom-by-atom interpolation makes sense. We don't enforce it
        here (you can write a chemistry-changing trajectory like a
        bond-cleavage if you really want), but a UserWarning is
        emitted if frame N has a different atom count than frame 1.
    comments:
        Optional per-frame comment strings (one per frame). The
        XYZ format puts comments on line 2 of each frame; viewers
        often display them as labels. Default: numeric ``"frame N"``.
    append:
        If ``True``, open in append mode so the new frames extend
        an existing trajectory file. Default: overwrite.

    Returns
    -------
    int
        The number of frames written.

    Examples
    --------
    Geometry-opt animation::

        opts = []                                    # collect mols
        for it in range(50):
            mol = step(mol, gradient_at(mol))
            opts.append(mol)
        vq.write_xyz_trajectory("opt.xyz", opts)

    Normal-mode movie (helper below produces the frames for you)::

        frames = vq.normal_mode_trajectory(mol, hess, mode_index=8)
        vq.write_xyz_trajectory("h2o-asym-stretch.xyz", frames)
    """
    frame_list = list(frames)
    if not frame_list:
        raise ValueError(
            "write_xyz_trajectory: frames is empty; need at least 1 frame."
        )

    n_atoms_first = len(frame_list[0].atoms)

    if comments is None:
        comments = [f"frame {i}" for i in range(len(frame_list))]
    elif len(comments) != len(frame_list):
        raise ValueError(
            f"write_xyz_trajectory: comments length {len(comments)} "
            f"!= frames length {len(frame_list)}."
        )

    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as fh:
        for i, (mol, cmt) in enumerate(zip(frame_list, comments)):
            if len(mol.atoms) != n_atoms_first:
                import warnings
                warnings.warn(
                    f"write_xyz_trajectory: frame {i} has "
                    f"{len(mol.atoms)} atoms vs frame 0's "
                    f"{n_atoms_first}. Most viewers assume a fixed "
                    f"atom count -- output may not animate cleanly.",
                    UserWarning,
                    stacklevel=2,
                )
            _write_one_xyz_frame(fh, mol, str(cmt))
    return len(frame_list)


# ----------------------------------------------------------------------
# Geometry-optimization history -- convenience wrapper that auto-
# formats the comment line with the per-step energy.
# ----------------------------------------------------------------------

def write_opt_trajectory(
    path: Union[str, os.PathLike],
    frames: Sequence[Molecule],
    energies_ha: Optional[Sequence[float]] = None,
    *,
    rms_grad: Optional[Sequence[float]] = None,
    append: bool = False,
) -> int:
    """Write a geometry-optimization history as a multi-XYZ trajectory
    with per-step energies on the comment line.

    Same on-disk format as :func:`write_xyz_trajectory` -- multi-XYZ
    with one frame per optimization iteration. The convention here
    is just the comment-line content: each frame's comment reads::

        step N  E = -76.026123456789 Ha  |grad| = 1.23e-04

    so the file self-documents. moltui, OVITO, ASE, and Avogadro all
    happily read this back as a regular multi-XYZ trajectory; the
    ``E = ...`` token in the comment is also recognized by ASE
    (``ase.io.read`` parses it into the per-frame ``Atoms.info``
    dict).

    Standard extension is ``.opt`` (matching the ORCA-style
    geometry-history convention) but the writer doesn't enforce it --
    pass any path you like.

    Parameters
    ----------
    path:
        Output file path (default extension ``.opt`` by convention).
    frames:
        Optimization iterates, ``frames[k]`` = geometry at step ``k``.
    energies_ha:
        Optional ``(N,)`` per-step total energies in Hartree. Goes
        into the comment line. If ``None``, the comment is just
        ``"step N"``.
    rms_grad:
        Optional ``(N,)`` per-step RMS gradient (any consistent unit;
        we just print the number). When supplied, joined into the
        comment as ``"|grad| = ..."`` (literal pipe characters
        around ``grad``) so converged steps are easy to spot.
    append:
        If ``True``, extend an existing ``.opt`` file rather than
        overwriting. Useful when an optimizer writes incrementally.

    Returns
    -------
    int
        Number of frames written.

    Examples
    --------
    >>> opt_geoms, opt_energies = run_optimization(mol)
    >>> vq.write_opt_trajectory("h2o.opt", opt_geoms, opt_energies)
    """
    n = len(frames)
    if energies_ha is not None and len(energies_ha) != n:
        raise ValueError(
            f"write_opt_trajectory: energies_ha length {len(energies_ha)} "
            f"!= frames length {n}."
        )
    if rms_grad is not None and len(rms_grad) != n:
        raise ValueError(
            f"write_opt_trajectory: rms_grad length {len(rms_grad)} "
            f"!= frames length {n}."
        )
    comments: List[str] = []
    for k in range(n):
        bits = [f"step {k}"]
        if energies_ha is not None:
            bits.append(f"E = {float(energies_ha[k]):.10f} Ha")
        if rms_grad is not None:
            bits.append(f"|grad| = {float(rms_grad[k]):.3e}")
        comments.append("  ".join(bits))
    return write_xyz_trajectory(path, frames, comments=comments,
                                  append=append)


# ----------------------------------------------------------------------
# Normal-mode trajectory helper -- discretise a vibrational mode for
# animation in moltui / chemcraft / etc.
# ----------------------------------------------------------------------

def normal_mode_trajectory(
    mol: Molecule,
    hessian_result,
    mode_index: int,
    *,
    amplitude: float = 0.5,
    n_frames: int = 20,
) -> List[Molecule]:
    """Generate a list of :class:`Molecule` frames sampling one
    vibrational normal mode for animated visualization.

    The displacement pattern for atom ``i`` in mode ``p`` is the
    standard Wilson-FG normalisation::

        Δr_i = (L_{ip} / sqrt(M_i)) . A . sin(2pi . t / T)

    where ``L`` is the mass-weighted normal-mode matrix, ``M_i`` the
    atomic mass (in electron-mass units), ``A`` is the amplitude
    (controlled by ``amplitude``), and ``t`` runs through ``n_frames``
    points over one full vibration period ``T``.

    Parameters
    ----------
    mol:
        Reference :class:`Molecule` (the equilibrium geometry).
    hessian_result:
        :class:`HessianResult` from any of the analytic-Hessian
        drivers or :func:`compute_hessian_fd`. Reads
        ``normal_modes``, ``frequencies_cm1``, and ``masses_amu``.
    mode_index:
        Which mode to animate (0 = lowest, ``3N - 1`` = highest).
        Trans/rot zero modes (typically indices 0..5) animate as
        rigid translations / rotations and are usually not
        physically interesting; pick a higher index to see a
        vibration.
    amplitude:
        Maximum atomic displacement in **bohr** along the mode
        eigenvector. Default 0.5 bohr ≈ 0.26 Å -- large enough to
        be visible in a molecular animation, small enough to stay
        in the harmonic regime.
    n_frames:
        Number of frames per period. Default 20. moltui plays back
        in a loop, so this controls both temporal resolution and
        playback speed.

    Returns
    -------
    list[Molecule]
        ``n_frames`` molecules, each with the equilibrium geometry
        + a sinusoidal displacement along mode ``p``. Pair with
        :func:`write_xyz_trajectory` to drop into moltui::

            frames = vq.normal_mode_trajectory(mol, hess, 8)
            vq.write_xyz_trajectory("mode-8.xyz", frames)

    Notes
    -----
    For zero modes (frequencies near 0), the eigenvector L_p still
    has well-defined norm but the "vibration" is an unphysical
    rigid translation/rotation. The function still writes a valid
    trajectory; visual inspection will show a translating /
    rotating cluster, which is sometimes useful as a sanity check
    on the Hessian projection.
    """
    L = np.asarray(hessian_result.normal_modes, dtype=np.float64)
    masses_amu = np.asarray(hessian_result.masses_amu, dtype=np.float64)
    n_dof, n_modes = L.shape
    n_atoms = n_dof // 3
    if 3 * n_atoms != n_dof:
        raise ValueError(
            f"normal_mode_trajectory: 3N={n_dof} not divisible by 3."
        )
    if not (0 <= mode_index < n_modes):
        raise ValueError(
            f"normal_mode_trajectory: mode_index={mode_index} out of "
            f"range [0, {n_modes})."
        )
    if amplitude <= 0:
        raise ValueError(
            f"normal_mode_trajectory: amplitude must be positive; got "
            f"{amplitude}."
        )
    if n_frames < 2:
        raise ValueError(
            f"normal_mode_trajectory: n_frames must be >= 2; got "
            f"{n_frames}."
        )
    if len(mol.atoms) != n_atoms:
        raise ValueError(
            f"normal_mode_trajectory: molecule has {len(mol.atoms)} "
            f"atoms but Hessian implies {n_atoms}."
        )

    # AMU -> electron-mass units (matches the convention used inside
    # HessianResult.hessian_mw / normal_modes).
    AMU_TO_E = 1822.8884862
    masses_e = masses_amu * AMU_TO_E
    inv_sqrt_m_per_atom = 1.0 / np.sqrt(masses_e)
    # Cartesian-displacement pattern: take the mass-weighted column,
    # then divide by sqrt(M_i) per atom triplet to recover Cartesian.
    L_p = L[:, mode_index].reshape(n_atoms, 3)
    cart_pattern = L_p * inv_sqrt_m_per_atom[:, None]   # (n_atoms, 3)

    # Reference geometry in bohr.
    ref = np.asarray(
        [[a.xyz[0], a.xyz[1], a.xyz[2]] for a in mol.atoms],
        dtype=np.float64,
    )
    ref_z = np.asarray([int(a.Z) for a in mol.atoms], dtype=np.int32)

    frames: List[Molecule] = []
    for f in range(n_frames):
        t = f / n_frames
        scale = amplitude * np.sin(2.0 * np.pi * t)
        new_xyz = ref + scale * cart_pattern
        atoms = [
            Atom(int(z), pos.tolist())
            for z, pos in zip(ref_z, new_xyz)
        ]
        frames.append(Molecule(atoms,
                                charge=mol.charge,
                                multiplicity=mol.multiplicity))
    return frames
