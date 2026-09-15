"""ORCA ``.hess`` ASCII writer for vibe-qc :class:`HessianResult` objects.

Phase M1. The ``.hess`` format is ORCA's documented ASCII Hessian
container, consumed natively by:

  - **moltui** (``moltui out.hess`` for animated normal-mode display);
  - **chemcraft**, **avogadro**, **gabedit** (vibrational-mode
    visualization);
  - **VMD** (via the molefacture / nmwiz plugin).

This is the cleanest interchange path between vibe-qc and the
visualization ecosystem because ``.hess`` is fully ASCII (no
proprietary binary structure like ``.gbw``) and version-stable.

Format reference: extracted from a real ORCA 6.1.1 NumFreq run on
H₂O / HF / STO-3G. The 14 sections ORCA writes, in order:

  1. ``$orca_hessian_file`` -- header
  2. ``$act_atom`` -- atom index for partial Hessians (``0`` for full)
  3. ``$act_coord`` -- coordinate index (``0`` for full)
  4. ``$act_energy`` -- total electronic energy in Ha
  5. ``$multiplicity``
  6. ``$hessian`` -- (3N, 3N) Cartesian Hessian in Ha/bohr^2, printed in
     5-columns-per-block layout with row labels
  7. ``$vibrational_frequencies`` -- 3N freqs in cm⁻¹ (zero modes 0.0,
     imaginary modes negative)
  8. ``$normal_modes`` -- (3N, 3N) mass-weighted normal-mode matrix
     (each *column* is a mode), 5-columns-per-block layout
  9. ``$atoms`` -- N atoms with ``label mass(amu) x y z`` in bohr
 10. ``$actual_temperature`` -- finite-T data temperature; 0.0 by default
 11. ``$frequency_scale_factor`` -- ORCA's harmonic-to-empirical scale
 12. ``$dipole_derivatives`` -- (3N, 3) dmu/dR tensor in atomic units;
     **only written when present on the HessianResult**
 13. ``$ir_spectrum`` -- N rows of ``wavenumber eps Int TX TY TZ``;
     **only written when dipole derivatives are present**
 14. ``$end``

When ``HessianResult.dipole_derivatives is None`` the writer emits a
zeroed ``$dipole_derivatives`` block + a zeroed ``$ir_spectrum`` so
the file remains readable by tools that scan for those sections; the
mode-amplitude information (which moltui actually uses) is unaffected.
"""

from __future__ import annotations

import os
from typing import Optional, Union

import numpy as np

from .._vibeqc_core import Atom, Molecule
from ..hessian import HessianResult, ir_intensities


__all__ = ["write_orca_hess"]


# ----------------------------------------------------------------------
# Element symbols -- covers Z = 1..118
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
# Block-printing helpers -- match ORCA's exact layout so character-by-
# character diff against an ORCA-produced file passes.
# ----------------------------------------------------------------------

def _write_block_matrix(fh, M: np.ndarray, *, cols_per_block: int = 5) -> None:
    """Write a 2-D matrix in ORCA's ``$hessian`` / ``$normal_modes``
    layout -- split columns into chunks of ``cols_per_block``, each
    chunk preceded by a header row of column indices, followed by
    rows labeled by row index.

    Layout (matches ORCA 6.x exactly so the file is a drop-in for any
    consumer that does positional parsing rather than tokenising on
    whitespace):

      - Row label: ``%5d``
      - First column number: ``%22.10E`` (6-space-leading for positive,
        5-space + sign for negative) -- extra wide so the row label
        and the first datum read clearly.
      - Subsequent column numbers: ``%19.10E`` (3-space-leading for
        positive, 2-space + sign for negative).

    Header row mirrors the column-number layout: 5-space row-label gap,
    then per-column ``%22d`` for the first column and ``%19d`` for the
    rest, with 8 trailing spaces (ORCA's habit).
    """
    n_rows, n_cols = M.shape
    for c0 in range(0, n_cols, cols_per_block):
        c1 = min(c0 + cols_per_block, n_cols)
        # Header line follows ORCA's Fortran format:
        #   (18X, I3, 4(16X, I3), 8X)
        # i.e. 18 leading spaces, col index in 3 chars, then 16-space
        # gap + 3-char col index per subsequent column, with 8 trailing
        # spaces. Total = 18 + 3 + 4*(16+3) + 8 = 105 chars for a
        # 5-wide block.
        header = " " * 18
        for j, c in enumerate(range(c0, c1)):
            if j > 0:
                header += " " * 16
            header += f"{c:>3d}"
        header += " " * 8
        fh.write(header + "\n")
        # Data lines:
        #   (I5, 6X, ES16.10, 4(3X, ES16.10))
        # i.e. 5-wide row label, 6 spaces, first 16-char number, then
        # 3-space gap + 16-char number per subsequent column. The
        # %22.10E and %19.10E formats produce identical output for
        # positive numbers; for negatives the sign consumes 1 of the
        # leading spaces (5 + 16 instead of 6 + 16).
        for r in range(n_rows):
            line = f"{r:>5d}"
            for j, c in enumerate(range(c0, c1)):
                width = 22 if j == 0 else 19
                line += f"{M[r, c]:>{width}.10E}"
            fh.write(line + "\n")


def _write_dipole_derivatives(fh, D: np.ndarray) -> None:
    """``$dipole_derivatives`` block: 3N rows x 3 cols, no labels."""
    n = D.shape[0]
    fh.write("$dipole_derivatives\n")
    fh.write(f"{n}\n")
    for r in range(n):
        fh.write(
            f"   {D[r, 0]:.10E}   {D[r, 1]:.10E}   {D[r, 2]:.10E}\n"
        )


# ----------------------------------------------------------------------
# Public writer
# ----------------------------------------------------------------------

def write_orca_hess(
    path: Union[str, os.PathLike],
    molecule: Molecule,
    result: HessianResult,
    *,
    energy_ha: float = 0.0,
    multiplicity: int = 1,
    actual_temperature_K: float = 0.0,
    frequency_scale_factor: float = 1.0,
) -> None:
    """Write a vibe-qc :class:`HessianResult` to an ORCA-format
    ``.hess`` file.

    Parameters
    ----------
    path:
        Output file path (overwritten if it exists).
    molecule:
        The :class:`Molecule` the Hessian was computed for. Atom
        species, masses, and Cartesian positions go into the
        ``$atoms`` block.
    result:
        :class:`HessianResult` carrying the Cartesian Hessian, mass-
        weighted normal modes, and frequencies.
    energy_ha:
        Total electronic energy at the reference geometry, in Hartree.
        Goes into ``$act_energy``. Pass the SCF total energy here.
    multiplicity:
        Spin multiplicity 2S+1. Default 1 (singlet).
    actual_temperature_K:
        Finite-temperature data temperature in K. ORCA writes 0.0 by
        default; we follow.
    frequency_scale_factor:
        Empirical harmonic-to-experimental frequency scaling factor
        (e.g. 0.95 for HF, 0.97 for DFT). 1.0 = no scaling. Default 1.0.

    Notes
    -----
    moltui consumes ``$atoms``, ``$vibrational_frequencies``, and
    ``$normal_modes`` to animate vibrational modes. The Hessian +
    dipole-derivative + IR-spectrum blocks are written unconditionally
    (zeroed when not available on the HessianResult) so the file is a
    drop-in for any consumer that scans for them.

    Examples
    --------
    >>> from vibeqc import (
    ...     Molecule, BasisSet, run_rhf, RHFOptions,
    ...     compute_hessian_rhf_analytic,
    ... )
    >>> from vibeqc.io import write_orca_hess
    >>> mol = Molecule.from_xyz("h2o.xyz")
    >>> basis = BasisSet(mol, "sto-3g")
    >>> rhf = run_rhf(mol, basis, RHFOptions())
    >>> hess = compute_hessian_rhf_analytic(mol, basis, rhf,
    ...                                       basis_name="sto-3g")
    >>> write_orca_hess("h2o.hess", mol, hess, energy_ha=rhf.energy)
    """
    H = np.asarray(result.hessian, dtype=np.float64)
    L = np.asarray(result.normal_modes, dtype=np.float64)
    freqs = np.asarray(result.frequencies_cm1, dtype=np.float64)

    n_dof = H.shape[0]
    if H.shape != (n_dof, n_dof):
        raise ValueError(
            f"write_orca_hess: hessian has shape {H.shape}; expected square."
        )
    if L.shape != (n_dof, n_dof) or freqs.shape != (n_dof,):
        raise ValueError(
            f"write_orca_hess: normal_modes shape {L.shape} / frequencies "
            f"shape {freqs.shape} inconsistent with Hessian shape {H.shape}."
        )

    n_atoms = n_dof // 3
    if 3 * n_atoms != n_dof:
        raise ValueError(
            f"write_orca_hess: 3N = {n_dof} is not divisible by 3."
        )
    if len(molecule.atoms) != n_atoms:
        raise ValueError(
            f"write_orca_hess: molecule has {len(molecule.atoms)} atoms but "
            f"Hessian implies {n_atoms}."
        )

    masses_amu = (
        np.asarray(result.masses_amu, dtype=np.float64)
        if result.masses_amu is not None
        else _default_masses_amu(molecule)
    )
    if masses_amu.shape != (n_atoms,):
        raise ValueError(
            f"write_orca_hess: masses_amu shape {masses_amu.shape}; "
            f"expected ({n_atoms},)."
        )

    # Dipole derivatives (3N, 3) -- zero pad if missing.
    if result.dipole_derivatives is not None:
        D = np.asarray(result.dipole_derivatives, dtype=np.float64)
        ir_int = ir_intensities(result)
        have_dipole = True
    else:
        D = np.zeros((n_dof, 3), dtype=np.float64)
        ir_int = np.zeros(n_dof, dtype=np.float64)
        have_dipole = False

    with open(path, "w", encoding="utf-8") as fh:
        # Section 1 -- header
        fh.write("\n")
        fh.write("$orca_hessian_file\n")
        fh.write("\n")

        # Sections 2 + 3 -- partial-Hessian markers (zero = full Hessian)
        fh.write("$act_atom\n  0\n\n")
        fh.write("$act_coord\n  0\n\n")

        # Section 4 -- total energy
        fh.write("$act_energy\n")
        fh.write(f"      {energy_ha:.6f}\n\n")

        # Section 5 -- multiplicity
        fh.write("$multiplicity\n")
        fh.write(f"  {int(multiplicity)}\n\n")

        # Section 6 -- Cartesian Hessian
        fh.write("$hessian\n")
        fh.write(f"{n_dof}\n")
        _write_block_matrix(fh, H)
        fh.write("\n")

        # Section 7 -- frequencies
        fh.write("$vibrational_frequencies\n")
        fh.write(f"{n_dof}\n")
        for i, f in enumerate(freqs):
            fh.write(f"    {i:<5d}{f:>22.16f}\n")
        fh.write("\n")

        # Section 8 -- normal modes (mass-weighted, columns)
        fh.write("$normal_modes\n")
        fh.write(f"{n_dof} {n_dof}\n")
        _write_block_matrix(fh, L)
        fh.write("\n")

        # Section 9 -- atoms (label, mass[amu], x, y, z[bohr])
        fh.write("#\n# The atoms: label  mass x y z (in bohrs)\n#\n")
        fh.write("$atoms\n")
        fh.write(f"{n_atoms}\n")
        for i, a in enumerate(molecule.atoms):
            sym = _element_symbol(a.Z)
            x, y, z = a.xyz
            fh.write(
                f" {sym:<4s} {masses_amu[i]:>10.5f}     "
                f"{x: .12f}     {y: .12f}     {z: .12f}\n"
            )
        fh.write("\n")

        # Section 10 -- finite-T temperature
        fh.write("$actual_temperature\n")
        fh.write(f"  {actual_temperature_K:.6f}\n\n")

        # Section 11 -- frequency scale factor
        fh.write("$frequency_scale_factor\n")
        fh.write(f"  {frequency_scale_factor:.6f}\n\n")

        # Section 12 -- dipole derivatives (3N rows x 3 cols)
        _write_dipole_derivatives(fh, D)
        fh.write("\n")

        # Section 13 -- IR spectrum (N rows: wavenumber, eps, Int, TX, TY, TZ)
        fh.write("#\n# The IR spectrum\n#  wavenumber eps Int TX TY TZ\n#\n")
        fh.write("$ir_spectrum\n")
        fh.write(f"{n_dof}\n")
        for i in range(n_dof):
            wavenum = freqs[i] if freqs[i] >= 0 else 0.0
            intensity = ir_int[i] if have_dipole else 0.0
            # ``eps`` (extinction coefficient) is intensity / (some
            # frequency-dependent prefactor); we leave it 0 unless we
            # have IR intensities -- moltui doesn't read it anyway.
            eps = 0.0
            tx, ty, tz = (D[i, 0], D[i, 1], D[i, 2]) if have_dipole \
                else (0.0, 0.0, 0.0)
            fh.write(
                f"   {wavenum:>7.2f}   {eps:.8f}   {intensity:.8f}     "
                f"{tx: .6f}     {ty: .6f}     {tz: .6f}\n"
            )
        fh.write("\n")

        # Section 14 -- sentinel
        fh.write("\n$end\n")
        fh.write("\n")


# ----------------------------------------------------------------------
# Internal -- fallback masses when HessianResult.masses_amu is missing
# (older code paths).
# ----------------------------------------------------------------------

def _default_masses_amu(molecule: Molecule) -> np.ndarray:
    """Per-atom isotope-averaged masses keyed by atomic number. We
    duplicate the small table from ``hessian._resolve_masses`` rather
    than importing it (private helper, but identical numerical
    contents) -- keeps this module self-contained for the writer.
    """
    # Standard atomic weights (IUPAC 2021), most-abundant-isotope-
    # averaged. Same reference table used elsewhere in vibe-qc.
    table = {
        1:   1.00800,    2:   4.00260,
        3:   6.94000,    4:   9.01218,
        5:  10.81000,    6:  12.01100,
        7:  14.00700,    8:  15.99900,
        9:  18.99800,   10:  20.18000,
        11: 22.98977,   12: 24.30500,
        13: 26.98154,   14: 28.08500,
        15: 30.97376,   16: 32.06000,
        17: 35.45000,   18: 39.94800,
        19: 39.09830,   20: 40.07800,
    }
    masses = np.zeros(len(molecule.atoms), dtype=np.float64)
    for i, atom in enumerate(molecule.atoms):
        masses[i] = table.get(int(atom.Z), float(atom.Z) * 2.0)
    return masses
