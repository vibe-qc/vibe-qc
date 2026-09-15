"""Molden-format writer for vibe-qc SCF results.

Molden files carry the geometry, the basis set, and the molecular orbitals
in a single plain-text document. Any orbital viewer that understands the
format (Molden itself, Jmol, Avogadro, MolView, IQmol, ...) can visualize
the occupied and virtual MOs without further processing.

Supports RHF, UHF, RKS, and UKS results. Restricted results produce one
[MO] block; unrestricted results produce two (alpha followed by beta).

Periodic (Bloch) orbitals
-------------------------

Periodic drivers return complex MO coefficients even at Gamma, where H(k=0)
and S(k=0) are real symmetric and every orbital is real up to one arbitrary
global phase per column. Molden's ``[MO]`` block carries only real
coefficients, so this writer divides that phase out and emits the real part
(:func:`_real_mo_coefficients`). An orbital whose imaginary part survives
phase removal -- a genuine Bloch orbital at k != 0 -- is refused rather than
written, because a viewer cannot tell a corrupted file from a valid one.

Basis-function ordering
-----------------------

Molden and libint disagree on the ordering of real spherical harmonics
within a shell of angular momentum L >= 1. We reorder MO coefficients
at write time so the emitted file matches Molden's convention:

- p (L=1): molden (px, py, pz) <- libint (py, pz, px)
- d (L=2): molden (d0, d+1, d-1, d+2, d-2) <- libint (d-2, d-1, d0, d+1, d+2)
- f, g, h: molden (m=0, +1, -1, +2, -2, ..., +L, -L) <- libint (m = -L..+L)

vibe-qc forces ``set_pure(true)`` on the BasisSet at construction, so every
shell is spherical-harmonic in this writer -- Cartesian d/f are not emitted.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..._primitive_norm import libint_primitive_norm as _primitive_normalisation
from ..._vibeqc_core import BasisSet, Molecule
from ...correlation_conventions import effective_electron_count
from .._text_safety import scrub_output_text


# Largest imaginary residual, relative to the orbital's own largest
# coefficient, still accepted as "real up to a global phase".
#
# Two regimes have to be told apart, and they are separated by four orders
# of magnitude:
#
#   * A Gamma orbital is real up to one global phase, because H(k=0) and
#     S(k=0) are real symmetric. A *non-degenerate* one comes back at
#     complex-eigensolver roundoff, ~1e-16. Inside a *degenerate* block the
#     eigenvector is ill-conditioned -- the solver may return any unitary
#     mixture of the partners -- and roundoff is amplified by the
#     degeneracy: Ne/STO-3G in a 7-bohr box puts its exactly degenerate 2p
#     pair at 7.5e-06, and that residual does not shrink as the SCF
#     converges, because it is conditioning and not convergence error.
#     Taking the real part is still exact here: Re(c) of a vector in a real
#     invariant subspace stays in that subspace, so it remains an
#     eigenvector at the same eigenvalue, off-normalised only at
#     O(residual^2).
#
#   * A genuine Bloch orbital at k != 0 is irreducibly complex and lands at
#     O(1) -- typically 0.1 to 1.
#
# 1e-4 sits two orders above the worst degeneracy amplification measured
# and three below the smallest genuinely complex orbital, so it separates
# the two cleanly without being delicate at either end.
_MOLDEN_IMAG_RTOL = 1.0e-4

# Orbital-energy separation below which two levels are treated as one
# degenerate block. Measured separation on Ne/STO-3G k222: the exactly
# degenerate 2p pair sits at ~1e-15 Ha, the nearest distinct level at
# 2.2e-3 Ha, so anything in between partitions the spectrum identically.
_MOLDEN_DEGENERACY_TOL_HA = 1.0e-6


# Element-symbol table; indexed by atomic number (0 is unused).
_ELEMENTS: Tuple[str, ...] = (
    "X",
    "H",  "He",
    "Li", "Be", "B",  "C",  "N",  "O",  "F",  "Ne",
    "Na", "Mg", "Al", "Si", "P",  "S",  "Cl", "Ar",
    "K",  "Ca", "Sc", "Ti", "V",  "Cr", "Mn", "Fe", "Co", "Ni",
    "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y",  "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
    "Ag", "Cd", "In", "Sn", "Sb", "Te", "I",  "Xe",
)


def _element(z: int) -> str:
    if 0 < z < len(_ELEMENTS):
        return _ELEMENTS[z]
    return f"Z{z}"


# Molden-spec shell-type letters (lowercase).
_SHELL_LETTER = ("s", "p", "d", "f", "g", "h", "i", "k", "l")


def _pure_molden_to_libint(L: int) -> List[int]:
    """Index permutation: molden_order[j] = libint slot of molden basis j.

    For L=0 a single s function, identity.

    For L=1 libint's ``pure`` p-shell indexes m = -1, 0, +1 which in real
    solid harmonics is (py, pz, px). Molden wants (px, py, pz).

    For L >= 2 libint uses m = -L, -L+1, ..., 0, ..., +L, while molden uses
    (m=0, +1, -1, +2, -2, ..., +L, -L).
    """
    if L == 0:
        return [0]
    if L == 1:
        return [2, 0, 1]  # px(m=+1=libint 2), py(m=-1=libint 0), pz(m=0=libint 1)
    idx = [L]  # m = 0
    for k in range(1, L + 1):
        idx.append(L + k)  # m = +k
        idx.append(L - k)  # m = -k
    return idx


def _build_ao_permutation(basis: BasisSet) -> np.ndarray:
    """Global AO permutation: ``C_molden = C_libint[ao_perm, :]``.

    Each shell contributes 2L+1 consecutive entries; we walk shells in
    libint's native order and stitch per-shell spherical-harmonic
    reorderings together.
    """
    perm: List[int] = []
    offset = 0
    for shell in basis.shells():
        if not shell.pure:
            raise ValueError(
                "write_molden: Cartesian shells are not supported yet "
                f"(shell L={shell.l} pure={shell.pure}). vibeqc forces "
                "set_pure(true) at construction -- this error means the "
                "BasisSet was created outside the normal path."
            )
        local = _pure_molden_to_libint(shell.l)
        perm.extend(offset + i for i in local)
        offset += len(local)
    return np.asarray(perm, dtype=np.intp)


def _write_header(f, title: str) -> None:
    f.write("[Molden Format]\n")
    f.write("[Title]\n")
    # Molden tolerates a blank title line; keep it non-empty to be safe.
    # The title is user-controlled -- run_job / run_periodic_job pass the
    # output basename here -- so neutralise any bidi / zero-width / control
    # character to a visible \uXXXX token before it lands in the file (the
    # Trojan-Source display-deception class, same hardening the cube header
    # uses; see vibeqc.output._text_safety). The file is UTF-8, so a
    # legitimate accented basename passes through unchanged.
    f.write(f" {scrub_output_text(title) or 'vibe-qc output'}\n")


def _write_atoms(f, mol: Molecule) -> None:
    f.write("[Atoms] (AU)\n")
    for idx, atom in enumerate(mol.atoms, start=1):
        sym = _element(atom.Z)
        x, y, z = atom.xyz[0], atom.xyz[1], atom.xyz[2]
        f.write(
            f" {sym:<4s} {idx:5d} {atom.Z:5d} "
            f"{x:20.12f} {y:20.12f} {z:20.12f}\n"
        )


def _write_gto(f, basis: BasisSet, n_atoms: int) -> None:
    """[GTO] block: per-atom shell listings, in atom index order."""
    # Group shells by atom (libint emits shells atom-by-atom but we don't
    # rely on that -- sort explicitly).
    shells_by_atom: List[List] = [[] for _ in range(n_atoms)]
    for shell in basis.shells():
        shells_by_atom[shell.atom_index].append(shell)

    f.write("[GTO]\n")
    for atom_idx in range(n_atoms):
        # Trailing zero is the historical "scaling factor" field -- always 0.
        f.write(f"  {atom_idx + 1} 0\n")
        for shell in shells_by_atom[atom_idx]:
            letter = _SHELL_LETTER[shell.l]
            n_prim = len(shell.exponents)
            f.write(f" {letter}   {n_prim}  1.00\n")
            for alpha, c_eff in zip(shell.exponents, shell.coefficients):
                # libint stores primitive-normalized coefficients; molden
                # readers expect the raw basis-set-file values and apply
                # primitive normalisation themselves.
                c_raw = c_eff / _primitive_normalisation(alpha, shell.l)
                f.write(f"    {alpha:18.10E}    {c_raw:18.10E}\n")
        f.write("\n")


def _write_pure_flags(f, basis: BasisSet) -> None:
    """Molden spec: ``[5D]`` / ``[7F]`` / ``[9G]`` announce spherical."""
    lmax = 0
    for shell in basis.shells():
        if shell.l > lmax:
            lmax = shell.l
    # Emit the flags relevant to the basis. [5D] implies [7F] [9G] ...
    # in some readers, but writing them explicitly is the safe bet.
    if lmax >= 2:
        f.write("[5D]\n")
    if lmax >= 3:
        f.write("[7F]\n")
    if lmax >= 4:
        f.write("[9G]\n")


def _degenerate_groups(
    energies: np.ndarray, tol: float = _MOLDEN_DEGENERACY_TOL_HA
) -> List[List[int]]:
    """Partition ascending orbital energies into degenerate blocks.

    Each column is compared against the energy that opened its block, so a
    slowly drifting run of near-degenerate levels does not chain into one
    giant group.
    """
    groups: List[List[int]] = []
    start = 0
    for j in range(1, len(energies) + 1):
        if j == len(energies) or abs(energies[j] - energies[start]) > tol:
            groups.append(list(range(start, j)))
            start = j
    return groups


def _realify_degenerate_block(
    block: np.ndarray, overlap: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Return a real orthonormal basis of a degenerate block's span.

    Within a degenerate eigenspace the solver is free to return any unitary
    mixture of the partners, and dividing out one global phase per column
    cannot undo a rotation *between* columns -- which is why the residual
    imaginary part scales with degeneracy rather than with convergence.

    The span itself is real, because H(k=0) and S(k=0) are real symmetric.
    So for a block ``C = U W`` with ``U`` real and ``W`` unitary, both
    ``Re C = U Re W`` and ``Im C = U Im W`` lie in ``span(U)``, and the real
    ``n x 2m`` matrix ``[Re C | Im C]`` has exactly ``span(U)`` as its column
    space. Diagonalising its Gram matrix in the overlap metric and keeping
    the ``m`` non-null directions returns a real basis with
    ``B^T S B = I`` -- an S-orthonormal set of eigenvectors at the same
    eigenvalue, which is what a real-arithmetic code would have produced.

    Any basis of the eigenspace is as valid as any other, so which rotation
    comes back is not meaningful. Returns ``None`` when the block does not
    have the expected rank, leaving the caller to refuse rather than emit a
    basis it could not verify.
    """
    n_ao, m = block.shape
    real_span = np.concatenate([block.real, block.imag], axis=1)  # n x 2m
    if overlap is not None:
        gram = real_span.T @ (overlap @ real_span)
    else:
        # No overlap available (a molecular result, which is real anyway).
        # The identity metric still yields a correct real basis of the span;
        # only the normalisation convention differs.
        gram = real_span.T @ real_span
    gram = 0.5 * (gram + gram.T)

    eigenvalues, vectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    keep = order[:m]
    discard = order[m:]
    kept_values = eigenvalues[keep]
    eigenvalue_scale = float(np.max(eigenvalues))
    rank_tolerance = 1.0e-12 * eigenvalue_scale
    if eigenvalue_scale <= 0.0 or float(np.min(kept_values)) <= rank_tolerance:
        return None  # rank deficient: not the real eigenspace we assumed
    if discard.size and float(np.max(eigenvalues[discard])) > rank_tolerance:
        # A genuinely complex m-dimensional Bloch subspace can span up to
        # 2m independent real directions.  Selecting an arbitrary m of them
        # would manufacture plausible-looking real orbitals at k != Gamma.
        return None
    basis = real_span @ (vectors[:, keep] / np.sqrt(kept_values))
    if basis.shape != (n_ao, m) or not np.isfinite(basis).all():
        return None
    return basis


def _real_mo_coefficients(
    coeffs: np.ndarray,
    spin: str,
    energies: Optional[np.ndarray] = None,
    overlap: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return real ``(nbasis, nmo)`` MO coefficients for the [MO] block.

    Real input passes through. Complex input -- what every periodic Bloch
    driver returns, including at Gamma -- has its arbitrary per-orbital
    global phase divided out, and the real part is kept.

    At k = 0 the Bloch Fock and overlap matrices are real symmetric, so each
    eigenvector is real up to one global phase. A *degenerate* block needs
    more than a per-column phase, because the solver may return any unitary
    mixture of its partners; those blocks are re-expressed on a real basis
    of their own span by :func:`_realify_degenerate_block`, which is exact.
    At k != 0 the orbital is genuinely complex and no basis choice inside a
    degenerate block makes it real; Molden cannot represent it, so we raise
    rather than write a file whose coefficients a viewer would read as
    something they are not.
    """
    arr = np.asarray(coeffs)
    if arr.ndim != 2:
        raise ValueError(
            f"write_molden: {spin} MO coefficients must be a 2-D "
            f"(nbasis, nmo) array; got shape {arr.shape}. A multi-k periodic "
            "result carries one such block per k-point -- select the Gamma "
            "block before writing."
        )
    if not np.iscomplexobj(arr):
        return np.ascontiguousarray(arr, dtype=float)

    arr = _realify_degenerate_blocks(arr, spin, energies, overlap)
    if not np.iscomplexobj(arr):
        return arr

    out = np.empty(arr.shape, dtype=float)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        magnitudes = np.abs(col)
        scale = float(np.max(magnitudes))
        if scale == 0.0:
            out[:, j] = 0.0
            continue
        if float(np.max(np.abs(col.imag))) > _MOLDEN_IMAG_RTOL * scale:
            # Rotate the largest coefficient onto the positive real axis;
            # for a real-up-to-phase orbital that makes the whole column
            # real. Skipped when the column is already real so the driver's
            # own sign convention survives the export unchanged -- rotating
            # a real column with a negative pivot would flip every sign.
            pivot = int(np.argmax(magnitudes))
            col = col * (np.conj(col[pivot]) / float(magnitudes[pivot]))
        imag_residual = float(np.max(np.abs(col.imag)))
        if imag_residual > _MOLDEN_IMAG_RTOL * scale:
            raise ValueError(
                f"write_molden: {spin} orbital {j + 1} is not real after "
                "removing its global phase "
                f"(max|Im|/max|C| = {imag_residual / scale:.2e}). Molden's "
                "[MO] block carries only real coefficients, so a genuine "
                "Bloch orbital at k != 0 cannot be written. Export orbitals "
                "at Gamma, or use the QVF archive, which carries the complex "
                "Bloch wavefunction."
            )
        out[:, j] = col.real
    return out


def _realify_degenerate_blocks(
    arr: np.ndarray,
    spin: str,
    energies: Optional[np.ndarray],
    overlap: Optional[np.ndarray],
) -> np.ndarray:
    """Rewrite each multi-column degenerate block on a real basis.

    Only blocks that a per-column phase cannot fix are touched, so a run
    whose orbitals are already real up to phase keeps byte-identical
    coefficients and the driver's own sign convention.
    """
    if energies is None:
        return arr
    values = np.asarray(energies)
    if np.iscomplexobj(values):
        values = values.real
    values = np.asarray(values, dtype=float).reshape(-1)
    if values.shape[0] != arr.shape[1]:
        return arr

    overlap_matrix = None
    if overlap is not None:
        candidate = np.asarray(overlap)
        if np.iscomplexobj(candidate):
            # S(k=0) is real symmetric; a complex container holds it exactly.
            candidate = candidate.real
        if candidate.ndim == 2 and candidate.shape == (arr.shape[0],) * 2:
            overlap_matrix = np.asarray(candidate, dtype=float)

    out = arr
    for group in _degenerate_groups(values):
        if len(group) < 2:
            continue
        block = arr[:, group]
        scale = float(np.max(np.abs(block)))
        if scale == 0.0:
            continue
        if float(np.max(np.abs(block.imag))) <= _MOLDEN_IMAG_RTOL * scale:
            # A small imaginary component is not by itself enough to make
            # ``block.real`` a valid Molden basis.  In an exactly degenerate
            # eigenspace a complex eigensolver may return a tiny unitary
            # rotation between partner columns.  Dropping that rotation
            # changes C.T S C by O(|Im C|**2), which is large enough to lose
            # orthonormality on a different BLAS even while every imaginary
            # coefficient is comfortably below the format refusal threshold.
            # Preserve an already-real block only when its real projection is
            # itself orthonormal in the Gamma overlap metric.
            projected = block.real
            if overlap_matrix is not None:
                projected_gram = projected.T @ (
                    overlap_matrix @ projected
                )
            else:
                projected_gram = projected.T @ projected
            projected_error = float(
                np.max(np.abs(projected_gram - np.eye(len(group))))
            )
            roundoff = (
                64.0
                * np.finfo(float).eps
                * max(1, arr.shape[0], len(group))
            )
            if projected_error <= roundoff:
                continue
        realified = _realify_degenerate_block(block, overlap_matrix)
        if realified is None:
            continue  # fall through to the per-column check, which refuses
        if out is arr:
            out = arr.copy()
        # Stays a complex container holding real values; the per-column pass
        # below then sees a zero imaginary part and keeps the real part.
        out[:, group] = realified
    return out


def _real_mo_energies(energies: np.ndarray) -> np.ndarray:
    """Return real orbital eigenvalues, tolerating a complex-typed array."""
    arr = np.asarray(energies)
    if np.iscomplexobj(arr):
        # Eigenvalues of a Hermitian F(k) are real; some drivers keep them in
        # the same complex container as the coefficients.
        arr = arr.real
    return np.ascontiguousarray(arr, dtype=float).reshape(-1)


def _write_mo_block(
    f,
    mo_energies: np.ndarray,
    mo_coeffs: np.ndarray,   # shape (nbasis, nmo), columns are MOs
    occupations: Sequence[float],
    spin: str,               # "Alpha" or "Beta"
    ao_perm: np.ndarray,
) -> None:
    """Write one [MO] block -- a list of orbitals with energy / spin / occ."""
    n_ao, n_mo = mo_coeffs.shape
    reordered = mo_coeffs[ao_perm, :]  # AO permutation to molden ordering

    for j in range(n_mo):
        f.write(f" Sym=  A\n")
        f.write(f" Ene= {mo_energies[j]:18.10f}\n")
        f.write(f" Spin= {spin}\n")
        f.write(f" Occup= {occupations[j]:12.8f}\n")
        col = reordered[:, j]
        for i in range(n_ao):
            f.write(f" {i + 1:5d}   {col[i]:20.12E}\n")


def _rhf_occupations(n_mo: int, n_electrons: int) -> List[float]:
    n_occ = n_electrons // 2
    return [2.0 if i < n_occ else 0.0 for i in range(n_mo)]


def _uhf_occupations(
    n_mo: int, multiplicity: int, n_electrons: int
) -> Tuple[List[float], List[float]]:
    n_alpha = (n_electrons + multiplicity - 1) // 2
    n_beta = n_electrons - n_alpha
    alpha = [1.0 if i < n_alpha else 0.0 for i in range(n_mo)]
    beta = [1.0 if i < n_beta else 0.0 for i in range(n_mo)]
    return alpha, beta


def _result_occupations(result, attr: str, n_mo: int) -> Optional[List[float]]:
    """Return per-orbital occupations carried by the result, if it has any.

    Molecular SCF results do not expose occupations, so the aufbau fallbacks
    above stay in force for them. Periodic results do, and theirs are the
    only correct answer under finite-T smearing, where the frontier orbitals
    of the exported k-block are fractionally occupied and an aufbau guess
    would misreport them.
    """
    values = getattr(result, attr, None)
    if values is None:
        return None
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.shape[0] != n_mo:
        return None
    if np.iscomplexobj(arr):
        arr = arr.real
    return [float(value) for value in arr]


def write_molden(
    path: os.PathLike | str,
    molecule: Molecule,
    basis: BasisSet,
    result,
    *,
    title: str = "",
) -> None:
    """Write geometry, basis, and molecular orbitals to a Molden-format file.

    Parameters
    ----------
    path
        Output file path (``.molden`` is the conventional suffix).
    molecule
        The :class:`Molecule` used for the SCF run.
    basis
        The :class:`BasisSet` used for the SCF run.
    result
        RHFResult / UHFResult / RKSResult / UKSResult (or any object with
        matching ``mo_coeffs`` / ``mo_energies`` attributes, restricted or
        split into ``_alpha`` / ``_beta``).
    title
        Optional one-line title written to the ``[Title]`` block.

    Notes
    -----
    The basis must use pure spherical harmonics (vibe-qc's default). MO
    coefficients are reordered from libint's native per-shell index to
    molden's ``(m=0, +1, -1, +2, -2, ...)`` convention on the fly.
    """
    path = os.fspath(path)
    ao_perm = _build_ao_permutation(basis)
    n_atoms = len(molecule.atoms)
    n_electrons = effective_electron_count(molecule, result)
    multiplicity = molecule.multiplicity

    # Dispatch: restricted has `mo_coeffs`; unrestricted has `_alpha` / `_beta`.
    has_alpha = hasattr(result, "mo_coeffs_alpha")
    has_restricted = hasattr(result, "mo_coeffs") and not has_alpha

    if not (has_alpha or has_restricted):
        raise TypeError(
            "write_molden: result must expose either `mo_coeffs` "
            "(restricted) or `mo_coeffs_alpha` + `mo_coeffs_beta` "
            "(unrestricted); got " + type(result).__name__
        )

    # UTF-8 (not ASCII): a legitimate non-ASCII output basename -- which
    # reaches the [Title] field -- must not raise UnicodeEncodeError
    # mid-write and abort the artifact. The only free-text field, the
    # title, is scrubbed of bidi / control bytes in _write_header, so
    # widening the encoding cannot leak a Trojan-Source byte into the file.
    with open(path, "w", encoding="utf-8") as f:
        _write_header(f, title)
        _write_atoms(f, molecule)
        _write_gto(f, basis, n_atoms)
        _write_pure_flags(f, basis)

        f.write("[MO]\n")
        if has_restricted:
            energies = _real_mo_energies(result.mo_energies)
            coeffs = _real_mo_coefficients(
                result.mo_coeffs,
                "alpha",
                energies,
                getattr(result, "overlap", None),
            )
            n_mo = coeffs.shape[1]
            occs = _result_occupations(
                result, "occupations", n_mo
            ) or _rhf_occupations(n_mo, n_electrons)
            _write_mo_block(f, energies, coeffs, occs, "Alpha", ao_perm)
        else:
            ea = _real_mo_energies(result.mo_energies_alpha)
            eb = _real_mo_energies(result.mo_energies_beta)
            overlap = getattr(result, "overlap", None)
            ca = _real_mo_coefficients(
                result.mo_coeffs_alpha, "alpha", ea, overlap
            )
            cb = _real_mo_coefficients(
                result.mo_coeffs_beta, "beta", eb, overlap
            )
            default_a, default_b = _uhf_occupations(
                ca.shape[1], multiplicity, n_electrons
            )
            occ_a = (
                _result_occupations(result, "occupations_alpha", ca.shape[1])
                or default_a
            )
            occ_b = (
                _result_occupations(result, "occupations_beta", cb.shape[1])
                or default_b
            )
            _write_mo_block(f, ea, ca, occ_a, "Alpha", ao_perm)
            _write_mo_block(f, eb, cb, occ_b, "Beta", ao_perm)
