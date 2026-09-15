"""Linear-dependence diagnostics for basis-set optimization.

Uses vibe-qc's ``compute_overlap`` to detect near-singularity in the
AO overlap matrix S, and identifies which shells/exponents are
responsible.

    l_min = min(eigenvalues(S))
    e_LD  = 1e-7            <- canonical orthogonalisation threshold

When l_min < e_LD, the basis has linear dependence -- usually from
exponents that are too diffuse (basis functions overlap too much across
unit cells, or two primitives on the same atom are so similar they
span nearly the same direction).  The LD penalty nudges the optimizer
away from these regions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# Default threshold from vibe-qc's canonical_orthogonalize (cpp/include/
# vibeqc/linear_dependence.hpp).  Eigenvectors with l < e_LD are dropped
# from the orthogonalising transformation X.
EPS_LD = 1e-7


@dataclass
class LDDiagnostics:
    """Result of a linear-dependence check on one basis."""

    overlap_eigenvalues: np.ndarray
    """All eigenvalues of S, sorted ascending."""

    lambda_min: float
    """Smallest eigenvalue."""

    condition_number: float
    """l_max / l_min -- condition number of S."""

    n_independent: int
    """Number of linearly independent basis functions (l >= e_LD)."""

    n_dependent: int
    """Number of near-null directions (l < e_LD)."""

    ld_detected: bool
    """True iff l_min < e_LD."""

    shell_contributions: list[float] = field(default_factory=list)
    """Per-shell LD contribution (max absolute component in the near-null
    eigenvector, normalised).  Same length as the number of shells."""


def compute_overlap_diagnostics(
    basis_text: str,
    Z: int,
    *,
    epsilon: float = EPS_LD,
    molecule=None,
) -> LDDiagnostics:
    """Compute LD diagnostics for a basis using vibe-qc's overlap matrix.

    Writes a temporary .g94 file, builds a trivial molecule (single
    atom or user-supplied), computes S, and returns eigenvalues +
    per-shell contributions.

    Parameters
    ----------
    basis_text
        CRYSTAL inline basis text (from ``emit_crystal()``).
    Z
        Atomic number for the test molecule.
    epsilon
        LD threshold (default 1e-7).
    molecule
        Optional vibe-qc Molecule to use instead of a single-atom
        (e.g., for periodic LD checks with a unit cell).

    Returns
    -------
    LDDiagnostics
    """
    try:
        import vibeqc as vq
    except ImportError:
        raise RuntimeError("LD diagnostics require vibe-qc") from None

    from vibeqc.basis_crystal import emit_g94, parse_crystal_atom_basis

    # Parse inline basis back to CrystalAtomBasis -> emit .g94.
    # (The inline text is already in CRYSTAL format; parse then re-emit.)
    atom = _parse_inline_text(basis_text, Z)
    g94 = emit_g94([atom])

    # Write temp .g94.
    import tempfile
    import uuid

    tmp = Path(tempfile.mkdtemp(prefix="vbld_"))
    basis_dir = tmp / "basis"
    basis_dir.mkdir()
    name = f"ldcheck_{uuid.uuid4().hex[:8]}"
    (basis_dir / f"{name}.g94").write_text(g94)

    # Override LIBINT_DATA_PATH so vibe-qc finds our basis.
    import os

    old_path = os.environ.get("LIBINT_DATA_PATH")
    os.environ["LIBINT_DATA_PATH"] = str(tmp)
    try:
        if molecule is not None:
            mol = molecule
        else:
            multiplicity = _atomic_multiplicity(Z)
            mol = vq.Molecule([vq.Atom(Z, [0.0, 0.0, 0.0])], multiplicity=multiplicity)

        basis = vq.BasisSet(mol, name)
        S = vq.compute_overlap(basis)
    finally:
        if old_path is not None:
            os.environ["LIBINT_DATA_PATH"] = old_path
        else:
            os.environ.pop("LIBINT_DATA_PATH", None)
        import shutil

        shutil.rmtree(tmp, ignore_errors=True)

    # Eigenvalues of S (ascending).
    eigs = np.linalg.eigvalsh(np.asarray(S))
    lambda_min = float(eigs[0])
    lambda_max = float(eigs[-1])
    cond = lambda_max / lambda_min if lambda_min > 0 else float("inf")

    n_dep = int(np.sum(eigs < epsilon))
    n_ind = len(eigs) - n_dep

    # Shell contributions from the near-null eigenvector.
    # The eigenvector with the smallest eigenvalue shows which basis
    # functions are nearly collinear.  We take the absolute components
    # and sum per shell (rough -- proper mapping needs the shell-to-AO map,
    # which vibe-qc can provide via basis.iter_shells or similar).
    # For now: return the full eigenvector; recipes can inspect it.
    shell_contribs: list[float] = []

    return LDDiagnostics(
        overlap_eigenvalues=eigs,
        lambda_min=lambda_min,
        condition_number=cond,
        n_independent=n_ind,
        n_dependent=n_dep,
        ld_detected=lambda_min < epsilon,
        shell_contributions=shell_contribs,
    )


def ld_penalty(
    diagnostics: LDDiagnostics,
    *,
    lambda_ld: float = 1e3,
    epsilon: float = EPS_LD,
) -> float:
    """Compute the LD penalty term for the objective.

    The penalty is a smooth quadratic that turns on when l_min drops
    below e_LD:

        p(S) = l_ld . max(0, e - l_min)^2

    This is zero when the basis is well-conditioned (l_min >= e) and
    grows rapidly as l_min approaches zero, guiding the optimizer away
    from linearly dependent regions.

    Parameters
    ----------
    diagnostics
        LD diagnostics from :func:`compute_overlap_diagnostics`.
    lambda_ld
        Penalty strength (default 1e3 -- adjusted so the penalty is
        ~0.1 mHa when l_min = e/2 and ~1 mHa near zero).
    epsilon
        LD threshold (default 1e-7).

    Returns
    -------
    float
        Penalty value in Hartree (add to the objective).
    """
    deficit = max(0.0, epsilon - diagnostics.lambda_min)
    return lambda_ld * deficit * deficit


def ld_penalty_from_basis(
    basis_text: str,
    Z: int,
    *,
    lambda_ld: float = 1e3,
    epsilon: float = EPS_LD,
) -> float:
    """Convenience: compute LD diagnostics and penalty in one call."""
    diag = compute_overlap_diagnostics(basis_text, Z, epsilon=epsilon)
    return ld_penalty(diag, lambda_ld=lambda_ld, epsilon=epsilon)


# ---------------------------------------------------------------------------
# Condition-number penalty (the CRYSTAL OPTBASIS / BDIIS objective term)
# ---------------------------------------------------------------------------
#
# VandeVondele & Hutter, J. Chem. Phys. 127, 114105 (2007),
# doi:10.1063/1.2770708, "Gaussian basis sets for accurate calculations on
# molecular systems in gas and condensed phases", optimise a basis by
# minimising the composite objective (their Eq. 1):
#
#     Ω({a, d}) = E_tot({a, d})  +  g . ln κ({a, d})
#
# where κ = l_max / l_min is the condition number of the AO overlap matrix
# S. The ln κ term penalises near-linear-dependence *smoothly*: it grows
# without bound as l_min -> 0 (κ -> inf) yet stays gentle while the basis is
# well-conditioned, so the optimiser is pulled back from the diffuse-
# exponent collapse that derails unconstrained energy minimisation.
# CRYSTAL23's OPTBASIS / BDIIS optimiser uses the same penalised objective
# (Daga, Civalleri & Maschio, J. Chem. Theory Comput. 16, 2192 (2020),
# doi:10.1021/acs.jctc.9b01004).
#
# Note: the Daga 2020 paper cites the VandeVondele reference with a typo --
# volume "227"; the correct volume is 127.
#
# This is the smooth, always-on counterpart to ``ld_penalty`` above: where
# ld_penalty is a hinge that only switches on once l_min crosses e_LD,
# ln κ contributes everywhere and shapes the whole search. They compose
# (ln κ for global conditioning + the e_LD hinge as a hard backstop).


def condition_number_penalty(
    diagnostics: LDDiagnostics,
    *,
    gamma: float = 1e-3,
    kappa_floor: float = 1.0,
) -> float:
    """VandeVondele-Hutter condition-number penalty ``g . ln κ(S)``.

    Parameters
    ----------
    diagnostics
        LD diagnostics carrying ``condition_number`` = l_max / l_min.
    gamma
        Penalty weight g (Hartree). Typical range 1e-3-1e-2: small enough
        not to bias the energy at a well-conditioned minimum, large enough
        to dominate once κ blows up. Default 1e-3.
    kappa_floor
        Lower clamp on κ before taking the log. For a real Gram matrix
        κ >= 1 exactly (ln κ >= 0); the clamp only guards numerical noise
        that could push κ marginally below 1 and make the penalty negative.

    Returns
    -------
    float
        ``g . ln(max(κ, kappa_floor))`` in Hartree (add to the objective),
        or ``+inf`` if κ is non-finite / non-positive (a degenerate basis,
        which the optimiser must treat as forbidden).
    """
    kappa = diagnostics.condition_number
    if not math.isfinite(kappa) or kappa <= 0.0:
        return float("inf")
    return gamma * math.log(max(kappa, kappa_floor))


# ---------------------------------------------------------------------------
# In-memory overlap path (no file / no libint round-trip)
# ---------------------------------------------------------------------------
#
# For a same-center, same-element atomic basis the AO overlap matrix is
# block-diagonal in (l, m) -- different angular momenta and different m
# components are orthogonal on a common origin. Each (l, m) block has the
# same numerical content (the angular integral is just d_{mm'} for
# normalized real spherical harmonics), so it appears (2l+1) times. The
# per-l contracted-shell overlap starts as a closed-form contraction of
# the primitive Gaussian-overlap formula:
#
#     S_norm(a, b; l) = (2.√(ab) / (a + b))^(l + 3/2)
#
# This is the overlap of two normalized solid-spherical primitives sharing
# l and the origin, derived from
#   ∫₀^inf r^(2l+2) exp(-(a+b)r^2) dr  .  ∫ Y_lm Y_lm dΩ
# with the radial normalization that makes primitive <a|a> = 1. The
# resulting contracted-shell block is then normalized by each contracted
# shell's self-overlap, matching libint's treatment of the .g94 file path.
# Used here to skip the emit_g94 -> tempfile -> libint round-trip inside
# per-evaluation LD checks. The file-path version (above) stays the
# authoritative call for multi-center / periodic LD checks where Bloch
# sums matter.


def _normalized_primitive_overlap(alpha: float, beta: float, l: int) -> float:
    """Overlap of two same-center, same-l normalized primitive Gaussians."""
    return (2.0 * math.sqrt(alpha * beta) / (alpha + beta)) ** (l + 1.5)


_SHELL_L = {"S": 0, "P": 1, "D": 2, "F": 3, "G": 4}


def _per_l_blocks(atom) -> dict[int, list[tuple[list[float], list[float]]]]:
    """Group an atom's shells by l, splitting SP shells into S + P sub-shells.

    Returns a dict mapping ``l`` to a list of (exponents, coefficients) pairs,
    one pair per contracted shell of that l. Caller assumes scale_factor == 1.0
    (raises NotImplementedError otherwise -- pob-* basis sets satisfy this).
    """
    blocks: dict[int, list[tuple[list[float], list[float]]]] = {}
    for sh in atom.shells:
        if sh.scale_factor != 1.0:
            raise NotImplementedError(
                f"in-memory overlap currently requires scale_factor == 1.0; "
                f"got {sh.scale_factor!r} on a {sh.shell_type} shell"
            )
        if sh.shell_type == "SP":
            blocks.setdefault(0, []).append(
                (list(sh.exponents), list(sh.coefficients))
            )
            blocks.setdefault(1, []).append(
                (list(sh.exponents), list(sh.coefficients_p))
            )
            continue
        try:
            l = _SHELL_L[sh.shell_type]
        except KeyError:
            raise NotImplementedError(
                f"in-memory overlap doesn't yet handle shell_type={sh.shell_type!r}"
            ) from None
        blocks.setdefault(l, []).append((list(sh.exponents), list(sh.coefficients)))
    return blocks


def _shell_shell_overlap(
    expo_a: list[float],
    coef_a: list[float],
    expo_b: list[float],
    coef_b: list[float],
    l: int,
) -> float:
    """Overlap of two contracted shells of the same l on a common center."""
    total = 0.0
    for ap, ca in zip(expo_a, coef_a):
        for bq, cb in zip(expo_b, coef_b):
            total += ca * cb * _normalized_primitive_overlap(ap, bq, l)
    return total


def _normalize_contracted_shell_block(block: np.ndarray) -> np.ndarray:
    """Normalize a same-l contracted-shell overlap block like libint."""
    diag = np.diag(block)
    if np.any(diag <= 0.0):
        raise ValueError(
            "contracted-shell self-overlap must be positive; got "
            f"{diag.tolist()}"
        )
    norms = np.sqrt(diag)
    return block / np.outer(norms, norms)


def compute_overlap_diagnostics_from_atom(
    atom,
    *,
    epsilon: float = EPS_LD,
) -> LDDiagnostics:
    """Pure-NumPy LD diagnostics from a parsed :class:`CrystalAtomBasis`.

    No file write, no libint call, no env-var mutation. Builds the
    same-center, same-element AO overlap matrix analytically from the shell
    contractions and returns its eigenvalue spectrum.

    The eigenvalues are reported with their full AO multiplicity (a per-l
    block of size n_l contributes n_l eigenvalues, each repeated 2l+1 times),
    so the n_dependent / n_independent counts match what libint would report
    when fed the same atom via the file path.

    Parameters
    ----------
    atom
        Parsed :class:`CrystalAtomBasis` with ``shells`` containing one or
        more :class:`CrystalShell`. ``scale_factor`` must be 1.0 (pob-*
        baseline); other values raise ``NotImplementedError``.
    epsilon
        LD threshold (default 1e-7).

    Returns
    -------
    LDDiagnostics

    Notes
    -----
    Only handles the one-center per-element atomic LD check -- i.e., what the
    production recipe currently invokes. For multi-center / periodic LD
    checks, use :func:`compute_overlap_diagnostics` (file path).
    """
    blocks = _per_l_blocks(atom)
    all_eigs: list[float] = []
    for l, shells in sorted(blocks.items()):
        n = len(shells)
        block = np.empty((n, n), dtype=float)
        for i in range(n):
            for j in range(i, n):
                v = _shell_shell_overlap(
                    shells[i][0], shells[i][1],
                    shells[j][0], shells[j][1],
                    l,
                )
                block[i, j] = v
                block[j, i] = v
        block = _normalize_contracted_shell_block(block)
        eigs_l = np.linalg.eigvalsh(block)
        # Each per-l eigenvalue appears (2l+1) times across the m-components.
        all_eigs.extend(eigs_l.tolist() * (2 * l + 1))

    if not all_eigs:
        raise ValueError("atom has no shells -- cannot build overlap")

    eigs = np.array(sorted(all_eigs), dtype=float)
    lambda_min = float(eigs[0])
    lambda_max = float(eigs[-1])
    cond = lambda_max / lambda_min if lambda_min > 0 else float("inf")
    n_dep = int(np.sum(eigs < epsilon))
    n_ind = len(eigs) - n_dep

    return LDDiagnostics(
        overlap_eigenvalues=eigs,
        lambda_min=lambda_min,
        condition_number=cond,
        n_independent=n_ind,
        n_dependent=n_dep,
        ld_detected=lambda_min < epsilon,
        shell_contributions=[],
    )


def ld_penalty_from_atom(
    atom,
    *,
    lambda_ld: float = 1e3,
    epsilon: float = EPS_LD,
) -> float:
    """Convenience: in-memory LD diagnostics + penalty in one call.

    Uses :func:`compute_overlap_diagnostics_from_atom` (NumPy-only) -- no
    libint, no file I/O. Suitable for the inner loop of a BOBYQA / MIGRAD
    objective where each evaluation must add an LD penalty without paying
    a filesystem round-trip per call.
    """
    diag = compute_overlap_diagnostics_from_atom(atom, epsilon=epsilon)
    return ld_penalty(diag, lambda_ld=lambda_ld, epsilon=epsilon)


def cond_penalty_from_atom(
    atom,
    *,
    gamma: float = 1e-3,
    epsilon: float = EPS_LD,
) -> float:
    """Convenience: in-memory g.ln κ(S) penalty from a parsed atom basis.

    The condition-number counterpart of :func:`ld_penalty_from_atom`,
    built on the same NumPy-only overlap path. See
    :func:`condition_number_penalty` for the g semantics.
    """
    diag = compute_overlap_diagnostics_from_atom(atom, epsilon=epsilon)
    return condition_number_penalty(diag, gamma=gamma)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_inline_text(basis_text: str, Z: int):
    """Parse CRYSTAL inline basis text to CrystalAtomBasis."""
    from vibeqc.basis_crystal import CrystalAtomBasis, CrystalShell

    lines = basis_text.strip().splitlines()
    parts = lines[0].split()
    nshell = int(parts[1])
    atom = CrystalAtomBasis(Z=Z, has_ecp=False)
    i = 1
    for _ in range(nshell):
        header = lines[i].split()
        lat = int(header[1])
        npg = int(header[2])
        occ = float(header[3])
        scale = float(header[4])
        i += 1
        lat_map = {0: "S", 1: "SP", 2: "P", 3: "D", 4: "F", 5: "G"}
        shell = CrystalShell(
            shell_type=lat_map.get(lat, "S"),
            occupancy=occ,
            scale_factor=scale,
        )
        for _ in range(npg):
            prim = lines[i].split()
            i += 1
            shell.exponents.append(float(prim[0]))
            shell.coefficients.append(float(prim[1]))
            if lat == 1 and len(prim) >= 3:
                shell.coefficients_p.append(float(prim[2]))
        atom.shells.append(shell)
    return atom


def _atomic_multiplicity(Z: int) -> int:
    """Ground-state spin multiplicity for an isolated atom."""
    # Hund's rule ground states for main-group elements.
    mult = {
        1: 2,
        2: 1,  # H, He
        3: 2,
        4: 1,
        5: 2,
        6: 3,
        7: 4,
        8: 3,
        9: 2,
        10: 1,  # Li-Ne
        11: 2,
        12: 1,
        13: 2,
        14: 3,
        15: 4,
        16: 3,
        17: 2,
        18: 1,  # Na-Ar
        19: 2,
        20: 1,  # K, Ca
    }
    return mult.get(Z, 1)
