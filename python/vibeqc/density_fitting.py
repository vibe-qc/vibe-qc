"""Density fitting / resolution-of-the-identity (RI) machinery.

The math, in brief
------------------

The four-index electron-repulsion integral

  (muν|ls) = ∫∫ chi_mu(r1) chi_ν(r1) (1/r12) chi_l(r2) chi_s(r2) dr1 dr2

is replaced by a three-index / two-index factorisation in an auxiliary
basis {w_P}:

  (muν|ls)  ≈  S_{PQ} (muν|P) [V^{-1}]_{PQ} (Q|ls),
  V_{PQ}    =  (P|Q)  =  Coulomb metric on {w_P}.

Equivalently, fitting the orbital pair density chi_mu chi_ν in {w_P} by
minimising the Coulomb self-error ‖r - r̃‖_J yields

  r_muν(r) = chi_mu(r) chi_ν(r)  ≈  S_P d^{muν}_P w_P(r),
  d^{muν}_P = S_Q [V^{-1}]_{PQ} (Q|muν).

Cholesky-factorising the metric V = L L^T gives the half-transformed
B-tensor

  B^P_{muν} = S_Q [L^{-1}]_{PQ} (Q|muν),

so that

  (muν|ls)  ≈  S_P B^P_{muν} B^P_{ls}.

This single object -- `B` of shape (n_aux, n_orb, n_orb) -- is the
*universal currency* of density-fitting. Every method (RI-J, RI-K,
RI-MP2, future RIJ-COSX, future DF-CC) is one or two contractions of
B against densities, MO coefficients, or amplitudes:

  J_{muν}     = S_P B^P_{muν} g_P,           g_P = S_{ls} B^P_{ls} D_{ls}
  K_{muν}     = S_{P,i} (B C_occ)^P_{mui} (B C_occ)^P_{νi}
  (ia|jb)_RI = S_P (B^MO)^P_{ia} (B^MO)^P_{jb}

References
----------

Whitten,  J. Chem. Phys. 58, 4496 (1973).
Dunlap, Connolly, Sabin, Int. J. Quantum Chem. 16, 81 (1979).
Eichkorn, Treutler, Öhm, Häser, Ahlrichs, CPL 240, 283 (1995).
Vahtras, Almlöf, Feyereisen, CPL 213, 514 (1993)  -- RI-MP2.
Weigend, JCC 29, 167 (2008)                       -- def2 JKfit family.
Weigend, Häser, Patzelt, Ahlrichs, CPL 294, 143 (1998) -- RI-MP2 aux.
Neese, Wennmohs, Hansen, Becker, Chem. Phys. 356, 98 (2009)
                                                  -- RIJ-COSX.

Bundled auxiliary bases
-----------------------

In ``python/vibeqc/basis_library/basis/``:

  def2 JKfit (libint/ORCA per-zeta convention):
               def2-svp-jk, def2-tzvp-jk, def2-tzvpp-jk,
               def2-qzvp-jk, def2-qzvpp-jk, def2-sv(p)-jkfit
  def2 universal (Weigend, BSE/PySCF convention; one fit for the whole
  def2 family -- vibe-qc treats these as overrides):
               def2-universal-jkfit
               def2-universal-jfit  (Coulomb-only fit, smaller)
  def2 RIfit (per-zeta, MP2 / correlation):
               def2-svp-rifit, def2-svpd-rifit, def2-sv(p)-rifit,
               def2-tzvp-rifit, def2-tzvpd-rifit,
               def2-tzvpp-rifit, def2-tzvppd-rifit,
               def2-qzvp-rifit, def2-qzvpp-rifit, def2-qzvppd-rifit
               (def2-qzvp-rifit is BSE's 42-element main-group record;
               the def2-qzvp orbital basis auto-resolves to
               def2-qzvpp-rifit, whose blocks it duplicates -- #483)
  Dunning JK:  cc-pvdz-jkfit, cc-pvtz-jkfit, cc-pvqz-jkfit,
               cc-pv5z-jkfit
               + augmentation-cc-pvNz-jkfit (paired with the cc-pvNz-jkfit)
  Dunning RI:  cc-pvdz-ri, cc-pvtz-ri, cc-pvqz-ri, cc-pv5z-ri
               + augmentation-cc-pv5z-ri (paired with the cc-pv5z-ri)

The def2 universal-jkfit / universal-jfit / per-zeta rifit families
are fetched from the Basis Set Exchange (https://basissetexchange.org)
via ``scripts/fetch_bse_aux_bases.py`` so vibe-qc ships out of the box
with all standard def2 auxiliaries used by ORCA, PySCF, and Psi4.

``default_aux_basis_for(name, kind)`` autodetects an appropriate aux
for an orbital basis name. ``pob-*`` orbital bases (CRYSTAL-style
periodic) raise ``NotImplementedError`` -- designing a pob-suitable
JKfit aux is a separate (publication-worthy) research project; pass
``aux_basis=`` explicitly to override.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, NamedTuple, Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular

from ._vibeqc_core import compute_2c_eri, compute_3c_eri  # noqa: F401
from ._vibeqc_core import BasisSet


_logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Auxiliary-basis autodetection.
# -----------------------------------------------------------------------------

# Orbital-basis aliases that should map to the same canonical aux. Adding
# diffuse functions ("pd", "+d") doesn't change the aux fit; the aux family
# is sized to the parent (svp / tzvp / etc.).
def default_aux_basis_for(orbital_basis_name: str, kind: str = "jk") -> str:
    """Auto-select an auxiliary basis name for a given orbital basis.

    Thin wrapper over :func:`vibeqc.basis_registry.default_aux_basis`, the
    one registry every default-fit decision goes through (the periodic
    :func:`vibeqc.aux_basis.default_aux_for` reads the same table).

    Parameters
    ----------
    orbital_basis_name
        libint-recognised orbital basis name (case-insensitive). Examples:
        ``def2-svp``, ``cc-pvtz``, ``def2-tzvp(d)``.
    kind
        ``"jk"`` selects a JK-fit aux for J + K builds (HF, hybrid DFT
        energies / gradients). ``"ri"`` selects a smaller RI-fit aux for
        MP2 / CC correlation.

    Raises
    ------
    NotImplementedError
        For ``pob-*`` orbital bases (no upstream JKfit aux exists; this
        is a tracked research direction). For any orbital basis without
        a registered default mapping.
    ValueError
        If ``kind`` is not one of ``"jk"`` / ``"ri"``.
    """
    from .basis_registry import default_aux_basis

    return default_aux_basis(orbital_basis_name, kind, allow_standin=False)


# -----------------------------------------------------------------------------
# Auxiliary-basis element coverage (GitLab #480).
# -----------------------------------------------------------------------------
#
# Not every shipped fitting family covers every element of the orbital
# basis it is auto-selected for. ``cc-pV{D,T,Q,5}Z-JKFIT`` step straight
# from H to B -- no He, Li, Be, Na or Mg -- and stop before the 3d row;
# the ``cc-pVnZ-RI`` / ``-RIFIT`` sets stop before K; ``def2-qzvp-rifit``
# (BSE's 42-element record; no longer auto-selected, see ``_DEF2_RI``)
# skips Sc..Zn, Y..Cd, La and Hf..Hg; the ``cc-pwCVnZ-RIFIT``
# core-correlation family skips H, He, Li, Be, Na and Mg.
#
# libint2 does not raise for an element a basis file omits: it returns a
# BasisSet with **zero shells on that centre**. The fit is then built with
# no auxiliary functions on the uncovered atom, and the SCF converges
# cleanly -- tight gradient, no warning -- to a grossly over-bound energy
# (measured: -1.15 Ha on LiH/cc-pVQZ, -49.56 Ha on NaH/cc-pVDZ).
#
# A *completely* absent auxiliary basis already fails loudly: libint2
# returns an empty BasisSet and the C++ ``BasisSet`` ctor rejects it (see
# ``cpp/src/basis.cpp``), which is what closed #11. This guard closes the
# asymmetry for the partially-covering case.
#
# Refuse rather than substitute. Swapping in a covering auxiliary would
# change the fit under a user who named a specific one, and by the time a
# BasisSet exists an auto-resolved name is indistinguishable from an
# explicit one -- so a fallback would silently re-pair *explicitly*
# requested auxiliaries too. Refusing costs one keyword to recover from;
# a silent substitution is undetectable in the output.


class AuxCoverageGap(NamedTuple):
    """One atom the auxiliary basis contributes no functions to."""

    atom_index: int
    Z: int
    symbol: str


class AuxiliaryBasisCoverageError(ValueError):
    """An auxiliary basis has no functions on an atom that needs them.

    Raised before any fitting work happens. ``missing`` carries the
    per-atom gaps so callers can report them without re-parsing the
    message.
    """

    def __init__(
        self,
        message: str,
        *,
        aux_basis_name: str = "",
        missing: Optional[List[AuxCoverageGap]] = None,
    ) -> None:
        super().__init__(message)
        self.aux_basis_name = aux_basis_name
        self.missing: List[AuxCoverageGap] = list(missing or ())


def _atoms_carrying_shells(basis: BasisSet) -> set:
    """Indices of the atoms that own at least one shell in ``basis``."""
    return {int(shell.atom_index) for shell in basis.shells()}


def aux_basis_coverage_gaps(
    aux_basis: BasisSet,
    *,
    molecule=None,
    orbital_basis: Optional[BasisSet] = None,
) -> List[AuxCoverageGap]:
    """Atoms that need auxiliary functions but were given none.

    An atom needs auxiliary functions exactly when it carries orbital
    functions: it is the orbital pair density chi_mu chi_nu that the
    auxiliary set has to span. A centre with no orbital shells (a ghost
    or dummy) contributes no pair density and correctly gets no
    auxiliary functions, so it is never reported -- which is what keeps
    this check from firing on systems that are in fact fully covered.

    Parameters
    ----------
    aux_basis
        The constructed auxiliary basis to audit.
    molecule
        Optional; supplies ``Z`` and the element symbol for the message.
    orbital_basis
        Optional; when given, defines which atoms need coverage. Without
        it every atom of ``molecule`` with ``Z > 0`` is taken to need
        coverage. At least one of ``molecule`` / ``orbital_basis`` must
        be supplied.

    Returns
    -------
    list of AuxCoverageGap
        Empty when the auxiliary basis covers everything it must.
    """
    if molecule is None and orbital_basis is None:
        raise ValueError(
            "aux_basis_coverage_gaps: pass molecule= and/or orbital_basis= "
            "to determine which atoms need auxiliary coverage."
        )

    atoms = list(getattr(molecule, "atoms", ())) if molecule is not None else []

    if orbital_basis is not None:
        needed = _atoms_carrying_shells(orbital_basis)
    else:
        # Without an orbital basis the atom list is the only source of
        # "which centres need coverage". An object that does not expose
        # one would make this check a silent no-op -- which is the exact
        # failure class the guard exists to remove -- so refuse instead.
        if not atoms:
            raise ValueError(
                "aux_basis_coverage_gaps: cannot determine which atoms need "
                f"auxiliary coverage from {type(molecule).__name__} (no "
                "non-empty .atoms). Pass orbital_basis=, or a Molecule."
            )
        needed = {i for i, atom in enumerate(atoms) if int(atom.Z) > 0}

    covered = _atoms_carrying_shells(aux_basis)

    gaps: List[AuxCoverageGap] = []
    for index in sorted(needed - covered):
        if 0 <= index < len(atoms):
            atom = atoms[index]
            gaps.append(
                AuxCoverageGap(int(index), int(atom.Z), str(atom.symbol))
            )
        else:
            gaps.append(AuxCoverageGap(int(index), 0, f"atom {index}"))
    return gaps


def _coverage_error_message(
    gaps: List[AuxCoverageGap],
    *,
    aux_basis_name: str,
    orbital_basis_name: str,
    route: str,
) -> str:
    aux_label = aux_basis_name or "<unnamed>"
    where = f"{route}: " if route else ""

    listed = ", ".join(
        f"{gap.symbol} (atom {gap.atom_index})" for gap in gaps
    )
    plural = "atom" if len(gaps) == 1 else "atoms"
    orbital_clause = (
        f" auto-selected for orbital basis '{orbital_basis_name}'"
        if orbital_basis_name
        else ""
    )

    return (
        f"{where}auxiliary basis '{aux_label}'{orbital_clause} has no "
        f"functions on {len(gaps)} {plural} of this molecule: {listed}.\n"
        f"\n"
        f"Density fitting would build the fit with zero auxiliary "
        f"functions on that centre. The SCF then converges cleanly -- no "
        f"warning, tight gradient -- to a grossly wrong energy (measured "
        f"on the affected sets: -1.15 Ha on LiH/cc-pVQZ, -49.56 Ha on "
        f"NaH/cc-pVDZ). Refusing is the only safe behaviour, so the run "
        f"stops here rather than reporting a converged answer.\n"
        f"\n"
        f"Not every bundled fitting family covers every element:\n"
        f"  * cc-pV{{D,T,Q,5}}Z-JKFIT ship no He, Li, Be, Na or Mg, and\n"
        f"    stop before the 3d row (no K, Ca, Sc..Zn);\n"
        f"  * cc-pVnZ-RI / -RIFIT stop before K;\n"
        f"  * def2-qzvp-rifit (BSE's 42-element record) skips Sc..Zn,\n"
        f"    Y..Cd, La and Hf..Hg -- use def2-qzvpp-rifit, which is\n"
        f"    what a def2-qzvp orbital basis auto-resolves to;\n"
        f"  * cc-pwCVnZ-RIFIT / aug-cc-pwCVnZ-RIFIT skip H, He, Li, Be,\n"
        f"    Na and Mg.\n"
        f"\n"
        f"Options:\n"
        f"  1. Pass an auxiliary basis that covers every element present:\n"
        f"     aux_basis='def2-universal-jkfit' for the JK / SCF route,\n"
        f"     aux_basis='def2-tzvpp-rifit' (or another def2 -rifit) for\n"
        f"     the RI / correlation route. Both cover H-Rn. On the four\n"
        f"     systems above def2-universal-jkfit reproduces the\n"
        f"     conventional SCF energy to ~1e-5 Ha.\n"
        f"  2. Switch to a def2 orbital basis, whose auto-resolved\n"
        f"     auxiliaries cover the same range.\n"
        f"  3. Run without density fitting (density_fit=False) to use the\n"
        f"     direct four-index route, which needs no auxiliary basis.\n"
        f"\n"
        f"vibe-qc does not substitute a covering auxiliary automatically: "
        f"that would change the fit under a caller who named a specific "
        f"one. See GitLab #480."
    )


def check_aux_basis_coverage(
    aux_basis: BasisSet,
    *,
    aux_basis_name: str = "",
    molecule=None,
    orbital_basis: Optional[BasisSet] = None,
    orbital_basis_name: str = "",
    route: str = "",
) -> None:
    """Raise unless ``aux_basis`` covers every atom that needs fitting.

    Call this before any two- or three-centre integral work. See
    :func:`aux_basis_coverage_gaps` for the coverage rule.

    Raises
    ------
    AuxiliaryBasisCoverageError
        Naming the uncovered elements, the auxiliary basis, and the
        covering alternatives.
    """
    gaps = aux_basis_coverage_gaps(
        aux_basis, molecule=molecule, orbital_basis=orbital_basis
    )
    if not gaps:
        return
    raise AuxiliaryBasisCoverageError(
        _coverage_error_message(
            gaps,
            aux_basis_name=aux_basis_name,
            orbital_basis_name=orbital_basis_name,
            route=route,
        ),
        aux_basis_name=aux_basis_name,
        missing=gaps,
    )


def guard_aux_coverage_for_options(
    options,
    molecule,
    orbital_basis: Optional[BasisSet] = None,
    *,
    route: str = "",
) -> None:
    """Pre-flight the coverage check for a driver that builds its own aux.

    The native C++ SCF, MP2, CC and gradient drivers each construct the
    auxiliary ``BasisSet`` from ``options.aux_basis`` themselves, so the
    Python-side :class:`DensityFitting` guard never sees it. Call this
    just before dispatching to one of them.

    Deliberately permissive: it may only *add* the partial-coverage
    refusal, never change which error a caller already saw. It is a
    silent no-op unless density fitting (or COSX, which fits Coulomb the
    same way) is active with a resolved, constructible auxiliary name.
    An unresolved name is the driver's own explicit "aux_basis required"
    error; an unbuildable one is #11's already-loud path, and the driver
    may object for a more specific reason first.
    """
    wants_fit = bool(getattr(options, "density_fit", False)) or bool(
        getattr(options, "cosx", False)
    )
    if not wants_fit:
        return
    aux_name = getattr(options, "aux_basis", "") or ""
    if not aux_name:
        return

    try:
        aux = BasisSet(molecule, aux_name, require_all_atoms=False)
    except Exception:
        return

    check_aux_basis_coverage(
        aux,
        aux_basis_name=aux_name,
        molecule=molecule,
        orbital_basis=orbital_basis,
        orbital_basis_name=(
            getattr(orbital_basis, "name", "") or ""
            if orbital_basis is not None
            else ""
        ),
        route=route,
    )


def make_checked_aux_basis(
    molecule,
    aux_basis_name: str,
    *,
    orbital_basis: Optional[BasisSet] = None,
    orbital_basis_name: str = "",
    route: str = "",
) -> BasisSet:
    """Build an auxiliary ``BasisSet`` and refuse an uncovered element.

    The chokepoint every density-fitting route should use in place of a
    bare ``BasisSet(molecule, aux_name)``: it fails closed on the #480
    partial-coverage case before any fitting work happens.
    """
    aux = BasisSet(molecule, aux_basis_name, require_all_atoms=False)
    check_aux_basis_coverage(
        aux,
        aux_basis_name=aux_basis_name,
        molecule=molecule,
        orbital_basis=orbital_basis,
        orbital_basis_name=orbital_basis_name,
        route=route,
    )
    return aux


# -----------------------------------------------------------------------------
# DensityFitting object -- the universal RI artifact.
# -----------------------------------------------------------------------------

@dataclass
class DensityFittingInfo:
    """Diagnostics about a constructed DensityFitting object.

    Stored as a separate dataclass so logging / banner code can summarise
    a DF without poking at private fields.
    """
    n_orb: int
    n_aux: int
    aux_basis_name: str
    metric_min_eigenvalue: float    # smallest eigenvalue of V (= L_ii^2 if SPD)
    metric_condition_estimate: float  # max(eig(V)) / min(eig(V))


class DensityFitting:
    """Resolution-of-the-identity infrastructure for one (orbital, aux) pair.

    Owns the two libint2-driven integral tensors for the geometry:

      * ``V`` = (P|Q),  the (n_aux, n_aux) Coulomb metric;
      * ``T`` = (P|muν), the (n_aux, n_orb, n_orb) three-centre ERI.

    Computes the Cholesky factor V = L L^T once at construction, then
    caches the half-transformed B-tensor

      B^P_{muν} = S_Q [L^{-1}]_{PQ} (Q|muν)

    so that every downstream contraction (J, K, MO transforms) is a pure
    matrix-multiplication on a 3-tensor of the same shape as T. All
    methods that consume B are stateless w.r.t. the SCF iteration --
    construct once per geometry, contract many times.

    The raw ``T`` and Cholesky factor ``L`` are kept on the side
    (``self.three_center``, ``self.cholesky_factor``) because the
    analytic-gradient kernels need them in their unrotated form.
    """

    def __init__(
        self,
        orbital_basis: BasisSet,
        aux_basis: BasisSet,
        *,
        aux_basis_name: str = "",
        molecule=None,
    ) -> None:
        self.orbital_basis = orbital_basis
        self.aux_basis = aux_basis
        self.aux_basis_name = aux_basis_name
        self.n_orb = orbital_basis.nbasis
        self.n_aux = aux_basis.nbasis

        # #480: refuse an auxiliary basis that contributes zero functions
        # to a centre carrying orbital functions, before any integral
        # work. ``molecule`` is optional -- it only supplies element
        # symbols for the message; the coverage question itself is
        # decidable from the two bases' shell-to-atom maps alone.
        check_aux_basis_coverage(
            aux_basis,
            aux_basis_name=aux_basis_name,
            molecule=molecule,
            orbital_basis=orbital_basis,
            route="DensityFitting",
        )

        # 2-centre Coulomb metric V_PQ = (P|Q). SPD when the aux basis is
        # linearly independent, which holds for the bundled JKfit / RIfit
        # families on physical geometries.
        V = compute_2c_eri(aux_basis)

        # Cholesky V = L L^T. scipy's cho_factor returns (cho, lower) where
        # cho is L (with the lower triangle filled and upper undefined).
        # If V is *not* SPD, Cholesky raises -- callers see a clear error
        # before any J/K work happens.
        try:
            cho, lower = cho_factor(V, lower=True)
        except np.linalg.LinAlgError as exc:
            # Diagnose: report the smallest eigenvalue so the user can
            # decide whether to drop a basis or pick a different aux.
            eigvals = np.linalg.eigvalsh(V)
            raise np.linalg.LinAlgError(
                f"DensityFitting: 2-centre metric V = (P|Q) is not "
                f"positive-definite for aux basis "
                f"'{aux_basis_name or '<unnamed>'}' "
                f"(min eig = {eigvals[0]:.3e}, max eig = {eigvals[-1]:.3e}). "
                f"Either the aux basis itself is over-complete (rare for "
                f"published JKfit families) or the geometry collapsed two "
                f"centres on top of each other."
            ) from exc

        # Lower-triangular Cholesky factor L. We keep both ``cho`` (raw
        # scipy output, for cho_solve) and ``L`` (zeroed upper triangle,
        # for explicit triangular solves below).
        L = np.tril(cho)

        # Three-centre tensor T_{P, muν} = (P | muν). Shape (n_aux, n_orb, n_orb).
        T = compute_3c_eri(orbital_basis, aux_basis)

        # B^P_{muν} = (L^{-1} T)^P_{muν} via a single triangular solve over
        # the first axis. Reshape to (n_aux, n_orb^2) for the solve, then
        # back to (n_aux, n_orb, n_orb).
        T_flat = T.reshape(self.n_aux, -1)
        B_flat = solve_triangular(L, T_flat, lower=True, check_finite=False)
        B = B_flat.reshape(self.n_aux, self.n_orb, self.n_orb)

        # Diagnostics.
        eigvals = np.linalg.eigvalsh(V)
        info = DensityFittingInfo(
            n_orb=self.n_orb,
            n_aux=self.n_aux,
            aux_basis_name=aux_basis_name,
            metric_min_eigenvalue=float(eigvals[0]),
            metric_condition_estimate=float(eigvals[-1] / max(eigvals[0], 1e-30)),
        )

        self.metric = V
        self.cholesky_factor = L
        self._cho = cho
        self._lower = lower
        self.three_center = T
        self.B = B
        self.info = info

    # -------------------------------------------------------------------------
    # J / K builders. All consume self.B; cost is matrix-multiplication only.
    # -------------------------------------------------------------------------

    def build_J(self, D: np.ndarray) -> np.ndarray:
        """RI Coulomb matrix.

            g_P    = S_{muν} B^P_{muν} D_{muν}
            J_{muν} = S_P    B^P_{muν} g_P

        D must be the total density (closed-shell convention: D = 2 C C^T,
        trace D.S = n_electrons). For UHF/UKS pass D = D_a + D_b; the J
        operator is spin-symmetric and treats a + b identically.
        """
        if D.shape != (self.n_orb, self.n_orb):
            raise ValueError(
                f"build_J: D shape {D.shape} != "
                f"(n_orb, n_orb) = ({self.n_orb}, {self.n_orb})"
            )
        gamma = np.einsum("Pmn,mn->P", self.B, D, optimize=True)
        return np.einsum("Pmn,P->mn", self.B, gamma, optimize=True)

    def build_K(self, C_occ: np.ndarray) -> np.ndarray:
        """RI Exchange matrix from a list of occupied MOs.

            Bocc^P_{mui} = S_ν B^P_{muν} C_occ_{νi}
            K_{muν}      = S_{P,i} Bocc^P_{mui} Bocc^P_{νi}

        C_occ is shape (n_orb, n_occ); the columns are the occupied MOs
        in the AO basis. For closed-shell K_RHF you want C_occ to carry
        the doubly-occupied orbitals AND multiply the result by 2 (since
        the standard RHF K convention is K = S_i 2 (mui|νi)).
        Conventions vary by code; vibe-qc's RHF F = Hcore + 2J - K
        expects this builder to return S_i (mui|νi) -- i.e. *without* the
        factor of 2.
        """
        if C_occ.ndim != 2 or C_occ.shape[0] != self.n_orb:
            raise ValueError(
                f"build_K: C_occ must be (n_orb, n_occ); got {C_occ.shape}"
            )
        # B[P, mu, ν] . C_occ[ν, i] -> B_occ[P, mu, i]
        B_occ = np.einsum("Pmn,ni->Pmi", self.B, C_occ, optimize=True)
        # K[mu, ν] = sum_{P, i} B_occ[P, mu, i] B_occ[P, ν, i]
        return np.einsum("Pmi,Pni->mn", B_occ, B_occ, optimize=True)

    def build_K_density(self, D: np.ndarray) -> np.ndarray:
        """RI Exchange matrix directly from a density matrix.

            X^P_{νs}  = S_l B^P_{νl} D_{ls}
            K_{muν}    = S_{P, s} B^P_{mus} X^P_{νs}

        Useful when occupied MOs are unavailable (e.g. SAD-density guess
        or fractional-occupation cases). A factor of ~2 more work than
        ``build_K(C_occ)`` because there's no n_occ truncation.
        """
        if D.shape != (self.n_orb, self.n_orb):
            raise ValueError(
                f"build_K_density: D shape {D.shape} != "
                f"({self.n_orb}, {self.n_orb})"
            )
        X = np.einsum("Pnl,ls->Pns", self.B, D, optimize=True)
        return np.einsum("Pms,Pns->mn", self.B, X, optimize=True)

    # -------------------------------------------------------------------------
    # MO transforms -- the substrate for RI-MP2 / DF-CC.
    # -------------------------------------------------------------------------

    def mo_transform(
        self,
        C_left: np.ndarray,
        C_right: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Half- or full-MO-transformed B-tensor.

            B^P_{pq} = S_{muν} C_left_{mup} B^P_{muν} C_right_{νq}

        If ``C_right`` is None it defaults to ``C_left`` (symmetric
        transform, useful for canonical MP2 (ia|jb) factorisation:
        ``mo_transform(C_occ, C_virt)`` returns the (n_aux, n_occ, n_virt)
        tensor whose contraction S_P B^P_ia B^P_jb gives the (ia|jb)
        integral block).

        Returns a (n_aux, n_p, n_q) array.
        """
        if C_left.ndim != 2 or C_left.shape[0] != self.n_orb:
            raise ValueError(
                f"mo_transform: C_left must be (n_orb, n_p); got {C_left.shape}"
            )
        if C_right is None:
            C_right = C_left
        elif C_right.ndim != 2 or C_right.shape[0] != self.n_orb:
            raise ValueError(
                f"mo_transform: C_right must be (n_orb, n_q); got "
                f"{C_right.shape}"
            )
        # B[P, mu, ν] @ C_right[ν, q] -> tmp[P, mu, q]
        tmp = np.einsum("Pmn,nq->Pmq", self.B, C_right, optimize=True)
        # C_left[mu, p].T @ tmp[P, mu, q] -> out[P, p, q]
        return np.einsum("mp,Pmq->Ppq", C_left, tmp, optimize=True)

    # -------------------------------------------------------------------------
    # Convenience constructor that pairs autodetection with libint loading.
    # -------------------------------------------------------------------------

    @classmethod
    def from_orbital_basis(
        cls,
        molecule,
        orbital_basis: BasisSet,
        *,
        aux_basis: str = "",
        kind: str = "jk",
        orbital_basis_name: str = "",
    ) -> "DensityFitting":
        """Construct a DensityFitting given a molecule + orbital basis.

        ``aux_basis`` may be:
          * a non-empty string -- used directly as the libint basis name;
          * empty -- autodetect via ``default_aux_basis_for(orbital_basis_name, kind)``.

        ``orbital_basis_name`` is needed for autodetection because the
        C++ ``BasisSet`` object's ``.name`` attribute is not exposed in
        all cases. If it's empty AND ``aux_basis`` is empty, a
        ``ValueError`` is raised.
        """
        if not aux_basis:
            if not orbital_basis_name:
                raise ValueError(
                    "from_orbital_basis: pass either aux_basis= "
                    "explicitly or orbital_basis_name= for autodetection."
                )
            aux_basis = default_aux_basis_for(orbital_basis_name, kind=kind)

        aux = make_checked_aux_basis(
            molecule,
            aux_basis,
            orbital_basis=orbital_basis,
            orbital_basis_name=orbital_basis_name,
            route="DensityFitting.from_orbital_basis",
        )
        return cls(
            orbital_basis, aux, aux_basis_name=aux_basis, molecule=molecule
        )
