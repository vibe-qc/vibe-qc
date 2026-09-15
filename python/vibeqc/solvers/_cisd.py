"""Fixed-space configuration interaction with singles and doubles (CISD).

Truncated, single-reference CI: the Hamiltonian is diagonalised in the space
spanned by the closed-shell (or ROHF-aufbau) reference plus all single and
double substitutions out of it.  Unlike :func:`~vibeqc.solvers.casci` (full CI
in an active window) the determinant space here is capped at double
excitations, so CISD recovers most -- but not all -- of the correlation a full CI
would, at a cost that grows only as ``(n_occ.n_vir)^2`` rather than
combinatorially.

The space is built by :func:`~vibeqc.solvers._determinant.generate_cisd_determinants`
and the matrix by the validated unrestricted Slater-Condon engine
(:func:`~vibeqc.solvers._slater_condon.build_hamiltonian_matrix_unrestricted`),
exactly as CASCI/MRCI do -- so this solver is reference-agnostic: any source of
MO-basis integrals (HF, DFT, or a semiempirical engine such as MSINDO) feeds it.

Two limiting checks the test-suite pins:

* For a **two-electron** system CISD is *exact* -- no triple/quadruple
  substitutions exist -- so ``cisd`` reproduces ``casci`` (full CI) to machine
  precision.
* The CISD energy is variational and bracketed, ``E_FCI <= E_CISD <= E_ref``;
  ``max_excitation=1`` collapses the space to CIS (reference + singles).

Reference: J. A. Pople, R. Seeger, R. Krishnan, "Variational configuration
interaction methods and comparison with perturbation theory", Int. J. Quantum
Chem. 12 (S11), 149 (1977).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.linalg import eigh

from ._casci import _frozen_core_dressing
from ._determinant import Det, SpinDet, generate_cisd_determinants
from ._slater_condon import (
    build_hamiltonian_matrix_unrestricted,
    diagonal_matrix_element_unrestricted,
)


def _describe_configuration(
    det: SpinDet,
    ref: SpinDet,
    n_core: int,
) -> str:
    """Human-readable label of ``det`` relative to the reference ``ref``.

    Orbital indices are reported in the **full-MO** numbering (active index +
    ``n_core``).  Examples: ``"reference"``, ``"3a->4a"``, ``"3a->5a, 3b->5b"``.
    """
    a_occ, b_occ = det
    a_ref, b_ref = ref
    holes_a = sorted(set(a_ref) - set(a_occ))
    parts_a = sorted(set(a_occ) - set(a_ref))
    holes_b = sorted(set(b_ref) - set(b_occ))
    parts_b = sorted(set(b_occ) - set(b_ref))
    if not holes_a and not holes_b:
        return "reference"
    terms = []
    for i, a in zip(holes_a, parts_a):
        terms.append(f"{i + n_core}a->{a + n_core}a")
    for i, a in zip(holes_b, parts_b):
        terms.append(f"{i + n_core}b->{a + n_core}b")
    return ", ".join(terms)


@dataclass(kw_only=True)
class CISDOptions:
    """CISD options for ``run_job(method="cisd")`` / ``citype="cisd"``.

    The standalone :func:`cisd` solver keeps the same controls as
    explicit keyword arguments.  This options struct is the high-level
    driver surface, mirroring the CASCI/CASSCF option pattern.

    Attributes
    ----------
    max_excitation : int
        2 for CISD (default), 1 for CIS / TDA-style reference+singles.
    nroots : int
        Number of CI roots to solve for (default 1).  With ``nroots > 1``
        the headline ``SolverResult.energy`` stays root 0 and the .out
        solver block prints the root-energy table.
    max_det : int | None
        Guard on the determinant-space size (default 20 000).  Raise this
        deliberately for larger dense eigensolves.
    """

    max_excitation: int = 2
    nroots: int = 1
    max_det: Optional[int] = 20_000


@dataclass
class CISDResult:
    """Result of a fixed-space CISD (or CIS, ``max_excitation=1``).

    Attributes
    ----------
    e_total : float
        Ground-state total energy (CI eigenvalue + nuclear repulsion + frozen-
        core energy).
    e_corr : float
        Correlation energy ``E_total - E_ref`` (E_ref = energy of the reference
        determinant in the same integrals; this is the SCF energy when the
        orbitals are the SCF MOs).
    e_ref : float
        Reference-determinant total energy.
    ci_coeffs : np.ndarray
        Ground-state CI vector over :attr:`determinants` (shape ``(n_det,)``).
    determinants : list[SpinDet]
        The ``(alpha_occ, beta_occ)`` determinants spanning the CISD space, in
        the **active**-space orbital numbering.
    n_det : int
        Determinant-space size.
    reference_weight : float
        Squared CI coefficient of the reference determinant (closeness to a
        single-reference picture).
    max_excitation : int
        1 for CIS, 2 for CISD.
    n_core : int
        Frozen (doubly-occupied, uncorrelated) core orbitals.
    converged : bool
        Always ``True`` -- the space is diagonalised exactly.
    eigenvalues : np.ndarray
        All CI total energies (ascending), useful for excited roots / spectra.
    e_totals : list[float]
        Total energies of the lowest ``nroots`` roots.
    ci_coeffs_all : np.ndarray or None
        ``(n_det, nroots)`` eigenvectors when ``nroots > 1`` (else ``None``).
    """

    e_total: float
    e_corr: float
    e_ref: float
    ci_coeffs: np.ndarray
    determinants: list
    n_det: int = 0
    reference_weight: float = 0.0
    max_excitation: int = 2
    n_core: int = 0
    converged: bool = True
    eigenvalues: Optional[np.ndarray] = None
    e_totals: list = field(default_factory=list)
    ci_coeffs_all: Optional[np.ndarray] = None

    def dominant_configurations(
        self,
        n: int = 5,
        *,
        root: int = 0,
        min_weight: float = 0.0,
    ) -> list[tuple[str, float, float]]:
        """Leading ``(description, coefficient, weight)`` of a CI root.

        Sorted by descending weight (squared coefficient).  ``description`` is
        relative to the reference (see :func:`_describe_configuration`).
        """
        vec = self.ci_coeffs if root == 0 or self.ci_coeffs_all is None \
            else self.ci_coeffs_all[:, root]
        a_ref = tuple(range(self._nalpha))
        b_ref = tuple(range(self._nbeta))
        ref = (a_ref, b_ref)
        order = np.argsort(vec ** 2)[::-1]
        out: list[tuple[str, float, float]] = []
        for idx in order:
            w = float(vec[idx] ** 2)
            if w < min_weight:
                break
            out.append((
                _describe_configuration(self.determinants[idx], ref, self.n_core),
                float(vec[idx]), w))
            if len(out) >= n:
                break
        return out

    # Reference occupation (set by ``cisd`` for the labelling helper).
    _nalpha: int = 0
    _nbeta: int = 0


def cisd(
    h1e_mo: np.ndarray,
    h2e_mo: np.ndarray,
    nelec: int,
    norb: int,
    n_core: int = 0,
    nuclear_repulsion: float = 0.0,
    ms2: Optional[int] = None,
    *,
    max_excitation: int = 2,
    nroots: int = 1,
    max_det: Optional[int] = 20_000,
) -> CISDResult:
    """Run fixed-space CISD on an MO-basis Hamiltonian.

    Parameters
    ----------
    h1e_mo : (norb, norb) ndarray
        One-electron Hamiltonian in the MO basis.
    h2e_mo : (norb, norb, norb, norb) ndarray
        Two-electron integrals, **physicist's** notation, MO basis.
    nelec : int
        Total number of electrons.
    norb : int
        Total number of spatial orbitals.
    n_core : int
        Doubly-occupied orbitals frozen out of the correlation treatment.  The
        remaining ``nelec - 2.n_core`` electrons are correlated in the upper
        ``norb - n_core`` orbitals; the core enters as a constant energy plus a
        mean-field dressing of the one-electron term (shared with CASCI).
    nuclear_repulsion : float
        Nuclear repulsion energy (added to every total energy).
    ms2 : int, optional
        ``2 S_z`` of the correlated electrons; defaults to
        ``(nelec - 2.n_core) % 2`` (closed-shell singlet / minimal-spin doublet).
    max_excitation : int
        2 for CISD (default), 1 for CIS (reference + singles only).
    nroots : int
        Number of CI roots to report energies for (default 1; the ground state
        is root 0).
    max_det : int or None
        Guard on the determinant-space size.

    Returns
    -------
    CISDResult
    """
    norb_total = int(h1e_mo.shape[0])
    if norb != norb_total:
        raise ValueError(f"norb ({norb}) != h1e dimension ({norb_total})")
    if n_core < 0 or 2 * n_core > nelec:
        raise ValueError(f"invalid n_core={n_core} for nelec={nelec}")

    n_active_orb = norb_total - n_core
    n_active_elec = nelec - 2 * n_core
    if ms2 is None:
        ms2 = n_active_elec % 2
    n_alpha = (n_active_elec + ms2) // 2
    n_beta = n_active_elec - n_alpha
    if not (0 <= n_alpha <= n_active_orb and 0 <= n_beta <= n_active_orb):
        raise ValueError(
            f"cannot place {n_active_elec} electrons (ms2={ms2}) in "
            f"{n_active_orb} active orbitals")

    # Frozen-core dressing (shared with CASCI): constant E_core + 1e dressing.
    active = slice(n_core, n_core + n_active_orb)
    e_core_dress, h1e_active = _frozen_core_dressing(
        h1e_mo, h2e_mo, n_core, active)
    h2e_active = np.ascontiguousarray(h2e_mo[active, active, active, active])

    dets = generate_cisd_determinants(
        n_active_orb, n_alpha, n_beta, max_excitation=max_excitation)
    n_det = len(dets)
    if max_det is not None and n_det > max_det:
        raise ValueError(
            f"CISD determinant space ({n_det}) exceeds max_det ({max_det}). "
            "Freeze more core orbitals, shrink the basis, or raise max_det.")
    if nroots < 1 or nroots > n_det:
        raise ValueError(f"nroots must be in [1, {n_det}], got {nroots}")

    H = build_hamiltonian_matrix_unrestricted(dets, h1e_active, h2e_active)
    eigvals, eigvecs = eigh(H)

    e_const = e_core_dress + nuclear_repulsion
    e_total = float(eigvals[0]) + e_const

    # Reference-determinant energy (== SCF energy for SCF MOs) for e_corr.
    a_ref, b_ref = tuple(range(n_alpha)), tuple(range(n_beta))
    e_ref = diagonal_matrix_element_unrestricted(
        a_ref, b_ref, h1e_active, h2e_active) + e_const

    det_to_idx = {d: i for i, d in enumerate(dets)}
    ref_idx = det_to_idx[(a_ref, b_ref)]
    ref_weight = float(eigvecs[ref_idx, 0] ** 2)

    result = CISDResult(
        e_total=e_total,
        e_corr=e_total - e_ref,
        e_ref=e_ref,
        ci_coeffs=eigvecs[:, 0].copy(),
        determinants=dets,
        n_det=n_det,
        reference_weight=ref_weight,
        max_excitation=max_excitation,
        n_core=n_core,
        converged=True,
        eigenvalues=eigvals + e_const,
        e_totals=[float(eigvals[i]) + e_const for i in range(nroots)],
        ci_coeffs_all=eigvecs[:, :nroots] if nroots > 1 else None,
    )
    result._nalpha = n_alpha
    result._nbeta = n_beta
    return result
