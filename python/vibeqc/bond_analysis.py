"""Bond analysis: Wiberg bond orders, delocalization index, and bond summaries.

Extends the existing Mayer bond orders in :mod:`vibeqc.properties` with
complementary bond-order metrics for both molecular and periodic systems.

Public API
----------

.. autofunction:: wiberg_bond_orders
.. autofunction:: delocalization_index
.. autofunction:: bond_order_summary
.. autofunction:: periodic_wiberg_bond_orders
.. autofunction:: periodic_delocalization_index

Theory references
-----------------

- Wiberg, K. B., Tetrahedron 24, 1083 (1968). DOI: 10.1016/0040-4020(68)88057-3
  (Wiberg bond index: W_AB = sum_{mu in A} sum_{nu in B} |P_munu|^2)
- Mayer, I., Chem. Phys. Lett. 97, 270 (1983). DOI: 10.1016/0009-2614(83)80005-0
  (Mayer bond order: B_AB = sum (PS)_munu (PS)_numu)
- Outeiral, C.; Vincent, M. A.; Martín Pendás, Á.; Popelier, P. L. A.,
  Chem. Sci. 9, 5517 (2018). DOI: 10.1039/C8SC01338A
  (Delocalization index as a bond order from QTAIM basins; AO-approximated here)
- Matito, E.; Solà, M.; Salvador, P.; Duran, M., Faraday Discuss. 135, 325
  (2007). DOI: 10.1039/B605086G
  (Electron sharing indexes / delocalization index at the correlated level)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from ._vibeqc_core import compute_overlap
from .spin_channels import spin_densities

if TYPE_CHECKING:
    from ._vibeqc_core import BasisSet, Molecule

__all__ = [
    "BondOrderSummary",
    "wiberg_bond_orders",
    "iao_wiberg_bond_orders",
    "delocalization_index",
    "bond_order_summary",
    "periodic_wiberg_bond_orders",
    "periodic_delocalization_index",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _shell_to_atom(basis) -> list[int]:
    """Map each AO to its parent atom index, as a plain list.

    Delegates to :func:`vibeqc.properties._shell_to_atom`, the canonical
    derivation; the local copy this replaces assumed pure spherical AOs
    and so came out short on a Cartesian basis. The list return type is
    kept for the loop-indexing call sites below.
    """
    from .properties import _shell_to_atom as _canonical

    return [int(a) for a in _canonical(basis)]


def _real_if_hermitian(mat: np.ndarray, what: str = "density matrix") -> np.ndarray:
    """Real part of a complex-but-Hermitian density (canonical impl).

    Delegates to :func:`vibeqc.properties._real_if_hermitian` -- quiet for
    Hermitian complex densities (periodic Bloch phases), warns only on a
    genuine Hermiticity violation.
    """
    from .properties import _real_if_hermitian as _canonical

    return _canonical(np.asarray(mat), what=what)


# ---------------------------------------------------------------------------
# Wiberg bond orders
# ---------------------------------------------------------------------------


def iao_wiberg_bond_orders(
    density_alpha: np.ndarray,
    density_beta: np.ndarray,
    atom_indices: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    """Spin-resolved Wiberg indices in labeled orthonormal IAO bases.

    ``B[a,b] = 2 sum_sigma sum_mu_in_a,nu_in_b |D_sigma[mu,nu]|^2``.
    Alpha and beta may use different IAO bases, with the same atom labels.
    For a determinant, Wick contraction gives number covariance
    ``Cov(N_a,N_b) = -sum_sigma ||D_sigma[a,b]||_F^2`` for a != b;
    this index is minus twice that covariance. In a restricted determinant
    it is the Wiberg square of the total IAO density. General correlated
    number covariances require the 2-RDM and are not implemented here.

    Wiberg, Tetrahedron 24, 1083 (1968),
    doi:10.1016/0040-4020(68)88057-3; determinant interpretation discussed
    by de Giambiagi et al., Theor. Chim. Acta 68, 337 (1985),
    doi:10.1007/BF00529054. No original-AO Loewdin transform is performed.
    """
    from .iao_population import _atom_labels, _hermitian

    if isinstance(n_atoms, bool) or not isinstance(n_atoms, (int, np.integer)) or n_atoms < 1:
        raise ValueError("n_atoms must be a positive integer")
    da = _hermitian(density_alpha, "alpha IAO density")
    db = _hermitian(density_beta, "beta IAO density", da.shape)
    labels = _atom_labels(atom_indices, len(da), n_atoms)
    with np.errstate(over="ignore", invalid="ignore"):
        residuals = [np.linalg.norm(d @ d - d) for d in (da, db)]
    if not np.isfinite(residuals).all() or max(residuals) > 1e-7:
        raise ValueError("IAO-Wiberg analysis requires idempotent spin determinant densities")
    weights = 2. * (np.abs(da) ** 2 + np.abs(db) ** 2)
    bonds = np.zeros((n_atoms, n_atoms))
    np.add.at(bonds, (labels[:, None], labels[None, :]), weights)
    np.fill_diagonal(bonds, 0.)
    return bonds


def wiberg_bond_orders(
    result,
    basis: BasisSet,
    molecule: Molecule,
) -> np.ndarray:
    """Wiberg bond-order matrix, shape ``(n_atoms, n_atoms)``.

    The Wiberg bond index between atoms A and B is::

        W_AB = sum_{mu in A} sum_{nu in B} |P_munu|^2

    where P is the spin-summed density matrix transformed to the orthonormal
    (Löwdin) basis.

    The Wiberg index differs from the Mayer bond order in using *squared
    absolute values* of the Löwdin-basis density matrix elements rather than
    the AO-basis ``(P S)(P S)`` product. The two values are distinct metrics
    and must not be substituted or relabeled based on overlap conditioning.
    """
    S = np.asarray(compute_overlap(basis))
    ao_to_atom = _shell_to_atom(basis)
    n_atoms = len(molecule.atoms)
    n_ao = S.shape[0]

    # Density in the Löwdin orthonormal AO basis:
    # P' = S^{1/2} P S^{1/2}.  Using S^{-1/2} here applies the inverse
    # basis transform to an AO density and explodes in flexible bases.
    evals, evecs = np.linalg.eigh(S)
    mask = evals > 1e-14
    sqrt_evals = np.zeros_like(evals)
    sqrt_evals[mask] = np.sqrt(evals[mask])
    S_half = evecs @ np.diag(sqrt_evals) @ evecs.T

    alpha, beta = spin_densities(result)
    if alpha is not None:
        Pa = _real_if_hermitian(alpha, "alpha density")
        Pb = _real_if_hermitian(beta, "beta density")
        P_tilde = S_half @ (Pa + Pb) @ S_half
    else:
        P = _real_if_hermitian(result.density)
        P_tilde = S_half @ P @ S_half

    bond_orders = np.zeros((n_atoms, n_atoms), dtype=np.float64)
    for mu in range(n_ao):
        a = ao_to_atom[mu]
        for nu in range(n_ao):
            b = ao_to_atom[nu]
            if a != b:
                bond_orders[a, b] += P_tilde[mu, nu] ** 2

    bond_orders = 0.5 * (bond_orders + bond_orders.T)
    return bond_orders


# ---------------------------------------------------------------------------
# Delocalization index (AO-approximated)
# ---------------------------------------------------------------------------


def delocalization_index(
    result,
    basis: BasisSet,
    molecule: Molecule,
) -> np.ndarray:
    r"""Delocalization index matrix, shape ``(n_atoms, n_atoms)``.

    The delocalization index (DI) between atoms A and B is defined in
    QTAIM as the double integral of the exchange-correlation density
    over the atomic basins. Here we approximate it from the AO density
    matrix following the Mayer-like formula but using the squared
    density in the non-orthogonal AO basis::

        DI_AB = 2 * sum_{mu in A} sum_{nu in B} (PS)_{munu} (PS)_{numu}
                - sum_{mu in A} sum_{nu in B} (P^S)_{munu} (P^S)_{numu}

    where P^S = S^{1/2} P S^{1/2} is the density in the Löwdin basis.

    For closed-shell systems the result is halved to match the
    QTAIM convention (electron pairs shared).

    References
    ----------
    Matito, E.; Solà, M.; Salvador, P.; Duran, M.,
    Faraday Discuss. 135, 325 (2007). DOI: 10.1039/B605086G
    Outeiral, C. et al., Chem. Sci. 9, 5517 (2018). DOI: 10.1039/C8SC01338A
    """
    S = np.asarray(compute_overlap(basis))
    ao_to_atom = _shell_to_atom(basis)
    n_atoms = len(molecule.atoms)
    n_ao = S.shape[0]

    alpha, beta = spin_densities(result)
    if alpha is not None:
        Pa = _real_if_hermitian(alpha, "alpha density")
        Pb = _real_if_hermitian(beta, "beta density")
        P = Pa + Pb
    else:
        P = _real_if_hermitian(result.density)

    # Löwdin-basis density
    evals, evecs = np.linalg.eigh(S)
    mask = evals > 1e-14
    sqrt_evals = np.zeros_like(evals)
    sqrt_evals[mask] = np.sqrt(evals[mask])
    S_half = evecs @ np.diag(sqrt_evals) @ evecs.T
    P_tilde = S_half @ P @ S_half

    deloc = np.zeros((n_atoms, n_atoms), dtype=np.float64)
    for mu in range(n_ao):
        a = ao_to_atom[mu]
        for nu in range(n_ao):
            b = ao_to_atom[nu]
            if a != b:
                # Off-diagonal exchange: 2*(PS)_{munu}*(PS)_{numu}
                # minus the Löwdin-basis self-term
                term_ps = 0.0
                term_tilde = 0.0
                for lam in range(n_ao):
                    term_ps += P[mu, lam] * S[lam, nu] * P[nu, lam] * S[lam, mu]
                deloc[a, b] += 2.0 * term_ps - P_tilde[mu, nu] ** 2

    deloc = 0.5 * (deloc + deloc.T)
    return deloc


# ---------------------------------------------------------------------------
# Periodic Wiberg bond orders
# ---------------------------------------------------------------------------


def periodic_wiberg_bond_orders(
    result,
    basis: BasisSet,
    molecule: Molecule,
) -> np.ndarray:
    """Wiberg bond orders for periodic (Gamma-point) SCF results.

    Uses the same definition as :func:`wiberg_bond_orders` but handles
    complex density matrices (Bloch-summed to Gamma) by taking the
    real part of P_tilde.
    """
    return wiberg_bond_orders(result, basis, molecule)


# ---------------------------------------------------------------------------
# Periodic delocalization index
# ---------------------------------------------------------------------------


def periodic_delocalization_index(
    result,
    basis: BasisSet,
    molecule: Molecule,
) -> np.ndarray:
    """Delocalization index for periodic (Gamma-point) SCF results."""
    return delocalization_index(result, basis, molecule)


# ---------------------------------------------------------------------------
# Bond order summary
# ---------------------------------------------------------------------------


@dataclass
class BondOrderSummary:
    """Container for multiple bond-order metrics on the same molecule.

    Each field is a ``(n_atoms, n_atoms)`` numpy array or None if the
    computation failed or was skipped. Diagonal entries are free valences;
    off-diagonals are bond orders.
    """

    mayer: Optional[np.ndarray] = None
    wiberg: Optional[np.ndarray] = None
    delocalization: Optional[np.ndarray] = None
    errors: dict[str, str] = field(default_factory=dict)

    def top_bonds(
        self,
        metric: str = "mayer",
        molecule: Optional[Molecule] = None,
        *,
        threshold: float = 0.10,
        n_top: int = 20,
    ) -> list[tuple[int, int, float, str, str]]:
        """Return ``[(i, j, order, sym_i, sym_j)]`` sorted descending.

        Parameters
        ----------
        metric : str
            Which bond-order matrix to use: ``"mayer"``, ``"wiberg"``,
            or ``"delocalization"``.
        molecule : Molecule, optional
            Needed to resolve element symbols. If not given, symbols
            are empty strings.
        threshold : float
            Minimum bond order to include.
        n_top : int
            Maximum number of bonds to return.
        """
        arr = getattr(self, metric, None)
        if arr is None:
            return []
        bo = np.asarray(arr)
        n = bo.shape[0]
        entries: list[tuple[int, int, float, str, str]] = []
        for i in range(n):
            for j in range(i + 1, n):
                v = float(bo[i, j])
                if v >= threshold:
                    si = ""
                    sj = ""
                    if molecule is not None:
                        atoms = list(molecule.atoms)
                        from .output.formats.xyz import _symbol

                        si = _symbol(int(atoms[i].Z))
                        sj = _symbol(int(atoms[j].Z))
                    entries.append((i, j, v, si, sj))
        entries.sort(key=lambda t: -t[2])
        return entries[:n_top]


def bond_order_summary(
    result,
    basis: BasisSet,
    molecule: Molecule,
    *,
    compute_mayer: bool = True,
    compute_wiberg: bool = True,
    compute_delocalization: bool = False,
) -> BondOrderSummary:
    """Compute multiple bond-order metrics in one pass.

    Returns a :class:`BondOrderSummary` with each successfully-computed
    metric populated. Failures are captured in ``.errors``.
    """
    summary = BondOrderSummary()

    if compute_mayer:
        try:
            from .properties import mayer_bond_orders

            summary.mayer = mayer_bond_orders(result, basis, molecule)
        except Exception as exc:
            summary.errors["mayer"] = f"{type(exc).__name__}: {exc}"

    if compute_wiberg:
        try:
            summary.wiberg = wiberg_bond_orders(result, basis, molecule)
        except Exception as exc:
            summary.errors["wiberg"] = f"{type(exc).__name__}: {exc}"

    if compute_delocalization:
        try:
            summary.delocalization = delocalization_index(result, basis, molecule)
        except Exception as exc:
            summary.errors["delocalization"] = f"{type(exc).__name__}: {exc}"

    return summary
