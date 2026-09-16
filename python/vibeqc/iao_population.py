"""Molecular determinant IAO populations and spin-resolved IAO-Wiberg indices.

Construction: Knizia, JCTC 9, 4834 (2013), doi:10.1021/ct400687b.
The partition uses symmetric MINI IAOs separately for each occupied spin
space. It does not require orbital localization or dipole integrals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .iao import (
    IAO_REFERENCE_BASIS, IAOReference, build_iaos, iao_reference,
    iao_unsupported_reason,
)

__all__ = ["IAOAnalysis", "analyse_iao", "analyse_iao_occupied"]

# Absolute tolerance for populations and metric identities; relative tolerance
# for comparing an AO density to its occupied-MO factorization. No eigenvectors
# or electrons are discarded to force these checks to pass.
_VALIDATION_TOL = 1e-7
_OCCUPATION_TOL = 1e-10
PERIODIC_IAO_UNAVAILABLE = (
    "Periodic IAO population analysis is not implemented: normalized k weights, "
    "spin occupations and lattice-resolved bond orders require a periodic adapter"
)


@dataclass(frozen=True)
class IAOAnalysis:
    """A full atom partition, or an explicit reason it is unavailable.

    Charges are in elementary charge units, populations and spins in electrons,
    and bond orders dimensionless. ``bond_orders`` is dense and unthresholded;
    its diagonal is zero. ``spin_populations`` is absent for restricted states.
    The coefficient/reference fields allow reuse by molecular localization and
    are deliberately excluded from the structured output payload.
    """

    populations: np.ndarray | None = None
    charges: np.ndarray | None = None
    spin_populations: np.ndarray | None = None
    bond_orders: np.ndarray | None = None
    reference_basis: str = IAO_REFERENCE_BASIS
    orthogonalization: str = "symmetric"
    method: str = "iao-wiberg"
    density_convention: str = "integer-occupied all-electron SCF determinant"
    spin_convention: str = "2 sum_sigma sum_mu_in_a,nu_in_b |D_sigma[mu,nu]|^2"
    partition: str = "separate spin IAOs with common reference atom labels"
    diagnostics: dict[str, float] = field(default_factory=dict)
    unavailable_reason: str | None = None
    iaos_alpha: np.ndarray | None = field(default=None, repr=False, compare=False)
    iaos_beta: np.ndarray | None = field(default=None, repr=False, compare=False)
    reference: IAOReference | None = field(default=None, repr=False, compare=False)

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None and self.charges is not None

    def to_dict(self) -> dict[str, Any]:
        """Portable numerical payload; atom order is the input molecule order."""
        return {
            "schema_version": 1,
            "available": self.available,
            "atom_indices": None if self.charges is None else list(range(len(self.charges))),
            "unavailable_reason": self.unavailable_reason,
            "method": self.method,
            "reference_basis": self.reference_basis,
            "orthogonalization": self.orthogonalization,
            "density_convention": self.density_convention,
            "spin_convention": self.spin_convention,
            "partition": self.partition,
            "units": {"charges": "e", "populations": "electrons",
                      "spin_populations": "alpha minus beta electrons",
                      "bond_orders": "dimensionless"},
            "populations": None if self.populations is None else self.populations.tolist(),
            "charges": None if self.charges is None else self.charges.tolist(),
            "spin_populations": (None if self.spin_populations is None
                                 else self.spin_populations.tolist()),
            "bond_orders": None if self.bond_orders is None else self.bond_orders.tolist(),
            "diagnostics": dict(self.diagnostics),
        }


def _matrix(value: Any, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    a = np.asarray(value)
    if a.ndim != 2 or a.dtype.kind not in "fci" or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite numeric matrix")
    if shape is not None and a.shape != shape:
        raise ValueError(f"{name} shape {a.shape} does not match {shape} in basis AO order")
    return a


def _hermitian(value: Any, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    a = _matrix(value, name, shape)
    if a.shape[0] != a.shape[1]:
        raise ValueError(f"{name} must be square")
    if np.linalg.norm(a - a.conj().T) > _VALIDATION_TOL * max(1., np.linalg.norm(a)):
        raise ValueError(f"{name} is not Hermitian")
    return a * 0.5 + a.conj().T * 0.5


def _real_scalar(value: complex, name: str) -> float:
    if not np.isfinite(value) or abs(np.imag(value)) > _VALIDATION_TOL:
        raise ValueError(f"{name} has a non-finite or non-numerical imaginary component")
    return float(np.real(value))


def _atom_labels(atom_indices: Any, n_ref: int, n_atoms: int) -> np.ndarray:
    labels = np.asarray(atom_indices)
    if (labels.shape != (n_ref,) or labels.dtype.kind not in "iu"
            or np.any(labels < 0) or np.any(labels >= n_atoms)
            or set(labels.tolist()) != set(range(n_atoms))):
        raise ValueError("Reference atom assignments must label every atom in input order")
    return labels


def analyse_iao_occupied(
    occupied_alpha: np.ndarray,
    overlap: np.ndarray,
    reference_overlap: np.ndarray,
    cross_overlap: np.ndarray,
    nuclear_charges: np.ndarray,
    atom_indices: np.ndarray,
    *,
    occupied_beta: np.ndarray | None = None,
    reference_basis: str = IAO_REFERENCE_BASIS,
) -> IAOAnalysis:
    """Analyze unit-occupied, S-orthonormal MO columns in the supplied AO order.

    ``occupied_beta=None`` denotes a restricted determinant: alpha and beta
    share the input occupied set and the IAO construction. To represent an
    empty beta spin, explicitly pass an ``(n_ao, 0)`` matrix. All occupied
    orbitals, including core orbitals, must be supplied. Invalid or ill-defined
    spaces raise ValueError/LinAlgError; :func:`analyse_iao` converts those
    failures into an unavailable result at the molecular adapter boundary.
    """
    from .bond_analysis import iao_wiberg_bond_orders

    s = _hermitian(overlap, "AO overlap")
    s2 = _hermitian(reference_overlap, "reference overlap")
    s12 = _matrix(cross_overlap, "cross overlap", (len(s), len(s2)))
    z = np.asarray(nuclear_charges)
    if (z.ndim != 1 or z.size == 0 or z.dtype.kind not in "fiu"
            or not np.isfinite(z).all() or np.any(z <= 0)
            or np.any(z != np.floor(z))):
        raise ValueError("All-electron nuclear charges must be positive integers; ghosts are unsupported")
    labels = _atom_labels(atom_indices, len(s2), len(z))
    diagnostics: dict[str, float] = {
        "validation_tolerance": _VALIDATION_TOL,
        "metric_eigenvalue_floor_factor": 1e-10,
        "occupation_tolerance": _OCCUPATION_TOL,
    }

    def channel(occupied, spin):
        c = _matrix(occupied, f"{spin} occupied coefficients")
        if c.shape[0] != len(s):
            raise ValueError(f"{spin} coefficients must have AO rows in basis order")
        if c.shape[1] > len(s2):
            raise ValueError(f"{spin} occupied space exceeds the minimal reference dimension")
        error = float(np.linalg.norm(c.conj().T @ s @ c - np.eye(c.shape[1])))
        diagnostics[f"{spin}_occupied_metric_residual"] = error
        if error > _VALIDATION_TOL:
            raise ValueError(f"{spin} occupied coefficients are not S-orthonormal")
        a = build_iaos(c, s, s2, s12)
        x = a.conj().T @ s @ c
        d = _hermitian(x @ x.conj().T, f"{spin} IAO density")
        metric_error = float(np.linalg.norm(a.conj().T @ s @ a - np.eye(len(s2))))
        residual = c - a @ x
        span_error = float(np.sqrt(max(0., _real_scalar(
            np.trace(residual.conj().T @ s @ residual), "occupied span norm"))))
        trace = _real_scalar(np.trace(d), f"{spin} density trace")
        idem = float(np.linalg.norm(d @ d - d))
        diagnostics.update({
            f"{spin}_iao_metric_residual": metric_error,
            f"{spin}_occupied_span_residual": span_error,
            f"{spin}_electrons": trace,
            f"{spin}_electron_residual": trace - c.shape[1],
            f"{spin}_idempotency_residual": idem,
        })
        if max(metric_error, span_error, abs(trace - c.shape[1]), idem) > _VALIDATION_TOL:
            raise ValueError(f"{spin} IAOs fail orthonormality, occupied-span or electron conservation checks")
        return a, d

    a, da = channel(occupied_alpha, "alpha")
    restricted = occupied_beta is None
    if restricted:
        b, db = a, da
        diagnostics.update({k.replace("alpha_", "beta_"): v
                            for k, v in list(diagnostics.items()) if k.startswith("alpha_")})
    else:
        b, db = channel(occupied_beta, "beta")
    pa = np.bincount(labels, weights=da.diagonal().real, minlength=len(z))
    pb = np.bincount(labels, weights=db.diagonal().real, minlength=len(z))
    populations = pa + pb
    bonds = iao_wiberg_bond_orders(da, db, labels, len(z))
    return IAOAnalysis(
        populations=populations, charges=z - populations,
        spin_populations=None if restricted else pa - pb, bond_orders=bonds,
        reference_basis=reference_basis, diagnostics=diagnostics,
        partition=("shared restricted IAOs; total density split equally" if restricted
                   else "separate spin IAOs with common reference atom labels"),
        iaos_alpha=a, iaos_beta=b,
    )


def _occupied_from_result(result, suffix, count, overlap, occupancy, diagnostics):
    coeff = _matrix(getattr(result, f"mo_coeffs{suffix}", None), f"MO coefficients{suffix}")
    if coeff.shape[0] != len(overlap) or coeff.shape[1] < count:
        raise ValueError("MO coefficients must contain all occupied orbitals with AO rows in basis order")
    # Native RHF/RKS and UHF/UKS results use leading occupied columns and
    # carry no occupation vector. Imported adapters may supply one explicitly.
    occ = getattr(result, f"occupations{suffix}", None)
    if occ is None:
        selected = coeff[:, :count]
    else:
        occ = np.asarray(occ)
        if (occ.shape != (coeff.shape[1],) or occ.dtype.kind not in "fiu"
                or not np.isfinite(occ).all()):
            raise ValueError("Occupations must be a finite vector matching the MO columns")
        empty = np.abs(occ) <= _OCCUPATION_TOL
        full = np.abs(occ - occupancy) <= _OCCUPATION_TOL
        if not np.all(empty | full):
            raise ValueError("Fractional occupations are unsupported; integer determinants are required")
        if np.count_nonzero(full) != count:
            raise ValueError("Occupation count disagrees with the molecular electron/spin count")
        selected = coeff[:, full]
    p = _hermitian(getattr(result, f"density{suffix}", None), f"AO density{suffix}", overlap.shape)
    expected = occupancy * (selected @ selected.conj().T)
    error = float(np.linalg.norm(p - expected) / max(1., np.linalg.norm(expected)))
    count_error = _real_scalar(np.einsum("ij,ji->", p, overlap), "AO electron count") - count * occupancy
    diagnostics[f"ao_density{suffix}_factorization_residual"] = error
    diagnostics[f"ao_density{suffix}_electron_residual"] = count_error
    if error > _VALIDATION_TOL or abs(count_error) > _VALIDATION_TOL:
        raise ValueError("AO density disagrees with integer occupied MOs or electron count; fractional/correlated densities and AO reordering are unsupported")
    return selected


def analyse_iao(
    result: Any, basis: Any, molecule: Any, *, method: str | None = None,
    uses_ecp: bool = False, periodic: bool = False,
) -> IAOAnalysis:
    """Analyze a converged molecular RHF/RKS/UHF/UKS result, without localization.

    The AO density convention is verified against ``f C_occ C_occ^H``:
    ``f=2`` for restricted total densities and ``f=1`` for each unrestricted
    channel. Molecular charge/multiplicity determine all-electron counts.
    Unsupported requests return an :class:`IAOAnalysis` with no numerical
    populations and an explicit ``unavailable_reason``.
    """
    from dataclasses import replace
    from ._vibeqc_core import compute_overlap
    from .ecp_metadata import effective_nuclear_charges_from

    try:
        if (periodic or getattr(molecule, "lattice", None) is not None
                or "periodic" in type(result).__name__.lower()
                or any(getattr(result, key, None) is not None
                       for key in ("kpoints", "k_points", "kpts", "kmesh", "lattice"))):
            raise ValueError(PERIODIC_IAO_UNAVAILABLE)
        reason = iao_unsupported_reason(molecule, uses_ecp=uses_ecp)
        if reason:
            raise ValueError(reason)
        z = np.array([atom.Z for atom in molecule.atoms])
        effective = np.asarray(effective_nuclear_charges_from(molecule, result))
        if not np.array_equal(effective, z):
            raise ValueError("IAO analysis requires all-electron densities; ECPs are unsupported")
        route = (method or getattr(result, "method", None)
                 or type(result).__name__.removesuffix("Result")).lower()
        if (route not in ("rhf", "rks", "uhf", "uks")
                or any(getattr(result, key, None) is not None
                       for key in ("mp2", "ccsd", "dlpno_mp2", "dlpno_ccsd"))):
            raise ValueError("IAO analysis supports molecular RHF/RKS/UHF/UKS determinants only")
        if not bool(getattr(result, "converged", False)):
            raise ValueError("IAO analysis requires a converged SCF determinant")
        if basis is None:
            raise ValueError("IAO analysis requires the SCF Gaussian basis in native AO order")
        s = _hermitian(compute_overlap(basis), "AO overlap")
        ne = molecule.n_electrons()
        spin = molecule.multiplicity - 1
        if (ne < 0 or spin < 0 or spin > ne or (ne + spin) % 2):
            raise ValueError("Inconsistent molecular electron count and multiplicity")
        na, nb = (ne + spin) // 2, (ne - spin) // 2
        for key, expected in (("n_alpha", na), ("n_beta", nb)):
            declared = getattr(result, key, None)
            if declared is not None and (
                isinstance(declared, (bool, np.bool_))
                or not isinstance(declared, (int, np.integer)) or declared != expected
            ):
                raise ValueError(f"{key} disagrees with the molecular electron/spin count")
        values = [getattr(result, key, None) for key in
                  ("density_alpha", "density_beta", "mo_coeffs_alpha", "mo_coeffs_beta")]
        unrestricted = route in ("uhf", "uks")
        if (any(v is not None for v in values) and not all(v is not None for v in values)):
            raise ValueError("Incomplete alpha/beta density or MO coefficient pair")
        if unrestricted != all(v is not None for v in values):
            raise ValueError("SCF method and alpha/beta channels are inconsistent")
        diagnostics: dict[str, float] = {}
        if unrestricted:
            ca = _occupied_from_result(result, "_alpha", na, s, 1., diagnostics)
            cb = _occupied_from_result(result, "_beta", nb, s, 1., diagnostics)
        else:
            if spin != 0:
                raise ValueError("Restricted IAO analysis requires a closed-shell determinant")
            ca = _occupied_from_result(result, "", na, s, 2., diagnostics)
            cb = None
        # Row dimensions alone cannot identify a Gaussian basis. Verify the
        # exposed atom labels and shell origins against this molecule too.
        from .properties import _shell_to_atom
        _atom_labels(_shell_to_atom(basis), len(s), len(z))
        for shell in basis.shells():
            if not np.allclose(shell.origin, molecule.atoms[shell.atom_index].xyz,
                               atol=1e-10, rtol=0):
                raise ValueError("Basis shell origins/atom labels do not match the input molecule")
        ref = iao_reference(molecule, basis)
        analysis = analyse_iao_occupied(ca, s, ref.overlap, ref.cross_overlap, z,
                                       ref.atom_indices, occupied_beta=cb,
                                       reference_basis=ref.name)
        diagnostics.update(analysis.diagnostics)
        diagnostics["total_charge_residual"] = float(analysis.charges.sum() - molecule.charge)
        diagnostics["spin_sum_residual"] = (0. if analysis.spin_populations is None else
                                             float(analysis.spin_populations.sum() - spin))
        if max(abs(diagnostics["total_charge_residual"]), abs(diagnostics["spin_sum_residual"])) > _VALIDATION_TOL:
            raise ValueError("IAO charges or spin populations do not conserve the molecular totals")
        return replace(analysis, reference=ref, diagnostics=diagnostics)
    except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
        return IAOAnalysis(unavailable_reason=str(exc))
