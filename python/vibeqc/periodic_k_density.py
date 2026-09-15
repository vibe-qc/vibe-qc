"""Shared per-k density helpers for periodic SCF drivers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import List, Sequence

import numpy as np

from ._vibeqc_core import (
    BlochKMesh,
    LatticeMatrixSet,
    real_space_density_from_kpoints_fractional,
)

__all__ = [
    "density_matrices_per_k",
    "real_space_density_from_per_k_density",
]


def density_matrices_per_k(
    C_per_k: Sequence[np.ndarray],
    occ_per_k: Sequence[np.ndarray],
) -> List[np.ndarray]:
    """Build Hermitian AO density matrices ``D(k)`` for each k-point."""
    if len(C_per_k) != len(occ_per_k):
        raise ValueError(
            "per-k density construction requires matching coefficient and "
            f"occupation lists; got {len(C_per_k)} and {len(occ_per_k)}"
        )

    densities: List[np.ndarray] = []
    for C_raw, occ_raw in zip(C_per_k, occ_per_k):
        C = np.asarray(C_raw, dtype=complex)
        occ = np.asarray(occ_raw, dtype=float).reshape(-1)
        D = (C * occ[None, :].astype(complex)) @ C.conj().T
        D = 0.5 * (D + D.conj().T)
        densities.append(D)
    return densities


class _FermiDensityDomainError(ValueError):
    """A density seed is outside the domain of the Fermi entropy."""


def _needs_fermi_seed_step(densities, overlaps, orthogonalizers, weights, capacity):
    """An atomic/restart guess may seed a Fock without defining an entropy."""
    try:
        _fermi_density_entropy(densities, overlaps, orthogonalizers, weights, capacity)
    except _FermiDensityDomainError:
        return True
    return False


def _fermi_density_entropy(densities, overlaps, orthogonalizers, weights, capacity):
    """Fermi entropy S/k_B of the actual (possibly mixed) AO density.

    Mermin (1965), Eqs. (1)-(4): energy and entropy are functionals of the
    same accepted density operator. Its natural occupations in the retained
    orthonormal space are eig(X^H S D S X), not the occupations of the next
    canonical Fock refill. Capacity is two for restricted and one per spin.
    Only roundoff outside the fermionic occupation interval is clipped.
    """
    if not (len(densities) == len(overlaps) == len(orthogonalizers) == len(weights)):
        raise ValueError("Fermi density entropy: inconsistent k-point counts")
    entropy = 0.0
    for density, overlap, x, weight in zip(densities, overlaps, orthogonalizers, weights):
        sx = np.asarray(overlap) @ np.asarray(x)
        occupation_matrix = sx.conj().T @ np.asarray(density) @ sx / capacity
        tolerance = 256 * np.finfo(float).eps * max(1, len(occupation_matrix))
        if (
            not np.isfinite(occupation_matrix).all()
            or np.max(np.abs(occupation_matrix - occupation_matrix.conj().T)) > tolerance
        ):
            raise ValueError("Fermi density entropy requires a finite Hermitian density")
        values = np.linalg.eigvalsh(occupation_matrix)
        if values.min() < -tolerance or values.max() > 1 + tolerance:
            raise _FermiDensityDomainError(
                "Fermi density entropy is undefined for accepted natural "
                "occupations outside the fermionic interval"
            )
        values = np.clip(values, 0., 1.)
        live = (values > 0.) & (values < 1.)
        f = values[live]
        entropy -= float(weight) * capacity * float(np.sum(
            f * np.log(f) + (1 - f) * np.log1p(-f)
        ))
    return entropy


def real_space_density_from_per_k_density(
    D_per_k: Sequence[np.ndarray],
    kmesh: BlochKMesh,
    cells: Sequence[object],
) -> LatticeMatrixSet:
    """Inverse-Bloch fold already-formed per-k AO densities.

    ``real_space_density_from_kpoints_fractional`` owns the canonical
    C++ fold from Bloch orbitals.  Diagonalising each Hermitian
    ``D(k)`` lets SCF loops reuse that fold for mixed or damped
    densities, where no MO coefficient/occupation pair exists.
    """
    if len(D_per_k) == 0:
        raise ValueError("cannot fold an empty per-k density list")

    C_pseudo: List[np.ndarray] = []
    occ_pseudo: List[np.ndarray] = []
    for D_raw in D_per_k:
        D = np.asarray(D_raw, dtype=complex)
        D = 0.5 * (D + D.conj().T)
        occ, C = np.linalg.eigh(D)
        C_pseudo.append(C.astype(complex))
        occ_pseudo.append(np.asarray(occ, dtype=float))
    return real_space_density_from_kpoints_fractional(
        C_pseudo,
        occ_pseudo,
        kmesh,
        cells,
    )


@dataclass(frozen=True)
class _LatticeDensityReturnPlan:
    """An admitted complete BvK representation of an accepted SCF state.

    This representation is independent of integral/overlap cutoffs. One
    integer residue system and its inverse cells suffice, including on
    shifted meshes where the fold is antiperiodic across a supercell.
    ``reserved_peak_bytes`` covers both spin channels, Python/native copies,
    transform scratch and bookkeeping; it excludes the caller's SCF state.
    No dense cell-by-k phase table or overlap lattice is constructed.
    """

    cells: tuple
    torus_count: int
    kpoints: np.ndarray
    weights: np.ndarray
    partner_indices: tuple[int, ...]
    mesh: tuple[int, int, int]
    nbf: int
    n_channels: int
    reserved_peak_bytes: int

    def fold(self, matrices: Sequence[np.ndarray]) -> LatticeMatrixSet:
        """Certify, fold, and round-trip the original matrices losslessly."""
        from ._vibeqc_core import make_lattice_matrix_set

        if len(matrices) != len(self.kpoints):
            raise ValueError("GDF lattice density: density/k-point count mismatch")
        shape = (self.nbf, self.nbf)
        scale = 1.0
        for matrix in matrices:
            d = np.asarray(matrix)
            if d.shape != shape or not np.isfinite(d).all():
                raise ValueError("GDF lattice density: invalid accepted density")
            scale = max(scale, float(np.max(np.abs(d))))
            if np.max(np.abs(d - d.conj().T)) > 5e-11 * scale:
                raise ValueError("GDF lattice density: accepted density is not Hermitian")
        tolerance = max(5e-12, 256 * np.finfo(float).eps * len(matrices)) * scale
        for i, j in enumerate(self.partner_indices):
            if np.max(np.abs(np.asarray(matrices[j]) - np.asarray(matrices[i]).conj())) > tolerance:
                raise ValueError(
                    "GDF lattice density: accepted density violates time reversal; "
                    "a real lattice representation would change the SCF state"
                )
        blocks = []
        for cell in self.cells:
            block = np.zeros(shape, dtype=complex)
            for k, w, d in zip(self.kpoints, self.weights, matrices):
                block += (w * np.exp(-1j * np.dot(k, cell.r_cart))) * np.asarray(d)
            if np.max(np.abs(block.imag)) > tolerance:
                raise ValueError("GDF lattice density: inverse Bloch fold is not real")
            blocks.append(np.array(block.real, order="F"))
        # Do not Hermitize this inverse: doing so could hide information lost
        # by a real projection. Compare directly with every accepted D(k).
        for k, original in zip(self.kpoints, matrices):
            recovered = np.zeros(shape, dtype=complex)
            for cell, block in zip(
                self.cells[:self.torus_count], blocks[:self.torus_count]
            ):
                recovered += np.exp(1j * np.dot(k, cell.r_cart)) * block
            if np.max(np.abs(recovered - original)) > tolerance:
                raise ValueError(
                    "GDF lattice density: inverse Bloch round trip changed "
                    "the accepted per-k density"
                )
        return make_lattice_matrix_set(self.nbf, list(self.cells), blocks)

    def attach(self, result):
        """Attach only after every accepted spin channel passes certification."""
        if (
            np.shape(result.kpoints_cart) != self.kpoints.shape
            or np.shape(result.kpoint_weights) != self.weights.shape
            or not np.allclose(result.kpoints_cart, self.kpoints, rtol=0, atol=1e-12)
            or not np.allclose(result.kpoint_weights, self.weights, rtol=0, atol=1e-14)
        ):
            raise ValueError("GDF lattice density: returned mesh differs from the producer mesh")
        if self.n_channels == 2:
            alpha = self.fold(result.density_alpha)
            beta = self.fold(result.density_beta)
            result.density_alpha_lattice = alpha
            result.density_beta_lattice = beta
        else:
            result.density_lattice = self.fold(result.density)
        result.density_lattice_mesh = self.mesh
        result.density_lattice_reserved_peak_bytes = self.reserved_peak_bytes
        return result


def _lattice_density_return_reservation(mesh, nbf, n_channels, memory_byte_cap):
    """Admit the transform before expanding either cells or k points."""
    mesh_raw = tuple(mesh)
    if len(mesh_raw) != 3 or any(int(n) != n or n < 1 for n in mesh_raw):
        raise ValueError("GDF lattice density requires a three-axis integer BvK mesh")
    mesh = tuple(int(n) for n in mesh_raw)
    nk = prod(mesh)
    if nbf < 1 or n_channels not in (1, 2):
        raise ValueError("GDF lattice density: invalid basis/spin dimensions")
    # At most two complete tori; account for both native and Python storage
    # and twelve complex AO scratch matrices. Refuse before index expansion.
    reserved = (
        32 * n_channels * nk * nbf**2 + 192 * nbf**2
        + 1536 * nk + 4096
    )
    if reserved > memory_byte_cap:
        raise MemoryError(
            f"GDF lattice density requires {reserved} reserved bytes; "
            f"cap is {memory_byte_cap}"
        )
    return mesh, nk, reserved


def _plan_lattice_density_return(
    system,
    kpoints,
    weights,
    mesh,
    *,
    nbf: int,
    n_channels: int,
    memory_byte_cap: int,
) -> _LatticeDensityReturnPlan:
    """Validate the producer's full uniform MP mesh before SCF allocation.

    The density convention is D(R)=sum_k w_k exp(-ik.R)D(k), with the
    full finite character set. It has an exact inverse on one residue
    system. Real storage additionally requires D(-k)=conj(D(k)); that
    property is checked on the accepted state by :meth:`fold`.
    """
    from ._vibeqc_core import LatticeCell

    mesh, nk, reserved = _lattice_density_return_reservation(
        mesh, nbf, n_channels, memory_byte_cap,
    )
    if np.iscomplexobj(kpoints) or np.iscomplexobj(weights):
        raise ValueError("GDF lattice density requires real mesh coordinates and weights")
    kpoints = np.array(kpoints, dtype=float, copy=True)
    weights = np.array(weights, dtype=float, copy=True)
    if (
        kpoints.shape != (nk, 3) or weights.shape != (nk,)
        or not np.isfinite(kpoints).all() or not np.isfinite(weights).all()
        or not np.allclose(weights, 1 / nk, rtol=0, atol=2e-13)
    ):
        raise ValueError("GDF lattice density requires the full uniform BvK mesh")
    fractional = kpoints @ np.asarray(system.lattice) / (2 * np.pi)
    indices = np.empty((nk, 3), dtype=np.int64)
    shifts = []
    for axis, period in enumerate(mesh):
        scaled = fractional[:, axis] * period
        centered = scaled - np.rint(scaled)
        if np.all(np.abs(centered) < 2e-10):
            shift = 0
        elif np.all(np.abs(np.abs(centered) - 0.5) < 2e-10):
            shift = 1
        else:
            raise ValueError("GDF lattice density requires an unshifted or half-shifted MP mesh")
        if axis >= int(system.dim) and (period != 1 or shift or np.any(np.abs(fractional[:, axis]) > 2e-10)):
            raise ValueError("GDF lattice density has sampling on a nonperiodic axis")
        indices[:, axis] = np.rint(scaled - 0.5 * shift).astype(np.int64) % period
        shifts.append(shift)
    address = {tuple(row): i for i, row in enumerate(indices)}
    if len(address) != nk:
        raise ValueError("GDF lattice density requires every BvK residue exactly once")
    partners = tuple(
        address[tuple((-row[a] - shifts[a]) % mesh[a] for a in range(3))]
        for row in indices
    )
    keys = list(product(*(range(-((n - 1) // 2), n // 2 + 1) for n in mesh)))
    known = set(keys)
    keys.extend(sorted({tuple(-x for x in key) for key in keys} - known))
    cells = tuple(LatticeCell(system, key) for key in keys)
    kpoints.flags.writeable = False
    weights.flags.writeable = False
    return _LatticeDensityReturnPlan(
        cells, nk, kpoints, weights, partners, mesh, nbf, n_channels, reserved
    )
