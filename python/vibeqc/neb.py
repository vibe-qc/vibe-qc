"""Nudged Elastic Band (NEB) -- minimum energy path finder.

Public surface
==============

Path construction (Increment 1):

* :class:`NEBImage`, :class:`NEBPath`
* :func:`interpolate_linear`, :func:`interpolate_idpp`

Driver:

* :func:`run_neb` -- improved-tangent NEB with parallel per-image
  SCFs and a quick-min outer loop on the concatenated NEB force.
  Supports molecular and periodic endpoints, climbing image
  (``climbing_image=True``), density warm-start across outer
  iterations (six molecular mean-field methods; four periodic), and
  DFT+U.
* :class:`NEBResult` -- converged path + energies + transition-state
  index + iteration count; ``write_qvf`` for vibe-view rendering.

Both interpolators accept either :class:`vibeqc.Molecule` or
:class:`vibeqc.PeriodicSystem`. For periodic systems, IDPP pair
distances and the NEB force (tangent + spring) use the minimum-image
convention, so a band whose images straddle a cell boundary -- e.g.
an adatom hopping across the PBC in surface self-diffusion --
interpolates and relaxes along the short, through-the-boundary path
rather than being dragged across the cell. (``interpolate_linear``
is a plain Cartesian straight line; use IDPP for cross-boundary
hops.) Minkowski reduction followed by a bounded closest-vector search
handles skewed and unreduced cells without assuming an orthorhombic basis.

Gaussian periodic per-image gradients are computed by central differences
(6N + 1 BIPOLE SCFs per image per outer iteration): the J^LR
(reciprocal-Ewald) contribution is still missing from the analytic
BIPOLE gradient, so the FD fallback is used to keep saddle-point
forces honest. Periodic semiempirical routes use their validated analytic or
total-energy finite-difference gradient. Full-k DFTB0/SCC-DFTB use the native
batched finite-difference Bloch kernel, while MSINDO SECCM uses its analytic
cyclic gradient. See ``docs/user_guide/neb.md``.

References
==========
* Henkelman & Jónsson, "Improved tangent estimate in the nudged
  elastic band method for finding minimum energy paths and saddle
  points", J. Chem. Phys. 113, 9978 (2000). doi:10.1063/1.1323224.
* Smidstrup, Pedersen, Stokbro, Jónsson,
  "Improved initial guess for minimum energy path calculations",
  J. Chem. Phys. 140, 214106 (2014). doi:10.1063/1.4878664.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
from scipy.optimize import minimize

from ._vibeqc_core import Atom, BasisSet, Molecule, PeriodicSystem
from .output import (
    OutputPlan,
    dry_run_manifest,
    is_dry_run_estimate_requested,
    is_dry_run_requested,
    write,
)

System = Union[Molecule, PeriodicSystem]


class NEBImageSCFError(RuntimeError):
    """Raised when a NEB image lacks a usable SCF stationary point.

    The band cannot be propagated from a non-converged image, an internally
    unstable image, or an image whose stability eigensolve is inconclusive:
    none supplies a certified energy and gradient on the intended electronic
    surface. The NEB driver surfaces these cases as clear, actionable errors
    naming the offending image and geometry.
    """


def _nonconverged_image_error(
    method: str,
    scf_result: Any,
    image_index: Optional[int],
    positions: np.ndarray,
) -> "NEBImageSCFError":
    """Build a clear :class:`NEBImageSCFError` for a non-converged image."""
    n_iter = getattr(scf_result, "n_iter", None)
    where = f"image {image_index}" if image_index is not None else "an image"
    iters = f" after {n_iter} iterations" if n_iter is not None else ""
    geom = np.array2string(np.asarray(positions, dtype=float), precision=4)
    return NEBImageSCFError(
        f"NEB {where}: the {method.upper()} SCF did not converge{iters}. "
        f"A non-converged density yields no valid energy or gradient, so "
        f"the band evaluation was aborted at this image. Raise the per-image "
        f"SCF iteration limit ({method.upper()}Options(max_iter=...), default "
        f"100) or improve the initial interpolation (more images / IDPP). "
        f"Geometry (bohr):\n{geom}"
    )


def _unstable_image_error(
    method: str,
    scf_result: Any,
    image_index: Optional[int],
    positions: np.ndarray,
) -> "NEBImageSCFError":
    """Build a loud error for a converged but unstable NEB image."""
    where = f"image {image_index}" if image_index is not None else "an image"
    eigenvalue = float(getattr(scf_result, "stability_eigenvalue", float("nan")))
    geom = np.array2string(np.asarray(positions, dtype=float), precision=4)
    return NEBImageSCFError(
        f"NEB {where}: the {method.upper()} SCF converged to an internally "
        f"unstable stationary point (lowest Hessian eigenvalue "
        f"{eigenvalue:.6e} Ha). Per-image corrective following is disabled "
        f"because independent basin switches can break electronic-state "
        f"continuity along the band, so this image cannot provide a valid "
        f"default NEB energy or gradient. Supply a continuous stable-state "
        f"initialisation for the path, or explicitly set "
        f"stability_check=False only when intentionally following an "
        f"uncertified diabatic branch. Geometry (bohr):\n{geom}"
    )


def _unverified_image_error(
    method: str,
    image_index: Optional[int],
    positions: np.ndarray,
) -> "NEBImageSCFError":
    """Build a loud error when an image stability solve did not settle."""
    where = f"image {image_index}" if image_index is not None else "an image"
    geom = np.array2string(np.asarray(positions, dtype=float), precision=4)
    return NEBImageSCFError(
        f"NEB {where}: the {method.upper()} stability eigensolver did not "
        f"converge, so the stationary-point character is UNVERIFIED. The "
        f"band evaluation was aborted before using this image's energy or "
        f"gradient. Increase stability_davidson_max_iter or explicitly set "
        f"stability_check=False only when intentionally following an "
        f"uncertified diabatic branch. Geometry (bohr):\n{geom}"
    )


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class NEBImage:
    """One image on a NEB path.

    ``energy``, ``gradient``, and ``tangent`` are populated by the
    NEB driver during optimisation. In Increment 1 they are all
    ``None`` for freshly interpolated images.
    """

    system: System
    energy: Optional[float] = None
    gradient: Optional[np.ndarray] = None  # (n_atoms, 3) Ha/bohr
    tangent: Optional[np.ndarray] = None  # (n_atoms, 3) unit-norm


@dataclass
class NEBPath:
    """A NEB path as an ordered list of images.

    First and last entries are the reactant and product respectively
    (fixed by default during optimisation). ``spring_constant`` is in
    Ha/bohr^2. ``climbing_image_index`` is None for standard NEB; the
    CI-NEB driver sets it to the highest-energy intermediate after
    the warm-up phase.
    """

    images: List[NEBImage]
    spring_constant: float = 0.1
    climbing_image_index: Optional[int] = None

    @property
    def n_images(self) -> int:
        return len(self.images)

    @property
    def n_intermediate(self) -> int:
        return max(0, len(self.images) - 2)

    def energies(self) -> np.ndarray:
        """Per-image energies as an array; NaN where unset."""
        return np.array(
            [
                img.energy if img.energy is not None else np.nan
                for img in self.images
            ],
            dtype=float,
        )


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _atom_iter(system: System):
    return system.atoms if isinstance(system, Molecule) else system.unit_cell


def _positions_of(system: System) -> np.ndarray:
    return np.array([list(a.xyz) for a in _atom_iter(system)], dtype=float)


def _atomic_numbers_of(system: System) -> np.ndarray:
    return np.array([int(a.Z) for a in _atom_iter(system)], dtype=int)


def _rebuild_with_positions(template: System, positions: np.ndarray) -> System:
    """Return a copy of ``template`` with Cartesian ``positions`` (bohr)."""
    if isinstance(template, Molecule):
        new_atoms = [
            Atom(int(a.Z), list(p))
            for a, p in zip(template.atoms, positions)
        ]
        return Molecule(new_atoms, template.charge, template.multiplicity)
    new_atoms = [
        Atom(int(a.Z), list(p))
        for a, p in zip(template.unit_cell, positions)
    ]
    return PeriodicSystem(
        template.dim,
        np.asarray(template.lattice, dtype=float),
        new_atoms,
        charge=template.charge,
        multiplicity=template.multiplicity,
    )


def _check_compatible(reactant: System, product: System) -> None:
    if type(reactant) is not type(product):
        raise ValueError(
            "reactant and product must be the same system type; got "
            f"{type(reactant).__name__} and {type(product).__name__}"
        )
    zr = _atomic_numbers_of(reactant)
    zp = _atomic_numbers_of(product)
    if zr.shape != zp.shape or not np.array_equal(zr, zp):
        raise ValueError(
            "reactant and product must have matching atomic-number "
            "sequences in the same order (NEB does not reorder atoms; "
            "pre-align if needed)."
        )
    if int(reactant.charge) != int(product.charge):
        raise ValueError(
            "reactant and product must have the same charge; got "
            f"{reactant.charge} and {product.charge}"
        )
    if int(reactant.multiplicity) != int(product.multiplicity):
        raise ValueError(
            "reactant and product must have the same multiplicity; got "
            f"{reactant.multiplicity} and {product.multiplicity}"
        )
    if isinstance(reactant, PeriodicSystem):
        if int(reactant.dim) != int(product.dim):
            raise ValueError(
                "reactant and product must have the same periodic dimension; "
                f"got dim={reactant.dim} and dim={product.dim}."
            )
        lr = np.asarray(reactant.lattice, dtype=float)
        lp = np.asarray(product.lattice, dtype=float)
        if not np.allclose(lr, lp):
            raise ValueError(
                "reactant and product must share the same lattice -- "
                "variable-cell NEB is out of scope. Fix the cell to the "
                "reactant's lattice before constructing endpoints."
            )


# ---------------------------------------------------------------------------
# Linear interpolation
# ---------------------------------------------------------------------------


def interpolate_linear(
    reactant: System,
    product: System,
    n_images: int,
) -> List[System]:
    """Linear Cartesian interpolation between two endpoints.

    Parameters
    ----------
    reactant, product
        Same-type Molecule or PeriodicSystem with matching atom
        ordering. For PeriodicSystem the lattices must agree.
    n_images
        Number of *intermediate* images. The returned list has length
        ``n_images + 2`` (endpoints included).

    Returns
    -------
    list[System]
        ``[reactant, img_1, ..., img_n, product]``. Endpoints are
        returned as the original objects (not copies).
    """
    if n_images < 0:
        raise ValueError(f"n_images must be >= 0; got {n_images}")
    _check_compatible(reactant, product)
    r0 = _positions_of(reactant)
    r1 = _positions_of(product)
    out: List[System] = [reactant]
    for k in range(1, n_images + 1):
        t = k / (n_images + 1)
        pos = (1.0 - t) * r0 + t * r1
        out.append(_rebuild_with_positions(reactant, pos))
    out.append(product)
    return out


# ---------------------------------------------------------------------------
# IDPP interpolation -- Smidstrup et al. 2014
# ---------------------------------------------------------------------------
#
# For each intermediate image i = 1..n_images we define a target
# pair-distance matrix
#
#     d^(i)_{jk} = (1 - t_i) * d^(R)_{jk} + t_i * d^(P)_{jk}
#
# (linear in pair-distance space between reactant and product). We
# then minimise the image-dependent pair-potential objective
#
#     S^(i)(R) = sum_{j<k}  (d^(i)_{jk} - r_{jk}(R))^2 / r_{jk}(R)^4
#
# over each image's Cartesian coordinates independently. The 1/r^4
# weighting drives images strongly to relieve close atom-atom
# contacts while still tracking the interpolated distance manifold.
# Analytic gradient is provided to L-BFGS-B for speed.


def _minimum_image_diff(
    diff: np.ndarray, lattice: np.ndarray, dim: int
) -> np.ndarray:
    """Wrap Cartesian displacements ``diff`` (..., 3) to the minimum image.

    The first ``dim`` columns of ``lattice`` are the periodic lattice
    vectors. The active lattice basis is Minkowski-reduced first, using the
    same reduction as the periodic CCM Wigner-Seitz geometry engine. A fixed
    neighbour shell around the reduced fractional coordinate then solves the
    closest-vector problem without depending on how strongly the input basis
    is sheared. Directions outside the periodic column span (e.g. the vacuum
    axis of a ``dim < 3`` slab) are left untouched.
    """
    displacement = np.asarray(diff, dtype=float)
    lattice_rows = np.asarray(lattice, dtype=float).T
    if dim == 3:
        from .periodic.ccm.wigner_seitz import _minkowski_reduce

        reduced_rows, _ = _minkowski_reduce(lattice_rows)
    else:
        from ase.geometry import minkowski_reduce

        pbc = np.arange(3) < dim
        reduced_rows, _ = minkowski_reduce(lattice_rows, pbc=pbc)
    periodic_vectors = np.asarray(reduced_rows, dtype=float)[:dim]
    flat = displacement.reshape(-1, 3)
    frac = flat @ np.linalg.pinv(periodic_vectors)
    centre = np.rint(frac).astype(np.int64)

    best = flat - centre @ periodic_vectors
    best_norm_sq = np.einsum("ij,ij->i", best, best)
    for neighbour_index in np.ndindex(*(5,) * dim):
        offset = np.asarray(neighbour_index, dtype=np.int64) - 2
        image = centre + offset
        candidate = flat - image @ periodic_vectors
        norm_sq = np.einsum("ij,ij->i", candidate, candidate)
        take = norm_sq < best_norm_sq
        best[take] = candidate[take]
        best_norm_sq[take] = norm_sq[take]

    return best.reshape(displacement.shape)


def _pair_distance_matrix(
    positions: np.ndarray,
    lattice: Optional[np.ndarray] = None,
    dim: int = 3,
) -> np.ndarray:
    diff = positions[:, None, :] - positions[None, :, :]
    if lattice is not None:
        diff = _minimum_image_diff(diff, lattice, dim)
    return np.sqrt(np.einsum("ijc,ijc->ij", diff, diff))


def _idpp_value_and_grad(
    flat: np.ndarray,
    target: np.ndarray,
    n_atoms: int,
    lattice: Optional[np.ndarray] = None,
    dim: int = 3,
) -> tuple[float, np.ndarray]:
    positions = flat.reshape(n_atoms, 3)
    diff = positions[:, None, :] - positions[None, :, :]
    if lattice is not None:
        # Minimum-image displacements so a pair interacting across the PBC
        # tracks its nearest image, not the in-cell Cartesian vector.
        # The selected lattice image is locally constant, so
        # d(mic diff)/dR = ddiff/dR. The analytic gradient below is therefore
        # unchanged apart from operating on the wrapped displacement.
        diff = _minimum_image_diff(diff, lattice, dim)
    r2 = np.einsum("ijc,ijc->ij", diff, diff)
    # Off-diagonal mask; we never read the diagonal because it's masked
    # out of every aggregation below.
    mask = ~np.eye(n_atoms, dtype=bool)
    # Replace diagonal r2=0 with 1.0 so divisions are finite -- masked
    # back to zero before any sum. Also floor off-diagonal r2 at a
    # small positive value so an L-BFGS-B step that briefly drives two
    # atoms onto each other produces a finite (very large, repulsive)
    # gradient instead of NaN -- the optimiser can then escape.
    _R_FLOOR2 = 1e-6  # bohr^2
    r2_off = np.maximum(r2, _R_FLOOR2)
    safe_r2 = np.where(mask, r2_off, 1.0)
    r = np.sqrt(safe_r2)
    inv_r2 = 1.0 / safe_r2
    inv_r4 = inv_r2 * inv_r2
    inv_r5 = inv_r4 / r
    delta = target - r  # (d - r)
    # Energy: 0.5 * sum over all (i,j) of (d-r)^2 / r^4 (symmetric ->
    # the 1/2 converts to the j<k sum).
    pair_energy = delta * delta * inv_r4
    energy = 0.5 * float(np.sum(np.where(mask, pair_energy, 0.0)))
    # d/dr [(d-r)^2 / r^4] = -2 (d-r) / r^4 - 4 (d-r)^2 / r^5
    dS_dr = -2.0 * delta * inv_r4 - 4.0 * delta * delta * inv_r5
    dS_dr = np.where(mask, dS_dr, 0.0)
    # Chain rule: dr_{jk}/dR_j = diff[j,k] / r[j,k], dr/dR_k = - of that.
    # grad[k] = sum_j dS_dr[k,j] / r[k,j] * diff[k,j]
    inv_r = np.where(mask, 1.0 / r, 0.0)
    factor = dS_dr * inv_r
    grad = np.einsum("ij,ijc->ic", factor, diff)
    return energy, grad.ravel()


def interpolate_idpp(
    reactant: System,
    product: System,
    n_images: int,
    *,
    max_iter: int = 1000,
    tol: float = 1e-5,
) -> List[System]:
    """Image-Dependent Pair Potential interpolation (Smidstrup 2014).

    Builds a linear-Cartesian starting path, then for each
    intermediate image independently minimises the IDPP objective
    ``S^(i)(R) = sum_{j<k} (d^(i)_{jk} - r_{jk}(R))^2 / r_{jk}(R)^4``
    with target distances interpolated linearly between reactant and
    product pair-distance matrices. Endpoints are returned unchanged.

    Cites: Smidstrup, Pedersen, Stokbro, Jónsson,
    J. Chem. Phys. 140, 214106 (2014). doi:10.1063/1.4878664.
    """
    if n_images < 0:
        raise ValueError(f"n_images must be >= 0; got {n_images}")
    _check_compatible(reactant, product)
    if n_images == 0:
        return [reactant, product]

    n_atoms = len(_positions_of(reactant))
    # Periodic IDPP uses minimum-image pair distances (lattice from the
    # reactant; _check_compatible has already enforced a shared cell).
    is_periodic = isinstance(reactant, PeriodicSystem)
    lattice = (
        np.asarray(reactant.lattice, dtype=float) if is_periodic else None
    )
    dim = int(reactant.dim) if is_periodic else 3
    d_R = _pair_distance_matrix(_positions_of(reactant), lattice, dim)
    d_P = _pair_distance_matrix(_positions_of(product), lattice, dim)

    linear_path = interpolate_linear(reactant, product, n_images)
    out: List[System] = [reactant]
    for k in range(1, n_images + 1):
        t = k / (n_images + 1)
        target = (1.0 - t) * d_R + t * d_P
        x0 = _positions_of(linear_path[k]).ravel()
        res = minimize(
            _idpp_value_and_grad,
            x0,
            args=(target, n_atoms, lattice, dim),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter, "gtol": tol, "ftol": tol},
        )
        out.append(
            _rebuild_with_positions(reactant, res.x.reshape(n_atoms, 3))
        )
    out.append(product)
    return out


# ===========================================================================
# Driver (Increment 2) -- improved-tangent NEB, molecular only.
# ===========================================================================
#
# The driver below implements the textbook formulation of NEB:
#
#   F_total_i = F_spring_∥_i  +  F_true_⊥_i
#
# with the improved-tangent estimator of Henkelman & Jónsson 2000
# replacing the original "central-difference" tangent of Mills/
# Jónsson/Schenter 1995 -- energy ordering of neighbours decides
# whether t_i points uphill or downhill, with a transitional weighted
# mix when the central image is the local extremum (eqs. 8-11 of JCP
# 113, 9978, 2000). Spring forces are projected onto t_i; true
# nuclear gradients are projected *off* t_i. The result is a
# discretised elastic band that relaxes to the minimum energy path.
#
# Outer loop: damped quick-min (also Henkelman+Jónsson). Each image
# carries a velocity; per step it is projected onto F_total before
# advancing -- the projection keeps the band from drifting away from
# the MEP when v has accumulated tangential momentum. L-BFGS-B on
# the concatenated coordinate vector would *also* work, but only if
# the NEB force were a true gradient. It isn't: F_spring_∥ depends on
# the parallel projection of the difference vector, which is not the
# derivative of a scalar potential. Quick-min is the standard choice
# in ASE / VASP / Quantum ESPRESSO for the same reason.
#
# Parallelism: joblib.Parallel over images per outer iteration.
# Endpoints are evaluated once and cached. n_jobs=0 selects a bounded
# auto default; pass n_jobs=-1 for joblib's all-core behavior or
# n_jobs=1 for serial.
#
# Shipped on top of the Increment-2 core below:
#   * Climbing image ("climbing_image=True").
#   * Periodic dispatch (PeriodicSystem endpoints; per-image BIPOLE
#     SCF + finite-difference gradient, k-mesh via "kpoints=").
#   * SCF density warm-start across outer iterations ("warm_start=True",
#     six molecular mean-field methods; four periodic methods).
#   * DFT+U ("dft_plus_u=[HubbardSite(...)]", molecular + periodic).
#
# References:
#   Henkelman, Jónsson, J. Chem. Phys. 113, 9978 (2000).
#   doi:10.1063/1.1323224.


def _positive_int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _resolve_neb_n_jobs(
    n_jobs: int,
    *,
    n_images: int,
    is_periodic: bool,
) -> int:
    """Resolve the process count for per-image NEB evaluation."""
    if n_jobs != 0:
        return n_jobs

    default_cap = 1 if is_periodic else 4
    cap = _positive_int_env("VIBEQC_NEB_MAX_JOBS") or default_cap
    cpu_count = os.cpu_count() or 1
    return max(1, min(n_images, cpu_count, cap))


@dataclass
class NEBResult:
    """Outcome of a :func:`run_neb` run.

    Attributes
    ----------
    path
        The converged (or last-evaluated) :class:`NEBPath` -- endpoints
        plus intermediate images with their final geometry, energy,
        gradient, and tangent.
    energies
        Per-image energy array (Hartree), length ``n_images + 2``.
        Endpoints are at indices 0 and -1.
    converged
        True iff the max-norm of the NEB force fell below
        ``conv_tol_force`` within ``max_iter`` outer iterations.
    transition_state_index
        Index into ``path.images`` of the highest-energy image -- the
        best non-climbing estimate of the saddle. ``None`` if energies
        are unset.
    n_iter
        Number of completed outer iterations.
    max_force
        Final maximum-norm NEB force (Ha/bohr) over intermediate
        images. Compare against ``conv_tol_force`` to gauge how close
        to converged a non-converged run was.
    """

    path: NEBPath
    energies: np.ndarray
    converged: bool
    transition_state_index: Optional[int]
    n_iter: int
    max_force: float
    # Captured at run_neb call time so write_qvf can build the
    # citation surface + manifest provenance without the user
    # having to repeat the SCF flavour. These can be None / default if
    # NEBResult is constructed by hand (e.g. in tests).
    method: Optional[str] = None
    basis: Optional[str] = None
    functional: Optional[str] = None
    is_periodic: bool = False
    # Truthy => write_qvf fires the ``routes.methods.dft_plus_u``
    # citation route (Dudarev 1998 + Cococcioni-Gironcoli 2005).
    # We store a bool rather than the HubbardSite list because
    # ``write_qvf`` only needs the on/off bit for citation
    # assembly; the actual sites have already done their job in
    # each per-image SCF.
    used_dft_plus_u: bool = False
    # For ``method="mace"``: the per-model foundation-model citation key
    # (e.g. ``batatia_mace_mp_2024``) so write_qvf can fire the MACE
    # references (the method paper via the static ``routes.methods.mace``
    # route, this key as an extra entry). Empty for the SCF methods.
    mace_model_citation: Optional[str] = None
    # Citation/manifest route for a semiempirical Hamiltonian whose public
    # result method name alone does not encode its boundary.
    semiempirical_route_method: Optional[str] = None
    # Preserve a method variant that changes the Hamiltonian or citation
    # surface even when ``method`` is the canonical dispatcher key.
    semiempirical_variant: Optional[str] = None
    semiempirical_citation_entries: tuple[str, ...] = ()

    def write_qvf(
        self,
        stem: Any,
        *,
        compression: Optional[int] = None,
    ) -> Any:
        """Emit a vibe-view ``reaction.path`` QVF archive.

        Builds an ``OutputPlan`` + context dict from this result and
        delegates to :func:`vibeqc.output.formats.qvf.write_qvf`. The
        archive contains a ``structure`` section (reactant geometry),
        a ``reaction.path`` section (every image's coords + energies
        + waypoints), and a ``citations`` section (BibTeX assembled
        with ``uses_neb=True`` -- plus ``uses_ci_neb=True`` when this
        result came from a climbing-image run).

        For periodic NEB results the archive ships as QVF v2 -- the
        ``reaction.path`` section additionally carries the per-frame
        lattice + dim (see ``docs/user_guide/vibe_view.md`` Sec.
        "Periodic reaction paths"). The writer detects periodic
        frames automatically.

        Parameters
        ----------
        stem
            Path stem; the writer appends ``.qvf``.
        compression
            Optional ``zipfile`` compression constant; if ``None``
            the writer uses its default (``ZIP_DEFLATED`` or
            ``ZIP_ZSTANDARD`` when ``zipfile-zstd`` is installed).

        Returns
        -------
        pathlib.Path
            The written archive path.
        """
        from .output.formats.qvf import write_reaction_path_qvf

        n_total = len(self.path.images)

        # Waypoints: reactant, product, TS (climbing image when
        # available, else the highest-energy intermediate).
        waypoints: list[dict[str, Any]] = [
            {
                "frame_index": 0,
                "label": "reactant",
                "kind": "reactant",
                "energy_eh": float(self.energies[0]),
            },
            {
                "frame_index": n_total - 1,
                "label": "product",
                "kind": "product",
                "energy_eh": float(self.energies[-1]),
            },
        ]
        ts_idx = (
            self.path.climbing_image_index
            if self.path.climbing_image_index is not None
            else self.transition_state_index
        )
        if ts_idx is not None and 0 < ts_idx < n_total - 1:
            waypoints.append(
                {
                    "frame_index": int(ts_idx),
                    "label": "TS",
                    "kind": "transition_state",
                    "energy_eh": float(self.energies[ts_idx]),
                }
            )

        # Reaction coordinate: cumulative arc length over frame
        # geometries, normalised to [0, 1].
        positions = [
            _positions_of(img.system) for img in self.path.images
        ]
        arc = [0.0]
        for i in range(1, n_total):
            arc.append(
                arc[-1]
                + float(np.linalg.norm(positions[i] - positions[i - 1]))
            )
        rc = (
            [a / arc[-1] for a in arc]
            if arc[-1] > 0.0
            else [0.0] * n_total
        )

        from .semiempirical.routes import (
            is_semiempirical_method,
            normalise_semiempirical_method,
        )

        method_name = self.method or ""
        is_mace = method_name.lower() == "mace"
        is_semiempirical = is_semiempirical_method(method_name)
        semiempirical_method = (
            normalise_semiempirical_method(method_name)
            if is_semiempirical
            else None
        )
        extra: dict[str, Any] = {
            "uses_neb": True,
            "uses_ci_neb": self.path.climbing_image_index is not None,
            "dft_plus_u": bool(self.used_dft_plus_u),
        }
        if is_mace:
            # MACE evaluates no Gaussian integrals and runs no SCF -- suppress
            # the always-on libint + DIIS routes so the references reflect
            # what ran. The MACE *method* paper fires via routes.methods.mace
            # (method="mace"); the per-model foundation paper is an extra.
            extra["uses_integrals"] = False
            extra["uses_scf"] = False
            # A periodic MACE frame has a cell, but it does not execute the
            # crystalline-orbital LCAO route or spglib-backed symmetry work.
            extra["periodic"] = False
            if self.mace_model_citation:
                extra["extra_entries"] = [self.mace_model_citation]
        elif is_semiempirical:
            # Only DFTB consumes the shared Gaussian-overlap machinery. Other
            # semiempirical families have no libint route, and DFTB0 has no
            # self-consistent density iteration. Explicitly suppress the
            # generic defaults so the archive cites only what actually ran.
            extra["uses_integrals"] = semiempirical_method in {
                "dftb0",
                "scc_dftb",
            }
            extra["uses_scf"] = semiempirical_method != "dftb0"
            extra["basis"] = None
            if semiempirical_method == "ccm":
                # The SECCM route identifies the cyclic-boundary theory; the
                # current adapter is MSINDO and must carry its method papers.
                extra["extra_entries"] = [
                    "ahlswede_jug_msindo_1_1999",
                    "ahlswede_jug_msindo_2_1999",
                ]
                # SECCM has a periodic cell but no Bloch LCAO or spglib route;
                # its cyclic-boundary papers are selected by method="seccm".
                extra["periodic"] = False

            if self.semiempirical_citation_entries:
                existing_entries = list(extra.get("extra_entries", ()))
                extra["extra_entries"] = list(
                    dict.fromkeys(
                        [
                            *existing_entries,
                            *self.semiempirical_citation_entries,
                        ]
                    )
                )

        is_msindo_nddo = bool(
            semiempirical_method == "msindo"
            and self.semiempirical_variant == "nddo"
        )
        if is_msindo_nddo:
            # The archive names the actual NDDO Hamiltonian, while citation
            # assembly uses the MSINDO base-method route plus the variant's
            # Dewar--Thiel entry captured from the route plan.
            extra["method"] = "msindo"

        qvf_method = (
            "msindo-nddo"
            if is_msindo_nddo
            else self.semiempirical_route_method or self.method or "RHF"
        )
        return write_reaction_path_qvf(
            stem,
            frames=[img.system for img in self.path.images],
            energies=[float(e) for e in self.energies],
            waypoints=waypoints,
            reaction_coordinate=rc,
            method=qvf_method.upper(),
            # MACE / semiempirical methods have no user-selected Gaussian
            # basis. This is a manifest placeholder; ``extra['basis']=None``
            # suppresses the citation route for semiempirical calculations.
            basis=(
                "mace" if is_mace
                else semiempirical_method if is_semiempirical
                else (self.basis or "sto-3g")
            ),
            # The run_neb default functional="pbe" must not leak into MACE or
            # semiempirical provenance.
            functional=(
                None if (is_mace or is_semiempirical) else self.functional
            ),
            extra_assemble_kwargs=extra,
            compression=compression,
        )


# --- improved tangent (Henkelman + Jónsson 2000) ---------------------------


def _improved_tangent(
    R_prev: np.ndarray,
    R_curr: np.ndarray,
    R_next: np.ndarray,
    E_prev: float,
    E_curr: float,
    E_next: float,
    lattice: Optional[np.ndarray] = None,
    dim: int = 3,
) -> np.ndarray:
    """Improved tangent t_i per Henkelman+Jónsson 2000 eq. 8-11.

    Returns a unit-norm Cartesian vector with the same shape as the
    input positions. Geometry-only fallback (no energy data) is the
    central-difference tangent ``(R_next - R_curr) + (R_curr - R_prev)``.
    For periodic systems (``lattice`` given) the two inter-image
    half-steps use the minimum-image convention, so a band whose images
    straddle a cell boundary still gets a short, sensible tangent
    instead of one that points the long way across the cell.
    """
    tau_plus = R_next - R_curr
    tau_minus = R_curr - R_prev
    if lattice is not None:
        tau_plus = _minimum_image_diff(tau_plus, lattice, dim)
        tau_minus = _minimum_image_diff(tau_minus, lattice, dim)
    if E_next > E_curr > E_prev:
        tau = tau_plus
    elif E_next < E_curr < E_prev:
        tau = tau_minus
    else:
        dE_max = max(abs(E_next - E_curr), abs(E_prev - E_curr))
        dE_min = min(abs(E_next - E_curr), abs(E_prev - E_curr))
        if E_next > E_prev:
            tau = tau_plus * dE_max + tau_minus * dE_min
        else:
            tau = tau_plus * dE_min + tau_minus * dE_max
    norm = float(np.linalg.norm(tau))
    if norm < 1e-12:
        # Degenerate band (all three images coincident); fall back to
        # the central-difference tangent (minimum-image half-steps, so
        # equal to R_next - R_prev for the molecular case). If that's
        # also zero, return zero -- the outer loop is at a fixed point.
        tau = tau_plus + tau_minus
        norm = float(np.linalg.norm(tau))
        if norm < 1e-12:
            return np.zeros_like(tau)
    return tau / norm


# --- per-image SCF + gradient (the worker called inside joblib) ------------


def _nuclear_repulsion_molecular(mol: Molecule) -> float:
    """Sum Z_i Z_j / r_ij for a molecule (bohr in, Ha out).

    The high-level SCF entry points compute this internally; the
    low-level ``run_*_scf_with_jk`` path wants ``E_nuc`` as a scalar
    so the NEB driver's warm-start helper computes it here.
    """
    atoms = list(mol.atoms)
    n = len(atoms)
    e = 0.0
    for i in range(n):
        zi = int(atoms[i].Z)
        xi = np.asarray(atoms[i].xyz, dtype=float)
        for j in range(i + 1, n):
            zj = int(atoms[j].Z)
            xj = np.asarray(atoms[j].xyz, dtype=float)
            r = float(np.linalg.norm(xi - xj))
            if r > 0:
                e += zi * zj / r
    return e


def _build_scf_common_pieces(
    mol: Molecule,
    basis: Any,
) -> tuple[np.ndarray, np.ndarray, float, Any]:
    """Shared S / Hcore / E_nuc / JKBuilder construction for the
    warm-start path. All four methods (RHF / UHF / RKS / UKS) need
    the same building blocks before calling their low-level
    ``run_*_scf_with_jk`` entry point.
    """
    from ._vibeqc_core import (
        compute_kinetic,
        compute_nuclear,
        compute_overlap,
        make_direct_jk_builder,
    )

    S = compute_overlap(basis)
    Hcore = compute_kinetic(basis) + compute_nuclear(basis, mol)
    e_nuc = _nuclear_repulsion_molecular(mol)
    jk = make_direct_jk_builder(basis)
    return S, Hcore, e_nuc, jk


def _empty_density() -> np.ndarray:
    return np.zeros((0, 0), dtype=np.float64)


def _sad_cold_start_closed(mol: Molecule, basis: Any) -> np.ndarray:
    """Build an explicit SAD density; construction failures propagate."""
    from .guess import initial_density_closed_shell
    from . import InitialGuess, compute_overlap

    return initial_density_closed_shell(
        mol, basis, mol.n_electrons() // 2, InitialGuess.SAD, is_periodic=False,
        overlap=np.asarray(compute_overlap(basis)),
    )


def _sad_cold_start_open(
    mol: Molecule, basis: Any, n_alpha: int, n_beta: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build explicit per-spin SAD densities with exact metric populations."""
    from .guess import initial_densities_open_shell
    from . import InitialGuess, compute_overlap

    return initial_densities_open_shell(
        mol, basis, n_alpha, n_beta, InitialGuess.SAD, is_periodic=False,
        overlap=np.asarray(compute_overlap(basis)),
    )


def _molecular_cold_start_closed(
    mol: Molecule,
    basis: Any,
    options: Any,
    S: np.ndarray,
    Hcore: np.ndarray,
    jk: Any,
) -> np.ndarray:
    """Build the cold density requested by a molecular SCF options object."""
    from ._vibeqc_core import (
        InitialGuess,
        _guess_closed_shell_density_with_jk,
    )

    kind = options.initial_guess
    if kind in (InitialGuess.READ, InitialGuess.FRAGMO):
        density = np.asarray(options.read_density, dtype=float)
        nbf = int(basis.nbasis)
        if density.shape != (nbf, nbf):
            raise RuntimeError(
                "run_neb: molecular READ/FRAGMO requires a complete "
                "options.read_density matching basis.nbasis()."
            )
        return density.copy()
    guess = _guess_closed_shell_density_with_jk(
        mol,
        basis,
        mol.n_electrons() // 2,
        kind,
        S,
        Hcore,
        jk,
        float(getattr(options, "linear_dep_threshold", 1.0e-7)),
    )
    if guess is None:
        return _empty_density()
    return np.asarray(guess, dtype=float)


def _molecular_cold_start_open(
    mol: Molecule,
    basis: Any,
    options: Any,
    n_alpha: int,
    n_beta: int,
    S: np.ndarray,
    Hcore: np.ndarray,
    jk: Any,
    *,
    use_jk_for_density_mode: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the requested molecular open-shell cold density."""
    from ._vibeqc_core import (
        InitialGuess,
        _guess_open_shell_density_with_jk,
    )

    kind = options.initial_guess
    if kind in (InitialGuess.READ, InitialGuess.FRAGMO):
        d_alpha = np.asarray(options.read_density_alpha, dtype=float)
        d_beta = np.asarray(options.read_density_beta, dtype=float)
        nbf = int(basis.nbasis)
        expected = (nbf, nbf)
        if d_alpha.shape != expected or d_beta.shape != expected:
            raise RuntimeError(
                "run_neb: molecular READ/FRAGMO requires complete "
                "options.read_density_alpha/read_density_beta matrices "
                "matching basis.nbasis()."
            )
        return d_alpha.copy(), d_beta.copy()
    guess = _guess_open_shell_density_with_jk(
        mol,
        basis,
        n_alpha,
        n_beta,
        kind,
        S,
        Hcore,
        jk,
        use_jk_for_density_mode,
        list(getattr(options, "atomic_spins", []) or []),
        float(getattr(options, "linear_dep_threshold", 1.0e-7)),
    )
    if guess is None:
        return _empty_density(), _empty_density()
    d_alpha, d_beta = guess
    return np.asarray(d_alpha, dtype=float), np.asarray(d_beta, dtype=float)


def _molecular_warm_selection(mol, opts, initial_density, *, open_shell):
    from .guess import GuessSelection, select_initial_guess
    from . import InitialGuess

    selection = select_initial_guess(
        mol, opts.initial_guess, is_open_shell=open_shell,
        atomic_spins=getattr(opts, "atomic_spins", None),
    )
    if initial_density is None:
        return selection
    return GuessSelection(selection.requested, selection.effective, InitialGuess.READ)



def _run_restricted_open_image(
    driver, mol, basis, options, initial_density, *, source_selection=None,
    **kwargs,
):
    """Execute an RO image while preserving construction across warm transport."""
    from .guess import GuessSelection
    from . import InitialGuess

    selection = _molecular_warm_selection(
        mol, options, initial_density, open_shell=True,
    )
    tags = getattr(options, "atomic_spins", None)
    try:
        if initial_density is not None and tags:
            options.atomic_spins = []
        result = driver(
            mol, basis, options, initial_density=initial_density, **kwargs,
        )
    finally:
        if initial_density is not None and tags:
            options.atomic_spins = tags
    if initial_density is not None:
        physical = source_selection or selection
        result.guess_selection = GuessSelection(
            physical.requested, physical.effective, InitialGuess.READ,
        )
    return result


def _run_rhf_warm_start(
    mol: Molecule,
    basis: Any,
    options: Any,
    initial_density: Optional[np.ndarray],
) -> Any:
    """RHF SCF via the low-level ``run_rhf_scf_with_jk`` entry point.

    Used by :func:`_evaluate_image` to seed the SCF from a previous
    outer-iteration's converged density (within-image warm-start --
    the NEB warm-start milestone). The high-level ``run_rhf`` doesn't
    expose ``initial_density``, so the NEB driver routes RHF through
    the lower-level entry that does.

    When ``initial_density is None`` the native engine constructs the
    selected guess using the same molecule and JK operator as ``run_rhf``. The cost difference vs. a cold
    ``run_rhf`` is the Python-side construction of S/Hcore/E_nuc/JK
    -- a single-pass set of compute_* calls, much cheaper than the
    SCF iterations themselves.
    """
    from ._vibeqc_core import RHFOptions, run_rhf_scf_with_jk

    opts = options if options is not None else RHFOptions()
    S, Hcore, e_nuc, jk = _build_scf_common_pieces(mol, basis)
    init = initial_density if initial_density is not None else _empty_density()
    return run_rhf_scf_with_jk(
        basis,
        mol.n_electrons(),
        S,
        Hcore,
        e_nuc,
        jk,
        opts,
        initial_density=init,
        molecule=mol,
        guess_selection=_molecular_warm_selection(
            mol, opts, initial_density, open_shell=False,
        ),
    )


class _VerifiedAllElectronSCFResult:
    """Read-only provenance view for NEB's internally built SCF Hcore.

    The public low-level ``run_*_scf_with_jk`` boundary accepts an arbitrary
    caller-supplied one-electron matrix and therefore correctly returns
    unverified ECP provenance.  NEB's warm-start helpers are a narrower
    internal boundary: they construct ``T + V_ne`` with bare nuclear charges
    and the molecule's full electron count themselves.  Record that exact
    all-electron fact for the public gradient contract without mutating or
    self-certifying the native low-level result.
    """

    __slots__ = (
        "_vibeqc_native_result",
        "ecp_operator_applied",
        "ecp_provenance_verified",
        "ecp_xml_centers",
        "ecp_xml_library",
        "ecp_primitive_blocks",
        "ecp_primitive_centers",
        "ecp_effective_charges",
        "ecp_total_ncore",
    )

    def __init__(self, native_result: Any) -> None:
        object.__setattr__(self, "_vibeqc_native_result", native_result)
        object.__setattr__(self, "ecp_operator_applied", False)
        object.__setattr__(self, "ecp_provenance_verified", True)
        object.__setattr__(self, "ecp_xml_centers", ())
        object.__setattr__(self, "ecp_xml_library", "")
        object.__setattr__(self, "ecp_primitive_blocks", ())
        object.__setattr__(self, "ecp_primitive_centers", ())
        object.__setattr__(self, "ecp_effective_charges", ())
        object.__setattr__(self, "ecp_total_ncore", 0)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._vibeqc_native_result, name)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"_VerifiedAllElectronSCFResult is read-only; cannot set {name!r}"
        )


def _run_uhf_warm_start(
    mol: Molecule,
    basis: Any,
    options: Any,
    initial_density: Optional[tuple[np.ndarray, np.ndarray]],
) -> Any:
    """UHF SCF via ``run_uhf_scf_with_jk``. Density cache is the
    ``(alpha, beta)`` tuple of per-spin density matrices."""
    from ._vibeqc_core import UHFOptions, run_uhf_scf_with_jk

    opts = options if options is not None else UHFOptions()
    S, Hcore, e_nuc, jk = _build_scf_common_pieces(mol, basis)
    n_total = mol.n_electrons()
    mult = mol.multiplicity
    # Standard alpha/beta partition: n_alpha = (N + 2S) / 2,
    # n_beta = N - n_alpha. Matches the molecular runner's
    # convention.
    n_alpha = (n_total + (mult - 1)) // 2
    n_beta = n_total - n_alpha
    init_a, init_b = (
        initial_density if initial_density is not None
        else (_empty_density(), _empty_density())
    )
    return run_uhf_scf_with_jk(
        basis,
        n_alpha,
        n_beta,
        S,
        Hcore,
        e_nuc,
        jk,
        opts,
        init_alpha=init_a,
        init_beta=init_b,
        molecule=mol,
        guess_selection=_molecular_warm_selection(
            mol, opts, initial_density, open_shell=True,
        ),
    )


def _resolve_xc_grid(
    mol: Molecule,
    grid_options: Any,
) -> Any:
    """Build (or pass through) the XC integration grid for KS-DFT."""
    from ._vibeqc_core import GridOptions, build_grid

    gopt = grid_options if grid_options is not None else GridOptions()
    return build_grid(mol, gopt)


def _run_rks_warm_start(
    mol: Molecule,
    basis: Any,
    options: Any,
    functional: Optional[str],
    grid_options: Any,
    initial_density: Optional[np.ndarray],
) -> Any:
    """RKS SCF via ``run_rks_scf_with_jk``. Same closed-shell density
    convention as RHF; additionally needs an XC integration grid."""
    from ._vibeqc_core import RKSOptions, run_rks_scf_with_jk

    opts = options if options is not None else RKSOptions()
    if functional is not None:
        opts.functional = functional
    S, Hcore, e_nuc, jk = _build_scf_common_pieces(mol, basis)
    grid = _resolve_xc_grid(mol, grid_options)
    init = initial_density if initial_density is not None else _empty_density()
    return run_rks_scf_with_jk(
        basis,
        mol.n_electrons(),
        S,
        Hcore,
        e_nuc,
        jk,
        grid,
        opts,
        initial_density=init,
        molecule=mol,
        guess_selection=_molecular_warm_selection(
            mol, opts, initial_density, open_shell=False,
        ),
    )


def _run_uks_warm_start(
    mol: Molecule,
    basis: Any,
    options: Any,
    functional: Optional[str],
    grid_options: Any,
    initial_density: Optional[tuple[np.ndarray, np.ndarray]],
) -> Any:
    """UKS SCF via ``run_uks_scf_with_jk``. Open-shell a/b densities +
    XC grid."""
    from ._vibeqc_core import UKSOptions, run_uks_scf_with_jk

    # Each image is a deliberately warm-started state on one continuous
    # reaction path. Analyse its stationary point, but never let the generic
    # wrong-root follower jump that image independently to another basin.
    # ``stability_max_retries=0`` keeps the verdict while disabling escape.
    opts = UKSOptions(options) if options is not None else UKSOptions()
    opts.stability_max_retries = 0
    if functional is not None:
        opts.functional = functional
    S, Hcore, e_nuc, jk = _build_scf_common_pieces(mol, basis)
    grid = _resolve_xc_grid(mol, grid_options)
    n_total = mol.n_electrons()
    mult = mol.multiplicity
    n_alpha = (n_total + (mult - 1)) // 2
    n_beta = n_total - n_alpha
    init_a, init_b = (
        initial_density if initial_density is not None
        else (_empty_density(), _empty_density())
    )
    return run_uks_scf_with_jk(
        basis,
        n_alpha,
        n_beta,
        S,
        Hcore,
        e_nuc,
        jk,
        grid,
        opts,
        init_alpha=init_a,
        init_beta=init_b,
        molecule=mol,
        guess_selection=_molecular_warm_selection(
            mol, opts, initial_density, open_shell=True,
        ),
    )


def _evaluate_image(
    positions: np.ndarray,
    template: Molecule,
    basis_name: str,
    method: str,
    *,
    functional: Optional[str],
    rhf_options: Any,
    uhf_options: Any,
    rks_options: Any,
    uks_options: Any,
    gradient_options: Any,
    grid_options: Any,
    dispersion_params: Any,
    rohf_options: Any = None,
    roks_options: Any = None,
    grid_level: str = "orca-defgrid3",
    fd_step_bohr: float = 1e-3,
    initial_density: Optional[np.ndarray] = None,
    image_index: Optional[int] = None,
    dft_plus_u: Optional[Sequence[Any]] = None,
) -> tuple[float, np.ndarray, Optional[np.ndarray]]:
    """Run SCF + gradient at one geometry.

    Returns ``(energy, gradient, converged_density)``. Energies in Ha;
    gradients in Ha/bohr, shape (n_atoms, 3). ``converged_density`` is
    the converged SCF result carried back to the outer loop for within-image
    warm starts. Its source basis permits projection after a geometry step;
    raw density inputs remain an explicit already-in-target-basis seam. RHF/UHF/RKS/UKS route through the
    low-level ``run_*_scf_with_jk`` entry points that accept an
    external initial density. ROHF uses its native Python warm-start seam;
    ROKS uses the same seam for the reference state and central differences
    of the ROKS energy because an analytic molecular ROKS gradient is not yet
    available.

    The molecule is reconstructed from ``positions`` (bohr) using
    ``template`` for atomic numbers + charge + multiplicity. The basis
    set is rebuilt per geometry -- vibe-qc's BasisSet is bound to the
    nuclei it was constructed with.

    When ``initial_density`` is provided the SCF starts from that
    density (or ``(alpha, beta)`` pair for open-shell) instead of
    SAD/Hcore -- within-image density warm-start across NEB outer
    iterations.

    ECP systems (#643): the ECP centres on the RHF/UHF/RKS/UKS options are
    absolute coordinates built for ``template``; they are moved onto this
    image's atoms for the duration of the call and restored afterwards
    (:func:`vibeqc.ecp_metadata.ecp_centres_follow_atoms`), the SCF runs
    through the high-level ECP-aware drivers instead of the all-electron
    warm-start seam (no density warm-start on that branch), and a ``None``
    ``gradient_options`` is replaced by the mirror of the SCF options.
    """
    from ._vibeqc_core import BasisSet
    from .molecular_optimize import _compute_molecular_gradient, _run_molecular_scf

    mol = _rebuild_with_positions(template, positions)
    method_lower = method.lower()
    basis = BasisSet(mol, basis_name)

    # Resolve the KS options and integration grid once per image.  When the
    # caller does not provide a top-level grid override, the grid configured
    # on the selected RKS/UKS options is the molecular-SCF contract.  Pass the
    # same object to both the warm-start SCF and analytic gradient so the two
    # evaluate one numerical surface.
    from .runner import apply_ks_grid_default
    from ._vibeqc_core import RKSOptions, UKSOptions
    from .roks import ROKSOptions

    effective_grid_options = grid_options
    if method_lower == "rks":
        rks_options = rks_options or RKSOptions()
        ks_options = rks_options
    elif method_lower == "uks":
        uks_options = uks_options or UKSOptions()
        ks_options = uks_options
    elif method_lower == "roks":
        roks_options = roks_options or ROKSOptions()
        ks_options = roks_options
    else:
        ks_options = None
    if ks_options is not None:
        if grid_options is not None:
            ks_options.grid = grid_options
        else:
            apply_ks_grid_default(ks_options, grid_level)
        if effective_grid_options is None:
            effective_grid_options = ks_options.grid

    # The density cache slot is one of:
    #   - None (no warm-start density available)
    #   - np.ndarray for closed-shell (RHF / RKS)
    #   - (alpha, beta) tuple of np.ndarrays for open-shell (UHF / UKS)
    # _evaluate_image returns the cache in whichever shape matches
    # the method, and the outer loop carries it back per image.
    # #643: both ECP routes pin the potential to absolute coordinates on the
    # SCF options (ecp_centers[i].xyz / ecp_primitive_centers[i]); the
    # template is the geometry they were built for. Move them onto this
    # image's atoms for the SCF + gradient below and hand the caller's
    # options back unchanged afterwards. When the caller passed no
    # gradient_options, an ECP image differentiates the mirror of the SCF
    # options (JK backend + ECP fields, vibeqc.gradient_options); an
    # explicit object has its ECP fields synchronised inside the block.
    from .ecp_metadata import ecp_centres_follow_atoms, options_carry_ecp
    from .gradient_options import gradient_options_from_scf

    _image_scf_opts = {
        "rhf": rhf_options,
        "uhf": uhf_options,
        "rks": rks_options,
        "uks": uks_options,
        "rohf": rohf_options,
        "roks": roks_options,
    }.get(method_lower)
    source_selection = getattr(initial_density, "guess_selection", None)
    _ecp_active = options_carry_ecp(_image_scf_opts)
    with ecp_centres_follow_atoms(
        _image_scf_opts, template, mol, gradient_options
    ):
        if initial_density is not None and getattr(initial_density, "restart_basis", None) is not None:
            # The cache owns the source AO basis. Equal matrix dimensions do
            # not imply equal basis functions after an image moves.
            from .guess_read import resolve_read_density_closed, resolve_read_densities_open
            from types import SimpleNamespace

            read_opts = _image_scf_opts or SimpleNamespace()
            if method_lower in ("rhf", "rks"):
                initial_density = resolve_read_density_closed(
                    read_opts, mol, basis, initial_density,
                )
            else:
                initial_density = resolve_read_densities_open(
                    read_opts, mol, basis, initial_density,
                )
        _image_gradient_options = gradient_options
        if _image_gradient_options is None and _ecp_active:
            _image_gradient_options = gradient_options_from_scf(_image_scf_opts)
        converged_density: Optional[Any] = None
        if _ecp_active and method_lower in ("rhf", "uhf", "rks", "uks"):
            # ECP image (#643): the warm-start path below builds an all-electron
            # Hcore (T + V_ne with bare Z, full electron count) and never reads
            # the ECP fields, so it cannot describe this Hamiltonian. Route the
            # image through the high-level drivers, which build Z_eff nuclear
            # attraction + V_ECP and fill valence electrons only -- the same
            # SCF the runner's single points run. The options' centres have
            # been moved onto this image's atoms by the enclosing context.
            if grid_options is not None and method_lower in ("rks", "uks"):
                raise NotImplementedError(
                    "ECP image evaluation runs the SCF through the high-level "
                    "KS driver, whose grid is the one on the RKS/UKS options; a "
                    "separate top-level grid_options override is not supported "
                    "on ECP systems. Set the grid on rks_options / uks_options."
                )
            energy, scf_result = _run_molecular_scf(
                mol,
                basis,
                method_lower,
                functional=functional,
                rhf_options=rhf_options,
                uhf_options=uhf_options,
                rks_options=rks_options,
                uks_options=uks_options,
                grid_level="legacy" if grid_options is not None else grid_level,
            )
            energy = float(energy)
            if method_lower in ("rhf", "rks"):
                converged_density = np.asarray(
                    scf_result.density, dtype=np.float64
                ).copy()
            else:
                converged_density = (
                    np.asarray(scf_result.density_alpha, dtype=np.float64).copy(),
                    np.asarray(scf_result.density_beta, dtype=np.float64).copy(),
                )
        elif method_lower == "rhf":
            scf_result = _VerifiedAllElectronSCFResult(
                _run_rhf_warm_start(
                    mol, basis, rhf_options, initial_density
                )
            )
            energy = float(scf_result.energy)
            converged_density = np.asarray(
                scf_result.density, dtype=np.float64
            ).copy()
        elif method_lower == "uhf":
            scf_result = _VerifiedAllElectronSCFResult(
                _run_uhf_warm_start(
                    mol, basis, uhf_options, initial_density
                )
            )
            energy = float(scf_result.energy)
            converged_density = (
                np.asarray(scf_result.density_alpha, dtype=np.float64).copy(),
                np.asarray(scf_result.density_beta, dtype=np.float64).copy(),
            )
        elif method_lower == "rks":
            scf_result = _VerifiedAllElectronSCFResult(
                _run_rks_warm_start(
                    mol,
                    basis,
                    rks_options,
                    functional,
                    effective_grid_options,
                    initial_density,
                )
            )
            energy = float(scf_result.energy)
            converged_density = np.asarray(
                scf_result.density, dtype=np.float64
            ).copy()
        elif method_lower == "uks":
            scf_result = _VerifiedAllElectronSCFResult(
                _run_uks_warm_start(
                    mol,
                    basis,
                    uks_options,
                    functional,
                    effective_grid_options,
                    initial_density,
                )
            )
            energy = float(scf_result.energy)
            converged_density = (
                np.asarray(scf_result.density_alpha, dtype=np.float64).copy(),
                np.asarray(scf_result.density_beta, dtype=np.float64).copy(),
            )
        elif method_lower == "rohf":
            from .rohf import ROHFOptions, run_rohf

            opts = rohf_options if rohf_options is not None else ROHFOptions()
            scf_result = _run_restricted_open_image(
                run_rohf, mol, basis, opts, initial_density,
                source_selection=source_selection,
            )
            energy = float(scf_result.energy)
            converged_density = (
                np.asarray(scf_result.density_alpha, dtype=np.float64).copy(),
                np.asarray(scf_result.density_beta, dtype=np.float64).copy(),
            )
        elif method_lower == "roks":
            from .roks import ROKSOptions, run_roks

            opts = roks_options if roks_options is not None else ROKSOptions()
            if grid_options is not None and opts.grid is None:
                opts.grid = grid_options
            scf_result = _run_restricted_open_image(
                run_roks, mol, basis, opts, initial_density,
                source_selection=source_selection, functional=functional,
            )
            energy = float(scf_result.energy)
            converged_density = (
                np.asarray(scf_result.density_alpha, dtype=np.float64).copy(),
                np.asarray(scf_result.density_beta, dtype=np.float64).copy(),
            )
        else:
            energy, scf_result = _run_molecular_scf(
                mol,
                basis,
                method_lower,
                functional=functional,
                rhf_options=rhf_options,
                uhf_options=uhf_options,
                rks_options=rks_options,
                uks_options=uks_options,
                grid_level="legacy" if grid_options is not None else grid_level,
            )
        # A non-converged image has no valid gradient -- the C++ gradient
        # builders reject a non-converged density with a cryptic RuntimeError.
        # Catch it here and raise a clear, image-named NEB error (F2).
        if not getattr(scf_result, "converged", True):
            raise _nonconverged_image_error(method, scf_result, image_index, positions)
        if (
            method_lower == "uks"
            and getattr(scf_result, "stability_checked", False)
        ):
            if not getattr(scf_result, "stability_analysis_converged", False):
                raise _unverified_image_error(method, image_index, positions)
            if getattr(scf_result, "internal_instability", False):
                raise _unstable_image_error(
                    method, scf_result, image_index, positions
                )
        if method_lower == "roks":
            from .guess_read import resolve_read_densities_open
            if not np.isfinite(fd_step_bohr) or fd_step_bohr <= 0.0:
                raise ValueError(
                    "run_neb: fd_step_bohr must be finite and positive for ROKS; "
                    f"got {fd_step_bohr!r}"
                )
            grad = np.zeros_like(positions, dtype=float)
            for atom_index in range(len(positions)):
                for axis in range(3):
                    displaced_energies = []
                    for sign in (1.0, -1.0):
                        displaced = np.asarray(positions, dtype=float).copy()
                        displaced[atom_index, axis] += sign * fd_step_bohr
                        displaced_mol = _rebuild_with_positions(template, displaced)
                        displaced_basis = BasisSet(displaced_mol, basis_name)
                        displaced_result = _run_restricted_open_image(
                            run_roks, displaced_mol, displaced_basis, opts,
                            resolve_read_densities_open(
                                opts, displaced_mol, displaced_basis, scf_result,
                            ),
                            source_selection=scf_result.guess_selection,
                            functional=functional,
                        )
                        if not getattr(displaced_result, "converged", True):
                            raise _nonconverged_image_error(
                                method,
                                displaced_result,
                                image_index,
                                displaced,
                            )
                        displaced_energy = float(displaced_result.energy)
                        if dispersion_params is not None:
                            from .dispersion import compute_d3bj

                            displaced_energy += float(
                                compute_d3bj(displaced_mol, dispersion_params).energy
                            )
                        displaced_energies.append(displaced_energy)
                    grad[atom_index, axis] = (
                        displaced_energies[0] - displaced_energies[1]
                    ) / (2.0 * fd_step_bohr)
        else:
            grad = _compute_molecular_gradient(
                mol,
                basis,
                scf_result,
                method_lower,
                gradient_options=_image_gradient_options,
                grid_options=effective_grid_options,
                dispersion_params=dispersion_params,
                dft_plus_u=dft_plus_u,
            )
        if dispersion_params is not None:
            from .dispersion import compute_d3bj

            disp = compute_d3bj(mol, dispersion_params)
            energy = energy + float(disp.energy)
        return float(energy), np.asarray(grad, dtype=float), scf_result


def _evaluate_image_periodic(
    positions: np.ndarray,
    template: PeriodicSystem,
    basis_name: str,
    method: str,
    *,
    kmesh: Any,
    functional: Optional[str],
    rhf_options: Any,
    uhf_options: Any,
    rks_options: Any,
    uks_options: Any,
    fd_step_bohr: float,
    sr_image_precision: Optional[float],
    initial_density: Optional[np.ndarray] = None,
    dft_plus_u: Optional[Sequence[Any]] = None,
    image_index: Optional[int] = None,
) -> tuple[float, np.ndarray, Optional[np.ndarray]]:
    """Run a periodic BIPOLE SCF + finite-difference gradient.

    Returns (energy, gradient). Gradient shape (n_atoms, 3), Ha/bohr,
    in Cartesian coordinates -- *not* fractional, even though the
    underlying SCF is BIPOLE. The NEB outer loop works in Cartesian
    space throughout (positions, tangents, spring distances).

    The gradient is computed by central-differencing the SCF energy
    along each Cartesian degree of freedom. This is the path the
    the implementation keeps until the J^LR reciprocal-Ewald
    contribution lands in the analytic BIPOLE gradient (see
    ``python/vibeqc/bipole_gradient.py`` and ``docs/user_guide/neb.md``).
    Cost: 6N + 1 BIPOLE SCFs per
    image per outer iteration. Correctness is exact in the limit
    ``fd_step_bohr -> 0``.
    """
    from ._vibeqc_core import BasisSet
    from .bipole_optimize import _run_scf

    sys_current = _rebuild_with_positions(template, positions)
    basis_current = BasisSet(sys_current.unit_cell_molecule(), basis_name)
    method_upper = method.upper()
    opts = {
        "rhf": rhf_options,
        "uhf": uhf_options,
        "rks": rks_options,
        "uks": uks_options,
    }[method.lower()]

    # BUG 99 guard: periodic NEB passes user-supplied options directly
    # to the BIPOLE SCF driver, bypassing the periodic runner's validation.
    if opts is not None:
        from .ecp_metadata import validate_ecp_required as _v_ecp

        _v_ecp(opts, sys_current.unit_cell_molecule(), basis_name)

    # Periodic SCF density warm-start. The outer-loop cache feeds in
    # the previous outer iter's converged density at this geometry's
    # image, and the 6N FD-displaced SCFs below additionally
    # warm-start from the reference SCF's converged density
    # (within-image FD speedup). Density shape varies by method:
    #
    #   * RHF / RKS -- single closed-shell density (list of cell
    #     blocks); the SCF driver takes ``initial_density=blocks``.
    #   * UHF / UKS -- open-shell (alpha_blocks, beta_blocks) tuple;
    #     the SCF driver takes ``init_alpha=`` + ``init_beta=``.
    is_closed_shell = method_upper in ("RHF", "RKS")
    is_open_shell = method_upper in ("UHF", "UKS")

    def _warm_kwargs(density: Any, target_system, target_basis) -> dict[str, Any]:
        if density is None:
            return {}
        if getattr(density, "restart_basis", None) is not None:
            # Source-aware Gamma projection is defined. A displaced full-k
            # basis needs a Bloch cross-basis projector; execute the chosen
            # physical construction afresh in that case.
            points = np.asarray(getattr(density, "restart_kpoints", []))
            if points.shape != (1, 3) or not np.allclose(points, 0, atol=1e-10):
                return {}
            from .guess_read import (
                resolve_periodic_read_density_k_closed,
                resolve_periodic_read_densities_k_open,
            )
            loader = (resolve_periodic_read_density_k_closed if is_closed_shell else
                      resolve_periodic_read_densities_k_open)
            return {"initial_density_k": loader(
                read_from=density, basis=target_basis, system=target_system, kmesh=kmesh,
            )}
        # Explicit prepared blocks are already in the current AO basis.
        if is_closed_shell:
            return {"initial_density": density}
        alpha_blocks, beta_blocks = density
        return {"init_alpha": alpha_blocks, "init_beta": beta_blocks}

    def _image_scf(target_system, target_basis, source):
        warm = _warm_kwargs(source, target_system, target_basis)
        tags = list(getattr(opts, "atomic_spins", []))
        try:
            if warm and tags:
                opts.atomic_spins = []
            energy, result = _run_scf(
                target_system, target_basis, kmesh, opts, method_upper, functional,
                **bipole_domain_kwargs, **warm, **dft_plus_u_kwargs,
            )
        finally:
            if warm and tags:
                opts.atomic_spins = tags
        selection = getattr(source, "guess_selection", None)
        if warm and selection is not None:
            from ._vibeqc_core import GuessSelection, InitialGuess
            result.guess_selection = GuessSelection(
                selection.requested, selection.effective, InitialGuess.READ,
            )
        return energy, result

    # DFT+U kwargs for the BIPOLE drivers. All four periodic BIPOLE
    # entries (``run_pbc_bipole_{rhf,uhf,rks,uks}``) accept
    # ``dft_plus_u=[HubbardSite, ...]`` as of the closed-shell
    # BIPOLE +U landing on v0.9.0 main.
    dft_plus_u_kwargs: dict[str, Any] = (
        {"dft_plus_u": list(dft_plus_u)}
        if dft_plus_u
        else {}
    )
    bipole_domain_kwargs = {"sr_image_precision": sr_image_precision}

    energy, scf_result = _image_scf(sys_current, basis_current, initial_density)
    # Abort on a non-converged reference SCF before the 6N FD-displaced
    # SCFs run against a meaningless density (F2; mirrors the molecular
    # path). Safe no-op if the periodic result lacks a `converged` flag.
    if not getattr(scf_result, "converged", True):
        raise _nonconverged_image_error(method, scf_result, image_index, positions)

    # Reference density: the converged density from the just-run
    # reference SCF, in the shape this method requires. Used to
    # warm-start the FD-displaced SCFs *and* returned to the outer
    # loop's per-image cache for the next iteration.
    reference_density = scf_result

    n_atoms = positions.shape[0]
    grad = np.zeros((n_atoms, 3), dtype=float)
    for a in range(n_atoms):
        for c in range(3):
            disp_plus = positions.copy()
            disp_plus[a, c] += fd_step_bohr
            sys_p = _rebuild_with_positions(template, disp_plus)
            basis_p = BasisSet(sys_p.unit_cell_molecule(), basis_name)
            e_p, result_p = _image_scf(sys_p, basis_p, reference_density)
            if not getattr(result_p, "converged", True):
                raise _nonconverged_image_error(
                    method,
                    result_p,
                    image_index,
                    disp_plus,
                )
            disp_minus = positions.copy()
            disp_minus[a, c] -= fd_step_bohr
            sys_m = _rebuild_with_positions(template, disp_minus)
            basis_m = BasisSet(sys_m.unit_cell_molecule(), basis_name)
            e_m, result_m = _image_scf(sys_m, basis_m, reference_density)
            if not getattr(result_m, "converged", True):
                raise _nonconverged_image_error(
                    method,
                    result_m,
                    image_index,
                    disp_minus,
                )
            grad[a, c] = (e_p - e_m) / (2.0 * fd_step_bohr)
    return float(energy), grad, reference_density


# --- MACE (machine-learned interatomic potential) backend ------------------
#
# MACE provides analytic energy + forces from a pre-trained model, with no
# SCF and no Gaussian basis (CLAUDE.md Sec.10 maintainer-approved external
# pre-trained model -- see ``vibeqc.mlip.mace``). For NEB this is a much
# cheaper per-image evaluation than the SCF path, and -- importantly for the
# periodic case -- it sidesteps the 6N+1 finite-difference SCFs entirely:
# MACE returns the gradient directly.
#
# The model (torch weights) is loaded **once** and the ASE calculator reused
# for every image and outer iteration; constructing a fresh ``MACEModel`` per
# evaluation would reload the model each call. Because the loaded calculator
# is a live torch object, MACE-NEB runs the per-image loop serially
# (``n_jobs=1`` in ``run_neb``) rather than pickling the model across joblib
# worker processes -- each evaluation is a single forward pass, so serial is
# cheap.


def _load_mace_model(
    template: System,
    mlip_options: Any,
    is_periodic: bool,
) -> tuple[Any, np.ndarray, Optional[np.ndarray], str]:
    """Load the MACE model once and return ``(calc, numbers, cell, citation)``.

    ``calc`` is the reusable ASE calculator (eV / Angstrom); ``numbers`` the
    atomic numbers; ``cell`` the 3x3 lattice in bohr (``None`` for molecular);
    ``citation`` the per-model foundation-model citation key (for the
    references block -- e.g. ``batatia_mace_mp_2024``). The ASL gate (academic,
    non-commercial models) fires inside :class:`vibeqc.mlip.mace.MACEModel`.
    """
    from .mlip.mace import MACEModel

    cell = np.asarray(template.lattice, dtype=float) if is_periodic else None
    seed = template.unit_cell_molecule() if is_periodic else template
    model = MACEModel(seed, mlip_options, cell=cell)
    citation = getattr(getattr(model, "info", None), "citation", "") or ""
    return model.calculator, _atomic_numbers_of(template), cell, citation


def _evaluate_image_mace(
    positions: np.ndarray,
    *,
    calc: Any,
    numbers: np.ndarray,
    cell: Optional[np.ndarray],
    **_ignored: Any,
) -> tuple[float, np.ndarray, None]:
    """Energy + analytic gradient at one geometry from a pre-loaded MACE
    ASE calculator.

    Reuses ``calc`` (the loaded model); only the forward pass runs per call.
    Returns ``(energy_ha, gradient_ha_bohr, None)`` -- MACE keeps no SCF state,
    so there is no warm-start density (the ``initial_density=`` kwarg the
    outer loop passes is absorbed by ``**_ignored``).
    """
    from ase import Atoms
    from ase.units import Bohr, Hartree

    atoms = Atoms(
        numbers=[int(z) for z in numbers],
        positions=np.asarray(positions, dtype=float) * Bohr,
    )
    if cell is not None:
        atoms.set_cell(np.asarray(cell, dtype=float) * Bohr)
        atoms.set_pbc(True)
    atoms.calc = calc
    energy_ha = float(atoms.get_potential_energy()) / Hartree
    # ASE forces are eV/Angstrom; gradient = -force in Ha/bohr.
    grad = -np.asarray(atoms.get_forces(), dtype=float) * Bohr / Hartree
    return energy_ha, grad, None


# --- MSINDO (semiempirical INDO) backend -----------------------------------
#
# MSINDO is vibe-qc's own Bredow/Geudtner/Jug INDO re-implementation
# (CLAUDE.md Sec.10; ``vibeqc.semiempirical.methods.msindo``). It supplies a
# molecular total energy + a nuclear gradient with no Gaussian basis and no
# libint -- the STO/INDO Fock is built from the parameter tables, and the
# gradient is the central-difference derivative of that energy
# (``msindo_gradient_fd``, oracle-validated to <= 1e-4 Ha/bohr against
# MSINDO's analytic ``CARTOPT ANALY`` gradient). For NEB this is the first
# *semiempirical* image path.
#
# Two ways it differs from the SCF and MACE paths:
#   * No basis / functional / k-mesh (like MACE; unlike the SCF methods).
#   * No reusable live object (unlike MACE's torch calculator):
#     ``run_msindo`` / ``msindo_gradient_fd`` are stateless module-level
#     functions, so the band evaluates *in parallel* across images (joblib
#     processes) exactly like the SCF path -- MSINDO is not forced to the
#     serial ``n_jobs=1`` MACE uses.
#
# Cost note: the FD gradient is 6N ``run_msindo`` SCFs per image per outer
# iteration (central differences over the 3N Cartesian DOF). Keep
# MSINDO-NEB systems small; parallelism across images is the main lever.


def _evaluate_image_msindo(
    positions: np.ndarray,
    *,
    numbers: np.ndarray,
    charge: int,
    multiplicity: int,
    fd_step_bohr: float,
    dispersion_params: Any = None,
    image_index: Optional[int] = None,
    **_ignored: Any,
) -> tuple[float, np.ndarray, None]:
    """Energy + finite-difference gradient at one geometry from MSINDO.

    ``positions`` are bohr (NEB's working units); the MSINDO engine takes
    Angstrom, so they are converted with MSINDO's own constant for a bit-exact
    round-trip (the same constant ``run_job`` uses -- runner.py). Returns
    ``(energy_ha, gradient_ha_bohr, None)``: MSINDO carries no SCF state across
    geometries, so there is no warm-start density (the ``initial_density=`` the
    outer loop passes is absorbed by ``**_ignored``).

    The gradient is the central-difference nuclear gradient of the MSINDO total
    energy (``msindo_gradient_fd``); it is already Ha/bohr. ``fd_step_bohr`` is
    the FD half-step, converted to the engine's Angstrom. A non-converged
    MSINDO SCF raises the same image-named :class:`NEBImageSCFError` the SCF
    path uses. When ``dispersion_params`` is given, the D3-BJ energy + gradient
    are folded in (the FD gradient sees only the bare MSINDO energy, so the
    dispersion derivative is added explicitly here).
    """
    from .semiempirical.methods.msindo import (
        ANGSTROM_TO_BOHR as _A2B,
    )
    from .semiempirical.methods.msindo import (
        msindo_gradient_fd,
        run_msindo,
    )

    Z = [int(z) for z in numbers]
    coords_ang = np.asarray(positions, dtype=float) / _A2B
    fd_step_ang = float(fd_step_bohr) / _A2B

    result = run_msindo(
        Z, coords_ang, charge=int(charge), multiplicity=int(multiplicity)
    )
    # A non-converged SCF has no valid gradient (the FD displacements would
    # difference meaningless energies). Raise the same clear, image-named error
    # the SCF path uses (F2).
    if not getattr(result, "converged", True):
        raise _nonconverged_image_error("msindo", result, image_index, positions)
    energy = float(result.total_energy)

    grad = np.asarray(
        msindo_gradient_fd(
            Z,
            coords_ang,
            charge=int(charge),
            multiplicity=int(multiplicity),
            step=fd_step_ang,
        ),
        dtype=float,
    )

    if dispersion_params is not None:
        from .dispersion import compute_d3bj

        mol = Molecule(
            [Atom(int(z), [float(p[0]), float(p[1]), float(p[2])])
             for z, p in zip(Z, positions)],
            int(charge),
            int(multiplicity),
        )
        disp = compute_d3bj(mol, dispersion_params, with_gradient=True)
        energy += float(disp.energy)
        if getattr(disp, "gradient", None) is not None:
            grad = grad + np.asarray(disp.gradient, dtype=float)

    return energy, grad, None


def _evaluate_image_semiempirical(
    positions: np.ndarray,
    *,
    template: System,
    route_plan: Any,
    ccm_options: Any = None,
    seccm_topology: Any = None,
    seccm_max_tie_score_excursion: float | None = None,
    dispersion_params: Any = None,
    image_index: Optional[int] = None,
    **_ignored: Any,
) -> tuple[float, np.ndarray, None]:
    """Evaluate one molecular or explicitly selected SECCM image."""
    from .semiempirical.routes import BOUNDARY_SECCM_DIRECT_TORUS

    current = _rebuild_with_positions(template, positions)
    molecule = (
        current.unit_cell_molecule()
        if isinstance(current, PeriodicSystem)
        else current
    )
    if (
        route_plan.boundary == BOUNDARY_SECCM_DIRECT_TORUS
        and route_plan.method_key == "dftb0"
    ):
        from .molecule import ANGSTROM_TO_BOHR
        from .semiempirical.seccm import run_dftb0_seccm

        topology_coords = np.asarray(positions, dtype=float)
        if seccm_topology.length_unit == "angstrom":
            topology_coords = topology_coords / ANGSTROM_TO_BOHR
        rebuild_options = (
            {
                "max_tie_score_excursion": float(
                    seccm_max_tie_score_excursion
                )
            }
            if seccm_topology.has_reference_ties
            else {}
        )
        current_topology = seccm_topology.rebuild_displacements(
            topology_coords,
            **rebuild_options,
        )
        native = run_dftb0_seccm(
            molecule,
            current_topology,
            compute_gradient=True,
        )
        assert native.gradient is not None
        return float(native.energy), np.asarray(native.gradient), None

    from .semiempirical.runner import _run_semiempirical_plan

    result = _run_semiempirical_plan(
        route_plan,
        molecule,
        ccm_options=ccm_options,
    )
    if not result.converged:
        raise _nonconverged_image_error(
            route_plan.variant,
            result,
            image_index,
            positions,
        )
    gradient = result.gradient()
    if gradient is None:
        raise NotImplementedError(
            "the validated semiempirical NEB route returned no gradient for "
            f"method={route_plan.method_key!r}, boundary={route_plan.boundary!r}"
        )
    energy = float(result.energy)
    grad = np.asarray(gradient, dtype=float)

    if dispersion_params is not None:
        from .dispersion import compute_d3bj

        disp = compute_d3bj(molecule, dispersion_params, with_gradient=True)
        energy += float(disp.energy)
        if getattr(disp, "gradient", None) is not None:
            grad = grad + np.asarray(disp.gradient, dtype=float)

    return energy, grad, None


def _evaluate_image_periodic_semiempirical(
    positions: np.ndarray,
    *,
    template: PeriodicSystem,
    route_plan: Any,
    kpoints: Any = None,
    cutoff_bohr: float,
    fd_step_bohr: float,
    **_ignored: Any,
) -> tuple[float, np.ndarray, None]:
    """Evaluate one route-planned periodic semiempirical image."""
    from .semiempirical.periodic import evaluate_periodic_energy_gradient

    current = _rebuild_with_positions(template, positions)
    energy, gradient = evaluate_periodic_energy_gradient(
        route_plan,
        current,
        kpoints=kpoints,
        cutoff_bohr=cutoff_bohr,
        fd_step_bohr=fd_step_bohr,
    )
    return float(energy), np.asarray(gradient, dtype=float), None


def _periodic_gaussian_neb_kmesh(
    system: PeriodicSystem,
    kpoints: Any,
) -> Any:
    """Materialize the BIPOLE k mesh before dry-run capability checks."""
    from ._vibeqc_core import monkhorst_pack as _mp

    if kpoints is None:
        return _mp(system, (1, 1, 1))
    if hasattr(kpoints, "to_bloch_kmesh") or (
        hasattr(kpoints, "kpoints") and hasattr(kpoints, "weights")
    ):
        from .kpoints import as_bloch_kmesh

        return as_bloch_kmesh(kpoints)
    if not isinstance(kpoints, (tuple, list)) or len(kpoints) != 3:
        raise ValueError(
            "run_neb: kpoints mesh must contain exactly three positive "
            "integers."
        )
    if any(
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or int(value) < 1
        for value in kpoints
    ):
        raise ValueError(
            "run_neb: kpoints mesh must contain exactly three positive "
            "integers."
        )
    return _mp(system, tuple(int(value) for value in kpoints))


def _periodic_semiempirical_neb_boundary(kpoints: Any) -> str:
    """Resolve Gamma versus full-k before loading method parameters."""
    from .semiempirical.routes import (
        BOUNDARY_PERIODIC_GAMMA,
        BOUNDARY_PERIODIC_K,
    )

    if kpoints is None:
        return BOUNDARY_PERIODIC_GAMMA
    if isinstance(kpoints, (tuple, list, np.ndarray)):
        try:
            values = np.asarray(kpoints, dtype=float)
        except (TypeError, ValueError):
            values = np.empty(0, dtype=float)
        if values.shape != (3,) or not np.all(np.isfinite(values)):
            raise ValueError(
                "run_neb: periodic semiempirical kpoints tuples must contain "
                "three finite positive integers."
            )
        if np.any(values <= 0.0) or not np.array_equal(values, np.floor(values)):
            raise ValueError(
                "run_neb: periodic semiempirical kpoints tuples must contain "
                "three finite positive integers."
            )
        if np.array_equal(values, np.ones(3)):
            return BOUNDARY_PERIODIC_GAMMA
        return BOUNDARY_PERIODIC_K
    else:
        try:
            from .kpoints import as_bloch_kmesh

            mesh = as_bloch_kmesh(kpoints)
            points = np.asarray(mesh.kpoints, dtype=float)
            if points.shape == (1, 3) and np.allclose(points, 0.0):
                return BOUNDARY_PERIODIC_GAMMA
        except (AttributeError, TypeError, ValueError):
            raise ValueError(
                "run_neb: kpoints must be a three-integer mesh or a valid "
                "KPoints/BlochKMesh object."
            ) from None
        return BOUNDARY_PERIODIC_K


def _validate_periodic_seccm_lattice(
    system: PeriodicSystem,
    ccm_options: Any,
) -> None:
    """Require the QVF cell and the explicit cyclic translations to agree."""
    from .molecule import ANGSTROM_TO_BOHR

    translations = getattr(ccm_options, "translations", None)
    if translations is None:
        raise ValueError(
            "run_neb: method='seccm' requires explicit ccm_options.translations"
        )
    vectors = np.asarray(translations, dtype=float)
    expected_shape = (int(system.dim), 3)
    if vectors.shape != expected_shape:
        raise ValueError(
            "run_neb: periodic SECCM requires one cyclic translation per "
            f"active lattice dimension; expected {expected_shape}, got "
            f"{vectors.shape}."
        )
    active_lattice = np.asarray(system.lattice, dtype=float).T[: system.dim]
    if not np.allclose(
        vectors * ANGSTROM_TO_BOHR,
        active_lattice,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError(
            "run_neb: periodic SECCM ccm_options.translations must match the "
            "active PeriodicSystem lattice vectors; the translations define "
            "the cyclic Hamiltonian while the lattice is written to the QVF."
        )


def _validate_periodic_dftb0_seccm_lattice(
    system: PeriodicSystem,
    topology: Any,
) -> None:
    """Require the QVF cell and DFTB0 cyclic translations to agree."""
    from .molecule import ANGSTROM_TO_BOHR

    vectors = np.asarray(topology.translations, dtype=float)
    if topology.length_unit == "angstrom":
        vectors = vectors * ANGSTROM_TO_BOHR
    expected_shape = (int(system.dim), 3)
    if vectors.shape != expected_shape:
        raise ValueError(
            "run_neb: DFTB0-SECCM requires one cyclic translation per active "
            f"lattice dimension; expected {expected_shape}, got {vectors.shape}."
        )
    active_lattice = np.asarray(system.lattice, dtype=float).T[: system.dim]
    if not np.allclose(vectors, active_lattice, rtol=0.0, atol=1.0e-10):
        raise ValueError(
            "run_neb: DFTB0-SECCM topology translations must match the active "
            "PeriodicSystem lattice vectors."
        )


# --- NEB force kernel ------------------------------------------------------


def _neb_forces(
    positions: list[np.ndarray],
    energies: list[float],
    gradients: list[np.ndarray],
    spring_constant: float,
    frozen_mask: Optional[np.ndarray],
    climbing_index: Optional[int] = None,
    lattice: Optional[np.ndarray] = None,
    dim: int = 3,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Compute NEB total forces + tangents for every intermediate image.

    For periodic systems (``lattice`` given) the tangent and the spring
    distances use minimum-image inter-image displacements, so a band
    whose images straddle a cell boundary (e.g. surface self-diffusion)
    is handled correctly. This is a no-op for a non-crossing band (the
    minimum image of a small displacement is the displacement itself).

    Returns (forces, tangents). Both are length n_intermediate
    (excluding endpoints). Frozen atoms (per ``frozen_mask`` of shape
    (n_atoms,) bool, True = frozen) have their force components set
    to zero before return.

    ``climbing_index`` is the index (into ``positions``, i.e. global
    image index -- 0 is reactant, ``len(positions) - 1`` is product)
    of the climbing image. For that image the standard
    ``F_spring_∥ + F_true_⊥`` is replaced by the climbing-image force
    ``F_climb = -gradE + 2 (gradE . t) t`` (Henkelman, Uberuaga, Jónsson
    2000): the true force with its tangent-parallel component
    inverted, no spring contribution. The climbing image then
    relaxes uphill *along* t while still relaxing perpendicular,
    landing on the saddle. ``None`` => standard NEB on every
    intermediate image.
    """
    n_total = len(positions)
    forces: list[np.ndarray] = []
    tangents: list[np.ndarray] = []
    for i in range(1, n_total - 1):
        tau = _improved_tangent(
            positions[i - 1],
            positions[i],
            positions[i + 1],
            energies[i - 1],
            energies[i],
            energies[i + 1],
            lattice,
            dim,
        )
        true_force = -gradients[i]
        parallel = float(np.sum(true_force * tau))
        if climbing_index is not None and i == climbing_index:
            # Climbing image: invert the parallel component of the
            # true force and drop the spring contribution.
            #   F_climb = -gradE + 2 (gradE . t) t
            #           = (true_force - parallel t) + (-parallel t)
            #           = F_true_⊥ - parallel.t
            f_total = true_force - 2.0 * parallel * tau
        else:
            # Spring force projected onto t (signed). Minimum-image
            # inter-image displacements for periodic bands (no-op when
            # consecutive images are within half a cell, i.e. always
            # except across a PBC hop).
            disp_next = positions[i + 1] - positions[i]
            disp_prev = positions[i] - positions[i - 1]
            if lattice is not None:
                disp_next = _minimum_image_diff(disp_next, lattice, dim)
                disp_prev = _minimum_image_diff(disp_prev, lattice, dim)
            d_next = float(np.linalg.norm(disp_next))
            d_prev = float(np.linalg.norm(disp_prev))
            f_spring = spring_constant * (d_next - d_prev) * tau
            # True force perpendicular to t.
            f_perp = true_force - parallel * tau
            f_total = f_spring + f_perp
        if frozen_mask is not None:
            f_total = f_total.copy()
            f_total[frozen_mask] = 0.0
        forces.append(f_total)
        tangents.append(tau)
    return forces, tangents


# --- public entry point ----------------------------------------------------


def _neb_scf_options(
    method_lower: str,
    *,
    rhf_options: Any = None,
    uhf_options: Any = None,
    rks_options: Any = None,
    uks_options: Any = None,
    rohf_options: Any = None,
    roks_options: Any = None,
) -> Any:
    if method_lower == "rhf":
        return rhf_options
    if method_lower == "uhf":
        return uhf_options
    if method_lower == "rks":
        return rks_options
    if method_lower == "uks":
        return uks_options
    if method_lower == "rohf":
        return rohf_options
    if method_lower == "roks":
        return roks_options
    return None


def _molecular_neb_manual_ecp_requested(options: Any) -> bool:
    """Whether ``options`` carries any explicit molecular ECP request.

    ``ecp_total_ncore`` is provenance rather than a complete ECP
    configuration. A zero value by itself therefore remains the ordinary
    all-electron default; malformed or nonzero values fail closed. Any
    nonempty ECP data field is an explicit request even when its companion
    count is absent or zero.
    """
    from .ecp_metadata import molecular_options_request_ecp_operator

    return molecular_options_request_ecp_operator(options)


def _refuse_molecular_neb_ecp_configuration(
    molecule: Molecule,
    basis_name: str,
    *option_sources: Any,
) -> None:
    """Reject ECP-backed top-level NEB before its warm image loop runs."""
    from .ecp_metadata import basis_sidecar_has_ecp_operator

    if basis_sidecar_has_ecp_operator(molecule, basis_name) or any(
        _molecular_neb_manual_ecp_requested(source)
        for source in option_sources
    ):
        raise NotImplementedError(
            "run_neb: molecular Gaussian ECP reaction paths are not yet "
            "end-to-end certified. Individual image evaluation preserves "
            "the ECP Hamiltonian and moving centres, but full band evolution "
            "and convergence have not been validated as one ECP reaction "
            "surface. Use an all-electron basis."
        )


def _reject_molecular_gaussian_neb_options(
    reactant: Molecule,
    basis_name: str,
    method_lower: str,
    options: Any,
    gradient_options: Any = None,
) -> None:
    """Fail closed on high-level SCF features bypassed by the warm loop."""
    _refuse_molecular_neb_ecp_configuration(
        reactant,
        basis_name,
        options,
        gradient_options,
    )

    if options is None:
        return
    if method_lower != "roks" and bool(getattr(options, "density_fit", False)):
        raise NotImplementedError(
            "run_neb: molecular Gaussian density fitting is not implemented "
            "by the warm-start image builder. Disable density_fit."
        )
    if method_lower != "roks" and bool(getattr(options, "cosx", False)):
        raise NotImplementedError(
            "run_neb: molecular Gaussian COSX is not implemented by the "
            "warm-start image builder. Disable COSX."
        )
    if list(getattr(options, "multi_guess_seeds", []) or []):
        raise NotImplementedError(
            "run_neb: molecular multi-guess basin selection is unavailable. "
            "Selecting a separate lowest basin at each image would not "
            "define one continuous reaction path."
        )
    if method_lower in ("uhf", "uks"):
        from ._vibeqc_core import SpinlockMode

        if (
            getattr(options, "spinlock_mode", SpinlockMode.OFF)
            == SpinlockMode.SPIN_SCHEDULE
        ):
            raise NotImplementedError(
                "run_neb: molecular SPIN_SCHEDULE is not implemented by the "
                "warm-start image loop. Use PATTERN_HOLD or disable spinlock."
            )


def _validate_neb_dft_plus_u_sites(
    system: System,
    basis_name: str,
    sites: Optional[Sequence[Any]],
) -> None:
    """Validate every Hubbard channel before dry-run writes a manifest."""
    if not sites:
        return
    from ._vibeqc_core import BasisSet
    from .dft_plus_u import ao_group_indices

    molecule = (
        system.unit_cell_molecule()
        if isinstance(system, PeriodicSystem)
        else system
    )
    available_groups = ao_group_indices(BasisSet(molecule, basis_name))
    for site in sites:
        key = (int(site.atom_index), int(site.l))
        if key not in available_groups:
            raise ValueError(
                f"HubbardSite(atom_index={site.atom_index}, l={site.l}) "
                "has no AOs in the basis. Available (atom_index, l) "
                f"channels: {sorted(available_groups.keys())}"
            )


def _resolve_molecular_neb_read_options(
    reactant: Molecule,
    basis_name: str,
    method_lower: str,
    options: Any,
) -> None:
    """Resolve path-backed READ once, before dry-run or image dispatch."""
    if options is None or method_lower not in ("rhf", "uhf", "rks", "uks"):
        return
    from ._vibeqc_core import BasisSet, InitialGuess

    kind = getattr(options, "initial_guess", InitialGuess.AUTO)
    if kind not in (InitialGuess.READ, InitialGuess.FRAGMO):
        return
    basis = BasisSet(reactant, basis_name)
    nbf = int(basis.nbasis)
    read_path = str(getattr(options, "read_path", "") or "")
    if method_lower in ("rhf", "rks"):
        density = np.asarray(getattr(options, "read_density", []), dtype=float)
        if kind == InitialGuess.FRAGMO:
            if density.shape == (nbf, nbf):
                return
            raise ValueError(
                "run_neb: FRAGMO requires a complete precomputed "
                "options.read_density because run_neb has no fragments= "
                "input seam."
            )
        if not read_path and density.shape == (nbf, nbf):
            return
        from .guess_read import resolve_read_density_closed

        options.read_density = resolve_read_density_closed(
            options,
            reactant,
            basis,
            None,
        )
        return

    density_alpha = np.asarray(
        getattr(options, "read_density_alpha", []), dtype=float
    )
    density_beta = np.asarray(
        getattr(options, "read_density_beta", []), dtype=float
    )
    if kind == InitialGuess.FRAGMO:
        if (
            density_alpha.shape == (nbf, nbf)
            and density_beta.shape == (nbf, nbf)
        ):
            return
        raise ValueError(
            "run_neb: FRAGMO requires complete precomputed "
            "options.read_density_alpha/read_density_beta matrices because "
            "run_neb has no fragments= input seam."
        )
    if (
        not read_path
        and density_alpha.shape == (nbf, nbf)
        and density_beta.shape == (nbf, nbf)
    ):
        return
    from .guess_read import resolve_read_densities_open

    density_alpha, density_beta = resolve_read_densities_open(
        options,
        reactant,
        basis,
        None,
    )
    options.read_density_alpha = density_alpha
    options.read_density_beta = density_beta


def _neb_dry_run_estimate_bytes(
    reactant: System,
    *,
    basis: str | None,
    method_lower: str,
    n_images: int,
    n_jobs: int,
    n_atoms: int,
    is_periodic: bool,
    warm_start: bool,
    rhf_options: Any = None,
    uhf_options: Any = None,
    rks_options: Any = None,
    uks_options: Any = None,
) -> int | None:
    if basis is None or method_lower not in ("rhf", "uhf", "rks", "uks"):
        return None
    try:
        from .memory import estimate_memory, estimate_neb_memory

        molecule = (
            reactant.unit_cell_molecule()
            if isinstance(reactant, PeriodicSystem)
            else reactant
        )
        basis_obj = BasisSet(molecule, basis)
        per_image = estimate_memory(
            molecule,
            basis_obj,
            method=method_lower,
            options=_neb_scf_options(
                method_lower,
                rhf_options=rhf_options,
                uhf_options=uhf_options,
                rks_options=rks_options,
                uks_options=uks_options,
            ),
        )
        fd_evaluations = 6 * n_atoms + 1 if is_periodic else 1
        return estimate_neb_memory(
            per_image,
            n_images=n_images,
            n_jobs=n_jobs,
            n_atoms=n_atoms,
            n_basis=basis_obj.nbasis,
            open_shell=method_lower in ("uhf", "uks"),
            finite_difference_evaluations=fd_evaluations,
            warm_start=warm_start,
        ).total_bytes
    except Exception:
        return None


def _write_neb_dry_run_manifest(
    *,
    output: str | os.PathLike,
    method: str,
    basis: str | None,
    functional: str | None,
    record_hostname: bool,
    estimate_bytes: int | None,
) -> None:
    output_stem = Path(os.fspath(output))
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    plan = OutputPlan(
        stem=output_stem,
        job_kind="neb",
        method=method,
        basis=basis or "(none)",
        functional=functional,
        files=(),
    )
    dry_run_manifest(
        plan,
        record_hostname=record_hostname,
        estimate_bytes=estimate_bytes,
    )


def run_neb(
    reactant: System,
    product: System,
    basis: Optional[str] = None,
    n_images: int = 7,
    *,
    method: str = "RKS",
    functional: Optional[str] = "pbe",
    spring_constant: float = 0.1,
    interpolation: str = "idpp",
    max_iter: int = 100,
    conv_tol_force: float = 1e-3,
    freeze_indices: Optional[Sequence[int]] = None,
    dispersion_params: Any = None,
    rhf_options: Any = None,
    uhf_options: Any = None,
    rks_options: Any = None,
    uks_options: Any = None,
    rohf_options: Any = None,
    roks_options: Any = None,
    grid_level: str = "orca-defgrid3",
    gradient_options: Any = None,
    grid_options: Any = None,
    n_jobs: int = 0,
    initial_step: float = 0.05,
    max_step: float = 0.2,
    progress: bool = False,
    output: Union[str, os.PathLike] = "output",
    dry_run: bool = False,
    record_hostname: bool = True,
    climbing_image: bool = False,
    climbing_image_start_fraction: float = 0.3,
    kpoints: Optional[Any] = None,
    fd_step_bohr: float = 1e-3,
    sr_image_precision: Optional[float] = 1e-6,
    warm_start: bool = True,
    dft_plus_u: Optional[Sequence[Any]] = None,
    ccm_options: Any = None,
    seccm_topology: Any = None,
    seccm_max_tie_score_excursion: float | None = None,
    semiempirical_cutoff_bohr: float = 15.0,
    mlip_options: Any = None,
) -> NEBResult | None:
    """Find a minimum-energy path with improved-tangent NEB.

    Parameters
    ----------
    reactant, product
        :class:`Molecule` or :class:`PeriodicSystem` endpoints (both
        the same type). Same atom-number sequence and length; the
        order is preserved through the path.
    basis
        Basis-set name passed to :class:`BasisSet`, rebuilt per image
        per outer iteration. Required for the SCF methods; optional and
        ignored for ``method="mace"`` and all semiempirical methods.
    n_images
        Number of *intermediate* images. The returned path has
        ``n_images + 2`` images (endpoints included). Endpoints are
        fixed by default.
    method
        ``"RHF"`` / ``"UHF"`` / ``"ROHF"`` / ``"RKS"`` / ``"UKS"`` /
        ``"ROKS"`` for self-consistent-field per-image energies + gradients;
        ROHF uses its analytic gradient and ROKS uses central energy
        differences. ``"MACE"`` drives a
        pre-trained MACE machine-learned interatomic potential
        (``mlip_options=``; analytic energy + forces, no SCF / basis /
        k-mesh); or any route-planned semiempirical method: DFTB0, SCC-DFTB,
        GFN2-xTB, PM6/UPM6, OM1/OM2/OM3, molecular MSINDO, and the existing
        MSINDO-SECCM route. Periodic semiempirical NEB supports Gamma
        DFTB0/SCC-DFTB/GFN2-xTB/PM6 and the validated zero-temperature full-k
        DFTB0/SCC-DFTB derivative routes. Bloch-periodic OMx and all PM7 routes
        fail closed until their published Hamiltonians are implemented. Each
        executable family keeps its validated analytic or
        explicitly finite-difference derivative. Case-insensitive. Same
        dispatch as :func:`vibeqc.run_job` and
        :func:`vibeqc.molecular_optimize.optimize_molecule`.
    functional
        XC functional for KS-DFT (e.g. ``"pbe"``, ``"b3lyp"``).
        Ignored for HF methods.
    spring_constant
        ``k`` in Ha/bohr^2 for the spring term
        ``F_spring_i = k (|R_{i+1} - R_i| - |R_i - R_{i-1}|) t_i``.
        0.1 is the canonical default; turn down if the path is
        chemically smooth, up if it kinks.
    interpolation
        ``"idpp"`` (default, recommended for bonded paths) or
        ``"linear"``.
    max_iter
        Hard cap on outer iterations. Quick-min iterations are cheap
        in this loop's terms (one parallel SCF batch each).
    conv_tol_force
        Convergence threshold on the max-norm of the NEB force over
        all intermediate atoms (Ha/bohr). 1e-3 ≈ 0.05 eV/Å -- the
        Kolsbjerg 2016 recommendation.
    freeze_indices
        Atom indices to freeze (NEB-local; the SCF + gradient still
        sees them, but their force components are zeroed before the
        step). Useful for slab substrate atoms in surface NEB.
    dispersion_params
        Optional D3-BJ parameters; the dispersion energy and gradient
        are folded into each image's energy + gradient.
    n_jobs
        joblib.Parallel ``n_jobs``. ``0`` (default) selects a bounded
        automatic worker count; ``-1`` explicitly requests all cores;
        ``1`` is serial.
    initial_step, max_step
        Quick-min step sizes (bohr). The integrator starts at
        ``initial_step`` and scales up (capped at ``max_step``) when
        the velocity is aligned with the force.
    progress
        If True, writes one line per outer iteration through the ambient
        :mod:`vibeqc.output` channel.
    output
        Output stem used by dry-run preflight. ``run_neb`` does not yet write
        a live NEB output bundle, but ``dry_run=True`` (or
        ``VIBEQC_DRY_RUN=1``) writes ``{output}.system`` so ``vq submit auto``
        can read the planned job kind and optional memory estimate.
    dry_run
        If True, write the dry-run manifest and return ``None`` without
        interpolation, endpoint evaluation, or any per-image SCF. Setting
        ``VIBEQC_DRY_RUN=1`` has the same effect; setting
        ``VIBEQC_DRY_RUN_ESTIMATE=1`` additionally records
        ``[memory].estimate_bytes`` when the Gaussian SCF route is estimable.
    climbing_image
        Enable climbing-image NEB (CI-NEB, Henkelman+Uberuaga+Jónsson
        2000). After a warm-up phase the highest-energy intermediate
        image is promoted to "climbing": its spring contribution is
        dropped and the tangent-parallel component of its true force
        is inverted, so it climbs uphill along t to the saddle while
        still relaxing perpendicular. Other images keep standard NEB
        dynamics. Default ``False`` (plain improved-tangent NEB).
    climbing_image_start_fraction
        Fraction of ``max_iter`` to spend in the plain-NEB warm-up
        phase before promoting an image to climbing. Default 0.3 --
        the band has typically found its rough shape by this point,
        so the identity of the highest-energy image is reliable.
        Ignored when ``climbing_image=False``.
    kpoints
        Periodic-only. Either a ``BlochKMesh`` or ``KPoints`` object, or a
        3-tuple of positive ints (the Monkhorst-Pack mesh sizes). Full-k
        semiempirical NEB is currently restricted to zero-temperature,
        closed-shell DFTB0 and SCC-DFTB. Ignored when
        ``reactant``/``product`` are ``Molecule``. ``None`` =>
        Γ-only mesh (``(1, 1, 1)``) for a sanity-check periodic
        run; pick a real k-mesh for production.
    fd_step_bohr
        Half-step for finite-difference per-image gradients. It is used by
        periodic Gaussian SCF, molecular ``method="msindo"``, periodic
        SCC-DFTB and GFN2-xTB total-energy differences, and the PM6/OMx
        native finite-difference batches.
        Periodic: the J^LR reciprocal contribution is still missing from
        the analytic BIPOLE gradient, so the periodic NEB driver uses the
        FD fallback to keep things bit-exact (cost per image 6N + 1 BIPOLE
        SCFs; the NEB user guide tracks the analytic-gradient
        switch). Molecular MSINDO: the engine exposes the FD gradient used
        by its existing NEB route
        (``msindo_gradient_fd``; cost 6N ``run_msindo`` SCFs per image).
        Converted to the MSINDO engine's Angstrom internally. Default
        1e-3 bohr -- same value the BIPOLE-FD gradient unit tests use. DFTB0
        and MSINDO-SECCM use analytic gradients at their validated domains.
        Ignored for molecular Gaussian SCF and other molecular semiempirical
        analytic routes, and for ``method="mace"`` (analytic forces).
    sr_image_precision
        Periodic BIPOLE ket-image precision forwarded to every reference and
        finite-difference SCF. The production default is ``1e-6``. Pass
        ``None`` only for an explicit historical-domain diagnostic or a
        bounded-cost dispatch test whose numerical accuracy is not under
        test; this does not cap or change the production default. Ignored for
        molecular, MACE, and MSINDO paths.
    warm_start
        Within-image density warm-start across outer iterations
        (the NEB warm-start milestone -- the 2-4x SCF-cost reduction
        this path was built to capture). When True (default) the
        converged density from outer iter N is fed in as the SCF
        initial guess at iter N+1 for the same image -- the geometry
        change between outer iters is small relative to a SAD/Hcore
        guess, so the SCF converges in fewer iterations. Active for all six
        molecular mean-field methods (RHF / UHF / ROHF / RKS / UKS / ROKS)
        and for RHF / UHF / RKS / UKS periodic BIPOLE; periodic also starts
        6N FD-displaced SCFs from the reference SCF's converged
        density. Bit-exact vs. cold-start in every case. Pass
        ``False`` to force every SCF to cold-start (useful for
        benchmarking the speedup).
    dft_plus_u
        Optional iterable of :class:`vibeqc.HubbardSite` objects.
        Each entry adds the Dudarev rotationally-invariant per-spin
        potential ``V_U^A = U_eff (1/2 d - n^A_l)`` on that
        ``(atom_index, l)`` channel for every per-image SCF in the
        NEB run. The Hubbard energy ``E_U`` contributes to each
        image's ``e_dft_plus_u`` (and to ``result.energy`` through
        the standard +U bookkeeping); the NEB driver picks this up
        uniformly through the SCF dispatch. Supported for all four
        methods, molecular and periodic -- the molecular leg applies
        +U Options-side via ``_apply_dft_plus_u_to_options``; the
        periodic leg forwards ``dft_plus_u=`` straight to
        ``run_pbc_bipole_{rhf,uhf,rks,uks}``. See the small-cell
        +U projector caveat in ``docs/user_guide/neb.md`` Sec. DFT+U.
        Rejected with ``method="mace"`` (the SCF-only correction).
    ccm_options
        Required for ``method="seccm"``. For PeriodicSystem endpoints the
        explicit cyclic translations must match the active lattice vectors so
        the Hamiltonian boundary and periodic reaction-path QVF describe the
        same cell.
    seccm_topology
        Explicit frozen :class:`SECCMTopology` selecting DFTB0 on the SECCM
        boundary. This is separate from MSINDO ``ccm_options`` and currently
        requires PeriodicSystem endpoints in the gated one-dimensional H/C
        envelope.
    seccm_max_tie_score_excursion
        Positive fixed-topology trust bound required when ``seccm_topology``
        contains exact face/edge/corner ties. It is checked before endpoint
        evaluation and forwarded unchanged to every image rebuild.
    semiempirical_cutoff_bohr
        Real-space image cutoff for periodic semiempirical methods. Default
        15 bohr. Gamma DFTB0 uses its analytic gradient at that validated
        domain and falls back to an explicitly labelled total-energy finite
        difference for another cutoff. Full-k DFTB0/SCC-DFTB use the native
        batched finite-difference Bloch kernel.
    mlip_options
        Used only when ``method="mace"``: a
        :class:`vibeqc.mlip.MLIPOptions` selecting the MACE foundation
        model, device, dtype, and (for academic-only ASL models) the
        license acknowledgment. ``None`` => the MIT MACE-MPA-0 default.
        The model is loaded once and its calculator reused for every
        image, so MACE-NEB runs serially (``n_jobs`` is forced to 1).
        MACE returns analytic forces, so the periodic FD-gradient path
        is bypassed entirely. Requires the optional ``[mace]`` extra
        (PyTorch + e3nn; Python <= 3.13) -- see :mod:`vibeqc.mlip.mace`.

    Returns
    -------
    :class:`NEBResult`, or ``None`` for dry-run preflight.

    Notes
    -----
    The outer loop is the quick-min (damped MD) integrator of
    Henkelman+Jónsson 2000. Each intermediate image carries a
    velocity; at each step the velocity is projected onto the NEB
    force, zeroed if the projection is negative (preventing climbing
    against the force). Step length grows by a factor of 1.1 when
    aligned and is capped at ``max_step``; it is reset to
    ``initial_step`` on direction flips. This is the textbook
    "FIRE-lite" cousin used by ASE's MDMin.
    """
    is_periodic = isinstance(reactant, PeriodicSystem)
    if is_periodic and not isinstance(product, PeriodicSystem):
        raise ValueError(
            "run_neb: reactant and product must be the same system "
            "type -- got PeriodicSystem and Molecule."
        )
    if not is_periodic and not isinstance(reactant, Molecule):
        raise NotImplementedError(
            "run_neb: only Molecule and PeriodicSystem endpoints are "
            "supported."
        )
    if n_images < 1:
        raise ValueError(f"n_images must be >= 1; got {n_images}")
    if (
        isinstance(max_iter, (bool, np.bool_))
        or not isinstance(max_iter, (int, np.integer))
        or int(max_iter) < 1
    ):
        raise ValueError(
            f"max_iter must be an integer >= 1; got {max_iter!r}"
        )

    def _finite_neb_scalar(
        name: str,
        value: Any,
        *,
        allow_zero: bool = False,
    ) -> float:
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(
                f"{name} must be finite and "
                f"{'non-negative' if allow_zero else 'positive'}; got {value!r}"
            )
        try:
            scalar = float(value)
        except (OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} must be finite and "
                f"{'non-negative' if allow_zero else 'positive'}; got {value!r}"
            ) from exc
        valid_range = scalar >= 0.0 if allow_zero else scalar > 0.0
        if not np.isfinite(scalar) or not valid_range:
            raise ValueError(
                f"{name} must be finite and "
                f"{'non-negative' if allow_zero else 'positive'}; got {value!r}"
            )
        return scalar

    spring_constant = _finite_neb_scalar(
        "spring_constant", spring_constant, allow_zero=True
    )
    conv_tol_force = _finite_neb_scalar("conv_tol_force", conv_tol_force)
    initial_step = _finite_neb_scalar("initial_step", initial_step)
    max_step = _finite_neb_scalar("max_step", max_step)
    fd_step_bohr = _finite_neb_scalar("fd_step_bohr", fd_step_bohr)
    if initial_step > max_step:
        raise ValueError(
            "initial_step must be <= max_step; got "
            f"initial_step={initial_step!r}, max_step={max_step!r}"
        )
    if isinstance(climbing_image_start_fraction, (bool, np.bool_)):
        raise ValueError(
            "climbing_image_start_fraction must be finite and in [0, 1]; "
            f"got {climbing_image_start_fraction!r}"
        )
    try:
        climbing_image_start_fraction = float(climbing_image_start_fraction)
    except (OverflowError, TypeError, ValueError) as exc:
        raise ValueError(
            "climbing_image_start_fraction must be finite and in [0, 1]; "
            f"got {climbing_image_start_fraction!r}"
        ) from exc
    if (
        not np.isfinite(climbing_image_start_fraction)
        or not 0.0 <= climbing_image_start_fraction <= 1.0
    ):
        raise ValueError(
            "climbing_image_start_fraction must be finite and in [0, 1]; "
            f"got {climbing_image_start_fraction!r}"
        )
    _check_compatible(reactant, product)

    # Lattice + periodic dimensionality for minimum-image inter-image
    # displacements in the NEB force (tangent + spring). None for the
    # molecular case (plain Cartesian); the shared cell is enforced by
    # _check_compatible.
    neb_lattice = (
        np.asarray(reactant.lattice, dtype=float) if is_periodic else None
    )
    neb_dim = int(reactant.dim) if is_periodic else 3

    method_lower = method.lower()
    from .semiempirical.routes import (
        BOUNDARY_PERIODIC_K,
        BOUNDARY_SECCM_DIRECT_TORUS,
        SemiempiricalRoutePlan,
        is_semiempirical_method,
        plan_periodic_semiempirical_route,
    )

    is_semiempirical = is_semiempirical_method(method)
    if (
        method_lower not in ("rhf", "uhf", "rohf", "rks", "uks", "roks", "mace")
        and not is_semiempirical
    ):
        raise ValueError(
            f"run_neb: unsupported method {method!r}. "
            "Use RHF/UHF/ROHF/RKS/UKS/ROKS, MACE, DFTB0, SCC-DFTB, GFN2-xTB, "
            "PM6/UPM6, OM1/OM2/OM3, MSINDO, or SECCM."
        )
    if is_periodic and method_lower in ("rohf", "roks"):
        raise NotImplementedError(
            "run_neb: ROHF/ROKS reaction paths are currently molecular-only; "
            "periodic restricted-open-shell NEB remains gated."
        )
    is_mace = method_lower == "mace"
    if seccm_topology is not None and not is_semiempirical:
        raise ValueError(
            "run_neb: seccm_topology is only supported by a route-planned "
            "semiempirical method."
        )
    if (
        seccm_max_tie_score_excursion is not None
        and seccm_topology is None
    ):
        raise ValueError(
            "run_neb: seccm_max_tie_score_excursion requires seccm_topology."
        )
    semiempirical_plan = None
    if is_semiempirical:
        preview = SemiempiricalRoutePlan.from_request(
            method,
            boundary="seccm" if seccm_topology is not None else None,
            properties=("energy", "gradient"),
            charge=int(getattr(reactant, "charge", 0)),
            multiplicity=int(getattr(reactant, "multiplicity", 1)),
            ccm_options=ccm_options,
        )
        if is_periodic and preview.boundary != BOUNDARY_SECCM_DIRECT_TORUS:
            if preview.method_key == "msindo":
                raise NotImplementedError(
                    "run_neb: MSINDO is molecular-only on this public route; "
                    "use method='seccm' with explicit ccm_options for the "
                    "current periodic MSINDO cyclic boundary."
                )
            boundary = _periodic_semiempirical_neb_boundary(kpoints)
            semiempirical_plan = plan_periodic_semiempirical_route(
                method,
                reactant,
                boundary=boundary,
                properties=("energy", "gradient"),
            )
        else:
            semiempirical_plan = preview

    is_seccm = bool(
        semiempirical_plan is not None
        and semiempirical_plan.boundary == BOUNDARY_SECCM_DIRECT_TORUS
    )
    is_msindo = bool(
        semiempirical_plan is not None
        and semiempirical_plan.method_key == "msindo"
        and semiempirical_plan.variant == "indo"
        and not is_seccm
    )
    is_dftb0_seccm = bool(
        is_seccm
        and semiempirical_plan is not None
        and semiempirical_plan.method_key == "dftb0"
    )
    if is_dftb0_seccm:
        from .semiempirical.seccm import SECCMTopology

        if not isinstance(seccm_topology, SECCMTopology):
            raise TypeError(
                "run_neb: DFTB0-SECCM requires seccm_topology=SECCMTopology"
            )
        if not is_periodic:
            raise NotImplementedError(
                "run_neb: the public DFTB0-SECCM reaction-path route requires "
                "PeriodicSystem endpoints so the QVF records the cyclic cell."
            )
        if ccm_options is not None:
            raise ValueError(
                "run_neb: ccm_options belongs to MSINDO-SECCM and cannot be "
                "combined with DFTB0 seccm_topology."
            )
        if seccm_topology.has_reference_ties:
            if (
                seccm_max_tie_score_excursion is None
                or not np.isfinite(seccm_max_tie_score_excursion)
                or seccm_max_tie_score_excursion <= 0.0
            ):
                raise ValueError(
                    "run_neb: tied DFTB0-SECCM topologies require a positive "
                    "finite seccm_max_tie_score_excursion."
                )
        elif seccm_max_tie_score_excursion is not None and (
            not np.isfinite(seccm_max_tie_score_excursion)
            or seccm_max_tie_score_excursion <= 0.0
        ):
            raise ValueError(
                "run_neb: seccm_max_tie_score_excursion must be positive and "
                "finite when provided."
            )
        _validate_periodic_dftb0_seccm_lattice(reactant, seccm_topology)
    elif is_seccm:
        if seccm_topology is not None:
            raise ValueError(
                "run_neb: seccm_topology currently belongs only to the "
                "DFTB0-SECCM adapter."
            )
        if ccm_options is None:
            raise ValueError("run_neb: method='seccm' requires ccm_options")
        if is_periodic:
            _validate_periodic_seccm_lattice(reactant, ccm_options)

    n_jobs = _resolve_neb_n_jobs(
        n_jobs,
        n_images=n_images,
        is_periodic=is_periodic,
    )
    # Only the Gaussian SCF routes need a user-selected basis.
    needs_basis = method_lower in ("rhf", "uhf", "rohf", "rks", "uks", "roks")
    if needs_basis and basis is None:
        raise ValueError(
            f"run_neb: a basis set is required for method={method!r}. "
            "(basis is optional for MACE and semiempirical routes.)"
        )
    if (
        not is_periodic
        and method_lower in ("rhf", "rks")
        and int(reactant.multiplicity) != 1
    ):
        raise ValueError(
            f"run_neb: molecular {method_lower.upper()} requires a singlet "
            f"(multiplicity=1); got multiplicity={reactant.multiplicity}. "
            "Use UHF/UKS for an open-shell path."
        )
    selected_scf_options = _neb_scf_options(
        method_lower,
        rhf_options=rhf_options,
        uhf_options=uhf_options,
        rks_options=rks_options,
        uks_options=uks_options,
        rohf_options=rohf_options,
        roks_options=roks_options,
    )
    effective_functional = functional
    if method_lower in ("rks", "uks", "roks"):
        from ._vibeqc_core import Functional

        method_options = (
            roks_options if method_lower == "roks" else selected_scf_options
        )
        effective_functional = (
            functional
            or getattr(method_options, "functional", "")
            or "lda"
        )
        functional_info = Functional(str(effective_functional))
        if bool(getattr(functional_info, "is_double_hybrid", False)):
            raise NotImplementedError(
                "run_neb: double-hybrid KS reaction paths are unavailable. "
                "The image SCF loop does not include the required MP2 "
                "correlation energy or gradient."
            )
        if (
            method_lower in ("rks", "uks")
            and bool(getattr(functional_info, "needs_vv10", False))
        ):
            raise NotImplementedError(
                "run_neb: VV10/nonlocal-correlation KS reaction paths are "
                "unavailable because the analytic molecular gradient does "
                "not include the VV10 derivative."
            )
    if needs_basis and not is_periodic:
        _reject_molecular_gaussian_neb_options(
            reactant,
            str(basis),
            method_lower,
            selected_scf_options,
            gradient_options,
        )
        _resolve_molecular_neb_read_options(
            reactant,
            str(basis),
            method_lower,
            selected_scf_options,
        )
        if (
            selected_scf_options is not None
            and list(
                getattr(selected_scf_options, "dft_plus_u_sites", []) or []
            )
            and not dft_plus_u
        ):
            raise NotImplementedError(
                "run_neb: options-side DFT+U metadata requires the matching "
                "dft_plus_u=[HubbardSite(...)] request so image gradients "
                "include the explicit Hubbard derivative."
            )
        if method_lower != "roks" and gradient_options is not None:
            gradient_bad = bool(
                getattr(gradient_options, "density_fit", False)
                or getattr(gradient_options, "cosx", False)
                or getattr(gradient_options, "ecp_centers", [])
                or getattr(gradient_options, "ecp_library", "")
                # Inline primitive ECPs are expressible on GradientOptions
                # since #574; they are an ECP derivative just as much as the
                # XML-library centres and must not slip past this guard.
                or getattr(gradient_options, "ecp_primitive_blocks", [])
            )
            if gradient_bad:
                raise NotImplementedError(
                    "run_neb: molecular Gaussian gradient_options request a "
                    "two-electron or ECP derivative that does not match the "
                    "maintained direct all-electron image Hamiltonian. Use "
                    "default direct all-electron gradient options."
                )
    if needs_basis:
        _validate_neb_dft_plus_u_sites(reactant, str(basis), dft_plus_u)
    kmesh = None
    is_periodic_gaussian = is_periodic and needs_basis
    if is_periodic_gaussian:
        if sr_image_precision is not None:
            try:
                sr_image_precision = float(sr_image_precision)
            except (OverflowError, TypeError, ValueError) as exc:
                raise ValueError(
                    "run_neb: sr_image_precision must be finite and in (0, 1) "
                    f"or None; got {sr_image_precision!r}"
                ) from exc
            if (
                not np.isfinite(sr_image_precision)
                or not 0.0 < sr_image_precision < 1.0
            ):
                raise ValueError(
                    "run_neb: sr_image_precision must be finite and in (0, 1) "
                    f"or None; got {sr_image_precision!r}"
                )
        if neb_dim != 3:
            raise NotImplementedError(
                "run_neb: periodic Gaussian BIPOLE NEB requires a 3-D "
                f"periodic system (dim=3); got dim={neb_dim}."
            )
        if dispersion_params is not None:
            raise NotImplementedError(
                "run_neb: dispersion_params is not supported for periodic "
                "Gaussian BIPOLE NEB because periodic D3 image energies "
                "and gradients are not implemented."
            )
        basis_name = str(basis).strip().lower()
        if basis_name.startswith("pob-"):
            raise NotImplementedError(
                "run_neb: POB basis families are not supported for periodic "
                "Gaussian BIPOLE NEB because this route does not resolve "
                "their possible inline ECP data before constructing the "
                "BIPOLE Hamiltonian. Use an all-electron non-POB basis."
            )

        from types import SimpleNamespace

        from .pbc_bipole_common import (
            reject_bipole_ecp_options,
            reject_bipole_lone_non_gamma_kpoint,
            reject_bipole_solver_options,
            validate_bipole_kmesh,
        )

        scf_options = selected_scf_options
        if float(getattr(scf_options, "smearing_temperature", 0.0) or 0.0) > 0.0:
            raise NotImplementedError(
                "run_neb: finite-temperature periodic Gaussian BIPOLE NEB "
                "is unavailable. The image objective would be the Mermin "
                "free energy, but the NEB/QVF result schema currently labels "
                "it only as generic energy. Use T=0 occupations until the "
                "objective kind and internal energy are represented explicitly."
            )
        if getattr(kpoints, "smearing", None) is not None:
            raise NotImplementedError(
                "run_neb: KPoints.smearing metadata is not supported for "
                "periodic Gaussian BIPOLE NEB. This route materializes only "
                "the k-point vectors and weights, so accepting the metadata "
                "would silently run a different occupation objective."
            )
        if getattr(kpoints, "bz_integration", None) is not None:
            raise NotImplementedError(
                "run_neb: KPoints.bz_integration metadata is not supported "
                "for periodic Gaussian BIPOLE NEB. This route materializes "
                "only the k-point vectors and weights, so accepting the "
                "metadata would silently drop the requested BZ integrator."
            )
        preflight_options = (
            scf_options if scf_options is not None else SimpleNamespace()
        )
        kmesh = _periodic_gaussian_neb_kmesh(reactant, kpoints)
        reject_bipole_solver_options(preflight_options, driver="run_neb")
        (
            _stored_kpoints,
            _mesh_size,
            true_multik,
            _is_ibz,
        ) = validate_bipole_kmesh(
            kmesh,
            driver="run_neb",
            require_complete=True,
        )
        reject_bipole_lone_non_gamma_kpoint(kmesh, driver="run_neb")

        from ._vibeqc_core import InitialGuess, SpinlockMode

        _periodic_neb_guess = getattr(
            preflight_options, "initial_guess", InitialGuess.AUTO
        )
        if _periodic_neb_guess in (InitialGuess.READ, InitialGuess.FRAGMO):
            _periodic_neb_guess_name = (
                "READ"
                if _periodic_neb_guess == InitialGuess.READ
                else "FRAGMO"
            )
            raise NotImplementedError(
                "run_neb: periodic Gaussian BIPOLE NEB does not support "
                f"InitialGuess.{_periodic_neb_guess_name}. The image and "
                "finite-difference "
                "displacement loop has no geometry-projected full-lattice "
                "density/fragment seam, so accepting it would either fail "
                "after dry-run or seed a different multi-k state. Run the "
                "requested guess as a supported single point first, then "
                "start NEB from SAD or HCORE; UHF/UKS may also use PATOM or "
                "ATOMSPIN."
            )
        if (
            method_lower in ("uhf", "uks")
            and getattr(preflight_options, "spinlock_mode", SpinlockMode.OFF)
            == SpinlockMode.SPIN_SCHEDULE
            and true_multik
        ):
            raise NotImplementedError(
                "run_neb: periodic Gaussian BIPOLE SPIN_SCHEDULE is "
                "Gamma-only because its phase-2 restart cannot reconstruct "
                "the complete multi-k spin density. Use PATTERN_HOLD or a "
                "Gamma mesh."
            )
        reject_bipole_ecp_options(
            preflight_options,
            driver="run_neb",
            basis=SimpleNamespace(name=basis_name),
            system=reactant,
        )
    if (is_mace or is_semiempirical) and dft_plus_u:
        raise ValueError(
            f"run_neb: dft_plus_u is not supported with method={method!r} "
            "(the Hubbard +U correction applies to the Gaussian SCF/KS methods "
            "RHF / UHF / RKS / UKS only)."
        )
    if method_lower in ("rohf", "roks") and dft_plus_u:
        raise NotImplementedError(
            "run_neb: DFT+U is not implemented for ROHF/ROKS reaction paths."
        )
    if is_semiempirical and is_periodic and dispersion_params is not None:
        raise NotImplementedError(
            "run_neb: periodic semiempirical D3 image sums are not implemented; "
            "dispersion_params cannot be used on this boundary."
        )
    if is_seccm and dispersion_params is not None:
        raise NotImplementedError(
            "run_neb: SECCM dispersion image ownership is not implemented."
        )
    if is_semiempirical and is_periodic and not is_seccm:
        if (
            not np.isfinite(semiempirical_cutoff_bohr)
            or semiempirical_cutoff_bohr <= 0.0
        ):
            raise ValueError(
                "run_neb: semiempirical_cutoff_bohr must be finite and "
                f"positive; got {semiempirical_cutoff_bohr!r}"
            )
    if is_seccm:
        resolved_method = "dftb0" if is_dftb0_seccm else "seccm"
    elif semiempirical_plan is not None:
        resolved_method = (
            "upm6"
            if semiempirical_plan.variant == "upm6"
            else semiempirical_plan.method_key
        )
    else:
        resolved_method = method_lower

    n_atoms = len(_positions_of(reactant))
    frozen_mask: Optional[np.ndarray] = None
    if freeze_indices is not None:
        fi = {int(i) for i in freeze_indices}
        bad = [i for i in fi if i < 0 or i >= n_atoms]
        if bad:
            raise ValueError(
                f"run_neb: freeze_indices {bad} out of range "
                f"[0, {n_atoms})"
            )
        frozen_mask = np.zeros(n_atoms, dtype=bool)
        for i in fi:
            frozen_mask[i] = True

    if dry_run or is_dry_run_requested():
        estimate_bytes = (
            _neb_dry_run_estimate_bytes(
                reactant,
                basis=basis,
                method_lower=method_lower,
                n_images=n_images,
                n_jobs=n_jobs,
                n_atoms=n_atoms,
                is_periodic=is_periodic,
                warm_start=warm_start,
                rhf_options=rhf_options,
                uhf_options=uhf_options,
                rks_options=rks_options,
                uks_options=uks_options,
            )
            if is_dry_run_estimate_requested()
            else None
        )
        _write_neb_dry_run_manifest(
            output=output,
            method=(
                "dftb0_seccm"
                if is_dftb0_seccm
                else "msindo-nddo"
                if (
                    semiempirical_plan is not None
                    and semiempirical_plan.method_key == "msindo"
                    and semiempirical_plan.variant == "nddo"
                )
                else resolved_method
            ),
            basis=None if is_semiempirical else basis,
            functional=(
                effective_functional
                if method_lower in ("rks", "uks", "roks")
                else None
            ),
            record_hostname=record_hostname,
            estimate_bytes=estimate_bytes,
        )
        return None

    # DFT+U coverage matrix (full surface; no periodic guard
    # remains as of v0.9.0):
    #   * Molecular RHF / UHF / RKS / UKS -- Options-side
    #     (`dft_plus_u_sites` populated by
    #     `_apply_dft_plus_u_to_options`).
    #   * Periodic RHF / UHF / RKS / UKS via BIPOLE -- kwarg-side
    #     (`run_pbc_bipole_{rhf,uhf,rks,uks}(..., dft_plus_u=[...])`,
    #     all four landed on `main`).
    # The periodic worker `_evaluate_image_periodic` builds the
    # forwarded kwargs dict below -- it now fires for any method,
    # not just open-shell.

    # --- 1. Initial path ---------------------------------------------------
    if interpolation == "idpp":
        initial_systems = interpolate_idpp(reactant, product, n_images)
    elif interpolation == "linear":
        initial_systems = interpolate_linear(reactant, product, n_images)
    else:
        raise ValueError(
            f"interpolation must be 'idpp' or 'linear'; got {interpolation!r}"
        )

    positions: list[np.ndarray] = [
        _positions_of(s) for s in initial_systems
    ]

    # DFT+U setup (molecular path only). The molecular
    # ``run_*_scf_with_jk`` entry points read the Dudarev fields
    # off Options, so we apply once here using a reactant-built
    # basis and reuse for every per-image SCF.
    # AO grouping depends only on basis structure (geometry-
    # invariant for a fixed basis name + atom ordering).
    # Periodic UHF/UKS BIPOLE drivers consume ``dft_plus_u`` as a
    # direct kwarg (not via Options), so they bypass this block
    # -- the periodic worker forwards ``dft_plus_u`` per-call.
    if dft_plus_u and not is_periodic:
        from ._vibeqc_core import (
            BasisSet,
            RHFOptions,
            RKSOptions,
            UHFOptions,
            UKSOptions,
        )
        from .dft_plus_u import _apply_dft_plus_u_to_options

        _reactant_basis = BasisSet(reactant, basis)
        if method_lower == "rhf":
            if rhf_options is None:
                rhf_options = RHFOptions()
            _apply_dft_plus_u_to_options(
                rhf_options, _reactant_basis, dft_plus_u
            )
        elif method_lower == "uhf":
            if uhf_options is None:
                uhf_options = UHFOptions()
            _apply_dft_plus_u_to_options(
                uhf_options, _reactant_basis, dft_plus_u
            )
        elif method_lower == "rks":
            if rks_options is None:
                rks_options = RKSOptions()
                from .runner import _apply_grid_level

                _apply_grid_level(rks_options.grid, grid_level)
            _apply_dft_plus_u_to_options(
                rks_options, _reactant_basis, dft_plus_u
            )
        elif method_lower == "uks":
            if uks_options is None:
                uks_options = UKSOptions()
                from .runner import _apply_grid_level

                _apply_grid_level(uks_options.grid, grid_level)
            _apply_dft_plus_u_to_options(
                uks_options, _reactant_basis, dft_plus_u
            )

    # --- 2. Evaluate endpoints once + cache --------------------------------
    mace_citation = ""
    if is_mace:
        # Load the MACE model once and reuse the calculator for every image.
        # Per-image work is a single forward pass (analytic energy + forces),
        # so the band is evaluated serially rather than pickling the torch
        # model across joblib processes. No SCF, no k-mesh, no basis.
        _mace_calc, _mace_numbers, _mace_cell, mace_citation = _load_mace_model(
            reactant, mlip_options, is_periodic
        )
        eval_kwargs: dict[str, Any] = {
            "calc": _mace_calc,
            "numbers": _mace_numbers,
            "cell": _mace_cell,
        }
        evaluator = _evaluate_image_mace
        n_jobs = 1
    elif is_seccm:
        eval_kwargs = {
            "template": reactant,
            "route_plan": semiempirical_plan,
            "ccm_options": ccm_options,
            "seccm_topology": seccm_topology,
            "seccm_max_tie_score_excursion": (
                seccm_max_tie_score_excursion
            ),
            "dispersion_params": None,
        }
        evaluator = _evaluate_image_semiempirical
    elif is_msindo:
        # MSINDO (molecular, INDO over Slater orbitals). Composition is fixed
        # across the band, so the atomic numbers + charge + multiplicity are
        # read once from the reactant and reused for every image. No basis /
        # functional / k-mesh. Stateless engine => keep the user's ``n_jobs``
        # (parallel across images), unlike MACE's live-torch serial path.
        eval_kwargs = {
            "numbers": _atomic_numbers_of(reactant),
            "charge": int(reactant.charge),
            "multiplicity": int(reactant.multiplicity),
            "fd_step_bohr": fd_step_bohr,
            "dispersion_params": dispersion_params,
        }
        evaluator = _evaluate_image_msindo
    elif is_semiempirical and is_periodic:
        eval_kwargs = {
            "template": reactant,
            "route_plan": semiempirical_plan,
            "kpoints": (
                kpoints
                if semiempirical_plan.boundary == BOUNDARY_PERIODIC_K
                else None
            ),
            "cutoff_bohr": semiempirical_cutoff_bohr,
            "fd_step_bohr": fd_step_bohr,
        }
        evaluator = _evaluate_image_periodic_semiempirical
    elif is_semiempirical:
        eval_kwargs = {
            "template": reactant,
            "route_plan": semiempirical_plan,
            "ccm_options": None,
            "dispersion_params": dispersion_params,
        }
        evaluator = _evaluate_image_semiempirical
    elif is_periodic:
        eval_kwargs = {
            "template": reactant,
            "basis_name": basis,
            "method": method_lower,
            "kmesh": kmesh,
            "functional": effective_functional,
            "rhf_options": rhf_options,
            "uhf_options": uhf_options,
            "rks_options": rks_options,
            "uks_options": uks_options,
            "fd_step_bohr": fd_step_bohr,
            "sr_image_precision": sr_image_precision,
            "dft_plus_u": dft_plus_u,
        }
        evaluator = _evaluate_image_periodic
    else:
        eval_kwargs = {
            "template": reactant,
            "basis_name": basis,
            "method": method_lower,
            "functional": effective_functional,
            "rhf_options": rhf_options,
            "uhf_options": uhf_options,
            "rks_options": rks_options,
            "uks_options": uks_options,
            "rohf_options": rohf_options,
            "roks_options": roks_options,
            "grid_level": grid_level,
            "gradient_options": gradient_options,
            "grid_options": grid_options,
            "dispersion_params": dispersion_params,
            "fd_step_bohr": fd_step_bohr,
            "dft_plus_u": dft_plus_u,
        }
        evaluator = _evaluate_image
    e_R, g_R, _d_R = evaluator(positions[0], image_index=0, **eval_kwargs)
    e_P, g_P, _d_P = evaluator(
        positions[-1], image_index=len(positions) - 1, **eval_kwargs
    )

    energies: list[float] = [e_R] + [0.0] * n_images + [e_P]
    gradients: list[np.ndarray] = (
        [g_R] + [np.zeros((n_atoms, 3))] * n_images + [g_P]
    )

    # Per-image density cache for within-image SCF warm-start. Molecular
    # RHF/RKS return one density and UHF/UKS/ROHF/ROKS return a per-spin pair;
    # the evaluator passes that shape back to the matching SCF seam.
    # ``warm_start`` defaults to True; pass False to force cold starts.
    image_densities: list[Optional[np.ndarray]] = [None] * (n_images + 2)

    # --- 3. Outer loop (quick-min) -----------------------------------------
    from joblib import Parallel, delayed

    velocities: list[np.ndarray] = [
        np.zeros((n_atoms, 3)) for _ in range(n_images)
    ]
    step = initial_step
    max_force = float("inf")
    converged = False
    n_iter = 0
    # CI-NEB warm-up: the band needs to be roughly settled before
    # we promote its highest-energy image to climbing; otherwise the
    # climbing-image selection can flip between outer iterations and
    # destabilise the loop.
    climbing_warmup_iters = (
        int(round(max_iter * climbing_image_start_fraction))
        if climbing_image
        else 0
    )
    climbing_index: Optional[int] = None

    for outer in range(max_iter):
        n_iter = outer + 1
        # Per-image warm-start density: pass last iter's converged
        # density when ``warm_start`` is on; otherwise pass None
        # which falls back to the cold initial guess (SAD/Hcore).
        results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(evaluator)(
                positions[i],
                **eval_kwargs,
                initial_density=image_densities[i] if warm_start else None,
                image_index=i,
            )
            for i in range(1, n_images + 1)
        )
        for k, (e_k, g_k, d_k) in enumerate(results):
            energies[k + 1] = e_k
            gradients[k + 1] = g_k
            image_densities[k + 1] = d_k  # None for non-RHF; harmless.

        # Promote a climbing image once we are out of the warm-up
        # phase. Henkelman+Uberuaga+Jónsson 2000: pick the highest-
        # energy intermediate image at the moment of promotion;
        # keep that selection fixed for the remainder of the run so
        # the climber doesn't lose its momentum to a re-selection.
        if (
            climbing_image
            and climbing_index is None
            and outer >= climbing_warmup_iters
        ):
            climbing_index = int(np.argmax(energies[1:-1])) + 1

        forces, tangents = _neb_forces(
            positions,
            energies,
            gradients,
            spring_constant,
            frozen_mask,
            climbing_index=climbing_index,
            lattice=neb_lattice,
            dim=neb_dim,
        )
        max_force = max(float(np.max(np.abs(f))) for f in forces)

        if progress:
            ts_idx = int(np.argmax(energies[1:-1])) + 1
            path_len = sum(
                float(np.linalg.norm(positions[i + 1] - positions[i]))
                for i in range(len(positions) - 1)
            )
            climb_tag = (
                f" [CI={climbing_index}]" if climbing_index is not None else ""
            )
            write(
                f"neb iter {n_iter:3d}{climb_tag}: max|F| = {max_force:.4e} "
                f"Ha/bohr, E_TS = {energies[ts_idx]:.6f} Ha, "
                f"path length = {path_len:.3f} bohr\n"
            )

        if max_force < conv_tol_force:
            converged = True
            break

        # On the last allowed iteration, ``energies`` and ``gradients``
        # describe the positions evaluated above. Do not advance to an
        # unevaluated R_(N+1) and then attach the R_N observables to it.
        if outer + 1 == max_iter:
            break

        # Quick-min velocity update + step. Each intermediate image:
        #   v <- v + Δt F
        #   if v.F < 0: v <- 0
        #   else:       v <- (v.F̂) F̂           (project onto F direction)
        #   R <- R + Δt v
        # Adaptive Δt: grow by 1.1 when aligned, reset on flips. This
        # is the standard MDMin / quick-min recipe (Henkelman+Jónsson
        # 2000 Sec. III.C and ASE's MDMin optimiser).
        aligned = True
        for k in range(n_images):
            f_k = forces[k]
            f_flat = f_k.ravel()
            f_norm = float(np.linalg.norm(f_flat))
            if f_norm < 1e-15:
                velocities[k] = np.zeros_like(f_k)
                continue
            v_k = velocities[k] + step * f_k
            dot = float(np.sum(v_k * f_k))
            if dot < 0.0:
                v_k = np.zeros_like(v_k)
                aligned = False
            else:
                # Project velocity onto force direction.
                f_hat = f_k / f_norm
                v_k = float(np.sum(v_k * f_hat)) * f_hat
            # Cap displacement: |Δx| <= max_step per atom.
            disp = step * v_k
            disp_norm = float(np.max(np.abs(disp)))
            if disp_norm > max_step:
                disp = disp * (max_step / disp_norm)
                v_k = np.zeros_like(v_k)
            velocities[k] = v_k
            positions[k + 1] = positions[k + 1] + disp
        # Grow step when every image stayed aligned; reset on any
        # flip. Cap at 10x initial_step to keep the integrator stable
        # near convergence.
        if aligned:
            step = min(step * 1.1, 10.0 * initial_step)
        else:
            step = initial_step

    # --- 4. Build NEBResult ------------------------------------------------
    images: list[NEBImage] = []
    final_systems = [
        _rebuild_with_positions(reactant, p) for p in positions
    ]
    final_systems[0] = reactant
    final_systems[-1] = product
    # Tangents for endpoints aren't defined; populate intermediates only.
    forces_for_tangents, tangents = _neb_forces(
        positions,
        energies,
        gradients,
        spring_constant,
        frozen_mask,
        climbing_index=climbing_index,
        lattice=neb_lattice,
        dim=neb_dim,
    )
    for i, s in enumerate(final_systems):
        img = NEBImage(system=s)
        img.energy = energies[i]
        img.gradient = gradients[i].copy()
        if 1 <= i <= n_images:
            img.tangent = tangents[i - 1]
        images.append(img)
    path = NEBPath(
        images=images,
        spring_constant=spring_constant,
        climbing_image_index=climbing_index,
    )

    energies_arr = np.array(energies, dtype=float)
    ts_index: Optional[int] = None
    if n_images >= 1:
        # Highest-energy intermediate image (excluding endpoints).
        inner = energies_arr[1:-1]
        ts_index = int(np.argmax(inner)) + 1

    return NEBResult(
        path=path,
        energies=energies_arr,
        converged=converged,
        transition_state_index=ts_index,
        n_iter=n_iter,
        max_force=max_force,
        method=resolved_method,
        basis=None if is_semiempirical else basis,
        functional=(
            effective_functional
            if method_lower in ("rks", "uks", "roks")
            else None
        ),
        is_periodic=is_periodic,
        used_dft_plus_u=bool(dft_plus_u),
        mace_model_citation=(mace_citation or None),
        semiempirical_route_method=(
            "dftb0_seccm" if is_dftb0_seccm else None
        ),
        semiempirical_variant=(
            semiempirical_plan.variant
            if semiempirical_plan is not None
            else None
        ),
        semiempirical_citation_entries=(
            tuple(
                semiempirical_plan.citation_assemble_kwargs.get(
                    "extra_entries", ()
                )
            )
            if semiempirical_plan is not None
            else ()
        ),
    )


__all__ = [
    "NEBImage",
    "NEBPath",
    "NEBResult",
    "NEBImageSCFError",
    "interpolate_linear",
    "interpolate_idpp",
    "run_neb",
]
