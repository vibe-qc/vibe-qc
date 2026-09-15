"""Cyclic Cluster Model (CCM) geometry for periodic MSINDO.

The CCM replaces an infinite crystal by a finite *cyclic* cluster (a supercell)
whose every atom carries a Wigner-Seitz (WS) cell describing its periodic
environment.  The Hamiltonian-independent image topology lives in
``vibeqc.semiempirical.seccm.topology``; this module supplies the MSINDO family
adapter for its Fock, repulsion, Madelung, SCF, energy, and derivative terms.

Reference: Bredow, Geudtner & Jug, J. Comput. Chem. 22, 89 (2001); Peintinger &
Bredow, J. Comput. Chem. 35, 839 (2014) (the cluster-choice problem).

The defining validity rule (``neighbors.f:413``): for every cluster atom the
summed WS weight ``round(WTOT)`` must equal ``NATOMS-1`` -- every *other* atom
appears exactly once (counting fractional shares) in its WS cell.  A cluster
that violates this is not a valid cyclic cluster (MSINDO aborts with
``*** ERROR: WRONG NUMBER OF NEIGHBORS ***``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
from scipy.special import erf, erfc, erfcx

from ..seccm.topology import (
    WSNeighbor,
    WignerSeitzCells,
    _rund,
    build_wigner_seitz,
)
from . import msindo
from .msindo import (
    ANGSTROM_TO_BOHR,
    _atom_blocks,
    _gamma_shell,
    _pair_blocks,
    ateng,
    eff_core_charge,
    eneg,
    n_basis,
    one_center_gmunu,
)
from .msindo_ccm_stability import scf_rhf_ccm

# Local-frame shell label for each of the 9 STO basis functions (s, 3xp, 5xd) --
# used to look up the monopole two-centre g per orbital pair.
_SHELLS = ("s", "p", "p", "p", "d", "d", "d", "d", "d")


@lru_cache(maxsize=1)
def _cpp_ccm_energy_kernel():
    try:
        from vibeqc._vibeqc_core.semiempirical import indo as _indo
    except ImportError:
        return None
    try:
        run_ccm_cpp = _indo.run_ccm
        load_params_from_json = _indo.load_params_from_json
    except AttributeError:
        return None
    params = load_params_from_json(
        Path(__file__).with_name("msindo_params.json").read_text()
    )
    return run_ccm_cpp, params


@lru_cache(maxsize=1)
def _cpp_ccm_gradient_fd_kernel():
    try:
        from vibeqc._vibeqc_core.semiempirical import indo as _indo
    except ImportError:
        return None
    try:
        ccm_gradient_fd_cpp = _indo.ccm_gradient_fd
        load_params_from_json = _indo.load_params_from_json
    except AttributeError:
        return None
    params = load_params_from_json(
        Path(__file__).with_name("msindo_params.json").read_text()
    )
    return ccm_gradient_fd_cpp, params


def _require_supported_msindo_elements(atomic_numbers) -> None:
    missing = sorted({z for z in atomic_numbers if z not in msindo._SUPPORTED})
    if missing:
        raise NotImplementedError(
            f"MSINDO engine supports {sorted(msindo._SUPPORTED)}; got Z={missing}."
        )


def _ccm_translations_angstrom_arrays(translations_angstrom) -> list[np.ndarray]:
    if translations_angstrom is None:
        raise ValueError(
            "CCM translations_angstrom must be 1, 2, or 3 three-component "
            "numeric vectors in Angstrom."
        )
    try:
        raw_vectors = list(translations_angstrom)
    except TypeError as exc:
        raise ValueError(
            "CCM translations_angstrom must be 1, 2, or 3 three-component "
            "numeric vectors in Angstrom."
        ) from exc
    if len(raw_vectors) not in (1, 2, 3):
        raise ValueError(
            "CCM translations_angstrom must contain 1, 2, or 3 vectors."
        )
    vectors: list[np.ndarray] = []
    for index, vector in enumerate(raw_vectors, start=1):
        try:
            array = np.asarray(vector, float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "CCM translations_angstrom vectors must have exactly three "
                f"numeric components; vector {index} could not be converted."
            ) from exc
        if array.shape != (3,):
            raise ValueError(
                "CCM translations_angstrom vectors must have exactly three "
                f"numeric components; vector {index} has shape {array.shape}."
            )
        vectors.append(array)
    return vectors


# --------------------------------------------------------------------------- #
# Periodic INDO build + SCF
# --------------------------------------------------------------------------- #


@dataclass
class CCMOptions:
    """Options for a periodic CCM job through ``run_job(method="ccm", ...)``.

    ``translations`` are the 1-3 supercell lattice vectors in Angstrom (1 -> a
    polymer/CCM1D, 2 -> a surface/CCM2D, 3 -> bulk/CCM3D).  ``madelung=True`` adds
    the long-range Ewald/Madelung embedding (recommended for ionic systems).
    ``cluster_radius`` is accepted as a future-facing alias, but ``run_job`` still
    needs explicit translations today.

    Scope through ``run_job`` (each limit is a clean error, never a silent
    wrong answer -- CLAUDE.md Sec.7):

    * **Closed-shell (RHF) only.** A cell with ``multiplicity != 1`` raises
      ``NotImplementedError``; open-shell / ROHF CCM is not implemented.
    * **Single-point only.** ``optimize=True`` raises ``NotImplementedError``;
      relax CCM geometries with
      :func:`vibeqc.semiempirical.methods.msindo_ccm.ccm_optimize`
      (finite-difference, fixed Wigner-Seitz) instead.
    * **Elements per the MSINDO engine** (currently H-Xe, ``msindo._SUPPORTED``).
      A heavier element raises ``NotImplementedError`` naming the supported set.
    """

    translations: list | None = None
    madelung: bool = True
    cluster_radius: float | None = None


@dataclass
class CcmResult:
    """Result of a periodic-CCM MSINDO SCF (energies in Hartree)."""

    total_energy: float = 0.0
    electronic_energy: float = 0.0
    binding_energy: float = 0.0
    madelung_nuclear_energy: float = 0.0
    mo_energies: np.ndarray = field(default=None)
    density: np.ndarray = field(default=None)
    n_iter: int = 0
    converged: bool = False
    stability_checked: bool = False
    stability_analysis_converged: bool = False
    stability_eigenvalue: float = 0.0
    n_stability_restarts: int = 0


def _build_core_and_gamma_ccm(Z, coords, blocks, nsto, ws: WignerSeitzCells):
    """Periodic core Hamiltonian ``H`` and two-electron ``GMUNU`` for the CCM.

    Structurally the molecular build (``msindo._build_core_and_gamma``) with the
    two-centre terms replaced by Wigner-Seitz-weighted sums over each atom's
    neighbour images -- the port of ``ccmintov.f``: every central atom K
    accumulates the core attraction ``HK1`` from each WS neighbour (weighted),
    and for the smaller-index member of each pair the resonance ``HKL2`` and the
    monopole Coulomb ``g`` go into the off-diagonal block (the other ordering and
    the upper triangle follow by transpose).  One-centre blocks (ENEG diagonal +
    ``gij1`` 9x9) are exactly the molecular ones.
    """
    H = np.zeros((nsto, nsto))
    G = np.zeros((nsto, nsto))

    # One-centre blocks -- identical to the molecular engine.
    for i, z in enumerate(Z):
        lo, hi = blocks[i]
        nb = hi - lo
        u = eneg(z)
        H[lo, lo] += u[0]
        if nb >= 4:
            for p in range(1, 4):
                H[lo + p, lo + p] += u[1]
        if nb >= 9:
            for d in range(4, 9):
                H[lo + d, lo + d] += u[2]
        G[lo:hi, lo:hi] = one_center_gmunu(z)[:nb, :nb]

    # Two-centre periodic terms -- WS-weighted sum over neighbour images.
    for k in range(len(Z)):
        lk, hk = blocks[k]
        nk = hk - lk
        for nb in ws.cells[k]:
            j = nb.origin
            w = nb.weight
            lj, hj = blocks[j]
            nj = hj - lj
            pb = _pair_blocks(Z[k], Z[j], coords[k], coords[k] + nb.disp)
            # core attraction on the central atom K from this image (HK1 only --
            # atom J's own block is filled when J is the central atom).
            H[lk:hk, lk:hk] += w * pb.HK1[:nk, :nk]
            if k < j:
                # resonance K-J and monopole g K-J (built once, from K<J).
                H[lk:hk, lj:hj] += w * pb.HKL2[:nk, :nj]
                H[lj:hj, lk:hk] += w * pb.HKL2[:nk, :nj].T
                R = pb.R
                gcache: dict[tuple[str, str], float] = {}
                for ia in range(nk):
                    for ib in range(nj):
                        key = (_SHELLS[ia], _SHELLS[ib])
                        if key not in gcache:
                            gcache[key] = _gamma_shell(Z[k], Z[j], key[0], key[1], R)
                        G[lk + ia, lj + ib] += w * gcache[key]
                        G[lj + ib, lk + ia] += w * gcache[key]
    return H, G


def _ewald_ws_cells(ws: WignerSeitzCells):
    """The Ewald WS neighbour set (``neighborewald.f``): the Fock WS neighbours
    of each atom *plus the atom itself* (weight 1.0, zero displacement) -- the
    Madelung potential at I includes the periodic images of I's own charge.  The
    comment in ``ccm1dmadelsum.f`` ("EWALDWSZ enthält das Zentralatom") is this
    difference from the Fock WS set."""
    return [list(cell) + [WSNeighbor(origin=i, weight=1.0, disp=np.zeros(3))]
            for i, cell in enumerate(ws.cells)]


def _madelung_potential_1d(net_charges, ews, translation):
    """1-D Madelung potential at every atom (``ccm1dmadelsum.f``).

    A finite point-charge lattice sum: for each Ewald-WS neighbour (origin charge
    ``q``, weight ``w``, displacement ``d`` from the central atom) it adds the
    potential of that charge's ±1 and ±2 translational images,
    ``S w.q.(1/|d+T| + 1/|d-T| + 1/|d+2T| + 1/|d-2T|)`` (the in-cell zeroth
    shell is already in the INDO two-centre g).  ``T`` is the 1-D lattice vector
    (bohr).  The cluster must be electroneutral for this truncated sum to be
    meaningful (``ewaldcharges.f``)."""
    T = np.asarray(translation, float)
    shells = [T, -T, 2.0 * T, -2.0 * T]
    pot = np.zeros(len(ews))
    for i, cell in enumerate(ews):
        s = 0.0
        for nb in cell:
            d = nb.disp
            contrib = sum(1.0 / np.linalg.norm(d + sh) for sh in shells)
            s += nb.weight * net_charges[nb.origin] * contrib
        pot[i] = s
    return pot


def _reciprocal_basis(translations):
    """Reciprocal lattice vectors ``b_i`` (with the 2pi factor, so K = S n_i.b_i)
    and the cell volume, from the 3 direct lattice vectors."""
    a1, a2, a3 = (np.asarray(t, float) for t in translations)
    vol_signed = float(np.dot(a1, np.cross(a2, a3)))
    twopi = 2.0 * np.pi
    b1 = twopi * np.cross(a2, a3) / vol_signed
    b2 = twopi * np.cross(a3, a1) / vol_signed
    b3 = twopi * np.cross(a1, a2) / vol_signed
    return np.array([b1, b2, b3]), abs(vol_signed)


_EWALD_3D_MAX_CANDIDATES = 500_000


def _checked_ewald_bounds_3d(unrounded_bounds, *, space):
    values = np.asarray(unrounded_bounds, dtype=float)
    if values.shape != (3,) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(
            f"3-D CCM Ewald {space} bounds are not finite; use a reduced "
            "lattice basis"
        )
    bounds = tuple(math.ceil(float(value)) + 1 for value in values)
    candidate_vectors = math.prod(2 * bound + 1 for bound in bounds)
    if candidate_vectors > _EWALD_3D_MAX_CANDIDATES:
        raise ValueError(
            f"3-D CCM Ewald {space} enumeration exceeds the safe candidate "
            "budget; use a reduced lattice basis"
        )
    return np.asarray(bounds, dtype=int)


def _ewald_lattice_3d(ews, translations):
    """Return a cutoff-complete 3-D Ewald vector inventory.

    The Ewald formulas sum physical vectors, not coefficient tuples.  Bounds
    derived from the direct/reciprocal dual plane heights include every vector
    inside the ``exp(-30)`` convergence spheres even when an equivalent
    unimodular basis represents it with large near-cancelling coefficients.
    The direct bounds also include every fractional Ewald-WS displacement so a
    shifted atom pair cannot move an in-cutoff vector outside the coefficient
    box.
    """
    exponent_cutoff = 30.0
    twopi = 2.0 * np.pi
    direct_basis = np.asarray(translations, dtype=float)
    recip, volume = _reciprocal_basis(translations)
    alpha = np.sqrt(np.pi) / volume ** (1.0 / 3.0)

    reciprocal_cutoff = alpha * np.sqrt(4.0 * exponent_cutoff)
    reciprocal_bounds = _checked_ewald_bounds_3d(
        reciprocal_cutoff * np.linalg.norm(direct_basis, axis=1) / twopi,
        space="reciprocal-space",
    )
    reciprocal_ranges = [
        range(-int(bound), int(bound) + 1) for bound in reciprocal_bounds
    ]
    reciprocal_grid = np.asarray(
        list(product(*reciprocal_ranges)), dtype=float
    )
    reciprocal_grid = reciprocal_grid[np.any(reciprocal_grid != 0.0, axis=1)]
    reciprocal_vectors = reciprocal_grid @ recip
    reciprocal_squared = np.einsum(
        "ij,ij->i", reciprocal_vectors, reciprocal_vectors
    )
    inside = reciprocal_squared <= reciprocal_cutoff**2
    reciprocal_vectors = reciprocal_vectors[inside]
    reciprocal_squared = reciprocal_squared[inside]

    direct_cutoff = np.sqrt(exponent_cutoff) / alpha
    max_fractional_displacement = np.zeros(3)
    for cell in ews:
        for neighbor in cell:
            fractional = recip @ np.asarray(neighbor.disp, dtype=float) / twopi
            max_fractional_displacement = np.maximum(
                max_fractional_displacement, np.abs(fractional)
            )
    direct_bounds = _checked_ewald_bounds_3d(
        direct_cutoff * np.linalg.norm(recip, axis=1) / twopi
        + max_fractional_displacement,
        space="direct-space",
    )
    direct_ranges = [range(-int(bound), int(bound) + 1) for bound in direct_bounds]
    direct_grid = np.asarray(list(product(*direct_ranges)), dtype=float)
    direct_vectors = direct_grid @ direct_basis

    return (
        reciprocal_vectors,
        reciprocal_squared,
        direct_vectors,
        alpha,
        volume,
        direct_cutoff,
    )


def _madkonst_3d(ews, translations, n_atoms):
    """3-D Ewald Madelung-constant matrix ``MADKONST[I, J]`` (``madelkonst.f``):
    the lattice-summed potential at atom I from a unit point charge at atom J and
    all its periodic images, split into reciprocal + direct (erfc) parts with the
    self-term and uniform-background (tin-foil boundary) corrections.  Summed over
    J's Wigner-Seitz images with the Ewald-WS weights.  Geometry-only, so built
    once before the SCF.  The convergence parameter a = √pi / V^(1/3)
    (``CONFAC``)."""
    kvecs, kb, dvecs, alpha, vol, direct_cutoff = _ewald_lattice_3d(
        ews, translations
    )
    faktor = kb / (4.0 * alpha ** 2)
    kpref = np.exp(-faktor) / faktor * np.pi / alpha ** 2 / vol  # xcos(K.r) per K

    self_const = -2.0 * alpha / np.sqrt(np.pi)
    background = -np.pi / vol / alpha ** 2

    mad = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for nb in ews[i]:
            vij = -nb.disp  # C(I) - image position
            recip_term = float(kpref @ np.cos(kvecs @ vij))
            pos = vij + dvecs
            dist = np.linalg.norm(pos, axis=1)
            near = dist < 1e-8
            far = (~near) & (dist <= direct_cutoff)
            direct_term = float(np.sum(erfc(alpha * dist[far]) / dist[far]))
            direct_term += self_const * int(np.count_nonzero(near))
            mad[i, nb.origin] += nb.weight * (recip_term + direct_term + background)
    return mad


_EWALD_2D_MAX_CANDIDATES = 500_000


def _checked_ewald_bounds_2d(unrounded_bounds, *, space):
    values = np.asarray(unrounded_bounds, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError(
            f"2-D CCM Ewald {space} bounds are not finite; use a reduced "
            "lattice basis"
        )
    bounds = tuple(math.ceil(float(value)) + 1 for value in values)
    candidate_vectors = math.prod(2 * bound + 1 for bound in bounds)
    if candidate_vectors > _EWALD_2D_MAX_CANDIDATES:
        raise ValueError(
            f"2-D CCM Ewald {space} enumeration exceeds the safe candidate "
            "budget; use a reduced lattice basis"
        )
    return np.asarray(bounds, dtype=int)


def _surface_basis(translations):
    """In-plane direct/reciprocal basis, unit normal, mesh area and the Parry
    convergence parameter ``a = 0.85.√pi/√A`` for a 2-D surface cell."""
    a1, a2 = (np.asarray(t, float) for t in translations)
    normal = np.cross(a1, a2)
    area = float(np.linalg.norm(normal))
    nhat = normal / area
    twopi = 2.0 * np.pi
    # b_i . a_j = 2pi delta_ij with both b_i in the surface plane (b_i . nhat = 0).
    b1 = twopi * np.cross(a2, nhat) / area
    b2 = twopi * np.cross(nhat, a1) / area
    alpha = 0.85 * np.sqrt(np.pi) / np.sqrt(area)
    return np.array([a1, a2]), np.array([b1, b2]), nhat, area, alpha


def _ewald_lattice_2d(ews, translations):
    """Return a cutoff-complete 2-D (Parry) Ewald vector inventory.

    Both halves of the Parry split converge *absolutely* and exponentially --
    the reciprocal summand carries the factor
    ``exp(-a^2 z^2 - |K|^2/4a^2)`` (see :func:`_parry_recip_pair`) and the
    direct summand ``erfc(a r)/r`` -- so truncating each at ``exp(-30)`` is
    well defined.  The *conditional* convergence Parry 1975 s1 warns about
    lives in the unsplit Coulomb sum, whose value depends on the shape of the
    summation region; it is not a property of eq. (7), and no summation-order
    choice is needed here.

    What this buys, and what the old fixed +/-12 coefficient box did not: the
    two truncated sets are ``{K != 0 : |K| <= K_cut}`` and
    ``{L : |v + L| <= r_cut}``, which are sets of *physical vectors*.  They are
    therefore identical for every unimodular choice of the surface basis, and
    the Madelung constant comes out basis-independent.  A coefficient box is
    not: a large unimodular shear hides a short lattice vector behind a
    coefficient outside the box (issue #187 -- shear 13 needed coefficient 13,
    and the required radius grows with the shear, so no fixed radius is
    correct).  Bounds come from the direct/reciprocal dual metric, exactly as
    :func:`_ewald_lattice_3d` does for the 3-D kernel; the direct bounds also
    absorb every fractional Ewald-WS displacement so a shifted atom pair cannot
    push an in-cutoff vector outside the enumerated box.

    A pathologically unreduced basis whose candidate box would exceed
    ``_EWALD_2D_MAX_CANDIDATES`` fails closed rather than allocating it.
    """
    exponent_cutoff = 30.0
    twopi = 2.0 * np.pi
    direct_basis, recip, nhat, area, alpha = _surface_basis(translations)

    # |K| <= K_cut makes the reciprocal summand's exp(-|K|^2/4a^2) <= exp(-30).
    reciprocal_cutoff = alpha * np.sqrt(4.0 * exponent_cutoff)
    # n_i = K.a_i/2pi, so |n_i| <= K_cut.|a_i|/2pi (Cauchy-Schwarz).
    reciprocal_bounds = _checked_ewald_bounds_2d(
        reciprocal_cutoff * np.linalg.norm(direct_basis, axis=1) / twopi,
        space="reciprocal-space",
    )
    reciprocal_ranges = [
        range(-int(bound), int(bound) + 1) for bound in reciprocal_bounds
    ]
    reciprocal_grid = np.asarray(list(product(*reciprocal_ranges)), dtype=float)
    reciprocal_grid = reciprocal_grid[np.any(reciprocal_grid != 0.0, axis=1)]
    reciprocal_vectors = reciprocal_grid @ recip
    reciprocal_magnitudes = np.linalg.norm(reciprocal_vectors, axis=1)
    inside = reciprocal_magnitudes <= reciprocal_cutoff
    reciprocal_vectors = reciprocal_vectors[inside]
    reciprocal_magnitudes = reciprocal_magnitudes[inside]

    # erfc(a r)/r <= exp(-30)-scale beyond r_cut = sqrt(30)/a.
    direct_cutoff = np.sqrt(exponent_cutoff) / alpha
    max_fractional_displacement = np.zeros(2)
    for cell in ews:
        for neighbor in cell:
            # b_i . disp picks up only the in-plane part of the displacement.
            fractional = recip @ np.asarray(neighbor.disp, dtype=float) / twopi
            max_fractional_displacement = np.maximum(
                max_fractional_displacement, np.abs(fractional)
            )
    direct_bounds = _checked_ewald_bounds_2d(
        direct_cutoff * np.linalg.norm(recip, axis=1) / twopi
        + max_fractional_displacement,
        space="direct-space",
    )
    direct_ranges = [range(-int(bound), int(bound) + 1) for bound in direct_bounds]
    direct_vectors = np.asarray(list(product(*direct_ranges)), dtype=float) @ (
        direct_basis
    )

    return (
        reciprocal_vectors,
        reciprocal_magnitudes,
        direct_vectors,
        alpha,
        area,
        nhat,
        direct_cutoff,
    )


def _parry_recip_pair(kmag, alpha, z):
    """The two Parry reciprocal-space factors, evaluated without overflow.

    Returns ``(e^{|K|z} erfc(az + |K|/2a), e^{-|K|z} erfc(-az + |K|/2a))`` --
    the pair inside eq. (7) of Parry, Surf. Sci. 49, 433 (1975) (= ``F_kl(z)``
    of de Leeuw & Perram, Mol. Phys. 37, 1313 (1979) eq. (11)).  Evaluated as
    written, the ``e^{|K|z}`` factor overflows while its ``erfc`` underflows,
    so the product becomes ``inf * 0 = nan``.  Using
    ``erfc(x) = e^{-x^2} erfcx(x)`` the growing exponentials cancel exactly:

        e^{|K|z} erfc(az + |K|/2a) = e^{-a^2 z^2 - |K|^2/4a^2} erfcx(az + |K|/2a)

    and likewise with ``z -> -z``, which is the same stable form the shared
    slab kernel already uses (``vibeqc::ewald_2d_point_charge_potential``,
    ``cpp/include/vibeqc/ewald.hpp``).  ``erfcx`` is only evaluated at
    non-negative arguments (where it is bounded by 1); the partner factor is
    taken in its already-bounded ``e^{-|K||z|} erfc(...)`` form.  The pair is
    symmetric under ``z -> -z`` with the two members swapped, so it is computed
    at ``|z|`` and swapped back.
    """
    half = kmag / (2.0 * alpha)
    az = alpha * abs(z)
    common = np.exp(-(az * az) - half * half)
    rising = common * erfcx(half + az)
    falling = np.exp(-kmag * abs(z)) * erfc(half - az)
    if z < 0.0:
        return falling, rising
    return rising, falling


def _madkonst_2d(ews, translations, n_atoms):
    """2-D (surface) Ewald Madelung-constant matrix ``MADKONST[I,J]``
    (``madelkonst.f`` CCM2D branch -- the Parry/Heyes 2-D Ewald).  The lattice is
    periodic in the plane of the two translation vectors; the perpendicular
    (out-of-plane) separation ``z`` decays through the ``exp(±|K|z).erfc(...)``
    reciprocal terms and a K=0 term.  a = 0.85.√pi/√A (A = 2-D cell area).

    Eq. (7) of Parry, Surf. Sci. 49, 433 (1975) (erratum Surf. Sci. 54, 195
    (1976)), with the K=0 term Parry drops by cell neutrality restored in the
    per-pair form of de Leeuw & Perram, Mol. Phys. 37, 1313 (1979) eqs. (8) and
    (12); the reciprocal factor is their eq. (11) ``F_kl(z)`` and the direct
    sum with its ``-2a/√pi`` self term is their eqs. (14a)-(14b)::

        M(v) = (pi/A) S_{K!=0} cos(K.v)/|K| . [e^{|K|z} erfc(az + |K|/2a)
                                               + e^{-|K|z} erfc(-az + |K|/2a)]
               - (2pi/A) [z.erf(az) + e^{-a^2 z^2}/(a√pi)]
               + S_L erfc(a r)/r,   r = |v + L|,   z = v.n̂

    Convergence guarantee: both lattice sums are truncated on the *physical*
    vectors inside the ``exp(-30)`` Ewald cutoffs, not on a coefficient box, so
    the result is invariant under any unimodular change of surface basis (that
    is a relabelling of the same lattice, not a change of physics) to ~1e-13.
    Not guaranteed: this is a truncation at ``exp(-30)``, not an exact sum --
    the residual tail is real, uniform across bases, and the reason the
    invariance is pinned at 1e-10 rather than at machine epsilon.  See
    :func:`_ewald_lattice_2d` on why the split is absolutely convergent even
    though the unsplit Coulomb sum is only conditionally so.
    """
    kvecs, kmag, dvecs, alpha, area, nhat, direct_cutoff = _ewald_lattice_2d(
        ews, translations
    )
    self_const = -2.0 * alpha / np.sqrt(np.pi)

    mad = np.zeros((n_atoms, n_atoms))
    for i in range(n_atoms):
        for nb in ews[i]:
            vij = -nb.disp
            zc = float(vij @ nhat)                            # out-of-plane sep
            krij = kvecs @ vij                                # K.R (K⊥nhat)
            rising, falling = _parry_recip_pair(kmag, alpha, zc)
            recip = np.sum(
                np.cos(krij) / kmag * (rising + falling)
            ) * np.pi / area
            # K=0 term
            k0 = -2.0 * np.pi / area * (
                zc * erf(alpha * zc)
                + np.exp(-(alpha ** 2) * zc ** 2) / (alpha * np.sqrt(np.pi)))
            # direct (real-space) 2-D sum
            pos = vij + dvecs
            dist = np.linalg.norm(pos, axis=1)
            near = dist < 1e-8
            far = (~near) & (dist <= direct_cutoff)
            direct = float(np.sum(erfc(alpha * dist[far]) / dist[far]))
            direct += self_const * int(np.count_nonzero(near))
            mad[i, nb.origin] += nb.weight * (float(recip) + k0 + direct)
    return mad


def _madelung_potential_ewald(net_charges, madkonst, ews):
    """Madelung potential at every atom (``madelsum.f`` with ``SMADEL``), for the
    2-D and 3-D Ewald paths: the full Ewald potential ``S_J q_J.MADKONST[I,J]``
    minus the direct 1/r part of the in-WS-cell neighbours (which the INDO
    two-centre g already carries), leaving the long-range tail."""
    full = madkonst @ net_charges
    smadel = np.zeros(len(ews))
    for i, cell in enumerate(ews):
        for nb in cell:
            d = float(np.linalg.norm(nb.disp))
            if d > 1e-8:
                smadel[i] += net_charges[nb.origin] * nb.weight / d
    return full - smadel


def _net_charges(P, blocks, cz):
    """Net atomic charges ``CZ_I - S_{KinI} P(K,K)`` (``ewaldcharges.f``); in the
    Löwdin-orthonormal MSINDO basis the diagonal density is the Mulliken
    population."""
    return np.array([cz[i] - float(np.trace(P[lo:hi, lo:hi]))
                     for i, (lo, hi) in enumerate(blocks)])


def _core_repulsion_ccm(Z, coords, cz, ws: WignerSeitzCells) -> float:
    """Periodic core-core (MADELUNG-NUCLEAR with NOEWALD), the WS-weighted
    point-charge sum ``S_I S_{imginWS(I)} 1/2.CZ_I.CZ_origin.weight / R``
    (``ccmscfclo.f:255-266``; the 1/2 removes the I<->J double counting)."""
    e = 0.0
    for i in range(len(Z)):
        for nb in ws.cells[i]:
            R = float(np.linalg.norm(nb.disp))
            e += 0.5 * cz[i] * cz[nb.origin] * nb.weight / R
    return e


def run_ccm(atomic_numbers, coords_angstrom, translations_angstrom, *,
            charge=0, madelung=False, max_iter=200, conv_tol=1e-9) -> CcmResult:
    """Periodic MSINDO closed-shell (RHF) SCF in the Cyclic Cluster Model.

    ``translations_angstrom`` are the 1-3 supercell lattice vectors (Angstrom).
    With ``madelung=False`` (``NOEWALD``) the long-range electrostatics are
    omitted and the only periodic term beyond the WS-folded INDO Fock is the
    WS-weighted core-core sum.  With ``madelung=True`` the long-range Madelung
    embedding is added self-consistently: the 1-D finite lattice sum follows
    ``ccm1dmadelsum.f``, the 2-D route uses the Parry/Heyes slab kernel, and
    the 3-D route uses the bulk Ewald kernel.
    """
    Z = list(atomic_numbers)
    _require_supported_msindo_elements(Z)

    coords_arr = np.asarray(coords_angstrom, float)
    translations_arr = _ccm_translations_angstrom_arrays(translations_angstrom)
    C = coords_arr * ANGSTROM_TO_BOHR
    T = [t * ANGSTROM_TO_BOHR for t in translations_arr]
    blocks, nsto = _atom_blocks(Z)
    cz = [eff_core_charge(z) for z in Z]
    nelec = sum(cz) - charge
    if nelec % 2 != 0:
        raise NotImplementedError(
            "periodic CCM currently supports closed-shell (even-electron) "
            "clusters only; open-shell CCM is a later increment.")

    ws = build_wigner_seitz(C, T)
    if not ws.is_valid(len(Z)):
        raise ValueError(
            "not a valid cyclic cluster: round(S WS weight) != NATOMS-1 for "
            "some atom (neighbors.f:413). Choose a supercell with no atom on a "
            "WS-cell face -- e.g. an odd-N evenly-spaced chain.")

    if all(1 <= z <= 36 for z in Z):
        kernel = _cpp_ccm_energy_kernel()
        if kernel is not None:
            run_ccm_cpp, params = kernel
            result_obj = run_ccm_cpp(
                Z,
                coords_arr.tolist(),
                [t.tolist() for t in translations_arr],
                params,
                madelung=madelung,
                max_iter=max_iter,
                conv_tol=conv_tol,
                charge=charge,
            )
            if getattr(result_obj, "converged", False):
                density = np.asarray(result_obj.density, dtype=float)
                if madelung:
                    net = _net_charges(density, blocks, cz)
                    if abs(float(np.sum(net))) > 1e-3:
                        raise ValueError(
                            "CCM Madelung needs an electroneutral cluster; net "
                            f"cell charge = {float(np.sum(net)):.4f} "
                            "(ewaldcharges.f)."
                        )
                return CcmResult(
                    total_energy=float(result_obj.total_energy),
                    electronic_energy=float(result_obj.electronic_energy),
                    binding_energy=float(result_obj.binding_energy),
                    madelung_nuclear_energy=(
                        float(result_obj.total_energy)
                        - float(result_obj.electronic_energy)
                    ),
                    mo_energies=np.asarray(result_obj.mo_energies, dtype=float),
                    density=density,
                    n_iter=int(result_obj.n_iter),
                    converged=True,
                    **{name: getattr(result_obj, name, default) for name, default in (
                        ("stability_checked", False),
                        ("stability_analysis_converged", False),
                        ("stability_eigenvalue", 0.0),
                        ("n_stability_restarts", 0),
                    )},
                )
            raise RuntimeError(
                "C++ MSINDO CCM did not converge within the iteration budget"
            )

    nocc = nelec // 2
    diagnostics = {}
    total, e_elec, e_core, eps, P, converged, it = _ccm_total_energy(
        Z, C, blocks, nsto, cz, nocc, ws, T,
        madelung=madelung, max_iter=max_iter, conv_tol=conv_tol,
        diagnostics=diagnostics)

    if madelung:
        net = _net_charges(P, blocks, cz)
        if abs(float(np.sum(net))) > 1e-3:
            raise ValueError(
                f"CCM Madelung needs an electroneutral cluster; net cell charge "
                f"= {float(np.sum(net)):.4f} (ewaldcharges.f).")

    binding = total - sum(ateng(z) for z in Z)
    return CcmResult(
        total_energy=total, electronic_energy=e_elec, binding_energy=binding,
        madelung_nuclear_energy=e_core, mo_energies=eps, density=P,
        n_iter=it, converged=converged, **diagnostics)


def _ccm_total_energy(Z, C, blocks, nsto, cz, nocc, ws, T, *,
                      madelung, max_iter=200, conv_tol=1e-9, diagnostics=None):
    """Total CCM energy for an already-built Wigner-Seitz set ``ws`` at geometry
    ``C`` (bohr).  Factored out of :func:`run_ccm` so the finite-difference
    gradient can hold the WS topology fixed while displacing atoms (matching
    MSINDO's fixed-weight analytic gradient).  Returns
    ``(total, e_elec, e_core, eps, P, converged, n_iter)``."""
    H, G = _build_core_and_gamma_ccm(Z, C, blocks, nsto, ws)
    e_core = _core_repulsion_ccm(Z, C, cz, ws)

    fock_extra = None
    if madelung:
        ews = _ewald_ws_cells(ws)
        madkonst = (_madkonst_2d(ews, T, len(Z)) if len(T) == 2 else
                    _madkonst_3d(ews, T, len(Z)) if len(T) == 3 else None)

        def _potential(net):
            if len(T) == 1:
                return _madelung_potential_1d(net, ews, T[0])
            return _madelung_potential_ewald(net, madkonst, ews)

        def fock_extra(P):
            net = _net_charges(P, blocks, cz)
            mad = _potential(net)
            F_add = np.zeros((nsto, nsto))
            for i, (lo, hi) in enumerate(blocks):
                for k in range(lo, hi):
                    F_add[k, k] = -mad[i]
            e_add = 0.5 * float(np.dot(cz, mad))  # MADELENRG = S 1/2 CZ_I.V_I
            return F_add, e_add

    P, _F, e_elec, eps, converged, it = scf_rhf_ccm(
        H, G, blocks, Z, nocc, max_iter=max_iter, conv_tol=conv_tol,
        fock_extra=fock_extra, diagnostics=diagnostics)
    return e_elec + e_core, e_elec, e_core, eps, P, converged, it


def ccm_gradient_fd(atomic_numbers, coords_angstrom, translations_angstrom, *,
                    charge=0, madelung=False, atoms=None, max_iter=200,
                    conv_tol=1e-10, step=1e-3):
    """Finite-difference nuclear gradient (Ha/bohr) of the CCM total energy.

    The Wigner-Seitz topology is built once at the reference geometry and held
    **fixed** while each atom is displaced (only the periodic-image positions
    follow the atoms) -- this matches MSINDO's fixed-weight analytic CCM gradient
    and avoids the energy kink a WS-membership flip would introduce when an atom
    sits on a cell boundary.  ``step`` is the central-difference displacement in
    Angstrom; tighten ``conv_tol`` for low SCF noise in the differences.
    ``atoms`` (default all) restricts the differentiation to a subset of atom
    indices -- the rest of the returned gradient stays zero -- which makes relaxing
    only an adsorbate cheap.
    """
    Z = list(atomic_numbers)
    _require_supported_msindo_elements(Z)
    coords_arr = np.asarray(coords_angstrom, float)
    translations_arr = _ccm_translations_angstrom_arrays(translations_angstrom)
    if not math.isfinite(step) or step <= 0.0:
        raise ValueError("CCM FD gradient step must be positive and finite.")
    C0 = coords_arr * ANGSTROM_TO_BOHR
    T = [t * ANGSTROM_TO_BOHR for t in translations_arr]
    blocks, nsto = _atom_blocks(Z)
    cz = [eff_core_charge(z) for z in Z]
    nelec = sum(cz) - charge
    if nelec % 2 != 0:
        raise NotImplementedError("CCM FD gradient supports closed-shell only.")
    nocc = nelec // 2
    ws0 = build_wigner_seitz(C0, T)
    if not ws0.is_valid(len(Z)):
        raise ValueError("not a valid cyclic cluster (see run_ccm).")
    h = step * ANGSTROM_TO_BOHR  # displace in bohr -> gradient in Ha/bohr

    # The legacy score is Lipschitz-bounded by |delta r| over the shortest
    # active first-shell vector.  This narrowly padded bound describes this FD
    # batch; it is not a reusable global permission for tied WS images.
    shell_norms = []
    for coefficients in product((-1, 0, 1), repeat=len(T)):
        if all(value == 0 for value in coefficients):
            continue
        vector = sum(
            (value * T[axis] for axis, value in enumerate(coefficients)),
            start=np.zeros(3),
        )
        squared_norm = float(vector @ vector)
        if _rund(squared_norm, 1) != 0.0:
            shell_norms.append(math.sqrt(squared_norm))
    if not shell_norms:
        raise ValueError("CCM translations do not define an active image shell.")
    tie_score_bound = (
        h / min(shell_norms) * (1.0 + 1.0e-10) + np.finfo(float).eps
    )

    atoms_list = None if atoms is None else [int(a) for a in atoms]
    atoms_valid = atoms_list is None or all(0 <= a < len(Z) for a in atoms_list)
    if atoms_valid and all(1 <= z <= 36 for z in Z):
        kernel = _cpp_ccm_gradient_fd_kernel()
        if kernel is not None:
            ccm_gradient_fd_cpp, params = kernel
            grad_cpp = ccm_gradient_fd_cpp(
                Z,
                coords_arr.tolist(),
                [t.tolist() for t in translations_arr],
                params,
                madelung=madelung,
                atoms=[] if atoms_list is None else atoms_list,
                max_iter=max_iter,
                conv_tol=conv_tol,
                step=step,
                charge=charge,
            )
            if len(grad_cpp) == 0 and len(Z) != 0:
                raise RuntimeError("SCF not converged; cannot compute CCM FD gradient.")
            grad_cpp_arr = np.asarray(grad_cpp, dtype=float)
            if (
                grad_cpp_arr.shape == (len(Z), 3)
                and np.all(np.isfinite(grad_cpp_arr))
            ):
                return grad_cpp_arr

    def _energy(C):
        ws = ws0.rebuild_displacements(
            C, max_tie_score_excursion=tie_score_bound
        )
        return _ccm_total_energy(Z, C, blocks, nsto, cz, nocc, ws, T,
                                 madelung=madelung, max_iter=max_iter,
                                 conv_tol=conv_tol)[0]

    which = range(len(Z)) if atoms_list is None else atoms_list
    grad = np.zeros((len(Z), 3))
    for i in which:
        for d in range(3):
            cp = C0.copy(); cp[i, d] += h
            cm = C0.copy(); cm[i, d] -= h
            grad[i, d] = (_energy(cp) - _energy(cm)) / (2.0 * h)
    return grad


def ccm_optimize(atomic_numbers, coords_angstrom, translations_angstrom, *,
                 charge=0, madelung=False, frozen=None, fmax=2e-3,
                 max_steps=200, conv_tol=1e-10):
    """Relax atomic positions (fixed cell) on the CCM energy surface with the
    finite-difference gradient (L-BFGS-B).  ``frozen`` = atom indices held fixed
    (e.g. the substrate slab in an adsorption calculation); the rest are relaxed.
    ``fmax`` is the max-force convergence target (Ha/bohr).  Returns
    ``(relaxed_coords_angstrom, CcmResult at the relaxed geometry)``.

    The WS topology is rebuilt at each step (as MSINDO does during optimization),
    while the gradient holds it fixed within a step (:func:`ccm_gradient_fd`) --
    consistent for an off-lattice adsorbate, which never crosses a cell face."""
    Z = list(atomic_numbers)
    _require_supported_msindo_elements(Z)
    C0 = np.asarray(coords_angstrom, float)
    translations_arr = _ccm_translations_angstrom_arrays(translations_angstrom)
    frozen = set(frozen or [])
    free = [i for i in range(len(Z)) if i not in frozen]
    if not free:
        raise ValueError("ccm_optimize: all atoms frozen")

    from scipy.optimize import minimize

    def _periodic_image_bounds():
        axis_lengths: dict[int, float] = {}
        for t in translations_arr:
            norm = float(np.linalg.norm(t))
            if norm == 0.0:
                continue
            axis = int(np.argmax(np.abs(t)))
            off_axis = np.delete(t, axis)
            if np.linalg.norm(off_axis) > 1e-10 * norm or axis in axis_lengths:
                return None
            axis_lengths[axis] = abs(float(t[axis]))
        if not axis_lengths:
            return None
        eps = 1e-6
        bounds = []
        for atom in free:
            for dim in range(3):
                length = axis_lengths.get(dim)
                if length is None:
                    bounds.append((None, None))
                    continue
                half_width = 0.5 * length - eps
                if half_width <= 0.0:
                    bounds.append((None, None))
                    continue
                bounds.append((
                    float(C0[atom, dim] - half_width),
                    float(C0[atom, dim] + half_width),
                ))
        return bounds

    def _coords(x):
        C = C0.copy()
        C[free] = x.reshape(-1, 3)
        return C

    best_converged = None
    best_valid = None
    last_good = None

    def _remember(C, e, g):
        nonlocal best_converged, best_valid, last_good
        max_force = float(np.max(np.abs(g[free])))
        item = (float(e), C.copy(), g.copy(), max_force)
        last_good = item
        if (best_valid is None or max_force < best_valid[3]
                or (max_force == best_valid[3] and e < best_valid[0])):
            best_valid = item
        if max_force <= fmax and (
            best_converged is None or e < best_converged[0]
        ):
            best_converged = item
        return item

    def _recoverable_trial_error(exc):
        msg = str(exc)
        return (
            isinstance(exc, RuntimeError)
            or "not a valid cyclic cluster" in msg
        )

    def _penalty(x):
        center = last_good[1][free].ravel()
        dx = x - center
        return last_good[0] + 1.0 + float(dx @ dx), 2.0 * dx

    def fun(x):
        C = _coords(x)
        try:
            e = run_ccm(Z, C, translations_angstrom, charge=charge,
                        madelung=madelung, conv_tol=conv_tol).total_energy
            g = ccm_gradient_fd(Z, C, translations_angstrom, charge=charge,
                                madelung=madelung, atoms=free,
                                conv_tol=conv_tol)
        except (RuntimeError, ValueError) as exc:
            if last_good is None or not _recoverable_trial_error(exc):
                raise
            return _penalty(x)
        _remember(C, e, g)
        return e, (g[free] * ANGSTROM_TO_BOHR).ravel()  # Ha/Angstrom for scipy

    res = minimize(
        fun,
        C0[free].ravel(),
        jac=True,
        method="L-BFGS-B",
        bounds=_periodic_image_bounds(),
        options={
            "gtol": fmax * ANGSTROM_TO_BOHR,
            "maxiter": max_steps,
            "maxls": 40,
        },
    )
    try:
        C_final = _coords(res.x)
        e_final = run_ccm(Z, C_final, translations_angstrom, charge=charge,
                          madelung=madelung, conv_tol=conv_tol).total_energy
        g_final = ccm_gradient_fd(Z, C_final, translations_angstrom,
                                  charge=charge, madelung=madelung,
                                  atoms=free, conv_tol=conv_tol)
        final_item = _remember(C_final, e_final, g_final)
    except (RuntimeError, ValueError) as exc:
        if not _recoverable_trial_error(exc):
            raise
        final_item = None

    chosen = best_converged or final_item or best_valid
    if chosen is None:
        raise RuntimeError("ccm_optimize: no valid CCM geometry was evaluated")
    C = chosen[1]
    final = run_ccm(Z, C, translations_angstrom, charge=charge,
                    madelung=madelung, conv_tol=conv_tol)
    return C, final
