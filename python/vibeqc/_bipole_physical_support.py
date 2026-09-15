"""Private finite AO-product and SR quartet domains for HF/correlation source development.

The rule is on the actual ERI ordering (a,b|c,d): both AO products have
radius r_pair and their unweighted geometric midpoints have radius r_mid.
Exchange must call it on (a,c|b,d); there is no additional output-pair mask.
This is a new explicit support policy, not the legacy HF O/C/D traversal.
It supplies labels to the existing native erfc integrals, not a Hamiltonian
or symmetry certificate. Product domains also feed explicit-cell native
Fourier/LR panels; production consumers still need integration. Python is limited to bounded geometry/label orchestration.

All distance decisions use exact rational values of the supplied binary64
geometry and radii. Exact inverse-row bounds admit the complete candidate
boxes, without a floating boundary slack or modulo-BvK wrapping. Numerical
Seitz fits can still fail exact retained-label covariance at a boundary.
Budgets count conservative logical storage/work, not allocator/RSS bounds.
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import itertools
import math
import struct

import numpy as np

from .symmetry_shared import Budget

_POLICY = b"vibeqc.bipole.physical-quartet-support/exact-binary64-midpoints-v1"
_PRODUCT_POLICY = b"vibeqc.bipole.physical-product-support/exact-binary64-distance-v1"
_FIXED_BYTES = 262144
_MAX_INPUT_BITS = 256
_MAX_LABEL = 2**50


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if value < 1 or value > 2**63 - 1:
        raise ValueError(f"{name} outside positive int64 range")
    return int(value)


def _rational(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("support geometry and radii must be finite")
    result = Fraction(value)
    if max(abs(result.numerator).bit_length(), result.denominator.bit_length()) > _MAX_INPUT_BITS:
        raise ValueError("support binary64 geometry exceeds exact-arithmetic bit bound")
    return result


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _matvec(a, v):
    return tuple(_dot(row, v) for row in a)


def _subtract(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _inverse(a):
    cofactors = tuple(tuple(
        a[(i+1) % 3][(j+1) % 3] * a[(i+2) % 3][(j+2) % 3]
        - a[(i+1) % 3][(j+2) % 3] * a[(i+2) % 3][(j+1) % 3]
        for j in range(3)) for i in range(3))
    # Cyclic minors already carry their cofactor signs.
    determinant = _dot(a[0], cofactors[0])
    if determinant == 0:
        raise ValueError("support lattice must be nonsingular")
    return tuple(tuple(cofactors[j][i] / determinant for j in range(3)) for i in range(3))


def _box(offset, radius, inverse):
    center = _matvec(inverse, offset)
    bounds = []
    for row, coordinate in zip(inverse, center):
        squared = radius * radius * _dot(row, row)
        bound = math.isqrt(squared.numerator // squared.denominator)
        if bound * bound * squared.denominator < squared.numerator:
            bound += 1
        lower, upper = math.floor(coordinate) - bound, math.ceil(coordinate) + bound
        if lower < -_MAX_LABEL or upper > _MAX_LABEL:
            raise ValueError("support candidate label exceeds exact-label bound")
        bounds.append((lower, upper))
    return tuple(bounds)


def _labels(box):
    return itertools.product(*(range(lo, hi + 1) for lo, hi in box))


def _box_size(box):
    return math.prod(hi - lo + 1 for lo, hi in box)


@dataclass(frozen=True, slots=True)
class QuartetSupportPlan:
    candidate_count: int
    image_count: int
    inventoried_bytes: int
    work_units: int
    support_identity_sha256: str


@dataclass(frozen=True, slots=True, init=False, eq=False)
class QuartetSupport:
    """Immutable integer [image,3,3] labels, ordered by (g,h,p), s=p+h.

    Centers are geometry declarations, not authenticated basis/atom maps.
    The caller must bind shell ownership, basis, lattice and kernel to its
    numerical source. Retaining these labels alone grants no pair skipping.
    """
    memory: QuartetSupportPlan
    images: np.ndarray

    def __init__(self, *args, **kwargs):
        raise TypeError("construct support with build_quartet_support")

    @property
    def physical_hamiltonian_certified(self):
        return False

    @property
    def symmetry_certified(self):
        return False


def _prepare(lattice, centers, pair_radius, midpoint_radius, budget, maximum_candidates, maximum_images,
             *, pair_only=False):
    if not isinstance(budget, Budget):
        raise TypeError("support requires an explicit shared Budget")
    budget.admit(_FIXED_BYTES, 65536)
    for array, shape in ((lattice, (3, 3)), (centers, (2 if pair_only else 4, 3))):
        if not isinstance(array, np.ndarray) or array.dtype != np.dtype("float64") or array.shape != shape:
            raise TypeError(f"support requires a float64 array of shape {shape}")
        if not array.flags.c_contiguous or not array.flags.aligned:
            raise ValueError("support arrays must be aligned and C-contiguous")
    for radius in (pair_radius, midpoint_radius):
        if isinstance(radius, (bool, np.bool_)) or not isinstance(radius, (int, float, np.integer, np.floating)):
            raise TypeError("support radii must be real numbers")
    maximum_candidates = _positive_integer(maximum_candidates, "maximum_candidates")
    maximum_images = _positive_integer(maximum_images, "maximum_cells" if pair_only else "maximum_images")
    # Snapshot the small geometry after fixed admission. All later passes
    # use these immutable rationals, so replay never re-reads the arrays.
    a = tuple(tuple(_rational(x) for x in row) for row in lattice)
    xyz = tuple(tuple(_rational(x) for x in row) for row in centers)
    rp, rm = _rational(pair_radius), _rational(midpoint_radius)
    if rp < 0 or rm < 0:
        raise ValueError("support radii must be nonnegative")
    inverse = _inverse(a)
    bits = max(max(abs(x.numerator).bit_length(), x.denominator.bit_length())
               for row in a + xyz + inverse + ((rp, rm),) for x in row)
    # Scale abstract work with rational operand width; includes two passes,
    # box construction and distance tests. This is not an instruction count.
    unit = 4096 * (1 + bits // 64)**2
    digest = hashlib.sha256(_PRODUCT_POLICY if pair_only else _POLICY)
    for row in a + xyz + ((rp,) if pair_only else (rp, rm),):
        for value in row:
            digest.update(struct.pack(">d", float(value)))
    return a, xyz, inverse, rp, rm, unit, digest.hexdigest(), maximum_candidates, maximum_images


def _walk(prepared, budget, output=None):
    a, xyz, inverse, rp, rm, unit, identity, maximum_candidates, maximum_images = prepared
    candidate_count = image_count = 0

    def admit_candidates(count):
        nonlocal candidate_count
        candidate_count += count
        if candidate_count > maximum_candidates:
            raise ValueError("quartet support candidate cap exceeded")
        budget.admit(_FIXED_BYTES + 144 * image_count, 65536 + 2 * unit * candidate_count)

    ab, cd = _subtract(xyz[0], xyz[1]), _subtract(xyz[2], xyz[3])
    g_box, h_box = _box(ab, rp, inverse), _box(cd, rp, inverse)
    admit_candidates(_box_size(g_box))
    for g in _labels(g_box):
        ag = _matvec(a, g)
        delta = _subtract(ab, ag)
        if _dot(delta, delta) > rp * rp:
            continue
        admit_candidates(_box_size(h_box))
        for h in _labels(h_box):
            ah = _matvec(a, h)
            delta = _subtract(cd, ah)
            if _dot(delta, delta) > rp * rp:
                continue
            midpoint = tuple((xyz[0][d] + xyz[1][d] - xyz[2][d] - xyz[3][d]
                              + ag[d] - ah[d]) / 2 for d in range(3))
            p_box = _box(midpoint, rm, inverse)
            admit_candidates(_box_size(p_box))
            for p in _labels(p_box):
                delta = _subtract(midpoint, _matvec(a, p))
                if _dot(delta, delta) > rm * rm:
                    continue
                image_count += 1
                if image_count > maximum_images:
                    raise ValueError("quartet support retained-image cap exceeded")
                budget.admit(_FIXED_BYTES + 144 * image_count, 65536 + 2 * unit * candidate_count)
                if output is not None:
                    output[image_count - 1] = (g, p, tuple(p[d] + h[d] for d in range(3)))
    return QuartetSupportPlan(candidate_count, image_count, _FIXED_BYTES + 144 * image_count,
                              65536 + 2 * unit * candidate_count, identity)


def plan_quartet_support(lattice, centers, *, pair_radius, midpoint_radius,
                         budget, maximum_candidates, maximum_images):
    """Geometry preflight: count a complete support before output allocation.

    Unlike a shape-only native integral plan this reads geometry and performs
    exact domain tests. Each candidate box is admitted before walking it.
    """
    prepared = _prepare(lattice, centers, pair_radius, midpoint_radius, budget,
                        maximum_candidates, maximum_images)
    return _walk(prepared, budget)


def build_quartet_support(lattice, centers, *, pair_radius, midpoint_radius,
                          budget, maximum_candidates, maximum_images):
    prepared = _prepare(lattice, centers, pair_radius, midpoint_radius, budget,
                        maximum_candidates, maximum_images)
    plan = _walk(prepared, budget)
    output = np.empty((plan.image_count, 3, 3), dtype=np.int64)
    replay = _walk(prepared, budget, output)
    if replay != plan:
        raise RuntimeError("quartet support replay disagrees with its admitted plan")
    # Both numerical buffers coexist within the 144 bytes/image allowance.
    images = np.frombuffer(output.tobytes(), dtype=np.int64).reshape(output.shape)
    result = object.__new__(QuartetSupport)
    object.__setattr__(result, "memory", plan)
    object.__setattr__(result, "images", images)
    return result


@dataclass(frozen=True, slots=True)
class ProductSupportPlan:
    candidate_count: int
    cell_count: int
    inventoried_bytes: int
    work_units: int
    support_identity_sha256: str


@dataclass(frozen=True, slots=True, init=False, eq=False)
class ProductSupport:
    """Immutable lexicographic [cell,3] labels for |ra-rb-A R| <= r_pair.

    Reversed products have C_ba = -C_ab; an individual off-diagonal list need
    not be invariant under inversion. This is the same product predicate used
    on each axis of QuartetSupport, without its additional midpoint mask.
    Centers declare geometry only: native callers must bind selected AO rows
    to these centers and inventory the retained support alongside their panels.
    """
    memory: ProductSupportPlan
    cells: np.ndarray

    def __init__(self, *args, **kwargs):
        raise TypeError("construct support with build_product_support")

    @property
    def physical_hamiltonian_certified(self):
        return False

    @property
    def symmetry_certified(self):
        return False


def _walk_product(prepared, budget, output=None):
    a, xyz, inverse, radius, _, unit, identity, maximum_candidates, maximum_cells = prepared
    offset = _subtract(xyz[0], xyz[1])
    box = _box(offset, radius, inverse)
    candidates = _box_size(box)
    if candidates > maximum_candidates:
        raise ValueError("product support candidate cap exceeded")
    work = 65536 + 2 * unit * candidates
    budget.admit(_FIXED_BYTES, work)
    count = 0
    for label in _labels(box):
        delta = _subtract(offset, _matvec(a, label))
        if _dot(delta, delta) > radius * radius:
            continue
        count += 1
        if count > maximum_cells:
            raise ValueError("product support retained-cell cap exceeded")
        budget.admit(_FIXED_BYTES + 48 * count, work)
        if output is not None:
            output[count - 1] = label
    return ProductSupportPlan(candidates, count, _FIXED_BYTES + 48 * count, work, identity)


def plan_product_support(lattice, centers, *, pair_radius, budget, maximum_candidates, maximum_cells):
    """Exact geometry preflight for one product, before cell output allocation."""
    prepared = _prepare(lattice, centers, pair_radius, 0, budget,
                        maximum_candidates, maximum_cells, pair_only=True)
    return _walk_product(prepared, budget)


def build_product_support(lattice, centers, *, pair_radius, budget, maximum_candidates, maximum_cells):
    prepared = _prepare(lattice, centers, pair_radius, 0, budget,
                        maximum_candidates, maximum_cells, pair_only=True)
    plan = _walk_product(prepared, budget)
    output = np.empty((plan.cell_count, 3), dtype=np.int64)
    if _walk_product(prepared, budget, output) != plan:
        raise RuntimeError("product support replay disagrees with its admitted plan")
    cells = np.frombuffer(output.tobytes(), dtype=np.int64).reshape(output.shape)
    result = object.__new__(ProductSupport)
    object.__setattr__(result, "memory", plan)
    object.__setattr__(result, "cells", cells)
    return result
