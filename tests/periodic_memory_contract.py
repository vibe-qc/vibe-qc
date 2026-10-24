"""Independent allocation inventory for controlled periodic preflight tests.

No production estimator is imported here. Dimensions describe arrays and
planner declarations, so dropping a charged category cannot change the oracle.
"""

from math import ceil, pi, prod, sqrt


MIB = 1024**2
RUNTIME = "Python runtime + NumPy overhead"


def reciprocal_capacity(lengths, cutoff):
    """Biorthogonal index bound, enumerator slack, then half-cell q padding."""
    return prod(2 * (ceil(sqrt(2 * cutoff) * length / (2 * pi)) + 2) + 1
                for length in lengths)


def xc_value_inventory(n, atoms, points, cells, active, bras, workers, spin=1):
    # GGA values/derivatives for each active image, thread contractions and
    # libxc vectors share one bounded point batch. Three sets of cell matrices
    # cover Python density, its native copy and the output potential.
    per_point = 8 * (4 * active * n + workers * (6 * n + 20) + 64)
    fixed = 8 * workers * n * n * (6 if spin == 2 else 4)
    batch = min(points, 4096, (256 * MIB - fixed) // per_point)
    return {
        "Periodic-XC value numerical workspace": fixed + batch * per_point,
        "Periodic-XC value input grid": 44 * points + 32 * atoms,
        "Periodic-XC value density and potential blocks": 3 * spin * cells * n * n * 8,
        "Periodic-XC value cell-pair metadata": 20 * active * bras + 512 * spin * cells,
    }


def xc_gradient_inventory(n, atoms, points, cells, active, bras, workers, batch, spin=1):
    # GGA: ten AO value/derivative/Hessian tables per image plus home copy.
    # Shifted-grid response needs 24 closed-shell or 40 open-shell vectors.
    b = min(points, batch)
    return {
        "Periodic-XC gradient AO/gradient/Hessian tables": (active + 1) * 10 * b * n * 8,
        "Periodic-XC gradient density blocks": spin * cells * n * n * 8,
        "Periodic-XC gradient pair scratch": workers * (18 if spin == 2 else 10) * b * n * 8,
        "Periodic-XC gradient grid vectors": ((40 if spin == 2 else 24) + 4 + 3 * workers) * b * 8,
        "Periodic-XC gradient input grid": 44 * points + 32 * atoms,
        "Periodic-XC gradient cell-pair metadata": active * bras * 16 + cells * 384,
        "Periodic-XC gradient thread buffers": (workers + 2) * atoms * 3 * 8 + n * (32 * workers + 8),
    }


def controlled_gdf_inventory(n, aux, atoms, length, *, exchange, open_shell, gradient=False):
    """Inventory with cutoff=2 Ha, two k points and declared planner fixtures.

    The source workspace is 80 MiB, greater than the declared 1 KiB
    eigensolver workspace. One-electron cutoffs are 4 and 3 bohr, without
    pair-complete enlargement. XC has 31 points, five density images, five
    active images, three bra images, two workers and a 31-point gradient batch.
    """
    k, workers, points, cells, active, bras, batch = 2, 2, 31, 5, 5, 3, 31
    spin = 2 if open_shell else 1
    pairs = k * k if exchange else k
    tensor = 16 * k * aux * n * n
    metric = 16 * aux * aux
    source = 4096 + 128 * (aux + k) + max(
        4096 + 128 * 512,
        tensor + 2 * metric + 80 * MIB,
        tensor + 2 * metric + 1024,
        2 * tensor + 3 * metric,
    )
    capacity = (2 * ceil(4 / length) + 3)**3
    # Tiling cells centred in the radius-4 sphere fit inside radius 4+3L/2.
    # These fixtures are far from integer boundaries; conservative roundoff
    # padding in the production planner cannot change this ceiling.
    matrix_cells = min(capacity, ceil(4 * pi / 3 * (4 + 1.5 * length)**3 / length**3))
    categories = {
        "GDF Lpq factor cache": pairs * aux * n * n * 16,
        "per-k complex buffers": (4 + 2 * spin) * k * n * n * 16,
        "SCF accelerator history peak": (4 * 8 + 3) * spin * k * n * n * 16,
        "GDF contraction workspace": (32 if exchange else 1) * MIB,
        RUNTIME: 100 * MIB,
        "GDF SR/LR source and whitening workspace": source,
        "GDF reciprocal vectors and binding copy": 48 * reciprocal_capacity([length]*3, 2),
        "GDF one-electron lattice matrices and metadata": 8 * (matrix_cells * 8 * n * n + capacity * 256),
        "GDF one-electron Bloch matrices": 16 * 8 * k * n * n,
        "GDF nuclear images": 256 * (2 * ceil(3 / length) + 3)**3,
        "GDF nuclear Fourier workspace": 256 * MIB,
        "GDF retained quadrature": 0,
    }
    if atoms > 1:  # the two-H fixture is RKS/PBE; the one-H fixture is ROHF
        value = sum(xc_value_inventory(n, atoms, points, cells, active, bras, workers).values())
        response = sum(xc_gradient_inventory(n, atoms, points, cells, active, bras, workers, batch).values())
        categories["Periodic-XC retained data and phase workspace"] = max(value, response) if gradient else value
    return categories


def headline(categories):
    """The tests declare 1.5 headroom and exactly one runtime allowance."""
    return int(sum(categories.values()) * 1.5)


def wedge_cache_bytes(n, aux, k, *, exchange, bras=None):
    """One complex auxiliary/AO-pair block per exchange pair or J diagonal."""
    if not exchange:
        pairs = k
    else:
        rows = k if bras is None else bras
        pairs = rows * k + k - rows
    return pairs * aux * n * n * 16


def controlled_wedge_gdf_inventory(n, aux, a, *, bras=None):
    """RHF on the fcc primitive cell, with declared allocator inputs.

    The mesh has 64 points; cutoff=2 Ha, AO/nuclear radii=4/3 bohr,
    no pair-complete enlargement, native workspace=80 MiB, eigensolver=1 KiB.
    Primitive vectors have length a/sqrt(2), inverse rows sqrt(3)/a,
    and volume a**3/4. These declarations are independent of the estimator.
    """
    k = 64
    tensor, metric = 16 * k * aux * n * n, 16 * aux * aux
    capacity = (2 * ceil(4 * sqrt(3) / a) + 3)**3
    cells = min(capacity, ceil(4 * pi / 3 * (4 + 1.5 * a / sqrt(2))**3 / (a**3 / 4)))
    return {
        "GDF Lpq factor cache": wedge_cache_bytes(n, aux, k, exchange=True, bras=bras),
        "per-k complex buffers": 6 * k * n * n * 16,
        "SCF accelerator history peak": 35 * k * n * n * 16,
        "GDF contraction workspace": 32 * MIB,
        RUNTIME: 100 * MIB,
        "GDF SR/LR source and whitening workspace": 4096 + 128 * (aux + k) + max(
            4096 + 128 * 512, tensor + 2 * metric + 80 * MIB,
            tensor + 2 * metric + 1024, 2 * tensor + 3 * metric,
        ),
        "GDF reciprocal vectors and binding copy": 48 * reciprocal_capacity([a / sqrt(2)] * 3, 2),
        "GDF one-electron lattice matrices and metadata": 8 * (cells * 8 * n * n + capacity * 256),
        "GDF one-electron Bloch matrices": 16 * 8 * k * n * n,
        "GDF nuclear images": 256 * (2 * ceil(3 * sqrt(3) / a) + 3)**3,
        "GDF nuclear Fourier workspace": 256 * MIB,
        "GDF retained quadrature": 0,
    }
