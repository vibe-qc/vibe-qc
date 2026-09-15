"""Mixed-boundary (wire/slab) electrostatic Green's functions — scalar kernels (D3 M2).

The explicit reciprocal-space kernels of the Hamiltonian-first low-D derivation
(`docs/aiccm2026dev_a_lowd_greens.md`, item 6 of the converged `-a`/`-b` result):
the potential of a charge that is **periodic** in the lattice directions and
**open** (free-space, decaying) in the transverse directions, with a neutralizing
**line** (1-D wire) / **sheet** (2-D slab) background over the periodic directions.

* **Wire** (periodic along `z`, period `L`; open transverse `ρ=(x,y)`):

      G^{1D}(ρ, z) = (2/L) Σ_{G_z≠0} K₀(|G_z|ρ) e^{iG_z z}  −  (2/L) ln(ρ/ρ₀)  +  C
                   = (4/L) Σ_{m≥1} K₀(2πmρ/L) cos(2πmz/L)   −  (2/L) ln(ρ/ρ₀)  +  C

* **Slab** (periodic in the plane, area `A`; open normal `z`):

      G^{2D}(ρ⃗, z) = (2π/A) Σ_{G∥≠0} e^{iG∥·ρ⃗} e^{−|G∥||z|} / |G∥|  −  (2π/A)|z|  +  C

``ρ₀`` and ``C`` are the gauge of the conditional (zero-transverse-mean) channel,
fixed downstream by the transverse-periodic reduction to ``v_E^{3D}`` (M3) — they
do **not** affect any harmonic / Poisson property, so they default to a bare
``ρ₀=1``, ``C=0`` here.

**Correctness gate (dependency-free).** Both kernels solve Poisson's equation with
the source on the charge axis / plane only, so **away from the source they are
harmonic**: ``∇²G^{1D}=0`` for ``ρ>0`` and ``∇²G^{2D}=0`` for ``z≠0`` (every
reciprocal mode is harmonic by the modified-Bessel / screened-Poisson equation,
and the ``ln ρ`` / ``|z|`` background term is harmonic off the source). With the
right lattice periodicity and transverse-reflection symmetry, this pins the
kernel form without any external reference (`tests/test_lowd_greens.py`). The
four-center routing and the M3 reduction to ``v_E^{3D}`` build on these.

Refs: Parry 1975; de Leeuw–Perram 1979; Bertaut; Rozzi et al. 2006. The 1-D wire
kernel is new to vibe-qc; the 2-D slab Coulomb side is also realized (at the
Hartree-`J` level) by :mod:`vibeqc.ewald_composed_slab`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["wire_greens", "slab_greens", "wire_self_energy", "slab_self_energy"]


def wire_greens(
    rho: float,
    z: float,
    *,
    period: float,
    n_recip: int = 200,
    rho0: float = 1.0,
    const: float = 0.0,
) -> float:
    """1-D wire Green's function ``G^{1D}(ρ, z)`` (periodic along ``z``).

    Parameters
    ----------
    rho : float
        Transverse distance from the wire axis (bohr), ``> 0``.
    z : float
        Coordinate along the periodic axis (bohr).
    period : float
        Lattice period ``L`` along ``z`` (bohr).
    n_recip : int
        Number of reciprocal terms ``m = 1..n_recip`` (the ``K₀(2πmρ/L)`` sum
        converges exponentially in ``mρ/L``; the default is converged for
        ``ρ/L ≳ 0.05``).
    rho0, const : float
        Gauge of the conditional channel (``−(2/L)ln(ρ/ρ₀) + C``). Fixed by the
        M3 reduction; irrelevant to the harmonic property.

    Returns
    -------
    float
        ``G^{1D}(ρ, z)`` (Hartree per unit charge).
    """
    from scipy.special import k0

    if rho <= 0.0:
        raise ValueError(f"wire_greens: rho must be > 0 (on-axis is singular); got {rho}")
    L = float(period)
    m = np.arange(1, int(n_recip) + 1, dtype=float)
    gz = 2.0 * np.pi * m / L
    osc = (4.0 / L) * np.sum(k0(gz * rho) * np.cos(gz * z))
    line = -(2.0 / L) * np.log(rho / rho0)
    return float(osc + line + const)


def _recip_2d(lattice2d: np.ndarray) -> tuple[np.ndarray, float]:
    """Reciprocal vectors (rows) + cell area for a 2-D real lattice (rows a1,a2)."""
    a = np.asarray(lattice2d, dtype=float).reshape(2, 2)
    area = float(abs(a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]))
    if area <= 0.0:
        raise ValueError("slab lattice is singular (zero area)")
    b = 2.0 * np.pi * np.linalg.inv(a).T  # rows are reciprocal vectors
    return b, area


def slab_greens(
    r_par: np.ndarray,
    z: float,
    *,
    lattice2d: np.ndarray,
    n_shell: int = 40,
    const: float = 0.0,
) -> float:
    """2-D slab Green's function ``G^{2D}(ρ⃗, z)`` (periodic in the plane).

    Parameters
    ----------
    r_par : array-like, shape (2,)
        In-plane coordinate ``ρ⃗ = (x, y)`` (bohr).
    z : float
        Coordinate along the open normal (bohr), ``≠ 0`` for the harmonic region.
    lattice2d : array-like, shape (2, 2)
        In-plane real-lattice vectors as rows ``(a1; a2)`` (bohr); the reciprocal
        lattice and cell area ``A`` are derived internally.
    n_shell : int
        Reciprocal-lattice shell half-width (``|n1|,|n2| ≤ n_shell``). The
        ``e^{−|G∥||z|}`` factor makes the sum converge exponentially for ``z≠0``.
    const : float
        Gauge constant ``C``.

    Returns
    -------
    float
        ``G^{2D}(ρ⃗, z)``.
    """
    b, area = _recip_2d(lattice2d)
    rp = np.asarray(r_par, dtype=float).reshape(2)
    az = abs(float(z))
    ns = int(n_shell)
    n = np.arange(-ns, ns + 1)
    n1, n2 = np.meshgrid(n, n, indexing="ij")
    sel = ~((n1 == 0) & (n2 == 0))
    G = (n1[sel, None] * b[0] + n2[sel, None] * b[1])  # (n_G, 2)
    gnorm = np.linalg.norm(G, axis=1)
    phase = G @ rp                                      # G∥·ρ⃗
    osc = (2.0 * np.pi / area) * np.sum(np.cos(phase) * np.exp(-gnorm * az) / gnorm)
    sheet = -(2.0 * np.pi / area) * az
    return float(osc + sheet + const)


def wire_self_energy(period: float, *, rho0: float = 1.0,
                     n_recip_floor: int = 400, _rel: float = 0.01) -> float:
    """Wire self-energy ``ξ = lim_{ρ→0}[G^{1D}(ρ,0) − 1/ρ]`` — the low-D Madelung-
    constant analog and the foundation for the M4 exchange-``q=0`` seam.

    The regular (self) part of the wire kernel once the ``1/ρ`` Coulomb
    singularity (which ``G^{1D}`` reconstructs from its ``K₀`` sum) is removed. The
    point-pair counterpart of the 3-D Ewald self-potential
    (:func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_background_constant`); it is
    what the bare-vs-neutral gap is built on (in 3-D this constant drives the
    bare-four-center Madelung over-binding, here it is the 1-D analog).

    **Gauge-dependent** through the conditional channel ``−(2/L)ln(ρ/ρ₀)``: the
    default ``ρ₀ = 1`` (bohr) fixes it; the physical gauge is set downstream by the
    transverse-periodic reduction (M3/M4). Extracted by **r² Richardson**
    extrapolation (the leading ``ρ → 0`` correction is ``O(ρ²)``, verified).
    """
    L = float(period)

    def xi(r: float) -> float:
        nrec = max(int(n_recip_floor), int(40.0 * L / r))
        return wire_greens(r, 0.0, period=L, n_recip=nrec, rho0=rho0) - 1.0 / r

    r = _rel * L
    return (4.0 * xi(r / 2.0) - xi(r)) / 3.0


def slab_self_energy(lattice2d: np.ndarray, *, const: float = 0.0,
                     n_shell_floor: int = 80, _rel: float = 0.01) -> float:
    """Slab self-energy ``ξ = lim_{z→0}[G^{2D}(0,z) − 1/z]`` — the 2-D Madelung-
    constant analog and the M4 exchange-``q=0`` seam foundation.

    The slab counterpart of :func:`wire_self_energy`: the regular part of the slab
    kernel at the source plane, gauge-fixed by ``const`` (default 0). Extracted by
    r² Richardson extrapolation along the open normal.
    """
    _, area = _recip_2d(lattice2d)
    a = float(area) ** 0.5

    def xi(z: float) -> float:
        ns = max(int(n_shell_floor), int(2.0 * a / z))
        return (slab_greens([0.0, 0.0], z, lattice2d=lattice2d, n_shell=ns, const=const)
                - 1.0 / z)

    z = _rel * a
    return (4.0 * xi(z / 2.0) - xi(z)) / 3.0
