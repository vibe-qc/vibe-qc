"""Image domain of the periodic XC density on the 3-D GDF Kohn-Sham routes.

``build_xc_periodic`` takes its image domain from the lattice options it is
handed: image cells whose atoms lie within ``cutoff_bohr`` of a home atom are
active, and a bra/ket image pair contributes only if its lattice difference is
among the density cells, a lattice-vector ball of the same radius. The GDF KS
drivers used to hand it the caller's ``cutoff_bohr`` (15 bohr by default)
while their one-electron domain is grown to a converged bound, so the XC
density was truncated on compact cells (#465): diamond def2-SVP PBE0 at Γ
was 0.38 mHa off, and the cubic Al2O3 model (STO-3G, PBE) 1.55 mHa off.

The bound here closes the density cells over every AO pair that can overlap
at one grid point. A density term χ_μ(r - R_a) D χ_ν(r - R_s) is negligible
unless both functions exceed the amplitude ``amplitude`` at ``r``, so its AO
centres are at most ``2 r_ao`` apart, where ``r_ao`` is the largest radius at
which a shell's absolute Gaussian envelope reaches ``amplitude``. The lattice
difference ``R_s - R_a`` is then at most ``2 r_ao`` plus the largest in-cell
distance between shell centres, the same span the one-electron bound adds for
a lattice-vector ball. The active-image radius this implies (an image atom
within ``r_ao`` of a weighted grid point) is smaller, so one cutoff serves
both.

The kernel's own AO-reach screen (``kPeriodicXCAOScreeningTol``) keeps the cost
of the wider domain to the shells that reach each grid batch.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ._vibeqc_core import BasisSet, LatticeSumOptions, PeriodicSystem

__all__ = ["xc_image_cutoff_bound", "xc_lattice_opts", "xc_lattice_opts_at"]

# Absolute AO amplitude below which a shell is taken not to reach a point.
# Measured on fixed converged densities (PBE, STO-3G). Cubic Al2O3 model:
# E_xc moves 1.6e-3 Ha from 15 to 20 bohr and 1.4e-8 Ha from 20 to 25; the
# bound is 23.2 bohr. Rocksalt LiH (fcc a = 7.72 bohr, diffuse Li 2sp): E_xc
# is 1.5e-4 Ha off at 15 bohr, 5.1e-8 at 26.75 (the 1e-3 bound) and 5e-11 at
# 31.4 (this bound), against 38.55 bohr (the 1e-6 bound) at twice this cost.
# Diamond def2-SVP gets 18.5 bohr; #465 found it converged by 20.
XC_IMAGE_AMPLITUDE = 1e-4


def _shell_envelope(exponents, coefficients, l: int, r: float) -> float:
    return float(sum(
        abs(c) * r**l * np.exp(-a * r * r)
        for a, c in zip(exponents, coefficients)
    ))


def _shell_reach(shell, amplitude: float) -> float:
    """Largest radius at which the shell's absolute envelope reaches ``amplitude``."""
    alphas = np.asarray(shell.exponents, dtype=float)
    coefficients = np.asarray(shell.coefficients, dtype=float)
    l = int(shell.l)
    # The envelope falls monotonically beyond its outermost primitive peak.
    r = float(np.sqrt(l / (2.0 * alphas.min()))) if l > 0 else 0.0
    if _shell_envelope(alphas, coefficients, l, r) <= amplitude:
        return r
    hi = max(r, 1.0)
    while _shell_envelope(alphas, coefficients, l, hi) > amplitude:
        hi *= 2.0
        if not np.isfinite(hi) or hi > 1e6:
            raise ValueError("periodic XC domain: shell envelope does not decay")
    lo = r
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _shell_envelope(alphas, coefficients, l, mid) > amplitude:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-6:
            break
    return hi


def xc_image_cutoff_bound(
    system: PeriodicSystem,
    basis: BasisSet,
    *,
    amplitude: float = XC_IMAGE_AMPLITUDE,
) -> float:
    """Lattice cutoff (bohr) that closes the XC density over overlapping AO pairs.

    Returns ``2 * max_shell r_ao(amplitude) + span``, where ``span`` is the
    largest distance between two shell centres of the home cell.
    """
    if not (np.isfinite(amplitude) and 0.0 < amplitude < 1.0):
        raise ValueError("periodic XC domain: amplitude must lie in (0, 1)")
    shells = list(basis.shells())
    reach = max(_shell_reach(shell, amplitude) for shell in shells)
    origins = [np.asarray(shell.origin, dtype=float) for shell in shells]
    span = max(float(np.linalg.norm(a - b)) for a in origins for b in origins)
    radius = 2.0 * reach + span
    if not np.isfinite(radius):
        raise ValueError("periodic XC domain: cutoff is not representable")
    return float(np.nextafter(radius, np.inf))


def xc_lattice_opts(
    system: PeriodicSystem,
    basis: BasisSet,
    base_opts: LatticeSumOptions,
    *,
    rcut_strategy: object,
    plog=None,
) -> LatticeSumOptions:
    """Lattice options for the periodic XC term of a 3-D GDF KS calculation.

    Under the default ``rcut_strategy`` (PySCF-style automatic cutoffs) the
    caller's ``cutoff_bohr`` is raised to :func:`xc_image_cutoff_bound`, as the
    one-electron domain is raised to its own bound. An explicit flat strategy
    keeps the caller's finite domain, and so does a non-3-D cell.
    """
    from .periodic_k_gdf import _is_pyscf_auto

    if int(system.dim) != 3 or not _is_pyscf_auto(rcut_strategy):
        return base_opts
    base = float(base_opts.cutoff_bohr)
    grown = max(base, xc_image_cutoff_bound(system, basis))
    opts = xc_lattice_opts_at(basis, base_opts, grown)
    if plog is not None and grown > base:
        plog.info(f"XC density image cutoff {base:.2f} -> {grown:.2f} bohr "
                  f"(AO pairs overlapping at amplitude {XC_IMAGE_AMPLITUDE:g})")
    return opts


def xc_lattice_opts_at(
    basis: BasisSet,
    base_opts: LatticeSumOptions,
    cutoff_bohr: Optional[float],
) -> LatticeSumOptions:
    """``base_opts`` with the XC image cutoff ``cutoff_bohr``.

    Rebuilds the domain an SCF recorded (``gdf_xc_lattice_cutoff_bohr``) so
    that its XC gradient differentiates the same quadrature. A missing or
    zero ``cutoff_bohr`` (results from before #465, HF results), or one not
    above the base cutoff, returns ``base_opts`` unchanged, as the SCF used it.
    A grown domain is a lattice-vector ball (``pair_complete_1e`` off), the
    set ``build_xc_periodic`` resolves bra/ket differences against.
    """
    from .lattice_screening import RcutStrategy, make_lattice_opts

    if cutoff_bohr is None or not float(cutoff_bohr) > float(base_opts.cutoff_bohr):
        return base_opts
    opts = make_lattice_opts(basis, strategy=RcutStrategy.FLAT,
                             base_opts=base_opts, cutoff_bohr=float(cutoff_bohr))
    if hasattr(opts, "pair_complete_1e"):
        opts.pair_complete_1e = False
    return opts
