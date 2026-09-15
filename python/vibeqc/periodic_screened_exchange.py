"""Shared CAM / range-separated exact-exchange resolution for periodic routes.

Every periodic SCF driver that builds exact exchange resolves its
functional through :func:`resolve_periodic_exchange`, which maps the
libxc CAM parameters onto the two exchange kernels the periodic code
can build natively:

    K_HF = c_full * K_full  +  c_sr * K_erfc(omega_screen)

``K_full`` is the full-range 1/r exchange (the trailing ``omega``
argument of ``build_jk_gamma_molecular_limit`` / ``build_jk_2e_real_space``
set to 0) and ``K_erfc(omega)`` is the short-range erfc-screened
exchange (the same builders with ``omega > 0``; see
``cpp/include/vibeqc/periodic_fock.hpp`` -- ``omega > 0`` selects
``erfc(omega*r_12)/r_12`` via libint's ``Operator::erfc_coulomb``).

The libxc/CAM convention exposed on :class:`Functional` is

    K_op(r) = cam_alpha * 1/r + cam_beta * erf(rsh_omega*r)/r

(see the ``cam_alpha`` property docstring). With erf = 1 - erfc this is

    K_op(r) = (cam_alpha + cam_beta) * 1/r
              - cam_beta * erfc(rsh_omega*r)/r

so ``c_full = cam_alpha + cam_beta`` and ``c_sr = -cam_beta``.

Worked check -- HSE06 (Heyd, Scuseria & Ernzerhof, J. Chem. Phys. 118,
8207 (2003), Eq. 2: exact exchange enters only through the short-range
erfc(omega*r)/r kernel; omega = 0.11 bohr^-1 per Krukau, Vydrov,
Izmaylov & Scuseria, J. Chem. Phys. 125, 224106 (2006)):
cam_alpha = 0.25, cam_beta = -0.25, rsh_omega = 0.11 gives
c_full = 0, c_sr = 0.25 -- i.e. K_HF = 0.25 * K_erfc(0.11), a pure
short-range screened exchange with NO full-range arm and therefore no
G -> 0 exchange divergence (the erfc kernel's zero mode is the finite
pi/omega^2), so no Madelung / exxdiv seam applies. This matches the
CRYSTAL treatment (CRYSTAL23 manual pp. 136-137: the SR operator is
assembled directly, not as full-minus-long-range).

Policy (maintainer decision 2026-07-09, HANDOVER_SLAB_2D_ROUTING.md):
periodic routes ship

* global hybrids (``is_range_separated`` false): ``c_full =
  hf_exchange_fraction``, unchanged behaviour; and
* short-range-only screened hybrids (HSE-type, ``c_full == 0``);

and FAIL CLOSED on range-separated functionals with a nonzero
full-range arm (wB97X, CAM-B3LYP, LC-wPBE, ...): their ``K_full``
contribution needs an exxdiv (G -> 0 exchange divergence) treatment
that has not been validated on the periodic routes.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "PeriodicExchangeAssembly",
    "resolve_periodic_exchange",
    "reject_periodic_gdf_unsupported_functional",
    "reject_unscreened_range_separated",
    "build_exchange_gamma",
    "build_exchange_blocks",
]


def reject_periodic_gdf_unsupported_functional(func, *, where: str) -> None:
    """Reject correlation terms absent from every periodic GDF KS driver."""
    if func is None:
        return
    if bool(getattr(func, "is_double_hybrid", False)):
        raise NotImplementedError(
            f"{where}: double-hybrid functionals are unavailable because "
            "periodic GDF does not add the required MP2 correlation energy."
        )
    if bool(getattr(func, "needs_vv10", False)):
        raise NotImplementedError(
            f"{where}: VV10/nonlocal-correlation functionals are unavailable "
            "because periodic GDF does not evaluate the nonlocal correlation "
            "energy or potential."
        )


def reject_unscreened_range_separated(func, *, where: str) -> None:
    """Fail closed on ANY range-separated functional (HSE-type included).

    For periodic routes whose exchange build is full-range only
    (GDF-fitted K, GPW/GAPW grid K, full-range gradients): running an
    RS functional there would silently apply ``hf_exchange_fraction``
    full-range -- HSE06 silently becomes PBE0. Routes that DO build the
    erfc kernel use :func:`resolve_periodic_exchange` instead, which
    admits HSE-type.
    """
    if func is None:
        return
    if bool(getattr(func, "is_range_separated", False)):
        raise NotImplementedError(
            f"{where}: range-separated functionals are not supported on "
            "this route -- its exact exchange is built full-range only, "
            "which would silently treat a screened hybrid like hse06 as "
            "its full-range twin (pbe0). Screened hybrids run on "
            "jk_method='bipole' (3D) and the slab_ewald_2d/AUTO ewald "
            "routes (dim=2 slabs); otherwise use a global hybrid "
            "(e.g. pbe0) or a pure functional."
        )

# cam_alpha + cam_beta below this magnitude counts as "no full-range
# arm". The CAM parameters are few-digit published constants (0.25,
# 0.11, ...), so any nonzero arm is >= O(0.01); 1e-10 only absorbs
# float noise from the C++/libxc round trip.
_C_FULL_TOL = 1e-10


@dataclass(frozen=True)
class PeriodicExchangeAssembly:
    """Coefficients of the periodic exact-exchange assembly.

    ``K_HF = c_full * K_full + c_sr * K_erfc(omega_screen)``.

    ``omega_screen`` is the *physical* screening parameter of the
    functional (bohr^-1) -- do not confuse it with the numerical Ewald
    split parameter (``slab_ewald_alpha`` / ``crystal_default_ewald_alpha``)
    that the drivers also call ``omega``.
    """

    c_full: float
    c_sr: float
    omega_screen: float

    @property
    def needs_exchange(self) -> bool:
        """True when any exact-exchange build is required."""
        return self.c_full != 0.0 or self.c_sr != 0.0

    @property
    def is_screened(self) -> bool:
        """True when the short-range erfc kernel contributes."""
        return self.c_sr != 0.0


def resolve_periodic_exchange(func, *, where: str) -> PeriodicExchangeAssembly:
    """Resolve ``func`` into the periodic exchange assembly, or fail closed.

    Parameters
    ----------
    func
        A ``Functional`` (or None for pure HF, which maps to
        ``c_full = 1``).
    where
        Caller name used in error messages (e.g.
        ``"run_rks_periodic_gamma_ewald3d"``).

    Raises
    ------
    NotImplementedError
        For range-separated functionals with a nonzero full-range
        exact-exchange arm (see module docstring for the policy).
    """
    if func is None:
        return PeriodicExchangeAssembly(1.0, 0.0, 0.0)

    if not bool(getattr(func, "is_range_separated", False)):
        return PeriodicExchangeAssembly(
            float(func.hf_exchange_fraction), 0.0, 0.0
        )

    cam_alpha = float(func.cam_alpha)
    cam_beta = float(func.cam_beta)
    omega = float(func.rsh_omega)
    c_full = cam_alpha + cam_beta

    if abs(c_full) > _C_FULL_TOL:
        raise NotImplementedError(
            f"{where}: the range-separated functional has a full-range "
            f"exact-exchange arm (cam_alpha + cam_beta = {c_full:.6g}, "
            f"cam_alpha = {cam_alpha:.6g}, cam_beta = {cam_beta:.6g}, "
            f"omega = {omega:.6g} bohr^-1), whose G -> 0 exchange "
            "divergence (exxdiv) treatment is not validated on the "
            "periodic routes. Supported here: global hybrids (e.g. "
            "pbe0) and short-range screened hybrids (e.g. hse06). "
            "Long-range-corrected functionals (wb97x, cam-b3lyp, "
            "lc-wpbe, ...) are not supported periodically yet."
        )

    if omega <= 0.0:
        # is_range_separated with no screening length is a functional
        # metadata inconsistency, not a user error.
        raise ValueError(
            f"{where}: functional reports is_range_separated but "
            f"rsh_omega = {omega:.6g} <= 0; cannot build the erfc-"
            "screened exchange kernel."
        )

    return PeriodicExchangeAssembly(0.0, -cam_beta, omega)


def build_exchange_gamma(basis, system, lat_opts, D, exx):
    """Coefficient-folded Γ exact exchange for the ewald-family drivers.

    Returns ``c_full * K_full + c_sr * K_erfc(omega_screen)`` as a dense
    ``(n_bf, n_bf)`` array, or ``None`` when the assembly needs no
    exchange. The coefficients are folded in HERE so callers fold the
    result into the Fock matrix with a bare ``- 0.5 * K`` (closed
    shell) / ``- K_s`` (per spin on ``K(2 D_s)/2``) and no further
    functional-dependent factor.

    Only ``.K`` of the native builder is consumed: with ``omega > 0``
    the builder's ``.J`` is the erfc-screened Hartree matrix, which
    must NOT replace the full-range J (J always comes from the
    Ewald-composed builders). The erfc kernel is bounded by 1/r, so a
    lattice/Schwarz truncation that converges the full-range K
    converges the screened K a fortiori.
    """
    import numpy as np

    from ._vibeqc_core import build_jk_gamma_molecular_limit

    K = None
    if exx.c_full != 0.0:
        jk = build_jk_gamma_molecular_limit(basis, system, lat_opts, D, 0.0)
        K = exx.c_full * np.asarray(jk.K)
    if exx.c_sr != 0.0:
        # omega > 0 selects erfc(omega*r_12)/r_12 (periodic_fock.hpp).
        # The erfc kernel has no G -> 0 divergence (finite pi/omega^2
        # zero mode, included by the real-space sum), so no Madelung /
        # exxdiv seam is added -- and none must be.
        jk_sr = build_jk_gamma_molecular_limit(
            basis, system, lat_opts, D, float(exx.omega_screen)
        )
        K_sr = exx.c_sr * np.asarray(jk_sr.K)
        K = K_sr if K is None else K + K_sr
    return K


def build_exchange_blocks(
    basis,
    system,
    lat_opts,
    D_real,
    exx,
    *,
    full_range_alpha=None,
    symmetry_reduction=None,
):
    """Coefficient-folded real-space exchange blocks for multi-k drivers.

    Multi-k analogue of :func:`build_exchange_gamma` on the lattice-block
    K builder ``build_jk_2e_real_space`` (same trailing screening-omega
    convention). Returns a list of per-cell ``(n_bf, n_bf)`` arrays
    ``c_full * K_full(g) + c_sr * K_erfc(g)`` over ``D_real.cells``, or
    ``None`` when no exchange is needed. Only ``.K`` is consumed; J
    always comes from the (unscreened) Ewald J-block builders.

    ``full_range_alpha`` switches the ``c_full`` arm from the bare
    ``1/r`` kernel to the Ewald-split **short-range** part
    ``erfc(alpha r)/r``. Pass it whenever the caller completes the split
    with the reciprocal and ``q + G = 0`` arms from
    :mod:`vibeqc.periodic_corrected_exchange`; the bare-kernel default is
    only valid where the density decays with image distance (the Gamma
    molecular limit), because on a finite k mesh the density is
    Born-von-Karman-torus periodic and the bare sum does not converge.
    The ``c_sr`` arm is unaffected: ``erfc(omega_screen r)/r`` is already
    short-ranged, so a screened hybrid such as hse06 (``c_full = 0``)
    never needs the split.

    ``symmetry_reduction`` is an optional ``(mapping, rep_cell_indices)``
    pair from :mod:`vibeqc.bipole_symmetry_fock` (a legacy-radial
    ``cell_orbit_mapping`` built on ``D_real.cells``). When given, the
    **erfc-screened** arms route through the SYM3b orbit-reduced builder
    :func:`~vibeqc.bipole_symmetry_fock.build_jk_reduced_symmetrized`,
    which evaluates K only at orbit-representative shell-pair blocks and
    reconstructs the rest by Wigner-D rotation --- measured 8.5x fewer
    built blocks on LiH rock-salt at a 12-bohr cutoff, growing with
    cutoff, against a build that is otherwise 94 % of a wedge SCF
    iteration. Two deliberate boundaries: the **bare** ``1/r`` arm
    (``full_range_alpha is None`` on the Gamma molecular-limit path)
    always uses the full builder --- SYM3b's bit-identity suite pins
    only erfc kernels --- and the reduced result equals the
    *symmetrized* full build, which differs from the raw build by the
    operator truncation asymmetry on shift-bearing cells (radial cell
    lists are not closed under the pair action ``R.g + s_b - s_a``).
    Callers must therefore pass it only where the symmetrized answer is
    the intended one: the wedge-native IBZ path, whose density is
    already the orbit-symmetrized object.
    """
    import numpy as np

    from ._vibeqc_core import build_jk_2e_real_space

    if not exx.needs_exchange:
        return None

    def _k_build(kernel_omega: float):
        if symmetry_reduction is not None and kernel_omega > 0.0:
            from .bipole_symmetry_fock import build_jk_reduced_symmetrized

            mapping, rep_cell_indices = symmetry_reduction
            return build_jk_reduced_symmetrized(
                basis,
                system,
                lat_opts,
                D_real,
                float(kernel_omega),
                mapping,
                rep_cell_indices,
            )
        return build_jk_2e_real_space(
            basis, system, lat_opts, D_real, float(kernel_omega)
        )

    n_cells = len(D_real.cells)

    def _on_density_cells(K_set, scale: float):
        # Align the builder's K(g) with ``D_real.cells`` by cell key. The
        # two lists coincide today; under LatticeSumOptions.pair_complete_1e
        # (#429) the density rides the longer pair-complete one-electron
        # list and the builder keeps the plain |g| ball, where K is zero
        # beyond its own support by its own truncation. A builder cell the
        # density does not carry is a bookkeeping error and raises.
        built = {
            tuple(int(v) for v in np.asarray(cell.index).reshape(3)): blk
            for cell, blk in zip(K_set.cells, K_set.blocks)
        }
        nbf = int(basis.nbasis)
        out = []
        for cell in D_real.cells:
            key = tuple(int(v) for v in np.asarray(cell.index).reshape(3))
            blk = built.pop(key, None)
            if blk is None:
                out.append(np.zeros((nbf, nbf), dtype=float))
            else:
                out.append(scale * np.asarray(blk, dtype=float))
        if built:
            raise RuntimeError(
                "periodic screened exchange: the K builder returned "
                f"{len(built)} cells the density list does not carry "
                f"(density has {n_cells} cells)"
            )
        return out

    K_blocks = None
    if exx.c_full != 0.0:
        full_kernel_omega = (
            0.0 if full_range_alpha is None else float(full_range_alpha)
        )
        jk_full = _k_build(full_kernel_omega)
        K_blocks = _on_density_cells(jk_full.K, float(exx.c_full))
    if exx.c_sr != 0.0:
        jk_sr = _k_build(float(exx.omega_screen))
        sr = _on_density_cells(jk_sr.K, float(exx.c_sr))
        K_blocks = (
            sr
            if K_blocks is None
            else [K_blocks[g] + sr[g] for g in range(n_cells)]
        )
    return K_blocks
