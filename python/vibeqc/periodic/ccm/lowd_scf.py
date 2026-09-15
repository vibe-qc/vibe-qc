"""A-line mixed-boundary wire SCF control (D3 M4/M5).

The self-consistent field on the mixed-boundary **wire** Hamiltonian: the finite
cluster supercell with its electrostatics carried by the wire Green's function
``G^{1D}`` (periodic along ``axis``, open in the transverse plane), *not* the
3-D-torus Ewald kernel of the neutral fitted-torus control on a vacuum-padded
cell. This is
the SCF that pairs with :mod:`~vibeqc.periodic.ccm.lowd_four_center` -- the whole
Hamiltonian rides one gauge:

This A-line implementation has no proved binding to the
union-and-weight/Wigner--Seitz integral-weighting construction, so it is not
classified as Γ-CCM. Running the finite cluster in a real-Gamma representation,
reusing this module from an A-line namespace, or feeding its result to an
existing post-HF solver does not confer a construction identity. It also must
not be substituted for the separately constructed χ-CCM-B Hamiltonian.

* ``S``/``T`` -- plain cluster-supercell overlap/kinetic (short-ranged, gauge-free);
  the periodicity lives entirely in the Coulomb kernel, consistently with the wire
  four-center / ``V_ne`` / ``E_nn`` (all built on the plain supercell AO pairs).
* ``V_ne``/``E_nn``/``J``/``K`` -- the wire builders
  (:func:`~vibeqc.periodic.ccm.lowd_four_center.ccm_wire_v_ne`,
  :func:`~vibeqc.periodic.ccm.lowd_four_center.ccm_wire_e_nn`, and the wire cderi
  ``B``), all sharing the single conditional-channel gauge constant ``c`` (set by
  ``g_perp_min``). For a **neutral** cell the Hartree gauge cancels exactly
  (``½c(N_e-N_nuc)²=0``; gated in ``tests/test_ccm_lowd_four_center.py``), so the
  Hartree side is gauge-free.

Exchange has no nuclear counterpart, so the exchange-``q=0`` term is the one
surviving convention (the low-D image of the D1/D2 finding):

Every convention is one constant times ``S D S`` added to ``K``, and the SCF response
is exactly linear: ``E_supercell(ξ) = E_strict − (N_e/2)·ξ`` (verified
``dE/dξ = −2.000000`` on the H₂ chain). What separates them is the **limit**.

* ``exxdiv="strict"`` -- the **strict-zero-mode**: the whole ``S⊗S``
  monopole of the four-center is projected out of ``K`` (``K → K − c*·S D S``,
  ``c* = <g,S⊗S>/<S⊗S,S⊗S>``). Infrared-stable (4e-6 across a 100× ``g_perp_min``
  sweep) but **not physical for absolute energies**: ``c*`` contains the ordinary
  free-space (molecular) ``S⊗S`` exchange as well as the IR-divergent conditional
  channel, and projecting both out leaves a constant error that does *not* vanish
  with the period. On a chain of well-separated neutral H₂/STO-3G it converges to
  ``E(isolated H₂) + 0.337`` Ha and is still drifting at ``L = 36`` bohr
  (measured 2026-07-10). Equivalent to the direct route's ``exxdiv=None``
  (:data:`~vibeqc.periodic.exchange_convention.STRICT_ZERO`).
* ``exxdiv="free"`` (default) -- subtract only the IR-divergent part, i.e. seam
  ``ξ = c*_free`` (:func:`~vibeqc.periodic.ccm.lowd_four_center.ccm_wire_free_ss_weight`).
  Infrared-stable (3e-6 across a decade) **and** correct in the non-interacting
  limit: isolated H₂ is approached as ``0.194/L`` with no constant offset
  (``dE·L`` = 0.1955/0.1942/0.1937/0.1936 at ``L`` = 12/18/26/36 bohr). This is the
  physically motivated choice for a ``d=1`` absolute energy.
* ``exxdiv=None`` -- the raw quadrature exchange (diagnostic only; keeps the whole
  ``½c·N_e`` seam, so the total drifts with ``g_perp_min`` -- measured 0.128 Ha per
  H₂ per decade -- even though its ``L→∞`` limit is right).

The **wire-Madelung** exchange seam (``exxdiv="ewald"``, the low-D analogue of
the direct route's ``ξ_N·S D S``) adds ``ξ_wire·S D S`` on top of the strict-zero
mode. Its ``ξ_wire`` vanishes as ``L → ∞``, so no choice of ``ρ₀`` can restore the
free-space ``S⊗S`` content that ``"strict"`` removed: the ``ρ₀`` question
(``handovers/HANDOVER_D2_EXXDIV.md``) is not what stands between this route and a
correct absolute energy. ``"free"`` is.

.. note::
   ``"free"`` became the default on 2026-07-10 (maintainer-approved). Callers that
   want the old absolute energies must pass ``exxdiv="strict"`` explicitly; the shift
   is ``c*_free`` (~0.35 Ha per H₂ at ``L = 12`` bohr) and ``.exchange_q0``
   distinguishes the two in the manifest. No ``d<3`` wire number for a core-bearing
   system may be quoted regardless, until the converged ``g_max × n_ang`` study lands
   (``FINDINGS.md`` F5(b′)).

Validation (all reference-free, ``tests/test_ccm_lowd_scf.py``):

* the strict total is ``g_perp_min``-independent;
* dense four-center SCF ``==`` RI-cderi SCF to ~1e-12 -- the **M5 gap-collapse**:
  the wire route's ``4c − RI`` is machine-zero, versus the ~22 mHa gauge offset a
  3-D vacuum-padded cell shows (``docs/aiccm2026dev_a_lowd_greens.md`` §0);
* the strict total from the quadrature ``==`` the strict total assembled from
  **exact ``erf``/``erfc`` image-sum integrals** (~1e-6) -- end-to-end proof the
  quadrature computes the correct wire electrostatics, not merely a self-consistent
  number.

Small-cluster / validation path (dense ``ρ̃``); see the module for memory guards.
"""

from __future__ import annotations

import numpy as np

from .experimental import _warn_experimental
from .lowd_four_center import ccm_wire_cderi, ccm_wire_e_nn, ccm_wire_v_ne
from .ri import _rhf_loop, ccm_ri_j_neutral, ccm_ri_k_neutral
from .scf import _ccm_initial_guess, _with_ccm_guess, _validate_conv_tol_grad

__all__ = ["run_ccm_rhf_wire"]


class _StreamedJK:
    """One streamed quadrature pass per density, serving both ``J`` and ``K``.

    ``_rhf_loop`` evaluates ``F = h + j_of_D(D) - 0.5*k_of_D(D)`` with the *same* ``D``
    object back to back, so a one-slot memo keyed on identity halves the work: the
    integral-direct path rebuilds the pair transform once per SCF iteration rather
    than twice.
    """

    def __init__(self, ccm, **qkw):
        self._ccm, self._qkw = ccm, qkw
        self._d = None
        self._jk = None

    def _ensure(self, D):
        if self._d is not D:
            from .lowd_four_center import ccm_wire_jk_stream

            self._jk = ccm_wire_jk_stream(self._ccm, D, **self._qkw)
            self._d = D
        return self._jk

    def j(self, D):
        return self._ensure(D)[0]

    def k(self, D):
        return self._ensure(D)[1]


def _wire_ss_gauge_coefficient(B, S):
    """``c* = <g,S⊗S>/<S⊗S,S⊗S>`` for ``g = Σ_Q B⊗B`` -- the S⊗S monopole weight
    of the wire four-center, needed only through the ``S``-projection (no dense
    ``g``)."""
    SdotB = np.einsum("mn,Qmn->Q", S, B, optimize=True)     # <S, B_Q>
    ss2 = float(np.sum(S * S))
    return float(np.sum(SdotB * SdotB)) / (ss2 * ss2)


def run_ccm_rhf_wire(ccm, *, initial_guess: object = "AUTO", axis=0, cderi=None, exxdiv="free", exxdiv_xi=None,
                     g_max=None, n_rad=64, n_ang=64, m_max=None, g_perp_min=1e-4,
                     e_nn_beta=None, max_iter=128, conv_tol=1e-9,
                     conv_tol_grad=1e-6, diis_dim=8,
                     lindep_tol=1e-7, stream=None):
    """Closed-shell HF-CCM on the wire kernel (see the module docstring).

    Parameters
    ----------
    axis : int
        Periodic (chain) direction; the cluster lattice must be axis-aligned.
    cderi : ndarray, optional
        Reuse a prebuilt wire cderi ``B`` (:func:`ccm_wire_cderi`) -- **must** be
        built with the same ``axis``/quadrature kwargs.
    exxdiv : {"strict", "free", "ewald", None}
        Exchange-``q=0`` convention; see the module docstring for the measurements.
        All four differ only by a constant times ``S D S``, converge to the same
        density, and shift the energy by ``−(N_e/2)·ξ`` per supercell.
        ``"strict"`` projects the whole ``S⊗S`` monopole out of ``K``:
        IR-stable, but it also deletes the free-space exchange content and so misses
        the non-interacting limit by a constant. ``"free"`` subtracts only the
        IR-divergent part (``ξ = c*_free``): IR-stable *and* correct in that limit
        (``O(1/L)``) -- the default since 2026-07-10. ``"ewald"`` adds the
        wire-Madelung seam ``ξ_wire`` on top of ``"strict"``. ``None`` keeps the raw
        quadrature exchange (diagnostic; ``g_perp_min``-dependent).
    exxdiv_xi : float, optional
        Override the seam constant. For ``exxdiv="ewald"`` it defaults to
        :func:`~vibeqc.periodic.lowd_greens.wire_self_energy` at the **cluster**
        (supercell) period, not the unit period. For ``exxdiv="free"`` it defaults to
        :func:`~vibeqc.periodic.ccm.lowd_four_center.ccm_wire_free_ss_weight`.
        Note ``ξ_wire → 0`` as ``L → ∞`` while the free-space ``S⊗S`` weight does
        not, so the ``ρ₀`` freedom in ``ξ_wire``
        (``handovers/HANDOVER_D2_EXXDIV.md``) cannot repair ``"strict"``'s constant
        offset; that is what ``"free"`` is for.
    stream : bool, optional
        Integral-direct: never form the dense cderi ``B``, rebuilding the quadrature
        in chunks for each ``J``/``K`` pass instead. ``None`` (default) auto-selects
        it when the dense ``B`` would exceed the 2 GiB guard. Because
        :func:`~vibeqc.periodic.ccm.lowd_four_center.required_g_max` scales with the
        **basis** and not the cell, this is what makes core-bearing elements
        reachable at all: polyethylene/STO-3G needs ``g_max ≈ 109``, whose dense
        ``B`` is 8.1 GiB, while the streamed peak is 0.25 GiB and independent of the
        quadrature size. Trades memory for one quadrature rebuild per SCF iteration.
        Incompatible with a prebuilt ``cderi``.

    Returns
    -------
    CCMSCFResult
        Energies per supercell; ``.exchange_q0`` records the convention label
        (``strict-zero-mode`` / ``BvK-ewald``).
    """
    selection = _ccm_initial_guess(ccm, initial_guess, driver="run_ccm_rhf_wire")
    conv_tol_grad = _validate_conv_tol_grad(
        conv_tol_grad, who="run_ccm_rhf_wire"
    )
    _warn_experimental()
    from vibeqc._vibeqc_core import compute_kinetic, compute_overlap
    from vibeqc.periodic.exchange_convention import exchange_q0_label

    if exxdiv not in ("strict", "free", "ewald", None):
        raise ValueError(
            "run_ccm_rhf_wire: exxdiv must be 'strict'/'free'/'ewald'/None, "
            f"got {exxdiv!r}")

    # The reciprocal cutoff is basis-adaptive, but the transverse angular grid is
    # not yet proved converged for tight core orbitals.  Carbon-chain calculations
    # consequently reach the SCF and oscillate at a quadrature-dependent energy.
    # Fail before allocating/building the very large streamed pair transform: a
    # visible convergence failure after minutes of setup is not a capability gate.
    # H/He remain the validated small-system control envelope; all atoms with a
    # frozen/tight core stay gated until the joint g_max x n_ang study lands.
    core_atoms = sorted({int(atom.Z) for atom in ccm.supercell.atoms if int(atom.Z) > 2})
    if core_atoms:
        symbols = ", ".join(str(z) for z in core_atoms)
        raise NotImplementedError(
            "run_ccm_rhf_wire: core-bearing wire SCF is not validated "
            f"(atomic numbers: {symbols}); the basis-adaptive radial cutoff is "
            "implemented, but the transverse angular quadrature is not yet "
            "converged for tight cores. Use this experimental route only for "
            "H/He until the g_max x n_ang validation gate is closed."
        )

    # Resolve g_max ONCE from the basis: J, V_ne and E_nn must ride the *same*
    # quadrature or the conditional gauge does not cancel (see the module docstring).
    from .lowd_four_center import (
        _MAX_BYTES,
        _resolve_g_max,
        ccm_wire_free_ss_weight,
        max_transverse_offset,
        ccm_wire_ss_gauge_stream,
        ccm_wire_v_ne_stream,
        wire_quadrature,
    )

    g_max = _resolve_g_max(ccm.basis, g_max, what="run_ccm_rhf_wire")
    # Offset-adaptive azimuthal grid (G-CCM-004): the azimuthal bandwidth of
    # an off-axis pair scales as |G_perp| x transverse offset (Jacobi-Anger;
    # trapezoid convergence per Trefethen & Weideman 2014), so a flat n_ang
    # silently under-resolves tight-core off-axis pairs -- the recorded
    # +22.69 Ha/atom polyethylene garbage at n_ang = 32..64. b_max = 0 (all
    # atoms on the axis, the validated H/He wires) keeps the flat grid.
    b_max = max_transverse_offset(ccm.supercell, axis=axis)
    qkw = dict(axis=axis, g_max=g_max, n_rad=n_rad, n_ang=n_ang, m_max=m_max,
               g_perp_min=g_perp_min, b_max=b_max)

    # Auto-select the integral-direct path when the dense cderi would not fit. The
    # cutoff scales with the BASIS, not the cell (:func:`required_g_max`), so
    # core-bearing elements blow the dense guard: C/STO-3G on polyethylene needs
    # g_max ≈ 109, an 8.1 GiB B. Streaming caps residency at one chunk (0.25 GiB)
    # and pays a quadrature rebuild per SCF iteration instead -- the integral-direct
    # bargain (Almlöf; Neese, Wennmohs, Hansen & Becker, Chem. Phys. 356, 98 (2009)).
    if stream is None:
        if cderi is not None:
            stream = False
        else:
            n_q = wire_quadrature(
                float(ccm.cluster_vectors[axis, axis]), g_max=g_max, n_rad=n_rad,
                n_ang=n_ang, m_max=m_max, g_perp_min=g_perp_min, axis=axis,
                b_max=b_max)[0].shape[0]
            stream = 16 * n_q * int(ccm.nbf) ** 2 > _MAX_BYTES
    if stream and cderi is not None:
        raise ValueError(
            "run_ccm_rhf_wire: stream=True cannot reuse a prebuilt cderi (the "
            "streamed path never forms B). Pass stream=False or drop cderi.")

    B = None if stream else (
        ccm_wire_cderi(ccm, **qkw) if cderi is None else np.asarray(cderi, float))

    S = np.asarray(compute_overlap(ccm.basis), dtype=float)
    T = np.asarray(compute_kinetic(ccm.basis), dtype=float)
    h = T + (ccm_wire_v_ne_stream(ccm, **qkw) if stream
             else ccm_wire_v_ne(ccm, cderi=B, **qkw))
    e_nn = ccm_wire_e_nn(ccm, axis=axis, beta=e_nn_beta, g_max=g_max, n_rad=n_rad,
                         n_ang=n_ang, m_max=m_max, g_perp_min=g_perp_min,
                         b_max=b_max)

    if stream:
        jk = _StreamedJK(ccm, **qkw)
        j_of_D, k0 = jk.j, jk.k
    else:
        j_of_D = lambda D: ccm_ri_j_neutral(B, D)
        k0 = lambda D: ccm_ri_k_neutral(B, D)

    if exxdiv in ("strict", "free", "ewald"):
        cstar = (ccm_wire_ss_gauge_stream(ccm, S, **qkw) if stream
                 else _wire_ss_gauge_coefficient(B, S))
        seam = -cstar
        if exxdiv == "free":
            # subtract ONLY the IR-divergent part of c*: keep the free-space S⊗S
            # content, which is ordinary molecular exchange, not an artifact.
            seam += ccm_wire_free_ss_weight(ccm) if exxdiv_xi is None else float(exxdiv_xi)
        elif exxdiv == "ewald":
            if exxdiv_xi is None:
                from ..lowd_greens import wire_self_energy

                exxdiv_xi = wire_self_energy(float(ccm.cluster_vectors[axis, axis]))
            seam += float(exxdiv_xi)             # K_strict + ξ_wire·S D S
        k_of_D = lambda D: k0(D) + seam * (S @ D @ S)
    else:
        k_of_D = k0

    res = _rhf_loop(ccm, S, h, e_nn, j_of_D, k_of_D,
                    max_iter=max_iter, conv_tol=conv_tol,
                    conv_tol_grad=conv_tol_grad, diis_dim=diis_dim,
                    lindep_tol=lindep_tol)
    res.exchange_q0 = exchange_q0_label(
        {"ewald": "ewald", "free": "free", "strict": None}.get(exxdiv, exxdiv))
    return _with_ccm_guess(res, selection)
