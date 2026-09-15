"""Low-dimensional neutral four-center for the A-line wire control (D3 M2b).

The mixed-boundary four-center ``(μν|G^{1D}|rs)`` for a 1-D-periodic cluster:
periodic along the chain axis (period = the *cluster* length ``L_c``), **free**
(open) in the transverse plane. This is the four-center whose kernel is the wire
Green's function of :mod:`vibeqc.periodic.lowd_greens` -- no spurious transverse
periodic images, unlike the 3-D-periodic ``ccm_eri_neutral`` on a vacuum-padded
cell.

**Construction identity.** This is an A-line mixed-boundary wire
construction/control. It is not currently classified as Γ-CCM because no
binding to the union-and-weight/Wigner--Seitz integral-weighting construction
has been derived. Neither A-line namespace ownership nor evaluation of the
finite cluster in a real-Gamma representation confers that identity. The same
implementation also does not define the separately constructed χ-CCM-B
Hamiltonian.

Construction (``docs/aiccm2026dev_a_lowd_greens.md`` § 9). The kernel is
translation-invariant with the reciprocal representation

.. math::

    G^{1D}(r) = \\frac{1}{L_c} \\sum_{m\\in\\mathbb{Z}} \\int
        \\frac{d^2 G_\\perp}{(2\\pi)^2}\\; e^{i G\\cdot r}\\,
        \\frac{4\\pi}{|G|^2},\\qquad G = (G_\\perp,\\, 2\\pi m/L_c),

i.e. a **discrete** sum over the periodic direction and a **continuous** integral
over the open (transverse) directions -- replacing that integral by a discrete
lattice sum is exactly the spurious-image error of the 3-D route. The four-center
is then the standard reciprocal contraction of AO-pair Fourier transforms
(:func:`vibeqc._aopair_ft.ao_pair_fourier_transform`, ``ρ̃(G→0) = S`` verified):

.. math::

    (\\mu\\nu|G^{1D}|rs) = \\frac{1}{L_c}\\sum_m \\int
        \\frac{d^2G_\\perp}{(2\\pi)^2}\\;
        \\tilde\\rho_{\\mu\\nu}(G)^*\\, \\frac{4\\pi}{|G|^2}\\,
        \\tilde\\rho_{rs}(G),

evaluated on a log-radial × uniform-angular transverse quadrature (validated to
~1e-9 against the analytic ``Σ_n erf(p d_n)/d_n`` image sum for Gaussian charges,
in gauge-cancelling difference form). Because ``v(G)`` is diagonal, the quadrature
**factorises as a real cderi** ``B[Q,μν]`` with ``g = Σ_Q B⊗B`` -- so the entire
CCM RI/DLPNO correlation stack rides it unchanged (``run_ccm_mp2(cderi=B)``, …).

**Gauge.** The ``m=0, G_\\perp→0`` region is infrared-log-divergent (the
conditional line/neutralising channel of the wire kernel); the quadrature cuts it
at ``g_perp_min``, leaving the physical tensor defined up to ``c·S⊗S`` -- the same
single-constant gauge freedom the M3a scalar reduction and the M3b gate allow.
The omitted non-``S⊗S`` content is ``O(g_perp_min²)``.

**Validation (the M3b gate, reference-free).** For an s-only basis the exact wire
four-center is the analytic image sum ``Σ_n erf(μ d_n)/d_n`` (Gaussian product
theorem; ``d→0`` limit ``2μ/√π``): the quadrature matches it to **2.8e-8**
Frobenius up to the single ``c·S⊗S`` constant
(``tests/test_ccm_lowd_four_center.py``). Note the 3-D vacuum-ladder comparison
(``ccm_eri_neutral`` on ``[L,D,D]`` with growing ``D``) is *not* a clean gate: its
gauge drifts as the theory predicts (``c ∝ −(2/L)ln D``), but the non-gauge
residual is dominated by the 3-D route's own finite-``D`` vacuum-cell artifacts
(~``D^{2/3}``-scaling numerical error, aux/FFT), not by this tensor.

Small-cluster / validation path: dense ``ρ̃`` over the full quadrature. Reference:
Rozzi et al., Phys. Rev. B 73, 205119 (2006) (mixed-boundary Coulomb cutoffs);
Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014) (CCM).
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc

__all__ = ["wire_quadrature", "required_n_ang", "max_transverse_offset",
           "ccm_wire_cderi", "ccm_eri_wire", "ccm_wire_v_ne",
           "ccm_wire_e_nn", "required_g_max", "wire_cderi_chunks",
           "ccm_wire_jk_stream", "ccm_wire_ss_gauge_stream", "ccm_wire_v_ne_stream",
           "ccm_wire_free_ss_weight"]

_MAX_BYTES = 2 * 1024**3     # dense-ρ̃ guard (complex128 nQ·nbf²)
_CHUNK_BYTES = 256 * 1024**2  # streamed-ρ̃ residency per chunk
_G_MAX_TOL = 1e-9            # target relative truncation of the pair-FT tail


def _pair_ft_libint(basis, G_pts):
    """AO-pair FT in **libint's** AO convention: ``(nbf, nbf, nG)``.

    :func:`vibeqc._aopair_ft.ao_pair_fourier_transform` returns the pair
    transform in the raw real-solid-harmonic convention, in which
    ``rho(G=0)`` equals the overlap for ``l = 0`` but is scaled by
    ``(2l+1)/(4 pi)`` for ``l >= 1`` (measured exactly: 0.23873241 = 3/4pi
    for p, 0.39788736 = 5/4pi for d, identical across H/He/C and three
    basis sets). libint absorbs ``Y_00 = 1/sqrt(4 pi)`` into the s-shell
    contraction coefficient but does **not** absorb ``Y_lm`` for
    ``l > 0``, so every such AO needs ``sqrt(4 pi / (2l+1))`` to match the
    overlap/density convention the rest of the code uses. See
    :func:`vibeqc.bipole_ext_el_pole._libint_ylm_correction_per_ao`,
    which the BIPOLE stack applies for the same reason.

    Omitting it is what made every core-bearing wire total wrong by
    Hartrees per atom while the H/He envelope stayed correct: the factor
    is exactly 1.0 for an s-only basis, so every H/He identity, both erf
    image-sum oracles and the M3b/M5 gates were structurally blind to it
    (root-caused 2026-08-02; ``HANDOVER_OPEN_BUGS_V015.md``). Because of
    that, an s-only basis must come out **bit-identical** through this
    helper -- the regression that proves it is a pure convention fix.
    """
    from ..._aopair_ft import ao_pair_fourier_transform
    from ...bipole_ext_el_pole import _libint_ylm_correction_per_ao

    rho = ao_pair_fourier_transform(basis, G_pts)      # (nbf, nbf, nG)
    f = _libint_ylm_correction_per_ao(basis)
    if np.all(f == 1.0):                               # s-only: untouched
        return rho
    return rho * f[:, None, None] * f[None, :, None]


def required_g_max(basis, tol=_G_MAX_TOL):
    """Smallest reciprocal cutoff that resolves this basis's tightest AO pair.

    Conventions, stated explicitly because the arithmetic is easy to redo with the
    wrong one (a 2026-07-10 theory-chat audit caught this function mixing two):

    * ``e_max`` is the **largest primitive exponent anywhere in the basis**, not a
      per-shell or contracted value.
    * ``τ = tol`` is the **relative** tail left in the pair transform (default 1e-9).
    * the decaying object is the **AO pair**, not the AO.

    By the Gaussian product theorem a primitive pair with exponents ``a``, ``b`` has
    combined exponent ``p = a + b`` and pair transform ``ρ̃(G) ∝ exp(-G²/4p)``, so
    truncating the quadrature at ``g_max`` leaves a relative tail ``exp(-g_max²/4p)``.
    Larger ``p`` decays *slower*, so the **tightest pair** ``p = 2·e_max`` (the max
    primitive exponent with itself) sets the cutoff:

    .. math::  g_\\text{max} \\;>\\; \\sqrt{4\\,p\\,\\ln(1/\\tau)}
                              \\;=\\; \\sqrt{8\\,e_\\text{max}\\,\\ln(1/\\tau)} .

    **This scales with the basis, not the cell.** At ``τ=1e-9``: H/STO-3G
    (``e_max=3.4253``) needs ``g_max > 23.8``; C/STO-3G (``e_max=71.6168``) needs
    ``g_max > 109.0``.

    A fixed default silently under-resolves core functions and produces a garbage
    four-center (measured 2026-07-09: the polyethylene wire SCF returned a *positive*
    energy and two IR gauges differed by 18 Ha/atom at ``g_max=16``). Hence the hard
    guard in :func:`_resolve_g_max`.

    .. note::
       Until 2026-07-10 this substituted ``p = e_max`` rather than ``p = 2·e_max``,
       returning a cutoff too small by ``√2`` (16.85 for H, 77.05 for C) while the
       docstring quoted 109 for C from the correct pair convention. The polyethylene
       sweep corroborates the correction: ``g_max = 80`` clears the old criterion's
       77.05 and the SCF still does not converge.
    """
    e_max = max(float(e) for sh in basis.shells() for e in np.atleast_1d(sh.exponents))
    p_max = 2.0 * e_max                      # tightest AO pair: e_max with itself
    return float(np.sqrt(4.0 * p_max * np.log(1.0 / float(tol))))


def _resolve_g_max(basis, g_max, *, what):
    """Auto-select ``g_max`` from the basis, or reject a value that cannot resolve it."""
    need = required_g_max(basis)
    if g_max is None:
        return need
    if float(g_max) < need:
        raise ValueError(
            f"{what}: g_max={float(g_max):.3g} cannot resolve this basis (needs "
            f">= {need:.3g} for a {_G_MAX_TOL:g} pair-FT tail; max primitive exponent "
            f"{max(float(e) for sh in basis.shells() for e in np.atleast_1d(sh.exponents)):.1f}). "
            "An under-resolved wire four-center is NOT merely inaccurate -- it loses "
            "positivity and the IR-gauge projection, so the SCF can converge to a "
            "positive energy. Pass g_max=None to auto-select, or raise it explicitly."
        )
    return float(g_max)


def required_n_ang(g, b_max, *, n_ang_min=8, safety=1.25, pad=8):
    """Azimuthal points needed at transverse radius ``g`` for offsets ``b_max``.

    The azimuthal integrand at fixed ``|G_perp| = g`` is a **product of two**
    reciprocal-space objects, each carrying a translation phase
    ``e^{-i G_perp . b}`` from its transverse offset: for the four-center
    ``rho*_{munu}(G) rho_{rs}(G)``, for ``V_ne`` a pair transform against
    the bare nuclear structure factor, for ``E_nn`` ``|S(G)|^2``. By the
    Jacobi-Anger expansion each phase has Bessel-weighted azimuthal
    content ``J_k(g b)``, negligible beyond ``k ~ g b``, so the
    **product** has bandwidth ``g (b_1 + b_2)``, bounded by ``2 g b_max``
    -- not ``g b_max``. The uniform-azimuth trapezoid rule is exact for a
    trigonometric polynomial of degree below the point count and
    converges geometrically once the count exceeds the bandwidth
    (Trefethen & Weideman, SIAM Rev. 56, 385 (2014),
    doi:10.1137/130932132, SS 3 and SS 7 -- the periodic-analytic case),
    so the requirement is ``n_ang(g) ~ 2 g b_max`` plus a margin, NOT a
    constant.

    .. note::
       The factor 2 landed 2026-08-02, a day after the rule itself: the
       first version used the single-object bandwidth ``g b_max``, which
       is short by up to 1.75x in the band where the pair-FT envelope
       ``e^{-g^2/4p}`` is 1e-4..1e-7 (a ~1e-5 accuracy leak on the
       polyethylene-shaped fixture, not a Ha-scale one -- the flat
       ``n_ang = 32`` that produced the recorded +22.69 Ha/atom was short
       in a band where the envelope is still ~0.14). Caught by costing
       the fixture properly rather than by any test, because the H
       exponents in the oracle fixture kill the envelope before the
       shortfall bites.

    An **envelope-aware cap** (skipping nodes where ``e^{-g^2/4 p_max}``
    is already below the target precision) does not pay here and was
    measured, not assumed: it costs slightly *more* (1.476x vs 1.435x of
    the old flat grid on the polyethylene fixture) because ``V_ne``'s
    bare nuclear structure factor carries no Gaussian envelope at all and
    sets the large-``g`` bandwidth on its own.

    On-axis systems (``b_max = 0``: every atom on the wire axis, e.g. the
    validated H/He wires) collapse to ``n_ang_min``, which is also a
    ~8x cheaper build than the old flat default.
    """
    n = int(np.ceil(2.0 * float(safety) * float(g) * float(b_max))) + int(pad)
    n = max(int(n_ang_min), n)
    return n + (n % 2)  # even count keeps the +/-phi symmetry exact


def max_transverse_offset(system, *, axis=0):
    """Largest transverse distance of any atom from the wire axis.

    Pair centres are convex combinations of atom centres, so the atomic
    maximum bounds every pair's translation offset. The pair-FT *envelope*
    (Gaussian width) carries no additional azimuthal oscillation -- only
    the centre phase does -- so this bound is the right bandwidth input
    for :func:`required_n_ang`.
    """
    tr = [i for i in range(3) if i != int(axis)]
    b = 0.0
    for atom in system.atoms:
        r = np.asarray(atom.xyz, dtype=float)
        b = max(b, float(np.hypot(r[tr[0]], r[tr[1]])))
    return b


def wire_quadrature(period, *, g_max=16.0, n_rad=64, n_ang=64,
                    m_max=None, g_perp_min=1e-4, axis=0,
                    b_max=None):
    """Reciprocal quadrature for the wire kernel: points ``G_Q`` and weights
    ``w_Q · 4π/|G_Q|² / L`` (the kernel folded into the weight).

    ``m_max`` defaults to covering ``|g_m| ≤ g_max`` (isotropic G coverage).
    Returns ``(G_pts (nQ,3), wv (nQ,))``.

    ``b_max`` switches the azimuthal grid from the flat ``n_ang`` to the
    offset-adaptive per-radial-node count :func:`required_n_ang` --
    ``n_ang`` then acts as the floor. Pass the system's
    :func:`max_transverse_offset` (0.0 is valid and gives the floor
    everywhere). ``None`` preserves the historical flat grid exactly.
    """
    from numpy.polynomial.legendre import leggauss

    L = float(period)
    if m_max is None:
        m_max = int(np.ceil(g_max * L / (2.0 * np.pi)))
    xs, ws = leggauss(int(n_rad))
    lo, hi = np.log(float(g_perp_min)), np.log(float(g_max))
    t = 0.5 * (xs + 1.0) * (hi - lo) + lo
    wt = ws * 0.5 * (hi - lo)
    Gr = np.exp(t)                                   # radial |G_perp| (log grid)

    def _angular(n):
        return (
            2.0 * np.pi * np.arange(int(n)) / int(n),
            2.0 * np.pi / int(n),
        )

    tr = [i for i in range(3) if i != axis]          # transverse Cartesian axes
    pts, wv = [], []
    for m in range(0, int(m_max) + 1):
        gz = 2.0 * np.pi * m / L
        wm = 1.0 if m == 0 else 2.0                  # ±m folded (real AOs)
        for ir in range(len(Gr)):
            G = Gr[ir]
            if b_max is None:
                phis, wphi = _angular(int(n_ang))
            else:
                phis, wphi = _angular(
                    required_n_ang(G, b_max, n_ang_min=int(n_ang))
                )
            v = 4.0 * np.pi / (G * G + gz * gz)
            # d²G_perp = G dG dφ = G² d(lnG) dφ
            w = wm * G * G * wt[ir] * wphi / (2.0 * np.pi) ** 2 * v / L
            for ia in range(len(phis)):
                p = np.zeros(3)
                p[axis] = gz
                p[tr[0]] = G * np.cos(phis[ia])
                p[tr[1]] = G * np.sin(phis[ia])
                pts.append(p)
                wv.append(w)
    return np.asarray(pts, dtype=float), np.asarray(wv, dtype=float)


def ccm_wire_cderi(ccm, *, axis=0, g_max=None, n_rad=64, n_ang=64,
                   m_max=None, g_perp_min=1e-4, b_max=None):
    """Real wire-kernel cderi ``B[Q,μν]`` on the cluster basis: ``g^{1D} = Σ_Q B⊗B``.

    ``axis`` is the periodic (chain) direction; the cluster lattice must be
    axis-aligned (diagonal) so the transverse plane is Cartesian. Two rows per
    quadrature point (``√w·Re ρ̃``, ``√w·Im ρ̃``). Drop-in for every ``cderi=``
    consumer of the CCM correlation stack (defined up to the ``c·S⊗S`` gauge).
    """
    from ..._aopair_ft import ao_pair_fourier_transform

    Ac = np.asarray(ccm.cluster_vectors, dtype=float)
    off = Ac - np.diag(np.diag(Ac))
    if np.max(np.abs(off)) > 1e-10 * max(np.max(np.abs(Ac)), 1.0):
        raise NotImplementedError(
            "ccm_wire_cderi: needs an axis-aligned (diagonal) cluster lattice "
            "so the open transverse plane is Cartesian.")
    L = float(Ac[axis, axis])
    g_max = _resolve_g_max(ccm.basis, g_max, what="ccm_wire_cderi")

    G_pts, wv = wire_quadrature(L, g_max=g_max, n_rad=n_rad, n_ang=n_ang,
                                m_max=m_max, g_perp_min=g_perp_min, axis=axis,
                                b_max=b_max)
    nbf = int(ccm.nbf)
    need = 16 * G_pts.shape[0] * nbf * nbf
    if need > _MAX_BYTES:
        raise MemoryError(
            f"ccm_wire_cderi: dense AO-pair FT would need {need/1024**3:.1f} GiB "
            f"({G_pts.shape[0]} quadrature points x {nbf}^2 AO pairs); reduce "
            "n_rad/n_ang/m_max or the basis (small-cluster validation path).")
    rho = _pair_ft_libint(ccm.basis, G_pts)                # (nbf, nbf, nQ)
    rho = np.ascontiguousarray(np.moveaxis(rho, -1, 0))    # (nQ, nbf, nbf)
    sw = np.sqrt(wv)[:, None, None]
    return np.concatenate([sw * rho.real, sw * rho.imag], axis=0)


def _wire_quadrature_for(ccm, axis, g_max, n_rad, n_ang, m_max, g_perp_min,
                         what, b_max=None):
    """Shared setup: validate the lattice, resolve ``g_max``, build the quadrature."""
    Ac = np.asarray(ccm.cluster_vectors, dtype=float)
    off = Ac - np.diag(np.diag(Ac))
    if np.max(np.abs(off)) > 1e-10 * max(np.max(np.abs(Ac)), 1.0):
        raise NotImplementedError(
            f"{what}: needs an axis-aligned (diagonal) cluster lattice "
            "so the open transverse plane is Cartesian.")
    L = float(Ac[axis, axis])
    g_max = _resolve_g_max(ccm.basis, g_max, what=what)
    G_pts, wv = wire_quadrature(L, g_max=g_max, n_rad=n_rad, n_ang=n_ang,
                                m_max=m_max, g_perp_min=g_perp_min, axis=axis,
                                b_max=b_max)
    return G_pts, wv


def wire_cderi_chunks(ccm, *, axis=0, g_max=None, n_rad=64, n_ang=64, m_max=None,
                      g_perp_min=1e-4, max_bytes=_CHUNK_BYTES, b_max=None):
    """Stream :func:`ccm_wire_cderi` in slices of the quadrature index ``Q``.

    Yields ``B_chunk`` of shape ``(2·nq_chunk, nbf, nbf)`` -- the same two real rows
    per quadrature point (``√w·Re ρ̃``, ``√w·Im ρ̃``) the dense builder emits, for a
    contiguous block of ``Q``. Concatenating every chunk along axis 0 reproduces
    :func:`ccm_wire_cderi` up to the ordering of its two half-blocks, which no
    consumer depends on: ``J``, ``K``, ``V_ne`` and the ``S⊗S`` gauge coefficient are
    all **sums over Q**, so none of them ever needs the whole tensor resident.

    This is what makes ``d = 1`` tractable for core-bearing elements. The dense
    tensor is ``2·nQ·nbf²·8`` bytes, and the cutoff scales with the basis, not the
    cell (:func:`required_g_max`): C/STO-3G needs ``g_max ≈ 109``, which is GiB-scale
    and trips the dense guard. Streaming caps residency at ``max_bytes`` and trades
    it for recomputation, the classic integral-direct bargain (Almlöf; see
    Neese, Wennmohs, Hansen & Becker, *Chem. Phys.* **356**, 98 (2009) for the
    modern SCF form).
    """
    from ..._aopair_ft import ao_pair_fourier_transform

    G_pts, wv = _wire_quadrature_for(
        ccm, axis, g_max, n_rad, n_ang, m_max, g_perp_min, "wire_cderi_chunks",
        b_max=b_max)
    nbf = int(ccm.nbf)
    per_q = 16 * nbf * nbf                       # complex128 rho for one Q
    nq_chunk = max(1, int(max_bytes) // per_q)
    n_q = G_pts.shape[0]
    for lo in range(0, n_q, nq_chunk):
        hi = min(lo + nq_chunk, n_q)
        rho = _pair_ft_libint(ccm.basis, G_pts[lo:hi])             # (nbf,nbf,nq)
        rho = np.ascontiguousarray(np.moveaxis(rho, -1, 0))        # (nq,nbf,nbf)
        sw = np.sqrt(wv[lo:hi])[:, None, None]
        yield np.concatenate([sw * rho.real, sw * rho.imag], axis=0)


def ccm_wire_jk_stream(ccm, D, *, axis=0, exchange=True, **kw):
    """Integral-direct wire ``J`` and ``K`` from a density, never forming ``B``.

    ``J = Σ_Q B_Q ⟨B_Q, D⟩`` and ``K = Σ_Q B_Q D B_Q`` are both accumulations over
    the quadrature index, so a single streamed pass over
    :func:`wire_cderi_chunks` yields both. Returns ``(J, K)``; ``K`` is ``None``
    when ``exchange=False``.

    Equals the dense ``ccm_ri_j_neutral``/``ccm_ri_k_neutral`` contraction of
    :func:`ccm_wire_cderi` to round-off (gated in
    ``tests/test_ccm_lowd_four_center.py``). Cost is one quadrature rebuild per
    call, so an SCF pays it once per iteration.
    """
    D = np.asarray(D, dtype=float)
    nbf = D.shape[0]
    J = np.zeros((nbf, nbf))
    K = np.zeros((nbf, nbf)) if exchange else None
    for B in wire_cderi_chunks(ccm, axis=axis, **kw):
        J += np.einsum("Qmn,Q->mn", B, np.einsum("Qrs,rs->Q", B, D, optimize=True),
                       optimize=True)
        if exchange:
            K += np.einsum("Qmr,rs,Qsn->mn", B, D, B, optimize=True)
    return J, K


def ccm_wire_v_ne_stream(ccm, *, axis=0, g_max=None, n_rad=64, n_ang=64, m_max=None,
                         g_perp_min=1e-4, max_bytes=_CHUNK_BYTES, b_max=None):
    """Streamed :func:`ccm_wire_v_ne`: contract the nuclear structure factor against
    the wire cderi one quadrature chunk at a time, never forming ``B``.

    Rides the identical quadrature as :func:`ccm_wire_jk_stream`, so it inherits the
    same conditional-channel gauge and the neutral-cell Hartree cancellation still
    holds exactly (``½c(N_e - N_nuc)² = 0``).
    """
    from ..._aopair_ft import ao_pair_fourier_transform

    G_pts, wv = _wire_quadrature_for(
        ccm, axis, g_max, n_rad, n_ang, m_max, g_perp_min, "ccm_wire_v_ne_stream",
        b_max=b_max)
    Z = np.array([int(a.Z) for a in ccm.supercell.atoms], dtype=float)
    R = np.asarray(ccm.atom_positions, dtype=float)
    nbf = int(ccm.nbf)
    per_q = 16 * nbf * nbf
    nq_chunk = max(1, int(max_bytes) // per_q)

    V = np.zeros((nbf, nbf))
    for lo in range(0, G_pts.shape[0], nq_chunk):
        hi = min(lo + nq_chunk, G_pts.shape[0])
        Gc = G_pts[lo:hi]
        rho = _pair_ft_libint(ccm.basis, Gc)                        # (nbf,nbf,nq)
        rho = np.ascontiguousarray(np.moveaxis(rho, -1, 0))         # (nq,nbf,nbf)
        sw = np.sqrt(wv[lo:hi])
        B = np.concatenate([sw[:, None, None] * rho.real,
                            sw[:, None, None] * rho.imag], axis=0)
        zc = np.exp(-1j * (Gc @ R.T)) @ Z
        v_row = np.concatenate([sw * zc.real, sw * zc.imag])
        V -= np.einsum("qmn,q->mn", B, v_row, optimize=True)
    return 0.5 * (V + V.T)


def ccm_wire_ss_gauge_stream(ccm, S, *, axis=0, **kw):
    """``c* = Σ_Q ⟨S, B_Q⟩² / (Σ S²)²`` -- the ``S⊗S`` monopole weight of the wire
    four-center, streamed. The strict-zero-mode exchange convention needs only this
    projection of ``g``, never ``g`` itself."""
    S = np.asarray(S, dtype=float)
    acc = 0.0
    for B in wire_cderi_chunks(ccm, axis=axis, **kw):
        sdotb = np.einsum("mn,Qmn->Q", S, B, optimize=True)
        acc += float(np.sum(sdotb * sdotb))
    ss2 = float(np.sum(S * S))
    return acc / (ss2 * ss2)


def ccm_wire_free_ss_weight(ccm, *, schwarz_threshold=1e-12):
    """``c*_free`` -- the ``S⊗S`` weight of the cluster's **free-space** four-center.

    ``c*_free = ⟨g_free, S⊗S⟩ / ⟨S⊗S, S⊗S⟩``, evaluated without forming ``g_free``:
    ``⟨g, S⊗S⟩ = Σ_{μν} S_{μν} J[S]_{μν}`` for the Coulomb matrix built from ``S`` as
    a density. Uses the **integral-direct** (Schwarz-screened) J builder, not the
    four-index one, so it does not reintroduce an ``O(n_bf⁴)`` wall in exactly the
    regime the streamed wire path exists for. Agrees with the four-index ``J[S]``
    to 4e-16 on the H₂ chain.

    **Why this constant is the physical exchange-``q=0`` seam of the wire.** The wire
    four-center's ``S⊗S`` weight ``c*`` splits into two pieces:

    * an infrared-divergent part from the ``m=0, G_⊥→0`` conditional channel, going
      like ``ln(1/g_perp_min)/L`` -- an artifact of where the quadrature is cut;
    * the **free-space** part, which is just the ordinary molecular ERI's ``S⊗S``
      content and is emphatically *not* an artifact.

    ``exxdiv="strict"`` projects out **both**, so it deletes real exchange and its
    error does not vanish with the period: on a chain of well-separated neutral H₂
    (STO-3G) it converges to ``E(isolated H₂) + 0.337`` Ha and is still drifting at
    ``L = 36`` bohr. ``exxdiv=None`` keeps both, so its total diverges with the
    infrared cut (measured 0.128 Ha per H₂ per decade of ``g_perp_min``).

    Subtracting only the divergent part -- i.e. the seam ``ξ = c*_free`` -- is both
    infrared-stable (3e-6 across a decade of ``g_perp_min``) and correct in the
    non-interacting limit, approaching isolated H₂ as ``0.194/L`` with no constant
    offset (measured 2026-07-10 at L = 12/18/26/36; ``dE·L`` = 0.1955/0.1942/0.1937/
    0.1936).

    The exact response is ``E_supercell(ξ) = E_strict − (N_e/2)·ξ``, verified to
    ``dE/dξ = −2.000000`` on the H₂ chain.
    """
    from vibeqc._vibeqc_core import compute_overlap, make_direct_jk_builder

    S = np.asarray(compute_overlap(ccm.basis), dtype=float)
    builder = make_direct_jk_builder(ccm.basis, float(schwarz_threshold))
    j_of_S = np.asarray(builder.build_J(S), dtype=float)
    return float(np.sum(S * j_of_S)) / float(np.sum(S * S)) ** 2


def ccm_eri_wire(ccm, *, cderi=None, **kw):
    """Dense wire four-center ``g^{1D}[μν,rs] = Σ_Q B⊗B`` (small clusters; the
    M3b gate path). ``cderi`` reuses a prebuilt :func:`ccm_wire_cderi`."""
    B = ccm_wire_cderi(ccm, **kw) if cderi is None else np.asarray(cderi, float)
    return np.einsum("Qmn,Qrs->mnrs", B, B, optimize=True)


def ccm_wire_v_ne(ccm, *, axis=0, cderi=None, g_max=None, n_rad=64, n_ang=64,
                  m_max=None, g_perp_min=1e-4, b_max=None):
    """Wire-kernel electron-nuclear attraction ``V_ne[μν]`` (D3 M4, two-center form).

    The nuclear side of the same mixed-boundary Poisson problem as
    :func:`ccm_wire_cderi`: the cluster nuclei ``{Z_A, R_A}`` (all ``N_c`` cells)
    seen through the wire kernel ``G^{1D}`` (periodic along ``axis``, open
    transverse). Built by contracting the **same real wire cderi** ``B`` against the
    nuclear structure factor ``Σ_A Z_A e^{-iG·R_A}`` on the shared quadrature:

    .. math::

        V_{ne}[\\mu\\nu] = -\\sum_A Z_A \\sum_Q w_Q\\,
            \\mathrm{Re}\\big[\\tilde\\rho_{\\mu\\nu}(G_Q)^*\\, e^{-iG_Q\\cdot R_A}\\big].

    Because it rides the identical ``B`` (hence the identical ``g_perp_min`` IR cut),
    ``V_ne`` inherits ``J``/``K``'s conditional-channel gauge **exactly**: its
    residual against the analytic image sum is a single ``c'·S`` with
    ``c' = -N_nuc·c`` (``c`` the four-center's ``c·S⊗S`` constant; verified
    ``Δc'/Δc = -N_nuc`` to 6 digits). That equality is what makes the neutral-cell
    Hartree gauge cancel in the total energy (``½c N_e² - c N_nuc N_e + ½c N_nuc²
    = ½c(N_e-N_nuc)² = 0``); the surviving gauge freedom is the exchange-``q=0``
    seam alone (``docs/aiccm2026dev_a_lowd_greens.md`` §10). ``cderi`` reuses a
    prebuilt :func:`ccm_wire_cderi` (built at the **same** quadrature kwargs).

    Validation (reference-free, s-only): matches the analytic image sum
    ``-Σ_A Z_A Σ_n erf(√p · d_n)/d_n`` (point-nucleus limit ``μ=√p`` of the pair
    Gaussian) up to that single ``c'·S`` (``tests/test_ccm_lowd_four_center.py``).
    """
    from ..._aopair_ft import ao_pair_fourier_transform  # noqa: F401 (parity import)

    Ac = np.asarray(ccm.cluster_vectors, dtype=float)
    off = Ac - np.diag(np.diag(Ac))
    if np.max(np.abs(off)) > 1e-10 * max(np.max(np.abs(Ac)), 1.0):
        raise NotImplementedError(
            "ccm_wire_v_ne: needs an axis-aligned (diagonal) cluster lattice "
            "so the open transverse plane is Cartesian.")
    L = float(Ac[axis, axis])
    g_max = _resolve_g_max(ccm.basis, g_max, what="ccm_wire_v_ne")

    G_pts, wv = wire_quadrature(L, g_max=g_max, n_rad=n_rad, n_ang=n_ang,
                                m_max=m_max, g_perp_min=g_perp_min, axis=axis,
                                b_max=b_max)
    B = (ccm_wire_cderi(ccm, axis=axis, g_max=g_max, n_rad=n_rad, n_ang=n_ang,
                        m_max=m_max, g_perp_min=g_perp_min, b_max=b_max)
         if cderi is None else np.asarray(cderi, dtype=float))
    if B.shape[0] != 2 * G_pts.shape[0]:
        raise ValueError(
            "ccm_wire_v_ne: cderi row count does not match the quadrature; pass a "
            "cderi built with the SAME g_max/n_rad/n_ang/m_max/g_perp_min/axis.")

    Z = np.array([int(a.Z) for a in ccm.supercell.atoms], dtype=float)
    R = np.asarray(ccm.atom_positions, dtype=float)
    sw = np.sqrt(wv)
    # Nuclear structure factor e^{-iG·R_A} weighted into B's (Re | Im) row layout,
    # so the real-cderi inner product B·v_row reproduces Σ_Q w_Q Re[ρ̃* Σ_A Z_A e^{-iG·R_A}].
    zc = np.exp(-1j * (G_pts @ R.T)) @ Z                 # (nQ,)  Σ_A Z_A e^{-iG·R_A}
    v_row = np.concatenate([sw * zc.real, sw * zc.imag])  # (2nQ,) matches B rows
    V = -np.einsum("qmn,q->mn", B, v_row, optimize=True)
    return 0.5 * (V + V.T)


def ccm_wire_e_nn(ccm, *, axis=0, beta=None, g_max=None, n_rad=64, n_ang=64,
                  m_max=None, g_perp_min=1e-4, n_real=None, b_max=None):
    """Wire-kernel nuclear repulsion per supercell (D3 M4, zero-center form).

    The nuclear point charges seen through the same mixed-boundary kernel
    ``G^{1D}``, by a **mini-Ewald on the 1-D lattice** (period ``L``, axis
    ``axis``): the short-range self / real-space parts use bare ``1/r`` (the wire
    kernel's short-range limit *is* ``1/r``, kernel-independent), while the
    long-range part rides the **same wire quadrature** as ``J``/``V_ne`` so the
    ``m=0, G_⊥→0`` region contributes ``½c·N_nuc²`` with the four-center's ``c``:

    .. math::

        E_{nn} = \\tfrac12\\sum_Q w_Q\\, e^{-|G_Q|^2/4\\beta^2}\\,|S(G_Q)|^2
               + \\tfrac12\\!\\!\\sum_{A,B,n}{}'\\, Z_A Z_B\\,
                 \\frac{\\mathrm{erfc}(\\beta d_{ABn})}{d_{ABn}}
               - \\frac{\\beta}{\\sqrt\\pi}\\sum_A Z_A^2,

    with ``S(G)=Σ_A Z_A e^{-iG·R_A}`` and the primed sum excluding the
    ``A=B, n=0`` self term. The split parameter ``β`` drops out (verified flat to
    ~1e-8), and ``E_nn``'s gauge coefficient ``½N_nuc²`` combines with ``J``'s
    (``½N_e²``) and ``V_ne``'s (``-N_nuc N_e``) so the neutral-cell Hartree total
    ``E_nn + Tr[D V_ne] + ½Tr[D J]`` is ``g_perp_min``-independent
    (``½c(N_e-N_nuc)²=0``); see :func:`ccm_wire_v_ne` and
    ``docs/aiccm2026dev_a_lowd_greens.md`` §10.

    ``β`` defaults to a value matched to ``L`` and ``g_max``; ``n_real`` defaults
    to the erfc cutoff. Returns a float (Ha per supercell).
    """
    Ac = np.asarray(ccm.cluster_vectors, dtype=float)
    off = Ac - np.diag(np.diag(Ac))
    if np.max(np.abs(off)) > 1e-10 * max(np.max(np.abs(Ac)), 1.0):
        raise NotImplementedError(
            "ccm_wire_e_nn: needs an axis-aligned (diagonal) cluster lattice.")
    L = float(Ac[axis, axis])
    g_max = _resolve_g_max(ccm.basis, g_max, what="ccm_wire_e_nn")
    Z = np.array([int(a.Z) for a in ccm.supercell.atoms], dtype=float)
    R = np.asarray(ccm.atom_positions, dtype=float)

    if beta is None:
        # real-space converges in a few cells; reciprocal Gaussian fits under g_max
        beta = float(max(0.2, min(3.0 / L, g_max / 8.0)))
    if g_max < 6.0 * beta:
        raise ValueError(
            f"ccm_wire_e_nn: g_max={g_max} too small for beta={beta:.3g} "
            "(need g_max >= 6*beta so the reciprocal Gaussian is covered).")
    if n_real is None:
        n_real = int(np.ceil(7.0 / (beta * L))) + 2

    # Reciprocal: Gaussian-nuclei Hartree on the shared quadrature (gauge = c).
    # The nuclear structure factor e^{-iG.R_A} oscillates azimuthally with
    # |G_perp| x offset exactly like an AO pair's phase, so it needs the
    # same offset-adaptive angular grid.
    G_pts, wv = wire_quadrature(L, g_max=g_max, n_rad=n_rad, n_ang=n_ang,
                                m_max=m_max, g_perp_min=g_perp_min, axis=axis,
                                b_max=b_max)
    G2 = np.einsum("qi,qi->q", G_pts, G_pts)
    Sq = np.exp(-1j * (G_pts @ R.T)) @ Z
    e_recip = 0.5 * float(np.sum(wv * np.exp(-G2 / (4.0 * beta**2)) * np.abs(Sq)**2))

    # Real space: erfc complement over 1-D images (bare 1/r short range).
    ns = np.arange(-int(n_real), int(n_real) + 1).astype(float)
    e_real = 0.0
    for iA in range(len(Z)):
        for iB in range(len(Z)):
            d = R[iA] - R[iB]
            dn = np.sqrt((d[0] + ns * L) ** 2 + d[1] ** 2 + d[2] ** 2)
            mask = dn > 1e-12                         # exclude A=B, n=0
            e_real += 0.5 * Z[iA] * Z[iB] * float(
                np.sum(erfc(beta * dn[mask]) / dn[mask]))

    e_self = (beta / np.sqrt(np.pi)) * float(np.sum(Z**2))
    return e_recip + e_real - e_self
