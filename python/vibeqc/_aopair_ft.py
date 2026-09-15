"""Analytic Fourier transform of AO pair densities chi_mu(r).chi_ν(r).

Used by the RSGDF long-range 3c tensor in :mod:`vibeqc.aux_basis`
(:func:`rsgdf_lr_3c_tensor`). See ``docs/design_rsgdf_3c_lr.md``
for the full math derivation and references.

This module is **incrementally** implemented, milestone-by-milestone:

* **Milestone 1 (TODAY)**: s x s pair (l_mu = l_ν = 0). No
  McMurchie-Davidson recursion needed -- closed-form Gaussian
  product theorem suffices.
* **Milestone 2 (NEXT)**: arbitrary L via McMurchie-Davidson
  Hermite expansion. Cartesian Gaussians.
* **Milestone 3 (NEXT)**: Cartesian -> spherical AO transform to
  match libint's basis-function ordering.

Until milestones 2 + 3 land, :func:`ao_pair_fourier_transform`
raises :class:`NotImplementedError` for any shell with L > 0.

Conventions
-----------

The AO basis function for libint convention (cf.
:func:`vibeqc.aux_basis._real_sph_harm` documentation):

* l = 0:  chi_s(r; a, R) = c . exp(-a |r-R|^2)               (no Y_00)
* l > 0:  chi_lm(r; a, R) = c . r_R^l . Y_lm(r̂_R) . exp(-a |r-R|^2)

where r_R = |r - R|, Y_lm is the real spherical harmonic, and c is
the libint-normalised contraction coefficient (which absorbs Y_00
for s-shells).

The Fourier transform convention matches the rest of the LR
machinery: F[f(r)](G) = ∫ f(r) e^{-i G . r} d^3r.

References
----------

- McMurchie, L. E. & Davidson, E. R. (1978). *J. Comput. Phys.*
  **26**, 218. DOI 10.1016/0021-9991(78)90092-X.
- Helgaker, T., Jorgensen, P. & Olsen, J. *Molecular
  Electronic-Structure Theory* (Wiley, 2000), Sec.9.3-9.5.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ._vibeqc_core import BasisSet


def _aopair_ft_backend() -> str:
    """Dispatch knob for the C++ <-> Python pair-FT split.

    ``VIBEQC_AOPAIR_FT_BACKEND``:

    * ``auto`` (default) -- use the C++ kernel when its preconditions
      are met (currently: s-only basis, no caller-supplied
      ``ft_per_cell``); fall back to pure Python otherwise.
    * ``python`` -- force the pure-Python reference path. Used by
      ``tests/test_aopair_ft_parity.py`` and as an emergency
      escape hatch.
    * ``cxx`` -- force the C++ kernel; raise if its preconditions
      aren't met. Useful for shaking out cases that should be
      dispatched but aren't.
    """
    return os.environ.get("VIBEQC_AOPAIR_FT_BACKEND", "auto").lower()


__all__ = [
    "ao_pair_fourier_transform",
    "ao_pair_fourier_transform_ss_only",
    "ao_pair_fourier_transform_shifted_ket",
    "ao_pair_fourier_transform_at_cells",
    "ao_pair_fourier_transform_bloch",
    "ao_pair_fourier_transform_bloch_multi",
    "ao_pair_fourier_transform_bloch_gradient_gweighted",
    # Internal/exported for testing:
    "md_e_coefficients_1d",
    "cartesian_gaussian_product_ft",
    "cartesian_components_for_l",
    "cart_to_sph_matrix",
]


def ao_pair_fourier_transform_ss_only(
    ao_basis: BasisSet,
    G_vectors: np.ndarray,
) -> np.ndarray:
    """FT of all AO-pair densities for an s-only basis.

    Computes ``r̂_muν(G) = ∫ chi_mu(r) chi_ν(r) e^{-iG.r} dr`` for every
    AO pair (mu, ν) and every G vector. For two contracted s-shells
    at centres A and B with primitives (a_mu_p, c_mu_p) and
    (a_ν_q, c_ν_q), the Gaussian product theorem gives:

        chi_mu(r) chi_ν(r) = S_{p,q} K_pq . exp(-g_pq |r - P_pq|^2)

    with g_pq = a_mu_p + a_ν_q,
         P_pq = (a_mu_p A + a_ν_q B) / g_pq,
         K_pq = c_mu_p c_ν_q . exp(-a_mu_p a_ν_q / g_pq . |A-B|^2).

    The FT of each Gaussian product term:

        F[K_pq . e^{-g_pq (r-P_pq)^2}](G) =
            K_pq . (pi/g_pq)^{3/2} . exp(-G^2/(4g_pq)) . exp(-iG.P_pq)

    Returns
    -------
    out : (n_orb, n_orb, n_G) complex128
        ``out[mu, ν, k]`` is the FT of (chi_mu . chi_ν) at G_vectors[k].

    Raises
    ------
    NotImplementedError
        If any shell in ``ao_basis`` has L > 0. Use
        :func:`ao_pair_fourier_transform` for general L (lands in
        milestone 2).
    """
    G_vectors = np.ascontiguousarray(G_vectors, dtype=float)
    n_G = G_vectors.shape[0]
    n_orb = ao_basis.nbasis
    out = np.zeros((n_orb, n_orb, n_G), dtype=np.complex128)

    shells = ao_basis.shells()
    # Validate s-only.
    for sh in shells:
        if int(sh.l) != 0:
            raise NotImplementedError(
                f"ao_pair_fourier_transform_ss_only: shell with L="
                f"{sh.l} encountered. This function handles s-shells "
                f"only; use ao_pair_fourier_transform for general L "
                f"(milestone 2).")

    # Precompute G^2/4 for damping factor reuse.
    G2 = (G_vectors ** 2).sum(axis=1)            # (n_G,)

    # Walk shell pairs. Each s-shell contributes 1 AO. The shell
    # ordering matches libint's basis-function ordering: AOs are
    # numbered in the order shells appear.
    n_shells = len(shells)
    bf_offsets = []
    off = 0
    for sh in shells:
        bf_offsets.append(off)
        off += 1                                   # s = 1 AO
    assert off == n_orb

    for sM in range(n_shells):
        shM = shells[sM]
        bfM = bf_offsets[sM]
        AM = np.asarray(shM.origin, dtype=float)
        es_M = np.asarray(shM.exponents, dtype=float)        # (nprim_M,)
        cs_M = np.asarray(shM.coefficients, dtype=float)     # (nprim_M,)

        for sN in range(n_shells):
            shN = shells[sN]
            bfN = bf_offsets[sN]
            BN = np.asarray(shN.origin, dtype=float)
            es_N = np.asarray(shN.exponents, dtype=float)    # (nprim_N,)
            cs_N = np.asarray(shN.coefficients, dtype=float) # (nprim_N,)

            ABsq = float(((AM - BN) ** 2).sum())

            # Gaussian-product setup: broadcast primitives.
            alpha_M = es_M[:, None]                          # (nprim_M, 1)
            alpha_N = es_N[None, :]                          # (1, nprim_N)
            gamma = alpha_M + alpha_N                        # (nprim_M, nprim_N)
            # K_pq prefactor (Gaussian-product overlap-contribution).
            K = (cs_M[:, None] * cs_N[None, :]               # (nprim_M, nprim_N)
                 * np.exp(-alpha_M * alpha_N / gamma * ABsq))

            # P_pq centres: (nprim_M, nprim_N, 3).
            P = (alpha_M[..., None] * AM[None, None, :]
                 + alpha_N[..., None] * BN[None, None, :]) / gamma[..., None]
            # Reshape for batched G.P: stack all (p, q) into one axis.
            n_pq = P.shape[0] * P.shape[1]
            P_flat = P.reshape(n_pq, 3)                      # (n_pq, 3)
            K_flat = K.reshape(n_pq)                         # (n_pq,)
            gamma_flat = gamma.reshape(n_pq)                 # (n_pq,)

            # FT contribution per primitive pair, batched over G:
            #   r̂_pq(G) = K_pq . (pi/g_pq)^{3/2} . exp(-G^2/4g_pq) . exp(-iG.P_pq)
            radial = (
                K_flat[:, None]                              # (n_pq, 1)
                * (np.pi / gamma_flat)[:, None] ** 1.5       # (n_pq, 1)
                * np.exp(-G2[None, :] / (4.0 * gamma_flat[:, None]))   # (n_pq, n_G)
            )                                                # (n_pq, n_G) real
            phase = np.exp(-1j * (P_flat @ G_vectors.T))     # (n_pq, n_G) complex

            # Sum over primitive pairs.
            ft_munu = np.einsum("pk,pk->k", radial, phase)   # (n_G,) complex

            out[bfM, bfN, :] = ft_munu
    return out


def md_e_coefficients_1d(
    la: int, lb: int,
    gamma: float, P_minus_A: float, P_minus_B: float,
) -> np.ndarray:
    """McMurchie-Davidson Hermite expansion coefficients (1-D).

    For two Cartesian Gaussians with angular indices (la, lb) on a
    single Cartesian axis, the product
    ``(x-A)^la . (x-B)^lb . exp(-g (x-P)^2)`` re-expresses as

        S_t E^{la,lb}_t . H_t(x-P; g) . exp(-g(x-P)^2)

    where ``H_t`` is the t-th Hermite polynomial / Hermite Gaussian
    derivative. The coefficients ``E^{i,j}_t`` are built recursively:

      E^{i+1, j}_t = (1/(2g)).E^{i,j}_{t-1} + (P-A).E^{i,j}_t
                   + (t+1).E^{i,j}_{t+1}
      E^{i, j+1}_t = (1/(2g)).E^{i,j}_{t-1} + (P-B).E^{i,j}_t
                   + (t+1).E^{i,j}_{t+1}

    with seed E^{0,0}_0 = 1, E^{0,0}_t = 0 for t > 0. (See Helgaker,
    Jorgensen & Olsen, *Molecular Electronic-Structure Theory*,
    eq. 9.5.6.)

    Note: the prefactor ``exp(-ab/g |A-B|^2)`` from the Gaussian
    product theorem is FACTORED OUT and supplied separately by the
    caller; this function only handles the polynomial-recursion part.

    Parameters
    ----------
    la, lb
        Angular momentum on this axis for the two Gaussians.
    gamma
        Combined exponent ``a + b``.
    P_minus_A, P_minus_B
        ``P - A`` and ``P - B`` along this axis, where
        ``P = (aA + bB)/g`` is the Gaussian-product centre.

    Returns
    -------
    E : np.ndarray, shape ``(la+1, lb+1, la+lb+1)``, real
        ``E[i, j, t]`` is the McMurchie-Davidson coefficient.
    """
    t_max = la + lb
    E = np.zeros((la + 1, lb + 1, t_max + 1))
    E[0, 0, 0] = 1.0

    # First, walk the i-axis: compute E[i, 0, t] for i = 1..la using
    # i-incrementing recursion on E[i-1, 0, t].
    inv2g = 1.0 / (2.0 * gamma)
    for i in range(la):
        for t in range(t_max + 1):
            term = P_minus_A * E[i, 0, t]
            if t > 0:
                term += inv2g * E[i, 0, t - 1]
            if t + 1 <= t_max:
                term += (t + 1) * E[i, 0, t + 1]
            E[i + 1, 0, t] = term

    # Then walk the j-axis from each (i, 0): compute E[i, j, t] for
    # j = 1..lb using j-incrementing recursion on E[i, j-1, t].
    for j in range(lb):
        for i in range(la + 1):
            for t in range(t_max + 1):
                term = P_minus_B * E[i, j, t]
                if t > 0:
                    term += inv2g * E[i, j, t - 1]
                if t + 1 <= t_max:
                    term += (t + 1) * E[i, j, t + 1]
                E[i, j + 1, t] = term

    return E


def cartesian_gaussian_product_ft_grad(
    A: np.ndarray, B: np.ndarray,
    cart_A: tuple[int, int, int], cart_B: tuple[int, int, int],
    alpha: float, beta: float,
    G_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Atomic-position gradient of :func:`cartesian_gaussian_product_ft`.

    Returns ``(gA, gB)``, each ``(3, n_G)`` complex, where
    ``gA[mu] = dFT/dA_mu`` and ``gB[mu] = dFT/dB_mu`` -- the derivatives of the
    Gaussian-product FT with respect to the *bra* centre ``A`` and the
    *ket* centre ``B``.

    Both derivatives follow the standard angular-momentum-shift relation
    for a Cartesian Gaussian differentiated w.r.t. its own centre. For one
    axis mu and the bra Gaussian ``(x-A_mu)^{i} e^{-a(x-A_mu)^2}``,

        d/dA_mu [(x-A_mu)^{i} e^{-a(x-A_mu)^2}]
              = -i . (x-A_mu)^{i-1} e^{-a(x-A_mu)^2}
                + 2a . (x-A_mu)^{i+1} e^{-a(x-A_mu)^2},

    i.e. lower the angular index by one (weight -i) plus raise it by one
    (weight +2a). Because the Fourier transform is linear it commutes with
    d/dA, so the same combination of *shifted-index* FTs gives dFT/dA --
    and it captures the **full** centre dependence (the polynomial origin,
    the product centre ``P``, the prefactor ``K`` and the phase) without
    differentiating the McMurchie-Davidson recursion by hand. (Helgaker,
    Jorgensen & Olsen, *Molecular Electronic-Structure Theory*, Sec.9.10, the
    Cartesian Gaussian centre-derivative recurrence.) The ket centre ``B``
    uses the identical relation with ``cart_B`` and ``2b``.
    """
    n_G = G_vectors.shape[0]
    gA = np.zeros((3, n_G), dtype=np.complex128)
    gB = np.zeros((3, n_G), dtype=np.complex128)

    for mu in range(3):
        # --- bra centre A ---
        i_mu = cart_A[mu]
        raised = list(cart_A)
        raised[mu] += 1
        term = 2.0 * alpha * cartesian_gaussian_product_ft(
            A, B, tuple(raised), cart_B, alpha, beta, G_vectors)
        if i_mu > 0:
            lowered = list(cart_A)
            lowered[mu] -= 1
            term = term - float(i_mu) * cartesian_gaussian_product_ft(
                A, B, tuple(lowered), cart_B, alpha, beta, G_vectors)
        gA[mu, :] = term

        # --- ket centre B ---
        j_mu = cart_B[mu]
        raised = list(cart_B)
        raised[mu] += 1
        term = 2.0 * beta * cartesian_gaussian_product_ft(
            A, B, cart_A, tuple(raised), alpha, beta, G_vectors)
        if j_mu > 0:
            lowered = list(cart_B)
            lowered[mu] -= 1
            term = term - float(j_mu) * cartesian_gaussian_product_ft(
                A, B, cart_A, tuple(lowered), alpha, beta, G_vectors)
        gB[mu, :] = term

    return gA, gB


def cartesian_gaussian_product_ft(
    A: np.ndarray, B: np.ndarray,
    cart_A: tuple[int, int, int], cart_B: tuple[int, int, int],
    alpha: float, beta: float,
    G_vectors: np.ndarray,
) -> np.ndarray:
    """FT of one Cartesian Gaussian product, via McMurchie-Davidson.

    Computes
    ::

        F[ (x-A_x)^{ix_A} (y-A_y)^{iy_A} (z-A_z)^{iz_A} e^{-a (r-A)^2}
         . (x-B_x)^{ix_B} (y-B_y)^{iy_B} (z-B_z)^{iz_B} e^{-b (r-B)^2} ](G)

    using the Gaussian-product theorem + Hermite expansion. The
    Cartesian indices encode the angular monomial ``x^i y^j z^k``
    with ``i + j + k = l`` for a Cartesian Gaussian of angular
    momentum l.

    Parameters
    ----------
    A, B : (3,) arrays
        Centres of the two Gaussians (bohr).
    cart_A, cart_B : (3,)-tuples of int
        Cartesian indices ``(ix, iy, iz)`` for each Gaussian. Sum =
        angular momentum.
    alpha, beta : float
        Primitive exponents.
    G_vectors : (n_G, 3) array
        Reciprocal-lattice vectors at which to evaluate the FT.

    Returns
    -------
    out : (n_G,) complex128
        FT of the single Gaussian product at each G.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    G_vectors = np.ascontiguousarray(G_vectors, dtype=float)
    AB = A - B
    AB_sq = float(np.dot(AB, AB))

    gamma = alpha + beta
    P = (alpha * A + beta * B) / gamma                                  # (3,)
    PA = P - A
    PB = P - B

    # 1-D Hermite expansions per axis.
    Ex = md_e_coefficients_1d(cart_A[0], cart_B[0], gamma, PA[0], PB[0])
    Ey = md_e_coefficients_1d(cart_A[1], cart_B[1], gamma, PA[1], PB[1])
    Ez = md_e_coefficients_1d(cart_A[2], cart_B[2], gamma, PA[2], PB[2])

    tmax = cart_A[0] + cart_B[0]
    umax = cart_A[1] + cart_B[1]
    vmax = cart_A[2] + cart_B[2]

    # E_total[t, u, v] = Ex[ia, ib, t] . Ey[ja, jb, u] . Ez[ka, kb, v]
    # picked at the requested (cart_A, cart_B). Fold the full
    # outer-product-and-pick into a single (tmax+1, umax+1, vmax+1)
    # tensor.
    E_total = np.einsum(
        "t,u,v->tuv",
        Ex[cart_A[0], cart_B[0], :tmax + 1],
        Ey[cart_A[1], cart_B[1], :umax + 1],
        Ez[cart_A[2], cart_B[2], :vmax + 1],
    )

    # Prefactor from Gaussian-product theorem.
    K = np.exp(-alpha * beta / gamma * AB_sq)

    # FT of single Gaussian x Hermite derivative chain:
    #   F[H_{tuv}(r-P; g)](G) = (-iG_x)^t (-iG_y)^u (-iG_z)^v
    #                          . (pi/g)^{3/2} . exp(-G^2/4g) . exp(-iG.P)
    # The (-iG)^t sign comes from (d/dP)^t . exp(-iG.P) = (-iG)^t exp(-iG.P).
    # (2026-05-18 fix: the historical code used (+iG)^t which is off by
    # (-1)^t for odd t. For SAME-center pairs only t=L_total contributes
    # and the L_total-odd term flips sign -- exactly what the per-AO
    # (-1)^L sign factor in _per_ao_pair_libint_to_libcint_scale was
    # patching. For OFF-center pairs both even and odd t contribute and
    # the per-AO blanket sign over-corrected the even-t terms, which
    # appeared as the LiH primitive multi-k AFT ~109 Ha residue tracked
    # at examples/debug/gdf_pair_ft_pyscf_diff.py.)
    G2 = (G_vectors ** 2).sum(axis=1)                                # (n_G,)
    Gx = G_vectors[:, 0]
    Gy = G_vectors[:, 1]
    Gz = G_vectors[:, 2]
    miGx = -1j * Gx
    miGy = -1j * Gy
    miGz = -1j * Gz

    # (-iG_x)^t over t=0..tmax; same for u, v.
    iGx_pow = np.array([miGx ** t for t in range(tmax + 1)])           # (tmax+1, n_G) complex
    iGy_pow = np.array([miGy ** u for u in range(umax + 1)])           # (umax+1, n_G)
    iGz_pow = np.array([miGz ** v for v in range(vmax + 1)])           # (vmax+1, n_G)

    # FT polynomial: S_{tuv} E_total[t,u,v] . iGx^t . iGy^u . iGz^v
    poly_ft = np.einsum(
        "tuv,tk,uk,vk->k",
        E_total, iGx_pow, iGy_pow, iGz_pow,
    )                                                                 # (n_G,) complex

    radial_ft = (np.pi / gamma) ** 1.5 * np.exp(-G2 / (4.0 * gamma))   # (n_G,) real
    phase = np.exp(-1j * (G_vectors @ P))                              # (n_G,) complex

    return K * poly_ft * radial_ft * phase


def cartesian_components_for_l(l: int) -> list[tuple[int, int, int]]:
    """Enumerate the ``(l+1)(l+2)/2`` Cartesian indices for shell of
    angular momentum l, in the order that maps to standard real
    spherical harmonics.

    Order: (i, j, k) with i + j + k = l, sorted by ``(-i, -j, -k)``
    (i.e. x-power decreases first; y next; z last). This is the
    libint Cartesian ordering when ``shell.pure = false`` is used,
    matching the McMurchie-Davidson literature convention.

    Examples
    --------
    >>> cartesian_components_for_l(0)
    [(0, 0, 0)]
    >>> cartesian_components_for_l(1)
    [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    >>> cartesian_components_for_l(2)
    [(2, 0, 0), (1, 1, 0), (1, 0, 1), (0, 2, 0), (0, 1, 1), (0, 0, 2)]
    """
    out: list[tuple[int, int, int]] = []
    for i in range(l, -1, -1):
        for j in range(l - i, -1, -1):
            k = l - i - j
            out.append((i, j, k))
    return out


def _evaluate_real_sph_harm_times_rl(
    l: int, m: int, x: np.ndarray, y: np.ndarray, z: np.ndarray,
) -> np.ndarray:
    """Evaluate ``r^l . Y_lm(r̂)`` at (x, y, z) points.

    Uses the same real-spherical-harmonic convention as
    :func:`vibeqc.aux_basis._real_sph_harm` (libint convention with
    Y_00 absorbed into the s-shell coefficient -- for s-shells we
    return ``1`` directly to match libint's "no Y_00" basis function
    treatment).
    """
    if l == 0:
        # libint convention: s-shell basis function is just c . e^{-ar^2}
        # (Y_00 absorbed into c). So r^0 . "spherical-l0" = 1.
        return np.ones_like(x)

    r = np.sqrt(x * x + y * y + z * z)
    # Avoid 0 division at r=0 (we evaluate away from the origin).
    safe_r = np.where(r > 0, r, 1.0)
    theta = np.arccos(np.clip(z / safe_r, -1.0, 1.0))
    phi = np.arctan2(y, x)
    from .aux_basis import _real_sph_harm
    Ylm = _real_sph_harm(l, m, theta, phi)
    return r ** l * Ylm


def cart_to_sph_matrix(l: int) -> np.ndarray:
    """Cartesian-to-spherical transform matrix for an angular shell.

    Computes the matrix ``C[m, a]`` such that

    ::

        r^l . Y_lm(r̂)  =  S_a  C[m, a] . x^{i_a} y^{j_a} z^{k_a}

    where ``(i_a, j_a, k_a)`` enumerates ``cartesian_components_for_l(l)``
    and ``Y_lm`` follows libint's real-spherical convention (cf.
    :func:`vibeqc.aux_basis._real_sph_harm`; for ``l = 0`` returns
    the trivial ``[[1]]`` matrix since libint absorbs Y_00 into the
    s-shell coefficient).

    Implementation: numerical fit. Sample ``r^l Y_lm`` and the
    Cartesian monomials at a Lebedev-like set of unit-sphere points,
    then solve the over-determined linear system via least squares.
    The numerical fit gives ``C`` to 1e-12 precision (verified;
    rank-revealing if l <= 6).

    Returns
    -------
    C : np.ndarray, shape ``(2l+1, (l+1)(l+2)/2)``, real
        ``C[m+l, a]`` is the coefficient for the a-th Cartesian
        monomial in the (l, m) spherical AO. Note: m index is
        offset to ``m+l`` (so 0 corresponds to ``m=-l``,
        ``2l`` corresponds to ``m=+l``).
    """
    if l == 0:
        return np.array([[1.0]])

    cart = cartesian_components_for_l(l)
    n_cart = len(cart)            # = (l+1)(l+2)/2
    n_sph = 2 * l + 1

    # Sample points on the unit sphere -- use a deterministic
    # quasi-random set so the fit is reproducible.
    n_samp = max(200, 4 * n_cart)
    rng = np.random.default_rng(seed=l * 997 + 17)
    # Uniform on sphere via z in [-1, 1], phi in [0, 2pi)
    z = rng.uniform(-1.0, 1.0, n_samp)
    phi_s = rng.uniform(0.0, 2.0 * np.pi, n_samp)
    sin_t = np.sqrt(1.0 - z * z)
    x = sin_t * np.cos(phi_s)
    y = sin_t * np.sin(phi_s)

    # Build the design matrix A[n, a] = x_n^{i_a} y_n^{j_a} z_n^{k_a}
    A = np.empty((n_samp, n_cart))
    for a, (i, j, k) in enumerate(cart):
        A[:, a] = (x ** i) * (y ** j) * (z ** k)

    # Build the RHS B[n, m+l] = Y_lm(r̂_n) at unit r (so r^l Y_lm = Y_lm).
    B = np.empty((n_samp, n_sph))
    for m in range(-l, l + 1):
        B[:, m + l] = _evaluate_real_sph_harm_times_rl(l, m, x, y, z)

    # Solve A . C^T = B (each column of B fits a linear combo of
    # Cartesian monomials).  Result shape: C^T = (n_cart, n_sph),
    # C = (n_sph, n_cart).
    Csol, _residuals, rank, _sv = np.linalg.lstsq(A, B, rcond=None)
    if rank < n_cart:
        raise RuntimeError(
            f"cart_to_sph_matrix(l={l}): design matrix has rank "
            f"{rank} < n_cart={n_cart}; sample-point set may be "
            f"degenerate. Bump n_samp.")
    C = Csol.T   # (n_sph, n_cart)
    return C


def _shell_pair_ao_pair_ft_at_origins(
    lM: int, lN: int,
    AM: np.ndarray, BN: np.ndarray,
    es_M: np.ndarray, cs_M: np.ndarray,
    es_N: np.ndarray, cs_N: np.ndarray,
    G_vectors: np.ndarray,
) -> np.ndarray:
    """Core single-shell-pair FT -- accepts explicit origins.

    Returns ``(2lM+1, 2lN+1, n_G)`` complex array. Same algorithm as
    :func:`_shell_pair_ao_pair_ft` but with the second shell's origin
    overridden -- used by the shifted-ν / lattice variants where the
    ket Gaussian is translated by an arbitrary R_g.
    """
    cart_M = cartesian_components_for_l(lM)
    cart_N = cartesian_components_for_l(lN)
    n_cart_M = len(cart_M)
    n_cart_N = len(cart_N)
    n_G = G_vectors.shape[0]

    ft_cart = np.zeros((n_cart_M, n_cart_N, n_G), dtype=np.complex128)
    for ip, (alpha, c_a) in enumerate(zip(es_M, cs_M)):
        for iq, (beta, c_b) in enumerate(zip(es_N, cs_N)):
            cc = c_a * c_b
            for a, ca in enumerate(cart_M):
                for b, cb in enumerate(cart_N):
                    ft_pq = cartesian_gaussian_product_ft(
                        AM, BN, ca, cb, alpha, beta, G_vectors)
                    ft_cart[a, b, :] += cc * ft_pq

    Csph_M = cart_to_sph_matrix(lM)
    Csph_N = cart_to_sph_matrix(lN)
    ft_sph = np.einsum("ma,nb,abk->mnk", Csph_M, Csph_N, ft_cart)
    return ft_sph


def _shell_pair_ao_pair_ft_grad_at_origins(
    lM: int, lN: int,
    AM: np.ndarray, BN: np.ndarray,
    es_M: np.ndarray, cs_M: np.ndarray,
    es_N: np.ndarray, cs_N: np.ndarray,
    G_vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Origin-gradient of :func:`_shell_pair_ao_pair_ft_at_origins`.

    Returns ``(gradA, gradB)``, each ``(3, 2lM+1, 2lN+1, n_G)`` complex:
    ``gradA[mu]`` is d(shell-pair FT)/dAM_mu (the bra origin) and
    ``gradB[mu]`` d/dBN_mu (the ket origin). Same contraction (over
    primitive pairs + Cartesian->spherical) as the value routine, with
    each primitive FT replaced by its centre-gradient
    (:func:`cartesian_gaussian_product_ft_grad`).
    """
    cart_M = cartesian_components_for_l(lM)
    cart_N = cartesian_components_for_l(lN)
    n_cart_M = len(cart_M)
    n_cart_N = len(cart_N)
    n_G = G_vectors.shape[0]

    gA_cart = np.zeros((3, n_cart_M, n_cart_N, n_G), dtype=np.complex128)
    gB_cart = np.zeros((3, n_cart_M, n_cart_N, n_G), dtype=np.complex128)
    for alpha, c_a in zip(es_M, cs_M):
        for beta, c_b in zip(es_N, cs_N):
            cc = c_a * c_b
            for a, ca in enumerate(cart_M):
                for b, cb in enumerate(cart_N):
                    gA_pq, gB_pq = cartesian_gaussian_product_ft_grad(
                        AM, BN, ca, cb, alpha, beta, G_vectors)
                    gA_cart[:, a, b, :] += cc * gA_pq
                    gB_cart[:, a, b, :] += cc * gB_pq

    Csph_M = cart_to_sph_matrix(lM)
    Csph_N = cart_to_sph_matrix(lN)
    gA_sph = np.einsum("ma,nb,xabk->xmnk", Csph_M, Csph_N, gA_cart)
    gB_sph = np.einsum("ma,nb,xabk->xmnk", Csph_M, Csph_N, gB_cart)
    return gA_sph, gB_sph


def _shell_pair_ao_pair_ft(
    shM, shN, G_vectors: np.ndarray,
) -> np.ndarray:
    """FT of all AO-pair densities for one shell pair (sM, sN).

    Returns ``(2lM+1, 2lN+1, n_G)`` complex array, with the AO
    indices in libint's spherical ordering (m runs from -l to +l).

    Algorithm:

    1. For each (Cart_M, Cart_N) Cartesian-component pair, compute
       the FT of the Cartesian-Gaussian-product density via
       :func:`cartesian_gaussian_product_ft`, summed over all
       contracted primitive pairs.
    2. Apply the Cartesian-to-spherical transform to both axes via
       :func:`cart_to_sph_matrix`.

    The contraction-coefficient sum and primitive-pair loop happen
    INSIDE the per-Cartesian-pair loop so the MD recursion is run
    once per primitive-pair (not per spherical-AO-pair x primitive-
    pair, which would be wasteful).
    """
    return _shell_pair_ao_pair_ft_at_origins(
        int(shM.l), int(shN.l),
        np.asarray(shM.origin, dtype=float),
        np.asarray(shN.origin, dtype=float),
        np.asarray(shM.exponents, dtype=float),
        np.asarray(shM.coefficients, dtype=float),
        np.asarray(shN.exponents, dtype=float),
        np.asarray(shN.coefficients, dtype=float),
        G_vectors,
    )


def _ss_only_ao_pair_ft_at_BN(
    ao_basis: BasisSet,
    G_vectors: np.ndarray,
    BN_offsets: np.ndarray,
) -> np.ndarray:
    """Fast path: s-only basis, ν origin overridden per-shell.

    Same as :func:`ao_pair_fourier_transform_ss_only` but the second
    Gaussian centre BN is replaced with ``shN.origin + BN_offsets``,
    independently of the basis's own shell centres. The offset is a
    single global R_g shift applied to every ν AO (the bra mu stays
    at home).

    Returns ``(n_orb, n_orb, n_G)`` complex.
    """
    G_vectors = np.ascontiguousarray(G_vectors, dtype=float)
    n_G = G_vectors.shape[0]
    n_orb = ao_basis.nbasis
    out = np.zeros((n_orb, n_orb, n_G), dtype=np.complex128)

    shells = ao_basis.shells()
    for sh in shells:
        if int(sh.l) != 0:
            raise NotImplementedError(
                "_ss_only_ao_pair_ft_at_BN: s-only fast path only")

    R_g = np.asarray(BN_offsets, dtype=float).reshape(3)
    G2 = (G_vectors ** 2).sum(axis=1)

    bf_offsets = []
    off = 0
    for sh in shells:
        bf_offsets.append(off)
        off += 1

    n_shells = len(shells)
    for sM in range(n_shells):
        shM = shells[sM]
        bfM = bf_offsets[sM]
        AM = np.asarray(shM.origin, dtype=float)
        es_M = np.asarray(shM.exponents, dtype=float)
        cs_M = np.asarray(shM.coefficients, dtype=float)

        for sN in range(n_shells):
            shN = shells[sN]
            bfN = bf_offsets[sN]
            BN = np.asarray(shN.origin, dtype=float) + R_g
            es_N = np.asarray(shN.exponents, dtype=float)
            cs_N = np.asarray(shN.coefficients, dtype=float)

            ABsq = float(((AM - BN) ** 2).sum())

            alpha_M = es_M[:, None]
            alpha_N = es_N[None, :]
            gamma = alpha_M + alpha_N
            K = (cs_M[:, None] * cs_N[None, :]
                 * np.exp(-alpha_M * alpha_N / gamma * ABsq))
            P = (alpha_M[..., None] * AM[None, None, :]
                 + alpha_N[..., None] * BN[None, None, :]) / gamma[..., None]

            n_pq = P.shape[0] * P.shape[1]
            P_flat = P.reshape(n_pq, 3)
            K_flat = K.reshape(n_pq)
            gamma_flat = gamma.reshape(n_pq)

            radial = (K_flat[:, None]
                      * (np.pi / gamma_flat)[:, None] ** 1.5
                      * np.exp(-G2[None, :] / (4.0 * gamma_flat[:, None])))
            phase = np.exp(-1j * (P_flat @ G_vectors.T))

            ft_munu = np.einsum("pk,pk->k", radial, phase)
            out[bfM, bfN, :] = ft_munu
    return out


def ao_pair_fourier_transform_shifted_ket(
    ao_basis: BasisSet,
    G_vectors: np.ndarray,
    R_g: np.ndarray,
) -> np.ndarray:
    """FT of AO pair densities chi_mu(r).chi_ν(r-R_g) -- ν shifted by R_g.

    For Gaussian primitives, shifting ν by R_g changes the Gaussian-
    product centre **non-linearly**::

        P_g = (a_mu A_mu + a_ν (A_ν + R_g)) / g
            = P_0 + (a_ν / g) . R_g

    and the overlap prefactor::

        K_pq^g = c_mu c_ν . exp(-a_mu a_ν / g . |A_mu - A_ν - R_g|^2)

    so ``FT_muν(K; R_g) != FT_muν(K; 0) . exp(-iK.R_g)`` (the "bra-pair-
    at-home phase approximation" that
    :func:`vibeqc.bipole_ext_el_pole.compute_cell_density_fourier`
    used before this routine landed).

    Output is in libint's spherical AO ordering. For ``R_g = 0`` this
    reduces to :func:`ao_pair_fourier_transform`.
    """
    R_g_arr = np.asarray(R_g, dtype=float).reshape(3)
    shells = ao_basis.shells()
    if all(int(sh.l) == 0 for sh in shells):
        return _ss_only_ao_pair_ft_at_BN(ao_basis, G_vectors, R_g_arr)

    G_vectors = np.ascontiguousarray(G_vectors, dtype=float)
    n_G = G_vectors.shape[0]
    n_orb = ao_basis.nbasis
    out = np.zeros((n_orb, n_orb, n_G), dtype=np.complex128)

    bf_offsets: list[int] = []
    off = 0
    for sh in shells:
        bf_offsets.append(off)
        if not bool(sh.pure):
            raise NotImplementedError(
                "ao_pair_fourier_transform_shifted_ket: Cartesian "
                "(non-pure) shells with L > 0 not supported.")
        off += 2 * int(sh.l) + 1
    assert off == n_orb

    n_shells = len(shells)
    for sM in range(n_shells):
        bfM = bf_offsets[sM]
        szM = 2 * int(shells[sM].l) + 1
        AM = np.asarray(shells[sM].origin, dtype=float)
        es_M = np.asarray(shells[sM].exponents, dtype=float)
        cs_M = np.asarray(shells[sM].coefficients, dtype=float)
        lM = int(shells[sM].l)
        for sN in range(n_shells):
            bfN = bf_offsets[sN]
            szN = 2 * int(shells[sN].l) + 1
            BN_eff = np.asarray(shells[sN].origin, dtype=float) + R_g_arr
            es_N = np.asarray(shells[sN].exponents, dtype=float)
            cs_N = np.asarray(shells[sN].coefficients, dtype=float)
            lN = int(shells[sN].l)
            block = _shell_pair_ao_pair_ft_at_origins(
                lM, lN, AM, BN_eff, es_M, cs_M, es_N, cs_N, G_vectors,
            )
            out[bfM:bfM + szM, bfN:bfN + szN, :] = block
    return out


def ao_pair_fourier_transform_shifted_ket_grad(
    ao_basis: BasisSet,
    G_vectors: np.ndarray,
    R_g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Atomic-origin gradient of :func:`ao_pair_fourier_transform_shifted_ket`.

    Returns ``(grad_bra, grad_ket)``, each ``(n_orb, n_orb, 3, n_G)``
    complex:

    * ``grad_bra[mu, ν, axis, k] = dFT_muν(G_k; R_g)/dA_mu^axis`` -- derivative
      w.r.t. the **bra** AO's centre (which sits at its home atom).
    * ``grad_ket[mu, ν, axis, k] = dFT_muν(G_k; R_g)/d(A_ν + R_g)^axis`` --
      derivative w.r.t. the **ket** AO's (shifted) centre. Since every
      lattice image of an atom moves rigidly with it, this is also
      dFT/dA_ν^home (d(A_ν+R_g)/dA_ν = I), so the caller scatters it onto
      the ket AO's home atom.

    The caller resolves "which atom" via the bra/ket AO->atom map; this
    routine only needs the centre derivatives, which it gets per shell
    pair from :func:`_shell_pair_ao_pair_ft_grad_at_origins`.
    """
    R_g_arr = np.asarray(R_g, dtype=float).reshape(3)
    shells = ao_basis.shells()
    G_vectors = np.ascontiguousarray(G_vectors, dtype=float)
    n_G = G_vectors.shape[0]
    n_orb = ao_basis.nbasis
    grad_bra = np.zeros((n_orb, n_orb, 3, n_G), dtype=np.complex128)
    grad_ket = np.zeros((n_orb, n_orb, 3, n_G), dtype=np.complex128)

    bf_offsets: list[int] = []
    off = 0
    for sh in shells:
        bf_offsets.append(off)
        if int(sh.l) > 0 and not bool(sh.pure):
            raise NotImplementedError(
                "ao_pair_fourier_transform_shifted_ket_grad: Cartesian "
                "(non-pure) shells with L > 0 not supported.")
        off += 2 * int(sh.l) + 1
    assert off == n_orb

    n_shells = len(shells)
    for sM in range(n_shells):
        bfM = bf_offsets[sM]
        szM = 2 * int(shells[sM].l) + 1
        AM = np.asarray(shells[sM].origin, dtype=float)
        es_M = np.asarray(shells[sM].exponents, dtype=float)
        cs_M = np.asarray(shells[sM].coefficients, dtype=float)
        lM = int(shells[sM].l)
        for sN in range(n_shells):
            bfN = bf_offsets[sN]
            szN = 2 * int(shells[sN].l) + 1
            BN_eff = np.asarray(shells[sN].origin, dtype=float) + R_g_arr
            es_N = np.asarray(shells[sN].exponents, dtype=float)
            cs_N = np.asarray(shells[sN].coefficients, dtype=float)
            lN = int(shells[sN].l)
            gA, gB = _shell_pair_ao_pair_ft_grad_at_origins(
                lM, lN, AM, BN_eff, es_M, cs_M, es_N, cs_N, G_vectors,
            )
            # gA/gB are (3, szM, szN, n_G); store as (szM, szN, 3, n_G).
            grad_bra[bfM:bfM + szM, bfN:bfN + szN, :, :] = \
                np.transpose(gA, (1, 2, 0, 3))
            grad_ket[bfM:bfM + szM, bfN:bfN + szN, :, :] = \
                np.transpose(gB, (1, 2, 0, 3))
    return grad_bra, grad_ket


def ao_pair_fourier_transform_grad_at_cells(
    ao_basis: BasisSet,
    G_vectors: np.ndarray,
    R_g_list: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell atomic-origin gradient of the AO-pair FT.

    Stacks :func:`ao_pair_fourier_transform_shifted_ket_grad` over the
    lattice translation vectors ``R_g_list``. Returns ``(grad_bra,
    grad_ket)``, each ``(n_g, n_orb, n_orb, 3, n_G)`` complex -- the
    gradient counterpart of :func:`ao_pair_fourier_transform_at_cells`.
    """
    R_g_arr = np.ascontiguousarray(R_g_list, dtype=float)
    if R_g_arr.ndim != 2 or R_g_arr.shape[1] != 3:
        raise ValueError(
            f"R_g_list must be (n_g, 3); got shape {R_g_arr.shape}")
    n_g = R_g_arr.shape[0]
    n_orb = ao_basis.nbasis
    G_vectors = np.ascontiguousarray(G_vectors, dtype=float)
    n_G = G_vectors.shape[0]
    grad_bra = np.empty((n_g, n_orb, n_orb, 3, n_G), dtype=np.complex128)
    grad_ket = np.empty((n_g, n_orb, n_orb, 3, n_G), dtype=np.complex128)
    for ig in range(n_g):
        gb, gk = ao_pair_fourier_transform_shifted_ket_grad(
            ao_basis, G_vectors, R_g_arr[ig])
        grad_bra[ig] = gb
        grad_ket[ig] = gk
    return grad_bra, grad_ket


def ao_pair_fourier_transform_at_cells(
    ao_basis: BasisSet,
    G_vectors: np.ndarray,
    R_g_list: np.ndarray,
) -> np.ndarray:
    """Per-cell AO-pair FT -- returns ``FT_muν(K; R_g)`` for every g.

    Computes the shifted-ν FT once per lattice translation vector
    ``R_g`` in ``R_g_list`` and stacks them. This is the workhorse
    builder for both r̂(K) at a multi-k density (sum over g with
    weights D(g)) and the per-k Bloch-summed bra-pair FT (sum over
    g with phases ``exp(+ik.R_g)``).

    Parameters
    ----------
    ao_basis
        AO basis (s-only fast path active for pure-s bases).
    G_vectors : (n_G, 3) float
        Reciprocal-space sampling points.
    R_g_list : (n_g, 3) float
        Lattice translation vectors for the ket Gaussian.

    Returns
    -------
    out : (n_g, n_orb, n_orb, n_G) complex128
        ``out[g, mu, ν, k]`` = ``FT_muν(G_vectors[k]; R_g_list[g])``.
    """
    R_g_arr = np.ascontiguousarray(R_g_list, dtype=float)
    if R_g_arr.ndim != 2 or R_g_arr.shape[1] != 3:
        raise ValueError(
            f"R_g_list must be (n_g, 3); got shape {R_g_arr.shape}")
    n_g = R_g_arr.shape[0]
    n_orb = ao_basis.nbasis
    G_vectors = np.ascontiguousarray(G_vectors, dtype=float)
    n_G = G_vectors.shape[0]
    out = np.empty((n_g, n_orb, n_orb, n_G), dtype=np.complex128)
    for ig in range(n_g):
        out[ig] = ao_pair_fourier_transform_shifted_ket(
            ao_basis, G_vectors, R_g_arr[ig])
    return out


def ao_pair_fourier_transform_bloch(
    ao_basis: BasisSet,
    G_vectors: np.ndarray,
    R_g_list: np.ndarray,
    k_cart: np.ndarray,
    *,
    ft_per_cell: Optional[np.ndarray] = None,
    screen_tol: float = 0.0,
) -> np.ndarray:
    """Bloch-summed AO-pair FT::

        FT^{(+k)}_muν(K) = S_g exp(+ik.R_g) . FT_muν(K; R_g)

    This is the Fourier coefficient of the Bloch AO-pair density at
    crystal momentum ``k`` and reciprocal point ``K``. The ``+ik``
    sign matches vibe-qc's standard Bloch-sum convention
    (``F(k) = S_g exp(+ik.R_g) F(g)``); see
    :func:`vibeqc.pbc_bipole._bloch_sum_blocks`.

    Parameters
    ----------
    ao_basis
        AO basis.
    G_vectors : (n_G, 3) float
    R_g_list : (n_g, 3) float
        Lattice-cell coordinates (e.g. ``cells[c].r_cart`` from a
        ``LatticeMatrixSet``).
    k_cart : (3,) float
        Crystal momentum in Cartesian inverse bohr.
    ft_per_cell : (n_g, n_orb, n_orb, n_G) complex, optional
        Pre-computed per-cell FT from
        :func:`ao_pair_fourier_transform_at_cells` -- pass this when
        the same set of (basis, G_vectors, R_g_list) is reused for
        multiple k-points to avoid re-doing the MD recursion.
    screen_tol
        Optional high-|G| shell/cell screen for the native general-L
        backend. ``0.0`` (default) is exact; positive values skip only
        shell/cell blocks whose conservative primitive-pair Fourier
        bound is below the tolerance. Used by the RSGDF tail completion
        where diffuse inter-cell AO-pair contributions are exponentially
        dead.

    Returns
    -------
    out : (n_orb, n_orb, n_G) complex128
    """
    R_g_arr = np.ascontiguousarray(R_g_list, dtype=float)
    if R_g_arr.ndim != 2 or R_g_arr.shape[1] != 3:
        raise ValueError(
            f"R_g_list must be (n_g, 3); got shape {R_g_arr.shape}")
    k = np.asarray(k_cart, dtype=float).reshape(3)

    # Dispatch to the C++ kernel when applicable. The native path runs
    # the Bloch sum streaming over R_g, never materialising the
    # (n_g, n_orb, n_orb, n_G) per-cell intermediate that the pure-
    # Python ``ao_pair_fourier_transform_at_cells`` builds -- that
    # tensor is what OOM-kills MgO 8-atom rocksalt under the
    # Python path (see ``handovers/HANDOVER_GDF_V0_11_2026_05_29.md``).
    #
    # Two C++ entry points:
    #   * ``ao_pair_fourier_transform_bloch_ss`` -- closed-form, no
    #     Hermite recursion, used when every shell is L = 0.
    #   * ``ao_pair_fourier_transform_bloch_cxx`` -- general L via
    #     McMurchie-Davidson chain + cart->sph; the static table at
    #     ``cpp/include/vibeqc/cart_to_sph_data.hpp`` covers L <= 6.
    G_vectors_arr = np.ascontiguousarray(G_vectors, dtype=float)
    backend = _aopair_ft_backend()
    screen = float(screen_tol)
    if backend != "python" and ft_per_cell is None:
        shells = ao_basis.shells()
        all_s = all(int(sh.l) == 0 for sh in shells)
        if all_s and screen <= 0.0:
            from ._vibeqc_core import ao_pair_fourier_transform_bloch_ss
            return ao_pair_fourier_transform_bloch_ss(
                ao_basis, G_vectors_arr, R_g_arr, k)
        max_l = max(int(sh.l) for sh in shells)
        all_pure = all(bool(sh.pure) or int(sh.l) == 0 for sh in shells)
        if all_pure and max_l <= 6:
            from ._vibeqc_core import ao_pair_fourier_transform_bloch_cxx
            return ao_pair_fourier_transform_bloch_cxx(
                ao_basis, G_vectors_arr, R_g_arr, k, screen)
        if backend == "cxx":
            reason = []
            if not all_pure:
                reason.append("a Cartesian (non-pure) shell with L > 0")
            if max_l > 6:
                reason.append(
                    f"max L = {max_l} exceeds the C++ cart-to-sph "
                    f"table coverage (L <= 6)"
                )
            raise NotImplementedError(
                f"VIBEQC_AOPAIR_FT_BACKEND=cxx requested but basis has "
                f"{' and '.join(reason)}; the C++ general-L kernel only "
                f"handles pure-spherical L <= 6 (see "
                f"docs/design_rsgdf_3c_lr.md and "
                f"cpp/include/vibeqc/cart_to_sph_data.hpp)."
            )
        # Fall through to the Python path for unsupported cases.

    if ft_per_cell is None:
        ft_per_cell = ao_pair_fourier_transform_at_cells(
            ao_basis, G_vectors_arr, R_g_arr)
    phases = np.exp(1j * (R_g_arr @ k))            # (n_g,) complex
    # ft_per_cell shape: (n_g, n_orb, n_orb, n_G)
    return np.einsum("g,gmnk->mnk", phases, ft_per_cell)


def ao_pair_fourier_transform_bloch_multi(
    ao_basis: BasisSet,
    G_vectors: np.ndarray,
    R_g_list: np.ndarray,
    k_carts: np.ndarray,
    *,
    screen_tol: float = 0.0,
    pair_weights: Optional[np.ndarray] = None,
) -> list[np.ndarray]:
    """Batched Bloch-summed AO-pair FT for several crystal momenta.

    Evaluates :func:`ao_pair_fourier_transform_bloch` for every row of
    ``k_carts`` on ONE shared reciprocal support. The per-cell
    McMurchie-Davidson pair-FT work is k-independent — the ket momentum
    enters only through the Bloch phase ``exp(+i k·R)`` — so the batch
    costs one pair-FT pass plus one cheap phase fold per k instead of
    ``n_k`` full passes. This is the 3c-side companion of the RSGDF
    q-metric cache: the multi-k GDF drivers batch the ``(k_bra, k_ket)``
    pairs sharing one momentum transfer q through this entry point.

    Numerical note: the batched C++ kernel applies the phase fold after
    the cart→sph transform instead of fusing it, so results agree with
    per-k :func:`ao_pair_fourier_transform_bloch` calls to floating-point
    rounding (~1e-15 relative), not bit-for-bit.

    Parameters
    ----------
    ao_basis, G_vectors, R_g_list, screen_tol
        As for :func:`ao_pair_fourier_transform_bloch`.
    k_carts : (n_k, 3) float
        Crystal momenta in Cartesian inverse bohr.
    pair_weights : (n_orb, n_orb) float, optional
        Per-AO-pair real weight applied as the kernel stores its
        result. Bit-identical to scaling the returned tensors with
        ``tensor *= pair_weights[:, :, None]`` afterwards, except that
        an EXACTLY ZERO weight stores a true ``+0.0`` — matching a
        NumPy masked assignment (``tensor[~keep] = 0.0``), which is
        what a fit screen does, rather than a multiply by zero.

        Every consumer of this tensor scales it this way before
        contracting, and that scaling is a single-threaded NumPy pass
        over the largest array in the RSGDF cderi build — measurably
        anti-scaling against the threads this kernel itself uses
        (``handovers/HANDOVER_GDF_OUTSTANDING.md``, OpenMP profile
        2026-08-05). Passing it here costs nothing: the store already
        touches every element, inside the parallel region.

    Returns
    -------
    list of (n_orb, n_orb, n_G) complex128, one per row of ``k_carts``.
    """
    R_g_arr = np.ascontiguousarray(R_g_list, dtype=float)
    if R_g_arr.ndim != 2 or R_g_arr.shape[1] != 3:
        raise ValueError(
            f"R_g_list must be (n_g, 3); got shape {R_g_arr.shape}")
    k_arr = np.ascontiguousarray(k_carts, dtype=float)
    if k_arr.ndim != 2 or k_arr.shape[1] != 3:
        raise ValueError(
            f"k_carts must be (n_k, 3); got shape {k_arr.shape}")
    if k_arr.shape[0] == 0:
        return []

    G_vectors_arr = np.ascontiguousarray(G_vectors, dtype=float)
    weights_arr: Optional[np.ndarray] = None
    if pair_weights is not None:
        weights_arr = np.ascontiguousarray(pair_weights, dtype=float)
        n_orb = int(ao_basis.nbasis)
        if weights_arr.shape != (n_orb, n_orb):
            raise ValueError(
                f"pair_weights must be ({n_orb}, {n_orb}); got shape "
                f"{weights_arr.shape}")
    backend = _aopair_ft_backend()
    screen = float(screen_tol)
    if backend != "python":
        shells = ao_basis.shells()
        max_l = max(int(sh.l) for sh in shells)
        all_pure = all(bool(sh.pure) or int(sh.l) == 0 for sh in shells)
        if all_pure and max_l <= 6:
            # The general-L batched kernel also covers all-s bases (the
            # MD chain reduces to the closed form at L = 0); sharing the
            # per-cell pass across k beats the per-k closed-form path
            # for any n_k > 1.
            from ._vibeqc_core import ao_pair_fourier_transform_bloch_multi_cxx
            return list(ao_pair_fourier_transform_bloch_multi_cxx(
                ao_basis, G_vectors_arr, R_g_arr, k_arr, screen,
                weights_arr))
        if backend == "cxx":
            reason = []
            if not all_pure:
                reason.append("a Cartesian (non-pure) shell with L > 0")
            if max_l > 6:
                reason.append(
                    f"max L = {max_l} exceeds the C++ cart-to-sph "
                    f"table coverage (L <= 6)"
                )
            raise NotImplementedError(
                f"VIBEQC_AOPAIR_FT_BACKEND=cxx requested but basis has "
                f"{' and '.join(reason)}; the C++ general-L kernel only "
                f"handles pure-spherical L <= 6 (see "
                f"docs/design_rsgdf_3c_lr.md and "
                f"cpp/include/vibeqc/cart_to_sph_data.hpp)."
            )

    # Pure-Python fallback: materialise the per-cell FT once and fold
    # every k against it — the same work-sharing, at the memory cost of
    # the (n_g, n_orb, n_orb, n_G) intermediate the reference path has
    # always carried.
    ft_per_cell = ao_pair_fourier_transform_at_cells(
        ao_basis, G_vectors_arr, R_g_arr)
    masked = None if weights_arr is None else (weights_arr == 0.0)
    out: list[np.ndarray] = []
    for ik in range(k_arr.shape[0]):
        phases = np.exp(1j * (R_g_arr @ k_arr[ik]))
        tensor = np.einsum("g,gmnk->mnk", phases, ft_per_cell)
        if weights_arr is not None:
            # Same two operations the C++ store fuses, in the same
            # order, so the reference path carries the same contract.
            tensor *= weights_arr[:, :, None]
            if masked is not None and masked.any():
                tensor[masked] = 0.0
        out.append(tensor)
    return out


def ao_pair_fourier_transform_bloch_gradient_gweighted(
    ao_basis: BasisSet,
    G_vectors: np.ndarray,
    R_g_list: np.ndarray,
    k_cart: np.ndarray,
    pair_weights_g: np.ndarray,
    n_atoms: int,
) -> np.ndarray:
    """Bloch-phased G-resolved-weighted AO-pair FT centre gradient.

    Computes (handovers/HANDOVER_GDF_GRADIENT_DEFERRED.md § 4, rung 1)::

        grad[A, d] = d/dR_{A,d} Re S_G S_muν Q_muν(G)
                         conj( S_g exp(+ik.R_g) FT_muν(G; R_g) )

    where the conjugated Bloch sum is exactly what
    :func:`ao_pair_fourier_transform_bloch` evaluates. The Bloch phase
    is atom-position independent; the derivative acts on the
    Gaussian-product centres only. ``k_cart = 0`` reduces to the C++
    ``ao_pair_fourier_transform_gamma_gradient_gweighted`` kernel
    bit-for-bit. C++-only (no pure-Python fallback), matching the
    Gamma gradient kernels.

    Parameters
    ----------
    ao_basis
        AO basis (pure-spherical shells, L <= 6).
    G_vectors : (n_G, 3) float
    R_g_list : (n_g, 3) float
    k_cart : (3,) float
        Crystal momentum in Cartesian inverse bohr; ``+ik`` ket sign
        as in :func:`ao_pair_fourier_transform_bloch`.
    pair_weights_g : (n_orb, n_orb, n_G) complex
        G-resolved complex pair weights Q, C-contiguous.
    n_atoms
        Number of atoms to scatter into; may exceed the atoms
        represented in ``ao_basis``.

    Returns
    -------
    out : (n_atoms, 3) float64
    """
    from ._vibeqc_core import (
        ao_pair_fourier_transform_bloch_gradient_gweighted as _cxx_kernel,
    )

    G_arr = np.ascontiguousarray(G_vectors, dtype=float)
    R_g_arr = np.ascontiguousarray(R_g_list, dtype=float)
    if R_g_arr.ndim != 2 or R_g_arr.shape[1] != 3:
        raise ValueError(
            f"R_g_list must be (n_g, 3); got shape {R_g_arr.shape}")
    k = np.ascontiguousarray(k_cart, dtype=float).reshape(3)
    Q = np.ascontiguousarray(pair_weights_g, dtype=np.complex128)
    return np.asarray(
        _cxx_kernel(ao_basis, G_arr, R_g_arr, k, Q, int(n_atoms)),
        dtype=np.float64,
    )


def ao_pair_fourier_transform(
    ao_basis: BasisSet,
    G_vectors: np.ndarray,
) -> np.ndarray:
    """FT of all AO-pair densities chi_mu(r) chi_ν(r) at given G-vectors.

    General-L implementation via McMurchie-Davidson Hermite expansion
    of the Cartesian Gaussian product, followed by Cartesian-to-
    spherical transform per shell. For pure-s bases dispatches to the
    closed-form fast path :func:`ao_pair_fourier_transform_ss_only`.

    Output is in libint's spherical AO ordering (within-shell m runs
    from -l to +l). For l=1 this gives (py, pz, px) per the libint
    convention.

    Returns
    -------
    out : (n_orb, n_orb, n_G) complex128
        Symmetric in (mu, ν) on output (analytic property of the AO
        pair density).
    """
    shells = ao_basis.shells()
    if all(int(sh.l) == 0 for sh in shells):
        return ao_pair_fourier_transform_ss_only(ao_basis, G_vectors)

    G_vectors = np.ascontiguousarray(G_vectors, dtype=float)
    n_G = G_vectors.shape[0]
    n_orb = ao_basis.nbasis
    out = np.zeros((n_orb, n_orb, n_G), dtype=np.complex128)

    # Shell offsets in the AO basis.
    bf_offsets: list[int] = []
    off = 0
    for sh in shells:
        bf_offsets.append(off)
        if not bool(sh.pure):
            raise NotImplementedError(
                "ao_pair_fourier_transform: Cartesian (non-pure) shells "
                "with L > 0 not supported. vibe-qc forces pure spherical "
                "for L >= 2 by default; this should not arise in practice.")
        off += 2 * int(sh.l) + 1
    assert off == n_orb

    n_shells = len(shells)
    for sM in range(n_shells):
        bfM = bf_offsets[sM]
        szM = 2 * int(shells[sM].l) + 1
        for sN in range(n_shells):
            bfN = bf_offsets[sN]
            szN = 2 * int(shells[sN].l) + 1
            block = _shell_pair_ao_pair_ft(shells[sM], shells[sN], G_vectors)
            out[bfM:bfM + szM, bfN:bfN + szN, :] = block
    return out
