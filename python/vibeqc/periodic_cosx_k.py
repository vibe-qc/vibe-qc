"""Multi-k periodic COSX exchange bridge (M3b-3).

Connects the real-space COSX K(g) block engine
(``compute_cosx_k_blocks``, M3b-2) to k-space SCF loops:

    D(k) per k-point
      -> D(g) = (1/N_k) S_k e^{-i k.g} D(k)      (inverse Bloch fold;
                                                  real for TRS meshes)
      -> K(g) via the cell-pair COSX engine        (per-d analytic
                                                  A-blocks on a
                                                  periodic Becke grid)
      -> K(k) = S_g e^{i k.g} K(g)                 (``bloch_sum``)

The K(g) build cost is independent of the k-mesh size -- the mesh only
enters through the two folds. This is the structural advantage over
the k-space GDF exchange contraction, which scales with the number of
(k, k') pairs (see handovers/HANDOVER_RIJCOSX_M3A.md Sec. "M3b direction", scope
block). The exxdiv='ewald' Madelung finite-size correction is applied
by the caller at the K(k) level, exactly as for the GDF exchange
(``vibeqc.madelung.apply_exxdiv_ewald_to_K``).

All lattice-sum conventions match the direct-ERI reference
(``build_jk_2e_real_space``): density blocks keyed by integer cell
difference, K(g) blocks on the truncated ``direct_lattice_cells``
list, ``bloch_sum`` with the e^{+ik.g} phase.

Validated against the ERI-exact references at every level
(tests/test_periodic_rijcosx.py): per-block parity (M3b-2), and
multi-k SCF parity vs the GDF exchange backend (M3b-3).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

from . import BasisSet, GridOptions, PeriodicSystem, build_grid
from ._vibeqc_core import (
    LatticeSumOptions,
    bloch_sum,
    build_cosx_cell_pair_caches,
    build_cosx_q,
    compute_cosx_k_blocks,
    compute_overlap_lattice,
    compute_shell_radial_cutoffs,
    direct_lattice_cells,
    make_lattice_matrix_set,
)

__all__ = ["KPointCosxK"]


class KPointCosxK:
    """Persistent multi-k COSX exchange builder.

    Builds the SCF-invariant inputs once per geometry -- the truncated
    cell list, the per-relative-shift analytic-integral caches, the
    periodic Becke grid, and the Q-junction -- and turns per-iteration
    k-space densities into k-space exchange matrices.

    Parameters
    ----------
    basis
        AO basis.
    system
        Periodic system (lattice + home cell).
    lat_opts
        :class:`LatticeSumOptions`; ``cutoff_bohr`` bounds both
        lattice indices of the exchange double sum (default 15 bohr).
    grid_options
        COSX integration grid controls. ``None`` uses the standard
        COSX tier (n_radial 35, n_theta 9, n_phi 18 -- the same
        default as molecular RIJCOSX and the C++ periodic builders,
        ~8x sparser than the XC tier; Neese 2009 Sec.3 reports sub-mHa
        per-atom accuracy at this class of quadrature). Pass an
        XC-tier :class:`GridOptions()` for tighter quadrature
        studies.

        The grid is the **molecular** (full-space) Becke grid around
        the home-cell atoms -- NOT the periodic-partition grid. The
        cell-pair engine's quadrature is an R^3 integral with a
        home-pinned bra (``S_G w_G chi_mu(r_G)....`` with home-cell mu);
        the periodic Becke partition restricts the weights to one
        unit-cell-equivalent region and silently undercounts the bra
        tails (measured ~20 % per-block error on the H-chain anchor
        before this was caught -- the M3b-2 validations all used the
        molecular grid).
    schwarz_drop_tol
        Per-shift Schwarz drop threshold for the d difference set.
    omega
        Coulomb-kernel range separation (default 0 = full 1/r). With
        w > 0 the exchange uses the erfc(w.r)/r SHORT-RANGE kernel
        (M3b-4a): absolutely convergent for any density (the kernel
        range 1/w bounds every lattice sum -- alias-immune for
        1/w well inside the BvK half-super-period), validated
        ERI-exact against the erfc direct-ERI reference. Note an
        SR-only K is not the full exchange -- the long-range erf
        complement must be added by the caller (M3b-4b composition).
    """

    # Practical ceiling for the SR cell-domain radius. Bases whose
    # extents push the derived radius past this are outside the
    # real-space SR envelope (ultra-diffuse exponents -- the same
    # constraint CRYSTAL-style codes impose); a warning points at the
    # k-space GDF route / periodic-adapted (pob-class) bases.
    SR_CELL_RADIUS_CAP = 40.0

    def __init__(
        self,
        basis: BasisSet,
        system: PeriodicSystem,
        *,
        lat_opts: Optional[LatticeSumOptions] = None,
        grid_options: Optional[GridOptions] = None,
        schwarz_drop_tol: float = 1e-10,
        omega: float = 0.0,
        sr_cell_cutoff_bohr: Optional[float] = None,
    ) -> None:
        import warnings

        opts = lat_opts if lat_opts is not None else LatticeSumOptions()
        self.basis = basis
        self.omega = float(omega)

        # SR cell-domain radius: with omega > 0 the exchange sums are
        # bounded by basis pair extents + the erfc kernel range -- NOT
        # by the one-electron lattice cutoff. Derive it from the
        # tabulated shell extents (screening tolerance 1e-6; the
        # per-pair SR cutoff tables prune within the domain), capped
        # at SR_CELL_RADIUS_CAP with a loud warning when the basis
        # demands more (LiH/sto-3g: Li 2sp extents of 50-56 bohr at
        # 1e-7 -- ultra-diffuse exponents outside the real-space SR
        # envelope; measured consequence at the 15-bohr domain:
        # broken omega-invariance + 534 mHa, handover Sec. M3b-5).
        if sr_cell_cutoff_bohr is not None:
            r_sr = float(sr_cell_cutoff_bohr)
        elif self.omega > 0.0:
            # Domain scale from the tabulated shell extents at the
            # 1e-4 screening level, empirically anchored: the
            # H/sto-3g chain (extent 20.6 bohr) measures w-invariant
            # at 1e-4 already with a 15-bohr domain, while LiH/sto-3g
            # (Li 2p extent 42.8 bohr) stays broken even at a 40-bohr
            # domain (2e-2...3.6e-2). The per-pair SR cutoff tables
            # prune rigorously *within* the domain.
            r_shell = float(
                np.max(compute_shell_radial_cutoffs(basis, 1e-4))
            )
            atoms_xyz = np.array(
                [np.asarray(a.xyz, dtype=float)
                 for a in system.unit_cell],
                dtype=float,
            )
            spread = (
                float(np.linalg.norm(atoms_xyz, axis=1).max())
                if atoms_xyz.size
                else 0.0
            )
            r_derived = r_shell + 5.0 / self.omega + spread
            r_sr = min(r_derived, self.SR_CELL_RADIUS_CAP)
            # Envelope criterion (CRYSTAL-style diffuse-exponent
            # constraint, measured on LiH/sto-3g -- handover Sec. M3b-5/6):
            # shells reaching past ~25 bohr at the 1e-4 level are
            # outside the real-space SR exchange envelope
            # (H/sto-3g: 20.6 -- inside, chain-validated; Li 2p
            # sto-3g: 42.8 -- outside, measured broken).
            if r_shell > 25.0:
                warnings.warn(
                    "KPointCosxK: ultra-diffuse basis -- max shell "
                    f"extent {r_shell:.1f} bohr at the 1e-4 level "
                    "(> 12). This is outside the real-space SR "
                    "exchange envelope (measured on LiH/sto-3g: "
                    "w-invariance broken at 2e-2...3.6e-2 for 15...40-"
                    "bohr domains; handovers/HANDOVER_RIJCOSX_M3A.md Sec. M3b-6). "
                    "Use the k-space GDF exchange or a periodic-"
                    "adapted (pob-class) basis, or set "
                    "sr_cell_cutoff_bohr explicitly at your own "
                    "risk.",
                    stacklevel=2,
                )
            # Never below the one-electron cutoff (compact bases keep
            # at least the historical domain).
            r_sr = max(r_sr, float(opts.cutoff_bohr))
        else:
            r_sr = float(opts.cutoff_bohr)
        self.sr_cell_cutoff_bohr = r_sr

        self.cells = direct_lattice_cells(system, r_sr)
        self.caches = build_cosx_cell_pair_caches(
            basis, self.cells, schwarz_drop_tol, self.omega
        )
        if grid_options is None:
            # Standard COSX quadrature tier (matches the molecular
            # RIJCOSX default and cpp default_cosx_grid_options()).
            grid_options = GridOptions()
            grid_options.n_radial = 35
            grid_options.n_theta = 9
            grid_options.n_phi = 18
        self.grid = build_grid(
            system.unit_cell_molecule(), grid_options
        )
        self.q = np.asarray(build_cosx_q(basis, self.grid))
        self._cell_r = np.array(
            [np.asarray(c.r_cart, dtype=float) for c in self.cells]
        )
        self._system = system
        self._lat_opts = opts
        self._volume = float(
            abs(np.linalg.det(np.asarray(system.lattice, dtype=float)))
        )
        # Per-(k_i, k_j) LR pair-FT caches, built lazily on the first
        # k_matrices call with lr_complement (SCF-invariant for a
        # fixed k-point set).
        self._lr_cache_key = None
        self._lr_rho = None
        self._lr_w = None
        self._lr_g2 = None
        self._lr_S_k = None

    # ------------------------------------------------------------------
    @staticmethod
    def _bz_weights(
        weights: Optional[Sequence[float]], n_k: int, label: str
    ) -> np.ndarray:
        """Normalise/validate BZ quadrature weights (None = uniform)."""
        if weights is None:
            return np.full(n_k, 1.0 / n_k)
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != n_k:
            raise ValueError(
                f"{label}: weights must align with the k-point list; "
                f"got {w.shape[0]} weights for {n_k} k-points"
            )
        if np.any(w <= 0.0) or not np.isclose(float(w.sum()), 1.0, atol=1e-9):
            raise ValueError(
                f"{label}: weights must be positive and sum to 1; got "
                f"sum {float(w.sum()):.6f}"
            )
        return w

    def real_space_density(
        self,
        D_k: Sequence[np.ndarray],
        kpoints_cart: Sequence[np.ndarray],
        *,
        weights: Optional[Sequence[float]] = None,
    ):
        """Inverse Bloch fold of per-k densities onto the cell list.

        D(g) = S_k w_k e^{-i k.g} D(k), with ``weights`` the BZ
        quadrature weights (``None`` = uniform 1/N_k -- the historical
        behavior, bit-identical). For a time-reversal-symmetric mesh
        (any unshifted Monkhorst-Pack mesh; weighted meshes must keep
        w(k) = w(-k)) the result is real; a large imaginary residue
        signals an inconsistent mesh/density/weight pairing and raises.
        """
        n_k = len(D_k)
        if n_k == 0 or len(kpoints_cart) != n_k:
            raise ValueError(
                "real_space_density: D_k and kpoints_cart must be "
                f"non-empty and aligned; got {n_k} densities, "
                f"{len(kpoints_cart)} k-points"
            )
        w = self._bz_weights(weights, n_k, "real_space_density")
        kpts = [np.asarray(k, dtype=float).reshape(3) for k in kpoints_cart]
        blocks = []
        for rg in self._cell_r:
            P_g = np.zeros(
                (self.basis.nbasis, self.basis.nbasis), dtype=complex
            )
            for i in range(n_k):
                P_g += (
                    w[i]
                    * np.exp(-1j * float(kpts[i] @ rg))
                    * np.asarray(D_k[i])
                )
            imag_norm = float(np.linalg.norm(P_g.imag))
            real_norm = float(np.linalg.norm(P_g.real))
            # The exact P(g) is real for TRS meshes; taking the real
            # part below IS the TRS projection (same convention as the
            # C++ real_space_density_from_kpoints). Mid-SCF densities
            # carry per-k DIIS-extrapolation noise that breaks exact
            # k/-k conjugation -- observed up to ~1e-3 relative on
            # hard-converging configurations, and harmless: the
            # TRS-odd component is projected away here and the
            # converged density is TRS-symmetric. A genuine pairing
            # error (k without -k in the mesh) shows up at O(1)
            # relative; that is what this guard is for.
            if imag_norm > 0.1 * max(1.0, real_norm):
                raise ValueError(
                    "real_space_density: non-negligible imaginary part "
                    f"(||Im|| = {imag_norm:.3e} vs ||Re|| = "
                    f"{real_norm:.3e}) -- the k-mesh/density pairing is "
                    "not time-reversal consistent"
                )
            blocks.append(np.ascontiguousarray(P_g.real))
        return make_lattice_matrix_set(self.basis.nbasis, self.cells, blocks)

    # ------------------------------------------------------------------
    def one_center_correction(
        self,
        D_k: Sequence[np.ndarray],
        *,
        weights: Optional[Sequence[float]] = None,
    ) -> np.ndarray:
        """Analytic one-center replacement for the final exchange.

        Returns the real correction matrix
        ``C = K_exact^(1c) - K_quadrature^(1c)`` for the same-atom
        home-cell exchange blocks, evaluated with this bridge's grid
        and SR kernel (``omega``) on the folded ``D(g=0)`` block
        (``(1/N_k) S_k D(k)``, TRS-projected to real). Adding ``C`` to
        ``K(g=0)`` adds it identically to every ``K(k)`` through the
        Bloch fold, so callers apply it directly at the K(k) level.

        Lifecycle contract (molecular RIJCOSX + the dedicated Gamma
        builders, commit 40dad042 + the 2026-07-15 increment in
        handovers/HANDOVER_RIJCOSX_PBC.md): the SCF iterates and
        reports its energy on the UNCORRECTED seminumerical surface;
        the correction upgrades only the post-convergence returned
        Fock and orbitals.

        Only the real-space SR part carries the same-atom cusp
        quadrature error this replaces -- the reciprocal-space LR-erf
        complement (``lr_complement=True``) is smooth at the nucleus
        and needs no replacement, which is why the correction uses the
        SR kernel at this bridge's ``omega``.
        """
        from ._vibeqc_core import compute_cosx_one_center_correction

        n_k = len(D_k)
        if n_k == 0:
            raise ValueError(
                "one_center_correction: D_k must be non-empty"
            )
        w = self._bz_weights(weights, n_k, "one_center_correction")
        D0 = np.zeros(
            (self.basis.nbasis, self.basis.nbasis), dtype=complex
        )
        for i, D in enumerate(D_k):
            D0 += w[i] * np.asarray(D)
        corr = np.asarray(
            compute_cosx_one_center_correction(
                self.basis,
                np.ascontiguousarray(D0.real),
                self.grid,
                self.omega,
            )
        )
        # The exact K and the true quadrature block are both symmetric;
        # the raw subtraction carries quadrature-level asymmetry from
        # the one-sided bra accumulation. Symmetrise so the correction
        # commutes with the drivers' Hermitian Fock symmetrisation.
        return 0.5 * (corr + corr.T)

    # ------------------------------------------------------------------
    def k_matrices(
        self,
        D_k: Sequence[np.ndarray],
        kpoints_cart: Sequence[np.ndarray],
        *,
        lr_complement: bool = False,
        lr_ke_cutoff: Optional[float] = None,
        weights: Optional[Sequence[float]] = None,
        screened_omega: Optional[float] = None,
    ) -> List[np.ndarray]:
        """Exchange matrices K(k) for every k-point.

        One real-space K(g) build per call (mesh-size independent),
        then a Bloch fold per k. Returns complex Hermitian matrices
        (Hermiticity follows from the engine's pairwise
        K(g) = K(-g)ᵀ symmetrisation). No exxdiv correction is applied
        here -- apply ``apply_exxdiv_ewald_to_K`` at the call site,
        same as for the GDF exchange backend.

        With ``lr_complement=True`` (requires ``omega > 0``) the
        smooth long-range erf complement is added in reciprocal space
        (M3b-4b), so the returned K is the FULL exchange in the same
        G=0-dropped gauge as the GDF backend:

          K = K_SR-w (real space, this object's caches)
            + S_{|G+q|!=0} (4pi/V).e^{-|G+q|^2/4w^2}/|G+q|^2
                . r̂(G+q) D r̂(G+q)+/ N_k
            - (pi/(w^2.V.N_k)) . S(k) D(k) S(k)+

        The last term restores the finite part of the LR kernel at
        G = 0 (ṽ_LR(G->0) = 4pi/G^2 - pi/w^2; the divergent piece is
        dropped exactly as in GDF and handled by the caller's
        exxdiv='ewald' Madelung shift; the real-space SR part already
        contains its own G = 0 physics). r̂ are the ket-Bloch-summed
        AO-pair FTs (same convention and per-AO calibration as
        ``build_lpq_bloch_native_fft``); pair-FT tensors are cached
        per (k_i, k_j) on first use -- SCF-invariant.

        With ``screened_omega = w_s`` (requires ``lr_complement=True``)
        the returned K is the PHYSICAL screened exchange
        ``K_erfc(w_s)`` of an HSE-type hybrid instead of the full 1/r
        exchange: the real-space SR part stays at the bridge's
        numerical split ``w_p = self.omega`` (alias-safety criterion
        unchanged), and the reciprocal complement becomes the BAND
        kernel ``[erf(w_p r) - erf(w_s r)]/r`` with a FINITE G = 0
        value -- no exxdiv Madelung shift applies to the result
        (``periodic_screened_exchange.py``). ``w_s == self.omega``
        short-circuits to the pure real-space SR evaluation (band = 0);
        ``w_s > self.omega`` raises (the SR caches truncate at the
        split reach).
        """
        if screened_omega is not None:
            w_s = float(screened_omega)
            if w_s <= 0.0:
                raise ValueError(
                    "k_matrices: screened_omega must be > 0; got "
                    f"{screened_omega}"
                )
            if not lr_complement:
                raise ValueError(
                    "k_matrices: screened_omega requires "
                    "lr_complement=True (the screened exchange is the "
                    "SR + band composition)"
                )
            if self.omega <= 0.0:
                raise ValueError(
                    "k_matrices: screened_omega requires a bridge built "
                    "with omega > 0 (the numerical SR split)"
                )
            if w_s > self.omega + 1e-12:
                raise ValueError(
                    "k_matrices: screened_omega must not exceed the "
                    "bridge's SR split omega (the real-space caches "
                    f"truncate at the {self.omega:.4g} bohr^-1 reach); "
                    f"got screened_omega={w_s:.4g}. Build the bridge "
                    "with omega=screened_omega for the pure real-space "
                    "screened evaluation."
                )

        P_set = self.real_space_density(D_k, kpoints_cart, weights=weights)
        K_set = compute_cosx_k_blocks(
            self.basis, P_set, self.grid, self.caches, q_cached=self.q
        )
        K_k = []
        for k in kpoints_cart:
            k_arr = np.asarray(k, dtype=float).reshape(3)
            Kk = np.asarray(bloch_sum(K_set, k_arr))
            K_k.append(0.5 * (Kk + Kk.conj().T))

        if screened_omega is not None and abs(
            float(screened_omega) - self.omega
        ) < 1e-12:
            # The physical screening IS the split: the erfc(w_s r)/r
            # kernel is evaluated entirely in real space (band == 0,
            # G = 0 constant == 0) -- the CRYSTAL-style direct SR
            # assembly. Valid whenever the caller ensured the w_s reach
            # fits the BvK torus (the driver's envelope criterion).
            return K_k

        if not lr_complement:
            return K_k
        if self.omega <= 0.0:
            raise ValueError(
                "k_matrices: lr_complement=True requires omega > 0 "
                "(the SR/LR split parameter)"
            )

        K_lr = self._lr_k_matrices(
            D_k,
            kpoints_cart,
            lr_ke_cutoff,
            weights=weights,
            screened_omega=screened_omega,
        )
        return [
            0.5 * (Ks + Kl + (Ks + Kl).conj().T)
            for Ks, Kl in zip(K_k, K_lr)
        ]

    # ------------------------------------------------------------------
    def _lr_setup(
        self,
        kpoints_cart: Sequence[np.ndarray],
        lr_ke_cutoff: Optional[float],
    ) -> None:
        """Build (lazily, once per k-point set) the LR pair-FT caches.

        Per (k_i, k_j): the ket-Bloch-summed pair FT r̂ at the shifted
        mesh G + q (q = k_j - k_i, k_cart = k_j -- the
        ``build_lpq_bloch_native_fft`` convention, including the
        per-AO libint-calibration scales), and the LR Coulomb weights
        (4pi/V).exp(-|G+q|^2/4w^2)/|G+q|^2 with the |G+q| = 0 mode
        dropped. Also the Bloch overlaps S(k) for the G = 0 finite-
        part correction.
        """
        from ._aopair_ft import ao_pair_fourier_transform_bloch
        from .aux_basis import _ao_scales_for_rsgdf, rsgdf_dense_g_mesh

        kpts = [np.asarray(k, dtype=float).reshape(3) for k in kpoints_cart]
        key = (
            tuple(tuple(np.round(k, 12)) for k in kpts),
            float(lr_ke_cutoff) if lr_ke_cutoff is not None else None,
        )
        if self._lr_cache_key == key:
            return

        # The LR kernel kills |G| ≳ ~6w (exp(-9) ≈ 1e-4 of the peak
        # weight before the 1/G^2 damping); ke = G^2/2 with 2x margin.
        ke = (
            float(lr_ke_cutoff)
            if lr_ke_cutoff is not None
            else max(2.0, 0.5 * (8.0 * self.omega) ** 2)
        )
        G_all = rsgdf_dense_g_mesh(self._system, ke)

        ao_scales = _ao_scales_for_rsgdf(self.basis)
        pair_scales = np.outer(ao_scales, ao_scales)

        cells = direct_lattice_cells(
            self._system, float(self.sr_cell_cutoff_bohr)
        )
        R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
        if R_g.size == 0:
            R_g = np.zeros((1, 3), dtype=float)

        n_k = len(kpts)
        n_bf = self.basis.nbasis
        rho = {}
        w = {}
        g2_kept = {}
        for i in range(n_k):
            for j in range(n_k):
                q_ij = kpts[j] - kpts[i]
                Gq = G_all + q_ij[None, :]
                g2 = (Gq**2).sum(axis=1)
                nz = g2 > 1e-12
                Gq = Gq[nz]
                g2 = g2[nz]
                w_ij = (
                    (4.0 * np.pi / self._volume)
                    * np.exp(-g2 / (4.0 * self.omega**2))
                    / g2
                )
                # Weight prefilter: the Gaussian damping kills most of
                # the ke-ball; the pair-FT (the expensive general-L
                # Hermite-chain kernel) only needs the G's that carry
                # LR weight. Dropped terms are bounded by
                # 1e-12 . max(w) . ‖r̂‖^2 each -- far below quadrature
                # error. Cuts the per-(k_i, k_j) pair-FT cost several-
                # fold on 3D meshes.
                keep = w_ij > 1e-12 * float(w_ij.max())
                Gq = np.ascontiguousarray(Gq[keep])
                w[(i, j)] = w_ij[keep]
                g2_kept[(i, j)] = g2[keep]
                ft = ao_pair_fourier_transform_bloch(
                    self.basis, Gq, R_g, k_cart=kpts[j]
                )
                rho_ij = np.asarray(ft.data, dtype=complex).reshape(
                    n_bf, n_bf, -1
                )
                rho[(i, j)] = rho_ij * pair_scales[:, :, None]

        # Bloch overlaps for the G = 0 finite-part correction.
        S_set = compute_overlap_lattice(
            self.basis, self._system, self._lat_opts
        )
        S_k = [np.asarray(bloch_sum(S_set, k)) for k in kpts]

        self._lr_cache_key = key
        self._lr_rho = rho
        self._lr_w = w
        self._lr_g2 = g2_kept
        self._lr_S_k = S_k

    def _lr_k_matrices(
        self,
        D_k: Sequence[np.ndarray],
        kpoints_cart: Sequence[np.ndarray],
        lr_ke_cutoff: Optional[float],
        weights: Optional[Sequence[float]] = None,
        screened_omega: Optional[float] = None,
    ) -> List[np.ndarray]:
        """Reciprocal-space complement at every k.

        With ``screened_omega=None`` (default): the LR-erf complement
        of the full 1/r kernel (G=0-dropped exxdiv gauge + the
        -pi/w^2 finite-part restoration). See the ``k_matrices``
        docstring for the formula.

        With ``screened_omega = w_s`` (HSE-type screened exchange): the
        BAND kernel ``[erf(w_p r) - erf(w_s r)]/r`` completing the
        real-space SR part to the PHYSICAL screened kernel
        ``erfc(w_s r)/r``. Reciprocal weights
        ``(4pi/V)[exp(-|G+q|^2/4w_p^2) - exp(-|G+q|^2/4w_s^2)]/|G+q|^2``;
        the G = 0 value is FINITE, ``pi(1/w_s^2 - 1/w_p^2)/V`` via the
        Bloch overlaps -- no divergent piece is dropped, so NO exxdiv
        Madelung shift applies to the composed screened exchange (the
        erfc kernel has no G -> 0 divergence; see
        ``periodic_screened_exchange.py``).

        The BZ average over the source point ``k_j`` uses the supplied
        quadrature ``weights`` (``None`` = uniform 1/N_k, the historical
        behavior); the G = 0 term is the ``k_j = k_i`` contribution of
        the same sum and carries ``w_i``.
        """
        self._lr_setup(kpoints_cart, lr_ke_cutoff)
        n_k = len(kpoints_cart)
        w_bz = self._bz_weights(weights, n_k, "_lr_k_matrices")
        n_bf = self.basis.nbasis
        w_s = None if screened_omega is None else float(screened_omega)
        K_lr = [
            np.zeros((n_bf, n_bf), dtype=complex) for _ in range(n_k)
        ]
        for i in range(n_k):
            for j in range(n_k):
                rho = self._lr_rho[(i, j)]
                wij = self._lr_w[(i, j)]
                if w_s is not None:
                    # Band weights: subtract the screened Gaussian at
                    # the SAME kept G's. G's dropped by the w_p
                    # prefilter carry even smaller w_s terms
                    # (w_s < w_p => steeper Gaussian), so the kept set
                    # covers the band exactly.
                    g2 = self._lr_g2[(i, j)]
                    wij = wij - (
                        (4.0 * np.pi / self._volume)
                        * np.exp(-g2 / (4.0 * w_s**2))
                        / g2
                    )
                Dj = np.asarray(D_k[j])
                # K(k_i)[p,q] += w_j S_G w_G r[p,r,G] D[r,s] r*[q,s,G]
                tmp = np.einsum("prG,rs->psG", rho, Dj, optimize=True)
                K_lr[i] += w_bz[j] * np.einsum(
                    "psG,qsG,G->pq", tmp, rho.conj(), wij, optimize=True
                )
        # G = 0 term (q -> 0 k-pairs). Full-range mode: the divergent
        # 4pi/G^2 piece is dropped (exxdiv gauge, restored by the
        # caller's Madelung shift) and the finite -pi/w_p^2 piece
        # belongs in the sum. Screened mode: the band kernel's G -> 0
        # limit is FINITE, +pi(1/w_s^2 - 1/w_p^2) -- nothing is
        # dropped. r̂(q=0, G=0) = S(k). Detecting q = 0 numerically
        # (not by index equality) keeps duplicated-k quadratures
        # consistent.
        if w_s is None:
            g0_coeff = -np.pi / (self.omega**2 * self._volume)
        else:
            g0_coeff = (
                np.pi
                * (1.0 / w_s**2 - 1.0 / self.omega**2)
                / self._volume
            )
        kpts = [np.asarray(k, dtype=float).reshape(3) for k in kpoints_cart]
        for i in range(n_k):
            S = self._lr_S_k[i]
            for j in range(n_k):
                if float(np.linalg.norm(kpts[j] - kpts[i])) > 1e-10:
                    continue
                K_lr[i] += (g0_coeff * w_bz[j]) * (
                    S @ np.asarray(D_k[j]) @ S.conj().T
                )
        return [0.5 * (K + K.conj().T) for K in K_lr]
