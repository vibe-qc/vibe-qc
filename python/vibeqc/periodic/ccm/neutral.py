"""Neutral (Ewald) effective four-center for a fitted-torus control.

This module supplies the **charge-neutral** effective four-center ``g_eff`` of
the separately declared neutral fitted-torus control and tools for comparing it
with the union-and-weight Γ-CCM construction.  It does not implement Γ-CCM, and
agreement with it does not prove that the two constructions are identical.

The background -- why a neutral four-center
-----------------------------------------
The WSSC four-center
(:func:`vibeqc.periodic.ccm.padded.ccm_eri_symmetric`) belongs to the
union-and-weight Γ-CCM construction.  This module instead fits the neutral
periodic Green's function ``v_E`` on a finite torus.  Three-dimensional
comparisons have shown large, Madelung-scale route gaps, but those gaps compare
different declared operators and do not establish that one is a correction to
the other.  For a scalable calculation of the same Γ-CCM construction use
:func:`~vibeqc.periodic.ccm.scf.run_ccm_rhf_scalable`.  Use
:func:`~vibeqc.periodic.ccm.ri.run_ccm_rhf_gdf` only when the neutral
fitted-torus control itself is the object being studied, and label it as such.
The neutral reciprocal-mesh control is not defined below three dimensions.

What the neutral-control ``g_eff`` is
-------------------------------------
``g_eff[muν,rs] = (muν|v_E|rs)`` over the home (supercell) AOs, realised as the
**RI / density fit of ``v_E``**.  The real-Γ supercell cderi and the character-
mesh GDF form are Fourier representations of this one already specified,
block-circulant neutral-control Hamiltonian:

    # GDF cderi reconstruction (Sun, Berkelbach, McClain & Chan, J. Chem. Phys.
    # 147, 164119 (2017), doi:10.1063/1.4998644 -- range-separated periodic GDF):
    #   (muν|v_E|rs)  =  S_P L_{P,muν} . L_{P,rs}        (k_bra = k_ket = 0, Γ)
    # where L is build_lpq_bloch_native_fft(cluster_system, ao, aux, 0, 0); the
    # G+q=0 mode is dropped (the neutralising-background / Ewald convention),
    # which is precisely what makes the kernel v_E charge-neutral.

``g_eff`` is **exactly 8-fold permutationally symmetric** (it is ``S_P L L`` with
``L_{P,muν}=L_{P,νmu}`` real at Γ) -- the property a single variational RHF/MP2/CCSD
energy functional requires, and the Sec.13.6 3-D soundness test.

The Madelung gap is a mean-field background ``ξ.S⊗S`` -- a fixed-reference diagnostic
------------------------------------------------------------------------------------
In the molecular / isolated limit the neutral and the full (bare) four-center
differ by a **rank-1 background**:

    # g_eff[muν,rs] = (muν|rs)_full - ξ . S_muν . S_rs ,
    #   q_muν == ∫ phi_mu phi_ν = S_muν   (the AO-pair "charge"),
    #   ξ = the cell's neutralising-background (Madelung) constant
    #       = madelung_constant_for_cell(cluster_system).
    # Verified: fitted ξ -> madelung_constant_for_cell as the box grows
    # (ratio 0.975 -> 0.984 -> 0.991 over L = 12 -> 15 -> 20 bohr; the residual is
    # the cderi RI fitting error), STO-3G H₂.

Two consequences follow, and they are the resolution of the gap:

1. **HF / KS energy.** ``ξ.S⊗S`` is ``∝ S`` in the Fock (``J`` background
   ``-ξ.S.tr(SD)``), i.e. a *uniform mean-field shift*. For an energy under the
   neutral fitted-torus control use
   :func:`vibeqc.periodic.ccm.ri.run_ccm_rhf_gdf` /
   :func:`~vibeqc.periodic.ccm.ri.run_ccm_rks_gdf`.  Multi-k GDF and the real-Γ
   supercell fit are representations of that control, not substitutes for the
   union-and-weight Γ-CCM construction.  Do not assume that an energy difference
   cancels the route gap; demonstrate cancellation under a fixed operator.

2. **Correlation -- fixed reference only (CORRECTED, do not over-claim).**
   Transformed to the MO basis the background is
   ``ξ.(CᵀSC)⊗(CᵀSC) = ξ.I⊗I`` -- diagonal in each pair index, so it has **no
   occupied-virtual element** (``(CᵀSC)_ia = d_ia = 0``). At a **fixed**
   reference it therefore contributes **exactly zero** to the MP2/CCSD
   correlation *numerator* (machine-zero for any ``ξ``;
   :func:`tests <tests.test_ccm_neutral>`). The ``(CᵀSC)_ia = 0`` step is exact
   only under the block-orthogonality condition ``Cₒᵀ S_plain C_v = 0`` against
   the *plain* AO overlap of the dropped mode -- guaranteed when ``C`` is
   orthonormal w.r.t. that same ``S_plain`` (a plain-``S`` / molecular reference),
   and observed to hold to machine precision on the centrosymmetric H₂-chain
   fixture (evidence for that fixture, **not** a general centrosymmetry theorem).
   A general CCM ``C`` is ``S^CCM``-orthonormal (``S^CCM ≠ S_plain`` at finite
   cluster), so ``Cₒᵀ S_plain C_v`` is in general a small non-zero leak ``∝
   ‖S^CCM − S_plain‖`` that *vanishes as the cluster grows* -- measured
   ``≤ 1e-9`` Ha at physical ``ξ`` for the smallest inversion-broken cluster,
   machine-zero by nrep ``(3,1,1)`` (FINDINGS.md F6, 2026-07-13). Negligible in
   practice; the higher-symmetry benchmark crystals sit far inside that bound.
   It does **not** follow that
   correlation is Madelung-invariant: the background is block-diagonal in
   occ/virt, so it leaves the HF *orbitals* invariant but **shifts the
   occ-virt gap** (every MP2/CCSD *denominator*) by ``O(ξ)``. Independent
   bare-vs-neutral SCF->correlation runs therefore **do not agree** -- measured
   ``+67 %`` MP2 shift at ``ξ=0.5`` (``+17 %`` at the H₂-cell ``ξ≈0.186``) for
   independent references on H₂/STO-3G/12-bohr. **A calculation that declares
   the neutral (Ewald) control must use its neutral reference for correlation as
   well as HF**; the rank-1 result is a *fixed-reference diagnostic* that
   isolates the route difference, **not** an
   all-post-HF invariance theorem (the earlier "rigorously Madelung-robust"
   wording is retracted -- 2026-06-22/24 head-to-head with ``aiccm2026dev-b``).
   Within a neutral-control study, run the post-HF stack
   (:mod:`~vibeqc.periodic.ccm.mp2`,
   :mod:`~vibeqc.periodic.ccm.ump2`, :mod:`~vibeqc.periodic.ccm.ccsd`) on the
   **neutral** reference and four-center / cderi throughout.  This consistency
   requirement is not an instruction to replace the Γ-CCM reference. Also note
   ``ξ.S⊗S`` is the
   *leading* molecular-limit term only: ``v_E − 1/r`` is not spatially constant
   on a finite torus, so an image / multipolar / RI-fitting remainder survives
   beyond rank-1.

Scope
-----
``g_eff`` is a dense ``n_AO**4`` object built from a Γ cderi on the supercell --
a **small-cluster analysis / validation tool for the neutral control** (it
checks 8-fold symmetry, the ``ξ.S⊗S`` decomposition, and the *fixed-reference*
correlation cancellation -- not all-reference robustness, see above).  The
scalable form of the same control is the GDF route above.  Neither form is the
union-and-weight Γ-CCM construction or a proof of it. Cite Peintinger &
Bredow, *J. Comput. Chem.* **35**, 839 (2014) for Γ-CCM and
Sun *et al.* (2017) for the periodic GDF.
"""

from __future__ import annotations

import numpy as np

from vibeqc.output import Level, write

__all__ = [
    "ccm_neutral_background_constant",
    "ccm_neutral_cderi",
    "ccm_neutral_cderi_fold",
    "ccm_neutral_tail_ke_cutoff",
    "ccm_sr_exchange_zero_mode_constant",
    "ccm_eri_neutral",
]


def _fold_thread_count(fold_threads, n_c):
    """Resolve the fold's WSC/q-axis thread count.

    ``None`` reads ``VIBEQC_CCM_FOLD_THREADS`` and otherwise stays serial
    (1). Opt-in on purpose: the per-(ka, kb) fits are BLAS-heavy, so with a
    multi-threaded BLAS an outer thread pool oversubscribes the node and can
    be SLOWER than serial. The measured win needs the two levels separated
    -- pin BLAS to one thread (``OMP_NUM_THREADS=1``) and give the pool the
    cores. Flipping the default is a measurement-gated decision, not a code
    default (CLAUDE.md § 15 makes full-node scaling a deliverable, so the
    recipe belongs in the docs, and the fleet ladder is what settles it).
    """
    import os

    if fold_threads is None:
        env = os.environ.get("VIBEQC_CCM_FOLD_THREADS", "").strip()
        fold_threads = int(env) if env else 1
    fold_threads = int(fold_threads)
    if fold_threads <= 0:                       # 0 / negative = "all cores"
        fold_threads = os.cpu_count() or 1
    return max(1, min(fold_threads, int(n_c)))


def _supercell_wrap_lat_opts(ccm, base=None):
    """Lattice-sum options with the cutoff widened for cross-torus AO pairs.

    The Γ-supercell cderi build stores AO pairs at their *absolute* supercell
    positions -- a pair ``(mu at cell 0, ν at cell N-1)`` sits at up to one
    supercell vector of stored separation, while its true torus (minimum-image)
    separation is small. The builder's real-space image sums must therefore
    reach the image cell that wraps the pair back: the default
    ``cutoff_bohr`` (~15, tuned for *unit*-cell pair extents) silently drops
    that image once the supercell outgrows it, and the fitted kernel then
    **breaks torus translation invariance** (measured: J/K asymmetry 4e-2 on
    an 18-bohr (3,1,1) chain supercell; machine ε once the wrap image is
    included). Widen by the worst-case wrap vector: the longest
    ``||S_i ε_i A_i^SC||`` over ``ε_i ∈ {0,1}`` restricted to replicated axes
    (``N_i > 1``; unreplicated axes have no cross-cell pairs and are covered
    by the base cutoff's overlap reach).
    """
    from vibeqc import LatticeSumOptions
    from vibeqc.lattice_screening import RcutStrategy, make_lattice_opts

    base = base if base is not None else LatticeSumOptions()
    a_sc = np.asarray(ccm.cluster_system.lattice, dtype=float)  # columns = vectors
    axes = [i for i in range(3) if int(ccm.nrep[i]) > 1]
    reach = 0.0
    for bits in range(1, 1 << len(axes)):
        v = np.zeros(3)
        for j, ax in enumerate(axes):
            if bits & (1 << j):
                v += a_sc[:, ax]
        reach = max(reach, float(np.linalg.norm(v)))
    if reach == 0.0:
        return base
    return make_lattice_opts(
        ccm.basis,
        strategy=RcutStrategy.FLAT,
        base_opts=base,
        cutoff_bohr=float(base.cutoff_bohr) + reach,
    )


def _supercell_modrho_aux(ccm, aux_basis=None):
    """The modrho-rescaled auxiliary BasisSet of the cluster supercell.

    The exact aux construction the Γ-supercell fit uses -- shared with the
    symmetry machinery, which needs the same BasisSet object to build the
    auxiliary AO rotation matrices (:func:`ccm_symmetry_basis_rotations`)
    that reconstruct non-representative pair columns of the cderi.
    """
    from vibeqc.aux_basis import (
        default_aux_for,
        make_aux_basis_set,
        make_modrho_aux_basis,
    )

    mol = ccm.cluster_system.unit_cell_molecule()
    aux_name = aux_basis or default_aux_for(ccm.basis.name)
    aux = make_aux_basis_set(mol, aux_name=aux_name, drop_eta=0.0)
    return make_modrho_aux_basis(aux, mol)


def ccm_neutral_background_constant(ccm) -> float:
    """The neutralising-background (Madelung) constant ``ξ`` of the cluster cell.

    ``ξ`` is the constant of the rank-1 background that separates the neutral
    four-center from the full one in the molecular limit
    (``g_eff = (muν|rs)_full - ξ.S_muν.S_rs``); it is the Madelung constant of the
    cluster (supercell) cell, i.e. the ``G=0`` term that the neutral ``v_E``
    drops and replaces by the compensating uniform background. Lattice-general.
    """
    from vibeqc.madelung import madelung_constant_for_cell

    return float(madelung_constant_for_cell(ccm.cluster_system))


def ccm_sr_exchange_zero_mode_constant(ccm, omega_screen: float) -> float:
    """The finite ``G+q=0`` weight of the screened (erfc-SR) torus kernel.

    The Fourier transform of the short-range kernel ``erfc(w r)/r`` is
    ``(4 pi / p^2)(1 - exp(-p^2 / (4 w^2)))``, whose ``p -> 0`` limit is the
    FINITE ``pi / w^2`` -- unlike the bare-Coulomb ``4 pi / p^2`` divergence,
    no neutralising background and no exchange-``q=0`` convention is needed
    (the resolver policy: no Madelung / exxdiv seam attaches to the screened
    kernel, and none must be). The screened-attenuated cderi builders
    (``omega_screen > 0``) still exclude the ``p = 0`` point from the fit,
    so the fitted four-center misses exactly the rank-1 term

        (mu nu | erfc | rho sigma)  ⊃  sigma_0 . S_{mu nu} . S_{rho sigma},
        sigma_0 = pi / (omega_screen^2 . V_supercell)

    which this constant restores analytically: the SR exchange operator is
    ``K_sr(D) = K[L_sr](D) + sigma_0 . S D S`` -- exact (no RI error on the
    zero mode), linear + symmetric in ``D`` so ``F = dE/dD`` holds through
    the standard functional, the same plumbing as the ``ξ_N . S D S``
    exchange-q=0 seam but *physical kernel content*, not a convention
    (experimental: HSE-class screened hybrids on the direct-torus route).
    """
    w = float(omega_screen)
    if w <= 0.0:
        raise ValueError(
            "ccm_sr_exchange_zero_mode_constant: omega_screen must be > 0 "
            f"(got {w}); the full-range Coulomb kernel's G+q=0 mode is the "
            "divergent Madelung/exxdiv seam, not this finite constant."
        )
    v_sc = float(abs(np.linalg.det(
        np.asarray(ccm.cluster_system.lattice, dtype=float))))
    return np.pi / (w * w * v_sc)


def ccm_neutral_cderi(
    ccm, *, ke_cutoff: float = 200.0, tail_ke_cutoff=None, aux_basis=None,
    lat_opts=None, canonical_auxiliary_basis: bool = False, symmetry=None,
    omega_screen: float = 0.0,
) -> np.ndarray:
    """Neutral (Ewald) GDF cderi ``L[P,muν]`` on the cluster supercell at Γ.

    The **RI-consistent three-center** of the separately declared neutral
    fitted-torus control: the density-fit tensor of ``v_E``, from which the
    neutral four-center, RI-J and RI-K all follow by contraction (it is the
    ``B.V^{-1/2}`` cderi -- the metric is already folded in, so there is no
    separate ``V^{-1}``):

        L = build_lpq_bloch_native_fft(ccm.cluster_system, ao, aux, 0, 0)
        (muν|v_E|rs) = S_P L[P,muν].L[P,rs]        (k_bra=k_ket=0)

    The real-Γ and character-mesh forms are related by a finite Fourier
    transform only after this block-circulant neutral-control Hamiltonian has
    been specified.  That representation theorem neither constructs nor proves
    equivalence to union-and-weight Γ-CCM.

    The ``G+q=0`` mode is dropped (Ewald / neutralising-background convention),
    so the fitted kernel is charge-neutral and -- unlike the bare-1/r WSSC
    three-center (:func:`~vibeqc.periodic.ccm.ri.ccm_ri_tensors`, the eq-13 union
    weight) -- **RI-consistent with the four-center**: its RI-J/RI-K reproduce the
    neutral four-center exactly, where the bare-1/r union RI-J floors at ~1-2 %
    because the bare-1/r four-center is non-separable (Sec.13.7).

    Returns the ``(n_aux, nbf, nbf)`` cderi. Heavy (all-FT GDF on the supercell)
    -- small-cluster path; the scalable neutral-control energy is the multi-k
    GDF route (:func:`~vibeqc.periodic.ccm.ri.run_ccm_rhf_gdf`).

    ``lat_opts`` overrides the real-space lattice-sum options of the build.
    Default (``None``): the base options **widened for cross-torus AO pairs**
    (:func:`_supercell_wrap_lat_opts`) -- without the widening, supercells
    outgrowing the flat ~15-bohr cutoff lose the image cell that wraps
    edge pairs back onto the torus, and the fitted kernel silently breaks
    torus translation invariance (fixed 2026-07-02; measured 4e-2 J/K
    asymmetry / 1e-4 Ha-per-cell energy error on an 18-bohr chain supercell).

    ``canonical_auxiliary_basis`` is forwarded to the builder: the default
    (False) returns the compact eigenmode factor (``n_kept`` rows); True
    returns the fit rotated back to the auxiliary-AO frame (``n_aux`` rows).
    All contractions (``g_eff``, J/K) are identical between the two; the
    canonical frame is required whenever cderi *columns* are compared or
    reconstructed across independent builds -- in particular by the
    space-group pair reduction, whose auxiliary rotation matrices are
    defined in the aux-AO frame.

    ``symmetry`` routes the fit through the space-group pair reduction
    (:func:`~vibeqc.periodic.ccm.symmetry.ccm_symmetry_unique_atom_pairs`):
    only orbit-representative atom-pair AO blocks are fit, the rest are
    reconstructed by the group action. The covariance identity and its
    quantitative finite-cell-list floors are documented at the reconstruction
    site below. Accepts ``None``/``False`` (off, the default), ``True``/"auto"
    (analyze the cluster), or a pre-computed
    :class:`~vibeqc.periodic.ccm.symmetry.CCMSymmetry`. A cluster with no
    symmetry beyond the identity (|G_c| = 1, e.g. P1/triclinic) falls back
    to the full build with no reduction claim. Any ``symmetry=...`` call
    returns the **canonical auxiliary frame** (the frame the reconstruction
    is defined in) whether or not a reduction was possible, so the frame is
    predictable downstream; contractions are identical to the un-reduced
    build's either way.

    ``tail_ke_cutoff`` (default ``None``) completes the RSGDF fit over the
    high-``|G+q|`` shell above ``ke_cutoff``. ``None`` resolves it through
    :func:`ccm_neutral_tail_ke_cutoff`, which mirrors the multi-k GDF
    sibling's decision so the two controls evaluate the same Hamiltonian
    (GitLab IID 307); pass an explicit value (``0`` disables) to override.

    ``omega_screen`` (experimental, default ``0.0`` = full Coulomb,
    bit-identical historical build) fits the **screened erfc-SR kernel**
    ``erfc(omega_screen * r_12)/r_12`` instead -- the exact-exchange kernel
    of HSE-class screened hybrids (see the builder docstring for the
    Fourier form). The fitted tensor still excludes the ``G+q=0`` point,
    whose screened weight is FINITE; consumers restore it analytically via
    :func:`ccm_sr_exchange_zero_mode_constant` (rank-1 ``sigma_0 S⊗S``).
    Screened fits are exchange-only objects: never use one for ``J`` (the
    Hartree channel is always the full-range neutral kernel).
    """
    from vibeqc.aux_basis import build_lpq_bloch_native_fft

    sysc = ccm.cluster_system
    aux_modrho = _supercell_modrho_aux(ccm, aux_basis)
    gamma = np.zeros(3)
    opts = lat_opts if lat_opts is not None else _supercell_wrap_lat_opts(ccm)
    # Same Hamiltonian-identity seam as the fold build (IID 307): both
    # cderi_build modes must resolve the tail the multi-k GDF sibling would.
    tail = ccm_neutral_tail_ke_cutoff(
        ccm, ke_cutoff=ke_cutoff, tail_ke_cutoff=tail_ke_cutoff
    )

    sym = None
    if symmetry is not None and symmetry is not False:
        from .symmetry import analyze_ccm_symmetry

        sym = (symmetry if not isinstance(symmetry, (bool, str))
               else analyze_ccm_symmetry(ccm))

    if sym is not None:
        from .symmetry import ccm_symmetry_unique_atom_pairs

        orbits = ccm_symmetry_unique_atom_pairs(ccm, sym)
        if orbits.n_unique < orbits.n_total:
            return _ccm_neutral_cderi_symm_reduced(
                ccm, sym, orbits, aux_modrho, opts, float(ke_cutoff),
                omega_screen=float(omega_screen), tail_ke_cutoff=tail)
        write(
            f"  ccm_neutral_cderi: no space-group pair reduction available "
            f"(|G_c| = {sym.cluster_order}, {orbits.n_unique}/"
            f"{orbits.n_total} unique pairs) -- full build\n",
            Level.VERBOSE)
        # Keep the symmetry contract's frame predictable: a symmetry=...
        # caller always gets the canonical aux frame, whether or not a
        # reduction was possible (the fallback is the full build either way).
        canonical_auxiliary_basis = True

    # k_bra = k_ket = 0 -> the Γ cderi over the supercell AOs; the all-FT route
    # is the validated (PySCF-µHa parity) builder used by the scalable multi-k
    # GDF control, and drops G+q=0 (the neutralising-background convention).
    return np.real(
        np.asarray(
            build_lpq_bloch_native_fft(
                sysc, ccm.basis, aux_modrho, gamma, gamma,
                ke_cutoff=float(ke_cutoff), tail_ke_cutoff=tail,
                lat_opts=opts,
                canonical_auxiliary_basis=bool(canonical_auxiliary_basis),
                omega_screen=float(omega_screen),
            )
        )
    )


def _ccm_neutral_cderi_symm_reduced(
    ccm, sym, orbits, aux_modrho, opts, ke_cutoff: float,
    omega_screen: float = 0.0, tail_ke_cutoff=None,
) -> np.ndarray:
    """Space-group pair-reduced Γ-supercell neutral cderi.

    Fits only the orbit-representative atom-pair AO blocks (``fit_pair_list``
    on :func:`~vibeqc.aux_basis.build_lpq_bloch_native_fft`) and reconstructs
    every other pair block by the group action. The reconstruction identity
    -- the space-group covariance of the density-fit 3-index tensor at Γ in
    the canonical auxiliary frame -- is, for a cluster-invariant op ``g``
    with orbital AO rotation ``P`` (atom permutation x real Wigner-D,
    ``vibeqc.symmetry_ao.build_ao_permutation_matrix``) and the analogous
    auxiliary rotation ``P_aux``:

        L_{Q, a, b} = S_{P,mu,ν} (P_aux)_{Q P} P_{a mu} P_{b ν} L_{P, mu, ν}
        with (a, b) on the atom pair (g A, g B) for (mu, ν) on (A, B).

    It holds for the *computed* tensor to the accuracy of ONE ingredient:
    the Bloch-sum lattice cell list. The reciprocal |G| <= G_max ball, the
    Coulomb kernel (and its erfc-screened attenuation, a function of
    ``|G+q|`` only, hence equally point-group invariant), the per-shell
    calibration scales, and the 2c metric
    ``M`` (hence its matrix function ``M^{-1/2}``, which commutes with
    ``P_aux``; measured invariant to 2e-14) are exactly point-group
    invariant, and at Γ the fractional-translation phases of non-symmorphic
    ops cancel between the aux and AO-pair Fourier factors. The cell list,
    however, enters pair covariance through *atom-shifted* copies of
    itself, so a finite radius leaves a truncated inter-cell tail that is
    not orbit-symmetric: the covariance residual equals the tail size at
    the cutoff (measured on C-diamond sto-3g at Γ: 8e-9 relative at the
    15-bohr default, 2e-15 at 20 bohr, 6e-16 at 25+ bohr -- exponential in
    the cutoff, same knob as the builder's documented lattice-cutoff
    accuracy floor). Consequently the reduced build equals the un-reduced
    one to machine precision at a cell-list-converged cutoff (pinned
    < 1e-10 in ``tests/test_ccm_symmetry.py``), and to the truncation
    floor -- the un-reduced build's own accuracy floor, ~5e-9 relative on
    the cderi, ~1e-12 on the SCF energy -- at the standard default.

    The petite-list strategy itself (compute one representative integral
    block per orbit, scatter the rest by the group action) is the
    symmetry-exploitation tradition of periodic LCAO codes (cf. Dovesi,
    *Int. J. Quantum Chem.* **29**, 1755 (1986), doi:10.1002/qua.560290608
    -- CRYSTAL); the covariance derivation above is self-contained.
    """
    from vibeqc.aux_basis import build_lpq_bloch_native_fft

    from .symmetry import ccm_symmetry_basis_rotations

    n = int(ccm.n_atoms)
    nbf = int(ccm.basis.nbasis)
    ao_atom = np.asarray(ccm.ao_atom, dtype=int)
    atom_ao = [np.flatnonzero(ao_atom == a) for a in range(n)]

    # Restrict the fit to the orbit-representative atom-pair blocks via the
    # builder's ``fit_pair_list`` shell-pair seam (shared with the Schwarz
    # fit screen -- symmetry drops *redundant* pairs, the screen drops
    # *negligible* ones; one build path, HANDOVER_GDF_FIT_SCREENING.md).
    rep_atom_pairs = set(orbits.representatives)
    shell_atom = [int(sh.atom_index) for sh in ccm.basis.shells()]
    n_sh = len(shell_atom)
    pair_list = np.array(
        [(i, j) for i in range(n_sh) for j in range(n_sh)
         if (shell_atom[i], shell_atom[j]) in rep_atom_pairs],
        dtype=int)
    write(
        f"  ccm_neutral_cderi: space-group pair reduction "
        f"|G_c| = {sym.cluster_order}: {orbits.n_unique}/{orbits.n_total} "
        f"atom pairs fit (reduction {orbits.reduction_factor:.2f}x, "
        f"{len(pair_list)}/{n_sh * n_sh} shell pairs)\n",
        Level.VERBOSE)

    gamma = np.zeros(3)
    # Canonical aux frame is REQUIRED: the covariance above is stated in the
    # aux-AO basis; the compact eigenmode frame carries arbitrary per-build
    # rotations (see the builder docstring) under which pair columns are not
    # comparable.
    L = np.ascontiguousarray(np.real(np.asarray(build_lpq_bloch_native_fft(
        ccm.cluster_system, ccm.basis, aux_modrho, gamma, gamma,
        ke_cutoff=ke_cutoff, tail_ke_cutoff=tail_ke_cutoff, lat_opts=opts,
        canonical_auxiliary_basis=True, fit_pair_list=pair_list,
        omega_screen=float(omega_screen),
    ))))

    # Scatter: batch all member pairs of one op into a single aux-rotation
    # GEMM, then apply the small per-pair orbital Wigner-D blocks.
    rots_aux = ccm_symmetry_basis_rotations(sym, aux_modrho)
    members_by_op: dict[int, list[tuple[int, int]]] = {}
    for A in range(n):
        for B in range(n):
            g = int(orbits.orbit_op[A, B])
            if g >= 0:
                members_by_op.setdefault(g, []).append((A, B))

    naux = L.shape[0]
    lf = L.reshape(naux, nbf * nbf)
    for g, mems in members_by_op.items():
        op = sym.invariant_ops[g]
        src_cols = []
        for (A, B) in mems:
            ra, rb = orbits.representatives[int(orbits.orbit_rep[A, B])]
            src_cols.append(
                (atom_ao[ra][:, None] * nbf + atom_ao[rb][None, :]).ravel())
        pl = rots_aux[g] @ lf[:, np.concatenate(src_cols)]
        lo = 0
        for (A, B), sc in zip(mems, src_cols):
            ra, rb = orbits.representatives[int(orbits.orbit_rep[A, B])]
            ia, ib = atom_ao[A], atom_ao[B]
            blk = pl[:, lo:lo + sc.size].reshape(naux, ia.size, ib.size)
            lo += sc.size
            # L_{Q,a,b} = S_{mu,ν} P[a,mu] P[b,ν] (P_aux L)_{Q,mu,ν}
            L[:, ia[:, None], ib[None, :]] = np.einsum(
                "Qmn,am,bn->Qab", blk,
                op.P[np.ix_(ia, atom_ao[ra])],
                op.P[np.ix_(ib, atom_ao[rb])],
                optimize=True)
    return L


def _fold_star_reconstruct(l_rep, op, ka, kb, trs, ao_at, aux_at):
    """Reconstruct ``L(R ka, R kb)`` (optionally time-reversed) from ``L(ka, kb)``.

    The algebraic k-pair covariance of the all-FT GDF fit in the canonical
    auxiliary frame, for an infinite or symmetry-closed Bloch cell list
    (probe-pinned 2026-07-11 by a brute-force phase scan on the H2 chain +
    diamond, then derived; finite-list errors 8.8e-13 / 3.8e-10 at a
    converged cell list, non-symmorphic glides included). For a space-group op
    ``g = {R|t}`` with atom permutation ``perm`` and lattice shifts
    ``S_A = L_u @ shift_A`` (``x_perm[A] = R x_A + t + L_u shift_A``),
    orbital / auxiliary AO rotations ``P`` / ``P_aux`` (atom permutation x
    real Wigner-D), and ``q = kb - ka``:

        L(R ka, R kb)[P',u',v'] =
            sum_{P,u,v} (P_aux)[P',P] e^{+i (Rq).S_P}
                        P[u',u]       e^{+i (R ka).S_u}
                        P[v',v]       e^{-i (R kb).S_v}   L(ka, kb)[P,u,v]

    Derivation sketch: substituting the op into the builder's M / T G-sums,
    the rotation maps the G-ball onto itself (|RG| = |G|, R maps the
    reciprocal lattice to itself -- this is why the mesh must be the full
    |G| <= G_max ball, the rsgdf_dense_g_mesh clipping fix), the Coulomb
    kernel and per-shell scales are invariant, and each Gaussian FT obeys
    F(Rp) = D e^{-i(Rp).t} e^{i(Rp).T_A} F(p) with T_A = -S_A. The
    G-dependent and the non-symmorphic e^{i(Rp).t} phases cancel between
    the aux CONJUGATE and the pair factor of both M and T (which is why no
    global tau phase appears above), leaving the three per-atom phases; at
    Gamma all of them collapse to 1 (the M3 Gamma covariance). The metric
    transforms as M(Rq) = Omega M(q) Omega^dagger with the same unitary
    Omega = P_aux diag(e^{+i(Rq).S_P}), so the whitening carries through
    as M(Rq)^{-1/2} = Omega M(q)^{-1/2} Omega^dagger and the identity
    holds for the final canonical-frame cderi, not just the raw 3c tensor. A
    finite origin-centred Bloch cell list is not generally closed under the
    atom-dependent shifts used in this reindexing; that is the vibeqc#337
    boundary, rather than an error in the identity above. The planner keeps
    this function on the closed side of it by admitting only ops whose
    per-atom shifts agree
    (:func:`~vibeqc.periodic.ccm.symmetry.ccm_symmetry_op_preserves_cell_list`).

    Time reversal composes as ``L(-k1, -k2) = conj(L(k1, k2))`` (exact,
    1e-14 class: real AOs conjugate every FT ingredient under momentum
    negation and the G-ball is inversion-symmetric).
    """
    R, _perm, s_cart, p_ao, p_aux = op
    kap, kbp = R @ np.asarray(ka, float), R @ np.asarray(kb, float)
    qp = kbp - kap
    ph_a = np.exp(1j * (s_cart[ao_at] @ kap))
    ph_b = np.exp(-1j * (s_cart[ao_at] @ kbp))
    ph_q = np.exp(1j * (s_cart[aux_at] @ qp))
    out = np.einsum(
        "QP,Pmn,am,bn->Qab",
        p_aux * ph_q[None, :], l_rep,
        p_ao * ph_a[None, :], p_ao * ph_b[None, :],
        optimize=True)
    return np.conj(out) if trs else out


def _ao_atom_indices(basis) -> np.ndarray:
    """Per-AO atom index of a (pure solid-harmonic) BasisSet."""
    out: list[int] = []
    for sh in basis.shells():
        out += [int(sh.atom_index)] * (2 * int(sh.l) + 1)
    return np.asarray(out, dtype=int)


def ccm_neutral_tail_ke_cutoff(
    ccm, *, ke_cutoff: float = 200.0, tail_ke_cutoff=None
):
    """Resolve the RSGDF high-``|G|`` tail the neutral cderi builders must use.

    **This is the Hamiltonian-identity seam between the two neutral controls**
    (GitLab IID 307). The neutral cderi builders and the multi-k GDF sibling
    (:func:`~vibeqc.periodic.ccm.ri.run_ccm_rhf_gdf`) are documented as
    evaluating the *same* block-circulant Hamiltonian, so they must resolve the
    same tail. Before this resolver the direct/fold route could not request a
    tail at all, while ``run_krhf_periodic_gdf`` at a Γ-only mesh delegates to
    :func:`~vibeqc.pbc_gdf.run_pbc_gdf_rhf`, which auto-sizes one -- so the two
    routes silently evaluated different Hamiltonians and disagreed by
    **-4.99e-01 Ha/cell** on MgO fcc (a = 7.956 bohr) / STO-3G at
    ``nrep=(1,1,1)``, with the tailed side the accurate one (0.3 mHa from the
    PySCF MDF reference -271.0499 Ha; the untailed side carries the documented
    dense-core P01 offset).

    The rule mirrors the sibling route exactly:

    * an explicit ``tail_ke_cutoff`` overrides the classifier (``0`` opts out,
      deliberately keeping the parity hold for fast diagnostic runs);
    * at ``N_c = 1`` the sibling takes the Γ fast path, which auto-sizes the
      tail via :func:`~vibeqc.pbc_gdf._auto_rsgdf_tail_ke_cutoff` -- so this
      returns the same classifier's answer on the same (unit system, AO basis,
      ``ke_cutoff``) inputs. ``ke_cutoff`` *is* the sibling's
      ``rsgdf_ke_cutoff`` (both default ``200.0``, and both reach the builder
      as its ``ke_cutoff``), so the mirror is like-for-like;
    * at ``N_c > 1`` the sibling takes the multi-k loop, which deliberately
      does **not** auto-tail (``_warn_multik_dense_core_gdf_parity_hold``
      records the hold and calls multi-k auto-tailing an open maintainer call),
      so neither does this. Both routes stay on the held-but-agreeing side
      rather than one of them moving unilaterally.

    Returns ``None`` when no tail applies -- the builders' "base mesh only"
    value, and the truthful provenance to record on a result.

    **A request the builder would ignore is normalised to ``None``.** The
    builders complete the tail only when ``tail_ke_cutoff > ke_cutoff``
    (``aux_basis.py``), so ``0`` -- and any value at or below the base mesh --
    means "no tail". Collapsing those to ``None`` here buys two things:
    a caller opting out is byte-for-byte the historical call, including at the
    handful of downstream sites where ``0.0`` and ``None`` are *not*
    interchangeable (``_rsgdf_weighted_3c_tensor_gradient_bloch`` rejects any
    non-``None`` tail alongside external kernel weights); and the value
    recorded on a result can never claim a tail the builder did not apply,
    which is the reporting failure this whole issue is about.
    """
    if tail_ke_cutoff is not None:
        return _tail_above_base_mesh(float(tail_ke_cutoff), float(ke_cutoff))
    if int(np.prod(np.asarray(ccm.nrep, dtype=int))) != 1:
        return None
    from vibeqc._vibeqc_core import BasisSet
    from vibeqc.pbc_gdf import _auto_rsgdf_tail_ke_cutoff

    unit = ccm.unit_system
    ubasis = BasisSet(unit.unit_cell_molecule(), ccm.basis_name)
    resolved = _auto_rsgdf_tail_ke_cutoff(
        unit, "rsgdf", ubasis, None, rsgdf_ke_cutoff=float(ke_cutoff)
    )
    if resolved is None:
        return None
    return _tail_above_base_mesh(float(resolved), float(ke_cutoff))


def _tail_above_base_mesh(tail: float, ke_cutoff: float):
    """``tail`` if the builders would actually sweep it, else ``None``."""
    return tail if tail > ke_cutoff else None


def ccm_neutral_cderi_fold(
    ccm, *, ke_cutoff: float = 200.0, tail_ke_cutoff=None, aux_basis=None,
    lat_opts=None, symmetry=None, omega_screen: float = 0.0,
    fold_threads=None,
) -> np.ndarray:
    """Real Γ-supercell neutral cderi from **per-q unit-cell fits** (the fold).

    Same object as :func:`ccm_neutral_cderi` -- a real ``(n_aux_sc, nbf_sc,
    nbf_sc)`` tensor whose self-contraction is the neutral torus four-center
    ``(muν|v_E|rs)`` -- built without ever running an FFT fit on the supercell.
    The supercell fit's memory wall (the dense supercell G-mesh: measured
    **49 GB** for a compact H₂ (3,3,3) torus, 54 AOs, 18³ bohr) is replaced by
    ``N_c × n_q`` *unit-cell* Bloch-pair fits (the multi-k GDF's own builder
    and cost), folded onto the torus:

        # Fold identity (this module, 2026-07-02; the RI-level finite Fourier
        # representation of this specified neutral-control Hamiltonian, using
        # periodic GDF factorisation per
        # Sun, Berkelbach, McClain & Chan, J. Chem. Phys. 147, 164119 (2017),
        # doi:10.1063/1.4998644):
        #   B_q[P,(mu c),(ν c')] = N_c^{-3/2} · Σ_{ka∈mesh}
        #       e^{+i ka·R_c − i kb·R_c'} · L_{ka,kb}[P,mu,ν],  kb = ka + q
        #   (mu c, ν c' | v_E | ρ d, σ d')
        #       = Σ_q Σ_P  B_q[P, mu c, ν c'] · conj(B_q[P, σ d', ρ d])
        # Torus ERI is real with q ↔ −q conjugate channels, so the real cderi
        # is the ±q-combined stack {w·Re B_q, w·Im B_q} (w = 1 for self-paired
        # labels, √2 for one representative of each ±q pair), null rows
        # pruned. Verified: max|g_fold − g_supercell| ≈ 1e-10 on compact H₂
        # (2,1,1)/(3,1,1)/(2,2,2) toruses at ke_cutoff=200.

    .. warning::

       **Frame consistency requires the UNFOLDED ket momentum.** All ``ka`` in
       one q-channel must be built at the *same cartesian* ``kb = ka + q``
       arithmetic (never mod-folded back into the first BZ): the builder's
       auxiliary whitening depends on the cartesian momentum transfer, so
       mod-folding gives fits in *different aux frames* (even different
       pruned ranks -- LiH: 62 vs 59) and the cross-``ka`` products of the
       fold identity are then silently wrong (measured 1.9e-3 on H₂ (2,2,2),
       0.146 Ha on LiH). The multi-k GDF never crosses frames (each ``L`` is
       contracted with itself only), which is why this constraint is
       invisible there.

    Notes
    -----
    * **Not elementwise equal** to :func:`ccm_neutral_cderi` -- an RI cderi is
      defined up to an orthogonal rotation of the auxiliary index; the two
      builders agree on every contraction (``g_eff``, J/K, B-tensors), which
      is what all consumers use.
    * **More accurate than the supercell fit on ultra-diffuse bases.** On
      LiH/sto-3g (2,1,1) the supercell-FFT cderi carries a cutoff-resistant
      ~1e-4 torus-translation asymmetry (open finding, 2026-07-02) and the
      direct SCF misses the multi-k GDF by −1.35e-4 Ha/cell; with the fold
      cderi the same SCF matches the GDF to **4e-14** and J/K commute with
      the torus translation to 1e-16 -- frame-consistent per-q fits are the
      GDF's own objects, so agreement is by construction.
    * The unit-cell fits need **no cross-torus wrap widening** (pairs live in
      the unit cell; Bloch phases carry the torus) -- ``lat_opts`` defaults to
      the plain builder options, exactly as the multi-k GDF driver uses.
    * Cost: ``N_c·(n_q_kept)`` unit-cell ``build_lpq_bloch_native_fft`` calls
      (≈ the multi-k GDF's exchange-cache setup); memory ``O(N_c·n_aux·nbf_sc²)``
      -- MBs where the supercell fit needs GBs (measured: 49 GB → tens of MB
      on the (3,3,3) torus).

    ``symmetry`` routes the per-``(k_a, k_b)`` builder calls through the
    space-group + time-reversal star reduction
    (:func:`~vibeqc.periodic.ccm.symmetry.ccm_symmetry_fold_kpair_plan`):
    only orbit-representative momentum pairs are fit, the rest are
    reconstructed by k-pair covariance. The planner admits a space-group
    relation only when its op also maps the finite Bloch cell list onto itself
    (:func:`~vibeqc.periodic.ccm.symmetry.ccm_symmetry_op_preserves_cell_list`),
    which is the vibeqc#337 fix: the reconstruction is then exact on the
    truncated build, not merely convergent to it, so the reduction no longer
    depends on the replica mesh's dimensionality.
    Accepts ``None``/``False`` (off, the default), ``True``/"auto"
    (analyze the cluster), or a pre-computed
    :class:`~vibeqc.periodic.ccm.symmetry.CCMSymmetry`. Only relations admitted
    by the current conservative plan are used: a rotated channel must coincide
    with a kept channel exactly in cartesian coordinates.
    The shared builder now constructs one physical ``|G+q|`` support, so
    mod-``G`` channel relations are also operator-exact; extending
    the optional star inventory to exploit those additional relations is a
    separate performance step. Small (2,1,1)-class meshes therefore still
    reduce nothing in the current plan, while the existing wins start at
    (2,2,1)/(3,1,1)-class meshes. The builder-call reduction is logged either
    way; on enabled meshes the difference from the un-reduced fold is bounded
    by the cell-list/whitening floors pinned in the parity tests.

    ``tail_ke_cutoff`` (default ``None``) completes the RSGDF fit over the
    high-``|G+q|`` shell above ``ke_cutoff``. ``None`` resolves it through
    :func:`ccm_neutral_tail_ke_cutoff`, which mirrors the multi-k GDF
    sibling's decision so the two controls evaluate the same Hamiltonian
    (GitLab IID 307); pass an explicit value (``0`` disables) to override.

    ``omega_screen`` (experimental, default ``0.0`` = full Coulomb,
    bit-identical historical build): fold the **screened erfc-SR kernel**
    instead, per the :func:`ccm_neutral_cderi` contract. The attenuation is
    a function of ``|G+q|`` on the *unfolded supercell* reciprocal set
    ``{G_unit + q}``, so the fold identity and the k-pair star reduction
    carry over unchanged; the dropped supercell ``p = 0`` mode lives in the
    ``q = 0`` channel's ``G = 0`` term with weight ``w(0)/(N_c V_unit) =
    w(0)/V_sc`` -- the same rank-1 ``sigma_0 S⊗S`` restoration as the
    supercell build (:func:`ccm_sr_exchange_zero_mode_constant`). Screened
    folds are exchange-only objects (never ``J``).

    ``fold_threads`` parallelizes the per-``(k_a, k_b)`` fits of each
    q-channel -- the CCM analogue of k-point parallelism (CLAUDE.md § 15).
    The fits are independent and every C++ primitive they call releases the
    GIL, so a thread pool scales; star MEMBERS are still reconstructed after
    their channel's builds complete, preserving the serial sweep's
    representative-before-member ordering exactly. ``None`` (default) reads
    ``VIBEQC_CCM_FOLD_THREADS`` and otherwise stays serial; ``0`` means all
    cores. **Pin BLAS first** (``OMP_NUM_THREADS=1``): the fits are
    BLAS-heavy, so threading on top of a multi-threaded BLAS oversubscribes
    and can be slower than serial. Output is unchanged elementwise either
    way (same operations, same order of accumulation per channel).
    """
    from vibeqc import LatticeSumOptions
    from vibeqc.aux_basis import (
        build_lpq_bloch_native_fft,
        default_aux_for,
        make_aux_basis_set,
        make_modrho_aux_basis,
    )
    from vibeqc.periodic_k_gdf import _kmesh_to_kpoints_weights

    unit = ccm.unit_system
    mol = unit.unit_cell_molecule()
    from vibeqc._vibeqc_core import BasisSet

    ubasis = BasisSet(mol, ccm.basis_name)
    aux_name = aux_basis or default_aux_for(ccm.basis_name)
    aux = make_aux_basis_set(mol, aux_name=aux_name, drop_eta=0.0)
    aux_modrho = make_modrho_aux_basis(aux, mol)
    opts = lat_opts if lat_opts is not None else LatticeSumOptions()
    # Hamiltonian-identity seam with the multi-k GDF sibling (IID 307) --
    # see ccm_neutral_tail_ke_cutoff for the rule and the measured offset.
    tail = ccm_neutral_tail_ke_cutoff(
        ccm, ke_cutoff=ke_cutoff, tail_ke_cutoff=tail_ke_cutoff
    )
    if tail is not None:
        write(
            f"  ccm_neutral_cderi_fold: RSGDF high-|G| tail cutoff "
            f"{float(tail):g} Ha (base mesh {float(ke_cutoff):g} Ha)\n",
            Level.VERBOSE)

    kpts, _ = _kmesh_to_kpoints_weights(unit, list(ccm.nrep))
    kpts = np.asarray(kpts, dtype=float)
    nrep = np.asarray(ccm.nrep, dtype=int)
    n_c = int(np.prod(nrep))
    if len(kpts) != n_c:
        raise ValueError(
            f"ccm_neutral_cderi_fold: mesh size {len(kpts)} != N_c {n_c}"
        )
    n_mu = int(ubasis.nbasis)
    a_lat = np.asarray(unit.lattice, dtype=float)  # columns = lattice vectors
    b_lat = 2.0 * np.pi * np.linalg.inv(a_lat).T   # columns = reciprocal vectors

    # Mesh bookkeeping: integer (fractional·nrep) labels, mod-folded -- used
    # only to pair q with −q; the builder always gets unfolded cartesian k's.
    frac = np.round(np.linalg.solve(b_lat, kpts.T).T * nrep[None, :]).astype(int)
    frac %= nrep

    # Supercell cell list in CCMSystem ordering ((i·n2+j)·n3+k), cartesian.
    cells = np.array(
        [(i, j, k) for i in range(nrep[0]) for j in range(nrep[1])
         for k in range(nrep[2])], dtype=int)
    r_c = cells @ a_lat.T

    # Kept channels: one representative per ±q pair, in fold order.
    kept: list[int] = []
    seen: set = set()
    for qi in range(n_c):
        qf = tuple(frac[qi])
        if qf in seen:
            continue
        kept.append(qi)
        seen.add(qf)
        seen.add(tuple((-frac[qi]) % nrep))

    # Optional space-group + TRS star reduction of the builder calls.
    plan = None
    if symmetry is not None and symmetry is not False:
        from .symmetry import analyze_ccm_symmetry, ccm_symmetry_fold_kpair_plan

        sym = (symmetry if not isinstance(symmetry, (bool, str))
               else analyze_ccm_symmetry(ccm))
        plan = ccm_symmetry_fold_kpair_plan(
            ccm, sym, kpts, kept, ubasis, aux_modrho)
        if plan.n_reps < plan.n_builds:
            write(
                f"  ccm_neutral_cderi_fold: k-pair star reduction "
                f"|G_c| = {sym.cluster_order} (+TRS): {plan.n_reps}/"
                f"{plan.n_builds} builder calls "
                f"({plan.reduction_factor:.2f}x)\n",
                Level.VERBOSE)
        else:
            write(
                f"  ccm_neutral_cderi_fold: no k-pair star reduction "
                f"available (|G_c| = {sym.cluster_order}, {plan.n_reps}/"
                f"{plan.n_builds} builds) -- full build\n",
                Level.VERBOSE)
        ao_at = _ao_atom_indices(ubasis)
        aux_at = _ao_atom_indices(aux_modrho)
        # Cache only representatives that other builds reconstruct from.
        referenced = {e[1] for e in plan.entries.values() if e[0] == "recon"}
        rep_cache: dict = {}

    alpha = n_c ** -1.5
    n_sc = n_c * n_mu
    n_threads = _fold_thread_count(fold_threads, n_c)
    blocks: list[np.ndarray] = []
    for qi in kept:
        qf = tuple(frac[qi])
        mqf = tuple((-frac[qi]) % nrep)
        acc = None

        # ---- WSC/q-axis parallelism -------------------------------------
        # The per-(ka, kb) fits of one q-channel are independent, and every
        # C++ primitive they call releases the GIL, so a thread pool over
        # this axis is the CCM analogue of k-point parallelism (CLAUDE.md
        # § 15). Builds go first and as a batch: a star MEMBER may
        # reconstruct from a representative in this same channel, and the
        # serial sweep's "representatives always precede members" ordering
        # is only guaranteed once every build of the channel is done.
        # Representatives in EARLIER channels are already in rep_cache.
        build_ais = [
            ai for ai in range(n_c)
            if plan is None or plan.entries[(qi, ai)][0] == "build"
        ]

        def _build_one(ai, _qi=qi):
            return ai, np.asarray(build_lpq_bloch_native_fft(
                unit, ubasis, aux_modrho, kpts[ai], kpts[ai] + kpts[_qi],
                ke_cutoff=float(ke_cutoff), tail_ke_cutoff=tail,
                lat_opts=opts,
                canonical_auxiliary_basis=True,
                omega_screen=float(omega_screen)))

        built: dict = {}
        if n_threads > 1 and len(build_ais) > 1:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=n_threads) as pool:
                for ai, l_ab in pool.map(_build_one, build_ais):
                    built[ai] = l_ab
        else:
            for ai in build_ais:
                built[ai] = _build_one(ai)[1]
        if plan is not None:
            for ai, l_ab in built.items():
                if (qi, ai) in referenced:
                    rep_cache[(qi, ai)] = l_ab

        for ai in range(n_c):
            # UNFOLDED cartesian ket momentum -- one whitening frame per
            # q-channel (see the docstring warning; folding kb into the BZ
            # breaks the cross-ka products of the fold identity).
            kb_cart = kpts[ai] + kpts[qi]
            entry = plan.entries[(qi, ai)] if plan is not None else ("build",)
            if entry[0] == "build":
                # canonical_auxiliary_basis=True: the fit is returned in the
                # fixed aux-AO frame (matrix-function M(q)^{-1/2},
                # gauge-stable) instead of the compact eigenmode frame, whose
                # arbitrary eigenvector phases/rotations differ between
                # independently built blocks -- the builder documents it as
                # required exactly for cross-block contractions like this
                # fold (measured without it: 8.5e-4 channel error on a vacuum
                # cell from FP q-jitter across ka). It is also what makes the
                # star reconstruction below frame-stable.
                l_ab = built[ai]
            else:
                # Star member: covariance reconstruction from the cached orbit
                # representative (identity + measured finite-list floors at
                # _fold_star_reconstruct). Representatives always precede
                # members in this (kept q, ai) sweep by construction. The
                # reconstructed L sits at momenta (R ka, R kb) which differ
                # from (kpts[ai], kb_cart) by a COMMON reciprocal shift G0;
                # the builder is exactly G0-periodic in both momenta and the
                # fold phases below see G0.r_c in 2*pi*Z, so no correction
                # appears here.
                _, (rqi, rai), oi, trs = entry
                l_ab = _fold_star_reconstruct(
                    rep_cache[(rqi, rai)], plan.ops[oi],
                    kpts[rai], kpts[rai] + kpts[rqi], trs, ao_at, aux_at)
            if acc is None:
                acc = np.zeros((l_ab.shape[0], n_sc, n_sc), dtype=complex)
            pha = np.exp(+1j * (r_c @ kpts[ai]))
            phb = np.exp(-1j * (r_c @ kb_cart))
            # Block-structured scatter, vectorized: the (c, d) double loop
            # is exactly acc += kron(outer(pha, phb), l_ab) on the torus
            # cell blocks. The Python form cost N_c^2 tiny GEMMs per builder
            # call (9.4M iterations at (4,4,4)); this is one einsum over the
            # same disjoint blocks, so it is arithmetically identical.
            acc.reshape(-1, n_c, n_mu, n_c, n_mu)[...] += np.einsum(
                "c,d,Amn->Acmdn", pha, phb, l_ab, optimize=True)
        acc *= alpha
        # {Re, Im} stack -- weight 1 for a self-paired (q ≡ −q mod G) label
        # (the channel enters the q-sum once; no realness assumption needed),
        # √2 for one representative of a genuine ±q pair.
        w = 1.0 if qf == mqf else np.sqrt(2.0)
        blocks.append(w * np.ascontiguousarray(acc.real))
        blocks.append(w * np.ascontiguousarray(acc.imag))
    out = np.concatenate(blocks, axis=0)
    # Prune numerically-null aux rows (exact-zero Im blocks of real channels).
    keep = np.einsum("Amn,Amn->A", out, out) > 1e-14
    return np.ascontiguousarray(out[keep])


def ccm_eri_neutral(ccm, *, ke_cutoff: float = 200.0, aux_basis=None) -> np.ndarray:
    """Neutral (Ewald) effective four-center ``g_eff[muν,rs] = (muν|v_E|rs)`` (Sec.13.7).

    The RI / density fit of the neutral torus Coulomb ``v_E`` over the home
    (supercell) AOs: ``g_eff[mu,ν,r,s] = S_P L[P,muν].L[P,rs]`` with ``L`` the
    neutral cderi (:func:`ccm_neutral_cderi`). The kernel is charge-neutral (the
    ``G+q=0`` mode is dropped), so there is no Madelung self-image leak for any
    lattice. The result contracts like ordinary chemists' ERIs
    (``J_muν = S_rs D_rs g[mu,ν,r,s]``, ``K_muν = S_rs D_rs g[mu,s,r,ν]``) and is
    **exactly 8-fold permutationally symmetric**.

    Parameters
    ----------
    ccm : CCMSystem
        The cyclic cluster (its ``cluster_system`` supercell is fitted at Γ).
    ke_cutoff : float
        Plane-wave kinetic-energy cutoff (Ha) for the all-FT GDF fit. 200 is
        converged for STO-3G valence cells (the cderi is cutoff-stable; the
        residual vs the exact neutral ERI is the auxiliary RI error).
    aux_basis : str, optional
        Auxiliary (fitting) basis name; defaults to
        ``default_aux_for(ccm.basis.name)``.

    Returns
    -------
    numpy.ndarray
        ``(nbf, nbf, nbf, nbf)`` neutral effective four-center.

    Notes
    -----
    Dense ``nbf**4`` from a Γ cderi -- a small-cluster analysis/validation tool
    (it OOMs on dense ionic 3-D, like the padded bare four-center). The
    scalable neutral-control HF/KS *energy* is :func:`~vibeqc.periodic.ccm.ri.run_ccm_rhf_gdf`
    / :func:`~vibeqc.periodic.ccm.ri.run_ccm_rks_gdf`. See the module docstring
    for the route-gap decomposition and its fixed-reference scope.
    """
    lpq = ccm_neutral_cderi(ccm, ke_cutoff=ke_cutoff, aux_basis=aux_basis)
    g_eff = np.einsum("Pmn,Prs->mnrs", lpq, lpq, optimize=True)
    # 8-fold symmetric by construction (L_{P,muν}=L_{P,νmu}); symmetrise to clean
    # the last ULP of FP accumulation noise so downstream asserts see exact.
    g_eff = 0.5 * (g_eff + np.transpose(g_eff, (1, 0, 3, 2)))
    g_eff = 0.5 * (g_eff + np.transpose(g_eff, (2, 3, 0, 1)))
    return g_eff
