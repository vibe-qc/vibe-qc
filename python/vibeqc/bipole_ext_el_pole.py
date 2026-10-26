"""BIPOLE cell-density Fourier and Ewald diagnostic helpers.

The AO-pair Fourier infrastructure in this module evaluates the positive
electron-number density ``rho_hat(K)``.  Its zero mode obeys
``rho_hat(0) = sum_Rmn P_mn(R) S_mn(R) = N_e``; nuclei are not inputs
here.

The reciprocal electronic Ewald component

``(1/2) (4 pi / V) sum_{K != 0} |rho_hat(K)|^2
exp(-K^2 / 4 alpha^2) / K^2``

is available as :func:`compute_electronic_reciprocal_ewald_energy` for
diagnostics.  It is the non-negative, alpha-dependent ``K != 0`` reciprocal
electron-electron Ewald contribution, not CRYSTAL's printed
``EXT EL-POLE`` term.  Saunders et al., Molecular Physics 77, 629 (1992),
Eqs. 116-130, define that term through shell-partitioned multipoles and an
explicit penetration-zone subtraction.  The former implementation defined
neither the multipole order nor the penetration criterion, so
:func:`compute_ext_el_pole_reciprocal_sum` now fails closed instead of
returning a different Ewald contribution under the CRYSTAL label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Never, Optional

import numpy as np

from ._aopair_ft import (
    ao_pair_fourier_transform,
    ao_pair_fourier_transform_at_cells,
)
from ._primitive_norm import libint_primitive_norm as _libint_primitive_norm
from ._vibeqc_core import (
    BasisSet,
    LatticeMatrixSet,
    LatticeSumOptions,
    PeriodicSystem,
    ShellInfo,
)


def _libint_ylm_correction_per_ao(basis: BasisSet) -> np.ndarray:
    """Per-AO correction factor √(4pi/(2l+1)) reconciling
    :func:`ao_pair_fourier_transform`'s convention with libint's
    Y_lm normalisation.

    libint absorbs Y_00 = 1/√(4pi) into the s-shell contraction
    coefficient (so s-shell norm is independent of any 4pi factor),
    but does NOT absorb Y_lm for l > 0 -- those AOs carry an explicit
    Y_lm factor that differs from the "raw" spherical-harmonic
    convention used inside ``cart_to_sph_matrix`` by √(4pi/(2l+1)).

    Empirically verified on MgO STO-3G: at G = 0 the FT diagonal
    equals 1.0 for s-orbital AOs (correct) but 3/(4pi) ≈ 0.239 for
    p-orbital AOs (off by 4pi/3). The per-AO correction recovers
    the libint convention so ``S_mu FT_mumu(0) = N_electrons``.
    """
    factors = np.empty(basis.nbasis, dtype=float)
    bf_offset = 0
    for shell in basis.shells():
        l = int(shell.l)
        n_ao_in_shell = 2 * l + 1
        if l == 0:
            factor = 1.0
        else:
            factor = math.sqrt(4.0 * math.pi / (2 * l + 1))
        factors[bf_offset : bf_offset + n_ao_in_shell] = factor
        bf_offset += n_ao_in_shell
    if bf_offset != basis.nbasis:
        raise RuntimeError(
            f"_libint_ylm_correction_per_ao: walked {bf_offset} AOs but "
            f"basis.nbasis = {basis.nbasis}; basis layout mismatch."
        )
    return factors


__all__ = [
    "ElectronicReciprocalEwaldResult",
    "EwaldReciprocalSumResult",
    "compute_reciprocal_lattice_vectors",
    "compute_cell_density_fourier",
    "compute_cell_density_fourier_lattice",
    "compute_electronic_reciprocal_ewald_energy",
    "compute_ext_el_pole_reciprocal_sum",
    "compute_ext_el_spheropole",
    "bipole_ewald_reciprocal_cutoff",
    "crystal_default_ewald_alpha",
    "crystal_ewald_reciprocal_cutoff",
]


def _ao_origins(basis: BasisSet) -> np.ndarray:
    """Per-AO Cartesian origin (bohr) -- shape ``(nbasis, 3)``.

    All AOs in a single shell share the shell's origin (the atom
    centre); this just expands ``shell.origin`` along the (2l+1)
    spherical-harmonic AOs of each shell.
    """
    origins = np.zeros((basis.nbasis, 3), dtype=float)
    ao_idx = 0
    for shell in basis.shells():
        l = int(shell.l)
        n_ao_in_shell = 2 * l + 1
        origins[ao_idx : ao_idx + n_ao_in_shell, :] = np.asarray(
            shell.origin, dtype=float
        )
        ao_idx += n_ao_in_shell
    if ao_idx != basis.nbasis:
        raise RuntimeError(
            f"_ao_origins: walked {ao_idx} AOs but basis.nbasis = "
            f"{basis.nbasis}; basis layout mismatch."
        )
    return origins


def _split_basis_into_primitives(
    basis: BasisSet,
    system: PeriodicSystem,
) -> tuple[BasisSet, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split contracted basis into a primitive-per-shell basis.

    Each original shell with N primitives becomes N child shells,
    each with a single primitive carrying the libint-normalised
    coefficient of that primitive. The split preserves libint's
    integral conventions:

        chi_mu^orig  =  S_a (c_a^orig) . raw_primitive_a
                  =  S_a child_shell_a

    so any contracted libint integral equals the sum over child-
    shell pairs of the corresponding primitive integrals.

    Returns
    -------
    prim_basis : BasisSet
        Primitive-split basis. ``prim_basis.nbasis = S_s N_s.(2l_s+1)``.
    prim_alpha : ndarray (n_prim_AOs,)
        Per-primitive-AO Gaussian exponent a (bohr⁻^2).
    prim_origins : ndarray (n_prim_AOs, 3)
        Per-primitive-AO Cartesian origin (bohr).
    prim_N_lib : ndarray (n_prim_AOs,)
        Per-primitive-AO libint normalisation factor; needed to
        recover the raw (un-libint-normalised) contraction
        coefficient ``d^std = c^lib / N_lib`` from
        ``BasisSet.coefficients`` so CRYSTAL's primitive integrand
        (which uses ``d^std`` directly) can be reconstructed.
    contraction_map : ndarray (nbasis_orig, n_prim_AOs)
        Boolean matrix: ``contraction_map[mu_orig, ν_prim] = 1`` if
        primitive AO ``ν_prim`` is a constituent of the contracted
        AO ``mu_orig``.
    """
    prim_shells: list[ShellInfo] = []
    prim_alpha_list: list[float] = []
    prim_l_list: list[int] = []
    prim_origin_list: list[tuple[float, float, float]] = []
    orig_ao_to_prim_aos: list[list[int]] = []

    orig_ao_idx = 0
    prim_ao_idx = 0
    for shell in basis.shells():
        l = int(shell.l)
        n_ao_in_shell = 2 * l + 1
        exponents = list(shell.exponents)
        coefficients = list(shell.coefficients)
        atom_index = int(shell.atom_index)
        pure = bool(shell.pure)
        origin = tuple(shell.origin)

        per_orig_ao_prim_list: list[list[int]] = [[] for _ in range(n_ao_in_shell)]
        for a_idx, (alpha, coef) in enumerate(zip(exponents, coefficients)):
            child = ShellInfo(
                atom_index=atom_index,
                l=l,
                pure=pure,
                exponents=[float(alpha)],
                coefficients=[float(coef)],
                origin=origin,
            )
            prim_shells.append(child)
            for m in range(n_ao_in_shell):
                prim_alpha_list.append(float(alpha))
                prim_l_list.append(l)
                prim_origin_list.append(origin)
                per_orig_ao_prim_list[m].append(prim_ao_idx)
                prim_ao_idx += 1

        for m in range(n_ao_in_shell):
            orig_ao_to_prim_aos.append(per_orig_ao_prim_list[m])
            orig_ao_idx += 1

    prim_basis = BasisSet(
        system.unit_cell_molecule(),
        prim_shells,
        "<spheropole-primitive-split>",
        True,
    )
    n_prim_AOs = int(prim_basis.nbasis)
    prim_alpha = np.asarray(prim_alpha_list, dtype=float)
    prim_origins = np.asarray(prim_origin_list, dtype=float)
    prim_N_lib = np.array(
        [
            _libint_primitive_norm(float(prim_alpha_list[i]), int(prim_l_list[i]))
            for i in range(n_prim_AOs)
        ],
        dtype=float,
    )

    nbf_orig = int(basis.nbasis)
    contraction_map = np.zeros((nbf_orig, n_prim_AOs), dtype=float)
    for mu_orig, prim_list in enumerate(orig_ao_to_prim_aos):
        for ν_prim in prim_list:
            contraction_map[mu_orig, ν_prim] = 1.0
    return prim_basis, prim_alpha, prim_origins, prim_N_lib, contraction_map


def compute_ext_el_spheropole(
    P_real: LatticeMatrixSet,
    basis: BasisSet,
    system: PeriodicSystem,
    lat_opts: LatticeSumOptions,
    *,
    n_electrons: Optional[int] = None,
) -> float:
    """Compute CRYSTAL's ``::: EXT EL-SPHEROPOLE`` energy correction.

    CRYSTAL prints this term as a separate K=0 spheropole-coupling
    contribution to ``TOTAL E-E`` in the per-cycle ENECYCLE block
    (e.g. MgO/STO-3G CYC 0 -> ``EXT EL-SPHEROPOLE = +4.119 Ha``). It
    is the K=0 limit of the Ewald reciprocal-space sum, evaluated as
    a real-space integral with a custom per-primitive-pair kernel.

    Implementation (v3, 2026-06-01) -- exact bond-symmetrised 2nd moment
    --------------------------------------------------------------------
    The per-pair spheropole kernel is exactly the **bond-symmetrised
    second moment** of the AO pair about its two atom centres::

        K_{muν}(g) = < phi_mu | (r - A)^2 + (r - B - g)^2 | phi_ν >

    (the s-s case is verified against CRYSTAL's printed spheropole). It is
    computed directly -- for arbitrary (l_a, l_b), angular factors
    included -- from libint ``emultipole2`` via
    ``compute_multipole_moments_lattice`` (the same lattice machinery the
    multipole far-field uses), so no per-(l_a, l_b) reconstruction is
    needed and it is exact for s/p/d/f. About a single origin O the
    Cartesian moments give

        <(r - C)^2> = Tr<r^2> - 2.(C - O).<r> + |C - O|^2.<1>

    evaluated for C = A (bra centre) and C = B + g (ket centre), with
    O = 0.

    This supersedes -- and as of 2026-06-01 has fully replaced -- the earlier
    per-(l_a, l_b) spheropole C++ kernel, whose p-p term carried an empirical
    fudge factor and whose d/f shells fell back to the s-s formula (it
    diverged ~0.64 Ha from this exact form at the converged MgO density).
    The earlier ``cpp/src/spheropole.cpp`` (``compute_spheropole_{bare_integrals,
    kernel_lattice,kernel_grad_lattice}``) was removed in the same change.

    Prefactor
    ---------
    ``E = (pi . N_e / (6 . V)) . S_g S_{muν} P_{muν}(g) . K_{muν}(g)``; the
    ``pi/V`` factor with the ``N_e/6`` charge moment is the K->0
    (reciprocal-space) limit of the Ewald sum -- the "spheropole"
    correction of the Gaussian-charge electrostatic series (Saunders,
    Freyria-Fava, Dovesi, Salasco & Roetti, Mol. Phys. 77, 629 (1992)).
    Validated: this reproduces CRYSTAL's MgO/STO-3G spheropole
    (+4.1191890 Ha, CYC 0) to < 0.02 mHa -- vs ~7 mHa for the old
    approximate p-p kernel.

    Energy-only correction
    ----------------------
    CRYSTAL adds this to the energy decomposition (printed
    separately) but NOT to the Fock matrix -- vibe-qc follows this
    convention: ``compute_ext_el_spheropole`` is a post-SCF correction
    added to ``E_total``, with no derivative term entering F^2e.

    Parameters
    ----------
    P_real : LatticeMatrixSet
        Real-space density matrix per lattice cell.
    basis : BasisSet
        AO basis (original contracted basis).
    system : PeriodicSystem
        3D periodic system (raises for dim != 3).
    lat_opts : LatticeSumOptions
        Cutoff options for the lattice-sum integral build.
    n_electrons : int, optional
        Total electrons per cell. If None, taken
        from ``system.n_electrons()``.

    Returns
    -------
    float
        ``EXT EL-SPHEROPOLE`` value in Hartree.
    """
    if system.dim != 3:
        raise ValueError(
            f"compute_ext_el_spheropole requires dim=3; got dim={system.dim}"
        )
    a = np.asarray(system.lattice, dtype=float)
    V = float(abs(np.linalg.det(a)))
    if V <= 0.0:
        raise ValueError(f"degenerate lattice (V_cell = {V}); check system.lattice")
    if n_electrons is None:
        n_electrons = int(system.n_electrons())

    orig_nbf = int(basis.nbasis)
    if P_real.nbf != orig_nbf:
        raise ValueError(f"P_real.nbf={P_real.nbf} != basis.nbasis={orig_nbf}")

    # Charge moment N_e / 6 with the pi/V Ewald reciprocal K->0 coefficient
    # -- the spheropole correction of the Gaussian-charge electrostatic
    # series (Saunders et al., Mol. Phys. 77, 629 (1992)). Pinned by output
    # parity against CRYSTAL's sealed MgO value (< 0.02 mHa).
    totchr = float(n_electrons) / 6.0
    prefactor = math.pi * totchr / V

    # Bond-symmetrised second moment per AO pair at every lattice cell,
    # via libint emultipole2 -- exact for any angular momentum.
    from ._vibeqc_core import compute_multipole_moments_lattice

    M_lat = compute_multipole_moments_lattice(
        basis, system, lat_opts, 2, (0.0, 0.0, 0.0)
    )

    # The density may ride the (longer) pair-complete one-electron cell
    # list under LatticeSumOptions.pair_complete_1e (#429) while the
    # moments keep the plain |g| ball; the shorter list is an exact prefix
    # of the longer one and the loop below runs over the moments' length.
    # See bipole_cell_moments.compute_cell_multipole_moments.
    if len(P_real.cells) < len(M_lat.cells):
        raise ValueError(
            f"cell-list length mismatch: P_real has {len(P_real.cells)} "
            f"cells, multipole lattice has {len(M_lat.cells)}. The density "
            f"must cover at least the moment cell list."
        )

    # Per-AO atom centre (bohr), in the libint AO order the moment blocks
    # follow. Pure (spherical) shells contribute 2l+1 AOs each.
    centers: list[tuple[float, float, float]] = []
    for sh in basis.shells():
        for _ in range(2 * int(sh.l) + 1):
            centers.append(tuple(sh.origin))
    A = np.asarray(centers, dtype=float)  # (nbf, 3) bra & ket atom centres
    if A.shape[0] != orig_nbf:
        raise ValueError(f"AO-centre count {A.shape[0]} != nbf {orig_nbf}")
    A2 = np.einsum("mi,mi->m", A, A)  # |A|^2 per AO

    E_sphero = 0.0
    for c, cell in enumerate(M_lat.cells):
        if not (np.asarray(P_real.cells[c].index) == np.asarray(cell.index)).all():
            raise ValueError(
                f"cell-list ordering mismatch at index {c}: "
                f"P_real has {tuple(P_real.cells[c].index)}, "
                f"multipole lattice has {tuple(cell.index)}"
            )
        P_g = np.asarray(P_real.blocks[c], dtype=float)
        if P_g.size == 0 or not P_g.any():
            continue
        blk = M_lat.blocks[c]
        M0 = np.asarray(blk[0], dtype=float)  # <1>  (overlap)
        M1 = [np.asarray(blk[k], dtype=float) for k in (1, 2, 3)]  # <x>,<y>,<z>
        TrM2 = (
            np.asarray(blk[4], dtype=float)
            + np.asarray(blk[7], dtype=float)
            + np.asarray(blk[9], dtype=float)
        )  # <x^2+y^2+z^2>
        g = np.asarray(cell.r_cart, dtype=float)
        Bk = A + g  # ket atom centres, shifted by the lattice vector
        Bk2 = np.einsum("ni,ni->n", Bk, Bk)
        # K_{muν} = 2.Tr<r^2> - 2 S_i (A_i + Bk_i).<r_i> + (|A|^2 + |Bk|^2).<1>
        AdotM1 = sum(A[:, i][:, None] * M1[i] for i in range(3))
        BdotM1 = sum(Bk[:, i][None, :] * M1[i] for i in range(3))
        K_g = (
            2.0 * TrM2
            - 2.0 * (AdotM1 + BdotM1)
            + (A2[:, None] + Bk2[None, :]) * M0
        )
        E_sphero += float(np.einsum("mn,mn->", P_g, K_g))

    return prefactor * E_sphero


@dataclass
class EwaldReciprocalSumResult:
    """Legacy result schema for the retired ``EXT EL-POLE`` helper.

    Kept importable for source compatibility.  No supported routine returns
    this schema because its signed ``E_ext_el_pole`` field misidentified a
    reciprocal electronic Ewald component as CRYSTAL's decomposition term.

    Attributes
    ----------
    E_ext_el_pole : float
        Historical negative of ``E_self``.  This field is not a valid
        CRYSTAL ``EXT EL-POLE`` value and must not be added to a BIELET term.
    E_self : float
        Historical reciprocal electronic Ewald component.  Use
        :class:`ElectronicReciprocalEwaldResult` for new code.
    n_k_vectors : int
        Number of reciprocal-space K vectors summed (K != 0 within
        cutoff).
    alpha_bohr_inv : float
        Ewald a used (bohr⁻¹). Should match V_ne / E_nn a.
    K_max_bohr_inv : float
        Reciprocal-space cutoff radius (bohr⁻¹).
    """

    E_ext_el_pole: float
    E_self: float
    n_k_vectors: int
    alpha_bohr_inv: float
    K_max_bohr_inv: float


@dataclass
class ElectronicReciprocalEwaldResult:
    """The ``K != 0`` reciprocal Ewald contribution of the electron density.

    This contribution is non-negative and depends on the Ewald splitting
    parameter.  It is not a standalone physical Hartree energy or CRYSTAL's
    penetration-zone-dependent ``EXT EL-POLE`` diagnostic.

    Attributes
    ----------
    energy : float
        Non-negative reciprocal electron-electron Ewald contribution in Ha.
    n_k_vectors : int
        Number of nonzero reciprocal vectors in the sum.
    alpha_bohr_inv : float
        Ewald splitting parameter in bohr^-1.
    K_max_bohr_inv : float
        Reciprocal cutoff in bohr^-1.
    """

    energy: float
    n_k_vectors: int
    alpha_bohr_inv: float
    K_max_bohr_inv: float


def crystal_default_ewald_alpha(V_cell_bohr3: float) -> float:
    """Default Ewald a from cell volume (CRYSTAL convention).

    The a = 2.8 . V^{-1/3} splitting is CRYSTAL's documented default
    convergence convention. With ``a^2`` entering the reciprocal-space
    Gaussian as ``exp(-|G|^2 / (4.a^2))``, reproducing it lets vibe-qc's
    Ewald sum be compared against CRYSTAL output term-by-term.

    This is the *reciprocal* of the value previously returned by
    vibe-qc (fixed v0.10 A1 audit):

        a  =  2.8 / V^{1/3}

    For a typical FCC-primitive cell with V ≈ 126 bohr^3 (MgO):
    a ≈ 0.558 bohr⁻¹. For LiH (V ≈ 28.7): a ≈ 0.915 bohr⁻¹.
    """
    if V_cell_bohr3 <= 0.0:
        raise ValueError(f"V_cell must be positive; got {V_cell_bohr3}")
    return 2.8 / (V_cell_bohr3 ** (1.0 / 3.0))


def crystal_ewald_reciprocal_cutoff(
    V_cell_bohr3: float,
    *,
    madelind: int = 6,
) -> float:
    """Reciprocal-space Ewald ``K_max`` from cell volume, matching the
    CRYSTAL convergence convention for term-by-term parity.

    CRYSTAL's reciprocal-lattice enumeration (selected via the public
    ``MADELIND`` keyword) accepts reciprocal vectors out to a cutoff
    set by the cell volume::

        K_max  =  2pi . √[12.8 . ((100 + MADELIND)/100)^2 . V^{-2/3}]

    where the ``12.8`` and the ``(100 + MADELIND)/100`` tolerance
    factor are CRYSTAL's documented convergence convention
    (``MADELIND`` defaults to 6). Reproducing this Cartesian ``K_max``
    lets the nuclear Ewald, the V_ne reciprocal piece, and the J_LR
    cache share one envelope, so vibe-qc's Ewald sum can be compared
    against CRYSTAL output term-by-term at finite cutoff.

    Parameters
    ----------
    V_cell_bohr3 : float
        Unit-cell volume in bohr^3.
    madelind : int
        CRYSTAL's ``MADELIND`` convergence parameter.  Default 6
        matches CRYSTAL's out-of-the-box value.

    Returns
    -------
    float
        Reciprocal-space cutoff in bohr⁻¹.
    """
    if V_cell_bohr3 <= 0.0:
        raise ValueError(f"V_cell must be positive; got {V_cell_bohr3}")
    if madelind < 0:
        raise ValueError(f"madelind must be >= 0; got {madelind}")
    faci = (100.0 + float(madelind)) / 100.0 * (V_cell_bohr3 ** (-1.0 / 3.0))
    h9 = 12.8 * faci * faci
    return 2.0 * math.pi * math.sqrt(h9)


def bipole_ewald_reciprocal_cutoff(
    V_cell_bohr3: float,
    alpha_bohr_inv: float,
    *,
    madelind: int = 6,
) -> float:
    """Reciprocal Ewald ``K_max`` of the BIPOLE routes, resolved for the
    splitting ``alpha`` actually in use (GitLab #674).

    CRYSTAL's volume-only envelope (:func:`crystal_ewald_reciprocal_cutoff`)
    is calibrated to its own ``alpha = 2.8 / V^(1/3)``
    (:func:`crystal_default_ewald_alpha`): at that pair the reciprocal
    Gaussian ``exp(-K^2 / (4 alpha^2))`` has fallen to
    ``exp(-pi^2 . 12.8 . 1.06^2 / 2.8^2) = 1.4e-8`` at ``K_max``, i.e.
    ``K_max / alpha = 2 pi sqrt(12.8) . 1.06 / 2.8 = 8.51`` whatever the
    cell. Ewald's potential is invariant to the screening parameter only
    when both lattice series are summed to convergence (Saunders,
    Freyria-Fava, Dovesi and Roetti, Mol. Phys. 77, 629 (1992), Property
    11; the ``2.8 / V^(1/3)`` of their Eq. (55) is a cost optimum for fully
    summed series, not an accuracy requirement), so an ``alpha`` above
    CRYSTAL's -- the corrected-exchange default bounded by the real-space
    cutoff (:func:`vibeqc.pbc_bipole_common.resolve_bipole_ewald_alpha`)
    or an explicit ``ewald_omega`` -- needs an envelope that grows with
    it. Measured on H2/STO-3G in a 30-bohr box at (1,1,1) and lattice
    cutoff 12, the volume-only envelope at ``alpha = 0.358`` read
    -1.1032469816 Ha against the split-invariant -1.1170858 Ha.

    This helper scales CRYSTAL's envelope by ``max(1, alpha / alpha_CRYSTAL)``:
    ``K_max / alpha`` stays at CRYSTAL's 8.51 for every ``alpha`` at or
    above CRYSTAL's value (the same 1.4e-8 resolution of the Gaussian),
    and below it CRYSTAL's envelope is returned verbatim. A run whose
    ``alpha`` is CRYSTAL's keeps its reciprocal sum bit for bit. The
    energy drivers and every analytic-gradient site derive ``K_max`` from
    this one function, so the gradient differentiates the reciprocal sum
    the energy evaluated.
    """
    alpha = float(alpha_bohr_inv)
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(
            f"alpha must be finite and positive; got {alpha_bohr_inv!r}"
        )
    k_max_crystal = crystal_ewald_reciprocal_cutoff(
        V_cell_bohr3, madelind=madelind
    )
    alpha_crystal = crystal_default_ewald_alpha(V_cell_bohr3)
    return k_max_crystal * max(1.0, alpha / alpha_crystal)


def compute_reciprocal_lattice_vectors(
    system: PeriodicSystem,
    K_max_bohr_inv: float,
) -> np.ndarray:
    """Enumerate reciprocal-lattice vectors with ``|K| <= K_max``,
    EXCLUDING K = 0.

    For a 3D periodic system with direct lattice ``a`` (columns ``a_i``),
    the reciprocal lattice is ``b = 2pi . (a⁻¹)ᵀ`` (columns ``b_i``).
    Each ``K = n1.b1 + n2.b2 + n3.b3`` for ``(n1, n2, n3) in ℤ^3``.

    Parameters
    ----------
    system : PeriodicSystem
        3D periodic system (raises for dim != 3).
    K_max_bohr_inv : float
        Cutoff radius in inverse bohr.

    Returns
    -------
    ndarray of shape (n_K, 3)
        Cartesian K vectors with ``0 < |K| <= K_max``.
    """
    if system.dim != 3:
        raise ValueError(
            f"compute_reciprocal_lattice_vectors requires dim=3; got dim={system.dim}"
        )
    if K_max_bohr_inv <= 0.0:
        raise ValueError(f"K_max must be positive; got {K_max_bohr_inv}")
    a = np.asarray(system.lattice, dtype=float)
    # b = 2pi . (a⁻¹)ᵀ -- columns are reciprocal-lattice vectors.
    b = 2.0 * math.pi * np.linalg.inv(a).T
    # For K=B.n with |K| <= K_max, reciprocal duality gives
    # |n_i|=|a_i.K|/(2pi) <= K_max.|a_i|/(2pi).  The per-axis bounds
    # are complete even for a strongly skew lattice.
    a_norms = np.linalg.norm(a, axis=0)
    n_max = np.ceil(K_max_bohr_inv * a_norms / (2.0 * math.pi)).astype(int)
    grids = [np.arange(-n, n + 1) for n in n_max]
    n1, n2, n3 = np.meshgrid(*grids, indexing="ij")
    idx = np.stack(
        [n1.ravel(), n2.ravel(), n3.ravel()],
        axis=-1,
    ).astype(float)
    K = idx @ b.T
    K2 = (K**2).sum(axis=1)
    keep = (K2 > 1e-12) & (K2 <= K_max_bohr_inv * K_max_bohr_inv)
    return K[keep]


def compute_cell_density_fourier_lattice(
    P_real: LatticeMatrixSet,
    basis: BasisSet,
    K_vectors: np.ndarray,
    *,
    ft_per_cell: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fourier transform of the cell electron density at given K-vectors.

    For the iter-N real-space density block representation
    ``r_cell(r) = S_{muν, g} P_muν(g) . phi_mu(r) . phi_ν(r - R_g)``::

        r̂(K)  =  ∫ r_cell(r) . exp(-i K . r) dr
              =  S_{muν, g} P_muν(g) . FT_muν(K; R_g)

    where ``FT_muν(K; R_g) = ∫ phi_mu(r) phi_ν(r-R_g) exp(-iK.r) dr`` is
    the shifted-ν AO-pair FT (closed-form Gaussian-product translate,
    see :func:`vibeqc._aopair_ft.ao_pair_fourier_transform_shifted_ket`).

    This is the gauge-correct lattice form that handles non-trivial
    ``D(g!=0)`` blocks at multi-k. For iter-1 SAD (D nonzero only at
    g=0) it reduces to ``S_{muν} D_muν(0) . FT_muν(K; 0)`` -- matching the
    Γ-only behaviour of the legacy :func:`compute_cell_density_fourier`.

    Parameters
    ----------
    P_real : LatticeMatrixSet
        Real-space cell density.
    basis : BasisSet
        AO basis matching P_real.
    K_vectors : ndarray of shape (n_K, 3)
        Reciprocal-space sampling points.
    ft_per_cell : (n_g, nbf, nbf, n_K) complex, optional
        Pre-computed shifted-ν FT stack from
        :func:`vibeqc._aopair_ft.ao_pair_fourier_transform_at_cells`.
        Pass this when r̂(K) is recomputed against the same lattice
        with a different density -- typical in an SCF inner loop with
        the same (basis, K_vectors, cells) but updated P.

    Returns
    -------
    rho_hat : ndarray of shape (n_K,) complex
        Fourier coefficients of the cell electron density.
    """
    K_vectors = np.ascontiguousarray(K_vectors, dtype=float)
    n_K = K_vectors.shape[0]
    if P_real.nbf != basis.nbasis:
        raise ValueError(f"P_real.nbf={P_real.nbf} != basis.nbasis={basis.nbasis}")
    cells = list(P_real.cells)
    n_cells = len(cells)
    R_g_arr = np.array(
        [np.asarray(c.r_cart, dtype=float) for c in cells],
        dtype=float,
    )
    if ft_per_cell is None:
        ft_per_cell = ao_pair_fourier_transform_at_cells(
            basis,
            K_vectors,
            R_g_arr,
        )
    correction = _libint_ylm_correction_per_ao(basis)
    # Apply per-AO correction (mu x ν) on every (g, K) slice.
    ft_corrected = ft_per_cell * (
        correction[None, :, None, None] * correction[None, None, :, None]
    )

    rho_hat = np.zeros(n_K, dtype=np.complex128)
    for c in range(n_cells):
        P_block = np.asarray(P_real.blocks[c], dtype=float)
        if P_block.size == 0:
            continue
        # S_muν D(g)_muν . FT_muν(K; R_g)
        rho_hat = rho_hat + np.einsum(
            "mn,mnk->k",
            P_block,
            ft_corrected[c],
        )
    return rho_hat


def compute_cell_density_fourier(
    P_real: LatticeMatrixSet,
    basis: BasisSet,
    K_vectors: np.ndarray,
) -> np.ndarray:
    """Fourier transform of the cell electron density at given K-vectors.

    Thin wrapper around :func:`compute_cell_density_fourier_lattice` --
    kept for API compatibility with the Γ-only Phase 5 entry point.
    Both functions implement the same gauge-correct shifted-ν FT
    formula as of 2026-05-18 (v0.9.0 multi-k fix); the legacy
    bra-pair-at-home phase approximation
    ``FT_muν(K; g) ≈ FT_muν(K; 0).exp(-iK.R_g)`` is no longer used.
    """
    return compute_cell_density_fourier_lattice(P_real, basis, K_vectors)


def compute_ext_el_pole_reciprocal_sum(
    P_real: LatticeMatrixSet,
    basis: BasisSet,
    system: PeriodicSystem,
    *,
    alpha_bohr_inv: Optional[float] = None,
    precision: float = 1e-8,
    pad_factor: float = 1.5,
) -> Never:
    """Fail closed for the unimplemented CRYSTAL ``EXT EL-POLE`` term.

    The legacy implementation does not construct CRYSTAL's shell-multipole
    model and the API defines neither its multipole order nor its ``T2``
    penetration criterion (Saunders et al. 1992, Eqs. 116-130).  Its inputs
    therefore do not uniquely specify the reported CRYSTAL decomposition.

    Use :func:`compute_electronic_reciprocal_ewald_energy` only when the
    alpha-dependent ``K != 0`` reciprocal Ewald contribution itself is the
    intended diagnostic.
    """
    raise NotImplementedError(
        "CRYSTAL EXT EL-POLE requires a shell-partition multipole Ewald "
        "model, a multipole order, and an explicit T2 penetration-zone "
        "criterion. This legacy API does not define that decomposition; "
        "the former sum returned the complete AO electron density's K != 0 "
        "reciprocal electron-electron Ewald contribution. Use "
        "compute_electronic_reciprocal_ewald_energy for that reciprocal "
        "diagnostic."
    )


def compute_electronic_reciprocal_ewald_energy(
    P_real: LatticeMatrixSet,
    basis: BasisSet,
    system: PeriodicSystem,
    *,
    alpha_bohr_inv: Optional[float] = None,
    precision: float = 1e-8,
    pad_factor: float = 1.5,
) -> ElectronicReciprocalEwaldResult:
    """Compute the ``K != 0`` reciprocal Ewald contribution of the AO density.

    Formula::

        E_recip = (1/2) . (4pi / V) . S_{K != 0}  |rho_hat(K)|^2
                  . exp(-|K|^2 / 4a^2) / |K|^2

    This non-negative contribution depends on ``a``.  It omits the real-space
    ``B`` and constant pieces of the complete Ewald interaction and the
    multipole/penetration partition used for CRYSTAL's ``EXT EL-POLE``.

    The ``a`` parameter should match the Ewald state used by the other terms.
    If ``alpha_bohr_inv`` is None, ``a = 2.8 / V^(1/3)`` is used.

    The reciprocal-space cutoff is set adaptively: ``K_max`` such that
    ``exp(-K_max^2/4a^2) <= precision`` (with a pad_factor for safety).

    Parameters
    ----------
    P_real : LatticeMatrixSet
        Real-space cell density (iter-N).
    basis : BasisSet
        AO basis.
    system : PeriodicSystem
        3D periodic system.
    alpha_bohr_inv : float, optional
        Ewald a in inverse bohr. If None, uses CRYSTAL default.
    precision : float
        Cutoff on the reciprocal Gaussian screening factor (default 1e-8).
        This is not an energy-error guarantee.
    pad_factor : float
        Multiplicative K_max safety factor, finite and at least 1
        (default 1.5).

    Returns
    -------
    ElectronicReciprocalEwaldResult
        Non-negative reciprocal electron-electron contribution and metadata.
    """
    if system.dim != 3:
        raise ValueError(
            "compute_electronic_reciprocal_ewald_energy requires dim=3; "
            f"got dim={system.dim}"
        )
    a = np.asarray(system.lattice, dtype=float)
    V = float(abs(np.linalg.det(a)))
    if not math.isfinite(V) or V <= 0.0:
        raise ValueError(f"degenerate lattice (V_cell = {V}); check system.lattice")

    alpha = (
        alpha_bohr_inv if alpha_bohr_inv is not None else crystal_default_ewald_alpha(V)
    )
    if not math.isfinite(alpha) or alpha <= 0.0:
        raise ValueError(f"alpha must be finite and positive; got {alpha}")
    if not math.isfinite(precision) or not 0.0 < precision < 1.0:
        raise ValueError(
            "precision must be finite and satisfy 0 < precision < 1; "
            f"got {precision}"
        )
    if not math.isfinite(pad_factor) or pad_factor < 1.0:
        raise ValueError(
            f"pad_factor must be finite and >= 1; got {pad_factor}"
        )

    # K_max sized by exp(-K^2/4a^2) <= precision -> K^2 >= -4a^2.ln(precision)
    # -> K_max = 2a . √(-ln(precision)) . pad_factor
    K_max = (
        2.0 * float(alpha) * float(math.sqrt(-math.log(precision))) * float(pad_factor)
    )

    K_vectors = compute_reciprocal_lattice_vectors(system, K_max)
    n_K = K_vectors.shape[0]
    if n_K == 0:
        raise RuntimeError(
            f"compute_electronic_reciprocal_ewald_energy: no K vectors within "
            f"K_max={K_max:.3f} (a={alpha:.3f}, V={V:.3f}); check inputs"
        )

    rho_hat = compute_cell_density_fourier(P_real, basis, K_vectors)
    K2 = (K_vectors**2).sum(axis=1)
    # Ewald-screened reciprocal kernel.
    kernel = np.exp(-K2 / (4.0 * alpha * alpha)) / K2  # (n_K,)
    rho_abs_sq = rho_hat.real**2 + rho_hat.imag**2  # (n_K,)
    E_recip = 0.5 * (4.0 * math.pi / V) * float(np.dot(kernel, rho_abs_sq))

    return ElectronicReciprocalEwaldResult(
        energy=E_recip,
        n_k_vectors=int(n_K),
        alpha_bohr_inv=float(alpha),
        K_max_bohr_inv=float(K_max),
    )
