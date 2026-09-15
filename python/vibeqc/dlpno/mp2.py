"""DLPNO-MP2 -- local MP2 with real PAO/PNO physics (M2/M2b).

This module implements the first *physically correct* rung of the DLPNO
ladder (blueprint: Pinski, Riplinger, Valeev, Neese, J. Chem. Phys. 143,
034108 (2015) -- "Sparse maps I"):

1. Occupied orbitals are (optionally) Foster-Boys localised; frozen-core
   orbitals are excluded before localisation.
2. Every surviving pair (i <= j) -- **including diagonal pairs** -- gets a
   PAO domain, an S-orthonormal semicanonical virtual basis
   (`pao.semicanonical_pao_basis`), and real density-fitted exchange
   integrals K_ij = (ia|jb) from a single batched half-transform of the
   DF B-tensor.
3. PNOs are built from the semicanonical MP2 pair density; the
   truncation error is compensated by the standard semicanonical
   correction ΔE_PNO = S_ij [e_ij(full PAO) - e_ij(truncated PNO)].
4. The residual iteration includes the localised-occupied Fock coupling
   -S_k [F_ik T_kj + T_ik F_kj] with amplitudes of neighbouring pairs
   projected between PNO bases through the AO overlap.
5. The closed-shell pair energy carries the antisymmetrised weights:
   e_ij = (2 - d_ij) S_ab K_ab (2 T_ab - T_ba).
6. Distant pairs (centroid separation >= ``dipole_r_min`` and estimated
   pair energy below ``tcut_pairs``) are excluded from the PNO machinery;
   their semicanonical dipole-dipole estimate (Riplinger & Neese 2013,
   Sec.II.C) is accumulated into ``e_distant`` and added to E_corr.

Exactness limit (the M2 gate, `tests/test_dlpno_mp2.py`): with full
domains (tcut_mkn = 0), no PNO truncation (tcut_pno = 0) and no pair
screening (tcut_pairs = 0), the iterated energy equals canonical DF-MP2
with the same fitting basis to <=1 µHa -- for canonical *and* localised
occupied orbitals, with and without frozen core.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pao import (
    build_atom_basis_map,
    build_projection_matrix,
    select_domain_atoms_mulliken,
    semicanonical_pao_basis,
)
from .pno_density import pair_density, resolve_pno_norm


@dataclass
class DLPNOMP2Options:
    """Options for the DLPNO-MP2 driver.

    Attributes
    ----------
    localise : str
        "boys" (default) -- Foster-Boys localise the active occupied
        orbitals with exact dipole integrals; "none" -- keep canonical
        orbitals (validation: semicanonical amplitudes are then exact
        without iteration).
    n_frozen : int | None
        Number of frozen-core orbitals (lowest canonical MOs), excluded
        before localisation and from all pair lists. ``None`` selects the
        shared published chemical-core convention; 0 is explicitly
        all-electron.
    tcut_pno : float
        PNO occupation truncation for strong pairs. 0 disables.
    tcut_pno_weak : float
        PNO occupation truncation for weak pairs. 0 disables.
    tcut_mkn : float
        Mulliken threshold for domain atom selection. 0 -> full domains.
    tcut_pairs : float
        Distant-pair screening threshold (Ha) on the semicanonical
        dipole-dipole pair-energy estimate. Pairs with centroid
        separation >= ``dipole_r_min`` and |estimate| < tcut_pairs are
        treated at the estimate level only (accumulated in
        ``e_distant``), not with the full PNO machinery -- turning the
        O(N^2) pair list into O(N) for extended systems. The default is the
        cross-route NormalPNO project convention. Set 0 to disable screening
        for exactness-limit calculations. The dipole estimate is crude; a
        calibrated MP2-pair-energy screen is a later refinement.
    tcut_pairs_weak : float
        Estimate threshold separating strong from weak pairs (which
        TCutPNO applies). Diagonal pairs are always strong.
    dipole_r_min : float
        Minimum centroid separation (bohr) for the dipole estimate to be
        trusted; closer pairs are never screened. Default 8.0.
    pair_distance_fn : callable, optional
        Override for the centroid pair distance used by the distant-pair
        screen: ``pair_distance_fn(r_i, r_j) -> float`` (bohr) on the two
        Cartesian centroids. ``None`` (default) uses the Euclidean norm
        -- bit-identical to the historical behavior. Periodic toroidal
        wrappers pass the exact minimum-image distance
        (``vibeqc.periodic_toroidal_mp2.toroidal_pair_distance_fn``) so
        wrap-around pairs are not falsely screened by their open-cluster
        Euclidean separation (which overestimates the true toroidal
        distance by up to the half-super-period; see
        handovers/HANDOVER_PERIODIC_LOCAL_CORRELATION.md Stage 5c).
    lindep : float
        PAO redundancy threshold (overlap eigenvalues).
    local_df : bool
        Use domain-restricted (local) density fitting: each pair's
        exchange integrals are refit with only the auxiliary functions
        in its fit domain, with a per-pair local metric V_[ij], instead
        of the global RI metric. This is local density fitting in a
        correlation method -- Werner, Manby & Knowles, J. Chem. Phys.
        118, 8149 (2003); the DLPNO sparse-maps form is Riplinger,
        Pinski, Becker, Valeev & Neese, J. Chem. Phys. 144, 024109
        (2016). Default False (global RI -- bit-for-bit the original
        path). With a full fit domain the local fit *is* the
        global RI (the 2-/3-centre integrals refit exactly), so the
        exactness limit is unchanged; ``fit_buffer`` controls the
        domain size. The fit dimension per pair is recorded in
        ``DLPNOMP2Result.fit_dim_per_pair`` -- bounded, not O(N_aux),
        which is the locality that makes DLPNO scale.
    fit_buffer : float
        Fit-domain extension (bohr) beyond each pair's PAO domain: an
        atom enters the fit domain if it lies within ``fit_buffer`` of
        any PAO-domain atom. Only used when ``local_df=True``. A very
        large value (>= system diameter) recovers the global RI exactly
        -- the exactness gate sets ``fit_buffer=1e9``. Default 4.0.
    max_iter : int
        Maximum Jacobi macro-iterations for the coupled LMP2 equations.
    conv_tol_energy : float
        Convergence on the correlation-energy change (Ha).
    conv_tol_residual : float
        Convergence on the max residual element.
    damping : float
        Amplitude-update damping (1.0 = undamped Jacobi).
    aux_basis : str | None
        RI fitting basis requested for the correlation integrals. ``None``
        lets :func:`vibeqc.run_job` resolve the orbital-basis default.
    """

    localise: str = "boys"
    n_frozen: int | None = None
    # Issue #448 deliberately uses the Liakos 2015 CC-family NormalPNO
    # triple as one cross-route vibe-qc policy. Pinski et al.'s published
    # DLPNO-MP2-specific Normal setting uses TCutPNO=1e-8; that remains
    # expressible as a custom value, but it is not the unified default.
    tcut_pno: float = 3.33e-7
    tcut_pno_weak: float = 3.33e-6
    tcut_mkn: float = 1e-3
    tcut_pairs: float = 1e-4
    tcut_pairs_weak: float = 1e-4
    #: Pair density whose eigenvalues are compared against ``tcut_pno``.
    #: ``"mp2"`` (default) is Riplinger and Neese 2013 Eq. 23, ORCA's
    #: ``PNONorm MP2Norm`` default and the density the published ``TCutPNO``
    #: presets are calibrated against; ``"legacy"`` is vibe-qc's historical
    #: density of the bare amplitudes, ``"iepa"`` the pre-2013 LPNO
    #: convention. Defaulted to ``"mp2"`` by the #65 (old #701) ruling. See
    #: :mod:`vibeqc.dlpno.pno_density`.
    pno_norm: str = "mp2"
    dipole_r_min: float = 8.0
    pair_distance_fn: object = None
    lindep: float = 1e-8
    local_df: bool = False
    fit_buffer: float = 4.0
    max_iter: int = 100
    conv_tol_energy: float = 1e-9
    conv_tol_residual: float = 1e-7
    damping: float = 1.0
    aux_basis: str | None = None


@dataclass
class _PairData:
    """Per-pair working data (PNO basis); indices are active-relative."""

    i: int
    j: int
    V: np.ndarray  # (nbf, n_pno) AO-expansion of the PNOs, S-orthonormal
    K: np.ndarray  # (n_pno, n_pno) exchange integrals (ia|jb)
    eps: np.ndarray  # (n_pno,) quasi-canonical PNO Fock eigenvalues
    T: np.ndarray  # (n_pno, n_pno) current amplitudes (i-side rows)
    n_pno: int = 0
    n_pao: int = 0
    e_full_sc: float = 0.0
    e_pno_sc: float = 0.0
    fit_dim: int = 0


@dataclass
class DLPNOMP2Result:
    """Result of a DLPNO-MP2 calculation.

    ``e_corr = e_corr_iterated + e_pno_correction + e_distant``.
    ``pair_energies`` keys are absolute occupied indices (frozen core
    counted), as are ``pno_per_pair`` keys.
    """

    e_hf: float = 0.0
    e_corr: float = 0.0
    e_corr_iterated: float = 0.0
    e_pno_correction: float = 0.0
    e_distant: float = 0.0
    e_total: float = 0.0
    n_frozen: int = 0
    n_pairs: int = 0
    n_pairs_screened: int = 0
    n_iter: int = 0
    converged: bool = False
    pair_energies: dict = field(default_factory=dict)
    pno_per_pair: dict = field(default_factory=dict)
    fit_dim_per_pair: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)


def _pair_energy(K: np.ndarray, T: np.ndarray, i: int, j: int) -> float:
    """Closed-shell pair energy e_ij = (2-d_ij).S_ab K_ab (2T_ab - T_ba)."""
    w = 1.0 if i == j else 2.0
    return w * float(np.sum(K * (2.0 * T - T.T)))


def _fit_domain_atoms(
    domain_atoms: np.ndarray,
    atom_coords: np.ndarray,
    fit_buffer: float,
) -> tuple[int, ...]:
    """Atom indices in a pair's extended local fitting domain."""
    natom = atom_coords.shape[0]
    if fit_buffer >= 1e8:
        return tuple(range(natom))
    keep_atom = np.zeros(natom, dtype=bool)
    for a in np.asarray(domain_atoms, dtype=int):
        within = np.linalg.norm(atom_coords - atom_coords[a], axis=1) <= fit_buffer
        keep_atom |= within
    return tuple(int(a) for a in np.where(keep_atom)[0])


def _aux_subbasis_for_atoms(molecule, aux_basis, fit_atoms: tuple[int, ...]):
    """Build an auxiliary BasisSet containing only shells on ``fit_atoms``."""
    from vibeqc._vibeqc_core import BasisSet

    fit_atom_set = set(fit_atoms)
    shells = [sh for sh in aux_basis.shells() if int(sh.atom_index) in fit_atom_set]
    if not shells:
        raise ValueError("local density-fitting domain contains no auxiliary shells")
    atom_label = ",".join(str(a) for a in fit_atoms)
    return BasisSet(
        molecule,
        shells,
        f"{aux_basis.name}:local[{atom_label}]",
        True,
    )


def _semicanonical_T(
    K: np.ndarray, f_ii: float, f_jj: float, eps: np.ndarray
) -> np.ndarray:
    """First-order amplitudes T_ab = K_ab / (f_ii + f_jj - e_a - e_b)."""
    denom = f_ii + f_jj - eps[:, None] - eps[None, :]
    return K / denom


def _dipole_pair_estimate(
    mu2_i: np.ndarray,
    eps_i: np.ndarray,
    f_ii: float,
    mu2_j: np.ndarray,
    eps_j: np.ndarray,
    f_jj: float,
    r_ij: float,
) -> float:
    """Semicanonical dipole-dipole pair-energy estimate.

    Spherically averaged second-order dispersion form
    (Riplinger & Neese, J. Chem. Phys. 138, 034106 (2013), Sec.II.C):

        e_ij ≈ -(4 / R⁶) S_{ain[i]} S_{bin[j]}
                |mu_ia|^2 |mu_jb|^2 / (e_a + e_b - F_ii - F_jj)

    with |mu_ia|^2 the squared transition-dipole norm from orbital i into
    its own orbital-domain semicanonical virtuals.
    """
    denom = eps_i[:, None] + eps_j[None, :] - f_ii - f_jj
    return -4.0 / r_ij**6 * float(np.sum(mu2_i[:, None] * mu2_j[None, :] / denom))


def run_dlpno_mp2(
    molecule,
    basis,
    rhf,
    df,
    options: DLPNOMP2Options | None = None,
) -> DLPNOMP2Result:
    """Run DLPNO-MP2 on a converged closed-shell RHF reference.

    Parameters
    ----------
    molecule, basis : Molecule, BasisSet
    rhf : RHFResult
        Converged RHF (``mo_coeffs``, ``mo_energies``, ``fock``, ``energy``).
    df : vibeqc.density_fitting.DensityFitting
        DF object for the orbital basis with a *real* RI fitting basis.
    options : DLPNOMP2Options

    Returns
    -------
    DLPNOMP2Result
    """
    from vibeqc._vibeqc_core import compute_overlap
    from vibeqc.correlation_conventions import effective_electron_count

    if options is None:
        options = DLPNOMP2Options()
    pno_norm = resolve_pno_norm(getattr(options, "pno_norm", "mp2"))

    F_ao = np.asarray(rhf.fock).copy()
    S_ao = np.asarray(compute_overlap(basis)).copy()
    C = np.asarray(rhf.mo_coeffs).copy()
    n_occ = effective_electron_count(molecule, rhf) // 2
    nbf = C.shape[0]

    from vibeqc.correlation_conventions import resolve_frozen_core_count

    nf = resolve_frozen_core_count(molecule, options.n_frozen, reference=rhf)
    if nf < 0 or nf >= n_occ:
        raise ValueError(f"n_frozen={nf} out of range for n_occ={n_occ}")
    n_act = n_occ - nf
    C_occ_full = C[:, :n_occ]
    C_act = C[:, nf:n_occ]

    # ----- active occupieds: localise or keep canonical -------------------
    dipoles = None
    if (
        options.localise == "boys"
        or options.tcut_pairs > 0.0
        or options.tcut_pairs_weak > 0.0
    ):
        from vibeqc import compute_dipole

        dip = compute_dipole(basis)
        dipoles = np.zeros((nbf, nbf, 3))
        dipoles[:, :, 0] = np.asarray(dip.x)
        dipoles[:, :, 1] = np.asarray(dip.y)
        dipoles[:, :, 2] = np.asarray(dip.z)

    if options.localise == "boys":
        from vibeqc.localise import foster_boys_localise

        C_loc = foster_boys_localise(C_act, dipoles, max_iter=200)
    elif options.localise == "none":
        C_loc = C_act
    else:
        raise ValueError(f"unknown localise option: {options.localise!r}")

    F_oo = C_loc.T @ F_ao @ C_loc  # active block; not diagonal if localised

    # ----- domains + projector (project out ALL occupieds, incl. core) ----
    atom_first, _ = build_atom_basis_map(molecule, basis)
    natom = len(atom_first) - 1
    Q_vir = build_projection_matrix(C_occ_full, S_ao)

    # ----- DF integral source: global B-tensor, or local-fit primitives --
    # Global path (default): one batched half-transform with the global RI
    # metric baked in. Local path: build a restricted auxiliary sub-basis for
    # each unique fit domain and evaluate the raw 2c/3c integrals directly.
    if options.local_df:
        from vibeqc._vibeqc_core import compute_2c_eri, compute_3c_eri

        aux_basis = getattr(df, "aux_basis", None)
        if aux_basis is None:
            raise ValueError("local_df=True requires a df object with aux_basis")
        atom_coords = np.array(
            [np.asarray(at.xyz, dtype=float) for at in molecule.atoms]
        )
        B_half = None
        local_fit_cache: dict[
            tuple[int, ...],
            tuple[np.ndarray, np.ndarray, int],
        ] = {}
    else:
        B_half = df.mo_transform(C_loc, np.eye(nbf))  # (naux, n_act, nbf)
        local_fit_cache = {}

    result = DLPNOMP2Result(e_hf=float(rhf.energy), n_frozen=nf)

    # ----- distant-pair screening via the dipole estimate -----------------
    all_pairs = [(i, j) for i in range(n_act) for j in range(i, n_act)]
    pair_estimates: dict[tuple[int, int], float] = {}
    screened: set[tuple[int, int]] = set()

    needs_estimates = options.tcut_pairs > 0.0 or options.tcut_pairs_weak > 0.0
    if needs_estimates and dipoles is not None:
        # Orbital domains: semicanonical virtuals + transition dipoles per
        # active orbital i (domain of the diagonal pair (i,i)).
        centroids = np.zeros((n_act, 3))
        for c in range(3):
            centroids[:, c] = np.einsum("mi,mn,ni->i", C_loc, dipoles[:, :, c], C_loc)
        orb_mu2: list[np.ndarray] = []
        orb_eps: list[np.ndarray] = []
        for i in range(n_act):
            if options.tcut_mkn > 0.0:
                atoms_i = select_domain_atoms_mulliken(
                    C_loc, S_ao, atom_first, i, i, options.tcut_mkn
                )
            else:
                atoms_i = np.arange(natom, dtype=int)
            mask = np.zeros(nbf, dtype=bool)
            for a in atoms_i:
                mask[atom_first[a] : atom_first[a + 1]] = True
            V_i, eps_i = semicanonical_pao_basis(
                F_ao, S_ao, Q_vir, np.where(mask)[0], options.lindep
            )
            mu2 = np.zeros(V_i.shape[1])
            for c in range(3):
                mu = C_loc[:, i] @ dipoles[:, :, c] @ V_i
                mu2 += mu**2
            orb_mu2.append(mu2)
            orb_eps.append(eps_i)

        for i, j in all_pairs:
            if i == j:
                continue
            if options.pair_distance_fn is not None:
                r_ij = float(options.pair_distance_fn(centroids[i], centroids[j]))
            else:
                r_ij = float(np.linalg.norm(centroids[i] - centroids[j]))
            if r_ij < options.dipole_r_min:
                continue
            if orb_eps[i].size == 0 or orb_eps[j].size == 0:
                continue
            e_est = _dipole_pair_estimate(
                orb_mu2[i],
                orb_eps[i],
                float(F_oo[i, i]),
                orb_mu2[j],
                orb_eps[j],
                float(F_oo[j, j]),
                r_ij,
            )
            pair_estimates[(i, j)] = e_est
            if options.tcut_pairs > 0.0 and abs(e_est) < options.tcut_pairs:
                screened.add((i, j))
                result.e_distant += e_est

    kept_pairs = [p for p in all_pairs if p not in screened]
    result.n_pairs_screened = len(screened)

    # ----- per-pair: domain -> semicanonical PAOs -> DF integrals -> PNOs ----
    pairs: dict[tuple[int, int], _PairData] = {}
    e_pno_correction = 0.0

    for i, j in kept_pairs:
        if options.tcut_mkn > 0.0:
            domain_atoms = select_domain_atoms_mulliken(
                C_loc, S_ao, atom_first, i, j, options.tcut_mkn
            )
        else:
            domain_atoms = np.arange(natom, dtype=int)
        ao_mask = np.zeros(nbf, dtype=bool)
        for a in domain_atoms:
            ao_mask[atom_first[a] : atom_first[a + 1]] = True
        ao_idx = np.where(ao_mask)[0]

        V_semi, eps_pao = semicanonical_pao_basis(
            F_ao, S_ao, Q_vir, ao_idx, lindep_thresh=options.lindep
        )
        n_pao = V_semi.shape[1]
        if n_pao == 0:
            continue

        # Pair exchange integrals K_ab = (ia|jb).
        if options.local_df:
            # Domain-restricted (local) fit: refit (ia|jb) using only the
            # aux functions in this pair's fit domain, with the *local*
            # metric V_[ij] = (P|Q)_{P,Q in [ij]}. A full fit domain (all
            # aux) reproduces the global RI exactly.
            fit_atoms = _fit_domain_atoms(domain_atoms, atom_coords, options.fit_buffer)
            if fit_atoms not in local_fit_cache:
                aux_loc = _aux_subbasis_for_atoms(molecule, aux_basis, fit_atoms)
                T_loc = np.asarray(compute_3c_eri(basis, aux_loc))
                V_fit = np.asarray(compute_2c_eri(aux_loc))
                local_fit_cache[fit_atoms] = (T_loc, V_fit, int(aux_loc.nbasis))
            T_loc, V_fit, fit_dim = local_fit_cache[fit_atoms]
            # M_i[P,ν] = sum_mu C_i[mu] (muν|P), then rotate ν into PAOs.
            M_i = (
                np.einsum("m,Pmn->Pn", C_loc[:, i], T_loc, optimize=True)
                @ V_semi
            )
            M_j = (
                np.einsum("m,Pmn->Pn", C_loc[:, j], T_loc, optimize=True)
                @ V_semi
            )
            K_pao = M_i.T @ np.linalg.solve(V_fit, M_j)
        else:
            B_i = B_half[:, i, :] @ V_semi  # (naux, n_pao)
            B_j = B_half[:, j, :] @ V_semi
            K_pao = B_i.T @ B_j
            fit_dim = int(B_half.shape[0])

        f_ii, f_jj = float(F_oo[i, i]), float(F_oo[j, j])
        T_pao = _semicanonical_T(K_pao, f_ii, f_jj, eps_pao)
        e_full = _pair_energy(K_pao, T_pao, i, j)

        # PNOs from the semicanonical pair density.
        delta = 1.0 if i == j else 0.0
        D_pair = pair_density(T_pao, delta, pno_norm)
        occs, d = np.linalg.eigh(D_pair)
        order = np.argsort(-occs)
        occs, d = occs[order], d[:, order]

        is_weak = (
            i != j
            and options.tcut_pairs_weak > 0.0
            and abs(pair_estimates.get((i, j), np.inf)) < options.tcut_pairs_weak
        )
        tcut = options.tcut_pno_weak if is_weak else options.tcut_pno
        keep = occs > tcut if tcut > 0.0 else np.ones_like(occs, dtype=bool)
        n_pno = int(np.sum(keep))
        if n_pno == 0:
            keep[0] = True
            n_pno = 1
        d = d[:, keep]

        # Quasi-canonicalise within the retained PNO space.
        F_pno = d.T @ np.diag(eps_pao) @ d
        F_pno = 0.5 * (F_pno + F_pno.T)
        eps_pno, u = np.linalg.eigh(F_pno)
        d = d @ u

        V_pno = V_semi @ d
        K_pno = d.T @ K_pao @ d
        T_pno = _semicanonical_T(K_pno, f_ii, f_jj, eps_pno)
        e_trunc = _pair_energy(K_pno, T_pno, i, j)
        e_pno_correction += e_full - e_trunc

        pairs[(i, j)] = _PairData(
            i=i,
            j=j,
            V=V_pno,
            K=K_pno,
            eps=eps_pno,
            T=T_pno,
            n_pno=n_pno,
            n_pao=n_pao,
            e_full_sc=e_full,
            e_pno_sc=e_trunc,
            fit_dim=fit_dim,
        )

    # The DF half-transforms (naux * n_act * nbf) are only needed by the
    # per-pair build loop above; free them before the long coupled-LMP2
    # iteration below.
    del B_half, local_fit_cache

    result.n_pairs = len(pairs)
    result.e_pno_correction = e_pno_correction
    result.pno_per_pair = {(k[0] + nf, k[1] + nf): p.n_pno for k, p in pairs.items()}
    result.fit_dim_per_pair = {
        (k[0] + nf, k[1] + nf): p.fit_dim for k, p in pairs.items()
    }

    if not pairs:
        result.converged = True
        result.e_corr = result.e_distant
        result.e_total = result.e_hf + result.e_corr
        return result

    # ----- coupled LMP2 iteration -----------------------------------------
    # Residual in each pair's quasi-canonical PNO basis:
    #   R_ij = K_ij + (e_a + e_b)∘T_ij - S_k [ F_ik . P(T_kj) + F_kj . P(T_ik) ]
    # where P projects neighbour amplitudes into pair (i,j)'s PNO basis via
    # the AO overlap: P(T_q) = S_pq T_q S_pqᵀ with S_pq = V_pᵀ S V_q.
    # For canonical occupieds F is diagonal and the k-sums collapse to the
    # (e_a+e_b-f_ii-f_jj) semicanonical form, converging in one step.

    overlap_cache: dict[tuple[tuple[int, int], tuple[int, int]], np.ndarray] = {}

    def S_pq(p: tuple[int, int], q: tuple[int, int]) -> np.ndarray:
        key = (p, q)
        if key not in overlap_cache:
            overlap_cache[key] = pairs[p].V.T @ S_ao @ pairs[q].V
        return overlap_cache[key]

    def get_T(k: int, l: int) -> tuple[np.ndarray, tuple[int, int]] | None:
        """Amplitudes T_kl (k-side rows) and the storage key, or None."""
        key = (k, l) if k <= l else (l, k)
        p = pairs.get(key)
        if p is None:
            return None
        return (p.T if k <= l else p.T.T), key

    # Try fused C++ iteration kernel (import once).
    try:
        from .._vibeqc_core import dlpno_mp2_iterate as _lmp2_iter
    except ImportError:
        _lmp2_iter = None

    e_corr_prev = 0.0
    for iteration in range(options.max_iter):
        if _lmp2_iter is not None:
            # Pack pair data into flat lists for the C++ kernel.
            n_pairs = len(pairs)
            pi = [p.i for p in pairs.values()]
            pj = [p.j for p in pairs.values()]
            K = [np.asfortranarray(p.K) for p in pairs.values()]
            eps = [np.asarray(p.eps) for p in pairs.values()]
            T_list = [np.asfortranarray(p.T) for p in pairs.values()]
            V_list = [np.asfortranarray(p.V) for p in pairs.values()]
            F_oo_arr = np.asfortranarray(F_oo)
            S_arr = np.asfortranarray(S_ao)

            T_new, max_r, e_corr = _lmp2_iter(
                pi,
                pj,
                K,
                eps,
                T_list,
                V_list,
                F_oo_arr,
                S_arr,
                options.damping,
            )
            # Write back amplitudes.
            for idx, p in enumerate(pairs.values()):
                p.T = np.asarray(T_new[idx])
                if p.i == p.j:
                    p.T = 0.5 * (p.T + p.T.T)
        else:
            # Python fallback.
            max_r = 0.0
            new_T: dict[tuple[int, int], np.ndarray] = {}

            for (i, j), p in pairs.items():
                R = p.K + (p.eps[:, None] + p.eps[None, :]) * p.T
                for k in range(n_act):
                    f_ik = float(F_oo[i, k])
                    if abs(f_ik) > 1e-14:
                        got = get_T(k, j)
                        if got is not None:
                            T_kj, q = got
                            if q == (i, j):
                                R -= f_ik * T_kj
                            else:
                                S_link = S_pq((i, j), q)
                                R -= f_ik * (S_link @ T_kj @ S_link.T)
                    f_kj = float(F_oo[k, j])
                    if abs(f_kj) > 1e-14:
                        got = get_T(i, k)
                        if got is not None:
                            T_ik, q = got
                            if q == (i, j):
                                R -= f_kj * T_ik
                            else:
                                S_link = S_pq((i, j), q)
                                R -= f_kj * (S_link @ T_ik @ S_link.T)
                max_r = max(max_r, float(np.max(np.abs(R))))

                denom = (
                    float(F_oo[i, i])
                    + float(F_oo[j, j])
                    - p.eps[:, None]
                    - p.eps[None, :]
                )
                new_T[(i, j)] = p.T + options.damping * R / denom

            for key, T in new_T.items():
                pairs[key].T = T
            for (i, j), p in pairs.items():
                if i == j:
                    p.T = 0.5 * (p.T + p.T.T)

            e_corr = sum(_pair_energy(p.K, p.T, p.i, p.j) for p in pairs.values())
        delta_e = e_corr - e_corr_prev
        result.trace.append(
            {
                "iter": iteration + 1,
                "e_corr": e_corr,
                "delta_e": delta_e,
                "max_r": max_r,
            }
        )
        if (
            iteration > 0
            and abs(delta_e) < options.conv_tol_energy
            and max_r < options.conv_tol_residual
        ):
            result.converged = True
            result.n_iter = iteration + 1
            break
        e_corr_prev = e_corr
    else:
        result.n_iter = options.max_iter

    result.e_corr_iterated = sum(
        _pair_energy(p.K, p.T, p.i, p.j) for p in pairs.values()
    )
    result.pair_energies = {
        (p.i + nf, p.j + nf): _pair_energy(p.K, p.T, p.i, p.j) for p in pairs.values()
    }
    result.e_corr = result.e_corr_iterated + result.e_pno_correction + result.e_distant
    result.e_total = result.e_hf + result.e_corr
    return result
