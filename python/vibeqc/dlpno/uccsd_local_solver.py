"""Local-domain open-shell DLPNO-UCCSD solver.

This is the reduced-scaling sibling of :mod:`vibeqc.dlpno.uccsd`'s dense
O(N^6) correctness pilot.  Each occupied pair owns a PNO basis, amplitudes
from neighbouring pairs are projected into that basis, and the native
``dlpno_uccsd_pair_residual`` kernel evaluates one bounded spin-orbital
domain at a time. The CCSD phase forms no global four-index
electron-repulsion tensor. The opt-in ``triples_mode="t1-iterative"``
currently materializes dense spin-orbital triples source blocks and is an
experimental, ``max_nbf``-capped correctness/generalization path rather than
a reduced-scaling triples implementation.

The full-domain limit is the correctness ratchet: with ``tcut_mkn=0``,
``tcut_pno=0``, ``tcut_pno_singles=0``, ``tcut_pairs=0``, and
``tcut_tno=0``, and ``coupling_radius=0`` every virtual and occupied is
retained, so the local projections are unitary and the result reproduces the
dense pilot. PAO/PNO, singles-PNO, weak-pair, TNO, and occupied-coupling
projections become exact in that limit. Pair, PNO, and PAO-domain truncation
use the shared NormalPNO default; singles-PNO, TNO, and occupied-coupling
truncation remain opt-in while broader radical benchmarks are accumulated.
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
from .uccsd import _boys_localise


@dataclass
class LocalUCCSDOptions:
    """Controls for the local-domain open-shell UCCSD solver.

    ``tcut_pairs=0`` keeps every pair in the CCSD iteration. A positive value
    screens pairs whose full-virtual MP2 estimate is smaller in magnitude than
    the threshold and retains their MP2 energy. ``tcut_mkn=0`` keeps the full
    canonical virtual space; a positive value selects pair atoms by the
    Mulliken populations of the two occupied orbitals and builds separate
    alpha/beta semicanonical PAOs on their union. ``tcut_pno_singles=0`` keeps
    every singles amplitude in the full spin-virtual space. A positive value
    builds occupied-specific, same-spin natural orbitals from the sum of that
    occupied's MP2 pair densities, optionally inside its Mulliken-selected PAO
    domain. ``coupling_radius=0`` keeps the full occupied coupling set and all
    triples. ``compute_triples=False`` leaves the perturbative triples path
    off; when enabled, ``tcut_tno=0`` keeps the full union of the three pair
    spaces for each distinct occupied triple. On the OH/def2-SVP calibration
    under `(T1)`, positive ``tcut_tno`` raises the energy by +12 uHa at
    ``1e-9`` (the ``T_CutTNO`` default of Guo et al. 2018), rising to +759 uHa
    at ``1e-4``; ``1e-12`` already changes it by +11 uHa. These measurements
    are calibration behavior, not a universal sign or monotonicity guarantee
    for a nonvariational triples correction.
    Watch ``LocalUCCSDResult.n_degenerate_tno_triples``: a triple retaining
    fewer than three spin virtuals cannot host a triple excitation at all, so a
    nonzero count means the correction has been truncated out of existence
    rather than approximated. The pair/PNO/domain defaults form the shared
    NormalPNO policy; the open-shell radical benchmark evidence remains less
    extensive than for closed-shell routes. ``triples_mode="t0"`` uses the
    diagonal localised occupied Fock, while the opt-in ``"t1"`` mode restores
    its off-diagonal coupling by spin-block semicanonicalisation. ``"t1"``
    rotates the occupied indices out of the localised basis and so cannot carry
    a screened triple list: it fails closed when ``tcut_pairs`` or
    ``coupling_radius`` actually screens a triple, and runs the exact full
    triple list when they do not. The singles, TNO, and coupling controls
    remain outside the published three-threshold convention.
    """

    localise: str = "boys"
    n_frozen: int | None = None
    # This operational route consumes all three members of the Liakos 2015
    # NormalPNO cross-route project policy. Pinski's MP2-specific threshold
    # set is not used for UCCSD. Explicit zero values retain the exactness
    # settings for callers that need full domains and no pair screening.
    tcut_pno: float = 3.33e-7
    tcut_pairs: float = 1e-4
    tcut_mkn: float = 1e-3
    tcut_pno_singles: float = 0.0
    compute_triples: bool = False
    tcut_tno: float = 0.0
    triples_mode: str = "t0"
    lindep: float = 1e-8
    coupling_radius: float = 0.0
    max_iter: int = 100
    conv_tol_energy: float = 1e-9
    conv_tol_residual: float = 1e-7
    diis_size: int = 6
    max_nbf: int = 160
    # RI fitting basis for the correlation integrals. None lets run_job
    # resolve the orbital-basis default before preflight and execution.
    aux_basis: str | None = None


@dataclass
class LocalUCCSDResult:
    e_hf: float = 0.0
    e_corr: float = 0.0
    e_t: float = 0.0
    e_total: float = 0.0
    n_iter: int = 0
    converged: bool = False
    n_pairs: int = 0
    n_frozen: int = 0
    localise: str = "boys"
    triples_mode: str = "t0"
    t1_norm: float = 0.0
    avg_pno: float = 0.0
    avg_pao: float = 0.0
    avg_singles_domain: float = 0.0
    avg_tno: float = 0.0
    avg_coupled_occ: float = 0.0
    n_screened: int = 0
    n_triples: int = 0
    n_screened_triples: int = 0
    #: Evaluated triples whose TNO domain retained fewer than three spin
    #: virtuals.  A triple excitation needs three distinct virtuals, so such a
    #: triple contributes exactly zero by construction rather than
    #: approximately: a nonzero count means ``tcut_tno`` has truncated past the
    #: point where the correction exists at all, not that it is small.
    n_degenerate_tno_triples: int = 0
    e_screened_mp2: float = 0.0
    pno_per_pair: dict[tuple[int, int], int] = field(default_factory=dict)
    pao_per_pair: dict[tuple[int, int], int] = field(default_factory=dict)
    singles_per_occ: dict[int, int] = field(default_factory=dict)
    tno_per_triple: dict[tuple[int, int, int], int] = field(
        default_factory=dict
    )
    screened_pair_energies: dict[tuple[int, int], float] = field(
        default_factory=dict
    )
    trace: list[dict[str, float | int]] = field(default_factory=list)
    #: True only when the requested triples contraction was actually entered.
    #: Non-converged CCSD or insufficient occupied space leaves it false.
    triples_executed: bool = False


def _domain_from_density(
    density: np.ndarray,
    f_vv: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build and quasi-canonicalise a virtual domain from a PSD density."""
    occ, vectors = np.linalg.eigh(0.5 * (density + density.T))
    order = np.argsort(-occ)
    occ = occ[order]
    vectors = vectors[:, order]
    keep = (
        occ > threshold
        if threshold > 0.0
        else np.ones_like(occ, dtype=bool)
    )
    if not np.any(keep):
        keep[0] = True
    vectors = vectors[:, keep]
    f_domain = vectors.T @ f_vv @ vectors
    eps, rotation = np.linalg.eigh(0.5 * (f_domain + f_domain.T))
    return np.ascontiguousarray(vectors @ rotation), eps


def run_local_dlpno_uccsd(
    molecule,
    basis,
    uhf,
    df,
    options: LocalUCCSDOptions | None = None,
) -> LocalUCCSDResult:
    """Run local-domain DLPNO-UCCSD on a converged UHF reference.

    This is the reduced-scaling production engine selected by ``run_job`` for
    open-shell ``dlpno-ccsd`` and ``dlpno-ccsd(t)`` calculations. Local PNO
    assembly and iteration remain ratcheted against the independent dense
    ``run_dlpno_uccsd_pilot`` oracle at the full-domain limit.
    """
    from vibeqc._vibeqc_core import (
        compute_overlap,
        dlpno_spin_orbital_triple_energy,
        dlpno_uccsd_pair_residual,
    )

    if options is None:
        options = LocalUCCSDOptions()
    from vibeqc.correlation_conventions import effective_electron_count
    if options.localise not in ("none", "boys", "external"):
        raise ValueError(f"unknown localise option: {options.localise!r}")
    triples_mode = str(options.triples_mode).lower()
    if triples_mode not in ("t0", "t1", "t1-iterative"):
        raise ValueError(f"unknown triples_mode: {options.triples_mode!r}")
    if options.tcut_pno < 0.0:
        raise ValueError("tcut_pno must be non-negative")
    if options.tcut_pairs < 0.0:
        raise ValueError("tcut_pairs must be non-negative")
    if options.tcut_mkn < 0.0:
        raise ValueError("tcut_mkn must be non-negative")
    if options.tcut_pno_singles < 0.0:
        raise ValueError("tcut_pno_singles must be non-negative")
    if options.tcut_tno < 0.0:
        raise ValueError("tcut_tno must be non-negative")
    if options.lindep < 0.0:
        raise ValueError("lindep must be non-negative")

    ca = np.asarray(uhf.mo_coeffs_alpha)
    cb = np.asarray(uhf.mo_coeffs_beta)
    fa_ao = np.asarray(uhf.fock_alpha)
    fb_ao = np.asarray(uhf.fock_beta)
    nbf = ca.shape[0]
    if nbf > options.max_nbf:
        raise ValueError(
            f"run_local_dlpno_uccsd: {nbf} basis functions exceeds the "
            f"max_nbf={options.max_nbf} Python-engine guard"
        )

    ne = effective_electron_count(molecule, uhf)
    two_s = molecule.multiplicity - 1
    nalpha = (ne + two_s) // 2
    nbeta = (ne - two_s) // 2
    from vibeqc.correlation_conventions import resolve_frozen_core_count

    nf = resolve_frozen_core_count(molecule, options.n_frozen, reference=uhf)
    if nf < 0 or nf > nbeta:
        raise ValueError(f"n_frozen={nf} out of range for n_beta={nbeta}")

    noa = nalpha - nf
    nob = nbeta - nf
    nva = nbf - nalpha
    nvb = nbf - nbeta
    no = noa + nob
    nv = nva + nvb
    result = LocalUCCSDResult(
        e_hf=float(uhf.energy),
        n_frozen=nf,
        localise=options.localise,
        triples_mode=triples_mode,
    )
    if no < 2 or nv == 0:
        result.converged = True
        result.e_total = result.e_hf
        return result

    cao, cav = ca[:, nf:nalpha], ca[:, nalpha:]
    cbo, cbv = cb[:, nf:nbeta], cb[:, nbeta:]
    if options.localise == "boys":
        cao_use = _boys_localise(cao, basis)
        cbo_use = _boys_localise(cbo, basis)
    else:
        cao_use, cbo_use = cao, cbo

    overlap = np.asarray(compute_overlap(basis))
    c_occ = np.hstack([cao_use, cbo_use])
    atom_first, _ = build_atom_basis_map(molecule, basis)
    q_alpha = build_projection_matrix(ca[:, :nalpha], overlap)
    q_beta = build_projection_matrix(cb[:, :nbeta], overlap)

    # Occupied-first spin-orbital blocks.  Cross-spin charge-density and
    # Fock blocks are exactly zero; alpha and beta orbitals retain their
    # distinct UHF spatial coefficients.
    f_oo = np.zeros((no, no))
    f_ov = np.zeros((no, nv))
    f_vv = np.zeros((nv, nv))
    f_oo[:noa, :noa] = cao_use.T @ fa_ao @ cao_use
    f_oo[noa:, noa:] = cbo_use.T @ fb_ao @ cbo_use
    f_ov[:noa, :nva] = cao_use.T @ fa_ao @ cav
    f_ov[noa:, nva:] = cbo_use.T @ fb_ao @ cbv
    f_vv[:nva, :nva] = cav.T @ fa_ao @ cav
    f_vv[nva:, nva:] = cbv.T @ fb_ao @ cbv

    boo_a = np.asarray(df.mo_transform(cao_use, cao_use))
    boo_b = np.asarray(df.mo_transform(cbo_use, cbo_use))
    bov_a = np.asarray(df.mo_transform(cao_use, cav))
    bov_b = np.asarray(df.mo_transform(cbo_use, cbv))
    bvv_a = np.asarray(df.mo_transform(cav, cav))
    bvv_b = np.asarray(df.mo_transform(cbv, cbv))
    naux = bov_a.shape[0]
    b_oo = np.zeros((naux, no, no))
    b_ov = np.zeros((naux, no, nv))
    b_vv = np.zeros((naux, nv, nv))
    b_oo[:, :noa, :noa] = boo_a
    b_oo[:, noa:, noa:] = boo_b
    b_ov[:, :noa, :nva] = bov_a
    b_ov[:, noa:, nva:] = bov_b
    b_vv[:, :nva, :nva] = bvv_a
    b_vv[:, nva:, nva:] = bvv_b

    # Localised occupied centroids define bounded occupied coupling sets.
    # Spin does not enter the position expectation value.
    from vibeqc import compute_dipole

    dip = compute_dipole(basis)
    position = [np.asarray(dip.x), np.asarray(dip.y), np.asarray(dip.z)]
    centroids = np.column_stack(
        [
            np.einsum("mi,mn,ni->i", c_occ, axis, c_occ, optimize=True)
            for axis in position
        ]
    )
    distances = np.linalg.norm(
        centroids[:, None, :] - centroids[None, :, :], axis=2
    )
    radius = float(options.coupling_radius)

    def coupled_occupieds(*centres: int) -> np.ndarray:
        if radius <= 0.0:
            return np.arange(no, dtype=int)
        near = np.zeros(no, dtype=bool)
        for centre in centres:
            near |= distances[centre] < radius
            near[centre] = True
        return np.where(near)[0]

    eps_o = np.diag(f_oo)
    eps_v = np.diag(f_vv)
    all_pair_keys = [(p, q) for p in range(no) for q in range(p + 1, no)]
    pair_u: dict[tuple[int, int], np.ndarray] = {}
    pair_eps: dict[tuple[int, int], np.ndarray] = {}
    pair_k: dict[tuple[int, int], np.ndarray] = {}
    t2: dict[tuple[int, int], np.ndarray] = {}
    pair_pao_size: dict[tuple[int, int], int] = {}
    pair_mp2_energy: dict[tuple[int, int], float] = {}
    screened_pairs: set[tuple[int, int]] = set()
    singles_density = (
        np.zeros((no, nv, nv))
        if options.tcut_pno_singles > 0.0
        else None
    )

    def pair_pao_basis(p: int, q: int) -> np.ndarray | None:
        """Return combined-spin PAOs in canonical virtual coordinates."""
        if options.tcut_mkn <= 0.0:
            return None
        atoms = select_domain_atoms_mulliken(
            c_occ,
            overlap,
            atom_first,
            p,
            q,
            options.tcut_mkn,
        )
        if atoms.size == 0:
            raise ValueError(
                f"empty PAO atom domain for occupied pair {(p, q)} at "
                f"tcut_mkn={options.tcut_mkn}"
            )
        ao_indices = np.concatenate(
            [
                np.arange(atom_first[atom], atom_first[atom + 1])
                for atom in atoms
            ]
        )
        va, _ = semicanonical_pao_basis(
            fa_ao,
            overlap,
            q_alpha,
            ao_indices,
            options.lindep,
        )
        vb, _ = semicanonical_pao_basis(
            fb_ao,
            overlap,
            q_beta,
            ao_indices,
            options.lindep,
        )
        domain = np.zeros((nv, va.shape[1] + vb.shape[1]))
        domain[:nva, : va.shape[1]] = cav.T @ overlap @ va
        domain[nva:, va.shape[1] :] = cbv.T @ overlap @ vb
        if domain.shape[1] == 0:
            raise ValueError(
                f"empty PAO virtual domain for occupied pair {(p, q)} at "
                f"tcut_mkn={options.tcut_mkn}"
            )
        return np.ascontiguousarray(domain)

    def occupied_pao_basis(p: int) -> np.ndarray | None:
        """Return one occupied's same-spin PAOs in spin-virtual coordinates."""
        if options.tcut_mkn <= 0.0:
            return None
        atoms = select_domain_atoms_mulliken(
            c_occ,
            overlap,
            atom_first,
            p,
            p,
            options.tcut_mkn,
        )
        if atoms.size == 0:
            raise ValueError(
                f"empty singles PAO atom domain for occupied {p} at "
                f"tcut_mkn={options.tcut_mkn}"
            )
        ao_indices = np.concatenate(
            [
                np.arange(atom_first[atom], atom_first[atom + 1])
                for atom in atoms
            ]
        )
        if p < noa:
            pao, _ = semicanonical_pao_basis(
                fa_ao,
                overlap,
                q_alpha,
                ao_indices,
                options.lindep,
            )
            domain = np.zeros((nv, pao.shape[1]))
            domain[:nva] = cav.T @ overlap @ pao
        else:
            pao, _ = semicanonical_pao_basis(
                fb_ao,
                overlap,
                q_beta,
                ao_indices,
                options.lindep,
            )
            domain = np.zeros((nv, pao.shape[1]))
            domain[nva:] = cbv.T @ overlap @ pao
        if domain.shape[1] == 0:
            raise ValueError(
                f"empty singles PAO virtual domain for occupied {p} at "
                f"tcut_mkn={options.tcut_mkn}"
            )
        return np.ascontiguousarray(domain)

    for p, q in all_pair_keys:
        coulomb = b_ov[:, p, :].T @ b_ov[:, q, :]
        anti = coulomb - coulomb.T
        denominator = (
            eps_o[p] + eps_o[q] - eps_v[:, None] - eps_v[None, :]
        )
        t_mp2 = anti / denominator
        if singles_density is not None:
            # Occupied-specific MP2 virtual densities. For the ordered pair
            # orientation p < q, the first virtual index belongs to p and the
            # second to q. Summing these pair contributions gives the
            # singles-specific natural-orbital spaces of Riplinger and Neese,
            # J. Chem. Phys. 138, 034106 (2013), doi:10.1063/1.4773581.
            singles_density[p] += t_mp2 @ t_mp2.T
            singles_density[q] += t_mp2.T @ t_mp2
        # The energy of one unique spin-orbital pair carries 1/2; summing both
        # occupied orders in the conventional expression would carry 1/4.
        e_mp2 = 0.5 * float(np.sum(anti * t_mp2))
        pair_mp2_energy[(p, q)] = e_mp2
        if options.tcut_pairs > 0.0 and abs(e_mp2) < options.tcut_pairs:
            screened_pairs.add((p, q))
            continue
        pao = pair_pao_basis(p, q)
        if pao is None:
            pair_pao_size[(p, q)] = nv
            anti_pao = anti
            f_pao = f_vv
        else:
            pair_pao_size[(p, q)] = pao.shape[1]
            anti_pao = pao.T @ anti @ pao
            f_pao = pao.T @ f_vv @ pao
        eps_pao = np.diag(f_pao)
        denominator_pao = (
            eps_o[p]
            + eps_o[q]
            - eps_pao[:, None]
            - eps_pao[None, :]
        )
        t_mp2_pao = anti_pao / denominator_pao
        density = t_mp2_pao @ t_mp2_pao.T + t_mp2_pao.T @ t_mp2_pao
        density = 0.5 * (density + density.T)
        d, ep = _domain_from_density(density, f_pao, options.tcut_pno)
        u = d if pao is None else np.ascontiguousarray(pao @ d)
        kp = u.T @ anti @ u
        pair_u[(p, q)] = u
        pair_eps[(p, q)] = ep
        pair_k[(p, q)] = kp
        t2[(p, q)] = kp / (
            eps_o[p] + eps_o[q] - ep[:, None] - ep[None, :]
        )

    pair_keys = [key for key in all_pair_keys if key not in screened_pairs]

    def triple_is_screened(i: int, j: int, k: int) -> bool:
        """Would local screening drop the LMO triple ``(i, j, k)``?

        Weak-pair and occupied-distance screening are both defined on the
        *localised* occupied indices: a triple survives only when none of its
        three pairs was screened out of the CCSD iteration and all three
        occupied centroids lie inside ``coupling_radius``.
        """
        if {(i, j), (i, k), (j, k)} & screened_pairs:
            return True
        return radius > 0.0 and (
            distances[i, j] >= radius
            or distances[i, k] >= radius
            or distances[j, k] >= radius
        )

    # (T1) rotates the occupied indices into the per-spin canonical basis, in
    # which a "distance" or a weak-pair label has no meaning: each canonical
    # occupied is a mixture of every LMO. So a screened triple list cannot be
    # carried through the rotation, and this route fails closed whenever the
    # requested thresholds actually remove one. Deciding on the realised set
    # rather than on the raw threshold values keeps the exact full-list (T1)
    # reachable when the thresholds happen to screen nothing -- which is what
    # a generous coupling_radius does, though not a positive tcut_pairs on any
    # system carrying symmetry-zero same-spin pair energies (see
    # tests/test_dlpno_uccsd_local.py). Recovering the screened local (T1)
    # itself needs an iterative local-basis treatment. ``t1-iterative`` uses a
    # spin-orbital generalization of Guo et al. 2018 Eq. (2), coupling
    # surviving triples through the off-diagonal occupied Fock *inside* the
    # LMO triple list instead of rotating the list away. Guo et al. 2020 is
    # the dedicated open-shell method-family source.
    screened_triples = (
        [
            (i, j, k)
            for i in range(no)
            for j in range(i + 1, no)
            for k in range(j + 1, no)
            if triple_is_screened(i, j, k)
        ]
        if options.compute_triples and triples_mode == "t1" and no >= 3
        else []
    )
    if screened_triples:
        raise NotImplementedError(
            "open-shell local (T1) does not support weak-pair or "
            "occupied-distance triple screening: the requested "
            f"tcut_pairs={options.tcut_pairs:g} / "
            f"coupling_radius={options.coupling_radius:g} screen "
            f"{len(screened_triples)} of "
            f"{no * (no - 1) * (no - 2) // 6} occupied triples, and the "
            "canonical occupied rotation cannot represent that list. Use "
            'triples_mode="t0" for a screened semicanonical run, or '
            '"t1-iterative", which keeps the triples in the localised basis '
            "and so can carry a screened list, or drop the screening "
            "thresholds."
        )

    singles_u: dict[int, np.ndarray] = {}
    singles_eps: dict[int, np.ndarray] = {}
    t1: dict[int, np.ndarray] = {}
    for p in range(no):
        if singles_density is None:
            u, ep = _domain_from_density(np.eye(nv), f_vv, 0.0)
        else:
            pao = occupied_pao_basis(p)
            if pao is None:
                density = singles_density[p]
                f_singles = f_vv
            else:
                density = pao.T @ singles_density[p] @ pao
                f_singles = pao.T @ f_vv @ pao
            d, ep = _domain_from_density(
                density,
                f_singles,
                options.tcut_pno_singles,
            )
            u = d if pao is None else np.ascontiguousarray(pao @ d)
        singles_u[p] = u
        singles_eps[p] = ep
        t1[p] = np.zeros(u.shape[1])

    result.n_pairs = len(pair_keys)
    result.n_screened = len(screened_pairs)
    result.screened_pair_energies = {
        key: pair_mp2_energy[key] for key in sorted(screened_pairs)
    }
    result.e_screened_mp2 = float(
        sum(result.screened_pair_energies.values())
    )
    result.pno_per_pair = {key: pair_u[key].shape[1] for key in pair_keys}
    result.pao_per_pair = {key: pair_pao_size[key] for key in pair_keys}
    result.singles_per_occ = {p: singles_u[p].shape[1] for p in range(no)}
    result.avg_pno = (
        float(np.mean(list(result.pno_per_pair.values())))
        if result.pno_per_pair
        else 0.0
    )
    result.avg_pao = (
        float(np.mean(list(result.pao_per_pair.values())))
        if result.pao_per_pair
        else 0.0
    )
    result.avg_singles_domain = float(
        np.mean(list(result.singles_per_occ.values()))
    )

    def source_t2(m: int, n: int) -> tuple[np.ndarray, np.ndarray] | None:
        if m == n:
            return None
        key = (m, n) if m < n else (n, m)
        if key in screened_pairs:
            return None
        sign = 1.0 if m < n else -1.0
        return pair_u[key], sign * t2[key]

    def residual_domain(
        required: np.ndarray,
        occupieds: np.ndarray,
    ) -> np.ndarray:
        """Union pair PNOs needed by a local residual contraction.

        DLPNO residual intermediates need a larger virtual span than the
        target amplitude alone.  Form the orthonormal union of pair domains
        in the local occupied set, always including the target.  Compact
        full-coupling systems may recover the full virtual span; extended
        systems retain a bounded domain once occupied locality is enabled.
        """
        pieces = [required]
        for a, m in enumerate(occupieds):
            for n in occupieds[a + 1 :]:
                pair = pair_u.get((int(m), int(n)))
                if pair is not None:
                    pieces.append(pair)
        stacked = np.hstack(pieces)
        left, singular, _ = np.linalg.svd(stacked, full_matrices=False)
        if singular.size == 0:
            return required
        cutoff = max(1e-12, 1e-10 * singular[0])
        rank = max(required.shape[1], int(np.count_nonzero(singular > cutoff)))
        return np.ascontiguousarray(left[:, : min(rank, nv)])

    def local_problem(
        target_u: np.ndarray,
        occupieds: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Assemble amplitudes, DF tensor, and Fock in one target domain."""
        nl = len(occupieds)
        nd = target_u.shape[1]
        t1_local = np.zeros((nl, nd))
        t2_local = np.zeros((nl, nl, nd, nd))
        for a, m in enumerate(occupieds):
            overlap = target_u.T @ singles_u[m]
            t1_local[a] = overlap @ t1[m]
            for b, n in enumerate(occupieds):
                source = source_t2(m, n)
                if source is None:
                    continue
                source_u, source_amp = source
                pair_overlap = target_u.T @ source_u
                t2_local[a, b] = (
                    pair_overlap @ source_amp @ pair_overlap.T
                )

        bov = np.einsum(
            "Piv,va->Pia", b_ov[:, occupieds, :], target_u, optimize=True
        )
        bvv = np.einsum(
            "Puv,ua,vb->Pab", b_vv, target_u, target_u, optimize=True
        )
        size = nl + nd
        b_local = np.zeros((naux, size, size))
        b_local[:, :nl, :nl] = b_oo[:, occupieds][:, :, occupieds]
        b_local[:, :nl, nl:] = bov
        b_local[:, nl:, :nl] = bov.transpose(0, 2, 1)
        b_local[:, nl:, nl:] = bvv

        fov = f_ov[occupieds] @ target_u
        f_local = np.zeros((size, size))
        f_local[:nl, :nl] = f_oo[np.ix_(occupieds, occupieds)]
        f_local[:nl, nl:] = fov
        f_local[nl:, :nl] = fov.T
        f_local[nl:, nl:] = target_u.T @ f_vv @ target_u
        return (
            np.ascontiguousarray(t1_local),
            np.ascontiguousarray(t2_local.reshape(nl * nl, nd * nd)),
            np.ascontiguousarray(b_local.reshape(naux, size * size)),
            np.ascontiguousarray(f_local),
        )

    def energy() -> float:
        value = result.e_screened_mp2
        for p in range(no):
            value += float(np.dot(f_ov[p] @ singles_u[p], t1[p]))
        for p, q in pair_keys:
            u = pair_u[(p, q)]
            tp = (u.T @ singles_u[p]) @ t1[p]
            tq = (u.T @ singles_u[q]) @ t1[q]
            k = pair_k[(p, q)]
            value += 0.5 * float(np.sum(k * t2[(p, q)]))
            value += float(np.sum(k * np.outer(tp, tq)))
        return value

    def flatten() -> np.ndarray:
        return np.concatenate(
            [t1[p].ravel() for p in range(no)]
            + [t2[key].ravel() for key in pair_keys]
        )

    def unflatten(vector: np.ndarray) -> None:
        offset = 0
        for p in range(no):
            count = t1[p].size
            t1[p] = vector[offset : offset + count].copy()
            offset += count
        for key in pair_keys:
            count = t2[key].size
            t2[key] = vector[offset : offset + count].reshape(
                t2[key].shape
            ).copy()
            offset += count

    amplitude_history: list[np.ndarray] = []
    residual_history: list[np.ndarray] = []
    coupled_sizes: list[int] = []
    e_prev = energy()
    for iteration in range(options.max_iter):
        old = flatten()
        new_t1: dict[int, np.ndarray] = {}
        new_t2: dict[tuple[int, int], np.ndarray] = {}

        for p in range(no):
            occupieds = coupled_occupieds(p)
            coupled_sizes.append(len(occupieds))
            domain = residual_domain(singles_u[p], occupieds)
            local = local_problem(domain, occupieds)
            r1, _ = dlpno_uccsd_pair_residual(*local)
            lp = int(np.searchsorted(occupieds, p))
            projected_r1 = (singles_u[p].T @ domain) @ np.asarray(r1)[lp]
            step = projected_r1 / (eps_o[p] - singles_eps[p])
            new_t1[p] = t1[p] + step

        for p, q in pair_keys:
            occupieds = coupled_occupieds(p, q)
            coupled_sizes.append(len(occupieds))
            domain = residual_domain(pair_u[(p, q)], occupieds)
            local = local_problem(domain, occupieds)
            _, r2_flat = dlpno_uccsd_pair_residual(*local)
            nd = domain.shape[1]
            r2 = np.asarray(r2_flat).reshape(
                len(occupieds), len(occupieds), nd, nd
            )
            lp = int(np.searchsorted(occupieds, p))
            lq = int(np.searchsorted(occupieds, q))
            denominator = (
                eps_o[p]
                + eps_o[q]
                - pair_eps[(p, q)][:, None]
                - pair_eps[(p, q)][None, :]
            )
            projection = pair_u[(p, q)].T @ domain
            projected_r2 = projection @ r2[lp, lq] @ projection.T
            updated = t2[(p, q)] + projected_r2 / denominator
            new_t2[(p, q)] = 0.5 * (updated - updated.T)

        for p in range(no):
            t1[p] = new_t1[p]
        for key in pair_keys:
            t2[key] = new_t2[key]

        amplitude_history.append(flatten())
        residual_history.append(amplitude_history[-1] - old)
        if len(amplitude_history) > options.diis_size:
            amplitude_history.pop(0)
            residual_history.pop(0)
        nh = len(amplitude_history)
        if nh >= 2:
            metric = np.full((nh + 1, nh + 1), -1.0)
            metric[-1, -1] = 0.0
            for a in range(nh):
                for b in range(nh):
                    metric[a, b] = float(
                        np.dot(residual_history[a], residual_history[b])
                    )
            rhs = np.zeros(nh + 1)
            rhs[-1] = -1.0
            try:
                weights = np.linalg.solve(metric, rhs)[:nh]
                unflatten(
                    sum(
                        weights[k] * amplitude_history[k]
                        for k in range(nh)
                    )
                )
            except np.linalg.LinAlgError:
                pass

        e_corr = energy()
        residual_norm = float(np.max(np.abs(residual_history[-1])))
        result.trace.append(
            {
                "iter": iteration + 1,
                "e_corr": e_corr,
                "delta_e": e_corr - e_prev,
                "r_norm": residual_norm,
            }
        )
        if (
            iteration > 0
            and abs(e_corr - e_prev) < options.conv_tol_energy
            and residual_norm < options.conv_tol_residual
        ):
            result.converged = True
            result.n_iter = iteration + 1
            break
        e_prev = e_corr
    else:
        result.n_iter = options.max_iter

    result.e_corr = energy()
    result.t1_norm = float(
        np.sqrt(sum(float(np.dot(amplitude, amplitude)) for amplitude in t1.values()))
    )
    result.avg_coupled_occ = float(np.mean(coupled_sizes)) if coupled_sizes else 0.0
    if options.compute_triples and result.converged and no >= 3:
        t1_full: np.ndarray | None = None
        t2_full: np.ndarray | None = None
        b_ov_triples = b_ov
        b_oo_triples = b_oo
        eps_o_triples = eps_o
        if triples_mode == "t1":
            t1_local = np.zeros((no, nv))
            for m in range(no):
                t1_local[m] = singles_u[m] @ t1[m]
            t2_local = np.zeros((no, no, nv, nv))
            for m in range(no):
                for n in range(no):
                    source = source_t2(m, n)
                    if source is None:
                        continue
                    source_u, source_amp = source
                    t2_local[m, n] = source_u @ source_amp @ source_u.T

            rotation = np.zeros((no, no))
            eps_o_triples = np.zeros(no)
            for occupied_slice in (slice(0, noa), slice(noa, no)):
                f_spin = f_oo[occupied_slice, occupied_slice]
                eps_spin, rotation_spin = np.linalg.eigh(
                    0.5 * (f_spin + f_spin.T)
                )
                rotation[occupied_slice, occupied_slice] = rotation_spin
                eps_o_triples[occupied_slice] = eps_spin
            t1_full = np.ascontiguousarray(rotation.T @ t1_local)
            t2_full = np.ascontiguousarray(
                np.einsum(
                    "mM,nN,mnuv->MNuv",
                    rotation,
                    rotation,
                    t2_local,
                    optimize=True,
                )
            )
            b_ov_triples = np.ascontiguousarray(
                np.einsum("Pmv,mM->PMv", b_ov, rotation, optimize=True)
            )
            b_oo_triples = np.ascontiguousarray(
                np.einsum(
                    "Pmn,mM,nN->PMN",
                    b_oo,
                    rotation,
                    rotation,
                    optimize=True,
                )
            )

        # The rotated-occupied ``t1`` route reaches this point only when the
        # screened set is empty (the guard above fails closed otherwise).
        # ``t0`` and ``t1-iterative`` retain this realised local triple list.
        triples: list[tuple[int, int, int]] = []
        for i in range(no):
            for j in range(i + 1, no):
                for k in range(j + 1, no):
                    if triple_is_screened(i, j, k):
                        result.n_screened_triples += 1
                    else:
                        triples.append((i, j, k))

        def triple_domain(i: int, j: int, k: int) -> tuple[np.ndarray, np.ndarray]:
            """Build and semicanonicalise one triple's TNO domain."""
            if t2_full is not None:
                density = np.zeros((nv, nv))
                for p, q in ((i, j), (i, k), (j, k)):
                    amplitude = t2_full[p, q]
                    density += amplitude @ amplitude.T + amplitude.T @ amplitude
                return _domain_from_density(
                    density,
                    f_vv,
                    options.tcut_tno,
                )

            pairs = [(i, j), (i, k), (j, k)]
            stacked = np.hstack([pair_u[key] for key in pairs])
            left, singular, _ = np.linalg.svd(stacked, full_matrices=False)
            cutoff = options.lindep * max(float(singular[0]), 1e-300)
            rank = max(1, int(np.count_nonzero(singular > cutoff)))
            precursor = np.ascontiguousarray(left[:, :rank])
            density = np.zeros((rank, rank))
            for key in pairs:
                overlap_t = precursor.T @ pair_u[key]
                amplitude = overlap_t @ t2[key] @ overlap_t.T
                density += amplitude @ amplitude.T + amplitude.T @ amplitude
            vectors, eps_t = _domain_from_density(
                density,
                precursor.T @ f_vv @ precursor,
                options.tcut_tno,
            )
            return np.ascontiguousarray(precursor @ vectors), eps_t

        if triples_mode == "t1-iterative":
            # Spin-orbital generalization of Guo et al. (2018) Eq. (2) in the
            # LOCALISED occupied basis. Guo et al. (2020) is the dedicated
            # open-shell method-family source. Iterating the off-diagonal
            # occupied-Fock coupling preserves the screened triple list,
            # unlike the canonical-rotation "t1" route.
            from .triples_iterative import iterative_t1_triples_correction

            tno_domains = {}
            for key in triples:
                vectors, eps_key = triple_domain(*key)
                tno_domains[key] = (vectors, eps_key)
                result.tno_per_triple[key] = int(vectors.shape[1])
            b_ov_full = b_ov
            eri_vovv_f = np.einsum(
                "Peb,Pic->eibc", b_vv, b_ov_full, optimize=True
            ) - np.einsum("Pec,Pib->eibc", b_vv, b_ov_full, optimize=True)
            eri_ovoo_f = np.einsum(
                "Pmj,Pka->majk", b_oo, b_ov_full, optimize=True
            ) - np.einsum("Pmk,Pja->majk", b_oo, b_ov_full, optimize=True)
            eri_oovv_f = np.einsum(
                "Pjb,Pkc->jkbc", b_ov_full, b_ov_full, optimize=True
            ) - np.einsum("Pjc,Pkb->jkbc", b_ov_full, b_ov_full, optimize=True)
            t1_it = np.zeros((no, nv))
            for m in range(no):
                t1_it[m] = singles_u[m] @ t1[m]
            t2_it = np.zeros((no, no, nv, nv))
            for m in range(no):
                for n in range(no):
                    source = source_t2(m, n)
                    if source is None:
                        continue
                    source_u, source_amp = source
                    t2_it[m, n] = source_u @ source_amp @ source_u.T
            e_t_iter, n_sweeps, converged_iter = (
                iterative_t1_triples_correction(
                    t1_it,
                    t2_it,
                    eri_vovv_f,
                    eri_ovoo_f,
                    eri_oovv_f,
                    f_oo,
                    eps_v,
                    triple_list=triples,
                    tno_domains=tno_domains,
                )
            )
            if not converged_iter:
                raise RuntimeError(
                    "iterative (T1) triples did not converge in "
                    f"{n_sweeps} sweeps; use the unscreened rotated "
                    'triples_mode="t1" route when applicable, or fall back '
                    'to triples_mode="t0"'
                )
            result.e_t = e_t_iter
            result.n_triples = len(triples)
            result.avg_tno = (
                float(np.mean(list(result.tno_per_triple.values())))
                if result.tno_per_triple
                else float(nv)
            )
            result.n_degenerate_tno_triples = sum(
                1 for s in result.tno_per_triple.values() if s < 3
            )
            result.triples_executed = bool(triples)
            result.e_total = result.e_hf + result.e_corr + result.e_t
            return result

        for i, j, k in triples:
            tno, eps_t = triple_domain(i, j, k)
            nt = tno.shape[1]

            if t1_full is not None and t2_full is not None:
                t1_t = np.ascontiguousarray(t1_full @ tno)
                t2_t = np.ascontiguousarray(
                    np.einsum(
                        "mnuv,ua,vb->mnab",
                        t2_full,
                        tno,
                        tno,
                        optimize=True,
                    )
                )
            else:
                t1_t = np.zeros((no, nt))
                for m in range(no):
                    t1_t[m] = (tno.T @ singles_u[m]) @ t1[m]
                t2_t = np.zeros((no, no, nt, nt))
                for m in range(no):
                    for n in range(no):
                        source = source_t2(m, n)
                        if source is None:
                            continue
                        source_u, source_amp = source
                        overlap_t = tno.T @ source_u
                        t2_t[m, n] = overlap_t @ source_amp @ overlap_t.T

            bov_t = np.einsum(
                "Piv,va->Pia", b_ov_triples, tno, optimize=True
            )
            bvv_t = np.einsum("Puv,ua,vb->Pab", b_vv, tno, tno, optimize=True)
            eri_vovv = np.einsum(
                "Peb,Pic->eibc", bvv_t, bov_t, optimize=True
            ) - np.einsum("Pec,Pib->eibc", bvv_t, bov_t, optimize=True)
            eri_ovoo = np.einsum(
                "Pmj,Pka->majk", b_oo_triples, bov_t, optimize=True
            ) - np.einsum(
                "Pmk,Pja->majk", b_oo_triples, bov_t, optimize=True
            )
            eri_oovv = np.einsum(
                "Pjb,Pkc->jkbc", bov_t, bov_t, optimize=True
            ) - np.einsum("Pjc,Pkb->jkbc", bov_t, bov_t, optimize=True)
            result.e_t += float(
                dlpno_spin_orbital_triple_energy(
                    i,
                    j,
                    k,
                    np.ascontiguousarray(t1_t),
                    np.ascontiguousarray(t2_t.reshape(no * no, nt * nt)),
                    np.ascontiguousarray(eri_vovv.reshape(nt * no, nt * nt)),
                    np.ascontiguousarray(eri_ovoo.reshape(no * nt, no * no)),
                    np.ascontiguousarray(eri_oovv.reshape(no * no, nt * nt)),
                    np.ascontiguousarray(eps_o_triples),
                    np.ascontiguousarray(eps_t),
                )
            )
            result.tno_per_triple[(i, j, k)] = nt
        result.n_triples = len(triples)
        result.avg_tno = (
            float(np.mean(list(result.tno_per_triple.values())))
            if result.tno_per_triple
            else 0.0
        )
        # A triple excitation promotes three electrons into three *distinct*
        # spin virtuals, so a TNO domain holding fewer than three cannot host
        # one and its energy is identically zero. Counting those separates
        # "tcut_tno truncated a little" from "tcut_tno truncated past the point
        # where the correction exists", which the energy alone cannot show.
        result.n_degenerate_tno_triples = sum(
            1 for size in result.tno_per_triple.values() if size < 3
        )
        result.triples_executed = bool(triples)
    result.e_total = result.e_hf + result.e_corr + result.e_t
    return result
