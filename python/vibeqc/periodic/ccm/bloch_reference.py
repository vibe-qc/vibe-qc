"""The Bloch producer's SCF as a real supercell-Γ reference (Theorem 1, executable).

Ruling R1 (2026-08-21): Γ-CCM is the neutral finite-Born-von-Karman-torus
construction, with two producers of one Hamiltonian. The Bloch producer
(:func:`~vibeqc.periodic.ccm.ri.run_ccm_rhf_gdf` into the multi-k GDF driver)
diagonalises ``N_k`` matrices of size ``n`` on the unit cell's Γ-centred
``nrep`` mesh; the real-Γ producer (:func:`~vibeqc.periodic.ccm.direct.
run_ccm_rhf_direct`) diagonalises one matrix of size ``N_c × n`` on the
supercell. Paper 1, Theorem 1 (M. F. Peintinger, the Γ-CCM paper, in
preparation; pinned numerically in ``tests/test_ccm_theorem1_3d.py``) states
that the two are related exactly by the finite Fourier transform of the torus:

    U_{kR} = N_c^{-1/2} e^{i k·R_c},   M_sc[(c,μ),(c',ν)] = N_c^{-1} Σ_k e^{ik(R_c − R_c')} M_k[μ,ν]

for every block-circulant operator ``M`` (overlap, core Hamiltonian, Fock,
density), and the per-k orbitals ``ψ_{p,k} = N_c^{-1/2} e^{ik·R_c} C_k[μ,p]``
are eigenfunctions of the folded Fock operator with the per-k eigenvalues.

This module makes that executable so the real-space correlated methods of
this package (:func:`~vibeqc.periodic.ccm.mp2.run_ccm_mp2`,
:func:`~vibeqc.periodic.ccm.dlpno.ccm_dlpno_mp2`,
:func:`~vibeqc.periodic.ccm.ccsd.run_ccm_ccsd` and their siblings) can
correlate on the Bloch producer's SCF, which is the cheaper of the two on
every cell measured (3 to 12 times at (2,2,2)) and the better converged.
:func:`ccm_reference_from_bloch` takes the producer's result and returns the
same object the direct route returns (:class:`~vibeqc.periodic.ccm.scf.
CCMSCFResult`, or :class:`~vibeqc.periodic.ccm.uhf.CCMUHFResult` for the
open-shell producers), carrying **real** S-orthonormal canonical supercell
orbitals, the folded Fock/overlap/core Hamiltonian, the density, the total
cyclic energies and, by default, the neutral cderi of the same fitted
operator (consumed from the driver's retained fit when the SCF ran with
``retain_fit_cache=True``, else refitted by the fold).

Realification. The complex Bloch orbitals come in time-reversal pairs
``ψ_{p,-k} = ψ_{p,k}^*``. A mesh point with ``k ≡ −k`` (a TRIM, the
prototype's "trim" kind) is self-paired: its eigenspaces are closed under
complex conjugation. Every other point is grouped with its partner ``−k``
(the "pair" kind) and the union of the two eigenspaces is closed. Within
each such group the orbitals are cut into blocks of (near-)degenerate
eigenvalues; each block is replaced by real S-orthonormal orbitals spanning
the same space (the real and imaginary parts, Löwdin-orthonormalised over
their Gram matrix), and the folded Fock operator is then re-diagonalised
inside the block so the orbitals are exactly canonical even when the block
was only nearly degenerate. Degeneracies **across** groups (point-group
partners such as the three X points of a cubic (2,2,2) mesh) need no
treatment: orbitals of different ``±k`` groups are S-orthogonal by
translational symmetry, and each is an eigenvector of the folded Fock
operator on its own.

Every claim above is checked on the produced object, with the tolerances in
:data:`BLOCH_REFERENCE_TOLERANCES`, and a failure raises rather than
returning a reference that would silently correlate wrongly. The prototype
that these checks come from (the paper campaign's ``bloch_transform.py``, kept
with the private campaign material) measured on LiH/STO-3G (2,2,1): folded-Fock
off-diagonal 1.7e-15, MP2 on the transformed orbitals equal to real-Γ MP2
to 5e-10 and DLPNO-MP2 to 6e-10.

The real-Γ producer stays the independent validation producer: nothing here
replaces :func:`~vibeqc.periodic.ccm.direct.run_ccm_rhf_direct`, and
``tests/test_ccm_bloch_reference.py`` pins the correlation energies from
the two references against each other at 1e-9 Ha/cell.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

__all__ = [
    "BLOCH_REFERENCE_SCF_CONV_TOL_GRAD",
    "BLOCH_REFERENCE_TOLERANCES",
    "BlochFoldReport",
    "as_real_reference",
    "bloch_reference_and_cderi",
    "ccm_reference_from_bloch",
    "fold_block_circulant",
    "is_bloch_result",
    "real_supercell_orbitals",
]

#: Tolerances of the checks :func:`ccm_reference_from_bloch` performs on the
#: reference it returns (absolute; Hartree where the quantity is an energy).
#: Each is a factor 1e2 to 1e6 above what the prototype measured, and a
#: factor 10 or more below what would move a correlation energy at the
#: 1e-9 Ha/cell class the identity tests pin.
BLOCH_REFERENCE_TOLERANCES = {
    # max |Im| of a folded operator (S, h, F, D): the mesh's time-reversal
    # closure makes every fold real; measured 1e-15 class.
    "fold_imag": 1e-10,
    # max |C^T S C − 1| of the real supercell orbitals.
    "orthonormality": 1e-8,
    # max off-diagonal |C^T F C| (canonical orbitals of the folded Fock).
    "fock_offdiag": 1e-8,
    # max |ε(block re-diagonalisation) − ε(driver)|, per orbital.
    "eigenvalue": 1e-8,
    # max |D(occupied orbitals) − fold(driver density)|.
    "density": 1e-8,
    # |Tr(D S) − N_e| on the supercell.
    "electron_count": 1e-8,
    # |occ − round(occ)| of every orbital, occ = ⟨ψ|S D S|ψ⟩ (integer
    # aufbau; a fractional value means smearing or a block straddling the
    # Fermi level, and the reference is refused).
    "occupation": 1e-8,
    # relative Gram-eigenvalue threshold of the realification.
    "realify": 1e-8,
}

#: Orbital-gradient convergence (the multi-k ``conv_tol_grad``) of the Bloch
#: SCF behind a reference. The ``density`` check compares the density of the
#: occupied orbitals with the density the driver returns, and the two differ
#: by the SCF's last step: at the multi-k default (1e-6) the check refused
#: LiH/pob-DZVP-rev2 (1,1,1) at 4.0e-7 and MgO/pob-DZVP-rev2 (1,1,1) at
#: 1.7e-7, while LiH at this value passed and its real-Γ READ restart agreed
#: with the Bloch energy to 1.6e-12 Ha/cell (2026-10-19, main 780bec8).
#: :func:`bloch_reference_and_cderi` runs the Bloch SCF at this value unless
#: the caller passes ``options``; ``tests/test_ccm_bloch_reference.py``
#: converges its references to it as well.
BLOCH_REFERENCE_SCF_CONV_TOL_GRAD = 1e-8

#: Orbitals within one ``±k`` group closer than this (Ha) are treated as one
#: degenerate block. Larger blocks are always safe (the block is re-
#: diagonalised), smaller ones are cheaper.
DEGENERACY_TOL = 1e-7


@dataclass(frozen=True)
class BlochFoldReport:
    """What the transform did and what it measured, stamped on the result.

    Every residual below was compared against :attr:`tolerances`; the
    reference was refused if any exceeded it, so a report on a returned
    reference is a record of passed checks.
    """

    n_k: int
    n_cells: int
    n_orbitals: int
    #: Per-spin occupied counts: ``(n_occ,)`` closed-shell, ``(n_alpha,
    #: n_beta)`` open-shell.
    n_occupied: tuple
    #: ``{"trim": ..., "pair": ...}``: orbitals from self-paired mesh points
    #: (``k ≡ −k``) and from ``±k`` pairs.
    kinds: dict
    n_blocks: int
    max_block: int
    degeneracy_tol: float
    fold_imag_max: float
    orthonormality_max_dev: float
    fock_offdiag_max: float
    eigenvalue_shift_max: float
    density_max_dev: float
    electron_count_dev: float
    occupation_max_dev: float
    tolerances: dict
    source_backend: str
    source_fit_source: str
    source_achieved_convergence: dict
    #: ``cderi_info["build"]`` of the cderi the reference carries (``None``
    #: when none was built).
    cderi_build: Optional[str]
    #: ``⟨S²⟩`` of the transformed determinant (open shell only).
    s_squared: Optional[float] = None


def _tolerances(overrides):
    tol = dict(BLOCH_REFERENCE_TOLERANCES)
    if overrides:
        unknown = set(overrides) - set(tol)
        if unknown:
            raise ValueError(f"unknown tolerance key(s): {sorted(unknown)}")
        tol.update({k: float(v) for k, v in overrides.items()})
    return tol


def _cells_and_phases(ccm, kpts):
    """Cell offsets ``R_c`` in the supercell builder's order and ``e^{ik·R_c}``.

    The order is ``for i: for j: for k`` over ``nrep``, exactly
    :meth:`~vibeqc.periodic.ccm.system.CCMSystem._build_supercell`'s, so a
    folded matrix indexed ``(c, μ)`` is in the supercell's own AO order.
    """
    n1, n2, n3 = (int(n) for n in ccm.nrep)
    cells = np.array([(i, j, k) for i in range(n1) for j in range(n2)
                      for k in range(n3)], dtype=int)
    a_rows = np.asarray(ccm.unit_vectors, dtype=float)      # rows = a_j
    r_c = cells @ a_rows                                     # (N_c, 3)
    phase = np.exp(1j * (r_c @ np.asarray(kpts, dtype=float).T))   # (N_c, n_k)
    return cells, r_c, phase


def _mesh_labels(ccm, kpts):
    """Integer mesh labels of the k-points and the index of each one's ``−k``.

    Raises when the points are not exactly the Γ-centred ``nrep`` mesh of the
    torus (a shifted or symmetry-reduced mesh has no fold).
    """
    a_rows = np.asarray(ccm.unit_vectors, dtype=float)
    nrep = np.asarray(ccm.nrep, dtype=int)
    kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
    frac = (kpts @ a_rows.T) / (2.0 * np.pi) * nrep[None, :]
    lab = np.rint(frac).astype(int)
    off = float(np.max(np.abs(frac - lab))) if len(kpts) else 0.0
    if off > 1e-8:
        raise ValueError(
            "ccm_reference_from_bloch: the k-points are not on the Γ-centred "
            f"nrep={tuple(int(n) for n in nrep)} mesh of the torus (max "
            f"fractional offset {off:.2e}); a shifted mesh has no fold.")
    lab = lab % nrep[None, :]
    index = {tuple(int(x) for x in row): i for i, row in enumerate(lab)}
    n_c = int(np.prod(nrep))
    if len(index) != len(lab) or len(lab) != n_c:
        raise ValueError(
            "ccm_reference_from_bloch: expected the full Γ-centred mesh of "
            f"{n_c} distinct points for nrep={tuple(int(n) for n in nrep)}, "
            f"got {len(lab)} points ({len(index)} distinct); a symmetry-"
            "reduced mesh cannot be folded.")
    partner = np.array([index[tuple(int(x) for x in ((-row) % nrep))]
                        for row in lab], dtype=int)
    return lab, partner


def fold_block_circulant(ccm, kpts, mats_k, *, tol=None):
    """``M_sc[(c,μ),(c',ν)] = N_c^{-1} Σ_k e^{ik(R_c − R_c')} M_k[μ,ν]`` (real).

    The inverse finite Fourier transform of a per-k operator onto the
    supercell, in the supercell's AO order. Returns ``(M_sc, max_imag)``;
    raises when the fold is not real to ``tol`` (default
    ``BLOCH_REFERENCE_TOLERANCES["fold_imag"]``), which happens only when the
    per-k matrices are not time-reversal closed (``M_{−k} ≠ M_k^*``).
    """
    tol = BLOCH_REFERENCE_TOLERANCES["fold_imag"] if tol is None else float(tol)
    _, _, ph = _cells_and_phases(ccm, kpts)
    n_c, n_k = ph.shape
    mats = np.asarray([np.asarray(m) for m in mats_k])
    if mats.ndim != 3 or mats.shape[0] != n_k:
        raise ValueError(
            f"fold_block_circulant: expected {n_k} square matrices, got an "
            f"array of shape {mats.shape}")
    n_mu = int(mats.shape[1])
    out = np.einsum("ck,dk,kmn->cmdn", ph, ph.conj(), mats, optimize=True)
    out = out.reshape(n_c * n_mu, n_c * n_mu) / n_c
    imag = float(np.max(np.abs(out.imag))) if out.size else 0.0
    if imag > tol:
        raise ValueError(
            f"fold_block_circulant: folded operator is not real (max |Im| "
            f"{imag:.2e} > {tol:.1e}); the per-k matrices are not time-"
            "reversal closed, so this is not a block-circulant operator of "
            "the torus.")
    return np.ascontiguousarray(out.real), imag


def _realify_block(psi, S_sc, tol):
    """Real S-orthonormal orbitals spanning the same space as ``psi``.

    Real and imaginary parts are stacked and Löwdin-orthonormalised over
    their Gram matrix; the retained count must equal the block size, which
    fails loudly when the block is not closed under complex conjugation (a
    full degenerate block of a time-reversal-symmetric Hamiltonian, taken
    over ``k`` and ``−k`` together, always is).
    """
    V = np.concatenate([psi.real, psi.imag], axis=1)
    G = V.T @ S_sc @ V
    w, U = np.linalg.eigh(G)
    keep = w > tol * max(1.0, float(w.max()))
    n_keep = int(np.sum(keep))
    if n_keep != psi.shape[1]:
        raise ValueError(
            f"realification: a block of {psi.shape[1]} Bloch orbitals spans "
            f"{n_keep} real directions; the block is not closed under "
            "complex conjugation (an eigenvalue window that cuts a "
            "time-reversal pair, or per-k orbitals that are not an "
            "eigenbasis of a block-circulant operator).")
    return V @ (U[:, keep] / np.sqrt(w[keep]))


def real_supercell_orbitals(ccm, kpts, C_k, eps_k, S_sc, F_sc=None, *,
                            degeneracy_tol=DEGENERACY_TOL, realify_tol=None):
    """Real canonical supercell-Γ orbitals from per-k Bloch orbitals.

    Returns ``(C_sc, eps, eps_source, labels, blocks)`` sorted by energy:
    ``C_sc`` is ``(N_c·n, N_c·n_orb)`` real and S-orthonormal, ``eps`` are
    the eigenvalues of ``F_sc`` in each realified block (``eps_source``
    when ``F_sc`` is ``None``, the driver's own eigenvalue of each orbital
    sorted alongside), ``labels[p] = (k index, kind)`` with ``kind`` in
    ``{"trim", "pair"}``, and ``blocks`` the list of block sizes.

    Orbitals are grouped by mesh point (self-paired, ``k ≡ −k``) or by
    ``±k`` pair, cut into blocks of eigenvalues closer than
    ``degeneracy_tol``, realified per block and, when ``F_sc`` is given,
    re-diagonalised inside the block so the result is an exact eigenbasis of
    the folded Fock operator whether or not the block was exactly
    degenerate.
    """
    tol_r = (BLOCH_REFERENCE_TOLERANCES["realify"] if realify_tol is None
             else float(realify_tol))
    kpts = np.asarray(kpts, dtype=float).reshape(-1, 3)
    _, _, phase = _cells_and_phases(ccm, kpts)
    n_c, n_k = phase.shape
    _, partner = _mesh_labels(ccm, kpts)
    S_sc = np.asarray(S_sc, dtype=float)
    cols, eps_out, eps_src, labels, blocks = [], [], [], [], []
    done = set()
    for ki in range(n_k):
        if ki in done:
            continue
        pk = int(partner[ki])
        members = [ki] if pk == ki else [ki, pk]
        done.update(members)
        kind = "trim" if len(members) == 1 else "pair"
        psi_all, e_all = [], []
        for m in members:
            C = np.asarray(C_k[m])
            e = np.asarray(eps_k[m]).real.astype(float)
            psi_all.append(np.kron(phase[:, m][:, None], C) / np.sqrt(n_c))
            e_all.append(e)
        psi_all = np.concatenate(psi_all, axis=1)
        e_all = np.concatenate(e_all)
        order = np.argsort(e_all, kind="stable")
        psi_all, e_all = psi_all[:, order], e_all[order]
        start = 0
        while start < len(e_all):
            stop = start + 1
            while (stop < len(e_all)
                   and abs(e_all[stop] - e_all[stop - 1]) < degeneracy_tol):
                stop += 1
            X = _realify_block(psi_all[:, start:stop], S_sc, tol_r)
            e_blk = e_all[start:stop]
            if F_sc is not None:
                Fb = X.T @ np.asarray(F_sc, dtype=float) @ X
                Fb = 0.5 * (Fb + Fb.T)
                w, U = np.linalg.eigh(Fb)
                X = X @ U
                e_new = w
            else:
                e_new = np.full(stop - start, float(np.mean(e_blk)))
            for b in range(X.shape[1]):
                cols.append(X[:, b])
                eps_out.append(float(e_new[b]))
                eps_src.append(float(e_blk[b]))
                labels.append((ki, kind))
            blocks.append(stop - start)
            start = stop
    C_sc = np.array(cols).T
    eps_out = np.array(eps_out)
    eps_src = np.array(eps_src)
    order = np.argsort(eps_out, kind="stable")
    return (C_sc[:, order], eps_out[order], eps_src[order],
            [labels[i] for i in order], blocks)


def is_bloch_result(obj) -> bool:
    """Whether ``obj`` is a Bloch producer result this module can transform.

    True for a :class:`~vibeqc.periodic.ccm.ri.CCMGDFResult` and for a
    multi-k driver result that carries one as ``ccm_result`` (what the
    ``variant="neutral-bloch"`` runner adapter returns).
    """
    if obj is None:
        return False
    raw = getattr(obj, "raw", None)
    if raw is not None and hasattr(raw, "kpoints_cart"):
        return True
    wrapper = getattr(obj, "ccm_result", None)
    return (hasattr(obj, "kpoints_cart") and wrapper is not None
            and getattr(wrapper, "raw", None) is obj)


def as_real_reference(ccm, scf_result, **kwargs):
    """``scf_result`` as a real supercell-Γ reference, transforming a Bloch one.

    The hook the correlated entry points call on their reference argument:
    a real-Γ result (``CCMSCFResult`` / ``CCMUHFResult``) or ``None`` is
    returned unchanged; a Bloch producer result is passed through
    :func:`ccm_reference_from_bloch` (``kwargs`` forwarded), so
    ``run_ccm_mp2(ccm, run_ccm_rhf_gdf(ccm, retain_fit_cache=True))``
    correlates in the Bloch representation without a k-space MP2 code.
    """
    if not is_bloch_result(scf_result):
        return scf_result
    return ccm_reference_from_bloch(ccm, scf_result, **kwargs)


def _projector_occupations(C, S, D):
    """``⟨ψ_p|S D S|ψ_p⟩`` for S-orthonormal columns: the occupation of each."""
    SDS = S @ D @ S
    return np.einsum("ip,ij,jp->p", C, SDS, C, optimize=True)


def _occupations(C, S_sc, D_fold, fill, tol_occ, who):
    """Integer occupations of the transformed orbitals, read off the density.

    Returns ``(occ_int, max_dev, n_occ)`` with ``occ_int`` in ``{0, fill}``;
    raises on fractional occupations (smearing, or a degenerate block that
    straddles the Fermi level) and when an occupied orbital lies above a
    virtual one in the energy order (the orbitals are energy-sorted, so a
    global aufbau over the mesh puts every occupied orbital first).
    """
    occ = _projector_occupations(C, S_sc, D_fold) / float(fill)
    occ_int = np.rint(occ)
    dev = float(np.max(np.abs(occ - occ_int))) if occ.size else 0.0
    if dev > tol_occ or np.any(occ_int < 0) or np.any(occ_int > 1):
        raise ValueError(
            f"{who}: the Bloch reference has non-integer occupations (max "
            f"deviation {dev:.2e} > {tol_occ:.1e}); the real-space correlated "
            "methods need an integer-aufbau determinant. Smearing, or a "
            "degenerate block straddling the Fermi level, is the cause.")
    n_occ = int(np.sum(occ_int))
    if not np.all(occ_int[:n_occ] == 1) or not np.all(occ_int[n_occ:] == 0):
        raise ValueError(
            f"{who}: an occupied orbital lies above a virtual one in the "
            "energy-sorted supercell spectrum; the mesh's occupation is not "
            "a global aufbau (a per-k filling of an inverted band), which "
            "the correlated methods do not describe.")
    return occ_int, dev, n_occ


def _spin_channel(ccm, kpts, C_k, eps_k, S_sc, F_sc, D_k, fill, tol,
                  degeneracy_tol, who):
    """Transform one spin channel and run the per-channel checks."""
    D_fold, imag_d = fold_block_circulant(ccm, kpts, D_k, tol=tol["fold_imag"])
    C, eps, eps_src, labels, blocks = real_supercell_orbitals(
        ccm, kpts, C_k, eps_k, S_sc, F_sc, degeneracy_tol=degeneracy_tol,
        realify_tol=tol["realify"])
    n_orb = C.shape[1]
    ortho = C.T @ S_sc @ C
    ortho_dev = float(np.max(np.abs(ortho - np.eye(n_orb))))
    fmo = C.T @ F_sc @ C
    offdiag = float(np.max(np.abs(fmo - np.diag(np.diag(fmo)))))
    eig_shift = float(np.max(np.abs(np.diag(fmo) - eps)))
    eig_vs_src = float(np.max(np.abs(eps - eps_src)))
    occ_int, occ_dev, n_occ = _occupations(C, S_sc, D_fold, fill,
                                           tol["occupation"], who)
    D = float(fill) * (C[:, :n_occ] @ C[:, :n_occ].T)
    dens_dev = float(np.max(np.abs(D - D_fold)))
    checks = {
        "orthonormality": ortho_dev,
        "fock_offdiag": offdiag,
        "eigenvalue": max(eig_shift, eig_vs_src),
        "density": dens_dev,
    }
    for key, value in checks.items():
        if value > tol[key]:
            hint = ""
            if key != "orthonormality":
                hint = (" A Bloch SCF stopped at the multi-k default "
                        "conv_tol_grad (1e-6) fails this check by its last "
                        "step; converge it to options.conv_tol_grad <= "
                        f"{BLOCH_REFERENCE_SCF_CONV_TOL_GRAD:g}.")
            raise ValueError(
                f"{who}: check {key!r} failed on the transformed reference "
                f"({value:.2e} > {tol[key]:.1e}); the per-k orbitals are not "
                "an eigenbasis of the folded operator, so this result "
                "cannot be a real-Γ reference of the same Hamiltonian."
                + hint)
    stats = {
        "fold_imag": imag_d, "orthonormality": ortho_dev,
        "fock_offdiag": offdiag, "eigenvalue": max(eig_shift, eig_vs_src),
        "density": dens_dev, "occupation": occ_dev,
        "kinds": {k: sum(1 for l in labels if l[1] == k)
                  for k in ("trim", "pair")},
        "blocks": blocks,
    }
    return C, eps, D, n_occ, stats


def _source_pieces(result, who):
    """``(wrapper, raw)`` for a CCMGDFResult or a raw result carrying one."""
    raw = getattr(result, "raw", None)
    if raw is not None and hasattr(raw, "kpoints_cart"):
        return result, raw
    wrapper = getattr(result, "ccm_result", None)
    if wrapper is not None and hasattr(result, "kpoints_cart"):
        return wrapper, result
    raise TypeError(
        f"{who}: expected the Bloch producer's result (a CCMGDFResult from "
        "run_ccm_{rhf,rks,uhf,uks}_gdf, or the multi-k result the "
        "neutral-bloch runner adapter returns with ccm_result attached); "
        f"got {type(result).__name__}.")


def _cderi_for(ccm, wrapper, raw, *, cderi, build_cderi, fit_cache, lat_opts,
               fold_threads, who):
    """The neutral cderi of the reference's own fitted operator, with its record."""
    from .direct import _neutral_cderi_with_info

    if cderi is not None:
        L = np.asarray(cderi, dtype=float)
        return L, {"build": "caller",
                   "shape": tuple(int(n) for n in L.shape)}
    if not build_cderi:
        return None, None
    fit_source = str(getattr(wrapper, "fit_source", "") or "")
    if fit_source != "classic":
        raise ValueError(
            f"{who}: the Bloch reference was fitted with fit_source="
            f"{fit_source or 'unrecorded'!r}, but the real-space correlation "
            "kernel is the classic fold (ccm_neutral_cderi_fold), a different "
            "fitted operator (vibe-qc #378: 1.4e-4 Ha/cell apart on "
            "LiH/STO-3G). Run the Bloch SCF with the producer default "
            "(fit_source='classic', gdf_method='rsgdf'), or pass cderi= / "
            "build_cderi=False explicitly.")
    support = dict(getattr(getattr(raw, "executed_fit_source", None),
                           "numerical_support", ()) or ())
    policy = support.get("classic_support_policy", "historical_index_ball")
    if policy != "historical_index_ball":
        raise ValueError(
            f"{who}: the Bloch reference's classic fit summed the {policy!r} "
            "AO-pair support, but the classic fold sums the historical index "
            "ball (#466), a different fitted operator. Run the Bloch SCF with "
            "the producer default (classic_support_policy="
            "'historical_index_ball'), or pass cderi= / build_cderi=False "
            "explicitly.")
    if fit_cache is None:
        fit_cache = getattr(raw, "fit_cache", None)
    if fit_cache is not None and not bool(getattr(fit_cache, "need_k_pairs", True)):
        # A pure-functional (or COSX-exchange) run fits only the diagonal
        # (k, k) pairs, which is all its Coulomb term needs; the fold needs
        # every (k_a, k_a + q), so the cache cannot serve it and the fold
        # refits at the same operator instead of raising on the first
        # missing pair. The record says so (cderi_info["build"] == "fold").
        fit_cache = None
    ke_cutoff = float(getattr(raw, "rsgdf_ke_cutoff", 200.0) or 200.0)
    aux_name = getattr(raw, "aux_basis_name", None) or None
    tail = getattr(wrapper, "rsgdf_tail_ke_cutoff", None)
    return _neutral_cderi_with_info(
        ccm, None, "fold", ke_cutoff, aux_name, tail_ke_cutoff=tail,
        lat_opts=lat_opts, fit_cache=fit_cache, fold_threads=fold_threads)


def ccm_reference_from_bloch(ccm, result, *, cderi=None, build_cderi=True,
                             fit_cache=None, lat_opts=None, fold_threads=None,
                             degeneracy_tol=DEGENERACY_TOL, tolerances=None):
    """The Bloch producer's converged SCF as a real supercell-Γ reference.

    Parameters
    ----------
    ccm : CCMSystem
        The torus the Bloch SCF ran on (the same ``nrep``; the mesh is
        checked against it).
    result
        The :class:`~vibeqc.periodic.ccm.ri.CCMGDFResult` of
        :func:`~vibeqc.periodic.ccm.ri.run_ccm_rhf_gdf` /
        ``run_ccm_rks_gdf`` (closed shell) or ``run_ccm_uhf_gdf`` /
        ``run_ccm_uks_gdf`` (open shell), or the multi-k result the
        ``variant="neutral-bloch"`` runner adapter returns (it carries the
        wrapper as ``ccm_result``).
    cderi, build_cderi
        The neutral cderi the reference should carry (``result.cderi``, what
        the correlated entry points consume). ``cderi=`` uses the caller's
        array; otherwise (``build_cderi=True``, the default) the fold builds
        it at the driver's ``ke_cutoff``, auxiliary basis and executed tail,
        consuming the driver's retained fit (``run_ccm_rhf_gdf(...,
        retain_fit_cache=True)``, or ``fit_cache=``) when one is available,
        which costs a rotation rather than a refit. ``build_cderi=False``
        returns an SCF-only reference. A reference fitted with any source
        other than the classic one is refused here, because the fold would
        be a different operator.
    lat_opts, fold_threads
        Forwarded to the fold (the driver's lattice options are not on its
        result; pass the same object the SCF used when it was not the
        default, or the fit-cache consumer refuses the mismatch).
    degeneracy_tol
        Eigenvalue window (Ha) of one realification block.
    tolerances
        Overrides of :data:`BLOCH_REFERENCE_TOLERANCES` by key.

    Returns
    -------
    CCMSCFResult or CCMUHFResult
        The object the direct route returns, with TOTAL cyclic energies
        (``result.energy × N_c``), real S-orthonormal canonical supercell
        orbitals, the folded Fock/overlap/core Hamiltonian, the density, the
        cderi (when built), the producer's convention fields (exchange
        seam, fit source, tail, vacuum opt-in, guess selection) and a
        :class:`BlochFoldReport` under ``bloch_transform``. Its ``backend``
        is the driver's, suffixed ``+bloch-fold``, so a ``+PARITY_HELD``
        marker survives and ``parity_held`` still reads it.
    """
    who = "ccm_reference_from_bloch"
    tol = _tolerances(tolerances)
    wrapper, raw = _source_pieces(result, who)
    if not bool(getattr(raw, "converged", False)):
        raise ValueError(f"{who}: the Bloch SCF did not converge; refusing to "
                         "build a correlation reference on it.")
    if float(getattr(raw, "smearing_temperature", 0.0) or 0.0) > 0.0:
        raise ValueError(
            f"{who}: the Bloch SCF ran with smearing "
            f"(T = {float(raw.smearing_temperature):g}); the real-space "
            "correlated methods need an integer-occupation determinant.")
    kpts = np.asarray(raw.kpoints_cart, dtype=float).reshape(-1, 3)
    n_c = int(ccm.n_cells)
    if len(kpts) != n_c:
        raise ValueError(
            f"{who}: the Bloch result has {len(kpts)} k-points but the torus "
            f"nrep={tuple(int(n) for n in ccm.nrep)} has {n_c} cells; the "
            "result was not produced on this CCMSystem's mesh.")
    weights = np.asarray(getattr(raw, "kpoint_weights", np.full(n_c, 1.0 / n_c)),
                         dtype=float).reshape(-1)
    if weights.size and float(np.max(np.abs(weights - weights[0]))) > 1e-12:
        raise ValueError(
            f"{who}: the k-point weights are not uniform; a symmetry-reduced "
            "mesh cannot be folded onto the torus.")
    n_e = int(ccm.supercell.n_electrons())
    open_shell = hasattr(raw, "mo_coeffs_alpha")

    S_sc, imag_s = fold_block_circulant(ccm, kpts, raw.overlap, tol=tol["fold_imag"])
    H_sc, imag_h = fold_block_circulant(ccm, kpts, raw.hcore, tol=tol["fold_imag"])
    imag_max = max(imag_s, imag_h)
    n_bf_sc = S_sc.shape[0]
    if n_bf_sc != int(ccm.nbf):
        raise ValueError(
            f"{who}: folded overlap is {n_bf_sc}×{n_bf_sc} but the supercell "
            f"carries {int(ccm.nbf)} AOs.")

    energy = float(raw.energy) * n_c
    e_el = float(getattr(raw, "e_electronic", raw.energy)) * n_c
    e_nn = float(getattr(raw, "e_nuclear", 0.0)) * n_c
    backend = f"{str(getattr(raw, 'backend', '') or '')}+bloch-fold"
    common = dict(
        converged=True, n_iter=int(getattr(raw, "n_iter", 0)),
        energy=energy, energy_per_atom=energy / int(ccm.n_atoms),
        e_electronic=e_el, e_nuclear=e_nn, overlap=S_sc, backend=backend,
        rsgdf_tail_ke_cutoff=getattr(wrapper, "rsgdf_tail_ke_cutoff", None),
        fit_source=str(getattr(wrapper, "fit_source", "") or ""),
        guess_selection=getattr(wrapper, "guess_selection", None),
        vacuum_padding_allowed=bool(getattr(wrapper, "vacuum_padding_allowed", False)),
        vacuum_padded_axes=tuple(getattr(wrapper, "vacuum_padded_axes", ()) or ()),
    )
    achieved = dict(getattr(wrapper, "achieved_convergence", {}) or {})
    L, info = _cderi_for(ccm, wrapper, raw, cderi=cderi, build_cderi=build_cderi,
                         fit_cache=fit_cache, lat_opts=lat_opts,
                         fold_threads=fold_threads, who=who)

    if not open_shell:
        F_sc, imag_f = fold_block_circulant(ccm, kpts, raw.fock, tol=tol["fold_imag"])
        imag_max = max(imag_max, imag_f)
        C, eps, D, n_occ, st = _spin_channel(
            ccm, kpts, raw.mo_coeffs, raw.mo_energies, S_sc, F_sc, raw.density,
            2.0, tol, degeneracy_tol, who)
        n_count = float(np.einsum("ij,ji->", D, S_sc))
        count_dev = abs(n_count - n_e)
        if 2 * n_occ != n_e or count_dev > tol["electron_count"]:
            raise ValueError(
                f"{who}: the transformed determinant holds {n_count:.10f} "
                f"electrons ({n_occ} doubly occupied orbitals) but the "
                f"supercell has {n_e}.")
        from .scf import CCMSCFResult

        report = BlochFoldReport(
            n_k=len(kpts), n_cells=n_c, n_orbitals=int(C.shape[1]),
            n_occupied=(n_occ,), kinds=st["kinds"], n_blocks=len(st["blocks"]),
            max_block=int(max(st["blocks"])), degeneracy_tol=float(degeneracy_tol),
            fold_imag_max=max(imag_max, st["fold_imag"]),
            orthonormality_max_dev=st["orthonormality"],
            fock_offdiag_max=st["fock_offdiag"],
            eigenvalue_shift_max=st["eigenvalue"], density_max_dev=st["density"],
            electron_count_dev=count_dev, occupation_max_dev=st["occupation"],
            tolerances=dict(tol), source_backend=str(getattr(raw, "backend", "") or ""),
            source_fit_source=common["fit_source"],
            source_achieved_convergence=achieved,
            cderi_build=(None if info is None else str(info.get("build"))))
        return CCMSCFResult(
            mo_energies=eps, mo_coeffs=C, density=D, fock=F_sc, hcore=H_sc,
            idempotency_error=float(np.linalg.norm(D @ S_sc @ D - 2.0 * D)),
            scf_trace=(), exchange_q0=(getattr(wrapper, "exchange_q0", None) or None),
            cderi=L, cderi_info=info, bloch_transform=report, **common)

    # ---- open shell: the same transform per spin channel ----------------
    Fa, imag_fa = fold_block_circulant(ccm, kpts, raw.fock_alpha, tol=tol["fold_imag"])
    Fb, imag_fb = fold_block_circulant(ccm, kpts, raw.fock_beta, tol=tol["fold_imag"])
    imag_max = max(imag_max, imag_fa, imag_fb)
    Ca, ea, Pa, n_a, sa = _spin_channel(
        ccm, kpts, raw.mo_coeffs_alpha, raw.mo_energies_alpha, S_sc, Fa,
        raw.density_alpha, 1.0, tol, degeneracy_tol, who)
    Cb, eb, Pb, n_b, sb = _spin_channel(
        ccm, kpts, raw.mo_coeffs_beta, raw.mo_energies_beta, S_sc, Fb,
        raw.density_beta, 1.0, tol, degeneracy_tol, who)
    n_count = float(np.einsum("ij,ji->", Pa + Pb, S_sc))
    count_dev = abs(n_count - n_e)
    if n_a + n_b != n_e or count_dev > tol["electron_count"]:
        raise ValueError(
            f"{who}: the transformed determinant holds {n_count:.10f} "
            f"electrons ({n_a} alpha, {n_b} beta) but the supercell has {n_e}.")
    mult_sc = int(ccm.supercell.multiplicity)
    if n_a - n_b != mult_sc - 1:
        # The multi-k driver replicates the unit cell's spin over the mesh
        # (ferromagnetically); the CCMSystem's supercell multiplicity is
        # parity-derived unless the cell carries an explicit high spin. The
        # transformed reference reports what the Bloch SCF converged; the
        # direct producer of this CCMSystem would run the other spin state.
        warnings.warn(
            f"{who}: the Bloch reference has {n_a - n_b} unpaired electrons "
            f"on the supercell, the CCMSystem's supercell multiplicity "
            f"{mult_sc} implies {mult_sc - 1}; run_ccm_uhf_direct on this "
            "CCMSystem converges a different spin state (per-unit-cell spin "
            "replication, see run_ccm_uhf_gdf). The transformed reference "
            "carries the Bloch SCF's own spin state.", stacklevel=2)
    sz = 0.5 * (n_a - n_b)
    ov = Ca[:, :n_a].T @ S_sc @ Cb[:, :n_b]
    s_squared = float(sz * (sz + 1.0) + n_b - np.sum(ov * ov))
    from .uhf import CCMUHFResult

    st = {k: max(sa[k], sb[k]) for k in ("fold_imag", "orthonormality",
                                         "fock_offdiag", "eigenvalue",
                                         "density", "occupation")}
    kinds = {k: sa["kinds"][k] + sb["kinds"][k] for k in ("trim", "pair")}
    blocks = list(sa["blocks"]) + list(sb["blocks"])
    report = BlochFoldReport(
        n_k=len(kpts), n_cells=n_c, n_orbitals=int(Ca.shape[1]),
        n_occupied=(n_a, n_b), kinds=kinds, n_blocks=len(blocks),
        max_block=int(max(blocks)), degeneracy_tol=float(degeneracy_tol),
        fold_imag_max=max(imag_max, st["fold_imag"]),
        orthonormality_max_dev=st["orthonormality"],
        fock_offdiag_max=st["fock_offdiag"], eigenvalue_shift_max=st["eigenvalue"],
        density_max_dev=st["density"], electron_count_dev=count_dev,
        occupation_max_dev=st["occupation"], tolerances=dict(tol),
        source_backend=str(getattr(raw, "backend", "") or ""),
        source_fit_source=common["fit_source"],
        source_achieved_convergence=achieved,
        cderi_build=(None if info is None else str(info.get("build"))),
        s_squared=s_squared)
    idem = max(float(np.linalg.norm(P @ S_sc @ P - P)) for P in (Pa, Pb))
    return CCMUHFResult(
        n_alpha=n_a, n_beta=n_b, s_squared=s_squared,
        mo_energies_alpha=ea, mo_energies_beta=eb,
        mo_coeffs_alpha=Ca, mo_coeffs_beta=Cb,
        density_alpha=Pa, density_beta=Pb, idempotency_error=idem,
        cderi=L, cderi_info=info, bloch_transform=report, **common)


def bloch_reference_and_cderi(ccm, *, initial_guess="AUTO", ke_cutoff=200.0,
                              aux_basis=None, open_shell=False, who,
                              allow_vacuum_padding=False, fold_threads=None,
                              lat_opts=None, **driver_kwargs):
    """The Bloch SCF of ``ccm``, transformed, with the cderi it was fitted on.

    The ``reference="bloch"`` route of the one-call correlation wrappers
    (:func:`~vibeqc.periodic.ccm.mp2.run_ccm_ri_mp2`,
    :func:`~vibeqc.periodic.ccm.ccsd.run_ccm_ri_ccsd`,
    :func:`~vibeqc.periodic.ccm.direct.run_ccm_dlpno_mp2_direct` and their
    open-shell siblings): runs :func:`~vibeqc.periodic.ccm.ri.run_ccm_rhf_gdf`
    (or ``run_ccm_uhf_gdf``) with ``retain_fit_cache=True`` so the driver's
    classic per-pair fit survives, folds it to the real supercell cderi
    (a rotation, not a refit) and returns ``(reference, L)`` exactly as the
    direct route's helper does, the reference converged on that same fitted
    operator by construction (Theorem 1).

    The Bloch SCF runs at ``conv_tol_grad =``
    :data:`BLOCH_REFERENCE_SCF_CONV_TOL_GRAD` unless ``options`` is passed:
    the transform's density check needs the occupied orbitals to reproduce
    the density the driver returns, which the multi-k default (1e-6) leaves
    short of on LiH and MgO pob-DZVP-rev2 (1,1,1).

    The multi-k drivers describe the **unit cell**: ``reference="bloch"``
    therefore needs a closed-shell unit cell for the closed-shell wrappers,
    and for the open-shell ones a supercell spin that is the unit cell's
    replicated over the torus (the drivers' per-cell convention, which is
    also what :class:`~vibeqc.periodic.ccm.system.CCMSystem` does for an
    explicit high-spin cell); anything else is refused here with a pointer
    to ``reference="direct"`` rather than run on a different spin state.
    """
    from .ri import run_ccm_rhf_gdf, run_ccm_uhf_gdf

    unit = ccm.unit_system
    n_cell = int(unit.n_electrons())
    mult_cell = int(getattr(unit, "multiplicity", 1) or 1)
    n_c = int(ccm.n_cells)
    mult_sc = int(ccm.supercell.multiplicity)
    if not open_shell:
        if n_cell % 2 != 0 or mult_cell != 1:
            raise ValueError(
                f"{who}: reference='bloch' runs the multi-k RHF driver on the "
                f"unit cell, which has {n_cell} electrons and multiplicity "
                f"{mult_cell}; a closed-shell supercell built from an open-"
                "shell cell has no Bloch RHF producer. Use reference='direct'.")
    elif mult_sc - 1 != n_c * (mult_cell - 1):
        raise ValueError(
            f"{who}: reference='bloch' replicates the unit cell's spin "
            f"({mult_cell - 1} unpaired) over the {n_c} cells of the torus, "
            f"but the CCMSystem's supercell multiplicity {mult_sc} asks for "
            f"{mult_sc - 1} unpaired electrons; the two producers would "
            "converge different spin states. Use reference='direct'.")
    kw = dict(driver_kwargs)
    kw.setdefault("rsgdf_ke_cutoff", float(ke_cutoff))
    kw["retain_fit_cache"] = True
    if kw.get("options") is None:
        from ..._vibeqc_core import PeriodicRHFOptions

        options = PeriodicRHFOptions()
        options.conv_tol_grad = BLOCH_REFERENCE_SCF_CONV_TOL_GRAD
        kw["options"] = options
    producer = run_ccm_uhf_gdf if open_shell else run_ccm_rhf_gdf
    res = producer(ccm, aux_basis=aux_basis, initial_guess=initial_guess,
                   allow_vacuum_padding=allow_vacuum_padding, **kw)
    if not bool(res.converged):
        raise ValueError(
            f"{who}: the Bloch SCF reference did not converge; refusing to "
            "correlate a non-converged state.")
    ref = ccm_reference_from_bloch(ccm, res, lat_opts=lat_opts,
                                   fold_threads=fold_threads)
    return ref, ref.cderi
