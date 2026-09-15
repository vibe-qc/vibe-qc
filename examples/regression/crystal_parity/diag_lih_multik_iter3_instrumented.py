"""Instrumented multi-k SCF for the LiH/STO-3G primitive iter-3 collapse.

Mirrors `run_rhf_periodic_multi_k_ewald3d` (see
`python/vibeqc/periodic_rhf_multi_k_ewald.py`) but adds per-iter /
per-k JSON dumps so we can identify WHICH spectral feature breaks at
the iter-2 → iter-3 transition documented in the retired periodic-JK
forensic note in git history.

Output: `/tmp/lih_multik_iter3_dump.json`. ~2 hr wall on the reference compute host
(14 cpus). Each iter logs full F(k) and D(k) spectra, the F(k)
HOMO-LUMO gap per k, the [F,DS] commutator norm per k, and the
occupied-subspace MO overlap with the previous iter (which reveals
orbital flipping / mis-ordering between iters).

Three hypotheses we're trying to discriminate:
  (1) D(k) eigenvalue rises above 2.0 — density loses physical
      interpretation, F gets corrupted, runaway.
  (2) F(k) HOMO-LUMO gap closes — insulator becomes effectively
      metallic mid-SCF, Aufbau picks wrong orbitals.
  (3) Aufbau orbital ordering flips iter-to-iter — wrong occupied
      states locked in by hard energy-sort.

Run this input with a configured vibe-qc interpreter. Site-specific submission,
affinity and interpreter selection are maintained in private operations; the
private LiH helper runs from a staged copy of this input directory.
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List

os.environ.setdefault("PYSCF_TMPDIR", "/tmp/pyscf_vibeqc")
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import vibeqc as vq
from vibeqc import (
    CoulombMethod,
    InitialGuess,
    LatticeSumOptions,
    bloch_sum,
    compute_kinetic_lattice,
    compute_nuclear_lattice,
    compute_overlap_lattice,
    nuclear_repulsion_per_cell,
    real_space_density_from_kpoints,
)
from vibeqc._vibeqc_core import PeriodicRHFOptions
from vibeqc.guess import initial_density_closed_shell
from vibeqc.ewald_j import auto_grid
from vibeqc.madelung import (
    madelung_energy_correction_for_lat as _madelung_energy_correction_for_lat,
)
from vibeqc.periodic_fock_multi_k import build_periodic_fock_ewald3d_k
from vibeqc.periodic_rhf_multi_k_ewald import (
    _canonical_orthogonalizer_complex,
    _damp_lattice_matrix,
    _diag_in_orth_basis,
    _g0_block,
)


# Schedule + MOM helpers inlined for portability: compute-reference's vibe-qc
# install is on a different branch and won't have the new
# `vibeqc.level_shift_schedule` / `vibeqc.mom` modules until the
# branch lands. Canonical sources at
# `python/vibeqc/level_shift_schedule.py` and `python/vibeqc/mom.py`
# on this worktree; keep these inline copies 1:1 if you change them.
class LevelShiftSchedule:
    def __init__(self, shifts):
        if not shifts:
            raise ValueError("LevelShiftSchedule: shifts must be non-empty")
        for i, s in enumerate(shifts):
            if s < 0:
                raise ValueError(
                    f"LevelShiftSchedule: shifts must be non-negative; "
                    f"got shifts[{i}] = {s}"
                )
        self.shifts = list(shifts)

    def at(self, iter_idx):
        if iter_idx < 1:
            raise ValueError(
                f"LevelShiftSchedule.at: iter_idx must be >= 1; got {iter_idx}"
            )
        return float(self.shifts[min(iter_idx - 1, len(self.shifts) - 1)])

    def as_list(self):
        return list(self.shifts)

    @classmethod
    def crystal_default(cls):
        return cls([5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.0])


def mom_select(C_new, S_k, C_prev_occ, n_occ, eps_new=None):
    """Maximum-Overlap selection of n_occ columns from C_new with
    largest summed S(k)-overlap onto C_prev_occ. Returns column
    indices, sorted by eps_new within (if supplied) so the
    `[:, :n_occ]` slicing convention downstream still gives
    energy-sorted occupied MOs."""
    if C_prev_occ.shape[1] != n_occ:
        raise ValueError(f"mom_select: C_prev_occ has {C_prev_occ.shape[1]} "
                         f"columns but n_occ={n_occ}")
    O = C_prev_occ.conj().T @ S_k @ C_new
    proj = np.sum(np.abs(O) ** 2, axis=0)
    sel = np.argsort(-proj)[:n_occ]
    if eps_new is not None:
        sel = sel[np.argsort(np.real(eps_new[sel]))]
    else:
        sel = np.sort(sel)
    return sel.astype(int)

ANG2BOHR = 1.0 / 0.529177210903

# ---- Knobs (override via env) -----------------------------------------------
# `BIPOLE_DIAG_MODE`:
#   "no_aids"        → Hcore guess, LEVSHIFT off (baseline collapse).
#   "levshift"       → Hcore guess, iter-decreasing LEVSHIFT
#                      [5,4,3,2,1,0.5,0].
#   "sad"            → SAD guess (periodic; lifts iter 1 to ~-6.83
#                      per handover), LEVSHIFT off.
#   "sad_levshift"   → SAD guess + iter-decreasing LEVSHIFT — the
#                      combination most likely to converge to the
#                      physical basin.
import os
MODE = os.environ.get("BIPOLE_DIAG_MODE", "no_aids")
A_ANG = 4.084                  # LiH conventional lattice parameter (Å)
KMESH = (2, 2, 2)              # The kmesh that triggers the collapse
MAX_ITER = int(os.environ.get("BIPOLE_DIAG_MAX_ITER", "15"))
CUTOFF_BOHR = 12.0             # ≥ 12 needed for LiH primitive PSD overlap
DAMPING = 0.0                  # No damping — we want to see the raw collapse
USE_DIIS = False               # Off — collapse happens with DIIS off too
FMIXING = 0.0                  # Off

USE_LEVSHIFT = MODE in ("levshift", "sad_levshift", "sad_levshift_mom")
USE_SAD = MODE in (
    "sad", "sad_levshift", "sad_mom", "sad_levshift_mom",
)
USE_MOM = MODE in ("mom", "sad_mom", "sad_levshift_mom")

LEVEL_SHIFT_SCHEDULE = (
    LevelShiftSchedule.crystal_default() if USE_LEVSHIFT else None
)
OUTPUT_PATH = f"/tmp/lih_multik_iter3_dump_{MODE}.json"


def build_lih_primitive() -> tuple:
    """FCC primitive cell, 2 atoms (Li + H), 1 formula unit.

    Primitive lattice vectors:  a/2 · (0,1,1), (1,0,1), (1,1,0).
    Li at primitive-fractional (0,0,0); H at primitive-fractional
    (1/2, 1/2, 1/2). The primitive (1/2)·(a1 + a2 + a3) Cartesian
    is (a/2)·(1,1,1) bohr — i.e. (a_conv/2)·(1,1,1), the standard
    rocksalt nearest-neighbour vector for SG 225 with conv-cell
    parameters Li(0,0,0) / H(0.5, 0.5, 0.5). Matches the CRYSTAL14
    input in `baseline_sto3g/lih-rhf-sto3g.d12`.
    """
    a = A_ANG * ANG2BOHR
    lattice = (a / 2.0) * np.array([
        [0.0, 1.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
    ])
    li_cart = [0.0, 0.0, 0.0]
    h_cart = [a / 2.0, a / 2.0, a / 2.0]
    atoms = [vq.Atom(3, li_cart), vq.Atom(1, h_cart)]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis, a


def occ_subspace_overlap(
    C_new: np.ndarray, C_prev: np.ndarray, S: np.ndarray, n_occ: int,
) -> Dict[str, float]:
    """Compare occupied subspaces across iters.

    Returns ``det(O O^†)`` and ``min_sigma`` where
    ``O = C_prev[:, :n_occ]^† S C_new[:, :n_occ]``. ``det → 1`` and
    ``min_sigma → 1`` when the occupied subspace is stable; either
    dropping toward 0 means orbital-flipping or new occupied-virtual
    swap.
    """
    O = C_prev[:, :n_occ].conj().T @ S @ C_new[:, :n_occ]
    # Singular values of O describe principal angles between subspaces.
    sigmas = np.linalg.svd(O, compute_uv=False)
    det = float(np.real(np.linalg.det(O @ O.conj().T)))
    return {
        "det_OOdag": det,
        "min_sigma": float(np.min(np.abs(sigmas))),
        "max_sigma": float(np.max(np.abs(sigmas))),
    }


def dump_iter(
    iter_idx: int,
    *,
    E_total: float,
    E_elec: float,
    E_nuc: float,
    E_madelung_fix: float,
    dE: float,
    grad_norm_sum: float,
    F_k_list: List[np.ndarray],
    D_k_list: List[np.ndarray],
    S_k_list: List[np.ndarray],
    eps_per_k: List[np.ndarray],
    C_per_k: List[np.ndarray],
    C_prev_per_k: List[np.ndarray] | None,
    D_real,
    n_occ: int,
) -> Dict[str, Any]:
    n_k = len(F_k_list)
    per_k: List[Dict[str, Any]] = []
    for kx in range(n_k):
        F_k = F_k_list[kx]
        D_k = D_k_list[kx]
        S_k = S_k_list[kx]
        eps_k = eps_per_k[kx]
        # AO-basis eigvals — depend on S(k); not directly the purity
        # check (a small S eigval inflates D eigvals).
        D_eigvals = np.linalg.eigvalsh(0.5 * (D_k + D_k.conj().T))
        F_eigvals = np.linalg.eigvalsh(0.5 * (F_k + F_k.conj().T))
        # S(k) eigvals — if any drop near 0 the basis is locally
        # linearly dependent at this k, and the canonical
        # orthogonaliser amplifies noise inversely.
        S_eigvals = np.linalg.eigvalsh(0.5 * (S_k + S_k.conj().T))
        # The physical purity check: eigvals of S@D should be 2 (n_occ
        # times) and 0 (n_virt times) for a pure RHF density built
        # from S-orthonormal MOs. Any drift means D has lost
        # idempotency (or the MOs aren't S-orthonormal). Compute via
        # eigvals of a Hermitian similarity: M = S^{1/2} D S^{1/2}
        # has the same eigvals as S·D and is guaranteed Hermitian.
        S_eigh_vals, S_eigh_vecs = np.linalg.eigh(0.5 * (S_k + S_k.conj().T))
        # Guard against tiny / negative S eigvals on rounding noise.
        S_eigh_vals_safe = np.maximum(S_eigh_vals, 1e-14)
        S_sqrt = S_eigh_vecs @ np.diag(np.sqrt(S_eigh_vals_safe)) \
            @ S_eigh_vecs.conj().T
        M = S_sqrt @ D_k @ S_sqrt
        SD_eigvals = np.linalg.eigvalsh(0.5 * (M + M.conj().T))
        # HOMO-LUMO gap from the diagonalised MO energies (eps_per_k).
        n_bf = eps_k.shape[0]
        homo = float(np.real(eps_k[n_occ - 1])) if n_occ > 0 else float("-inf")
        lumo = float(np.real(eps_k[n_occ])) if n_occ < n_bf else float("inf")
        gap = lumo - homo
        # Commutator norm.
        FDS = F_k @ D_k @ S_k
        comm_norm = float(np.linalg.norm(FDS - FDS.conj().T))
        entry: Dict[str, Any] = {
            "k_idx": kx,
            "F_eigvals": [float(v) for v in np.real(F_eigvals)],
            "D_eigvals": [float(v) for v in np.real(D_eigvals)],
            "S_eigvals": [float(v) for v in np.real(S_eigvals)],
            "SD_eigvals": [float(v) for v in np.real(SD_eigvals)],
            "mo_energies": [float(v) for v in np.real(eps_k)],
            "homo": homo,
            "lumo": lumo,
            "gap": float(gap),
            "comm_norm": comm_norm,
            "D_eigval_max": float(np.max(np.real(D_eigvals))),
            "S_eigval_min": float(np.min(np.real(S_eigvals))),
            "SD_eigval_max": float(np.max(np.real(SD_eigvals))),
            "SD_eigval_min": float(np.min(np.real(SD_eigvals))),
            "D_trace": float(np.real(np.trace(D_k))),
            "SD_trace": float(np.real(np.trace(S_sqrt @ D_k @ S_sqrt))),
        }
        if C_prev_per_k is not None:
            entry["occ_overlap_vs_prev"] = occ_subspace_overlap(
                C_per_k[kx], C_prev_per_k[kx], S_k, n_occ,
            )
        per_k.append(entry)
    # Per-cell density traces (cheap, useful for cross-cell density leak).
    cell_traces: List[Dict[str, Any]] = []
    for g_idx in range(len(D_real.cells)):
        cell = D_real.cells[g_idx]
        blk = np.asarray(D_real.blocks[g_idx])
        cell_traces.append({
            "cell_index": [int(c) for c in cell.index],
            "trace": float(np.real(np.trace(blk))),
            "frobenius": float(np.linalg.norm(blk)),
        })
    return {
        "iter": iter_idx,
        "E_total": float(E_total),
        "E_elec": float(E_elec),
        "E_nuc": float(E_nuc),
        "E_madelung_fix": float(E_madelung_fix),
        "dE": float(dE),
        "grad_norm_sum_k": float(grad_norm_sum),
        "per_k": per_k,
        "cell_density_traces": cell_traces,
    }


def main() -> int:
    print(f"=== iter-3 collapse diagnostic on LiH primitive ===")
    print(f"  MODE = {MODE}")
    print(f"  A_ANG = {A_ANG}, KMESH = {KMESH}, MAX_ITER = {MAX_ITER}")
    print(f"  guess = {'SAD' if USE_SAD else 'HCORE'}")
    if LEVEL_SHIFT_SCHEDULE is not None:
        print(f"  LEVSHIFT schedule = {LEVEL_SHIFT_SCHEDULE.as_list()}")
    else:
        print(f"  LEVSHIFT = OFF")
    print(f"  MOM = {'ON' if USE_MOM else 'OFF'}")
    print(f"  CUTOFF_BOHR = {CUTOFF_BOHR}, DIIS = {USE_DIIS}, "
          f"FMIXING = {FMIXING}")
    print(f"  output → {OUTPUT_PATH}")
    sys.stdout.flush()

    system, basis, a = build_lih_primitive()
    print(f"  primitive a = {a:.6f} bohr ({A_ANG} Å)")
    print(f"  basis: STO-3G  ({basis.nbasis} BFs / {basis.nshells} shells)")
    n_elec = system.n_electrons()
    n_occ = n_elec // 2
    print(f"  n_electrons = {n_elec} (n_occ = {n_occ})")
    sys.stdout.flush()

    kmesh = vq.monkhorst_pack(system, list(KMESH))
    k_points = list(kmesh.kpoints)
    weights = np.asarray(kmesh.weights, dtype=float)
    n_k = len(k_points)
    print(f"  k-mesh: {n_k} k-points; weights sum = {weights.sum():.4f}")
    sys.stdout.flush()

    lat_opts = LatticeSumOptions()
    lat_opts.cutoff_bohr = CUTOFF_BOHR
    lat_opts.nuclear_cutoff_bohr = CUTOFF_BOHR
    lat_opts.coulomb_method = CoulombMethod.EWALD_3D

    omega = 0.5
    grid_shape_t = auto_grid(np.asarray(system.lattice), 0.3)
    print(f"  FFT grid {grid_shape_t}")
    sys.stdout.flush()

    # ---- Real-space one-electron integrals
    t0 = time.time()
    print("  building S/T/V real-space lattice integrals…")
    S_lat = compute_overlap_lattice(basis, system, lat_opts)
    T_lat = compute_kinetic_lattice(basis, system, lat_opts)
    from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch
    V_lat = compute_nuclear_lattice_dispatch(basis, system, lat_opts)
    cells = list(S_lat.cells)
    print(f"  one-elec lattice integrals built in {time.time() - t0:.1f}s; "
          f"n_cells = {len(cells)}")
    sys.stdout.flush()

    S_k_list: List[np.ndarray] = []
    Hcore_k_list: List[np.ndarray] = []
    X_k_list: List[np.ndarray] = []
    for k_idx, k in enumerate(k_points):
        k_arr = np.asarray(k, dtype=float).reshape(3)
        S_k = np.asarray(bloch_sum(S_lat, k_arr))
        T_k = np.asarray(bloch_sum(T_lat, k_arr))
        V_k = np.asarray(bloch_sum(V_lat, k_arr))
        H_k = T_k + V_k
        S_k = 0.5 * (S_k + S_k.conj().T)
        H_k = 0.5 * (H_k + H_k.conj().T)
        X_k, n_kept = _canonical_orthogonalizer_complex(
            S_k, threshold=1e-7, normalize_diag_first=True,
        )
        if n_occ > n_kept:
            raise RuntimeError(
                f"k={k_idx}: canonical orth dropped too many directions "
                f"(n_occ={n_occ}, n_kept={n_kept})"
            )
        S_k_list.append(S_k)
        Hcore_k_list.append(H_k)
        X_k_list.append(X_k)

    e_nuc = float(nuclear_repulsion_per_cell(system, lat_opts))
    print(f"  E_nuc per cell = {e_nuc:.10f} Ha (Ewald-3D)")
    sys.stdout.flush()

    # ---- Initial guess: diagonalise Hcore(k) per k → C, then build
    # D_real via the inverse-Bloch fold. This is the production
    # driver's HCORE path.
    C_per_k: List[np.ndarray] = []
    eps_per_k: List[np.ndarray] = []
    for H_k, X_k in zip(Hcore_k_list, X_k_list):
        C_k, eps_k = _diag_in_orth_basis(H_k, X_k)
        C_per_k.append(C_k.astype(complex))
        eps_per_k.append(eps_k)
    n_occ_per_k = [n_occ] * n_k
    D_real = real_space_density_from_kpoints(C_per_k, n_occ_per_k, kmesh, cells)

    # SAD override (multi-k convention from periodic_rhf_multi_k_ewald.py):
    # GuessEngine produces a unit-cell density; we overwrite D(g=0)
    # with it and zero the other cell blocks. Iter 1 builds F(D_SAD).
    if USE_SAD:
        D_engine = initial_density_closed_shell(
            system.unit_cell_molecule(), basis, n_occ,
            InitialGuess.SAD, is_periodic=True,
        )
        if D_engine is None:
            print("  WARNING: SAD requested but engine returned None; "
                  "falling back to Hcore guess")
        else:
            print(f"  initial guess: SAD (D_engine.shape = {D_engine.shape})")
            for g_idx in range(len(D_real.cells)):
                if (D_real.cells[g_idx].index == np.array([0, 0, 0])).all():
                    D_real.set_block(g_idx, D_engine)
                else:
                    D_real.set_block(g_idx,
                                     np.zeros_like(D_engine, dtype=float))
    else:
        print(f"  initial guess: HCORE (Hcore-diagonalise per k)")
    sys.stdout.flush()

    D_real_prev = None
    C_prev_per_k: List[np.ndarray] | None = None

    # ---- SCF loop with per-iter JSON dump
    iters_log: List[Dict[str, Any]] = []
    E_prev = 0.0

    for iter_idx in range(1, MAX_ITER + 1):
        t_iter = time.time()
        # Density damping (off by default here).
        D_used = D_real
        if iter_idx > 1 and DAMPING > 0.0:
            D_used = _damp_lattice_matrix(D_real, D_real_prev, DAMPING)

        F_k_list = build_periodic_fock_ewald3d_k(
            basis, system, D_used, omega=omega,
            k_points_cart=[np.asarray(k) for k in k_points],
            Hcore_k=Hcore_k_list,
            lattice_opts=lat_opts,
            grid_shape=grid_shape_t, origin=None, spacing_bohr=0.3,
        )

        # Per-k D(k), energy, gradient.
        E_elec = 0.0
        grad_norm_sum = 0.0
        D_k_list: List[np.ndarray] = []
        for idx in range(n_k):
            C_k = C_per_k[idx]
            C_occ = C_k[:, :n_occ]
            D_k = 2.0 * (C_occ @ C_occ.conj().T)
            D_k_list.append(D_k)
            H_k = Hcore_k_list[idx]
            F_k = F_k_list[idx]
            w = float(weights[idx])
            E_elec += w * 0.5 * np.real(np.trace(D_k @ (H_k + F_k)))
            S_k = S_k_list[idx]
            FDS = F_k @ D_k @ S_k
            grad = FDS - FDS.conj().T
            grad_norm_sum += w * float(np.linalg.norm(grad))
        D_g0 = np.asarray(_g0_block(D_real))
        S_g0 = np.asarray(_g0_block(S_lat))
        E_mad = _madelung_energy_correction_for_lat(D_g0, S_g0, system, lat_opts)
        E_total = float(E_elec) + e_nuc + E_mad
        dE = E_total - E_prev if iter_idx > 1 else 0.0

        # Apply LEVSHIFT before diagonalisation. Off when MODE=no_aids;
        # iter-decreasing per LevelShiftSchedule when MODE=levshift.
        level_shift_b = (
            LEVEL_SHIFT_SCHEDULE.at(iter_idx)
            if LEVEL_SHIFT_SCHEDULE is not None
            else 0.0
        )
        if level_shift_b != 0.0:
            F_for_diag: List[np.ndarray] = []
            for idx in range(n_k):
                D_k = D_k_list[idx]
                S_k = S_k_list[idx]
                F_shift = (
                    F_k_list[idx]
                    + level_shift_b * S_k
                    - (level_shift_b / 2.0) * (S_k @ D_k @ S_k)
                )
                F_shift = 0.5 * (F_shift + F_shift.conj().T)
                F_for_diag.append(F_shift)
        else:
            F_for_diag = F_k_list

        # Diagonalise each F(k) → C(k), ε(k).
        new_C_per_k: List[np.ndarray] = []
        new_eps_per_k: List[np.ndarray] = []
        for idx in range(n_k):
            C_k, eps_k = _diag_in_orth_basis(F_for_diag[idx], X_k_list[idx])
            new_C_per_k.append(C_k)
            new_eps_per_k.append(eps_k)

        # MOM: pick occupied columns by max S(k)-overlap with iter
        # k-1's occupied subspace. At iter 1 there's no previous to
        # compare against (C_prev_per_k is None), so default to Aufbau.
        if USE_MOM and C_prev_per_k is not None:
            for idx in range(n_k):
                C_k = new_C_per_k[idx]
                eps_k = new_eps_per_k[idx]
                S_k = S_k_list[idx]
                C_prev_occ = C_prev_per_k[idx][:, :n_occ]
                sel = mom_select(C_k, S_k, C_prev_occ, n_occ, eps_new=eps_k)
                n_kept_idx = C_k.shape[1]
                virt_mask = np.ones(n_kept_idx, dtype=bool)
                virt_mask[sel] = False
                virt_sel = np.where(virt_mask)[0]
                virt_sel = virt_sel[np.argsort(np.real(eps_k[virt_sel]))]
                order = np.concatenate([sel, virt_sel])
                new_C_per_k[idx] = C_k[:, order]
                new_eps_per_k[idx] = eps_k[order]

        # Dump BEFORE rotating C_per_k so the dump reflects this iter's
        # pre-extrapolation state.
        entry = dump_iter(
            iter_idx,
            E_total=E_total, E_elec=E_elec, E_nuc=e_nuc, E_madelung_fix=E_mad,
            dE=dE, grad_norm_sum=grad_norm_sum,
            F_k_list=F_k_list, D_k_list=D_k_list, S_k_list=S_k_list,
            eps_per_k=eps_per_k,        # MO energies from PREVIOUS diag
            C_per_k=C_per_k,            # MO coeffs from PREVIOUS diag
            C_prev_per_k=C_prev_per_k,
            D_real=D_real, n_occ=n_occ,
        )
        iters_log.append(entry)
        wall_iter = time.time() - t_iter
        sdmax = entry['per_k'][0].get('SD_eigval_max')
        sdmax_str = f"SDmax = {sdmax:.4f}" if sdmax is not None else ""
        mom_flag = "MOM" if (USE_MOM and C_prev_per_k is not None) else "   "
        print(
            f"  iter {iter_idx:>2}  E = {E_total:+.6f}   dE = {dE:+.3e}   "
            f"|FDS-SDF| = {grad_norm_sum:.3e}   "
            f"gap = {entry['per_k'][0]['gap']:+.2e}   "
            f"b = {level_shift_b:.2f}   {mom_flag}   "
            f"{sdmax_str}   ({wall_iter:.1f}s)"
        )
        sys.stdout.flush()

        # Rotate state for the next iter.
        C_prev_per_k = C_per_k
        C_per_k = new_C_per_k
        eps_per_k = new_eps_per_k
        D_real_new = real_space_density_from_kpoints(
            C_per_k, [n_occ] * n_k, kmesh, cells,
        )
        D_real_prev = D_used
        D_real = D_real_new
        E_prev = E_total

        # Flush partial JSON every iter so a hung job still gives data.
        with open(OUTPUT_PATH, "w") as f:
            json.dump({
                "system": "LiH primitive (FCC, a=4.084 Å)",
                "basis": "STO-3G",
                "kmesh": list(KMESH),
                "cutoff_bohr": CUTOFF_BOHR,
                "damping": DAMPING,
                "level_shift_schedule": (
                    LEVEL_SHIFT_SCHEDULE.as_list()
                    if LEVEL_SHIFT_SCHEDULE is not None else None
                ),
                "initial_guess": "SAD" if USE_SAD else "HCORE",
                "use_mom": USE_MOM,
                "mode": MODE,
                "use_diis": USE_DIIS,
                "fmixing": FMIXING,
                "n_occ": n_occ,
                "n_bf": int(basis.nbasis),
                "n_cells_in_lattice_sum": int(len(cells)),
                "iters": iters_log,
            }, f, indent=2)

    print(f"\n=== done. {len(iters_log)} iters dumped to {OUTPUT_PATH} ===")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
