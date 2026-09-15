"""Real-Γ direct-torus efficiency ladder v2 — factored-K re-measure + adsorption probes.

Requested by the real-Γ AICCM development chat (2026-08-21) for the loop's
validation campaign. Two purposes:

  A. Re-measure the July ladder's five rungs now that the occupied-rank
     factored exchange landed (main 2effb3ad7) — quantify the lever
     end-to-end at production scale against the recorded baseline:

       rung                 fold(s)  SCF(s)  K/call(ms)  E/cell (Ha)
       R1 LiH-pob (2,2,2)     27.7     4.6        30.3   -8.058884213121
       R2 LiH-pob (3,3,3)    144.4    27.0      2749.3   -8.067814104022
       R3 H2 (4,4,4)         516.9     1.4       213.6   -1.116666998359
       R4a slab (3,3,1)       13.0     0.5         0.4   -1.115312290858
       R4b slab (4,4,1)       42.9     0.5         1.1   -1.114674520779

     ENERGIES MUST REPRODUCE. The factored K is machine-exact, so on the
     SAME host a drift beyond ~1e-9 is a REGRESSION — report loudly, stop.
     ACROSS hosts expect ~1e-8 (measured 2026-08-21: workstation reproduces
     the compute-study R1 row to 1.2e-8 with the factored path live); judge the
     cross-host rows at 1e-7 and only flag beyond that.

  B. Adsorption-class cost probes (the standing mission target): a molecule
     on 3x3 and 4x4 in-plane slab supercells with real vacuum. These are
     COST/FEASIBILITY probes, not chemistry — do not quote their energies
     as adsorption energies (no BSSE, no geometry relaxation, minimal
     basis on the larger rungs).

Per-rung records: fold time (symmetry=True), SCF time + iterations,
E/cell, exchange_q0, mean K/call for BOTH the factored and dense
contraction (the speedup number), L shape/GB, peak RSS.

RSS UNITS: ru_maxrss is KiB on Linux, bytes on macOS — corrected below.

Local pre-flight (workstation, 2026-08-21, factored path live) so the fleet
run has sanity anchors:
  A1  fold 75.3 s, SCF 9.2 s, K 189.2 -> 82.4 ms (2.30x), |dK| 3.2e-14,
      E/cell -8.058884200989 (compute-study baseline -8.058884213121; 1.2e-8 apart
      cross-host, as expected)
  A2  fold 641 s, SCF 42.5 s, K 16250.8 -> 4253.9 ms (3.82x), |dK| 2.3e-13,
      E/cell -8.067814091890 (baseline -8.067814104022)
  B0  (an extra (1,1,1) rung of the B fixture, not in RUNGS): 10 atoms,
      26 bf/cell, 18 e-, closed shell; fold 155.9 s, SCF 19.4 s, 18 iters,
      E -32.360911, L(312,26,26). The 34-bohr vacuum box dominates the
      fold cost -- B2/B3 are the rungs that will find the wall. Size the
      job generously (see the submit note in the request).
"""
import json, os, platform, resource, sys, time
import numpy as np

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.neutral import ccm_neutral_cderi_fold
from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct
from vibeqc.periodic.ccm.ri import (
    ccm_ri_k_neutral, ccm_ri_k_neutral_factored, psd_density_factor)

OUT = os.environ.get("VQ_WORKDIR", ".") + "/eff_ladder_v2.json"


def rss_gb():
    m = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return m / 1024**2 if sys.platform != "darwin" else m / 1024**3


def lih_pob(nrep):
    a = 7.72
    lat = 0.5 * a * np.array([[0., 1, 1], [1, 0, 1], [1, 1, 0]]).T
    cell = PeriodicSystem(3, lat, [Atom(3, [0, 0, 0]),
                                   Atom(1, [0.5 * a] * 3)], 0, 1)
    return CCMSystem(cell, nrep, "pob-tzvp-rev2")


def h2_cubic(nrep):
    cell = PeriodicSystem(3, np.diag([6., 6., 6.]),
                          [Atom(1, [3., 3., 2.3]), Atom(1, [3., 3., 3.7])], 0, 1)
    return CCMSystem(cell, nrep, "sto-3g")


def h2_slab(nrep):
    cell = PeriodicSystem(3, np.diag([5., 5., 25.]),
                          [Atom(1, [2.5, 2.5, 11.8]), Atom(1, [2.5, 2.5, 13.2])], 0, 1)
    return CCMSystem(cell, nrep, "sto-3g")


def lih_slab_with_adsorbate(nrep, basis="sto-3g", adsorbate=True):
    """2-layer LiH(100)-like slab, 30-bohr vacuum, optional H2 adsorbate.

    Declared-3D periodic-in-vacuum (the route's supported slab posture);
    in-plane supercell comes from nrep. Cost probe, not chemistry.
    """
    a = 4.08  # in-plane Li-H spacing (bohr), rocksalt-like
    atoms = [Atom(3, [0., 0., 10.0]), Atom(1, [a, 0., 10.0]),
             Atom(1, [0., a, 10.0]), Atom(3, [a, a, 10.0]),
             Atom(3, [0., 0., 10.0 + a]), Atom(1, [a, 0., 10.0 + a]),
             Atom(1, [0., a, 10.0 + a]), Atom(3, [a, a, 10.0 + a])]
    if adsorbate:  # H2 above the surface
        atoms += [Atom(1, [0., 0., 10.0 + a + 4.0]),
                  Atom(1, [0., 0., 10.0 + a + 5.4])]
    cell = PeriodicSystem(3, np.diag([2 * a, 2 * a, 34.0]), atoms, 0, 1)
    return CCMSystem(cell, nrep, basis)


# Fold-threading sweep (the WSC/q-axis lever, landed 2026-08-21 after this
# payload's first version). Runs on the two rungs where the fold dominates.
# BLAS MUST be pinned for this to mean anything -- an outer pool over a
# multi-threaded BLAS oversubscribes; the submit sets OMP_NUM_THREADS=1.
THREAD_SWEEP = [1, 2, 4, 8, 16]

RUNGS = [
    ("A1-LiH-pob-222",  lambda: lih_pob((2, 2, 2))),
    ("A2-LiH-pob-333",  lambda: lih_pob((3, 3, 3))),
    ("A3-H2-444",       lambda: h2_cubic((4, 4, 4))),
    ("A4a-slab-331",    lambda: h2_slab((3, 3, 1))),
    ("A4b-slab-441",    lambda: h2_slab((4, 4, 1))),
    # Adsorption-class cost probes (mission target sizes).
    ("B1-LiHslab+H2-221", lambda: lih_slab_with_adsorbate((2, 2, 1))),
    ("B2-LiHslab+H2-331", lambda: lih_slab_with_adsorbate((3, 3, 1))),
    ("B3-LiHslab+H2-441", lambda: lih_slab_with_adsorbate((4, 4, 1))),
]

def main():
    records = []
    for name, mk in RUNGS:
        try:
            ccm = mk()
            t0 = time.perf_counter()
            L = ccm_neutral_cderi_fold(ccm, symmetry=True)
            t_fold = time.perf_counter() - t0

            t0 = time.perf_counter()
            scf = run_ccm_rhf_direct(ccm, cderi=L)
            t_scf = time.perf_counter() - t0

            # K contraction: factored (production) vs dense (reference).
            D = scf.density
            t0 = time.perf_counter(); X = psd_density_factor(D)
            K_f = ccm_ri_k_neutral_factored(L, X); t_fact = time.perf_counter() - t0
            t0 = time.perf_counter(); K_d = ccm_ri_k_neutral(L, D)
            t_dense = time.perf_counter() - t0

            records.append(dict(
                rung=name, n_atoms_sc=int(ccm.n_atoms), n_cells=int(ccm.n_cells),
                basis=ccm.basis_name, nrep=list(ccm.nrep),
                t_fold_s=round(t_fold, 1), t_scf_s=round(t_scf, 1),
                n_iter=int(scf.n_iter), converged=bool(scf.converged),
                e_per_cell=float(scf.energy) / int(ccm.n_cells),
                exchange_q0=str(scf.exchange_q0),
                l_shape=list(L.shape), l_gb=round(L.nbytes / 1024**3, 3),
                occ_rank=int(X.shape[1]),
                t_k_factored_ms=round(t_fact * 1e3, 2),
                t_k_dense_ms=round(t_dense * 1e3, 2),
                k_speedup=round(t_dense / max(t_fact, 1e-9), 2),
                k_max_abs_diff=float(np.max(np.abs(K_f - K_d))),
                rss_gb=round(rss_gb(), 2)))
        except Exception as exc:
            records.append(dict(rung=name, failed=f"{type(exc).__name__}: {exc}"))
        print(json.dumps(records[-1]), flush=True)

        # --- fold-threading sweep -------------------------------------------
    thread_rows = []
    for label, mk, nrep in (("H2-331", h2_slab, (3, 3, 1)),
                            ("LiH-pob-222", lih_pob, (2, 2, 2))):
        try:
            ccm = mk(nrep)
            base = None
            for nt in THREAD_SWEEP:
                t0 = time.perf_counter()
                L = ccm_neutral_cderi_fold(ccm, symmetry=True, fold_threads=nt)
                dt = time.perf_counter() - t0
                if base is None:
                    base, L_ref = dt, L
                    same = True
                else:
                    same = bool(np.array_equal(L, L_ref))
                row = dict(fixture=label, threads=nt, t_fold_s=round(dt, 1),
                           speedup=round(base / max(dt, 1e-9), 2),
                           bit_identical_to_serial=same)
                thread_rows.append(row)
                print(json.dumps(row), flush=True)
        except Exception as exc:
            row = dict(fixture=label, failed=f"{type(exc).__name__}: {exc}")
            thread_rows.append(row)
            print(json.dumps(row), flush=True)

    with open(OUT, "w") as fh:
        json.dump(dict(host=platform.node(), records=records,
                       thread_sweep=thread_rows,
                       omp_num_threads=os.environ.get("OMP_NUM_THREADS")),
                  fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
