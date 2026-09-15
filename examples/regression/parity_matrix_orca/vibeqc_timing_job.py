"""Runs ON compute-reference: time one vibe-qc SCF, emit a machine-readable result.

Submitted by ``run_speed_benchmark.py`` as a ``vq`` job in the
vibeqc-dev venv. It reads a ``cell.json`` (written into the job
workspace next to this file), runs the SCF, and prints one
``VIBEQC-TIMING-RESULT:{json}`` line — the same marker-line pattern
``runner_pyscf.py`` uses for its external PySCF process.

This file is the *only* part of the speed benchmark that imports
vibe-qc, and it does so on compute-reference — never in the orchestrating
process (which only shells out to ``vq``).

cell.json schema::

    {
      "atoms": [[Z, [x, y, z]], ...],   # nuclear coords in BOHR
      "multiplicity": 1,
      "basis": "def2-svp",
      "method": "RHF",                  # RHF|UHF|RKS-*|UKS-*
      "df": false,                      # density-fitted (RIJK) Fock build
      "cosx": false,                    # RIJCOSX (RI-J + COSX-K); implies df
      "aux_basis": "def2-universal-jkfit",
      "conv_tol_energy": 1e-9,
      "label": "H2O__def2-svp__RHF"     # echoed back for sanity
    }
"""

from __future__ import annotations

import json
import sys
import time
import traceback

RESULT_MARKER = "VIBEQC-TIMING-RESULT:"


def _emit(payload: dict) -> None:
    print(RESULT_MARKER + json.dumps(payload, sort_keys=True), flush=True)


def _run(spec: dict) -> dict:
    import vibeqc
    from vibeqc import (
        Atom,
        BasisSet,
        InitialGuess,
        Molecule,
        RHFOptions,
        RKSOptions,
        UHFOptions,
        UKSOptions,
        run_rhf,
        run_rks,
        run_uhf,
        run_uks,
    )

    mol = Molecule(
        [Atom(int(z), list(xyz)) for z, xyz in spec["atoms"]],
        multiplicity=int(spec["multiplicity"]),
    )
    basis = BasisSet(mol, spec["basis"])
    cosx = bool(spec.get("cosx", False))
    # cosx (RIJCOSX) is built on the DF-J machinery, so it implies df.
    df = bool(spec.get("df", False)) or cosx
    aux = spec.get("aux_basis", "def2-universal-jkfit") if df else ""
    method = spec["method"]
    conv = float(spec.get("conv_tol_energy", 1e-9))

    def _apply_fock_path(o):
        o.density_fit = df
        o.aux_basis = aux
        o.cosx = cosx
        # Pin SAD as the initial guess. Since commit `c31c9f5`
        # (``guess: AUTO is now the default initial guess``, v0.9.x)
        # the default AUTO resolves to SAP for closed-shell light-atom
        # systems; SAP doesn't converge well on long n-alkanes
        # (n-decane, n-hexadecane in the benchmark's speed ladder).
        # SAD converges those in ~26 iters at the benchmark's
        # ``conv_tol_energy=1e-9``. Pinning it here keeps the speed
        # benchmark calibrated to a guess that exercises every cell —
        # the benchmark measures Fock-build wall time, not guess
        # heuristics. When the AUTO resolver is fixed for long
        # alkanes, this pin can be removed.
        o.initial_guess = InitialGuess.SAD
        # For the direct path, opt in to the Almlöf-style incremental
        # ΔP Fock build (off by default; pair with the two-phase
        # Schwarz tightening that's on by default). The incremental
        # ΔD path has an intrinsic drift floor of ~1e-7 Ha — clamp
        # conv_tol_energy to 1e-7 so the SCF converges cleanly.
        # density_fit / cosx paths have their own amortisation and
        # ignore this flag.
        if not df:
            o.incremental_fock = True
            if o.conv_tol_energy < 1e-7:
                o.conv_tol_energy = 1e-7

    if method == "RHF":
        o = RHFOptions()
        o.conv_tol_energy = conv
        _apply_fock_path(o)
        runner, args = run_rhf, (mol, basis, o)
    elif method == "UHF":
        o = UHFOptions()
        o.conv_tol_energy = conv
        o.max_iter = 300
        _apply_fock_path(o)
        runner, args = run_uhf, (mol, basis, o)
    elif method.startswith("RKS-"):
        o = RKSOptions()
        o.functional = method.split("-", 1)[1]
        o.conv_tol_energy = conv
        o.max_iter = 300
        _apply_fock_path(o)
        runner, args = run_rks, (mol, basis, o)
    elif method.startswith("UKS-"):
        o = UKSOptions()
        o.functional = method.split("-", 1)[1]
        o.conv_tol_energy = conv
        o.max_iter = 500
        o.damping = 0.7
        _apply_fock_path(o)
        runner, args = run_uks, (mol, basis, o)
    else:
        raise ValueError(f"vibeqc_timing_job: unknown method {method!r}")

    # Time the SCF only — Molecule / BasisSet construction is cheap and
    # not the thing the benchmark compares.
    t0 = time.perf_counter()
    result = runner(*args)
    wall = time.perf_counter() - t0

    return {
        "status": "ok",
        "label": spec.get("label", ""),
        "code": "vibeqc",
        "code_version": getattr(vibeqc, "__version__", "unknown"),
        "method": method,
        "basis": spec["basis"],
        "df": df,
        "cosx": cosx,
        "wall_s": wall,
        "energy_ha": float(result.energy),
        "converged": bool(result.converged),
        "n_iter": int(result.n_iter),
    }


def main() -> int:
    if len(sys.argv) != 2:
        _emit({"status": "error", "note": f"usage: {sys.argv[0]} <cell.json>"})
        return 2
    try:
        spec = json.loads(open(sys.argv[1], encoding="utf-8").read())
    except Exception as exc:  # noqa: BLE001
        _emit({"status": "error", "note": f"cannot read cell.json: {exc}"})
        return 2
    try:
        _emit(_run(spec))
    except Exception as exc:  # noqa: BLE001
        _emit(
            {
                "status": "error",
                "label": spec.get("label", ""),
                "note": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
