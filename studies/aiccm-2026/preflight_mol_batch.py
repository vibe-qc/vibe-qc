#!/usr/bin/env python
"""Pre-flight check for the molecular benchmark batch.

Run before ``python batch_full_matrix.py | sh`` to verify:

  1. vibe-qc is importable and the version banner prints
  2. A trivial test calculation (H₂O / RHF / STO-3G) completes
  3. Each converger setting produces valid options without errors
  4. COSX auto-detection of the auxiliary basis works
  5. vq is configured correctly (if not --local)

Exits 0 on success, non-zero on any failure.

Usage:
    python preflight_mol_batch.py              # check everything
    python preflight_mol_batch.py --local       # skip vq checks
    python preflight_mol_batch.py --quick       # import check only (fastest)
"""

from __future__ import annotations

import site_settings

import argparse
import subprocess
import sys
import textwrap
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"
BOLD = "\033[1m"

_ok = 0


def _pass(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def _fail(msg: str) -> None:
    global _ok
    _ok += 1
    print(f"  {RED}✗{RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET} {msg}")


# ── Step 1: vibe-qc import ──────────────────────────────────────────────────

def check_import() -> bool:
    print(f"\n{BOLD}[1/5] vibe-qc import{RESET}")
    try:
        import vibeqc as vq
        _pass(f"vibe-qc {vq.VIBEQC_VERSION} imported")
        return True
    except ImportError as e:
        _fail(f"vibe-qc not importable: {e}")
        print("  → Check that the vibeqc-dev venv is active, or set PYTHONPATH.")
        return False


# ── Step 2: trivial test calculation ────────────────────────────────────────

def check_trivial_calc() -> bool:
    print(f"\n{BOLD}[2/5] Trivial test calculation (H₂O / RHF / STO-3G){RESET}")
    try:
        import vibeqc as vq
    except ImportError:
        return False

    mol = vq.Molecule([
        vq.Atom(8, [0.0, 0.00, 0.00]),
        vq.Atom(1, [0.0, 1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ])

    t0 = time.perf_counter()
    try:
        result = vq.run_rhf(mol, "sto-3g")
    except Exception as e:
        _fail(f"RHF/STO-3G failed: {e}")
        return False

    dt = time.perf_counter() - t0
    if not result.converged:
        _fail(f"RHF did not converge in {result.n_iter} iterations")
        return False

    _pass(f"RHF converged: E = {result.energy:.8f} Ha  "
          f"({result.n_iter} iters, {dt*1000:.0f} ms)")
    return True


# ── Step 3: converger options validation ────────────────────────────────────

_CONVERGERS = [
    ("ediis-diis", "EDIIS+DIIS", "opts.scf_accelerator = vq.SCFAccelerator.EDIIS_DIIS"),
    ("kdiis", "KDIIS", "opts.scf_accelerator = vq.SCFAccelerator.KDIIS"),
    ("ad-cdiis", "AD-CDIIS", "opts.scf_accelerator = vq.SCFAccelerator.AD_CDIIS\nopts.diis_adaptive_delta = 1e-4"),
    ("newton", "Newton", "opts.newton_threshold = 1.0"),
    ("trah", "TRAH", "opts.trah_threshold = 1.0"),
    ("quadratic", "Quadratic", "opts.quadratic_fallback_iter = 30"),
    ("damping", "Damping-only", "opts.use_diis = False\nopts.damping = 0.7\nopts.fock_mixing = 0.3"),
]

def check_convergers() -> bool:
    print(f"\n{BOLD}[3/5] Converger options validation{RESET}")
    try:
        import vibeqc as vq
    except ImportError:
        return False

    all_ok = True
    for key, label, code in _CONVERGERS:
        try:
            opts = vq.RHFOptions()
            exec(textwrap.dedent(code), {"opts": opts, "vq": vq})
            _pass(f"{label}: valid")
        except Exception as e:
            _fail(f"{label}: {e}")
            all_ok = False
    return all_ok


# ── Step 4: COSX / aux_basis auto-detection ─────────────────────────────────

_BASES = ["sto-3g", "6-31g*", "def2-svp", "cc-pvdz"]

def check_cosx_aux() -> bool:
    print(f"\n{BOLD}[4/5] COSX auxiliary basis auto-detection{RESET}")
    try:
        import vibeqc as vq
    except ImportError:
        return False

    all_ok = True
    for basis in _BASES:
        try:
            aux = vq.default_aux_basis_for(basis, kind="jk")
            _pass(f"{basis} → {aux}")
        except Exception as e:
            _warn(f"{basis}: auto-detect failed ({e}) — will use def2-universal-jfit fallback")
    return all_ok


# ── Step 5: vq configuration ────────────────────────────────────────────────

def check_vq() -> bool:
    print(f"\n{BOLD}[5/5] vq cluster configuration{RESET}")
    try:
        hosts = site_settings.molecular_hosts()
    except ValueError as error:
        _fail(str(error))
        return False
    try:
        result = subprocess.run(
            ["vq", "status"], capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        _warn("vq not on PATH — use --local mode or install vq")
        return True  # not a failure, just a warning
    except subprocess.TimeoutExpired:
        _fail("vq status timed out")
        return False

    if result.returncode != 0:
        _fail(f"vq status returned {result.returncode}")
        print(f"  stderr: {result.stderr.strip()[:200]}")
        return False

    # Check only the explicitly selected private molecular host rotation.
    for host in hosts:
        r2 = subprocess.run(
            ["vq", "status", "--host", host],
            capture_output=True, text=True, timeout=10)
        if r2.returncode == 0 and host in r2.stdout:
            _pass(f"vq can reach {host}")
        else:
            _warn(f"vq cannot reach {host} — jobs targeting it will queue")

    return True


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Pre-flight check for the molecular benchmark batch")
    ap.add_argument("--local", action="store_true",
                    help="skip vq cluster checks")
    ap.add_argument("--quick", action="store_true",
                    help="import check only (fastest)")
    args = ap.parse_args()

    print(f"{BOLD}vibe-qc molecular batch — pre-flight check{RESET}")
    print(f"  Scripts dir: {HERE / 'batch_scripts'}")
    n_scripts = len(list((HERE / 'batch_scripts').glob('*.py'))) if (HERE / 'batch_scripts').exists() else 0
    print(f"  Jobs ready:  {n_scripts}")

    check_import()
    if args.quick:
        sys.exit(_ok)

    if not check_trivial_calc():
        print(f"\n{RED}Trivial calculation failed — fix before launching the batch.{RESET}")
        sys.exit(1)

    check_convergers()
    check_cosx_aux()

    if not args.local:
        check_vq()
    else:
        print(f"\n{BOLD}[5/5] vq cluster configuration{RESET}")
        _pass("skipped (--local mode)")

    # Final verdict
    if _ok == 0:
        print(f"\n{GREEN}{BOLD}All checks passed. Ready to launch:{RESET}")
        print(f"  python batch_full_matrix.py | sh")
    else:
        print(f"\n{RED}{BOLD}{_ok} check(s) failed. Fix before launching.{RESET}")

    sys.exit(_ok)


if __name__ == "__main__":
    main()
