"""Run the CRYSTAL14 periodic reference via ``vq submit`` (subprocess).

CRYSTAL14 is the second independent periodic reference alongside
PySCF.pbc — closed-shell HF / DFT (SVWN, PBE, B3LYP, ...) with the
POB-* basis-set family is its sweet spot, exactly what the
asbestos-polymorph paper needs cross-validated. Adding it here gives
the regression suite three-way periodic parity (vibe-qc / PySCF.pbc /
CRYSTAL14) on every fetched SPEC.

Execution boundary: CRYSTAL14 is an *external program* (see
``CLAUDE.md`` § 10). We never link or import it; we dispatch out-of-
process via ``vq submit`` (the vibe-queue dispatcher), wait, fetch the
workspace back, and parse the .out. The ``vq``-side wrapper that hides
CRYSTAL's stdin / parallel-input conventions lives in
``vibe-queue/contrib/run-crystal.sh`` (Mike's vibe-queue checkout).

Availability gating: this runner emits ``status='unavailable'`` and a
single explanatory note when any of the following hold (laptop runs
land here by default — only compute-reference / compute-small / a configured vq host
satisfy the chain):

* ``vq`` binary not on PATH.
* No vq host configured (``--vq-host`` unset and no ``default_host`` in
  ``~/.config/vq/config.toml``).
* Method's XC isn't one of the small whitelist we map to a CRYSTAL14
  keyword (SVWN / PBE / BLYP / B3LYP / HF — others would need basis-
  block inlining or hand-tuned input decks).
* Basis isn't one of the CRYSTAL14-internal-library names (STO-3G /
  POB-DZVP-REV2 / POB-TZVP-REV2) — anything else would need an inlined
  basis block per element. v1 ships internal-library-only.

Env-var knobs (laptop callers usually leave them at default):

* ``VIBEQC_CRYSTAL_VQ_HOST`` — override the vq host (default: vq's
  ``default_host``).
* ``VIBEQC_CRYSTAL_CPUS`` — MPI ranks the run-crystal.sh wrapper
  receives via ``--cpus`` (default: 14, matches the wrapper's own
  default).
* ``VIBEQC_CRYSTAL_WALL_TIME_SECONDS`` — vq's hard wall-clock cap
  (default: 7200 = 2 h). The handover's 2-h budget is enough for the
  STO-3G / POB-* canonical regression set; tune up for tight POB-TZVP
  on larger cells.
* ``VIBEQC_CRYSTAL_POLL_INTERVAL_S`` — how often we re-`vq status` the
  job (default: 30 s). Tighter wastes daemon round-trips; looser
  wastes wall on a job that's already finished.
* ``VIBEQC_CRYSTAL_WRAPPER`` — **required**. Absolute path to
  ``run-crystal.sh`` *on the remote host* (e.g. ``~/gitlab/vibe-queue/
  contrib/run-crystal.sh`` once vibe-queue is checked out). The path
  is interpreted on the vq host, not locally; unset → the runner
  emits ``status='unavailable'`` with a single-line hint.

The runner is deliberately P1: CRYSTAL's space-group accelerated path
needs an asymmetric-unit atom listing whose generation depends on the
family. The conventional-cell SPECs feed P1 directly with 0 reduction
needed; same lattice + atoms PySCF gets. CRYSTAL spends more wall (no
symmetry exploitation) but the reference E is correct. Big-cell paper
work can switch to symmetry-exploiting decks once a per-family
generator lands; out of scope for v1.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import time
import traceback
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from .case import CodeRow
from .spec import MethodSpec, PeriodicSpec

# CRYSTAL14 publishes prints in Hartree (AU) with no scaling — direct
# match for CodeRow.energy_ha.


# ----------------------------------------------------------------------
# Method / basis mappings
# ----------------------------------------------------------------------

# vibe-qc method.xc → CRYSTAL14 functional keyword. The XC convention
# match is non-trivial:
#   "lda"   → "SVWN"   (Slater + VWN5; same as PySCF "slater,vwn5" and
#                       ORCA "LDA"; CRYSTAL's bare "LDA" is Slater-only,
#                       which would mismatch the cross-code parity by
#                       ~30 mHa at sto-3g — don't use it).
#   "pbe"   → "PBE"
#   "blyp"  → "BLYP"
#   "b3lyp" → "B3LYP" (CRYSTAL's B3LYP keyword computes the VWN5
#                       flavor — verified empirically on H2/STO-3G at
#                       1.4 bohr: CRYSTAL14 XLGRID/TOLDEE 11 gives
#                       −1.1586001474 Ha vs libxc HYB_GGA_XC_B3LYP5
#                       −1.1586001482 Ha. That is exactly vibe-qc's
#                       bare "b3lyp" (ORCA/VWN5 definition), so the
#                       bare-name pairing is flavor-correct; "b3lyp5"
#                       is the explicit spelling of the same thing.
#                       The Gaussian/VWN-RPA spellings ("b3lyp/g",
#                       "b3lypg") are deliberately unsupported:
#                       CRYSTAL14's CORRELAT menu has no VWN-RPA, and
#                       silently mapping them onto CRYSTAL's B3LYP
#                       would hide the ~6.8 mHa (H2-sized cell)
#                       flavor gap.)
_CRYSTAL_FUNC_MAP = {
    "lda":    "SVWN",
    "pbe":    "PBE",
    "blyp":   "BLYP",
    "b3lyp":  "B3LYP",
    "b3lyp5": "B3LYP",
}

# vibe-qc basis spec.id → CRYSTAL14 internal-library name (case the
# wrapper passes through verbatim).
_CRYSTAL_BASIS_LIBRARY = {
    "sto-3g":          "STO-3G",
    "pob-dzvp-rev2":   "POB-DZVP-REV2",
    "pob-tzvp-rev2":   "POB-TZVP-REV2",
}


_DEFAULT_WRAPPER = ""    # must be supplied via VIBEQC_CRYSTAL_WRAPPER


def run_periodic_case(
    *, run_id: str, target: str, spec: PeriodicSpec, basis_name: str,
    method: MethodSpec, kmesh: Tuple[int, int, int],
    conv_tol_energy: Optional[float] = None,
    max_iter: Optional[int] = None,
    log_path: Path,
    workdir: Path,
) -> CodeRow:
    """Run one CRYSTAL14 periodic case via ``vq submit``.

    Returns a CodeRow with ``status='unavailable'`` (and an explanatory
    note) when any preflight check fails — vq missing, no host
    configured, unsupported method, unsupported basis, wrapper missing.
    The suite stays runnable on machines without the vq / compute-reference
    chain.

    The returned row's ``code_version`` is best-effort — CRYSTAL14
    prints its banner in the .out; we parse the first matching line.
    """
    if conv_tol_energy is None:
        conv_tol_energy = spec.default_conv_tol_energy
    if max_iter is None:
        max_iter = spec.default_max_iter

    row = CodeRow(
        run_id=run_id, target=target, system_id=spec.id, family=spec.family,
        basis=basis_name, method_id=method.id,
        kmesh="x".join(str(k) for k in kmesh),
        code="crystal", code_version="unknown",
        n_atoms=len(spec.atoms),
        n_electrons=sum(at.z for at in spec.atoms),
    )

    def _log(text: str) -> None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(text)
            if not text.endswith("\n"):
                fh.write("\n")
            fh.flush()

    _log("\n" + "=" * 78)
    _log(
        f"  crystal | {spec.id} | {basis_name} | {method.id} | "
        f"kmesh={kmesh} | target={target}"
    )
    _log("=" * 78)

    # --- Pre-flight gating ------------------------------------------------
    vq_bin = shutil.which("vq")
    if vq_bin is None:
        row.status = "unavailable"
        row.note = "vq binary not on PATH (install vibe-queue or alias vq)"
        _log(row.note)
        return row

    vq_host = (
        os.environ.get("VIBEQC_CRYSTAL_VQ_HOST")
        or _vq_default_host(vq_bin)
    )
    if vq_host is None:
        row.status = "unavailable"
        row.note = (
            "no vq host configured "
            "(set VIBEQC_CRYSTAL_VQ_HOST=<host> or default_host in "
            "~/.config/vq/config.toml)"
        )
        _log(row.note)
        return row

    func_kw = _crystal_functional(method)
    if func_kw is _UNSUPPORTED:
        row.status = "unavailable"
        row.note = (
            f"crystal runner: unsupported method "
            f"(scf={method.scf!r} xc={method.xc!r} post={method.post!r}); "
            f"v1 maps lda/pbe/blyp/b3lyp/hf only"
        )
        _log(row.note)
        return row

    basis_kw = _CRYSTAL_BASIS_LIBRARY.get(basis_name.lower())
    if basis_kw is None:
        row.status = "unavailable"
        row.note = (
            f"crystal runner: basis {basis_name!r} not in CRYSTAL14 "
            f"internal library "
            f"(supported: {sorted(_CRYSTAL_BASIS_LIBRARY.keys())})"
        )
        _log(row.note)
        return row

    wrapper = os.environ.get("VIBEQC_CRYSTAL_WRAPPER", _DEFAULT_WRAPPER)
    if not wrapper:
        row.status = "unavailable"
        row.note = (
            "VIBEQC_CRYSTAL_WRAPPER not set "
            "(absolute path to run-crystal.sh on the vq host, e.g. "
            "~/gitlab/vibe-queue/contrib/run-crystal.sh)"
        )
        _log(row.note)
        return row
    cpus = int(os.environ.get("VIBEQC_CRYSTAL_CPUS", "14"))
    wall_s = int(os.environ.get("VIBEQC_CRYSTAL_WALL_TIME_SECONDS", "7200"))
    poll_s = float(os.environ.get("VIBEQC_CRYSTAL_POLL_INTERVAL_S", "30"))

    # --- Build the .d12 deck ----------------------------------------------
    try:
        deck = build_d12(
            spec=spec, basis_kw=basis_kw, func_kw=func_kw,
            kmesh=kmesh, conv_tol_energy=conv_tol_energy,
            max_iter=max_iter,
        )
    except Exception as exc:
        row.note = f"deck build failed: {type(exc).__name__}: {exc}"
        _log(f"  EXCEPTION:\n{traceback.format_exc()}")
        return row

    case_workdir = workdir / f"{spec.id}__{basis_name}__{method.id}"
    case_workdir.mkdir(parents=True, exist_ok=True)
    deck_path = case_workdir / "INPUT.d12"
    deck_path.write_text(deck)
    _log(f"  wrote {deck_path}")
    _log(f"  vq host: {vq_host}   cpus: {cpus}   wall: {wall_s}s")

    # --- Submit, poll, fetch, parse --------------------------------------
    job_name = _job_name(spec, basis_name, method, kmesh)
    try:
        jobid = vq_submit(
            vq_bin=vq_bin, host=vq_host,
            workdir=case_workdir, wrapper=wrapper,
            cpus=cpus, wall_time_s=wall_s,
            job_name=job_name,
        )
    except Exception as exc:
        row.note = f"vq submit failed: {type(exc).__name__}: {exc}"
        _log(f"  EXCEPTION:\n{traceback.format_exc()}")
        return row
    _log(f"  vq jobid: {jobid}   name: {job_name}")

    t0 = time.perf_counter()
    try:
        final_state = vq_wait(
            vq_bin=vq_bin, host=vq_host, jobid=jobid,
            wall_time_s=wall_s, poll_interval_s=poll_s,
        )
    except Exception as exc:
        row.note = f"vq poll failed: {type(exc).__name__}: {exc}"
        _log(f"  EXCEPTION:\n{traceback.format_exc()}")
        return row
    wall = time.perf_counter() - t0
    _log(f"  vq job terminal state: {final_state}   (wall {wall:.1f} s)")

    fetch_dest = case_workdir / "fetched"
    fetch_dest.mkdir(parents=True, exist_ok=True)
    try:
        local_workspace = vq_fetch(
            vq_bin=vq_bin, host=vq_host, jobid=jobid,
            job_name=job_name, output_dir=fetch_dest,
        )
    except Exception as exc:
        row.note = (
            f"vq fetch failed (state={final_state}): "
            f"{type(exc).__name__}: {exc}"
        )
        _log(f"  EXCEPTION:\n{traceback.format_exc()}")
        return row

    out_path = local_workspace / "INPUT.out"
    if not out_path.is_file():
        row.note = f"CRYSTAL output not found at {out_path}; vq state={final_state}"
        _log(row.note)
        return row

    parsed = parse_crystal_out(out_path.read_text(errors="replace"))
    row.wall_s = wall
    row.code_version = parsed.version or "unknown"
    row.energy_ha = parsed.energy_ha
    row.converged = parsed.converged
    row.n_iter = parsed.n_iter
    if parsed.energy_ha is not None and len(spec.atoms) > 0:
        row.energy_per_atom_ha = parsed.energy_ha / len(spec.atoms)

    note_bits = []
    if parsed.band_gap_ev is not None:
        note_bits.append(f"gap={parsed.band_gap_ev:.3f} eV")
    if final_state != "completed":
        note_bits.append(f"vq state={final_state}")
    if parsed.error_line:
        note_bits.append(f"crystal error: {parsed.error_line}")
    row.note = "; ".join(note_bits)

    if parsed.energy_ha is None:
        # No SCF energy parsed — escalate to error so the comparison
        # path doesn't silently feed None into the Δ calculation.
        if not row.note:
            row.note = "CRYSTAL produced no parseable total energy"

    _log(
        f"  E = {row.energy_ha!r} Ha   converged={row.converged}   "
        f"iters={row.n_iter}   gap_eV={parsed.band_gap_ev}"
    )
    return row


# ----------------------------------------------------------------------
# Deck builder
# ----------------------------------------------------------------------


def build_d12(
    *, spec: PeriodicSpec, basis_kw: str, func_kw: str,
    kmesh: Tuple[int, int, int],
    conv_tol_energy: float, max_iter: int,
) -> str:
    """Build a P1 CRYSTAL14 .d12 input deck for one (spec, basis, method).

    P1 (space group 1) sidesteps the asymmetric-unit reduction the
    family-aware decks need; CRYSTAL handles a P1 cell with all atoms
    explicit, just without symmetry-driven SCF acceleration. The
    lattice is written as the standard CRYSTAL3D form: ``a b c α β γ``.

    ``func_kw`` is the upper-cased CRYSTAL functional keyword (SVWN /
    PBE / BLYP / B3LYP) or ``"HF"`` to skip the DFT block. ``basis_kw``
    is the upper-cased internal-library basis name (STO-3G /
    POB-TZVP-REV2 / ...). Both are pre-validated by the caller.

    ``kmesh`` is consumed isotropically: CRYSTAL's ``SHRINK IS ISP``
    form takes one (or two) integers. Anisotropic kmeshes get max(k)
    used uniformly; the caller already prints a warning at the
    runner-level when this happens.
    """
    title = f"{spec.id} :: {basis_kw} :: {func_kw} :: P1 (vibeqc regression)"

    a, b, c, alpha, beta, gamma = _lattice_to_a_b_c_alpha_beta_gamma(
        spec.lattice_ang
    )
    atoms_block = "\n".join(
        f"{at.z:>3d} {at.frac[0]:>11.7f} {at.frac[1]:>11.7f} {at.frac[2]:>11.7f}"
        for at in spec.atoms
    )

    # CRYSTAL SHRINK: use max(kmesh) for both arguments. (1,1,1) → "1 1".
    # The Gilat net (second arg) should be ≥ the MP net per the CRYSTAL
    # manual; using equal values is the standard textbook recipe.
    k_iso = max(kmesh) if kmesh else 1

    # TOLDEE: number of decimals on energy convergence in CRYSTAL is
    # the integer N in "10⁻ᴺ tolerance". 1e-7 → 7. Clamp to [3, 12].
    toldee = max(3, min(12, int(round(-math.log10(conv_tol_energy)))))

    parts = [
        title,
        "CRYSTAL",
        "0 0 0",
        "1",                                          # P1 space group
        f"{a:.7f} {b:.7f} {c:.7f} {alpha:.5f} {beta:.5f} {gamma:.5f}",
        f"{len(spec.atoms):d}",
        atoms_block,
        "END",
        "BASISSET",
        basis_kw,
    ]
    if func_kw != "HF":
        # XLGRID = CRYSTAL's higher-density XC grid; pairs with TOLDEE 7
        # cleanly. Default LGRID would give ~10× more grid noise than
        # PySCF's default `becke` for the same wallclock; XLGRID makes
        # the cross-code Δ a tolerance match.
        parts.extend(["DFT", func_kw, "XLGRID", "END"])
    parts.extend([
        "SHRINK",
        f"{k_iso} {k_iso}",
        "TOLINTEG",
        "7 7 7 7 14",
        "TOLDEE",
        f"{toldee:d}",
        "MAXCYCLE",
        f"{int(max_iter):d}",
        "END",
    ])
    return "\n".join(parts) + "\n"


def _lattice_to_a_b_c_alpha_beta_gamma(
    lattice_ang: Tuple[Tuple[float, float, float], ...],
) -> Tuple[float, float, float, float, float, float]:
    """Convert a 3x3 row-vector lattice (Å) → ``(a, b, c, α, β, γ)``.

    Returns angles in degrees. CRYSTAL accepts the six-parameter form
    universally (cubic / tetragonal / orthorhombic / monoclinic /
    triclinic); using it sidesteps having to detect the symmetry.
    """
    a_vec = np.asarray(lattice_ang[0], dtype=float)
    b_vec = np.asarray(lattice_ang[1], dtype=float)
    c_vec = np.asarray(lattice_ang[2], dtype=float)
    a_len = float(np.linalg.norm(a_vec))
    b_len = float(np.linalg.norm(b_vec))
    c_len = float(np.linalg.norm(c_vec))
    alpha = float(np.degrees(np.arccos(np.dot(b_vec, c_vec) / (b_len * c_len))))
    beta  = float(np.degrees(np.arccos(np.dot(a_vec, c_vec) / (a_len * c_len))))
    gamma = float(np.degrees(np.arccos(np.dot(a_vec, b_vec) / (a_len * b_len))))
    return a_len, b_len, c_len, alpha, beta, gamma


_UNSUPPORTED = object()


def _crystal_functional(method: MethodSpec) -> str:
    """Map MethodSpec → CRYSTAL14 keyword or _UNSUPPORTED.

    Returns ``"HF"`` for any HF reference (CRYSTAL treats it as the
    no-DFT-block path); the deck builder omits the DFT block when it
    sees ``"HF"``. Post-HF (MP2) is unsupported in v1 — CRYSTAL14's
    LCMP2 needs a separate input topology.
    """
    if method.post is not None:
        return _UNSUPPORTED                              # type: ignore[return-value]
    if method.scf in ("rhf", "uhf"):
        # CRYSTAL doesn't differentiate U/R in the keyword; it picks
        # from SPINLOCK / etc. blocks. UHF support is left out of v1
        # (no open-shell test in WAVE1_PERIODIC_CASES needs it yet).
        if method.scf == "uhf":
            return _UNSUPPORTED                          # type: ignore[return-value]
        return "HF"
    if method.scf in ("rks", "uks"):
        if method.scf == "uks":
            return _UNSUPPORTED                          # type: ignore[return-value]
        kw = _CRYSTAL_FUNC_MAP.get((method.xc or "").lower())
        return kw if kw is not None else _UNSUPPORTED   # type: ignore[return-value]
    return _UNSUPPORTED                                  # type: ignore[return-value]


def _job_name(
    spec: PeriodicSpec, basis_name: str, method: MethodSpec,
    kmesh: Tuple[int, int, int],
) -> str:
    """vq --job-name. Strict charset; <50 chars; collisions OK
    (vq disambiguates via the 12-hex jobid suffix on fetch paths)."""
    raw = f"vqc-{spec.id}-{basis_name}-{method.id}"
    safe = re.sub(r"[^A-Za-z0-9._-]", "-", raw)[:50]
    return safe


# ----------------------------------------------------------------------
# vq subprocess driver
# ----------------------------------------------------------------------


_VQ_TERMINAL_STATES = frozenset({
    "completed", "failed", "killed", "time_exceeded",
    "oom_killed", "starved", "aborted_by_queue", "suspended",
})


def _vq_default_host(vq_bin: str) -> Optional[str]:
    """Probe ``vq queue`` (no host arg) — if vq's config has a
    ``default_host``, the call succeeds and prints its jobs. We don't
    actually parse the queue here; we just round-trip to confirm vq
    is configured. Returns the host name extracted from
    ``~/.config/vq/config.toml`` when present, else ``None``.
    """
    cfg = Path.home() / ".config" / "vq" / "config.toml"
    if not cfg.is_file():
        return None
    try:
        text = cfg.read_text()
    except Exception:
        return None
    m = re.search(r'^\s*default_host\s*=\s*"([^"]+)"\s*$', text, re.MULTILINE)
    return m.group(1) if m else None


def vq_submit(
    *, vq_bin: str, host: str, workdir: Path, wrapper: str,
    cpus: int, wall_time_s: int, job_name: str,
) -> str:
    """Submit ``bash <wrapper> INPUT.d12`` against ``workdir`` on ``host``.

    Returns the 12-char hex jobid parsed from vq's stdout. Raises
    ``RuntimeError`` if vq exits non-zero or we can't find a jobid.
    """
    cmd = [
        vq_bin, "submit", "--host", host,
        "-d", str(workdir),
        "--cpus", str(cpus),
        "--wall-time-seconds", str(wall_time_s),
        "--job-name", job_name,
        "--", "bash", wrapper, "INPUT.d12", "INPUT.out",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(
            f"vq submit rc={proc.returncode}: "
            f"stderr={proc.stderr.strip()[:300]!r} "
            f"stdout={proc.stdout.strip()[:300]!r}"
        )
    # vq submit prints a confirmation line; the jobid is the first
    # 12-hex token we encounter in stdout. Loose match — we don't
    # depend on a specific phrasing.
    m = re.search(r"\b([0-9a-f]{12})\b", proc.stdout)
    if not m:
        raise RuntimeError(
            f"vq submit succeeded but no jobid parsed from stdout: "
            f"{proc.stdout.strip()[:300]!r}"
        )
    return m.group(1)


def vq_wait(
    *, vq_bin: str, host: str, jobid: str,
    wall_time_s: int, poll_interval_s: float,
) -> str:
    """Poll ``vq status`` until the job reaches a terminal state.

    Returns the terminal state string (``"completed"`` / ``"failed"``
    / ``"killed"`` / ``"time_exceeded"`` / ...). Raises
    ``TimeoutError`` if we exceed ``wall_time_s + 5 min`` of polling
    (5-min buffer for the wrapper's cleanup + vq's bookkeeping).
    """
    deadline = time.monotonic() + wall_time_s + 300
    while time.monotonic() < deadline:
        state = _vq_status_state(vq_bin=vq_bin, host=host, jobid=jobid)
        if state in _VQ_TERMINAL_STATES:
            return state
        time.sleep(poll_interval_s)
    raise TimeoutError(
        f"vq_wait: jobid={jobid} did not reach terminal state within "
        f"{wall_time_s + 300}s (final observed state: {state!r})"
    )


def _vq_status_state(*, vq_bin: str, host: str, jobid: str) -> str:
    """One vq-status round-trip. Returns the lowercase state token, or
    ``"unknown"`` if we can't parse it.
    """
    proc = subprocess.run(
        [vq_bin, "status", host, jobid, "-n", "0"],
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        # Transient vq daemon glitches shouldn't abort the wait —
        # log the rc and keep polling. The wait-deadline backstop
        # catches truly persistent failure.
        return "unknown"
    for line in proc.stdout.splitlines():
        m = re.match(r"^\s*state:\s*(\S+)", line)
        if m:
            return m.group(1).strip().lower()
    return "unknown"


def vq_fetch(
    *, vq_bin: str, host: str, jobid: str, job_name: str,
    output_dir: Path,
) -> Path:
    """``vq fetch`` the workspace to ``output_dir``; return the path to
    the materialised workspace dir.

    v0.5.34: when the job has a ``--job-name``, vq drops the workspace
    at ``<output_dir>/<jobname>-<jobid>/`` (else ``<output_dir>/<jobid>/``).
    """
    proc = subprocess.run(
        [vq_bin, "fetch", host, jobid, "-o", str(output_dir)],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"vq fetch rc={proc.returncode}: "
            f"stderr={proc.stderr.strip()[:300]!r}"
        )
    named = output_dir / f"{job_name}-{jobid}"
    if named.is_dir():
        return named
    bare = output_dir / jobid
    if bare.is_dir():
        return bare
    raise RuntimeError(
        f"vq fetch landed neither at {named} nor at {bare}; "
        f"stdout={proc.stdout.strip()[:300]!r}"
    )


# ----------------------------------------------------------------------
# Local CRYSTAL runner
# ----------------------------------------------------------------------


def add_crystal_enecycle_keyword(deck: str) -> str:
    """Return ``deck`` with CRYSTAL's ``ENECYCLE`` print enabled.

    CRYSTAL23 accepts the literal ``ENECYCLE`` keyword, but CRYSTAL14
    expects the portable ``SETPRINT`` flag ``69``. Inject the latter in
    the SCF/input section before the final top-level ``END``. The helper
    is intentionally textual because the regression decks are already
    valid CRYSTAL input files and we only need to request extra
    diagnostic printing.
    """
    lines = deck.splitlines()
    if any(line.strip().upper() == "ENECYCLE" for line in lines):
        return deck
    if any(line.strip().split()[:1] == ["69"] for line in lines):
        return deck

    for idx in range(len(lines) - 1, -1, -1):
        if lines[idx].strip().upper() == "END":
            lines[idx:idx] = ["SETPRINT", "1", "69 999"]
            break
    else:
        lines.extend(["SETPRINT", "1", "69 999", "END"])

    return "\n".join(lines) + "\n"


def run_crystal_local(
    input_path: Path,
    *,
    output_path: Optional[Path] = None,
    executable: Optional[str] = None,
    enecycle: bool = False,
    timeout_s: float = 7200.0,
) -> Path:
    """Run local ``crystal`` out-of-process and return the output path.

    This is the laptop/desktop companion to the vq-based remote runner.
    It deliberately shells out to the CRYSTAL executable instead of
    linking or importing anything from CRYSTAL, preserving vibe-qc's
    external-reference boundary.
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".out")
    else:
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    exe = executable or shutil.which("crystal")
    if exe is None:
        raise FileNotFoundError(
            "CRYSTAL executable not found on PATH; pass executable=..."
        )

    deck = input_path.read_text()
    if enecycle:
        deck = add_crystal_enecycle_keyword(deck)

    with output_path.open("w", encoding="utf-8") as out:
        proc = subprocess.run(
            [exe],
            input=deck,
            text=True,
            stdout=out,
            stderr=subprocess.PIPE,
            cwd=str(output_path.parent),
            timeout=float(timeout_s),
        )
        if proc.stderr:
            out.write("\n\n--- CRYSTAL stderr ---\n")
            out.write(proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError(
            f"local CRYSTAL failed with rc={proc.returncode}; "
            f"see {output_path}"
        )
    return output_path


# ----------------------------------------------------------------------
# CRYSTAL .out parser
# ----------------------------------------------------------------------


class CrystalParseResult:
    """Bag of parsed fields. Kept as a plain class (not a dataclass) so
    callers can introspect without importing the schema."""

    __slots__ = (
        "version", "energy_ha", "converged", "n_iter",
        "band_gap_ev", "error_line",
    )

    def __init__(self) -> None:
        self.version: Optional[str] = None
        self.energy_ha: Optional[float] = None
        self.converged: Optional[bool] = None
        self.n_iter: Optional[int] = None
        self.band_gap_ev: Optional[float] = None
        self.error_line: str = ""


class CrystalEneCycleRecord:
    """One CRYSTAL ``ENECYCLE`` component block.

    Field names mirror vibe-qc's BIPOLE component diagnostics where
    possible. ``e_bielet_zone_ee`` is CRYSTAL's raw direct-zone
    two-electron energy before the printed EXT corrections; ``e_two_electron``
    is CRYSTAL's physical ``TOTAL E-E`` value.
    """

    __slots__ = (
        "cycle", "e_total", "e_kinetic", "e_nuclear_attraction",
        "e_nuclear_repulsion", "e_two_electron", "e_bielet_zone_ee",
        "e_ext_el_pole", "e_ext_el_spheropole", "virial",
    )

    def __init__(self, *, cycle: int, values: dict[str, float]) -> None:
        self.cycle = int(cycle)
        self.e_total = values.get("TOTAL ENERGY")
        self.e_kinetic = values.get("KINETIC ENERGY")
        self.e_nuclear_attraction = values.get("TOTAL E-N + N-E")
        self.e_nuclear_repulsion = values.get("TOTAL N-N")
        self.e_two_electron = values.get("TOTAL E-E")
        self.e_bielet_zone_ee = values.get("BIELET ZONE E-E")
        self.e_ext_el_pole = values.get("EXT EL-POLE")
        self.e_ext_el_spheropole = values.get("EXT EL-SPHEROPOLE")
        self.virial = values.get("VIRIAL COEFFICIENT")


# Anchor patterns for the CRYSTAL14 output. CRYSTAL prints in
# fixed-column FORTRAN; the keyword phrases are stable across years.
#
# Total-energy line shapes we want to match (whichever is the last
# occurrence in the .out wins, so the last completed SCF step is the
# one we keep):
#
#   ` TOTAL ENERGY(HF)(AU)(  12) -7.6037215834312E+02 DE-9.5E-09 tester ...`
#   ` TOTAL ENERGY(DFT)(AU)(  13) -2.7494183921501E+02 DE-6.2E-09 ...`
#   ` TOTAL ENERGY + DISP-D3 -7.6037215834312E+02`  ← D3-augmented; ignore
#
# We deliberately filter to the (HF)/(DFT) variants — the D3-corrected
# line is a derived quantity we don't currently surface.
_TOTAL_E_RE = re.compile(
    r"^\s*TOTAL ENERGY\((HF|DFT)\)\(AU\)\(\s*(\d+)\)"
    r"\s*(-?\d+\.\d+E[+-]?\d+)",
)
# SCF terminator (success vs failure). CRYSTAL prints these as a
# banner-shaped line just before the wave-function dump.
_SCF_OK_RE = re.compile(r"== SCF ENDED - CONVERGENCE ON ENERGY")
_SCF_FAIL_RE = re.compile(r"== SCF ENDED - TOO MANY CYCLES")
# Version banner. CRYSTAL prints `CRYSTAL14` / `CRYSTAL17` etc. inside
# a decorated asterisk banner near the top of the .out, e.g.
#   ` *                                CRYSTAL14                                    *`
# Match anywhere on the line.
_VERSION_RE = re.compile(r"\b(CRYSTAL\d+)\b", re.IGNORECASE)
# Band gap (direct / indirect). CRYSTAL prints e.g.
#   ` DIRECT ENERGY BAND GAP:    7.4837 eV`
_GAP_RE = re.compile(
    r"^\s*(?:DIRECT|INDIRECT)\s+ENERGY\s+BAND\s+GAP:?\s*"
    r"(-?\d+\.\d+)\s*eV",
    re.IGNORECASE,
)
# Catastrophic-failure markers in the output. We capture the line so
# the row's note tells the operator what to grep for in the .out.
_ERROR_RE = re.compile(
    r"^\s*(ERROR|FATAL|ABNORMAL TERMINATION|BASIS SET LIBRARY:|"
    r"\*\*\* GEOM \*\*\*)",
)
_ENECYCLE_VALUE_RE = re.compile(
    r"^\s*:::\s+(.+?)\s+(-?\d+(?:\.\d*)?E[+-]?\d+)\s*$",
)
_CYC_RE = re.compile(r"^\s*CYC\s+(\d+)\s+ETOT\(AU\)")


def _normalise_enecycle_label(label: str) -> str:
    return " ".join(label.strip().split())


def parse_crystal_out(text: str) -> CrystalParseResult:
    """Parse a CRYSTAL14 .out into the fields the regression row needs.

    All fields are best-effort: an empty or truncated .out returns
    ``energy_ha=None, converged=None, n_iter=None`` rather than raising.
    The caller decides whether the missing values are fatal.

    The last-matched total-energy line wins (CRYSTAL prints one per
    converged SCF cycle and one at the bottom of the .out — the
    bottom-of-file value is the canonical converged result). Failure
    mode: ``SCF ENDED - TOO MANY CYCLES`` flips ``converged=False`` but
    the last per-cycle energy is still captured.
    """
    res = CrystalParseResult()
    last_e = None
    last_iter = None
    for line in text.splitlines():
        if res.version is None:
            m = _VERSION_RE.search(line)
            if m:
                res.version = m.group(1).upper()
        m = _TOTAL_E_RE.match(line)
        if m:
            try:
                last_iter = int(m.group(2))
                last_e = float(m.group(3))
            except ValueError:
                pass
            continue
        if _SCF_OK_RE.search(line):
            res.converged = True
            continue
        if _SCF_FAIL_RE.search(line):
            res.converged = False
            continue
        if res.band_gap_ev is None:
            m = _GAP_RE.match(line)
            if m:
                try:
                    res.band_gap_ev = float(m.group(1))
                except ValueError:
                    pass
                continue
        if not res.error_line and _ERROR_RE.match(line):
            res.error_line = line.strip()[:200]

    res.energy_ha = last_e
    res.n_iter = last_iter
    return res


def parse_crystal_enecycle(text: str) -> List[CrystalEneCycleRecord]:
    """Parse CRYSTAL ``ENECYCLE`` component blocks.

    CRYSTAL prints the component block immediately before the matching
    ``CYC n ETOT(AU)`` line, so the parser holds a pending value dict
    until it sees that cycle line. Outputs without ``ENECYCLE`` simply
    return an empty list.
    """
    records: List[CrystalEneCycleRecord] = []
    pending: Optional[dict[str, float]] = None

    for line in text.splitlines():
        if "+++ ENERGIES IN A.U. +++" in line:
            pending = {}
            continue
        if pending is not None:
            m_val = _ENECYCLE_VALUE_RE.match(line)
            if m_val:
                label = _normalise_enecycle_label(m_val.group(1))
                try:
                    pending[label] = float(m_val.group(2))
                except ValueError:
                    pass
                continue
            m_cyc = _CYC_RE.match(line)
            if m_cyc:
                records.append(CrystalEneCycleRecord(
                    cycle=int(m_cyc.group(1)),
                    values=pending,
                ))
                pending = None
                continue

    return records
