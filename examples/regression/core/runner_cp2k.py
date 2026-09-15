"""Run the CP2K periodic reference via local subprocess.

CP2K is the canonical reference for Gaussian + plane-wave (GPW) and
Gaussian-augmented plane-wave (GAPW) calculations — Lippert & Hutter
1999 implemented in CP2K is the upstream the v0.10.x GAPW chat
(``docs/design_periodic_gapw.md``) is reproducing. Neither
PySCF.pbc (Gaussian-only) nor CRYSTAL14 (Gaussian-only) exercises
the GPW/GAPW J build, so CP2K is the only honest parity oracle.

Execution boundary: CP2K is an **external program** (CLAUDE.md
§ 10). We never link or import CP2K; this runner shells out to the
``cp2k.psmp`` / ``cp2k.popt`` / ``cp2k.sopt`` executable, parses the
.out, and emits a :class:`CodeRow`. No ``import cp2k`` anywhere in
``python/vibeqc/`` or ``cpp/``.

Availability gating: emits ``status='unavailable'`` with a single
explanatory note when any of the following hold (laptop runs without
CP2K installed land here cleanly — no spurious failures):

* No CP2K executable on PATH (``cp2k.psmp``, ``cp2k.popt``,
  ``cp2k.sopt``, or plain ``cp2k``).
* Method's XC isn't one of the small initial whitelist (LDA / PBE
  / BLYP / B3LYP — flavor-matched: CP2K's B3LYP preset is the VWN5
  flavor, exactly vibe-qc's bare ``b3lyp``). UHF / UKS / MP2 are out of M1f scope; the GPW
  open-shell parity is M3 work.
* Basis isn't one of the GTH-pseudopotential basis names the
  shipped data files cover (DZVP-MOLOPT-SR-GTH /
  TZVP-MOLOPT-GTH / SZV-MOLOPT-SR-GTH). Adding more is a one-line
  mapping change.

Env-var knobs (laptop callers usually leave them at default):

* ``VIBEQC_CP2K_EXECUTABLE`` — explicit path to the binary. The
  default probes the PATH in the order
  ``cp2k.psmp → cp2k.popt → cp2k.sopt → cp2k``.
* ``VIBEQC_CP2K_BASIS_SET_FILE`` — path to the
  ``BASIS_MOLOPT`` file shipped with CP2K (the default points at
  CP2K's standard install location, then falls back to a
  ``CP2K_DATA_DIR`` env var that the user can set without changing
  this runner).
* ``VIBEQC_CP2K_POTENTIAL_FILE`` — path to ``GTH_POTENTIALS``;
  same fallback semantics.
* ``VIBEQC_CP2K_CPUS`` — MPI ranks to launch (default: 1, i.e.
  serial). Only consulted when the binary is ``cp2k.psmp``;
  ``cp2k.popt`` / ``cp2k.sopt`` ignore it.
* ``VIBEQC_CP2K_PW_CUTOFF_RY`` — plane-wave cutoff in Ry
  (default: 300). The M2 GPW driver's cutoff sweep against this
  runner will probe convergence behaviour.
* ``VIBEQC_CP2K_REL_CUTOFF_RY`` — rel-cutoff for the multigrid
  (default: 60, CP2K's standard).
* ``VIBEQC_CP2K_WALL_TIME_SECONDS`` — subprocess timeout
  (default: 1800 = 30 min).

M1f scope (this commit): GPW closed-shell RKS + RHF on simple
cells with LDA / PBE / BLYP / B3LYP. GAPW (METHOD GAPW), open-shell
UHF / UKS, and MP2 are deliberately deferred — the gapw chat's
M3 / M4 milestones extend this runner as their first task.

Mirror of ``runner_crystal.py`` for the file shape + availability-
gating + parser pattern; mirror of ``runner_pyscf.py`` for the
emit-CodeRow convention.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
import traceback
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from .case import CodeRow
from .spec import MethodSpec, PeriodicSpec

# ----------------------------------------------------------------------
# Method / basis mappings
# ----------------------------------------------------------------------

# vibe-qc method.xc → CP2K XC_FUNCTIONAL keyword. The CP2K
# convention is the upper-case "official" functional name; for
# composite functionals (B3LYP, ...) it picks the most-cited variant
# unless an explicit override is given.
#
# Notes on convention matching:
#   "lda"   → "LDA"   (CP2K's bare LDA is Slater + VWN5, matching
#                      ORCA/CRYSTAL/PySCF "lda,vwn5"; this is the
#                      consistent vibe-qc reference convention.)
#   "pbe"   → "PBE"
#   "blyp"  → "BLYP"
#   "b3lyp" → "B3LYP" (CP2K's B3LYP preset uses VWN5 per the CP2K
#                      docs — same camp as CRYSTAL's B3LYP keyword and
#                      ORCA's bare B3LYP, and exactly vibe-qc's bare
#                      "b3lyp" (ORCA/VWN5 definition); "b3lyp5" is
#                      the explicit spelling of the same flavor. The
#                      Gaussian/VWN-RPA spellings ("b3lyp/g",
#                      "b3lypg") are deliberately unsupported —
#                      mapping them onto CP2K's preset would hide the
#                      ~10-15 mHa/heavy-atom flavor gap, mirroring
#                      the CRYSTAL runner.)
_CP2K_FUNC_MAP = {
    "lda": "LDA",
    "pbe": "PBE",
    "blyp": "BLYP",
    "b3lyp": "B3LYP",
    "b3lyp5": "B3LYP",
}

# vibe-qc basis.id → CP2K basis-set name (from BASIS_MOLOPT). These
# are GTH-pseudopotential basis sets — the GPW route requires
# pseudopotentials by construction, so we only expose pseudo-paired
# basis sets here. GAPW (all-electron) will get a separate mapping
# at M3 with ALLELECTRON_GAPW basis files.
_CP2K_BASIS_MAP = {
    # MOLOPT family — the workhorse for production GPW. SR = single
    # set for s + p; STR = "short range" variants for solids.
    "dzvp-molopt-sr-gth": "DZVP-MOLOPT-SR-GTH",
    "tzvp-molopt-gth": "TZVP-MOLOPT-GTH",
    "tzv2p-molopt-gth": "TZV2P-MOLOPT-GTH",
    "szv-molopt-sr-gth": "SZV-MOLOPT-SR-GTH",
}

# vibe-qc xc → CP2K GTH-pseudopotential suffix. Each pseudopotential
# is generated against a specific functional; using the LDA pseudo
# with a PBE calculation introduces a ~10 mHa systematic shift, so
# the mapping is per-XC.
_CP2K_PSEUDO_SUFFIX = {
    "lda": "PADE",  # GTH-PADE = LDA pseudopotential
    "pbe": "PBE",
    "blyp": "BLYP",
    "b3lyp": "BLYP",   # no separate B3LYP GTH; BLYP pseudo is the
    "b3lyp5": "BLYP",  # closest match in the GTH set. Documented in
    # the note.
}


_UNSUPPORTED = object()


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def run_periodic_case(
    *,
    run_id: str,
    target: str,
    spec: PeriodicSpec,
    basis_name: str,
    method: MethodSpec,
    kmesh: Tuple[int, int, int],
    conv_tol_energy: Optional[float] = None,
    max_iter: Optional[int] = None,
    log_path: Path,
    workdir: Path,
) -> CodeRow:
    """Run one CP2K GPW periodic case via local ``cp2k.*`` subprocess.

    Returns a CodeRow with ``status='unavailable'`` (+ explanatory
    note) when CP2K isn't installed, the method isn't on the
    whitelist, or the basis isn't a known GTH-pseudopotential set.
    The suite stays runnable on machines without CP2K.

    Returns ``status='error'`` (+ note) when the CP2K run fails or
    the .out has no parseable energy.

    Returns ``status='pending'`` (the CodeRow default) on success;
    the comparison layer flips it to ``pass`` / ``marginal`` /
    ``fail`` against the reference.
    """
    if conv_tol_energy is None:
        conv_tol_energy = spec.default_conv_tol_energy
    if max_iter is None:
        max_iter = spec.default_max_iter

    row = CodeRow(
        run_id=run_id,
        target=target,
        system_id=spec.id,
        family=spec.family,
        basis=basis_name,
        method_id=method.id,
        kmesh="x".join(str(k) for k in kmesh),
        code="cp2k",
        code_version="unknown",
        n_atoms=len(spec.atoms),
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
        f"  cp2k | {spec.id} | {basis_name} | {method.id} | "
        f"kmesh={kmesh} | target={target}"
    )
    _log("=" * 78)

    # --- Pre-flight gating -----------------------------------------------
    exe = _cp2k_executable()
    if exe is None:
        row.status = "unavailable"
        row.note = (
            "cp2k executable not on PATH "
            "(install CP2K or set VIBEQC_CP2K_EXECUTABLE; tried "
            "cp2k.psmp / cp2k.popt / cp2k.sopt / cp2k)"
        )
        _log(row.note)
        return row

    func_kw = _cp2k_functional(method)
    if func_kw is _UNSUPPORTED:
        row.status = "unavailable"
        row.note = (
            f"cp2k runner: unsupported method "
            f"(scf={method.scf!r} xc={method.xc!r} post={method.post!r}); "
            f"M1f whitelist: rks/rhf with lda/pbe/blyp/b3lyp. "
            f"UHF/UKS + MP2 are M3+ scope."
        )
        _log(row.note)
        return row

    basis_kw = _CP2K_BASIS_MAP.get(basis_name.lower())
    if basis_kw is None:
        row.status = "unavailable"
        row.note = (
            f"cp2k runner: basis {basis_name!r} not in GTH-pseudo set "
            f"(supported: {sorted(_CP2K_BASIS_MAP.keys())}). "
            f"GAPW all-electron basis sets are M3+ scope."
        )
        _log(row.note)
        return row

    pseudo_suffix = _CP2K_PSEUDO_SUFFIX[(method.xc or "lda").lower()]

    basis_file = os.environ.get(
        "VIBEQC_CP2K_BASIS_SET_FILE",
        _cp2k_default_data_file("BASIS_MOLOPT"),
    )
    potential_file = os.environ.get(
        "VIBEQC_CP2K_POTENTIAL_FILE",
        _cp2k_default_data_file("GTH_POTENTIALS"),
    )
    if not Path(basis_file).is_file():
        row.status = "unavailable"
        row.note = (
            f"cp2k basis file not found at {basis_file!r}; "
            f"set VIBEQC_CP2K_BASIS_SET_FILE or CP2K_DATA_DIR"
        )
        _log(row.note)
        return row
    if not Path(potential_file).is_file():
        row.status = "unavailable"
        row.note = (
            f"cp2k potential file not found at {potential_file!r}; "
            f"set VIBEQC_CP2K_POTENTIAL_FILE or CP2K_DATA_DIR"
        )
        _log(row.note)
        return row

    pw_cutoff_ry = float(os.environ.get("VIBEQC_CP2K_PW_CUTOFF_RY", "300"))
    rel_cutoff_ry = float(os.environ.get("VIBEQC_CP2K_REL_CUTOFF_RY", "60"))
    wall_s = int(os.environ.get("VIBEQC_CP2K_WALL_TIME_SECONDS", "1800"))
    cpus = int(os.environ.get("VIBEQC_CP2K_CPUS", "1"))

    # --- Build the input deck -------------------------------------------
    try:
        deck = build_cp2k_input(
            spec=spec,
            basis_kw=basis_kw,
            func_kw=func_kw,
            pseudo_suffix=pseudo_suffix,
            kmesh=kmesh,
            conv_tol_energy=conv_tol_energy,
            max_iter=max_iter,
            pw_cutoff_ry=pw_cutoff_ry,
            rel_cutoff_ry=rel_cutoff_ry,
            basis_file=basis_file,
            potential_file=potential_file,
        )
    except Exception as exc:
        row.status = "error"
        row.note = f"deck build failed: {type(exc).__name__}: {exc}"
        _log(f"  EXCEPTION:\n{traceback.format_exc()}")
        return row

    case_workdir = workdir / f"{spec.id}__{basis_name}__{method.id}"
    case_workdir.mkdir(parents=True, exist_ok=True)
    deck_path = case_workdir / "vibeqc_parity.inp"
    deck_path.write_text(deck)
    out_path = case_workdir / "vibeqc_parity.out"
    _log(f"  wrote {deck_path}")
    _log(f"  cp2k exe: {exe}   cpus: {cpus}   wall: {wall_s}s")

    # --- Run + parse ----------------------------------------------------
    t0 = time.perf_counter()
    try:
        proc = _run_cp2k_subprocess(
            exe=exe,
            deck_path=deck_path,
            out_path=out_path,
            cpus=cpus,
            wall_s=wall_s,
        )
    except subprocess.TimeoutExpired:
        wall = time.perf_counter() - t0
        row.status = "error"
        row.note = f"cp2k subprocess timed out after {wall:.0f}s"
        _log(row.note)
        return row
    except Exception as exc:
        wall = time.perf_counter() - t0
        row.status = "error"
        row.note = f"cp2k subprocess failed: {type(exc).__name__}: {exc}"
        _log(f"  EXCEPTION:\n{traceback.format_exc()}")
        return row
    wall = time.perf_counter() - t0
    _log(f"  cp2k rc={proc.returncode}   wall {wall:.1f}s")

    if not out_path.is_file():
        row.status = "error"
        row.note = f"cp2k output not at {out_path}; rc={proc.returncode}"
        _log(row.note)
        return row

    parsed = parse_cp2k_out(out_path.read_text(errors="replace"))
    row.wall_s = wall
    row.code_version = parsed.version or "unknown"
    row.energy_ha = parsed.energy_ha
    row.converged = parsed.converged
    row.n_iter = parsed.n_iter
    if parsed.energy_ha is not None and len(spec.atoms) > 0:
        row.energy_per_atom_ha = parsed.energy_ha / len(spec.atoms)

    note_bits = []
    if proc.returncode != 0:
        note_bits.append(f"rc={proc.returncode}")
    if parsed.error_line:
        note_bits.append(f"cp2k error: {parsed.error_line}")
    if parsed.energy_ha is None:
        note_bits.append("cp2k produced no parseable total energy")
        row.status = "error"
    if (method.xc or "lda").lower() in ("b3lyp", "b3lyp5"):
        note_bits.append("GTH-BLYP pseudo used for B3LYP (no B3LYP GTH)")
    row.note = "; ".join(note_bits)

    _log(
        f"  E = {row.energy_ha!r} Ha   converged={row.converged}   "
        f"iters={row.n_iter}   version={row.code_version}"
    )
    return row


def run_cp2k_energy(
    spec: PeriodicSpec,
    basis_name: str = "dzvp-molopt-sr-gth",
    functional: str = "pbe",
    kmesh: Tuple[int, int, int] = (1, 1, 1),
    *,
    conv_tol_energy: float = 1e-7,
    max_iter: int = 40,
    pw_cutoff_ry: float = 300.0,
    workdir: Optional[Path] = None,
) -> float:
    """Run CP2K and return the total energy in Hartree (standalone convenience wrapper).

        This is a simpler entry point than :func:`run_periodic_case` for
    aud-hoc parity checks, scripts, or interactive use. It accepts the
    same :class:`PeriodicSpec` and returns just the energy float, raising
    :class:`RuntimeError` (with an informative message) if CP2K is not
    available, the input is invalid, or the calculation fails.

        Parameters
        ----------
        spec : PeriodicSpec
            Periodic system specification with lattice + atoms.
        basis_name : str
            vibe-qc basis name (mapped to the GTH-pseudopotential set, e.g.
            ``"dzvp-molopt-sr-gth"``, ``"tzvp-molopt-gth"``).
        functional : str
            XC functional (``"lda"``, ``"pbe"``, ``"blyp"``, ``"b3lyp"``,
            or ``"hf"`` for pure Hartree-Fock).
        kmesh : tuple of int
            Monkhorst-Pack k-mesh. Default (1, 1, 1) = \u0393-only.
        conv_tol_energy : float
            SCF convergence threshold (Hartree).
        max_iter : int
            Maximum SCF iterations.
        pw_cutoff_ry : float
            Plane-wave cutoff in Rydberg.
        workdir : Path or None
            Working directory for input/output files. If None, a temporary
            directory is created and cleaned up on success; on failure it
            is left for inspection.

        Returns
        -------
        float
            Total energy in Hartree.

        Raises
        ------
        RuntimeError
            If CP2K is not installed, the method/basis is unsupported,
            the calculation fails, or the energy cannot be parsed.
    """
    exe = _cp2k_executable()
    if exe is None:
        raise RuntimeError(
            "CP2K executable not found on PATH. Install CP2K or set "
            "the VIBEQC_CP2K_EXECUTABLE environment variable."
        )

    xc_key = functional.lower()
    if xc_key == "hf":
        func_kw = "HF"
    elif xc_key in _CP2K_FUNC_MAP:
        func_kw = _CP2K_FUNC_MAP[xc_key]
    else:
        raise RuntimeError(
            f"Unsupported functional {functional!r}. Supported: "
            f"lda, pbe, blyp, b3lyp, hf. (The Gaussian-flavor spellings "
            f"b3lyp/g and b3lypg are unsupported: CP2K's B3LYP preset "
            f"is the VWN5 flavor.)"
        )

    basis_kw = _CP2K_BASIS_MAP.get(basis_name.lower())
    if basis_kw is None:
        raise RuntimeError(
            f"Unsupported basis {basis_name!r}. Supported: "
            f"{sorted(_CP2K_BASIS_MAP.keys())}."
        )

    pseudo_suffix = _CP2K_PSEUDO_SUFFIX.get(xc_key, _CP2K_PSEUDO_SUFFIX["lda"])

    basis_file = os.environ.get(
        "VIBEQC_CP2K_BASIS_SET_FILE",
        _cp2k_default_data_file("BASIS_MOLOPT"),
    )
    potential_file = os.environ.get(
        "VIBEQC_CP2K_POTENTIAL_FILE",
        _cp2k_default_data_file("GTH_POTENTIALS"),
    )
    if not Path(basis_file).is_file():
        raise RuntimeError(
            f"CP2K basis file not found at {basis_file!r}. "
            "Set VIBEQC_CP2K_BASIS_SET_FILE or CP2K_DATA_DIR."
        )
    if not Path(potential_file).is_file():
        raise RuntimeError(
            f"CP2K potential file not found at {potential_file!r}. "
            "Set VIBEQC_CP2K_POTENTIAL_FILE or CP2K_DATA_DIR."
        )

    rel_cutoff_ry = float(os.environ.get("VIBEQC_CP2K_REL_CUTOFF_RY", "60"))

    if workdir is None:
        import tempfile

        tmp = tempfile.mkdtemp(prefix="vibeqc_cp2k_")
        workdir = Path(tmp)
        _cleanup = True
    else:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        _cleanup = False

    deck = build_cp2k_input(
        spec=spec,
        basis_kw=basis_kw,
        func_kw=func_kw,
        pseudo_suffix=pseudo_suffix,
        kmesh=kmesh,
        conv_tol_energy=conv_tol_energy,
        max_iter=max_iter,
        pw_cutoff_ry=pw_cutoff_ry,
        rel_cutoff_ry=rel_cutoff_ry,
        basis_file=basis_file,
        potential_file=potential_file,
    )

    deck_path = workdir / "vibeqc_cp2k.inp"
    out_path = workdir / "vibeqc_cp2k.out"
    deck_path.write_text(deck)

    cpus = int(os.environ.get("VIBEQC_CP2K_CPUS", "1"))
    wall_s = int(os.environ.get("VIBEQC_CP2K_WALL_TIME_SECONDS", "1800"))

    try:
        _run_cp2k_subprocess(
            exe=exe,
            deck_path=deck_path,
            out_path=out_path,
            cpus=cpus,
            wall_s=wall_s,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"CP2K subprocess timed out after {wall_s}s. Output left at {out_path}."
        )
    except Exception as exc:
        raise RuntimeError(f"CP2K subprocess failed: {exc}") from exc

    if not out_path.is_file():
        raise RuntimeError(f"CP2K produced no output at {out_path}.")

    parsed = parse_cp2k_out(out_path.read_text(errors="replace"))

    if parsed.energy_ha is None:
        raise RuntimeError(
            f"CP2K calculation did not produce a parseable total energy. "
            f"Check {out_path} for details. "
            f"{f'Error: {parsed.error_line}' if parsed.error_line else ''}"
        )

    if not parsed.converged:
        import warnings

        warnings.warn(f"CP2K SCF did not converge (energy={parsed.energy_ha:.10f} Ha)")

    if _cleanup:
        import shutil

        shutil.rmtree(workdir, ignore_errors=True)

    return parsed.energy_ha


# ----------------------------------------------------------------------
# CP2K executable + data-file discovery
# ----------------------------------------------------------------------


def _cp2k_executable() -> Optional[str]:
    """Return the path to a CP2K binary, or ``None`` if not found.

    Probe order (matches CP2K's own install convention): explicit
    env override first, then the parallel-MPI binary, then the
    OpenMP/serial fallbacks.
    """
    explicit = os.environ.get("VIBEQC_CP2K_EXECUTABLE")
    if explicit:
        if shutil.which(explicit) or Path(explicit).is_file():
            return explicit
        return None
    for name in ("cp2k.psmp", "cp2k.popt", "cp2k.sopt", "cp2k"):
        found = shutil.which(name)
        if found is not None:
            return found
    return None


def _cp2k_default_data_file(name: str) -> str:
    """Locate one of CP2K's shipped data files.

    Order: ``$CP2K_DATA_DIR/<name>``, then common Linux install paths
    (``/usr/share/cp2k/``, ``/opt/cp2k/data/``), then a homebrew path
    on macOS. Returns the first hit or — failing all — the bare
    ``name`` (which will fail the ``is_file()`` check upstream and
    produce a clean ``unavailable`` row, not a cryptic crash).
    """
    data_dir = os.environ.get("CP2K_DATA_DIR")
    if data_dir:
        candidate = Path(data_dir) / name
        if candidate.is_file():
            return str(candidate)
    candidates = [
        Path("/usr/share/cp2k") / name,
        Path("/usr/local/share/cp2k") / name,
        Path("/opt/cp2k/data") / name,
        Path("/opt/homebrew/share/cp2k") / name,  # homebrew arm64
        Path("/usr/local/Cellar/cp2k/current/share/cp2k") / name,  # homebrew x86
    ]
    for c in candidates:
        if c.is_file():
            return str(c)
    return name


# ----------------------------------------------------------------------
# Subprocess driver
# ----------------------------------------------------------------------


def _run_cp2k_subprocess(
    *,
    exe: str,
    deck_path: Path,
    out_path: Path,
    cpus: int,
    wall_s: int,
) -> subprocess.CompletedProcess:
    """Spawn ``cp2k -i <deck> -o <out>`` (or its MPI equivalent) and
    wait for it to finish.

    The ``-i / -o`` CLI form is the portable input/output convention
    supported by every CP2K build since 4.x. We work-dir into the
    deck's parent so any auxiliary files CP2K writes (``*.cp2k`` /
    ``*-pos-1.xyz`` / ``RESTART``) land alongside the deck.

    Output goes to ``out_path``; stderr is appended for diagnostics.
    Caller catches ``subprocess.TimeoutExpired``.
    """
    cwd = deck_path.parent
    if exe.endswith("cp2k.psmp") and cpus > 1:
        cmd = [
            "mpirun",
            "-n",
            str(cpus),
            exe,
            "-i",
            deck_path.name,
            "-o",
            out_path.name,
        ]
    else:
        cmd = [exe, "-i", deck_path.name, "-o", out_path.name]
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=float(wall_s),
    )


# ----------------------------------------------------------------------
# Deck builder
# ----------------------------------------------------------------------


def build_cp2k_input(
    *,
    spec: PeriodicSpec,
    basis_kw: str,
    func_kw: str,
    pseudo_suffix: str,
    kmesh: Tuple[int, int, int],
    conv_tol_energy: float,
    max_iter: int,
    pw_cutoff_ry: float,
    rel_cutoff_ry: float,
    basis_file: str,
    potential_file: str,
) -> str:
    """Compose the GPW CP2K input deck (closed-shell RKS / RHF).

    Lattice goes in as the explicit row-vector ``A / B / C`` block;
    coordinates as scaled (fractional) — same convention the rest of
    the regression suite uses. The cell is always treated as
    ``PERIODIC XYZ``; lower-dimensional cells (slabs / wires) are
    M3+ work.

    The GPW knob is ``METHOD GPW`` inside the ``&QS`` block. To
    switch to GAPW (M3), set ``METHOD GAPW`` and replace the
    GTH pseudo with ``ALL`` / ``ALL_ELECTRON`` per element.
    """
    by_element = _group_atoms_by_element(spec)
    is_hf = func_kw == "HF"
    cell_lines = _cell_block(spec.lattice_ang)
    coord_lines = _coord_block(spec)
    kind_lines = _kind_block(by_element, basis_kw=basis_kw, pseudo_suffix=pseudo_suffix)
    kpoints_block = _kpoints_block(kmesh)

    parts = [
        "&GLOBAL",
        "  PROJECT vibeqc_parity",
        "  RUN_TYPE ENERGY",
        "  PRINT_LEVEL LOW",
        "&END GLOBAL",
        "",
        "&FORCE_EVAL",
        "  METHOD QUICKSTEP",
        "  &DFT",
        f"    BASIS_SET_FILE_NAME {basis_file}",
        f"    POTENTIAL_FILE_NAME {potential_file}",
        "    &MGRID",
        f"      CUTOFF {pw_cutoff_ry:.1f}",
        f"      REL_CUTOFF {rel_cutoff_ry:.1f}",
        "    &END MGRID",
        "    &QS",
        "      METHOD GPW",
        "      EPS_DEFAULT 1.0E-12",
        "    &END QS",
    ]
    if not is_hf:
        parts.extend(
            [
                "    &XC",
                f"      &XC_FUNCTIONAL {func_kw}",
                "      &END XC_FUNCTIONAL",
                "    &END XC",
            ]
        )
    else:
        # Hartree-Fock — request HFX exchange explicitly. The default
        # &XC block is LDA, which is not what we want here.
        parts.extend(
            [
                "    &XC",
                "      &XC_FUNCTIONAL NONE",
                "      &END XC_FUNCTIONAL",
                "      &HF",
                "        FRACTION 1.0",
                "      &END HF",
                "    &END XC",
            ]
        )
    parts.extend(
        [
            "    &SCF",
            f"      EPS_SCF {conv_tol_energy:.2e}",
            f"      MAX_SCF {int(max_iter):d}",
            "      &OUTER_SCF",
            "        MAX_SCF 50",
            f"        EPS_SCF {conv_tol_energy:.2e}",
            "      &END OUTER_SCF",
            "    &END SCF",
        ]
    )
    parts.extend(kpoints_block)
    parts.extend(
        [
            "  &END DFT",
            "  &SUBSYS",
            "    &CELL",
        ]
    )
    parts.extend(cell_lines)
    parts.extend(
        [
            "      PERIODIC XYZ",
            "    &END CELL",
            "    &COORD",
            "      SCALED",
        ]
    )
    parts.extend(coord_lines)
    parts.append("    &END COORD")
    parts.extend(kind_lines)
    parts.extend(
        [
            "  &END SUBSYS",
            "&END FORCE_EVAL",
        ]
    )
    return "\n".join(parts) + "\n"


def _cell_block(lattice_ang: Tuple[Tuple[float, float, float], ...]) -> list:
    a, b, c = (np.asarray(v, dtype=float) for v in lattice_ang)
    return [
        f"      A [angstrom] {a[0]:.10f} {a[1]:.10f} {a[2]:.10f}",
        f"      B [angstrom] {b[0]:.10f} {b[1]:.10f} {b[2]:.10f}",
        f"      C [angstrom] {c[0]:.10f} {c[1]:.10f} {c[2]:.10f}",
    ]


def _coord_block(spec: PeriodicSpec) -> list:
    return [
        f"      {at.symbol} {at.frac[0]:.10f} {at.frac[1]:.10f} {at.frac[2]:.10f}"
        for at in spec.atoms
    ]


def _kind_block(
    by_element: dict,
    *,
    basis_kw: str,
    pseudo_suffix: str,
) -> list:
    out = []
    for symbol in sorted(by_element):
        n_val = _n_valence_electrons(symbol)
        out.extend(
            [
                f"    &KIND {symbol}",
                f"      BASIS_SET {basis_kw}",
                f"      POTENTIAL GTH-{pseudo_suffix}-q{n_val}",
                "    &END KIND",
            ]
        )
    return out


def _kpoints_block(kmesh: Tuple[int, int, int]) -> list:
    if tuple(kmesh) == (1, 1, 1):
        return []  # Γ-only — CP2K defaults to Γ if KPOINTS block absent.
    return [
        "    &KPOINTS",
        f"      SCHEME MONKHORST-PACK {kmesh[0]} {kmesh[1]} {kmesh[2]}",
        "    &END KPOINTS",
    ]


def _group_atoms_by_element(spec: PeriodicSpec) -> dict:
    """Map element symbol → list of (frac coordinates). Used only to
    enumerate the &KIND blocks; CP2K wants exactly one &KIND per
    distinct element in the cell."""
    out: dict = {}
    for at in spec.atoms:
        out.setdefault(at.symbol, []).append(at.frac)
    return out


# Number of valence electrons used by the standard CP2K GTH
# pseudopotentials. Covers the elements the M1f / M2 demo systems
# touch (Si, MgO, Al, Cu, NaCl, …). Adding more is a one-line
# extension; the M3+ chat will sweep the full periodic table.
_GTH_VALENCE = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 3,
    "C": 4,
    "N": 5,
    "O": 6,
    "F": 7,
    "Ne": 8,
    "Na": 9,
    "Mg": 10,
    "Al": 3,
    "Si": 4,
    "P": 5,
    "S": 6,
    "Cl": 7,
    "Ar": 8,
    "K": 9,
    "Ca": 10,
    "Sc": 11,
    "Ti": 12,
    "Cu": 11,
    "Zn": 12,
    "Ga": 13,
    "Ge": 4,
    "As": 5,
    "Se": 6,
    "Br": 7,
    "Ag": 11,
    "Cd": 12,
    "In": 13,
    "Sn": 4,
    "Sb": 5,
    "Te": 6,
    "I": 7,
    "Au": 11,
    "Pt": 18,
    "Pb": 4,
}


def _n_valence_electrons(symbol: str) -> int:
    """Look up the GTH-pseudopotential valence count for an element.

    Raises ``KeyError`` with a clear message when an element falls
    outside the M1f bootstrap set — the caller catches that and
    surfaces a ``status='unavailable'`` row, no silent
    misconfiguration.
    """
    try:
        return _GTH_VALENCE[symbol]
    except KeyError:
        raise KeyError(
            f"cp2k runner: no GTH-pseudopotential valence count "
            f"registered for element {symbol!r}. Add an entry to "
            f"_GTH_VALENCE in runner_cp2k.py (the M3 chat will "
            f"sweep this in)."
        )


# ----------------------------------------------------------------------
# Method whitelist
# ----------------------------------------------------------------------


def _cp2k_functional(method: MethodSpec) -> str:
    """Map MethodSpec → CP2K XC keyword or ``_UNSUPPORTED``.

    M1f scope: rhf + rks with lda / pbe / blyp / b3lyp. UHF / UKS and
    MP2 are intentionally not yet wired — the gapw chat's M3 / M4
    milestones extend this whitelist as their first task.
    """
    if method.post is not None:
        return _UNSUPPORTED  # type: ignore[return-value]
    if method.scf == "rhf":
        return "HF"
    if method.scf == "rks":
        kw = _CP2K_FUNC_MAP.get((method.xc or "").lower())
        return kw if kw is not None else _UNSUPPORTED  # type: ignore[return-value]
    return _UNSUPPORTED  # type: ignore[return-value]


# ----------------------------------------------------------------------
# CP2K .out parser
# ----------------------------------------------------------------------


class Cp2kParseResult:
    """Bag of parsed fields. Mirrors :class:`CrystalParseResult` so
    callers can introspect uniformly. All fields default to ``None``;
    a partial / truncated .out returns the bag with whatever could be
    parsed rather than raising — the caller decides what's fatal."""

    __slots__ = ("version", "energy_ha", "converged", "n_iter", "error_line")

    def __init__(self) -> None:
        self.version: Optional[str] = None
        self.energy_ha: Optional[float] = None
        self.converged: Optional[bool] = None
        self.n_iter: Optional[int] = None
        self.error_line: str = ""


# CP2K version banner. Lines like:
#   CP2K version 2024.1 (Development Version)
#   CP2K| version string: CP2K version 9.1
_VERSION_RE = re.compile(
    r"CP2K\s+version\s+(\S+)",
    re.IGNORECASE,
)
# Final total energy. CP2K prints near the end of the .out:
#   ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:        -7.6037215834312
# The "[a.u.]" tag pins atomic units. There can be multiple matches
# in a single run (one per SCF restart); the last one wins.
_TOTAL_E_RE = re.compile(
    r"ENERGY\|\s*Total\s+FORCE_EVAL\s*\(\s*QS\s*\)\s*energy\s*\[a\.u\.\]:\s*(-?\d+\.\d+)",
    re.IGNORECASE,
)
# SCF convergence verdict. CP2K prints:
#   *** SCF run converged in     12 steps ***
#   *** SCF run NOT converged ***
_SCF_OK_RE = re.compile(
    r"\*\*\*\s*SCF\s+run\s+converged\s+in\s+(\d+)\s+steps",
    re.IGNORECASE,
)
_SCF_FAIL_RE = re.compile(
    r"\*\*\*\s*SCF\s+run\s+NOT\s+converged",
    re.IGNORECASE,
)
# Catastrophic error markers. CP2K's reporting is verbose but the
# leading-asterisk banner lines mark abort / fatal conditions:
#   *** Fatal error in qs_environment ...
#   *** ERROR in atom_basis: ...
_ERROR_RE = re.compile(
    r"^\s*\*+\s*(ERROR|Fatal error|ABORT|Abnormal Termination)",
    re.IGNORECASE,
)


def parse_cp2k_out(text: str) -> Cp2kParseResult:
    """Parse a CP2K .out into the fields the regression row needs.

    All fields are best-effort. A truncated .out (SCF crash, OOM,
    timeout mid-iteration) returns ``energy_ha=None`` rather than
    raising. The caller decides whether to flag the row.

    Last-occurrence-wins on the total-energy line — same convention
    as the CRYSTAL parser, for the same reason (CP2K prints one
    energy per converged SCF; the bottom-of-file value is the
    canonical converged result).
    """
    res = Cp2kParseResult()
    last_e: Optional[float] = None

    for line in text.splitlines():
        if res.version is None:
            m = _VERSION_RE.search(line)
            if m:
                res.version = m.group(1).strip().rstrip(",")
        m = _TOTAL_E_RE.search(line)
        if m:
            try:
                last_e = float(m.group(1))
            except ValueError:
                pass
            continue
        m_ok = _SCF_OK_RE.search(line)
        if m_ok:
            res.converged = True
            try:
                res.n_iter = int(m_ok.group(1))
            except ValueError:
                pass
            continue
        if _SCF_FAIL_RE.search(line):
            res.converged = False
            continue
        if not res.error_line and _ERROR_RE.match(line):
            res.error_line = line.strip()[:200]

    res.energy_ha = last_e
    return res


# ----------------------------------------------------------------------
# Discovery / smoke API for tests
# ----------------------------------------------------------------------


def is_available() -> bool:
    """Return True iff CP2K is invocable on this machine.

    Cheaper than running a full case: just probes the PATH for one
    of the CP2K executable names. Tests use this to skip CP2K
    parity assertions on machines without it.
    """
    return _cp2k_executable() is not None
