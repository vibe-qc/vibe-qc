"""Cross-calculator benchmarking and validation against external QC codes.

This module wraps multiple ASE-compatible calculators (vibe-qc itself
plus optional external codes: ORCA, NWChem, Gaussian, Q-Chem, GAMESS-US,
Turbomole) and runs the same calculation on each, returning a tidy
comparison table. The two main entry points are:

* :func:`compare_calculators` runs a list of (label, calculator) pairs
  on a single ``ase.Atoms``, gathers the requested properties, and
  returns a :class:`BenchmarkResults` object.
* :func:`detect_calculators` returns which external QC codes are
  importable + have their binaries on ``$PATH``. Useful for
  "skip if missing" branches in cross-validation scripts.

PySCF, Psi4, and CRYSTAL are treated as external references. This
module does not import any of them in-process (CLAUDE.md Sec. 10); parity
harnesses spawn the reference program as a subprocess and parse its
output. See the runners under ``examples/regression/core/`` for the
canonical pattern.

Design choices
==============

* **Calculators are passed in as configured objects, not as factories.**
  The caller is responsible for setting the basis / functional / charge
  / multiplicity to match. The framework just runs them and reports.

* **Missing calculators degrade gracefully.** If the user passes
  ``("ORCA/HF/sto-3g", OrcaCalculator(...))`` and the orca binary
  isn't on PATH, the row appears in the result table with status
  "unavailable" and the script can continue. This makes
  cross-validation scripts safe to ship as repository examples --
  they run on developer laptops without ORCA installed and just
  skip those rows.

* **Result units are ASE-native** (eV, eV/Å, eÅ, Å^3). Every backend's
  outputs are funnelled through ASE's ``get_potential_energy()`` and
  friends, so the units are consistent regardless of what each code
  uses internally.

* **Tolerances on assertions are explicit.** :meth:`BenchmarkResults.
  assert_agreement` takes per-property tolerances; defaults are loose
  enough that legitimate method differences (DFT grid, MP2 vs HF) pass,
  tight enough that numerical bugs fail loudly.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from ase import Atoms
    from ase.calculators.calculator import Calculator

_log = logging.getLogger("vibeqc.benchmark")

# Default per-property tolerances. Tight enough to catch a real bug,
# loose enough that legitimate method/grid differences pass. Override
# per-call via :meth:`BenchmarkResults.assert_agreement(tol_*=...)`.
DEFAULT_TOL = {
    "energy": 1e-4,  # eV  -- chemical accuracy is ~0.04 eV (1 kcal/mol),
    #       so 1e-4 catches real bugs without flagging
    #       basis/method differences as errors
    "forces": 1e-3,  # eV/Å -- same logic
    "dipole": 5e-3,  # eÅ
    "polarizability": 5e-2,  # Å^3
}


# ---------------------------------------------------------------------------
# Calculator availability detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _CalcAvailability:
    name: str
    importable: bool
    binary_path: Path | None
    notes: str = ""

    @property
    def available(self) -> bool:
        return self.importable and (
            self.binary_path is not None or self._is_python_native
        )

    @property
    def _is_python_native(self) -> bool:
        # VibeQC and Psi4 (despite a binary
        # existing) are accessed via Python bindings -- ASE's wrappers
        # `import` them rather than spawning subprocesses. So
        # availability is gated on the Python module being
        # importable in the current interpreter, not on a binary
        # being on $PATH.
        return self.name.lower() in {"vibeqc", "psi4"}


def detect_calculators() -> dict[str, _CalcAvailability]:
    """Probe the runtime for available QC calculators.

    Returns a dict keyed by lowercased calculator name. Each value
    reports whether the Python-side wrapper is importable, and whether
    the external binary (where applicable) is found on ``$PATH``.
    Used by example scripts to skip cross-validation rows when a
    code isn't installed.
    """
    found: dict[str, _CalcAvailability] = {}

    # vibe-qc -- always present in this codebase.
    try:
        import vibeqc  # noqa: F401

        found["vibeqc"] = _CalcAvailability(
            "vibeqc",
            importable=True,
            binary_path=None,
            notes="Python-native; no external binary",
        )
    except ImportError:
        # Shouldn't happen, but be safe.
        found["vibeqc"] = _CalcAvailability(
            "vibeqc",
            importable=False,
            binary_path=None,
        )

    # PySCF is intentionally not imported here. Use an explicit
    # out-of-process reference runner if parity data are needed.
    pyscf_bin = shutil.which("pyscf")
    found["pyscf"] = _CalcAvailability(
        "pyscf",
        importable=bool(pyscf_bin),
        binary_path=Path(pyscf_bin) if pyscf_bin else None,
        notes=(
            "External reference only; vibe-qc does not import PySCF. "
            "Use a subprocess runner/script and parse its output."
        ),
    )

    # Psi4 is intentionally not imported here. Like PySCF, it is
    # treated as an external reference and driven out-of-process via
    # the regression-suite subprocess runner. CLAUDE.md Sec. 10.
    psi4_bin = shutil.which("psi4")
    found["psi4"] = _CalcAvailability(
        "psi4",
        importable=bool(psi4_bin),
        binary_path=Path(psi4_bin) if psi4_bin else None,
        notes=(
            "External reference only; vibe-qc does not import Psi4. "
            "Use a subprocess runner/script and parse its output."
        ),
    )

    # Subprocess-based external codes (FileIOCalculator pattern).
    # ASE wrapper imports cheaply; the actual code runs as a child
    # process so we need both the Python wrapper AND the binary.
    external_codes: list[tuple[str, str, str]] = [
        # (name,           ase_module_path,                 binary_basename)
        ("orca", "ase.calculators.orca", "orca"),
        ("nwchem", "ase.calculators.nwchem", "nwchem"),
        ("gaussian", "ase.calculators.gaussian", "g16"),
        ("qchem", "ase.calculators.qchem", "qchem"),
        ("gamess_us", "ase.calculators.gamess_us", "rungms"),
        ("turbomole", "ase.calculators.turbomole", "dscf"),
    ]

    for name, module_path, binary in external_codes:
        importable = True
        try:
            __import__(module_path)
        except ImportError:
            importable = False

        path = shutil.which(binary)
        binary_path = Path(path) if path is not None else None

        notes = ""
        if not importable:
            notes = f"ASE wrapper {module_path!r} not importable"
        elif binary_path is None:
            notes = (
                f"Python wrapper present, but {binary!r} not on $PATH "
                f"-- install the {name.upper()} binary"
            )

        found[name] = _CalcAvailability(
            name=name,
            importable=importable,
            binary_path=binary_path,
            notes=notes,
        )

    return found


def find_orca_command() -> str | None:
    """Return the path to the ORCA binary, or None if not found.

    Resolution order:

    1. ``ASE_ORCA_COMMAND`` environment variable (set by user).
       ASE's ORCA wrapper accepts a full command template like
       ``orca PREFIX.inp > PREFIX.out``; we extract the executable.
    2. ``ORCA_COMMAND`` environment variable (vibe-qc-specific).
    3. ``ORCA_PATH`` environment variable, expected to point at the
       ORCA install directory; we look for ``$ORCA_PATH/orca``.
    4. ``shutil.which("orca")`` -- lookup on ``$PATH``.

    Returns the resolved binary path as a string, or None if no
    candidate works.
    """
    import os

    # ASE_ORCA_COMMAND can be a full command template; take the first token.
    cmd = os.environ.get("ASE_ORCA_COMMAND")
    if cmd:
        first = cmd.split()[0]
        if Path(first).is_file():
            return first
    cmd = os.environ.get("ORCA_COMMAND")
    if cmd and Path(cmd).is_file():
        return cmd
    orca_dir = os.environ.get("ORCA_PATH")
    if orca_dir:
        candidate = Path(orca_dir) / "orca"
        if candidate.is_file():
            return str(candidate)
    return shutil.which("orca")


def make_orca_calculator(
    *,
    orcasimpleinput: str,
    orcablocks: str = "",
    orca_command: str | None = None,
    label: str = "orca",
    **kwargs: Any,
) -> "Calculator | None":
    """Build a ready-to-run ASE ORCA calculator, or return None.

    Returns ``None`` if ORCA can't be located (so calling scripts can
    branch with ``if orca is not None:`` rather than try/except).

    Parameters
    ----------
    orcasimpleinput
        ORCA "simple input" line -- method, basis, options. E.g.
        ``"HF STO-3G EnGrad"`` or ``"PBE D3BJ def2-SVP EnGrad"``.
        Must end with ``EnGrad`` if forces are wanted (otherwise
        ASE's wrapper falls back to FD on energies).
    orcablocks
        Optional ``%basis``, ``%scf``, etc. block input. Joined into
        the ORCA input file as-is.
    orca_command
        Path to the orca binary. If None, resolved via
        :func:`find_orca_command`.
    label
        ASE label, used as the working-directory / filename stem.

    Examples
    --------
    >>> from vibeqc.benchmark import make_orca_calculator
    >>> orca = make_orca_calculator(
    ...     orcasimpleinput="HF STO-3G EnGrad",
    ... )
    >>> if orca is None:
    ...     print("ORCA not found, skipping")
    """
    cmd = orca_command or find_orca_command()
    if cmd is None:
        return None

    try:
        from ase.calculators.orca import ORCA, OrcaProfile
    except ImportError:
        return None

    return ORCA(
        profile=OrcaProfile(command=cmd),
        orcasimpleinput=orcasimpleinput,
        orcablocks=orcablocks,
        label=label,
        **kwargs,
    )


def print_calculator_availability() -> None:
    """Print a tabulated report of calculator availability.

    Convenience helper for diagnostic scripts. Output looks like::

        vibeqc     ✓ available  (Python-native)
        pyscf      ✓ available  (Python-native)
        orca       ✗ missing    (orca not on $PATH)
        psi4       ✓ available  (/usr/local/bin/psi4)
        ...
    """
    avail = detect_calculators()
    name_width = max(len(n) for n in avail) + 2
    for name, info in sorted(avail.items()):
        marker = "✓ available" if info.available else "✗ missing  "
        if info.binary_path is not None:
            details = str(info.binary_path)
        elif info._is_python_native and info.importable:
            details = "Python-native"
        else:
            details = info.notes
        print(f"  {name:<{name_width}}{marker}  ({details})")


# ---------------------------------------------------------------------------
# BenchmarkResults
# ---------------------------------------------------------------------------


@dataclass
class _CalcRow:
    """One row in the benchmark table -- one calculator's outputs.

    All values are in ASE-native units (eV, eV/Å, eÅ, Å^3). Status:

    * ``ok`` -- every requested property computed cleanly.
    * ``partial`` -- at least one property succeeded and at least one
      raised; ``error`` lists which properties failed and why.
    * ``failed`` -- every property raised during ``calculate()``.
    * ``unavailable`` -- calculator binary or Python wrapper missing;
      ``skip_unavailable=True`` reported it and moved on.
    * ``timeout`` -- only with ``timeout_s`` set; the child process
      was alive past the deadline and got terminated. No properties
      are recorded.
    * ``crashed`` -- only with ``timeout_s`` set; the child process
      died (nonzero exit code or no result on the queue) without a
      Python exception we could capture. No properties are recorded.
    """

    label: str
    status: str
    properties: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    wall_time_s: float | None = None


@dataclass
class BenchmarkResults:
    """Container for the per-calculator output of
    :func:`compare_calculators`. The class is intentionally simple --
    it just stores rows; methods provide pretty-printing, CSV export,
    and assertion helpers used by tests / example scripts.
    """

    atoms: "Atoms"
    properties_requested: tuple[str, ...]
    rows: list[_CalcRow] = field(default_factory=list)

    # ---- output / introspection -------------------------------------

    def print_table(self) -> None:
        """Pretty-print the comparison table to stdout."""
        if not self.rows:
            print("(no calculators)")
            return

        # Collect column set: status + each requested property.
        cols: list[str] = ["status", "wall_time_s"]
        for p in self.properties_requested:
            if p == "forces":
                cols.append("|F|max")
            elif p == "polarizability":
                cols.append("a_iso")
            else:
                cols.append(p)

        # Format helpers -- trade compactness for readability.
        def _fmt(value: Any, name: str) -> str:
            if value is None:
                return "--"
            if name == "status":
                return str(value)
            if name == "wall_time_s":
                return f"{value:5.2f}s"
            if name == "|F|max":
                return f"{float(value):.4e}"
            if name == "a_iso":
                return f"{float(value):.4f}"
            if name == "energy":
                return f"{float(value):.6f} eV"
            if name == "dipole":
                return f"{float(value):.4f} eÅ"
            return str(value)

        # Build per-row dict for tabulating.
        body: list[dict[str, str]] = []
        for r in self.rows:
            d: dict[str, str] = {"label": r.label}
            d["status"] = r.status
            d["wall_time_s"] = (
                _fmt(r.wall_time_s, "wall_time_s") if r.wall_time_s is not None else "--"
            )
            for p in self.properties_requested:
                v = r.properties.get(p)
                if p == "forces" and v is not None:
                    v = float(np.abs(v).max())
                    name = "|F|max"
                elif p == "polarizability" and v is not None:
                    v = float(np.asarray(v).trace() / 3.0)
                    name = "a_iso"
                elif p == "dipole" and v is not None:
                    v = float(np.linalg.norm(v))
                    name = "dipole"
                else:
                    name = p
                d[name] = _fmt(v, name)
            body.append(d)

        # Compute column widths.
        all_cols = ["label"] + cols
        widths = {c: len(c) for c in all_cols}
        for d in body:
            for c in all_cols:
                widths[c] = max(widths[c], len(d.get(c, "")))

        # Print header.
        sep = " │ "
        header = sep.join(c.ljust(widths[c]) for c in all_cols)
        print(header)
        print("─" * len(header))
        for d in body:
            print(sep.join(d.get(c, "").ljust(widths[c]) for c in all_cols))
            if d["status"] in ("failed", "partial", "timeout", "crashed"):
                err_row = next(
                    (r for r in self.rows if r.label == d["label"]),
                    None,
                )
                if err_row and err_row.error:
                    print(f"  └─ error: {err_row.error}")

    def to_csv(self, path: str | Path) -> Path:
        """Write the comparison table to a CSV file. Returns the path."""
        import csv

        out_path = Path(path)
        with out_path.open("w", newline="") as f:
            w = csv.writer(f)
            cols = ["label", "status", "wall_time_s", *self.properties_requested]
            w.writerow(cols)
            for r in self.rows:
                row: list[str] = [r.label, r.status, ""]
                row[2] = "" if r.wall_time_s is None else f"{r.wall_time_s:.4f}"
                for p in self.properties_requested:
                    v = r.properties.get(p)
                    if v is None:
                        row.append("")
                    elif isinstance(v, np.ndarray):
                        row.append(np.array2string(v, separator=";"))
                    else:
                        row.append(repr(v))
                w.writerow(row)
        return out_path

    # ---- assertions for tests ---------------------------------------

    def assert_agreement(
        self,
        reference: str | None = None,
        *,
        tol: Mapping[str, float] | None = None,
    ) -> None:
        """Assert all successful rows agree with *reference* within
        per-property tolerances.

        ``reference`` is the label of the row to compare against. If
        omitted, the first row with ``status='ok'`` wins. Properties
        are compared in ASE units. ``forces`` uses element-wise max
        absolute difference; ``polarizability`` uses Frobenius norm
        of the difference; scalars just use absolute difference.

        Raises :class:`AssertionError` on first violation, naming the
        property + the offending calculator + the gap.
        """
        # Both fully-ok rows and partial rows are compared (partial
        # rows might have the property under test even if they failed
        # on a different property -- e.g. PySCF energy succeeded but
        # forces failed).
        ok_rows = [r for r in self.rows if r.status in ("ok", "partial")]
        if not ok_rows:
            raise AssertionError("benchmark: no successful rows to compare")

        if reference is None:
            ref_row = ok_rows[0]
        else:
            try:
                ref_row = next(r for r in ok_rows if r.label == reference)
            except StopIteration:
                raise AssertionError(
                    f"benchmark: reference {reference!r} not found among "
                    f"successful rows: {[r.label for r in ok_rows]}"
                )

        tols = {**DEFAULT_TOL, **(tol or {})}
        for r in ok_rows:
            if r is ref_row:
                continue
            for prop in self.properties_requested:
                ref_v = ref_row.properties.get(prop)
                cur_v = r.properties.get(prop)
                if ref_v is None or cur_v is None:
                    continue
                tol_v = tols.get(prop, 0.0)
                if prop == "forces":
                    delta = float(np.abs(np.asarray(cur_v) - np.asarray(ref_v)).max())
                elif prop == "polarizability":
                    delta = float(np.linalg.norm(np.asarray(cur_v) - np.asarray(ref_v)))
                elif prop == "dipole":
                    delta = float(np.linalg.norm(np.asarray(cur_v) - np.asarray(ref_v)))
                else:
                    delta = float(abs(cur_v - ref_v))
                if delta > tol_v:
                    raise AssertionError(
                        f"{r.label} disagrees with {ref_row.label} on "
                        f"{prop}: |Δ| = {delta:.6e} > tol {tol_v:.2e} "
                        f"({ref_row.label}={ref_v!r}, {r.label}={cur_v!r})"
                    )

    # ---- iteration ---------------------------------------------------

    def __iter__(self) -> Iterable[_CalcRow]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)


# ---------------------------------------------------------------------------
# compare_calculators
# ---------------------------------------------------------------------------


def _is_calculator_instance(obj: Any) -> bool:
    """Heuristic: looks like an ASE Calculator instance (not a factory)."""
    return hasattr(obj, "get_potential_energy") or hasattr(obj, "calculate")


def _compute_properties(
    calc: "Calculator",
    atoms: "Atoms",
    properties: Sequence[str],
    *,
    label_for_log: str | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Compute *properties* on *atoms* via *calc*, capturing per-property errors.

    Returns ``(properties_dict, errors_dict)`` so a failure on one
    property doesn't lose successfully-computed others. The caller maps
    these to row status ('ok' / 'partial' / 'failed').

    The caller is expected to have already attached *calc* to *atoms*
    (``atoms.calc = calc``) and to have handed in a private copy of
    *atoms* -- this helper does not copy.
    """
    props: dict[str, Any] = {}
    errors: dict[str, str] = {}
    for prop in properties:
        try:
            if prop == "energy":
                props["energy"] = atoms.get_potential_energy()
            elif prop == "forces":
                props["forces"] = atoms.get_forces()
            elif prop == "dipole":
                props["dipole"] = atoms.get_dipole_moment()
            elif prop == "polarizability":
                props["polarizability"] = calc.get_property("polarizability", atoms)
            elif prop == "hessian":
                props["hessian"] = calc.get_property("hessian", atoms)
            else:
                props[prop] = calc.get_property(prop, atoms)
        except Exception as e:  # pylint: disable=broad-except
            errors[prop] = f"{type(e).__name__}: {e}"
            if label_for_log is not None:
                _log.warning(
                    "calc %s failed on property %s: %s",
                    label_for_log,
                    prop,
                    errors[prop],
                )
    return props, errors


def _finalize_row(
    label: str,
    properties: dict[str, Any],
    errors: dict[str, str],
    wall_time_s: float,
) -> _CalcRow:
    """Build a _CalcRow with status derived from properties/errors split."""
    row = _CalcRow(
        label=label,
        status="ok",
        properties=properties,
        wall_time_s=wall_time_s,
    )
    if errors and not row.properties:
        row.status = "failed"
    elif errors:
        row.status = "partial"
    if errors:
        # Compose a single human-readable error: which property(ies)
        # failed and why. Used by print_table() to render the
        # ``└─ error: ...`` line under a failed/partial row.
        row.error = "; ".join(f"{p}: {msg}" for p, msg in errors.items())
    return row


# ---------------------------------------------------------------------------
# Subprocess timeout machinery
# ---------------------------------------------------------------------------
#
# In-process serial execution (timeout_s=None) cannot enforce a
# wall-time cap on a hung calculator -- Python threads cannot be killed
# from outside, and an external code blocked in C/Fortran won't respond
# to signals. When the caller asks for a real timeout we run each
# calculator in a child process via multiprocessing's 'spawn' context
# (clean import state, no inherited C++ globals) and SIGTERM/SIGKILL it
# past the deadline. Both the calculator (or a zero-arg factory that
# constructs one) and the Atoms object must be picklable to reach the
# child; we pickle-check eagerly so users get a clear, actionable error
# instead of an opaque pickle traceback from the multiprocessing layer.


def _subprocess_worker(  # pragma: no cover -- runs in a child process
    calc_or_factory: Any,
    atoms: "Atoms",
    properties: tuple[str, ...],
    queue: Any,
) -> None:
    """Child-process entry point. Compute properties; put result on queue.

    Always puts a single dict on the queue (even on setup failure) so
    the parent can distinguish "child finished but failed" from
    "child crashed without sending anything".
    """
    payload: dict[str, Any] = {"properties": {}, "errors": {}}
    try:
        if _is_calculator_instance(calc_or_factory):
            calc = calc_or_factory
        elif callable(calc_or_factory):
            calc = calc_or_factory()
        else:
            raise TypeError(
                f"expected an ASE Calculator instance or a zero-arg "
                f"factory callable, got {type(calc_or_factory).__name__}"
            )
        atoms.calc = calc
        props, errors = _compute_properties(calc, atoms, properties)
        payload["properties"] = props
        payload["errors"] = errors
    except Exception as e:  # pylint: disable=broad-except
        # Setup-time failure (factory raised, bad input, ...) -> every
        # property is reported failed with the same root cause.
        msg = f"{type(e).__name__}: {e}"
        payload["errors"] = {p: msg for p in properties}
    try:
        queue.put(payload)
    except Exception:  # pylint: disable=broad-except
        # If pickling the payload back fails, leave the queue empty so
        # the parent reports a "no result" condition rather than hanging.
        pass


def _run_with_timeout(
    label: str,
    calc_or_factory: Any,
    atoms: "Atoms",
    properties: Sequence[str],
    timeout_s: float,
) -> _CalcRow:
    """Run one calculator out-of-process with a hard wall-time cap."""
    import multiprocessing as mp
    import pickle
    import time

    properties = tuple(properties)
    ctx = mp.get_context("spawn")

    # Eager pickle check -- give a useful error rather than the
    # multiprocessing pickler's traceback. Most failures here are
    # closures / lambdas / unpicklable C++-bound state on configured
    # calculator instances; the fix is a module-level factory.
    try:
        pickle.dumps((calc_or_factory, atoms, properties))
    except (pickle.PicklingError, TypeError, AttributeError) as e:
        raise ValueError(
            f"compare_calculators(timeout_s=...): cannot send calculator "
            f"{label!r} to a child process -- {type(e).__name__}: {e}. "
            f"Pass a module-level zero-arg factory function "
            f"(``(label, my_factory)``) instead of a pre-built "
            f"calculator instance. Lambdas and closures are not "
            f"picklable; use a top-level ``def`` in your script or "
            f"a helper module. See "
            f"docs/user_guide/external_codes.md Sec. \"Per-calculator "
            f"timeouts\"."
        ) from e

    queue: Any = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_subprocess_worker,
        args=(calc_or_factory, atoms, properties, queue),
        daemon=False,
    )
    t0 = time.perf_counter()
    proc.start()
    proc.join(timeout=timeout_s)
    wall = time.perf_counter() - t0

    if proc.is_alive():
        # Past the deadline: SIGTERM, then SIGKILL if it ignored us.
        proc.terminate()
        proc.join(timeout=2.0)
        if proc.is_alive():
            proc.kill()
            proc.join(timeout=2.0)
        # Drain queue if the child managed to push between deadline
        # and terminate -- we discard partial results rather than
        # half-reporting; status is unambiguously "timeout".
        try:
            while not queue.empty():
                queue.get_nowait()
        except Exception:  # pylint: disable=broad-except
            pass
        _log.warning(
            "calc %s timed out after %.3fs (limit %.3fs)",
            label,
            wall,
            timeout_s,
        )
        return _CalcRow(
            label=label,
            status="timeout",
            error=(
                f"calculator did not finish within timeout_s={timeout_s:.3f}s "
                f"(child process terminated)"
            ),
            wall_time_s=wall,
        )

    # Child finished within the deadline. Pull its result (with a
    # short get-timeout in case the queue's pipe is still flushing).
    payload: dict[str, Any] | None = None
    try:
        payload = queue.get(timeout=5.0)
    except Exception:  # pylint: disable=broad-except
        payload = None

    if payload is None:
        # No result on the queue -> treat as a crash; report the
        # exitcode so the user can correlate with stderr.
        exitcode = proc.exitcode
        _log.warning(
            "calc %s child finished with no result (exitcode=%s)",
            label,
            exitcode,
        )
        return _CalcRow(
            label=label,
            status="crashed",
            error=(
                f"child process produced no result "
                f"(exitcode={exitcode!r})"
            ),
            wall_time_s=wall,
        )

    return _finalize_row(
        label=label,
        properties=payload.get("properties", {}),
        errors=payload.get("errors", {}),
        wall_time_s=wall,
    )


# ---------------------------------------------------------------------------
# compare_calculators driver
# ---------------------------------------------------------------------------


def compare_calculators(
    atoms: "Atoms",
    calculators: Sequence[tuple[str, "Calculator | Any"]],
    properties: Sequence[str] = ("energy", "forces"),
    skip_unavailable: bool = True,
    timeout_s: float | None = None,
) -> BenchmarkResults:
    """Run *atoms* through each calculator, return a
    :class:`BenchmarkResults` for printing / assertions.

    Parameters
    ----------
    atoms
        The system to compute on. **Each calculator gets a fresh
        copy** -- running the same Atoms through multiple calculators
        directly clobbers ``atoms.calc`` between calls, leading to
        confusing "wrong result" debugging.
    calculators
        Sequence of ``(label, calc_or_factory)`` tuples. The second
        element is normally a configured ASE Calculator instance.
        When ``timeout_s`` is set, you may instead pass a **zero-arg
        factory callable** that constructs the calculator inside the
        child process -- this avoids pickling unpicklable bindings
        (e.g. Psi4) across the process boundary. Labels are free-form;
        convention is ``"<code>/<method>/<basis>"`` so the comparison
        table reads cleanly (e.g. ``"vibe-qc/RHF/6-31G*"``).
    properties
        Which ASE properties to compute on each. Common choices:
        ``("energy",)`` for fastest sanity check;
        ``("energy", "forces")`` for opt-relevant validation;
        ``("energy", "forces", "dipole")`` for property comparison;
        plus ``"polarizability"`` / ``"hessian"`` where supported.
    skip_unavailable
        If True (default), calculators whose Python wrapper or
        external binary are missing are reported as "unavailable"
        and the comparison continues. If False, the missing
        calculator raises during the run.
    timeout_s
        Per-calculator wall-time cap in seconds. If ``None`` (default),
        calculators run **in-process and serially**; a hung calculator
        will hang the whole benchmark. If a positive number is given,
        each calculator runs in its own ``multiprocessing.spawn``
        child process and is SIGTERM/SIGKILLed past the deadline,
        producing a row with ``status='timeout'``. The calculator (or
        its factory) and the Atoms object must be picklable; an
        unpicklable instance raises :class:`ValueError` with migration
        guidance pointing at the factory pattern. Child processes
        that die without returning a result get ``status='crashed'``.
        Threads are intentionally **not** used as a fake-timeout
        fallback -- a worker blocked in C/Fortran cannot be cancelled
        from another thread, and reporting "timeout" while the worker
        still consumes CPU would be a lie.

    Returns
    -------
    :class:`BenchmarkResults`

    Examples
    --------
    >>> from ase.build import molecule
    >>> from vibeqc.ase import VibeQC
    >>> from vibeqc.benchmark import compare_calculators
    >>> atoms = molecule("H2O")
    >>> results = compare_calculators(
    ...     atoms,
    ...     [
    ...         ("vibe-qc/RHF/6-31G*", VibeQC(basis="6-31g*")),
    ...     ],
    ...     properties=("energy", "forces", "dipole"),
    ... )
    >>> results.print_table()  # doctest: +SKIP
    >>> results.assert_agreement(tol={"energy": 1e-6})

    With a per-calculator timeout (factory pattern for the
    process-safe path):

    >>> def make_vibeqc_rhf():
    ...     from vibeqc.ase import VibeQC
    ...     return VibeQC(basis="6-31g*")
    >>> results = compare_calculators(  # doctest: +SKIP
    ...     atoms,
    ...     [("vibe-qc/RHF/6-31G*", make_vibeqc_rhf)],
    ...     timeout_s=300.0,
    ... )
    """
    import time

    avail = detect_calculators()
    rows: list[_CalcRow] = []
    use_subprocess = timeout_s is not None and timeout_s > 0.0
    if timeout_s is not None and timeout_s <= 0.0:
        raise ValueError(
            f"timeout_s must be a positive number of seconds or None; "
            f"got {timeout_s!r}"
        )

    for label, calc in calculators:
        # Sniff the calculator name from its class for availability check.
        # Factories (plain functions) have no useful class name; skip
        # availability gating for them and let the child surface any
        # missing-dep error inside its own try/except.
        if _is_calculator_instance(calc):
            cls_name = type(calc).__name__.lower()
            avail_key = next(
                (k for k in avail if k in cls_name),
                None,
            )
            if (
                skip_unavailable
                and avail_key is not None
                and not avail[avail_key].available
            ):
                rows.append(
                    _CalcRow(
                        label=label,
                        status="unavailable",
                        error=avail[avail_key].notes,
                    )
                )
                _log.info("skip %s -- %s", label, avail[avail_key].notes)
                continue

        if use_subprocess:
            rows.append(
                _run_with_timeout(
                    label=label,
                    calc_or_factory=calc,
                    atoms=atoms,
                    properties=properties,
                    timeout_s=float(timeout_s),  # type: ignore[arg-type]
                )
            )
            continue

        # In-process path (timeout_s=None): historical behaviour.
        # Always work on a copy -- multiple calcs sharing one Atoms is
        # a footgun (the second .calc assignment invalidates the first
        # calc's cached results in subtle ways).
        if not _is_calculator_instance(calc):
            # Without a timeout we still let callers pass a factory,
            # for symmetry with the timeout path. Construct it once
            # in-process here.
            calc = calc()
        atoms_copy = atoms.copy()
        atoms_copy.calc = calc
        t0 = time.perf_counter()
        props, errors = _compute_properties(
            calc, atoms_copy, properties, label_for_log=label
        )
        wall = time.perf_counter() - t0
        rows.append(_finalize_row(label, props, errors, wall))

    return BenchmarkResults(
        atoms=atoms,
        properties_requested=tuple(properties),
        rows=rows,
    )


__all__ = [
    "compare_calculators",
    "BenchmarkResults",
    "detect_calculators",
    "print_calculator_availability",
    "find_orca_command",
    "make_orca_calculator",
    "DEFAULT_TOL",
]
