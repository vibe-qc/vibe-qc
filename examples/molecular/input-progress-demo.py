"""ProgressLogger showcase — every v0.5.1 logging surface in one file.

A short, **fast** demo of vibe-qc's live-logging API. Designed to
finish in seconds while exercising every public method on
:class:`vibeqc.ProgressLogger` plus the ``progress=`` kwarg on
``run_job`` and the ``VIBEQC_LIVE_LOGGING=0`` env-var opt-out.

The four sections demonstrate, in order:

    1. **Default-on logging via ``run_job(progress=True)``** — the
       canonical molecular workflow. Live banner, native SCF rows,
       and summary on stdout; ``.out`` / ``.molden`` / ``.system``
       siblings written to disk.

    2. **Direct ``ProgressLogger`` API** — every public method
       called by hand (``banner``, ``info``, ``warn``, ``stage``,
       ``iteration``, ``write_raw``, ``converged``). No SCF; just
       shows what each call produces. Useful as a copy-paste
       reference if you're wiring ``ProgressLogger`` into your own
       script.

    3. **Custom logger that tees to a persistent file** — a
       ``ProgressLogger(log_path=...)`` that mirrors stdout to a
       file users can ``tail -f`` from a second terminal. Threaded
       through ``progress=plog`` on a real molecular SCF so the
       per-stage banners and final summary all land in both places.

    4. **The off-switch** — ``progress=False`` and
       ``VIBEQC_LIVE_LOGGING=0`` for batch scripts that want a clean
       stdout for downstream parsing.

Wall: <5 s. The molecular SCFs are tiny (H2 / sto-3g) and there's
no periodic SCF here — periodic-SCF live logging is demonstrated
separately by the long ``examples/periodic/input-lih-pob-tzvp.py`` (the
real "extensive logging" use case where logging genuinely matters).

Run live:
    .venv/bin/python examples/molecular/input-progress-demo.py

Run silent:
    VIBEQC_LIVE_LOGGING=0 \\
        .venv/bin/python examples/molecular/input-progress-demo.py

Run under nohup with tail-able output:
    nohup .venv/bin/python examples/molecular/input-progress-demo.py \\
        > demo.log 2>&1 &
    tail -f demo.log

See ``docs/user_guide/output_files.md`` § "Progress logging" and
``docs/tutorial/reference_outputs.md`` for the full API + the
``.system`` manifest schema.
"""

from __future__ import annotations

import time
from pathlib import Path

import vibeqc as vq
from vibeqc.progress import ProgressLogger


HERE = Path(__file__).resolve().parent
OUTDIR = HERE / "output-progress-demo"
OUTDIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------
# 1. Molecular run_job — default-on progress, .out + .molden + .system
# ---------------------------------------------------------------------

print("=" * 72)
print(" 1. Molecular run_job (default-on progress)")
print("=" * 72)

mol = vq.Molecule([
    vq.Atom(1, [0.0, 0.0, 0.0]),
    vq.Atom(1, [0.0, 0.0, 1.4]),
])

# progress= defaults to True for run_job. The .out / .molden /
# .system files are still written either way; only the live mirror
# to stdout is suppressed when you pass progress=False.
result = vq.run_job(
    mol,
    basis="sto-3g",
    method="rhf",
    output=str(OUTDIR / "h2-rhf"),
    record_hostname=False,           # bundle-friendly default
    progress=True,                   # explicit, for clarity
)
print(f"\n  E(SCF) = {result.energy:.10f} Ha")
print(f"  outputs: {OUTDIR.name}/h2-rhf.{{out,molden,system}}")


# ---------------------------------------------------------------------
# 2. Direct ProgressLogger API — every public method, by hand
# ---------------------------------------------------------------------

print()
print("=" * 72)
print(" 2. ProgressLogger API surface (direct calls, no SCF)")
print("=" * 72)

# Construct a logger writing to stdout. The same object accepts a
# log_path= kwarg to ALSO write to a file (see section 3).
plog = ProgressLogger(verbose=True)

plog.banner("phase A - integrals")
plog.info("S, T, V lattice matrices at cutoff 12 bohr")
plog.info("ERIs to be assembled inside the Fock build (dynamic AO screening)")

with plog.stage("integrals_lattice", detail="overlap + kinetic + nuclear"):
    time.sleep(0.05)        # stand-in for a real compute step

plog.banner("phase B - SCF")
plog.info("Pulay DIIS, max 80 iters, conv_tol_energy = 1e-8")

# Iteration lines — the per-SCF-step format every Python-driven
# periodic SCF emits on stdout. Recognized fields: energy, dE,
# grad (||[F,DS]||), diis (subspace dim). Wall time since logger
# construction is appended automatically.
plog.iteration(1, energy=-1.0337, dE=None, grad=3.18e0, diis=0)
plog.iteration(2, energy=-1.1156, dE=-8.19e-2, grad=4.21e-1, diis=1)
plog.iteration(3, energy=-1.1167, dE=-1.10e-3, grad=2.34e-3, diis=2)
plog.iteration(4, energy=-1.1167, dE=-3.44e-7, grad=8.91e-7, diis=3)

plog.converged(n_iter=4, energy=-1.1167143, converged=True)

plog.banner("phase C - properties")
with plog.stage("dipole"):
    time.sleep(0.02)
with plog.stage("mulliken"):
    time.sleep(0.02)

plog.warn("this is what a warn message looks like - prefixed WARN:")

# write_raw splices in pre-formatted blocks (e.g. a geometry table
# from vibeqc.scf_log.format_scf_trace) without the per-line
# indentation that info() applies.
plog.write_raw(
    "  ---- raw block (e.g. format_scf_trace output) ----\n"
    "  iter   E_total           dE         ||[F,DS]||\n"
    "     1   -1.0337            --         3.18e+00\n"
    "     2   -1.1156   -8.19e-02         4.21e-01\n"
    "     3   -1.1167   -1.10e-03         2.34e-03\n"
    "     4   -1.1167   -3.44e-07         8.91e-07\n"
    "  ----------------------------------------------\n"
)


# ---------------------------------------------------------------------
# 3. Custom ProgressLogger — tee stdout + persistent file
# ---------------------------------------------------------------------

print()
print("=" * 72)
print(" 3. ProgressLogger(log_path=...) - tee to a tail-able file")
print("=" * 72)

# log_path= writes to disk in addition to stdout. The file is
# truncated on construction and flushed after every write, so a
# `tail -f` from another terminal sees lines appear in real time.
log_path = OUTDIR / "demo.log"
tee_log = ProgressLogger(log_path=log_path, verbose=True)

tee_log.banner("custom tee'd logger")
tee_log.info(f"writing to {log_path.name} alongside stdout")

# Thread the same instance through a real molecular job so its banner,
# native iteration rows, stages, and terminal summary also land in demo.log.
with tee_log.stage("scf", detail="H2 / sto-3g, run_job"):
    rhf_result = vq.run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        output=OUTDIR / "h2-rhf-tee",
        progress=tee_log,
        record_hostname=False,
        citations=False,
        write_molden_file=False,
        output_qvf=False,
    )

with tee_log.stage("summary"):
    tee_log.info(f"E      = {rhf_result.energy:.8f} Ha")
    tee_log.info(f"iters  = {rhf_result.n_iter}")
    tee_log.info(f"both stdout AND {log_path.name} captured this run")

tee_log.banner("done")

print(f"\n  Live mirror saved to: {log_path}")


# ---------------------------------------------------------------------
# 4. Off switch
# ---------------------------------------------------------------------

print()
print("=" * 72)
print(" 4. Silent mode (progress=False)")
print("=" * 72)

# progress=False suppresses ALL live output - no banner, no
# per-iteration trace, no done message. The result object is still
# returned, the SCF still converges, the .out file (if you used
# run_job) is still written. Useful for batch scripts that want a
# clean stdout for downstream parsing.
quiet = vq.run_job(
    mol,
    basis="sto-3g",
    method="rhf",
    output=str(OUTDIR / "h2-rhf-silent"),
    record_hostname=False,
    progress=False,                   # <-- nothing on stdout
)
print(f"  E (silent run) = {quiet.energy:.10f} Ha")
print(f"  (no banner, no SCF trace - the .out file still got written)")
print(f"  Same result as section 1, just no live mirror.")
print()
print("  The env-var equivalent silences everything globally:")
print("      VIBEQC_LIVE_LOGGING=0 python this_script.py")


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

print()
print("=" * 72)
print(" Summary")
print("=" * 72)
print(f"  generated outputs: {OUTDIR.name}/h2-rhf.{{out,molden,system}}")
print(f"  silent run:       {OUTDIR.name}/h2-rhf-silent.{{out,molden,system}}")
print(f"  custom log:       {OUTDIR.name}/demo.log")
print()
print("  See examples/periodic/input-lih-pob-tzvp.py for the slow-SCF use case")
print("  where live progress actually matters - a multi-minute periodic")
print("  pob-TZVP run that without progress=True would look frozen.")
