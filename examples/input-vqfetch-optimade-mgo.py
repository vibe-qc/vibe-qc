"""vqfetch — programmatic OPTIMADE fetch of MgO, emit + run.

Walks the v0.8.0 ``vibeqc.fetch`` Python API end-to-end:

  1. ``fetch_optimade()``        — pull MgO rocksalt from the Materials
                                    Project OPTIMADE endpoint.
  2. Inspect ``PeriodicSpec``    — lattice, atoms, recommended basis,
                                    Provenance (DOI + license).
  3. ``emit_spec_module()``      — write a regression-suite SPEC.
  4. ``emit_input_script()``     — write a standalone vibe-qc input.
  5. Run vibe-qc on the result   — Γ-only RKS / LDA / STO-3G, the
                                    fastest periodic smoke vibe-qc has.

Cache-friendly: the first run hits Materials Project's OPTIMADE
endpoint (~1 s); subsequent runs replay from
``$XDG_CACHE_HOME/vibeqc/fetch/`` (default ``~/.cache/vibeqc/fetch/``)
unless ``--no-cache`` is set.

Output files land in ``/tmp/vqfetch_optimade_mgo/`` so re-running
doesn't clutter ``examples/``.

Run:
    .venv/bin/python examples/input-vqfetch-optimade-mgo.py

Paired with the user-facing walkthrough in
``docs/user_guide/external_structures.md`` and tutorial 30
(``docs/tutorial/external_data_fetcher.md``).
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc.fetch import (
    emit_input_script,
    emit_spec_module,
    fetch_optimade,
)
from vibeqc.progress import ProgressLogger


OUTDIR = Path("/tmp/vqfetch_optimade_mgo")
OUTDIR.mkdir(parents=True, exist_ok=True)
SPEC_DIR = OUTDIR / "specs"
INPUT_DIR = OUTDIR / "inputs"
SPEC_DIR.mkdir(exist_ok=True)
INPUT_DIR.mkdir(exist_ok=True)


# ---- 1. Fetch ---------------------------------------------------------------

# Why ID-lookup ('mp/mp-1265') instead of formula='MgO':
# ----------------------------------------------------------------------
# MgO has multiple polymorphs in the Materials Project database:
#
#   mp-1265     rocksalt   Fm-3m   8 atoms / conventional cell  ← canonical
#   mp-1009127  CsCl-like  Pm-3m   2 atoms / conventional cell    (HP phase)
#   mp-1191789  hexagonal  P6_3mc                                  (HP phase)
#
# A formula query like `fetch_optimade(formula='MgO')` returns
# whichever entry MP serves first, which may NOT be the textbook
# rocksalt. The fetcher's plurality-voting picker needs multiple
# providers (mp + oqmd + aflow + …) to agree before it can
# disambiguate — single-provider formula queries are ambiguous.
#
# Three robust patterns:
#   (a) ID-lookup           fetch_optimade(optimade_id="mp/mp-1265")
#   (b) Federation default  fetch_optimade(formula="MgO")              # no provider
#   (c) Canonical set       vqfetch canonical mgo_rocksalt               # CLI
#
# This example uses (a). See § 11 of the structure-fetcher handover
# for the failure mode in full, and the multi-candidate API below
# (`max_results=N`) for the interactive-disambiguation path.

print("\n  [1/5] fetch_optimade(optimade_id='mp/mp-1265', quick=True)")
t0 = time.perf_counter()
specs = fetch_optimade(optimade_id="mp/mp-1265", quick=True)
print(f"        returned {len(specs)} candidate(s) in {time.perf_counter()-t0:.2f}s "
      f"(first run hits network; subsequent runs replay from cache)")
spec = specs[0]

# Sidebar — multi-candidate API (commented out, doesn't run):
#
#   candidates = fetch_optimade(formula="MgO", max_results=5)
#   for s in candidates:
#       print(f"  sg={s.space_group}  a={s.lattice_ang[0][0]:.3f}  "
#             f"n_atoms={len(s.atoms)}  {s.provenance.source_id}")
#
# Returns a deduped list of distinct polymorph candidates ranked by
# space-group plurality. The caller picks by inspection — useful in
# interactive notebooks. See VFETCH-X1 in docs/roadmap.md for the
# `vqfetch list-candidates` CLI surface (v0.8.x).


# ---- 2. Inspect what we got -------------------------------------------------

print("\n  [2/5] Inspect the PeriodicSpec")
a = spec.lattice_ang[0][0]
print(f"        id                 = {spec.id!r}")
print(f"        family             = {spec.family!r}")
print(f"        space_group        = {spec.space_group!r}")
print(f"        lattice a (Å)      = {a:.4f}  (MP-relaxed; experimental ≈ 4.211)")
print(f"        n_atoms            = {len(spec.atoms)}")
print(f"        recommended_basis  = {spec.recommended_basis!r}")
print(f"        default_initial_guess = {spec.default_initial_guess!r}")
print(f"        default_damping    = {spec.default_damping}")

print("\n        Provenance:")
p = spec.provenance
print(f"          source_db          = {p.source_db}")
print(f"          source_id          = {p.source_id}")
print(f"          source_url         = {p.source_url}")
print(f"          original_reference = {p.original_reference or '(none)'}")
print(f"          license            = {p.license}")
print(f"          fetched_at         = {p.fetched_at}")
print(f"          fetcher_version    = {p.fetcher_version}")


# ---- 3. Emit a regression-suite SPEC ----------------------------------------

print("\n  [3/5] emit_spec_module(spec, examples/regression/systems/periodic/)")
spec_path = emit_spec_module(spec, SPEC_DIR)
print(f"        wrote {spec_path}")
print( "        consumable by examples.regression.run_suite — the emitted")
print( "        module re-imports cleanly because we re-print floats via")
print( "        repr() (shortest round-tripping decimal).")


# ---- 4. Emit a standalone executable input script ---------------------------

print("\n  [4/5] emit_input_script(spec, examples/, basis='sto-3g', method='rks-lda')")
input_path = emit_input_script(
    spec, INPUT_DIR, basis="sto-3g", method="rks-lda",
)
print(f"        wrote {input_path}")
print( "        runnable end-to-end with `.venv/bin/python <that path>`")


# ---- 5. Run vibe-qc Γ-only RKS/LDA on the fetched cell (opt-in) -------------

# Default: skip the SCF — the conventional 8-atom rocksalt cell at
# STO-3G / RKS-LDA takes ~2 h on a laptop (~30 min at Γ-only on 16
# cores). Set VQFETCH_RUN_SCF=1 to opt in. The expected result, for
# reference: compute-reference (16 cores, 2026-05-09) converged the same SPEC
# in 13 iters to E/cell = -950.4204308512 Ha (HOMO-LUMO gap 6.486 eV).

if os.environ.get("VQFETCH_RUN_SCF") == "1":
    print("\n  [5/5] vibe-qc Γ-only RKS / LDA / STO-3G on the fetched cell")
    print( "        (uses the SPEC's defaults: SAD guess + damping 0.85 for ionic)")
    t0 = time.perf_counter()

    ANGSTROM_TO_BOHR = 1.0 / 0.529177210903
    lat_bohr = np.array(spec.lattice_ang, dtype=float) * ANGSTROM_TO_BOHR
    unit_cell = []
    for at in spec.atoms:
        cart_bohr = lat_bohr @ np.asarray(at.frac, dtype=float)
        unit_cell.append(vq.Atom(at.z, list(cart_bohr)))

    system = vq.PeriodicSystem(dim=3, lattice=lat_bohr, unit_cell=unit_cell)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    vq.attach_symmetry(system, symprec=1e-4)

    opts = vq.PeriodicKSOptions()
    opts.functional = "LDA"
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.cutoff_bohr = spec.default_cutoff_bohr
    opts.conv_tol_energy = spec.default_conv_tol_energy
    opts.max_iter = spec.default_max_iter
    opts.damping = spec.default_damping
    opts.initial_guess = vq.InitialGuess.SAD

    kpts = vq.KPoints.monkhorst_pack(system, [1, 1, 1], symmetry=True)
    plog = ProgressLogger(log_path=OUTDIR / "scf.log", verbose=False)
    result = vq.run_rks_periodic_scf(system, basis, kpts, opts, progress=plog)

    wall = time.perf_counter() - t0
    print(f"        E/cell = {result.energy:.8f} Ha  ({result.n_iter} iters, {wall:.1f}s)")
    print( "        Reference: compute-reference 16-core, 13 iters, E = -950.4204308512 Ha.")
else:
    print("\n  [5/5] SCF — skipped (set VQFETCH_RUN_SCF=1 to enable).")
    print( "        Conventional 8-atom rocksalt at STO-3G / RKS-LDA is slow")
    print( "        on laptop (~2 h); the compute-reference 16-core reference run on this")
    print( "        same fetched SPEC converged in 13 iters to")
    print( "        E/cell = -950.4204308512 Ha (HOMO-LUMO gap 6.486 eV).")
    print(f"        To run the emitted input on a beefier box:")
    print(f"            ssh compute-reference '.venv/bin/python {input_path.name}'")
    print( "        or use `vq submit ... --cpus 16 --wall-time-seconds 7200`.")

print("\n  Done. Artefacts at:")
print(f"    SPEC          {spec_path}")
print(f"    input script  {input_path}")
if os.environ.get("VQFETCH_RUN_SCF") == "1":
    print(f"    SCF log       {OUTDIR / 'scf.log'}")
print()
print( "  Cache: set VIBEQC_FETCH_CACHE_ONLY=1 to verify subsequent runs")
print( "  replay offline from $XDG_CACHE_HOME/vibeqc/fetch/.")
