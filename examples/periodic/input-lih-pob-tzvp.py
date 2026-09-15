"""LiH cubic rocksalt — pob-TZVP / RHF / EWALD_3D.

End-to-end periodic SCF + property surface + crystalline-orbital
cube on a real solid. Produces:

  - Per-cell SCF energy + bandgap (HOMO over IBZ vs LUMO over IBZ)
  - Hcore band structure along the standard cubic Γ-X-W-K-Γ-L-U-W-L-K|U-X
    HPKOT path
  - Hcore density of states + atom-l projected DOS (Li-s, H-s)
  - HOMO Bloch orbital at Γ as a .cube for VESTA / XCrySDen / VMD
  - **Three sibling output files** demonstrating every v0.5.x
    observability surface:
        output-lih-pob-tzvp.out       — line-buffered SCF log
        output-lih-pob-tzvp.system    — runtime manifest (TOML; v0.5.1)
        output-lih-pob-tzvp.perf      — post-mortem perf report (v0.5.2)

Wall: ~5–15 min on a laptop, ~2–3 min on a 32-core box. Set
``OMP_NUM_THREADS=8`` first if you want the timing block in the
.perf to mean anything.

Observability surfaces (all three composing freely):

  - **v0.5.1 — live progress.** ``progress=True`` on every SCF
    entry point streams a flushed banner + per-stage timings +
    per-iteration SCF lines (``iter N  E = …  dE = …  ||[F,DS]|| = …
    DIIS=k  [wall]``) so multi-minute jobs don't go silent.
    ``VIBEQC_LIVE_LOGGING=0`` opts out.

  - **v0.5.1 — runtime manifest.** ``run_job`` writes a third
    sibling ``.system`` file pinning the runtime environment
    (vibe-qc version + git SHA, host + CPU + RAM, OMP threads,
    library versions) so wall-time numbers are interpretable.
    For periodic SCFs (which don't go through ``run_job`` today)
    we call ``vq.write_system_manifest`` directly.

  - **v0.5.2 — post-mortem perf log.** ``with vq.perf_log(...):``
    accumulates per-phase wall/CPU timings, per-iteration SCF
    rows, and memory snapshots into a sortable plain-text report.
    Pairs with the live progress: ``progress`` answers "is it
    stuck?", ``perf_log`` answers "where did the time go?".

Run live + persistent + perf:
    .venv/bin/python examples/periodic/input-lih-pob-tzvp.py

Run under nohup with live tail + perf:
    nohup .venv/bin/python examples/periodic/input-lih-pob-tzvp.py \\
        > LiH-stdout.log 2>&1 &
    tail -f examples/periodic/output-lih-pob-tzvp.out          # v0.5.1 live
    cat examples/periodic/output-lih-pob-tzvp.perf             # v0.5.2 post-mortem
    cat examples/periodic/output-lih-pob-tzvp.system           # v0.5.1 manifest

Disable progress (still writes .out / .system / .perf):
    VIBEQC_LIVE_LOGGING=0 .venv/bin/python \\
        examples/periodic/input-lih-pob-tzvp.py
"""

import time
from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc.progress import ProgressLogger

HERE = Path(__file__).resolve().parent
OUT_STEM = HERE / "output-lih-pob-tzvp"

HARTREE_TO_EV = 27.211386245988

t_start = time.perf_counter()
# v0.5.1 ProgressLogger that mirrors to BOTH stdout AND the .out file.
# The same instance is threaded through every SCF (progress=plog) so
# per-iteration trace lines land in the persistent log.
plog = ProgressLogger(log_path=OUT_STEM.with_suffix(".out"), verbose=True)

# v0.5.2 perf_log — context manager around the whole script. Per-phase
# timings, per-iteration SCF rows, memory snapshots accumulate into
# the .perf file at exit. Independent of plog: progress is "is it
# stuck?", perf is "where did the time go?".
with vq.perf_log(OUT_STEM.with_suffix(".perf")):

    plog.banner("LiH rocksalt  /  RHF  /  pob-TZVP  /  EWALD_3D")

    # ---- 1. Build the LiH conventional cubic cell --------------------
    a_ang = 4.084                                 # experimental
    a = a_ang / 0.529177210903                    # bohr
    lat = a * np.eye(3)

    # 4 Li at FCC sites + 4 H at the displaced FCC (rocksalt)
    unit_cell = []
    for fx, fy, fz in [(0, 0, 0), (0, 0.5, 0.5),
                       (0.5, 0, 0.5), (0.5, 0.5, 0)]:
        unit_cell.append(vq.Atom(3, [fx*a, fy*a, fz*a]))    # Li
    for fx, fy, fz in [(0.5, 0.5, 0.5), (0.5, 0, 0),
                       (0, 0.5, 0), (0, 0, 0.5)]:
        unit_cell.append(vq.Atom(1, [fx*a, fy*a, fz*a]))    # H

    system = vq.PeriodicSystem(dim=3, lattice=lat, unit_cell=unit_cell)
    basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp")
    vq.attach_symmetry(system, symprec=1e-4)
    plog.info(
        f"spacegroup: {system.symmetry.international_symbol} "
        f"(SG {system.symmetry.number}, "
        f"point group {system.symmetry.point_group})"
    )
    plog.info(f"basis: pob-TZVP, {basis.nbasis} bf per cell")

    # ---- 2. SCF with EWALD_3D + IBZ-reduced 4×4×4 mesh --------------
    opts = vq.PeriodicSCFOptions()
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    opts.conv_tol_energy = 1e-8

    kpts = vq.KPoints.monkhorst_pack(system, [4, 4, 4], symmetry=True)
    plog.info(
        f"k-mesh: 4×4×4 → {len(kpts)} IBZ points "
        f"(out of 64 in the full mesh)"
    )

    # progress=plog threads the live logger; per-iter SCF lines land
    # in BOTH stdout and the .out file. Inside the perf_log block,
    # the SCF also reports per-phase timings into the .perf file.
    result = vq.run_rhf_periodic_scf(
        system, basis, kpts, opts, progress=plog,
    )
    plog.info(
        f"E/cell = {result.energy:.8f} Ha   ({result.n_iter} iters)"
    )

    # ---- 2a. HOMO-LUMO gap from per-k orbital energies ---------------
    # `bands.gap_at(0)` from the original sketch is a v0.6 helper —
    # today we read the gap straight off the SCF's per-k mo_energies
    # which works on every shipped version of vibe-qc.
    n_occ = sum(int(at.Z) for at in system.unit_cell) // 2
    homo = max(float(eps[n_occ - 1]) for eps in result.mo_energies)
    lumo = min(float(eps[n_occ])     for eps in result.mo_energies)
    plog.info(
        f"HOMO max = {homo*HARTREE_TO_EV:.3f} eV   "
        f"LUMO min = {lumo*HARTREE_TO_EV:.3f} eV"
    )
    plog.info(
        f"Gap     = {(lumo - homo) * HARTREE_TO_EV:.3f} eV "
        f"(HF over-estimates by ~3 eV; LiH experimental optical ~5 eV)"
    )

    # ---- 3. Bands + DOS + PDOS along Γ-X-W-K-Γ-L-... ----------------
    band_path = vq.KPoints.band_path(system)        # HPKOT via seekpath
    plog.info(
        f"band path: {' → '.join(l for _, l in band_path.labels)}"
    )

    # Hcore (non-interacting) along the HPKOT path. Right band shapes
    # / orbital characters; absolute eigenvalues differ from SCF by
    # the missing Hartree + exchange contributions. SCF-Fock-based
    # bands need F_real exposed as a LatticeMatrixSet from the
    # converged SCF — small v0.6 wiring task.
    with plog.stage("bands_hcore"):
        bands = vq.band_structure_hcore(
            system, basis, band_path.to_kpath(),
            n_electrons_per_cell=2 * n_occ,
        )
    with plog.stage("dos_hcore"):
        dos = vq.density_of_states_hcore(
            system, basis, [4, 4, 4],
            sigma=0.01, n_electrons_per_cell=2 * n_occ,
        )
    with plog.stage("pdos_hcore"):
        pdos = vq.density_of_states_projected_hcore(
            system, basis, [4, 4, 4],
            projection="atoms_l",
            sigma=0.01, n_electrons_per_cell=2 * n_occ,
        )

    direct_gap_eV = (
        float(bands.energies[0, n_occ])
        - float(bands.energies[0, n_occ - 1])
    ) * HARTREE_TO_EV
    plog.info(
        f"bandgap (Hcore, direct at first k = Γ): {direct_gap_eV:.2f} eV"
    )

    # ---- 4. HOMO Bloch orbital cube at Γ -----------------------------
    # The 4×4×4 MP mesh doesn't include exact Γ. Re-run a single Γ
    # SCF to get clean Γ-point orbitals for the cube; progress=plog
    # again so the second SCF lands in the same .out file.
    plog.banner("Γ-only SCF for HOMO cube")
    gamma = vq.KPoints.gamma(system)
    gamma_result = vq.run_rhf_periodic_scf(
        system, basis, gamma, opts, progress=plog,
    )

    cube_path = OUT_STEM.parent / "output-lih-pob-tzvp-homo.cube"
    with plog.stage("write_cube_mo_periodic", detail="2×2×2 supercell tile"):
        vq.write_cube_mo_periodic(
            cube_path,
            system, basis,
            np.asarray(gamma_result.mo_coeffs[0]),
            np.zeros(3),                              # k = Γ
            n_occ - 1,                                # HOMO band index
            spacing_bohr=0.20,
            n_replica=(2, 2, 2),                      # show 2×2×2 supercell
        )
    plog.info(f"wrote {cube_path.name}")
    plog.info("open with VESTA / XCrySDen / VMD — H 1s combinations")
    plog.info("bonded between Li⁺ centers across the 2×2×2 tile")

    # ---- 5. v0.5.1 .system manifest sibling --------------------------
    # run_job writes this automatically; for periodic SCFs we have to
    # call write_system_manifest by hand. The TOML pins the runtime
    # environment (vibe-qc version + git SHA, host CPU/RAM/OMP, library
    # versions) so the wall-time numbers in the .out / .perf are
    # interpretable across machines.
    with plog.stage("write_system_manifest"):
        import os
        vq.write_system_manifest(
            OUT_STEM.with_suffix(".system"),
            wall_seconds=time.perf_counter() - t_start,
            basename=OUT_STEM.name,
            record_hostname=os.environ.get(
                "VIBEQC_NO_HOSTNAME", ""
            ).strip().lower() not in ("1", "true", "yes", "on"),
        )
    plog.info(f"wrote {OUT_STEM.with_suffix('.system').name}")

    plog.banner("Done")
    plog.info(f".out:    {OUT_STEM.with_suffix('.out').name}")
    plog.info(f".system: {OUT_STEM.with_suffix('.system').name}")
    plog.info(f".perf:   {OUT_STEM.with_suffix('.perf').name}  "
              "(written when this perf_log block exits)")
    plog.info(f"cube:    {cube_path.name}")
