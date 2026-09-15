"""NaCl rocksalt — STO-3G / RKS-LDA / EWALD_3D — debug-friendly solid demo.

Solid-state demo for iterating on the v0.5.x logging + plotting
surfaces. Different from the LiH and MgO siblings:

  - **STO-3G** (not pob-TZVP) — minimal basis, 72 bf per
    conventional unit cell, no linear-dependency issues. Bands /
    DOS shapes are rough at minimal basis but the workflow is
    identical to a production-quality run.
  - **RKS / LDA** (Slater + VWN5) — exercises the periodic XC
    integration path (Phase 12f periodic Becke partition) on top
    of the EWALD_3D Coulomb dispatch. LDA is cheaper than GGA
    (no density-gradient evaluation in the XC kernel).
  - **NaCl rocksalt** — Na+ vs Cl- is chemically distinct from LiH
    and MgO: more polarizable, narrower gap, different DOS shape.
  - **2×2×2 IBZ-reduced mesh** (Fm-3m → 1 unique k-point) gives a
    Γ-equivalent SCF cell. Coarsened FFT grid (spacing_bohr=0.6)
    keeps the per-iteration wall well under a minute on a laptop.

What this exercises (every v0.5.x observability + visualization
surface):

  - v0.5.1 ProgressLogger threading: live SCF iteration trace
    mirrored to stdout AND a persistent .out file
  - v0.5.1 .system manifest sibling pinning the runtime environment
  - v0.5.2 perf log accumulator: per-phase timings + per-iter SCF
    rows + memory snapshots into the .perf file
  - Hcore band structure along the auto-detected HPKOT k-path
    (cubic-F: Γ-X-W-K-Γ-L-U-W-L-K|U-X)
  - Hcore total DOS + atom/l-projected DOS (Na-s, Na-p, Cl-s, Cl-p)
  - HOMO Bloch orbital at Γ as a Gaussian cube file for VESTA /
    XCrySDen / VMD
  - matplotlib triple-panel bands + DOS + PDOS plot

Outputs (next to this script in ``examples/periodic/``):
    output-nacl-sto3g-dft.out         — line-buffered SCF log
    output-nacl-sto3g-dft.system      — runtime manifest (TOML; v0.5.1)
    output-nacl-sto3g-dft.perf        — post-mortem perf report (v0.5.2)
    output-nacl-sto3g-dft-bands.png   — bands + DOS + PDOS triple panel
    output-nacl-sto3g-dft-homo.cube   — HOMO Bloch orbital, 2x2x2 tile

Wall: ~1-2 min on a laptop, ~30 s on a 32-core box. Designed to be
short enough to iterate on quickly when debugging logging or plot
output without waiting for a multi-minute SCF.

Run live + persistent + perf:
    .venv/bin/python examples/periodic/input-nacl-sto3g-dft.py

Run under nohup with tail-able output:
    nohup .venv/bin/python examples/periodic/input-nacl-sto3g-dft.py \\
        > nacl-stdout.log 2>&1 &
    tail -f examples/periodic/output-nacl-sto3g-dft.out

Disable progress (still writes .out / .system / .perf):
    VIBEQC_LIVE_LOGGING=0 .venv/bin/python \\
        examples/periodic/input-nacl-sto3g-dft.py

Fast-debug profile (aggressively coarsens every accuracy knob —
spacing 0.6→0.8, cutoff 12→8, nuclear cutoff 25→15, omega 0.5→0.7,
conv tol 1e-7→1e-5, max iters 80→25). Use to iterate on the logging
surface; numbers will be wrong but the workflow is the same:
    VIBEQC_FAST_DEBUG=1 .venv/bin/python \\
        examples/periodic/input-nacl-sto3g-dft.py

Note: per-iteration EWALD_3D wall is dominated by the FFT + JK build.
Parallel work on Schwarz screening (in another v0.5.x patch) will
collapse the K cost dramatically and reduce the need for FAST_DEBUG.
"""

import os
import time
from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc.progress import ProgressLogger

HERE = Path(__file__).resolve().parent
OUT_STEM = HERE / "output-nacl-sto3g-dft"

HARTREE_TO_EV = 27.211386245988

# Lattice parameter (experimental, room temperature, NaCl).
A_ANG = 5.640
A_BOHR = A_ANG / 0.529177210903

# Rocksalt: Na at FCC sites, Cl at displaced FCC.
NA_FRAC = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
CL_FRAC = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]


# ---------------------------------------------------------------------
# Convergence / accuracy knobs - two profiles for fast iteration
# ---------------------------------------------------------------------
#
# Default profile is "debug-friendly" — already aggressively coarsened
# vs production-precision values. Set ``VIBEQC_FAST_DEBUG=1`` in the
# environment to push every knob harder for the fastest possible
# round-trip, at the cost of any pretense to physical accuracy. Use
# the FAST profile when iterating on plot output / log formatting /
# script structure; flip it OFF (and ideally bump to spacing_bohr=0.3)
# for any number you'd actually report.
#
#                                  default     FAST_DEBUG    production
#                                                            (drop FAST,
#                                                             override)
# spacing_bohr (FFT grid)            0.6        0.8           0.3
# cutoff_bohr (AO real-space)        12.0       8.0           15-20
# nuclear_cutoff_bohr                25.0       15.0          30-40
# omega (Ewald α split)              0.5        0.7           0.5
# conv_tol_energy                    1e-7       1e-5          1e-9
# max_iter                           80         25            150+
_FAST_DEBUG = os.environ.get(
    "VIBEQC_FAST_DEBUG", ""
).strip().lower() in ("1", "true", "yes", "on")

if _FAST_DEBUG:
    SPACING_BOHR     = 0.8
    CUTOFF_BOHR      = 8.0
    NUCLEAR_CUTOFF   = 15.0
    OMEGA            = 0.7
    CONV_TOL_ENERGY  = 1e-5
    MAX_ITER         = 25
    PROFILE_LABEL    = "FAST_DEBUG (numbers may be wrong; logging surface verified)"
else:
    SPACING_BOHR     = 0.6
    CUTOFF_BOHR      = 12.0
    NUCLEAR_CUTOFF   = 25.0
    OMEGA            = 0.5
    CONV_TOL_ENERGY  = 1e-7
    MAX_ITER         = 80
    PROFILE_LABEL    = "default (debug-friendly; not production precision)"


t_start = time.perf_counter()
plog = ProgressLogger(log_path=OUT_STEM.with_suffix(".out"), verbose=True)

with vq.perf_log(OUT_STEM.with_suffix(".perf")):

    plog.banner("NaCl rocksalt  /  RKS / LDA  /  STO-3G  /  EWALD_3D")
    plog.info(f"profile:    {PROFILE_LABEL}")

    # ---- 1. Build the NaCl conventional cubic cell ------------------
    lat = A_BOHR * np.eye(3)
    unit_cell = []
    for fx, fy, fz in NA_FRAC:
        unit_cell.append(vq.Atom(11, [fx*A_BOHR, fy*A_BOHR, fz*A_BOHR]))
    for fx, fy, fz in CL_FRAC:
        unit_cell.append(vq.Atom(17, [fx*A_BOHR, fy*A_BOHR, fz*A_BOHR]))

    system = vq.PeriodicSystem(dim=3, lattice=lat, unit_cell=unit_cell)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    vq.attach_symmetry(system, symprec=1e-4)
    plog.info(
        f"cell:       {A_ANG} Å cubic, {len(system.unit_cell)} atoms (4 Na + 4 Cl)"
    )
    plog.info(
        f"spacegroup: {system.symmetry.international_symbol} "
        f"(SG {system.symmetry.number}, "
        f"point group {system.symmetry.point_group})"
    )
    plog.info(f"basis:      sto-3g, {basis.nbasis} bf per cell")

    # ---- 2. RKS / LDA SCF with EWALD_3D + IBZ-reduced 2×2×2 ---------
    opts = vq.PeriodicKSOptions()
    opts.functional = "LDA"      # Slater exchange + VWN5 correlation
    opts.use_periodic_becke = True              # tight 3D bulk recommendation
    opts.becke_image_radius_bohr = 12.0
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.cutoff_bohr = CUTOFF_BOHR
    opts.lattice_opts.nuclear_cutoff_bohr = NUCLEAR_CUTOFF
    opts.conv_tol_energy = CONV_TOL_ENERGY
    opts.max_iter = MAX_ITER

    kpts = vq.KPoints.monkhorst_pack(system, [2, 2, 2], symmetry=True)
    plog.info(
        f"k-mesh:     2×2×2 → {len(kpts)} IBZ point(s) (Fm-3m: ~Γ-only)"
    )

    # FFT-grid spacing + Ewald α split go in via dispatcher kwargs.
    # SPACING_BOHR / OMEGA come from the FAST_DEBUG profile selection
    # at the top of the file - the live-logging output is the same
    # regardless of profile, so use FAST_DEBUG to iterate on the
    # logging surface and the default for chemistry-grade numbers.
    result = vq.run_rks_periodic_scf(
        system, basis, kpts, opts, progress=plog,
        spacing_bohr=SPACING_BOHR,
        omega=OMEGA,
    )
    plog.info(f"E/cell = {result.energy:.8f} Ha   ({result.n_iter} iters)")

    # ---- 2a. HOMO-LUMO gap from per-k orbital energies --------------
    n_occ = sum(int(at.Z) for at in system.unit_cell) // 2
    homo = max(float(eps[n_occ - 1]) for eps in result.mo_energies)
    lumo = min(float(eps[n_occ])     for eps in result.mo_energies)
    plog.info(
        f"HOMO max = {homo*HARTREE_TO_EV:.3f} eV   "
        f"LUMO min = {lumo*HARTREE_TO_EV:.3f} eV"
    )
    plog.info(
        f"Gap     = {(lumo - homo) * HARTREE_TO_EV:.3f} eV "
        f"(LDA under-estimates ionic gaps; NaCl experimental optical ~8.7 eV)"
    )

    # ---- 3. Hcore bands + DOS + PDOS -------------------------------
    band_path = vq.KPoints.band_path(system)
    plog.info(
        f"band path: {' → '.join(l for _, l in band_path.labels)}"
    )

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

    # ---- 4. HOMO Bloch orbital cube at Γ ----------------------------
    plog.banner("Γ-only SCF for HOMO cube")
    gamma = vq.KPoints.gamma(system)
    gamma_result = vq.run_rks_periodic_scf(
        system, basis, gamma, opts, progress=plog,
        spacing_bohr=SPACING_BOHR,
        omega=OMEGA,
    )

    cube_path = OUT_STEM.parent / "output-nacl-sto3g-dft-homo.cube"
    with plog.stage("write_cube_mo_periodic", detail="2×2×2 supercell tile"):
        vq.write_cube_mo_periodic(
            cube_path,
            system, basis,
            np.asarray(gamma_result.mo_coeffs[0]),
            np.zeros(3),
            n_occ - 1,
            spacing_bohr=0.25,
            n_replica=(2, 2, 2),
            title=f"NaCl HOMO @ Γ (RKS/LDA/STO-3G)",
        )
    plog.info(f"wrote {cube_path.name}")

    # ---- 5. Plot bands + DOS + PDOS --------------------------------
    with plog.stage("plot", detail="bands + DOS + PDOS triple panel"):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            plog.warn("matplotlib not available - skipping plot")
        else:
            fig = plt.figure(figsize=(11, 4.6), dpi=150)
            gs = fig.add_gridspec(
                1, 3, width_ratios=[3.0, 1.0, 1.6], wspace=0.05,
            )
            ax_b = fig.add_subplot(gs[0, 0])
            ax_d = fig.add_subplot(gs[0, 1], sharey=ax_b)
            ax_p = fig.add_subplot(gs[0, 2], sharey=ax_b)

            e_fermi = bands.e_fermi if bands.e_fermi is not None else homo
            e_b = (bands.energies - e_fermi) * HARTREE_TO_EV
            e_d = (dos.energies - e_fermi) * HARTREE_TO_EV
            e_p = (pdos.energies - e_fermi) * HARTREE_TO_EV

            for i in range(e_b.shape[1]):
                color = "#1f77b4" if i < n_occ else "#aaaaaa"
                ax_b.plot(bands.kpath.distances, e_b[:, i],
                          color=color, linewidth=1.0)
            ax_b.axhline(0.0, color="black", linewidth=0.8,
                         linestyle="--", alpha=0.7)
            tick_pos, tick_txt = zip(*bands.kpath.labels)
            ax_b.set_xticks(tick_pos)
            ax_b.set_xticklabels(tick_txt, fontsize=10)
            ax_b.set_xlim(bands.kpath.distances[0],
                          bands.kpath.distances[-1])
            ax_b.set_ylabel(r"$E - E_F$  /  eV")
            ax_b.set_title("Hcore bands (HPKOT)")
            ax_b.grid(axis="y", alpha=0.3, linestyle=":")

            ax_d.fill_betweenx(e_d, dos.dos, color="#ff7f0e",
                               alpha=0.55, linewidth=0)
            ax_d.plot(dos.dos, e_d, color="#ff7f0e", linewidth=1.4)
            ax_d.axhline(0.0, color="black", linewidth=0.8,
                         linestyle="--", alpha=0.7)
            ax_d.set_xlim(left=0)
            ax_d.set_xlabel("DOS")
            ax_d.set_title("Total DOS")
            ax_d.tick_params(labelleft=False)

            # PDOS — aggregate Na vs Cl by l character.
            def aggregate(species, lchar):
                return sum(
                    (c for label, c in pdos.contributions.items()
                     if label.startswith(species) and label.endswith(lchar)),
                    start=np.zeros_like(e_p),
                )

            for spec, lchar, color, alpha in [
                ("Na", "-s", "#1f77b4", 1.0),
                ("Na", "-p", "#9467bd", 0.7),
                ("Cl", "-s", "#d62728", 1.0),
                ("Cl", "-p", "#8c564b", 0.7),
            ]:
                vals = aggregate(spec, lchar)
                if vals.any():
                    ax_p.plot(vals, e_p, color=color, alpha=alpha,
                              linewidth=1.4, label=f"{spec}{lchar}")
            ax_p.axhline(0.0, color="black", linewidth=0.8,
                         linestyle="--", alpha=0.7)
            ax_p.set_xlabel("PDOS")
            ax_p.set_xlim(left=0)
            ax_p.set_title("Atom/l-projected DOS")
            ax_p.tick_params(labelleft=False)
            ax_p.legend(loc="upper right", fontsize=8, frameon=True)

            ax_b.set_ylim(max(e_b.min(), -25.0), min(e_b.max(), 25.0))
            fig.suptitle(
                f"NaCl rocksalt - RKS / LDA / STO-3G - a = {A_ANG} Å",
                fontsize=12, y=1.0,
            )
            png_path = OUT_STEM.parent / "output-nacl-sto3g-dft-bands.png"
            fig.savefig(png_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            plog.info(f"wrote {png_path.name}")

    # ---- 6. Runtime manifest ----------------------------------------
    with plog.stage("write_system_manifest"):
        vq.write_system_manifest(
            OUT_STEM.with_suffix(".system"),
            wall_seconds=time.perf_counter() - t_start,
            basename=OUT_STEM.name,
            record_hostname=os.environ.get(
                "VIBEQC_NO_HOSTNAME", ""
            ).strip().lower() not in ("1", "true", "yes", "on"),
        )

    plog.banner("Done")
    plog.info(f".out:    {OUT_STEM.with_suffix('.out').name}")
    plog.info(f".system: {OUT_STEM.with_suffix('.system').name}")
    plog.info(f".perf:   {OUT_STEM.with_suffix('.perf').name}  "
              "(written when this perf_log block exits)")
    plog.info(f"bands:   output-nacl-sto3g-dft-bands.png")
    plog.info(f"cube:    output-nacl-sto3g-dft-homo.cube")
