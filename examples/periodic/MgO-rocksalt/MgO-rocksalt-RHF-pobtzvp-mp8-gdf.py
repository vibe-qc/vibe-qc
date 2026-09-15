"""MgO rocksalt primitive cell -- RHF / pob-TZVP / 8x8x8 native GDF target.

Standalone vibe-qc input for side-by-side CRYSTAL14 parity.

CRYSTAL14 partner:
    MgO-rocksalt-RHF-pobtzvp-mp8-crystal14.d12

The CRYSTAL input lists the Fm-3m asymmetric unit in the conventional
cell. CRYSTAL then works internally with the 2-atom FCC primitive cell.
This vibe-qc input builds that same primitive cell explicitly and runs
the 8x8x8 target on the native multi-k GDF backend end to end.

Run:
    .venv/bin/python examples/periodic/MgO-rocksalt/MgO-rocksalt-RHF-pobtzvp-mp8-gdf.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc.progress import ProgressLogger


ANG2BOHR = 1.0 / 0.529177210903
HARTREE_TO_EV = 27.211386245988

A_ANG = 4.21
A_BOHR = A_ANG * ANG2BOHR

OUT_STEM = Path(__file__).resolve().with_suffix("")


def build_mgo_primitive() -> vq.PeriodicSystem:
    """2-atom FCC primitive MgO cell used internally by CRYSTAL."""
    lattice = np.array([
        [0.0, A_BOHR / 2.0, A_BOHR / 2.0],
        [A_BOHR / 2.0, 0.0, A_BOHR / 2.0],
        [A_BOHR / 2.0, A_BOHR / 2.0, 0.0],
    ])
    o_xyz = lattice @ np.array([0.5, 0.5, 0.5])
    return vq.PeriodicSystem(3, lattice, [
        vq.Atom(12, [0.0, 0.0, 0.0]),
        vq.Atom(8, o_xyz.tolist()),
    ])


def make_options() -> vq.PeriodicRHFOptions:
    opts = vq.PeriodicRHFOptions()
    # CRYSTAL14 parity controls:
    #   FMIXING 30   -> 30% Fock-matrix mixing
    #   LEVSHIFT 6 0 -> 0.6 Ha level-shift warm-up, then unshifted
    #   TOLDEE 8     -> 1e-8 Ha energy convergence
    opts.conv_tol_energy = 1e-8
    opts.max_iter = 50
    opts.use_diis = True
    opts.damping = 0.0
    opts.fock_mixing = 0.30
    opts.level_shift = 0.60
    opts.initial_guess = vq.InitialGuess.SAD
    return opts


if __name__ == "__main__":
    t_start = time.perf_counter()
    plog = ProgressLogger(log_path=OUT_STEM.with_suffix(".out"), verbose=True)

    with vq.perf_log(OUT_STEM.with_suffix(".perf")):
        plog.banner("MgO rocksalt / primitive / RHF / pob-TZVP / native GDF target")

        system = build_mgo_primitive()
        vq.attach_symmetry(system, symprec=1e-4)
        basis = vq.BasisSet(system.unit_cell_molecule(), "pob-tzvp")

        plog.info(f"cell:       primitive FCC, a(conventional)={A_ANG} A")
        plog.info(f"atoms:      {len(system.unit_cell)} (Mg + O)")
        plog.info(
            f"spacegroup: {system.symmetry.international_symbol} "
            f"(SG {system.symmetry.number})"
        )
        plog.info(f"basis:      pob-TZVP, {basis.nbasis} bf per primitive cell")
        plog.info(
            "SCF:        CRYSTAL FMIXING 30 / LEVSHIFT 6 0 "
            "-> fock_mixing=0.30, level_shift=0.60 Ha, warm-up=5"
        )

        opts = make_options()
        kmesh = vq.KPoints.gamma_centred(system, [8, 8, 8], symmetry=True)
        plog.info(
            f"k-mesh:     Gamma-centred 8x8x8 -> {len(kmesh)} IBZ "
            "points (512 full)"
        )

        result = vq.run_krhf_periodic_gdf(
            system, basis, kmesh,
            options=opts,
            level_shift_warmup_cycles=5,
            progress=plog,
        )

        plog.info("")
        plog.info(f"E/cell           = {result.energy:.10f} Ha")
        plog.info(f"  E_electronic   = {result.e_electronic:.10f} Ha")
        plog.info(f"  E_nuclear      = {result.e_nuclear:.10f} Ha")
        plog.info(f"converged        = {result.converged}")
        plog.info(f"SCF iters        = {result.n_iter}")
        plog.info(f"expanded IBZ     = {result.expanded_from_ibz}")

        n_occ = system.n_electrons() // 2
        homo = max(float(eps[n_occ - 1]) for eps in result.mo_energies)
        lumo = min(float(eps[n_occ]) for eps in result.mo_energies)
        plog.info(
            f"HOMO max         = {homo * HARTREE_TO_EV:.3f} eV   "
            f"LUMO min         = {lumo * HARTREE_TO_EV:.3f} eV"
        )
        plog.info(
            f"Gap              = {(lumo - homo) * HARTREE_TO_EV:.3f} eV"
        )

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
        plog.info(f".perf:   {OUT_STEM.with_suffix('.perf').name}")
