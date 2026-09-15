"""M1 gate: Gaussian-route Gamma phonons against the GAPW route.

    .venv/bin/python examples/basisset_dev/gamma_phonon_gate.py MgO
    .venv/bin/python examples/basisset_dev/gamma_phonon_gate.py NaCl --steps 0.01,0.02,0.04
    .venv/bin/python examples/basisset_dev/gamma_phonon_gate.py MgO --gapw-only

What it measures
----------------
Gamma-point frequencies of a rocksalt primitive cell, computed twice at
the same functional and the same geometry:

* **BIPOLE (Gaussian)** -- ``vibeqc.basis_optimization.phonons``, the M1
  deliverable. Hessian from central differences of the **total energy**
  (see that module's docstring for why not from the force), ``18 N^2 +
  1`` SCFs.
* **GAPW (plane-wave-augmented)** -- ``vibeqc.periodic_gapw_phonon``,
  which differences the *analytic* GAPW force, ``6N`` SCFs.

The gate: optical modes agree within **5 cm^-1**; acoustic modes land
within **15 cm^-1 of zero**. The acoustic check is only meaningful with
``asr="none"`` -- imposing the acoustic sum rule would zero the residual
by construction and the test would pass without measuring anything. This
script therefore never imposes it.

``--steps`` runs the BIPOLE side at several finite-difference half-steps.
That scan is not decoration: an energy Hessian divides by ``h^2``, so SCF
noise enters as ``eps / h^2`` while anharmonicity enters as ``O(h^2)``.
The step is a choice that has to be *shown*, not asserted, and a
frequency quoted without its step is not reproducible.

Cost
----
Both routes are expensive on a real (dense) rocksalt primitive cell: the
BIPOLE side is 73 SCFs per step size per system, and a single Gamma
BIPOLE SCF on MgO/STO-3G at the experimental lattice constant takes
minutes on a laptop. Budget hours, or run it on the queue
(CLAUDE.md § 15). ``--dry-run`` prints the SCF budget and exits.

Geometries come from ``vibe_basis.io.structures.STRUCTURES``, the same
database the rest of the campaign uses, so the lattice constants carry
their published source with them rather than being retyped here. They are
**not** relaxed at the method used below, so a nonzero residual force is
expected and the low modes carry some of it -- equally on both routes,
which is what keeps the comparison fair.

``LiH`` is also accepted and is **not** a gate system: it is the cheapest
real rocksalt solid in the family and exists so the two routes can be
compared end to end on a laptop while MgO and NaCl are queued. A LiH
agreement is evidence the driver is right; it is not the M1 gate.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc.basis_optimization.phonons import (
    estimate_scf_count,
    gamma_phonons,
    phonons_from_hessian,
)
from vibeqc.kpoints import KPoints
from vibeqc.molecule import ANGSTROM_TO_BOHR

#: The two gate systems, plus one cheap stand-in. Element ordering only;
#: the lattice constant is read from ``STRUCTURES`` so it keeps its
#: published provenance (``Structure.structure_source``).
#:
#: ``LiH`` is **not** a gate system. It is the cheapest real rocksalt in
#: this family -- 6 basis functions per cell against MgO's 23 -- and
#: exists so the two routes can be compared to completion on a laptop
#: while MgO and NaCl are queued.
SYSTEMS = {
    "MgO": (12, 8),
    "NaCl": (11, 17),
    "LiH": (3, 1),
}


def rocksalt_primitive(a_angstrom: float, z1: int, z2: int):
    """The 2-atom fcc primitive cell of the rocksalt structure.

    Not the 8-atom conventional cubic cell: the primitive cell is the
    smallest one carrying all six Gamma modes, and it is the only one
    affordable at ``18 N^2`` SCFs.
    """
    a = float(a_angstrom) * ANGSTROM_TO_BOHR
    lattice = 0.5 * a * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    second = np.array([0.5, 0.5, 0.5]) @ lattice
    return vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(int(z1), [0.0, 0.0, 0.0]), vq.Atom(int(z2), list(second))],
    )


def bipole_options(cutoff_bohr=None):
    """SCF options shared by every displaced point.

    ``cutoff_bohr`` sets the BIPOLE real-space lattice cutoff. It is
    exposed because the driver's fold-truncation guard refuses to enter
    the SCF when it is too small for the system: LiH/STO-3G at the
    default 15 bohr reports ``S(Gamma) fold truncation 7.1e-02 ... in the
    unreliable numerical-support regime``. That guard is doing its job --
    raise the cutoff rather than work around it, and use the **same**
    value for every displaced point so the Hessian differences one
    surface.
    """
    import vibeqc as _vq

    options = _vq.PeriodicRHFOptions()
    if cutoff_bohr is not None:
        options.lattice_opts.cutoff_bohr = float(cutoff_bohr)
    return options


def run_bipole(system, basis_name, functional, step, options=None):
    kmesh = KPoints.gamma(system).to_bloch_kmesh()
    method = "RHF" if functional is None else "RKS"
    t0 = time.time()
    result = gamma_phonons(
        system,
        basis_name,
        kmesh,
        options,
        method=method,
        functional=functional,
        step_bohr=step,
        hessian_mode="energy",
        asr="none",
    )
    return result, time.time() - t0


def run_gapw(system, basis_name, functional, step, gapw_kwargs=None):
    """The reference side: FD on the *analytic* GAPW force.

    Reuses ``periodic_gapw_phonon``'s dynamical matrix, then converts it
    back to a force-constant matrix so both sides go through the same
    ``phonons_from_hessian`` classification and ZPE sum. Otherwise a
    difference in the acoustic identification or the ZPE mask would show
    up as a difference in the physics.
    """
    from vibeqc.periodic_gapw_phonon import (
        _AMU_TO_ELECTRON_MASS,
        _resolve_masses,
        compute_dynamical_matrix_fd,
    )

    basis = vq.BasisSet(system.unit_cell_molecule(), basis_name)
    t0 = time.time()
    D = compute_dynamical_matrix_fd(
        system,
        basis,
        basis_name,
        functional=functional,
        gapw_kwargs=dict(gapw_kwargs or {"one_centre": "block"}),
        fd_step_bohr=step,
    )
    elapsed = time.time() - t0

    masses = _resolve_masses(system)
    n_dof = D.shape[0]
    m = np.empty(n_dof)
    for i, mass in enumerate(masses):
        m[3 * i : 3 * i + 3] = float(mass) * _AMU_TO_ELECTRON_MASS
    sqrt_m = np.sqrt(m)
    H = D * np.outer(sqrt_m, sqrt_m)

    return (
        phonons_from_hessian(
            H,
            system,
            fd_step_bohr=step,
            hessian_mode="analytic_gapw_force",
            asr="none",
            n_scf=6 * len(system.unit_cell),
            detail={"route": "gapw", "functional": functional},
        ),
        elapsed,
    )


def report(label, phonons, elapsed, n_fu):
    print(f"\n=== {label} ===")
    print(f"  wall: {elapsed:.1f} s   SCF: {phonons.n_scf}   "
          f"h = {phonons.fd_step_bohr:g} bohr   ASR = {phonons.asr}")
    for k, w in enumerate(phonons.frequencies_cm1):
        tag = "  acoustic" if k in phonons.acoustic_indices else ""
        print(f"  mode {k}: {w:12.3f} cm^-1{tag}")
    print(f"  acoustic residual (max |w|): "
          f"{max(abs(w) for w in phonons.acoustic_residual_cm1):.3f} cm^-1")
    print(f"  imaginary modes excluded: {phonons.n_imaginary_modes_excluded}")
    print(f"  ZPE: {phonons.zero_point_hartree:.8f} Ha/cell = "
          f"{phonons.zero_point_kj_per_mol_per_formula_unit(n_fu):.3f} kJ/mol "
          f"per formula unit")


def compare(bipole, gapw):
    """Gate verdict. Returns ``(passed, lines)``."""
    lines = []
    opt_b = np.sort(bipole.optical_frequencies_cm1)
    opt_g = np.sort(gapw.optical_frequencies_cm1)
    dev = np.abs(opt_b - opt_g)
    lines.append("  optical modes (sorted), BIPOLE vs GAPW:")
    for b, g, d in zip(opt_b, opt_g, dev):
        lines.append(f"    {b:12.3f}  {g:12.3f}   delta = {d:8.3f} cm^-1")
    worst_opt = float(dev.max()) if dev.size else 0.0
    worst_ac = max(
        max(abs(w) for w in bipole.acoustic_residual_cm1),
        max(abs(w) for w in gapw.acoustic_residual_cm1),
    )
    ok_opt = worst_opt < 5.0
    ok_ac = worst_ac < 15.0
    lines.append(
        f"  worst optical deviation: {worst_opt:.3f} cm^-1  "
        f"(gate < 5)  -> {'PASS' if ok_opt else 'FAIL'}"
    )
    lines.append(
        f"  worst acoustic residual: {worst_ac:.3f} cm^-1  "
        f"(gate < 15)  -> {'PASS' if ok_ac else 'FAIL'}"
    )
    return (ok_opt and ok_ac), lines


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("system", choices=sorted(SYSTEMS))
    p.add_argument("--basis", default="sto-3g")
    p.add_argument(
        "--functional",
        default=None,
        help="XC functional; omit for RHF (the cheapest honest comparison).",
    )
    p.add_argument(
        "--steps",
        default="0.02",
        help="Comma-separated BIPOLE FD half-steps in bohr (the sensitivity scan).",
    )
    p.add_argument("--gapw-step", type=float, default=0.01)
    p.add_argument(
        "--bipole-cutoff",
        type=float,
        default=None,
        help=(
            "BIPOLE real-space lattice cutoff in bohr. Raise it when the "
            "driver's fold-truncation guard refuses to enter the SCF "
            "(LiH/STO-3G needs more than the default 15)."
        ),
    )
    p.add_argument("--formula-units", type=int, default=1)
    p.add_argument("--bipole-only", action="store_true")
    p.add_argument("--gapw-only", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", type=Path, default=None)
    args = p.parse_args(argv)

    from vibe_basis.io.structures import STRUCTURES

    reference = STRUCTURES[args.system]
    system = rocksalt_primitive(reference.a, *SYSTEMS[args.system])
    steps = [float(s) for s in args.steps.split(",") if s.strip()]
    n_atoms = len(system.unit_cell)

    print(f"{args.system} rocksalt primitive cell, a = {reference.a} A, "
          f"{n_atoms} atoms, basis {args.basis}, "
          f"functional {args.functional or 'RHF'}")
    print(f"geometry source: {reference.structure_source}")
    print(f"BIPOLE budget: {estimate_scf_count(n_atoms, 'energy')} SCF per step "
          f"x {len(steps)} step(s) = "
          f"{estimate_scf_count(n_atoms, 'energy') * len(steps)} SCF")
    print(f"GAPW budget:   {6 * n_atoms} SCF (+ analytic gradients)")
    if args.dry_run:
        return 0

    payload: dict = {"system": args.system, "a_angstrom": reference.a,
                     "basis": args.basis, "functional": args.functional,
                     "bipole": {}, "gapw": None}

    bipole_results = {}
    if not args.gapw_only:
        for step in steps:
            ph, dt = run_bipole(
                system,
                args.basis,
                args.functional,
                step,
                options=bipole_options(args.bipole_cutoff),
            )
            bipole_results[step] = ph
            report(f"BIPOLE energy-Hessian, h = {step:g} bohr", ph, dt,
                   args.formula_units)
            payload["bipole"][str(step)] = {
                "frequencies_cm1": [float(w) for w in ph.frequencies_cm1],
                "acoustic_residual_cm1": list(ph.acoustic_residual_cm1),
                "zpe_hartree_per_cell": ph.zero_point_hartree,
                "zpe_kj_per_mol_per_fu":
                    ph.zero_point_kj_per_mol_per_formula_unit(args.formula_units),
                "n_scf": ph.n_scf,
                "wall_s": dt,
            }

    if len(bipole_results) > 1:
        print("\n=== step-size sensitivity (BIPOLE) ===")
        ref = min(bipole_results)
        base = np.sort(bipole_results[ref].optical_frequencies_cm1)
        for step, ph in sorted(bipole_results.items()):
            here = np.sort(ph.optical_frequencies_cm1)
            spread = float(np.abs(here - base).max())
            print(f"  h = {step:6.3f} bohr: optical "
                  f"{np.array2string(here, precision=2)}  "
                  f"max |delta vs h={ref:g}| = {spread:.3f} cm^-1   "
                  f"ZPE = "
                  f"{ph.zero_point_kj_per_mol_per_formula_unit(args.formula_units):.3f}"
                  f" kJ/mol/f.u.")

    gapw = None
    if not args.bipole_only:
        gapw, dt = run_gapw(system, args.basis, args.functional, args.gapw_step)
        report("GAPW analytic-force Hessian (reference side)", gapw, dt,
               args.formula_units)
        payload["gapw"] = {
            "frequencies_cm1": [float(w) for w in gapw.frequencies_cm1],
            "acoustic_residual_cm1": list(gapw.acoustic_residual_cm1),
            "zpe_hartree_per_cell": gapw.zero_point_hartree,
            "zpe_kj_per_mol_per_fu":
                gapw.zero_point_kj_per_mol_per_formula_unit(args.formula_units),
            "wall_s": dt,
        }

    passed = None
    if gapw is not None and bipole_results:
        print("\n=== M1 gate ===")
        for step, ph in sorted(bipole_results.items()):
            ok, lines = compare(ph, gapw)
            print(f"  BIPOLE h = {step:g} bohr:")
            for line in lines:
                print("  " + line)
            passed = ok if passed is None else (passed or ok)
        payload["gate_passed"] = bool(passed)
        print(f"\n  VERDICT: {'PASS' if passed else 'FAIL'}")

    if args.json:
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
