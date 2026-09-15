"""NH₃ umbrella inversion via climbing-image Nudged Elastic Band.

Run:
    .venv/bin/python input-nh3-umbrella-neb.py

Produces:
    output-nh3-umbrella-neb.traj   — full reaction path; `ase gui` animates it
    output-nh3-umbrella-neb.qvf    — vibe-view reaction-path archive
    output-nh3-umbrella-neb.out    — text summary + MEP energies + TS geometry

The ammonia molecule inverts by flipping its three hydrogens through a
planar (D₃h) transition state. This script:

  1. Builds pyramidal NH₃ in two enantiomers (H's below vs. above N).
  2. Relaxes both endpoints with vibe-qc's native optimiser.
  3. Runs vibe-qc's native climbing-image NEB (``vq.run_neb``) to
     locate the TS.
  4. Writes the MEP energies, the TS geometry, the trajectory, and a
     vibe-view QVF archive.

The whole calculation is vibe-qc's own code — ``run_neb`` does the
improved-tangent band relaxation with a quick-min outer loop. ASE is
used only to write the ``.traj`` output that ``ase gui`` and the
tutorial's plotting scripts consume (structure file I/O, not a QC
engine; see CLAUDE.md § 10).

The experimental inversion barrier is ~5.8 kcal/mol. HF/STO-3G
over-estimates at ~11 kcal/mol; this is the usual HF-with-tiny-basis
overshoot and is a known qualitative-only limit. For quantitative
agreement use MP2 or a hybrid DFT functional in a def2-TZVP basis.
"""

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.data import chemical_symbols
from ase.io import write as ase_write

import vibeqc as vq

HERE = Path(__file__).parent
TRAJ_OUT = HERE / "output-nh3-umbrella-neb.traj"
QVF_OUT = HERE / "output-nh3-umbrella-neb"          # .qvf appended by writer
TEXT_OUT = HERE / "output-nh3-umbrella-neb.out"

ANG_TO_BOHR = 1.8897259886       # Å → bohr (vibe-qc coordinates are bohr)
BOHR_TO_ANG = 0.52917721067
EV_PER_HARTREE = 27.211386245988
KCAL_PER_EV = 23.0605


# --- geometry builders ----------------------------------------------------

def build_nh3(flip: bool = False) -> vq.Molecule:
    """Pyramidal NH₃ (vibe-qc Molecule, bohr) — H's below (``flip=False``)
    or above the nitrogen."""
    theta = np.deg2rad(106.67)              # HNH angle, experimental
    r_nh = 1.012 * ANG_TO_BOHR              # N-H bond length, experimental

    sin_alpha = np.sqrt(2.0 * (1.0 - np.cos(theta)) / 3.0)
    cos_alpha = np.sqrt(1.0 - sin_alpha ** 2)
    z_sign = +1 if flip else -1

    atoms = [vq.Atom(7, [0.0, 0.0, 0.0])]   # nitrogen at origin
    for k in range(3):
        phi = k * 2.0 * np.pi / 3.0
        atoms.append(vq.Atom(1, [
            r_nh * sin_alpha * np.cos(phi),
            r_nh * sin_alpha * np.sin(phi),
            z_sign * r_nh * cos_alpha,
        ]))
    return vq.Molecule(atoms, 0, 1)         # neutral singlet


def to_ase(system: vq.Molecule, energy_hartree: float) -> Atoms:
    """vibe-qc Molecule (+ energy) → ASE Atoms for trajectory output.

    Positions bohr → Å; energy Hartree → eV, attached via a
    SinglePointCalculator so ``ase gui`` and the plotting scripts can
    read ``get_potential_energy()`` back off the trajectory.
    """
    numbers = [int(a.Z) for a in system.atoms]
    symbols = "".join(chemical_symbols[z] for z in numbers)
    positions = np.array([list(a.xyz) for a in system.atoms]) * BOHR_TO_ANG
    atoms = Atoms(symbols, positions=positions)
    atoms.calc = SinglePointCalculator(
        atoms, energy=energy_hartree * EV_PER_HARTREE
    )
    return atoms


# --- run ------------------------------------------------------------------

# 1. Relaxed endpoints (vibe-qc native optimiser).
reactant = vq.optimize_molecule(build_nh3(flip=False), "sto-3g", method="rhf").system
product = vq.optimize_molecule(build_nh3(flip=True), "sto-3g", method="rhf").system

# 2. Native climbing-image NEB. 5 interior images + the two endpoints;
#    linear interpolation across the band (the path is a smooth flip, no
#    bonds break, so IDPP isn't needed); the highest-energy image climbs
#    to the saddle.
result = vq.run_neb(
    reactant, product,
    basis="sto-3g",
    n_images=5,
    method="RHF",
    interpolation="linear",
    climbing_image=True,
    climbing_image_start_fraction=0.1,  # promote the climber early: this
                                        # small symmetric band converges in
                                        # ~56 iters, sooner than the default
                                        # 0.3·max_iter warm-up would engage.
    conv_tol_force=1e-3,        # ≈ 0.05 eV/Å, the Kolsbjerg 2016 default
    max_iter=200,
    n_jobs=1,                   # serial; the 7-image band is tiny. For
                                # larger systems n_jobs=-1 parallelises the
                                # per-image SCFs across cores.
)

images = result.path.images
energies_ha = np.asarray(result.energies)
e_initial = energies_ha[0]
ts_index = result.transition_state_index

# 3. Trajectory (ASE I/O) + QVF archive (native vibe-view output).
ase_write(str(TRAJ_OUT), [to_ase(img.system, e) for img, e in zip(images, energies_ha)])
qvf_path = result.write_qvf(str(QVF_OUT))

# 4. Text report.
with open(TEXT_OUT, "w") as f:
    f.write("NH3 umbrella inversion via native CI-NEB @ HF/STO-3G\n")
    f.write("=" * 52 + "\n\n")
    f.write(f"converged: {result.converged}   outer iterations: {result.n_iter}\n")
    f.write(f"max |F_NEB|: {result.max_force:.2e} Ha/bohr\n\n")
    f.write("Minimum energy path:\n")
    f.write(f"  {'image':>6s}  {'ΔE (eV)':>10s}  {'ΔE (kcal/mol)':>14s}\n")
    for i, e in enumerate(energies_ha):
        de_ha = e - e_initial
        de_ev = de_ha * EV_PER_HARTREE
        f.write(f"  {i:6d}  {de_ev:10.4f}  {de_ev * KCAL_PER_EV:14.3f}\n")

    barrier_ha = energies_ha[ts_index] - e_initial
    barrier_kcal = barrier_ha * EV_PER_HARTREE * KCAL_PER_EV

    f.write("\n")
    f.write(f"Transition state: image {ts_index}"
            f"  (climbing image {result.path.climbing_image_index})\n")
    f.write(f"Barrier: {barrier_ha:.4f} Ha  ({barrier_kcal:.2f} kcal/mol)\n")
    f.write("Experimental barrier: ~5.8 kcal/mol\n")
    f.write("HF/STO-3G over-estimates by ~2× — expected for this level of theory.\n\n")

    f.write(f"TS geometry (image {ts_index}, Å):\n")
    for a in images[ts_index].system.atoms:
        p = np.array(list(a.xyz)) * BOHR_TO_ANG
        f.write(f"  {chemical_symbols[int(a.Z)]:<2s}  {p[0]:+.5f}  {p[1]:+.5f}  {p[2]:+.5f}\n")

print(f"Wrote {TRAJ_OUT.name},  {Path(qvf_path).name},  and  {TEXT_OUT.name}")
