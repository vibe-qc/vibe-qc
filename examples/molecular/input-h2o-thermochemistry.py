"""Water, harmonic frequencies and RRHO thermochemistry at 298.15 K and 500 K.

System: H2O at RHF/6-31G*, first relaxed with ``vibeqc.optimize_molecule``
so the Hessian is taken at a true minimum.
Method: ``run_job(method="rhf", basis="6-31g*", hessian=True,
thermo_options=ThermoOptions(symmetry_number=2))``. ``hessian=True`` builds
the Hessian by central finite differences of the analytic RHF gradient
(6N = 18 displacements), prints the harmonic frequencies with IR
intensities, and appends an RRHO ideal-gas thermochemistry block at
298.15 K and 1 atm to the .out. C2v water has rotational symmetry number 2,
which must be given by hand (the default is 1).

The returned RHFResult is a C++ object and carries no Hessian or thermo
attributes; those numbers live in the .out (and the QVF). To have them in
Python this script calls the same engine directly, ``vibeqc.compute_hessian_fd``
and ``vibeqc.compute_thermochemistry``, at 298.15 K and again at 500 K with
``ThermoOptions(temperature=500.0, symmetry_number=2)``.

Files (next to this script):
    input-h2o-thermochemistry.out      SCF, frequency table, thermochemistry block
    input-h2o-thermochemistry.molden   RHF orbitals
    input-h2o-thermochemistry.system   provenance manifest
    plus .xyz / .population.* / .qvf sidecars

Success: harmonic frequencies 1826.6, 4055.3 and 4173.3 cm^-1 (no
imaginary mode); ZPE = 0.022907 Ha (14.375 kcal/mol); at 298.15 K the
thermal corrections are H_corr = 0.026685 Ha and G_corr = 0.005309 Ha with
S = 71.70 microHa/K (44.99 cal/mol/K); at 500 K H_corr = 0.029285 Ha,
S = 78.34 microHa/K and G_corr = -0.009887 Ha. The library values must
match the .out block digit for digit at 298.15 K. Tolerance: 0.1 cm^-1 on
the frequencies and 1e-6 Ha on the corrections (vibe-qc 0.17.x development
build, Apple M5 Max, about 15 s).
"""

from pathlib import Path

import numpy as np

import vibeqc as vq

HERE = Path(__file__).parent
STEM = Path(__file__).stem
KCAL = 627.5094740631
CAL_PER_MOL_K = 627.5094740631 * 1000.0      # Ha/K -> cal/(mol K)

start = vq.Molecule([
    vq.Atom(8, [0.000,  0.000,  0.000]),
    vq.Atom(1, [0.000,  1.499, -1.160]),
    vq.Atom(1, [0.000, -1.499, -1.160]),
])
BASIS = "6-31g*"
SIGMA = 2                                   # C2v rotational symmetry number

opt = vq.optimize_molecule(start, BASIS, method="rhf")
mol = opt.system
print(f"\n  relaxed RHF/{BASIS}: E = {opt.energy:.8f} Ha, converged={opt.converged}")

result = vq.run_job(
    mol,
    basis=BASIS,
    method="rhf",
    hessian=True,
    thermo_options=vq.ThermoOptions(symmetry_number=SIGMA),
    output=str(HERE / STEM),
)
e_el = float(result.energy)

# Echo the block the runner wrote.
out_text = (HERE / f"{STEM}.out").read_text(encoding="utf-8")
block = out_text.split("## Vibrational Frequencies", 1)[1].split("Timings", 1)[0]
print("\n  --- from the .out ---")
print("\n".join("  " + ln for ln in ("## Vibrational Frequencies" + block).rstrip().splitlines()))

# The same numbers through the library, then 500 K.
hess = vq.compute_hessian_fd(mol, BASIS, method="RHF")
freqs = hess.frequencies_cm1[-(3 * len(list(mol.atoms)) - 6):]
print("\n  --- from compute_hessian_fd / compute_thermochemistry ---")
print(f"  frequencies (cm^-1): {np.round(freqs, 1).tolist()}; "
      f"imaginary modes: {hess.imaginary_count}; displacements: {hess.n_displacements}")
for temperature in (298.15, 500.0):
    th = vq.compute_thermochemistry(
        mol, hess, options=vq.ThermoOptions(temperature=temperature, symmetry_number=SIGMA))
    print(f"\n  T = {temperature:.2f} K, p = {th.pressure:.0f} Pa, rotor = {th.rotor_type}, "
          f"imaginary modes excluded = {th.n_imaginary_modes_excluded}")
    print(f"    ZPE            = {th.zpe:14.8f} Ha  ({th.zpe * KCAL:7.3f} kcal/mol)")
    print(f"    H_corr         = {th.h_thermal:14.8f} Ha    H = {e_el + th.h_thermal:.8f} Ha")
    print(f"    S_total        = {th.s_total * 1e6:14.4f} microHa/K  "
          f"({th.s_total * CAL_PER_MOL_K:6.3f} cal/mol/K)")
    print(f"    G_corr         = {th.g_thermal:14.8f} Ha    G = {e_el + th.g_thermal:.8f} Ha")
    print(f"    Cv_total       = {th.cv_total * CAL_PER_MOL_K:14.3f} cal/mol/K")
