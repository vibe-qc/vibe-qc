"""Water, TD-DFT excited states: TDA and full Casida on PBE0/def2-SVP, with NTOs.

System: H2O at the fixed geometry shared by these examples (bohr).
Method: RKS with the PBE0 hybrid in def2-SVP, then five singlet excitations
twice: once in the Tamm-Dancoff approximation (``tddft_type="tda"``) and
once with the full Casida response (``tddft_type="casida"``). Both runs ask
for natural transition orbitals (``nto=True``) and a Gaussian-broadened
UV/Vis stick spectrum (``tddft_spectrum=True``).

Files (next to this script):
    input-h2o-tddft-tda-nto-tda.out       SCF trace, TD-DFT table, UV/Vis spectrum block
    input-h2o-tddft-tda-nto-tda.qvf       QVF archive with the excited states and NTOs
    input-h2o-tddft-tda-nto-casida.*      the same pair for the Casida run
    plus the usual .molden / .system / .xyz / .population.* sidecars per run

The converged RKSResult is a C++ object that cannot carry extra Python
attributes, so ``result.tddft`` is not set; the runner writes the excited
states into the .out table and the QVF, and the NTOs into the QVF only. This
script therefore reads the two tables back from the .out files and prints
them side by side. To get the NTOs as numbers, take a ``TDDFTState`` from
``vibeqc.run_tddft_tda(mol, basis, mo_energies, mo_coeffs, n_occ, ...)`` and
call ``vibeqc.compute_nto(state, mo_coeffs, n_occ)``; that is a third
response solve, so it is not repeated here. In a separate check with the
same orbitals it gave a leading S1 hole/particle pair of weight 0.9999.

Known defect: at this development build the printed oscillator strengths are
over-scaled (about twice the spin-adapted value on a closed-shell reference,
from the transition-dipole normalisation), so compare them only relative to
each other. Excitation energies are unaffected.

Success: TDA S1 = 7.574 eV with f = 0.031, dominated by HOMO->LUMO (5->1,
weight 0.999); Casida S1 = 7.542 eV, 0.032 eV below TDA as the
de-excitation coupling requires, and Casida lies below TDA for all five
states (by 0.009 to 0.077 eV). The oscillator strengths move by 8 to 30
percent between the two (S1: 0.031 TDA, 0.027 Casida; S4: 0.155 and 0.202;
S5: 0.708 and 0.762). E(RKS/PBE0) = -76.273301 Ha. Tolerance: 0.01 eV on
the excitation energies.

Cost, and why this is the slowest example in the directory: a profile of
one TDA solve shows 99.9 percent of the time in the adiabatic XC kernel on
the DFT grid (``vibeqc.tddft._compute_alda_kernel_mo``), not in the
integral transformation or the diagonalisation. All timings here come from
a heavily loaded shared machine (load average 180 to 280 on 18 cores), so a
quiet machine will be faster by an amount not measured. Under that load one
TDA solve takes 4.5 s in STO-3G, 16 s in 6-31G, 81 s in 6-31G* and 150 s
in def2-SVP for water. This script, two complete SCF plus response jobs,
took 114 s wall time in its final run (jobs of 45 s and 59 s) and 108 to
150 s per job at higher load. Set ``BASIS = "6-31g"`` below for a much
cheaper run (S1 shifts to 7.68 eV in TDA). vibe-qc 0.17.x development
build, Apple M5 Max.
"""

from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).parent
STEM = Path(__file__).stem

mol = vq.Molecule([
    vq.Atom(8, [0.000,  0.000,  0.000]),
    vq.Atom(1, [0.000,  1.499, -1.160]),
    vq.Atom(1, [0.000, -1.499, -1.160]),
])
N_STATES = 5
FUNCTIONAL = "pbe0"
BASIS = "def2-svp"      # "6-31g" is several times cheaper; see the cost note above


def read_tddft_table(out_path: Path):
    """Return [(state, E_eV, lambda_nm, f_osc, dominant)] from the runner's .out table."""
    lines = out_path.read_text(encoding="utf-8").splitlines()
    start = next(i for i, ln in enumerate(lines) if "## TD-DFT excited states" in ln)
    rows = []
    for ln in lines[start + 4:]:                 # header, rule, column titles, dashes
        parts = ln.split()
        if len(parts) < 4 or not parts[0].isdigit():
            break
        rows.append((int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]),
                     " ".join(parts[4:])))
    return rows


scf = {}
tables = {}
for kind in ("tda", "casida"):
    scf[kind] = vq.run_job(
        mol,
        basis=BASIS,
        method="rks",
        functional=FUNCTIONAL,
        tddft=True,
        tddft_type=kind,
        tddft_n_states=N_STATES,
        nto=True,
        tddft_spectrum=True,
        output=str(HERE / f"{STEM}-{kind}"),
    )
    tables[kind] = read_tddft_table(HERE / f"{STEM}-{kind}.out")

print(f"\n  E(RKS/{FUNCTIONAL}/{BASIS}) = {scf['tda'].energy:16.8f} Ha")
print("\n  excited states read back from the two .out tables (dominant transitions are "
      "occupied->virtual orbital numbers, so 5->1 is HOMO->LUMO)")
print(f"  {'state':>5s} {'TDA E (eV)':>11s} {'f':>7s} {'Casida E (eV)':>14s} {'f':>7s} "
      f"{'shift (eV)':>10s}  dominant transitions (TDA)")
for (n, e_t, _l, f_t, dom), (_n, e_c, _lc, f_c, _d) in zip(tables["tda"], tables["casida"]):
    print(f"  {n:5d} {e_t:11.4f} {f_t:7.4f} {e_c:14.4f} {f_c:7.4f} {e_c - e_t:10.4f}  {dom}")
print("\n  natural transition orbitals and the UV/Vis spectrum were written to the .qvf files "
      "(open them in vibe-view); the spectrum block is also in each .out")
