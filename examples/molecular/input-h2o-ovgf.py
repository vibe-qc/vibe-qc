"""Water, second-order Green's function IPs and the renormalised HOMO value against Koopmans.

System: H2O/cc-pVDZ at the fixed geometry shared by these examples (bohr).
Method: ``run_job(method="ovgf", basis="cc-pvdz")`` runs RHF and then the
propagator route. What it computes needs stating exactly. ``result.ovgf``
holds the quasiparticle energies of the diagonal second-order self-energy
(GF2), from the Dyson equation omega = eps_p + Sigma_pp(omega) solved by
Newton iteration, for up to the three highest occupied orbitals and the
LUMO. The .out labels that table "OVGF / GF2" and prints the same numbers
as "OVGF ionization potential". Separately, when 2 n_bf <= 60 the .out
prints a renormalised HOMO ionisation potential that adds the diagonal
third-order self-energy through a per-channel geometric resummation (the
"GF2(renorm)" line); it is not stored in ``result.ovgf`` and is read back
from the .out here. This is not a literal implementation of the OVGF A/B/C
recipes; see docs/tutorial/ovgf_ionization.md.

Each ``QuasiparticleResult`` in ``result.ovgf`` has ``orbital`` (0-based MO
index), ``eps_scf`` (the Koopmans value), ``eps_qp`` (the quasiparticle
energy), ``sigma`` (the self-energy at the pole), ``pole_strength``,
``iterations`` and ``converged``, all energies in Hartree.

Files (next to this script):
    input-h2o-ovgf.out       RHF trace, GF2 quasiparticle table, IP/EA summary, renormalised HOMO IP
    input-h2o-ovgf.molden    RHF orbitals
    input-h2o-ovgf.system    provenance manifest
    plus .xyz and .population.* sidecars

Success: for the HOMO (MO 5, the 1b1 lone pair) Koopmans gives IP = 13.339
eV and GF2 IP = 10.750 eV with pole strength Z = 0.901; the renormalised
third-order value printed in the .out is 11.453 eV (Z = 0.920), against an
experimental vertical IP of 12.62 eV. The next two orbitals (MO 4, 3a1, and
MO 3, 1b2) shift from 15.187 to 13.195 eV and from 18.385 to 17.346 eV
with Z = 0.910 and 0.927. Bare GF2 overcorrects Koopmans in this basis and
the third-order resummation pulls it back toward experiment. Tolerance:
0.005 eV on each quasiparticle energy (vibe-qc 0.17.x development build,
Apple M5 Max, 10 to 40 s depending on machine load).
"""

import re
from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).parent
STEM = Path(__file__).stem
EV = 27.211386245988

mol = vq.Molecule([
    vq.Atom(8, [0.000,  0.000,  0.000]),
    vq.Atom(1, [0.000,  1.499, -1.160]),
    vq.Atom(1, [0.000, -1.499, -1.160]),
])
N_OCC = 5
EXPT_IP = {5: 12.62, 4: 14.74, 3: 18.55}       # vertical IPs of water (1-based MO number), eV

result = vq.run_job(mol, basis="cc-pvdz", method="ovgf", output=str(HERE / STEM))

print("\n  GF2 quasiparticle energies (result.ovgf), H2O/cc-pVDZ")
print(f"  {'MO':>4s} {'role':6s} {'Koopmans (eV)':>14s} {'GF2 (eV)':>10s} {'shift (eV)':>11s} "
      f"{'Sigma (eV)':>11s} {'pole Z':>7s} {'iters':>6s} {'expt IP (eV)':>13s}")
for q in result.ovgf:
    mo = q.orbital + 1                                     # 1-based, as in the .out
    role = "HOMO" if q.orbital == N_OCC - 1 else ("LUMO" if q.orbital == N_OCC else "")
    # IP = -eps for an occupied orbital; for the LUMO the same sign gives -EA.
    k, g = -q.eps_scf * EV, -q.eps_qp * EV
    expt = f"{EXPT_IP[mo]:13.2f}" if mo in EXPT_IP else f"{'':13s}"
    print(f"  {mo:4d} {role:6s} {k:14.3f} {g:10.3f} {g - k:11.3f} {q.sigma * EV:11.3f} "
          f"{q.pole_strength:7.3f} {q.iterations:6d} {expt}"
          + ("" if q.converged else "   (not converged)"))
print("  (the LUMO row lists -eps, i.e. minus the electron affinity)")

homo = next(q for q in result.ovgf if q.orbital == N_OCC - 1)
text = (HERE / f"{STEM}.out").read_text(encoding="utf-8")
match = re.search(r"GF2\(renorm\) ionization potential \(HOMO\) = ([\d.]+) eV\s+"
                  r"\(renormalized; pole Z=([\d.]+)\)", text)
print(f"\n  first IP: Koopmans {-homo.eps_scf * EV:.3f} eV, GF2 {-homo.eps_qp * EV:.3f} eV "
      f"(Z = {homo.pole_strength:.3f})", end="")
if match:
    print(f", renormalised third order {float(match.group(1)):.3f} eV (Z = {float(match.group(2)):.3f}); "
          f"experiment 12.62 eV")
else:
    print("; no renormalised line in the .out (it is printed only when 2 n_bf <= 60)")
