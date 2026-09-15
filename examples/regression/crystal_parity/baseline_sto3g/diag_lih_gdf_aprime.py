"""LiH GDF Option A' validation — live SCF trace.

Option A' = Steps 1+2+3' gauge fix: V_ne/e_nuc Ewald (Step 1),
J via Ewald-3D (Step 2), K via full-range real-space (Step 3',
replacing the broken native Lpq-K).  Target: converge to the
EWALD_3D reference -29.33212849 Ha (11 iter).

progress=True so the per-iteration SCF trace streams live to
vq's stdout.log — no more blind runs.
"""
import os, sys
os.environ.setdefault("PYSCF_TMPDIR", "/tmp/pyscf_vibeqc")
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import numpy as np
import vibeqc as vq

ANG2BOHR = 1.0 / 0.529177210903
a = 4.084 * ANG2BOHR
cation = [(0,0,0),(0,0.5,0.5),(0.5,0,0.5),(0.5,0.5,0)]
anion = [(0.5,0.5,0.5),(0.5,0,0),(0,0.5,0),(0,0,0.5)]
atoms = [vq.Atom(3,[fx*a,fy*a,fz*a]) for fx,fy,fz in cation]
atoms += [vq.Atom(1,[fx*a,fy*a,fz*a]) for fx,fy,fz in anion]
system = vq.PeriodicSystem(3, np.diag([a,a,a]), atoms)
basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
print(f"LiH 8-atom conv: {basis.nbasis} BFs, {system.n_electrons()} elec", flush=True)

opts = vq.PeriodicRHFOptions()
opts.use_diis = True
opts.damping = 0.0
opts.max_iter = 30
opts.conv_tol_energy = 1e-8

r = vq.run_rhf_periodic_gamma_gdf(system, basis, opts, progress=True, verbose=1)
print(flush=True)
print(f"RESULT  E_total={r.energy:.8f}  conv={r.converged}  n_iter={r.n_iter}", flush=True)
print(f"        e_nuclear={r.e_nuclear:.6f}  e_coulomb={r.e_coulomb:.6f}  "
      f"e_hf_exchange={r.e_hf_exchange:.6f}", flush=True)
print(f"        EWALD_3D ref -29.33212849   dE={r.energy-(-29.33212849):+.6f} Ha", flush=True)
