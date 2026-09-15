"""Li4 square stalls at T=0, so it reaches the ladder's finite-temperature
rung -- the system needed to pin the explicit-temperature contract (#3)."""
import sys; sys.path.insert(0, "tests")
from test_gfn2_xtb import load_gfn2_params
from vibeqc.molecule import Molecule, Atom
import vibeqc._vibeqc_core.semiempirical.xtb as _xtb
params = load_gfn2_params()
A = 1.0/0.529177210903
LI4 = [(3,0,0,0),(3,3.0,0,0),(3,3.0,3.0,0),(3,0,3.0,0)]
mol = Molecule([Atom(z,[x*A,y*A,zz*A]) for z,x,y,zz in LI4], 0, 1)

def run(explicit_T=None, max_iter=6000):
    o=_xtb.XTBSccOptions(); o.max_iter=max_iter; o.conv_tol_charge=1e-8
    o.auto_stabilize=True
    if explicit_T is not None:
        o.electronic_temperature = explicit_T
    r=_xtb.run_gfn2_xtb(mol,params,o)
    return o, r

for label, T in [("implicit (default)", None), ("explicit T=0.0", 0.0)]:
    for rep in (1, 2):                      # determinism check
        o, r = run(T)
        temps=[a.electronic_temperature for a in r.attempts]
        iters=[a.n_iter for a in r.attempts]
        print(f"{label:>20} rep{rep} explicit={o.electronic_temperature_explicit!s:>5} "
              f"conv={r.converged!s:>5} smear_T={r.smearing_temperature:<6} "
              f"E={r.energy:>14.9f} temps={temps} iters={iters}")
