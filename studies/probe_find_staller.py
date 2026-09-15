"""Find a molecular system whose T=0 primary attempt genuinely STALLS, so the
ladder is entered and the explicit-temperature contract can be pinned."""
import sys; sys.path.insert(0, "tests")
from test_gfn2_xtb import load_gfn2_params
from vibeqc.molecule import Molecule, Atom
import vibeqc._vibeqc_core.semiempirical.xtb as _xtb
params = load_gfn2_params()
A = 1.0/0.529177210903

def mol(spec, charge=0):
    return Molecule([Atom(z,[x*A,y*A,zz*A]) for z,x,y,zz in spec], charge, 1)

cands = {
    # stretched closed-shell systems -> collapsing frontier gap
    "H2 @3.0A":   [(1,0,0,0),(1,3.0,0,0)],
    "H2 @5.0A":   [(1,0,0,0),(1,5.0,0,0)],
    "H4 chain":   [(1,0,0,0),(1,1.0,0,0),(1,2.4,0,0),(1,3.4,0,0)],
    "H6 ring2.0": [(1,2.0,0,0),(1,1.0,1.732,0),(1,-1.0,1.732,0),
                   (1,-2.0,0,0),(1,-1.0,-1.732,0),(1,1.0,-1.732,0)],
    "Li4 square": [(3,0,0,0),(3,3.0,0,0),(3,3.0,3.0,0),(3,0,3.0,0)],
    "C2 @1.9A":   [(6,0,0,0),(6,1.9,0,0)],
}
print(f"{'system':>12} {'conv':>5} {'attempts':>9} {'iters':>16} {'temps':>18} {'smear_T':>8}")
for name, spec in cands.items():
    o=_xtb.XTBSccOptions(); o.max_iter=3000; o.conv_tol_charge=1e-8
    o.auto_stabilize=True
    try:
        r=_xtb.run_gfn2_xtb(mol(spec), params, o)
        temps=[a.electronic_temperature for a in r.attempts]
        iters=[a.n_iter for a in r.attempts]
        print(f"{name:>12} {r.converged!s:>5} {len(r.attempts):>9} {str(iters):>16} "
              f"{str(temps):>18} {r.smearing_temperature:>8}")
    except Exception as e:
        print(f"{name:>12} ERROR {str(e)[:60]}")
