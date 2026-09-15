"""#125: adenine's iteration count post-fix, across tolerances.

#125 records 514 iterations, 24.5x xtb (so xtb ~21).  The fix removed the
ladder's 2887-iteration path; the question is whether the T=0 solve itself is
still an outlier.
"""
import sys; sys.path.insert(0, "tests")
from test_gfn2_xtb import _ADENINE_ANGSTROM, load_gfn2_params, _water, _mol, _bug45_pyridine
from vibeqc.molecule import Molecule, Atom
import vibeqc._vibeqc_core.semiempirical.xtb as _xtb
BOHR = 1.0/0.529177210903
adenine = Molecule([Atom(z,[x*BOHR,y*BOHR,w*BOHR]) for z,x,y,w in _ADENINE_ANGSTROM],0,1)
params = load_gfn2_params()

print(f"{'system':>10} {'conv_tol':>10} {'iters':>7} {'attempts':>9} {'E (Ha)':>16}")
for name, mol in (("adenine", adenine), ("pyridine", _mol(_bug45_pyridine())), ("water", _mol(_water()))):
    for tol in (1e-6, 1e-7, 1e-8):
        o=_xtb.XTBSccOptions(); o.max_iter=4000; o.conv_tol_charge=tol
        o.aes_faithful=False; o.aes_damping=0.5; o.auto_stabilize=True
        r=_xtb.run_gfn2_xtb(mol,params,o)
        print(f"{name:>10} {tol:>10.0e} {r.n_iter:>7} {len(r.attempts):>9} {r.energy:>16.9f}")
    # default tolerance
    o=_xtb.XTBSccOptions(); o.max_iter=4000
    r=_xtb.run_gfn2_xtb(mol,params,o)
    print(f"{name:>10} {'default':>10} {r.n_iter:>7} {len(r.attempts):>9} {r.energy:>16.9f}")
