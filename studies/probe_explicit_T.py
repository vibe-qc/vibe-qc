"""#3 part 2: does the ladder honour an explicitly requested temperature?

An unreachable conv_tol forces every rung to run, so the attempt ledger shows
which temperatures the ladder used -- without needing a system that stalls.
"""
import sys; sys.path.insert(0, "tests")
from test_gfn2_xtb import _ADENINE_ANGSTROM, load_gfn2_params
from vibeqc.molecule import Molecule, Atom
import vibeqc._vibeqc_core.semiempirical.xtb as _xtb
BOHR = 1.0/0.529177210903
mol = Molecule([Atom(z,[x*BOHR,y*BOHR,w*BOHR]) for z,x,y,w in _ADENINE_ANGSTROM],0,1)
params = load_gfn2_params()

def run(explicit_T=None, tol=1e-15, max_iter=3000):
    o=_xtb.XTBSccOptions(); o.max_iter=max_iter; o.conv_tol_charge=tol
    o.aes_faithful=False; o.aes_damping=0.5; o.auto_stabilize=True
    if explicit_T is not None:
        o.electronic_temperature = explicit_T      # marks the explicit flag
    r=_xtb.run_gfn2_xtb(mol,params,o)
    return o, r

for label, T in [("implicit (default)", None),
                 ("explicit T=0.0", 0.0),
                 ("explicit T=0.005", 0.005)]:
    o, r = run(T)
    temps = [a.electronic_temperature for a in r.attempts]
    iters = [a.n_iter for a in r.attempts]
    print(f"{label:>20} explicit={o.electronic_temperature_explicit!s:>5} "
          f"conv={r.converged!s:>5} attempts={len(r.attempts)} "
          f"smear_T={r.smearing_temperature} temps={temps} iters={iters}")
