"""#3 probe: is the adenine basin selected by the auto_stabilize 700-iteration cap?

use_stabilization = has_gam3 && auto_stabilize && total_max_iter >= 2500
primary_max_iter  = use_stabilization ? min(700, total_max_iter) : total_max_iter

So max_iter=2499 and max_iter=2500 differ ONLY in whether the primary attempt
is cut off at 700 iterations.  If the basin flips across that boundary, the
answer is decided by the cutoff, not by any stabilisation physics.
"""
import sys
sys.path.insert(0, "tests")
from test_gfn2_xtb import _ADENINE_ANGSTROM, load_gfn2_params
from vibeqc.molecule import Molecule, Atom
import vibeqc._vibeqc_core.semiempirical.xtb as _xtb

BOHR = 1.0 / 0.529177210903
mol = Molecule([Atom(z, [x*BOHR, y*BOHR, w*BOHR]) for z, x, y, w in _ADENINE_ANGSTROM], 0, 1)
params = load_gfn2_params()

def run(max_iter, stabilize):
    o = _xtb.XTBSccOptions()
    o.max_iter = max_iter
    o.conv_tol_charge = 1.0e-8
    o.aes_faithful = False
    o.auto_stabilize = stabilize
    o.aes_damping = 0.5
    r = _xtb.run_gfn2_xtb(mol, params, o)
    return r

print(f"{'max_iter':>9} {'stab':>6} {'conv':>5} {'n_iter':>7} {'energy (Ha)':>16}  {'vs xtb (mHa)':>13}")
XTB = -27.49957548
for max_iter, stab in [(2499, True), (2500, True), (4000, True), (4000, False), (2499, False)]:
    r = run(max_iter, stab)
    print(f"{max_iter:>9} {str(stab):>6} {str(r.converged):>5} {r.n_iter:>7} "
          f"{r.energy:>16.7f}  {(r.energy-XTB)*1e3:>+13.3f}")
