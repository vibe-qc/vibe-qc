"""#3: the attempt ledger for the shipped adenine path."""
import sys
sys.path.insert(0, "tests")
from test_gfn2_xtb import _ADENINE_ANGSTROM, load_gfn2_params
from vibeqc.molecule import Molecule, Atom
import vibeqc._vibeqc_core.semiempirical.xtb as _xtb

BOHR = 1.0 / 0.529177210903
mol = Molecule([Atom(z, [x*BOHR, y*BOHR, w*BOHR]) for z, x, y, w in _ADENINE_ANGSTROM], 0, 1)
params = load_gfn2_params()

for max_iter in (2499, 2500, 4000):
    o = _xtb.XTBSccOptions()
    o.max_iter = max_iter; o.conv_tol_charge = 1e-8
    o.aes_faithful = False; o.auto_stabilize = True; o.aes_damping = 0.5
    r = _xtb.run_gfn2_xtb(mol, params, o)
    print(f"\n=== max_iter={max_iter}  converged={r.converged}  E={r.energy:.7f}  "
          f"selected={r.selected_attempt_index} ===")
    for i, a in enumerate(r.attempts):
        mark = " <== ACCEPTED" if i == r.selected_attempt_index else ""
        print(f"  [{i}] {a.solver:<20} alloc={a.allocated_max_iter:>5} ran={a.n_iter:>5} "
              f"mix={a.ladder_charge_mixing:<6} T={a.electronic_temperature:<7} "
              f"{a.exit_reason}{mark}")
