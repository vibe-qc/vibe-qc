"""#3 post-fix: the basin sweep, the monotonicity boundary, and the ledger."""
import sys; sys.path.insert(0, "tests")
from test_gfn2_xtb import _ADENINE_ANGSTROM, load_gfn2_params
from vibeqc.molecule import Molecule, Atom
import vibeqc._vibeqc_core.semiempirical.xtb as _xtb
BOHR = 1.0/0.529177210903
mol = Molecule([Atom(z,[x*BOHR,y*BOHR,w*BOHR]) for z,x,y,w in _ADENINE_ANGSTROM],0,1)
params = load_gfn2_params()
XTB = -27.49957548

def run(max_iter, stab, faithful, damp):
    o=_xtb.XTBSccOptions(); o.max_iter=max_iter; o.conv_tol_charge=1e-8
    o.aes_faithful=faithful; o.auto_stabilize=stab; o.aes_damping=damp
    return _xtb.run_gfn2_xtb(mol,params,o)

print("=== monotonicity boundary (ad-hoc, auto_stabilize=True) ===")
for mi in (2499, 2500, 2501, 3000, 4000):
    r = run(mi, True, False, 0.5)
    acc = r.attempts[r.selected_attempt_index].electronic_temperature if r.selected_attempt_index >= 0 else None
    print(f"  max_iter={mi:>5} conv={str(r.converged):>5} n_iter={r.n_iter:>5} "
          f"E={r.energy:>14.7f}  T_acc={acc}  attempts={len(r.attempts)}")

print("\n=== the #3 four-way sweep (conv_tol 1e-8, max_iter 4000) ===")
energies=set()
for faithful in (False, True):
    for stab in (True, False):
        r = run(4000, stab, faithful, 0.25 if faithful else 0.5)
        acc = r.attempts[r.selected_attempt_index].electronic_temperature if r.selected_attempt_index>=0 else None
        energies.add(round(r.energy,7))
        print(f"  faithful={str(faithful):>5} stab={str(stab):>5} conv={str(r.converged):>5} "
              f"n_iter={r.n_iter:>5} E={r.energy:>14.7f} T_acc={acc} "
              f"vs_xtb={(r.energy-XTB)*1e3:>+8.3f} mHa")
print(f"\n  distinct energies: {len(energies)}  spread: "
      f"{(max(energies)-min(energies))*1e3:.4f} mHa")
print(f"  sorted: {sorted(energies)}")
