import sys; sys.path.insert(0, "tests")
from test_gfn2_xtb import _ADENINE_ANGSTROM, load_gfn2_params
from vibeqc.molecule import Molecule, Atom
import vibeqc._vibeqc_core.semiempirical.xtb as _xtb
BOHR = 1.0/0.529177210903
mol = Molecule([Atom(z,[x*BOHR,y*BOHR,w*BOHR]) for z,x,y,w in _ADENINE_ANGSTROM],0,1)
params = load_gfn2_params()
print(f"{'max_iter':>8} {'E (Ha)':>14} {'free_E':>14} {'entropy':>11} {'smear_T':>9} {'accepted_T':>10}")
for mi in (2499, 4000):
    o=_xtb.XTBSccOptions(); o.max_iter=mi; o.conv_tol_charge=1e-8
    o.aes_faithful=False; o.auto_stabilize=True; o.aes_damping=0.5
    r=_xtb.run_gfn2_xtb(mol,params,o)
    acc = r.attempts[r.selected_attempt_index].electronic_temperature
    print(f"{mi:>8} {r.energy:>14.7f} {r.free_energy:>14.7f} {r.entropy:>11.3e} "
          f"{r.smearing_temperature:>9.4f} {acc:>10.4f}")
