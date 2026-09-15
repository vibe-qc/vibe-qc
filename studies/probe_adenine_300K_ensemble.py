"""#184: does vibe-qc at xtb's T=300 K agree with its own T=0 number?"""
import sys; sys.path.insert(0, "tests")
from test_gfn2_xtb import _ADENINE_ANGSTROM, load_gfn2_params
from vibeqc.molecule import Molecule, Atom
import vibeqc._vibeqc_core.semiempirical.xtb as _xtb
BOHR = 1.0/0.529177210903
KB_HA_PER_K = 3.166811563e-6
mol = Molecule([Atom(z,[x*BOHR,y*BOHR,w*BOHR]) for z,x,y,w in _ADENINE_ANGSTROM],0,1)
params = load_gfn2_params()
XTB = -27.49957548
print(f"{'T (K)':>8} {'T (Ha)':>12} {'E (Ha)':>16} {'free_E':>16} {'S':>10} {'vs xtb mHa':>11} {'iters':>6}")
for TK in (0.0, 300.0):
    o=_xtb.XTBSccOptions(); o.max_iter=4000; o.conv_tol_charge=1e-8
    o.aes_faithful=False; o.aes_damping=0.5; o.auto_stabilize=True
    o.electronic_temperature = TK * KB_HA_PER_K
    r=_xtb.run_gfn2_xtb(mol,params,o)
    print(f"{TK:>8.0f} {TK*KB_HA_PER_K:>12.3e} {r.energy:>16.9f} {r.free_energy:>16.9f} "
          f"{r.entropy:>10.3e} {(r.energy-XTB)*1e3:>+11.3f} {r.n_iter:>6}")
