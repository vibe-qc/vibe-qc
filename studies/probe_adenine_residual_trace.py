"""#3: is the adenine primary attempt contracting when the 700-cap fires?"""
import sys; sys.path.insert(0, "tests")
import numpy as np
from test_gfn2_xtb import _ADENINE_ANGSTROM, load_gfn2_params
from vibeqc.molecule import Molecule, Atom
import vibeqc._vibeqc_core.semiempirical.xtb as _xtb
BOHR = 1.0/0.529177210903
mol = Molecule([Atom(z,[x*BOHR,y*BOHR,w*BOHR]) for z,x,y,w in _ADENINE_ANGSTROM],0,1)
params = load_gfn2_params()

o=_xtb.XTBSccOptions(); o.max_iter=4000; o.conv_tol_charge=1e-8
o.aes_faithful=False; o.auto_stabilize=True; o.aes_damping=0.5
r=_xtb.run_gfn2_xtb(mol,params,o)
tr = np.asarray(r.attempts[0].max_change_trace)
print(f"attempt[0]: {len(tr)} iterations, exit={r.attempts[0].exit_reason}")
print(f"  residual at   1: {tr[0]:.6e}")
for i in (100, 300, 500, 600, 650, 680, 690, 695, 699):
    print(f"  residual at {i:>3}: {tr[i-1]:.6e}")
print(f"  best over whole attempt: {tr.min():.6e} at iter {int(tr.argmin())+1}")
print(f"  tolerance: {o.conv_tol_charge:.1e}")
# contraction over the last windows
for w in (15, 50, 100):
    print(f"  min(last {w:>3}) = {tr[-w:].min():.6e}   min(prev {w:>3}) = {tr[-2*w:-w].min():.6e}")

# and the attempt that DOES converge, at max_iter=2499
o2=_xtb.XTBSccOptions(); o2.max_iter=2499; o2.conv_tol_charge=1e-8
o2.aes_faithful=False; o2.auto_stabilize=True; o2.aes_damping=0.5
r2=_xtb.run_gfn2_xtb(mol,params,o2)
t2 = np.asarray(r2.attempts[0].max_change_trace)
print(f"\nconverging attempt: {len(t2)} iters")
for i in (690, 700, 705, 710, 715, 719, 720):
    if i <= len(t2): print(f"  residual at {i:>3}: {t2[i-1]:.6e}")
