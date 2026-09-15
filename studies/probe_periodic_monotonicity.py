"""Is the periodic GFN2 stabilization-reserve defect reachable, or latent?

periodic_gfn2.cpp:429  primary_max_iter = max_iter - (max_iter >= 2500 ? 2000 : 0)
so max_iter=2499 gives the primary 2499 iterations and 2500 gives it 500.
A system converging in 501..2499 iterations is cut off by the LARGER budget.
"""
import sys; sys.path.insert(0, "tests")
import numpy as np
from test_periodic_gfn2_aes import _polar_hf_cell, _hbn
from test_gfn2_xtb import load_gfn2_params
import vibeqc._vibeqc_core.semiempirical.xtb as _xtb

params = load_gfn2_params()
cases = {"polar_hf": (_polar_hf_cell(), 12.0), "hbn": (_hbn(0.1), 12.0)}

for name, (system, cutoff) in cases.items():
    print(f"--- {name} ---")
    for max_iter in (2499, 2500):
        o = _xtb.XTBSccOptions()
        o.max_iter = max_iter
        o.conv_tol_charge = 1.0e-9
        o.auto_stabilize = True
        r = _xtb.run_gfn2_xtb_gamma(system, params, o, cutoff)
        print(f"  max_iter={max_iter:>5} conv={str(r.converged):>5} "
              f"n_iter={r.n_iter:>5} E={r.energy:>16.9f}")
