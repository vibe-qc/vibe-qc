"""Periodic GFN2 #244: fast controls and a measured slow-primary witness.

Before the fix, max_iter=2500 cut the primary to 500 iterations. The polar
HF cell with explicit Simple/Aitken mixing 0.02 or 0.01 converged in about
1099/2205 iterations at max_iter=2499, but failed with the larger budget.
Run from the repository root with a source-matched native extension.
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

from vibeqc._vibeqc_core import semiempirical as _se

for mixing in (0.01, 0.02):
    print(f"--- slow polar HF, mixing={mixing} ---")
    for max_iter in (500, 2499, 2500, 2501, 3000):
        o = _xtb.XTBSccOptions()
        o.max_iter = max_iter
        o.conv_tol_charge = 1e-9
        o.auto_stabilize = True
        o.electronic_temperature = 0.001
        o.scc_mixer = _se.SCCMixer.Simple
        o.mixer_damping = 1.0
        o.charge_mixing = mixing
        r = _xtb.run_gfn2_xtb_gamma(_polar_hf_cell(), params, o, 12.0)
        print(f"  max_iter={max_iter:>5} conv={str(r.converged):>5} "
              f"n_iter={r.n_iter:>5} trace={len(r.scc_max_change_trace):>5} "
              f"E={r.energy:.15f} T={r.smearing_temperature}")
