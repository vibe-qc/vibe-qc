"""MgO is the handover's slow-converging periodic cell (Eyert 87 iters at
alpha 0.4, 437 at alpha 0.2).  Does any setting land it in the 501-2499
window where the stabilization reserve cuts the primary attempt?"""
import sys; sys.path.insert(0, "tests")
from test_periodic_gfn2_aes import _mgo
from test_gfn2_xtb import load_gfn2_params
import vibeqc._vibeqc_core.semiempirical.xtb as _xtb
import vibeqc._vibeqc_core.semiempirical as _sem
params = load_gfn2_params()

for label, mixer, damping in [("default", None, 0.0),
                              ("broyden a=0.2", _sem.SCCMixer.Broyden, 0.2),
                              ("simple a=0.1", _sem.SCCMixer.Simple, 0.0)]:
    row = []
    for max_iter in (2499, 2500):
        o = _xtb.XTBSccOptions()
        o.max_iter = max_iter; o.conv_tol_charge = 1.0e-9
        o.auto_stabilize = True; o.electronic_temperature = 0.001
        if mixer is not None:
            o.scc_mixer = mixer
            if damping: o.mixer_damping = damping
        try:
            r = _xtb.run_gfn2_xtb_gamma(_mgo(), params, o, 12.0)
            row.append((max_iter, r.converged, r.n_iter, r.energy))
        except Exception as e:
            row.append((max_iter, "ERR", str(e)[:40], 0.0))
    for max_iter, conv, n, e in row:
        print(f"{label:>14} max_iter={max_iter:>5} conv={str(conv):>5} n_iter={n:>6} E={e:>16.9f}"
              if isinstance(e, float) else f"{label:>14} max_iter={max_iter} {conv} {n}")
