"""vibe-qc#3 closure criterion 1: one converged adenine state per AES model,
independent of auto_stabilize AND of the SCC mixer choice.  The re-pinned
test sweeps AES x auto_stabilize only; this measures the mixer axis."""
import sys; sys.path.insert(0, "tests")
from test_gfn2_xtb import _ADENINE_ANGSTROM, load_gfn2_params
from vibeqc.molecule import Molecule, Atom
import vibeqc._vibeqc_core.semiempirical.xtb as _xtb
import vibeqc._vibeqc_core.semiempirical as _sem
BOHR = 1.0 / 0.529177210903
mol = Molecule([Atom(z, [x*BOHR, y*BOHR, w*BOHR]) for z, x, y, w in _ADENINE_ANGSTROM], 0, 1)
params = load_gfn2_params()

mixers = [("default (hybrid)", None)] + [(m, getattr(_sem.SCCMixer, m))
          for m in ("Simple", "Broyden", "BroydenEyert", "DIIS", "Newton")]
print(f"{'aes':>9} {'stab':>5} {'mixer':>18} {'conv':>5} {'iters':>6} {'att':>4} {'T_acc':>6} {'E (Ha)':>16}")
by_model = {False: set(), True: set()}
for faithful in (False, True):
    for stab in (True, False):
        for label, mixer in mixers:
            o = _xtb.XTBSccOptions(); o.max_iter = 4000; o.conv_tol_charge = 1e-8
            o.aes_faithful = faithful; o.aes_damping = 0.25 if faithful else 0.5
            o.auto_stabilize = stab
            if mixer is not None:
                o.scc_mixer = mixer
            try:
                r = _xtb.run_gfn2_xtb(mol, params, o)
            except Exception as e:
                print(f"{('faithful' if faithful else 'ad-hoc'):>9} {str(stab):>5} {label:>18} ERROR {str(e)[:60]}")
                continue
            acc = r.attempts[r.selected_attempt_index].electronic_temperature if r.selected_attempt_index >= 0 else None
            if r.converged:
                by_model[faithful].add(round(r.energy, 7))
            print(f"{('faithful' if faithful else 'ad-hoc'):>9} {str(stab):>5} {label:>18} {str(r.converged):>5} "
                  f"{r.n_iter:>6} {len(r.attempts):>4} {str(acc):>6} {r.energy:>16.9f}")
for faithful, es in by_model.items():
    print(f"{'faithful' if faithful else 'ad-hoc'}: distinct converged energies = {sorted(es)}")
