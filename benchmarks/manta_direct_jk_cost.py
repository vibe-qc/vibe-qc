"""Matched MANTA versus libint direct J/K cost on small molecules (tracker #638).

Runs one selected RHF (exact J/K) per fixture on both backends with identical
options, records wall time, J/K shell calls and primitive evaluations from the
engine provenance, and the energy agreement. Writes a JSON line per run.
Usage: .venv/bin/python benchmarks/manta_direct_jk_cost.py <out.jsonl> [fixture ...]
"""
import json, sys, time, platform, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import vibeqc as vq
from vibeqc import manta

FIXTURES = {
    "h2o-sto3g": (lambda: vq.Molecule([vq.Atom(8, [0, 0, .1173]), vq.Atom(1, [0, 1.4315, -.9314]), vq.Atom(1, [0, -1.4315, -.9314])], 0, 1), "sto-3g"),
    "h2o-631g": (lambda: vq.Molecule([vq.Atom(8, [0, 0, .1173]), vq.Atom(1, [0, 1.4315, -.9314]), vq.Atom(1, [0, -1.4315, -.9314])], 0, 1), "6-31g"),
    "h2o-def2svp": (lambda: vq.Molecule([vq.Atom(8, [0, 0, .1173]), vq.Atom(1, [0, 1.4315, -.9314]), vq.Atom(1, [0, -1.4315, -.9314])], 0, 1), "def2-svp"),
    "hbr-sto3g": (lambda: vq.Molecule([vq.Atom(35, [0, 0, 0]), vq.Atom(1, [0, 0, 2.7])], 0, 1), "sto-3g"),
    "znh+-sto3g": (lambda: vq.Molecule([vq.Atom(30, [0, 0, 0]), vq.Atom(1, [0, 0, 3.0])], 1, 1), "sto-3g"),
}

def options():
    o = vq.RHFOptions(); o.initial_guess = vq.InitialGuess.HCORE
    o.conv_tol_energy, o.conv_tol_grad, o.stability_check = 1e-10, 1e-8, False
    o.max_iter = 100
    return o

def run(name, backend):
    make, basis_name = FIXTURES[name]
    mol = make(); basis = vq.BasisSet(mol, basis_name)
    budget = manta.ResourceBudget(); budget.memory_bytes = 4 << 30
    budget.kernel_terms = budget.scalar_terms = budget.primitive_quartets = budget.shell_calls = 1 << 50
    t0 = time.perf_counter()
    r = vq.run_rhf(mol, basis, options(), integral_engine=vq.IntegralEngineOptions(backend=backend, budget=budget))
    wall = time.perf_counter()-t0
    p = manta.molecular_engine_provenance(r)
    return dict(fixture=name, backend=backend, basis=basis_name, nbf=int(basis.nbasis() if callable(basis.nbasis) else basis.nbasis), nshell=int(basis.nshells() if callable(basis.nshells) else basis.nshells),
                wall_s=wall, converged=bool(r.converged), n_iter=int(r.n_iter), energy=float(r.energy),
                jk_build_calls=p["jk_build_calls"], jk_shell_calls=p["jk_shell_calls"],
                primitive_integrals_evaluated=p["primitive_integrals_evaluated"],
                jk_wall_seconds=p["jk_wall_seconds"], one_electron_wall_seconds=p["one_electron_wall_seconds"])

if __name__ == "__main__":
    out = sys.argv[1]; names = sys.argv[2:] or list(FIXTURES)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    with open(out, "a") as f:
        f.write(json.dumps(dict(meta=True, main=head, machine=platform.machine(), python=platform.python_version(),
                                vibeqc_file=vq.__file__))+"\n")
        for name in names:
            for backend in ("libint", "manta"):
                row = run(name, backend); row["main"] = head
                print(json.dumps(row)); f.write(json.dumps(row)+"\n"); f.flush()
