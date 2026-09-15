#!/usr/bin/env python3
"""Define a custom functional at runtime — ``vq.define_functional`` demo.

This script defines a PW1PW clone ("my-pw1pw") from the Python input,
then runs a quick molecular RKS/H₂O/def2-SVP total energy and verifies
it matches the built-in PW1PW.

The same ``define_functional`` call works for periodic jobs — any driver
that accepts ``functional="..."`` will resolve the runtime alias.

Run:
    python examples/molecular/define_functional_demo.py

Requires: vibe-qc >= v0.11.3
"""

from __future__ import annotations

import numpy as np
import vibeqc as vq

# ---------------------------------------------------------------------------
# 1. Register the custom functional
# ---------------------------------------------------------------------------

vq.define_functional(
    "my-pw1pw",
    [("GGA_X_PW91", 0.80), ("GGA_C_PW91", 1.00)],
    hf_exchange_fraction=0.20,
)

# ---------------------------------------------------------------------------
# 2. Verify it matches the built-in PW1PW
# ---------------------------------------------------------------------------

f_custom = vq.Functional("my-pw1pw")
f_builtin = vq.Functional("pw1pw")

print(
    f"Custom  : {f_custom.name}, kind={f_custom.kind}, HF={f_custom.hf_exchange_fraction}"
)
print(
    f"Built-in: {f_builtin.name}, kind={f_builtin.kind}, HF={f_builtin.hf_exchange_fraction}"
)

assert f_custom.hf_exchange_fraction == 0.20
assert f_custom.is_hybrid
assert f_custom.kind == f_builtin.kind

# XC energy density on a test grid — must match the built-in.
rho = np.array([0.1, 0.5, 1.0])
sigma = np.array([0.01, 0.05, 0.2])
exc_custom, _, _ = f_custom.eval_unpolarised(rho, sigma)
exc_builtin, _, _ = f_builtin.eval_unpolarised(rho, sigma)
assert np.allclose(exc_custom, exc_builtin, atol=1e-12), (
    f"XC mismatch: custom={exc_custom}, built-in={exc_builtin}"
)
print("XC energies match built-in PW1PW ✓")

# ---------------------------------------------------------------------------
# 3. Run a molecular SCF with the custom functional
# ---------------------------------------------------------------------------

# H₂O geometry, positions in bohr.
bohr = 1.0 / 0.529177210903
h2o = vq.Molecule(
    [
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.793353 * bohr, -0.613510 * bohr]),
        vq.Atom(1, [0.0, -0.793353 * bohr, -0.613510 * bohr]),
    ],
    charge=0,
    multiplicity=1,
)
basis = vq.BasisSet(h2o, "def2-svp")

opts_custom = vq.RKSOptions()
opts_custom.functional = "my-pw1pw"
opts_custom.max_iter = 100
opts_custom.conv_tol_energy = 1e-8

opts_builtin = vq.RKSOptions()
opts_builtin.functional = "pw1pw"
opts_builtin.max_iter = 100
opts_builtin.conv_tol_energy = 1e-8

r_custom = vq.run_rks(h2o, basis, opts_custom)
r_builtin = vq.run_rks(h2o, basis, opts_builtin)

print(f"\nH₂O PW1PW/def2-SVP")
print(
    f"  Custom  total energy : {r_custom.energy:.10f} Ha  converged={r_custom.converged}"
)
print(
    f"  Built-in total energy: {r_builtin.energy:.10f} Ha  converged={r_builtin.converged}"
)

assert r_custom.converged and r_builtin.converged, "SCF did not converge"
assert abs(r_custom.energy - r_builtin.energy) < 1e-10, (
    f"Energy mismatch: {r_custom.energy} vs {r_builtin.energy}"
)
print("Molecular SCF energies match ✓")
print("\nDone — custom functional resolution works end-to-end.")
