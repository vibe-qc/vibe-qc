"""Fast MgO diagnostic: RHF first (fast, no DFT grid), then RKS trajectory."""

from __future__ import annotations

import os
import warnings

import numpy as np

os.environ.setdefault("VIBEQC_AOPAIR_FT_BACKEND", "python")

import vibeqc as vq
from vibeqc.pbc_bipole import run_pbc_bipole_rhf
from vibeqc.pbc_bipole_rks import PeriodicKSOptions, run_pbc_bipole_rks

ANG2BOHR = 1.0 / 0.529177210903
a = 4.21 * ANG2BOHR
lattice = (a / 2.0) * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
atoms = [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [0.25 * a, 0.25 * a, 0.25 * a])]
sysp = vq.PeriodicSystem(3, lattice, atoms)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
kmesh = vq.monkhorst_pack(sysp, [1, 1, 1])
V = float(abs(np.linalg.det(lattice)))
nbf = basis.nbasis
print(f"MgO primitive: V={V:.2f} bohr^3, nbf={nbf}, n_elec={sysp.n_electrons()}")

# ---- RHF ----
print("\n=== RHF BIPOLE ===")
opts = vq.PeriodicSCFOptions()
opts.lattice_opts.cutoff_bohr = 10.0
opts.lattice_opts.nuclear_cutoff_bohr = 10.0
opts.max_iter = 80
opts.use_diis = True
opts.conv_tol_energy = 1e-8
opts.conv_tol_grad = 1e-5

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result = run_pbc_bipole_rhf(
        sysp,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
        progress=False,
    )
print(f"  converged={result.converged}, n_iter={result.n_iter}, E={result.energy:.8f}")
print(f"  ewald_alpha={getattr(result, 'ewald_alpha_bohr_inv', None)}")

# Energy components
for attr in dir(result):
    if attr.startswith("e_") and not attr.startswith("__"):
        try:
            print(f"  {attr}={getattr(result, attr):.10f}")
        except:
            pass

# ---- RKS SVWN ----
print("\n=== RKS SVWN BIPOLE ===")
opts = PeriodicKSOptions()
opts.lattice_opts.cutoff_bohr = 10.0
opts.lattice_opts.nuclear_cutoff_bohr = 10.0
opts.max_iter = 80
opts.use_diis = True
opts.scf_accelerator = vq.SCFAccelerator.KDIIS
opts.conv_tol_energy = 1e-6
opts.functional = "svwn"
opts.initial_guess = vq.InitialGuess.SAD

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result_ks = run_pbc_bipole_rks(
        sysp,
        basis,
        kmesh,
        opts,
        use_ewald_j_split=True,
        ewald_precision=1e-8,
        progress=True,
    )
print(
    f"\n  converged={result_ks.converged}, n_iter={result_ks.n_iter}, E={result_ks.energy:.8f}"
)
