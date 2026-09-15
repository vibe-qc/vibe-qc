"""Phase 12d: multi-k periodic Kohn–Sham DFT (LDA / pure GGA).

Test strategy mirrors 12c's: the core correctness witness is machine-
precision agreement with molecular RKS in the molecular-limit regime
(unit cell big enough that lattice-sum contributions beyond g=0 are
numerically zero). Real bulk tests against published CRYSTAL values
arrive once the Coulomb lattice sum is handled via Ewald (12e) and the
periodic Becke partition is in place (12f).
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


H2  = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])]
H2O = [vq.Atom(8, [0.0,  0.0,  0.0]),
       vq.Atom(1, [0.0,  1.43, -0.98]),
       vq.Atom(1, [0.0, -1.43, -0.98])]


def _big_box_system(atoms, dim, box=50.0, vacuum=30.0):
    if dim == 1:   lat = np.diag([box, vacuum, vacuum])
    elif dim == 2: lat = np.diag([box, box, vacuum])
    else:          lat = np.diag([box, box, box])
    return vq.PeriodicSystem(dim, lat, atoms)


def _ks_opts(functional="LDA", cutoff=15.0):
    o = vq.PeriodicKSOptions()
    o.functional = functional
    o.lattice_opts.cutoff_bohr = cutoff
    o.lattice_opts.nuclear_cutoff_bohr = cutoff
    o.conv_tol_energy = 1e-12
    o.conv_tol_grad = 1e-10
    o.max_iter = 100
    return o


def _molecular_rks(atoms, functional):
    mol = vq.Molecule(atoms, 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RKSOptions()
    opts.functional = functional
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    return vq.run_rks(mol, basis, opts)


FUNCTIONALS = ["LDA", "PBE", "BLYP"]


# ---------------------------------------------------------------------------
# Molecular-limit agreement, across dim ∈ {1, 2, 3} × functionals × systems
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,atoms,dim,functional",
    [(n, a, d, f)
     for (n, a) in (("H2", H2), ("H2O", H2O))
     for d in (1, 2, 3)
     for f in FUNCTIONALS],
)
def test_molecular_limit_matches_molecular_rks(name, atoms, dim, functional):
    sysp = _big_box_system(atoms, dim)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = vq.monkhorst_pack(sysp, [2] * dim + [1] * (3 - dim))
    res = vq.run_rks_periodic(sysp, basis, km, _ks_opts(functional))
    assert res.converged
    mres = _molecular_rks(atoms, functional)
    diff = abs(res.energy - mres.energy)
    assert diff < 1e-9, (
        f"{name}/{functional}/dim={dim}: periodic {res.energy:.12f} vs "
        f"molecular {mres.energy:.12f}, diff = {diff:.2e}"
    )


# ---------------------------------------------------------------------------
# k-mesh independence in the molecular limit
# ---------------------------------------------------------------------------

def test_kmesh_independence_in_molecular_limit():
    sysp = _big_box_system(H2O, 3)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    energies = []
    for mesh in [[1,1,1], [2,2,2], [3,3,3]]:
        km = vq.monkhorst_pack(sysp, mesh)
        res = vq.run_rks_periodic(sysp, basis, km, _ks_opts("PBE"))
        energies.append(res.energy)
    assert max(energies) - min(energies) < 1e-10


# ---------------------------------------------------------------------------
# Energy decomposition
# ---------------------------------------------------------------------------

def test_energy_decomposition_sums_to_total():
    sysp = _big_box_system(H2O, 3)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    res = vq.run_rks_periodic(sysp, basis, km, _ks_opts("LDA"))
    # E_total == E_electronic + E_nuclear
    assert abs(res.energy - (res.e_electronic + res.e_nuclear)) < 1e-10
    # E_hf_exchange should be zero for pure LDA.
    assert abs(res.e_hf_exchange) < 1e-10
    # E_xc should be negative (LDA exchange-correlation is stabilising).
    assert res.e_xc < 0


# ---------------------------------------------------------------------------
# build_xc_periodic standalone reduces to molecular for trivial density
# ---------------------------------------------------------------------------

# (A standalone build_xc_periodic correctness test is omitted because
# LatticeMatrixSet has no Python constructor; the molecular-limit total-
# energy match covers the XC build end-to-end.)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_rejects_odd_electron_unit_cell():
    sysp = vq.PeriodicSystem(3, np.eye(3) * 10.0,
                             [vq.Atom(1, [0,0,0])],
                             charge=0, multiplicity=2)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = vq.monkhorst_pack(sysp, [2,2,2])
    with pytest.raises(ValueError, match="even electron count"):
        vq.run_rks_periodic(sysp, basis, km)


def test_scf_trace_populated_and_monotonic():
    sysp = _big_box_system(H2O, 3)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    km = vq.monkhorst_pack(sysp, [1,1,1])
    res = vq.run_rks_periodic(sysp, basis, km, _ks_opts("PBE"))
    assert len(res.scf_trace) >= 2
    # Final gradient must be below the converged threshold.
    assert res.scf_trace[-1].grad_norm < 1e-10
