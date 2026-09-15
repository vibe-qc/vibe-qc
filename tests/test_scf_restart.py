"""BUG 64 — regression coverage for deterministic SCF restart + SOSCF."""

from __future__ import annotations

import pytest
from vibeqc import Atom, BasisSet, Molecule, RHFOptions, UHFOptions, run_rhf, run_uhf


def _make_mol(atoms_bohr, charge=0, multiplicity=1):
    return Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms_bohr], charge, multiplicity)


ACETAMIDE_ATOMS = [
    (6,  [-4.612171, 1.076977, 0.032256]), (6,  [-2.800419, 0.782592, 0.019328]),
    (7,  [-2.074253, -0.405067, -0.088761]), (8,  [-2.140144, 1.847441, 0.110168]),
    (1,  [-4.600118, 2.152023, 0.278278]), (1,  [-5.158507, 1.009373, -0.919968]),
    (1,  [-5.159677, 0.574792, 0.853413]), (1,  [-1.098423, -0.287567, -0.081448]),
    (1,  [-2.499831, -1.312388, -0.174913]),
]

FECL3_ATOMS = [
    (26, [0.0, 0.0, 0.0]), (17, [0.0, 0.0, 4.025916]),
    (17, [3.486550, 0.0, -2.012958]), (17, [-3.486550, 0.0, -2.012958]),
]

CYCLOPROPENE_ATOMS = [
    (6, [0.0, 1.256167, 0.442893]), (6, [0.0, 0.0, -1.240592]),
    (6, [0.0, -1.256167, 0.442893]), (1, [0.0, 2.172384, 1.022354]),
    (1, [0.0, -2.172384, 1.022354]), (1, [0.914226, 0.0, -1.854756]),
    (1, [-0.914226, 0.0, -1.854756]),
]


@pytest.fixture
def default_rhf_opts():
    o = RHFOptions(); o.max_iter = 120; return o

@pytest.fixture
def default_uhf_opts():
    o = UHFOptions(); o.max_iter = 120; return o


@pytest.mark.parametrize("basis_name", ["sto-3g", "def2-svp"])
def test_acetamide_rhf_converges(default_rhf_opts, basis_name):
    mol = _make_mol(ACETAMIDE_ATOMS); basis = BasisSet(mol, basis_name)
    assert run_rhf(mol, basis, default_rhf_opts).converged

@pytest.mark.parametrize("basis_name", ["sto-3g", "def2-svp"])
def test_cyclopropene_rhf_converges(default_rhf_opts, basis_name):
    mol = _make_mol(CYCLOPROPENE_ATOMS); basis = BasisSet(mol, basis_name)
    assert run_rhf(mol, basis, default_rhf_opts).converged

@pytest.mark.parametrize("basis_name", ["sto-3g", "def2-svp"])
def test_fecl3_uhf_converges(default_uhf_opts, basis_name):
    mol = _make_mol(FECL3_ATOMS, multiplicity=6); basis = BasisSet(mol, basis_name)
    assert run_uhf(mol, basis, default_uhf_opts).converged

def test_restart_triggers_on_hard_case():
    mol = _make_mol(ACETAMIDE_ATOMS); basis = BasisSet(mol, "sto-3g")
    opts = RHFOptions(); opts.max_iter = 200
    opts.use_diis = False; opts.damping = 0.0; opts.dynamic_damping = False
    opts.restart_opts.max_stall_iters = 10
    assert run_rhf(mol, basis, opts).converged

def test_restart_options_surface():
    o = RHFOptions()
    assert o.restart_opts.enabled is True
    assert o.restart_opts.max_stall_iters == 15
    assert o.restart_opts.seed == 42
    assert o.restart_opts.max_restarts == 3
    o.restart_opts.enabled = False; o.restart_opts.max_stall_iters = 7
    assert o.restart_opts.enabled is False
    assert o.restart_opts.max_stall_iters == 7

def test_restart_disabled_is_noop():
    mol = _make_mol(ACETAMIDE_ATOMS); basis = BasisSet(mol, "sto-3g")
    opts = RHFOptions(); opts.restart_opts.enabled = False
    result = run_rhf(mol, basis, opts)
    assert result.converged
    ref = -196.21888492
    assert abs(result.energy - ref) < 1e-6

def test_restart_conserves_energy():
    mol = _make_mol(ACETAMIDE_ATOMS); basis = BasisSet(mol, "sto-3g")
    opts_off = RHFOptions(); opts_off.restart_opts.enabled = False
    opts_on = RHFOptions(); opts_on.restart_opts.enabled = True
    r_off = run_rhf(mol, basis, opts_off)
    r_on = run_rhf(mol, basis, opts_on)
    assert r_off.converged and r_on.converged
    assert abs(r_off.energy - r_on.energy) < 1e-8

def test_restart_available_on_all_drivers():
    from vibeqc import RKSOptions, UKSOptions
    for cls in (RHFOptions, UHFOptions, RKSOptions, UKSOptions):
        o = cls()
        assert o.restart_opts.enabled is True
        o.restart_opts.enabled = False
        assert o.restart_opts.enabled is False
