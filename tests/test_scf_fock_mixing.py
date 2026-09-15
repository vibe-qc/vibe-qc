"""Generic SCF Fock-mixing controls.

CRYSTAL's ``FMIXING 30`` means "diagonalize 70% current Fock plus
30% previous Fock". vibe-qc exposes that as ``options.fock_mixing`` on
the shared molecular and periodic option structs. These tests keep the
control generic and disabled by default; MgO-style examples may tune the
value, but no method should need a hard-coded special case.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


def _h2_sto3g():
    mol = vq.Molecule([
        vq.Atom(1, [0.0, 0.0, -0.7]),
        vq.Atom(1, [0.0, 0.0, 0.7]),
    ])
    return mol, vq.BasisSet(mol, "sto-3g")


def _tighten(opts):
    opts.conv_tol_energy = 1.0e-10
    opts.conv_tol_grad = 1.0e-8
    opts.max_iter = 80
    return opts


@pytest.fixture(scope="module")
def _low_level_h2_parts():
    mol, basis = _h2_sto3g()
    overlap = np.asarray(vq.compute_overlap(basis), dtype=float)
    hcore = np.asarray(vq.compute_kinetic(basis), dtype=float) + np.asarray(
        vq.compute_nuclear(basis, mol), dtype=float
    )
    e_nuc = 1.0 / 1.4
    jk = vq.make_direct_jk_builder(basis)
    grid = vq.build_grid(mol, vq.GridOptions())
    return basis, overlap, hcore, e_nuc, jk, grid


@pytest.mark.parametrize("cls", [
    vq.RHFOptions,
    vq.RKSOptions,
    vq.UHFOptions,
    vq.UKSOptions,
    vq.PeriodicRHFOptions,
    vq.PeriodicSCFOptions,
    vq.PeriodicKSOptions,
])
def test_fock_mixing_option_default_is_off(cls):
    opts = cls()
    assert hasattr(opts, "fock_mixing"), cls.__name__
    assert opts.fock_mixing == 0.0
    opts.fock_mixing = 0.30
    assert opts.fock_mixing == pytest.approx(0.30)


@pytest.mark.parametrize("runner, cls, functional", [
    (vq.run_rhf, vq.RHFOptions, None),
    (vq.run_rks, vq.RKSOptions, "LDA"),
    (vq.run_uhf, vq.UHFOptions, None),
    (vq.run_uks, vq.UKSOptions, "LDA"),
])
def test_molecular_fock_mixing_preserves_fixed_point(
    runner, cls, functional,
):
    mol, basis = _h2_sto3g()

    baseline_opts = _tighten(cls())
    if functional is not None:
        baseline_opts.functional = functional
    baseline = runner(mol, basis, baseline_opts)

    mixed_opts = _tighten(cls())
    mixed_opts.fock_mixing = 0.30
    if functional is not None:
        mixed_opts.functional = functional
    mixed = runner(mol, basis, mixed_opts)

    assert baseline.converged
    assert mixed.converged
    assert mixed.energy == pytest.approx(baseline.energy, abs=1.0e-8)


def test_molecular_fock_mixing_rejects_out_of_range():
    mol, basis = _h2_sto3g()
    opts = vq.RHFOptions()
    opts.fock_mixing = 1.0
    with pytest.raises(ValueError, match="fock_mixing"):
        vq.run_rhf(mol, basis, opts)


@pytest.mark.parametrize("field", ["damping", "fock_mixing"])
@pytest.mark.parametrize("runner, cls, functional", [
    (vq.run_rhf, vq.RHFOptions, None),
    (vq.run_rks, vq.RKSOptions, "LDA"),
    (vq.run_uhf, vq.UHFOptions, None),
    (vq.run_uks, vq.UKSOptions, "LDA"),
])
def test_molecular_mixing_controls_reject_nan(
    field, runner, cls, functional,
):
    """NaN must not be ignored or reach a later Fock diagonalization."""
    mol, basis = _h2_sto3g()
    opts = cls()
    if functional is not None:
        opts.functional = functional
    setattr(opts, field, float("nan"))

    with pytest.raises(ValueError, match=field):
        runner(mol, basis, opts)


@pytest.mark.parametrize("field", ["damping", "fock_mixing"])
@pytest.mark.parametrize("flavor", ["rhf", "rks", "uhf", "uks"])
def test_low_level_molecular_mixing_controls_reject_nan(
    field, flavor, _low_level_h2_parts,
):
    """Every public externally-built-Fock entry validates both controls."""
    basis, overlap, hcore, e_nuc, jk, grid = _low_level_h2_parts
    options_cls = {
        "rhf": vq.RHFOptions,
        "rks": vq.RKSOptions,
        "uhf": vq.UHFOptions,
        "uks": vq.UKSOptions,
    }[flavor]
    opts = options_cls()
    setattr(opts, field, float("nan"))

    with pytest.raises(ValueError, match=field):
        if flavor == "rhf":
            vq.run_rhf_scf_with_jk(
                basis, 2, overlap, hcore, e_nuc, jk, opts
            )
        elif flavor == "rks":
            vq.run_rks_scf_with_jk(
                basis, 2, overlap, hcore, e_nuc, jk, grid, opts
            )
        elif flavor == "uhf":
            vq.run_uhf_scf_with_jk(
                basis, 1, 1, overlap, hcore, e_nuc, jk, opts
            )
        else:
            vq.run_uks_scf_with_jk(
                basis, 1, 1, overlap, hcore, e_nuc, jk, grid, opts
            )
