"""Python/C++ binding contract smoke tests."""

from __future__ import annotations

import pytest

import vibeqc as vq
import vibeqc.cc as cc
import vibeqc.solvers as solvers
from vibeqc import _vibeqc_core as core


def test_python_exports_required_native_symbols():
    """Catch stale editable builds after pybind11 API changes."""
    required = [
        "ExternalChargeGradient",
        "RHFOptions",
        "RKSOptions",
        "UHFOptions",
        "UKSOptions",
        "MP2Options",
        "UMP2Options",
        "CCSDOptions",
        "run_rhf",
        "run_rks",
        "run_uhf",
        "run_uks",
        "run_mp2",
        "run_ump2",
        "run_ccsd",
        "compute_overlap",
        "compute_eri",
        "run_rhf_periodic_gamma",
    ]
    missing = [name for name in required if not hasattr(core, name)]
    assert missing == []

    # A stale extension can still import the private module but fail the
    # public package re-export layer.
    assert vq.ExternalChargeGradient is core.ExternalChargeGradient
    assert vq.RHFOptions is core.RHFOptions
    assert vq.CCSDIteration is core.CCSDIteration
    assert vq.CCSDResult is core.CCSDResult
    assert vq.CCSDOptions is cc.CCSDOptions
    assert vq.run_ccsd is cc.run_ccsd
    assert vq.chemical_core_orbital_count is cc.chemical_core_orbital_count
    assert vq.CISDOptions is solvers.CISDOptions
    assert vq.CISDResult is solvers.CISDResult
    assert vq.cisd is solvers.cisd


@pytest.mark.parametrize("method", ["rhf", "rks", "uhf", "uks"])
def test_direct_scf_wrappers_accept_basis_name(method):
    """The public direct wrappers accept the documented string shorthand."""
    mol = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])]
    )
    options = {
        "rhf": vq.RHFOptions(),
        "rks": vq.RKSOptions(),
        "uhf": vq.UHFOptions(),
        "uks": vq.UKSOptions(),
    }[method]
    if method in ("rks", "uks"):
        options.functional = "lda"
    result = getattr(vq, f"run_{method}")(mol, "sto-3g", options)
    assert result.converged
