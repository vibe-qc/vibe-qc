"""Every documented CAS-family run_job surface is actually reachable.

Guards the v0.12-cut regression class: multi-root CASCI was *documented*
but *unreachable* through run_job until 6406e2f7 (the CASCIOptions wiring).
This file exhaustively drives each CAS-family method string and its option
object through the public run_job entry point and asserts it runs and
returns a sensible energy, so a future refactor cannot silently strand a
documented path again.

These are reachability/wiring checks (does the public surface dispatch and
run), not numerical-accuracy checks: the engines' physics is pinned in
tests/test_solvers_*.py / test_caspt2_*.py / test_selected_casci.py.
"""

from __future__ import annotations

import typing

import pytest
from vibeqc import Atom, Molecule
from vibeqc.runner import Method, run_job
from vibeqc.solvers import (
    CASCIOptions,
    CASPT2Options,
    CASSCFOptions,
    SelectedCIOptions,
    SelectedCIPT2Options,
)

H2 = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
LIH = Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])])


def _exact_sci():
    return SelectedCIOptions(
        target_size=100000, max_iter=80, conv_tol_energy=1e-12,
        pt2_threshold=1e-13, max_det_per_iter=100000, significant_coeff=1e-9,
    )


def _run(tmp_path, mol, **kw):
    return run_job(
        mol, basis="sto-3g", output=tmp_path / "job", citations=False,
        write_molden_file=False, write_xyz_file=False,
        write_population_file=False, **kw,
    )


# Each entry: (id, molecule, run_job kwargs).  Covers every CAS-family
# method string + every documented option object that selects a distinct
# code path through run_job.
_CASES = [
    ("casci", H2, dict(method="casci", active_space=(2, 2))),
    ("casci_multiroot", H2, dict(method="casci", active_space=(2, 2),
                                 casci_options=CASCIOptions(nroots=2))),
    ("casscf", H2, dict(method="casscf", active_space=(2, 2))),
    ("casscf_sa", H2, dict(method="casscf", active_space=(2, 2),
                           casscf_options=CASSCFOptions(nroots=2))),
    ("casscf_selci", LIH, dict(
        method="casscf", active_space=(2, 2),
        casscf_options=CASSCFOptions(ci_solver="selected_ci",
                                     selected_ci_options=_exact_sci()))),
    ("casscf_selci_pt2", LIH, dict(
        method="casscf", active_space=(2, 2),
        casscf_options=CASSCFOptions(
            ci_solver="selected_ci", selected_ci_options=_exact_sci(),
            pt2=SelectedCIPT2Options(eps2=0.0)))),
    ("nevpt2", H2, dict(method="nevpt2", active_space=(2, 2))),
    ("nevpt2_casscf", H2, dict(method="nevpt2", active_space=(2, 2),
                               casscf_options=CASSCFOptions())),
    ("nevpt2_selected_ref", LIH, dict(
        method="nevpt2", active_space=(2, 2),
        casscf_options=CASSCFOptions(ci_solver="selected_ci",
                                     selected_ci_options=_exact_sci()))),
    ("caspt2", H2, dict(method="caspt2", active_space=(2, 2))),
    ("caspt2_imaginary", H2, dict(method="caspt2", active_space=(2, 2),
                                  caspt2_options=CASPT2Options(imaginary=0.1))),
    ("caspt2_casscf", H2, dict(method="caspt2", active_space=(2, 2),
                               casscf_options=CASSCFOptions())),
    ("caspt2_ms", LIH, dict(method="caspt2", active_space=(2, 2),
                            caspt2_options=CASPT2Options(multistate="ms",
                                                         nroots=2))),
    ("caspt2_xms", LIH, dict(method="caspt2", active_space=(2, 2),
                             caspt2_options=CASPT2Options(multistate="xms",
                                                          nroots=2))),
    ("caspt2_selected_ref", LIH, dict(
        method="caspt2", active_space=(2, 2),
        casscf_options=CASSCFOptions(ci_solver="selected_ci",
                                     selected_ci_options=_exact_sci()))),
    ("caspt2_cases", H2, dict(
        method="caspt2", active_space=(2, 2),
        caspt2_options=CASPT2Options(engine="cases"))),
    ("mrci", H2, dict(method="mrci", active_space=(2, 2))),
    ("mrci_casscf", H2, dict(method="mrci", active_space=(2, 2),
                             casscf_options=CASSCFOptions())),
    ("fci", H2, dict(method="fci")),
    ("selected_ci", H2, dict(method="selected_ci")),
    ("dmrg", H2, dict(method="dmrg")),
    ("v2rdm", H2, dict(method="v2rdm")),
    ("transcorrelated_ci", H2, dict(method="transcorrelated_ci")),
]


@pytest.mark.parametrize("name,mol,kw", _CASES, ids=[c[0] for c in _CASES])
def test_cas_method_reachable(tmp_path, name, mol, kw):
    res = _run(tmp_path, mol, **kw)
    assert res.energy < 0.0          # a bound molecular energy came back
    assert res.method                # a method label was assigned
    assert res.converged


def test_caspt2_cases_engine_matches_auto(tmp_path):
    # The case-partitioned matrix-free engine, reached through run_job via
    # CASPT2Options(engine="cases"), reproduces the default engine and labels
    # the run distinctly.
    auto = _run(tmp_path, H2, method="caspt2", active_space=(2, 2),
                caspt2_options=CASPT2Options(engine="auto"))
    cases = _run(tmp_path, H2, method="caspt2", active_space=(2, 2),
                 caspt2_options=CASPT2Options(engine="cases"))
    assert abs(auto.energy - cases.energy) < 1e-8
    assert cases.method.endswith("_cases")


def test_multiroot_casci_surfaces_all_roots(tmp_path):
    # The exact v0.12 bug: CASCIOptions(nroots=N) must actually surface N
    # root energies through run_job, not silently collapse to one root.
    res = _run(tmp_path, Molecule([
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 1.43, -0.93]),
        Atom(1, [0.0, -1.43, -0.93]),
    ]), method="casci", active_space=(4, 4),
        casci_options=CASCIOptions(nroots=3))
    assert res.root_energies is not None
    assert len(res.root_energies) == 3
    assert res.root_energies == sorted(res.root_energies)  # ascending roots


def test_caspt2_nevpt2_are_in_the_public_method_type():
    # Documented + dispatched through run_job, so they must be part of the
    # public Method literal (the doc/dispatch/type surfaces must agree).
    literal_values = set(typing.get_args(Method))
    for m in ("casci", "casscf", "nevpt2", "caspt2", "mrci", "fci",
              "selected_ci", "dmrg", "v2rdm", "transcorrelated_ci"):
        assert m in literal_values, f"{m!r} missing from Method literal"
