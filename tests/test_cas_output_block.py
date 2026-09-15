"""The .out solver block for the CAS family (M4 output polish).

CASCI / CASSCF / CASPT2 / NEVPT2 runs surface their reference
wavefunction in the .out: leading determinant configurations, per-root
energies (multi-root / SA runs), and active natural-orbital occupations.
"""

from __future__ import annotations

import pytest
from vibeqc import Atom, Molecule
from vibeqc.runner import run_job
from vibeqc.solvers import CASSCFOptions

H2O = Molecule(
    [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 1.43, -0.93]),
        Atom(1, [0.0, -1.43, -0.93]),
    ]
)


def _run(method, tmp_path, **kw):
    return run_job(
        H2O,
        basis="sto-3g",
        method=method,
        output=tmp_path / method,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        citations=False,
        **kw,
    )


def test_sa_casscf_out_block_has_roots_configs_and_occupations(tmp_path):
    res = _run(
        "casscf",
        tmp_path,
        active_space=(4, 4),
        casscf_options=CASSCFOptions(nroots=2),
    )
    out = (tmp_path / "casscf.out").read_text()
    assert "Leading configurations:" in out
    assert "Root energies:" in out and "root 1:" in out
    # Per-root <S^2> column (visibility surface of the spin-pure SA
    # default): both averaged roots are singlets under the default.
    assert out.count("<S^2> =  0.0000") == 2
    assert "Natural occupations (active):" in out
    # The diagnostics also live on the result object.
    assert res.rdm1 is not None and res.rdm1.shape == (4, 4)
    assert res.ci_labels is not None and len(res.ci_labels) == 36


def test_caspt2_out_block_shows_reference_diagnostics(tmp_path):
    _run("caspt2", tmp_path, active_space=(2, 2))
    out = (tmp_path / "caspt2.out").read_text()
    assert "Leading configurations:" in out
    assert "Natural occupations (active):" in out


def test_casci_natural_occupations_sum_to_electron_count(tmp_path):
    import numpy as np

    res = _run("casci", tmp_path, active_space=(4, 4))
    occ = np.linalg.eigvalsh(res.rdm1)
    assert abs(occ.sum() - 4.0) < 1e-8  # trace of the active 1-RDM
