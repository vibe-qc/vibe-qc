"""State-pair nonadiabatic couplings for MS/XMS-CASPT2."""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, Molecule
from vibeqc.gradient._ms_caspt2_nac import (
    _backtransform_state,
    compute_ms_caspt2_nac,
)
from vibeqc.runner import run_job
from vibeqc.solvers import CASPT2Options, CASSCFOptions


LIH = Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])])

# Live OpenMolcas 26.02, &GATEWAY RICD + full SA2-&RASSCF + XMS-&CASPT2
# NAC + &MCLR/&ALASKA, LiH/STO-3G R=3 bohr. OpenMolcas requires RI/CD for
# analytic MS-CASPT2 derivative couplings, whereas vibe-qc uses exact ERIs;
# the resulting 0.9% norm difference is covered by the component tolerance.
_OPENMOLCAS_XMS_NAC = np.array(
    [
        [0.0, 0.0, 0.194851114390451],
        [0.0, 0.0, -0.0719983958016190],
    ]
)
_OPENMOLCAS_MS_NAC = np.array(
    [
        [0.0, 0.0, 0.176900304453],
        [0.0, 0.0, -0.0582026780316975],
    ]
)


def _run_nac(
    tmp_path,
    name: str,
    pair: tuple[int, int],
    *,
    mode="xms",
    citations=False,
):
    return run_job(
        LIH,
        basis="sto-3g",
        method="caspt2",
        active_space=(2, 2),
        caspt2_options=CASPT2Options(
            multistate=mode,
            nac_pair=pair,
            nac_fd_step=2e-3,
        ),
        casscf_options=CASSCFOptions(nroots=2, weights=[0.5, 0.5]),
        output=tmp_path / name,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        citations=citations,
    )


def test_xms_caspt2_nac_public_route_and_citation(tmp_path):
    result = _run_nac(tmp_path, "xms_nac", (0, 1), citations=True)
    coupling = np.asarray(result.nonadiabatic_coupling)

    assert result.nonadiabatic_pair == (0, 1)
    assert coupling.shape == (2, 3)
    assert np.all(np.isfinite(coupling))
    # Linear LiH: only the bond-axis component survives. The global sign is
    # an arbitrary electronic-state phase, so compare both phase choices to
    # the independent OpenMolcas vector recorded above.
    assert np.max(np.abs(coupling[:, :2])) < 1e-4
    phase = 1.0 if np.vdot(coupling, _OPENMOLCAS_XMS_NAC) >= 0.0 else -1.0
    assert np.allclose(phase * coupling, _OPENMOLCAS_XMS_NAC, atol=2e-3)

    bibtex = (tmp_path / "xms_nac.bibtex").read_text()
    assert "park_shiozaki_ms_caspt2_nac_2017" in bibtex


def test_ms_caspt2_nac_route_is_finite_and_symmetry_adapted(tmp_path):
    result = _run_nac(tmp_path, "ms_nac", (0, 1), mode="ms")
    coupling = np.asarray(result.nonadiabatic_coupling)

    assert coupling.shape == (2, 3)
    assert np.all(np.isfinite(coupling))
    assert np.max(np.abs(coupling[:, :2])) < 1e-8
    assert np.linalg.norm(coupling[:, 2]) > 1e-3
    phase = 1.0 if np.vdot(coupling, _OPENMOLCAS_MS_NAC) >= 0.0 else -1.0
    assert np.allclose(phase * coupling, _OPENMOLCAS_MS_NAC, atol=7e-3)


def test_ordered_pair_is_antisymmetric(tmp_path):
    forward = _run_nac(tmp_path, "forward", (0, 1)).nonadiabatic_coupling
    reverse = _run_nac(tmp_path, "reverse", (1, 0)).nonadiabatic_coupling
    assert np.allclose(forward, -np.asarray(reverse), atol=2e-7)


def test_nac_requires_sa_casscf_reference(tmp_path):
    with pytest.raises(ValueError, match="state-averaged CASSCF"):
        run_job(
            LIH,
            basis="sto-3g",
            method="caspt2",
            active_space=(2, 2),
            caspt2_options=CASPT2Options(
                multistate="xms", nroots=2, nac_pair=(0, 1)
            ),
            output=tmp_path / "no_sa",
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
        )


def test_nac_requires_multistate_mode(tmp_path):
    with pytest.raises(ValueError, match="requires multistate"):
        run_job(
            LIH,
            basis="sto-3g",
            method="caspt2",
            active_space=(2, 2),
            caspt2_options=CASPT2Options(nac_pair=(0, 1)),
            casscf_options=CASSCFOptions(nroots=2, weights=[0.5, 0.5]),
            output=tmp_path / "no_multistate",
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
        )


def test_semicanonical_first_order_state_is_backtransformed():
    angle = 0.37
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )

    transformed = _backtransform_state(
        {1: 1.0}, rotation, n_core=0, n_active_orb=0
    )

    assert transformed == pytest.approx(
        {1: np.cos(angle), 2: np.sin(angle)}, abs=1e-14
    )


def test_low_level_pair_validation_precedes_integral_work():
    with pytest.raises(ValueError, match="two distinct roots"):
        compute_ms_caspt2_nac(
            None,
            None,
            None,
            None,
            None,
            0,
            2,
            n_active_elec=2,
            sa_weights=[0.5, 0.5],
            nroots=2,
            state_pair=(0, 0),
        )
