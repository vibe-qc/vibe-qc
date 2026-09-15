"""Analytic CIS/TDA excited-state gradient — MSINDO/INDO (Phase 6, M2).

Validates :func:`vibeqc.semiempirical.methods.msindo_cis.cis_gradient`
(analytic, CPHF Z-vector path) against the finite-difference reference
:func:`cis_gradient_fd` for closed-shell singlet and triplet states.

The analytic gradient reproduces FD to ~1e-6 Ha/bohr for non-degenerate
states; the geometries below are deliberately symmetry-broken so the CIS
states are non-degenerate (per-state gradients are ill-defined at degeneracies).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc.semiempirical.methods.msindo_cis import (
    cis_gradient,
    cis_gradient_fd,
    run_cis,
)

# Non-degenerate, symmetry-broken closed-shell geometries (Angstrom).
H2O = ([8, 1, 1], [[0.0, 0.0, 0.0], [0.97, 0.0, 0.12], [-0.25, 0.91, 0.05]])
NH3 = (
    [7, 1, 1, 1],
    [[0.0, 0.0, 0.0], [0.10, 0.98, 0.30], [0.85, -0.40, 0.42], [-0.78, -0.52, 0.35]],
)
H2CO = (
    [6, 8, 1, 1],
    [[0.0, 0.0, 0.0], [1.21, 0.05, 0.03], [-0.58, 0.94, 0.02], [-0.55, -0.92, -0.05]],
)
HF = ([9, 1], [[0.0, 0.0, 0.0], [0.0, 0.30, 0.85]])


@pytest.mark.parametrize("mol", [H2O, NH3, H2CO, HF], ids=["H2O", "NH3", "H2CO", "HF"])
@pytest.mark.parametrize("spin", ["singlet", "triplet"])
@pytest.mark.parametrize("state", [0, 1])
def test_cis_gradient_matches_fd(mol, spin, state):
    """Analytic CIS gradient == central-difference FD to ~1e-5 Ha/bohr.

    Published reference for the analytic CIS gradient: Foresman, Head-Gordon,
    Pople & Frisch, J. Phys. Chem. 96, 135 (1992).
    """
    Z, coords = mol
    cis = run_cis(Z, coords, n_states=max(state + 2, 4), spin=spin)
    g_an = cis_gradient(Z, coords, cis, state=state, spin=spin)
    g_fd = cis_gradient_fd(Z, coords, state=state, spin=spin, step=1e-3)
    assert np.max(np.abs(g_an - g_fd)) < 1e-5, (
        f"{spin} S{state + 1}: analytic vs FD differ by "
        f"{np.max(np.abs(g_an - g_fd)):.2e} Ha/bohr\nanalytic={g_an}\nfd={g_fd}"
    )


# MSINDO oracle (2025e) analytic CIS *singlet* gradient for the H2O geometry
# above, parsed from "FIRST DERIVATIVES [H/BOHR]" of
#   CARTES RHF CIS DAVIDSONCIS SROI 6 CISGRAD(state+2) GRADONLY
# (examples/regression/msindo/runner_msindo.run_cis_gradient). The oracle and
# analytic excitation energies are identical to the printed precision (S1 6.76,
# S2 9.12 eV across all 6 roots), and the analytic gradient reproduces the
# oracle's own analytic gradient to ~3e-6 (S1) / ~6e-6 (S2) Ha/bohr. The values
# are pinned so the cross-check survives without the oracle binary at test time.
_ORACLE_H2O_SINGLET = {
    0: np.array(
        [
            [0.01126668, 0.05582804, 0.00635870],
            [-0.07078558, 0.03295556, -0.00582619],
            [0.05951890, -0.08878359, -0.00053251],
        ]
    ),
    1: np.array(
        [
            [0.07667791, 0.15767927, 0.02350861],
            [-0.08343328, -0.02727270, -0.01274705],
            [0.00675537, -0.13040657, -0.01076156],
        ]
    ),
}


@pytest.mark.parametrize("state", [0, 1])
def test_cis_gradient_matches_msindo_oracle_singlet(state):
    """Analytic singlet CIS gradient reproduces the MSINDO oracle's *own*
    analytic gradient (the rigorous DAVIDSONCIS / CISGRAD path) — Ha/bohr.

    Independent of the FD check above: this pins the port against the reference
    Fortran (clean-room: reading + citing ``cisgrad.f`` etc., never copying).
    """
    Z, coords = H2O
    cis = run_cis(Z, coords, n_states=6, spin="singlet")
    g = cis_gradient(Z, coords, cis, state=state, spin="singlet")
    assert np.max(np.abs(g - _ORACLE_H2O_SINGLET[state])) < 1e-4, (
        f"S{state + 1}: analytic vs oracle differ by "
        f"{np.max(np.abs(g - _ORACLE_H2O_SINGLET[state])):.2e} Ha/bohr"
    )


@pytest.mark.parametrize("spin", ["singlet", "triplet"])
def test_cis_gradient_translationally_invariant(spin):
    """Sum of per-atom gradients vanishes (no net force)."""
    Z, coords = H2O
    cis = run_cis(Z, coords, n_states=4, spin=spin)
    g = cis_gradient(Z, coords, cis, state=0, spin=spin)
    assert np.max(np.abs(g.sum(axis=0))) < 1e-8, f"net force {g.sum(axis=0)}"


def test_cis_gradient_open_shell_raises():
    """Open-shell (odd electron) references are not supported by the analytic path."""
    # N atom: 5 valence electrons -> odd.
    with pytest.raises(NotImplementedError):
        cis_gradient([7], [[0.0, 0.0, 0.0]], None, 0)


def test_cis_gradient_singlet_triplet_differ():
    """Singlet and triplet S1 gradients are genuinely different surfaces."""
    Z, coords = H2O
    gs = cis_gradient(Z, coords, run_cis(Z, coords, spin="singlet"), 0, spin="singlet")
    gt = cis_gradient(Z, coords, run_cis(Z, coords, spin="triplet"), 0, spin="triplet")
    assert np.max(np.abs(gs - gt)) > 1e-3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
