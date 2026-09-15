"""RIJCOSX analytic gradient vs finite differences of the RIJCOSX energy.

Reproducer for the RIJCOSX analytic-gradient parity defect (2026-07-29):
on glycine/def2-TZVP the RIJCOSX SCF energy matches ORCA 6.1.1 to
0.2648 mHa, but the analytic nuclear gradient disagrees componentwise by
RMS 1.603e-2 / max 7.151e-2 Ha/bohr (|g| 0.1090 vs 0.0870), with the
per-atom |g_vqc|/|g_orca| ratio tracking nuclear charge (N 1.47, C 2.12,
C 1.39, O 1.08, O 1.59 vs 1.05-1.14 for H) — the fingerprint of a
missing or incomplete derivative term that grows with core density.

The decisive oracle here is finite differences of vibe-qc's OWN RIJCOSX
total energy with the SAME options at every displaced geometry: the
energy is known correct against ORCA, so any analytic-vs-FD gap on the
same code isolates the incomplete analytic term without ORCA in the
loop.  Central differences at h = 1e-4 bohr have a truncation floor of
~1e-7 Ha/bohr.

Post-fix state (2026-07-29, mixed-density bilinear form + overlap-fit
Q response landed; see cpp/src/cosx.cpp and the worktree EVIDENCE):

* ``test_cosx_k_gradient_matches_frozen_surface_fd`` is the SHARP
  formula gate: the analytic K contribution must match FD of the
  exchange energy at fixed density with a frozen (fixed-in-space) grid
  to 1e-5 Ha/bohr (measured ~1e-7; the pre-fix formula failed this at
  2.4e-3 on the GridX tier-2 grid).
* ``test_rijcosx_rhf_gradient_matches_fd`` gates the TOTAL surface.
  Its bands are the measured, documented residuals that remain by
  design: the FITTED-variant Fock is not the functional derivative of
  the reported energy (Hellmann-Feynman response, ~1.3e-3 on
  H2O/def2-tzvp; needs an SCF-side self-adjoint fitted K — maintainer
  proposal pending) plus the combined grid-point-motion/Becke-weight
  neglect (~2e-4 typical, 3e-3 measured worst case).  Do NOT widen
  these bands to absorb a new regression — tighten them toward 1e-4
  when the stationarity fix lands.

Cases deliberately mirror the failing route (density_fit + cosx with a
def2 -jk aux basis) on small systems: H2O/def2-svp (fast smoke) and
H2O/def2-tzvp (same basis family as the failing glycine case; triggers
the same COSX grid-cardinality resolution).
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import (
    Atom,
    BasisSet,
    GradientOptions,
    Molecule,
    RHFOptions,
    SCFAccelerator,
    compute_gradient,
    run_rhf,
)

_A = 1.8897259886  # angstrom -> bohr

H2O = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.793353 * _A, -0.613510 * _A]),
    (1, [0.0, -0.793353 * _A, -0.613510 * _A]),
]

#: (orbital basis, aux basis, total-surface tolerance). The tolerances
#: are 2x the measured post-fix residual (svp 6.5e-4, tzvp 1.43e-3 on
#: 2026-07-29), which is the characterized response + grid/weight
#: neglect documented in the module docstring — NOT a free parameter.
FD_CASES = [
    ("def2-svp", "def2-svp-jk", 1.5e-3),
    ("def2-tzvp", "def2-tzvp-jk", 3.0e-3),
]


def _rijcosx_rhf_at(positions, atom_Zs, basis_name, aux):
    mol = Molecule([Atom(int(Z), list(p)) for Z, p in zip(atom_Zs, positions)])
    basis = BasisSet(mol, basis_name)
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    opts.scf_accelerator = SCFAccelerator.EDIIS_DIIS
    opts.max_iter = 200
    opts.density_fit = True
    opts.aux_basis = aux
    opts.cosx = True
    return mol, basis, run_rhf(mol, basis, opts)


def _fd_total_energy(energy_at, positions, h=1e-4):
    n_atoms = len(positions)
    grad = np.zeros((n_atoms, 3))
    for A in range(n_atoms):
        for c in range(3):
            pp = [list(p) for p in positions]; pp[A][c] += h
            pm = [list(p) for p in positions]; pm[A][c] -= h
            grad[A, c] = (energy_at(pp) - energy_at(pm)) / (2 * h)
    return grad


@pytest.mark.parametrize(
    "orb,aux,tol", FD_CASES, ids=[f"H2O-{o}" for o, _, _ in FD_CASES],
)
def test_rijcosx_rhf_gradient_matches_fd(orb, aux, tol):
    """RIJCOSX analytic gradient must match FD of the RIJCOSX energy.

    Each FD step re-runs the full RIJCOSX SCF (DF-J + COSX-K) at the
    displaced geometry.  Because the COSX grid is atom-centered, the FD
    gradient automatically contains the grid-point-motion, grid-weight,
    and overlap-fit (Q) response contributions AND the density-response
    (non-stationarity) term; the per-case tolerances are the measured,
    documented residual bands (see the module docstring) — the pre-fix
    defect showed up here as a formula-family error on top of them.
    """
    Zs = [Z for Z, _ in H2O]
    pos = [list(xyz) for _, xyz in H2O]
    mol, basis, rhf = _rijcosx_rhf_at(pos, Zs, orb, aux)
    assert rhf.converged

    go = GradientOptions()
    go.density_fit = True
    go.aux_basis = aux
    go.cosx = True
    g_an = np.array(compute_gradient(mol, basis, rhf, go))

    def energy_at(positions):
        _, _, hf = _rijcosx_rhf_at(positions, Zs, orb, aux)
        assert hf.converged
        return hf.energy

    g_fd = _fd_total_energy(energy_at, pos)
    delta = np.abs(g_an - g_fd).max()
    assert delta < tol, (
        f"H2O/{orb}/{aux}: RIJCOSX analytic vs FD gradient max abs diff "
        f"= {delta:.3e} Ha/bohr (band {tol:.1e})\n"
        f"analytic =\n{g_an}\nfd =\n{g_fd}"
    )


def test_cosx_k_gradient_matches_frozen_surface_fd():
    """Sharp formula gate: analytic K gradient vs frozen-grid FD at fixed D.

    Freezes the converged density AND the grid (points + weights fixed
    in space; Q rebuilt from the displaced basis on the frozen grid, so
    the overlap-fit response is exercised) and finite-differences
    E_K = -(1/4) tr(D K_cosx).  At this level the analytic
    ``compute_cosx_k_gradient_contribution`` is an EXACT derivative:
    measured agreement ~1e-7 Ha/bohr.  The pre-2026-07-29 formula
    (fitted density on both sides of the bilinear form, frozen Q)
    failed this at 2.4e-3 on the GridX tier-2 grid — the root cause of
    the glycine/def2-TZVP ORCA-parity defect.

    Uses the tier-2 GridX grid deliberately: that is the grid the SCF's
    AUTO default converges on, and where the pre-fix error was largest.
    """
    from vibeqc._vibeqc_core import (
        build_grid,
        compute_cosx_k,
        compute_cosx_k_gradient_contribution,
        cosx_grid_options_for_level,
    )

    Zs = [Z for Z, _ in H2O]
    pos = [list(xyz) for _, xyz in H2O]
    mol0, basis0, rhf = _rijcosx_rhf_at(pos, Zs, "def2-svp", "def2-svp-jk")
    assert rhf.converged
    D = np.asarray(rhf.density)

    grid0 = build_grid(mol0, cosx_grid_options_for_level(2))

    def e_k(positions):
        mol = Molecule(
            [Atom(int(Z), list(p)) for Z, p in zip(Zs, positions)]
        )
        basis = BasisSet(mol, "def2-svp")
        K = compute_cosx_k(basis, D, grid0)  # frozen grid, Q rebuilt
        return -0.25 * float((D * np.asarray(K)).sum())

    g_an = np.asarray(compute_cosx_k_gradient_contribution(
        mol0, basis0, D, grid0, 1.0, np.empty((0, 0)), True))

    h = 2e-4
    g_fd = np.zeros((len(pos), 3))
    for a in range(len(pos)):
        for c in range(3):
            pp = [list(p) for p in pos]; pp[a][c] += h
            pm = [list(p) for p in pos]; pm[a][c] -= h
            g_fd[a, c] = (e_k(pp) - e_k(pm)) / (2 * h)

    delta = np.abs(g_an - g_fd).max()
    assert delta < 1e-5, (
        f"COSX K gradient vs frozen-surface FD max abs diff = "
        f"{delta:.3e} Ha/bohr (exact-derivative gate; measured ~1e-7)\n"
        f"analytic =\n{g_an}\nfd =\n{g_fd}"
    )
