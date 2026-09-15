"""Validation of the Green's-function embedding numerical backbone on the
analytic 1D semi-infinite tight-binding substrate (open design question 4
of the adsorbate-embedding feature; see ``handovers/HANDOVER_GF_EMBEDDING.md``).

The 1D semi-infinite chain has closed forms for the surface Green
function, the surface LDOS (a Wigner semicircle), the surface-site
occupation and the embedding self-energy, so every piece of the backbone
is pinned to an analytic target — and, for the occupation, additionally
cross-checked against a fully independent finite-chain diagonalization.

Test surface:

1. Sancho-Rubio decimation reproduces the analytic surface (and bulk)
   Green function to ~1e-10, and converges in a handful of iterations
   (the super-exponential principal-layer renormalization).
2. Surface LDOS matches the analytic semicircle, and the semicircle
   integrates to exactly one state.
3. Complex-contour occupation matches the analytic ``N(E_F)`` — exactly
   0.5 at half filling — and converges geometrically in the node count.
4. Contour band energy ``\\int E rho dE`` matches its closed form.
5. The Ishida energy-linearized embedding potential reproduces the full
   ``Sigma_emb(z)`` to second order in ``(z - z0)``.
6. Lloyd's formula ``Delta_N(E) = -(1/pi) Im ln det(1 - G0 Delta_V)``
   matches a direct integration of the total DOS change for a weak
   (no-bound-state) perturbation, and conserves states.
7. Lloyd's formula counts a bound state pulled out of the band by a
   strong attractive perturbation (a +1 step in ``Delta_N``), while still
   conserving the total state count.
8. Independent cross-check: the end-site occupation of a large finite
   chain (direct diagonalization) converges to the contour occupation.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc.periodic_embedding import (
    EmbeddingPotential,
    EnergyContour,
    dyson_solve,
    lloyd_integrated_dos_change,
    sancho_rubio_surface_gf,
)
from vibeqc.periodic_embedding.models import SemiInfiniteChain1D

# Asymmetric parameters (eps != 0, t != 1) so no accidental symmetry hides
# a sign or branch error.
EPS = 0.2
HOP = 1.1
CHAIN = SemiInfiniteChain1D(onsite=EPS, hop=HOP)


# --- 1. Sancho-Rubio decimation vs analytic surface/bulk GF -------------
@pytest.mark.parametrize(
    "z", [0.30 + 0.50j, -1.20 + 0.80j, 1.50 + 0.20j, 0.20 + 0.05j]
)
def test_sancho_rubio_matches_analytic_surface_gf(z):
    h00 = np.array([[EPS]], dtype=complex)
    h01 = np.array([[HOP]], dtype=complex)
    g_surface, g_bulk = sancho_rubio_surface_gf(h00, h01, z)

    assert g_surface[0, 0] == pytest.approx(CHAIN.surface_gf(z), abs=1e-10)
    assert g_bulk[0, 0] == pytest.approx(CHAIN.bulk_gf(z), abs=1e-10)


def test_sancho_rubio_converges_fast():
    # A handful of doublings is enough even close to the real axis.
    z = EPS + 0.5 * HOP + 0.02j
    h00 = np.array([[EPS]], dtype=complex)
    h01 = np.array([[HOP]], dtype=complex)
    # max_iter=12 -> effective depth 2^12 layers; must already converge.
    g_surface, _ = sancho_rubio_surface_gf(h00, h01, z, max_iter=12)
    assert g_surface[0, 0] == pytest.approx(CHAIN.surface_gf(z), abs=1e-9)


# --- 2. Surface LDOS vs analytic semicircle -----------------------------
def test_ldos_matches_semicircle():
    # Interior of the band, away from the edges where eta-broadening leaks.
    e = np.linspace(CHAIN.band_bottom + 0.3, CHAIN.band_top - 0.3, 60)
    num = CHAIN.ldos(e, eta=1e-5)
    ana = CHAIN.ldos_analytic(e)
    assert np.allclose(num, ana, atol=2e-3)


def test_ldos_normalizes_to_one_state():
    e = np.linspace(CHAIN.band_bottom, CHAIN.band_top, 200_001)
    integral = np.trapezoid(CHAIN.ldos_analytic(e), e)
    assert integral == pytest.approx(1.0, abs=1e-4)


# --- 3. Contour occupation vs analytic N(E_F) ---------------------------
def test_contour_occupation_half_filling_is_exactly_half():
    e_bottom = CHAIN.band_bottom - 0.5
    contour = EnergyContour.semicircle(e_bottom, EPS, n_nodes=64)
    g = CHAIN.surface_gf(contour.nodes)
    assert contour.occupation(g) == pytest.approx(0.5, abs=1e-7)


@pytest.mark.parametrize("e_fermi", [-1.3, -0.4, 0.2, 0.9, 1.7])
def test_contour_occupation_matches_analytic(e_fermi):
    e_bottom = CHAIN.band_bottom - 0.5
    contour = EnergyContour.semicircle(e_bottom, e_fermi, n_nodes=80)
    g = CHAIN.surface_gf(contour.nodes)
    assert contour.occupation(g) == pytest.approx(
        CHAIN.occupation_analytic(e_fermi), abs=1e-6
    )


def test_contour_occupation_converges_in_node_count():
    e_bottom = CHAIN.band_bottom - 0.5
    target = CHAIN.occupation_analytic(EPS)  # 0.5
    errors = []
    for n in (8, 16, 32, 64):
        contour = EnergyContour.semicircle(e_bottom, EPS, n_nodes=n)
        g = CHAIN.surface_gf(contour.nodes)
        errors.append(abs(contour.occupation(g) - target))
    # Monotone improvement and machine-precision-class at 64 nodes.
    assert errors[0] > errors[-1]
    assert errors[-1] < 1e-7


# --- 4. Contour band energy ---------------------------------------------
@pytest.mark.parametrize("e_fermi", [-0.4, 0.2, 1.0])
def test_contour_band_energy_matches_analytic(e_fermi):
    e_bottom = CHAIN.band_bottom - 0.5
    contour = EnergyContour.semicircle(e_bottom, e_fermi, n_nodes=96)
    g = CHAIN.surface_gf(contour.nodes)
    assert contour.band_energy(g) == pytest.approx(
        CHAIN.band_energy_analytic(e_fermi), abs=1e-5
    )


# --- 5. Ishida energy linearization of the embedding potential ----------
def test_embedding_potential_linearization_is_second_order():
    sigma = EmbeddingPotential(CHAIN.embedding_self_energy, label="1d-chain")
    z0 = EPS + 0.5j
    lin = sigma.linearize(z0)

    def err(dz):
        z = z0 + dz
        return abs(complex(lin.at(z)[0, 0]) - CHAIN.embedding_self_energy(z))

    e_small, e_big = err(0.01), err(0.02)
    assert e_small < 1e-3
    # Doubling dz quadruples the error (second-order remainder).
    assert e_big / e_small == pytest.approx(4.0, rel=0.25)


# --- 6. Lloyd's formula, weak perturbation (no bound state) -------------
def test_lloyd_matches_direct_integration_weak_perturbation():
    delta = 0.5 * HOP  # |delta| < |t| -> no bound state
    eta = 1e-3
    energies = np.linspace(CHAIN.band_bottom - 1.0, CHAIN.band_top + 1.0, 6000)
    z = energies + 1j * eta
    g0 = CHAIN.surface_gf(z)

    dn_lloyd = lloyd_integrated_dos_change(g0, delta)

    # Independent path: integrate the *total* DOS change. For a single-site
    # perturbation Tr[G_pert - G0] = -delta g0' / (1 - g0 delta), with
    # g0' the analytic-continued derivative (central difference here).
    dz = 1e-4
    g0p = (CHAIN.surface_gf(z + dz) - CHAIN.surface_gf(z - dz)) / (2.0 * dz)
    tr_dg = -delta * g0p / (1.0 - g0 * delta)
    d_rho = -(1.0 / np.pi) * np.imag(tr_dg)
    dn_direct = np.concatenate(
        [[0.0], np.cumsum(0.5 * (d_rho[1:] + d_rho[:-1]) * np.diff(energies))]
    )

    assert np.max(np.abs(dn_lloyd - dn_direct)) < 5e-3
    # States conserved: Delta_N returns to zero above the band.
    assert abs(dn_lloyd[-1]) < 1e-2


def test_dyson_solve_matches_scalar_formula():
    delta = 0.5 * HOP
    z = EPS + 0.3 + 0.1j
    g0 = CHAIN.surface_gf(z)
    g = dyson_solve(np.array([[g0]]), np.array([[delta]]))
    assert g[0, 0] == pytest.approx(g0 / (1.0 - g0 * delta), abs=1e-12)


# --- 7. Lloyd's formula, strong attractive perturbation (bound state) ---
def test_lloyd_counts_bound_state():
    delta = -1.5 * HOP  # |delta| > |t| -> one bound state below the band
    e_b = CHAIN.bound_state_energy(delta)
    assert e_b is not None and e_b < CHAIN.band_bottom

    eta = 1e-3
    energies = np.linspace(e_b - 1.0, CHAIN.band_top + 1.0, 12000)
    g0 = CHAIN.surface_gf(energies + 1j * eta)
    dn = lloyd_integrated_dos_change(g0, delta)

    # Well below the bound state: nothing displaced yet.
    i_below_eb = np.searchsorted(energies, e_b - 0.3)
    # Just below the band bottom but above the bound state: the bound
    # state has been counted -> a clean step of one state.
    i_below_band = np.searchsorted(energies, CHAIN.band_bottom - 0.05)
    step = dn[i_below_band] - dn[i_below_eb]
    assert abs(step) == pytest.approx(1.0, abs=0.1)

    # Total state count conserved across the whole spectrum.
    assert abs(dn[-1]) < 5e-2


# --- 8. Independent finite-chain diagonalization cross-check ------------
@pytest.mark.parametrize("e_fermi", [-0.4, 0.2, 1.0])
def test_finite_chain_end_site_occupation(e_fermi):
    n = 2000
    h = CHAIN.finite_chain_hamiltonian(n)
    evals, evecs = np.linalg.eigh(h)
    occ_mask = evals <= e_fermi
    # Occupation projected onto the end (surface) site 0.
    end_site_weight = np.abs(evecs[0, occ_mask]) ** 2
    occ_finite = float(end_site_weight.sum())

    e_bottom = CHAIN.band_bottom - 0.5
    contour = EnergyContour.semicircle(e_bottom, e_fermi, n_nodes=80)
    occ_contour = contour.occupation(CHAIN.surface_gf(contour.nodes))

    assert occ_finite == pytest.approx(CHAIN.occupation_analytic(e_fermi), abs=1.5e-2)
    assert occ_contour == pytest.approx(occ_finite, abs=1.5e-2)
