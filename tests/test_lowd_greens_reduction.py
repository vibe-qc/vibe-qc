"""M3 reduction gate (scalar level): the low-D kernels reduce to the 3-D `v_E`.

The hard, dependency-free check of the mixed-boundary construction
(`docs/aiccm2026dev_a_lowd_greens.md` §5): the 3-D torus kernel is the
transverse-periodic completion of the mixed-boundary kernel, so each low-D
Green's function is the **large-transverse-period limit** of the (already
validated) 3-D neutralized Ewald potential ``v_E^{3D}``, up to a gauge constant:

    G^{1D}(ρ,z) = lim_{D⊥→∞} [ v_E^{3D}(ρ,0,z; z-period L, transverse period D⊥) − c(D⊥) ]
    G^{2D}(ρ⃗,z) = lim_{L_z→∞} [ v_E^{3D}(ρ⃗,z; in-plane lattice, z-period L_z) − c(L_z) ]

The reference ``v_E^{3D}`` is an independent analytic Ewald sum (no GDF, no
external code), self-validated here by η-invariance and by its self-potential
reproducing the lattice Madelung constant. Gate = the residual (after removing
the single gauge constant) **decreases monotonically** as the transverse period
grows — the kernel-level precursor of the four-center M3 (`g_eff^{(d)}` re-
periodized == ``ccm_eri_neutral``), mirroring the passing slab-`J` reduction
test ``test_j_matches_3d_vacuum_limit_up_to_gauge``.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erfc

from vibeqc import Atom, PeriodicSystem
from vibeqc.madelung import madelung_constant_for_cell
from vibeqc.periodic.lowd_greens import slab_greens, wire_greens, wire_self_energy


def _ewald_pot_3d(r, lat, eta, nr, ngrid):
    """Neutralized 3-D Ewald potential at displacement ``r`` (vectorized).

    ``nr`` / ``ngrid`` are per-axis tuples of half-widths for the real / reciprocal
    sums; η-invariance is the convergence check.
    """
    r = np.asarray(r, float)
    lat = np.asarray(lat, float)
    V = abs(np.linalg.det(lat))
    b = 2.0 * np.pi * np.linalg.inv(lat).T  # reciprocal rows

    ax = [np.arange(-n, n + 1) for n in nr]
    I, J, K = np.meshgrid(*ax, indexing="ij")
    T = (I[..., None] * lat[0] + J[..., None] * lat[1] + K[..., None] * lat[2])
    d = np.linalg.norm(r + T, axis=-1).ravel()
    d = d[d > 1e-12]
    real = np.sum(erfc(eta * d) / d)

    gx = [np.arange(-n, n + 1) for n in ngrid]
    GI, GJ, GK = np.meshgrid(*gx, indexing="ij")
    G = (GI[..., None] * b[0] + GJ[..., None] * b[1] + GK[..., None] * b[2]).reshape(-1, 3)
    G2 = np.einsum("ij,ij->i", G, G)
    nz = G2 > 1e-12
    G2, G = G2[nz], G[nz]
    recip = np.sum(np.exp(-G2 / (4 * eta**2)) / G2 * np.cos(G @ r))
    return float(real + (4.0 * np.pi / V) * recip - np.pi / (eta**2 * V))


# --------------------------------------------------------------------------- #
# The reference Ewald potential is itself validated (η-invariance + Madelung).
# --------------------------------------------------------------------------- #
def test_ewald_reference_eta_invariant_and_madelung():
    lat = np.diag([5.0, 5.0, 5.0])
    r = np.array([1.3, 0.7, 2.1])
    vals = [_ewald_pot_3d(r, lat, eta, (7, 7, 7), (7, 7, 7)) for eta in (0.3, 0.5, 0.8)]
    assert max(vals) - min(vals) < 1e-7          # η-invariant
    # Self-potential v_E(r→0) − 1/r = −ξ (the lattice Madelung constant).
    xi = madelung_constant_for_cell(
        PeriodicSystem(3, lat, [Atom(1, [0, 0, 0])], charge=0, multiplicity=2))
    eps = 1e-4
    v_self = _ewald_pot_3d([eps, 0, 0], lat, 0.5, (9, 9, 9), (9, 9, 9)) - 1.0 / eps
    assert v_self == pytest.approx(-xi, abs=1e-4)


# --------------------------------------------------------------------------- #
# Wire: G^{1D} == large-transverse-period limit of v_E^{3D} (fast, ~1/D²).
# --------------------------------------------------------------------------- #
def test_wire_reduces_to_3d_ewald():
    L = 2.0
    pts = [(0.5, 0.0), (0.7, 0.4), (0.6, 0.9), (0.9, 0.55)]
    g1d = np.array([wire_greens(ro, z, period=L, n_recip=400) for ro, z in pts])

    resids = []
    for D in (8.0, 16.0, 32.0):
        lat = np.diag([D, D, L])
        eta = 0.9                                 # tied to the short axis L
        nr = (2, 2, int(np.ceil(4.0 / eta / L)) + 1)
        ng = (int(np.ceil(4.0 * eta / (2 * np.pi / D))) + 1,) * 2 + (2,)
        v3 = np.array([_ewald_pot_3d([ro, 0.0, z], lat, eta, nr, ng) for ro, z in pts])
        diff = v3 - g1d
        resids.append(float(np.max(np.abs(diff - diff[0]))))  # remove gauge const

    assert resids[0] > resids[1] > resids[2], f"not converging: {resids}"
    assert resids[-1] < 2e-3                      # ~1/D²: well-converged by D=32


# --------------------------------------------------------------------------- #
# Slab: G^{2D} == large-z-period limit of v_E^{3D} (monotone, ~1/L_z).
# --------------------------------------------------------------------------- #
def test_wire_self_energy_anchors_to_3d_madelung():
    """The wire self-energy ξ_wire(L) is the large-transverse-period limit of the
    validated 3-D Ewald self-potential, up to the **exact** line-vs-uniform-
    background gauge ``(2/L)·ln(D⊥)``.

    The 3-D self-potential of a charge in the cell ``diag([D⊥,D⊥,L])`` is
    ``v_E^{3D}(r→0) − 1/r = −madelung``; as D⊥→∞ the transverse images recede and
    it must approach ξ_wire(L) up to the conditional-channel gauge, which here is
    the line-vs-sheet/uniform difference and is *predicted* to be exactly
    ``(2/L)·ln(D⊥)``. So ``−madelung(D⊥) − ξ_wire − (2/L)·ln(D⊥)`` is
    **D⊥-independent** (a constant). This ties the new low-D Madelung-analog
    constant to ``madelung_constant_for_cell`` (itself bit-exact vs PySCF).
    """
    L = 2.0
    xw = wire_self_energy(L)
    consts = []
    for D in (8.0, 16.0, 32.0, 64.0):
        sysc = PeriodicSystem(3, np.diag([D, D, L]),
                              [Atom(2, [0, 0, 0])], charge=0, multiplicity=1)
        xi3 = -madelung_constant_for_cell(sysc)
        consts.append(xi3 - xw - (2.0 / L) * np.log(D))
    # D⊥-independent ⇒ the residual gauge is exactly (2/L)·ln(D⊥), nothing else.
    assert max(consts) - min(consts) < 2e-3


def test_slab_reduces_to_3d_ewald():
    a = 2.0
    lat2d = np.array([[a, 0.0], [0.0, a]])
    pts = [(0.3, 0.2, 0.5), (0.5, 0.1, 0.8), (0.2, 0.4, 1.1), (0.6, 0.3, 0.6)]
    g2d = np.array([slab_greens([x, y], z, lattice2d=lat2d, n_shell=60) for x, y, z in pts])

    resids = []
    for Lz in (8.0, 16.0, 32.0):
        lat = np.diag([a, a, Lz])
        eta = 0.9                                 # tied to the short in-plane axis a
        nr = (int(np.ceil(4.0 / eta / a)) + 1,) * 2 + (2,)
        ng = (2, 2, int(np.ceil(4.0 * eta / (2 * np.pi / Lz))) + 1)
        v3 = np.array([_ewald_pot_3d([x, y, z], lat, eta, nr, ng) for x, y, z in pts])
        diff = v3 - g2d
        resids.append(float(np.max(np.abs(diff - diff[0]))))

    # Slab reduction is the slow (~1/L_z) limit; the gate is monotone convergence.
    assert resids[0] > resids[1] > resids[2], f"not converging: {resids}"
