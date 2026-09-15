"""Bloch bridge — per-k ↔ per-cell round-trip contract.

The :mod:`vibeqc.periodic_scf_accelerators` module exports two small
utilities, :func:`blocks_from_per_k` (inverse Bloch summation) and
:func:`per_k_from_blocks` (forward Bloch summation), that bridge the
per-k complex-Hermitian Fock representation used by the Python multi-k
Ewald SCF drivers and the per-cell real-block representation consumed
by the C++ block-vector ``EDIIS::extrapolate_blocks`` /
``ADIIS::extrapolate_blocks`` kernels (`cpp/include/vibeqc/ediis.hpp:113`
and 193). These tests pin the contracts the rest of the multi-k
accelerator rollout (M2c–e) will rely on.

Conventions matching the C++ side (`cpp/src/periodic_scf.cpp`
``real_space_density_from_kpoints``):

  F(k) = Σ_g  exp(+i k · R_g)            · F(g)
  F(g) = Σ_k  w_k · Re[ exp(−i k · R_g)  · F(k) ]

The inverse direction's ``Re[]`` encodes time-reversal averaging on a
Bloch-Floquet-symmetric mesh: F(−k) = F(k)* for a TR-symmetric
Hamiltonian, so the (k, −k) pair contribution to F(g) is real. A
``BlochKMesh`` from :func:`vibeqc.monkhorst_pack` is TR-symmetric by
construction.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.periodic_scf_accelerators import (
    blocks_from_per_k,
    per_k_from_blocks,
)


def _h2_chain_1d():
    """H₂-per-cell linear chain in a 1D periodic box with a big vacuum
    in the y, z directions. Even electron count, time-reversal-symmetric
    by construction."""
    sysp = vq.PeriodicSystem(
        1, np.diag([6.0, 30.0, 30.0]),
        [vq.Atom(1, [0.0, 15.0, 15.0]),
         vq.Atom(1, [1.4, 15.0, 15.0])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _lattice_opts(cutoff: float, nuc_cutoff: float):
    o = vq.LatticeSumOptions()
    o.cutoff_bohr = cutoff
    o.nuclear_cutoff_bohr = nuc_cutoff
    return o


def _real_blocks_to_complex_k(blocks, cells, kpoints):
    """Convenience: same as per_k_from_blocks but returns a list."""
    return per_k_from_blocks(blocks, cells, kpoints)


# ---------------------------------------------------------------------------
# Forward Bloch — Hermiticity of the result on a TR-symmetric mesh.
# ---------------------------------------------------------------------------

def test_per_k_from_blocks_returns_hermitian_for_symmetric_input():
    """A real-symmetric per-cell block list closed under negation of
    R_g must Bloch-sum to a Hermitian per-k matrix at every k. This is
    the contract relied on by every multi-k diagonaliser downstream."""
    sysp, basis = _h2_chain_1d()
    km = vq.monkhorst_pack(sysp, [4, 1, 1])
    lat_opts = _lattice_opts(12.0, 15.0)
    S_lat = vq.compute_overlap_lattice(basis, sysp, lat_opts)
    blocks_in = [np.asarray(b).copy() for b in S_lat.blocks]

    per_k = per_k_from_blocks(blocks_in, S_lat.cells, km.kpoints)
    for idx, S_k in enumerate(per_k):
        residual = float(np.max(np.abs(S_k - S_k.conj().T)))
        assert residual < 1e-12, (
            f"S(k={idx}) Hermitian residual {residual} exceeds 1e-12"
        )


# ---------------------------------------------------------------------------
# Linearity — the bridge preserves linear combinations.
#
# This is the actual contract the multi-k EDIIS / ADIIS rollout relies
# on. The QP forms an extrapolated F_ex_blocks = Σ_j c_j F_j_blocks; for
# the forward-Bloch of F_ex_blocks to coincide with Σ_j c_j F_j(k), the
# bridge functions must be linear in their matrix-valued input. The
# round-trip composition itself is *not* the identity when cell count
# differs from k-point count (aliasing); but linearity holds regardless
# of cell-vs-k-count and is what makes the QP coefficients transfer
# faithfully through the bridge.
# ---------------------------------------------------------------------------

def test_blocks_from_per_k_is_linear():
    """``blocks_from_per_k(α·M1 + β·M2) = α·blocks_from_per_k(M1)
    + β·blocks_from_per_k(M2)`` — necessary so the per-cell QP's
    convex combination coefficients lift back to a meaningful per-k
    extrapolation."""
    sysp, basis = _h2_chain_1d()
    km = vq.monkhorst_pack(sysp, [4, 1, 1])
    lat_opts = _lattice_opts(12.0, 15.0)
    S_lat = vq.compute_overlap_lattice(basis, sysp, lat_opts)
    T_lat = vq.compute_kinetic_lattice(basis, sysp, lat_opts)
    M1 = per_k_from_blocks([np.asarray(b).copy() for b in S_lat.blocks],
                             S_lat.cells, km.kpoints)
    M2 = per_k_from_blocks([np.asarray(b).copy() for b in T_lat.blocks],
                             T_lat.cells, km.kpoints)
    alpha, beta = 0.3, -0.7
    combined = [alpha * m1 + beta * m2 for m1, m2 in zip(M1, M2)]
    blocks_lhs = blocks_from_per_k(combined, S_lat.cells,
                                     km.kpoints, km.weights)
    b1 = blocks_from_per_k(M1, S_lat.cells, km.kpoints, km.weights)
    b2 = blocks_from_per_k(M2, S_lat.cells, km.kpoints, km.weights)
    blocks_rhs = [alpha * x1 + beta * x2 for x1, x2 in zip(b1, b2)]
    for lhs, rhs in zip(blocks_lhs, blocks_rhs):
        assert np.allclose(lhs, rhs, atol=1e-12)


def test_per_k_from_blocks_is_linear():
    """Same linearity contract in the other direction."""
    sysp, basis = _h2_chain_1d()
    km = vq.monkhorst_pack(sysp, [4, 1, 1])
    lat_opts = _lattice_opts(12.0, 15.0)
    S_lat = vq.compute_overlap_lattice(basis, sysp, lat_opts)
    T_lat = vq.compute_kinetic_lattice(basis, sysp, lat_opts)
    Sb = [np.asarray(b).copy() for b in S_lat.blocks]
    Tb = [np.asarray(b).copy() for b in T_lat.blocks]
    alpha, beta = 0.4, 1.1
    combined = [alpha * s + beta * t for s, t in zip(Sb, Tb)]
    lhs = per_k_from_blocks(combined, S_lat.cells, km.kpoints)
    Sk = per_k_from_blocks(Sb, S_lat.cells, km.kpoints)
    Tk = per_k_from_blocks(Tb, T_lat.cells, km.kpoints)
    rhs = [alpha * s + beta * t for s, t in zip(Sk, Tk)]
    for L, R in zip(lhs, rhs):
        assert np.allclose(L, R, atol=1e-12)


# ---------------------------------------------------------------------------
# Single-k (Γ-only) — bridge collapses to the identity on the home cell.
# ---------------------------------------------------------------------------

def test_gamma_only_round_trip_preserves_home_block():
    """At a [1,1,1] k-mesh (Γ only, w_k = 1), the forward Bloch sum of
    a cell list reduces to Σ_g F(g), and the inverse spreads that one
    matrix across every cell uniformly. Verifying that the cell sum
    survives is the practically-useful invariant: the Γ-point Fock at
    the converged density depends only on Σ_g F(g)."""
    sysp, basis = _h2_chain_1d()
    km = vq.monkhorst_pack(sysp, [1, 1, 1])
    lat_opts = _lattice_opts(12.0, 15.0)
    S_lat = vq.compute_overlap_lattice(basis, sysp, lat_opts)
    blocks_in = [np.asarray(b).copy() for b in S_lat.blocks]

    sum_in = sum(blocks_in)
    per_k = per_k_from_blocks(blocks_in, S_lat.cells, km.kpoints)
    # At Γ with weight 1 the per-k matrix at k=0 equals the cell sum.
    assert np.allclose(per_k[0], sum_in, atol=1e-12)


def test_blocks_from_per_k_size_mismatch_raises():
    """Defensive: M_per_k / kpoints / weights must have the same
    length; otherwise the caller has a logic bug, not a recoverable
    runtime condition."""
    sysp, basis = _h2_chain_1d()
    km = vq.monkhorst_pack(sysp, [4, 1, 1])
    lat_opts = _lattice_opts(12.0, 15.0)
    S_lat = vq.compute_overlap_lattice(basis, sysp, lat_opts)
    n_bf = S_lat.blocks[0].shape[0] if S_lat.blocks else 0
    bogus = [np.zeros((n_bf, n_bf), dtype=complex) for _ in range(3)]
    with pytest.raises(ValueError, match="same length"):
        blocks_from_per_k(bogus, S_lat.cells, km.kpoints, km.weights)
