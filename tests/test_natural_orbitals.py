"""Natural orbitals + idempotency diagnostic.

Pin both the algebraic identities (RHF gives integer occupations and
spans the same occupied subspace as the canonical MOs) and the physical
content (triplet O₂ has two unpaired-spin natural orbitals).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq


# ---------------------------------------------------------------------------
# Closed-shell RHF: NOs are integer-occupied
# ---------------------------------------------------------------------------

@pytest.fixture
def h2o_rhf():
    mol = vq.Molecule.from_xyz(
        str(Path(__file__).parent.parent / "examples" / "h2o.xyz")
    )
    basis = vq.BasisSet(mol, "sto-3g")
    res = vq.run_rhf(mol, basis)
    return mol, basis, res


def test_natural_orbitals_rhf_integer_occupations(h2o_rhf):
    mol, basis, res = h2o_rhf
    no = vq.natural_orbitals(res, basis)
    assert no.kind == "rhf"
    n_occ = mol.n_electrons() // 2

    # First n_occ NO occupations are exactly 2; the rest are exactly 0
    # (single-determinant RHF density is idempotent).
    assert np.allclose(no.occupations[:n_occ], 2.0, atol=1e-10)
    assert np.allclose(no.occupations[n_occ:], 0.0, atol=1e-10)

    # Sum of occupations equals the electron count.
    assert no.n_electrons == pytest.approx(float(mol.n_electrons()), abs=1e-9)
    # Idempotency diagnostic vanishes for a single-determinant state.
    assert abs(vq.idempotency_deviation(no)) < 1e-9


def test_natural_orbitals_descending(h2o_rhf):
    mol, basis, res = h2o_rhf
    no = vq.natural_orbitals(res, basis)
    diffs = np.diff(no.occupations)
    # Strictly non-increasing (we sort descending; ties are allowed).
    assert np.all(diffs <= 1e-10)


def test_natural_orbitals_s_normalised(h2o_rhf):
    """``C^T S C = I`` — the AO-basis NO transform is S-orthonormal."""
    mol, basis, res = h2o_rhf
    no = vq.natural_orbitals(res, basis)
    S = np.asarray(vq.compute_overlap(basis))
    G = no.coefficients.T @ S @ no.coefficients
    assert np.allclose(G, np.eye(G.shape[0]), atol=1e-10)


def test_natural_orbitals_density_round_trip(h2o_rhf):
    """``Σ_i n_i c_i c_i^T = D`` — the spectral form rebuilds the
    original density matrix."""
    mol, basis, res = h2o_rhf
    no = vq.natural_orbitals(res, basis)
    D = np.asarray(res.density)
    D_rebuilt = (no.coefficients * no.occupations) @ no.coefficients.T
    assert np.allclose(D_rebuilt, D, atol=1e-10)


def test_natural_orbitals_span_matches_canonical_occupied(h2o_rhf):
    """Closed-shell RHF: the natural-orbital occupied subspace is the
    same as the canonical MO occupied subspace (NOs differ from MOs
    only by an orthogonal rotation within the occupied / virtual block).

    Witness: the ``n_occ × n_occ`` overlap of the two bases on the
    occupied side must be unitary (its singular values are all 1).
    """
    mol, basis, res = h2o_rhf
    no = vq.natural_orbitals(res, basis)
    n_occ = mol.n_electrons() // 2
    C_mo = np.asarray(res.mo_coeffs)
    C_no = no.coefficients
    S = np.asarray(vq.compute_overlap(basis))
    overlap = C_no[:, :n_occ].T @ S @ C_mo[:, :n_occ]    # (n_occ, n_occ)
    sv = np.linalg.svd(overlap, compute_uv=False)
    assert np.allclose(sv, 1.0, atol=1e-9)


def test_natural_orbitals_auto_picks_rhf(h2o_rhf):
    _, basis, res = h2o_rhf
    no = vq.natural_orbitals(res, basis, kind="auto")
    assert no.kind == "rhf"


# ---------------------------------------------------------------------------
# Open-shell UHF: triplet O₂ has 2 unpaired electrons
# ---------------------------------------------------------------------------

@pytest.fixture
def o2_triplet_uhf():
    mol = vq.Molecule(
        [vq.Atom(8, [0, 0, 0]), vq.Atom(8, [0, 0, 2.28])],
        multiplicity=3,
    )
    basis = vq.BasisSet(mol, "sto-3g")
    res = vq.run_uhf(mol, basis)
    assert res.converged
    return mol, basis, res


def test_uhf_total_no_sum_rule(o2_triplet_uhf):
    mol, basis, res = o2_triplet_uhf
    no = vq.natural_orbitals(res, basis, kind="uhf-total")
    assert no.kind == "uhf-total"
    assert no.n_electrons == pytest.approx(float(mol.n_electrons()), abs=1e-9)
    # Occupations bracketed in [0, 2].
    assert no.occupations.max() <= 2.0 + 1e-10
    assert no.occupations.min() >= -1e-10


def test_uhf_alpha_beta_no_sum_rules(o2_triplet_uhf):
    mol, basis, res = o2_triplet_uhf
    no_a = vq.natural_orbitals(res, basis, kind="uhf-alpha")
    no_b = vq.natural_orbitals(res, basis, kind="uhf-beta")
    n_alpha = (mol.n_electrons() + (mol.multiplicity - 1)) // 2
    n_beta = (mol.n_electrons() - (mol.multiplicity - 1)) // 2
    assert no_a.n_electrons == pytest.approx(float(n_alpha), abs=1e-9)
    assert no_b.n_electrons == pytest.approx(float(n_beta), abs=1e-9)
    # Per-spin densities are idempotent in UHF (each spin channel is a
    # single Slater determinant), so per-spin NO occupations are
    # exactly {0, 1}.
    assert vq.idempotency_deviation(no_a) == pytest.approx(0.0, abs=1e-9)
    assert vq.idempotency_deviation(no_b) == pytest.approx(0.0, abs=1e-9)


def test_uhf_spin_no_sum_is_2sz(o2_triplet_uhf):
    mol, basis, res = o2_triplet_uhf
    no = vq.natural_orbitals(res, basis, kind="uhf-spin")
    two_sz = mol.multiplicity - 1   # 2 for triplet
    assert no.n_electrons == pytest.approx(float(two_sz), abs=1e-9)
    # The two largest |occupations| are very close to ±1 — those are the
    # genuinely unpaired electrons.
    biggest = np.sort(np.abs(no.occupations))[::-1]
    assert biggest[0] == pytest.approx(1.0, abs=5e-3)
    assert biggest[1] == pytest.approx(1.0, abs=5e-3)


def test_uhf_total_idempotency_reflects_unpaired(o2_triplet_uhf):
    """Σ n_i (2 − n_i)/2 ≈ N_unpaired/2 × 2 = N_unpaired for a system
    that's "essentially two unpaired electrons" on top of an otherwise
    closed-shell core. For O₂ triplet UHF/STO-3G the diagnostic comes
    in just above 1 (the σ pair contributes a small spin contamination
    too)."""
    _, basis, res = o2_triplet_uhf
    no = vq.natural_orbitals(res, basis, kind="uhf-total")
    delta = vq.idempotency_deviation(no)
    # Allow a generous bracket; the exact value is basis- and
    # convergence-dependent.
    assert 0.5 < delta < 2.0


def test_uhf_auto_picks_total(o2_triplet_uhf):
    _, basis, res = o2_triplet_uhf
    no = vq.natural_orbitals(res, basis, kind="auto")
    assert no.kind == "uhf-total"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_rhf_kind_on_uhf_raises(o2_triplet_uhf):
    _, basis, res = o2_triplet_uhf
    with pytest.raises(ValueError, match="closed-shell"):
        vq.natural_orbitals(res, basis, kind="rhf")


def test_uhf_kinds_on_rhf_raise(h2o_rhf):
    _, basis, res = h2o_rhf
    for kind in ("uhf-total", "uhf-alpha", "uhf-beta", "uhf-spin"):
        with pytest.raises(ValueError, match="open-shell"):
            vq.natural_orbitals(res, basis, kind=kind)


def test_unknown_kind_raises(h2o_rhf):
    _, basis, res = h2o_rhf
    with pytest.raises(ValueError, match="unknown kind"):
        vq.natural_orbitals(res, basis, kind="bogus")


def test_idempotency_deviation_unknown_kind(h2o_rhf):
    _, basis, res = h2o_rhf
    no = vq.natural_orbitals(res, basis)
    no.kind = "bogus"
    with pytest.raises(ValueError, match="unknown NO kind"):
        vq.idempotency_deviation(no)


# ---------------------------------------------------------------------------
# Reuse: NO matrix flows directly into existing cube/molden writers
# ---------------------------------------------------------------------------

def test_natural_orbitals_into_cube_writer(h2o_rhf, tmp_path):
    """NOs are just another (n_bf, n_bf) AO-basis MO matrix, so
    ``write_cube_mo`` should accept ``no.coefficients`` unchanged. We
    write the NO with the largest occupation and check the cube
    integrates to ~1 (the orbital is S-normalized; on a generous grid
    the per-voxel sum approximates ⟨ψ | ψ⟩ = 1)."""
    mol, basis, res = h2o_rhf
    no = vq.natural_orbitals(res, basis)
    grid = vq.make_uniform_grid(mol, spacing=0.2, padding=4.0)
    p = vq.write_cube_mo(
        tmp_path / "no_homo.cube",
        no.coefficients, 0,        # column 0 is the highest-occupation NO
        basis, mol, grid=grid,
    )
    assert p.exists()
    chi = vq.evaluate_ao(basis, grid.points())
    psi = chi @ no.coefficients[:, 0]
    norm = float((psi ** 2).sum() * np.prod(grid.spacing))
    assert norm == pytest.approx(1.0, rel=5e-2)
