"""Projected DOS (Phase V5b).

Pin both the physics (sum rules, group consistency) and the plumbing
(grouping helpers, custom-projection acceptance, total-DOS equivalence
when groups partition the AOs).
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_A = 6.0
_D = 1.4
_Y0 = _Z0 = 15.0


def _h2_chain():
    return vq.PeriodicSystem(
        1,
        [[_A, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]],
        [vq.Atom(1, [(_A - _D) / 2, _Y0, _Z0]),
         vq.Atom(1, [(_A + _D) / 2, _Y0, _Z0])],
    )


def _h_he_chain():
    """Asymmetric H–He chain — bonding/antibonding orbitals pick up
    nontrivial weight on each atom (not the trivial 1/1 split of H₂)."""
    return vq.PeriodicSystem(
        1,
        [[_A, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]],
        [vq.Atom(1, [(_A - _D) / 2, _Y0, _Z0]),
         vq.Atom(2, [(_A + _D) / 2, _Y0, _Z0])],
        multiplicity=2,
    )


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def test_ao_groups_per_atom_h2():
    sys = _h2_chain()
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    g = vq.ao_groups_per_atom(sys, basis)
    # STO-3G H = 1 s-shell × 2 atoms → 2 AOs total.
    assert g == {"H1": [0], "H2": [1]}


def test_ao_groups_per_atom_l_h2():
    sys = _h2_chain()
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    g = vq.ao_groups_per_atom_l(sys, basis)
    assert g == {"H1-s": [0], "H2-s": [1]}


def test_ao_groups_per_atom_l_split_by_l():
    """3-21G on H gives one s-shell per H (still no p), but a heavier
    element with multiple shells should split into per-l buckets. Use
    Ne / 3-21G as a quick witness — l covers s and p."""
    sys = vq.PeriodicSystem(
        3,
        [[8.0, 0, 0], [0, 8.0, 0], [0, 0, 8.0]],
        [vq.Atom(10, [0, 0, 0])],
    )
    basis = vq.BasisSet(sys.unit_cell_molecule(), "3-21g")
    g = vq.ao_groups_per_atom_l(sys, basis)
    # Ne(3-21G) is (6s,3p)/[3s,2p] in Pople notation:
    # 3 s-shells (3 AOs) and 2 p-shells (6 AOs) → 9 AOs, both buckets present.
    assert "Ne1-s" in g and "Ne1-p" in g
    assert len(g["Ne1-s"]) == 3
    assert len(g["Ne1-p"]) == 6


# ---------------------------------------------------------------------------
# PDOS sum rules
# ---------------------------------------------------------------------------

def test_pdos_total_matches_unprojected_dos_h2():
    """When the grouping covers every AO, the PDOS total is bit-equal to
    the plain DOS (modulo FP noise from extra arithmetic)."""
    sys = _h2_chain()
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    pdos = vq.density_of_states_projected_hcore(
        sys, basis, [12, 1, 1], projection="atoms",
        sigma=0.02, n_grid=201, n_electrons_per_cell=2,
    )
    dos = vq.density_of_states_hcore(
        sys, basis, [12, 1, 1],
        sigma=0.02, n_grid=201, n_electrons_per_cell=2,
    )
    # Same Gaussian-smeared sum, just with an extra Mulliken-weight
    # decomposition; pin to FP machine precision.
    assert np.allclose(pdos.total, dos.dos, atol=1e-12, rtol=0)


def test_pdos_per_atom_integrals_sum_to_band_count():
    """Each band integrates to 1; per-atom Mulliken weights of one band
    sum to 1 by construction (C^† S C = I). Therefore the per-atom PDOS
    integrals must sum to ``n_bands`` per cell."""
    sys = _h2_chain()
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    n_bands = basis.nbasis   # 2 for STO-3G H₂
    pdos = vq.density_of_states_projected_hcore(
        sys, basis, [12, 1, 1], projection="atoms",
        sigma=0.01, n_grid=801, pad=10.0,
    )
    integrals = {
        l: float(np.trapezoid(c, pdos.energies))
        for l, c in pdos.contributions.items()
    }
    total = sum(integrals.values())
    assert total == pytest.approx(float(n_bands), abs=2e-3)
    # Symmetric chain → per-atom weights split exactly in half.
    assert integrals["H1"] == pytest.approx(integrals["H2"], abs=1e-6)


def test_pdos_h_he_energy_resolved_asymmetry():
    """Mulliken sum rule forces ``Σ_n w_{μ,n} = 1`` per AO (basis
    completeness on a S-normalized eigenbasis), so per-atom *integrals*
    always equal the AO count regardless of how strong the asymmetry
    is. The asymmetry shows up in the *energy distribution*: at the
    bonding-band peak energy the lower-energy atom (He) carries more
    PDOS weight, and the antibonding peak is the mirror image.

    We verify by sampling each per-atom PDOS at the global maxima of
    the two contributions: if the curves were identical the argmax
    energies would coincide; for a heteronuclear chain they don't."""
    sys = _h_he_chain()
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    pdos = vq.density_of_states_projected_hcore(
        sys, basis, [12, 1, 1], projection="atoms",
        sigma=0.01, n_grid=2001, pad=10.0,
    )
    h_curve = pdos.contributions["H1"]
    he_curve = pdos.contributions["He2"]
    # Non-negative everywhere.
    assert np.all(h_curve >= -1e-12)
    assert np.all(he_curve >= -1e-12)
    # Curves are NOT identical pointwise — the heteronuclear pair has a
    # nontrivial bonding/antibonding asymmetry.
    assert np.max(np.abs(h_curve - he_curve)) > 0.05
    # The two atoms peak at *different* energies (bonding vs antibonding).
    e_h_peak = pdos.energies[int(np.argmax(h_curve))]
    e_he_peak = pdos.energies[int(np.argmax(he_curve))]
    assert e_h_peak != e_he_peak
    # Sum rule still holds in the integrals (1.0 per AO ≡ 1.0 per atom).
    h_int = float(np.trapezoid(h_curve, pdos.energies))
    he_int = float(np.trapezoid(he_curve, pdos.energies))
    assert h_int == pytest.approx(1.0, abs=2e-3)
    assert he_int == pytest.approx(1.0, abs=2e-3)


def test_pdos_atoms_l_partition_matches_atoms():
    """Atom × l partition is finer than the atom-only partition; summing
    the l-resolved channels of one atom must equal that atom's total."""
    sys = _h2_chain()
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    pdos_a = vq.density_of_states_projected_hcore(
        sys, basis, [8, 1, 1], projection="atoms",
        sigma=0.02, n_grid=151,
    )
    pdos_al = vq.density_of_states_projected_hcore(
        sys, basis, [8, 1, 1], projection="atoms_l",
        sigma=0.02, n_grid=151,
    )
    # Sum H1-s = H1 (only an s-shell exists for H/STO-3G).
    assert np.allclose(
        pdos_al.contributions["H1-s"], pdos_a.contributions["H1"],
        atol=1e-12,
    )


# ---------------------------------------------------------------------------
# Custom projection / error paths
# ---------------------------------------------------------------------------

def test_pdos_custom_groups_pass_through():
    """A custom dict with a single ``{label: [all AOs]}`` group must
    produce a single contribution equal to the total DOS."""
    sys = _h2_chain()
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    nbf = basis.nbasis
    pdos = vq.density_of_states_projected_hcore(
        sys, basis, [8, 1, 1],
        projection={"all": list(range(nbf))},
        sigma=0.02, n_grid=151,
    )
    assert list(pdos.contributions.keys()) == ["all"]
    assert np.allclose(pdos.contributions["all"], pdos.total, atol=1e-12)


def test_pdos_unknown_projection_string_raises():
    sys = _h2_chain()
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    with pytest.raises(ValueError, match="projection"):
        vq.density_of_states_projected_hcore(
            sys, basis, [4, 1, 1], projection="atoms_xy",
        )


def test_pdos_out_of_range_ao_raises():
    sys = _h2_chain()
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    with pytest.raises(IndexError, match="out-of-range"):
        vq.density_of_states_projected_hcore(
            sys, basis, [4, 1, 1],
            projection={"bogus": [0, 99]},
        )


def test_pdos_empty_groups_raises():
    sys = _h2_chain()
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    with pytest.raises(ValueError, match="empty|no.*groups"):
        vq.density_of_states_projected_hcore(
            sys, basis, [4, 1, 1], projection={},
        )


# ---------------------------------------------------------------------------
# Dataclass conveniences
# ---------------------------------------------------------------------------

def test_as_density_of_states_round_trip():
    sys = _h2_chain()
    basis = vq.BasisSet(sys.unit_cell_molecule(), "sto-3g")
    pdos = vq.density_of_states_projected_hcore(
        sys, basis, [4, 1, 1], projection="atoms",
        sigma=0.02, n_grid=51, n_electrons_per_cell=2,
    )
    dos = pdos.as_density_of_states()
    assert isinstance(dos, vq.DensityOfStates)
    assert np.allclose(dos.dos, pdos.total)
    assert dos.sigma == pdos.sigma
    assert dos.e_fermi == pdos.e_fermi
