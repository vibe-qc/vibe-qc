"""Phase D4b-3 — structural validation of the D4 reference-system
catalogue (``vibeqc.dispersion_d4_reference_systems``).

These tests do not run any SCF — they pin the *catalogue*: that every
reference system is internally consistent, builds a valid spin-labelled
:class:`Molecule`, has a sane geometry, and that the registry helpers
behave. The reference-C6 *values* are Phase D4b-4's concern.
"""

from __future__ import annotations

import math

import pytest

from vibeqc.dispersion_d4_reference_systems import (
    ANGSTROM_TO_BOHR,
    ReferenceSystem,
    all_reference_elements,
    all_reference_systems,
    build_molecule,
    reference_systems_for,
)

# Elements Phase D4b-3 commits to covering.
EXPECTED_ELEMENTS = (1, 2, 5, 6, 7, 8, 9, 10)  # H He B C N O F Ne


def _distance(a, b) -> float:
    """Å distance between two (Z, x, y, z) atom tuples."""
    return math.sqrt(sum((a[i + 1] - b[i + 1]) ** 2 for i in range(3)))


# ---------------------------------------------------------------------
# Catalogue coverage.
# ---------------------------------------------------------------------

def test_catalogue_is_nonempty():
    systems = all_reference_systems()
    assert len(systems) >= 15
    assert all(isinstance(s, ReferenceSystem) for s in systems)


def test_expected_elements_are_covered():
    assert all_reference_elements() == EXPECTED_ELEMENTS


def test_every_covered_element_has_a_system():
    for z in EXPECTED_ELEMENTS:
        systems = reference_systems_for(z)
        assert len(systems) >= 1
        assert all(s.element == z for s in systems)


def test_labels_are_unique():
    labels = [s.label for s in all_reference_systems()]
    assert len(labels) == len(set(labels))


def test_carbon_spans_three_hybridisations():
    """C must carry sp³ / sp² / sp references — the variety that
    makes the D4 CN interpolation meaningful."""
    cns = sorted({s.nominal_cn for s in reference_systems_for(6)})
    assert 4.0 in cns and 3.0 in cns and 2.0 in cns


def test_reference_systems_for_rejects_out_of_scope():
    with pytest.raises(KeyError, match="no D4 reference systems"):
        reference_systems_for(26)  # iron — not in the D4b-3 scope


# ---------------------------------------------------------------------
# Per-system internal consistency.
# ---------------------------------------------------------------------

@pytest.mark.parametrize("rs", all_reference_systems(),
                         ids=[s.label for s in all_reference_systems()])
class TestEverySystem:

    def test_target_atom_matches_element(self, rs):
        assert rs.atoms_angstrom[rs.target_index][0] == rs.element

    def test_catalogue_spin_matches_electron_count(self, rs):
        n_elec = sum(z for z, *_ in rs.atoms_angstrom) - rs.charge
        expected_multiplicity = 2 if rs.label == "C/CH-doublet" else 1
        assert rs.multiplicity == expected_multiplicity
        assert (n_elec - (expected_multiplicity - 1)) % 2 == 0

    def test_builds_a_molecule(self, rs):
        mol = build_molecule(rs)
        assert len(mol.atoms) == rs.n_atoms

    def test_no_overlapping_atoms(self, rs):
        atoms = rs.atoms_angstrom
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                d = _distance(atoms[i], atoms[j])
                assert d > 0.5, (
                    f"{rs.label}: atoms {i},{j} only {d:.3f} Å apart")

    def test_neighbour_count_matches_nominal_cn(self, rs):
        """The number of bonded neighbours must equal ``nominal_cn``
        exactly; a free atom (nominal_cn == 0) is alone.

        Covalent cutoff 1.45 Å — chosen for this fixed catalogue to
        sit above the longest real bond (F-F, 1.412 Å) and below the
        shortest 1-3 non-bonded contact (H···H in water, 1.515 Å)."""
        atoms = rs.atoms_angstrom
        target = atoms[rs.target_index]
        neighbours = sum(
            1 for k, a in enumerate(atoms)
            if k != rs.target_index and _distance(target, a) < 1.45
        )
        if rs.nominal_cn == 0.0:
            assert len(atoms) == 1
        else:
            assert neighbours == rs.nominal_cn


# ---------------------------------------------------------------------
# Geometry builders — shape correctness.
# ---------------------------------------------------------------------

def _angle(centre, a, b) -> float:
    """Angle a-centre-b in degrees."""
    va = [a[i + 1] - centre[i + 1] for i in range(3)]
    vb = [b[i + 1] - centre[i + 1] for i in range(3)]
    na = math.sqrt(sum(c * c for c in va))
    nb = math.sqrt(sum(c * c for c in vb))
    cos = sum(va[i] * vb[i] for i in range(3)) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def test_tetrahedral_methane_angles():
    ch4 = next(s for s in all_reference_systems() if s.label == "C/CH4-sp3")
    atoms = ch4.atoms_angstrom
    c = atoms[0]
    # All C-H bonds 1.0870 Å; all H-C-H angles 109.47°.
    for h in atoms[1:]:
        assert _distance(c, h) == pytest.approx(1.0870, abs=1e-3)
    for i in range(1, 5):
        for j in range(i + 1, 5):
            assert _angle(c, atoms[i], atoms[j]) == pytest.approx(
                109.4712, abs=1e-2)


def test_water_geometry():
    h2o = next(s for s in all_reference_systems() if s.label == "O/H2O-sp3")
    atoms = h2o.atoms_angstrom
    o = atoms[0]
    assert _distance(o, atoms[1]) == pytest.approx(0.9578, abs=1e-3)
    assert _distance(o, atoms[2]) == pytest.approx(0.9578, abs=1e-3)
    assert _angle(o, atoms[1], atoms[2]) == pytest.approx(104.48, abs=1e-2)


def test_ammonia_geometry():
    nh3 = next(s for s in all_reference_systems() if s.label == "N/NH3-sp3")
    atoms = nh3.atoms_angstrom
    n = atoms[0]
    for h in atoms[1:]:
        assert _distance(n, h) == pytest.approx(1.0124, abs=1e-3)
    for i in range(1, 4):
        for j in range(i + 1, 4):
            assert _angle(n, atoms[i], atoms[j]) == pytest.approx(
                106.67, abs=1e-2)


def test_trigonal_planar_bf3():
    bf3 = next(s for s in all_reference_systems() if s.label == "B/BF3-sp2")
    atoms = bf3.atoms_angstrom
    b = atoms[0]
    for f in atoms[1:]:
        assert _distance(b, f) == pytest.approx(1.3070, abs=1e-3)
    for i in range(1, 4):
        for j in range(i + 1, 4):
            assert _angle(b, atoms[i], atoms[j]) == pytest.approx(
                120.0, abs=1e-2)
    # Planar — all four atoms share z = 0.
    assert all(abs(a[3]) < 1e-9 for a in atoms)


def test_build_molecule_converts_to_bohr():
    h2 = next(s for s in all_reference_systems() if s.label == "H/H2")
    mol = build_molecule(h2)
    assert len(mol.atoms) == 2
    # H2 bond is 0.7414 Å → 1.4011 bohr.
    p0 = mol.atoms[0].xyz
    p1 = mol.atoms[1].xyz
    bond_bohr = math.sqrt(sum((p0[i] - p1[i]) ** 2 for i in range(3)))
    assert bond_bohr == pytest.approx(0.7414 * ANGSTROM_TO_BOHR, abs=1e-6)


def test_custom_reference_spin_and_charge_validation():
    ref = ReferenceSystem(
        element=2, label="He+/atom", atoms_angstrom=((2, 0., 0., 0.),),
        target_index=0, nominal_cn=0., charge=1, multiplicity=2,
        geometry_source="free atom",
    )
    assert build_molecule(ref).n_electrons() == 1
    for multiplicity in (0, 1, 3):
        with pytest.raises(ValueError, match="inconsistent"):
            ReferenceSystem(
                element=2, label="invalid", atoms_angstrom=ref.atoms_angstrom,
                target_index=0, nominal_cn=0., charge=1,
                multiplicity=multiplicity,
            )


def test_generate_custom_open_shell_charged_reference():
    from vibeqc.dispersion_d4_reference_data import generate_reference_dataset

    ref = ReferenceSystem(
        element=2, label="He+/atom", atoms_angstrom=((2, 0., 0., 0.),),
        target_index=0, nominal_cn=0., charge=1, multiplicity=2,
        geometry_source="free atom",
    )
    data = generate_reference_dataset(
        "cc-pvdz", reference_systems=[ref], n_grid=8, verbose=False,
    )
    assert data.supported_z == [2]
    assert data.get_qs(2) == pytest.approx([1.])
    assert data.get_cns(2) == [0.]
    assert data.get_c6_ref(2, 0, 2, 0) > 0
    alpha = data.elements[2].alpha_iw[0]
    assert all(value > 0 for value in alpha)
    assert all(a > b for a, b in zip(alpha[:-1], alpha[1:]))
    with pytest.raises(ValueError, match="unique"):
        generate_reference_dataset(reference_systems=[ref, ref], verbose=False)
    with pytest.raises(ValueError, match="ReferenceSystem"):
        generate_reference_dataset(reference_systems=[], verbose=False)


def test_canonical_free_atom_precedes_hydride_when_both_are_present():
    """Adding a coordination host must not replace the free-atom canonical."""
    import numpy as np
    from vibeqc.dispersion_d4_reference_data import _build_canonical_references

    hydrogen = next(r for r in reference_systems_for(1) if r.label == "H/H2")
    helium = reference_systems_for(2)[0]
    hydride = ReferenceSystem(
        element=2, label="He/HeH+", atoms_angstrom=((2, 0., 0., 0.), (1, 0., 0., .8)),
        target_index=0, nominal_cn=1., charge=1,
    )
    # Controlled spectra test extraction precedence, not measured chemistry.
    def record(ref, alpha, charges):
        return dict(ref=ref, zs=[a[0] for a in ref.atoms_angstrom],
                    alpha=np.asarray(alpha), charges=np.asarray(charges))
    records = [record(hydrogen, [8., 4.], [0., 0.]),
               record(hydride, [6., 3.], [.5, .5]),
               record(helium, [1.4, .7], [0.])]
    for ordered in (records, list(reversed(records))):
        canonical = _build_canonical_references(ordered)
        np.testing.assert_array_equal(canonical[2][0], [1.4, .7])
        assert canonical[2][1] == 0.


def test_carbon_canonical_host_remains_methane_with_ch_doublet():
    """Adding a low-CN reference must not change partner subtraction hosts."""
    import numpy as np
    from vibeqc.dispersion_d4_reference_data import _build_canonical_references

    host_alpha = {"H/H2": 4., "C/CH4-sp3": 16., "C/CH-doublet": 8.}
    systems = []
    for ref in all_reference_systems():
        if ref.label in host_alpha:
            systems.append({"ref": ref, "zs": [a[0] for a in ref.atoms_angstrom],
                            "alpha": np.array([host_alpha[ref.label]]),
                            "charges": np.zeros(ref.n_atoms)})
    assert len(systems) == 3
    canonical = _build_canonical_references(systems)
    np.testing.assert_allclose(canonical[6][0], [8.], atol=1e-14)
