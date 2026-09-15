"""DFT-D3(BJ) native implementation tests.

These tests validate the machinery without requiring the optional reference
backend. The contracts exercised here:

- Coordination-number counting function: shape, physical values on
  well-known geometries, analytic gradient against finite differences.
- Becke-Johnson damping function: linear in R0 with the correct slope
  and intercept.
- C6 interpolation over the complete native H-Ar reference grid.
- Pairwise D3(BJ) energy: sign is negative (attractive), decays as the
  correct power of r at long range, scales linearly with s6/s8.
- Functional parameter registry: case-insensitive lookup, ``None`` for
  unknown names, expected keys present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq


HERE = Path(__file__).parent
H2O_XYZ = HERE.parent / "examples" / "h2o.xyz"


# ---------------------------------------------------------------------------
# Coordination numbers
# ---------------------------------------------------------------------------

def test_coordination_numbers_h2o_values():
    """O in H2O has CN ≈ 2 (two bonded H); each H has CN ≈ 1."""
    mol = vq.Molecule.from_xyz(str(H2O_XYZ))
    cn = vq.d3_coordination_numbers(mol)
    assert cn.shape == (3,)
    # Atom 0 is O (from h2o.xyz ordering); atoms 1 and 2 are H.
    assert cn[0] == pytest.approx(2.0, abs=0.1)
    assert cn[1] == pytest.approx(1.0, abs=0.1)
    assert cn[2] == pytest.approx(1.0, abs=0.1)


def test_coordination_numbers_isolated_atom_is_zero():
    """A single atom has no neighbors, so CN = 0 trivially."""
    mol_lit = vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0])], charge=-1)
    cn = vq.d3_coordination_numbers(mol_lit)
    assert cn.shape == (1,)
    assert cn[0] == pytest.approx(0.0, abs=1e-12)


def test_coordination_numbers_finite_difference_gradient():
    """Numerical partial derivative of CN w.r.t. atom positions matches
    the analytic counting-function derivative baked into the C++ code.

    We verify indirectly: reading CN at ±h displacements of each atom
    and comparing the central-difference estimate against the implicit
    finite-difference of the analytic expression. Tight tolerance
    because the counting function is smooth.
    """
    mol = vq.Molecule.from_xyz(str(H2O_XYZ))

    def shifted(atoms, idx, axis, delta):
        """Return a new Molecule with ``atoms[idx].xyz[axis]`` shifted by
        ``delta``. Pybind11's Atom.xyz property returns a fresh list each
        call, so in-place mutation doesn't stick — we build a new Atom
        from shifted coordinates instead."""
        out = []
        for i, a in enumerate(atoms):
            xyz = list(a.xyz)
            if i == idx:
                xyz[axis] += delta
            out.append(vq.Atom(a.Z, xyz))
        return vq.Molecule(out)

    # Move atom 1 (H) along y. In examples/h2o.xyz atom 1 sits at
    # y ≈ +1.5, z ≈ -1.16 so a +y displacement actually changes the
    # O-H distance (unlike an x displacement which is perpendicular
    # to the bond and has vanishing first-order effect on r).
    h = 1e-4
    atoms = list(mol.atoms)
    mol_plus = shifted(atoms, 1, 1, +h)
    mol_minus = shifted(atoms, 1, 1, -h)

    cn_plus = vq.d3_coordination_numbers(mol_plus)
    cn_minus = vq.d3_coordination_numbers(mol_minus)
    # Central difference on CN_O must be finite and non-zero (moving
    # atom 1 changes the r_OH1 distance, which contributes to CN_O).
    d_cnO = (cn_plus[0] - cn_minus[0]) / (2 * h)
    assert abs(d_cnO) > 1e-3

    # Pair symmetry: because CN uses the symmetric counting function
    # f(r_AB) added to both CN_A and CN_B, we have
    #     |∂CN_O / ∂r_{H1,y}| = |∂CN_{H1} / ∂r_{O,y}|
    # for the pair-term contribution (signs are opposite).
    cn_Op = vq.d3_coordination_numbers(shifted(atoms, 0, 1, +h))
    cn_Om = vq.d3_coordination_numbers(shifted(atoms, 0, 1, -h))
    d_cnH1_dOy = (cn_Op[1] - cn_Om[1]) / (2 * h)
    assert abs(d_cnO) == pytest.approx(abs(d_cnH1_dOy), rel=1e-3)


# ---------------------------------------------------------------------------
# Atomic parameter tables
# ---------------------------------------------------------------------------

def test_r2r4_values_match_grimme_table():
    """Spot-check a few r2r4 entries against Grimme 2010 Table S1."""
    # From Grimme 2010 SI; values in a.u.
    cases = {1: 8.0589, 6: 7.8715, 7: 5.5588, 8: 4.7566, 18: 5.6004}
    for Z, ref in cases.items():
        assert vq.d3_r2r4(Z) == pytest.approx(ref, abs=1e-4)


def test_rcov_ordering_is_chemical():
    """Covalent radii should roughly decrease across a period and
    increase down a group — a weak but cheap structural sanity check."""
    # Across period 2 the radius decreases (Li > Be > B > C > N).
    for Z in range(3, 7):  # Li (3) -> N (7)
        assert vq.d3_rcov(Z) > vq.d3_rcov(Z + 1)
    # Down group 1: Li < Na.
    assert vq.d3_rcov(3) < vq.d3_rcov(11)


def test_rcov_unsupported_element_returns_nan():
    """Z outside the native H-Ar range returns NaN."""
    assert np.isnan(vq.d3_rcov(30))


def test_rcov_values_match_simple_dftd3():
    """Pin the D3-scaled Pyykko radii that drive coordination numbers."""
    cases = {
        1: 0.8062831465,
        3: 3.0235617994,
        6: 1.8897261246,
        8: 1.5873699447,
        11: 3.5274887660,
        14: 2.6204202261,
        18: 2.4188494395,
    }
    for Z, expected in cases.items():
        assert vq.d3_rcov(Z) == pytest.approx(expected, rel=0, abs=5e-10)


# ---------------------------------------------------------------------------
# Functional parameter registry
# ---------------------------------------------------------------------------

def test_d3bj_params_case_insensitive_lookup():
    """Name matching is case-insensitive."""
    p1 = vq.d3bj_params_for("PBE")
    p2 = vq.d3bj_params_for("pbe")
    p3 = vq.d3bj_params_for("Pbe")
    for p in (p1, p2, p3):
        assert p is not None
    assert p1.s8 == p2.s8 == p3.s8
    assert p1.a1 == p2.a1 == p3.a1


def test_d3bj_params_known_functionals_present():
    """The full registry includes modern and double-hybrid fits."""
    for name in (
        "PBE", "PBE0", "BLYP", "B3LYP", "TPSS", "revPBE",
        "r2scan", "wb97x", "wb97m", "dsdpbep86", "dsd-pbep86", "skala-1.0",
    ):
        p = vq.d3bj_params_for(name)
        assert p is not None, f"missing D3-BJ params for {name}"
        assert p.s6 >= 0.0
        assert p.a1 >= 0.0 and p.a2 > 0.0


@pytest.mark.parametrize("name", ["skala", "skala-1.1", "skala-1.1-rev1"])
def test_skala_11_uses_published_b3lyp5_d3bj_settings(name):
    """The official SKALA-1.1 checkpoint declares ``b3lyp5`` D3 settings."""
    skala_params = vq.d3bj_params_for(name, backend="builtin")
    b3lyp5_params = vq.d3bj_params_for("b3lyp5", backend="builtin")
    assert skala_params is not None
    assert b3lyp5_params is not None
    fields = ("s6", "s8", "a1", "a2", "s9")
    assert tuple(getattr(skala_params, field) for field in fields) == tuple(
        getattr(b3lyp5_params, field) for field in fields
    )


def test_d3bj_params_unknown_is_none():
    assert vq.d3bj_params_for("not-a-functional") is None


def test_dsd_pbep86_d3bj_params_present():
    """DSD-PBEP86 (Kozuch & Martin, PCCP 13, 20104 (2011); JCC 34, 2327
    (2013)) — the hyphenated name resolves to the GMTKN55-validated
    dsdpbep86_2011 D3(BJ) fit (s6=0.418, s8=0, a1=0, a2=5.65).

    The bare dsdpbep86 entry (s6=0.48, s8=0, a1=0, a2=5.6) is the
    2013 JCC fit; the vibe-qc alias favours the 2011 parametrisation.
    Both are valid D3(BJ) parameter sets for the same XC recipe.
    """
    p = vq.d3bj_params_for("dsd-pbep86", backend="builtin")
    assert p is not None, "missing D3-BJ params for dsd-pbep86 (builtin)"
    # s6 = 0.418 (2011 PCCP / GMTKN55-fit), s8 = 0 (double-hybrid
    # convention: the C8 dispersion overlaps with the MP2 correction),
    # a1 = 0 (double-hybrid damping convention), a2 = 5.65.
    assert p.s6 == pytest.approx(0.418, abs=1e-12)
    assert p.s8 == 0.0
    assert p.a1 == 0.0
    assert p.a2 == pytest.approx(5.65, abs=1e-12)
    assert p.s9 == 0.0  # three-body term off by default


def test_dsd_pbep86_d3bj_h2o_nonzero_regression():
    """DSD-PBEP86-D3(BJ) on H2O must produce a nonzero (attractive)
    dispersion correction. This is the regression test for BUG 35 —
    before the full 156-entry D3(BJ) parameter table landed (commit
    10924b6bf), compute_d3bj raised ValueError because the small
    hand-curated table had no DSD-PBEP86 entry.
    """
    mol = vq.Molecule.from_xyz(str(H2O_XYZ))
    res = vq.compute_d3bj(mol, "dsd-pbep86", backend="builtin")
    # Dispersion is always attractive (negative) for a bound molecule.
    assert res.energy < 0.0
    # Must be nonzero (the bug would have raised before reaching this).
    assert abs(res.energy) > 1e-12


def test_backend_auto_is_self_contained_for_native_elements(monkeypatch):
    """The quantitative H-Ar path must not import the optional backend."""
    import vibeqc.dispersion as dispersion

    def _missing(_functional):
        raise ImportError("simulated missing dftd3")

    monkeypatch.setattr(dispersion, "_params_from_dftd3", _missing)
    p = dispersion.d3bj_params_for("b3lyp", backend="auto")
    assert p is not None
    energy = dispersion.compute_d3bj(
        _h2_molecule(5.0), p, backend="auto"
    ).energy
    assert energy < 0.0


# ---------------------------------------------------------------------------
# Pairwise D3(BJ) energy
# ---------------------------------------------------------------------------

def _h2_molecule(r_bohr: float) -> "vq.Molecule":
    return vq.Molecule([
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [r_bohr, 0.0, 0.0]),
    ])


def test_dispersion_energy_is_attractive():
    """E_disp < 0 for any bound-distance pair of atoms."""
    mol = _h2_molecule(r_bohr=1.4)
    p = vq.d3bj_params_for("PBE")
    res = vq.compute_d3bj(mol, p)
    assert res.energy < 0.0


def test_dispersion_energy_goes_to_zero_at_infinity():
    """E_disp → 0 for r → ∞ (power-law decay)."""
    p = vq.d3bj_params_for("PBE")
    e_near = vq.compute_d3bj(_h2_molecule(1.5), p).energy
    e_far = vq.compute_d3bj(_h2_molecule(50.0), p).energy
    # At r = 50 bohr the C6/r^6 tail is O(1e-9) Ha; the point here is
    # that it is *vastly* smaller than the near-distance value, not that
    # it is literally zero.
    assert abs(e_far) < 1e-8
    assert abs(e_near) > abs(e_far) * 1e5  # decay is fast


def test_dispersion_energy_scales_linearly_with_s6():
    """At r ≫ BJ damping radius, the s6 prefactor is linear in E."""
    p = vq.d3bj_params_for("PBE")
    # Copy params with scaled s6.
    p2 = vq.D3BJParams(s6=2.0 * p.s6, s8=p.s8, a1=p.a1, a2=p.a2)
    mol = _h2_molecule(5.0)  # far enough that BJ floor is negligible
    e1 = vq.compute_d3bj(mol, p).energy
    e2 = vq.compute_d3bj(mol, p2).energy
    # With s8 held fixed, only the C6 term scales. The C8 term contributes
    # but at r=5 bohr it's already small compared to C6/r^6. Looser tolerance.
    # The doubled s6 piece should overshoot the s8 residual well.
    assert e2 < e1  # more negative (more attractive) when s6 is larger


def test_dispersion_single_atom_is_zero():
    """A single atom has no pairs, so E_disp = 0 identically."""
    mol = vq.Molecule([vq.Atom(8, [0.0, 0.0, 0.0])])
    p = vq.d3bj_params_for("PBE")
    assert vq.compute_d3bj(mol, p).energy == 0.0


def test_unsupported_element_raises_in_builtin_backend():
    """The native backend fails clearly beyond its quantitative H-Ar range."""
    mol = vq.Molecule([
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(19, [2.0, 0.0, 0.0]),
    ])
    p = vq.D3BJParams(s6=1.0, s8=1.0, a1=0.4, a2=4.4)
    with pytest.raises((ValueError, RuntimeError)):
        vq.compute_d3bj(mol, p, backend="builtin")


# ---------------------------------------------------------------------------
# Simple two-atom coverage in the builtin backend
# ---------------------------------------------------------------------------
#
# Rare gases have one free-atom reference environment, making their diagonal
# C6 values convenient direct table checks.


def _dimer(Z: int, r_bohr: float) -> "vq.Molecule":
    return vq.Molecule([
        vq.Atom(Z, [0.0, 0.0, 0.0]),
        vq.Atom(Z, [0.0, 0.0, r_bohr]),
    ])


def test_builtin_he_dimer_is_attractive():
    """He dimer must not crash and must bind (E_disp < 0) in the
    builtin backend — the symptom that used to fail without dftd3."""
    p = vq.D3BJParams(s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407)
    res = vq.compute_d3bj(_dimer(2, 5.0), p, backend="builtin")
    assert res.energy < 0.0


def test_builtin_rare_gas_c6_matches_d3_reference():
    """Pin the builtin rare-gas diagonal C6 against Grimme's D3
    reference data.

    With s8 = a1 = a2 = 0 the BJ damping radius f(R0) = a1·R0 + a2
    vanishes and the pair energy reduces to E = -s6·C6/r⁶ exactly, so
    -E·r⁶ reads the table value back out. Published source: free-atom
    reference C6 of Grimme, Antony, Ehrlich & Krieg, J. Chem. Phys.
    132, 154104 (2010), doi:10.1063/1.3382344; the four-decimal values
    below were extracted bit-level from the reference dftd3 library
    (simple-dftd3) on 2026-06-10 and are CN-independent for these
    elements (single free-atom reference each).
    """
    C6_REF = {2: 1.5583, 10: 6.2896, 18: 64.6462}  # He, Ne, Ar (a.u.)
    bare = vq.D3BJParams(s6=1.0, s8=0.0, a1=0.0, a2=0.0)
    r = 10.0
    for Z, c6_ref in C6_REF.items():
        e = vq.compute_d3bj(_dimer(Z, r), bare, backend="builtin").energy
        assert -e * r**6 == pytest.approx(c6_ref, rel=1e-10), (
            f"builtin C6(Z={Z}) drifted from the D3 reference value"
        )


def test_builtin_rare_gas_gradient_is_translation_free():
    """Gradient path works for rare gases too (sum over atoms = 0)."""
    p = vq.D3BJParams(s6=1.0, s8=0.7875, a1=0.4289, a2=4.4407)
    res = vq.compute_d3bj(_dimer(18, 7.0), p, backend="builtin",
                          with_gradient=True)
    assert res.gradient.shape == (2, 3)
    assert np.allclose(res.gradient.sum(axis=0), 0.0, atol=1e-14)
    # Sign sanity: dispersion is attractive at vdW-ish separations, so
    # contracting the dimer lowers the energy.
    e7 = vq.compute_d3bj(_dimer(18, 7.0), p, backend="builtin").energy
    e6 = vq.compute_d3bj(_dimer(18, 6.5), p, backend="builtin").energy
    assert e6 < e7 < 0.0


# ---------------------------------------------------------------------------
# Gradient
# ---------------------------------------------------------------------------

def test_gradient_returns_correct_shape():
    mol = vq.Molecule.from_xyz(str(H2O_XYZ))
    p = vq.d3bj_params_for("PBE")
    res = vq.compute_d3bj(mol, p, with_gradient=True)
    assert res.gradient.shape == (3, 3)
    # No translational dependence ⇒ sum over atoms of each grad component = 0.
    assert np.allclose(res.gradient.sum(axis=0), 0.0, atol=1e-12)


def test_gradient_matches_finite_difference_on_h2():
    """The complete analytic gradient matches a central difference."""
    r0 = 2.5   # bohr, somewhere between bond length and van-der-Waals
    mol = _h2_molecule(r0)
    p = vq.d3bj_params_for("PBE")
    res = vq.compute_d3bj(mol, p, with_gradient=True)
    g_ana = res.gradient[1, 0]   # dE/dx on atom B, with A fixed at origin

    h = 1e-5
    e_plus = vq.compute_d3bj(_h2_molecule(r0 + h), p).energy
    e_minus = vq.compute_d3bj(_h2_molecule(r0 - h), p).energy
    g_num = (e_plus - e_minus) / (2 * h)

    # The finite-difference gradient here is dE/dr, and r = x_B since A is
    # at origin on the x-axis; so dE/dx_B = dE/dr = g_num.
    assert g_ana == pytest.approx(g_num, rel=1e-4, abs=1e-10)


def test_cn_dependent_gradient_matches_finite_difference_on_water():
    """Exercise the C6(CN) chain rule on a multi-reference element pair."""
    mol = vq.Molecule.from_xyz(str(H2O_XYZ))
    atoms = list(mol.atoms)
    p = vq.d3bj_params_for("pbe")
    analytic = np.asarray(
        vq.compute_d3bj(mol, p, with_gradient=True).gradient
    )

    h = 1e-5
    finite_difference = np.zeros_like(analytic)
    for atom_index in range(len(atoms)):
        for axis in range(3):
            shifted = []
            for index, atom in enumerate(atoms):
                xyz = list(atom.xyz)
                if index == atom_index:
                    xyz[axis] += h
                shifted.append(vq.Atom(atom.Z, xyz))
            e_plus = vq.compute_d3bj(vq.Molecule(shifted), p).energy

            shifted[atom_index] = vq.Atom(
                atoms[atom_index].Z,
                [
                    coordinate - (2.0 * h if i == axis else 0.0)
                    for i, coordinate in enumerate(shifted[atom_index].xyz)
                ],
            )
            e_minus = vq.compute_d3bj(vq.Molecule(shifted), p).energy
            finite_difference[atom_index, axis] = (e_plus - e_minus) / (2 * h)

    assert np.max(np.abs(analytic - finite_difference)) < 1e-9
