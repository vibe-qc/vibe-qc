"""B3LYP flavor-convention regression pins.

"B3LYP" is one fixed recipe (Stephens-Devlin-Chabalowski-Frisch 1994:
0.20 HF + 0.80 LDA_x + 0.72 ΔB88 + 0.81 LYP + 0.19 VWN), but codes
split on WHICH Vosko-Wilk-Nusair parametrisation fills the 0.19 local-
correlation slot (Hertwig & Koch, Chem. Phys. Lett. 268, 345 (1997),
doi:10.1016/S0009-2614(97)00207-8):

* VWN5 (Ceperley-Alder fit) — ORCA, TURBOMOLE, ADF, CRYSTAL.
* VWN fit to RPA data ("VWN(III)" in Gaussian's naming; libxc
  LDA_C_VWN_RPA) — Gaussian, libxc, PySCF (both ``b3lyp`` and
  ``b3lypg`` as of 2.13), Psi4 (>= 1.2), NWChem, Q-Chem.

**vibe-qc ships the ORCA definition** (maintainer ruling, reaffirmed
2026-06-11 after the cross-code audit below): bare ``b3lyp`` is the
VWN5 variant (libxc HYB_GGA_XC_B3LYP5, id 475), with ``b3lyp5`` as
its explicit spelling for unambiguous cross-code tables. ``b3lyp/g``
(ORCA spelling) and ``b3lypg`` (PySCF spelling) select the Gaussian
variant (libxc HYB_GGA_XC_B3LYP, id 402) for parity with that camp.

Cross-code reference values, H2/STO-3G at 1.4 bohr, RKS converged to
1e-12 (this file's geometry):

* VWN5 flavor: PySCF 2.13.0 / libxc 7.0.0 ``b3lyp5`` -1.1586001482 Ha
  (vibe-qc matches to < 1e-9 on its own grid); CRYSTAL14 ``B3LYP``
  -1.1586001474 Ha (XLGRID, TOLDEE 11, MOLECULE input); ORCA
  ``B3LYP`` -1.158669 Ha (DefGrid2).
* Gaussian flavor: PySCF 2.13.0 ``b3lyp`` == ``b3lypg`` ==
  -1.1654009284 Ha; ORCA ``B3LYP/G`` -1.165470 Ha (DefGrid2).
* Flavor gap = 0.19 * (E_c[VWN_RPA] - E_c[VWN5]) = -6.8008 mHa on
  the converged density (measured with PySCF's NumInt on this
  system; the SCF energies differ by exactly the same amount).

There is deliberately NO ``b3lyp3`` alias: libxc's ``LDA_C_VWN_3`` is
the Ceperley-Alder fit III — numerically identical to VWN5 on closed-
shell densities — NOT Gaussian's "VWN(III)" (the RPA fit), so the name
would select the wrong flavor exactly when a user reaches for it.
"""
from __future__ import annotations

import pytest

from vibeqc import (
    Atom,
    BasisSet,
    Functional,
    Molecule,
    RKSOptions,
    run_rks,
)

# Cross-code targets quoted above (PySCF 2.13.0, libxc 7.0.0,
# conv_tol 1e-12; CRYSTAL14 and ORCA values agree per the docstring).
E_B3LYP_VWN5_FLAVOR = -1.1586001482      # bare b3lyp == b3lyp5
E_B3LYP_GAUSSIAN_FLAVOR = -1.1654009284  # b3lyp/g == b3lypg

# vibe-qc reproduced both PySCF numbers to <1e-9 Ha on its own default
# grid when this pin was recorded; 5e-7 leaves room for benign grid /
# libxc revisions while staying 4 orders below the 6.8 mHa flavor gap.
TOL = 5e-7


def _h2_energy(functional: str) -> float:
    mol = Molecule([Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
    basis = BasisSet(mol, "sto-3g")
    opts = RKSOptions()
    opts.functional = functional
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-9
    result = run_rks(mol, basis, opts)
    assert result.converged
    return result.energy


@pytest.mark.parametrize("spelling", ["b3lyp", "b3lyp5", "B3LYP5"])
def test_bare_b3lyp_is_the_vwn5_flavor(spelling):
    """The ORCA-definition default, under both spellings."""
    assert _h2_energy(spelling) == pytest.approx(
        E_B3LYP_VWN5_FLAVOR, abs=TOL)


@pytest.mark.parametrize("spelling", ["b3lyp/g", "b3lypg", "B3LYPG"])
def test_gaussian_flavor_spellings_agree(spelling):
    assert _h2_energy(spelling) == pytest.approx(
        E_B3LYP_GAUSSIAN_FLAVOR, abs=TOL)


def test_flavor_gap_is_the_vwn_slot():
    # 0.19·(E_c[VWN_RPA] − E_c[VWN5]) = −6.8008 mHa on this system.
    gap = _h2_energy("b3lyp/g") - _h2_energy("b3lyp")
    assert gap == pytest.approx(-6.8008e-3, abs=2e-5)


def test_both_flavors_are_20_percent_hybrids():
    for name in ("b3lyp", "b3lyp5", "b3lyp/g", "b3lypg"):
        fn = Functional(name)
        assert fn.is_hybrid
        assert fn.hf_exchange_fraction == pytest.approx(0.20, abs=1e-12)
        assert not fn.is_range_separated


def test_b3lyp3_does_not_resolve():
    """Guard the deliberate naming decision documented above: if a
    'b3lyp3' alias ever appears it must be a conscious choice, not a
    libxc fall-through to the wrong (Ceperley-Alder fit III) flavor."""
    with pytest.raises(Exception):
        Functional("b3lyp3")


def test_b3lyp_3c_composite_uses_the_default_flavor():
    """The 3c composite lineage is TURBOMOLE-culture (Grimme group),
    i.e. VWN5-B3LYP — which is exactly vibe-qc's bare ``b3lyp``."""
    from vibeqc.composites import resolve_composite

    recipe = resolve_composite("b3lyp-3c")
    assert recipe is not None
    assert recipe.functional == "b3lyp"
    assert Functional(recipe.functional).hf_exchange_fraction == (
        pytest.approx(0.20, abs=1e-12))


def test_dispersion_params_resolve_for_all_flavor_spellings():
    """D3(BJ) and D4 damping parameters are keyed to the B3LYP recipe,
    not the VWN flavor — every spelling must find them."""
    from vibeqc.dispersion import d3bj_params_for
    from vibeqc.dispersion_d4_parameters import get_d4_params

    reference = d3bj_params_for("b3lyp", backend="builtin")
    assert reference is not None
    d4_reference = get_d4_params("b3lyp")
    for spelling in ("b3lyp5", "b3lyp/g", "b3lypg", "B3LYP5"):
        p = d3bj_params_for(spelling, backend="builtin")
        assert p is not None
        assert (p.s6, p.s8, p.a1, p.a2) == (
            reference.s6, reference.s8, reference.a1, reference.a2)
        d4 = get_d4_params(spelling)
        assert d4 == d4_reference
