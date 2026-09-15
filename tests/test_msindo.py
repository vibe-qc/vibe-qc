"""Parity tests for the MSINDO engine (vibeqc.semiempirical.methods.msindo).

Reference values are from a reference MSINDO build, run out-of-process via
examples/regression/msindo/ (CLAUDE.md §10).  Scope: closed-shell, s- and
p-shell main-group elements (H, He, C, N, O, F).
"""

import json
from pathlib import Path

import pytest
from vibeqc.semiempirical.methods import msindo

_SYMBOL_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
}
_REF = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "examples/regression/msindo/molecular_reference.json"
    ).read_text()
)
_ACETAMIDE_Z = [6, 8, 7, 6, 1, 1, 1, 1, 1]
_ACETAMIDE_XYZ = [
    [0.000, 0.000, 0.195],
    [0.000, 1.221, -0.014],
    [0.000, -1.170, -0.450],
    [1.284, -0.325, -0.645],
    [2.130, -1.015, -0.640],
    [1.315, 0.220, -1.595],
    [1.400, 0.398, 0.165],
    [-0.820, -1.680, -0.730],
    [0.820, -1.680, -0.730],
]
_CYCLOPROPENE_Z = [6, 6, 6, 1, 1, 1, 1]
_CYCLOPROPENE_XYZ = [
    [0.000, 0.000, 0.865],
    [0.000, 0.664, -0.321],
    [0.000, -0.664, -0.321],
    [0.000, 0.000, 1.942],
    [0.000, 1.510, -0.670],
    [0.916, 0.000, -0.670],
    [-0.916, 0.000, -0.670],
]


def _run(mol):
    Z = [_SYMBOL_Z[s] for s, *_ in mol["geometry"]]
    xyz = [[x, y, z] for _, x, y, z in mol["geometry"]]
    return msindo.run_msindo(
        Z,
        xyz,
        charge=mol.get("charge", 0),
        multiplicity=mol.get("multiplicity", 1),
    )


@pytest.mark.parametrize("mol", _REF["molecules"], ids=lambda m: m["name"])
def test_total_energy_parity(mol):
    """Every reference molecule reproduces MSINDO's total energy to ~1e-6 Ha."""
    r = _run(mol)
    assert r.converged
    assert r.total_energy == pytest.approx(mol["reference"]["total_energy"], abs=1e-6)


@pytest.mark.parametrize("mol", _REF["molecules"], ids=lambda m: m["name"])
def test_binding_energy_parity(mol):
    """Binding energies (TOTAL − Σ ATENG) match reference MSINDO."""
    r = _run(mol)
    assert r.binding_energy == pytest.approx(
        mol["reference"]["binding_energy"], abs=1e-6
    )


def test_h2_total_energy_parity():
    """H2 @ 0.741 A reproduces reference MSINDO total energy to ~1e-9 Ha."""
    r = msindo.run_msindo([1, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.741]])
    assert r.converged
    # Aggregate work includes the primary, WICHT probe, and strict refinement.
    assert r.n_iter == 6
    assert r.total_energy == pytest.approx(-1.1732166872, abs=1e-7)
    assert r.binding_energy == pytest.approx(-0.1732166872, abs=1e-7)
    assert r.mo_energies[0] == pytest.approx(-0.6492, abs=1e-3)
    assert r.mo_energies[1] == pytest.approx(0.3161, abs=1e-3)


def test_h2_intermediates_match_oracle():
    """Block-level checks against the oracle's PRINTOPTS dumps for H2."""
    R = 0.741 * msindo.ANGSTROM_TO_BOHR
    zs = msindo.MUS[1]
    assert msindo.gss(1) == pytest.approx(0.62875, abs=1e-5)
    assert msindo.coulomb_1s1s(zs, zs, R) == pytest.approx(0.54918, abs=1e-4)
    assert msindo.overlap_1s1s(zs, zs, R) == pytest.approx(0.69134, abs=1e-4)
    assert msindo.lmunu_ss(zs, zs, R, True, True) == pytest.approx(-0.16036, abs=1e-4)


def test_fluorine_one_center_integrals():
    """F one-center 2e integrals match the oracle GMUNU dump (gij1.f relations)."""
    oc = msindo.one_center_2e(9)
    assert oc["GSS"] == pytest.approx(0.85037, abs=1e-5)
    assert oc["GSP"] == pytest.approx(0.83271, abs=1e-5)
    assert oc["HSP"] == pytest.approx(0.18378, abs=1e-5)
    assert oc["GPP"] == pytest.approx(0.87929, abs=1e-5)
    assert oc["GP2"] == pytest.approx(0.78452, abs=1e-5)
    assert oc["HPP"] == pytest.approx(0.04739, abs=1e-5)
    # INDO relation GPP − GP2 = 2·HPP.
    assert oc["GPP"] - oc["GP2"] == pytest.approx(2.0 * oc["HPP"], abs=1e-9)


def test_hf_core_hamiltonian_block():
    """HF core Hamiltonian H⁰ matches the oracle HMAT dump (σ/π split, FCP)."""
    import numpy as np

    Z = [9, 1]
    C = np.array([[0, 0, 0], [0, 0, 0.917]]) * msindo.ANGSTROM_TO_BOHR
    blocks, nsto = msindo._atom_blocks(Z)
    H, _ = msindo._build_core_and_gamma(Z, C, blocks, nsto)
    # order: Fs, Fpx, Fpy, Fpz, Hs
    assert H[0, 0] == pytest.approx(-7.0707, abs=1e-3)  # Fs
    assert H[1, 1] == pytest.approx(-5.8505, abs=1e-3)  # Fpx (π)
    assert H[3, 3] == pytest.approx(-5.8728, abs=1e-3)  # Fpz (σ)
    assert H[4, 4] == pytest.approx(-4.1491, abs=1e-3)  # Hs (frozen-core)
    assert H[0, 4] == pytest.approx(-0.3247, abs=1e-3)  # Fs–Hs
    assert abs(H[1, 4]) == pytest.approx(0.0, abs=1e-9)  # Fpx–Hs (π, no overlap)


def test_unsupported_element():
    """The MSINDO engine now covers the full H–Xe table (Z 1–54); elements beyond
    Xe have no bundled parameters and raise.  Cs (Z55) is the first such."""
    with pytest.raises(NotImplementedError):
        msindo.run_msindo([55, 9], [[0, 0, 0], [0, 0, 2.3]])  # CsF (Cs=55, no params)


def test_run_job_unsupported_element_fails_closed(tmp_path):
    """Molecular run_job(method="msindo") guards element scope BEFORE dispatch.

    A closed-shell molecule routes to the C++ run_msindo_full, which does NOT
    itself validate element scope — without the runner-level guard an
    unsupported Z (Cs, Z=55) would reach the engine with
    MsindoParameterSet::get(...) returning 0.0 for the missing params and
    produce a meaningless number.  The guard makes molecular run_job fail closed
    for Z>54 exactly like the direct run_msindo call (test_unsupported_element)
    and the CCM run_job path (test_msindo_ccm) already do.  A supported heavy
    element (TcCl, Z=43) still runs end-to-end through C++ to the oracle energy,
    proving the guard does not over-reject."""
    import vibeqc as vq

    a2b = msindo.ANGSTROM_TO_BOHR  # vq.Atom coords are bohr; runner re-derives Å

    # Cs (Z=55): closed-shell CsF, no bundled params → clean NotImplementedError
    # (not a silent wrong answer from the C++ engine).
    csf = vq.Molecule(
        [vq.Atom(55, [0.0, 0.0, 0.0]), vq.Atom(9, [0.0, 0.0, 2.3 * a2b])], 0, 1
    )
    with pytest.raises(NotImplementedError, match="MSINDO engine supports"):
        vq.run_job(csf, method="msindo", output=str(tmp_path / "csf"), verbose=0)

    # TcCl (Z=43, a supported _MSINDO_TRAJECTORY_SCF heavy element): the guard
    # must NOT over-reject — run_job still reaches the C++ closed-shell path and
    # returns the MSINDO oracle total energy (matches the TcCl reference in
    # test_heavy_4d_5p_total_energy_parity).
    tccl = vq.Molecule(
        [vq.Atom(43, [0.0, 0.0, 0.0]), vq.Atom(17, [0.0, 0.0, 2.25 * a2b])], 0, 1
    )
    r = vq.run_job(tccl, method="msindo", output=str(tmp_path / "tccl"), verbose=0)
    assert r.converged
    assert float(r.energy) == pytest.approx(-25.0060838476, abs=1e-6)


def test_sn_total_energy_parity():
    """Sn (Z50, 5p) total-energy parity vs the MSINDO oracle — enabled via the
    [Kr]/[Kr]4d¹⁰ frozen-core penetration (4s/4p/4d) + the In–Xe d-shell ENEG
    formula (atomic_reference.f).  Reference energies from the MSINDO oracle
    (examples/regression/msindo/build_oracle.sh)."""
    import math

    import numpy as np

    def _tet(b):
        d = b / math.sqrt(3.0)
        return [(d, d, d), (d, -d, -d), (-d, d, -d), (-d, -d, d)]

    cases = [
        ([50, 1, 1, 1, 1], [(0.0, 0.0, 0.0)] + _tet(1.711), -4.8628737828),
        ([50, 8], [(0.0, 0.0, 0.0), (0.0, 0.0, 1.833)], -17.9791802532),
        ([50, 9, 9, 9, 9], [(0.0, 0.0, 0.0)] + _tet(1.880), -97.4075592296),
    ]
    for Z, coords, ref in cases:
        r = msindo.run_msindo(Z, np.array(coords, float))
        assert r.total_energy == pytest.approx(ref, abs=1e-6), (
            f"Sn parity Z={Z}: got {r.total_energy}, expected {ref}"
        )


def test_in_total_energy_parity():
    """In (Z49, 5p) total-energy parity vs the MSINDO oracle — group-13 planar
    trihydride/trihalides (full valence, no lone-pair d-occupation).  References
    from the MSINDO oracle (examples/regression/msindo/build_oracle.sh)."""
    import numpy as np

    P = 0.8660254  # sin(60°), planar D3h vertices

    def _planar3(r):
        return [(r, 0.0, 0.0), (-r / 2, r * P, 0.0), (-r / 2, -r * P, 0.0)]

    cases = [
        ([49, 1, 1, 1], [(0.0, 0.0, 0.0)] + _planar3(1.71), -3.0064025383),
        ([49, 9, 9, 9], [(0.0, 0.0, 0.0)] + _planar3(1.79), -72.422082203),
        ([49, 17, 17, 17], [(0.0, 0.0, 0.0)] + _planar3(2.05), -43.3718274211),
    ]
    for Z, coords, ref in cases:
        r = msindo.run_msindo(Z, np.array(coords, float))
        assert r.total_energy == pytest.approx(ref, abs=1e-6), (
            f"In parity Z={Z}: got {r.total_energy}, expected {ref}"
        )


def test_ag_cd_total_energy_parity():
    """Ag (Z47, 4d¹⁰5s¹) and Cd (Z48, 4d¹⁰5s²) total-energy parity vs the MSINDO
    oracle.  These are occupied-4d elements that atomic_reference.f computes via
    the In–Xe loop (DO L=47,54), which OVERWRITES the Sc–Zn-style Y–Cd-loop values
    for L=47,48 — so the diagonal U_ss/U_pp/U_dd use the In–Xe p-block formula
    carrying the occupied d shell (DEL=ND=10), not the Y–Pd occupied-d TM form.
    Closed-shell metal hydride/oxide/halides (varied H/O/F/Cl partners); references
    from the MSINDO oracle (examples/regression/msindo/build_oracle.sh)."""
    import numpy as np

    cases = [
        ([47, 1], [(0, 0, 0), (0, 0, 1.62)], -13.3529501543),  # AgH
        ([47, 9], [(0, 0, 0), (0, 0, 1.98)], -36.3888699675),  # AgF
        ([47, 17], [(0, 0, 0), (0, 0, 2.28)], -32.1734629531),  # AgCl
        ([48, 8], [(0, 0, 0), (0, 0, 1.92)], -31.4763143748),  # CdO
        ([48, 9, 9], [(0, 0, 0), (0, 0, 1.97), (0, 0, -1.97)], -63.4616154930),  # CdF2
        (
            [48, 17, 17],
            [(0, 0, 0), (0, 0, 2.21), (0, 0, -2.21)],
            -49.8912994610,
        ),  # CdCl2
    ]
    for Z, coords, ref in cases:
        r = msindo.run_msindo(Z, np.array(coords, float))
        assert r.total_energy == pytest.approx(ref, abs=1e-6), (
            f"Ag/Cd parity Z={Z[0]}: got {r.total_energy}, expected {ref}"
        )


def test_heavy_4d_5p_total_energy_parity():
    """Tc–Pd (open-d 4d metals) and Sb–Xe (heavier lone-pair 5p) total-energy
    parity vs the MSINDO oracle.  These reach the reference SCF stationary point
    only via the MSINDO-faithful Hückel-guess + WICHT-damped SCF (msindo.
    _scf_rhf_msindo, dispatched through _MSINDO_TRAJECTORY_SCF); the production
    DIIS path lands on a different stationary point (higher for Tc/Rh, lower with
    the polarization d wrongly occupied for Pd/Sb/Xe).  Two diverse closed-shell
    molecules per element; references from the MSINDO oracle (see
    docs/user_guide/msindo.md, examples/regression/msindo/build_oracle.sh)."""
    import math

    import numpy as np

    def _octa(r):
        return [(r, 0, 0), (-r, 0, 0), (0, r, 0), (0, -r, 0), (0, 0, r), (0, 0, -r)]

    def _tet(r):
        d = r / math.sqrt(3.0)
        return [(d, d, d), (d, -d, -d), (-d, d, -d), (-d, -d, d)]

    def _tbp(r):
        p = 0.8660254
        return [(r, 0, 0), (-r / 2, r * p, 0), (-r / 2, -r * p, 0), (0, 0, r), (0, 0, -r)]

    def _pl3(r):
        p = 0.8660254
        return [(r, 0, 0), (-r / 2, r * p, 0), (-r / 2, -r * p, 0)]

    cases = [
        ([43, 17], [(0, 0, 0), (0, 0, 2.25)], -25.0060838476),  # TcCl
        ([43, 8, 8, 8, 9],
         [(0, 0, 0), (0, 0, 1.65), (1.55, 0, -0.55), (-1.55, 0, -0.55), (0, 0, -1.8)],
         -81.6110075367),  # TcO3F
        ([44, 8, 8, 8, 8], [(0, 0, 0)] + _tet(1.71), -78.1614421070),  # RuO4
        ([44, 8, 8, 8],
         [(0, 0, 0), (0, 0, 1.7), (1.47, 0, -0.85), (-1.47, 0, -0.85)],
         -62.5281411289),  # RuO3
        ([45, 9], [(0, 0, 0), (0, 0, 1.80)], -44.8136548780),  # RhF
        ([45, 17], [(0, 0, 0), (0, 0, 2.20)], -35.2211229168),  # RhCl
        ([46, 17, 17], [(0, 0, 0), (0, 0, 2.30), (0, 0, -2.30)], -59.4646491771),  # PdCl2
        ([46, 9, 9], [(0, 0, 0), (0, 0, 1.95), (0, 0, -1.95)], -78.6966563434),  # PdF2
        ([51, 9, 9, 9],
         [(0, 0, 0), (0, 0, 1.88), (1.78, 0, -0.6), (-1.78, 0, -0.6)],
         -75.3664492072),  # SbF3
        ([51, 17, 17, 17],
         [(0, 0, 0), (0, 0, 2.33), (2.2, 0, -0.75), (-2.2, 0, -0.75)],
         -46.4479613695),  # SbCl3
        ([52, 9, 9, 9, 9, 9, 9], [(0, 0, 0)] + _octa(1.82), -149.2768395396),  # TeF6
        ([52, 8, 8], [(0, 0, 0), (0, 0, 1.8), (1.7, 0, -0.6)], -38.4187534226),  # TeO2
        ([53, 17], [(0, 0, 0), (0, 0, 2.32)], -24.4450047669),  # ICl
        ([53, 9, 9, 9, 9, 9], [(0, 0, 0)] + _tbp(1.85), -128.8178410536),  # IF5
        ([54, 9, 9], [(0, 0, 0), (0, 0, 2.0), (0, 0, -2.0)], -62.8880352420),  # XeF2
        ([54, 8, 8, 8], [(0, 0, 0)] + _pl3(1.76), -62.9780463090),  # XeO3
    ]
    for Z, coords, ref in cases:
        r = msindo.run_msindo(Z, np.array(coords, float))
        assert r.converged, f"heavy parity Z={Z[0]}: SCF not converged"
        assert r.total_energy == pytest.approx(ref, abs=1e-6), (
            f"heavy parity Z={Z[0]}: got {r.total_energy}, expected {ref}"
        )


def test_fifth_row_sblock_and_4d_parity():
    """Rb, Sr (5s) + Y, Zr, Nb, Mo (4d transition metals) total-energy parity vs
    the MSINDO oracle.  Enabled by adding the 5th-row analogs to the alkali /
    alkaline-earth / transition-metal ENEG categories (atomic_reference.f: the
    Rb/Sr block + the Y-Cd block, which is term-for-term identical to Sc-Zn).
    References from the MSINDO oracle (examples/regression/msindo/build_oracle.sh)."""
    import numpy as np

    P = 0.8660254

    def _planar3(r):
        return [(r, 0.0, 0.0), (-r / 2, r * P, 0.0), (-r / 2, -r * P, 0.0)]

    def _tet(r):
        d = r / 3.0**0.5
        return [(d, d, d), (d, -d, -d), (-d, d, -d), (-d, -d, d)]

    def _octa(r):
        return [(r, 0, 0), (-r, 0, 0), (0, r, 0), (0, -r, 0), (0, 0, r), (0, 0, -r)]

    def _tbp(r):
        return _planar3(r) + [(0, 0, r), (0, 0, -r)]

    cases = [
        ([37, 9], [(0, 0, 0), (0, 0, 2.27)], -23.6798426651),  # RbF
        ([38, 9, 9], [(0, 0, 0), (0, 0, 2.10), (0, 0, -2.10)], -47.7033579329),  # SrF2
        ([39, 9, 9, 9], [(0, 0, 0)] + _planar3(2.0), -72.2032148014),  # YF3
        ([40, 9, 9, 9, 9], [(0, 0, 0)] + _tet(1.9), -97.1358343812),  # ZrF4
        ([41] + [9] * 5, [(0, 0, 0)] + _tbp(1.88), -122.4882160924),  # NbF5
        ([42] + [9] * 6, [(0, 0, 0)] + _octa(1.82), -149.1104634227),  # MoF6
    ]
    for Z, coords, ref in cases:
        r = msindo.run_msindo(Z, np.array(coords, float))
        assert r.total_energy == pytest.approx(ref, abs=1e-6), (
            f"5th-row parity Z={Z[0]}: got {r.total_energy}, expected {ref}"
        )


def test_homonuclear_4d_dimers_parity():
    """Nb₂, Mo₂ (homonuclear 4d dimers) reach the MSINDO oracle SCF state.

    These near-degenerate metal dimers converge to a *different* stationary point
    under the DIIS path (Mo₂ jumps to a lower spurious basin −14.5171, Nb₂ to a
    higher one −6.7264) than MSINDO finds.  Routing Nb/Mo to the Hückel-guess +
    WICHT path (msindo._MSINDO_TRAJECTORY_SCF, 2026-06-17) reaches the reference
    basin — the same default SCF MSINDO uses; their well-behaved halides (NbF₅,
    MoF₆) reproduce the oracle on this path too (test_fifth_row_…, no regression).

    Tolerance 5e-6: Mo₂'s sextuply-bonded valence space is so near-degenerate that
    its converged energy sits on a flat plateau where the energy-only stop
    (DELEN=1e-8) lands numpy and MSINDO's LAPACK ~1.2 µHa apart (the oracle's own
    per-cycle energy oscillates ±6e-8 there); the basin — not a sub-µHa match — is
    what this guards.  Nb₂ matches to ~2e-8.  References from the MSINDO oracle
    (examples/regression/msindo/build_oracle.sh; bond lengths Mo₂ 1.94 Å, Nb₂ 2.08 Å)."""
    import numpy as np

    cases = [
        ([41, 41], [(0, 0, -1.04), (0, 0, 1.04)], -6.9007186359),  # Nb₂
        ([42, 42], [(0, 0, -0.97), (0, 0, 0.97)], -14.4178688549),  # Mo₂
    ]
    for Z, coords, ref in cases:
        r = msindo.run_msindo(Z, np.array(coords, float))
        assert r.converged, f"dimer Z={Z[0]}: SCF not converged"
        assert r.total_energy == pytest.approx(ref, abs=5e-6), (
            f"4d-dimer parity Z={Z[0]}: got {r.total_energy}, expected {ref}"
        )


def test_param_bundle_covers_all_54_elements():
    """The bundled MSINDO parameter table (datas.f) carries all Z=1..54."""
    assert len(msindo._PARAMS) == 54
    assert all(str(z) in msindo._PARAMS for z in range(1, 55))
    # Spot-check a value against datas.f (C two-center s exponent).
    assert msindo.MUS[6] == pytest.approx(1.7874, abs=1e-12)


def test_supported_elements():
    """The MSINDO engine covers the full first six rows, H–Xe (Z 1–54): H-Kr
    (incl. the Sc-Zn 3d metals and Ga-Br 4th-row p-block), the 5th-row s-block
    Rb/Sr, the 4d transition metals Y-Cd (with the d¹⁰ Ag/Cd In-Xe-loop ENEG and
    the open-d Tc-Pd via the Hückel+WICHT SCF), and the 5p In-Xe ([Kr]/[Kr]4d¹⁰
    frozen core + 5d polarization).  All oracle-validated."""
    assert msindo._SUPPORTED == set(range(1, 55))


@pytest.mark.parametrize("name", ["ScF3", "TiF4", "CrF6", "CuF", "ZnO"])
def test_transition_metal_parity(name):
    """Closed-shell transition-metal molecules reproduce reference MSINDO,
    exercising the occupied-d ENEG/ATENG and the d-electron core penetration."""
    mol = next(m for m in _REF["molecules"] if m["name"] == name)
    Z = [_SYMBOL_Z[s] for s, *_ in mol["geometry"]]
    coords = [[x, y, z] for _s, x, y, z in mol["geometry"]]
    r = msindo.run_msindo(Z, coords)
    assert r.total_energy == pytest.approx(mol["reference"]["total_energy"], abs=1e-6)
    assert r.binding_energy == pytest.approx(
        mol["reference"]["binding_energy"], abs=1e-6
    )


# 1-octanol (OC8H18, 27 atoms, 54 basis functions) — a realistically sized,
# floppy C/H/O molecule.  The bare fixed-point SCF oscillates and never
# converges on a system this size; this guards the Pulay-DIIS accelerator that
# fixes it (run_msindo, CPL 73, 393 (1980)).  NOTE: the pinned energy is this
# engine's own converged *INDO* single-point value — an internal regression
# anchor, NOT a reference-MSINDO parity number (the reference octanol.out is an
# NDDO + D3 geometry optimisation, a different theory level; see CLAUDE.md §10).
_OCTANOL_Z = [8] + [6] * 8 + [1] * 18
_OCTANOL_XYZ = [
    [0.0, 0.0, 0.0],
    [0.0, 1.424, 0.0],
    [1.431, 1.959, 0.0],
    [1.485, 3.477, 0.0],
    [2.917, 4.012, 0.0],
    [2.972, 5.531, 0.0],
    [4.403, 6.066, 0.0],
    [4.458, 7.585, 0.0],
    [5.889, 8.120, 0.0],
    [-0.517, 1.769, -0.897],
    [-0.517, 1.769, 0.897],
    [1.959, 1.614, 0.897],
    [1.959, 1.614, -0.897],
    [0.958, 3.822, 0.897],
    [0.958, 3.822, -0.897],
    [3.444, 3.667, 0.897],
    [3.444, 3.667, -0.897],
    [2.445, 5.876, 0.897],
    [2.445, 5.876, -0.897],
    [4.931, 5.720, 0.897],
    [4.931, 5.720, -0.897],
    [3.930, 7.931, 0.897],
    [3.930, 7.931, -0.897],
    [6.416, 7.775, 0.897],
    [6.416, 7.775, -0.897],
    [5.929, 9.208, 0.0],
    [-0.539, -0.314, 0.741],
]


def test_diis_converges_large_molecule():
    """DIIS drives a 27-atom C/H/O SCF to convergence (bare iteration diverges)."""
    r = msindo.run_msindo(_OCTANOL_Z, _OCTANOL_XYZ, max_iter=200)
    assert r.converged
    assert r.n_iter < 60
    # Internal regression anchor (this engine's INDO single-point, NOT a
    # reference-MSINDO value — see comment above).
    assert r.total_energy == pytest.approx(-73.7831132577, abs=1e-6)


@pytest.mark.parametrize("backend", ["native", "python"])
@pytest.mark.parametrize(
    ("name", "Z", "xyz", "reference_energy", "iteration_range"),
    [
        pytest.param(
            "acetamide",
            _ACETAMIDE_Z,
            _ACETAMIDE_XYZ,
            -39.7743880637,
            range(90, 111),
            id="acetamide",
        ),
        pytest.param(
            "cyclopropene",
            _CYCLOPROPENE_Z,
            _CYCLOPROPENE_XYZ,
            -19.4510059183,
            range(120, 141),
            id="cyclopropene",
        ),
        pytest.param(
            "propanamide",
            [6, 8, 7, 6, 6, 1, 1, 1, 1, 1, 1, 1],
            [
                [0.000, 0.543, 0.000],
                [0.000, 1.765, 0.000],
                [0.000, -0.633, 0.000],
                [1.273, -0.048, -0.449],
                [2.461, -0.004, 0.494],
                [1.473, -0.487, -1.430],
                [1.025, 0.995, -0.656],
                [3.361, 0.271, -0.058],
                [2.637, -0.998, 0.912],
                [2.269, 0.680, 1.316],
                [-0.819, -1.151, -0.283],
                [0.819, -1.151, -0.283],
            ],
            -46.7216952448,
            range(90, 111),
            id="propanamide",
        ),
        pytest.param(
            "pyridazine",
            [7, 7, 6, 6, 6, 6, 1, 1, 1, 1],
            [
                [0.000, 0.659, 0.000],
                [0.000, -0.659, 0.000],
                [1.170, 1.347, 0.000],
                [-1.170, -1.347, 0.000],
                [-1.170, 1.347, 0.000],
                [1.170, -1.347, 0.000],
                [1.086, 2.424, 0.000],
                [-1.086, -2.424, 0.000],
                [2.078, -0.818, 0.000],
                [-2.078, 0.818, 0.000],
            ],
            -44.7506965235,
            range(40, 61),
            id="pyridazine",
        ),
    ],
)
def test_light_rhf_checks_wicht_candidate_and_keeps_lower_strict_basin(
    monkeypatch, backend, name, Z, xyz, reference_energy, iteration_range
):
    """Ordinary RHF must compare strict Hcore and WICHT-seeded solutions."""
    import numpy as np

    if backend == "native":
        if msindo._cpp_msindo_kernel(nddo=False) is None:
            pytest.skip("native MSINDO extension is unavailable")

        def _forbid_python_scf(*_args, **_kwargs):
            raise AssertionError("native MSINDO silently fell back to Python")

        monkeypatch.setattr(msindo, "_scf_rhf_molecular", _forbid_python_scf)
    else:
        monkeypatch.setattr(
            msindo, "_cpp_msindo_kernel", lambda *, nddo=False: None
        )

    result = msindo.run_msindo(Z, xyz, max_iter=200, conv_tol=1e-9)
    coords_bohr = np.asarray(xyz) * msindo.ANGSTROM_TO_BOHR
    blocks, nsto = msindo._atom_blocks(Z)
    H, G = msindo._build_core_and_gamma(Z, coords_bohr, blocks, nsto)
    final_fock = msindo._build_fock(H, G, result.density, blocks, Z)
    commutator = final_fock @ result.density - result.density @ final_fock

    assert result.converged, f"{backend} {name} did not converge"
    assert result.total_energy == pytest.approx(reference_energy, abs=1e-9)
    assert result.n_iter in iteration_range
    assert np.max(np.abs(commutator)) < 1e-9


@pytest.mark.parametrize("backend", ["native", "python"])
def test_acetamide_root_selection_is_stable_across_primary_caps(monkeypatch, backend):
    """Crossing primary convergence must not disable alternate-basin search."""
    if backend == "native":
        if msindo._cpp_msindo_kernel(nddo=False) is None:
            pytest.skip("native MSINDO extension is unavailable")

        def _forbid_python_scf(*_args, **_kwargs):
            raise AssertionError("native MSINDO silently fell back to Python")

        monkeypatch.setattr(msindo, "_scf_rhf_molecular", _forbid_python_scf)
    else:
        monkeypatch.setattr(
            msindo, "_cpp_msindo_kernel", lambda *, nddo=False: None
        )

    results = [
        msindo.run_msindo(
            _ACETAMIDE_Z,
            _ACETAMIDE_XYZ,
            max_iter=cap,
            conv_tol=1e-9,
        )
        for cap in (20, 33, 200)
    ]

    assert all(result.converged for result in results)
    assert [result.total_energy for result in results] == pytest.approx(
        [-39.7743880637] * 3, abs=1e-9
    )


@pytest.mark.parametrize("backend", ["native", "python"])
def test_converged_non_light_primary_skips_unvalidated_probe(monkeypatch, backend):
    """AgH keeps its established DIIS result without a 2000-cycle WICHT cost."""
    if backend == "native":
        if msindo._cpp_msindo_kernel(nddo=False) is None:
            pytest.skip("native MSINDO extension is unavailable")

        def _forbid_python_scf(*_args, **_kwargs):
            raise AssertionError("native MSINDO silently fell back to Python")

        monkeypatch.setattr(msindo, "_scf_rhf_molecular", _forbid_python_scf)
    else:
        monkeypatch.setattr(
            msindo, "_cpp_msindo_kernel", lambda *, nddo=False: None
        )

    result = msindo.run_msindo(
        [47, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 1.62]], conv_tol=1e-9
    )

    assert result.converged
    assert result.total_energy == pytest.approx(-13.3529501543, abs=1e-9)
    assert result.n_iter == 5


@pytest.mark.parametrize("backend", ["native", "python"])
def test_acetamide_analytic_gradient_uses_selected_basin(monkeypatch, backend):
    """The analytic gradient must use the density selected by the energy path."""
    import numpy as np
    from vibeqc.semiempirical.methods import msindo_gradient_analytic as grad_mod

    if backend == "native":
        if (
            msindo._cpp_msindo_kernel(nddo=False) is None
            or grad_mod._cpp_gradient_kernel() is None
        ):
            pytest.skip("native MSINDO extension is unavailable")

        def _forbid_python_scf(*_args, **_kwargs):
            raise AssertionError("native MSINDO silently fell back to Python")

        monkeypatch.setattr(msindo, "_scf_rhf_molecular", _forbid_python_scf)
    else:
        monkeypatch.setattr(
            msindo, "_cpp_msindo_kernel", lambda *, nddo=False: None
        )
        monkeypatch.setattr(grad_mod, "_cpp_gradient_kernel", lambda: None)

    xyz = np.asarray(_ACETAMIDE_XYZ, dtype=float)
    gradient = grad_mod.msindo_gradient_analytic(
        _ACETAMIDE_Z, xyz, conv_tol=1e-10
    )
    step = 1e-4
    plus = xyz.copy()
    minus = xyz.copy()
    plus[2, 0] += step
    minus[2, 0] -= step
    e_plus = msindo.run_msindo(
        _ACETAMIDE_Z, plus, conv_tol=1e-10
    ).total_energy
    e_minus = msindo.run_msindo(
        _ACETAMIDE_Z, minus, conv_tol=1e-10
    ).total_energy
    fd_nx = (e_plus - e_minus) / (2.0 * step) / msindo.ANGSTROM_TO_BOHR

    assert gradient[2, 0] == pytest.approx(0.14627202, abs=1e-6)
    assert gradient[2, 0] == pytest.approx(fd_nx, abs=1e-6)


def test_heavy_analytic_gradient_preserves_stationary_diis_route(monkeypatch):
    """IID 191 must not feed nonstationary heavy WICHT densities to gradients."""
    import numpy as np
    from vibeqc.semiempirical.methods import msindo_gradient_analytic as grad_mod

    Z = [43, 17]
    xyz = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.25]]
    native_kernel = grad_mod._cpp_gradient_kernel()
    native_gradient = None
    if native_kernel is not None:
        native_gradient = grad_mod.msindo_gradient_analytic(
            Z, xyz, conv_tol=1e-10
        )

    def _forbid_selector(*_args, **_kwargs):
        raise AssertionError("heavy analytic gradient used the energy selector")

    monkeypatch.setattr(grad_mod, "_cpp_gradient_kernel", lambda: None)
    monkeypatch.setattr(grad_mod, "_scf_rhf_molecular", _forbid_selector)
    python_gradient = grad_mod.msindo_gradient_analytic(
        Z, xyz, conv_tol=1e-10
    )

    assert python_gradient[1, 2] == pytest.approx(0.07541543, abs=1e-8)
    if native_gradient is not None:
        assert np.max(np.abs(native_gradient - python_gradient)) < 1e-8


def test_non_singlet_analytic_gradient_stays_outside_rhf_selector(monkeypatch):
    """IID 191 must not alter the pre-existing non-singlet gradient route."""
    import numpy as np
    from vibeqc.semiempirical.methods import msindo_gradient_analytic as grad_mod

    def _forbid_selector(*_args, **_kwargs):
        raise AssertionError("non-singlet gradient used the RHF basin selector")

    monkeypatch.setattr(grad_mod, "_scf_rhf_molecular", _forbid_selector)
    gradient = grad_mod.msindo_gradient_analytic(
        [1, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.741]], multiplicity=3
    )

    assert gradient.shape == (2, 3)
    assert np.all(np.isfinite(gradient))


def test_tetrazine_recovers_diis_cycle_via_strict_wicht_seed():
    """The archived Thiel frame must finish at a stationary MSINDO root."""
    import numpy as np

    Z = [7, 7, 7, 7, 6, 6, 1, 1]
    coords_bohr = np.asarray(
        [
            [0.0, 2.26767119, 0.0],
            [2.26767119, 0.0, 0.0],
            [0.0, -2.26767119, 0.0],
            [-2.26767119, 0.0, 0.0],
            [2.26767119, 2.26767119, 0.0],
            [-2.26767119, 2.26767119, 0.0],
            [3.91173280, 3.91173280, 0.0],
            [-3.91173280, 3.91173280, 0.0],
        ]
    )
    xyz = [
        list(row / msindo.ANGSTROM_TO_BOHR)
        for row in coords_bohr
    ]
    blocks, nsto = msindo._atom_blocks(Z)
    H, G = msindo._build_core_and_gamma(Z, coords_bohr, blocks, nsto)
    nocc = sum(msindo.eff_core_charge(z) for z in Z) // 2
    _P, _F, _e, _eps, cold_converged, cold_iter = msindo._scf_rhf(
        H, G, blocks, Z, nocc, max_iter=200, conv_tol=1e-9
    )

    result = msindo.run_msindo(Z, xyz, max_iter=200)
    final_fock = msindo._build_fock(H, G, result.density, blocks, Z)
    commutator = final_fock @ result.density - result.density @ final_fock

    assert not cold_converged
    assert cold_iter == 200
    assert result.converged
    assert result.n_iter == 270
    assert result.total_energy == pytest.approx(-50.737177619704, abs=1e-10)
    assert np.trace(result.density) == pytest.approx(30.0)
    assert np.max(np.abs(commutator)) < 1e-9


# ---- open-shell UHF (fockop.f), s/p elements ----


def test_uhf_reduces_to_rhf_for_closed_shell():
    """The UHF path with PA=PB must reproduce the RHF energy bit-for-bit.

    This is the load-bearing check on the spin generalisation: setting
    PA = PB = P/2 in _build_fock_uhf is algebraically identical to _build_fock,
    so a closed-shell singlet forced through _run_uhf must equal the RHF result.
    """
    import numpy as np

    Z = [8, 1, 1]
    xyz = [[0.0, 0.0, 0.0], [0.7572, 0.5865, 0.0], [-0.7572, 0.5865, 0.0]]
    rhf = msindo.run_msindo(Z, xyz)  # RHF path (multiplicity 1, even nelec)

    C = np.asarray(xyz, float) * msindo.ANGSTROM_TO_BOHR
    blocks, nsto = msindo._atom_blocks(Z)
    cz = [msindo.eff_core_charge(z) for z in Z]
    H, G = msindo._build_core_and_gamma(Z, C, blocks, nsto)
    uhf = msindo._run_uhf(
        Z, H, G, blocks, sum(cz), 1, msindo._core_repulsion(C, cz), 200, 1e-9
    )
    assert uhf.converged
    assert uhf.total_energy == pytest.approx(rhf.total_energy, abs=1e-9)
    assert uhf.electronic_energy == pytest.approx(rhf.electronic_energy, abs=1e-9)


def test_uhf_oh_doublet_matches_oracle():
    """OH• doublet (the simplest radical) reproduces reference MSINDO UHF."""
    r = msindo.run_msindo([8, 1], [[0, 0, 0], [0, 0, 0.97]], multiplicity=2)
    assert r.converged
    assert r.total_energy == pytest.approx(-16.3330865643, abs=1e-6)
    assert r.binding_energy == pytest.approx(-0.177045199, abs=1e-6)


def test_uhf_inconsistent_multiplicity_raises():
    """A multiplicity incompatible with the electron count is rejected."""
    # OH has an odd electron count; a singlet (multiplicity 1) is impossible.
    with pytest.raises(ValueError):
        msindo.run_msindo([8, 1], [[0, 0, 0], [0, 0, 0.97]], multiplicity=1)


def test_uhf_dshell_radical_converges():
    """Open-shell on a d-shell element (SH radical) now converges."""
    r = msindo.run_msindo([16, 1], [[0, 0, 0], [0, 0, 1.34]], multiplicity=2)
    assert r.converged
    assert r.total_energy < 0


# Heavy-element open-shell (UHF) parity, one+ molecule per _MSINDO_TRAJECTORY_SCF
# element (Nb–Pd 4d, Sb–Xe 5p) plus triplets.  These reach the reference SCF
# stationary point only via the MSINDO-faithful Hückel-guess + WICHT-damped UHF
# SCF (msindo._scf_uhf_msindo, dispatched through _MSINDO_TRAJECTORY_SCF in
# _run_uhf) — the production DIIS path lands on a different stationary point, the
# same basin-selection problem the closed-shell heavy elements have.  The
# one-centre d Fock is the faithful fockop.f port (_add_einzi_dblock_uhf), which
# the closed-shell EINZI-per-spin shortcut gets wrong.  References from the MSINDO
# oracle (CARTES UHF MULTIP m; fast-converging linear/symmetric geometries — see
# docs/user_guide/msindo.md).
_UHF_HEAVY = [
    # (name, Z, coords Å, multiplicity, oracle_total)
    ("NbO2", [41, 8, 8], [(0, 0, 0), (0, 0, 1.70), (0, 0, -1.70)], 2, -35.4496629687),
    ("MoCl", [42, 17], [(0, 0, 0), (0, 0, 2.30)], 2, -21.4390039753),
    ("MoO2", [42, 8, 8], [(0, 0, 0), (0, 0, 1.70), (0, 0, -1.70)], 3, -38.9214397584),  # triplet
    ("TcCl2", [43, 17, 17], [(0, 0, 0), (0, 0, 2.25), (0, 0, -2.25)], 2, -39.2198082755),
    ("RuCl", [44, 17], [(0, 0, 0), (0, 0, 2.20)], 2, -29.6808033415),
    ("RuF2", [44, 9, 9], [(0, 0, 0), (0, 0, 1.82), (0, 0, -1.82)], 3, -62.9657257886),  # triplet
    ("RhF2", [45, 9, 9], [(0, 0, 0), (0, 0, 1.85), (0, 1.60, -0.92)], 2, -68.6846477540),
    ("PdCl", [46, 17], [(0, 0, 0), (0, 0, 2.30)], 2, -45.6424788740),
    ("SbF2", [51, 9, 9], [(0, 0, 0), (0, 0, 1.88), (0, 0, -1.88)], 2, -51.5332968692),
    ("TeCl", [52, 17], [(0, 0, 0), (0, 0, 2.25)], 2, -20.8891995841),
    ("ICl2", [53, 17, 17], [(0, 0, 0), (0, 0, 2.32), (0, 0, -2.32)], 2, -38.5687534466),
    ("XeOF", [54, 8, 9], [(0, 0, 0), (0, 0, 1.75), (0, 0, -2.0)], 2, -55.1886280567),
]


@pytest.mark.parametrize("name,Z,coords,mult,ref", _UHF_HEAVY, ids=[c[0] for c in _UHF_HEAVY])
def test_uhf_heavy_4d_5p_total_energy_parity(name, Z, coords, mult, ref):
    """Open-shell UHF total-energy parity vs the MSINDO oracle for the heavy
    d/p-block radicals (Tc–Pd, Sb–Xe), doublets and a triplet."""
    import numpy as np

    r = msindo.run_msindo(Z, np.array(coords, float), multiplicity=mult)
    assert r.converged, f"{name}: SCF not converged"
    assert r.total_energy == pytest.approx(ref, abs=1e-6), (
        f"{name}: got {r.total_energy}, expected {ref}"
    )


def test_uhf_msindo_reduces_to_rhf_heavy():
    """For a closed-shell heavy molecule the trajectory dispatch is consistent:
    forced UHF(M=1) (_scf_uhf_msindo) reduces to RHF (_scf_rhf_msindo).  Uses a
    linear molecule (PdCl₂) where fockop.f's one-centre d Fock reduces exactly to
    fockcl.f's (the off-axis d-d couplings that differ vanish by symmetry)."""
    import numpy as np

    Z = [46, 17, 17]
    xyz = [[0, 0, 0], [0, 0, 2.30], [0, 0, -2.30]]
    rhf = msindo.run_msindo(Z, np.array(xyz, float))  # trajectory RHF path

    C = np.asarray(xyz, float) * msindo.ANGSTROM_TO_BOHR
    blocks, nsto = msindo._atom_blocks(Z)
    cz = [msindo.eff_core_charge(z) for z in Z]
    H, G = msindo._build_core_and_gamma(Z, C, blocks, nsto)
    uhf = msindo._run_uhf(
        Z, H, G, blocks, sum(cz), 1, msindo._core_repulsion(C, cz), 200, 1e-9
    )
    assert uhf.converged
    assert uhf.total_energy == pytest.approx(rhf.total_energy, abs=1e-7)


# --------------------------------------------------------------------------- #
# NDDO mode — separate parametrization (nddoparam.f).                          #
#                                                                              #
# MSINDO's NDDO mode (its default) is a separate parameter set: IF(NDDO) CALL  #
# NDDOPARAM selects the Slater exponents / IPs / K betas / AL.  With those     #
# exponents the EXISTING one-centre machinery reproduces the oracle NDDO       #
# GMUNU exactly. These pin the parameter bundle and context-local overlay.      #
# --------------------------------------------------------------------------- #

# Oracle NDDO one-centre 2e integrals (PRINTOPTS=GMAT on HF, CARTES RHF NDDO):
# the GMUNU one-centre block for H (1s) and F (2s2p).  Reproduced to 5 decimals
# by one_center_2e under the NDDO parameter set (the cracked "NDDO crux").
_NDDO_OC_H = {"GSS": 0.59869}
_NDDO_OC_F = {
    "GSS": 0.92313,
    "GSP": 0.81668,
    "HSP": 0.17299,
    "GPP": 0.79761,
    "GP2": 0.71164,
    "HPP": 0.04298,
}


def test_nddo_one_center_2e_matches_oracle():
    """Under the NDDO parameter overrides, one_center_2e reproduces the oracle
    NDDO GMUNU one-centre block (H + F) to 5 decimals — the verification that
    NDDO's one-centre integrals are simply the INDO machinery on the NDDO
    exponents (nddoparam.f), not a separate recompute."""
    with msindo._nddo_params():
        oc_h = msindo.one_center_2e(1)
        oc_f = msindo.one_center_2e(9)
    for k, v in _NDDO_OC_H.items():
        assert oc_h[k] == pytest.approx(v, abs=1e-5), f"H {k}"
    for k, v in _NDDO_OC_F.items():
        assert oc_f[k] == pytest.approx(v, abs=1e-5), f"F {k}"


def test_nddo_params_override_and_restore():
    """_nddo_params() selects NDDO exponents/IPs and restores INDO on exit."""
    indo_muse_f, indo_ipots_f = msindo.MUSE[9], msindo.IPOTS[9]
    with msindo._nddo_params():
        assert msindo.MUSE[9] == pytest.approx(2.5411)  # NDDO F MUSE
        assert msindo.MUSE[1] == pytest.approx(0.9579)  # NDDO H MUSE
        assert msindo.IPOTS[9] == pytest.approx(-1.7860)  # NDDO F IPOTS
    # Restored.
    assert msindo.MUSE[9] == indo_muse_f
    assert msindo.IPOTS[9] == indo_ipots_f


def test_nddo_parameter_context_is_thread_local_and_restores():
    """Overlapping NDDO contexts cannot corrupt another calculation's tables."""
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    indo_muse_f = msindo.MUSE[9]
    nddo_muse_f = msindo._NDDO_PARAMS["9"]["MUSE"]
    first_entered = Event()
    second_entered = Event()
    allow_second_read = Event()

    def first():
        with msindo._nddo_params():
            first_entered.set()
            assert second_entered.wait(timeout=5.0)
            return msindo.MUSE[9]

    def second():
        assert first_entered.wait(timeout=5.0)
        with msindo._nddo_params():
            second_entered.set()
            assert allow_second_read.wait(timeout=5.0)
            return msindo.MUSE[9]

    with ThreadPoolExecutor(max_workers=2) as pool:
        first_result = pool.submit(first)
        assert first_entered.wait(timeout=5.0)
        # This thread represents an overlapping ordinary INDO calculation.
        assert msindo.MUSE[9] == indo_muse_f
        second_result = pool.submit(second)
        assert first_result.result(timeout=5.0) == nddo_muse_f
        allow_second_read.set()
        assert second_result.result(timeout=5.0) == nddo_muse_f

    assert msindo.MUSE[9] == indo_muse_f


def test_nddo_param_bundle_covers_h_through_cl():
    """The NDDO bundle is parametrized for H, Li–F, Na–Cl (noble gases excluded,
    matching einzentren.f's 'keine Berechnung fuer Edelgase')."""
    assert msindo._SUPPORTED_NDDO == frozenset(
        {1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17}
    )
    # No noble gases.
    assert not ({2, 10, 18} & msindo._SUPPORTED_NDDO)


def test_nddo_spdd_kernel_matches_source_and_radial_derivative():
    """The asymmetric dipole/d-monopole kernel follows SPDD_SI/DSPDD_SI."""
    import numpy as np

    distance = 2.10 * msindo.ANGSTROM_TO_BOHR
    source = {
        (13, 17): (0.0550522676034613, -0.0156655974038640),
        (17, 13): (0.0362017794645199, -0.0112884923676598),
    }
    with msindo._nddo_params():
        for atoms, (value_ref, derivative_ref) in source.items():
            value = msindo.nddo_spdd_si(*atoms, distance)
            derivative = msindo.nddo_dspdd_si(*atoms, distance)
            assert value == pytest.approx(value_ref, abs=1e-14)
            assert derivative == pytest.approx(derivative_ref, abs=1e-14)
            step = 1e-5
            finite_difference = (
                msindo.nddo_spdd_si(*atoms, distance + step)
                - msindo.nddo_spdd_si(*atoms, distance - step)
            ) / (2.0 * step)
            assert derivative == pytest.approx(finite_difference, abs=1e-10)
            assert np.isfinite(value)


def test_nddo_spdd_fock_branches_match_source_equations():
    """Al-Cl d-diagonal and s-p Fock updates retain source factors/signs."""
    import numpy as np

    atomic_numbers = [13, 17]
    coordinates = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.10]])
    with msindo._nddo_params():
        coordinates_bohr = coordinates * msindo.ANGSTROM_TO_BOHR
        blocks, nsto = msindo._atom_blocks(atomic_numbers)
        fock_extra = msindo._nddo_fock_extra(
            atomic_numbers, coordinates_bohr, blocks
        )
        lo_al, _ = blocks[0]
        lo_cl, _ = blocks[1]

        p_dipole = np.zeros((nsto, nsto))
        p_dipole[lo_cl + 3, lo_cl] = p_dipole[lo_cl, lo_cl + 3] = 0.2
        f_dipole, _ = fock_extra(p_dipole)
        ddsp = msindo.nddo_spdd_si(17, 13, 2.10 * msindo.ANGSTROM_TO_BOHR)
        for d in range(4, 9):
            assert f_dipole[lo_al + d, lo_al + d] == pytest.approx(
                -2.0 * ddsp * 0.2, abs=1e-14
            )

        p_d_monopole = np.zeros((nsto, nsto))
        for d in range(4, 9):
            p_d_monopole[lo_cl + d, lo_cl + d] = 0.3
        f_d_monopole, _ = fock_extra(p_d_monopole)
        spdd = msindo.nddo_spdd_si(13, 17, 2.10 * msindo.ANGSTROM_TO_BOHR)
        assert f_d_monopole[lo_al + 3, lo_al] == pytest.approx(
            spdd * 1.5, abs=1e-14
        )


def test_nddo_open_shell_rejected():
    """NDDO mode is closed-shell (RHF) only — open-shell NDDO (UHF) is not
    implemented, so a multiplicity > 1 (or odd electron count) raises cleanly."""
    with pytest.raises(NotImplementedError, match="closed-shell"):
        # NO• radical-like: O + N with multiplicity 2.
        msindo.run_msindo([8, 7], [[0, 0, 0], [0, 0, 1.15]], multiplicity=2, nddo=True)


def test_nddo_unsupported_element_named():
    """An element outside the NDDO parameter set is reported as such."""
    with pytest.raises(NotImplementedError, match="NDDO mode is parametrized"):
        # P (Z=15) is fine; Ca (Z=20) is not NDDO-parametrized.
        msindo.run_msindo([20, 9, 9], [[0, 0, 0], [0, 0, 2.0], [0, 0, -2.0]], nddo=True)


def test_nddo_core_hamiltonian_matches_oracle_hf():
    """The NDDO core Hamiltonian (INDO core under the NDDO parameter set + the
    new HSP s-pσ term) reproduces the oracle CORE HAMILTONIAN for HF to the
    dump's F9.4 precision — the verification that the NDDO 1e-core is complete
    (ENEG + V2INT monopole core + VCORRK penetration + HSP, all on the NDDO
    params).  Basis order: H s, F s, F px, F py, F pz."""
    import numpy as np

    Z = [1, 9]
    C = np.array([[0, 0, 0], [0, 0, 0.917]], float) * msindo.ANGSTROM_TO_BOHR
    # Oracle dump: CARTES RHF NDDO PRINTOPTS=HMAT, "CORE HAMILTONIAN" (F9.4).
    oracle = np.array(
        [
            [-4.1842, -0.2705, 0.0000, 0.0000, 0.2332],
            [-0.2705, -6.8705, 0.0000, 0.0000, 0.0789],
            [0.0000, 0.0000, -5.5300, 0.0000, 0.0000],
            [0.0000, 0.0000, 0.0000, -5.5300, 0.0000],
            [0.2332, 0.0789, 0.0000, 0.0000, -5.5491],
        ]
    )
    with msindo._nddo_params():
        blocks, nsto = msindo._atom_blocks(Z)
        H, _G = msindo._build_core_and_gamma(Z, C, blocks, nsto, nddo=True)
    assert np.max(np.abs(H - oracle)) < 1.5e-4, f"max diff {np.max(np.abs(H - oracle))}"


@pytest.mark.parametrize(
    "name,Z,coords,ref",
    [
        # Reference MSINDO NDDO total energies at these exact geometries (oracle,
        # CARTES RHF NDDO).  HF/H₂O/CH₄ at the locked-target geometries; N₂/CO at
        # z=1.098/1.128 (the oracle totals there — the locked targets are at a
        # marginally different bond length).
        ("HF", [1, 9], [[0, 0, 0], [0, 0, 0.917]], -23.0436304974),
        (
            "H2O",
            [8, 1, 1],
            [[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]],
            -17.0717535443,
        ),
        (
            "CH4",
            [6, 1, 1, 1, 1],
            [
                [0, 0, 0],
                [0.6276, 0.6276, 0.6276],
                [-0.6276, -0.6276, 0.6276],
                [0.6276, -0.6276, -0.6276],
                [-0.6276, 0.6276, -0.6276],
            ],
            -8.2337262049,
        ),
        ("N2", [7, 7], [[0, 0, 0], [0, 0, 1.098]], -19.8447373916),
        ("CO", [6, 8], [[0, 0, 0], [0, 0, 1.128]], -21.6762159213),
        ("AlCl", [13, 17], [[0, 0, 0], [0, 0, 2.10]], -16.4566380534),
    ],
)
def test_nddo_total_energy_matches_oracle(name, Z, coords, ref):
    """``run_msindo(nddo=True)`` reproduces reference MSINDO NDDO total energies
    to ≤1 µHa — the separate NDDO parametrization (nddoparam.f) + the HSP
    one-centre core + the two-centre multipole 2e Fock (dipole-monopole
    nddofockcl + dipole-dipole spspfockcl). Spans single-p (HF/H₂O/CH₄),
    p-p (N₂/CO), and source SPDD d-monopole (AlCl) systems."""
    r = msindo.run_msindo(Z, coords, nddo=True)
    assert r.converged
    assert r.total_energy == pytest.approx(ref, abs=1e-6)


# --------------------------------------------------------------------------- #
# Molecular nuclear gradient + geometry optimization.                         #
# --------------------------------------------------------------------------- #


def test_msindo_molecular_gradient_matches_oracle():
    """The finite-difference molecular nuclear gradient reproduces the oracle's
    analytic gradient (CARTES RHF CARTOPT ANALY GRADONLY) on a distorted
    (off-equilibrium) H₂O — Ha/bohr, to the oracle dump's precision."""
    import numpy as np

    Z = [8, 1, 1]
    C = [[0, 0, 0], [0.95, 0.0, 0.10], [-0.20, 0.92, 0.0]]
    # Oracle "FIRST DERIVATIVES [H/BOHR]" at this geometry.
    oracle = np.array(
        [
            [-0.00460, 0.01701, -0.00010],
            [-0.00776, 0.00693, -0.00066],
            [0.01236, -0.02394, 0.00075],
        ]
    )
    g = msindo.msindo_gradient_fd(Z, C)
    assert np.max(np.abs(g - oracle)) < 2e-4, f"max diff {np.max(np.abs(g - oracle))}"


def test_msindo_pair_blocks_derivatives_match_fd():
    """Phase 2: local-frame R-derivatives of per-pair blocks match finite
    differences of the global-frame _pair_blocks to ~1e-8.

    Validates the s-s element of HK1/HL1/HKL2 for representative pairs
    (H-H, O-H, O-O, C-O, N-N) at several distances."""
    import numpy as np
    from vibeqc.semiempirical.methods.msindo import _pair_blocks
    from vibeqc.semiempirical.methods.msindo_pair_deriv import _pair_blocks_deriv

    h = 1e-4
    for zk, zl in [(1, 1), (8, 1), (8, 8), (6, 8), (7, 7)]:
        for R in [0.95, 1.5, 2.5]:
            rk = np.array([0.0, 0.0, 0.0])
            rl = np.array([R, 0.0, 0.0])
            E = np.array([1.0, 0.0, 0.0])
            pd = _pair_blocks_deriv(zk, zl, rk, rl)
            pb_p = _pair_blocks(zk, zl, rk, rk + (R + h) * E)
            pb_m = _pair_blocks(zk, zl, rk, rk + (R - h) * E)
            # s-s elements are directly comparable (s orbitals don't rotate)
            fd_Hk = (pb_p.HK1[0, 0] - pb_m.HK1[0, 0]) / (2 * h)
            fd_Hl = (pb_p.HL1[0, 0] - pb_m.HL1[0, 0]) / (2 * h)
            fd_Core = (pb_p.HKL2[0, 0] - pb_m.HKL2[0, 0]) / (2 * h)
            assert abs(fd_Hk - pd.dHK1_local[0, 0]) < 1e-8, (
                f"{zk},{zl} R={R}: dHK1 FD={fd_Hk:.10f} AN={pd.dHK1_local[0, 0]:.10f}"
            )
            assert abs(fd_Hl - pd.dHL1_local[0, 0]) < 1e-8
            assert abs(fd_Core - pd.dHKL2_local[0, 0]) < 1e-8


def test_msindo_analytic_gradient_matches_fd():
    """Phase 4: analytic nuclear gradient matches FD for H2, CO, and H2O."""
    import numpy as np
    from vibeqc.semiempirical.methods.msindo import msindo_gradient_fd
    from vibeqc.semiempirical.methods.msindo_gradient_analytic import (
        msindo_gradient_analytic,
    )

    # H2 (s-only)
    Z = [1, 1]
    C = [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]
    g_fd = msindo_gradient_fd(Z, C, step=1e-4)
    g_an = msindo_gradient_analytic(Z, C)
    assert np.max(np.abs(g_an - g_fd)) < 1e-7, (
        f"H2 max diff {np.max(np.abs(g_an - g_fd))}"
    )

    # CO (p-orbitals, two-center)
    Z = [6, 8]
    C = [[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]]
    g_fd = msindo_gradient_fd(Z, C, step=1e-4)
    g_an = msindo_gradient_analytic(Z, C)
    assert np.max(np.abs(g_an - g_fd)) < 1e-7, (
        f"CO max diff {np.max(np.abs(g_an - g_fd))}"
    )

    # H2O (3-atom, p-orbitals, non-axial)
    Z = [8, 1, 1]
    C = [[0, 0, 0], [0.95, 0.0, 0.10], [-0.20, 0.92, 0.0]]
    g_fd = msindo_gradient_fd(Z, C, step=1e-4)
    g_an = msindo_gradient_analytic(Z, C)
    assert np.max(np.abs(g_an - g_fd)) < 1e-7, (
        f"H2O max diff {np.max(np.abs(g_an - g_fd))}"
    )


def test_msindo_nddo_analytic_gradient_matches_exact_energy_fd_ladder():
    """NDDO differentiates every term in the NDDO energy it reports (#538)."""
    import numpy as np
    from vibeqc.semiempirical.methods.msindo import (
        ANGSTROM_TO_BOHR,
        msindo_gradient_fd,
    )
    from vibeqc.semiempirical.methods.msindo_gradient_analytic import (
        msindo_gradient_analytic,
    )

    cases = [
        (
            "HF-axial",
            [1, 9],
            [[0, 0, 0], [0, 0, 0.917]],
            [[0.0, 0.0, 0.0047202298], [0.0, 0.0, -0.0047202298]],
        ),
        (
            "H2O-nonaxial",
            [8, 1, 1],
            [[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]],
            [
                [-0.0, 0.0, 0.0003840477],
                [0.0, -0.0070198950, -0.0001920239],
                [-0.0, 0.0070198950, -0.0001920239],
            ],
        ),
        (
            "CO-axial",
            [6, 8],
            [[0, 0, 0], [0, 0, 1.128]],
            [[-0.0, 0.0, 0.1476292686], [0.0, -0.0, -0.1476292686]],
        ),
    ]
    for name, atomic_numbers, coordinates, source_gradient in cases:
        analytic = msindo_gradient_analytic(
            atomic_numbers,
            coordinates,
            nddo=True,
            conv_tol=1e-12,
        )
        assert np.max(np.abs(np.sum(analytic, axis=0))) < 1e-10, name
        # Frozen MSINDO 2025 source oracle: CARTES RHF NDDO CARTOPT ANALY
        # GRADONLY NOSYM DELEN 1D-12. The executable is never a test runtime
        # dependency; its FIRST DERIVATIVES values are in Hartree/bohr.
        assert np.max(np.abs(analytic - np.asarray(source_gradient))) < 5e-8, name
        for step_bohr in (2e-4, 1e-4, 5e-5):
            finite_difference = msindo_gradient_fd(
                atomic_numbers,
                coordinates,
                nddo=True,
                step=step_bohr / ANGSTROM_TO_BOHR,
                conv_tol=1e-12,
            )
            residual = np.max(np.abs(analytic - finite_difference))
            assert residual < 1e-6, (
                f"{name}, h={step_bohr} bohr: {residual:.3e} Ha/bohr"
            )


def test_msindo_nddo_frozen_core_analytic_gradient_matches_exact_energy_fd():
    """Na exercises the NDDO frozen-core parameter and penetration branches."""
    import numpy as np
    from vibeqc.semiempirical.methods.msindo import (
        ANGSTROM_TO_BOHR,
        msindo_gradient_fd,
    )
    from vibeqc.semiempirical.methods.msindo_gradient_analytic import (
        msindo_gradient_analytic,
    )

    atomic_numbers = [11, 1]
    coordinates = [[0.0, 0.0, 0.0], [0.21, -0.12, 1.89]]
    analytic = msindo_gradient_analytic(
        atomic_numbers,
        coordinates,
        nddo=True,
        conv_tol=1e-12,
    )
    finite_difference = msindo_gradient_fd(
        atomic_numbers,
        coordinates,
        nddo=True,
        step=1e-4 / ANGSTROM_TO_BOHR,
        conv_tol=1e-12,
    )

    assert np.max(np.abs(analytic - finite_difference)) < 1e-8
    assert np.max(np.abs(np.sum(analytic, axis=0))) < 1e-12


def test_msindo_nddo_d_shell_analytic_gradient_matches_source_and_energy_fd():
    """Al-Cl SPDD energy derivatives match the source and shipped energy."""
    import numpy as np

    from vibeqc.semiempirical.methods.msindo import (
        ANGSTROM_TO_BOHR,
        msindo_gradient_fd,
    )
    from vibeqc.semiempirical.methods.msindo_gradient_analytic import (
        msindo_gradient_analytic,
    )

    atomic_numbers = [13, 17]
    coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.10]]
    analytic = msindo_gradient_analytic(
        atomic_numbers,
        coordinates,
        nddo=True,
        conv_tol=1e-12,
    )
    source = np.array(
        [[0.0, 0.0, 0.0317797727], [0.0, 0.0, -0.0317797727]]
    )
    assert np.max(np.abs(analytic - source)) < 5e-8
    assert np.max(np.abs(np.sum(analytic, axis=0))) < 1e-12
    for step_bohr in (2e-4, 1e-4, 5e-5):
        finite_difference = msindo_gradient_fd(
            atomic_numbers,
            coordinates,
            nddo=True,
            step=step_bohr / ANGSTROM_TO_BOHR,
            conv_tol=1e-12,
        )
        residual = np.max(np.abs(analytic - finite_difference))
        assert residual < 1e-6, (
            f"h={step_bohr} bohr: {residual:.3e} Ha/bohr"
        )


def test_msindo_nddo_d_shell_charged_gradient_matches_energy_fd():
    """The Al-Cl SPDD derivative follows the charged closed-shell energy."""
    import numpy as np

    from vibeqc.semiempirical.methods.msindo import (
        ANGSTROM_TO_BOHR,
        msindo_gradient_fd,
    )
    from vibeqc.semiempirical.methods.msindo_gradient_analytic import (
        msindo_gradient_analytic,
    )

    atomic_numbers = [13, 17]
    coordinates = [[0.0, 0.0, 0.0], [0.31, -0.17, 2.13]]
    analytic = msindo_gradient_analytic(
        atomic_numbers,
        coordinates,
        charge=2,
        nddo=True,
        conv_tol=1e-12,
    )
    finite_difference = msindo_gradient_fd(
        atomic_numbers,
        coordinates,
        charge=2,
        nddo=True,
        step=1e-4 / ANGSTROM_TO_BOHR,
        conv_tol=1e-12,
    )

    assert np.max(np.abs(analytic - finite_difference)) < 1e-6
    assert np.max(np.abs(np.sum(analytic, axis=0))) < 1e-12


def test_msindo_nddo_analytic_gradient_rejects_open_shell_multiplicity():
    """An even-electron triplet must not reuse the closed-shell NDDO gradient."""
    from vibeqc.semiempirical.methods.msindo_gradient_analytic import (
        msindo_gradient_analytic,
    )

    with pytest.raises(NotImplementedError, match="closed-shell"):
        msindo_gradient_analytic(
            [8, 8],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 1.208]],
            multiplicity=3,
            nddo=True,
        )


def test_msindo_nddo_analytic_gradient_rejects_excessive_charge():
    """The direct gradient mirrors the energy API's electron-count guard."""
    from vibeqc.semiempirical.methods.msindo_gradient_analytic import (
        msindo_gradient_analytic,
    )

    with pytest.raises(ValueError, match="exceeds the 2 valence electrons"):
        msindo_gradient_analytic(
            [1, 1],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]],
            charge=4,
            nddo=True,
        )
    with pytest.raises(NotImplementedError, match="valence-electron count is odd"):
        msindo_gradient_analytic(
            [1, 9],
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.917]],
            charge=1,
            nddo=True,
        )


def test_msindo_nddo_analytic_gradient_rotation_and_permutation_covariance():
    """Al-Cl SPDD derivatives are rigid-motion and atom-order covariant."""
    import numpy as np
    from vibeqc.semiempirical.methods.msindo_gradient_analytic import (
        msindo_gradient_analytic,
    )

    atomic_numbers = [13, 17]
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [0.31, -0.17, 2.13]]
    )
    reference = msindo_gradient_analytic(
        atomic_numbers, coordinates, nddo=True, conv_tol=1e-12
    )

    angle = 0.713
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotated = msindo_gradient_analytic(
        atomic_numbers,
        coordinates @ rotation.T,
        nddo=True,
        conv_tol=1e-12,
    )
    assert rotated == pytest.approx(reference @ rotation.T, abs=1e-9)

    translated = msindo_gradient_analytic(
        atomic_numbers,
        coordinates + np.array([1.7, -0.4, 2.3]),
        nddo=True,
        conv_tol=1e-12,
    )
    assert translated == pytest.approx(reference, abs=1e-9)

    permutation = [1, 0]
    permuted = msindo_gradient_analytic(
        [atomic_numbers[i] for i in permutation],
        coordinates[permutation],
        nddo=True,
        conv_tol=1e-12,
    )
    assert permuted == pytest.approx(reference[permutation], abs=1e-9)


def test_msindo_pair_blocks_derivatives_global_frame_match_fd():
    """Phase 3: global-frame (R,θ,φ) derivatives of per-pair blocks match FD.

    Validates the s-s element of HK1_DR/DT/DP for O-H at non-axial orientations."""
    import math

    import numpy as np
    from vibeqc.semiempirical.methods.msindo import _pair_blocks
    from vibeqc.semiempirical.methods.msindo_pair_deriv import _pair_blocks_deriv

    zk, zl = 8, 1
    R = 0.95
    rk = np.array([0.0, 0.0, 0.0])
    h = 1e-4
    ph = 1e-3

    for theta in [45.0 * math.pi / 180.0, 60.0 * math.pi / 180.0]:
        for phi in [0.0, 45.0 * math.pi / 180.0]:
            E_test = np.array(
                [
                    math.sin(theta) * math.cos(phi),
                    math.sin(theta) * math.sin(phi),
                    math.cos(theta),
                ]
            )
            assert abs(np.linalg.norm(E_test) - 1.0) < 1e-15
            rl = rk + R * E_test
            pd = _pair_blocks_deriv(zk, zl, rk, rl)

            # Validate R-derivative
            pb_p = _pair_blocks(zk, zl, rk, rk + (R + h) * E_test)
            pb_m = _pair_blocks(zk, zl, rk, rk + (R - h) * E_test)
            fd_dr = (pb_p.HK1[0, 0] - pb_m.HK1[0, 0]) / (2 * h)
            assert abs(fd_dr - pd.HK1_DR[0, 0]) < 1e-8, (
                f"θ={theta:.2f} φ={phi:.2f}: DR FD={fd_dr:.10f} AN={pd.HK1_DR[0, 0]:.10f}"
            )

            # Validate θ-derivative: rotate bond direction around y-axis
            Et_p = np.array(
                [
                    math.sin(theta + h) * math.cos(phi),
                    math.sin(theta + h) * math.sin(phi),
                    math.cos(theta + h),
                ]
            )
            Et_m = np.array(
                [
                    math.sin(theta - h) * math.cos(phi),
                    math.sin(theta - h) * math.sin(phi),
                    math.cos(theta - h),
                ]
            )
            pb_tp = _pair_blocks(zk, zl, rk, rk + R * Et_p)
            pb_tm = _pair_blocks(zk, zl, rk, rk + R * Et_m)
            fd_dt = (pb_tp.HK1[0, 0] - pb_tm.HK1[0, 0]) / (2 * h)
            assert abs(fd_dt - pd.HK1_DT[0, 0]) < 1e-7, (
                f"θ={theta:.2f}: DT FD={fd_dt:.10f} AN={pd.HK1_DT[0, 0]:.10f}"
            )

            # Validate φ-derivative
            Ep_p = np.array(
                [
                    math.sin(theta) * math.cos(phi + ph),
                    math.sin(theta) * math.sin(phi + ph),
                    math.cos(theta),
                ]
            )
            Ep_m = np.array(
                [
                    math.sin(theta) * math.cos(phi - ph),
                    math.sin(theta) * math.sin(phi - ph),
                    math.cos(theta),
                ]
            )
            pb_pp = _pair_blocks(zk, zl, rk, rk + R * Ep_p)
            pb_pm = _pair_blocks(zk, zl, rk, rk + R * Ep_m)
            fd_dp = (pb_pp.HK1[0, 0] - pb_pm.HK1[0, 0]) / (2 * ph)
            assert abs(fd_dp - pd.HK1_DP[0, 0]) < 1e-7, (
                f"φ={phi:.2f}: DP FD={fd_dp:.10f} AN={pd.HK1_DP[0, 0]:.10f}"
            )


def test_msindo_geometry_optimization_lowers_energy():
    """msindo_optimize relaxes a distorted H₂O to a lower-energy stationary point
    (final energy below the start; relaxed gradient near zero)."""
    import numpy as np

    Z = [8, 1, 1]
    C = [[0, 0, 0], [0.95, 0.0, 0.10], [-0.20, 0.92, 0.0]]
    e0 = msindo.run_msindo(Z, C).total_energy
    Crel, final = msindo.msindo_optimize(Z, C, fmax=1e-3)
    assert final.converged
    assert final.total_energy < e0 - 1e-5
    g = msindo.msindo_gradient_fd(Z, Crel)
    assert np.max(np.abs(g)) < 5e-3


def test_run_job_msindo_nddo_routes_to_nddo_engine():
    """run_job(method='msindo', nddo=True) routes to the NDDO engine (distinct
    from INDO) and exposes the source-validated NDDO energy and gradient."""
    import os
    import tempfile

    import numpy as np
    from vibeqc._vibeqc_core import Atom, Molecule
    from vibeqc.runner import run_job
    from vibeqc.semiempirical.methods.msindo import ANGSTROM_TO_BOHR as A2B

    hf = Molecule([Atom(1, [0, 0, 0]), Atom(9, [0, 0, 0.917 * A2B])], 0, 1)
    with tempfile.TemporaryDirectory() as d:
        r_indo = run_job(hf, method="msindo", output=os.path.join(d, "i"), verbose=0)
        r_nddo = run_job(
            hf, method="msindo", nddo=True, output=os.path.join(d, "n"), verbose=0
        )
        gradient = r_nddo.gradient()
    assert bool(r_nddo.converged)
    assert float(r_nddo.energy) == pytest.approx(-23.0436304974, abs=1e-6)
    assert gradient == pytest.approx(
        np.array([[0.0, 0.0, 0.0047202298], [0.0, 0.0, -0.0047202298]]),
        abs=5e-8,
    )
    assert r_nddo.gradient() is gradient
    # NDDO differs from the default INDO energy.
    assert abs(float(r_nddo.energy) - float(r_indo.energy)) > 1.0


@pytest.mark.parametrize(
    "name,Z,coords,ref",
    [
        # 4th-row p-block Ga–Br (Z 31–35): 4s4p4d valence over an [Ar]3d¹⁰ frozen
        # core.  Reference MSINDO totals (CARTES RHF) at the fixed geometries below.
        ("GaF", [31, 9], [[0, 0, 0], [0, 0, 1.77]], -25.8426638586),
        (
            "GeH4",
            [32, 1, 1, 1, 1],
            [
                [0, 0, 0],
                [0.88, 0.88, 0.88],
                [-0.88, -0.88, 0.88],
                [0.88, -0.88, -0.88],
                [-0.88, 0.88, -0.88],
            ],
            -6.2879933199,
        ),
        (
            "AsH3",
            [33, 1, 1, 1],
            [[0, 0, 0.30], [1.0, 0, 0], [-0.5, 0.866, 0], [-0.5, -0.866, 0]],
            -6.7714734084,
        ),
        ("H2Se", [34, 1, 1], [[0, 0, 0], [1.40, 0, 0], [0, 1.40, 0]], -10.7196002756),
        ("HBr", [1, 35], [[0, 0, 0], [0, 0, 1.41]], -13.0039510422),
        ("Br2", [35, 35], [[0, 0, 0], [0, 0, 2.28]], -24.7959682859),
    ],
)
def test_ga_br_total_energy_matches_oracle(name, Z, coords, ref):
    """Ga–Br (4s4p4d valence + [Ar]3d¹⁰ frozen core, with the 3d core shell in
    V2CORE and the Ga–Kr d-ENEG) reproduce reference MSINDO to ≤1 µHa."""
    r = msindo.run_msindo(Z, coords)
    assert r.converged
    assert r.total_energy == pytest.approx(ref, abs=1e-6)


# --------------------------------------------------------------------------- #
# MSINDO MP2 (INDO reference) via the general correlation kernel.              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,Z,coords,e_corr_ref",
    [
        (
            "H2O",
            [8, 1, 1],
            [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]],
            -0.0104151464,
        ),
        (
            "CH4",
            [6, 1, 1, 1, 1],
            [
                [0, 0, 0],
                [0.629, 0.629, 0.629],
                [-0.629, -0.629, 0.629],
                [0.629, -0.629, -0.629],
                [-0.629, 0.629, -0.629],
            ],
            -0.0317873051,
        ),
        (
            "NH3",
            [7, 1, 1, 1],
            [
                [0, 0, 0.12],
                [0.94, 0, -0.27],
                [-0.47, 0.81, -0.27],
                [-0.47, -0.81, -0.27],
            ],
            -0.0207023548,
        ),
        ("HF", [1, 9], [[0, 0, 0], [0, 0, 0.917]], -0.0070712440),
        ("CO", [6, 8], [[0, 0, 0], [0, 0, 1.128]], -0.0557616046),
    ],
)
def test_msindo_mp2_matches_oracle(name, Z, coords, e_corr_ref):
    """INDO MP2 (the general correlation kernel + the INDO ERIProvider)
    reproduces reference MSINDO MP2 correlation energies to sub-µHa
    (SCF-convergence-limited).  Spans first/second-row s/p molecules."""
    r = msindo.msindo_mp2(Z, coords)
    assert r.converged
    assert r.e_corr == pytest.approx(e_corr_ref, abs=1e-6)
    assert r.e_total == pytest.approx(r.e_scf + r.e_corr)


def test_msindo_scs_mp2_scales_components():
    """SCS-MP2 on the MSINDO reference rescales the spin components (6/5, 1/3)."""
    Z = [8, 1, 1]
    coords = [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]]
    plain = msindo.msindo_mp2(Z, coords, variant="mp2")
    scs = msindo.msindo_mp2(Z, coords, variant="scs-mp2")
    assert scs.e_os == pytest.approx(plain.e_os)
    assert scs.e_corr == pytest.approx(1.2 * plain.e_os + plain.e_ss / 3.0)
    assert scs.e_corr != pytest.approx(plain.e_corr)


# --------------------------------------------------------------------------- #
# MSINDO NDDO-MP2 — MP2 on the NDDO reference (INDO integral set + NDDO MOs).   #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,Z,coords,e_corr_ref",
    [
        # Reference MSINDO NDDO MP2 correlation energies (oracle, CARTES RHF NDDO MP2)
        # at the NDDO geometries of test_nddo_total_energy_matches_oracle.
        ("HF", [1, 9], [[0, 0, 0], [0, 0, 0.917]], -0.0048223855),
        (
            "H2O",
            [8, 1, 1],
            [[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]],
            -0.0089744362,
        ),
        (
            "CH4",
            [6, 1, 1, 1, 1],
            [
                [0, 0, 0],
                [0.6276, 0.6276, 0.6276],
                [-0.6276, -0.6276, 0.6276],
                [0.6276, -0.6276, -0.6276],
                [-0.6276, 0.6276, -0.6276],
            ],
            -0.0285982267,
        ),
        ("N2", [7, 7], [[0, 0, 0], [0, 0, 1.098]], -0.0864054042),
        ("CO", [6, 8], [[0, 0, 0], [0, 0, 1.128]], -0.0500258701),
    ],
)
def test_msindo_nddo_mp2_matches_oracle(name, Z, coords, e_corr_ref):
    """NDDO-MP2 (MP2 on the NDDO reference) reproduces reference MSINDO NDDO MP2
    correlation energies to sub-µHa.  The NDDO multipoles enter the SCF Fock (and
    hence the MOs) but NOT the MP2 integral transform — MSINDO's ``mp2rhf.f`` reads
    only the INDO ``GMUNU`` set even under NDDO.  Spans single-p (HF/H₂O/CH₄) and
    p–p (N₂/CO)."""
    r = msindo.msindo_mp2(Z, coords, nddo=True)
    assert r.converged
    assert r.e_corr == pytest.approx(e_corr_ref, abs=1e-6)
    assert r.e_total == pytest.approx(r.e_scf + r.e_corr)


def test_nddo_ao_eri_reproduces_nddo_fock():
    """The rigorous NDDO multipole AO tensor (``_nddo_ao_eri``), contracted as
    ``J − ½K``, reproduces the NDDO two-electron Fock (``_build_fock`` +
    ``_nddo_fock_extra``) to machine precision — the verification that the
    dipole-monopole + dipole-dipole integrals are transcribed correctly (the
    INDO-only tensor is off by ~0.4 Ha here).  Checked on N₂, where the two-centre
    multipoles are largest."""
    import numpy as np

    Z = [7, 7]
    coords = [[0, 0, 0], [0, 0, 1.098]]
    with msindo._nddo_params():
        C = np.asarray(coords, float) * msindo.ANGSTROM_TO_BOHR
        blocks, nsto = msindo._atom_blocks(Z)
        nocc = sum(msindo.eff_core_charge(z) for z in Z) // 2
        H, G = msindo._build_core_and_gamma(Z, C, blocks, nsto, nddo=True)
        fock_extra = msindo._nddo_fock_extra(Z, C, blocks)
        P, _F, _e, _eps, conv, _it = msindo._scf_rhf(
            H, G, blocks, Z, nocc, fock_extra=fock_extra, conv_tol=1e-10
        )
        f2_ref = (msindo._build_fock(H, G, P, blocks, Z) - H) + fock_extra(P)[0]
        g = msindo._nddo_ao_eri(Z, C, blocks, G)
        f2_tensor = np.einsum("mnls,ls->mn", g, P, optimize=True) - 0.5 * np.einsum(
            "mlns,ls->mn", g, P, optimize=True
        )
        gi = msindo._indo_ao_eri(G, blocks)
        f2_indo = np.einsum("mnls,ls->mn", gi, P, optimize=True) - 0.5 * np.einsum(
            "mlns,ls->mn", gi, P, optimize=True
        )
    assert conv
    assert np.max(np.abs(f2_tensor - f2_ref)) < 1e-11
    assert np.max(np.abs(f2_indo - f2_ref)) > 0.1  # multipoles are essential


def test_nddo_mp2_uses_indo_not_multipole_integrals():
    """Documents the integral-set finding: NDDO-MP2 over the INDO tensor matches
    the oracle, while MP2 over the rigorous NDDO multipole tensor (``_nddo_ao_eri``)
    gives a *different* number — confirming MSINDO's MP2 transform omits the
    two-centre multipoles even in NDDO mode (parity-first: the oracle is ground
    truth)."""
    import numpy as np
    from vibeqc.correlation import mp2_energy

    Z = [6, 8]  # CO — large multipole effect
    coords = [[0, 0, 0], [0, 0, 1.128]]
    oracle = -0.0500258701
    shipped = msindo.msindo_mp2(Z, coords, nddo=True).e_corr  # INDO tensor
    with msindo._nddo_params():
        C = np.asarray(coords, float) * msindo.ANGSTROM_TO_BOHR
        blocks, nsto = msindo._atom_blocks(Z)
        nocc = sum(msindo.eff_core_charge(z) for z in Z) // 2
        H, G = msindo._build_core_and_gamma(Z, C, blocks, nsto, nddo=True)
        fe = msindo._nddo_fock_extra(Z, C, blocks)
        P, _F, _e, _eps, _c, _it = msindo._scf_rhf(
            H, G, blocks, Z, nocc, fock_extra=fe, conv_tol=1e-10
        )
        eps, cmo = np.linalg.eigh(msindo._build_fock(H, G, P, blocks, Z) + fe(P)[0])
        g = msindo._nddo_ao_eri(Z, C, blocks, G)
        co, cv = cmo[:, :nocc], cmo[:, nocc:]
        ovov = np.einsum("mnls,mi,na,lj,sb->iajb", g, co, cv, co, cv, optimize=True)
        cand_b = mp2_energy(eps[:nocc], eps[nocc:], ovov).e_corr
    assert shipped == pytest.approx(oracle, abs=1e-6)  # INDO tensor = oracle
    assert abs(cand_b - oracle) > 1e-3  # multipole tensor ≠ oracle


# --------------------------------------------------------------------------- #
# MSINDO open-shell UMP2 (UHF reference) via vibeqc.correlation.ump2_energy.    #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name,Z,coords,mult,oracle_buggy",
    [
        # Reference MSINDO UMP2 correlation energies (oracle, CARTES UHF MULTIP n MP2).
        # These oracle numbers carry MSINDO's mp2uhf.f αα-channel ¼ bug (see
        # msindo_ump2): oracle_total = 0.25·E_αα + E_ββ + E_αβ.
        ("OH", [8, 1], [[0, 0, 0], [0, 0, 0.97]], 2, -0.0058675061),
        ("O2", [8, 8], [[0, 0, 0], [0, 0, 1.21]], 3, -0.0498883152),
        (
            "NH2",
            [7, 1, 1],
            [[0, 0, 0], [0, 0.80, 0.58], [0, -0.80, 0.58]],
            2,
            -0.0127228774,
        ),
        (
            "CH3",
            [6, 1, 1, 1],
            [[0, 0, 0], [0, 1.08, 0], [0.935, -0.54, 0], [-0.935, -0.54, 0]],
            2,
            -0.0242667793,
        ),
        ("NO", [7, 8], [[0, 0, 0], [0, 0, 1.15]], 2, -0.0550171551),
    ],
)
def test_msindo_ump2_matches_oracle_modulo_aa_bug(name, Z, coords, mult, oracle_buggy):
    """Open-shell INDO UMP2.  vibe-qc ships the *correct* UMP2; the oracle carries
    MSINDO's ``mp2uhf.f`` αα ¼ bug.  So the parity check folds the bug back in: the
    oracle total equals ``0.25·E_αα + E_ββ + E_αβ`` to sub-µHa — which validates
    vibe-qc's ββ and αβ channels (and the αα channel structure) against the
    oracle.  The reported ``e_corr`` is the bug-free ``E_αα+E_ββ+E_αβ``.  OH/O₂
    have no αα channel (one α-virtual), so correct == oracle there."""
    r = msindo.msindo_ump2(Z, coords, multiplicity=mult)
    assert r.converged
    bug_compatible = 0.25 * r.e_aa + r.e_bb + r.e_ab
    assert bug_compatible == pytest.approx(oracle_buggy, abs=1e-5)
    assert r.e_corr == pytest.approx(r.e_aa + r.e_bb + r.e_ab)  # correct UMP2
    assert r.e_total == pytest.approx(r.e_scf + r.e_corr)


def test_msindo_ump2_aa_bug_only_when_aa_channel_exists():
    """The αα-bug correction ¾·E_αα is nonzero only when an αα channel exists
    (≥2 α-occupied + ≥2 α-virtual).  OH (one α-virtual) → E_αα=0 → vibe-qc ==
    oracle; NO has a real αα channel → vibe-qc's correct total lies below the
    oracle by ¾·E_αα."""
    oh = msindo.msindo_ump2([8, 1], [[0, 0, 0], [0, 0, 0.97]], multiplicity=2)
    assert oh.e_aa == pytest.approx(0.0, abs=1e-12)  # no αα channel
    assert oh.e_corr == pytest.approx(-0.0058675061, abs=1e-5)  # == oracle
    no = msindo.msindo_ump2([7, 8], [[0, 0, 0], [0, 0, 1.15]], multiplicity=2)
    assert no.e_aa < -1e-4  # real αα channel
    assert no.e_corr < (0.25 * no.e_aa + no.e_bb + no.e_ab)  # correct < oracle


def test_msindo_ump2_reduces_to_rhf_mp2():
    """On a closed-shell molecule run as a UHF singlet, UMP2 reproduces RHF-MP2
    (the kernel's α=β reduction) — H₂O to near machine precision."""
    Z = [8, 1, 1]
    xyz = [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]]
    rhf = msindo.msindo_mp2(Z, xyz)
    uhf = msindo.msindo_ump2(Z, xyz, multiplicity=1)
    assert uhf.e_corr == pytest.approx(rhf.e_corr, abs=1e-8)
    assert uhf.e_aa == pytest.approx(uhf.e_bb)  # symmetric spins


# --------------------------------------------------------------------------- #
# MSINDO Green's-function quasiparticle IPs (GF2 / OVGF) via vibeqc.propagator #
# --------------------------------------------------------------------------- #


def test_msindo_ovgf_rigorous_quasiparticle():
    """INDO quasiparticle energies via the general diagonal second-order
    self-energy (vibeqc.propagator) on H₂O.

    These pin vibe-qc's *rigorous* ZDO self-energy (the same INDO integral set
    that drives MSINDO-MP2) — an internal regression anchor, NOT a reference-
    MSINDO parity number.  Reference MSINDO's ``ovgfrhf_neu`` uses a *factorized*
    integral approximation (it pairs the (p,i) and (a,j) AO densities on the
    same centres for both Coulomb and exchange), giving a HOMO IP of 12.586 eV;
    that approximation is cruder than, and inconsistent with, MSINDO's own MP2
    integrals, so vibe-qc ships the rigorous value instead (see
    vibeqc/propagator.py and docs/user_guide/msindo.md)."""
    from vibeqc.semiempirical.methods.msindo import msindo_ovgf

    EV = 27.211386245988
    Z = [8, 1, 1]
    xyz = [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]]

    # Non-iterative Σ(ε_p): rigorous HOMO correction is +0.581 eV.
    r0 = msindo_ovgf(Z, xyz, iterate=False)
    assert r0.orbitals == (3, 4)  # HOMO, LUMO
    assert r0.homo == 3
    sigma_homo_ev = (r0.eps_qp[0] - r0.eps_scf[0]) * EV
    assert sigma_homo_ev == pytest.approx(0.581, abs=2e-3)
    assert r0.eps_scf[0] * EV == pytest.approx(-13.698, abs=1e-2)  # SCF HOMO

    # Iterative Dyson solution + physical pole strengths.
    r = msindo_ovgf(Z, xyz, iterate=True)
    assert r.converged
    assert r.ip_homo_ev == pytest.approx(13.129, abs=5e-3)
    assert all(0.90 < z <= 1.0 + 1e-12 for z in r.pole_strength)  # strong QP


def test_msindo_ovgf_electron_affinity():
    """EA = -eps_qp(LUMO) is surfaced and sane: water has no bound anion, so the
    GF2 EA is negative (the LUMO quasiparticle sits above vacuum)."""
    from vibeqc.semiempirical.methods.msindo import msindo_ovgf

    EV = 27.211386245988
    Z = [8, 1, 1]
    xyz = [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]]
    r = msindo_ovgf(Z, xyz, iterate=True)
    lumo = r.homo + 1
    assert lumo in r.orbitals
    # EA matches -eps_qp(LUMO) exactly.
    i = list(r.orbitals).index(lumo)
    assert r.ea_lumo_ev == pytest.approx(-r.eps_qp[i] * EV)
    # Water binds no extra electron: EA < 0 (qualitative, minimal basis).
    assert r.ea_lumo_ev < 0.0
    assert r.ea_lumo_ev == pytest.approx(-7.08, abs=0.1)


def test_msindo_ovgf_ea_requires_lumo_corrected():
    """ea_lumo_ev raises a clear error if the LUMO was not among the corrected
    orbitals (e.g. only occupied orbitals were requested)."""
    from vibeqc.semiempirical.methods.msindo import msindo_ovgf

    Z = [8, 1, 1]
    xyz = [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]]
    r = msindo_ovgf(Z, xyz, orbitals=(3,))  # HOMO only, no LUMO
    with pytest.raises(ValueError, match="LUMO"):
        _ = r.ea_lumo_ev


def test_msindo_ovgf_open_shell_radical():
    """Open-shell (UHF) INDO quasiparticles for a doublet radical (CH₃): the
    spin-resolved second-order self-energy returns a MsindoUOVGFResult over the
    α/β frontier orbitals; the α-SOMO IP corrects Koopmans downward (a strong
    quasiparticle) and the EA is surfaced.  s/p elements only."""
    import numpy as np
    from vibeqc.semiempirical.methods.msindo import MsindoUOVGFResult, msindo_ovgf

    d = 1.079
    ch3 = [[0, 0, 0]] + [
        [d * np.cos(a), d * np.sin(a), 0] for a in (0.0, 2 * np.pi / 3, 4 * np.pi / 3)
    ]
    r = msindo_ovgf([6, 1, 1, 1], ch3)  # 7 valence e⁻ → doublet
    assert isinstance(r, MsindoUOVGFResult)
    assert r.converged
    assert r.n_alpha == 4 and r.n_beta == 3
    somo = next(
        i
        for i, (sp, ix) in enumerate(r.orbitals)
        if sp == "alpha" and ix == r.n_alpha - 1
    )
    assert -r.eps_qp[somo] < -r.eps_scf[somo]  # GF2 lowers the SOMO IP
    assert 0.85 < r.pole_strength[somo] <= 1.0 + 1e-12
    assert 8.0 < r.first_ip_ev < 13.0  # most weakly bound electron
    assert r.ea_ev < 0.0  # CH₃ binds no extra electron


def test_msindo_ovgf_open_shell_explicit_multiplicity():
    """An explicit multiplicity routes through the UHF propagator even for an
    even valence count (O₂ triplet); inconsistent spin/electron parity raises."""
    from vibeqc.semiempirical.methods.msindo import MsindoUOVGFResult, msindo_ovgf

    r = msindo_ovgf([8, 8], [[0, 0, 0], [0, 0, 1.21]], multiplicity=3)
    assert isinstance(r, MsindoUOVGFResult)
    assert r.n_alpha - r.n_beta == 2  # two unpaired electrons
    # multiplicity=2 with an even valence count is unphysical → clear error.
    with pytest.raises(ValueError, match="inconsistent"):
        msindo_ovgf(
            [8, 1, 1],
            [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]],
            multiplicity=2,
        )


def test_msindo_ovgf_closed_shell_still_restricted():
    """A closed-shell singlet keeps the restricted (RHF) path + result type even
    though the open-shell branch now exists."""
    from vibeqc.semiempirical.methods.msindo import MsindoOVGFResult, msindo_ovgf

    r = msindo_ovgf([8, 1, 1], [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]])
    assert isinstance(r, MsindoOVGFResult)
    assert r.ip_homo_ev == pytest.approx(13.129, abs=5e-3)  # unchanged


def test_msindo_ovgf_renormalized():
    """renormalize=True swaps the bare GF2 self-energy for the renormalized one
    (full third-order + geometric screening), moving the H₂O HOMO IP relative to
    bare GF2 (13.13 eV) toward experiment (12.6) with a still-strong pole."""
    from vibeqc.semiempirical.methods.msindo import MsindoOVGFResult, msindo_ovgf

    xyz = [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]]
    bare = msindo_ovgf([8, 1, 1], xyz)
    ren = msindo_ovgf([8, 1, 1], xyz, renormalize=True)
    assert isinstance(ren, MsindoOVGFResult) and ren.converged
    # the renormalization shifts the IP off the bare value, toward experiment.
    assert ren.ip_homo_ev != pytest.approx(bare.ip_homo_ev, abs=1e-2)
    assert abs(ren.ip_homo_ev - 12.6) < abs(bare.ip_homo_ev - 12.6)
    assert 0.85 < ren.pole_strength[0] <= 1.0 + 1e-12


# --------------------------------------------------------------------------- #
# MSINDO CIS / TDA excited states via vibeqc.excited                           #
# --------------------------------------------------------------------------- #


def test_msindo_cis_singlet_triplet():
    """INDO CIS via the general kernel (vibeqc.excited) on H₂O: the lowest
    singlet is HOMO→LUMO and lies above the lowest triplet (Hund).

    Pins the *rigorous* INDO CIS as a regression anchor — reference MSINDO
    applies empirical CIS scaling (SCALEDCIS/UJCORR), so this is NOT a
    reference-MSINDO parity number (see msindo_cis docstring)."""
    from vibeqc.semiempirical.methods.msindo import msindo_cis

    Z = [8, 1, 1]
    xyz = [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]]
    s = msindo_cis(Z, xyz, spin="singlet", n_states=4)
    t = msindo_cis(Z, xyz, spin="triplet", n_states=4)
    assert t.excitation_energies[0] < s.excitation_energies[0]  # Hund
    # S1 is the HOMO (occ index nocc-1=3) → LUMO (vir index 0) transition.
    occ, vir, wt = s.dominant_transition(0)
    assert (occ, vir) == (3, 0)
    assert wt > 0.9
    # Regression anchor for the rigorous INDO CIS S1 (eV).
    assert s.excitation_energies_ev[0] == pytest.approx(6.806, abs=1e-2)


def test_msindo_cis_odd_electron_raises():
    """Open-shell CIS is not wired — odd valence count fails fast."""
    from vibeqc.semiempirical.methods.msindo import msindo_cis

    with pytest.raises(NotImplementedError, match="closed-shell"):
        msindo_cis([7], [[0, 0, 0]])  # N atom


# --------------------------------------------------------------------------- #
# MSINDO CISD via vibe-qc's GENERIC CI solver (vibeqc.solvers.cisd).           #
# The reference MSINDO CISD (rhfcisd.f) is a selecting driver with empirical   #
# scalings — deliberately NOT reproduced; these tests validate the CI          #
# mechanics against vibe-qc's own FCI / CIS instead (HANDOVER §5).             #
# --------------------------------------------------------------------------- #

_H2O = ([8, 1, 1], [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]])


def test_msindo_cisd_reference_equals_scf():
    """Integral-consistency keystone: the CI reference-determinant energy built
    from the MSINDO MO integrals equals the MSINDO SCF energy to machine
    precision (the INDO 2e tensor reproduces _build_fock's J−½K exactly)."""
    from vibeqc.semiempirical.methods.msindo import msindo_cisd

    r = msindo_cisd(*_H2O)
    assert r.converged
    assert r.ci.e_ref == pytest.approx(r.e_scf, abs=1e-9)
    # e_corr is the lowering relative to that reference.
    assert r.e_total == pytest.approx(r.e_scf + r.e_corr, abs=1e-12)


def test_msindo_cisd_equals_fci_for_two_electrons():
    """For a 2-electron system CISD spans the whole FCI space, so MSINDO CISD
    reproduces the INDO FCI (casci full space) on the same MO integrals."""
    from vibeqc.semiempirical.methods.msindo import _msindo_mo_hamiltonian, msindo_cisd
    from vibeqc.solvers import casci

    Z, xyz = [1, 1], [[0, 0, 0], [0, 0, 0.74]]
    r = msindo_cisd(Z, xyz)
    ham, _e, _no, nelec, _escf, _c = _msindo_mo_hamiltonian(Z, xyz)
    fci = casci(
        ham.h1e,
        ham.h2e,
        n_active_elec=nelec,
        n_active_orb=ham.norb,
        nuclear_repulsion=ham.nuclear_repulsion,
    )
    assert r.e_total == pytest.approx(fci.e_total, abs=1e-10)
    assert r.n_det == fci.n_det  # full 2e2o space


def test_msindo_cisd_brackets_fci():
    """E_FCI ≤ E_CISD ≤ E_SCF on H₂O, and CISD recovers most of the INDO
    correlation while staying reference-dominated."""
    from vibeqc.semiempirical.methods.msindo import _msindo_mo_hamiltonian, msindo_cisd
    from vibeqc.solvers import casci

    r = msindo_cisd(*_H2O)
    ham, _e, _no, nelec, e_scf, _c = _msindo_mo_hamiltonian(*_H2O)
    fci = casci(
        ham.h1e,
        ham.h2e,
        n_active_elec=nelec,
        n_active_orb=ham.norb,
        nuclear_repulsion=ham.nuclear_repulsion,
    )
    assert fci.e_total <= r.e_total + 1e-10  # FCI lower bound
    assert r.e_total <= e_scf + 1e-10  # CISD ≤ HF
    assert r.e_corr < -1e-3  # real correlation
    assert 0.95 < r.reference_weight <= 1.0


def test_msindo_cisd_superset_of_cis():
    """CISD ⊇ CIS: the singles-only subspace (max_excitation=1) of the
    determinant CI reproduces the msindo_cis excitation structure exactly — its
    spectrum is the union of the spin-adapted singlet and triplet CIS energies
    (Brillouin decouples the reference; M_s=0 determinants span both spins)."""
    import numpy as np
    from vibeqc.semiempirical.methods.msindo import _msindo_mo_hamiltonian, msindo_cis
    from vibeqc.solvers import cisd

    Z, xyz = _H2O
    ham, _e, nocc, nelec, _escf, _c = _msindo_mo_hamiltonian(Z, xyz)
    ndet_cis = 1 + 2 * nocc * (ham.norb - nocc)
    det_cis = cisd(
        ham.h1e,
        ham.h2e,
        nelec,
        ham.norb,
        nuclear_repulsion=ham.nuclear_repulsion,
        max_excitation=1,
        nroots=ndet_cis,
    )
    ev = np.asarray(det_cis.eigenvalues)
    omega = (ev - ev[0]) * 27.211386245988  # excitation energies (eV)
    s = msindo_cis(Z, xyz, spin="singlet", n_states=4).excitation_energies_ev
    t = msindo_cis(Z, xyz, spin="triplet", n_states=4).excitation_energies_ev
    for arr in (s, t):
        for x in arr:
            assert np.min(np.abs(omega - x)) < 1e-6  # appears in det-CIS


def test_msindo_cisd_dominant_configurations():
    """The H₂O CISD ground state is reference-dominated; the leading double is a
    closed-shell αβ pair excitation into the same spatial virtual."""
    from vibeqc.semiempirical.methods.msindo import msindo_cisd

    r = msindo_cisd(*_H2O)
    configs = r.dominant_configurations(4)
    assert configs[0][0] == "reference"
    assert configs[0][2] > 0.95  # reference weight
    # The next configs are closed-shell doubles (one α + one β term each).
    desc1 = configs[1][0]
    assert "a->" in desc1 and "b->" in desc1


def test_msindo_cisd_citations(tmp_path):
    """msindo_cisd(output=) surfaces MSINDO + the variational-CISD method
    (Pople–Seeger–Krishnan 1977) in the .bibtex / .references siblings."""
    from vibeqc.semiempirical.methods.msindo import msindo_cisd

    stem = tmp_path / "h2o_cisd"
    msindo_cisd(*_H2O, output=str(stem))
    bib = (tmp_path / "h2o_cisd.bibtex").read_text()
    assert "pople_cisd_1977" in bib
    assert "ahlswede_jug_msindo_1_1999" in bib
    assert "valeev_libint" not in bib  # INDO: no Gaussian ints


def test_msindo_ovgf_citations(tmp_path):
    """msindo_ovgf(output=) surfaces MSINDO + the NDDO Green's-function IP paper
    (Danovich 1997) over the OVGF formalism (Cederbaum 1975, von Niessen 1984).

    The routes.methods.ovgf row cannot cover this: it keys off method="ovgf",
    the *ab-initio* run_job path, whereas the semiempirical route reports
    method="msindo". Without the extra_entries emit here, the NDDO OVGF paper is
    unreachable from any output file."""
    from vibeqc.semiempirical.methods.msindo import msindo_ovgf

    stem = tmp_path / "h2o_ovgf"
    msindo_ovgf(*_H2O, output=str(stem))
    bib = (tmp_path / "h2o_ovgf.bibtex").read_text()
    assert "danovich_ovgf_1997" in bib
    assert "cederbaum_ovgf_1975" in bib
    assert "vonniessen_ovgf_1984" in bib
    assert "ahlswede_jug_msindo_1_1999" in bib
    assert "valeev_libint" not in bib  # INDO: no Gaussian ints


def test_msindo_uovgf_citations(tmp_path):
    """The open-shell (UOVGF) return site emits the same bundle as the
    closed-shell one -- they are separate `return` statements."""
    from vibeqc.semiempirical.methods.msindo import msindo_ovgf

    stem = tmp_path / "oh_uovgf"
    msindo_ovgf([8, 1], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]],
                multiplicity=2, output=str(stem))
    refs = (tmp_path / "oh_uovgf.references").read_text()
    assert "danovich_ovgf_1997" in refs


def test_msindo_ovgf_without_output_writes_nothing(tmp_path):
    """output=None stays the default: no citation sidecars, no behaviour change."""
    from vibeqc.semiempirical.methods.msindo import msindo_ovgf

    msindo_ovgf(*_H2O)
    assert not list(tmp_path.iterdir())


def test_msindo_cisd_odd_electron_raises():
    """Open-shell CISD is not wired — odd valence count fails fast."""
    from vibeqc.semiempirical.methods.msindo import msindo_cisd

    with pytest.raises(NotImplementedError, match="closed-shell"):
        msindo_cisd([7], [[0, 0, 0]])  # N atom


def test_run_job_msindo_surfaces_binding_energy(tmp_path):
    """run_job(method="msindo") writes the binding/atomization energy block and
    exposes result.binding_energy (the semiempirical analogue of the mean-field
    atomization block)."""
    import vibeqc as vq

    mol = vq.Molecule(
        [
            vq.Atom(8, [0, 0, 0]),
            vq.Atom(1, [0, 0.957 * msindo.ANGSTROM_TO_BOHR, 0]),
            vq.Atom(
                1, [0, -0.24 * msindo.ANGSTROM_TO_BOHR, 0.927 * msindo.ANGSTROM_TO_BOHR]
            ),
        ],
        0,
        1,
    )
    r = vq.run_job(mol, method="msindo", output=str(tmp_path / "h2o"), verbose=0)
    assert r.binding_energy < 0.0  # bound
    out = (tmp_path / "h2o.out").read_text()
    assert "## Atomization / binding energy" in out
    assert "Binding energy (E" in out
    assert "kcal/mol" in out
