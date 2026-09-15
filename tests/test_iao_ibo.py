"""Intrinsic atomic orbitals (IAO) and intrinsic bond orbitals (IBO).

Three families of assertion:

1. **Published partial charges.** Knizia, *J. Chem. Theory Comput.* 9, 4834
   (2013), doi:10.1021/ct400687b, Table 1 -- Hartree-Fock IAO charges for CH4
   and HCN, which the paper shows are insensitive to the orbital basis.
2. **Emergent Lewis structures.** Methane must come out as one core plus four
   equivalent C-H bonds; benzene as 6 cores + 12 two-centre sigma bonds + a
   multi-centre pi system (Knizia Figure 5).
3. **The Jacobi increments.** Knizia's Appendix D prints them with corrupted
   subscripts, so they are *derived* in ``localise.ibo_localise`` and pinned
   here against a brute-force scan of the rotation functional rather than
   transcribed from the PDF. See handovers/HANDOVER_IBO.md § 4b.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest
from vibeqc import BasisSet, RHFOptions, compute_overlap, run_rhf
from vibeqc._vibeqc_core import Atom, Molecule
from vibeqc.iao import (
    IAO_REFERENCE_MAX_Z,
    analyse_ibo,
    analyse_localization,
    build_iaos,
    iao_charges,
    iao_reference,
    iao_unsupported_reason,
)
from vibeqc.localise import ibo_localise, ibo_objective

ANGSTROM_TO_BOHR = 1.8897261254578281

# Knizia's test molecules were optimised at DF-RKS/PBE/def2-TZVPP; these are
# standard experimental geometries, which is why the charges below are
# compared with a 0.03 e window rather than to the printed digits.
_CH4_D = 1.087 * ANGSTROM_TO_BOHR / np.sqrt(3.0)
CH4_ATOMS = [
    (6, [0.0, 0.0, 0.0]),
    (1, [_CH4_D, _CH4_D, _CH4_D]),
    (1, [_CH4_D, -_CH4_D, -_CH4_D]),
    (1, [-_CH4_D, _CH4_D, -_CH4_D]),
    (1, [-_CH4_D, -_CH4_D, _CH4_D]),
]

HCN_ATOMS = [
    (1, [0.0, 0.0, -1.0655 * ANGSTROM_TO_BOHR]),
    (6, [0.0, 0.0, 0.0]),
    (7, [0.0, 0.0, 1.153 * ANGSTROM_TO_BOHR]),
]


def _benzene_atoms():
    """D6h benzene, r(CC) = 1.39 A, r(CH) = 1.09 A."""
    r_c = 1.39 * ANGSTROM_TO_BOHR
    r_h = (1.39 + 1.09) * ANGSTROM_TO_BOHR
    atoms = []
    for k in range(6):
        a = k * np.pi / 3.0
        atoms.append((6, [r_c * np.cos(a), r_c * np.sin(a), 0.0]))
        atoms.append((1, [r_h * np.cos(a), r_h * np.sin(a), 0.0]))
    return atoms


def _rhf(atoms, basis_name):
    """Converged closed-shell RHF plus the pieces an IAO analysis needs."""
    mol = Molecule([Atom(z, pos) for z, pos in atoms], charge=0, multiplicity=1)
    basis = BasisSet(mol, basis_name)
    opts = RHFOptions()
    opts.max_iter = 100
    result = run_rhf(mol, basis, opts)
    assert result.converged, f"{basis_name} RHF did not converge"
    n_occ = mol.n_electrons() // 2
    return mol, basis, np.asarray(result.mo_coeffs)[:, :n_occ].copy()


# ---------------------------------------------------------------------------
# 1. Published IAO partial charges (Knizia Table 1)
# ---------------------------------------------------------------------------

# Published target: Hartree-Fock IAO partial charges, Knizia (2013) Table 1.
# vibe-qc uses Huzinaga MINI as the minimal reference basis B2 where the
# printed row uses MINAO; Knizia's own footnote c shows that swap moves CH4
# carbon from -0.52 to -0.49, so the window below is the physically
# meaningful one, not slack.
IAO_CHARGE_TOL = 0.03

CHARGE_CASES = [
    # (label, atoms, basis, {atom index: published charge})
    ("CH4/def2-SVP", CH4_ATOMS, "def2-svp", {0: -0.49, 1: 0.12}),
    ("CH4/def2-TZVPP", CH4_ATOMS, "def2-tzvpp", {0: -0.52, 1: 0.13}),
    ("CH4/cc-pVTZ", CH4_ATOMS, "cc-pvtz", {0: -0.52, 1: 0.13}),
    ("HCN/def2-SVP", HCN_ATOMS, "def2-svp", {0: 0.21, 1: -0.01, 2: -0.20}),
    ("HCN/def2-TZVPP", HCN_ATOMS, "def2-tzvpp", {0: 0.22, 1: -0.01, 2: -0.21}),
]


@pytest.mark.parametrize("label,atoms,basis_name,expected", CHARGE_CASES)
def test_iao_charges_match_published_values(label, atoms, basis_name, expected):
    mol, basis, occupied = _rhf(atoms, basis_name)
    analysis = analyse_ibo(mol, basis, occupied)
    for index, published in expected.items():
        assert analysis.charges[index] == pytest.approx(
            published, abs=IAO_CHARGE_TOL
        ), f"{label}: atom {index}"


def test_iao_charges_conserve_total_charge():
    """Charges must sum to the molecular charge -- IAOs partition all electrons."""
    mol, basis, occupied = _rhf(CH4_ATOMS, "def2-svp")
    analysis = analyse_ibo(mol, basis, occupied)
    assert analysis.charges.sum() == pytest.approx(0.0, abs=1e-10)


def test_iao_charges_are_basis_set_stable():
    """The property that motivates IAOs over Mulliken.

    Knizia Table 1: CH4 carbon moves only -0.52 -> -0.52 from def2-TZVPP to
    cc-pVTZ, where Mulliken swings by an entire electron across comparable
    basis changes.
    """
    charges = []
    for basis_name in ("def2-tzvpp", "cc-pvtz"):
        mol, basis, occupied = _rhf(CH4_ATOMS, basis_name)
        charges.append(analyse_ibo(mol, basis, occupied).charges[0])
    assert abs(charges[0] - charges[1]) < 0.01


# ---------------------------------------------------------------------------
# 2. IAO construction invariants
# ---------------------------------------------------------------------------


def test_iaos_are_orthonormal_and_span_the_occupied_space():
    """The defining property of the construction (Knizia Appendix C).

    IAOs are orthonormal in S1, and they express the occupied MOs *exactly* --
    so each occupied orbital's atomic populations sum to 1.
    """
    mol, basis, occupied = _rhf(CH4_ATOMS, "def2-svp")
    overlap = np.asarray(compute_overlap(basis))
    reference = iao_reference(mol, basis)
    iaos = build_iaos(
        occupied, overlap, reference.overlap, reference.cross_overlap
    )

    identity = iaos.T @ overlap @ iaos
    assert np.abs(identity - np.eye(iaos.shape[1])).max() < 1e-10

    amplitudes = iaos.T @ overlap @ occupied
    assert np.abs(np.einsum("ri,ri->i", amplitudes, amplitudes) - 1.0).max() < 1e-10


def test_iao_reference_uses_huzinaga_mini_not_the_minao_guess_basis():
    """Regression guard on the reference-basis choice.

    ``ano-rcc-mb`` is what the MINAO *initial guess* uses and is the obvious
    thing to reach for, but it is a poor IAO reference: it shifts the CH4
    carbon charge from -0.47 to -0.39, roughly 20 %, and out of agreement
    with Knizia Table 1. See handovers/HANDOVER_IBO.md § 4a.
    """
    mol, basis, occupied = _rhf(CH4_ATOMS, "def2-svp")
    overlap = np.asarray(compute_overlap(basis))
    nuclear = np.array([z for z, _ in CH4_ATOMS], dtype=np.float64)

    measured = {}
    for name in ("mini", "ano-rcc-mb"):
        reference = iao_reference(mol, basis, name=name)
        iaos = build_iaos(
            occupied, overlap, reference.overlap, reference.cross_overlap
        )
        measured[name] = iao_charges(
            occupied, iaos, overlap, nuclear, reference.atom_indices
        )[0]

    assert measured["mini"] == pytest.approx(-0.473, abs=0.01)
    assert measured["ano-rcc-mb"] == pytest.approx(-0.389, abs=0.01)
    # The default must be the one that reproduces the paper.
    assert analyse_ibo(mol, basis, occupied).reference_basis == "mini"


# ---------------------------------------------------------------------------
# 3. Emergent Lewis structures
# ---------------------------------------------------------------------------


def test_methane_gives_one_core_and_four_equivalent_bonds():
    mol, basis, occupied = _rhf(CH4_ATOMS, "def2-svp")
    analysis = analyse_ibo(mol, basis, occupied)

    assert sorted(analysis.n_centres.tolist()) == [1, 2, 2, 2, 2]
    assert analysis.objective_final >= analysis.objective_initial

    populations = analysis.atom_populations
    core = np.flatnonzero(analysis.n_centres == 1)
    assert core.size == 1
    assert populations[core[0], 0] == pytest.approx(1.0, abs=1e-3)

    # The four C-H bonds are symmetry-equivalent: same carbon weight, and one
    # distinct hydrogen each.
    bonds = np.flatnonzero(analysis.n_centres == 2)
    carbon_weights = populations[bonds, 0]
    assert np.ptp(carbon_weights) < 1e-6
    assert carbon_weights[0] == pytest.approx(0.549, abs=0.02)
    assert sorted(int(np.argmax(populations[b, 1:])) for b in bonds) == [0, 1, 2, 3]


def test_benzene_reproduces_the_published_orbital_census():
    """Knizia Figure 5: six CC sigma and six CH sigma bonds, both localized,
    plus a pi system that cannot be expressed on few centres."""
    atoms = _benzene_atoms()
    mol, basis, occupied = _rhf(atoms, "def2-svp")
    analysis = analyse_ibo(mol, basis, occupied)

    counts = np.bincount(analysis.n_centres, minlength=4)
    assert counts[1] == 6, "six carbon 1s cores"
    assert counts[2] == 12, "six CC sigma + six CH sigma"
    assert analysis.n_centres.max() >= 3, "the pi system must be multi-centre"
    assert int(analysis.n_centres.size) == 21


def test_localisation_never_decreases_the_objective():
    mol, basis, occupied = _rhf(HCN_ATOMS, "def2-svp")
    analysis = analyse_ibo(mol, basis, occupied)
    assert analysis.objective_final >= analysis.objective_initial
    assert analysis.n_sweeps <= 50


# ---------------------------------------------------------------------------
# 4. The Jacobi increments, pinned numerically (not transcribed)
# ---------------------------------------------------------------------------


def _pair_objective(q_ii, q_jj, q_ij, phi, power):
    """L(phi) for a single (i, j) rotation, summed over atoms."""
    c, s = np.cos(phi), np.sin(phi)
    new_ii = c * c * q_ii + s * s * q_jj + 2 * c * s * q_ij
    new_jj = s * s * q_ii + c * c * q_jj - 2 * c * s * q_ij
    return float(np.sum(new_ii**power) + np.sum(new_jj**power))


def _brute_force_optimum(q_ii, q_jj, q_ij, power, n=40001):
    grid = np.linspace(-np.pi / 4, np.pi / 4, n)
    c, s = np.cos(grid)[:, None], np.sin(grid)[:, None]
    new_ii = c * c * q_ii + s * s * q_jj + 2 * c * s * q_ij
    new_jj = s * s * q_ii + c * c * q_jj - 2 * c * s * q_ij
    values = (new_ii**power).sum(1) + (new_jj**power).sum(1)
    best = int(np.argmax(values))
    return float(grid[best]), float(values[best])


def _random_pair_block(rng, n_atoms=6):
    """A realistic IAO pair block: two orbitals each mostly on one atom.

    Rows sum to 1 (IAOs span the occupied space) and the transition
    populations sum to zero (the two orbitals are orthogonal).
    """
    q_ii = np.full(n_atoms, 0.01)
    q_jj = np.full(n_atoms, 0.01)
    a, b = rng.choice(n_atoms, 2, replace=False)
    q_ii[a] = 0.9
    q_jj[b] = 0.9
    q_ii += rng.random(n_atoms) * 0.05
    q_jj += rng.random(n_atoms) * 0.05
    q_ii /= q_ii.sum()
    q_jj /= q_jj.sum()
    q_ij = rng.normal(scale=0.08, size=n_atoms)
    q_ij -= q_ij.mean()
    return q_ii, q_jj, q_ij


def _rotation_angle(q_ii, q_jj, q_ij, power):
    """Reproduce the increments used inside ``ibo_localise`` for one pair."""
    if power == 2:
        a = np.sum(4.0 * q_ij**2 - (q_ii - q_jj) ** 2)
        b = np.sum(4.0 * q_ij * (q_ii - q_jj))
    else:
        a = np.sum(
            -(q_ii**4)
            - q_jj**4
            + 6.0 * (q_ii**2 + q_jj**2) * q_ij**2
            + q_ii**3 * q_jj
            + q_ii * q_jj**3
        )
        b = np.sum(4.0 * q_ij * (q_ii**3 - q_jj**3))
    return 0.25 * np.arctan2(b, -a)


def test_p2_increment_is_the_exact_maximiser():
    """At p=2 the closed form is exact -- this pins the ``(Q_ii - Q_jj)^2``
    reading against the paper's mangled ``(Q_ii - Q_ij^2)``."""
    rng = np.random.default_rng(20260805)
    for _ in range(50):
        q_ii, q_jj, q_ij = _random_pair_block(rng)
        phi = _rotation_angle(q_ii, q_jj, q_ij, 2)
        phi_bf, best = _brute_force_optimum(q_ii, q_jj, q_ij, 2)
        assert _pair_objective(q_ii, q_jj, q_ij, phi, 2) == pytest.approx(
            best, abs=1e-9
        )
        assert abs(phi - phi_bf) < 1e-4


def test_p4_increment_never_decreases_the_objective():
    """At p=4 the rotation is approximate by construction -- Knizia neglects
    high-order terms in phi -- so the contract is that it never decreases L
    and captures nearly all of the achievable gain."""
    rng = np.random.default_rng(20260805)
    worst_shortfall = 0.0
    for _ in range(100):
        q_ii, q_jj, q_ij = _random_pair_block(rng)
        phi = _rotation_angle(q_ii, q_jj, q_ij, 4)
        start = _pair_objective(q_ii, q_jj, q_ij, 0.0, 4)
        achieved = _pair_objective(q_ii, q_jj, q_ij, phi, 4)
        _, best = _brute_force_optimum(q_ii, q_jj, q_ij, 4)
        assert achieved >= start - 1e-12
        gain = best - start
        if gain > 1e-10:
            worst_shortfall = max(worst_shortfall, (best - achieved) / gain)
    # Measured 4.7e-3 over this generator; an order of magnitude of headroom.
    assert worst_shortfall < 5e-2


def test_ibo_localise_rejects_unsupported_power():
    rng = np.random.default_rng(0)
    coefficients = rng.random((4, 2))
    atoms = np.array([0, 0, 1, 1])
    with pytest.raises(ValueError, match="power must be 2 or 4"):
        ibo_localise(coefficients, atoms, power=3)


def test_ibo_localise_does_not_mutate_its_input():
    rng = np.random.default_rng(1)
    coefficients = rng.random((6, 3))
    original = coefficients.copy()
    atoms = np.array([0, 0, 1, 1, 2, 2])
    ibo_localise(coefficients, atoms)
    assert np.array_equal(coefficients, original)


def test_ibo_objective_increases_under_localisation():
    rng = np.random.default_rng(2)
    atoms = np.array([0, 0, 1, 1, 2, 2])
    raw = rng.random((6, 3))
    # Orthonormalise so the block is a legitimate orbital set.
    coefficients, _ = np.linalg.qr(raw)
    before = ibo_objective(coefficients, atoms)
    localised, _ = ibo_localise(coefficients, atoms)
    assert ibo_objective(localised, atoms) >= before


# ---------------------------------------------------------------------------
# 5. Gates
# ---------------------------------------------------------------------------


def test_ecp_jobs_are_refused():
    """An all-electron minimal reference cannot partition a valence-only
    target basis: for Au, lanl2dz carries 22 functions for 19 explicit
    electrons against 43 all-electron reference functions."""
    mol = Molecule([Atom(z, pos) for z, pos in CH4_ATOMS], charge=0, multiplicity=1)
    reason = iao_unsupported_reason(mol, uses_ecp=True)
    assert reason is not None and "ECP" in reason


def test_elements_beyond_the_reference_basis_are_refused():
    mol = Molecule([Atom(92, [0.0, 0.0, 0.0])], charge=0, multiplicity=1)
    reason = iao_unsupported_reason(mol)
    assert reason is not None
    assert str(IAO_REFERENCE_MAX_Z) in reason


def test_supported_system_is_not_refused():
    mol = Molecule([Atom(z, pos) for z, pos in CH4_ATOMS], charge=0, multiplicity=1)
    assert iao_unsupported_reason(mol) is None


# ---------------------------------------------------------------------------
# 6. QVF surface
# ---------------------------------------------------------------------------


def test_iao_charges_ride_atom_properties_and_validate(tmp_path):
    """End-to-end: localize="ibo" emits both the localized wavefunction
    section and an ``iao_charge`` member in ``atom_properties``, and the
    result validates against the manifest schema."""
    import vibeqc as vq
    from vibeqc.output.formats.qvf import validate_qvf

    mol = Molecule([Atom(z, pos) for z, pos in CH4_ATOMS], charge=0, multiplicity=1)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        result = vq.run_job(
            molecule=mol, basis="def2-svp", method="rhf", localize="ibo"
        )
        assert result.converged
        qvf_path = next(tmp_path.glob("*.qvf"))
        validate_qvf(str(qvf_path))

        with zipfile.ZipFile(qvf_path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        by_id = {s["id"]: s for s in manifest["sections"]}

        # Asking for one criterion emits exactly that criterion's section.
        assert by_id["wf_localized_ibo"]["kind"] == "wavefunction.gto"
        assert "wf_localized_boys" not in by_id
        assert "iao_charge" in by_id["props0"]["members"]
    finally:
        os.chdir(cwd)


def test_localize_resolver_accepts_the_documented_spellings():
    from vibeqc.runner import DEFAULT_LOCALIZE_METHODS, _resolve_localize_methods

    assert _resolve_localize_methods(None) == DEFAULT_LOCALIZE_METHODS
    assert _resolve_localize_methods(True) == DEFAULT_LOCALIZE_METHODS
    assert _resolve_localize_methods(False) == ()
    assert _resolve_localize_methods("none") == ()
    assert _resolve_localize_methods("ibo") == ("ibo",)
    assert _resolve_localize_methods("pm") == ("pipek-mezey",)
    assert _resolve_localize_methods(["boys", "pipek_mezey"]) == (
        "boys",
        "pipek-mezey",
    )
    # Duplicates collapse rather than emitting the same section twice.
    assert _resolve_localize_methods(["ibo", "ibo"]) == ("ibo",)
    with pytest.raises(ValueError, match="unknown localize criterion"):
        _resolve_localize_methods("nope")


def test_boys_and_ibo_disagree_on_benzene_as_the_literature_says():
    """The reason it is worth shipping all three.

    Boys maximises the orbital dipole spread and so mixes sigma and pi into
    equivalent "banana" bonds; IBO and Pipek-Mezey maximise atomic
    populations and keep the pi system separate. The p=2 PM objective has
    a degenerate benzene pi manifold (Knizia 2013, Appendix D), so its atom-
    centre census is not unique. Reflection through the molecular plane
    tests sigma/pi separation without selecting a member of that manifold.
    """
    mol, basis, occupied = _rhf(_benzene_atoms(), "def2-svp")
    census = {}
    analyses = {}
    for method in ("ibo", "boys", "pipek-mezey"):
        analysis = analyse_localization(mol, basis, occupied, method=method)
        analyses[method] = analysis
        counts = np.bincount(analysis.n_centres, minlength=5)
        census[method] = (int(counts[1]), int(counts[2]), int(counts[3:].sum()))

    # All three agree on the six carbon cores and on the total orbital count.
    for method, (cores, _, _) in census.items():
        assert cores == 6, method

    assert census["ibo"] == (6, 12, 3), "IBO keeps a three-centre pi system"
    # Pipek-Mezey 1989 Sec. V (10.1063/1.456588) establishes sigma/pi
    # separation. Knizia 2013 Appendix D (10.1021/ct400687b) explains the
    # p=2 zero mode; Lehtola 2014 p.646 (10.1021/ct401016x) shows distinct
    # benzene pi centre patterns. Probe spatial parity directly.
    from vibeqc import evaluate_ao

    points = np.random.default_rng(456).normal(size=(256, 3)) * 2.5
    reflected = points.copy()
    reflected[:, 2] *= -1
    values = np.asarray(evaluate_ao(basis, points))
    mirror_values = np.asarray(evaluate_ao(basis, reflected))
    for method, analysis in analyses.items():
        orbitals = values @ analysis.coefficients
        mirror = mirror_values @ analysis.coefficients
        norms = np.linalg.norm(orbitals, axis=0)
        even = np.linalg.norm(orbitals - mirror, axis=0) / norms < 1e-9
        odd = np.linalg.norm(orbitals + mirror, axis=0) / norms < 1e-9
        if method in ("ibo", "pipek-mezey"):
            assert np.count_nonzero(even) == 18, method
            assert np.count_nonzero(odd) == 3, method
            assert np.all(even | odd), method
        else:
            assert np.any(~(even | odd)), "Boys mixes sigma and pi"
    assert census["boys"][2] == 0, "Boys leaves no multi-centre orbital"
    assert census["boys"][1] > census["ibo"][1], "Boys makes extra 2c bonds"


def test_every_criterion_describes_the_same_occupied_space():
    """Localization is a unitary rotation within the occupied space, so
    quantities that are properties of that space -- the IAO charges -- must
    not depend on which criterion ran."""
    mol, basis, occupied = _rhf(CH4_ATOMS, "def2-svp")
    charges = [
        analyse_localization(mol, basis, occupied, method=m).charges
        for m in ("ibo", "boys", "pipek-mezey")
    ]
    for other in charges[1:]:
        assert np.abs(other - charges[0]).max() < 1e-10


def test_analyse_localization_rejects_unknown_method():
    mol, basis, occupied = _rhf(CH4_ATOMS, "def2-svp")
    with pytest.raises(ValueError, match="unknown localization method"):
        analyse_localization(mol, basis, occupied, method="edmiston")


def test_default_run_emits_one_section_per_criterion(tmp_path):
    import vibeqc as vq
    from vibeqc.output.formats.qvf import validate_qvf

    mol = Molecule([Atom(z, pos) for z, pos in CH4_ATOMS], charge=0, multiplicity=1)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        # No localize= argument at all: the default set must fire.
        result = vq.run_job(molecule=mol, basis="def2-svp", method="rhf")
        assert result.converged
        qvf_path = next(tmp_path.glob("*.qvf"))
        validate_qvf(str(qvf_path))
        with zipfile.ZipFile(qvf_path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        ids = {s["id"] for s in manifest["sections"]}
        assert {
            "wf_localized_ibo",
            "wf_localized_boys",
            "wf_localized_pipek_mezey",
        } <= ids
    finally:
        os.chdir(cwd)


def test_localize_false_emits_no_localized_section(tmp_path):
    import vibeqc as vq

    mol = Molecule([Atom(z, pos) for z, pos in CH4_ATOMS], charge=0, multiplicity=1)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        vq.run_job(
            molecule=mol, basis="def2-svp", method="rhf", localize=False
        )
        qvf_path = next(tmp_path.glob("*.qvf"))
        with zipfile.ZipFile(qvf_path) as archive:
            manifest = json.loads(archive.read("manifest.json"))
        ids = {s["id"] for s in manifest["sections"]}
        assert not any(i.startswith("wf_localized") for i in ids)
        props = next(s for s in manifest["sections"] if s["id"] == "props0")
        assert "iao_charge" not in props["members"]
    finally:
        os.chdir(cwd)


def test_manifest_schema_copies_stay_byte_identical():
    """``iao_charge`` had to be added to a closed member set that lives in
    three places; the two qvf-writer copies are real files, not symlinks."""
    root = Path(__file__).resolve().parent.parent
    canonical = (
        root / "python" / "vibeqc" / "output" / "formats" / "qvf_manifest.schema.json"
    )
    reference = canonical.read_bytes()
    assert b'"iao_charge"' in reference

    for relative in (
        Path("qvf-writer") / "spec" / "qvf_manifest.schema.json",
        Path("qvf-writer") / "python" / "qvf_manifest.schema.json",
        Path("vibe-view") / "src" / "vibeview" / "schema.json",
    ):
        copy = root / relative
        if not copy.exists():
            continue
        assert copy.read_bytes() == reference, f"{relative} drifted from canonical"
