"""Classifier + strategy resolver for automatic periodic convergence.

Pins the transparency contract (explicit knobs win; mode labels) and
the v1 strategy table for the four canonical system classes:
MgO (ionic), diamond (covalent), Na bcc (metallic), H2-in-a-box
(molecular-limit).
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.periodic_convergence_auto import (
    CONVERGENCE_KNOBS,
    _VACUUM_AXIS_BOHR,
    _lattice_lengths,
    classify_periodic_system,
    insulator_smearing_warning,
    resolve_convergence_strategy,
)

ANG2BOHR = 1.0 / 0.529177210903


def _mgo():
    a = 4.21 * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    return vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [a / 2, a / 2, a / 2])],
    )


def _diamond():
    a = 3.567 * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    return vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(6, [0, 0, 0]), vq.Atom(6, [a / 4, a / 4, a / 4])],
    )


def _na_bcc():
    a = 4.29 * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [[-1.0, 1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, -1.0]]
    )
    return vq.PeriodicSystem(3, lattice, [vq.Atom(11, [0, 0, 0])])


def _h2_box():
    return vq.PeriodicSystem(
        3,
        np.eye(3) * 25.0,
        [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])],
    )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def test_classify_mgo_ionic():
    cls = classify_periodic_system(_mgo())
    assert cls.profile == "ionic-insulator"
    assert cls.en_spread == pytest.approx(3.44 - 1.31, abs=1e-6)
    assert any("ionic" in r for r in cls.reasons)


def test_classify_diamond_covalent():
    cls = classify_periodic_system(_diamond())
    assert cls.profile == "covalent-insulator"
    assert cls.en_spread == pytest.approx(0.0, abs=1e-12)


def test_classify_na_metallic():
    cls = classify_periodic_system(_na_bcc())
    assert cls.profile == "metallic-candidate"
    assert cls.open_shell  # 11 electrons/cell — odd


def test_classify_h2_box_molecular_limit():
    cls = classify_periodic_system(_h2_box())
    assert cls.profile == "molecular-limit"


def _skewed_compact():
    # Deliberately non-symmetric compact cell: `system.lattice` columns are
    # the Cartesian lattice vectors (cpp/include/vibeqc/periodic.hpp:32,
    # "Columns = Cartesian lattice vectors"), so the axis lengths are the
    # column norms [14.84, 14.84, 12.5] bohr -- all under the 20-bohr vacuum
    # threshold, i.e. a compact crystal. The row norms are [21.65, 8, 8]:
    # reading axis lengths off the rows spuriously sees a 21.65-bohr vacuum
    # axis and mis-classifies the cell molecular-limit.
    lattice = np.array(
        [[12.5, 12.5, 12.5], [8.0, 0.0, 0.0], [0.0, 8.0, 0.0]]
    )
    return vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(6, [0, 0, 0]), vq.Atom(6, [2.0, 2.0, 2.0])],
    )


def test_lattice_lengths_uses_column_vectors():
    """Axis lengths are the column norms of `system.lattice`, not row norms.

    Row norms equal column norms only for a symmetric (cubic/orthorhombic)
    lattice matrix; on this skewed cell they differ, which is what makes the
    row-vs-column bug observable.
    """
    system = _skewed_compact()
    lattice = np.asarray(system.lattice, dtype=float)
    np.testing.assert_allclose(
        np.sort(_lattice_lengths(system)),
        np.sort(np.linalg.norm(lattice, axis=0)),
    )
    # The longest true lattice vector is < 20 bohr; the buggy row-norm path
    # would report a spurious 21.65-bohr axis over the vacuum threshold.
    assert _lattice_lengths(system).max() < _VACUUM_AXIS_BOHR


def test_classify_skewed_compact_not_molecular_limit():
    """A skewed compact crystal must not be mis-read as a vacuum box."""
    cls = classify_periodic_system(_skewed_compact())
    assert cls.profile != "molecular-limit"
    assert not any("vacuum" in r for r in cls.reasons)


# ---------------------------------------------------------------------------
# Strategy table
# ---------------------------------------------------------------------------
def test_strategy_mgo_rks_auto_default():
    s = resolve_convergence_strategy(_mgo(), method="RKS", convergence=None)
    assert s.mode == "auto-default"
    assert s.value("fock_mixing") == pytest.approx(0.30)
    assert s.value("smearing_temperature") == 0.0
    assert s.value("level_shift") == 0.0
    assert s.knobs["fock_mixing"].source == "auto"


def test_strategy_mgo_rhf_no_smearing():
    """HF drivers reject finite temperature — auto must never set it."""
    s = resolve_convergence_strategy(_mgo(), method="RHF", convergence=None)
    assert s.value("smearing_temperature") == 0.0
    assert s.value("fock_mixing") == pytest.approx(0.30)


def test_strategy_diamond_rhf_plain():
    s = resolve_convergence_strategy(
        _diamond(), method="RHF", convergence=None
    )
    for name in CONVERGENCE_KNOBS:
        assert s.value(name) == 0.0, name


def test_strategy_diamond_rks_reports_driver_fmixing_floor():
    """The BIPOLE KS drivers apply FMIXING 30% for DFT functionals only
    when DIIS is off (Gap-B validation 2026-07-13) — by default (DIIS
    on, no floor requested) the strategy reports 0; with the floor flag
    (a DIIS-off BIPOLE KS run) it reports the driver's 30%."""
    s = resolve_convergence_strategy(
        _diamond(), method="RKS", convergence=None
    )
    assert s.value("fock_mixing") == 0.0
    assert s.value("smearing_temperature") == 0.0
    s_no_diis = resolve_convergence_strategy(
        _diamond(),
        method="RKS",
        convergence=None,
        ks_driver_fock_mixing_floor=True,
    )
    assert s_no_diis.value("fock_mixing") == pytest.approx(0.30)
    assert "driver" in s_no_diis.knobs["fock_mixing"].reason


def test_strategy_na_ks_metal():
    s = resolve_convergence_strategy(_na_bcc(), method="RKS", convergence=None)
    assert s.value("smearing_temperature") == pytest.approx(0.005)
    assert s.value("fock_mixing") == pytest.approx(0.50)


def test_strategy_na_hf_level_shift_not_smearing():
    s = resolve_convergence_strategy(_na_bcc(), method="UHF", convergence=None)
    assert s.value("smearing_temperature") == 0.0
    assert s.value("level_shift") == pytest.approx(0.2)


def test_strategy_h2_box_untouched():
    s = resolve_convergence_strategy(_h2_box(), method="RHF", convergence=None)
    for name in CONVERGENCE_KNOBS:
        assert s.value(name) == 0.0, name


def test_strategy_h2_box_ks_floor_only():
    """Molecular-limit KS: no aids by default (DIIS on); the driver
    floor appears only when explicitly requested (DIIS-off runs)."""
    s = resolve_convergence_strategy(_h2_box(), method="RKS", convergence=None)
    assert s.value("fock_mixing") == 0.0
    assert s.value("smearing_temperature") == 0.0
    assert s.value("level_shift") == 0.0
    s_no_diis = resolve_convergence_strategy(
        _h2_box(),
        method="RKS",
        convergence=None,
        ks_driver_fock_mixing_floor=True,
    )
    assert s_no_diis.value("fock_mixing") == pytest.approx(0.30)  # driver floor


# ---------------------------------------------------------------------------
# The transparency contract
# ---------------------------------------------------------------------------
def test_explicit_knob_disables_auto_by_default():
    """User gave a knob and no convergence= → manual mode, nothing
    auto-filled (today's behaviour for explicit options)."""
    s = resolve_convergence_strategy(
        _mgo(),
        method="RKS",
        convergence=None,
        explicit={"fock_mixing": 0.10},
    )
    assert s.mode == "manual"
    assert s.value("fock_mixing") == pytest.approx(0.10)
    assert s.knobs["fock_mixing"].source == "explicit"
    # auto did NOT fill smearing
    assert s.value("smearing_temperature") == 0.0
    assert s.knobs["smearing_temperature"].source == "default"
    assert s.classification is None


def test_convergence_auto_keyword_fills_around_explicit():
    """convergence="auto" + explicit knob → explicit wins per-knob,
    auto fills the rest, mode says requested."""
    s = resolve_convergence_strategy(
        _mgo(),
        method="RKS",
        convergence="auto",
        explicit={"smearing_temperature": 0.02},
    )
    assert s.mode == "auto-requested"
    assert s.value("smearing_temperature") == pytest.approx(0.02)
    assert s.knobs["smearing_temperature"].source == "explicit"
    assert s.value("fock_mixing") == pytest.approx(0.30)
    assert s.knobs["fock_mixing"].source == "auto"


def test_convergence_off_plain_defaults():
    s = resolve_convergence_strategy(_mgo(), method="RKS", convergence="off")
    assert s.mode == "off"
    for name in CONVERGENCE_KNOBS:
        assert s.value(name) == 0.0
        assert s.knobs[name].source == "default"


def test_unknown_knob_rejected():
    with pytest.raises(ValueError, match="unknown knob"):
        resolve_convergence_strategy(
            _mgo(), method="RKS", convergence=None,
            explicit={"bogus": 1.0},
        )


def test_bad_convergence_keyword_rejected():
    with pytest.raises(ValueError, match="convergence must be"):
        resolve_convergence_strategy(
            _mgo(), method="RKS", convergence="magic"
        )


def test_log_lines_state_mode_and_reasons():
    s = resolve_convergence_strategy(_mgo(), method="RKS", convergence=None)
    text = "\n".join(s.log_lines())
    assert "AUTO (default" in text
    assert "ionic-insulator" in text
    assert "fock_mixing = 0.3" in text
    assert "smearing" in text
    assert "integer occupations" in text
    assert "[auto]" in text

    s2 = resolve_convergence_strategy(
        _mgo(), method="RKS", convergence=None,
        explicit={"level_shift": 0.5},
    )
    text2 = "\n".join(s2.log_lines())
    assert "manual" in text2
    assert "[explicit]" in text2


def test_insulator_smearing_warning_for_mgo():
    message = insulator_smearing_warning(_mgo(), 0.01)
    assert message is not None
    assert "ionic-insulator" in message
    assert "wrong-energy basin" in message


def test_insulator_smearing_warning_for_supplied_gap():
    message = insulator_smearing_warning(
        _na_bcc(),
        0.005,
        band_gap_hartree=0.2,
        metallic=True,
    )
    assert message is not None
    assert "supplied band gap" in message


def test_insulator_smearing_warning_silent_when_safe_or_disabled():
    assert insulator_smearing_warning(_mgo(), 0.0) is None
    assert insulator_smearing_warning(_na_bcc(), 0.005, metallic=True) is None


def test_run_periodic_job_emits_insulator_smearing_guard(tmp_path, monkeypatch):
    """The helper is wired into the public runner before SCF dispatch."""
    from vibeqc.periodic_runner import run_periodic_job

    system = _mgo()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    class StopBeforeSCF(Exception):
        pass

    def stop_after_warning(*args, **kwargs):
        raise StopBeforeSCF

    monkeypatch.setattr(
        "vibeqc.pbc_bipole_rks.run_pbc_bipole_rks", stop_after_warning
    )
    with pytest.warns(UserWarning, match="wrong-energy basin"):
        # A valid selector reaches the warning before SCF. Malformed selectors
        # now fail before calculation-related warnings or output assembly.
        with pytest.raises(StopBeforeSCF):
            run_periodic_job(
                system,
                basis,
                method="RKS",
                functional="lda",
                jk_method="bipole",
                kpoints=(1, 1, 1),
                smearing_temperature=0.01,
                initial_guess="AUTO",
                output=tmp_path / "smearing-guard",
                write_molden_file=False,
            )


# ---------------------------------------------------------------------------
# Strategy v2: post-SCF re-classification
# ---------------------------------------------------------------------------
from types import SimpleNamespace  # noqa: E402

from vibeqc.periodic_convergence_auto import (  # noqa: E402
    converged_gap_hartree,
    post_scf_profile_check,
)


def _fake_closed(eps_per_k):
    return SimpleNamespace(mo_energies=[np.asarray(e) for e in eps_per_k])


def test_converged_gap_closed_shell_indirect():
    # VBM at k0 (-0.1), CBM at k1 (+0.05) -> indirect gap 0.15
    r = _fake_closed([[-0.5, -0.1, 0.4], [-0.6, -0.2, 0.05]])
    assert converged_gap_hartree(r, n_alpha=2) == pytest.approx(0.15)


def test_converged_gap_open_shell_min_over_spins():
    r = SimpleNamespace(
        mo_energies_alpha=[np.array([-0.5, -0.1, 0.4])],
        mo_energies_beta=[np.array([-0.6, 0.02, 0.5])],
    )
    # alpha gap (n=2): 0.4-(-0.1)=0.5 ; beta gap (n=1): 0.02-(-0.6)=0.62
    assert converged_gap_hartree(r, n_alpha=2, n_beta=1) == pytest.approx(0.5)


def test_converged_gap_none_without_virtuals():
    r = _fake_closed([[-0.5, -0.1]])
    assert converged_gap_hartree(r, n_alpha=2) is None


def test_converged_gap_handles_degenerate_band_manifolds():
    r = _fake_closed(
        [
            [[-0.6, -0.6], [-0.10, -0.08], [0.40, 0.42]],
            [[-0.7, -0.7], [-0.20, -0.18], [0.05, 0.07]],
        ]
    )

    # The VBM is the highest member of the occupied manifold at k0;
    # the CBM is the lowest member of the virtual manifold at k1.
    assert converged_gap_hartree(r, n_alpha=2) == pytest.approx(0.13)


def test_post_scf_check_warns_on_conducting_insulator_guess():
    s = resolve_convergence_strategy(_mgo(), method="RKS", convergence=None)
    out = post_scf_profile_check(s, 1e-5)
    assert out is not None
    level, msg = out
    assert level == "warning"
    assert "conducting" in msg


def test_post_scf_check_notes_insulating_metal_guess():
    s = resolve_convergence_strategy(_na_bcc(), method="RKS", convergence=None)
    out = post_scf_profile_check(s, 0.25)
    assert out is not None
    level, msg = out
    assert level == "note"
    assert "insulating" in msg


def test_post_scf_check_silent_when_profile_confirmed():
    s = resolve_convergence_strategy(_mgo(), method="RKS", convergence=None)
    assert post_scf_profile_check(s, 0.2) is None


def test_post_scf_check_silent_in_manual_mode():
    s = resolve_convergence_strategy(
        _mgo(), method="RKS", convergence=None,
        explicit={"fock_mixing": 0.1},
    )
    assert post_scf_profile_check(s, 1e-5) is None


# Monolayer sheets, a = 2.504 A (h-BN) / 2.461 A (graphene), 20 A of vacuum.
#
# ``LATTICE_VECTORS_AS_ROWS`` is the PySCF `cell.a` convention. PeriodicSystem
# takes lattice vectors as COLUMNS, so every constructor below transposes.
# Handing the row array straight to PeriodicSystem is not a loud failure: it
# builds a well-formed but DIFFERENT cell (|a1| 4.7319 -> 5.2904, angle 60 ->
# 63.43 deg) of identical volume, so a volume-per-atom profiler check cannot
# see it. `test_sheet_fixtures_are_the_hexagonal_cells_they_claim_to_be`
# below is the guard.
_HBN_LATTICE_VECTORS_AS_ROWS = np.array(
    [
        [4.731874216062929, 0.0, 0.0],
        [2.3659371080314644, 4.097923278623072, 0.0],
        [0.0, 0.0, 37.794522492515405],
    ]
)
_GRAPHENE_LATTICE_VECTORS_AS_ROWS = np.array(
    [
        [4.6511, 0.0, 0.0],
        [2.32555, 4.02797, 0.0],
        [0.0, 0.0, 37.794522492515405],
    ]
)
_SHEET_Z = 18.897261246257703


def _hbn_sheet():
    """Monolayer h-BN: N at (0,0), B at the (1/3,1/3) honeycomb site."""
    a1, a2 = (
        _HBN_LATTICE_VECTORS_AS_ROWS[0],
        _HBN_LATTICE_VECTORS_AS_ROWS[1],
    )
    boron = (a1 + a2) / 3.0
    return vq.PeriodicSystem(
        2,
        _HBN_LATTICE_VECTORS_AS_ROWS.T,
        [
            vq.Atom(5, [boron[0], boron[1], _SHEET_Z]),
            vq.Atom(7, [0.0, 0.0, _SHEET_Z]),
        ],
        0,
        1,
    )


def _graphene_sheet():
    """Monolayer graphene: C at (0,0) and at the (1/3,1/3) honeycomb site."""
    return vq.PeriodicSystem(
        2,
        _GRAPHENE_LATTICE_VECTORS_AS_ROWS.T,
        [
            vq.Atom(6, [0.0, 0.0, _SHEET_Z]),
            vq.Atom(6, [2.32555, 1.342657, _SHEET_Z]),
        ],
        0,
        1,
    )


def test_sheet_fixtures_are_the_hexagonal_cells_they_claim_to_be():
    """The 2D fixtures must be the sheets their names promise.

    These fixtures shipped with two independent geometry defects, and no test
    could see either one because the profiler only reads dimensionality and
    volume per atom -- both invariant under a transpose.

    1. The lattice array was handed to ``PeriodicSystem`` in PySCF's
       rows-are-vectors convention while ``PeriodicSystem`` reads COLUMNS,
       producing a well-formed 63.43-degree cell instead of the hexagonal
       60-degree one.
    2. h-BN's boron sat at fractional (1/3, 2/3) -- the second honeycomb site
       of a **120-degree** cell -- inside a **60-degree** cell, where the
       second site is (1/3, 1/3). That put B-N at 0.835 A against a real
       1.446 A: not a strained h-BN, just not h-BN.

    Both are the well-formed-halves / wrong-pairing class: each half is
    individually correct and their combination is silently wrong. So assert
    on cell *content* -- the lattice-vector lengths, the enclosed angle, and
    the nearest-neighbour bond -- not on shape or volume.

    Literature: a(h-BN) = 2.504 A, a(graphene) = 2.461 A; both honeycombs put
    the nearest neighbour at a/sqrt(3), i.e. 1.4457 A and 1.4210 A.
    """
    bohr_to_angstrom = 0.5291772109

    for system, a_angstrom, bond_angstrom in (
        (_hbn_sheet(), 2.504, 1.4457),
        (_graphene_sheet(), 2.4612, 1.4210),
    ):
        lattice = np.asarray(system.lattice, dtype=float)
        a1, a2 = lattice[:, 0], lattice[:, 1]  # COLUMNS are the vectors
        n1 = np.linalg.norm(a1) * bohr_to_angstrom
        n2 = np.linalg.norm(a2) * bohr_to_angstrom
        assert abs(n1 - a_angstrom) < 1e-3, f"|a1| = {n1:.4f} A"
        assert abs(n2 - a_angstrom) < 1e-3, f"|a2| = {n2:.4f} A"
        cosine = float(a1 @ a2 / (np.linalg.norm(a1) * np.linalg.norm(a2)))
        # 1e-4 deg, not exact: the graphene vectors are rounded to 5 decimals
        # in bohr, which lands the angle at 59.9999953. The transposed cell it
        # has to separate this from sits at 63.43 deg, five orders away.
        angle = np.degrees(np.arccos(cosine))
        assert abs(angle - 60.0) < 1e-4, f"angle(a1, a2) = {angle:.6f} deg"

        positions = [np.asarray(list(atom.xyz), dtype=float)
                     for atom in system.unit_cell]
        shortest = min(
            np.linalg.norm(p - q - m * a1 - n * a2)
            for i, p in enumerate(positions)
            for j, q in enumerate(positions)
            for m in (-1, 0, 1)
            for n in (-1, 0, 1)
            if not (i == j and m == 0 and n == 0)
        ) * bohr_to_angstrom
        assert abs(shortest - bond_angstrom) < 1e-3, (
            f"nearest-neighbour distance {shortest:.4f} A, expected "
            f"{bond_angstrom:.4f} A"
        )


@pytest.mark.parametrize("system", [_hbn_sheet(), _graphene_sheet()])
def test_low_dimensional_cells_are_named_unclassified_not_inconclusive(system):
    """A 2D cell must not be reported as a weighed-and-inconclusive call.

    The covalent / ionic / metallic branches are all gated on ``tight``,
    which is ``dim == 3`` by construction, so NO dim=1 or dim=2 cell can
    reach them. Reporting that as "no decisive signal" implies the
    classifier looked; in fact it has no low-dimensional branch, and the
    distinction matters because the unknown profile then selects integer
    occupations, which a semimetallic sheet like graphene cannot converge
    on. Naming a limitation instead of dressing it as a confident default
    is CLAUDE.md section 7 applied to the profiler.
    """
    cls = classify_periodic_system(system)

    assert cls.profile == "unknown"
    assert cls.dim == 2
    reason = " ".join(cls.reasons)
    assert "no low-dimensional branch" in reason
    assert "unclassified by construction" in reason
    assert "explicit smearing choice" in reason
    assert "no decisive signal" not in reason


def test_low_dimensional_classification_does_not_change_any_knob():
    """The truthful reason string must be inert on the strategy table.

    Finite-temperature smearing is not implemented on the slab GDF route
    that 2D systems take (it raises ``NotImplementedError``), so a smeared
    2D default would fail closed on every such run. The reason string is
    corrected; the knobs deliberately are not.
    """
    strategy = resolve_convergence_strategy(
        _hbn_sheet(), method="RKS", convergence=None
    )

    assert strategy.classification.profile == "unknown"
    for name in CONVERGENCE_KNOBS:
        assert strategy.knobs[name].value == 0.0
        assert strategy.knobs[name].source == "auto"


def test_three_dimensional_inconclusive_wording_is_unchanged():
    """A genuinely inconclusive 3D cell keeps the original message."""
    lattice = np.diag([14.0, 14.0, 14.0]).astype(float)
    loose = vq.PeriodicSystem(
        3, lattice, [vq.Atom(6, [0.0, 0.0, 0.0]), vq.Atom(6, [2.6, 0.0, 0.0])]
    )
    cls = classify_periodic_system(loose)
    if cls.profile == "unknown":
        reason = " ".join(cls.reasons)
        assert "no decisive signal" in reason
        assert "no low-dimensional branch" not in reason
