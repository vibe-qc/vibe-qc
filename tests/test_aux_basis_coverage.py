"""Auxiliary-basis element coverage -- density fitting must refuse, not drop.

GitLab #480. The shipped Dunning fitting families do not cover every
element they are auto-selected for:

  * ``cc-pV{D,T,Q,5}Z-JKFIT`` ship no He, Li, Be, Na or Mg block, and
    stop before the 3d row (no K, Ca, Sc..Zn);
  * ``cc-pVnZ-RI`` / ``-RIFIT`` stop before K;
  * ``def2-qzvp-rifit`` skips Sc..Zn, Y..Cd, La and Hf..Hg (BSE's own
    42-element record; the def2-qzvp orbital basis therefore auto-resolves
    to ``def2-qzvpp-rifit`` -- GitLab #483);
  * the ``cc-pwCVnZ-RIFIT`` / ``aug-cc-pwCVnZ-RIFIT`` core-correlation
    families skip H, He, Li, Be, Na and Mg.

libint2 returns an auxiliary ``BasisSet`` with **zero shells on the
uncovered centre** rather than raising, so before this guard the SCF
converged cleanly -- tight gradient, no warning anywhere in the ``.out``
-- to a grossly over-bound energy. Measured on the pre-fix tree:

    LiH  / cc-pVQZ   -1.15 Ha
    BeH2 / cc-pVQZ   -1.20 Ha
    NaH  / cc-pVDZ  -49.56 Ha
    MgH2 / cc-pVDZ  -45.43 Ha

A -49.56 Ha converged answer is ~31,000 kcal/mol, and the tell is that
NaH's auxiliary basis has the *same* function count as its orbital basis
(23/23): the Na centre contributes nothing at all.

The contrast that defines the defect: a *completely* absent auxiliary
basis already fails loudly (closed #11 -- CCSD(T)/STO-3G). A *partially*
covering one failed silently. These tests pin the asymmetry closed on
both density-fitting routes:

  1. the SCF JK route (``density_fit=True``, ``kind="jk"``), and
  2. the post-SCF correlation route (``DensityFitting`` /
     DF-MP2 / DF-CC / DLPNO, ``kind="ri"``) -- reached with no
     ``density_fit`` flag at all by the DLPNO methods, which are
     density-fitted by construction.

Both public SCF entry points are pinned separately, ``run_rhf`` and
``run_job``. ``vibeqc.runner`` is imported from ``vibeqc/__init__.py``
*before* the Python SCF wrappers are defined, so it binds the **bare
C++ entry points** (``vibeqc.runner.run_rhf is vibeqc._run_rhf_cxx``).
A guard placed only in the wrappers therefore passes a ``run_rhf``
regression while leaving ``run_job`` -- the entry point the issue was
filed against -- silently wrong.

Refusal, not substitution, is the contract: silently swapping in a
covering auxiliary would change the method under a user who named a
specific one. The error message points at ``def2-universal-jkfit``
(JK) / ``def2-tzvpp-rifit`` (RI), which do cover H-Rn, and the control
tests below pin that those recover the conventional answer to ~1e-5 Ha.

The guard is one-directional: it may only *add* the partial-coverage
refusal, never change which error a caller already saw. An auxiliary
basis that cannot be constructed at all keeps raising its own error.
"""

from __future__ import annotations

import pytest

from vibeqc import (
    Atom,
    BasisSet,
    Molecule,
    RHFOptions,
    default_aux_basis_for,
    run_rhf,
)
from vibeqc.density_fitting import (
    AuxiliaryBasisCoverageError,
    DensityFitting,
    aux_basis_coverage_gaps,
)

# Tier / maturity markers come from scripts/test_gate/suite_manifest.json
# via tests/conftest.py; this file is a CURATED T1 row there.
pytestmark = [pytest.mark.molecular]


ANGSTROM_TO_BOHR = 1.8897261254578281


def _ang(value: float) -> float:
    return value * ANGSTROM_TO_BOHR


# Geometries in bohr. Experimental r_e where available; the exact bond
# length is immaterial to the coverage question, but pinning it keeps the
# conventional reference energies below reproducible.
GEOMETRIES = {
    # LiH r_e = 1.5949 A (Huber & Herzberg).
    "LiH": [(3, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, _ang(1.5949)])],
    # BeH2, linear D_inf_h, r(Be-H) = 1.3264 A.
    "BeH2": [
        (4, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.0, _ang(1.3264)]),
        (1, [0.0, 0.0, -_ang(1.3264)]),
    ],
    # NaH r_e = 1.8873 A (Huber & Herzberg).
    "NaH": [(11, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, _ang(1.8873)])],
    # MgH2, linear D_inf_h, r(Mg-H) = 1.700 A.
    "MgH2": [
        (12, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.0, _ang(1.7)]),
        (1, [0.0, 0.0, -_ang(1.7)]),
    ],
    # Covered-element controls.
    "H2O": [
        (8, [0.0, 0.0, 0.0]),
        (1, [0.0, _ang(0.793353), -_ang(0.613510)]),
        (1, [0.0, -_ang(0.793353), -_ang(0.613510)]),
    ],
    # CaH2 -- covered by cc-pVDZ but *not* by cc-pvdz-ri / cc-pvdz-jkfit,
    # which both stop before K. Exercises the post-SCF RI route.
    "CaH2": [
        (20, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.0, _ang(2.0)]),
        (1, [0.0, 0.0, -_ang(2.0)]),
    ],
}


# The four systems from #480: (name, orbital basis, uncovered element
# symbol, conventional RHF energy in Ha). The conventional energies are
# the *covered* reference -- they use no density fitting, so this guard
# must leave them untouched.
UNCOVERED_CASES = [
    ("LiH", "cc-pvqz", "Li", -7.98717679),
    ("BeH2", "cc-pvqz", "Be", -15.77298272),
    ("NaH", "cc-pvdz", "Na", -162.38394862),
    ("MgH2", "cc-pvdz", "Mg", -200.72938787),
]


def _molecule(name: str) -> Molecule:
    return Molecule([Atom(Z, list(xyz)) for Z, xyz in GEOMETRIES[name]])


def _rhf(name: str, basis_name: str, *, density_fit: bool, aux: str = ""):
    mol = _molecule(name)
    basis = BasisSet(mol, basis_name)
    opts = RHFOptions()
    opts.conv_tol_energy = 1e-12
    opts.conv_tol_grad = 1e-10
    opts.density_fit = density_fit
    opts.aux_basis = aux
    return run_rhf(mol, basis, opts)


# -----------------------------------------------------------------------------
# 1. The defect: DF-SCF on an uncovered element.
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,basis_name,element,e_conventional",
    UNCOVERED_CASES,
    ids=[c[0] for c in UNCOVERED_CASES],
)
def test_df_scf_never_silently_disagrees(
    name, basis_name, element, e_conventional
):
    """DF-SCF must refuse an uncovered element -- or match conventional.

    Formulated as the invariant rather than as ``pytest.raises`` so that
    the pre-fix failure reads as the filed defect: on the unguarded tree
    the run *converges* and lands 1.15-49.56 Ha below the conventional
    answer, and this assertion reports that magnitude. It is not an
    import or resolution error.
    """
    reference = _rhf(name, basis_name, density_fit=False)
    assert reference.converged
    assert reference.energy == pytest.approx(e_conventional, abs=1e-7)

    aux_name = default_aux_basis_for(basis_name, kind="jk")

    try:
        result = _rhf(name, basis_name, density_fit=True, aux=aux_name)
    except AuxiliaryBasisCoverageError as exc:
        message = str(exc)
        assert element in message, (
            f"coverage refusal must name the uncovered element {element!r}; "
            f"got: {message}"
        )
        assert aux_name in message, (
            f"coverage refusal must name the auxiliary basis {aux_name!r}; "
            f"got: {message}"
        )
        return

    delta = result.energy - reference.energy
    pytest.fail(
        f"{name}/{basis_name}: DF-SCF with auto-resolved aux {aux_name!r} "
        f"neither refused nor agreed with the conventional SCF. It "
        f"converged={result.converged} to {result.energy:.10f} Ha against "
        f"{reference.energy:.10f} Ha conventional -- a silent error of "
        f"{delta:+.4f} Ha. The auxiliary basis has no functions on "
        f"{element}, so that centre is omitted from the fit (#480)."
    )


@pytest.mark.parametrize(
    "name,basis_name,element,e_conventional",
    UNCOVERED_CASES,
    ids=[c[0] for c in UNCOVERED_CASES],
)
def test_run_job_density_fit_refuses_uncovered_element(
    name, basis_name, element, e_conventional, tmp_path
):
    """``run_job(..., density_fit=True)`` -- the filed envelope.

    ``runner`` binds the *bare* C++ SCF entry points: it is imported from
    ``vibeqc/__init__.py`` before the Python wrappers are defined, so the
    guard inside ``run_rhf`` and friends never runs on this path. Pinned
    separately from the ``run_rhf`` case because a guard placed only in
    the wrappers passes that test while leaving ``run_job`` -- the entry
    point the issue was filed against -- silently wrong.
    """
    from vibeqc import run_job

    mol = _molecule(name)
    with pytest.raises(AuxiliaryBasisCoverageError) as excinfo:
        run_job(
            mol,
            method="rhf",
            basis=basis_name,
            density_fit=True,
            output=str(tmp_path / name.lower()),
        )
    message = str(excinfo.value)
    assert element in message
    assert "jkfit" in message


def test_run_job_without_density_fit_is_unaffected(tmp_path):
    """The default path stays open: no density fitting, no refusal."""
    from vibeqc import run_job

    result = run_job(
        _molecule("NaH"),
        method="rhf",
        basis="cc-pvdz",
        output=str(tmp_path / "nah_conventional"),
    )
    assert result.energy == pytest.approx(-162.38394862, abs=1e-6)


@pytest.mark.parametrize(
    "name,basis_name,element,e_conventional",
    UNCOVERED_CASES,
    ids=[c[0] for c in UNCOVERED_CASES],
)
def test_covering_aux_control_still_recovers_the_answer(
    name, basis_name, element, e_conventional
):
    """Positive control: a covering aux must keep working, unchanged.

    ``def2-universal-jkfit`` covers H-Rn. The DF machinery itself is
    sound -- this is what makes #480 a coverage defect and not a
    density-fitting defect -- so the guard must not disturb it.
    """
    result = _rhf(
        name, basis_name, density_fit=True, aux="def2-universal-jkfit"
    )
    assert result.converged
    assert result.energy == pytest.approx(e_conventional, abs=1e-4)


# -----------------------------------------------------------------------------
# 2. Covered systems must be completely unaffected.
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("basis_name", ["cc-pvdz", "def2-svp"])
def test_covered_system_df_scf_unchanged(basis_name):
    """H2O -- every element covered -- keeps running with the auto aux."""
    reference = _rhf("H2O", basis_name, density_fit=False)
    aux_name = default_aux_basis_for(basis_name, kind="jk")
    result = _rhf("H2O", basis_name, density_fit=True, aux=aux_name)
    assert result.converged
    assert result.energy == pytest.approx(reference.energy, abs=1e-4)


def test_covered_system_post_scf_df_unchanged():
    """A covered molecule still builds a DensityFitting on the RI route."""
    mol = _molecule("H2O")
    basis = BasisSet(mol, "cc-pvdz")
    aux_name = default_aux_basis_for("cc-pvdz", kind="ri")
    aux = BasisSet(mol, aux_name, require_all_atoms=False)
    df = DensityFitting(basis, aux, aux_basis_name=aux_name, molecule=mol)
    assert df.n_aux > df.n_orb


# -----------------------------------------------------------------------------
# 3. The post-SCF (RI) route -- reached by DLPNO with no density_fit flag.
# -----------------------------------------------------------------------------

def test_post_scf_ri_route_refuses_uncovered_element():
    """CaH2 + cc-pvdz-ri: the RI families stop before K.

    This is the route the DLPNO methods and ``run_rohf_mp2`` take by
    construction -- no ``density_fit=True`` is needed to reach it -- so
    the same silent drop was available on a default-configured run.
    """
    mol = _molecule("CaH2")
    basis = BasisSet(mol, "cc-pvdz")
    aux_name = default_aux_basis_for("cc-pvdz", kind="ri")
    aux = BasisSet(mol, aux_name, require_all_atoms=False)

    with pytest.raises(AuxiliaryBasisCoverageError) as excinfo:
        DensityFitting(basis, aux, aux_basis_name=aux_name, molecule=mol)

    message = str(excinfo.value)
    assert "Ca" in message
    assert aux_name in message


def test_post_scf_ri_route_refuses_without_a_molecule():
    """The guard still fires when the caller passes no Molecule.

    ``DensityFitting`` is constructed from two ``BasisSet`` objects in
    several places. Without a molecule the element symbol is
    unavailable, but "an atom carrying orbital functions carries no
    auxiliary functions" is decidable from the two bases alone, so the
    refusal must still happen -- reported by atom index.
    """
    mol = _molecule("CaH2")
    basis = BasisSet(mol, "cc-pvdz")
    aux = BasisSet(mol, "cc-pvdz-ri", require_all_atoms=False)

    with pytest.raises(AuxiliaryBasisCoverageError) as excinfo:
        DensityFitting(basis, aux, aux_basis_name="cc-pvdz-ri")

    assert "cc-pvdz-ri" in str(excinfo.value)


# -----------------------------------------------------------------------------
# 4. Diagnostics: consistent with #11's completely-absent case.
# -----------------------------------------------------------------------------

def test_completely_absent_aux_still_fails_loudly():
    """#11's path is unchanged: no registered aux -> NotImplementedError."""
    with pytest.raises(NotImplementedError) as excinfo:
        default_aux_basis_for("sto-3g", kind="ri")
    assert "sto-3g" in str(excinfo.value)


def test_guard_does_not_preempt_an_unbuildable_aux_basis():
    """The guard may only *add* a refusal, never reorder existing ones.

    The SCF guard builds a throwaway auxiliary basis to inspect it. If
    that construction fails -- a misspelt name, or one no ``.g94``
    matches -- it must give up silently: that case is already loud
    downstream (#11), and the driver may have a more specific objection
    that should reach the caller first. Pinned as a general ordering
    invariant, not for one call site: the range-separated-hybrid
    rejection in ``tests/test_xc.py`` originally motivated it, but that
    test now names a real auxiliary basis, so this test is the only
    thing holding the invariant.
    """
    mol = _molecule("H2O")
    basis = BasisSet(mol, "cc-pvdz")
    opts = RHFOptions()
    opts.density_fit = True
    opts.aux_basis = "not-a-real-aux-basis-name"

    with pytest.raises(Exception) as excinfo:
        run_rhf(mol, basis, opts)

    assert not isinstance(excinfo.value, AuxiliaryBasisCoverageError), (
        "an unbuildable auxiliary basis must keep raising its own error, "
        f"not the coverage refusal; got: {excinfo.value}"
    )


def test_refusal_message_offers_a_covering_alternative():
    """The message must be actionable, like #11's."""
    mol = _molecule("NaH")
    basis = BasisSet(mol, "cc-pvdz")
    aux = BasisSet(mol, "cc-pvdz-jkfit", require_all_atoms=False)

    with pytest.raises(AuxiliaryBasisCoverageError) as excinfo:
        DensityFitting(basis, aux, aux_basis_name="cc-pvdz-jkfit", molecule=mol)

    message = str(excinfo.value)
    assert "Na" in message
    assert "cc-pvdz-jkfit" in message
    # Names a covering family the user can switch to, and the opt-out.
    assert "def2-universal-jkfit" in message
    assert "density_fit" in message


# -----------------------------------------------------------------------------
# 5. The coverage predicate itself.
# -----------------------------------------------------------------------------

def test_coverage_gaps_reports_symbol_and_index():
    mol = _molecule("NaH")
    aux = BasisSet(mol, "cc-pvdz-jkfit", require_all_atoms=False)
    gaps = aux_basis_coverage_gaps(aux, molecule=mol)
    assert [(g.atom_index, g.symbol) for g in gaps] == [(0, "Na")]


def test_coverage_gaps_empty_for_a_covering_aux():
    mol = _molecule("NaH")
    aux = BasisSet(mol, "def2-universal-jkfit", require_all_atoms=False)
    assert aux_basis_coverage_gaps(aux, molecule=mol) == []


def test_coverage_gaps_ignores_atoms_the_orbital_basis_also_skips():
    """An atom with no orbital functions contributes no pair density.

    Such a centre needs no auxiliary functions, so it must not trip the
    guard -- this is what keeps the check from firing on covered
    systems that carry a dummy or ghost centre.
    """
    mol = _molecule("H2O")
    basis = BasisSet(mol, "cc-pvdz")
    aux = BasisSet(mol, "cc-pvdz-jkfit", require_all_atoms=False)
    assert aux_basis_coverage_gaps(aux, orbital_basis=basis) == []


# -----------------------------------------------------------------------------
# 6. Shipped-data audit pin.
# -----------------------------------------------------------------------------

# Element blocks the shipped auxiliary files are known to lack, pinned so
# that a future backfill (or a regression in the fetch scripts) shows up
# as a named failure rather than as a silent change in which molecules
# are refused. Verified by direct construction, not by parsing .g94.
#
# The def2-qzvp-rifit row was filed as GitLab #483 on the hypothesis that
# its 42-of-72 element count was a truncated BSE fetch. It is not: BSE's
# ``def2-QZVP-RIFIT`` version 1 carries exactly those 42 main-group elements
# (H..Ca, Ga..Sr, In..Ba, Tl..Rn) -- the transition metals are absent at the
# source (Turbomole 7.3 data), and the record is a byte-identical subset of
# ``def2-qzvpp-rifit``. The file is kept as fetched, so the gap stays pinned
# here as *detected*; what changed is that ``default_aux_basis_for`` no
# longer routes a def2-qzvp orbital basis to it (see
# ``test_def2_qzvp_auto_resolves_to_the_covering_qzvpp_fit`` below).
KNOWN_GAPS = [
    # (aux basis, Z, symbol)  -- auto-selected JK families
    ("cc-pvdz-jkfit", 3, "Li"),
    ("cc-pvdz-jkfit", 11, "Na"),
    ("cc-pvtz-jkfit", 4, "Be"),
    ("cc-pvqz-jkfit", 12, "Mg"),
    ("cc-pv5z-jkfit", 3, "Li"),
    # auto-selected RI families -- the post-SCF route
    ("cc-pvdz-ri", 20, "Ca"),
    ("cc-pvtz-ri", 21, "Sc"),
    ("cc-pvqz-ri", 30, "Zn"),
    ("def2-qzvp-rifit", 21, "Sc"),
    # core-correlation RI families
    ("cc-pwcvtz-rifit", 1, "H"),
    ("aug-cc-pwcvtz-rifit", 11, "Na"),
]


#: Partner element for the single-gap probes below. Carbon is present in
#: every family audited here, so the probe molecule always loads at least
#: one auxiliary shell -- which keeps the *partial*-coverage path under
#: test. An all-absent auxiliary takes #11's path instead: libint2
#: returns an empty BasisSet and the C++ ctor rejects it outright.
_PARTNER_Z = 6


def _probe_molecule(Z: int) -> Molecule:
    """A two-atom probe carrying ``Z`` and a covered partner element."""
    n_electrons = Z + _PARTNER_Z
    multiplicity = 1 if n_electrons % 2 == 0 else 2
    return Molecule(
        [Atom(Z, [0.0, 0.0, 0.0]), Atom(_PARTNER_Z, [0.0, 0.0, 4.0])],
        0,
        multiplicity,
    )


@pytest.mark.parametrize(
    "aux_name,Z,symbol",
    KNOWN_GAPS,
    ids=[f"{a}-{s}" for a, _, s in KNOWN_GAPS],
)
def test_known_shipped_aux_gaps_are_detected(aux_name, Z, symbol):
    """Each known gap must be *detected*, so the guard refuses the run."""
    mol = _probe_molecule(Z)
    aux = BasisSet(mol, aux_name, require_all_atoms=False)
    gaps = aux_basis_coverage_gaps(aux, molecule=mol)
    assert symbol in {g.symbol for g in gaps}, (
        f"{aux_name} was expected to lack {symbol}; if it was backfilled, "
        f"drop this row from KNOWN_GAPS"
    )


DEF2_COVERING = ["def2-universal-jkfit", "def2-universal-jfit"]


@pytest.mark.parametrize("aux_name", DEF2_COVERING)
@pytest.mark.parametrize("Z", [1, 2, 3, 4, 11, 12, 20, 21, 26, 30])
def test_def2_universal_families_cover_the_gapped_elements(aux_name, Z):
    """The recommended fallbacks must actually cover what they promise."""
    mol = _probe_molecule(Z)
    aux = BasisSet(mol, aux_name, require_all_atoms=False)
    assert aux_basis_coverage_gaps(aux, molecule=mol) == []


# ---------------------------------------------------------------------------
# GitLab #483 -- def2-qzvp must not be paired with the 42-element record.
# ---------------------------------------------------------------------------

#: 3d / 4d / 5d probes: every one is absent from ``def2-qzvp-rifit`` and
#: present in ``def2-qzvpp-rifit``.
_TRANSITION_METAL_PROBES = [21, 26, 30, 39, 48, 72, 80]


def test_def2_qzvp_auto_resolves_to_the_covering_qzvpp_fit():
    """The RI default for def2-qzvp is the 72-element QZVPP fit (#483)."""
    from vibeqc.density_fitting import default_aux_basis_for

    assert default_aux_basis_for("def2-qzvp", kind="ri") == "def2-qzvpp-rifit"
    assert default_aux_basis_for("def2-QZVP", kind="ri") == "def2-qzvpp-rifit"


@pytest.mark.parametrize("Z", _TRANSITION_METAL_PROBES)
def test_def2_qzvp_ri_default_covers_the_transition_metals(Z):
    """What def2-qzvp auto-resolves to covers the metals the record lacks."""
    from vibeqc.density_fitting import default_aux_basis_for

    mol = _probe_molecule(Z)
    aux = BasisSet(mol, default_aux_basis_for("def2-qzvp", kind="ri"), require_all_atoms=False)
    assert aux_basis_coverage_gaps(aux, molecule=mol) == []
    # ... and the by-name record really does lack it (the #483 gap).
    named = BasisSet(mol, "def2-qzvp-rifit", require_all_atoms=False)
    assert Z in {g.Z for g in aux_basis_coverage_gaps(named, molecule=mol)}


@pytest.mark.parametrize("Z", [1, 6, 8, 20, 35, 53, 86])
def test_def2_qzvp_rifit_is_a_subset_of_the_qzvpp_fit(Z):
    """On every element it ships, def2-qzvp-rifit is def2-qzvpp-rifit.

    This is what makes the re-route in ``_DEF2_RI`` a no-op for main-group
    molecules: the two files carry identical shells there, so the fitted
    numbers are bit-identical. Compared at the libint level (shell count
    and function count), not by parsing the .g94 files.
    """
    mol = _probe_molecule(Z)
    a = BasisSet(mol, "def2-qzvp-rifit", require_all_atoms=False)
    b = BasisSet(mol, "def2-qzvpp-rifit", require_all_atoms=False)
    assert a.nbasis == b.nbasis
    assert a.nshells == b.nshells
