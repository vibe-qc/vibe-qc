"""``VibeQcEngine`` periodic wiring and fail-closed relaxation boundary.

vibe-qc is the **primary** engine of the basis-optimization roadmap
(``vibe-basis/ROADMAP.md`` § 2), so this is the implementation every
milestone ultimately runs on. What this file pins, in the spirit of
``vibe-basis/tests/test_pipeline.py``: the marshalling, the failure
contract, and the conventions -- deterministically, with the SCF stubbed
out -- plus one real Gamma-point run to prove the wiring actually
reaches the native core.

It does **not** pin any physical number. Agreement with CRYSTAL23 on a
cohesive energy is M0's acceptance gate; it needs converged settings and
reference geometries, and a wiring test that asserted an energy would
quietly become a regression lock on a number nobody has certified.

The failure contract is the part worth stating plainly. A basis
optimizer walks into infeasible regions constantly -- an exponent
collapses, the overlap goes singular, the SCF stalls. Every one of those
must come back as ``ok=False`` with a specific ``failure_mode`` so a
derivative-free driver can score the point ``inf`` and continue. An
exception would abort a campaign that is days long, and returning the
last-cycle energy of a stalled SCF would poison the objective with a
number that looks converged.
"""

from __future__ import annotations

import math

import pytest

pytest.importorskip("vibe_basis")

from vibe_basis.engine import EngineEnergy  # noqa: E402
from vibe_basis.io.structures import Structure, StructureAtom  # noqa: E402

from vibeqc.basis_crystal import (  # noqa: E402
    emit_crystal,
    parse_crystal_atom_basis,
    parse_crystal_inline_basis,
)
from vibeqc.basis_optimization.calculators import (  # noqa: E402
    VibeQcEngine,
    _parse_inline_basis_atoms,
    _parse_inline_basis_text,
    _resolve_method,
    _structure_from_periodic_system,
    _to_periodic_system,
)

# ---------------------------------------------------------------------------
# Fixtures: a two-element candidate basis and a rocksalt cell
# ---------------------------------------------------------------------------
#
# STO-3G exponents/coefficients for Mg and O (Hehre, Stewart & Pople,
# J. Chem. Phys. 51, 2657 (1969)). Small on purpose: this file tests
# wiring, and every second here is paid on every run of the suite.

STO3G_MG = """12 3
0 0 3 2.0 1.0
 299.2374000  0.15432897
  54.5064700  0.53532814
  14.7515800  0.44463454
0 1 3 8.0 1.0
  15.1218200 -0.09996723  0.15591627
   3.5139870  0.39951283  0.60768372
   1.1428570  0.70011547  0.39195739
0 1 3 2.0 1.0
   1.3954480 -0.21962037  0.01058760
   0.3893260  0.22559543  0.59516701
   0.1523800  0.90039843  0.46200101
"""

STO3G_O = """8 2
0 0 3 2.0 1.0
 130.7093200  0.15432897
  23.8088610  0.53532814
   6.4436083  0.44463454
0 1 3 6.0 1.0
   5.0331513 -0.09996723  0.15591627
   1.1695961  0.39951283  0.60768372
   0.3803890  0.70011547  0.39195739
"""


@pytest.fixture(scope="module")
def basis_text() -> str:
    """A two-element candidate basis, in the engines' interchange format."""
    return emit_crystal(
        [parse_crystal_atom_basis(STO3G_MG), parse_crystal_atom_basis(STO3G_O)]
    )


def rocksalt(name: str = "MgO", a: float = 4.21, **kw) -> Structure:
    """A one-formula-unit cubic MgO stand-in. Not a reference geometry.

    GitLab #128: the space group here is ``Pm-3m`` (221), not rocksalt's
    ``Fm-3m`` (225). One cation at the origin and one anion at the body centre
    of a cubic cell is the CsCl arrangement; rocksalt needs four formula units
    in the conventional cell, and its primitive cell is rhombohedral. The
    label was wrong and the marshalling check now catches it. The geometry is
    deliberately unchanged, so no energy in this file moves -- this fixture
    exercises wiring, not physics.
    """
    return Structure(
        name=name,
        formula=name,
        spacegroup="Pm-3m",
        crystal_system="cubic",
        a=a,
        b=a,
        c=a,
        unit_cell=(
            StructureAtom(Z=12, fxyz=(0.0, 0.0, 0.0)),
            StructureAtom(Z=8, fxyz=(0.5, 0.5, 0.5)),
        ),
        crystal_spacegroup=221,
        crystal_asymm_unit=(
            StructureAtom(Z=12, fxyz=(0.0, 0.0, 0.0)),
            StructureAtom(Z=8, fxyz=(0.5, 0.5, 0.5)),
        ),
        **kw,
    )


class _FakeResult:
    """Stand-in for a periodic SCF / optimizer result."""

    def __init__(self, energy, converged, system=None, n_iter=7):
        self.energy = energy
        self.converged = converged
        self.system = system
        self.n_iter = n_iter


# ---------------------------------------------------------------------------
# Basis marshalling
# ---------------------------------------------------------------------------


def test_inline_basis_splits_into_one_block_per_element(basis_text):
    atoms = _parse_inline_basis_atoms(basis_text)
    assert [a.Z for a in atoms] == [12, 8]
    assert [len(a.shells) for a in atoms] == [3, 2]


def test_atom_block_is_selected_by_Z_not_by_position(basis_text):
    """The regression that matters: O must not be handed the Mg basis.

    The engine used to parse the *first* block of ``basis_text`` and
    stamp the requested ``Z`` on it. For MgO that gives oxygen
    magnesium's exponents -- an SCF that converges happily to a
    thoroughly wrong number, with nothing in the output to say so.
    """
    o_block = _parse_inline_basis_text(basis_text, 8)
    assert o_block.Z == 8
    # 130.709 is oxygen's tightest s primitive; magnesium's is 299.237.
    assert o_block.shells[0].exponents[0] == pytest.approx(130.7093200)

    mg_block = _parse_inline_basis_text(basis_text, 12)
    assert mg_block.shells[0].exponents[0] == pytest.approx(299.2374000)


def test_missing_element_is_an_error_not_a_silent_substitution(basis_text):
    with pytest.raises(KeyError):
        _parse_inline_basis_text(basis_text, 3)  # no Li block


def test_emit_crystal_round_trips_through_the_inline_parser(basis_text):
    """``emit_crystal`` -> ``parse_crystal_inline_basis`` is lossless.

    It was not: ``emit_crystal`` writes ``SCAL = 0.0`` and the parser
    read that as a literal scale factor, multiplying every exponent by
    ``0.0**2``. Per the CRYSTAL23 manual ("Basis set input", p. 25) SCAL
    is meaningful only for the standard Pople sets (``ITYB=1``); for a
    general user basis both 0.0 and 1.0 mean unscaled.
    """
    reparsed = parse_crystal_inline_basis(basis_text)
    original = [parse_crystal_atom_basis(STO3G_MG), parse_crystal_atom_basis(STO3G_O)]
    for want, got in zip(original, reparsed):
        assert want.Z == got.Z
        for s_want, s_got in zip(want.shells, got.shells):
            assert s_want.shell_type == s_got.shell_type
            assert s_got.exponents == pytest.approx(s_want.exponents)
            assert s_got.coefficients == pytest.approx(s_want.coefficients, rel=1e-6)


def test_inline_parser_rejects_a_duplicated_element():
    doubled = emit_crystal(
        [parse_crystal_atom_basis(STO3G_O), parse_crystal_atom_basis(STO3G_O)]
    )
    with pytest.raises(ValueError, match="more than once"):
        parse_crystal_inline_basis(doubled)


def test_inline_parser_rejects_an_inline_ecp():
    """Rejected rather than skipped -- see the function's docstring."""
    text = "208 1\nINPUT\n8.0 1 0 0 0 0 0\n1.0 1.0 0\n"
    with pytest.raises(ValueError, match="INPUT"):
        parse_crystal_inline_basis(text)


# ---------------------------------------------------------------------------
# Geometry marshalling
# ---------------------------------------------------------------------------


def test_structure_survives_a_round_trip_through_periodic_system():
    struct = rocksalt()
    system = _to_periodic_system(struct)
    back = _structure_from_periodic_system(system, struct)

    assert back.a == pytest.approx(struct.a)
    assert back.b == pytest.approx(struct.b)
    assert back.c == pytest.approx(struct.c)
    assert back.alpha == pytest.approx(90.0)
    assert back.gamma == pytest.approx(90.0)
    assert [s.Z for s in back.unit_cell] == [12, 8]
    for want, got in zip(struct.unit_cell, back.unit_cell):
        assert got.fxyz == pytest.approx(want.fxyz, abs=1e-10)


def test_relaxed_structure_clears_the_stale_symmetry_fields():
    """A relaxation invalidates the asymmetric unit; it must not survive.

    ``emit_input_inline`` builds a CRYSTAL deck from
    ``crystal_asymm_unit`` + ``crystal_spacegroup``. Carrying those over
    from the *input* geometry would let a downstream engine compute a
    different crystal than the one that was relaxed. Cleared, that
    emitter returns ``None`` and the caller sees an honest failure.
    """
    struct = rocksalt()
    assert struct.crystal_spacegroup == 221  # the input has them

    back = _structure_from_periodic_system(_to_periodic_system(struct), struct)
    assert back.crystal_spacegroup == 0
    assert back.crystal_asymm_unit == ()
    assert "relaxed" in back.notes


def test_periodic_system_is_passed_through_unchanged():
    struct = rocksalt()
    system = _to_periodic_system(struct)
    assert _to_periodic_system(system) is system


def test_a_non_structure_is_a_type_error():
    with pytest.raises(TypeError):
        _to_periodic_system(object())


# ---------------------------------------------------------------------------
# Method resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, multiplicity, expected",
    [
        ("rhf", 1, ("RHF", None)),
        ("hf", 1, ("RHF", None)),
        ("rhf", 3, ("UHF", None)),
        ("pbe", 1, ("RKS", "pbe")),
        ("r2scan", 3, ("UKS", "r2scan")),
        ("PW1PW", 1, ("RKS", "PW1PW")),
    ],
)
def test_method_string_maps_to_solver_and_functional(method, multiplicity, expected):
    """Spin comes from the multiplicity, the functional from the string.

    So a caller writes ``"r2scan"`` and never has to know whether the
    system wants RKS or UKS -- which is what lets one ``method=`` value
    describe a whole campaign of open- and closed-shell systems.
    """
    assert _resolve_method(method, multiplicity) == expected


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


def test_engine_declares_zero_point_but_not_relax_or_counterpoise():
    """Only currently executable optional stages are advertised.

    Variable-cell BIPOLE optimization is fail-closed, and the engine has no
    ghost-basis atom path. The pipeline must record both stages as skipped.
    """
    engine = VibeQcEngine()
    assert engine.supports("relax") is False
    assert engine.supports("zero_point") is True
    assert engine.supports("counterpoise") is False


def test_engine_reports_the_installed_vibeqc_version():
    import vibeqc as vq

    assert VibeQcEngine().version() == vq.__version__


# ---------------------------------------------------------------------------
# crystal_energy: the failure contract
# ---------------------------------------------------------------------------


def test_crystal_energy_refuses_a_magnetic_ordering_it_cannot_express(basis_text):
    engine = VibeQcEngine()
    result = engine.crystal_energy(basis_text, rocksalt(afm_pattern="AF2 (111)"))
    assert result.ok is False
    assert result.failure_mode == "unsupported_magnetic_structure"


def test_crystal_energy_refuses_an_ecp_structure(basis_text):
    engine = VibeQcEngine()
    result = engine.crystal_energy(basis_text, rocksalt(ecp_library="pob-tzvp-rev2"))
    assert result.ok is False
    assert result.failure_mode == "unsupported_ecp"


def test_crystal_energy_refuses_a_basis_missing_an_element(basis_text):
    """Fail closed. libint would otherwise pick some other basis by name."""
    only_o = emit_crystal([parse_crystal_atom_basis(STO3G_O)])
    result = VibeQcEngine().crystal_energy(only_o, rocksalt())
    assert result.ok is False
    assert result.failure_mode == "basis_missing_element"
    assert result.detail["missing"] == [12]


def test_crystal_energy_reports_malformed_basis_text():
    result = VibeQcEngine().crystal_energy("not a basis", rocksalt())
    assert result.ok is False
    assert result.failure_mode == "bad_input"


def test_crystal_energy_reports_non_convergence_and_withholds_the_energy(
    monkeypatch, basis_text
):
    """The last cycle of a stalled SCF is diagnostic, never a result."""
    import vibeqc.periodic_runner as pr

    monkeypatch.setattr(
        pr, "run_periodic_job", lambda *a, **k: _FakeResult(-270.5, converged=False)
    )
    result = VibeQcEngine().crystal_energy(basis_text, rocksalt())

    assert result.ok is False
    assert result.failure_mode == "non_converged"
    assert result.energy == pytest.approx(-270.5)   # kept for diagnosis
    assert result.energy_if_ok() is None            # never handed out


def test_crystal_energy_reports_an_scf_exception_rather_than_raising(
    monkeypatch, basis_text
):
    import vibeqc.periodic_runner as pr

    def boom(*a, **k):
        raise RuntimeError("the integrals fell over")

    monkeypatch.setattr(pr, "run_periodic_job", boom)
    result = VibeQcEngine().crystal_energy(basis_text, rocksalt())

    assert result.ok is False
    assert result.failure_mode == "scf_error"
    assert "fell over" in result.detail["error"]


def test_crystal_energy_carries_its_provenance(monkeypatch, basis_text):
    import vibeqc.periodic_runner as pr

    monkeypatch.setattr(
        pr, "run_periodic_job", lambda *a, **k: _FakeResult(-270.8, converged=True)
    )
    result = VibeQcEngine(kpoints=(2, 2, 2)).crystal_energy(
        basis_text, rocksalt(), method="pbe"
    )

    assert result.ok is True
    assert result.engine == "vibeqc"
    assert result.detail["scf_method"] == "RKS"
    assert result.detail["functional"] == "pbe"
    assert result.detail["system"] == "MgO"


def test_bipole_is_the_default_route_and_is_overridable(monkeypatch, basis_text):
    """A candidate basis has no density-fitting auxiliary basis.

    Borrowing another basis's JKFIT set would lay a fitting error on
    top of the basis-incompleteness signal the objective measures, so
    the engine defaults to the four-centre BIPOLE route -- the same one
    :meth:`relax` drives.
    """
    import vibeqc.periodic_runner as pr

    seen: dict = {}

    def record(*a, **k):
        seen.update(k)
        return _FakeResult(-270.8, converged=True)

    monkeypatch.setattr(pr, "run_periodic_job", record)

    VibeQcEngine().crystal_energy(basis_text, rocksalt())
    assert seen["jk_method"] == "bipole"

    VibeQcEngine(periodic_kwargs={"jk_method": "rijcosx"}).crystal_energy(
        basis_text, rocksalt()
    )
    assert seen["jk_method"] == "rijcosx"


# ---------------------------------------------------------------------------
# relax: unsupported until a certified variable-cell route exists
# ---------------------------------------------------------------------------


def test_direct_relax_request_fails_closed_before_dispatch(
    monkeypatch, basis_text
):
    import vibeqc.bipole_optimize as bo

    called = False

    def unexpected_dispatch(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("relax_full must not be dispatched")

    monkeypatch.setattr(bo, "relax_full", unexpected_dispatch)
    engine = VibeQcEngine(relax_kwargs={"max_iter": 1})

    with pytest.raises(NotImplementedError, match="variable-cell") as exc:
        engine.relax(basis_text, rocksalt())

    assert "relax=False" in str(exc.value)
    assert called is False


def test_pipeline_plans_vibeqc_relaxation_as_skipped(monkeypatch, basis_text):
    from vibe_basis.pipeline import CohesivePipeline

    engine = VibeQcEngine()
    monkeypatch.setattr(
        engine,
        "crystal_energy",
        lambda *args, **kwargs: EngineEnergy(
            energy=-270.0, ok=True, engine=engine.name
        ),
    )
    monkeypatch.setattr(
        engine,
        "atom_energy",
        lambda basis, Z, **kwargs: EngineEnergy(
            energy=-100.0 - float(Z), ok=True, engine=engine.name
        ),
    )

    result = CohesivePipeline(
        engine, method="rhf", relax=True, zero_point=False
    ).run(basis_text, rocksalt())
    stage = next(record for record in result.stages if record.name == "relax")

    assert stage.status == "skipped"
    assert "no relaxer" in stage.reason


# ---------------------------------------------------------------------------
# zero_point_energy (M1)
# ---------------------------------------------------------------------------
#
# The phonon numerics are pinned in `test_gamma_phonons.py` against
# closed-form Hessians. What is pinned here is the *engine contract*:
# units, sign, per-cell convention, provenance, and the failure mode.


def _patch_gamma_phonons(monkeypatch, result_or_exc):
    """Stub the phonon driver at the point the engine imports it from."""
    import vibeqc.basis_optimization.phonons as ph

    def fake(*a, **k):
        if isinstance(result_or_exc, Exception):
            raise result_or_exc
        fake.seen = dict(k)
        fake.args = a
        return result_or_exc

    fake.seen = {}
    fake.args = ()
    monkeypatch.setattr(ph, "gamma_phonons", fake)
    return fake


class _FakePhonons:
    def __init__(self, zpe=0.004, imaginary=0):
        self.zero_point_hartree = zpe
        self.frequencies_cm1 = [0.4, -0.3, 0.9, 400.0, 401.0, 402.0]
        self.acoustic_residual_cm1 = (0.4, -0.3, 0.9)
        self.acoustic_projection = (0.999, 0.998, 0.999)
        self.n_imaginary_modes_excluded = imaginary
        self.n_acoustic_modes_excluded = 3
        self.asr = "none"
        self.hessian_mode = "energy"
        self.fd_step_bohr = 0.02
        self.n_scf = 73


def test_zero_point_energy_is_positive_hartree_per_unit_cell(
    monkeypatch, basis_text
):
    """Sign convention: positive, and the pipeline SUBTRACTS it.

    ``CohesiveResult`` computes ``E_atoms - E_bulk/f.u. - ZPE/f.u.``,
    because a vibrating solid is less bound than a static one. An engine
    returning a negative ZPE would silently flip that.
    """
    _patch_gamma_phonons(monkeypatch, _FakePhonons(zpe=0.004))
    result = VibeQcEngine().zero_point_energy(basis_text, rocksalt())

    assert result.ok is True
    assert result.energy == pytest.approx(0.004)
    assert result.energy > 0.0
    assert result.engine == "vibeqc"


def test_zero_point_energy_records_the_acoustic_residual_and_the_asr_choice(
    monkeypatch, basis_text
):
    """No ASR by default, so the residual means something.

    Silently imposing the sum rule while reporting a residual would make
    the calculation's noise floor unmeasurable.
    """
    _patch_gamma_phonons(monkeypatch, _FakePhonons())
    detail = VibeQcEngine().zero_point_energy(basis_text, rocksalt()).detail

    assert detail["asr"] == "none"
    assert len(detail["acoustic_residual_cm1"]) == 3
    assert min(detail["acoustic_projection"]) > 0.9
    assert detail["approximation"] == "gamma_point_only"
    assert detail["hessian_mode"] == "energy"
    assert detail["fd_step_bohr"] == 0.02


def test_zero_point_energy_reports_imaginary_modes(monkeypatch, basis_text):
    """Imaginary modes say the geometry is not a minimum. Not noise."""
    _patch_gamma_phonons(monkeypatch, _FakePhonons(imaginary=2))
    detail = VibeQcEngine().zero_point_energy(basis_text, rocksalt()).detail
    assert detail["n_imaginary_modes_excluded"] == 2


def test_zero_point_energy_quotes_its_cost_before_running(monkeypatch, basis_text):
    """18 N^2 + 1 grows fast; a campaign has to be able to budget it."""
    _patch_gamma_phonons(monkeypatch, _FakePhonons())
    detail = VibeQcEngine().zero_point_energy(basis_text, rocksalt()).detail
    assert detail["n_scf_estimated"] == 73  # 2-atom primitive cell


def test_zero_point_energy_uses_the_engine_method_kmesh_and_step(
    monkeypatch, basis_text
):
    """The phonon and single-point stages must share method and k-mesh."""
    from vibeqc.periodic_runner import _bloch_kmesh_size

    fake = _patch_gamma_phonons(monkeypatch, _FakePhonons())
    engine = VibeQcEngine(
        kpoints=(2, 2, 2), phonon_kwargs={"step_bohr": 0.03, "asr": "rowsum"}
    )
    engine.zero_point_energy(basis_text, rocksalt(), method="pbe")

    assert fake.seen["method"] == "RKS"
    assert fake.seen["functional"] == "pbe"
    assert fake.seen["step_bohr"] == 0.03
    assert fake.seen["asr"] == "rowsum"
    assert _bloch_kmesh_size(fake.args[2]) == 8


def test_zero_point_energy_reports_a_phonon_failure_rather_than_raising(
    monkeypatch, basis_text
):
    """A failed ZPE downgrades the run; it does not sink a campaign.

    A static-lattice cohesive energy is still a well-defined number, and
    ``zero_point_applied=False`` says which one you got.
    """
    _patch_gamma_phonons(monkeypatch, RuntimeError("displaced SCF stalled"))
    result = VibeQcEngine().zero_point_energy(basis_text, rocksalt())

    assert result.ok is False
    assert result.failure_mode == "phonon_error"
    assert "stalled" in result.detail["error"]
    assert result.energy_if_ok() is None


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"afm_pattern": "AF2 (111)"}, "unsupported_magnetic_structure"),
        ({"ecp_library": "pob-tzvp-rev2"}, "unsupported_ecp"),
    ],
)
def test_zero_point_energy_fails_closed_on_structures_it_cannot_express(
    basis_text, kwargs, expected
):
    result = VibeQcEngine().zero_point_energy(basis_text, rocksalt(**kwargs))
    assert result.ok is False
    assert result.failure_mode == expected


def test_zero_point_energy_refuses_a_basis_missing_an_element(basis_text):
    only_o = emit_crystal([parse_crystal_atom_basis(STO3G_O)])
    result = VibeQcEngine().zero_point_energy(only_o, rocksalt())
    assert result.ok is False
    assert result.failure_mode == "basis_missing_element"


def test_pipeline_applies_the_zero_point_and_it_reduces_the_cohesive_energy(
    monkeypatch, basis_text
):
    """End to end: ``zero_point=True`` now produces an applied correction.

    Before M1 this stage recorded ``skipped -- engine 'vibeqc' has no
    phonons``. The assertion that matters is the *direction*: a ZPE makes
    a solid less bound, so the cohesive energy must go DOWN by exactly
    ZPE per formula unit.
    """
    import vibeqc as vq
    import vibeqc.periodic_runner as pr
    from vibe_basis.pipeline import CohesivePipeline

    monkeypatch.setattr(
        pr, "run_periodic_job", lambda *a, **k: _FakeResult(-270.0, converged=True)
    )
    monkeypatch.setattr(
        vq, "run_rhf", lambda *a, **k: _FakeResult(-100.0, converged=True)
    )
    monkeypatch.setattr(
        vq, "run_uhf", lambda *a, **k: _FakeResult(-74.0, converged=True)
    )
    _patch_gamma_phonons(monkeypatch, _FakePhonons(zpe=0.004))

    engine = VibeQcEngine()
    with_zpe = CohesivePipeline(engine, method="rhf", relax=False).run(
        basis_text, rocksalt()
    )
    without = CohesivePipeline(
        engine, method="rhf", relax=False, zero_point=False
    ).run(basis_text, rocksalt())

    assert with_zpe.ok and without.ok
    assert with_zpe.zero_point_applied is True
    assert without.zero_point_applied is False
    assert with_zpe.formula_units == 1
    assert with_zpe.zero_point == pytest.approx(0.004)
    assert with_zpe.cohesive_hartree == pytest.approx(
        without.cohesive_hartree - 0.004
    )
    stages = {s.name: s.status for s in with_zpe.stages}
    assert stages["zero_point"] == "ok"
    assert "with ZPE" in with_zpe.summary()


# ---------------------------------------------------------------------------
# atom_energy: the reference convention
# ---------------------------------------------------------------------------


def test_atom_multiplicities_come_from_the_shared_ground_state_table(
    monkeypatch, basis_text
):
    """One table, shared with ``vibeqc.atomization``.

    The engine used to carry a hand-written copy that put silicon in a
    *singlet*. Sharing the table means the periodic cohesive energy and
    the molecular atomization energy cannot disagree about what the
    ground state of an element is.
    """
    import vibeqc as vq
    from vibeqc.atomization import _GROUND_STATE_MULTIPLICITY

    assert _GROUND_STATE_MULTIPLICITY[14] == 3  # Si: 3P, not a singlet
    assert _GROUND_STATE_MULTIPLICITY[8] == 3   # O:  3P
    assert _GROUND_STATE_MULTIPLICITY[12] == 1  # Mg: 1S

    seen: dict = {}

    def record(mol, basis, opts):
        seen["multiplicity"] = mol.multiplicity
        return _FakeResult(-74.0, converged=True)

    monkeypatch.setattr(vq, "run_uhf", record)
    result = VibeQcEngine().atom_energy(basis_text, 8)

    assert result.ok is True
    assert seen["multiplicity"] == 3


def test_atom_energy_records_the_reference_convention(monkeypatch, basis_text):
    """Aspherical + spin-polarised, and it says so.

    This has to match whatever reference the cohesive energy is compared
    against: spin-restricting the Br reference alone moved KBr's
    atomization by 37 kJ/mol (``handovers/HANDOVER_GPAW_PW_REFERENCE.md``,
    GPAW-PWREF-002). An engine that did not record its convention could
    not be audited for that.
    """
    import vibeqc as vq

    monkeypatch.setattr(
        vq, "run_uhf", lambda *a, **k: _FakeResult(-74.0, converged=True)
    )
    result = VibeQcEngine().atom_energy(basis_text, 8)
    assert (
        result.detail["atom_reference"]
        == "aspherical_spin_polarised_ground_multiplicity"
    )


def test_atom_energy_refuses_an_element_with_no_tabulated_ground_state(basis_text):
    result = VibeQcEngine().atom_energy(basis_text, 26)  # Fe
    assert result.ok is False
    assert result.failure_mode == "no_ground_state"


def test_atom_energy_reports_non_convergence(monkeypatch, basis_text):
    import vibeqc as vq

    monkeypatch.setattr(
        vq, "run_uhf", lambda *a, **k: _FakeResult(-73.0, converged=False)
    )
    result = VibeQcEngine().atom_energy(basis_text, 8)
    assert result.ok is False
    assert result.failure_mode == "non_converged"
    assert result.energy_if_ok() is None


def test_atom_energy_routes_a_functional_to_the_ks_solver(monkeypatch, basis_text):
    """The DFT atom path used to drop the functional on the floor.

    ``run_rks``/``run_uks`` take it on the options object, not as a
    keyword, so the old ``runner(mol, basis, **scf_kwargs)`` call ran
    every "DFT" atom at the default LDA -- while the bulk ran at the
    requested functional. A cohesive energy assembled from those two is
    wrong by the difference between the functionals.
    """
    import vibeqc as vq

    seen: dict = {}

    def record(mol, basis, opts):
        seen["functional"] = opts.functional
        return _FakeResult(-74.0, converged=True)

    monkeypatch.setattr(vq, "run_uks", record)
    result = VibeQcEngine().atom_energy(basis_text, 8, method="pbe")

    assert result.ok is True
    assert seen["functional"] == "pbe"


# ---------------------------------------------------------------------------
# One real run, to prove the wiring reaches the native core
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.periodic
@pytest.mark.bipole
def test_gamma_point_rhf_runs_end_to_end_through_the_pipeline(basis_text):
    """MgO/STO-3G at Gamma, assembled by ``CohesivePipeline``.

    Deliberately asserts no energy. Gamma-only on a one-formula-unit
    cell with a minimal basis is nowhere near a converged cohesive
    energy, and pinning the number it happens to produce would create a
    regression lock on a value nobody has certified. What is pinned is
    that every stage ran, the arithmetic composed, and the result is
    honest about excluding ZPE.

    The certified numbers are M0's acceptance gate: CRYSTAL23
    reproducing the ``E_pob`` column to < 1 kJ/mol, then vibe-qc
    reproducing CRYSTAL23 to < 2 kJ/mol, at reference geometries and
    converged settings.
    """
    from vibe_basis.pipeline import CohesivePipeline

    engine = VibeQcEngine()
    pipeline = CohesivePipeline(engine, method="rhf", relax=False, zero_point=False)
    result = pipeline.run(basis_text, rocksalt())

    assert result.ok, result.failure_mode
    assert result.engine == "vibeqc"
    assert result.formula_units == 1
    assert set(result.atomic_energies) == {8, 12}

    # ZPE is M1: the number is a static-lattice one and must say so.
    assert result.zero_point_applied is False
    assert "static lattice" in result.summary()

    # The assembly is E_atoms - E_bulk, in kJ/mol per formula unit.
    from vibe_basis.pipeline import HARTREE_TO_KJ_PER_MOL

    expected = result.e_atoms_sum - result.e_bulk
    assert result.cohesive_hartree == pytest.approx(expected)
    assert result.cohesive_kj_per_mol == pytest.approx(
        expected * HARTREE_TO_KJ_PER_MOL
    )
    assert math.isfinite(result.cohesive_kj_per_mol)

    stages = {s.name: s.status for s in result.stages}
    assert stages["single_point"] == "ok"
    assert stages["relax"] == "skipped"
    assert stages["zero_point"] == "skipped"
    assert stages["atom_Z8"] == "ok"
    assert stages["atom_Z12"] == "ok"


@pytest.mark.slow
@pytest.mark.periodic
@pytest.mark.bipole
def test_free_atom_energies_are_shared_across_systems(basis_text):
    """The cache win that makes a campaign affordable.

    Atoms are the expensive stage and the atom set is small and shared,
    so O computed for MgO must be reused for the next system that
    contains O. ``atom_key`` carries no system identifier for exactly
    this reason.
    """
    from vibe_basis.cache import EnergyCache
    from vibe_basis.pipeline import CohesivePipeline

    engine = VibeQcEngine()
    cache = EnergyCache()
    pipeline = CohesivePipeline(
        engine, method="rhf", relax=False, zero_point=False, cache=cache
    )

    first = pipeline.run(basis_text, rocksalt())
    assert first.ok, first.failure_mode

    second = pipeline.run(basis_text, rocksalt(name="MgO-again"))
    assert second.ok, second.failure_mode

    stages = {s.name: s.status for s in second.stages}
    assert stages["atom_Z8"] == "cached"
    assert stages["atom_Z12"] == "cached"
    assert second.atomic_energies == first.atomic_energies


# ---------------------------------------------------------------------------
# GitLab #128: the marshalling transposed the lattice, and the cubic fixture
# above could not see it.
#
# `Structure.lattice_matrix_angstrom` documents "rows = a, b, c vectors";
# `PeriodicSystem.lattice` holds them as COLUMNS. The forward step handed the
# row matrix over unchanged and the reverse step unpacked the column matrix as
# rows, so the two cancelled on a round trip -- and rocksalt is cubic, whose
# matrix is diagonal and therefore its own transpose, so every assertion above
# passes either way. On a hexagonal cell the error is a different crystal:
# a = b = 2.504 Angstrom, gamma = 120 arrived as 2.7996 / 2.1685 and 116.565.
# ---------------------------------------------------------------------------


def hexagonal_sheet(a: float = 2.504, c: float = 15.0) -> Structure:
    """A hexagonal cell, whose lattice matrix is NOT its own transpose."""
    return Structure(
        name="BN",
        formula="BN",
        # P-6m2, not P6/mmm: B and N are different elements, so the
        # horizontal mirror and the inversion of the elemental lattice are
        # gone. The declaration check caught this label too (#128).
        spacegroup="P-6m2",
        crystal_system="hexagonal",
        a=a,
        b=a,
        c=c,
        alpha=90.0,
        beta=90.0,
        gamma=120.0,
        unit_cell=(
            StructureAtom(Z=5, fxyz=(0.0, 0.0, 0.0)),
            StructureAtom(Z=7, fxyz=(1.0 / 3.0, 2.0 / 3.0, 0.0)),
        ),
        crystal_spacegroup=187,
        crystal_asymm_unit=(StructureAtom(Z=5, fxyz=(0.0, 0.0, 0.0)),),
    )


def test_hexagonal_structure_keeps_its_cell_through_periodic_system():
    """The marshalled cell must be the cell the Structure declared."""
    import vibeqc as vq

    struct = hexagonal_sheet()
    system = _to_periodic_system(struct)
    params = vq.cell_parameters(system)
    bohr_to_ang = 0.529177210903

    assert params.a * bohr_to_ang == pytest.approx(struct.a, rel=1e-12)
    assert params.b * bohr_to_ang == pytest.approx(struct.b, rel=1e-12)
    assert params.c * bohr_to_ang == pytest.approx(struct.c, rel=1e-12)
    assert params.gamma == pytest.approx(120.0, abs=1e-9)
    # The declaration the Structure already carries must hold on the result.
    vq.check_crystal_system(system, struct.crystal_system)


def test_hexagonal_structure_survives_a_round_trip():
    """Round-tripping a non-orthogonal cell must return the same cell.

    This pins the two directions as a PAIR. A round trip cannot see the
    original defect on its own -- the forward and reverse transposes cancelled,
    which is how it survived -- but it does fail the moment only one side is
    corrected, which is the easy mistake when fixing it. Verified: with the
    forward transpose alone, this test fails and the other two pass.
    """
    struct = hexagonal_sheet()
    back = _structure_from_periodic_system(_to_periodic_system(struct), struct)

    assert back.a == pytest.approx(struct.a, rel=1e-10)
    assert back.b == pytest.approx(struct.b, rel=1e-10)
    assert back.c == pytest.approx(struct.c, rel=1e-10)
    assert back.alpha == pytest.approx(90.0, abs=1e-9)
    assert back.beta == pytest.approx(90.0, abs=1e-9)
    assert back.gamma == pytest.approx(120.0, abs=1e-9)
    assert [s.Z for s in back.unit_cell] == [5, 7]
    for want, got in zip(struct.unit_cell, back.unit_cell):
        assert got.fxyz == pytest.approx(want.fxyz, abs=1e-10)


def test_a_structure_whose_declaration_is_contradicted_is_refused():
    """A Structure declaring one system and carrying another fails closed."""
    import dataclasses

    import vibeqc as vq

    struct = dataclasses.replace(hexagonal_sheet(), crystal_system="cubic")
    with pytest.raises(vq.LatticeDeclarationError):
        _to_periodic_system(struct)


def test_a_structure_whose_space_group_is_contradicted_is_refused():
    """GitLab #128: the marshaller enforces the group the record declares.

    This is the tighter of the two declarations, and it is what caught the
    mislabelled fixtures in this file: a 1 Mg + 1 O cubic cell is Pm-3m, not
    rocksalt's Fm-3m, and a B/N sheet is P-6m2, not P6/mmm.
    """
    import dataclasses

    import vibeqc as vq

    wrong = dataclasses.replace(rocksalt(), crystal_spacegroup=225)  # Fm-3m
    with pytest.raises(vq.LatticeDeclarationError) as excinfo:
        _to_periodic_system(wrong)
    assert "detected : 221" in str(excinfo.value)


def test_a_relaxed_structure_is_not_held_to_its_stale_symmetry():
    """The check keys on the field a relaxation clears, not the one it keeps.

    ``_structure_from_periodic_system`` zeroes ``crystal_spacegroup`` because
    it describes the *input* geometry and a relaxation has moved the atoms,
    but the ``spacegroup`` string is carried over and goes stale. Enforcing
    the string would fail closed on every relaxed structure fed back in;
    enforcing the integer skips exactly those records, which is the point.
    """
    import vibeqc as vq

    struct = rocksalt()
    system = _to_periodic_system(struct)
    # A relaxation moves atoms; displace one so the stale label genuinely no
    # longer describes the structure. Without this the test cannot tell the
    # two implementations apart.
    moved = vq.PeriodicSystem(
        3,
        system.lattice,
        [
            vq.Atom(12, [0.0, 0.0, 0.0]),
            vq.Atom(8, [x + 0.30 for x in system.unit_cell[1].xyz]),
        ],
    )
    relaxed = _structure_from_periodic_system(moved, struct)

    assert relaxed.crystal_spacegroup == 0        # cleared by the relaxation
    assert relaxed.spacegroup == "Pm-3m"          # carried over, now stale
    from vibeqc.periodic_symmetrize import detect_spacegroup
    assert detect_spacegroup(moved).international_symbol != "Pm-3m"

    # Feeding it back in must not raise on that stale string. Keying the
    # check on it instead of on the cleared integer fails here.
    _to_periodic_system(relaxed)
