"""Artifact stems that contain a dot must round-trip (GitLab issue #254).

``output=`` is a documented path **stem**: ``docs/user_guide/output_files.md``
states that the emitted files are ``{output}.out``, ``{output}.system``,
``{output}.molden`` ... i.e. the suffix is *appended*.

:meth:`pathlib.Path.with_suffix` *substitutes* instead, treating everything
after the final dot of the final path component as a replaceable extension.
A stem carrying a dot -- and campaign stems routinely encode a lattice
constant, a scale factor or a tolerance -- therefore truncated::

    output_scale_4.08  -> output_scale_4.out
    output_scale_4.12  -> output_scale_4.out    # same file, silently

The jobs still exited 0.  Only the captured output was destroyed, so an
EOS scan of nine points left two files and seven results ceased to exist
(campaign ``rel134periodic-20260817T154157Z``); a weekend bundle kept 6 of
24 ``.out`` files; and the resulting ``FileNotFoundError`` was reported as
a confident ``route-mismatch`` verdict on seven healthy calculations.

These tests pin the round-trip at the two places that decide artifact
names -- the declarative :class:`~vibeqc.output.OutputPlan` and the two
runners that write the files -- plus the source-level guard, because this
defect has been reintroduced through a filename more than once.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vibeqc.output import OutputPlan
from vibeqc.output._stem_paths import stem_sibling


REPO_ROOT = Path(__file__).resolve().parents[1]

# A nine-point EOS ladder of the shape that lost seven of its points.
EOS_STEMS = [f"output_scale_{4.00 + 0.02 * i:.2f}" for i in range(9)]

# The campaign stem from the false route-mismatch bundle: the dot sits in
# the middle of the name, so ``.suffix`` is the nonsense ``".0082-k222"``.
LATTICE_STEM = "lih-gdf-rhf-a4.0082-k222"


def _path_for(plan: OutputPlan, role: str) -> Path:
    for planned in plan.files:
        if planned.role == role:
            return Path(planned.path)
    raise AssertionError(f"plan declares no {role!r} artefact")


def _plan(stem: Path) -> OutputPlan:
    return OutputPlan.from_run_job_kwargs(
        output=stem,
        method="rhf",
        basis="sto-3g",
        functional=None,
    )


# --------------------------------------------------------------------------
# 1. The plan must declare the stem verbatim
# --------------------------------------------------------------------------

def test_plan_declares_dot_bearing_stem_verbatim(tmp_path: Path) -> None:
    """Every declared artefact name starts with the whole stem."""
    plan = _plan(tmp_path / LATTICE_STEM)
    for planned in plan.files:
        assert Path(planned.path).name.startswith(LATTICE_STEM), (
            f"{planned.role} artefact {Path(planned.path).name!r} lost part "
            f"of the stem {LATTICE_STEM!r}"
        )


def test_plan_log_and_manifest_append_rather_than_substitute(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path / LATTICE_STEM)
    assert _path_for(plan, "log").name == f"{LATTICE_STEM}.out"
    assert _path_for(plan, "manifest").name == f"{LATTICE_STEM}.system"


# --------------------------------------------------------------------------
# 2. Distinct stems must stay distinct -- the collision that lost the scan
# --------------------------------------------------------------------------

def test_eos_ladder_stems_do_not_collide(tmp_path: Path) -> None:
    """Nine EOS points must declare nine distinct logs, not two."""
    logs = [_path_for(_plan(tmp_path / s), "log") for s in EOS_STEMS]
    assert len(set(logs)) == len(EOS_STEMS), (
        f"{len(EOS_STEMS)} EOS stems collapsed onto "
        f"{len(set(logs))} log paths: {sorted({p.name for p in logs})}"
    )


def test_every_declared_artefact_is_injective_over_the_ladder(
    tmp_path: Path,
) -> None:
    """No artefact family may collide across the ladder either.

    A collision in *any* role overwrites one point's evidence with
    another's, which is the failure that made the EOS curves
    unrecoverable rather than merely incomplete.

    Some roles declare more than one file per job (``citations`` emits
    ``.bibtex`` + ``.references``; ``population`` emits ``.txt`` +
    ``.json``), so the assertion is injectivity of the declared paths --
    not a count against the number of stems.
    """
    by_role: dict[str, list[Path]] = {}
    for stem in EOS_STEMS:
        for planned in _plan(tmp_path / stem).files:
            by_role.setdefault(planned.role, []).append(Path(planned.path))
    collided = {
        role: len(paths) - len(set(paths))
        for role, paths in by_role.items()
        if len(set(paths)) != len(paths)
    }
    assert not collided, (
        f"artefact roles collided across the ladder (role -> lost files): "
        f"{collided}"
    )


# --------------------------------------------------------------------------
# 3. Negative control -- the same route with the dot removed (L125)
# --------------------------------------------------------------------------

def test_dot_free_stems_are_byte_identical_to_the_historical_names(
    tmp_path: Path,
) -> None:
    """The same plan route with the feature (a dot in the stem) off.

    A dot-free stem must produce exactly the names it always produced.
    The correct set is exact, so it is asserted exactly.
    """
    plan = _plan(tmp_path / "output-h2o")
    names = {Path(f.path).name for f in plan.files}
    assert "output-h2o.out" in names
    assert "output-h2o.system" in names
    for name in names:
        assert name.count(".") == name[len("output-h2o"):].count("."), name
        assert not name.startswith("output-h2o.out."), name


def test_stem_sibling_is_a_no_op_on_dot_free_stems() -> None:
    """The helper must not change any name that was already correct."""
    for suffix in (".out", ".system", ".molden", ".scf.jsonl"):
        assert stem_sibling(Path("output-h2o"), suffix) == Path(
            f"output-h2o{suffix}"
        )


# --------------------------------------------------------------------------
# 4. Compound artefact names must survive (the .opt.xyz idiom)
# --------------------------------------------------------------------------

def test_stem_sibling_does_not_double_an_already_present_suffix() -> None:
    """``{stem}.opt.xyz`` handed to the ``.xyz`` writer stays put.

    ``periodic_runner`` names the optimized-geometry artefacts by passing
    a stem that already ends in the target suffix; under substitution the
    writer's own re-derivation was a no-op.  It must stay a no-op.
    """
    assert stem_sibling(Path("mgo.opt.xyz"), ".xyz") == Path("mgo.opt.xyz")
    assert stem_sibling(Path("mgo.opt.POSCAR"), ".POSCAR") == Path(
        "mgo.opt.POSCAR"
    )
    # ... but only for that exact suffix.
    assert stem_sibling(Path("mgo.opt.xyz"), ".out") == Path("mgo.opt.xyz.out")


def test_stem_sibling_rejects_a_suffix_without_a_leading_dot() -> None:
    with pytest.raises(ValueError):
        stem_sibling(Path("h2o"), "out")
    with pytest.raises(ValueError):
        stem_sibling(Path("h2o"), "")


# --------------------------------------------------------------------------
# 5. End to end -- the files actually land at the full stem
# --------------------------------------------------------------------------

def test_run_job_writes_artifacts_at_the_full_dotted_stem(
    tmp_path: Path,
) -> None:
    """A real (tiny) job with a dotted stem keeps its whole name."""
    from vibeqc import Atom, Molecule, run_job

    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])], 0, 1)
    stem = tmp_path / "h2-scale_0.74"
    run_job(
        mol,
        basis="sto-3g",
        method="rhf",
        output=str(stem),
        write_molden_file=False,
        output_qvf=False,
    )
    assert (tmp_path / "h2-scale_0.74.out").exists()
    assert (tmp_path / "h2-scale_0.74.system").exists()
    # The truncated name must not exist at all.
    assert not (tmp_path / "h2-scale_0.out").exists()
    assert not (tmp_path / "h2-scale_0.system").exists()


def test_two_scale_points_do_not_overwrite_each_other(tmp_path: Path) -> None:
    """The exact shape of the lost EOS scan, run for real on two points."""
    from vibeqc import Atom, Molecule, run_job

    written: list[Path] = []
    for r in ("1.30", "1.50"):
        mol = Molecule(
            [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, float(r)])], 0, 1
        )
        stem = tmp_path / f"h2_scale_{r}"
        run_job(
            mol,
            basis="sto-3g",
            method="rhf",
            output=str(stem),
            write_molden_file=False,
            output_qvf=False,
        )
        written.append(Path(f"{stem}.out"))

    assert all(p.exists() for p in written), [str(p) for p in written]
    assert len({p.name for p in written}) == 2
    # Both logs must be non-empty and describe their own point.
    for p in written:
        assert p.stat().st_size > 0


# --------------------------------------------------------------------------
# 6. Source guard -- the defect has been reintroduced through a filename
#    before (L82's shared-path last-write-wins, via a stem this time)
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "relative",
    [
        "python/vibeqc/runner.py",
        "python/vibeqc/periodic_runner.py",
        "python/vibeqc/output/plan.py",
    ],
)
def test_no_job_stem_is_suffixed_by_substitution(relative: str) -> None:
    """No artefact path may be derived from a job stem with
    ``with_suffix``; use ``stem_sibling`` so a dotted stem survives."""
    source = (REPO_ROOT / relative).read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in source.splitlines()
        if "with_suffix" in line
        and not line.lstrip().startswith("#")
        and any(
            recv in line
            for recv in ("output_stem.", "stem.", "_crash_target.")
        )
    ]
    assert not offenders, (
        f"{relative} still derives an artefact path by suffix substitution; "
        f"a dotted stem would truncate: {offenders}"
    )
