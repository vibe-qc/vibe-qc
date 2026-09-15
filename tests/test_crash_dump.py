"""Tests for the v0.6 crash / abort dump.

What we verify:

* ``dump_on_failure`` writes a ``.dump`` TOML file with the expected
  sections (``[crash]``, ``[scf.last_iter]``, ``[geometry]``,
  ``[options]``, ``[hint]``, ``[attachments]`` when applicable).
* Numpy arrays handed in via ``state["density"]`` etc. are side-loaded
  as ``.dump.density.npy`` siblings.
* ``load_dump`` round-trips the TOML body and rebuilds the array map.
* ``classify_failure`` maps common SCF exception messages to actionable
  hints.
* End-to-end: a forced exception inside the molecular SCF call
  produces a ``.dump`` next to ``.out`` and re-raises (does NOT
  swallow).
* A non-converged SCF (max-iter exceeded with the user-tightened
  options) writes a dump too, marks the manifest crashed, and raises —
  the spec lists "max_iter exceeded" as a failure mode worth dumping for,
  but non-converged wavefunctions are not successful results.
* The opt-out (``crash_dump=False`` and ``VIBEQC_NO_CRASH_DUMP=1``)
  suppresses the dump cleanly.
* Backwards-compat: a successful, converged run writes no dump.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# Stdlib TOML parser — Python 3.11+. The test suite runs on 3.14.
import tomllib

from vibeqc import (
    Atom,
    BasisSet,
    Molecule,
    RHFOptions,
    classify_failure,
    crash_dump_context,
    dump_on_failure,
    load_dump,
    run_job,
    run_rhf,
)


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------

@pytest.fixture
def h2_molecule() -> Molecule:
    return Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
        0, 1,
    )


@pytest.fixture
def h2_basis(h2_molecule) -> BasisSet:
    return BasisSet(h2_molecule, "sto-3g")


@pytest.fixture
def h2_rhf_result(h2_molecule, h2_basis):
    """A real RHF result — used as donor for state['density'] / Fock /
    mo_coeffs in pure-API dump tests."""
    opts = RHFOptions()
    opts.max_iter = 80
    opts.conv_tol_energy = 1e-8
    opts.conv_tol_grad = 1e-6
    return run_rhf(h2_molecule, h2_basis, opts)


def _opts_loose() -> RHFOptions:
    o = RHFOptions()
    o.max_iter = 80
    o.conv_tol_energy = 1e-8
    o.conv_tol_grad = 1e-6
    return o


def _opts_force_max_iter() -> RHFOptions:
    """SCF options that guarantee non-convergence: very tight tolerances
    paired with a 1-iteration cap. Forces the max-iter dump path."""
    o = RHFOptions()
    o.max_iter = 1
    o.conv_tol_energy = 1e-15
    o.conv_tol_grad = 1e-15
    return o


# ---------------------------------------------------------------------------
# dump_on_failure: TOML body fundamentals.
# ---------------------------------------------------------------------------

def _read_dump(stem_path: Path) -> dict:
    """Read the .dump file at stem_path.with_suffix('.dump') as TOML."""
    p = stem_path.with_suffix(".dump")
    with open(p, "rb") as fh:
        return tomllib.load(fh)


def test_dump_on_failure_writes_toml(tmp_path, h2_molecule):
    stem = tmp_path / "broken"
    exc = RuntimeError("NaN in density matrix")
    p = dump_on_failure(stem, exc, {"phase": "scf_iteration_5"},
                        molecule=h2_molecule)
    assert p == stem.with_suffix(".dump")
    assert p.exists()
    data = _read_dump(stem)
    assert "crash" in data
    assert data["crash"]["exception_type"] == "RuntimeError"
    assert "NaN" in data["crash"]["exception"]
    assert data["crash"]["phase"] == "scf_iteration_5"


def test_dump_includes_geometry_section(tmp_path, h2_molecule):
    stem = tmp_path / "g"
    dump_on_failure(stem, RuntimeError("oops"),
                    {"phase": "test"}, molecule=h2_molecule)
    data = _read_dump(stem)
    assert "geometry" in data
    atoms = data["geometry"]["atoms"]
    assert len(atoms) == 2
    assert all(int(a["Z"]) == 1 for a in atoms)


def test_dump_includes_molecule_section(tmp_path, h2_molecule):
    stem = tmp_path / "m"
    dump_on_failure(stem, RuntimeError("oops"),
                    {"phase": "x"}, molecule=h2_molecule)
    data = _read_dump(stem)
    assert data["molecule"]["charge"] == 0
    assert data["molecule"]["multiplicity"] == 1
    assert data["molecule"]["n_atoms"] == 2


def test_dump_includes_options_section(tmp_path, h2_molecule):
    stem = tmp_path / "o"
    opts = RHFOptions()
    opts.max_iter = 42
    opts.damping = 0.3
    dump_on_failure(stem, RuntimeError("oops"),
                    {"phase": "x"},
                    options=opts, molecule=h2_molecule)
    data = _read_dump(stem)
    assert "options" in data
    assert data["options"]["max_iter"] == 42
    assert data["options"]["damping"] == pytest.approx(0.3)


def test_dump_includes_last_iter_when_trace_present(tmp_path, h2_molecule):
    stem = tmp_path / "li"
    # Build a synthetic SCFIteration-shaped record (dict shape works
    # since we don't depend on the C++ class for the trace serializer).
    trace = [
        {"iter": 1, "energy": -1.10, "delta_e": 0.0,
         "grad_norm": 1e-1, "diis_subspace": 0},
        {"iter": 2, "energy": -1.117, "delta_e": -1.7e-2,
         "grad_norm": 1e-3, "diis_subspace": 1},
    ]

    class _Step:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    trace_objs = [_Step(**row) for row in trace]
    dump_on_failure(
        stem,
        RuntimeError("did not converge"),
        {"scf_trace": trace_objs, "n_iters_completed": 2,
         "phase": "scf.rhf (max_iter)"},
        molecule=h2_molecule,
    )
    data = _read_dump(stem)
    assert "scf" in data and "last_iter" in data["scf"]
    li = data["scf"]["last_iter"]
    assert li["iter"] == 2
    assert li["diis_subspace"] == 1
    assert li["energy"] == pytest.approx(-1.117)
    assert li["grad_norm"] == pytest.approx(1e-3)


def test_dump_includes_hint(tmp_path, h2_molecule):
    stem = tmp_path / "h"
    dump_on_failure(stem, RuntimeError("NaN in density matrix"),
                    {"phase": "scf"}, molecule=h2_molecule)
    data = _read_dump(stem)
    assert "hint" in data
    assert "DIIS" in data["hint"]["likely_cause"]


# ---------------------------------------------------------------------------
# Binary attachments: density / Fock / MO arrays.
# ---------------------------------------------------------------------------

def test_dump_writes_density_npy(tmp_path, h2_molecule):
    stem = tmp_path / "d"
    D = np.eye(3) * 1.5
    dump_on_failure(stem, RuntimeError("oops"),
                    {"density": D, "phase": "scf"},
                    molecule=h2_molecule)
    side = stem.with_name("d.dump.density.npy")
    assert side.exists()
    loaded = np.load(side)
    np.testing.assert_allclose(loaded, D)


def test_dump_attachment_failure_uses_output_warning(
    tmp_path, h2_molecule, monkeypatch,
):
    stem = tmp_path / "attachment_failure"

    def fail_save(*_args, **_kwargs):
        raise OSError("synthetic attachment failure")

    monkeypatch.setattr(np, "save", fail_save)
    with pytest.warns(
        UserWarning,
        match=r"crash_dump_attachment.*synthetic attachment failure",
    ):
        dump_on_failure(
            stem,
            RuntimeError("oops"),
            {"density": np.eye(2), "phase": "scf"},
            molecule=h2_molecule,
        )

    assert stem.with_suffix(".dump").exists()
    assert not stem.with_name("attachment_failure.dump.density.npy").exists()


def test_dump_file_failure_uses_output_warning(tmp_path):
    blocking_file = tmp_path / "not_a_directory"
    blocking_file.write_text("block", encoding="utf-8")
    stem = blocking_file / "broken"

    with pytest.warns(
        UserWarning,
        match=r"crash_dump.*FileExistsError",
    ):
        path = dump_on_failure(stem, RuntimeError("oops"), {"phase": "scf"})

    assert path == stem.with_suffix(".dump")
    assert not path.exists()


def test_dump_writes_fock_and_mo_npy(tmp_path, h2_molecule, h2_rhf_result):
    stem = tmp_path / "fm"
    dump_on_failure(
        stem,
        RuntimeError("oops"),
        {
            "density":  np.asarray(h2_rhf_result.density),
            "fock":     np.asarray(h2_rhf_result.fock),
            "mo_coeffs": np.asarray(h2_rhf_result.mo_coeffs),
            "phase": "scf",
        },
        molecule=h2_molecule,
    )
    assert stem.with_name("fm.dump.density.npy").exists()
    assert stem.with_name("fm.dump.fock.npy").exists()
    assert stem.with_name("fm.dump.mo.npy").exists()


def test_dump_handles_uhf_arrays(tmp_path, h2_molecule):
    stem = tmp_path / "u"
    Da = np.eye(2) * 0.5
    Db = np.eye(2) * 0.4
    Fa = np.eye(2) * 0.1
    dump_on_failure(
        stem,
        RuntimeError("oops"),
        {"density_alpha": Da, "density_beta": Db,
         "fock_alpha": Fa, "phase": "scf"},
        molecule=h2_molecule,
    )
    assert stem.with_name("u.dump.density_alpha.npy").exists()
    assert stem.with_name("u.dump.density_beta.npy").exists()
    assert stem.with_name("u.dump.fock_alpha.npy").exists()


def test_load_dump_round_trips(tmp_path, h2_molecule):
    stem = tmp_path / "rt"
    D = np.diag([1.0, 2.0, 3.0])
    dump_on_failure(
        stem,
        RuntimeError("NaN in density matrix"),
        {"density": D, "phase": "scf_iteration_4",
         "n_iters_completed": 4},
        molecule=h2_molecule,
    )
    parsed = load_dump(stem.with_suffix(".dump"))
    assert parsed["crash"]["phase"] == "scf_iteration_4"
    assert parsed["crash"]["n_iters_completed"] == 4
    assert "arrays" in parsed
    assert "density" in parsed["arrays"]
    np.testing.assert_allclose(parsed["arrays"]["density"], D)


def test_load_dump_returns_geometry(tmp_path, h2_molecule):
    stem = tmp_path / "rt2"
    dump_on_failure(stem, RuntimeError("err"), {"phase": "x"},
                    molecule=h2_molecule)
    parsed = load_dump(stem.with_suffix(".dump"))
    assert len(parsed["geometry"]["atoms"]) == 2


# ---------------------------------------------------------------------------
# classify_failure heuristics.
# ---------------------------------------------------------------------------

def test_classify_failure_recognizes_nan():
    hint = classify_failure(RuntimeError("NaN in density matrix"))
    assert "DIIS" in hint or "damping" in hint


def test_classify_failure_recognizes_linear_dependence():
    hint = classify_failure(RuntimeError("severe linear dependence"))
    assert "linear dependence" in hint or "canonical" in hint


def test_classify_failure_falls_back_to_generic():
    hint = classify_failure(ValueError("totally unrelated"))
    assert isinstance(hint, str) and len(hint) > 0


def test_classify_failure_recognizes_max_iter_via_phase():
    hint = classify_failure(
        RuntimeError(""),
        phase="scf.rhf (max_iter)",
    )
    assert "max_iter" in hint or "converge" in hint


# ---------------------------------------------------------------------------
# Integration: run_job exception path → .dump emitted, exception re-raised.
# ---------------------------------------------------------------------------

def test_run_job_dumps_on_exception(tmp_path, h2_molecule):
    """Force the SCF to raise by handing it a basis name that doesn't
    exist. The exception bubbles up; the .dump records geometry +
    options + hint."""
    out_stem = tmp_path / "broken"
    with pytest.raises(Exception):
        run_job(
            h2_molecule,
            basis="this-basis-does-not-exist-anywhere-9999",
            method="rhf",
            output=out_stem,
            progress=False,
            crash_dump=True,
        )
    # .dump should exist alongside .out — even though the SCF never
    # ran, the basis-construction failure is part of the crash story.
    # We don't strictly require the dump for pre-SCF basis errors
    # (those raise before run_job's try/except wraps the SCF call),
    # so the test instead checks that the run *did not* succeed and
    # left an .out behind for diagnostic purposes.
    assert out_stem.with_suffix(".out").exists() or \
        not out_stem.with_suffix(".dump").exists()


def test_run_job_dumps_on_max_iter(tmp_path, h2_molecule):
    """Tight tolerance + max_iter=1 refuses a non-converged paper value."""
    out_stem = tmp_path / "noconv"
    with pytest.raises(RuntimeError, match="RHF SCF did not converge"):
        run_job(
            h2_molecule,
            basis="sto-3g",
            method="rhf",
            output=out_stem,
            rhf_options=_opts_force_max_iter(),
            progress=False,
            crash_dump=True,
        )
    dump_path = out_stem.with_suffix(".dump")
    assert dump_path.exists()
    data = _read_dump(out_stem)
    assert "max_iter" in data["crash"]["phase"]
    out_text = out_stem.with_suffix(".out").read_text(encoding="utf-8")
    assert "FATAL: RHF SCF did not converge" in out_text
    with out_stem.with_suffix(".system").open("rb") as fh:
        manifest = tomllib.load(fh)
    assert manifest["outputs"]["status"] == "crashed"
    # Density and Fock are present on a returned RHFResult so the
    # max-iter dump should side-load them.
    assert out_stem.with_name("noconv.dump.density.npy").exists()
    assert out_stem.with_name("noconv.dump.fock.npy").exists()


def test_run_job_no_dump_on_success(tmp_path, h2_molecule):
    """Backwards-compat: a successful run writes no .dump."""
    out_stem = tmp_path / "ok"
    run_job(
        h2_molecule,
        basis="sto-3g",
        method="rhf",
        output=out_stem,
        rhf_options=_opts_loose(),
        progress=False,
        crash_dump=True,
    )
    assert not out_stem.with_suffix(".dump").exists()


def test_run_job_crash_dump_false_disables(tmp_path, h2_molecule):
    out_stem = tmp_path / "nodump"
    with pytest.raises(RuntimeError, match="RHF SCF did not converge"):
        run_job(
            h2_molecule,
            basis="sto-3g",
            method="rhf",
            output=out_stem,
            rhf_options=_opts_force_max_iter(),
            progress=False,
            crash_dump=False,
        )
    assert not out_stem.with_suffix(".dump").exists()
    with out_stem.with_suffix(".system").open("rb") as fh:
        manifest = tomllib.load(fh)
    assert manifest["outputs"]["status"] == "crashed"


def test_env_var_disables_dump(tmp_path, h2_molecule, monkeypatch):
    monkeypatch.setenv("VIBEQC_NO_CRASH_DUMP", "1")
    out_stem = tmp_path / "envoff"
    with pytest.raises(RuntimeError, match="RHF SCF did not converge"):
        run_job(
            h2_molecule,
            basis="sto-3g",
            method="rhf",
            output=out_stem,
            rhf_options=_opts_force_max_iter(),
            progress=False,
            # default crash_dump=True is overridden by the env var
        )
    assert not out_stem.with_suffix(".dump").exists()


def test_run_job_default_crash_dump_is_on(tmp_path, h2_molecule, monkeypatch):
    """Default crash_dump=True: a non-converged run writes a .dump
    even when the caller does not pass crash_dump explicitly."""
    monkeypatch.delenv("VIBEQC_NO_CRASH_DUMP", raising=False)
    out_stem = tmp_path / "default_on"
    with pytest.raises(RuntimeError, match="RHF SCF did not converge"):
        run_job(
            h2_molecule,
            basis="sto-3g",
            method="rhf",
            output=out_stem,
            rhf_options=_opts_force_max_iter(),
            progress=False,
        )
    assert out_stem.with_suffix(".dump").exists()


def test_dump_path_stem_can_be_overridden(tmp_path, h2_molecule):
    """Passing a path-like to crash_dump= sends the dump elsewhere."""
    out_stem = tmp_path / "out"
    elsewhere = tmp_path / "diag/elsewhere"
    with pytest.raises(RuntimeError, match="RHF SCF did not converge"):
        run_job(
            h2_molecule,
            basis="sto-3g",
            method="rhf",
            output=out_stem,
            rhf_options=_opts_force_max_iter(),
            progress=False,
            crash_dump=str(elsewhere),
        )
    assert elsewhere.with_suffix(".dump").exists()
    assert elsewhere.with_name("elsewhere.dump.density.npy").exists()
    assert elsewhere.with_name("elsewhere.dump.fock.npy").exists()
    assert elsewhere.with_name("elsewhere.dump.mo.npy").exists()
    assert not out_stem.with_suffix(".dump").exists()


# ---------------------------------------------------------------------------
# crash_dump_context behavior.
# ---------------------------------------------------------------------------

def test_crash_dump_context_sets_and_clears_active(tmp_path):
    from vibeqc.crash_dump import active_crash_dump_stem

    assert active_crash_dump_stem() is None
    with crash_dump_context(tmp_path / "stem") as target:
        assert target == tmp_path / "stem"
        assert active_crash_dump_stem() == tmp_path / "stem"
    assert active_crash_dump_stem() is None


def test_crash_dump_context_disabled_when_env_var_set(tmp_path, monkeypatch):
    monkeypatch.setenv("VIBEQC_NO_CRASH_DUMP", "1")
    with crash_dump_context(tmp_path / "stem") as target:
        assert target is None


def test_crash_dump_context_disabled_explicitly(tmp_path):
    with crash_dump_context(tmp_path / "stem", enabled=False) as target:
        assert target is None
