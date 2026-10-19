"""run_job(optimize=True) fails closed on the native and Brent backends.

``optimizer_backend="native"`` (scipy L-BFGS-B) and ``"brent"`` (steepest
descent with a Brent line search) used to ignore ``MolecularOptimizeResult.
converged``: an optimizer that hit ``max_opt_steps`` still wrote "Optimized
geometry" and went on to the final single point and the frequencies, so a
job that never reached a stationary point reported energies and frequencies
as if it had. The ASE backend raised on a capped run; these two did not.

They now behave like the ASE backend. The ``.out`` says ``CONVERGED`` or
``NOT CONVERGED`` with the final largest gradient component, and a run that
did not converge writes the last geometry, marks the manifest crashed and
raises :class:`vibeqc.geomopt.GeomOptNotConvergedError` before the single
point.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

import vibeqc.molecular_optimize as mo
from vibeqc import Atom, Molecule, run_job
from vibeqc.geomopt import GeomOptFailedError, GeomOptNotConvergedError

BACKENDS = ["native", "brent"]
OPTIMIZER_KEY = {"native": "native_lbfgsb", "brent": "brent_sd"}

BOHR_IN_ANGSTROM = 0.52917721092
# fmax = 0.05 eV/A in Ha/bohr.
DEFAULT_GMAX = 0.05 * 0.036749322 / 1.8897261

QUIET = dict(
    basis="sto-3g",
    method="rhf",
    optimize=True,
    output_qvf=False,
    write_molden_file=False,
    write_population_file=False,
    citations=False,
    progress=False,
)


def _water() -> Molecule:
    """H2O stretched and bent away from its STO-3G minimum (bohr)."""
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.60, -1.20]),
            Atom(1, [0.0, -1.60, -1.20]),
        ]
    )


def _verdict_numbers(out: str) -> tuple[float, float]:
    """The ``final max|g|`` and its threshold, as the .out prints them."""
    match = re.search(
        r"final max\|g\| = ([0-9.eE+-]+) Ha/bohr \(threshold ([0-9.eE+-]+)\)", out
    )
    assert match, f"no verdict line in the .out:\n{out}"
    return float(match.group(1)), float(match.group(2))


def _xyz_bohr(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    rows = [ln.split() for ln in lines[2 : 2 + int(lines[0])]]
    return np.array([[float(x) for x in r[1:4]] for r in rows]) / BOHR_IN_ANGSTROM


@pytest.mark.parametrize("backend", BACKENDS)
def test_capped_run_fails_closed(tmp_path, backend):
    """One step cannot converge water from here: the job must stop, say so,
    keep the last geometry, and not run the single point or the frequencies."""
    start = _water()
    stem = tmp_path / f"h2o_{backend}"
    with pytest.raises(GeomOptNotConvergedError) as excinfo:
        run_job(
            start,
            output=stem,
            optimizer_backend=backend,
            max_opt_steps=1,
            hessian=True,
            **QUIET,
        )
    exc = excinfo.value

    # One error type to catch, and still a RuntimeError like the ASE backend's.
    assert isinstance(exc, GeomOptFailedError)
    assert isinstance(exc, RuntimeError)
    assert "did not converge" in str(exc)
    assert exc.result.converged is False
    assert exc.result.optimizer == OPTIMIZER_KEY[backend]
    assert exc.result.n_iter >= 1
    assert exc.result.method == "rhf"
    assert exc.result.basis == "sto-3g"
    assert exc.system is exc.result.system

    out = stem.with_suffix(".out").read_text()
    assert "Geometry optimization NOT CONVERGED" in out
    assert "step limit" in out
    assert "FATAL" in out
    # The verdict quotes the gradient the optimizer stopped at, against the
    # threshold it had to meet.
    final, threshold = _verdict_numbers(out)
    assert threshold == pytest.approx(DEFAULT_GMAX, rel=1e-3)
    assert final > threshold
    assert final == pytest.approx(
        float(np.max(np.abs(exc.result.gradient))), rel=1e-3
    )
    # Nothing downstream ran as if the geometry were optimized.
    assert "Optimized geometry" not in out
    assert "## Vibrational Frequencies" not in out
    assert "Thermochemistry" not in out

    # The last geometry is written, and it is the geometry the error carries,
    # not the starting one.
    xyz = _xyz_bohr(stem.with_suffix(".xyz"))
    assert "Last non-converged geometry written to" in out
    expected = np.array([list(a.xyz) for a in exc.system.atoms])
    np.testing.assert_allclose(xyz, expected, atol=1e-5)
    moved = np.array([list(a.xyz) for a in start.atoms])
    assert np.max(np.abs(xyz - moved)) > 1e-3

    assert 'status           = "crashed"' in stem.with_suffix(
        ".system"
    ).read_text()


def test_capped_run_without_xyz_request_writes_none(tmp_path):
    stem = tmp_path / "h2o_noxyz"
    with pytest.raises(GeomOptNotConvergedError):
        run_job(
            _water(),
            output=stem,
            optimizer_backend="native",
            max_opt_steps=1,
            write_xyz_file=False,
            **QUIET,
        )
    assert not stem.with_suffix(".xyz").exists()
    assert "NOT CONVERGED" in stem.with_suffix(".out").read_text()


@pytest.mark.parametrize("backend", BACKENDS)
def test_converged_run_says_converged_and_continues(tmp_path, backend):
    stem = tmp_path / f"h2o_{backend}_ok"
    result = run_job(
        _water(), output=stem, optimizer_backend=backend, **QUIET
    )
    out = stem.with_suffix(".out").read_text()
    assert "Geometry optimization CONVERGED" in out
    assert "NOT CONVERGED" not in out
    final, threshold = _verdict_numbers(out)
    assert final <= threshold
    # The verdict comes before the geometry it vouches for, and the job
    # carries on to the final single point.
    assert out.index("Geometry optimization CONVERGED") < out.index(
        "Optimized geometry"
    )
    assert result.converged
    assert stem.with_suffix(".xyz").is_file()
    assert 'status           = "complete"' in stem.with_suffix(
        ".system"
    ).read_text()


@pytest.mark.parametrize("backend", BACKENDS)
def test_unconverged_result_from_any_optimizer_fails_closed(
    tmp_path, monkeypatch, backend
):
    """The runner trusts ``converged``, whatever produced the result: a result
    built without a ``message`` (as the stubs in other tests are) still fails
    closed, with a generic reason."""
    start = _water()

    def _stalled(mol, basis_name, **kwargs):
        return mo.MolecularOptimizeResult(
            system=mol,
            energy=-74.9,
            gradient=np.full((3, 3), 5e-3),
            n_iter=7,
            converged=False,
        )

    monkeypatch.setattr(
        mo, "optimize_molecule" if backend == "native" else "optimize_molecule_brent", _stalled
    )
    stem = tmp_path / f"h2o_{backend}_stalled"
    with pytest.raises(GeomOptNotConvergedError, match="stopped before converging"):
        run_job(start, output=stem, optimizer_backend=backend, **QUIET)

    out = stem.with_suffix(".out").read_text()
    assert "Geometry optimization NOT CONVERGED" in out
    assert "7 iterations" in out
    final, _ = _verdict_numbers(out)
    assert final == pytest.approx(5e-3)
    assert "Optimized geometry" not in out


def test_molecular_result_converts_to_geomopt_result():
    mol = _water()
    res = mo.MolecularOptimizeResult(
        system=mol,
        energy=-74.9,
        gradient=np.arange(9, dtype=float).reshape(3, 3),
        n_iter=4,
        converged=False,
        trajectory_frames=[mol],
        trajectory_energies=[-74.8],
        message="stopped at the step limit",
    )
    out = res.to_geomopt_result(
        "native_lbfgsb", method="rhf", basis="sto-3g", functional=None
    )
    assert out.system is mol
    assert out.gradient.shape == (9,)
    assert out.converged is False
    assert out.n_iter == 4
    assert out.optimizer == "native_lbfgsb"
    assert out.message == "stopped at the step limit"
    assert out.trajectory_energies == [-74.8]
