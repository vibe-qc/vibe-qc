"""Tests for the ``vibeqc`` CLI (:mod:`vibeqc._cli`).

The CLI ships three GPAW-equivalent sub-commands:

* ``vibeqc info`` — version + linked libraries + platform.
* ``vibeqc gpw <restart.npz>`` — summarise a GPW restart file
  (kind, energy, atoms, lattice) plus the Lippert-Hutter 1997
  citation pointer for the GPW route.
* ``vibeqc dos <restart.npz>`` — Gaussian-broadened DOS from the
  GPW restart's MO energies, written as ``(energy_Ha, dos)``
  pairs to stdout.

We exercise main() directly via ``sys.argv`` patching (covers the
import path) — that's the path the installed console script
delegates to. A small Γ-only RHF GPW SCF on H₂ / STO-3G provides
the restart artefact the gpw / dos tests parse.
"""

from __future__ import annotations

import io
import warnings
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc._cli import main as cli_main
from vibeqc.periodic_gapw_grid import GAPWExperimentalWarning, PlaneWaveGrid
from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw
from vibeqc.periodic_gapw_restart import save_gpw_result


pytestmark = pytest.mark.filterwarnings(
    "ignore::vibeqc.periodic_gapw_grid.GAPWExperimentalWarning"
)


# ---------- Fixtures ------------------------------------------------------


def _build_h2_restart(tmp_path: Path, L: float = 12.0, n: int = 32) -> Path:
    """Run a small Γ-only H₂/STO-3G GPW SCF and serialise it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GAPWExperimentalWarning)
        system = core.PeriodicSystem()
        system.dim = 3
        system.lattice = np.eye(3) * L
        system.unit_cell = [
            core.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
            core.Atom(1, [L / 2 + 0.7, L / 2, L / 2]),
        ]
        mol = vq.Molecule(
            [
                vq.Atom(1, [L / 2 - 0.7, L / 2, L / 2]),
                vq.Atom(1, [L / 2 + 0.7, L / 2, L / 2]),
            ],
            charge=0,
            multiplicity=1,
        )
        basis = vq.BasisSet(mol, "sto-3g")
        grid = PlaneWaveGrid(np.eye(3) * L, n, n, n)
        result = run_periodic_rhf_gpw(system, basis, grid=grid, quiet=True)
        assert result.converged

    path = tmp_path / "h2_rhf.npz"
    return save_gpw_result(path, result, basis, system)


@pytest.fixture(scope="module")
def h2_restart(tmp_path_factory) -> Path:
    tmp_path = tmp_path_factory.mktemp("vibeqc_cli")
    return _build_h2_restart(tmp_path)


def _run_cli(argv: list[str]) -> tuple[int, str, str]:
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    with redirect_stdout(out_buf), redirect_stderr(err_buf):
        try:
            rc = cli_main(argv)
        except SystemExit as exc:
            rc = int(exc.code) if exc.code is not None else 0
    return rc, out_buf.getvalue(), err_buf.getvalue()


# ---------- Tests ---------------------------------------------------------


def test_info_prints_vibeqc_version():
    """``vibeqc info`` exits 0 and prints a line mentioning ``vibeqc``."""
    rc, out, _err = _run_cli(["info"])
    assert rc == 0
    assert "vibeqc" in out
    # Should also surface a couple of linked library entries.
    assert "libint" in out
    assert "libxc" in out


def test_gpw_prints_energy_and_lippert_citation(h2_restart: Path):
    """``vibeqc gpw <existing.npz>`` shows the converged energy
    and the Lippert-Hutter 1997 GPW citation pointer."""
    rc, out, _err = _run_cli(["gpw", str(h2_restart)])
    assert rc == 0
    # Energy in Hartree, with the literal "Ha" unit label.
    assert "energy:" in out
    assert "Ha" in out
    # Lippert citation surface.
    assert "Lippert" in out
    assert "10.1080/00268979709482119" in out
    # Sanity check the restart-shape fields we render.
    assert "basis_name:" in out
    assert "sto-3g" in out


def test_dos_prints_energy_dos_pairs(h2_restart: Path):
    """``vibeqc dos <existing.npz>`` writes ``(energy_Ha, dos)`` pairs
    to stdout. Sum integrates the closed-shell electron count
    (2.0 e⁻ for H₂) to within the Gaussian-tail truncation."""
    rc, out, _err = _run_cli(["dos", str(h2_restart), "--sigma=0.01", "--bins=200"])
    assert rc == 0
    lines = [ln for ln in out.splitlines() if ln and not ln.startswith("#")]
    assert len(lines) == 200
    energies = []
    dos = []
    for ln in lines:
        e_str, d_str = ln.split()
        energies.append(float(e_str))
        dos.append(float(d_str))
    energies = np.asarray(energies)
    dos = np.asarray(dos)
    # Monotonic grid + non-negative DOS.
    assert np.all(np.diff(energies) > 0)
    assert np.all(dos >= 0.0)
    # Closed-shell H₂ carries 2 electrons; the integrated DOS
    # recovers it to within Gaussian-tail truncation.
    integrated = float(np.trapezoid(dos, energies))
    assert integrated == pytest.approx(2.0, rel=0.05)


def test_unknown_subcommand_exits_non_zero():
    """An unknown sub-command must exit non-zero (argparse's own
    error path) and print usage hints on stderr."""
    rc, _out, err = _run_cli(["nope-this-isnt-a-subcommand"])
    assert rc != 0
    assert "usage" in err.lower() or "invalid choice" in err.lower()


def test_no_subcommand_prints_help_non_zero():
    """``vibeqc`` with no sub-command prints help on stderr and
    exits non-zero so a wrapping shell can detect the no-op."""
    rc, _out, err = _run_cli([])
    assert rc != 0
    assert "usage" in err.lower()
    # Sub-command names should appear in the help block.
    assert "info" in err
    assert "gpw" in err
    assert "dos" in err


# ---------- v0.12 R3 sub-commands (run / opt / bands) --------------------


def test_run_executes_user_script(tmp_path: Path):
    """``vibeqc run <script.py>`` runs the script under ``__main__``
    with ``vibeqc`` preloaded; rc 0 + script stdout reaches our buffer."""
    script = tmp_path / "hello.py"
    script.write_text(
        "import vibeqc\n"
        "assert hasattr(vibeqc, '__version__')\n"
        "print('ok')\n"
    )
    rc, out, _err = _run_cli(["run", str(script)])
    assert rc == 0
    assert "ok" in out


# The `opt` subcommand runs a hardcoded BFGS(fmax=0.05, steps=20) under
# VibeqcGPW — each step is a plane-wave-grid GPW SCF + gradient whose grid
# collocation (build_J / evaluate_ao on the FFT grid) dominates the runtime,
# so the whole opt is minutes even on H₂/STO-3G (the GPW grid-efficiency
# follow-up tracked with the gapw_* files; docs/pbc_audit_2026-06.md). The
# other CLI subcommands stay on the fast lane; this end-to-end opt runs on
# the slow/nightly lane.
@pytest.mark.slow
def test_opt_optimises_h2_and_writes_xyz(tmp_path: Path):
    """``vibeqc opt <h2.xyz>`` runs a BFGS optimisation and writes
    ``<stem>.opt.xyz`` next to the input."""
    xyz_path = tmp_path / "h2.xyz"
    # H₂ at a slightly stretched bond — BFGS should pull it inwards.
    xyz_path.write_text(
        "2\n"
        "H2 stretched\n"
        "H  0.000000  0.000000  0.000000\n"
        "H  0.000000  0.000000  0.900000\n"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GAPWExperimentalWarning)
        rc, out, _err = _run_cli(
            [
                "opt", str(xyz_path),
                "--basis", "sto-3g",
                "--functional", "lda",
                "--cutoff", "150",
            ]
        )
    assert rc == 0, f"opt CLI failed; stdout=\n{out}"
    assert "initial energy" in out
    assert "final energy" in out
    out_path = tmp_path / "h2.opt.xyz"
    assert out_path.exists()
    # The optimised file should still hold a 2-atom H₂.
    body = out_path.read_text().splitlines()
    assert int(body[0].strip()) == 2
    syms = [line.split()[0] for line in body[2:4]]
    assert syms == ["H", "H"]


def test_bands_prints_eigenvalues(h2_restart: Path):
    """``vibeqc bands <restart.npz> --k-path G,X`` walks two
    high-symmetry points and prints eigenvalues for each."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", GAPWExperimentalWarning)
        rc, out, _err = _run_cli(
            [
                "bands", str(h2_restart),
                "--k-path", "G,X",
                "--n-points", "3",
            ]
        )
    assert rc == 0, f"bands CLI failed; stdout=\n{out}"
    # Header survives.
    assert "# vibeqc bands" in out
    assert "k_path" in out
    # Both endpoint labels appear in the table.
    body = [ln for ln in out.splitlines() if not ln.startswith("#") and ln.strip()]
    assert any(ln.lstrip().startswith("G ") for ln in body)
    assert any(ln.lstrip().startswith("X ") for ln in body)
    # Each printed row carries at least one numeric eigenvalue.
    for ln in body:
        tokens = ln.split()
        # label + 3 k-components + >=1 eigenvalue
        assert len(tokens) >= 5
        # The trailing eigenvalues parse as floats.
        float(tokens[-1])
