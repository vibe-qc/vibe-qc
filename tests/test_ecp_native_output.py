"""ECP integral builds write nothing to the process's stdout or stderr (#392).

The vendored libecpint reported a radial quadrature that did not meet its
convergence test by printing "Quadrature failed" to stdout ("Failed to
converge" / "Failed at second attempt" to stderr) and then used the value.
vibe-qc's native code never writes to a stream; diagnostics go through
``vibeqc::diagnostic`` into the ``.out`` (cpp/include/vibeqc/diagnostics.hpp).
tests/test_cpp_emits_no_user_output.py scans only ``cpp/``, so it never saw
``third_party/``. These tests look at behaviour instead: each build runs in a
child process whose file descriptors 1 and 2 are captured, so any byte a
native library writes is seen, whatever buffering Python applies.

The cases are ones the unpatched library is known to flag: a W2 dimer in
pob-TZVP-rev2 at the bcc nearest-neighbour distance (64 lines per build
when #358 was verified) and H2S in LANL2DZ, the #326 molecule.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

import numpy as np
import pytest

_PRELUDE = textwrap.dedent(
    """
    import json, sys
    import numpy as np
    import vibeqc as vq
    """
)

_W2 = _PRELUDE + textwrap.dedent(
    """
    from vibeqc.basis_crystal import crystal_ecp_to_libecpint_arrays, fetch_bredow_basis_sets

    d = 5.222234  # W bcc nearest-neighbour distance, bohr
    mol = vq.Molecule([vq.Atom(74, [0.0, 0.0, 0.0]), vq.Atom(74, [0.0, 0.0, d])])
    basis = vq.BasisSet(mol, "pob-tzvp-rev2")
    _, atoms = fetch_bredow_basis_sets(names=["pob-tzvp-rev2"], verbose=False, return_atoms=True)
    record = next(atom for atom in atoms["pob-tzvp-rev2"] if int(atom.Z) == 74)
    arrays = crystal_ecp_to_libecpint_arrays(record.ecp)
    blocks = []
    for _ in range(2):
        block = vq._vibeqc_core.ECPPrimitiveBlock()
        block.n_primitive = arrays.n_primitives
        block.exponents = list(arrays.exponents)
        block.coefficients = list(arrays.coefficients)
        block.ams = list(arrays.ams)
        block.ns = list(arrays.ns)
        blocks.append(block)
    V = np.asarray(vq.compute_ecp_matrix_from_primitives(basis, [0.0, 0.0, 0.0, 0.0, 0.0, d], blocks))
    """
)

_H2S = _PRELUDE + textwrap.dedent(
    """
    from vibeqc.ecp_metadata import inline_ecp_data_for

    mol = vq.Molecule([
        vq.Atom(16, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 1.815, 1.425]),
        vq.Atom(1, [0.0, -1.815, 1.425]),
    ])
    basis = vq.BasisSet(mol, "lanl2dz")
    blocks, centers, _, _ = inline_ecp_data_for(mol, "lanl2dz")
    flat = [x for center in centers for x in center]
    V = np.asarray(vq.compute_ecp_matrix_from_primitives(basis, flat, blocks))
    """
)

_EMIT = textwrap.dedent(
    """
    sys.stderr.flush()
    print("RESULT " + json.dumps({"finite": bool(np.isfinite(V).all()),
                                  "max_abs": float(np.abs(V).max())}), flush=True)
    """
)


def _run_child(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", body + _EMIT],
        capture_output=True, text=True, timeout=600,
    )


def _native_lines(stream: str) -> list[str]:
    return [line for line in stream.splitlines() if line.strip() and not line.startswith("RESULT ")]


@pytest.mark.parametrize("case", ["w2_pob_tzvp_rev2", "h2s_lanl2dz"])
def test_ecp_integral_build_writes_nothing_to_stdout_or_stderr(case):
    body = {"w2_pob_tzvp_rev2": _W2, "h2s_lanl2dz": _H2S}[case]
    child = _run_child(body)
    assert child.returncode == 0, child.stderr[-2000:]
    result = next(line for line in child.stdout.splitlines() if line.startswith("RESULT "))
    payload = json.loads(result[len("RESULT "):])
    assert payload["finite"] and payload["max_abs"] > 0.0
    stdout, stderr = _native_lines(child.stdout), _native_lines(child.stderr)
    assert not stdout, f"{len(stdout)} native stdout lines, first: {stdout[:3]}"
    assert not stderr, f"{len(stderr)} native stderr lines, first: {stderr[:3]}"


def _w2_blocks():
    import vibeqc as vq
    from vibeqc.basis_crystal import crystal_ecp_to_libecpint_arrays, fetch_bredow_basis_sets

    d = 5.222234
    mol = vq.Molecule([vq.Atom(74, [0.0, 0.0, 0.0]), vq.Atom(74, [0.0, 0.0, d])])
    basis = vq.BasisSet(mol, "pob-tzvp-rev2")
    _, atoms = fetch_bredow_basis_sets(names=["pob-tzvp-rev2"], verbose=False, return_atoms=True)
    record = next(atom for atom in atoms["pob-tzvp-rev2"] if int(atom.Z) == 74)
    arrays = crystal_ecp_to_libecpint_arrays(record.ecp)
    blocks = []
    for _ in range(2):
        block = vq._vibeqc_core.ECPPrimitiveBlock()
        block.n_primitive = arrays.n_primitives
        block.exponents = list(arrays.exponents)
        block.coefficients = list(arrays.coefficients)
        block.ams = list(arrays.ams)
        block.ns = list(arrays.ns)
        blocks.append(block)
    return basis, [0.0, 0.0, 0.0, 0.0, 0.0, d], blocks


def test_quadrature_report_counts_the_failures_it_no_longer_prints():
    """The build's report carries what libecpint used to print (#392).

    The unpatched library wrote one "Quadrature failed" line per
    integrate_small call that missed its convergence test; the report must
    count those calls, by ECP channel, with a magnitude, so the silent use
    of the value is at least visible and measurable.
    """
    import vibeqc as vq

    basis, centers, blocks = _w2_blocks()
    vq.compute_ecp_matrix_from_primitives(basis, centers, blocks)
    report = vq._vibeqc_core._ecp_quadrature_report()
    assert report["builder"] == "molecular"
    small = sum(report["failures"][0])
    assert small > 0, report
    assert report["total_failures"] == sum(sum(site) for site in report["failures"])
    assert np.isfinite(report["max_change"][0]) and report["max_change"][0] > 0.0
    assert report["bessel_lmax_exceeded"] == 0
    # A second build reports its own failures, not an accumulation.
    vq.compute_ecp_matrix_from_primitives(basis, centers, blocks)
    assert vq._vibeqc_core._ecp_quadrature_report()["total_failures"] == report["total_failures"]


def _nacl_lanl2dz():
    import vibeqc as vq
    from vibeqc.periodic_runner import _resolve_ecp_data

    a = 5.64 * 1.8897261246257702
    lattice = np.array([[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]])
    chlorine = np.array([a / 2, a / 2, a / 2]) - lattice[:, 0]
    system = vq.PeriodicSystem(
        3, lattice, [vq.Atom(11, [0.0, 0.0, 0.0]), vq.Atom(17, list(chlorine))], 0, 1,
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "lanl2dz")
    blocks, centers, _, _ = _resolve_ecp_data(system, basis)
    return system, basis, blocks, centers


def test_lattice_screen_skips_only_exact_zeros(monkeypatch):
    """Skipping libecpint-negligible image pairs leaves V_ECP(g) bit-identical (#382).

    compute_ecp_lattice_from_primitives evaluated every (shell pair,
    projector image, ECP centre); libecpint's own screen then returned an
    all-zero block for most of them after a full call setup. The builder now
    asks libecpint's screen first and skips those calls. Every retained term
    is added in the same order, so the blocks must be equal, not close, and
    the screen must actually have skipped something on this domain.
    """
    import vibeqc as vq

    system, basis, blocks, centers = _nacl_lanl2dz()
    options = vq.LatticeSumOptions()
    options.cutoff_bohr = 12.0
    options.nuclear_cutoff_bohr = 12.0
    core = vq._vibeqc_core

    monkeypatch.setenv("VIBEQC_ECP_LATTICE_SCREEN", "off")
    full = core.compute_ecp_lattice_from_primitives(basis, system, options, centers, blocks)
    full_report = core._ecp_quadrature_report()
    monkeypatch.delenv("VIBEQC_ECP_LATTICE_SCREEN")
    screened = core.compute_ecp_lattice_from_primitives(basis, system, options, centers, blocks)
    report = core._ecp_quadrature_report()

    assert report["builder"] == "periodic lattice"
    assert full_report["skipped"] == 0
    assert report["candidates"] == full_report["candidates"] > 0
    assert 0 < report["skipped"] < report["candidates"]
    assert report["total_failures"] == full_report["total_failures"]
    assert len(screened.cells) == len(full.cells) > 1
    for a, b in zip(screened.blocks, full.blocks):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_job_output_carries_one_quadrature_summary_line(tmp_path, capfd):
    """A job reports the build's quadrature misses once, in its .out (#392).

    H2S/LANL2DZ's ECP build misses the integrate_small test ten times. The
    unpatched library printed ten lines to stdout and the .out said nothing;
    now stdout carries none and the .out carries one summary line with the
    count, and the energy is the one the unpatched build gave (#269).
    """
    import vibeqc as vq

    molecule = vq.Molecule(
        [
            vq.Atom(16, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 1.815, 1.425]),
            vq.Atom(1, [0.0, -1.815, 1.425]),
        ],
        charge=0,
        multiplicity=1,
    )
    result = vq.run_job(molecule, basis="lanl2dz", method="rhf", output=tmp_path / "h2s")
    captured = capfd.readouterr()
    assert result.converged
    assert result.energy == pytest.approx(-11.0094523536, abs=1e-9)
    assert "Quadrature failed" not in captured.out + captured.err
    lines = [
        line for line in (tmp_path / "h2s.out").read_text().splitlines()
        if "libecpint radial quadrature" in line
    ]
    assert len(lines) == 1, lines
    assert lines[0].startswith("[C++/ecp] ") and "integrate_small" in lines[0]
