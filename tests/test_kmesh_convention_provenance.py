"""KPOINT-CONVENTION-UNPRINTED regressions.

``kpoints=(N, N, N)`` names a different Brillouin-zone sampling in
different codes, and printing the tuple alone lets a cross-code
comparison silently compare different meshes. These tests pin both halves
of the fix: the convention vibe-qc actually builds, and the fact that the
``.out`` states it.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.output import Level, OutputChannel
from vibeqc.periodic_runner import (
    KMESH_CONVENTION_GAMMA,
    KMESH_CONVENTION_SHIFTED,
    KMESH_CONVENTION_UNKNOWN,
    write_kmesh_convention,
    write_kmesh_line,
)


def _cubic_system(a_bohr: float = 6.0):
    return vq.PeriodicSystem(
        3, np.eye(3) * a_bohr, [vq.Atom(1, [0.0, 0.0, 0.0])]
    )


def _frac_wrapped(system, kpoints_cart):
    a_mat = np.asarray(system.lattice, dtype=float)
    b_mat = 2.0 * np.pi * np.linalg.inv(a_mat).T
    frac = np.asarray(kpoints_cart, dtype=float) @ np.linalg.inv(b_mat)
    return np.round((frac + 0.5) % 1.0 - 0.5, 8) + 0.0


def test_vibeqc_mesh_is_gamma_centred_not_ase_shifted():
    """The convention divergence is even-N only -- pin both halves.

    vibe-qc builds {0, 1/N, ..., (N-1)/N} per axis; ASE/GPAW build the
    classical mesh offset by half a step. For odd N the offset wraps onto
    the same lattice (identical sets); for even N it lands exactly between
    vibe-qc's points (disjoint sets). Measured 2026-08-02.

    This is the property that makes the trap survive testing: a cross-code
    check validated at (3,3,3) passes and says nothing about (4,4,4).
    """
    ase_mp = pytest.importorskip(
        "ase.dft.kpoints", reason="ASE is an optional extra"
    ).monkhorst_pack

    system = _cubic_system()
    for n, expect_shared in ((2, 0), (3, 27), (4, 0), (5, 125), (6, 0)):
        mesh = (n, n, n)
        ours = {
            tuple(r)
            for r in _frac_wrapped(system, vq.monkhorst_pack(system, mesh).kpoints)
        }
        theirs = {tuple(r) for r in np.round(
            (np.asarray(ase_mp(mesh)) + 0.5) % 1.0 - 0.5, 8) + 0.0}
        assert len(ours) == n**3
        assert len(ours & theirs) == expect_shared, (
            f"mesh {mesh}: expected {expect_shared} shared k-points with "
            f"ASE, got {len(ours & theirs)}. Odd N must agree exactly and "
            f"even N must be disjoint; a change here means the mesh "
            f"convention moved and every cross-code reference is suspect."
        )

    # Gamma is in vibe-qc's set for every N -- the defining property of the
    # gamma-centred convention, and false for the ASE mesh at even N.
    for n in (2, 3, 4):
        ours = _frac_wrapped(
            _cubic_system(), vq.monkhorst_pack(_cubic_system(), (n, n, n)).kpoints
        )
        assert np.any(np.all(np.abs(ours) < 1e-12, axis=1)), (
            f"mesh ({n},{n},{n}) does not contain Gamma; "
            f"KMESH_CONVENTION_GAMMA claims {KMESH_CONVENTION_GAMMA!r}"
        )


def _capture(fn, level=Level.STANDARD):
    import io

    buf = io.StringIO()
    with OutputChannel(stream=buf, level=level):
        fn()
    return buf.getvalue()


def test_out_states_the_kmesh_convention():
    """The mesh line must never appear without the convention beside it."""
    system = _cubic_system()
    text = _capture(lambda: write_kmesh_line((4, 4, 4), system=system))
    assert "kpoints" in text and "(4, 4, 4)" in text
    assert KMESH_CONVENTION_GAMMA in text, (
        f"the mesh line must carry its convention; got:\n{text}"
    )
    assert "even N" in text, (
        "the convention line must say the ASE/GPAW mesh is disjoint for "
        f"even N -- that is the actionable half; got:\n{text}"
    )


def test_convention_is_measured_from_the_mesh_not_assumed():
    """Two identical-looking (2,2,2) specs get different, truthful labels.

    `kpoints=` accepts both the mesh tuple (always gamma-centred) and a
    `KPoints` built with the classical auto-shift (shifted at even N). If
    the label were hardcoded, the second caller's .out would be
    confidently wrong -- worse than printing nothing, since the whole
    point is to make cross-code comparison trustworthy.
    """
    from vibeqc.kpoints import KPoints

    system = _cubic_system()
    mesh = (2, 2, 2)

    gamma = _capture(lambda: write_kmesh_line(mesh, system=system))
    assert KMESH_CONVENTION_GAMMA in gamma
    assert KMESH_CONVENTION_SHIFTED not in gamma

    classical = KPoints.monkhorst_pack(system, mesh).to_bloch_kmesh()
    assert list(classical.is_shift) == [1, 1, 1], (
        "precondition: KPoints.monkhorst_pack must auto-shift at even N"
    )
    shifted = _capture(
        lambda: write_kmesh_line(
            mesh, system=system, kpoints_cart=classical.kpoints
        )
    )
    assert KMESH_CONVENTION_SHIFTED in shifted, (
        f"a shifted mesh must not be labelled gamma-centred; got:\n{shifted}"
    )
    assert "Gamma is NOT sampled" in shifted


def test_unresolvable_mesh_says_so_rather_than_guessing():
    """With no k-list to measure, report 'unresolved', never a guess."""
    text = _capture(lambda: write_kmesh_line((4, 4, 4)))
    assert KMESH_CONVENTION_UNKNOWN in text
    assert KMESH_CONVENTION_GAMMA not in text
    assert KMESH_CONVENTION_SHIFTED not in text


def test_kmesh_convention_line_is_shared_not_duplicated():
    """Routes reporting a k-point count state the same convention."""
    system = _cubic_system()
    from_line = _capture(lambda: write_kmesh_line((4, 4, 4), system=system))
    standalone = _capture(
        lambda: write_kmesh_convention(KMESH_CONVENTION_GAMMA)
    )
    assert standalone.strip() in from_line, (
        "write_kmesh_line must emit exactly the shared convention string, "
        "so the two cannot drift apart"
    )


def test_kmesh_klist_is_verbose_only_and_reconstructable():
    """The resolved k-list reaches the .out at VERBOSE, not STANDARD."""
    system = _cubic_system()
    kmesh = vq.monkhorst_pack(system, (2, 2, 2))

    def emit():
        write_kmesh_line(
            (2, 2, 2), system=system, kpoints_cart=kmesh.kpoints
        )

    standard = _capture(emit, level=Level.STANDARD)
    verbose = _capture(emit, level=Level.VERBOSE)

    assert "k-points resolved  = 8" in standard
    assert "fractional" not in standard, (
        f"the full k-list must not land in a default .out; got:\n{standard}"
    )
    assert "fractional" in verbose

    # Every fractional coordinate must be recoverable from the verbose text.
    got = [
        tuple(round(float(x), 8) + 0.0 for x in line.split()[1:4])
        for line in verbose.splitlines()
        if line.strip() and line.split()[0].isdigit()
    ]
    expected = {tuple(r) for r in _frac_wrapped(system, kmesh.kpoints)}
    assert len(got) == 8
    assert set(got) == expected, (
        "the printed k-list must reproduce the mesh actually used"
    )


def test_kmesh_provenance_never_breaks_a_job():
    """Provenance output is best-effort: a bad lattice must not raise."""

    class _Broken:
        lattice = np.zeros((3, 3))  # singular -> inv() raises

    text = _capture(
        lambda: write_kmesh_line(
            (2, 2, 2), system=_Broken(), kpoints_cart=np.zeros((8, 3))
        )
    )
    # Degrades to mesh + "unresolved" rather than propagating -- and must
    # not invent a convention it could not measure.
    assert KMESH_CONVENTION_UNKNOWN in text
    assert KMESH_CONVENTION_GAMMA not in text
    assert "k-points resolved" not in text
