"""The post-SCF phase of the Gamma-CCM runner arms, and an honest ``Job total``.

EXPERIMENTAL (AICCM / Gamma-CCM research line; T3, never release blocking).

Everything pinned here is output-stage behaviour. None of it can move an
energy, and the end-to-end cases assert exactly that.

**What was wrong.** ``run_periodic_job`` fills the QVF's ``dos/`` sections
from one of two builders. The accepted-state adapter
(:func:`vibeqc.periodic_gdf_properties.gdf_properties_from_result`) projects
the eigenpairs the SCF returned. The generic payload instead assembles its
*own* Ewald/HF-like operator at a hard-wired 18-bohr cutoff and diagonalises
it on a hard-wired 8x8x8 mesh. Every Gamma-CCM route used to take the second
path, which is wrong twice over:

* it is not the operator those runs converged. Ruling R1 (2026-08-21,
  :mod:`vibeqc.periodic.ccm.route`) makes Gamma-CCM the neutral
  finite-BvK-torus construction, whose exchange channel carries the q=0
  Madelung seam ``K -> K + xi_N S D S``. The rebuild has no seam, and fits
  its Coulomb operator at its own cutoff rather than from the torus's fitted
  cderi -- the same objection the tree already records for BIPOLE and
  AICCM2026DEV-B;
* it is a four-centre real-space Fock build, and it dominated these jobs.
  Profiled with cProfile over ``run_periodic_job`` on LiH/STO-3G (1,1,1),
  ``variant="neutral-bloch"``: 1322 s of post-SCF work, of which one
  ``build_fock_2e_real_space`` call was 1306.995 s (98.9 %). It never grew
  with the BvK mesh, because neither the 18-bohr cutoff nor the 8x8x8 DOS
  mesh is a function of it.

**What it now does.** ``variant="neutral-bloch"`` returns the production
multi-k GDF driver's own accepted state, so it is served by the accepted-state
adapter and *keeps* its DOS. ``variant="real-gamma"`` and the four-centre
route have no such adapter, so they skip the payload with a stated reason in
``property_hamiltonian`` rather than answering for an operator they never
solved.

**And ``Job total`` covers the whole job.** It used to be printed straight
after the SCF, so the 1322 s above sat entirely outside the number the
``.out`` called the total: LiH (1,1,1) ``variant="real-gamma"`` reported
``Job total 156.168`` against ``wall_seconds 1480.118``.
"""

from __future__ import annotations

import re
import tomllib
import zipfile
from pathlib import Path

import numpy as np
import pytest

from vibeqc import BasisSet
from vibeqc.periodic_jk_method import PeriodicJKMethod
from vibeqc.periodic_runner import (
    _QVF_ACCEPTED_STATE_PROPERTY_ROUTES,
    _qvf_periodic_property_payload_skip_reason,
    _qvf_periodic_property_payload_supported,
    run_periodic_job,
)

pytestmark = pytest.mark.filterwarnings("ignore")

ANG = 1.0 / 0.529177210903

#: The two producers of the neutral finite-BvK-torus Hamiltonian (ruling R1).
_NEUTRAL_TORUS = (
    PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA,
    PeriodicJKMethod.NEUTRAL_BLOCH,
)


def _h2_cube6():
    """H2 in a 6-bohr cube, d = 1.4 bohr.

    The compact measured-3-D control of the AICCM campaign (its
    ``h2_cube6`` geometry). Chosen here because it is the cheapest cell on which the
    Bloch arm runs at all: on LiH the same arm needs ~20000 s of SCF on this
    tree, for reasons that belong to the SR/LR fit, not to this change.
    """
    from vibeqc import Atom, PeriodicSystem

    return PeriodicSystem(
        3,
        np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])],
        0,
        1,
    )


def _lih():
    """LiH, rocksalt fcc primitive cell, a = 4.0840 A (the campaign's cell)."""
    from vibeqc import Atom, PeriodicSystem

    a = 4.0840 * ANG
    lattice = 0.5 * a * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    return PeriodicSystem(
        3,
        lattice,
        [Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.5 * a, 0.5 * a, 0.5 * a])],
        0,
        1,
    )


def _run(system, variant, stem, mesh=(1, 1, 1)):
    basis = BasisSet(system.unit_cell_molecule(), "sto-3g")
    return run_periodic_job(
        system, basis, method="aiccm", variant=variant,
        aiccm_lattice_extension=mesh, output=str(stem), progress=False,
        citations=False, write_xyz_file=False, write_cif_file=False,
        write_xsf_structure_file=False, write_molden_file=False,
    )


def _out_timing(stem: Path, label: str):
    text = Path(str(stem) + ".out").read_text()
    match = re.search(rf"^\s*{re.escape(label)}\s+([0-9.]+)\s*$", text, re.M)
    return float(match.group(1)) if match else None


def _manifest(stem: Path) -> dict:
    return tomllib.loads(Path(str(stem) + ".system").read_text())


def _qvf_members(stem: Path, prefix: str) -> list[str]:
    with zipfile.ZipFile(str(stem) + ".qvf") as archive:
        return sorted(n for n in archive.namelist() if n.startswith(prefix))


def _assert_job_total_is_last_and_covers_the_job(stem: Path, wall: float):
    """``Job total`` is written once, at the end, and accounts for the job.

    Ordering is the whole point: a phase printed after ``Job total`` is a
    phase outside it, which is how a 1322 s payload stayed invisible. Asserted
    from the runs the end-to-end cases already make rather than from a job of
    its own -- a third run of this cell costs minutes and proves nothing new.
    """
    text = Path(str(stem) + ".out").read_text()
    assert text.count("Job total") == 1
    assert text.index("SCF total") < text.index("Post-SCF and output")
    assert text.index("Post-SCF and output") < text.index("Job total")
    tail = text[text.index("Job total"):]
    for marker in ("Population dump", "Timings", "SCF total", "References"):
        assert marker not in tail, f"{marker!r} is printed after Job total"

    job = _out_timing(stem, "Job total")
    scf = _out_timing(stem, "SCF total")
    post = _out_timing(stem, "Post-SCF and output")
    assert job is not None and scf is not None and post is not None
    assert post == pytest.approx(job - scf, abs=0.05)
    # What is allowed to sit outside Job total is the QVF archive write and
    # the manifest record: 0.47 s on LiH (1,1,1).
    assert 0.0 <= wall - job <= max(5.0, 0.1 * job), (
        f"wall_seconds {wall:.1f} s against .out Job total {job:.1f} s: "
        f"{wall - job:.1f} s of the job is outside the number the .out calls "
        "the total"
    )


# --------------------------------------------------------------- routing --
# Cheap and deterministic: these decide which builder a route reaches, which
# is the whole of the cost fix. They hold under any machine load.

def test_neutral_bloch_is_served_by_the_accepted_state_adapter():
    """The Bloch producer projects the state it converged, and keeps its DOS.

    It is the production multi-k GDF driver under another front door
    (``periodic.ccm.ri.run_ccm_rhf_gdf`` into ``run_krhf_periodic_gdf``), and
    its runner adapter returns that driver's own result object, so the per-k
    eigenpairs, Fock, density and overlap the adapter needs are all present
    and carry the exchange q=0 seam.
    """
    assert PeriodicJKMethod.NEUTRAL_BLOCH in _QVF_ACCEPTED_STATE_PROPERTY_ROUTES
    assert PeriodicJKMethod.GDF in _QVF_ACCEPTED_STATE_PROPERTY_ROUTES


@pytest.mark.parametrize(
    "route",
    [
        PeriodicJKMethod.AICCM2026DEV_A,
        PeriodicJKMethod.AICCM2026DEV_A_REAL_GAMMA,
        PeriodicJKMethod.NEUTRAL_BLOCH,
    ],
    ids=lambda r: r.value,
)
def test_no_ccm_route_accepts_the_generic_property_rebuild(route):
    """No cyclic-cluster route rebuilds its own property Hamiltonian.

    ``NEUTRAL_BLOCH`` does not reach this predicate from ``run_periodic_job``
    (the adapter takes it first); it is asserted here so the branch stays
    fail-closed if that routing is ever changed.
    """
    reason = _qvf_periodic_property_payload_skip_reason(route)
    assert reason is not None, (
        f"{route.value} would rebuild a fixed-cutoff Ewald/HF-like operator "
        "instead of using the operator it converged"
    )
    assert not _qvf_periodic_property_payload_supported(route)
    if route in _NEUTRAL_TORUS:
        # The reason has to name the defect, not just decline: the seam is
        # what makes the rebuilt operator a different Hamiltonian.
        assert "Madelung seam" in reason
        assert "finite-BvK-torus" in reason


def test_the_routes_that_did_get_the_payload_still_do():
    """The fix is route-scoped: it must not widen into unrelated routes."""
    for route in (
        PeriodicJKMethod.GPW,
        PeriodicJKMethod.GAPW,
        PeriodicJKMethod.RIJCOSX,
        PeriodicJKMethod.FFT_POISSON,
    ):
        assert _qvf_periodic_property_payload_skip_reason(route) is None
    # ... and the two that were already excluded stay excluded.
    for route in (PeriodicJKMethod.BIPOLE, PeriodicJKMethod.AICCM2026DEV_B):
        assert _qvf_periodic_property_payload_skip_reason(route) is not None


# ------------------------------------------------------------ end to end --

def test_real_gamma_lih_wall_tracks_its_scf_and_the_energy_is_unchanged(tmp_path):
    """LiH (1,1,1): ``wall_seconds`` within 2x the SCF wall, energy to the
    last digit, and no four-centre rebuild after the SCF.

    Measured on this cell with ``OMP_NUM_THREADS=2`` on a shared machine at a
    load of 50 to 120 (vibe-qc 0.17.6.dev0):

    ==========================  ==========  =========
    ..                          before      after
    ==========================  ==========  =========
    SCF total (s)                  155.828    110.552
    ``wall_seconds`` (s)          1480.118    111.233
    hidden after ``Job total``    1323.949      0.467
    wall / SCF                        9.50      1.006
    ==========================  ==========  =========

    The wall ratio is the assertion the speed-up is accepted on, but it is a
    timing, so the deterministic half of the pin is ``calls == 0``: the
    payload's four-centre build must not be entered at all.
    """
    import vibeqc._vibeqc_core as core

    calls = []
    original = core.build_fock_2e_real_space

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    stem = tmp_path / "lih_real_gamma"
    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(core, "build_fock_2e_real_space", counting)
        result = _run(_lih(), "real-gamma", stem)
    finally:
        monkey.undo()

    assert result.converged
    # The energy of this cell and producer, to every digit it prints. The
    # change is output-stage only, so any movement here is a real regression.
    assert float(result.energy) == pytest.approx(-8.3350437600853, abs=1e-12)

    assert not calls, (
        f"the generic QVF property rebuild ran {len(calls)} four-centre "
        "real-space Fock build(s) on a Gamma-CCM route"
    )

    tables = _manifest(stem)
    wall = float(tables["run"]["wall_seconds"])
    scf = _out_timing(stem, "SCF total")
    assert scf is not None

    assert wall <= 2.0 * scf, (
        f"wall_seconds {wall:.1f} s is more than 2x the SCF wall {scf:.1f} s; "
        "a post-SCF analysis has taken over the job again"
    )
    _assert_job_total_is_last_and_covers_the_job(stem, wall)
    # Declined rather than silently absent, and the reason names the operator.
    hamiltonian = tables["run"].get("property_hamiltonian", "")
    assert hamiltonian.startswith("omitted ("), hamiltonian
    assert "Madelung seam" in hamiltonian
    assert _qvf_members(stem, "dos/") == []


def test_neutral_bloch_keeps_its_dos_and_gets_it_from_the_accepted_state(tmp_path):
    """The Bloch arm still ships DOS/PDOS -- now from the state it converged.

    H2 in a 6-bohr cube rather than LiH: the Bloch producer's SR/LR fit needs
    ~20000 s of SCF on LiH on this tree, which is prompt 2's subject and not
    this change's. The routing under test is the same on either cell.

    Measured here (same machine and load as above): ``wall_seconds``
    67.417 -> 56.977 s, of which the part hidden after ``Job total``
    6.863 -> 0.257 s, energy unchanged at -1.2385562902099365.
    """
    stem = tmp_path / "h2cube6_neutral_bloch"
    result = _run(_h2_cube6(), "neutral-bloch", stem)

    assert result.converged
    assert float(result.energy) == pytest.approx(
        -1.2385562902099365, abs=1e-12
    )

    tables = _manifest(stem)
    assert tables["run"]["property_hamiltonian"] == "accepted-scf"
    # The artefact survives the fix: this route did not lose its DOS, it
    # stopped computing it from an operator it had not solved.
    assert "dos/total.bin" in _qvf_members(stem, "dos/")
    assert "dos/projections.bin" in _qvf_members(stem, "dos/")

    wall = float(tables["run"]["wall_seconds"])
    scf = _out_timing(stem, "SCF total")
    assert scf is not None
    assert wall <= 2.0 * scf, f"wall_seconds {wall:.1f} s against SCF {scf:.1f} s"
    _assert_job_total_is_last_and_covers_the_job(stem, wall)
