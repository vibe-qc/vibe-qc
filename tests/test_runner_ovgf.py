"""End-to-end ``run_job`` exposure of OVGF / GF2 Green's-function IPs.

Pins the dispatcher wiring for ``method="ovgf"``: the RHF SCF runs, the diagonal
second-order self-energy (vibeqc.propagator) corrects the outer-valence
orbitals, an OVGF block is written to the ``.out`` with quasiparticle energies +
pole strengths, the result exposes ``result.ovgf``, the §8 citations fire, and
open-shell fails fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vibeqc import run_job
from vibeqc.molecule import Atom, Molecule

_A2B = 1.8897259886


def _h2o() -> Molecule:
    return Molecule(
        [Atom(8, [0.0, 0.0, 0.0]),
         Atom(1, [0.0, 0.7572 * _A2B, 0.5865 * _A2B]),
         Atom(1, [0.0, -0.7572 * _A2B, 0.5865 * _A2B])], 0, 1)


def test_run_job_ovgf_runs_and_corrects_koopmans(tmp_path: Path) -> None:
    """method="ovgf" runs RHF + the self-energy and corrects the HOMO IP toward
    experiment (Koopmans overestimates; OVGF shifts down).  6-31G is the
    smallest basis with a sensible virtual space."""
    r = run_job(_h2o(), basis="6-31g", method="ovgf",
                output=str(tmp_path / "h2o"), verbose=0)
    qps = r.ovgf
    assert len(qps) >= 2                       # outer valence + LUMO
    nocc = 5
    homo = next(q for q in qps if q.orbital == nocc - 1)
    assert homo.converged
    EV = 27.211386245988
    koopmans_ip = -homo.eps_scf * EV
    ovgf_ip = -homo.eps_qp * EV
    assert koopmans_ip == pytest.approx(13.6, abs=1.0)   # HF/6-31G Koopmans
    assert 9.0 < ovgf_ip < koopmans_ip                   # corrects downward
    assert 0.80 < homo.pole_strength <= 1.0 + 1e-12      # strong quasiparticle

    out = (tmp_path / "h2o.out").read_text()
    assert "RHF + OVGF" in out                            # header
    assert "OVGF / GF2 quasiparticle energies" in out
    assert "ionization potential (HOMO)" in out


def test_run_job_ovgf_reports_electron_affinity(tmp_path: Path) -> None:
    """The LUMO quasiparticle energy is surfaced as EA = -eps_qp(LUMO): the .out
    carries an 'electron affinity (LUMO)' line (flagged small-basis-qualitative)
    and the result exposes the LUMO correction.  Water binds no anion → EA<0."""
    r = run_job(_h2o(), basis="6-31g", method="ovgf",
                output=str(tmp_path / "h2o"), verbose=0)
    nocc = 5
    lumo = next((q for q in r.ovgf if q.orbital == nocc), None)
    assert lumo is not None and lumo.converged
    EV = 27.211386245988
    ea = -lumo.eps_qp * EV
    assert ea < 0.0                                      # water: unbound anion

    out = (tmp_path / "h2o.out").read_text()
    assert "electron affinity   (LUMO)" in out
    assert "qualitative only" in out                     # small-basis caveat


def test_run_job_ovgf_fires_citations(tmp_path: Path) -> None:
    """The §8 OVGF citations (Cederbaum 1975 + von Niessen 1984) reach the
    .references sibling — the route keys off the original method."""
    run_job(_h2o(), basis="sto-3g", method="ovgf",
            output=str(tmp_path / "h2o"), verbose=0)
    refs = (tmp_path / "h2o.references").read_text()
    assert "10.1088/0022-3700/8/2/018" in refs            # Cederbaum 1975
    assert "10.1016/0167-7977(84)90002-9" in refs         # von Niessen 1984


def test_run_job_ovgf_open_shell_runs(tmp_path: Path) -> None:
    """Open-shell OVGF runs over a UHF reference (spin-resolved spin-orbital
    self-energy): the .out carries a spin-resolved quasiparticle block + a
    first-IP line, and result.ovgf holds (spin, QuasiparticleResult) pairs."""
    oh = Molecule([Atom(8, [0, 0, 0]), Atom(1, [0, 0, 1.834])], 0, 2)  # doublet
    r = run_job(oh, basis="sto-3g", method="ovgf",
                output=str(tmp_path / "oh"), verbose=0)
    assert len(r.ovgf) >= 2
    spin, q = r.ovgf[0]
    assert spin in ("alpha", "beta")
    assert q.converged
    out = (tmp_path / "oh.out").read_text()
    assert "open-shell, spin-resolved" in out
    assert "first ionization potential" in out
    assert "alpha" in out and "beta" in out               # spin column present
