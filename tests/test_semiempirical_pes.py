"""PES-shape regressions for the semiempirical platform (DFTB, PM6, OMx).

Every test here scans a bond coordinate and asserts the well EXISTS and sits
in a physical window.  History shows why this file must stay: the PM6/OMx
PES was dissociative until 2026-06 (five bugs, see
the semiempirical PES hardening pass), and commit 65ed3112 later re-broke it
silently by stubbing out the n-dependent STO overlap "to fix a build break"
— nothing failed because no committed test pinned the PES shape.  These
scans are that pin.  Single points are sub-millisecond, so the whole file
runs in seconds.

The DFTB scans below were added 2026-07-09 after the vibe-view live-opt
chat found SCC-DFTB water dissociating DOWNHILL (energy monotonically
decreasing with stretch, no repulsive wall): a wrong-sign SCC Hamiltonian
correction made charge separation self-amplifying, and core electrons were
counted into the valence band filling.  No committed test scanned a DFTB
bond coordinate, so both bugs shipped.

Reference bond lengths (bohr):
  H2O r(O-H) exp 1.81 (0.957 A); PM6 publishes 0.949 A (Stewart, J. Mol.
  Model. 13, 1173 (2007), Table 4); OM2/OM3 reproduce ~0.96 A (Dral et al.,
  J. Chem. Theory Comput. 12, 1097 (2016)).
  N2 exp 2.074;  CO exp 2.132;  CH4 r(C-H) exp 2.05.

The windows below are deliberately generous: this codebase's NDDO platform
evaluates the two-centre Coulomb class with STO densities + Klopman-Ohno
scaling instead of the published Gaussian integrals (documented in
omx_fock.cpp / pm6_fock.cpp), which shifts minima by up to ~0.2 bohr.  The
windows catch the failure modes that matter — collapse (min at the scan's
short edge), dissociation (min at the long edge), or a wandering minimum —
without pinning the residual model bias.
"""

from __future__ import annotations

import math

import pytest
from vibeqc import Atom, Molecule
from vibeqc._vibeqc_core import semiempirical as _se


def water(r_oh: float) -> Molecule:
    a = math.radians(104.5 / 2)
    y = r_oh * math.sin(a)
    z = r_oh * math.cos(a)
    return Molecule(
        [Atom(8, [0, 0, 0]), Atom(1, [0, y, z]), Atom(1, [0, -y, z])], 0, 1
    )


def diatomic(z1: int, z2: int, r: float) -> Molecule:
    return Molecule([Atom(z1, [0, 0, 0]), Atom(z2, [r, 0, 0])], 0, 1)


def methane(r_ch: float) -> Molecule:
    d = r_ch / math.sqrt(3.0)
    return Molecule(
        [
            Atom(6, [0, 0, 0]),
            Atom(1, [d, d, d]),
            Atom(1, [-d, -d, d]),
            Atom(1, [d, -d, -d]),
            Atom(1, [-d, d, -d]),
        ],
        0,
        1,
    )


def scan_minimum(run, geoms):
    """Run `run` over (r, Molecule) pairs; assert convergence; return r_min
    and the (r, E) list."""
    es = []
    for r, mol in geoms:
        res = run(mol)
        assert res.converged, f"SCF not converged at r={r}"
        es.append((r, float(res.energy)))
    r_min = min(es, key=lambda t: t[1])[0]
    return r_min, es


def assert_interior_minimum(r_min, es, lo, hi, tag):
    rs = [r for r, _ in es]
    assert r_min != min(rs), f"{tag}: collapse — minimum at short scan edge {r_min}"
    assert r_min != max(rs), f"{tag}: dissociative — minimum at long scan edge {r_min}"
    assert lo <= r_min <= hi, f"{tag}: minimum at {r_min} bohr, outside [{lo}, {hi}]"


# ---------------------------------------------------------------------------
# DFTB0 / SCC-DFTB (default in-house parameter set)
# ---------------------------------------------------------------------------


class _ConvergedShim:
    """Adapt DFTB0Result (no .converged field) to scan_minimum."""

    def __init__(self, result):
        self.energy = result.energy
        self.converged = True


class TestDFTBWellShapes:
    """Water O-H well must exist for both DFTB flavours.

    The in-house R⁻¹² repulsive + Wolfsberg-Helmholtz H⁰ give a shallow,
    slightly short well — the generous window pins existence and rough
    position, i.e. the collapse/dissociation failure modes, not the
    residual parameter-set bias."""

    @staticmethod
    def _params():
        from vibeqc.semiempirical.parameters import default_parameters

        return default_parameters()

    RS = (1.30, 1.50, 1.70, 1.81, 2.00, 2.30, 2.70, 3.20, 3.80)
    # SCC scan stops at 3.02 bohr (1.6 Å): beyond that the stretched
    # O-H HOMO-LUMO gap closes and integer-occupation SCC has no stable
    # fixed point (honest converged=False; Fermi smearing is the future
    # fix). The convergent window comfortably brackets the well.
    RS_SCC = (1.42, 1.60, 1.81, 2.08, 2.46, 2.80, 3.02)

    def test_dftb0_water_oh_well(self):
        params = self._params()
        r_min, es = scan_minimum(
            lambda mol: _ConvergedShim(_se.run_dftb0(mol, params)),
            [(r, water(r)) for r in self.RS],
        )
        assert_interior_minimum(r_min, es, 1.50, 2.30, "DFTB0 water")

    def test_scc_dftb_water_oh_well(self):
        params = self._params()
        r_min, es = scan_minimum(
            lambda mol: _se.run_scc_dftb(mol, params),
            [(r, water(r)) for r in self.RS_SCC],
        )
        assert_interior_minimum(r_min, es, 1.50, 2.30, "SCC-DFTB water")

    def test_scc_dftb_water_has_repulsive_wall_and_binding(self):
        """Compressed and stretched geometries must both sit above the
        equilibrium region (pre-fix: E fell monotonically with stretch)."""
        params = self._params()
        e = {r: float(_se.run_scc_dftb(water(r), params).energy) for r in self.RS_SCC}
        e_min = min(e.values())
        assert e[self.RS_SCC[0]] > e_min + 1e-3, "no repulsive wall at short r"
        assert e[self.RS_SCC[-1]] > e_min + 1e-3, "no binding: stretched below minimum"


# ---------------------------------------------------------------------------
# PM6
# ---------------------------------------------------------------------------


class TestPM6WellShapes:
    @staticmethod
    def _params():
        from vibeqc.semiempirical.methods.pm6_params import load_pm6_params

        return load_pm6_params()

    def _run(self, mol):
        return _se.nddo.run_pm6(mol, self._params(), 500)

    def test_water_oh_well(self):
        rs = (1.50, 1.65, 1.81, 2.00, 2.20, 2.40, 2.80, 3.50)
        r_min, es = scan_minimum(self._run, [(r, water(r)) for r in rs])
        assert_interior_minimum(r_min, es, 1.81, 2.40, "PM6 water")

    def test_n2_well(self):
        rs = (1.70, 1.90, 2.07, 2.30, 2.60, 3.20)
        r_min, es = scan_minimum(self._run, [(r, diatomic(7, 7, r)) for r in rs])
        assert_interior_minimum(r_min, es, 1.90, 2.60, "PM6 N2")

    def test_co_well(self):
        rs = (1.70, 1.90, 2.13, 2.30, 2.60, 3.20)
        r_min, es = scan_minimum(self._run, [(r, diatomic(6, 8, r)) for r in rs])
        assert_interior_minimum(r_min, es, 1.90, 2.60, "PM6 CO")

    def test_ch4_well(self):
        rs = (1.70, 1.90, 2.06, 2.20, 2.40, 2.80)
        r_min, es = scan_minimum(self._run, [(r, methane(r)) for r in rs])
        assert_interior_minimum(r_min, es, 1.90, 2.40, "PM6 CH4")

    def test_geometry_optimization_sane(self, tmp_path):
        """BFGS from a perturbed start lands both O-H bonds in the well window."""
        pytest.importorskip("ase")
        import numpy as np
        from ase import Atoms
        from ase.optimize import BFGSLineSearch
        from ase.units import Bohr

        from vibeqc.runner import _make_semiempirical_ase_calculator

        mol = water(2.05)  # start off-minimum
        atoms = Atoms(
            numbers=[a.Z for a in mol.atoms],
            positions=[[c * Bohr for c in a.xyz] for a in mol.atoms],
        )
        atoms.calc = _make_semiempirical_ase_calculator(mol, "pm6")
        BFGSLineSearch(atoms, logfile=None).run(fmax=0.1, steps=40)
        pos = atoms.get_positions() / Bohr
        # Stewart's PM6 optimum is 0.949 A = 1.793 bohr, slightly shorter
        # than the experimental 1.81-bohr value used to label the coarse
        # scan above.  Keep the optimizer guard broad enough to include the
        # published PM6 minimum while still rejecting the historical collapse.
        for ih in (1, 2):
            r_oh = float(np.linalg.norm(pos[ih] - pos[0]))
            assert 1.75 <= r_oh <= 2.45, f"PM6 opt O-H{ih} = {r_oh:.3f} bohr"


# ---------------------------------------------------------------------------
# OMx (v2 — the run_job production path)
# ---------------------------------------------------------------------------


class TestOM2WellShapes:
    # Published OM2 minima: water r(O-H) 1.81, N2 2.07, CH4 2.05 bohr
    # (Dral 2016 II / exp.).  With the documented spherical-monopole + STO
    # integral stand-ins (omx_fock.cpp header) vibe-qc's minima sit
    # systematically ~0.1-0.2 A long: water ~2.20, N2 ~2.30, CH4 ~2.20.
    # The windows pin that documented state; the interior-minimum asserts
    # are the collapse/dissociation guard.

    @staticmethod
    def _params():
        from vibeqc.semiempirical.methods.omx_params import load_om2_params

        return load_om2_params()

    def _run(self, mol):
        return _se.nddo.run_omx_v2(mol, self._params(), 500)

    def test_water_oh_well(self):
        rs = (1.60, 1.80, 2.00, 2.20, 2.40, 2.70, 3.10)
        r_min, es = scan_minimum(self._run, [(r, water(r)) for r in rs])
        assert_interior_minimum(r_min, es, 2.00, 2.40, "OM2 water")

    def test_n2_well(self):
        rs = (1.90, 2.10, 2.30, 2.50, 2.80, 3.20)
        r_min, es = scan_minimum(self._run, [(r, diatomic(7, 7, r)) for r in rs])
        assert_interior_minimum(r_min, es, 2.10, 2.50, "OM2 N2")

    def test_ch4_well(self):
        rs = (1.80, 2.00, 2.20, 2.40, 2.70, 3.00)
        r_min, es = scan_minimum(self._run, [(r, methane(r)) for r in rs])
        assert_interior_minimum(r_min, es, 2.00, 2.45, "OM2 CH4")


class TestOM3WellShapes:
    @staticmethod
    def _params():
        from vibeqc.semiempirical.methods.omx_params import load_om3_params

        return load_om3_params()

    def _run(self, mol):
        return _se.nddo.run_omx_v2(mol, self._params(), 500)

    def test_water_oh_well(self):
        # Published OM3 r(O-H) ~ 0.96 A = 1.81 bohr; ours sits ~2.35 with
        # the documented integral stand-ins (see TestOM2WellShapes note).
        rs = (1.80, 2.00, 2.20, 2.35, 2.55, 2.80, 3.20)
        r_min, es = scan_minimum(self._run, [(r, water(r)) for r in rs])
        assert_interior_minimum(r_min, es, 2.10, 2.60, "OM3 water")


class TestOM1KnownLimitation:
    """OM1 lacks its analytically-evaluated core-valence ECP (Kolb & Thiel
    1993; no semiempirical ECP parameters are published for OM1).  Without
    that repulsion the O-H well sits ~0.6 bohr short.  These tests pin the
    documented behaviour: a well EXISTS (no dissociation), it is short, and
    run_job warns.  When the analytic OM1 ECP lands, tighten the window to
    [1.70, 1.95] and drop the warning test."""

    @staticmethod
    def _params():
        from vibeqc.semiempirical.methods.omx_params import load_om1_params

        return load_om1_params()

    def _run(self, mol):
        return _se.nddo.run_omx_v2(mol, self._params(), 500)

    def test_water_well_exists_but_short(self):
        rs = (0.90, 1.05, 1.20, 1.35, 1.50, 1.81, 2.20)
        r_min, es = scan_minimum(self._run, [(r, water(r)) for r in rs])
        assert_interior_minimum(r_min, es, 1.00, 1.50, "OM1 water (documented short)")

    def test_run_job_warns_om1_only(self, tmp_path):
        import warnings

        from vibeqc import run_job
        from vibeqc.semiempirical import NDDOExperimentalWarning

        mol = water(1.81)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            run_job(mol, method="om1", output=str(tmp_path / "om1"))
        assert any(issubclass(w.category, NDDOExperimentalWarning) for w in rec)

        for method in ("pm6", "om2", "om3"):
            with warnings.catch_warnings(record=True) as rec:
                warnings.simplefilter("always")
                run_job(mol, method=method, output=str(tmp_path / method))
            assert not any(
                issubclass(w.category, NDDOExperimentalWarning) for w in rec
            ), f"{method} should not emit NDDOExperimentalWarning"
