"""Guardrails and ergonomic fixes for common wrong-input mistakes.

Covers the input-validation hardening + several release-paper bug-report
rough edges:

* wrong-runner calls (Molecule -> run_periodic_job, PeriodicSystem ->
  run_job) fail fast with an actionable message rather than a deep
  AttributeError;
* odd-electron closed-shell periodic calls suggest UHF/UKS (bug 23);
* semiempirical methods do not require an explicit basis (bug 18);
* Atom exposes ASE-parity .symbol / .number (bug 16);
* RIJCOSX / density-fitted SCF auto-resolves the JK aux basis (bug 21);
* Mermin smearing spellings resolve to their canonical flavor (bug 9);
* extra libxc functional spellings resolve (bug 22).
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import Atom, BasisSet, Functional, Molecule, run_job
from vibeqc.smearing.options import SmearingOptions

ANGSTROM_TO_BOHR = 1.8897259886


def _h2():
    return Molecule(
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
        charge=0,
        multiplicity=1,
    )


def _h2o():
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.5, -1.1]),
            Atom(1, [0.0, -1.5, -1.1]),
        ],
        charge=0,
        multiplicity=1,
    )


class TestWrongRunnerGuards:
    def test_periodic_runner_rejects_molecule(self):
        from vibeqc import run_periodic_job

        with pytest.raises(TypeError, match="Molecule.*run_job|run_job"):
            run_periodic_job(_h2(), "sto-3g", method="RHF")

    def test_molecular_runner_rejects_periodic_system(self, tmp_path):
        lat = np.diag([6.0, 6.0, 6.0])
        sysp = vq.PeriodicSystem(3, lat, [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])])
        with pytest.raises(TypeError, match="periodic|run_periodic_job"):
            run_job(sysp, basis="sto-3g", method="rhf",
                    output=str(tmp_path / "x"))


class TestOddElectronPeriodicGuard:
    def test_odd_electron_rhf_suggests_uhf(self):
        from vibeqc import run_periodic_job

        # single Cu atom in a box: 29 electrons, odd.
        lat = np.diag([10.0, 10.0, 10.0])
        sysp = vq.PeriodicSystem(3, lat, [Atom(29, [0, 0, 0])])
        with pytest.raises(ValueError, match="odd electron count|UHF"):
            run_periodic_job(sysp, "sto-3g", method="RHF")


class TestAtomAliases:
    def test_number_alias(self):
        assert Atom(29, [0, 0, 0]).number == 29

    def test_symbol_lookup(self):
        assert Atom(29, [0, 0, 0]).symbol == "Cu"
        assert Atom(8, [0, 0, 0]).symbol == "O"
        assert Atom(1, [0, 0, 0]).symbol == "H"

    def test_symbol_out_of_range_raises(self):
        with pytest.raises(Exception):
            _ = Atom(200, [0, 0, 0]).symbol


class TestSemiempiricalNoBasis:
    @pytest.mark.parametrize("method", ["gfn2-xtb", "gfn2_xtb", "scc-dftb"])
    def test_semiempirical_runs_without_basis(self, tmp_path, method):
        # ethane-like C-C + H cage; the release-paper report saw a spurious
        # "basis is required" on multi-carbon molecules.
        b = ANGSTROM_TO_BOHR
        c2h6 = Molecule(
            [
                Atom(6, [0, 0, 0]),
                Atom(6, [0, 0, 1.5 * b]),
                Atom(1, [1.0 * b, 0, -0.3 * b]),
                Atom(1, [-0.5 * b, 0.87 * b, -0.3 * b]),
                Atom(1, [-0.5 * b, -0.87 * b, -0.3 * b]),
                Atom(1, [1.0 * b, 0, 1.8 * b]),
                Atom(1, [-0.5 * b, 0.87 * b, 1.8 * b]),
                Atom(1, [-0.5 * b, -0.87 * b, 1.8 * b]),
            ],
            charge=0,
            multiplicity=1,
        )
        res = run_job(c2h6, method=method, output=str(tmp_path / "se"))
        assert res is not None


class TestRIJCOSXAuxAutoResolve:
    def test_cosx_without_aux_basis_resolves(self, tmp_path):
        from vibeqc import RKSOptions

        opts = RKSOptions()
        opts.cosx = True
        opts.functional = "pbe"
        # No aux_basis set: run_job should auto-resolve the JK aux instead
        # of the C++ driver rejecting the run.
        res = run_job(
            _h2o(),
            basis="def2-svp",
            method="rks",
            functional="pbe",
            rks_options=opts,
            output=str(tmp_path / "rijcosx"),
        )
        assert res.converged

    def test_density_fit_without_aux_basis_resolves(self, tmp_path):
        from vibeqc import RHFOptions

        opts = RHFOptions()
        opts.density_fit = True
        res = run_job(
            _h2o(),
            basis="def2-svp",
            method="rhf",
            rhf_options=opts,
            output=str(tmp_path / "rij"),
        )
        assert res.converged


class TestMerminSmearing:
    @pytest.mark.parametrize(
        "name,canonical",
        [
            # "mermin" is a first-class flavor since 38d538799 (Mermin
            # free-energy functional A = E - TS, own citation route) --
            # not a Fermi-Dirac alias. "meremin" is its spelling alias;
            # "fermi"/"fd" resolve to the Fermi-Dirac flavor.
            ("mermin", "mermin"),
            ("meremin", "mermin"),
            ("fermi", "fermi-dirac"),
            ("fd", "fermi-dirac"),
        ],
    )
    def test_flavor_spellings_resolve(self, name, canonical):
        opts = SmearingOptions(temperature=0.005, flavor=name)
        assert opts.flavor == canonical

    def test_unknown_flavor_lists_aliases(self):
        with pytest.raises(ValueError, match="mermin|fermi-dirac"):
            SmearingOptions(temperature=0.005, flavor="bogus")


class TestFunctionalAliases:
    @pytest.mark.parametrize(
        "name,is_hybrid,exx",
        [
            ("revtpss", False, 0.0),
            ("rev-tpss", False, 0.0),
            ("pbesol", False, 0.0),
            ("pbe_sol", False, 0.0),
            ("bhlyp", True, 0.5),
            ("bhandhlyp", True, 0.5),
            ("wb97", True, 0.0),  # RSH: HF fraction lives in the range split
        ],
    )
    def test_functional_name_resolves(self, name, is_hybrid, exx):
        f = Functional(name)
        assert bool(f.is_hybrid) == is_hybrid
        assert f.hf_exchange_fraction == pytest.approx(exx, abs=1e-9)
