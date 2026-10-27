"""M2 rung 2b recipe (#224): each reference run recomputed from our own decks.

A stand-in engine records what it is asked for and answers with the reference
energy plus a known offset. Pinned here: the bulk is a single point at the
relaxed cell with the deck's SHRINK pair, each counterpoise atom carries its
deck's IAT label, element, NSTAR and RMAX, a keyword deck gets the bundled
pob record as its inline basis, a run without a deck is reported rather than
guessed, and the cohesive energy is assembled exactly as the reference was.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

pytest.importorskip("vibe_basis")

from vibe_basis.engine import EngineEnergy  # noqa: E402
from vibe_basis.io.cohesive_reference import (  # noqa: E402
    DeckSettings,
    ReferenceComponent,
    ReferenceSystem,
)
from vibe_basis.io.structures import STRUCTURES  # noqa: E402
from vibe_basis.pipeline import HARTREE_TO_KJ_PER_MOL  # noqa: E402

from vibeqc.basis_optimization.recipes.cohesive_reference import (  # noqa: E402
    crystal23_reference_factory,
    reproduce_reference_system,
)

K2O = STRUCTURES["K2O"]
K_FIRST = replace(K2O, crystal_asymm_unit=K2O.crystal_asymm_unit[::-1])
E = {"bulk.out": -1203.25, "k.out": -599.6, "o.out": -75.06}


def _settings(atombsse=None, keyword="pob-tzvp-rev2", inline=None, shrink=(8, 24)):
    return DeckSettings(
        spacegroup=225, lattice=(6.4,), optgeom=atombsse is None, atombsse=atombsse,
        basis_keyword=None if inline else keyword, inline_basis=inline,
        functional="r2SCAN", tolinteg=(9, 9, 9, 18, 54), shrink=shrink if atombsse is None else None,
    )


def _system(o_deck=True):
    relaxed = replace(K_FIRST, a=6.36, b=6.36, c=6.36)
    comps = [
        ReferenceComponent("bulk", "bulk.out", None, None, E["bulk.out"], "converged", E["bulk.out"],
                           _settings(), relaxed),
        ReferenceComponent("atom", "k.out", None, None, E["k.out"], "converged", E["k.out"],
                           _settings((1, 20, 7.5)), K_FIRST, Z=19),
        ReferenceComponent("atom", "o.out", None, None, E["o.out"], "crashed", E["o.out"],
                           _settings((3, 10, 7.5), inline="8 0\n19 0\n99 0\n") if o_deck else None,
                           K_FIRST if o_deck else None, Z=8),
    ]
    published = (2 * E["k.out"] + E["o.out"] - E["bulk.out"]) * HARTREE_TO_KJ_PER_MOL
    return ReferenceSystem("K2O", round(published, 1), "E_atom K2O = ...", tuple(comps), {19: 2, 8: 1})


class StandIn:
    """Reference energy + 1 mHa for every atom; records each request."""

    name = "stand-in"
    calls: list = []

    def __init__(self, run_dir, **settings):
        self.settings = settings

    def crystal_energy(self, basis_text, structure, method="rhf"):
        StandIn.calls.append(("bulk", self.settings, structure.a, method, basis_text))
        return EngineEnergy(energy=E["bulk.out"], ok=True, engine=self.name)

    def atom_energy(self, basis_text, Z, method="rhf", *, host=None, atom_label=None):
        StandIn.calls.append(("atom", self.settings, Z, atom_label, method, basis_text))
        e = E["k.out"] if Z == 19 else E["o.out"]
        return EngineEnergy(energy=e + 0.001, ok=True, engine=self.name)


def test_each_run_is_recomputed_with_its_own_deck_settings(tmp_path):
    StandIn.calls = []
    report = reproduce_reference_system(_system(), StandIn, tmp_path)
    bulk, k, o = StandIn.calls
    assert bulk[:4] == ("bulk", {"shrink": (8, 24), "optgeom": False}, 6.36, "r2scan")
    # The keyword deck got the bundled pob-TZVP-rev2 record of both elements.
    assert bulk[4].startswith("19 ") and "\n8 " in bulk[4]
    assert k[:5] == ("atom", {"bsse_nstar": 20, "bsse_rmax": 7.5}, 19, 1, "r2scan")
    assert o[:5] == ("atom", {"bsse_nstar": 10, "bsse_rmax": 7.5}, 8, 3, "r2scan")
    assert o[5] == "8 0\n19 0\n99 0\n"
    # Three atoms (2 K + 1 O) each 1 mHa higher raise sum(n E_atom) - E_bulk
    # by 3 mHa: the cohesive energy grows.
    mha = 1e-3 * HARTREE_TO_KJ_PER_MOL
    assert report.delta_kj == pytest.approx(3 * mha)
    assert [r.delta_kj for r in report.runs] == pytest.approx([0.0, mha, mha])
    assert report.runs[2].reference_status == "crashed"
    saved = json.loads((tmp_path / "K2O" / "reproduction.json").read_text())
    assert saved["delta_kj"] == pytest.approx(report.delta_kj)


def test_a_consistent_variant_changes_only_the_named_runs(tmp_path):
    """Bundled basis for the O atom; the bulk relaxed from its deck's start."""
    StandIn.calls = []
    report = reproduce_reference_system(
        _system(), StandIn, tmp_path,
        bundled_basis={"o.out": "pob-TZVP-rev2"}, relax={"bulk.out"}, label="consistent",
    )
    bulk, k, o = StandIn.calls
    # Relaxed from the deck's starting cell, not the reference's relaxed 6.36.
    assert bulk[:4] == ("bulk", {"shrink": (8, 24), "optgeom": True}, 6.4, "r2scan")
    assert report.runs[0].settings["optgeom"] is True
    # The K atom keeps its deck; the O atom gets the bundled record.
    assert k[5] == bulk[4] and o[5] != "8 0\n19 0\n99 0\n" and o[5].startswith("19 ")
    assert report.runs[2].basis_source.startswith("bundled pob-TZVP-rev2")
    assert (tmp_path / "K2O" / "consistent.json").exists()


def test_a_run_without_its_deck_is_reported_not_guessed(tmp_path):
    StandIn.calls = []
    report = reproduce_reference_system(_system(o_deck=False), StandIn, tmp_path, record=False)
    assert report.reproduced_kj is None
    assert report.runs[2].failure_mode == "no_deck" and report.runs[2].basis_source == "no deck"
    assert len(StandIn.calls) == 2


def test_the_crystal23_factory_carries_the_per_run_settings(tmp_path):
    make = crystal23_reference_factory(object(), crystal_binary="Pcrystal23", invocation="mpi", cpus=4)
    eng = make(tmp_path, shrink=(8, 24), bsse_nstar=20, bsse_rmax=7.5)
    assert (eng.shrink, eng.bsse_nstar, eng.bsse_rmax, eng.invocation, eng.cpus) == ((8, 24), 20, 7.5, "mpi", 4)
    assert eng.strict_version
