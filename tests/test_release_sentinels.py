"""Minimal shipped-route sentinels for the pre-cut release gate."""

from __future__ import annotations

import pytest

from vibeqc import (
    Atom,
    BasisSet,
    Molecule,
    RKSOptions,
    UKSOptions,
    run_mp2,
    run_rhf,
    run_rks,
    run_uhf,
    run_uks,
)


def test_molecular_production_routes_reach_pinned_fixed_points() -> None:
    h2 = Molecule([
        Atom(1, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 0.0, 1.4]),
    ])
    h2_basis = BasisSet(h2, "sto-3g")

    rhf = run_rhf(h2, h2_basis)
    assert rhf.converged
    assert rhf.energy == pytest.approx(-1.1167143250625702, abs=1e-10)
    assert run_mp2(h2, h2_basis, rhf).e_total == pytest.approx(
        -1.1298721951152082,
        abs=1e-10,
    )

    rks_options = RKSOptions()
    rks_options.functional = "LDA"
    rks = run_rks(h2, h2_basis, rks_options)
    assert rks.converged
    assert rks.energy == pytest.approx(-1.121200704006311, abs=1e-10)

    hydrogen = Molecule([Atom(1, [0.0, 0.0, 0.0])], multiplicity=2)
    hydrogen_basis = BasisSet(hydrogen, "sto-3g")

    uhf = run_uhf(hydrogen, hydrogen_basis)
    assert uhf.converged
    assert uhf.energy == pytest.approx(-0.46658184955727544, abs=1e-10)
    assert uhf.s_squared == pytest.approx(0.75, abs=1e-12)

    uks_options = UKSOptions()
    uks_options.functional = "LDA"
    uks = run_uks(hydrogen, hydrogen_basis, uks_options)
    assert uks.converged
    assert uks.energy == pytest.approx(-0.4356702329795724, abs=1e-10)
    assert uks.s_squared == pytest.approx(0.75, abs=1e-12)
