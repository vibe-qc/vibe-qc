"""Release pins for the MSINDO-SECCM systems of the semi-empirical article.

Blocking T1, so a release cannot change these numbers silently. The route
itself is not declared production here: its maturity is a separate decision.
These are change-safety pins of the audited stable states (issue #249); the
multi-start uniqueness, 3x3x3 and force checks live in test_msindo_ccm.py.
About ten seconds of native SCF on an unloaded machine.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.msindo_seccm_article_systems import (
    ARTICLE_222_PINS,
    BENT_CHAIN_ENERGIES,
    BENT_CHAIN_REFERENCE_3E3,
    BENT_CHAIN_SWEEP_BOHR,
    article_cluster,
    bent_c3_chain,
    pin_id,
)
from vibeqc.semiempirical.methods import msindo_ccm as ccm


@pytest.fixture(scope="module", autouse=True)
def _native_kernel():
    if ccm._cpp_ccm_energy_kernel() is None:
        pytest.fail("the native MSINDO CCM kernel is required for the release pins")


@pytest.mark.parametrize("madelung", [False, True], ids=["bare", "madelung"])
def test_bent_c3_chain_stays_on_one_branch(madelung):
    """Under the lowest-stable policy E(dy) of the bent C3 chain is
    continuous and pinned (#249)."""
    energies = []
    for dy in BENT_CHAIN_SWEEP_BOHR:
        coords, translations = bent_c3_chain(dy)
        r = ccm.run_ccm([6, 6, 6], coords, translations, madelung=madelung,
                        max_iter=400, conv_tol=1e-9, stability="lowest")
        assert r.converged and r.stability_checked and r.stability_analysis_converged
        assert not r.internal_instability
        energies.append(r.total_energy)
    np.testing.assert_allclose(energies, BENT_CHAIN_ENERGIES[madelung], rtol=0, atol=1e-8)
    assert np.all(np.abs(np.diff(energies)) < 2e-4)


@pytest.mark.parametrize("madelung", [False, True], ids=["bare", "madelung"])
def test_bent_c3_chain_reference_policy_flags_the_saddle(madelung):
    """The default policy keeps the reference trajectory's state but never
    returns a detected saddle silently (#249)."""
    coords, translations = bent_c3_chain(3e-3)
    r = ccm.run_ccm([6, 6, 6], coords, translations, madelung=madelung,
                    max_iter=400, conv_tol=1e-9)
    assert r.converged and r.stability_checked and r.internal_instability
    assert r.stability_eigenvalue < -1e-2
    assert r.total_energy == pytest.approx(BENT_CHAIN_REFERENCE_3E3[madelung], abs=1e-8)


@pytest.mark.parametrize(
    "name, madelung, stability, energy, max_charge, restarts, flagged",
    ARTICLE_222_PINS, ids=[pin_id(entry) for entry in ARTICLE_222_PINS],
)
def test_article_222_cluster_pinned(
    name, madelung, stability, energy, max_charge, restarts, flagged
):
    """MgO (B1) and diamond 2x2x2, bare and Madelung-embedded."""
    Z, coords, translations = article_cluster(name, 2)
    r = ccm.run_ccm(Z, coords, translations, madelung=madelung,
                    max_iter=400, conv_tol=1e-9, stability=stability)
    assert r.converged and r.stability_checked and r.stability_analysis_converged
    assert r.internal_instability == flagged
    if flagged:
        assert r.stability_eigenvalue < -1.0
    else:
        assert r.stability_eigenvalue > 1e-3
    assert r.n_stability_restarts == restarts
    assert r.total_energy == pytest.approx(energy, abs=1e-8)
    charges = ccm.net_atomic_charges(Z, r.density)
    assert np.max(np.abs(charges)) == pytest.approx(max_charge, abs=1e-4)
    assert abs(float(np.sum(charges))) < 1e-8
