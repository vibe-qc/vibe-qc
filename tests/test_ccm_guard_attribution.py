"""Every neutral CCM entry point names *itself* when it fails closed (IID 498).

The IID 291 vacuum-padding guard fires correctly on all of them; what drifted
was the *diagnostic*. ``run_ccm_rhf_gdf`` and ``run_ccm_rks_gdf`` both delegate
to the shared ``_ccm_gdf`` helper, which hardcoded ``who="run_ccm_rhf_gdf"`` --
so a caller of ``run_ccm_rks_gdf`` was told to look at a function it never
called. The four unshared entry points each passed their own name and were
fine, which is exactly why the one shared path went unnoticed.

A wrong function name in a fail-closed message is cheap to ship and expensive
to debug: it sends the reader into the wrong module. This pins the attribution
for every public neutral route, so adding a sixth caller to a shared helper
cannot silently inherit a fifth's identity.

The guard itself (that these routes refuse a vacuum-padded declared-3-D cell
at all) is covered by the IID 291 tests; this file is only about *who* the
refusal blames.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.direct import (
    run_ccm_rhf_direct,
    run_ccm_rks_direct,
    run_ccm_uhf_direct,
    run_ccm_uks_direct,
)
from vibeqc.periodic.ccm.ri import (
    run_ccm_rhf_gdf,
    run_ccm_rhf_ri_neutral,
    run_ccm_rks_gdf,
    run_ccm_uhf_gdf,
    run_ccm_uks_gdf,
)

pytestmark = pytest.mark.experimental  # Γ-CCM (aiccm2026dev-a) research lane


def _vacuum_padded_chain():
    """H₂ chain declared 3-D with two vacuum-padded directions.

    The 6 x 30 x 30 bohr cell IID 498 was measured on: physically 1-D, declared
    3-D, so every neutral route must refuse it (IID 291).
    """
    lat = np.diag([6.0, 30.0, 30.0])
    atoms = [Atom(1, [0, 0, 0]), Atom(1, [0, 0, 1.4])]
    return CCMSystem(PeriodicSystem(3, lat, atoms, 0, 1), (2, 1, 1), "sto-3g")


# (public callable, the name its own refusal must carry)
_ROUTES = [
    (lambda c: run_ccm_rhf_gdf(c), "run_ccm_rhf_gdf"),
    (lambda c: run_ccm_rks_gdf(c, "pbe"), "run_ccm_rks_gdf"),
    (lambda c: run_ccm_uhf_gdf(c), "run_ccm_uhf_gdf"),
    (lambda c: run_ccm_uks_gdf(c, "pbe"), "run_ccm_uks_gdf"),
    (lambda c: run_ccm_rhf_ri_neutral(c), "run_ccm_rhf_ri_neutral"),
    (lambda c: run_ccm_rhf_direct(c), "run_ccm_rhf_direct"),
    (lambda c: run_ccm_rks_direct(c, "pbe"), "run_ccm_rks_direct"),
    (lambda c: run_ccm_uhf_direct(c), "run_ccm_uhf_direct"),
    (lambda c: run_ccm_uks_direct(c, "pbe"), "run_ccm_uks_direct"),
]


@pytest.mark.parametrize("call,expected", _ROUTES, ids=[n for _, n in _ROUTES])
def test_vacuum_padding_refusal_names_the_route_the_caller_used(call, expected):
    """The refusal must open with the entry point that was actually called.

    Asserted on the message *prefix*, because that is the part a reader acts
    on. The shared-helper routes (rhf/rks GDF) are the ones that can regress:
    the others cannot, since they pass their own ``who`` directly.
    """
    ccm = _vacuum_padded_chain()
    with pytest.raises(NotImplementedError) as excinfo:
        call(ccm)
    msg = str(excinfo.value)
    assert msg.startswith(f"{expected}: "), (
        f"{expected} refused, but the message blames a different route: {msg[:120]}"
    )


def test_the_two_shared_helper_routes_do_not_share_an_identity():
    """The specific regression: rhf and rks GDF go through one helper.

    Pinned as its own assertion rather than left implicit in the table above,
    so the failure reads as "these two collapsed onto one name" rather than as
    two unrelated parametrised failures.
    """
    ccm = _vacuum_padded_chain()
    names = []
    for call in (lambda c: run_ccm_rhf_gdf(c), lambda c: run_ccm_rks_gdf(c, "pbe")):
        with pytest.raises(NotImplementedError) as excinfo:
            call(ccm)
        names.append(str(excinfo.value).split(":", 1)[0])
    assert names == ["run_ccm_rhf_gdf", "run_ccm_rks_gdf"], names
