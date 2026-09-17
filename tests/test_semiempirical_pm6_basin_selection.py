"""#154: PM6 must refuse a basin-ambiguous system, not report one of its answers.

The filed symptom was a silent ``E = 0.0`` on one host, and the tripwire for
that (``_reject_non_result_energy``) is landed elsewhere.  What remained is
why the #154 verification failed: four fleet hosts returned four different
*converged* PM6/norbornadiene energies for one input and one build
(-33.0848/49 it, -33.3644/359, -33.8172/199, -33.9207/482).

Measured locally, that is not a deployment quirk.  Norbornadiene has at least
five genuine zero-temperature PM6 fixed points spanning 0.836 Ha (22.7 eV),
and hard occupations leave the SCF selecting between them on arithmetic
noise: the direct iteration converges to -33.0848096 Ha, and to -33.9206690
Ha after a 1e-9 A nudge of one bridgehead carbon, while a hot-to-cold
annealed path reaches -33.8410399, -33.8171396, -33.3644211 or -33.9206691
on inputs differing by at most 1e-7 A.  Thread count changes nothing
(identical to 16 digits from 1 to 18 threads), so no arithmetic-determinism
fix reaches it either.

Since no path selects the basin reliably, the route probes with a second
independent path and refuses when the two converged fixed points disagree.
Every host then reaches the same outcome rather than a different number.
"""

from __future__ import annotations

import pytest
import vibeqc.semiempirical.routes as semi_routes
import vibeqc.semiempirical.runner as semi_runner
from vibeqc._vibeqc_core import Atom, Molecule
from vibeqc.semiempirical.runner import SemiempiricalEnergyError

BOHR_PER_ANGSTROM = 1.8897259886

# C7H8, C2v, B3LYP/6-31G(d) (NIST CCCBDB): the geometry of the filed case and
# of scripts/debug/bug024_norbornadiene_pm6.py.
NORBORNADIENE = [
    (6, 0.000000, 0.000000, 1.268695),
    (6, 0.000000, 1.215139, 0.805739),
    (6, 0.000000, -1.215139, 0.805739),
    (6, 0.000000, 1.215139, -0.805739),
    (6, 0.000000, -1.215139, -0.805739),
    (6, 0.000000, 0.000000, -1.268695),
    (6, 0.000000, 0.000000, 0.000000),
    (1, 0.000000, 2.146163, 1.365586),
    (1, 0.000000, -2.146163, 1.365586),
    (1, 0.000000, 2.146163, -1.365586),
    (1, 0.000000, -2.146163, -1.365586),
    (1, 0.000000, 0.000000, 2.356519),
    (1, 0.000000, 0.000000, -2.356519),
    (1, 0.882000, 0.000000, 0.000000),
    (1, -0.882000, 0.000000, 0.000000),
]

# Water, well away from any frontier degeneracy: the negative control.
WATER = [
    (8, 0.000000, 0.000000, 0.117300),
    (1, 0.000000, 0.757200, -0.469200),
    (1, 0.000000, -0.757200, -0.469200),
]

DIRECT_REFERENCE_ENERGY = -33.0848095873


def _molecule(geometry, nudge=None):
    """Build the molecule, optionally moving ``(index, component, delta_A)``."""
    atoms = []
    for index, (z, x, y, zz) in enumerate(geometry):
        xyz = [x, y, zz]
        if nudge is not None and index == nudge[0]:
            xyz[nudge[1]] += nudge[2]
        atoms.append(
            Atom(z, [component * BOHR_PER_ANGSTROM for component in xyz])
        )
    return Molecule(atoms, charge=0, multiplicity=1)


@pytest.fixture(scope="module")
def pm6_params():
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_params

    return load_pm6_params()


@pytest.fixture(scope="module")
def native_run_pm6():
    from vibeqc._vibeqc_core.semiempirical.nddo import run_pm6

    return run_pm6


def _route(molecule, native_run_pm6):
    plan = semi_routes.SemiempiricalRoutePlan.from_request(
        "pm6", charge=0, multiplicity=1
    )
    return semi_runner._run_pm6(plan, molecule, native_run_pm6)


@pytest.mark.slow
def test_route_refuses_a_basin_ambiguous_system(pm6_params, native_run_pm6):
    """The direct SCF converges cleanly; it is still not an answer."""
    direct = native_run_pm6(_molecule(NORBORNADIENE), pm6_params, max_iter=200)
    assert direct.converged
    assert float(direct.energy) == pytest.approx(
        DIRECT_REFERENCE_ENERGY, abs=1.0e-6
    )

    with pytest.raises(SemiempiricalEnergyError) as excinfo:
        _route(_molecule(NORBORNADIENE), native_run_pm6)

    message = str(excinfo.value)
    assert "#154" in message
    # The refusal must name both solutions it saw, so the reader can tell a
    # basin gap from convergence noise without rerunning anything.
    assert f"{DIRECT_REFERENCE_ENERGY:.10f}" in message


@pytest.mark.slow
def test_refusal_does_not_depend_on_noise_level_input_changes(
    pm6_params, native_run_pm6
):
    """The property the four fleet hosts violated.

    The direct SCF flips basin under a 1e-9 A nudge, and on two of these
    inputs it does not converge at all.  The route's OUTCOME must be the same
    for all of them: refused, not four different converged energies.
    """
    for nudge in (None, (0, 2, 1.0e-9), (6, 0, 1.0e-9), (0, 2, 1.0e-7)):
        with pytest.raises(SemiempiricalEnergyError) as excinfo:
            _route(_molecule(NORBORNADIENE, nudge=nudge), native_run_pm6)
        assert "#154" in str(excinfo.value), nudge


@pytest.mark.slow
def test_a_single_basin_system_is_unaffected(pm6_params, native_run_pm6):
    """Negative control: agreement between the two paths reports normally.

    Water's frontier is nowhere near degenerate, so the direct SCF and the
    homotopy reach the same fixed point and the guard stays silent.
    """
    result = _route(_molecule(WATER), native_run_pm6)

    assert result.converged
    direct = native_run_pm6(_molecule(WATER), pm6_params, max_iter=200)
    assert result.energy == pytest.approx(float(direct.energy), abs=1.0e-8)


@pytest.mark.slow
def test_direct_scf_is_independent_of_the_iteration_budget(
    pm6_params, native_run_pm6
):
    """#154: a budget bounds the work; it must not select the answer.

    The damped-mixing window used to start at ``max_iter / 3``, so budgets of
    100 or less began damping before this system's iteration 49, derailed the
    very trajectory that would have converged, and returned non-converged
    (exactly 0.0 Ha), while budgets of 150 or more converged at iteration 49.
    A fixed window makes every budget a prefix of one path.
    """
    energies = {}
    for max_iter in (60, 100, 200, 400):
        result = native_run_pm6(
            _molecule(NORBORNADIENE), pm6_params, max_iter=max_iter
        )
        assert result.converged, f"max_iter={max_iter} did not converge"
        energies[max_iter] = float(result.energy)

    spread = max(energies.values()) - min(energies.values())
    assert spread < 1.0e-9, energies
