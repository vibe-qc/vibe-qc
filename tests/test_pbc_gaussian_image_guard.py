"""Issue #133: the Gaussian periodic drivers must fail closed on a collapsed
image list instead of silently reporting a free-boundary cluster.

``direct_lattice_cells`` selects image cells by the lattice-TRANSLATION
length ``|g| <= R``.  An interaction cutoff, however, bounds the PAIR
separation ``|R_A - R_B - g|`` (Sharma & Beylkin, J. Chem. Theory Comput.
17, 3916 (2021), doi:10.1021/acs.jctc.0c01195, Eqs. 17-18: the lattice
factor depends on the translation only through ``T(P) = rho |A - B - P|^2``,
so the contributing translations form a ball centred on the intra-cell
offset, not on the origin).  The full repair of that convention is staged
(``LatticeSumOptions.pair_complete_1e``, ``lattice_pair_cells.hpp``).

This file pins the *stopgap* half, which is the high-severity half: when
the shortest lattice vector exceeds the cutoff, the translation ball holds
nothing but ``g = 0``.  Every image coupling vanishes, ``F(k) = F(0)`` at
every k, the bands are exactly flat, and the driver returns the isolated
cluster's energy with no error and no warning.  That is issue #183
(toroidal MP2 supercells selecting molecular-limit HF) observed in the
wild, and the Gaussian analogue of the semiempirical defect fixed under
issue #316 (``tests/test_semiempirical_pair_image_selection.py``).

Fail-first evidence, measured on main @ 335caf4 BEFORE the guard existed
(``EVIDENCE.md``, "Finding 1").  An H2 chain with ``a = 8.0`` bohr at
``cutoff_bohr = nuclear_cutoff_bohr = 6.0``:

* ``direct_lattice_cells`` returned exactly one cell, ``(0, 0, 0)``;
* ``run_rhf_periodic`` converged without complaint at every k mesh tried,
  giving -1.1167143250625706 Ha at 1x1x1, 2x1x1 AND 4x1x1 -- identical to
  the last bit, i.e. no dispersion at all;
* isolated H2 ``run_rhf`` gave -1.1167143250625706 Ha, so
  periodic - molecular was **+0.00000000000000000e+00**;
* ``run_rhf_periodic_gamma`` and ``run_rks_periodic`` likewise returned
  and raised nothing.

The same chain one lattice constant tighter (``a = 4.0``, 3 cells) moved
by 7.3e-01 Ha between a 1-point and a 2-point mesh -- real dispersion,
which is what the collapsed case is missing.  ``pytest`` on this file
then reported ``8 failed, 1 passed``: everything but the enumerator fact
below.

The escape hatch is ``LatticeSumOptions.gamma_only_0``, named for and
behaving like the semiempirical ``PeriodicDFTB0Options.gamma_only_0`` /
``PeriodicPM6Options.gamma_only_0``: the molecular limit stays reachable,
it just has to be asked for.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import direct_lattice_cells


BOND = 1.4          # bohr, H2
CHAIN_WIDE = 8.0    # bohr lattice constant; > CUTOFF -> collapsed cell list
CHAIN_TIGHT = 4.0   # bohr lattice constant; < CUTOFF -> genuinely periodic
CUTOFF = 6.0        # bohr, both the AO and the nuclear cutoff


def _h2_chain(a):
    """1D H2 chain of lattice constant ``a`` bohr along x."""
    return vq.PeriodicSystem(
        1,
        np.diag([a, 30.0, 30.0]),
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [BOND, 0.0, 0.0])],
        0,
        1,
    )


def _scf_opts(gamma_only_0=False, cutoff=CUTOFF):
    o = vq.PeriodicSCFOptions()
    o.lattice_opts.cutoff_bohr = cutoff
    o.lattice_opts.nuclear_cutoff_bohr = cutoff
    o.lattice_opts.gamma_only_0 = gamma_only_0
    o.conv_tol_energy = 1e-12
    o.conv_tol_grad = 1e-10
    return o


def _ks_opts(gamma_only_0=False, cutoff=CUTOFF):
    o = vq.PeriodicKSOptions()
    o.functional = "LDA"
    o.lattice_opts.cutoff_bohr = cutoff
    o.lattice_opts.nuclear_cutoff_bohr = cutoff
    o.lattice_opts.gamma_only_0 = gamma_only_0
    o.conv_tol_energy = 1e-10
    o.conv_tol_grad = 1e-8
    return o


def _gamma_opts(gamma_only_0=False, cutoff=CUTOFF):
    o = vq.PeriodicRHFOptions()
    o.lattice_opts.cutoff_bohr = cutoff
    o.lattice_opts.nuclear_cutoff_bohr = cutoff
    o.lattice_opts.gamma_only_0 = gamma_only_0
    o.conv_tol_energy = 1e-12
    o.conv_tol_grad = 1e-10
    return o


def _basis(system):
    return vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


def _molecular_h2():
    mol = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [BOND, 0.0, 0.0])], 0, 1
    )
    o = vq.RHFOptions()
    o.conv_tol_energy = 1e-12
    o.conv_tol_grad = 1e-10
    return vq.run_rhf(mol, vq.BasisSet(mol, "sto-3g"), o)


# ---------------------------------------------------------------------------
# The mechanism, stated as a fact about the enumerator
# ---------------------------------------------------------------------------

def test_translation_ball_collapses_when_the_cell_exceeds_the_cutoff():
    """``|g| <= R`` holds nothing but the home cell once a > R.

    ``direct_lattice_cells`` keeps its plain geometric contract -- this is
    not the bug, it is the input to it.  The bug is a *driver* treating
    this list as a periodic calculation.
    """
    collapsed = direct_lattice_cells(_h2_chain(CHAIN_WIDE), CUTOFF)
    assert len(collapsed) == 1
    assert np.array_equal(np.asarray(collapsed[0].index), [0, 0, 0])

    # The same chain one lattice constant tighter keeps real neighbours.
    populated = direct_lattice_cells(_h2_chain(CHAIN_TIGHT), CUTOFF)
    assert len(populated) > 1
    assert any((np.asarray(c.index) != 0).any() for c in populated)


# ---------------------------------------------------------------------------
# The refusal, on each of the three Gaussian driver entry points
# ---------------------------------------------------------------------------

def test_run_rhf_periodic_refuses_a_collapsed_image_list():
    system = _h2_chain(CHAIN_WIDE)
    kmesh = vq.monkhorst_pack(system, [2, 1, 1])
    with pytest.raises(ValueError, match="no nonzero lattice image"):
        vq.run_rhf_periodic(system, _basis(system), kmesh, _scf_opts())


def test_run_rks_periodic_refuses_a_collapsed_image_list():
    system = _h2_chain(CHAIN_WIDE)
    kmesh = vq.monkhorst_pack(system, [2, 1, 1])
    with pytest.raises(ValueError, match="no nonzero lattice image"):
        vq.run_rks_periodic(system, _basis(system), kmesh, _ks_opts())


def test_run_rhf_periodic_gamma_refuses_a_collapsed_image_list():
    system = _h2_chain(CHAIN_WIDE)
    with pytest.raises(ValueError, match="no nonzero lattice image"):
        vq.run_rhf_periodic_gamma(system, _basis(system), _gamma_opts())


def test_the_refusal_names_the_cutoff_and_the_ways_out():
    """A refusal that does not say what to change is a worse bug report."""
    system = _h2_chain(CHAIN_WIDE)
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])
    with pytest.raises(ValueError) as excinfo:
        vq.run_rhf_periodic(system, _basis(system), kmesh, _scf_opts())
    message = str(excinfo.value)
    assert "run_rhf_periodic" in message
    assert "free-boundary cluster" in message
    assert "gamma_only_0" in message


# ---------------------------------------------------------------------------
# The escape hatch, and what it used to happen silently
# ---------------------------------------------------------------------------

def test_gamma_only_0_reproduces_the_molecular_limit_exactly():
    """The molecular limit stays reachable -- it just has to be asked for.

    The equality asserted here is precisely the wrong answer the driver
    used to return WITHOUT being asked: a periodic SCF on an 8-bohr chain
    giving the isolated H2 molecule's energy.  Legitimate when requested
    (vacuum-box molecular-limit fixtures depend on it), a silent
    wrong-answer when not.
    """
    system = _h2_chain(CHAIN_WIDE)
    kmesh = vq.monkhorst_pack(system, [1, 1, 1])
    result = vq.run_rhf_periodic(
        system, _basis(system), kmesh, _scf_opts(gamma_only_0=True)
    )
    assert result.converged
    assert result.energy == pytest.approx(_molecular_h2().energy, abs=1e-10)


def test_gamma_only_0_is_k_independent_because_there_is_no_dispersion():
    """No image coupling means F(k) = F(0): the bands are exactly flat.

    This is the signature of the collapse.  A driver returning a
    k-independent energy for a system the user believes is periodic is
    reporting a cluster.
    """
    system = _h2_chain(CHAIN_WIDE)
    basis = _basis(system)
    energies = [
        vq.run_rhf_periodic(
            system, basis, vq.monkhorst_pack(system, [nk, 1, 1]),
            _scf_opts(gamma_only_0=True),
        ).energy
        for nk in (1, 2, 4)
    ]
    assert max(energies) - min(energies) < 1e-12


# ---------------------------------------------------------------------------
# The guard must not overreach: genuine periodic work is untouched
# ---------------------------------------------------------------------------

def test_a_cell_inside_the_cutoff_runs_without_the_flag():
    system = _h2_chain(CHAIN_TIGHT)
    kmesh = vq.monkhorst_pack(system, [2, 1, 1])
    result = vq.run_rhf_periodic(system, _basis(system), kmesh, _scf_opts())
    assert result.converged


def test_a_cell_inside_the_cutoff_actually_disperses():
    """The control for the test above: this chain is not secretly a cluster.

    If the in-range case were also collapsing, the k-mesh independence
    check would pass vacuously and the guard's scope would be untested.
    """
    system = _h2_chain(CHAIN_TIGHT)
    basis = _basis(system)
    energies = [
        vq.run_rhf_periodic(
            system, basis, vq.monkhorst_pack(system, [nk, 1, 1]), _scf_opts()
        ).energy
        for nk in (1, 4)
    ]
    assert abs(energies[0] - energies[1]) > 1e-6
