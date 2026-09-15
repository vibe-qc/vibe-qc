"""Issue #316 root cause: pair-distance-correct image selection.

``direct_lattice_cells`` selects image cells by the lattice-TRANSLATION
length.  An interaction cutoff, however, bounds the PAIR distance
|R_A - R_B - g| (Bredow, Geudtner & Jug, J. Comput. Chem. 22, 89 (2001),
pp. 90-91: the CCM interaction region is constructed symmetrically around
each atom, not around the cell origin).  Selecting by |g| alone means a
Gamma supercell wider than the cutoff keeps no image at all - a
free-boundary cluster - and even a narrower cell silently drops corner
pairs whose separation is within the cutoff while their translation is
not.

These regressions pin the repaired semantics for the semiempirical
periodic routes:

* the band-folding identity Gamma-supercell(n) == matched n-point
  Monkhorst-Pack mesh holds at a supercell width EXCEEDING the cutoff;
* the image list used by a wide supercell contains more than the home
  cell;
* the energy is invariant under relabeling an atom by a lattice vector
  (the pair-interaction set is translation invariant by construction);
* genuinely isolated molecular-limit boxes still fail closed without the
  explicit ``gamma_only_0`` molecular mode (issue #316 gen-1 semantics),
  and ``gamma_only_0`` still reproduces the g=0-only molecular limit.

Fail-first evidence (recorded before the fix, main @ db06420d8): the two
folding tests and both wide-supercell recovery tests died on the gen-1
fail-closed guard (``no nonzero lattice image``), and the relabeling
test moved the DFTB0 chain energy by far more than the tolerance because
the translation ball is not translation invariant.
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc._vibeqc_core import (
    Atom,
    PeriodicSystem,
    direct_lattice_cells,
    monkhorst_pack,
)
from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc.molecule import ANGSTROM_TO_BOHR

_gfn2_available = False
try:
    from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    _gfn2_available = True
except Exception:
    pass

requires_gfn2 = pytest.mark.skipif(not _gfn2_available, reason="GFN2-xTB unavailable")

_CHAIN_LENGTH = 4.1
_PRIMITIVE_ATOMS = (
    (1, np.array([0.17, 0.31, 0.0])),
    (3, np.array([1.39, -0.22, 0.0])),
)


def _hli_chain(repeats: int = 1) -> PeriodicSystem:
    atoms = []
    for image in range(repeats):
        shift = np.array([image * _CHAIN_LENGTH, 0.0, 0.0])
        atoms.extend(Atom(z, xyz + shift) for z, xyz in _PRIMITIVE_ATOMS)

    system = PeriodicSystem()
    system.dim = 1
    system.lattice = np.diag([repeats * _CHAIN_LENGTH, 30.0, 30.0])
    system.unit_cell = atoms
    system.charge = 0
    system.multiplicity = 1
    return system


def _diamond_si(repeats: int = 1) -> PeriodicSystem:
    a = 5.43 * ANGSTROM_TO_BOHR
    primitive_lattice = np.array(
        [
            [0.0, a / 2.0, a / 2.0],
            [a / 2.0, 0.0, a / 2.0],
            [a / 2.0, a / 2.0, 0.0],
        ]
    )
    primitive_atoms = (
        np.zeros(3),
        np.full(3, a / 4.0),
    )
    atoms = []
    for image in np.ndindex((repeats, repeats, repeats)):
        shift = primitive_lattice @ np.asarray(image, dtype=float)
        atoms.extend(Atom(14, xyz + shift) for xyz in primitive_atoms)
    return PeriodicSystem(
        3,
        repeats * primitive_lattice,
        atoms,
        0,
        1,
    )


def _mgo_conventional(repeats: int = 1) -> PeriodicSystem:
    """MgO rocksalt, conventional cubic cell a = 4.212 Angstrom (issue #316)."""
    a = 4.212 * ANGSTROM_TO_BOHR
    base = (
        (12, np.array([0.0, 0.0, 0.0])),
        (12, np.array([0.0, 0.5, 0.5])),
        (12, np.array([0.5, 0.0, 0.5])),
        (12, np.array([0.5, 0.5, 0.0])),
        (8, np.array([0.5, 0.0, 0.0])),
        (8, np.array([0.0, 0.5, 0.0])),
        (8, np.array([0.0, 0.0, 0.5])),
        (8, np.array([0.5, 0.5, 0.5])),
    )
    atoms = []
    for image in np.ndindex((repeats, repeats, repeats)):
        shift = (np.asarray(image, dtype=float)) * a
        atoms.extend(Atom(z, (frac * a) + shift) for z, frac in base)
    return PeriodicSystem(
        3,
        np.eye(3) * (repeats * a),
        atoms,
        0,
        1,
    )


@pytest.fixture(scope="module")
def parameters():
    return _se.SemiempiricalParameters.dftb0_default()


def test_dftb0_folding_identity_supercell_wider_than_cutoff(parameters):
    """Gamma-supercell(4) == 4-point mesh at a width exceeding the cutoff.

    Chain width 4 x 4.1 = 16.4 bohr > cutoff 15 bohr: the translation
    ball is empty beyond g=0, so pre-fix this was a free-boundary
    cluster (686 uHa/cell-scale surface error against the mesh; fixed
    fail-closed by gen-1).  Pair-distance selection keeps every image
    pair within the cutoff and the folding identity is exact.
    """
    primitive = _hli_chain()
    repeats = 4
    supercell = _hli_chain(repeats)
    cutoff = 15.0

    # The translation ball still collapses - that enumerator is
    # deliberately untouched (shared by the Gaussian stack).
    assert len(direct_lattice_cells(supercell, cutoff)) == 1

    kmesh = monkhorst_pack(primitive, (repeats, 1, 1))
    kpoint = _se.run_dftb0_kpoints(primitive, parameters, kmesh, cutoff)

    gamma_options = _se.PeriodicDFTB0Options()
    gamma_options.cutoff_bohr = cutoff
    gamma = _se.run_dftb0_gamma(supercell, parameters, gamma_options)

    assert gamma.n_cells > 1
    assert kpoint.energy == pytest.approx(gamma.energy / repeats, abs=1.0e-9)
    assert kpoint.e_electronic == pytest.approx(
        gamma.e_electronic / repeats, abs=1.0e-9
    )
    assert kpoint.e_repulsive == pytest.approx(
        gamma.e_repulsive / repeats, abs=1.0e-12
    )


def test_scc_dftb_folding_identity_supercell_wider_than_cutoff(parameters):
    """SCC-DFTB folding identity at a width exceeding the cutoff.

    Exercises the second-order gamma-matrix lattice sum as well: with
    pair-distance selection the primitive gamma sum folds term-for-term
    onto the supercell sum, so the identity is limited only by the SCC
    convergence tolerance.
    """
    primitive = _hli_chain()
    repeats = 4
    supercell = _hli_chain(repeats)
    cutoff = 15.0

    kpoint_options = _se.SCCOptions()
    kpoint_options.max_iter = 500
    kpoint_options.conv_tol_charge = 1.0e-11
    kmesh = monkhorst_pack(primitive, (repeats, 1, 1))
    kpoint = _se.run_scc_dftb_kpoints(
        primitive, parameters, kmesh, kpoint_options, cutoff
    )

    gamma_options = _se.PeriodicSCCOptions()
    gamma_options.cutoff_bohr = cutoff
    gamma_options.max_iter = kpoint_options.max_iter
    gamma_options.conv_tol_charge = kpoint_options.conv_tol_charge
    gamma = _se.run_scc_dftb_gamma(supercell, parameters, gamma_options)

    assert kpoint.converged
    assert gamma.converged
    assert kpoint.energy == pytest.approx(gamma.energy / repeats, abs=1.0e-8)
    gamma_charges = np.asarray(gamma.charges).reshape(repeats, 2).mean(axis=0)
    np.testing.assert_allclose(kpoint.charges, gamma_charges, atol=1.0e-7)


def test_dftb0_supercell_image_list_exceeds_home_cell(parameters):
    """A 2x supercell wider than the cutoff must keep image cells."""
    supercell = _diamond_si(2)
    cutoff = 12.0

    # Documented collapse of the translation ball for this geometry:
    # every diamond-Si 2x primitive supercell vector is 14.5 bohr long.
    assert len(direct_lattice_cells(supercell, cutoff)) == 1

    gamma_options = _se.PeriodicDFTB0Options()
    gamma_options.cutoff_bohr = cutoff
    result = _se.run_dftb0_gamma(supercell, parameters, gamma_options)
    assert result.n_cells > 1
    assert np.isfinite(result.energy)


def test_dftb0_energy_invariant_under_atom_relabeling(parameters):
    """Relabeling an atom by a lattice vector must not move the energy.

    The pair-interaction set {(A, B, g) : |R_A - R_B - g| <= cutoff} is
    translation invariant by construction; the pre-fix translation-ball
    set is not (see cpp/include/vibeqc/lattice_pair_cells.hpp for the
    same defect class measured in the Gaussian stack).
    """
    cutoff = 8.0
    reference = _hli_chain()

    relabeled = _hli_chain()
    moved = [
        Atom(atom.Z, np.asarray(atom.xyz) + np.array([_CHAIN_LENGTH, 0.0, 0.0]))
        if index == 1
        else atom
        for index, atom in enumerate(relabeled.unit_cell)
    ]
    relabeled.unit_cell = moved

    options = _se.PeriodicDFTB0Options()
    options.cutoff_bohr = cutoff
    energy_reference = _se.run_dftb0_gamma(reference, parameters, options).energy
    energy_relabeled = _se.run_dftb0_gamma(relabeled, parameters, options).energy

    assert energy_relabeled == pytest.approx(energy_reference, abs=1.0e-10)


def test_mgo_wide_supercell_interaction_cells_exceed_home_cell():
    """Issue #316 reproduction row: MgO conventional n=2 at cutoff 15.

    Cell width 15.92 bohr > 15 bohr: the translation ball collapses to
    the home cell (the free-boundary-cluster defect measured at up to
    +686 uHa/cell on the Gamma ladder).  The pair-distance selection the
    semiempirical routes now sum over must retain the interacting
    images.
    """
    supercell = _mgo_conventional(2)
    cutoff = 15.0

    assert len(direct_lattice_cells(supercell, cutoff)) == 1
    interaction_cells = _se.atom_pair_interaction_cells(supercell, cutoff)
    assert len(interaction_cells) > 1
    # The home cell leads the ascending-|g| order (gamma_only_0 relies
    # on cells[0] being g=0).
    assert np.array_equal(
        np.asarray(interaction_cells[0].index), [0, 0, 0]
    )
    # Every retained translation carries at least one atom pair within
    # the cutoff (rule (*)) - no cell is retained by translation length
    # alone.
    positions = np.array([atom.xyz for atom in supercell.unit_cell])
    for cell in interaction_cells:
        separations = np.linalg.norm(
            positions[:, None, :]
            - positions[None, :, :]
            - np.asarray(cell.r_cart)[None, None, :],
            axis=-1,
        )
        assert separations.min() <= cutoff + 1.0e-12


@requires_gfn2
def test_gfn2_si_wide_supercell_recovers_periodicity():
    """GFN2 end-to-end on a supercell wider than the cutoff.

    The 16-atom diamond-Si 2x primitive supercell (translation lengths
    14.5 bohr) at a 12-bohr cutoff has an empty translation ball:
    pre-fix this was refused (gen-1 guard) after silently degrading to a
    free-boundary cluster; pair-distance selection must retain the
    images and converge.  (The polar MgO wide-supercell analogue is
    covered as a selection-rule pin above: its GFN2 SCC does not yet
    converge under the damped third-order mixing in this newly reachable
    image-summed regime - measured 16-atom primitive-based 2x2x2 at
    cutoff 10: unconverged after 3000 iterations - which is a
    convergence follow-up on the issue, not an image-selection defect.)
    """
    supercell = _diamond_si(2)
    relabeled = _diamond_si(2)
    lattice_vector = np.asarray(relabeled.lattice)[:, 0]
    relabeled.unit_cell = [
        Atom(atom.Z, np.asarray(atom.xyz) + lattice_vector)
        if index == 1
        else atom
        for index, atom in enumerate(relabeled.unit_cell)
    ]
    cutoff = 12.0

    assert len(direct_lattice_cells(supercell, cutoff)) == 1

    params = load_gfn2_params()
    opts = _xtb.XTBSccOptions()
    opts.max_iter = 1000
    reference = _xtb.run_gfn2_xtb_gamma(supercell, params, opts, cutoff)
    translated = _xtb.run_gfn2_xtb_gamma(relabeled, params, opts, cutoff)
    assert reference.converged
    assert translated.converged
    assert reference.n_cells > 1
    # The padded cell list is a bounding-box artefact of where the input
    # representative sits, not an invariant: relabelling atom 1 one lattice
    # vector over widens the box that has to enclose every retained pair
    # image (25 -> 31 here) while retaining exactly the same pair images.
    # The invariant is the interaction set, asserted through the energy
    # decomposition below.
    assert translated.n_cells > 1
    np.testing.assert_allclose(
        [
            translated.energy,
            translated.free_energy,
            translated.e_electronic,
            translated.e_repulsive,
            translated.e_scc,
            translated.e_band0,
            translated.e_aes,
            translated.e_3rd,
        ],
        [
            reference.energy,
            reference.free_energy,
            reference.e_electronic,
            reference.e_repulsive,
            reference.e_scc,
            reference.e_band0,
            reference.e_aes,
            reference.e_3rd,
        ],
        rtol=0.0,
        atol=1.0e-10,
    )


def test_isolated_molecular_limit_box_still_fails_closed(parameters):
    """H2 in a 50-bohr box has no in-range image pair: fail closed.

    Pair-distance selection must NOT weaken the gen-1 guard for
    genuinely isolated systems - a tiny molecule in a huge box has no
    image interaction within the cutoff, and reporting the resulting
    free-boundary cluster as periodic remains an error unless the
    explicit ``gamma_only_0`` molecular mode is selected.
    """
    system = PeriodicSystem()
    system.dim = 3
    system.lattice = np.eye(3) * 50.0
    system.unit_cell = [
        Atom(1, [0.0, 0.0, 0.0]),
        Atom(1, [1.4, 0.0, 0.0]),
    ]
    system.charge = 0
    system.multiplicity = 1

    options = _se.PeriodicDFTB0Options()
    options.cutoff_bohr = 15.0
    with pytest.raises(
        ValueError,
        match=r"no nonzero lattice image.*free-boundary cluster",
    ):
        _se.run_dftb0_gamma(system, parameters, options)

    options.gamma_only_0 = True
    molecular_limit = _se.run_dftb0_gamma(system, parameters, options)
    assert molecular_limit.n_cells == 1
    assert np.isfinite(molecular_limit.energy)
