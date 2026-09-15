"""Tests for embedded localisation (embedded_localise.py).

Regression context (2026-08-05)
-------------------------------
Both branches of :func:`embedded_localise` were dead code.  Neither
:func:`vibeqc.localise.foster_boys_localise` nor
:func:`vibeqc.localise.pipek_mezey_localise` returns a ``(C_loc, U)`` pair --
both return the localised coefficients alone -- yet both call sites unpacked
two values.  That raises ``ValueError: too many values to unpack`` for every
region-I size except ``n_i == 2``, where a ``(2, n_occ)`` array unpacks into
its two *rows* and silently yields a 1-D ``C_loc`` plus a nonsense ``U``.
The only test in this file used a two-AO region and asserted
``C_loc.ndim >= 1``, so it passed on that garbage.

The ``boys`` branch additionally passed the region-I *overlap* matrix where
``foster_boys_localise`` expects the Cartesian dipole integrals
``(n_i, n_i, 3)``, so it could not have run even at ``n_i == 2``.

The tests below therefore pin, at ``n_i != 2``: 2-D shapes for both methods,
that ``U`` is a genuine orthogonal mixing matrix, that ``C_loc`` still spans
the natural-orbital space, and that ``boys`` maximises the true Boys
objective built from real dipole integrals.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc._vibeqc_core import (
    Atom,
    BasisSet,
    LatticeSumOptions,
    PeriodicSystem,
    compute_dipole,
)
from vibeqc.localise import boys_objective
from vibeqc.periodic_embedding.embedded_localise import (
    _region_dipoles,
    embedded_localise,
)
from vibeqc.periodic_embedding.region import TAG_REGION_I, TAG_SUBSTRATE
from vibeqc.periodic_embedding.runner import (
    EmbeddedSurfaceExperimentalWarning,
    run_embedded_surface,
)
from vibeqc.periodic_embedding.substrate_gf import _build_hk_sk

BOX = 30.0
N_SUBSTRATE = 4
METHODS = ("pipek-mezey", "boys")


def _atom_z(i: int) -> float:
    """z (bohr) of the i-th H in the stacked chain."""
    return 2.0 + i * 2.0


def _build(n_region: int) -> SimpleNamespace:
    """Run the embedded-surface driver for a chain with ``n_region`` region-I H.

    STO-3G H contributes one AO per atom, so ``n_i == n_region`` -- which is
    how these tests select the degenerate (``n_i == 2``) and non-degenerate
    (``n_i == 3``) regimes.
    """
    lattice = np.eye(3) * BOX
    n_atom = N_SUBSTRATE + n_region
    atoms = [Atom(1, [BOX / 2, BOX / 2, _atom_z(i)]) for i in range(n_atom)]
    system = PeriodicSystem(dim=3, lattice=lattice, unit_cell=atoms, multiplicity=1)
    basis = BasisSet(system.unit_cell_molecule(), "sto-3g")
    tags = [TAG_SUBSTRATE] * N_SUBSTRATE + [TAG_REGION_I] * n_region

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", EmbeddedSurfaceExperimentalWarning)
        result = run_embedded_surface(
            system,
            basis,
            tags,
            surface_k_mesh=(1, 1),
            contour_n_nodes=24,
            e_bottom=-2.2,
            e_fermi=-1.0,
        )

    opts = LatticeSumOptions()
    opts.cutoff_bohr = 15
    opts.nuclear_cutoff_bohr = 15
    _, Sk = _build_hk_sk(system, basis, np.zeros(3), opts)
    i_ao = result.region.i_ao
    s_ii = np.asarray(Sk[np.ix_(i_ao, i_ao)], dtype=np.complex128).real

    return SimpleNamespace(
        system=system,
        basis=basis,
        result=result,
        i_ao=i_ao,
        s_ii=s_ii,
        n_i=s_ii.shape[0],
        region_z=[_atom_z(i) for i in range(N_SUBSTRATE, n_atom)],
    )


@pytest.fixture(scope="module")
def embedded3() -> SimpleNamespace:
    """Three region-I AOs -- the shape-unambiguous case the old test missed."""
    return _build(3)


@pytest.fixture(scope="module")
def embedded2() -> SimpleNamespace:
    """Two region-I AOs -- the degenerate case that hid the unpacking bug."""
    return _build(2)


# ---------------------------------------------------------------------------
# Shape / return-contract regressions (defect 1: the bogus 2-tuple unpacking)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_localise_returns_two_dimensional_coefficients(embedded3, method):
    """``C_loc`` is ``(n_i, n_occ)`` and ``U`` is ``(n_occ, n_occ)``.

    Pre-fix this raised ``ValueError: too many values to unpack`` for both
    methods, because the localisers return one array rather than a pair.
    """
    loc = embedded_localise(
        embedded3.result.density_local,
        embedded3.s_ii,
        embedded3.basis,
        embedded3.i_ao,
        method=method,
    )

    assert loc.method == method
    assert loc.C_loc.ndim == 2
    assert loc.C_loc.shape[0] == embedded3.n_i == 3
    n_occ = loc.C_loc.shape[1]
    assert 1 <= n_occ <= embedded3.n_i
    assert loc.U.shape == (n_occ, n_occ)
    assert loc.centers.shape == (n_occ, 3)
    assert loc.centroids.shape == (n_occ, 3)
    assert loc.spreads.shape == (n_occ,)
    assert np.all(np.isfinite(loc.C_loc))
    assert np.all(np.isfinite(loc.U))


@pytest.mark.parametrize("method", METHODS)
def test_localise_two_ao_region_is_not_row_unpacked(embedded2, method):
    """``n_i == 2`` returns a matrix, not two rows of one.

    This is the exact configuration the old test used.  Pre-fix, the
    ``pipek-mezey`` branch "succeeded" here by unpacking the ``(2, 2)``
    result into its rows, so ``C_loc`` came back with shape ``(2,)``.
    """
    loc = embedded_localise(
        embedded2.result.density_local,
        embedded2.s_ii,
        embedded2.basis,
        embedded2.i_ao,
        method=method,
    )

    assert embedded2.n_i == 2
    assert loc.C_loc.ndim == 2, "C_loc collapsed to 1-D: the unpacking bug is back"
    assert loc.C_loc.shape[0] == 2
    assert loc.U.ndim == 2


# ---------------------------------------------------------------------------
# The recovered mixing matrix and the preserved natural-orbital space
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_mixing_matrix_is_orthogonal(embedded3, method):
    """``U`` is a genuine unitary mixing matrix, and ``C_loc`` stays S-orthonormal."""
    loc = embedded_localise(
        embedded3.result.density_local,
        embedded3.s_ii,
        embedded3.basis,
        embedded3.i_ao,
        method=method,
    )
    n_occ = loc.C_loc.shape[1]

    np.testing.assert_allclose(loc.U.T @ loc.U, np.eye(n_occ), atol=1e-10)
    np.testing.assert_allclose(
        loc.C_loc.T @ embedded3.s_ii @ loc.C_loc, np.eye(n_occ), atol=1e-10
    )


@pytest.mark.parametrize("method", METHODS)
def test_localised_span_is_a_ds_invariant_subspace(embedded3, method):
    """Localisation rotates within the natural-orbital space, it does not leave it.

    ``C_nat`` columns are eigenvectors of ``D S`` (``D S C_nat = C_nat n``), and
    ``C_loc = C_nat @ U`` with orthogonal ``U``, so ``span(C_loc)`` is invariant
    under ``D S``.  With ``C_loc`` S-orthonormal, ``C_loc C_loc^T S`` is the
    S-orthogonal projector onto that span, and the residual below must vanish.
    """
    d_local = embedded3.result.density_local
    s_ii = embedded3.s_ii
    loc = embedded_localise(
        d_local, s_ii, embedded3.basis, embedded3.i_ao, method=method
    )

    projector = loc.C_loc @ loc.C_loc.T @ s_ii
    image = np.asarray(d_local) @ s_ii @ loc.C_loc
    residual = (np.eye(embedded3.n_i) - projector) @ image
    assert np.max(np.abs(residual)) < 1e-10


# ---------------------------------------------------------------------------
# Dipole integrals (defect 2: the overlap matrix was passed as `dipoles`)
# ---------------------------------------------------------------------------


def test_region_dipoles_shape_and_slicing(embedded3):
    """``_region_dipoles`` returns the ``(n_i, n_i, 3)`` region-I block.

    Pre-fix the ``boys`` branch handed ``foster_boys_localise`` an
    ``(n_i, n_i)`` overlap matrix instead, which fails at ``dipoles[:, :, c]``.
    """
    i_ao = np.asarray(embedded3.i_ao, dtype=np.int64)
    dipoles = _region_dipoles(embedded3.basis, i_ao)

    assert dipoles.shape == (embedded3.n_i, embedded3.n_i, 3)

    nbf = int(embedded3.basis.nbasis)
    dip = compute_dipole(embedded3.basis)
    full = np.zeros((nbf, nbf, 3))
    full[:, :, 0] = np.asarray(dip.x)
    full[:, :, 1] = np.asarray(dip.y)
    full[:, :, 2] = np.asarray(dip.z)
    np.testing.assert_array_equal(dipoles, full[np.ix_(i_ao, i_ao)])

    # An overlap matrix would be near-identity on this basis; real dipole
    # integrals carry the ~15 bohr offset of the atoms from the origin.
    assert np.max(np.abs(dipoles)) > 1.0


def test_region_dipoles_match_gamma_localiser_convention(embedded3):
    """Same g=0 block, same origin, as ``periodic_localise._home_cell_moments``.

    The Γ-point localiser reaches the home-cell dipoles through
    ``compute_multipole_moments_lattice`` with a 1-bohr lattice cutoff;
    ``_region_dipoles`` reaches them through ``compute_dipole``.  Pinning the
    two together keeps the embedded Boys criterion on the same position
    operator as the Γ-point one.
    """
    from vibeqc.periodic_localise import _home_cell_moments

    nbf = int(embedded3.basis.nbasis)
    all_ao = np.arange(nbf, dtype=np.int64)
    mine = _region_dipoles(embedded3.basis, all_ao)
    sibling, _ = _home_cell_moments(embedded3.basis, embedded3.system)

    np.testing.assert_allclose(mine, sibling, atol=1e-12)


def test_boys_maximises_the_true_boys_objective(embedded3):
    """Boys increases ``S_i <i|r|i>^2`` built from real dipole integrals.

    A Jacobi sweep can never *decrease* the objective it is driving, so
    ``obj(C_loc) >= obj(C_nat)`` is exact rather than empirical.  Feeding the
    localiser anything other than the dipole integrals -- the overlap matrix,
    or a broadcast of it -- would drive a different objective and break the
    comparison against Pipek-Mezey below.
    """
    dipoles = _region_dipoles(
        embedded3.basis, np.asarray(embedded3.i_ao, dtype=np.int64)
    )
    loc = embedded_localise(
        embedded3.result.density_local,
        embedded3.s_ii,
        embedded3.basis,
        embedded3.i_ao,
        method="boys",
    )
    pm = embedded_localise(
        embedded3.result.density_local,
        embedded3.s_ii,
        embedded3.basis,
        embedded3.i_ao,
        method="pipek-mezey",
    )

    # C_loc = C_nat @ U with U orthogonal, so C_nat = C_loc @ U.T.
    c_nat = loc.C_loc @ loc.U.T
    obj_nat = boys_objective(c_nat, dipoles)
    obj_boys = boys_objective(loc.C_loc, dipoles)
    obj_pm = boys_objective(pm.C_loc, dipoles)

    assert obj_boys >= obj_nat - 1e-10
    assert obj_boys > obj_nat  # this system is genuinely delocalised to start
    assert obj_boys >= obj_pm - 1e-10


# ---------------------------------------------------------------------------
# Reported Wannier descriptors
# ---------------------------------------------------------------------------


def test_boys_centroids_are_position_expectation_values(embedded3):
    """Boys reports true ``<i|r|i>`` centroids that sit on the region-I chain.

    Pre-fix this field was a copy of the Mulliken ``centers``.  The atoms lie
    on the x = y = BOX/2 axis, so a genuine position expectation value has to
    land there too, with z inside the region-I span.
    """
    loc = embedded_localise(
        embedded3.result.density_local,
        embedded3.s_ii,
        embedded3.basis,
        embedded3.i_ao,
        method="boys",
    )

    np.testing.assert_allclose(loc.centroids[:, 0], BOX / 2, atol=1e-6)
    np.testing.assert_allclose(loc.centroids[:, 1], BOX / 2, atol=1e-6)
    z = loc.centroids[:, 2]
    assert np.all(z > min(embedded3.region_z) - 1.0)
    assert np.all(z < max(embedded3.region_z) + 1.0)


def test_pipek_mezey_reports_no_centroids(embedded3):
    """PM never forms the position operator, so its centroids stay zero.

    Documented behaviour, pinned so it cannot silently become stale garbage.
    """
    loc = embedded_localise(
        embedded3.result.density_local,
        embedded3.s_ii,
        embedded3.basis,
        embedded3.i_ao,
        method="pipek-mezey",
    )
    np.testing.assert_array_equal(loc.centroids, 0.0)


@pytest.mark.parametrize("method", METHODS)
def test_mulliken_centers_lie_on_the_region_chain(embedded3, method):
    """Mulliken charge-weighted centers are PBC-robust and reported for both methods."""
    loc = embedded_localise(
        embedded3.result.density_local,
        embedded3.s_ii,
        embedded3.basis,
        embedded3.i_ao,
        method=method,
    )
    np.testing.assert_allclose(loc.centers[:, 0], BOX / 2, atol=1e-6)
    np.testing.assert_allclose(loc.centers[:, 1], BOX / 2, atol=1e-6)
    z = loc.centers[:, 2]
    assert np.all(z > min(embedded3.region_z) - 1.0)
    assert np.all(z < max(embedded3.region_z) + 1.0)


@pytest.mark.parametrize("method", METHODS)
def test_spreads_are_reported_as_zero(embedded3, method):
    """``spreads`` needs region-I quadrupoles, which this entry point cannot build.

    Zero is the documented placeholder (see the module docstring).  If a later
    change wires up ``<r^2>``, this expectation is the one to revisit.
    """
    loc = embedded_localise(
        embedded3.result.density_local,
        embedded3.s_ii,
        embedded3.basis,
        embedded3.i_ao,
        method=method,
    )
    np.testing.assert_array_equal(loc.spreads, 0.0)


def test_unknown_method_is_rejected(embedded3):
    with pytest.raises(ValueError, match="Unknown method"):
        embedded_localise(
            embedded3.result.density_local,
            embedded3.s_ii,
            embedded3.basis,
            embedded3.i_ao,
            method="edmiston-ruedenberg",
        )
