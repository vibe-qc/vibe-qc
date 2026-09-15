"""Per-k band extraction is a public accessor, not a guess (issue #505).

Reading "the bands of this result" looks trivial and is not: the layout
differs by driver, and there used to be no way to ask.  Every consumer
duck-typed the result object, and the same class of mistake was made
twice in five days, in two different campaign carriers, each time after
a certified SCF and each time burning a full-node wall:

* **#15 (BUG 124)** -- ``float(b[n_occ - 1])`` where the selection was a
  degenerate manifold, i.e. an ndarray rather than a scalar.
* **#135** -- a carrier branched on ``mo_energies_k``, did not find it on
  a ``PBCBipoleRHFResult``, and fell back to ``eps_k = [mo_energies]``,
  wrapping the whole per-k list as ONE pseudo-k.  Its guard
  ``len(b) > n_occ`` then counted *k-points* rather than *bands* and
  passed, so ``float(row)`` raised.

The real layouts, which is why neither naive test works:

    PBCBipoleRHFResult   mo_energies   : List[np.ndarray], one per k
                         mo_energies_k : ABSENT
    GPW / GAPW multi-k   mo_energies_k : tuple of per-k arrays
    Gamma-only drivers   mo_energies   : a single 1-D array

The reference numbers below come from a real converged run, not from a
constructed object -- the whole point of #135 is that the *shape* of a
real result was not what a reader assumed.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import InitialGuess, PeriodicRHFOptions, monkhorst_pack
from vibeqc.pbc_bipole import run_pbc_bipole_rhf


# Measured 2026-08-28 on this fixture: H2/STO-3G in a 6.0-bohr cube at
# separation 1.4 bohr, k = (2,1,1), cutoff 8.5, SAD guess, DIIS.
# Converged E = -1.1944690859739715 Ha.  The frontier values are the ones
# the #135 carrier was trying and failing to produce.
#
# 2026-09-05, #674: the corrected split's default alpha on this 6-bohr cell
# at the 8.5-bohr cutoff is the erfc bound 0.505 instead of CRYSTAL's
# 0.467, and the total moved -1.1944690859739715 -> -1.1944715044284 Ha.
# This fixture pins its SR ket-image ball at 12 bohr (an M4a oracle
# contract, under-converged for a 6-bohr cell: the total still moves with
# alpha up to 1.0 by 7.5e-6 Ha, and by 3e-5 Ha with the cutoff), so the
# energy here is a drift guard on the accessor's SCF, not a converged
# reference; the accessor contract below is what the file tests.
# #704/D131: displaced Gaussian-product nuclear support changes this
# under-converged SCF by -7.3047944e-7 Ha, without changing band selection.
E_REF_HA = -1.1944722349078716
HOMO_REF_HA = -0.5958704285585078
LUMO_REF_HA = 0.7875581932570849
TOL_HA = 1.0e-6


@pytest.fixture(scope="module")
def bipole_multik_result():
    system = vq.PeriodicSystem(
        3,
        np.diag([6.0, 6.0, 6.0]),
        [vq.Atom(1, [0, 0, -0.7]), vq.Atom(1, [0, 0, 0.7])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [2, 1, 1])
    opts = PeriodicRHFOptions()
    opts.lattice_opts.cutoff_bohr = 8.5
    opts.lattice_opts.nuclear_cutoff_bohr = 8.5
    opts.max_iter = 60
    opts.use_diis = True
    opts.initial_guess = InitialGuess.SAD
    result = run_pbc_bipole_rhf(
        system,
        basis,
        kmesh,
        opts,
        ewald_precision=1e-8,
        sr_image_extent_bohr=12.0,
        progress=False,
        use_fock_symmetry=False,
    )
    assert result.converged
    assert abs(result.energy - E_REF_HA) < TOL_HA
    return result


# --------------------------------------------------------------------------
# 0. Why an accessor was needed: BOTH routes available without one are
#    wrong on this result, and only one of them is loud.
#    These two run unchanged at the parent -- they measure the defect
#    rather than the presence of the new symbols (L124).
# --------------------------------------------------------------------------

def _reference_frontier(result, n_occ: int) -> tuple[float, float]:
    """Frontier bands derived explicitly, using no vibe-qc helper."""
    per_k = [np.sort(np.asarray(block, dtype=float)) for block in result.mo_energies]
    return (
        max(float(block[n_occ - 1]) for block in per_k),
        min(float(block[n_occ]) for block in per_k),
    )


def test_the_wrapping_route_crashes_after_a_certified_scf(
    bipole_multik_result,
):
    """Issue #135, reproduced on a real converged result.

    ``mo_energies_k`` is absent, so the carrier's fallback wrapped the
    per-k list as one pseudo-k. Its guard ``len(b) > n_occ`` counted
    k-points (2 > 1) and passed; ``float`` then received a row.
    """
    eps_k = getattr(bipole_multik_result, "mo_energies_k", None)
    if eps_k is None:
        eps_k = [bipole_multik_result.mo_energies]
    assert len(eps_k) == 1  # one pseudo-k where there are two real ones
    block = np.sort(np.asarray(eps_k[0], dtype=float))
    assert block.ndim == 2
    assert len(block) > 1  # the guard that passed, counting the wrong axis
    with pytest.raises(TypeError):
        float(block[0])


def test_the_flattening_route_reports_a_wrong_gap_and_does_not_crash(
    bipole_multik_result,
):
    """The quieter half, and the reason this needed a fixed accessor.

    The other pattern in the tree -- ``np.asarray(mo_energies).ravel()``
    when ``mo_energies_k`` is absent -- raises nothing. It mixes the two
    k-points into one list, so the second-lowest eigenvalue (an OCCUPIED
    band at the other k-point) is reported as the conduction band:

        true      homo -0.5958704286  lumo  0.7875581933  gap 1.3834286218
        flattened homo -0.6193343306  lumo -0.5958704286  gap 0.0234639021
        (pre-#674: homo -0.6193305389  lumo -0.5958697315  gap 0.0234608073)

    A 1.3834 Ha insulator rendered as a 0.0235 Ha near-metal, on a
    converged SCF, with rc 0 and nothing in the log.
    """
    n_occ = 1
    true_homo, true_lumo = _reference_frontier(bipole_multik_result, n_occ)
    flat = np.sort(
        np.asarray(bipole_multik_result.mo_energies, dtype=float).ravel()
    )
    flat_homo, flat_lumo = float(flat[n_occ - 1]), float(flat[n_occ])

    # -0.6193335339190602 before the #704 nuclear-support repair.
    assert flat_homo == pytest.approx(-0.6193343306336518, abs=TOL_HA)
    assert flat_lumo == pytest.approx(HOMO_REF_HA, abs=TOL_HA)
    # The reported "conduction band" is in fact an occupied band.
    assert flat_lumo == pytest.approx(true_homo, abs=TOL_HA)
    assert (flat_lumo - flat_homo) == pytest.approx(0.02346390207514404, abs=1e-8)
    assert abs((flat_lumo - flat_homo) - (true_lumo - true_homo)) > 1.0


# --------------------------------------------------------------------------
# 1. The shape a real BIPOLE result actually has
# --------------------------------------------------------------------------

def test_bipole_result_has_no_mo_energies_k_attribute(bipole_multik_result):
    """The premise of the #135 defect, pinned so it cannot drift silently.

    A consumer that branches on ``mo_energies_k`` takes the wrong branch
    on every BIPOLE result. If this attribute is ever added, the fallback
    path below stops being exercised and this lane should be revisited.
    """
    assert not hasattr(bipole_multik_result, "mo_energies_k")
    assert isinstance(bipole_multik_result.mo_energies, list)
    assert len(bipole_multik_result.mo_energies) == 2


def test_per_k_accessor_returns_one_block_per_kpoint(bipole_multik_result):
    blocks = vq.per_k_band_energies(bipole_multik_result)
    assert len(blocks) == 2
    for block in blocks:
        assert block.ndim == 1
        assert block.size == 2


# --------------------------------------------------------------------------
# 2. The frontier values the carrier could not produce
# --------------------------------------------------------------------------

def test_frontier_bands_on_a_real_converged_multik_result(
    bipole_multik_result,
):
    frontier = vq.frontier_bands(bipole_multik_result, n_occ=1)
    assert frontier.n_kpoints == 2
    assert frontier.n_bands == 2
    assert frontier.homo == pytest.approx(HOMO_REF_HA, abs=TOL_HA)
    assert frontier.lumo == pytest.approx(LUMO_REF_HA, abs=TOL_HA)
    assert frontier.gap == pytest.approx(
        LUMO_REF_HA - HOMO_REF_HA, abs=TOL_HA
    )
    assert 0 <= frontier.homo_k < 2
    assert 0 <= frontier.lumo_k < 2


def test_the_135_carrier_shape_is_refused_loudly(bipole_multik_result):
    """The exact wrong input, rejected with a message that names it.

    ``eps_k = [result.mo_energies]`` used to reach ``float(row)`` and
    raise an opaque numpy TypeError after the SCF had already succeeded.
    """
    wrapped = SimpleNamespace(
        mo_energies=[bipole_multik_result.mo_energies]
    )
    with pytest.raises(TypeError, match="one pseudo-k"):
        vq.per_k_band_energies(wrapped)


# --------------------------------------------------------------------------
# 3. The guard is on the band axis, not the k axis
# --------------------------------------------------------------------------

def test_n_occ_is_checked_against_bands_not_kpoints(bipole_multik_result):
    """#135's accomplice: ``len(b) > n_occ`` counted k-points and passed.

    This result has 2 k-points and 2 bands, so a k-counting guard accepts
    ``n_occ = 2`` -- for which there is no virtual band to report.
    """
    with pytest.raises(ValueError, match="no virtual band"):
        vq.frontier_bands(bipole_multik_result, n_occ=2)
    with pytest.raises(ValueError, match="n_occ must be"):
        vq.frontier_bands(bipole_multik_result, n_occ=0)


# --------------------------------------------------------------------------
# 4. The other shapes, and the #15 degenerate manifold
# --------------------------------------------------------------------------

def test_gamma_only_single_array_is_a_one_element_kset():
    """Negative control: the same accessor with multi-k turned off.

    A Gamma-only result is a single 1-D array, and must come back as
    exactly one block carrying exactly those values -- not flattened,
    not wrapped twice.
    """
    eps = np.array([-0.6, 0.1, 0.8, 1.2])
    blocks = vq.per_k_band_energies(SimpleNamespace(mo_energies=eps))
    assert len(blocks) == 1
    assert np.array_equal(blocks[0], eps)

    frontier = vq.frontier_bands(SimpleNamespace(mo_energies=eps), n_occ=2)
    assert frontier.n_kpoints == 1
    assert frontier.homo == 0.1
    assert frontier.lumo == 0.8
    assert frontier.homo_k == 0 and frontier.lumo_k == 0


def test_mo_energies_k_layout_is_read_when_present():
    result = SimpleNamespace(
        mo_energies_k=(np.array([-0.5, 0.4]), np.array([-0.3, 0.2])),
        mo_energies=np.array([-0.5, 0.4]),
    )
    blocks = vq.per_k_band_energies(result)
    assert len(blocks) == 2
    frontier = vq.frontier_bands(result, n_occ=1)
    # Indirect: the VBM sits at k=1 and the CBM at k=1 too here, but the
    # selection must be a max/min over k rather than a single-k read.
    assert frontier.homo == -0.3
    assert frontier.lumo == 0.2
    assert frontier.homo_k == 1


def test_two_dimensional_block_is_read_as_n_k_by_n_bands():
    result = SimpleNamespace(
        mo_energies_k=np.array([[-0.5, 0.4], [-0.3, 0.2]])
    )
    blocks = vq.per_k_band_energies(result)
    assert len(blocks) == 2
    assert blocks[0].tolist() == [-0.5, 0.4]


def test_degenerate_manifold_reduces_to_a_scalar():
    """Issue #15 (BUG 124): the VBM sat in a degenerate manifold.

    The reduction is a max/min over scalars rather than an index into a
    manifold, so a degeneracy is arithmetic rather than a TypeError.
    """
    eps = np.array([-0.4, -0.4, -0.4, 0.9, 0.9])
    frontier = vq.frontier_bands(SimpleNamespace(mo_energies=eps), n_occ=3)
    assert isinstance(frontier.homo, float)
    assert frontier.homo == -0.4
    assert frontier.lumo == 0.9
    assert frontier.gap == pytest.approx(1.3)


def test_a_ragged_kset_is_refused_rather_than_padded():
    """Different band counts per k misalign every band index."""
    result = SimpleNamespace(
        mo_energies=[np.array([-0.5, 0.4]), np.array([-0.3, 0.2, 0.9])]
    )
    with pytest.raises(ValueError, match="different band counts"):
        vq.per_k_band_energies(result)


def test_a_result_without_bands_says_so():
    with pytest.raises(TypeError, match="neither 'mo_energies_k'"):
        vq.per_k_band_energies(SimpleNamespace(energy=-1.0))
