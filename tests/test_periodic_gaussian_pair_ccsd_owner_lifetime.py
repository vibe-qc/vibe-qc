"""Equal-content native owner replacement must precede no stale F scan."""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_correlation_real_local_basis import _make as _basis
from tests.test_periodic_gaussian_pair_mp2 import _physical, _run as _mp2
from tests.test_periodic_gaussian_pair_ccsd import _controls


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("stage", ["BEGIN", "SINGLES_COMPLETE", "SOLVER", "FINISHED"])
def test_equal_content_basis_move_replacement_is_rejected_before_borrowed_f_scan(split, stage):
    b, provider = _physical("he", 1)
    warm = _mp2(b, provider)
    replacement = _basis(b, b.rows, b.selected, options=b.basis.options)
    original_receipt = b.basis.identity_sha256
    original_fock = b.basis.fock_copy()
    # The native diagnostic checks exact shared-state ownership directly;
    # this numerical basis intentionally has no Python state accessor.
    assert replacement.identity_sha256 == original_receipt
    assert replacement.local_basis_identity_sha256 == b.basis.local_basis_identity_sha256
    np.testing.assert_array_equal(replacement.fock_copy(), original_fock)
    options, live, caps = _controls(provider, iterations=2)
    solver = caps.solver
    solver.maximum_particle_hole_calls = 1024
    caps.solver = solver
    with pytest.raises(ValueError, match="original basis storage changed across progress"):
        core._run_periodic_gaussian_pair_ccsd_basis_replacement_diagnostic(
            b.reference, b.basis, replacement, provider, warm, options, live, caps,
            getattr(core._PeriodicGaussianPairCCSDStage, stage), split,
        )
    # This failure cannot be explained by altered F, states or receipts:
    # the move changed ONLY addresses/ownership and consumed the other wrapper.
    assert b.basis.identity_sha256 == original_receipt
    np.testing.assert_array_equal(b.basis.fock_copy(), original_fock)
    with pytest.raises(RuntimeError, match="consumed"):
        replacement.fock_copy()


def test_replacement_seam_requires_distinct_native_equal_content_owners():
    b, provider = _physical("he", 1)
    warm = _mp2(b, provider)
    options, live, caps = _controls(provider, iterations=1)
    with pytest.raises(ValueError, match="distinct equal-content native owners"):
        core._run_periodic_gaussian_pair_ccsd_basis_replacement_diagnostic(
            b.reference, b.basis, b.basis, provider, warm, options, live, caps,
            core._PeriodicGaussianPairCCSDStage.SOLVER,
        )
