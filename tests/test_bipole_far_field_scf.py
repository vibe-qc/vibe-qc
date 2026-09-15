"""RKS guards for the unavailable BIPOLE quartet far-field."""

from __future__ import annotations

import inspect

import pytest

from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks


def test_rks_far_field_default_is_exact_route():
    parameter = inspect.signature(run_pbc_bipole_rks).parameters[
        "use_multipole_far_field"
    ]
    assert parameter.default is False


def test_rks_explicit_far_field_fails_before_scf_setup():
    with pytest.raises(NotImplementedError, match="three-translation"):
        run_pbc_bipole_rks(None, None, None, use_multipole_far_field=True)
