"""Open-shell detection must test spin-density VALUES, not attribute presence.

The runner-contract adapters for the supercell-Gamma CCM routes
(:class:`CCMRealGammaResult`, :class:`CCMFourCentreResult`) and the CCM KS
result (:class:`CCMKSResult`) declare ``density_alpha`` / ``density_beta`` as
dataclass *fields* that are ``None`` on a closed-shell run.  A consumer that
detects open shell with ``hasattr(result, "density_alpha")`` therefore takes
the open-shell branch for a closed-shell result and evaluates ``None + None``,
raising ``TypeError: unsupported operand type(s) for +: 'NoneType' and
'NoneType'``.

Every property consumer wraps its section in ``except Exception`` and records
the message, so the failure does not crash the run -- it surfaces as a
plausible-looking ``N/A -- TypeError`` row in the population sidecar.  That is
how the same defect shipped three times (GitLab #679, and again when it turned
the whole population sidecar of a real-Gamma job into ``N/A`` rows).

These tests pin the contract from both ends:

  1. the three result classes really do default both spin fields to ``None``,
     so ``hasattr`` is structurally useless on them; and
  2. a result carrying ``density_alpha = density_beta = None`` is treated by
     every consumer exactly like one that omits the attributes entirely.
"""

from __future__ import annotations

import dataclasses as dc
from types import SimpleNamespace

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.spin_channels import is_open_shell_result, spin_densities


# --------------------------------------------------------------------------
# 1. The class contract that makes ``hasattr`` useless
# --------------------------------------------------------------------------

def _nullable_spin_result_classes():
    from vibeqc.periodic.ccm.dft import CCMKSResult
    from vibeqc.periodic.ccm.four_center_runner import CCMFourCentreResult
    from vibeqc.periodic.ccm.real_gamma_runner import CCMRealGammaResult

    return [CCMRealGammaResult, CCMFourCentreResult, CCMKSResult]


@pytest.mark.parametrize(
    "cls", _nullable_spin_result_classes(), ids=lambda c: c.__name__
)
def test_spin_fields_default_to_none_so_hasattr_is_useless(cls) -> None:
    fields = {f.name: f for f in dc.fields(cls)}
    for name in ("density_alpha", "density_beta"):
        assert name in fields, f"{cls.__name__} should declare {name}"
        assert fields[name].default is None, (
            f"{cls.__name__}.{name} must default to None -- the closed-shell "
            "case is what makes a value-based open-shell test necessary"
        )


def test_helper_reads_values_not_attribute_presence() -> None:
    closed = SimpleNamespace(density=np.eye(2), density_alpha=None, density_beta=None)
    assert hasattr(closed, "density_alpha")  # the trap the old test fell into
    assert is_open_shell_result(closed) is False
    assert spin_densities(closed) == (None, None)

    pa, pb = np.eye(2), 0.5 * np.eye(2)
    openshell = SimpleNamespace(density=pa + pb, density_alpha=pa, density_beta=pb)
    assert is_open_shell_result(openshell) is True
    assert spin_densities(openshell) == (pa, pb)

    absent = SimpleNamespace(density=np.eye(2))
    assert is_open_shell_result(absent) is False

    # A half-populated pair falls back to the combined density rather than
    # raising: the consumers of this helper are diagnostics, not the SCF.
    half = SimpleNamespace(density=np.eye(2), density_alpha=pa, density_beta=None)
    assert is_open_shell_result(half) is False


# --------------------------------------------------------------------------
# 2. Consumers treat a None-valued pair exactly like an absent one
# --------------------------------------------------------------------------

def _h2o() -> vq.Molecule:
    return vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 1.43, -0.98]),
        vq.Atom(1, [0.0, -1.43, -0.98]),
    ])


@pytest.fixture(scope="module")
def closed_shell_pair():
    """``(plain, ccm_style, basis, molecule)`` for one converged RHF density.

    ``ccm_style`` carries the identical density but also the ``None``-valued
    spin fields the CCM runner adapters declare, so any behavioural difference
    between the two is exactly the open-shell-detection bug.
    """
    mol = _h2o()
    basis = vq.BasisSet(mol, "sto-3g")
    res = vq.run_rhf(mol, basis)
    assert not hasattr(res, "density_alpha"), (
        "an RHF result should not carry spin densities at all"
    )

    ccm_style = SimpleNamespace(
        **{k: getattr(res, k) for k in ("density", "overlap", "converged")
           if hasattr(res, k)},
        density_alpha=None,
        density_beta=None,
    )
    return res, ccm_style, basis, mol


def test_mulliken_charges_ignore_none_valued_spin_fields(closed_shell_pair) -> None:
    plain, ccm_style, basis, mol = closed_shell_pair
    np.testing.assert_allclose(
        vq.mulliken_charges(ccm_style, basis, mol),
        vq.mulliken_charges(plain, basis, mol),
        atol=1e-12,
    )


def test_mayer_bond_orders_ignore_none_valued_spin_fields(closed_shell_pair) -> None:
    from vibeqc.properties import mayer_bond_orders

    plain, ccm_style, basis, mol = closed_shell_pair
    np.testing.assert_allclose(
        mayer_bond_orders(ccm_style, basis, mol),
        mayer_bond_orders(plain, basis, mol),
        atol=1e-12,
    )


def test_wiberg_and_delocalization_ignore_none_valued_spin_fields(
    closed_shell_pair,
) -> None:
    from vibeqc.bond_analysis import delocalization_index, wiberg_bond_orders

    plain, ccm_style, basis, mol = closed_shell_pair
    np.testing.assert_allclose(
        wiberg_bond_orders(ccm_style, basis, mol),
        wiberg_bond_orders(plain, basis, mol),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        delocalization_index(ccm_style, basis, mol),
        delocalization_index(plain, basis, mol),
        atol=1e-12,
    )


def test_nbo_charges_reach_the_same_dead_end(closed_shell_pair) -> None:
    """``nbo_charges`` is dead-ended behind ``npa_charges``'s NotImplementedError.

    The branch selection is still what this pins: before the fix a
    ``None``-valued pair died with ``AttributeError: 'NoneType' object has no
    attribute 'real'`` while choosing the branch; now it reaches the same
    ``NotImplementedError`` as a result that omits the spin fields entirely.
    """
    from vibeqc.nbo import nbo_charges

    plain, ccm_style, basis, mol = closed_shell_pair
    with pytest.raises(NotImplementedError):
        nbo_charges(plain, basis, mol)
    with pytest.raises(NotImplementedError):
        nbo_charges(ccm_style, basis, mol)


def test_natural_orbitals_pick_the_closed_shell_kind(closed_shell_pair) -> None:
    from vibeqc.properties import natural_orbitals

    plain, ccm_style, basis, _ = closed_shell_pair
    got = natural_orbitals(ccm_style, basis)
    ref = natural_orbitals(plain, basis)
    np.testing.assert_allclose(got.occupations, ref.occupations, atol=1e-12)


# --------------------------------------------------------------------------
# 3. The population sidecar: the shape the defect actually shipped in
# --------------------------------------------------------------------------

def test_population_sidecar_has_no_typeerror_sections(closed_shell_pair) -> None:
    """The regression that turned a real-Gamma sidecar into ``N/A`` rows.

    Each section is wrapped in ``except Exception``, so the bug never raised
    -- it filled ``summary.errors`` with ``TypeError`` messages that rendered
    as ``# <section>: N/A -- TypeError: ...``.
    """
    from vibeqc.output.formats.population import (
        compute_population_summary,
        format_population_txt,
    )

    _, ccm_style, basis, mol = closed_shell_pair
    summary = compute_population_summary(ccm_style, basis, mol)

    type_errors = {k: v for k, v in summary.errors.items() if "TypeError" in v}
    assert not type_errors, (
        f"closed-shell result took an open-shell branch: {type_errors}"
    )
    assert summary.mulliken_atoms, "Mulliken section should be populated"

    text = format_population_txt(summary)
    assert "N/A -- TypeError" not in text


def test_population_sidecar_matches_the_attribute_free_result(
    closed_shell_pair,
) -> None:
    from vibeqc.output.formats.population import compute_population_summary

    plain, ccm_style, basis, mol = closed_shell_pair
    got = compute_population_summary(ccm_style, basis, mol)
    ref = compute_population_summary(plain, basis, mol)

    assert got.errors == ref.errors
    np.testing.assert_allclose(
        [row[3] for row in got.mulliken_atoms],
        [row[3] for row in ref.mulliken_atoms],
        atol=1e-12,
    )


# --------------------------------------------------------------------------
# 4. The in-memory restart reader (``read_from=prior_result``)
# --------------------------------------------------------------------------

def test_restart_reader_treats_none_valued_pair_as_restricted() -> None:
    """``read_from=<closed-shell CCM result>`` must take the restricted branch.

    ``_prior_from_result`` splits ``density`` in half for a restricted source
    and reads the two spin channels for an unrestricted one.  With an
    attribute test, a closed-shell CCM adapter took the unrestricted branch
    and handed ``None`` to ``gamma_density``.
    """
    from vibeqc.guess_read import _is_restricted_result, _is_unrestricted_result

    ccm_style = SimpleNamespace(
        density=np.eye(4), density_alpha=None, density_beta=None
    )
    assert _is_unrestricted_result(ccm_style) is False
    assert _is_restricted_result(ccm_style) is True

    pa, pb = np.eye(4), 0.5 * np.eye(4)
    openshell = SimpleNamespace(density=pa + pb, density_alpha=pa, density_beta=pb)
    assert _is_unrestricted_result(openshell) is True
    assert _is_restricted_result(openshell) is False
