"""The neutral fitted-torus source identity (#519, RG-2).

Both producers of the neutral torus record the Hamiltonian they evaluated as
one :class:`~vibeqc.fock_state.SourceProvenance`
(:mod:`vibeqc.periodic.ccm.source`). A Theorem-1 pair is checked by identity
before its energies are compared, and a pair that evaluated different
operators is refused with the differing fields named. These tests pin that
the two producers agree on the record for one torus, that every knob which
selects a different Hamiltonian changes the identity, that a caller-supplied
array can never pass as a built fit, and that both runner arms write the
identity into their ``.system`` manifests.
"""
from __future__ import annotations

import re
import warnings

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.fock_state import SourceProvenance
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct, run_ccm_rks_direct
from vibeqc.periodic.ccm.ri import run_ccm_rhf_gdf, run_ccm_rks_gdf
from vibeqc.periodic.ccm.source import (
    NEUTRAL_TORUS_PROVIDER,
    assert_matched_neutral_source,
    neutral_source_of,
)

pytestmark = [pytest.mark.experimental, pytest.mark.aiccm]


def _h2_cube(nrep=(2, 1, 1)):
    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)
    return CCMSystem(cell, nrep, "sto-3g")


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


@pytest.fixture(scope="module")
def h2_pair():
    """One direct and one Bloch run of the H2 cube (2,1,1) at the defaults."""
    ccm = _h2_cube()
    return ccm, _quiet(run_ccm_rhf_direct, ccm), _quiet(run_ccm_rhf_gdf, ccm)


def test_both_producers_record_one_operator(h2_pair):
    ccm, direct, bloch = h2_pair
    assert direct.converged and bloch.converged
    for res in (direct, bloch):
        record = neutral_source_of(res)
        assert isinstance(record, SourceProvenance)
        assert record.provider == NEUTRAL_TORUS_PROVIDER
        details = record.as_dict()["details"]
        assert details["torus"] == {"nrep": [2, 1, 1], "n_cells": 2}
        assert details["fit"]["source"] == "classic"
        assert details["fit"]["engine"] == "native-classic-fft-shared-q"
        assert details["fit"]["auxiliary_basis"]
        assert details["fit"]["base_reciprocal_cutoff_ha"] == 200.0
        assert details["fit"]["lattice_cutoff_bohr"] == 15.0
        assert details["exchange"] == {
            "q0": "BvK-ewald", "applicability": "active", "screened": None}
        assert details["xc"] is None
    identity = assert_matched_neutral_source([direct, bloch], context="h2 pair")
    assert identity == direct.neutral_source.identity == bloch.neutral_source.identity
    assert direct.neutral_source.as_dict() == bloch.neutral_source.as_dict()
    # The identity is a precondition of the energy identity, not a substitute.
    assert direct.energy / ccm.n_cells == pytest.approx(bloch.energy, abs=1e-8)


@pytest.mark.parametrize("kwargs, field", [
    ({"exxdiv": None}, "exchange.q0"),
    ({"ke_cutoff": 100.0}, "fit.base_reciprocal_cutoff_ha"),
    ({"cderi_build": "supercell"}, "fit.engine"),
])
def test_a_different_hamiltonian_changes_the_identity(h2_pair, kwargs, field):
    ccm, direct, _ = h2_pair
    other = _quiet(run_ccm_rhf_direct, ccm, **kwargs)
    assert other.neutral_source.identity != direct.neutral_source.identity
    with pytest.raises(ValueError, match=re.escape(field)):
        assert_matched_neutral_source([direct, other], context="knob")


def test_a_caller_supplied_array_is_not_a_built_fit(h2_pair):
    """The array does not know what fitted it: an injected cderi records
    fit_source='injected' and cannot pass as either producer's operator."""
    ccm, direct, bloch = h2_pair
    injected = _quiet(run_ccm_rhf_direct, ccm, cderi=np.asarray(direct.cderi))
    assert injected.fit_source == "injected"
    details = injected.neutral_source.as_dict()["details"]
    assert details["fit"]["engine"] == "caller"
    assert details["fit"]["auxiliary_basis"] is None
    assert details["fit"]["caller_shape"] == list(np.shape(direct.cderi))
    with pytest.raises(ValueError, match="fit.source"):
        assert_matched_neutral_source([bloch, injected], context="injected")
    # Same energy, by construction; the refusal is about provenance.
    assert injected.energy == pytest.approx(direct.energy, abs=1e-12)


def test_a_result_without_a_record_is_refused(h2_pair):
    _, direct, _ = h2_pair
    with pytest.raises(ValueError, match="no neutral-source record"):
        assert_matched_neutral_source([direct, object()], context="bare")


@pytest.mark.slow
@pytest.mark.parametrize("functional", ["pbe", "pbe0"])
def test_ks_producers_record_one_operator(functional):
    """KS adds the functional and the executed XC image reach to the record;
    the two producers grow the same domain (#512), so they still agree."""
    ccm = _h2_cube()
    direct = _quiet(run_ccm_rks_direct, ccm, functional)
    bloch = _quiet(run_ccm_rks_gdf, ccm, functional)
    details = direct.neutral_source.as_dict()["details"]
    assert details["xc"]["functional"] == functional
    assert details["xc"]["image_cutoff_bohr"] == pytest.approx(
        direct.xc_lattice_cutoff_bohr)
    expected = "active" if functional == "pbe0" else "inactive"
    assert details["exchange"]["applicability"] == expected
    assert_matched_neutral_source([direct, bloch], context=f"h2 {functional}")
    assert direct.energy / ccm.n_cells == pytest.approx(bloch.energy, abs=1e-8)


def _manifest_identity(path):
    text = path.read_text(encoding="utf-8")
    match = re.search(r'neutral_source_identity\s*=\s*"([0-9a-f]{64})"', text)
    assert match, text
    return match.group(1)


def test_both_runner_arms_write_the_identity(tmp_path):
    """Through the front door: the two arms' .system manifests carry one
    identity, and the QVF convention sections carry the whole record."""
    from vibeqc import run_periodic_job
    from vibeqc._vibeqc_core import BasisSet

    cell = _h2_cube().unit_system
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    rg = _quiet(run_periodic_job, cell, basis, method="aiccm",
                variant="real-gamma", aiccm_lattice_extension=(2, 1, 1),
                initial_guess="HCORE", output=str(tmp_path / "rg"),
                progress=False)
    nb = _quiet(run_periodic_job, cell, basis, method="aiccm",
                variant="neutral-bloch", aiccm_lattice_extension=(2, 1, 1),
                output=str(tmp_path / "nb"), progress=False)
    assert rg.converged and nb.converged
    rg_id = _manifest_identity(tmp_path / "rg.system")
    nb_id = _manifest_identity(tmp_path / "nb.system")
    assert rg_id == nb_id
    assert rg.real_gamma.neutral_source["identity"] == rg_id
    assert nb.neutral_bloch.neutral_source["identity"] == nb_id
    assert rg.real_gamma.neutral_source["details"] == \
        nb.neutral_bloch.neutral_source["details"]
    assert_matched_neutral_source([rg, nb], context="runner arms")
    assert nb.energy == pytest.approx(rg.energy, abs=1e-9)
