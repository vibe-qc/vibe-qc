"""GitLab IID 257: the ``rsgdf_ke_cutoff`` default must not floor the planner.

The range-separated (SR/LR) GDF source plans its reciprocal cutoff from the
raw integral error target and treats an explicit ``rsgdf_ke_cutoff`` as a
lower bound. A shipped default of 200 Ha therefore silently replaced the
planner's bound with 200 Ha everywhere: 68,999 instead of 2,007 reciprocal
vectors on the 8-bohr H2 cell below, and a candidate-cap refusal
(513**3 = 135,005,697 > 10,000,000) on the 80-bohr CCM RI cell.

These tests pin the *resolved* cutoff, read back from the executed source,
against the planner.

GitLab IID 206 (maintainer decision, 2026-09-25): an explicit request is used
exactly as given; it is no longer a floor over the planner. A request below
the planner's error-bounded cutoff is honoured with a ``RuntimeWarning``.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc.aux_basis import _plan_range_separated_gdf_cutoffs, make_aux_basis_set

pytestmark = [pytest.mark.periodic, pytest.mark.gdf]

AUX = "def2-svp-jk"


def _h2_box(box_bohr: float = 8.0, sep_bohr: float = 1.4):
    half = 0.5 * sep_bohr
    system = vq.PeriodicSystem(
        3,
        np.diag([box_bohr, box_bohr, box_bohr]),
        [vq.Atom(1, [0, 0, -half]), vq.Atom(1, [0, 0, half])],
    )
    return system, vq.BasisSet(system.unit_cell_molecule(), "sto-3g")


def _planner_ke_cutoff(system, basis):
    # The drivers' defaults: rsgdf_omega=0.4, rsgdf_g_precision=1e-10. The
    # planner's reciprocal tail is uniform in q, so q=0 bounds every k mesh.
    auxiliary = make_aux_basis_set(system.unit_cell_molecule(), aux_name=AUX)
    return _plan_range_separated_gdf_cutoffs(
        system, basis, auxiliary, np.zeros(3), omega=0.4, raw_integral_error=1e-10,
    ).ke_cutoff


def _opts():
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.max_iter = 40
    opts.conv_tol_energy = 1e-9
    opts.lattice_opts.cutoff_bohr = 30.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0
    return opts


def test_resolve_rsgdf_ke_cutoff_splits_request_from_legacy_mesh():
    from vibeqc.periodic_k_gdf import RSGDF_KE_CUTOFF_DEFAULT, _resolve_rsgdf_ke_cutoff

    assert RSGDF_KE_CUTOFF_DEFAULT == 200.0
    assert _resolve_rsgdf_ke_cutoff(None) == (None, 200.0)
    assert _resolve_rsgdf_ke_cutoff(80.0) == (80.0, 80.0)


def test_gamma_default_resolves_to_planner_cutoff():
    from vibeqc.pbc_gdf import run_pbc_gdf_rhf

    system, basis = _h2_box()
    planned = _planner_ke_cutoff(system, basis)
    assert planned < 200.0  # the cell is only meaningful if the floor would bind
    result = run_pbc_gdf_rhf(
        system, basis, _opts(), aux_basis=AUX, gdf_method="rsgdf",
        exxdiv="ewald", progress=False,
    )
    assert result.converged
    assert result.rsgdf_ke_cutoff == pytest.approx(planned, rel=1e-12)


def test_gamma_explicit_cutoff_above_planner_is_used_exactly():
    from vibeqc.pbc_gdf import run_pbc_gdf_rhf

    system, basis = _h2_box()
    result = run_pbc_gdf_rhf(
        system, basis, _opts(), aux_basis=AUX, gdf_method="rsgdf",
        exxdiv="ewald", rsgdf_ke_cutoff=200.0, progress=False,
    )
    assert result.converged
    assert result.rsgdf_ke_cutoff == pytest.approx(200.0)


# GitLab IID 206: below the planner's cutoff (18.86 Ha on this cell) the
# request used to be raised to the planner. It is now used exactly and warns.
BELOW_PLANNER_KE = 12.0


def test_gamma_explicit_cutoff_below_planner_is_used_exactly_and_warns():
    from vibeqc.pbc_gdf import run_pbc_gdf_rhf

    system, basis = _h2_box()
    assert BELOW_PLANNER_KE < _planner_ke_cutoff(system, basis)
    with pytest.warns(RuntimeWarning, match="below the .* SR/LR planner requires"):
        result = run_pbc_gdf_rhf(
            system, basis, _opts(), aux_basis=AUX, gdf_method="rsgdf",
            exxdiv="ewald", rsgdf_ke_cutoff=BELOW_PLANNER_KE, progress=False,
        )
    assert result.converged
    assert result.rsgdf_ke_cutoff == BELOW_PLANNER_KE


def test_multik_explicit_cutoff_below_planner_is_used_exactly_and_warns():
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

    system, basis = _h2_box()
    assert BELOW_PLANNER_KE < _planner_ke_cutoff(system, basis)
    with pytest.warns(RuntimeWarning, match="below the .* SR/LR planner requires"):
        result = run_krhf_periodic_gdf(
            system, basis, (2, 1, 1), _opts(), aux_basis=AUX,
            rsgdf_ke_cutoff=BELOW_PLANNER_KE, progress=False,
        )
    assert result.converged
    assert result.rsgdf_ke_cutoff == BELOW_PLANNER_KE


@pytest.mark.parametrize("request_ke", [None, BELOW_PLANNER_KE, 200.0])
def test_runner_memory_estimate_sizes_the_executed_cutoff(monkeypatch, request_ke):
    """The runner's reciprocal-vector estimate mirrors the source: the planner
    without a request, the request exactly otherwise (IID 206)."""
    import vibeqc.aux_basis as aux_basis
    from vibeqc.periodic_runner import PeriodicJKMethod, _periodic_gdf_estimate

    system, basis = _h2_box()
    reaches = []
    real_box = aux_basis._reciprocal_index_box_per_axis

    def spy(lattice, reach, dim):
        reaches.append(float(reach))
        return real_box(lattice, reach, dim)

    monkeypatch.setattr(aux_basis, "_reciprocal_index_box_per_axis", spy)
    _periodic_gdf_estimate(
        system, basis, resolved_jk=PeriodicJKMethod.GDF, method_upper="RHF",
        functional=None, kpoints=(1, 1, 1), aux_basis=AUX, rsgdf_ke_cutoff=request_ke,
    )
    expected = _planner_ke_cutoff(system, basis) if request_ke is None else request_ke
    assert len(reaches) == 1
    assert reaches[0] ** 2 / 2 == pytest.approx(expected, rel=1e-12)


def test_multik_default_resolves_to_planner_cutoff():
    from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

    system, basis = _h2_box()
    planned = _planner_ke_cutoff(system, basis)
    result = run_krhf_periodic_gdf(
        system, basis, (2, 1, 1), _opts(), aux_basis=AUX, progress=False,
    )
    assert result.converged
    assert result.rsgdf_ke_cutoff == pytest.approx(planned, rel=1e-12)


@pytest.mark.slow
def test_explicit_cutoff_still_refuses_the_ccm_ri_cell():
    """An explicit 200 Ha request is used as given, so the 80-bohr CCM RI cell
    still reaches the product's own candidate-cap refusal."""
    from vibeqc.aux_basis import _RangeSeparatedGdfAdmissionError
    from vibeqc.pbc_gdf import run_pbc_gdf_rhf

    system, basis = _h2_box(box_bohr=80.0)
    with pytest.raises(_RangeSeparatedGdfAdmissionError, match="candidates=135005697"):
        run_pbc_gdf_rhf(
            system, basis, _opts(), aux_basis=AUX, gdf_method="rsgdf",
            exxdiv="ewald", rsgdf_ke_cutoff=200.0, progress=False,
        )
