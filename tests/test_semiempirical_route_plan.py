from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc.semiempirical import SemiempiricalRoutePlan
from vibeqc.semiempirical.routes import (
    BOUNDARY_CCM_DIRECT_TORUS,
    BOUNDARY_MOLECULE,
    BOUNDARY_PERIODIC_GAMMA,
    BOUNDARY_SECCM_DIRECT_TORUS,
    EXECUTION_NATIVE,
    EXECUTION_NATIVE_BATCHED_FD,
    EXECUTION_PYTHON_ORCHESTRATION,
    GFN2SECCMHamiltonianIdentity,
    GFN2SECCMRestartIdentity,
    GFN2SECCMRunControls,
    MATURITY_EXPERIMENTAL,
    MATURITY_MIXED_NATIVE,
    is_semiempirical_method,
    plan_periodic_semiempirical_route,
)
from vibeqc.semiempirical.status import semiempirical_route_status


@pytest.mark.parametrize(
    "aliases",
    [
        ("dftb0", "dftb", "dftb-0"),
        ("scc_dftb", "scc-dftb", "sccdftb"),
        ("gfn2_xtb", "gfn2", "gfn2-xtb", "gfn2xtb", "gfn2 xtb"),
        ("om2",),
    ],
)
def test_route_plan_method_aliases_share_one_representation(aliases):
    plans = [SemiempiricalRoutePlan.from_request(alias) for alias in aliases]
    assert {plan.route_key for plan in plans} == {plans[0].route_key}
    assert {plan.boundary for plan in plans} == {BOUNDARY_MOLECULE}
    assert {plan.properties for plan in plans} == {("energy",)}


@pytest.mark.parametrize("method", ["om1", "om2", "om3"])
def test_molecular_omx_routes_are_explicitly_experimental(method):
    energy = SemiempiricalRoutePlan.from_request(method)
    gradient = SemiempiricalRoutePlan.from_request(
        method,
        properties=("energy", "gradient"),
    )

    assert energy.maturity == MATURITY_EXPERIMENTAL
    assert energy.execution == EXECUTION_NATIVE
    assert gradient.maturity == MATURITY_EXPERIMENTAL
    assert gradient.execution == EXECUTION_NATIVE_BATCHED_FD


def test_route_plan_canonicalizes_spin_and_msindo_aliases():
    assert SemiempiricalRoutePlan.from_request(
        "upm6"
    ) == SemiempiricalRoutePlan.from_request("pm6", multiplicity=2)

    assert SemiempiricalRoutePlan.from_request(
        "nddo"
    ) == SemiempiricalRoutePlan.from_request("msindo", nddo=True)

    seccm = SemiempiricalRoutePlan.from_request("seccm")
    aliases = (
        SemiempiricalRoutePlan.from_request("se-ccm"),
        SemiempiricalRoutePlan.from_request("ccm"),
        SemiempiricalRoutePlan.from_request("msindo-ccm", boundary="ccm"),
        SemiempiricalRoutePlan.from_request("msindo", boundary="seccm"),
    )
    assert all(plan == seccm for plan in aliases)
    assert seccm.method_key == "msindo"
    assert seccm.variant == "indo"
    assert seccm.boundary == BOUNDARY_SECCM_DIRECT_TORUS
    assert BOUNDARY_CCM_DIRECT_TORUS == BOUNDARY_SECCM_DIRECT_TORUS


def test_molecular_msindo_nddo_gradient_route_is_explicit_and_cited():
    plan = SemiempiricalRoutePlan.from_request(
        "msindo",
        nddo=True,
        properties=("energy", "gradient"),
    )

    assert plan.status_route == "msindo-nddo-gradient-analytic"
    assert plan.maturity == MATURITY_MIXED_NATIVE
    assert plan.execution == EXECUTION_PYTHON_ORCHESTRATION
    assert plan.citation_assemble_kwargs["extra_entries"] == (
        "dewar_thiel_1977",
        "voigt_1973",
    )
    status = semiempirical_route_status(plan)
    assert status.route == "msindo-nddo-gradient-analytic"
    assert status.backend == "mixed-native"
    assert status.production


@pytest.mark.parametrize(
    ("method", "dimension", "family", "expected_kernel"),
    [
        ("scc_dftb", 1, "madelung", "madelung_wire_1d"),
        ("scc_dftb", 2, "madelung", "madelung_parry_2d"),
        ("gfn2", 2, "ewald_gamma", "ewald_gamma_parry_2d"),
        ("gfn2", 3, "ewald_gamma", "ewald_gamma_ewald_3d"),
        ("om2", 2, "none", "none"),
        # MSINDO keeps the frozen truncated 1-D lattice sum
        # (indo::_madelung_potential_1d), not the Parry-type wire Ewald the
        # other adapters run -- so it carries its own kernel name and never
        # reaches routes.methods.wire_ewald_1d. Its 2-D/3-D kernels are the
        # shared _madkonst_2d / _madkonst_3d machinery. GitLab #442.
        ("ccm", 1, "madelung", "madelung_truncated_1d"),
        ("ccm", 2, "madelung", "madelung_parry_2d"),
        ("ccm", 3, "madelung", "madelung_ewald_3d"),
        ("ccm", 1, "none", "none"),
    ],
)
def test_seccm_runtime_plan_records_concrete_citation_context(
    method, dimension, family, expected_kernel
):
    plan = SemiempiricalRoutePlan.from_request(
        method, boundary="seccm"
    ).with_seccm_runtime(
        periodic_dimension=dimension,
        electrostatics_family=family,
        electronic_temperature=0.002,
    )

    assert plan.periodic_dimension == dimension
    assert plan.electrostatics_kernel == expected_kernel
    assert plan.electronic_temperature == pytest.approx(0.002)
    assert not plan.truncated_electrostatics_acknowledged
    assert plan.citation_assemble_kwargs["seccm_dimension"] == dimension
    assert (
        plan.citation_assemble_kwargs["seccm_electrostatics_kernel"]
        == expected_kernel
    )
    assert plan.citation_assemble_kwargs["electronic_temperature"] == 0.002


def test_omx_seccm_runtime_plan_records_truncated_electrostatics_ack():
    base = SemiempiricalRoutePlan.from_request("om2", boundary="seccm")
    acknowledged = base.with_seccm_runtime(
        periodic_dimension=1,
        truncated_electrostatics_acknowledged=True,
    )

    assert acknowledged.electrostatics_kernel == "none"
    assert acknowledged.truncated_electrostatics_acknowledged
    assert acknowledged != base.with_seccm_runtime(periodic_dimension=1)

    with pytest.raises(ValueError, match="must be boolean"):
        base.with_seccm_runtime(
            periodic_dimension=1,
            truncated_electrostatics_acknowledged="true",
        )
    with pytest.raises(ValueError, match="only valid for OM2-/OM3-SECCM"):
        SemiempiricalRoutePlan.from_request(
            "pm6", boundary="seccm"
        ).with_seccm_runtime(
            periodic_dimension=1,
            truncated_electrostatics_acknowledged=True,
        )


def test_seccm_runtime_plan_rejects_invalid_dimension_and_temperature():
    plan = SemiempiricalRoutePlan.from_request("gfn2", boundary="seccm")
    with pytest.raises(ValueError, match="dimension must be 1, 2, or 3"):
        plan.with_seccm_runtime(periodic_dimension=0)
    with pytest.raises(ValueError, match="finite and non-negative"):
        plan.with_seccm_runtime(
            periodic_dimension=2,
            electronic_temperature=-0.001,
        )


def test_gfn2_seccm_hamiltonian_identity_is_canonical_and_round_trippable():
    base = SemiempiricalRoutePlan.from_request("gfn2", boundary="seccm")
    plan = base.with_gfn2_seccm_runtime(
        periodic_dimension=2,
        requested_electrostatics_family=" EWALD_GAMMA ",
        resolved_electrostatics_family="none",
        requested_electronic_temperature=0.0,
        resolved_electronic_temperature=0.005,
        include_aes=True,
        madelung_s_weighted=False,
        madelung_no_self=False,
        ewald_gamma_molecular_onsite=True,
        molecular_delegated=True,
        run_controls=GFN2SECCMRunControls.from_request(),
    )
    identity = plan.gfn2_hamiltonian_identity
    assert identity is not None
    assert identity.requested_electrostatics_family == "ewald_gamma"
    assert identity.resolved_electrostatics_kernel == "none"
    assert identity.requested_electronic_temperature == 0.0
    assert identity.resolved_electronic_temperature == pytest.approx(0.005)
    assert identity.ewald_gamma_molecular_onsite
    assert identity.molecular_delegated
    assert plan.electrostatics_kernel == "none"
    assert plan.electronic_temperature == pytest.approx(0.005)

    payload = json.loads(json.dumps(identity.to_dict()))
    restored = GFN2SECCMHamiltonianIdentity.from_dict(payload)
    assert restored == identity
    assert hash(restored) == hash(identity)

    changed_runtime = plan.with_seccm_runtime(
        periodic_dimension=3,
        electrostatics_family="none",
        electronic_temperature=0.1,
    )
    assert changed_runtime.gfn2_hamiltonian_identity is None

    with pytest.raises(ValueError, match="include_aes must be boolean"):
        GFN2SECCMHamiltonianIdentity.from_dict(
            {**payload, "include_aes": "false"}
        )
    with pytest.raises(ValueError, match="resolved electrostatics kernel"):
        GFN2SECCMHamiltonianIdentity.from_dict(
            {**payload, "resolved_electrostatics_kernel": "invented"}
        )
    with pytest.raises(ValueError, match="include_aes=True"):
        GFN2SECCMHamiltonianIdentity.from_dict(
            {**payload, "include_aes": False}
        )


def test_gfn2_seccm_hamiltonian_identity_validates_dependent_options():
    base = SemiempiricalRoutePlan.from_request("gfn2", boundary="seccm")
    with pytest.raises(ValueError, match="Madelung modifiers require"):
        base.with_gfn2_seccm_runtime(
            periodic_dimension=2,
            requested_electrostatics_family="none",
            resolved_electrostatics_family="none",
            requested_electronic_temperature=0.0,
            resolved_electronic_temperature=0.0,
            include_aes=True,
            madelung_s_weighted=True,
            madelung_no_self=False,
            ewald_gamma_molecular_onsite=False,
            molecular_delegated=False,
            run_controls=GFN2SECCMRunControls.from_request(),
        )


def test_gfn2_seccm_restart_identity_is_canonical_and_round_trippable():
    omitted = GFN2SECCMRestartIdentity.from_initial_shell_charges(None)
    empty = GFN2SECCMRestartIdentity.from_initial_shell_charges([])
    assert empty == omitted
    assert omitted.primary_source == "neutral"
    assert omitted.n_shell_charges == 0
    assert omitted.shell_charges_sha256 is None
    assert omitted.automatic_retry_source == "neutral"

    supplied = GFN2SECCMRestartIdentity.from_initial_shell_charges(
        [0.0, -0.0, 0.25]
    )
    equivalent = GFN2SECCMRestartIdentity.from_initial_shell_charges(
        [-0.0, 0.0, 0.25]
    )
    altered = GFN2SECCMRestartIdentity.from_initial_shell_charges(
        [0.0, 0.0, 0.5]
    )
    assert supplied == equivalent
    assert supplied != altered
    assert supplied.primary_source == "supplied"
    assert supplied.n_shell_charges == 3
    assert len(supplied.shell_charges_sha256 or "") == 64

    payload = json.loads(json.dumps(supplied.to_dict()))
    restored = GFN2SECCMRestartIdentity.from_dict(payload)
    assert restored == supplied
    assert hash(restored) == hash(supplied)


@pytest.mark.parametrize(
    "payload",
    [
        {
            "schema_version": 2,
            "primary_source": "neutral",
            "n_shell_charges": 0,
            "shell_charges_sha256": None,
            "automatic_retry_source": "neutral",
        },
        {
            "schema_version": 1,
            "primary_source": "neutral",
            "n_shell_charges": 1,
            "shell_charges_sha256": None,
            "automatic_retry_source": "neutral",
        },
        {
            "schema_version": 1,
            "primary_source": "supplied",
            "n_shell_charges": 1,
            "shell_charges_sha256": "not-a-sha256",
            "automatic_retry_source": "neutral",
        },
        {
            "schema_version": 1,
            "primary_source": "supplied",
            "n_shell_charges": 1,
            "shell_charges_sha256": "a" * 64,
            "automatic_retry_source": "supplied",
        },
    ],
)
def test_gfn2_seccm_restart_identity_rejects_inconsistent_payloads(payload):
    with pytest.raises(ValueError):
        GFN2SECCMRestartIdentity.from_dict(payload)


@pytest.mark.parametrize(
    "restart",
    [np.array([np.nan]), np.array([np.inf]), np.zeros((1, 1)), "0.1"],
    ids=["nan", "infinity", "two-dimensional", "text"],
)
def test_gfn2_seccm_restart_identity_rejects_invalid_vectors(restart):
    with pytest.raises(ValueError):
        GFN2SECCMRestartIdentity.from_initial_shell_charges(restart)


def test_gfn2_seccm_run_controls_defaults_round_trip_and_attach_to_route():
    default = GFN2SECCMRunControls.from_request()
    explicit = GFN2SECCMRunControls.from_request(
        parameter_set="gfn2-xtb-parameter-registry",
        max_iter=3600,
        conv_tol_charge=1.0e-6,
        charge_mixing=0.1,
        scc_mixer="simple",
        ewald_gamma_k0_global=False,
        initial_shell_charges=None,
    )
    assert explicit == default
    assert default.finite_torus_gap_tolerance == 1.0e-8
    payload = json.loads(json.dumps(default.to_dict()))
    restored = GFN2SECCMRunControls.from_dict(payload)
    assert restored == default
    assert hash(restored) == hash(default)

    base = SemiempiricalRoutePlan.from_request("gfn2", boundary="seccm")
    plan = base.with_gfn2_seccm_runtime(
        periodic_dimension=2,
        requested_electrostatics_family="ewald_gamma",
        resolved_electrostatics_family="ewald_gamma",
        requested_electronic_temperature=0.0,
        resolved_electronic_temperature=0.0,
        include_aes=True,
        madelung_s_weighted=False,
        madelung_no_self=False,
        ewald_gamma_molecular_onsite=False,
        molecular_delegated=False,
        run_controls=restored,
    )
    assert plan.gfn2_run_controls is restored
    assert plan.route_key == base.route_key

    changed_runtime = plan.with_seccm_runtime(
        periodic_dimension=3,
        electrostatics_family="none",
    )
    assert changed_runtime.gfn2_hamiltonian_identity is None
    assert changed_runtime.gfn2_run_controls is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_iter": 3599},
        {"conv_tol_charge": 2.0e-6},
        {"charge_mixing": 0.2},
        {"scc_mixer": "diis"},
        {"ewald_gamma_k0_global": True},
        {"initial_shell_charges": [0.1, -0.1]},
        {"initial_shell_charges": [0.0, 0.0]},
    ],
)
def test_gfn2_seccm_run_controls_distinguish_altered_requests(overrides):
    default = GFN2SECCMRunControls.from_request()
    altered = GFN2SECCMRunControls.from_request(**overrides)
    assert altered != default

    base = SemiempiricalRoutePlan.from_request("gfn2", boundary="seccm")
    common = {
        "periodic_dimension": 2,
        "requested_electrostatics_family": "ewald_gamma",
        "resolved_electrostatics_family": "ewald_gamma",
        "requested_electronic_temperature": 0.0,
        "resolved_electronic_temperature": 0.0,
        "include_aes": True,
        "madelung_s_weighted": False,
        "madelung_no_self": False,
        "ewald_gamma_molecular_onsite": False,
        "molecular_delegated": False,
    }
    default_plan = base.with_gfn2_seccm_runtime(
        **common,
        run_controls=default,
    )
    altered_plan = base.with_gfn2_seccm_runtime(
        **common,
        run_controls=altered,
    )
    assert altered_plan.gfn2_hamiltonian_identity == (
        default_plan.gfn2_hamiltonian_identity
    )
    assert altered_plan.route_key == default_plan.route_key
    assert altered_plan != default_plan


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", 2, "schema_version"),
        ("parameter_set", "invented", "parameter_set"),
        ("max_iter", 0, "max_iter"),
        ("max_iter", True, "max_iter"),
        ("conv_tol_charge", float("nan"), "conv_tol_charge"),
        ("charge_mixing", 1.1, "charge_mixing"),
        ("scc_mixer", "anderson", "scc_mixer"),
        (
            "requested_ewald_gamma_k0_global",
            "false",
            "must be boolean",
        ),
        (
            "resolved_ewald_gamma_k0_policy",
            "rank-one-dipole",
            "resolved Ewald K=0 policy",
        ),
        (
            "finite_torus_gap_tolerance",
            2.0e-8,
            "finite_torus_gap_tolerance",
        ),
        ("restart", None, "restart"),
    ],
)
def test_gfn2_seccm_run_controls_reject_invalid_payloads(
    field, value, message
):
    payload = GFN2SECCMRunControls.from_request().to_dict()
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        GFN2SECCMRunControls.from_dict(payload)


def test_route_plan_records_charge_in_canonical_representation():
    neutral = SemiempiricalRoutePlan.from_request("pm6")
    cation = SemiempiricalRoutePlan.from_request("pm6", charge=1)

    assert neutral.charge == 0
    assert cation.charge == 1
    assert neutral.route_key != cation.route_key


def test_route_plan_periodic_gamma_aliases_share_route():
    plan = SemiempiricalRoutePlan.from_request("scc-dftb", boundary="periodic")
    assert plan.method_key == "scc_dftb"
    assert plan.boundary == BOUNDARY_PERIODIC_GAMMA
    assert plan.status_route == "dftb"

    gfn2 = SemiempiricalRoutePlan.from_request(
        "gfn2 xtb",
        boundary="periodic-gamma",
    )
    assert gfn2.method_key == "gfn2_xtb"
    assert gfn2.boundary == BOUNDARY_PERIODIC_GAMMA


def test_periodic_system_planner_owns_alias_charge_and_spin():
    system = SimpleNamespace(
        lattice=object(),
        unit_cell=(),
        charge=1,
        multiplicity=2,
        n_electrons=lambda: 1,
    )
    plan = plan_periodic_semiempirical_route("scc-dftb", system)

    assert plan.method_key == "scc_dftb"
    assert plan.boundary == BOUNDARY_PERIODIC_GAMMA
    assert plan.charge == 1
    assert plan.spin == "unrestricted"
    assert plan_periodic_semiempirical_route(plan, system) is plan


def test_periodic_system_planner_rejects_mismatch_and_unrestricted_nddo():
    closed = SimpleNamespace(
        lattice=object(),
        unit_cell=(),
        charge=0,
        multiplicity=1,
        n_electrons=lambda: 2,
    )
    charged = SimpleNamespace(
        lattice=object(),
        unit_cell=(),
        charge=1,
        multiplicity=1,
        n_electrons=lambda: 2,
    )
    plan = plan_periodic_semiempirical_route("dftb0", closed)

    with pytest.raises(ValueError, match="does not match the periodic system"):
        plan_periodic_semiempirical_route(plan, charged)

    radical = SimpleNamespace(
        lattice=object(),
        unit_cell=(),
        charge=0,
        multiplicity=2,
        n_electrons=lambda: 1,
    )
    with pytest.raises(NotImplementedError, match="no Bloch-periodic Gamma"):
        plan_periodic_semiempirical_route("om2", radical)


def test_known_semiempirical_method_detection_includes_boundary_aliases():
    assert is_semiempirical_method("gfn2 xtb")
    assert is_semiempirical_method("upm6")
    assert is_semiempirical_method("seccm")
    assert not is_semiempirical_method("RHF")


def test_route_plan_rejects_unsupported_boundaries_and_properties():
    dftb_k = SemiempiricalRoutePlan.from_request("scc-dftb", boundary="periodic_k")
    assert dftb_k.boundary == "periodic_k"
    assert dftb_k.status_route == "periodic-scc-dftb-kpoint"
    assert dftb_k.maturity == "experimental"

    bands = SemiempiricalRoutePlan.from_request(
        "dftb0",
        boundary="periodic_k",
        properties=("bands",),
    )
    assert bands.properties == ("bands",)
    assert bands.status_route == "periodic-dftb0-kpoint"

    with pytest.raises(NotImplementedError, match="energy/band/gradient/stress"):
        SemiempiricalRoutePlan.from_request("gfn2", boundary="periodic_k")

    gradient = SemiempiricalRoutePlan.from_request(
        "dftb0",
        boundary="periodic_k",
        properties=("energy", "gradient"),
    )
    assert gradient.status_route == "periodic-dftb0-kpoint-gradient-fd"
    assert gradient.execution == EXECUTION_NATIVE_BATCHED_FD
    assert semiempirical_route_status(gradient).route == (
        "periodic-dftb0-kpoint-gradient-fd"
    )

    stress = SemiempiricalRoutePlan.from_request(
        "scc-dftb",
        boundary="periodic_k",
        properties=("energy", "stress"),
    )
    assert stress.status_route == "periodic-scc-dftb-kpoint-gradient-fd"
    assert stress.execution == EXECUTION_NATIVE_BATCHED_FD
    assert semiempirical_route_status(stress).route == (
        "periodic-scc-dftb-kpoint-gradient-fd"
    )

    with pytest.raises(NotImplementedError, match="no Bloch-periodic Gamma"):
        SemiempiricalRoutePlan.from_request("msindo", boundary="periodic_gamma")

    for method in ("pm7", "om1", "om2", "om3"):
        with pytest.raises(NotImplementedError, match="no Bloch-periodic Gamma"):
            SemiempiricalRoutePlan.from_request(
                method,
                boundary="periodic_gamma",
            )

    with pytest.raises(NotImplementedError, match="bands"):
        SemiempiricalRoutePlan.from_request("gfn2", properties=("bands",))

    with pytest.raises(ValueError, match="ccm_options"):
        SemiempiricalRoutePlan.from_request("pm6", ccm_options=object())

    pm6_seccm = SemiempiricalRoutePlan.from_request("pm6", boundary="seccm")
    assert pm6_seccm.status_route == "pm6-seccm-energy"
    assert pm6_seccm.maturity == "experimental"

    # pm7 has no SECCM adapter yet (the PM6-SECCM engine is PM6-parameter
    # scoped; PM7 SECCM follows the same engine later).
    with pytest.raises(NotImplementedError, match="current SECCM implementation"):
        SemiempiricalRoutePlan.from_request("pm7", boundary="seccm")

    with pytest.raises(ValueError, match="denotes the SECCM boundary"):
        SemiempiricalRoutePlan.from_request("ccm", boundary="molecule")

    with pytest.raises(NotImplementedError, match="closed-shell only"):
        SemiempiricalRoutePlan.from_request("seccm", multiplicity=2)


def test_route_plan_gates_unvalidated_molecular_spin_variants():
    assert SemiempiricalRoutePlan.from_request(
        "scc-dftb",
        multiplicity=2,
    ).spin == "unrestricted"
    assert SemiempiricalRoutePlan.from_request(
        "pm6",
        multiplicity=3,
    ).variant == "upm6"

    with pytest.raises(NotImplementedError, match="GFN2-xTB route is closed-shell"):
        SemiempiricalRoutePlan.from_request("gfn2", multiplicity=2)

    with pytest.raises(NotImplementedError, match="NDDO mode is closed-shell"):
        SemiempiricalRoutePlan.from_request("msindo", multiplicity=2, nddo=True)


def test_route_status_consumes_route_plan_status_route():
    dftb_k = SemiempiricalRoutePlan.from_request(
        "dftb0",
        boundary="periodic_k",
    )
    assert semiempirical_route_status(dftb_k).route == "periodic-dftb0-kpoint"

    periodic_pm6 = SemiempiricalRoutePlan.from_request(
        "pm6",
        boundary="periodic_gamma",
    )
    assert semiempirical_route_status(periodic_pm6).route == "periodic-pm6"

    pm6_gradient = SemiempiricalRoutePlan.from_request(
        "pm6",
        properties=("forces",),
    )
    assert pm6_gradient.properties == ("gradient",)
    assert pm6_gradient.execution == EXECUTION_NATIVE_BATCHED_FD
    assert semiempirical_route_status(pm6_gradient).route == "pm6-gradient-fd"


def test_seccm_route_status_describes_current_native_capabilities():
    scc_gradient = SemiempiricalRoutePlan.from_request(
        "scc_dftb",
        boundary="seccm",
        properties=("energy", "gradient"),
    )
    scc_summary = semiempirical_route_status(scc_gradient).summary
    assert "analytic fixed-charge Madelung derivative" in scc_summary
    assert "no coupled-perturbed SCC charge term remains" in scc_summary
    assert "central differences" not in scc_summary

    omx = SemiempiricalRoutePlan.from_request("om2", boundary="seccm")
    omx_summary = semiempirical_route_status(omx).summary
    assert "explicit truncated-electrostatics acknowledgement" in omx_summary
    assert "not a quantitative solid-state result" in omx_summary

    gfn2 = SemiempiricalRoutePlan.from_request("gfn2", boundary="seccm")
    gfn2_summary = semiempirical_route_status(gfn2).summary
    assert "multi-replica" in gfn2_summary
    assert "native shell-resolved SCC engine" in gfn2_summary
    assert "fail closed" not in gfn2_summary


@pytest.mark.parametrize(
    "method",
    ("dftb0", "scc_dftb", "pm6", "om2", "om3", "gfn2"),
)
def test_generic_executor_fails_closed_for_topology_bound_seccm_methods(
    monkeypatch, method
):
    from vibeqc.semiempirical import runner

    plan = SemiempiricalRoutePlan.from_request(method, boundary="seccm")

    def unexpected_msindo_call(*args, **kwargs):
        pytest.fail("non-MSINDO SECCM plan was misrouted to MSINDO")

    monkeypatch.setattr(runner, "run_ccm", unexpected_msindo_call)
    with pytest.raises(
        NotImplementedError,
        match="must use its explicit run_\\*_seccm adapter",
    ):
        runner._run_semiempirical_plan(plan, object())


def test_pm6_seccm_fd_gradient_is_invariant_to_topology_length_unit():
    from vibeqc import Atom, Molecule
    from vibeqc.molecule import ANGSTROM_TO_BOHR
    from vibeqc.semiempirical import run_pm6_seccm
    from vibeqc.semiempirical.seccm import (
        bind_finite_group,
        build_seccm_topology,
    )

    coords_bohr = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0]])
    cyclic_translation_bohr = np.array([40.0, 0.0, 0.0])
    molecule = Molecule(
        [Atom(1, coordinate.tolist()) for coordinate in coords_bohr],
        0,
        1,
    )

    def molecular_limit_topology(length_unit):
        unit_scale = (
            1.0 if length_unit == "bohr" else 1.0 / ANGSTROM_TO_BOHR
        )
        coords = coords_bohr * unit_scale
        cyclic_translation = cyclic_translation_bohr * unit_scale
        topology = build_seccm_topology(
            coords,
            [cyclic_translation],
            length_unit=length_unit,
            geometry_quantum=1.0e-10,
        )
        return bind_finite_group(
            topology,
            primitive_vectors=[cyclic_translation],
            replicas=(1, 1, 1),
            geometry_tolerance=1.0e-9,
            length_unit=length_unit,
        )

    results = [
        run_pm6_seccm(
            molecule,
            molecular_limit_topology(length_unit),
            max_iter=200,
            conv_tol=1.0e-9,
            compute_gradient=True,
            gradient_fd_step=1.0e-4,
        )
        for length_unit in ("bohr", "angstrom")
    ]

    assert results[0].energy == results[1].energy
    np.testing.assert_allclose(results[0].gradient, results[1].gradient, atol=1.0e-10)


# --------------------------------------------------------------------------- #
# One canonical maturity vocabulary (#150).
# --------------------------------------------------------------------------- #


def _every_reachable_plan():
    """Yield every route plan reachable through the public planner.

    The sweep is the enumeration the #150 audit ran by hand: each known
    method, crossed with every boundary, property set, and the charge/spin
    combinations that select a distinct route.  Requests the validators
    refuse are skipped -- they never reach a maturity.
    """
    import itertools

    from vibeqc.semiempirical.routes import (
        BOUNDARY_PERIODIC_K,
        SEMIEMPIRICAL_METHODS,
    )

    methods = sorted(
        SEMIEMPIRICAL_METHODS | {"upm6", "upm7", "msindo", "om1", "om2", "om3"}
    )
    boundaries = (
        BOUNDARY_MOLECULE,
        BOUNDARY_PERIODIC_GAMMA,
        BOUNDARY_PERIODIC_K,
        BOUNDARY_SECCM_DIRECT_TORUS,
    )
    propsets = (
        None,
        ("energy",),
        ("gradient",),
        ("stress",),
        ("gradient", "stress"),
    )
    spins = ((0, 1, False), (0, 3, True), (1, 2, True))

    for method, boundary, properties in itertools.product(
        methods, boundaries, propsets
    ):
        for charge, multiplicity, unrestricted in spins:
            try:
                yield SemiempiricalRoutePlan.from_request(
                    method,
                    boundary=boundary,
                    properties=properties,
                    charge=charge,
                    multiplicity=multiplicity,
                    unrestricted=unrestricted,
                )
            except (NotImplementedError, ValueError, KeyError, TypeError):
                continue


def test_every_plan_maturity_is_in_the_canonical_vocabulary():
    """No route invents a maturity word outside the one vocabulary."""
    from vibeqc.semiempirical.routes import SEMIEMPIRICAL_MATURITIES

    plans = list(_every_reachable_plan())
    assert plans, "route sweep produced no plans"
    seen = {plan.maturity for plan in plans}
    assert seen <= set(SEMIEMPIRICAL_MATURITIES), (
        f"maturities outside the canonical vocabulary: "
        f"{sorted(seen - set(SEMIEMPIRICAL_MATURITIES))}"
    )


def test_route_status_production_flag_tracks_plan_maturity():
    """``status.py`` and ``routes.py`` state one maturity, not two (#150).

    ``production`` is the single field the two tables share.  Before the #150
    ruling they disagreed on molecular PM6: the plan said ``production`` while
    the status entry said ``production=False`` and its summary said
    heavy-heavy two-center parity was still open.
    """
    from vibeqc.semiempirical.routes import MATURITY_PRODUCTION

    disagreements = {}
    for plan in _every_reachable_plan():
        status = semiempirical_route_status(plan)
        expected = plan.maturity == MATURITY_PRODUCTION
        if status.production != expected:
            disagreements[plan.status_route] = (
                plan.maturity,
                status.production,
            )

    assert not disagreements, (
        "routes.py maturity and status.py production disagree for "
        f"{disagreements}"
    )


def test_molecular_pm6_energy_is_not_a_production_claim():
    """Molecular PM6 is a prescreening route, not a production claim (#150)."""
    plan = SemiempiricalRoutePlan.from_request(
        "pm6", boundary=BOUNDARY_MOLECULE, properties=("energy",)
    )
    assert plan.maturity == MATURITY_EXPERIMENTAL
    assert plan.execution == EXECUTION_NATIVE
    assert not semiempirical_route_status(plan).production


def test_native_registry_carries_no_second_maturity_vocabulary():
    """The dormant C++ ``PeriodicTier`` is gone (#150).

    It was written in one place, read nowhere, and disagreed with the live
    Python classification for gfn2-xtb, msindo, pm6, and om1-3.
    """
    _semi = pytest.importorskip("vibeqc._vibeqc_core.semiempirical")

    assert not hasattr(_semi, "PeriodicTier")
    config = _semi.SemiempiricalMethodConfig()
    assert not hasattr(config, "periodic_tier")
