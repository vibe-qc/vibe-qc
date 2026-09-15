from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, Molecule
from vibeqc._vibeqc_core import semiempirical as native
from vibeqc.semiempirical import (
    NDDOExperimentalWarning,
    SemiempiricalRoutePlan,
    native_route_descriptor,
    run_native_energy,
)


def _h2() -> Molecule:
    return Molecule(
        [Atom(1, [0.0, 0.0, -0.7]), Atom(1, [0.0, 0.0, 0.7])]
    )


def _h_doublet() -> Molecule:
    return Molecule([Atom(1, [0.0, 0.0, 0.0])], multiplicity=2)


@pytest.mark.parametrize(
    ("method", "native_method"),
    [
        ("dftb0", native.NativeMethod.DFTB0),
        ("scc-dftb", native.NativeMethod.SCCDFTB),
        ("gfn2", native.NativeMethod.GFN2XTB),
        ("pm6", native.NativeMethod.PM6),
        ("upm6", native.NativeMethod.PM6),
        ("om1", native.NativeMethod.OM1),
        ("om2", native.NativeMethod.OM2),
        ("om3", native.NativeMethod.OM3),
        ("msindo", native.NativeMethod.MSINDO),
        ("nddo", native.NativeMethod.MSINDONDDO),
    ],
)
def test_native_descriptor_maps_canonical_route_variants(method, native_method):
    plan = SemiempiricalRoutePlan.from_request(method)
    if method == "om1":
        with pytest.warns(NDDOExperimentalWarning, match="core-valence ECP"):
            descriptor = native_route_descriptor(plan)
    else:
        descriptor = native_route_descriptor(plan)

    assert descriptor.method == native_method
    assert descriptor.spin == (
        native.NativeSpin.Unrestricted
        if plan.spin == "unrestricted"
        else native.NativeSpin.ClosedShell
    )


def test_native_descriptor_rejects_non_energy_or_orchestrated_routes():
    with pytest.raises(NotImplementedError, match="energy plans only"):
        native_route_descriptor(
            SemiempiricalRoutePlan.from_request(
                "dftb0",
                properties=("gradient",),
            )
        )

    with pytest.raises(NotImplementedError, match="COSMO remains"):
        native_route_descriptor(
            SemiempiricalRoutePlan.from_request("msindo", solvent="water")
        )

    with pytest.raises(NotImplementedError, match="molecular routes only"):
        native_route_descriptor(SemiempiricalRoutePlan.from_request("seccm"))


def test_native_dftb_and_scc_match_direct_bindings():
    from vibeqc.semiempirical.parameters import default_parameters

    molecule = _h2()
    parameters = default_parameters()

    dftb = run_native_energy(
        SemiempiricalRoutePlan.from_request("dftb0"),
        molecule,
        parameters,
    )
    dftb_direct = native.run_dftb0(molecule, parameters)
    assert dftb.energy == dftb_direct.energy
    assert dftb.e_electronic == dftb_direct.e_electronic
    assert dftb.e_repulsive == dftb_direct.e_repulsive
    assert dftb.e_core == 0.0
    assert dftb.e_scc == 0.0
    assert dftb.binding_energy == 0.0
    assert dftb.parameter_identity == dftb_direct.parameter_identity
    assert dftb.parameter_sha256 == dftb_direct.parameter_sha256
    assert dftb.retained_result_bytes > 0
    assert dftb.workspace_bytes == 0
    assert not dftb.memory_counters_complete

    scc = run_native_energy(
        SemiempiricalRoutePlan.from_request("scc-dftb"),
        molecule,
        parameters,
        max_iter=80,
        conv_tol=1.0e-8,
        charge_mixing=0.3,
    )
    options = native.SCCOptions()
    options.max_iter = 80
    options.conv_tol_charge = 1.0e-8
    options.charge_mixing = 0.3
    scc_direct = native.run_scc_dftb(molecule, parameters, options)
    assert scc.energy == scc_direct.energy
    assert scc.e_scc == scc_direct.e_scc
    assert scc.n_iter == scc_direct.n_iter
    assert scc.converged == scc_direct.converged
    assert scc.parameter_identity == scc_direct.parameter_identity
    assert scc.parameter_sha256 == scc_direct.parameter_sha256


def test_native_unrestricted_dftb_matches_direct_binding():
    from vibeqc.semiempirical.parameters import default_parameters

    molecule = _h_doublet()
    parameters = default_parameters()
    plan = SemiempiricalRoutePlan.from_request("dftb0", multiplicity=2)

    facade = run_native_energy(plan, molecule, parameters)
    direct = native.run_udftb0(molecule, parameters)

    assert facade.route.spin == native.NativeSpin.Unrestricted
    assert facade.energy == direct.energy
    assert facade.e_electronic == direct.e_electronic
    assert facade.e_repulsive == direct.e_repulsive


def test_native_gfn2_matches_direct_binding():
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    molecule = _h2()
    parameters = load_gfn2_params()
    facade = run_native_energy(
        SemiempiricalRoutePlan.from_request("gfn2"),
        molecule,
        parameters,
        max_iter=80,
        conv_tol=1.0e-7,
        charge_mixing=0.15,
    )

    options = native.xtb.XTBSccOptions()
    options.max_iter = 80
    options.conv_tol_charge = 1.0e-7
    options.charge_mixing = 0.15
    direct = native.xtb.run_gfn2_xtb(molecule, parameters, options)

    assert facade.energy == direct.energy
    assert facade.e_electronic == direct.e_electronic
    assert facade.e_repulsive == direct.e_repulsive
    assert facade.e_scc == direct.e_scc
    assert facade.n_iter == direct.n_iter
    assert facade.converged == direct.converged
    assert facade.parameter_identity == direct.parameter_identity
    assert facade.parameter_sha256 == direct.parameter_sha256


def test_native_pm6_and_om2_match_direct_bindings():
    from vibeqc.semiempirical.methods.omx_params import load_om2_params
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_params

    molecule = _h2()
    pm6_parameters = load_pm6_params()
    pm6 = run_native_energy(
        SemiempiricalRoutePlan.from_request("pm6"),
        molecule,
        pm6_parameters,
        max_iter=60,
        conv_tol=1.0e-8,
    )
    pm6_direct = native.nddo.run_pm6(
        molecule,
        pm6_parameters,
        60,
        1.0e-8,
    )
    assert pm6.energy == pm6_direct.energy
    assert pm6.e_electronic == pm6_direct.e_electronic
    assert pm6.e_core == pm6_direct.e_core
    assert pm6.e_electronic + pm6.e_core == pytest.approx(
        pm6.energy, abs=1.0e-13
    )
    assert pm6.n_iter == pm6_direct.n_iter
    assert pm6.parameter_identity == pm6_direct.parameter_identity
    assert pm6.parameter_sha256 == pm6_direct.parameter_sha256

    om2_parameters = load_om2_params()
    om2 = run_native_energy(
        SemiempiricalRoutePlan.from_request("om2"),
        molecule,
        om2_parameters,
        max_iter=60,
        conv_tol=1.0e-8,
    )
    om2_direct = native.nddo.run_omx_v2(
        molecule,
        om2_parameters,
        60,
        1.0e-8,
    )
    assert om2.energy == om2_direct.energy
    assert om2.e_electronic == om2_direct.e_electronic
    assert om2.e_core == om2_direct.e_core
    assert om2.e_electronic + om2.e_core == pytest.approx(
        om2.energy, abs=1.0e-13
    )
    assert om2.n_iter == om2_direct.n_iter
    assert om2.parameter_identity == om2_direct.parameter_identity
    assert om2.parameter_sha256 == om2_direct.parameter_sha256


def test_native_unrestricted_pm6_and_om2_preserve_energy_components():
    from vibeqc.semiempirical.methods.omx_params import load_om2_params
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_params

    molecule = _h_doublet()
    pm6_parameters = load_pm6_params()
    upm6 = run_native_energy(
        SemiempiricalRoutePlan.from_request("upm6"),
        molecule,
        pm6_parameters,
        max_iter=80,
        conv_tol=1.0e-8,
    )
    upm6_direct = native.nddo.run_upm6(
        molecule,
        pm6_parameters,
        80,
        1.0e-8,
    )
    assert upm6.energy == upm6_direct.energy
    assert upm6.e_electronic == upm6_direct.e_electronic
    assert upm6.e_core == upm6_direct.e_core
    assert upm6.e_electronic + upm6.e_core == pytest.approx(
        upm6.energy, abs=1.0e-13
    )
    assert upm6.n_basis == upm6_direct.n_basis
    assert upm6.retained_result_bytes > 0

    om2_parameters = load_om2_params()
    uom2 = run_native_energy(
        SemiempiricalRoutePlan.from_request("om2", multiplicity=2),
        molecule,
        om2_parameters,
        max_iter=80,
        conv_tol=1.0e-8,
    )
    uom2_direct = native.nddo.run_uomx_v2(
        molecule,
        om2_parameters,
        80,
        1.0e-8,
    )
    assert uom2.energy == uom2_direct.energy
    assert uom2.e_electronic == uom2_direct.e_electronic
    assert uom2.e_core == uom2_direct.e_core
    assert uom2.e_electronic + uom2.e_core == pytest.approx(
        uom2.energy, abs=1.0e-13
    )
    assert uom2.n_basis == uom2_direct.n_basis
    assert uom2.retained_result_bytes > 0


@pytest.mark.parametrize(
    ("method", "nddo"), [("msindo", False), ("nddo", True)]
)
def test_native_msindo_matches_direct_binding(method, nddo):
    from vibeqc.semiempirical.methods.msindo import (
        ANGSTROM_TO_BOHR,
        _cpp_msindo_kernel,
    )

    molecule = _h2()
    run_msindo_full, _run_msindo_uhf, parameters = _cpp_msindo_kernel(
        nddo=nddo
    )
    facade = run_native_energy(
        SemiempiricalRoutePlan.from_request(method),
        molecule,
        parameters,
        max_iter=80,
        conv_tol=1.0e-9,
    )
    # The direct MSINDO API consumes Angstrom while Molecule stores bohr.
    # Use the engine's pinned 1986-CODATA convention so this comparison also
    # guards that the facade preserves the physical geometry at the boundary.
    coordinates_angstrom = [
        np.asarray(atom.xyz, dtype=float) / ANGSTROM_TO_BOHR
        for atom in molecule.atoms
    ]
    direct = run_msindo_full(
        [atom.Z for atom in molecule.atoms],
        coordinates_angstrom,
        parameters,
        80,
        1.0e-9,
        nddo,
        molecule.charge,
    )

    assert facade.energy == direct.total_energy
    assert facade.e_electronic == direct.electronic_energy
    assert facade.binding_energy == direct.binding_energy
    assert facade.n_iter == direct.n_iter
    assert facade.converged == direct.converged


def test_native_nddo_spdd_energy_matches_source_alcl():
    """The native facade retains the Al-Cl SPDD contribution (#660)."""
    from vibeqc.semiempirical.methods.msindo import (
        ANGSTROM_TO_BOHR,
        _cpp_msindo_kernel,
    )

    molecule = Molecule(
        [
            Atom(13, [0.0, 0.0, 0.0]),
            Atom(17, [0.0, 0.0, 2.10 * ANGSTROM_TO_BOHR]),
        ]
    )
    _run_msindo_full, _run_msindo_uhf, parameters = _cpp_msindo_kernel(
        nddo=True
    )
    result = run_native_energy(
        SemiempiricalRoutePlan.from_request("nddo"),
        molecule,
        parameters,
        max_iter=200,
        conv_tol=1.0e-12,
    )

    assert result.converged
    assert result.energy == pytest.approx(-16.4566380534, abs=1.0e-6)


def test_native_facade_rejects_parameter_family_mismatch():
    from vibeqc.semiempirical.methods.pm6_params import load_pm6_params

    descriptor = native_route_descriptor(
        SemiempiricalRoutePlan.from_request("dftb0")
    )
    with pytest.raises(ValueError, match="does not match PM6 parameters"):
        native.run_native(descriptor, _h2(), load_pm6_params())


def test_native_facade_rejects_invalid_descriptor_before_dispatch():
    from vibeqc.semiempirical.parameters import default_parameters

    parameters = default_parameters()
    descriptor = native.NativeRoute()
    descriptor.method = native.NativeMethod.DFTB0
    descriptor.spin = native.NativeSpin.Unrestricted
    with pytest.raises(ValueError, match="spin does not match"):
        native.run_native(descriptor, _h2(), parameters)

    descriptor.spin = native.NativeSpin.ClosedShell
    descriptor.max_iter = 0
    with pytest.raises(ValueError, match="max_iter must be >= 1"):
        native.run_native(descriptor, _h2(), parameters)
