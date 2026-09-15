"""ECP derivative workflows must preserve the exact SCF Hamiltonian."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import vibeqc
from vibeqc import Atom, BasisSet, ECPCenter, Molecule, RHFOptions


_GRADIENT_WRAPPERS = (
    ("compute_gradient", "_compute_gradient_core"),
    ("compute_gradient_rks", "_compute_gradient_rks_core"),
    ("compute_gradient_uhf", "_compute_gradient_uhf_core"),
    ("compute_gradient_uks", "_compute_gradient_uks_core"),
)


def _h2o() -> Molecule:
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.4, 1.1]),
            Atom(1, [0.0, -1.4, 1.1]),
        ]
    )


def _h2s() -> Molecule:
    return Molecule(
        [
            Atom(16, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.8, 1.3]),
            Atom(1, [0.0, -1.8, 1.3]),
        ]
    )


def _s2() -> Molecule:
    return Molecule(
        [
            Atom(16, [-1.5, 0.0, 0.0]),
            Atom(16, [1.5, 0.0, 0.0]),
        ]
    )


def _one_primitive_inline_block():
    block = vibeqc.ECPPrimitiveBlock()
    block.n_primitive = 1
    block.exponents = [1.25]
    block.coefficients = [-2.0]
    block.ams = [0]
    block.ns = [2]
    return block


@pytest.mark.parametrize(
    ("runner_name", "options_name", "gradient_name", "needs_grid"),
    [
        ("run_rhf", "RHFOptions", "compute_gradient", False),
        ("run_uhf", "UHFOptions", "compute_gradient_uhf", False),
        ("run_rks", "RKSOptions", "compute_gradient_rks", True),
        ("run_uks", "UKSOptions", "compute_gradient_uks", True),
    ],
)
def test_native_gradient_entry_points_reject_mixed_ecp_routes_before_work(
    runner_name,
    options_name,
    gradient_name,
    needs_grid,
):
    """The direct native API has the same mutually-exclusive ECP contract."""
    from vibeqc import _vibeqc_core as core

    molecule = _h2o()
    basis = BasisSet(molecule, "sto-3g")
    scf_options = getattr(core, options_name)()
    # The mixed-route validation must precede even the result-convergence
    # check, so no converged reference or derivative work is needed here.
    scf_options.max_iter = 0
    result = getattr(core, runner_name)(molecule, basis, scf_options)

    gradient_options = core.GradientOptions()
    gradient_options.ecp_centers = [
        ECPCenter(Z=8, xyz=[0.0, 0.0, 0.0])
    ]
    gradient_options.ecp_library = "ecp10mdf"
    gradient_options.ecp_primitive_blocks = [_one_primitive_inline_block()]

    args = [molecule, basis, result]
    if needs_grid:
        args.append(core.GridOptions())
    args.append(gradient_options)
    with pytest.raises(
        ValueError,
        match="XML-library and inline-primitive ECP inputs are mutually exclusive",
    ):
        getattr(core, gradient_name)(*args)


def test_zero_core_result_operator_provenance_is_authoritative():
    from vibeqc.ecp_metadata import molecular_result_has_ecp_operator

    result = SimpleNamespace(
        ecp_operator_applied=True,
        ecp_provenance_verified=True,
        ecp_total_ncore=0,
    )
    assert molecular_result_has_ecp_operator(result)


@pytest.mark.parametrize(
    ("wrapper_name", "core_name"),
    _GRADIENT_WRAPPERS,
)
def test_public_gradient_wrappers_refuse_missing_ecp_operator_options(
    monkeypatch,
    wrapper_name,
    core_name,
):
    def must_not_run(*args, **kwargs):
        raise AssertionError("native gradient ran before the ECP contract check")

    monkeypatch.setattr(vibeqc, core_name, must_not_run)
    result = SimpleNamespace(
        ecp_operator_applied=True,
        ecp_provenance_verified=True,
        ecp_total_ncore=10,
        ecp_xml_centers=[ECPCenter(Z=16, xyz=[0.0, 0.0, 0.0])],
        ecp_xml_library="lanl2dz",
    )
    wrapper = getattr(vibeqc, wrapper_name)
    with pytest.raises(ValueError, match="ECP route does not match"):
        wrapper(_h2s(), object(), result)


def test_public_gradient_refuses_ecp_options_for_all_electron_result(monkeypatch):
    def must_not_run(*args, **kwargs):
        raise AssertionError("native gradient ran before the ECP contract check")

    monkeypatch.setattr(vibeqc, "_compute_gradient_core", must_not_run)
    options = vibeqc.GradientOptions()
    options.ecp_centers = [ECPCenter(Z=8, xyz=[0.0, 0.0, 0.0])]
    options.ecp_library = "ecp10mdf"
    result = SimpleNamespace(
        ecp_operator_applied=False,
        ecp_provenance_verified=True,
        ecp_total_ncore=0,
    )
    with pytest.raises(
        ValueError,
        match="verified SCF result is all-electron",
    ):
        vibeqc.compute_gradient(_h2o(), object(), result, options)


@pytest.mark.parametrize(("wrapper_name", "core_name"), _GRADIENT_WRAPPERS)
def test_public_gradient_refuses_same_ncore_wrong_xml_library(
    monkeypatch,
    wrapper_name,
    core_name,
):
    def must_not_run(*args, **kwargs):
        raise AssertionError("aggregate core count or native gradient was used")

    monkeypatch.setattr(vibeqc, "ecp_effective_charges", must_not_run)
    monkeypatch.setattr(vibeqc, core_name, must_not_run)
    sulfur = ECPCenter(Z=16, xyz=[0.0, 0.0, 0.0])
    result = SimpleNamespace(
        ecp_operator_applied=True,
        ecp_provenance_verified=True,
        ecp_total_ncore=10,
        ecp_xml_centers=[sulfur],
        ecp_xml_library="lanl2dz",
    )
    options = vibeqc.GradientOptions()
    options.ecp_centers = [ECPCenter(Z=16, xyz=[0.0, 0.0, 0.0])]
    options.ecp_library = "ecp10mdf"

    with pytest.raises(ValueError, match="library .* does not match"):
        getattr(vibeqc, wrapper_name)(
            _h2s(), object(), result, options=options
        )


@pytest.mark.parametrize(("wrapper_name", "core_name"), _GRADIENT_WRAPPERS)
def test_public_gradient_refuses_same_z_wrong_xml_center(
    monkeypatch,
    wrapper_name,
    core_name,
):
    def must_not_run(*args, **kwargs):
        raise AssertionError("aggregate core count or native gradient was used")

    monkeypatch.setattr(vibeqc, "ecp_effective_charges", must_not_run)
    monkeypatch.setattr(vibeqc, core_name, must_not_run)
    result = SimpleNamespace(
        ecp_operator_applied=True,
        ecp_provenance_verified=True,
        ecp_total_ncore=10,
        ecp_xml_centers=[ECPCenter(Z=16, xyz=[-1.5, 0.0, 0.0])],
        ecp_xml_library="lanl2dz",
    )
    options = vibeqc.GradientOptions()
    options.ecp_centers = [ECPCenter(Z=16, xyz=[1.5, 0.0, 0.0])]
    options.ecp_library = "lanl2dz"

    with pytest.raises(ValueError, match="centers do not match"):
        getattr(vibeqc, wrapper_name)(
            _s2(), object(), result, options=options
        )


@pytest.mark.parametrize(("wrapper_name", "core_name"), _GRADIENT_WRAPPERS)
def test_public_gradient_accepts_exact_xml_provenance(
    monkeypatch,
    wrapper_name,
    core_name,
):
    expected = np.arange(9, dtype=float).reshape(3, 3)
    monkeypatch.setattr(
        vibeqc,
        "ecp_effective_charges",
        lambda *args, **kwargs: [6.0, 1.0, 1.0],
    )
    monkeypatch.setattr(
        vibeqc,
        core_name,
        lambda *args, **kwargs: expected,
    )
    result = SimpleNamespace(
        ecp_operator_applied=True,
        ecp_provenance_verified=True,
        ecp_total_ncore=10,
        ecp_xml_centers=[ECPCenter(Z=16, xyz=[0.0, 0.0, 0.0])],
        ecp_xml_library="lanl2dz",
    )
    options = vibeqc.GradientOptions()
    options.ecp_centers = [ECPCenter(Z=16, xyz=[0.0, 0.0, 0.0])]
    options.ecp_library = "lanl2dz"

    actual = getattr(vibeqc, wrapper_name)(
        _h2s(), object(), result, options=options
    )
    assert np.array_equal(actual, expected)


def test_public_gradient_refuses_unverified_scf_provenance(monkeypatch):
    def must_not_run(*args, **kwargs):
        raise AssertionError("native gradient ran without exact ECP provenance")

    monkeypatch.setattr(vibeqc, "_compute_gradient_core", must_not_run)
    result = SimpleNamespace(
        ecp_operator_applied=False,
        ecp_provenance_verified=False,
        ecp_total_ncore=0,
    )

    with pytest.raises(
        NotImplementedError,
        match="SCF ECP provenance is unknown or unverified",
    ):
        vibeqc.compute_gradient(_h2o(), object(), result)


@pytest.mark.parametrize(("wrapper_name", "core_name"), _GRADIENT_WRAPPERS)
def test_public_gradient_accepts_exact_inline_provenance(
    monkeypatch,
    wrapper_name,
    core_name,
):
    expected = np.arange(9, dtype=float).reshape(3, 3)
    monkeypatch.setattr(vibeqc, core_name, lambda *args, **kwargs: expected)

    block = _one_primitive_inline_block()
    centers = [[0.0, 0.0, 0.0]]
    charges = [6.0, 1.0, 1.0]
    result = SimpleNamespace(
        ecp_operator_applied=True,
        ecp_provenance_verified=True,
        ecp_xml_centers=[],
        ecp_xml_library="",
        ecp_primitive_blocks=[block],
        ecp_primitive_centers=centers,
        ecp_effective_charges=charges,
        ecp_total_ncore=10,
    )
    options = vibeqc.GradientOptions()
    options.ecp_primitive_blocks = [block]
    options.ecp_primitive_centers = centers
    options.ecp_effective_charges = charges
    options.ecp_total_ncore = 10

    actual = getattr(vibeqc, wrapper_name)(
        _h2s(), object(), result, options=options
    )
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        pytest.param(
            "ecp_primitive_centers",
            [[0.0, 0.0, 0.1]],
            "block/center pairs do not match",
            id="center",
        ),
        pytest.param(
            "ecp_effective_charges",
            [5.0, 1.0, 1.0],
            "effective charges do not match",
            id="charges",
        ),
        pytest.param(
            "ecp_total_ncore",
            9,
            "total_ncore .* does not match",
            id="ncore",
        ),
    ],
)
def test_inline_gradient_provenance_rejects_exact_route_mismatch(
    field,
    value,
    message,
):
    from vibeqc.ecp_metadata import validate_gradient_ecp_contract

    block = _one_primitive_inline_block()
    payload = {
        "ecp_primitive_blocks": [block],
        "ecp_primitive_centers": [[0.0, 0.0, 0.0]],
        "ecp_effective_charges": [6.0, 1.0, 1.0],
        "ecp_total_ncore": 10,
    }
    result = SimpleNamespace(
        ecp_operator_applied=True,
        ecp_provenance_verified=True,
        ecp_xml_centers=[],
        ecp_xml_library="",
        **payload,
    )
    option_payload = dict(payload)
    option_payload[field] = value
    options = SimpleNamespace(
        ecp_centers=[],
        ecp_library="",
        **option_payload,
    )

    with pytest.raises(ValueError, match=message):
        validate_gradient_ecp_contract(
            _h2s(),
            result,
            options,
            route="test-gradient",
        )


def test_arbitrary_low_level_hcore_cannot_certify_xml_ecp_provenance():
    """ECP options alone are not evidence about a caller-built Hcore."""
    from vibeqc import _vibeqc_core as core

    molecule = _h2o()
    basis = BasisSet(molecule, "sto-3g")
    overlap = np.asarray(core.compute_overlap(basis))
    hcore = np.asarray(core.compute_kinetic(basis)) + np.asarray(
        core.compute_nuclear(basis, molecule)
    )
    options = RHFOptions()
    options.max_iter = 0
    options.ecp_centers = [ECPCenter(Z=8, xyz=[0.0, 0.0, 0.0])]
    options.ecp_library = "ecp10mdf"
    result = core.run_rhf_scf_with_jk(
        basis,
        molecule.n_electrons(),
        overlap,
        hcore,
        float(molecule.nuclear_repulsion()),
        core.make_direct_jk_builder(basis),
        options,
    )

    assert not result.ecp_operator_applied
    assert not result.ecp_provenance_verified
    assert list(result.ecp_xml_centers) == []
    assert result.ecp_xml_library == ""
    assert list(result.ecp_primitive_blocks) == []
    assert list(result.ecp_primitive_centers) == []
    assert list(result.ecp_effective_charges) == []
    assert result.ecp_total_ncore == 0
    gradient_options = vibeqc.GradientOptions()
    gradient_options.ecp_centers = list(options.ecp_centers)
    gradient_options.ecp_library = options.ecp_library
    with pytest.raises(
        NotImplementedError,
        match="SCF ECP provenance is unknown or unverified",
    ):
        vibeqc.compute_gradient(
            molecule,
            basis,
            result,
            gradient_options,
        )


@pytest.mark.parametrize("manual_ecp", [False, True], ids=["sidecar", "manual"])
def test_run_job_refuses_hf_cis_ecp_gradient_before_output_or_scf(
    tmp_path,
    monkeypatch,
    manual_ecp,
):
    import vibeqc.runner as runner_module

    def work_must_not_start(*args, **kwargs):
        raise AssertionError("output or SCF started before the TD-ECP guard")

    monkeypatch.setattr(runner_module, "OutputWriter", work_must_not_start)
    monkeypatch.setattr(runner_module, "run_rhf", work_must_not_start)

    options = None
    basis_name = "lanl2dz"
    if manual_ecp:
        basis_name = "def2-svp"
        options = RHFOptions()
        options.ecp_centers = [ECPCenter(Z=16, xyz=[0.0, 0.0, 0.0])]
        options.ecp_library = "lanl2dz"

    with pytest.raises(
        NotImplementedError,
        match="does not yet preserve a molecular ECP Hamiltonian",
    ):
        vibeqc.run_job(
            _h2s(),
            basis=basis_name,
            method="rhf",
            rhf_options=options,
            tddft=True,
            tddft_type="tda",
            tddft_gradient=True,
            output=tmp_path / f"td-ecp-{manual_ecp}",
        )
    assert not list(tmp_path.iterdir())


def test_run_job_refuses_zero_core_mp2_optimization_before_output_or_scf(
    tmp_path,
    monkeypatch,
):
    """A safe MP2 ECP single point does not make its derivative supported."""
    import vibeqc.runner as runner_module

    def work_must_not_start(*args, **kwargs):
        raise AssertionError("output or SCF started before the MP2-ECP guard")

    monkeypatch.setattr(runner_module, "OutputWriter", work_must_not_start)
    monkeypatch.setattr(runner_module, "run_rhf", work_must_not_start)
    options = RHFOptions()
    options.ecp_primitive_blocks = [_one_primitive_inline_block()]
    options.ecp_total_ncore = 0

    with pytest.raises(
        NotImplementedError,
        match="only RHF/UHF/RKS/UKS mean-field ECP gradients are supported",
    ):
        vibeqc.run_job(
            _h2s(),
            basis="def2-svp",
            method="mp2",
            rhf_options=options,
            optimize=True,
            output=tmp_path / "zero-core-mp2-opt",
        )
    assert not list(tmp_path.iterdir())


def test_run_job_refuses_ecp_cpcm_hessian_before_output_or_scf(
    tmp_path,
    monkeypatch,
):
    """An ECP finite-difference Hessian cannot use gas-phase gradients."""
    import vibeqc.runner as runner_module

    def work_must_not_start(*args, **kwargs):
        raise AssertionError("output or SCF started before the ECP+CPCM guard")

    monkeypatch.setattr(runner_module, "OutputWriter", work_must_not_start)
    monkeypatch.setattr(runner_module, "run_rhf", work_must_not_start)

    with pytest.raises(
        NotImplementedError,
        match=r"ECP\+CPCM molecular gradients are not implemented",
    ):
        vibeqc.run_job(
            _h2s(),
            basis="lanl2dz",
            method="rhf",
            solvent="water",
            hessian=True,
            output=tmp_path / "ecp-cpcm-hessian",
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("manual_ecp", [False, True])
def test_hf_cis_energy_fn_refuses_ecp_before_basis_or_scf(
    monkeypatch,
    manual_ecp,
):
    import vibeqc._vibeqc_core as core
    from vibeqc.excited_gradient import make_hf_cis_energy_fn

    def basis_must_not_run(*args, **kwargs):
        raise AssertionError("BasisSet constructed before the ECP contract check")

    monkeypatch.setattr(core, "BasisSet", basis_must_not_run)
    options = None
    basis_name = "lanl2dz"
    if manual_ecp:
        basis_name = "sto-3g"
        options = RHFOptions()
        options.ecp_centers = [ECPCenter(Z=16, xyz=[0.0, 0.0, 0.0])]
        options.ecp_library = "ecp10mdf"
    energy_fn = make_hf_cis_energy_fn(
        [16, 1, 1],
        basis_name,
        rhf_options=options,
    )
    coords_angstrom = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.9, 0.7], [0.0, -0.9, 0.7]]
    )
    with pytest.raises(
        NotImplementedError,
        match="molecular ECP excited-state gradients and optimizations",
    ):
        energy_fn(coords_angstrom)


@pytest.mark.parametrize(
    ("module_name", "function_name"),
    [
        ("vibeqc.hessian_casscf", "compute_hessian_casscf"),
        ("vibeqc.hessian_caspt2", "compute_hessian_caspt2"),
    ],
)
def test_multireference_hessians_refuse_ecp_before_basis(
    monkeypatch,
    module_name,
    function_name,
):
    import importlib

    module = importlib.import_module(module_name)

    def basis_must_not_run(*args, **kwargs):
        raise AssertionError("BasisSet constructed before the ECP contract check")

    monkeypatch.setattr(module, "BasisSet", basis_must_not_run)
    with pytest.raises(NotImplementedError, match="molecular ECP derivatives"):
        getattr(module, function_name)(
            _h2s(),
            "lanl2dz",
            active_space=(2, 2),
        )


@pytest.mark.parametrize(
    ("module_name", "function_name", "extra_kwargs"),
    [
        (
            "vibeqc.gradient._casscf",
            "compute_casscf_gradient",
            {"rdm1": np.zeros((1, 1)), "rdm2": np.zeros((1, 1, 1, 1))},
        ),
        (
            "vibeqc.gradient._caspt2",
            "compute_caspt2_gradient",
            {"rdm1": np.zeros((1, 1)), "rdm2": np.zeros((1, 1, 1, 1))},
        ),
        (
            "vibeqc.gradient._nevpt2",
            "compute_nevpt2_gradient",
            {"rdm1": np.zeros((1, 1)), "rdm2": np.zeros((1, 1, 1, 1))},
        ),
        (
            "vibeqc.gradient._ms_caspt2_nac",
            "compute_ms_caspt2_nac",
            {
                "n_active_elec": 2,
                "sa_weights": [0.5, 0.5],
                "nroots": 2,
                "state_pair": (0, 1),
            },
        ),
        (
            "vibeqc.gradient._ms_caspt2_grad",
            "compute_ms_caspt2_gradient",
            {
                "n_active_elec": 2,
                "sa_weights": [0.5, 0.5],
                "nroots": 2,
            },
        ),
    ],
)
def test_direct_multireference_derivatives_refuse_ecp_before_work(
    module_name,
    function_name,
    extra_kwargs,
):
    import importlib

    function = getattr(importlib.import_module(module_name), function_name)
    with pytest.raises(NotImplementedError, match="molecular ECP derivatives"):
        function(
            _h2s(),
            SimpleNamespace(name="lanl2dz"),
            np.zeros((1, 1)),
            np.zeros((1, 1)),
            np.zeros((1, 1, 1, 1)),
            0,
            1,
            **extra_kwargs,
        )


def test_direct_rohf_gradient_refuses_ecp_before_work(monkeypatch):
    from vibeqc import _vibeqc_core as core
    from vibeqc.rohf import compute_rohf_gradient

    def derivative_must_not_run(*args, **kwargs):
        raise AssertionError("ROHF derivative work ran before the ECP guard")

    monkeypatch.setattr(
        core,
        "one_electron_gradient_contribution",
        derivative_must_not_run,
    )
    with pytest.raises(NotImplementedError, match="molecular ECP derivatives"):
        compute_rohf_gradient(
            _h2s(),
            SimpleNamespace(name="lanl2dz"),
            SimpleNamespace(method="rohf"),
        )


@pytest.mark.parametrize(
    ("module_name", "function_name", "result_name"),
    [
        ("vibeqc.hessian_analytic", "compute_hessian_rhf_analytic", "rhf_result"),
        (
            "vibeqc.hessian_analytic_uhf",
            "compute_hessian_uhf_analytic",
            "uhf_result",
        ),
        (
            "vibeqc.hessian_analytic_rks",
            "compute_hessian_rks_analytic",
            "rks_result",
        ),
        (
            "vibeqc.hessian_analytic_uks",
            "compute_hessian_uks_analytic",
            "uks_result",
        ),
    ],
)
def test_analytic_hessians_refuse_ecp_result_before_integrals(
    module_name,
    function_name,
    result_name,
):
    import importlib

    function = getattr(importlib.import_module(module_name), function_name)
    result = SimpleNamespace(
        ecp_operator_applied=True,
        ecp_total_ncore=0,
    )
    with pytest.raises(NotImplementedError, match="molecular ECP derivatives"):
        function(
            _h2o(),
            object(),
            **{result_name: result},
            basis_name="unused",
        )


def test_cpcm_gradient_refuses_effective_nuclear_charges_before_work():
    from vibeqc.solvation.gradient import cpcm_gradient

    result = SimpleNamespace(
        ecp_operator_applied=False,
        ecp_total_ncore=0,
    )
    solvent_result = SimpleNamespace(z_eff=np.array([6.0, 1.0, 1.0]))
    with pytest.raises(NotImplementedError, match="ECP nuclear derivatives"):
        cpcm_gradient(result, _h2o(), object(), solvent_result)


def test_qvf_requested_mo_occupation_uses_effective_electron_count(monkeypatch):
    import vibeqc.cube as cube
    from vibeqc.output.formats.qvf import qvf_mo_data

    grid = SimpleNamespace(origin=np.zeros(3), spacing=np.ones(3))
    monkeypatch.setattr(cube, "make_uniform_grid", lambda *args, **kwargs: grid)
    monkeypatch.setattr(
        cube,
        "_mo_on_grid",
        lambda *args, **kwargs: np.zeros((1, 1, 1)),
    )
    result = SimpleNamespace(
        mo_coeffs=np.ones((1, 5)),
        mo_energies=np.arange(5, dtype=float),
        ecp_total_ncore=2,
    )
    [orbital] = qvf_mo_data(result, object(), _h2o(), [4])
    assert orbital["occupation"] == 0.0


def test_qvf_requested_alpha_mo_uses_effective_spin_occupation(monkeypatch):
    import vibeqc.cube as cube
    from vibeqc.output.formats.qvf import qvf_mo_data

    grid = SimpleNamespace(origin=np.zeros(3), spacing=np.ones(3))
    monkeypatch.setattr(cube, "make_uniform_grid", lambda *args, **kwargs: grid)
    monkeypatch.setattr(
        cube,
        "_mo_on_grid",
        lambda *args, **kwargs: np.zeros((1, 1, 1)),
    )
    molecule = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.4, 1.1]),
            Atom(1, [0.0, -1.4, 1.1]),
        ],
        charge=0,
        multiplicity=3,
    )
    result = SimpleNamespace(
        mo_coeffs_alpha=np.ones((1, 6)),
        mo_energies_alpha=np.arange(6, dtype=float),
        ecp_total_ncore=2,
    )
    [orbital] = qvf_mo_data(result, object(), molecule, [4])
    assert orbital["occupation"] == 1.0


def test_periodic_qvf_adapter_supplies_ecp_count_to_occupation_fallbacks():
    from vibeqc.output.formats.qvf import qvf_bloch_wf_data, qvf_wf_data
    from vibeqc.periodic_runner import _result_with_ecp_ncore

    class _Shell:
        l = 0
        pure = True
        atom_index = 0
        exponents = [0.5]
        coefficients = [1.0]

    class _Basis:
        def shells(self):
            return [_Shell()]

    class _PhysicalSixElectronCell:
        def n_electrons(self):
            return 6

    result = SimpleNamespace(
        mo_coeffs=np.ones((1, 4)),
        mo_energies=np.arange(4, dtype=float),
        occupations=np.empty(0),
    )
    adapted = _result_with_ecp_ncore(result, 2)
    molecule = _PhysicalSixElectronCell()

    wf_data = qvf_wf_data(adapted, _Basis(), molecule)
    assert wf_data["mo_metadata"]["occupations"] == pytest.approx(
        [2.0, 2.0, 0.0, 0.0]
    )

    bloch_data = qvf_bloch_wf_data(
        adapted,
        _Basis(),
        molecule,
        k_points=[[0.0, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        bloch_data["occupations"][0],
        [2.0, 2.0, 0.0, 0.0],
    )
