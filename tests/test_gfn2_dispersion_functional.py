"""Internal D4 energy, potential, SCC and nuclear-derivative acceptance."""

from dataclasses import replace
import json
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import Atom, Molecule
from vibeqc._vibeqc_core import eeq_coordination_numbers
from vibeqc.dispersion_d4_model import (
    compute_c8, compute_d4_c6_pair, d4_bj_damping_energy,
)
from vibeqc.semiempirical.methods.gfn2 import _gfn2_d4_execution_snapshot
from vibeqc.semiempirical.methods.gfn2_dispersion import build_d4_charge_functional


@pytest.fixture
def operands():
    mol = Molecule([
        Atom(8, [0., 0., 0.]), Atom(1, [1.5, 0., 1.1]),
        Atom(1, [-1.4, 0.1, 1.2]), Atom(8, [0.2, 0.3, 5.5]),
        Atom(1, [1.6, 0.4, 6.8]), Atom(1, [-1.3, 0.2, 6.7]),
    ])
    execution = _gfn2_d4_execution_snapshot()
    kernel = build_d4_charge_functional(
        mol, execution.reference_data, execution.parameters,
    )
    return mol, execution, kernel


@pytest.mark.parametrize("q", [np.zeros(6), np.array([-.4, .2, .2, -.5, .3, .2])])
def test_matrix_energy_matches_independent_pair_sum(operands, q):
    mol, execution, kernel = operands
    cn = eeq_coordination_numbers(mol)
    p = execution.parameters
    pair_sum = 0.
    for a, atom_a in enumerate(mol.atoms):
        for b, atom_b in enumerate(list(mol.atoms)[:a]):
            c6 = compute_d4_c6_pair(
                cn[a], q[a], atom_a.Z, cn[b], q[b], atom_b.Z,
                execution.reference_data,
            )
            distance = np.linalg.norm(np.array(atom_a.xyz)-atom_b.xyz)
            pair_sum += d4_bj_damping_energy(
                distance, c6, compute_c8(c6, atom_a.Z, atom_b.Z),
                p.s6, p.s8, p.a1, p.a2,
            )
    energy, _ = kernel.energy_and_potential(q)
    assert energy-kernel.neutral_atm_energy == pytest.approx(pair_sum, abs=1e-15, rel=0)
    assert kernel.neutral_atm_energy > 1e-10


@pytest.mark.parametrize("q", [np.zeros(6), np.array([-.4, .2, .2, -.5, .3, .2])])
def test_charge_potential_is_energy_derivative(operands, q):
    _, _, kernel = operands
    _, potential = kernel.energy_and_potential(q)
    for step in (1e-4, 5e-5):
        numeric = np.empty(6)
        for a in range(6):
            dq = np.eye(6)[a] * step
            numeric[a] = (kernel.energy_and_potential(q+dq)[0]
                          - kernel.energy_and_potential(q-dq)[0])/(2*step)
        np.testing.assert_allclose(potential, numeric, atol=1e-10, rtol=0)


def test_mulliken_fock_has_variational_sign_and_pair_factor(operands):
    """A density perturbation independently checks -1/2 S(v_A+v_B)."""
    _, _, kernel = operands
    owners = np.array([0, 0, 1, 2, 3, 3, 4, 5])
    rng = np.random.default_rng(184)
    matrix = rng.normal(size=(8, 8)) * .03
    overlap = np.eye(8) + matrix + matrix.T
    density = np.diag([3., 3., 1., 1., 3., 3., 1., 1.])
    valence = np.array([6., 1., 1., 6., 1., 1.])

    def charges(p):
        return valence-np.bincount(owners, weights=np.diag(p@overlap), minlength=6)

    _, v = kernel.energy_and_potential(charges(density))
    fock = -.5 * overlap * (v[owners, None] + v[None, owners])
    for _ in range(3):
        direction = rng.normal(size=(8, 8))
        direction = (direction+direction.T)/2
        direction -= np.eye(8) * np.trace(direction@overlap)/np.trace(overlap)
        analytic = np.sum(direction*fock)
        h = 1e-5
        numeric = (kernel.energy_and_potential(charges(density+h*direction))[0]
                   - kernel.energy_and_potential(charges(density-h*direction))[0])/(2*h)
        assert analytic == pytest.approx(numeric, abs=1e-10, rel=0)
        assert abs(analytic) > 1e-8


def test_neutral_atm_has_no_charge_potential(operands):
    _, _, kernel = operands
    pair_only = replace(kernel, neutral_atm_energy=0.)
    for q in (np.zeros(6), np.array([-.5, .1, .4, -.1, 0., .1])):
        energy, potential = kernel.energy_and_potential(q)
        e2, v2 = pair_only.energy_and_potential(q)
        assert energy-e2 == pytest.approx(kernel.neutral_atm_energy, abs=1e-16, rel=0)
        np.testing.assert_array_equal(potential, v2)


def test_kernel_is_immutable_and_permutation_covariant(operands):
    mol, execution, kernel = operands
    for name in ("kernel", "reference_atoms", "cn_weights", "positions_bohr"):
        with pytest.raises(ValueError):
            getattr(kernel, name).setflags(write=True)
    order = [4, 2, 0, 5, 3, 1]
    permuted = build_d4_charge_functional(
        Molecule([list(mol.atoms)[i] for i in order]),
        execution.reference_data, execution.parameters,
    )
    q = np.array([-.4, .2, .2, -.5, .3, .2])
    e, v = kernel.energy_and_potential(q)
    ep, vp = permuted.energy_and_potential(q[order])
    assert ep == pytest.approx(e, abs=1e-15, rel=0)
    np.testing.assert_allclose(vp, v[order], atol=1e-15, rtol=0)


@pytest.mark.parametrize("charges", [[0.], [0.]*5+[float("nan")], [[0.]*6]])
def test_invalid_charge_arrays_are_rejected(operands, charges):
    with pytest.raises(ValueError, match="finite atom-sized"):
        operands[2].energy_and_potential(charges)


def test_saturated_anionic_charge_is_finite(operands):
    energy, potential = operands[2].energy_and_potential(np.full(6, -100.))
    assert np.isfinite(energy)
    np.testing.assert_array_equal(potential, np.zeros(6))


@pytest.mark.parametrize("value", [float("nan"), -1.])
def test_invalid_reference_coefficients_are_rejected(operands, value):
    mol, execution, _ = operands
    ref = execution.reference_data
    invalid = SimpleNamespace(
        get_cns=ref.get_cns, get_qs=ref.get_qs,
        get_n_gaussian=ref.get_n_gaussian, get_c6_ref=lambda *args: value,
    )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        build_d4_charge_functional(mol, invalid, execution.parameters)


def test_missing_element_is_not_silently_omitted(operands):
    _, execution, _ = operands
    mol = Molecule([Atom(11, [0., 0., 0.])], 1, 1)
    with pytest.raises(ValueError, match="unavailable for Z=11"):
        build_d4_charge_functional(mol, execution.reference_data, execution.parameters)


def test_atomic_limit_has_no_interatomic_dispersion(operands):
    _, execution, _ = operands
    atom = Molecule([Atom(8, [0., 0., 0.])])
    kernel = build_d4_charge_functional(atom, execution.reference_data, execution.parameters)
    for q in (-1., 0., 1.):
        energy, potential = kernel.energy_and_potential([q])
        assert energy == 0.
        np.testing.assert_array_equal(potential, np.zeros(1))


def test_fixed_charge_geometry_gradient_matches_two_steps(operands):
    mol, execution, kernel = operands
    q = np.array([-.4, .2, .2, -.5, .3, .2])
    _, _, gradient = kernel.energy_potential_gradient(q)
    xyz = np.array([a.xyz for a in mol.atoms])
    numbers = [a.Z for a in mol.atoms]
    for h in (2e-4, 1e-4):
        numeric = np.empty_like(xyz)
        for a in range(len(xyz)):
            for axis in range(3):
                values = []
                for sign in (1, -1):
                    moved = xyz.copy()
                    moved[a, axis] += sign*h
                    displaced = Molecule([Atom(z, p) for z, p in zip(numbers, moved)])
                    candidate = build_d4_charge_functional(
                        displaced, execution.reference_data, execution.parameters,
                    )
                    values.append(candidate.energy_and_potential(q)[0])
                numeric[a, axis] = (values[0]-values[1])/(2*h)
        np.testing.assert_allclose(gradient, numeric, atol=2e-10, rtol=0)
    np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=1e-15)
    np.testing.assert_allclose(np.cross(xyz, gradient).sum(axis=0), 0., atol=1e-15)


def test_native_operands_match_python_energy_potential_and_gradient(operands):
    kernel = operands[2]
    native = kernel.to_native()
    assert len(native.sha256) == 64
    for q in (np.zeros(6), np.array([-.4, .2, .2, -.5, .3, .2])):
        for actual, expected in zip(native.evaluate(q), kernel.energy_potential_gradient(q)):
            np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0)


def test_native_snapshot_identity_binds_derivatives_and_detaches_arrays(operands):
    kernel = operands[2]
    mutable = replace(kernel, cn_weights=kernel.cn_weights.copy())
    native = mutable.to_native()
    before = native.evaluate(np.zeros(6))
    mutable.cn_weights[:] *= .9
    assert native.sha256 != mutable.to_native().sha256
    for value, expected in zip(native.evaluate(np.zeros(6)), before):
        np.testing.assert_array_equal(value, expected)
    changed_derivative = replace(kernel, cn_jacobian=kernel.cn_jacobian*.9)
    assert changed_derivative.to_native().sha256 != kernel.to_native().sha256


@pytest.mark.parametrize("field,value", [
    ("cn_jacobian", np.zeros((1, 1))),
    ("reference_atoms", np.array([-1])),
    ("neutral_atm_energy", float("nan")),
])
def test_native_rejects_malformed_operand_packets(operands, field, value):
    with pytest.raises(ValueError, match="GFN2 D4"):
        replace(operands[2], **{field: value}).to_native()


def _coupled_case(name):
    if name == "ch_doublet":
        return Molecule([Atom(6, [0., 0., 0.]),
                         Atom(1, [0., 0., 1.1199*1.8897261254535])],
                        charge=0, multiplicity=2), 300*3.166811563e-6
    if name in ("h2_cation_zero", "h2_cation_smeared"):
        return Molecule([Atom(1, [0., 0., 0.]), Atom(1, [0., 0., 4.])],
                        charge=1, multiplicity=2), (0. if name.endswith("zero") else .02)
    if name == "water":
        return Molecule([Atom(8, [0., 0., 0.]), Atom(1, [1.5, 0., 1.1]),
                         Atom(1, [-1.4, .1, 1.2])]), 300*3.166811563e-6
    if name == "formaldehyde":
        return Molecule([Atom(6, [0., 0., 0.]), Atom(8, [0., 0., 2.3]),
                         Atom(1, [1.8, 0., -1.1]), Atom(1, [-1.7, .1, -1.2])]), 300*3.166811563e-6
    return Molecule([Atom(1, [0., 0., 0.]), Atom(1, [0., 0., 4.])],
                    charge=2 if name == "bare_protons" else 0), .02


def _coupled_options(temperature):
    from vibeqc._vibeqc_core.semiempirical import xtb
    opts = xtb.XTBSccOptions()
    opts.aes_faithful = True
    opts.max_iter = 1200
    opts.conv_tol_charge = 1e-10
    opts.electronic_temperature = temperature
    opts.auto_stabilize = False
    return opts


@pytest.mark.parametrize("case", [
    "water", "formaldehyde", "fractional_h2", "bare_protons", "ch_doublet",
    "h2_cation_zero", "h2_cation_smeared",
])
def test_coupled_scc_gradient_matches_free_energy(case, record_property):
    from vibeqc._vibeqc_core import semiempirical as semi
    from vibeqc._vibeqc_core.semiempirical import xtb
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    mol, temperature = _coupled_case(case)
    execution = _gfn2_d4_execution_snapshot()
    params = load_gfn2_params()
    opts = _coupled_options(temperature)

    def run(molecule):
        kernel = build_d4_charge_functional(
            molecule, execution.reference_data, execution.parameters,
        )
        data = kernel.to_native()
        result = xtb.run_gfn2_xtb(molecule, params, opts, data)
        assert result.converged
        assert result.dispersion_sha256 == data.sha256
        assert result.e_dispersion == pytest.approx(
            kernel.energy_and_potential(result.charges)[0], abs=1e-14, rel=0,
        )
        assert result.energy == pytest.approx(sum(getattr(result, field) for field in (
            "e_band0", "e_scc", "e_aes", "e_3rd", "e_repulsive", "e_dispersion",
        )), abs=1e-12, rel=0)
        return result

    result = run(mol)
    gradient = semi.compute_gfn2_gradient(mol, result, params)
    if case in ("fractional_h2", "h2_cation_smeared"):
        assert np.any((result.occupations > 1e-3) & (result.occupations < 1.999))
        assert result.entropy > 1e-3
    xyz = np.array([a.xyz for a in mol.atoms])
    numbers = [a.Z for a in mol.atoms]
    samples = []
    for h in (2e-4, 1e-4):
        numeric = np.empty_like(xyz)
        for a in range(len(xyz)):
            for axis in range(3):
                values = []
                for sign in (1, -1):
                    moved = xyz.copy()
                    moved[a, axis] += sign*h
                    displaced = Molecule([Atom(z, p) for z, p in zip(numbers, moved)],
                                         mol.charge, mol.multiplicity)
                    values.append(run(displaced).free_energy)
                numeric[a, axis] = (values[0]-values[1])/(2*h)
        samples.append(numeric.tolist())
        np.testing.assert_allclose(gradient, numeric, atol=2e-6, rtol=0)
    record_property("analytic_gradient", json.dumps(gradient.tolist()))
    record_property("finite_difference_gradients", json.dumps(samples))
    record_property("dispersion_hartree", result.e_dispersion)
    record_property("free_energy_hartree", result.free_energy)
    record_property("charges", json.dumps(result.charges.tolist()))
    record_property("occupations_alpha", json.dumps(result.occupations_alpha.tolist()))
    record_property("occupations_beta", json.dumps(result.occupations_beta.tolist()))
    record_property("entropy", result.entropy)
    np.testing.assert_allclose(gradient.sum(axis=0), 0., atol=1e-8)


def test_coupling_changes_the_density_and_lowers_variational_energy():
    from vibeqc._vibeqc_core.semiempirical import xtb
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    mol, temperature = _coupled_case("water")
    execution = _gfn2_d4_execution_snapshot()
    kernel = build_d4_charge_functional(mol, execution.reference_data, execution.parameters)
    params = load_gfn2_params()
    opts = _coupled_options(temperature)
    bare = xtb.run_gfn2_xtb(mol, params, opts)
    coupled = xtb.run_gfn2_xtb(mol, params, opts, kernel.to_native())
    assert bare.converged and coupled.converged
    assert bare.e_dispersion == 0. and bare.dispersion_sha256 == ""
    post = bare.free_energy + kernel.energy_and_potential(bare.charges)[0]
    assert coupled.free_energy < post-1e-9
    assert np.max(np.abs(coupled.charges-bare.charges)) > 1e-5


def test_dispersion_requires_matching_geometry_and_variational_aes(operands):
    from vibeqc._vibeqc_core import semiempirical as semi
    from vibeqc._vibeqc_core.semiempirical import xtb
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    mol, _, kernel = operands
    data, params = kernel.to_native(), load_gfn2_params()
    opts = _coupled_options(0.)
    opts.aes_faithful = False
    with pytest.raises(ValueError, match="faithful AES"):
        xtb.run_gfn2_xtb(mol, params, opts, data)
    opts.aes_faithful = True
    displaced = Molecule([Atom(a.Z, np.array(a.xyz)+[.1, 0., 0.]) for a in mol.atoms])
    with pytest.raises(ValueError, match="geometry"):
        xtb.run_gfn2_xtb(displaced, params, opts, data)
    result = xtb.run_gfn2_xtb(mol, params, opts, data)
    assert result.converged
    with pytest.raises(ValueError, match="geometry"):
        semi.compute_gfn2_gradient(displaced, result, params)


def test_exhausted_scc_retains_dispersion_identity(operands):
    from vibeqc._vibeqc_core.semiempirical import xtb
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    mol, _, kernel = operands
    data = kernel.to_native()
    opts = _coupled_options(0.)
    opts.max_iter = 1
    result = xtb.run_gfn2_xtb(mol, load_gfn2_params(), opts, data)
    assert not result.converged
    assert result.dispersion_sha256 == data.sha256


def test_failed_retry_ladder_retains_coupled_functional_identity():
    from vibeqc._vibeqc_core import semiempirical as semi
    from vibeqc._vibeqc_core.semiempirical import xtb
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    mol, temperature = _coupled_case("water")
    execution = _gfn2_d4_execution_snapshot()
    data = build_d4_charge_functional(
        mol, execution.reference_data, execution.parameters,
    ).to_native()
    opts = _coupled_options(temperature)
    opts.auto_stabilize = True
    opts.max_iter = 2500
    opts.conv_tol_charge = 0.  # Strict '< 0' forces all retry paths to exhaust.
    params = load_gfn2_params()
    result = xtb.run_gfn2_xtb(mol, params, opts, data)
    assert not result.converged and len(result.attempts) > 1
    assert result.selected_attempt_index == -1
    assert result.dispersion_sha256 == data.sha256
    assert result.e_dispersion == pytest.approx(data.evaluate(result.charges)[0], abs=1e-14, rel=0)
    last = result.attempts[-1]
    assert result.n_iter == sum(attempt.n_iter for attempt in result.attempts)
    assert result.n_iter <= opts.max_iter
    opts.auto_stabilize = False
    opts.max_iter = last.allocated_max_iter
    opts.charge_mixing = last.ladder_charge_mixing
    opts.electronic_temperature = last.electronic_temperature
    opts.scc_mixer = last.scc_mixer
    direct = xtb.run_gfn2_xtb(mol, params, opts, data)
    assert not direct.converged
    assert result.free_energy == pytest.approx(direct.free_energy, abs=1e-12, rel=0)
    np.testing.assert_allclose(result.charges, direct.charges, atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match="converged full SCC Fock"):
        semi.compute_gfn2_gradient(mol, result, params)
