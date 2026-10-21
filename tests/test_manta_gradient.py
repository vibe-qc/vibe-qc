"""Bounded all-electron HF Lagrangian derivatives and stationary HF forces."""
import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc._vibeqc_core import ShellInfo


def _fixture(pure=True):
    positions = [[0., .1, -.2], [.2, -.3, 1.1], [-.4, .5, .7]]
    mol = vq.Molecule([vq.Atom(2, p) for p in positions], 0, 1)
    shells = []
    for i, l in enumerate([1, 0, 2]):
        s = ShellInfo(); s.atom_index, s.l, s.pure, s.origin = i, l, pure, positions[i]
        s.exponents, s.coefficients = [.7+.1*i, 1.4+.1*i], [.8, -.2]
        shells.append(s)
    basis = vq.BasisSet(mol, shells, "manta-gradient", False)
    rng = np.random.default_rng(329)
    matrices = []
    for _ in range(3):
        d = rng.normal(size=(basis.nbasis, basis.nbasis))*.1
        matrices.append(np.ascontiguousarray(d+d.T))
    return mol, basis, matrices


def _moved(mol, basis, direction, h):
    atoms = [vq.Atom(a.Z, (np.array(a.xyz)+h*direction[i]).tolist()) for i, a in enumerate(mol.atoms)]
    moved = vq.Molecule(atoms, mol.charge, mol.multiplicity)
    shells = basis.shells()
    for s in shells:
        s.origin = atoms[s.atom_index].xyz
    return moved, vq.BasisSet(moved, shells, basis.name, True)


def _one(basis, operation, mol=None):
    request = manta.OneElectronRequest(); request.operation = operation
    if mol is not None:
        request.charges = [(a.Z, a.xyz) for a in mol.atoms]
    return manta.compute_one_electron(basis, basis, request).values


def _energy_components(mol, basis, matrices):
    a, b, w = matrices; d = a+b
    builder = manta.DirectJKBuilder(basis)
    j = builder.build_J(d); ka, kb = builder.build_K_pair(a, b)
    return np.array([mol.nuclear_repulsion(), np.sum(d*(_one(basis, "kinetic")+_one(basis, "potential", mol))),
                     .5*(np.sum(d*j)-np.sum(a*ka)-np.sum(b*kb)), -np.sum(w*_one(basis, "overlap"))])


@pytest.mark.parametrize("pure", [False, True])
def test_arbitrary_density_components_against_five_point_lagrangian_differences(pure):
    mol, basis, matrices = _fixture(pure)
    result = manta.compute_hf_lagrangian_gradient(mol, basis, *matrices)
    actual = [result.nuclear, result.one_electron, result.two_electron, result.pulay]
    np.testing.assert_allclose(result.values, sum(actual), atol=0, rtol=0)
    direction = np.array([[.4, -.3, .2], [-.1, .2, .3], [.2, .1, -.4]])
    for h in [2e-4, 1e-4]:
        energies = [_energy_components(*_moved(mol, basis, direction, factor*h), matrices) for factor in [1, -1, 2, -2]]
        expected = (8*(energies[0]-energies[1])-(energies[2]-energies[3]))/(12*h)
        np.testing.assert_allclose([np.sum(g*direction) for g in actual], expected, atol=4e-9, rtol=2e-8)
    for component in actual:
        np.testing.assert_allclose(component.sum(axis=0), 0., atol=8e-12, rtol=0)
    plan = manta.plan_hf_lagrangian_gradient(mol, basis)
    assert result.receipt.shell_calls == plan.shell_calls
    assert result.receipt.primitive_integrals_evaluated <= plan.primitive_evaluations_upper_bound
    assert result.receipt.modeled_peak_bytes == plan.modeled_peak_bytes


def _scf(mol, basis, unrestricted):
    options = vq.UHFOptions() if unrestricted else vq.RHFOptions()
    options.initial_guess = vq.InitialGuess.HCORE
    options.conv_tol_energy, options.conv_tol_grad, options.max_iter = 1e-12, 1e-10, 150
    options.stability_check = False
    s = _one(basis, "overlap")
    h = _one(basis, "kinetic")+_one(basis, "potential", mol)
    builder = manta.DirectJKBuilder(basis)
    ne = mol.n_electrons()
    if unrestricted:
        na, nb = (ne+mol.multiplicity-1)//2, (ne-mol.multiplicity+1)//2
        result = vq.run_uhf_scf_with_jk(basis, na, nb, s, h, mol.nuclear_repulsion(), builder, options)
        a, b = result.density_alpha, result.density_beta
        ca, cb = result.mo_coeffs_alpha[:, :na], result.mo_coeffs_beta[:, :nb]
        w = (ca*result.mo_energies_alpha[:na])@ca.T+(cb*result.mo_energies_beta[:nb])@cb.T
    else:
        result = vq.run_rhf_scf_with_jk(basis, ne, s, h, mol.nuclear_repulsion(), builder, options)
        a = b = result.density*.5
        c = result.mo_coeffs[:, :ne//2]
        w = 2*(c*result.mo_energies[:ne//2])@c.T
    assert result.converged
    return result, [np.ascontiguousarray(x) for x in [a, b, w]], options


@pytest.mark.parametrize("unrestricted", [False, True])
def test_stationary_hf_gradient_against_independent_libcint_and_relaxed_energy(unrestricted):
    gto = pytest.importorskip("pyscf.gto")
    scf = pytest.importorskip("pyscf.scf")
    positions = [[0., 0., 0.], [.2, -.1, 1.8]]+([[1.3, .4, .7]] if unrestricted else [])
    mol = vq.Molecule([vq.Atom(1, p) for p in positions], 0, 2 if unrestricted else 1)
    basis = vq.BasisSet(mol, "sto-3g")
    result, matrices, options = _scf(mol, basis, unrestricted)
    gradient = manta.compute_hf_lagrangian_gradient(mol, basis, *matrices)
    refmol = gto.M(atom=[("H", p) for p in positions], basis="sto-3g", unit="Bohr", spin=int(unrestricted), verbose=0)
    reference = scf.UHF(refmol) if unrestricted else scf.RHF(refmol)
    reference.conv_tol, reference.conv_tol_grad = 1e-12, 1e-10
    reference.kernel(dm0=np.array(matrices[:2]) if unrestricted else sum(matrices[:2]))
    assert reference.converged
    assert result.energy == pytest.approx(reference.e_tot, abs=3e-10)
    np.testing.assert_allclose(gradient.values, reference.nuc_grad_method().kernel(), atol=3e-8, rtol=3e-8)
    native = vq.run_uhf(mol, basis, options) if unrestricted else vq.run_rhf(mol, basis, options)
    native_gradient = vq.compute_gradient_uhf(mol, basis, native) if unrestricted else vq.compute_gradient(mol, basis, native)
    np.testing.assert_allclose(gradient.values, native_gradient, atol=3e-8, rtol=3e-8)
    direction = np.array([[.1, -.2, .3], [-.3, .2, .1]]+([[.2, -.1, -.2]] if unrestricted else []))
    for h in [2e-4, 1e-4]:
        energies = [_scf(*_moved(mol, basis, direction, sign*h), unrestricted)[0].energy for sign in [1, -1]]
        assert np.sum(gradient.values*direction) == pytest.approx((energies[0]-energies[1])/(2*h), abs=5e-8)
    np.testing.assert_allclose(gradient.values.sum(axis=0), 0., atol=4e-12)
    np.testing.assert_allclose(np.cross(positions, gradient.values).sum(axis=0), 0., atol=3e-10)
    assert np.linalg.norm(gradient.pulay) > 1e-3


@pytest.mark.parametrize("field,plan_field", [("memory_bytes", "modeled_peak_bytes"),
    ("shell_calls", "shell_calls"), ("primitive_quartets", "primitive_evaluations_upper_bound"),
    ("scalar_terms", "scalar_terms"), ("kernel_terms", "kernel_terms")])
def test_gradient_combined_budget_boundaries(field, plan_field):
    mol = vq.Molecule([vq.Atom(1, [0., 0., 0.]), vq.Atom(1, [0., 0., 1.8])], 0, 1)
    basis = vq.BasisSet(mol, "sto-3g"); d = np.eye(basis.nbasis)
    plan = manta.plan_hf_lagrangian_gradient(mol, basis)
    budget = manta.ResourceBudget(); setattr(budget, field, getattr(plan, plan_field))
    manta.compute_hf_lagrangian_gradient(mol, basis, d, d, d, budget)
    setattr(budget, field, getattr(plan, plan_field)-1)
    with pytest.raises(ValueError, match="budget"):
        manta.compute_hf_lagrangian_gradient(mol, basis, d, d, d, budget)


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("kind", ["float32", "complex", "strided", "shape", "nan", "asymmetric"])
def test_gradient_input_matrices_are_strict(index, kind):
    mol, basis, matrices = _fixture(); n = basis.nbasis
    matrices[index] = {"float32": np.eye(n, dtype=np.float32), "complex": np.eye(n, dtype=complex),
                       "strided": np.eye(2*n)[::2, ::2], "shape": np.eye(n+1), "nan": np.eye(n)*np.nan,
                       "asymmetric": np.triu(np.ones((n, n)))}[kind]
    with pytest.raises(ValueError):
        manta.compute_hf_lagrangian_gradient(mol, basis, *matrices)


def test_gradient_geometry_consistency_ownership_and_source_identity():
    mol, basis, matrices = _fixture()
    result = manta.compute_hf_lagrangian_gradient(mol, basis, *matrices)
    before = result.values.copy()
    alternate = manta.compute_hf_lagrangian_gradient(mol, basis, *(np.asfortranarray(m) for m in matrices))
    np.testing.assert_array_equal(result.values, alternate.values)
    assert alternate.receipt.density_identity == result.receipt.density_identity
    matrices[0] *= .7
    alternate = manta.compute_hf_lagrangian_gradient(mol, basis, *matrices)
    assert alternate.receipt.density_identity != result.receipt.density_identity
    assert alternate.receipt.source_identity == result.receipt.source_identity
    np.testing.assert_array_equal(before, result.values)
    moved, _ = _moved(mol, basis, np.ones((3, 3)), .1)
    with pytest.raises(ValueError, match="centers/atom map"):
        manta.plan_hf_lagrangian_gradient(moved, basis)
    matrices[1][:] = np.nan
    budget = manta.ResourceBudget(); budget.memory_bytes = 1
    with pytest.raises(ValueError, match="memory budget"):
        manta.compute_hf_lagrangian_gradient(mol, basis, *matrices, budget)


def test_ghost_center_gradient_and_unaligned_storage():
    mol, basis, matrices = _fixture()
    atoms = [vq.Atom(0 if i == 1 else a.Z, a.xyz) for i, a in enumerate(mol.atoms)]
    mol = vq.Molecule(atoms, 0, 1)
    basis = vq.BasisSet(mol, basis.shells(), "ghost-gradient", True)
    unaligned = []
    for matrix in matrices:
        out = np.ndarray(matrix.shape, dtype=np.float64, buffer=bytearray(matrix.nbytes+1), offset=1)
        out[:] = matrix; unaligned.append(out)
    result = manta.compute_hf_lagrangian_gradient(mol, basis, *unaligned)
    reference = manta.compute_hf_lagrangian_gradient(mol, basis, *matrices)
    np.testing.assert_array_equal(result.values, reference.values)
    np.testing.assert_array_equal(result.nuclear[1], 0.)
    direction = np.zeros((3, 3)); direction[1] = [.2, -.3, .1]
    h = 1e-4
    energies = [_energy_components(*_moved(mol, basis, direction, sign*h), matrices).sum() for sign in [1, -1]]
    assert np.sum(result.values*direction) == pytest.approx((energies[0]-energies[1])/(2*h), abs=3e-8)


def test_coincident_charged_nuclei_are_refused():
    mol, basis, _ = _fixture()
    atoms = [vq.Atom(a.Z, a.xyz) for a in mol.atoms]+[vq.Atom(1, mol.atoms[0].xyz)]
    with pytest.raises(ValueError, match="coincident"):
        manta.plan_hf_lagrangian_gradient(vq.Molecule(atoms, 0, 2), basis)


def test_hf_gradient_citation_route():
    from vibeqc.output.citations import load_default_database
    references = load_default_database().assemble(method="RHF", extra_methods=["manta_hf_gradient"])
    assert {r.key for r in references} >= {"mcmurchie_davidson_1978", "neese_shark_2023", "pople_derivatives_1979"}
