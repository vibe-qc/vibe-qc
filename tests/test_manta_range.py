"""Full, erf and erfc MD kernels with independent high-precision references."""
from decimal import Decimal, localcontext
from functools import lru_cache
from itertools import product
from math import pi

import numpy as np
import pytest

import vibeqc as vq
from vibeqc import manta
from vibeqc._vibeqc_core import ShellInfo


def _request(kernel, omega, centers=4):
    request = manta.KernelRequest()
    request.kernel, request.omega, request.centers = kernel, omega, centers
    return request


def _bases(angular, pure=True, contracted=False):
    positions = [[0., .1, -.2], [.2, -.3, 1.1], [-.4, .5, .7], [.7, .1, -.2]]
    mol = vq.Molecule([vq.Atom(2, p) for p in positions], 0, 1)
    bases = []
    for i, l in enumerate(angular):
        shell = ShellInfo()
        shell.atom_index, shell.l, shell.pure, shell.origin = i, l, pure, positions[i]
        shell.exponents = [.7+.1*i, 1.6+.1*i] if contracted else [.7+.1*i]
        shell.coefficients = [.8, -.18] if contracted else [1.]
        bases.append(vq.BasisSet(mol, [shell], "manta-range", False))
    merged = vq.BasisSet(mol, [b.shells()[0] for b in bases], "manta-range", True)
    return bases, merged


def _tile(bases, merged, kernel, omega):
    request = _request(kernel, omega, len(bases))
    if len(bases) == 4:
        return manta.DirectJKBuilder(merged, request).evaluate(tuple(range(4)))
    return manta.FittingTiles(bases, request=request).evaluate((0,)*len(bases))


@pytest.mark.parametrize("maximum_n", [0, 4, 24, 32])
@pytest.mark.parametrize("kernel", ["erf", "erfc"])
def test_range_boys_against_300_digit_reference(maximum_n, kernel):
    with localcontext() as context:
        context.prec = 300

        @lru_cache(None)
        def boys(n, t):
            term = total = Decimal(1)
            for k in range(1, 4000):
                term *= 2*t/Decimal(2*n+2*k+1)
                total += term
                if term < total*Decimal("1e-295"):
                    break
            else:
                raise AssertionError("Independent high-precision reference did not converge")
            return (-t).exp()*total/Decimal(2*n+1)

        for t in [0., 1e-8, 1., 40., 100., 500.]:
            for omega in [1e-6, .1, 1., 10., 1e6, 1e8]:
                actual = manta.range_boys(maximum_n, t, .7, kernel, omega)
                kappa = Decimal(omega)**2/(Decimal(omega)**2+Decimal(.7))
                for n in {0, maximum_n}:
                    lr = kappa.sqrt()**(2*n+1)*boys(n, kappa*Decimal(t))
                    expected = float(lr if kernel == "erf" else boys(n, Decimal(t))-lr)
                    assert actual[n] == pytest.approx(expected, rel=2e-12, abs=0.)


@pytest.mark.parametrize("angular", [(2, 3), (3, 2, 1), (2, 1, 2, 1)])
@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("omega", [0., 1e-6, .3, 3., 1e6])
def test_contracted_full_equals_erf_plus_erfc_and_range_limits(angular, pure, omega):
    bases, merged = _bases(angular, pure, contracted=True)
    full = _tile(bases, merged, "coulomb", 0.)
    lr, sr = (_tile(bases, merged, kernel, omega) for kernel in ["erf", "erfc"])
    np.testing.assert_allclose(lr["values"]+sr["values"], full["values"], atol=2e-11, rtol=2e-11)
    assert lr["kernel"] == "erf" and sr["kernel"] == "erfc"
    assert lr["omega"] == sr["omega"] == omega
    assert len({v["source_identity"] for v in [full, lr, sr]}) == 3
    if omega == 0:
        np.testing.assert_array_equal(lr["values"], np.zeros_like(lr["values"]))
        np.testing.assert_array_equal(sr["values"], full["values"])
    if omega == 1e6:
        np.testing.assert_allclose(lr["values"], full["values"], atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize("kernel", ["erf", "erfc"])
@pytest.mark.parametrize("pure", [False, True])
def test_contracted_range_tiles_and_jk_against_matched_libint(kernel, pure):
    """Matched libint witness through the generated ERI limit, signed contractions included."""
    limit = min(manta.capability().maximum_l, 4)
    angular = (limit, 1, max(limit-1, 0), 2)
    bases, merged = _bases(angular, pure, contracted=True)
    rng = np.random.default_rng(6306)
    d = rng.normal(size=(merged.nbasis, merged.nbasis)); d += d.T
    for omega in [.1, .7, 2.]:
        request = _request(kernel, omega)
        md, li = manta.DirectJKBuilder(merged, request), manta.ReferenceJKBuilder(merged, request)
        assert md.source_identity != li.source_identity
        for quartet in product(range(merged.nshells), repeat=4):
            np.testing.assert_allclose(md.evaluate(quartet)["values"], li.evaluate(quartet)["values"], atol=1e-11, rtol=5e-12)
        actual, reference = md.build(d), li.build(d)
        np.testing.assert_allclose(actual.J, reference.J, atol=2e-10, rtol=1e-11)
        np.testing.assert_allclose(actual.K, reference.K, atol=2e-10, rtol=1e-11)
        assert (actual.receipt.kernel, actual.receipt.omega) == (reference.receipt.kernel, reference.receipt.omega) == (kernel, omega)
        assert actual.receipt.provider == "manta" and reference.receipt.provider == "libint"


def _libcint(bases, kernel, omega, pure):
    gto = pytest.importorskip("pyscf.gto")
    shells = [b.shells()[0] for b in bases]
    molecule = gto.M(atom=[(f"He{i}", s.origin) for i, s in enumerate(shells)],
                     basis={f"He{i}": [[s.l, [s.exponents[0], 1.]]] for i, s in enumerate(shells)},
                     unit="Bohr", cart=not pure, verbose=0)
    arity = len(shells)
    name = {2: "int2c2e", 3: "int3c2e", 4: "int2e"}[arity]+("_sph" if pure else "_cart")
    indices = (1, 2, 0) if arity == 3 else tuple(range(arity))
    with molecule.with_range_coulomb(omega if kernel == "erf" else -omega):
        reference = molecule.intor_by_shell(name, indices)
    if arity == 3:
        reference = reference.transpose(2, 0, 1)
    for axis, shell in enumerate(shells):
        if pure and shell.l == 1:
            reference = np.take(reference, [1, 2, 0], axis=axis)
        elif not pure and shell.l >= 2:
            reference *= np.sqrt((2*shell.l+1)/(4*pi))
    return reference


@pytest.mark.parametrize("angular", [(7, 7), (7, 6, 6), (6, 1, 5, 2), (6, 6, 6, 6)])
@pytest.mark.parametrize("pure", [False, True])
@pytest.mark.parametrize("kernel", ["erf", "erfc"])
def test_high_angular_range_tiles_against_optional_libcint(angular, pure, kernel):
    bases, merged = _bases(angular, pure)
    for omega in [.1, .7, 2.]:
        reference = _libcint(bases, kernel, omega, pure)
        actual = _tile(bases, merged, kernel, omega)["values"]
        np.testing.assert_allclose(actual, reference, atol=4e-11, rtol=3e-10)


def test_arbitrary_density_range_jk_and_native_erf_entry():
    _, basis = _bases((0, 1), contracted=True)
    # Ordered full tensor is a small test oracle; production builds stream tiles.
    n, omega = basis.nbasis, .4
    rng = np.random.default_rng(7412)
    density = rng.normal(size=(n, n)); density += density.T
    matrices = []
    for kernel, range_parameter in [("coulomb", 0.), ("erf", omega), ("erfc", omega)]:
        builder = manta.DirectJKBuilder(basis, _request(kernel, range_parameter))
        eri = np.zeros((n,)*4)
        slices = [slice(0, 1), slice(1, n)]
        for indices in np.ndindex((2,)*4):
            eri[tuple(slices[i] for i in indices)] = builder.evaluate(indices)["values"]
        result = builder.build(density)
        np.testing.assert_allclose(result.J, np.einsum("abcd,cd->ab", eri, density), atol=2e-12)
        np.testing.assert_allclose(result.K, np.einsum("abcd,bd->ac", eri, density), atol=2e-12)
        assert result.receipt.kernel == kernel and result.receipt.omega == range_parameter
        matrices.append(result)
    np.testing.assert_allclose(matrices[0].J, matrices[1].J+matrices[2].J, atol=2e-12)
    np.testing.assert_allclose(matrices[0].K, matrices[1].K+matrices[2].K, atol=2e-12)
    np.testing.assert_allclose(manta.DirectJKBuilder(basis).build_K_erf(density, omega), matrices[1].K, atol=2e-12)
    for invalid in [density.astype(np.complex128), density.astype(np.float32), np.eye(2*n)[::2, ::2]]:
        with pytest.raises(ValueError):
            manta.DirectJKBuilder(basis).build_K_erf(invalid, omega)


def test_binary64_range_identity_and_request_snapshot():
    axes, basis = _bases((0, 1))
    for arity in [2, 4]:
        request = _request("erf", .3, arity)
        make = (lambda r: manta.FittingTiles(axes, request=r)) if arity == 2 else (lambda r: manta.DirectJKBuilder(basis, r))
        first = make(request)
        before = first.source_identity
        request.omega = np.nextafter(.3, 1.)
        assert make(request).source_identity != before
        assert first.source_identity == before
        assert make(_request("erf", 0., arity)).source_identity == make(_request("erf", -0., arity)).source_identity


@pytest.mark.parametrize("kernel,omega", [("coulomb", .3), ("erf", -1.), ("erfc", np.nan), ("erf", np.inf), ("unknown", 0.)])
def test_unadmitted_range_requests(kernel, omega):
    axes, basis = _bases((0, 0))
    request = _request(kernel, omega)
    assert not manta.md_capability(request).supported
    with pytest.raises(ValueError):
        manta.DirectJKBuilder(basis, request)
    request.centers = 2
    with pytest.raises(ValueError):
        manta.FittingTiles(axes, request=request)


@pytest.mark.parametrize("arity", [2, 3, 4])
def test_range_quadrature_work_and_workspace_caps(arity):
    axes, basis = _bases((0,)*arity)
    request = _request("erfc", .4, arity)
    if arity == 4:
        plan = manta.md_plan(basis, request)
        make = lambda budget: manta.DirectJKBuilder(basis, request, budget)
    else:
        plan = manta.plan_fitting(axes, request=request)
        make = lambda budget: manta.FittingTiles(axes, budget, request)
    for field, extent in [("memory_bytes", plan.modeled_peak_bytes), ("kernel_terms", plan.kernel_terms)]:
        budget = manta.ResourceBudget()
        setattr(budget, field, extent)
        make(budget)
        setattr(budget, field, extent-1)
        with pytest.raises(ValueError, match="budget"):
            make(budget)


@pytest.mark.parametrize("field,value", [
    ("centers", 2), ("scalar", "complex"), ("derivative", 1),
    ("images", [[1, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]),
])
def test_range_selection_does_not_expand_other_capabilities(field, value):
    _, basis = _bases((0,))
    request = _request("erf", .3)
    setattr(request, field, value)
    # Other center counts and first derivatives belong to the fitting and
    # derivative planners; the four-center builder still refuses them.
    assert manta.md_capability(request).supported == (field in ["centers", "derivative"])
    with pytest.raises(ValueError):
        manta.DirectJKBuilder(basis, request)


def test_native_erf_entry_rejects_invalid_density_and_range():
    _, basis = _bases((0, 1))
    builder = manta.DirectJKBuilder(basis)
    density = np.eye(basis.nbasis)
    for omega in [-1., np.nan, np.inf]:
        with pytest.raises(ValueError):
            builder.build_K_erf(density, omega)
    bad = density.copy(); bad[0, 0] = np.nan
    asymmetric = density.copy(); asymmetric[0, 1] = 1.
    for invalid in [bad, asymmetric, np.eye(1)]:
        with pytest.raises(ValueError):
            builder.build_K_erf(invalid, .3)


@pytest.mark.parametrize("n,t,rho,kernel,omega", [
    (-1, 0., 1., "erf", .3), (33, 0., 1., "erfc", .3),
    (0, -1., 1., "erf", .3), (0, np.nan, 1., "erfc", .3),
    (0, 0., 0., "erfc", .3), (0, 0., np.inf, "erf", .3),
    (0, 0., 1., "coulomb", .3),
])
def test_radial_diagnostic_refuses_invalid_parameters(n, t, rho, kernel, omega):
    with pytest.raises(ValueError):
        manta.range_boys(n, t, rho, kernel, omega)


@pytest.mark.parametrize("unrestricted", [False, True])
def test_native_range_separated_ks_host_matches_libint(unrestricted):
    from vibeqc import _vibeqc_core as core

    positions = [[0., 0., 0.], [0., 0., 1.4]]
    if unrestricted:
        positions.append([0., 0., 2.9])
    molecule = vq.Molecule([vq.Atom(1, p) for p in positions], 0, 2 if unrestricted else 1)
    basis = vq.BasisSet(molecule, "sto-3g")
    overlap = vq.compute_overlap(basis)
    hcore = vq.compute_kinetic(basis)+vq.compute_nuclear(basis, molecule)
    enuc = sum(1/np.linalg.norm(np.array(a)-b) for i, a in enumerate(positions) for b in positions[i+1:])
    options = vq.UKSOptions() if unrestricted else vq.RKSOptions()
    options.functional = "cam-b3lyp"
    options.initial_guess = vq.InitialGuess.HCORE
    options.max_iter = 100
    options.conv_tol_energy, options.conv_tol_grad = 1e-11, 1e-9
    grid_options = vq.GridOptions()
    grid_options.n_radial = 20
    grid_options.angular = "lebedev"
    grid_options.lebedev_order = 11
    grid = core.build_grid(molecule, grid_options)
    results = []
    for builder in [core.make_direct_jk_builder(basis, 0.), manta.DirectJKBuilder(basis)]:
        if unrestricted:
            result = core.run_uks_scf_with_jk(basis, 2, 1, overlap, hcore, enuc, builder, grid, options)
        else:
            result = core.run_rks_scf_with_jk(basis, 2, overlap, hcore, enuc, builder, grid, options)
        assert result.converged
        results.append(result)
    assert results[0].energy == pytest.approx(results[1].energy, abs=1e-10)
    fields = ["density_alpha", "density_beta"] if unrestricted else ["density"]
    for field in fields:
        np.testing.assert_allclose(getattr(results[0], field), getattr(results[1], field), atol=1e-9)


def _libint_short_range(bases, omega):
    """Libint erfc_coulomb two-/three-center tiles through the Gamma lattice sum.

    A 100 bohr vacuum cell with a 30 bohr pair cutoff keeps only the zero
    translation; every image would carry exp(-omega^2 * 1e4) or less.
    """
    shells = [b.shells()[0] for b in bases]
    for i, shell in enumerate(shells):
        shell.atom_index = i
    atoms = [vq.Atom(2, s.origin) for s in shells]
    mol = vq.Molecule(atoms, 0, 1)
    system = vq.PeriodicSystem(3, np.eye(3)*100., atoms)
    opts = vq.LatticeSumOptions(); opts.cutoff_bohr = 30.
    sizes = [2*s.l+1 if s.pure else (s.l+1)*(s.l+2)//2 for s in shells]
    if len(bases) == 2:
        aux = vq.BasisSet(mol, shells, "libint-sr-aux", True)
        full = np.asarray(vq.compute_2c_eri(aux))
        short = np.asarray(vq.compute_2c_eri_lattice_sr(aux, system, opts, omega))
        block = (slice(0, sizes[0]), slice(sizes[0], sizes[0]+sizes[1]))
    else:
        aux = vq.BasisSet(mol, shells[:1], "libint-sr-aux", True)
        orbital = vq.BasisSet(mol, shells[1:], "libint-sr-orbital", True)
        full = np.asarray(vq.compute_3c_eri(orbital, aux))
        short = np.asarray(vq.compute_3c_eri_lattice_sr(orbital, aux, system, opts, omega))
        block = (slice(0, sizes[0]), slice(0, sizes[1]), slice(sizes[1], sizes[1]+sizes[2]))
    return {"coulomb": full[block], "erfc": short[block], "erf": (full-short)[block]}


@pytest.mark.parametrize("angular", [(3, 2), (3, 3, 1), (2, 3, 3)])
@pytest.mark.parametrize("pure", [False, True])
def test_contracted_range_fitting_tiles_against_matched_libint(angular, pure):
    """Libint's erfc core evaluator witnesses the MD two-/three-center range tiles."""
    bases, merged = _bases(angular, pure, contracted=True)
    for omega in [.1, .7, 2.]:
        reference = _libint_short_range(bases, omega)
        for kernel in ["coulomb", "erf", "erfc"]:
            actual = _tile(bases, merged, kernel, 0. if kernel == "coulomb" else omega)
            assert actual["kernel"] == kernel
            np.testing.assert_allclose(actual["values"], reference[kernel], atol=1e-11, rtol=5e-12)
