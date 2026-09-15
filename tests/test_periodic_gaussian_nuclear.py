"""Tiny physical nuclear integrals and explicit finite-source accounting.

No SCF, target-system job, image-tail or whole-HF matching certification.
"""

from __future__ import annotations

import gc
import itertools
import math
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import _vibeqc_core as core
from tests.test_periodic_aopair_fourier_panel import _basis, _primitive_pair, _system
from tests.test_periodic_gaussian_source_context import (
    _exact_caps, _make as _context, _options as _source_options,
)


def _bundle(*, basis=None, lattice=None, mesh=(2, 3, 1), ao_cutoff=1.2,
            atoms=None):
    basis = _basis([(0, (.1, -.2, .15), [.7], [.8], True),
                    (0, (.65, .25, -.1), [.9], [.6], True)]) if basis is None else basis
    system = _system(np.array([[3.7, .2, .1], [0, 4.1, .3], [0, 0, 3.9]])
                     if lattice is None else lattice)
    system.unit_cell = [core.Atom(2, [.3, .1, -.2])] if atoms is None else atoms
    auxiliary = _basis([(0, (.1, .2, .3), [.8], [.6], True)])
    options = _source_options()
    options.ao_pair_image_cutoff_bohr = ao_cutoff
    context = _context(system=system, ao=basis, auxiliary=auxiliary,
                       mesh=core._RegularKMesh(mesh), options=options)
    return SimpleNamespace(ao=basis, auxiliary=auxiliary, system=system, context=context)


def _options(b, *, alpha=.6, rcut=1.6, gcut=2.0):
    o = core._PeriodicGaussianNuclearOptions()
    o.alpha, o.real_cutoff_bohr, o.reciprocal_cutoff_bohr_inverse = alpha, rcut, gcut
    o.structural_absolute_tolerance = o.structural_relative_tolerance = 3e-12
    o.basis_verification_caps = _exact_caps(b.context)
    return o


def _live():
    live = core._PeriodicGaussianNuclearLiveInventory()
    live.replicas_per_node = 1
    live.fixed_backend_margin_bytes_per_replica = 16384
    return live


def _caps():
    c = core._PeriodicGaussianNuclearCaps()
    c.maximum_owned_numeric_bytes = 2**24
    c.maximum_per_replica_inventoried_bytes = 2**25
    c.maximum_node_inventoried_bytes = 2**27
    c.maximum_atom_count = 8
    c.maximum_pair_count = 256
    c.maximum_ao_image_candidates = c.maximum_nuclear_image_candidates = 5000000
    c.maximum_reciprocal_candidates = 1000000
    c.maximum_ewald_pair_candidates = 5000000
    c.maximum_work_units = 1000000000000
    return c


def _call(b, k=0, begin=0, count=None, *, options=None, live=None, caps=None,
          plan=False, ewald=False):
    options = _options(b) if options is None else options
    live, caps = _live() if live is None else live, _caps() if caps is None else caps
    if ewald:
        f = (core._plan_periodic_gaussian_nuclear_ewald if plan
             else core._build_periodic_gaussian_nuclear_ewald)
        return f(b.context, b.system, options, live, caps)
    f = (core._plan_periodic_gaussian_nuclear_panel if plan
         else core._build_periodic_gaussian_nuclear_panel)
    return f(b.context, b.ao, b.auxiliary, b.system, k, begin,
             b.ao.nbasis**2-begin if count is None else count, options, live, caps)


def _decimal_boys(n, t):
    """Independent alternating entire-function series, 240-digit arithmetic.

    This is NOT the production positive top-order/downward recurrence.
    At T<=200 the cancellation costs fewer than 90 decimal digits.
    """
    with localcontext() as ctx:
        ctx.prec = 240
        t = t if isinstance(t, Decimal) else Decimal.from_float(float(t))
        power = Decimal(1)
        total = power / (2*n+1)
        for k in range(1, 2000):
            power *= -t / k
            term = power / (2*n+2*k+1)
            total += term
            if abs(term) < Decimal("1e-200"):
                return +total
        raise AssertionError("independent Decimal Boys series failed")


def _boys0(t):
    return 1.0 if t == 0 else .5*math.sqrt(math.pi/t)*math.erf(math.sqrt(t))


def _sr_s(alpha, beta, a, b, c, omega):
    gamma = alpha+beta
    p = (alpha*a+beta*b)/gamma
    t = gamma*np.dot(p-c, p-c)
    rho = omega**2/(omega**2+gamma)
    return (2*math.pi/gamma*math.exp(-alpha*beta/gamma*np.dot(a-b, a-b))
            * (_boys0(t)-math.sqrt(rho)*_boys0(rho*t)))


def _oracle(b, k, options, image_bound=2, nuclear_bound=2, reciprocal_bound=3):
    """Direct independent s-Gaussian sums over oversized integer cubes."""
    shells = b.ao.shells()
    assert all(s.l == 0 and len(s.exponents) == 1 for s in shells)
    a = np.asarray(b.system.lattice)
    reciprocal = 2*np.pi*np.linalg.inv(a).T
    volume = abs(np.linalg.det(a))
    kv = np.array(b.context.k_record(k).cartesian)
    nuclear = [(atom.Z, np.asarray(atom.xyz)) for atom in b.system.unit_cell]
    images = list(itertools.product(range(-image_bound, image_bound+1), repeat=3))
    nimages = list(itertools.product(range(-nuclear_bound, nuclear_bound+1), repeat=3))
    vectors = [reciprocal@label for label in itertools.product(
        range(-reciprocal_bound, reciprocal_bound+1), repeat=3) if any(label)]
    vectors = [v for v in vectors if v@v <= options.reciprocal_cutoff_bohr_inverse**2]
    out = np.zeros((4, len(shells), len(shells)), complex)
    one = {(0, 0, 0): 1.0}
    for mu, sa in enumerate(shells):
        for nu, sb in enumerate(shells):
            alpha, beta = sa.exponents[0], sb.exponents[0]
            ca, cb = sa.coefficients[0], sb.coefficients[0]
            ac = np.asarray(sa.origin)
            overlap, long = 0j, 0j
            for label in images:
                translation = a@label
                bc = np.asarray(sb.origin)+translation
                if np.dot(ac-bc, ac-bc) > b.context.options.ao_pair_image_cutoff_bohr**2:
                    continue
                phase = np.exp(1j*kv@translation)*ca*cb
                p = (alpha*ac+beta*bc)/(alpha+beta)
                overlap += phase*_primitive_pair(alpha, beta, ac, bc, one, one, np.zeros(3))
                for z, c in nuclear:
                    for nlabel in nimages:
                        translated = c+a@nlabel
                        if np.dot(p-translated, p-translated) <= options.real_cutoff_bohr**2:
                            out[0, mu, nu] -= z*phase*_sr_s(alpha, beta, ac, bc, translated, options.alpha)
                for g in vectors:
                    sf = sum(z*np.exp(-1j*g@c) for z, c in nuclear)
                    rho = _primitive_pair(alpha, beta, ac, bc, one, one, -g)
                    long += (-4*np.pi/volume)*math.exp(-(g@g)/(4*options.alpha**2))/(g@g)*sf*phase*rho
            out[1, mu, nu] = long
            out[2, mu, nu] = np.pi*sum(z for z, _ in nuclear)/(volume*options.alpha**2)*overlap
    out[3] = out[:3].sum(axis=0)
    return out


def _ewald_oracle(b, o, bound=5):
    a = np.asarray(b.system.lattice)
    reciprocal = 2*np.pi*np.linalg.inv(a).T
    volume = abs(np.linalg.det(a))
    atoms = [(atom.Z, np.asarray(atom.xyz)) for atom in b.system.unit_cell]
    labels = list(itertools.product(range(-bound, bound+1), repeat=3))
    real = 0.0
    for ia, (za, ra) in enumerate(atoms):
        for ib, (zb, rb) in enumerate(atoms):
            for n in labels:
                if ia == ib and not any(n):
                    continue
                distance = np.linalg.norm(ra-rb+a@n)
                if distance <= o.real_cutoff_bohr:
                    real += .5*za*zb*math.erfc(o.alpha*distance)/distance
    long = 0.0
    for n in labels:
        if not any(n):
            continue
        g = reciprocal@n
        squared = g@g
        if squared <= o.reciprocal_cutoff_bohr_inverse**2:
            sf = sum(z*np.exp(1j*g@r) for z, r in atoms)
            long += 2*np.pi/volume*abs(sf)**2*math.exp(-squared/(4*o.alpha**2))/squared
    self = -o.alpha/math.sqrt(np.pi)*sum(z*z for z, _ in atoms)
    background = -np.pi*sum(z for z, _ in atoms)**2/(2*volume*o.alpha**2)
    return real+long+self+background


def _stable_single_s_oracle(b, o, bound=5):
    """One home-AO s pair: independent stable erfc-convolution formula.

    For a coincident Gaussian pair with exponent gamma, the short-range
    attraction at distance r is proportional to
      [erfc(alpha*sqrt(gamma/(gamma+alpha**2))*r)-erfc(sqrt(gamma)*r)]/r.
    In particular its tail does NOT decay as erfc(alpha*r)/r. This oracle
    avoids the cancellation in a difference of two Boys F0 evaluations.
    """
    shell, = b.ao.shells()
    assert shell.l == 0 and len(shell.exponents) == 1
    lattice = np.asarray(b.system.lattice)
    reciprocal = 2*np.pi*np.linalg.inv(lattice).T
    volume = abs(np.linalg.det(lattice))
    gamma = 2*shell.exponents[0]
    p = np.asarray(shell.origin)
    overlap = shell.coefficients[0]**2*(np.pi/gamma)**1.5
    effective = o.alpha*math.sqrt(gamma/(gamma+o.alpha**2))
    nuclei = [(a.Z, np.asarray(a.xyz)) for a in b.system.unit_cell]
    short, long = 0., 0j
    for label in itertools.product(range(-bound, bound+1), repeat=3):
        for z, c in nuclei:
            distance = np.linalg.norm(p-c-lattice@label)
            if distance <= o.real_cutoff_bohr:
                quotient = (2/math.sqrt(np.pi)*(math.sqrt(gamma)-effective) if distance == 0
                            else (math.erfc(effective*distance)-math.erfc(math.sqrt(gamma)*distance))/distance)
                short -= z*overlap*quotient
        if any(label):
            g = reciprocal@label
            squared = g@g
            if squared <= o.reciprocal_cutoff_bohr_inverse**2:
                structure = sum(z*np.exp(-1j*g@c) for z, c in nuclei)
                long += (-4*np.pi/volume)*math.exp(-squared/(4*o.alpha**2))/squared*structure*overlap*math.exp(-squared/(4*gamma))*np.exp(1j*g@p)
    background = np.pi*sum(z for z, _ in nuclei)/(volume*o.alpha**2)*overlap
    return np.array([short, long, background, short+long+background])


@pytest.mark.parametrize("n", range(13))
def test_boys_table_all_orders_grid_edges_and_large_T_against_decimal(n):
    t = np.array([0., np.nextafter(0., 1.), .009999999, .01, .010000001,
                  49.999, 50., 50.001, 65.99, 66., 99.9999, 100.,
                  np.nextafter(100., np.inf), 140., 200.])
    actual = core._periodic_gaussian_nuclear_boys_diagnostic(n, t, 2**21, 2_000_000_000)
    expected = [float(_decimal_boys(n, x)) for x in t]
    np.testing.assert_allclose(actual, expected, rtol=5e-15, atol=1e-30)
    if n == 12:
        asymptotic = math.prod(range(1, 24, 2))/(2*50.001)**12*math.sqrt(np.pi/(4*50.001))
        assert abs(asymptotic/float(_decimal_boys(12, 50.001))-1) > 1e-12


@pytest.mark.parametrize("mesh,k", [((1, 1, 1), 0), ((2, 3, 1), 4), ((2, 2, 2), 7)])
def test_actual_s_gaussian_components_sign_volume_background_and_nuclear_scalar(mesh, k):
    b = _bundle(mesh=mesh)
    o = _options(b)
    result = _call(b, k, options=o)
    expected = _oracle(b, k, o)
    np.testing.assert_allclose(result.values_copy().reshape(expected.shape), expected, rtol=2e-12, atol=3e-13)
    scalar = _call(b, options=o, ewald=True)
    assert scalar.energy == pytest.approx(_ewald_oracle(b, o), rel=2e-13, abs=2e-13)
    assert scalar.nuclei_policy_identity_sha256 == result.nuclei_policy_identity_sha256
    assert not result.infinite_image_tail_certified
    assert not result.whole_hf_hamiltonian_certified
    assert not result.propagated_roundoff_error_certified
    assert not scalar.cross_rank_bitwise_replay_certified


@pytest.mark.parametrize("alpha", [.03, .6, 3.0])
def test_erfc_single_primitive_subtraction_against_high_precision_integral(alpha):
    basis = _basis([(0, (0., 0., 0.), [.7], [.8], True)])
    b = _bundle(basis=basis, lattice=np.eye(3)*50, mesh=(1, 1, 1), ao_cutoff=.5,
                atoms=[core.Atom(2, [.7, -.2, .15])])
    result = _call(b, options=_options(b, alpha=alpha, rcut=2.0, gcut=.05))
    with localcontext() as ctx:
        ctx.prec = 240
        exponent, coefficient = Decimal.from_float(.7), Decimal.from_float(.8)
        gamma, omega = 2*exponent, Decimal.from_float(alpha)
        radius2 = sum(Decimal.from_float(x)**2 for x in [.7, -.2, .15])
        rho = omega**2/(omega**2+gamma)
        integral = (_decimal_boys(0, gamma*radius2)-rho.sqrt()*_decimal_boys(0, rho*gamma*radius2))
        expected = float(-2*coefficient**2*Decimal.from_float(2*np.pi)/gamma*integral)
    assert result.element(0, 0).real == pytest.approx(expected, rel=2e-13, abs=2e-15)
    d = result.diagnostics
    assert d.maximum_erfc_seed_term_magnitude > 0
    assert 0 < d.minimum_nonzero_erfc_seed_difference_ratio <= 1


@pytest.mark.parametrize("angular", range(6))
def test_original_normalized_shell_convention_and_erfc_against_native_libint(angular):
    basis = _basis([(angular, (.1, -.2, .15), [.7], [.8], True)])
    b = _bundle(basis=basis, lattice=np.eye(3)*40, mesh=(1, 1, 1), ao_cutoff=.3,
                atoms=[core.Atom(2, [.3, .1, -.2])])
    o = _options(b, rcut=1., gcut=.05)
    reference = -2*core._cosx_nuclear_pair_libint(b.ao, 0, 0, [.3, .1, -.2], omega=o.alpha)
    # Selected entries only: no full high-angular-momentum panel required.
    n = b.ao.nbasis
    caps = _caps()
    if angular >= 3:
        caps.maximum_work_units = 10**15
    for index in sorted({0, n//2, (n//2)*n+n//2, n*n-1}):
        actual = _call(b, begin=index, count=1, options=o, caps=caps).element(0, 0)
        assert actual == pytest.approx(reference.ravel()[index], rel=2e-11, abs=3e-13)


def test_L6_pure_shell_erfc_against_independent_radial_closed_form():
    # Native libint one-electron Engine is compiled only through L=5. This
    # L=6 witness uses a radial erfc integral and spherical orthogonality.
    angular, coefficient, exponent, omega = 6, .8, .7, .6
    basis = _basis([(angular, (0., 0., 0.), [exponent], [coefficient], True)])
    b = _bundle(basis=basis, lattice=np.eye(3)*40, mesh=(1, 1, 1), ao_cutoff=.3,
                atoms=[core.Atom(2, [0., 0., 0.])])
    o = _options(b, alpha=omega, rcut=1., gcut=.05)
    caps = _caps()
    caps.maximum_work_units = 10**15
    with localcontext() as ctx:
        ctx.prec = 100
        gamma, w = 2*Decimal.from_float(exponent), Decimal.from_float(omega)
        x = gamma/(gamma+w*w)
        series = sum(Decimal(math.comb(2*k, k))*x**k/(4**k) for k in range(angular+1))
        radial = Decimal(math.factorial(angular))/(2*gamma**(angular+1))*(1-w/(gamma+w*w).sqrt()*series)
        expected = float(-2*Decimal.from_float(coefficient)**2
                         * Decimal.from_float(4*np.pi)/(2*angular+1)*radial)
    n = 2*angular+1
    assert _call(b, begin=6*n+6, count=1, options=o, caps=caps).element(0, 0) == pytest.approx(expected, rel=2e-11, abs=2e-13)
    assert abs(_call(b, begin=6*n+5, count=1, options=o, caps=caps).element(0, 0)) < 2e-13


@pytest.mark.parametrize("mesh", [(3, 1, 1), (2, 1, 1)])
def test_multik_image_phase_reversal_opposite_k_and_partition(mesh):
    basis = _basis([(0, (.1, -.2, .15), [.7], [.8], True),
                    (0, (1.8, .25, -.1), [.9], [.6], True)])
    b = _bundle(basis=basis, lattice=np.diag([2.4, 4.1, 3.9]), mesh=mesh, ao_cutoff=2.6)
    o = _options(b)
    whole = _call(b, 1, options=o)
    expected = _oracle(b, 1, o, image_bound=2, nuclear_bound=2)
    np.testing.assert_allclose(whole.values_copy().reshape(expected.shape), expected, rtol=3e-12, atol=4e-13)
    if mesh[0] == 3:
        assert abs(expected[3, 0, 1].imag) > 1e-4
    else:
        assert np.max(abs(expected.imag)) < 1e-12
    blocks = [_call(b, 1, begin=i, count=1, options=o) for i in range(4)]
    np.testing.assert_array_equal(np.hstack([x.values_copy() for x in blocks]), whole.values_copy())
    assert all(x.nuclei_policy_identity_sha256 == whole.nuclei_policy_identity_sha256 for x in blocks)
    assert all(x.panel_source_identity_sha256 != whole.panel_source_identity_sha256 for x in blocks)
    assert _call(b, 0, options=o).nuclei_policy_identity_sha256 == whole.nuclei_policy_identity_sha256


def test_primitive_product_centered_nuclear_cutoff_is_closed_not_tolerance_padded():
    basis = _basis([(0, (0., 0., 0.), [.7], [.8], True)])
    b = _bundle(basis=basis, lattice=np.eye(3)*40, mesh=(1, 1, 1), ao_cutoff=.3,
                atoms=[core.Atom(2, [1., 0., 0.])])
    values = [_call(b, options=_options(b, rcut=r, gcut=.05)).element(0, 0)
              for r in [np.nextafter(1., 0.), 1., np.nextafter(1., np.inf)]]
    assert values[0] == 0
    assert values[1].real < 0
    assert values[1] == values[2]


def test_reciprocal_cutoff_boundary_uses_original_Ewald_closed_sphere():
    basis = _basis([(0, (0., 0., 0.), [.7], [.8], True)])
    b = _bundle(basis=basis, lattice=np.eye(3)*8, mesh=(1, 1, 1), ao_cutoff=.3,
                atoms=[core.Atom(2, [0., 0., 0.])])
    radius = 2*np.pi/8
    outputs = [_call(b, options=_options(b, rcut=.5, gcut=cut))
               for cut in [np.nextafter(radius, 0.), radius, np.nextafter(radius, np.inf)]]
    assert outputs[0].element(1, 0) == 0
    assert outputs[1].diagnostics.retained_reciprocal_vectors == 18  # six vectors, three audits
    assert outputs[1].element(1, 0).real < 0
    assert outputs[1].element(1, 0) == outputs[2].element(1, 0)


def test_alpha_and_finite_cut_convergence_are_measured_not_certified():
    basis = _basis([(0, (.1, -.2, .15), [.7], [.8], True)])
    b = _bundle(basis=basis, lattice=np.eye(3)*4., mesh=(1, 1, 1), ao_cutoff=.5)
    energies, potentials = [], []
    for alpha in [.5, .8]:
        o = _options(b, alpha=alpha, rcut=12., gcut=8.)
        energies.append(_call(b, options=o, ewald=True).energy)
        result = _call(b, options=o)
        potentials.append(result.element(3, 0))
        np.testing.assert_allclose(result.values_copy()[:, 0], _stable_single_s_oracle(b, o),
                                   rtol=0, atol=3e-13)
        larger = _options(b, alpha=alpha, rcut=14., gcut=8.)
        plateau = _call(b, options=larger)
        np.testing.assert_allclose(plateau.values_copy()[:, 0], _stable_single_s_oracle(b, larger),
                                   rtol=0, atol=3e-13)
        assert result.element(3, 0) == pytest.approx(plateau.element(3, 0), abs=3e-13)
    assert energies[0] == pytest.approx(energies[1], abs=3e-11)
    assert potentials[0] == pytest.approx(potentials[1], abs=3e-11)
    # Preserve the failed R=10 witness: the diffuse convolved Gaussian tail
    # is measurable even though the point-nuclear Ewald energy is converged.
    old_options = _options(b, alpha=.5, rcut=10., gcut=8.)
    old = _call(b, options=old_options)
    np.testing.assert_allclose(old.values_copy()[:, 0], _stable_single_s_oracle(b, old_options),
                               rtol=0, atol=3e-13)
    assert 8e-11 < abs(old.element(3, 0)-potentials[0]) < 1.1e-10
    coarse = _call(b, options=_options(b, rcut=1., gcut=.5)).element(3, 0)
    assert abs(coarse-potentials[0]) > 1e-4


def test_far_unwrapped_nuclei_use_same_native_Ewald_representative_in_every_component():
    lattice = np.eye(3)*10.
    raw = np.array([.17+2**35*10., .2, -.1])
    inverse = np.linalg.inv(lattice)
    fractional = inverse@raw
    representative = lattice@(fractional-np.rint(fractional))
    # Do not compare to the unrecoverable original decimal .17: the physical
    # source is the represented binary64 input and the named native map.
    assert abs(representative[0]-.17) > 1e-6
    basis = _basis([(0, (.1, -.2, .15), [.7], [.8], True)])
    atoms = [core.Atom(2, raw.tolist()), core.Atom(2, [1.4, .2, .1])]
    b = _bundle(basis=basis, lattice=lattice, mesh=(1, 1, 1), ao_cutoff=.5, atoms=atoms)
    wrapped = _bundle(basis=basis, lattice=lattice, mesh=(1, 1, 1), ao_cutoff=.5,
                      atoms=[core.Atom(2, representative.tolist()), atoms[1]])
    o, ow = _options(b, rcut=2., gcut=1.4), _options(wrapped, rcut=2., gcut=1.4)
    actual, expected = _call(b, options=o), _call(wrapped, options=ow)
    np.testing.assert_allclose(actual.values_copy(), expected.values_copy(), rtol=0, atol=3e-13)
    oracle = _stable_single_s_oracle(wrapped, ow, bound=3)
    np.testing.assert_allclose(actual.values_copy()[:, 0], oracle, rtol=0, atol=3e-13)
    # A raw-coordinate phase really is wrong at the requested tolerance.
    wrong_long = _stable_single_s_oracle(b, o, bound=3)[1]
    assert abs(wrong_long-oracle[1]) > 1e-8
    scalar, scalar_wrapped = _call(b, options=o, ewald=True), _call(wrapped, options=ow, ewald=True)
    assert scalar.energy == pytest.approx(scalar_wrapped.energy, abs=3e-13)
    assert scalar.energy == pytest.approx(_ewald_oracle(wrapped, ow, bound=3), abs=3e-13)
    assert actual.nuclei_policy_identity_sha256 == scalar.nuclei_policy_identity_sha256
    assert actual.nuclei_policy_identity_sha256 != expected.nuclei_policy_identity_sha256
    assert "native-Ewald-centered-representatives" in core._PERIODIC_GAUSSIAN_NUCLEAR_POLICY
    assert "FLT_EVAL_METHOD=0" in core._PERIODIC_GAUSSIAN_NUCLEAR_FLOATING_POINT_POLICY


def test_actual_nuclei_charge_multiplicity_and_cut_controls_are_sealed():
    b = _bundle()
    first = _call(b, count=1)
    b.system.charge = 1
    assert _call(b, count=1).nuclei_policy_identity_sha256 != first.nuclei_policy_identity_sha256
    b.system.charge = 0
    b.system.multiplicity = 3
    assert _call(b, count=1).nuclei_policy_identity_sha256 != first.nuclei_policy_identity_sha256
    b.system.multiplicity = 1
    b.system.unit_cell = [core.Atom(3, [.3, .1, -.2])]
    assert _call(b, count=1).nuclei_policy_identity_sha256 != first.nuclei_policy_identity_sha256
    b.system.unit_cell = [core.Atom(2, [.3, .1, -.2])]
    o = _options(b)
    o.structural_absolute_tolerance *= 2
    qualified = _call(b, count=1, options=o)
    assert qualified.nuclei_policy_identity_sha256 == first.nuclei_policy_identity_sha256
    assert qualified.panel_source_identity_sha256 == first.panel_source_identity_sha256
    assert qualified.payload_identity_sha256 != first.payload_identity_sha256
    o.alpha *= 1.01
    assert _call(b, count=1, options=o).nuclei_policy_identity_sha256 != first.nuclei_policy_identity_sha256


@pytest.mark.parametrize("ewald", [False, True])
def test_exact_known_memory_live_inventory_and_each_budget_boundary(ewald):
    b = _bundle()
    p = _call(b, count=1, ewald=ewald, plan=True)
    assert p.borrowed_system_active_numeric_bytes == 28*len(b.system.unit_cell)
    if ewald:
        assert p.owned_numeric_peak_bytes == 56*p.atom_count+24*p.reciprocal_box_candidates
    else:
        assert p.output_numeric_bytes == 64
        assert p.owned_numeric_peak_bytes == sum(getattr(p, f) for f in [
            "output_numeric_bytes", "boys_table_numeric_bytes", "os_workspace_numeric_bytes",
            "selected_shell_numeric_bytes_upper_bound", "primitive_and_shell_output_numeric_bytes",
            "fourier_numeric_workspace_bytes"])
    for cap, field in [("maximum_owned_numeric_bytes", "owned_numeric_peak_bytes"),
                       ("maximum_per_replica_inventoried_bytes", "per_replica_inventoried_bytes"),
                       ("maximum_node_inventoried_bytes", "node_inventoried_bytes"),
                       ("maximum_work_units", "work_units_upper_bound")]:
        c = _caps()
        setattr(c, cap, getattr(p, field))
        _call(b, count=1, ewald=ewald, caps=c, plan=True)
        setattr(c, cap, getattr(p, field)-1)
        with pytest.raises((ValueError, RuntimeError), match="cap exceeded"):
            _call(b, count=1, ewald=ewald, caps=c, plan=True)
    live = _live()
    live.other_retained_bytes_per_replica = 1234
    live.other_transient_bytes_per_replica = 567
    live.replicas_per_node = 3
    live.external_node_bytes = 789
    expanded = _call(b, count=1, ewald=ewald, live=live, plan=True)
    assert expanded.per_replica_inventoried_bytes == p.per_replica_inventoried_bytes+1801
    assert expanded.node_inventoried_bytes == 3*expanded.per_replica_inventoried_bytes+789


@pytest.mark.parametrize("field", ["maximum_atom_count", "maximum_pair_count",
    "maximum_ao_image_candidates", "maximum_nuclear_image_candidates",
    "maximum_reciprocal_candidates", "maximum_work_units"])
def test_incremental_count_caps_before_numerical_allocations(field):
    b = _bundle()
    c = _caps()
    setattr(c, field, 1 if field != "maximum_atom_count" else 0)
    with pytest.raises((ValueError, RuntimeError), match="cap|positive"):
        _call(b, caps=c, plan=True)


@pytest.mark.parametrize("field", ["alpha", "real_cutoff_bohr", "reciprocal_cutoff_bohr_inverse"])
@pytest.mark.parametrize("value", [0., -1., math.nan, math.inf, 1e200, 1e-200])
def test_nonfinite_zero_underflow_and_overflow_controls_fail_closed(field, value):
    b = _bundle()
    o = _options(b)
    setattr(o, field, value)
    with pytest.raises((ValueError, RuntimeError), match="finite|positive|underflow|overflow"):
        _call(b, options=o, plan=True)


def test_ewald_rejects_periodically_coincident_nonzero_nuclei_and_integer_overflow():
    b = _bundle(lattice=np.eye(3)*4., atoms=[core.Atom(2, [0., 0., 0.]), core.Atom(2, [4., 0., 0.])])
    with pytest.raises(ValueError, match="coincident"):
        _call(b, ewald=True, plan=True)
    b = _bundle()
    o = _options(b, gcut=1e20)
    with pytest.raises((ValueError, OverflowError, RuntimeError), match="integer|overflow|count"):
        _call(b, options=o, ewald=True, plan=True)


def test_actual_basis_and_original_lattice_revalidation_and_context_lifetime():
    b = _bundle()
    result = _call(b, count=1)
    old = b.context
    b.ao = _basis([(0, (.1, -.2, .15), [.71], [.8], True),
                   (0, (.65, .25, -.1), [.9], [.6], True)])
    with pytest.raises((ValueError, RuntimeError), match="content|identity|match"):
        _call(b, plan=True)
    del old, b
    gc.collect()
    assert result.context is not None
    assert math.isfinite(result.element(3, 0).real)
    with pytest.raises(IndexError):
        result.element(4, 0)
    b = _bundle()
    b.system.lattice = np.eye(3)*9.
    with pytest.raises(ValueError, match="cell.*match"):
        _call(b, plan=True)


def test_boys_diagnostic_shape_finite_and_exact_budget_gates():
    values = np.array([0., .5, 100.])
    required = 8*10001*19+8*len(values)
    work = 256*10001*512+1024*len(values)
    core._periodic_gaussian_nuclear_boys_diagnostic(12, values, required, work)
    for cap, effort in [(required-1, work), (required, work-1)]:
        with pytest.raises((ValueError, RuntimeError), match="cap exceeded"):
            core._periodic_gaussian_nuclear_boys_diagnostic(12, values, cap, effort)
    for bad in [np.array([math.nan]), np.array([-1.]), np.array([math.inf])]:
        with pytest.raises(ValueError, match="finite|nonnegative"):
            core._periodic_gaussian_nuclear_boys_diagnostic(0, bad, 2**21, 2_000_000_000)
    with pytest.raises(ValueError, match="tiny|bounded"):
        core._periodic_gaussian_nuclear_boys_diagnostic(13, values, 2**21, 2_000_000_000)
