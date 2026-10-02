"""Complete native D4 derivatives of the existing EEQ/reference-data model.

No external D4 package or SCC calculation is required. The scalar evaluator
is the oracle for full Cartesian differences; pair/ATM closed forms and
EEQ response contractions independently isolate the derivative layers.
"""

import numpy as np
import pytest

from vibeqc import Atom, Molecule
from vibeqc import _vibeqc_core as core
from vibeqc.dispersion_d4 import compute_d4, _charge_scaling_options, _resolve_refdata_path
from vibeqc.dispersion_d4_model import (
    _d4_atm_distance_derivatives,
    compute_c9,
    compute_d4_energy_total,
    compute_d4_gradient,
    d4_atm_triple_energy,
    d4_bj_damping_energy,
    d4_bj_damping_gradient,
    d4_reference_weights,
    d4_reference_weights_derivs,
)
from vibeqc.dispersion_d4_reference_data import D4ReferenceDataset


def _molecule(case):
    specs = {
        "methanol": [(6, (0,0,0)), (8, (2.7,.15,-.2)), (1, (-.5,1.8,.2)),
                     (1, (-.7,-.9,1.5)), (1, (-.8,-.8,-1.7)), (1, (3.3,1.6,.2))],
        "co2": [(8, (-2.2,0,0)), (6, (0,0,0)), (8, (2.35,0,0))],
        "helium3": [(2, (0,0,0)), (2, (6.7,.1,0)), (2, (3.1,5.3,.2))],
        "water": [(8, (0,0,0)), (1, (1.4,1.1,.1)), (1, (-1.5,1.0,-.1))],
        "single": [(2, (0,0,0))],
        "empty": [],
    }
    return Molecule([Atom(z, list(xyz)) for z, xyz in specs[case]])


def _displaced(mol, atom, axis, step):
    xyz = np.array([a.xyz for a in mol.atoms])
    xyz[atom,axis] += step
    return Molecule([Atom(a.Z, x) for a, x in zip(mol.atoms, xyz)],
                    mol.charge, mol.multiplicity)


def _finite_difference(mol, energy, step):
    grad = np.zeros((len(mol.atoms), 3))
    for atom in range(len(mol.atoms)):
        for axis in range(3):
            grad[atom,axis] = (energy(_displaced(mol,atom,axis,step))
                               - energy(_displaced(mol,atom,axis,-step))) / (2*step)
    return grad


@pytest.mark.parametrize("radius", [2.0, 6.0, 14.0])
def test_pair_gradient_matches_radial_derivative(radius):
    """Fixed C6/C8 isolates the former factor-of-two geometric error."""
    c6, ratio = 15.0, 20.0
    c8 = c6 * ratio
    pars = (1.0, 2.7, .52, 5.0)
    energy, grad, sensitivity = d4_bj_damping_gradient(
        radius, np.array([radius,0.,0.]), c6, c8, *pars
    )
    assert energy == pytest.approx(d4_bj_damping_energy(radius,c6,c8,*pars), abs=1e-16)
    for h in (1e-4, 5e-5):
        radial = (d4_bj_damping_energy(radius+h,c6,c8,*pars)
                  - d4_bj_damping_energy(radius-h,c6,c8,*pars)) / (2*h)
        assert grad[0] == pytest.approx(radial, abs=1e-12, rel=1e-7)
    assert sensitivity == pytest.approx(energy/c6, abs=1e-16)
    assert np.array_equal(grad[1:], np.zeros(2))


@pytest.mark.parametrize("edges", [(3.,4.,5.), (4.,5.,6.), (4.,8.,4.), (12.,13.,14.)])
def test_atm_geometry_derivative_including_collinear(edges):
    """Differentiate the independently evaluated scalar edge kernel."""
    edges = np.array(edges)
    squared = edges**2
    radii = np.array([4.,5.,6.])
    c9 = 5.0 * compute_c9(10.,20.,30.)
    energy, deriv = _d4_atm_distance_derivatives(squared, np.prod(radii), c9, 16.)
    assert energy == pytest.approx(d4_atm_triple_energy(*edges,c9,*radii), abs=1e-16)
    for i in range(3):
        for h in (1e-4, 5e-5):
            plus, minus = squared.copy(), squared.copy()
            plus[i] += h
            minus[i] -= h
            fd = (d4_atm_triple_energy(*np.sqrt(plus),c9,*radii)
                  - d4_atm_triple_energy(*np.sqrt(minus),c9,*radii)) / (2*h)
            assert deriv[i] == pytest.approx(fd, abs=2e-12, rel=2e-7)


def test_reference_weight_underflow_retains_charge_response():
    """The scalar's nearest-CN fallback still applies charge scaling."""
    args = (1000., .4, [1.,2.], [.1,-.2], 6)
    w, dcn, dq = d4_reference_weights_derivs(*args)
    assert np.allclose(w, d4_reference_weights(*args), rtol=0, atol=1e-15)
    h = 1e-5
    fd = (d4_reference_weights(args[0],args[1]+h,*args[2:])
          - d4_reference_weights(args[0],args[1]-h,*args[2:])) / (2*h)
    assert np.allclose(dq, fd, rtol=0, atol=1e-10)
    assert np.array_equal(dcn, np.zeros(2))


@pytest.mark.parametrize("charge", [0., 1., -.5])
@pytest.mark.parametrize("component", ["cn", "q", "both"])
def test_eeq_adjoint_response(charge, component):
    mol = _molecule("methanol")
    n = len(mol.atoms)
    dcn = np.linspace(-.2,.3,n) if component != "q" else np.zeros(n)
    dq = np.linspace(.4,-.1,n) if component != "cn" else np.zeros(n)
    grad = core.eeq_response_gradient(mol,dcn,dq,total_charge=charge)
    def scalar(m):
        return dcn @ core.eeq_coordination_numbers(m) + dq @ core.eeq_charges(m,total_charge=charge).charges
    for h in (1e-4, 5e-5):
        assert np.max(np.abs(grad-_finite_difference(mol,scalar,h))) < 2e-8
    assert np.max(np.abs(grad.sum(axis=0))) < 1e-13
    constant_charge_response = core.eeq_response_gradient(mol,np.zeros(n),np.ones(n),total_charge=charge)
    assert np.max(np.abs(constant_charge_response)) < 1e-13


def test_eeq_response_through_saturated_cn():
    """The response includes the soft cap, not just the raw pair count."""
    # Twelve distinct H neighbours force CN(C) above its saturation scale.
    directions = np.array([(a,b,c) for a in (-1,1) for b in (-1,1) for c in (0,)]
                          + [(a,b,c) for a in (-1,1) for c in (-1,1) for b in (0,)]
                          + [(a,b,c) for b in (-1,1) for c in (-1,1) for a in (0,)])
    mol = Molecule([Atom(6,[0.,0.,0.])]+[Atom(1,list(1.4*d)) for d in directions])
    n = len(mol.atoms)
    dcn = np.linspace(-.1,.2,n)
    dq = np.linspace(.2,-.2,n)
    grad = core.eeq_response_gradient(mol,dcn,dq)
    scalar = lambda m: dcn @ core.eeq_coordination_numbers(m) + dq @ core.eeq_charges(m).charges
    assert np.max(np.abs(grad-_finite_difference(mol,scalar,5e-5))) < 2e-8


@pytest.mark.parametrize("cutoff", [4.0, 100.0])
def test_eeq_response_preserves_counting_options(cutoff):
    mol = _molecule("methanol")
    options = core.EEQOptions()
    options.cn_exp = 2.5
    options.cn_max = 1.2
    options.cn_cutoff = cutoff
    n = len(mol.atoms)
    dcn, dq = np.linspace(-.1,.2,n), np.linspace(.2,-.2,n)
    grad = core.eeq_response_gradient(mol,dcn,dq,total_charge=.5,options=options)
    def scalar(m):
        return (dcn @ core.eeq_coordination_numbers(m,options=options)
                + dq @ core.eeq_charges(m,total_charge=.5,options=options).charges)
    assert np.max(np.abs(grad-_finite_difference(mol,scalar,5e-5))) < 2e-8


@pytest.mark.parametrize("case,functional,charge", [
    ("methanol","gfn2xtb",0.), ("water","pbe",1.),
    ("water","r2scan-3c",-.5), ("co2","gfn2xtb",0.),
    ("helium3","gfn2xtb",0.), ("single","pbe",1.), ("empty","pbe",0.),
])
def test_native_public_complete_gradient(case, functional, charge):
    mol = _molecule(case)
    scalar = lambda m: compute_d4(m,functional,charge=charge,backend="native").energy
    result = compute_d4(mol,functional,charge=charge,backend="native",with_gradient=True)
    assert result.energy == pytest.approx(scalar(mol), abs=2e-15)
    assert result.gradient.shape == (len(mol.atoms),3)
    for h in (1e-4,5e-5):
        fd = _finite_difference(mol,scalar,h)
        assert np.max(np.abs(result.gradient-fd),initial=0.) < 2e-9
    assert np.max(np.abs(result.gradient.sum(axis=0)),initial=0.) < 1e-13
    if mol.atoms:
        xyz = np.array([a.xyz for a in mol.atoms])
        assert np.max(np.abs(np.cross(xyz,result.gradient).sum(axis=0))) < 1e-13
    data = D4ReferenceDataset.load_json(_resolve_refdata_path(None))
    expected = compute_d4_energy_total(mol,data,functional,total_charge=charge,
                                       **_charge_scaling_options(functional))
    assert result.energy == pytest.approx(expected, abs=2e-15)
    if charge and len(mol.atoms)>1:
        neutral = compute_d4(mol,functional,backend="native").energy
        assert abs(neutral-result.energy) > 1e-7


@pytest.mark.parametrize("atm", [False, True])
def test_low_level_gradient_preserves_atm_and_parameter_options(atm):
    from vibeqc.dispersion_d4_parameters import get_d4_params
    mol = _molecule("methanol")
    data = D4ReferenceDataset.load_json(_resolve_refdata_path(None))
    params = get_d4_params("gfn2xtb")._replace(s9=20., s8=1.7)
    kw = dict(total_charge=.5,atm=atm,alp=14.,ga=2.,gc=1.,_damping_parameters=params)
    energy, grad, dcn, dq = compute_d4_gradient(mol,data,"gfn2xtb",**kw)
    scalar = lambda m: compute_d4_energy_total(m,data,"gfn2xtb",**kw)
    assert energy == pytest.approx(scalar(mol),abs=2e-15)
    assert dcn.shape == dq.shape == (len(mol.atoms),)
    assert np.max(np.abs(grad-_finite_difference(mol,scalar,5e-5))) < 2e-9


def test_eeq_response_rejects_bad_sensitivities():
    mol = _molecule("water")
    with pytest.raises(ValueError,match="sensitivities"):
        core.eeq_response_gradient(mol,np.zeros(2),np.zeros(3))
    with pytest.raises(ValueError,match="sensitivities"):
        core.eeq_response_gradient(mol,np.array([np.nan,0.,0.]),np.zeros(3))


def test_native_gradient_rotation_and_permutation_covariance():
    mol = _molecule("methanol")
    xyz = np.array([a.xyz for a in mol.atoms])
    angle = .47
    rotation = np.array([[np.cos(angle),-np.sin(angle),0.],
                         [np.sin(angle),np.cos(angle),0.], [0.,0.,1.]])
    order = np.array([3,1,5,0,4,2])
    transformed = (xyz @ rotation.T + np.array([1.2,-.7,2.1]))[order]
    moved = Molecule([Atom(mol.atoms[i].Z,x) for i,x in zip(order,transformed)])
    kw = dict(backend="native",charge=.5,with_gradient=True)
    first = compute_d4(mol,"r2scan-3c",**kw)
    second = compute_d4(moved,"r2scan-3c",**kw)
    assert first.energy == pytest.approx(second.energy,abs=2e-15)
    assert np.max(np.abs(second.gradient-(first.gradient @ rotation.T)[order])) < 1e-13
