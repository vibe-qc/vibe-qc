"""Constrained-energy checks for the padded RHF orbital response."""

from types import SimpleNamespace

import numpy as np
import pytest

import vibeqc.bipole_gradient as gradient


@pytest.mark.parametrize("weights", [[0.5, 0.5], [0.2, 0.8]])
@pytest.mark.parametrize("complex_response", [False, True])
def test_padded_rhf_response_differentiates_constrained_energy(
    monkeypatch, weights, complex_response
):
    """Differentiate an energy along a coupled, non-self-adjoint SCF map.

    The oracle solves the forward constraint at displaced geometries and
    contracts the resulting normalized AO densities with an energy operator.
    Complex rotations and unequal k weights distinguish its adjoint from a
    forward solve, a symmetrized solve, or an unweighted transpose solve.
    """
    # Coordinates are Re/Im of the virtual component of each occupied MO.
    # Even a diagonal Fock gives opposite signs in the imaginary B_ov block.
    H = np.array([[2.0, 0.7, 0.0, 0.0],
                  [-0.2, 1.4, 0.0, 0.0],
                  [0.0, 0.0, -1.7, 0.3],
                  [0.0, 0.0, -0.1, -2.3]])
    rhs = np.array([0.3, -0.2], dtype=complex)
    motion = np.array([[0.2, -0.3, 0.1], [-0.4, 0.1, 0.2],
                       [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    if complex_response:
        H[:2, 2:] = [[0.4, -0.2], [0.1, 0.3]]
        H[2:, :2] = [[-0.3, 0.2], [0.2, 0.1]]
        rhs += [0.15j, -0.25j]
        motion[2:] = [[0.1, 0.3, -0.2], [0.2, -0.1, 0.4]]
    deltas = [np.array([[0.0, b], [b.conjugate(), 0.0]]) for b in rhs]
    system = gradient.PeriodicSystem(
        3, 10.0 * np.eye(3), [gradient.Atom(2, [0.0, 0.0, 0.0])]
    )
    basis = SimpleNamespace(name="model", nbasis=2)
    kmesh = SimpleNamespace(kpoints=[np.zeros(3), np.array([0.1, 0, 0])],
                            weights=weights)
    opts = SimpleNamespace(pair_complete_1e=False)
    monkeypatch.setattr(gradient, "compute_overlap_lattice", lambda *args: None)
    monkeypatch.setattr(gradient, "BasisSet", lambda *args: basis)

    def b0(displaced, _basis, coeffs, *args, **kwargs):
        t = np.array([C[1, 0] for C in coeffs])
        x = np.concatenate((t.real, t.imag))
        b = H @ x + motion @ np.asarray(displaced.unit_cell[0].xyz)
        return [np.array([[v]]) for v in b[:2] + 1j * b[2:]]

    monkeypatch.setattr(gradient, "_build_multi_k_bipole_b0_closed", b0)
    analytic = gradient._multi_k_orbital_relaxation_closed_diag(
        system, basis, [np.eye(2)] * 2, [np.array([-1.0, 1.0])] * 2,
        1, kmesh, opts, 0.4, energy_fock_delta_k=deltas,
        sr_image_extent_bohr=20.0, sr_density_cells=[]
    )

    def energy(position):
        x = np.linalg.solve(H, -motion @ position)
        t = x[:2] + 1j * x[2:]
        value = 0.0
        for weight, amplitude, delta in zip(weights, t, deltas):
            occupied = np.array([1.0, amplitude])
            occupied /= np.linalg.norm(occupied)
            density = 2.0 * np.outer(occupied, occupied.conjugate())
            value += weight * np.trace(density @ delta).real
        return value

    step = 1e-5
    fd = np.array([(energy(step * e) - energy(-step * e)) / (2 * step)
                   for e in np.eye(3)])
    np.testing.assert_allclose(analytic[0], fd, atol=1e-9, rtol=0)


def test_padded_rhf_response_refuses_inconsistent_singular_constraint(monkeypatch):
    """A least-squares result must not silently drop an unsatisfied RHS."""
    system = gradient.PeriodicSystem(
        3, 10.0 * np.eye(3), [gradient.Atom(2, [0.0, 0.0, 0.0])]
    )
    monkeypatch.setattr(gradient, "compute_overlap_lattice", lambda *args: None)
    monkeypatch.setattr(gradient, "_build_multi_k_bipole_b0_closed",
                        lambda *args, **kwargs: [np.zeros((1, 1), complex)])
    with pytest.raises(NotImplementedError, match="orbital response"):
        gradient._multi_k_orbital_relaxation_closed_diag(
            system, SimpleNamespace(name="model"), [np.eye(2)],
            [np.array([-1.0, 1.0])], 1,
            SimpleNamespace(kpoints=[np.zeros(3)], weights=[1.0]),
            SimpleNamespace(pair_complete_1e=False), 0.4,
            energy_fock_delta_k=[np.array([[0.0, 0.1], [0.1, 0.0]])],
            sr_image_extent_bohr=20.0, sr_density_cells=[]
        )
