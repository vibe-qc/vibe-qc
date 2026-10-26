"""Internal charge-dependent D4 operands for molecular GFN2 development.

The two-body functional is E(q) = 1/2 W(q).T K W(q), with the derivative
with respect to *positive* atomic partial charges returned alongside it.
This is the energy/potential construction of Bannwarth, Ehlert and Grimme,
JCTC 15, 1652 (2019), doi:10.1021/acs.jctc.8b01176, using the charge scaling
of Caldeweyher et al., JCP 150, 154122 (2019), doi:10.1063/1.5090222.

These are fixed-geometry operands, not a self-consistent calculation. The
builder currently consumes the independently generated native references
and native EEQ coordination numbers. Neither an EEQ charge solve nor a
charge-dependent ATM term belongs in this charge functional. Its native
snapshot supplies both the SCC potential and matched geometry derivatives.
Published reference conventions and the high-level model/output integration
remain separate work; the public GFN2 model does not select this module yet.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vibeqc.dispersion_d4_model import (
    D4_CHARGE_HEIGHT,
    D4_CHARGE_STEEPNESS,
    D4_WEIGHTING_FACTOR,
    D4_ALP_DEFAULT,
    _dzeta_dq,
    _cn_gaussian_weights_derivative,
    _d4_atm_distance_derivatives,
    compute_c8,
    compute_c9,
    compute_d4_c6_pair,
    compute_d4_c6_pair_derivs,
    d4_bj_damping_energy,
    d4_bj_damping_gradient,
    effective_nuclear_charge,
    r4r2_val,
    zeta_charge_scaling,
    zeta_hardness,
)


def _frozen_array(values, dtype=float):
    """Detach operands from caller-owned data and make storage immutable."""
    data = np.ascontiguousarray(values, dtype=dtype)
    # A bytes backing buffer also prevents setflags(write=True).
    return np.frombuffer(data.tobytes(), dtype=data.dtype).reshape(data.shape)


@dataclass(frozen=True)
class D4ChargeFunctional:
    """Immutable fixed-geometry two-body kernel and neutral ATM energy.

    ``kernel`` contains reference C6 coefficients times negative BJ damping;
    same-atom blocks are zero. CN weights and charge scaling are applied
    separately. The 1/2 in the symmetric contraction counts each atom pair
    once. Arrays are detached from reference datasets at construction.
    """

    atom_numbers: np.ndarray
    positions_bohr: np.ndarray
    reference_atoms: np.ndarray
    reference_charges_shifted: np.ndarray
    effective_charges: np.ndarray
    hardness: np.ndarray
    cn_weights: np.ndarray
    kernel: np.ndarray
    neutral_atm_energy: float
    cn_weight_derivatives: np.ndarray
    cn_jacobian: np.ndarray
    radial_kernel: np.ndarray
    neutral_atm_gradient: np.ndarray

    def to_native(self):
        """Copy validated operands into an immutable native SCC snapshot."""
        from vibeqc._vibeqc_core.semiempirical.xtb import _GFN2D4Data

        return _GFN2D4Data(self)

    def energy_and_potential(self, charges):
        """Return E in Ha and dE/dq_A in Ha/e, with geometry held fixed.

        For Mulliken q_A = Zval_A - sum_{mu in A,nu} P_mu,nu S_mu,nu,
        the corresponding Fock contribution is
        -1/2 S_mu,nu (dE/dq_A + dE/dq_B). The minus sign is essential.
        ATM is charge-independent here and contributes no potential.
        """
        weights, derivative, _ = self._weights(charges)
        field = self.kernel @ weights
        energy = 0.5 * float(weights @ field) + self.neutral_atm_energy
        potential = np.bincount(
            self.reference_atoms, weights=derivative * field,
            minlength=len(self.atom_numbers),
        )
        if not np.isfinite(energy) or not np.all(np.isfinite(potential)):
            raise ValueError("D4 charge functional produced nonfinite operands")
        return energy, potential

    def _weights(self, charges):
        q = np.asarray(charges, dtype=float)
        if q.shape != self.atom_numbers.shape or not np.all(np.isfinite(q)):
            raise ValueError("D4 charge functional requires finite atom-sized charges")
        nref = len(self.reference_atoms)
        weights = np.empty(nref)
        derivative = np.empty(nref)
        cn_derivative = np.empty(nref)
        for i, atom in enumerate(self.reference_atoms):
            qmod = q[atom] + self.effective_charges[i]
            qref = self.reference_charges_shifted[i]
            zeta = zeta_charge_scaling(
                D4_CHARGE_HEIGHT, self.hardness[i], qref, qmod,
            )
            weights[i] = self.cn_weights[i] * zeta
            cn_derivative[i] = self.cn_weight_derivatives[i] * zeta
            derivative[i] = self.cn_weights[i] * _dzeta_dq(
                D4_CHARGE_HEIGHT, self.hardness[i], qref, qmod,
            )
        return weights, derivative, cn_derivative

    def energy_potential_gradient(self, charges):
        """Also return the explicit nuclear gradient at fixed partial charges.

        This includes CN, BJ damping and neutral ATM. The Mulliken-overlap
        contribution belongs in the SCC gradient with the charge potential;
        there is no EEQ charge response to add to this functional.
        """
        energy, potential = self.energy_and_potential(charges)
        weights, _, cn_derivative = self._weights(charges)
        sensitivity = np.bincount(
            self.reference_atoms, weights=cn_derivative * (self.kernel @ weights),
            minlength=len(self.atom_numbers),
        )
        gradient = (self.cn_jacobian @ sensitivity).reshape(-1, 3)
        gradient += self.neutral_atm_gradient
        for i, a in enumerate(self.reference_atoms):
            for j, b in enumerate(self.reference_atoms[:i]):
                if a == b:
                    continue
                vector = self.positions_bohr[a]-self.positions_bohr[b]
                value = weights[i]*weights[j]*self.radial_kernel[i, j]
                contribution = value * vector / np.linalg.norm(vector)
                gradient[a] += contribution
                gradient[b] -= contribution
        return energy, potential, gradient


def build_d4_charge_functional(mol, reference_data, damping_parameters):
    """Build internal SCC operands with native reference/CN conventions.

    No reference data are fetched or generated here. Unsupported elements
    and malformed coefficients raise instead of silently dropping dispersion.
    The neutral-charge ATM convention is that of molecular GFN2, not the
    charge-dependent ATM convention of the existing post-SCF EEQ model.
    """
    from vibeqc._vibeqc_core import eeq_coordination_numbers, eeq_response_gradient

    numbers = np.asarray([a.Z for a in mol.atoms], dtype=int)
    positions = np.asarray([a.xyz for a in mol.atoms], dtype=float)
    if not len(numbers) or not np.all(np.isfinite(positions)):
        raise ValueError("D4 charge functional requires a finite nonempty geometry")
    p = damping_parameters
    values = np.asarray([p.s6, p.s8, p.s9, p.a1, p.a2])
    if not np.all(np.isfinite(values)) or np.any(values < 0) or p.a2 <= 0:
        raise ValueError("D4 charge functional requires physical damping parameters")
    cn = eeq_coordination_numbers(mol)
    # Unit CN sensitivities with zero charge sensitivity isolate the exact
    # native CN Jacobian. Built once per geometry, outside SCC iterations.
    cn_jacobian = np.column_stack([
        eeq_response_gradient(mol, row, np.zeros(len(numbers))).reshape(-1)
        for row in np.eye(len(numbers))
    ])
    owners, shifted_q, effective, hardness, weights, offsets = [], [], [], [], [], [0]
    cn_derivatives = []
    for a, z in enumerate(numbers):
        try:
            ref_cn = np.asarray(reference_data.get_cns(int(z)), dtype=float)
            ref_q = np.asarray(reference_data.get_qs(int(z)), dtype=float)
        except KeyError:
            raise ValueError(f"D4 reference data unavailable for Z={z}") from None
        if (ref_cn.ndim != 1 or not len(ref_cn) or ref_cn.shape != ref_q.shape
                or not np.all(np.isfinite(ref_q))):
            raise ValueError("D4 reference charges must match the reference environments")
        z_eff = effective_nuclear_charge(int(z))
        if np.any(ref_q + z_eff <= 0):
            raise ValueError("D4 reference effective charges must be positive")
        w, dw = _cn_gaussian_weights_derivative(
            cn[a], ref_cn, D4_WEIGHTING_FACTOR, reference_data.get_n_gaussian(int(z)),
        )
        weights.extend(w)
        cn_derivatives.extend(dw)
        owners.extend([a] * len(ref_cn))
        shifted_q.extend(ref_q + z_eff)
        effective.extend([z_eff] * len(ref_cn))
        hardness.extend([zeta_hardness(int(z)) * D4_CHARGE_STEEPNESS] * len(ref_cn))
        offsets.append(len(owners))

    kernel = np.zeros((len(owners), len(owners)))
    radial_kernel = np.zeros_like(kernel)
    neutral_c6 = np.zeros((len(numbers), len(numbers)))
    neutral_c6_cn = np.zeros_like(neutral_c6)
    distances = np.zeros_like(neutral_c6)
    radii = np.zeros_like(neutral_c6)
    for a, za in enumerate(numbers):
        for b, zb in enumerate(numbers[:a]):
            distance = float(np.linalg.norm(positions[a] - positions[b]))
            if distance <= 1e-10:
                raise ValueError("D4 charge functional requires distinct atom positions")
            factor = d4_bj_damping_energy(
                distance, 1., compute_c8(1., za, zb), p.s6, p.s8, p.a1, p.a2,
            )
            _, radial_vector, _ = d4_bj_damping_gradient(
                distance, np.array([1., 0., 0.]), 1., compute_c8(1., za, zb),
                p.s6, p.s8, p.a1, p.a2,
            )
            for ia, i in enumerate(range(offsets[a], offsets[a+1])):
                for ib, j in enumerate(range(offsets[b], offsets[b+1])):
                    c6 = float(reference_data.get_c6_ref(int(za), ia, int(zb), ib))
                    if not np.isfinite(c6) or c6 < 0:
                        raise ValueError("D4 reference C6 coefficients must be finite and nonnegative")
                    kernel[i, j] = kernel[j, i] = factor * c6
                    radial_kernel[i, j] = radial_kernel[j, i] = radial_vector[0]*distance*c6
            neutral_c6[a, b] = neutral_c6[b, a] = compute_d4_c6_pair(
                cn[a], 0., int(za), cn[b], 0., int(zb), reference_data,
            )
            _, ca, cb, _, _ = compute_d4_c6_pair_derivs(
                cn[a], 0., int(za), cn[b], 0., int(zb), reference_data,
            )
            neutral_c6_cn[a, b], neutral_c6_cn[b, a] = ca, cb
            distances[a, b] = distances[b, a] = distance
            radii[a, b] = radii[b, a] = p.a1 * np.sqrt(3*r4r2_val(za)*r4r2_val(zb)) + p.a2

    atm = 0.
    atm_gradient = np.zeros_like(positions)
    atm_cn = np.zeros(len(numbers))
    for a in range(len(numbers)):
        for b in range(a):
            for c in range(b):
                if min(neutral_c6[a, b], neutral_c6[a, c], neutral_c6[b, c]) <= 0.:
                    continue
                pairs = ((a, b), (b, c), (a, c))
                triple, derivatives = _d4_atm_distance_derivatives(
                    np.array([distances[i, j]**2 for i, j in pairs]),
                    radii[a, b]*radii[b, c]*radii[a, c],
                    p.s9*compute_c9(neutral_c6[a, b], neutral_c6[a, c], neutral_c6[b, c]),
                    D4_ALP_DEFAULT,
                )
                atm += triple
                for (i, j), derivative in zip(pairs, derivatives):
                    vector = 2.*derivative*(positions[i]-positions[j])
                    atm_gradient[i] += vector
                    atm_gradient[j] -= vector
                    sensitivity = triple/(2.*neutral_c6[i, j])
                    atm_cn[i] += sensitivity*neutral_c6_cn[i, j]
                    atm_cn[j] += sensitivity*neutral_c6_cn[j, i]
    atm_gradient += (cn_jacobian @ atm_cn).reshape(-1, 3)
    return D4ChargeFunctional(
        _frozen_array(numbers, int), _frozen_array(positions),
        _frozen_array(owners, int), _frozen_array(shifted_q),
        _frozen_array(effective), _frozen_array(hardness),
        _frozen_array(weights), _frozen_array(kernel), float(atm),
        _frozen_array(cn_derivatives), _frozen_array(cn_jacobian),
        _frozen_array(radial_kernel), _frozen_array(atm_gradient),
    )
