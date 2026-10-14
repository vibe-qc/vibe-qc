"""Experimental chi-owned semilocal quadrature and its density adjoint.

Bloch AOs use exp(+ik.R), D = C occ C†, and tau = |grad psi|²/2.
The returned matrices have no k weight: dE = sum_k w_k Tr(V_k dD_k).
The hard, physical AO-image support is independent of Coulomb translations
and of the Becke partition. Its spatial boundary is not differentiated;
this module supplies density derivatives, not nuclear forces.

The GGA and kinetic-energy-density chain rules follow Neumann, Nobes and
Handy, Mol. Phys. 87, 1-36 (1996). Functional definitions belong to libxc.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from copy import deepcopy
from hashlib import sha256
from itertools import product
import math

import numpy as np


@dataclass(frozen=True)
class ChiXCControls:
    """Explicit physical support; radii must be converged by the caller.

    The candidate cap applies to the sum of enclosing integer boxes over
    AO centers in one point block. Exceeding it refuses the calculation.
    """

    density_image_radius_bohr: float
    partition_radius_bohr: float
    block_size: int = 128
    image_candidate_cap: int = 1_000_000

    def __post_init__(self):
        for name in ("density_image_radius_bohr", "partition_radius_bohr"):
            value = getattr(self, name)
            if isinstance(value, (bool, str, complex)) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"chi XC {name} must be finite and positive")
        for name in ("block_size", "image_candidate_cap"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"chi XC {name} must be a positive integer")


def _real_array(value, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"chi XC {name} must be real")
    result = np.array(raw, dtype=float, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"chi XC {name} must be finite")
    result.flags.writeable = False
    return result


def _array_hash(*arrays):
    digest = sha256()
    for array in arrays:
        data = np.ascontiguousarray(array, dtype="<f8")
        digest.update(str(data.shape).encode("ascii"))
        digest.update(data.tobytes())
    return digest.hexdigest()


def _grid_options(options):
    """Portable values, including enum names, without object repr addresses."""
    result = {}
    for key in dir(options):
        if key.startswith("_"):
            continue
        value = getattr(options, key)
        if callable(value):
            continue
        if isinstance(value, (str, int, float, bool, list, tuple)) or value is None:
            result[key] = value
        elif hasattr(value, "name"):
            result[key] = str(value.name)
    return result


@dataclass(frozen=True)
class ChiExchangeAssembly:
    """Exact exchange ``c_full/r + c_sr*erfc(omega*r)/r``.

    The CAM convention is alpha/r + beta*erf(omega*r)/r (Yanai,
    Tew and Handy, Chem. Phys. Lett. 393, 51-57, 2004). Thus the
    short-distance fraction is c_full+c_sr and the long-distance
    fraction is c_full. A negative c_sr is required by LC hybrids.
    This contract does not admit a kernel on an unimplemented backend.
    """

    c_full: float
    c_sr: float = 0.0
    omega_screen: float = 0.0

    @property
    def schema(self):
        return "vibeqc.chi.exchange-kernels/v1"

    def __post_init__(self):
        for name in ("c_full", "c_sr", "omega_screen"):
            value = getattr(self, name)
            if isinstance(value, (bool, str, complex)) or not np.isfinite(value):
                raise ValueError(f"chi exchange {name} must be finite real")
            object.__setattr__(self, name, float(value))
        if self.omega_screen < 0 or (self.c_sr != 0 and self.omega_screen <= 0):
            raise ValueError("chi screened exchange requires positive omega")
        if self.c_sr == 0 and self.omega_screen != 0:
            raise ValueError("inactive chi screened exchange requires zero omega")
        if self.c_full < 0 or self.c_full + self.c_sr < 0:
            raise NotImplementedError("chi exchange requires nonnegative short- and long-distance fractions")

    @property
    def needs_exchange(self):
        return self.c_full != 0 or self.c_sr != 0

    @property
    def is_screened(self):
        return self.c_sr != 0

    def as_dict(self):
        return dict(schema="vibeqc.chi.exchange-kernels/v1",
                    full_range_coefficient=self.c_full,
                    short_range_coefficient=self.c_sr,
                    long_range_coefficient=0.0,
                    omega_bohr_inv=self.omega_screen,
                    equivalent_full_plus_erf=dict(full=self.c_full + self.c_sr,
                                                  erf=-self.c_sr),
                    full_range_zero_mode="Born-von-Karman Ewald",
                    short_range_zero_mode="finite pi/(omega^2 V_BvK)",
                    spin_convention="RKS: -K(D)/2; UKS: -K(D_spin)",
                    energy_convention="RKS: -Tr(D K(D))/4; UKS: -sum_spin Tr(D_spin K(D_spin))/2",
                    normalization="primitive cell; k weights sum to one")


def exchange_assembly(functional):
    """Use native capabilities, never a functional-name allow list."""
    if not hasattr(functional, "exchange_components"):
        raise RuntimeError("chi XC requires native Functional.exchange_components")
    full, short, ranges = 0.0, 0.0, set()
    for kernel, coefficient, omega in functional.exchange_components:
        for value in (coefficient, omega):
            if isinstance(value, (bool, str, complex)) or not np.isfinite(value):
                raise ValueError("chi exchange components must be finite real")
        if kernel == "coulomb":
            if omega != 0:
                raise ValueError("full-range chi exchange component has nonzero range")
            full += coefficient
        elif kernel == "erfc":
            if omega <= 0:
                raise ValueError("chi erfc exchange requires positive omega")
            if coefficient != 0:
                ranges.add(float(omega))
            short += coefficient
        else:
            raise NotImplementedError(f"chi exchange kernel {kernel!r} is unsupported")
    if len(ranges) > 1:
        raise NotImplementedError("chi hybrid functionals with multiple screening ranges are unsupported")
    return ChiExchangeAssembly(full, short, next(iter(ranges)) if short != 0 else 0.0)


def functional_identity(functional, spin):
    """Resolve supported native components without allocating a grid."""
    from ... import _vibeqc_core as core

    if isinstance(functional, str):
        functional = core.Functional(functional, spin)
    if hasattr(functional, "spin") and int(functional.spin) != spin:
        raise ValueError("chi XC functional spin does not match requested spin")
    for attribute in ("is_external", "is_double_hybrid", "needs_vv10",
                      "needs_laplacian"):
        if bool(getattr(functional, attribute, False)):
            raise NotImplementedError(f"chi semilocal XC does not support {attribute}")
    exchange = exchange_assembly(functional)
    kind = str(functional.kind).rsplit(".", 1)[-1].upper()
    if kind not in ("LDA", "GGA", "MGGA"):
        raise NotImplementedError(f"unsupported chi XC kind {kind}")
    if not hasattr(functional, "components"):
        raise RuntimeError("chi XC requires a native core exposing Functional.components")
    if not hasattr(functional, "component_capabilities"):
        raise RuntimeError("chi XC requires native Functional.component_capabilities")
    for _, component_kind, energy, potential, dim3, laplacian in functional.component_capabilities:
        if (component_kind not in ("exchange", "correlation", "exchange-correlation")
                or not energy or not potential or not dim3 or laplacian):
            raise NotImplementedError("chi XC requires 3D exchange/correlation energy and first derivatives without Laplacian")
    return {"name": str(functional.name), "kind": kind, "spin": int(spin),
            "libxc_version": core.libxc_version(),
            "components": list(functional.components),
            "component_capabilities": list(functional.component_capabilities),
            "exchange_components": list(functional.exchange_components),
            "exchange": exchange.as_dict(),
            "range_parameter_bohr_inv": float(getattr(functional, "rsh_omega", 0.0)),
            "semilocal_convention": "native component weights, including hybrid subtraction"}


class ChiSemilocalXC:
    """Stream a periodic grid and one k point at a time in two AO passes."""

    def __init__(self, system, basis, kpoints_cart, weights, functional,
                 spin, options, controls):
        from ... import _vibeqc_core as core
        from ...periodic_grid import build_periodic_becke_grid

        if not isinstance(controls, ChiXCControls):
            raise TypeError("chi XC requires explicit ChiXCControls")
        if spin not in (1, 2):
            raise ValueError("chi XC spin must be 1 (restricted) or 2 (unrestricted)")
        if isinstance(functional, str):
            functional = core.Functional(functional, spin)
        identity = functional_identity(functional, spin)
        self._functional_identity = deepcopy(identity)
        self.exchange_assembly = exchange_assembly(functional)
        self.kind = identity["kind"]
        if int(system.dim) != 3:
            raise NotImplementedError("chi semilocal XC currently requires 3D periodicity")
        self.lattice = _real_array(system.lattice, "lattice")
        if self.lattice.shape != (3, 3):
            raise ValueError("chi XC lattice must be 3 by 3 with column vectors")
        try:
            self.inverse = np.linalg.inv(self.lattice)
        except np.linalg.LinAlgError as error:
            raise ValueError("chi XC lattice must be nonsingular") from error
        self.kpoints = _real_array(kpoints_cart, "k points")
        self.weights = _real_array(weights, "k weights")
        if self.kpoints.ndim != 2 or self.kpoints.shape[1] != 3 or not len(self.kpoints):
            raise ValueError("chi XC k points must have nonempty shape (nk, 3)")
        if (self.weights.shape != (len(self.kpoints),) or np.any(self.weights < 0)
                or not np.isclose(self.weights.sum(), 1.0, atol=1e-12, rtol=0)):
            raise ValueError("chi XC k weights must be nonnegative and sum to one")
        self.basis, self.functional, self.spin = basis, functional, spin
        self.controls, self.nbasis = controls, int(basis.nbasis)
        self._evaluate_ao = core.evaluate_ao_with_gradient
        groups = {}
        offset = 0
        for shell in basis.shells():
            size = 2 * shell.l + 1 if shell.pure else (shell.l + 1) * (shell.l + 2) // 2
            origin = _real_array(shell.origin, "AO center")
            if origin.shape != (3,):
                raise ValueError("chi XC AO center must have length three")
            groups.setdefault(tuple(origin), []).extend(range(offset, offset + size))
            offset += size
        if offset != self.nbasis or self.nbasis < 1:
            raise ValueError("chi XC shell dimensions disagree with basis size")
        self.centers = [(np.asarray(center), np.asarray(indices)) for center, indices in groups.items()]
        grid_options = getattr(options, "grid", None)
        if grid_options is None:
            grid_options = core.GridOptions()
        grid = build_periodic_becke_grid(
            system, grid_options=grid_options,
            image_radius_bohr=controls.partition_radius_bohr,
        )
        self.points = _real_array(grid.points, "grid points")
        self.grid_weights = _real_array(grid.weights, "grid weights")
        if self.points.ndim != 2 or self.points.shape[1] != 3 or self.grid_weights.shape != (len(self.points),):
            raise ValueError("chi XC grid dimensions disagree")
        self.metadata = {
            "implementation": "chi-semilocal-xc-v1", "experimental": True,
            "functional": identity,
            "density_support": "point-centered physical AO-image spheres; hard cutoff",
            "controls": asdict(controls),
            "grid_point_count": len(self.points),
            "grid_sha256": _array_hash(self.points, self.grid_weights),
            "grid_options": _grid_options(grid_options),
            "kpoints_sha256": _array_hash(self.kpoints, self.weights),
            "executed_image_count": 0, "executed_image_sha256": None,
            "integrated_charge": None,
        }

    def require_functional(self, functional):
        """Refuse attaching this evaluator to a different XC request."""
        if functional_identity(functional, self.spin) != self._functional_identity:
            raise ValueError("chi XC evaluator functional identity does not match request")

    def _image_boxes(self, points):
        radius = self.controls.density_image_radius_bohr
        widths = radius * np.linalg.norm(self.inverse, axis=1)
        boxes, count = [], 0
        for center, indices in self.centers:
            fractional = (points - center) @ self.inverse.T
            # Outward rounding also retains points on the physical boundary.
            lower = np.floor(fractional.min(axis=0) - widths).astype(object)
            upper = np.ceil(fractional.max(axis=0) + widths).astype(object)
            if not np.all(np.isfinite(np.asarray([lower, upper], dtype=float))):
                raise ValueError("chi XC image bounds are not finite")
            bounds = [(int(lo), int(hi)) for lo, hi in zip(lower, upper)]
            count += math.prod(hi - lo + 1 for lo, hi in bounds)
            if count > self.controls.image_candidate_cap:
                raise ValueError("chi XC image candidate cap exceeded; increase cap or reduce block size")
            boxes.append((center, indices, bounds))
        return boxes

    def _bloch(self, points, kpoint, boxes, record=None):
        ao = np.zeros((len(points), self.nbasis), dtype=complex)
        gradient = np.zeros((3, len(points), self.nbasis), dtype=complex)
        radius2 = self.controls.density_image_radius_bohr ** 2
        for center_id, (center, indices, bounds) in enumerate(boxes):
            for integer in product(*(range(lo, hi + 1) for lo, hi in bounds)):
                shift = self.lattice @ np.asarray(integer, dtype=float)
                displacement = points - center - shift
                # A closed sphere includes its boundary despite roundoff from
                # equivalent lattice/atom representatives. This is an ulp
                # allowance, not an AO-magnitude screening threshold.
                distance2 = np.einsum("pi,pi->p", displacement, displacement)
                mask = distance2 <= radius2 * (1 + 32 * np.finfo(float).eps)
                if not np.any(mask):
                    continue
                if record is not None:
                    record[0] += 1
                    record[1].update(f"{center_id}:{integer}:".encode("ascii"))
                    record[1].update(mask.tobytes())
                values = self._evaluate_ao(self.basis, np.ascontiguousarray(points[mask] - shift))
                phase = np.exp(1j * np.dot(kpoint, shift))
                selection = np.ix_(mask, indices)
                ao[selection] += phase * np.asarray(values[0])[:, indices]
                for axis in range(3):
                    gradient[axis][selection] += phase * np.asarray(values[axis + 1])[:, indices]
        return ao, gradient

    def _density(self, density):
        value = np.asarray(density, dtype=complex)
        shape = (len(self.kpoints), self.nbasis, self.nbasis)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"chi XC density must be finite with shape {shape}")
        if not np.allclose(value, value.swapaxes(-1, -2).conj(), atol=1e-12, rtol=1e-12):
            raise ValueError("chi XC density must be Hermitian")
        return value

    def restricted(self, density):
        if self.spin != 1:
            raise ValueError("restricted chi XC requires spin=1")
        energy, potentials = self._evaluate([self._density(density)])
        return energy, potentials[0]

    def unrestricted(self, density_a, density_b):
        if self.spin != 2:
            raise ValueError("unrestricted chi XC requires spin=2")
        energy, potentials = self._evaluate([self._density(density_a), self._density(density_b)])
        return energy, potentials[0], potentials[1]

    def _evaluate(self, densities):
        potentials = [np.zeros_like(density) for density in densities]
        energy, charge = 0.0, np.zeros(len(densities))
        record = [0, sha256()]
        for start in range(0, len(self.points), self.controls.block_size):
            points = self.points[start:start + self.controls.block_size]
            weights = self.grid_weights[start:start + self.controls.block_size]
            boxes = self._image_boxes(points)
            record[1].update(f"block:{start}:{len(points)}:".encode("ascii"))
            rho = np.zeros((len(densities), len(points)))
            grad = np.zeros((len(densities), 3, len(points)))
            tau = np.zeros_like(rho)
            for k, (kpoint, wk) in enumerate(zip(self.kpoints, self.weights)):
                ao, derivatives = self._bloch(points, kpoint, boxes, record if k == 0 else None)
                for s, density in enumerate(densities):
                    ad = ao @ density[k]
                    rho[s] += wk * np.einsum("pi,pi->p", ad, ao.conj()).real
                    if self.kind != "LDA":
                        for axis, derivative in enumerate(derivatives):
                            grad[s, axis] += 2 * wk * np.einsum("pi,pi->p", derivative, ad.conj()).real
                            if self.kind == "MGGA":
                                tau[s] += .5 * wk * np.einsum("pi,pi->p", derivative @ density[k], derivative.conj()).real
            exc, vrho, q, vtau = self._functional(rho, grad, tau)
            energy += float(weights @ exc)
            charge += rho @ weights
            for k, kpoint in enumerate(self.kpoints):
                ao, derivatives = self._bloch(points, kpoint, boxes)
                for s, potential in enumerate(potentials):
                    matrix = ao.conj().T @ ((weights * vrho[s])[:, None] * ao)
                    if self.kind != "LDA":
                        for axis, derivative in enumerate(derivatives):
                            cross = derivative.conj().T @ ((weights * q[s, axis])[:, None] * ao)
                            matrix += cross + cross.conj().T
                            if self.kind == "MGGA":
                                matrix += .5 * derivative.conj().T @ ((weights * vtau[s])[:, None] * derivative)
                    potential[k] += matrix
        if not np.isfinite(energy) or not all(np.all(np.isfinite(v)) for v in potentials):
            raise FloatingPointError("chi XC produced nonfinite energy or potential")
        self.metadata.update(executed_image_count=record[0],
                             executed_image_sha256=record[1].hexdigest(),
                             integrated_charge=float(charge.sum()),
                             electron_count=float(charge.sum()),
                             integrated_spin_charges=charge.tolist(),
                             executed_image_count_definition="center-image contributions summed over grid blocks; one AO pass, independent of k count")
        return energy, potentials

    def _functional(self, rho, grad, tau):
        q, vtau = np.zeros_like(grad), np.zeros_like(rho)
        if len(rho) == 1:
            sigma = np.sum(grad[0] ** 2, axis=0)
            if self.kind == "MGGA":
                exc, vr, vs, vt = self.functional.eval_unpolarised_mgga(rho[0], sigma, tau[0])
                vtau[0] = vt
            else:
                exc, vr, vs = self.functional.eval_unpolarised(rho[0], sigma)
            if self.kind != "LDA":
                q[0] = 2 * np.asarray(vs) * grad[0]
            return np.asarray(exc), np.asarray(vr)[None, :], q, vtau
        aa = np.sum(grad[0] ** 2, axis=0)
        ab = np.sum(grad[0] * grad[1], axis=0)
        bb = np.sum(grad[1] ** 2, axis=0)
        arguments = (rho[0], rho[1], aa, ab, bb)
        if self.kind == "MGGA":
            exc, va, vb, vsaa, vsab, vsbb, vta, vtb = self.functional.eval_polarised_mgga(*arguments, tau[0], tau[1])
            vtau[:] = vta, vtb
        else:
            exc, va, vb, vsaa, vsab, vsbb = self.functional.eval_polarised(*arguments)
        if self.kind != "LDA":
            q[0] = 2 * np.asarray(vsaa) * grad[0] + np.asarray(vsab) * grad[1]
            q[1] = 2 * np.asarray(vsbb) * grad[1] + np.asarray(vsab) * grad[0]
        return np.asarray(exc), np.asarray([va, vb]), q, vtau
