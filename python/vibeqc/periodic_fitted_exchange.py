"""Kernel-specific fitted exchange for the explicit experimental KS path.

All fitting metrics omit p=0. The exact overlap term restores the finite
erfc zero mode separately from the full-range Born-von-Karman Ewald term.
This is the full-minus-physical-erfc convention for long-range exchange.
The finite auxiliary approximations to different kernels need not add:
the executed CAM approximation is explicitly full plus signed erfc fits.
See Yanai, Tew and Handy, Chem. Phys. Lett. 393, 51-57 (2004), and Sun
et al., J. Chem. Phys. 147, 164119 (2017), for CAM and periodic fitting.
"""
from types import MappingProxyType

import numpy as np


class ChiFittedExchange:
    """Immutable kernel factors and exact zero-mode actions, per primitive cell."""

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("chi fitted exchange identity is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, full, screened, assembly, overlap, weights, volume, madelung):
        self.full, self.screened, self.assembly = full, screened, assembly
        s = np.ascontiguousarray(overlap, dtype=np.complex128)
        w = np.ascontiguousarray(weights, dtype=float)
        if (s.ndim != 3 or s.shape[1] != s.shape[2] or len(s) != len(w)
                or not np.isfinite(s).all() or not np.isfinite(w).all()
                or np.any(w < 0) or abs(float(w.sum()) - 1) > 1e-12
                or not np.allclose(s, s.conj().transpose(0, 2, 1), atol=1e-12, rtol=1e-12)):
            raise ValueError("chi fitted exchange needs finite Hermitian overlaps and normalized weights")
        if not np.isfinite(volume) or volume <= 0 or not np.isfinite(madelung):
            raise ValueError("chi fitted exchange needs positive volume and finite Madelung")
        if assembly.c_sr != 0 and screened is None:
            raise ValueError("missing physical erfc exchange factors")
        if full.kernel != "coulomb" or full.omega_screen != 0:
            raise ValueError("full exchange requires Coulomb factors")
        if screened is not None and (screened.kernel != "erfc"
                or screened.omega_screen != assembly.omega_screen):
            raise ValueError("screened exchange factor kernel or range mismatch")
        for operator in (full, screened):
            if operator is not None:
                if (tuple(operator._weights) != tuple(w)
                        or set(operator) != {(i, j) for i in range(len(w)) for j in range(len(w))}
                        or any(a.shape[-2:] != s.shape[-2:] for a in operator.values())):
                    raise ValueError("exchange factors disagree with density mesh, weights or AO shape")
        self.overlap = np.frombuffer(s.tobytes(), dtype=s.dtype).reshape(s.shape)
        self.weights = np.frombuffer(w.tobytes(), dtype=w.dtype)
        self.volume, self.madelung = float(volume), float(madelung)
        self.factors = MappingProxyType({"coulomb": full, **({"erfc": screened} if screened is not None else {})})
        self._sealed = True

    def as_dict(self):
        return dict(schema="vibeqc.chi.fitted-exchange/v1",
                    assembly=self.assembly.as_dict(),
                    factor_decomposition="independent full plus signed erfc fits",
                    sources={name: value.as_dict() for name, value in self.factors.items()},
                    zero_mode=dict(full=self.madelung,
                        erfc_per_k=(np.pi * self.weights / (self.volume * self.assembly.omega_screen**2)).tolist()
                        if self.assembly.c_sr else [0.0] * len(self.weights)),
                    method_maturity="experimental", quantitative_status="not-qualified")

    def components(self, density):
        # Reuse source shape/Hermiticity validation even for zero coefficients.
        d = self.full._density(density)
        sds = self.overlap @ d @ self.overlap
        result = {}
        if self.assembly.c_full != 0:
            result["full_nonzero"] = self.assembly.c_full * self.full.exchange(d)
            result["full_zero"] = self.assembly.c_full * self.madelung * sds
        if self.assembly.c_sr != 0:
            result["erfc_nonzero"] = self.assembly.c_sr * self.screened.exchange(d)
            constant = self.assembly.c_sr * np.pi * self.weights / (self.volume * self.assembly.omega_screen**2)
            result["erfc_zero"] = constant[:, None, None] * sds
        return result

    def exchange(self, density):
        components = self.components(density)
        return sum(components.values(), np.zeros_like(self.overlap))

    def energies(self, *spin_densities):
        """Coefficient-weighted exchange components for spin-one densities."""
        energies = dict(full_nonzero=0.0, full_zero=0.0, erfc_nonzero=0.0, erfc_zero=0.0)
        for density in spin_densities:
            for name, value in self.components(density).items():
                energies[name] -= .5 * float(np.einsum("k,kij,kji->", self.weights, density, value).real)
        return energies


def build_chi_fitted_exchange(system, basis, auxiliary, kpoints, weights,
                              full, assembly, overlap, madelung, *, source,
                              controls, retained_bytes=0):
    """Build actual erfc M/T factors under the same requested support controls."""
    from .periodic_fitted_support import build_chi_fitted_operator
    screened = None
    if assembly.c_sr != 0:
        retained = int(retained_bytes) + sum(value.nbytes for value in full.values())
        screened = build_chi_fitted_operator(
            system, basis, auxiliary, kpoints, weights, source=source,
            controls=controls, kernel="erfc", omega_screen=assembly.omega_screen,
            _retained_bytes=retained)
    return ChiFittedExchange(full, screened, assembly, overlap, weights,
                             abs(float(np.linalg.det(system.lattice))), madelung)
