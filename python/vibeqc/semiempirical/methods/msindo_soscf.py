"""NumPy twin of the native stateful SOSCF orbital step.

See cpp/src/soscf.cpp, Neese, Chem. Phys. Lett. 325, 93-98 (2000),
and the existing SOSCF citation route. This changes the search direction,
not the Fock operator, energy, occupations, or convergence criteria.
"""

import numpy as np

from vibeqc.quadratic_scf import expm_skew


class OrbitalOptimizer:
    def __init__(self):
        self.history = []
        self.previous_gradient = None
        self.previous_step = None

    def step(self, F, C, eps, nocc):
        nvir = C.shape[1] - nocc
        if nocc == 0 or nvir == 0:
            return C, np.diag(C.T @ F @ C).copy()
        Fmo = C.T @ F @ C
        Fmo = .5 * (Fmo + Fmo.T)
        gradient = Fmo[nocc:, :nocc].ravel(order="F")
        diagonal = np.maximum(eps[nocc:, None] - eps[None, :nocc], 1e-6).ravel(order="F")
        if self.previous_gradient is not None:
            s = self.previous_step
            y = gradient - self.previous_gradient
            if np.dot(s, y) > 1e-10 * np.linalg.norm(s) * np.linalg.norm(y):
                self.history.append((s.copy(), y.copy()))
                self.history = self.history[-10:]

        def direction():
            q = gradient.copy()
            alpha, rho = [], []
            for s, y in reversed(self.history):
                r = 1. / np.dot(y, s)
                a = r * np.dot(s, q)
                alpha.append(a)
                rho.append(r)
                q -= a * y
            result = q / diagonal
            for (s, y), a, r in zip(self.history, reversed(alpha), reversed(rho)):
                result += s * (a - r * np.dot(y, result))
            return -result

        step = direction()
        if np.dot(gradient, step) >= 0 and self.history:
            self.history.clear()
            step = direction()
            if np.dot(gradient, step) >= 0:
                return C, np.diag(Fmo).copy()
        norm = np.linalg.norm(step)
        if norm > .3:
            step *= .3 / norm
        self.previous_gradient = gradient.copy()
        self.previous_step = step.copy()
        rotation = np.zeros((C.shape[1], C.shape[1]))
        block = step.reshape((nvir, nocc), order="F")
        rotation[nocc:, :nocc] = block
        rotation[:nocc, nocc:] = -block.T
        updated = C @ expm_skew(rotation)
        return updated, np.diag(updated.T @ F @ updated).copy()
