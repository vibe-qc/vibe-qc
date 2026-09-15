"""Analytic gradients of the overlap-based penalty terms.

The composite basis-optimisation objective is Ω = E + penalty, where the
penalty is built from the same-center atomic overlap matrix S (the
condition-number term g.ln κ(S) and the linear-dependence hinge on
l_min). This module differentiates the *penalty* part analytically, so a
BDIIS / scipy gradient need not finite-difference it.

Why this matters
----------------
* The penalty is exactly the regime where finite differences are worst:
  near linear dependence κ blows up and dκ/da is large and stiff, so an
  FD step that is fine for the smooth energy is noisy for ln κ.
* The kernel here -- the analytic same-center overlap derivative dS/da --
  is the *same* object the eventual analytic SCF-energy gradient needs
  for its Pulay term -tr(W.dS/da). Building it once, FD-verified, pays
  off twice.

Scope (first increment)
-----------------------
Same-center, single-element atomic overlap, segmented shells
(S/P/D/F/G) with ``scale_factor == 1.0`` -- i.e. exactly what
:func:`vibeqc.basis_optimization.ld_diagnostics.compute_overlap_diagnostics_from_atom`
handles, and what the pob-* targets use. **SP shells raise**
``NotImplementedError`` (their shared exponent couples two angular
blocks); fall back to finite-difference gradients for SP bases. Extreme
eigenvalues are assumed simple (true for the atomic overlaps that arise
in practice); Hellmann-Feynman dl = vᵀ(dS)v is exact there.

Math
----
For two same-center, same-l *normalised* primitive Gaussians the overlap
is (see ld_diagnostics)::

    s(a, b; l) = (2.√(ab) / (a + b))^(l + 3/2)

so, with p = l + 3/2 and u = 2√(ab)/(a+b),

    ds/da = p . u^(p-1) . du/da,
    du/da = √b . (b - a) / (√a . (a + b)^2).

A contracted-shell block ``S0_ij = S_pq c_ip c_jq s(a_ip, a_jq; l)`` is
normalised to ``M_ij = S0_ij / √(S0_ii S0_jj)``; its derivative follows by
the quotient rule (the diagonal ``M_ii = 1`` has zero derivative, a handy
self-check). Eigenvalue derivatives are Hellmann-Feynman; the global
extremes live in one block each, and a parameter only perturbs the block
of its own shell, so off-block contributions vanish.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import numpy as np

from .ld_diagnostics import EPS_LD, _SHELL_L, _normalized_primitive_overlap

if TYPE_CHECKING:
    from .parametrise import BasisParametrisation


def _ds_prim_da(a: float, b: float, l: int) -> float:
    """d/da of the normalised same-l primitive overlap ``s(a, b; l)``."""
    p = l + 1.5
    apb = a + b
    u = 2.0 * math.sqrt(a * b) / apb
    du_da = math.sqrt(b) * (b - a) / (math.sqrt(a) * apb * apb)
    return p * (u ** (p - 1.0)) * du_da


# Block = list of (original_shell_index, exponents, coefficients) for one l.
_Block = list[tuple[int, list[float], list[float]]]


def _atom_blocks(atom) -> dict[int, _Block]:
    """Group an atom's segmented shells by l (SP / scaled shells raise)."""
    blocks: dict[int, _Block] = {}
    for idx, sh in enumerate(atom.shells):
        if sh.scale_factor != 1.0:
            raise NotImplementedError(
                "analytic overlap gradient requires scale_factor == 1.0; "
                f"shell {idx} has {sh.scale_factor!r} -- use finite differences"
            )
        if sh.shell_type == "SP":
            raise NotImplementedError(
                "analytic overlap gradient is not implemented for SP shells "
                "(shared exponent couples two angular blocks); omit grad= to "
                "finite-difference the penalty for SP bases"
            )
        l = _SHELL_L.get(sh.shell_type)
        if l is None:
            raise NotImplementedError(f"unsupported shell_type {sh.shell_type!r}")
        blocks.setdefault(l, []).append(
            (idx, list(sh.exponents), list(sh.coefficients))
        )
    return blocks


def _block_S0(rows: _Block, l: int) -> np.ndarray:
    n = len(rows)
    s0 = np.empty((n, n))
    for i in range(n):
        ei, ci = rows[i][1], rows[i][2]
        for j in range(i, n):
            ej, cj = rows[j][1], rows[j][2]
            v = 0.0
            for p in range(len(ei)):
                for q in range(len(ej)):
                    v += ci[p] * cj[q] * _normalized_primitive_overlap(ei[p], ej[q], l)
            s0[i, j] = v
            s0[j, i] = v
    return s0


def _normalize(s0: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.diag(s0))
    return s0 / np.outer(d, d)


def _dS0_dparam(
    rows: _Block, l: int, k: int, r: int, field: str
) -> np.ndarray:
    """dS0 (unnormalised block) w.r.t. one parameter of block-row ``k``.

    ``field`` is ``"exponent"`` or ``"coeff"``; ``r`` is the primitive
    index within that shell. Only rows/cols touching ``k`` are non-zero;
    the diagonal entry picks up the factor of two automatically via the
    symmetric accumulation.
    """
    n = len(rows)
    ds0 = np.zeros((n, n))
    ek, ck = rows[k][1], rows[k][2]
    akr = ek[r]
    for j in range(n):
        ej, cj = rows[j][1], rows[j][2]
        if field == "exponent":
            val = ck[r] * sum(
                cj[q] * _ds_prim_da(akr, ej[q], l) for q in range(len(ej))
            )
        elif field in ("coeff", "coeff_s"):
            val = sum(
                cj[q] * _normalized_primitive_overlap(akr, ej[q], l)
                for q in range(len(ej))
            )
        else:
            raise NotImplementedError(
                f"analytic overlap gradient for field {field!r} not supported "
                "(non-SP shells expose 'exponent' / 'coeff' only)"
            )
        ds0[k, j] += val
        ds0[j, k] += val
    return ds0


def _dM_from_dS0(s0: np.ndarray, ds0: np.ndarray) -> np.ndarray:
    """Quotient-rule derivative of the normalised block ``M = S0/√(d⊗d)``."""
    diag = np.diag(s0)
    ddiag = np.diag(ds0)
    inv_sqrt = 1.0 / np.sqrt(diag)
    ratio = ddiag / diag
    return np.outer(inv_sqrt, inv_sqrt) * (
        ds0 - 0.5 * s0 * np.add.outer(ratio, ratio)
    )


class _AtomSpectrum:
    """Cached eigendecomposition of an atom's per-l overlap blocks."""

    def __init__(self, atom) -> None:
        self.blocks = _atom_blocks(atom)
        self._cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        lam_min = math.inf
        lam_max = -math.inf
        self.vmin: Optional[np.ndarray] = None
        self.vmax: Optional[np.ndarray] = None
        self.l_min: Optional[int] = None
        self.l_max: Optional[int] = None
        for l, rows in self.blocks.items():
            s0 = _block_S0(rows, l)
            m = _normalize(s0)
            w, vecs = np.linalg.eigh(m)
            self._cache[l] = (s0, m, vecs)
            if w[0] < lam_min:
                lam_min, self.vmin, self.l_min = float(w[0]), vecs[:, 0], l
            if w[-1] > lam_max:
                lam_max, self.vmax, self.l_max = float(w[-1]), vecs[:, -1], l
        self.lambda_min = lam_min
        self.lambda_max = lam_max
        self.condition_number = (
            lam_max / lam_min if lam_min > 0 else math.inf
        )

    def _row_of_shell(self, l: int, shell_idx: int) -> int:
        for row, (orig, _e, _c) in enumerate(self.blocks[l]):
            if orig == shell_idx:
                return row
        raise KeyError(f"shell {shell_idx} not found in l={l} block")

    def _block_of_shell(self, shell_idx: int) -> int:
        for l, rows in self.blocks.items():
            if any(orig == shell_idx for orig, _e, _c in rows):
                return l
        raise KeyError(f"shell {shell_idx} not in any block")

    def extreme_gradients(
        self, shell_idx: int, prim_idx: int, field: str
    ) -> tuple[float, float]:
        """Return (dl_min/dparam, dl_max/dparam) for one physical parameter."""
        l_var = self._block_of_shell(shell_idx)
        s0, _m, _vecs = self._cache[l_var]
        k = self._row_of_shell(l_var, shell_idx)
        ds0 = _dS0_dparam(self.blocks[l_var], l_var, k, prim_idx, field)
        dm = _dM_from_dS0(s0, ds0)
        dlam_min = (
            float(self.vmin @ dm @ self.vmin) if l_var == self.l_min else 0.0
        )
        dlam_max = (
            float(self.vmax @ dm @ self.vmax) if l_var == self.l_max else 0.0
        )
        return dlam_min, dlam_max


def _penalty_grad_for_atom(
    atom,
    specs: list[tuple[int, int, str]],
    *,
    kind: str,
    gamma: float,
    lambda_ld: float,
    epsilon: float,
) -> list[float]:
    """dpenalty/d(physical param) for each (shell_idx, prim_idx, field) spec."""
    spec = _AtomSpectrum(atom)
    out: list[float] = []
    for shell_idx, prim_idx, field in specs:
        dmin, dmax = spec.extreme_gradients(shell_idx, prim_idx, field)
        if kind == "cond":
            # d/dx of g.ln(l_max/l_min) = g(dl_max/l_max - dl_min/l_min).
            g = gamma * (dmax / spec.lambda_max - dmin / spec.lambda_min)
        elif kind == "ld":
            # d/dx of l_ld.max(0, e - l_min)^2  = -2 l_ld (e - l_min) dl_min,
            # active only while l_min < e.
            if spec.lambda_min < epsilon:
                g = -2.0 * lambda_ld * (epsilon - spec.lambda_min) * dmin
            else:
                g = 0.0
        else:  # pragma: no cover - guarded by callers
            raise ValueError(f"unknown penalty kind {kind!r}")
        out.append(g)
    return out


def _penalty_gradient_vector(
    parametrisation: "BasisParametrisation",
    x: np.ndarray,
    *,
    kind: str,
    gamma: float = 1e-3,
    lambda_ld: float = 1e3,
    epsilon: float = EPS_LD,
    gated_symbols: Optional[list[str]] = None,
) -> np.ndarray:
    """Penalty gradient in *optimiser* space for a parametrisation at ``x``.

    Mirrors the per-element summed penalty used by the objective factories:
    a free parameter of element X only feeds X's atomic-overlap penalty, so
    cross-element terms vanish. The per-spec chain rule turns d/d(physical)
    into d/d(optimiser) -- multiply by the physical value for a ``LOG``
    parameter (da/d(ln a) = a), by 1 for ``LINEAR``.
    """
    atoms = parametrisation.unpack(np.asarray(x, dtype=float))
    gated = set(gated_symbols) if gated_symbols is not None else None

    # Group free-spec indices by element so each atom is diagonalised once.
    by_symbol: dict[str, list[tuple[int, int, int, str]]] = {}
    for i, fs in enumerate(parametrisation.free):
        by_symbol.setdefault(fs.symbol, []).append(
            (i, fs.shell_idx, fs.prim_idx, fs.field)
        )

    grad = np.zeros(len(parametrisation.free))
    for symbol, items in by_symbol.items():
        if gated is not None and symbol not in gated:
            continue
        specs = [(sidx, pidx, field) for (_i, sidx, pidx, field) in items]
        phys_grads = _penalty_grad_for_atom(
            atoms[symbol], specs,
            kind=kind, gamma=gamma, lambda_ld=lambda_ld, epsilon=epsilon,
        )
        for (i, sidx, pidx, field), gphys in zip(items, phys_grads):
            fs = parametrisation.free[i]
            phys = fs.transform.from_optim(float(x[i]))
            # Compare by .name, not enum identity: a duplicate-loaded
            # parametrise module (the arch test's shim swap) yields a second
            # Transform class, and `is` would silently miss the LOG branch.
            jac = phys if getattr(fs.transform, "name", "") == "LOG" else 1.0
            grad[i] = gphys * jac
    return grad


def condition_number_penalty_gradient(
    parametrisation: "BasisParametrisation",
    x: np.ndarray,
    *,
    gamma: float = 1e-3,
    gated_symbols: Optional[list[str]] = None,
) -> np.ndarray:
    """Optimiser-space gradient of the g.ln κ(S) penalty at ``x``.

    The analytic counterpart of finite-differencing
    :func:`vibeqc.basis_optimization.ld_diagnostics.cond_penalty_from_atom`
    summed over elements. See module docstring for scope (segmented,
    non-SP shells).
    """
    return _penalty_gradient_vector(
        parametrisation, x, kind="cond", gamma=gamma, gated_symbols=gated_symbols
    )


def ld_penalty_gradient(
    parametrisation: "BasisParametrisation",
    x: np.ndarray,
    *,
    lambda_ld: float = 1e3,
    epsilon: float = EPS_LD,
    gated_symbols: Optional[list[str]] = None,
) -> np.ndarray:
    """Optimiser-space gradient of the l_min linear-dependence hinge at ``x``."""
    return _penalty_gradient_vector(
        parametrisation, x, kind="ld",
        lambda_ld=lambda_ld, epsilon=epsilon, gated_symbols=gated_symbols,
    )


def penalty_gradient(
    parametrisation: "BasisParametrisation",
    x: np.ndarray,
    *,
    use_cond_penalty: bool = True,
    gamma: float = 1e-3,
    use_ld_penalty: bool = False,
    lambda_ld: float = 1e3,
    epsilon: float = EPS_LD,
    gated_symbols: Optional[list[str]] = None,
) -> np.ndarray:
    """Combined optimiser-space gradient of the active penalty terms.

    Mirrors the ``use_cond_penalty`` / ``use_ld_penalty`` flags of the
    objective factories
    (:mod:`vibeqc.basis_optimization.recipes.objective`): sums whichever
    penalties the objective includes, so a BDIIS ``grad=`` for that
    objective is simply ``energy_grad(x) + penalty_gradient(p, x, ...same
    flags...)``. Returns the zero vector if no penalty is active.
    """
    x = np.asarray(x, dtype=float)
    total = np.zeros(len(parametrisation.free))
    if use_cond_penalty:
        total = total + condition_number_penalty_gradient(
            parametrisation, x, gamma=gamma, gated_symbols=gated_symbols
        )
    if use_ld_penalty:
        total = total + ld_penalty_gradient(
            parametrisation, x, lambda_ld=lambda_ld, epsilon=epsilon,
            gated_symbols=gated_symbols,
        )
    return total
