"""Analytic penalty-gradient tests for basis optimisation.

Every analytic derivative in ``basis_optimization.gradients`` is checked
against a central finite difference of the corresponding scalar/matrix.
Pure-Python (guarded vibeqc shim, like test_bdiis_driver.py) so it runs
without a built C++ extension. Run with:

    .venv/bin/python -m pytest tests/basisset_dev/test_basis_opt_gradients.py -v --noconftest
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
try:
    import vibeqc  # noqa: F401
except Exception:
    PKG_PARENT = REPO_ROOT / "python"
    if str(PKG_PARENT) not in sys.path:
        sys.path.insert(0, str(PKG_PARENT))
    for mod_name in list(sys.modules):
        # Keep vibeqc._vibeqc_core* registered even here: the single-phase-init
        # C extension never re-runs PyInit on re-import, so deleting those
        # entries would strip the pybind11 def_submodule registrations
        # (...semiempirical.{nddo,xtb,indo}) for the rest of the pytest process.
        # Reachable when ``import vibeqc`` fails *after* the extension itself
        # loaded.
        if mod_name == "vibeqc._vibeqc_core" or mod_name.startswith(
            "vibeqc._vibeqc_core."
        ):
            continue
        if mod_name == "vibeqc" or mod_name.startswith("vibeqc."):
            del sys.modules[mod_name]
    _shim = types.ModuleType("vibeqc")
    _shim.__path__ = [str(PKG_PARENT / "vibeqc")]
    sys.modules["vibeqc"] = _shim

from vibeqc.basis_crystal import parse_crystal_atom_basis_file  # noqa: E402
from vibeqc.basis_optimization import (  # noqa: E402
    BasisParametrisation,
    FreeSpec,
    Transform,
    condition_number_penalty_gradient,
    ld_penalty_gradient,
    optimize_bdiis,
    penalty_gradient,
)
from vibeqc.basis_optimization.gradients import (  # noqa: E402
    _atom_blocks,
    _block_S0,
    _dM_from_dS0,
    _ds_prim_da,
    _dS0_dparam,
    _normalize,
)
from vibeqc.basis_optimization.ld_diagnostics import (  # noqa: E402
    _normalized_primitive_overlap,
    cond_penalty_from_atom,
    compute_overlap_diagnostics_from_atom,
    ld_penalty,
)

SOURCES = REPO_ROOT / "python" / "vibeqc" / "basis_library" / "sources" / "pob-TZVP"


def _pob(symbol_file: str):
    return parse_crystal_atom_basis_file(SOURCES / symbol_file)


def _central_fd(f, x, i, h=1e-6):
    xp = np.array(x, dtype=float)
    xm = np.array(x, dtype=float)
    xp[i] += h
    xm[i] -= h
    return (f(xp) - f(xm)) / (2.0 * h)


# --------------------------------------------------------------------------
# Kernel: primitive overlap + normalised block derivatives
# --------------------------------------------------------------------------


def test_ds_prim_da_matches_fd():
    a, b, l = 0.8, 0.3, 0
    ana = _ds_prim_da(a, b, l)
    h = 1e-7
    fd = (
        _normalized_primitive_overlap(a + h, b, l)
        - _normalized_primitive_overlap(a - h, b, l)
    ) / (2 * h)
    assert ana == pytest.approx(fd, rel=1e-6)


def test_dM_normalised_diagonal_has_zero_derivative():
    # Three s-primitives treated as three 1-primitive shells of l=0.
    rows = [(0, [1.2], [1.0]), (1, [0.5], [1.0]), (2, [0.18], [1.0])]
    s0 = _block_S0(rows, 0)
    ds0 = _dS0_dparam(rows, 0, k=2, r=0, field="exponent")
    dm = _dM_from_dS0(s0, ds0)
    np.testing.assert_allclose(np.diag(dm), 0.0, atol=1e-12)


def test_dM_block_matches_fd():
    rows = [(0, [1.2], [1.0]), (1, [0.5], [1.0]), (2, [0.18], [1.0])]
    l = 0
    s0 = _block_S0(rows, l)
    dm_ana = _dM_from_dS0(s0, _dS0_dparam(rows, l, k=2, r=0, field="exponent"))

    def m_of(alpha2: float) -> np.ndarray:
        r2 = [(0, [1.2], [1.0]), (1, [0.5], [1.0]), (2, [alpha2], [1.0])]
        return _normalize(_block_S0(r2, l))

    h = 1e-7
    dm_fd = (m_of(0.18 + h) - m_of(0.18 - h)) / (2 * h)
    np.testing.assert_allclose(dm_ana, dm_fd, atol=1e-6)


# --------------------------------------------------------------------------
# Condition-number penalty gradient vs finite difference
# --------------------------------------------------------------------------


def _cond_scalar(p: BasisParametrisation, x: np.ndarray, gamma: float) -> float:
    atoms = p.unpack(np.asarray(x, dtype=float))
    return sum(cond_penalty_from_atom(atoms[s], gamma=gamma) for s in atoms)


def test_cond_gradient_matches_fd_on_diffuse_s_exponent():
    """pob-TZVP H, diffuse s (shell 2) — lives in the 3-shell l=0 block."""
    p = BasisParametrisation(
        atoms={"H": _pob("01_H")},
        free=[FreeSpec("H", shell_idx=2, prim_idx=0, field="exponent",
                       transform=Transform.LOG)],
    )
    x0 = p.pack()
    gamma = 1e-3
    ana = condition_number_penalty_gradient(p, x0, gamma=gamma)
    fd = np.array([_central_fd(lambda x: _cond_scalar(p, x, gamma), x0, 0)])
    assert abs(ana[0]) > 1e-6  # genuinely sensitive — not a trivial zero
    np.testing.assert_allclose(ana, fd, rtol=1e-5, atol=1e-9)


def test_cond_gradient_matches_fd_on_contracted_shell_exponent():
    """Vary one primitive exponent of the 5-primitive contracted s shell."""
    p = BasisParametrisation(
        atoms={"H": _pob("01_H")},
        free=[FreeSpec("H", shell_idx=0, prim_idx=0, field="exponent",
                       transform=Transform.LOG)],
    )
    x0 = p.pack()
    ana = condition_number_penalty_gradient(p, x0, gamma=1e-3)
    fd = np.array([_central_fd(lambda x: _cond_scalar(p, x, 1e-3), x0, 0)])
    np.testing.assert_allclose(ana, fd, rtol=1e-5, atol=1e-9)


def test_cond_gradient_matches_fd_on_coefficient():
    p = BasisParametrisation(
        atoms={"H": _pob("01_H")},
        free=[FreeSpec("H", shell_idx=0, prim_idx=1, field="coeff",
                       transform=Transform.LINEAR)],
    )
    x0 = p.pack()
    ana = condition_number_penalty_gradient(p, x0, gamma=1e-3)
    fd = np.array([_central_fd(lambda x: _cond_scalar(p, x, 1e-3), x0, 0)])
    np.testing.assert_allclose(ana, fd, rtol=1e-5, atol=1e-9)


def test_cond_gradient_zero_for_lone_p_shell():
    """A single p shell is its own 1x1 block (normalised overlap ≡ 1), so
    its exponent cannot move κ — analytic and FD must both be ~0."""
    p = BasisParametrisation(
        atoms={"H": _pob("01_H")},
        free=[FreeSpec("H", shell_idx=3, prim_idx=0, field="exponent",
                       transform=Transform.LOG)],
    )
    x0 = p.pack()
    ana = condition_number_penalty_gradient(p, x0, gamma=1e-3)
    fd = _central_fd(lambda x: _cond_scalar(p, x, 1e-3), x0, 0)
    assert ana[0] == pytest.approx(0.0, abs=1e-10)
    assert fd == pytest.approx(0.0, abs=1e-7)


def test_cond_gradient_multi_element_and_log_chain_rule():
    p = BasisParametrisation(
        atoms={"H": _pob("01_H"), "O": _pob("08_O")},
        free=[
            FreeSpec("H", shell_idx=2, prim_idx=0, field="exponent",
                     transform=Transform.LOG),
            FreeSpec("O", shell_idx=1, prim_idx=0, field="exponent",
                     transform=Transform.LOG),
        ],
    )
    x0 = p.pack()
    ana = condition_number_penalty_gradient(p, x0, gamma=2e-3)
    fd = np.array([
        _central_fd(lambda x: _cond_scalar(p, x, 2e-3), x0, i) for i in range(2)
    ])
    np.testing.assert_allclose(ana, fd, rtol=1e-5, atol=1e-9)


def test_cond_gradient_respects_gating():
    p = BasisParametrisation(
        atoms={"H": _pob("01_H"), "O": _pob("08_O")},
        free=[
            FreeSpec("H", shell_idx=2, prim_idx=0, field="exponent",
                     transform=Transform.LOG),
            FreeSpec("O", shell_idx=1, prim_idx=0, field="exponent",
                     transform=Transform.LOG),
        ],
    )
    x0 = p.pack()
    g = condition_number_penalty_gradient(p, x0, gamma=1e-3, gated_symbols=["O"])
    assert g[0] == 0.0  # H not gated → no contribution
    assert abs(g[1]) > 0.0


def test_linear_vs_log_chain_rule_consistency():
    """Same physical exponent, LOG vs LINEAR free spec: gradients differ by
    exactly the Jacobian dα/d(ln α) = α."""
    atom = _pob("01_H")
    alpha = atom.shells[2].exponents[0]
    p_log = BasisParametrisation(
        atoms={"H": _pob("01_H")},
        free=[FreeSpec("H", 2, 0, "exponent", transform=Transform.LOG)],
    )
    p_lin = BasisParametrisation(
        atoms={"H": _pob("01_H")},
        free=[FreeSpec("H", 2, 0, "exponent", transform=Transform.LINEAR)],
    )
    g_log = condition_number_penalty_gradient(p_log, p_log.pack(), gamma=1e-3)[0]
    g_lin = condition_number_penalty_gradient(p_lin, p_lin.pack(), gamma=1e-3)[0]
    assert g_log == pytest.approx(g_lin * alpha, rel=1e-9)


# --------------------------------------------------------------------------
# LD-hinge gradient
# --------------------------------------------------------------------------


def _ld_scalar(p, x, lambda_ld, epsilon):
    atoms = p.unpack(np.asarray(x, dtype=float))
    total = 0.0
    for s in atoms:
        diag = compute_overlap_diagnostics_from_atom(atoms[s])
        total += ld_penalty(diag, lambda_ld=lambda_ld, epsilon=epsilon)
    return total


def test_ld_gradient_matches_fd_in_active_region():
    """Force the hinge on with a large ε so λ_min < ε for the well-conditioned
    pob-H, and check the analytic derivative against FD."""
    p = BasisParametrisation(
        atoms={"H": _pob("01_H")},
        free=[FreeSpec("H", 2, 0, "exponent", transform=Transform.LOG)],
    )
    x0 = p.pack()
    eps = 0.5  # pob-H λ_min is well below 0.5 → hinge active
    lam = 1e3
    ana = ld_penalty_gradient(p, x0, lambda_ld=lam, epsilon=eps)
    fd = np.array([_central_fd(lambda x: _ld_scalar(p, x, lam, eps), x0, 0)])
    assert abs(ana[0]) > 1e-3
    np.testing.assert_allclose(ana, fd, rtol=1e-5, atol=1e-8)


def test_ld_gradient_zero_when_inactive():
    """At the real ε_LD the basis is well-conditioned → hinge off → grad 0."""
    p = BasisParametrisation(
        atoms={"H": _pob("01_H")},
        free=[FreeSpec("H", 2, 0, "exponent", transform=Transform.LOG)],
    )
    g = ld_penalty_gradient(p, p.pack())  # default epsilon = EPS_LD
    assert g[0] == 0.0


# --------------------------------------------------------------------------
# Scope guard
# --------------------------------------------------------------------------


def test_penalty_gradient_combines_active_terms():
    """penalty_gradient mirrors the objective's use_* flags: it sums whichever
    penalty terms are active and returns zero when none are."""
    p = BasisParametrisation(
        atoms={"H": _pob("01_H")},
        free=[FreeSpec("H", 2, 0, "exponent", transform=Transform.LOG)],
    )
    x0 = p.pack()
    cond = condition_number_penalty_gradient(p, x0, gamma=1e-3)
    eps = 0.5
    ld = ld_penalty_gradient(p, x0, lambda_ld=1e3, epsilon=eps)

    # cond only (default)
    np.testing.assert_allclose(penalty_gradient(p, x0, gamma=1e-3), cond)
    # both terms → sum
    both = penalty_gradient(
        p, x0, gamma=1e-3, use_ld_penalty=True, lambda_ld=1e3, epsilon=eps
    )
    np.testing.assert_allclose(both, cond + ld)
    # neither → zero
    np.testing.assert_allclose(
        penalty_gradient(p, x0, use_cond_penalty=False), np.zeros(1)
    )


def test_bdiis_uses_analytic_penalty_gradient_end_to_end():
    """The intended composition: BDIIS gradient = analytic energy grad +
    analytic condition-number-penalty grad. Verify the composite gradient
    matches FD and that BDIIS converges with it."""
    p = BasisParametrisation(
        atoms={"H": _pob("01_H")},
        free=[FreeSpec("H", 2, 0, "exponent", transform=Transform.LOG,
                       bounds=(0.05, 2.0))],
    )
    x0 = p.pack()
    target = float(np.log(0.40))  # toy energy bowl pulling ln(α) → ln(0.40)
    gamma = 1e-3

    class Obj:
        n_params = 1

        def __init__(self) -> None:
            self.n_calls = 0

        def __call__(self, x: np.ndarray) -> float:
            self.n_calls += 1
            atoms = p.unpack(np.asarray(x, dtype=float))
            energy = 0.5 * (float(x[0]) - target) ** 2
            return energy + cond_penalty_from_atom(atoms["H"], gamma=gamma)

    obj = Obj()

    def grad(x: np.ndarray) -> np.ndarray:
        energy_grad = np.array([float(x[0]) - target])
        return energy_grad + condition_number_penalty_gradient(p, x, gamma=gamma)

    # Composite analytic gradient agrees with FD of the composite objective.
    g_fd = np.array([_central_fd(obj, x0, 0)])
    np.testing.assert_allclose(grad(x0), g_fd, rtol=1e-5, atol=1e-7)

    res = optimize_bdiis(obj, x0, bounds=p.optim_bounds(), grad=grad)
    assert res.success
    assert res.x[0] == pytest.approx(target, abs=0.05)  # energy dominates tiny γ


def test_sp_shell_raises_not_implemented():
    from copy import deepcopy
    from vibeqc.basis_crystal import CrystalShell

    atom = _pob("01_H")
    atom.shells[0] = CrystalShell(
        shell_type="SP", occupancy=1.0, scale_factor=1.0,
        exponents=[1.0], coefficients=[1.0], coefficients_p=[1.0],
    )
    with pytest.raises(NotImplementedError, match="SP shells"):
        _atom_blocks(atom)
