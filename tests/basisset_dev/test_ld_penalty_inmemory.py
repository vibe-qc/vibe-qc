"""In-memory LD penalty — overlap parity, knob behaviour, objective wiring.

Covers the basis-optimizer audit follow-up that wired
``ld_penalty_from_atom`` into ``make_multi_objective`` so the LD penalty
is part of every objective evaluation rather than a post-hoc final-point
diagnostic.

The parity test against the file-path ``compute_overlap_diagnostics``
needs a built vibe-qc (libint + ``compute_overlap``); it skips when
that isn't importable so this file remains stdlib-only-runnable in CI
configurations that don't build the C++ extension.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Isolated import path — load the pure-Python subpackages we need without
# triggering the full ``vibeqc/__init__.py`` (which pulls the C++ extension).
#
# The copies are registered under a **private root package name**, never under
# ``vibeqc``.  Registering them as ``vibeqc.basis_crystal`` replaced the real
# entry in ``sys.modules`` for the rest of the pytest process while the already
# imported ``vibeqc`` package object kept its original submodule attribute —
# two live copies of one module, reachable by two different routes.  A later
# test that monkeypatched ``vibeqc.basis_crystal`` then patched the copy its own
# ``import vibeqc.basis_crystal`` statement resolved (via the package
# attribute), while ``from .basis_crystal import ...`` *inside* ``vibeqc``
# resolves through ``sys.modules`` and kept calling the unpatched original.
# That silently disarmed the fail-closed ECP-provenance guard in
# ``tests/test_periodic_ecp.py`` (CLAUDE.md § 1) whenever this file was
# collected first: its monkeypatched offline fetcher was never the one called.
#
# The relative imports inside the loaded modules (``...basis_crystal``,
# ``..ld_diagnostics``, ``..parametrise``) resolve against this private root,
# so the copies stay internally consistent.  Their *absolute* lazy imports
# (``import vibeqc as vq`` inside functions) still reach the real, built
# package — which is what the parity test at the bottom of this file wants.
#
# Sister files under this directory (test_crystal_ecp_parser.py and friends)
# instead shim ``sys.modules["vibeqc"]`` and restore the snapshot afterwards.
# A private root needs no snapshot, no restore, and no carve-out for the
# single-phase-init C extension, so prefer it for new files.
# ---------------------------------------------------------------------------

_ROOT = "_vibeqc_ld_penalty_test"
_REPO_PYTHON = Path(__file__).resolve().parents[2] / "python"


def _namespace(modname: str, relpath: str) -> types.ModuleType:
    """Register a synthetic package whose ``__path__`` is a real source dir."""
    if modname not in sys.modules:
        pkg = types.ModuleType(modname)
        pkg.__path__ = [str(_REPO_PYTHON / relpath)]
        sys.modules[modname] = pkg
    return sys.modules[modname]


def _load(modname: str, relpath: str):
    spec = importlib.util.spec_from_file_location(
        modname, str(_REPO_PYTHON / relpath)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    # Mirror what the real import machinery does, so ``getattr(parent, child)``
    # and the ``sys.modules`` entry can never disagree for these copies either.
    parent_name, _, child = modname.rpartition(".")
    if parent_name:
        setattr(sys.modules[parent_name], child, mod)
    return mod


_namespace(_ROOT, "vibeqc")
bc = _load(f"{_ROOT}.basis_crystal", "vibeqc/basis_crystal.py")

_namespace(f"{_ROOT}.basis_optimization", "vibeqc/basis_optimization")
ld = _load(
    f"{_ROOT}.basis_optimization.ld_diagnostics",
    "vibeqc/basis_optimization/ld_diagnostics.py",
)
pm = _load(
    f"{_ROOT}.basis_optimization.parametrise",
    "vibeqc/basis_optimization/parametrise.py",
)

# ---------------------------------------------------------------------------
# Closed-form sanity: single normalized primitive overlaps to itself = 1.
# ---------------------------------------------------------------------------


def test_self_overlap_normalised():
    """⟨α|α⟩ = 1 for any (α, l)."""
    for alpha in (0.1, 1.0, 17.3, 423.0):
        for l in (0, 1, 2, 3, 4):
            v = ld._normalized_primitive_overlap(alpha, alpha, l)
            assert abs(v - 1.0) < 1e-13, (alpha, l, v)


def test_two_close_exponents_near_one():
    """Two nearly-identical exponents have near-unit overlap (≈ 1 − O(δ²))."""
    a, b, l = 1.0, 1.001, 0
    v = ld._normalized_primitive_overlap(a, b, l)
    # Closed form: (2√(ab)/(a+b))^(3/2). For a=1, b=1+δ this is
    # (1 − δ²/16 + O(δ⁴))^(3/2) ≈ 1 − 3δ²/32, so the gap is microscopic.
    assert 1.0 - v < 1e-6 and 1.0 - v > 0


# ---------------------------------------------------------------------------
# Knob check — LD detection on a basis with two near-collinear primitives.
# ---------------------------------------------------------------------------


def _make_atom(shells):
    return bc.CrystalAtomBasis(Z=1, has_ecp=False, shells=list(shells))


def _s_shell(exponents, coefficients):
    return bc.CrystalShell(
        shell_type="S",
        occupancy=1.0,
        scale_factor=1.0,
        exponents=list(exponents),
        coefficients=list(coefficients),
    )


def test_well_conditioned_basis_no_ld():
    atom = _make_atom([
        _s_shell([34.0, 5.12, 1.16], [0.0060, 0.0450, 0.2019]),
        _s_shell([0.41], [1.0]),
        _s_shell([0.17], [1.0]),
    ])
    diag = ld.compute_overlap_diagnostics_from_atom(atom)
    assert not diag.ld_detected, diag.lambda_min
    assert diag.lambda_min > ld.EPS_LD
    assert diag.n_dependent == 0


def test_two_collinear_primitives_trip_ld():
    """Two single-primitive S shells with nearly identical exponents."""
    atom = _make_atom([
        _s_shell([1.0], [1.0]),
        _s_shell([1.0 + 1e-9], [1.0]),  # essentially identical
    ])
    diag = ld.compute_overlap_diagnostics_from_atom(atom)
    assert diag.ld_detected, (diag.lambda_min, diag.overlap_eigenvalues)
    assert diag.n_dependent >= 1
    # Penalty is positive and grows monotonically as λ_min shrinks.
    p = ld.ld_penalty_from_atom(atom, lambda_ld=1e3)
    assert p > 0


def test_ld_penalty_quadratic_growth():
    """``λ_ld · max(0, ε − λ_min)²`` — verify the quadratic shape."""
    base = ld.LDDiagnostics(
        overlap_eigenvalues=np.array([1e-9]),
        lambda_min=1e-9,
        condition_number=1.0,
        n_independent=0,
        n_dependent=1,
        ld_detected=True,
    )
    # Expected: 1e3 · (1e-7 − 1e-9)² = 1e3 · (9.9e-8)² ≈ 9.8e-12.
    p = ld.ld_penalty(base, lambda_ld=1e3, epsilon=1e-7)
    expected = 1e3 * (1e-7 - 1e-9) ** 2
    assert math.isclose(p, expected, rel_tol=1e-12)


def test_eigenvalue_multiplicity_matches_2l_plus_1():
    """A single P shell contributes 3 eigenvalues (m = −1, 0, +1)."""
    atom = bc.CrystalAtomBasis(
        Z=6, has_ecp=False,
        shells=[bc.CrystalShell(
            shell_type="P", occupancy=2.0, scale_factor=1.0,
            exponents=[1.0], coefficients=[1.0],
        )],
    )
    diag = ld.compute_overlap_diagnostics_from_atom(atom)
    assert diag.overlap_eigenvalues.shape == (3,)
    # All three are the same self-overlap (= 1).
    assert np.allclose(diag.overlap_eigenvalues, 1.0)


def test_sp_shell_decomposes_into_s_plus_p_block():
    """An SP shell contributes 1 S eigenvalue + 3 P eigenvalues."""
    atom = bc.CrystalAtomBasis(
        Z=6, has_ecp=False,
        shells=[bc.CrystalShell(
            shell_type="SP", occupancy=4.0, scale_factor=1.0,
            exponents=[1.0],
            coefficients=[1.0],         # S side
            coefficients_p=[1.0],       # P side
        )],
    )
    diag = ld.compute_overlap_diagnostics_from_atom(atom)
    assert diag.overlap_eigenvalues.shape == (4,)


def test_scale_factor_other_than_one_is_not_silent():
    """A SCAL != 1.0 shell must raise rather than silently mis-scale."""
    atom = _make_atom([bc.CrystalShell(
        shell_type="S", occupancy=1.0, scale_factor=1.5,
        exponents=[1.0], coefficients=[1.0],
    )])
    with pytest.raises(NotImplementedError, match="scale_factor"):
        ld.compute_overlap_diagnostics_from_atom(atom)


# ---------------------------------------------------------------------------
# Objective wiring — the in-objective penalty actually moves the value.
# ---------------------------------------------------------------------------


class _ConstantCalculator:
    """Returns a fixed energy regardless of basis_text / structure — lets us
    isolate the LD-penalty contribution to the objective."""
    def __init__(self, e=-100.0):
        self._e = e
    def evaluate_crystal(self, basis_text, structure, *, method):
        return self._e
    def evaluate_atom(self, basis_text, Z, *, method):
        return self._e


def _load_objective_module():
    # objective.py imports from ``..ld_diagnostics`` (already loaded) and
    # ``...basis_crystal`` (already loaded). emit_crystal lives in basis_crystal.
    return _load(
        f"{_ROOT}.basis_optimization.recipes.objective",
        "vibeqc/basis_optimization/recipes/objective.py",
    )


def test_make_multi_objective_in_loop_penalty_moves_value():
    _namespace(
        f"{_ROOT}.basis_optimization.recipes",
        "vibeqc/basis_optimization/recipes",
    )
    obj_mod = _load_objective_module()

    # Two identical-exponent S shells on H → LD-tripped basis.
    atom = bc.CrystalAtomBasis(Z=1, has_ecp=False, shells=[
        bc.CrystalShell(
            shell_type="S", occupancy=1.0, scale_factor=1.0,
            exponents=[1.0], coefficients=[1.0],
        ),
        bc.CrystalShell(
            shell_type="S", occupancy=0.0, scale_factor=1.0,
            exponents=[1.0 + 1e-10], coefficients=[1.0],
        ),
    ])
    p = pm.BasisParametrisation(atoms={"H": atom}, free=[])

    fake_structure = object()  # constant calculator ignores it
    base = obj_mod.make_multi_objective(
        p, _ConstantCalculator(-100.0), [fake_structure],
        use_ld_penalty=False,
    )
    penalised = obj_mod.make_multi_objective(
        p, _ConstantCalculator(-100.0), [fake_structure],
        use_ld_penalty=True, ld_lambda=1e6,
    )
    x0 = p.pack()
    assert base(x0) == -100.0
    val = penalised(x0)
    # Penalty is positive (basis is LD), so penalised > base.
    assert val > -100.0, (val, base(x0))


# ---------------------------------------------------------------------------
# Parity gate — the in-memory overlap must match libint's per-element compute
# to within ~1e-9 relative on a representative shipped basis. Skipped when
# vibeqc isn't importable; intended to actually run in CI (where the C++
# extension is built).
# ---------------------------------------------------------------------------


def _have_vibeqc() -> bool:
    """True when the built vibe-qc package (libint + C++ extension) imports.

    Probes for the attributes ``compute_overlap_diagnostics`` actually calls
    rather than trusting a bare ``import vibeqc``.  In a build-less checkout a
    sibling file in this directory (test_crystal_ecp_parser.py and friends)
    installs a bare namespace shim under the real ``vibeqc`` name so it can
    reach the pure-Python submodules; the import then *succeeds* against that
    shim, and the parity test below would run without libint and fail rather
    than skip.

    It must not mutate ``sys.modules``: the version before this popped the
    ``vibeqc`` entry and re-imported, which re-executed ``vibeqc/__init__.py``
    and left a second package object behind with its own submodule attributes.
    """
    try:
        mod = importlib.import_module("vibeqc")
    except Exception:
        return False
    # vq.Molecule / vq.Atom / vq.BasisSet / vq.compute_overlap are what
    # ld_diagnostics.compute_overlap_diagnostics needs from the built package.
    return all(
        hasattr(mod, attr)
        for attr in ("Molecule", "Atom", "BasisSet", "compute_overlap")
    )


@pytest.mark.skipif(
    not _have_vibeqc(), reason="needs the built vibeqc C++ extension"
)
def test_parity_against_file_path_overlap():
    """In-memory overlap eigenvalues match libint's to ~1e-9 relative."""
    # Build a representative one-element basis (Hydrogen, pob-TZVP-like:
    # one contracted S of 3 primitives + two uncontracted Ss + one P).
    atom = bc.CrystalAtomBasis(Z=1, has_ecp=False, shells=[
        bc.CrystalShell(
            shell_type="S", occupancy=1.0, scale_factor=1.0,
            exponents=[34.061341, 5.1235746, 1.1646626],
            coefficients=[0.00602519780, 0.04502109400, 0.20189726000],
        ),
        bc.CrystalShell(
            shell_type="S", occupancy=1.0, scale_factor=1.0,
            exponents=[0.41574551], coefficients=[1.0],
        ),
        bc.CrystalShell(
            shell_type="S", occupancy=0.0, scale_factor=1.0,
            exponents=[0.1795111], coefficients=[1.0],
        ),
        bc.CrystalShell(
            shell_type="P", occupancy=0.0, scale_factor=1.0,
            exponents=[0.8], coefficients=[1.0],
        ),
    ])
    inmem = ld.compute_overlap_diagnostics_from_atom(atom)

    from vibeqc.basis_crystal import emit_crystal
    basis_text = emit_crystal([atom])
    libint_diag = ld.compute_overlap_diagnostics(basis_text, Z=1)

    a = np.sort(inmem.overlap_eigenvalues)
    b = np.sort(libint_diag.overlap_eigenvalues)
    assert a.shape == b.shape, (a.shape, b.shape)
    # Relative match — eigenvalues span many orders of magnitude.
    assert np.allclose(a, b, rtol=1e-9, atol=1e-12), (a, b)
