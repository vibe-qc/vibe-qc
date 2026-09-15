"""Parity: native ``prune_tight_primitives`` ≡ the Python softened-basis fallback.

The GAPW hard/soft density split (``vibeqc.periodic_gapw_augment``) builds a
"softened" basis by dropping tight Gaussian primitives from each contracted
shell. ``softened_basis`` prefers the native kernel
``_vibeqc_core.prune_tight_primitives`` and falls back to a pure-Python
implementation when the binding is absent. The two paths MUST agree
bit-for-bit — otherwise rebuilding the extension (which flips
``softened_basis`` from the Python path to the C++ path) would silently move
every GAPW SCF number. This module pins that equivalence on a core-bearing
basis (O/cc-pvdz, whose 1s primitives reach ~1.2e4 bohr⁻²) across the three
pruning regimes the contract distinguishes:

  * ``cutoff = 3.0`` — no shell fully pruned, but the contracted s/s/p shells
    collapse to a single surviving primitive. Catches any coefficient
    renormalisation divergence: the fallback keeps coefficients **as-is**
    (``coefficients_pre_normalized=True``), it does **not** rescale them.
  * ``cutoff = 1.0`` — four of six shells are fully pruned and **dropped**
    entirely (nbasis 14 → 4). Catches a "keep the loosest primitive instead
    of dropping the shell" divergence.
  * ``cutoff = 0.1`` — every shell is pruned; the **original basis** is
    returned unchanged.

The authoritative contract is ``softened_basis``'s fallback in
``python/vibeqc/periodic_gapw_augment.py``; ``_python_fallback`` below is a
verbatim copy of it so this test pins the semantics independently of the
dispatch and without reaching into that module's internals.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import _vibeqc_core as core

_HAS_CPP_PRUNE = hasattr(core, "prune_tight_primitives")

pytestmark = pytest.mark.skipif(
    not _HAS_CPP_PRUNE,
    reason=(
        "native _vibeqc_core.prune_tight_primitives is not built — "
        "softened_basis falls back to pure Python; rebuild the extension "
        "to exercise (and validate) the C++ fast path."
    ),
)


def _oxygen_ccpvdz():
    """O atom (triplet) + its cc-pvdz BasisSet — a core-bearing basis."""
    mol = vq.Molecule([vq.Atom(8, [0.0, 0.0, 0.0])], 0, 3)
    basis = core.BasisSet(mol, "cc-pvdz")
    return mol, basis


def _python_fallback(basis, mol, soft_cutoff):
    """Verbatim copy of softened_basis's pure-Python fallback (the contract)."""
    pruned_shells = []
    for sh in basis.shells():
        kept_exp = []
        kept_coef = []
        for e, c in zip(sh.exponents, sh.coefficients):
            if e <= soft_cutoff:
                kept_exp.append(e)
                kept_coef.append(c)
        if not kept_exp:
            continue  # shell has no soft primitives — drop it
        pruned = core.ShellInfo()
        pruned.atom_index = sh.atom_index
        pruned.l = sh.l
        pruned.pure = sh.pure
        pruned.exponents = kept_exp
        pruned.coefficients = kept_coef
        pruned.origin = sh.origin
        pruned_shells.append(pruned)
    if not pruned_shells:
        # all primitives pruned — the fallback returns the original basis
        return basis
    return core.BasisSet(
        mol,
        pruned_shells,
        f"{basis.name}-soft",
        coefficients_pre_normalized=True,
    )


def _grid_points(seed=0, n=64, half=3.0):
    """Deterministic cloud of points around the (origin-centred) atom."""
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.uniform(-half, half, size=(n, 3)))


def _assert_shells_identical(a, b):
    """Same shell list: l / pure / atom / origin / exponents / coefficients."""
    sa, sb = a.shells(), b.shells()
    assert len(sa) == len(sb), f"nshells differ: {len(sa)} vs {len(sb)}"
    for i, (x, y) in enumerate(zip(sa, sb)):
        assert x.atom_index == y.atom_index, f"shell {i}: atom_index"
        assert x.l == y.l, f"shell {i}: l"
        assert x.pure == y.pure, f"shell {i}: pure"
        np.testing.assert_allclose(
            list(x.origin), list(y.origin), atol=1e-14, err_msg=f"shell {i}: origin"
        )
        assert len(x.exponents) == len(y.exponents), f"shell {i}: nprim"
        np.testing.assert_allclose(
            list(x.exponents), list(y.exponents),
            rtol=1e-13, atol=1e-14, err_msg=f"shell {i}: exponents",
        )
        np.testing.assert_allclose(
            list(x.coefficients), list(y.coefficients),
            rtol=1e-13, atol=1e-14, err_msg=f"shell {i}: coefficients",
        )


def _assert_density_identical(a, b, seed=0):
    """Same AOs on a grid, and the same AO density ρ(r)=Σ χ_m D_mn χ_n."""
    pts = _grid_points(seed=seed)
    chi_a = np.asarray(core.evaluate_ao(a, pts))
    chi_b = np.asarray(core.evaluate_ao(b, pts))
    assert chi_a.shape == chi_b.shape, (chi_a.shape, chi_b.shape)
    np.testing.assert_allclose(chi_a, chi_b, rtol=1e-12, atol=1e-12)
    # Density with a deterministic symmetric D, exactly as the GAPW
    # augmentation consumes the softened AOs.
    n = chi_a.shape[1]
    rng = np.random.default_rng(seed + 1)
    m = rng.standard_normal((n, n))
    dens = m + m.T
    rho_a = np.einsum("gm,mn,gn->g", chi_a, dens, chi_a, optimize=True)
    rho_b = np.einsum("gm,mn,gn->g", chi_b, dens, chi_b, optimize=True)
    np.testing.assert_allclose(rho_a, rho_b, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("cutoff", [3.0, 1.0, 0.1])
def test_cpp_matches_python_fallback(cutoff):
    """The C++ kernel produces a basis IDENTICAL to the Python fallback."""
    mol, basis = _oxygen_ccpvdz()
    soft_cpp = core.prune_tight_primitives(basis, mol, cutoff)
    soft_py = _python_fallback(basis, mol, cutoff)
    assert soft_cpp.nshells == soft_py.nshells
    assert soft_cpp.nbasis == soft_py.nbasis
    _assert_shells_identical(soft_cpp, soft_py)
    _assert_density_identical(soft_cpp, soft_py)


def test_pruning_regimes_are_distinct():
    """Sanity: the three cutoffs genuinely exercise the three code branches."""
    mol, basis = _oxygen_ccpvdz()

    # cutoff 3.0 — no shell dropped, but primitives are actually removed.
    s3 = core.prune_tight_primitives(basis, mol, 3.0)
    assert s3.nshells == basis.nshells
    assert s3.nbasis == basis.nbasis
    n_prim_orig = sum(len(sh.exponents) for sh in basis.shells())
    n_prim_soft = sum(len(sh.exponents) for sh in s3.shells())
    assert n_prim_soft < n_prim_orig, "cutoff 3.0 should prune some primitives"

    # cutoff 1.0 — some shells fully pruned → dropped → nbasis shrinks.
    s1 = core.prune_tight_primitives(basis, mol, 1.0)
    assert s1.nshells < basis.nshells, "cutoff 1.0 should drop whole shells"
    assert s1.nbasis < basis.nbasis

    # cutoff 0.1 — everything pruned → original basis returned unchanged.
    s0 = core.prune_tight_primitives(basis, mol, 0.1)
    assert s0.nshells == basis.nshells
    assert s0.nbasis == basis.nbasis


def test_softened_basis_prefers_cpp_path():
    """softened_basis() routes through the C++ kernel and matches it exactly."""
    from vibeqc.periodic_gapw_augment import softened_basis

    mol, basis = _oxygen_ccpvdz()
    soft = softened_basis(basis, mol, soft_cutoff=3.0)
    soft_cpp = core.prune_tight_primitives(basis, mol, 3.0)
    _assert_shells_identical(soft, soft_cpp)
    _assert_density_identical(soft, soft_cpp)
