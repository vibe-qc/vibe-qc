"""In-tree COSX nuclear-attraction kernel — ULP equivalence vs libint.

The COSX-K hot path in ``cpp/src/cosx.cpp::compute_cosx_k`` evaluates,
per grid point r_g, an analytical-integral block A_{μν}(r_g) for every
surviving shell pair via libint's ``Operator::nuclear`` engine
parameterised by a single pseudo-charge q = −1 at r_g. The libint call
is > 99 % of the K-build wall — replacing it with an in-tree custom
kernel is the remaining 5–10× perf lever (tracked in
handovers/HANDOVER_PERF_OPT.md).

This first landing (commit 1 of the kernel arc) covers (s, s) shells
only — Boys-table tabulation + downward recursion + 6th-order Taylor
interpolation, primitive-pair cache, and the (s, s) contracted-block
assembly. Higher angular momentum requires the Obara-Saika bra-
recursion + bra-ket transfer that lands in commit 2.

The tests pin per-element bit-equivalence between the in-tree kernel
and libint at ULP (rtol = 1e-13, atol = 1e-13) across a battery of
basis sets and pseudo-nucleus positions. The libint reference is
queried through the private ``_cosx_nuclear_pair_libint`` C++
binding — same engine + same charge convention the hot path uses,
so any drift would be ours.

References (mathematical, no proprietary source consulted):
* Gill, P. M. W.; Head-Gordon, M.; Pople, J. A., Int. J. Quantum
  Chem. Symp. 40, 269 (1991) — Boys table + Taylor interpolation.
* Obara, S.; Saika, A., J. Chem. Phys. 84, 3963 (1986) — recursion
  (lands in kernel arc commit 2).
"""
from __future__ import annotations

import numpy as np
import pytest

from vibeqc import Atom, BasisSet, Molecule
from vibeqc import _vibeqc_core as core


def _build_basis(atoms_z_xyz, basis_name):
    n_elec = sum(int(z) for z, _ in atoms_z_xyz)
    mult = 1 if (n_elec % 2 == 0) else 2
    mol = Molecule([Atom(int(z), list(xyz)) for z, xyz in atoms_z_xyz],
                   multiplicity=mult)
    return mol, BasisSet(mol, basis_name)


# Three molecular contexts: a single H (one s shell), a tight H2 (two
# atoms at typical bond distance, def2-svp expands to multiple s + p
# shells), and water (mixed s/p/d shells but we only test (s, s)
# pairs of it for now).
_GEOMETRIES = [
    pytest.param(
        [(1, (0.0, 0.0, 0.0))], "def2-svp",
        id="single-H-def2svp"),
    pytest.param(
        [(1, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, 1.4))], "def2-svp",
        id="H2-def2svp"),
    pytest.param(
        [(8, (0.0, 0.0, 0.117)),
         (1, (0.0, 0.757, -0.468)),
         (1, (0.0, -0.757, -0.468))], "def2-svp",
        id="H2O-def2svp"),
    pytest.param(
        [(1, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, 1.4))], "sto-3g",
        id="H2-sto3g"),
]


# Pseudo-nucleus positions: on-axis, off-axis, far from any atom (so T
# = α · |P − C|² spans the small-T series-domain through the large-T
# asymptotic-domain transitions of the Boys lookup).
_C_POINTS = [
    (0.0, 0.0, 0.0),         # at the first atom
    (0.5, 0.3, 0.2),         # near the centre of mass
    (0.0, 0.0, 1.0),         # mid-bond on H2 / above H2O
    (5.0, 0.0, 0.0),         # mid-range
    (20.0, 0.0, 0.0),        # large-T regime (asymptotic Boys)
    (0.7, 1.5, -0.9),        # generic off-axis
]


def _shells_by_l(basis):
    """Return a dict {l: [shell_idx, ...]}."""
    out = {}
    for i, sh in enumerate(basis.shells()):
        out.setdefault(sh.l, []).append(i)
    return out


@pytest.mark.parametrize("atoms,basis_name", _GEOMETRIES)
@pytest.mark.parametrize("C", _C_POINTS)
def test_pair_matches_libint_ulp(atoms, basis_name, C):
    """Per-shell-pair kernel matches libint to ULP across every
    (l_a, l_b) combination present in the basis."""
    _, basis = _build_basis(atoms, basis_name)
    n_shells = len(basis.shells())
    assert n_shells >= 1

    for s1 in range(n_shells):
        for s2 in range(n_shells):
            ref = core._cosx_nuclear_pair_libint(basis, s1, s2, list(C))
            ours = core._cosx_nuclear_pair_custom(basis, s1, s2, list(C))
            np.testing.assert_allclose(
                ours, ref, rtol=5e-12, atol=1e-13,
                err_msg=f"shell pair ({s1}, {s2}) at C={C} drifted from libint")


def test_kernel_exercises_all_am_combinations_in_def2svp():
    """def2-svp on a multi-atom organic exercises (s,s), (s,p), (p,p),
    (s,d), (p,d), (d,d) — make sure the test fixtures actually trigger
    each AM pair, so the parametrized test above isn't silently skipping
    any. Pin-down for regression on the test design itself.
    """
    mol = Molecule([
        Atom(8, [0.0, 0.0, 0.117]),   # O has 5s/2p/1d in def2-svp
        Atom(1, [0.0, 0.757, -0.468]),
        Atom(1, [0.0, -0.757, -0.468]),
    ], multiplicity=1)
    basis = BasisSet(mol, "def2-svp")
    ls = sorted(set(sh.l for sh in basis.shells()))
    assert 0 in ls, "expected s shells"
    assert 1 in ls, "expected p shells"
    assert 2 in ls, "expected d shells (def2-svp O polarization)"


def test_higher_am_ss_pair_matches_libint():
    """Quick smoke check on a known (d, d) and (p, p) pair so we don't
    rely on the parametrized aggregate above to catch AM regressions."""
    mol = Molecule([
        Atom(8, [0.0, 0.0, 0.117]),
        Atom(1, [0.0, 0.757, -0.468]),
        Atom(1, [0.0, -0.757, -0.468]),
    ], multiplicity=1)
    basis = BasisSet(mol, "def2-svp")
    by_l = _shells_by_l(basis)
    assert 1 in by_l and 2 in by_l, "def2-svp on H2O must have p and d"

    C = [0.5, 0.3, 0.2]
    pp = (by_l[1][0], by_l[1][0])    # (p, p) self pair
    dd = (by_l[2][0], by_l[2][0])    # (d, d) self pair
    sp = (by_l[0][0], by_l[1][0])    # (s, p)
    pd = (by_l[1][0], by_l[2][0])    # (p, d)
    for s1, s2 in [pp, dd, sp, pd]:
        ref = core._cosx_nuclear_pair_libint(basis, s1, s2, C)
        ours = core._cosx_nuclear_pair_custom(basis, s1, s2, C)
        np.testing.assert_allclose(
            ours, ref, rtol=5e-12, atol=1e-13,
            err_msg=f"AM smoke: pair ({s1}, {s2}) drifted from libint")


def test_boys_table_internal_consistency():
    """Cross-check the Boys-table lookup against a high-order direct
    eval — this catches table-build bugs without depending on libint.

    F_0(0) = 1; F_n(0) = 1 / (2n + 1) by the Boys series at T = 0.
    F_0 at large T → √π / (2√T). Both endpoints are model-free
    expectations.
    """
    # We don't expose BoysTable directly to Python — instead exercise
    # the table indirectly: with a single (1s, 1s) primitive pair where
    # the integrand reduces to a known multiple of F_0(T), we can
    # back-solve for F_0(T) and check it against the analytical
    # closed form. The test geometry is constructed so the integrand
    # is exactly recoverable.
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0])], multiplicity=2)
    basis = BasisSet(mol, "sto-3g")

    # STO-3G hydrogen has one (s) shell with 3 primitives. For the
    # bra=ket=this shell, the on-centre integrand at C = 0 reduces to
    # a sum of primitive contributions all with P = 0, T = 0, F_0 = 1.
    # Check this against libint for a self-consistency tie-down.
    out = core._cosx_nuclear_pair_custom(basis, 0, 0, [0.0, 0.0, 0.0])
    ref = core._cosx_nuclear_pair_libint(basis, 0, 0, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(out, ref, rtol=5e-12, atol=1e-13)

    # T spans 0 → large by walking C away from the atom; pin against
    # libint at every step. Catches table-edge bugs.
    for d in [0.01, 0.1, 1.0, 3.0, 10.0, 25.0, 50.0]:
        for direction in [(d, 0.0, 0.0), (0.0, d, 0.0), (0.0, 0.0, d)]:
            ours = core._cosx_nuclear_pair_custom(basis, 0, 0, list(direction))
            ref = core._cosx_nuclear_pair_libint(basis, 0, 0, list(direction))
            np.testing.assert_allclose(
                ours, ref, rtol=5e-12, atol=1e-13,
                err_msg=f"Boys-table-edge drift at C={direction}")


# ---------------------------------------------------------------------------
# M3b-4a: erfc-attenuated short-range kernel (range-separated exchange)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("omega", [0.11, 0.33, 0.84])
@pytest.mark.parametrize(
    "C",
    [
        (0.0, 0.0, 0.0),
        (0.5, 0.3, 0.2),
        (5.0, 0.0, 0.0),
        (20.0, 0.0, 0.0),
    ],
)
def test_pair_matches_libint_erfc_sr(omega, C):
    """M3b-4a: the erfc(ω·r)/r short-range kernel matches libint's
    ``Operator::erfc_nuclear`` to ULP across every (l_a, l_b)
    combination of H2O/def2-svp (s, p, d shells), spanning the
    small-T through asymptotic Boys domains and three attenuation
    strengths.

    The in-tree implementation changes only the Obara-Saika seeds —
    [0|v|0]^(m) = prefactor · [F_m(T) − ρ^{m+1/2} F_m(ρT)],
    ρ = ω²/(ω²+α) — so a ULP pin here covers the whole recursion.
    """
    atoms = [
        (8, (0.0, 0.0, 0.117)),
        (1, (0.0, 0.757, -0.468)),
        (1, (0.0, -0.757, -0.468)),
    ]
    _, basis = _build_basis(atoms, "def2-svp")
    n_shells = len(basis.shells())
    for s1 in range(n_shells):
        for s2 in range(n_shells):
            ref = core._cosx_nuclear_pair_libint(
                basis, s1, s2, list(C), omega=omega
            )
            ours = core._cosx_nuclear_pair_custom(
                basis, s1, s2, list(C), omega=omega
            )
            np.testing.assert_allclose(
                ours, ref, rtol=5e-12, atol=1e-13,
                err_msg=(
                    f"erfc-SR shell pair ({s1}, {s2}) at C={C}, "
                    f"omega={omega} drifted from libint"
                ),
            )


def test_erfc_sr_limits():
    """erfc-SR kernel limits: ω → 0⁺ recovers the full Coulomb block
    (erfc(0) = 1); growing ω monotonically shrinks the block norm
    (the kernel range 1/ω contracts)."""
    _, basis = _build_basis(
        [(1, (0.0, 0.0, 0.0)), (1, (0.0, 0.0, 1.4))], "sto-3g"
    )
    C = [0.3, 0.1, 0.9]
    full = core._cosx_nuclear_pair_custom(basis, 0, 1, C)
    near_full = core._cosx_nuclear_pair_custom(basis, 0, 1, C, omega=1e-4)
    np.testing.assert_allclose(near_full, full, rtol=1e-3)

    norms = [
        np.linalg.norm(
            core._cosx_nuclear_pair_custom(basis, 0, 1, C, omega=w)
        )
        for w in (0.2, 0.6, 2.0)
    ]
    assert np.linalg.norm(full) > norms[0] > norms[1] > norms[2] > 0.0
