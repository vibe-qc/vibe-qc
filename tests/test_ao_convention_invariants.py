"""AO-convention invariants at the libint boundary.

Two conventions are assumed throughout vibe-qc without being checked at the
places that assume them. Both are silent failure modes: nothing raises, the
numbers just come out wrong.

1. **Solid-harmonic ordering.** libint must emit the 2l+1 functions of a pure
   shell as m = -l..+l (STANDARD). ``cpp/src/ao_eval.cpp``, the molden writer,
   and the QVF writer each hardcode that independently.

2. **Cartesian primitive normalization.** libint applies one normalization per
   shell, derived from the total l, so a mixed Cartesian component is *not*
   unit-normalized -- a d shell has <xy|xy> = 1/3. QVF spec Appendix A.1
   specifies the same factor. Both AO evaluators used to add a per-component
   correction on top, making Cartesian densities 3x too large; they were
   corrected on 2026-08-05 and these tests pin all four surfaces together.

See ``cpp/src/ao_eval.cpp``, ``cpp/include/vibeqc/basis.hpp``,
``python/vibeqc/output/formats/qvf.py`` (``_basis_shell_payload``), and
``vibe-view/src/vibeview/renderers/wavefunction.py``.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from vibeqc import Atom, Molecule, compute_overlap, evaluate_ao
from vibeqc._vibeqc_core import BasisSet, ShellInfo
from vibeqc.output.formats.qvf import (
    _basis_shell_payload,
    _primitive_norm,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _shell(atom_index: int, l: int, *, pure: bool, alpha: float, origin) -> ShellInfo:
    s = ShellInfo()
    s.atom_index = atom_index
    s.l = l
    s.pure = pure
    s.exponents = [alpha]
    s.coefficients = [1.0]
    s.origin = list(origin)
    return s


def _one_atom(z: int = 1, multiplicity: int = 2) -> Molecule:
    return Molecule([Atom(z, [0.0, 0.0, 0.0])], 0, multiplicity)


# ---------------------------------------------------------------------------
# 1. Solid-harmonic ordering: m = -l .. +l
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("l", [1, 2, 3])
def test_integral_engine_uses_standard_shg_ordering(l):
    """libint's integral engine must place m = 0 at index l within a pure shell.

    Probed behaviourally rather than by reading a build macro, because the
    macro is not what the engine consults: ``SolidHarmonicsCoefficients``
    reads the *runtime* ``libint2::solid_harmonics_ordering()`` (see
    ``third_party/libint/install/include/libint2/solidharmonics.h``).

    The probe: a pure shell at the origin overlapped against an s function
    displaced along +z. Only the m = 0 (z-axial) component of the shell has
    nonzero overlap with it, so the index at which the overlap block peaks
    *is* the slot the engine assigned to m = 0.

    STANDARD ordering (m = -l..+l) puts m = 0 at index l.
    GAUSSIAN ordering (m = 0, +1, -1, ...) would put it at index 0.
    """
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])], 0, 1)
    basis = BasisSet(
        mol,
        [
            _shell(0, l, pure=True, alpha=0.9, origin=[0.0, 0.0, 0.0]),
            _shell(1, 0, pure=True, alpha=0.9, origin=[0.0, 0.0, 1.4]),
        ],
        "<shg-probe>",
        False,
    )
    n_pure = 2 * l + 1
    assert basis.nbasis == n_pure + 1

    block = np.abs(np.asarray(compute_overlap(basis))[:n_pure, n_pure])
    assert int(np.argmax(block)) == l, (
        f"libint put m=0 at index {int(np.argmax(block))} of a pure l={l} shell, "
        f"expected {l} (STANDARD, m=-l..+l). vibe-qc hardcodes STANDARD in "
        f"ao_eval.cpp, the molden writer, and the QVF writer -- see the "
        f"static_assert in cpp/include/vibeqc/basis.hpp and the runtime check "
        f"in ensure_libint_initialized()."
    )
    # Exactly one component may respond; a permutation that merely moved the
    # peak would still be caught above, but a transform that smeared m = 0
    # across components would not.
    assert np.count_nonzero(block > 1e-10) == 1


@pytest.mark.parametrize("l", [1, 2, 3])
def test_evaluate_ao_ordering_matches_integral_engine(l):
    """``evaluate_ao`` must agree with the integral engine on in-shell order.

    ``cart_to_pure_transform`` builds its rows with a literal
    ``for (int m = -l; m <= l; ++m)`` loop and never asks libint what
    ordering it is using, so this pins the two together. MO coefficients come
    from the integral basis and are evaluated with ``evaluate_ao``; if the two
    orderings ever diverge, every rendered orbital silently permutes.
    """
    mol = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])], 0, 1)
    basis = BasisSet(
        mol,
        [
            _shell(0, l, pure=True, alpha=0.9, origin=[0.0, 0.0, 0.0]),
            _shell(1, 0, pure=True, alpha=0.9, origin=[0.0, 0.0, 1.4]),
        ],
        "<shg-probe>",
        False,
    )
    n_pure = 2 * l + 1

    engine_m0 = int(np.argmax(np.abs(np.asarray(compute_overlap(basis))[:n_pure, n_pure])))

    # On the +z axis every real solid harmonic with m != 0 vanishes.
    chi = np.asarray(evaluate_ao(basis, np.array([[0.0, 0.0, 0.7]])))[0, :n_pure]
    eval_m0 = int(np.argmax(np.abs(chi)))

    assert eval_m0 == engine_m0 == l
    assert np.count_nonzero(np.abs(chi) > 1e-10) == 1


def test_vendored_libint_is_built_standard():
    """The vendored libint build macro must say STANDARD.

    Secondary to the behavioural tests above -- the macro seeds only the
    legacy ``FOR_SOLIDHARM`` / ``INT_SOLIDHARMINDEX`` helpers, not the engine
    -- but it is the thing a rebuild would change, and it mirrors the
    ``static_assert`` in ``cpp/include/vibeqc/basis.hpp`` so the Python side
    fails with the same diagnosis rather than a mystery numerical drift.
    """
    config = REPO_ROOT / "third_party/libint/install/include/libint2/config.h"
    if not config.is_file():
        pytest.skip("vendored libint headers not present in this checkout")

    values: dict[str, int] = {}
    for line in config.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "#define" and "SHGSHELL_ORDERING" in parts[1]:
            try:
                values[parts[1]] = int(parts[2])
            except ValueError:
                pass

    assert "LIBINT_SHGSHELL_ORDERING" in values, (
        "LIBINT_SHGSHELL_ORDERING is absent from the vendored libint config.h. "
        "Upstream marks it retired; if it is gone for good, drop the guarded "
        "static_assert in cpp/include/vibeqc/basis.hpp and rely on the runtime "
        "check in ensure_libint_initialized() plus the behavioural tests here."
    )
    assert values["LIBINT_SHGSHELL_ORDERING"] == values["LIBINT_SHGSHELL_ORDERING_STANDARD"]


def test_vendored_libint_uses_standard_cartesian_ordering():
    """The vendored libint's Cartesian (CGShell) ordering must be STANDARD.

    Companion to the ``LIBINT_CGSHELL_ORDERING`` ``static_assert`` in
    ``cpp/include/vibeqc/basis.hpp``. Unlike the solid-harmonic ordering,
    this macro has no runtime accessor and no deprecation shim -- libint's
    generated engine code is specialized to it at build time -- so the macro
    IS the convention. vibe-qc hardcodes the standard order (lx descending,
    then ly descending) in ``enumerate_cartesians`` (ao_eval.cpp),
    cosx_kernel.cpp, QVF spec Appendix A.2,
    ``_aopair_ft.cartesian_components_for_l``, and cart_to_sph_data.hpp.

    The behavioural anchor is ``test_evaluate_ao_reproduces_the_integral_basis``
    below: an engine with a permuted Cartesian order would break grid-vs-engine
    overlap parity off the diagonal for Cartesian shells with l >= 2 (the
    orderings do not differ on p shells).
    """
    config = REPO_ROOT / "third_party/libint/install/include/libint2/config.h"
    if not config.is_file():
        pytest.skip("vendored libint headers not present in this checkout")

    values: dict[str, int] = {}
    for line in config.read_text().splitlines():
        parts = line.split()
        if len(parts) == 3 and parts[0] == "#define" and "CGSHELL_ORDERING" in parts[1]:
            try:
                values[parts[1]] = int(parts[2])
            except ValueError:
                pass

    assert "LIBINT_CGSHELL_ORDERING" in values, (
        "LIBINT_CGSHELL_ORDERING is absent from the vendored libint config.h; "
        "the static_assert in cpp/include/vibeqc/basis.hpp can no longer see "
        "the convention. Find where upstream moved it before trusting a build."
    )
    assert values["LIBINT_CGSHELL_ORDERING"] == values["LIBINT_CGSHELL_ORDERING_STANDARD"]


# ---------------------------------------------------------------------------
# 2. Cartesian normalization: the split the QVF writer refuses to cross
# ---------------------------------------------------------------------------


def _cart_d_basis(alpha: float = 0.8) -> BasisSet:
    return BasisSet(
        _one_atom(),
        [_shell(0, 2, pure=False, alpha=alpha, origin=[0.0, 0.0, 0.0])],
        "<cart-d>",
        False,
    )


def test_integral_basis_cartesian_d_is_not_unit_normalized():
    """Pins libint's Cartesian convention, which QVF spec A.1's N_i encodes.

    The shell's six components come out in libint lexicographic order
    (xx, xy, xz, yy, yz, zz). Only the axial ones are unit-normalized; the
    mixed ones sit at 1/3. If this ever changes, spec Appendix A.1 and the
    QVF writer's ``_primitive_norm`` both need revisiting.
    """
    S = np.asarray(compute_overlap(_cart_d_basis()))
    np.testing.assert_allclose(
        np.diag(S), [1.0, 1 / 3, 1 / 3, 1.0, 1 / 3, 1.0], rtol=0, atol=1e-12
    )


def test_qvf_primitive_norm_is_the_axial_norm_not_the_per_component_norm():
    """``_primitive_norm`` must equal QVF spec A.1's ``N_i`` exactly.

    That is the axial (l, 0, 0) normalization applied shell-wide -- not the
    per-component unit norm, which differs by √3 for d_xy. The writer divides
    libint's stored coefficients by this factor, so writer and spec agree
    only as long as this identity holds.
    """
    alpha, l = 0.8, 2

    def axial(a: float, ll: int) -> float:
        df = 1.0
        for k in range(1, 2 * ll, 2):
            df *= k
        return (2.0 * a / math.pi) ** 0.75 * (4.0 * a) ** (ll / 2.0) / math.sqrt(df)

    def per_component(a: float, lx: int, ly: int, lz: int) -> float:
        def df(n: int) -> float:
            r = 1.0
            for k in range(1, 2 * n, 2):
                r *= k
            return r

        ll = lx + ly + lz
        return (
            (2.0 * a / math.pi) ** 0.75
            * (4.0 * a) ** (ll / 2.0)
            / math.sqrt(df(lx) * df(ly) * df(lz))
        )

    assert _primitive_norm(alpha, l) == pytest.approx(axial(alpha, l), rel=1e-14)
    assert _primitive_norm(alpha, l) == pytest.approx(
        per_component(alpha, 2, 0, 0), rel=1e-14
    )
    assert per_component(alpha, 1, 1, 0) / _primitive_norm(alpha, l) == pytest.approx(
        math.sqrt(3.0), rel=1e-14
    )


def _gauss_hermite_cube(alpha: float, n: int = 40):
    """Product Gauss-Hermite nodes/weights, exact for polynomial x exp(-2a r^2).

    Returns (points, weights) already de-weighted, so
    ``sum_g w_g f(r_g)`` integrates ``f`` over all space.
    """
    x, w = np.polynomial.hermite_e.hermegauss(n)
    scale = math.sqrt(1.0 / (4.0 * alpha))
    r, wr = x * scale, w * scale
    X, Y, Z = np.meshgrid(r, r, r, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    weights = np.einsum("i,j,k->ijk", wr, wr, wr).ravel()
    # Undo the exp(-t^2/2) already folded into the quadrature weights.
    return pts, weights * np.exp(2.0 * alpha * (pts**2).sum(axis=1))


@pytest.mark.parametrize("l,pure", [(0, False), (1, False), (2, False), (3, False), (2, True)])
def test_evaluate_ao_reproduces_the_integral_basis(l, pure):
    """``evaluate_ao`` on a grid must integrate to ``compute_overlap``.

    This is the invariant that makes grid evaluation meaningful at all: MO
    coefficients, density matrices and every operator matrix are expressed
    in libint's basis, so the evaluator has to produce *those* functions.

    It held for spherical shells and was broken for Cartesian ones until
    2026-08-05: ``ao_eval.cpp`` multiplied Cartesian output by
    ``cart_relative_norm`` = √((2l-1)!!/((2lx-1)!!(2ly-1)!!(2lz-1)!!)) to
    reach true unit norm per component, which libint's engine does not do.
    A d_xy came out √3 too large, so any density built from it was 3x.
    """
    alpha = 0.8
    basis = BasisSet(
        _one_atom(),
        [_shell(0, l, pure=pure, alpha=alpha, origin=[0.0, 0.0, 0.0])],
        "<cmp>",
        False,
    )
    pts, weights = _gauss_hermite_cube(alpha)
    chi = np.asarray(evaluate_ao(basis, pts))
    S_grid = np.einsum("gi,gj,g->ij", chi, chi, weights)
    np.testing.assert_allclose(
        S_grid, np.asarray(compute_overlap(basis)), rtol=1e-8, atol=1e-9
    )


def test_cartesian_d_xy_normalised_in_integral_basis_evaluates_to_one():
    """The decisive end-to-end statement, stated in the units that matter.

    Normalise a d_xy coefficient vector where MO coefficients live
    (c^T S c = 1), then evaluate it where a renderer evaluates. Before the
    2026-08-05 fix this integrated to exactly 3.0.
    """
    alpha = 0.8
    basis = _cart_d_basis(alpha)
    S = np.asarray(compute_overlap(basis))
    pts, weights = _gauss_hermite_cube(alpha)
    chi = np.asarray(evaluate_ao(basis, pts))
    S_grid = np.einsum("gi,gj,g->ij", chi, chi, weights)

    c = np.zeros(6)
    c[1] = 1.0  # d_xy, libint lexicographic index 1
    c /= math.sqrt(c @ S @ c)
    assert c @ S @ c == pytest.approx(1.0, rel=1e-10)
    assert c @ S_grid @ c == pytest.approx(1.0, rel=1e-6)


def test_qvf_writer_emits_cartesian_shells():
    """Cartesian shells are emitted again (the 2026-08-05 rejection is lifted).

    The rejection existed only because both readers mis-scaled Cartesian
    payloads; with them corrected the payload round-trips, so refusing to
    write one would block a correct path. See
    :func:`test_qvf_cartesian_payload_round_trips_through_vibe_view`.
    """
    shells, pure_top, n_ao = _basis_shell_payload(_cart_d_basis())
    assert pure_top is False
    assert n_ao == 6  # (l+1)(l+2)/2, not 2l+1
    assert len(shells) == 1
    assert shells[0]["pure"] is False
    assert shells[0]["l"] == 2


def test_qvf_writer_reports_mixed_purity_honestly():
    basis = BasisSet(
        _one_atom(),
        [
            _shell(0, 0, pure=True, alpha=0.8, origin=[0.0, 0.0, 0.0]),
            _shell(0, 2, pure=False, alpha=0.8, origin=[0.0, 0.0, 0.0]),
        ],
        "<mixed>",
        False,
    )
    shells, pure_top, n_ao = _basis_shell_payload(basis)
    assert [sh["pure"] for sh in shells] == [True, False]
    assert pure_top is False
    assert n_ao == 1 + 6


def test_qvf_cartesian_payload_round_trips_through_vibe_view():
    """The loop this whole thread is about, closed end to end.

    Take a Cartesian d_xy normalised in vibe-qc's integral basis, emit the
    QVF payload, hand it to vibe-view's own evaluator, and integrate the
    density. It must come back as 1 electron.

    Before 2026-08-05 this returned 3.0, and neither side looked wrong on
    its own: the writer followed spec A.1's formula, the reader followed
    A.1's prose.
    """
    vibeview_wf = pytest.importorskip(
        "vibeview.renderers.wavefunction",
        reason="vibe-view not installed in this environment",
    )
    alpha = 0.8
    basis = _cart_d_basis(alpha)
    S = np.asarray(compute_overlap(basis))
    shells, _pure_top, n_ao = _basis_shell_payload(basis)
    assert n_ao == 6

    c = np.zeros(6)
    c[1] = 1.0
    c /= math.sqrt(c @ S @ c)

    # Rebuild the AO on a grid the way a QVF consumer must: N_i from the
    # emitted (exponent, l), times the bare Cartesian monomials.
    n = 140
    half = 8.0
    ax = np.linspace(-half, half, n)
    d = float(ax[1] - ax[0])
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    r2 = gx * gx + gy * gy + gz * gz

    shell = shells[0]
    radial = np.zeros_like(r2)
    for a, coeff in zip(shell["exponents"], shell["coefficients"]):
        radial += coeff * vibeview_wf._primitive_norm(int(shell["l"]), float(a)) * np.exp(
            -a * r2
        )
    angs = list(vibeview_wf._cartesian_factors(int(shell["l"]), gx, gy, gz))
    assert len(angs) == 6

    psi = np.zeros_like(r2)
    for k, ang in enumerate(angs):
        if c[k] != 0.0:
            psi += c[k] * radial * ang

    assert float(np.sum(psi * psi) * d**3) == pytest.approx(1.0, abs=0.01)


def test_qvf_writer_still_emits_pure_shells():
    """The rejection must not disturb the spherical path, which is every
    basis vibe-qc builds by name."""
    basis = BasisSet(
        _one_atom(),
        [_shell(0, 2, pure=True, alpha=0.8, origin=[0.0, 0.0, 0.0])],
        "<pure-d>",
        False,
    )
    shells, pure_top, n_ao = _basis_shell_payload(basis)
    assert pure_top is True
    assert n_ao == 5
    assert len(shells) == 1
    assert shells[0]["pure"] is True
    assert shells[0]["l"] == 2

    # Coefficients are divided by N_i on the way out (spec Appendix A.1).
    native = float(list(basis.shells())[0].coefficients[0])
    assert shells[0]["coefficients"][0] == pytest.approx(
        native / _primitive_norm(0.8, 2), rel=1e-12
    )


def test_named_basis_route_never_produces_cartesian_shells():
    """``set_pure(true)`` (cpp/src/basis.cpp) keeps every *named-basis*
    archive spherical, which is why the Cartesian convention error sat
    undetected on the ordinary test surface. It is not the only production
    route: ``vibeqc.basis_toolkit.to_libint_basis`` honors an imported
    basis's ``harmonic_type`` and does build Cartesian shells (shipped since
    v0.15.0), which is how the error was reachable in the field -- see
    handovers/HANDOVER_AO_CONVENTION.md.

    Worth pinning independently of the Cartesian fixes: the named-basis
    route is the reason a regression on the Cartesian path would again be
    invisible to the whole normal test surface."""
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 1.5, -1.16]),
            Atom(1, [0.0, -1.5, -1.16]),
        ]
    )
    for name in ("sto-3g", "6-31g*", "cc-pvdz"):
        basis = BasisSet(mol, name)
        assert all(bool(s.pure) for s in basis.shells()), name
        _shells, pure_top, _n_ao = _basis_shell_payload(basis)
        assert pure_top is True
