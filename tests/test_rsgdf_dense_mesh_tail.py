"""RSGDF dense-mesh completeness + high-|G| tail completion (prompt 11 / P01).

Two related defects on dense-core Gamma cells (MgO/STO-3G class):

1. ``rsgdf_dense_g_mesh`` sized its integer index box as
   ``ceil(G_max / |b_i|) + 1``, which is exact only for orthogonal cells.
   For oblique lattices the correct per-axis bound follows from
   biorthogonality (``n_i = G . a_i / 2pi``), i.e.
   ``|n_i| <= G_max |a_i| / 2pi`` -- strictly larger, so the historical box
   silently DROPPED reciprocal-lattice points near the sphere boundary
   (fcc primitive: ~15% per-axis deficit, growing with the cutoff). Every
   "ke_cutoff" mesh on the P01/LiH fcc-class cells was subtly incomplete.

2. Tight core AO products need reciprocal support far beyond any affordable
   base mesh; ``tail_ke_cutoff`` extends BOTH the 2c metric and the 3c
   tensor G-sums over the exact complementary shell with the same analytic
   kernels (no real-space/FT representation mixing). By construction
   ``Lpq(ke=a, tail=b)`` spans the same G set as ``Lpq(ke=b)``.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import _vibeqc_core as core
from vibeqc._vibeqc_core import LatticeSumOptions
from vibeqc.aux_basis import (
    _aft_g_mesh,
    _rsgdf_shifted_dense_g_mesh,
    _rsgdf_shifted_sphere_boundary_tolerance,
    build_lpq_bloch_native_fft,
    build_lpq_native_fft,
    make_aux_basis_set,
    make_modrho_aux_basis,
    rsgdf_dense_g_mesh,
    rsgdf_g_mesh,
)
from vibeqc._aopair_ft import ao_pair_fourier_transform_bloch


def _fcc_system(a: float, atoms):
    sysp = core.PeriodicSystem()
    sysp.dim = 3
    sysp.lattice = np.array(
        [[0.0, a, a], [a, 0.0, a], [a, a, 0.0]], dtype=float
    )
    sysp.unit_cell = atoms
    return sysp


def _brute_force_count(system, ke_cutoff: float) -> int:
    A = np.asarray(system.lattice, dtype=float)
    B = 2.0 * np.pi * np.linalg.inv(A).T
    G_max = np.sqrt(2.0 * ke_cutoff)
    nmax = int(np.ceil(G_max * np.linalg.norm(A, axis=1).max() / (2 * np.pi))) + 5
    r = np.arange(-nmax, nmax + 1)
    n1, n2, n3 = np.meshgrid(r, r, r, indexing="ij")
    idx = np.stack([n1.ravel(), n2.ravel(), n3.ravel()], -1).astype(float)
    G = idx @ B.T
    return int((np.linalg.norm(G, axis=1) <= G_max).sum())


class _ProgressCapture:
    def __init__(self):
        self.lines = []

    def info(self, message: str) -> None:
        self.lines.append(message)


@pytest.mark.parametrize("ke", [50.0, 200.0])
def test_dense_g_mesh_complete_on_oblique_fcc(ke):
    """The mesh contains EVERY reciprocal-lattice point with
    |G| <= sqrt(2 ke) on an oblique (fcc-primitive) cell -- the historical
    norm-based index box dropped boundary points here."""
    a = 3.97976322  # P01 MgO primitive
    sysp = _fcc_system(a, [core.Atom(12, [0, 0, 0]), core.Atom(8, [a, a, a])])
    mesh = rsgdf_dense_g_mesh(sysp, ke)
    assert mesh.shape[0] == _brute_force_count(sysp, ke)


def test_dense_g_mesh_complete_on_cubic_cell():
    """Orthogonal cells were unaffected by the box bug -- pin no change."""
    L = 10.0
    sysp = core.PeriodicSystem()
    sysp.dim = 3
    sysp.lattice = np.eye(3) * L
    sysp.unit_cell = [core.Atom(1, [L / 2, L / 2, L / 2])]
    mesh = rsgdf_dense_g_mesh(sysp, 100.0)
    assert mesh.shape[0] == _brute_force_count(sysp, 100.0)


def _nonsymmetric_system():
    """A genuinely NON-symmetric (lower-triangular) lattice matrix.

    The fcc-primitive controls above use a symmetric lattice matrix (rows ==
    columns), so neither the legacy single-``n_max`` box nor a row-norm
    variant clips them -- only a non-symmetric matrix exercises the
    column-norm per-axis bound (``_reciprocal_index_box_per_axis``). Regression
    for the 2026-07-14 GDF audit: the column-norm fix that landed in
    ``rsgdf_dense_g_mesh`` (2026-07-10) had not propagated to its siblings
    ``rsgdf_g_mesh`` / ``_aft_g_mesh``, which kept the clipping ``min|b_i|``
    bound.
    """
    sysp = core.PeriodicSystem()
    sysp.dim = 3
    sysp.lattice = np.array(
        [[6.0, 0.0, 0.0], [2.0, 6.0, 0.0], [0.0, 2.0, 6.0]], dtype=float
    )
    sysp.unit_cell = [core.Atom(1, [0.0, 0.0, 0.0])]
    return sysp


def _ball_keys(system, g_max):
    """Set of every reciprocal-lattice point with |G| <= g_max (brute force)."""
    A = np.asarray(system.lattice, dtype=float)
    B = 2.0 * np.pi * np.linalg.inv(A).T
    nmax = int(np.ceil(g_max * np.linalg.norm(A, axis=0).max() / (2 * np.pi))) + 5
    r = np.arange(-nmax, nmax + 1)
    n1, n2, n3 = np.meshgrid(r, r, r, indexing="ij")
    idx = np.stack([n1.ravel(), n2.ravel(), n3.ravel()], -1).astype(float)
    G = idx @ B.T
    inb = G[np.linalg.norm(G, axis=1) <= g_max - 1e-9]
    return {(round(g[0], 6), round(g[1], 6), round(g[2], 6)) for g in inb}


def _mesh_keys(mesh):
    return {
        (round(g[0], 6), round(g[1], 6), round(g[2], 6))
        for g in np.asarray(mesh, dtype=float)
    }


def test_rsgdf_g_mesh_complete_on_nonsymmetric_cell():
    """rsgdf_g_mesh must cover the full |G| <= G_max ball on a non-symmetric
    lattice (audit fix: shared column-norm per-axis box)."""
    sysp = _nonsymmetric_system()
    omega, precision, pad = 0.5, 1e-8, 1.5
    g_max = pad * 2.0 * omega * np.sqrt(np.log(1.0 / precision))
    mesh = rsgdf_g_mesh(sysp, omega=omega, precision=precision, pad_factor=pad)
    assert _ball_keys(sysp, g_max) <= _mesh_keys(mesh)


def test_aft_g_mesh_complete_on_nonsymmetric_cell():
    """_aft_g_mesh must cover the ball (minus the intentional G = 0 exclusion)
    on a non-symmetric lattice."""
    sysp = _nonsymmetric_system()
    eta, precision, pad = 1.0, 1e-8, 1.5
    g_max = 2.0 * pad * np.sqrt(eta * np.log(1.0 / precision))
    mesh = _aft_g_mesh(sysp, eta=eta, precision=precision, pad_factor=pad)
    ball = _ball_keys(sysp, g_max) - {(0.0, 0.0, 0.0)}
    assert ball <= _mesh_keys(mesh)


def test_shifted_g_sphere_preserves_gamma_order_for_reciprocal_labels() -> None:
    """The Γ equivalence class keeps the historical accumulation order."""

    system, _, _, reciprocal, _ = _skew_h2_bloch_fit_fixture()
    gamma = rsgdf_dense_g_mesh(system, 20.0)
    shifted_gamma = _rsgdf_shifted_dense_g_mesh(system, np.zeros(3), 20.0)
    relabelled_gamma = _rsgdf_shifted_dense_g_mesh(
        system,
        reciprocal[:, 0],
        20.0,
    )

    assert np.array_equal(shifted_gamma, gamma)
    assert np.array_equal(relabelled_gamma, gamma)


def test_shifted_g_sphere_boundary_is_relabelling_and_inversion_stable() -> None:
    """Roundoff at the kinetic boundary cannot split one physical shell."""

    lattice = np.array(
        [
            [5.79928919736845, 0.0, 0.0],
            [-0.4555060828230417, 7.042178120136812, 0.0],
            [-0.7296927872617867, -0.7429115539622462, 6.1699533848892445],
        ],
        dtype=float,
    )
    system = vq.PeriodicSystem(3, lattice, [vq.Atom(1, [0.0, 0.0, 0.0])])
    reciprocal = np.asarray(system.reciprocal_lattice(), dtype=float)
    q = reciprocal @ np.array([0.0, -0.5, 0.0])
    q_relabelled = q + reciprocal @ np.array([-1.0, -2.0, 3.0])
    cutoff = 14.745712347058529

    reference = _rsgdf_shifted_dense_g_mesh(system, q, cutoff)
    relabelled = _rsgdf_shifted_dense_g_mesh(system, q_relabelled, cutoff)
    reversed_mesh = _rsgdf_shifted_dense_g_mesh(system, -q, cutoff)
    larger_cutoff = 1.25 * cutoff
    larger = _rsgdf_shifted_dense_g_mesh(system, q_relabelled, larger_cutoff)
    p_max = float(np.sqrt(2.0 * cutoff))
    base_tolerance = _rsgdf_shifted_sphere_boundary_tolerance(
        system,
        p_max,
        q_relabelled,
    )
    shell = larger[
        np.linalg.norm(larger, axis=1) > p_max + base_tolerance
    ]

    def coordinate_order(vectors):
        rounded = np.round(np.asarray(vectors), 12)
        return rounded[np.lexsort((rounded[:, 2], rounded[:, 1], rounded[:, 0]))]

    np.testing.assert_allclose(relabelled, reference, atol=2e-12, rtol=0.0)
    np.testing.assert_allclose(
        coordinate_order(reversed_mesh),
        coordinate_order(-reference),
        atol=2e-12,
        rtol=0.0,
    )
    partition = np.concatenate((relabelled, shell), axis=0)
    assert len(partition) == len(larger)
    np.testing.assert_allclose(
        coordinate_order(partition),
        coordinate_order(larger),
        atol=2e-12,
        rtol=0.0,
    )

    # A second shell lands between the raw-q-dependent slacks of equivalent
    # labels. Canonicalising q before the tolerance calculation keeps it in
    # or out on both paths.
    generic_q = reciprocal @ np.array([0.173, -0.287, 0.119])
    generic_relabelled = generic_q + reciprocal @ np.array([-1.0, -2.0, 3.0])
    generic_cutoff = 4.0985580584936665
    generic_reference = _rsgdf_shifted_dense_g_mesh(
        system,
        generic_q,
        generic_cutoff,
    )
    generic_shifted = _rsgdf_shifted_dense_g_mesh(
        system,
        generic_relabelled,
        generic_cutoff,
    )
    np.testing.assert_allclose(
        generic_shifted,
        generic_reference,
        atol=2e-12,
        rtol=0.0,
    )


def _skew_h2_bloch_fit_fixture():
    lattice = np.array(
        [[6.0, 0.3, 0.2], [0.0, 6.5, 0.4], [0.0, 0.0, 7.0]],
        dtype=float,
    ).T
    system = vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(1, [0.2, 0.3, 0.4]), vq.Atom(1, [1.6, 0.3, 0.4])],
    )
    molecule = system.unit_cell_molecule()
    basis = vq.BasisSet(molecule, "sto-3g")
    auxiliary = make_modrho_aux_basis(
        make_aux_basis_set(molecule, aux_name="def2-svp-jk"),
        molecule,
    )
    reciprocal = np.asarray(system.reciprocal_lattice(), dtype=float)
    options = LatticeSumOptions()
    options.cutoff_bohr = 8.0
    return system, basis, auxiliary, reciprocal, options


def test_bloch_lpq_reuses_only_q_dependent_metric_state():
    """Pairs sharing q reuse M(q), while their ket-resolved fits stay exact."""

    system, basis, auxiliary, reciprocal, options = _skew_h2_bloch_fit_fixture()
    q = (reciprocal[:, 0] + reciprocal[:, 2]) / 3.0
    first_bra = 0.17 * reciprocal[:, 1]
    second_bra = -0.11 * reciprocal[:, 1]
    metric_cache = {}

    build_lpq_bloch_native_fft(
        system,
        basis,
        auxiliary,
        first_bra,
        first_bra + q,
        ke_cutoff=20.0,
        lat_opts=options,
        canonical_auxiliary_basis=True,
        _q_metric_cache=metric_cache,
    )
    cached = build_lpq_bloch_native_fft(
        system,
        basis,
        auxiliary,
        second_bra,
        second_bra + q,
        ke_cutoff=20.0,
        lat_opts=options,
        canonical_auxiliary_basis=True,
        _q_metric_cache=metric_cache,
    )
    fresh = build_lpq_bloch_native_fft(
        system,
        basis,
        auxiliary,
        second_bra,
        second_bra + q,
        ke_cutoff=20.0,
        lat_opts=options,
        canonical_auxiliary_basis=True,
    )

    assert len(metric_cache) == 1
    np.testing.assert_allclose(cached, fresh, atol=2e-12, rtol=2e-12)


@pytest.mark.parametrize("canonical_auxiliary_basis", (False, True))
def test_bloch_lpq_is_invariant_to_ket_reciprocal_relabelling(
    canonical_auxiliary_basis,
):
    """A non-Nyquist ket-only ``G`` shift is one Bloch label, not a new fit.

    The finite builder historically shifted a fixed ``|G|`` ball by the raw
    momentum transfer. ``q`` and ``q + G0`` then differed by the boundary
    crescent of that ball even though their ket cell phases are identical.
    Exercise a non-symmetric lattice so the fractional/cartesian conversion
    cannot pass through a row/column convention accident, and pin both the
    compact SCF fitted operator and canonical auxiliary-AO factor.
    """

    system, basis, auxiliary, reciprocal, options = _skew_h2_bloch_fit_fixture()
    k_bra = 0.17 * reciprocal[:, 1]
    q = (reciprocal[:, 0] + reciprocal[:, 2]) / 3.0
    k_ket = k_bra + q

    reference = build_lpq_bloch_native_fft(
        system,
        basis,
        auxiliary,
        k_bra,
        k_ket,
        ke_cutoff=20.0,
        lat_opts=options,
        canonical_auxiliary_basis=canonical_auxiliary_basis,
    )
    relabelled = build_lpq_bloch_native_fft(
        system,
        basis,
        auxiliary,
        k_bra,
        k_ket + reciprocal[:, 0],
        ke_cutoff=20.0,
        lat_opts=options,
        canonical_auxiliary_basis=canonical_auxiliary_basis,
    )

    if canonical_auxiliary_basis:
        np.testing.assert_allclose(
            relabelled,
            reference,
            atol=2e-12,
            rtol=2e-12,
        )
    else:
        reference_flat = reference.reshape(reference.shape[0], -1)
        relabelled_flat = relabelled.reshape(relabelled.shape[0], -1)
        np.testing.assert_allclose(
            relabelled_flat.conj().T @ relabelled_flat,
            reference_flat.conj().T @ reference_flat,
            atol=2e-12,
            rtol=2e-12,
        )


def test_canonical_bloch_lpq_keeps_nyquist_time_reversal() -> None:
    """The physical shifted sphere is time-reversal covariant at ``+/-G/2``.

    Mapping both Nyquist signs to one half-open representative makes the two
    calls use the same finite shifted ball instead of conjugate balls. The
    canonical auxiliary-AO frame must retain the exact time-reversal relation
    needed by the real ``q/-q`` torus fold.
    """

    system, basis, auxiliary, reciprocal, options = _skew_h2_bloch_fit_fixture()
    k_bra = 0.17 * reciprocal[:, 1]
    q = 0.5 * reciprocal[:, 0]
    plus = build_lpq_bloch_native_fft(
        system,
        basis,
        auxiliary,
        k_bra,
        k_bra + q,
        ke_cutoff=20.0,
        lat_opts=options,
        canonical_auxiliary_basis=True,
    )
    minus = build_lpq_bloch_native_fft(
        system,
        basis,
        auxiliary,
        -k_bra,
        -(k_bra + q),
        ke_cutoff=20.0,
        lat_opts=options,
        canonical_auxiliary_basis=True,
    )

    np.testing.assert_allclose(minus, plus.conj(), atol=2e-12, rtol=2e-12)


def test_shifted_bloch_lpq_tail_completion_matches_big_mesh() -> None:
    """The shifted-sphere base plus tail is the larger shifted sphere."""

    system, basis, auxiliary, reciprocal, options = _skew_h2_bloch_fit_fixture()
    k_bra = 0.17 * reciprocal[:, 1]
    k_ket = k_bra + (reciprocal[:, 0] + reciprocal[:, 2]) / 3.0
    big = build_lpq_bloch_native_fft(
        system,
        basis,
        auxiliary,
        k_bra,
        k_ket,
        ke_cutoff=20.0,
        lat_opts=options,
        canonical_auxiliary_basis=True,
    )
    completed = build_lpq_bloch_native_fft(
        system,
        basis,
        auxiliary,
        k_bra,
        k_ket,
        ke_cutoff=10.0,
        tail_ke_cutoff=20.0,
        lat_opts=options,
        canonical_auxiliary_basis=True,
    )
    big_flat = big.reshape(big.shape[0], -1)
    completed_flat = completed.reshape(completed.shape[0], -1)

    np.testing.assert_allclose(
        completed_flat.conj().T @ completed_flat,
        big_flat.conj().T @ big_flat,
        atol=2e-12,
        rtol=2e-12,
    )


def test_lpq_tail_completion_matches_big_mesh():
    """Lpq(ke=a, tail_ke_cutoff=b) reproduces Lpq(ke=b) exactly (same G
    set), compared through the gauge-invariant fitted-ERI tensor
    W = sum_P L[P,mn] L[P,kl]. Uses the P01 MgO fcc cell at small cutoffs
    to keep the pair-FT cheap."""
    a = 3.97976322
    sysp = _fcc_system(a, [core.Atom(12, [0, 0, 0]), core.Atom(8, [a, a, a])])
    mol = vq.Molecule(list(sysp.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    aux = make_modrho_aux_basis(
        make_aux_basis_set(mol, aux_name="def2-svp-jk"), mol
    )
    lo = LatticeSumOptions()
    lo.cutoff_bohr = 15.0
    progress = _ProgressCapture()

    L_big = build_lpq_native_fft(sysp, basis, aux, ke_cutoff=60.0, lat_opts=lo)
    L_tail = build_lpq_native_fft(
        sysp,
        basis,
        aux,
        ke_cutoff=20.0,
        tail_ke_cutoff=60.0,
        lat_opts=lo,
        progress=progress,
    )
    W_big = np.einsum("Pmn,Pkl->mnkl", L_big, L_big, optimize=True)
    W_tail = np.einsum("Pmn,Pkl->mnkl", L_tail, L_tail, optimize=True)
    assert np.max(np.abs(W_big - W_tail)) < 1e-9
    assert any("RSGDF high-|G| tail chunk" in line for line in progress.lines)


def test_tail_pair_ft_screen_matches_exact_high_g_samples():
    """The production tail screen only removes exponentially dead
    shell/cell work on the high-|G| P01 shell."""
    a = 3.97976322
    sysp = _fcc_system(a, [core.Atom(12, [0, 0, 0]), core.Atom(8, [a, a, a])])
    mol = vq.Molecule(list(sysp.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    lo = LatticeSumOptions()
    lo.cutoff_bohr = 15.0
    cells = core.direct_lattice_cells(sysp, lo.cutoff_bohr)
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    G_all = rsgdf_dense_g_mesh(sysp, 400.0)
    norms = np.linalg.norm(G_all, axis=1)
    G_tail = G_all[norms > np.sqrt(2.0 * 200.0)]
    # A small deterministic slice keeps the exact path cheap while still
    # exercising the same high-|G| regime as the P01 tail completion.
    G_sample = G_tail[:12]

    exact = ao_pair_fourier_transform_bloch(
        basis, G_sample, R_g, k_cart=np.zeros(3), screen_tol=0.0
    )
    screened = ao_pair_fourier_transform_bloch(
        basis, G_sample, R_g, k_cart=np.zeros(3), screen_tol=1.0e-10
    )
    np.testing.assert_allclose(screened, exact, atol=1e-9, rtol=1e-8)


def test_gamma_symmetric_cxx_tail_pair_ft_matches_python_reference(monkeypatch):
    """The Gamma C++ fast path mirrors shell pairs only when the cell list
    is inversion-symmetric; compare it against the pure-Python reference on
    the same high-|G| samples so the transpose convention stays pinned."""
    a = 3.97976322
    sysp = _fcc_system(a, [core.Atom(12, [0, 0, 0]), core.Atom(8, [a, a, a])])
    mol = vq.Molecule(list(sysp.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    lo = LatticeSumOptions()
    lo.cutoff_bohr = 15.0
    cells = core.direct_lattice_cells(sysp, lo.cutoff_bohr)
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)
    G_all = rsgdf_dense_g_mesh(sysp, 400.0)
    norms = np.linalg.norm(G_all, axis=1)
    G_tail = G_all[norms > np.sqrt(2.0 * 200.0)]
    G_sample = G_tail[:16]

    monkeypatch.setenv("VIBEQC_AOPAIR_FT_BACKEND", "cxx")
    fast = ao_pair_fourier_transform_bloch(
        basis, G_sample, R_g, k_cart=np.zeros(3), screen_tol=1.0e-10
    )
    monkeypatch.setenv("VIBEQC_AOPAIR_FT_BACKEND", "python")
    reference = ao_pair_fourier_transform_bloch(
        basis, G_sample, R_g, k_cart=np.zeros(3), screen_tol=0.0
    )

    np.testing.assert_allclose(fast, reference, atol=1e-9, rtol=1e-8)
    np.testing.assert_allclose(fast, fast.transpose(1, 0, 2), atol=1e-12)


def test_gamma_mirror_disabled_on_momentum_shifted_mesh(monkeypatch):
    """k == 0 alone must NOT enable the shell-pair mirror: on a
    momentum-shifted mesh G+q (q not a reciprocal-lattice vector) the swap
    picks up exp(i q.R) phases on the R != 0 inter-cell terms, so
    FT_mu,nu(G+q) != FT_nu,mu(G+q).

    Exactly this call shape reaches the C++ kernel from
    ``build_lpq_bloch_native_fft`` for every (k_bra != Gamma,
    k_ket = Gamma) multi-k exchange pair (``k_cart = k_ket = 0``, mesh
    shifted by ``q = -k_bra``). Mirroring there corrupted the multi-k GDF
    exchange by 2.0e-2 Ha/cell on LiH/sto-3g (2,1,1) — the 2026-07-10
    three-way HF disagreement of
    ``handovers/HANDOVER_AICCM_DIRECT_TORUS.md`` Finding 2. Ultra-diffuse
    Li makes the R != 0 terms large, which is what let the corruption
    through every vacuum-padded gate.
    """
    a = 7.72  # rocksalt LiH primitive fcc — the incident fixture
    sysp = _fcc_system(0.5 * a, [
        core.Atom(3, [0.0, 0.0, 0.0]),
        core.Atom(1, [0.5 * a, 0.5 * a, 0.5 * a]),
    ])
    mol = vq.Molecule(list(sysp.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")
    lo = LatticeSumOptions()
    cells = core.direct_lattice_cells(sysp, lo.cutoff_bohr)
    R_g = np.array([list(c.r_cart) for c in cells], dtype=float)

    # q = half a reciprocal vector — the (2,1,1)-mesh momentum transfer.
    B = 2.0 * np.pi * np.linalg.inv(np.asarray(sysp.lattice, dtype=float)).T
    q = 0.5 * B[0]
    G_all = rsgdf_dense_g_mesh(sysp, 60.0)
    G_sample = G_all[:16] + q[None, :]

    monkeypatch.setenv("VIBEQC_AOPAIR_FT_BACKEND", "cxx")
    fast = ao_pair_fourier_transform_bloch(
        basis, G_sample, R_g, k_cart=np.zeros(3), screen_tol=0.0
    )
    monkeypatch.setenv("VIBEQC_AOPAIR_FT_BACKEND", "python")
    reference = ao_pair_fourier_transform_bloch(
        basis, G_sample, R_g, k_cart=np.zeros(3), screen_tol=0.0
    )

    np.testing.assert_allclose(fast, reference, atol=1e-9, rtol=1e-8)
    # And the object genuinely is not mirror-symmetric here — guards the
    # test itself against a future fixture change that would trivialise it.
    assert np.max(np.abs(reference - reference.transpose(1, 0, 2))) > 1e-3


def test_parity_hold_lifts_with_sufficient_tail():
    """The dense-core Gamma GDF parity hold stays active at the default
    mesh but lifts when the tail completion is sized to the steepest AO
    primitive (tail_ke_cutoff >= 10 x zeta_max; P01 zeta_max = 299.24 ->
    3200 Ha suffices, validated at +1.3 uHa vs PySCF GDF)."""
    from vibeqc.pbc_gdf import _gamma_dense_core_gdf_parity_held

    a = 3.97976322
    sysp = _fcc_system(a, [core.Atom(12, [0, 0, 0]), core.Atom(8, [a, a, a])])
    mol = vq.Molecule(list(sysp.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")

    # No tail: held (the historical P01 hold).
    assert _gamma_dense_core_gdf_parity_held(sysp, "rsgdf") is True
    # Undersized tail: still held (1600 Ha ~ 5.3 x zeta_max -> -1.7 mHa).
    assert (
        _gamma_dense_core_gdf_parity_held(
            sysp, "rsgdf", ao_basis=basis, tail_ke_cutoff=1600.0
        )
        is True
    )
    # Parity-sized tail: lifted (3200 Ha ~ 10.7 x zeta_max -> ~1 uHa).
    assert (
        _gamma_dense_core_gdf_parity_held(
            sysp, "rsgdf", ao_basis=basis, tail_ke_cutoff=3200.0
        )
        is False
    )
    # mdf has no tail plumbing: tail does not lift its hold.
    assert (
        _gamma_dense_core_gdf_parity_held(
            sysp, "mdf", ao_basis=basis, tail_ke_cutoff=3200.0
        )
        is True
    )


def test_dense_core_rsgdf_auto_tail_lifts_default_hold():
    from vibeqc.pbc_gdf import (
        _auto_rsgdf_tail_ke_cutoff,
        _gamma_dense_core_gdf_parity_held,
    )

    a = 3.97976322
    sysp = _fcc_system(a, [core.Atom(12, [0, 0, 0]), core.Atom(8, [a, a, a])])
    mol = vq.Molecule(list(sysp.unit_cell), 0, 1)
    basis = vq.BasisSet(mol, "sto-3g")

    auto_tail = _auto_rsgdf_tail_ke_cutoff(sysp, "rsgdf", basis, None)
    assert auto_tail is not None
    assert auto_tail > 3200.0
    assert (
        _gamma_dense_core_gdf_parity_held(
            sysp, "rsgdf", ao_basis=basis, tail_ke_cutoff=auto_tail
        )
        is False
    )
    assert _auto_rsgdf_tail_ke_cutoff(sysp, "rsgdf", basis, 0.0) == 0.0


def test_sparse_tight_core_rsgdf_auto_tail_lifts_default_hold():
    """Sparse molecular-limit boxes can still need the high-G tail.

    The open-shell OH/STO-3G Gamma UKS validation row used a 20 bohr box:
    not a dense ionic solid by volume, but the O 1s STO-3G primitive still
    needs reciprocal support above the 200 Ha base mesh. Before the classifier
    became basis-driven this returned ``None`` and the user-facing GDF route
    under-bound the Coulomb energy by ~0.11 Ha vs PySCF.
    """
    from vibeqc.pbc_gdf import (
        _RSGDF_PARITY_TAIL_RATIO,
        _auto_rsgdf_tail_ke_cutoff,
        _gamma_dense_core_gdf_parity_held,
    )

    sysp = vq.PeriodicSystem(
        3,
        np.diag([20.0, 20.0, 20.0]),
        [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [1.85, 0.0, 0.0])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    zeta_max = max(float(max(shell.exponents)) for shell in basis.shells())

    assert _gamma_dense_core_gdf_parity_held(
        sysp,
        "rsgdf",
        ao_basis=basis,
        tail_ke_cutoff=None,
        rsgdf_ke_cutoff=200.0,
    )
    auto_tail = _auto_rsgdf_tail_ke_cutoff(
        sysp,
        "rsgdf",
        basis,
        None,
        rsgdf_ke_cutoff=200.0,
    )
    assert auto_tail is not None
    assert auto_tail > _RSGDF_PARITY_TAIL_RATIO * zeta_max
    assert (
        _gamma_dense_core_gdf_parity_held(
            sysp,
            "rsgdf",
            ao_basis=basis,
            tail_ke_cutoff=auto_tail,
            rsgdf_ke_cutoff=200.0,
        )
        is False
    )
    assert _auto_rsgdf_tail_ke_cutoff(sysp, "rsgdf", basis, 0.0) == 0.0
