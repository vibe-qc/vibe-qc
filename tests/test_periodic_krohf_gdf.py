"""Restricted-open-shell multi-k GDF driver: ``run_krohf_periodic_gdf``.

Validation strategy (mirroring ``tests/test_periodic_kuhf_gdf.py``):

  * **closed-shell limit** -- ROHF at multiplicity 1 reproduces the
    trusted ``run_krhf_periodic_gdf`` (which carries µHa PySCF KRHF
    parity) to machine precision. That gates J, the per-spin K, the
    exxdiv Madelung shift, the energy expression and the SCF at once.
  * **single-open-shell limit** -- with one unpaired electron (or a
    fully polarised shell) ROHF and UHF describe the *same*
    determinant, so ``run_kuhf_periodic_gdf`` is an exact partner.
  * **external anchor** -- Li/STO-3G in a 10-bohr box at Gamma sits
    +2.86e-5 Ha from out-of-process PySCF 2.13.1 ``KROHF/GDF``
    (``exxdiv='ewald'``, ``def2-svp-jkfit``) at the default rsgdf mesh,
    and the residual collapses with the mesh (see the pin below), which
    is the signature of shared fit truncation rather than a convention
    error.
  * **spin** -- ``<S^2> = S(S+1)`` exactly, by construction.
  * **factored exchange** -- the occupied-index builder is the same
    operator as the density-form one, and damping (which reuses the
    previous bare K by linearity) reaches the same fixed point.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.periodic_k_gdf import (
    _build_k_from_lpq_factors,
    _k_from_densities_dense,
    run_krhf_periodic_gdf,
    run_kuhf_periodic_gdf,
)
from vibeqc.periodic_rohf_gdf import run_krohf_periodic_gdf

_COMMON = dict(
    aux_basis="def2-svp-jk",
    gdf_method="rsgdf",
    rsgdf_ke_cutoff=200.0,
    progress=False,
)


def _cell(atoms, lattice, *, mult=1, charge=0, basis="sto-3g"):
    system = vq.PeriodicSystem(3, np.asarray(lattice, dtype=float), atoms)
    system.multiplicity = int(mult)
    system.charge = int(charge)
    return system, vq.BasisSet(system.unit_cell_molecule(), basis)


def _h2_box(box_bohr: float = 12.0, sep_bohr: float = 1.4, **kw):
    half = 0.5 * sep_bohr
    return _cell(
        [vq.Atom(1, [0, 0, -half]), vq.Atom(1, [0, 0, half])],
        np.diag([box_bohr] * 3),
        **kw,
    )


def _opts(*, damping=0.0, diis=True, cutoff=30.0, tol=1e-10):
    o = vq.PeriodicRHFOptions()
    o.initial_guess = vq.InitialGuess.HCORE
    o.use_diis = diis
    o.damping = damping
    o.max_iter = 200
    o.conv_tol_energy = tol
    o.lattice_opts.cutoff_bohr = cutoff
    o.lattice_opts.nuclear_cutoff_bohr = cutoff
    return o


@pytest.mark.parametrize(
    "guess",
    [vq.InitialGuess.FRAGMO],
    ids=lambda guess: guess.name.lower(),
)
def test_krohf_unsupported_source_fails_closed_before_gdf_setup(
    monkeypatch,
    guess,
):
    """Direct callers must not receive Hcore under another guess label."""
    system, basis = _h2_box(mult=3)
    options = _opts()
    options.initial_guess = guess

    def unexpected_setup(*_args, **_kwargs):
        pytest.fail("GDF setup ran before the unsupported selector failed")

    monkeypatch.setattr(
        "vibeqc.periodic_rohf_gdf.make_aux_basis_set",
        unexpected_setup,
    )
    with pytest.raises(NotImplementedError, match=f"initial_guess={guess.name}"):
        run_krohf_periodic_gdf(
            system,
            basis,
            kmesh=(1, 1, 1),
            options=options,
            **_COMMON,
        )


# ---------------------------------------------------------------------
# Closed-shell and single-open-shell limits
# ---------------------------------------------------------------------


def test_krohf_m1_equals_krhf():
    """ROHF at multiplicity 1 IS RHF: same energy to machine precision.

    The closed-shell-limit gate. ``run_krhf_periodic_gdf`` carries µHa
    PySCF KRHF parity, so agreement here validates the whole multi-k
    machinery of the new driver -- J from the BZ-summed density, the
    per-spin fitted K, the BvK-supercell exxdiv shift and the energy
    expression -- against a trusted reference.
    """
    system, basis = _h2_box()
    ref = run_krhf_periodic_gdf(
        system, basis, kmesh=(2, 1, 1), options=_opts(), **_COMMON
    )
    got = run_krohf_periodic_gdf(
        system, basis, kmesh=(2, 1, 1), options=_opts(), **_COMMON
    )
    assert ref.converged and got.converged
    assert abs(got.energy - ref.energy) < 1e-10, (
        f"KROHF(M=1) {got.energy!r} != KRHF {ref.energy!r}"
    )
    assert got.s_squared == pytest.approx(0.0, abs=1e-12)
    assert isinstance(got, vq.PeriodicKROHFGDFResult)
    # Doubly occupied ground state, one occupied orbital per k.
    for occ in got.mo_occupations:
        assert np.count_nonzero(occ == 2.0) == 1
        assert np.count_nonzero(occ == 1.0) == 0


@pytest.mark.parametrize(
    "mult, charge, s2", [(2, 1, 0.75), (3, 0, 2.0)]
)
def test_krohf_matches_kuhf_on_a_single_open_shell(mult, charge, s2):
    """One unpaired electron / a fully polarised shell: ROHF == UHF.

    With no doubly occupied shell left for the two spins to relax
    differently, the restricted-open and unrestricted determinants
    coincide, so the PySCF-pinned ``run_kuhf_periodic_gdf`` is an exact
    partner rather than merely an upper bound. Any drift here is a bug
    in the ROHF Fock assembly, not a variational difference.
    """
    system, basis = _h2_box(mult=mult, charge=charge)
    ro = run_krohf_periodic_gdf(
        system, basis, kmesh=(2, 1, 1), options=_opts(), **_COMMON
    )
    uhf = run_kuhf_periodic_gdf(
        system, basis, kmesh=(2, 1, 1), options=_opts(), **_COMMON
    )
    assert ro.converged and uhf.converged
    assert abs(ro.energy - uhf.energy) < 1e-10, (
        f"KROHF {ro.energy!r} != KUHF {uhf.energy!r}"
    )
    assert ro.s_squared == pytest.approx(s2, abs=1e-12)
    assert ro.s_squared == pytest.approx(ro.s_squared_ideal, abs=1e-12)


def test_krohf_gamma_is_the_one_one_one_mesh():
    """Gamma is reached through the same code path as any other mesh."""
    system, basis = _h2_box(mult=3)
    got = run_krohf_periodic_gdf(
        system, basis, kmesh=(1, 1, 1), options=_opts(), **_COMMON
    )
    assert got.converged
    assert got.kpoints_cart.shape == (1, 3)
    assert np.allclose(got.kpoints_cart[0], 0.0)
    assert got.backend == "native-multi-k-gdf-rohf"


# ---------------------------------------------------------------------
# External anchor
# ---------------------------------------------------------------------


def test_krohf_li_box_matches_pyscf_krohf_gdf():
    """Li(2S) in a 10-bohr box vs out-of-process PySCF 2.13.1 KROHF/GDF.

    Reference (``pyscf.pbc.scf.KROHF(cell, kpts, exxdiv='ewald')
    .density_fit(auxbasis='def2-svp-jkfit')``, ``cell.unit='B'``,
    ``mf.nelec=(2, 1)``): ``-7.387277021304971 Ha``.

    The default rsgdf mesh sits +2.86e-5 Ha from it. That residual is the
    shared fit-truncation ladder, not a convention error -- measured on
    this exact cell it collapses monotonically with the mesh:
    ``ke=200 -> +2.860e-05``, ``ke=400 -> +3.987e-06``,
    ``ke=800 (cutoff 40) -> +2.208e-07`` Ha. The tolerance below pins the
    default rung; the collapse is recorded here rather than run in the
    fast lane because the ke=800 leg is a fine-mesh FT.
    """
    system, basis = _cell(
        [vq.Atom(3, [0, 0, 0])], np.diag([10.0] * 3), mult=2
    )
    got = run_krohf_periodic_gdf(
        system, basis, kmesh=(1, 1, 1), options=_opts(), **_COMMON
    )
    assert got.converged
    e_pyscf = -7.387277021304971
    assert got.energy - e_pyscf == pytest.approx(2.860e-5, abs=2e-6), (
        f"KROHF {got.energy!r} moved off the recorded PySCF residual"
    )
    assert got.s_squared == pytest.approx(0.75, abs=1e-12)
    assert list(got.mo_occupations[0][:2]) == [2.0, 1.0]


# ---------------------------------------------------------------------
# The factored (occupied-index) exchange build
# ---------------------------------------------------------------------


def test_factored_exchange_equals_the_density_form():
    """``_build_k_from_lpq_factors`` is the density-form builder's operator.

    Same contraction, reassociated so the cderi meets the occupied MO
    block (``n_occ`` columns) instead of the ``nbf x nbf`` density. The
    multi-factor form (a convex mix of two densities) is checked too,
    since that is what expresses a damped iterate.

    The reference is deliberately ``_k_from_densities_dense``, the
    historical ``2 naux nbf^3`` contraction, and NOT
    ``_build_k_from_lpq_cache``: since 2026-08-02 that wrapper is itself
    factored, so comparing against it would compare the factored path
    with itself and pass vacuously.
    """
    rng = np.random.default_rng(20260801)
    n_k, naux, nbf, n_occ = 3, 12, 7, 2
    lpq = {
        (i, j): rng.standard_normal((naux, nbf, nbf))
        + 1j * rng.standard_normal((naux, nbf, nbf))
        for i in range(n_k)
        for j in range(n_k)
    }
    W = [
        rng.standard_normal((nbf, n_occ))
        + 1j * rng.standard_normal((nbf, n_occ))
        for _ in range(n_k)
    ]
    W2 = [
        rng.standard_normal((nbf, n_occ))
        + 1j * rng.standard_normal((nbf, n_occ))
        for _ in range(n_k)
    ]
    weights = np.full(n_k, 1.0 / n_k)

    D = [w @ w.conj().T for w in W]
    ref = _k_from_densities_dense(
        lpq, D, weights, range(n_k), nbasis=nbf
    )
    got = _build_k_from_lpq_factors(lpq, [[w] for w in W], weights, nbasis=nbf)
    scale = max(float(np.max(np.abs(K))) for K in ref)
    for a, b in zip(ref, got):
        assert np.max(np.abs(a - b)) < 1e-12 * scale
        # Gram sum by construction, so exactly Hermitian.
        assert np.max(np.abs(b - b.conj().T)) < 1e-13 * scale

    mix = 0.3
    D_mix = [
        mix * w @ w.conj().T + (1.0 - mix) * w2 @ w2.conj().T
        for w, w2 in zip(W, W2)
    ]
    ref_mix = _k_from_densities_dense(
        lpq, D_mix, weights, range(n_k), nbasis=nbf
    )
    got_mix = _build_k_from_lpq_factors(
        lpq,
        [
            [np.sqrt(mix) * w, np.sqrt(1.0 - mix) * w2]
            for w, w2 in zip(W, W2)
        ],
        weights,
        nbasis=nbf,
    )
    for a, b in zip(ref_mix, got_mix):
        assert np.max(np.abs(a - b)) < 1e-12 * scale


def test_factored_exchange_rejects_mismatched_weights():
    lpq = {(0, 0): np.zeros((2, 3, 3), dtype=complex)}
    with pytest.raises(ValueError, match="weights length must match"):
        _build_k_from_lpq_factors(
            lpq, [[np.zeros((3, 1))]], [0.5, 0.5], nbasis=3
        )


def test_krohf_damping_reaches_the_same_fixed_point():
    """Damped iterates reuse the previous bare K by linearity.

    ``K`` is linear in the density, so the damped exchange is the damped
    combination of the two builds -- which lets every build stay at
    ``n_occ`` columns instead of accumulating factor rank. If that
    recursion were wrong the damped SCF would converge somewhere else.
    """
    system, basis = _cell(
        [vq.Atom(3, [0, 0, 0]), vq.Atom(1, [0, 0, 3.0])],
        np.diag([6.0, 12.0, 12.0]),
        mult=2,
        charge=1,
    )
    energies = []
    for damping in (0.0, 0.4, 0.7):
        got = run_krohf_periodic_gdf(
            system,
            basis,
            kmesh=(3, 1, 1),
            options=_opts(damping=damping, diis=False, cutoff=24.0),
            **_COMMON,
        )
        assert got.converged, f"damping={damping} did not converge"
        # The damping branch must actually have been exercised.
        assert got.n_iter > 2
        energies.append(got.energy)
    spread = max(energies) - min(energies)
    assert spread < 1e-9, f"damped fixed points differ by {spread:.3e} Ha"


# ---------------------------------------------------------------------
# Fail-closed surface
# ---------------------------------------------------------------------


def test_krohf_rejects_smearing():
    system, basis = _h2_box(mult=3)
    opts = _opts()
    opts.smearing_temperature = 0.01
    with pytest.raises(NotImplementedError, match="smearing_temperature"):
        run_krohf_periodic_gdf(
            system, basis, kmesh=(1, 1, 1), options=opts, **_COMMON
        )


def test_krohf_rejects_a_functional():
    """A functional selects ROKS, which is a separate increment."""
    system, basis = _h2_box(mult=3)
    opts = vq.PeriodicKSOptions()
    opts.functional = "pbe"
    opts.lattice_opts.cutoff_bohr = 30.0
    with pytest.raises(NotImplementedError, match="ROKS"):
        run_krohf_periodic_gdf(
            system, basis, kmesh=(1, 1, 1), options=opts, **_COMMON
        )


def test_krohf_rejects_unknown_gdf_method():
    system, basis = _h2_box(mult=3)
    with pytest.raises(ValueError, match="gdf_method"):
        run_krohf_periodic_gdf(
            system,
            basis,
            kmesh=(1, 1, 1),
            options=_opts(),
            aux_basis="def2-svp-jk",
            gdf_method="nope",
            progress=False,
        )


def test_krohf_rejects_the_tail_knob_off_rsgdf():
    system, basis = _h2_box(mult=3)
    with pytest.raises(NotImplementedError, match="rsgdf_tail_ke_cutoff"):
        run_krohf_periodic_gdf(
            system,
            basis,
            kmesh=(1, 1, 1),
            options=_opts(),
            aux_basis="def2-svp-jk",
            gdf_method="compcell",
            rsgdf_tail_ke_cutoff=800.0,
            progress=False,
        )


def test_krohf_rejects_an_impossible_multiplicity():
    system, basis = _h2_box(mult=2)  # 2 electrons cannot be a doublet
    with pytest.raises(ValueError, match="integer a/b occupations"):
        run_krohf_periodic_gdf(
            system, basis, kmesh=(1, 1, 1), options=_opts(), **_COMMON
        )


def test_krohf_rejects_a_2d_slab():
    system = vq.PeriodicSystem(
        2, np.diag([6.0, 6.0, 20.0]), [vq.Atom(1, [0, 0, 0])]
    )
    system.multiplicity = 2
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    with pytest.raises((NotImplementedError, ValueError)):
        run_krohf_periodic_gdf(
            system, basis, kmesh=(2, 2, 1), options=_opts(), **_COMMON
        )


@pytest.mark.parametrize("guess", ["AUTO", "SAD", "SAP", "HUECKEL", "MINAO", "PATOM"])
@pytest.mark.parametrize("mesh", [(1, 1, 1), (2, 1, 1)])
def test_krohf_constructed_guess_converges_in_closed_shell_basin(guess, mesh):
    from vibeqc._initial_guess import coerce_initial_guess

    system, basis = _h2_box()
    options = _opts(cutoff=12)
    options.initial_guess = coerce_initial_guess(guess)
    result = run_krohf_periodic_gdf(
        system, basis, kmesh=mesh, options=options, **_COMMON
    )
    reference = run_krohf_periodic_gdf(
        system, basis, kmesh=mesh, options=_opts(cutoff=12), **_COMMON
    )
    assert result.converged and reference.converged
    assert result.energy == pytest.approx(reference.energy, abs=2e-8)


@pytest.mark.parametrize("guess", [vq.InitialGuess.SAD, vq.InitialGuess.PATOM])
@pytest.mark.parametrize("mesh", [(1, 1, 1), (3, 1, 1)])
def test_open_shell_guess_density_and_exchange_factors_agree(monkeypatch, guess, mesh):
    """First ROHF/GDF Fock sees the selected spin densities, including PATOM."""
    import vibeqc.periodic_rohf_gdf as driver

    system, basis = _cell([vq.Atom(3, [0, 0, 0])], np.eye(3) * 10, mult=2)
    options = _opts(cutoff=15)
    options.initial_guess = guess
    options.max_iter = 1
    seen = []
    build = driver._build_k_from_lpq_factors

    def exchange(lpq, factors, weights, **kwargs):
        seen.append([sum(w @ w.conj().T for w in blocks) for blocks in factors])
        return build(lpq, factors, weights, **kwargs)

    monkeypatch.setattr(driver, "_build_k_from_lpq_factors", exchange)
    result = run_krohf_periodic_gdf(system, basis, kmesh=mesh, options=options, **_COMMON)
    assert len(seen) == 2
    for factors, densities, target in zip(
        seen, (result.density_alpha, result.density_beta), (2, 1)
    ):
        for actual, density in zip(factors, densities):
            np.testing.assert_allclose(actual, density, atol=1e-11)
        count = sum(w * np.trace(d @ s) for w, d, s in zip(
            result.kpoint_weights, densities, result.overlap
        ))
        assert count == pytest.approx(target, abs=1e-11)
    # The accepted non-idempotent guess remains above; the spectral
    # fields now describe its effective Fock, not old Hcore orbitals.
    assert len(result.mo_coeffs) == len(result.kpoint_weights)
    for c, eps, fock, overlap in zip(
        result.mo_coeffs, result.mo_energies, result.fock, result.overlap
    ):
        np.testing.assert_allclose(c.conj().T @ overlap @ c, np.eye(c.shape[1]), atol=1e-10)
        np.testing.assert_allclose(c.conj().T @ fock @ c, np.diag(eps), atol=1e-10)
