"""Cauchy-Schwarz screening of the three-centre GDF fit.

Pins the ``fit_screen_threshold`` option of
:func:`vibeqc.aux_basis.build_lpq_bloch_native_fft` and its driver
plumbing (``run_krhf_periodic_gdf`` / ``run_krks_periodic_gdf`` /
``run_kuhf_periodic_gdf``), per
``handovers/HANDOVER_GDF_FIT_SCREENING.md``:

* Energy invariance: screened == unscreened to < 1e-8 Ha/cell at
  threshold 1e-10 on the H₂ anchor (c-diamond + MgO run in the slow
  gate below).
* The screen actually drops pairs on a system with well-separated
  units, and the dropped pairs' Lpq entries are exactly zero.
* No silent truncation: the kept/total pair counts are logged.
* The ``fit_pair_list`` seam (symmetry-pair-reduction composition,
  ``handovers/HANDOVER_SYMMETRY_PAIR_REDUCTION.md``): a full pair list
  is bit-identical to no list; a restricted list zeroes the complement.
* Loud NotImplementedError/ValueError on unsupported combinations --
  never a silent no-op.
* M2 G-chunked accumulation: small-chunk sweeps reproduce the
  one-pass build (canonical-aux comparison) and log their chunk
  count.
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.aux_basis import (
    _rsgdf_dense_pair_ft,
    build_lpq_bloch_native_fft,
    build_lpq_native_fft,
    make_aux_basis_set,
    make_modrho_aux_basis,
)
from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf

_A = 1.0 / 0.529177210903


def _h2_cubic_box(box_bohr: float = 12.0):
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * float(box_bohr),
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _two_h2_units_box(separation_bohr: float = 10.0, box_bohr: float = 20.0):
    """Two H₂ units far apart in one cell -- inter-unit AO pairs are
    overlap-dead, so the Schwarz screen has something real to drop.

    The second unit sits at the body-diagonal half (s, s, s): with a
    cubic box of side 2s every one of the eight lattice images is
    equidistant, so the minimum-image separation is the maximal s√3
    (17.3 bohr at the default) -- the diffuse sto-3g H pair bound is
    then ~1e-11, comfortably below a 1e-8 screen, while at (s, s, 0)
    the 14.1-bohr minimum image only reaches ~1e-7 (measured; the
    conservative bound direction)."""
    s = float(separation_bohr)
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * float(box_bohr),
        [
            vq.Atom(1, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 0.0, 1.4]),
            vq.Atom(1, [s, s, s]),
            vq.Atom(1, [s, s, s + 1.4]),
        ],
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _c_diamond():
    """fcc diamond primitive cell (a = 3.567 Å), 2 C -- the tight-cell
    control of the handover's c-diamond ladder."""
    a = 3.567 * _A
    lattice = (a / 2.0) * np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    atoms = [
        vq.Atom(6, [0.0, 0.0, 0.0]),
        vq.Atom(6, [a / 4.0, a / 4.0, a / 4.0]),
    ]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _mgo():
    """Rocksalt MgO primitive cell (a = 4.21 Å), Mg + O -- the ionic
    control (same fixture as tests/test_bipole_fock_ewald.py)."""
    a = 4.21 * _A
    lattice = (a / 2.0) * np.array(
        [
            [0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ]
    )
    atoms = [
        vq.Atom(12, [0.0, 0.0, 0.0]),
        vq.Atom(8, [a / 2.0, a / 2.0, a / 2.0]),
    ]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _rhf_opts(max_iter: int = 30) -> vq.PeriodicRHFOptions:
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.diis_start_iter = 2
    opts.diis_subspace_size = 8
    opts.max_iter = int(max_iter)
    opts.conv_tol_energy = 1e-8
    opts.lattice_opts.cutoff_bohr = 14.0
    opts.lattice_opts.nuclear_cutoff_bohr = 16.0
    return opts


def _bloch_fit(
    system,
    basis,
    *,
    threshold=0.0,
    pair_list=None,
    progress=None,
    canonical=False,
    ke_cutoff=100.0,
    **extra,
):
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk", drop_eta=0.0)
    aux_modrho = make_modrho_aux_basis(aux, mol)
    k0 = np.zeros(3)
    return build_lpq_bloch_native_fft(
        system,
        basis,
        aux_modrho,
        k0,
        k0,
        ke_cutoff=float(ke_cutoff),
        fit_screen_threshold=threshold,
        fit_pair_list=pair_list,
        progress=progress,
        canonical_auxiliary_basis=canonical,
        **extra,
    )


class _RecordingProgress:
    def __init__(self):
        self.lines: list[str] = []

    def info(self, message: str) -> None:
        self.lines.append(str(message))


# =====================================================================
#                      Builder-level invariance
# =====================================================================


def test_threshold_zero_is_bit_identical_to_default():
    system, basis = _h2_cubic_box()
    L_default = _bloch_fit(system, basis)
    L_zero = _bloch_fit(system, basis, threshold=0.0)
    np.testing.assert_array_equal(L_default, L_zero)


def test_screened_fit_drops_pairs_and_zeroes_them():
    """Two H₂ units 17.3 bohr apart (minimum image): the inter-unit AO
    pairs' Schwarz bound is ~1e-11, so a 1e-8 screen must drop them,
    zero exactly their Lpq entries, and leave every ERI reconstruction
    element within screening accuracy."""
    system, basis = _two_h2_units_box()
    prog = _RecordingProgress()
    L_exact = _bloch_fit(system, basis)
    L_scr = _bloch_fit(system, basis, threshold=1e-8, progress=prog)

    # The screen log line is emitted (no silent truncation) ...
    screen_lines = [ln for ln in prog.lines if "GDF fit screen" in ln]
    assert len(screen_lines) == 1
    # ... and something was actually dropped on this geometry.
    n_orb = basis.nbasis
    dropped = np.all(np.abs(L_scr) == 0.0, axis=0)
    assert int(np.count_nonzero(dropped)) > 0, (
        "well-separated inter-unit pairs should be screened at 1e-8; "
        f"log: {screen_lines}"
    )
    # Intra-unit pairs (large) must never be dropped.
    assert not dropped[0, 1] and not dropped[0, 0]

    # ERI reconstruction W = Σ_P L L† agrees within screening accuracy
    # (the dropped pairs' true T elements sit below the 1e-8 Schwarz
    # bound; the whitened-fit propagation keeps the W error within a
    # couple of orders of the threshold).
    W_exact = np.einsum("Pmn,Prs->mnrs", L_exact, np.conj(L_exact))
    W_scr = np.einsum("Pmn,Prs->mnrs", L_scr, np.conj(L_scr))
    assert float(np.max(np.abs(W_exact - W_scr))) < 1e-6
    assert L_scr.shape[1:] == (n_orb, n_orb)


def test_full_pair_list_is_bit_identical_to_none():
    system, basis = _two_h2_units_box()
    n_sh = basis.nshells
    full = np.array(
        [(i, j) for i in range(n_sh) for j in range(n_sh)], dtype=int
    )
    L_none = _bloch_fit(system, basis)
    L_full = _bloch_fit(system, basis, pair_list=full)
    np.testing.assert_array_equal(L_none, L_full)


def test_restricted_pair_list_zeroes_the_complement():
    system, basis = _two_h2_units_box()
    n_sh = basis.nshells
    # Keep only shell-diagonal pairs.
    diag = np.array([(i, i) for i in range(n_sh)], dtype=int)
    L = _bloch_fit(system, basis, pair_list=diag)
    # sto-3g H: one shell per atom, one AO per shell -- the AO-pair
    # mask is exactly the shell-pair mask.
    off_diag = ~np.eye(basis.nbasis, dtype=bool)
    assert np.all(L[:, off_diag] == 0.0)
    assert np.any(L[:, np.eye(basis.nbasis, dtype=bool)] != 0.0)


# =====================================================================
#                      M2: G-chunked accumulation
# =====================================================================


def test_chunked_accumulation_matches_single_pass():
    """M2: sweeping the base mesh in small G-chunks must reproduce the
    coarse-chunk build to numerical noise -- compared in the canonical
    auxiliary basis (chunk-order round-off perturbs the metric
    eigenvectors, so the compact eigenmode factor is phase-unstable;
    the canonical rotation is basis-stable). Screened and unscreened;
    odd prime chunk size to exercise ragged chunk boundaries. The
    chunk-count log line must appear (no silent memory caps)."""
    system, basis = _two_h2_units_box()
    for thr in (0.0, 1e-8):
        L_ref = _bloch_fit(system, basis, threshold=thr, canonical=True)
        prog = _RecordingProgress()
        L_chunk = _bloch_fit(
            system,
            basis,
            threshold=thr,
            canonical=True,
            tail_chunk_g=10007,
            progress=prog,
        )
        assert L_ref.shape == L_chunk.shape
        np.testing.assert_allclose(L_chunk, L_ref, atol=2e-8, rtol=0.0)
        chunk_lines = [ln for ln in prog.lines if "GDF fit build" in ln]
        assert len(chunk_lines) == 1 and "chunks" in chunk_lines[0], (
            f"chunked build must log its chunking; got {prog.lines}"
        )


# =====================================================================
#            Γ-only builder (build_lpq_native_fft) delegation
# =====================================================================


def _gamma_fit(system, basis, *, ke_cutoff=100.0, **extra):
    mol = system.unit_cell_molecule()
    aux = make_aux_basis_set(mol, aux_name="def2-svp-jk", drop_eta=0.0)
    aux_modrho = make_modrho_aux_basis(aux, mol)
    return build_lpq_native_fft(
        system,
        basis,
        aux_modrho,
        ke_cutoff=float(ke_cutoff),
        **extra,
    )


def _four_center(L: np.ndarray) -> np.ndarray:
    """The fit-invariant W = Σ_P L[P] ⊗ L[P] -- basis- and phase-stable
    where the eigenmode factor itself is not."""
    Lf = np.asarray(L).reshape(L.shape[0], -1)
    return np.real(Lf.T.conj() @ Lf)


def test_gamma_screened_fit_matches_dense_four_center():
    """A negligible threshold on the Γ builder (Bloch-delegated path)
    reproduces the dense unscreened Γ build's fitted four-center."""
    system, basis = _h2_cubic_box()
    W_dense = _four_center(_gamma_fit(system, basis))
    W_scr = _four_center(_gamma_fit(system, basis, fit_screen_threshold=1e-14))
    np.testing.assert_allclose(W_scr, W_dense, atol=1e-8, rtol=0.0)


def test_gamma_screened_fit_drops_pairs_and_zeroes_them():
    """On the two-separated-units box the Γ screened build must drop
    the inter-unit AO pairs (their Lpq entries exactly zero) while the
    intra-unit four-center matches the dense build."""
    system, basis = _two_h2_units_box()
    L_scr = _gamma_fit(system, basis, fit_screen_threshold=1e-8)
    # AOs 0-1 = unit A, AOs 2-3 = unit B (sto-3g H = 1 AO per atom).
    inter = np.abs(L_scr[:, :2, 2:])
    assert np.all(inter == 0.0), (
        f"inter-unit Lpq entries not zeroed: max {inter.max():.3e}"
    )
    W_dense = _four_center(_gamma_fit(system, basis))
    W_scr = _four_center(L_scr)
    intra = (slice(None, 2), slice(None, 2))
    n = basis.nbasis
    W_dense_intra = W_dense.reshape(n, n, n, n)[intra + intra]
    W_scr_intra = W_scr.reshape(n, n, n, n)[intra + intra]
    np.testing.assert_allclose(W_scr_intra, W_dense_intra, atol=1e-7, rtol=0.0)


def test_gamma_screen_conflicts_with_shared_bundle():
    """Memory-lean screened mode + precomputed dense bundle is a
    contradiction -- must raise, never silently ignore either."""
    system, basis = _h2_cubic_box()
    lat_opts = vq.LatticeSumOptions()
    bundle = _rsgdf_dense_pair_ft(basis, system, 100.0, lat_opts)
    with pytest.raises(ValueError, match="pair_ft_shared"):
        _gamma_fit(
            system,
            basis,
            fit_screen_threshold=1e-10,
            pair_ft_shared=bundle,
            lat_opts=lat_opts,
        )


def test_gamma_screen_conflicts_with_tail_pair_ft_screen():
    system, basis = _h2_cubic_box()
    with pytest.raises(NotImplementedError, match="tail_pair_ft_screen"):
        _gamma_fit(
            system,
            basis,
            fit_screen_threshold=1e-10,
            tail_ke_cutoff=150.0,
            tail_pair_ft_screen=1e-12,
        )


def _h2_line_box(n_units: int, spacing_bohr: float = 20.0):
    """``n_units`` H₂ units on a z-line, ``spacing_bohr`` apart, in a
    box that grows linearly with the unit count -- the local-system
    ladder for the significant-pair-count scaling gate. Neighbouring
    units (and their z-periodic images) sit >= 18.6 bohr apart, where
    the sto-3g H pair bound is far below a 1e-8 screen (the measured
    17.3-bohr bound is already ~1e-11, see ``_two_h2_units_box``)."""
    s = float(spacing_bohr)
    atoms = []
    for u in range(int(n_units)):
        z = u * s
        atoms.append(vq.Atom(1, [0.0, 0.0, z]))
        atoms.append(vq.Atom(1, [0.0, 0.0, z + 1.4]))
    system = vq.PeriodicSystem(
        3, np.diag([s, s, s * int(n_units)]), atoms
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _kept_ao_pairs(lines: list[str]) -> tuple[int, int]:
    """Parse (kept, total) AO-pair counts from the fit-screen log."""
    import re

    for ln in lines:
        m = re.search(r"\((\d+)/(\d+) AO pairs\)", ln)
        if "GDF fit screen" in ln and m:
            return int(m.group(1)), int(m.group(2))
    raise AssertionError(f"no fit-screen log line found in {lines}")


def test_significant_pair_count_scales_subquadratically():
    """Gate 3: for a local system (well-separated H₂ units) the
    significant-pair count under the Schwarz screen grows ~linearly
    with cluster size while the total pair count grows quadratically.
    Asserted loudly from the no-silent-truncation log line."""
    kept = {}
    for n in (1, 2, 3):
        system, basis = _h2_line_box(n)
        prog = _RecordingProgress()
        # ke_cutoff 40: the pair-count gate needs the screen's verdict,
        # not a converged fit -- the n=3 box is 24000 bohr³ and the
        # default 100-Ha mesh alone costs a minute of pure FT.
        _bloch_fit(system, basis, threshold=1e-8, progress=prog, ke_cutoff=40.0)
        n_kept, n_total = _kept_ao_pairs(prog.lines)
        assert n_total == (2 * n) ** 2
        kept[n] = n_kept
    # Intra-unit pairs must survive (4 AO pairs per unit)...
    assert kept[1] >= 4
    # ...and the kept count must stay linear-with-slack in the unit
    # count: quadratic growth would give kept[n] ~ n^2 * kept[1].
    for n in (2, 3):
        assert kept[n] <= 2 * n * kept[1], (
            f"significant-pair count not sub-quadratic: kept={kept} "
            f"(unit count {n}: {kept[n]} > 2*{n}*{kept[1]})"
        )


# =====================================================================
#                      Driver-level invariance (H₂ anchor)
# =====================================================================


def test_krhf_multik_energy_invariant_under_1e10_screen():
    """Gate 1 (H₂ anchor): screened energy == unscreened to < 1e-8
    Ha/cell at fit_screen_threshold=1e-10."""
    system, basis = _h2_cubic_box()
    r_exact = run_krhf_periodic_gdf(system, basis, (2, 1, 1), _rhf_opts())
    r_scr = run_krhf_periodic_gdf(
        system,
        basis,
        (2, 1, 1),
        _rhf_opts(),
        fit_screen_threshold=1e-10,
    )
    assert r_exact.converged and r_scr.converged
    assert abs(r_exact.energy - r_scr.energy) < 1e-8, (
        f"screened {r_scr.energy} vs unscreened {r_exact.energy}: "
        f"|dE| = {abs(r_exact.energy - r_scr.energy):.3e}"
    )


def _energy_invariance_pin(system, basis, *, conv_tol: float = 1e-10):
    """Gate 1 body: screened == unscreened total energy to < 1e-8
    Ha/cell at fit_screen_threshold = 1e-10 on a (2,1,1) k-mesh."""
    opts_exact = _rhf_opts(max_iter=60)
    opts_exact.conv_tol_energy = conv_tol
    opts_scr = _rhf_opts(max_iter=60)
    opts_scr.conv_tol_energy = conv_tol
    r_exact = run_krhf_periodic_gdf(system, basis, (2, 1, 1), opts_exact)
    r_scr = run_krhf_periodic_gdf(
        system,
        basis,
        (2, 1, 1),
        opts_scr,
        fit_screen_threshold=1e-10,
    )
    assert r_exact.converged and r_scr.converged
    assert abs(r_exact.energy - r_scr.energy) < 1e-8, (
        f"screened {r_scr.energy} vs unscreened {r_exact.energy}: "
        f"|dE| = {abs(r_exact.energy - r_scr.energy):.3e}"
    )
    return r_exact.energy


@pytest.mark.slow
def test_krhf_c_diamond_energy_invariant_under_1e10_screen():
    """Gate 1 (tight-cell control): c-diamond fcc primitive, sto-3g,
    kmesh (2,1,1). On a tight cell every AO pair may survive the
    1e-10 screen -- the pin then verifies the screen machinery
    perturbs nothing, which is exactly the gate."""
    system, basis = _c_diamond()
    _energy_invariance_pin(system, basis)


@pytest.mark.slow
def test_krhf_mgo_energy_invariant_under_1e10_screen():
    """Gate 1 (ionic control): rocksalt MgO primitive, sto-3g,
    kmesh (2,1,1)."""
    system, basis = _mgo()
    _energy_invariance_pin(system, basis)


# =====================================================================
#                      Loud failure modes
# =====================================================================


def test_negative_threshold_raises():
    system, basis = _h2_cubic_box()
    with pytest.raises(ValueError, match="fit_screen_threshold"):
        run_krhf_periodic_gdf(
            system, basis, (2, 1, 1), _rhf_opts(), fit_screen_threshold=-1.0
        )


def test_threshold_on_compcell_method_raises():
    system, basis = _h2_cubic_box()
    with pytest.raises(NotImplementedError, match="rsgdf"):
        run_krhf_periodic_gdf(
            system,
            basis,
            (2, 1, 1),
            _rhf_opts(),
            gdf_method="compcell",
            fit_screen_threshold=1e-10,
        )


def test_gamma_fast_path_energy_invariant_under_1e10_screen():
    """The Γ (1,1,1) k-mesh takes the pure PBC-GDF fast path
    (run_pbc_gdf_rhf), which supports the screen via the Bloch-delegated
    build_lpq_native_fft + streamed V_ne: screened == unscreened to
    < 1e-8 Ha/cell (M4 of the handover; this raised NotImplementedError
    before the Γ plumbing landed)."""
    system, basis = _h2_cubic_box()
    r_exact = run_krhf_periodic_gdf(system, basis, (1, 1, 1), _rhf_opts())
    r_scr = run_krhf_periodic_gdf(
        system,
        basis,
        (1, 1, 1),
        _rhf_opts(),
        fit_screen_threshold=1e-10,
    )
    assert r_exact.converged and r_scr.converged
    assert abs(r_exact.energy - r_scr.energy) < 1e-8, (
        f"screened {r_scr.energy} vs unscreened {r_exact.energy}: "
        f"|dE| = {abs(r_exact.energy - r_scr.energy):.3e}"
    )


def test_gamma_level_shift_keeps_the_screened_range_separated_source(monkeypatch):
    """A convergence option cannot select a different fitting Hamiltonian."""
    import vibeqc.periodic_k_gdf as k_module

    system, basis = _h2_cubic_box()
    reference = run_krhf_periodic_gdf(system, basis, (1, 1, 1), _rhf_opts())
    original = k_module._build_scf_range_separated_lpq_cache
    seen = []
    def record_source(*args, **kwargs):
        cache = original(*args, **kwargs)
        seen.append((kwargs['fit_screen_threshold'], cache.source_parameters))
        return cache
    monkeypatch.setattr(k_module, '_build_scf_range_separated_lpq_cache', record_source)
    opts = _rhf_opts()
    opts.level_shift = 0.2
    actual = run_krhf_periodic_gdf(
        system, basis, (1, 1, 1), opts, fit_screen_threshold=1e-10,
    )
    assert reference.converged and actual.converged
    assert actual.energy == pytest.approx(reference.energy, abs=1e-8, rel=0)
    assert len(seen) == 1 and seen[0][0] == 1e-10
    assert 0 < seen[0][1][-1] <= 1e-10
