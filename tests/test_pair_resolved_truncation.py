"""M2 (pair-resolved truncation): domain, orbit closure, builds, enforcement.

The SYM3b root cause (HANDOVER_BIPOLE_PRODUCTION.md Sec. 0a 2026-07-12):
the radial cell list is closed under the bare point-group action, but
the symmetry machinery acts on atom-pair triples
``(a, b, h) -> (perm[a], perm[b], R.h + s_a - s_b)`` whose per-atom
lattice shifts push orbit partners of EVERY MgO cell outside any radial
list. The pair-resolved criterion ``|r_b + L.h - r_a| <= cutoff`` is
preserved by the action, so the triple space is group-invariant by
construction and ``identify_atom_pair_orbits`` runs with
``require_closed=True``.

Pinned here:

* the MgO c6 domain reproduces the probe: 19 cells (vs 13 radial),
  38 triples, 6 orbits (6.33x compression), all 6 symmetry-equivalent
  nearest-neighbour Mg-O couplings present (a radial list keeps 3);
* exact group-invariance of the domain; closure succeeds on the
  pair-resolved triple set and RAISES on the radial product space;
* reduced masked build == full pair-resolved build + enforcement,
  bit-identical (0.0), through the M1 full-domain binding;
* non-qualifying sub-blocks of the full pair-resolved build are exact
  zeros; the J-only variant matches and skips K;
* ``symmetrize_fock_blocks`` associates blocks with cells BY KEY when
  the caller template differs from the mapping's (the pre-M2
  positional association silently scrambled the exact-J pure-RKS +
  enforcement combination: radial lists at different radii are NOT
  positional prefixes of each other).
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq
from vibeqc._vibeqc_core import (
    LatticeSumOptions,
    attach_symmetry,
    compute_overlap_lattice,
    direct_lattice_cells,
    make_lattice_matrix_set,
)
from vibeqc.bipole_ext_el_pole import crystal_default_ewald_alpha
from vibeqc.bipole_symmetry_fock import (
    build_jk_pair_resolved,
    build_jk_reduced_symmetrized,
    pair_resolved_fock_mapping,
    representative_cell_indices,
    symmetrize_fock_blocks,
)
from vibeqc.pair_resolved_truncation import (
    atom_pair_span,
    domain_triples,
    pair_resolved_domain,
)
from vibeqc.symmetry_integrals import symmorphic_operations
from vibeqc.symmetry_lattice_c import (
    _atom_ao_slices,
    identify_atom_pair_orbits,
    operator_triple_actions,
)

ANG2BOHR = 1.0 / 0.529177210903


def _mgo(attach: bool = True):
    a = 4.21 * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    system = vq.PeriodicSystem(
        3, lattice, [vq.Atom(12, [0, 0, 0]), vq.Atom(8, [a / 2, a / 2, a / 2])]
    )
    if attach:
        attach_symmetry(system)
    return system, a


def _lat(cutoff: float) -> LatticeSumOptions:
    opts = LatticeSumOptions()
    opts.cutoff_bohr = float(cutoff)
    opts.nuclear_cutoff_bohr = float(cutoff)
    return opts


def _key(cell):
    return tuple(int(x) for x in np.asarray(cell.index).reshape(3))


def test_mgo_domain_reproduces_probe_numbers():
    system, a = _mgo()
    ops = symmorphic_operations(system.symmetry.operations)
    dom = pair_resolved_domain(system, 6.0, operations=ops)
    radial = list(direct_lattice_cells(system, 6.0))
    # Diagonal (a, a) pairs keep every radial cell; Mg-O pairs pull in
    # cells out to cutoff + pair_span = 12.89 bohr.
    assert {_key(c) for c in radial} <= {_key(c) for c in dom.cells}
    assert len(dom.cells) == 19
    assert dom.n_triples == 38
    assert abs(atom_pair_span(system) - np.sqrt(3.0) * a / 2.0) < 1e-9

    # All 6 symmetry-equivalent nearest-neighbour Mg->O couplings are
    # in the domain (the radial density list chops 3 of them -- the
    # dominant near-field asymmetry source of the SYM3b probe).
    r_o = np.array([a / 2.0, a / 2.0, a / 2.0])
    n_nn = 0
    for cell, pairs in zip(dom.cells, dom.pairs_by_cell):
        if (0, 1) in pairs:
            d = float(np.linalg.norm(r_o + np.asarray(cell.r_cart)))
            if abs(d - a / 2.0) < 1e-9:
                n_nn += 1
    assert n_nn == 6


def test_mgo_domain_is_exactly_group_invariant():
    system, _a = _mgo()
    ops = symmorphic_operations(system.symmetry.operations)
    dom = pair_resolved_domain(system, 6.0, operations=ops)
    triples = domain_triples(dom)
    op_R, op_perm, op_shift, _ident = operator_triple_actions(ops, system)
    for a_at, b_at, h in triples:
        for i in range(len(op_R)):
            h_img = tuple(
                int(x)
                for x in (op_R[i] @ np.array(h) + op_shift[i][a_at] - op_shift[i][b_at])
            )
            assert (int(op_perm[i][a_at]), int(op_perm[i][b_at]), h_img) in triples


def test_orbit_closure_pair_resolved_succeeds_radial_raises():
    system, _a = _mgo()
    ops = symmorphic_operations(system.symmetry.operations)
    dom = pair_resolved_domain(system, 6.0, operations=ops)
    orbits = identify_atom_pair_orbits(
        list(dom.cells), ops, system, require_closed=True,
        triples=domain_triples(dom),
    )
    assert orbits.n_orbits == 6
    assert abs(orbits.compression_ratio - 38 / 6) < 1e-12

    # The radial product space is orbit-open (the root cause): closure
    # must raise. This pins the fact that raising the cutoff is NOT a
    # fix -- the openness comes from the atom-offset shifts.
    for cutoff in (6.0, 12.0):
        cells = list(direct_lattice_cells(system, cutoff))
        with pytest.raises(ValueError, match="not symmetry-closed"):
            identify_atom_pair_orbits(cells, ops, system, require_closed=True)


def test_he_domain_equals_radial_ball():
    system = vq.PeriodicSystem(
        3, np.eye(3) * 4.0, [vq.Atom(2, [0.0, 0.0, 0.0])]
    )
    attach_symmetry(system)
    ops = symmorphic_operations(system.symmetry.operations)
    dom = pair_resolved_domain(system, 6.0, operations=ops)
    radial_keys = {_key(c) for c in direct_lattice_cells(system, 6.0)}
    assert {_key(c) for c in dom.cells} == radial_keys
    assert all(pairs == frozenset({(0, 0)}) for pairs in dom.pairs_by_cell)


def _mgo_build_fixture(cutoff: float = 6.0):
    system, _a = _mgo()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    lopts = _lat(cutoff)
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    omega = float(crystal_default_ewald_alpha(V))
    mapping = pair_resolved_fock_mapping(system, basis, lopts)
    assert mapping is not None and mapping.pair_resolved
    # Density with content on every cell of a wide list: D(h) = S(h).
    sw = _lat(2.0 * cutoff + atom_pair_span(system) + 1.0)
    S = compute_overlap_lattice(basis, system, sw)
    density = make_lattice_matrix_set(
        int(basis.nbasis),
        list(S.cells),
        [np.asarray(b, dtype=float) for b in S.blocks],
    )
    return system, basis, lopts, omega, mapping, density


def test_reduced_build_bit_identical_to_full_plus_enforcement():
    system, basis, lopts, omega, mapping, density = _mgo_build_fixture()
    full = build_jk_pair_resolved(basis, system, lopts, density, omega, mapping)
    j_ref = [np.array(b, dtype=float, copy=True) for b in full.J.blocks]
    k_ref = [np.array(b, dtype=float, copy=True) for b in full.K.blocks]
    symmetrize_fock_blocks(j_ref, mapping)
    symmetrize_fock_blocks(k_ref, mapping)

    reps = representative_cell_indices(mapping)
    red = build_jk_reduced_symmetrized(
        basis, system, lopts, density, omega, mapping, reps
    )
    assert list(map(_key, red.J.cells)) == list(map(_key, mapping.cells))
    for i in range(len(j_ref)):
        assert float(np.max(np.abs(np.asarray(red.J.blocks[i]) - j_ref[i]))) == 0.0
        assert float(np.max(np.abs(np.asarray(red.K.blocks[i]) - k_ref[i]))) == 0.0


def test_full_build_zeroes_nonqualifying_and_j_only_matches():
    system, basis, lopts, omega, mapping, density = _mgo_build_fixture()
    full = build_jk_pair_resolved(basis, system, lopts, density, omega, mapping)
    farmed = build_jk_pair_resolved(
        basis,
        system,
        lopts,
        density,
        omega,
        mapping,
        output_cell_farming_task_kind="chi-direct-output-cell",
    )
    slices = _atom_ao_slices(basis)
    checked = 0
    for i, pairs in enumerate(mapping.domain.pairs_by_cell):
        J_b = np.asarray(full.J.blocks[i])
        K_b = np.asarray(full.K.blocks[i])
        for p in range(2):
            for q in range(2):
                if (p, q) not in pairs:
                    checked += 1
                    assert float(np.max(np.abs(J_b[slices[p], slices[q]]))) == 0.0
                    assert float(np.max(np.abs(K_b[slices[p], slices[q]]))) == 0.0
    assert checked > 0  # the domain genuinely masks something
    for got_j, want_j, got_k, want_k in zip(
        farmed.J.blocks,
        full.J.blocks,
        farmed.K.blocks,
        full.K.blocks,
    ):
        np.testing.assert_array_equal(np.asarray(got_j), np.asarray(want_j))
        np.testing.assert_array_equal(np.asarray(got_k), np.asarray(want_k))
    assert (
        farmed.output_cell_farming_execution.global_task_count
        == len(mapping.output_positions)
    )

    j_only = build_jk_pair_resolved(
        basis, system, lopts, density, omega, mapping, compute_exchange=False
    )
    assert j_only.K is None
    for i in range(len(mapping.cells)):
        assert (
            float(
                np.max(
                    np.abs(
                        np.asarray(j_only.J.blocks[i])
                        - np.asarray(full.J.blocks[i])
                    )
                )
            )
            == 0.0
        )


def test_mask_lattice_blocks_to_domain():
    system, _a = _mgo()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    ops = symmorphic_operations(system.symmetry.operations)
    dom = pair_resolved_domain(system, 6.0, operations=ops)
    from vibeqc._vibeqc_core import make_lattice_matrix_set
    from vibeqc.pair_resolved_truncation import mask_lattice_blocks_to_domain
    from vibeqc.symmetry_lattice_c import _atom_ao_slices

    nbf = int(basis.nbasis)
    rng = np.random.default_rng(3)
    blocks = [rng.standard_normal((nbf, nbf)) for _ in dom.cells]
    lms = make_lattice_matrix_set(nbf, list(dom.cells), [b.copy() for b in blocks])
    mask_lattice_blocks_to_domain(basis, dom, lms)
    slices = _atom_ao_slices(basis)
    masked_something = False
    for i, pairs in enumerate(dom.pairs_by_cell):
        got = np.asarray(lms.blocks[i])
        for p in range(2):
            for q in range(2):
                sub = got[slices[p], slices[q]]
                if (p, q) in pairs:
                    assert np.array_equal(
                        sub, blocks[i][slices[p], slices[q]]
                    )
                else:
                    masked_something = True
                    assert float(np.max(np.abs(sub))) == 0.0
    assert masked_something

    short = make_lattice_matrix_set(
        nbf, list(dom.cells)[:3], [b.copy() for b in blocks[:3]]
    )
    with pytest.raises(ValueError, match="blocks for"):
        mask_lattice_blocks_to_domain(basis, dom, short)


def _gamma_fold_invariant_density(nbf, domain, basis):
    """Group-covariant Gamma-fold-like density: D(h) = I on every domain
    cell (the identity home block is invariant under the phaseless
    Gamma rep), masked to the qualifying pairs."""
    from vibeqc._vibeqc_core import make_lattice_matrix_set
    from vibeqc.pair_resolved_truncation import mask_lattice_blocks_to_domain

    D = make_lattice_matrix_set(
        nbf, list(domain.cells), [np.eye(nbf) for _ in domain.cells]
    )
    mask_lattice_blocks_to_domain(basis, domain, D)
    return D


def test_orbit_asymmetry_collapses_with_pair_resolved_domains():
    """The M3 payoff, measured at the block level with a group-covariant
    Gamma-fold model density (D(h) = I):

    * legacy radial domains: J orbit asymmetry ~3.6e-2 (the SYM3b
      enforcement-shift scale that keeps the flags opt-in);
    * pair-resolved output + density domains (the M3 wiring): ~1.6e-3;
    * + an M4-style padded internal ball (16 bohr): ~1e-6 -- the
      remaining floor is the internal summation domain, which M4b
      makes interaction-resolved.
    """
    from vibeqc._vibeqc_core import (
        build_jk_2e_real_space,
        build_jk_2e_real_space_domains,
        make_lattice_matrix_set,
    )
    from vibeqc.bipole_symmetry_fock import cell_orbit_mapping

    system, _a = _mgo()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    lopts = _lat(6.0)
    V = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    omega = float(crystal_default_ewald_alpha(V))
    mapping = pair_resolved_fock_mapping(system, basis, lopts)
    nbf = int(basis.nbasis)

    def _j_asym(jk, out_positions, m):
        jf = [
            np.array(jk.J.blocks[p], dtype=float, copy=True)
            for p in out_positions
        ]
        js = [b.copy() for b in jf]
        symmetrize_fock_blocks(js, m)
        return max(
            float(np.max(np.abs(jf[i] - js[i]))) for i in range(len(jf))
        )

    # Legacy radial domains (control).
    cells_rad = list(direct_lattice_cells(system, 12.0))
    D_rad = make_lattice_matrix_set(
        nbf, cells_rad, [np.eye(nbf) for _ in cells_rad]
    )
    cells6 = list(direct_lattice_cells(system, 6.0))
    m_leg = cell_orbit_mapping(system, basis, cells6)
    jk_leg = build_jk_2e_real_space(basis, system, lopts, D_rad, omega)
    asym_legacy = _j_asym(jk_leg, range(len(cells6)), m_leg)
    assert asym_legacy > 1e-2

    # M3 wiring: pair-resolved output + density domains.
    D_pr = _gamma_fold_invariant_density(nbf, mapping.density_domain, basis)
    jk_m3 = build_jk_2e_real_space_domains(
        basis, system, lopts, D_pr, mapping.cells_internal,
        mapping.output_positions, mapping.output_masks, omega,
    )
    asym_m3 = _j_asym(jk_m3, mapping.output_positions, mapping)
    assert asym_m3 < asym_legacy / 10.0
    assert asym_m3 < 5e-3

    # M4b preview: a padded internal ball collapses the residual further.
    pad = 16.0
    cells_int = list(direct_lattice_cells(system, pad))
    pos = {_key(c): i for i, c in enumerate(cells_int)}
    out_idx = [pos[_key(c)] for c in mapping.cells]
    jk_pad = build_jk_2e_real_space_domains(
        basis, system, _lat(pad), D_pr, cells_int, out_idx,
        mapping.output_masks, omega,
    )
    asym_pad = _j_asym(jk_pad, out_idx, mapping)
    assert asym_pad < 1e-5


def test_pair_mode_end_to_end_enforce_equals_reduce():
    """Driver-level M3: MgO RHF with use_fock_symmetry vs
    use_fock_symmetry_reduce runs on the pair-resolved domains and the
    two are numerically identical (reduce reconstructs exactly what
    enforcement projects onto). UHF reduces to the RHF value on the
    closed-shell cell. The 12-bohr image domain is the smallest measured
    fixture cutoff below the legacy-gauge fold-refusal threshold. This is an
    algorithm-parity check, not a production-energy witness."""
    from vibeqc._vibeqc_core import PeriodicRHFOptions, monkhorst_pack

    system, _a = _mgo()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1], use_symmetry=False)

    def _opts():
        opts = PeriodicRHFOptions()
        opts.lattice_opts.cutoff_bohr = 12.0
        opts.max_iter = 2
        opts.use_diis = False
        return opts

    from vibeqc.pbc_bipole import run_pbc_bipole_rhf
    from vibeqc.pbc_bipole_uhf import run_pbc_bipole_uhf

    e_enf = float(
        run_pbc_bipole_rhf(
            system, basis, kmesh, _opts(),
            use_fock_symmetry=True,
            use_fock_symmetry_reduce=False,
            use_exchange_ewald_split=False,
            sr_image_precision=None,
            progress=False,
        ).energy
    )
    e_red = float(
        run_pbc_bipole_rhf(
            system, basis, kmesh, _opts(),
            use_fock_symmetry_reduce=True,
            use_exchange_ewald_split=False,
            sr_image_precision=None,
            progress=False,
        ).energy
    )
    assert np.isfinite(e_enf)
    assert abs(e_enf - e_red) < 1e-12

    e_uhf = float(
        run_pbc_bipole_uhf(
            system, basis, kmesh, _opts(),
            use_fock_symmetry_reduce=True,
            use_exchange_ewald_split=False,
            sr_image_precision=None,
            progress=False,
        ).energy
    )
    assert abs(e_uhf - e_red) < 1e-8


def test_pair_mode_pure_rks_exact_j_runs():
    """Pure-functional pair mode exercises the exact-FT J route on the
    pair-resolved density template (mapping/legacy templates coincide
    for He-at-origin, keeping this cheap); the key-based enforcement
    must accept the density-template blocks (the pre-M2 positional
    association scrambled exactly this combination)."""
    from vibeqc._vibeqc_core import PeriodicKSOptions, monkhorst_pack
    from vibeqc.pbc_bipole_rks import run_pbc_bipole_rks

    system = vq.PeriodicSystem(
        3, np.eye(3) * 4.0, [vq.Atom(2, [0.0, 0.0, 0.0])]
    )
    attach_symmetry(system)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    kmesh = monkhorst_pack(system, [1, 1, 1], use_symmetry=False)
    opts = PeriodicKSOptions()
    opts.functional = "lda"
    opts.lattice_opts.cutoff_bohr = 6.0
    opts.max_iter = 2
    opts.use_diis = False

    e = float(
        run_pbc_bipole_rks(
            system, basis, kmesh, opts,
            use_fock_symmetry=True,
            use_fock_symmetry_reduce=False,
            progress=False,
        ).energy
    )
    assert np.isfinite(e)
    # He-at-origin: every orbit stays inside the domain, so the
    # pair-resolved run must match the plain run to numerical noise
    # (enforcement is a no-op; the truncation set is unchanged).
    e_plain = float(
        run_pbc_bipole_rks(
            system,
            basis,
            kmesh,
            opts,
            use_fock_symmetry_reduce=False,
            progress=False,
        ).energy
    )
    assert abs(e - e_plain) < 1e-9


def test_pair_mode_composes_with_sr_image_extent():
    """M4b: the padded internal ball composes with the pair-resolved
    domains. On the QUALIFYING home sub-blocks the padded pair-resolved
    build equals the M4a radial padded build bitwise (same internal
    quartet set for the home cell -- the M4a oracle expressed through
    the pair machinery); the non-qualifying home sub-blocks (the
    in-cell Mg-O pair sits at 6.89 bohr > cutoff 6 -- MgO's home
    cross-block is genuinely OUTSIDE the pair criterion; the 6 NN
    couplings live in neighbour cells) are exact zeros."""
    system, basis, lopts, omega, mapping, density = _mgo_build_fixture()
    from vibeqc.pbc_bipole_fock import _sr_image_padded_jk

    slices = _atom_ao_slices(basis)
    home_pairs = mapping.domain.pairs_by_cell[0]
    assert (0, 0) in home_pairs and (1, 1) in home_pairs
    assert (0, 1) not in home_pairs  # in-cell Mg-O at 6.89 bohr > 6

    extent = 14.0
    pair_pad = build_jk_pair_resolved(
        basis, system, lopts, density, omega, mapping,
        internal_extent_bohr=extent,
    )
    m4a_pad = _sr_image_padded_jk(
        basis, system, lopts, density, omega, extent
    )
    for lat_pair, lat_m4a in ((pair_pad.J, m4a_pad.J), (pair_pad.K, m4a_pad.K)):
        got = np.asarray(lat_pair.blocks[0], dtype=float)
        ref = np.asarray(lat_m4a.blocks[0], dtype=float)
        for p in range(2):
            for q in range(2):
                sub = got[slices[p], slices[q]] - ref[slices[p], slices[q]]
                if (p, q) in home_pairs:
                    assert float(np.max(np.abs(sub))) == 0.0
                else:
                    assert (
                        float(np.max(np.abs(got[slices[p], slices[q]])))
                        == 0.0
                    )
    # And the pad genuinely extends the pair-resolved internal ball.
    unpadded = build_jk_pair_resolved(
        basis, system, lopts, density, omega, mapping
    )
    assert (
        float(
            np.max(
                np.abs(
                    np.asarray(pair_pad.J.blocks[0])
                    - np.asarray(unpadded.J.blocks[0])
                )
            )
        )
        > 1e-6
    )


def test_symmetrize_key_based_on_wider_template():
    """Enforcement on a template wider than the mapping's cell list.

    The exact-J pure-RKS path hands enforcement blocks on the DENSITY
    template (2x-cutoff radial list). The enumerator is prefix-stable, so
    deliberately reverse paired cell/block values to exercise key-based
    association independently of list position. Enforcement must (a) leave
    exact symmetric content unchanged on covered cells and (b) leave
    uncovered cells untouched.
    """
    system, _a = _mgo()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    from vibeqc.bipole_symmetry_fock import cell_orbit_mapping

    cells6 = list(direct_lattice_cells(system, 6.0))
    mapping = cell_orbit_mapping(system, basis, cells6)
    assert mapping is not None

    S12 = compute_overlap_lattice(basis, system, _lat(12.0))
    paired = list(
        zip(
            list(S12.cells),
            [np.asarray(b, dtype=float).copy() for b in S12.blocks],
            strict=True,
        )
    )
    paired.reverse()
    template = [cell for cell, _ in paired]
    blocks = [block for _, block in paired]
    assert [_key(c) for c in template][: len(cells6)] != [
        _key(c) for c in cells6
    ], "fixture must exercise the non-prefix case"
    before = [b.copy() for b in blocks]

    symmetrize_fock_blocks(blocks, mapping, cells=template)

    covered = {_key(c) for c in cells6}
    for i, cell in enumerate(template):
        d = float(np.max(np.abs(blocks[i] - before[i])))
        if _key(cell) in covered:
            # S sub-blocks are exact integrals: the orbit action maps
            # them onto each other exactly, so enforcement is a no-op
            # up to roundoff.
            assert d < 1e-11
        else:
            assert d == 0.0

    # Fail-closed contracts: a template that misses mapping cells, and
    # a positional call whose block count disagrees with the mapping.
    with pytest.raises(ValueError, match="missing"):
        symmetrize_fock_blocks(
            [b.copy() for b in blocks[:5]],
            mapping,
            cells=template[:5],
        )
    with pytest.raises(ValueError, match="blocks for"):
        symmetrize_fock_blocks([b.copy() for b in blocks], mapping)
