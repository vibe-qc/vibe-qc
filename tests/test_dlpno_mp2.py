"""DLPNO-MP2 correctness gates (M2).

Validation ladder (each rung isolates one layer of the machinery):

1. Canonical occupieds + full domains + no PNO truncation: the
   semicanonical amplitudes are exact, no coupling is exercised —
   must equal canonical DF-MP2 (same fitting basis) to <= 1 uHa.
2. Boys-localised occupieds, same no-truncation limit: the localised
   F_oo off-diagonal coupling (with PNO-basis projections) must
   iterate back to the same canonical answer to <= 1 uHa.
3. Same exactness on a DZ basis (def2-SVP) where the virtual space is
   non-trivial, and on a second molecule (NH3).
4. Default thresholds on the DZ case: measured recovery tiers
   (>= 99.8% at TCutPNO=1e-8; >= 99.9% at 1e-9) with real PNO
   compression and a negative-definite truncation correction.

All references are computed in-process with vibe-qc's own canonical
DF-MP2 (C++), same geometry, same fitting basis.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from vibeqc import BasisSet, MP2Options, RHFOptions, run_mp2, run_rhf
from vibeqc._vibeqc_core import Atom, Molecule
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno.mp2 import (
    DLPNOMP2Options as _DLPNOMP2Options,
    DLPNOMP2Result,
    run_dlpno_mp2,
)

ANGSTROM_TO_BOHR = 1.8897259886
AUX = "def2-svp-rifit"
MICRO_HA = 1e-6

# These ratchets retain the pre-unification all-electron/Pinski-style MP2
# convention as historical numerical evidence. Current published defaults are
# pinned independently in test_dlpno.py and test_frozen_core_convention.py.
DLPNOMP2Options = partial(
    _DLPNOMP2Options,
    n_frozen=0,
    tcut_pno=1e-8,
    tcut_pno_weak=1e-7,
    tcut_mkn=1e-3,
    tcut_pairs=1e-6,
    tcut_pairs_weak=1e-4,
    domain_criterion="mulliken",  # retained with Mulliken domains; DOI became the default in #190
)

H2O_ATOMS = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
    (1, [0.0, -0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
]
NH3_ATOMS = [
    (7, [0.0, 0.0, 0.0]),
    (1, [0.0, 1.772, -0.721]),
    (1, [1.535, -0.886, -0.721]),
    (1, [-1.535, -0.886, -0.721]),
]

NO_TRUNCATION = dict(
    tcut_pno=0.0, tcut_pno_weak=0.0, tcut_mkn=0.0, tcut_pairs=0.0
)


def _system(atoms, basis_name):
    mol = Molecule(
        [Atom(z, pos) for z, pos in atoms], charge=0, multiplicity=1
    )
    basis = BasisSet(mol, basis_name)
    opts = RHFOptions()
    opts.max_iter = 100
    rhf = run_rhf(mol, basis, opts)
    assert rhf.converged
    mp2_opts = MP2Options()
    mp2_opts.n_frozen_core = 0
    mp2_opts.density_fit = True
    mp2_opts.aux_basis = AUX
    e_ref = run_mp2(mol, basis, rhf, mp2_opts).e_correlation
    df = DensityFitting(basis, BasisSet(mol, AUX), aux_basis_name=AUX)
    return mol, basis, rhf, df, e_ref


@pytest.fixture(scope="module")
def h2o_sto3g():
    return _system(H2O_ATOMS, "sto-3g")


@pytest.fixture(scope="module")
def h2o_dz():
    return _system(H2O_ATOMS, "def2-svp")


@pytest.mark.parametrize("localise", ["none", "boys"])
@pytest.mark.parametrize("tcut", [0.0, 1e-4])
def test_shared_pair_fock_matches_previous_solve(monkeypatch, h2o_dz, localise, tcut):
    """Pair frames, truncation correction and coupled energy survive migration."""
    from types import SimpleNamespace
    from scipy.linalg import eigvalsh
    from vibeqc.dlpno import mp2 as module

    mol, basis, rhf, df, _ = h2o_dz
    opts = DLPNOMP2Options(
        localise=localise, n_frozen=1, tcut_pno=tcut, tcut_pno_weak=tcut,
        tcut_mkn=0.0, tcut_pairs=0.0,
    )
    solve = module.solve_dense_fock
    sizes = {}

    def checked_solve(matrix, **kwargs):
        result = solve(matrix, **kwargs)
        metadata = result.operator_metadata
        assert metadata.kind == "fock" and metadata.spin == "restricted"
        assert result.residual_frame == "orthogonal"
        assert result.selection_verified
        assert max(result.residuals) < 1e-11
        assert result.orthogonality_error < 1e-12
        np.testing.assert_allclose(result.eigenvalues, eigvalsh(matrix), atol=1e-12)
        assert metadata.basis_id not in sizes
        sizes[metadata.basis_id] = len(result.eigenvalues)
        return result

    monkeypatch.setattr(module, "solve_dense_fock", checked_solve)
    actual = run_dlpno_mp2(mol, basis, rhf, df, opts)
    assert sizes == {f"dlpno-mp2-pair-{i}-{j}-pno": n
                     for (i, j), n in actual.pno_per_pair.items()}
    assert sizes
    assert all(i >= 1 and j >= 1 for i, j in actual.pno_per_pair)

    def previous_solve(matrix, **kwargs):
        eps, C = np.linalg.eigh(matrix)
        return SimpleNamespace(eigenvalues=eps, eigenvectors=C)

    monkeypatch.setattr(module, "solve_dense_fock", previous_solve)
    previous = run_dlpno_mp2(mol, basis, rhf, df, opts)
    assert actual.converged and previous.converged
    assert actual.pno_per_pair == previous.pno_per_pair
    assert actual.n_pairs_screened == previous.n_pairs_screened
    assert actual.pair_energies == pytest.approx(previous.pair_energies, abs=1e-11)
    for name in ("e_corr", "e_corr_iterated", "e_pno_correction", "e_distant"):
        assert getattr(actual, name) == pytest.approx(getattr(previous, name), abs=1e-11)


def test_explicit_sources_keep_interleaved_ao_frames(monkeypatch, h2o_sto3g):
    """Nested MP2 runs retain their own nonorthogonal AO frame and core."""
    from dataclasses import asdict, replace
    from types import SimpleNamespace
    from vibeqc import _vibeqc_core as core
    from vibeqc.dlpno.source import CorrelationSource

    mol, basis, rhf, df, _ = h2o_sto3g
    overlap = np.asarray(core.compute_overlap(basis))
    options = DLPNOMP2Options(
        localise="none", n_frozen=1, tcut_pno=0.0,
        tcut_pno_weak=0.0, tcut_mkn=0.0, tcut_pairs=0.0,
        tcut_pairs_weak=0.0,
    )
    before_options = asdict(options)
    expected = run_dlpno_mp2(mol, basis, rhf, df, options)

    def make_source(scale):
        # An invertible diagonal AO change leaves the finite Hamiltonian
        # unchanged if S, F, C and the factors all transform together.
        transform = np.diag(scale)
        reference = SimpleNamespace(
            energy=rhf.energy,
            mo_coeffs=np.asarray(rhf.mo_coeffs) / scale[:, None],
            fock=transform @ np.asarray(rhf.fock) @ transform,
        )

        class Factors:
            on_transform = None

            def mo_transform(self, left, right=None):
                if self.on_transform is not None:
                    callback, self.on_transform = self.on_transform, None
                    callback()
                right = left if right is None else right
                return df.mo_transform(transform @ left, transform @ right)

        source = CorrelationSource(
            kind="test-ao-frame", reference_kind="rhf", molecule=mol,
            basis=basis, reference=reference, factors=Factors(),
            overlap=transform @ overlap @ transform,
            provenance={"ao_scale": scale.tolist()},
        )
        for array in (reference.mo_coeffs, reference.fock, source.overlap):
            array.flags.writeable = False
        return source

    outer = make_source(np.ones(basis.nbasis))
    inner = make_source(np.linspace(0.7, 1.3, basis.nbasis))
    nested = []
    outer.factors.on_transform = lambda: nested.append(
        run_dlpno_mp2(None, None, None, None, options, source=inner)
    )

    def forbidden_overlap(_):
        raise AssertionError("explicit sources must not reconstruct AO overlap")

    monkeypatch.setattr(core, "compute_overlap", forbidden_overlap)
    actual = run_dlpno_mp2(None, None, None, None, options, source=outer)
    for result, source in ((actual, outer), (nested[0], inner)):
        assert result.converged and result.n_frozen == 1
        assert result.e_corr == pytest.approx(expected.e_corr, abs=1e-11)
        assert result.pair_energies == pytest.approx(expected.pair_energies, abs=1e-11)
        assert result.source_kind == source.kind
        assert result.source_identity == source.identity
    assert outer.identity != inner.identity
    assert core.compute_overlap is forbidden_overlap
    assert asdict(options) == before_options

    class FailedFactors:
        def mo_transform(self, left, right=None):
            raise RuntimeError("factor transform failed")

    with pytest.raises(RuntimeError, match="factor transform failed"):
        run_dlpno_mp2(
            None, None, None, None, options,
            source=replace(outer, factors=FailedFactors()),
        )
    assert asdict(options) == before_options
    assert core.compute_overlap is forbidden_overlap
    assert not outer.reference.mo_coeffs.flags.writeable
    assert not outer.reference.fock.flags.writeable
    assert not outer.overlap.flags.writeable


class TestExactnessLimit:
    """Full domains + no truncation must reproduce canonical DF-MP2."""

    def test_rung1_canonical_occupieds(self, h2o_sto3g):
        mol, basis, rhf, df, e_ref = h2o_sto3g
        r = run_dlpno_mp2(
            mol, basis, rhf, df,
            DLPNOMP2Options(localise="none", **NO_TRUNCATION),
        )
        assert r.converged
        assert abs(r.e_corr - e_ref) < 1.0 * MICRO_HA
        # Canonical orbitals: semicanonical amplitudes are already exact,
        # so the coupled iteration must terminate immediately.
        assert r.n_iter <= 3

    def test_rung2_boys_localised(self, h2o_sto3g):
        mol, basis, rhf, df, e_ref = h2o_sto3g
        r = run_dlpno_mp2(
            mol, basis, rhf, df,
            DLPNOMP2Options(localise="boys", **NO_TRUNCATION),
        )
        assert r.converged
        assert abs(r.e_corr - e_ref) < 1.0 * MICRO_HA
        # Localised occupieds genuinely exercise the F_oo coupling.
        assert r.n_iter > 3

    def test_rung3_dz_basis(self, h2o_dz):
        mol, basis, rhf, df, e_ref = h2o_dz
        r = run_dlpno_mp2(
            mol, basis, rhf, df,
            DLPNOMP2Options(localise="boys", **NO_TRUNCATION),
        )
        assert r.converged
        assert abs(r.e_corr - e_ref) < 1.0 * MICRO_HA

    def test_rung3_second_molecule_nh3(self):
        mol, basis, rhf, df, e_ref = _system(NH3_ATOMS, "sto-3g")
        r = run_dlpno_mp2(
            mol, basis, rhf, df,
            DLPNOMP2Options(localise="boys", **NO_TRUNCATION),
        )
        assert r.converged
        assert abs(r.e_corr - e_ref) < 1.0 * MICRO_HA


class TestStructure:
    """Structural invariants of the new pipeline."""

    def test_diagonal_pairs_present_and_counted(self, h2o_sto3g):
        mol, basis, rhf, df, _ = h2o_sto3g
        r = run_dlpno_mp2(
            mol, basis, rhf, df,
            DLPNOMP2Options(localise="boys", **NO_TRUNCATION),
        )
        n_occ = mol.n_electrons() // 2
        assert r.n_pairs == n_occ * (n_occ + 1) // 2  # includes (i,i)
        diag = [k for k in r.pair_energies if k[0] == k[1]]
        assert len(diag) == n_occ
        # Diagonal pairs carry real correlation energy.
        assert all(r.pair_energies[k] < -1e-5 for k in diag)

    def test_pair_energies_sum_to_iterated_energy(self, h2o_sto3g):
        mol, basis, rhf, df, _ = h2o_sto3g
        r = run_dlpno_mp2(
            mol, basis, rhf, df,
            DLPNOMP2Options(localise="boys", **NO_TRUNCATION),
        )
        assert sum(r.pair_energies.values()) == pytest.approx(
            r.e_corr_iterated, abs=1e-12
        )

    def test_energy_composition(self, h2o_dz):
        mol, basis, rhf, df, _ = h2o_dz
        r = run_dlpno_mp2(mol, basis, rhf, df, DLPNOMP2Options())
        assert r.e_corr == pytest.approx(
            r.e_corr_iterated + r.e_pno_correction, abs=1e-14
        )
        assert r.e_total == pytest.approx(r.e_hf + r.e_corr, abs=1e-14)


class TestTruncationRecovery:
    """Pre-unification threshold tiers, retained as measured evidence."""

    def test_pre_unification_thresholds_recover_99_8(self, h2o_dz):
        mol, basis, rhf, df, e_ref = h2o_dz
        r = run_dlpno_mp2(mol, basis, rhf, df, DLPNOMP2Options())
        assert r.converged
        recovery = r.e_corr / e_ref
        assert recovery > 0.998, f"recovery {recovery:.5%}"
        assert recovery < 1.005, f"recovery {recovery:.5%} (overshoot)"
        # PNO compression must happen at the recorded historical thresholds.
        n_occ = mol.n_electrons() // 2
        n_vir_full = basis.nbasis - n_occ
        avg_pno = sum(r.pno_per_pair.values()) / len(r.pno_per_pair)
        assert avg_pno < 0.9 * n_vir_full
        # Truncation correction restores energy (negative contribution).
        assert r.e_pno_correction < 0.0

    def test_tight_pno_recovers_99_9(self, h2o_dz):
        mol, basis, rhf, df, e_ref = h2o_dz
        r = run_dlpno_mp2(
            mol, basis, rhf, df,
            DLPNOMP2Options(tcut_pno=1e-9, tcut_pno_weak=1e-8),
        )
        assert r.converged
        recovery = r.e_corr / e_ref
        assert recovery > 0.999, f"recovery {recovery:.5%}"

    def test_truncation_error_monotone_in_tcut_pno(self, h2o_dz):
        """Tighter TCutPNO must not worsen the recovered energy."""
        mol, basis, rhf, df, e_ref = h2o_dz
        errs = []
        for tcut in (1e-7, 1e-8, 1e-9):
            r = run_dlpno_mp2(
                mol, basis, rhf, df,
                DLPNOMP2Options(tcut_pno=tcut, tcut_pno_weak=tcut * 10),
            )
            errs.append(abs(r.e_corr - e_ref))
        assert errs[0] >= errs[1] >= errs[2]


def _canonical_frozen_core_dfmp2(rhf, df, n_frozen: int, n_occ: int) -> float:
    """Canonical frozen-core DF-MP2 built directly from the DF B-tensor."""
    C = np.asarray(rhf.mo_coeffs)
    eps = np.asarray(rhf.mo_energies)
    C_act = C[:, n_frozen:n_occ]
    C_vir = C[:, n_occ:]
    B = df.mo_transform(C_act, C_vir)  # (naux, n_act, n_vir)
    K = np.einsum("Pia,Pjb->iajb", B, B)
    eo = eps[n_frozen:n_occ]
    ev = eps[n_occ:]
    D = (
        eo[:, None, None, None]
        + eo[None, None, :, None]
        - ev[None, :, None, None]
        - ev[None, None, None, :]
    )
    T = K / D
    return float(
        2.0 * np.einsum("iajb,iajb->", K, T) - np.einsum("iajb,ibja->", K, T)
    )


class TestFrozenCore:
    """Frozen-core DLPNO-MP2 against a canonical frozen-core reference."""

    def test_frozen_core_exactness_limit(self, h2o_dz):
        mol, basis, rhf, df, _ = h2o_dz
        n_occ = mol.n_electrons() // 2
        e_ref_fc = _canonical_frozen_core_dfmp2(rhf, df, 1, n_occ)
        r = run_dlpno_mp2(
            mol, basis, rhf, df,
            DLPNOMP2Options(localise="boys", n_frozen=1, **NO_TRUNCATION),
        )
        assert r.converged
        assert abs(r.e_corr - e_ref_fc) < 1.0 * MICRO_HA
        # The reference helper itself must agree with the all-electron
        # C++ value when nothing is frozen (guards the helper).
        _, _, _, _, e_ref_all = h2o_dz
        assert _canonical_frozen_core_dfmp2(rhf, df, 0, n_occ) == pytest.approx(
            e_ref_all, abs=1e-9
        )

    def test_frozen_core_structure(self, h2o_dz):
        mol, basis, rhf, df, _ = h2o_dz
        n_occ = mol.n_electrons() // 2
        r = run_dlpno_mp2(
            mol, basis, rhf, df,
            DLPNOMP2Options(localise="boys", n_frozen=1, **NO_TRUNCATION),
        )
        n_act = n_occ - 1
        assert r.n_pairs == n_act * (n_act + 1) // 2
        # Absolute occupied indexing: the core orbital (index 0) never
        # appears in pair keys.
        assert all(i >= 1 and j >= 1 for (i, j) in r.pair_energies)
        # Freezing the O 1s must remove only a small part of E_corr.
        r_all = run_dlpno_mp2(
            mol, basis, rhf, df,
            DLPNOMP2Options(localise="boys", **NO_TRUNCATION),
        )
        assert abs(r.e_corr) < abs(r_all.e_corr)
        assert abs(r.e_corr - r_all.e_corr) < 0.05 * abs(r_all.e_corr)


class TestDistantPairs:
    """Dipole-estimate screening on well-separated H2 units."""

    @pytest.fixture(scope="class")
    @classmethod
    def h2_chain(cls):
        # Three H2 units along z, 12 bohr apart: inter-unit sigma-sigma
        # pairs at R = 12 and 24 bohr are genuine distant pairs.
        atoms = []
        for k in range(3):
            z0 = 12.0 * k
            atoms += [(1, [0.0, 0.0, z0]), (1, [0.0, 0.0, z0 + 1.4])]
        return _system(atoms, "sto-3g")

    def test_screening_loss_is_bounded_and_accounted(self, h2_chain):
        mol, basis, rhf, df, e_ref = h2_chain
        full = run_dlpno_mp2(
            mol, basis, rhf, df,
            DLPNOMP2Options(localise="boys", **NO_TRUNCATION),
        )
        assert abs(full.e_corr - e_ref) < 1.0 * MICRO_HA  # exactness holds

        screened = run_dlpno_mp2(
            mol, basis, rhf, df,
            DLPNOMP2Options(
                localise="boys",
                tcut_pno=0.0, tcut_pno_weak=0.0, tcut_mkn=0.0,
                tcut_pairs=1e-6, dipole_r_min=8.0,
            ),
        )
        assert screened.n_pairs_screened >= 1
        assert screened.n_pairs + screened.n_pairs_screened == full.n_pairs
        # The dipole estimate stands in for the screened pairs: it must
        # be attractive and the total must stay close to the full result.
        assert screened.e_distant <= 0.0
        assert abs(screened.e_corr - full.e_corr) < 20.0 * MICRO_HA

    @pytest.mark.parametrize("localise", ("boys", "none"))
    def test_custom_weak_interval_changes_retained_pno_spaces(
        self,
        h2_chain,
        localise,
    ):
        """The weak PNO cutoff is live only above the screening boundary."""
        mol, basis, rhf, df, _ = h2_chain
        full = run_dlpno_mp2(
            mol,
            basis,
            rhf,
            df,
            DLPNOMP2Options(
                localise=localise,
                tcut_pno=0.0,
                tcut_pno_weak=0.0,
                tcut_mkn=0.0,
                tcut_pairs=0.0,
                tcut_pairs_weak=1.0,
                dipole_r_min=0.0,
                pair_distance_fn=lambda *_: 10.0,
            ),
        )
        weak = run_dlpno_mp2(
            mol,
            basis,
            rhf,
            df,
            DLPNOMP2Options(
                localise=localise,
                tcut_pno=0.0,
                tcut_pno_weak=1.0,
                tcut_mkn=0.0,
                tcut_pairs=0.0,
                tcut_pairs_weak=1.0,
                dipole_r_min=0.0,
                pair_distance_fn=lambda *_: 10.0,
            ),
        )

        off_diagonal = [pair for pair in full.pno_per_pair if pair[0] != pair[1]]
        assert full.n_pairs_screened == weak.n_pairs_screened == 0
        assert any(
            weak.pno_per_pair[pair] < full.pno_per_pair[pair]
            for pair in off_diagonal
        )

    def test_close_pairs_never_screened(self, h2_chain):
        mol, basis, rhf, df, _ = h2_chain
        r = run_dlpno_mp2(
            mol, basis, rhf, df,
            DLPNOMP2Options(
                localise="boys",
                tcut_pno=0.0, tcut_pno_weak=0.0, tcut_mkn=0.0,
                # Absurdly loose threshold: would screen everything the
                # estimate covers — the r_min guard must protect the
                # intra-unit diagonal+near pairs.
                tcut_pairs=1.0, dipole_r_min=8.0,
            ),
        )
        # Diagonal pairs (3) survive unconditionally; every surviving
        # off-diagonal pair must be a close one (here: none closer than
        # 8 bohr between units, so exactly the diagonals remain).
        assert r.n_pairs >= 3
        assert all(
            r.pair_energies[k] < 0.0 for k in r.pair_energies if k[0] == k[1]
        )


def _translation_mp2_model(mesh, n_bands, *, full):
    """A finite periodic Hamiltonian with independent dense translation algebra."""
    from types import SimpleNamespace
    from vibeqc.dlpno.translation import TranslationPairSpaces
    from vibeqc.periodic.chi.pno import build_translation_pair_plan

    cells = tuple(np.ndindex(mesh))
    lookup = {cell: index for index, cell in enumerate(cells)}
    nk, n_virtual_bands = len(cells), 2
    n_local = n_bands + n_virtual_bands
    rng = np.random.default_rng(150 + nk + n_bands)

    def permutation(shift, block):
        result = np.zeros((nk * block, nk * block))
        for cell, coordinate in enumerate(cells):
            translated = lookup[tuple((a + b) % n for a, b, n in zip(coordinate, shift, mesh))]
            for orbital in range(block):
                result[translated * block + orbital, cell * block + orbital] = 1.0
        return result

    ao_translations = [permutation(cell, n_local) for cell in cells]
    occ_translations = [permutation(cell, n_bands) for cell in cells]
    vir_translations = [permutation(cell, n_virtual_bands) for cell in cells]

    def invariant_matrix(translations, diagonal, scale):
        raw = rng.normal(size=(len(diagonal), len(diagonal)))
        raw = (raw + raw.T) * scale
        return np.diag(diagonal) + sum(p @ raw @ p.T for p in translations) / nk

    overlap = invariant_matrix(ao_translations, np.full(nk * n_local, 1.4), 0.01)
    eigenvalues, vectors = np.linalg.eigh(overlap)
    sqrt = (vectors * np.sqrt(eigenvalues)) @ vectors.T
    inverse_sqrt = (vectors / np.sqrt(eigenvalues)) @ vectors.T
    occupied_rows = [cell * n_local + band for cell in range(nk) for band in range(n_bands)]
    virtual_rows = [cell * n_local + n_bands + band for cell in range(nk)
                    for band in range(n_virtual_bands)]
    occupied = inverse_sqrt[:, occupied_rows]
    virtual = inverse_sqrt[:, virtual_rows]
    foo = invariant_matrix(occ_translations, np.tile(np.linspace(-1.6, -1.1, n_bands), nk), 0.035)
    fvv = invariant_matrix(vir_translations, np.tile([0.7, 1.3], nk), 0.04)
    fock_orthogonal = np.zeros_like(overlap)
    fock_orthogonal[np.ix_(occupied_rows, occupied_rows)] = foo
    fock_orthogonal[np.ix_(virtual_rows, virtual_rows)] = fvv
    fock = sqrt @ fock_orthogonal @ sqrt

    # Translating both orthonormal orbital indices permutes the factors. Thus the
    # actual four-index Hamiltonian is covariant without assuming an energy.
    seed = rng.normal(size=overlap.shape) * 0.18
    seed = 0.5 * (seed + seed.T)
    factors = np.array([p @ seed @ p.T for p in ao_translations])
    bov = factors[:, occupied_rows][:, :, virtual_rows]
    exchange = np.einsum("Pia,Pjb->ijab", bov, bov)

    plan = build_translation_pair_plan(n_bands, mesh)
    frames, keys = [], []
    for row, ((band_i, band_j), shift) in enumerate(zip(plan.band_pairs, plan.displacements)):
        partner_cell = lookup[tuple(shift)]
        keys.append((int(band_i), partner_cell * n_bands + int(band_j)))
        if full:
            selected = np.arange(nk * n_virtual_bands)
        else:
            channels = range(1 + row % 2)
            selected = sorted({cell * n_virtual_bands + channel
                               for cell in (0, partner_cell) for channel in channels})
        rotation, _ = np.linalg.qr(rng.normal(size=(len(selected), len(selected))))
        frame = np.eye(nk * n_virtual_bands)[:, selected] @ rotation
        frames.append(frame)
    spaces = TranslationPairSpaces(
        mesh, overlap, fock, occupied,
        tuple(virtual @ frame for frame in frames),
    )
    assert spaces.representative_pairs == tuple(keys)
    return SimpleNamespace(
        spaces=spaces, frames=frames, exchange=exchange, foo=foo, fvv=fvv,
        occupied=occupied, virtual=virtual, overlap=overlap, fock=fock,
        occ_translations=occ_translations, vir_translations=vir_translations,
        rng=rng, n_cells=nk,
        ao_factors=np.einsum("ma,Pab,bn->Pmn", sqrt, factors, sqrt, optimize=True),
    )


def _translation_dense_lift(model, tensors):
    """Expand by explicit physical translations, without the pair resolver."""
    no, nv = len(model.foo), len(model.fvv)
    lifted = np.full((no, no, nv, nv), np.nan)
    for (i, j), frame, tensor in zip(model.spaces.representative_pairs, model.frames, tensors):
        block = frame @ tensor @ frame.T
        for occupied_shift, virtual_shift in zip(model.occ_translations, model.vir_translations):
            ti = int(np.argmax(occupied_shift[:, i]))
            tj = int(np.argmax(occupied_shift[:, j]))
            translated = virtual_shift @ block @ virtual_shift.T
            for a, b, value in ((ti, tj, translated), (tj, ti, translated.T)):
                if np.isfinite(lifted[a, b]).all():
                    np.testing.assert_allclose(lifted[a, b], value, atol=2e-12)
                lifted[a, b] = value
    assert np.isfinite(lifted).all()
    return lifted


@pytest.mark.parametrize("mesh,n_bands", [((1, 1, 1), 2), ((2, 1, 1), 1),
                                         ((3, 1, 1), 2), ((2, 2, 1), 1)])
def test_translation_pair_residual_matches_independent_dense_lift(mesh, n_bands):
    from vibeqc.dlpno.translation_mp2 import (
        translation_mp2_residual, translation_pair_energy_per_cell,
    )

    model = _translation_mp2_model(mesh, n_bands, full=False)
    no, nv = len(model.foo), len(model.fvv)
    raw = model.rng.normal(size=(no, no, nv, nv)) * 0.03
    raw = 0.5 * (raw + raw.transpose(1, 0, 3, 2))
    # Average a dense trial over physical translations before truncating it.
    trial = np.zeros_like(raw)
    for po, pv in zip(model.occ_translations, model.vir_translations):
        trial += np.einsum("Ii,Jj,Aa,Bb,ijab->IJAB", po, po, pv, pv, raw, optimize=True)
    trial /= model.n_cells
    amplitudes, integrals = [], []
    for (i, j), frame in zip(model.spaces.representative_pairs, model.frames):
        amplitudes.append(frame.T @ trial[i, j] @ frame)
        integrals.append(frame.T @ model.exchange[i, j] @ frame)
    saved = tuple(value.copy() for value in (*integrals, *amplitudes))
    lifted = _translation_dense_lift(model, amplitudes)
    expected_dense = (
        model.exchange
        + np.einsum("ac,ijcb->ijab", model.fvv, lifted)
        + np.einsum("ijac,cb->ijab", lifted, model.fvv)
        - np.einsum("ik,kjab->ijab", model.foo, lifted)
        - np.einsum("kj,ikab->ijab", model.foo, lifted)
    )
    actual = translation_mp2_residual(model.spaces, integrals, amplitudes)
    assert len(actual) == model.spaces.plan.n_representatives
    for (i, j), frame, residual in zip(model.spaces.representative_pairs, model.frames, actual):
        np.testing.assert_allclose(residual, frame.T @ expected_dense[i, j] @ frame,
                                   atol=3e-12, rtol=3e-12)
    expected_energy = np.sum(model.exchange * (2 * lifted - lifted.swapaxes(2, 3))) / model.n_cells
    assert translation_pair_energy_per_cell(model.spaces, integrals, amplitudes) == pytest.approx(
        expected_energy, abs=3e-12,
    )
    for original, value in zip(saved, (*integrals, *amplitudes)):
        np.testing.assert_array_equal(value, original)
    if model.n_cells > 1:
        assert model.spaces.plan.n_representatives < no * (no + 1) // 2
    assert max(np.linalg.norm(value) for value in actual) > 1e-3


@pytest.mark.parametrize("mesh,n_bands", [((2, 1, 1), 1), ((3, 1, 1), 2),
                                         ((2, 2, 1), 1)])
def test_translation_pair_full_space_matches_canonical_mp2(mesh, n_bands):
    from vibeqc.correlation import mp2_energy
    from vibeqc.dlpno.translation_mp2 import (
        translation_mp2_residual, translation_pair_energy_per_cell,
    )

    model = _translation_mp2_model(mesh, n_bands, full=True)
    eo, uo = np.linalg.eigh(model.foo)
    ev, uv = np.linalg.eigh(model.fvv)
    canonical_exchange = np.einsum("pi,qj,ar,bs,pqab->ijrs", uo, uo, uv, uv,
                                   model.exchange, optimize=True)
    denominator = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
    canonical_amplitude = canonical_exchange / denominator
    trial = np.einsum("pi,qj,ar,bs,ijrs->pqab", uo, uo, uv, uv, canonical_amplitude, optimize=True)
    amplitudes, integrals = [], []
    for (i, j), frame in zip(model.spaces.representative_pairs, model.frames):
        amplitudes.append(frame.T @ trial[i, j] @ frame)
        integrals.append(frame.T @ model.exchange[i, j] @ frame)
    actual = translation_mp2_residual(model.spaces, integrals, amplitudes)
    assert np.linalg.norm(model.foo - np.diag(np.diag(model.foo))) > 1e-3
    assert max(np.max(np.abs(value)) for value in actual) < 3e-12
    expected = mp2_energy(eo, ev, canonical_exchange.transpose(0, 2, 1, 3))
    assert translation_pair_energy_per_cell(model.spaces, integrals, amplitudes) == pytest.approx(
        expected.e_corr / model.n_cells, abs=3e-12,
    )


def _translation_pno_source(mesh, *, frozen=False):
    from types import SimpleNamespace
    from vibeqc.dlpno.source import CorrelationSource, LpqFactors, SourceConventions

    model = _translation_mp2_model(mesh, 2, full=True)
    nbf, no = model.occupied.shape
    active_indices = np.arange(no)[1::2] if frozen else np.arange(no)
    frozen_indices = np.arange(no)[::2] if frozen else np.empty(0, dtype=int)
    active, core = model.occupied[:, active_indices], model.occupied[:, frozen_indices]
    reference = SimpleNamespace(
        converged=True, energy=-2.0 * no, fock=model.fock,
        mo_coeffs=np.column_stack((core, active, model.virtual)),
    )
    source = CorrelationSource(
        kind="finite-translation-model", reference_kind="rhf",
        molecule=SimpleNamespace(multiplicity=1, n_electrons=lambda: 2 * no),
        basis=SimpleNamespace(nbasis=nbf), reference=reference,
        factors=LpqFactors(model.ao_factors), overlap=model.overlap,
        conventions=SourceConventions(kmesh=mesh, bloch_normalisation="real-bvk-supercell"),
    )
    return model, source, active, core, active_indices


@pytest.mark.parametrize("mesh,frozen,trimmed,norm", [
    ((2, 1, 1), False, False, "mp2"),
    ((3, 1, 1), True, False, "legacy"),
    ((2, 1, 1), True, True, "mp2"),
    ((3, 1, 1), False, True, "legacy"),
])
def test_translation_pno_builder_matches_dense_pair_construction(mesh, frozen, trimmed, norm):
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos
    from vibeqc.dlpno.translation_mp2 import translation_pair_energy_per_cell
    from vibeqc.periodic.chi.pno import build_translation_pair_plan

    model, source, active, core, indices = _translation_pno_source(mesh, frozen=frozen)
    nk, nbf = model.n_cells, len(model.overlap)
    plan = build_translation_pair_plan(active.shape[1] // nk, mesh)
    local = nbf // nk
    domains = []
    for shift in plan.displacements:
        cell = int(np.ravel_multi_index(tuple(shift), mesh))
        domains.append(np.array(sorted({r for c in (0, cell) for r in range(c * local, (c + 1) * local)}))
                       if trimmed else np.arange(nbf))
    cutoff = 1e-3
    built = build_translation_pair_pnos(
        source, mesh, active, domains, frozen_occupied=core,
        tcut_pno=cutoff, pno_norm=norm,
    )
    S, F = model.overlap, model.fock
    projector = np.eye(nbf) - model.occupied @ model.occupied.T @ S
    focc = active.T @ F @ active
    expected_correction = 0.0
    for row, ((i, j), domain, actual) in enumerate(zip(
        built.spaces.representative_pairs, domains, built.spaces.pair_coefficients,
    )):
        # Independent redundant-PAO Gram and semicanonical spectral construction.
        raw = projector[:, domain]
        gram = raw.T @ S @ raw
        ge, gu = np.linalg.eigh((gram + gram.T) / 2)
        base = raw @ (gu[:, ge > 1e-8] / np.sqrt(ge[ge > 1e-8]))
        eps, rotation = np.linalg.eigh((base.T @ F @ base + base.T @ F.T @ base) / 2)
        base = base @ rotation
        factors = np.einsum("Pmn,mi,na->Pia", model.ao_factors, active[:, [i, j]], base)
        exchange = factors[:, 0].T @ factors[:, 1]
        amplitude = exchange / (focc[i, i] + focc[j, j] - eps[:, None] - eps[None, :])
        if norm == "legacy":
            density = (amplitude @ amplitude.T + amplitude.T @ amplitude) / (1 + (i == j))
        else:
            dual = (4 * amplitude - 2 * amplitude.T) / (1 + (i == j))
            density = dual @ amplitude.T + dual.T @ amplitude
        values, vectors = np.linalg.eigh((density + density.T) / 2)
        keep = values > cutoff
        if not np.any(keep):
            keep[-1] = True
        selected = vectors[:, keep]
        fe, fu = np.linalg.eigh(selected.T @ np.diag(eps) @ selected)
        selected = selected @ fu
        expected = base @ selected
        expected_k = selected.T @ exchange @ selected
        expected_t = expected_k / (focc[i, i] + focc[j, j] - fe[:, None] - fe[None, :])
        assert built.pao_ranks[row] == len(eps)
        assert built.pno_ranks[row] == int(np.sum(keep))
        np.testing.assert_allclose(built.occupations[row], values[::-1], atol=3e-12)
        np.testing.assert_allclose(actual @ actual.T @ S, expected @ expected.T @ S, atol=3e-11)
        gauge = expected.T @ S @ actual
        np.testing.assert_allclose(built.integrals[row], gauge.T @ expected_k @ gauge, atol=3e-12)
        np.testing.assert_allclose(built.amplitudes[row], gauge.T @ expected_t @ gauge, atol=3e-12)
        q_full = np.sum(exchange * (2 * amplitude - amplitude.T))
        q_pno = np.sum(expected_k * (2 * expected_t - expected_t.T))
        assert built.pao_pair_energies[row] == pytest.approx(q_full, abs=3e-12)
        assert built.pno_pair_energies[row] == pytest.approx(q_pno, abs=3e-12)
        expected_correction += plan.multiplicities[row] / nk * (1 if i == j else 2) * (q_full - q_pno)
    assert built.pno_correction_per_cell == pytest.approx(expected_correction, abs=3e-12)
    seeded = translation_pair_energy_per_cell(built.spaces, built.integrals, built.amplitudes)
    full_seeded = np.dot(built.spaces.energy_weights, built.pao_pair_energies)
    assert seeded + built.pno_correction_per_cell == pytest.approx(full_seeded, abs=3e-12)
    assert any(a < b for a, b in zip(built.pno_ranks, built.pao_ranks))


@pytest.mark.parametrize("frozen", [False, True])
def test_translation_pno_builder_full_rank_keeps_canonical_hamiltonian(frozen):
    from vibeqc.correlation import mp2_energy
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos
    from vibeqc.dlpno.translation_mp2 import translation_mp2_residual, translation_pair_energy_per_cell

    mesh = (3, 1, 1)
    model, source, active, core, indices = _translation_pno_source(mesh, frozen=frozen)
    built = build_translation_pair_pnos(source, mesh, active, frozen_occupied=core)
    eo, uo = np.linalg.eigh(model.foo[np.ix_(indices, indices)])
    ev, uv = np.linalg.eigh(model.fvv)
    exchange = model.exchange[np.ix_(indices, indices, np.arange(len(ev)), np.arange(len(ev)))]
    canonical = np.einsum("pi,qj,ar,bs,pqab->ijrs", uo, uo, uv, uv, exchange, optimize=True)
    denom = eo[:, None, None, None] + eo[None, :, None, None] - ev[None, None, :, None] - ev[None, None, None, :]
    trial = np.einsum("pi,qj,ar,bs,ijrs->pqab", uo, uo, uv, uv, canonical / denom, optimize=True)
    amplitudes = []
    for (i, j), basis in zip(built.spaces.representative_pairs, built.spaces.pair_coefficients):
        embedding = model.virtual.T @ model.overlap @ basis
        amplitudes.append(embedding.T @ trial[i, j] @ embedding)
    residual = translation_mp2_residual(built.spaces, built.integrals, amplitudes)
    assert max(np.max(np.abs(value)) for value in residual) < 4e-12
    expected = mp2_energy(eo, ev, canonical.transpose(0, 2, 1, 3)).e_corr / model.n_cells
    assert translation_pair_energy_per_cell(built.spaces, built.integrals, amplitudes) == pytest.approx(expected, abs=4e-12)
    assert built.pno_correction_per_cell == pytest.approx(0.0, abs=4e-12)


@pytest.mark.parametrize("fault", ["duplicate", "negative", "float", "empty", "missing", "denominator"])
def test_translation_pno_builder_rejects_bad_domains_and_zero_denominator(fault):
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos
    from vibeqc.periodic.chi.pno import build_translation_pair_plan

    mesh = (1, 1, 1)
    _, source, active, core, _ = _translation_pno_source(mesh)
    plan = build_translation_pair_plan(active.shape[1], mesh)
    domains = [np.arange(source.basis.nbasis) for _ in range(plan.n_representatives)]
    if fault == "duplicate":
        domains[0] = np.array([0, 0])
    elif fault == "negative":
        domains[0] = np.array([-1, 0])
    elif fault == "float":
        domains[0] = np.array([0., 1.])
    elif fault == "empty":
        domains[0] = np.array([], dtype=int)
    elif fault == "missing":
        domains.pop()
    else:
        source.reference.fock = np.zeros_like(source.reference.fock)
    with pytest.raises((ValueError, FloatingPointError)):
        build_translation_pair_pnos(source, mesh, active, domains)


def test_translation_pno_builder_preserves_strict_cutoff_and_minimum_rank():
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos

    mesh = (1, 1, 1)
    _, source, active, _, _ = _translation_pno_source(mesh)
    full = build_translation_pair_pnos(source, mesh, active)
    cutoff = float(full.occupations[0][-1])
    exact = build_translation_pair_pnos(source, mesh, active, tcut_pno=cutoff)
    assert exact.pno_ranks[0] == sum(full.occupations[0] > cutoff)
    minimal = build_translation_pair_pnos(source, mesh, active, tcut_pno=1e100)
    assert set(minimal.pno_ranks) == {1}


@pytest.mark.parametrize("mesh,frozen", [((2, 1, 1), False), ((3, 1, 1), True),
                                        ((2, 2, 1), True)])
def test_translation_native_solver_matches_canonical_mp2(mesh, frozen, record_property):
    from vibeqc.correlation import mp2_energy
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos
    from vibeqc.dlpno.translation_mp2 import translation_mp2_residual
    from vibeqc.dlpno.translation_solver import solve_translation_mp2

    model, source, active, core, indices = _translation_pno_source(mesh, frozen=frozen)
    built = build_translation_pair_pnos(source, mesh, active, frozen_occupied=core)
    saved = tuple(value.copy() for value in (*built.integrals, *built.amplitudes))
    result = solve_translation_mp2(built, conv_tol_energy=1e-13, conv_tol_residual=1e-11)
    assert result.converged and 1 < result.n_iter < 100
    eo, uo = np.linalg.eigh(model.foo[np.ix_(indices, indices)])
    ev, uv = np.linalg.eigh(model.fvv)
    exchange = model.exchange[np.ix_(indices, indices, np.arange(len(ev)), np.arange(len(ev)))]
    canonical = np.einsum("pi,qj,ar,bs,pqab->ijrs", uo, uo, uv, uv, exchange, optimize=True)
    expected = mp2_energy(eo, ev, canonical.transpose(0, 2, 1, 3)).e_corr / model.n_cells
    record_property('e_pair_per_cell', result.e_pair_per_cell)
    record_property('unreduced_e_corr_per_cell', expected)
    record_property('absolute_energy_error', abs(result.e_pair_per_cell-expected))
    record_property('maximum_raw_residual', result.max_residual)
    assert result.e_pair_per_cell == pytest.approx(expected, abs=2e-12)
    assert result.e_corr_per_cell == pytest.approx(expected, abs=2e-12)
    assert result.e_hf_per_cell == source.reference.energy / model.n_cells
    assert result.e_tot_per_cell == pytest.approx(result.e_hf_per_cell + expected, abs=2e-12)
    raw = translation_mp2_residual(built.spaces, built.integrals, result.amplitudes)
    for native, python in zip(result.residuals, raw):
        np.testing.assert_allclose(native, python, atol=3e-15)
    assert result.max_residual == pytest.approx(max(np.max(np.abs(value)) for value in raw), abs=3e-15)
    assert result.trace[0].energy_change is None
    assert abs(result.trace[-1].energy_change) <= 1e-13
    assert result.build_identity == built.identity
    assert len(result.amplitudes) == built.spaces.plan.n_representatives
    for value, original in zip((*built.integrals, *built.amplitudes), saved):
        np.testing.assert_array_equal(value, original)
    with pytest.raises(ValueError):
        result.amplitudes[0].setflags(write=True)


def _translation_unreduced_galerkin_solution(model, built, indices):
    """Solve all ordered placed pairs in dense common virtual coordinates.

    Independent physical permutation matrices construct each projector.
    The complementary-space identity removes redundant zero modes; no
    representative resolver, cross-pair overlap or native residual is used.
    """
    no, nv = len(indices), len(model.fvv)
    projectors = np.full((no, no, nv, nv), np.nan)
    for (i, j), basis in zip(built.spaces.representative_pairs, built.spaces.pair_coefficients):
        frame = model.virtual.T @ model.overlap @ basis
        projector = frame @ frame.T
        for full_po, pv in zip(model.occ_translations, model.vir_translations):
            po = full_po[np.ix_(indices, indices)]
            ti, tj = int(np.argmax(po[:, i])), int(np.argmax(po[:, j]))
            translated = pv @ projector @ pv.T
            for a, b in ((ti, tj), (tj, ti)):
                if np.isfinite(projectors[a, b]).all():
                    np.testing.assert_allclose(projectors[a, b], translated, atol=2e-11)
                projectors[a, b] = translated
    assert np.isfinite(projectors).all()
    exchange = model.exchange[np.ix_(indices, indices, np.arange(nv), np.arange(nv))]
    foo = model.foo[np.ix_(indices, indices)]
    def project(tensor):
        return np.einsum("ijac,ijcd,ijdb->ijab", projectors, tensor, projectors, optimize=True)
    def linear(tensor):
        physical = project(tensor)
        raw = (
            np.einsum("ac,ijcb->ijab", model.fvv, physical)
            + np.einsum("ijac,cb->ijab", physical, model.fvv)
            - np.einsum("ik,kjab->ijab", foo, physical)
            - np.einsum("kj,ikab->ijab", foo, physical)
        )
        return project(raw) + tensor - physical
    shape = (no, no, nv, nv)
    size = no * no * nv * nv
    operator = np.empty((size, size))
    for column in range(size):
        unit = np.zeros(size)
        unit[column] = 1
        operator[:, column] = linear(unit.reshape(shape)).ravel()
    amplitude = np.linalg.solve(operator, -project(exchange).ravel()).reshape(shape)
    np.testing.assert_allclose(linear(amplitude) + project(exchange), 0, atol=3e-13)
    energy = np.sum(exchange * (2 * amplitude - amplitude.swapaxes(2, 3))) / model.n_cells
    return amplitude, energy


@pytest.mark.parametrize("mesh,frozen", [((2, 1, 1), False), ((3, 1, 1), True)])
def test_translation_native_truncated_solver_matches_unreduced_galerkin(mesh, frozen, record_property):
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos
    from vibeqc.dlpno.translation_solver import solve_translation_mp2

    model, source, active, core, indices = _translation_pno_source(mesh, frozen=frozen)
    built = build_translation_pair_pnos(source, mesh, active, frozen_occupied=core, tcut_pno=1e-3)
    assert min(built.pno_ranks) < len(model.fvv)
    result = solve_translation_mp2(built, conv_tol_energy=1e-13, conv_tol_residual=1e-11)
    assert result.converged
    dense, expected = _translation_unreduced_galerkin_solution(model, built, indices)
    record_property('e_pair_per_cell', result.e_pair_per_cell)
    record_property('unreduced_pair_energy_per_cell', expected)
    record_property('absolute_energy_error', abs(result.e_pair_per_cell-expected))
    record_property('maximum_raw_residual', result.max_residual)
    assert result.e_pair_per_cell == pytest.approx(expected, abs=3e-12)
    assert result.e_pno_correction_per_cell == built.pno_correction_per_cell
    assert result.e_corr_per_cell == pytest.approx(expected + built.pno_correction_per_cell, abs=3e-12)
    for (i, j), basis, actual in zip(built.spaces.representative_pairs,
                                    built.spaces.pair_coefficients, result.amplitudes):
        frame = model.virtual.T @ model.overlap @ basis
        np.testing.assert_allclose(actual, frame.T @ dense[i, j] @ frame, atol=3e-12)


def test_translation_native_solver_exhaustion_is_evaluated_snapshot():
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos
    from vibeqc.dlpno.translation_mp2 import translation_mp2_residual, translation_pair_energy_per_cell
    from vibeqc.dlpno.translation_solver import solve_translation_mp2

    _, source, active, core, _ = _translation_pno_source((2, 1, 1))
    built = build_translation_pair_pnos(source, (2, 1, 1), active, frozen_occupied=core)
    first = solve_translation_mp2(built, max_iter=1)
    assert not first.converged and first.n_iter == 1 and first.max_residual > 1e-7
    for actual, expected in zip(first.amplitudes, built.amplitudes):
        np.testing.assert_array_equal(actual, expected)
    second = solve_translation_mp2(built, max_iter=2, conv_tol_energy=0, conv_tol_residual=0, damping=.7)
    assert not second.converged and second.n_iter == 2
    expected_residual = translation_mp2_residual(built.spaces, built.integrals, second.amplitudes)
    for actual, expected in zip(second.residuals, expected_residual):
        np.testing.assert_allclose(actual, expected, atol=3e-15)
    expected_energy = translation_pair_energy_per_cell(built.spaces, built.integrals, second.amplitudes)
    assert second.e_pair_per_cell == pytest.approx(expected_energy, abs=3e-15)
    assert second.trace[-1].energy_change == pytest.approx(expected_energy - first.e_pair_per_cell, abs=3e-15)


def _translation_native_arguments(built):
    spaces = built.spaces
    return dict(
        mesh=list(spaces.mesh), n_bands=spaces.plan.n_bands,
        pair_i=[i for i, _ in spaces.representative_pairs],
        pair_j=[j for _, j in spaces.representative_pairs],
        K=[value.copy() for value in built.integrals],
        F_vv=[spaces.fock_between(pair, pair) for pair in spaces.representative_pairs],
        T=[value.copy() for value in built.amplitudes],
        V=[value.copy() for value in spaces.pair_coefficients],
        F_oo=spaces.occupied_fock.copy(), S_ao=spaces._overlap.copy(),
        energy_weights=spaces.energy_weights.copy(), maximum_iterations=2,
        energy_tolerance=1e-9, residual_tolerance=1e-7, damping=1., symmetry_tolerance=spaces.tolerance,
    )


def test_translation_native_raw_residual_retains_full_virtual_fock():
    from types import SimpleNamespace
    from vibeqc._vibeqc_core import dlpno_translation_mp2_solve
    from vibeqc.dlpno.translation_mp2 import translation_mp2_residual

    model = _translation_mp2_model((3, 1, 1), 2, full=False)
    integrals = tuple(frame.T @ model.exchange[i, j] @ frame
                      for (i, j), frame in zip(model.spaces.representative_pairs, model.frames))
    # A covariant trial, with nonsymmetric off-diagonal exchange blocks.
    amplitudes = tuple(-.2 * value for value in integrals)
    arguments = _translation_native_arguments(SimpleNamespace(
        spaces=model.spaces, integrals=integrals, amplitudes=amplitudes,
    ))
    arguments["maximum_iterations"] = 1
    assert max(np.linalg.norm(f - np.diag(np.diag(f))) for f in arguments["F_vv"]) > 1e-3
    _, native, converged, trace = dlpno_translation_mp2_solve(**arguments)
    expected = translation_mp2_residual(model.spaces, integrals, amplitudes)
    assert not converged and len(trace) == 1
    for actual, wanted in zip(native, expected):
        np.testing.assert_allclose(actual, wanted, atol=3e-14)


@pytest.mark.parametrize("fault", ["shape", "indices", "nonfinite", "complex", "weights",
                                   "denominator", "overlap_covariance", "fock_covariance"])
def test_translation_native_solver_rejects_bad_inputs_without_mutation(fault):
    from copy import deepcopy
    from vibeqc._vibeqc_core import dlpno_translation_mp2_solve
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos

    _, source, active, core, _ = _translation_pno_source((2, 1, 1))
    built = build_translation_pair_pnos(source, (2, 1, 1), active, frozen_occupied=core)
    arguments = _translation_native_arguments(built)
    if fault == "shape":
        arguments["K"][0] = arguments["K"][0][:, :-1]
    elif fault == "indices":
        arguments["pair_i"][0] = -1
    elif fault == "nonfinite":
        arguments["T"][0][0, 0] = np.nan
    elif fault == "complex":
        arguments["K"][0] = arguments["K"][0].astype(complex) + 1j
    elif fault == "weights":
        arguments["energy_weights"][0] = 3
    elif fault == "denominator":
        for block in arguments["F_vv"]:
            block[:] = 0
        arguments["F_oo"][:] = 0
    elif fault == "overlap_covariance":
        arguments["S_ao"][0, 0] += .1
    else:
        arguments["F_oo"][0, 0] += .1
    saved = deepcopy(arguments)
    with pytest.raises((ValueError, RuntimeError)):
        dlpno_translation_mp2_solve(**arguments)
    for key, value in arguments.items():
        if isinstance(value, list):
            for actual, expected in zip(value, saved[key]):
                np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_array_equal(value, saved[key])


@pytest.mark.parametrize("options", [{"max_iter": 0}, {"max_iter": True}, {"max_iter": 1.5},
                                     {"conv_tol_energy": np.nan}, {"conv_tol_residual": -1},
                                     {"damping": 0}, {"damping": 1.01}])
def test_translation_solver_rejects_invalid_controls(options):
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos
    from vibeqc.dlpno.translation_solver import solve_translation_mp2

    _, source, active, _, _ = _translation_pno_source((1, 1, 1))
    built = build_translation_pair_pnos(source, (1, 1, 1), active)
    with pytest.raises(ValueError):
        solve_translation_mp2(built, **options)


@pytest.mark.parametrize("mesh,frozen,cutoff", [((2, 1, 1), False, 0.),
                                               ((3, 1, 1), True, 1e-3),
                                               ((2, 2, 1), True, 1e-3)])
def test_translation_prepared_compact_dense_and_ao_backends_match(mesh, frozen, cutoff):
    from dataclasses import replace
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos
    from vibeqc.dlpno.translation_solver import solve_translation_mp2
    from vibeqc.dlpno.translation_transport import prepare_translation_transport

    _, source, active, core, _ = _translation_pno_source(mesh, frozen=frozen)
    built = build_translation_pair_pnos(source, mesh, active, frozen_occupied=core, tcut_pno=cutoff)
    compact = prepare_translation_transport(built)
    assert compact.compact
    assert compact.n_couplings == 2 * built.spaces.plan.n_occupied * built.spaces.plan.n_representatives
    assert compact.n_links < compact.n_couplings
    assert compact.required_bytes == compact.retained_bytes + compact.workspace_bytes
    dense_build = replace(built, projected_pair_bases=None)
    dense = prepare_translation_transport(dense_build)
    assert not dense.compact
    assert dense.n_links == compact.n_links
    controls = dict(max_iter=4, conv_tol_energy=0., conv_tol_residual=0., damping=.8)
    reference = solve_translation_mp2(built, transport_backend="ao", **controls)
    for candidate, transport in ((built, compact), (dense_build, dense)):
        actual = solve_translation_mp2(candidate, transport=transport, **controls)
        assert actual.n_iter == reference.n_iter == 4
        assert not actual.converged and not reference.converged
        for actual_blocks, expected_blocks in ((actual.amplitudes, reference.amplitudes),
                                               (actual.residuals, reference.residuals)):
            for value, expected in zip(actual_blocks, expected_blocks):
                np.testing.assert_allclose(value, expected, atol=3e-13, rtol=2e-12)
        assert actual.e_corr_per_cell == pytest.approx(reference.e_corr_per_cell, abs=3e-13)
        for a, b in zip(actual.trace, reference.trace):
            assert a.energy_per_cell == pytest.approx(b.energy_per_cell, abs=3e-13)
            assert a.maximum_residual == pytest.approx(b.maximum_residual, abs=3e-13)
        assert actual.provenance["transport_identity"] == transport.identity


def test_translation_prepared_reuse_performs_no_ao_contractions(monkeypatch):
    from vibeqc import _vibeqc_core
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos
    from vibeqc.dlpno.translation_solver import solve_translation_mp2
    from vibeqc.dlpno.translation_transport import prepare_translation_transport

    _, source, active, core, _ = _translation_pno_source((3, 1, 1), frozen=True)
    built = build_translation_pair_pnos(source, (3, 1, 1), active, frozen_occupied=core)
    transport = prepare_translation_transport(built)
    expected = solve_translation_mp2(built, transport_backend="ao")
    def forbidden(*args, **kwargs):
        raise AssertionError("prepared reuse must not contract or translate AO arrays")
    for method in ("fock_between", "overlap_between", "coefficients"):
        monkeypatch.setattr(built.spaces, method, forbidden)
    monkeypatch.setattr(_vibeqc_core, "dlpno_translation_mp2_solve", forbidden)
    monkeypatch.setattr(_vibeqc_core, "dlpno_prepare_translation_transport", forbidden)
    actual = solve_translation_mp2(built, transport=transport)
    assert actual.converged and expected.converged
    assert actual.e_corr_per_cell == pytest.approx(expected.e_corr_per_cell, abs=3e-13)
    assert not hasattr(transport._payload, "overlap")
    assert not hasattr(transport._payload, "coefficients")
    with pytest.raises(TypeError):
        type(transport._payload)()
    with pytest.raises(AttributeError):
        transport._payload.compact = False


def test_translation_prepared_budget_is_explicit_and_preserves_inputs():
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos
    from vibeqc.dlpno.translation_solver import solve_translation_mp2
    from vibeqc.dlpno.translation_transport import prepare_translation_transport

    _, source, active, core, _ = _translation_pno_source((2, 1, 1))
    built = build_translation_pair_pnos(source, (2, 1, 1), active, frozen_occupied=core)
    snapshot = tuple(value.copy() for value in built.spaces.pair_coefficients)
    transport = prepare_translation_transport(built)
    exact = prepare_translation_transport(built, max_bytes=transport.required_bytes)
    assert exact.required_bytes == transport.required_bytes
    with pytest.raises(ValueError, match="before link allocation"):
        prepare_translation_transport(built, max_bytes=transport.required_bytes - 1)
    with pytest.raises(ValueError, match="before link allocation"):
        prepare_translation_transport(built, max_bytes=1)
    with pytest.raises(ValueError, match="exceeds max_transport_bytes"):
        solve_translation_mp2(built, transport=transport, max_transport_bytes=transport.required_bytes - 1)
    for invalid in (0, True, 1.5):
        with pytest.raises(ValueError, match="positive unsigned"):
            prepare_translation_transport(built, max_bytes=invalid)
        for backend in ("prepared", "ao"):
            with pytest.raises(ValueError, match="positive unsigned"):
                solve_translation_mp2(built, transport_backend=backend, max_transport_bytes=invalid)
        with pytest.raises(ValueError, match="positive unsigned"):
            solve_translation_mp2(built, transport=transport, max_transport_bytes=invalid)
    for value, saved in zip(built.spaces.pair_coefficients, snapshot):
        np.testing.assert_array_equal(value, saved)


@pytest.mark.parametrize("fault", ["occupied_tail", "local_coefficients", "domain"])
def test_translation_prepared_rejects_inconsistent_projected_metadata(fault):
    from dataclasses import replace
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos
    from vibeqc.dlpno.translation_transport import prepare_translation_transport

    _, source, active, core, _ = _translation_pno_source((2, 1, 1), frozen=True)
    built = build_translation_pair_pnos(source, (2, 1, 1), active, frozen_occupied=core)
    projected = list(built.projected_pair_bases)
    value = projected[0]
    if fault == "occupied_tail":
        changed = value.occupied_coefficients.copy()
        changed[0, 0] += .01
        projected[0] = replace(value, occupied_coefficients=changed)
    elif fault == "local_coefficients":
        changed = value.local_coefficients.copy()
        changed[0, 0] += .01
        projected[0] = replace(value, local_coefficients=changed)
    else:
        changed = value.ao_indices.copy()
        changed[0] = changed[1]
        projected[0] = replace(value, ao_indices=changed)
    with pytest.raises(ValueError):
        prepare_translation_transport(replace(built, projected_pair_bases=tuple(projected)))
    assert prepare_translation_transport(built).compact


def test_translation_prepared_transport_rejects_another_calculation():
    from dataclasses import replace
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos
    from vibeqc.dlpno.translation_solver import solve_translation_mp2
    from vibeqc.dlpno.translation_transport import prepare_translation_transport

    _, source, active, core, _ = _translation_pno_source((2, 1, 1))
    built = build_translation_pair_pnos(source, (2, 1, 1), active, frozen_occupied=core)
    transport = prepare_translation_transport(built)
    other = build_translation_pair_pnos(source, (2, 1, 1), active, frozen_occupied=core, tcut_pno=1e-3)
    with pytest.raises(ValueError, match="does not match"):
        solve_translation_mp2(other, transport=transport)
    with pytest.raises(ValueError, match="does not match"):
        solve_translation_mp2(replace(built, projected_pair_bases=None), transport=transport)
    with pytest.raises(ValueError, match="requires"):
        solve_translation_mp2(built, transport=transport, transport_backend="ao")


def test_translation_prepared_stabilizes_ill_conditioned_frozen_domain_links(record_property):
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos
    from vibeqc.dlpno.translation_solver import solve_translation_mp2
    from vibeqc.dlpno.translation_transport import prepare_translation_transport
    from vibeqc.periodic.chi.pno import build_translation_pair_plan

    mesh = (2, 1, 1)
    model, source, active, core, indices = _translation_pno_source(mesh, frozen=True)
    plan = build_translation_pair_plan(1, mesh)
    local = len(model.overlap) // 2
    domains = [np.array(sorted({ao for cell in (0, int(np.ravel_multi_index(tuple(shift), mesh)))
                               for ao in range(cell * local, (cell + 1) * local)}))
               for shift in plan.displacements]
    built = build_translation_pair_pnos(source, mesh, active, domains, frozen_occupied=core, tcut_pno=0.)
    assert built.pno_ranks == built.pao_ranks == (4, 4)
    # A near-null projected PAO direction yields large A/B coefficients. The
    # unmodified two-term overlap would amplify occupied Gram roundoff above
    # the 1e-10 admission tolerance; no orbital may be dropped to hide that.
    assert max(np.max(np.abs(value.local_coefficients)) for value in built.projected_pair_bases) > 100
    transport = prepare_translation_transport(built)
    assert transport.compact and 0 < transport.stable_links <= transport.n_links
    assert transport.provenance["stable_selected_column_links"] == transport.stable_links
    controls = dict(conv_tol_energy=1e-13, conv_tol_residual=1e-11)
    actual = solve_translation_mp2(built, transport=transport, **controls)
    reference = solve_translation_mp2(built, transport_backend="ao", **controls)
    assert actual.converged and reference.converged
    dense, energy = _translation_unreduced_galerkin_solution(model, built, indices)
    record_property('e_pair_per_cell', actual.e_pair_per_cell)
    record_property('unreduced_pair_energy_per_cell', energy)
    record_property('absolute_energy_error', abs(actual.e_pair_per_cell-energy))
    record_property('maximum_raw_residual', actual.max_residual)
    record_property('stable_selected_column_links', transport.stable_links)
    assert actual.e_pair_per_cell == pytest.approx(energy, abs=3e-12)
    assert actual.e_pair_per_cell == pytest.approx(reference.e_pair_per_cell, abs=3e-12)
    for (i, j), basis, amplitude, residual, reference_residual in zip(
        built.spaces.representative_pairs, built.spaces.pair_coefficients,
        actual.amplitudes, actual.residuals, reference.residuals,
    ):
        frame = model.virtual.T @ model.overlap @ basis
        np.testing.assert_allclose(amplitude, frame.T @ dense[i, j] @ frame, atol=3e-12)
        np.testing.assert_allclose(residual, reference_residual, atol=3e-12)


def _translation_cyclic_source(mesh, *, frozen=False):
    from copy import copy
    from dataclasses import replace
    from vibeqc.dlpno.translation_operators import CyclicAOOperator

    model, dense, active, core, indices = _translation_pno_source(mesh, frozen=frozen)
    local = len(model.overlap) // model.n_cells
    def operator(matrix):
        return CyclicAOOperator(np.array([matrix[:local, cell * local:(cell + 1) * local]
                                           for cell in range(model.n_cells)]), mesh)
    reference = copy(dense.reference)
    reference.fock = operator(model.fock)
    cyclic = replace(dense, overlap=operator(model.overlap), reference=reference)
    return model, dense, cyclic, active, core, indices


@pytest.mark.parametrize("mesh,frozen,cutoff", [((2, 1, 1), False, 0.),
                                               ((3, 1, 1), True, 1e-3),
                                               ((2, 2, 1), True, 1e-3)])
def test_translation_block_metric_solver_matches_dense_finite_hamiltonian(mesh, frozen, cutoff, monkeypatch):
    from vibeqc import _vibeqc_core
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos
    from vibeqc.dlpno.translation_solver import solve_translation_mp2
    from vibeqc.dlpno.translation_transport import prepare_translation_transport

    model, dense, cyclic, active, core, _ = _translation_cyclic_source(mesh, frozen=frozen)
    controls = dict(conv_tol_energy=1e-13, conv_tol_residual=1e-11)
    original = build_translation_pair_pnos(dense, mesh, active, frozen_occupied=core, tcut_pno=cutoff)
    reference = solve_translation_mp2(original, transport_backend="ao", **controls)
    def forbidden(*args, **kwargs):
        raise AssertionError("cyclic metric path must not invoke dense native preparation or solve")
    monkeypatch.setattr(_vibeqc_core, "dlpno_prepare_translation_transport", forbidden)
    monkeypatch.setattr(_vibeqc_core, "dlpno_translation_mp2_solve", forbidden)
    built = build_translation_pair_pnos(cyclic, mesh, active, frozen_occupied=core, tcut_pno=cutoff)
    transport = prepare_translation_transport(built)
    actual = solve_translation_mp2(built, transport=transport, **controls)
    assert reference.converged and actual.converged
    assert transport.metric_backend == "cyclic-blocks" and transport._payload.cyclic_metric
    assert actual.provenance["transport_preparation"]["metric_backend"] == "cyclic-blocks"
    assert actual.provenance["input_snapshot"]["overlap"]["kind"] == "cyclic-ao-home-row-v1"
    assert built.pao_ranks == original.pao_ranks and built.pno_ranks == original.pno_ranks
    assert actual.e_pair_per_cell == pytest.approx(reference.e_pair_per_cell, abs=4e-12)
    assert actual.e_corr_per_cell == pytest.approx(reference.e_corr_per_cell, abs=4e-12)
    for actual_basis, reference_basis, amplitude, reference_amplitude in zip(
        built.spaces.pair_coefficients, original.spaces.pair_coefficients,
        actual.amplitudes, reference.amplitudes,
    ):
        gauge = reference_basis.T @ model.overlap @ actual_basis
        np.testing.assert_allclose(amplitude, gauge.T @ reference_amplitude @ gauge, atol=4e-12)
    with pytest.raises(ValueError, match="explicit dense reference"):
        solve_translation_mp2(built, transport_backend="ao")


def test_translation_block_metric_ill_conditioned_domain_matches_unreduced_oracle(record_property):
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos
    from vibeqc.dlpno.translation_solver import solve_translation_mp2
    from vibeqc.dlpno.translation_transport import prepare_translation_transport
    from vibeqc.periodic.chi.pno import build_translation_pair_plan

    mesh = (2, 1, 1)
    model, _, cyclic, active, core, indices = _translation_cyclic_source(mesh, frozen=True)
    local = len(model.overlap) // 2
    plan = build_translation_pair_plan(1, mesh)
    domains = [np.array(sorted({ao for cell in (0, int(np.ravel_multi_index(tuple(shift), mesh)))
                               for ao in range(cell * local, (cell + 1) * local)}))
               for shift in plan.displacements]
    built = build_translation_pair_pnos(cyclic, mesh, active, domains, frozen_occupied=core)
    transport = prepare_translation_transport(built)
    assert transport.metric_backend == "cyclic-blocks" and transport.stable_links > 0
    assert built.pno_ranks == built.pao_ranks == (4, 4)
    solved = solve_translation_mp2(built, transport=transport, conv_tol_energy=1e-13, conv_tol_residual=1e-11)
    expected_t, expected_e = _translation_unreduced_galerkin_solution(model, built, indices)
    assert solved.converged
    assert solved.e_pair_per_cell == pytest.approx(expected_e, abs=4e-12)
    record_property("e_pair_per_cell", solved.e_pair_per_cell)
    record_property("unreduced_pair_energy_per_cell", expected_e)
    record_property("absolute_energy_error", abs(solved.e_pair_per_cell - expected_e))
    record_property("maximum_raw_residual", solved.max_residual)
    for (i, j), basis, actual in zip(built.spaces.representative_pairs, built.spaces.pair_coefficients, solved.amplitudes):
        frame = model.virtual.T @ model.overlap @ basis
        np.testing.assert_allclose(actual, frame.T @ expected_t[i, j] @ frame, atol=4e-12)


def _translation_native_block_preparation_arguments(built):
    from vibeqc.dlpno.translation_operators import CyclicAOOperator

    spaces = built.spaces
    assert type(spaces._overlap) is CyclicAOOperator
    return dict(
        mesh=list(spaces.mesh), n_bands=spaces.plan.n_bands,
        pair_i=[i for i, _ in spaces.representative_pairs],
        pair_j=[j for _, j in spaces.representative_pairs],
        F_vv=[spaces.fock_between(pair, pair) for pair in spaces.representative_pairs],
        V=[value.copy() for value in spaces.pair_coefficients],
        F_oo=spaces.occupied_fock.copy(), S_home_blocks=[value.copy() for value in spaces._overlap.blocks],
        energy_weights=spaces.energy_weights.copy(), symmetry_tolerance=spaces.tolerance,
        maximum_bytes=256 * 1024**2,
    )


@pytest.mark.parametrize("fault", ["missing", "shape", "nonfinite", "complex", "symmetry", "nonpositive_symbol"])
def test_translation_block_metric_native_rejects_invalid_symbols(fault):
    from copy import deepcopy
    from vibeqc._vibeqc_core import dlpno_prepare_translation_transport_blocks
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos

    mesh = (3, 1, 1)
    _, _, source, active, core, _ = _translation_cyclic_source(mesh, frozen=True)
    built = build_translation_pair_pnos(source, mesh, active, frozen_occupied=core)
    arguments = _translation_native_block_preparation_arguments(built)
    blocks = arguments["S_home_blocks"]
    if fault == "missing":
        blocks.pop()
    elif fault == "shape":
        blocks[0] = blocks[0][:-1]
    elif fault == "nonfinite":
        blocks[0][0, 0] = np.inf
    elif fault == "complex":
        blocks[0] = blocks[0].astype(complex) + .1j
    elif fault == "symmetry":
        blocks[1][0, 1] += .01
    else:
        # Gamma is positive (3.4 I), but the nonzero-k symbols are negative
        # (-0.2 I). Checking only the home block or Gamma would accept it.
        blocks[:] = [scale * np.eye(len(blocks[0])) for scale in (1., 1.2, 1.2)]
    saved = deepcopy(arguments)
    with pytest.raises(ValueError, match="positive definite" if fault == "nonpositive_symbol" else None):
        dlpno_prepare_translation_transport_blocks(**arguments)
    for key, value in arguments.items():
        if isinstance(value, list):
            for actual, expected in zip(value, saved[key]):
                np.testing.assert_array_equal(actual, expected)
        else:
            np.testing.assert_array_equal(value, saved[key])


def test_translation_block_metric_budget_uses_primitive_symbol_workspace():
    from dataclasses import replace
    from vibeqc.dlpno.translation import TranslationPairSpaces
    from vibeqc.dlpno.translation_builder import build_translation_pair_pnos
    from vibeqc.dlpno.translation_transport import prepare_translation_transport

    mesh = (3, 1, 1)
    _, source, cyclic, active, core, _ = _translation_cyclic_source(mesh, frozen=True)
    dense_build = build_translation_pair_pnos(source, mesh, active, frozen_occupied=core)
    spaces = TranslationPairSpaces(mesh, cyclic.overlap, cyclic.reference.fock, active,
                                   dense_build.spaces.pair_coefficients, frozen_occupied=core)
    block_build = replace(dense_build, spaces=spaces)
    dense = prepare_translation_transport(dense_build)
    block = prepare_translation_transport(block_build)
    assert block.retained_bytes == dense.retained_bytes
    assert block.required_bytes < dense.required_bytes
    assert prepare_translation_transport(block_build, max_bytes=block.required_bytes).required_bytes == block.required_bytes
    with pytest.raises(ValueError, match="before link allocation"):
        prepare_translation_transport(dense_build, max_bytes=block.required_bytes)
    generic = prepare_translation_transport(replace(block_build, projected_pair_bases=None))
    assert not generic.compact and generic.metric_backend == "cyclic-blocks"
