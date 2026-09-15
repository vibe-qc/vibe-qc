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
    def h2_chain(self):
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
