"""DLPNO-CCSD pilot correctness gates (M3a).

Two-layer validation:

1. **The anchor** (`dlpno._ccsd_ref`, spin-orbital SGB-1991 CCSD) is
   pinned against exact references: CCSD ≡ FCI for two-electron
   systems (H2, difference bounded by the independently measured RI-fit
   error), and for H2O/STO-3G the anchor must sit just above the
   full-space CASCI (=FCI) correlation energy by the expected
   triples-and-higher margin.
2. **The pilot** (`dlpno.ccsd`, subspace-projected DLPNO-CCSD) must
   reproduce the anchor exactly in the untruncated limit — canonical
   and Boys-localised occupieds — and meet measured recovery tiers
   with PNO truncation on a DZ basis (slow lane).
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from vibeqc import BasisSet, MP2Options, RHFOptions, run_mp2, run_rhf
from vibeqc._vibeqc_core import Atom, Molecule
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno._ccsd_ref import run_ref_ccsd
from vibeqc.dlpno.ccsd import (
    DLPNOCCSDPilotOptions as _DLPNOCCSDPilotOptions,
    run_dlpno_ccsd_pilot,
)

ANGSTROM_TO_BOHR = 1.8897259886
AUX = "def2-svp-rifit"
MICRO_HA = 1e-6

# Keep the all-electron, pre-#448 pilot settings behind the historical
# exactness/recovery anchors in this module.
DLPNOCCSDPilotOptions = partial(
    _DLPNOCCSDPilotOptions,
    n_frozen=0,
    tcut_pno=1e-8,
    tcut_mkn=1e-3,
)

H2_ATOMS = [(1, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 1.4])]
H2O_ATOMS = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
    (1, [0.0, -0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
]

# Full-space CASCI(7,10)/STO-3G correlation energy for the geometry above
# (vibe-qc direct CAS-CI engine, 2026-06-10) — FCI for this basis.
E_FCI_CORR_H2O_STO3G = -0.055587908

NO_TRUNCATION = dict(tcut_pno=0.0, tcut_mkn=0.0)


def _system(atoms, basis_name):
    mol = Molecule([Atom(z, p) for z, p in atoms], charge=0, multiplicity=1)
    basis = BasisSet(mol, basis_name)
    opts = RHFOptions()
    opts.max_iter = 100
    rhf = run_rhf(mol, basis, opts)
    assert rhf.converged
    df = DensityFitting(basis, BasisSet(mol, AUX), aux_basis_name=AUX)
    return mol, basis, rhf, df


def _ref(rhf, df, n_occ):
    C = np.asarray(rhf.mo_coeffs)
    F = np.asarray(rhf.fock)
    return run_ref_ccsd(
        C.T @ F @ C, df.mo_transform(C, C), n_occ=n_occ, e_hf=rhf.energy
    )


@pytest.fixture(scope="module")
def h2():
    return _system(H2_ATOMS, "sto-3g")


@pytest.fixture(scope="module")
def h2o():
    return _system(H2O_ATOMS, "sto-3g")


class TestAnchor:
    """The spin-orbital reference kernel against exact references."""

    def test_h2_ccsd_equals_fci_within_df_error(self, h2, tmp_path, monkeypatch):
        mol, basis, rhf, df = h2
        ref = _ref(rhf, df, 1)
        assert ref.converged

        # Exact FCI via the runner (conventional integrals). Run in
        # tmp_path: run_job writes .out/.system artifacts into the cwd.
        monkeypatch.chdir(tmp_path)
        from vibeqc import run_job

        fci = run_job(mol, basis="sto-3g", method="fci", output="h2_fci_anchor")
        e_fci_corr = fci.energy - rhf.energy

        # The only difference CCSD(=FCI for 2e) vs the DF kernel is the
        # RI fit; bound it by the independently measured DF error at the
        # MP2 level on the same system, with headroom.
        m_conv_options = MP2Options()
        m_conv_options.n_frozen_core = 0
        m_conv = run_mp2(mol, basis, rhf, m_conv_options)
        m_df_opts = MP2Options()
        m_df_opts.n_frozen_core = 0
        m_df_opts.density_fit = True
        m_df_opts.aux_basis = AUX
        m_df = run_mp2(mol, basis, rhf, m_df_opts)
        df_error_scale = abs(m_conv.e_correlation - m_df.e_correlation)

        assert abs(ref.e_corr - e_fci_corr) < 5.0 * max(df_error_scale, MICRO_HA)

    def test_h2o_anchor_sits_above_fci_by_triples_margin(self, h2o):
        """CCSD recovers slightly less correlation than FCI — never more.

        This is the test that catches catastrophically wrong CCSD
        equations (the experimental C++ kernel overshoots FCI by 41 mHa
        on this system as of 2026-06-10).
        """
        mol, basis, rhf, df = h2o
        ref = _ref(rhf, df, 5)
        assert ref.converged
        # Less correlation than FCI...
        assert ref.e_corr > E_FCI_CORR_H2O_STO3G
        # ...but by no more than ~1% (triples+quadruples for H2O) plus
        # DF-fit headroom.
        assert ref.e_corr < E_FCI_CORR_H2O_STO3G * 0.985


class TestPilotExactnessLimit:
    """Untruncated DLPNO-CCSD pilot ≡ the anchor."""

    def test_rung1_canonical(self, h2o):
        mol, basis, rhf, df = h2o
        ref = _ref(rhf, df, 5)
        r = run_dlpno_ccsd_pilot(
            mol, basis, rhf, df,
            DLPNOCCSDPilotOptions(localise="none", **NO_TRUNCATION),
        )
        assert r.converged
        assert abs(r.e_corr - ref.e_corr) < 1.0 * MICRO_HA

    def test_rung2_boys_localised(self, h2o):
        mol, basis, rhf, df = h2o
        ref = _ref(rhf, df, 5)
        r = run_dlpno_ccsd_pilot(
            mol, basis, rhf, df,
            DLPNOCCSDPilotOptions(localise="boys", **NO_TRUNCATION),
        )
        assert r.converged
        assert abs(r.e_corr - ref.e_corr) < 1.0 * MICRO_HA
        # Localised occupieds genuinely exercise the projections.
        assert r.t1_norm > 1e-4

    def test_diagonal_pairs_present(self, h2o):
        mol, basis, rhf, df = h2o
        r = run_dlpno_ccsd_pilot(
            mol, basis, rhf, df,
            DLPNOCCSDPilotOptions(localise="boys", **NO_TRUNCATION),
        )
        n_occ = mol.n_electrons() // 2
        assert r.n_pairs == n_occ * (n_occ + 1) // 2


@pytest.mark.slow
class TestPilotTruncationDZ:
    """Recovery tiers on H2O/def2-SVP (O(N^6) pilot — slow lane)."""

    @pytest.fixture(scope="class")
    def h2o_dz(self):
        return _system(H2O_ATOMS, "def2-svp")

    def test_exactness_limit_dz(self, h2o_dz):
        mol, basis, rhf, df = h2o_dz
        ref = _ref(rhf, df, 5)
        r = run_dlpno_ccsd_pilot(
            mol, basis, rhf, df,
            DLPNOCCSDPilotOptions(localise="boys", **NO_TRUNCATION),
        )
        assert r.converged
        assert abs(r.e_corr - ref.e_corr) < 1.0 * MICRO_HA

    def test_default_pno_truncation_recovery(self, h2o_dz):
        mol, basis, rhf, df = h2o_dz
        ref = _ref(rhf, df, 5)
        r = run_dlpno_ccsd_pilot(
            mol, basis, rhf, df,
            DLPNOCCSDPilotOptions(localise="boys", tcut_pno=1e-8, tcut_mkn=1e-3),
        )
        assert r.converged
        recovery = r.e_corr / ref.e_corr
        # Measured 2026-06-10: 99.858% — assert the tier with margin.
        assert recovery > 0.998, f"recovery {recovery:.5%}"
        assert recovery < 1.003, f"recovery {recovery:.5%} (overshoot)"
        # Real compression at default thresholds.
        n_vir = basis.nbasis - 5
        avg = sum(r.pno_per_pair.values()) / len(r.pno_per_pair)
        assert avg < 0.9 * n_vir


class TestTriples:
    """(T) correction gates (M4): null test, FCI bounds, pilot exactness."""

    def test_h2_triples_vanish(self, h2):
        """Two electrons cannot make triple excitations: (T) ≡ 0."""
        mol, basis, rhf, df = h2
        C = np.asarray(rhf.mo_coeffs)
        F = np.asarray(rhf.fock)
        ref = run_ref_ccsd(
            C.T @ F @ C, df.mo_transform(C, C), n_occ=1, e_hf=rhf.energy,
            compute_triples=True,
        )
        assert ref.e_t == 0.0

    def test_h2o_ccsd_t_improves_toward_fci_without_overshoot(self, h2o):
        mol, basis, rhf, df = h2o
        C = np.asarray(rhf.mo_coeffs)
        F = np.asarray(rhf.fock)
        ref = run_ref_ccsd(
            C.T @ F @ C, df.mo_transform(C, C), n_occ=5, e_hf=rhf.energy,
            compute_triples=True,
        )
        assert ref.converged
        assert ref.e_t < 0.0
        e_ccsd_t = ref.e_corr + ref.e_t
        # (T) must move CCSD toward FCI...
        assert abs(e_ccsd_t - E_FCI_CORR_H2O_STO3G) < abs(
            ref.e_corr - E_FCI_CORR_H2O_STO3G
        )
        # ...without catastrophic overshoot (small non-variational
        # excursions below FCI are physical; 20 µHa is generous here).
        assert e_ccsd_t > E_FCI_CORR_H2O_STO3G - 20.0 * MICRO_HA

    def test_pilot_triples_exact_in_untruncated_limit(self, h2o):
        """Localised DLPNO pilot + canonicalised (T) ≡ canonical (T)."""
        mol, basis, rhf, df = h2o
        C = np.asarray(rhf.mo_coeffs)
        F = np.asarray(rhf.fock)
        ref = run_ref_ccsd(
            C.T @ F @ C, df.mo_transform(C, C), n_occ=5, e_hf=rhf.energy,
            compute_triples=True,
        )
        r = run_dlpno_ccsd_pilot(
            mol, basis, rhf, df,
            DLPNOCCSDPilotOptions(
                localise="boys", compute_triples=True, **NO_TRUNCATION
            ),
        )
        assert r.converged
        assert abs(r.e_corr - ref.e_corr) < 1.0 * MICRO_HA
        assert abs(r.e_t - ref.e_t) < 0.05 * MICRO_HA
        assert r.e_total == pytest.approx(
            r.e_hf + r.e_corr + r.e_t, abs=1e-12
        )

    def test_truncated_triples_sane(self, h2o):
        """(T) on truncated-PNO amplitudes stays negative and close."""
        mol, basis, rhf, df = h2o
        r_full = run_dlpno_ccsd_pilot(
            mol, basis, rhf, df,
            DLPNOCCSDPilotOptions(
                localise="boys", compute_triples=True, **NO_TRUNCATION
            ),
        )
        r_trunc = run_dlpno_ccsd_pilot(
            mol, basis, rhf, df,
            DLPNOCCSDPilotOptions(
                localise="boys", compute_triples=True,
                tcut_pno=1e-8, tcut_mkn=1e-3,
            ),
        )
        assert r_trunc.converged
        assert r_trunc.e_t < 0.0
        assert abs(r_trunc.e_t - r_full.e_t) < 0.3 * abs(r_full.e_t)
