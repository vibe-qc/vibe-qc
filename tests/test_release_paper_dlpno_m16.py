"""Release-paper M16 DLPNO n-butane/cc-pVDZ compatibility tests.

The ORCA comparison for M16 is an out-of-process reference calculation with
``TightPNO NoFrozenCore``.  Its vibe-qc comparison predates the coordinated
#140/#448 frozen-core and threshold-default change.  These tests preserve that
historical recipe explicitly, separately from the current project-default
contract, so the release paper cannot accidentally report the DLPNO
approximation gap as canonical microhartree parity.
"""

from __future__ import annotations

import math

import pytest

from vibeqc import BasisSet, RHFOptions, run_job, run_rhf
from vibeqc._vibeqc_core import Atom, Molecule
from vibeqc.density_fitting import DensityFitting, default_aux_basis_for
from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions, LocalCCSDResult
from vibeqc.dlpno.mp2 import DLPNOMP2Options, run_dlpno_mp2


ANG2BOHR = 1.0 / 0.529177210903
ORCA_M16_RHF = -155.63996717880431
ORCA_M16_DLPNO_MP2_CORR = -0.710031986501
ORCA_M16_DLPNO_CCSD_CORR = -0.747704054
ORCA_M16_DLPNO_TRIPLES = -0.031184442
VIBEQC_M16_DLPNO_MP2_PRE_140_448_CORR = -0.7062699636
VIBEQC_M16_DLPNO_MP2_ORCA_LIKE_CORR = -0.7075590024


def _n_butane() -> Molecule:
    """Return the release-paper n-butane geometry in bohr."""
    d_cc, d_ch = 1.54, 1.09
    th = math.radians(109.47 / 2.0)
    dx, dz = d_cc * math.sin(th), d_cc * math.cos(th)
    carbons = [(i * dx, 0.0, (i % 2) * dz) for i in range(4)]
    atoms = [(6, p) for p in carbons]
    hy = d_ch * math.sin(math.radians(54.75))
    hz = d_ch * math.cos(math.radians(54.75))
    for i, (x, _y, z) in enumerate(carbons):
        up = i % 2 == 0
        if i in (0, 3):
            sgn = -1.0 if i == 0 else 1.0
            atoms += [
                (1, (x - sgn * 0.7 * d_ch, 0.0, z)),
                (1, (x + 0.3 * d_ch, hy, z - (hz if up else -hz))),
                (1, (x + 0.3 * d_ch, -hy, z - (hz if up else -hz))),
            ]
        else:
            atoms += [
                (1, (x, hy, z + (hz if up else -hz))),
                (1, (x, -hy, z + (hz if up else -hz))),
            ]
    return Molecule(
        [Atom(z, [c * ANG2BOHR for c in xyz]) for z, xyz in atoms],
        charge=0,
        multiplicity=1,
    )


def _pre_140_448_dlpno_mp2_options() -> DLPNOMP2Options:
    """Return the exact all-electron MP2 recipe behind the retained M16 row."""
    return DLPNOMP2Options(
        n_frozen=0,
        tcut_pno=1e-8,
        tcut_pno_weak=1e-7,
        tcut_mkn=1e-3,
        tcut_pairs=1e-6,
        tcut_pairs_weak=1e-4,
        # Retained rows predate the #65 ruling that made "mp2" the
        # default pair density, so the convention is pinned here too:
        # this helper must reproduce the recorded row exactly, and a
        # tcut_pno only means what the density it was cut against
        # means. Under the current default this row moves by
        # -0.878 mHa.
        pno_norm="legacy",
    )


def _pre_140_448_local_ccsd_options() -> LocalCCSDOptions:
    """Return the exact all-electron local-CCSD recipe behind the M16 row."""
    return LocalCCSDOptions(
        n_frozen=0,
        tcut_pno=1e-7,
        tcut_mkn=0.0,
        tcut_pairs=1e-4,
        residual_domain="pair",
        compute_triples=True,
        pno_norm="legacy",  # retained row predates #65; see the MP2 helper
    )


def test_dlpno_project_defaults_match_published_normalpno_contract():
    """The new default contract is published NormalPNO, not a legacy energy."""
    mp2_opts = DLPNOMP2Options()
    ccsd_opts = LocalCCSDOptions()

    assert mp2_opts.tcut_pno == pytest.approx(3.33e-7)
    assert mp2_opts.tcut_pno_weak == pytest.approx(3.33e-6)
    assert mp2_opts.tcut_pairs == pytest.approx(1e-4)
    assert mp2_opts.tcut_pairs_weak == pytest.approx(1e-4)
    assert mp2_opts.tcut_mkn == pytest.approx(1e-3)
    assert ccsd_opts.tcut_pno == pytest.approx(3.33e-7)
    assert ccsd_opts.tcut_pairs == pytest.approx(1e-4)
    assert ccsd_opts.tcut_mkn == pytest.approx(1e-3)
    assert ccsd_opts.residual_domain == "extended"
    assert ccsd_opts.triples_mode == "t1"


@pytest.mark.slow
def test_m16_nbutane_ccpvdz_dlpno_mp2_records_legacy_orca_tightpno_gap():
    """Retain the pre-#140/#448 NoFrozenCore comparison without rebaselining."""
    mol = _n_butane()
    basis = BasisSet(mol, "cc-pvdz")
    assert basis.nbasis == 106
    aux_name = default_aux_basis_for("cc-pvdz", kind="ri")
    assert aux_name == "cc-pvdz-ri"

    opts = RHFOptions()
    opts.max_iter = 100
    rhf = run_rhf(mol, basis, opts)
    assert rhf.converged
    assert rhf.energy == pytest.approx(ORCA_M16_RHF, abs=5e-7)

    df = DensityFitting(basis, BasisSet(mol, aux_name), aux_basis_name=aux_name)
    legacy = run_dlpno_mp2(
        mol, basis, rhf, df, _pre_140_448_dlpno_mp2_options()
    )
    orca_like = run_dlpno_mp2(
        mol,
        basis,
        rhf,
        df,
        DLPNOMP2Options(
            n_frozen=0,
            tcut_pno=1e-9,
            tcut_pno_weak=1e-9,
            tcut_pairs=0.0,
            tcut_pairs_weak=0.0,
            tcut_mkn=1e-3,
            # Retained value, measured before #65. Note the irony recorded
            # for whoever revisits this: with "mp2" now the default, an
            # "ORCA-like" row could finally use ORCA's own pair density --
            # but that is a new comparison to measure, not a licence to
            # rebaseline a row this test exists to hold fixed.
            pno_norm="legacy",
        ),
    )

    assert legacy.converged
    assert orca_like.converged
    assert legacy.n_frozen == 0
    assert legacy.e_corr == pytest.approx(
        VIBEQC_M16_DLPNO_MP2_PRE_140_448_CORR, abs=5e-7
    )
    assert orca_like.e_corr == pytest.approx(
        VIBEQC_M16_DLPNO_MP2_ORCA_LIKE_CORR, abs=5e-7
    )
    legacy_avg_pno = sum(legacy.pno_per_pair.values()) / len(
        legacy.pno_per_pair
    )
    orca_like_avg_pno = sum(orca_like.pno_per_pair.values()) / len(
        orca_like.pno_per_pair
    )
    assert orca_like_avg_pno > legacy_avg_pno

    legacy_gap = legacy.e_corr - ORCA_M16_DLPNO_MP2_CORR
    orca_like_gap = orca_like.e_corr - ORCA_M16_DLPNO_MP2_CORR
    assert legacy_gap == pytest.approx(0.0037620229, abs=1e-6)
    assert orca_like_gap == pytest.approx(0.0024729841, abs=1e-6)


@pytest.mark.slow
def test_m16_legacy_nbutane_ccpvdz_dlpno_ccsdt_out_emits_audit_components(
    tmp_path, monkeypatch
):
    """Keep the legacy all-electron audit shape and ORCA pins reproducible."""
    import vibeqc.dlpno.ccsd_local_solver as local_solver

    seen = {}

    def fake_local_solver(molecule, basis, rhf, df, options):
        assert molecule.n_electrons() == 34
        assert basis.name.lower() == "cc-pvdz"
        assert options.compute_triples is True
        assert options.n_frozen == 0
        assert options.tcut_pno == pytest.approx(1e-7)
        assert options.tcut_pairs == pytest.approx(1e-4)
        assert options.tcut_mkn == pytest.approx(0.0)
        assert options.residual_domain == "pair"
        e_hf = float(rhf.energy)
        e_t = ORCA_M16_DLPNO_TRIPLES
        e_corr = ORCA_M16_DLPNO_CCSD_CORR
        e_total = e_hf + e_corr + e_t
        seen["e_total"] = e_total
        return LocalCCSDResult(
            e_hf=e_hf,
            e_corr=e_corr,
            e_t=e_t,
            e_total=e_total,
            n_iter=12,
            converged=True,
            n_pairs=153,
            n_frozen=0,
            t1_norm=0.0,
            pno_per_pair={(0, i): 36 for i in range(153)},
        )

    monkeypatch.setattr(local_solver, "run_local_dlpno_ccsd", fake_local_solver)
    monkeypatch.chdir(tmp_path)

    rhf_options = RHFOptions()
    rhf_options.max_iter = 100
    res = run_job(
        _n_butane(),
        basis="cc-pvdz",
        method="dlpno-ccsd(t)",
        output="m16",
        rhf_options=rhf_options,
        dlpno_ccsd_options=_pre_140_448_local_ccsd_options(),
    )

    assert res.dlpno_ccsd.e_corr == pytest.approx(ORCA_M16_DLPNO_CCSD_CORR)
    assert res.dlpno_ccsd.e_t == pytest.approx(ORCA_M16_DLPNO_TRIPLES)
    assert res.energy_total == pytest.approx(seen["e_total"])

    out = (tmp_path / "m16.out").read_text()
    assert "DLPNO-CCSD(T) local reduced-scaling (M3c)" in out
    assert "E(CCSD correlation)" in out
    assert f"{ORCA_M16_DLPNO_CCSD_CORR:16.10f} Ha" in out
    assert "E((T) correction)" in out
    assert f"{ORCA_M16_DLPNO_TRIPLES:16.10f} Ha" in out
    assert "E(DLPNO-CCSD(T) corr)" in out
    final_corr = ORCA_M16_DLPNO_CCSD_CORR + ORCA_M16_DLPNO_TRIPLES
    assert f"{final_corr:16.10f} Ha" in out
    assert "E(DLPNO-CCSD(T) total)" in out
    assert f"{seen['e_total']:16.10f} Ha" in out
