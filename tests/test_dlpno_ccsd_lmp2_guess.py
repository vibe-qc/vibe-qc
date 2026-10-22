"""Full local MP2 PNO guess of the local DLPNO-CCSD route (pno_guess="lmp2").

The coupled local MP2 equations are invariant to the occupied localization,
so in full PAO domains with every pair and no Fock cutoff they reproduce
canonical DF-MP2, and with no PNO truncation the LMP2 truncation correction
vanishes while CCSD stays canonical. The dense and compiled LMP2 sweeps are
two contractions of one residual and must agree.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import BasisSet, RHFOptions, run_rhf
from vibeqc._vibeqc_core import Atom, Molecule
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno._ccsd_cs import run_cs_ccsd
from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions, run_local_dlpno_ccsd

ANGSTROM_TO_BOHR = 1.8897259886
AUX = "def2-svp-rifit"

H2O_ATOMS = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
    (1, [0.0, -0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
]

TIGHT_LMP2 = dict(
    pno_guess="lmp2",
    lmp2_conv_tol_energy=1e-12,
    lmp2_conv_tol_residual=1e-10,
    lmp2_max_iter=200,
)


@pytest.fixture(scope="module")
def water():
    mol = Molecule([Atom(z, p) for z, p in H2O_ATOMS], charge=0, multiplicity=1)
    basis = BasisSet(mol, "def2-svp")
    opts = RHFOptions()
    opts.max_iter = 100
    rhf = run_rhf(mol, basis, opts)
    assert rhf.converged
    df = DensityFitting(basis, BasisSet(mol, AUX), aux_basis_name=AUX)
    C = np.asarray(rhf.mo_coeffs)
    n_occ = mol.n_electrons() // 2
    Co, Cv = C[:, :n_occ], C[:, n_occ:]
    f_mo = C.T @ np.asarray(rhf.fock) @ C
    B_ov = np.asarray(df.mo_transform(Co, Cv))
    eps = np.diag(f_mo)
    K = np.einsum("Pia,Pjb->iajb", B_ov, B_ov, optimize=True)
    D = (
        eps[:n_occ, None, None, None]
        + eps[None, None, :n_occ, None]
        - eps[None, n_occ:, None, None]
        - eps[None, None, None, n_occ:]
    )
    e_mp2 = float(np.sum(K * (2.0 * K - K.transpose(0, 3, 2, 1)) / D))
    e_ccsd = run_cs_ccsd(
        f_mo,
        B_ov,
        np.asarray(df.mo_transform(Cv, Cv)),
        np.asarray(df.mo_transform(Co, Co)),
        n_occ,
        e_hf=rhf.energy,
    ).e_corr
    return mol, basis, rhf, df, e_mp2, e_ccsd


def _run(water, **kw):
    mol, basis, rhf, df, _, _ = water
    return run_local_dlpno_ccsd(mol, basis, rhf, df, LocalCCSDOptions(n_frozen=0, **kw))


FULL = dict(tcut_pno=0.0, tcut_mkn=0.0, coupling_radius=0.0, lmp2_fock_cut=0.0)


def test_full_domain_lmp2_is_canonical_mp2_and_ccsd_stays_canonical(water):
    *_, e_mp2, e_ccsd = water
    r = _run(water, tcut_pairs=0.0, **FULL, **TIGHT_LMP2)
    assert r.pno_guess == "lmp2"
    assert r.lmp2_converged
    assert r.n_lmp2_pairs == r.n_pairs == 15
    assert len(r.lmp2_iterations) == 2
    assert "lmp2_guess" in r.phase_seconds
    assert r.e_lmp2 == pytest.approx(e_mp2, abs=1e-9)
    assert r.e_pno_correction == pytest.approx(0.0, abs=1e-9)
    assert r.converged
    assert r.e_corr == pytest.approx(e_ccsd, abs=1e-6)


def test_untruncated_weak_pairs_leave_no_truncation_correction(water):
    # Weak pairs keep their intermediate PNOs in the final LMP2 solve; with
    # nothing truncated, the two solves span the same spaces.
    r = _run(water, tcut_pairs=1e-3, **FULL, **TIGHT_LMP2)
    assert r.n_screened > 0
    assert r.n_lmp2_pairs == 15
    assert r.e_pno_correction == pytest.approx(0.0, abs=1e-9)


def test_dense_and_compiled_lmp2_sweeps_agree(water):
    kw =dict(tcut_pno=1e-6, tcut_pairs=1e-5, **TIGHT_LMP2)
    dense = _run(water, **kw)
    compiled = _run(water, lmp2_dense_max_gb=0.0, **kw)
    assert dense.lmp2_converged and compiled.lmp2_converged
    assert compiled.e_lmp2 == pytest.approx(dense.e_lmp2, abs=1e-10)
    assert compiled.e_pno_correction == pytest.approx(dense.e_pno_correction, abs=1e-10)
    assert compiled.e_corr == pytest.approx(dense.e_corr, abs=1e-8)
    assert compiled.pno_per_pair == dense.pno_per_pair


def test_semicanonical_guess_stays_the_default(water):
    assert LocalCCSDOptions().pno_guess == "semicanonical"
    r = _run(water, tcut_pno=1e-6)
    assert r.pno_guess == "semicanonical"
    assert r.n_lmp2_pairs == 0 and r.lmp2_iterations == ()


def test_unknown_pno_guess_is_refused():
    with pytest.raises(ValueError, match="pno_guess"):
        LocalCCSDOptions(pno_guess="canonical")


def test_run_job_discloses_the_lmp2_guess(water, tmp_path, monkeypatch):
    import json

    from vibeqc import run_job

    monkeypatch.chdir(tmp_path)
    mol = water[0]
    res = run_job(
        mol,
        basis="def2-svp",
        method="dlpno-ccsd",
        output="w",
        structured_log=True,
        dlpno_ccsd_options=LocalCCSDOptions(tcut_pno=1e-6, pno_guess="lmp2"),
    )
    r = res.dlpno_ccsd
    assert r.converged and r.pno_guess == "lmp2" and r.lmp2_converged
    out = (tmp_path / "w.out").read_text()
    assert "PNO guess" in out and "full local MP2" in out
    records = [
        json.loads(line)
        for line in (tmp_path / "w.scf.jsonl").read_text().splitlines()
        if line.strip()
    ]
    done = next(row for row in records if row["event"] == "dlpno_ccsd_done")
    assert done["pno_guess"] == "lmp2"

    default = run_job(mol, basis="def2-svp", method="dlpno-ccsd", output="d")
    assert default.dlpno_ccsd.pno_guess == "semicanonical"
    assert "PNO guess" not in (tmp_path / "d.out").read_text()
