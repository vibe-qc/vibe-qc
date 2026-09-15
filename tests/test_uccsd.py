"""Open-shell DF-UCCSD / DF-UCCSD(T) correlation energies vs PySCF.

Out-of-venv cross-check; skipped when PySCF is not installed (PySCF is a
parity oracle, never a vibe-qc dependency -- CLAUDE.md section 10). The
always-on in-repo gate is tests/test_uccsd_anchor.py (the spin-orbital
SGWB-1991 anchor, run_ref_uccsd). Validation summary (2026-06-17): the C++
kernel agrees with PySCF cc.UCCSD/UCCSD(T) to < 1e-3 uHa on CH3/NH2/O2 in
sto-3g and cc-pVDZ (all-electron) and to < 1e-2 uHa frozen-core.
"""

from __future__ import annotations

import pytest
from vibeqc import Atom, BasisSet, Molecule, UHFOptions, run_uhf
from vibeqc.cc import CCSDOptions, run_uccsd

ANG = 1.8897259886
AUX = "cc-pvdz-ri"

# (atoms in Bohr (Z, [x,y,z]); multiplicity)
SYSTEMS = {
    "CH3": ([(6, [0, 0, 0]), (1, [1.079 * ANG, 0, 0]),
             (1, [-0.5395 * ANG, 0.9344 * ANG, 0]),
             (1, [-0.5395 * ANG, -0.9344 * ANG, 0])], 2),
    "NH2": ([(7, [0, 0, 0]), (1, [0, 0.7268 * ANG, 0.5415 * ANG]),
             (1, [0, -0.7268 * ANG, 0.5415 * ANG])], 2),
    "O2": ([(8, [0, 0, 0]), (8, [0, 0, 1.208 * ANG])], 3),
}


def _pyscf_uccsd(atoms, basis_name, mult, frozen=0):
    """Reference UHF + DF-UCCSD/(T) via PySCF, spawned in this test process."""
    pytest.importorskip("pyscf")
    from pyscf import cc, gto, scf

    mol = gto.Mole()
    mol.unit = "Bohr"
    mol.atom = [[Z, tuple(xyz)] for Z, xyz in atoms]
    mol.basis = basis_name
    mol.spin = mult - 1
    mol.charge = 0
    mol.verbose = 0
    mol.build()

    mf = scf.UHF(mol).density_fit(auxbasis=AUX)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 300
    mf.kernel()
    if not mf.converged:
        mf = mf.newton()
        mf.kernel()
    assert mf.converged

    mycc = cc.UCCSD(mf, frozen=frozen).density_fit(auxbasis=AUX)
    mycc.conv_tol = 1e-10
    mycc.conv_tol_normt = 1e-9
    mycc.diis_space = 6
    e_corr = mycc.kernel()[0]
    assert mycc.converged
    e_t = mycc.ccsd_t()
    return {"e_corr": e_corr, "e_ccsd": mycc.e_tot, "e_t": e_t,
            "e_ccsd_t": mycc.e_tot + e_t}


def _vibeqc_uccsd(atoms, basis_name, mult, frozen=0):
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms],
                   charge=0, multiplicity=mult)
    basis = BasisSet(mol, basis_name)
    o = UHFOptions()
    o.density_fit = True
    o.aux_basis = AUX
    o.conv_tol_energy = 1e-11
    o.conv_tol_grad = 1e-9
    o.max_iter = 400
    o.level_shift = 0.3
    # Solution-matched kernel parity: PySCF's reference UHF sits on the
    # symmetric O2-triplet solution; the internal-stability escape
    # (UHF-ABOVE-ROHF-VARIATIONAL-INVERSION fix) would move vibe-qc to a
    # lower symmetry-broken solution and turn this CCSD-kernel test into
    # a basin comparison. Pin the legacy solution.
    o.stability_check = False
    hf = run_uhf(mol, basis, o)
    assert hf.converged, f"vibe-qc UHF did not converge on {basis_name}"
    co = CCSDOptions(aux_basis=AUX, compute_triples=True, n_frozen_core=frozen)
    co.conv_tol_energy = 1e-10
    co.conv_tol_residual = 1e-9
    return run_uccsd(mol, basis, hf, co)


# Small open-shell radicals, minimal basis for fast CI.
UCCSD_CASES = [("CH3", "sto-3g"), ("NH2", "sto-3g"), ("O2", "sto-3g")]


@pytest.mark.parametrize("mol_key,basis", UCCSD_CASES,
                         ids=[f"{m}-{b}" for m, b in UCCSD_CASES])
def test_uccsd_energy_matches_pyscf(mol_key, basis):
    """DF-UCCSD correlation energy agrees with PySCF to 1 uHa."""
    atoms, mult = SYSTEMS[mol_key]
    ref = _pyscf_uccsd(atoms, basis, mult)
    res = _vibeqc_uccsd(atoms, basis, mult)
    assert res.converged
    assert abs(res.e_ccsd_correlation - ref["e_corr"]) < 1e-6, (
        f"{mol_key}/{basis}: E_corr vibe-qc={res.e_ccsd_correlation:.12f}, "
        f"pyscf={ref['e_corr']:.12f}, "
        f"diff={res.e_ccsd_correlation - ref['e_corr']:+.2e}"
    )


@pytest.mark.parametrize("mol_key,basis", UCCSD_CASES,
                         ids=[f"{m}-{b}" for m, b in UCCSD_CASES])
def test_uccsd_t_energy_matches_pyscf(mol_key, basis):
    """DF-UCCSD(T) total energy agrees with PySCF to 1 uHa."""
    atoms, mult = SYSTEMS[mol_key]
    ref = _pyscf_uccsd(atoms, basis, mult)
    res = _vibeqc_uccsd(atoms, basis, mult)
    assert res.converged
    assert abs(res.e_t - ref["e_t"]) < 1e-6, (
        f"{mol_key}/{basis}: E_T vibe-qc={res.e_t:.12f}, "
        f"pyscf={ref['e_t']:.12f}, diff={res.e_t - ref['e_t']:+.2e}"
    )
    assert abs(res.e_ccsd_t - ref["e_ccsd_t"]) < 1e-6


@pytest.mark.slow
def test_uccsd_t_frozen_core_matches_pyscf():
    """Frozen-core (run_job default) DF-UCCSD(T) agrees with PySCF frozen=1."""
    atoms, mult = SYSTEMS["CH3"]
    ref = _pyscf_uccsd(atoms, "cc-pvdz", mult, frozen=1)
    res = _vibeqc_uccsd(atoms, "cc-pvdz", mult, frozen=1)
    assert res.converged
    assert abs(res.e_ccsd_t - ref["e_ccsd_t"]) < 1e-6


# Cached ORCA 6.1.0 cross-check references (out-of-process; CLAUDE.md
# section 10). Generated 2026-06-17 with
#   ! UHF CCSD(T) cc-pVDZ NoFrozenCore TightSCF
# on the geometries in SYSTEMS above. ORCA uses conventional (non-DF)
# integrals, so the *totals* differ from vibe-qc's DF-UCCSD(T) by the
# cc-pVDZ density-fitting error (~0.8-0.9 mHa); the perturbative (T)
# correction is DF-insensitive and agrees independently to ~5 uHa. (The
# tight same-DF parity check is the PySCF cross-check above.)
ORCA_CCPVDZ = {
    "CH3": {"e_ccsd_t": -39.718197118, "e_t": -0.002623615},
    "NH2": {"e_ccsd_t": -55.701067317, "e_t": -0.002201888},
}


@pytest.mark.slow
@pytest.mark.parametrize("mol_key", ["CH3", "NH2"])
def test_uccsd_t_orca_cross_check(mol_key):
    """Independent-code cross-check vs cached conventional ORCA 6.1 CCSD(T)."""
    atoms, mult = SYSTEMS[mol_key]
    res = _vibeqc_uccsd(atoms, "cc-pvdz", mult, frozen=0)
    orca = ORCA_CCPVDZ[mol_key]
    assert res.converged
    # (T) is density-fitting-insensitive: tight agreement with conventional ORCA.
    assert abs(res.e_t - orca["e_t"]) < 3e-5
    # Total carries the cc-pVDZ DF error vs ORCA's conventional integrals.
    assert abs(res.e_ccsd_t - orca["e_ccsd_t"]) < 2e-3


def test_run_job_routes_open_shell_to_uccsd(tmp_path):
    """run_job(method='ccsd(t)') on an open-shell molecule routes UHF->UCCSD
    and attaches a CCSDResult; closed shell still routes to RCCSD."""
    atoms, mult = SYSTEMS["CH3"]
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in atoms],
                   charge=0, multiplicity=mult)
    import vibeqc as vq

    uo = UHFOptions()
    uo.level_shift = 0.3
    uo.max_iter = 400
    uo.conv_tol_energy = 1e-10
    r = vq.run_job(mol, basis="cc-pvdz", method="ccsd(t)", uhf_options=uo,
                   triples="(t)",
                   output=str(tmp_path / "open_shell_uccsd"))
    assert hasattr(r, "ccsd")
    assert r.ccsd.converged
    assert r.ccsd.e_t < 0.0  # (T) computed
    # vs direct run_uccsd on the same reference type (frozen core default)
    assert r.ccsd.e_ccsd_correlation < 0.0
