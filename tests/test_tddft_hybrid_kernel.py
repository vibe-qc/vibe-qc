"""Hybrid TDDFT/TDA exact-exchange kernel regression — pinned vs PySCF.

Regression for the v0.12.0 post-release audit bug (2026-06-12): the
TDDFT drivers hardcoded the exact-exchange coefficient as
``c_x = 1.0 if functional is None else 0.0``, so every HYBRID
functional (B3LYP α=0.20, PBE0 α=0.25, …) ran its response kernel
with ZERO exact exchange — only the ALDA/AGGA f_xc — systematically
overshooting hybrid excitation energies. On this file's H2O/6-31G
system the TDA-PBE0 lowest excitation came out 0.3975 Ha pre-fix vs
0.3001 Ha correct: a 2.6 eV error. Pure functionals (c_x = 0) and
HF/CIS (c_x = 1) were unaffected.

The correct global-hybrid response kernel (Bauernschmitt & Ahlrichs,
Chem. Phys. Lett. 256, 454 (1996), doi:10.1016/0009-2614(96)00440-X):

    A_{ia,jb} = δ_ij δ_ab (ε_a − ε_i) + 2 (ia|jb) − c_x (ij|ab) + (ia|f_xc|jb)
    B_{ia,jb} = 2 (ia|jb) − c_x (ib|ja) + (ia|f_xc|jb)

with c_x = α, the functional's exact-exchange admixture
(``Functional.hf_exchange_fraction``), and f_xc the adiabatic XC
kernel of the functional's DFT part. Range-separated hybrids
(ωB97X, HSE06, …) are NOT representable by a single global c_x over
regular ERIs and must be refused, not run wrong.

**Reference provenance (CLAUDE.md §10 — generated OUT-OF-PROCESS;
PySCF is never imported by vibe-qc code or tests):** PySCF 2.13.0 /
libxc 7.0.0, run 2026-06-12 in a standalone interpreter process.
RKS with conv_tol=1e-12 on PySCF's default (level-3) grids, then
``tdscf.TDA`` / ``tdscf.TDDFT`` with nstates=4; all SCF and Davidson
roots converged. Geometry and basis exactly as in ``_h2o()`` below —
coordinates specified in BOHR on both sides, basis 6-31G. Sanity
anchor: vibe-qc's pure-functional TDA energies (LDA, PBE — where the
broken c_x was coincidentally correct) matched these PySCF references
to ~1e-7 Ha before the fix, validating grid + f_xc plumbing; only the
hybrid exchange admixture was broken.

B3LYP flavor note (see tests/test_b3lyp_convention.py): vibe-qc's
bare ``b3lyp`` is the ORCA/VWN5 flavor — the PySCF reference below
was generated with xc='b3lyp5' (PySCF's bare 'b3lyp' is the
Gaussian/VWN-RPA variant). PBE0 has no flavor trap and is the
primary pin.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.tddft import (
    _eri_oovv,
    eri_ao_to_mo,
    run_tddft_casida,
    run_tddft_casida_uhf,
    run_tddft_tda,
    run_tddft_tda_uhf,
)

# ---------------------------------------------------------------------------
# PySCF 2.13.0 reference values (provenance in module docstring).
# Excitation energies in Hartree, lowest four singlet states.
# ---------------------------------------------------------------------------

# RKS-PBE0, E_SCF = -76.3009965269 Ha (vibe-qc: -76.3009968183, Δ 3e-7
# from grid differences).
PYSCF_PBE0_TDA = [0.3001181503, 0.3781959381, 0.3805832935, 0.4699865199]
PYSCF_PBE0_TDDFT = [0.2993615380, 0.3777468903, 0.3779558212, 0.4679017228]

# RKS-B3LYP (VWN5 flavor == PySCF 'b3lyp5'), E_SCF = -76.3477719186 Ha.
PYSCF_B3LYP5_TDA = [0.2879923476, 0.3660259744, 0.3672814814, 0.4569072063]
PYSCF_B3LYP5_TDDFT = [0.2872070309, 0.3643050929, 0.3658521869, 0.4548378381]

# RKS-LDA (slater,vwn5) — pure functional, unaffected by the c_x fix;
# guards the f_xc kernel convention itself.
PYSCF_LDA_TDA = [0.2788820227, 0.3503903975, 0.3603405491, 0.4464360472]
PYSCF_LDA_TDDFT = [0.2779637415, 0.3472723898, 0.3601752432, 0.4438057380]

# Agreement floor: with the fixed kernel, all six functional/solver
# combinations below matched these PySCF references to ≤ 2.4e-7 Ha on
# vibe-qc's default grid (recorded 2026-06-12). 5e-5 Ha absorbs benign
# grid/libxc drift while sitting 3 orders below the ~0.1 Ha bug signal
# and well below the 2.4 mHa spacing of the closest pinned states.
TOL = 5e-5


def _h2o() -> vq.Molecule:
    """H2O, coordinates in bohr (r_OH = 1.8090 bohr, ∠HOH = 104.5°) —
    identical literals to the PySCF reference run (unit='Bohr')."""
    return vq.Molecule(
        [
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 1.43, 1.108]),
            vq.Atom(1, [0.0, -1.43, 1.108]),
        ],
        0,
        1,
    )


_scf_cache: dict[str, tuple] = {}


def _h2o_rks(functional: str):
    """Cached H2O/6-31G RKS ground state for the given functional."""
    if functional not in _scf_cache:
        mol = _h2o()
        basis = vq.BasisSet(mol, "6-31g")
        opts = vq.RKSOptions()
        opts.functional = functional
        opts.conv_tol_energy = 1e-12
        res = vq.run_rks(mol, basis, opts)
        assert res.converged
        _scf_cache[functional] = (mol, basis, res)
    return _scf_cache[functional]


def _excitations_tda(functional: str, n_states: int = 4) -> list[float]:
    mol, basis, res = _h2o_rks(functional)
    out = run_tddft_tda(
        mol,
        basis,
        res.mo_energies,
        res.mo_coeffs,
        mol.n_electrons() // 2,
        n_states=n_states,
        functional=functional,
        density_ao=np.asarray(res.density),
    )
    return [s.excitation_energy for s in out.states]


def _excitations_casida(functional: str, n_states: int = 4) -> list[float]:
    mol, basis, res = _h2o_rks(functional)
    out = run_tddft_casida(
        mol,
        basis,
        res.mo_energies,
        res.mo_coeffs,
        mol.n_electrons() // 2,
        n_states=n_states,
        functional=functional,
        density_ao=np.asarray(res.density),
    )
    return [s.excitation_energy for s in out.states]


# ---------------------------------------------------------------------------
# The bug: hybrids must carry their exact-exchange fraction
# ---------------------------------------------------------------------------


def test_tda_pbe0_matches_pyscf():
    """TDA-PBE0 (α = 0.25): the primary hybrid pin."""
    w = _excitations_tda("pbe0")
    for k, ref in enumerate(PYSCF_PBE0_TDA):
        assert w[k] == pytest.approx(ref, abs=TOL), (
            f"state {k + 1}: {w[k]:.8f} vs PySCF {ref:.8f}"
        )


def test_tddft_casida_pbe0_matches_pyscf():
    """Full-response (Casida) TDDFT-PBE0: α·HF exchange and f_xc must
    both enter A and B."""
    w = _excitations_casida("pbe0")
    for k, ref in enumerate(PYSCF_PBE0_TDDFT):
        assert w[k] == pytest.approx(ref, abs=TOL), (
            f"state {k + 1}: {w[k]:.8f} vs PySCF {ref:.8f}"
        )


def test_tda_b3lyp_matches_pyscf_b3lyp5():
    """TDA-B3LYP (α = 0.20), VWN5-flavor cross-pin against PySCF 'b3lyp5'."""
    w = _excitations_tda("b3lyp")
    for k, ref in enumerate(PYSCF_B3LYP5_TDA):
        assert w[k] == pytest.approx(ref, abs=TOL), (
            f"state {k + 1}: {w[k]:.8f} vs PySCF {ref:.8f}"
        )


def test_tddft_casida_b3lyp_matches_pyscf_b3lyp5():
    w = _excitations_casida("b3lyp")
    for k, ref in enumerate(PYSCF_B3LYP5_TDDFT):
        assert w[k] == pytest.approx(ref, abs=TOL), (
            f"state {k + 1}: {w[k]:.8f} vs PySCF {ref:.8f}"
        )


# ---------------------------------------------------------------------------
# Pure functionals: c_x = 0 unchanged; guards the f_xc convention
# ---------------------------------------------------------------------------


def test_tda_lda_matches_pyscf():
    """Pure-functional control: matched PySCF before the fix and must
    keep matching after it (the fix may not disturb c_x = 0)."""
    w = _excitations_tda("lda")
    for k, ref in enumerate(PYSCF_LDA_TDA):
        assert w[k] == pytest.approx(ref, abs=TOL), (
            f"state {k + 1}: {w[k]:.8f} vs PySCF {ref:.8f}"
        )


def test_tddft_casida_lda_matches_pyscf():
    """Casida with density_ao composes f_xc into both A and B: full
    TDDFT-LDA must hit the PySCF TDDFT reference (pre-fix the Casida
    driver had no f_xc support at all and gave 0.2898 Ha here)."""
    w = _excitations_casida("lda")
    for k, ref in enumerate(PYSCF_LDA_TDDFT):
        assert w[k] == pytest.approx(ref, abs=TOL), (
            f"state {k + 1}: {w[k]:.8f} vs PySCF {ref:.8f}"
        )


# ---------------------------------------------------------------------------
# UHF/UKS-TDA site: same c_x admixture as the restricted driver
# ---------------------------------------------------------------------------


def test_uhf_tda_hybrid_uses_exchange_fraction():
    """The UHF-TDA driver must apply c_x = α in its same-spin exchange
    blocks. For a closed-shell reference fed identical α/β MOs, the
    in-phase (singlet) eigenvalues of the UHF TDA matrix equal the
    eigenvalues of the spin-adapted singlet matrix

        A_{ia,jb} = δ_ij δ_ab (ε_a − ε_i) + 2 (ia|jb) − α (ij|ab)

    built here directly from the same MOs with α = 0.25 (PBE0). No
    f_xc on either side (the UHF driver has no polarised-kernel path
    yet), so this isolates exactly the exchange admixture."""
    mol, basis, res = _h2o_rks("pbe0")
    n_occ = mol.n_electrons() // 2
    n_virt = basis.nbasis - n_occ

    eri_ao = vq._vibeqc_core.compute_eri(basis)
    ovov = eri_ao_to_mo(eri_ao, res.mo_coeffs, n_occ)
    oovv = _eri_oovv(eri_ao, res.mo_coeffs, n_occ)
    eps_o = np.asarray(res.mo_energies)[:n_occ]
    eps_v = np.asarray(res.mo_energies)[n_occ:]

    alpha = 0.25  # PBE0 exact-exchange admixture
    n_pair = n_occ * n_virt
    a_singlet = np.zeros((n_pair, n_pair))
    for i in range(n_occ):
        for a in range(n_virt):
            ia = i * n_virt + a
            a_singlet[ia, ia] = eps_v[a] - eps_o[i]
            for j in range(n_occ):
                for b in range(n_virt):
                    jb = j * n_virt + b
                    a_singlet[ia, jb] += (
                        2.0 * ovov[i, a, j, b] - alpha * oovv[i, j, a, b]
                    )
    w_singlet = np.linalg.eigvalsh(a_singlet)[:3]

    uhf = run_tddft_tda_uhf(
        mol,
        basis,
        res.mo_energies,
        res.mo_energies,
        res.mo_coeffs,
        res.mo_coeffs,
        n_occ,
        n_occ,
        n_states=2 * n_pair,
        functional="pbe0",
    )
    w_uhf = np.array([s.excitation_energy for s in uhf.states])

    # Each singlet eigenvalue must appear in the UHF spectrum (which
    # also contains the triplet/out-of-phase combinations).
    for w in w_singlet:
        assert np.min(np.abs(w_uhf - w)) < 1e-8, (
            f"singlet root {w:.10f} missing from UHF-TDA spectrum "
            f"(closest: {w_uhf[np.argmin(np.abs(w_uhf - w))]:.10f})"
        )


@pytest.mark.parametrize(
    ("restricted_runner", "unrestricted_runner"),
    [
        (run_tddft_tda, run_tddft_tda_uhf),
        (run_tddft_casida, run_tddft_casida_uhf),
    ],
    ids=["tda", "casida"],
)
def test_uhf_response_with_polarised_fxc_reduces_to_restricted(
    restricted_runner,
    unrestricted_runner,
):
    """With identical alpha/beta orbitals and spin densities, the
    spin-polarised UHF/UKS response kernel must contain the restricted
    singlet root, including the adiabatic f_xc contribution."""
    mol = vq.Molecule(
        [
            vq.Atom(1, [0.0, 0.0, -0.7]),
            vq.Atom(1, [0.0, 0.0, 0.7]),
        ]
    )
    basis = vq.BasisSet(mol, "sto-3g")
    opts = vq.RKSOptions()
    opts.functional = "pbe0"
    opts.conv_tol_energy = 1e-12
    res = vq.run_rks(mol, basis, opts)
    assert res.converged

    n_occ = mol.n_electrons() // 2
    restricted = restricted_runner(
        mol,
        basis,
        res.mo_energies,
        res.mo_coeffs,
        n_occ,
        n_states=1,
        functional="pbe0",
        density_ao=np.asarray(res.density),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        unrestricted = unrestricted_runner(
            mol,
            basis,
            res.mo_energies,
            res.mo_energies,
            res.mo_coeffs,
            res.mo_coeffs,
            n_occ,
            n_occ,
            n_states=2,
            functional="pbe0",
            density_alpha_ao=0.5 * np.asarray(res.density),
            density_beta_ao=0.5 * np.asarray(res.density),
        )

    assert not any("f_xc" in str(w.message) for w in caught)
    roots = np.array([state.excitation_energy for state in unrestricted.states])
    ref = restricted.states[0].excitation_energy
    assert np.min(np.abs(roots - ref)) < 1e-8


# ---------------------------------------------------------------------------
# Range-separated hybrids: refuse, don't run wrong
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rsh", ["wb97x", "hse06"])
def test_rsh_functionals_refused(rsh):
    """ω-split exchange (α + β·erf(ωr)) is not representable by one
    global c_x over regular ERIs — all three drivers must refuse."""
    mol, basis, res = _h2o_rks("pbe0")
    n_occ = mol.n_electrons() // 2

    with pytest.raises(NotImplementedError, match="[Rr]ange-separated"):
        run_tddft_tda(
            mol, basis, res.mo_energies, res.mo_coeffs, n_occ,
            n_states=1, functional=rsh,
        )
    with pytest.raises(NotImplementedError, match="[Rr]ange-separated"):
        run_tddft_casida(
            mol, basis, res.mo_energies, res.mo_coeffs, n_occ,
            n_states=1, functional=rsh,
        )
    with pytest.raises(NotImplementedError, match="[Rr]ange-separated"):
        run_tddft_tda_uhf(
            mol, basis, res.mo_energies, res.mo_energies,
            res.mo_coeffs, res.mo_coeffs, n_occ, n_occ,
            n_states=1, functional=rsh,
        )
    with pytest.raises(NotImplementedError, match="[Rr]ange-separated"):
        run_tddft_casida_uhf(
            mol, basis, res.mo_energies, res.mo_energies,
            res.mo_coeffs, res.mo_coeffs, n_occ, n_occ,
            n_states=1, functional=rsh,
        )


# ---------------------------------------------------------------------------
# Partial-kernel honesty: hybrid without density_ao warns
# ---------------------------------------------------------------------------


def test_hybrid_without_density_warns():
    """A hybrid functional without density_ao gets its α·HF exchange
    but no f_xc — a partial kernel that must not pass silently as
    TDA-PBE0 (CLAUDE.md §7)."""
    mol, basis, res = _h2o_rks("pbe0")
    n_occ = mol.n_electrons() // 2
    with pytest.warns(UserWarning, match="f_xc"):
        run_tddft_tda(
            mol, basis, res.mo_energies, res.mo_coeffs, n_occ,
            n_states=1, functional="pbe0",
        )
