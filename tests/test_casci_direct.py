"""Parity tests for the C++ direct determinant CAS-CI engine.

The direct engine (cpp/src/casci.cpp: string-based sigma builds per
Knowles & Handy, CPL 111, 315 (1984) + block Davidson) must reproduce the
validated Python dense Slater-Condon engine (the historical
``VIBEQC_CASCI_BACKEND=python`` path) to numerical precision — same
determinant ordering, same per-spin-sector phase conventions, same RDM
conventions (solvers._rdm.make_rdm12 / PySCF fci.direct_spin1).

PySCF is an out-of-process-style oracle via importorskip (CLAUDE.md §10:
test-only, never imported by vibeqc itself).
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import build_hamiltonian_mo, get_hf_orbital_provider
from vibeqc.solvers._casci import casci
from vibeqc.solvers._rdm import make_rdm12
from vibeqc.solvers._slater_condon import build_hamiltonian_matrix_unrestricted

pytest.importorskip("vibeqc._vibeqc_core")
core = pytest.importorskip("vibeqc._vibeqc_core")
if not hasattr(core, "casci_direct_solve"):  # stale extension build
    pytest.skip("extension lacks casci_direct_solve", allow_module_level=True)

H2O = Molecule(
    [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 1.43, -0.93]),
        Atom(1, [0.0, -1.43, -0.93]),
    ]
)
N2 = Molecule([Atom(7, [0.0, 0.0, 0.0]), Atom(7, [0.0, 0.0, 2.074])])


@pytest.fixture(scope="module")
def h2o_ham():
    b = BasisSet(H2O, "sto-3g")
    C = get_hf_orbital_provider(H2O, b, method="rhf")
    return build_hamiltonian_mo(H2O, b, C)


def _both_backends(H, ne, no, nc, nroots=1, ms2=None, monkeypatch=None):
    if ms2 is None:
        ms2 = ne % 2
    kw = dict(nuclear_repulsion=H.nuclear_repulsion, ms2=ms2, nroots=nroots)
    monkeypatch.setenv("VIBEQC_CASCI_BACKEND", "python")
    rp = casci(H.h1e, H.h2e, ne, no, nc, **kw)
    monkeypatch.setenv("VIBEQC_CASCI_BACKEND", "cpp")
    rc = casci(H.h1e, H.h2e, ne, no, nc, **kw)
    return rp, rc


class TestDirectEngineParity:
    @pytest.mark.parametrize(
        "ne,no,nc,nroots,ms2",
        [
            (4, 4, 3, 1, 0),  # closed-shell singlet
            (4, 4, 3, 3, 0),  # multi-root
            (6, 6, 1, 2, 0),  # 400-det space, 2 roots
            (5, 5, 2, 1, 1),  # open-shell doublet (n_alpha != n_beta)
            (10, 7, 0, 2, 0),  # full space (FCI limit), 441 dets
        ],
    )
    def test_eigenvalues_match_dense(self, h2o_ham, monkeypatch, ne, no, nc, nroots, ms2):
        rp, rc = _both_backends(
            h2o_ham, ne, no, nc, nroots=nroots, ms2=ms2, monkeypatch=monkeypatch
        )
        assert rc.n_det == rp.n_det
        for ep, ec in zip(rp.e_totals, rc.e_totals):
            assert abs(ep - ec) < 1e-9
        # Root-0 wavefunctions identical up to a global sign.
        assert abs(abs(np.dot(rp.ci_coeffs, rc.ci_coeffs)) - 1.0) < 1e-8

    def test_sigma_matches_dense_hamiltonian(self, h2o_ham, monkeypatch):
        # σ = H·c elementwise against the dense Slater-Condon matrix.
        from vibeqc.solvers._casci import _frozen_core_dressing
        from vibeqc.solvers._determinant import generate_determinants

        H = h2o_ham
        ne, no, nc = 4, 4, 3
        act = slice(nc, nc + no)
        _, h1a = _frozen_core_dressing(H.h1e, H.h2e, nc, act)
        g_act = np.ascontiguousarray(H.h2e[act, act, act, act])
        dets = generate_determinants(no, 2, 2)
        ham = build_hamiltonian_matrix_unrestricted(dets, h1a, g_act)
        rng = np.random.default_rng(7)
        c = rng.standard_normal(len(dets))
        c /= np.linalg.norm(c)
        eri_chem = np.ascontiguousarray(g_act.transpose(0, 2, 1, 3))
        sig = core.casci_direct_sigma(c, np.ascontiguousarray(h1a), eri_chem, no, 2, 2)
        assert np.max(np.abs(sig - ham @ c)) < 1e-11

    def test_rdm12_matches_python_oracle(self, h2o_ham, monkeypatch):
        rp, rc = _both_backends(h2o_ham, 6, 6, 1, monkeypatch=monkeypatch)
        d1p, d2p = make_rdm12(rp.ci_coeffs, rp.determinants, 6)
        d1c, d2c = core.casci_direct_rdm12(
            np.ascontiguousarray(rc.ci_coeffs), 6, 3, 3
        )
        assert np.max(np.abs(d1p - d1c)) < 1e-9
        assert np.max(np.abs(d2p - d2c)) < 1e-9

    def test_energy_from_direct_rdms_reproduces_eigenvalue(self, h2o_ham, monkeypatch):
        # ⟨H⟩ from the C++ RDMs must reproduce the CI eigenvalue (the same
        # self-test contract as solvers._rdm.energy_from_rdms).
        from vibeqc.solvers._casci import _frozen_core_dressing
        from vibeqc.solvers._rdm import energy_from_rdms

        H = h2o_ham
        ne, no, nc = 4, 4, 3
        monkeypatch.setenv("VIBEQC_CASCI_BACKEND", "cpp")
        r = casci(
            H.h1e, H.h2e, ne, no, nc,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
        )
        act = slice(nc, nc + no)
        _, h1a = _frozen_core_dressing(H.h1e, H.h2e, nc, act)
        g_act = np.ascontiguousarray(H.h2e[act, act, act, act])
        d1, d2 = core.casci_direct_rdm12(np.ascontiguousarray(r.ci_coeffs), no, 2, 2)
        e_active = energy_from_rdms(d1, d2, h1a, g_act)
        assert abs(e_active + r.e_core - r.e_total) < 1e-9

    def test_fci_limit_vs_pyscf(self, monkeypatch):
        # Direct engine in the full space == FCI (PySCF oracle).
        pytest.importorskip("pyscf")
        from pyscf import fci, gto, scf

        b = BasisSet(H2O, "sto-3g")
        C = get_hf_orbital_provider(H2O, b, method="rhf")
        H = build_hamiltonian_mo(H2O, b, C)
        monkeypatch.setenv("VIBEQC_CASCI_BACKEND", "cpp")
        r = casci(
            H.h1e, H.h2e, H.nelec, H.norb, 0,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
        )
        mf = scf.RHF(
            gto.M(
                atom="O 0 0 0; H 0 1.43 -0.93; H 0 -1.43 -0.93",
                unit="Bohr", basis="sto-3g", verbose=0,
            )
        ).run(verbose=0)
        e_fci = fci.FCI(mf).kernel()[0]
        assert abs(r.e_total - e_fci) < 1e-8

    def test_auto_threshold_dispatch(self, h2o_ham, monkeypatch):
        # Above the threshold, backend=auto must give the same physics as
        # the forced python path (here we just confirm auto == cpp == python
        # on a space that straddles nothing: behavioral guard, small space).
        monkeypatch.setenv("VIBEQC_CASCI_BACKEND", "auto")
        H = h2o_ham
        r_auto = casci(
            H.h1e, H.h2e, 4, 4, 3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
        )
        monkeypatch.setenv("VIBEQC_CASCI_BACKEND", "python")
        r_py = casci(
            H.h1e, H.h2e, 4, 4, 3,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
        )
        assert abs(r_auto.e_total - r_py.e_total) < 1e-12

    def test_invalid_backend_env_raises(self, h2o_ham, monkeypatch):
        monkeypatch.setenv("VIBEQC_CASCI_BACKEND", "fortran")
        with pytest.raises(ValueError, match="VIBEQC_CASCI_BACKEND"):
            casci(
                h2o_ham.h1e, h2o_ham.h2e, 4, 4, 3,
                nuclear_repulsion=h2o_ham.nuclear_repulsion, ms2=0,
            )


@pytest.mark.slow
class TestDirectEngineScale:
    def test_n2_cas66_vs_dense_and_beyond_dense_cap(self, monkeypatch):
        # N2 CAS(6,6): parity with dense; then the full space CAS(14,10) —
        # C(10,7)^2 = 14 400 determinants, beyond the historical
        # max_det=10 000 dense cap — must auto-dispatch to the direct
        # engine and land below the smaller CAS variationally (it is the
        # FCI of N2/STO-3G).
        b = BasisSet(N2, "sto-3g")
        C = get_hf_orbital_provider(N2, b, method="rhf")
        H = build_hamiltonian_mo(N2, b, C)
        kw = dict(nuclear_repulsion=H.nuclear_repulsion, ms2=0)
        monkeypatch.setenv("VIBEQC_CASCI_BACKEND", "python")
        rp = casci(H.h1e, H.h2e, 6, 6, 4, **kw)
        monkeypatch.setenv("VIBEQC_CASCI_BACKEND", "cpp")
        rc = casci(H.h1e, H.h2e, 6, 6, 4, **kw)
        assert abs(rp.e_total - rc.e_total) < 1e-9

        monkeypatch.setenv("VIBEQC_CASCI_BACKEND", "auto")
        r_big = casci(H.h1e, H.h2e, 14, 10, 0, **kw)  # full space, 14 400 dets
        assert r_big.n_det == 14_400
        # Variational: the full space is below the CAS(6,6) truncation.
        assert r_big.e_total < rc.e_total + 1e-12
