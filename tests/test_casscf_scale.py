"""Large-active-space CASSCF on the C++ direct CI core (roadmap 25d-e).

Covers the production-scale pieces layered on the direct determinant
engine (tests/test_casci_direct.py):

* the RDM backend dispatch (solvers._rdm.make_rdm12 routes full-CAS
  determinant spaces through the C++ kernel, truncated CI lists stay on
  the Python path);
* the robustified Super-CI step (|H_diag| clamp + steepest-descent
  line-search fallback) and the trust-region Newton-CG ``orbital_step="nr"``
  (Hessian-vector products by central FD of the analytic gradient);
* the headline capability: CAS(10,10) CASSCF — 63 504 determinants per
  CI solve — converging to a genuine stationary point.

CASSCF is non-convex; on this surface vibe-qc converges to a stationary
point 8.13 mHa BELOW PySCF ``mcscf.CASSCF``'s converged result
(-109.0614472 vs -109.0533171, both |g|-converged — different basins),
so the regression pins vibe-qc's own stationary point plus variational
bounds rather than cross-code equality (same policy as the basin-rich
H2O cases in test_solvers_mrpt_parity.py).
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import build_hamiltonian_mo, get_hf_orbital_provider
from vibeqc.solvers._casci import casci
from vibeqc.solvers._casscf import casscf

core = pytest.importorskip("vibeqc._vibeqc_core")
if not hasattr(core, "casci_direct_rdm12"):  # stale extension build
    pytest.skip("extension lacks casci_direct_rdm12", allow_module_level=True)

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


class TestRDMDirectDispatch:
    """solvers._rdm.make_rdm12 routes full-CAS spaces through the C++ kernel."""

    def test_make_rdm12_backends_agree(self, h2o_ham, monkeypatch):
        from vibeqc.solvers._rdm import make_rdm12

        H = h2o_ham
        monkeypatch.setenv("VIBEQC_CASCI_BACKEND", "python")
        r = casci(
            H.h1e, H.h2e, 6, 6, 1,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
        )
        monkeypatch.setenv("VIBEQC_RDM_BACKEND", "python")
        d1p, d2p = make_rdm12(r.ci_coeffs, r.determinants, 6)
        monkeypatch.setenv("VIBEQC_RDM_BACKEND", "cpp")
        d1c, d2c = make_rdm12(r.ci_coeffs, r.determinants, 6)
        assert np.max(np.abs(d1p - d1c)) < 1e-12
        assert np.max(np.abs(d2p - d2c)) < 1e-12

    def test_sa_rdms_route_through_dispatch(self, h2o_ham, monkeypatch):
        from vibeqc.solvers._rdm import make_rdm12_sa

        H = h2o_ham
        monkeypatch.setenv("VIBEQC_CASCI_BACKEND", "python")
        r = casci(
            H.h1e, H.h2e, 6, 6, 1,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0, nroots=2,
        )
        monkeypatch.setenv("VIBEQC_RDM_BACKEND", "python")
        s1p, s2p = make_rdm12_sa(r.ci_coeffs_all, r.determinants, 6, [0.5, 0.5])
        monkeypatch.setenv("VIBEQC_RDM_BACKEND", "cpp")
        s1c, s2c = make_rdm12_sa(r.ci_coeffs_all, r.determinants, 6, [0.5, 0.5])
        assert np.max(np.abs(s1p - s1c)) < 1e-12
        assert np.max(np.abs(s2p - s2c)) < 1e-12

    def test_truncated_det_list_stays_python(self):
        # The C++ kernel assumes the canonical full CAS space; truncated
        # lists (selected CI / CISD) must be detected and refused.
        from vibeqc.solvers._determinant import generate_determinants
        from vibeqc.solvers._rdm import _full_cas_counts

        full = generate_determinants(6, 3, 3)
        assert _full_cas_counts(full, 6) == (3, 3)
        assert _full_cas_counts(full[:-5], 6) is None
        assert _full_cas_counts(full[5:], 6) is None


class TestOrbitalSteps:
    def test_all_orbital_steps_agree_on_unique_minimum(self):
        # LiH/STO-3G CAS(2,2) has a unique CASSCF minimum (used by the
        # PySCF parity suite): pure Super-CI, auto (Super-CI -> Newton-CG)
        # and pure Newton-CG must all converge to the same energy.
        LIH = Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])])
        b = BasisSet(LIH, "sto-3g")
        C = get_hf_orbital_provider(LIH, b, method="rhf")
        H = build_hamiltonian_mo(LIH, b, C)
        kw = dict(nuclear_repulsion=H.nuclear_repulsion, ms2=0)
        res = {
            step: casscf(H.h1e, H.h2e, 2, 2, 1, orbital_step=step, **kw)
            for step in ("superci", "auto", "nr")
        }
        # The second-order steps must converge tightly (Newton-CG endgame);
        # pure Super-CI is the robust far-field method with a slow
        # first-order tail — it must reach the same energy, but |g| <= 1e-6
        # within max_macro is not part of its contract.
        assert res["auto"].converged and res["nr"].converged
        e = [r.e_total for r in res.values()]
        assert max(e) - min(e) < 1e-8

    def test_steihaug_boundary_helper(self):
        from vibeqc.solvers._casscf import _to_trust_boundary

        x = np.array([0.1, 0.0])
        p = np.array([1.0, 0.0])
        out = _to_trust_boundary(x, p, 0.5)
        assert abs(np.linalg.norm(out) - 0.5) < 1e-12


class TestDavidsonWarmStart:
    def test_warm_start_reproduces_cold_start(self, h2o_ham, monkeypatch):
        # Seeding Davidson with the converged CI vector (or any reasonable
        # guess) must converge to the same eigenpairs as the cold start.
        from vibeqc.solvers._casci import casci

        H = h2o_ham
        monkeypatch.setenv("VIBEQC_CASCI_BACKEND", "cpp")
        kw = dict(nuclear_repulsion=H.nuclear_repulsion, ms2=0, nroots=2)
        cold = casci(H.h1e, H.h2e, 6, 6, 1, **kw)
        warm = casci(H.h1e, H.h2e, 6, 6, 1, ci_guess=cold.ci_coeffs_all, **kw)
        for ec, ew in zip(cold.e_totals, warm.e_totals):
            assert abs(ec - ew) < 1e-9
        assert abs(abs(np.dot(cold.ci_coeffs, warm.ci_coeffs)) - 1.0) < 1e-8

    def test_wrong_guess_shape_raises(self, h2o_ham, monkeypatch):
        from vibeqc.solvers._casci import casci

        monkeypatch.setenv("VIBEQC_CASCI_BACKEND", "cpp")
        H = h2o_ham
        with pytest.raises(ValueError, match="ci_guess"):
            casci(
                H.h1e, H.h2e, 6, 6, 1,
                nuclear_repulsion=H.nuclear_repulsion, ms2=0,
                ci_guess=np.ones(7),
            )

    def test_casscf_macro_iterations_warm_start_consistent(self):
        # The CASSCF loop threads the previous CI vector into each CASCI
        # solve; the converged result must match the recorded LiH minimum
        # (same value as the cold-start orbital-step agreement test).
        LIH = Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])])
        b = BasisSet(LIH, "sto-3g")
        C = get_hf_orbital_provider(LIH, b, method="rhf")
        H = build_hamiltonian_mo(LIH, b, C)
        r = casscf(
            H.h1e, H.h2e, 2, 2, 1,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
        )
        assert r.converged
        assert abs(r.e_total - (-7.8812143431)) < 1e-8


@pytest.mark.slow
class TestLargeCASSCF:
    def test_n2_cas1010_converges_on_direct_core(self):
        # The 25d-e headline: CAS(10,10) CASSCF (63 504 determinants per CI
        # solve) — impossible on the dense CI path — converges through the
        # C++ engine with the auto Super-CI -> Newton-CG pipeline.
        b = BasisSet(N2, "6-31g")
        C = get_hf_orbital_provider(N2, b, method="rhf")
        H = build_hamiltonian_mo(N2, b, C)
        sc = casscf(
            H.h1e, H.h2e, 10, 10, 2,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0, max_macro=100,
        )
        assert sc.converged and sc.grad_norm < 1e-6
        # vibe-qc's stationary point (recorded 2026-06-10); PySCF converges
        # to a different, 8.13 mHa higher basin (-109.0533171021) — see
        # module docstring.  Variational floor: must stay above the
        # (deeper-CAS) FCI; here just pin the recorded value.
        assert abs(sc.e_total - (-109.0614472204)) < 5e-6
        # CASCI on HF orbitals (the starting point) must lie above.
        e_casci = casci(
            H.h1e, H.h2e, 10, 10, 2,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
        ).e_total
        assert sc.e_total < e_casci
