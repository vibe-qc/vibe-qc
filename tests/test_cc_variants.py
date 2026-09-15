"""Coupled-pair / QCI variants of the canonical DF-CCSD kernel (cc_variant=).

Covers CCD, LCCD, LCCSD (= CEPA(0)) and CEPA(1)/(2)/(3) in
cpp/src/ccsd.cpp, plus QCISD / QCISD(T).  Three validation layers:

1. **Spin-orbital oracle (live)** -- CCD / LCCD / LCCSD are re-derived here
   at spin-orbital level from the FCI-anchored SGWB reference residual
   (``vibeqc.dlpno._ccsd_ref.so_residuals``) under the same variant rules
   (T1 pinned at zero; exact linear-part extraction; QCISD monomial
   projection) and the converged correlation energies must agree with the
   C++ kernel.

2. **Linearization exactness (live)** -- the linear variants extract the
   degree-1 part of the canonical residual with the five-point odd
   stencil, exact because the residual is a polynomial of degree <= 4 in
   the joint amplitudes.  Exactness is proven by h-independence: two
   stencil step sizes must give the same linear part to machine
   precision.  If someone raises the polynomial degree of
   compute_residuals, this test fails loudly.

3. **Out-of-process ORCA parity (pinned)** -- the CEPA(1)/(2)/(3) EPV
   shift convention and QCISD/QCISD(T) corrections are pinned against
   ORCA 6.1 RI-MDCI runs with the IDENTICAL auxiliary basis, canonical
   orbitals and no frozen core.  ORCA input used (2026-07-02 for CEPA,
   2026-07-05 for QCISD, ORCA 6.1, out-of-process per CLAUDE.md
   section 10)::

       ! HF RI-CEPA/n def2-SVP def2-SVP/C VeryTightSCF NoFrozenCore
       ! HF RI-QCISD(T) def2-SVP def2-SVP/C VeryTightSCF NoFrozenCore
       %mdci Localize false end
       * xyz 0 1
       O 0 0 0
       H 0  0.793353 -0.613510
       H 0 -0.793353 -0.613510
       *

   NOTE: ORCA localizes internal valence orbitals (Foster-Boys) for CEPA
   by default; CEPA(n>=1) is not invariant under occupied rotations.
   vibe-qc's CEPA uses canonical MOs == ORCA's ``Localize false``
   (default-ORCA differs by ~0.3 mHa on this system from that choice).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from vibeqc import BasisSet, RHFOptions, run_rhf
from vibeqc._vibeqc_core import Atom, Molecule, dlpno_pair_residual
from vibeqc.cc import CCSDOptions, run_bccd, run_ccsd, run_uccsd
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno import _ccsd_ref as ref

ANGSTROM_TO_BOHR = 1.8897259886
AUX = "def2-svp-rifit"
MICRO_HA = 1e-6

H2O_ATOMS = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
    (1, [0.0, -0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
]

# ORCA 6.1 RI-CEPA/n correlation energies for the recipe in the module
# docstring (H2O/def2-SVP, def2-SVP/C aux, canonical orbitals, no frozen
# core).  E(HF): ORCA -75.954759703, vibe-qc -75.954759719 (1.6e-8 apart).
E_ORCA_RI_CEPA1 = -0.216130472
E_ORCA_RI_CEPA2 = -0.218146586
E_ORCA_RI_CEPA3 = -0.214203833

# ORCA 6.1 RI-QCISD(T) for the same recipe.  QCISD(T) in the original
# Pople/Head-Gordon/Raghavachari definition is E[T] + 2 E_ST on QCISD
# amplitudes; this pin catches accidental fallback to the CCSD(T)
# E[T] + E_ST combination.
E_ORCA_RI_QCISD = -0.216260599
E_ORCA_RI_QCISD_T = -0.003127162

# ORCA 6.1 AUTOCI-CC2 correlation energy for the same geometry and orbital
# basis, using canonical four-index integrals and no frozen core. ORCA input:
# ``! RHF AUTOCI-CC2 def2-SVP VeryTightSCF NoFrozenCore`` with
# ``%autoci STol 1e-9; MaxIter 200 end``.
E_ORCA_CC2 = -0.207664939

# Water/STO-3G all-electron BCCD(T), conventional integrals.  ORCA 6.1's
# one-shot ``BCCD(T)`` keyword is anomalous for this case (-0.001877196 Ha),
# but rerunning CCD(T) with singles disabled on its saved, converged Brueckner
# orbitals gives -0.000086104 Ha.  PySCF 2.13.1 BCCD followed by ccsd_t() gives
# -0.0000861036 Ha independently.  Both generalized triples kernels retain
# the noncanonical F_ov*T2 numerator that survives when Brueckner T1 is zero.
E_BCCD_T_H2O_STO3G = -0.000086103597


def _molecule():
    return Molecule(
        [Atom(z, p) for z, p in H2O_ATOMS], charge=0, multiplicity=1
    )


def _system(basis_name):
    mol = _molecule()
    basis = BasisSet(mol, basis_name)
    opts = RHFOptions()
    opts.max_iter = 100
    opts.conv_tol_energy = 1e-11
    rhf = run_rhf(mol, basis, opts)
    assert rhf.converged
    return mol, basis, rhf


def _run_variant(mol, basis, rhf, variant, **kw):
    opts = CCSDOptions(
        aux_basis=AUX,
        compute_triples=False,
        n_frozen_core=0,
        cc_variant=variant,
        **kw,
    )
    res = run_ccsd(mol, basis, rhf, opts)
    assert res.converged
    return res


# ---------------------------------------------------------------------------
# Polynomial projectors shared by the QCISD oracle
# ---------------------------------------------------------------------------


def _coefficient_weights(nodes, degree):
    powers = np.vander(np.asarray(nodes, dtype=float), increasing=True).T
    rhs = np.zeros(len(nodes))
    rhs[degree] = 1.0
    return np.linalg.solve(powers, rhs)


def _selected_monomial_residual(residual, t1, t2, keep_r1, keep_r2):
    x_nodes = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    y_nodes = np.array([-1.0, 0.0, 1.0])
    wx = [_coefficient_weights(x_nodes, d) for d in range(len(x_nodes))]
    wy = [_coefficient_weights(y_nodes, d) for d in range(len(y_nodes))]

    r1_acc = np.zeros_like(t1)
    r2_acc = np.zeros_like(t2)
    for ix, x in enumerate(x_nodes):
        for iy, y in enumerate(y_nodes):
            r1, r2 = residual(x * t1, y * t2)
            w1 = sum(wx[d1][ix] * wy[d2][iy] for d1, d2 in keep_r1)
            w2 = sum(wx[d1][ix] * wy[d2][iy] for d1, d2 in keep_r2)
            r1_acc += w1 * r1
            r2_acc += w2 * r2
    return r1_acc, r2_acc


# ---------------------------------------------------------------------------
# Spin-orbital oracle for the variants
# ---------------------------------------------------------------------------


def _so_setup(mol, basis, rhf):
    """Spin-orbital Fock + antisymmetrised ERI (occupied block first)."""
    df = DensityFitting(basis, BasisSet(mol, AUX), aux_basis_name=AUX)
    C = np.asarray(rhf.mo_coeffs)
    F = np.asarray(rhf.fock)
    f_mo = C.T @ F @ C
    B_mo = np.asarray(df.mo_transform(C, C))
    n_mo = f_mo.shape[0]
    nocc_sp = mol.n_electrons() // 2
    eri = ref._spin_orbital_eri(B_mo, n_mo)
    f_so = np.kron(f_mo, np.eye(2))
    occ = [2 * p + s for p in range(nocc_sp) for s in range(2)]
    vir = [2 * p + s for p in range(nocc_sp, n_mo) for s in range(2)]
    perm = occ + vir
    return (
        f_so[np.ix_(perm, perm)],
        eri[np.ix_(perm, perm, perm, perm)],
        len(occ),
        len(vir),
    )


def _so_iterate_variant(f_so, eri, no, nv, *, singles, linear, qci=False,
                        cc2=False,
                        max_iter=200, tol=1e-10):
    """Iterate the spin-orbital SGWB residual under the variant rules."""
    o, v = slice(0, no), slice(no, no + nv)
    fock_od = f_so.copy()
    np.fill_diagonal(fock_od, 0.0)
    eps = np.diag(f_so)
    D1 = eps[o, None] - eps[None, v]
    D2 = (
        eps[o][:, None, None, None]
        + eps[o][None, :, None, None]
        - eps[v][None, None, :, None]
        - eps[v][None, None, None, :]
    )

    def residual(t1, t2):
        return ref.so_residuals(f_so, fock_od, eri, t1, t2, o, v, D1, D2)

    z1 = np.zeros((no, nv))
    z2 = np.zeros((no, no, nv, nv))
    r1_0, r2_0 = residual(z1, z2)

    def linear_residual(t1, t2):
        # Five-point odd stencil: exact for the degree-4 polynomial
        # residual (same construction as compute_linear_residuals in
        # cpp/src/ccsd.cpp).
        r1p, r2p = residual(t1, t2)
        r1m, r2m = residual(-t1, -t2)
        r1p2, r2p2 = residual(2 * t1, 2 * t2)
        r1m2, r2m2 = residual(-2 * t1, -2 * t2)
        return (
            r1_0 + (8 * (r1p - r1m) - (r1p2 - r1m2)) / 12,
            r2_0 + (8 * (r2p - r2m) - (r2p2 - r2m2)) / 12,
        )

    def qci_residual(t1, t2):
        # QCISD term selection from Pople/Head-Gordon/Raghavachari:
        # keep T2^2 disconnected quadruples, drop T1*T2 disconnected
        # triples from the doubles projection, and use the CI-like energy.
        keep_r1 = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1))
        keep_r2 = ((0, 0), (1, 0), (0, 1), (2, 0), (0, 2))
        return _selected_monomial_residual(
            residual, t1, t2, keep_r1, keep_r2
        )

    def cc2_residual(t1, t2):
        # Christiansen-Koch-Jorgensen CC2: the singles projection is the
        # CCSD singles residual (already linear in T2), while doubles retain
        # only exp(-T1) H exp(T1) plus the diagonal [F,T2] commutator.
        r1, _ = residual(t1, t2)
        _, r2_t1 = residual(t1, z2)
        return r1, r2_t1 - D2 * t2

    def lin_energy(t1, t2):
        # Linear energy functional: no quadratic t1 t1 term.
        return float(
            np.einsum("ia,ia->", f_so[o, v], t1)
            + 0.25 * np.einsum("ijab,ijab->", eri[o, o, v, v], t2)
        )

    t1 = np.zeros((no, nv))
    t2 = eri[o, o, v, v] / D2
    hist_a, hist_r = [], []
    e_prev = 0.0
    for _ in range(max_iter):
        if cc2:
            r1, r2 = cc2_residual(t1, t2)
        elif qci:
            r1, r2 = qci_residual(t1, t2)
        elif linear:
            r1, r2 = linear_residual(t1, t2)
        else:
            r1, r2 = residual(t1, t2)
        if not singles:
            r1 = np.zeros_like(r1)
        t1n, t2n = t1 + r1 / D1, t2 + r2 / D2
        hist_a.append(np.concatenate([t1n.ravel(), t2n.ravel()]))
        hist_r.append(np.concatenate([r1.ravel(), r2.ravel()]))
        if len(hist_a) > 6:
            hist_a.pop(0)
            hist_r.pop(0)
        n = len(hist_a)
        if n >= 2:
            B = np.empty((n + 1, n + 1))
            B[-1, :] = -1.0
            B[:, -1] = -1.0
            B[-1, -1] = 0.0
            for a in range(n):
                for b in range(n):
                    B[a, b] = hist_r[a] @ hist_r[b]
            rhs = np.zeros(n + 1)
            rhs[-1] = -1.0
            try:
                c = np.linalg.solve(B, rhs)[:n]
                flat = sum(c[k] * hist_a[k] for k in range(n))
                t1n = flat[: t1.size].reshape(t1.shape)
                t2n = flat[t1.size:].reshape(t2.shape)
            except np.linalg.LinAlgError:
                pass
        t1, t2 = t1n, t2n
        e = (
            lin_energy(t1, t2)
            if linear or qci
            else ref.so_energy(f_so, eri, t1, t2, o, v)
        )
        rnorm = np.linalg.norm(r1) + np.linalg.norm(r2)
        if abs(e - e_prev) < tol and rnorm < 1e-8:
            return e, t1, t2
        e_prev = e
    raise RuntimeError("spin-orbital variant iteration did not converge")


class TestOracleSTO3G:
    """CC2 / CCD / LCCD / LCCSD vs the live spin-orbital oracle."""

    @pytest.fixture(scope="class")
    @classmethod
    def h2o(cls):
        mol, basis, rhf = _system("sto-3g")
        f_so, eri, no, nv = _so_setup(mol, basis, rhf)
        return mol, basis, rhf, f_so, eri, no, nv

    @pytest.mark.parametrize(
        "variant,singles,linear",
        [("ccd", False, False), ("lccd", False, True), ("lccsd", True, True)],
    )
    def test_variant_matches_spin_orbital_oracle(
        self, h2o, variant, singles, linear
    ):
        mol, basis, rhf, f_so, eri, no, nv = h2o
        e_so, _, _ = _so_iterate_variant(
            f_so, eri, no, nv, singles=singles, linear=linear
        )
        res = _run_variant(mol, basis, rhf, variant)
        assert abs(res.e_ccsd_correlation - e_so) < 5.0 * MICRO_HA

    def test_cc2_matches_spin_orbital_oracle(self, h2o):
        mol, basis, rhf, f_so, eri, no, nv = h2o
        e_so, t1_so, _ = _so_iterate_variant(
            f_so, eri, no, nv, singles=True, linear=False, cc2=True
        )
        res = _run_variant(mol, basis, rhf, "cc2")
        assert abs(res.e_ccsd_correlation - e_so) < 5.0 * MICRO_HA
        assert res.t1_norm == pytest.approx(
            np.linalg.norm(t1_so) / np.sqrt(2.0), abs=1e-7
        )

    def test_qcisd_matches_spin_orbital_oracle(self, h2o):
        mol, basis, rhf, f_so, eri, no, nv = h2o
        e_so, t1_so, _ = _so_iterate_variant(
            f_so, eri, no, nv, singles=True, linear=False, qci=True
        )
        res = _run_variant(mol, basis, rhf, "qcisd")
        assert abs(res.e_ccsd_correlation - e_so) < 5.0 * MICRO_HA
        assert res.t1_norm == pytest.approx(
            np.linalg.norm(t1_so) / np.sqrt(2.0), abs=1e-7
        )


class TestLinearExtractionExactness:
    """The stencil linearization is algebraically exact (h-independent)."""

    def test_h_independence_of_linear_part(self):
        mol, basis, rhf = _system("sto-3g")
        df = DensityFitting(basis, BasisSet(mol, AUX), aux_basis_name=AUX)
        C = np.asarray(rhf.mo_coeffs)
        F = np.asarray(rhf.fock)
        f_mo = C.T @ F @ C
        no = mol.n_electrons() // 2
        n_mo = C.shape[1]
        nv = n_mo - no
        B = np.asarray(df.mo_transform(C, C))
        naux = B.shape[0]
        B_ov = np.ascontiguousarray(B[:, :no, no:].reshape(naux, no * nv))
        B_oo = np.ascontiguousarray(B[:, :no, :no].reshape(naux, no * no))
        B_vv = np.ascontiguousarray(B[:, no:, no:].reshape(naux, nv * nv))
        f_oo = np.ascontiguousarray(f_mo[:no, :no])
        f_vv = np.ascontiguousarray(f_mo[no:, no:])
        f_ov = np.ascontiguousarray(f_mo[:no, no:])

        def resid(t1, t2f):
            r1, r2 = dlpno_pair_residual(
                np.ascontiguousarray(t1),
                np.ascontiguousarray(t2f),
                B_ov, B_oo, B_vv, f_oo, f_vv, f_ov,
            )
            return np.asarray(r1), np.asarray(r2)

        rng = np.random.default_rng(7)
        t1 = 0.05 * rng.standard_normal((no, nv))
        t2 = 0.05 * rng.standard_normal((no * no, nv * nv))
        # enforce t_ij^ab = t_ji^ba (the alpha-beta amplitude symmetry)
        t2v = t2.reshape(no, no, nv, nv)
        t2v = 0.5 * (t2v + t2v.transpose(1, 0, 3, 2))
        t2 = t2v.reshape(no * no, nv * nv)

        def lin(h):
            r1p, r2p = resid(h * t1, h * t2)
            r1m, r2m = resid(-h * t1, -h * t2)
            r1p2, r2p2 = resid(2 * h * t1, 2 * h * t2)
            r1m2, r2m2 = resid(-2 * h * t1, -2 * h * t2)
            return (
                (8 * (r1p - r1m) - (r1p2 - r1m2)) / (12 * h),
                (8 * (r2p - r2m) - (r2p2 - r2m2)) / (12 * h),
            )

        l1a, l2a = lin(1.0)
        l1b, l2b = lin(0.5)
        assert np.abs(l1a - l1b).max() < 1e-9
        assert np.abs(l2a - l2b).max() < 1e-9


class TestCEPAOrcaParity:
    """CEPA(1)/(2)/(3) pinned to out-of-process ORCA RI-CEPA/n."""

    @pytest.fixture(scope="class")
    @classmethod
    def h2o(cls):
        return _system("def2-svp")

    @pytest.mark.parametrize(
        "variant,e_orca",
        [
            ("cepa(1)", E_ORCA_RI_CEPA1),
            ("cepa(2)", E_ORCA_RI_CEPA2),
            ("cepa(3)", E_ORCA_RI_CEPA3),
        ],
    )
    def test_cepa_matches_orca(self, h2o, variant, e_orca):
        mol, basis, rhf = h2o
        res = _run_variant(mol, basis, rhf, variant)
        # 0.02/0.03/0.00 microHa at generation time; 1 microHa tolerance
        # leaves headroom for platform noise while catching any change in
        # the shift convention (smallest known convention error: ~50 uHa).
        assert abs(res.e_ccsd_correlation - e_orca) < 1.0 * MICRO_HA


class TestCC2OrcaParity:
    """Ground-state CC2 pinned to out-of-process ORCA AUTOCI-CC2."""

    def test_cc2_matches_orca(self):
        mol, basis, rhf = _system("def2-svp")
        res = run_ccsd(
            mol,
            basis,
            rhf,
            CCSDOptions(
                density_fit=False,
                compute_triples=False,
                n_frozen_core=0,
                cc_variant="cc2",
                conv_tol_energy=1e-11,
                conv_tol_residual=1e-9,
            ),
        )
        assert res.converged
        assert abs(res.e_ccsd_correlation - E_ORCA_CC2) < 0.02 * MICRO_HA


class TestQCISDOrcaParity:
    """QCISD/QCISD(T) pinned to out-of-process ORCA RI-QCISD(T)."""

    @pytest.fixture(scope="class")
    @classmethod
    def h2o(cls):
        return _system("def2-svp")

    def test_qcisd_matches_orca(self, h2o):
        mol, basis, rhf = h2o
        res = _run_variant(mol, basis, rhf, "qcisd")
        # 3.84 microHa at generation time vs ORCA RI-QCISD; keep this
        # cross-code pin tight enough to catch term-selection drift while
        # leaving room for the small ORCA/vibe-qc implementation spread.
        assert abs(res.e_ccsd_correlation - E_ORCA_RI_QCISD) < 10.0 * MICRO_HA

    def test_qcisd_t_matches_orca_triples_increment(self, h2o):
        mol, basis, rhf = h2o
        opts = CCSDOptions(
            aux_basis=AUX,
            compute_triples=True,
            n_frozen_core=0,
            cc_variant="qcisd",
        )
        res = run_ccsd(mol, basis, rhf, opts)
        assert res.converged
        assert abs(res.e_ccsd_correlation - E_ORCA_RI_QCISD) < 10.0 * MICRO_HA
        assert abs(res.e_t - E_ORCA_RI_QCISD_T) < 2.0 * MICRO_HA


class TestVariantStructure:
    @pytest.fixture(scope="class")
    @classmethod
    def h2o(cls):
        return _system("sto-3g")

    def test_lccsd_is_cepa0(self, h2o):
        mol, basis, rhf = h2o
        e_lccsd = _run_variant(mol, basis, rhf, "lccsd").e_ccsd_correlation
        e_cepa0 = _run_variant(mol, basis, rhf, "cepa(0)").e_ccsd_correlation
        assert e_lccsd == pytest.approx(e_cepa0, abs=1e-12)

    def test_doubles_only_variants_keep_t1_zero(self, h2o):
        mol, basis, rhf = h2o
        for variant in ("ccd", "lccd"):
            assert _run_variant(mol, basis, rhf, variant).t1_norm == 0.0

    def test_physical_ordering(self, h2o):
        """|CCD| < |CCSD|; linearization overbinds: CEPA(0) most negative;
        the EPV shifts damp it back toward (and CEPA(3) past) CCSD."""
        mol, basis, rhf = h2o
        e = {
            v: _run_variant(mol, basis, rhf, v).e_ccsd_correlation
            for v in ("ccsd", "ccd", "lccd", "lccsd", "cepa(1)", "cepa(2)",
                      "cepa(3)")
        }
        assert e["ccd"] > e["ccsd"]              # dropping singles loses corr
        assert e["lccd"] < e["ccd"]              # linearization overbinds
        assert e["lccsd"] < e["lccd"]            # linear singles add more
        # standard CEPA ordering on a well-behaved closed shell
        assert e["lccsd"] < e["cepa(2)"] < e["cepa(1)"] < e["cepa(3)"]

    def test_ccsd_default_variant_unchanged(self, h2o):
        """cc_variant='ccsd' is the default and matches an explicit run."""
        mol, basis, rhf = h2o
        opts_default = CCSDOptions(
            aux_basis=AUX,
            compute_triples=False,
            n_frozen_core=0,
        )
        e_default = run_ccsd(mol, basis, rhf, opts_default).e_ccsd_correlation
        e_explicit = _run_variant(mol, basis, rhf, "ccsd").e_ccsd_correlation
        assert e_default == pytest.approx(e_explicit, abs=1e-12)


class TestBruecknerCCD:
    @pytest.mark.parametrize(
        ("field", "value"),
        [
            (field, value)
            for field in ("brueckner_tol", "brueckner_damping")
            for value in (-1.0, 0.0, float("nan"), float("inf"), float("-inf"))
        ],
    )
    def test_bccd_rejects_invalid_controls_before_integrals(
        self, monkeypatch, field, value
    ):
        mol = SimpleNamespace(multiplicity=1, n_electrons=lambda: 2)
        basis = SimpleNamespace(name="sto-3g")
        rhf = SimpleNamespace(converged=True)

        def unexpected_integral_work(*args, **kwargs):
            pytest.fail("BCCD integral work started before option validation")

        monkeypatch.setattr(
            "vibeqc._vibeqc_core.compute_kinetic", unexpected_integral_work
        )
        options = CCSDOptions(
            density_fit=False,
            compute_triples=False,
            n_frozen_core=0,
            **{field: value},
        )

        with pytest.raises(
            ValueError,
            match=rf"run_bccd: {field} must be finite and positive",
        ):
            run_bccd(mol, basis, rhf, options)

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("brueckner_tol", 1.0e-12),
            ("brueckner_damping", 2.0),
        ],
    )
    def test_bccd_preserves_positive_finite_controls(
        self, monkeypatch, field, value
    ):
        class IntegralWorkReached(RuntimeError):
            pass

        mol = SimpleNamespace(multiplicity=1, n_electrons=lambda: 2)
        basis = SimpleNamespace(name="sto-3g")
        rhf = SimpleNamespace(converged=True)

        def stop_at_integral_work(*args, **kwargs):
            raise IntegralWorkReached

        monkeypatch.setattr(
            "vibeqc._vibeqc_core.compute_kinetic", stop_at_integral_work
        )
        options = CCSDOptions(
            density_fit=False,
            compute_triples=False,
            n_frozen_core=0,
            **{field: value},
        )

        with pytest.raises(IntegralWorkReached):
            run_bccd(mol, basis, rhf, options)

    def test_bccd_drives_ccsd_singles_below_tolerance(self):
        mol = Molecule(
            [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
            charge=0,
            multiplicity=1,
        )
        basis = BasisSet(mol, "cc-pvdz")
        rhf = run_rhf(mol, basis, RHFOptions())
        assert rhf.converged

        probe = run_ccsd(
            mol, basis, rhf,
            CCSDOptions(compute_triples=False, n_frozen_core=0),
        )
        bccd = run_bccd(
            mol,
            basis,
            rhf,
            CCSDOptions(
                compute_triples=False,
                n_frozen_core=0,
                brueckner_max_iter=20,
                brueckner_tol=1e-6,
            ),
        )

        assert probe.t1_norm > 1e-3
        assert bccd.brueckner_t1_norm < 1e-6
        assert bccd.t1_norm == pytest.approx(bccd.brueckner_t1_norm)
        assert bccd.brueckner_iterations > 0
        assert bccd.converged
        assert all(
            step.next_t1_norm < step.t1_norm
            for step in bccd.brueckner_trace
        )

    def test_bccd_t_includes_noncanonical_fock_doubles_term(self):
        mol, basis, rhf = _system("sto-3g")
        bccd_t = run_bccd(
            mol,
            basis,
            rhf,
            CCSDOptions(
                density_fit=False,
                compute_triples=True,
                n_frozen_core=0,
                brueckner_max_iter=40,
                brueckner_tol=1e-8,
            ),
        )

        assert bccd_t.converged
        assert abs(bccd_t.e_t - E_BCCD_T_H2O_STO3G) < 0.01 * MICRO_HA
        assert bccd_t.e_t == pytest.approx(
            bccd_t.e_t4 + bccd_t.e_t5_st, abs=1e-12
        )
        assert bccd_t.e_t5_st < 0.0
        assert bccd_t.triples_memory_mode_used == "fast"
        assert bccd_t.triples_tile_size_used > 0
        assert bccd_t.triples_threads_used > 0
        assert bccd_t.triples_workspace_bytes > 0
        assert bccd_t.triples_disk_bytes == 0

    def test_bccd_t_noncanonical_split_matches_bounded_modes(self, tmp_path):
        mol, basis, rhf = _system("sto-3g")

        def run(mode):
            scratch = tmp_path / mode
            scratch.mkdir()
            result = run_bccd(
                mol,
                basis,
                rhf,
                CCSDOptions(
                    density_fit=True,
                    aux_basis=AUX,
                    compute_triples=True,
                    n_frozen_core=0,
                    brueckner_max_iter=40,
                    brueckner_tol=1e-8,
                    triples_memory_mode=mode,
                    requested_memory_bytes=64 * 1024**2,
                    triples_tile_size=1,
                    triples_max_threads=1,
                    triples_scratch_directory=str(scratch),
                ),
            )
            assert list(scratch.iterdir()) == []
            return result

        reference = run("fast")
        assert reference.e_t5_st < 0.0
        for mode in ("blocked", "direct", "disk"):
            bounded = run(mode)
            assert bounded.triples_memory_mode_used == mode
            assert bounded.e_t4 == pytest.approx(reference.e_t4, abs=1e-12)
            assert bounded.e_t5_st == pytest.approx(
                reference.e_t5_st,
                abs=1e-12,
            )
            assert bounded.e_t == pytest.approx(reference.e_t, abs=1e-12)


class TestBracketTriples:
    """CCSD[T] (= CCSD+T(CCSD)): the fourth-order piece of (T).

    The kernel reports E(T) split into the fourth-order bracket piece
    E[T] (e_t4) and the fifth-order singles-triples coupling E_ST
    (e_t5_st).  The split is validated against the spin-orbital oracle:
    On canonical RHF orbitals f_ov is zero, so Wd is linear in t1 and
    evaluating the oracle (T) formula with t1 zeroed gives exactly E[T].
    """

    @pytest.fixture(scope="class")
    @classmethod
    def h2o(cls):
        mol, basis, rhf = _system("sto-3g")
        f_so, eri, no, nv = _so_setup(mol, basis, rhf)
        return mol, basis, rhf, f_so, eri, no, nv

    def test_split_identity_and_selection(self, h2o):
        mol, basis, rhf = h2o[:3]
        res_t = run_ccsd(
            mol, basis, rhf,
            CCSDOptions(
                aux_basis=AUX,
                triples="(t)",
                n_frozen_core=0,
            ),
        )
        res_b = run_ccsd(
            mol, basis, rhf,
            CCSDOptions(
                aux_basis=AUX,
                triples="[t]",
                n_frozen_core=0,
            ),
        )
        # the amplitudes are identical, so the pieces are too
        assert res_t.e_t4 == pytest.approx(res_b.e_t4, abs=1e-12)
        assert res_t.e_t5_st == pytest.approx(res_b.e_t5_st, abs=1e-12)
        # selection: (t) sums both pieces, [t] takes the fourth order only
        assert res_t.e_t == pytest.approx(res_t.e_t4 + res_t.e_t5_st,
                                          abs=1e-12)
        assert res_b.e_t == pytest.approx(res_b.e_t4, abs=1e-12)
        assert res_b.e_t5_st != 0.0

    def test_bracket_matches_spin_orbital_oracle(self, h2o):
        mol, basis, rhf, f_so, eri, no, nv = h2o
        _, t1, t2 = _so_iterate_variant(
            f_so, eri, no, nv, singles=True, linear=False
        )
        o, v = slice(0, no), slice(no, no + nv)
        eps_so = np.diag(f_so)
        e_t_so = ref.so_triples_correction(eps_so, eri, t1, t2, o, v)
        e_bracket_so = ref.so_triples_correction(
            eps_so, eri, np.zeros_like(t1), t2, o, v
        )
        res = run_ccsd(
            mol,
            basis,
            rhf,
            CCSDOptions(
                aux_basis=AUX,
                triples="(t)",
                n_frozen_core=0,
            ),
        )
        assert abs(res.e_t4 - e_bracket_so) < 1.0 * MICRO_HA
        assert abs(res.e_t - e_t_so) < 1.0 * MICRO_HA
        assert abs(res.e_t5_st - (e_t_so - e_bracket_so)) < 1.0 * MICRO_HA

    def test_selector_spellings(self):
        opts = CCSDOptions(triples="[t]")
        assert opts.compute_triples and opts.triples == "[t]"
        opts2 = CCSDOptions(triples="+T(CCSD)")
        assert opts2.compute_triples and opts2.triples == "[t]"
        opts3 = CCSDOptions(triples="(t)")
        assert opts3.triples == "(t)"
        opts4 = CCSDOptions(triples="a-ccsd(t)")
        assert opts4.compute_triples and opts4.triples == "a-ccsd(t)"

    def test_run_job_bracket_block_and_citation(self, tmp_path):
        from vibeqc import run_job

        mol = _molecule()
        out = tmp_path / "h2o_bracket"
        res = run_job(
            mol,
            basis="cc-pvdz",
            method="ccsd",
            triples="[t]",
            output=str(out),
        )
        assert res.ccsd.converged
        assert res.ccsd.e_t == pytest.approx(res.ccsd.e_t4, abs=1e-12)
        text = (tmp_path / "h2o_bracket.out").read_text()
        assert "Coupled-Cluster CCSD[T]" in text
        assert "E[T] correction" in text
        assert "Urban" in text  # urban_ccsd_t_1985 fired

    def test_open_shell_bracket_raises(self, tmp_path):
        from vibeqc import run_job

        oh = Molecule(
            [Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.8])],
            charge=0,
            multiplicity=2,
        )
        with pytest.raises(NotImplementedError, match="closed-shell"):
            run_job(
                oh,
                basis="cc-pvdz",
                method="ccsd(t)",
                triples="[t]",
                output=str(tmp_path / "oh_bracket"),
            )


class TestRunJobRouting:
    """run_job dispatch, output block, and citation firing for variants."""

    def test_run_job_cc2_block_and_citation(self, tmp_path):
        from vibeqc import run_job

        mol = _molecule()
        out = tmp_path / "h2o_cc2"
        res = run_job(mol, basis="cc-pvdz", method="cc2", output=str(out))
        assert res.ccsd.converged
        assert res.ccsd.t1_norm > 0.0
        text = (tmp_path / "h2o_cc2.out").read_text()
        assert "Approximate Coupled-Cluster CC2" in text
        assert "Christiansen" in text
        bib = (tmp_path / "h2o_cc2.bibtex").read_text()
        assert "christiansen_cc2_1995" in bib

    def test_run_job_cepa1_block_and_citations(self, tmp_path):
        from vibeqc import run_job

        mol = _molecule()
        out = tmp_path / "h2o_cepa1"
        res = run_job(mol, basis="cc-pvdz", method="cepa(1)", output=str(out))
        assert res.ccsd.converged
        assert res.ccsd.t1_norm > 0.0
        text = (tmp_path / "h2o_cepa1.out").read_text()
        assert "Coupled-Pair / Coupled-Cluster CEPA(1)" in text
        assert "Meyer" in text            # meyer_cepa_1973 fired
        assert "Wennmohs" in text         # wennmohs_neese_cepa_2008 fired
        bib = (tmp_path / "h2o_cepa1.bibtex").read_text()
        assert "meyer_cepa_1973" in bib

    def test_run_job_citype_routes_ccd(self, tmp_path):
        from vibeqc import run_job

        mol = _molecule()
        out = tmp_path / "h2o_ccd"
        res = run_job(
            mol, basis="cc-pvdz", method="ci", citype="ccd", output=str(out)
        )
        assert res.ccsd.converged
        assert res.ccsd.t1_norm == 0.0
        text = (tmp_path / "h2o_ccd.out").read_text()
        assert "Coupled-Pair / Coupled-Cluster CCD" in text

    def test_run_job_variant_rejects_triples(self, tmp_path):
        from vibeqc import run_job

        mol = _molecule()
        with pytest.raises(ValueError, match="triples="):
            run_job(
                mol,
                basis="cc-pvdz",
                method="ccd",
                triples="(t)",
                output=str(tmp_path / "x"),
            )

    def test_run_job_variant_rejects_open_shell(self, tmp_path):
        from vibeqc import run_job

        oh = Molecule(
            [Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.8])],
            charge=0,
            multiplicity=2,
        )
        with pytest.raises(NotImplementedError, match="closed-shell"):
            run_job(
                oh,
                basis="cc-pvdz",
                method="cepa(1)",
                output=str(tmp_path / "y"),
            )


class TestGates:
    @pytest.fixture(scope="class")
    @classmethod
    def h2o(cls):
        return _system("sto-3g")

    def test_triples_with_variant_raises(self, h2o):
        mol, basis, rhf = h2o
        opts = CCSDOptions(aux_basis=AUX, compute_triples=True,
                           cc_variant="ccd")
        with pytest.raises(Exception, match="compute_triples|\\(T\\)"):
            run_ccsd(mol, basis, rhf, opts)

    def test_qcisd_rejects_bracket_triples(self, h2o):
        mol, basis, rhf = h2o
        opts = CCSDOptions(aux_basis=AUX, triples="[t]", cc_variant="qcisd")
        with pytest.raises(Exception, match="triples_variant|QCISD\\(T\\)"):
            run_ccsd(mol, basis, rhf, opts)

    def test_unknown_variant_raises(self, h2o):
        mol, basis, rhf = h2o
        opts = CCSDOptions(aux_basis=AUX, compute_triples=False,
                           cc_variant="qcisd(tq)")
        with pytest.raises(Exception, match="cc_variant"):
            run_ccsd(mol, basis, rhf, opts)

    def test_uccsd_rejects_variants(self):
        from vibeqc import UHFOptions, run_uhf

        mol = Molecule(
            [Atom(8, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.8])],
            charge=0,
            multiplicity=2,
        )
        basis = BasisSet(mol, "sto-3g")
        uhf = run_uhf(mol, basis, UHFOptions())
        assert uhf.converged
        opts = CCSDOptions(aux_basis=AUX, compute_triples=False,
                           cc_variant="cepa(1)")
        with pytest.raises(Exception, match="closed-shell"):
            run_uccsd(mol, basis, uhf, opts)
