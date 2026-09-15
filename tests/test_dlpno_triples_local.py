"""Local DLPNO-(T) on converged local DLPNO-CCSD amplitudes (M3c).

`triples_local.local_triples_correction` evaluates the (T) correction
per occupied triple in a TNO domain, reusing the FCI-anchored per-triple
kernel `_ccsd_ref.so_triple_energy`. The (T) analogue of the CCSD parity
ratchet:

* with **canonical occupieds and full domains** every triple spans the
  full virtual space and the sum reproduces canonical CCSD(T) exactly;
* the **Boys-localised** production path carries the standard DLPNO-(T0)
  semicanonical approximation (diagonal localised Fock in the
  denominators) plus the TNO-domain truncation, validated by recovery.

Canonical (T) reference: `_ccsd_ref.run_ref_ccsd(compute_triples=True)`,
FCI-anchored (CCSD(T) ⊃ the exact (T) for the closed-shell reference).
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from vibeqc import BasisSet, RHFOptions, run_rhf
from vibeqc._vibeqc_core import Atom, Molecule
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno._ccsd_ref import run_ref_ccsd
from vibeqc.dlpno.ccsd_local_solver import (
    LocalCCSDOptions as _LocalCCSDOptions,
    run_local_dlpno_ccsd,
)
from vibeqc.dlpno import triples_local
from vibeqc.dlpno.triples_local import (
    _HAVE_CPP_SPATIAL_TRIPLE,
    _HAVE_CPP_SPATIAL_TRIPLES,
    _local_distinct_triple_keys,
    _ordered_triples_for_key,
    _spatial_triple_energy,
    _spatial_triples_correction,
    _t1_triples_energy,
)

A = 1.8897259886
AUX = "def2-svp-rifit"

# The local-(T) recovery figures were established before #140/#448. Preserve
# their all-electron LocalCCSD threshold convention rather than rebaselining.
LocalCCSDOptions = partial(
    _LocalCCSDOptions,
    n_frozen=0,
    tcut_pno=1e-7,
    tcut_mkn=0.0,
    tcut_pairs=1e-4,
    residual_domain="pair",
)
H2O = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.793353 * A, -0.613510 * A]),
    (1, [0.0, -0.793353 * A, -0.613510 * A]),
]
H2 = [(1, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 1.4])]


class TestTripleIteration:
    """Memory-bounded iteration over unique TNO builds."""

    def test_distinct_keys_cover_full_ordered_triple_space(self):
        n_act = 4
        keys = list(_local_distinct_triple_keys(n_act, None, 0.0))
        triples = {
            triple for key in keys for triple in _ordered_triples_for_key(key)
        }
        expected = {
            (i, j, k)
            for i in range(n_act)
            for j in range(n_act)
            for k in range(n_act)
        }

        assert len(keys) == 14  # C(4,1) + C(4,2) + C(4,3)
        assert triples == expected

    def test_distinct_keys_apply_occupied_locality_screen(self):
        occ_dist = np.array(
            [
                [0.0, 1.0, 5.0],
                [1.0, 0.0, 5.0],
                [5.0, 5.0, 0.0],
            ]
        )

        keys = list(_local_distinct_triple_keys(3, occ_dist, 2.0))

        assert keys == [(0,), (1,), (2,), (0, 1)]
        triples = {
            triple for key in keys for triple in _ordered_triples_for_key(key)
        }
        assert (0, 1, 0) in triples
        assert (0, 2, 0) not in triples

    def test_native_spatial_triples_matches_python_oracle(self):
        """The DLPNO triples native helper is the Python oracle, bit-tight."""
        assert _HAVE_CPP_SPATIAL_TRIPLES
        rng = np.random.default_rng(20260709)
        no, nv, naux = 3, 4, 6
        t1 = 0.02 * rng.normal(size=(no, nv))
        t2_raw = 0.03 * rng.normal(size=(no, no, nv, nv))
        t2 = 0.5 * (t2_raw + t2_raw.transpose(1, 0, 3, 2))
        B_ov = rng.normal(size=(naux, no, nv))
        B_oo_raw = rng.normal(size=(naux, no, no))
        B_oo = 0.5 * (B_oo_raw + B_oo_raw.transpose(0, 2, 1))
        B_vv_raw = rng.normal(size=(naux, nv, nv))
        B_vv = 0.5 * (B_vv_raw + B_vv_raw.transpose(0, 2, 1))
        eps_o = np.array([-0.72, -0.54, -0.39])
        eps_v = np.array([0.21, 0.34, 0.58, 0.91])

        cpp = _spatial_triples_correction(
            t1, t2, B_ov, B_oo, B_vv, eps_o, eps_v, use_cpp=True
        )
        py = _spatial_triples_correction(
            t1, t2, B_ov, B_oo, B_vv, eps_o, eps_v, use_cpp=False
        )

        assert cpp == pytest.approx(py, abs=1e-11)

    def test_native_spatial_triple_matches_python_oracle(self):
        """The ordered-triple native microkernel matches the Python oracle."""
        assert _HAVE_CPP_SPATIAL_TRIPLE
        rng = np.random.default_rng(20260710)
        no, nv = 3, 4
        t1 = 0.02 * rng.normal(size=(no, nv))
        t2 = 0.03 * rng.normal(size=(no, no, nv, nv))
        ovvv = rng.normal(size=(no, nv, nv, nv))
        ooov = rng.normal(size=(no, no, no, nv))
        ovov = rng.normal(size=(no, nv, no, nv))
        eps_o = np.array([-0.72, -0.54, -0.39])
        eps_v = np.array([0.21, 0.34, 0.58, 0.91])

        for triple in [(0, 0, 0), (0, 1, 2), (2, 1, 0)]:
            cpp = _spatial_triple_energy(
                *triple, t1, t2, ovvv, ooov, ovov, eps_o, eps_v, use_cpp=True
            )
            py = _spatial_triple_energy(
                *triple, t1, t2, ovvv, ooov, ovov, eps_o, eps_v, use_cpp=False
            )

            assert cpp == pytest.approx(py, abs=1e-11)

    def test_t1_tno_branch_uses_native_ordered_triple_kernel(self, monkeypatch):
        """The TNO-truncated `(T1)` loop does not call the Python oracle."""
        assert _HAVE_CPP_SPATIAL_TRIPLE
        rng = np.random.default_rng(20260711)
        no, nv, naux = 3, 4, 5
        nbf = no + nv
        C_loc = np.eye(nbf, no)
        C_vir = np.eye(nbf, nv, k=no)

        U = {}
        T2 = {}
        for i in range(no):
            for j in range(i, no):
                U[(i, j)] = np.eye(nv)
                raw = 0.02 * rng.normal(size=(nv, nv))
                T2[(i, j)] = 0.5 * (raw + raw.T) if i == j else raw
        t1 = {i: 0.01 * rng.normal(size=nv) for i in range(no)}

        B_ov = rng.normal(size=(naux, no, nv))
        B_oo_raw = rng.normal(size=(naux, no, no))
        B_oo = 0.5 * (B_oo_raw + B_oo_raw.transpose(0, 2, 1))
        B_vv_raw = rng.normal(size=(naux, nv, nv))
        B_vv = 0.5 * (B_vv_raw + B_vv_raw.transpose(0, 2, 1))

        class DummyDF:
            def mo_transform(self, left, right):
                if left is C_loc and right is C_vir:
                    return B_ov
                if left is C_loc and right is C_loc:
                    return B_oo
                if left is C_vir and right is C_vir:
                    return B_vv
                raise AssertionError("unexpected MO transform block")

        def fail_python_oracle(*_args, **_kwargs):
            raise AssertionError("Python cs_triple_energy should not be called")

        monkeypatch.setattr(triples_local, "cs_triple_energy", fail_python_oracle)

        e_t = _t1_triples_energy(
            U,
            T2,
            t1,
            np.eye(no),
            C_loc,
            C_vir,
            np.eye(nbf),
            np.diag(np.array([0.21, 0.34, 0.58, 0.91])),
            np.array([-0.72, -0.54, -0.39]),
            DummyDF(),
            no,
            tcut_tno=1e-4,
        )

        assert np.isfinite(e_t)

    def test_native_tno_density_matches_numpy_oracle(self, monkeypatch):
        """The local-(T0) native TNO density is live and matches NumPy."""
        assert triples_local._HAVE_CPP_TNO_DENSITY
        assert triples_local._cpp_build_tno_density is not None
        rng = np.random.default_rng(20260825)
        no, nv, naux = 3, 4, 5
        nbf = no + nv
        C_loc = np.eye(nbf, no)
        C_vir = np.eye(nbf, nv, k=no)

        U = {}
        T2 = {}
        for i in range(no):
            for j in range(i, no):
                U[(i, j)] = np.eye(nv)
                raw = 0.03 * rng.normal(size=(nv, nv))
                T2[(i, j)] = 0.5 * (raw + raw.T) if i == j else raw
        t1 = {i: 0.01 * rng.normal(size=nv) for i in range(no)}

        B_ov = 0.1 * rng.normal(size=(naux, no, nv))
        B_oo_raw = 0.1 * rng.normal(size=(naux, no, no))
        B_oo = 0.5 * (B_oo_raw + B_oo_raw.transpose(0, 2, 1))
        B_vv_raw = 0.1 * rng.normal(size=(naux, nv, nv))
        B_vv = 0.5 * (B_vv_raw + B_vv_raw.transpose(0, 2, 1))

        class DummyDF:
            def mo_transform(self, left, right):
                if left is C_loc and right is C_vir:
                    return B_ov
                if left is C_loc and right is C_loc:
                    return B_oo
                if left is C_vir and right is C_vir:
                    return B_vv
                raise AssertionError("unexpected MO transform block")

        native_density = triples_local._cpp_build_tno_density
        density_keys = []

        def spy_native_density(*args, **kwargs):
            density_keys.append(tuple(args[1]))
            return native_density(*args, **kwargs)

        monkeypatch.setattr(
            triples_local, "_cpp_build_tno_density", spy_native_density
        )
        common = dict(
            U=U,
            T2=T2,
            t1=t1,
            C_loc=C_loc,
            C_vir=C_vir,
            S=np.eye(nbf),
            f_vv_full=np.diag([0.21, 0.34, 0.58, 0.91]),
            f_dd=np.array([-0.72, -0.54, -0.39]),
            df=DummyDF(),
            n_act=no,
            tcut_tno=1e-4,
        )

        native_stats = {}
        native_energy = triples_local.local_triples_correction(
            **common, stats=native_stats
        )

        expected_keys = list(_local_distinct_triple_keys(no, None, 0.0))
        assert density_keys == expected_keys

        monkeypatch.setattr(triples_local, "_HAVE_CPP_TNO_DENSITY", False)
        numpy_stats = {}
        numpy_energy = triples_local.local_triples_correction(
            **common, stats=numpy_stats
        )

        assert native_energy == pytest.approx(numpy_energy, abs=1e-14)
        assert native_stats == numpy_stats


def _setup(atoms, basis_name):
    mol = Molecule([Atom(z, p) for z, p in atoms], charge=0, multiplicity=1)
    basis = BasisSet(mol, basis_name)
    opts = RHFOptions()
    opts.max_iter = 100
    opts.conv_tol_energy = 1e-11
    rhf = run_rhf(mol, basis, opts)
    assert rhf.converged
    df = DensityFitting(basis, BasisSet(mol, AUX), aux_basis_name=AUX)
    C = np.asarray(rhf.mo_coeffs)
    F = np.asarray(rhf.fock)
    n_occ = mol.n_electrons() // 2
    ref = run_ref_ccsd(
        C.T @ F @ C,
        np.asarray(df.mo_transform(C, C)),
        n_occ,
        e_hf=rhf.energy,
        compute_triples=True,
    )
    return mol, basis, rhf, df, ref


class TestLocalTriplesRatchet:
    """Canonical-occupied full-domain local (T) == canonical (T)."""

    def test_full_domain_equals_canonical_t(self):
        mol, b, rhf, df, ref = _setup(H2O, "sto-3g")
        r = run_local_dlpno_ccsd(
            mol,
            b,
            rhf,
            df,
            LocalCCSDOptions(
                localise="none",
                tcut_pno=0.0,
                tcut_mkn=0.0,
                coupling_radius=0.0,
                tcut_pairs=0.0,
                compute_triples=True,
            ),
        )
        assert abs(r.e_t - ref.e_t) < 1e-9, (r.e_t, ref.e_t)

    def test_full_domain_equals_canonical_t_dz(self):
        mol, b, rhf, df, ref = _setup(H2O, "def2-svp")
        r = run_local_dlpno_ccsd(
            mol,
            b,
            rhf,
            df,
            LocalCCSDOptions(
                localise="none",
                tcut_pno=0.0,
                tcut_mkn=0.0,
                coupling_radius=0.0,
                tcut_pairs=0.0,
                compute_triples=True,
            ),
        )
        assert abs(r.e_t - ref.e_t) < 1e-9, (r.e_t, ref.e_t)

    def test_two_electrons_no_triples(self):
        # H2 has one spatial occupied: the only triple (0,0,0) is degenerate
        # and cancels to floating-point noise, so (T) vanishes.
        mol, b, rhf, df, ref = _setup(H2, "def2-svp")
        r = run_local_dlpno_ccsd(
            mol,
            b,
            rhf,
            df,
            LocalCCSDOptions(
                localise="boys",
                tcut_pno=0.0,
                tcut_mkn=0.0,
                coupling_radius=0.0,
                tcut_pairs=0.0,
                compute_triples=True,
            ),
        )
        assert abs(r.e_t) < 1e-12
        assert ref.e_t == pytest.approx(0.0, abs=1e-12)


class TestLocalTriplesRecovery:
    """Boys-localised production (T): DLPNO-(T0) + TNO truncation."""

    def test_t0_recovery(self):
        mol, b, rhf, df, ref = _setup(H2O, "def2-svp")
        r = run_local_dlpno_ccsd(
            mol,
            b,
            rhf,
            df,
            LocalCCSDOptions(localise="boys", tcut_pno=1e-7, compute_triples=True),
        )
        # (T0) semicanonical + TNO-domain approximation: a documented
        # fraction of canonical (T), not the exact value. The absolute
        # deficit is sub-0.2 mHa here; the relative recovery is amplified
        # on this tiny (T) (~3 mHa). Larger systems recover more.
        recovery = r.e_t / ref.e_t
        assert 0.90 < recovery < 1.05, f"recovery {recovery:.4%}"
        assert r.e_t < 0.0  # (T) is stabilising

    def test_exact_mode_removes_t0_error(self):
        # triples_mode="exact" runs the canonical (T) on the converged
        # amplitudes (occupied rotated to canonical) — it removes the
        # DLPNO-(T0) semicanonical error, so its (T) is far closer to the
        # canonical (T) than the default local-(T0).
        mol, b, rhf, df, ref = _setup(H2O, "def2-svp")
        base = dict(localise="boys", tcut_pno=1e-7, compute_triples=True)
        t0 = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(triples_mode="local", **base)
        )
        ex = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(triples_mode="exact", **base)
        )
        assert abs(ex.e_t - ref.e_t) < abs(t0.e_t - ref.e_t)  # closer to canonical
        assert abs(ex.e_t - ref.e_t) < 5e-5  # ~0.03 kcal/mol of exact (T)


class TestTNOTruncation:
    """Occupation-number TNO truncation (tcut_tno) of the (T) virtual domain."""

    def test_tcut_tno_truncates_and_preserves_full_at_zero(self):
        mol, b, rhf, df, _ = _setup(H2O, "def2-svp")
        base = dict(localise="boys", tcut_pno=1e-7, compute_triples=True)
        full = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(tcut_tno=0.0, **base)
        ).e_t
        tiny = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(tcut_tno=1e-12, **base)
        ).e_t
        loose = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(tcut_tno=1e-4, **base)
        ).e_t
        # A negligible occupation cut keeps the full union span (≈ tcut_tno=0).
        assert abs(tiny - full) < 1e-7
        # A loose cut drops TNOs → a smaller (still stabilising) (T), proving
        # the truncation is live (the old SVD-on-union cut was inert here).
        assert full < 0.0 and loose < 0.0
        assert abs(loose) < abs(full)  # truncation removes correlation
        assert abs(loose) > 0.5 * abs(full)  # but degrades gracefully


class TestLocalTriplesScreening:
    """(T) triple screening by occupied locality (coupling_radius)."""

    def test_default_radius_compact_noop(self):
        # On a compact molecule every triple is within the 12-bohr default,
        # so screening is a bit-identical no-op vs full (coupling_radius=0).
        mol, b, rhf, df, _ = _setup(H2O, "def2-svp")
        base = dict(localise="boys", tcut_pno=1e-7, compute_triples=True)
        full = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(coupling_radius=0.0, **base)
        )
        deflt = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(coupling_radius=12.0, **base)
        )
        assert abs(deflt.e_t - full.e_t) < 1e-12

    @pytest.mark.slow  # full-coupling reference on 10 occupieds (O(N⁴) path)
    def test_separated_fragments_screen_losslessly(self):
        # Two water molecules 16 bohr apart (5 occupieds each → real
        # intra-molecular triples). A 10-bohr radius screens every
        # cross-fragment triple; each water's triples stay intra-local, so
        # the (T) is unchanged — the dropped cross-fragment triples vanish.
        atoms = [
            (8, [0, 0, 0]),
            (1, [0, 0.79 * A, -0.61 * A]),
            (1, [0, -0.79 * A, -0.61 * A]),
            (8, [0, 0, 16.0]),
            (1, [0, 0.79 * A, 16 - 0.61 * A]),
            (1, [0, -0.79 * A, 16 - 0.61 * A]),
        ]
        mol, b, rhf, df, _ = _setup(atoms, "def2-svp")
        # tcut_pairs=0 keeps every cross-fragment pair at CCSD level, so the
        # cross-fragment triples genuinely exist and the coupling_radius is
        # what screens them (isolating the (T) triple-screening behaviour).
        base = dict(
            localise="boys",
            tcut_pno=1e-7,
            compute_triples=True,
            max_nbf=300,
            tcut_pairs=0.0,
        )
        full = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(coupling_radius=0.0, **base)
        )
        scr = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(coupling_radius=10.0, **base)
        )
        assert abs(scr.e_t - full.e_t) < 1e-6  # lossless across the gap (sub-µHa)


class TestT1:
    """Rotated-occupied (T1) ratchet: full domains equal exact (T)."""

    def test_t1_equals_exact_full_domains_boys(self):
        """Boys-localised, full PNO domains: (T1) removes the (T0) error."""
        mol, b, rhf, df, ref = _setup(H2O, "def2-svp")
        base = dict(
            localise="boys",
            tcut_pno=0.0,
            tcut_mkn=0.0,
            coupling_radius=0.0,
            tcut_pairs=0.0,
            compute_triples=True,
        )
        r_ex = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(triples_mode="exact", **base)
        )
        r_t1 = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(triples_mode="t1", **base)
        )
        assert abs(r_t1.e_t - r_ex.e_t) < 1e-9, (
            f"(T1) error {abs(r_t1.e_t - r_ex.e_t):.2e}"
        )

    def test_t1_equals_exact_full_domains_sto3g(self):
        """H2O/STO-3G: same ratchet on a minimal basis."""
        mol, b, rhf, df, ref = _setup(H2O, "sto-3g")
        base = dict(
            localise="boys",
            tcut_pno=0.0,
            tcut_mkn=0.0,
            coupling_radius=0.0,
            tcut_pairs=0.0,
            compute_triples=True,
        )
        r_ex = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(triples_mode="exact", **base)
        )
        r_t1 = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(triples_mode="t1", **base)
        )
        assert abs(r_t1.e_t - r_ex.e_t) < 1e-9, (
            f"(T1) error {abs(r_t1.e_t - r_ex.e_t):.2e}"
        )

    def test_t1_idempotent_on_canonical_occupieds(self):
        """With canonical occupieds (localise='none'), (T1)==(T0)==exact.

        The Jacobi eigensystem is the identity (f_oo is already diagonal),
        so (T1) is bitwise identical to (T0) which is exact here.
        """
        mol, b, rhf, df, ref = _setup(H2O, "def2-svp")
        base = dict(
            localise="none",
            tcut_pno=0.0,
            tcut_mkn=0.0,
            coupling_radius=0.0,
            tcut_pairs=0.0,
            compute_triples=True,
        )
        r_local = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(triples_mode="local", **base)
        )
        r_t1 = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(triples_mode="t1", **base)
        )
        # Both are exact here (canonical occupieds, full domains)
        assert abs(r_t1.e_t - r_local.e_t) < 1e-9
        assert abs(r_t1.e_t - ref.e_t) < 1e-9

    def test_t1_positive_radius_noop_matches_full_result(self):
        """A nonzero radius that screens nothing is identical to full (T1)."""
        mol, b, rhf, df, _ = _setup(H2O, "sto-3g")
        base = dict(
            localise="boys",
            tcut_pno=0.0,
            tcut_mkn=0.0,
            tcut_pairs=0.0,
            compute_triples=True,
            triples_mode="t1",
        )
        full = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(coupling_radius=0.0, **base)
        )
        positive_noop = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(coupling_radius=1.0e6, **base)
        )

        assert abs(positive_noop.e_t - full.e_t) < 1e-12

    def test_t1_closes_t0_error_at_default_thresholds(self):
        """At default tcut_pno=1e-7, (T1) substantially closes the (T0) error.

        The (T0) error is ~0.1 kcal/mol; (T1) reduces it by ~90%.
        """
        mol, b, rhf, df, ref = _setup(H2O, "def2-svp")
        base = dict(localise="boys", compute_triples=True, coupling_radius=0.0)
        r_t0 = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(triples_mode="local", **base)
        )
        r_t1 = run_local_dlpno_ccsd(
            mol, b, rhf, df, LocalCCSDOptions(triples_mode="t1", **base)
        )
        err_t0 = abs(r_t0.e_t - ref.e_t)
        err_t1 = abs(r_t1.e_t - ref.e_t)
        # (T1) must be at least 50% closer to canonical than (T0)
        assert err_t1 < 0.5 * err_t0, (
            f"(T1) error {err_t1:.2e} not substantially smaller than (T0) {err_t0:.2e}"
        )
