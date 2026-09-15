"""Spatial closed-shell DF-CCSD — parity with the FCI-anchored kernel.

`dlpno._ccsd_cs.run_cs_ccsd` is the spatial closed-shell residual+solver
that the per-pair *local* DLPNO-CCSD residual will mirror (same
contractions, evaluated in PNO bases with projections). It must
reproduce the spin-orbital reference kernel (`_ccsd_ref`, CCSD ≡ FCI for
two electrons) to machine precision — at a quarter of the basis size.
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import BasisSet, RHFOptions, run_rhf
from vibeqc._vibeqc_core import Atom, Molecule
from vibeqc.density_fitting import DensityFitting
from vibeqc.dlpno._ccsd_cs import cs_ccsd_residual, run_cs_ccsd, _blocks
from vibeqc.dlpno._ccsd_ref import run_ref_ccsd

ANGSTROM_TO_BOHR = 1.8897259886
AUX = "def2-svp-rifit"
MICRO_HA = 1e-6
E_FCI_CORR_H2O_STO3G = -0.055587908  # full-space CASCI(7,10), from test_dlpno_ccsd

H2_ATOMS = [(1, [0.0, 0.0, 0.0]), (1, [0.0, 0.0, 1.4])]
H2O_ATOMS = [
    (8, [0.0, 0.0, 0.0]),
    (1, [0.0, 0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
    (1, [0.0, -0.793353 * ANGSTROM_TO_BOHR, -0.613510 * ANGSTROM_TO_BOHR]),
]


def _setup(atoms, basis_name):
    mol = Molecule([Atom(z, p) for z, p in atoms], charge=0, multiplicity=1)
    basis = BasisSet(mol, basis_name)
    opts = RHFOptions()
    opts.max_iter = 100
    rhf = run_rhf(mol, basis, opts)
    assert rhf.converged
    df = DensityFitting(basis, BasisSet(mol, AUX), aux_basis_name=AUX)
    C = np.asarray(rhf.mo_coeffs)
    F = np.asarray(rhf.fock)
    n_occ = mol.n_electrons() // 2
    f_mo = C.T @ F @ C
    Co, Cv = C[:, :n_occ], C[:, n_occ:]
    return dict(
        rhf=rhf, df=df, f_mo=f_mo, n_occ=n_occ,
        B_full=np.asarray(df.mo_transform(C, C)),
        B_ov=np.asarray(df.mo_transform(Co, Cv)),
        B_vv=np.asarray(df.mo_transform(Cv, Cv)),
        B_oo=np.asarray(df.mo_transform(Co, Co)),
    )


def _spatial(d):
    return run_cs_ccsd(
        d["f_mo"], d["B_ov"], d["B_vv"], d["B_oo"], d["n_occ"],
        e_hf=d["rhf"].energy,
    )


def _spin_orbital(d):
    return run_ref_ccsd(
        d["f_mo"], d["B_full"], n_occ=d["n_occ"], e_hf=d["rhf"].energy
    )


class TestParity:
    @pytest.mark.parametrize(
        "atoms,basis",
        [(H2_ATOMS, "sto-3g"), (H2O_ATOMS, "sto-3g"), (H2O_ATOMS, "def2-svp")],
    )
    def test_matches_spin_orbital(self, atoms, basis):
        d = _setup(atoms, basis)
        cs = _spatial(d)
        so = _spin_orbital(d)
        assert cs.converged
        assert abs(cs.e_corr - so.e_corr) < 1.0 * MICRO_HA

    def test_h2o_sits_above_fci(self):
        """The same anchor gate that caught the broken C++ CCSD."""
        d = _setup(H2O_ATOMS, "sto-3g")
        cs = _spatial(d)
        assert cs.e_corr > E_FCI_CORR_H2O_STO3G
        assert cs.e_corr < E_FCI_CORR_H2O_STO3G * 0.985


class TestSpinOrbitalParity:
    """The spatial residual equals the spin-orbital αβ block element-wise.

    This is what makes the DLPNO-CCSD pilot's swap (so_residuals →
    cs_ccsd_residual) exact, not merely same-fixed-point.
    """

    def test_residual_matches_spin_orbital_block(self):
        from vibeqc.dlpno._ccsd_ref import _spin_orbital_eri, so_residuals

        d = _setup(H2O_ATOMS, "sto-3g")
        no = d["n_occ"]
        C = np.asarray(d["rhf"].mo_coeffs)
        F = np.asarray(d["rhf"].fock)
        f_mo = d["f_mo"]
        nv = C.shape[1] - no
        n_mo = no + nv
        V = _blocks(d["B_ov"], d["B_vv"], d["B_oo"])

        # Non-trivial closed-shell amplitudes (not a fixed point).
        rng_t1 = np.cos(np.arange(no * nv)).reshape(no, nv) * 0.01
        raw = np.sin(np.arange(no * no * nv * nv)).reshape(no, no, nv, nv)
        t2 = 0.01 * (raw + raw.transpose(1, 0, 3, 2))  # t_ij^ab = t_ji^ba

        f_oo, f_vv, f_ov = f_mo[:no, :no], f_mo[no:, no:], f_mo[:no, no:]
        R1, R2 = cs_ccsd_residual(rng_t1, t2, f_oo, f_vv, f_ov, V)

        # Spin-orbital residual on the same amplitudes.
        eri = _spin_orbital_eri(np.asarray(d["B_full"]), n_mo)
        f_so = np.zeros((2 * n_mo, 2 * n_mo))
        f_so[0::2, 0::2] = f_mo
        f_so[1::2, 1::2] = f_mo
        no_so, nv_so = 2 * no, 2 * nv
        o, v = slice(0, no_so), slice(no_so, no_so + nv_so)
        fock_od = f_so.copy()
        np.fill_diagonal(fock_od, 0.0)
        eps = np.diag(f_so)
        D1 = eps[o, None] - eps[None, v]
        D2 = (eps[o][:, None, None, None] + eps[o][None, :, None, None]
              - eps[v][None, None, :, None] - eps[v][None, None, None, :])
        t1_so = np.zeros((no_so, nv_so))
        t1_so[0::2, 0::2] = rng_t1
        t1_so[1::2, 1::2] = rng_t1
        t2_so = np.zeros((no_so, no_so, nv_so, nv_so))
        for si in (0, 1):
            for sj in (0, 1):
                t2_so[si::2, sj::2, si::2, sj::2] += t2
                t2_so[si::2, sj::2, sj::2, si::2] -= t2.transpose(0, 1, 3, 2)
        r1_so, r2_so = so_residuals(f_so, fock_od, eri, t1_so, t2_so, o, v, D1, D2)

        assert np.allclose(R1, r1_so[0::2, 0::2], atol=1e-10)
        assert np.allclose(R2, r2_so[0::2, 1::2, 0::2, 1::2], atol=1e-10)


class TestTriplesParity:
    """Spatial closed-shell (T) == the spin-orbital reference (T).

    `cs_triples_correction` is the spatial transcription of the validated
    C++ (T) kernel; it must reproduce `_ccsd_ref.so_triples_correction` on
    the same amplitudes to machine precision, at a quarter the dimensions
    and an eighth the triples.
    """

    @pytest.mark.parametrize("atoms,basis", [(H2O_ATOMS, "sto-3g"), (H2O_ATOMS, "def2-svp")])
    def test_matches_spin_orbital_triples(self, atoms, basis):
        from vibeqc.dlpno._ccsd_cs import cs_triples_correction
        from vibeqc.dlpno._ccsd_ref import _spin_orbital_eri, so_triples_correction

        d = _setup(atoms, basis)
        cs = _spatial(d)
        no = d["n_occ"]
        f_mo = d["f_mo"]
        nv = np.asarray(d["rhf"].mo_coeffs).shape[1] - no
        n_mo = no + nv
        V = _blocks(d["B_ov"], d["B_vv"], d["B_oo"])
        eps_o, eps_v = np.diag(f_mo)[:no], np.diag(f_mo)[no:]

        e_spatial = cs_triples_correction(
            cs.t1, cs.t2, V["ovvv"], V["ooov"], V["ovov"], eps_o, eps_v
        )

        # Spin-orbital reference (T) on the same amplitudes.
        eri = _spin_orbital_eri(np.asarray(d["B_full"]), n_mo)
        eps_so = np.zeros(2 * n_mo)
        eps_so[0::2] = np.diag(f_mo)
        eps_so[1::2] = np.diag(f_mo)
        t1_so = np.zeros((2 * no, 2 * nv))
        t1_so[0::2, 0::2] = cs.t1
        t1_so[1::2, 1::2] = cs.t1
        t2_so = np.zeros((2 * no, 2 * no, 2 * nv, 2 * nv))
        for si in (0, 1):
            for sj in (0, 1):
                t2_so[si::2, sj::2, si::2, sj::2] += cs.t2
                t2_so[si::2, sj::2, sj::2, si::2] -= cs.t2.transpose(0, 1, 3, 2)
        e_so = so_triples_correction(
            eps_so, eri, t1_so, t2_so, slice(0, 2 * no), slice(2 * no, 2 * n_mo)
        )
        assert e_spatial < 0.0
        assert abs(e_spatial - e_so) < 1e-12, (e_spatial, e_so)


class TestSelfConsistency:
    def test_residual_vanishes_at_convergence(self):
        d = _setup(H2O_ATOMS, "sto-3g")
        cs = _spatial(d)
        V = _blocks(d["B_ov"], d["B_vv"], d["B_oo"])
        no = d["n_occ"]
        f_oo = d["f_mo"][:no, :no]
        f_vv = d["f_mo"][no:, no:]
        f_ov = d["f_mo"][:no, no:]
        R1, R2 = cs_ccsd_residual(cs.t1, cs.t2, f_oo, f_vv, f_ov, V)
        assert np.max(np.abs(R1)) < 1e-7
        assert np.max(np.abs(R2)) < 1e-7
