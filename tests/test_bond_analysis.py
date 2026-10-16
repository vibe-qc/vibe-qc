"""Tests for bond analysis methods: Wiberg, EDA, entanglement, NBO.

Covers both molecular and periodic pathways. COHP/COOP is NOT here --
the canonical periodic COHP/COOP implementation is
:mod:`vibeqc.coop_cohp` (tests in ``test_coop_cohp.py``).
"""

from __future__ import annotations

import functools
from types import SimpleNamespace

import numpy as np
import pytest
import vibeqc as vq

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _h2o() -> vq.Molecule:
    return vq.Molecule(
        [
            vq.Atom(8, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 1.43, -0.98]),
            vq.Atom(1, [0.0, -1.43, -0.98]),
        ]
    )


def _co() -> vq.Molecule:
    return vq.Molecule(
        [
            vq.Atom(6, [0.0, 0.0, 0.0]),
            vq.Atom(8, [0.0, 0.0, 2.132]),
        ]
    )


def _h2() -> vq.Molecule:
    return vq.Molecule(
        [
            vq.Atom(1, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 0.0, 1.4]),
        ]
    )


# ---------------------------------------------------------------------------
# Wiberg bond orders
# ---------------------------------------------------------------------------


class TestWibergBondOrders:
    """Wiberg bond index from density matrix."""

    def test_wiberg_water_sto3g(self):
        """Wiberg bond orders for H2O / STO-3G RHF."""
        from vibeqc.bond_analysis import wiberg_bond_orders

        mol = _h2o()
        basis = vq.BasisSet(mol, "sto-3g")
        result = vq.run_rhf(mol, basis)
        bo = wiberg_bond_orders(result, basis, mol)

        assert bo.shape == (3, 3)
        assert bo[0, 1] > 0.3
        assert bo[0, 2] > 0.3
        assert bo[1, 2] < 0.15
        assert np.all(np.diag(bo) >= 0)

    def test_wiberg_symmetric(self):
        """Wiberg matrix must be symmetric."""
        from vibeqc.bond_analysis import wiberg_bond_orders

        mol = _h2()
        basis = vq.BasisSet(mol, "sto-3g")
        result = vq.run_rhf(mol, basis)
        bo = wiberg_bond_orders(result, basis, mol)

        assert bo.shape == (2, 2)
        assert abs(bo[0, 1] - bo[1, 0]) < 1e-14

    def test_wiberg_uhf(self):
        """Wiberg for open-shell (UHF) system."""
        from vibeqc.bond_analysis import wiberg_bond_orders

        mol = vq.Molecule(
            [vq.Atom(8, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.832])],
            charge=0,
            multiplicity=2,
        )
        basis = vq.BasisSet(mol, "sto-3g")
        result = vq.run_uhf(mol, basis)
        bo = wiberg_bond_orders(result, basis, mol)

        assert bo.shape == (2, 2)
        assert bo[0, 1] > 0.2


# ---------------------------------------------------------------------------
# Bond order summary
# ---------------------------------------------------------------------------


class TestBondOrderSummary:
    def test_summary_all_metrics(self):
        """bond_order_summary computes Mayer + Wiberg without error."""
        from vibeqc.bond_analysis import bond_order_summary

        mol = _co()
        basis = vq.BasisSet(mol, "sto-3g")
        result = vq.run_rhf(mol, basis)

        summary = bond_order_summary(result, basis, mol)
        assert summary.mayer is not None
        assert summary.wiberg is not None
        assert summary.mayer.shape == (2, 2)
        assert summary.wiberg.shape == (2, 2)

    def test_top_bonds(self):
        """Top bonds listing returns ordered entries."""
        from vibeqc.bond_analysis import bond_order_summary

        mol = _co()
        basis = vq.BasisSet(mol, "sto-3g")
        result = vq.run_rhf(mol, basis)

        summary = bond_order_summary(result, basis, mol)
        top = summary.top_bonds(metric="mayer", molecule=mol, threshold=0.05)
        assert len(top) >= 1
        assert top[0][0] == 0
        assert top[0][1] == 1
        assert top[0][2] > 0.5


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

_ANG = 1.8897259886
_KCAL = 627.5094740631

# Water dimer, donor first (examples/h2o_dimer.xyz), R(O..O) = 2.95 A, in Angstrom.
_DIMER_XYZ = [
    (8, [0.000, 0.000, 0.000]),
    (1, [0.000, 0.000, 0.960]),
    (1, [0.000, 0.929, -0.240]),
    (8, [0.000, 0.000, 2.950]),
    (1, [0.000, 0.759, 3.538]),
    (1, [0.000, -0.759, 3.538]),
]
_DONOR, _ACCEPTOR = [0, 1, 2], [3, 4, 5]


@functools.lru_cache(maxsize=None)
def _water_dimer_eda(shift: float = 0.0):
    """Three RHF/6-31G* runs in the dimer basis (ghost atoms for the fragments).

    ``shift`` moves the acceptor along z, in Angstrom. Returns a namespace with
    everything ``eda_lmo`` takes.
    """
    coords = [
        (z, [p[0], p[1], p[2] + (shift if i in _ACCEPTOR else 0.0)])
        for i, (z, p) in enumerate(_DIMER_XYZ)
    ]

    def molecule(real):
        return vq.Molecule(
            [
                vq.Atom(z if i in real else 0, [c * _ANG for c in p])
                for i, (z, p) in enumerate(coords)
            ],
            0,
            1,
        )

    dimer = molecule(_DONOR + _ACCEPTOR)
    basis = vq.BasisSet(dimer, "6-31g*")
    shells = basis.shells()

    def scf(mol):
        opts = vq.RHFOptions()
        if any(a.Z == 0 for a in mol.atoms):
            opts.initial_guess = vq.InitialGuess.HCORE  # SAD has no density for Z = 0
            b = vq.BasisSet(mol, shells, name="6-31g*")
        else:
            b = basis
        res = vq.run_rhf(mol, b, opts)
        assert res.converged
        return res

    ab, a, b = scf(dimer), scf(molecule(_DONOR)), scf(molecule(_ACCEPTOR))
    overlap = np.asarray(vq.compute_overlap(basis))
    return SimpleNamespace(
        basis=basis,
        dimer=dimer,
        ab=ab,
        a=a,
        b=b,
        overlap=overlap,
        args=(
            ab.energy, a.energy, b.energy, None, None, None,
            np.asarray(ab.density), np.asarray(a.density), np.asarray(b.density),
            overlap,
        ),  # fmt: skip
        kw=dict(
            basis=basis, molecule=dimer, frag1_atoms=_DONOR, frag2_atoms=_ACCEPTOR
        ),
    )


class TestEDA:
    """Energy decomposition analysis of the water-dimer hydrogen bond."""

    def test_water_dimer_kitaura_morokuma_terms(self):
        """ES, EX and PL+CT at RHF/6-31G*, checked against an independent run.

        The reference numbers come from a separate implementation of the same
        partition on PySCF integrals and SCF (see ``test_matches_pyscf``); the
        placeholder this replaces returned ES = +75.96 Ha, the fragment
        kinetic energy.
        """
        from vibeqc.eda import eda_morokuma

        d = _water_dimer_eda()
        km = eda_morokuma(*d.args, **d.kw)
        assert km.method == "morokuma"
        assert km.e_elstat * _KCAL == pytest.approx(-7.8802, abs=2e-3)
        assert km.e_exch * _KCAL == pytest.approx(5.3675, abs=2e-3)
        assert km.e_pol * _KCAL == pytest.approx(-1.6200, abs=2e-3)
        assert km.e_int * _KCAL == pytest.approx(-4.1327, abs=2e-3)
        assert km.e_rep == 0.0
        comp = km.extra["morokuma_components"]
        assert comp["ES"] == km.e_elstat
        assert comp["EX"] == km.e_exch
        assert comp["PL+CT+MIX"] == km.e_pol
        assert "CT+MIX" not in comp

    def test_water_dimer_lmo_terms(self):
        """LMO-EDA splits the Pauli term into exchange and repulsion."""
        from vibeqc.eda import eda_lmo

        d = _water_dimer_eda()
        lmo = eda_lmo(*d.args, **d.kw)
        assert lmo.method == "lmo"
        assert lmo.e_elstat * _KCAL == pytest.approx(-7.8802, abs=2e-3)
        assert lmo.e_exch * _KCAL == pytest.approx(-7.1308, abs=2e-3)
        assert lmo.e_rep * _KCAL == pytest.approx(12.4983, abs=2e-3)
        assert lmo.e_pol * _KCAL == pytest.approx(-1.6200, abs=2e-3)
        assert lmo.e_disp == 0.0
        # The energy check of the inputs is part of the result.
        assert max(abs(v) for v in lmo.extra["consistency"].values()) < 1e-9

    def test_components_sum_to_the_interaction_energy(self):
        from vibeqc.eda import eda_lmo, eda_morokuma

        d = _water_dimer_eda()
        e_int = d.ab.energy - d.a.energy - d.b.energy
        lmo = eda_lmo(*d.args, **d.kw)
        km = eda_morokuma(*d.args, **d.kw)
        for res in (lmo, km):
            assert res.e_int == pytest.approx(e_int, abs=1e-14)
            total = res.e_elstat + res.e_exch + res.e_rep + res.e_pol + res.e_disp
            assert total == pytest.approx(e_int, abs=1e-12)
        # The two schemes partition the same states.
        assert km.e_elstat == pytest.approx(lmo.e_elstat, abs=1e-14)
        assert km.e_exch == pytest.approx(lmo.e_exch + lmo.e_rep, abs=1e-12)
        assert km.e_pol == pytest.approx(lmo.e_pol, abs=1e-14)

    def test_matches_dense_integral_construction(self):
        """The library terms equal the textbook construction on dense integrals.

        Different code path: the four-index tensor, ``einsum`` contractions and a
        Loewdin transformation of the occupied MOs, where the library uses the
        direct J/K builder and derives the orbitals from the densities.
        """
        from vibeqc.eda import eda_lmo

        d = _water_dimer_eda()
        basis, S = d.basis, d.overlap
        n = basis.nbasis
        eri = np.asarray(vq.compute_eri(basis)).reshape(n, n, n, n)
        J = lambda P: np.einsum("pqrs,rs->pq", eri, P, optimize=True)  # noqa: E731
        K = lambda P: np.einsum("prqs,rs->pq", eri, P, optimize=True)  # noqa: E731
        atoms = list(d.dimer.atoms)

        def charges(idx):
            return [list(atoms[i].xyz) for i in idx], [float(atoms[i].Z) for i in idx]

        T = np.asarray(vq.compute_kinetic(basis))
        V_A = np.asarray(vq.compute_nuclear_with_charges(basis, *charges(_DONOR)))
        V_B = np.asarray(vq.compute_nuclear_with_charges(basis, *charges(_ACCEPTOR)))

        def e_nn(idx_a, idx_b=None):
            ra = [np.asarray(atoms[i].xyz) for i in idx_a]
            za = [atoms[i].Z for i in idx_a]
            if idx_b is None:
                return sum(
                    za[i] * za[j] / np.linalg.norm(ra[i] - ra[j])
                    for i in range(len(za))
                    for j in range(i + 1, len(za))
                )
            rb = [np.asarray(atoms[i].xyz) for i in idx_b]
            zb = [atoms[i].Z for i in idx_b]
            return sum(
                za[i] * zb[j] / np.linalg.norm(ra[i] - rb[j])
                for i in range(len(za))
                for j in range(len(zb))
            )

        e_nn_ab = e_nn(_DONOR, _ACCEPTOR)
        P_a, P_b = np.asarray(d.a.density), np.asarray(d.b.density)
        es = (
            np.trace(P_a @ V_B) + np.trace(P_b @ V_A) + np.trace(P_a @ J(P_b)) + e_nn_ab
        )
        exch = -0.5 * np.trace(P_a @ K(P_b))

        nocc = 5
        C = np.hstack(
            [
                np.asarray(d.a.mo_coeffs)[:, :nocc],
                np.asarray(d.b.mo_coeffs)[:, :nocc],
            ]
        )
        w, U = np.linalg.eigh(C.T @ S @ C)
        C_orth = C @ U @ np.diag(w**-0.5) @ U.T
        P_fr = 2.0 * C_orth @ C_orth.T
        h = T + V_A + V_B
        e_frozen = (
            np.trace(P_fr @ h)
            + 0.5 * np.trace(P_fr @ (J(P_fr) - 0.5 * K(P_fr)))
            + e_nn(_DONOR)
            + e_nn(_ACCEPTOR)
            + e_nn_ab
        )
        e_pauli = e_frozen - d.a.energy - d.b.energy - es

        lmo = eda_lmo(*d.args, **d.kw)
        assert lmo.e_elstat == pytest.approx(es, abs=1e-8)
        assert lmo.e_exch == pytest.approx(exch, abs=1e-8)
        assert lmo.e_exch + lmo.e_rep == pytest.approx(e_pauli, abs=1e-8)
        assert lmo.extra["e_frozen"] == pytest.approx(e_frozen, abs=1e-8)
        assert lmo.e_pol == pytest.approx(d.ab.energy - e_frozen, abs=1e-8)
        # The promolecule energy is the HF energy expression of P_A + P_B.
        P_pro = P_a + P_b
        e_pro = (
            np.trace(P_pro @ h)
            + 0.5 * np.trace(P_pro @ (J(P_pro) - 0.5 * K(P_pro)))
            + e_nn(_DONOR)
            + e_nn(_ACCEPTOR)
            + e_nn_ab
        )
        assert lmo.extra["e_promolecule"] == pytest.approx(e_pro, abs=1e-8)

    def test_matches_pyscf(self):
        """Same partition on PySCF's integrals and SCF (independent program)."""
        scipy_linalg = pytest.importorskip("scipy.linalg")
        gto = pytest.importorskip("pyscf.gto")
        pyscf_scf = pytest.importorskip("pyscf.scf")
        from vibeqc.eda import eda_lmo, eda_morokuma

        def mol(real):
            atoms = [
                (("" if i in real else "ghost-") + vq.Atom(z, [0, 0, 0]).symbol, p)
                for i, (z, p) in enumerate(_DIMER_XYZ)
            ]
            return gto.M(atom=atoms, basis="6-31g*", unit="Angstrom", verbose=0)

        def rhf(m):
            mf = pyscf_scf.RHF(m)
            mf.conv_tol = 1e-12
            mf.kernel()
            assert mf.converged
            return mf

        m_ab, m_a, m_b = mol(_DONOR + _ACCEPTOR), mol(_DONOR), mol(_ACCEPTOR)
        mf_ab, mf_a, mf_b = rhf(m_ab), rhf(m_a), rhf(m_b)
        P_a, P_b = mf_a.make_rdm1(), mf_b.make_rdm1()
        S = m_ab.intor("int1e_ovlp")
        e_nn_ab = m_ab.energy_nuc() - m_a.energy_nuc() - m_b.energy_nuc()
        es = (
            np.sum(P_a * m_b.intor("int1e_nuc"))  # ghost atoms carry no charge
            + np.sum(P_b * m_a.intor("int1e_nuc"))
            + np.sum(P_a * mf_ab.get_j(m_ab, P_b))
            + e_nn_ab
        )
        exch = -0.5 * np.sum(P_a * mf_ab.get_k(m_ab, P_b))
        C = np.hstack(
            [
                mf_a.mo_coeff[:, mf_a.mo_occ > 0],
                mf_b.mo_coeff[:, mf_b.mo_occ > 0],
            ]
        )
        C = C @ scipy_linalg.fractional_matrix_power(C.T @ S @ C, -0.5).real
        e_frozen = mf_ab.energy_tot(dm=2.0 * C @ C.T)
        pauli = e_frozen - mf_a.e_tot - mf_b.e_tot - es
        pol = mf_ab.e_tot - e_frozen

        d = _water_dimer_eda()
        lmo = eda_lmo(*d.args, **d.kw)
        km = eda_morokuma(*d.args, **d.kw)
        assert d.ab.energy == pytest.approx(mf_ab.e_tot, abs=1e-7)
        assert lmo.e_elstat == pytest.approx(es, abs=5e-8)
        assert lmo.e_exch == pytest.approx(exch, abs=5e-8)
        assert km.e_exch == pytest.approx(pauli, abs=5e-8)
        assert lmo.e_rep == pytest.approx(pauli - exch, abs=5e-8)
        assert km.e_pol == pytest.approx(pol, abs=5e-8)

    def test_pauli_term_vanishes_and_electrostatics_becomes_classical(self):
        """Limits that need no reference values.

        Pauli repulsion decays exponentially with separation, and at large
        separation the electrostatic term approaches the classical
        dipole-dipole energy of the two isolated monomers, with a correction
        that falls off as 1/R.
        """
        from vibeqc.eda import eda_lmo

        def ratio_to_dipole_dipole(shift):
            d = _water_dimer_eda(shift)
            lmo = eda_lmo(*d.args, **d.kw)
            dip = vq.compute_dipole(d.basis)

            def moment(res, real):
                P = np.asarray(res.density)
                electrons = np.array([np.sum(P * np.asarray(m)) for m in (dip.x, dip.y, dip.z)])
                nuclei = sum(
                    np.asarray(d.dimer.atoms[i].xyz) * d.dimer.atoms[i].Z for i in real
                )
                return nuclei - electrons

            mu_a, mu_b = moment(d.a, _DONOR), moment(d.b, _ACCEPTOR)
            r_vec = np.asarray(d.dimer.atoms[3].xyz) - np.asarray(d.dimer.atoms[0].xyz)
            r = np.linalg.norm(r_vec)
            n = r_vec / r
            e_dd = (mu_a @ mu_b - 3.0 * (mu_a @ n) * (mu_b @ n)) / r**3
            return lmo, lmo.e_elstat / e_dd

        near, _ = ratio_to_dipole_dipole(0.0)
        mid, _ = ratio_to_dipole_dipole(3.0)
        far, _ = ratio_to_dipole_dipole(6.0)
        assert abs(near.e_exch + near.e_rep) > 100 * abs(mid.e_exch + mid.e_rep)
        assert abs(mid.e_exch + mid.e_rep) > 1e3 * abs(far.e_exch + far.e_rep)
        assert abs(far.e_exch + far.e_rep) < 1e-7  # Ha

        _, ratio_20 = ratio_to_dipole_dipole(17.0)
        _, ratio_40 = ratio_to_dipole_dipole(37.0)
        assert abs(ratio_20 - 1.0) < 0.12
        assert abs(ratio_40 - 1.0) < 0.6 * abs(ratio_20 - 1.0)  # ~1/R convergence

    def test_dispersion_correction_is_reported_separately(self):
        """Additive dispersion is removed before the partition and reported."""
        from vibeqc.eda import eda_lmo

        d = _water_dimer_eda()
        base = eda_lmo(*d.args, **d.kw)
        disp_ab, disp_a, disp_b = -0.0031, -0.0012, -0.0013
        args = list(d.args)
        args[0] += disp_ab
        args[1] += disp_a
        args[2] += disp_b
        res = eda_lmo(
            *args, **d.kw, e_disp=disp_ab, e_disp_frag1=disp_a, e_disp_frag2=disp_b
        )
        assert res.e_disp == pytest.approx(disp_ab - disp_a - disp_b, abs=1e-15)
        for name in ("e_elstat", "e_exch", "e_rep", "e_pol"):
            assert getattr(res, name) == pytest.approx(getattr(base, name), abs=1e-10)
        total = res.e_elstat + res.e_exch + res.e_rep + res.e_pol + res.e_disp
        assert total == pytest.approx(res.e_int, abs=1e-12)
        assert "Dispersion" in res.summary()

    def test_summary(self):
        from vibeqc.eda import eda_lmo, eda_morokuma

        d = _water_dimer_eda()
        s = eda_lmo(*d.args, **d.kw).summary()
        for label in ("Electrostatic", "Exchange", "Repulsion", "Polarisation"):
            assert label in s
        assert "-7.88 kcal/mol" in s
        sm = eda_morokuma(*d.args, **d.kw).summary()
        assert "MOROKUMA" in sm and "EX" in sm and "Polarisation + CT" in sm
        assert "Repulsion" not in sm

    def test_refuses_the_old_call_without_integrals(self):
        """Fock matrices and densities alone cannot give the terms."""
        from vibeqc.eda import eda_lmo

        d = _water_dimer_eda()
        with pytest.raises(TypeError, match="basis"):
            eda_lmo(*d.args)

    def test_refuses_non_idempotent_fragment_density(self):
        """A projected fragment density is not a closed-shell SCF density."""
        from vibeqc.eda import eda_lmo, fragment_density_matrix

        d = _water_dimer_eda()
        n = d.basis.nbasis
        ao_atom = [0] * (n // 2) + [1] * (n - n // 2)
        projected = fragment_density_matrix(
            np.asarray(d.ab.mo_coeffs), d.overlap, 10, [0], ao_to_atom=ao_atom
        )
        args = list(d.args)
        args[7] = projected
        with pytest.raises(ValueError, match="closed-shell density"):
            eda_lmo(*args, **d.kw, consistency_tol=None)

    def test_refuses_energies_that_are_not_hartree_fock(self):
        """A Kohn-Sham (or otherwise foreign) energy fails the consistency check."""
        from vibeqc.eda import eda_lmo, eda_morokuma

        d = _water_dimer_eda()
        args = list(d.args)
        args[1] += 2.0e-3  # not the HF energy of density_frag1
        for func in (eda_lmo, eda_morokuma):
            with pytest.raises(ValueError, match="not the closed-shell Hartree-Fock"):
                func(*args, **d.kw)
        # The check is a switch, not a silent default: with it off the run goes through.
        res = eda_lmo(*args, **d.kw, consistency_tol=None)
        assert res.e_int == pytest.approx(args[0] - args[1] - args[2], abs=1e-14)

    def test_refuses_bad_fragment_definitions(self):
        from vibeqc.eda import eda_lmo

        d = _water_dimer_eda()
        base = dict(d.kw)
        for frag1, frag2, msg in (
            ([0, 1, 2], [2, 3, 4, 5], "overlap"),
            ([0, 1, 2], [3, 4], "cover every atom"),
            ([0, 1, 2], [3, 4, 9], "outside"),
            ([], [0, 1, 2, 3, 4, 5], "empty"),
            ([0, 0, 1, 2], [3, 4, 5], "twice"),
        ):
            with pytest.raises(ValueError, match=msg):
                eda_lmo(*d.args, **{**base, "frag1_atoms": frag1, "frag2_atoms": frag2})

    def test_refuses_monomer_basis_densities(self):
        """Fragments computed in their own basis do not fit the dimer basis."""
        from vibeqc.eda import eda_lmo

        d = _water_dimer_eda()
        atoms = [vq.Atom(z, [c * _ANG for c in p]) for z, p in _DIMER_XYZ[:3]]
        monomer = vq.Molecule(atoms, 0, 1)
        res = vq.run_rhf(monomer, vq.BasisSet(monomer, "6-31g*"))
        args = list(d.args)
        args[7] = np.asarray(res.density)
        with pytest.raises(ValueError, match="dimer basis"):
            eda_lmo(*args, **d.kw)


# ---------------------------------------------------------------------------
# Entanglement
# ---------------------------------------------------------------------------


class TestEntanglement:
    """Orbital entanglement measures."""

    def test_single_orbital_entropy_closed_shell(self):
        """Single-orbital entropy for fully occupied / empty orbitals."""
        from vibeqc.entanglement import single_orbital_entropy

        rho_occ = np.diag([0.0, 0.0, 0.0, 1.0])
        s = single_orbital_entropy(np.array([rho_occ]))
        assert s[0] == 0.0

        rho_empty = np.diag([1.0, 0.0, 0.0, 0.0])
        s = single_orbital_entropy(np.array([rho_empty]))
        assert s[0] == 0.0

        # Pure one-/two-orbital RDMs must not acquire entropy from a
        # probability floor, especially before bond-order normalisation.
        from vibeqc.entanglement import entanglement_bond_order, mutual_information

        rho = np.array([rho_empty, rho_occ])
        rho2 = np.zeros((2, 2, 16, 16))
        rho2[0, 1] = np.kron(rho_empty, rho_occ)
        for two_orbital_densities in (None, rho2):
            mi = mutual_information(rho, two_orbital_densities)
            np.testing.assert_array_equal(mi, 0.0)
            np.testing.assert_array_equal(entanglement_bond_order(mi), 0.0)

    def test_entanglement_from_density(self):
        """Entanglement from single-determinant density."""
        from vibeqc.entanglement import entanglement_from_density

        n_ao = 4
        P = np.zeros((n_ao, n_ao), dtype=np.float64)
        P[0, 0] = 2.0
        P[1, 1] = 2.0
        S = np.eye(n_ao)

        result = entanglement_from_density(P, S)
        assert result.n_orbitals == n_ao
        np.testing.assert_array_equal(result.single_orbital_entropies, 0.0)
        np.testing.assert_array_equal(result.mutual_information, 0.0)
        np.testing.assert_array_equal(result.entanglement_bond_orders, 0.0)
        assert result.total_information == 0.0

    def test_entanglement_paired_orbital_rdms(self):
        """An entangled empty/pair superposition has known RDM entropies."""
        from vibeqc.entanglement import (
            entanglement_bond_order,
            mutual_information,
            single_orbital_entropy,
        )

        rho = np.array([np.diag([0.5, 0.0, 0.0, 0.5])] * 2)
        psi = np.zeros(16)
        psi[[3, 12]] = 1.0 / np.sqrt(2.0)
        rho2 = np.zeros((2, 2, 16, 16))
        rho2[0, 1] = np.outer(psi, psi)
        np.testing.assert_allclose(
            single_orbital_entropy(rho), np.log(2.0), atol=1e-14
        )
        mi = mutual_information(rho, rho2)
        np.testing.assert_allclose(mi, 2.0 * np.log(2.0), atol=1e-14)
        np.testing.assert_allclose(entanglement_bond_order(mi), 1.0, atol=1e-14)

    def test_entanglement_rhf_water(self):
        """A nonorthogonal RHF AO density has only 0/2 occupations."""
        from vibeqc.entanglement import entanglement_from_density

        mol = _h2o()
        basis = vq.BasisSet(mol, "sto-3g")
        rhf = vq.run_rhf(mol, basis)
        assert rhf.converged
        S = np.asarray(vq.compute_overlap(basis))
        P = np.asarray(rhf.density)
        # Independent occupation oracle: PS is similar to the orthogonal
        # density, without using the production square-root transform.
        occ = np.sort(np.linalg.eigvals(P @ S).real)
        np.testing.assert_allclose(occ, [0, 0, 2, 2, 2, 2, 2], atol=1e-12)
        result = entanglement_from_density(P, S)
        np.testing.assert_array_equal(result.single_orbital_entropies, 0.0)
        np.testing.assert_array_equal(result.mutual_information, 0.0)
        np.testing.assert_array_equal(result.entanglement_bond_orders, 0.0)
        assert result.total_information == 0.0

    @pytest.mark.parametrize("n_active", [None, 2])
    def test_entanglement_fractional_occupations(self, n_active):
        """All density-based measures use the same per-spin occupations."""
        from vibeqc.entanglement import (
            entanglement_bond_order,
            entanglement_from_density,
            single_orbital_entropy,
        )

        # S = A.T A and C = A^-1 give C.T S C = I, so these are known
        # natural occupations in a nonorthogonal basis without an eigh oracle.
        A = np.array([[1.2, 0.3, -0.1], [0.0, 0.9, 0.2], [0.0, 0.0, 1.1]])
        C = np.linalg.inv(A)
        occ = np.array([0.4, 0.8, 1.6])
        P = (C * occ) @ C.T
        result = entanglement_from_density(P, A.T @ A, n_active=n_active)
        n = occ[:n_active] / 2.0
        expected_s = -n * np.log(n) - (1.0 - n) * np.log1p(-n)
        expected_mi = 4.0 * np.outer(n * (1.0 - n), n * (1.0 - n))
        np.fill_diagonal(expected_mi, 0.0)
        np.testing.assert_allclose(
            result.single_orbital_entropies, expected_s, atol=1e-14
        )
        np.testing.assert_allclose(result.mutual_information, expected_mi, atol=1e-14)
        np.testing.assert_allclose(
            result.entanglement_bond_orders,
            entanglement_bond_order(expected_mi),
            atol=1e-14,
        )
        assert result.total_information == pytest.approx(expected_s.sum(), abs=1e-14)
        # A paired empty/doubly-occupied orbital has this same binary entropy.
        rho = np.array([np.diag([1.0 - p, 0.0, 0.0, p]) for p in n])
        np.testing.assert_allclose(single_orbital_entropy(rho), expected_s, atol=1e-14)

    def test_entanglement_casci_stretched_h2(self):
        """A correlated CAS(2,2) density retains positive entropy and MI."""
        from vibeqc.entanglement import entanglement_from_density
        from vibeqc.solvers import build_hamiltonian_mo, casci, get_hf_orbital_provider
        from vibeqc.solvers._rdm import make_rdm1

        mol = vq.Molecule(
            [
                vq.Atom(1, [0.0, 0.0, 0.0]),
                vq.Atom(1, [0.0, 0.0, 4.0]),
            ]
        )
        basis = vq.BasisSet(mol, "sto-3g")
        C = get_hf_orbital_provider(mol, basis)
        H = build_hamiltonian_mo(mol, basis, C)
        cas = casci(
            H.h1e, H.h2e, n_active_elec=2, n_active_orb=2,
            n_core=0, nuclear_repulsion=H.nuclear_repulsion,
        )
        D = make_rdm1(cas.ci_coeffs, cas.determinants, 2)
        n = np.linalg.eigvalsh(D) / 2.0
        assert np.all((n > 0.1) & (n < 0.9))
        expected_s = -n * np.log(n) - (1.0 - n) * np.log1p(-n)
        result = entanglement_from_density(C @ D @ C.T, vq.compute_overlap(basis))
        np.testing.assert_allclose(
            result.single_orbital_entropies, expected_s, atol=1e-12
        )
        assert result.total_information == pytest.approx(expected_s.sum(), abs=1e-12)
        assert result.total_information > 1.0
        assert result.mutual_information[0, 1] == pytest.approx(
            4.0 * n[0] * (1.0 - n[0]) * n[1] * (1.0 - n[1]), abs=1e-12
        )
        np.testing.assert_allclose(result.entanglement_bond_orders, [[0, 1], [1, 0]])

    def test_correlation_clusters_empty(self):
        """No clusters when MI is zero."""
        from vibeqc.entanglement import correlation_clusters

        mi = np.zeros((4, 4), dtype=np.float64)
        clusters = correlation_clusters(mi, threshold=0.1)
        assert clusters == []

    def test_correlation_clusters_connected(self):
        """Find connected clusters above threshold."""
        from vibeqc.entanglement import correlation_clusters

        mi = np.zeros((4, 4), dtype=np.float64)
        mi[0, 1] = 0.5
        mi[1, 0] = 0.5
        mi[1, 2] = 0.5
        mi[2, 1] = 0.5
        clusters = correlation_clusters(mi, threshold=0.1)
        assert len(clusters) == 1
        assert clusters[0] == {0, 1, 2}


# ---------------------------------------------------------------------------
# NBO
# ---------------------------------------------------------------------------


class TestNBO:
    """Natural Bond Orbital analysis."""

    def test_npa_charges_fail_closed_without_full_naos(self):
        """A Löwdin population must not be returned under the NPA label."""
        from vibeqc._vibeqc_core import compute_overlap
        from vibeqc.nbo import npa_charges

        mol = _h2o()
        basis = vq.BasisSet(mol, "sto-3g")
        result = vq.run_rhf(mol, basis)

        S = np.asarray(compute_overlap(basis))
        P = np.asarray(result.density)
        with pytest.raises(NotImplementedError, match="Natural Atomic Orbital"):
            npa_charges(P, S, basis, mol)

    def test_nbo_search_water(self):
        """NBO search finds bonding and lone-pair orbitals."""
        from vibeqc._vibeqc_core import compute_overlap
        from vibeqc.nbo import nbo_search

        mol = _h2o()
        basis = vq.BasisSet(mol, "sto-3g")
        result = vq.run_rhf(mol, basis)

        S = np.asarray(compute_overlap(basis))
        P = np.asarray(result.density)
        nbo_result = nbo_search(P, S, basis, mol)

        assert len(nbo_result.nbo_orbitals) > 0
        assert nbo_result.npa_charges.size == 0
        types = [n["type"] for n in nbo_result.nbo_orbitals]
        assert "BD" in types or "LP" in types

    def test_nbo_summary_nonempty(self):
        """NBO summary string is non-empty."""
        from vibeqc._vibeqc_core import compute_overlap
        from vibeqc.nbo import nbo_search

        mol = _h2o()
        basis = vq.BasisSet(mol, "sto-3g")
        result = vq.run_rhf(mol, basis)

        S = np.asarray(compute_overlap(basis))
        P = np.asarray(result.density)
        nbo_result = nbo_search(P, S, basis, mol)
        s = nbo_result.summary()
        assert "NBO" in s
        assert "NPA Atomic Charges" not in s
        assert len(s) > 50


# ---------------------------------------------------------------------------
# Periodic bond analysis
# ---------------------------------------------------------------------------


class TestPeriodicBondAnalysis:
    """Periodic bond analysis: Wiberg for Gamma-point SCF."""

    def test_periodic_wiberg_trivial(self):
        """periodic_wiberg_bond_orders delegates to wiberg_bond_orders."""
        from vibeqc.bond_analysis import periodic_wiberg_bond_orders

        mol = _h2()
        basis = vq.BasisSet(mol, "sto-3g")
        result = vq.run_rhf(mol, basis)

        bo = periodic_wiberg_bond_orders(result, basis, mol)
        assert bo.shape == (2, 2)
        assert bo[0, 1] > 0.1
