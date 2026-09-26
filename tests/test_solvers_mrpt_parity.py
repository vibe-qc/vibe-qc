"""Parity tests for the multireference methods CASCI / NEVPT2 / CASPT2.

History: these landed broken in ``0c8f0fcf`` (CASCI ≈ −104 Ha; NEVPT2
wrong-sign; CASPT2 ≡ 0) and were rebuilt + validated (audit 2026-05-30):

* **CASCI** — rebuilt on the validated unrestricted Slater–Condon engine
  with a correct frozen-core dressing; matches PySCF ``mcscf.CASCI``.
* **NEVPT2** — strongly-contracted NEVPT2 (Angeli 2001) via the uniform
  perturber-projection definition; matches PySCF ``mrpt.NEVPT`` per class.
* **CASPT2** — two variants share ``caspt2()``.  ``variant="ic"`` (default)
  is internally-contracted CASPT2 (Andersson 1992), validated against
  OpenMolcas ``&CASPT2`` to ≤2 µHa and **un-gated**; the OpenMolcas
  reference values are recorded below.  Its canonical eight-class IPEA
  contraction is also OpenMolcas-pinned.  ``variant="sc"``
  (strongly-contracted, Andersson 1990) stays gated behind
  ``VIBEQC_EXPERIMENTAL_MULTIREF=1``.  See ``handovers/HANDOVER_MULTIREF.md``.

PySCF and OpenMolcas are out-of-process references (PySCF via
``importorskip``; OpenMolcas values recorded as constants); neither is
imported into ``vibeqc`` itself (AGENTS.md ground rule 5).
"""

from __future__ import annotations

import numpy as np
import pytest
from vibeqc import Atom, BasisSet, Molecule
from vibeqc.solvers import (
    CASPT2Options,
    SelectedCIOptions,
    build_hamiltonian_mo,
    casci,
    caspt2,
    casscf,
    get_hf_orbital_provider,
    nevpt2,
    selected_casci,
)
from vibeqc.solvers import _mrpt

# Geometries (bohr) — match tests/test_solvers_active_space_api.py.
H2 = Molecule([Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])])
H2_ATOM = "H 0 0 0; H 0 0 1.4"
H2O = Molecule(
    [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [0.0, 1.43, -0.93]),
        Atom(1, [0.0, -1.43, -0.93]),
    ]
)
H2O_ATOM = "O 0 0 0; H 0 1.43 -0.93; H 0 -1.43 -0.93"
LIH = Molecule([Atom(3, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 3.0])])
LIH_ATOM = "Li 0 0 0; H 0 0 3.0"
HF_MOL = Molecule([Atom(9, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.7])])
HF_ATOM = "F 0 0 0; H 0 0 1.7"
BE = Molecule([Atom(4, [0.0, 0.0, 0.0])])
BE_ATOM = "Be 0 0 0"
FLAG = "VIBEQC_EXPERIMENTAL_MULTIREF"
TOL = 1e-7  # Ha — parity tolerance vs PySCF


def _ham(mol, basis_name):
    b = BasisSet(mol, basis_name)
    c = get_hf_orbital_provider(mol, b, method="rhf")
    return build_hamiltonian_mo(mol, b, c)


def _pyscf_casci(atom, basis, ncas, nelecas):
    pytest.importorskip("pyscf")
    from pyscf import gto, mcscf, scf

    mf = scf.RHF(gto.M(atom=atom, unit="Bohr", basis=basis, verbose=0)).run(verbose=0)
    return mcscf.CASCI(mf, ncas, nelecas).run(verbose=0).e_tot


def _pyscf_nevpt2(atom, basis, ncas, nelecas):
    pytest.importorskip("pyscf")
    from pyscf import gto, mcscf, mrpt, scf

    mf = scf.RHF(gto.M(atom=atom, unit="Bohr", basis=basis, verbose=0)).run(verbose=0)
    mc = mcscf.CASCI(mf, ncas, nelecas).run(verbose=0)
    return mc.e_tot, mc.e_tot + mrpt.NEVPT(mc).kernel()


def _pyscf_casscf(atom, basis, ncas, nelecas):
    pytest.importorskip("pyscf")
    from pyscf import gto, mcscf, scf

    mf = scf.RHF(gto.M(atom=atom, unit="Bohr", basis=basis, verbose=0)).run(verbose=0)
    return mcscf.CASSCF(mf, ncas, nelecas).run(verbose=0).e_tot


# ── CASCI ───────────────────────────────────────────────────────────────────


class TestCASCIParity:
    @pytest.mark.parametrize(
        "mol,atom,basis,n_core,n_act,n_elec",
        [
            (H2O, H2O_ATOM, "sto-3g", 4, 2, 2),  # CAS(2,2)
            (H2O, H2O_ATOM, "sto-3g", 3, 4, 4),  # CAS(4,4)
            (H2, H2_ATOM, "6-31g", 0, 2, 2),  # no-core CAS(2,2)
        ],
    )
    def test_casci_vs_pyscf(self, mol, atom, basis, n_core, n_act, n_elec):
        H = _ham(mol, basis)
        e = casci(
            H.h1e,
            H.h2e,
            n_active_elec=n_elec,
            n_active_orb=n_act,
            n_core=n_core,
            nuclear_repulsion=H.nuclear_repulsion,
        ).e_total
        assert abs(e - _pyscf_casci(atom, basis, n_act, n_elec)) < TOL

    def test_casci_full_space_is_fci(self):
        # full active space → CASCI == FCI == PySCF FCI
        pytest.importorskip("pyscf")
        from pyscf import fci, gto, scf

        H = _ham(H2, "6-31g")
        e = casci(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=H.norb,
            n_core=0,
            nuclear_repulsion=H.nuclear_repulsion,
        ).e_total
        mf = scf.RHF(gto.M(atom=H2_ATOM, unit="Bohr", basis="6-31g", verbose=0)).run()
        assert abs(e - (fci.FCI(mf).kernel()[0])) < TOL

    def test_casci_cored_is_variational(self):
        # The original bug returned −104 Ha (below FCI); a correct CASCI(n,m)
        # must lie at or below HF and at or above FCI.
        H = _ham(H2O, "sto-3g")
        e = casci(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=2,
            n_core=4,
            nuclear_repulsion=H.nuclear_repulsion,
        ).e_total
        assert -74.99 < e < -74.0


# ── NEVPT2 ────────────────────────────────────────────────────────────────


class TestNEVPT2Parity:
    @pytest.mark.parametrize(
        "mol,atom,basis,n_core,n_act,n_elec",
        [
            (H2, H2_ATOM, "6-31g", 0, 2, 2),
            (H2O, H2O_ATOM, "sto-3g", 4, 2, 2),
            (H2O, H2O_ATOM, "sto-3g", 3, 4, 4),
        ],
    )
    def test_nevpt2_total_vs_pyscf(self, mol, atom, basis, n_core, n_act, n_elec):
        H = _ham(mol, basis)
        n_virt = H.norb - n_core - n_act
        cas = casci(
            H.h1e,
            H.h2e,
            n_active_elec=n_elec,
            n_active_orb=n_act,
            n_core=n_core,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        pt = nevpt2(cas, H.h1e, H.h2e, n_core=n_core, n_virt=n_virt)
        _, e_ref = _pyscf_nevpt2(atom, basis, n_act, n_elec)
        assert abs(pt.e_total - e_ref) < TOL


# ── CASPT2 — internally-contracted (validated vs OpenMolcas) + SC (gated) ──

# OpenMolcas &CASPT2 reference totals on a fixed CASCI(HF) reference (&SCF +
# &RASSCF CIonly), Group=C1, IPEAshift=0.0, Bohr geometry, Frozen as noted.
# Generated locally with OpenMolcas (see examples/regression/core/
# runner_openmolcas.py).  vibe-qc's explicit internally-contracted CASPT2
# reproduces these to ≤2 µHa (worst observed: N2/STO-3G CAS(6,6), 1.8 µHa).
#   key = (mol, basis, n_core, n_act, n_elec, n_frozen) -> E_total
_OM_IC = {
    ("H2", "6-31g", 0, 2, 2, 0): -1.14670322,  # virtual excitations, no core
    ("H2O", "sto-3g", 3, 4, 4, 0): -74.97302712,  # core hole, all-electron
    ("H2O", "sto-3g", 3, 4, 4, 1): -74.97293321,  # frozen O 1s
}

# Same external oracle and fixed CASCI(HF) reference as _OM_IC, now with
# IPEAshift=0.25.  H2O/6-31G exercises all eight canonical IPEA classes.
_OM_IPEA = {
    ("H2", "6-31g", 0, 2, 2): -1.14616891,
    ("H2O", "6-31g", 3, 4, 4): -76.10385263,
}


class TestICCASPT2Parity:
    """Internally-contracted CASPT2 (default) vs OpenMolcas &CASPT2.

    The rigorous IC method (``ipea=0``) is un-gated: validated against
    OpenMolcas to ≤2 µHa (handovers/HANDOVER_MULTIREF.md, un-gating audit).
    """

    def test_direct_dispatch_threshold(self):
        assert not _mrpt._ic_caspt2_uses_direct_active_ci(
            norb=7,
            n_core=3,
            n_act=4,
        )
        assert _mrpt._ic_caspt2_uses_direct_active_ci(
            norb=24,
            n_core=3,
            n_act=4,
        )
        # Since 2026-06-10 the imaginary shift rides the direct path
        # (level-shift-corrected SPD solve); only IPEA pins jobs to the
        # explicit engine.
        assert _mrpt._ic_caspt2_uses_direct_active_ci(
            norb=24,
            n_core=3,
            n_act=4,
            imaginary=0.1,
        )
        assert not _mrpt._ic_caspt2_uses_direct_active_ci(
            norb=24,
            n_core=3,
            n_act=4,
            ipea=0.25,
        )

    @pytest.mark.parametrize("key", list(_OM_IC))
    def test_ic_vs_openmolcas(self, key):
        name, basis, n_core, n_act, n_elec, n_frozen = key
        mol = {"H2": H2, "H2O": H2O}[name]
        H = _ham(mol, basis)
        n_virt = H.norb - n_core - n_act
        cas = casci(
            H.h1e,
            H.h2e,
            n_active_elec=n_elec,
            n_active_orb=n_act,
            n_core=n_core,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        e = caspt2(
            cas,
            H.h1e,
            H.h2e,
            n_core=n_core,
            n_virt=n_virt,
            variant="ic",
            n_frozen=n_frozen,
        ).e_total
        assert abs(e - _OM_IC[key]) < 5e-6  # ≤5 µHa vs OpenMolcas

    def test_ic_ungated_by_default(self, monkeypatch):
        # The rigorous IC method (ipea=0) needs no experimental opt-in.
        monkeypatch.delenv(FLAG, raising=False)
        H = _ham(H2, "6-31g")
        cas = casci(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=2,
            n_core=0,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        e = caspt2(cas, H.h1e, H.h2e, n_core=0, n_virt=H.norb - 2).e_total
        assert abs(e - (-1.14670322)) < 5e-6

    @pytest.mark.parametrize("key", list(_OM_IPEA))
    def test_ipea_vs_openmolcas_ungated(self, monkeypatch, key):
        monkeypatch.delenv(FLAG, raising=False)
        name, basis, n_core, n_act, n_elec = key
        mol = {"H2": H2, "H2O": H2O}[name]
        H = _ham(mol, basis)
        cas = casci(
            H.h1e,
            H.h2e,
            n_active_elec=n_elec,
            n_active_orb=n_act,
            n_core=n_core,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        result = caspt2(
            cas,
            H.h1e,
            H.h2e,
            n_core=n_core,
            n_virt=H.norb - n_core - n_act,
            ipea=0.25,
        )
        assert abs(result.e_total - _OM_IPEA[key]) < 2e-6

    def test_ipea_basis_is_canonical_eight_class_set(self):
        specs = _mrpt._ipea_icb_specs(
            inactive=[0], active=[1, 2], secondary=[3, 4]
        )
        counts = {
            case: sum(spec[0] == case for spec in specs)
            for case in {spec[0] for spec in specs}
        }
        assert counts == {
            "SS<-II": 3,
            "SS<-AI": 8,
            "AS<-II": 4,
            "SS<-AA": 10,
            "AS<-AI": 16,
            "AA<-II": 3,
            "AS<-AA": 16,
            "AA<-AI": 8,
        }
        assert any(
            case == "AS<-AA" and u == v
            for case, _a, _t, u, v in specs
        )
        assert any(
            case == "AA<-AI" and u == v
            for case, _t, _i, u, v in specs
        )

    def test_ic_direct_path_ungated(self, monkeypatch):
        monkeypatch.delenv(FLAG, raising=False)
        monkeypatch.setattr(
            _mrpt,
            "_ic_caspt2_uses_direct_active_ci",
            lambda *args, **kwargs: True,
        )
        H = _ham(H2, "6-31g")
        cas = casci(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=2,
            n_core=0,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        e = caspt2(cas, H.h1e, H.h2e, n_core=0, n_virt=H.norb - 2).e_total
        assert abs(e - (-1.14670322)) < 5e-6

    def test_imaginary_shift_raises_energy(self, monkeypatch):
        # An imaginary level shift (Forsberg 1997) raises the energy by reducing
        # |E2|; on a well-behaved reference the change is tiny.
        monkeypatch.delenv(FLAG, raising=False)
        H = _ham(H2O, "sto-3g")
        cas = casci(
            H.h1e,
            H.h2e,
            n_active_elec=4,
            n_active_orb=4,
            n_core=3,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        e0 = caspt2(cas, H.h1e, H.h2e, n_core=3, n_virt=H.norb - 7).e_total
        es = caspt2(
            cas, H.h1e, H.h2e, n_core=3, n_virt=H.norb - 7, imaginary=0.1
        ).e_total
        assert e0 < es and (es - e0) < 1e-3

    def test_imaginary_shift_zero_limit(self, monkeypatch):
        # The Hylleraas-corrected imaginary-shift energy (Forsberg Eq 11) must
        # reduce to the unshifted energy as σ→0 — a regression guard on the
        # shift formula itself (an early draft used the wrong Eq 8, which did
        # not have this limit at finite σ but is moot here; this pins the
        # σ→0 reduction of the shipped Eq 11).
        monkeypatch.delenv(FLAG, raising=False)
        H = _ham(H2O, "sto-3g")
        cas = casci(
            H.h1e,
            H.h2e,
            n_active_elec=4,
            n_active_orb=4,
            n_core=3,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        e_unshifted = caspt2(cas, H.h1e, H.h2e, n_core=3, n_virt=H.norb - 7).e_total
        e_tiny = caspt2(
            cas, H.h1e, H.h2e, n_core=3, n_virt=H.norb - 7, imaginary=1e-8
        ).e_total
        assert abs(e_tiny - e_unshifted) < 1e-9

    def test_invalid_variant_raises(self):
        # Guards the variant dispatch branch in caspt2().
        H = _ham(H2, "6-31g")
        cas = casci(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=2,
            n_core=0,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        with pytest.raises(ValueError, match="variant"):
            caspt2(cas, H.h1e, H.h2e, n_core=0, n_virt=H.norb - 2, variant="bogus")


def _exact_sel_opts(**kw):
    """Selected-CI options driving the selection to the full-CI limit."""
    base = dict(
        target_size=100000, max_iter=80, conv_tol_energy=1e-13,
        pt2_threshold=1e-14, max_det_per_iter=100000, significant_coeff=1e-9,
    )
    base.update(kw)
    return SelectedCIOptions(**base)


class TestSelectedReferenceMRPT2:
    """Selected-CI reference -> MR-PT2 (roadmap 25i, M21/M22).

    The PT2 engines are determinant-based (no high-order RDMs), so a
    selected (possibly truncated) reference threads through
    _semicanonical_prep without re-solving the full CAS.  At the
    full-selection limit the result must reproduce the dense full-CAS
    MR-PT2 to machine precision (hence OpenMolcas for CASPT2); a truncated
    reference converges to it from above as the selection tightens.

    H2O/STO-3G CAS(4,4) on HF orbitals matches the recorded OpenMolcas
    constant in _OM_IC, so the dense and selected(full) CASPT2 are both
    pinned to the external oracle.
    """

    def _h2o_cas44(self):
        H = _ham(H2O, "sto-3g")
        n_core, n_act, n_elec = 3, 4, 4
        return H, n_core, n_act, n_elec, H.norb - n_core - n_act

    def test_full_limit_caspt2_matches_dense_and_openmolcas(self):
        H, n_core, n_act, n_elec, n_virt = self._h2o_cas44()
        dense = caspt2(
            casci(H.h1e, H.h2e, n_active_elec=n_elec, n_active_orb=n_act,
                  n_core=n_core, nuclear_repulsion=H.nuclear_repulsion),
            H.h1e, H.h2e, n_core=n_core, n_virt=n_virt,
        ).e_total
        sel = selected_casci(
            H.h1e, H.h2e, n_elec, n_act, n_core,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
            options=_exact_sel_opts(),
        )
        got = caspt2(
            sel, H.h1e, H.h2e, n_core=n_core, n_virt=n_virt,
            selected_reference=True,
        ).e_total
        assert abs(got - dense) < 1e-10           # == dense explicit engine
        assert abs(got - _OM_IC[("H2O", "sto-3g", 3, 4, 4, 0)]) < 5e-6  # == OM

    def test_full_limit_nevpt2_matches_dense(self):
        H, n_core, n_act, n_elec, n_virt = self._h2o_cas44()
        dense = nevpt2(
            casci(H.h1e, H.h2e, n_active_elec=n_elec, n_active_orb=n_act,
                  n_core=n_core, nuclear_repulsion=H.nuclear_repulsion),
            H.h1e, H.h2e, n_core=n_core, n_virt=n_virt,
        ).e_total
        sel = selected_casci(
            H.h1e, H.h2e, n_elec, n_act, n_core,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
            options=_exact_sel_opts(),
        )
        got = nevpt2(
            sel, H.h1e, H.h2e, n_core=n_core, n_virt=n_virt,
            selected_reference=True,
        ).e_total
        assert abs(got - dense) < 1e-10

    def test_exact_ci_as_selected_reference_is_identical(self):
        # Feeding the EXACT casci wavefunction through the selected_reference
        # path must reproduce the default (full-CAS re-solve) path bitwise:
        # the threading is the only difference, the math is the same.
        H, n_core, n_act, n_elec, n_virt = self._h2o_cas44()
        cas = casci(H.h1e, H.h2e, n_active_elec=n_elec, n_active_orb=n_act,
                    n_core=n_core, nuclear_repulsion=H.nuclear_repulsion)
        for fn in (nevpt2, caspt2):
            default = fn(cas, H.h1e, H.h2e, n_core=n_core, n_virt=n_virt).e_total
            threaded = fn(cas, H.h1e, H.h2e, n_core=n_core, n_virt=n_virt,
                          selected_reference=True).e_total
            assert abs(default - threaded) < 1e-12

    def test_truncated_reference_converges_to_dense(self):
        H, n_core, n_act, n_elec, n_virt = self._h2o_cas44()
        dense = caspt2(
            casci(H.h1e, H.h2e, n_active_elec=n_elec, n_active_orb=n_act,
                  n_core=n_core, nuclear_repulsion=H.nuclear_repulsion),
            H.h1e, H.h2e, n_core=n_core, n_virt=n_virt,
        ).e_total
        prev_gap = None
        for ts in (6, 10, 36):
            sel = selected_casci(
                H.h1e, H.h2e, n_elec, n_act, n_core,
                nuclear_repulsion=H.nuclear_repulsion, ms2=0,
                options=_exact_sel_opts(target_size=ts, max_det_per_iter=ts),
            )
            e = caspt2(sel, H.h1e, H.h2e, n_core=n_core, n_virt=n_virt,
                       selected_reference=True).e_total
            gap = e - dense
            assert gap > -1e-10                    # truncated ref is above dense
            if prev_gap is not None:
                assert gap <= prev_gap + 1e-10     # converges monotonically
            prev_gap = gap
        assert abs(prev_gap) < 1e-9                # full selection == dense

    def test_selected_reference_validation_raises(self):
        H, n_core, n_act, n_elec, n_virt = self._h2o_cas44()
        sel = selected_casci(
            H.h1e, H.h2e, n_elec, n_act, n_core,
            nuclear_repulsion=H.nuclear_repulsion, ms2=0,
            options=_exact_sel_opts(target_size=10, max_det_per_iter=10),
        )
        with pytest.raises(ValueError, match="variant='ic'"):
            caspt2(sel, H.h1e, H.h2e, n_core=n_core, n_virt=n_virt,
                   variant="sc", selected_reference=True)
        with pytest.raises(ValueError, match="IPEA"):
            caspt2(sel, H.h1e, H.h2e, n_core=n_core, n_virt=n_virt,
                   ipea=0.25, selected_reference=True)


class TestICCASPT2Limits:
    @pytest.mark.parametrize("variant", ["ic", "sc"])
    def test_mp2_limit(self, monkeypatch, variant):
        # Empty active space ⇒ CASPT2 must equal MP2 exactly (both variants).
        monkeypatch.setenv(FLAG, "1")  # required for sc, harmless for ic
        pytest.importorskip("pyscf")
        from pyscf import gto, mp, scf

        H = _ham(H2, "6-31g")
        cas = casci(
            H.h1e,
            H.h2e,
            n_active_elec=0,
            n_active_orb=0,
            n_core=1,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        e_corr = caspt2(
            cas, H.h1e, H.h2e, n_core=1, n_virt=H.norb - 1, variant=variant
        ).e_corr
        mf = scf.RHF(gto.M(atom=H2_ATOM, unit="Bohr", basis="6-31g", verbose=0)).run()
        assert abs(e_corr - mp.MP2(mf, frozen=0).run().e_corr) < TOL

    @pytest.mark.parametrize("variant", ["ic", "sc"])
    def test_full_space_is_zero(self, monkeypatch, variant):
        monkeypatch.setenv(FLAG, "1")
        H = _ham(H2, "6-31g")
        cas = casci(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=H.norb,
            n_core=0,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        assert (
            abs(caspt2(cas, H.h1e, H.h2e, n_core=0, n_virt=0, variant=variant).e_corr)
            < 1e-10
        )


class TestSCCASPT2Gated:
    def test_sc_gated_by_default(self, monkeypatch):
        monkeypatch.delenv(FLAG, raising=False)
        H = _ham(H2, "6-31g")
        cas = casci(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=2,
            n_core=0,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        with pytest.raises(RuntimeError, match=FLAG):
            caspt2(cas, H.h1e, H.h2e, n_core=0, n_virt=H.norb - 2, variant="sc")

    def test_cross_checks_nevpt2(self, monkeypatch):
        # SC CASPT2 and NEVPT2 differ only in H0 (generalized Fock vs Dyall),
        # so on a well-behaved reference their second-order corrections track
        # each other closely (~1.5 mHa across H2O / H2 CAS(2,2)/(4,4)).
        monkeypatch.setenv(FLAG, "1")
        H = _ham(H2O, "sto-3g")
        n_core, n_act, n_elec = 4, 2, 2
        n_virt = H.norb - n_core - n_act
        cas = casci(
            H.h1e,
            H.h2e,
            n_active_elec=n_elec,
            n_active_orb=n_act,
            n_core=n_core,
            nuclear_repulsion=H.nuclear_repulsion,
        )
        e_nevpt2 = nevpt2(cas, H.h1e, H.h2e, n_core=n_core, n_virt=n_virt).e_corr
        e_caspt2 = caspt2(
            cas, H.h1e, H.h2e, n_core=n_core, n_virt=n_virt, variant="sc"
        ).e_corr
        assert e_nevpt2 < 0 and e_caspt2 < 0  # both recover dynamic correlation
        assert abs(e_caspt2 - e_nevpt2) < 5e-3  # and track each other (~1.5 mHa)


# ── CASSCF (orbital-optimized CASCI; validated vs PySCF mcscf.CASSCF) ────────


class TestCASSCFParity:
    @pytest.mark.parametrize(
        "mol,atom,basis,n_core,n_act,n_elec",
        [
            (H2, H2_ATOM, "6-31g", 0, 2, 2),  # no core, 6-31g
            (LIH, LIH_ATOM, "sto-3g", 1, 2, 2),  # 1 core orbital
            (HF_MOL, HF_ATOM, "sto-3g", 4, 2, 2),  # 4 core orbitals
            (BE, BE_ATOM, "sto-3g", 1, 2, 2),  # 2s/2p near-degeneracy
        ],
    )
    def test_casscf_vs_pyscf(self, mol, atom, basis, n_core, n_act, n_elec):
        # Unique-minimum systems where both codes converge to the same basin.
        H = _ham(mol, basis)
        res = casscf(
            H.h1e,
            H.h2e,
            n_active_elec=n_elec,
            n_active_orb=n_act,
            n_core=n_core,
            nuclear_repulsion=H.nuclear_repulsion,
            ms2=0,
        )
        assert res.converged and res.grad_norm < 1e-6  # stationary point reached
        assert abs(res.e_total - _pyscf_casscf(atom, basis, n_act, n_elec)) < TOL

    def test_full_active_space_is_fci(self):
        # No inactive/virtual => no orbital rotations => CASSCF == CASCI == FCI.
        pytest.importorskip("pyscf")
        from pyscf import fci, gto, scf

        H = _ham(H2, "6-31g")
        res = casscf(
            H.h1e,
            H.h2e,
            n_active_elec=2,
            n_active_orb=H.norb,
            n_core=0,
            nuclear_repulsion=H.nuclear_repulsion,
            ms2=0,
        )
        mf = scf.RHF(gto.M(atom=H2_ATOM, unit="Bohr", basis="6-31g", verbose=0)).run()
        assert abs(res.e_total - fci.FCI(mf).kernel()[0]) < TOL

    def test_variational_sandwich(self):
        # Basis-independent correctness check that holds even where CASSCF finds
        # a deeper minimum than PySCF's HF-start: E_FCI <= E_CASSCF <= E_CASCI.
        H = _ham(H2O, "sto-3g")
        kw = dict(nuclear_repulsion=H.nuclear_repulsion, ms2=0)
        e_casci = casci(H.h1e, H.h2e, 2, 2, 4, **kw).e_total
        e_fci = casci(H.h1e, H.h2e, H.nelec, H.norb, 0, **kw).e_total
        res = casscf(H.h1e, H.h2e, 2, 2, 4, **kw)
        assert e_fci - 1e-9 <= res.e_total <= e_casci + 1e-9
        assert res.e_total < e_casci  # orbital optimization strictly improves here

    def test_analytic_orbital_gradient_matches_fd(self):
        # The one new numeric kernel: g_pq = 2(F_qp - F_pq).  Validate it
        # elementwise against finite differences of the trusted CASCI energy.
        from scipy.linalg import expm
        from vibeqc.solvers._casscf import (
            _build_kappa,
            _nonredundant_pairs,
            _orbital_gradient_and_fock,
            _rotate_integrals,
        )
        from vibeqc.solvers._rdm import make_rdm12

        H = _ham(H2, "6-31g")
        n_core, n_act, n_elec, norb = 0, 2, 2, H.norb
        cas = casci(H.h1e, H.h2e, n_elec, n_act, n_core, H.nuclear_repulsion, ms2=0)
        dm1, dm2 = make_rdm12(cas.ci_coeffs, cas.determinants, n_act)
        pairs = _nonredundant_pairs(n_core, n_act, norb)
        g_an, _, _, _ = _orbital_gradient_and_fock(
            H.h1e, H.h2e.transpose(0, 2, 1, 3), dm1, dm2, n_core, n_act, norb, pairs
        )

        def energy(x):
            U = expm(_build_kappa(x, pairs, norb))
            h1, g2 = _rotate_integrals(H.h1e, H.h2e, U)
            return casci(
                h1, g2, n_elec, n_act, n_core, H.nuclear_repulsion, ms2=0
            ).e_total

        eps = 1e-5
        ident = np.eye(len(pairs))
        g_fd = np.array(
            [
                (energy(ident[k] * eps) - energy(-ident[k] * eps)) / (2 * eps)
                for k in range(len(pairs))
            ]
        )
        assert np.max(np.abs(g_an - g_fd)) < 1e-6

    def test_nevpt2_runs_on_casscf_reference(self):
        # The purpose of CASSCF: NEVPT2 then runs on the optimized reference
        # (res.cas in the optimized basis res.h1e_cas / res.h2e_cas).
        H = _ham(H2O, "sto-3g")
        n_core, n_act, n_elec = 4, 2, 2
        n_virt = H.norb - n_core - n_act
        res = casscf(
            H.h1e,
            H.h2e,
            n_elec,
            n_act,
            n_core,
            nuclear_repulsion=H.nuclear_repulsion,
            ms2=0,
        )
        pt = nevpt2(res.cas, res.h1e_cas, res.h2e_cas, n_core=n_core, n_virt=n_virt)
        assert pt.e_total < res.e_total  # PT2 lowers the CASSCF energy


# ── run_job dispatch ────────────────────────────────────────────────────────


def _run(method, tmp_path, **kw):
    from vibeqc.runner import run_job

    return run_job(
        H2O,
        basis="sto-3g",
        method=method,
        output=tmp_path / method,
        write_molden_file=False,
        write_xyz_file=False,
        write_population_file=False,
        citations=False,
        **kw,
    )


class TestRunJob:
    def test_casci_and_nevpt2_ungated(self, tmp_path):
        e_cas = _run("casci", tmp_path, active_space=(2, 2)).energy
        e_nev = _run("nevpt2", tmp_path, active_space=(2, 2)).energy
        assert e_cas < -74.0 and e_nev < e_cas  # PT2 lowers the energy

    def test_caspt2_ungated_and_lowers_energy(self, tmp_path, monkeypatch):
        # method="caspt2" runs the validated IC variant un-gated; PT2 lowers
        # the CASCI energy.
        monkeypatch.delenv(FLAG, raising=False)
        e_casci = _run("casci", tmp_path, active_space=(2, 2)).energy
        e_caspt2 = _run("caspt2", tmp_path, active_space=(2, 2)).energy
        assert e_caspt2 < e_casci

    def test_casscf_ungated_and_improves_on_casci(self, tmp_path):
        # CASSCF is validated vs PySCF, so it ships ungated like CASCI/NEVPT2.
        e_casci = _run("casci", tmp_path, active_space=(2, 2)).energy
        e_casscf = _run("casscf", tmp_path, active_space=(2, 2)).energy
        assert e_casscf <= e_casci + 1e-9  # orbital optimization never worse

    def test_caspt2_options_reach_ic_engine(self, tmp_path):
        # caspt2_options threads the imaginary shift + frozen core through
        # run_job into the IC engine.
        e_def = _run("caspt2", tmp_path, active_space=(4, 4)).energy
        e_imag = _run(
            "caspt2",
            tmp_path,
            active_space=(4, 4),
            caspt2_options=CASPT2Options(imaginary=0.1),
        ).energy
        e_froz = _run(
            "caspt2",
            tmp_path,
            active_space=(4, 4),
            caspt2_options=CASPT2Options(n_frozen=1),
        ).energy
        assert e_imag > e_def  # imaginary shift raises the energy
        assert abs(e_froz - e_def) > 1e-6  # frozen core changes the energy

    def test_caspt2_options_sc_variant_gated(self, tmp_path, monkeypatch):
        # variant="sc" through run_job is still gated.
        monkeypatch.delenv(FLAG, raising=False)
        with pytest.raises(RuntimeError, match=FLAG):
            _run(
                "caspt2",
                tmp_path,
                active_space=(4, 4),
                caspt2_options=CASPT2Options(variant="sc"),
            )


# ── CASSCF-referenced CASPT2 / NEVPT2 (the standard composition) ───────────

N2 = Molecule([Atom(7, [0.0, 0.0, 0.0]), Atom(7, [0.0, 0.0, 2.074])])

# OpenMolcas full &RASSCF (orbital-optimized CASSCF) + &CASPT2 totals,
# Frozen=0, IPEAshift=0, Group=C1.  Generated 2026-06-10 with
# examples/regression/core/runner_openmolcas.py::run_caspt2(ci_only=False)
# against a local OpenMolcas build (out-of-process per CLAUDE.md §10).
# Key: (molecule, basis, n_core, n_act, n_elec) -> (E_CASSCF, E_CASPT2).
#
# Basin note (2026-06-10): H2 and N2 have a unique CAS minimum and all codes
# agree to ≤5e-6 Ha.  H2O CAS(4,4) is basin-rich — three codes converge to
# three different stationary points (OpenMolcas -74.95274480, PySCF CASSCF
# -74.95975401, vibe-qc -74.97931835 on STO-3G; OpenMolcas -75.99466564,
# PySCF -76.02721390 on 6-31G), so equality with OpenMolcas is asserted only
# for H2/N2; for H2O vibe-qc must land at least as deep as the recorded
# OpenMolcas basin (deeper-or-equal is the variationally meaningful check).
_OM_CASSCF = {
    ("H2", "6-31g", 0, 2, 2): (-1.14624788, -1.15031051),
    ("H2O", "sto-3g", 3, 4, 4): (-74.95274480, -74.97461239),
    ("H2O", "6-31g", 3, 4, 4): (-75.99466564, -76.10397578),
    ("N2", "sto-3g", 4, 6, 6): (-107.63683522, -107.64834018),
}
_OM_TOL = 5e-6  # Ha — same gate as the recorded CIonly OpenMolcas parity


class TestCASSCFReferencedCASPT2:
    """CASSCF→CASPT2 composition vs OpenMolcas full &RASSCF + &CASPT2.

    The historical (and still default) ``method="caspt2"`` reference is
    CASCI on HF orbitals (matching ``RASSCF … CIonly``).  Supplying
    ``casscf_options`` switches the reference to an orbital-optimized
    CASSCF — the standard CASPT2 workflow.  On unique-minimum systems
    (H2, N2) both the CASSCF reference and the PT2 total must match
    OpenMolcas's RASSCF/CASPT2 pair; on the basin-rich H2O cases the
    deeper-or-equal + variational-bound checks apply (see _OM_CASSCF).
    """

    @pytest.mark.parametrize(
        "mol,key",
        [
            (H2, ("H2", "6-31g", 0, 2, 2)),
            (N2, ("N2", "sto-3g", 4, 6, 6)),
        ],
    )
    def test_casscf_caspt2_vs_openmolcas(self, mol, key):
        _, basis, n_core, n_act, n_elec = key
        e_om_casscf, e_om_caspt2 = _OM_CASSCF[key]
        H = _ham(mol, basis)
        n_virt = H.norb - n_core - n_act
        sc = casscf(
            H.h1e,
            H.h2e,
            n_active_elec=n_elec,
            n_active_orb=n_act,
            n_core=n_core,
            nuclear_repulsion=H.nuclear_repulsion,
            ms2=0,
        )
        assert sc.converged
        assert abs(sc.e_total - e_om_casscf) < _OM_TOL
        pt = caspt2(sc.cas, sc.h1e_cas, sc.h2e_cas, n_core=n_core, n_virt=n_virt)
        assert abs(pt.e_total - e_om_caspt2) < _OM_TOL

    def test_casscf_h2o_lands_at_least_as_deep_as_openmolcas(self):
        # H2O CAS(4,4) is basin-rich (see _OM_CASSCF note): vibe-qc's
        # optimizer must land at least as deep as OpenMolcas's recorded
        # stationary point, bounded below by FCI, and the PT2 on that
        # reference must still lower the energy.
        e_om_casscf, _ = _OM_CASSCF[("H2O", "sto-3g", 3, 4, 4)]
        H = _ham(H2O, "sto-3g")
        sc = casscf(
            H.h1e,
            H.h2e,
            n_active_elec=4,
            n_active_orb=4,
            n_core=3,
            nuclear_repulsion=H.nuclear_repulsion,
            ms2=0,
        )
        assert sc.converged
        e_fci = casci(
            H.h1e,
            H.h2e,
            H.nelec,
            H.norb,
            0,
            nuclear_repulsion=H.nuclear_repulsion,
            ms2=0,
        ).e_total
        assert e_fci - 1e-9 <= sc.e_total <= e_om_casscf + _OM_TOL
        pt = caspt2(sc.cas, sc.h1e_cas, sc.h2e_cas, n_core=3, n_virt=H.norb - 7)
        assert pt.e_total < sc.e_total

    def test_run_job_caspt2_casscf_reference(self, tmp_path):
        # casscf_options through run_job switches the reference: the label
        # records it, the CASSCF reference is at least as deep as the
        # recorded OpenMolcas basin, and PT2 lowers it (basin-rich H2O —
        # see _OM_CASSCF note; exact OM equality is pinned on H2/N2).
        from vibeqc.solvers import CASSCFOptions

        e_om_casscf, _ = _OM_CASSCF[("H2O", "sto-3g", 3, 4, 4)]
        res = _run(
            "caspt2",
            tmp_path,
            active_space=(4, 4),
            casscf_options=CASSCFOptions(),
        )
        assert res.method.endswith("_casscf")
        assert res.energy < e_om_casscf  # PT2 below the CASSCF reference

    def test_run_job_nevpt2_casscf_reference_vs_pyscf(self, tmp_path):
        # NEVPT2 on the CASSCF reference vs PySCF mcscf.CASSCF + mrpt.NEVPT
        # (unique-minimum system so both codes land in the same basin).
        pytest.importorskip("pyscf")
        from pyscf import gto, mcscf, mrpt, scf
        from vibeqc.runner import run_job
        from vibeqc.solvers import CASSCFOptions

        res = run_job(
            LIH,
            basis="sto-3g",
            method="nevpt2",
            active_space=(2, 2),
            casscf_options=CASSCFOptions(),
            output=tmp_path / "nevpt2_casscf",
            write_molden_file=False,
            write_xyz_file=False,
            write_population_file=False,
            citations=False,
        )
        assert res.method.endswith("_casscf")
        mf = scf.RHF(
            gto.M(atom=LIH_ATOM, unit="Bohr", basis="sto-3g", verbose=0)
        ).run(verbose=0)
        mc = mcscf.CASSCF(mf, 2, 2).run(verbose=0)
        e_ref = mc.e_tot + mrpt.NEVPT(mc).kernel()
        assert abs(res.energy - e_ref) < 1e-6

    def test_run_job_sa_casscf_caspt2_surfaces_roots(self, tmp_path):
        # SA-CASSCF reference: PT2 runs on the lowest root of the
        # state-averaged orbitals; the per-root CASSCF energies are surfaced.
        from vibeqc.solvers import CASSCFOptions

        res = _run(
            "caspt2",
            tmp_path,
            active_space=(2, 2),
            casscf_options=CASSCFOptions(nroots=2),
        )
        assert res.method.endswith("_casscf")
        assert res.root_energies is not None and len(res.root_energies) == 2
        # PT2 lowers the energy relative to the root it perturbs (root 0).
        assert res.energy < res.root_energies[0]

    def test_casscf_reference_rejects_active_orbitals_window(self, tmp_path):
        from vibeqc.solvers import CASSCFOptions

        with pytest.raises(ValueError, match="active_orbitals"):
            _run(
                "caspt2",
                tmp_path,
                active_space=(2, 2),
                casscf_options=CASSCFOptions(active_orbitals=[3, 4]),
            )

    @pytest.mark.parametrize("method", ["caspt2", "nevpt2"])
    def test_run_job_selected_reference_routes_and_lowers_its_own_root(
        self, tmp_path, method
    ):
        """Selected-CI reference -> MR-PT2 through run_job (M23).

        What run_job owns here is the plumbing: that
        ``casscf_options(ci_solver="selected_ci")`` actually reaches the
        CASSCF reference and that the MR-PT2 engine then perturbs it.

        It deliberately does NOT compare the two legs' energies. H2O CAS(4,4)
        is basin-rich (#56), and the two legs optimize orbitals
        independently, so they land in different CASSCF stationary points --
        measured here, and bit-identical to the pair #56 recorded:

            dense    -74.95975393685424
            selected -74.95274477996531   (gap 7.009156888927e-03 Ha)

        The MR-PT2 layer partly compensates that (the higher reference takes
        the larger correction), leaving totals 1.998886868748e-03 apart for
        CASPT2 and 4.167662930428e-04 for NEVPT2. An earlier revision
        asserted those totals equal to 1e-9 and was red on main for exactly
        the reason #56 documents (#277).

        The full-selection parity claim is real, but it only means anything
        at FIXED orbitals, which is where ``TestSelectedReferenceMRPT2``
        tests it: ``test_full_limit_caspt2_matches_dense`` and
        ``test_full_limit_nevpt2_matches_dense`` compare the dense and
        selected-CI engines on one Hamiltonian to 1e-10. This mirrors the
        precedent ``tests/test_selected_casci.py`` set for #56.
        """
        from vibeqc.solvers import CASSCFOptions

        common = dict(active_space=(4, 4))

        def both_legs(job):
            dense = _run(
                job, tmp_path / f"d_{job}", casscf_options=CASSCFOptions(),
                **common,
            )
            sel = _run(
                job, tmp_path / f"s_{job}",
                casscf_options=CASSCFOptions(
                    ci_solver="selected_ci",
                    selected_ci_options=_exact_sel_opts(),
                ),
                **common,
            )
            return dense, sel

        ref_dense, ref_sel = both_legs("casscf")
        pt_dense, pt_sel = both_legs(method)

        # The selected-CI backend really drove the reference, and the PT2
        # engine really ran on a CASSCF (not a CASCI-on-HF) reference.
        assert ref_sel.method.endswith("_selci")
        assert not ref_dense.method.endswith("_selci")
        assert pt_dense.method.endswith("_casscf")
        assert pt_sel.method.endswith("_casscf")

        # Each leg's PT2 lowers the root it actually perturbed. Comparing a
        # total against its OWN reference is basin-safe; comparing the two
        # legs against each other is not. Measured margins are 15-22 mHa.
        assert pt_dense.energy < ref_dense.energy
        assert pt_sel.energy < ref_sel.energy

    def test_run_job_selected_reference_multistate_rejected(self, tmp_path):
        from vibeqc.solvers import CASPT2Options, CASSCFOptions

        with pytest.raises(ValueError, match="multi-state"):
            _run(
                "caspt2",
                tmp_path,
                active_space=(4, 4),
                caspt2_options=CASPT2Options(multistate="ms"),
                casscf_options=CASSCFOptions(
                    nroots=2, ci_solver="selected_ci",
                    selected_ci_options=_exact_sel_opts(),
                ),
            )
