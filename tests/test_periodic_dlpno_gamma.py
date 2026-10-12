"""The GDF route of DLPNO-CCSD(T) on crystals at the Gamma point (increment
B1 of docs/design_dlpno_correlation_source.md) and on a Gamma-centred mesh
folded to the real BvK supercell (increment B2).

Pins: the adapter's factors are the SCF's own fit (the Coulomb energy
rebuilt from them equals the SCF's), full domains reproduce the canonical
CCSD on the same source, the route runs with triples and records its source,
the folded mesh equals the explicit Gamma supercell, the run_periodic_job
front door (increment B2b) lands the result on every artifact, and the
refused references and requests fail closed with their reason.
"""

from __future__ import annotations

import json
import tomllib
import zipfile
from copy import copy
from types import SimpleNamespace

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.dlpno._ccsd_cs import run_cs_ccsd
from vibeqc.dlpno._ccsd_ref import run_ref_uccsd
from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions, run_local_dlpno_ccsd
from vibeqc.dlpno.uccsd_local_solver import LocalUCCSDOptions
from vibeqc.kpoints import KPoints
from vibeqc.dlpno.source import (
    CorrelationSourceError,
    bvk_supercell_source,
    gamma_gdf_source,
    refuse_unsupported_periodic_reference,
)
from vibeqc.periodic.chi.scf import cyclic_gamma_mesh
from vibeqc.periodic_dlpno import correlate_periodic_reference, prepare_periodic_cc_options, run_periodic_dlpno_ccsd
from vibeqc.periodic_k_gdf import run_krhf_periodic_gdf, run_kuhf_periodic_gdf
from vibeqc.periodic_toroidal_mp2 import build_toroidal_supercell

FULL = dict(
    localise="none",
    n_frozen=0,
    tcut_pno=0.0,
    tcut_mkn=0.0,
    tcut_pairs=0.0,
    coupling_radius=0.0,
)


def _h2_chain_cell(spacing_bohr=6.0, hh_bond=1.4, transverse=12.0):
    """One H2 per cell, periodic along z, vacuum padding in x and y."""
    return vq.PeriodicSystem(
        3,
        np.diag([transverse, transverse, spacing_bohr]),
        [
            vq.Atom(1, [transverse / 2, transverse / 2, spacing_bohr / 2 - hh_bond / 2]),
            vq.Atom(1, [transverse / 2, transverse / 2, spacing_bohr / 2 + hh_bond / 2]),
        ],
    )


def _h2_cube(a=6.0):
    """H2 in a cubic cell: a bulk-like Gamma reference with two AOs per cell."""
    return vq.PeriodicSystem(
        3, np.diag([a, a, a]),
        [vq.Atom(1, [a / 2, a / 2, a / 2 - 0.7]), vq.Atom(1, [a / 2, a / 2, a / 2 + 0.7])],
    )


@pytest.fixture(scope="module")
def gamma_reference():
    system = _h2_chain_cell()
    basis = vq.BasisSet(system.unit_cell_molecule(), "def2-svp")
    hf = run_krhf_periodic_gdf(
        system, basis, (1, 1, 1), fit_source="classic", retain_fit_cache=True, progress=False,
    )
    assert hf.converged
    return system, basis, hf


class TestGammaSource:
    def test_factors_are_the_scf_fit(self, gamma_reference):
        """J rebuilt from the adapter's factors reproduces the SCF Coulomb energy."""
        system, basis, hf = gamma_reference
        source = gamma_gdf_source(system, basis, hf, localise="none")
        assert source.kind == "gamma-gdf"
        assert source.conventions.same_source
        assert source.conventions.exchange_q0 == str(hf.exchange_q0)
        assert source.provenance["fit"]["factor_sha256"] == hf.executed_fit_source.factor_sha256
        L = source.factors.B
        D = np.real(np.asarray(hf.density[0]))
        rho = np.einsum("Pmn,mn->P", L, D, optimize=True)
        J = np.einsum("P,Pmn->mn", rho, L, optimize=True)
        e_j = 0.5 * float(np.einsum("mn,mn->", D, J, optimize=True))
        assert e_j == pytest.approx(float(hf.e_coulomb), abs=1e-6)

    def test_identity_names_lattice_bases_and_fit(self, gamma_reference):
        system, basis, hf = gamma_reference
        source = gamma_gdf_source(system, basis, hf)
        record = source.provenance
        assert record["kind"] == "gamma-gdf"
        assert record["basis"] == "def2-svp"
        assert len(record["lattice"]) == 3 and len(record["atoms"]) == 2
        assert record["fit"] is not None
        assert len(source.identity) == 64
        again = gamma_gdf_source(system, basis, hf)
        assert again.identity == source.identity
        assert gamma_gdf_source(system, basis, hf, localise="none").identity != source.identity

    def test_localised_reference_keeps_the_occupied_span(self, gamma_reference):
        system, basis, hf = gamma_reference
        canonical = gamma_gdf_source(system, basis, hf, localise="none").reference
        localised = gamma_gdf_source(system, basis, hf).reference
        S = canonical.overlap
        n_occ = int(system.unit_cell_molecule().n_electrons()) // 2
        c_can, c_loc = canonical.mo_coeffs[:, :n_occ], localised.mo_coeffs[:, :n_occ]
        U = c_can.T @ S @ c_loc
        assert np.allclose(U.T @ U, np.eye(n_occ), atol=1e-8)
        assert np.allclose(c_loc.T @ S @ c_loc, np.eye(n_occ), atol=1e-8)
        assert localised.localisation == "pipek-mezey"
        assert np.array_equal(canonical.mo_coeffs[:, n_occ:], localised.mo_coeffs[:, n_occ:])

    def test_full_domain_equals_canonical_ccsd_on_the_same_source(self, gamma_reference):
        system, basis, hf = gamma_reference
        source = gamma_gdf_source(system, basis, hf, localise="none")
        ref = source.reference
        C, F = ref.mo_coeffs, ref.fock
        n_occ = int(system.unit_cell_molecule().n_electrons()) // 2
        f_mo = C.T @ F @ C
        Co, Cv = C[:, :n_occ], C[:, n_occ:]
        canon = run_cs_ccsd(
            f_mo,
            np.asarray(source.factors.mo_transform(Co, Cv)),
            np.asarray(source.factors.mo_transform(Cv, Cv)),
            np.asarray(source.factors.mo_transform(Co, Co)),
            n_occ,
            e_hf=ref.energy,
        )
        local = run_local_dlpno_ccsd(None, None, None, None, LocalCCSDOptions(**FULL), source=source)
        assert local.converged
        assert local.e_corr == pytest.approx(canon.e_corr, abs=1e-8)
        assert local.source_kind == "gamma-gdf"
        assert local.source_identity == source.identity


class TestRoute:
    def test_runs_with_triples_and_records_the_source(self):
        res = run_periodic_dlpno_ccsd(_h2_cube(), "def2-svp", with_triples=True, progress=False)
        assert res.converged
        assert res.kmesh == (1, 1, 1) and res.n_cells == 1
        assert res.e_corr < 0.0
        assert res.e_t is not None and res.e_t <= 0.0
        assert res.e_total == pytest.approx(res.e_hf + res.e_corr + res.e_t)
        assert res.source_kind == "gamma-gdf" and len(res.source_identity) == 64
        assert res.same_source and res.exchange_q0 == str(res.hf_result.exchange_q0)
        assert res.localisation == "pipek-mezey"
        assert res.cc_result.source_identity == res.source_identity
        assert res.n_occ == 1 and res.n_vir == res.n_bf - 1

    def test_options_must_leave_localisation_to_the_adapter(self):
        """The dataclass default "boys" is replaced by "none" (the adapter
        localises); any other explicit localiser is refused with the reason."""
        from vibeqc.periodic_dlpno import prepare_periodic_cc_options

        prepared = prepare_periodic_cc_options(LocalCCSDOptions(), with_triples=True)
        assert prepared.localise == "none" and prepared.compute_triples
        with pytest.raises(ValueError, match="localise must be 'none'"):
            run_periodic_dlpno_ccsd(
                _h2_cube(), "sto-3g",
                cc_options=LocalCCSDOptions(localise="pipek-mezey"),
                progress=False,
            )

    def test_refuses_bad_meshes_and_low_dimensions(self):
        with pytest.raises(ValueError, match="three positive integers"):
            run_periodic_dlpno_ccsd(_h2_cube(), "sto-3g", kmesh=(0, 1, 1), progress=False)
        chain = vq.PeriodicSystem(
            1, np.diag([12.0, 12.0, 6.0]),
            [vq.Atom(1, [6.0, 6.0, 2.3]), vq.Atom(1, [6.0, 6.0, 3.7])],
        )
        with pytest.raises(NotImplementedError, match="dim=3"):
            run_periodic_dlpno_ccsd(chain, "sto-3g", progress=False)


class TestRefusals:
    def test_bipole_reference_is_refused_with_the_reason(self):
        fake = SimpleNamespace(runtime_backend="pbc-bipole", mo_coeffs=[np.eye(2)])
        with pytest.raises(CorrelationSourceError, match="BIPOLE"):
            refuse_unsupported_periodic_reference(fake)
        with pytest.raises(CorrelationSourceError, match="decision 2"):
            gamma_gdf_source(_h2_cube(), None, fake)

    def test_symmetry_reduced_reference_is_refused(self):
        fake = SimpleNamespace(
            backend="native-multi-k-ewald3d", used_kpoint_symmetry_unfolding=True,
            mo_coeffs=[np.eye(2)],
        )
        with pytest.raises(CorrelationSourceError, match="irreducible wedge"):
            refuse_unsupported_periodic_reference(fake)

    def test_multi_k_and_missing_fit_are_refused(self):
        multi = SimpleNamespace(backend="native-multi-k-gdf", mo_coeffs=[np.eye(2), np.eye(2)])
        with pytest.raises(CorrelationSourceError, match="bvk_supercell_source"):
            gamma_gdf_source(_h2_cube(), None, multi)
        nofit = SimpleNamespace(
            backend="native-multi-k-gdf", mo_coeffs=[np.eye(2)], overlap=[np.eye(2)],
            fock=[np.eye(2)], mo_energies=[np.zeros(2)], kpoints_cart=np.zeros((1, 3)),
            energy=-1.0, converged=True, fit_cache=None,
        )
        with pytest.raises(CorrelationSourceError, match="fit_source='classic'"):
            gamma_gdf_source(_h2_cube(), None, nofit)

    def test_default_fit_source_result_is_refused_for_want_of_its_fit(self):
        system = _h2_cube()
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        auto = run_krhf_periodic_gdf(system, basis, (1, 1, 1), retain_fit_cache=True, progress=False)
        assert auto.fit_cache is None
        with pytest.raises(CorrelationSourceError, match="Gamma fast path"):
            gamma_gdf_source(system, basis, auto)


@pytest.fixture(scope="module")
def chain():
    """A tighter H2 chain cell for the (1, 1, 2) fold: four AOs in STO-3G."""
    return _h2_chain_cell(spacing_bohr=5.0, transverse=10.0)


class TestBvKSupercell:
    """A Gamma-centred mesh folded to the real BvK supercell is the explicit
    Gamma supercell problem (Theorem 1 of the Bloch fold); both run through
    the same engine."""

    def test_fold_equals_the_explicit_gamma_supercell_at_full_domains(self, chain):
        mesh = (1, 1, 2)
        folded = run_periodic_dlpno_ccsd(
            chain, "sto-3g", kmesh=mesh, with_triples=False, localise="none",
            cc_options=LocalCCSDOptions(**FULL), progress=False,
        )
        explicit = run_periodic_dlpno_ccsd(
            build_toroidal_supercell(chain, mesh), "sto-3g", with_triples=False,
            localise="none", cc_options=LocalCCSDOptions(**FULL), progress=False,
        )
        assert folded.n_cells == 2 and explicit.n_cells == 1
        assert folded.source_kind == "bvk-supercell" and folded.same_source
        assert folded.e_hf_per_cell == pytest.approx(explicit.e_hf_per_cell / 2, abs=1e-6)
        assert folded.e_corr_per_cell == pytest.approx(explicit.e_corr_per_cell / 2, abs=1e-6)

    def test_folded_full_domain_equals_canonical_ccsd_on_the_folded_source(self, chain):
        mesh = (1, 1, 2)
        basis = vq.BasisSet(chain.unit_cell_molecule(), "sto-3g")
        hf = run_krhf_periodic_gdf(
            chain, basis, cyclic_gamma_mesh(chain, mesh), fit_source="classic",
            retain_fit_cache=True, progress=False,
        )
        source = bvk_supercell_source(chain, basis, hf, mesh=mesh, localise="none")
        ref = source.reference
        C, F = ref.mo_coeffs, ref.fock
        n_occ = int(source.molecule.n_electrons()) // 2
        Co, Cv = C[:, :n_occ], C[:, n_occ:]
        canon = run_cs_ccsd(
            C.T @ F @ C,
            np.asarray(source.factors.mo_transform(Co, Cv)),
            np.asarray(source.factors.mo_transform(Cv, Cv)),
            np.asarray(source.factors.mo_transform(Co, Co)),
            n_occ,
            e_hf=ref.energy,
        )
        local = run_local_dlpno_ccsd(None, None, None, None, LocalCCSDOptions(**FULL), source=source)
        assert local.converged
        assert local.e_corr == pytest.approx(canon.e_corr, abs=1e-8)
        assert ref.energy == pytest.approx(2 * float(hf.energy))
        assert source.provenance["mesh"] == [1, 1, 2]
        assert max(source.provenance["fold_residuals"].values()) < 1e-7

    def test_default_route_runs_localised_with_triples(self, chain):
        res = run_periodic_dlpno_ccsd(chain, "sto-3g", kmesh=(1, 1, 2), progress=False)
        assert res.converged and res.n_cells == 2
        assert res.localisation == "pipek-mezey"
        assert res.e_corr < 0.0 and res.e_t is not None
        assert res.e_total_per_cell == pytest.approx(res.e_total / 2)

    def test_half_shifted_mesh_is_refused(self, chain):
        """A Monkhorst-Pack (1, 1, 2) mesh sits at k_z = 1/4 and 3/4: not the
        finite translation group of the torus, so the fold refuses it."""
        basis = vq.BasisSet(chain.unit_cell_molecule(), "sto-3g")
        mp = KPoints.monkhorst_pack(chain, (1, 1, 2))
        assert np.allclose(np.sort(mp.kpoints_frac[:, 2]), [0.25, 0.75])
        shifted = run_krhf_periodic_gdf(
            chain, basis, mp, fit_source="classic", retain_fit_cache=True, progress=False,
        )
        with pytest.raises(CorrelationSourceError, match="Gamma-centred unreduced mesh"):
            bvk_supercell_source(chain, basis, shifted, mesh=(1, 1, 2))


class TestFrontDoor:
    """run_periodic_job(method="dlpno-ccsd(t)") on the GDF route (increment
    B2b, #508 decision 3): the reference runs on the k-mesh engine with its
    classic fit retained, the engine correlates it through the adapters, and
    the result lands on the .out, the .system manifest, the structured log,
    the citations and the QVF. Refusals fire before any SCF work."""

    def test_gamma_front_door_writes_every_artifact(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        system = _h2_cube()
        basis = vq.BasisSet(system.unit_cell_molecule(), "def2-svp")
        res = vq.run_periodic_job(
            system, basis, method="dlpno-ccsd(t)", output="h2g",
            structured_log=True, progress=False,
        )
        assert res.converged and res.dlpno_ccsd.converged
        pd = res.periodic_dlpno
        assert pd.n_cells == 1 and pd.source_kind == "gamma-gdf" and pd.same_source
        assert res.correlation.backend == "periodic-gdf-dlpno-ccsd(t)"
        assert res.energy_total == pytest.approx(pd.e_total_per_cell)
        assert res.energy == pytest.approx(pd.e_hf_per_cell)
        assert -0.03 < pd.e_corr_per_cell < -0.01
        # The library route on the same cell is the same number.
        lib = run_periodic_dlpno_ccsd(system, "def2-svp", progress=False)
        assert pd.e_corr_per_cell == pytest.approx(lib.e_corr_per_cell, abs=1e-8)

        out = (tmp_path / "h2g.out").read_text()
        assert "Job: PERIODIC DLPNO-CCSD(T) on RHF/GDF" in out
        assert "DLPNO-CCSD(T) local PNO-space (M3c) on the periodic GDF reference" in out
        assert "fit_source=classic" in out and "(same source)" in out
        assert "correlation source             = gamma-gdf" in out
        assert "E(CCSD correlation) / cell" in out
        assert "E(DLPNO-CCSD(T) total) / cell" in out
        assert "AICCM correlation" not in out

        run = tomllib.loads((tmp_path / "h2g.system").read_text())["run"]
        assert run["correlation_method"] == "dlpno-ccsd(t)"
        assert run["correlation_route"] == "periodic-gdf-dlpno-ccsd(t)"
        assert run["dlpno_source_kind"] == "gamma-gdf"
        assert run["dlpno_source_identity"] == pd.source_identity
        assert run["dlpno_kmesh"] == [1, 1, 1] and run["dlpno_n_cells"] == 1
        assert run["dlpno_same_source"] is True
        assert run["dlpno_exchange_q0"] == str(res.hf_result.exchange_q0) if hasattr(res, "hf_result") else True
        assert run["dlpno_e_corr_per_cell"] == pytest.approx(pd.e_corr_per_cell)
        assert run["gdf_engine"] == "k-mesh"
        assert "aiccm_correlation_route" not in run

        keys = {row["key"] for row in tomllib.loads((tmp_path / "h2g.system").read_text())["citations"]["entries"]}
        assert {"riplinger_dlpno_2013", "riplinger_triples_2013", "raghavachari_ccsdt_1989",
                "pipek_mezey_1989", "sun_berkelbach_gdf_2017"} <= keys
        assert "foster_boys_1960" not in keys
        bib = (tmp_path / "h2g.bibtex").read_text()
        assert "riplinger_dlpno_2013" in bib and "pipek_mezey_1989" in bib

        events = [json.loads(line) for line in (tmp_path / "h2g.scf.jsonl").read_text().splitlines() if line.strip()]
        done = [e for e in events if e.get("event") == "dlpno_ccsd_done"]
        assert len(done) == 1
        assert done[0]["source_kind"] == "gamma-gdf" and done[0]["n_cells"] == 1
        assert done[0]["e_corr_per_cell"] == pytest.approx(pd.e_corr_per_cell)

        with zipfile.ZipFile(tmp_path / "h2g.qvf") as z:
            manifest = json.loads(z.read("manifest.json"))
            assert "x_vibeqc" in manifest["extensions"]
            section = [s for s in manifest["sections"] if s.get("kind") == "x_vibeqc.dlpno_correlation"]
            assert len(section) == 1
            member = section[0]["members"]["dlpno_correlation"]["path"]
            payload = json.loads(z.read(member))
            assert payload["source_identity"] == pd.source_identity
            assert payload["e_corr_per_cell"] == pytest.approx(pd.e_corr_per_cell)
            assert manifest["provenance"]["run_fields"]["correlation_route"] == "periodic-gdf-dlpno-ccsd(t)"

    def test_mesh_front_door_folds_to_the_supercell(self, tmp_path, monkeypatch, chain):
        monkeypatch.chdir(tmp_path)
        basis = vq.BasisSet(chain.unit_cell_molecule(), "sto-3g")
        res = vq.run_periodic_job(
            chain, basis, method="dlpno-ccsd", kpoints=(1, 1, 2), output="h2m", progress=False,
        )
        pd = res.periodic_dlpno
        assert pd.n_cells == 2 and pd.source_kind == "bvk-supercell" and pd.same_source
        assert res.correlation.backend == "periodic-gdf-dlpno-ccsd"
        assert pd.e_t is None and res.dlpno_ccsd.e_t == 0.0
        assert res.energy_total == pytest.approx(pd.e_hf_per_cell + pd.e_corr_per_cell)
        lib = run_periodic_dlpno_ccsd(chain, "sto-3g", kmesh=(1, 1, 2), with_triples=False, progress=False)
        assert pd.e_corr_per_cell == pytest.approx(lib.e_corr_per_cell, abs=1e-8)
        out = (tmp_path / "h2m.out").read_text()
        assert "folded to the real BvK supercell" in out
        assert "supercell E(corr) / E(total)" in out
        assert "E(DLPNO-CCSD total) / cell" in out and "E((T) correction)" not in out
        run = tomllib.loads((tmp_path / "h2m.system").read_text())["run"]
        assert run["dlpno_kmesh"] == [1, 1, 2] and run["dlpno_n_cells"] == 2
        assert run["dlpno_source_kind"] == "bvk-supercell"
        assert run["dlpno_e_t_per_cell"] == ""
        keys = {row["key"] for row in tomllib.loads((tmp_path / "h2m.system").read_text())["citations"]["entries"]}
        assert "riplinger_dlpno_2013" in keys and "raghavachari_ccsdt_1989" not in keys

    @pytest.mark.parametrize(
        "kwargs, error, reason",
        [
            ({"functional": "pbe"}, ValueError, "Hartree-Fock reference"),
            ({"jk_method": "bipole"}, NotImplementedError, "decision 2"),
            ({"symmetry_reduce_k": True}, NotImplementedError, "irreducible"),
            ({"smearing_temperature": 0.01}, NotImplementedError, "integer-aufbau"),
            ({"optimize": True}, NotImplementedError, "gradients"),
            ({"fit_source": "mdf"}, ValueError, "fit_source='classic'"),
            ({"dlpno_ccsd_options": object()}, TypeError, "LocalCCSDOptions"),
        ],
    )
    def test_requests_the_front_door_cannot_run_are_refused_before_the_scf(
        self, tmp_path, monkeypatch, kwargs, error, reason
    ):
        monkeypatch.chdir(tmp_path)
        system = _h2_cube()
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        with pytest.raises(error, match=reason):
            vq.run_periodic_job(system, basis, method="dlpno-ccsd(t)", output="refused", progress=False, **kwargs)
        assert not (tmp_path / "refused.out").exists() or "DLPNO" not in (tmp_path / "refused.out").read_text()

    def test_kpoints_objects_and_low_dimensions_are_refused(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        a = 6.0
        system = _h2_cube(a)
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        with pytest.raises(ValueError, match="mesh tuple"):
            vq.run_periodic_job(
                system, basis, method="dlpno-ccsd(t)", output="kp", progress=False,
                kpoints=KPoints.monkhorst_pack(system, (1, 1, 2)),
            )
        wire = vq.PeriodicSystem(
            1, np.diag([12.0, 12.0, 6.0]),
            [vq.Atom(1, [6.0, 6.0, 2.3]), vq.Atom(1, [6.0, 6.0, 3.7])],
        )
        with pytest.raises(NotImplementedError, match="dim=3"):
            vq.run_periodic_job(
                wire, vq.BasisSet(wire.unit_cell_molecule(), "sto-3g"),
                method="dlpno-ccsd(t)", output="wire", progress=False,
            )
        with pytest.raises(ValueError, match="require method='dlpno-ccsd'"):
            vq.run_periodic_job(system, basis, method="RHF", output="rhf", progress=False, frozen_core=0)


def _h3_doublet():
    """A neutral three-electron cell with distinct alpha and beta orbitals."""
    return vq.PeriodicSystem(
        3, np.diag([8.0, 8.0, 8.0]),
        [vq.Atom(1, [3.3, 4.0, 4.0]), vq.Atom(1, [4.7, 4.0, 4.0]),
         vq.Atom(1, [4.0, 6.0, 4.0])], multiplicity=2,
    )


@pytest.fixture(scope="module", params=[(1, 1, 1), (1, 1, 2)])
def open_reference(request):
    system = _h3_doublet()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    mesh = request.param
    hf = run_kuhf_periodic_gdf(
        system, basis, cyclic_gamma_mesh(system, mesh), fit_source="classic",
        retain_fit_cache=True, progress=False,
    )
    assert hf.converged
    return system, basis, hf, mesh


class TestPeriodicOpenShell:
    @pytest.mark.parametrize("n_frozen_cell", [0, 1])
    def test_full_domains_match_canonical_uccsd_on_both_spins(self, open_reference, n_frozen_cell):
        system, basis, hf, mesh = open_reference
        settings = {**FULL, "n_frozen": n_frozen_cell}
        result = correlate_periodic_reference(
            system, basis, hf, kmesh=mesh, localise="none",
            options=LocalUCCSDOptions(**settings), with_triples=True,
        )
        source, ref = result.source, result.source.reference
        n = int(np.prod(mesh))
        nf = n * n_frozen_cell
        assert source.reference_kind == result.reference_kind == "uhf"
        assert source.molecule.multiplicity == n + 1
        assert (result.n_occ_alpha, result.n_occ_beta) == (2 * n, n)
        assert result.n_occ == 3 * n and result.n_vir == 3 * n
        assert result.cc_result.n_frozen == nf
        assert ref.energy == pytest.approx(n * hf.energy)
        assert result.same_source and result.exchange_q0 == str(hf.exchange_q0)
        ca, cb = ref.mo_coeffs_alpha[:, nf:], ref.mo_coeffs_beta[:, nf:]
        canonical = run_ref_uccsd(
            ca.T @ ref.fock_alpha @ ca, cb.T @ ref.fock_beta @ cb,
            source.factors.mo_transform(ca), source.factors.mo_transform(cb),
            2 * n - nf, n - nf, e_hf=ref.energy, compute_triples=True,
        )
        assert result.converged and canonical.converged
        assert result.e_corr == pytest.approx(canonical.e_corr, abs=1e-8)
        assert result.e_t == pytest.approx(canonical.e_t, abs=1e-8)
        assert result.e_total_per_cell == pytest.approx(result.e_total / n)
        assert result.cc_result.source_identity == source.identity

    def test_localisation_preserves_each_spin_occupied_projector(self, open_reference):
        system, basis, hf, mesh = open_reference
        n = int(np.prod(mesh))
        make_source = gamma_gdf_source if n == 1 else bvk_supercell_source
        kwargs = {} if n == 1 else {"mesh": mesh}
        canonical = make_source(system, basis, hf, localise="none", **kwargs)
        localised = make_source(system, basis, hf, **kwargs)
        S = canonical.overlap
        for spin, n_occ in (("alpha", 2 * n), ("beta", n)):
            c = getattr(canonical.reference, "mo_coeffs_" + spin)
            l = getattr(localised.reference, "mo_coeffs_" + spin)
            assert np.allclose(c[:, :n_occ] @ c[:, :n_occ].T,
                               l[:, :n_occ] @ l[:, :n_occ].T, atol=1e-9)
            assert np.allclose(l.T @ S @ l, np.eye(len(S)), atol=1e-8)
            assert np.array_equal(l[:, n_occ:], c[:, n_occ:])

    def test_gamma_complex_phases_preserve_spin_densities_and_coulomb(self, open_reference):
        system, basis, hf, mesh = open_reference
        if mesh != (1, 1, 1):
            return
        phased = copy(hf)
        phased.mo_coeffs_alpha = [hf.mo_coeffs_alpha[0] * np.exp(1j * np.array([0.3, -1.2, 2.0]))]
        phased.mo_coeffs_beta = [hf.mo_coeffs_beta[0] * np.exp(1j * np.array([-0.8, 2.1, 0.5]))]
        source = gamma_gdf_source(system, basis, phased, localise="none")
        D = np.zeros_like(source.overlap)
        for spin, n_occ in (("alpha", 2), ("beta", 1)):
            c = getattr(source.reference, "mo_coeffs_" + spin)[:, :n_occ]
            d = c @ c.T
            assert np.allclose(d, getattr(hf, "density_" + spin)[0], atol=1e-7)
            D += d
        rho = np.einsum("Pmn,mn->P", source.factors.B, D)
        assert 0.5 * rho @ rho == pytest.approx(hf.e_coulomb, abs=1e-6)
        assert not np.allclose(source.reference.fock_alpha, source.reference.fock_beta)

    @pytest.mark.parametrize("bad_field, reason", [
        ("occupations_beta", "integer aufbau"),
        ("mo_coeffs_beta", "all .* k-points"),
        ("kpoint_weights", "uniform"),
        ("functional", "Hartree-Fock"),
        ("orbital_rank_beta", "full-rank"),
        ("mo_energies_beta", "orbital energies"),
    ])
    def test_invalid_spin_reference_is_refused(self, open_reference, bad_field, reason):
        system, basis, hf, mesh = open_reference
        bad = copy(hf)
        bad.occupations_beta = [np.array(o, copy=True) for o in hf.occupations_beta]
        if bad_field == "occupations_beta":
            bad.occupations_beta[0][0] = 0.5
        elif bad_field == "mo_coeffs_beta":
            bad.mo_coeffs_beta = []
        elif bad_field == "kpoint_weights":
            bad.kpoint_weights = np.full(int(np.prod(mesh)), 0.2)
        elif bad_field == "orbital_rank_beta":
            bad.mo_coeffs_beta = [c[:, :-1] for c in hf.mo_coeffs_beta]
            bad.mo_energies_beta = [e[:-1] for e in hf.mo_energies_beta]
            bad.occupations_beta = [o[:-1] for o in hf.occupations_beta]
        elif bad_field == "mo_energies_beta":
            bad.mo_energies_beta = [e + 0.1 for e in hf.mo_energies_beta]
        else:
            bad.functional = "pbe"
        make_source = gamma_gdf_source if mesh == (1, 1, 1) else bvk_supercell_source
        kwargs = {} if mesh == (1, 1, 1) else {"mesh": mesh}
        with pytest.raises(CorrelationSourceError, match=reason):
            make_source(system, basis, bad, **kwargs)

    def test_options_follow_spin_and_requested_triples_without_mutation(self):
        requested = LocalUCCSDOptions(compute_triples=True)
        prepared = prepare_periodic_cc_options(requested, reference_kind="uhf", with_triples=False)
        assert prepared.localise == "none" and not prepared.compute_triples
        assert requested.localise == "boys" and requested.compute_triples
        with pytest.raises(TypeError, match="LocalUCCSDOptions"):
            run_periodic_dlpno_ccsd(_h3_doublet(), "sto-3g", cc_options=LocalCCSDOptions())

    def test_fold_matches_explicit_gamma_supercell(self, open_reference):
        system, basis, hf, mesh = open_reference
        if mesh == (1, 1, 1):
            return
        from vibeqc.periodic.chi.posthf import _build_supercell_system

        folded = correlate_periodic_reference(
            system, basis, hf, kmesh=mesh, options=LocalUCCSDOptions(**FULL),
            localise="none", with_triples=True,
        )
        explicit = run_periodic_dlpno_ccsd(
            _build_supercell_system(system, mesh, multiplicity=3), "sto-3g",
            cc_options=LocalUCCSDOptions(**FULL), localise="none", progress=False,
        )
        assert folded.converged and explicit.converged
        assert folded.e_hf == pytest.approx(explicit.e_hf, abs=1e-6)
        assert folded.e_corr == pytest.approx(explicit.e_corr, abs=1e-6)
        assert folded.e_t == pytest.approx(explicit.e_t, abs=1e-7)

    def test_empty_beta_occupied_space(self):
        system = vq.PeriodicSystem(
            3, np.diag([8.0] * 3), [vq.Atom(1, [4.0] * 3)], multiplicity=2,
        )
        result = run_periodic_dlpno_ccsd(system, "sto-3g", progress=False)
        assert result.converged and result.reference_kind == "uhf"
        assert (result.n_occ_alpha, result.n_occ_beta) == (1, 0)
        assert result.e_corr == result.e_t == 0.0

    def test_explicit_uhf_singlet_matches_closed_shell_limit(self):
        system = _h2_cube()
        closed = run_periodic_dlpno_ccsd(
            system, "sto-3g", cc_options=LocalCCSDOptions(**FULL),
            localise="none", with_triples=False, progress=False,
        )
        opened = run_periodic_dlpno_ccsd(
            system, "sto-3g", cc_options=LocalUCCSDOptions(**FULL),
            localise="none", with_triples=False, progress=False,
        )
        assert opened.reference_kind == "uhf" and opened.converged
        assert (opened.n_occ_alpha, opened.n_occ_beta) == (1, 1)
        assert opened.e_hf == pytest.approx(closed.e_hf, abs=1e-8)
        assert opened.e_corr == pytest.approx(closed.e_corr, abs=1e-8)
        assert opened.e_t is None

    def test_overlapping_occupied_and_virtual_bands_are_refused(self, open_reference):
        system, basis, hf, mesh = open_reference
        if mesh == (1, 1, 1):
            return
        bad = copy(hf)
        # Every k-point is individually sorted and integer occupied, but
        # alpha's occupied band at k=1 lies above a virtual band at k=0.
        bad.mo_energies_alpha = [np.array([-3., -2., -1.]), np.array([0., 1., 2.])]
        with pytest.raises(CorrelationSourceError, match="no global gap"):
            bvk_supercell_source(system, basis, bad, mesh=mesh)

    def test_frozen_active_band_overlap_is_refused(self, open_reference):
        system, basis, hf, mesh = open_reference
        bad = copy(hf)
        if mesh == (1, 1, 1):
            # A degenerate occupied block must not be realified across a
            # boundary that freezes only part of it.
            bad.mo_energies_alpha = [np.array([-2., -2., 1.])]
            make_source, kwargs = gamma_gdf_source, {"n_frozen": 1}
        else:
            # Global sorting would freeze both k=0 orbitals, replacing the
            # requested first occupied band of each k-point.
            bad.mo_energies_alpha = [np.array([-4., -3., 10.]), np.array([-2., -1., 11.])]
            make_source, kwargs = bvk_supercell_source, {"mesh": mesh, "n_frozen_per_cell": 1}
        with pytest.raises(CorrelationSourceError, match="frozen and active occupied bands"):
            make_source(system, basis, bad, **kwargs)

    def test_open_shell_front_door_records_spin_on_every_artifact(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        system = _h3_doublet()
        basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
        res = vq.run_periodic_job(
            system, basis, method="dlpno-ccsd(t)", kpoints=(1, 1, 2),
            dlpno_ccsd_options=LocalUCCSDOptions(**FULL), frozen_core=0,
            output="open", structured_log=True, progress=False,
        )
        pd = res.periodic_dlpno
        assert pd.converged and pd.reference_kind == "uhf"
        assert pd.n_occ_alpha == 4 and pd.n_occ_beta == 2
        assert pd.source.molecule.multiplicity == 3
        assert res.correlation.backend == "periodic-gdf-dlpno-uccsd(t)"
        assert res.energy_total == pytest.approx(pd.e_total_per_cell)
        out = (tmp_path / "open.out").read_text()
        assert "on UHF/GDF" in out and "KUHF/GDF, fit_source=classic" in out
        assert "occupied alpha / beta (total)" in out
        manifest = tomllib.loads((tmp_path / "open.system").read_text())
        run = manifest["run"]
        assert run["dlpno_reference_kind"] == "uhf"
        assert (run["dlpno_n_occ_alpha"], run["dlpno_n_occ_beta"]) == (4, 2)
        assert run["correlation_route"] == res.correlation.backend
        keys = {row["key"] for row in manifest["citations"]["entries"]}
        assert {"saitow_openshell_dlpno_2017", "pipek_mezey_1989",
                "ye_berkelbach_periodic_lno_2024", "guo_openshell_dlpno_triples_2020",
                "watts_gauss_bartlett_openshell_triples_1993"} <= keys
        assert "foster_boys_1960" not in keys
        events = [json.loads(line) for line in (tmp_path / "open.scf.jsonl").read_text().splitlines() if line.strip()]
        done = next(e for e in events if e.get("event") == "dlpno_ccsd_done")
        assert done["reference_kind"] == "uhf" and done["n_occ_beta"] == 2
        with zipfile.ZipFile(tmp_path / "open.qvf") as archive:
            qvf = json.loads(archive.read("manifest.json"))
            section = next(s for s in qvf["sections"] if s.get("kind") == "x_vibeqc.dlpno_correlation")
            payload = json.loads(archive.read(section["members"]["dlpno_correlation"]["path"]))
            assert payload["reference_kind"] == "uhf" and payload["n_occ_alpha"] == 4
            assert payload["source_identity"] == pd.source_identity

