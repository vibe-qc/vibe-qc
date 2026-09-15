"""EXPERIMENTAL: per-unit-cell adapter for the real-Γ direct route (2d).

Gates for :mod:`vibeqc.periodic.ccm.real_gamma_runner` — the runner-contract
adapter that backs ``jk_method='real-gamma'`` in ``run_periodic_job``.
The runner wiring landed with 2d step 2 (6ec349703, 2026-07-26):
``validate_jk_method`` accepts the route, and the tests below drive the
route through ``run_periodic_job`` itself (manifest + QVF records, parity
with the API driver). What still fails closed on the runner surface is
pinned in ``test_runner_real_gamma_fail_closed_surface``: variable-cell
relaxation, ``hessian``, DFT+U, dim<3, non-SCF methods. Analytic forces and
fixed-lattice relaxation landed at the API level in milestone 2c
(e2eabc769 2026-08-19, 4a3edc058 2026-08-20: ``run_ccm_direct_gradient`` /
``run_ccm_direct_optimize``) and ``optimize=True`` was wired through the
runner on 2026-08-23 -- see ``test_runner_real_gamma_optimize``, which
also pins that the relaxed geometry lands in the separate ``.opt.xyz``
artefact rather than overwriting the SCF ``.xyz``. Contract inventory +
history: ``HANDOVER_AICCM_DIRECT_TORUS.md`` § Milestone 2d and
§ Milestone 2c LANDED.
"""

from __future__ import annotations

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from vibeqc import Atom, PeriodicSystem
from vibeqc.periodic.ccm import CCMSystem
from vibeqc.periodic.ccm.direct import run_ccm_rhf_direct, run_ccm_rks_direct
from vibeqc.periodic.ccm.real_gamma_runner import (
    CCMRealGammaResult,
    run_real_gamma_scf,
)

pytestmark = pytest.mark.experimental  # neutral finite-BvK-torus research lane


def _h2_cell():
    return PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.3]), Atom(1, [3.0, 3.0, 3.7])], 0, 1)


def _li_doublet():
    return PeriodicSystem(
        3, np.diag([7.0, 7.0, 7.0]), [Atom(3, [3.5, 3.5, 3.5])], 0, 2)


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _assert_ks_operator_contract(result, *, unrestricted):
    """Pin the adapter's unscaled supercell-AO operator convention."""
    overlap = np.asarray(result.overlap)
    n_supercell_ao = overlap.shape[0]
    assert overlap.shape == (n_supercell_ao, n_supercell_ao)
    assert result.fock is not None
    assert result.hcore is not None
    assert np.asarray(result.fock).shape == overlap.shape
    assert np.asarray(result.hcore).shape == overlap.shape
    np.testing.assert_allclose(result.fock, result.ccm_result.fock, atol=0.0)
    np.testing.assert_allclose(result.hcore, result.ccm_result.hcore, atol=0.0)
    np.testing.assert_allclose(result.hcore, result.hcore.T, atol=1e-13)

    alpha_residual = (
        result.fock @ result.mo_coeffs
        - (overlap @ result.mo_coeffs) * result.mo_energies[None, :]
    )
    assert np.max(np.abs(alpha_residual)) < 5e-9
    if unrestricted:
        assert result.fock_beta is not None
        assert np.asarray(result.fock_beta).shape == overlap.shape
        np.testing.assert_allclose(
            result.fock_beta, result.ccm_result.fock_beta, atol=0.0
        )
        beta_residual = (
            result.fock_beta @ result.mo_coeffs_beta
            - (overlap @ result.mo_coeffs_beta)
            * result.mo_energies_beta[None, :]
        )
        assert np.max(np.abs(beta_residual)) < 5e-9
    else:
        assert result.fock_beta is None


def test_adapter_rhf_parity_and_contract():
    """Adapter energy == API driver energy / N_c exactly; required contract
    fields present; convention record carries the 2d field set."""
    cell = _h2_cell()
    r = _quiet(run_real_gamma_scf, cell, "sto-3g", "RHF", (2, 1, 1))
    a = _quiet(run_ccm_rhf_direct, CCMSystem(cell, (2, 1, 1), "sto-3g"))
    assert r.guess_selection == r.ccm_result.guess_selection
    assert r.guess_selection.requested.name == "AUTO"
    assert r.guess_selection.effective.name == "HCORE"
    assert isinstance(r, CCMRealGammaResult)
    assert r.converged
    assert r.energy == pytest.approx(a.energy / 2.0, abs=1e-14)
    assert r.backend == "ccm-direct-real-gamma"
    assert r.effective_n_electrons == 4
    c = r.real_gamma
    assert c.route == "real-gamma"
    assert c.exchange_q0 == "BvK-ewald"
    assert c.exchange_q0_applicability == "active"
    assert c.lattice_vector_convention == "columns"
    assert c.nrep == (2, 1, 1) and c.n_cells == 2
    # D86 executed values: the direct loops implement no mixing/damping/
    # level shift -- structural zeros, recorded not inferred.
    assert c.executed_fock_mixing == 0.0
    assert c.executed_damping == 0.0
    assert c.executed_level_shift == 0.0
    assert c.experimental is True
    assert c.screened_exchange is None
    # Folded density is a per-unit-cell LatticeMatrixSet.
    assert len(r.density.cells) > 0
    n_mu = int(np.asarray(r.density.blocks[0]).shape[0])
    assert r.mo_coeffs.shape[0] == c.n_cells * n_mu


def test_adapter_hse06_screened_record():
    """HSE-class runs record the SR-direct assembly + inactive seam."""
    r = _quiet(run_real_gamma_scf, _h2_cell(), "sto-3g", "RKS", (2, 1, 1),
               functional="hse06")
    a = _quiet(run_ccm_rks_direct, CCMSystem(_h2_cell(), (2, 1, 1), "sto-3g"),
               "hse06")
    assert r.energy == pytest.approx(a.energy / 2.0, abs=1e-14)
    c = r.real_gamma
    assert c.exchange_q0_applicability == "inactive"
    assert c.screened_exchange == (0.0, 0.25, 0.11)
    assert r.functional == "hse06"
    assert r.e_hf_exchange is not None
    _assert_ks_operator_contract(r, unrestricted=False)


def test_adapter_open_shell_fields():
    """UHF/UKS expose per-spin MOs and folded per-spin densities; the UHF
    result's missing total-density/plain-MO fields fall back correctly."""
    for method, functional in (("UHF", None), ("UKS", "pbe")):
        r = _quiet(run_real_gamma_scf, _li_doublet(), "sto-3g", method,
                   (1, 1, 1), functional=functional)
        assert r.converged
        assert r.mo_energies_beta is not None
        assert r.mo_coeffs_beta is not None
        assert r.density_alpha is not None and r.density_beta is not None
        if method == "UKS":
            _assert_ks_operator_contract(r, unrestricted=True)
        # total = alpha + beta on every folded block
        for da, db, dt in zip(r.density_alpha.blocks, r.density_beta.blocks,
                              r.density.blocks):
            np.testing.assert_allclose(
                np.asarray(da) + np.asarray(db), np.asarray(dt), atol=1e-13)


def test_adapter_forwards_periodic_xc_grid_controls(monkeypatch):
    """KS adapter preserves the negotiated grid and one image-radius value."""
    from vibeqc import GridOptions, LatticeSumOptions
    from vibeqc.periodic.ccm import direct as direct_module

    captured = {}

    def fake_rks(ccm, functional, **kwargs):
        captured.update(functional=functional, **kwargs)
        nbf = int(ccm.basis.nbasis)
        identity = np.eye(nbf)
        return SimpleNamespace(
            converged=True,
            n_iter=1,
            energy=-2.0,
            e_xc=-0.2,
            e_hf_exchange=0.0,
            mo_energies=np.zeros(nbf),
            mo_coeffs=identity,
            density=0.5 * identity,
            overlap=identity,
            fock=identity,
            hcore=identity,
            exchange_q0="BvK-ewald",
            exchange_q0_applicability="inactive",
        )

    monkeypatch.setattr(direct_module, "run_ccm_rks_direct", fake_rks)
    grid = GridOptions()
    grid.atomic_grid_profile = "pyscf-level3"
    lattice = LatticeSumOptions()
    lattice.cutoff_bohr = 7.0
    lattice.becke_image_radius_bohr = 1.0

    result = run_real_gamma_scf(
        _h2_cell(),
        "sto-3g",
        "RKS",
        (1, 1, 1),
        functional="pbe",
        grid_options=grid,
        becke_image_radius_bohr=7.25,
        lat_opts=lattice,
    )

    assert result.converged
    assert captured["functional"] == "pbe"
    assert captured["grid_options"] is grid
    assert captured["becke_image_radius_bohr"] == pytest.approx(7.25)
    assert captured["lat_opts"] is lattice


def test_adapter_boundary():
    """Bad mesh and unknown methods fail loudly; the runner surface accepts
    real-gamma since the 2d step-2 wiring (the historical fail-closed pin
    flipped 2026-07-26), while the Γ-CCM construction selector and the
    rejected A-prefixed control spelling keep failing closed."""
    with pytest.raises(ValueError, match="mesh"):
        _quiet(run_real_gamma_scf, _h2_cell(), "sto-3g", "RHF", (0, 1, 1))
    with pytest.raises(NotImplementedError, match="real-gamma"):
        _quiet(run_real_gamma_scf, _h2_cell(), "sto-3g", "MP2", (1, 1, 1))
    from vibeqc.periodic_jk_method import (
        resolve_jk_method_string,
        validate_jk_method,
    )

    lat = np.asarray(_h2_cell().lattice, dtype=float)
    with pytest.warns(DeprecationWarning, match="variant='real-gamma'"):
        jk = resolve_jk_method_string("real-gamma")  # legacy spelling (M1)
    validate_jk_method(jk, lattice=lat, basis_name="sto-3g")  # no raise
    with pytest.raises(ValueError, match="ruling R1"):
        resolve_jk_method_string("gamma-ccm")  # D-2: retired bare spelling
    with pytest.warns(DeprecationWarning, match="variant='four-center'"):
        a = resolve_jk_method_string("aiccm2026dev-a")
    # M3a: the four-centre construction is wired, so validation accepts it;
    # the arm's own envelope is pinned in
    # tests/test_periodic_ccm_four_center_adapter.py.
    validate_jk_method(a, lattice=lat, basis_name="sto-3g")  # no raise
    with pytest.raises(ValueError, match="rejected"):
        resolve_jk_method_string("aiccm2026dev-a-real-gamma")


# --- Runner-level gates (2d step 2, 2026-07-26) ------------------------------


def _run_job(tmp_path, method, functional=None, **kw):
    from vibeqc import run_periodic_job
    from vibeqc._vibeqc_core import BasisSet

    cell = kw.pop("cell", None) or _h2_cell()
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    kw.setdefault("initial_guess", "HCORE")
    return _quiet(
        run_periodic_job, cell, basis, method=method, functional=functional,
        jk_method="real-gamma", kpoints=kw.pop("kpoints", (2, 1, 1)),
        output=str(tmp_path / "rg"), **kw)


def test_runner_real_gamma_rhf_parity_and_manifest(tmp_path):
    """run_periodic_job(jk_method='real-gamma') == the API driver / N_c, and
    the .system manifest carries the full 2d convention record with
    RESULT-derived values."""
    r = _run_job(tmp_path, "RHF")
    a = _quiet(run_ccm_rhf_direct, CCMSystem(_h2_cell(), (2, 1, 1), "sto-3g"))
    assert r.converged
    assert r.energy == pytest.approx(a.energy / 2.0, abs=1e-13)
    manifest = (tmp_path / "rg.system").read_text(encoding="utf-8")
    for needle in (
        'exchange_q0    = "BvK-ewald"',
        'exchange_q0_applicability = "active"',
        'lattice_vector_convention = "columns"',
        "executed_fock_mixing = 0.0",
        "executed_damping = 0.0",
        "executed_level_shift = 0.0",
        "bvk_n_cells    = 2",
        'cderi_build    = "fold"',
        "experimental_route = true",
        'method_status  = "experimental"',
        'jk_method_executed = "real-gamma"',
    ):
        assert needle in manifest, needle


def test_runner_real_gamma_hse06_qvf_convention(tmp_path):
    """Screened hybrid through the runner: seam applicability 'inactive'
    (result-derived, NOT the requested exxdiv label), the SR-direct assembly
    recorded in the manifest, and the x_vibeqc.real_gamma_convention QVF
    vendor section present with the full record."""
    import json
    import zipfile

    r = _run_job(tmp_path, "RKS", functional="hse06")
    a = _quiet(run_ccm_rks_direct, CCMSystem(_h2_cell(), (2, 1, 1), "sto-3g"),
               "hse06")
    assert r.energy == pytest.approx(a.energy / 2.0, abs=1e-13)
    manifest = (tmp_path / "rg.system").read_text(encoding="utf-8")
    assert 'exchange_q0_applicability = "inactive"' in manifest
    assert "sr-direct (erfc-attenuated fitted kernel" in manifest
    assert "exact_exchange_c_sr = 0.25" in manifest

    z = zipfile.ZipFile(tmp_path / "rg.qvf")
    vendor = [n for n in z.namelist() if "x_vibeqc" in n]
    assert vendor, z.namelist()
    payload = json.loads(z.read(vendor[0]))
    blob = json.dumps(payload)
    assert "real_gamma_convention" in blob
    assert '"exchange_q0_applicability": "inactive"' in blob
    assert '"lattice_vector_convention": "columns"' in blob


def test_runner_real_gamma_optimize(tmp_path):
    """Runner-level geometry relaxation on the real-Γ surface (wired
    2026-08-23). The relaxation runs through run_ccm_direct_optimize, so
    the direct-vs-control parity premise is verified at both endpoints and
    the residuals are written into the .out. Measured on the asymmetric
    compact H₂ (2,1,1): 1.6000 -> 1.4736 bohr, E = -1.195551311,
    endpoint parities 1.3e-13 -- identical to the API driver.

    The SCF geometry stays in ``{stem}.xyz`` and the relaxed one is a
    SEPARATE ``{stem}.opt.xyz`` artefact (the runner's existing
    convention); both are asserted so a future change that silently
    overwrites one is caught."""
    from vibeqc import run_periodic_job
    from vibeqc._vibeqc_core import BasisSet

    cell = PeriodicSystem(
        3, np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [3.0, 3.0, 2.2]), Atom(1, [3.0, 3.0, 3.8])], 0, 1)
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    r = _quiet(run_periodic_job, cell, basis, method="RHF",
               jk_method="real-gamma", kpoints=(2, 1, 1), optimize=True,
               initial_guess="HCORE", output=str(tmp_path / "rg"))
    assert r.converged

    bohr = 1.8897261254578281

    def _bond(name):
        lines = (tmp_path / name).read_text(encoding="utf-8").splitlines()
        z = [float(lines[i].split()[3]) for i in (2, 3)]
        return abs(z[1] - z[0]) * bohr

    assert _bond("rg.xyz") == pytest.approx(1.60, abs=1e-3)      # SCF geom
    assert _bond("rg.opt.xyz") == pytest.approx(1.4736, abs=5e-3)  # relaxed
    out = (tmp_path / "rg.out").read_text(encoding="utf-8")
    assert "real-Γ direct-torus, parity-verified" in out
    assert "parity (initial)" in out and "parity (final)" in out
    # M0 (handovers/HANDOVER_AICCM_STANDARD_METHOD.md): the relaxation runs on
    # run_ccm_direct_gradient, so the citation surface owes Pulay 1969 and
    # Hellmann-Feynman via uses_gradient, next to the CCM lineage pair.
    from vibeqc.output.citations.registry import load_default_database

    entries = load_default_database().entries()
    surface = "\n".join(
        (tmp_path / f"rg{ext}").read_text(encoding="utf-8", errors="replace")
        for ext in (".out", ".references", ".bibtex")
        if (tmp_path / f"rg{ext}").is_file()
    )
    for key in ("pulay_forces_1969", "feynman_forces_1939",
                "bredow_geudtner_jug_ccm_2001", "peintinger_ccm_2014"):
        assert entries[key].bibtex_key in surface, key


def test_runner_real_gamma_fail_closed_surface(tmp_path):
    """The runner's real-gamma boundary: optimize/hessian/DFT+U off, dim<3
    rejected, non-SCF methods rejected, non-Γ-centred meshes rejected."""
    from vibeqc import run_periodic_job
    from vibeqc._vibeqc_core import BasisSet

    cell = _h2_cell()
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")

    def job(**kw):
        kw.setdefault("initial_guess", "HCORE")
        return _quiet(
            run_periodic_job, kw.pop("cell", cell), kw.pop("basis", basis),
            jk_method="real-gamma", output=str(tmp_path / "x"), **kw)

    # optimize=True is WIRED since 2026-08-23 (test_runner_real_gamma_optimize),
    # so IID 199's pin on the refusal TEXT is retired with the refusal it
    # pinned. What stays closed is variable-cell: the route is fixed-lattice
    # and has no analytic stress. The refusal must say so rather than claim
    # relaxation is unavailable in general.
    with pytest.raises(NotImplementedError, match="variable-cell") as exc:
        job(method="RHF", kpoints=(2, 1, 1), optimize=True,
            optimize_cell=True)
    msg = str(exc.value)
    assert "fixed-lattice" in msg, msg
    assert "analytic stress" in msg, msg
    with pytest.raises(NotImplementedError, match="force constants"):
        job(method="RHF", kpoints=(2, 1, 1), hessian=True)
    # Non-SCF and restricted-open-shell methods die on the runner's generic
    # guards before this route's own method boundary (which remains as
    # defense-in-depth in the validation block).
    with pytest.raises(NotImplementedError, match="not supported"):
        job(method="MP2", kpoints=(2, 1, 1))
    with pytest.raises(NotImplementedError, match="ROHF"):
        job(method="ROHF", kpoints=(2, 1, 1))
    chain = PeriodicSystem(
        1, np.diag([6.0, 15.0, 15.0]),
        [Atom(1, [0.0, 7.5, 7.5]), Atom(1, [1.4, 7.5, 7.5])], 0, 1)
    chain_basis = BasisSet(chain.unit_cell_molecule(), "sto-3g")
    with pytest.raises(NotImplementedError, match="dim"):
        job(cell=chain, basis=chain_basis, method="RHF", kpoints=(2, 1, 1))
    charged = PeriodicSystem(
        3,
        np.diag([6.0, 6.0, 6.0]),
        [Atom(1, [0.0, 0.0, 0.0]), Atom(1, [0.0, 0.0, 1.4])],
        2,
        1,
    )
    charged_basis = BasisSet(charged.unit_cell_molecule(), "sto-3g")
    with pytest.raises(NotImplementedError, match="neutral fitted-torus"):
        job(
            cell=charged,
            basis=charged_basis,
            method="RHF",
            kpoints=(1, 1, 1),
        )
    with pytest.raises(NotImplementedError, match="charged cells"):
        _quiet(
            run_real_gamma_scf,
            charged,
            "sto-3g",
            "RHF",
            (1, 1, 1),
        )


def test_runner_front_door_real_gamma_matches_legacy_selector(tmp_path):
    """M1 (handovers/HANDOVER_AICCM_STANDARD_METHOD.md): the front door
    ``method="aiccm", variant="real-gamma"`` with the torus keyword and the
    deprecated ``jk_method="real-gamma"`` spelling with the k-mesh alias
    execute the same dispatch: same converged energy, same executed route in
    the manifest, and the front door records the variant and the selector
    surface. No numerics change."""
    from vibeqc import run_periodic_job
    from vibeqc._vibeqc_core import BasisSet

    legacy = _run_job(tmp_path, "RHF")  # jk_method="real-gamma", kpoints=(2,1,1)
    cell = _h2_cell()
    basis = BasisSet(cell.unit_cell_molecule(), "sto-3g")
    front = _quiet(
        run_periodic_job, cell, basis, method="aiccm", variant="real-gamma",
        aiccm_lattice_extension=(2, 1, 1), initial_guess="HCORE",
        output=str(tmp_path / "fd"))
    assert front.converged and legacy.converged
    assert front.energy == pytest.approx(legacy.energy, abs=1e-10)
    manifest = (tmp_path / "fd.system").read_text(encoding="utf-8")
    for needle in (
        'jk_method_requested = "real-gamma"',
        'jk_method_executed = "real-gamma"',
        'aiccm_variant  = "real-gamma"',
        'aiccm_selector = "front-door"',
        'method_status  = "experimental"',
    ):
        assert needle in manifest, needle
    assert 'aiccm_selector = "legacy-jk_method"' in (
        tmp_path / "rg.system").read_text(encoding="utf-8")
    # The legacy .out carries the deprecation pointer (Python hides the
    # DeprecationWarning outside __main__ by default).
    assert (
        "(user-requested: 'real-gamma'; deprecated spelling of "
        "method='aiccm', variant='real-gamma')"
    ) in (tmp_path / "rg.out").read_text(encoding="utf-8")
    out = (tmp_path / "fd.out").read_text(encoding="utf-8")
    assert "J/K method: aiccm (real-gamma)" in out
    assert "selected by method='aiccm', variant='real-gamma'" in out
    assert "SCF reference RHF inferred" in out
    # The torus keyword reached the producer as its Γ-centred mesh.
    assert "BvK torus (nrep)    = (2, 1, 1)" in out
    assert "BvK torus (nrep)    = (2, 1, 1)" in (
        tmp_path / "rg.out"
    ).read_text(encoding="utf-8")


def test_supercell_gamma_variants_default_to_hcore_but_refuse_explicit_sad(
    tmp_path,
):
    """The front door's default must be reachable for every variant (#692).

    ``bbe7b9001`` gave the supercell-Gamma variants a guard that refuses any
    guess but HCORE, which their SCF loops are the only ones to implement.
    The historical runner-wide default was SAD, so both variants failed with *default
    arguments* -- half of a documented front door unreachable -- and eight
    tests on the #670 arm went red for the same reason.

    The distinction this pins: an omitted or explicit AUTO ``initial_guess`` resolves to the
    guess the variant executes, while an EXPLICIT ``initial_guess="SAD"``
    still fails before SCF. Substituting a guess the caller actually asked for
    is what the guard exists to prevent; supplying a working default is not
    that.
    """
    import vibeqc as vq

    a = 6.0
    cell = vq.PeriodicSystem(
        3,
        np.diag([a] * 3),
        [
            vq.Atom(1, [a / 2, a / 2, a / 2 - 0.7]),
            vq.Atom(1, [a / 2, a / 2, a / 2 + 0.7]),
        ],
        0,
        1,
    )
    basis = vq.BasisSet(cell.unit_cell_molecule(), "sto-3g")

    for variant in ("real-gamma", "four-center"):
        defaulted = vq.run_periodic_job(
            cell,
            basis,
            method="aiccm",
            variant=variant,
            output=tmp_path / f"default-{variant}",
            progress=False,
        )
        explicit = vq.run_periodic_job(
            cell,
            basis,
            method="aiccm",
            variant=variant,
            initial_guess="HCORE",
            output=tmp_path / f"hcore-{variant}",
            progress=False,
        )
        assert defaulted.guess_selection.requested == vq.InitialGuess.AUTO
        assert defaulted.guess_selection.effective == vq.InitialGuess.HCORE
        assert "AUTO -> HCORE" in (tmp_path / f"default-{variant}.out").read_text()
        assert explicit.guess_selection.requested == vq.InitialGuess.HCORE
        assert explicit.ccm_result.guess_selection.requested == vq.InitialGuess.HCORE
        # The default must not merely succeed -- it must be the same run.
        assert defaulted.energy == pytest.approx(explicit.energy, abs=1e-12)

        with pytest.raises(NotImplementedError, match="initial_guess"):
            vq.run_periodic_job(
                cell,
                basis,
                method="aiccm",
                variant=variant,
                initial_guess="SAD",
                output=tmp_path / f"sad-{variant}",
                progress=False,
            )


def test_ordinary_periodic_routes_keep_the_sad_default(tmp_path):
    """The variant-aware default must not leak into any other route (#692)."""
    import vibeqc as vq

    a = 6.0
    cell = vq.PeriodicSystem(
        3,
        np.diag([a] * 3),
        [
            vq.Atom(1, [a / 2, a / 2, a / 2 - 0.7]),
            vq.Atom(1, [a / 2, a / 2, a / 2 + 0.7]),
        ],
        0,
        1,
    )
    basis = vq.BasisSet(cell.unit_cell_molecule(), "sto-3g")
    defaulted = vq.run_periodic_job(
        cell, basis, method="RHF", jk_method="gdf",
        output=tmp_path / "gdf-default", progress=False,
    )
    explicit_sad = vq.run_periodic_job(
        cell, basis, method="RHF", jk_method="gdf", initial_guess="SAD",
        output=tmp_path / "gdf-sad", progress=False,
    )
    assert defaulted.guess_selection.requested == vq.InitialGuess.AUTO
    assert defaulted.guess_selection.effective == vq.InitialGuess.SAD
    assert defaulted.energy == pytest.approx(explicit_sad.energy, abs=1e-12)


@pytest.mark.parametrize("selector", ["SAD", "READ", "bad-guess"])
@pytest.mark.parametrize("variant", ["real-gamma", "four-center"])
def test_direct_ccm_adapters_validate_before_basis(selector, variant):
    from vibeqc.periodic.ccm.four_center_runner import run_four_center_scf
    adapter = run_real_gamma_scf if variant == "real-gamma" else run_four_center_scf
    with pytest.raises((ValueError, NotImplementedError), match="guess"):
        adapter(_h2_cell(), "intentionally-invalid-basis", "RHF", (1, 1, 1),
                initial_guess=selector)
