"""run_periodic_job × automatic convergence strategy (transparency).

End-to-end pins of the contract on the BIPOLE route:

* default (nothing given) → AUTO, stated in the .out with profile +
  per-knob reasons;
* any explicit knob → manual, auto fills nothing;
* convergence="off" → plain defaults, stated;
* non-BIPOLE routes: the block exists and is honest about v1 scope.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import vibeqc as vq
from vibeqc.pbc_gdf import PBCGDFUKSResult
from vibeqc.periodic_k_gdf import PeriodicKRKSGDFResult


def _h2_box(box: float = 25.0):
    return vq.PeriodicSystem(
        3,
        np.eye(3) * box,
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )


_QUIET = dict(
    write_molden_file=False,
    write_xyz_file=False,
    write_poscar_file=False,
    write_xsf_structure_file=False,
    write_cif_file=False,
    write_population_file=False,
    progress=False,
)


def _run(tmp_path, name, **kw):
    sysp = _h2_box(box=12.0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RHF",
        jk_method="bipole",
        output=str(tmp_path / name),
        max_iter=40,
        **_QUIET,
        **kw,
    )
    out_text = (tmp_path / name).with_suffix(".out").read_text()
    return result, out_text


def _mgo_cell():
    return vq.PeriodicSystem(
        3,
        np.eye(3) * 6.0,
        [vq.Atom(12, [0.0, 0.0, 0.0]), vq.Atom(8, [3.0, 3.0, 3.0])],
    )


def _li2_metal_cell():
    return vq.PeriodicSystem(
        3,
        np.eye(3) * 6.0,
        [vq.Atom(3, [0.0, 0.0, 0.0]), vq.Atom(3, [3.0, 3.0, 3.0])],
    )


def _diamond_c_cell():
    a = 6.74065308
    half = 0.5 * a
    quarter = 0.25 * a
    lattice = np.array(
        [
            [0.0, half, half],
            [half, 0.0, half],
            [half, half, 0.0],
        ]
    )
    return vq.PeriodicSystem(
        3,
        lattice,
        [vq.Atom(6, [0.0, 0.0, 0.0]), vq.Atom(6, [quarter, quarter, quarter])],
    )


def _fake_krks_result(kpoints, fock_mixing):
    n_k = int(np.prod(kpoints if isinstance(kpoints, (list, tuple)) else (1, 1, 1)))
    weights = np.full(n_k, 1.0 / n_k)
    zeros = [np.zeros((2, 2), dtype=complex) for _ in range(n_k)]
    overlap = [np.eye(2, dtype=complex) for _ in range(n_k)]
    density = [np.diag([2.0, 0.0]).astype(complex) for _ in range(n_k)]
    return PeriodicKRKSGDFResult(
        energy=-1.0,
        e_electronic=-1.5,
        e_nuclear=0.5,
        n_iter=2,
        converged=True,
        mo_energies=[np.array([-0.5, 0.5]) for _ in range(n_k)],
        mo_coeffs=[np.eye(2, dtype=complex) for _ in range(n_k)],
        fock=zeros,
        overlap=overlap,
        hcore=zeros,
        density=density,
        kpoints_cart=np.zeros((n_k, 3)),
        kpoint_weights=weights,
        functional="lda",
        e_xc=-0.1,
        fock_mixing=fock_mixing,
        occupations=[np.array([2.0, 0.0]) for _ in range(n_k)],
    )


def _fake_gamma_uks_result(fock_mixing):
    eye = np.eye(2)
    zeros = np.zeros((2, 2))
    density = np.diag([1.0, 0.0])
    return PBCGDFUKSResult(
        energy=-1.0,
        e_electronic=-1.5,
        e_nuclear=0.5,
        e_coulomb=0.0,
        e_hf_exchange=0.0,
        e_xc=-0.1,
        e_exxdiv=0.0,
        n_iter=2,
        converged=True,
        s_squared=0.0,
        s_squared_ideal=0.0,
        functional="lda",
        mo_energies_alpha=np.array([-0.5, 0.5]),
        mo_coeffs_alpha=eye.copy(),
        density_alpha=density.copy(),
        fock_alpha=zeros.copy(),
        mo_energies_beta=np.array([-0.5, 0.5]),
        mo_coeffs_beta=eye.copy(),
        density_beta=density.copy(),
        fock_beta=zeros.copy(),
        overlap=eye.copy(),
        hcore=zeros.copy(),
        fock_mixing=fock_mixing,
    )


def _fake_gpw_multik_result(kpoints):
    n_k = int(np.prod(kpoints if isinstance(kpoints, (list, tuple)) else (1, 1, 1)))
    eye = np.eye(2, dtype=complex)
    return SimpleNamespace(
        energy=-1.0,
        n_iter=1,
        converged=True,
        scf_trace=[
            {
                "iter": 1,
                "energy": -1.0,
                "delta_e": 0.0,
                "grad_norm": 0.0,
                "diis_subspace": 0,
            }
        ],
        mo_energies_k=[np.array([-0.5, 0.5]) for _ in range(n_k)],
        mo_coeffs_k=[eye.copy() for _ in range(n_k)],
        occupations_k=[np.array([2.0, 0.0]) for _ in range(n_k)],
        density=[np.diag([2.0, 0.0]).astype(complex) for _ in range(n_k)],
        fock_mixing=0.0,
    )


def test_auto_default_stated_in_out(tmp_path):
    result, out = _run(tmp_path, "auto_default")
    assert result.converged
    assert "Convergence strategy" in out
    assert "AUTO (default" in out
    assert "profile: molecular-limit" in out
    # H2-in-a-box: auto leaves everything plain for RHF.
    assert "fock_mixing = 0.3" not in out


def test_explicit_knob_switches_to_manual(tmp_path):
    result, out = _run(tmp_path, "manual", fmixing_percent=10.0)
    assert result.converged
    assert "manual (explicit user options" in out
    assert "[explicit]" in out
    # The explicit value flows through.
    assert "fmixing_percent     = 10.0" in out


def test_fock_mixing_alias_uses_fractional_scale(tmp_path):
    result, out = _run(tmp_path, "manual_alias", fock_mixing=0.10)
    assert result.converged
    assert result.fock_mixing == pytest.approx(0.10)
    assert "manual (explicit user options" in out
    assert "fmixing_percent     = 10.0" in out


def test_fock_mixing_and_fmixing_percent_are_exclusive(tmp_path):
    sysp = _h2_box(box=12.0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    with pytest.raises(ValueError, match="either fmixing_percent= or fock_mixing="):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RHF",
            jk_method="bipole",
            output=str(tmp_path / "bad_mix"),
            fmixing_percent=10.0,
            fock_mixing=0.1,
            **_QUIET,
        )


def test_convergence_off_stated(tmp_path):
    result, out = _run(tmp_path, "conv_off", convergence="off")
    assert result.converged
    assert 'off (convergence="off"' in out


def test_convergence_auto_requested_label(tmp_path):
    result, out = _run(tmp_path, "auto_req", convergence="auto")
    assert result.converged
    assert 'AUTO (requested via convergence="auto")' in out


def test_bipole_rhf_accepts_smearing_bz_metadata(tmp_path):
    result, out = _run(
        tmp_path,
        "rhf_smearing_bz_metadata",
        bz_integration="smearing",
        kpoints=(1, 1, 1),
    )
    assert result.converged
    assert "bz_integration     = smearing" in out


def test_bipole_rks_forwards_gilat_bz_integration(monkeypatch, tmp_path):
    """The BIPOLE RKS call must forward bz_integration to the driver.
    Pre-fix the runner validated gilat (documented BIPOLE-RKS-only),
    recorded it in the .out, then silently dropped the keyword -- the
    SCF ran without Gilat-net BZ integration."""
    sysp = _h2_box(box=12.0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    captured = {}

    class _Sentinel(RuntimeError):
        pass

    def fake_rks(_system, _basis, _kmesh, _opts, **kwargs):
        captured["bz_integration"] = kwargs.get("bz_integration")
        raise _Sentinel("captured; abort before SCF")

    # The runner imports the driver at call time
    # (``from .pbc_bipole_rks import run_pbc_bipole_rks``), so patch the
    # source module, not periodic_runner.
    monkeypatch.setattr(
        "vibeqc.pbc_bipole_rks.run_pbc_bipole_rks", fake_rks
    )
    with pytest.raises(_Sentinel):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RKS",
            functional="lda",
            jk_method="bipole",
            kpoints=(1, 1, 1),
            bz_integration="gilat",
            output=str(tmp_path / "rks_gilat"),
            citations=False,
            **_QUIET,
        )
    assert captured["bz_integration"] == "gilat"


def test_explicit_smearing_zero_is_respected_not_auto(tmp_path):
    """smearing_temperature=0.0 is an explicit user choice (off) and
    must put the run in manual mode — never auto-overridden."""
    result, out = _run(tmp_path, "smear_off", smearing_temperature=0.0)
    assert result.converged
    assert "manual (explicit user options" in out


def test_bad_convergence_keyword_raises(tmp_path):
    sysp = _h2_box(box=12.0)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    with pytest.raises(ValueError, match="convergence must be"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RHF",
            jk_method="bipole",
            output=str(tmp_path / "bad"),
            convergence="magic",
            **_QUIET,
        )


def test_gdf_route_auto_supported_no_ks_floor(tmp_path):
    """The GDF route participates in auto-convergence (queue #2), and —
    unlike BIPOLE — gets no KS FMIXING floor (its drivers honor the
    resolved value verbatim, no in-driver 30% default)."""
    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RKS",
        functional="lda",
        jk_method="gdf",
        output=str(tmp_path / "gdf_auto"),
        max_iter=60,
        **_QUIET,
    )
    out = (tmp_path / "gdf_auto").with_suffix(".out").read_text()
    assert result.converged
    assert "Convergence strategy" in out
    assert "AUTO (default" in out
    assert "profile: molecular-limit" in out
    # No BIPOLE-style KS floor on GDF: fock_mixing stays at 0.
    assert "fock_mixing = 0.3" not in out


def test_gdf_auto_fmixing_reaches_multi_k_ionic_driver(monkeypatch, tmp_path):
    sysp = _mgo_cell()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    captured = {}

    def fake_driver(_system, _basis, kpoints, options, **kwargs):
        captured["option_fock_mixing"] = float(options.fock_mixing)
        captured["kw_fock_mixing"] = float(kwargs["fock_mixing"])
        return _fake_krks_result(kpoints, captured["kw_fock_mixing"])

    monkeypatch.setattr(
        "vibeqc.periodic_runner.run_krks_periodic_gdf",
        fake_driver,
    )

    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RKS",
        functional="lda",
        jk_method="gdf",
        kpoints=(2, 2, 2),
        output=str(tmp_path / "mgo_gdf_auto"),
        max_iter=20,
        citations=False,
        **_QUIET,
    )

    assert result.converged
    assert captured["option_fock_mixing"] == pytest.approx(0.30)
    assert captured["kw_fock_mixing"] == pytest.approx(0.30)


def test_gdf_explicit_level_shift_reaches_multi_k_closed_shell_driver(
    monkeypatch, tmp_path
):
    sysp = _mgo_cell()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    captured = {}

    class _Captured(RuntimeError):
        pass

    def fake_driver(_system, _basis, _kpoints, options, **_kwargs):
        captured["level_shift"] = float(options.level_shift)
        captured["gdf_method"] = _kwargs["gdf_method"]
        raise _Captured("captured before SCF")

    monkeypatch.setattr(
        "vibeqc.periodic_runner.run_krhf_periodic_gdf",
        fake_driver,
    )

    with pytest.raises(_Captured, match="captured before SCF"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RHF",
            jk_method="gdf",
            kpoints=(1, 1, 2),
            gdf_method="mdf",
            level_shift=0.42,
            output=str(tmp_path / "gdf_explicit_level_shift"),
            output_qvf=False,
            citations=False,
            **_QUIET,
        )

    assert captured["level_shift"] == pytest.approx(0.42)
    assert captured["gdf_method"] == "mdf"


def test_gdf_explicit_level_shift_reaches_gamma_fallback(monkeypatch, tmp_path):
    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    captured = {}

    class _Captured(RuntimeError):
        pass

    def fake_driver(_system, _basis, options, **_kwargs):
        captured["level_shift"] = float(options.level_shift)
        raise _Captured("captured before SCF")

    monkeypatch.setattr(
        "vibeqc.periodic_runner.run_rhf_periodic_gamma_gdf",
        fake_driver,
    )

    with pytest.raises(_Captured, match="captured before SCF"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RHF",
            jk_method="gdf",
            level_shift=0.31,
            output=str(tmp_path / "gdf_gamma_fallback_level_shift"),
            output_qvf=False,
            citations=False,
            **_QUIET,
        )

    assert captured["level_shift"] == pytest.approx(0.31)


def test_gdf_explicit_level_shift_reaches_shifted_single_k_driver(
    monkeypatch, tmp_path
):
    sysp = _mgo_cell()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    shifted_kpoint = vq.KPoints.from_list(sysp, [0.25, 0.0, 0.0])
    captured = {}

    class _Captured(RuntimeError):
        pass

    def fake_driver(_system, _basis, _kpoints, options, **kwargs):
        captured["level_shift"] = float(options.level_shift)
        captured["gdf_method"] = kwargs["gdf_method"]
        raise _Captured("captured before SCF")

    monkeypatch.setattr(
        "vibeqc.periodic_runner.run_krhf_periodic_gdf",
        fake_driver,
    )

    with pytest.raises(_Captured, match="captured before SCF"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RHF",
            jk_method="gdf",
            kpoints=shifted_kpoint,
            gdf_method="mdf",
            level_shift=0.42,
            output=str(tmp_path / "gdf_shifted_single_k_level_shift"),
            output_qvf=False,
            citations=False,
            **_QUIET,
        )

    assert captured["level_shift"] == pytest.approx(0.42)
    assert captured["gdf_method"] == "mdf"


def test_gdf_auto_level_shift_reaches_multi_k_metallic_hf_driver(
    monkeypatch, tmp_path
):
    sysp = _li2_metal_cell()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    captured = {}

    class _Captured(RuntimeError):
        pass

    def fake_driver(_system, _basis, _kpoints, options, **_kwargs):
        captured["level_shift"] = float(options.level_shift)
        raise _Captured("captured before SCF")

    monkeypatch.setattr(
        "vibeqc.periodic_runner.run_krhf_periodic_gdf",
        fake_driver,
    )

    with pytest.raises(_Captured, match="captured before SCF"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RHF",
            jk_method="gdf",
            kpoints=(1, 1, 2),
            convergence="auto",
            output=str(tmp_path / "gdf_auto_level_shift"),
            output_qvf=False,
            citations=False,
            **_QUIET,
        )

    assert captured["level_shift"] == pytest.approx(0.20)


def test_gdf_level_shift_fails_closed_for_open_shell_driver(tmp_path):
    sysp = _h2_box()
    sysp.multiplicity = 3
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    with pytest.raises(NotImplementedError, match="open-shell UHF/GDF"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="UHF",
            jk_method="gdf",
            kpoints=(1, 1, 2),
            level_shift=0.25,
            output=str(tmp_path / "gdf_open_shell_level_shift"),
            output_qvf=False,
            citations=False,
            **_QUIET,
        )


def test_gdf_fock_mixing_fails_closed_for_open_shell_kmesh_driver(tmp_path):
    sysp = _h2_box()
    sysp.multiplicity = 3
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    with pytest.raises(NotImplementedError, match="open-shell UHF/GDF k-mesh"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="UHF",
            jk_method="gdf",
            kpoints=(1, 1, 2),
            fock_mixing=0.25,
            output=str(tmp_path / "gdf_open_shell_fock_mixing"),
            output_qvf=False,
            citations=False,
            **_QUIET,
        )


@pytest.mark.parametrize(
    "kpoints_factory",
    [
        lambda _sysp: None,
        lambda _sysp: (1, 1, 1),
        lambda sysp: vq.monkhorst_pack(sysp, [1, 1, 1]),
    ],
    ids=["implicit-gamma", "tuple-gamma", "bloch-gamma"],
)
def test_gdf_gamma_explicit_method_level_shift_fails_closed(
    tmp_path, kpoints_factory
):
    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    with pytest.raises(NotImplementedError, match="explicit gdf_method"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RHF",
            jk_method="gdf",
            gdf_method="rsgdf",
            kpoints=kpoints_factory(sysp),
            level_shift=0.25,
            output=str(tmp_path / "gdf_gamma_explicit_level_shift"),
            output_qvf=False,
            citations=False,
            **_QUIET,
        )


def test_gdf_gamma_explicit_method_fock_mixing_fails_closed(tmp_path):
    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    with pytest.raises(NotImplementedError, match="fock_mixing"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RHF",
            jk_method="gdf",
            gdf_method="rsgdf",
            fock_mixing=0.25,
            output=str(tmp_path / "gdf_gamma_explicit_fock_mixing"),
            output_qvf=False,
            citations=False,
            **_QUIET,
        )


@pytest.mark.parametrize("convergence", [None, "auto"], ids=["default", "requested"])
def test_gdf_gamma_explicit_method_auto_level_shift_is_filtered(
    monkeypatch, tmp_path, convergence
):
    sysp = _li2_metal_cell()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    captured = {}

    class _Captured(RuntimeError):
        pass

    def fake_driver(_system, _basis, options, **_kwargs):
        captured["level_shift"] = float(options.level_shift)
        captured["fock_mixing"] = float(options.fock_mixing)
        raise _Captured("captured before SCF")

    monkeypatch.setattr(
        "vibeqc.periodic_runner.run_pbc_gdf_rhf",
        fake_driver,
    )
    output_stem = tmp_path / f"gdf_gamma_explicit_auto_{convergence}"

    with pytest.raises(_Captured, match="captured before SCF"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="RHF",
            jk_method="gdf",
            gdf_method="rsgdf",
            convergence=convergence,
            output=str(output_stem),
            output_qvf=False,
            citations=False,
            **_QUIET,
        )

    assert captured["level_shift"] == 0.0
    assert captured["fock_mixing"] == 0.0
    out = output_stem.with_suffix(".out").read_text()
    assert "the exact-Gamma GDF driver" in out
    assert "does not implement Fock mixing" in out


@pytest.mark.parametrize("convergence", [None, "auto"], ids=["default", "requested"])
def test_gdf_auto_level_shift_is_filtered_for_open_shell_driver(
    monkeypatch, tmp_path, convergence
):
    sysp = _li2_metal_cell()
    sysp.multiplicity = 3
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    captured = {}

    class _Captured(RuntimeError):
        pass

    def fake_driver(_system, _basis, _kpoints, options, **_kwargs):
        captured["level_shift"] = float(options.level_shift)
        captured["fock_mixing"] = float(options.fock_mixing)
        raise _Captured("captured before SCF")

    monkeypatch.setattr(
        "vibeqc.periodic_runner.run_kuhf_periodic_gdf",
        fake_driver,
    )
    output_stem = tmp_path / f"gdf_open_shell_auto_{convergence}"

    with pytest.raises(_Captured, match="captured before SCF"):
        vq.run_periodic_job(
            sysp,
            basis,
            method="UHF",
            jk_method="gdf",
            kpoints=(1, 1, 2),
            convergence=convergence,
            output=str(output_stem),
            output_qvf=False,
            citations=False,
            **_QUIET,
        )

    assert captured["level_shift"] == 0.0
    assert captured["fock_mixing"] == 0.0
    out = output_stem.with_suffix(".out").read_text()
    assert "open-shell GDF drivers" in out
    assert "selected open-shell GDF driver" in out


def test_r2scan_compact_gdf_auto_density_mixer_manual_damping(
    monkeypatch, tmp_path
):
    sysp = _mgo_cell()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    captured = {}

    def fake_driver(_system, _basis, kpoints, _options, **kwargs):
        captured.update(kwargs)
        return _fake_krks_result(kpoints, kwargs["fock_mixing"])

    monkeypatch.setattr(
        "vibeqc.periodic_runner.run_krks_periodic_gdf",
        fake_driver,
    )

    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RKS",
        functional="r2scan",
        jk_method="gdf",
        kpoints=(1, 1, 1),
        damping=0.0,
        output=str(tmp_path / "mgo_r2scan_gdf_mixer"),
        output_qvf=False,
        max_iter=20,
        citations=False,
        **_QUIET,
    )

    out = (tmp_path / "mgo_r2scan_gdf_mixer").with_suffix(".out").read_text()
    assert result.converged
    assert captured["density_mixer"] == "anderson"
    assert captured["density_mixer_beta"] == pytest.approx(0.35)
    assert "compact periodic SCAN/r2SCAN RKS/GDF profile" in out
    assert "density_mixer='anderson'" in out


def test_r2scan_c_diamond_reproducer_auto_density_mixer(monkeypatch, tmp_path):
    sysp = _diamond_c_cell()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    captured = {}

    def fake_driver(_system, _basis, kpoints, _options, **kwargs):
        captured.update(kwargs)
        return _fake_krks_result(kpoints, kwargs["fock_mixing"])

    monkeypatch.setattr(
        "vibeqc.periodic_runner.run_krks_periodic_gdf",
        fake_driver,
    )

    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RKS",
        functional="r2scan",
        jk_method="gdf",
        kpoints=(1, 1, 1),
        output=str(tmp_path / "c_diamond_r2scan_gdf_mixer"),
        output_qvf=False,
        max_iter=120,
        citations=False,
        **_QUIET,
    )

    out = (tmp_path / "c_diamond_r2scan_gdf_mixer").with_suffix(".out").read_text()
    assert result.converged
    assert captured["density_mixer"] == "anderson"
    assert captured["density_mixer_beta"] == pytest.approx(0.35)
    assert captured["fock_mixing"] == pytest.approx(0.0)
    assert "profile: covalent-insulator" in out
    assert "density_mixer='anderson'" in out


def test_r2scan_molecular_limit_gdf_keeps_diis_route(monkeypatch, tmp_path):
    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    captured = {}

    def fake_driver(_system, _basis, kpoints, _options, **kwargs):
        captured.update(kwargs)
        return _fake_krks_result(kpoints, kwargs["fock_mixing"])

    monkeypatch.setattr(
        "vibeqc.periodic_runner.run_krks_periodic_gdf",
        fake_driver,
    )

    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RKS",
        functional="r2scan",
        jk_method="gdf",
        kpoints=(1, 1, 1),
        damping=0.0,
        output=str(tmp_path / "h2_r2scan_gdf_no_mixer"),
        output_qvf=False,
        max_iter=20,
        citations=False,
        **_QUIET,
    )

    out = (tmp_path / "h2_r2scan_gdf_no_mixer").with_suffix(".out").read_text()
    assert result.converged
    assert captured["density_mixer"] is None
    assert "compact periodic SCAN/r2SCAN RKS/GDF profile" not in out


def test_gdf_auto_fmixing_reaches_gamma_rks_pure_gdf_driver(monkeypatch, tmp_path):
    sysp = _mgo_cell()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    captured = {}

    def fake_driver(_system, _basis, options, **kwargs):
        captured["option_fock_mixing"] = float(options.fock_mixing)
        captured["gdf_method"] = kwargs["gdf_method"]
        return _fake_gamma_uks_result(captured["option_fock_mixing"])

    def legacy_bridge(*_args, **_kwargs):  # pragma: no cover - should not route here
        raise AssertionError("ionic Gamma RKS/GDF auto fell back to legacy bridge")

    monkeypatch.setattr(
        "vibeqc.periodic_runner.run_pbc_gdf_uks",
        fake_driver,
    )
    monkeypatch.setattr(
        "vibeqc.periodic_runner.run_rhf_periodic_gamma_gdf",
        legacy_bridge,
    )

    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RKS",
        functional="lda",
        jk_method="gdf",
        output=str(tmp_path / "mgo_gamma_gdf_auto"),
        output_qvf=False,
        max_iter=20,
        citations=False,
        **_QUIET,
    )

    out = (tmp_path / "mgo_gamma_gdf_auto").with_suffix(".out").read_text()
    assert result.converged
    assert captured["gdf_method"] == "rsgdf"
    assert captured["option_fock_mixing"] == pytest.approx(0.30)
    assert "fock_mixing         = 0.3" in out


def test_unsupported_auto_note_names_bipole_and_gdf(monkeypatch, tmp_path):
    sysp = _h2_box()
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    def fake_gpw(_system, _basis, kpoints, **_kwargs):
        return _fake_gpw_multik_result(kpoints.mesh)

    monkeypatch.setattr(
        "vibeqc.periodic_gapw_j.run_periodic_rks_gpw_multi_k",
        fake_gpw,
    )

    result = vq.run_periodic_job(
        sysp,
        basis,
        method="RKS",
        functional="lda",
        jk_method="gpw",
        kpoints=(2, 2, 2),
        convergence="auto",
        output=str(tmp_path / "gpw_auto_note"),
        output_qvf=False,
        citations=False,
        **_QUIET,
    )

    out = (tmp_path / "gpw_auto_note").with_suffix(".out").read_text()
    assert result.converged
    assert 'convergence="auto" is wired for jk_method="bipole" and' in out
    assert 'jk_method="gdf" only' in out
