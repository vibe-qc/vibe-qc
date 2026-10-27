"""CLI reference/response consistency for molecular TDDFT."""

from types import SimpleNamespace

import numpy as np
import pytest
import vibeqc as vq
from vibeqc import tddft
from vibeqc._cli import _cmd_tddft


@pytest.mark.parametrize("functional", [None, "pbe"])
@pytest.mark.parametrize("mult", [1, 2])
@pytest.mark.parametrize("casida", [False, True])
def test_cli_passes_reference_density_grid_and_spin(tmp_path, monkeypatch, functional, mult, casida):
    path = tmp_path / "atom.xyz"
    path.write_text("1\natom\n" + ("He" if mult == 1 else "Li") + " 0 0 0\n")
    reference = SimpleNamespace(
        converged=True, n_iter=1, energy=-1.0,
        mo_energies=np.arange(5.0), mo_coeffs=np.eye(5), density=np.eye(5),
        mo_energies_alpha=np.arange(5.0), mo_energies_beta=np.arange(5.0),
        mo_coeffs_alpha=np.eye(5), mo_coeffs_beta=np.eye(5),
        density_alpha=np.eye(5), density_beta=np.eye(5) * 0.5,
    )
    seen = {}

    def scf(mol, basis, opts):
        if functional:
            assert opts.functional == functional
            seen["grid"] = opts.grid
        return reference

    for name in ["run_rhf", "run_uhf", "run_rks", "run_uks"]:
        monkeypatch.setattr(vq, name, scf)

    def response(*args, **kwargs):
        seen["response"] = True
        assert kwargs["functional"] == functional
        grid = kwargs["grid_options"]
        if functional:
            assert grid.n_radial == seen["grid"].n_radial
            assert grid.lebedev_order == seen["grid"].lebedev_order
        else:
            assert grid is None
        if mult == 1:
            assert args[4] == 1
            assert kwargs["density_ao"] is (reference.density if functional else None)
        else:
            assert args[6:8] == (2, 1)
            assert kwargs["density_alpha_ao"] is (reference.density_alpha if functional else None)
            assert kwargs["density_beta_ao"] is (reference.density_beta if functional else None)
        return SimpleNamespace(method="test", n_states=0, states=[])

    def wrong_response(*args, **kwargs):
        pytest.fail("CLI called the wrong spin or TDA/Casida driver")

    selected = "run_tddft_" + ("casida" if casida else "tda") + ("_uhf" if mult == 2 else "")
    for name in ["run_tddft_tda", "run_tddft_casida", "run_tddft_tda_uhf", "run_tddft_casida_uhf"]:
        monkeypatch.setattr(tddft, name, response if name == selected else wrong_response)
    assert _cmd_tddft(SimpleNamespace(path=path, basis="sto-3g", functional=functional,
                                     charge=0, multiplicity=mult, n_states=1, casida=casida)) == 0
    assert seen["response"]


@pytest.mark.parametrize("name", ["run_tddft_tda", "run_tddft_casida", "run_tddft_tda_uhf", "run_tddft_casida_uhf"])
def test_response_driver_forwards_explicit_xc_grid(monkeypatch, name):
    grid = vq.GridOptions()
    grid.n_radial = 71
    eps, coeff, density = np.arange(3.0), np.eye(3), np.eye(3)
    monkeypatch.setattr(tddft, "compute_eri", lambda basis: np.zeros((3, 3, 3, 3)))
    monkeypatch.setattr(tddft, "eri_ao_to_mo_blocks", lambda *args: (None, None))

    class ReachedKernel(Exception):
        pass

    def kernel(*args, grid_options=None):
        assert grid_options is grid
        raise ReachedKernel()

    monkeypatch.setattr(tddft, "_compute_alda_kernel_mo", kernel)
    monkeypatch.setattr(tddft, "_compute_polarised_kernel_mo_uhf", kernel)
    with pytest.raises(ReachedKernel):
        if name.endswith("_uhf"):
            getattr(tddft, name)(None, None, eps, eps, coeff, coeff, 2, 1,
                                functional="pbe", density_alpha_ao=density,
                                density_beta_ao=density, grid_options=grid)
        else:
            getattr(tddft, name)(None, None, eps, coeff, 1,
                                functional="pbe", density_ao=density, grid_options=grid)


@pytest.mark.parametrize("mult,casida", [(1, False), (2, True)])
def test_cli_pbe_native_smoke(tmp_path, capsys, mult, casida):
    """Exercise the real SCF signature and XC response, not just mocks."""
    path = tmp_path / "molecule.xyz"
    path.write_text("2\nH2\nH 0 0 0\nH 0 0 0.74\n" if mult == 1 else "1\nLi\nLi 0 0 0\n")
    assert _cmd_tddft(SimpleNamespace(path=path, basis="sto-3g", functional="pbe",
                                     charge=0, multiplicity=mult, n_states=1, casida=casida)) == 0
    output = capsys.readouterr().out
    assert "SCF converged" in output
    assert "n_states=1" in output


@pytest.mark.parametrize("unrestricted", [False, True])
def test_nto_cli_reads_canonical_state_links(tmp_path, capsys, unrestricted):
    """NTO identity comes from metadata even when the section id is arbitrary."""
    import json
    import zipfile
    from vibeqc._cli import _cmd_nto

    path = tmp_path / "states.qvf"
    meta = {
        "orbital_kind": "nto", "nto_role": "hole",
        "state_ref": "response", "state_index": 2,
        "excitation_index": 99, "excitation_energy_ev": 999.0,
    }
    if unrestricted:
        meta.update(spin="unrestricted",
                    alpha={"n_mo": 2, "occupations": [0.6, 0.1]},
                    beta={"n_mo": 1, "occupations": [0.3]})
    else:
        meta.update(spin="restricted", n_mo=2, occupations=[0.9, 0.1])
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"qvf_version": 1, "sections": [
            {"id": "arbitrary", "kind": "wavefunction.gto",
             "members": {"mo_metadata": {"path": "orbitals.json"}}},
            {"id": "response", "kind": "excited_states",
             "members": {"states": {"path": "states.json"}}},
        ]}))
        zf.writestr("orbitals.json", json.dumps(meta))
        zf.writestr("states.json", json.dumps({"states": [{
            "index": 2, "excitation_energy": {"value": 0.25, "units": "hartree"},
        }]}))
    assert _cmd_nto(SimpleNamespace(path=path, json=True)) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == (2 if unrestricted else 1)
    for row in rows:
        assert row["id"] == "arbitrary"
        assert row["state_index"] == 2
        assert row["role"] == "hole"
        assert row["excitation_energy_ev"] == pytest.approx(0.25 * 27.211386245988)
    if unrestricted:
        assert [row["spin"] for row in rows] == ["alpha", "beta"]
        assert rows[0]["nto_weights"] == [0.6, 0.1]
        assert rows[1]["nto_weights"] == [0.3]
    else:
        assert rows[0]["nto_weights"] == [0.9, 0.1]


def test_nto_cli_preserves_legacy_occupations(tmp_path, capsys):
    import json
    import zipfile
    from vibeqc._cli import _cmd_nto

    path = tmp_path / "legacy.qvf"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("manifest.json", json.dumps({"qvf_version": 1, "sections": [{
            "id": "wf_nto_S3_beta_electron", "kind": "wavefunction.gto",
            "members": {"mo_metadata": {"path": "legacy.json"}},
        }]}))
        zf.writestr("legacy.json", json.dumps({
            "spin": "restricted", "orbital_kind": "natural",
            "excitation_index": 3, "excitation_energy_ev": 4.5,
            "n_mo": 2, "occupations": [0.8, 0.6],
        }))
    assert _cmd_nto(SimpleNamespace(path=path, json=True)) == 0
    row, = json.loads(capsys.readouterr().out)
    assert row["state_index"] == "3"
    assert row["spin"] == "beta"
    assert row["role"] == "electron"
    assert row["excitation_energy_ev"] == 4.5
    assert row["nto_weights"] == [0.8, 0.6]
