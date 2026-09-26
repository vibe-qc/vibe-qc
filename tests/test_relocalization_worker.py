"""Standalone protocol, strict refusal, and independently checked subspaces."""

from __future__ import annotations

import copy
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import vibeqc.relocalize as worker
from vibeqc import Atom, BasisSet, Molecule, RHFOptions, compute_overlap, run_rhf
from vibeqc.output.formats.qvf import _basis_shell_payload
from vibeqc_relocalize import RequestError, parse_request

DATA = Path(__file__).parent / "data" / "relocalize"


def decode(value):
    a = np.array(value["data"])
    return (
        a[..., 0] + 1j * a[..., 1]
        if value["encoding"] == "complex_split_last_axis"
        else a
    )


def fixture(name):
    return json.loads((DATA / (name + ".request.json")).read_text())


def launch(request=None, *args, env=None):
    return subprocess.run(
        [sys.executable, "-m", "vibeqc_relocalize", *args],
        input=None if request is None else json.dumps(request) + "\n",
        text=True,
        capture_output=True,
        timeout=120,
        env=env,
        check=False,
    )


def independent_checks(before, after, s):
    np.testing.assert_allclose(
        after.conj().T @ s @ after, np.eye(after.shape[1]), atol=2e-8
    )
    # Compare full occupied density kernels, independently of the worker's
    # low-rank residual. Nonorthogonal AO metric only enters orthonormality.
    np.testing.assert_allclose(
        after @ after.conj().T, before @ before.conj().T, atol=2e-8
    )


@pytest.fixture(scope="module")
def water_request():
    positions = [[0.0, 0.0, 0.0], [0.0, -1.43, 1.11], [0.0, 1.43, 1.11]]
    mol = Molecule(
        [Atom(z, p) for z, p in zip([8, 1, 1], positions)], charge=0, multiplicity=1
    )
    basis = BasisSet(mol, "sto-3g")
    result = run_rhf(mol, basis, RHFOptions())
    assert result.converged
    request = fixture("h2")
    request["system"].update(atomic_numbers=[8, 1, 1], positions_bohr=positions)
    request["basis"]["shells"] = _basis_shell_payload(basis)[0]
    request["orbitals"] = {
        "coefficients": worker._encode(np.asarray(result.mo_coeffs)[:, :5].T),
        "occupations": [2.0] * 5,
    }
    return request, np.asarray(compute_overlap(basis))


@pytest.mark.parametrize("method", ["ibo", "boys", "pipek-mezey"])
def test_molecular_supplied_subspace_never_calls_scf(
    water_request, monkeypatch, method
):
    request, overlap = water_request
    request = copy.deepcopy(request)
    request["method"] = method

    def forbidden(*args, **kwargs):
        pytest.fail("Supplied localization must not call SCF")

    monkeypatch.setattr(worker, "run_rhf", forbidden)
    result = worker.localize(request)
    before = decode(request["orbitals"]["coefficients"]).T
    after = decode(result["coefficients"]).T
    independent_checks(before, after, overlap)
    np.testing.assert_allclose(
        np.array(result["atom_populations"]).sum(axis=1), 1, atol=1e-8
    )
    assert sum(result["charges"]) == pytest.approx(0, abs=1e-8)
    assert not result["scf_performed"]
    assert result["population_model"] == "iao-mini"


@pytest.mark.parametrize("name", ["h2", "periodic_gamma", "periodic_complex"])
def test_reproducible_request_result_fixtures(name):
    request = fixture(name)
    result = worker.localize(request)
    expected = json.loads((DATA / (name + ".result.json")).read_text())
    assert result["representation"] == expected["representation"]
    np.testing.assert_allclose(result["charges"], expected["charges"], atol=1e-8)
    np.testing.assert_allclose(
        result["atom_populations"], expected["atom_populations"], atol=1e-8
    )
    # Sign/gauge can change across BLAS/compiler versions: compare projectors.
    c = decode(result["coefficients"]).T
    ref = decode(expected["coefficients"]).T
    np.testing.assert_allclose(c @ c.conj().T, ref @ ref.conj().T, atol=1e-8)


@pytest.mark.parametrize("name", ["periodic_gamma", "periodic_complex"])
def test_periodic_complex_full_torus_preserves_original_bloch_subspace(
    name, monkeypatch
):
    request = fixture(name)

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Periodic requests must not enter molecular SCF/integrals/localization"
        )

    for attr in ("run_rhf", "compute_overlap", "analyse_localization"):
        monkeypatch.setattr(worker, attr, forbidden)
    result = worker.localize(request)
    p = request["periodic"]
    mesh = p["mesh"]
    ts = np.asarray(list(itertools.product(*(range(n) for n in mesh))))
    k = np.asarray(p["kpoints_fractional"])
    phase = np.exp(2j * np.pi * k @ ts.T)
    ck = decode(request["orbitals"]["coefficients"]).swapaxes(-1, -2)
    sk = decode(p["overlap"])
    # Independent discrete Fourier oracle, not the backend's helper functions.
    c0 = np.einsum("kt,kmn,kr->tmrn", phase, ck, phase.conj()).reshape(
        len(ts) * ck.shape[1], -1
    ) / len(ts)
    s = np.einsum("kt,kmn,kr->tmrn", phase, sk, phase.conj()).reshape(
        c0.shape[0], c0.shape[0]
    ) / len(ts)
    c = decode(result["coefficients"]).T
    independent_checks(c0, c, s)
    assert result["coefficients"]["encoding"] == "complex_split_last_axis"
    assert result["experimental"] and result["warnings"]
    assert not result["scf_performed"]
    assert sum(result["charges"]) == pytest.approx(0, abs=1e-8)


def test_native_gaussian_periodic_overlap_fixture():
    """Gamma fixture is tied to a lattice overlap integral, not molecular S."""
    import vibeqc as vq

    request = fixture("periodic_gamma")
    mol = Molecule(
        [
            Atom(z, p)
            for z, p in zip(
                request["system"]["atomic_numbers"], request["system"]["positions_bohr"]
            )
        ],
        charge=0,
        multiplicity=1,
    )
    basis = BasisSet(mol, "sto-3g")
    system = vq.PeriodicSystem(
        1, np.array(request["periodic"]["lattice_bohr"]), list(mol.atoms)
    )
    cells = vq.direct_lattice_cells(system, 12.0)
    sg = vq.compute_overlap_lattice_explicit(basis, system, cells)
    s = np.asarray(vq.bloch_sum(sg, [0.0, 0.0, 0.0]))
    np.testing.assert_allclose(s, decode(request["periodic"]["overlap"])[0], atol=1e-12)
    assert np.max(np.abs(s - np.asarray(vq.compute_overlap(basis)))) > 1e-3
    assert _basis_shell_payload(basis)[0] == request["basis"]["shells"]


def test_explicit_fresh_rhf_is_distinguished(monkeypatch):
    request = fixture("h2")
    request.pop("orbitals")
    request["source"] = "fresh_rhf"
    request["basis"].pop("shells")
    actual = worker.run_rhf
    calls = []

    def counted(*args):
        calls.append(True)
        return actual(*args)

    monkeypatch.setattr(worker, "run_rhf", counted)
    result = worker.localize(request)
    assert calls == [True]
    assert result["scf_performed"] and result["source"] == "fresh_rhf"


@pytest.mark.parametrize(
    "change,code",
    [
        (lambda r: r.pop("orbitals"), "invalid_source"),
        (lambda r: r.update(source="fresh_rhf"), "invalid_source"),
        (lambda r: r["basis"].pop("shells"), "incomplete_basis"),
        (lambda r: r["basis"].update(uses_ecp=True), "unsupported_basis"),
        (lambda r: r["basis"].update(ao_convention="unknown"), "unsupported_basis"),
        (lambda r: r["system"].update(spin="unrestricted"), "unsupported_spin"),
        (lambda r: r["system"].update(multiplicity=3), "unsupported_spin"),
        (lambda r: r["orbitals"].update(occupations=[1.5]), "unsupported_occupations"),
        (
            lambda r: r["orbitals"].update(
                coefficients={"encoding": "real", "data": [[2.0, 0.0]]}
            ),
            "nonorthonormal_input",
        ),
        (
            lambda r: r["orbitals"].update(
                coefficients=worker._encode(np.array([[1j, 0j]]))
            ),
            "unsupported_complex",
        ),
        (lambda r: r.update(periodic={}), "invalid_request"),
        (lambda r: r.update(unknown="do not ignore"), "invalid_request"),
        (
            lambda r: r["system"].update(
                positions_bohr=[[float("nan"), 0, 0], [0, 0, 1]]
            ),
            "invalid_request",
        ),
        (
            lambda r: r["system"]["positions_bohr"][0].__setitem__(0, False),
            "invalid_request",
        ),
        (
            lambda r: r["basis"]["shells"][0]["exponents"].__setitem__(0, True),
            "invalid_request",
        ),
        (
            lambda r: r["orbitals"]["coefficients"]["data"][0].__setitem__(0, True),
            "invalid_request",
        ),
    ],
)
def test_molecular_refusals_never_run_scf(change, code, monkeypatch):
    request = fixture("h2")
    change(request)
    monkeypatch.setattr(
        worker, "run_rhf", lambda *a: pytest.fail("silent SCF fallback")
    )
    with pytest.raises(RequestError) as exc:
        worker.localize(request)
    assert exc.value.code == code


@pytest.mark.parametrize(
    "change,code",
    [
        (lambda r: r.update(method="ibo"), "unsupported_periodic_method"),
        (lambda r: r.update(method="boys"), "unsupported_periodic_method"),
        (lambda r: r.update(method="pipek-mezey"), "unsupported_periodic_method"),
        (lambda r: r.pop("allow_experimental"), "experimental_opt_in_required"),
        (lambda r: r["periodic"].pop("overlap"), "invalid_request"),
        (
            lambda r: r["periodic"].update(representation="infinite_crystal"),
            "unsupported_periodic_representation",
        ),
        (
            lambda r: r["periodic"].update(
                kpoints_fractional=[[0.25, 0, 0], [0.75, 0, 0]]
            ),
            "unsupported_kmesh",
        ),
        (lambda r: r["periodic"].update(weights=[0.2, 0.8]), "unsupported_kmesh"),
        (lambda r: r["periodic"].update(pbc=[False, True, False]), "unsupported_pbc"),
        (
            lambda r: r["periodic"].update(
                orbital_energies_hartree=[[-1.0, -1.0], [-1.0, -1.0]]
            ),
            "unsupported_metallic_manifold",
        ),
        (
            lambda r: r["periodic"].update(orbital_energies_hartree=[[-1.0], [-1.0]]),
            "incomplete_band_manifold",
        ),
    ],
)
def test_periodic_refusals(change, code):
    request = fixture("periodic_complex")
    change(request)
    with pytest.raises(RequestError) as exc:
        worker.localize(request)
    assert exc.value.code == code


def test_complex_metric_never_discarded():
    request = fixture("periodic_gamma")
    s = np.array([[1, 0.1j], [-0.1j, 1]])
    vals, vecs = np.linalg.eigh(s)
    c = vecs[:, :1] / np.sqrt(vals[:1])
    request["periodic"]["overlap"] = worker._encode(s[None])
    request["orbitals"]["coefficients"] = worker._encode(c.T[None])
    with pytest.raises(RequestError, match="time reversal") as exc:
        worker.localize(request)
    assert exc.value.code == "unsupported_complex_metric"


def test_subprocess_protocol_and_native_readiness():
    p = launch(None, "--probe")
    assert p.returncode == 0, p.stderr
    lines = p.stdout.splitlines()
    assert len(lines) == 1
    cap = json.loads(lines[0])["capabilities"]
    assert cap["native_core_ready"] and cap["fresh_rhf"]["ready"]
    assert all(m["ready"] for m in cap["methods"].values()), cap
    assert cap["methods"]["ibo"]["periodic"] is False
    p = launch(fixture("h2"))
    assert p.returncode == 0, p.stderr
    events = [json.loads(s) for s in p.stdout.splitlines()]
    assert [e["event"] for e in events] == ["started", "result"]
    assert all(e["id"] == "h2" for e in events)
    assert not events[-1]["result"]["scf_performed"]


def test_broken_backend_probe_is_structured_and_redirects_native_stdout(tmp_path):
    # Isolate the standard-library launcher from all installed numerical code.
    fake = tmp_path / "vibeqc"
    fake.mkdir()
    (fake / "__init__.py").write_text(
        "import os\nprint('python diagnostic')\nos.write(1, b'native diagnostic\\n')\nraise ImportError('missing native core')\n"
    )
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            [str(tmp_path), str(Path(__file__).parents[1] / "python")]
        ),
    )
    p = subprocess.run(
        [sys.executable, "-S", "-m", "vibeqc_relocalize", "--probe"],
        env=env,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert p.returncode == 3
    cap = json.loads(p.stdout)["capabilities"]
    assert not cap["ready"] and not cap["native_core_ready"]
    assert not any(m["ready"] for m in cap["methods"].values())
    assert cap["error"]["code"] == "backend_unavailable"
    assert "native diagnostic" in p.stderr and "python diagnostic" in p.stderr


@pytest.mark.parametrize("line", ["{}", "[]", '{"x":NaN}', '{"x":1,"x":2}', "not json"])
def test_malformed_protocol(line):
    with pytest.raises(RequestError):
        parse_request(line)


@pytest.mark.parametrize("encoding", ["utf-16", "utf-16-le", "utf-32", "utf-32-be"])
def test_protocol_requires_utf8(encoding):
    request = {
        "protocol": "vibeqc.relocalize",
        "protocol_version": 1,
        "id": "probe-\u03bb",
        "operation": "capabilities",
    }
    payload = json.dumps(request, ensure_ascii=False)
    assert parse_request(payload.encode("utf-8")) == request
    with pytest.raises(RequestError) as exc:
        parse_request(payload.encode(encoding))
    assert exc.value.code == "invalid_json"


def test_invalid_utf8_has_structured_terminal_error():
    p = subprocess.run(
        [sys.executable, "-m", "vibeqc_relocalize"],
        input=b"\xff\n",
        capture_output=True,
        timeout=20,
        check=False,
    )
    assert p.returncode == 2
    terminal = json.loads(p.stdout)
    assert terminal["event"] == "error"
    assert terminal["id"] is None
    assert terminal["error"]["code"] == "invalid_json"


def test_structured_refusal_exit_code():
    request = fixture("h2")
    request.pop("orbitals")
    p = launch(request)
    assert p.returncode == 2
    terminal = json.loads(p.stdout.splitlines()[-1])
    assert (
        terminal["event"] == "error" and terminal["error"]["code"] == "invalid_source"
    )


@pytest.mark.parametrize("name", ["missing_orbitals", "periodic_ibo"])
def test_rejection_fixtures(name):
    p = launch(fixture(name))
    assert p.returncode == 2
    result = json.loads(p.stdout.splitlines()[-1])["error"]
    expected = json.loads((DATA / (name + ".error.json")).read_text())
    assert {k: result[k] for k in expected} == expected


def test_molecular_archive_is_sufficient_without_viewer(monkeypatch):
    import zipfile

    archive = (
        Path(__file__).parents[1] / "docs/_static/examples/h2o-rhf/output-h2o-rhf.qvf"
    )
    with zipfile.ZipFile(archive) as z:
        sections = {
            s["id"]: s["members"]
            for s in json.loads(z.read("manifest.json"))["sections"]
        }
        geom = json.loads(z.read(sections["structure"]["structure"]["path"]))
        basis = json.loads(z.read(sections["wf"]["basis"]["path"]))
        mo = json.loads(z.read(sections["wf"]["mo_metadata"]["path"]))
        ref = sections["wf"]["mo_coefficients"]
        c = np.frombuffer(z.read(ref["path"]), dtype="<f8").reshape(ref["shape"])
    request = fixture("h2")
    request["system"].update(
        atomic_numbers=[a["atomic_number"] for a in geom["atoms"]],
        positions_bohr=(
            np.array([a["position"] for a in geom["atoms"]]) / 0.529177210903
        ).tolist(),
    )
    request["basis"].update(name="archived-custom", shells=basis["shells"])
    occ = np.array(mo["occupations"])
    request["orbitals"] = {
        "coefficients": worker._encode(c[occ == 2]),
        "occupations": occ[occ == 2].tolist(),
    }
    monkeypatch.setattr(
        worker, "run_rhf", lambda *a: pytest.fail("Archive relocalization ran SCF")
    )
    result = worker.localize(request)
    assert len(result["occupations"]) == 5
    assert result["validation"]["orthonormality_error"] < 1e-8
    assert sum(result["charges"]) == pytest.approx(0, abs=1e-8)


def test_exact_qvf_cartesian_and_spherical_basis_roundtrip():
    from vibeqc import ShellInfo

    mol = Molecule([Atom(2, [0.0, 0.0, 0.0])], charge=0, multiplicity=1)
    native = [
        ShellInfo(0, ell, pure, [1.3, 0.4], [0.6, 0.2], [0.0, 0.0, 0.0])
        for ell, pure in [(0, True), (1, True), (2, False), (3, True)]
    ]
    basis = BasisSet(mol, native, "mixed", coefficients_pre_normalized=False)
    shells = _basis_shell_payload(basis)[0]
    spec = {"shells": shells, "uses_ecp": False, "ao_convention": worker.AO_CONVENTION}
    rebuilt = worker._basis(spec, mol, np.zeros((1, 3)), True)
    np.testing.assert_allclose(
        compute_overlap(rebuilt), compute_overlap(basis), atol=1e-14
    )
    assert [s.pure for s in rebuilt.shells()] == [True, True, False, True]


def test_subspace_violation_is_rejected(monkeypatch):
    from types import SimpleNamespace

    request = fixture("h2")
    # A real orthonormal orbital outside the occupied bonding space.
    mol = Molecule(
        [Atom(1, p) for p in request["system"]["positions_bohr"]],
        charge=0,
        multiplicity=1,
    )
    s = np.asarray(compute_overlap(BasisSet(mol, "sto-3g")))
    c = np.array([[1.0], [-1.0]])
    c /= np.sqrt((c.T @ s @ c)[0, 0])
    monkeypatch.setattr(
        worker, "analyse_localization", lambda *a, **kw: SimpleNamespace(coefficients=c)
    )
    with pytest.raises(RequestError) as exc:
        worker.localize(request)
    assert exc.value.code == "numerical_validation_failed"


@pytest.mark.parametrize("mesh,dim", [([3, 1, 1], 1), ([2, 2, 1], 2), ([2, 1, 2], 3)])
def test_complete_cyclic_meshes_and_dimensions(mesh, dim):
    # Deliberately decoupled-cell metric with complex Bloch gauges. This
    # isolates the Fourier contract from image summation/SCF approximations.
    request = fixture("periodic_gamma")
    p = request["periodic"]
    nk = int(np.prod(mesh))
    p.update(
        mesh=mesh,
        dimension=dim,
        pbc=[i < dim for i in range(3)],
        kpoints_fractional=(
            np.array(list(itertools.product(*(range(n) for n in mesh)))) / mesh
        ).tolist(),
        weights=[1 / nk] * nk,
        orbital_energies_hartree=[[-1.0, 0.5]] * nk,
    )
    s = decode(p["overlap"])[0]
    c = decode(request["orbitals"]["coefficients"])[0]
    p["overlap"] = worker._encode(np.array([s] * nk))
    request["orbitals"] = {
        "coefficients": worker._encode(
            np.array([np.exp(0.37j * i) * c for i in range(nk)])
        ),
        "occupations": [[2.0]] * nk,
    }
    result = worker.localize(request)
    assert len(result["occupations"]) == nk
    assert result["validation"]["unitary_error"] < 1e-8


def test_probe_reports_method_failure_without_enabling_it(monkeypatch):
    from vibeqc_relocalize import capabilities

    actual = worker.analyse_localization

    def missing_boys(*args, **kwargs):
        if kwargs.get("method") == "boys":
            raise RuntimeError("unavailable dipole witness")
        return actual(*args, **kwargs)

    monkeypatch.setattr(worker, "analyse_localization", missing_boys)
    report = capabilities()
    assert report["ready"] and report["native_core_ready"]
    assert report["methods"]["ibo"]["ready"]
    assert not report["methods"]["boys"]["ready"]
    assert report["methods"]["boys"]["readiness_error"] == "unavailable dipole witness"


def test_periodic_fresh_scf_is_never_dispatched(monkeypatch):
    request = fixture("periodic_gamma")
    request.pop("orbitals")
    request["source"] = "fresh_rhf"
    monkeypatch.setattr(
        worker, "run_rhf", lambda *a: pytest.fail("Periodic fresh SCF dispatched")
    )
    with pytest.raises(RequestError) as exc:
        worker.localize(request)
    assert exc.value.code == "unsupported_source"
