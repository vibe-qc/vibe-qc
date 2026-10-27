"""vibeqc.mlip — MACE model registry + ASL gate + options.

These are pure-Python (no torch / mace required), so they run on any
vibe-qc environment without the optional ``[mace]`` extra. The ASL gate deliberately fires *before* any
torch import, which is what makes it testable here.
"""

import builtins
import importlib
import tomllib
from pathlib import Path

import pytest
from vibeqc.mlip import (
    MACE_MODELS,
    MLIPOptions,
    mace_model_registry,
    resolve_model,
    save_mace_model_registry,
)
from vibeqc.mlip._mace_models import DEFAULT_MODEL


def test_missing_backend_explains_supported_installation(monkeypatch):
    from vibeqc.mlip import mace as mace_mod

    original_import = builtins.__import__

    def missing_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("missing test backend")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_torch)
    monkeypatch.setattr(mace_mod, "_check_mace_api_series", lambda: None)
    monkeypatch.setattr(mace_mod, "_maybe_set_openmp_workaround", lambda: None)
    with pytest.raises(ImportError, match="pip install.*vibe-qc") as caught:
        mace_mod._require_backends()
    assert isinstance(caught.value.__cause__, ImportError)
    assert "missing test backend" in str(caught.value.__cause__)
    assert "ACEsuit MACE" in str(caught.value)
    assert "mace-torch>=0.3.16,<0.4" in str(caught.value)
    assert "Mendoza Automated Crystal Engine" in str(caught.value)
    assert "Python <= 3.13" not in str(caught.value)


@pytest.mark.parametrize("installed", ["0.2.0", "0.4.0", "1.0.0rc1", "1.1.5"])
def test_incompatible_mace_api_fails_before_runtime_changes(monkeypatch, installed):
    import importlib.metadata as metadata
    from vibeqc.mlip import mace as mace_mod

    monkeypatch.setattr(metadata, "version", lambda name: installed)

    def unexpected_runtime_setup():
        pytest.fail("incompatible MACE reached OpenMP or backend setup")

    monkeypatch.setattr(mace_mod, "_maybe_set_openmp_workaround", unexpected_runtime_setup)
    with pytest.raises(ImportError, match="Unsupported ACEsuit mace-torch API") as caught:
        mace_mod._require_backends()
    assert installed in str(caught.value)
    assert "pip install 'mace-torch>=0.3.16,<0.4'" in str(caught.value)


@pytest.mark.parametrize("installed", ["0.3.16", "0.3.17.dev1", "0.3.17+local"])
def test_supported_mace_api_check_does_not_import_backends(monkeypatch, installed):
    import importlib.metadata as metadata
    from vibeqc.mlip import mace as mace_mod

    monkeypatch.setattr(metadata, "version", lambda name: installed)
    original_import = builtins.__import__

    def no_backends(name, *args, **kwargs):
        if name.partition(".")[0] in {"torch", "e3nn", "ase", "mace"}:
            pytest.fail(f"version check imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_backends)
    mace_mod._check_mace_api_series()


def test_unversioned_source_install_reaches_calculator_imports(monkeypatch):
    import importlib.metadata as metadata
    from types import SimpleNamespace
    from vibeqc.mlip import mace as mace_mod

    def no_metadata(name):
        raise metadata.PackageNotFoundError(name)

    stubs = {
        "torch": SimpleNamespace(),
        "e3nn": SimpleNamespace(),
        "ase": SimpleNamespace(Atoms=object),
        "ase.units": SimpleNamespace(Bohr=1.0, Hartree=2.0),
        "mace.calculators": SimpleNamespace(MACECalculator=object),
    }
    original_import = builtins.__import__

    def source_backends(name, *args, **kwargs):
        if name in stubs:
            return stubs[name]
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(metadata, "version", no_metadata)
    monkeypatch.setattr(mace_mod, "_maybe_set_openmp_workaround", lambda: None)
    monkeypatch.setattr(builtins, "__import__", source_backends)
    mp, off, torch, atoms, bohr, hartree = mace_mod._require_backends()
    assert callable(mp) and callable(off)
    assert torch is stubs["torch"]
    assert (atoms, bohr, hartree) == (object, 1.0, 2.0)


@pytest.mark.parametrize("missing", ["e3nn", "mace", "mace.calculators"])
def test_missing_mace_calculator_or_dependency_has_actionable_error(monkeypatch, missing):
    from types import SimpleNamespace
    from vibeqc.mlip import mace as mace_mod

    original_import = builtins.__import__
    failure = ModuleNotFoundError(f"No module named {missing!r}", name=missing)
    stubs = {
        "torch": SimpleNamespace(),
        "e3nn": SimpleNamespace(),
        "ase": SimpleNamespace(Atoms=object),
        "ase.units": SimpleNamespace(Bohr=1.0, Hartree=1.0),
    }

    def missing_backend(name, *args, **kwargs):
        if name == missing or (missing == "mace" and name.startswith("mace.")):
            raise failure
        if name in stubs:
            return stubs[name]
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(mace_mod, "_check_mace_api_series", lambda: None)
    monkeypatch.setattr(mace_mod, "_maybe_set_openmp_workaround", lambda: None)
    monkeypatch.setattr(builtins, "__import__", missing_backend)
    with pytest.raises(ImportError, match="ACEsuit MACE") as caught:
        mace_mod._require_backends()
    assert caught.value.__cause__ is failure
    assert "separate environment" in str(caught.value)


@pytest.mark.parametrize("python_version", [(3, 13, 0), (3, 14, 0), (3, 15, 0)])
def test_healthcheck_reports_backend_installation_error(monkeypatch, python_version):
    import runpy
    import sys
    from vibeqc.mlip import mace as mace_mod

    def missing_backend():
        raise ImportError("ACEsuit MACE test installation error")

    monkeypatch.setattr(mace_mod, "_require_backends", missing_backend)
    script = Path(__file__).resolve().parents[1] / "examples/mlip/00_runtime_healthcheck.py"
    healthcheck = runpy.run_path(str(script))
    monkeypatch.setattr(sys, "version_info", python_version)
    with pytest.raises(SystemExit, match="ACEsuit MACE test installation error") as caught:
        healthcheck["main"]([])
    assert isinstance(caught.value.__cause__, ImportError)


def test_healthcheck_help_without_optional_stack(monkeypatch, capsys):
    import runpy

    original_import = builtins.__import__

    def no_backends(name, *args, **kwargs):
        if name.partition(".")[0] in {"torch", "e3nn", "ase", "mace"}:
            pytest.fail(f"healthcheck help imported {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_backends)
    script = Path(__file__).resolve().parents[1] / "examples/mlip/00_runtime_healthcheck.py"
    healthcheck = runpy.run_path(str(script))
    with pytest.raises(SystemExit) as caught:
        healthcheck["main"](["--help"])
    assert caught.value.code == 0
    assert "--require-main-current" in capsys.readouterr().out


def test_default_model_is_mit_mpa0():
    info = resolve_model(None)
    assert info.key == DEFAULT_MODEL == "medium-mpa-0"
    assert info.loader == "mace_mp"
    assert info.license == "MIT"
    assert not info.is_academic_only
    assert info.citation == "batatia_mace_mp_2024"


def test_mlip_output_record_does_not_grant_academic_license(monkeypatch):
    from types import SimpleNamespace
    from vibeqc._qvf_options import hydrate_option
    from vibeqc.mlip.mace import enforce_academic_license
    from vibeqc.output.formats.mlip_record import mlip_job_options, mlip_record_fields

    result = SimpleNamespace(
        model_info=resolve_model("off23"), device="mps", dtype="float32",
        model_options=MLIPOptions(model="off23", accept_academic_license=True),
    )
    recorded = mlip_job_options(result)["mlip_options"]
    assert recorded == {"model": "off23-medium", "device": "mps", "dtype": "float32"}
    assert mlip_record_fields(result)["mlip"]["license"] == "ASL"
    monkeypatch.delenv("VIBEQC_ACCEPT_ASL", raising=False)
    restored = hydrate_option("mlip_options", recorded)
    with pytest.raises(PermissionError, match="Academic Software License"):
        enforce_academic_license(result.model_info, restored)
    assert mlip_job_options(SimpleNamespace()) == {}
    assert mlip_record_fields(SimpleNamespace()) == {}


def test_off23_is_asl_organic():
    info = resolve_model("off23-medium")
    assert info.loader == "mace_off"
    assert info.loader_arg == "medium"
    assert info.license == "ASL"
    assert info.is_academic_only
    assert info.citation == "kovacs_mace_off_2023"


@pytest.mark.parametrize(
    "alias,canonical",
    [("mpa-0", "medium-mpa-0"), ("MPA", "medium-mpa-0"),
     ("off23", "off23-medium"), ("off", "off23-medium")],
)
def test_aliases_resolve(alias, canonical):
    assert resolve_model(alias).key == canonical


def test_unknown_model_is_rejected_even_with_license_acknowledgment():
    # A license acknowledgment is not a back door for arbitrary URLs, local
    # paths, or newer upstream model families outside the reviewed registry.
    with pytest.raises(ValueError, match="Unsupported MACE model"):
        resolve_model("mace-omat-medium-2099")
    with pytest.raises(ValueError, match="Arbitrary URLs"):
        resolve_model("https://example.invalid/arbitrary.model")


def test_license_doc_matches_fail_closed_unknown_model_policy():
    license_doc = (
        Path(__file__).resolve().parents[1] / "docs" / "license.md"
    ).read_text(encoding="utf-8")
    words = " ".join(license_doc.split())

    assert "Unknown MACE model keys are treated as academic-only" not in words
    assert (
        "Unknown or unregistered MACE model keys are rejected before backend "
        "import or weight download."
    ) in words
    assert (
        "Accepting the ASL does not authorize an unregistered model family, "
        "URL, or local weight path."
    ) in words


def test_all_registered_models_have_known_citation_keys():
    for info in MACE_MODELS.values():
        assert info.citation in ("batatia_mace_mp_2024", "kovacs_mace_off_2023")


def test_registry_doi_is_the_citation_entry_version_of_record():
    """The per-model ``doi`` the .out provenance block prints must be the
    same DOI the .bibtex sibling emits for that model's citation entry.

    #523 moved the citation database to the journal versions of record; the
    registry still carried the arXiv preprint DOIs, so the .out said
    ``Reference: doi:10.48550/arXiv.2401.00096 (cited in the .bibtex
    sibling)`` while the sibling cited 10.1063/5.0297006.
    """
    from vibeqc.output.citations import load_default_database

    entries = load_default_database().entries()
    for info in MACE_MODELS.values():
        assert info.doi, info.key
        assert info.doi == entries[info.citation].doi, info.key
        assert not info.doi.lower().startswith("10.48550/arxiv"), info.key


def test_mace_model_registry_export_roundtrip(tmp_path):
    registry = mace_model_registry()
    assert registry["metadata"]["default_model"] == "medium-mpa-0"
    assert [row["key"] for row in registry["model"]] == list(MACE_MODELS)
    assert registry["aliases"]["off23"] == "off23-medium"

    path = tmp_path / "mace_models.toml"
    save_mace_model_registry(path)
    exported = tomllib.loads(path.read_text(encoding="utf-8"))

    assert exported["kind"] == "vibeqc.mace.model_registry"
    assert [row["key"] for row in exported["model"]] == list(MACE_MODELS)
    off23 = {row["key"]: row for row in exported["model"]}["off23-medium"]
    assert off23["academic_only"] is True
    assert off23["elements_z"] == [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
    assert "omat" not in {row["key"] for row in exported["model"]}


def test_options_ack_via_flag():
    assert MLIPOptions(accept_academic_license=True).academic_license_acknowledged()
    assert not MLIPOptions().academic_license_acknowledged()


def test_options_ack_via_env(monkeypatch):
    monkeypatch.setenv("VIBEQC_ACCEPT_ASL", "1")
    assert MLIPOptions().academic_license_acknowledged()
    monkeypatch.setenv("VIBEQC_ACCEPT_ASL", "0")
    assert not MLIPOptions().academic_license_acknowledged()
    monkeypatch.delenv("VIBEQC_ACCEPT_ASL", raising=False)
    assert not MLIPOptions().academic_license_acknowledged()


def test_mace_cache_root_honors_xdg(monkeypatch, tmp_path):
    from vibeqc.mlip.mace import mace_cache_root

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert mace_cache_root() == tmp_path / "mace"


def test_molecular_provenance_records_loader_runtime_and_cache(monkeypatch, tmp_path):
    from vibeqc.mlip import mace as mace_mod
    from vibeqc.runner import _format_mlip_provenance

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    text = _format_mlip_provenance(
        resolve_model("medium-mpa-0"),
        MLIPOptions(device="cpu", dtype="float64"),
    )
    assert "mace_mp(model='medium-mpa-0')" in text
    assert "device=cpu   dtype=float64" in text
    assert str(mace_mod.mace_cache_root()) in text


class _FakeMol:
    """Minimal stand-in; the ASL gate fires before the molecule is used."""

    atoms = ()
    charge = 0
    multiplicity = 1


def test_asl_model_gated_without_acknowledgment():
    # PermissionError is raised BEFORE importing torch/mace, so this works
    # without the [mace] extra installed.
    from vibeqc.mlip.mace import MACEModel

    with pytest.raises(PermissionError, match="Academic Software License"):
        MACEModel(_FakeMol(), MLIPOptions(model="off23-medium"))


def test_periodic_mace_evaluator_gates_asl_before_backend(monkeypatch):
    from vibeqc.mlip import mace as mace_mod

    def backend_must_not_load():
        raise AssertionError("ASL gate must fire before backend loading")

    monkeypatch.setattr(mace_mod, "_require_backends", backend_must_not_load)
    with pytest.raises(PermissionError, match="Academic Software License"):
        mace_mod.PeriodicMACEEvaluator(MLIPOptions(model="off23-medium"))


def test_mit_model_not_gated():
    # A MIT model clears the gate; the next step is the backend import,
    # which raises ImportError when [mace] is absent (the dev env). The
    # point: it is NOT a PermissionError.
    from vibeqc.mlip.mace import MACEModel

    try:
        import mace  # noqa: F401

        pytest.skip("mace installed; constructing the model would download weights")
    except ImportError:
        pass
    with pytest.raises(ImportError):
        # a valid one-atom molecule clears both the gate and the empty-input
        # guard, so the next step is the (absent) backend import
        MACEModel(_fake_mol([8]), MLIPOptions(model="medium-mpa-0"))


def test_mace_macos_openmp_workaround_caps_threads(monkeypatch):
    """macOS MACE runs need the duplicate-runtime flag and a conservative
    thread cap before torch imports."""
    from vibeqc.mlip import mace as mace_mod

    monkeypatch.setattr(mace_mod.sys, "platform", "darwin")
    monkeypatch.setattr(mace_mod, "_OPENMP_WORKAROUND_DONE", False)
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    monkeypatch.delenv("VIBEQC_MACE_OPENMP_THREADS", raising=False)
    monkeypatch.setenv("OMP_NUM_THREADS", "4")

    with pytest.warns(RuntimeWarning, match="KMP_DUPLICATE_LIB_OK"):
        mace_mod._maybe_set_openmp_workaround()

    assert mace_mod.os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    for name in mace_mod._MACE_THREAD_ENV_VARS:
        assert mace_mod.os.environ[name] == "1"


def test_mace_macos_openmp_workaround_honors_thread_override(monkeypatch):
    from vibeqc.mlip import mace as mace_mod

    monkeypatch.setattr(mace_mod.sys, "platform", "darwin")
    monkeypatch.setattr(mace_mod, "_OPENMP_WORKAROUND_DONE", False)
    monkeypatch.delenv("KMP_DUPLICATE_LIB_OK", raising=False)
    monkeypatch.setenv("VIBEQC_MACE_OPENMP_THREADS", "2")
    for name in mace_mod._MACE_THREAD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    with pytest.warns(RuntimeWarning, match="VIBEQC_MACE_OPENMP_THREADS"):
        mace_mod._maybe_set_openmp_workaround()

    assert mace_mod.os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"
    for name in mace_mod._MACE_THREAD_ENV_VARS:
        assert mace_mod.os.environ[name] == "2"


def test_banner_probes_mlip_stack():
    """CLAUDE.md § 6: the banner records the MLIP stack versions. The keys
    are always present (probed); "none" when the [mace] extra is absent
    (e.g. this 3.14 dev env). The ``mlip:`` line appears iff the stack is
    present, and ``banner()`` always renders."""
    from vibeqc.banner import banner, library_versions

    v = library_versions()
    for pkg in ("torch", "mace", "e3nn"):
        assert pkg in v, f"banner does not probe {pkg!r}"
    b = banner()
    assert isinstance(b, str) and b
    has_stack = any(v[p] != "none" for p in ("torch", "mace", "e3nn"))
    assert ("mlip:" in b) == has_stack


def test_banner_probes_mlip_versions_without_importing_runtime(monkeypatch):
    """Version reporting must not load Torch before the MACE adapter.

    On macOS, importing the optional runtime from the banner can initialize a
    conflicting OpenMP library before MACE installs its process-local guard.
    Distribution metadata supplies the same provenance without executing the
    heavy stack.
    """
    banner_mod = importlib.import_module("vibeqc.banner")
    expected = {
        "torch": "2.test",
        "mace-torch": "0.3.test",
        "e3nn": "0.4.test",
    }
    monkeypatch.setattr(banner_mod, "_pkg_version", expected.__getitem__)

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"torch", "mace", "e3nn"}:
            raise AssertionError(f"banner imported MLIP runtime {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    versions = banner_mod.library_versions()

    assert versions["torch"] == "2.test"
    assert versions["mace"] == "0.3.test"
    assert versions["e3nn"] == "0.4.test"


def test_mlip_import_chain_stays_backend_free(monkeypatch):
    """The mlip import chain must stay lazy for every non-MACE job.

    ``import vibeqc.mlip`` / ``import vibeqc.mlip.mace`` are the modules
    ``run_job`` touches at dispatch time (MLIPOptions + the early ASL gate);
    the heavy [mace] extra (torch / mace / e3nn) must only load inside
    ``_require_backends()`` at the point of use -- otherwise every RHF/DFT
    job would pay the ~7 s MACE cold-import chain (MACE-COLD-IMPORT).
    """
    import sys

    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name in {"torch", "mace", "e3nn"}:
            raise AssertionError(f"mlip import loaded MLIP runtime {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    for module_name in ("vibeqc.mlip", "vibeqc.mlip.mace"):
        sys.modules.pop(module_name, None)
        importlib.import_module(module_name)


# --- input validation: element coverage + charge/spin (hardening) ---


class _FakeAtom:
    def __init__(self, z):
        self.Z = int(z)


def _fake_mol(zs, charge=0, multiplicity=1):
    m = _FakeMol()
    m.atoms = [_FakeAtom(z) for z in zs]
    m.charge = charge
    m.multiplicity = multiplicity
    return m


def test_off23_covers_only_organic_elements():
    off = resolve_model("off23-medium")
    assert off.elements_z is not None
    assert off.unsupported_elements([1, 6, 7, 8]) == []          # H C N O
    assert off.unsupported_elements([1, 6, 26, 14]) == [14, 26]  # Si, Fe out


def test_materials_model_uses_exact_upstream_89_element_table():
    mp = resolve_model("medium-mpa-0")
    assert mp.elements_z is not None
    assert len(mp.elements_z) == 89
    assert mp.unsupported_elements([1, 26, 78, 92]) == []
    assert mp.unsupported_elements([84, 88, 95]) == [84, 88, 95]


def test_element_coverage_error_message():
    from vibeqc.mlip.mace import element_coverage_error

    off = resolve_model("off23-medium")
    assert element_coverage_error(off, _fake_mol([1, 8])) is None
    msg = element_coverage_error(off, _fake_mol([26, 8]))  # Fe
    assert msg is not None and "26" in msg
    assert element_coverage_error(resolve_model("medium-mpa-0"), _fake_mol([26, 8])) is None
    msg = element_coverage_error(resolve_model("medium-mpa-0"), _fake_mol([84]))
    assert msg is not None and "No registered vibe-qc MACE model" in msg


def test_validate_raises_on_out_of_domain_element():
    from vibeqc.mlip.mace import _validate_mlip_input

    with pytest.raises(ValueError, match="does not cover"):
        _validate_mlip_input(_fake_mol([26, 8]), resolve_model("off23-medium"))


def test_validate_warns_on_charge_or_spin():
    import warnings

    from vibeqc.mlip.mace import _validate_mlip_input

    mp = resolve_model("medium-mpa-0")

    def _warn_text(mol):
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            _validate_mlip_input(mol, mp)
        msgs = [str(w.message) for w in rec if issubclass(w.category, RuntimeWarning)]
        return msgs[0] if msgs else ""

    # The warning names only the non-default quantum number(s), so it never
    # redundantly reports "total charge (0)" for a neutral open-shell system.
    t = _warn_text(_fake_mol([8, 1, 1], charge=-1))  # charge only
    assert "total charge (-1)" in t and "multiplicity" not in t
    t = _warn_text(_fake_mol([8, 8], multiplicity=3))  # spin only
    assert "spin multiplicity (3)" in t and "total charge" not in t
    t = _warn_text(_fake_mol([8, 1], charge=1, multiplicity=2))  # both
    assert "total charge (1)" in t and "spin multiplicity (2)" in t
    with warnings.catch_warnings():  # neutral singlet -> silent
        warnings.simplefilter("error")
        _validate_mlip_input(_fake_mol([8, 1, 1]), mp)


def test_validate_rejects_empty_molecule():
    # An empty structure gets a clear, actionable error before the backend
    # (MACE's own forward pass would otherwise raise a cryptic
    # "zero-size array to reduction" deep in torch).
    from vibeqc.mlip.mace import _validate_mlip_input

    with pytest.raises(ValueError, match="at least one atom"):
        _validate_mlip_input(_fake_mol([]), resolve_model("medium-mpa-0"))


def test_run_job_mace_rejects_out_of_domain_element_fast(tmp_path):
    """run_job's early gate raises a clear element error before any torch /
    mace import — so this runs on the 3.14 dev env (no [mace] needed)."""
    from vibeqc import Atom, Molecule
    from vibeqc.runner import run_job

    feo = Molecule([Atom(26, [0.0, 0.0, 0.0]), Atom(8, [0.0, 0.0, 3.0])], 0, 1)
    with pytest.raises(ValueError, match="does not cover"):
        run_job(
            feo, method="mace",
            mlip_options=MLIPOptions(model="off23-medium", accept_academic_license=True),
            output=str(tmp_path / "x"),
        )


def test_mace_calculator_applies_gate_before_backend():
    """mace_calculator() applies the ASL gate (pure-Python) before the
    backend import: on the 3.14 dev env an ASL model without ack raises
    PermissionError; a MIT model raises ImportError (no [mace] installed)."""
    from vibeqc.mlip.mace import mace_calculator

    with pytest.raises(PermissionError, match="Academic Software License"):
        mace_calculator(MLIPOptions(model="off23-medium"))
    try:
        import mace  # noqa: F401

        pytest.skip("mace installed; mace_calculator would load a model")
    except ImportError:
        pass
    with pytest.raises(ImportError):
        mace_calculator(MLIPOptions(model="medium-mpa-0"))


def test_periodic_mace_evaluator_reuses_one_loaded_model(monkeypatch):
    """A fixed-model scan must load MACE once, not once per structure."""
    import numpy as np
    from vibeqc.mlip import mace as mace_mod

    loads = []

    class FakeCalculator:
        """Mimics the ASE Calculator contract MACEModel._compute uses:
        one calculate() call fills .results with all properties (stress in
        Voigt-6 form, like the upstream MACECalculator)."""

        def calculate(self, atoms=None, properties=None, system_changes=None):
            assert atoms is not None
            cell_trace = float(np.trace(atoms.cell))
            self.results = {
                "energy": float(np.sum(atoms.positions)),
                "forces": np.asarray(atoms.positions),
                # voigt-6 of diag(cell_trace, cell_trace, cell_trace)
                "stress": np.asarray(
                    [cell_trace, cell_trace, cell_trace, 0.0, 0.0, 0.0]
                ),
            }

    def fake_loader(**kwargs):
        loads.append(kwargs)
        return FakeCalculator()

    class FakeAtoms:
        def __init__(self, *, numbers, positions):
            self.numbers = list(numbers)
            self.positions = np.asarray(positions, dtype=float)
            self.cell = None
            self.pbc = False

        def set_cell(self, cell):
            self.cell = np.asarray(cell, dtype=float)

        def set_pbc(self, value):
            self.pbc = tuple(bool(v) for v in value)

    monkeypatch.setattr(
        mace_mod,
        "_require_backends",
        lambda: (fake_loader, fake_loader, object(), FakeAtoms, 0.5, 2.0),
    )

    class Atom:
        def __init__(self, z, xyz):
            self.Z = z
            self.xyz = xyz

    class Molecule:
        charge = 0
        multiplicity = 1

        def __init__(self, z_shift):
            self.atoms = [Atom(14, [z_shift, 0.0, 0.0])]

    class System:
        dim = 3

        def __init__(self, z_shift):
            self.lattice = np.diag([10.0 + z_shift, 10.0, 10.0])
            self._molecule = Molecule(z_shift)

        def unit_cell_molecule(self):
            return self._molecule

    evaluator = mace_mod.PeriodicMACEEvaluator()
    first = evaluator.run(System(1.0))
    second = evaluator.run(System(2.0))

    assert len(loads) == 1
    assert loads == [
        {"model": "medium-mpa-0", "device": "cpu", "default_dtype": "float64"}
    ]
    assert first.energy == pytest.approx(0.25)
    assert second.energy == pytest.approx(0.5)
    assert first.energy != second.energy
    assert first.gradient()[0, 0] == pytest.approx(-0.125)
    assert second.gradient()[0, 0] == pytest.approx(-0.25)
    assert first.pbc == second.pbc == (True, True, True)
    assert first.device == "cpu"
    assert first.dtype == "float64"
    assert first.loader == "mace_mp"


@pytest.mark.parametrize("positions", [
    [[float("nan"), 0.0, 0.0]],
    [[0.0, float("inf"), 0.0]],
    [[0.0, 0.0]],
])
def test_invalid_mace_coordinates_precede_backend(monkeypatch, positions):
    from types import SimpleNamespace
    from vibeqc.mlip import mace as mace_mod

    molecule = SimpleNamespace(
        atoms=[SimpleNamespace(Z=1, xyz=xyz) for xyz in positions],
        charge=0, multiplicity=1,
    )
    monkeypatch.setattr(mace_mod, "_require_backends",
                        lambda: pytest.fail("Invalid geometry loaded the backend"))
    with pytest.raises(ValueError, match="MACE positions.*finite.*shape"):
        mace_mod.MACEModel(molecule)


@pytest.mark.parametrize("cell,pbc", [
    ([10.0, 10.0, 10.0], None),
    ([[10.0, 0.0, 0.0], [0.0, float("nan"), 0.0], [0.0, 0.0, 10.0]], None),
    ([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, float("inf")]], None),
])
def test_invalid_mace_cell_precedes_backend(monkeypatch, cell, pbc):
    from types import SimpleNamespace
    from vibeqc.mlip import mace as mace_mod

    molecule = SimpleNamespace(atoms=[SimpleNamespace(Z=14, xyz=[0., 0., 0.])])
    monkeypatch.setattr(mace_mod, "_require_backends",
                        lambda: pytest.fail("Invalid cell loaded the backend"))
    with pytest.raises(ValueError, match="MACE cell.*finite.*shape"):
        mace_mod.MACEModel(molecule, cell=cell, pbc=pbc)


@pytest.mark.parametrize("cell,pbc", [
    ([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], None),
    ([[4., 8., 0.], [0., 0., 0.], [0., 0., 6.]], None),
    ([[4., 8., 0.], [0., 0., 0.], [0., 0., 6.]], (True, True, False)),
    ([[0., 0., 0.], [0., 5., 0.], [0., 0., 6.]], (True, False, False)),
    ([[4., 0., 8.], [0., 5., 0.], [0., 0., 0.]], (True, False, True)),
])
def test_degenerate_mace_periodic_cell_precedes_backend(monkeypatch, cell, pbc):
    """A backend must not invent periodic translations for a singular cell."""
    from types import SimpleNamespace
    from vibeqc.mlip import mace as mace_mod

    molecule = SimpleNamespace(atoms=[SimpleNamespace(Z=14, xyz=[0., 0., 0.])])
    monkeypatch.setattr(mace_mod, "_require_backends",
                        lambda: pytest.fail("Degenerate cell reached backend loading"))
    with pytest.raises(ValueError, match="periodic lattice vectors.*nonzero.*independent"):
        mace_mod.MACEModel(molecule, cell=cell, pbc=pbc)


@pytest.mark.parametrize("cell,pbc", [
    ([[-4., 1., 0.], [0., 5., 1.], [0., 0., 6.]], None),
    ([[4., 1., 0.], [0., 5., 0.], [0., 0., 0.]], (True, True, False)),
    ([[4., 0., 0.], [0., 0., 0.], [0., 0., 0.]], (True, False, False)),
    ([[0., 0., 4.], [3., 0., 0.], [0., 0., 0.]], (True, False, True)),
    ([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]], (False, False, False)),
    ([[1e-200, 0., 0.], [0., 1e200, 0.], [0., 0., 1.]], None),
])
def test_mace_cell_rank_uses_only_periodic_columns(backend_results, cell, pbc):
    """Cell admission is independent of handedness and inactive bookkeeping axes."""
    import numpy as np
    from vibeqc.mlip import mace as mace_mod

    _, template = backend_results
    model = mace_mod.MACEModel(template._mol, cell=cell, pbc=pbc)
    np.testing.assert_array_equal(model._cell, cell)
    assert model.pbc == ((True, True, True) if pbc is None else pbc)


def test_degenerate_mace_update_preserves_cached_structure(backend_results):
    import numpy as np

    calc, model = backend_results
    energy, gradient, stress = model.energy(), model.gradient(), model.stress()
    cell, pbc = model._cell.copy(), model.pbc
    with pytest.raises(ValueError, match="periodic lattice vectors"):
        model._set_structure(model._mol, cell=np.zeros((3, 3)))
    np.testing.assert_array_equal(model._cell, cell)
    assert model.pbc == pbc
    assert model.energy() == energy
    np.testing.assert_array_equal(model.gradient(), gradient)
    np.testing.assert_array_equal(model.stress(), stress)
    assert calc.calls == 1


@pytest.mark.parametrize("pbc", [
    (True, True), (True, True, True, False), "abc",
    (True, False, "False"), (True, False, 2), (True, False, None),
])
def test_invalid_mace_pbc_precedes_backend(monkeypatch, pbc):
    from types import SimpleNamespace
    import numpy as np
    from vibeqc.mlip import mace as mace_mod

    molecule = SimpleNamespace(atoms=[SimpleNamespace(Z=14, xyz=[0., 0., 0.])])
    monkeypatch.setattr(mace_mod, "_require_backends",
                        lambda: pytest.fail("Invalid PBC loaded the backend"))
    with pytest.raises(ValueError, match="pbc must contain three boolean flags"):
        mace_mod.MACEModel(molecule, cell=np.eye(3) * 10., pbc=pbc)


def test_rejected_mace_structure_update_preserves_cached_structure(backend_results):
    from types import SimpleNamespace
    import numpy as np

    calc, model = backend_results
    energy, gradient = model.energy(), model.gradient()
    molecule, cell, pbc = model._mol, model._cell.copy(), model.pbc
    replacement = SimpleNamespace(atoms=[SimpleNamespace(Z=14, xyz=[3., 2., 1.])])
    with pytest.raises(ValueError, match="pbc requires"):
        model._set_structure(replacement, pbc=(True, True, True))
    assert model._mol is molecule
    np.testing.assert_array_equal(model._cell, cell)
    assert model.pbc == pbc
    assert model.energy() == energy
    np.testing.assert_array_equal(model.gradient(), gradient)
    assert calc.calls == 1


@pytest.mark.parametrize("pbc", [(True, False, False), (1, 1, 0), (True, True, True)])
def test_mace_cell_snapshot_and_valid_pbc(backend_results, pbc):
    import numpy as np
    from vibeqc.mlip import mace as mace_mod

    _, template = backend_results
    cell = np.array([[10., 1., 0.], [0., 11., 2.], [0., 0., 12.]])
    expected = cell.copy()
    model = mace_mod.MACEModel(template._mol, cell=cell, pbc=pbc)
    cell[0, 0] = 999.
    np.testing.assert_array_equal(model._cell, expected)
    np.testing.assert_allclose(model._ase_atoms().cell, expected.T * model._Bohr)
    assert model.pbc == tuple(bool(flag) for flag in pbc)


@pytest.fixture
def backend_results(monkeypatch):
    """An ASE-shaped backend with injectable results, without model weights."""
    from types import SimpleNamespace

    import numpy as np
    from ase import Atoms
    from ase.units import Bohr, Hartree
    from vibeqc.mlip import mace as mace_mod

    class Calculator:
        calls = 0

        def calculate(self, atoms, properties):
            self.calls += 1

    calc = Calculator()
    calc.results = {
        "energy": -2.0,
        "forces": np.array([[1., 2., 3.], [-1., -2., -3.]]),
        "stress": np.arange(6, dtype=float),
    }
    monkeypatch.setattr(
        mace_mod, "_require_backends",
        lambda: (lambda **kw: calc, lambda **kw: calc, None, Atoms, Bohr, Hartree),
    )
    mol = SimpleNamespace(
        atoms=[SimpleNamespace(Z=1, xyz=[0., 0., 0.]),
               SimpleNamespace(Z=1, xyz=[1.4, 0., 0.])],
        charge=0, multiplicity=1,
    )
    return calc, mace_mod.MACEModel(mol, cell=np.eye(3) * 10.)


@pytest.mark.parametrize("mutation", ["caller", "property", "backend_import"])
def test_mace_provenance_retains_loaded_runtime(backend_results, monkeypatch, mutation):
    """Reusing mutable configuration must not relabel an already loaded model."""
    from vibeqc.mlip import mace as mace_mod
    from vibeqc.runner import _format_mlip_provenance

    calc, template = backend_results
    options = MLIPOptions()
    loads = []
    backends = mace_mod._require_backends()

    def change(configuration):
        configuration.model = "small"
        configuration.device = "cuda"
        configuration.dtype = "float32"

    def loader(**kwargs):
        loads.append(kwargs)
        return calc

    def require_backends():
        if mutation == "backend_import":
            change(options)
        return (loader, loader, *backends[2:])

    monkeypatch.setattr(mace_mod, "_require_backends", require_backends)
    model = mace_mod.MACEModel(template._mol, options, cell=template._cell)
    if mutation == "caller":
        change(options)
    elif mutation == "property":
        change(model.options)

    result = mace_mod._periodic_result(model)
    assert loads == [{"model": "medium-mpa-0", "device": "cpu",
                      "default_dtype": "float64"}]
    assert result.model_info.key == model.options.model == "medium-mpa-0"
    assert result.device == "cpu"
    assert result.dtype == "float64"
    text = _format_mlip_provenance(model.info, model.options)
    assert "device=cpu" in text
    assert "dtype=float64" in text


@pytest.mark.parametrize("entry", ["model", "calculator"])
def test_mace_mps_converts_checkpoint_before_device_transfer(backend_results, monkeypatch, entry):
    """A float64 checkpoint cannot be deserialized directly onto MPS."""
    import sys
    from types import ModuleType, SimpleNamespace
    from vibeqc.mlip import mace as mace_mod
    from vibeqc.mlip._mace_checkpoint import CheckpointIdentity

    calc, template = backend_results
    backends = mace_mod._require_backends()
    events = []

    class Weights:
        dtype = "float64"

        def float(self):
            events.append("convert on cpu")
            self.dtype = "float32"
            return self

    weights = Weights()
    weights._vibeqc_checkpoint_identity = CheckpointIdentity(
        "a" * 64, 123, "https://example.invalid/model",
    )

    def loader(**kwargs):
        if kwargs["device"] != "cpu":
            raise TypeError("MPS cannot deserialize float64 checkpoint tensors")
        assert kwargs["return_raw_model"] is True
        assert kwargs["model"] == "medium-mpa-0"
        assert kwargs["expected_sha256"] == "a" * 64
        events.append("load on cpu")
        return weights

    def calculator(*, models, device, default_dtype):
        assert models is weights and models.dtype == "float32"
        assert device == "mps" and default_dtype == "float32"
        events.append("construct on mps")
        return calc

    package, calculators = ModuleType("mace"), ModuleType("mace.calculators")
    calculators.MACECalculator = calculator
    monkeypatch.setitem(sys.modules, "mace", package)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators)
    torch = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setattr(mace_mod, "_require_backends",
                        lambda: (loader, loader, torch, *backends[3:]))
    options = MLIPOptions(device="mps", dtype="float32", weight_sha256="a" * 64)
    if entry == "model":
        model = mace_mod.MACEModel(template._mol, options)
        assert model.calculator is calc
        assert model.options.device == "mps"
        assert model.weight_identity["sha256"] == "a" * 64
    else:
        assert mace_mod.mace_calculator(options) is calc
    assert events == ["load on cpu", "convert on cpu", "construct on mps"]
    assert calc._vibeqc_checkpoint_identity is weights._vibeqc_checkpoint_identity


@pytest.mark.parametrize("entry", ["model", "calculator", "evaluator"])
@pytest.mark.parametrize("device,dtype", [("mps", "float64"), ("cpu", "float16"), ("cpu", "")])
def test_mace_rejects_unsupported_precision_before_backend(backend_results, monkeypatch, entry, device, dtype):
    from vibeqc.mlip import mace as mace_mod

    _, template = backend_results

    def unexpected_backend():
        raise AssertionError("invalid precision reached backend import")

    monkeypatch.setattr(mace_mod, "_require_backends", unexpected_backend)
    options = MLIPOptions(device=device, dtype=dtype)
    with pytest.raises(ValueError, match="dtype"):
        if entry == "model":
            mace_mod.MACEModel(template._mol, options)
        elif entry == "calculator":
            mace_mod.mace_calculator(options)
        else:
            mace_mod.PeriodicMACEEvaluator(options)


@pytest.mark.parametrize("entry", ["model", "calculator"])
def test_mace_unavailable_mps_fails_before_weight_loading(backend_results, monkeypatch, entry):
    from types import SimpleNamespace
    from vibeqc.mlip import mace as mace_mod

    _, template = backend_results
    backends = mace_mod._require_backends()

    def unexpected_loader(**kwargs):
        raise AssertionError("unavailable MPS reached weight loading")

    torch = SimpleNamespace(backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setattr(mace_mod, "_require_backends",
                        lambda: (unexpected_loader, unexpected_loader, torch, *backends[3:]))
    options = MLIPOptions(device="mps", dtype="float32")
    with pytest.raises(RuntimeError, match="no available MPS device"):
        if entry == "model":
            mace_mod.MACEModel(template._mol, options)
        else:
            mace_mod.mace_calculator(options)


@pytest.mark.parametrize("property,value", [
    ("energy", float("nan")), ("energy", float("inf")),
    ("energy", [-2.0]), ("energy", None),
    ("forces", [[1., 2., float("nan")], [0., 0., 0.]]),
    ("forces", [[1., 2., 3.]]), ("forces", [[1., 2.], [3., 4.], [5., 6.]]),
    ("stress", [0., 0., 0., 0., 0., float("inf")]),
    ("stress", [0., 0., 0.]),
])
def test_mace_rejects_invalid_results_and_retries_without_partial_cache(
    backend_results, property, value,
):
    """Even energy() must refuse an incomplete energy/force/stress result."""
    from ase.units import Bohr, Hartree
    import numpy as np

    calc, model = backend_results
    valid = calc.results[property]
    calc.results[property] = value
    with pytest.raises(ValueError, match=f"MACE.*{property}"):
        model.energy()
    # A failed force/stress read must not leave an accepted energy behind.
    calc.results[property] = valid
    assert model.energy() == pytest.approx(-2. / Hartree)
    np.testing.assert_allclose(model.gradient(), -calc.results["forces"] * Bohr / Hartree)
    assert model.stress().shape == (3, 3)
    assert calc.calls == 2


@pytest.mark.parametrize("property", ["energy", "forces", "stress"])
def test_mace_rejects_missing_backend_results(backend_results, property):
    calc, model = backend_results
    del calc.results[property]
    with pytest.raises(ValueError, match=f"MACE.*{property}"):
        model.energy()


def test_mace_cached_result_does_not_alias_backend_buffers(backend_results):
    import numpy as np

    calc, model = backend_results
    energy, gradient, stress = model.energy(), model.gradient(), model.stress()
    calc.results["forces"][:] = 100.
    calc.results["stress"][:] = 100.
    assert model.energy() == energy
    np.testing.assert_array_equal(model.gradient(), gradient)
    np.testing.assert_array_equal(model.stress(), stress)
    assert calc.calls == 1


@pytest.mark.parametrize(
    "dim,expected",
    [
        (1, (True, False, False)),
        (2, (True, True, False)),
        (3, (True, True, True)),
    ],
)
def test_periodic_mace_preserves_dimensional_pbc(monkeypatch, dim, expected):
    """The direct wrapper must not turn a true slab into a 3D repeat."""
    import numpy as np
    from vibeqc.mlip import mace as mace_mod

    pbc_seen = []
    properties_seen = []

    class FakeCalculator:
        def calculate(self, atoms=None, properties=None, system_changes=None):
            properties_seen.append(list(properties or []))
            self.results = {
                "energy": 1.0,
                "forces": np.zeros((len(atoms.positions), 3)),
                "stress": np.zeros(6),
            }

    class FakeAtoms:
        def __init__(self, *, numbers, positions):
            self.positions = np.asarray(positions, dtype=float)

        def set_cell(self, _cell):
            pass

        def set_pbc(self, value):
            pbc_seen.append(tuple(bool(v) for v in value))

    monkeypatch.setattr(
        mace_mod,
        "_require_backends",
        lambda: (
            lambda **_kwargs: FakeCalculator(),
            lambda **_kwargs: FakeCalculator(),
            object(),
            FakeAtoms,
            0.5,
            2.0,
        ),
    )

    class Atom:
        Z = 14
        xyz = [0.0, 0.0, 0.0]

    class Molecule:
        atoms = [Atom()]
        charge = 0
        multiplicity = 1

    class System:
        lattice = np.eye(3) * 10.0

        def __init__(self):
            self.dim = dim

        def unit_cell_molecule(self):
            return Molecule()

    result = mace_mod.run_periodic_mace(System())
    assert pbc_seen == [expected]
    assert result.pbc == expected
    assert result.dim == dim
    if dim == 3:
        assert result.stress().shape == (3, 3)
        assert all("stress" in props for props in properties_seen)
    else:
        assert all("stress" not in props for props in properties_seen)
        with pytest.raises(ValueError, match="only for dim=3"):
            result.stress()


def test_periodic_cell_optimization_rejects_true_slab_before_backend(monkeypatch):
    from vibeqc.mlip import mace as mace_mod

    class Slab:
        dim = 2

    monkeypatch.setattr(
        mace_mod,
        "MACEModel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("backend must not load")
        ),
    )
    with pytest.raises(ValueError, match="optimize_periodic_mace_positions"):
        mace_mod.optimize_periodic_mace_cell(Slab())


@pytest.mark.parametrize(
    "max_steps",
    [
        True,
        False,
        1.0,
        1.9,
        -0.5,
        "2",
        None,
        float("nan"),
        float("inf"),
        -float("inf"),
    ],
)
@pytest.mark.parametrize("driver", ["optimize_periodic_mace_positions", "optimize_periodic_mace_cell"])
def test_periodic_position_optimization_rejects_noninteger_step_budgets_before_backend(
    monkeypatch, max_steps, driver
):
    from vibeqc.mlip import mace as mace_mod

    class System:
        def unit_cell_molecule(self):
            raise AssertionError("backend work must not start")

    monkeypatch.setattr(
        mace_mod,
        "MACEModel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("backend must not load")
        ),
    )
    with pytest.raises(ValueError, match="max_steps must be a non-negative integer"):
        getattr(mace_mod, driver)(System(), max_steps=max_steps)


@pytest.mark.parametrize("driver", ["optimize_periodic_mace_positions", "optimize_periodic_mace_cell"])
@pytest.mark.parametrize("fmax", [float("nan"), float("inf"), -float("inf"), 0., -1., True, "0.05", None])
def test_periodic_mace_rejects_invalid_force_targets_before_backend(driver, fmax):
    from vibeqc.mlip import mace as mace_mod

    # No structure inspection, backend import or model loading is needed to
    # reject these controls. In particular, +inf must not certify convergence.
    with pytest.raises(ValueError, match="fmax must be finite and positive"):
        getattr(mace_mod, driver)(object(), fmax=fmax)


@pytest.mark.parametrize("max_steps", [0, 2, pytest.param("numpy", id="numpy-int")])
def test_periodic_position_optimization_forwards_integer_step_budgets(
    monkeypatch, max_steps
):
    import ase.optimize
    import numpy as np

    import vibeqc as vq
    from vibeqc.mlip import mace as mace_mod

    if max_steps == "numpy":
        max_steps = np.int64(3)

    seen_steps = []

    class FakeAtoms:
        numbers = np.asarray([1])
        positions = np.zeros((1, 3))
        calc = None

        def set_constraint(self, _constraint):
            raise AssertionError("no fixed atom constraint expected")

        def get_forces(self, apply_constraint=True):
            return np.zeros((1, 3))

        def get_potential_energy(self):
            return 0.0

    class FakeModel:
        _Bohr = 1.0
        _Hartree = 1.0
        calculator = object()
        info = resolve_model("medium-mpa-0")
        options = MLIPOptions()
        weight_identity = {}

        def __init__(self, _molecule, _options, cell, *, pbc):
            assert cell.shape == (3, 3)
            self.pbc = pbc

        def _ase_atoms(self):
            return FakeAtoms()

    class FakeBFGS:
        nsteps = 0

        def __init__(self, _atoms, *, logfile):
            assert logfile is None

        def run(self, *, fmax, steps):
            assert fmax == 0.05
            seen_steps.append(steps)
            return True

    monkeypatch.setattr(mace_mod, "MACEModel", FakeModel)
    monkeypatch.setattr(ase.optimize, "BFGS", FakeBFGS)
    system = vq.PeriodicSystem(
        3,
        np.eye(3) * 8.0,
        [vq.Atom(1, [0.0, 0.0, 0.0])],
    )

    result = mace_mod.optimize_periodic_mace_positions(
        system, max_steps=max_steps
    )

    assert seen_steps == [int(max_steps)]
    assert result.n_steps == 0


@pytest.mark.parametrize(
    "log_volume_offset,max_steps,should_converge",
    [(0.5, 0, False), (0.5, 1, False), (0.0, 0, True), (0.5, 80, True)],
)
@pytest.mark.parametrize("charge,multiplicity", [(0, 1), (1, 2), (0, 3)])
def test_periodic_cell_optimization_requires_filter_convergence(
    monkeypatch, log_volume_offset, max_steps, should_converge, charge, multiplicity
):
    """Zero atomic forces do not imply a relaxed cell with nonzero stress.

    Exercise real ASE BFGS/FrechetCellFilter on E = log(V/V0)^2 / 2 eV.
    Its exact stress is log(V/V0) / V times the identity in eV/Angstrom^3.
    This is an optimizer contract control, not a MACE accuracy reference.
    """
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.filters import FrechetCellFilter
    from ase.units import Bohr, Hartree

    import vibeqc as vq
    from vibeqc.mlip import mace as mace_mod

    lattice = np.array([[6.0, 0.8, 0.2], [0.0, 5.0, 0.5], [0.0, 0.0, 4.0]])
    target_volume = np.linalg.det(lattice * Bohr) / np.exp(log_volume_offset)

    class VolumeCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            volume = atoms.get_volume()
            offset = np.log(volume / target_volume)
            self.results = {
                "energy": 0.5 * offset**2,
                "forces": np.zeros((len(atoms), 3)),
                "stress": offset / volume * np.array([1., 1., 1., 0., 0., 0.]),
            }

    def loader(**kwargs):
        assert kwargs == {
            "model": "medium-mpa-0", "device": "cpu", "default_dtype": "float64"
        }
        return VolumeCalculator()

    monkeypatch.setattr(
        mace_mod, "_require_backends",
        lambda: (loader, loader, object(), Atoms, Bohr, Hartree),
    )
    system = vq.PeriodicSystem(
        3, lattice, [vq.Atom(14, [0.0, 0.0, 0.0])],
        charge=charge, multiplicity=multiplicity,
    )
    if should_converge:
        final = mace_mod.optimize_periodic_mace_cell(
            system, fmax=0.01, max_steps=max_steps
        )
    else:
        with pytest.raises(RuntimeError, match="did not converge") as caught:
            mace_mod.optimize_periodic_mace_cell(
                system, fmax=0.01, max_steps=max_steps
            )
        error = caught.value
        assert error.n_steps == max_steps
        assert error.max_steps == max_steps
        assert error.target_max_force == 0.01
        final = error.system

    # Recompute both parts of ASE's convergence criterion independently of
    # the wrapper, including the cell degrees of freedom, at the terminal cell.
    model = mace_mod.MACEModel(final.unit_cell_molecule(), cell=final.lattice)
    atoms = model._ase_atoms()
    atoms.calc = model.calculator
    assert np.array_equal(atoms.get_forces(), np.zeros((1, 3)))
    residual = np.linalg.norm(FrechetCellFilter(atoms).get_forces(), axis=1).max()
    assert (residual < 0.01) == should_converge
    if not should_converge:
        assert error.final_max_force == pytest.approx(residual, abs=1e-12)
    assert np.array_equal(system.lattice, lattice)
    assert final.dim == 3
    assert [atom.Z for atom in final.unit_cell] == [14]
    # MACE ignores these labels, but the returned geometry is also an input
    # to later electronic-structure jobs. Preserve them on both success and
    # the last structure attached to a budget-exhaustion exception.
    assert final.charge == system.charge == charge
    assert final.multiplicity == system.multiplicity == multiplicity


def test_periodic_position_optimization_preserves_slab_and_fixed_atoms(monkeypatch):
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes

    import vibeqc as vq
    from vibeqc.mlip import mace as mace_mod

    class HarmonicCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            positions = np.asarray(atoms.positions, dtype=float)
            self.results = {
                "energy": 0.5 * float(np.sum(positions**2)),
                "forces": -positions,
            }

    pbc_seen = []

    class FakeModel:
        _Bohr = 0.5
        _Hartree = 2.0
        weight_identity = {}

        def __init__(self, molecule, options, cell, *, pbc):
            self.molecule = molecule
            self.options = options or MLIPOptions()
            self.info = resolve_model(self.options.model)
            self.cell = np.asarray(cell, dtype=float)
            self.pbc = tuple(pbc)
            self.calculator = HarmonicCalculator()
            pbc_seen.append(self.pbc)

        def _ase_atoms(self):
            return Atoms(
                numbers=[atom.Z for atom in self.molecule.atoms],
                positions=np.asarray(
                    [atom.xyz for atom in self.molecule.atoms], dtype=float
                )
                * self._Bohr,
                cell=self.cell.T * self._Bohr,
                pbc=self.pbc,
            )

    monkeypatch.setattr(mace_mod, "MACEModel", FakeModel)
    lattice = np.diag([8.0, 8.0, 30.0])
    slab = vq.PeriodicSystem(
        2,
        lattice,
        [vq.Atom(14, [1.0, 0.0, 0.0]), vq.Atom(14, [4.0, 0.0, 0.0])],
    )
    result = mace_mod.optimize_periodic_mace_positions(
        slab,
        fmax=0.01,
        max_steps=20,
        fixed_indices=(0,),
    )

    coords = np.asarray([atom.xyz for atom in result.system.unit_cell])
    assert pbc_seen == [(True, True, False)]
    assert result.system.dim == 2
    assert np.array_equal(result.system.lattice, lattice)
    assert coords[0] == pytest.approx([1.0, 0.0, 0.0])
    assert abs(coords[1, 0]) < 4.0
    assert result.fixed_indices == (0,)
    assert result.max_force_eva <= 0.01
    assert result.converged
    assert result.gradient().shape == (2, 3)
    # The physical energy derivative includes forces on frozen atoms. Only
    # the stopping residual is projected by ASE's position constraints.
    np.testing.assert_allclose(result.gradient(), coords * 0.5**2 / 2.0)
    assert result.gradient()[0, 0] != 0.0
    assert result.max_force_eva == pytest.approx(abs(coords[1, 0]) * 0.5)


@pytest.mark.parametrize("index,error,match", [
    (1, IndexError, "out-of-range"), (-1, IndexError, "out-of-range"),
    (0.5, ValueError, "integer"), ("0", ValueError, "integer"),
    (True, ValueError, "integer"),
])
def test_periodic_position_optimization_validates_fixed_indices(
    monkeypatch, index, error, match,
):
    import numpy as np
    import vibeqc as vq
    from vibeqc.mlip import mace as mace_mod

    system = vq.PeriodicSystem(
        2,
        np.diag([8.0, 8.0, 30.0]),
        [vq.Atom(14, [0.0, 0.0, 0.0])],
    )
    monkeypatch.setattr(
        mace_mod,
        "MACEModel",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("backend must not load")
        ),
    )
    with pytest.raises(error, match=match):
        mace_mod.optimize_periodic_mace_positions(system, fixed_indices=(index,))


@pytest.mark.parametrize("optimizer,property", [
    ("positions", "energy"), ("positions", "forces"),
    ("cell", "energy"), ("cell", "forces"), ("cell", "stress"),
])
def test_periodic_optimization_rejects_nonfinite_terminal_result(
    monkeypatch, optimizer, property,
):
    """Zero forces must not turn a NaN energy into an accepted relaxation."""
    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.units import Bohr, Hartree

    import vibeqc as vq
    from vibeqc.mlip import mace as mace_mod

    class InvalidCalculator(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "energy": 0.0, "forces": np.zeros((len(atoms), 3)),
                "stress": np.zeros(6),
            }
            self.results[property] = np.full_like(self.results[property], float("nan"))

    monkeypatch.setattr(
        mace_mod, "_require_backends",
        lambda: (lambda **kw: InvalidCalculator(), None, None, Atoms, Bohr, Hartree),
    )
    system = vq.PeriodicSystem(3, np.eye(3) * 10., [vq.Atom(14, [0., 0., 0.])])
    optimize = getattr(mace_mod, f"optimize_periodic_mace_{optimizer}")
    with pytest.raises(ValueError, match=f"MACE.*{property}"):
        optimize(system, max_steps=0)


@pytest.fixture
def checkpoint_backend(monkeypatch, tmp_path):
    """Model deserialization is observable without torch or any real weights."""
    import sys
    from types import SimpleNamespace
    from vibeqc.mlip import _mace_checkpoint as checkpoint

    path = tmp_path / "registered.model"
    path.write_bytes(b"original checkpoint bytes")
    consumed = []

    def deserialize(stream, **kwargs):
        consumed.append((stream.read(), kwargs))
        # Simulate a cache replacement during deserialization. Identity must
        # describe the consumed snapshot, not this newer file.
        path.write_bytes(b"replacement checkpoint bytes")
        return SimpleNamespace()

    monkeypatch.setattr(checkpoint, "_checkpoint_path", lambda info: path)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=deserialize))
    monkeypatch.setitem(sys.modules, "mace.calculators", SimpleNamespace(
        MACECalculator=lambda **kwargs: SimpleNamespace(**kwargs),
    ))
    return checkpoint, path, consumed


@pytest.mark.parametrize("raw", [False, True])
def test_checkpoint_identity_binds_deserialized_snapshot(checkpoint_backend, raw):
    import hashlib
    from dataclasses import FrozenInstanceError

    checkpoint, path, consumed = checkpoint_backend
    original = path.read_bytes()
    expected = hashlib.sha256(original).hexdigest()
    loaded = checkpoint.load_mp(model="small", device="cpu", default_dtype="float64",
                                return_raw_model=raw, expected_sha256=expected.upper())
    assert consumed == [(original, {"map_location": "cpu", "weights_only": False})]
    assert path.read_bytes() != original
    identity = loaded._vibeqc_checkpoint_identity
    assert identity.sha256 == expected
    assert identity.size_bytes == len(original)
    assert identity.registry_url == checkpoint._CHECKPOINT_URLS["small"]
    if not raw:
        assert loaded.models._vibeqc_checkpoint_identity is identity
        assert loaded.default_dtype == "float64"
    with pytest.raises(FrozenInstanceError):
        identity.sha256 = "0" * 64


@pytest.mark.parametrize("payload,expected,match", [
    (b"", None, "empty"), (b"changed bytes", "0" * 64, "SHA256"),
])
def test_checkpoint_failure_precedes_deserialization(checkpoint_backend, payload, expected, match):
    checkpoint, path, consumed = checkpoint_backend
    path.write_bytes(payload)
    with pytest.raises(ValueError, match=match):
        checkpoint.load_mp(model="medium", device="cpu", default_dtype="float64",
                           expected_sha256=expected)
    assert consumed == []
    assert path.read_bytes() == payload


def test_checkpoint_asl_gate_precedes_file_access(monkeypatch):
    from vibeqc.mlip import _mace_checkpoint as checkpoint

    def unexpected_path(info):
        pytest.fail("Unacknowledged OFF23 must not access or fetch weights")

    monkeypatch.setattr(checkpoint, "_checkpoint_path", unexpected_path)
    with pytest.raises(PermissionError, match="Academic Software License"):
        checkpoint.load_off(model="small", device="cpu", default_dtype="float64")


@pytest.mark.parametrize("digest", ["", "0" * 63, "g" * 64, 123, True])
def test_invalid_checkpoint_digest_precedes_backend(monkeypatch, digest):
    from vibeqc.mlip import mace as mace_mod

    def unexpected_backend():
        pytest.fail("Invalid checkpoint identity must fail before backend import")

    monkeypatch.setattr(mace_mod, "_require_backends", unexpected_backend)
    with pytest.raises(ValueError, match="weight_sha256"):
        mace_mod.PeriodicMACEEvaluator(MLIPOptions(weight_sha256=digest))


@pytest.mark.parametrize("payload,headers,raises", [
    (b"checkpoint", {"Content-Length": "10"}, None),
    (b"short", {"Content-Length": "10"}, "incomplete"),
    (b"", {}, "empty"),
    (b"<html>", {"Content-Type": "text/html; charset=utf-8"}, "HTML"),
])
def test_checkpoint_download_publishes_only_complete_bytes(monkeypatch, tmp_path, payload, headers, raises):
    import io
    from vibeqc.mlip import _mace_checkpoint as checkpoint

    response = io.BytesIO(payload)
    response.headers = headers
    monkeypatch.setattr(checkpoint.urllib.request, "urlopen", lambda *a, **k: response)
    target = tmp_path / "cache" / "registered.model"
    if raises:
        with pytest.raises(RuntimeError, match=raises):
            checkpoint._download("https://example.invalid/model", target)
        assert not target.exists()
    else:
        checkpoint._download("https://example.invalid/model", target)
        assert target.read_bytes() == payload
    assert list(target.parent.glob(".mace-*")) == []


def test_checkpoint_download_preserves_concurrent_winner(monkeypatch, tmp_path):
    import io
    from vibeqc.mlip import _mace_checkpoint as checkpoint

    target = tmp_path / "registered.model"

    class ConcurrentResponse(io.BytesIO):
        headers = {}

        def read(self, *args):
            target.write_bytes(b"concurrent complete checkpoint")
            return super().read(*args)

    monkeypatch.setattr(checkpoint.urllib.request, "urlopen",
                        lambda *a, **k: ConcurrentResponse(b"this download"))
    checkpoint._download("https://example.invalid/model", target)
    assert target.read_bytes() == b"concurrent complete checkpoint"
    assert list(tmp_path.glob(".mace-*")) == []


def test_loaded_checkpoint_identity_survives_result_and_replay(backend_results, monkeypatch):
    from vibeqc.mlip import mace as mace_mod
    from vibeqc.mlip._mace_checkpoint import CheckpointIdentity
    from vibeqc.runner import _run_mlip, _format_mlip_provenance
    from vibeqc.output.formats.mlip_record import mlip_job_options, mlip_record_fields
    from vibeqc._qvf_options import hydrate_option

    calc, template = backend_results
    identity = CheckpointIdentity("a" * 64, 123, "https://example.invalid/model")
    calc._vibeqc_checkpoint_identity = identity
    model = mace_mod.MACEModel(template._mol, cell=template._cell)
    # Neither a caller's detached dictionary nor a later calculator label may
    # relabel an existing model or its subsequent periodic results.
    model.weight_identity["sha256"] = "b" * 64
    calc._vibeqc_checkpoint_identity = None
    periodic = mace_mod._periodic_result(model)
    assert periodic.weight_identity["sha256"] == identity.sha256
    monkeypatch.setattr(mace_mod, "MACEModel", lambda *a, **k: model)
    result = _run_mlip("mace", template._mol, MLIPOptions())
    assert result.weight_identity == periodic.weight_identity
    fields = mlip_record_fields(result)["mlip"]
    assert fields["checkpoint"] == model.weight_identity
    fields["checkpoint"]["sha256"] = "c" * 64
    assert model.weight_identity["sha256"] == identity.sha256
    options = hydrate_option("mlip_options", mlip_job_options(result)["mlip_options"])
    assert options.weight_sha256 == identity.sha256
    assert not options.academic_license_acknowledged()
    text = _format_mlip_provenance(model.info, model.options, result.weight_identity)
    assert identity.sha256 in text
    assert "123 checkpoint bytes" in text


@pytest.mark.parametrize("model,filename", [
    ("medium-mpa-0", "macempa0mediummodel"),
    ("small", "20231210mace128L0_energy_epoch249model"),
    ("medium", "20231203mace128L1_epoch199model"),
    ("large", "MACE_MPtrj_20229model"),
    ("off23-small", "MACE-OFF23_small.model"),
    ("off23-medium", "MACE-OFF23_medium.model"),
    ("off23-large", "MACE-OFF23_large.model"),
])
def test_checkpoint_registry_preserves_upstream_cache_names(monkeypatch, tmp_path, model, filename):
    import sys
    from types import ModuleType
    from vibeqc.mlip import _mace_checkpoint as checkpoint

    foundations = ModuleType("mace.calculators.foundations_models")
    foundations.local_model_path = str(tmp_path / "absent-bundled.model")
    calculators = ModuleType("mace.calculators")
    calculators.foundations_models = foundations
    package = ModuleType("mace")
    package.calculators = calculators
    for name, module in [("mace", package), ("mace.calculators", calculators),
                         ("mace.calculators.foundations_models", foundations)]:
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    calls = []

    def download(url, target):
        calls.append((url, target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"fixture checkpoint")

    monkeypatch.setattr(checkpoint, "_download", download)
    expected = tmp_path / "mace" / filename
    assert checkpoint._checkpoint_path(resolve_model(model)) == expected
    assert checkpoint._checkpoint_path(resolve_model(model)) == expected
    assert calls == [(checkpoint._CHECKPOINT_URLS[model], expected)]
    if model == "medium-mpa-0":
        bundled = Path(foundations.local_model_path)
        bundled.write_bytes(b"bundled fixture")
        assert checkpoint._checkpoint_path(resolve_model(model)) == bundled
        assert len(calls) == 1


def test_checkpoint_interrupted_download_leaves_no_cache(monkeypatch, tmp_path):
    import io
    from vibeqc.mlip import _mace_checkpoint as checkpoint

    class InterruptedResponse(io.BytesIO):
        headers = {}

        def read(self, *args):
            if self.tell():
                raise OSError("connection interrupted")
            return super().read(*args)

    monkeypatch.setattr(checkpoint.urllib.request, "urlopen",
                        lambda *a, **k: InterruptedResponse(b"partial bytes"))
    target = tmp_path / "registered.model"
    with pytest.raises(OSError, match="interrupted"):
        checkpoint._download("https://example.invalid/model", target)
    assert not target.exists()
    assert list(tmp_path.glob(".mace-*")) == []


def test_periodic_evaluator_retains_checkpoint_pin():
    from vibeqc.mlip.mace import PeriodicMACEEvaluator

    options = MLIPOptions(weight_sha256="a" * 64)
    evaluator = PeriodicMACEEvaluator(options)
    options.weight_sha256 = None
    assert evaluator._options.weight_sha256 == "a" * 64


@pytest.mark.parametrize("optimizer_backend", ["auto", "ase"])
@pytest.mark.parametrize("controls,match", [
    ({"fmax": float("inf")}, "fmax"),
    ({"fmax": float("nan")}, "fmax"),
    ({"fmax": 0.0}, "fmax"),
    ({"fmax": -0.1}, "fmax"),
    ({"fmax": True}, "fmax"),
    ({"fmax": "0.1"}, "fmax"),
    ({"max_opt_steps": -1}, "max_opt_steps"),
    ({"max_opt_steps": 1.5}, "max_opt_steps"),
    ({"max_opt_steps": True}, "max_opt_steps"),
    ({"max_opt_steps": "1"}, "max_opt_steps"),
    ({"max_opt_steps": float("inf")}, "max_opt_steps"),
])
def test_molecular_mace_rejects_invalid_stopping_controls_before_loading(
    monkeypatch, tmp_path, optimizer_backend, controls, match,
):
    """Invalid targets must not accept an unrelaxed structure or load weights."""
    from vibeqc import Atom, Molecule, run_job
    from vibeqc.mlip import mace as mace_mod

    def unexpected_backend():
        raise AssertionError("invalid controls reached the MACE backend")

    monkeypatch.setattr(mace_mod, "_require_backends", unexpected_backend)
    molecule = Molecule([Atom(1, [0., 0., 0.]), Atom(1, [2., 0., 0.])])
    with pytest.raises(ValueError, match=match):
        run_job(
            molecule, method="mace", optimize=True,
            optimizer_backend=optimizer_backend, output=tmp_path / "invalid",
            record_hostname=False, **controls,
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("max_steps", [0, 1])
def test_molecular_mace_accepts_integer_stopping_budgets(monkeypatch, tmp_path, max_steps):
    import numpy as np
    from vibeqc import Atom, Molecule, run_job
    from vibeqc.mlip import mace as mace_mod

    def backend_reached():
        raise RuntimeError("valid controls reached the MACE backend")

    monkeypatch.setattr(mace_mod, "_require_backends", backend_reached)
    molecule = Molecule([Atom(1, [0., 0., 0.]), Atom(1, [2., 0., 0.])])
    with pytest.raises(RuntimeError, match="valid controls reached"):
        run_job(
            molecule, method="mace", optimize=True, optimizer_backend="ase",
            fmax=np.float64(0.01), max_opt_steps=np.int64(max_steps),
            output=tmp_path / "valid", record_hostname=False,
        )


@pytest.mark.parametrize("change", ["checkpoint", "caller_options"])
def test_molecular_optimization_retains_loaded_potential(monkeypatch, tmp_path, change):
    """The reported terminal energy must belong to the optimizer's surface."""
    import json
    import tomllib
    import zipfile

    import numpy as np
    from ase import Atoms
    from ase.calculators.calculator import Calculator, all_changes
    from ase.io.trajectory import Trajectory
    from ase.units import Bohr, Hartree

    from vibeqc import Atom, Molecule
    from vibeqc.mlip import mace as mace_mod
    from vibeqc.mlip._mace_checkpoint import CheckpointIdentity
    from vibeqc.runner import _ASEGeometryOptimizationError, run_job

    options = MLIPOptions()
    loads = []
    identity = CheckpointIdentity("a" * 64, 123, "https://example.invalid/model")

    class BondPotential(Calculator):
        implemented_properties = ["energy", "free_energy", "forces"]

        def __init__(self, offset):
            super().__init__()
            self.offset = offset
            self._vibeqc_checkpoint_identity = identity

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            vector = atoms.positions[1] - atoms.positions[0]
            distance = np.linalg.norm(vector)
            delta = distance - 1.0
            energy = self.offset + 0.5 * delta**2
            force = delta * vector / distance
            self.results = {
                "energy": energy, "free_energy": energy,
                "forces": np.array([force, -force]),
            }
            if change == "caller_options":
                options.model = "small"
                options.dtype = "float32"

    def loader(**kwargs):
        # A second load observes a changed cache file or caller configuration.
        # No actual model file is altered and no network/torch is needed.
        loads.append(kwargs)
        calc = BondPotential(-3.0 if len(loads) == 1 else 7.0)
        if len(loads) > 1:
            calc._vibeqc_checkpoint_identity = CheckpointIdentity(
                "b" * 64, 456, identity.registry_url,
            )
        return calc

    monkeypatch.setattr(
        mace_mod, "_require_backends",
        lambda: (loader, loader, None, Atoms, Bohr, Hartree),
    )
    molecule = Molecule([
        Atom(1, [0., 0., 0.]), Atom(1, [1.5 / Bohr, 0., 0.]),
    ])
    stem = tmp_path / "optimized"
    result = run_job(
        molecule, method="mace", mlip_options=options,
        optimize=True, optimizer_backend="ase", fmax=1e-5, max_opt_steps=20,
        output=stem, record_hostname=False,
    )
    with Trajectory(str(stem.with_suffix(".traj"))) as trajectory:
        terminal = trajectory[-1]
    assert result.energy == pytest.approx(terminal.get_potential_energy() / Hartree, abs=1e-12, rel=0)
    np.testing.assert_allclose(
        result.gradient(), -terminal.get_forces() * Bohr / Hartree,
        atol=1e-12, rtol=0,
    )
    assert len(loads) == 1
    assert result.model_info.key == "medium-mpa-0"
    assert result.dtype == "float64"
    assert result.weight_identity["sha256"] == identity.sha256
    with zipfile.ZipFile(stem.with_suffix(".qvf")) as archive:
        spec = json.loads(archive.read("job_spec/spec.json"))
        provenance = json.loads(archive.read("manifest.json"))["provenance"]
    assert spec["options"]["mlip_options"] == {
        "model": "medium-mpa-0", "device": "cpu", "dtype": "float64",
        "weight_sha256": identity.sha256,
    }
    assert provenance["mlip"]["checkpoint"] == result.weight_identity

    # Reusing the terminal model must not turn a capped walk into success.
    loads.clear()
    options = MLIPOptions()
    capped = tmp_path / "capped"
    with pytest.raises(_ASEGeometryOptimizationError) as failure:
        run_job(
            molecule, method="mace", mlip_options=options,
            optimize=True, optimizer_backend="ase", fmax=1e-5, max_opt_steps=0,
            output=capped, record_hostname=False,
        )
    assert len(loads) == 1
    assert failure.value.n_steps == 0
    assert failure.value.final_max_force > failure.value.target_max_force
    assert tomllib.loads(capped.with_suffix(".system").read_text())["outputs"]["status"] == "crashed"
