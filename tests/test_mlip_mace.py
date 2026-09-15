"""vibeqc.mlip — MACE model registry + ASL gate + options.

These are pure-Python (no torch / mace required), so they run on any
vibe-qc environment — including the 3.14 dev venv where the ``[mace]``
extra cannot be installed. The ASL gate deliberately fires *before* any
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


def test_default_model_is_mit_mpa0():
    info = resolve_model(None)
    assert info.key == DEFAULT_MODEL == "medium-mpa-0"
    assert info.loader == "mace_mp"
    assert info.license == "MIT"
    assert not info.is_academic_only
    assert info.citation == "batatia_mace_mp_2024"


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
def test_periodic_position_optimization_rejects_noninteger_step_budgets_before_backend(
    monkeypatch, max_steps
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
        mace_mod.optimize_periodic_mace_positions(System(), max_steps=max_steps)


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

        def get_forces(self):
            return np.zeros((1, 3))

        def get_potential_energy(self):
            return 0.0

    class FakeModel:
        _Bohr = 1.0
        _Hartree = 1.0
        calculator = object()
        info = resolve_model("medium-mpa-0")
        options = MLIPOptions()

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


def test_periodic_position_optimization_validates_fixed_indices(monkeypatch):
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
    with pytest.raises(IndexError, match="out-of-range"):
        mace_mod.optimize_periodic_mace_positions(system, fixed_indices=(1,))
