"""Parity tests for the C++ MSINDO engine (vibeqc._vibeqc_core.semiempirical.indo).

The C++ engine is a faithful port of the validated Python reference engine
(vibeqc.semiempirical.methods.msindo).  Covers closed-shell s/p/d (H–Xe),
open-shell UHF (s/p light elements + the heavy d/p-block trajectory elements
Tc–Pd / Sb–Xe), NDDO closed-shell (H, Li–F, Na–Cl), and periodic CCM (1D/2D/3D).
"""

import json
from pathlib import Path

import numpy as np
import pytest
from vibeqc.semiempirical.methods import msindo as py_msindo
from vibeqc.semiempirical.methods import msindo_ccm as py_ccm

_indo = pytest.importorskip("vibeqc._vibeqc_core.semiempirical.indo")
run_msindo_cpp = _indo.run_msindo
run_msindo_full = _indo.run_msindo_full
run_msindo_uhf = _indo.run_msindo_uhf
run_ccm_cpp = _indo.run_ccm
load_params = _indo.load_params_from_json
merge_nddo = _indo.merge_nddo_overrides

_SYMBOL_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
}
_REF = json.loads(
    (
        Path(__file__).resolve().parents[1]
        / "examples/regression/msindo/molecular_reference.json"
    ).read_text()
)
_CPP_ELEMENTS = {1, 2, 6, 7, 8, 9}
_CPP_MOLS = [
    m
    for m in _REF["molecules"]
    if m.get("multiplicity", 1) == 1
    and m.get("charge", 0) == 0
    and all(_SYMBOL_Z.get(s) in _CPP_ELEMENTS for s, *_ in m["geometry"])
]
_ALL_CS_MOLS = [
    m
    for m in _REF["molecules"]
    if m.get("multiplicity", 1) == 1 and m.get("charge", 0) == 0
]

_BASE = Path(__file__).resolve().parents[1]
_PARAM_PATH = (
    _BASE / "python" / "vibeqc" / "semiempirical" / "methods" / "msindo_params.json"
)
_NDDO_PATH = (
    _BASE
    / "python"
    / "vibeqc"
    / "semiempirical"
    / "methods"
    / "msindo_params_nddo.json"
)
_PARAMS = load_params(_PARAM_PATH.read_text())
_NDDO_PARAMS = load_params(_PARAM_PATH.read_text())
merge_nddo(_NDDO_PARAMS, _NDDO_PATH.read_text())


def _force_python_msindo(monkeypatch):
    monkeypatch.setattr(py_msindo, "_cpp_msindo_kernel", lambda *, nddo=False: None)


@pytest.mark.parametrize("charge", [0, 2])
def test_public_msindo_gradient_analytic_uses_cpp(monkeypatch, charge):
    """The public MSINDO gradient wrapper routes native-scope jobs to C++."""
    from vibeqc.semiempirical.methods import msindo_gradient_analytic as grad_mod

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("Python MSINDO gradient pair loop was used")

    Z = [8, 1, 1]
    xyz = [[0, 0, 0], [0.95, 0.0, 0.10], [-0.20, 0.92, 0.0]]
    monkeypatch.setattr(grad_mod._pd, "_pair_blocks_deriv", _forbidden)

    g_public = grad_mod.msindo_gradient_analytic(Z, xyz, charge=charge)
    g_cpp = np.asarray(_indo.gradient_analytic(Z, xyz, _PARAMS, charge=charge))

    assert g_public.shape == (3, 3)
    assert np.all(np.isfinite(g_public))
    assert np.max(np.abs(g_public - g_cpp)) < 1e-12


def test_cpp_msindo_charged_gradient_matches_python(monkeypatch):
    """C++ molecular MSINDO gradient honors charge and matches Python."""
    from vibeqc.semiempirical.methods import msindo_gradient_analytic as grad_mod

    Z = [8, 1, 1]
    xyz = [[0, 0, 0], [0.95, 0.0, 0.10], [-0.20, 0.92, 0.0]]
    monkeypatch.setattr(grad_mod, "_cpp_gradient_kernel", lambda: None)

    gc = np.asarray(_indo.gradient_analytic(Z, xyz, _PARAMS, charge=2))
    gp = grad_mod.msindo_gradient_analytic(Z, xyz, charge=2)
    gn = np.asarray(_indo.gradient_analytic(Z, xyz, _PARAMS))

    assert gc.shape == gp.shape
    assert np.max(np.abs(gc - gp)) < 1e-8
    assert np.max(np.abs(gc - gn)) > 1e-5


def _zxyz(mol):
    Z = [_SYMBOL_Z[s] for s, *_ in mol["geometry"]]
    xyz = [[x, y, z] for _, x, y, z in mol["geometry"]]
    return Z, xyz


def test_public_run_msindo_closed_shell_uses_cpp(monkeypatch):
    """The direct public MSINDO wrapper routes closed-shell jobs to C++."""
    def _forbidden(*_args, **_kwargs):
        raise AssertionError("Python MSINDO RHF SCF path was used")

    Z = [8, 1, 1]
    xyz = [[0, 0, 0], [0.95, 0.0, 0.10], [-0.20, 0.92, 0.0]]
    monkeypatch.setattr(py_msindo, "_scf_rhf", _forbidden)
    monkeypatch.setattr(py_msindo, "_scf_rhf_msindo", _forbidden)

    r_public = py_msindo.run_msindo(Z, xyz, charge=2)
    r_cpp = run_msindo_full(Z, xyz, _PARAMS, charge=2)

    assert r_public.converged
    assert r_public.total_energy == pytest.approx(r_cpp.total_energy, abs=1e-12)
    assert r_public.electronic_energy == pytest.approx(
        r_cpp.electronic_energy, abs=1e-12
    )


def test_public_run_msindo_open_shell_uses_cpp(monkeypatch):
    """The direct public MSINDO wrapper routes open-shell jobs to C++ UHF."""
    def _forbidden(*_args, **_kwargs):
        raise AssertionError("Python MSINDO UHF SCF path was used")

    Z = [8, 1, 1]
    xyz = [[0, 0, 0], [0.95, 0.0, 0.10], [-0.20, 0.92, 0.0]]
    monkeypatch.setattr(py_msindo, "_scf_uhf", _forbidden)
    monkeypatch.setattr(py_msindo, "_scf_uhf_msindo", _forbidden)

    r_public = py_msindo.run_msindo(Z, xyz, charge=1, multiplicity=2)
    r_cpp = run_msindo_uhf(Z, xyz, _PARAMS, charge=1, multiplicity=2)

    assert r_public.converged
    assert r_public.total_energy == pytest.approx(r_cpp.total_energy, abs=1e-12)


def test_public_run_msindo_nddo_uses_cpp(monkeypatch):
    """The direct public MSINDO wrapper routes closed-shell NDDO jobs to C++."""
    def _forbidden(*_args, **_kwargs):
        raise AssertionError("Python MSINDO NDDO path was used")

    Z = [1, 9]
    xyz = [[0, 0, 0], [0, 0, 0.917]]
    monkeypatch.setattr(py_msindo, "_run_nddo", _forbidden)

    r_public = py_msindo.run_msindo(Z, xyz, nddo=True)
    r_cpp = run_msindo_full(Z, xyz, _NDDO_PARAMS, nddo=True)

    assert r_public.converged
    assert r_public.total_energy == pytest.approx(r_cpp.total_energy, abs=1e-12)


def test_cpp_msindo_full_rejects_odd_electron_nddo_at_native_boundary():
    """The nominal RHF entry point cannot auto-dispatch NDDO into UHF."""
    Z = [8, 7]
    xyz = [[0, 0, 0], [0, 0, 1.15]]

    with pytest.raises(ValueError, match="closed-shell"):
        run_msindo_full(Z, xyz, _NDDO_PARAMS, nddo=True)


def test_cpp_msindo_uhf_rejects_nddo_at_native_boundary():
    """Direct callers cannot bypass the closed-shell-only NDDO contract."""
    Z = [8, 7]
    xyz = [[0, 0, 0], [0, 0, 1.15]]

    with pytest.raises(ValueError, match="closed-shell"):
        run_msindo_uhf(
            Z,
            xyz,
            _NDDO_PARAMS,
            multiplicity=2,
            nddo=True,
        )


@pytest.mark.parametrize("mol", _CPP_MOLS, ids=lambda m: m["name"])
def test_cpp_total_energy_parity(mol):
    """C++ MSINDO reproduces reference MSINDO total energy to ~1e-6 Ha."""
    Z, xyz = _zxyz(mol)
    r = run_msindo_cpp(Z, xyz)
    assert r.converged
    assert r.total_energy == pytest.approx(mol["reference"]["total_energy"], abs=1e-6)
    assert r.binding_energy == pytest.approx(
        mol["reference"]["binding_energy"], abs=1e-6
    )


@pytest.mark.parametrize("mol", _CPP_MOLS, ids=lambda m: m["name"])
def test_cpp_matches_python_engine(mol, monkeypatch):
    """C++ and Python engines agree to ~1e-9 Ha (same algorithm, two languages)."""
    _force_python_msindo(monkeypatch)
    Z, xyz = _zxyz(mol)
    rc = run_msindo_cpp(Z, xyz)
    rp = py_msindo.run_msindo(Z, xyz)
    assert rc.total_energy == pytest.approx(rp.total_energy, abs=1e-8)
    assert rc.electronic_energy == pytest.approx(rp.electronic_energy, abs=1e-8)


def test_cpp_full_charged_closed_shell_matches_python(monkeypatch):
    """C++ RHF MSINDO energy honors charge and matches Python."""
    _force_python_msindo(monkeypatch)
    Z = [8, 1, 1]
    xyz = [[0, 0, 0], [0.95, 0.0, 0.10], [-0.20, 0.92, 0.0]]

    rc = run_msindo_full(Z, xyz, _PARAMS, charge=2)
    rp = py_msindo.run_msindo(Z, xyz, charge=2)
    rn = run_msindo_full(Z, xyz, _PARAMS)

    assert rc.converged
    assert rc.total_energy == pytest.approx(rp.total_energy, abs=1e-8)
    assert abs(rc.total_energy - rn.total_energy) > 1e-3


def test_cpp_uhf_charged_matches_python(monkeypatch):
    """C++ UHF MSINDO energy honors charge and matches Python."""
    _force_python_msindo(monkeypatch)
    Z = [8, 1, 1]
    xyz = [[0, 0, 0], [0.95, 0.0, 0.10], [-0.20, 0.92, 0.0]]

    rc = run_msindo_uhf(Z, xyz, _PARAMS, charge=1, multiplicity=2)
    rp = py_msindo.run_msindo(Z, xyz, charge=1, multiplicity=2)
    rn = run_msindo_full(Z, xyz, _PARAMS)

    assert rc.converged
    assert rc.total_energy == pytest.approx(rp.total_energy, abs=1e-8)
    assert abs(rc.total_energy - rn.total_energy) > 1e-3


def test_runner_dispatch_msindo():
    """method='msindo' routes through runner._run_semiempirical to the C++ engine."""
    import vibeqc._vibeqc_core as core
    from vibeqc.runner import _run_semiempirical
    from vibeqc.semiempirical.methods.msindo import ANGSTROM_TO_BOHR as A2B

    mol = core.Molecule(
        [core.Atom(1, [0, 0, 0]), core.Atom(9, [0, 0, 0.917 * A2B])],
        charge=0,
        multiplicity=1,
    )
    r = _run_semiempirical("msindo", mol)
    assert r.converged
    assert r.method == "msindo"
    assert r.energy == pytest.approx(-24.3089965036, abs=1e-6)


def test_runner_dispatch_msindo_closed_shell_gradient_is_lazy(monkeypatch):
    """Unified MSINDO results expose the closed-shell analytic gradient lazily."""
    import vibeqc._vibeqc_core as core
    from vibeqc.runner import _run_semiempirical
    from vibeqc.semiempirical.methods import msindo_gradient_analytic as grad_mod
    from vibeqc.semiempirical.methods.msindo import ANGSTROM_TO_BOHR as A2B

    calls = []

    def fake_gradient(atomic_numbers, coords_angstrom, **kwargs):
        calls.append((atomic_numbers, coords_angstrom, kwargs))
        return np.ones((len(atomic_numbers), 3))

    monkeypatch.setattr(grad_mod, "msindo_gradient_analytic", fake_gradient)
    mol = core.Molecule(
        [core.Atom(1, [0, 0, 0]), core.Atom(9, [0, 0, 0.917 * A2B])],
        charge=0,
        multiplicity=1,
    )
    r = _run_semiempirical("msindo", mol)
    assert calls == []

    g1 = r.gradient()
    g2 = r.gradient()
    assert np.asarray(g1).shape == (2, 3)
    assert g2 is g1
    assert len(calls) == 1
    atomic_numbers, coords_angstrom, kwargs = calls[0]
    assert atomic_numbers == [1, 9]
    assert coords_angstrom[1][2] == pytest.approx(0.917)
    assert kwargs["charge"] == 0
    assert kwargs["multiplicity"] == 1


def test_runner_msindo_charged_closed_shell_uses_charged_cpp(monkeypatch):
    """method='msindo' passes charge through to the C++ RHF engine."""
    import vibeqc._vibeqc_core as core
    from vibeqc.runner import _run_semiempirical
    from vibeqc.semiempirical.methods.msindo import ANGSTROM_TO_BOHR as A2B

    original = _indo.run_msindo_full
    seen = {}

    def _recording_run_msindo_full(*args, **kwargs):
        seen["charge"] = kwargs.get("charge")
        return original(*args, **kwargs)

    monkeypatch.setattr(_indo, "run_msindo_full", _recording_run_msindo_full)
    mol = core.Molecule(
        [
            core.Atom(8, [0, 0, 0]),
            core.Atom(1, [0.95 * A2B, 0.0, 0.10 * A2B]),
            core.Atom(1, [-0.20 * A2B, 0.92 * A2B, 0.0]),
        ],
        charge=2,
        multiplicity=1,
    )
    expected = original(
        [8, 1, 1],
        [[0, 0, 0], [0.95, 0.0, 0.10], [-0.20, 0.92, 0.0]],
        _PARAMS,
        max_iter=200,
        charge=2,
    )

    r = _run_semiempirical("msindo", mol)
    assert seen["charge"] == 2
    assert r.energy == pytest.approx(expected.total_energy, abs=1e-9)


def test_runner_dispatch_msindo_dshell():
    """method='msindo' on a 3rd-row d molecule (H2S) routes to the C++ engine."""
    import vibeqc._vibeqc_core as core
    from vibeqc.runner import _run_semiempirical
    from vibeqc.semiempirical.methods.msindo import ANGSTROM_TO_BOHR as A2B

    mol = core.Molecule(
        [
            core.Atom(16, [0, 0, 0]),
            core.Atom(1, [0.9627 * A2B, 0, 0.9264 * A2B]),
            core.Atom(1, [-0.9627 * A2B, 0, 0.9264 * A2B]),
        ],
        charge=0,
        multiplicity=1,
    )
    r = _run_semiempirical("msindo", mol)
    assert r.converged
    assert r.energy == pytest.approx(-11.2354666980, abs=1e-6)


def test_runner_msindo_open_shell_uhf():
    """method='msindo' routes an open-shell molecule to the C++ UHF engine."""
    import vibeqc._vibeqc_core as core
    from vibeqc.runner import _run_semiempirical
    from vibeqc.semiempirical.methods.msindo import ANGSTROM_TO_BOHR as A2B

    # OH• doublet — open-shell, dispatched to C++ UHF.
    mol = core.Molecule(
        [core.Atom(8, [0, 0, 0]), core.Atom(1, [0, 0, 0.97 * A2B])],
        charge=0,
        multiplicity=2,
    )
    r = _run_semiempirical("msindo", mol)
    assert r.converged
    assert r.energy == pytest.approx(-16.3330865643, abs=1e-6)


def test_runner_msindo_open_shell_heavy_stays_on_cpp(monkeypatch):
    """Heavy open-shell run_job dispatch must not fall back to Python MSINDO."""
    import vibeqc._vibeqc_core as core
    from vibeqc.runner import _run_semiempirical
    from vibeqc.semiempirical.methods import msindo as py_msindo
    from vibeqc.semiempirical.methods.msindo import ANGSTROM_TO_BOHR as A2B

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("Python MSINDO fallback was used")

    monkeypatch.setattr(py_msindo, "run_msindo", _forbidden)
    mol = core.Molecule(
        [core.Atom(42, [0, 0, 0]), core.Atom(17, [0, 0, 2.30 * A2B])],
        charge=0,
        multiplicity=2,
    )
    r = _run_semiempirical("msindo", mol)
    assert r.converged
    assert r.energy == pytest.approx(-21.4390039753, abs=1e-6)


def test_msindo_citation_route_fires():
    """A msindo job emits the Ahlswede-Jug 1999 method papers (CLAUDE.md §8)."""
    from vibeqc.output.citations.registry import load_default_database

    db = load_default_database()
    keys = {c.key for c in db.assemble(method="msindo", basis="sto-3g").citations}
    assert {"ahlswede_jug_msindo_1_1999", "ahlswede_jug_msindo_2_1999"} <= keys


def test_msindo_in_method_registry():
    """MSINDO is registered as MethodFamily.INDO (closed-shell, H–F)."""
    from vibeqc._vibeqc_core.semiempirical import (
        MethodFamily,
        SemiempiricalMethodRegistry,
    )

    reg = SemiempiricalMethodRegistry.instance()
    if "msindo" not in reg.method_names():
        try:
            SemiempiricalMethodRegistry.register_dftb()  # registers all builtins
        except Exception:
            pass
    assert "msindo" in reg.method_names()
    cfg = reg.find_config("msindo")
    assert cfg.name == "msindo"
    assert cfg.family == MethodFamily.INDO
    # Open-shell UHF is a supported MSINDO route and has a native C++ kernel.
    assert cfg.supports_open_shell is True


# --------------------------------------------------------------------------- #
# Full parameter-driven closed-shell s/p/d parity.
# --------------------------------------------------------------------------- #

# Full parameter-driven closed-shell s/p/d parity (all elements H-Br).
# --------------------------------------------------------------------------- #

_CS_MOLS = [
    m
    for m in _REF["molecules"]
    if m.get("multiplicity", 1) == 1 and m.get("charge", 0) == 0
]


@pytest.mark.parametrize("mol", _CS_MOLS, ids=lambda m: m["name"])
def test_cpp_full_total_energy_parity(mol):
    """Param-driven C++ engine reproduces reference MSINDO to ~1e-6 Ha."""
    Z, xyz = _zxyz(mol)
    r = run_msindo_full(Z, xyz, _PARAMS)
    assert r.converged
    assert r.total_energy == pytest.approx(mol["reference"]["total_energy"], abs=1e-6)


@pytest.mark.parametrize("mol", _CS_MOLS, ids=lambda m: m["name"])
def test_cpp_full_matches_python(mol, monkeypatch):
    """Param-driven C++ and Python engines agree to ~1e-8 Ha."""
    _force_python_msindo(monkeypatch)
    Z, xyz = _zxyz(mol)
    rc = run_msindo_full(Z, xyz, _PARAMS)
    rp = py_msindo.run_msindo(Z, xyz)
    assert rc.total_energy == pytest.approx(rp.total_energy, abs=1e-8)


# --------------------------------------------------------------------------- #
# Full 5th row (Z 37–54): the C++ engine's 5th-row ENEG/ATENG + [Kr]/[Kr]4d¹⁰
# frozen-core penetration + the Hückel-guess/WICHT SCF for the open-d/lone-pair
# heavy elements.  Locks both the C++↔oracle parity for the gated Tc–Pd/Sb–Xe and
# the run_job 5th-row regression (closed-shell run_job routes to run_msindo_full;
# the C++ engine was H-Kr-only before this row landed).  References from the
# MSINDO oracle.  See docs/user_guide/msindo.md.
def _geo(builder, r):
    import math
    p = 0.8660254
    if builder == "octa":
        return [(r, 0, 0), (-r, 0, 0), (0, r, 0), (0, -r, 0), (0, 0, r), (0, 0, -r)]
    if builder == "tet":
        d = r / math.sqrt(3.0)
        return [(d, d, d), (d, -d, -d), (-d, d, -d), (-d, -d, d)]
    if builder == "tbp":
        return [(r, 0, 0), (-r / 2, r * p, 0), (-r / 2, -r * p, 0), (0, 0, r), (0, 0, -r)]
    if builder == "pl3":
        return [(r, 0, 0), (-r / 2, r * p, 0), (-r / 2, -r * p, 0)]
    raise ValueError(builder)


_FIFTH_ROW = [
    # (name, Z, coords, oracle_total).  Enabled s/4d/5p (DIIS path):
    ("RbF", [37, 9], [(0, 0, 0), (0, 0, 2.27)], -23.6798426651),
    ("SrF2", [38, 9, 9], [(0, 0, 0), (0, 0, 2.1), (0, 0, -2.1)], -47.7033579329),
    ("YF3", [39, 9, 9, 9], [(0, 0, 0)] + _geo("pl3", 2.0), -72.2032148014),
    ("ZrF4", [40, 9, 9, 9, 9], [(0, 0, 0)] + _geo("tet", 1.9), -97.1358343710),
    ("NbF5", [41, 9, 9, 9, 9, 9], [(0, 0, 0)] + _geo("tbp", 1.88), -122.4882161084),
    ("MoF6", [42, 9, 9, 9, 9, 9, 9], [(0, 0, 0)] + _geo("octa", 1.82), -149.1104634227),
    ("AgF", [47, 9], [(0, 0, 0), (0, 0, 1.98)], -36.3888699675),
    ("CdCl2", [48, 17, 17], [(0, 0, 0), (0, 0, 2.21), (0, 0, -2.21)], -49.8912994610),
    ("InF3", [49, 9, 9, 9], [(0, 0, 0)] + _geo("pl3", 1.99), -72.3547574341),
    ("SnF4", [50, 9, 9, 9, 9], [(0, 0, 0)] + _geo("tet", 1.88), -97.4075598412),
    # Open-d 4d (Tc–Pd) + lone-pair 5p (Sb–Xe): the Hückel+WICHT SCF path.
    ("TcCl", [43, 17], [(0, 0, 0), (0, 0, 2.25)], -25.0060838476),
    ("RuO4", [44, 8, 8, 8, 8], [(0, 0, 0)] + _geo("tet", 1.71), -78.1614421070),
    ("RhF", [45, 9], [(0, 0, 0), (0, 0, 1.80)], -44.8136548780),
    ("PdCl2", [46, 17, 17], [(0, 0, 0), (0, 0, 2.30), (0, 0, -2.30)], -59.4646491771),
    ("SbF3", [51, 9, 9, 9],
     [(0, 0, 0), (0, 0, 1.88), (1.78, 0, -0.6), (-1.78, 0, -0.6)], -75.3664492072),
    ("TeF6", [52, 9, 9, 9, 9, 9, 9], [(0, 0, 0)] + _geo("octa", 1.82), -149.2768395396),
    ("ICl", [53, 17], [(0, 0, 0), (0, 0, 2.32)], -24.4450047669),
    ("XeF2", [54, 9, 9], [(0, 0, 0), (0, 0, 2.0), (0, 0, -2.0)], -62.8880352420),
]


@pytest.mark.parametrize("name,Z,coords,ref", _FIFTH_ROW, ids=[c[0] for c in _FIFTH_ROW])
def test_cpp_fifth_row_oracle_and_python_parity(name, Z, coords, ref, monkeypatch):
    """C++ run_msindo_full reproduces the MSINDO oracle (≤1 µHa) and matches the
    Python engine for the full 5th row.  Tolerance 1e-6: the Tc–Pd/Sb–Xe metastable
    states converge on MSINDO's energy-only criterion (~1e-7 between numpy and
    Eigen diagonalizers), well within the oracle bar."""
    _force_python_msindo(monkeypatch)
    xyz = [[float(x), float(y), float(z)] for x, y, z in coords]
    rc = run_msindo_full(Z, xyz, _PARAMS)
    assert rc.converged, f"{name}: C++ SCF not converged"
    assert rc.total_energy == pytest.approx(ref, abs=1e-6), f"{name}: C++ vs oracle"
    rp = py_msindo.run_msindo(Z, xyz)
    assert rc.total_energy == pytest.approx(rp.total_energy, abs=1e-6), f"{name}: C++ vs Python"


# Homonuclear 4d dimers Nb₂/Mo₂: routed to the Hückel+WICHT path (Nb/Mo added to
# the trajectory set 2026-06-17) so they reach the MSINDO reference basin (DIIS
# jumps to a spurious one).  Separate from _FIFTH_ROW because Mo₂'s sextuply-bonded
# valence space is so near-degenerate that the energy-only stop lands numpy (Python)
# and Eigen (C++)/Netlib (oracle) ~1 µHa apart on a flat plateau — the basin, not a
# 1e-6 match, is the target.  C++↔Python parity stays tight (≤2e-8).
_DIMERS = [
    ("Nb2", [41, 41], [(0, 0, -1.04), (0, 0, 1.04)], -6.9007186359),
    ("Mo2", [42, 42], [(0, 0, -0.97), (0, 0, 0.97)], -14.4178688549),
]


@pytest.mark.parametrize("name,Z,coords,ref", _DIMERS, ids=[c[0] for c in _DIMERS])
def test_cpp_homonuclear_4d_dimers(name, Z, coords, ref, monkeypatch):
    """C++ Nb₂/Mo₂ reach the oracle basin (≤5 µHa) and match the Python engine."""
    _force_python_msindo(monkeypatch)
    xyz = [[float(x), float(y), float(z)] for x, y, z in coords]
    rc = run_msindo_full(Z, xyz, _PARAMS)
    assert rc.converged, f"{name}: C++ SCF not converged"
    assert rc.total_energy == pytest.approx(ref, abs=5e-6), f"{name}: C++ vs oracle"
    rp = py_msindo.run_msindo(Z, xyz)
    assert rc.total_energy == pytest.approx(rp.total_energy, abs=1e-6), f"{name}: C++ vs Python"


# --------------------------------------------------------------------------- #
# NDDO mode.
# --------------------------------------------------------------------------- #

_NDDO_TESTS = [
    ("HF", [1, 9], [[0, 0, 0], [0, 0, 0.917]], -23.0436304974),
    (
        "H2O",
        [8, 1, 1],
        [[0, 0, 0.1173], [0, 0.7572, -0.4692], [0, -0.7572, -0.4692]],
        -17.0717535443,
    ),
    ("N2", [7, 7], [[0, 0, 0], [0, 0, 1.098]], -19.8447373916),
    ("CO", [6, 8], [[0, 0, 0], [0, 0, 1.128]], -21.6762159213),
    ("AlCl", [13, 17], [[0, 0, 0], [0, 0, 2.10]], -16.4566380534),
    (
        "CH4",
        [6, 1, 1, 1, 1],
        [
            [0, 0, 0],
            [0.6276, 0.6276, 0.6276],
            [-0.6276, -0.6276, 0.6276],
            [0.6276, -0.6276, -0.6276],
            [-0.6276, 0.6276, -0.6276],
        ],
        -8.2337262049,
    ),
]


@pytest.mark.parametrize("name,Z,coords,ref", _NDDO_TESTS)
def test_cpp_nddo_total_energy_matches_oracle(name, Z, coords, ref):
    """C++ NDDO reproduces reference MSINDO NDDO to ~1 uHa."""
    r = run_msindo_full(Z, coords, _NDDO_PARAMS, nddo=True)
    assert r.converged
    assert r.total_energy == pytest.approx(ref, abs=1e-6)


@pytest.mark.parametrize("name,Z,coords,_", _NDDO_TESTS)
def test_cpp_nddo_matches_python(name, Z, coords, _, monkeypatch):
    """C++ and Python NDDO engines agree to ~1e-8 Ha."""
    _force_python_msindo(monkeypatch)
    rc = run_msindo_full(Z, coords, _NDDO_PARAMS, nddo=True)
    rp = py_msindo.run_msindo(Z, coords, nddo=True)
    assert rc.total_energy == pytest.approx(rp.total_energy, abs=1e-8)


# --------------------------------------------------------------------------- #
# UHF open-shell.  Geometries and reference values from molecular_reference.json.
# --------------------------------------------------------------------------- #

_UHF_MOLS = [m for m in _REF["molecules"] if m.get("multiplicity", 1) != 1]


@pytest.mark.parametrize("mol", _UHF_MOLS, ids=lambda m: m["name"])
def test_cpp_uhf_total_energy_matches_oracle(mol):
    """C++ UHF reproduces reference MSINDO UHF to ~2e-4 Ha."""
    Z, xyz = _zxyz(mol)
    r = run_msindo_uhf(Z, xyz, _PARAMS, multiplicity=mol["multiplicity"])
    assert r.converged
    assert r.total_energy == pytest.approx(mol["reference"]["total_energy"], abs=2e-4)


@pytest.mark.parametrize("mol", _UHF_MOLS, ids=lambda m: m["name"])
def test_cpp_uhf_matches_python(mol, monkeypatch):
    """C++ and Python UHF engines agree to ~1e-8 Ha."""
    _force_python_msindo(monkeypatch)
    Z, xyz = _zxyz(mol)
    rc = run_msindo_uhf(Z, xyz, _PARAMS, multiplicity=mol["multiplicity"])
    rp = py_msindo.run_msindo(Z, xyz, multiplicity=mol["multiplicity"])
    assert rc.total_energy == pytest.approx(rp.total_energy, abs=1e-8)


def test_cpp_uhf_reduces_to_rhf():
    """UHF singlet on a closed-shell molecule reproduces RHF."""
    Z, xyz = [8, 1, 1], [[0, 0, 0], [0, 0.757, 0.587], [0, -0.757, 0.587]]
    rc_rhf = run_msindo_full(Z, xyz, _PARAMS)
    rc_uhf = run_msindo_uhf(Z, xyz, _PARAMS, multiplicity=1)
    assert rc_uhf.total_energy == pytest.approx(rc_rhf.total_energy, abs=1e-9)


# Heavy-element open-shell UHF (Nb–Pd 4d, Sb–Xe 5p): the C++ Hückel-guess + WICHT
# UHF SCF (scf_uhf_msindo_driver, dispatched through is_msindo_trajectory_scf in
# run_msindo_core_uhf), including the faithful fockop.f one-centre d Fock
# (add_einzi_dblock_uhf).  Locks C++↔oracle and C++↔Python parity.  Explicit
# Z lists (heavy symbols are absent from _SYMBOL_Z), matching _FIFTH_ROW.  Same
# molecules / oracle references as tests/test_msindo.py::_UHF_HEAVY.
_UHF_HEAVY = [
    ("NbO2", [41, 8, 8], [(0, 0, 0), (0, 0, 1.70), (0, 0, -1.70)], 2, -35.4496629687),
    ("MoCl", [42, 17], [(0, 0, 0), (0, 0, 2.30)], 2, -21.4390039753),
    ("MoO2", [42, 8, 8], [(0, 0, 0), (0, 0, 1.70), (0, 0, -1.70)], 3, -38.9214397584),  # triplet
    ("TcCl2", [43, 17, 17], [(0, 0, 0), (0, 0, 2.25), (0, 0, -2.25)], 2, -39.2198082755),
    ("RuCl", [44, 17], [(0, 0, 0), (0, 0, 2.20)], 2, -29.6808033415),
    ("RuF2", [44, 9, 9], [(0, 0, 0), (0, 0, 1.82), (0, 0, -1.82)], 3, -62.9657257886),
    ("RhF2", [45, 9, 9], [(0, 0, 0), (0, 0, 1.85), (0, 1.60, -0.92)], 2, -68.6846477540),
    ("PdCl", [46, 17], [(0, 0, 0), (0, 0, 2.30)], 2, -45.6424788740),
    ("SbF2", [51, 9, 9], [(0, 0, 0), (0, 0, 1.88), (0, 0, -1.88)], 2, -51.5332968692),
    ("TeCl", [52, 17], [(0, 0, 0), (0, 0, 2.25)], 2, -20.8891995841),
    ("ICl2", [53, 17, 17], [(0, 0, 0), (0, 0, 2.32), (0, 0, -2.32)], 2, -38.5687534466),
    ("XeOF", [54, 8, 9], [(0, 0, 0), (0, 0, 1.75), (0, 0, -2.0)], 2, -55.1886280567),
]


@pytest.mark.parametrize("name,Z,coords,mult,ref", _UHF_HEAVY, ids=[c[0] for c in _UHF_HEAVY])
def test_cpp_uhf_heavy_oracle_and_python_parity(
    name, Z, coords, mult, ref, monkeypatch
):
    """C++ run_msindo_uhf reproduces the MSINDO oracle (≤1 µHa) and matches the
    Python engine for the heavy d/p-block open-shell radicals."""
    _force_python_msindo(monkeypatch)
    xyz = [[float(x), float(y), float(z)] for x, y, z in coords]
    rc = run_msindo_uhf(Z, xyz, _PARAMS, multiplicity=mult)
    assert rc.converged, f"{name}: C++ UHF SCF not converged"
    assert rc.total_energy == pytest.approx(ref, abs=1e-6), f"{name}: C++ vs oracle"
    rp = py_msindo.run_msindo(Z, xyz, multiplicity=mult)
    assert rc.total_energy == pytest.approx(rp.total_energy, abs=1e-6), f"{name}: C++ vs Python"


# --------------------------------------------------------------------------- #
# Periodic CCM — 1D / 2D / 3D.
# --------------------------------------------------------------------------- #

_He7_Z = [2, 2, 2, 2, 2, 2, 2]
_He7_coords = [[i * 2.0, 0.0, 0.0] for i in range(7)]
_He7_T = [[14.0, 0.0, 0.0]]
# Removing two electrons from the exactly uniform ring leaves a degenerate
# frontier pair and requires the reference executable's fractional-occupation
# path. A small physical distortion selects one integer-occupation root for
# the C++/Python charged-parity tests without changing their charge purpose.
_He7_charged_coords = [coord.copy() for coord in _He7_coords]
_He7_charged_coords[1][0] += 0.01


@pytest.mark.parametrize("charge", [0, 2])
def test_public_run_ccm_uses_cpp(monkeypatch, charge):
    """The public CCM energy wrapper routes native-scope jobs to C++."""
    def _forbidden(*_args, **_kwargs):
        raise AssertionError("Python CCM SCF path was used")

    monkeypatch.setattr(py_ccm, "_ccm_total_energy", _forbidden)

    r_public = py_ccm.run_ccm(
        _He7_Z, _He7_coords, _He7_T, charge=charge, madelung=False
    )
    r_cpp = run_ccm_cpp(
        _He7_Z, _He7_coords, _He7_T, _PARAMS, madelung=False, charge=charge
    )

    assert r_public.converged
    assert r_public.total_energy == pytest.approx(r_cpp.total_energy, abs=1e-12)
    assert r_public.electronic_energy == pytest.approx(
        r_cpp.electronic_energy, abs=1e-12
    )


def test_cpp_ccm_charged_energy_matches_python(monkeypatch):
    """C++ CCM energy honors charge and matches the Python reference path."""
    monkeypatch.setattr(py_ccm, "_cpp_ccm_energy_kernel", lambda: None)

    rc = run_ccm_cpp(
        _He7_Z, _He7_charged_coords, _He7_T, _PARAMS, madelung=False, charge=2
    )
    rp = py_ccm.run_ccm(
        _He7_Z, _He7_charged_coords, _He7_T, charge=2, madelung=False
    )
    rn = run_ccm_cpp(
        _He7_Z, _He7_charged_coords, _He7_T, _PARAMS, madelung=False
    )

    assert rc.converged
    assert rc.total_energy == pytest.approx(rp.total_energy, abs=1e-8)
    assert abs(rc.total_energy - rn.total_energy) > 1e-3


def test_ccm_he7_chain_noewald_matches_python(monkeypatch):
    """1-D He chain without Madelung matches Python CCM."""
    monkeypatch.setattr(py_ccm, "_cpp_ccm_energy_kernel", lambda: None)
    rc = run_ccm_cpp(_He7_Z, _He7_coords, _He7_T, _PARAMS, madelung=False)
    rp = py_ccm.run_ccm(_He7_Z, _He7_coords, _He7_T, madelung=False)
    assert rc.converged
    assert rc.total_energy == pytest.approx(rp.total_energy, abs=1e-10)


_HF6_Z = [1, 9, 1, 9, 1, 9]
_HF6_coords = [
    [0, 0, 0],
    [1.0, 0, 0],
    [2.5, 0, 0],
    [3.5, 0, 0],
    [5.0, 0, 0],
    [6.0, 0, 0],
]
_HF6_T = [[7.5, 0.0, 0.0]]


def test_ccm_hf_chain_madelung_matches_python(monkeypatch):
    """1-D H-F chain with Madelung matches Python CCM."""
    monkeypatch.setattr(py_ccm, "_cpp_ccm_energy_kernel", lambda: None)
    rc = run_ccm_cpp(_HF6_Z, _HF6_coords, _HF6_T, _PARAMS, madelung=True)
    rp = py_ccm.run_ccm(_HF6_Z, _HF6_coords, _HF6_T, madelung=True)
    assert rc.converged
    assert rp.converged
    assert rc.total_energy == pytest.approx(rp.total_energy, abs=1e-6)


# 2-D/3-D MgO.
a_mgo = 2.105
_MgO_Z = [12, 8, 12, 8, 8, 12, 8, 12]
_MgO_coords = [
    [0, 0, 0],
    [a_mgo, 0, 0],
    [0, a_mgo, 0],
    [a_mgo, a_mgo, 0],
    [0, 0, a_mgo],
    [a_mgo, 0, a_mgo],
    [0, a_mgo, a_mgo],
    [a_mgo, a_mgo, a_mgo],
]


def test_ccm_mgo_2d_matches_python(monkeypatch):
    """2-D MgO slab matches Python CCM."""
    monkeypatch.setattr(py_ccm, "_cpp_ccm_energy_kernel", lambda: None)
    rc = run_ccm_cpp(
        _MgO_Z,
        _MgO_coords,
        [[2 * a_mgo, 0, 0], [0, 2 * a_mgo, 0]],
        _PARAMS,
        madelung=False,
    )
    rp = py_ccm.run_ccm(
        _MgO_Z, _MgO_coords, [[2 * a_mgo, 0, 0], [0, 2 * a_mgo, 0]], madelung=False
    )
    assert rc.converged
    assert rc.total_energy == pytest.approx(rp.total_energy, abs=1e-6)


def test_ccm_mgo_3d_matches_python(monkeypatch):
    """3-D MgO bulk matches Python CCM."""
    monkeypatch.setattr(py_ccm, "_cpp_ccm_energy_kernel", lambda: None)
    T3d = [[2 * a_mgo, 0, 0], [0, 2 * a_mgo, 0], [0, 0, 2 * a_mgo]]
    rc = run_ccm_cpp(_MgO_Z, _MgO_coords, T3d, _PARAMS, madelung=False)
    rp = py_ccm.run_ccm(_MgO_Z, _MgO_coords, T3d, madelung=False)
    assert rc.converged
    assert rc.total_energy == pytest.approx(rp.total_energy, abs=1e-6)


def test_ccm_gradient_fd_matches_python(monkeypatch):
    """C++ CCM FD gradient matches Python on He7 chain."""
    ccm_grad = pytest.importorskip(
        "vibeqc._vibeqc_core.semiempirical.indo"
    ).ccm_gradient_fd
    monkeypatch.setattr(py_ccm, "_cpp_ccm_gradient_fd_kernel", lambda: None)
    gc = np.array(ccm_grad(_He7_Z, _He7_coords, _He7_T, _PARAMS, madelung=False))
    gp = py_ccm.ccm_gradient_fd(
        _He7_Z, _He7_coords, _He7_T, madelung=False, conv_tol=1e-10
    )
    assert gc.shape == gp.shape
    assert np.max(np.abs(gc - gp)) < 1e-8


def test_ccm_gradient_fd_mgo_3d_non_madelung_matches_python(monkeypatch):
    """C++ CCM FD gradient stays finite on 3-D MgO without Madelung."""
    ccm_grad = pytest.importorskip(
        "vibeqc._vibeqc_core.semiempirical.indo"
    ).ccm_gradient_fd
    T3d = [[2 * a_mgo, 0, 0], [0, 2 * a_mgo, 0], [0, 0, 2 * a_mgo]]

    gc = np.asarray(
        ccm_grad(
            _MgO_Z, _MgO_coords, T3d, _PARAMS, madelung=False, conv_tol=1e-10
        )
    )
    with monkeypatch.context() as mp:
        mp.setattr(py_ccm, "_cpp_ccm_gradient_fd_kernel", lambda: None)
        gp = py_ccm.ccm_gradient_fd(
            _MgO_Z, _MgO_coords, T3d, madelung=False, conv_tol=1e-10
        )

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("Python CCM FD energy loop was used")

    monkeypatch.setattr(py_ccm, "_ccm_total_energy", _forbidden)
    g_public = py_ccm.ccm_gradient_fd(
        _MgO_Z, _MgO_coords, T3d, madelung=False, conv_tol=1e-10
    )

    assert gc.shape == gp.shape
    assert np.all(np.isfinite(gc))
    assert np.max(np.abs(gc - gp)) < 1e-8
    assert np.max(np.abs(g_public - gc)) < 1e-12


def test_public_ccm_gradient_fd_uses_cpp(monkeypatch):
    """The public CCM FD-gradient wrapper routes neutral jobs to C++."""
    ccm_grad = pytest.importorskip(
        "vibeqc._vibeqc_core.semiempirical.indo"
    ).ccm_gradient_fd

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("Python CCM FD energy loop was used")

    monkeypatch.setattr(py_ccm, "_ccm_total_energy", _forbidden)

    g_public = py_ccm.ccm_gradient_fd(
        _He7_Z, _He7_coords, _He7_T, madelung=False, conv_tol=1e-10
    )
    g_cpp = np.asarray(
        ccm_grad(_He7_Z, _He7_coords, _He7_T, _PARAMS, madelung=False)
    )

    assert g_public.shape == g_cpp.shape
    assert np.all(np.isfinite(g_public))
    assert np.max(np.abs(g_public - g_cpp)) < 1e-12


def test_cpp_ccm_charged_fd_gradient_matches_python(monkeypatch):
    """C++ CCM FD gradient honors charge and matches the Python reference path."""
    ccm_grad = pytest.importorskip(
        "vibeqc._vibeqc_core.semiempirical.indo"
    ).ccm_gradient_fd
    monkeypatch.setattr(py_ccm, "_cpp_ccm_gradient_fd_kernel", lambda: None)

    gc = np.asarray(
        ccm_grad(
            _He7_Z,
            _He7_charged_coords,
            _He7_T,
            _PARAMS,
            madelung=False,
            atoms=[0],
            charge=2,
        )
    )
    gp = py_ccm.ccm_gradient_fd(
        _He7_Z,
        _He7_charged_coords,
        _He7_T,
        charge=2,
        madelung=False,
        atoms=[0],
    )

    assert gc.shape == gp.shape
    assert np.max(np.abs(gc - gp)) < 1e-8


def test_public_ccm_gradient_analytic_uses_cpp(monkeypatch):
    """The public CCM analytic-gradient wrapper routes to the C++ kernel."""
    from vibeqc.semiempirical.methods import msindo_ccm_gradient_analytic as pg

    def _forbidden(*_args, **_kwargs):
        raise AssertionError("Python CCM gradient pair loop was used")

    monkeypatch.setattr(pg._pd, "_pair_blocks_deriv", _forbidden)

    g_public = pg.ccm_gradient_analytic(
        _He7_Z, _He7_coords, _He7_T, madelung=False, conv_tol=1e-10
    )
    g_cpp = np.asarray(
        _indo.ccm_gradient_analytic(
            _He7_Z, _He7_coords, _He7_T, _PARAMS, madelung=False, conv_tol=1e-10
        )
    )

    assert g_public.shape == g_cpp.shape
    assert np.all(np.isfinite(g_public))
    assert np.max(np.abs(g_public - g_cpp)) < 1e-12


# Analytic CCM gradient: C++ vs Python parity, 1-D/2-D/3-D x {NOEWALD, Madelung}.
_CCM_REF = json.loads(
    (_BASE / "examples" / "regression" / "msindo" / "ccm_reference.json").read_text())
_CCM_GC = {c["name"]: c for c in _CCM_REF["gradient_clusters"]}
_CCM_CLU = {c["name"]: c for c in _CCM_REF["clusters"]}


def _distort_ccm(C):
    C = np.asarray(C, float).copy()
    for i in range(len(C)):
        C[i] += 0.04 * ((-1.0) ** i) * np.array([1.0, -0.7, 0.5]) * (1 + 0.1 * i)
    return C


@pytest.mark.parametrize("name,src,madelung,distort", [
    ("hfionic_chain_distorted", _CCM_GC, False, False),       # 1-D NOEWALD
    ("hfionic_a1.6_n2_madelung", _CCM_CLU, True, True),       # 1-D Madelung
    ("mgo_100_slab", _CCM_GC, True, False),                   # 2-D Madelung
    ("mgo_bulk_distorted", _CCM_GC, True, False),             # 3-D Madelung
])
def test_ccm_gradient_analytic_matches_python(name, src, madelung, distort, monkeypatch):
    """The C++ analytic CCM gradient is byte-parity (<1e-8 Ha/bohr) with the
    validated Python ccm_gradient_analytic across 1-D/2-D/3-D and the
    direct-assembly Madelung gradient (1-D lattice sum, 2-D Parry-Heyes, 3-D
    Ewald)."""
    ccm_grad = pytest.importorskip(
        "vibeqc._vibeqc_core.semiempirical.indo").ccm_gradient_analytic
    from vibeqc.semiempirical.methods import msindo_ccm_gradient_analytic as pg
    monkeypatch.setattr(pg, "_cpp_ccm_gradient_kernel", lambda: None)
    c = src[name]
    Z = [_SYMBOL_Z[s] for s, *_ in c["real_atoms"]]
    C = np.array([[x, y, z] for _s, x, y, z in c["real_atoms"]], float)
    if distort:
        C = _distort_ccm(C)
    gc = np.array(ccm_grad(Z, C.tolist(), c["translations"], _PARAMS,
                           madelung=madelung, conv_tol=1e-10))
    gp = pg.ccm_gradient_analytic(Z, C, c["translations"], madelung=madelung,
                                  conv_tol=1e-10)
    assert gc.shape == gp.shape
    assert np.max(np.abs(gc - gp)) < 1e-8, (
        f"{name} madelung={madelung}: max|C++-Py|={np.max(np.abs(gc - gp)):.2e}")


# --------------------------------------------------------------------------- #
# Runner dispatch: NDDO.
# --------------------------------------------------------------------------- #


def test_runner_msindo_nddo():
    """method='msindo' with nddo=True routes to the C++ NDDO engine."""
    import vibeqc._vibeqc_core as core
    from vibeqc.runner import _run_semiempirical
    from vibeqc.semiempirical.methods.msindo import ANGSTROM_TO_BOHR as A2B

    mol = core.Molecule(
        [core.Atom(1, [0, 0, 0]), core.Atom(9, [0, 0, 0.917 * A2B])],
        charge=0,
        multiplicity=1,
    )
    r_indo = _run_semiempirical("msindo", mol, nddo=False)
    r_nddo = _run_semiempirical("msindo", mol, nddo=True)
    assert r_nddo.converged
    assert r_nddo.energy == pytest.approx(-23.0436304974, abs=1e-6)
    assert abs(r_nddo.energy - r_indo.energy) > 1.0


def test_runner_msindo_nddo_result_exposes_validated_analytic_gradient():
    """The unified closed-shell s/p NDDO result exposes its analytic gradient."""
    import vibeqc._vibeqc_core as core
    from vibeqc.runner import _run_semiempirical
    from vibeqc.semiempirical.methods.msindo import ANGSTROM_TO_BOHR as A2B

    mol = core.Molecule(
        [core.Atom(1, [0, 0, 0]), core.Atom(9, [0, 0, 0.917 * A2B])],
        charge=0,
        multiplicity=1,
    )
    r = _run_semiempirical("msindo", mol, nddo=True)
    assert r.converged
    expected = np.array([[0.0, 0.0, 0.0047202298], [0.0, 0.0, -0.0047202298]])
    gradient = r.gradient()
    assert gradient == pytest.approx(expected, abs=5e-8)
    assert r.gradient() is gradient


def test_explicit_msindo_nddo_gradient_route_supports_d_shell_scope():
    """The unified Al-Cl NDDO route exposes the source SPDD gradient."""
    import vibeqc._vibeqc_core as core
    from vibeqc.semiempirical.methods.msindo import ANGSTROM_TO_BOHR as A2B
    from vibeqc.semiempirical.routes import SemiempiricalRoutePlan
    from vibeqc.semiempirical.runner import _run_semiempirical_plan

    molecule = core.Molecule(
        [core.Atom(13, [0, 0, 0]), core.Atom(17, [0, 0, 2.10 * A2B])],
        charge=0,
        multiplicity=1,
    )
    plan = SemiempiricalRoutePlan.from_request(
        "msindo",
        nddo=True,
        properties=("energy", "gradient"),
    )
    result = _run_semiempirical_plan(plan, molecule)
    assert result.converged
    assert result.energy == pytest.approx(-16.4566380534, abs=1e-6)
    np.testing.assert_allclose(
        result.gradient(),
        np.array([[0.0, 0.0, 0.0317797727], [0.0, 0.0, -0.0317797727]]),
        atol=5e-8,
        rtol=0.0,
    )


def test_run_job_msindo_nddo_d_shell_energy_and_gradient(tmp_path):
    """The public run_job route exposes the Al-Cl NDDO energy and gradient."""
    import vibeqc as vq
    from vibeqc.semiempirical.methods.msindo import ANGSTROM_TO_BOHR as A2B

    molecule = vq.Molecule(
        [vq.Atom(13, [0, 0, 0]), vq.Atom(17, [0, 0, 2.10 * A2B])],
        charge=0,
        multiplicity=1,
    )
    result = vq.run_job(
        molecule,
        method="msindo",
        nddo=True,
        output=str(tmp_path / "alcl-nddo"),
        verbose=0,
    )
    gradient = result.gradient()

    assert result.converged
    assert result.energy == pytest.approx(-16.4566380534, abs=1e-6)
    np.testing.assert_allclose(
        gradient,
        np.array([[0.0, 0.0, 0.0317797727], [0.0, 0.0, -0.0317797727]]),
        atol=5e-8,
        rtol=0.0,
    )
    assert result.gradient() is gradient


def test_runner_msindo_nddo_open_shell_rejected():
    """Runner keeps the direct API's closed-shell-only NDDO boundary."""
    import vibeqc._vibeqc_core as core
    from vibeqc.runner import _run_semiempirical
    from vibeqc.semiempirical.methods.msindo import ANGSTROM_TO_BOHR as A2B

    mol = core.Molecule(
        [core.Atom(8, [0, 0, 0]), core.Atom(7, [0, 0, 1.15 * A2B])],
        charge=0,
        multiplicity=2,
    )
    with pytest.raises(NotImplementedError, match="closed-shell"):
        _run_semiempirical("msindo", mol, nddo=True)
