"""GFN2-xTB regression suite — invariants + documented reference targets.

GFN2-xTB is **experimental** in vibe-qc (see
``python/vibeqc/semiempirical/methods/gfn2.py``): it implements H0, pairwise
repulsion, shell-resolved 2nd-order isotropic electrostatics, the third-order
on-site term, and an opt-in atom-resolved AES path, but it is still not at
quantitative tblite/xtb parity. This file pins the invariants the implementation
*does* guarantee
(size-consistency, charge signs, repulsion decay, determinism, the
experimental gate, isolated-atom energies) and records, as ``xfail``, the
remaining quantitative ``xtb`` reference targets the full method must
eventually reach.
The translated-water GFN2 oracle is no longer an xfail: it is a strict active
parity gate for the H0/AES short-range interaction.

The 2026-06-01 fix resolved the GFN2 H⁰ deep-state bug:
- H0 diagonal: H0_μμ = EN_l · S_μμ (EN_l is for normalized AOs)
- Wolfsberg-Helmholtz: k_ll' from GFN2-xTB table (k_ss=1.85, k_sp=2.04, etc.)
- H₂ and CH₄ are now within 0.5 Ha of xtb reference.
- Isolated Ne is at −5.93 Ha (was −4974 Ha).
- H₂O is inside the deliberately loose 0.5 Ha total-energy gate; the translated
  water-dimer oracle is the strict short-range H0/AES parity gate.

Run:
    .venv/bin/python -m pytest tests/test_gfn2_xtb.py -q
"""

from __future__ import annotations

import tomllib
import urllib.error
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_gfn2_available = False
_gfn2_params = None
try:
    from vibeqc import Atom, Molecule
    from vibeqc._vibeqc_core import compute_overlap as _compute_overlap
    from vibeqc._vibeqc_core import semiempirical as _semi
    from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
    from vibeqc.semiempirical.methods import gfn2 as _gfn2_method
    from vibeqc.semiempirical.methods import gfn2_params as _gfn2_params
    from vibeqc.semiempirical.methods.gfn2 import (
        GFN2D4UnsupportedWarning,
        GFN2ExperimentalWarning,
        GFN2Model,
    )
    from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

    _gfn2_available = True
except Exception:  # pragma: no cover - import guard
    pass

requires_gfn2 = pytest.mark.skipif(not _gfn2_available, reason="GFN2-xTB unavailable")


# ---------------------------------------------------------------------------
# History note (2026-08-14): the former _STO34G_GRADIENT_FD_REASON xfail block
# is retired.  The gradient-vs-FD residual it described had two layers:
# (1) two basis-plumbing bugs fixed in the 2026-08-06 pass (per-element
# primitive count; stale hardcoded-6-primitive basis inside the gradient/
# stress kernels); (2) the BUG-022 M-matrix SCC sign flip (+½ -> -½ at all
# three call sites) on the false premise that the Hamiltonian is
# H⁰ - ½S(v+v').  Both drivers actually build H_scc = H⁰ + ½S(v+v'), so the
# minus sign mis-differentiated the converged surface (max|analytic-FD| ~
# 3e-2 Ha/bohr).  Restoring the plus sign closes the residual to ~1e-9
# Ha/bohr and the two strict xfail gates below now run as hard passes.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Geometries (bohr)
# ---------------------------------------------------------------------------
def _h2(x0=0.0):
    return [Atom(1, [x0, 0, 0]), Atom(1, [x0 + 1.4, 0, 0])]


def _water(x0=0.0):
    theta = np.deg2rad(104.5 / 2)
    r = 1.81
    return [
        Atom(8, [x0, 0.0, 0.0]),
        Atom(1, [x0 + r * np.sin(theta), r * np.cos(theta), 0.0]),
        Atom(1, [x0 - r * np.sin(theta), r * np.cos(theta), 0.0]),
    ]


def _mgo_periodic_system(
    *, oxygen_image=0, magnesium_image=0, permuted=False
):
    """MgO primitive used for periodic GFN2 failure-result coverage."""
    from vibeqc._vibeqc_core import PeriodicSystem

    a = 4.211 * 1.8897261254535
    lattice = np.array(
        [
            [0.0, a / 2.0, a / 2.0],
            [a / 2.0, 0.0, a / 2.0],
            [a / 2.0, a / 2.0, 0.0],
        ]
    )
    atoms = [
        Atom(12, (magnesium_image * lattice[:, 1]).tolist()),
        Atom(
            8,
            (
                np.array([a / 2.0, 0.0, 0.0])
                + oxygen_image * lattice[:, 0]
            ).tolist(),
        ),
    ]
    if permuted:
        atoms.reverse()
    return PeriodicSystem(
        3,
        lattice,
        atoms,
        0,
        1,
    )


def _periodic_gfn2_components(result):
    return np.array(
        [
            result.energy,
            result.free_energy,
            result.e_electronic,
            result.e_repulsive,
            result.e_scc,
            result.e_band0,
            result.e_aes,
            result.e_3rd,
            result.fermi_level,
            result.entropy,
        ]
    )


def _water_axis_aligned():
    """Asymmetric water with one O-H bond exactly on the global +x axis.

    Regression geometry for GFN2-GRADIENT-FD-RESIDUAL: at this alignment
    S(p_y, s_H1) is exactly zero, and the H0 derivative F = dH0/dS must
    still contribute through dS/dR.  The former |S| < 1e-15 skip in
    gfn2_h0_image_derivative_terms dropped that pair at exactly 0 deg but
    kept it at any nonzero angle, so the analytic gradient jumped by
    ~2.7e-2 Ha/bohr across the alignment point while the symmetric suite
    geometries (no exact axis alignment) passed the FD gate.
    """
    return [
        Atom(8, [0.0, 0.0, 0.0]),
        Atom(1, [1.8089, 0.0, 0.0]),
        Atom(1, [-0.4730, 1.7390, 0.0]),
    ]


def _ch4():
    a = 2.0 / np.sqrt(3)
    return [
        Atom(6, [0, 0, 0]),
        Atom(1, [a, a, a]),
        Atom(1, [-a, -a, a]),
        Atom(1, [-a, a, -a]),
        Atom(1, [a, -a, -a]),
    ]


def _nh3():
    return [
        Atom(7, [0.0, 0.0, 0.22]),
        Atom(1, [1.77, 0.0, -0.52]),
        Atom(1, [-0.885, 1.5329, -0.52]),
        Atom(1, [-0.885, -1.5329, -0.52]),
    ]


_ANGSTROM_TO_BOHR = 1.8897259885789233
_TAE_PTCOMP_ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def _from_angstrom(spec):
    return [
        Atom(z, np.asarray(xyz, dtype=float) * _ANGSTROM_TO_BOHR)
        for z, xyz in spec
    ]


def _from_tae_ptcomp_angstrom(spec):
    return [
        Atom(z, np.asarray(xyz, dtype=float) * _TAE_PTCOMP_ANGSTROM_TO_BOHR)
        for z, xyz in spec
    ]


def _bug45_pyridine():
    """Minimal BUG 45 finite-temperature-retry reproducer."""
    return _from_angstrom(
        [
            (7, (0.0, 1.392, 0.0)),
            (6, (1.196, 0.697, 0.0)),
            (6, (-1.196, 0.697, 0.0)),
            (6, (1.201, -0.698, 0.0)),
            (6, (-1.201, -0.698, 0.0)),
            (6, (0.0, -1.39, 0.0)),
            (1, (2.145, 1.234, 0.0)),
            (1, (-2.145, 1.234, 0.0)),
            (1, (2.16, -1.223, 0.0)),
            (1, (-2.16, -1.223, 0.0)),
            (1, (0.0, -2.48, 0.0)),
        ]
    )


def _bug45_cyclopropene():
    """Minimal BUG 45 slow-zero-temperature-retry reproducer."""
    return _from_angstrom(
        [
            (6, (0.0, 0.0, 0.865)),
            (6, (0.0, 0.664, -0.321)),
            (6, (0.0, -0.664, -0.321)),
            (1, (0.0, 0.0, 1.942)),
            (1, (0.0, 1.51, -0.67)),
            (1, (0.916, 0.0, -0.67)),
            (1, (-0.916, 0.0, -0.67)),
        ]
    )


_TAE_PTCOMP_CASES = {
    "s_li_1": {
        "charge": -1,
        "mult": 1,
        "atoms": [
            (3, (-1.323942, -1.4110954, 0.0)),
            (6, (0.4784364, 1.1238911, 0.0)),
            (6, (0.514244, -0.4179763, 0.0)),
            (1, (1.4655421, 1.6342561, 0.0)),
            (1, (-0.0610212, 1.5126684, -0.8768538)),
            (1, (-0.0610212, 1.5126684, 0.8768538)),
            (1, (1.1141894, -0.7423707, -0.8706396)),
            (1, (1.1141894, -0.7423707, 0.8706396)),
            (17, (-3.240617, -2.4696708, 0.0)),
        ],
    },
    "s_k_1": {
        "charge": -1,
        "mult": 1,
        "atoms": [
            (19, (0.0, 0.0, 0.0)),
            (1, (0.0, 0.0, 2.4758559)),
            (1, (0.0, 0.0, -2.4758559)),
        ],
    },
    "pheavy_as_1": {
        "charge": -1,
        "mult": 1,
        "atoms": [
            (33, (0.0, 0.0, 0.0)),
            (17, (-2.2896461, 0.0, 0.0)),
            (17, (2.2896461, 0.0, 0.0)),
            (17, (0.0, 0.0, -2.2896461)),
            (17, (0.0, 0.0, 2.2896461)),
            (17, (0.0, 2.2896461, 0.0)),
            (17, (0.0, -2.2896461, 0.0)),
        ],
    },
    "pheavy_br_1": {
        "charge": -1,
        "mult": 1,
        "atoms": [
            (13, (0.0, 0.0, 0.0)),
            (35, (1.3497361, 1.3497361, -1.3497361)),
            (35, (-1.3497361, -1.3497361, -1.3497361)),
            (35, (1.3497361, -1.3497361, 1.3497361)),
            (35, (-1.3497361, 1.3497361, 1.3497361)),
        ],
    },
    "pheavy_rn_1": {
        "charge": -1,
        "mult": 1,
        "atoms": [
            (86, (0.0, 0.0, -3.2362433)),
            (6, (-0.9719961, -0.7061965, 0.3256841)),
            (6, (-0.9719961, 0.7061965, 0.3256841)),
            (6, (0.3712695, 1.1426499, 0.3256841)),
            (6, (1.2014532, 0.0, 0.3256841)),
            (6, (0.3712695, -1.1426499, 0.3256841)),
            (1, (0.7069889, -2.175888, 0.3215646)),
            (1, (2.287864, 0.0, 0.3215646)),
            (1, (0.7069889, 2.175888, 0.3215646)),
            (1, (-1.8509208, 1.3447727, 0.3215646)),
            (1, (-1.8509208, -1.3447727, 0.3215646)),
        ],
    },
}


def _tae_ptcomp_mol(name: str) -> Molecule:
    case = _TAE_PTCOMP_CASES[name]
    return _mol(
        _from_tae_ptcomp_angstrom(case["atoms"]),
        charge=case["charge"],
        mult=case["mult"],
    )


def _mol(atoms, charge=0, mult=1):
    return Molecule(atoms, charge=charge, multiplicity=mult)


def _z_minus_noble(Z: int) -> int:
    """The naive (wrong, past a completed d/f series) ``Z − previous noble gas``
    valence count — used only to make the corrected-count assertions informative.
    """
    core = 0
    for g in (2, 10, 18, 36, 54, 86):
        if g < Z:
            core = g
        else:
            break
    return Z - core


@pytest.fixture(scope="module")
def params():
    return load_gfn2_params()


def _energy(mol, params, max_iter=400, aes_faithful=False):
    opts = _xtb.XTBSccOptions()
    opts.charge_mixing = 0.2
    opts.max_iter = max_iter
    opts.aes_faithful = aes_faithful
    return _xtb.run_gfn2_xtb(mol, params, opts)


def _diatomic(z1: int, z2: int, r: float) -> Molecule:
    return _mol([Atom(z1, [0.0, 0.0, 0.0]), Atom(z2, [r, 0.0, 0.0])])


def _write_downgraded_gfn2_cache(cache_file) -> str:
    text = "\n".join(
        [
            "[[element]]",
            "Z = 1",
            "shells = [{ l = 0, en = -0.4, zeta = 1.0, k_en = 1.0 }]",
            "gam = 0.5",
            "gam3 = 0.0",
        ]
    )
    cache_file.write_text(text, encoding="utf-8")
    return text


def test_gfn2_cache_dir_honours_xdg_and_explicit_override(monkeypatch, tmp_path):
    """The GFN2 cache root is seedable, not hardwired to $HOME (issue #127).

    An offline compute node can only run GFN2 if a cache was put somewhere
    it will look. Every other vibe-qc cache resolves an override, then
    ``$XDG_CACHE_HOME``, then ``~/.cache``; GFN2 resolved ``Path.home()``
    unconditionally, so the documented "rsync ``$XDG_CACHE_HOME/vibeqc/``
    to the cluster" remedy could not reach it and every host had to fetch
    its own copy -- which is how 78 rows failed at once when the compute-cluster nodes
    lost outbound network.

    Not a full fix for #127: seeding still has to happen, and bundling the
    LGPL-3.0 upstream file is a maintainer licensing decision. What changes
    here is that seeding is now *possible*.
    """
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.delenv("VIBEQC_GFN2_CACHE_DIR", raising=False)
    assert _gfn2_params._default_cache_dir() == tmp_path / "xdg" / "vibeqc"

    # An explicit override wins over XDG, matching the precedence
    # vibeqc.fetch.cache.cache_root gives VIBEQC_FETCH_CACHE_ROOT.
    monkeypatch.setenv("VIBEQC_GFN2_CACHE_DIR", str(tmp_path / "explicit"))
    assert _gfn2_params._default_cache_dir() == tmp_path / "explicit"

    # Negative control -- the same call with the feature off. With neither
    # variable set the resolution must stay byte-identical to the pre-#127
    # behaviour, so no existing deployment's cache moves out from under it.
    monkeypatch.delenv("VIBEQC_GFN2_CACHE_DIR", raising=False)
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    assert (
        _gfn2_params._default_cache_dir()
        == Path.home() / ".cache" / "vibeqc"
    )


def test_gfn2_module_cache_constants_follow_the_environment(tmp_path):
    """``_CACHE_DIR`` / ``_CACHE_FILE`` themselves move, not just the helper.

    Every read and write in this module goes through those two constants, so
    a resolver nothing consults would be dead code. A fresh interpreter with
    the variable set is what a job on a compute node actually is: the
    environment comes from the scheduler and the import follows it.

    Run out of process rather than via ``importlib.reload`` on purpose --
    reloading would also reset this module's parameter cache and its
    ``WeakKeyDictionary`` lineage map, which other tests in this file hold
    live references into.
    """
    import json
    import os
    import subprocess
    import sys

    probe = (
        "import json;"
        "from vibeqc.semiempirical.methods import gfn2_params as g;"
        "print(json.dumps([str(g._CACHE_DIR), str(g._CACHE_FILE)]))"
    )

    env = dict(os.environ)
    env.pop("VIBEQC_GFN2_CACHE_DIR", None)
    env["XDG_CACHE_HOME"] = str(tmp_path / "seeded")
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, check=True, env=env,
    ).stdout
    cache_dir, cache_file = json.loads(out.strip().splitlines()[-1])
    assert Path(cache_dir) == tmp_path / "seeded" / "vibeqc"
    assert Path(cache_file) == (
        tmp_path / "seeded" / "vibeqc" / "gfn2_xtb_params.toml"
    )

    # Negative control: the same probe with the feature off resolves the
    # pre-#127 path, so no existing deployment's cache moves.
    env.pop("XDG_CACHE_HOME", None)
    out = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, check=True, env=env,
    ).stdout
    cache_dir, _ = json.loads(out.strip().splitlines()[-1])
    assert Path(cache_dir) == Path.home() / ".cache" / "vibeqc"


@requires_gfn2
def test_gfn2_parameter_cache_miss_offline_error_is_structured(monkeypatch, tmp_path):
    cache_file = tmp_path / "gfn2_xtb_params.toml"
    monkeypatch.setattr(_gfn2_params, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(_gfn2_params, "_CACHE_FILE", cache_file)

    def fail_urlopen(*_args, **_kwargs):
        raise urllib.error.URLError("temporary DNS failure")

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

    with pytest.raises(
        _gfn2_params.GFN2ParameterUnavailableError,
        match="GFN2ParameterUnavailable",
    ) as excinfo:
        _gfn2_params.load_gfn2_params()

    message = str(excinfo.value)
    assert str(cache_file) in message
    assert "not present in the local vibe-qc cache" in message
    assert "populate this cache" in message
    assert "An upstream refresh was attempted and failed." in message
    assert "URLError" not in message
    assert "temporary DNS failure" not in message


@requires_gfn2
def test_gfn2_parameter_cache_only_miss_does_not_fetch(monkeypatch, tmp_path):
    cache_file = tmp_path / "gfn2_xtb_params.toml"
    monkeypatch.setattr(_gfn2_params, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(_gfn2_params, "_CACHE_FILE", cache_file)

    def fail_fetch():
        raise AssertionError("cache-only GFN2 preflight must not fetch")

    monkeypatch.setattr(_gfn2_params, "_fetch_and_parse", fail_fetch)

    assert _gfn2_params.gfn2_parameter_cache_available() is False
    with pytest.raises(
        _gfn2_params.GFN2ParameterUnavailableError,
        match="not present in the local vibe-qc cache",
    ) as excinfo:
        _gfn2_params.load_gfn2_params(allow_fetch=False)
    assert str(cache_file) in str(excinfo.value)
    assert "populate this cache" in str(excinfo.value)


@requires_gfn2
@pytest.mark.parametrize(
    "cache_text",
    [
        "[[element]\nZ = 1\n",
        "element = [1]\n",
        "[[element]]\nZ = 'hydrogen'\nshells = [{ n = 1, l = 0 }]\n",
        "[[element]]\nZ = inf\nshells = [{ n = 1, l = 0 }]\n",
        "[[element]]\nZ = 1\nshells = [{ n = 1, l = 0, zeta = nan }]\n",
    ],
    ids=[
        "invalid-toml",
        "non-table-element",
        "invalid-numeric-record",
        "overflowing-integer-record",
        "nonfinite-float-record",
    ],
)
def test_gfn2_parameter_cache_only_reports_malformed_cache(
    monkeypatch,
    tmp_path,
    cache_text,
):
    cache_file = tmp_path / "gfn2_xtb_params.toml"
    cache_file.write_text(cache_text, encoding="utf-8")
    monkeypatch.setattr(_gfn2_params, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(_gfn2_params, "_CACHE_FILE", cache_file)

    def fail_fetch():
        raise AssertionError("cache-only GFN2 preflight must not fetch")

    monkeypatch.setattr(_gfn2_params, "_fetch_and_parse", fail_fetch)

    assert _gfn2_params.gfn2_parameter_cache_available() is False
    with pytest.raises(
        _gfn2_params.GFN2ParameterUnavailableError,
        match="present but unreadable or malformed",
    ) as excinfo:
        _gfn2_params.load_gfn2_params(allow_fetch=False)

    message = str(excinfo.value)
    assert str(cache_file) in message
    assert "remove or repair that file" in message
    assert "TOMLDecodeError" not in message


@requires_gfn2
def test_gfn2_parameter_cache_only_rejects_forced_refresh(monkeypatch):
    def fail_fetch():
        raise AssertionError("cache-only GFN2 preflight must not fetch")

    monkeypatch.setattr(_gfn2_params, "_fetch_and_parse", fail_fetch)
    with pytest.raises(ValueError, match="force_refetch=True"):
        _gfn2_params.load_gfn2_params(force_refetch=True, allow_fetch=False)


@requires_gfn2
def test_gfn2_parameter_cache_only_rejects_stale_cache(monkeypatch, tmp_path):
    cache_file = tmp_path / "gfn2_xtb_params.toml"
    _write_downgraded_gfn2_cache(cache_file)
    monkeypatch.setattr(_gfn2_params, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(_gfn2_params, "_CACHE_FILE", cache_file)
    monkeypatch.setattr(
        _gfn2_params,
        "_fetch_and_parse",
        lambda: (_ for _ in ()).throw(
            AssertionError("cache-only GFN2 preflight must not refresh")
        ),
    )

    assert _gfn2_params.gfn2_parameter_cache_available() is False
    with pytest.raises(
        _gfn2_params.GFN2ParameterUnavailableError,
        match="parameter cache is stale",
    ) as excinfo:
        _gfn2_params.require_gfn2_parameter_cache()
    message = str(excinfo.value)
    assert str(cache_file) in message
    assert "present but out of date" in message
    assert "predates the shell principal-quantum-number field" in message
    assert "refresh that file" in message
    with pytest.raises(
        _gfn2_params.GFN2ParameterUnavailableError,
        match="parameter cache is stale",
    ):
        _gfn2_params.load_gfn2_params(allow_fetch=False)


@requires_gfn2
def test_gfn2_stale_cache_diagnosis_survives_failed_refresh(monkeypatch, tmp_path):
    cache_file = tmp_path / "gfn2_xtb_params.toml"
    _write_downgraded_gfn2_cache(cache_file)
    monkeypatch.setattr(_gfn2_params, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(_gfn2_params, "_CACHE_FILE", cache_file)

    def fail_refresh():
        raise _gfn2_params._GFN2ParameterRefreshError

    monkeypatch.setattr(_gfn2_params, "_fetch_and_parse", fail_refresh)

    with pytest.raises(
        _gfn2_params.GFN2ParameterUnavailableError,
        match="parameter cache is stale",
    ) as excinfo:
        _gfn2_params.load_gfn2_params()

    message = str(excinfo.value)
    assert str(cache_file) in message
    assert "present but out of date" in message
    assert "predates the shell principal-quantum-number field" in message
    assert "An upstream refresh was attempted and failed." in message
    assert "refresh that file" in message
    assert "not present in the local vibe-qc cache" not in message


@requires_gfn2
def test_gfn2_stale_cache_diagnosis_survives_invalid_upstream(monkeypatch, tmp_path):
    cache_file = tmp_path / "gfn2_xtb_params.toml"
    original_text = _write_downgraded_gfn2_cache(cache_file)
    monkeypatch.setattr(_gfn2_params, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(_gfn2_params, "_CACHE_FILE", cache_file)

    class InvalidUTF8Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b"\xff"

    import urllib.request

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: InvalidUTF8Response(),
    )

    with pytest.raises(
        _gfn2_params.GFN2ParameterUnavailableError,
        match="parameter cache is stale",
    ) as excinfo:
        _gfn2_params.load_gfn2_params()

    message = str(excinfo.value)
    assert str(cache_file) in message
    assert "predates the shell principal-quantum-number field" in message
    assert "An upstream refresh was attempted and failed." in message
    assert "UnicodeDecodeError" not in message
    assert cache_file.read_text(encoding="utf-8") == original_text


@requires_gfn2
def test_gfn2_noncurrent_refresh_preserves_stale_cache(monkeypatch, tmp_path):
    cache_file = tmp_path / "gfn2_xtb_params.toml"
    original_text = _write_downgraded_gfn2_cache(cache_file)
    monkeypatch.setattr(_gfn2_params, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(_gfn2_params, "_CACHE_FILE", cache_file)

    incomplete_upstream = "\n".join(
        [
            "$Z=1",
            "ao=1s",
            "lev=-13.6",
            "exp=1.0",
            "GAM=0.5",
            "REPA=1.0",
            "REPB=1.0",
            "$end",
        ]
    )

    class IncompleteResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return incomplete_upstream.encode("utf-8")

    import urllib.request

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: IncompleteResponse(),
    )

    with pytest.raises(
        _gfn2_params.GFN2ParameterUnavailableError,
        match="parameter cache is stale",
    ) as excinfo:
        _gfn2_params.load_gfn2_params()

    message = str(excinfo.value)
    assert "predates the shell principal-quantum-number field" in message
    assert "predates shell-Hubbard multipliers" not in message
    assert "An upstream refresh was attempted and failed." in message
    assert cache_file.read_text(encoding="utf-8") == original_text
    assert list(tmp_path.glob(f".{cache_file.name}.*")) == []


@requires_gfn2
def test_gfn2_failed_cache_replace_preserves_stale_file(monkeypatch, tmp_path):
    cache_file = tmp_path / "gfn2_xtb_params.toml"
    original_text = _write_downgraded_gfn2_cache(cache_file)
    monkeypatch.setattr(_gfn2_params, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(_gfn2_params, "_CACHE_FILE", cache_file)

    upstream_lines = []
    for atomic_number in range(1, 87):
        upstream_lines.extend(
            [
                f"$Z={atomic_number}",
                "ao=1s 2p",
                "lev=-13.6 -5.0",
                "exp=1.0 1.0",
                "GAM=0.5",
                "REPA=1.0",
                "REPB=1.0",
                "$end",
            ]
        )
    upstream_text = "\n".join(upstream_lines)

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return upstream_text.encode("utf-8")

    import urllib.request

    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(),
    )
    path_type = type(cache_file)
    original_replace = path_type.replace

    def fail_cache_replace(path, target):
        if target == cache_file:
            raise OSError("simulated atomic replacement failure")
        return original_replace(path, target)

    monkeypatch.setattr(path_type, "replace", fail_cache_replace)

    with pytest.raises(
        _gfn2_params.GFN2ParameterUnavailableError,
        match="parameter cache is stale",
    ) as excinfo:
        _gfn2_params.load_gfn2_params()

    message = str(excinfo.value)
    assert str(cache_file) in message
    assert "predates the shell principal-quantum-number field" in message
    assert "local cache update could not be written" in message
    assert "cache directory is writable and has free space" in message
    assert "upstream refresh was attempted and failed" not in message
    assert cache_file.read_text(encoding="utf-8") == original_text
    assert list(tmp_path.glob(f".{cache_file.name}.*")) == []


@requires_gfn2
def test_gfn2_parameter_loader_reuses_validated_toml_cache(monkeypatch, tmp_path):
    cache_file = tmp_path / "gfn2_xtb_params.toml"
    cache_file.write_text(
        "\n".join(
            [
                # Current caches carry the per-l projection marker
                # (issue #43); without it the staleness gate refuses.
                'projection = "per-l"',
                "[[element]]",
                "Z = 1",
                "shells = [",
                "  { n = 1, l = 0, en = -0.4, zeta = 1.0, k_en = 1.0 },",
                "  { n = 2, l = 1, en = -0.2, zeta = 1.0, k_en = 1.1 },",
                "]",
                "gam = 0.5",
                "gam3 = 0.0",
                "repa = 1.0",
                "repb = 1.0",
                "dpol = 0.0",
                "qpol = 0.0",
                "mp_rad = 1.4",
                "mp_vcn = 1.0",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(_gfn2_params, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(_gfn2_params, "_CACHE_FILE", cache_file)
    monkeypatch.setattr(_gfn2_params, "_PARAM_TOML_CACHE", None)
    original_load = _gfn2_params.tomllib.load
    load_calls = []

    def counted_load(file_obj):
        load_calls.append(file_obj.name)
        return original_load(file_obj)

    monkeypatch.setattr(_gfn2_params.tomllib, "load", counted_load)

    first = _gfn2_params.load_gfn2_params(allow_fetch=False)
    second = _gfn2_params.load_gfn2_params(allow_fetch=False)

    assert first is not second
    assert first.n_elements() == 1
    assert second.n_elements() == 1
    assert load_calls == [str(cache_file)]


@requires_gfn2
def test_gfn2_route_runtime_availability_is_cache_only(monkeypatch, tmp_path):
    from vibeqc.semiempirical import semiempirical_route_runtime_available

    monkeypatch.setattr(_gfn2_params, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(_gfn2_params, "_CACHE_FILE", tmp_path / "gfn2_xtb_params.toml")

    assert semiempirical_route_runtime_available("gfn2") is False
    assert semiempirical_route_runtime_available("periodic-gfn2-gradient") is False
    assert semiempirical_route_runtime_available("scc-dftb") is True


def test_gfn2_multipole_vcn_table_matches_tblite_water_elements():
    assert _gfn2_params._MP_VCN[1] == pytest.approx(1.0)
    assert _gfn2_params._MP_VCN[8] == pytest.approx(2.0)


def _gfn2_shells(mol, params):
    basis = _semi.SemiempiricalBasis.build(mol, params, 6)
    return _semi.gfn2_enumerate_shells(basis, mol, params)


class TestGFN2STONGBasisNormalization:
    """Contracted self-overlap of the semiempirical STO-NG basis must be
    unity.  This is the cheapest guard on the whole basis-table class: the
    `coefficients_pre_normalized` flag defect (fbb23f74d) showed up as a
    135x inconsistency between O 2p and H 1s overlap diagonals, and any
    malformed table or normalization-convention flip trips it at once.

    The 3G/4G tables (GFN2) carry Stewart (1970) contraction weights with
    self-overlap 1.000000; the 6G rows are the known-bad non-HSP tables
    tracked as DFTB-STO-NG-TABLES-NOT-HSP in agentic-loop/bug-claims.md
    (raw self-overlap 0.832/0.650, pending a maintainer decision).
    libint's enforce-unit-normalization rescales every contraction at
    runtime, so the runtime diagonal is pinned to 1 +/- 1e-5 for all
    (NG, l); this test pins the runtime behaviour every consumer sees.
    """

    @staticmethod
    def _h2o_mol():
        return _mol(_water())

    @requires_gfn2
    @pytest.mark.parametrize("n_primitives", [3, 4, 6])
    def test_runtime_overlap_diagonal_is_unity(self, params, n_primitives):
        mol = self._h2o_mol()
        basis = _semi.SemiempiricalBasis.build(mol, params, n_primitives)
        S = np.asarray(_compute_overlap(basis))
        assert S.shape[0] == S.shape[1]
        diag = np.diag(S)
        assert np.max(np.abs(diag - 1.0)) < 1e-5, (
            f"STO-{n_primitives}G overlap diagonal is not unity: "
            f"max|S_mu,mu - 1| = {np.max(np.abs(diag - 1.0)):.3e}"
        )
        # Sanity: the off-diagonal is a real overlap, not a trivial identity.
        assert np.max(np.abs(S - np.eye(S.shape[0]))) > 1e-3

    @requires_gfn2
    def test_gfn2_auto_primitive_counts_build_unit_normalized_basis(self, params):
        """The GFN2 per-element auto count (n_primitives=0: H,He->3, else->4)
        must produce a unit-normalized runtime basis, matching the basis
        run_gfn2_xtb converges on."""
        mol = self._h2o_mol()
        basis = _semi.SemiempiricalBasis.build(mol, params, 0)
        S = np.asarray(_compute_overlap(basis))
        diag = np.diag(S)
        assert np.max(np.abs(diag - 1.0)) < 1e-5



class TestGFN2ShellCoulombKernel:
    @requires_gfn2
    def test_shell_hubbard_multipliers_are_parsed_from_lpar(self, params):
        """The official xtb GFN2 parameter file stores p/d shell-Hubbard
        multipliers as LPARP/LPARD deltas in tenths; they must not collapse to
        1.0 or be used raw."""
        shells = _gfn2_shells(_mol(_water()), params)
        oxygen_p = next(sh for sh in shells if sh.Z == 8 and sh.l == 1)
        assert oxygen_p.k_en == pytest.approx(1.149702, rel=1e-7)
        assert oxygen_p.hardness == pytest.approx(0.451896 * 1.149702, rel=1e-7)

    @requires_gfn2
    def test_shell_gamma_uses_arithmetic_effective_coulomb(self, params):
        """GFN2 second-order shell electrostatics use tblite's
        effective-Coulomb kernel with arithmetic-averaged hardnesses."""
        mol = Molecule([Atom(8, [0.0, 0.0, 0.0]), Atom(8, [3.0, 0.0, 0.0])])
        shells = _gfn2_shells(mol, params)
        gamma = np.asarray(_semi.build_gfn2_shell_gamma(shells, mol, params))

        oxygen_s = next(
            i for i, sh in enumerate(shells) if sh.Z == 8 and sh.l == 0 and sh.atom_idx == 0
        )
        oxygen_p = next(
            i for i, sh in enumerate(shells) if sh.Z == 8 and sh.l == 1 and sh.atom_idx == 0
        )
        oxygen_p_b = next(
            i for i, sh in enumerate(shells) if sh.Z == 8 and sh.l == 1 and sh.atom_idx == 1
        )

        hs = shells[oxygen_s].hardness
        hp = shells[oxygen_p].hardness

        assert gamma[oxygen_s, oxygen_p] == pytest.approx(0.5 * (hs + hp), rel=1e-12)

        r_oh = np.linalg.norm(
            np.asarray(mol.atoms[0].xyz, dtype=float)
            - np.asarray(mol.atoms[1].xyz, dtype=float)
        )
        gab = 0.5 * (hp + shells[oxygen_p_b].hardness)
        expected = 1.0 / np.sqrt(r_oh * r_oh + 1.0 / (gab * gab))
        assert gamma[oxygen_p, oxygen_p_b] == pytest.approx(expected, rel=1e-12)

    @requires_gfn2
    def test_positive_gamma3_shell_gamma_uses_published_arithmetic_kernel(self, params):
        """Positive-GAM3 elements use the same published shell-Hubbard scaling
        and arithmetic effective-Coulomb kernel as the rest of GFN2."""
        mol = _mol(_ch4())
        shells = _gfn2_shells(mol, params)
        gamma = np.asarray(_semi.build_gfn2_shell_gamma(shells, mol, params))

        carbon_p = next(i for i, sh in enumerate(shells) if sh.Z == 6 and sh.l == 1)
        hydrogen_s = next(i for i, sh in enumerate(shells) if sh.Z == 1)
        hp = shells[carbon_p].hardness
        hh = shells[hydrogen_s].hardness
        assert shells[carbon_p].k_en == pytest.approx(1.1056358, rel=1e-7)
        r_ch = np.linalg.norm(
            np.asarray(mol.atoms[0].xyz, dtype=float)
            - np.asarray(mol.atoms[shells[hydrogen_s].atom_idx].xyz, dtype=float)
        )
        gab = 0.5 * (hp + hh)
        expected = 1.0 / np.sqrt(r_ch * r_ch + 1.0 / (gab * gab))
        assert gamma[carbon_p, hydrogen_s] == pytest.approx(expected, rel=1e-12)


class TestGFN2GammaGradientFD:
    """Pin the gamma-derivative gradient term against central finite
    differences of the second-order energy E2 = ½ Δq·γ(R)·Δq at fixed
    charges.  This is the term that was halved pre-fix (2026-07-09): an
    ordered (a,b) loop deposits each pair's derivative into grad(a) only,
    so the correct per-atom force is Σ_B Δq_A Δq_B ∂γ_AB/∂R_A with NO ½
    prefactor — the ½ in E2 is cancelled by γ_AB appearing twice in the
    double sum.  Also pins ∂γ/∂R = −R·γ³ for the Klopman-Ohno kernel
    γ = 1/sqrt(R² + η⁻²) of build_gfn2_gamma (same defect family as the
    SCC-DFTB gamma force fixed the same day)."""

    @requires_gfn2
    def test_gamma_term_matches_fd_no_half_prefactor(self, params):
        coords0 = np.array([np.asarray(a.xyz, dtype=float) for a in _water()])
        dq = np.array([-0.6, 0.3, 0.3])  # representative fixed charges

        def mol_at(coords):
            return _mol([Atom(z, list(c)) for z, c in zip([8, 1, 1], coords)])

        def e2(coords):
            g = np.asarray(_semi.build_gfn2_gamma(mol_at(coords), params))
            return 0.5 * dq @ g @ dq

        h = 1e-5
        fd = np.zeros((3, 3))
        for a in range(3):
            for d in range(3):
                cp = coords0.copy()
                cp[a, d] += h
                cm = coords0.copy()
                cm[a, d] -= h
                fd[a, d] = (e2(cp) - e2(cm)) / (2 * h)

        gamma = np.asarray(_semi.build_gfn2_gamma(mol_at(coords0), params))
        ana = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                if a == b:
                    continue
                dR = coords0[a] - coords0[b]
                R = np.linalg.norm(dR)
                dgamma_dR = -R * gamma[a, b] ** 3
                ana[a] += dq[a] * dq[b] * dgamma_dR / R * dR

        # The analytic per-atom term (no ½) matches FD tightly; the
        # pre-fix halved form is off by exactly a factor 2.
        assert np.abs(ana - fd).max() < 1e-7
        assert np.abs(0.5 * ana - fd).max() > 1e-3

    @requires_gfn2
    def test_full_gradient_carries_unhalved_gamma_term(self, params):
        """Implementation value pin for the full analytic gradient on water.

        History (all Ha/bohr, O-atom y-component):
          * pre-2026-07-09 halved-gamma term:                +0.3014973
          * unhalved gamma term but H^0 shape derivatives still omitted:
                                                             +0.2930460
          * full analytic gradient (H^0 shape + shell-ES + 3rd-order + AES,
            landed 2026-07-09):                              -0.0171048
          * Stewart (1970) basis + normalization fix:         +0.0366827
          * M-matrix SCC sign fix (+0.5 -> -0.5, BUG-022):   -0.0094663
          * GAM3 l-dependent scaling (gam3s=1.0, gam3p=0.5, d=0.25):  -0.011100
          * third-order convention fix (BUG-027):             -0.007093
          * M-matrix SCC sign restored (BUG-022 regression, the Hamiltonian
            really is H^0 + 1/2 S(v+v')):                    +0.0233107
          * (n,l)-correct STO-NG basis rows (O 2s uses the 2s table,
            GFN2-MOL-PARITY):                                +0.0094999

        The full gradient agrees with central finite differences to
        ~1e-9 Ha/bohr (see test_gradient_matches_fd).  This is the cheap
        value pin (no FD loop); it catches regressions in any single term."""
        mol = _mol(_water())
        opts = _xtb.XTBSccOptions()
        opts.charge_mixing = 0.05
        opts.max_iter = 2000
        result = _xtb.run_gfn2_xtb(mol, params, opts)
        assert result.converged
        # 2026-08-14: re-pinned after the (n,l)-correct STO-NG basis rows
        # (GFN2-MOL-PARITY); the SCC fixed point moved onto the xtb surface
        # (O charge -0.5566 vs xtb's -0.5610).
        assert np.asarray(result.charges)[0] == pytest.approx(-0.556578, abs=1e-4)

        grad = np.asarray(_semi.compute_gfn2_gradient(mol, result, params))
        # 2026-08-14: value re-pinned with the (n,l)-correct basis; the
        # gradient is internally consistent and the FD gate
        # (test_gradient_matches_fd) passes.
        assert grad[0, 1] == pytest.approx(0.0094999, abs=5e-4)


class TestGFN2FullGradientFD:
    """Full-gradient FD consistency gate (2026-07-09 landing).

    compute_gfn2_gradient now differentiates every term of the
    run_gfn2_xtb energy functional: the H⁰ shape terms (CN-dependent
    self-energy chain, Slater/EN factors via ∂H⁰/∂S, shell distance
    polynomial), shell-resolved 2nd-order ES, the atom-resolved 3rd-order
    term (through the SCC potential channel), the ad-hoc AES (kernel +
    multipole-integral Pulay terms), and the AES Z-vector response
    correction required because the ad-hoc AES Fock is not variational.
    Gate: central FD of the total SCC energy to < 1e-5 Ha/bohr."""

    @staticmethod
    def _tight_opts():
        opts = _xtb.XTBSccOptions()
        opts.charge_mixing = 0.05
        opts.max_iter = 20000
        opts.conv_tol_charge = 1e-11
        return opts

    def _fd_gradient(self, spec, params, h=1e-4):
        def energy(atoms):
            r = _xtb.run_gfn2_xtb(_mol(atoms), params, self._tight_opts())
            assert r.converged
            return r.energy

        coords0 = np.array([np.asarray(a.xyz, dtype=float) for a in spec])
        zs = [a.Z for a in spec]
        fd = np.zeros_like(coords0)
        for a in range(len(spec)):
            for d in range(3):
                ep = []
                for sgn in (+1, -1):
                    c = coords0.copy()
                    c[a, d] += sgn * h
                    ep.append(energy([Atom(z, list(x)) for z, x in zip(zs, c)]))
                fd[a, d] = (ep[0] - ep[1]) / (2 * h)
        return fd

    @requires_gfn2
    @pytest.mark.parametrize(
        "make_geom",
        [_water, _water_axis_aligned, _ch4, _nh3],
        ids=["h2o", "h2o_axis_aligned", "ch4", "nh3"],
    )
    def test_gradient_matches_fd(self, params, make_geom):
        spec = make_geom()
        spec = make_geom()
        result = _xtb.run_gfn2_xtb(_mol(spec), params, self._tight_opts())
        assert result.converged

        ana = np.asarray(_semi.compute_gfn2_gradient(_mol(spec), result, params))
        fd = self._fd_gradient(spec, params)
        assert np.abs(ana - fd).max() < 1e-5, (
            f"max |analytic - FD| = {np.abs(ana - fd).max():.3e}"
        )

    @requires_gfn2
    def test_gradient_translation_invariance(self, params):
        """Net force vanishes for every term of the assembled gradient."""
        spec = _water()
        opts = _xtb.XTBSccOptions()
        opts.charge_mixing = 0.05
        opts.max_iter = 2000
        result = _xtb.run_gfn2_xtb(_mol(spec), params, opts)
        assert result.converged
        grad = np.asarray(_semi.compute_gfn2_gradient(_mol(spec), result, params))
        assert np.abs(grad.sum(axis=0)).max() < 1e-8


# ---------------------------------------------------------------------------
# Reference targets (xtb GFN2). The out-of-process oracle lives at
# examples/regression/core/runner_xtb.py (CLAUDE.md §10). It carries tight
# pinned monomer energies (xtb v6.7.1), imported below when available.
#
# 2026-08-05: H2 repinned from the ASCENT external-parity task
# (asc_258dc68174ee, ORCA→xTB 6.7.1 at r=0.741400 Å; our _h2() geometry is
# R=1.4 bohr=0.740848 Å, ΔR=0.000552 Å, <0.0002 Eh).  H2O and CH4 references
# are STALE: they were generated with xtb reading bohr numbers as ångström
# (pre-fix runner_xtb.py bug) and need regeneration on a Linux host.
# ---------------------------------------------------------------------------
_XTB_REF_EH = {"H2": -0.982017, "H2O": -5.070374, "CH4": -4.18}

# Guarded import of the out-of-process xtb oracle (skips its consumers if the
# examples package is not importable in this environment).
try:
    from examples.regression.core import runner_xtb as _xtb_oracle
except Exception:  # pragma: no cover - import guard
    _xtb_oracle = None


def _coords_from_xyz(xyz: str) -> list[tuple[str, np.ndarray]]:
    lines = [line.split() for line in xyz.splitlines()[2:] if line.strip()]
    return [
        (parts[0], np.asarray([float(parts[1]), float(parts[2]), float(parts[3])]))
        for parts in lines
    ]


@requires_gfn2
def test_xtb_oracle_h2o_reference_uses_water_fixture_geometry():
    """The H2O oracle geometry must match the _water() fixture geometry
    when both are expressed in the same units.  The oracle XYZ is now in
    ångström (standard XYZ convention); the fixture is in bohr."""
    if _xtb_oracle is None:
        pytest.skip("xtb oracle runner not importable")

    oracle_atoms = _coords_from_xyz(_xtb_oracle.geometry_xyz("H2O"))
    assert [symbol for symbol, _ in oracle_atoms] == ["O", "H", "H"]

    water_atoms = _water()
    for (_, xyz_angstrom), atom in zip(oracle_atoms, water_atoms):
        xyz_bohr = np.asarray(atom.xyz, dtype=float)
        xyz_expected_angstrom = xyz_bohr * (1.0 / _ANGSTROM_TO_BOHR)
        assert xyz_angstrom == pytest.approx(xyz_expected_angstrom, abs=1e-6)

    assert _xtb_oracle.energy("H2O") == pytest.approx(
        _XTB_REF_EH["H2O"], abs=5e-7
    )


# ===================================================================
# 1. Size-consistency — the headline guarantee of the structural fix
# ===================================================================
class TestGFN2SizeConsistency:
    @requires_gfn2
    def test_two_h2_far_apart_is_additive(self, params):
        """Two H₂ 40 bohr apart must equal 2·E(H₂) to ~machine precision.

        Homonuclear → all Δq = 0 by symmetry, so the only inter-fragment
        coupling is the (now correctly decaying) repulsion. Pre-fix this was
        +0.32 Ha because the repulsion did not decay with distance.
        """
        e1 = _energy(_mol(_h2()), params).energy
        e2 = _energy(_mol(_h2(0.0) + _h2(40.0)), params).energy
        assert abs(e2 - 2 * e1) < 1e-9, f"size-inconsistency {e2 - 2 * e1:.2e}"

    @requires_gfn2
    def test_water_dimer_residual_is_the_physical_1_over_r3_tail(self, params):
        """Two neutral (but polarized) waters interact via a real dipole-dipole
        tail that decays as 1/R³ → 0. This is physical, not a size-consistency
        bug: the residual must shrink ~8× per doubling of separation and be
        < 1e-6 once far enough apart."""
        e_mono = _energy(_mol(_water()), params).energy
        r100 = _energy(_mol(_water(0.0) + _water(100.0)), params).energy - 2 * e_mono
        r200 = _energy(_mol(_water(0.0) + _water(200.0)), params).energy - 2 * e_mono
        r400 = _energy(_mol(_water(0.0) + _water(400.0)), params).energy - 2 * e_mono
        # ~1/R³ decay
        assert abs(r200) < 0.2 * abs(r100)
        assert abs(r400) < 0.2 * abs(r200)
        # size-consistent in the limit
        assert abs(r400) < 1e-6


# ===================================================================
# 2. Charge signs — physically correct polarization direction
# ===================================================================
class TestGFN2ChargeSigns:
    @requires_gfn2
    def test_water_oxygen_negative_hydrogens_positive(self, params):
        r = _energy(_mol(_water()), params)
        q = np.asarray(r.charges)  # chemical convention: cation positive
        assert q[0] < 0.0, f"O should be negative, got {q[0]:.3f}"
        assert q[1] > 0.0 and q[2] > 0.0, f"H should be positive, got {q[1:]}"
        assert abs(q.sum()) < 1e-8, "total charge must be 0 (neutral molecule)"

    @requires_gfn2
    def test_co2_carbon_positive_oxygen_negative(self, params):
        co2 = _mol([Atom(6, [0, 0, 0]), Atom(8, [2.2, 0, 0]), Atom(8, [-2.2, 0, 0])])
        r = _energy(co2, params)
        if not r.converged:
            pytest.skip("CO2 SCC did not converge (AES work-in-progress)")
        q = np.asarray(r.charges)
        assert q[0] > 0.0, f"C should be positive in CO2, got {q[0]:.3f}"
        assert q[1] < 0.0 and q[2] < 0.0, f"O should be negative, got {q[1:]}"


# ===================================================================
# 3. Repulsion decays (the size-consistency root cause)
# ===================================================================
class TestGFN2Repulsion:
    @requires_gfn2
    def test_repulsion_decays_to_zero(self, params):
        near = params.repulsive_energy(1, 1, 1.4)
        far = params.repulsive_energy(1, 1, 40.0)
        assert near > 0.0
        assert far < 1e-9, f"repulsion must vanish at 40 bohr, got {far:.3e}"

    @requires_gfn2
    def test_light_pair_repulsion_uses_published_linear_exponent(self):
        p = _xtb.GFN2ParameterSet()
        rp = _xtb.GFN2RepulsivePair()
        rp.alpha = 0.7
        rp.k_ab = 3.0
        p.set_repulsive_pair(1, 1, rp)
        p.set_repulsive_pair(2, 2, rp)
        p.set_repulsive_pair(1, 8, rp)

        r = 4.0
        light_ref = rp.k_ab / r * np.exp(-rp.alpha * r)
        heavy_ref = rp.k_ab / r * np.exp(-rp.alpha * r**1.5)
        assert p.repulsive_energy(1, 1, r) == pytest.approx(light_ref)
        assert p.repulsive_energy(2, 2, r) == pytest.approx(light_ref)
        assert p.repulsive_energy(1, 8, r) == pytest.approx(heavy_ref)

    @requires_gfn2
    def test_repulsion_monotonic_decreasing(self, params):
        rs = [params.repulsive_energy(8, 8, r) for r in (2.0, 3.0, 5.0, 8.0)]
        assert all(a > b for a, b in zip(rs, rs[1:])), rs


# ===================================================================
# 4. Determinism
# ===================================================================
class TestGFN2Determinism:
    @requires_gfn2
    def test_repeated_runs_identical(self, params):
        m = _mol(_water())
        assert _energy(m, params).energy == _energy(m, params).energy


class TestGFN2SCCStabilization:
    """Regression coverage for third-order SCC bistability."""

    @requires_gfn2
    @pytest.mark.parametrize("case_name", ["s_k_1", "pheavy_as_1"])
    def test_tae_ptcomp_h0_and_overlap_are_finite_before_first_diag(
        self, params, case_name
    ):
        """TAE-PTComp post-Ar cases must not inject NaNs into first H0.

        The pilot failures were first-diagonalization errors, not SCC-update
        blow-ups: the H0 coordination-number radius lookup returned NaN for
        elements beyond Ar and poisoned the generalized eigensolver.
        """
        mol = _tae_ptcomp_mol(case_name)
        basis = _semi.SemiempiricalBasis.build(mol, params, 6)
        S = np.asarray(_compute_overlap(basis))
        H0 = np.asarray(_semi.build_gfn2_hamiltonian_zero(basis, S, mol, params))

        assert np.isfinite(S).all()
        assert np.isfinite(H0).all()
        assert np.abs(S - S.T).max() < 1e-12
        assert np.abs(H0 - H0.T).max() < 1e-12

        s_eigs = np.linalg.eigvalsh(0.5 * (S + S.T))
        assert s_eigs[0] > 0.0
        assert s_eigs[-1] / s_eigs[0] < 2.0e3

        opts = _xtb.XTBSccOptions()
        opts.max_iter = 1
        opts.auto_stabilize = False
        result = _xtb.run_gfn2_xtb(mol, params, opts)

        assert not result.converged
        assert result.n_iter == 1
        assert result.n_basis == S.shape[0]

    @requires_gfn2
    @pytest.mark.parametrize(
        "case_name,expected_energy,expected_n_occ",
        [
            # 2026-08-13: re-pinned after the third-order convention fix
            # (shell-resolved l-scaled Gamma q_sh^3/3, xtb thirdorder.f90
            # parity; BUG-027).  Convergence properties unchanged; the SCC
            # fixed point shifted.
            # 2026-08-14: re-pinned on the merged tree (third-order fix +
            # Bannwarth Table 2 k_sd/k_pd pair values, BUG-027, plus the
            # (n,l)-correct STO-NG basis rows, GFN2-MOL-PARITY).  Cases
            # with d-polarization basis functions (3rd-row and heavier
            # elements) shift; s_li_1's Cl carries the sd/pd correction,
            # s_k_1's K d-shell barely mixes.
            # 2026-08-14 (IID 101): s_li_1 re-pinned after the default SCC
            # path became the simple+DIIS polyalgorithm; the converged
            # basin is unchanged (max |dq| vs the 1e-7-tolerance run is
            # 2.5e-7), only the stopping point inside the tolerance ball
            # moved.
            # 2026-08-27 (issue #43): s_li_1, pheavy_br_1, and
            # pheavy_rn_1 re-pinned after the d-shell STO-NG primitive
            # count moved from 4 to 3 (xtb setGFN2NumberOfPrimitives
            # parity; part of the d-first parameter-projection fix).
            # These cases carry d-polarization shells; the shifts are
            # 2-75 uHa. s_k_1 is unchanged at the 1e-10 pin.
            # 2026-08-28 (issue #433): s_li_1, s_k_1, pheavy_br_1, and
            # pheavy_rn_1 re-pinned after the Pauling EN table behind the
            # off-site EN factor (1 + 0.02*dEN^2) was extended from the
            # Z <= 10 stub (fallback 1.0 for everything heavier) to the
            # full Z <= 86 mctc-lib set (tblite parity). Heteronuclear
            # heavy pairs gained a nonzero dEN, moving the SCC fixed
            # points; the shifts are the intended physics.
            ("s_li_1", -11.894496877097069, 11),
            ("s_k_1", -1.4143706414474126, 2),
            ("pheavy_br_1", -17.948575987174884, 16),
            ("pheavy_rn_1", -17.271222057926266, 17),
        ],
    )
    def test_tae_ptcomp_repaired_default_cases_converge(
        self, params, case_name, expected_energy, expected_n_occ
    ):
        """Representative TAE-PTComp light, K, heavy-p, and period-6 cases."""
        result = _xtb.run_gfn2_xtb(_tae_ptcomp_mol(case_name), params)

        assert result.converged, f"{case_name} SCC did not converge"
        assert np.isfinite(result.energy)
        assert result.energy == pytest.approx(expected_energy, abs=1e-10)
        assert result.n_occ == expected_n_occ
        assert np.isfinite(np.asarray(result.charges)).all()

    @requires_gfn2
    def test_tae_ptcomp_as_reproducer_converges_with_extended_period_retry(
        self, params
    ):
        """The explicit high-T retry repairs the slow AsCl6- SCC basin."""
        opts = _xtb.XTBSccOptions()
        opts.auto_stabilize = False
        opts.charge_mixing = 0.05
        opts.electronic_temperature = 0.05
        opts.max_iter = 600

        result = _xtb.run_gfn2_xtb(_tae_ptcomp_mol("pheavy_as_1"), params, opts)

        assert result.converged
        assert result.n_iter < 600
        assert result.n_occ == 24
        # 2026-08-13: re-pinned after the third-order convention fix
        # (shell-resolved l-scaled Gamma q_sh^3/3, xtb thirdorder.f90
        # parity; BUG-027).  2026-08-14: re-pinned on the merged tree
        # (third-order fix + Bannwarth Table 2 k_sd/k_pd pair values,
        # BUG-027, plus the (n,l)-correct STO-NG basis rows,
        # GFN2-MOL-PARITY).  2026-08-27 (issue #43): re-pinned after the
        # d-shell primitive count moved from 4 to 3 (xtb parity); As and
        # Cl both carry d-polarization shells.
        # 2026-08-28 (issue #433): re-pinned after the full Pauling EN
        # table (Z <= 86, tblite parity) reached the As-Cl pairs.
        assert result.energy == pytest.approx(-29.823540884921673, abs=1e-10)

    @requires_gfn2
    @pytest.mark.parametrize(
        "name,z1,z2,r",
        [
            ("LiH", 3, 1, 8.0),
            ("LiF", 3, 9, 12.0),
            ("NaCl", 11, 17, 12.0),
        ],
    )
    def test_long_range_ionic_diatomics_converge_by_default(
        self, params, name, z1, z2, r
    ):
        """NaCl/LiF/LiH used to exhaust SCC iterations on the T=0 Aufbau map."""
        result = _xtb.run_gfn2_xtb(_diatomic(z1, z2, r), params)
        assert result.converged, f"{name} at R={r:g} bohr did not converge"
        assert result.energy < 0.0
        assert result.selected_attempt_index == len(result.attempts) - 1
        assert sum(attempt.n_iter for attempt in result.attempts) == result.n_iter
        assert sum(
            len(np.asarray(attempt.max_change_trace))
            for attempt in result.attempts
        ) == result.n_iter
        assert len(np.asarray(result.scc_max_change_trace)) == result.n_iter

        charges = np.asarray(result.charges)
        assert abs(charges.sum()) < 1e-8
        assert charges[0] > 0.0
        assert charges[1] < 0.0
        assert 0.1 < abs(charges[0]) < 1.2

    @requires_gfn2
    @pytest.mark.parametrize(
        "name,z1,z2,r",
        [
            ("LiH", 3, 1, 8.0),
            ("LiF", 3, 9, 12.0),
            ("NaCl", 11, 17, 12.0),
        ],
    )
    def test_public_model_long_range_ionic_diatomics_converge(
        self, params, name, z1, z2, r
    ):
        """The user-facing model wrapper must preserve the native SCC fix."""
        model = GFN2Model(_diatomic(z1, z2, r), params, warn=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GFN2D4UnsupportedWarning)
            energy = model.energy()

        assert model.converged, f"{name} at R={r:g} bohr did not converge"
        assert model.n_iter > 0
        assert np.isfinite(energy)
        assert energy < 0.0

    @requires_gfn2
    def test_explicit_finite_temperature_retry_stabilizes_lif(self, params):
        opts = _xtb.XTBSccOptions()
        opts.auto_stabilize = False
        opts.electronic_temperature = 0.005
        opts.charge_mixing = 0.05
        opts.max_iter = 3000

        result = _xtb.run_gfn2_xtb(_diatomic(3, 9, 12.0), params, opts)
        assert result.converged
        assert result.energy < 0.0
        assert np.asarray(result.charges)[0] > 0.0

    @requires_gfn2
    def test_auto_stabilizer_leaves_converged_water_path_unchanged(self, params):
        mol = _mol(_water())
        default = _xtb.XTBSccOptions()
        disabled = _xtb.XTBSccOptions()
        disabled.auto_stabilize = False

        r_default = _xtb.run_gfn2_xtb(mol, params, default)
        r_disabled = _xtb.run_gfn2_xtb(mol, params, disabled)

        assert r_default.converged and r_disabled.converged
        assert r_default.n_iter == r_disabled.n_iter
        assert r_default.energy == pytest.approx(r_disabled.energy, abs=1e-12)
        assert np.asarray(r_default.charges) == pytest.approx(
            np.asarray(r_disabled.charges), abs=1e-12
        )

    @requires_gfn2
    @pytest.mark.parametrize(
        "name,atoms,expected_energy,min_iter",
        [
            ("pyridine", _bug45_pyridine, -16.154593899415765, 100),
            # 2026-08-14 (IID 101): cyclopropene's historical 1000-iteration
            # outlier floor is obsolete -- the simple+DIIS polyalgorithm
            # default converges it in ~470 iterations; the floor is kept at
            # 100 so the case still has to be a real SCC solve, and the
            # 3000-step budget cap remains the gate.  The energy pin moved
            # 8.9e-12 Ha to the polyalgorithm's stopping point (same basin;
            # within 8.1e-11 Ha of the 1e-7-tolerance run).
            ("cyclopropene", _bug45_cyclopropene, -7.9550601197461495, 100),
        ],
    )
    def test_bug45_validation_outliers_converge_within_default_budget(
        self, params, name, atoms, expected_energy, min_iter
    ):
        """The two BUG 45 retry classes must finish without the 10k--21k ladder."""
        result = _xtb.run_gfn2_xtb(_mol(atoms()), params)

        assert result.converged, f"{name} did not converge"
        assert min_iter < result.n_iter <= 3000
        assert result.energy == pytest.approx(expected_energy, abs=1e-12)

    @requires_gfn2
    def test_bug45_public_model_uses_single_native_retry_budget(self, params):
        model = GFN2Model(_mol(_bug45_pyridine()), params, warn=False)

        energy = model.energy()

        assert model.converged
        assert 100 < model.n_iter <= 3000
        assert np.isfinite(energy)

    @requires_gfn2
    def test_native_auto_stabilizer_honors_explicit_small_budget(self, params):
        """Explicit molecular GFN2 max_iter is a total SCC budget."""
        opts = _xtb.XTBSccOptions()
        opts.max_iter = 7

        result = _xtb.run_gfn2_xtb(_mol(_bug45_pyridine()), params, opts)

        assert not result.converged
        assert result.n_iter == opts.max_iter

    @requires_gfn2
    def test_public_model_honors_explicit_small_budget(self, params):
        model = GFN2Model(_mol(_bug45_pyridine()), params, warn=False, max_iter=7)

        with pytest.raises(RuntimeError, match="did not converge after 7 iterations"):
            model.energy()

        assert not model.converged
        assert model.n_iter == 7


# ===================================================================
# 5. Experimental gate — must not return wrong numbers silently
# ===================================================================
class TestGFN2ExperimentalGate:
    @requires_gfn2
    def test_default_params_delegate_to_loader(self, monkeypatch):
        sentinel = _xtb.GFN2ParameterSet()
        calls = []

        def fake_load_gfn2_params():
            calls.append(True)
            return sentinel

        monkeypatch.setattr(
            _gfn2_params,
            "load_gfn2_params",
            fake_load_gfn2_params,
        )

        model = GFN2Model(_mol(_water()), warn=False)

        assert model.params is sentinel
        assert calls == [True]

    @requires_gfn2
    def test_invalid_scc_mixer_is_rejected(self, params):
        with pytest.raises(ValueError, match="scc_mixer"):
            GFN2Model(
                _mol(_water()),
                params,
                scc_mixer="typo",
                warn=False,
            )

    @requires_gfn2
    def test_native_newton_scc_mixer_fails_closed(self, params):
        opts = _xtb.XTBSccOptions()
        opts.scc_mixer = _semi.SCCMixer.Newton

        with pytest.raises(ValueError, match=r"Newton.*molecular GFN2"):
            _xtb.run_gfn2_xtb(_mol(_h2()), params, opts)

    @requires_gfn2
    def test_native_eyert_scc_mixer_fails_closed(self, params):
        opts = _xtb.XTBSccOptions()
        opts.scc_mixer = _semi.SCCMixer.BroydenEyert

        with pytest.raises(
            ValueError, match=r"BroydenEyert.*molecular GFN2"
        ):
            _xtb.run_gfn2_xtb(_mol(_h2()), params, opts)

    @requires_gfn2
    def test_model_emits_experimental_warning(self, params):
        with pytest.warns(GFN2ExperimentalWarning):
            GFN2Model(_mol(_water()), params).energy()

    @requires_gfn2
    def test_warn_false_silences(self, params):
        with warnings.catch_warnings():
            warnings.simplefilter("error", GFN2ExperimentalWarning)
            GFN2Model(_mol(_water()), params, warn=False).energy()  # must not raise

    @requires_gfn2
    def test_run_job_path_warns(self, tmp_path):
        from vibeqc.runner import run_job

        with pytest.warns(GFN2ExperimentalWarning):
            run_job(
                _mol(_water()),
                method="gfn2_xtb",
                output=str(tmp_path / "gfn2_run_job"),
                citations=False,
                write_molden_file=False,
                write_xyz_file=False,
                write_population_file=False,
            )

    @requires_gfn2
    def test_model_reports_real_iteration_count(self, params):
        """Regression: the runner used to hard-code n_iter=1 ('converged in 1
        iterations'). A polar molecule must take several SCC iterations."""
        m = GFN2Model(_mol(_water()), params, warn=False)
        m.energy()
        assert m.converged
        assert m.n_iter > 1

    @requires_gfn2
    def test_model_gradient_uses_one_native_scc(self, params, monkeypatch):
        """The public gradient must not finite-difference full SCC+D4 energies."""
        native_result = SimpleNamespace(
            energy=-5.0,
            n_iter=3,
            converged=True,
        )
        calls = {"scc": 0, "scc_gradient": 0, "d4_gradient": 0}

        def fake_run_gfn2_xtb(mol, passed_params, opts):
            calls["scc"] += 1
            assert passed_params is params
            assert mol.n_electrons() == 10
            return native_result

        def fake_scc_gradient(mol, result, passed_params):
            calls["scc_gradient"] += 1
            assert result is native_result
            assert passed_params is params
            return np.full((len(mol.atoms), 3), 0.25)

        def fake_d4_gradient(mol, passed_params, *, execution=None):
            calls["d4_gradient"] += 1
            assert passed_params is params
            assert execution is not None
            return np.full((len(mol.atoms), 3), -0.05)

        monkeypatch.setattr(
            _gfn2_method,
            "_xtb",
            SimpleNamespace(
                XTBSccOptions=_xtb.XTBSccOptions,
                run_gfn2_xtb=fake_run_gfn2_xtb,
            ),
        )
        monkeypatch.setattr(
            _gfn2_method,
            "_se",
            SimpleNamespace(compute_gfn2_gradient=fake_scc_gradient),
        )
        monkeypatch.setattr(
            _gfn2_method,
            "_compute_gfn2_d4_gradient_fd",
            fake_d4_gradient,
        )

        gradient = GFN2Model(_mol(_water()), params, warn=False).gradient()

        assert calls == {"scc": 1, "scc_gradient": 1, "d4_gradient": 1}
        assert gradient == pytest.approx(np.full((3, 3), 0.20))

    @requires_gfn2
    def test_model_gradient_reuses_converged_energy_scc(self, params, monkeypatch):
        """Energy plus gradient should not repeat the same native SCC solve."""
        native_result = SimpleNamespace(
            energy=-5.0,
            n_iter=3,
            converged=True,
        )
        calls = {"scc": 0, "scc_gradient": 0, "d4": 0, "d4_gradient": 0}

        def fake_run_gfn2_xtb(mol, passed_params, opts):
            calls["scc"] += 1
            assert passed_params is params
            assert mol.n_electrons() == 10
            return native_result

        def fake_scc_gradient(mol, result, passed_params):
            calls["scc_gradient"] += 1
            assert result is native_result
            assert passed_params is params
            return np.full((len(mol.atoms), 3), 0.25)

        def fake_d4(mol, passed_params, *, execution=None):
            calls["d4"] += 1
            assert passed_params is params
            assert execution is not None
            return -0.10

        def fake_d4_gradient(mol, passed_params, *, execution=None):
            calls["d4_gradient"] += 1
            assert passed_params is params
            assert execution is not None
            return np.full((len(mol.atoms), 3), -0.05)

        monkeypatch.setattr(
            _gfn2_method,
            "_xtb",
            SimpleNamespace(
                XTBSccOptions=_xtb.XTBSccOptions,
                run_gfn2_xtb=fake_run_gfn2_xtb,
            ),
        )
        monkeypatch.setattr(
            _gfn2_method,
            "_se",
            SimpleNamespace(compute_gfn2_gradient=fake_scc_gradient),
        )
        monkeypatch.setattr(_gfn2_method, "_compute_gfn2_d4", fake_d4)
        monkeypatch.setattr(
            _gfn2_method,
            "_compute_gfn2_d4_gradient_fd",
            fake_d4_gradient,
        )

        model = GFN2Model(_mol(_water()), params, warn=False)

        assert model.energy() == pytest.approx(-5.10)
        gradient = model.gradient()

        assert calls == {
            "scc": 1,
            "scc_gradient": 1,
            "d4": 1,
            "d4_gradient": 1,
        }
        assert gradient == pytest.approx(np.full((3, 3), 0.20))

    @requires_gfn2
    def test_model_gradient_matches_d4_corrected_total_energy_fd(self, params):
        """Analytic SCC plus FD-D4 differentiates the public total energy."""
        mol = _mol(_water())
        grad = GFN2Model(
            mol,
            params,
            warn=False,
            charge_mixing=0.05,
            max_iter=2000,
        ).gradient()

        h = 1.0e-4
        atoms0 = list(mol.atoms)
        fd = np.zeros((len(atoms0), 3))

        def displaced_energy(atom_idx, coord_idx, delta):
            atoms = list(atoms0)
            xyz = list(atoms[atom_idx].xyz)
            xyz[coord_idx] += delta
            atoms[atom_idx] = Atom(atoms[atom_idx].Z, xyz)
            return GFN2Model(
                Molecule(atoms, mol.charge, mol.multiplicity),
                params,
                warn=False,
                charge_mixing=0.05,
                max_iter=2000,
            ).energy()

        for atom_idx in range(len(atoms0)):
            for coord_idx in range(3):
                ep = displaced_energy(atom_idx, coord_idx, h)
                em = displaced_energy(atom_idx, coord_idx, -h)
                fd[atom_idx, coord_idx] = (ep - em) / (2.0 * h)

        assert np.abs(np.asarray(grad) - fd).max() < 2.0e-4

    @requires_gfn2
    def test_model_rejects_nonconverged_native_result_before_d4(
        self,
        params,
        monkeypatch,
    ):
        """A failed SCC fixed point is not a valid D4-corrected energy."""
        native_result = SimpleNamespace(
            energy=-123.0,
            n_iter=7,
            converged=False,
        )

        def fake_run_gfn2_xtb(mol, passed_params, opts):
            assert passed_params is params
            assert int(opts.max_iter) == 7
            assert mol.n_electrons() == 10
            return native_result

        def fail_d4(*_args, **_kwargs):
            raise AssertionError("D4 must not run on a nonconverged SCC density")

        monkeypatch.setattr(
            _gfn2_method,
            "_xtb",
            SimpleNamespace(
                XTBSccOptions=_xtb.XTBSccOptions,
                run_gfn2_xtb=fake_run_gfn2_xtb,
            ),
        )
        monkeypatch.setattr(_gfn2_method, "_compute_gfn2_d4", fail_d4)

        model = GFN2Model(_mol(_water()), params, warn=False, max_iter=7)
        with pytest.raises(
            RuntimeError,
            match="GFN2-xTB SCC did not converge after 7 iterations",
        ):
            model.energy()

        assert model.converged is False
        assert model.n_iter == 7
        assert model._last_result is native_result


# ===================================================================
# 6. Quantitative accuracy — H⁰/overlap deep-state bug FIXED (2026-06-01)
# ===================================================================
# The H⁰ diagonal normalization (H0_μμ = EN_l · S_μμ) and the correct
# Wolfsberg-Helmholtz k_ll' table (k_ss=1.85, k_sp=2.04, … from GFN2-xTB)
# fix the deep-state bug.  Isolated Ne is no longer catastrophic and H2/CH4
# are within 0.5 Ha of the xtb reference.  H2O is now also inside the 0.5 Ha
# gate (E=-4.857 vs the pinned xtb ref -5.070374, a 0.21 Ha residual): its
# total-energy test is a plain assert and PASSES.  The old loose water-*dimer*
# target is retained only as a metadata guard; the real quantitative parity
# gate is the three-point xtb oracle below (see handovers/HANDOVER_GFN2_AES.md).
class TestGFN2QuantitativeAccuracy:
    @requires_gfn2
    @pytest.mark.parametrize(
        "name,atoms,ref,tol",
        [
            ("H2", _h2, -0.982017, 0.5),
            ("CH4", _ch4, -4.18, 0.5),
        ],
    )
    def test_total_energy_matches_xtb(self, params, name, atoms, ref, tol):
        r = _energy(_mol(atoms()), params)
        if not r.converged:
            pytest.skip(f"{name} SCC did not converge (AES work-in-progress)")
        e = r.energy
        assert abs(e - ref) < tol, (
            f"{name}: E={e:.6f} Ha vs xtb ref {ref:.6f} Eh"
        )

    @requires_gfn2
    def test_h2_parity_vs_xtb_oracle(self, params):
        """H₂ GFN2 energy vs the ASCENT external-parity oracle.

        ORCA→xTB 6.7.1 at r=0.741400 Å gives −0.982038430160 Eh
        (asc_258dc68174ee).  Our _h2() is at R=1.4 bohr = 0.740848 Å;
        ΔR=0.000552 Å, <0.0002 Eh.  With aes_faithful=True the H0-based
        electronic model matches xtb to ~80 µHa (2026-08-08: kCN fix +
        H0 verification against xtb v6.7.0).
        """
        r = _energy(_mol(_h2()), params, aes_faithful=True)
        if not r.converged:
            pytest.skip("H2 SCC did not converge")
        e = r.energy
        ref = -0.982017  # live xtb GFN2 at R=1.4 bohr (2026-08-05)
        gap = abs(e - ref)
        assert gap < 0.001, (
            f"H2: E={e:.12f} Ha vs xtb ref {ref:.12f} Eh, "
            f"gap={gap:.6f} Eh (aes_faithful)"
        )

    @requires_gfn2
    def test_h2o_energy_matches_xtb(self, params):
        e = _energy(_mol(_water()), params).energy
        assert abs(e - _XTB_REF_EH["H2O"]) < 0.5, (
            f"H2O: E={e:.4f} Ha vs xtb ref {_XTB_REF_EH['H2O']:.4f} Eh"
        )

    @requires_gfn2
    def test_isolated_neon_not_catastrophic(self, params):
        """An isolated closed-shell atom (Δq=0) must equal its valence band
        energy (~-5 Ha), not collapse onto spurious deep states."""
        e = _energy(_mol([Atom(10, [0, 0, 0])]), params).energy
        assert e > -50.0, f"Ne atom E={e:.1f} Ha — spurious deep occupied states"

    @requires_gfn2
    @pytest.mark.parametrize("Z,name", [(2, "He"), (10, "Ne"), (18, "Ar")])
    def test_isolated_atom_energies_physical(self, params, Z, name):
        """Every isolated closed-shell atom must have a bound (negative),
        physically reasonable total energy (|E| < 100 Ha)."""
        e = _energy(_mol([Atom(Z, [0, 0, 0])]), params).energy
        assert e < 0.0, f"{name} atom should be bound, E={e:.2f}"
        assert abs(e) < 100.0, f"{name} atom energy suspicious: E={e:.2f}"

    @requires_gfn2
    @pytest.mark.parametrize(
        "Z,name,mult",
        [
            (26, "Fe", 5),
            (27, "Co", 4),
            (28, "Ni", 3),
            (29, "Cu", 2),
            # Group 12 (Zn, Cd, Hg) are closed-shell ns² singlets.  GFN2 freezes
            # the filled (n−1)d¹⁰ — and, for Hg, the 4f¹⁴ — in the core and loads
            # only s,p valence shells, so gfn2_valence_electrons() returns 2, not
            # the noble-gas-core count (Zn,Cd → 12; Hg → 26).  Before that fix the
            # over-count forced 12/26 e⁻ into the 4-function s+p basis → n_occ
            # saturated the basis, a spurious +4 cation, and E>0 (Zn ≈ +48.6 Ha);
            # the SCC third-order sign fix had only been masking it.
            (30, "Zn", 1),
            (48, "Cd", 1),
            (80, "Hg", 1),
        ],
    )
    def test_transition_metal_atoms_converge(self, params, Z, name, mult):
        """Transition metal atoms must converge and give bound, finite energies."""
        mol = _mol([Atom(Z, [0, 0, 0])], mult=mult)
        if mult > 1:
            r = _xtb.run_ugfn2_xtb(mol, params)
        else:
            r = _xtb.run_gfn2_xtb(mol, params)
        assert r.converged, f"{name} SCF did not converge"
        assert r.energy < 0.0, f"{name} not bound: E={r.energy:.2f}"
        assert abs(r.energy) < 100.0, f"{name} energy suspicious: E={r.energy:.2f}"

    @requires_gfn2
    @pytest.mark.parametrize(
        "Z,name,mult,exp_val",
        [
            # Group 12 (frozen (n−1)d¹⁰, and 4f¹⁴ for Hg): ns² = 2.  No d shell
            # loaded, so the over-count had nowhere to hide → closed-shell Zn was
            # unbound (+48.6 Ha) before the fix.
            (30, "Zn", 1, 2),
            (48, "Cd", 1, 2),
            (80, "Hg", 1, 2),
            # Post-transition-metal p-block, period 4 (frozen 3d¹⁰):
            # group-equivalent valence 3…8 (NOT Z−Ar = 13…18).
            (32, "Ge", 1, 4),
            (35, "Br", 2, 7),
            (36, "Kr", 1, 8),
            # period 5 (frozen 4d¹⁰):
            (50, "Sn", 1, 4),
            (53, "I", 2, 7),
            (54, "Xe", 1, 8),
            # period-6 5d transition metals (frozen 4f¹⁴): valence = group no.
            # 4…11.  Pre-fix Os/Pt came out with POSITIVE total energy.
            (74, "W", 1, 6),
            (76, "Os", 1, 8),
            (78, "Pt", 1, 10),
            # period-6 p-block (frozen 4f¹⁴ AND 5d¹⁰): valence 3…8.  These load
            # only s,p (Tl–Po) or s,p,d (At,Rn).  Pre-fix the over-count was
            # catastrophic: closed-shell Pb → +530 Ha, Po → +6662 Ha, Rn → +85 Ha.
            (82, "Pb", 1, 4),
            (84, "Po", 1, 6),
            (86, "Rn", 1, 8),
            # f-in-core trivalent lanthanides (4fⁿ entirely in the core): valence
            # is a flat 3, like La's 5d¹6s².  (Odd-Z members chosen so the neutral
            # atom's total-electron parity admits the doublet the 3 valence
            # electrons need.)
            (59, "Pr", 2, 3),
            (63, "Eu", 2, 3),
            (71, "Lu", 2, 3),
        ],
    )
    def test_heavy_element_valence_is_frozen_core_corrected(
        self, params, Z, name, mult, exp_val
    ):
        """GFN2 freezes the filled (n−1)d¹⁰ — and, throughout period 6, the 4f¹⁴
        — in the core for every element past a completed d/f series, so the
        valence count is NOT ``Z − previous noble gas``.
        ``gfn2_valence_electrons`` subtracts the frozen shells (Bannwarth, Ehlert
        & Grimme, J. Chem. Theory Comput. 2019, 15, 1652,
        doi:10.1021/acs.jctc.8b01176).

        Pre-fix the over-count forced the frozen d/f electrons into the small
        valence basis: a spurious cation and, where the over-count exceeded the
        basis capacity, a positive total energy (closed-shell Pb → +530 Ha,
        Po → +6662 Ha, Os/Pt/Rn E > 0).  Each isolated neutral atom must now
        converge to a bound, finite energy and use exactly ``exp_val`` valence
        electrons.
        """
        mol = _mol([Atom(Z, [0, 0, 0])], mult=mult)
        if mult > 1:
            r = _xtb.run_ugfn2_xtb(mol, params)
            n_used = r.n_alpha + r.n_beta
        else:
            r = _xtb.run_gfn2_xtb(mol, params)
            n_used = 2 * r.n_occ
        assert r.converged, f"{name}: SCC did not converge"
        assert r.energy < 0.0, f"{name}: not bound, E={r.energy:.3f} Ha"
        assert abs(r.energy) < 100.0, (
            f"{name}: |E|={abs(r.energy):.1f} Ha — frozen-core over-count "
            f"(the Pb +530 / Po +6662 / Os,Pt,Rn E>0 symptom)"
        )
        assert n_used == exp_val, (
            f"{name}: GFN2 valence count {n_used} != expected {exp_val} "
            f"(Z−noble-gas would give the over-counted {_z_minus_noble(Z)})"
        )
        if mult == 1:
            # A neutral closed-shell atom must come out neutral — the pre-fix
            # over-count collapsed the Mulliken charge to a large spurious cation.
            assert abs(float(r.charges[0])) < 1e-6, (
                f"{name}: net charge {float(r.charges[0]):+.3f} — spurious cation"
            )

    @requires_gfn2
    def test_symmetry_equivalent_hydrogens_equal_charge(self, params):
        """Symmetry-equivalent atoms (the two H in water) must carry equal
        Mulliken charges."""
        r = _energy(_mol(_water()), params)
        q = np.asarray(r.charges)
        assert abs(q[1] - q[2]) < 1e-8, (
            f"sym H charges differ: {q[1]:.6f} vs {q[2]:.6f}"
        )

    @requires_gfn2
    def test_water_angle_minimum_near_104_degrees(self, params):
        """H2O rigid angle scan: minimum near 104 degrees (H0 shape terms)."""
        import math

        best_e, best_ang = None, None
        for ang in range(95, 120):
            a = math.radians(ang / 2)
            r = 1.81
            z = r * math.cos(a)
            y = r * math.sin(a)
            mol = _mol(
                [
                    Atom(8, [0, 0, 0]),
                    Atom(1, [0, y, z]),
                    Atom(1, [0, -y, z]),
                ]
            )
            e = _energy(mol, params).energy
            if best_e is None or e < best_e:
                best_e, best_ang = e, ang
        assert 98 <= best_ang <= 110, (
            f"angle minimum at {best_ang} deg (expected 98-110)"
        )

    def test_legacy_water_dimer_target_is_not_parity_gate(self):
        """The retired loose H-bond target must not be used as xtb parity."""
        if _xtb_oracle is None:
            pytest.skip("xtb oracle runner not importable")

        legacy_target = -0.008
        legacy_tol = 0.01
        assert legacy_target - legacy_tol < 0.0 < legacy_target + legacy_tol

        ref = _xtb_oracle.dimer_scc_interaction_ref("trans_5.6")
        assert ref > 0.0  # translated waters are repulsive in live xtb GFN2
        assert abs(ref - legacy_target) > legacy_tol


# ===================================================================
# 6b. D4 dispersion — path active and non-zero
# ===================================================================
class TestGFN2D4Dispersion:
    """Confirm the native D4 dispersion path fires and contributes
    physically reasonable energies (attractive, non-zero)."""

    @requires_gfn2
    def test_h2o_d4_is_attractive_and_nonzero(self, params):
        """H2O monomer must have negative D4 energy ~ -1 kcal/mol."""
        from vibeqc.semiempirical.methods.gfn2 import _compute_gfn2_d4

        e_d4 = _compute_gfn2_d4(_mol(_water()), params)
        assert e_d4 < 0.0, f"D4 should be attractive, got {e_d4:.6f}"
        assert abs(e_d4) > 1e-6, f"D4 should be non-negligible, got {e_d4:.10f}"
        # ~1 kcal/mol for H2O
        assert abs(e_d4) < 0.01, f"D4 energy {e_d4:.6f} too large for H2O"

    @requires_gfn2
    def test_d4_reference_data_is_cached_across_calls(self, params, monkeypatch):
        """Repeated GFN2-D4 energies must not reload immutable JSON data."""
        original_read_bytes = Path.read_bytes
        refdata_path = _gfn2_method._gfn2_d4_refdata_path()
        load_calls = []

        def counted_read_bytes(path):
            if path == refdata_path:
                load_calls.append(path)
            return original_read_bytes(path)

        monkeypatch.setattr(Path, "read_bytes", counted_read_bytes)
        _gfn2_method._load_gfn2_d4_reference_data.cache_clear()
        try:
            first = _gfn2_method._compute_gfn2_d4(_mol(_water()), params)
            second = _gfn2_method._compute_gfn2_d4(_mol(_water()), params)
        finally:
            _gfn2_method._load_gfn2_d4_reference_data.cache_clear()

        assert first == pytest.approx(second)
        assert len(load_calls) == 1

    @requires_gfn2
    def test_d4_included_in_total_energy(self, params):
        """Total energy must include D4 (total < SCC-only)."""
        from vibeqc._vibeqc_core.semiempirical.xtb import run_gfn2_xtb
        from vibeqc.semiempirical.methods.gfn2 import GFN2Model

        mol = _mol(_water())
        e_scc = float(run_gfn2_xtb(mol, params).energy)
        e_total = GFN2Model(mol, params, warn=False).energy()
        # D4 is attractive, so total < SCC-only
        assert e_total < e_scc, (
            f"Total {e_total:.6f} should be < SCC-only {e_scc:.6f} (D4 is attractive)"
        )

    @requires_gfn2
    def test_water_dimer_d4_binding_is_attractive(self, params):
        """Water dimer D4 binding contribution should be attractive."""
        from vibeqc.semiempirical.methods.gfn2 import _compute_gfn2_d4

        mono = _mol(_water())
        dimer = _mol(_water(0.0) + _water(5.6))
        d4_mono = _compute_gfn2_d4(mono, params)
        d4_dim = _compute_gfn2_d4(dimer, params)
        d4_bind = d4_dim - 2 * d4_mono
        assert d4_bind < 0.0, f"D4 binding should be attractive, got {d4_bind:.8f} Ha"

    @requires_gfn2
    def test_unsupported_elements_warn_and_return_zero(self, params):
        """GFN2's native D4 helper must not silently omit dispersion."""
        from vibeqc.semiempirical.methods.gfn2 import _compute_gfn2_d4

        with pytest.warns(GFN2D4UnsupportedWarning, match=r"Z=\[11, 17\]"):
            e_d4 = _compute_gfn2_d4(_diatomic(11, 17, 12.0), params)
        assert e_d4 == 0.0

    @requires_gfn2
    def test_experimental_warning_fires(self, params):
        """GFN2ExperimentalWarning must emit when warn=True (default)."""
        import warnings

        from vibeqc.semiempirical.methods.gfn2 import GFN2ExperimentalWarning, GFN2Model

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            GFN2Model(_mol(_water()), params).energy()
            assert any(issubclass(x.category, GFN2ExperimentalWarning) for x in w), (
                "Experimental warning not emitted"
            )

    @requires_gfn2
    def test_warn_false_silences_warning(self, params):
        """warn=False must suppress GFN2ExperimentalWarning."""
        import warnings

        from vibeqc.semiempirical.methods.gfn2 import GFN2ExperimentalWarning, GFN2Model

        with warnings.catch_warnings():
            warnings.simplefilter("error", GFN2ExperimentalWarning)
            GFN2Model(_mol(_water()), params, warn=False).energy()
        # No exception = success


# ===================================================================
# 6b-bis. The published GFN2 energy is SCC + post-SCF D4 (issue #502)
# ===================================================================
class TestGFN2PublicEnergyIsDispersionCorrected:
    """Pin the two GFN2 energies apart so they cannot be swapped silently.

    ``run_gfn2_xtb(...).energy`` is the native SCC energy; the published
    molecular GFN2 energy — what ``GFN2Model.energy()``, ``run_job`` and
    ``run_semiempirical`` return — is that value plus the post-SCF D4
    dispersion correction (Bannwarth, Ehlert & Grimme, J. Chem. Theory
    Comput. 15, 1652-1671 (2019), doi:10.1021/acs.jctc.8b01176; D4 is an
    a-posteriori correction there, not folded into the SCC).

    xtb's own total energy INCLUDES dispersion — its output prints it as a
    sub-term of the SCC energy (``-> dispersion``) — so an xtb parity
    comparison is only like-for-like against the D4-corrected value.

    Issue #502 was filed as a 13.28 mHa GFN2 regression on adenine
    "between main f3c4accd0 and 941caf57a". It was neither. Rebuilt at both
    SHAs against one pinned parameter cache
    (``~/.cache/vibeqc/gfn2_xtb_params.toml`` sha256 ``6be784c5...``), the
    archived THIEL adenine frame gives bit-identical numbers at both:

        e_scc  = -27.508108468392056 Ha   (native SCC, no dispersion)
        e_d4   =  -0.013278137295    Ha   (post-SCF D4)
        total  = -27.521386605687525 Ha   (published GFN2 energy)

    The two recorded "before/after" figures were e_scc and total, measured
    on the same code. For scale, xtb 6.7.0 --gfn 2 --acc 1.0 on that frame
    reports ``-> dispersion -0.013537661933 Eh`` inside its own SCC energy
    and a total of -27.499575434707 Eh, so the like-for-like residual is
    -21.811 mHa, not the -8.5 mHa that a dispersion-free comparison gives.
    A 13 mHa confusion is 133x issue #43's 0.1 mHa parity gate.
    """

    # Live xtb 6.7.0 GFN2 total energy for _h2() at R = 1.4 bohr, the same
    # reference _XTB_REF_EH["H2"] carries. It is a TOTAL, i.e. dispersion
    # is already in it, which is what makes the comparison below directional.
    _H2_XTB_TOTAL_EH = -0.982017

    @requires_gfn2
    @pytest.mark.parametrize("name,atoms", [("CH4", _ch4()), ("H2O", _water())])
    def test_public_energy_is_exactly_scc_plus_d4(self, params, name, atoms):
        """Every public entry point returns native SCC + post-SCF D4."""
        from vibeqc.semiempirical import run_semiempirical
        from vibeqc.semiempirical.methods.gfn2 import _compute_gfn2_d4

        mol = _mol(atoms)
        e_scc = float(_energy(mol, params).energy)
        e_d4 = float(_compute_gfn2_d4(mol, params))

        model_total = float(GFN2Model(mol, params, warn=False).energy())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", GFN2ExperimentalWarning)
            runner_total = float(run_semiempirical("gfn2_xtb", mol).energy)

        assert model_total == e_scc + e_d4, (
            f"{name}: GFN2Model.energy() {model_total!r} is not "
            f"SCC {e_scc!r} + D4 {e_d4!r}"
        )
        assert runner_total == model_total, (
            f"{name}: run_semiempirical {runner_total!r} disagrees with "
            f"GFN2Model.energy() {model_total!r}"
        )

    @requires_gfn2
    @pytest.mark.parametrize("name,atoms", [("CH4", _ch4()), ("H2O", _water())])
    def test_scc_and_total_are_not_interchangeable_at_the_parity_gate(
        self, params, name, atoms
    ):
        """The D4 term is larger than issue #43's 0.1 mHa parity gate.

        If it were not, quoting the SCC energy where the total belongs would
        be harmless. It is not: even on these small molecules the gap clears
        the gate, and it grows with system size (13.28 mHa on adenine).
        """
        from vibeqc.semiempirical.methods.gfn2 import _compute_gfn2_d4

        mol = _mol(atoms)
        e_d4 = float(_compute_gfn2_d4(mol, params))
        gate_ha = 1.0e-4  # issue #43's stated 0.1 mHa total-energy tolerance
        assert abs(e_d4) > gate_ha, (
            f"{name}: D4 {e_d4:.9f} Ha no longer clears the {gate_ha} Ha "
            "parity gate; this guard can no longer discriminate"
        )

    @requires_gfn2
    def test_dropping_d4_moves_h2_away_from_the_xtb_oracle(self, params):
        """Against an xtb TOTAL, the D4-corrected value is the closer one.

        Measured on main 9cca63c16 for _h2() with aes_faithful=True:
        the D4-corrected total is 0.96 uHa from the live xtb 6.7.0 total,
        the SCC-only value 50.47 uHa — a factor of ~52. Comparing SCC-only
        against an xtb total therefore reports a residual that is almost
        entirely the missing dispersion term, which is the mis-read that
        produced issue #502 and the -8.5 mHa adenine figure of record.
        """
        from vibeqc.semiempirical.methods.gfn2 import _compute_gfn2_d4

        mol = _mol(_h2())
        result = _energy(mol, params, aes_faithful=True)
        if not result.converged:
            pytest.skip("H2 SCC did not converge")
        e_scc = float(result.energy)
        e_total = e_scc + float(_compute_gfn2_d4(mol, params))

        gap_total = abs(e_total - self._H2_XTB_TOTAL_EH)
        gap_scc = abs(e_scc - self._H2_XTB_TOTAL_EH)

        assert gap_total < 5.0e-6, (
            f"H2 D4-corrected total is {gap_total * 1e6:.2f} uHa from the "
            f"xtb total {self._H2_XTB_TOTAL_EH}"
        )
        assert gap_scc > 4.0e-5, (
            f"H2 SCC-only is only {gap_scc * 1e6:.2f} uHa from the xtb total; "
            "the two quantities are no longer distinguishable here"
        )
        assert gap_scc > 10.0 * gap_total, (
            f"dropping D4 must measurably worsen the xtb residual: "
            f"SCC-only {gap_scc * 1e6:.2f} uHa vs total {gap_total * 1e6:.2f} uHa"
        )


# ===================================================================
# 6c. PES validation — water O-H stretch has proper minimum
# ===================================================================
class TestGFN2PESGate:
    """Verify the potential energy surface has a physically reasonable
    minimum near the experimental O-H bond length (1.81 bohr = 0.957 A).
    The H0 shape terms (CN self-energy + shell polynomial + EN factor,
    2026-06) fix the equilibrium angle to ~104 deg.
    """

    @requires_gfn2
    def test_water_pes_has_minimum_near_equilibrium(self, params):
        """Water O-H stretch must have minimum within 1.70-1.95 bohr."""
        import math

        from vibeqc import Atom, Molecule
        from vibeqc.semiempirical.methods.gfn2 import GFN2Model

        def water(r):
            a = math.radians(52.25)
            z = r * math.cos(a)
            y = r * math.sin(a)
            return Molecule(
                [Atom(8, [0, 0, 0]), Atom(1, [0, y, z]), Atom(1, [0, -y, z])],
                0,
                1,
            )

        energies = []
        for r in [1.50, 1.55, 1.60, 1.65, 1.70, 1.75, 1.81, 1.90, 2.00, 2.10, 2.20]:
            e = GFN2Model(water(r), params, warn=False).energy()
            energies.append((r, e))

        # Find minimum
        min_r, min_e = min(energies, key=lambda x: x[1])
        assert 1.70 <= min_r <= 1.95, (
            f"Water PES minimum at r={min_r:.2f} bohr, expected 1.70-1.95 (expt ~1.81)"
        )

    @requires_gfn2
    def test_water_pes_has_well_in_both_directions(self, params):
        """Water PES must have a proper well: energy up at both shorter
        and longer R from the minimum.  A collapse shows as E↓ when
        shortening from 1.81→1.50 (no left wall)."""
        import math

        from vibeqc import Atom, Molecule
        from vibeqc.semiempirical.methods.gfn2 import GFN2Model

        def water(r):
            a = math.radians(52.25)
            z = r * math.cos(a)
            y = r * math.sin(a)
            return Molecule(
                [Atom(8, [0, 0, 0]), Atom(1, [0, y, z]), Atom(1, [0, -y, z])],
                0,
                1,
            )

        e_short = GFN2Model(water(1.50), params, warn=False).energy()
        e_eq = GFN2Model(water(1.81), params, warn=False).energy()
        e_long = GFN2Model(water(2.00), params, warn=False).energy()

        # Both shorter and longer must be higher (less negative) than eq
        left_ok = e_short > e_eq
        right_ok = e_long > e_eq
        assert left_ok and right_ok, (
            f"PES must have a well: E(1.50)={e_short:.4f} "
            f"E(1.81)={e_eq:.4f} E(2.00)={e_long:.4f} "
            f"(left_wall={'OK' if left_ok else 'COLLAPSED'}, "
            f"right_wall={'OK' if right_ok else 'MISSING'})"
        )

    @requires_gfn2
    def test_h2o_geometry_optimization_converges_within_budget(self, params):
        """GFN2-GEOMOPT (IID 76): ASE BFGS on the paper-core H2O row must
        converge to fmax = 0.01 eV/A within 500 steps.  At v0.15.130 the
        analytic gradient carried the BUG-022 -1/2 S(v+v') sign, so BFGS
        stalled (final displacement ~7e-11 A at 0.26 eV/A); the gradient
        restore (5f4f8d9fa, IID 68) fixed the surface the optimizer walks."""
        pytest.importorskip("ase")
        from ase import Atoms
        from ase.optimize import BFGSLineSearch
        from ase.units import Bohr

        from vibeqc.runner import _make_semiempirical_ase_calculator

        mol = Molecule(
            [
                Atom(8, [0.0, 0.0, 0.0]),
                Atom(1, [1.6, 0.3, 0.0]),
                Atom(1, [-1.4, 0.5, 0.2]),
            ]
        )
        atoms = Atoms(
            numbers=[a.Z for a in mol.atoms],
            positions=[[c * Bohr for c in a.xyz] for a in mol.atoms],
        )
        atoms.calc = _make_semiempirical_ase_calculator(mol, "gfn2_xtb")

        opt = BFGSLineSearch(atoms, logfile=None)
        converged = bool(opt.run(fmax=0.01, steps=500))
        assert converged, f"BFGS stalled: nsteps={opt.nsteps}"

        forces = np.asarray(atoms.get_forces())
        assert float(np.max(np.linalg.norm(forces, axis=1))) < 0.01
        # The optimized O-H bonds must sit in the physical range
        # (positions are in Angstrom; 0.85-1.11 A = 1.61-2.10 bohr).
        oh = [float(np.linalg.norm(atoms.positions[i] - atoms.positions[0]))
              for i in (1, 2)]
        assert all(0.85 < r < 1.11 for r in oh), f"O-H bonds {oh} unphysical"


class TestGFN2MultiMolecule:
    """GFN2-xTB converges for diverse molecules with shape terms active."""

    @requires_gfn2
    def test_nh3_converges_physical_charges(self, params):
        """NH3: converge with |q| < 0.3 (Gamma_N <= 0, CN shift active)."""
        import numpy as np

        mol = Molecule(
            [
                Atom(7, [0, 0, 0]),
                Atom(1, [0.94, 0, 1.78]),
                Atom(1, [0.94, 1.54, -0.89]),
                Atom(1, [-1.88, 0, 0]),
            ],
            0,
            1,
        )
        r = _xtb.run_gfn2_xtb(mol, params)
        assert r.converged
        q = np.asarray(r.charges)
        assert np.max(np.abs(q)) < 0.8, (
            f"NH3 charges too polarized: max|q|={np.max(np.abs(q)):.3f}"
        )
        assert r.energy < -1.2

    @requires_gfn2
    def test_ch3oh_converges_physical_charges(self, params):
        """CH3OH: converge with bounded charges on the ungated GFN2 H0 path."""
        import numpy as np

        mol = Molecule(
            [
                Atom(6, [0, 0, 0]),
                Atom(8, [0, 0, 2.7]),
                Atom(1, [0, 0, 3.5]),
                Atom(1, [1.0, 1.0, -0.5]),
                Atom(1, [1.0, -1.0, -0.5]),
                Atom(1, [-1.4, 0, -0.5]),
            ],
            0,
            1,
        )
        r = _xtb.run_gfn2_xtb(mol, params)
        assert r.converged
        q = np.asarray(r.charges)
        assert np.max(np.abs(q)) < 0.75, (
            f"CH3OH charges too polarized: max|q|={np.max(np.abs(q)):.3f}"
        )
        assert r.energy < -4.0


# ===================================================================
# 7. Periodic GFN2-xTB -- Gamma-point smoke test
# ===================================================================
class TestGFN2Periodic:

    @staticmethod
    def _fcc_cu_supercell(nx, ny, nz, *, rows=False):
        """fcc Cu (a = 3.7958 A) nx x ny x nz primitive supercell, 16 atoms
        at 2x2x4.  ``rows=True`` feeds the lattice vectors as ROWS, the
        transposed input of issue #408's report; the contract is COLUMNS."""
        from vibeqc._vibeqc_core import Atom, PeriodicSystem

        a = 3.7958 * 1.8897261254535
        primitive = np.array([[0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0]]) * a
        vectors = np.array(
            [nx * primitive[0], ny * primitive[1], nz * primitive[2]]
        )
        lattice = vectors if rows else vectors.T
        atoms = [
            Atom(
                29,
                (i * primitive[0] + j * primitive[1] + k * primitive[2]).tolist(),
            )
            for i in range(nx)
            for j in range(ny)
            for k in range(nz)
        ]
        return PeriodicSystem(3, lattice, atoms, 0, 1)

    @requires_gfn2
    def test_anisotropic_supercell_assembles_and_row_fed_lattice_names_the_coincidence(
        self, params
    ):
        """Issue #408: the 2x2x4 refusal was a transposed lattice, not the driver.

        With the lattice as columns no two of the 16 atoms are lattice
        equivalent and the Gamma driver assembles at every cutoff; the
        original report fed rows, under which atoms k and k+2 along the
        four-fold axis coincide modulo a(1,1,0), tripping the H0 coordination
        pair-distance guard -- and 2x2x2 has no such pair, exactly as filed.
        The guard now names the coincident pair and the image instead of a
        bare 'must be finite and positive'.
        """
        from vibeqc._vibeqc_core.semiempirical.xtb import XTBSccOptions

        options = XTBSccOptions()
        options.max_iter = 1
        options.auto_stabilize = False
        for cutoff in (12.0, 20.0):
            # max_iter=1 is an assembly probe: the record is a non-converged
            # diagnostics result, which is the point -- it was assembled.
            result = _xtb.run_gfn2_xtb_gamma(
                self._fcc_cu_supercell(2, 2, 4), params, options, cutoff
            )
            assert not result.converged

        with pytest.raises(ValueError, match="coincide modulo the lattice"):
            _xtb.run_gfn2_xtb_gamma(
                self._fcc_cu_supercell(2, 2, 4, rows=True), params, options, 12.0
            )
        # The transposed 2x2x2 cell has no lattice-equivalent pair and still
        # assembles -- the size dependence that made the report look like an
        # enumeration defect.
        result = _xtb.run_gfn2_xtb_gamma(
            self._fcc_cu_supercell(2, 2, 2, rows=True), params, options, 12.0
        )
        assert not result.converged
    @requires_gfn2
    def test_periodic_h0_coordination_uses_physical_atom_images(self):
        """Eq. 18 CN is invariant to site relabelling and atom order."""
        reference = np.asarray(
            _semi._gfn2_h0_periodic_coordination_numbers(
                _mgo_periodic_system()
            )
        )
        translated = np.asarray(
            _semi._gfn2_h0_periodic_coordination_numbers(
                _mgo_periodic_system(oxygen_image=1)
            )
        )
        permuted = np.asarray(
            _semi._gfn2_h0_periodic_coordination_numbers(
                _mgo_periodic_system(oxygen_image=1, permuted=True)
            )
        )

        assert np.isfinite(reference).all()
        assert (reference > 0.0).all()
        np.testing.assert_allclose(
            translated, reference, rtol=0.0, atol=1.0e-13
        )
        np.testing.assert_allclose(
            permuted[::-1], reference, rtol=0.0, atol=1.0e-13
        )

    @requires_gfn2
    def test_periodic_gamma_energy_follows_lattice_site_relabelling(
        self, params
    ):
        """A full lattice translation may only relabel an atom's images.

        2026-08-28 (issue #433): the full Pauling EN table moved the MgO
        SCC basin, and the exact-zero-temperature Aufbau configuration this
        test used to pin no longer converges at any iteration budget -- the
        Gamma frontier flaps between SCC branches, the failure mode the
        driver's default frontier smearing exists to prevent. The test now
        runs the default smearing path (the shipped configuration) with a
        1000-iteration budget (the solve needs ~780 at tol 1e-8); the
        relabelling invariant it exists to pin is unchanged.

        2026-09-06 (issues #296/#338): the driver no longer canonicalises the
        input representative; the lattice-summed gamma, the image-resolved
        AES and the pair-cutoff H0/S are covariant by construction, so the
        invariance holds with every term (and the joint-state Eyert default
        converges in ~90 iterations).
        """
        options = _xtb.XTBSccOptions()
        options.conv_tol_charge = 1.0e-11
        options.max_iter = 1000

        reference_system = _mgo_periodic_system(oxygen_image=0)
        translated_system = _mgo_periodic_system(oxygen_image=1)
        translated_magnesium_system = _mgo_periodic_system(
            magnesium_image=-1
        )
        reference_oxygen = np.asarray(reference_system.unit_cell[1].xyz)
        translated_oxygen = np.asarray(translated_system.unit_cell[1].xyz)
        np.testing.assert_allclose(
            translated_oxygen - np.asarray(reference_system.lattice)[:, 0],
            reference_oxygen,
            rtol=0.0,
            atol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(translated_magnesium_system.unit_cell[0].xyz)
            + np.asarray(reference_system.lattice)[:, 1],
            np.asarray(reference_system.unit_cell[0].xyz),
            rtol=0.0,
            atol=0.0,
        )

        results = [
            _xtb.run_gfn2_xtb_gamma(
                periodic_system,
                params,
                options,
                cutoff_bohr=12.0,
            )
            for periodic_system in (
                reference_system,
                translated_system,
                translated_magnesium_system,
            )
        ]
        assert all(result.converged for result in results)

        for result in results:
            assert result.energy == result.e_electronic + result.e_repulsive
            assert result.e_electronic == (
                result.e_band0
                + result.e_scc
                + result.e_aes
                + result.e_3rd
            )
            assert result.free_energy == (
                result.energy
                - result.smearing_temperature * result.entropy
            )

        # Internal source regression only: this is not an external xtb parity
        # claim.  Pinning the decomposition makes a future translation-neutral
        # redistribution between terms visible as well as a total-energy move.
        # Re-pinned 2026-08-28 (issue #433): default-smearing fixed point of
        # the corrected EN table (was the exact-T=0 Aufbau state).
        # Re-pinned 2026-09-06 (issues #296/#338): the lattice-summed
        # Ewald-split Elstner shell gamma replaced the home-cell gamma and
        # the Bannwarth 2019 AES with image-resolved moments replaced the
        # ad-hoc shell-resolved kernel; e_aes is exactly zero on the cubic
        # site by symmetry.  Joint charge+moment Eyert Broyden converges the
        # cubic-symmetric state in 97 iterations at conv_tol_charge 1e-11.
        #
        # The tolerance is 1e-11 rather than the driver default because the
        # per-term decomposition is what is pinned here.  A converged total
        # energy is stationary and so quadratic in the charge residual, but
        # the individual band/SCC/third-order terms are linear in it: at the
        # 1e-8 this test used before, two representatives of the same crystal
        # agree to 1.3e-8 term by term while their totals already agree to
        # 5e-12.  Pinning terms at 1e-10 therefore requires resolving the
        # fixed point itself, not merely the energy.
        np.testing.assert_allclose(
            _periodic_gfn2_components(results[0])[:8],
            [
                -4.374421856512581,    # energy
                -4.377205019210195,    # free_energy
                -4.378263504645535,    # e_electronic
                0.0038416481329539703, # e_repulsive
                0.059423780460156894,  # e_scc
                -4.517696463772153,    # e_band0
                0.0,                   # e_aes (cubic symmetry)
                0.08000917866646179,   # e_3rd
            ],
            rtol=0.0,
            atol=2.0e-12,
        )

        # The padded cell list differs between representatives; the
        # interaction set, and hence every energy term, is what is invariant.
        for result in results[1:]:
            np.testing.assert_allclose(
                _periodic_gfn2_components(result),
                _periodic_gfn2_components(results[0]),
                rtol=0.0,
                atol=1.0e-10,
            )

    @requires_gfn2
    def test_periodic_gamma_default_cutoff_translation_components(
        self, params
    ):
        """The independently failing raw default-cutoff route is covariant.

        Everything here is the shipped default: default cutoff, default
        smearing and the default conv_tol_charge of 1e-6.  The covariance
        assertion is therefore made at the accuracy that solve delivers --
        the stationary total energy to 1e-8, the per-term decomposition to
        1e-5, since the terms are linear in the charge residual and the
        total is quadratic in it.  The tight per-term statement is made by
        test_periodic_gamma_energy_follows_lattice_site_relabelling, which
        resolves the fixed point to 1e-11 first.
        """
        results = [
            _xtb.run_gfn2_xtb_gamma(system, params)
            for system in (
                _mgo_periodic_system(oxygen_image=0),
                _mgo_periodic_system(oxygen_image=1),
            )
        ]
        assert all(result.converged for result in results)
        np.testing.assert_allclose(
            _periodic_gfn2_components(results[1])[:4],
            _periodic_gfn2_components(results[0])[:4],
            rtol=0.0,
            atol=1.0e-8,
        )
        np.testing.assert_allclose(
            _periodic_gfn2_components(results[1]),
            _periodic_gfn2_components(results[0]),
            rtol=0.0,
            atol=1.0e-5,
        )

    @requires_gfn2
    def test_periodic_energy_closure_matches_raw_gamma_free_energy(self, params):
        """The optimizer closure returns the same default-smeared surface."""
        from vibeqc.semiempirical.periodic import make_periodic_energy_function

        systems = [
            _mgo_periodic_system(oxygen_image=image) for image in (0, 1)
        ]
        raw_results = [
            _xtb.run_gfn2_xtb_gamma(
                system, params, cutoff_bohr=12.0
            )
            for system in systems
        ]
        assert all(result.converged for result in raw_results)

        energy_fn = make_periodic_energy_function(
            "gfn2_xtb", systems[0], cutoff_bohr=12.0
        )
        closure_energies = [energy_fn(system) for system in systems]
        assert closure_energies == [
            result.free_energy for result in raw_results
        ]
        np.testing.assert_allclose(
            closure_energies[1],
            closure_energies[0],
            rtol=0.0,
            atol=1.0e-10,
        )

    @requires_gfn2
    def test_molecular_energy_rigid_translation_control(self, params):
        """The free-boundary GFN2 control keeps its Cartesian gauge."""
        shift = np.array([37.0, -19.0, 11.0])
        reference_atoms = _water()
        translated_atoms = [
            Atom(atom.Z, (np.asarray(atom.xyz) + shift).tolist())
            for atom in reference_atoms
        ]
        reference = _xtb.run_gfn2_xtb(_mol(reference_atoms), params)
        translated = _xtb.run_gfn2_xtb(_mol(translated_atoms), params)
        assert reference.converged
        assert translated.converged
        np.testing.assert_allclose(
            [
                translated.energy,
                translated.free_energy,
                translated.e_electronic,
                translated.e_repulsive,
                translated.e_scc,
                translated.e_band0,
                translated.e_aes,
                translated.e_3rd,
            ],
            [
                reference.energy,
                reference.free_energy,
                reference.e_electronic,
                reference.e_repulsive,
                reference.e_scc,
                reference.e_band0,
                reference.e_aes,
                reference.e_3rd,
            ],
            rtol=0.0,
            atol=1.0e-10,
        )

    @requires_gfn2
    def test_nonconverged_periodic_result_has_no_numeric_energy(self, params):
        """A flag-ignoring consumer must not read a plausible zero energy."""
        from vibeqc._vibeqc_core.semiempirical.xtb import (
            XTBSccOptions,
            run_gfn2_xtb_gamma,
        )

        options = XTBSccOptions()
        options.max_iter = 1
        options.auto_stabilize = False
        result = run_gfn2_xtb_gamma(
            _mgo_periodic_system(),
            params,
            options,
            cutoff_bohr=12.0,
        )
        assert not result.converged
        assert result.n_iter == options.max_iter
        assert np.isnan(result.energy)
        assert np.isnan(result.free_energy)

    @requires_gfn2
    def test_periodic_result_exposes_gamma_assembly_diagnostics(self, params):
        """The Gamma-point assembly diagnostics (overlap_gamma,
        hamiltonian_gamma, mo_energies, charges, dq_shell, dimensions)
        are readable from Python on a converged result, for the
        SCC-map analysis probes (IID 130/300). The driver fills them on
        the converged exit only; a failed exit leaves them empty."""
        from vibeqc._vibeqc_core.semiempirical.xtb import (
            XTBSccOptions,
            run_gfn2_xtb_gamma,
        )

        options = XTBSccOptions()
        # 2026-08-28 (issue #433): the full Pauling EN table moved the MgO
        # SCC basin; the solve now takes ~530 iterations, so the 500-step
        # budget was no longer enough. max_iter is the total public budget
        # (500 also suppresses the >= 2500 neutral-restart reserve), and
        # 531 iterations fit well inside 1000.
        options.max_iter = 1000
        result = run_gfn2_xtb_gamma(
            _mgo_periodic_system(),
            params,
            options,
            cutoff_bohr=12.0,
        )
        assert result.converged
        n_basis = int(result.n_basis)
        assert n_basis > 0
        S = np.asarray(result.overlap_gamma)
        H = np.asarray(result.hamiltonian_gamma)
        assert S.shape == (n_basis, n_basis)
        assert H.shape == (n_basis, n_basis)
        np.testing.assert_allclose(S, S.T, atol=1.0e-12)
        assert np.asarray(result.mo_energies).size == n_basis
        assert np.asarray(result.charges).size == len(
            _mgo_periodic_system().unit_cell
        )
        assert np.asarray(result.dq_shell).size == int(result.n_shells)
        assert int(result.n_occ) > 0

    @requires_gfn2
    def test_periodic_image_h0_uses_gfn2_shape_terms(self, params):
        """Periodic GFN2 image-cell H0 blocks must use the same shell
        distance-polynomial shape terms as molecular two-centre H0 couplings."""
        import vibeqc._vibeqc_core as core

        system = core.PeriodicSystem()
        system.lattice = np.diag([3.0, 20.0, 20.0])
        system.unit_cell = [Atom(8, [0.0, 0.0, 0.0])]
        mol = system.unit_cell_molecule()
        basis = _semi.SemiempiricalBasis.build(mol, params, 6)

        opts = core.LatticeSumOptions()
        opts.cutoff_bohr = 4.0
        overlap = core.compute_overlap_lattice(basis, system, opts)

        home_idx = next(
            i for i, cell in enumerate(overlap.cells)
            if np.all(np.asarray(cell.index) == 0)
        )
        image_idx = next(
            i for i, cell in enumerate(overlap.cells)
            if np.array_equal(np.asarray(cell.index), np.array([1, 0, 0]))
        )

        S_home = np.asarray(overlap.blocks[home_idx])
        S_image = np.asarray(overlap.blocks[image_idx])
        shift = np.asarray(overlap.cells[image_idx].r_cart)
        H_home = np.asarray(
            _semi.build_gfn2_hamiltonian_zero(basis, S_home, mol, params)
        )
        H_image = np.asarray(
            _semi.build_gfn2_hamiltonian_zero_image(
                basis, S_image, mol, params, shift
            )
        )

        shell = next(sh for sh in _semi.gfn2_enumerate_shells(basis, mol, params) if sh.l == 1)
        mu = shell.bf_start  # p_x, whose image overlap is nonzero along x
        h_diag = H_home[mu, mu] / S_home[mu, mu]
        reduced_image_h0 = 2.23 * S_image[mu, mu] * h_diag
        # O CRC atomic radius (gfn2_h0_atomic_rad, Z=8): 1.209425 bohr.
        # p_rad's 1.8 was retired by e12a04aef (CRC radii replaced p_rad).
        rr = np.sqrt(np.linalg.norm(shift) / (2.0 * 1.209425))
        expected = reduced_image_h0 * (1.0 + shell.poly * rr) ** 2

        assert abs(expected - reduced_image_h0) > 1e-7
        assert H_image[mu, mu] == pytest.approx(expected, rel=1e-12, abs=1e-12)

    @requires_gfn2
    def test_h0_intercentre_uses_slater_compactness_factor(self):
        """The published H0 coupling reduces unlike-zeta AO pairs."""
        import vibeqc._vibeqc_core as core

        p = _xtb.GFN2ParameterSet()
        h = _xtb.GFN2ElementData()
        h.Z = 1
        h.add_shell(0, -0.5, 1.0, 1.0, 0.0, 0.10)
        li = _xtb.GFN2ElementData()
        li.Z = 3
        li.add_shell(0, -0.3, 4.0, 1.0, 0.0, -0.05)
        p.add_element(h)
        p.add_element(li)

        mol = _mol([Atom(1, [0.0, 0.0, 0.0]), Atom(3, [2.0, 0.0, 0.0])])
        basis = _semi.SemiempiricalBasis.build(mol, p, 6)
        S = core.compute_overlap(basis)
        H = np.asarray(_semi.build_gfn2_hamiltonian_zero(basis, S, mol, p))

        shells = _semi.gfn2_enumerate_shells(basis, mol, p)
        h_shell = next(sh for sh in shells if sh.atom_idx == 0 and sh.l == 0)
        li_shell = next(sh for sh in shells if sh.atom_idx == 1 and sh.l == 0)
        mu = h_shell.bf_start
        nu = li_shell.bf_start
        zeta_factor = np.sqrt(2.0 * np.sqrt(1.0 * 4.0) / (1.0 + 4.0))
        en_factor = 1.0 + 0.02 * (2.20 - 0.98) ** 2
        # H, Li CRC atomic radii (gfn2_h0_atomic_rad, Z=1,3): 0.604712,
        # 2.456644 bohr. p_rad's 1.4/5.0 was retired by e12a04aef.
        rr = np.sqrt(2.0 / (0.604712 + 2.456644))
        pi_factor = (1.0 + 0.10 * rr) * (1.0 - 0.05 * rr)
        h_diag_mu = H[mu, mu] / S[mu, mu]
        h_diag_nu = H[nu, nu] / S[nu, nu]
        expected = 0.5 * 1.85 * S[mu, nu] * (h_diag_mu + h_diag_nu)
        expected *= zeta_factor * en_factor * pi_factor

        assert zeta_factor < 1.0
        assert H[mu, nu] == pytest.approx(expected, rel=1e-12, abs=1e-12)

    @requires_gfn2
    def test_h0_na_cl_pair_uses_the_full_pauling_electronegativity_table(self):
        """Issue #433: the Pauling EN table behind the off-site EN factor
        (1 + 0.02 * dEN^2) stopped at Z = 10 and returned 1.0 for every
        heavier element, so a NaCl-class pair computed dEN = 0 and a flat
        en_factor instead of the tblite value.  Pin a synthetic Na-Cl pair
        against the tblite formula with the full mctc-lib values
        (Na 0.93, Cl 3.16), which the energy and derivative paths share."""
        p = _xtb.GFN2ParameterSet()
        na = _xtb.GFN2ElementData()
        na.Z = 11
        na.add_shell(0, -0.10, 1.0, 1.0, 0.0, 0.10)
        cl = _xtb.GFN2ElementData()
        cl.Z = 17
        cl.add_shell(0, -0.30, 1.0, 1.0, 0.0, -0.05)
        p.add_element(na)
        p.add_element(cl)

        mol = _mol([Atom(11, [0.0, 0.0, 0.0]), Atom(17, [2.0, 0.0, 0.0])])
        basis = _semi.SemiempiricalBasis.build(mol, p, 6)
        S = _compute_overlap(basis)
        H = np.asarray(_semi.build_gfn2_hamiltonian_zero(basis, S, mol, p))

        shells = _semi.gfn2_enumerate_shells(basis, mol, p)
        na_shell = next(sh for sh in shells if sh.atom_idx == 0 and sh.l == 0)
        cl_shell = next(sh for sh in shells if sh.atom_idx == 1 and sh.l == 0)
        mu = na_shell.bf_start
        nu = cl_shell.bf_start
        zeta_factor = np.sqrt(2.0 * np.sqrt(1.0 * 1.0) / (1.0 + 1.0))
        en_factor = 1.0 + 0.02 * (0.93 - 3.16) ** 2
        # Na, Cl CRC atomic radii (gfn2_h0_atomic_rad, Z=11,17): 3.023561,
        # 1.889726 bohr.
        rr = np.sqrt(2.0 / (3.023561 + 1.889726))
        pi_factor = (1.0 + 0.10 * rr) * (1.0 - 0.05 * rr)
        h_diag_mu = H[mu, mu] / S[mu, mu]
        h_diag_nu = H[nu, nu] / S[nu, nu]
        expected = 0.5 * 1.85 * S[mu, nu] * (h_diag_mu + h_diag_nu)
        expected *= zeta_factor * en_factor * pi_factor

        # The truncated table returned 1.0 for both, so dEN was exactly 0;
        # the full table must move the factor away from 1.
        assert en_factor != pytest.approx(1.0)
        assert H[mu, nu] == pytest.approx(expected, rel=1e-12, abs=1e-12)


# ===================================================================
# Parameter provenance — BUG 69 auditability gate
# ===================================================================
class TestGFN2ParameterProvenance:
    """The _gfn2_parameter_provenance helper must return a non-empty
    flat scalar dict with the expected keys, and the SemiempiricalResult
    from the GFN2 path must carry it so run_job can feed it into the
    output manifest."""

    _REQUIRED_KEYS = frozenset(
        {
            "gfn2_param_version",
            "gfn2_param_origin",
            "gfn2_param_license",
            "gfn2_param_doi",
            "gfn2_n_elements",
            "gfn2_cache_path",
            "gfn2_d4_s8",
            "gfn2_d4_a1",
            "gfn2_d4_a2",
            "gfn2_electronic_temperature",
            "gfn2_charge_mixing",
            "gfn2_scc_mixer",
            "gfn2_aes_faithful",
            "gfn2_cache_sha256",
            "gfn2_d4_refdata_sha256",
        }
    )

    @requires_gfn2
    def test_provenance_dict_is_non_empty_and_has_expected_keys(self, params):
        from vibeqc.semiempirical.runner import _gfn2_parameter_provenance

        model = GFN2Model(_mol(_water()), params, warn=False)
        model.energy()  # materialise the SCC result
        prov = _gfn2_parameter_provenance(model)

        assert isinstance(prov, dict)
        assert len(prov) >= len(self._REQUIRED_KEYS)
        missing = self._REQUIRED_KEYS - set(prov.keys())
        assert not missing, f"Missing provenance keys: {sorted(missing)}"

        # All values must be flat scalars (str, int, float, bool).
        for key, value in prov.items():
            assert isinstance(value, (str, int, float, bool)), (
                f"provenance[{key!r}] = {value!r} ({type(value).__name__}), "
                f"expected flat scalar"
            )

        # SHA-256 checksums must be 64 hex chars or "unavailable".
        for sha_key in ("gfn2_cache_sha256", "gfn2_d4_refdata_sha256"):
            val = prov[sha_key]
            assert isinstance(val, str)
            assert len(val) == 64 or val == "unavailable", (
                f"{sha_key}: expected 64-char hex or 'unavailable', got {val!r}"
            )

        # Numeric D4 params must be finite.
        for num_key in ("gfn2_d4_s8", "gfn2_d4_a1", "gfn2_d4_a2"):
            assert np.isfinite(prov[num_key]), f"{num_key} is not finite"

        # Strings must not be empty.
        for str_key in (
            "gfn2_param_version",
            "gfn2_param_origin",
            "gfn2_param_license",
            "gfn2_param_doi",
            "gfn2_cache_path",
        ):
            assert len(prov[str_key]) > 0, f"{str_key} is empty"

    @requires_gfn2
    def test_semiempirical_result_carries_provenance_from_gfn2_path(self, params):
        from vibeqc.semiempirical import SemiempiricalResult
        from vibeqc.semiempirical.routes import SemiempiricalRoutePlan
        from vibeqc.semiempirical.runner import _run_molecular_semiempirical

        plan = SemiempiricalRoutePlan.from_request(
            "gfn2_xtb",
            boundary="molecule",
            charge=0,
            multiplicity=1,
        )
        mol = _mol(_water())
        result = _run_molecular_semiempirical(plan, mol)

        assert isinstance(result, SemiempiricalResult)
        assert result.parameter_provenance is not None
        assert isinstance(result.parameter_provenance, dict)
        assert "gfn2_param_version" in result.parameter_provenance
        from vibeqc.dispersion_d4_parameters import get_d4_params

        assert result.parameter_provenance["gfn2_d4_s8"] == pytest.approx(
            float(get_d4_params("gfn2xtb").s8)
        )

    @requires_gfn2
    def test_dftb_result_carries_exact_native_parameter_provenance(self):
        from vibeqc.semiempirical import SemiempiricalResult
        from vibeqc.semiempirical.routes import SemiempiricalRoutePlan
        from vibeqc.semiempirical.runner import _run_molecular_semiempirical

        plan = SemiempiricalRoutePlan.from_request(
            "dftb0",
            boundary="molecule",
            charge=0,
            multiplicity=1,
        )
        mol = _mol(_water())
        result = _run_molecular_semiempirical(plan, mol)

        assert isinstance(result, SemiempiricalResult)
        assert result.parameter_provenance == {
            "parameter_identity": result.parameter_identity,
            "parameter_sha256": result.parameter_sha256,
        }
        assert result.parameter_identity
        assert len(result.parameter_sha256) == 64

    @requires_gfn2
    def test_provenance_is_distinct_per_model_instance(self, params):
        from vibeqc.semiempirical.runner import _gfn2_parameter_provenance

        m1 = GFN2Model(_mol(_water()), params, warn=False, charge_mixing=0.1)
        m1.energy()
        m2 = GFN2Model(_mol(_water()), params, warn=False, charge_mixing=0.3)
        m2.energy()

        p1 = _gfn2_parameter_provenance(m1)
        p2 = _gfn2_parameter_provenance(m2)

        # Different charge mixing → distinct provenance.
        assert p1["gfn2_charge_mixing"] == 0.1
        assert p2["gfn2_charge_mixing"] == 0.3

    def test_manual_result_accepts_provenance_kwarg(self):
        from vibeqc.semiempirical.runner import SemiempiricalResult

        r = SemiempiricalResult(energy=-1.0, parameter_provenance={"k": "v"})
        assert r.parameter_provenance == {"k": "v"}

        r_none = SemiempiricalResult(energy=-1.0)
        assert r_none.parameter_provenance is None

    @requires_gfn2
    def test_ar_fcc_fails_closed_until_physics_parity(self, params):
        """Rare-gas cells must not return the known attractive-collapse curve."""
        import numpy as np
        from vibeqc._vibeqc_core import (
            Atom,
            PeriodicSystem,
            bloch_kmesh_from_lists,
        )
        from vibeqc._vibeqc_core.semiempirical.xtb import (
            XTBSccOptions,
            run_gfn2_xtb_gamma,
            run_gfn2_xtb_kpoints,
        )

        a = 5.26
        sc = lambda x, y, z: [x * a, y * a, z * a]
        system = PeriodicSystem()
        system.lattice = np.diag([a, a, a])
        system.unit_cell = [
            Atom(18, sc(0.0, 0.0, 0.0)),
            Atom(18, sc(0.0, 0.5, 0.5)),
            Atom(18, sc(0.5, 0.0, 0.5)),
            Atom(18, sc(0.5, 0.5, 0.0)),
        ]
        opts = XTBSccOptions()
        opts.charge_mixing = 0.2
        opts.max_iter = 200
        with pytest.raises(RuntimeError, match="rare-gas.*attractive-collapse"):
            run_gfn2_xtb_gamma(system, params, opts, cutoff_bohr=10.0)
        gamma_mesh = bloch_kmesh_from_lists([[0.0, 0.0, 0.0]], [1.0])
        with pytest.raises(RuntimeError, match="rare-gas.*attractive-collapse"):
            run_gfn2_xtb_kpoints(
                system, params, gamma_mesh, opts, cutoff_bohr=10.0
            )

    @requires_gfn2
    def test_h2_molecular_limit_requires_explicit_molecular_route(self, params):
        """A home-only Gamma list must not masquerade as periodic H2."""
        import numpy as np
        from vibeqc._vibeqc_core import Atom, PeriodicSystem
        from vibeqc._vibeqc_core.semiempirical.xtb import (
            XTBSccOptions,
            run_gfn2_xtb_gamma,
        )

        ps = PeriodicSystem()
        ps.lattice = np.diag([20.0, 20.0, 20.0])
        ps.unit_cell = [Atom(1, [0, 0, 0]), Atom(1, [1.4, 0, 0])]
        opts = XTBSccOptions()
        opts.charge_mixing = 0.2
        opts.max_iter = 100
        with pytest.raises(
            ValueError,
            match=r"no nonzero lattice image.*free-boundary cluster",
        ):
            run_gfn2_xtb_gamma(ps, params, opts, cutoff_bohr=15.0)

    @requires_gfn2
    def test_skew_fcc_cu_fails_closed_on_overpolarized_basin(self, params):
        """The periodic driver's home-cell gamma carries no image screening,
        so skew transition-metal cells admit the spurious over-polarized
        intra-atomic shell-transfer fixed point (IID 300). The physical-basin
        gate must fail the run closed instead of returning the -2.39e6 Ha
        third-order runaway energy."""
        import numpy as np
        from vibeqc._vibeqc_core import Atom, PeriodicSystem
        from vibeqc._vibeqc_core.semiempirical.xtb import (
            XTBSccOptions,
            run_gfn2_xtb_gamma,
        )

        a = 3.615
        prim = [
            np.array([0.0, 0.5, 0.5]) * a,
            np.array([0.5, 0.0, 0.5]) * a,
            np.array([0.5, 0.5, 0.0]) * a,
        ]
        atoms = [
            i * prim[0] + j * prim[1] + k * prim[2]
            for i in range(2) for j in range(2) for k in range(2)
        ]
        lattice = np.column_stack([2.0 * p for p in prim])
        ps = PeriodicSystem(
            3, lattice, [Atom(29, c.tolist()) for c in atoms], 0, 1
        )
        opts = XTBSccOptions()
        opts.electronic_temperature = 0.005
        result = run_gfn2_xtb_gamma(ps, params, opts, cutoff_bohr=15.0)
        assert not result.converged


# ===================================================================
# 8. Faithful AES (EXPERIMENTAL, opt-in via XTBSccOptions.aes_faithful)
# ===================================================================
# The faithful atom-resolved GFN2 AES (Bannwarth, Ehlert & Grimme, JCTC 2019,
# doi:10.1021/acs.jctc.8b01176): on-site DPOL/QPOL XC kernel + CN-damped
# (fdmp3/fdmp5) inter-site charge-dipole / dipole-dipole / charge-quadrupole,
# added through an opt-in SCC Fock potential and converged-density energy.
# Default OFF (the shipped ad-hoc shell-resolved path is unchanged unless the
# flag is set). These tests pin the invariants the faithful path must satisfy;
# the quantitative xtb-parity gate is wired to the live oracle values in
# runner_xtb (handovers/HANDOVER_GFN2_AES.md §6).
def _energy_aes(mol, params, max_iter=400):
    opts = _xtb.XTBSccOptions()
    opts.charge_mixing = 0.2
    opts.max_iter = max_iter
    opts.aes_faithful = True
    return _xtb.run_gfn2_xtb(mol, params, opts)


class TestGFN2FaithfulAES:
    @requires_gfn2
    def test_flag_default_off_is_unchanged(self, params):
        """aes_faithful defaults to False and the default result is identical
        to a run that does not touch the flag (no silent behaviour change)."""
        opts = _xtb.XTBSccOptions()
        assert opts.aes_faithful is False
        e_plain = _energy(_mol(_water()), params).energy
        opts.charge_mixing = 0.2
        opts.max_iter = 400
        e_flagoff = _xtb.run_gfn2_xtb(_mol(_water()), params, opts).energy
        assert e_plain == e_flagoff

    @requires_gfn2
    def test_faithful_converges_and_deterministic(self, params):
        r1 = _energy_aes(_mol(_water()), params)
        r2 = _energy_aes(_mol(_water()), params)
        assert r1.converged
        assert r1.energy == r2.energy

    @requires_gfn2
    def test_faithful_path_is_actually_active(self, params):
        """The faithful AES energy must differ from the ad-hoc path — i.e. the
        flag is wired through to the energy, not a no-op."""
        e_default = _energy(_mol(_water()), params).energy
        e_faithful = _energy_aes(_mol(_water()), params).energy
        assert abs(e_faithful - e_default) > 1e-6

    @requires_gfn2
    def test_faithful_charges_physical(self, params):
        """The faithful atom-resolved AES path must keep physical water charges
        (O negative, H positive, neutral total)."""
        r = _energy_aes(_mol(_water()), params)
        q = np.asarray(r.charges)
        assert q[0] < 0.0 and q[1] > 0.0 and q[2] > 0.0
        assert abs(q.sum()) < 1e-8

    @requires_gfn2
    def test_faithful_inter_site_aes_is_size_consistent(self, params):
        """The defining check of the inter-site damped multipoles: the residual
        must decay as the physical ~1/R³ dipole-dipole tail (fdmp5 → 1 at large
        R) and vanish in the limit — NOT couple spuriously like the pre-fix
        non-decaying repulsion.  Mirrors test_water_dimer_residual_* for the
        faithful path."""
        e_mono = _energy_aes(_mol(_water()), params).energy
        r100 = _energy_aes(_mol(_water(0.0) + _water(100.0)), params).energy - 2 * e_mono
        r200 = _energy_aes(_mol(_water(0.0) + _water(200.0)), params).energy - 2 * e_mono
        r400 = _energy_aes(_mol(_water(0.0) + _water(400.0)), params).energy - 2 * e_mono
        assert abs(r200) < 0.3 * abs(r100)  # ~1/R³ decay
        if abs(r200) > 1e-8:
            assert abs(r400) < 0.3 * abs(r200)
        assert abs(r400) < 1e-6  # size-consistent in the limit

    @requires_gfn2
    @pytest.mark.parametrize(
        "tag,sep", [
            ("trans_5.6", 5.6),
            ("trans_6.0", 6.0),
            ("trans_7.0", 7.0),
        ]
    )
    def test_water_dimer_vs_xtb_oracle(self, params, tag, sep):
        """Discriminating parity gate: vibe-qc's SCC-only interaction (faithful
        AES) vs the live xtb GFN2 reference on the SAME geometry, ~1 kcal/mol.

        2026-08-14 (GFN2-MOL-PARITY): the gate failed for trans_5.6/6.0 until
        three defects were fixed -- (1) the STO-NG basis selected rows by l
        alone (the O 2s shell ran the 1s expansion), (2) the faithful-AES Fock
        used a chain rule over atom-centred traceless moments instead of xtb's
        global-origin setvsdq/buildIsoAnisotropicH1 construction, and (3) the
        first SCC iteration skipped the charge-only AES channels.  With the
        fixed basis + ported Fock the three interactions land within 0.5 mHa
        of the xtb oracle."""
        if _xtb_oracle is None:
            pytest.skip("xtb oracle runner not importable")
        ref = _xtb_oracle.dimer_scc_interaction_ref(tag)
        e_mono = _energy_aes(_mol(_water()), params).energy
        e_dim = _energy_aes(_mol(_water(0.0) + _water(sep)), params).energy
        e_int = e_dim - 2 * e_mono
        assert abs(e_int - ref) < 2.0e-3, (
            f"{tag}: vibe-qc {e_int * 1e3:+.2f} mHa vs xtb {ref * 1e3:+.2f} mHa"
        )

    @requires_gfn2
    def test_faithful_water_pes_minimum_near_equilibrium(self, params):
        """The faithful path must keep a physical O-H well near 1.81 bohr - a
        guard that the AES Fock potential does not wreck the PES."""
        import math

        best_r, best_e = None, None
        for r in [1.60, 1.70, 1.75, 1.81, 1.90, 2.00, 2.10]:
            a = math.radians(52.25)
            mol = _mol(
                [
                    Atom(8, [0, 0, 0]),
                    Atom(1, [0, r * math.sin(a), r * math.cos(a)]),
                    Atom(1, [0, -r * math.sin(a), r * math.cos(a)]),
                ]
            )
            e = _energy_aes(mol, params).energy
            if best_e is None or e < best_e:
                best_e, best_r = e, r
        assert 1.70 <= best_r <= 1.95, f"faithful-AES PES min at r={best_r}"


class TestGFN2DFirstParameterProjection:
    """Issue #43: per-angular-momentum parameter projection for d-first
    elements. The xtb parameter file lists transition-metal shells in
    occupation order (ao=3d4s4p: Z 21-29, 39-47, 57-79), while every
    consumer indexes the derived per-l arrays by angular momentum.
    Positional indexing therefore rotated the STO exponents and on-site
    levels (C++ core arrays) and the kcn/poly fields (TOML converter)
    across the shells of all 41 d-first elements, deviating Cu2 from the
    xtb oracle by +36 to +140 mHa, all in the H0/band channel."""

    @requires_gfn2
    def test_cu_shell_parameters_are_l_projected(self):
        from vibeqc.semiempirical.methods.gfn2_params import (
            _read_cached_toml,
        )

        data = _read_cached_toml()
        assert data.get("projection") == "per-l"
        cu = next(e for e in data["element"] if int(e["Z"]) == 29)
        by_l = {int(s["l"]): s for s in cu["shells"]}
        assert set(by_l) == {0, 1, 2}
        # KCNS/KCNP/KCND and POLYS/POLYP/POLYD are per-l named lines in
        # the upstream file; the converter scales kcn by 0.1/27.2114 and
        # poly by 0.01. The rotated cache carried the s value on the d
        # shell (and cyclically for the rest).
        assert by_l[0]["kcn"] == pytest.approx(+0.012308, abs=2.0e-6)
        assert by_l[1]["kcn"] == pytest.approx(-0.009626, abs=2.0e-6)
        assert by_l[2]["kcn"] == pytest.approx(-0.000638, abs=2.0e-6)
        assert by_l[0]["poly"] == pytest.approx(+0.177983, abs=2.0e-6)
        assert by_l[1]["poly"] == pytest.approx(+0.149778, abs=2.0e-6)
        assert by_l[2]["poly"] == pytest.approx(-0.265089, abs=2.0e-6)
        # The 3d STO must be the most contracted; the rotated core
        # arrays handed the 4s shell the 3d exponent.
        assert by_l[2]["zeta"] > by_l[1]["zeta"] > by_l[0]["zeta"]

    @requires_gfn2
    def test_cu2_dimer_total_energy_parity_vs_xtb(self, params):
        """Pinned xtb 6.7.0 totals (--gfn 2 --acc 1.0, T = 300 K). The
        pre-projection tree deviated +35.8/+56.8/+139.5 mHa, growing
        with distance and living entirely in e_band0; the projected
        parameters leave the known, tracked anisotropic-ES/D4 remainder
        (+19.1/+31.5/+47.8 mHa measured 2026-08-27). The bounds sit
        between the two regimes so a regression to positional indexing
        fails loudly at every distance."""
        xtb_totals = {
            2.2: -7.775577494527,
            2.5: -7.765435511614,
            3.0: -7.721422955823,
        }
        bounds_mha = {2.2: 27.0, 2.5: 42.0, 3.0: 70.0}
        bohr = 1.8897259886
        for d, ref in xtb_totals.items():
            mol = _mol(
                [Atom(29, [0, 0, 0]), Atom(29, [0, 0, d * bohr])]
            )
            opts = _xtb.XTBSccOptions()
            opts.electronic_temperature = 0.00095
            opts.max_iter = 600
            result = _xtb.run_gfn2_xtb(mol, load_gfn2_params(), opts)
            assert result.converged
            deviation_mha = abs(result.energy - ref) * 1000.0
            assert deviation_mha < bounds_mha[d], (
                f"Cu2 d={d}: {deviation_mha:.1f} mHa vs xtb "
                f"(bound {bounds_mha[d]})"
            )


# ---------------------------------------------------------------------------
# Issue #467: the TOML converter's shell-record keying.
#
# The synthetic block below is written here rather than sliced out of the
# upstream param_gfn2-xtb.txt: vibe-qc fetches that file at runtime under
# LGPL-3.0 and does not redistribute it (CLAUDE.md section 1). The values
# are invented and chosen only to be pairwise distinct, so a rotation
# cannot pass by coincidence.
_SYNTHETIC_KCN = {"S": 1.0, "P": -2.0, "D": -3.0}
_SYNTHETIC_POLY = {"S": 10.0, "P": -20.0, "D": -30.0}
# Converter scale factors, as applied in _render_parameter_cache.
_KCN_SCALE = 0.1 / 27.2114  # eV -> Ha, with xtb's 0.1 on KCN*
_POLY_SCALE = 0.01
# The rendered TOML carries 6 decimals, so half an ulp of that is the
# tightest honest tolerance.
_RENDER_TOL = 5.0e-7


def _synthetic_element_block(order):
    """A one-element upstream block, KCN*/POLY* emitted in ``order``.

    ``ao=3d4s4p`` is the d-first transition-metal layout, so the shells
    are listed l = 2, 0, 1 while lev/exp stay positional. That is the
    combination issue #43 got wrong; ``order`` additionally varies the
    line order of the per-angular-momentum records, which is what
    issue #467 is about.
    """
    lines = [
        "$Z=29 synthetic block (invented values, not upstream data)",
        " ao=3d4s4p",
        " lev=   -9.000000   -6.000000   -2.000000",
        " exp=    2.300000    1.500000    1.900000",
        " GAM=    0.200000",
    ]
    lines += [f" KCN{x}=   {_SYNTHETIC_KCN[x]:.6f}" for x in order]
    lines += [f" POLY{x}=  {_SYNTHETIC_POLY[x]:.6f}" for x in order]
    lines.append("$end")
    return "\n".join(lines) + "\n"


def _rendered_shells_by_l(order):
    """Render the synthetic block and index its shells by l."""
    rendered = _gfn2_params._render_parameter_cache(
        _synthetic_element_block(order)
    )
    data = tomllib.loads(rendered)
    assert data["projection"] == "per-l"
    (element,) = data["element"]
    return {int(shell["l"]): shell for shell in element["shells"]}


class TestGFN2ShellRecordKeying:
    """Issue #467: KCNS/KCNP/KCND and POLYS/POLYP/POLYD are keyed by
    their shell LETTER, not by their position in the element block.

    Collecting them positionally is correct only while every upstream
    block happens to list them in s, p, d order. That held for all 86
    blocks of the revision current at 2026-08-28, but nothing enforced
    it, and a revision that emitted KCND before KCNS would silently
    re-create the 3-cycle of issue #43 (re-attested under issue #446):
    the rotated cache stays structurally valid, so no test fails and the
    loaded content digest merely moves off GFN2_PUBLISHED_SHA256, which
    reads as tampering rather than as a mis-read source.
    """

    @requires_gfn2
    def test_out_of_order_records_land_on_their_own_l(self):
        by_l = _rendered_shells_by_l(("D", "S", "P"))

        assert set(by_l) == {0, 1, 2}
        for letter, lv in (("S", 0), ("P", 1), ("D", 2)):
            assert by_l[lv]["kcn"] == pytest.approx(
                _SYNTHETIC_KCN[letter] * _KCN_SCALE, abs=_RENDER_TOL
            ), f"KCN{letter} did not land on l = {lv}"
            assert by_l[lv]["poly"] == pytest.approx(
                _SYNTHETIC_POLY[letter] * _POLY_SCALE, abs=_RENDER_TOL
            ), f"POLY{letter} did not land on l = {lv}"

    @requires_gfn2
    def test_lev_and_exp_stay_positional(self):
        """The two conventions coexist, which is why the letter matters.

        ``lev``/``exp`` carry one value per shell in the block's own
        shell order (here d, s, p), so they index by position; only the
        named per-angular-momentum records index by l.
        """
        by_l = _rendered_shells_by_l(("D", "S", "P"))

        expected = ((2, -9.0, 2.3), (0, -6.0, 1.5), (1, -2.0, 1.9))
        for lv, en_ev, zeta in expected:
            assert by_l[lv]["en"] == pytest.approx(
                en_ev / 27.2114, abs=_RENDER_TOL
            )
            assert by_l[lv]["zeta"] == pytest.approx(zeta, abs=_RENDER_TOL)

    @requires_gfn2
    def test_record_line_order_does_not_change_the_rendered_shells(self):
        canonical = _rendered_shells_by_l(("S", "P", "D"))

        for order in (("D", "S", "P"), ("P", "D", "S"), ("D", "P", "S")):
            assert _rendered_shells_by_l(order) == canonical, (
                f"record order {order} changed the rendered shells"
            )

    @requires_gfn2
    def test_duplicate_shell_record_is_rejected(self):
        # Two KCNS records in one block: silently keeping the last would
        # leave l = 1 with no value of its own.
        text = _synthetic_element_block(("S", "P", "D")).replace(
            " KCNP=", " KCNS=", 1
        )

        with pytest.raises(
            _gfn2_params._GFN2ParameterSourceMalformedError, match="repeats"
        ):
            _gfn2_params._render_parameter_cache(text)

    @requires_gfn2
    def test_unrecognised_shell_letter_is_rejected(self):
        text = _synthetic_element_block(("S", "P", "D")).replace(
            " KCND=", " KCNF=", 1
        )

        with pytest.raises(
            _gfn2_params._GFN2ParameterSourceMalformedError,
            match="cannot map to an angular momentum",
        ):
            _gfn2_params._render_parameter_cache(text)

    @requires_gfn2
    def test_multi_valued_shell_record_is_rejected(self):
        text = _synthetic_element_block(("S", "P", "D")).replace(
            " KCNS=   1.000000", " KCNS=   1.000000   2.000000", 1
        )

        with pytest.raises(
            _gfn2_params._GFN2ParameterSourceMalformedError,
            match="carries 2 values",
        ):
            _gfn2_params._render_parameter_cache(text)

    @requires_gfn2
    def test_a_malformed_record_aborts_the_refresh(self, monkeypatch):
        """A record the converter cannot map must abort the refresh.

        The failure has to reach _fetch_and_parse as a source-refresh
        error so that no cache file is installed: a half-mapped cache
        would be indistinguishable from a tampered one.
        """
        text = _synthetic_element_block(("S", "P", "D")).replace(
            " KCND=", " KCNF=", 1
        )

        monkeypatch.setattr(
            _gfn2_params,
            "_fetch_and_parse_impl",
            lambda: _gfn2_params._render_parameter_cache(text),
        )
        with pytest.raises(_gfn2_params._GFN2ParameterSourceRefreshError):
            _gfn2_params._fetch_and_parse()


def test_physical_basin_gate_constants_are_defined_once() -> None:
    """Issue #411: the basin-gate bounds must not fork into per-driver copies.

    The five constants and their predicates were duplicated by the
    1fc1dc73b periodic-Gamma port. A recalibration touching one copy and not
    the other silently diverges the two routes' fail-closed behaviour, so the
    definition may only exist in the shared core header; the drivers only
    call through it.
    """
    root = Path(__file__).resolve().parent.parent
    header = root / "cpp" / "include" / "vibeqc" / "semiempirical" / "core" / "basin_gates.hpp"
    drivers = (
        root / "cpp" / "src" / "semiempirical" / "seccm" / "gfn2.cpp",
        root / "cpp" / "src" / "semiempirical" / "methods" / "xtb"
        / "periodic_gfn2.cpp",
    )
    names = (
        "kShellPopulationViolationTolerance",
        "kShellRmsPolarizationLimit",
        "kAtomicRmsPolarizationLimit",
        "kLocalShellPolarizationLimit",
        "kLocalAtomicPolarizationLimit",
    )
    header_text = header.read_text(encoding="utf-8")
    for name in names:
        assert f"constexpr double {name} =" in header_text
        for driver in drivers:
            text = driver.read_text(encoding="utf-8")
            assert f"constexpr double {name} =" not in text


# Adenine, qc-input-library scripts/_geometries.py :: adenine() (Angstrom);
# the same frame tests/test_scc_dftb_retry_ladder.py uses.
_ADENINE_ANGSTROM = [
    (7, 0.000, 1.280, 0.000),
    (6, 1.110, 0.552, 0.000),
    (6, -1.110, 0.552, 0.000),
    (7, 0.690, -0.806, 0.000),
    (6, -0.690, -0.806, 0.000),
    (7, 2.272, 1.250, 0.000),
    (7, -2.180, 1.400, 0.000),
    (6, -1.490, -1.710, 0.000),
    (7, -2.820, -1.120, 0.000),
    (6, 1.490, -1.710, 0.000),
    (1, 3.170, 0.792, 0.000),
    (1, 2.169, 2.252, 0.000),
    (1, -3.170, 0.900, 0.000),
    (1, -2.100, 2.390, 0.000),
    (1, -1.450, -2.790, 0.000),
]


# ---------------------------------------------------------------------------
# Faithful-AES external parity (issues #43, #475)
# ---------------------------------------------------------------------------
#
# References measured live against xtb 6.7.1 on 2026-09-08 through an
# out-of-process subprocess call (CLAUDE.md s10), on the exact geometries
# below.  These replace the reasoning in #43, whose headline figure is stale:
# it recorded a +245.7 mHa adenine residual, and the shipped path is now at
# -1.86 mHa.
_XTB_671_LIVE_EH = {
    "H2": -0.98201714,
    "H2O": -5.07036453,
    "NH3": -4.42554157,
    "adenine": -27.49957548,
}


@requires_gfn2
@pytest.mark.parametrize(
    ("name", "ad_hoc_mha", "faithful_mha"),
    [
        ("H2", -16.497, 0.051),
        ("H2O", 0.601, 0.138),
        ("NH3", -5.089, 0.362),
    ],
)
def test_faithful_aes_closes_the_molecular_xtb_residual(
    name, ad_hoc_mha, faithful_mha
):
    """Issue #43: the residual is the on-site DPOL/QPOL term, and this is it.

    The shipped ad-hoc shell-resolved multipole channel omits the on-site
    anisotropic XC term of Bannwarth 2019 Eq. 31.  Switching to the faithful
    atom-resolved model collapses the error against live xtb 6.7.1 by one to
    two orders of magnitude on every small polar system:

        system   ad-hoc      faithful
        H2       -16.497     +0.051    mHa
        H2O       +0.601     +0.138    mHa
        NH3       -5.089     +0.362    mHa

    This is why `aes_faithful` exists.  It is NOT yet the default; see
    test_adenine_molecular_scc_has_multiple_converged_basins for the reason.
    """
    geometries = {
        "H2": [(1, 0.0, 0.0, 0.0), (1, 0.740848, 0.0, 0.0)],
        "H2O": [
            (8, 0.0, 0.0, 0.0),
            (1, 0.757, 0.587, 0.0),
            (1, -0.757, 0.587, 0.0),
        ],
        "NH3": [
            (7, 0.0, 0.0, 0.0),
            (1, 0.94, 0.0, 0.33),
            (1, -0.47, 0.814, 0.33),
            (1, -0.47, -0.814, 0.33),
        ],
    }
    bohr = 1.0 / 0.529177210903
    molecule = Molecule(
        [
            Atom(z, [x * bohr, y * bohr, w * bohr])
            for z, x, y, w in geometries[name]
        ],
        0,
        1,
    )
    params = load_gfn2_params()
    reference = _XTB_671_LIVE_EH[name]

    errors = {}
    for label, faithful in (("ad_hoc", False), ("faithful", True)):
        options = _xtb.XTBSccOptions()
        options.max_iter = 3000
        options.conv_tol_charge = 1.0e-9
        options.aes_faithful = faithful
        result = _xtb.run_gfn2_xtb(molecule, params, options)
        assert result.converged
        errors[label] = (result.energy - reference) * 1.0e3

    assert errors["ad_hoc"] == pytest.approx(ad_hoc_mha, abs=0.01)
    assert errors["faithful"] == pytest.approx(faithful_mha, abs=0.01)
    # The point of the model change, stated as an inequality so it survives a
    # re-pin: the faithful channel is closer to xtb than the shipped one.
    assert abs(errors["faithful"]) < abs(errors["ad_hoc"])


@requires_gfn2
def test_adenine_molecular_scc_has_multiple_converged_basins():
    """Why the molecular AES default has not moved (issues #43, #475).

    Adenine is the one system where the faithful channel does not help, and
    the reason is that its molecular GFN2 SCC does not have a single
    converged state.  Sweeping the AES model, the auto-stabilisation retry
    and the mixer at conv_tol 1e-8 reaches four distinct fixed points
    spanning 29.4 mHa:

        E (Ha)         vs xtb 6.7.1   reached by
        -27.5081085      -8.533 mHa   ad-hoc,   no auto-stabilise
        -27.5014338      -1.858 mHa   ad-hoc,   auto-stabilise (shipped)
        -27.4860432     +13.532 mHa   faithful, no auto-stabilise
        -27.4787423     +20.833 mHa   faithful, auto-stabilise

    So on adenine the faithful model is *further* from xtb than the shipped
    one, but the comparison is confounded: which state is reached is decided
    by solver settings rather than by the Hamiltonian.  #43's 0.1 mHa
    tolerance cannot be assessed on this system until that is resolved, and
    the default must not move on evidence this ambiguous.  #475 reports the
    same system as its iteration-count outlier, which is unlikely to be a
    coincidence.

    This test pins the multiplicity so it cannot change silently.  It is a
    defect record, not an endorsement of any of the four numbers.
    """
    bohr = 1.0 / 0.529177210903
    molecule = Molecule(
        [
            Atom(z, [x * bohr, y * bohr, w * bohr])
            for z, x, y, w in _ADENINE_ANGSTROM
        ],
        0,
        1,
    )
    params = load_gfn2_params()

    energies = set()
    for faithful in (False, True):
        for stabilize in (True, False):
            options = _xtb.XTBSccOptions()
            options.max_iter = 4000
            options.conv_tol_charge = 1.0e-8
            options.aes_faithful = faithful
            options.auto_stabilize = stabilize
            options.aes_damping = 0.25 if faithful else 0.5
            result = _xtb.run_gfn2_xtb(molecule, params, options)
            assert result.converged
            energies.add(round(result.energy, 7))

    assert len(energies) == 4, sorted(energies)
    assert max(energies) - min(energies) == pytest.approx(0.0293662, abs=1e-6)
    # The shipped configuration is the ad-hoc channel with auto-stabilisation.
    assert min(abs(e - (-27.5014338)) for e in energies) < 1e-6
