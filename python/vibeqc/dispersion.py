"""Unified D3(BJ) dispersion correction entry point.

Two backends are available:

* **``"builtin"``** -- the native C++ implementation in vibe-qc's core.
  Self-contained and quantitative for H through Ar. It ships the complete
  H--Ar corner of Grimme's coordination-dependent C6 reference grid and
  all 156 D3(BJ) damping-parameter sets from the pinned simple-dftd3 data.

* **``"dftd3"``** -- a thin wrapper over `Grimme's reference Fortran
  library <https://github.com/dftd3/simple-dftd3>`_, distributed as
  the ``dftd3`` PyPI package (LGPL-3). Full element coverage up to
  plutonium and the canonical CN-dependent c6ab table baked into the
  shared library. It remains available as an independent reference and
  for elements beyond Ar.

  vibe-qc treats ``dftd3`` as an **optional** dependency -- it is
  listed in ``pyproject.toml`` under the ``dispersion`` extra, so
  ``pip install -e '.[dispersion]'`` from the vibe-qc checkout (or a
  plain ``pip install dftd3`` into the venv) pulls it in. If it is
  not installed, ``backend="dftd3"`` raises ``ImportError`` with a
  clear message.

* **``"auto"``** (default) -- selects the native backend for H--Ar.
  Molecules containing heavier elements use ``dftd3`` when installed and
  otherwise fail with an installation hint.

Three-body (ATM) term
---------------------

:class:`D3BJParams` carries ``s9``, the scale factor of the
Axilrod-Teller-Muto three-body dispersion term (``a3`` in Eq. (3) of
Grimme, Brandenburg, Bannwarth & Hansen, *J. Chem. Phys.* **143**,
054107 (2015)). It defaults to ``0.0``: the published per-functional
``(s8, a1, a2)`` sets were fit without the three-body term, so a plain
``compute_d3bj(mol, "pbe")`` is the two-body D3(BJ) energy. Methods
whose *definition* includes ATM -- the PBEh-3c and HSE-3c composites --
carry ``s9 = 1.0`` on their recipe (see :mod:`vibeqc.composites`).

Both backends honour ``s9`` and agree on the ATM term throughout the native
H--Ar range. ``s9`` is always passed to ``dftd3``
explicitly -- ``dftd3``'s own ``RationalDampingParam`` defaults it to
``1.0``, so relying on that default would silently apply ATM on the
``dftd3`` backend and not on ``builtin``, making the answer depend on
whether an optional package happens to be installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from ._vibeqc_core import (
    D3BJParams,
    DispersionResult,
    Molecule,
    compute_d3bj as _compute_d3bj_builtin,
    d3_max_supported_Z as _d3_max_supported_Z,
    d3bj_params_for as _d3bj_params_for_builtin,
)


__all__ = [
    "DispersionResult",
    "D3BJParams",
    "compute_d3bj",
    "d3bj_params_for",
    "dftd3_available",
]


def dftd3_available() -> bool:
    """Return True if the optional ``dftd3`` backend can be imported.

    Used by feature probes, the runtime banner, and tests that conditionally
    exercise the external reference implementation.
    """
    try:
        import dftd3.interface  # noqa: F401
        return True
    except ImportError:
        return False


def _resolve_backend(backend: str) -> str:
    if backend == "auto":
        return "builtin"
    if backend in ("builtin", "dftd3"):
        return backend
    raise ValueError(
        f"compute_d3bj: unknown backend {backend!r} "
        f"(expected 'auto', 'builtin', or 'dftd3')"
    )


# B3LYP flavor spellings (b3lyp5 / b3lyp/g / b3lypg) share the single
# "b3lyp" damping-parameter set: Grimme's fits are keyed to the B3LYP
# recipe, not to the VWN parametrisation inside it (the flavor moves
# E_xc by ~mHa -- far below the resolution of the damping fit).
#
# D3(BJ)-only aliases: the simple-dftd3 registry carries two entries
# for DSD-PBEP86 -- "dsdpbep86" (2013 JCC fit, s6=0.48) and
# "dsdpbep86_2011" (2011 PCCP / GMTKN55 fit, s6=0.418).  The vibe-qc
# dispatcher passes the hyphenated name "dsd-pbep86".  These aliases
# route only the D3(BJ) path to the 2011 entry; the D4 path uses its
# own key normalisation and must NOT see the _2011 suffix.
_D3BJ_NAME_ALIASES = {
    "dsd-pbep86": "dsdpbep86_2011",
    # The official SKALA-1.1 checkpoint embeds expected_d3_settings=
    # "b3lyp5". D3's registry keys that damping fit as "b3lyp"; keep this
    # D3-only so a request for D4 cannot silently substitute B3LYP-D4.
    "skala": "b3lyp",
    "skala-1.1": "b3lyp",
    "skala-1.1-rev1": "b3lyp",
}
_DISPERSION_NAME_ALIASES = {
    "bp86": "bp",
    "b3lyp5": "b3lyp",
    "b3lyp/g": "b3lyp",
    "b3lypg": "b3lyp",
}


def _canonical_dispersion_name(functional: str) -> str:
    key = functional.lower()
    return _DISPERSION_NAME_ALIASES.get(key, key)


def d3bj_params_for(
    functional: str,
    *,
    backend: str = "auto",
) -> Optional[D3BJParams]:
    """Look up the D3(BJ) damping parameters for a DFT functional.

    Both backends expose the 156 D3(BJ) sets from Grimme's canonical
    ``parameters.toml``. ``"auto"`` uses the native registry and does not
    require an optional package.

    Returns ``None`` if the functional is unknown to the chosen
    backend.
    """
    functional = _canonical_dispersion_name(functional)
    # D3(BJ)-only aliases: route per-functional overrides (e.g.
    # dsd-pbep86 → the GMTKN55-validated dsdpbep86_2011 entry).
    # These must NOT go into _DISPERSION_NAME_ALIASES because
    # _canonical_dispersion_name is shared with the D4 path.
    functional = _D3BJ_NAME_ALIASES.get(functional, functional)
    chosen = _resolve_backend(backend)
    if chosen == "dftd3":
        try:
            return _params_from_dftd3(functional)
        except ImportError as e:
            raise ImportError(
                "d3bj_params_for(backend='dftd3') requires the optional "
                "dftd3 package. Install it with "
                "`pip install dftd3` into your vibe-qc venv, or "
                "`pip install -e '.[dispersion]'` from the repo checkout."
            ) from e
    return _d3bj_params_for_builtin(functional)


def _params_from_dftd3(functional: str) -> Optional[D3BJParams]:
    """Read the canonical D3-BJ damping parameters for a functional
    from the ``dftd3`` package's ``parameters.toml``. Returns ``None``
    if the functional has no 'bj' entry."""
    import importlib.resources as _res
    try:
        # Python 3.11+: tomllib is stdlib
        import tomllib
    except ImportError:  # pragma: no cover
        import tomli as tomllib

    # The parameters file ships inside the dftd3 package.
    with _res.files("dftd3").joinpath("parameters.toml").open("rb") as fh:
        params = tomllib.load(fh)

    # Most historical TOML keys omit hyphens (b97d), while newer labels can
    # retain them (skala-1.0). Prefer the literal lowercase key, then the
    # legacy hyphen-stripped spelling.
    table = params.get("parameter", {})
    key = functional.lower()
    methods = table.get(key) or table.get(key.replace("-", ""), {})
    entry = methods.get("d3", {}).get("bj")
    if entry is None:
        return None

    # Canonical D3-BJ has s6 = 1.0 unless overridden (double hybrids).
    # s9 = 0.0: dftd3's parameters.toml carries no s9 field because these
    # damping sets were fit two-body-only. Callers wanting D3(BJ)-ATM set
    # s9 on the returned object (or use a composite recipe that does).
    return D3BJParams(
        s6=float(entry.get("s6", 1.0)),
        s8=float(entry["s8"]),
        a1=float(entry["a1"]),
        a2=float(entry["a2"]),
        s9=0.0,
    )


def compute_d3bj(
    mol: Molecule,
    params: Union[D3BJParams, str],
    *,
    with_gradient: bool = False,
    backend: str = "auto",
) -> DispersionResult:
    """Pairwise D3(BJ) dispersion energy (and optionally gradient).

    Parameters
    ----------
    mol
        :class:`vibeqc.Molecule`.
    params
        Either a :class:`D3BJParams` with explicit (s6, s8, a1, a2)
        values, or the lowercase name of a DFT functional
        (e.g. ``"pbe"``, ``"b3lyp"``). A string is resolved via
        :func:`d3bj_params_for` with the same ``backend``.
    with_gradient
        If True, the returned ``DispersionResult.gradient`` is a
        ``(n_atoms, 3)`` array of dE_disp / dr_A in Hartree / bohr.
        The derivative includes the explicit pair/triple geometry terms and
        the coordination-number dependence of the interpolated C6 values.
    backend
        ``"builtin"``, ``"dftd3"``, or ``"auto"`` (default). See the
        module docstring.
    """
    chosen = _resolve_backend(backend)
    if backend == "auto" and any(
        atom.Z > _d3_max_supported_Z() for atom in mol.atoms
    ):
        chosen = "dftd3"

    if isinstance(params, str):
        p = d3bj_params_for(params, backend=chosen)
        if p is None:
            raise ValueError(
                f"compute_d3bj: no D3-BJ parameters for functional "
                f"{params!r} (backend={chosen!r})"
            )
        params = p

    if chosen == "dftd3":
        return _compute_d3bj_dftd3(mol, params, with_gradient=with_gradient)
    return _compute_d3bj_builtin(mol, params, with_gradient)


def _compute_d3bj_dftd3(
    mol: Molecule,
    params: D3BJParams,
    *,
    with_gradient: bool,
) -> DispersionResult:
    """Bit-exact reference implementation via the ``dftd3`` package.

    Grimme's Fortran library takes positions in bohr and atomic numbers
    as integer arrays, matching vibe-qc's native conventions.
    """
    try:
        from dftd3.interface import DispersionModel, RationalDampingParam
    except ImportError as e:
        raise ImportError(
            "compute_d3bj(backend='dftd3') requires the optional "
            "dftd3 package. Install it with `pip install dftd3` into "
            "your vibe-qc venv, or `pip install -e '.[dispersion]'` "
            "from the repo checkout."
        ) from e

    atoms = mol.atoms
    n = len(atoms)
    numbers = np.array([a.Z for a in atoms], dtype=np.int32)
    positions = np.array([list(a.xyz) for a in atoms], dtype=float)
    if n == 0:
        return DispersionResult()

    model = DispersionModel(numbers, positions)
    # s9 MUST be passed explicitly: dftd3's RationalDampingParam defaults
    # s9=1.0, so omitting it silently switches the three-body ATM term on
    # for every functional -- and only on this backend.
    bj = RationalDampingParam(
        s6=params.s6, s8=params.s8, a1=params.a1, a2=params.a2,
        s9=params.s9,
    )
    res = model.get_dispersion(bj, grad=with_gradient)

    # Our DispersionResult is a pybind11-bound C++ struct that pybind11
    # lets us construct positionally from defaults and then set fields
    # on -- but both fields are readonly. Work around by building via
    # D3BJParams-style positional construction would require bindings
    # changes; instead we use an escape hatch: run the *builtin* call
    # with zeroed params to get a fresh DispersionResult whose fields
    # we then mutate... except they're def_readonly. So: declare the
    # dftd3 backend's return as a small shim that quacks like
    # DispersionResult for the public API.
    out = _DispersionResultShim(
        energy=float(res["energy"]),
        gradient=(np.asarray(res["gradient"], dtype=float)
                  if with_gradient else np.zeros((0, 3))),
    )
    return out  # type: ignore[return-value]


@dataclass
class _DispersionResultShim:
    """Python shim quacking like the pybind11 DispersionResult.

    We can't construct DispersionResult from Python (its fields are
    def_readonly), so the dftd3 backend returns this dataclass with
    the same attribute names. Callers treat the return opaquely via
    ``.energy`` and ``.gradient`` so the shim is indistinguishable
    from the native type in practice.
    """
    energy: float
    gradient: np.ndarray
