"""D4 dispersion correction (Caldeweyher, Bannwarth, Grimme,
*J. Chem. Phys.* **150**, 154122 (2019)).

Where D3(BJ) reads a fixed C6 reference table and applies
Becke-Johnson damping, D4 introduces an **electronegativity-
equilibrated charge-dependent C6** plus an Axilrod-Teller-Muto
three-body correction. The functional-form upgrades push D4 closer
to ab initio reference dispersion energies on the GMTKN benchmark
suites -- by ~30 % over D3(BJ) on average.

vibe-qc supports two backends:

* **dftd4** (default) -- the reference Fortran implementation
  (``dftd4`` PyPI package). Requires the optional dependency.
  Verified bit-exact to the upstream package (audit 2026-05-31).
* **native** -- vibe-qc's own MPL-2.0 implementation (Phase D4b).
  Requires a pre-generated reference dataset JSON file. The shipped
  ``d4_reference_data.json`` applies the per-atom Eq.-6 extraction of
  Caldeweyher et al. (2019) over a **correlated CPKS (PBE38/TD-DFT)**
  reference polarizability, and the r4r2 / C8 table is dftd4-consistent,
  so native pairwise C6 agree with the dftd4 backend to within a few
  percent and the CH4-dimer D4 energy to <0.05 kcal/mol -- the
  handover's production parity bars. **Supported elements: H-Ne** (the
  Phase-D4b reference catalogue); elements outside it raise a clear
  error. ``backend="dftd4"`` remains the default for full
  periodic-table coverage. See ``handovers/HANDOVER_D4_NATIVE.md``.

Both backends are accessed through the same :func:`compute_d4` API.

Typical use sits alongside a double-hybrid run -- pass
``dispersion="d4"`` to :func:`vibeqc.run_b2plyp` or
:func:`vibeqc.run_dsd_pbep86` to get the published
``B2PLYP-D4`` / ``DSD-PBEP86-D4`` total energy. The standalone
:func:`compute_d4` is for arbitrary functionals (any name ``dftd4``
recognises -- B3LYP-D4, PBE0-D4, PW1PW-D4, etc.) and for users that
want to manage the SCF + dispersion composition themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ._vibeqc_core import Molecule
from .dispersion_d4_parameters import normalize_d4_key

__all__ = [
    "D4NativeExperimentalWarning",
    "D4Result",
    "compute_d4",
    "dftd4_available",
]


class D4NativeExperimentalWarning(UserWarning):
    """Back-compat warning category for the native D4 backend.

    **No longer emitted** (2026-06-26): the native backend is now
    parity-validated for its supported element set (H-Ne) -- correlated
    CPKS (PBE38) reference polarizabilities + the corrected r4r2 table
    bring pairwise C6 within a few percent and the CH4-dimer energy
    within <0.05 kcal/mol of dftd4. The class is retained so existing
    ``warnings.simplefilter("ignore", D4NativeExperimentalWarning)``
    filters and ``except``/``catch_warnings`` blocks keep importing.
    """


def _charge_scaling_options(functional: str) -> dict[str, float]:
    """Return non-default D4 charge-scaling parameters for a method."""
    if normalize_d4_key(functional) == "r2scan3c":
        # Grimme et al., JCP 154, 064103 (2021), Sec. II.B: r2SCAN-3c
        # refits the charge-scaling (beta, gamma) from (3, 2) to (2, 1).
        # dftd4 exposes those same two coefficients as ``ga`` and ``gc``.
        # On the sealed water dimer this gives -0.000399206537858684 Ha
        # versus ORCA 6.1.1's printed -0.000399198 Ha reference value.
        return {"ga": 2.0, "gc": 1.0}
    return {}


@dataclass
class D4Result:
    """Result of a :func:`compute_d4` call.

    Attributes
    ----------
    energy : float
        Dispersion energy contribution (Hartree). Negative for bound
        systems.
    gradient : np.ndarray, optional
        ``(n_atoms, 3)`` array of nuclear gradients of the D4 energy
        in Hartree/bohr. Only present when ``compute_d4`` was called
        with ``with_gradient=True``; otherwise ``None``.
    functional : str
        Lowercase functional name that was used to look up the D4
        damping parameters (e.g. ``"b2plyp"``, ``"dsd-pbep86"``).
    """

    energy: float
    functional: str
    gradient: Optional[np.ndarray] = None

    def __repr__(self) -> str:
        return f"D4Result(functional={self.functional!r}, energy={self.energy:.10f}" + (
            ", gradient=<...>)" if self.gradient is not None else ")"
        )


def dftd4_available() -> bool:
    """Return True if the optional ``dftd4`` backend can be imported.

    Used in tests that conditionally exercise the D4 path, and by the
    double-hybrid dispatchers when ``dispersion="d4"`` is requested.
    """
    try:
        import dftd4.interface  # noqa: F401

        return True
    except ImportError:
        return False


def compute_d4(
    mol: Molecule,
    functional: str,
    *,
    charge: float = 0.0,
    with_gradient: bool = False,
    backend: str = "dftd4",
    ref_data_path: Optional[str] = None,
) -> D4Result:
    """D4 dispersion correction (Caldeweyher et al. 2019).

    Parameters
    ----------
    mol
        :class:`vibeqc.Molecule`. Atomic positions in bohr.
    functional
        Lowercase functional name (``"b2plyp"``, ``"dsd-pbep86"``,
        ``"b3lyp"``, ``"pbe0"``, ``"pw1pw"``, ...). For the ``dftd4``
        backend, anything the reference library catalogs is accepted.
        For the ``native`` backend, the name is looked up in
        :mod:`vibeqc.dispersion_d4_parameters`.
    charge
        Total molecular charge. D4 includes a charge-dependent C6
        scaling (the main upgrade over D3). Default 0.0 (neutral).
    with_gradient
        When True, the returned :class:`D4Result` carries the
        ``(n_atoms, 3)`` gradient of E_disp in Hartree/bohr.
    backend
        ``"dftd4"`` (default, requires optional dependency) or
        ``"native"`` (Phase D4b, requires generated reference dataset).
        The native backend is parity-validated for **H-Ne** (correlated
        CPKS/PBE38 reference C6 within a few percent of dftd4, CH4-dimer
        energy <0.05 kcal/mol); elements outside its reference catalogue
        raise a clear error. ``"dftd4"`` stays the default for full
        periodic-table coverage.
    ref_data_path
        Path to a JSON reference dataset file for the native backend.
        If ``None``, looks for ``VIBEQC_D4_REFDATA`` env var, then
        ``d4_reference_data.json`` in the current directory.

    Returns
    -------
    D4Result

    Raises
    ------
    ImportError
        If ``backend="dftd4"`` and the optional ``dftd4`` package is
        not installed.
    FileNotFoundError
        If ``backend="native"`` and no reference dataset can be found.
    RuntimeError
        If ``dftd4`` does not recognise the requested ``functional``
        (re-raised from ``dftd4.interface`` with a clearer message).
    """
    backend = backend.strip().lower()

    # B3LYP flavor spellings (b3lyp5 / b3lyp/g / b3lypg) share the
    # single "b3lyp" damping set; the external dftd4 catalog only
    # knows the bare name, so normalise before dispatching.
    from .dispersion import _canonical_dispersion_name
    functional = _canonical_dispersion_name(functional)

    if backend == "dftd4":
        return _compute_d4_dftd4(
            mol, functional, charge=charge, with_gradient=with_gradient
        )
    elif backend == "native":
        return _compute_d4_native(
            mol, functional, with_gradient=with_gradient, ref_data_path=ref_data_path
        )
    else:
        raise ValueError(
            f"compute_d4: unknown backend {backend!r}. Use 'dftd4' or 'native'."
        )


def _compute_d4_dftd4(
    mol: Molecule,
    functional: str,
    *,
    charge: float = 0.0,
    with_gradient: bool = False,
) -> D4Result:
    """D4 via the dftd4 reference library."""
    try:
        from dftd4.interface import DampingParam, DispersionModel
    except ImportError as e:
        raise ImportError(
            "compute_d4: the optional 'dftd4' package is not installed.\n"
            "  Install with `pip install -e '.[dispersion]'` from the "
            "vibe-qc checkout, or `pip install dftd4` in your venv."
        ) from e

    numbers = np.array([atom.Z for atom in mol.atoms], dtype=np.int32)
    positions = np.array([atom.xyz for atom in mol.atoms], dtype=np.float64)

    model = DispersionModel(
        numbers=numbers,
        positions=positions,
        charge=charge,
        **_charge_scaling_options(functional),
    )
    try:
        params = DampingParam(method=functional.lower())
    except Exception as e:
        raise RuntimeError(
            f"compute_d4: dftd4 does not recognise functional "
            f"{functional!r}. Underlying error: {e}"
        ) from e

    result = model.get_dispersion(params, grad=with_gradient)
    energy = float(result["energy"])
    gradient = (
        np.asarray(result["gradient"], dtype=np.float64) if with_gradient else None
    )
    return D4Result(
        energy=energy,
        functional=functional.lower(),
        gradient=gradient,
    )


def _compute_d4_native(
    mol: Molecule,
    functional: str,
    *,
    with_gradient: bool = False,
    ref_data_path: Optional[str] = None,
) -> D4Result:
    """D4 via the vibe-qc native Phase D4b backend.

    Un-gated (2026-06-26): the reference dataset is built from the
    per-atom Eq.-6 extraction over a **correlated CPKS (PBE38/TD-DFT)**
    alpha(iw), and the r4r2 / C8 table was corrected, so native pairwise
    C6 agree with dftd4 to within a few percent and the CH4-dimer D4
    energy to <0.05 kcal/mol (handover Sec. 5 bars met). No experimental
    warning is emitted for the supported element set (H-Ne); elements
    outside the reference catalogue raise a clear error from the dataset
    lookup. ``backend="dftd4"`` remains the default for full
    periodic-table coverage.
    """
    from .dispersion_d4_model import compute_d4_energy_total, compute_d4_gradient
    from .dispersion_d4_reference_data import D4ReferenceDataset

    # Resolve reference dataset path.
    path = _resolve_refdata_path(ref_data_path)
    ref_data = D4ReferenceDataset.load_json(path)
    charge_scaling = _charge_scaling_options(functional)

    if with_gradient:
        energy, gradient, _dEdcn, _dEdq = compute_d4_gradient(
            mol,
            ref_data,
            functional,
            **charge_scaling,
        )
        return D4Result(
            energy=float(energy),
            functional=functional.lower(),
            gradient=gradient,
        )
    else:
        energy = compute_d4_energy_total(
            mol,
            ref_data,
            functional,
            atm=True,
            **charge_scaling,
        )
        return D4Result(
            energy=float(energy),
            functional=functional.lower(),
        )


def _resolve_refdata_path(explicit: Optional[str] = None) -> str:
    """Resolve the path to the D4 reference dataset JSON file."""
    import os

    if explicit and os.path.isfile(explicit):
        return explicit
    env = os.environ.get("VIBEQC_D4_REFDATA", "")
    if env and os.path.isfile(env):
        return env
    # Try the shipped package copy (works for wheel/editable installs).
    try:
        from importlib.resources import files as _res_files

        pkg_path = _res_files("vibeqc.semiempirical.methods") / "d4_reference_data.json"
        if pkg_path.is_file():
            return str(pkg_path)
    except Exception:
        pass
    # Try current directory (legacy path).
    default = "d4_reference_data.json"
    if os.path.isfile(default):
        return default
    raise FileNotFoundError(
        "compute_d4 native backend: no reference dataset found.\n"
        "  Generate one with:\n"
        "    from vibeqc.dispersion_d4_reference_data import "
        "generate_reference_dataset\n"
        "    ds = generate_reference_dataset('def2-svp')\n"
        "    ds.save_json('d4_reference_data.json')\n"
        "  Or set VIBEQC_D4_REFDATA=/path/to/dataset.json"
    )
