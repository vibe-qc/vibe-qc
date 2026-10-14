"""GFN2-xTB model for molecules.

.. warning::

   **Experimental -- not quantitative.** This GFN2-xTB implementation is
   still gated and does *not* yet claim reference ``xtb`` parity. The
   molecular path includes shell-resolved SCC, AES dipole/quadrupole,
   GAM3 third-order on-site, pairwise repulsion, and native D4 dispersion
   (post-SCF). This is a native D4-style approximation, not the
   self-consistent D4 functional of Bannwarth, Ehlert & Grimme (JCTC
   2019). It uses EEQ charges and independently generated reference
   data for H, He, B, C, N, O, F, Ne, and Cu, and explicitly warns/returns
   zero outside that set. Upstream GFN2 uses SCC charges and their
   dispersion potential, distinct reference/CN conventions, and
   neutral-charge C6 for ATM. Selecting faithful AES does not change
   this D4 policy or establish total-energy parity. Remaining
   production gates: periodic AES image-cell multipole terms, periodic
   molecular-limit parity, and the full external-parity matrix.  See
   ``tests/test_gfn2_xtb.py`` for validated invariants and strict
   reference targets.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Optional
import warnings

import numpy as np

from vibeqc import Molecule
from vibeqc._vibeqc_core import Atom as _Atom
from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
from vibeqc.semiempirical.model import SemiempiricalModel
from vibeqc.semiempirical.routes import SemiempiricalRoutePlan


class GFN2ExperimentalWarning(UserWarning):
    """Emitted when the experimental GFN2-xTB model is evaluated."""


class GFN2D4UnsupportedWarning(UserWarning):
    """Emitted when GFN2's post-SCF native D4 term has no reference data."""


_GFN2_EXPERIMENTAL_MESSAGE = (
    "GFN2-xTB is EXPERIMENTAL and not quantitative: reference xtb parity "
    "is not closed yet (all-electron vs valence-only convention, periodic "
    "faithful AES electrostatics, and external parity matrix remain production "
    "gates), and "
    "the post-SCF D4-style term uses EEQ charges and independently "
    "generated CPKS/PBE38 reference data for H, He, B, C, N, O, F, Ne, and Cu; "
    "it is not upstream GFN2 self-consistent dispersion. "
    "Results are size-consistent but must not be used for production. "
    "Construct the model with warn=False to silence this warning."
)
_GFN2_SCC_MIXERS = {
    "broyden": _se.SCCMixer.Broyden,
    "diis": _se.SCCMixer.DIIS,
    "simple": _se.SCCMixer.Simple,
}


def _gfn2_d4_refdata_path() -> Path:
    """Return the path to the shipped D4 reference dataset.

    Uses ``importlib.resources`` so the file is found regardless of whether
    vibe-qc runs from a source checkout, an editable install, or a wheel.
    """
    from importlib.resources import files as _res_files

    # Prefer the package-shipped copy (works for wheel/editable installs).
    pkg_path = _res_files("vibeqc.semiempirical.methods") / "d4_reference_data.json"
    if pkg_path.is_file():
        return pkg_path
    # Fallback: source-tree-relative for development checkouts where the file
    # sits at the repo root.
    tree_path = Path(__file__).resolve().parents[4] / "d4_reference_data.json"
    if tree_path.is_file():
        return tree_path
    raise FileNotFoundError(
        "d4_reference_data.json not found in package or source tree; "
        "the GFN2-xTB D4 post-SCF correction cannot be evaluated."
    )


def _gfn2_d4_refdata_bytes() -> bytes:
    """Return the raw bytes of the shipped D4 reference dataset.

    Used by the runner provenance path which needs a SHA-256 hash of the
    on-disk file without holding a persistent ``Path`` reference across
    subprocess boundaries.
    """
    return _gfn2_d4_refdata_path().read_bytes()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _gfn2_d4_reference_content_sha256(ref_data: object) -> str:
    """Hash the exact in-memory D4 reference records used by a calculation."""
    payload = json.dumps(
        ref_data.to_dict(),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return _sha256_bytes(payload)


@dataclass(frozen=True)
class _GFN2D4Execution:
    """Immutable selection of the Python D4 inputs for one GFN2 result."""

    reference_data: object
    functional: str
    parameters: object
    refdata_path: str
    refdata_sha256: str
    refdata_source_sha256: str


@dataclass(frozen=True)
class _FrozenD4ReferenceDataset:
    """Detached immutable D4 records consumed by one GFN2 execution."""

    _cns: Mapping[int, tuple[float, ...]]
    _qs: Mapping[int, tuple[float, ...]]
    _alpha_iw: Mapping[int, tuple[tuple[float, ...], ...]]
    _c6_self: Mapping[int, tuple[tuple[float, ...], ...]]
    _n_gaussian: Mapping[int, tuple[int, ...] | None]
    _c6_cross: Mapping[
        tuple[int, int], tuple[tuple[float, ...], ...]
    ]

    @classmethod
    def from_dataset(cls, source: object) -> "_FrozenD4ReferenceDataset":
        cns: dict[int, tuple[float, ...]] = {}
        qs: dict[int, tuple[float, ...]] = {}
        alpha_iw: dict[int, tuple[tuple[float, ...], ...]] = {}
        c6_self: dict[int, tuple[tuple[float, ...], ...]] = {}
        n_gaussian: dict[int, tuple[int, ...] | None] = {}
        for atomic_number, element in source.elements.items():
            atomic_number = int(atomic_number)
            cns[atomic_number] = tuple(float(value) for value in element.cns)
            qs[atomic_number] = tuple(float(value) for value in element.qs)
            n_gaussian[atomic_number] = (
                None if element.n_gaussian is None else tuple(element.n_gaussian)
            )
            alpha_iw[atomic_number] = tuple(
                tuple(float(value) for value in row)
                for row in np.asarray(element.alpha_iw)
            )
            c6_self[atomic_number] = tuple(
                tuple(float(value) for value in row)
                for row in np.asarray(element.c6_self)
            )
        c6_cross = {
            (int(pair[0]), int(pair[1])): tuple(
                tuple(float(value) for value in row)
                for row in np.asarray(matrix)
            )
            for pair, matrix in source.c6_cross.items()
        }
        return cls(
            _cns=MappingProxyType(cns),
            _qs=MappingProxyType(qs),
            _alpha_iw=MappingProxyType(alpha_iw),
            _c6_self=MappingProxyType(c6_self),
            _n_gaussian=MappingProxyType(n_gaussian),
            _c6_cross=MappingProxyType(c6_cross),
        )

    @property
    def supported_z(self) -> list[int]:
        return sorted(self._cns)

    def get_cns(self, atomic_number: int) -> tuple[float, ...]:
        return self._cns[int(atomic_number)]

    def get_qs(self, atomic_number: int) -> tuple[float, ...]:
        return self._qs[int(atomic_number)]

    def get_n_gaussian(self, atomic_number: int) -> tuple[int, ...] | None:
        return self._n_gaussian[int(atomic_number)]

    def get_c6_ref(
        self,
        z_a: int,
        ref_a: int,
        z_b: int,
        ref_b: int,
    ) -> float:
        if z_a == z_b:
            return float(self._c6_self[z_a][ref_a][ref_b])
        key = (z_a, z_b) if z_a <= z_b else (z_b, z_a)
        matrix = self._c6_cross[key]
        if z_a <= z_b:
            return float(matrix[ref_a][ref_b])
        return float(matrix[ref_b][ref_a])

    def to_dict(self) -> dict[str, object]:
        elements = {
            str(atomic_number): {
                "cns": list(self._cns[atomic_number]),
                "qs": list(self._qs[atomic_number]),
                "alpha_iw": [
                    list(row) for row in self._alpha_iw[atomic_number]
                ],
                "c6_self": [
                    list(row) for row in self._c6_self[atomic_number]
                ],
            }
            for atomic_number in self.supported_z
        }
        for atomic_number, counts in self._n_gaussian.items():
            if counts is not None:
                elements[str(atomic_number)]["n_gaussian"] = list(counts)
        c6_cross = {
            f"{z_a},{z_b}": [list(row) for row in matrix]
            for (z_a, z_b), matrix in sorted(self._c6_cross.items())
        }
        return {"elements": elements, "c6_cross": c6_cross}


def _gfn2_d4_execution_snapshot() -> _GFN2D4Execution:
    """Freeze the D4 table row and reference data before native execution."""
    from vibeqc.dispersion_d4_parameters import get_d4_params

    source = _load_gfn2_d4_reference_data()
    ref_data = _FrozenD4ReferenceDataset.from_dataset(source)
    source_sha256 = str(
        getattr(source, "_vibeqc_source_sha256", "unavailable")
    )
    return _GFN2D4Execution(
        reference_data=ref_data,
        functional="gfn2xtb",
        parameters=get_d4_params("gfn2xtb"),
        refdata_path="vibeqc/semiempirical/methods/d4_reference_data.json",
        refdata_sha256=_gfn2_d4_reference_content_sha256(ref_data),
        refdata_source_sha256=source_sha256,
    )


def _portable_gfn2_cache_path(cache_path: Path) -> str:
    try:
        relative = cache_path.relative_to(Path.home())
    except ValueError:
        return f"<gfn2-parameter-cache>/{cache_path.name}"
    return f"~/{relative.as_posix()}"


def _gfn2_cache_lineage_snapshot(params: object) -> dict[str, str]:
    """Return only cache lineage bound to this exact parameter builder."""
    from vibeqc.semiempirical.methods import gfn2_params as _gfn2_params

    snapshot = {
        "cache_path": "<unbound-gfn2-parameter-cache>",
        "cache_sha256": "unavailable",
        "cache_source_sha256": "unavailable",
        "cache_source_url": "unavailable",
    }
    lineage = _gfn2_params.gfn2_parameter_lineage(params)
    if lineage is None:
        return snapshot
    snapshot.update(
        {
            "cache_path": _portable_gfn2_cache_path(
                Path(lineage["cache_path"])
            ),
            "cache_sha256": str(lineage["cache_sha256"]),
            "cache_source_sha256": str(lineage["cache_source_sha256"]),
            "cache_source_url": str(lineage["cache_source_url"]),
        }
    )
    return snapshot


def _gfn2_nonconvergence_message(mol: "Molecule", result: object) -> str:
    elements = ", ".join(str(int(atom.Z)) for atom in mol.atoms)
    n_iter = int(getattr(result, "n_iter", 0))
    return (
        f"GFN2-xTB SCC did not converge after {n_iter} iterations "
        f"(charge={int(mol.charge)}, multiplicity={int(mol.multiplicity)}, "
        f"elements=[{elements}]); refusing to apply post-SCF D4 or return "
        "a total energy from a nonstationary SCC density."
    )


def _reported_scc_energy(result: object) -> float:
    """The SCC total the molecular GFN2 route reports and differentiates.

    A GFN2 SCC that ran at a finite electronic temperature, whether because
    the caller asked for one or because the auto-stabilisation ladder
    accepted its finite-temperature rung (vibe-qc#3, #247), is stationary in
    the Mermin free energy ``A = E - T*S`` (Mermin, Phys. Rev. 137, A1441
    (1965); Bannwarth, Ehlert & Grimme, JCTC 2019, Eqs. 8 and 10), not in the
    internal energy ``E``. The shell-AES analytic gradient differentiates ``A``
    through the converged fractional occupations, so ``A`` is what the model
    pairs with that gradient (vibe-qc#343).  At ``T = 0`` the two coincide and
    the historical value is returned unchanged.
    """
    temperature = float(getattr(result, "smearing_temperature", 0.0) or 0.0)
    if temperature > 0.0:
        return float(result.free_energy)
    return float(result.energy)


def _gfn2_molecule_key(mol: "Molecule"):
    return (
        int(mol.charge),
        int(mol.multiplicity),
        tuple(
            (
                int(atom.Z),
                float(atom.xyz[0]),
                float(atom.xyz[1]),
                float(atom.xyz[2]),
            )
            for atom in mol.atoms
        ),
    )


def _freeze_gfn2_scc_options(source: object):
    """Copy every native SCC control into a private execution object."""
    frozen = _xtb.XTBSccOptions()
    frozen.max_iter = int(source.max_iter)
    frozen.conv_tol_charge = float(source.conv_tol_charge)
    frozen.charge_mixing = float(source.charge_mixing)
    frozen.scc_mixer = source.scc_mixer
    frozen.mixer_memory = int(source.mixer_memory)
    frozen.mixer_damping = float(source.mixer_damping)
    # The property setter marks finite temperature as explicit, while the
    # explicit flag itself is intentionally read-only in Python. An implicit
    # option object can therefore only carry the native default temperature.
    source_temperature = float(source.electronic_temperature)
    if bool(source.electronic_temperature_explicit):
        frozen.electronic_temperature = source_temperature
    elif source_temperature != float(frozen.electronic_temperature):
        raise RuntimeError(
            "GFN2-xTB received an implicit electronic temperature that "
            "differs from the native default"
        )
    frozen.auto_stabilize = bool(source.auto_stabilize)
    frozen.aes_faithful = bool(source.aes_faithful)
    frozen.aes_damping = float(source.aes_damping)
    return frozen


def _gfn2_scc_options_snapshot(options: object) -> dict[str, object]:
    mixer_name = getattr(options.scc_mixer, "name", None)
    if mixer_name is None:
        mixer_name = str(options.scc_mixer).rsplit(".", 1)[-1]
    return {
        "max_iter": int(options.max_iter),
        "conv_tol_charge": float(options.conv_tol_charge),
        "charge_mixing": float(options.charge_mixing),
        "scc_mixer": str(mixer_name).lower(),
        "mixer_memory": int(options.mixer_memory),
        "mixer_damping": float(options.mixer_damping),
        "electronic_temperature": float(options.electronic_temperature),
        "electronic_temperature_explicit": bool(
            options.electronic_temperature_explicit
        ),
        "auto_stabilize": bool(options.auto_stabilize),
        "aes_faithful": bool(options.aes_faithful),
        "aes_damping": float(options.aes_damping),
    }


@lru_cache(maxsize=1)
def _load_gfn2_d4_reference_data():
    """Load GFN2-D4 data and bind its source hash to the parsed object."""
    from vibeqc.dispersion_d4_reference_data import D4ReferenceDataset

    source_bytes = _gfn2_d4_refdata_path().read_bytes()
    source_payload = json.loads(source_bytes.decode("utf-8"))
    dataset = D4ReferenceDataset.from_dict(source_payload)
    dataset._vibeqc_source_sha256 = _sha256_bytes(source_bytes)
    return dataset


def _gfn2_d4_reference_support(
    mol: "Molecule",
    ref_data: object | None = None,
):
    if ref_data is None:
        ref_data = _load_gfn2_d4_reference_data()
    supported = set(ref_data.supported_z)
    unsupported = sorted(
        {int(atom.Z) for atom in mol.atoms if atom.Z not in supported}
    )
    return ref_data, supported, unsupported


def _gfn2_d4_unsupported_message(
    supported: set[int],
    unsupported: list[int],
) -> str:
    return (
        "GFN2-xTB post-SCF native D4 has reference data only for "
        f"Z={sorted(supported)}; returning zero D4 for unsupported "
        f"Z={unsupported}."
    )


def _validate_gfn2_d4_execution(execution: _GFN2D4Execution) -> None:
    """Fail closed if captured D4 inputs are malformed or later mutated."""
    try:
        damping_values = tuple(
            float(getattr(execution.parameters, field))
            for field in ("s6", "s8", "a1", "a2", "s9")
        )
        damping_doi = execution.parameters.doi
    except (AttributeError, TypeError, ValueError):
        raise RuntimeError(
            "GFN2-xTB captured malformed D4 damping parameters"
        ) from None
    s6, s8, a1, a2, s9 = damping_values
    if (
        not all(np.isfinite(value) for value in damping_values)
        or s6 < 0.0
        or s8 < 0.0
        or s9 < 0.0
        or a1 < 0.0
        or a2 <= 0.0
        or not isinstance(damping_doi, str)
        or not damping_doi
    ):
        raise RuntimeError(
            "GFN2-xTB captured nonphysical D4 damping parameters"
        )
    if (
        _gfn2_d4_reference_content_sha256(execution.reference_data)
        != execution.refdata_sha256
    ):
        raise RuntimeError(
            "GFN2-xTB D4 reference data changed after the immutable "
            "execution snapshot was captured"
        )


def _compute_gfn2_d4(
    mol: "Molecule",
    params: "_xtb.GFN2ParameterSet",
    *,
    execution: _GFN2D4Execution | None = None,
) -> float:
    """Post-SCF GFN2-D4-style dispersion energy using the native D4 backend.

    Uses vibe-qc's native D4 model (Phase D4b) with EEQ partial charges
    and coordination numbers to compute charge-dependent C6 coefficients
    from the pre-generated reference dataset. The EEQ charge constraint
    is the molecule's total charge, including for displaced geometries.
    GFN2-specific Becke-Johnson
    damping parameters (s8/a1/a2) are registered in the D4 parameter
    database under the key ``"gfn2xtb"``.

    This is the shipped post-SCF correction.  Elements outside the native
    reference dataset are treated explicitly below.

    This differs from upstream GFN2 D4: it does not add a dispersion
    potential to SCC, and uses native EEQ/reference/CN conventions for
    both two-body and ATM C6. Agreement of selected native C6 values
    within a few percent does not establish GFN2 total-energy parity.
    Faithful AES selects only the AES functional, not a D4 functional.
    """
    from vibeqc.dispersion_d4_model import compute_d4_energy_total

    if execution is None:
        execution = _gfn2_d4_execution_snapshot()
    _validate_gfn2_d4_execution(execution)
    ref_data, supported, unsupported = _gfn2_d4_reference_support(
        mol,
        execution.reference_data,
    )
    if unsupported:
        warnings.warn(
            _gfn2_d4_unsupported_message(supported, unsupported),
            category=GFN2D4UnsupportedWarning,
            stacklevel=2,
        )
        return 0.0

    return compute_d4_energy_total(
        mol,
        ref_data,
        execution.functional,
        total_charge=float(mol.charge),
        atm=True,
        _damping_parameters=execution.parameters,
    )


def _displaced_molecule(
    mol: "Molecule",
    atom_idx: int,
    coord_idx: int,
    delta: float,
) -> "Molecule":
    atoms = list(mol.atoms)
    atom = atoms[atom_idx]
    xyz = list(atom.xyz)
    xyz[coord_idx] += delta
    atoms[atom_idx] = _Atom(int(atom.Z), xyz)
    return Molecule(atoms, int(mol.charge), int(mol.multiplicity))


def _compute_gfn2_d4_gradient_fd(
    mol: "Molecule",
    params: "_xtb.GFN2ParameterSet",
    *,
    h: float = 0.001,
    execution: _GFN2D4Execution | None = None,
) -> np.ndarray:
    """Finite-difference gradient of the cheap post-SCF D4 correction only."""
    from vibeqc.dispersion_d4_model import compute_d4_energy_total

    if execution is None:
        execution = _gfn2_d4_execution_snapshot()
    _validate_gfn2_d4_execution(execution)
    ref_data, supported, unsupported = _gfn2_d4_reference_support(
        mol,
        execution.reference_data,
    )
    natom = len(mol.atoms)
    if unsupported:
        warnings.warn(
            _gfn2_d4_unsupported_message(supported, unsupported),
            category=GFN2D4UnsupportedWarning,
            stacklevel=2,
        )
        return np.zeros((natom, 3), dtype=float)

    grad = np.zeros((natom, 3), dtype=float)
    for atom_idx in range(natom):
        for coord_idx in range(3):
            ep = compute_d4_energy_total(
                _displaced_molecule(mol, atom_idx, coord_idx, h),
                ref_data,
                execution.functional,
                total_charge=float(mol.charge),
                atm=True,
                _damping_parameters=execution.parameters,
            )
            em = compute_d4_energy_total(
                _displaced_molecule(mol, atom_idx, coord_idx, -h),
                ref_data,
                execution.functional,
                total_charge=float(mol.charge),
                atm=True,
                _damping_parameters=execution.parameters,
            )
            grad[atom_idx, coord_idx] = (float(ep) - float(em)) / (2.0 * h)
    return grad


class GFN2Model(SemiempiricalModel):
    """GFN2-xTB tight-binding model (experimental -- see module docstring).

    Parameters
    ----------
    mol : Molecule
    params : GFN2ParameterSet or None
    charge_mixing : float (default 0.1)
    max_iter : int (default 3600)
        Total SCC iteration budget across the primary solve and any automatic
        stabilization retry. Explicit smaller values are honored as hard caps.
    scc_mixer : str (default "simple")
        SCC charge mixer: "simple", "diis", or "broyden".  The default
        "simple" runs the two-phase polyalgorithm (damped simple mixing
        with stall-adaptive step halving, then a guarded DIIS handoff;
        see the GFN2 section of the semiempirical user guide).  An
        explicit "diis"/"broyden" request bypasses the polyalgorithm
        and runs the unguarded accelerator from the neutral guess.
    mixer_memory : int (default 6)
        DIIS subspace size or Broyden history length.
    mixer_damping : float (default 0.0)
        DIIS damping or Broyden step damping.
    warn : bool (default True)
        Emit a :class:`GFN2ExperimentalWarning` on each energy evaluation.
    solver : str (default ``"dense"``)
        Eigensolver for the SCF step.  Currently a reserved keyword;
        the GFN2-xTB C++ backend always uses dense diagonalisation.

    Notes
    -----
    ``parameter_identity`` and ``parameter_sha256`` identify the native SCC
    parameter set. The post-SCF D4 row and reference dataset are separate
    execution inputs and are pinned in ``parameter_provenance``.
    """

    def __init__(
        self,
        mol: Molecule,
        params: Optional[_xtb.GFN2ParameterSet] = None,
        *,
        charge_mixing: float = 0.1,
        max_iter: int = 3600,
        scc_mixer: str = "simple",
        mixer_memory: int = 6,
        mixer_damping: float = 0.0,
        warn: bool = True,
        solver: str = "dense",
    ):
        super().__init__(mol, solver=solver)
        self._route_plan = SemiempiricalRoutePlan.from_request(
            "gfn2_xtb",
            boundary="molecule",
            charge=int(mol.charge),
            multiplicity=int(mol.multiplicity),
        )
        if params is None:
            from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

            params = load_gfn2_params()
        mixer_key = str(scc_mixer).strip().lower()
        if mixer_key not in _GFN2_SCC_MIXERS:
            known = ", ".join(sorted(_GFN2_SCC_MIXERS))
            raise ValueError(
                f"GFN2Model scc_mixer must be one of {known}; got "
                f"{scc_mixer!r}."
            )
        self._params = params
        self._charge_mixing = charge_mixing
        self._max_iter = max_iter
        self._scc_mixer = mixer_key
        self._mixer_memory = mixer_memory
        self._mixer_damping = mixer_damping
        self._warn = warn
        self._n_iter = 0
        self._converged = False
        self._last_result = None
        self._last_result_key = None
        self._last_parameter_snapshot = None
        self._last_d4_execution = None

    def _make_scc_options(self):
        opts = _xtb.XTBSccOptions()
        opts.charge_mixing = self._charge_mixing
        opts.max_iter = self._max_iter
        opts.scc_mixer = _GFN2_SCC_MIXERS[self._scc_mixer]
        opts.mixer_memory = self._mixer_memory
        opts.mixer_damping = self._mixer_damping
        return opts

    def _record_scc_result(
        self,
        mol: Molecule,
        result: object,
        parameter_snapshot: dict[str, object] | None = None,
        d4_execution: _GFN2D4Execution | None = None,
    ):
        result_sha256 = getattr(result, "parameter_sha256", None)
        result_identity = getattr(result, "parameter_identity", None)
        if (result_sha256 is None) != (result_identity is None):
            raise RuntimeError(
                "GFN2-xTB native result returned an incomplete parameter "
                "identity"
            )
        if result_sha256 is not None:
            result_sha256 = str(result_sha256)
            result_identity = str(result_identity)

        frozen_snapshot = None
        if (
            parameter_snapshot is not None
            and result_sha256 is not None
            and result_identity is not None
        ):
            captured_sha256 = str(parameter_snapshot.get("parameter_sha256"))
            if captured_sha256 != result_sha256:
                raise RuntimeError(
                    "GFN2-xTB native result parameter identity disagrees with "
                    "the immutable pre-execution parameter snapshot"
                )
            frozen_snapshot = dict(parameter_snapshot)
            # The native result is the authority for what the released-GIL
            # calculation consumed. Never reconstruct these fields from the
            # caller-owned mutable builder after execution.
            frozen_snapshot["parameter_sha256"] = result_sha256
            frozen_snapshot["parameter_identity"] = result_identity

        # Publish the new cache entry only after all identity checks succeed;
        # a rejected concurrent mutation must not leave a reusable partial
        # result behind.
        self._n_iter = int(result.n_iter)
        self._converged = bool(result.converged)
        self._last_result = result
        self._last_result_key = (_gfn2_molecule_key(mol), result_sha256)
        self._last_parameter_snapshot = frozen_snapshot
        self._last_d4_execution = d4_execution
        return result

    def _cached_scc_result_for(self, mol: Molecule):
        if self._last_result is None:
            return None
        result_sha256 = getattr(self._last_result, "parameter_sha256", None)
        if result_sha256 is None:
            # Preserve the model-like test-double contract. Production native
            # results always carry a snapshot hash.
            expected_key = (_gfn2_molecule_key(mol), None)
        else:
            result_sha256 = str(result_sha256)
            current_sha256 = str(self._params.content_sha256())
            if current_sha256 != result_sha256:
                return None
            expected_key = (_gfn2_molecule_key(mol), result_sha256)
        if self._last_result_key != expected_key:
            return None
        self._n_iter = int(self._last_result.n_iter)
        self._converged = bool(self._last_result.converged)
        return self._last_result

    def _run_scc_at(self, mol: Molecule):
        opts = _freeze_gfn2_scc_options(self._make_scc_options())
        option_snapshot = _gfn2_scc_options_snapshot(opts)
        d4_execution = _gfn2_d4_execution_snapshot()
        cache_lineage = _gfn2_cache_lineage_snapshot(self._params)
        metadata = self._params.metadata()
        d4_parameters = d4_execution.parameters
        parameter_snapshot: dict[str, object] = {
            "parameter_sha256": str(metadata.parameter_hash),
            "version": str(metadata.version),
            "origin": str(metadata.origin),
            "license": str(metadata.license),
            "doi": str(metadata.doi_or_url),
            "n_elements": int(metadata.n_elements),
            "d4_functional": d4_execution.functional,
            "d4_s6": float(d4_parameters.s6),
            "d4_s8": float(d4_parameters.s8),
            "d4_s9": float(d4_parameters.s9),
            "d4_a1": float(d4_parameters.a1),
            "d4_a2": float(d4_parameters.a2),
            "d4_doi": str(d4_parameters.doi or "unavailable"),
            "d4_refdata_path": d4_execution.refdata_path,
            "d4_refdata_sha256": d4_execution.refdata_sha256,
            "d4_refdata_source_sha256": d4_execution.refdata_source_sha256,
            **cache_lineage,
            **option_snapshot,
        }
        return self._record_scc_result(
            mol,
            _xtb.run_gfn2_xtb(mol, self._params, opts),
            parameter_snapshot,
            d4_execution,
        )

    @property
    def params(self) -> _xtb.GFN2ParameterSet:
        return self._params

    @property
    def parameter_identity(self) -> Optional[str]:
        """Native SCC parameter identity consumed by the last execution."""
        identity = getattr(self._last_result, "parameter_identity", None)
        return None if identity is None else str(identity)

    @property
    def parameter_sha256(self) -> Optional[str]:
        """Canonical native SCC parameter hash bound to the last result."""
        sha256 = getattr(self._last_result, "parameter_sha256", None)
        return None if sha256 is None else str(sha256)

    @property
    def n_iter(self) -> int:
        """SCC iterations taken by the most recent energy evaluation."""
        return self._n_iter

    @property
    def converged(self) -> bool:
        """Whether the most recent SCC evaluation converged."""
        return self._converged

    def energy(self) -> float:
        if self._warn:
            warnings.warn(
                _GFN2_EXPERIMENTAL_MESSAGE,
                category=GFN2ExperimentalWarning,
                stacklevel=2,
            )
        return self._energy_at(self._mol)

    def _energy_at(self, mol: Molecule) -> float:
        result = self._cached_scc_result_for(mol)
        if result is None:
            result = self._run_scc_at(mol)

        if not self._converged:
            raise RuntimeError(_gfn2_nonconvergence_message(mol, result))

        # GFN2-D4 dispersion: post-SCF correction (Bannwarth, Ehlert &
        # Grimme, JCTC 2019).  D4 is charge-dependent via EEQ charges
        # computed internally by compute_d4_energy_total.  By native
        # implementation policy, this differs from upstream self-consistent
        # GFN2 dispersion; faithful AES does not change the D4 functional.
        e_d4 = _compute_gfn2_d4(
            mol,
            self._params,
            execution=self._last_d4_execution,
        )

        # At T > 0 the reported total is the Mermin free energy of the SCC
        # plus D4: the ensemble the analytic gradient differentiates (#343).
        return float(_reported_scc_energy(result) + e_d4)

    @property
    def electronic_temperature(self) -> float:
        """Electronic temperature (Ha) the most recent SCC actually ran at.

        0.0 for an exact zero-temperature Aufbau solution.  A positive value
        means the reported energy is the Mermin free energy ``A = E - T*S``
        and the gradient is its derivative; the auto-stabilisation ladder can
        select such a rung without the caller asking for smearing (#247).
        """
        return float(
            getattr(self._last_result, "smearing_temperature", 0.0) or 0.0
        )

    @property
    def entropy(self) -> float:
        """Electronic entropy ``S`` (in k_B) of the most recent SCC; 0 at T = 0."""
        return float(getattr(self._last_result, "entropy", 0.0) or 0.0)

    def gradient(self) -> np.ndarray:
        """Return the D4-corrected GFN2 gradient without finite-differencing SCC.

        The native term is the analytic derivative of the same total
        :meth:`energy` reports: the internal energy at ``T = 0`` and the
        Mermin free energy ``A = E - T*S`` when the SCC ran smeared, built
        from the converged (possibly fractional) occupations (#343).
        Dispatches to the derivative of the executed shell or faithful AES
        functional. Results without that identity raise ValueError (#427).
        """
        if self._warn:
            warnings.warn(
                _GFN2_EXPERIMENTAL_MESSAGE,
                category=GFN2ExperimentalWarning,
                stacklevel=2,
            )
        result = self._cached_scc_result_for(self._mol)
        if result is None:
            result = self._run_scc_at(self._mol)

        if not self._converged:
            raise RuntimeError(_gfn2_nonconvergence_message(self._mol, result))

        # Older native results do not identify their executed functional.
        # Never infer it from current options or silently assume shell AES.
        functional = getattr(result, "aes_functional", "unknown")
        if functional not in ("shell", "faithful"):
            raise ValueError(
                "Molecular GFN2 gradient requires a known executed AES functional; "
                "rerun the SCC with a current native build."
            )

        scc_gradient = np.asarray(
            _se.compute_gfn2_gradient(self._mol, result, self._params),
            dtype=float,
        )
        return scc_gradient + _compute_gfn2_d4_gradient_fd(
            self._mol,
            self._params,
            execution=self._last_d4_execution,
        )
