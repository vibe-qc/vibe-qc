"""ACEsuit MACE -- machine-learning interatomic potential interface.

MACE (https://github.com/ACEsuit/mace) is a higher-order *E(3)*-
equivariant message-passing neural-network interatomic potential. It
predicts energy / forces / stress on a DFT-fitted potential-energy
surface -- it does **not** solve the Schrödinger equation and builds no
wavefunction. vibe-qc drives MACE's pre-trained forward pass and
attributes it as an external pre-trained model: a maintainer-approved
extension of ``CLAUDE.md`` Sec.10 (vibe-qc computes nothing here; it
marshals geometry in and reads energy / forces out).

Energy-scale caveat
-------------------
:meth:`MACEModel.energy` returns the model's DFT-surface energy in
Hartree on a **model-specific reference scale** (each model subtracts
its own per-element atomic energies *E0*), **not** a vibe-qc total
electronic energy, and **not comparable across models**. For H2O the
materials model MACE-MPA-0 gives ~ -0.51 Ha (heavily reference-shifted)
while the organic MACE-OFF23 gives ~ -76.5 Ha (close to its wB97M
total); the ab-initio HF/DFT total is ~ -76 Ha. MACE energies are
meaningful for *relative* energetics (geometry optimization, reaction
energies at fixed composition + model, MD), not as absolute totals.

Licensing + the ASL gate
-------------------------
MACE *code* (``mace-torch``) is MIT. Foundation-model *weights* are
licensed separately and fetched on demand into the XDG cache (never
bundled). The per-model license lives in :mod:`vibeqc.mlip._mace_models`:
MP-0 / MPA-0 (materials) are MIT and ungated; OFF23 is ASL (academic,
**non-commercial**) and gated. Selecting OFF23 raises
:class:`PermissionError` unless the caller acknowledges the academic license
(``MLIPOptions(accept_academic_license=True)`` or ``VIBEQC_ACCEPT_ASL=1``).
OMAT, MATPES, MH, MDP, arbitrary URLs, and local weights are not registered
and fail closed. Registry and license checks fire *before* any torch import or
weight download.

The heavy ML stack (PyTorch, e3nn) is the optional ``[mace]`` extra and
is import-gated -- using a MIT model without it installed raises a clear,
actionable error. The supported ``[mace]`` extra targets Python <= 3.13
and ``mace-torch>=0.3.16,<0.4``. ACEsuit's planned 1.0 rewrite changes
APIs and checkpoint loading (https://github.com/ACEsuit/mace#project-status).
Mendoza Automated Crystal Engine is a separate DFT workflow application;
its release versions and GUI commands do not describe this MLIP backend.

OpenMP note (macOS)
-------------------
vibe-qc's C++ core (libxc / BLAS) and PyTorch each link their own OpenMP
runtime. On macOS, loading both in one process aborts with "OMP: Error
#15: ... libomp.dylib already initialized". Set ``KMP_DUPLICATE_LIB_OK=
TRUE`` to allow it. A duplicate runtime can still crash when both sides
create OpenMP worker pools, so the MACE path also caps OpenMP/BLAS-style
thread environment variables to one thread by default before importing
torch. The MACE forward pass is verified to produce identical energies
with the duplicate-runtime workaround set (M5). On macOS, vibe-qc
**auto-sets** this (once, with a ``RuntimeWarning``) before importing
torch -- see :func:`_maybe_set_openmp_workaround`; set
``VIBEQC_MACE_OPENMP_THREADS`` yourself to override the conservative
thread cap.
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, replace
from numbers import Integral, Real
from pathlib import Path
from typing import TYPE_CHECKING

from ._mace_models import MaceModelInfo, resolve_model
from .options import MLIPOptions

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


_OPENMP_WORKAROUND_DONE = False
_MACE_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
_MACE_THREAD_OVERRIDE_ENV = "VIBEQC_MACE_OPENMP_THREADS"


def mace_cache_root() -> Path:
    """Return the cache directory used by upstream MACE model loaders.

    MACE honors ``XDG_CACHE_HOME`` and otherwise uses ``~/.cache/mace``.
    vibe-qc never bundles model weights; a registered checkpoint is fetched
    on first use, preserving upstream cache names.
    """
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return root / "mace"


def _pbc_for_dimension(dim: int) -> tuple[bool, bool, bool]:
    """Map a ``PeriodicSystem`` dimension to ASE's leading-axis PBC mask."""
    dim = int(dim)
    if dim not in (1, 2, 3):
        raise ValueError(
            f"Periodic MACE supports dim=1, 2, or 3; got dim={dim}."
        )
    return tuple(axis < dim for axis in range(3))


def _mace_openmp_thread_cap() -> str:
    value = os.environ.get(_MACE_THREAD_OVERRIDE_ENV, "1").strip()
    if not value:
        return "1"
    try:
        n_threads = int(value)
    except ValueError:
        return "1"
    return str(max(1, n_threads))


def _maybe_set_openmp_workaround() -> None:
    """On macOS, prepare the process for MACE's torch-backed runtime.

    vibe-qc's C++ core (libxc / BLAS) and PyTorch each link their own
    OpenMP runtime; on macOS loading both aborts with "OMP Error #15"
    unless this is set. The MACE forward pass is verified to produce
    identical energies with it set (M5). Duplicate runtimes can still
    crash when both try to spin OpenMP pools, so the MACE path also caps
    thread-like environment variables before torch loads. It is officially
    "unsafe" (it permits a duplicate OpenMP runtime), so we warn -- but
    for MACE inference vibe-qc's core runs no concurrent OpenMP work, so
    the risk is low. Use ``VIBEQC_MACE_OPENMP_THREADS`` to opt into a
    different thread cap.
    """
    global _OPENMP_WORKAROUND_DONE
    if _OPENMP_WORKAROUND_DONE or sys.platform != "darwin":
        return
    _OPENMP_WORKAROUND_DONE = True
    changed = []
    if "KMP_DUPLICATE_LIB_OK" not in os.environ:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        changed.append("KMP_DUPLICATE_LIB_OK=TRUE")
    thread_cap = _mace_openmp_thread_cap()
    for name in _MACE_THREAD_ENV_VARS:
        if os.environ.get(name) != thread_cap:
            os.environ[name] = thread_cap
            changed.append(f"{name}={thread_cap}")
    if not changed:
        return
    import warnings

    warnings.warn(
        "vibe-qc set " + ", ".join(changed) + " for the MACE run: on macOS, "
        "vibe-qc's core and PyTorch each link an OpenMP runtime and can "
        "otherwise abort or segfault. The MACE forward pass is verified "
        "correct with the duplicate-runtime workaround. Set "
        "VIBEQC_MACE_OPENMP_THREADS before importing vibeqc to choose a "
        "different MACE thread cap.",
        RuntimeWarning,
        stacklevel=3,
    )


def _mace_installation_hint() -> str:
    hint = (
        "vibeqc.mlip.mace requires ACEsuit MACE (the mace-torch distribution, "
        "with PyTorch + e3nn). Install it with `pip install 'vibe-qc[mace]'` "
        "(or `pip install 'mace-torch>=0.3.16,<0.4'`) into your vibe-qc venv. "
        "Mendoza Automated Crystal Engine and other packages named mace "
        "do not provide this MLIP backend. If another package occupies the "
        "mace import namespace, use a separate environment for ACEsuit MACE."
    )
    if sys.version_info >= (3, 14):
        hint += (
            f"  NOTE: this interpreter is Python {sys.version_info.major}."
            f"{sys.version_info.minor}; vibe-qc enables its supported "
            "[mace] extra only on Python <= 3.13. Use a Python <= 3.13 "
            "environment for the supported MACE installation."
        )
    return hint


def _check_mace_api_series() -> None:
    """Reject incompatible installed API series before importing the ML stack.

    The dependency specification enforces the patch-version floor. Source
    trees without distribution metadata still go through calculator imports.
    """
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("mace-torch")
    except PackageNotFoundError:
        return
    if not installed.startswith("0.3."):
        raise ImportError(
            f"Unsupported ACEsuit mace-torch API series: {installed!r}. "
            "This wrapper uses the 0.3.x calculator and checkpoint API. "
            + _mace_installation_hint()
        )


def _require_backends():
    """Import the ACEsuit MACE runtime stack, or raise an installation error.

    Returns ``(mace_mp, mace_off, torch, Atoms, Bohr, Hartree)``.
    """
    _check_mace_api_series()
    _maybe_set_openmp_workaround()
    # Upstream mace-torch sets this env var itself in mace/__init__.py and
    # mace/calculators/mace.py; set it here first, explicitly, so the
    # override is visible in vibe-qc's own code together with its
    # justification: torch >= 2.6 defaults torch.load to weights_only=True,
    # but the cached MACE foundation-model files are legacy pickles of
    # custom classes. Verified 2026-08-14 against the two cached weight
    # digests (MPA-0 medium 75428afe..., OFF23 medium 4842c52a...):
    # torch.load(..., weights_only=True) raises UnpicklingError on both, so
    # weights_only=True is NOT safe and the override must stay. The
    # per-run UserWarning it produces ("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
    # ... forcing weights_only=False") comes from upstream's
    # un-parameterized torch.load calls (mace.calculators.mace.py:226 and
    # e3nn/o3/_wigner.py) and is expected noise for locally digest-verified
    # weights -- do not remove the override to silence it.
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    try:
        import torch
        import e3nn  # noqa: F401 - dependency check
        from ase import Atoms
        from ase.units import Bohr, Hartree
        from mace.calculators import MACECalculator  # noqa: F401 - dependency check
        from ._mace_checkpoint import load_mp as mace_mp, load_off as mace_off
    except ImportError as exc:
        raise ImportError(_mace_installation_hint()) from exc
    return mace_mp, mace_off, torch, Atoms, Bohr, Hartree


def _asl_gate_message(info: MaceModelInfo) -> str:
    """The error shown when an ASL (academic, non-commercial) model is
    selected without acknowledgment."""
    return (
        f"MACE model {info.key!r} ({info.domain}; trained on "
        f"{info.training_data}) is distributed under the Academic Software "
        f"License (ASL) -- free for ACADEMIC, NON-COMMERCIAL use only "
        f"(https://github.com/gabor1/ASL). vibe-qc will not fetch or run it "
        f"until you acknowledge this. If your use is academic and "
        f"non-commercial, set MLIPOptions(accept_academic_license=True) or "
        f"the env var VIBEQC_ACCEPT_ASL=1. For commercial use, pick a "
        f"MIT-licensed model instead (e.g. model='medium-mpa-0' -- "
        f"MACE-MPA-0, materials)."
    )


def enforce_academic_license(info: MaceModelInfo, options: MLIPOptions) -> None:
    """Raise :class:`PermissionError` if an ASL (academic, non-commercial)
    model is selected without acknowledgment; MIT models pass silently.

    Shared by :class:`MACEModel` (defense in depth) and ``run_job`` (which
    calls it early to fail fast, before any output / SCF machinery)."""
    if info.is_academic_only and not options.academic_license_acknowledged():
        raise PermissionError(_asl_gate_message(info))


def element_coverage_error(info: MaceModelInfo, molecule) -> str | None:
    """Return a clear error message if ``molecule`` contains an element the
    model does not cover, else None. Used by ``run_job`` (fail-fast) and
    :class:`MACEModel`."""
    bad = info.unsupported_elements(int(a.Z) for a in molecule.atoms)
    if not bad:
        return None
    if info.domain == "materials":
        return (
            f"MACE model {info.key!r} ({info.domain}, {info.elements} elements) "
            f"does not cover element(s) Z={bad}. No registered vibe-qc MACE "
            "model covers those elements; use a suitable electronic method "
            "or stop and review a new model's domain and license before "
            "extending the registry."
        )
    return (
        f"MACE model {info.key!r} ({info.domain}, {info.elements} elements) "
        f"does not cover element(s) Z={bad}. Use a materials model "
        f"(e.g. model='medium-mpa-0', 89 elements) for inorganic / metal "
        f"systems; the organic MACE-OFF23 covers only H,C,N,O,F,P,S,Cl,Br,I."
    )


def _validate_mlip_input(molecule, info: MaceModelInfo) -> None:
    """Fail-fast input checks before loading / running a MACE model: a hard
    error for out-of-domain elements, and a warning for charge / spin
    (which MACE -- a neutral learned potential with no electronic structure --
    ignores)."""
    import warnings

    if not list(getattr(molecule, "atoms", ()) or ()):
        raise ValueError(
            "MACE requires a structure with at least one atom (got an empty "
            "molecule). Build the Molecule with its atoms before running."
        )
    msg = element_coverage_error(info, molecule)
    if msg is not None:
        raise ValueError(msg)
    charge = getattr(molecule, "charge", 0) or 0
    mult = getattr(molecule, "multiplicity", 1) or 1
    if charge != 0 or mult != 1:
        parts = []
        if charge != 0:
            parts.append(f"total charge ({charge})")
        if mult != 1:
            parts.append(f"spin multiplicity ({mult})")
        warnings.warn(
            f"MACE ignores {' and '.join(parts)}: it is a neutral learned "
            f"interatomic potential with no electronic structure, so the "
            f"energy is identical for any charge / spin state.",
            RuntimeWarning,
            stacklevel=3,
        )


def _validate_runtime_options(options: MLIPOptions) -> None:
    digest = options.weight_sha256
    if digest is not None and (
        not isinstance(digest, str) or len(digest) != 64
        or any(c not in "0123456789abcdefABCDEF" for c in digest)
    ):
        raise ValueError("MACE weight_sha256 must be a 64-character hexadecimal digest")
    if options.dtype not in ("float32", "float64"):
        raise ValueError("MACE dtype must be 'float32' or 'float64'")
    if str(options.device).split(":", 1)[0] == "mps" and options.dtype != "float32":
        raise ValueError("MACE on MPS requires dtype='float32'; MPS cannot represent float64")


def _load_calculator(info, options, mace_mp, mace_off, torch):
    loader = mace_mp if info.loader == "mace_mp" else mace_off
    identity_options = {}
    if options.weight_sha256 is not None:
        identity_options["expected_sha256"] = options.weight_sha256
    if info.is_academic_only:
        identity_options["accept_academic_license"] = options.academic_license_acknowledged()
    if str(options.device).split(":", 1)[0] != "mps":
        return loader(model=info.loader_arg, device=options.device,
                      default_dtype=options.dtype, **identity_options)
    if not torch.backends.mps.is_available():
        raise RuntimeError("MACE requested MPS, but this PyTorch runtime has no available MPS device")
    # MACE's ordinary loader maps checkpoint tensors to the target device
    # before casting their dtype. A float64 checkpoint cannot even deserialize
    # on MPS. Keep the registered loader and license gate, but cast on CPU
    # before MACECalculator transfers the model to the requested device.
    from mace.calculators import MACECalculator

    model = loader(model=info.loader_arg, device="cpu",
                   default_dtype="float32", return_raw_model=True, **identity_options)
    identity = getattr(model, "_vibeqc_checkpoint_identity", None)
    calculator = MACECalculator(models=model.float(), device=options.device,
                                default_dtype="float32")
    calculator._vibeqc_checkpoint_identity = identity
    return calculator


def mace_calculator(options: MLIPOptions | None = None):
    """Return the bare MACE ASE ``Calculator`` (eV / Angstrom) for the
    selected model -- for ASE-native workflows (custom optimizers, MD,
    slab / adsorption studies). Applies the ASL license gate; needs no
    molecule. For vibe-qc's energy/gradient/stress wrapper (Hartree units)
    use :class:`MACEModel` instead."""
    if options is None:
        options = MLIPOptions()
    else:
        options = replace(options)
    info = resolve_model(options.model)
    enforce_academic_license(info, options)
    _validate_runtime_options(options)
    mace_mp, mace_off, torch, _Atoms, _Bohr, _Hartree = _require_backends()
    return _load_calculator(info, options, mace_mp, mace_off, torch)


def _checked_backend_result(results, name, shape, model_name):
    """Copy one required finite property before accepting a backend result."""
    import numpy as np

    try:
        value = np.array(results[name], dtype=float, copy=True)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"MACE model {model_name!r} returned missing or nonnumeric {name}"
        ) from exc
    if value.shape != shape or not np.all(np.isfinite(value)):
        raise ValueError(
            f"MACE model {model_name!r} returned invalid {name}: "
            f"expected finite values with shape {shape}, got shape {value.shape}"
        )
    return value


class MACEModel:
    """vibe-qc wrapper around a MACE foundation model.

    Exposes the same energy/gradient contract as
    :class:`vibeqc.semiempirical.model.SemiempiricalModel` so the runner
    can treat it uniformly: :meth:`energy` returns Hartree, :meth:`gradient`
    returns Hartree/bohr. The energy is MACE's reference-shifted DFT-surface
    energy (see module docstring), not a total electronic energy.

    Parameters
    ----------
    molecule
        A vibe-qc ``Molecule`` (positions in bohr).
    options
        An :class:`vibeqc.mlip.MLIPOptions` selecting the model, device,
        dtype, and (for ASL models) the academic-license acknowledgment.
        Defaults to the MIT MACE-MPA-0 on CPU in float64. Values are copied
        at construction, so later configuration edits cannot relabel this model.
    """

    def __init__(
        self,
        molecule,
        options: MLIPOptions | None = None,
        cell=None,
        *,
        pbc=None,
    ):
        if options is None:
            options = MLIPOptions()
        options = replace(options)
        self._options = options
        self._info = resolve_model(options.model)
        # ASL gate (CLAUDE.md Sec.1/Sec.10): academic, non-commercial models
        # require explicit acknowledgment; MIT models are ungated. Fires
        # before any torch/mace import or weight download.
        enforce_academic_license(self._info, options)
        _validate_runtime_options(options)
        self._set_structure(molecule, cell=cell, pbc=pbc)
        mace_mp, mace_off, torch, Atoms, Bohr, Hartree = _require_backends()
        self._Atoms = Atoms
        self._Bohr = float(Bohr)
        self._Hartree = float(Hartree)
        self._calc = _load_calculator(self._info, options, mace_mp, mace_off, torch)
        self._weight_identity = getattr(self._calc, "_vibeqc_checkpoint_identity", None)

    def _set_structure(self, molecule, *, cell=None, pbc=None) -> None:
        """Replace the evaluated structure while retaining the loaded model.

        This is private because a caller holding a one-shot ``MACEModel``
        should not mutate it behind an existing result. The reusable periodic
        evaluator below owns one model and calls this sequentially between
        scan points.
        """
        import numpy as np

        _validate_mlip_input(molecule, self._info)
        position_error = "MACE positions must be finite with shape (n_atoms, 3)"
        try:
            positions = np.asarray([atom.xyz for atom in molecule.atoms], dtype=float)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(position_error) from exc
        if positions.shape != (len(molecule.atoms), 3) or not np.all(np.isfinite(positions)):
            raise ValueError(position_error)

        if cell is None:
            if pbc is not None:
                raise ValueError("pbc requires a periodic MACE cell")
            values = None
        else:
            cell_error = "MACE cell must be finite with shape (3, 3)"
            try:
                cell = np.array(cell, dtype=float, copy=True)
            except (TypeError, ValueError) as exc:
                raise ValueError(cell_error) from exc
            if cell.shape != (3, 3) or not np.all(np.isfinite(cell)):
                raise ValueError(cell_error)
            if pbc is None:
                # Supplying only a cell retains the lower-level 3-D default.
                # Public periodic drivers pass the exact dimensionality.
                values = (True, True, True)
            else:
                pbc_error = "pbc must contain three boolean flags (or integer 0/1)"
                try:
                    values = tuple(pbc)
                except TypeError as exc:
                    raise ValueError(pbc_error) from exc
                if len(values) != 3 or any(
                    not isinstance(value, (bool, np.bool_, Integral)) or value not in (0, 1)
                    for value in values
                ):
                    raise ValueError(pbc_error)
                values = tuple(bool(value) for value in values)

        # Validate before replacing any state. A rejected scan point must
        # leave the previous structure and its cached result consistent.
        self._mol = molecule
        self._cell = cell
        self._pbc = values
        self._e_eV = None
        self._f_eVA = None
        self._stress_eVA3 = None

    def _ase_atoms(self):
        """Build an ASE ``Atoms`` (Angstrom) from the bohr-valued molecule;
        with a unit cell + PBC when this is a periodic system."""
        import numpy as np

        bohr = self._Bohr
        numbers = [atom.Z for atom in self._mol.atoms]
        positions = np.asarray([atom.xyz for atom in self._mol.atoms],
                               dtype=float) * bohr
        atoms = self._Atoms(numbers=numbers, positions=positions)
        if self._cell is not None:
            # self._cell is the vibe-qc PeriodicSystem lattice: COLUMNS are the
            # Cartesian lattice vectors (periodic.hpp:32). ASE's set_cell expects
            # them as ROWS, so transpose. This also keeps relax_periodic's
            # round-trip consistent: _ase_atoms transposes columns->rows on the way
            # in and atoms_to_periodic_system transposes rows->columns on the way
            # out, so the relaxed lattice comes back in the same (correct)
            # convention. (Both omitting the transpose cancelled for the matrix but
            # fed MACE the wrong geometry on non-orthogonal cells.)
            atoms.set_cell(np.asarray(self._cell, dtype=float).T * bohr)
            atoms.set_pbc(self._pbc)
        return atoms

    def _compute(self) -> None:
        if self._e_eV is None:
            atoms = self._ase_atoms()
            # One forward pass for every property this evaluation needs:
            # MACECalculator computes energy + forces + stress in a single
            # graph evaluation and serves later ASE getters from .results, so
            # requesting them together avoids two extra get_property()
            # round-trips per point (~250 us of ASE bookkeeping per point on
            # top of the ~tens-of-ms forward pass; MACE-REUSABLE-TIMING).
            properties = ["energy", "forces"]
            if self._cell is not None and self.periodic_dimension == 3:
                properties.append("stress")
            self._calc.calculate(atoms, properties=properties)
            results = self._calc.results

            def checked(name, shape):
                return _checked_backend_result(results, name, shape, self.model_name)

            energy = float(checked("energy", ()))
            forces = checked("forces", (len(self._mol.atoms), 3))
            stress = None
            if "stress" in properties:
                # MACECalculator stores stress as the Voigt 6-vector; ASE's
                # Atoms.get_stress(voigt=False) converts it back to the 3x3
                # tensor. Reading .results directly does the same conversion
                # with the same public helper (no constraints or momenta on
                # this internal Atoms object, so the values are identical).
                from ase.stress import voigt_6_to_full_3x3_stress

                stress = voigt_6_to_full_3x3_stress(
                    checked("stress", (6,))
                )
            # Publish the cache only after the entire result is valid. A
            # failed force/stress read must not make the next energy() call
            # succeed with a partially initialized result. Own the arrays:
            # a reused ASE calculator may overwrite its result buffers.
            self._f_eVA = forces
            self._stress_eVA3 = stress
            self._e_eV = energy

    def energy(self) -> float:
        """Total energy in Hartree (reference-shifted DFT-surface energy)."""
        self._compute()
        return self._e_eV / self._Hartree

    def gradient(self) -> np.ndarray:
        """Nuclear gradient dE/dR in Hartree/bohr, shape ``(n_atoms, 3)``."""
        import numpy as np

        self._compute()
        # ASE forces are eV/Angstrom; gradient = -force, converted to
        # Ha/bohr:  g[Ha/bohr] = -F[eV/A] * Bohr[A/bohr] / Hartree[eV/Ha].
        return -np.asarray(self._f_eVA) * self._Bohr / self._Hartree

    def stress(self) -> np.ndarray:
        """Stress tensor in Hartree/bohr^3, shape (3, 3) -- periodic only.

        Stress is available only for fully 3D-periodic systems.  For a true
        2D slab, ASE/MACE would divide by the arbitrary bookkeeping cell
        volume along the non-periodic direction, so vibe-qc fails closed
        instead of reporting that quantity as a physical slab stress.
        """
        import numpy as np

        if self._cell is None:
            raise ValueError(
                "stress() is only defined for a periodic MACEModel "
                "(construct it with cell=<3x3 lattice in bohr>)."
            )
        if self.periodic_dimension != 3:
            raise ValueError(
                "MACE stress is exposed only for dim=3 systems. A dim=1/2 "
                "PeriodicSystem has a non-physical bookkeeping cell along "
                "its non-periodic axes; use energy/forces and fixed-cell "
                "position optimization instead."
            )
        self._compute()
        # ASE stress is eV/Angstrom^3; -> Ha/bohr^3 via x * Bohr^3 / Hartree.
        return np.asarray(self._stress_eVA3) * (self._Bohr ** 3) / self._Hartree

    @property
    def calculator(self):
        """The underlying ASE ``Calculator`` (eV/Angstrom) for ASE-native
        drivers (e.g. BFGS geometry optimization)."""
        return self._calc

    @property
    def info(self) -> MaceModelInfo:
        """Provenance for the selected model (key, license, citation, ...)."""
        return self._info

    @property
    def options(self) -> MLIPOptions:
        """A detached copy of the runtime choices used to load this model."""
        return replace(self._options)

    @property
    def weight_identity(self) -> dict:
        """Identity of the checkpoint bytes consumed at model loading."""
        return asdict(self._weight_identity) if self._weight_identity is not None else {}

    @property
    def pbc(self) -> tuple[bool, bool, bool] | None:
        """ASE periodic-boundary mask, or ``None`` for a molecule."""
        return self._pbc

    @property
    def periodic_dimension(self) -> int:
        """Number of periodic axes represented by :attr:`pbc`."""
        return 0 if self._pbc is None else sum(self._pbc)

    @property
    def model_name(self) -> str:
        return self._info.key


class _PeriodicMaceResult:
    """Duck-typed periodic MLIP result: energy (Hartree, model-specific
    reference scale), gradient (Hartree/bohr), and stress (Hartree/bohr^3),
    plus model provenance. Mirrors the molecular ``_MLIPResult`` shape with
    an added stress tensor."""

    def __init__(
        self,
        energy,
        gradient,
        stress,
        model_info,
        *,
        options,
        pbc,
        weight_identity=None,
        converged=True,
        n_iter=1,
    ):
        self.energy = energy
        self._gradient = gradient
        self._stress = stress
        self.method = "mace"
        self.converged = converged
        self.n_iter = n_iter
        self.scf_trace = []
        self.model_info = model_info
        self.weight_identity = dict(weight_identity or {})
        self.device = options.device
        self.dtype = options.dtype
        self.loader = model_info.loader
        self.loader_arg = model_info.loader_arg
        self.cache_root = mace_cache_root()
        self.pbc = tuple(pbc)
        self.dim = sum(self.pbc)

    def gradient(self):
        return self._gradient

    def stress(self):
        if self._stress is None:
            raise ValueError(
                "MACE stress is exposed only for dim=3 systems; this result "
                f"has dim={self.dim}, pbc={self.pbc}. Use energy/forces for "
                "a slab and keep its cell fixed."
            )
        return self._stress


def _periodic_result(model: MACEModel) -> _PeriodicMaceResult:
    stress = model.stress() if model.periodic_dimension == 3 else None
    return _PeriodicMaceResult(
        energy=model.energy(),
        gradient=model.gradient(),
        stress=stress,
        model_info=model.info,
        options=model.options,
        pbc=model.pbc,
        weight_identity=model.weight_identity,
    )


class PeriodicMACEEvaluator:
    """Reusable MACE evaluator for a sequential fixed-model periodic scan.

    The upstream calculator and its model weights are loaded lazily on the
    first :meth:`run`, then reused for later structures. Input validation,
    column-to-row lattice conversion, units, and model provenance match
    :func:`run_periodic_mace`. One evaluator is intentionally sequential;
    use one instance per worker for parallel scans.

    Parameters
    ----------
    options
        Fixed model, device, dtype, and license acknowledgment for the whole
        scan. The values are snapshotted when the evaluator is constructed.
    """

    def __init__(self, options: MLIPOptions | None = None):
        if options is None:
            options = MLIPOptions()
        self._options = replace(options)
        self._info = resolve_model(self._options.model)
        # Preserve the established fail-fast contract: an ASL model is
        # rejected when the evaluator is configured, before torch import,
        # weight download, or the first structure.
        enforce_academic_license(self._info, self._options)
        _validate_runtime_options(self._options)
        self._model: MACEModel | None = None

    def run(self, system) -> _PeriodicMaceResult:
        """Evaluate one ``PeriodicSystem`` with the retained calculator."""
        import numpy as np

        molecule = system.unit_cell_molecule()
        cell_bohr = np.asarray(system.lattice, dtype=float)
        pbc = _pbc_for_dimension(system.dim)
        if self._model is None:
            self._model = MACEModel(
                molecule,
                self._options,
                cell=cell_bohr,
                pbc=pbc,
            )
        else:
            self._model._set_structure(molecule, cell=cell_bohr, pbc=pbc)
        return _periodic_result(self._model)

    @property
    def model_info(self) -> MaceModelInfo:
        """Provenance for the fixed model used by this evaluator."""
        return self._info


def run_periodic_mace(system, options: MLIPOptions | None = None):
    """Periodic MACE single point for a ``PeriodicSystem``.

    Returns a result with ``.energy`` (Hartree, model-specific reference
    scale), ``.gradient()`` (Hartree/bohr, ``(n_atoms, 3)``), runtime/model
    provenance, and, for ``dim=3`` only, ``.stress()`` (Hartree/bohr^3,
    ``(3, 3)``). Energy and forces support true ``dim=1`` wires and ``dim=2``
    slabs with the corresponding ASE PBC mask. Slab stress is deliberately
    unavailable because the non-periodic cell volume is arbitrary.

    vibe-qc drives MACE's pre-trained periodic forward pass (``CLAUDE.md``
    Sec.10); the ASL gate fires inside :class:`MACEModel`. This is a direct
    driver, following the periodic-semiempirical precedent (e.g.
    ``run_pm6_gamma``) rather than routing through ``run_periodic_job``.
    """
    return PeriodicMACEEvaluator(options).run(system)


class PeriodicMACECellOptimizationError(RuntimeError):
    """ASE exhausted the cell-relaxation budget without converging.

    ``system`` retains the last geometry for inspection or an explicit restart.
    ``final_max_force`` is the maximum norm of the ASE cell filter's combined
    atomic/cell forces, using the same scaling and units as its ``fmax`` test;
    it is not a raw stress in Hartree/bohr^3 or an atomic-only force norm.
    """

    def __init__(self, *, system, n_steps, max_steps, final_max_force,
                 target_max_force):
        self.system = system
        self.n_steps = n_steps
        self.max_steps = max_steps
        self.final_max_force = final_max_force
        self.target_max_force = target_max_force
        super().__init__(
            "Periodic MACE cell optimization did not converge after "
            f"{n_steps} of {max_steps} allowed steps: final maximum "
            f"cell-filter force {final_max_force:.6g} does not meet the "
            f"{target_max_force:.6g} ASE fmax target. The last geometry "
            "is available as this exception's system attribute."
        )


def _validate_optimization_controls(fmax, max_steps, *, step_option="max_steps") -> int:
    """Reject invalid stopping criteria before model loading or ASE work."""
    import math

    if (
        isinstance(fmax, bool) or not isinstance(fmax, Real)
        or not math.isfinite(fmax) or fmax <= 0.0
    ):
        raise ValueError(f"fmax must be finite and positive; got {fmax!r}")
    if (
        isinstance(max_steps, bool) or not isinstance(max_steps, Integral)
        or max_steps < 0
    ):
        raise ValueError(
            f"{step_option} must be a non-negative integer; got {max_steps!r}"
        )
    return int(max_steps)


def optimize_periodic_mace_cell(system, options: MLIPOptions | None = None,
                                *, fmax: float = 0.05, max_steps: int = 200):
    """Variable-cell relaxation of a ``PeriodicSystem`` with MACE.

    BFGS on an ASE cell filter, driven by MACE's analytic energy / forces /
    stress (relaxing both atomic positions and lattice vectors). Returns the
    relaxed ``PeriodicSystem`` only if ASE's combined atomic/cell force
    criterion is met. Otherwise raises :class:`PeriodicMACECellOptimizationError`
    with the last structure and residual. ``fmax`` uses ASE's cell-filter
    force scaling (eV/Angstrom). This API is intentionally 3D-only; use
    :func:`optimize_periodic_mace_positions` for a slab or wire.
    Both the relaxed and last-on-failure structures retain the input charge
    and spin multiplicity for subsequent electronic-structure calculations.
    """
    max_steps = _validate_optimization_controls(fmax, max_steps)
    import numpy as np
    from ase.optimize import BFGS

    try:  # ASE >= 3.23 renamed the variable-cell filter
        from ase.filters import FrechetCellFilter as _CellFilter
    except ImportError:  # pragma: no cover - older ASE
        from ase.constraints import ExpCellFilter as _CellFilter

    if int(system.dim) != 3:
        raise ValueError(
            "optimize_periodic_mace_cell requires dim=3 because MACE/ASE "
            "cell stress uses a 3D volume. For dim=1/2 systems use "
            "optimize_periodic_mace_positions, which keeps the cell fixed."
        )

    from vibeqc.ase_periodic import atoms_to_periodic_system

    mol = system.unit_cell_molecule()
    cell_bohr = np.asarray(system.lattice, dtype=float)
    model = MACEModel(mol, options, cell=cell_bohr, pbc=(True, True, True))
    atoms = model._ase_atoms()  # cell + pbc, Angstrom
    atoms.calc = model.calculator
    cell_filter = _CellFilter(atoms)
    optimizer = BFGS(cell_filter, logfile=None)
    # ASE Dynamics.run returns convergence, not whether it finished its loop.
    # The FrechetCellFilter criterion includes cell forces: zero atomic forces
    # at a symmetric but strained cell are insufficient (ASE optimize/filters).
    converged = bool(optimizer.run(fmax=fmax, steps=max_steps))
    final_results = {
        "energy": atoms.get_potential_energy(),
        "forces": atoms.get_forces(apply_constraint=False),
        "stress": atoms.get_stress(),
    }
    for name, shape in (("energy", ()), ("forces", (len(atoms), 3)), ("stress", (6,))):
        _checked_backend_result(final_results, name, shape, model.info.key)
    final = atoms_to_periodic_system(
        atoms, charge=int(system.charge), multiplicity=int(system.multiplicity),
    )
    if not converged:
        final_max_force = float(
            np.linalg.norm(cell_filter.get_forces(), axis=1).max()
        )
        raise PeriodicMACECellOptimizationError(
            system=final,
            n_steps=int(optimizer.nsteps),
            max_steps=max_steps,
            final_max_force=final_max_force,
            target_max_force=fmax,
        )
    return final


class PeriodicMACEOptimizationResult:
    """Result of a fixed-cell periodic MACE position optimization.

    ``system`` is the relaxed :class:`vibeqc.PeriodicSystem`; ``energy`` is in
    Hartree and ``gradient()`` in Hartree/bohr. The gradient is the physical
    energy derivative, including frozen atoms. ``max_force_eva`` is the terminal
    constrained ASE force norm in eV/Angstrom, matching the input ``fmax``.
    """

    def __init__(
        self,
        *,
        system,
        energy,
        gradient,
        converged,
        n_steps,
        max_force_eva,
        model_info,
        options,
        pbc,
        fixed_indices,
        weight_identity=None,
    ):
        self.system = system
        self.energy = energy
        self._gradient = gradient
        self.method = "mace"
        self.converged = converged
        self.n_steps = n_steps
        self.n_iter = n_steps
        self.max_force_eva = max_force_eva
        self.model_info = model_info
        self.weight_identity = dict(weight_identity or {})
        self.device = options.device
        self.dtype = options.dtype
        self.loader = model_info.loader
        self.loader_arg = model_info.loader_arg
        self.cache_root = mace_cache_root()
        self.pbc = tuple(pbc)
        self.dim = sum(self.pbc)
        self.fixed_indices = tuple(fixed_indices)

    @property
    def relaxed_system(self):
        """Alias spelling out the role of :attr:`system`."""
        return self.system

    def gradient(self):
        return self._gradient


def optimize_periodic_mace_positions(
    system,
    options: MLIPOptions | None = None,
    *,
    fmax: float = 0.05,
    max_steps: int = 200,
    fixed_indices=(),
) -> PeriodicMACEOptimizationResult:
    """Relax atomic positions with MACE while keeping the cell fixed.

    This is the first-class geometry path for wires, true 2D slabs, and 3D
    crystals. ``fixed_indices`` accepts zero-based atom indices, typically
    ``SlabInfo.bottom_layer_indices(n)`` for substrate layers. The returned
    structure preserves the input dimensionality, lattice, charge, spin, and
    atom ordering, making it suitable for a subsequent periodic electronic-
    structure single point.

    The optimizer reports ``converged=False`` when ``max_steps`` is reached;
    a capped run is not silently presented as a relaxed structure.
    """
    max_steps_value = _validate_optimization_controls(fmax, max_steps)
    import numpy as np
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS

    from vibeqc import Atom, PeriodicSystem

    molecule = system.unit_cell_molecule()
    n_atoms = len(molecule.atoms)
    indices = tuple(fixed_indices)
    if any(
        isinstance(index, bool) or not isinstance(index, Integral)
        for index in indices
    ):
        raise ValueError("fixed_indices must contain integer atom indices")
    fixed = tuple(sorted({int(index) for index in indices}))
    invalid = [index for index in fixed if index < 0 or index >= n_atoms]
    if invalid:
        raise IndexError(
            f"fixed_indices contains out-of-range atom indices {invalid}; "
            f"the system has {n_atoms} atoms"
        )

    pbc = _pbc_for_dimension(system.dim)
    cell_bohr = np.asarray(system.lattice, dtype=float)
    model = MACEModel(molecule, options, cell=cell_bohr, pbc=pbc)
    atoms = model._ase_atoms()
    atoms.calc = model.calculator
    if fixed:
        atoms.set_constraint(FixAtoms(indices=fixed))

    optimizer = BFGS(atoms, logfile=None)
    converged = bool(optimizer.run(fmax=fmax, steps=max_steps_value))
    final_results = {
        "energy": atoms.get_potential_energy(),
        "forces": atoms.get_forces(apply_constraint=False),
    }
    energy_eV = float(
        _checked_backend_result(final_results, "energy", (), model.info.key)
    )
    forces_eva = _checked_backend_result(
        final_results, "forces", (n_atoms, 3), model.info.key,
    )
    gradient = -forces_eva * model._Bohr / model._Hartree
    # Constraints project the optimizer's stopping residual, not the physical
    # derivative of the reported energy at the returned structure.
    force_norms = np.linalg.norm(atoms.get_forces(), axis=1)
    max_force_eva = float(force_norms.max()) if force_norms.size else 0.0

    relaxed_atoms = [
        Atom(int(z), list(position / model._Bohr))
        for z, position in zip(atoms.numbers, atoms.positions)
    ]
    relaxed = PeriodicSystem(
        int(system.dim),
        cell_bohr,
        relaxed_atoms,
        charge=int(system.charge),
        multiplicity=int(system.multiplicity),
    )
    return PeriodicMACEOptimizationResult(
        system=relaxed,
        energy=energy_eV / model._Hartree,
        gradient=gradient,
        converged=converged,
        n_steps=int(getattr(optimizer, "nsteps", 0)),
        max_force_eva=max_force_eva,
        model_info=model.info,
        options=model.options,
        pbc=pbc,
        fixed_indices=fixed,
        weight_identity=model.weight_identity,
    )
