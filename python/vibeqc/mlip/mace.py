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
actionable error. MACE currently requires Python <= 3.13.

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
from numbers import Integral
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
    vibe-qc never bundles or copies model weights; the upstream loader fetches
    a registered weight on first use.
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


def _require_backends():
    """Import the MACE runtime stack, or raise a clear, actionable error.

    Returns ``(mace_mp, mace_off, torch, Atoms, Bohr, Hartree)``.
    """
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
        from ase import Atoms
        from ase.units import Bohr, Hartree
        from mace.calculators import mace_mp, mace_off
    except ImportError as exc:  # pragma: no cover - import-time guard
        hint = (
            "vibeqc.mlip.mace requires the MACE stack (PyTorch + e3nn). "
            "Install it with `pip install 'vibe-qc[mace]'` (or "
            "`pip install mace-torch`) into your vibe-qc venv."
        )
        if sys.version_info >= (3, 14):
            hint += (
                f"  NOTE: this interpreter is Python {sys.version_info.major}."
                f"{sys.version_info.minor}; MACE caps at Python 3.13 today "
                "(its matscipy dependency ships no 3.14 wheel). Use a "
                "Python <= 3.13 environment for MACE."
            )
        raise ImportError(hint) from exc
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


def mace_calculator(options: MLIPOptions | None = None):
    """Return the bare MACE ASE ``Calculator`` (eV / Angstrom) for the
    selected model -- for ASE-native workflows (custom optimizers, MD,
    slab / adsorption studies). Applies the ASL license gate; needs no
    molecule. For vibe-qc's energy/gradient/stress wrapper (Hartree units)
    use :class:`MACEModel` instead."""
    if options is None:
        options = MLIPOptions()
    info = resolve_model(options.model)
    enforce_academic_license(info, options)
    mace_mp, mace_off, _torch, _Atoms, _Bohr, _Hartree = _require_backends()
    loader = mace_mp if info.loader == "mace_mp" else mace_off
    return loader(model=info.loader_arg, device=options.device,
                  default_dtype=options.dtype)


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
        Defaults to the MIT MACE-MPA-0 on CPU in float64.
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
        self._options = options
        self._info = resolve_model(options.model)
        # ASL gate (CLAUDE.md Sec.1/Sec.10): academic, non-commercial models
        # require explicit acknowledgment; MIT models are ungated. Fires
        # before any torch/mace import or weight download.
        enforce_academic_license(self._info, options)
        self._set_structure(molecule, cell=cell, pbc=pbc)
        mace_mp, mace_off, _torch, Atoms, Bohr, Hartree = _require_backends()
        self._Atoms = Atoms
        self._Bohr = float(Bohr)
        self._Hartree = float(Hartree)
        loader = mace_mp if self._info.loader == "mace_mp" else mace_off
        self._calc = loader(
            model=self._info.loader_arg,
            device=options.device,
            default_dtype=options.dtype,
        )

    def _set_structure(self, molecule, *, cell=None, pbc=None) -> None:
        """Replace the evaluated structure while retaining the loaded model.

        This is private because a caller holding a one-shot ``MACEModel``
        should not mutate it behind an existing result. The reusable periodic
        evaluator below owns one model and calls this sequentially between
        scan points.
        """
        _validate_mlip_input(molecule, self._info)
        self._mol = molecule
        # 3x3 lattice vectors (bohr) for a periodic system; None = molecular.
        self._cell = cell
        if cell is None:
            if pbc is not None:
                raise ValueError("pbc requires a periodic MACE cell")
            self._pbc = None
        elif pbc is None:
            # Backward-compatible lower-level default: a caller that supplies
            # only a cell means a fully 3D-periodic system. Public periodic
            # drivers always pass the PeriodicSystem's exact dimensionality.
            self._pbc = (True, True, True)
        else:
            values = tuple(bool(value) for value in pbc)
            if len(values) != 3:
                raise ValueError(f"pbc must contain three flags; got {pbc!r}")
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
            self._e_eV = results["energy"]
            self._f_eVA = results["forces"]
            if "stress" in properties:
                # MACECalculator stores stress as the Voigt 6-vector; ASE's
                # Atoms.get_stress(voigt=False) converts it back to the 3x3
                # tensor. Reading .results directly does the same conversion
                # with the same public helper (no constraints or momenta on
                # this internal Atoms object, so the values are identical).
                from ase.stress import voigt_6_to_full_3x3_stress

                self._stress_eVA3 = voigt_6_to_full_3x3_stress(
                    results["stress"]
                )

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
        """Model runtime choices used by this wrapper."""
        return self._options

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
        self._options = MLIPOptions(
            model=options.model,
            device=options.device,
            dtype=options.dtype,
            accept_academic_license=options.accept_academic_license,
        )
        self._info = resolve_model(self._options.model)
        # Preserve the established fail-fast contract: an ASL model is
        # rejected when the evaluator is configured, before torch import,
        # weight download, or the first structure.
        enforce_academic_license(self._info, self._options)
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


def optimize_periodic_mace_cell(system, options: MLIPOptions | None = None,
                                *, fmax: float = 0.05, max_steps: int = 200):
    """Variable-cell relaxation of a ``PeriodicSystem`` with MACE.

    BFGS on an ASE cell filter, driven by MACE's analytic energy / forces /
    stress (relaxing both atomic positions and lattice vectors). Returns the
    relaxed ``PeriodicSystem``. ``fmax`` is the ASE force/stress convergence
    threshold (eV/Angstrom). This API is intentionally 3D-only; use
    :func:`optimize_periodic_mace_positions` for a slab or wire.
    """
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
    BFGS(_CellFilter(atoms), logfile=None).run(fmax=fmax, steps=max_steps)
    return atoms_to_periodic_system(atoms)


class PeriodicMACEOptimizationResult:
    """Result of a fixed-cell periodic MACE position optimization.

    ``system`` is the relaxed :class:`vibeqc.PeriodicSystem`; ``energy`` and
    ``gradient()`` use Hartree/bohr units. ``max_force_eva`` is the terminal
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
    import numpy as np
    from ase.constraints import FixAtoms
    from ase.optimize import BFGS

    from vibeqc import Atom, PeriodicSystem

    if fmax <= 0.0:
        raise ValueError(f"fmax must be positive; got {fmax}")
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, Integral)
        or max_steps < 0
    ):
        raise ValueError(
            f"max_steps must be a non-negative integer; got {max_steps!r}"
        )
    max_steps_value = int(max_steps)

    molecule = system.unit_cell_molecule()
    n_atoms = len(molecule.atoms)
    fixed = tuple(sorted({int(index) for index in fixed_indices}))
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
    forces_eva = np.asarray(atoms.get_forces(), dtype=float)
    gradient = -forces_eva * model._Bohr / model._Hartree
    force_norms = np.linalg.norm(forces_eva, axis=1)
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
        energy=float(atoms.get_potential_energy()) / model._Hartree,
        gradient=gradient,
        converged=converged,
        n_steps=int(getattr(optimizer, "nsteps", 0)),
        max_force_eva=max_force_eva,
        model_info=model.info,
        options=model.options,
        pbc=pbc,
        fixed_indices=fixed,
    )
