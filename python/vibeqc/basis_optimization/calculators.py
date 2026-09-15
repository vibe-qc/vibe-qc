"""The vibe-qc energy engine -- tier 2 of the basis-opt engine roster.

vibe-qc is the **primary** engine for basis optimization
(``vibe-basis/ROADMAP.md`` § 2): every milestone targets it, and the
external engines are fallback and cross-validation only. This module is
where that primary implementation lives.

It sits on the vibe-qc side, not in vibe-basis, for one reason: it needs
vibe-qc's internals and its compiled native core. The external engines
(:class:`~vibe_basis.engines.crystal23.Crystal23Engine`,
:class:`~vibe_basis.engines.gpaw.GpawEngine`) need neither, so they are
tier 1 and live in vibe-basis. The protocol they all implement --
:class:`~vibe_basis.engine.EnergyEngine` -- is declared in vibe-basis,
because the dependency arrow runs vibe-qc -> vibe-basis and never back.

Import note: this module imports ``vibe_basis`` at module level, so it
requires vibe-qc's ``[basisopt]`` extra. It is deliberately **not**
imported from ``vibeqc.basis_optimization.__init__`` -- that package's
import graph is free of third-party dependencies, which is what keeps
the native analytic-gradient molecular optimiser
(``optimize_bdiis``, ``energy_gradient``) usable on a plain
``pip install vibe-qc``. Keep it that way.

History: this module used to also define the ``Calculator`` ABC and
``CRYSTAL14Calculator``, both of which imported only from vibe-basis and
were therefore tier-1 code filed on the tier-2 side. They moved to
vibe-basis in 0.3.0 as ``EnergyEngine`` and ``Crystal23Engine``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Optional, Sequence

from vibe_basis.engine import EnergyEngine, EngineEnergy, RelaxedStructure

# ---------------------------------------------------------------------------
# Structure <-> PeriodicSystem marshalling
# ---------------------------------------------------------------------------
#
# The engine protocol's `structure` is a `vibe_basis.io.structures.Structure`
# (lattice parameters in Angstrom + fractional coordinates). vibe-qc's
# `PeriodicSystem` wants a Cartesian lattice matrix in bohr + Cartesian
# atoms in bohr. ``crystal_energy`` converts into ``PeriodicSystem``. The
# reverse helper remains available for callers handling a relaxed result from
# another certified engine, but ``VibeQcEngine`` does not advertise relaxation.


def _to_periodic_system(structure: Any):
    """Marshal a vibe-basis ``Structure`` into a ``vq.PeriodicSystem``.

    A ``PeriodicSystem`` is passed through unchanged, so the engine is
    also usable directly from a vibe-qc script or a test without going
    through the structure database.
    """
    import numpy as np

    import vibeqc as vq
    from vibeqc.molecule import ANGSTROM_TO_BOHR

    if hasattr(structure, "lattice") and hasattr(structure, "unit_cell_molecule"):
        return structure  # already a PeriodicSystem

    if not hasattr(structure, "lattice_matrix_angstrom"):
        raise TypeError(
            "VibeQcEngine needs a vibe_basis Structure (or a "
            f"vq.PeriodicSystem); got {type(structure).__name__}"
        )

    lattice = np.asarray(structure.lattice_matrix_angstrom(), dtype=float)
    lattice_bohr = lattice * ANGSTROM_TO_BOHR

    atoms = []
    for site in structure.unit_cell:
        frac = np.asarray(site.fxyz, dtype=float)
        cart = frac @ lattice_bohr
        atoms.append(vq.Atom(int(site.Z), [float(c) for c in cart]))

    return vq.PeriodicSystem(
        dim=3,
        lattice=lattice_bohr,
        unit_cell=atoms,
        multiplicity=int(getattr(structure, "multiplicity", 1) or 1),
    )


def _structure_from_periodic_system(system: Any, template: Any) -> Any:
    """Marshal a relaxed ``PeriodicSystem`` back into a ``Structure``.

    Two fields are deliberately **cleared** rather than carried over:
    ``crystal_spacegroup`` and ``crystal_asymm_unit``. They describe the
    *input* geometry's symmetry, and a relaxation has moved the atoms
    and the cell; carrying them would let
    :func:`vibe_basis.backends.crystal.emit_input_inline` build a
    CRYSTAL deck from stale symmetry data and silently compute a
    different crystal. Cleared, that emitter returns ``None`` and the
    caller gets an honest ``emit_failed`` instead.

    Orientation is not preserved: the returned ``Structure`` stores
    (a, b, c, alpha, beta, gamma), from which
    ``lattice_matrix_angstrom()`` rebuilds a canonically-oriented cell.
    That differs from the relaxed lattice by a rigid rotation at most,
    which no energy depends on.
    """
    import math

    import numpy as np

    from vibeqc.molecule import ANGSTROM_TO_BOHR

    lattice_bohr = np.asarray(system.lattice, dtype=float)
    lattice_ang = lattice_bohr / ANGSTROM_TO_BOHR

    a_vec, b_vec, c_vec = lattice_ang
    a = float(np.linalg.norm(a_vec))
    b = float(np.linalg.norm(b_vec))
    c = float(np.linalg.norm(c_vec))

    def _angle(u, v, nu, nv) -> float:
        cosine = float(np.dot(u, v) / (nu * nv))
        return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))

    alpha = _angle(b_vec, c_vec, b, c)
    beta = _angle(a_vec, c_vec, a, c)
    gamma = _angle(a_vec, b_vec, a, b)

    from vibe_basis.io.structures import StructureAtom

    inv = np.linalg.inv(lattice_bohr)
    sites = []
    for atom in system.unit_cell:
        frac = np.asarray(atom.xyz, dtype=float) @ inv
        sites.append(
            StructureAtom(Z=int(atom.Z), fxyz=tuple(float(f) for f in frac))
        )

    import dataclasses

    return dataclasses.replace(
        template,
        a=a,
        b=b,
        c=c,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        unit_cell=tuple(sites),
        crystal_spacegroup=0,
        crystal_asymm_unit=(),
        notes=(
            (template.notes + " " if template.notes else "")
            + "[geometry relaxed by VibeQcEngine; symmetry fields cleared]"
        ),
    )


# ---------------------------------------------------------------------------
# Basis marshalling
# ---------------------------------------------------------------------------


def _parse_inline_basis_atoms(basis_text: str) -> list:
    """Every element block in a CRYSTAL inline basis, in source order."""
    from vibeqc.basis_crystal import parse_crystal_inline_basis

    return parse_crystal_inline_basis(basis_text, source="<engine basis_text>")


def _parse_inline_basis_text(basis_text: str, Z: int):
    """The element block for *Z* out of a CRYSTAL inline basis text.

    Retained under its historical name because it is the one-element
    view callers already use. It no longer *assumes* one element: the
    text an engine receives is the whole candidate basis for a system,
    so picking the first block and stamping the requested ``Z`` on it --
    which this function used to do -- hands O the Mg basis for MgO. That
    converges, and it is wrong.
    """
    atoms = _parse_inline_basis_atoms(basis_text)
    for atom in atoms:
        if atom.Z == Z:
            return atom
    raise KeyError(
        f"no basis block for Z={Z} in the inline basis text "
        f"(found: {sorted(a.Z for a in atoms)})"
    )


def _write_basis(library, atoms: Sequence[Any], stem: str) -> str:
    """Emit *atoms* as a ``.g94`` inside *library*; return the basis name.

    The name is uuid-suffixed so libint's ``BasisSet`` constructor never
    returns a cached parse of a *previous* candidate that happened to
    share a name -- the failure mode there is an optimizer that appears
    to converge while every evaluation used the same basis.
    """
    import uuid

    from vibeqc.basis_crystal import emit_g94

    name = f"{stem}-{uuid.uuid4().hex[:8]}"
    (library.path / "basis" / f"{name}.g94").write_text(emit_g94(list(atoms)))
    return name


# ---------------------------------------------------------------------------
# Method mapping
# ---------------------------------------------------------------------------

#: ``method`` strings the engine reads as Hartree-Fock rather than as an
#: XC functional name.
_HF_ALIASES = frozenset({"rhf", "hf", "uhf"})


def _resolve_method(method: str, multiplicity: int) -> tuple[str, Optional[str]]:
    """``(vibe-qc method, functional)`` for an engine *method* string.

    The engine protocol's ``method`` is either ``"rhf"``/``"hf"`` or a
    functional name (``"pbe"``, ``"pw1pw"``, ``"r2scan"``, ...). The
    restricted/unrestricted branch comes from the spin multiplicity, not
    from the string, so a caller never has to spell ``"uks"`` --
    "r2SCAN on a triplet" is unambiguous.
    """
    restricted = multiplicity == 1
    if method.lower() in _HF_ALIASES:
        return ("RHF" if restricted else "UHF"), None
    return ("RKS" if restricted else "UKS"), method


# ---------------------------------------------------------------------------
# VibeQcEngine -- in-process SCF. The primary engine.
# ---------------------------------------------------------------------------


class VibeQcEngine(EnergyEngine):
    """The primary energy engine: vibe-qc, in-process.

    Covers the periodic single point (:meth:`crystal_energy`), the
    Gamma-point zero-point energy (:meth:`zero_point_energy`), and the
    isolated-atom reference (:meth:`atom_energy`). Variable-cell BIPOLE
    optimization has no certified stress and coupled atom/cell convergence
    path, so this engine does not advertise the ``relax`` capability and a
    direct :meth:`relax` request fails closed.

    Candidate bases are written as temporary ``.g94`` files under a
    managed ``LIBINT_DATA_PATH`` and addressed by name.

    Atom reference convention
    -------------------------
    Isolated atoms are computed **spin-polarised at the tabulated
    ground-state multiplicity, aspherical** (no spherical averaging, no
    fractional occupation): unrestricted for any open shell, restricted
    only for a closed-shell singlet. This is recorded in
    ``EngineEnergy.detail["atom_reference"]`` because it *must* match
    the convention of whatever reference the cohesive energy is compared
    against -- spin-restricting the Br reference alone moved KBr's
    atomization by 37 kJ/mol (``handovers/HANDOVER_GPAW_PW_REFERENCE.md``,
    GPAW-PWREF-002).

    Parameters
    ----------
    library
        A :class:`vibeqc.basis_optimization.io.TempBasisLibrary`
        instance, or ``None`` to create one per evaluation.
    kpoints
        k-mesh for the periodic stages: a ``(n1, n2, n3)`` tuple, a
        scalar, a ``KPoints``/``BlochKMesh``, or ``None`` for Gamma
        only. Gamma is the default because it is what makes a wiring
        test cheap; **it is not a converged setting** for a production
        cohesive energy, and the caller owns that choice.
    periodic_kwargs
        Extra kwargs forwarded to
        :func:`vibeqc.periodic_runner.run_periodic_job`. Anything here
        overrides this class's own defaults, which switch off the
        optional artefact writers (QVF, xyz, xsf, cif) -- an optimizer
        inner loop discards them, and they cost wall time per
        evaluation. None of them touches the energy.
    relax_kwargs
        Retained for constructor compatibility. These options are inactive
        while :meth:`relax` is unavailable.
    phonon_kwargs
        Extra kwargs forwarded to
        :func:`vibeqc.basis_optimization.phonons.gamma_phonons`
        (``step_bohr``, ``asr``, ``hessian_mode``, SCF ``options``, ...).
        The defaults are that function's own: the production
        energy-difference Hessian, ``step_bohr=0.02`` bohr, and **no**
        acoustic sum rule, so the acoustic residual stays visible in
        ``EngineEnergy.detail``.
    work_dir
        Where per-evaluation output stems are written. ``None`` (the
        default) uses a fresh temporary directory that is deleted after
        each evaluation; set it to keep the ``.out`` files for a
        post-mortem.
    scf_kwargs
        Extra kwargs forwarded to the **molecular** SCF runner used for
        isolated atoms.
    """

    name: str = "vibeqc"

    #: Retained for callers that predate the engine rename.
    label: str = "vibe-qc"

    #: ``zero_point`` uses the M1 Gamma-point phonon driver
    #: (``vibeqc.basis_optimization.phonons``). ``relax`` stays undeclared
    #: until BIPOLE has a certified variable-cell optimizer; ``counterpoise``
    #: stays undeclared because this engine has no ghost-basis atom path.
    #: The pipeline therefore records either unsupported stage as skipped.
    capabilities = frozenset({"zero_point"})

    def __init__(
        self,
        *,
        library=None,
        kpoints: Any = None,
        periodic_kwargs: Optional[dict] = None,
        relax_kwargs: Optional[dict] = None,
        phonon_kwargs: Optional[dict] = None,
        work_dir: Optional[Any] = None,
        **scf_kwargs,
    ) -> None:
        self._library = library
        self._kpoints = kpoints
        self._periodic_kwargs = dict(periodic_kwargs or {})
        self._relax_kwargs = dict(relax_kwargs or {})
        self._phonon_kwargs = dict(phonon_kwargs or {})
        self._work_dir = Path(work_dir) if work_dir is not None else None
        self._scf_kwargs = scf_kwargs

    def version(self) -> Optional[str]:
        """The installed vibe-qc version, or ``None`` if unimportable."""
        try:
            import vibeqc as vq
        except ImportError:
            return None
        return getattr(vq, "__version__", None)

    # -- helpers ---------------------------------------------------------

    def _new_library(self):
        from vibeqc.basis_optimization.io import TempBasisLibrary

        return self._library or TempBasisLibrary()

    def _bloch_kmesh(self, system):
        """This engine's ``kpoints=`` as a native ``BlochKMesh``.

        Goes through the periodic runner's own normaliser rather than
        :func:`vibeqc.kpoints.as_bloch_kmesh`, because that one passes a
        plain ``(n, n, n)`` tuple straight through -- it only converts a
        :class:`KPoints` builder -- and the BIPOLE drivers then fail on
        ``tuple.kpoints``. Using the runner's normaliser also guarantees
        the single point and phonons sample the same mesh.
        """
        from vibeqc.periodic_runner import _runner_bloch_kmesh

        return _runner_bloch_kmesh(system, self._kpoints)

    def _output_stem(self, stack, label: str) -> str:
        """A per-evaluation output stem, inside a managed directory."""
        if self._work_dir is not None:
            self._work_dir.mkdir(parents=True, exist_ok=True)
            return str(self._work_dir / label)
        tmp = stack.enter_context(tempfile.TemporaryDirectory(prefix="vqengine-"))
        return str(Path(tmp) / label)

    #: Defaults for the periodic runner. Overridable via
    #: ``periodic_kwargs``.
    #:
    #: ``jk_method="bipole"`` is the load-bearing one. The GDF route
    #: needs a density-fitting auxiliary basis, and a *candidate* basis
    #: has none: ``default_aux_for`` has nothing registered for a
    #: freshly-emitted optimizer basis, and borrowing some other basis's
    #: JKFIT set would lay a fitting error on top of the very
    #: basis-incompleteness signal the objective is measuring. BIPOLE is
    #: four-centre and needs no auxiliary basis, so the number depends on
    #: the candidate basis alone.
    #:
    #: The rest switch off artefact writers an optimizer discards; none
    #: of them touches the energy.
    _RUNNER_DEFAULTS = {
        "jk_method": "bipole",
        "output_qvf": False,
        "write_xyz_file": False,
        "write_xsf_structure_file": False,
        "write_cif_file": False,
        "progress": False,
    }

    # -- the periodic single point ---------------------------------------

    def crystal_energy(
        self,
        basis_text: str,
        structure: Any,
        method: str = "rhf",
    ) -> EngineEnergy:
        """Total energy of one unit cell, in Hartree.

        Never raises for an SCF that ran and failed: a basis optimizer
        walks into infeasible regions routinely and must be able to
        score the point and move on. A **request** it cannot express --
        an unknown structure type, a magnetic ordering it does not
        implement -- is reported as ``ok=False`` with a specific
        ``failure_mode`` for the same reason: a campaign should skip the
        system, not abort.
        """
        import contextlib

        try:
            import vibeqc as vq
        except ImportError:
            raise RuntimeError("VibeQcEngine needs vibe-qc installed") from None

        from vibeqc.periodic_runner import run_periodic_job

        name = getattr(structure, "name", "crystal")
        multiplicity = int(getattr(structure, "multiplicity", 1) or 1)
        scf_method, functional = _resolve_method(method, multiplicity)
        detail: dict[str, Any] = {
            "system": name,
            "method": method,
            "scf_method": scf_method,
            "functional": functional,
            "multiplicity": multiplicity,
            "kpoints": repr(self._kpoints),
        }

        # Fail closed on the two structure classes this engine does not
        # express, rather than computing a different crystal quietly.
        if getattr(structure, "afm_pattern", None):
            return EngineEnergy.failed(
                self.name,
                "unsupported_magnetic_structure",
                engine_version=self.version(),
                afm_pattern=structure.afm_pattern,
                **detail,
            )
        if getattr(structure, "ecp_library", None):
            return EngineEnergy.failed(
                self.name,
                "unsupported_ecp",
                engine_version=self.version(),
                ecp_library=structure.ecp_library,
                **detail,
            )

        try:
            system = _to_periodic_system(structure)
            atoms = _parse_inline_basis_atoms(basis_text)
        except (TypeError, ValueError, KeyError) as exc:
            return EngineEnergy.failed(
                self.name,
                "bad_input",
                engine_version=self.version(),
                error=repr(exc),
                **detail,
            )

        present = {int(a.Z) for a in system.unit_cell}
        supplied = {int(a.Z) for a in atoms}
        if not present <= supplied:
            return EngineEnergy.failed(
                self.name,
                "basis_missing_element",
                engine_version=self.version(),
                missing=sorted(present - supplied),
                **detail,
            )

        kwargs = {
            **self._RUNNER_DEFAULTS,
            "method": scf_method,
            "functional": functional,
            "kpoints": self._kpoints,
            **self._periodic_kwargs,
        }

        with contextlib.ExitStack() as stack:
            library = stack.enter_context(self._new_library())
            basis_name = _write_basis(library, atoms, f"vqe-{name}")
            basis = vq.BasisSet(system.unit_cell_molecule(), basis_name)
            stem = self._output_stem(stack, f"{name}")

            try:
                result = run_periodic_job(system, basis, output=stem, **kwargs)
            except Exception as exc:  # noqa: BLE001 -- reported, not raised
                return EngineEnergy.failed(
                    self.name,
                    "scf_error",
                    engine_version=self.version(),
                    error=repr(exc),
                    **detail,
                )

        if result is None:
            return EngineEnergy.failed(
                self.name, "no_result", engine_version=self.version(), **detail
            )
        detail["n_iter"] = getattr(result, "n_iter", None)
        if not bool(getattr(result, "converged", False)):
            return EngineEnergy.failed(
                self.name,
                "non_converged",
                engine_version=self.version(),
                energy=float(result.energy),  # diagnostic only
                **detail,
            )
        return EngineEnergy(
            energy=float(result.energy),
            ok=True,
            engine=self.name,
            engine_version=self.version(),
            detail=detail,
        )

    # -- the geometry relaxation -----------------------------------------

    def relax(
        self,
        basis_text: str,
        structure: Any,
        method: str = "rhf",
    ) -> RelaxedStructure:
        """Reject unsupported variable-cell relaxation before dispatch."""
        raise NotImplementedError(
            "VibeQcEngine.relax is unavailable because BIPOLE variable-cell "
            "optimization is fail-closed: no certified stress and coupled "
            "atom/cell optimizer are available. Use CohesivePipeline(..., "
            "relax=False) for a static-lattice calculation or select an "
            "engine with a certified cell optimizer."
        )

    # -- the zero-point correction ---------------------------------------

    def zero_point_energy(
        self,
        basis_text: str,
        structure: Any,
        method: str = "rhf",
    ) -> EngineEnergy:
        """Gamma-point zero-point energy, **Hartree per unit cell**, positive.

        Per unit cell is what the protocol asks for;
        :class:`~vibe_basis.pipeline.CohesivePipeline` divides by the
        formula-unit count and **subtracts** the result, because a
        vibrating solid is less bound than a static one.

        Driven by :func:`vibeqc.basis_optimization.phonons.gamma_phonons`
        on the same BIPOLE route, k-mesh and functional as
        :meth:`crystal_energy`.

        Two things the caller must read off ``detail`` rather than assume:

        * ``acoustic_residual_cm1`` -- the Gamma acoustic modes are zero
          in exact arithmetic, so their residual is this calculation's
          noise floor. **No acoustic sum rule is imposed by default**;
          pass ``phonon_kwargs={"asr": "rowsum"}`` to impose one, and
          ``detail["asr"]`` records which you got.
        * ``n_imaginary_modes_excluded`` -- imaginary modes mean the
          geometry is not a minimum. They are excluded from the sum and
          counted, never silently folded in.

        **Cost.** The production Hessian is built from total energies at
        ``18 N^2 + 1`` SCFs (73 for a 2-atom primitive cell) and grows
        quadratically; ``detail["n_scf_estimated"]`` says so before the
        run. See the phonon module's docstring for why it is not built on
        the BIPOLE force (which is itself a finite difference).

        **Gamma-only ZPE is an approximation** for a solid -- it samples
        one q-point and misses acoustic dispersion. It is recorded in
        ``detail["approximation"]``.
        """
        from vibeqc.basis_optimization.phonons import (
            estimate_scf_count,
            gamma_phonons,
        )

        import contextlib

        name = getattr(structure, "name", "crystal")
        multiplicity = int(getattr(structure, "multiplicity", 1) or 1)
        scf_method, functional = _resolve_method(method, multiplicity)
        detail: dict[str, Any] = {
            "system": name,
            "method": method,
            "scf_method": scf_method,
            "functional": functional,
            "multiplicity": multiplicity,
            "kpoints": repr(self._kpoints),
            "approximation": "gamma_point_only",
        }

        if getattr(structure, "afm_pattern", None):
            return EngineEnergy.failed(
                self.name,
                "unsupported_magnetic_structure",
                engine_version=self.version(),
                **detail,
            )
        if getattr(structure, "ecp_library", None):
            return EngineEnergy.failed(
                self.name,
                "unsupported_ecp",
                engine_version=self.version(),
                **detail,
            )

        try:
            system = _to_periodic_system(structure)
            atoms = _parse_inline_basis_atoms(basis_text)
        except (TypeError, ValueError, KeyError) as exc:
            return EngineEnergy.failed(
                self.name,
                "bad_input",
                engine_version=self.version(),
                error=repr(exc),
                **detail,
            )

        present = {int(a.Z) for a in system.unit_cell}
        supplied = {int(a.Z) for a in atoms}
        if not present <= supplied:
            return EngineEnergy.failed(
                self.name,
                "basis_missing_element",
                engine_version=self.version(),
                missing=sorted(present - supplied),
                **detail,
            )

        kwargs = dict(self._phonon_kwargs)
        mode = kwargs.get("hessian_mode", "energy")
        detail["n_scf_estimated"] = estimate_scf_count(
            len(system.unit_cell), mode
        )

        with contextlib.ExitStack() as stack:
            library = stack.enter_context(self._new_library())
            basis_name = _write_basis(library, atoms, f"vqe-ph-{name}")
            try:
                phonons = gamma_phonons(
                    system,
                    basis_name,
                    self._bloch_kmesh(system),
                    kwargs.pop("options", None),
                    method=scf_method,
                    functional=functional,
                    **kwargs,
                )
            except Exception as exc:  # noqa: BLE001 -- reported, not raised
                # A ZPE failure is survivable: the pipeline downgrades to
                # a static-lattice cohesive energy and says so. Aborting a
                # multi-day campaign because one displaced SCF stalled
                # would be the wrong trade.
                return EngineEnergy.failed(
                    self.name,
                    "phonon_error",
                    engine_version=self.version(),
                    error=repr(exc),
                    **detail,
                )

        detail.update(
            {
                "frequencies_cm1": [float(w) for w in phonons.frequencies_cm1],
                "acoustic_residual_cm1": list(phonons.acoustic_residual_cm1),
                "acoustic_projection": list(phonons.acoustic_projection),
                "n_imaginary_modes_excluded": phonons.n_imaginary_modes_excluded,
                "n_acoustic_modes_excluded": phonons.n_acoustic_modes_excluded,
                "asr": phonons.asr,
                "hessian_mode": phonons.hessian_mode,
                "fd_step_bohr": phonons.fd_step_bohr,
                "n_scf": phonons.n_scf,
            }
        )
        return EngineEnergy(
            energy=float(phonons.zero_point_hartree),
            ok=True,
            engine=self.name,
            engine_version=self.version(),
            detail=detail,
        )

    # -- the isolated-atom reference -------------------------------------

    def atom_energy(
        self,
        basis_text: str,
        Z: int,
        method: str = "rhf",
        *,
        host: Any = None,
    ) -> EngineEnergy:
        """Free-atom energy of element *Z*, in Hartree.

        *host* is accepted for protocol compatibility and must be
        ``None``: this engine does not declare the ``counterpoise``
        capability, so `CohesivePipeline` never asks it for a
        ghost-basis atom. Passing one anyway is a caller error rather
        than an infeasible point, so it raises instead of returning
        ``ok=False`` -- silently computing a *bare* atom when a
        counterpoise one was requested would substitute a different
        quantity, which is the exact class of error the counterpoise
        work exists to remove.

        Spin-polarised at the ground-state multiplicity, aspherical --
        see the class docstring for why that convention is load-bearing
        rather than a detail.
        """
        import contextlib

        if host is not None:
            raise NotImplementedError(
                "VibeQcEngine does not support counterpoise free atoms; it "
                "does not declare the 'counterpoise' capability. Use "
                "Crystal23Engine for ATOMBSSE-corrected atoms, or run the "
                "pipeline with counterpoise=False."
            )

        try:
            import vibeqc as vq
        except ImportError:
            raise RuntimeError("VibeQcEngine needs vibe-qc installed") from None

        # One table for the free-atom ground states, shared with
        # `vibeqc.atomization`, so the periodic cohesive energy and the
        # molecular atomization energy cannot disagree about what the
        # ground state of an element is.
        from vibeqc.atomization import _GROUND_STATE_MULTIPLICITY

        detail: dict[str, Any] = {
            "Z": Z,
            "method": method,
            "atom_reference": "aspherical_spin_polarised_ground_multiplicity",
        }

        multiplicity = _GROUND_STATE_MULTIPLICITY.get(Z)
        if multiplicity is None:
            return EngineEnergy.failed(
                self.name,
                "no_ground_state",
                engine_version=self.version(),
                supported=sorted(_GROUND_STATE_MULTIPLICITY),
                **detail,
            )
        detail["multiplicity"] = multiplicity

        try:
            atom = _parse_inline_basis_text(basis_text, Z)
        except (ValueError, KeyError) as exc:
            return EngineEnergy.failed(
                self.name,
                "bad_input",
                engine_version=self.version(),
                error=repr(exc),
                **detail,
            )

        scf_method, functional = _resolve_method(method, multiplicity)
        detail["scf_method"] = scf_method
        detail["functional"] = functional

        with contextlib.ExitStack() as stack:
            library = stack.enter_context(self._new_library())
            basis_name = _write_basis(library, [atom], f"vqe-atom{Z}")

            mol = vq.Molecule(
                [vq.Atom(Z, [0.0, 0.0, 0.0])], multiplicity=multiplicity
            )
            basis = vq.BasisSet(mol, basis_name)

            if scf_method == "RHF":
                runner, opts = vq.run_rhf, vq.RHFOptions()
            elif scf_method == "UHF":
                runner, opts = vq.run_uhf, vq.UHFOptions()
            elif scf_method == "RKS":
                runner, opts = vq.run_rks, vq.RKSOptions()
                opts.functional = functional
            else:
                runner, opts = vq.run_uks, vq.UKSOptions()
                opts.functional = functional

            for key, value in self._scf_kwargs.items():
                setattr(opts, key, value)

            try:
                result = runner(mol, basis, opts)
            except Exception as exc:  # noqa: BLE001 -- reported, not raised
                return EngineEnergy.failed(
                    self.name,
                    "scf_error",
                    engine_version=self.version(),
                    error=repr(exc),
                    **detail,
                )

        if result is None:
            return EngineEnergy.failed(
                self.name, "no_result", engine_version=self.version(), **detail
            )
        detail["n_iter"] = getattr(result, "n_iter", None)
        if not bool(getattr(result, "converged", False)):
            return EngineEnergy.failed(
                self.name,
                "non_converged",
                engine_version=self.version(),
                energy=float(result.energy),  # diagnostic only
                **detail,
            )
        return EngineEnergy(
            energy=float(result.energy),
            ok=True,
            engine=self.name,
            engine_version=self.version(),
            detail=detail,
        )
