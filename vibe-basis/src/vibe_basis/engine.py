"""The energy-engine abstraction — one protocol, several engines.

A basis-set optimization needs total energies: of a crystal, and of the
isolated atoms it is built from. *Which program computes them* is a
configuration choice, not a structural one. This module is the seam that
makes it so.

Engine policy
-------------
Per ``vibe-basis/ROADMAP.md`` § 2:

======================  ==============================================
Role                    Engine
======================  ==============================================
**Primary, Gaussian**   **vibe-qc** (BIPOLE / GDF)
**Primary, plane wave** **vibe-qc** (GPW / GAPW) -- not ready, BOC-7
Fallback, Gaussian      CRYSTAL23 -- pob-lineage continuity, baseline
Fallback, plane wave    GPAW -- today's validated PW-limit oracle
======================  ==============================================

**vibe-qc is the engine. External codes are fallback and
cross-validation only.** No milestone is *complete* until it runs on
``VibeQcEngine``; the fallbacks exist to unblock work whose vibe-qc
capability has not landed, and to provide the independent cross-code
check that is the whole point of a plane-wave reference.

That policy only holds if it is cheap to switch, which is why nothing
downstream may name a program. Recipes, objectives and pipelines take an
:class:`EnergyEngine`; swapping CRYSTAL23 out for vibe-qc when GDF /
BIPOLE is ready must be a config change, not a rewrite.

Where engines live
------------------
The tier rule (``ROADMAP.md`` § 3) decides, and it is not arbitrary:

* :class:`Crystal23Engine` and :class:`GpawEngine` are **tier 1**, here
  in vibe-basis, because driving an external program needs nothing from
  vibe-qc -- just subprocess and file I/O.
* ``VibeQcEngine`` is **tier 2**, in
  ``python/vibeqc/basis_optimization/``, because it needs vibe-qc's
  internals and its native build.

So the primary engine lives *outside* this package while the fallbacks
live inside it. That inversion is deliberate: it is what keeps
``pip install vibe-basis`` a ten-second install for a collaborator with
their own CRYSTAL, and it is what lets an independent engine behind the
same protocol *verify* the vibe-qc number rather than merely agree with
it by construction.

Provenance
----------
:class:`EngineEnergy` carries the engine and its version alongside every
number, because "MgO is -275.4776 Ha" is not a result -- "CRYSTAL23
computed -275.4776 Ha for MgO at RHF/pob-TZVP" is. A campaign that mixes
engines and records only floats cannot be audited afterwards, and the
mixing is exactly what this module makes easy.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class EngineEnergy:
    """One total energy plus the provenance needed to trust it.

    Attributes
    ----------
    energy
        Total energy in **Hartree**, per unit cell for a crystal and per
        atom for an isolated atom. ``None`` iff the evaluation failed.
    ok
        ``True`` iff the evaluation converged and completed. Optimizer
        code branches on this and substitutes ``inf`` for the objective
        when it is ``False`` -- a derivative-free driver routes around
        infeasible points, so a failed evaluation must be *reported*,
        never raised.
    engine
        Engine name, e.g. ``"crystal23"``, ``"vibeqc"``, ``"gpaw"``.
    engine_version
        Version string as reported by the engine itself, or ``None``
        when it could not be determined. Never guessed.
    failure_mode
        Short machine-readable tag when ``ok`` is ``False``
        (``"non_converged"``, ``"transport_error"``,
        ``"no_energy_line"``, ...). ``None`` on success.
    detail
        Free-form extras for diagnosis: cycle counts, wall time, the
        engine's own result object. Never load-bearing for control flow.
    """

    energy: Optional[float]
    ok: bool
    engine: str
    engine_version: Optional[str] = None
    failure_mode: Optional[str] = None
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ok:
            if self.energy is None:
                raise ValueError("EngineEnergy(ok=True) requires an energy")
            if self.failure_mode is not None:
                raise ValueError(
                    "EngineEnergy(ok=True) must not carry a failure_mode; "
                    f"got {self.failure_mode!r}"
                )
        elif self.failure_mode is None:
            raise ValueError("EngineEnergy(ok=False) requires a failure_mode")

    def energy_if_ok(self) -> Optional[float]:
        """The energy when usable, else ``None``.

        The narrowing view for callers that only branch on success. A
        non-converged evaluation may still carry its last-cycle energy in
        :attr:`energy` for diagnosis; this method refuses to hand that
        out, because the one thing it must never be is silently consumed
        as a converged value.
        """
        return self.energy if self.ok else None

    @classmethod
    def failed(
        cls,
        engine: str,
        failure_mode: str,
        *,
        engine_version: Optional[str] = None,
        energy: Optional[float] = None,
        **detail: Any,
    ) -> "EngineEnergy":
        """Construct a failed result. ``energy`` is diagnostic only."""
        return cls(
            energy=energy,
            ok=False,
            engine=engine,
            engine_version=engine_version,
            failure_mode=failure_mode,
            detail=detail,
        )


class EnergyEngine(abc.ABC):
    """Compute total energies of crystals and isolated atoms.

    Implementers override the two ``*_energy`` methods, which return a
    provenance-carrying :class:`EngineEnergy`. The ``evaluate_*`` methods
    are concrete float-returning views over them, so a caller that only
    wants a number does not have to unwrap.

    Two views, one abstraction -- not two abstractions. The float view
    exists because objective functions genuinely only need a float or
    ``None``, and it is derived from the rich view rather than
    implemented alongside it, so the two cannot disagree.

    An evaluation that fails returns ``ok=False``; it does not raise.
    Basis optimization walks into infeasible regions routinely (an
    exponent collapses, the overlap goes singular, the queue drops a
    job), and a derivative-free optimizer handles that by scoring the
    point ``inf`` and moving on. An exception would abort a campaign that
    is minutes-per-evaluation and days long.
    """

    #: Short engine name, used in provenance records. Override.
    name: str = "abstract"

    def version(self) -> Optional[str]:
        """Engine version, or ``None`` if it cannot be determined.

        Default is ``None``: an engine that cannot report its version
        says so rather than guessing. Overriding with a fabricated or
        assumed value would defeat the point of recording it.
        """
        return None

    @abc.abstractmethod
    def crystal_energy(
        self,
        basis_text: str,
        structure: Any,
        method: str = "rhf",
    ) -> EngineEnergy:
        """Total energy of a periodic crystal, per unit cell.

        Parameters
        ----------
        basis_text
            CRYSTAL inline basis text (from ``io.emit_crystal``). The
            interchange format between the optimizer and every engine,
            including the non-CRYSTAL ones, which convert on the way in.
        structure
            A :class:`~vibe_basis.io.structures.Structure`.
        method
            ``"rhf"`` / ``"hf"``, or a functional name (``"pw1pw"``,
            ``"pbe"``, ``"r2scan"``, ...).
        """

    @abc.abstractmethod
    def atom_energy(
        self,
        basis_text: str,
        Z: int,
        method: str = "rhf",
        *,
        host: Any = None,
    ) -> EngineEnergy:
        """Total energy of an isolated atom of atomic number *Z*.

        The atom reference convention (spherical vs aspherical, spin
        restricted vs polarised) is the engine's, and it **must** match
        the convention of whatever reference the result is compared
        against. This is the single biggest correctness trap in cohesive
        energies: spin-restricting the Br reference alone moved KBr's
        atomization by 37 kJ/mol (see
        ``handovers/HANDOVER_GPAW_PW_REFERENCE.md``, GPAW-PWREF-002).
        An engine that can vary the convention must record which one it
        used in ``EngineEnergy.detail``.

        *host*, when given, requests a **counterpoise-corrected** atom:
        the atom computed in the ghost basis of that host structure,
        rather than bare. Only passed when ``supports("counterpoise")``.
        The published pob cohesive energies are computed this way, so
        matching them requires it; the cost is that such an energy is
        system-specific and cannot be shared across compounds.
        """

    # -- float views: derived, never overridden ---------------------------

    def evaluate_crystal(
        self,
        basis_text: str,
        structure: Any,
        method: str = "rhf",
    ) -> Optional[float]:
        """Crystal energy in Hartree, or ``None`` on failure."""
        return self.crystal_energy(basis_text, structure, method=method).energy_if_ok()

    def evaluate_atom(
        self,
        basis_text: str,
        Z: int,
        method: str = "rhf",
        *,
        host: Any = None,
    ) -> Optional[float]:
        """Atom energy in Hartree, or ``None`` on failure."""
        return self.atom_energy(
            basis_text, Z, method=method, host=host
        ).energy_if_ok()

    # -- optional capabilities -------------------------------------------
    #
    # A cohesive energy needs more than single points: a relaxed geometry
    # and, for the physical number, a zero-point correction. Not every
    # engine can do either, and an engine that cannot must say so rather
    # than silently return the input geometry (which would look like a
    # converged relaxation) or zero (which would look like a solid with
    # no vibrations).
    #
    # Declared as optional capabilities on the one engine abstraction
    # rather than as separate protocols: the pipeline asks `supports()`
    # and records a skipped stage with its reason, so a partial result
    # is explicit in the output instead of being inferred from a
    # suspiciously round number.

    #: Capability names an engine may declare in :attr:`capabilities`.
    CAPABILITIES = frozenset({"relax", "zero_point", "counterpoise"})

    #: Which of :data:`CAPABILITIES` this engine implements. Override.
    capabilities: frozenset[str] = frozenset()

    def supports(self, capability: str) -> bool:
        """Does this engine implement *capability*?"""
        if capability not in self.CAPABILITIES:
            raise ValueError(
                f"unknown capability {capability!r}; "
                f"known: {sorted(self.CAPABILITIES)}"
            )
        return capability in self.capabilities

    def relax(
        self,
        basis_text: str,
        structure: Any,
        method: str = "rhf",
    ) -> "RelaxedStructure":
        """Relax *structure* (cell + internal coordinates) at *method*.

        Only called when ``supports("relax")``. Default raises, because
        an engine without a relaxer returning the unrelaxed input would
        make every downstream energy silently a fixed-geometry one.
        """
        raise EngineCapabilityError(self.name, "relax")

    def zero_point_energy(
        self,
        basis_text: str,
        structure: Any,
        method: str = "rhf",
    ) -> EngineEnergy:
        """Zero-point energy of *structure*, in Hartree per unit cell.

        Sign convention: **positive**. The pipeline subtracts it from
        the cohesive energy (a vibrating solid is less bound than a
        static one). Only called when ``supports("zero_point")``.
        """
        raise EngineCapabilityError(self.name, "zero_point")


class EngineCapabilityError(NotImplementedError):
    """An engine was asked for a capability it does not declare."""

    def __init__(self, engine: str, capability: str) -> None:
        super().__init__(
            f"engine {engine!r} does not support {capability!r}. "
            f"Check engine.supports({capability!r}) before calling, or pick "
            f"an engine that declares it in `capabilities`."
        )
        self.engine = engine
        self.capability = capability


@dataclass(frozen=True)
class RelaxedStructure:
    """Outcome of a geometry relaxation.

    Attributes
    ----------
    structure
        The relaxed structure, in whatever type the caller passed in.
    ok
        ``True`` iff the relaxation converged. When ``False``,
        :attr:`structure` is whatever the optimizer last held and must
        not be treated as a minimum.
    energy
        Total energy at the relaxed geometry, Hartree per unit cell, if
        the engine reported one.
    engine, engine_version
        Provenance, as on :class:`EngineEnergy`.
    failure_mode
        Why it failed, when ``ok`` is ``False``.
    detail
        Free-form extras: step count, max force, stress residual.
    """

    structure: Any
    ok: bool
    engine: str
    energy: Optional[float] = None
    engine_version: Optional[str] = None
    failure_mode: Optional[str] = None
    detail: dict[str, Any] = field(default_factory=dict)
