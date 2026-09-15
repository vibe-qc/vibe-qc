"""CRYSTAL23 energy engine — the Gaussian fallback.

Submits ``.d12`` decks through a :class:`~vibe_basis.transports.base.Transport`
to an external CRYSTAL binary and parses what comes back. Computes
nothing itself; CLAUDE.md § 10's out-of-process boundary is why this is
driver code rather than a quantum-chemistry dependency.

Role
----
CRYSTAL23 is the **fallback** Gaussian engine, not the target
(``ROADMAP.md`` § 2). It earns its place three ways:

1. **Methodology continuity.** pob-TZVP and its rev2 successors were
   optimized with CRYSTAL09 / CRYSTAL17 under the PT2013 / VO2019
   recipe. Reproducing a published pob number is how the *pipeline*
   gets certified before vibe-qc's number is trusted (M0 gate 1).
2. **The head-to-head.** Beating CRYSTAL23's ``OPTBASIS`` is the
   mission; running it is how the margin gets measured on its own terms.
3. **Unblocking.** Any milestone whose vibe-qc capability has not landed
   can proceed here and be re-certified on ``VibeQcEngine`` later.

This class was previously ``CRYSTAL14Calculator`` in
``python/vibeqc/basis_optimization/calculators.py`` -- tier-2 code that
imported only from vibe-basis, i.e. filed on the wrong side of the
boundary. Moved here in vibe-basis 0.3.0 with the CRYSTAL23 retarget.
"""

from __future__ import annotations

import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..backends.crystal import (
    DEFAULT_CRYSTAL_VERSION,
    emit_input_inline,
    parse_output_file,
)
from ..backends.crystal_atom import (
    GROUND_STATE_MULTIPLICITY,
    emit_input_atom,
    emit_input_atom_counterpoise,
)
from ..engine import EnergyEngine, EngineEnergy
from ..transports.base import Transport, TransportError


@dataclass
class Crystal23Engine(EnergyEngine):
    """Energy engine backed by an external CRYSTAL binary.

    Parameters
    ----------
    transport
        How decks reach a CPU: ``LocalTransport`` (subprocess) or
        ``VqTransport`` (submit to a vibe-queue daemon).
    crystal_wrapper
        Command or path for the CRYSTAL wrapper script on the target
        host.
    cpus, wall_time_s, timeout_s
        Passed through to ``transport.run``.
    workdir_prefix
        Prefix for the per-evaluation staging directory.
    expected_version
        CRYSTAL major version this engine claims to be driving. Parsed
        outputs are checked against it and a mismatch is reported (see
        :attr:`strict_version`), never silently accepted.
    strict_version
        When ``True``, an output whose banner reports a different major
        version fails the evaluation with
        ``failure_mode="version_mismatch"`` rather than returning its
        energy. Default ``False``: a CRYSTAL17 output is perfectly
        readable, and a parity study may legitimately want one. Turn it
        on for a production campaign, where a stray binary on one host
        silently contributing a few points is precisely the failure this
        guards.
    """

    transport: Transport
    crystal_wrapper: str = "crystal"
    cpus: int = 4
    wall_time_s: int = 7200
    timeout_s: float = 86_400.0
    workdir_prefix: str = "calc"
    bsse_nstar: int = 30
    bsse_rmax: float = 10.0
    expected_version: int = DEFAULT_CRYSTAL_VERSION
    strict_version: bool = False

    name: str = field(default="crystal23", init=False)
    capabilities: frozenset = field(
        default=frozenset({"counterpoise"}), init=False
    )

    #: Version seen on the most recent parsed output, if any. Populated
    #: per evaluation because it is a property of the *run*, not of this
    #: object -- the wrapper script on the far end is what decides.
    _observed_version: Optional[int] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.name = f"crystal{self.expected_version}"

    def version(self) -> Optional[str]:
        """Version observed on the last parsed output, or ``None``.

        Deliberately not ``expected_version``: this reports what the
        engine *saw*, and before the first evaluation it has seen
        nothing. Reporting the expectation here would turn a
        configuration value into fake evidence.
        """
        return None if self._observed_version is None else str(self._observed_version)

    def crystal_energy(
        self,
        basis_text: str,
        structure: Any,
        method: str = "rhf",
    ) -> EngineEnergy:
        deck = emit_input_inline(structure, basis_text, method=method)
        if deck is None:
            # emit_input_inline returns None for an unsupported method or
            # a structure it cannot express -- a bad *request*, caught
            # before a job is submitted.
            return EngineEnergy.failed(
                self.name,
                "emit_failed",
                engine_version=self.version(),
                method=method,
                structure=getattr(structure, "name", None),
            )
        return self._run_deck(deck, structure.name, method=method)

    def atom_energy(
        self,
        basis_text: str,
        Z: int,
        method: str = "rhf",
        *,
        host: Any = None,
    ) -> EngineEnergy:
        # CRYSTAL's atom reference here is a single atom in a large P1
        # box (see backends.crystal_atom), aspherical and spin-polarised
        # for an open shell (SYMMREMO + UHF/SPIN + SPINLOCK). The
        # convention is recorded in detail rather than assumed, because
        # a cohesive energy is only comparable to a reference computed
        # under the *same* atom convention: spin-restricting one
        # reference moved KBr's atomization by 37 kJ/mol.
        n_unpaired = GROUND_STATE_MULTIPLICITY.get(Z, 1) - 1
        shell = "aspherical_spin_polarised" if n_unpaired else "closed_shell"

        if host is None:
            deck = emit_input_atom(Z, basis_text, method=method)
            name = f"Z{Z}"
            reference = f"p1_box_{shell}"
            extra: dict[str, Any] = {}
        else:
            # Counterpoise: the atom in the ghost basis of its own
            # crystal, which is how the published pob values were
            # computed. Its energy is therefore host-specific.
            index = self._atom_index(host, Z)
            deck = emit_input_atom_counterpoise(
                host,
                index,
                basis_text,
                method=method,
                nstar=self.bsse_nstar,
                rmax=self.bsse_rmax,
            )
            if deck is None:
                return EngineEnergy.failed(
                    self.name,
                    "emit_failed",
                    Z=Z,
                    host=getattr(host, "name", None),
                    method=method,
                )
            name = f"{getattr(host, 'name', 'host')}_bsse_Z{Z}"
            reference = f"atombsse_{shell}"
            extra = {
                "host": getattr(host, "name", None),
                "bsse_nstar": self.bsse_nstar,
                "bsse_rmax": self.bsse_rmax,
                "atom_index": index,
            }

        return self._run_deck(
            deck,
            name,
            method=method,
            Z=Z,
            atom_reference=reference,
            n_unpaired=n_unpaired,
            **extra,
        )

    @staticmethod
    def _atom_index(host: Any, Z: int) -> int:
        """1-based label of the first atom of element *Z* in the cell.

        CRYSTAL's ``IAT`` indexes the reference cell as written. Picking
        the first match is right for the ordered binaries in the pob
        test sets, where every atom of an element is symmetry
        equivalent; a structure with inequivalent sites of the same
        element would need the caller to choose.
        """
        for i, atom in enumerate(host.crystal_asymm_unit, start=1):
            if atom.Z == Z:
                return i
        raise ValueError(
            f"element Z={Z} does not appear in the asymmetric unit of "
            f"{getattr(host, 'name', host)!r}"
        )

    # ------------------------------------------------------------------

    def _run_deck(
        self,
        deck: str,
        name: str,
        *,
        method: str,
        **detail: Any,
    ) -> EngineEnergy:
        """Stage a deck, submit it, parse the output, report."""
        wd = Path(f"{self.workdir_prefix}_{name}")
        if wd.exists():
            shutil.rmtree(wd)
        wd.mkdir(parents=True)

        d12_name = f"{name}.d12"
        (wd / d12_name).write_text(deck)

        # Unique label so concurrent evaluations of the same system (a
        # topology scan runs many at once) do not collide in the queue.
        label = f"calc/{name}/{uuid.uuid4().hex[:6]}"

        try:
            job = self.transport.run(
                wd,
                command=[self.crystal_wrapper, d12_name],
                dest=wd / "fetched",
                cpus=self.cpus,
                wall_time_s=self.wall_time_s,
                label=label,
                timeout_s=self.timeout_s,
            )
        except TransportError as exc:
            return EngineEnergy.failed(
                self.name, "transport_error", error=str(exc), label=label, **detail
            )

        try:
            parsed = parse_output_file(job.output_dir / f"{name}.out")
        except (FileNotFoundError, OSError) as exc:
            # The job "finished" but produced no readable output: a
            # fetch failure or a crash before CRYSTAL opened the file.
            return EngineEnergy.failed(
                self.name, "no_output_file", error=str(exc), label=label, **detail
            )

        self._observed_version = parsed.crystal_version
        seen = self.version()
        detail = {
            **detail,
            "method": method,
            "label": label,
            "last_cycle": parsed.last_cycle,
            "scf_method": parsed.method,
            "crystal_version": parsed.crystal_version,
        }

        if not parsed.ok:
            return EngineEnergy.failed(
                self.name,
                parsed.failure_mode or "unknown",
                engine_version=seen,
                energy=parsed.energy,  # diagnostic only; energy_if_ok() hides it
                **detail,
            )

        if self.strict_version and parsed.version_matches(self.expected_version) is False:
            return EngineEnergy.failed(
                self.name,
                "version_mismatch",
                engine_version=seen,
                energy=parsed.energy,
                expected_version=self.expected_version,
                **detail,
            )

        return EngineEnergy(
            energy=parsed.energy,
            ok=True,
            engine=self.name,
            engine_version=seen,
            detail=detail,
        )
