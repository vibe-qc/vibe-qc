"""Explicit spin-state contract of the neutral Γ-CCM producers (#519, RG-14).

A :class:`~vibeqc.periodic.ccm.system.CCMSystem` replicates one unit cell
over the Born-von Karman torus. The electron count of the torus is fixed by
the cell, ``N_c N_e``; the **spin sector** is not. Two conventions were in
use before this module existed, one per producer:

* the real-Gamma supercell drivers read the supercell multiplicity that
  ``CCMSystem`` derives: an explicit cell multiplicity ``>= 3`` replicates
  ferromagnetically, every other cell is paired by parity, so a doublet cell
  on an even torus runs as a **supercell singlet** and on an odd torus as a
  supercell doublet (one unpaired electron shared by ``N_c`` cells);
* the neutral-Bloch producer hands the unit cell to the multi-k drivers,
  whose state is the cell's ``(n_alpha, n_beta)`` **replicated uniformly** at
  every k point, so the same doublet cell runs with ``N_c`` unpaired
  electrons, aligned.

On the Li doublet cube at ``(2,1,1)`` these are different states of one
operator (the operator identity of :mod:`vibeqc.periodic.ccm.source` is
equal), and their energies differ by millihartrees. That is not a failure of
the representation identity: Theorem 2 of the theory notes (representation
equivalence on the finite torus) maps block-circulant supercell densities
with ``D_sigma S D_sigma = D_sigma`` and ``Tr[D_sigma S] = N_c n_sigma`` onto
the Bloch state space, so the Bloch producer can only represent sectors whose
per-spin counts are ``N_c`` times an integer pair, while the supercell
producer also admits the non-uniform sectors. The same-H comparison therefore
needs the same executed per-spin counts on both sides, besides the same
finite source; :func:`assert_compatible_spin_state` checks the former and
:func:`assert_comparable_neutral_results` both.

The contract is one keyword, ``spin_state=``, on :class:`CCMSystem` and the
producers (``aiccm_spin_state=`` on the runner):

``"legacy"`` (the default)
    Each producer keeps its historical convention above, unchanged number
    for number. The executed state is recorded on every result and the
    comparison helpers refuse a mismatched pair instead of averaging it.
``"per-cell"``
    The unit cell's ``(n_alpha, n_beta)`` is replicated to every cell with
    spins aligned: ``N_c (multiplicity - 1)`` unpaired electrons on the
    torus. Both producers execute this one sector, on every mesh, so a
    doublet cell is a uniform ferromagnetic torus at ``(2,1,1)``, ``(3,1,1)``
    and ``(2,2,1)`` alike.
an integer ``M``
    The **total** multiplicity of the torus, ``2S + 1`` with ``S`` the total
    spin projection ``(n_alpha - n_beta) / 2``. The real-Gamma producer runs
    it as given; the Bloch producer runs it only when ``M - 1`` is a multiple
    of ``N_c`` (then it is the uniform sector above) and refuses otherwise.
    A total multiplicity fixes ``n_alpha - n_beta`` and nothing else: it does
    not select an antiferromagnetic or any other broken-symmetry arrangement
    of local moments, which needs a per-cell seed the producers do not take.

Every resolved state records the requested policy, the per-cell and torus
counts, the executed arrangement and whether the sector is uniform per cell,
so manifests, the QVF record and restart densities carry what ran.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from typing import Optional

__all__ = [
    "CCMSpinState",
    "DEGENERATE_FRONTIER_TOL",
    "INTEGER_OCCUPATION_TOL",
    "OccupationRecord",
    "assert_integer_occupations",
    "bloch_occupation_record",
    "occupation_record_of",
    "supercell_occupation_record",
    "SPIN_STATE_POLICIES",
    "assert_comparable_neutral_results",
    "assert_compatible_spin_state",
    "bloch_executed_state",
    "normalize_spin_state_request",
    "require_closed_shell_state",
    "resolve_ccm_spin_state",
    "spin_state_of",
    "stamp_spin_state",
    "supercell_spin_counts",
]

#: The named policies; an integer total multiplicity is the third form.
SPIN_STATE_POLICIES = ("legacy", "per-cell")

_ARRANGEMENTS = ("parity-paired", "uniform", "explicit-total")


@dataclass(frozen=True)
class CCMSpinState:
    """The spin sector a neutral producer was asked for and executed.

    Counts are totals on the torus unless named ``per_cell``. ``multiplicity``
    is the torus total ``n_alpha - n_beta + 1``. ``arrangement`` names how the
    sector was built: ``"parity-paired"`` (the legacy supercell rule),
    ``"uniform"`` (the cell's counts replicated with spins aligned) or
    ``"explicit-total"`` (a requested torus multiplicity). ``uniform`` is
    whether the sector is ``n_cells`` copies of one integer pair, the
    condition for the Bloch producer to represent it; ``per_cell`` is that
    pair, or ``None``.
    """

    policy: str
    requested: object
    n_cells: int
    charge_per_cell: int
    n_electrons_per_cell: int
    unit_multiplicity: int
    n_electrons: int
    n_alpha: int
    n_beta: int
    arrangement: str
    producer: str = ""

    def __post_init__(self):
        if self.arrangement not in _ARRANGEMENTS:
            raise ValueError(
                f"CCMSpinState: arrangement must be one of {_ARRANGEMENTS}, "
                f"got {self.arrangement!r}")
        if self.n_alpha < 0 or self.n_beta < 0:
            raise ValueError("CCMSpinState: negative spin counts")
        if self.n_alpha + self.n_beta != self.n_electrons:
            raise ValueError(
                f"CCMSpinState: n_alpha + n_beta = {self.n_alpha + self.n_beta} "
                f"differs from the torus electron count {self.n_electrons}")

    @property
    def multiplicity(self) -> int:
        """Total torus multiplicity ``n_alpha - n_beta + 1``."""
        return int(self.n_alpha - self.n_beta + 1)

    @property
    def n_unpaired(self) -> int:
        return int(self.n_alpha - self.n_beta)

    @property
    def closed_shell(self) -> bool:
        """Whether the sector is spin-compensated (``n_alpha == n_beta``)."""
        return self.n_alpha == self.n_beta

    @property
    def uniform(self) -> bool:
        return (self.n_alpha % self.n_cells == 0
                and self.n_beta % self.n_cells == 0)

    @property
    def per_cell(self) -> Optional[tuple[int, int]]:
        if not self.uniform:
            return None
        return (self.n_alpha // self.n_cells, self.n_beta // self.n_cells)

    @property
    def per_cell_multiplicity(self) -> Optional[int]:
        pair = self.per_cell
        return None if pair is None else pair[0] - pair[1] + 1

    def executed_by(self, producer: str, *, n_alpha: int, n_beta: int,
                    arrangement: str) -> "CCMSpinState":
        """The state a producer executed, with the request carried along."""
        return replace(self, producer=str(producer), n_alpha=int(n_alpha),
                       n_beta=int(n_beta), arrangement=str(arrangement))

    def as_dict(self) -> dict:
        out = asdict(self)
        out["requested"] = (self.requested if isinstance(self.requested, (int, str))
                            else str(self.requested))
        # Manifest-safe (TOML has no null): a non-uniform sector records an
        # empty per-cell pair.
        out.update(
            multiplicity=self.multiplicity, n_unpaired=self.n_unpaired,
            closed_shell=self.closed_shell, uniform=self.uniform,
            per_cell=([] if self.per_cell is None else list(self.per_cell)),
            schema="vibeqc.ccm-spin-state.v1")
        return out

    @classmethod
    def from_dict(cls, payload: dict) -> "CCMSpinState":
        fields = {k: payload[k] for k in (
            "policy", "requested", "n_cells", "charge_per_cell",
            "n_electrons_per_cell", "unit_multiplicity", "n_electrons",
            "n_alpha", "n_beta", "arrangement")}
        fields["producer"] = str(payload.get("producer", "") or "")
        return cls(**fields)

    def describe(self) -> str:
        pair = self.per_cell
        per = ("not uniform per cell" if pair is None
               else f"({pair[0]}, {pair[1]}) per cell")
        return (f"{self.arrangement}: n_alpha={self.n_alpha}, n_beta={self.n_beta} "
                f"on {self.n_cells} cell(s), multiplicity {self.multiplicity}, {per}")


def _unit_cell_counts(n_electrons_per_cell: int, unit_multiplicity: int, *,
                      who: str) -> tuple[int, int, int]:
    """``(n_alpha, n_beta, multiplicity)`` of the unit cell.

    A ``PeriodicSystem`` defaults to multiplicity 1; on an odd-electron cell
    that default is not a request but a construction default, and it is
    floored to the doublet exactly as the legacy supercell rule and the
    runner's ``_system_with_valid_default_multiplicity`` do. Any explicit
    multiplicity must fit the count and parity.
    """
    n_e = int(n_electrons_per_cell)
    mult = int(unit_multiplicity)
    if mult < 1:
        raise ValueError(f"{who}: the unit-cell multiplicity must be >= 1, got {mult}")
    if mult == 1 and n_e % 2 == 1:
        mult = 2
    n_unp = mult - 1
    if (n_e - n_unp) % 2 != 0 or n_unp > n_e:
        raise ValueError(
            f"{who}: unit-cell electron count {n_e} and multiplicity {mult} "
            "are incompatible (n_alpha/n_beta non-integer).")
    n_a = (n_e + n_unp) // 2
    return n_a, n_e - n_a, mult


def _legacy_supercell_multiplicity(n_electrons_per_cell: int, unit_multiplicity: int,
                                   n_cells: int, *, who: str) -> int:
    """The historical ``CCMSystem._build_supercell`` rule, unchanged."""
    n_elec = int(n_electrons_per_cell) * int(n_cells)
    n_unpaired_unit = int(unit_multiplicity) - 1
    if n_unpaired_unit >= 2:
        n_elec_unit = int(n_electrons_per_cell)
        if ((n_elec_unit - n_unpaired_unit) % 2 != 0
                or n_unpaired_unit > n_elec_unit):
            raise ValueError(
                f"{who}: unit-cell electron count {n_elec_unit} and "
                f"multiplicity {unit_multiplicity} are "
                "incompatible (n_alpha/n_beta non-integer).")
        return n_unpaired_unit * int(n_cells) + 1
    return 1 if (n_elec % 2 == 0) else 2


def normalize_spin_state_request(spin_state, *, who: str = "spin_state"):
    """``"legacy"``, ``"per-cell"`` or a positive integer, from any spelling.

    The form of a request is mesh independent and is checked here; whether
    the counts fit a given torus is :func:`resolve_ccm_spin_state`'s job.
    """
    if isinstance(spin_state, CCMSpinState):
        return spin_state.requested
    if isinstance(spin_state, bool):
        raise ValueError(f"{who}: spin_state must be 'legacy', 'per-cell' or an "
                         f"integer multiplicity, got {spin_state!r}")
    if isinstance(spin_state, str):
        key = spin_state.strip().lower().replace("_", "-")
        if key in SPIN_STATE_POLICIES:
            return key
        try:
            spin_state = int(key)
        except ValueError:
            raise ValueError(
                f"{who}: spin_state must be 'legacy', 'per-cell' or an integer "
                f"total multiplicity, got {spin_state!r}") from None
    try:
        mult_total = int(spin_state)
    except (TypeError, ValueError):
        raise ValueError(
            f"{who}: spin_state must be 'legacy', 'per-cell' or an integer "
            f"total multiplicity, got {spin_state!r}") from None
    if mult_total < 1:
        raise ValueError(f"{who}: a total multiplicity must be >= 1, got {mult_total}")
    return mult_total


def resolve_ccm_spin_state(unit_system, nrep, spin_state="legacy", *,
                           who: str = "CCMSystem") -> CCMSpinState:
    """Resolve the torus spin sector of ``unit_system`` replicated over ``nrep``.

    ``spin_state`` is ``"legacy"``, ``"per-cell"`` or an integer total torus
    multiplicity (see the module docstring). Impossible requests (parity,
    count, a multiplicity below 1) are refused here, before any basis or
    integral work, independent of the mesh size. The returned state is what
    the real-Gamma supercell producer executes; under ``"legacy"`` the Bloch
    producer executes the unit-cell convention instead and records that.
    """
    spin_state = normalize_spin_state_request(spin_state, who=who)
    n1, n2, n3 = (int(n) for n in nrep)
    n_cells = n1 * n2 * n3
    charge = int(unit_system.charge)
    n_e_cell = int(sum(int(a.Z) for a in unit_system.unit_cell)) - charge
    if n_e_cell < 0:
        raise ValueError(f"{who}: the unit cell has a negative electron count")
    unit_mult = int(unit_system.multiplicity)
    n_elec = n_e_cell * n_cells
    common = dict(n_cells=n_cells, charge_per_cell=charge,
                  n_electrons_per_cell=n_e_cell, unit_multiplicity=unit_mult,
                  n_electrons=n_elec)

    if isinstance(spin_state, str):
        key = spin_state
        if key == "legacy":
            mult_sc = _legacy_supercell_multiplicity(
                n_e_cell, unit_mult, n_cells, who=who)
            n_unp = mult_sc - 1
            n_a = (n_elec + n_unp) // 2
            return CCMSpinState(
                policy="legacy", requested="legacy", n_alpha=n_a,
                n_beta=n_elec - n_a,
                arrangement=("uniform" if unit_mult - 1 >= 2 else "parity-paired"),
                **common)
        if key == "per-cell":
            n_a_cell, n_b_cell, _ = _unit_cell_counts(n_e_cell, unit_mult, who=who)
            return CCMSpinState(
                policy="per-cell", requested="per-cell",
                n_alpha=n_a_cell * n_cells, n_beta=n_b_cell * n_cells,
                arrangement="uniform", **common)
    mult_total = int(spin_state)
    n_unp = mult_total - 1
    if n_unp > n_elec:
        raise ValueError(
            f"{who}: total multiplicity {mult_total} needs {n_unp} unpaired "
            f"electrons but the torus holds {n_elec}.")
    if (n_elec - n_unp) % 2 != 0:
        raise ValueError(
            f"{who}: total multiplicity {mult_total} and the torus electron "
            f"count {n_elec} are incompatible (n_alpha/n_beta non-integer).")
    n_a = (n_elec + n_unp) // 2
    return CCMSpinState(
        policy="explicit", requested=mult_total, n_alpha=n_a,
        n_beta=n_elec - n_a, arrangement="explicit-total", **common)


def bloch_executed_state(state: CCMSpinState, *, who: str) -> CCMSpinState:
    """The sector the neutral-Bloch producer executes for a resolved state.

    Under ``"legacy"`` that is the unit-cell convention of the multi-k
    drivers (the cell's counts at every k point), whatever the supercell rule
    paired. Under ``"per-cell"`` and an explicit total it is the resolved
    sector itself, which the Bloch representation holds only when it is
    uniform per cell; a non-uniform sector is refused by name.
    """
    if state.policy == "legacy":
        n_a_cell, n_b_cell, _ = _unit_cell_counts(
            state.n_electrons_per_cell, state.unit_multiplicity, who=who)
        return state.executed_by(
            "neutral-bloch", n_alpha=n_a_cell * state.n_cells,
            n_beta=n_b_cell * state.n_cells, arrangement="uniform")
    if not state.uniform:
        raise ValueError(
            f"{who}: the requested spin sector ({state.describe()}) is not "
            f"uniform per cell: {state.n_unpaired} unpaired electron(s) cannot "
            f"be shared by {state.n_cells} cells at every k point. The Bloch "
            "representation holds only sectors whose per-spin counts are "
            "N_c times an integer pair (Theorem 2, representation "
            "equivalence); choose spin_state='per-cell' or a total "
            f"multiplicity M with (M - 1) a multiple of {state.n_cells}, or "
            "run this sector on the real-Gamma producer alone.")
    return state.executed_by("neutral-bloch", n_alpha=state.n_alpha,
                             n_beta=state.n_beta, arrangement=state.arrangement)


def spin_state_of(result) -> Optional[CCMSpinState]:
    """The executed spin state of a producer or runner result, or ``None``."""
    if isinstance(result, CCMSpinState):
        return result
    record = getattr(result, "spin_state", None)
    if isinstance(record, CCMSpinState):
        return record
    if isinstance(record, dict):
        return CCMSpinState.from_dict(record)
    convention = (getattr(result, "real_gamma", None)
                  or getattr(result, "neutral_bloch", None))
    payload = getattr(convention, "spin_state", None)
    if isinstance(payload, CCMSpinState):
        return payload
    if isinstance(payload, dict) and payload:
        return CCMSpinState.from_dict(payload)
    if isinstance(payload, str) and payload:
        return CCMSpinState.from_dict(json.loads(payload))
    return None


_COMPARED = ("n_cells", "charge_per_cell", "n_electrons", "n_alpha", "n_beta")


def assert_compatible_spin_state(items, *, context: str = "comparison") -> CCMSpinState:
    """Refuse a comparison of results that executed different spin sectors.

    Compares the executed torus counts (cells, charge, electrons, alpha and
    beta) of every item; the policy that produced them and the producer are
    not compared, so a ``"per-cell"`` real-Gamma run and a ``"legacy"`` Bloch
    run of one uniform sector are compatible. Items without a state record
    are refused: a count that is not recorded cannot be matched. Returns the
    first state.
    """
    states = []
    for index, item in enumerate(items):
        state = spin_state_of(item)
        if state is None:
            raise ValueError(
                f"{context}: item {index} carries no spin-state record "
                "(spin_state); its executed sector cannot be matched")
        states.append(state)
    if not states:
        raise ValueError(f"{context}: nothing to compare")
    first = states[0]
    differing = []
    for index, state in enumerate(states[1:], start=1):
        for key in _COMPARED:
            a, b = getattr(first, key), getattr(state, key)
            if a != b:
                differing.append(f"{key}: item 0 {a!r} vs item {index} {b!r}")
    if differing:
        raise ValueError(
            f"{context}: the results executed different spin sectors of the "
            "torus; these are different states, not a tolerance question. "
            f"Item 0 is {first.describe()}; differing fields: "
            + "; ".join(differing))
    return first


def assert_comparable_neutral_results(items, *, context: str = "comparison"):
    """Both halves of a same-state, same-operator comparison.

    The operator identity (:func:`~vibeqc.periodic.ccm.source.assert_matched_neutral_source`)
    and the executed spin sector are checked separately and both must hold:
    one operator in two sectors is as incomparable as two operators in one
    sector. Both sides must also be determinants
    (:func:`assert_integer_occupations`, RG-15). Returns
    ``(source_identity, spin_state)``.
    """
    from .source import assert_matched_neutral_source

    identity = assert_matched_neutral_source(items, context=context)
    state = assert_compatible_spin_state(items, context=context)
    assert_integer_occupations(items, context=context)
    return identity, state


def supercell_spin_counts(ccm, *, who: str) -> tuple[int, int, Optional[CCMSpinState]]:
    """``(n_alpha, n_beta, state)`` the supercell drivers execute.

    Read off ``ccm.spin_state`` when the cluster carries one. The direct
    drivers are also exercised on synthetic stand-ins without a resolved
    state; those derive the counts from the stand-in's supercell
    charge/multiplicity as the drivers always did and carry no record.
    """
    state = getattr(ccm, "spin_state", None)
    if isinstance(state, CCMSpinState):
        return int(state.n_alpha), int(state.n_beta), state
    n_elec = int(ccm.supercell.n_electrons())
    mult = int(ccm.supercell.multiplicity)
    n_unpaired = mult - 1
    if (n_elec - n_unpaired) % 2 != 0 or n_unpaired > n_elec:
        raise ValueError(
            f"{who}: cluster electron count {n_elec} and multiplicity {mult} "
            "are incompatible (n_alpha/n_beta non-integer).")
    n_alpha = (n_elec + n_unpaired) // 2
    return n_alpha, n_elec - n_alpha, None


def require_closed_shell_state(ccm, *, who: str) -> Optional[CCMSpinState]:
    """Refuse a closed-shell driver on a cluster whose resolved sector is open.

    Returns the state (or ``None`` on a stand-in). The closed-shell loops
    occupy ``n_elec // 2`` pairs and would silently run a different sector.
    """
    state = getattr(ccm, "spin_state", None)
    if isinstance(state, CCMSpinState) and not state.closed_shell:
        raise ValueError(
            f"{who}: the cluster's resolved spin sector is open shell "
            f"({state.describe()}); a closed-shell driver would occupy "
            f"{state.n_electrons // 2} pairs instead. Use the unrestricted "
            "sibling, or spin_state='legacy' / a singlet request.")
    return state if isinstance(state, CCMSpinState) else None


def stamp_spin_state(res, state, producer: str):
    """Record the executed sector on a producer result (``None`` on a stand-in)."""
    res.spin_state = (None if state is None
                      else state.executed_by(producer, n_alpha=state.n_alpha,
                                             n_beta=state.n_beta,
                                             arrangement=state.arrangement))
    return res


# ----------------------------------------------------------------------------
# Occupation identity (#519, RG-15)
#
# The sector fixes the per-spin counts; it does not fix how they are
# distributed over the levels. The real-Gamma supercell loop occupies an
# integer determinant (aufbau on the torus spectrum), the multi-k driver's
# zero-temperature fill gives every member of a frontier-degenerate group the
# same fraction (the symmetry-preserving ensemble limit). A fractional Bloch
# state has no determinant on the torus, so the two producers then evaluate
# different states of one sector, and a same-H row is refused by name before
# any fold rather than detected afterwards.

#: Largest departure from an integer an occupation may show and still count
#: as integer: the multi-k fill's own degeneracy tolerance is nanohartrees in
#: energy, which moves no occupation; a fraction below this is rounding.
INTEGER_OCCUPATION_TOL = 1.0e-8
#: A frontier closer than this (Ha) is recorded as degenerate: the
#: determinant chosen there is one of several, and the row says so.
DEGENERATE_FRONTIER_TOL = 1.0e-6


@dataclass(frozen=True)
class OccupationRecord:
    """How a producer distributed the sector's electrons over its levels.

    ``integer`` is whether every occupation is 0 or the spin capacity to
    :data:`INTEGER_OCCUPATION_TOL`; ``max_fractional_deviation`` the largest
    departure. ``counts_per_k`` are the per-k ``(alpha, beta)`` sums of the
    Bloch producer (``None`` on the supercell producer, which has one block).
    ``frontier_gap`` is LUMO minus HOMO of the returned spectrum in Ha and
    ``degenerate_frontier`` whether it is below :data:`DEGENERATE_FRONTIER_TOL`.
    """

    producer: str
    integer: bool
    max_fractional_deviation: float
    n_blocks: int
    counts_per_k: Optional[tuple] = None
    frontier_gap: Optional[float] = None
    degenerate_frontier: bool = False

    def as_dict(self) -> dict:
        out = asdict(self)
        out["counts_per_k"] = ([] if self.counts_per_k is None
                               else [list(c) for c in self.counts_per_k])
        out["frontier_gap"] = ("none" if self.frontier_gap is None
                               else float(self.frontier_gap))
        out["schema"] = "vibeqc.ccm-occupation.v1"
        return out

    @classmethod
    def from_dict(cls, payload: dict) -> "OccupationRecord":
        counts = payload.get("counts_per_k")
        gap = payload.get("frontier_gap")
        return cls(
            producer=str(payload.get("producer", "")),
            integer=bool(payload["integer"]),
            max_fractional_deviation=float(payload["max_fractional_deviation"]),
            n_blocks=int(payload.get("n_blocks", 1)),
            counts_per_k=(None if not counts else tuple(tuple(c) for c in counts)),
            frontier_gap=(None if gap in (None, "none") else float(gap)),
            degenerate_frontier=bool(payload.get("degenerate_frontier", False)))

    def describe(self) -> str:
        kind = ("integer" if self.integer
                else f"fractional (largest departure {self.max_fractional_deviation:.3g})")
        gap = ("" if self.frontier_gap is None
               else f", frontier gap {self.frontier_gap:.3e} Ha"
               + (" (degenerate)" if self.degenerate_frontier else ""))
        return f"{self.producer}: {kind} occupations over {self.n_blocks} block(s){gap}"


def _fractional_deviation(occupations, capacity):
    """Largest departure of ``occupations`` from ``{0, capacity}``."""
    import numpy as np

    worst = 0.0
    for occ in occupations:
        o = np.asarray(occ, dtype=float).reshape(-1)
        if o.size:
            worst = max(worst, float(np.max(np.minimum(np.abs(o), np.abs(o - capacity)))))
    return worst


def bloch_occupation_record(raw, *, open_shell: bool) -> Optional[OccupationRecord]:
    """The occupation record of a multi-k driver result."""
    import numpy as np

    if open_shell:
        occ_a = getattr(raw, "occupations_alpha", None)
        occ_b = getattr(raw, "occupations_beta", None)
        if occ_a is None or occ_b is None:
            return None
        sets = [(occ_a, getattr(raw, "mo_energies_alpha", None), 1.0),
                (occ_b, getattr(raw, "mo_energies_beta", None), 1.0)]
    else:
        occ = getattr(raw, "occupations", None)
        if occ is None:
            return None
        sets = [(occ, getattr(raw, "mo_energies", None), 2.0)]
    deviation = max(_fractional_deviation(o, cap) for o, _, cap in sets)
    integer = deviation <= INTEGER_OCCUPATION_TOL
    n_blocks = len(sets[0][0])
    counts = []
    for k in range(n_blocks):
        counts.append(tuple(float(np.sum(np.asarray(o[k], dtype=float)))
                            for o, _, _ in sets))
    # The spin-resolved frontier of an integer fill: the highest occupied
    # and the lowest empty level over both spins, as ccm_homo_lumo_gap
    # defines it. An ensemble has no gap to report.
    gap = None
    if integer and all(e is not None for _, e, _ in sets):
        gap = _spin_resolved_gap(sets)
    return OccupationRecord(
        producer="neutral-bloch", integer=integer,
        max_fractional_deviation=deviation, n_blocks=n_blocks,
        counts_per_k=tuple(counts), frontier_gap=gap,
        degenerate_frontier=(gap is not None and gap < DEGENERATE_FRONTIER_TOL))


def _spin_resolved_gap(sets):
    import numpy as np

    homo, lumo = -np.inf, np.inf
    for occ_k, eps_k, cap in sets:
        for occ, eps in zip(occ_k, eps_k):
            e = np.sort(np.asarray(np.real(eps), dtype=float).reshape(-1))
            n = int(round(float(np.sum(np.asarray(occ, dtype=float))) / cap))
            if n > 0:
                homo = max(homo, float(e[n - 1]))
            if n < e.size:
                lumo = min(lumo, float(e[n]))
    if not (np.isfinite(homo) and np.isfinite(lumo)):
        return None
    return lumo - homo


def supercell_occupation_record(res, n_alpha: int, n_beta: int) -> OccupationRecord:
    """The occupation record of a real-Gamma supercell result (integer by
    construction: the loops occupy aufbau determinants)."""
    import numpy as np

    # The UHF result names its spectra per spin; the open-shell KS result
    # keeps the alpha spectrum on ``mo_energies`` beside ``mo_energies_beta``.
    eb = getattr(res, "mo_energies_beta", None)
    if eb is not None:
        ea = getattr(res, "mo_energies_alpha", None)
        if ea is None:
            ea = res.mo_energies
        sets = [(np.ones(n_alpha), ea, 1.0), (np.ones(n_beta), eb, 1.0)]
        gap = _spin_resolved_gap([([o], [e], c) for o, e, c in sets])
    else:
        gap = _spin_resolved_gap([([np.full(n_alpha, 2.0)], [res.mo_energies], 2.0)])
    return OccupationRecord(
        producer="real-gamma", integer=True, max_fractional_deviation=0.0,
        n_blocks=1, counts_per_k=None, frontier_gap=gap,
        degenerate_frontier=(gap is not None and gap < DEGENERATE_FRONTIER_TOL))


def occupation_record_of(result) -> Optional[OccupationRecord]:
    """The occupation record of a producer or runner result, or ``None``."""
    if isinstance(result, OccupationRecord):
        return result
    record = getattr(result, "occupation", None)
    if isinstance(record, OccupationRecord):
        return record
    if isinstance(record, dict):
        return OccupationRecord.from_dict(record)
    convention = (getattr(result, "real_gamma", None)
                  or getattr(result, "neutral_bloch", None))
    payload = getattr(convention, "occupation", None)
    if isinstance(payload, OccupationRecord):
        return payload
    if isinstance(payload, dict) and payload:
        return OccupationRecord.from_dict(payload)
    return None


def assert_integer_occupations(items, *, context: str = "comparison"):
    """Refuse a comparison that includes an ensemble (fractional) state.

    Every item must carry an occupation record with integer occupations; a
    fractional record names its largest departure. Degenerate frontiers are
    allowed (a determinant was chosen) but the returned records say so.
    Returns the records.
    """
    records = []
    for index, item in enumerate(items):
        record = occupation_record_of(item)
        if record is None:
            raise ValueError(
                f"{context}: item {index} carries no occupation record "
                "(occupation); whether it is a determinant cannot be matched")
        if not record.integer:
            raise ValueError(
                f"{context}: item {index} is an ensemble state, not a "
                f"determinant ({record.describe()}). The multi-k fill shares "
                "a frontier-degenerate group equally; no supercell "
                "determinant corresponds to it, so the two producers evaluate "
                "different states of one sector. A same-Hamiltonian row needs "
                "integer occupations on both sides (a gapped frontier, or a "
                "fixed per-k fill the shared multi-k driver does not offer).")
        records.append(record)
    if not records:
        raise ValueError(f"{context}: nothing to compare")
    return records
