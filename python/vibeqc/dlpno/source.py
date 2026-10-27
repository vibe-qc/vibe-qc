"""The correlation-source contract the DLPNO engine reads.

Design record: ``docs/design_dlpno_correlation_source.md``. The engine,
:func:`~vibeqc.dlpno.ccsd_local_solver.run_local_dlpno_ccsd` and
:func:`~vibeqc.dlpno.uccsd_local_solver.run_local_dlpno_uccsd`, needs a
reference (MO coefficients, AO Fock matrices, energy, electron count), the AO
overlap, three-centre factors ``B[P, p, q]`` through ``mo_transform``, and,
on a crystal, the distance function its pair screens must use. A
:class:`CorrelationSource` carries exactly that, plus the conventions the
reference and the factors were built under and an identity: the SHA-256 of
the provenance record that defines them. The engine records the identity on
its result, and ``run_job`` writes it to the ``.system`` manifest and the
structured log, so a correlation energy names the Hamiltonian it was
computed on.

This module ships three adapters: :func:`molecular_source` (increment A),
:func:`gamma_gdf_source` for a 3-D GDF reference at the Gamma point with
its own retained fit (increment B1), and :func:`bvk_supercell_source`,
which folds a Gamma-centred k-mesh reference and its fit to the real
Born-von Karman supercell (increment B2). The periodic adapters apply the
same-source rule and refuse BIPOLE and symmetry-reduced references;
:class:`CorrelationSourceError` is the exception they raise.
Both periodic adapters also accept KUHF (increment C), preserving its
independent alpha/beta orbital and Fock spaces on the shared AO frame.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from .translation_operators import CyclicAOOperator

#: Version of the provenance record :func:`source_identity` hashes. Bump it
#: when a field is added or its meaning changes, so identities stay
#: comparable only within one contract version.
CONTRACT_VERSION = 2

_REFERENCE_KINDS = ("rhf", "uhf", "rohf")
_REAL_ORBITAL_DEGENERACY_TOL = 1e-8


class CorrelationSourceError(ValueError):
    """A reference or factor set the engine refuses, with the reason."""


@runtime_checkable
class ThreeCenterFactors(Protocol):
    """Three-centre factors over a real orthonormal auxiliary frame.

    ``mo_transform(C_left, C_right)`` returns ``B[P, p, q]`` with
    ``(pq|rs) = sum_P B[P, p, q] B[P, r, s]``. The molecular
    :class:`~vibeqc.density_fitting.DensityFitting` and the periodic Gamma
    tensor of :mod:`vibeqc.periodic_df` both satisfy it.
    """

    def mo_transform(self, C_left, C_right=None): ...


@dataclass(frozen=True)
class SourceConventions:
    """Conventions the reference and the factors were built under.

    ``exchange_q0`` names the exchange zero-mode convention of a periodic
    reference (``"molecular"`` for molecules). ``same_source`` is true only
    when reference and factors come from one Hamiltonian; the molecular
    adapter sets it false because the SCF fits J and K with the JK auxiliary
    basis while correlation uses the RI one.
    """

    exchange_q0: str = "molecular"
    kmesh: Optional[tuple[int, int, int]] = None
    bloch_normalisation: str = "molecular"
    same_source: bool = False

    def as_record(self) -> dict[str, Any]:
        return {
            "exchange_q0": str(self.exchange_q0),
            "kmesh": list(self.kmesh) if self.kmesh is not None else None,
            "bloch_normalisation": str(self.bloch_normalisation),
            "same_source": bool(self.same_source),
        }


@dataclass
class CorrelationSource:
    """Everything the DLPNO engine reads about a reference and its factors.

    Build one with an adapter such as :func:`molecular_source`; the engine
    functions accept it through their ``source`` keyword and otherwise build
    the molecular one from their positional arguments.
    """

    #: Adapter name, for example ``"molecular-df"``.
    kind: str
    #: ``"rhf"``, ``"uhf"`` or ``"rohf"``.
    reference_kind: str
    molecule: Any
    basis: Any
    #: The converged SCF result the engine reads coefficients, Fock
    #: matrices and the energy from.
    reference: Any
    factors: ThreeCenterFactors
    #: AO overlap of the reference's AO frame. ``None`` means the molecular
    #: overlap of ``basis``. The experimental translation route also accepts
    #: an exact immutable CyclicAOOperator; dense-only engines refuse it.
    overlap: Optional[np.ndarray | CyclicAOOperator] = None
    #: Distance between two localised-orbital centroids for the pair screens
    #: and ``coupling_radius``; ``None`` is the Euclidean distance.
    pair_distance_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None
    conventions: SourceConventions = field(default_factory=SourceConventions)
    #: The record the identity hashes. Adapters fill it.
    provenance: dict[str, Any] = field(default_factory=dict)
    #: SHA-256 hex digest of ``provenance``; computed when left empty.
    identity: str = ""
    _reference_state: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        kind = str(self.reference_kind).strip().lower()
        if kind not in _REFERENCE_KINDS:
            raise CorrelationSourceError(
                f"reference_kind must be one of {_REFERENCE_KINDS}, got "
                f"{self.reference_kind!r}"
            )
        self.reference_kind = kind
        if not callable(getattr(self.factors, "mo_transform", None)):
            raise CorrelationSourceError(
                "factors must provide mo_transform(C_left, C_right) returning "
                "three-centre factors B[P, p, q]"
            )
        if self.overlap is not None:
            from .translation_operators import CyclicAOOperator

            if type(self.overlap) is not CyclicAOOperator:
                overlap = np.asarray(self.overlap)
                if overlap.ndim != 2 or overlap.shape[0] != overlap.shape[1]:
                    raise CorrelationSourceError(
                        f"overlap must be a square matrix, got shape {overlap.shape}"
                    )
                self.overlap = overlap
        if not self.identity:
            self.identity = source_identity(self.provenance)
        from .translation_operators import CyclicAOOperator
        if (self.kind in {"molecular-df", "gamma-gdf", "bvk-supercell"}
                or type(self.overlap) is CyclicAOOperator):
            self._reference_state = self._reference_signature()

    def _reference_signature(self):
        """Bind reference data without dense conversion or factor tensor reads."""
        from .translation_operators import CyclicAOOperator
        from .translation_projectors import BlochOccupiedProjectors
        digest = hashlib.sha256()
        names = ("fock", "density", "mo_coeffs", "mo_energies", "occupations", "occupied_projectors")
        for suffix in ("", "_alpha", "_beta"):
            for name in names:
                value = getattr(self.reference, name + suffix, None)
                digest.update((name + suffix).encode())
                if value is None:
                    digest.update(b"absent")
                    continue
                if type(value) is CyclicAOOperator:
                    digest.update(value.identity.encode())
                    continue
                if type(value) is BlochOccupiedProjectors:
                    # Exact sealed operators and immutable scalar audit fields.
                    digest.update(repr((value.active.identity, value.frozen.identity,
                        *value[2:])).encode())
                    continue
                array = np.asarray(value)
                if array.dtype.kind not in "biufc":
                    raise CorrelationSourceError("reference must contain numeric dense arrays")
                digest.update(str((array.shape, array.dtype.str)).encode())
                # Bound temporary bytes even for strided or complex spin arrays.
                for block in np.nditer(array, flags=["external_loop", "buffered", "zerosize_ok"],
                                       order="C", buffersize=8192):
                    if not np.isfinite(block).all():
                        raise CorrelationSourceError("reference contains non-finite data")
                    digest.update(block.tobytes())
        if type(self.overlap) is CyclicAOOperator:
            digest.update(self.overlap.identity.encode())
        elif self.overlap is not None:
            overlap = np.asarray(self.overlap)
            digest.update(str((overlap.shape, overlap.dtype.str)).encode())
            for block in np.nditer(overlap, flags=["external_loop", "buffered", "zerosize_ok"],
                                   order="C", buffersize=8192):
                if not np.isfinite(block).all():
                    raise CorrelationSourceError("reference overlap contains non-finite data")
                digest.update(block.tobytes())
        energy = float(self.reference.energy)
        if not np.isfinite(energy):
            raise CorrelationSourceError("reference energy must be finite")
        return (digest.hexdigest(), energy, self.kind, self.reference_kind,
                detect_reference_kind(self.reference),
                getattr(self.reference, "ecp_total_ncore", 0),
                tuple(getattr(self.reference, name, None) for name in
                      ("n_alpha", "n_beta", "n_occ", "smearing_temperature")),
                json.dumps([[_atom_record(a) if a is not None else None for a in getattr(self.molecule, "atoms", [])],
                    getattr(self.molecule, "charge", 0), getattr(self.molecule, "multiplicity", 1)]),
                id(self.molecule), id(self.basis), id(self.factors),
                self.conventions, self.identity, source_identity(self.provenance))

    def validate_reference(self):
        """Recheck a bound reference at the synchronous consumer entry.

        This is a state-integrity guard, not factor certification or an SCF
        stationarity test. Compact numerical admission retains its existing gates.
        Concurrent caller mutation during an engine call remains unsupported.
        """
        if self._reference_state is None:
            return
        if not bool(getattr(self.reference, "converged", False)):
            raise CorrelationSourceError("correlation source requires a converged SCF reference")
        if detect_reference_kind(self.reference) != self.reference_kind:
            raise CorrelationSourceError("correlation source reference kind does not match its state")
        if self.kind == "molecular-df":
            _validate_molecular_occupations(self.molecule, self.reference, self.reference_kind)
        if self._reference_signature() != self._reference_state:
            raise CorrelationSourceError("correlation source reference changed after construction; rebuild the source")

    def overlap_matrix(self) -> np.ndarray:
        """The AO overlap the engine localises and projects with."""
        if self.overlap is not None:
            from .translation_operators import CyclicAOOperator

            if type(self.overlap) is CyclicAOOperator:
                raise CorrelationSourceError(
                    "cyclic AO operators require the experimental translation route; "
                    "this engine needs an explicit dense AO overlap"
                )
            return self.overlap
        from vibeqc._vibeqc_core import compute_overlap

        return np.asarray(compute_overlap(self.basis))


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"source provenance cannot carry {type(value).__name__}")


def source_identity(record: dict[str, Any]) -> str:
    """SHA-256 of a provenance record, canonicalised as sorted JSON."""
    payload = json.dumps(
        {"contract": CONTRACT_VERSION, **record},
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_identity(array):
    """Hash numerical contents, including shape, without a second tensor copy."""
    array = np.ascontiguousarray(array, dtype="<f8")
    return {
        "shape": list(array.shape),
        "sha256": hashlib.sha256(memoryview(array).cast("B")).hexdigest(),
    }


def _atom_record(atom: Any) -> list[Any]:
    z = getattr(atom, "Z", None)
    if z is None:
        z = getattr(atom, "z", None)
    xyz = [round(float(x), 10) for x in np.asarray(getattr(atom, "xyz")).tolist()]
    return [int(z) if z is not None else None, xyz]


def detect_reference_kind(reference: Any) -> str:
    """``"uhf"`` for a two-spin result, ``"rohf"`` when it says so, else ``"rhf"``."""
    declared = getattr(reference, "reference_kind", None)
    if isinstance(declared, str) and declared.strip().lower() in _REFERENCE_KINDS:
        return declared.strip().lower()
    if hasattr(reference, "mo_coeffs_alpha") and hasattr(reference, "mo_coeffs_beta"):
        return "uhf"
    return "rhf"


def _validate_molecular_occupations(molecule, reference, kind):
    """Check declared fillings against the partitions the molecular engines use.

    Riplinger and Neese, JCP 138, 034106 (2013), Sec. II A, Eqs. (1)-(3):
    occupied/virtual indices belong to a single determinant. No fractional
    occupation weighting is implemented. Legacy molecular results omit these
    vectors; their integer counts still come from charge, spin and ECP metadata.
    """
    kind = str(kind).strip().lower()
    if kind not in {"rhf", "uhf"}:
        return
    suffixes = ("_alpha", "_beta") if kind == "uhf" else ("",)
    declared = []
    for suffix in suffixes:
        value = getattr(reference, "occupations" + suffix, None)
        if value is None:
            continue
        occupations = np.asarray(value)
        if occupations.shape == (0,):
            continue
        declared.append((suffix, occupations))
    if not declared:
        return

    from vibeqc.correlation_conventions import effective_electron_count

    ne = effective_electron_count(molecule, reference)
    spin = int(molecule.multiplicity) - 1
    message = "molecular correlation requires integer aufbau occupations"
    if spin < 0 or spin > ne or (ne - spin) % 2 or (kind == "rhf" and spin):
        raise CorrelationSourceError(message + "; electron count and spin are inconsistent")
    for suffix, occupations in declared:
        count = (ne - spin) // 2 if suffix == "_beta" else (ne + spin) // 2
        coefficients = np.asarray(getattr(reference, "mo_coeffs" + suffix, None))
        if coefficients.ndim != 2 or count > coefficients.shape[1]:
            raise CorrelationSourceError(message + "; inconsistent orbital dimensions")
        target = np.zeros(coefficients.shape[1])
        target[:count] = 1.0 if suffix else 2.0
        # Same absolute filling tolerance as the periodic source adapters;
        # never round, reorder or mutate the caller's reference occupations.
        if (occupations.shape != target.shape or occupations.dtype.kind not in "biuf"
                or not np.isfinite(occupations).all()
                or not np.allclose(occupations, target, atol=1e-8, rtol=0)):
            raise CorrelationSourceError(
                message + f" in the {suffix.lstrip('_') or 'restricted'} channel "
                f"({count} occupied orbitals)"
            )


def molecular_source(
    molecule: Any,
    basis: Any,
    reference: Any,
    df: ThreeCenterFactors,
    *,
    reference_kind: Optional[str] = None,
    aux_basis_name: Optional[str] = None,
) -> CorrelationSource:
    """The molecular adapter: an SCF result plus a density-fitting object.

    The identity hashes the geometry, charge and multiplicity, the orbital
    and auxiliary basis names and sizes, the reference kind and energy, and
    the conventions, so two runs on the same inputs share it and any change
    of basis, auxiliary basis or geometry changes it.
    """
    if not bool(getattr(reference, "converged", False)):
        raise CorrelationSourceError(
            "molecular_source requires a converged SCF reference; "
            "stopped or unqualified orbital states are not supported"
        )
    kind = reference_kind or detect_reference_kind(reference)
    _validate_molecular_occupations(molecule, reference, kind)
    aux_name = aux_basis_name or getattr(df, "aux_basis_name", None)
    conventions = SourceConventions()
    record = {
        "kind": "molecular-df",
        "reference_kind": kind,
        "basis": getattr(basis, "name", None),
        "n_basis": int(getattr(basis, "nbasis", 0) or 0),
        "aux_basis": aux_name,
        "n_aux": int(getattr(df, "n_aux", 0) or 0),
        "atoms": [_atom_record(atom) for atom in getattr(molecule, "atoms", [])],
        "charge": int(getattr(molecule, "charge", 0)),
        "multiplicity": int(getattr(molecule, "multiplicity", 1)),
        "reference_energy": f"{float(getattr(reference, 'energy', 0.0)):.12f}",
        "conventions": conventions.as_record(),
    }
    return CorrelationSource(
        kind="molecular-df",
        reference_kind=kind,
        molecule=molecule,
        basis=basis,
        reference=reference,
        factors=df,
        conventions=conventions,
        provenance=record,
    )


class LocalFitFactors:
    """RI factors for the local-fit route, built globally only on demand.

    With ``DLPNOMP2Options(local_df=True)`` DLPNO-MP2 refits every pair in
    the auxiliary functions near its domain and reads only ``aux_basis``.
    Constructing :class:`~vibeqc.density_fitting.DensityFitting` up front
    would evaluate the global three-centre tensor and metric that route
    exists to avoid. This adapter carries the orbital and auxiliary bases
    and builds that fit the first time a consumer asks for
    :meth:`mo_transform`, so it is a full :class:`ThreeCenterFactors` of the
    same RI factorisation.
    """

    def __init__(self, orbital_basis, aux_basis, *, aux_basis_name=None):
        self.orbital_basis = orbital_basis
        self.aux_basis = aux_basis
        self.aux_basis_name = aux_basis_name
        self.n_orb = int(orbital_basis.nbasis)
        self.n_aux = int(aux_basis.nbasis)
        self._global_fit = None

    @property
    def global_fit_built(self) -> bool:
        """Whether :meth:`mo_transform` has built the global fit."""
        return self._global_fit is not None

    def mo_transform(self, C_left, C_right=None):
        if self._global_fit is None:
            from vibeqc.density_fitting import DensityFitting

            self._global_fit = DensityFitting(
                self.orbital_basis,
                self.aux_basis,
                aux_basis_name=self.aux_basis_name or "",
            )
        return self._global_fit.mo_transform(C_left, C_right)


# ---------------------------------------------------------------------------
# Crystals: increment B of the design page. Only the Gamma-point GDF adapter
# ships here; multi-k folding is increment B2.
# ---------------------------------------------------------------------------


@dataclass
class GammaReference:
    """A real Gamma-point reference in the shape the engine reads.

    Built by :func:`gamma_gdf_source` from the single k-point of a multi-k
    GDF result: real coefficients (occupied block localised), Fock, overlap
    and orbital energies, the SCF energy per cell, and the ECP provenance the
    electron count needs.
    """

    mo_coeffs: np.ndarray
    fock: np.ndarray
    overlap: np.ndarray
    mo_energies: np.ndarray
    energy: float
    converged: bool = True
    ecp_total_ncore: int = 0
    reference_kind: str = "rhf"
    #: ``"pipek-mezey"`` when the active occupied block was localised by the
    #: adapter, ``"none"`` when the canonical orbitals were kept.
    localisation: str = "none"


@dataclass
class GammaUHFReference:
    """Two real spin channels on the same periodic AO and auxiliary frames.

    These are independent UHF spaces, not the shared ROHF/QRO orbitals
    and NEV-PNOs of Saitow et al., JCP 146, 164105 (2017).
    """

    mo_coeffs_alpha: np.ndarray
    mo_coeffs_beta: np.ndarray
    fock_alpha: np.ndarray
    fock_beta: np.ndarray
    mo_energies_alpha: np.ndarray
    mo_energies_beta: np.ndarray
    overlap: np.ndarray
    energy: float
    converged: bool = True
    ecp_total_ncore: int = 0
    reference_kind: str = "uhf"
    localisation: str = "none"


def _periodic_channels(system, result, n_frozen):
    """Spin suffix, occupied count and occupation weight for each channel."""
    if getattr(result, "functional", None) is not None:
        raise CorrelationSourceError("periodic correlation needs a Hartree-Fock reference")
    kind = detect_reference_kind(result)
    if kind == "rohf":
        raise CorrelationSourceError("periodic ROHF correlation is not wired; use KUHF")
    ne = int(system.unit_cell_molecule().n_electrons()) - int(
        getattr(result, "ecp_total_ncore", 0) or 0
    )
    spin = int(system.multiplicity) - 1
    if ne < 0 or spin < 0 or spin > ne or (ne - spin) % 2:
        raise CorrelationSourceError("electron count and multiplicity do not define an integer determinant")
    na, nb = (ne + spin) // 2, (ne - spin) // 2
    if kind == "rhf" and na != nb:
        raise CorrelationSourceError("an open-shell cell needs both UHF spin channels")
    if int(n_frozen) != n_frozen or not 0 <= n_frozen <= nb:
        raise CorrelationSourceError(f"n_frozen must be an integer between 0 and {nb}")
    channels = (("_alpha", na, 1.0), ("_beta", nb, 1.0)) if kind == "uhf" else (("", na, 2.0),)
    return kind, channels


def _periodic_channel_data(result, suffix, n_k, n_occ, weight, nbf, *, n_frozen=0):
    """Audit a gapped integer filling before changing the orbital gauge.

    A gap in the *sorted folded* spectrum alone does not certify that the
    occupied per-k bands are its lowest states. Check the occupied and
    virtual bands across the entire mesh before re-diagonalising the Fock.
    """
    fields = [
        getattr(result, name + suffix, None)
        for name in ("mo_coeffs", "fock", "mo_energies")
    ]
    if any(not isinstance(v, (list, tuple)) or len(v) != n_k for v in fields):
        raise CorrelationSourceError(
            f"periodic {suffix.lstrip('_') or 'RHF'} channel needs coefficients, "
            f"Fock matrices and energies at all {n_k} k-points"
        )
    coeffs, focks, energies = fields
    weights = getattr(result, "kpoint_weights", None)
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n_k,) or not np.allclose(weights, 1.0 / n_k, atol=1e-12, rtol=0):
            raise CorrelationSourceError("periodic correlation needs uniform full-mesh k-point weights")
    occ = getattr(result, "occupations" + suffix, None)
    if occ is not None and len(occ) not in (0, n_k):
        raise CorrelationSourceError("periodic occupations do not cover the full mesh")
    occupied, virtual, frozen, active = [], [], [], []
    for k, (c, f, e) in enumerate(zip(coeffs, focks, energies)):
        c, f, e = np.asarray(c), np.asarray(f), np.asarray(e)
        if (c.ndim != 2 or c.shape[0] != nbf or f.shape != (nbf, nbf)
                or e.shape != (c.shape[1],) or n_occ > c.shape[1]):
            raise CorrelationSourceError(f"inconsistent periodic orbital dimensions in channel {suffix!r}, k={k}")
        if not all(np.isfinite(v).all() for v in (c, f, e)):
            raise CorrelationSourceError("periodic reference contains non-finite orbital data")
        if suffix and c.shape[1] != nbf:
            raise CorrelationSourceError("the periodic UHF engine needs full-rank orbital spaces")
        e = _real_array(e, name="mo_energies" + suffix)
        if np.any(np.diff(e) < -1e-10):
            raise CorrelationSourceError("periodic orbital energies must be in aufbau order")
        if occ is not None and len(occ):
            o = np.asarray(occ[k], dtype=float)
            target = np.zeros(c.shape[1])
            target[:n_occ] = weight
            if o.shape != target.shape or not np.allclose(o, target, atol=1e-8, rtol=0):
                raise CorrelationSourceError(
                    f"periodic {suffix.lstrip('_') or 'closed-shell'} occupations "
                    f"are not an integer aufbau filling of {n_occ} bands"
                )
        if n_occ:
            occupied.append(float(e[n_occ - 1]))
        if n_occ < e.size:
            virtual.append(float(e[n_occ]))
        if 0 < n_frozen < n_occ:
            frozen.append(float(e[n_frozen - 1]))
            active.append(float(e[n_frozen]))
    # Gamma realification may rotate a degenerate eigenspace. Neither an
    # occupied/virtual boundary nor a frozen/active boundary may bisect it.
    if occupied and virtual and min(virtual) - max(occupied) < _REAL_ORBITAL_DEGENERACY_TOL:
        raise CorrelationSourceError(
            f"periodic {suffix.lstrip('_') or 'RHF'} spectrum has no global "
            "gap between its occupied and virtual bands"
        )
    if frozen and active and min(active) - max(frozen) < _REAL_ORBITAL_DEGENERACY_TOL:
        raise CorrelationSourceError(
            f"periodic {suffix.lstrip('_') or 'RHF'} frozen and active occupied "
            "bands have no global gap; the adapter cannot preserve the frozen space"
        )
    return coeffs, focks, energies


class LpqFactors:
    """Three-centre factors ``L[P, mu, nu]`` over a canonical auxiliary frame.

    ``(mu nu|lambda sigma) = sum_P L[P, mu, nu] L[P, lambda, sigma]``; the
    metric is the identity, so ``mo_transform`` is a plain two-index
    transform, as for :class:`~vibeqc.density_fitting.DensityFitting`.
    """

    def __init__(self, L, *, aux_basis_name=None):
        L = np.asarray(L)
        if L.ndim != 3 or L.shape[1] != L.shape[2]:
            raise CorrelationSourceError(
                f"three-centre factors must have shape (n_aux, n, n), got {L.shape}"
            )
        self.B = np.ascontiguousarray(L, dtype=float)
        self.three_center = self.B
        self.n_aux = int(L.shape[0])
        self.n_orb = int(L.shape[1])
        self.aux_basis_name = aux_basis_name
        self.metric = None

    def mo_transform(self, C_left, C_right=None):
        if C_right is None:
            C_right = C_left
        tmp = np.einsum("Pmn,nq->Pmq", self.B, np.asarray(C_right, dtype=float), optimize=True)
        return np.einsum("mp,Pmq->Ppq", np.asarray(C_left, dtype=float), tmp, optimize=True)


def refuse_unsupported_periodic_reference(result) -> None:
    """Fail closed on the references the design refuses (#508, decision 2).

    A BIPOLE reference has no certified same-source correlation factors; a
    symmetry-reduced reference keeps its orbitals on the irreducible wedge
    only and no certified transport to the full mesh exists.
    """
    backend = " ".join(
        str(getattr(result, name, "") or "")
        for name in ("runtime_backend", "backend")
    ).lower()
    if "bipole" in backend or type(result).__name__.lower().startswith("pbcbipole"):
        raise CorrelationSourceError(
            "BIPOLE reference refused for correlation: the Ewald-split SCF "
            "route has no certified same-source correlation factors. Run the "
            "reference on the GDF route instead "
            "(docs/design_dlpno_correlation_source.md, decision 2)."
        )
    if bool(getattr(result, "used_kpoint_symmetry_unfolding", False)):
        raise CorrelationSourceError(
            "symmetry-reduced reference refused for correlation: its orbitals "
            "exist only on the irreducible wedge and no certified orbital "
            "transport to the full mesh exists. Run the SCF on the full mesh; "
            "symmetry_reduce_k only accelerates exchange and keeps full-mesh "
            "orbitals (docs/design_dlpno_correlation_source.md, decision 2)."
        )


def _real_array(value, *, name, tol=1e-8):
    arr = np.asarray(value)
    if not np.isfinite(arr).all():
        raise CorrelationSourceError(f"{name} contains non-finite values")
    if np.iscomplexobj(arr):
        imag = float(np.max(np.abs(arr.imag))) if arr.size else 0.0
        if imag > tol:
            raise CorrelationSourceError(
                f"{name} is not real at the Gamma point (largest imaginary "
                f"part {imag:.2e} > {tol:.0e})"
            )
        arr = arr.real
    return np.ascontiguousarray(arr, dtype=float)


def _real_torus_source(
    system, basis, reference, factors, *, kind, conventions,
    factor_tensor, factor_representation, aux_basis_name, provenance=None,
) -> CorrelationSource:
    """Package an already admitted real, restricted experimental torus.

    The caller owns folding, localisation and the same-source proof. This
    helper does not rebuild a Fock or fit: it supplies the cyclic AO metric
    explicitly instead of changing the process-wide molecular overlap.
    ``factor_tensor`` is the representation actually stored by ``factors``;
    hashing chi home-auxiliary factors must not expand translated AO copies.
    The representation and mesh are part of the identity along with the
    numerical tensors, so compressed and full auxiliary frames are distinct.
    """
    from ..periodic_toroidal_mp2 import toroidal_pair_distance_fn
    from .translation_operators import CyclicAOOperator, matrix_record
    from .translation_projectors import BlochOccupiedProjectors, reference_orbitals

    if detect_reference_kind(reference) != "rhf" or not reference.converged:
        raise CorrelationSourceError("a real torus source requires a converged RHF reference")
    block_source = type(reference.overlap) is CyclicAOOperator
    if block_source != (type(reference.fock) is CyclicAOOperator):
        raise CorrelationSourceError("torus overlap and Fock must use the same representation")
    S = reference.overlap if block_source else _real_array(reference.overlap, name="torus overlap")
    F = reference.fock if block_source else _real_array(reference.fock, name="torus Fock")
    if block_source and (S.mesh != tuple(conventions.kmesh) or F.mesh != S.mesh):
        raise CorrelationSourceError("torus operator mesh does not match its conventions")
    orbitals = reference_orbitals(reference)
    compact = type(orbitals) is BlochOccupiedProjectors
    C = None if compact else _real_array(orbitals, name="torus orbitals")
    if compact:
        orbitals.validate(conventions.kmesh, S, F, orbitals.n_occupied, orbitals.n_frozen, 1e-8)
    L = _real_array(factor_tensor, name="torus factors")
    nbf = int(basis.nbasis)
    if (S.shape != (nbf, nbf) or F.shape != S.shape
            or (orbitals.active.shape[0] != nbf if compact else C.ndim != 2 or C.shape[0] != nbf)
            or L.ndim != 3 or L.shape[1:] != S.shape):
        raise CorrelationSourceError("real torus tensors do not match the supercell AO basis")
    if not np.isfinite(reference.energy):
        raise CorrelationSourceError("real torus reference energy is not finite")
    molecule = system.unit_cell_molecule()
    record = {
        "kind": kind, "reference_kind": "rhf",
        "basis": basis.name, "n_basis": nbf,
        "aux_basis": aux_basis_name,
        "n_aux": int(getattr(factors, "n_aux", getattr(factors, "n_fit", 0))),
        "dim": int(system.dim), "lattice": np.asarray(system.lattice).tolist(),
        "atoms": [_atom_record(atom) for atom in molecule.atoms],
        "charge": int(molecule.charge), "multiplicity": int(molecule.multiplicity),
        "reference_energy": float(reference.energy),
        "fock": matrix_record(F) if block_source else _array_identity(F),
        "overlap": matrix_record(S) if block_source else _array_identity(S),
        "orbitals": orbitals.record() if compact else _array_identity(C), "factors": _array_identity(L),
        "factor_representation": factor_representation,
        "conventions": conventions.as_record(),
        "adapter": dict(provenance or {}),
    }
    return CorrelationSource(
        kind=kind, reference_kind="rhf", molecule=molecule, basis=basis,
        reference=reference, factors=factors, overlap=S,
        pair_distance_fn=toroidal_pair_distance_fn(system, (1, 1, 1)),
        conventions=conventions, provenance=record,
    )


def _real_uhf_torus_source(
    system, basis, reference, factors, *, kind, conventions,
    factor_tensor, factor_representation, aux_basis_name, provenance=None,
) -> CorrelationSource:
    """Package an admitted real torus with independent UHF spin spaces.

    As for ``_real_torus_source``, the wrapper owns admission, folding and
    the same-source proof. No shared-orbital or QRO conversion is made:
    each spin's Fock, coefficients and energies identify the Hamiltonian
    used by the UHF engines. Hash the stored factor representation directly,
    including compact chi home-auxiliary factors, without expanding it.
    """
    from ..periodic_toroidal_mp2 import toroidal_pair_distance_fn

    if detect_reference_kind(reference) != "uhf" or not reference.converged:
        raise CorrelationSourceError("a real UHF torus source requires a converged UHF reference")
    S = _real_array(reference.overlap, name="UHF torus overlap")
    L = _real_array(factor_tensor, name="UHF torus factors")
    nbf = int(basis.nbasis)
    if (S.shape != (nbf, nbf) or L.ndim != 3
            or L.shape[1:] != S.shape):
        raise CorrelationSourceError("real UHF torus tensors do not match the supercell AO basis")
    if not np.isfinite(reference.energy):
        raise CorrelationSourceError("real UHF torus reference energy is not finite")
    spin_record = {}
    for spin in ("alpha", "beta"):
        F = _real_array(getattr(reference, f"fock_{spin}"), name=f"torus {spin} Fock")
        C = _real_array(getattr(reference, f"mo_coeffs_{spin}"), name=f"torus {spin} orbitals")
        eps = _real_array(getattr(reference, f"mo_energies_{spin}"), name=f"torus {spin} energies")
        if (F.shape != S.shape or C.ndim != 2 or C.shape[0] != nbf
                or eps.shape != (C.shape[1],)):
            raise CorrelationSourceError(f"real UHF torus {spin} tensors have inconsistent dimensions")
        spin_record[spin] = {
            "fock": _array_identity(F), "orbitals": _array_identity(C),
            "orbital_energies": _array_identity(eps),
        }
    molecule = system.unit_cell_molecule()
    record = {
        "kind": kind, "reference_kind": "uhf",
        "basis": basis.name, "n_basis": nbf,
        "aux_basis": aux_basis_name,
        "n_aux": int(getattr(factors, "n_aux", getattr(factors, "n_fit", 0))),
        "dim": int(system.dim), "lattice": np.asarray(system.lattice).tolist(),
        "atoms": [_atom_record(atom) for atom in molecule.atoms],
        "charge": int(molecule.charge), "multiplicity": int(molecule.multiplicity),
        "ecp_total_ncore": int(getattr(reference, "ecp_total_ncore", 0)),
        "reference_energy": float(reference.energy),
        "overlap": _array_identity(S), "spin_channels": spin_record,
        "factors": _array_identity(L), "factor_representation": factor_representation,
        "conventions": conventions.as_record(),
        "adapter": dict(provenance or {}),
    }
    return CorrelationSource(
        kind=kind, reference_kind="uhf", molecule=molecule, basis=basis,
        reference=reference, factors=factors, overlap=S,
        pair_distance_fn=toroidal_pair_distance_fn(system, (1, 1, 1)),
        conventions=conventions, provenance=record,
    )


def _real_gamma_orbitals(
    C, eps, S, F, *, degeneracy_tol=_REAL_ORBITAL_DEGENERACY_TOL, eigenvalue_tol=1e-7
):
    """Real S-orthonormal Gamma orbitals from a complex eigenvector set.

    A Gamma-point eigensolver returns complex vectors with arbitrary phases;
    Fock and overlap are real, so every degenerate block spans a real space
    of its own dimension. Each block is rebuilt from the real span of its
    vectors (S-orthonormalised through the Gram eigenproblem) and the real
    Fock is re-diagonalised inside it, which keeps the SCF's rank and
    eigenvalues; a block whose real span has the wrong dimension, or whose
    eigenvalues move, is refused.
    """
    C = np.asarray(C)
    return _real_gamma_orbitals_from_columns(
        C.shape[0], eps, lambda indices: C[:, indices], S, F,
        degeneracy_tol=degeneracy_tol, eigenvalue_tol=eigenvalue_tol,
    )


def _real_gamma_orbitals_from_columns(
    nbf, eps, columns, S, F, *, degeneracy_tol=_REAL_ORBITAL_DEGENERACY_TOL,
    eigenvalue_tol=1e-7,
):
    """Realify complete eigenspaces supplied in original orbital order.

    ``columns(indices)`` supplies only the requested complex AO columns.
    Bloch sources can construct one degenerate space directly, without a
    full folded complex coefficient array or a globally reordered copy.
    The dense adapter above and this streamed adapter share every rank,
    degeneracy and eigenvalue check. Never split a degenerate space merely
    to satisfy a storage limit: its real span can require all its columns.
    """
    blocks = _RealGammaOrbitalBlocks(nbf, eps, columns, S, F,
        degeneracy_tol=degeneracy_tol, eigenvalue_tol=eigenvalue_tol)
    out = np.zeros((nbf, blocks.n_orb), dtype=float)
    eps_out = np.zeros(blocks.n_orb, dtype=float)
    for i, j in blocks.ranges:
        out[:, i:j], eps_out[i:j] = blocks.realify(i, j)
    return out, eps_out


class _RealGammaOrbitalBlocks:
    """Replay complete real eigenspaces without retaining their AO columns.

    Dense consumers assemble the blocks; compact consumers replay them for
    cross-space audits. An optional bounded cache stores only the small
    real-span/Fock transforms, never AO coefficients. Both paths use the
    same real-span and eigenvalue guards on each fresh decomposition.
    Cached recipes require stable columns and operators for this object's
    lifetime; the compact adapter supplies its private input snapshot.
    """

    def __init__(self, nbf, eps, columns, S, F, *,
                 degeneracy_tol=_REAL_ORBITAL_DEGENERACY_TOL, eigenvalue_tol=1e-7,
                 recipe_cache_bytes=0):
        eps = np.asarray(eps).real.astype(float)
        if eps.ndim != 1 or not eps.size or not np.isfinite(eps).all():
            raise CorrelationSourceError('realification requires finite nonempty orbital energies')
        if (isinstance(recipe_cache_bytes, (bool, np.bool_))
                or not isinstance(recipe_cache_bytes, (int, np.integer)) or recipe_cache_bytes < 0):
            raise ValueError('realification recipe cache budget must be a nonnegative integer')
        self.order = np.argsort(eps, kind="stable")
        self.eps = eps[self.order]
        self.n_orb = len(eps)
        self.nbf, self.columns, self.S, self.F = nbf, columns, S, F
        self.eigenvalue_tol = eigenvalue_tol
        self.evaluations = 0
        self.realifications = self.recipe_cache_hits = self.recipe_cache_bytes = 0
        self.recipe_cache_budget = int(recipe_cache_bytes)
        self._recipes = {}
        ranges = []
        i = 0
        while i < self.n_orb:
            j = i+1
            while j < self.n_orb and abs(self.eps[j]-self.eps[i]) < degeneracy_tol:
                j += 1
            ranges.append((i, j))
            i = j
        self.ranges = tuple(ranges)

    def recipe_cache_record(self):
        return dict(policy='first-fit-no-ao-coefficients', budget_bytes=self.recipe_cache_budget,
                    stored_bytes=self.recipe_cache_bytes, entries=len(self._recipes),
                    hits=self.recipe_cache_hits, realifications=self.realifications)

    def realify(self, i, j):
        from .translation_operators import CyclicAOOperator, matrix_apply

        self.evaluations += 1
        m = j - i
        block = self.columns(self.order[i:j])
        if block.shape != (self.nbf, m) or not np.isfinite(block).all():
            raise CorrelationSourceError('realification requires finite complete AO column blocks')
        S, F = self.S, self.F
        candidates = np.column_stack([block.real, block.imag])
        recipe = self._recipes.get((i, j))
        if recipe is not None:
            vectors, roots, rotation, energies = recipe
            self.recipe_cache_hits += 1
            # Preserve the original multiplication/division order; combining
            # these transforms would change floating-point cancellation.
            real_basis = candidates @ vectors / roots
            return real_basis @ rotation, energies.copy()
        self.realifications += 1
        gram = (candidates.T @ matrix_apply(S, candidates) if type(S) is CyclicAOOperator
                else candidates.T @ S @ candidates)
        w, V = np.linalg.eigh(0.5 * (gram + gram.T))
        keep = np.argsort(w)[::-1][:m]
        rest = np.argsort(w)[::-1][m:]
        if w[keep[-1]] < 1e-8 or (rest.size and w[rest[0]] > 1e-6):
            raise CorrelationSourceError(
                f"Gamma orbitals {i}..{j - 1} do not span a real space of "
                f"dimension {m} (Gram eigenvalues {np.sort(w)[::-1][: m + 1]})"
            )
        vectors, roots = V[:, keep], np.sqrt(w[keep])
        real_basis = candidates @ vectors / roots
        f_block = (real_basis.T @ matrix_apply(F, real_basis) if type(F) is CyclicAOOperator
                   else real_basis.T @ F @ real_basis)
        e_block, U = np.linalg.eigh(0.5 * (f_block + f_block.T))
        shift = float(np.max(np.abs(e_block-self.eps[i:j])))
        if shift > self.eigenvalue_tol:
            raise CorrelationSourceError(
                "realified Gamma orbitals do not reproduce the SCF orbital energies "
                f"(largest shift {shift:.2e} Ha)"
            )
        # A complete recipe costs 8*(3*m*m+2*m) bytes. First fit, no eviction:
        # a wide degeneracy cannot grow the cache or displace admitted small
        # blocks. Own the arrays so no candidate/Gram allocation is retained.
        recipe = (vectors, roots, U, e_block)
        size = sum(value.nbytes for value in recipe)
        if size <= self.recipe_cache_budget-self.recipe_cache_bytes:
            owned = tuple(value.copy(order='K') for value in recipe)
            for value in owned:
                value.setflags(write=False)
            self._recipes[i, j] = owned
            self.recipe_cache_bytes += size
        return real_basis @ U, e_block


def _localise_active_occupied(C, S, basis, system, n_occ, n_frozen):
    """Pipek-Mezey localise the active occupied block ``C[:, n_frozen:n_occ]``.

    The frozen core stays canonical, so the engine's ``n_frozen`` keeps its
    meaning; Pipek-Mezey needs only overlap and atom populations and is
    therefore safe on a crystal whose orbitals wrap the cell.
    """
    from vibeqc.periodic_localise import localise_periodic_gamma

    n_act = int(n_occ - n_frozen)
    if n_act <= 1:
        return np.array(C, dtype=float, copy=True)
    active_first = np.column_stack([C[:, n_frozen:n_occ], C[:, :n_frozen], C[:, n_occ:]])
    duck = type("_GammaDuck", (), {})()
    duck.mo_coeffs = active_first
    duck.overlap = S
    wannier = localise_periodic_gamma(duck, basis, system, method="pipek-mezey", n_occ=n_act)
    out = np.array(C, dtype=float, copy=True)
    out[:, n_frozen:n_occ] = np.asarray(wannier.C_loc, dtype=float)
    return out


def gamma_gdf_source(
    system,
    basis,
    result,
    *,
    aux_basis_name: Optional[str] = None,
    n_frozen: int = 0,
    localise: str = "pipek-mezey",
    factors: Optional[ThreeCenterFactors] = None,
) -> CorrelationSource:
    """The Gamma-point GDF adapter: a single-k GDF result plus its own fit.

    ``result`` is a :func:`~vibeqc.periodic_k_gdf.run_krhf_periodic_gdf` or
    :func:`~vibeqc.periodic_k_gdf.run_kuhf_periodic_gdf`
    result at ``kmesh=(1, 1, 1)`` run with ``fit_source="classic"`` and
    ``retain_fit_cache=True``: the factors are the SCF's own classic RSGDF
    fit in the canonical auxiliary frame, so reference and factors are one
    Hamiltonian (``same_source``; the Coulomb energy rebuilt from them
    equals the SCF's). The default fit source takes a Gamma fast path that
    retains nothing and is refused here with that reason.
    The Fock matrix keeps the Ewald exchange zero-mode term the SCF applied;
    the factors are the bare fitted Coulomb operator with G=0 omitted, as the
    SCF contracted them. ``localise="pipek-mezey"`` localises the active
    occupied block for the engine (pass ``localise="none"`` to keep canonical
    orbitals, the exactness-ratchet setting); the engine itself must then run
    with ``localise="none"``. KUHF spin channels are realified and localised
    independently, without rotating occupied into virtual or frozen spaces.
    """
    refuse_unsupported_periodic_reference(result)
    reference_kind, channels = _periodic_channels(system, result, n_frozen)
    mo_k = getattr(result, "mo_coeffs" + channels[0][0], None)
    if not isinstance(mo_k, (list, tuple)):
        raise CorrelationSourceError(
            "gamma_gdf_source needs a multi-k GDF result "
            "(run_krhf_periodic_gdf or run_kuhf_periodic_gdf at kmesh=(1, 1, 1))"
        )
    if len(mo_k) != 1:
        raise CorrelationSourceError(
            f"gamma_gdf_source needs a single-k (Gamma) result; this one "
            f"carries {len(mo_k)} k-points. A Gamma-centred mesh folds to the "
            "real BvK supercell through bvk_supercell_source."
        )
    kpts = np.asarray(getattr(result, "kpoints_cart", np.zeros((1, 3))), dtype=float).reshape(-1, 3)
    if kpts.size and float(np.max(np.abs(kpts))) > 1e-12:
        raise CorrelationSourceError("gamma_gdf_source: the single k-point is not Gamma")
    if not bool(getattr(result, "converged", False)):
        raise CorrelationSourceError("gamma_gdf_source: the reference SCF did not converge")
    if str(localise).lower() not in ("pipek-mezey", "none"):
        raise CorrelationSourceError(
            f"localise must be 'pipek-mezey' or 'none', got {localise!r}"
        )

    cache = getattr(result, "fit_cache", None)
    aux_name = aux_basis_name or getattr(result, "aux_basis_name", None)
    same_source = False
    if factors is None:
        if cache is None:
            raise CorrelationSourceError(
                "gamma_gdf_source needs the SCF's retained fit for "
                f"same-source factors: run_k{reference_kind}_periodic_gdf(..., "
                "fit_source='classic', retain_fit_cache=True). The default "
                "fit source takes a Gamma fast path that retains no fit. "
                "Explicit factors= bypass this with same_source=False."
            )
        factors = LpqFactors(_real_array(cache.canonical_pair(0, 0), name="fit factors"),
                             aux_basis_name=aux_name)
        same_source = True

    if float(getattr(result, "smearing_temperature", 0.0) or 0.0) > 0.0:
        raise CorrelationSourceError(
            "gamma_gdf_source: the reference uses fractional (smeared) "
            "occupations; the engine needs an integer-aufbau determinant"
        )
    molecule = system.unit_cell_molecule()
    S = _real_array(result.overlap[0], name="overlap")
    ncore = int(getattr(result, "ecp_total_ncore", 0) or 0)
    orbital_data = {}
    for suffix, n_occ, weight in channels:
        mo_k, f_k, eps_k = _periodic_channel_data(
            result, suffix, 1, n_occ, weight, S.shape[0], n_frozen=int(n_frozen)
        )
        F = _real_array(f_k[0], name="fock" + suffix)
        C, eps = _real_gamma_orbitals(mo_k[0], eps_k[0], S, F)
        if str(localise).lower() == "pipek-mezey":
            C = _localise_active_occupied(C, S, basis, system, n_occ, int(n_frozen))
        orbital_data.update({
            "mo_coeffs" + suffix: C, "fock" + suffix: F,
            "mo_energies" + suffix: eps,
        })
    reference_class = GammaUHFReference if reference_kind == "uhf" else GammaReference
    reference = reference_class(
        **orbital_data, overlap=S,
        energy=float(result.energy), converged=True, ecp_total_ncore=ncore,
        localisation=str(localise).lower(),
    )

    from vibeqc.periodic_toroidal_mp2 import toroidal_pair_distance_fn

    distance = toroidal_pair_distance_fn(system, (1, 1, 1))
    exchange_q0 = str(getattr(result, "exchange_q0", None) or "ewald")
    executed = getattr(result, "executed_fit_source", None)
    conventions = SourceConventions(
        exchange_q0=exchange_q0,
        kmesh=(1, 1, 1),
        bloch_normalisation="unnormalised-ao-bloch-sums-uniform-weights",
        same_source=same_source,
    )
    lattice = np.asarray(system.lattice, dtype=float)
    record = {
        "kind": "gamma-gdf",
        "reference_kind": reference_kind,
        "basis": getattr(basis, "name", None),
        "n_basis": int(getattr(basis, "nbasis", C.shape[0]) or C.shape[0]),
        "aux_basis": aux_name,
        "n_aux": int(getattr(factors, "n_aux", 0) or 0),
        "dim": int(system.dim),
        "lattice": [[round(float(x), 10) for x in row] for row in lattice.tolist()],
        "atoms": [_atom_record(atom) for atom in getattr(molecule, "atoms", [])],
        "charge": int(getattr(molecule, "charge", 0)),
        "multiplicity": int(getattr(molecule, "multiplicity", 1)),
        "backend": str(getattr(result, "backend", "") or ""),
        # The driver's executed-source record names the fit engine, its
        # numerical supports and the SHA-256 of the factors it contracted.
        "fit": {
            "engine": getattr(executed, "engine", None),
            "source": getattr(executed, "source", None),
            "source_signature": getattr(executed, "source_signature", None),
            "factor_sha256": getattr(executed, "factor_sha256", None),
            "auxiliary_frame": getattr(executed, "auxiliary_frame", None),
            "metric_threshold": getattr(executed, "metric_threshold", None),
            "cache_frame": getattr(cache, "frame", None),
            "ke_cutoff": getattr(cache, "ke_cutoff", None),
            "tail_ke_cutoff": getattr(cache, "tail_ke_cutoff", None),
            "support_policy": getattr(cache, "support_policy", None),
        },
        "n_frozen": int(n_frozen),
        "localisation": str(localise).lower(),
        "reference_energy": f"{float(result.energy):.12f}",
        "conventions": conventions.as_record(),
    }
    return CorrelationSource(
        kind="gamma-gdf",
        reference_kind=reference_kind,
        molecule=molecule,
        basis=basis,
        reference=reference,
        factors=factors,
        overlap=S,
        pair_distance_fn=distance,
        conventions=conventions,
        provenance=record,
    )


def bvk_supercell_source(
    system,
    basis,
    result,
    *,
    mesh,
    aux_basis_name: Optional[str] = None,
    n_frozen_per_cell: int = 0,
    localise: str = "pipek-mezey",
    residual_tol: float = 1e-7,
) -> CorrelationSource:
    """The BvK-supercell adapter: a full Gamma-centred mesh folded to the real torus.

    ``result`` is a :func:`~vibeqc.periodic_k_gdf.run_krhf_periodic_gdf` or
    :func:`~vibeqc.periodic_k_gdf.run_kuhf_periodic_gdf`
    result on the unreduced Gamma-centred mesh of
    :func:`~vibeqc.periodic.chi.scf.cyclic_gamma_mesh` (its cyclic k order,
    no half shift, no symmetry reduction), run with ``fit_source="classic"``
    and ``retain_fit_cache=True``. The per-k overlap and Fock matrices and
    every retained ``(k_i, k_j)`` factor are inverse-transformed to the real
    BvK supercell with the finite-character kernels the chi line uses
    (Theorem 1 of ``periodic/ccm/bloch_reference.py``: the real supercell
    Gamma problem of the folded Hamiltonian is the full-mesh Bloch
    problem), the real supercell Fock is diagonalised in the canonical
    orthogonal frame, and the eigenvalues are checked against the union of
    the per-k ones. Reference and factors stay one Hamiltonian
    (``same_source``). Energies on the returned reference are totals for
    the supercell; the engine's frozen-core count is ``n_frozen_per_cell``
    times the number of cells.

    KUHF folds the two Fock matrices separately on the same overlap and
    auxiliary frames. The supercell has multiplicity
    ``1 + n_cells * (system.multiplicity - 1)``. This uses the finite BvK
    equivalence discussed by Ye and Berkelbach, JCTC 20, 8948 (2024),
    doi:10.1021/acs.jctc.4c00936, not their fragment LNO approximation.
    """
    from vibeqc import BasisSet
    from vibeqc.aux_basis import make_aux_basis_set
    from vibeqc.periodic.chi.posthf import (
        _BRealDFAdapter,
        _build_supercell_system,
        _cyclic_translations,
        _lpq_cache_to_real_home_auxiliary,
        _periodic_matrix_to_real_supercell,
    )
    from vibeqc.periodic.chi.scf import cyclic_gamma_mesh
    from vibeqc.periodic_toroidal_mp2 import toroidal_pair_distance_fn

    refuse_unsupported_periodic_reference(result)
    mesh = tuple(int(x) for x in mesh)
    if len(mesh) != 3 or min(mesh) < 1:
        raise CorrelationSourceError(f"mesh must be three positive integers, got {mesh}")
    n_cells = int(np.prod(mesh))
    if str(localise).lower() not in ("pipek-mezey", "none"):
        raise CorrelationSourceError(
            f"localise must be 'pipek-mezey' or 'none', got {localise!r}"
        )
    if not bool(getattr(result, "converged", False)):
        raise CorrelationSourceError("bvk_supercell_source: the reference SCF did not converge")
    if float(getattr(result, "smearing_temperature", 0.0) or 0.0) > 0.0:
        raise CorrelationSourceError(
            "bvk_supercell_source: the reference uses fractional (smeared) "
            "occupations; the engine needs an integer-aufbau determinant"
        )
    reference_kind, channels = _periodic_channels(system, result, n_frozen_per_cell)
    mo_k = getattr(result, "mo_coeffs" + channels[0][0], None)
    if not isinstance(mo_k, (list, tuple)) or len(mo_k) != n_cells:
        n = len(mo_k) if isinstance(mo_k, (list, tuple)) else 0
        raise CorrelationSourceError(
            f"bvk_supercell_source: the reference carries {n} k-points, the "
            f"mesh {mesh} needs {n_cells}"
        )
    kpoints = cyclic_gamma_mesh(system, mesh)
    want = np.asarray(kpoints.kpoints_cart, dtype=float).reshape(-1, 3)
    have = np.asarray(getattr(result, "kpoints_cart", np.zeros((0, 3))), dtype=float).reshape(-1, 3)
    if have.shape != want.shape or float(np.max(np.abs(have - want))) > 1e-10:
        raise CorrelationSourceError(
            "bvk_supercell_source: the reference mesh is not the Gamma-centred "
            "unreduced mesh in cyclic order (a Monkhorst-Pack mesh carries a "
            "half shift on even axes, a symmetry-reduced mesh keeps the "
            "irreducible wedge only). Run the SCF with "
            "kmesh=cyclic_gamma_mesh(system, mesh)."
        )
    cache = getattr(result, "fit_cache", None)
    if cache is None:
        raise CorrelationSourceError(
            "bvk_supercell_source needs the SCF's retained fit: "
            f"run_k{reference_kind}_periodic_gdf(..., fit_source='classic', retain_fit_cache=True)"
        )
    pairs = {}
    for i in range(n_cells):
        for j in range(n_cells):
            try:
                pairs[(i, j)] = cache.canonical_pair(i, j)
            except KeyError as exc:
                raise CorrelationSourceError(
                    f"bvk_supercell_source: the retained fit holds no (k_{i}, k_{j}) "
                    "pair; the fold needs every pair (k_exchange='gdf')"
                ) from exc
    translations = _cyclic_translations(mesh)
    kfrac = np.asarray(kpoints.kpoints_frac, dtype=float)
    S_sc, s_imag = _periodic_matrix_to_real_supercell(result.overlap, kfrac, translations)
    folded_channels = {}
    for suffix, n_occ_cell, weight in channels:
        mo_k, f_k, eps_k = _periodic_channel_data(
            result, suffix, n_cells, n_occ_cell, weight, int(basis.nbasis),
            n_frozen=int(n_frozen_per_cell),
        )
        F_sc, f_imag = _periodic_matrix_to_real_supercell(f_k, kfrac, translations)
        folded_channels[suffix] = (F_sc, f_imag, mo_k, eps_k, n_occ_cell)
    f_imag = max(data[1] for data in folded_channels.values())
    home_factors, cderi_imag, cderi_sym = _lpq_cache_to_real_home_auxiliary(pairs, kfrac, mesh)
    pairs.clear()
    worst = max(s_imag, f_imag, cderi_imag, cderi_sym)
    if not np.isfinite(worst) or worst > residual_tol:
        raise CorrelationSourceError(
            "bvk_supercell_source: the real-torus fold is not Hermitian to "
            f"{residual_tol:.0e} (overlap {s_imag:.2e}, Fock {f_imag:.2e}, "
            f"factors imaginary {cderi_imag:.2e}, symmetry {cderi_sym:.2e})"
        )

    # Magnetisation, like charge and electron count, is extensive. Using
    # the cell's multiplicity unchanged would correlate a different spin
    # determinant whenever the mesh has more than one point.
    multiplicity = 1 + n_cells * (int(system.multiplicity) - 1)
    super_system = _build_supercell_system(system, mesh, multiplicity=multiplicity)
    molecule = super_system.unit_cell_molecule()
    super_basis = BasisSet(molecule, basis.name)
    nbf_sc = int(super_basis.nbasis)
    if nbf_sc != S_sc.shape[0] or nbf_sc != home_factors.shape[1]:
        raise CorrelationSourceError(
            "bvk_supercell_source: the supercell basis ordering does not match "
            "the finite-character fold"
        )
    aux_name = aux_basis_name or getattr(result, "aux_basis_name", None)
    super_aux = make_aux_basis_set(molecule, aux_name=aux_name)
    if int(super_aux.nbasis) != n_cells * int(home_factors.shape[0]):
        raise CorrelationSourceError(
            "bvk_supercell_source: the supercell auxiliary ordering does not "
            "match the home-auxiliary fold"
        )
    factors = _BRealDFAdapter(home_factors, super_aux, mesh)

    # Real canonical supercell orbitals: the folded Fock in the canonical
    # orthogonal frame. Rank and eigenvalues must reproduce the per-k SCF.
    s_eig, s_vec = np.linalg.eigh(0.5 * (S_sc + S_sc.T))
    keep = s_eig > 1e-10 * float(s_eig[-1])
    X = s_vec[:, keep] / np.sqrt(s_eig[keep])[None, :]
    ncore_cell = int(getattr(result, "ecp_total_ncore", 0) or 0)
    n_frozen = int(n_frozen_per_cell) * n_cells
    orbital_data = {}
    for suffix, (F_sc, _, mo_k, eps_k, n_occ_cell) in folded_channels.items():
        n_orb = sum(np.asarray(c).shape[1] for c in mo_k)
        if int(np.count_nonzero(keep)) != n_orb:
            raise CorrelationSourceError(
                "bvk_supercell_source: the folded overlap rank "
                f"{int(np.count_nonzero(keep))} differs from the per-k orbital "
                f"count {n_orb} in channel {suffix!r}"
            )
        f_red = X.T @ F_sc @ X
        eps, c_red = np.linalg.eigh(0.5 * (f_red + f_red.T))
        C = X @ c_red
        eps_k = np.sort(np.concatenate([np.asarray(e).real for e in eps_k]))
        if eps_k.shape != eps.shape or not np.allclose(eps_k, eps, atol=1e-6, rtol=0):
            raise CorrelationSourceError(
                "bvk_supercell_source: the folded supercell eigenvalues do not "
                f"reproduce the per-k orbital energies in channel {suffix!r}"
            )
        if str(localise).lower() == "pipek-mezey":
            C = _localise_active_occupied(
                C, S_sc, super_basis, super_system, n_cells * n_occ_cell, n_frozen
            )
        orbital_data.update({
            "mo_coeffs" + suffix: C, "fock" + suffix: F_sc,
            "mo_energies" + suffix: eps,
        })
    reference_class = GammaUHFReference if reference_kind == "uhf" else GammaReference
    reference = reference_class(
        **orbital_data, overlap=S_sc,
        energy=float(result.energy) * n_cells, converged=True,
        ecp_total_ncore=ncore_cell * n_cells, localisation=str(localise).lower(),
    )
    exchange_q0 = str(getattr(result, "exchange_q0", None) or "ewald")
    executed = getattr(result, "executed_fit_source", None)
    conventions = SourceConventions(
        exchange_q0=exchange_q0, kmesh=mesh,
        bloch_normalisation="unnormalised-ao-bloch-sums-uniform-weights",
        same_source=True,
    )
    lattice = np.asarray(system.lattice, dtype=float)
    record = {
        "kind": "bvk-supercell",
        "reference_kind": reference_kind,
        "basis": getattr(basis, "name", None),
        "n_basis_cell": int(getattr(basis, "nbasis", 0) or 0),
        "aux_basis": aux_name,
        "n_aux_cell": int(home_factors.shape[0]),
        "dim": int(system.dim),
        "lattice": [[round(float(x), 10) for x in row] for row in lattice.tolist()],
        "atoms": [_atom_record(atom) for atom in getattr(system.unit_cell_molecule(), "atoms", [])],
        "charge": int(getattr(system, "charge", 0)),
        "multiplicity": int(system.multiplicity),
        "supercell_multiplicity": multiplicity,
        "mesh": list(mesh),
        "backend": str(getattr(result, "backend", "") or ""),
        "fit": {
            "engine": getattr(executed, "engine", None),
            "source": getattr(executed, "source", None),
            "source_signature": getattr(executed, "source_signature", None),
            "factor_sha256": getattr(executed, "factor_sha256", None),
            "auxiliary_frame": getattr(executed, "auxiliary_frame", None),
            "cache_frame": getattr(cache, "frame", None),
            "support_policy": getattr(cache, "support_policy", None),
        },
        "fold_residuals": {"overlap": s_imag, "fock": f_imag,
                           "factors_imaginary": cderi_imag, "factors_symmetry": cderi_sym},
        "n_frozen_per_cell": int(n_frozen_per_cell),
        "localisation": str(localise).lower(),
        "reference_energy_per_cell": f"{float(result.energy):.12f}",
        "conventions": conventions.as_record(),
    }
    return CorrelationSource(
        kind="bvk-supercell",
        reference_kind=reference_kind,
        molecule=molecule,
        basis=super_basis,
        reference=reference,
        factors=factors,
        overlap=S_sc,
        pair_distance_fn=toroidal_pair_distance_fn(system, mesh),
        conventions=conventions,
        provenance=record,
    )


__all__ = [
    "CONTRACT_VERSION",
    "CorrelationSource",
    "CorrelationSourceError",
    "GammaReference",
    "GammaUHFReference",
    "LpqFactors",
    "SourceConventions",
    "ThreeCenterFactors",
    "bvk_supercell_source",
    "detect_reference_kind",
    "gamma_gdf_source",
    "molecular_source",
    "refuse_unsupported_periodic_reference",
    "source_identity",
]
