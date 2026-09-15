"""The conductor surface: what COSMO hands to COSMO-RS.

COSMO-RS does not see a Hamiltonian, a basis, or a density. It sees a set of
surface segments, each carrying an area and a screening charge density

    sigma_nu = q_nu / s_nu                                     [e / bohr^2]

computed in the **conductor limit** (``epsilon = inf``, screening factor
exactly 1). That is the "ideally screened molecule" of Klamt 1995 section 3.1:
the starting point of the whole theory, and the reason the surface record is
the layer boundary rather than an implementation detail.

Everything downstream -- sigma profiles, the sigma potential, chemical
potentials, activity coefficients -- is a functional of this record plus a
parameterization. Nothing downstream may reach back into the electronic
structure.

Provenance is not decoration
----------------------------
A COSMO-RS parameterization is inseparable from the electronic method, basis,
cavity construction, radii and post-processing it was fitted against. Klamt
et al. 1998 fitted the classic set against DMol/BPW91/DNP with COSMO at
``f(eps) = 1`` and NSPA = 92 segments on their own optimized radii; applying
those constants to a surface built another way is a transfer across protocols,
and the size of the resulting error is not knowable from the numbers alone.

So the record carries the recipe that produced it, and a parameterization
carries the recipe it was fitted to. That makes the comparison possible at all
-- and the run reports which pair was used, rather than leaving a reader to
assume they matched.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np

SURFACE_SCHEMA_VERSION = 1

# Segment areas below this (bohr^2) carry no meaningful screening charge
# density: sigma = q/s amplifies the noise in a near-zero-weight segment
# without contributing area to any profile.
MIN_SEGMENT_AREA_BOHR2 = 1e-10


@dataclass(frozen=True)
class SurfaceProvenance:
    """How a surface was produced. Compared against a parameterization's.

    Every field is what a later reader needs to decide whether a given
    parameterization may be applied. ``cavity_recipe`` is free-form because
    cavity constructions differ in kind, not merely in parameter values.
    """

    method: str
    basis: Optional[str]
    cavity_kind: str
    cavity_recipe: dict[str, Any]
    screening_variant: str
    epsilon: float
    charge: int
    multiplicity: int
    code: str = "vibe-qc"
    code_version: Optional[str] = None
    notes: Optional[str] = None

    def descriptor(self) -> str:
        """A short protocol token, e.g. ``rhf/sto-3g/lebedev-csc``."""
        basis = self.basis or "n/a"
        return f"{self.method}/{basis}/{self.cavity_kind}"


@dataclass(frozen=True)
class ConductorSurface:
    """Ideally screened molecular surface -- the COSMO-RS input record.

    Attributes
    ----------
    positions : ndarray (n_seg, 3), bohr
        Segment representation points.
    areas : ndarray (n_seg,), bohr^2
        Segment areas. Sum is :attr:`total_area`.
    normals : ndarray (n_seg, 3)
        Outward unit normals.
    segment_atom : ndarray (n_seg,), int
        Owning atom index. Segments move rigidly with their owner, so this is
        what makes a geometry derivative expressible.
    charges : ndarray (n_seg,), e
        Conductor screening charges ``q*`` (``epsilon = inf``).
    atomic_numbers : ndarray (n_atom,), int
    atom_positions : ndarray (n_atom, 3), bohr
    energy_gas, energy_conductor : float, Hartree
        The pair whose difference is the ideal screening energy
        ``Delta^X = E_gas - E_COSMO`` (Klamt 1998 eq. 4).
    provenance : SurfaceProvenance
    conformer_energy, conformer_degeneracy
        Reserved for ensembles; a single conformer is weight 1.
    """

    positions: np.ndarray
    areas: np.ndarray
    normals: np.ndarray
    segment_atom: np.ndarray
    charges: np.ndarray
    atomic_numbers: np.ndarray
    atom_positions: np.ndarray
    energy_gas: Optional[float]
    energy_conductor: Optional[float]
    provenance: SurfaceProvenance
    cavity_volume: Optional[float] = None
    conformer_energy: Optional[float] = None
    conformer_degeneracy: int = 1
    schema_version: int = SURFACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        n = self.positions.shape[0]
        for name, arr, shape in (
            ("areas", self.areas, (n,)),
            ("normals", self.normals, (n, 3)),
            ("segment_atom", self.segment_atom, (n,)),
            ("charges", self.charges, (n,)),
        ):
            if tuple(np.shape(arr)) != shape:
                raise ValueError(
                    f"ConductorSurface: {name} has shape "
                    f"{tuple(np.shape(arr))}, expected {shape} for "
                    f"{n} segments."
                )
        if np.any(np.asarray(self.areas) < 0.0):
            raise ValueError("ConductorSurface: negative segment area.")

    # -- derived quantities ------------------------------------------

    @property
    def n_segments(self) -> int:
        return int(self.positions.shape[0])

    @property
    def total_area(self) -> float:
        """``A^X`` in bohr^2 -- the COSMO area of Klamt 1998 Table 1."""
        return float(np.sum(self.areas))

    @property
    def sigma(self) -> np.ndarray:
        """Screening charge density ``sigma = q / s`` (e / bohr^2).

        Zero on degenerate segments rather than infinite: a segment with no
        area carries no charge to spread, and letting ``0/0`` through would
        poison every downstream profile with NaN.
        """
        areas = np.asarray(self.areas, dtype=np.float64)
        charges = np.asarray(self.charges, dtype=np.float64)
        out = np.zeros_like(charges)
        ok = areas > MIN_SEGMENT_AREA_BOHR2
        out[ok] = charges[ok] / areas[ok]
        return out

    @property
    def total_charge(self) -> float:
        return float(np.sum(self.charges))

    @property
    def ideal_screening_energy(self) -> Optional[float]:
        """``Delta^X = E_gas - E_COSMO`` (Klamt 1998 eq. 4), Hartree.

        ``None`` when either endpoint is absent; the caller must not silently
        substitute zero, which would read as "no screening gain".
        """
        if self.energy_gas is None or self.energy_conductor is None:
            return None
        return float(self.energy_gas) - float(self.energy_conductor)

    # -- serialization -----------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Plain-Python form; round-trips exactly through :meth:`from_dict`."""
        return {
            "schema_version": self.schema_version,
            "positions": np.asarray(self.positions).tolist(),
            "areas": np.asarray(self.areas).tolist(),
            "normals": np.asarray(self.normals).tolist(),
            "segment_atom": [int(i) for i in np.asarray(self.segment_atom)],
            "charges": np.asarray(self.charges).tolist(),
            "atomic_numbers": [int(z) for z in np.asarray(self.atomic_numbers)],
            "atom_positions": np.asarray(self.atom_positions).tolist(),
            "energy_gas": self.energy_gas,
            "energy_conductor": self.energy_conductor,
            "cavity_volume": self.cavity_volume,
            "conformer_energy": self.conformer_energy,
            "conformer_degeneracy": int(self.conformer_degeneracy),
            "provenance": asdict(self.provenance),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ConductorSurface":
        version = int(payload.get("schema_version", 0))
        if version != SURFACE_SCHEMA_VERSION:
            raise ValueError(
                f"ConductorSurface: unsupported schema version {version} "
                f"(this build reads version {SURFACE_SCHEMA_VERSION}). "
                f"Refusing to guess at a different layout."
            )
        return cls(
            positions=np.asarray(payload["positions"], dtype=np.float64),
            areas=np.asarray(payload["areas"], dtype=np.float64),
            normals=np.asarray(payload["normals"], dtype=np.float64),
            segment_atom=np.asarray(payload["segment_atom"], dtype=int),
            charges=np.asarray(payload["charges"], dtype=np.float64),
            atomic_numbers=np.asarray(payload["atomic_numbers"], dtype=int),
            atom_positions=np.asarray(payload["atom_positions"], dtype=np.float64),
            energy_gas=payload["energy_gas"],
            energy_conductor=payload["energy_conductor"],
            cavity_volume=payload.get("cavity_volume"),
            conformer_energy=payload.get("conformer_energy"),
            conformer_degeneracy=int(payload.get("conformer_degeneracy", 1)),
            provenance=SurfaceProvenance(**payload["provenance"]),
        )

    def to_json(self, *, indent: int = 1) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "ConductorSurface":
        return cls.from_dict(json.loads(text))


def segment_normals(
    positions: np.ndarray,
    segment_atom: np.ndarray,
    atom_positions: np.ndarray,
) -> np.ndarray:
    """Outward unit normals for sphere-based cavities.

    For a segment on atom ``a``, the outward normal is the unit vector from
    the atom centre to the segment. Exact for any union-of-spheres cavity,
    which is every construction vibe-qc ships. A segment coincident with its
    own atom centre has no defined normal and is returned as zero rather than
    NaN.
    """
    pos = np.asarray(positions, dtype=np.float64)
    owner = np.asarray(segment_atom, dtype=int)
    centres = np.asarray(atom_positions, dtype=np.float64)[owner]
    d = pos - centres
    norm = np.linalg.norm(d, axis=1)
    out = np.zeros_like(d)
    ok = norm > 1e-12
    out[ok] = d[ok] / norm[ok, None]
    return out


def build_conductor_surface(
    solvent_result,
    *,
    method: str,
    basis: Optional[str] = None,
    cavity_kind: str = "lebedev-csc",
    cavity_recipe: Optional[dict[str, Any]] = None,
    energy_gas: Optional[float] = None,
    energy_conductor: Optional[float] = None,
    charge: int = 0,
    multiplicity: int = 1,
    notes: Optional[str] = None,
) -> ConductorSurface:
    """Project a converged solvation result onto the COSMO-RS input record.

    The charges are taken as stored. **They are only the conductor charges
    ``q*`` when the run was made at ``epsilon = inf``**; at finite dielectric
    vibe-qc stores the scaled ``q = f q*``, and a sigma profile built from
    those is screened, not ideal. This function therefore records the
    screening variant and epsilon in the provenance instead of silently
    rescaling: undoing the scaling is only correct if the cavity and the
    outlying-charge treatment also match, and that is a decision for the
    caller who knows the protocol, not for a projection helper.
    """
    cavity = solvent_result.cavity
    positions = np.asarray(cavity.points, dtype=np.float64)
    owner = np.asarray(cavity.point_atom, dtype=int)
    atom_pos = np.asarray(cavity.atom_positions, dtype=np.float64)
    screening = getattr(solvent_result, "screening", None)

    recipe = dict(cavity_recipe or {})
    recipe.setdefault("n_points_per_sphere", getattr(cavity, "n_points_per_sphere", None))
    recipe.setdefault("atom_radii_bohr", np.asarray(cavity.atom_radii).tolist())

    return ConductorSurface(
        positions=positions,
        areas=np.asarray(cavity.weights, dtype=np.float64),
        normals=segment_normals(positions, owner, atom_pos),
        segment_atom=owner,
        charges=np.asarray(solvent_result.cpcm.q, dtype=np.float64),
        atomic_numbers=np.asarray(
            getattr(cavity, "atom_numbers", np.zeros(atom_pos.shape[0], dtype=int)),
            dtype=int,
        ),
        atom_positions=atom_pos,
        energy_gas=(
            float(energy_gas) if energy_gas is not None
            else _maybe_float(getattr(solvent_result, "e_gas", None))
        ),
        energy_conductor=(
            float(energy_conductor) if energy_conductor is not None else None
        ),
        provenance=SurfaceProvenance(
            method=method,
            basis=basis,
            cavity_kind=cavity_kind,
            cavity_recipe=recipe,
            screening_variant=(
                screening.variant if screening is not None
                else getattr(solvent_result, "solvent_variant", "unknown")
            ),
            epsilon=float(getattr(solvent_result, "epsilon", float("nan"))),
            charge=int(charge),
            multiplicity=int(multiplicity),
            notes=notes,
        ),
    )


def _maybe_float(value: Any) -> Optional[float]:
    return None if value is None else float(value)


__all__ = [
    "MIN_SEGMENT_AREA_BOHR2",
    "SURFACE_SCHEMA_VERSION",
    "ConductorSurface",
    "SurfaceProvenance",
    "build_conductor_surface",
    "segment_normals",
]
