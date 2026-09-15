"""MACE foundation-model registry -- names, licenses, citations, provenance.

Single source of truth mapping a user-facing model key to: which MACE
loader to use (``mace_mp`` materials vs ``mace_off`` organic), the argument
passed to that loader, the model's license (MIT vs ASL), domain + element
coverage, the foundation-model citation key, and a DOI. The ASL gate
(:mod:`vibeqc.mlip.mace`) and the citation / provenance plumbing both read
this table.

Licenses (verified against ``ACEsuit/mace-foundations`` 2026-06):

* **MIT** -- MACE-MP-0 / MPA-0 (materials). Commercial use OK; redistributable.
* **ASL** -- MACE-OFF23 (organic) + OMAT / MATPES / MH / MDP. Academic,
  **non-commercial** (https://github.com/gabor1/ASL). Gated behind an
  explicit acknowledgment.

Models not listed here are rejected.  A license acknowledgment is not a
general-purpose escape hatch for arbitrary URLs, local files, or newer
upstream model families: each supported weight family needs an explicit
registry entry, license decision, citation, and scientific scope.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vibeqc.output.formats.toml import write_toml


@dataclass(frozen=True)
class MaceModelInfo:
    """Provenance + routing for one MACE foundation model."""

    key: str  # canonical user-facing key (e.g. "medium-mpa-0")
    loader: str  # "mace_mp" (materials) | "mace_off" (organic)
    loader_arg: str  # value passed to the loader's model= argument
    license: str  # "MIT" | "ASL"
    domain: str  # "materials" | "organic" | "unknown"
    elements: int  # element-coverage count (0 = unknown)
    training_data: str
    theory: str
    citation: str  # foundation-model citation entry key (database.toml); "" if none
    doi: str
    elements_z: "frozenset[int] | None" = None  # covered Z; None = unrestricted/unknown

    @property
    def is_academic_only(self) -> bool:
        """True for ASL (academic, non-commercial) models -- gated."""
        return self.license.upper() == "ASL"

    def unsupported_elements(self, atomic_numbers) -> "list[int]":
        """Atomic numbers in the system this model does NOT cover (empty
        when its element coverage is unrestricted / unknown)."""
        if self.elements_z is None:
            return []
        return sorted({int(z) for z in atomic_numbers if int(z) not in self.elements_z})


# Canonical models. mace_mp materials models are MIT; mace_off organic
# models are ASL. The default (medium-mpa-0) is MIT.
# Each row's ``doi`` is the *version of record* of its citation entry
# (JCP 163, 184110 (2025) for the MP family; JACS 147, 17598 (2025) for
# OFF23) -- the same DOI the .bibtex sibling emits, because runner.py prints
# ``Reference: doi:<doi> (cited in the .bibtex sibling)`` from this field
# (#523). tests/test_mlip_mace.py pins registry doi == database doi.
_MP = "batatia_mace_mp_2024"
_OFF = "kovacs_mace_off_2023"
# MACE-OFF23 organic coverage: H C N O F P S Cl Br I -- the "ten most
# important chemical elements for organic chemistry" (Kovács et al. 2023,
# arXiv:2312.15211, Sec.II.A). Reference level is wB97M-D3(BJ)/def2-TZVPPD
# (the SPICE reference level, Sec.II.B) -- note def2-TZVPP*D*, the
# diffuse-augmented Rappoport-Furche basis, NOT plain def2-TZVPP.
_OFF23_Z = frozenset({1, 6, 7, 8, 9, 15, 16, 17, 35, 53})
# Current MACE-MP/MPA-0 loader table: H through Bi plus Ac through Pu
# (89 elements total). Po-Ra and elements beyond Pu are absent. This was
# checked against the upstream calculator's ``z_table`` for the registered
# medium-mpa-0 weights; keeping the explicit set lets vibe-qc reject an
# unsupported atomic number before loading Torch or downloading weights.
_MP_Z = frozenset((*range(1, 84), *range(89, 95)))
_MODELS = (
    # --- MIT materials (mace_mp) ---
    MaceModelInfo("medium-mpa-0", "mace_mp", "medium-mpa-0", "MIT", "materials",
                  89, "MPTrj + sAlex", "DFT (PBE+U)", _MP, "10.1063/5.0297006",
                  elements_z=_MP_Z),
    MaceModelInfo("small", "mace_mp", "small", "MIT", "materials",
                  89, "MPTrj", "DFT (PBE+U)", _MP, "10.1063/5.0297006",
                  elements_z=_MP_Z),
    MaceModelInfo("medium", "mace_mp", "medium", "MIT", "materials",
                  89, "MPTrj", "DFT (PBE+U)", _MP, "10.1063/5.0297006",
                  elements_z=_MP_Z),
    MaceModelInfo("large", "mace_mp", "large", "MIT", "materials",
                  89, "MPTrj", "DFT (PBE+U)", _MP, "10.1063/5.0297006",
                  elements_z=_MP_Z),
    # --- ASL organic (mace_off) -- academic, NON-COMMERCIAL ---
    MaceModelInfo("off23-small", "mace_off", "small", "ASL", "organic",
                  10, "SPICE", "DFT (wB97M-D3(BJ)/def2-TZVPPD)", _OFF, "10.1021/jacs.4c07099",
                  elements_z=_OFF23_Z),
    MaceModelInfo("off23-medium", "mace_off", "medium", "ASL", "organic",
                  10, "SPICE", "DFT (wB97M-D3(BJ)/def2-TZVPPD)", _OFF, "10.1021/jacs.4c07099",
                  elements_z=_OFF23_Z),
    MaceModelInfo("off23-large", "mace_off", "large", "ASL", "organic",
                  10, "SPICE", "DFT (wB97M-D3(BJ)/def2-TZVPPD)", _OFF, "10.1021/jacs.4c07099",
                  elements_z=_OFF23_Z),
)

MACE_MODELS: dict[str, MaceModelInfo] = {m.key: m for m in _MODELS}

# User-friendly aliases -> canonical key.
_ALIASES = {
    "mpa-0": "medium-mpa-0",
    "mpa": "medium-mpa-0",
    "mace-mp": "medium-mpa-0",
    "mp-0": "medium",
    "off23": "off23-medium",
    "off": "off23-medium",
    "mace-off": "off23-medium",
}

# MIT-licensed MACE-MPA-0 (materials, 89 elements).
DEFAULT_MODEL = "medium-mpa-0"


def resolve_model(key: str | None) -> MaceModelInfo:
    """Resolve a user model key (or alias) to a :class:`MaceModelInfo`.

    Unknown keys fail closed.  The supported wrapper scope is deliberately
    limited to the registered MACE-MP/MPA-0 and MACE-OFF23 weights; accepting
    an academic license does not authorize an unregistered model family,
    URL, or local weight path.
    """
    if key is None:
        key = DEFAULT_MODEL
    norm = key.strip().lower()
    norm = _ALIASES.get(norm, norm)
    info = MACE_MODELS.get(norm)
    if info is not None:
        return info
    supported = ", ".join(MACE_MODELS)
    raise ValueError(
        f"Unsupported MACE model {key!r}. vibe-qc accepts only registered "
        f"MACE-MP/MPA-0 and MACE-OFF23 weights: {supported}. Arbitrary "
        "URLs, local paths, OMAT, MATPES, MH, MDP, and other unregistered "
        "weights are outside the supported and license-reviewed scope."
    )


def mace_model_registry() -> dict[str, Any]:
    """Return the registered MACE foundation-model metadata.

    The mapping is TOML-serialisable and intentionally contains only the
    v0.15-supported wrapper scope: MACE-MP/MPA-0 and MACE-OFF23 families.
    """
    return {
        "schema_version": 1,
        "kind": "vibeqc.mace.model_registry",
        "metadata": {
            "scope": (
                "v0.15 MACE wrapper registry; newer upstream foundation weights "
                "require explicit implementation and license-gate entries."
            ),
            "default_model": DEFAULT_MODEL,
            "unknown_model_policy": "Unknown and unregistered keys are rejected.",
        },
        "aliases": dict(sorted(_ALIASES.items())),
        "model": [
            {
                "key": info.key,
                "loader": info.loader,
                "loader_arg": info.loader_arg,
                "license": info.license,
                "academic_only": info.is_academic_only,
                "domain": info.domain,
                "elements": info.elements,
                "elements_z": (
                    sorted(info.elements_z) if info.elements_z is not None else []
                ),
                "element_coverage": (
                    "listed" if info.elements_z is not None else "loader-defined"
                ),
                "training_data": info.training_data,
                "theory": info.theory,
                "citation": info.citation,
                "doi": info.doi,
            }
            for info in _MODELS
        ],
    }


def save_mace_model_registry(path: str | Path) -> None:
    """Write the registered MACE foundation-model metadata as TOML."""
    write_toml(mace_model_registry(), path)
