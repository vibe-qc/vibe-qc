"""vibe-qc data_library/ standard — schema + loader contracts.

Covers the v0.9.0 parameter-storage standard documented in
``python/vibeqc/data_library/README.md``. Tests live here (rather
than alongside the per-kind loaders) because they exercise
cross-cutting contracts every TOML file must satisfy regardless
of kind:

* Metadata block present + structurally complete (name, kind,
  citation, doi, license, status).
* File-stem matches metadata.name.
* Status enum is one of {complete, partial, pending}.
* `kind` discriminator routes to a known loader.
* File loads through tomllib without raising.
* Per-kind body validation delegates to that kind's loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

try:
    import tomllib
except ImportError:  # pragma: no cover
    import tomli as tomllib   # type: ignore[no-redef]

import vibeqc as vq


_DATA_LIB = Path(vq.__file__).resolve().parent / "data_library"

# Sub-directories that contain real parameter files (vs. future
# placeholders). Update as new kinds land in Phase 2.
_ACTIVE_KINDS = ["gcp"]


def _toml_files(subdir: str) -> Iterator[Path]:
    """Yield every TOML parameter file under data_library/<subdir>/,
    skipping templates / schemas (prefixed with ``_``)."""
    root = _DATA_LIB / subdir
    if not root.is_dir():
        return
    for path in sorted(root.glob("*.toml")):
        if path.name.startswith("_"):
            continue
        yield path


# ---------------------------------------------------------------------------
# data_library/ directory structure
# ---------------------------------------------------------------------------

def test_data_library_README_exists():
    assert (_DATA_LIB / "README.md").is_file()


def test_data_library_schema_exists():
    assert (_DATA_LIB / "_schema.toml").is_file()


def test_data_library_gcp_dir_exists():
    assert (_DATA_LIB / "gcp").is_dir()


def test_data_library_gcp_template_exists():
    assert (_DATA_LIB / "gcp" / "_template.toml").is_file()


# ---------------------------------------------------------------------------
# Per-file metadata contract — applies to every kind
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kind", _ACTIVE_KINDS)
def test_every_toml_has_required_metadata(kind):
    """Every shipped parameter file must declare the full metadata
    block. Loaders rely on these fields for license + citation
    surface; CI must catch a missing one."""
    required = {"name", "kind", "citation", "doi", "license", "status"}
    for path in _toml_files(kind):
        with path.open("rb") as fh:
            doc = tomllib.load(fh)
        meta = doc.get("metadata", {})
        missing = required - set(meta.keys())
        assert not missing, (
            f"{path.relative_to(_DATA_LIB)}: missing metadata fields "
            f"{sorted(missing)}"
        )


@pytest.mark.parametrize("kind", _ACTIVE_KINDS)
def test_file_stem_matches_metadata_name(kind):
    """Loaders resolve files by stem. The metadata.name field must
    match the stem so a `data_library/<kind>/<name>.toml` reverse-
    lookup always works."""
    for path in _toml_files(kind):
        with path.open("rb") as fh:
            doc = tomllib.load(fh)
        meta_name = doc["metadata"]["name"].lower()
        stem = path.stem.lower()
        assert stem == meta_name, (
            f"{path.relative_to(_DATA_LIB)}: stem {stem!r} != "
            f"metadata.name {meta_name!r}"
        )


@pytest.mark.parametrize("kind", _ACTIVE_KINDS)
def test_status_is_valid_enum(kind):
    """status must be one of complete / partial / pending — used by
    loaders to decide whether to populate-and-error or raise upfront."""
    valid = {"complete", "partial", "pending"}
    for path in _toml_files(kind):
        with path.open("rb") as fh:
            doc = tomllib.load(fh)
        status = doc["metadata"].get("status")
        assert status in valid, (
            f"{path.relative_to(_DATA_LIB)}: status {status!r} not in {valid}"
        )


@pytest.mark.parametrize("kind", _ACTIVE_KINDS)
def test_metadata_kind_matches_subdir(kind):
    """metadata.kind must align with the subdir the file lives in.
    Prevents misfiling a d3bj_damping file under gcp/ etc."""
    kind_to_metadata_kind = {
        "gcp": "gcp_parameters",
        # Future Phase 2 kinds extend here.
    }
    expected = kind_to_metadata_kind[kind]
    for path in _toml_files(kind):
        with path.open("rb") as fh:
            doc = tomllib.load(fh)
        meta_kind = doc["metadata"]["kind"]
        assert meta_kind == expected, (
            f"{path.relative_to(_DATA_LIB)}: kind {meta_kind!r} does not "
            f"match expected {expected!r} for subdir {kind!r}"
        )


# ---------------------------------------------------------------------------
# gCP-specific loader integration
# ---------------------------------------------------------------------------

def test_gcp_loader_loaded_all_toml_files():
    """The gCP TOML files in data_library/gcp/ all register through
    the gcp.py loader and surface via vq.available_gcp_basis_sets()."""
    toml_names = {p.stem for p in _toml_files("gcp")}
    bundled = set(vq.available_gcp_basis_sets())
    # The loader may also use lowercase canonical names from
    # metadata.name; compare in case-insensitive form.
    assert {n.lower() for n in toml_names} == {n.lower() for n in bundled}


def test_gcp_def2_svp_full_h_kr_coverage():
    """Post-v0.9.0 bundling: def2-svp ships full H–Kr (Z=1..36)
    coverage from the mctc-gcp Fortran reference. The (σ, η, α, β)
    fit constants are (0.2054, 1.3157, 0.8136, 1.2572) per the HF/SVP
    case in mctc-gcp src/gcp.f90."""
    p = vq.gcp_params_for("def2-svp")
    assert p is not None
    assert sorted(p.e_mis.keys()) == list(range(1, 37))
    assert sorted(p.n_virt.keys()) == list(range(1, 37))
    assert abs(p.sigma - 0.2054) < 1e-12
    assert abs(p.eta - 1.3157) < 1e-12
    assert abs(p.alpha - 0.8136) < 1e-12
    assert abs(p.beta - 1.2572) < 1e-12
    assert "Kruse & Grimme" in p.citation
    assert "10.1063/1.3700154" in p.citation


def test_gcp_minix_full_h_kr_coverage():
    """Post-v0.9.0: MINIX (HF-3c parent) now ships full per-element
    data extracted from mctc-gcp gcp.f90's 'hf3c'/'minix' case
    (composite of MINIS for H-Mg, MINIS+d for Al-Ar, SV for K-Zn,
    SVP for Ga-Kr, with p-augmentation on Li/Be/Na/Mg)."""
    p = vq.gcp_params_for("minix")
    assert p is not None
    assert sorted(p.e_mis.keys()) == list(range(1, 37))
    assert "Sure & Grimme" in p.citation
    # Sample published values per mctc-gcp src/gcp.f90 lines 922-928 +
    # 976-981. Cross-checks the extractor's data-block parsing.
    assert abs(p.e_mis[6] - 0.27995) < 1e-5      # C: from HFminis
    assert abs(p.e_mis[14] - 1.61098) < 1e-5     # Si: from HFminisd
    assert abs(p.e_mis[3] - 0.177871) < 1e-6     # Li: p-augmented override


def test_gcp_def2_mtzvpp_uses_canonical_mctc_gcp_params():
    """Post-v0.9.0: def2-mTZVPP (r²SCAN-3c) uses the canonical mctc-gcp
    parameters (σ, η, α, β) = (1.0, 1.315, 0.941, 1.4636) — NOT zero
    by design as the earlier first-cut assumed. Sample e_mis for B / O
    cross-checks the data-block parsing."""
    p = vq.gcp_params_for("def2-mtzvpp")
    assert p is not None
    assert (p.sigma, p.eta, p.alpha, p.beta) == (1.0, 1.315, 0.941, 1.4636)
    assert abs(p.e_mis[5] - 0.2) < 1e-12          # B
    assert abs(p.e_mis[8] - 0.08) < 1e-12         # O
