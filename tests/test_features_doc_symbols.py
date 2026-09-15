"""Guard against phantom APIs in docs/features.md.

The 2026-06 docs audit found docs/features.md advertising helper names that did
not exist in the public API (run_gf2, run_ovgf, compute_atomisation_energy,
compute_gf_ip, compute_gf_ea). This test makes that class of drift mechanical:
every backticked top-level ``run_*`` / ``compute_*`` helper named in
docs/features.md must resolve as a ``vibeqc.*`` attribute. A new phantom name
fails the build instead of shipping in the capability surface the release paper
cites.
"""

import re
from pathlib import Path

import vibeqc

_FEATURES = Path(__file__).resolve().parent.parent / "docs" / "features.md"

# Backticked top-level helper identifiers of the form `run_xxx` / `compute_xxx`.
# Wildcards (e.g. `compute_gradient_periodic_*`) and dotted/sub-module paths are
# intentionally not matched -- this guard targets the flat public-helper surface
# where the phantom names appeared.
_HELPER_RE = re.compile(r"`((?:run|compute)_[a-z0-9_]+)`")

# Names that are documented in features.md but live somewhere other than the
# top-level vibeqc namespace (sub-modules / result-object methods). Each entry
# must be a *real* symbol -- this is an allowlist for resolution location, not a
# licence to document things that do not exist.
_ALLOW: set[str] = {
    # Real CCM helpers under vibeqc.periodic.ccm (module-level, NOT top-level
    # public): documented in features.md as the experimental Gamma-CCM
    # (aiccm2026dev-a) stream, qualified by `vibeqc.periodic.ccm`. They exist as
    # functions (periodic/ccm/{scf,dft,mp2,ccsd,uccsd}.py), just not on the
    # top-level vibeqc namespace.
    "run_ccm_rhf",
    "run_ccm_rks",
    "run_ccm_mp2",
    "run_ccm_ccsd",
    "run_ccm_uccsd",
    # A-namespace neutral fitted-torus controls: real functions
    # (periodic/ccm/{direct,ri,ccsd,mp2,ump2}.py), documented in features.md's
    # "A-namespace neutral fitted-torus controls" row alongside the row above.
    "run_ccm_rhf_direct",
    "run_ccm_rhf_gdf",
    "run_ccm_ri_ccsd",
    "run_ccm_ri_mp2",
    "run_ccm_ump2",
    # Multi-k GPW UKS driver: a real function
    # (vibeqc.periodic_gapw_open_shell.run_periodic_uks_gpw_multi_k),
    # documented as library-level in features.md (unlike its top-level RKS
    # sibling run_periodic_rks_gpw_multi_k). Allowlisted here so the doc may
    # name it in bare backticks without tripping the phantom guard.
    "run_periodic_uks_gpw_multi_k",
}


def test_features_md_helper_names_resolve():
    text = _FEATURES.read_text(encoding="utf-8")
    names = sorted(set(_HELPER_RE.findall(text)))
    assert names, "no run_*/compute_* helpers parsed from features.md -- regex stale?"
    missing = [n for n in names if not hasattr(vibeqc, n) and n not in _ALLOW]
    assert not missing, (
        "docs/features.md names run_*/compute_* helpers that are absent from the "
        f"public vibeqc namespace (phantom API surface): {missing}. Either wire "
        "the helper, fix the doc, or (if it is a real non-top-level symbol) add "
        "it to _ALLOW with a comment."
    )


def test_features_md_semiempirical_status_is_current():
    text = _FEATURES.read_text(encoding="utf-8")

    assert "H-Br (Z 1-35)" not in text
    assert "INDO H-Xe (Z 1-54)" in text
    assert "energy and analytic gradient for H, Li-F, Na-Cl" in text
    assert "source-parity SPDD terms for Al through Cl" in text
    assert "NDDO gradients use a mixed native-energy/Python-analytic route" in text
    assert "Al-Cl energy remains callable but scientifically unsupported" not in text
    assert "Gated experimental tight-binding semiempirical route" in text
    assert "scoped post-SCF native D4" in text
    assert "external `xtb` parity remain production gates" in text
