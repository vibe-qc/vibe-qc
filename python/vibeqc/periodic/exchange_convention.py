"""Exchange-``q=0`` finite-size convention — a machine-recorded provenance gate.

On a finite Born–von-Kármán torus the Coulomb / ``J`` kernel is *uniquely forced*
(zero-average ``G=0`` removal), but the **exchange-``q=0``** (``G+q=0``) treatment
is an *additional* finite-size convention: ``strict-zero-mode`` and
``BvK-Madelung`` (``exxdiv="ewald"``) exchange are **distinct finite Hamiltonians**
with the *same* thermodynamic limit. A shared seam is a necessary comparison
constraint, not proof that Γ-CCM, χ-CCM, or a neutral fitted-torus control use
the same finite Hamiltonian. Historical neutral-control and external KRHF
agreements at ``exxdiv="ewald"`` are same-operator evidence only.

Crucially the seam shifts the HF occupied/virtual spectrum, so **finite-``N``
correlation** (MP2 / KMP2 / CCSD(T) / DLPNO) inherits the same convention
dependence *through the denominators*. An ``A``-vs-``B``-vs-``KMP2`` comparison is
therefore only meaningful at a **matched** convention. This module makes the
convention a ``.system`` manifest field plus a comparison-time assertion — a
reproducibility gate, not a footnote.

References
----------
``docs/manuscripts/aiccm_a_position.md`` §0.5 item 1 (the converged joint result)
and ``docs/manuscripts/aiccm_a_proposed_comparison_text.md`` point 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

__all__ = [
    "BVK_EWALD",
    "STRICT_ZERO",
    "WIRE_FREE_SEAM",
    "exchange_q0_label",
    "read_exchange_q0",
    "assert_matched_exchange_q0",
]

#: The BvK-Madelung exchange convention (``exxdiv="ewald"``), fixed by χ-CCM-B
#: and used by the neutral fitted-torus/KRHF controls. Γ-CCM routes must declare
#: their own active convention rather than inherit this identity by name.
BVK_EWALD = "BvK-ewald"

#: The legitimate alternative: the strict ``G+q=0`` zero-mode drop (the legacy
#: ``exxdiv=None`` Γ-supercell helper). A distinct finite Hamiltonian.
STRICT_ZERO = "strict-zero-mode"

#: Wire (``d=1`` mixed-boundary) only. Subtract just the **infrared-divergent** part
#: of the ``S⊗S`` monopole, keeping its free-space (molecular) content, which is
#: ordinary exchange rather than a quadrature artifact. :data:`STRICT_ZERO` deletes
#: both, so it misses the non-interacting limit by a constant that does not vanish
#: with the period; ``exxdiv=None`` keeps both, so its total diverges with the
#: infrared cut. This convention is infrared-stable *and* reaches the isolated-H₂
#: limit as ``O(1/L)`` (measured 2026-07-10; see
#: :func:`~vibeqc.periodic.ccm.lowd_four_center.ccm_wire_free_ss_weight`).
WIRE_FREE_SEAM = "wire-free-space-seam"

# Canonical map from an SCF ``exxdiv`` setting to the exchange-q=0 convention.
_EXXDIV_TO_Q0 = {
    "ewald": BVK_EWALD,
    "bvk-ewald": BVK_EWALD,
    None: STRICT_ZERO,
    "none": STRICT_ZERO,
    "": STRICT_ZERO,
    "strict-zero-mode": STRICT_ZERO,
    "strict-zero": STRICT_ZERO,
    "free": WIRE_FREE_SEAM,
    "wire-free-space-seam": WIRE_FREE_SEAM,
}


def exchange_q0_label(exxdiv: str | None) -> str:
    """Canonical exchange-``q=0`` convention label for an SCF ``exxdiv`` setting.

    ``"ewald"`` → :data:`BVK_EWALD`; ``None`` / ``"none"`` / ``""`` /
    ``"strict-zero[-mode]"`` → :data:`STRICT_ZERO`. An unrecognised value is
    recorded verbatim as ``f"exxdiv:{exxdiv}"`` (so it is never silently lost and
    still trips :func:`assert_matched_exchange_q0` against a mismatched run) rather
    than raising — the manifest field must always land.
    """
    key = exxdiv.lower() if isinstance(exxdiv, str) else exxdiv
    if key in _EXXDIV_TO_Q0:
        return _EXXDIV_TO_Q0[key]
    return f"exxdiv:{exxdiv}"


def read_exchange_q0(system_manifest: str | Path) -> str | None:
    """Read ``[run].exchange_q0`` from a ``.system`` manifest (``None`` if absent).

    ``system_manifest`` is a path to a ``{stem}.system`` TOML file (or a directory
    is **not** accepted — pass the file). Returns the recorded convention label, or
    ``None`` if the field / file is missing.
    """
    import tomllib

    p = Path(system_manifest)
    if not p.is_file():
        return None
    with p.open("rb") as fh:
        data = tomllib.load(fh)
    run = data.get("run")
    if not isinstance(run, dict):
        return None
    val = run.get("exchange_q0")
    return str(val) if val is not None else None


def assert_matched_exchange_q0(
    items: Iterable[str | Path], *, context: str = "comparison"
) -> str:
    """Assert every run shares one exchange-``q=0`` convention; return it.

    ``items`` is an iterable of either convention **labels** (e.g.
    ``"BvK-ewald"``) or paths to ``.system`` manifests (anything that exists on
    disk as a file is read via :func:`read_exchange_q0`). Raises :class:`ValueError`
    on a mismatch — finite-``N`` correlation is convention-dependent through the HF
    gap, so ``A``/``B``/``KMP2`` parity is only meaningful at a matched convention.

    Returns the single shared label. A run whose label could not be determined
    (manifest missing the field) is reported as ``"<unknown>"`` and, if it differs
    from the others, trips the assertion (an undeclared convention is itself a
    reproducibility failure).
    """
    labels: list[str] = []
    for it in items:
        label: str | None = None
        if isinstance(it, Path) or (isinstance(it, str) and Path(it).is_file()):
            label = read_exchange_q0(it)
        else:
            label = str(it)
        labels.append(label if label is not None else "<unknown>")

    if not labels:
        raise ValueError(f"{context}: no runs supplied to the exchange_q0 match check")

    unique = set(labels)
    if len(unique) > 1:
        raise ValueError(
            f"{context}: exchange-q=0 convention mismatch — runs carry {labels}. "
            f"The G+q=0 exchange seam shifts the HF gap, hence every MP2/KMP2/"
            f"CCSD/DLPNO denominator, so A/B/KMP2 parity is only meaningful at a "
            f"matched convention. Re-run the odd one(s) with the same declared "
            f"exxdiv (for BvK-Madelung: exxdiv='ewald' → '{BVK_EWALD}')."
        )
    return labels[0]
