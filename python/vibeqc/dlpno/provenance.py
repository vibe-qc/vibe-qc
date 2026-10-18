"""Provenance of an executed DLPNO coupled-cluster run, shared by the runners.

What a DLPNO-CCSD(T) result claims about itself has to match what ran:
which triples algorithm (if any) produced ``e_t``, which cutoffs were in
force, and therefore which published sources the run may cite. These
helpers answer those questions from the options and the result, so the
molecular runner (:mod:`vibeqc.runner`) and the periodic front door
(:mod:`vibeqc.periodic_runner`) describe one engine the same way in the
``.out`` block, the ``.system`` manifest, the structured log and the
citation artifacts.
"""

from __future__ import annotations

from typing import Any

#: The DLPNO truncation knobs, ordered as the publications tabulate them:
#: TCutPairs, TCutPNO, TCutMKN (Liakos, Sparta, Kesharwani, Martin and
#: Neese, J. Chem. Theory Comput. 11, 1525 (2015), doi:10.1021/ct501129s,
#: Table 1), followed by vibe-qc's own additional cutoffs.
THRESHOLD_ORDER = (
    "tcut_pairs",
    "tcut_pno",
    "tcut_mkn",
    "tcut_pairs_weak",
    "tcut_pno_weak",
    "tcut_pno_singles",
    "tcut_tno",
)


def threshold_settings(options: Any, provenance: Any) -> dict[str, float]:
    """Return every cutoff disclosed for the route in artifact order.

    ``provenance.applied`` owns the published triple and the derived MP2 weak
    cutoffs. Local CC routes additionally carry solver-specific singles/TNO
    cutoffs outside that published request. Keeping those settings beside the
    named provenance makes the human, JSONL, and TOML artifacts mutually
    reproducible without pretending they are Liakos preset coordinates.
    """

    applied = dict(provenance.applied)
    settings: dict[str, float] = {}
    for name in THRESHOLD_ORDER:
        if name in applied:
            settings[name] = float(applied[name])
        elif name in ("tcut_pno_singles", "tcut_tno"):
            if name == "tcut_tno" and (
                not bool(getattr(options, "compute_triples", False))
                or str(getattr(options, "triples_mode", "")).lower()
                == "exact"
            ):
                continue
            value = getattr(options, name, None)
            if value is not None:
                settings[name] = float(value)
    return settings


def triples_provenance(options: Any, result: Any = None) -> tuple[str, str]:
    """Return the executed triples mode and an artifact-safe description."""

    if not bool(getattr(options, "compute_triples", False)):
        return "none", "disabled"
    if (
        result is not None
        and hasattr(result, "triples_executed")
        and not bool(result.triples_executed)
    ):
        return "none", "not executed for this result"
    raw_mode = getattr(options, "triples_mode", None)
    mode = str(raw_mode).strip().lower() if raw_mode is not None else "pilot"
    algorithms = {
        "local": "DLPNO-(T0), diagonal local occupied-Fock",
        "t0": "DLPNO-(T0), diagonal local occupied-Fock",
        "t1": "rotated-occupied (T1) compatibility correction",
        "t1-iterative": (
            "experimental dense spin-orbital generalization of Guo Eq. (2), "
            "iterative local-basis DLPNO-(T1)"
        ),
        "exact": "canonical (T) contraction on local amplitudes",
        "pilot": "dense correctness-pilot (T)",
    }
    return mode, algorithms.get(mode, f"unrecognized mode {mode!r}")


def triples_citation_entries(
    method: str,
    options: Any,
    result: Any = None,
) -> tuple[str, ...]:
    """Return the triples sources that depend on the executed algorithm.

    Both Guo sources only when iterative local ``(T1)`` runs. The 2018
    closed-shell Eq. (2) keeps occupied triples in the localised basis and
    iterates their off-diagonal occupied-Fock coupling. The 2020 paper is the
    dedicated open-shell extension. The compatibility ``"t1"`` routes instead
    diagonalise/rotate the occupied space. Static method routes therefore
    cannot cite either source without over-claiming T0, rotated/canonical,
    exact, pilot, or skipped executions.

    Rendell, Lee and Komornicki (1991) only when the closed-shell local
    ``(T0)`` runs in its compiled kernel, which evaluates their W/V spatial
    expression; the canonical spatial (T) behind ``"t1"`` and ``"exact"`` is
    spin-integrated by spin class instead.

    The open-shell local ``(T0)`` builds per-triple TNO spaces from the
    three pair spaces and keeps the diagonal localised Fock in the
    denominators, Guo et al.'s (2020) open-shell DLPNO-CCSD(T0), and both
    it and the rotated open-shell ``"t1"`` evaluate the spin-orbital (T)
    expression for a UHF reference of Watts, Gauss and Bartlett (1993),
    so an executed open-shell ``(T0)`` adds both sources and an executed
    rotated ``(T1)`` adds the 1993 expression.
    """
    if method != "dlpno-ccsd(t)" or options is None or result is None:
        return ()
    from .ccsd_local_solver import LocalCCSDOptions
    from .uccsd_local_solver import LocalUCCSDOptions

    if not bool(getattr(result, "triples_executed", False)):
        return ()
    if isinstance(options, LocalCCSDOptions):
        from . import triples_local

        mode, _ = triples_provenance(options, result)
        if mode == "local" and triples_local._HAVE_CPP_LOCAL_TRIPLES:
            return ("rendell_lee_komornicki_triples_1991",)
        return ()
    if not isinstance(options, LocalUCCSDOptions):
        return ()
    mode, _ = triples_provenance(options, result)
    if mode == "t1-iterative":
        return (
            "guo_dlpno_t1_2018",
            "guo_openshell_dlpno_triples_2020",
        )
    if mode == "t0":
        return (
            "guo_openshell_dlpno_triples_2020",
            "watts_gauss_bartlett_openshell_triples_1993",
        )
    if mode == "t1":
        return ("watts_gauss_bartlett_openshell_triples_1993",)
    return ()


__all__ = [
    "THRESHOLD_ORDER",
    "threshold_settings",
    "triples_citation_entries",
    "triples_provenance",
]
