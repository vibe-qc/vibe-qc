"""Keyword selector for Γ-CCM and the 2014 four-centre lineage.

Ruling R1 (2026-08-21, AICCM article repo ``HANDOVER.md`` §2/§2b and
``ROUTES_AND_VALIDATION.md`` §0) fixes the paper-facing nomenclature, and this
module follows it:

* **Paper-1 Γ-CCM denotes the NEUTRAL finite-BvK-torus construction.** Its two
  admissible producers are ``"real-gamma"`` -- the k-free real-Γ-supercell
  representation
  (:func:`~vibeqc.periodic.ccm.direct.run_ccm_rhf_direct` /
  :func:`~vibeqc.periodic.ccm.direct.run_ccm_rks_direct`, or their open-shell
  :func:`~vibeqc.periodic.ccm.direct.run_ccm_uhf_direct` /
  :func:`~vibeqc.periodic.ccm.direct.run_ccm_uks_direct` siblings): one real
  generalized eigenproblem over the ``N_c·n_μ`` supercell AOs, no k-mesh, with
  the derived exchange-``q=0`` seam (applied per spin for UHF/UKS and scaled
  by the exact-exchange fraction for hybrid KS) -- and ``"neutral-bloch"``,
  the multi-k GDF representation of the same construction
  (:func:`~vibeqc.periodic.ccm.ri.run_ccm_rhf_gdf` /
  :func:`~vibeqc.periodic.ccm.ri.run_ccm_rks_gdf`). ``"gdf"``,
  ``"gdf-control"``, ``"bloch-control"``, and ``"aiccm-ri"`` are aliases for
  the ``"neutral-bloch"`` producer.

* **``"four-center"`` is the 2014 union-and-weight/Wigner--Seitz lineage**
  (:func:`~vibeqc.periodic.ccm.scf.run_ccm_rhf` /
  :func:`~vibeqc.periodic.ccm.dft.run_ccm_rks`, ``method="aiccm2026dev-a"``).
  Paper 1 discusses this object only in the section that diagnoses it as
  non-variational; it is **not** what Paper 1 calls Γ-CCM. It stays
  selectable under its own name for lineage / diagnosis work, but the
  paper-facing spellings ``"gamma"`` / ``"gamma-ccm"`` / ``"gamma_ccm"`` fail
  closed in :func:`resolve_ccm_route`: a single paper-facing keyword cannot
  choose between the two admissible neutral producers, and it must never
  silently select the diagnosis-only four-centre route.

The two neutral producers are Fourier-related representations of one specified
block-circulant neutral Hamiltonian and agree to numerical implementation
tolerance at the shared ``exxdiv="ewald"`` convention; any RI fitting error is
common to both. That same-H representation theorem applies generally to
block-circulant torus Hamiltonians; it does not identify the neutral Γ-CCM
construction with the finite-translation-group χ-CCM construction.

Every route is **shell-aware** (keyed on the supercell multiplicity + electron
parity): open-shell clusters dispatch to ``run_ccm_uhf`` / ``run_ccm_uks``
(``"four-center"``), ``run_ccm_uhf_gdf`` / ``run_ccm_uks_gdf``
(``"neutral-bloch"``), or ``run_ccm_uhf_direct`` / ``run_ccm_uks_direct``
(``"real-gamma"``).

Chi (χ-CCM, ``aiccm2026dev-b``) is a *separate* development line reached through
``run_periodic_job(jk_method="aiccm2026dev-b")`` -- it is intentionally not a
route here.
"""

from __future__ import annotations

__all__ = ["run_ccm_scf", "resolve_ccm_route", "CCM_ROUTES"]

#: The two neutral Γ-CCM construction producers plus the 2014 four-centre
#: lineage.
CCM_ROUTES = ("four-center", "neutral-bloch", "real-gamma")

# Spoken / jk_method-style / harness aliases → canonical route name.
_ROUTE_ALIASES = {
    "four-center": "four-center",
    "four_center": "four-center",
    "fourcenter": "four-center",
    "4c": "four-center",
    "bare": "four-center",
    "aiccm-hf": "four-center",
    "aiccm2026dev-a": "four-center",
    "neutral-bloch": "neutral-bloch",
    "neutral_bloch": "neutral-bloch",
    "bloch-control": "neutral-bloch",
    "gdf-control": "neutral-bloch",
    "gdf_control": "neutral-bloch",
    "gdf": "neutral-bloch",
    "aiccm-ri": "neutral-bloch",
    "real-gamma": "real-gamma",
    "real_gamma": "real-gamma",
    "direct": "real-gamma",
    "direct-torus": "real-gamma",
    "non-k": "real-gamma",
    "aiccm-hf-direct": "real-gamma",
}

# Paper-facing Γ-CCM spellings, rejected per ruling R1 (2026-08-21): Paper-1
# Γ-CCM is the NEUTRAL construction, which has two admissible producers
# ("real-gamma" and "neutral-bloch"); a single keyword cannot pick one, and it
# must never silently select the diagnosis-only four-centre lineage.
_R1_PAPER_FACING_GAMMA_ALIASES = frozenset({"gamma", "gamma-ccm", "gamma_ccm"})

# Selector spellings that force method='aiccm2026dev-a' on the four-centre
# lineage branch (the paper-facing gamma spellings are rejected earlier, in
# resolve_ccm_route -- ruling R1).
_DEV_A_FOUR_CENTER_SELECTOR_ALIASES = frozenset(
    {"aiccm2026dev-a", "aiccm-hf"}
)
_REJECTED_A_PREFIXED_CONTROL_ALIASES = frozenset(
    {"aiccm2026dev-a-real-gamma", "aiccm2026dev-a-direct"}
)


def resolve_ccm_route(route: str) -> str:
    """Normalise a route keyword (with aliases) to one of :data:`CCM_ROUTES`.

    ``"aiccm2026dev-a"`` / ``"aiccm-hf"`` select the 2014 union-and-weight
    four-centre lineage; ``"neutral-bloch"`` / ``"gdf"`` / ``"aiccm-ri"`` and
    ``"real-gamma"`` / ``"aiccm-hf-direct"`` select the two producers of the
    neutral Γ-CCM construction. The paper-facing spellings ``"gamma"`` /
    ``"gamma-ccm"`` / ``"gamma_ccm"`` are rejected (ruling R1): Paper-1 Γ-CCM
    is the neutral construction, and a single keyword cannot choose between
    its two admissible producers. Raises :class:`ValueError` on an unknown or
    rejected keyword.
    """
    key = str(route).strip().lower()
    if key in _REJECTED_A_PREFIXED_CONTROL_ALIASES:
        raise ValueError(
            f"CCM route {route!r} is rejected because the "
            "'aiccm2026dev-a' prefix denotes the union-and-weight four-centre "
            "lineage (2014), while real-Gamma is a producer of the neutral "
            "Γ-CCM construction (ruling R1). Use route='aiccm2026dev-a' for "
            "the four-centre lineage or route='aiccm-hf-direct'/'real-gamma' "
            "for the neutral construction."
        )
    if key in _R1_PAPER_FACING_GAMMA_ALIASES:
        raise ValueError(
            f"CCM route {route!r} is the paper-facing Γ-CCM spelling, "
            "rejected per ruling R1 (2026-08-21): Paper-1 Γ-CCM denotes the "
            "NEUTRAL finite-BvK-torus construction, which has two admissible "
            "producers -- route='real-gamma' (run_ccm_rhf_direct) and "
            "route='neutral-bloch' (run_ccm_rhf_gdf). The union-and-weight "
            "four-centre is the 2014 lineage (diagnosed non-variational in "
            "Paper 1) and stays selectable as route='four-center'. Name one "
            "producer explicitly."
        )
    if key in _ROUTE_ALIASES:
        return _ROUTE_ALIASES[key]
    raise ValueError(
        f"unknown CCM route {route!r}; choose from {CCM_ROUTES} "
        f"(aliases: {sorted(set(_ROUTE_ALIASES) - set(CCM_ROUTES))})."
    )


def run_ccm_scf(ccm, *, route="four-center", functional=None,
                method=None, **kwargs):
    """Run the neutral Γ-CCM construction or the 2014 four-centre lineage by
    route keyword.

    Dispatches to the concrete driver for ``route`` (see the module docstring
    and :data:`CCM_ROUTES`); ``functional`` selects KS-DFT (``None`` → HF).
    Extra ``**kwargs`` forward to the underlying driver (they differ per route
    -- e.g. ``aux_basis`` for ``"neutral-bloch"``, ``cderi`` / ``cderi_build`` /
    ``exxdiv`` for ``"real-gamma"``; a wrong kwarg for the chosen route surfaces
    as a clear ``TypeError`` from the target).

    The default ``"four-center"`` route, as well as aliases
    ``"aiccm2026dev-a"`` and ``"aiccm-hf"``, selects the 2014 union-and-weight
    four-centre lineage; ``method=None`` selects ``"aiccm2026dev-a"`` on that
    branch. The paper-facing spellings ``"gamma"`` / ``"gamma-ccm"`` /
    ``"gamma_ccm"`` are rejected by :func:`resolve_ccm_route` (ruling R1):
    Paper-1 Γ-CCM is the NEUTRAL construction, whose two admissible producers
    -- ``"neutral-bloch"`` and ``"real-gamma"`` -- must be named explicitly. An
    explicit ``method`` is accepted only on the four-center branch, and the
    dev-a selector aliases constrain it to ``"aiccm2026dev-a"``. The
    ``"neutral-bloch"`` / ``"real-gamma"`` routes are the two producers of the
    neutral Γ-CCM construction and reject ``method`` rather than silently
    ignoring a construction selector. Agreement between those two routes is
    same-H representation evidence for the neutral Γ-CCM construction, not
    Γ-CCM / χ-CCM construction evidence.

    Every route dispatches **by shell**: a closed-shell cluster (even electron
    count AND supercell multiplicity 1) reaches the restricted driver, anything
    else the unrestricted sibling (``run_ccm_uhf`` / ``run_ccm_uks`` on
    ``"four-center"``, ``run_ccm_uhf_gdf`` / ``run_ccm_uks_gdf`` on
    ``"neutral-bloch"``, ``run_ccm_uhf_direct`` / ``run_ccm_uks_direct`` on
    ``"real-gamma"``).

    Returns the chosen driver's native result. The result types differ
    (``CCMGDFResult`` for ``"neutral-bloch"``; ``CCMSCFResult`` /
    ``CCMKSResult`` for HF / KS on closed-shell ``"four-center"`` and
    ``"real-gamma"``; open-shell HF on those two routes returns a
    :class:`~vibeqc.periodic.ccm.uhf.CCMUHFResult` -- per-spin fields, no
    ``.fock``/``.hcore`` -- and open-shell KS a ``CCMKSResult`` with
    ``open_shell=True``) but all expose ``.energy``, ``.energy_per_atom``,
    and ``.converged`` for comparison; the ``"real-gamma"`` results
    additionally record ``.exchange_q0``.
    """
    route_key = str(route).strip().lower()
    r = resolve_ccm_route(route)

    from .direct import (
        run_ccm_rhf_direct,
        run_ccm_rks_direct,
        run_ccm_uhf_direct,
        run_ccm_uks_direct,
    )
    from .dft import run_ccm_rks, run_ccm_uks
    from .ri import (
        run_ccm_rhf_gdf,
        run_ccm_rks_gdf,
        run_ccm_uhf_gdf,
        run_ccm_uks_gdf,
    )
    from .scf import run_ccm_rhf
    from .uhf import run_ccm_uhf

    # Shell-aware dispatch on every route (real-gamma 2026-07-16; four-center
    # + neutral-bloch 2026-07-15). Keyed on multiplicity AND electron parity.
    # Two silent traps closed together: an explicit high-spin unit cell
    # (multiplicity >= 3) used to be DISCARDED by
    # CCMSystem._build_supercell's parity default (fixed there -- FM
    # replication), and an even-electron open-shell supercell would then
    # have fallen through to the closed-shell loops (n_occ = n_elec // 2
    # ignores multiplicity), returning a wrong closed-shell number; an
    # odd-electron cluster raised. Open shells now reach the UHF/UKS
    # siblings. The neutral-bloch multi-k drivers read the unit cell's charge
    # and multiplicity; their multi-cell spin-bookkeeping limitation is
    # documented on run_ccm_uhf_gdf/run_ccm_uks_gdf.
    n_elec = int(ccm.supercell.n_electrons())
    closed = (n_elec % 2 == 0) and int(ccm.supercell.multiplicity) == 1
    if r == "four-center":
        selected_method = "aiccm2026dev-a" if method is None else method
        if (
            route_key in _DEV_A_FOUR_CENTER_SELECTOR_ALIASES
            and str(selected_method).strip().lower() != "aiccm2026dev-a"
        ):
            raise ValueError(
                f"route={route!r} identifies the aiccm2026dev-a four-centre "
                "lineage (2014) and therefore requires "
                "method='aiccm2026dev-a'; "
                f"got method={method!r}. Use route='four-center' to select a "
                "different four-center weighting explicitly."
            )
        if route_key in _DEV_A_FOUR_CENTER_SELECTOR_ALIASES:
            selected_method = "aiccm2026dev-a"
        if functional is not None:
            if closed:
                return run_ccm_rks(
                    ccm, functional, method=selected_method, **kwargs
                )
            return run_ccm_uks(
                ccm, functional, method=selected_method, **kwargs
            )
        if closed:
            return run_ccm_rhf(ccm, method=selected_method, **kwargs)
        return run_ccm_uhf(ccm, method=selected_method, **kwargs)
    if method is not None:
        raise ValueError(
            f"method={method!r} applies only to route='four-center'; "
            f"route={route!r} selects the {r!r} producer of the neutral "
            "Γ-CCM construction. Omit method for construction routes."
        )
    if r == "neutral-bloch":
        if functional is not None:
            if closed:
                return run_ccm_rks_gdf(ccm, functional, **kwargs)
            return run_ccm_uks_gdf(ccm, functional, **kwargs)
        if closed:
            return run_ccm_rhf_gdf(ccm, **kwargs)
        return run_ccm_uhf_gdf(ccm, **kwargs)
    # real-gamma
    if functional is not None:
        if closed:
            return run_ccm_rks_direct(ccm, functional, **kwargs)
        return run_ccm_uks_direct(ccm, functional, **kwargs)
    if closed:
        return run_ccm_rhf_direct(ccm, **kwargs)
    return run_ccm_uhf_direct(ccm, **kwargs)
