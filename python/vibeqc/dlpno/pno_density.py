"""DLPNO pair-density conventions: what a PNO occupation number means.

Every DLPNO route builds the pair natural orbitals of pair ``ij`` by
diagonalising a pair density built from the semicanonical LMP2 amplitudes

    T^ij_uv = K^ij_uv / (F_ii + F_jj - eps_u - eps_v)

in the pair's non-redundant, Fock-diagonal PAO basis, and keeping the
eigenvectors whose eigenvalue (the *occupation number*) exceeds ``TCutPNO``.
``TCutPNO`` is therefore only as meaningful as the density it is compared
against, and the published Loose/Normal/Tight values are defined against one
specific density. This module holds the conventions so all three closed-shell
builders share one implementation and one disclosure.

Conventions
-----------
``"legacy"`` (vibe-qc's historical convention, the default before #65)
    ``D = (T T^T + T^T T) / (1 + delta_ij)``

    The density of the **bare** amplitudes. It is neither published
    convention below: both of those are built from the spin-adapted
    contravariant amplitude, and this one is not.

``"mp2"`` (Riplinger and Neese, *J. Chem. Phys.* **138**, 034106 (2013),
Eq. 23 and the line following it; ORCA's ``%mdci PNONorm MP2Norm`` default)

    ``Tt = (4 T - 2 T^T) / (1 + delta_ij)``
    ``D  = Tt T^T + Tt^T T``

    ``Tt`` is the spin-adapted contravariant amplitude that already carries
    the closed-shell pair energy: Eq. 14 of Neese, Wennmohs and Hansen,
    *J. Chem. Phys.* **130**, 114108 (2009) writes
    ``E_c = sum_{i<=j} sum_ab K^ij_ab Tt^ij_ab``, which is exactly what
    ``dlpno.mp2._pair_energy`` and the local solver's ``energy()`` evaluate as
    ``w * sum(T * (2 K - K^T))`` with ``w = 2 / (1 + delta_ij)``.

``"iepa"`` (Neese, Wennmohs and Hansen 2009, Eqs. 18-19; ORCA's
``PNONorm IEPANorm``, the pre-2013 LPNO convention)

    ``D = [(1 + delta_ij) / N_ij] (Tt T^T + Tt^T T)``,
    ``N_ij = 1 + <Tt^dagger T>`` with ``<A B> = sum_pq A_pq B_qp`` (their
    Eq. 13), i.e. ``N_ij = 1 + sum_ab Tt_ab T_ab``.

    A per-pair scalar on top of ``"mp2"``, nothing more. Riplinger and Neese
    moved to the MP2 norm "to keep future comparisons in the literature more
    transparent" after finding the convergence difference "insignificantly
    small (on average 1.4% per PNO included)" (their Fig. 2); the ORCA manual
    calls the two "near identical". Measured here ``N_ij`` runs 1.003-1.007,
    so ``"iepa"`` is ``"mp2"`` times ~1/1.005 off-diagonal and ~2/1.005 on the
    diagonal.

How much the choice moves the PNO count
---------------------------------------
Splitting ``T = S + A`` into symmetric and antisymmetric parts gives, exactly,

    ``D_mp2    = [4 S^2 + 12 A^T A] / (1 + delta_ij)``
    ``D_legacy = [2 S^2 +  2 A^T A] / (1 + delta_ij)``
    ``D_mp2    = 2 D_legacy + 8 A^T A / (1 + delta_ij)``

so every ``"mp2"`` occupation number is between **2x** (pure singlet-like
symmetric direction, and exactly 2x for any diagonal pair, whose ``T`` is
symmetric) and **6x** (pure triplet-like antisymmetric direction) the
``"legacy"`` one. Because the factor varies *within* one pair, no rescaling of
``TCutPNO`` reproduces the other convention's PNO set; only the density does.

Measured at NormalPNO from identical amplitudes in identical domains,
``"legacy"`` retains 4.6-18.5 % fewer PNOs than ``"mp2"``, growing with basis
size (13.5 % on H2O/cc-pVTZ, 18.5 % on the S22-02 water dimer at cc-pVDZ).
``TCutPNO`` under ``"legacy"`` is therefore effectively looser than the
published value it is named for.

**The default is** ``"mp2"``, by the ruling on GitLab issue **#65** (old
**#701**). The evidence behind that ruling is worth restating, because it is
not what it first looks like. At a *matched threshold* the published density
recovers more correlation energy -- but that only restates the count
difference. At *matched cost* the two are indistinguishable: over every
retained-PNO count reached by both conventions the ``mp2``/``legacy`` error
ratio has mean 1.005 and median 0.984, 68 % ties, the non-ties splitting both
ways. **The published density is not a better truncation criterion.** The
whole difference is a ``TCutPNO`` relabelling of 3.16x against a preset ladder
spacing of 3.330 -- one rung -- so ``legacy``@TightPNO and ``mp2``@NormalPNO
agree in both cost and accuracy. The defect was labelling: under ``"legacy"``
our NormalPNO delivered their LoosePNO. The default is ``"mp2"`` so that a
preset name means what the literature means by it, which is the entire purpose
of a named preset.

Closed-shell only
-----------------
The ``4 T - 2 T^T`` adaptation is a closed-shell spin-summation identity. The
open-shell routes work with spin-orbital amplitudes, where ORCA builds the
PNOs from ``T T^dagger`` and ``T^dagger T`` directly with no such factor (ORCA
manual, "Local MP2 calculations (DLPNO-MP2)", UHF subsection). They therefore
do not take a ``pno_norm`` and are untouched by this module.
"""

from __future__ import annotations

import numpy as np

#: Public spellings, in the order they are documented above.
PNO_NORMS = ("legacy", "mp2", "iepa")

# ORCA spells these MP2Norm / IEPANorm in %mdci PNONorm; accept both, plus the
# separator-insensitive forms, the way `thresholds.resolve_dlpno_thresholds`
# accepts "NormalPNO" / "normal".
_ALIASES = {
    "legacy": "legacy",
    "vibeqc": "legacy",
    "bare": "legacy",
    "mp2": "mp2",
    "mp2norm": "mp2",
    "iepa": "iepa",
    "iepanorm": "iepa",
}


def _normalise(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch not in " _-")


def resolve_pno_norm(spec: str | None) -> str:
    """Resolve a public spelling to one of :data:`PNO_NORMS`.

    ``None`` resolves to the project default, ``"legacy"``.
    """

    if spec is None:
        return "legacy"
    if not isinstance(spec, str):
        raise TypeError(
            f"pno_norm must be a string, got {type(spec).__name__}; "
            f"choose one of {', '.join(PNO_NORMS)}"
        )
    try:
        return _ALIASES[_normalise(spec)]
    except KeyError:
        raise ValueError(
            f"unknown pno_norm: {spec!r} (expected one of "
            f"{', '.join(repr(n) for n in PNO_NORMS)}; ORCA's MP2Norm and "
            "IEPANorm spellings are accepted for the latter two)"
        ) from None


def pair_density(
    T: np.ndarray,
    delta: float,
    norm: str | None = "legacy",
) -> np.ndarray:
    """Pair density whose eigenvalues are compared against ``tcut_pno``.

    Parameters
    ----------
    T : ndarray, shape (n, n)
        Semicanonical LMP2 amplitudes of the pair, in its PAO domain.
    delta : float
        ``1.0`` for a diagonal pair ``i == j``, ``0.0`` otherwise.
    norm : str, optional
        One of :data:`PNO_NORMS`; see the module docstring. Default
        ``"legacy"``, vibe-qc's historical density.

    Returns
    -------
    ndarray, shape (n, n)
        Symmetrised (numerical noise only) and positive semidefinite in every
        convention.
    """

    key = resolve_pno_norm(norm)
    scale = 1.0 + float(delta)
    if key == "legacy":
        D = (T @ T.T + T.T @ T) / scale
    else:
        # Riplinger and Neese 2013, the line under Eq. 23.
        T_tilde = (4.0 * T - 2.0 * T.T) / scale
        # Eq. 23 itself. The ORCA manual writes the transpose of this for
        # DLPNO-MP2; both are the same matrix because it is symmetric.
        D = T_tilde @ T.T + T_tilde.T @ T
        if key == "iepa":
            # LPNO 2009 Eqs. 18-19: <A B> = sum_pq A_pq B_qp (their Eq. 13),
            # so <Tt^dagger T> is the elementwise sum below.
            n_ij = 1.0 + float(np.sum(T_tilde * T))
            if n_ij != 0.0:
                D = D * (scale / n_ij)
    return 0.5 * (D + D.T)


__all__ = ["PNO_NORMS", "pair_density", "resolve_pno_norm"]
