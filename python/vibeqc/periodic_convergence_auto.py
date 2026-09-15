"""Automatic convergence-strategy selection for periodic SCF.

Classifies a :class:`PeriodicSystem` from cheap pre-SCF signals
(dimensionality, cell volume per atom, vacuum axes, Pauling
electronegativity spread, metallic composition, electron-count parity)
into a coarse profile, and resolves a full convergence strategy
(Fermi-Dirac smearing temperature, CRYSTAL-style FMIXING, level shift,
density damping) for that profile.

Design rules (the transparency contract):

* **Explicit user knobs always win.** A knob the user set is never
  overridden; its resolution is labelled ``source="explicit"``.
* **Auto fills only unset knobs**, each with a human-readable reason;
  the assembled strategy carries ``mode`` =
  ``"auto-default"`` (user gave nothing), ``"auto-requested"``
  (user passed ``convergence="auto"``), ``"manual"`` (user knobs only,
  no auto fill), or ``"off"`` (user passed ``convergence="off"``).
* **Conservative bias**: when classification is uncertain the resolver
  returns today's plain defaults (DIIS only) rather than changing the
  physics of an unknown system -- the same policy as
  :func:`vibeqc.smearing.auto.guess_smearing_temperature`, which this
  module composes for the gap/metallic smearing decision.

The numbers are grounded in measurements and existing practice:

* ionic-insulator -> FMIXING 30 % with integer occupations.  Smearing is
  deliberately excluded: on MgO it can metallize the insulating state and
  converge to a wrong-energy basin.  FMIXING 30 is CRYSTAL's long-standing
  recommended default for
  difficult ionic systems (Dovesi et al., Int. J. Quantum Chem. 114,
  1287 (2014) -- the FMIXING keyword).
* molecular-limit / covalent cells converge with plain DIIS in a few
  iterations (H₂-box suites; diamond/Si validation history) -- aids
  only slow them down, so the resolver leaves them untouched.
* metallic composition -> smearing 0.005 Ha (the ``"metal"`` preset
  shared with the smearing resolver) + FMIXING 50 %.

The tight-cell signal (``dim==3``, any Z>1, volume < 500 bohr^3)
matches the heuristic already used by the GDF route
(``pbc_gdf._warn_gamma_compcell_ionic`` /
``periodic_rijcosx._is_tight_cell``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

__all__ = [
    "SystemClassification",
    "KnobResolution",
    "ConvergenceStrategy",
    "classify_periodic_system",
    "resolve_convergence_strategy",
    "insulator_smearing_warning",
    "CONVERGENCE_KNOBS",
]

# The knobs the auto strategy manages. Anything else (DIIS subspace,
# conv tolerances, max_iter, accelerator family) keeps its existing
# default and is out of scope for v1.
CONVERGENCE_KNOBS = (
    "smearing_temperature",
    "fock_mixing",
    "level_shift",
    "damping",
)

# Pauling electronegativities (standard scale, dimensionless) for the
# elements that appear in periodic calculations in practice. Textbook
# reference-table constants (shared background data; see e.g. any CRC
# Handbook edition). Missing entries simply mute the spread signal.
_PAULING_EN: Dict[int, float] = {
    1: 2.20, 2: 0.0,
    3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98,
    10: 0.0,
    11: 0.93, 12: 1.31, 13: 1.61, 14: 1.90, 15: 2.19, 16: 2.58,
    17: 3.16, 18: 0.0,
    19: 0.82, 20: 1.00, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66,
    25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.90, 30: 1.65,
    31: 1.81, 32: 2.01, 33: 2.18, 34: 2.55, 35: 2.96, 36: 3.00,
    37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33, 41: 1.60, 42: 2.16,
    43: 1.90, 44: 2.20, 45: 2.28, 46: 2.20, 47: 1.93, 48: 1.69,
    49: 1.78, 50: 1.96, 51: 2.05, 52: 2.10, 53: 2.66, 54: 2.60,
    55: 0.79, 56: 0.89, 57: 1.10, 58: 1.12, 72: 1.30, 73: 1.50,
    74: 2.36, 75: 1.90, 76: 2.20, 77: 2.20, 78: 2.28, 79: 2.54,
    80: 2.00, 81: 1.62, 82: 2.33, 83: 2.02,
}

# Elements that are metals in the elemental-solid sense (alkali,
# alkaline earth, transition, post-transition metals + lanthanides).
# Used for the all-metallic-composition signal only.
_METALLIC_Z = (
    set(range(3, 5))        # Li, Be
    | {11, 12, 13}          # Na, Mg, Al
    | set(range(19, 32))    # K..Ga
    | set(range(37, 51))    # Rb..Sn
    | set(range(55, 84))    # Cs..Bi (incl. lanthanides)
)

# dim==3 cells with any lattice vector longer than this are treated as
# vacuum-padded molecular-limit boxes (the H₂-in-a-box pattern).
_VACUUM_AXIS_BOHR = 20.0

# The GDF/RIJCOSX tight-cell heuristic: 3D, beyond-H composition,
# volume below this -> condensed ionic/covalent crystal.
_TIGHT_CELL_VOLUME_BOHR3 = 500.0

# Electronegativity spread at or above this -> ionic bonding character.
_IONIC_EN_SPREAD = 1.4


@dataclass(frozen=True)
class SystemClassification:
    """Coarse pre-SCF profile of a periodic system."""

    profile: str  # molecular-limit | covalent-insulator | ionic-insulator
    #               | metallic-candidate | unknown
    reasons: List[str] = field(default_factory=list)
    open_shell: bool = False
    dim: int = 3
    volume_per_atom_bohr3: Optional[float] = None
    en_spread: Optional[float] = None


@dataclass(frozen=True)
class KnobResolution:
    """One resolved convergence knob with its provenance."""

    value: float
    source: str  # "explicit" | "auto" | "default"
    reason: str


@dataclass(frozen=True)
class ConvergenceStrategy:
    """The full resolved strategy, ready for the SCF options object."""

    mode: str  # "auto-default" | "auto-requested" | "manual" | "off"
    classification: Optional[SystemClassification]
    knobs: Dict[str, KnobResolution]

    def value(self, knob: str) -> float:
        return self.knobs[knob].value

    def log_lines(self) -> List[str]:
        """Human-readable block for the SCF log / .out file."""
        mode_text = {
            "auto-default": (
                "AUTO (default: no convergence options given; pass "
                'convergence="off" or any explicit knob to override)'
            ),
            "auto-requested": 'AUTO (requested via convergence="auto")',
            "manual": "manual (explicit user options; auto selection off)",
            "off": 'off (convergence="off": plain defaults)',
        }[self.mode]
        lines = [f"convergence strategy: {mode_text}"]
        if self.classification is not None and self.mode.startswith("auto"):
            cls = self.classification
            lines.append(f"  profile: {cls.profile}")
            for r in cls.reasons:
                lines.append(f"    - {r}")
        for name in CONVERGENCE_KNOBS:
            res = self.knobs[name]
            if res.source == "default" and res.value == 0.0:
                continue  # don't spam zero defaults
            lines.append(
                f"  {name} = {res.value:g}   [{res.source}] {res.reason}"
            )
        return lines


def _lattice_lengths(system) -> np.ndarray:
    lattice = np.asarray(system.lattice, dtype=float)
    # `system.lattice` columns are the Cartesian lattice vectors
    # (cpp/include/vibeqc/periodic.hpp:32, "Columns = Cartesian lattice
    # vectors"; periodic_runner._reduce_system_to_primitive uses
    # frac = inv(L) @ r_cart), so axis lengths are the column norms. Row and
    # column norms coincide only for a symmetric (cubic/orthorhombic) matrix;
    # on a skewed slab/rod or triclinic cell axis=1 gives wrong lengths.
    return np.linalg.norm(lattice, axis=0)


def classify_periodic_system(system) -> SystemClassification:
    """Classify a PeriodicSystem from cheap pre-SCF signals only.

    No integrals, no SCF -- composition + geometry + electron parity.
    """
    reasons: List[str] = []
    atoms = list(system.unit_cell)
    zs = [int(a.Z) for a in atoms]
    dim = int(system.dim)

    n_electrons = float(system.n_electrons())
    multiplicity = int(getattr(system, "multiplicity", 1) or 1)
    odd_electrons = int(round(n_electrons)) % 2 == 1
    open_shell = multiplicity > 1 or odd_electrons
    if open_shell:
        reasons.append(
            f"open-shell signals (multiplicity={multiplicity}, "
            f"{int(round(n_electrons))} electrons/cell)"
        )

    volume = float(abs(np.linalg.det(np.asarray(system.lattice, dtype=float))))
    v_per_atom = volume / max(len(atoms), 1)

    lengths = _lattice_lengths(system)
    has_vacuum_axis = dim == 3 and bool(np.any(lengths > _VACUUM_AXIS_BOHR))

    en_values = [_PAULING_EN.get(z) for z in zs]
    en_known = [e for e in en_values if e]
    en_spread = (max(en_known) - min(en_known)) if len(en_known) >= 2 else None

    all_metallic = bool(zs) and all(z in _METALLIC_Z for z in zs)
    heavy = any(z > 1 for z in zs)
    tight = dim == 3 and heavy and volume < _TIGHT_CELL_VOLUME_BOHR3

    if has_vacuum_axis or (dim == 3 and v_per_atom > 800.0):
        reasons.append(
            f"vacuum-padded cell (max lattice vector "
            f"{lengths.max():.1f} bohr, {v_per_atom:.0f} bohr^3/atom) -- "
            "molecular-limit box"
        )
        profile = "molecular-limit"
    elif all_metallic and tight:
        reasons.append(
            "all-metal composition in a tight cell "
            f"({volume:.0f} bohr^3 < {_TIGHT_CELL_VOLUME_BOHR3:.0f}) -- "
            "likely metallic"
        )
        profile = "metallic-candidate"
    elif tight and en_spread is not None and en_spread >= _IONIC_EN_SPREAD:
        reasons.append(
            f"tight 3D cell ({volume:.0f} bohr^3, "
            f"{v_per_atom:.0f} bohr^3/atom) with large Pauling "
            f"electronegativity spread Δchi = {en_spread:.2f} -- ionic "
            "charge-transfer crystal"
        )
        profile = "ionic-insulator"
    elif tight:
        reasons.append(
            f"tight 3D cell ({volume:.0f} bohr^3) with small "
            f"electronegativity spread (Δchi = "
            f"{en_spread if en_spread is not None else 0.0:.2f}) -- "
            "covalent crystal"
        )
        profile = "covalent-insulator"
    elif dim != 3:
        # Be honest about WHY there is no profile. The covalent / ionic /
        # metallic branches above are all gated on ``tight``, which is
        # ``dim == 3`` by construction, so no dim=1 or dim=2 cell can ever
        # reach them. Reporting that as "no decisive signal" reads as
        # "looked and found nothing" when the truth is "this classifier has
        # no low-dimensional branch at all" -- and the difference matters,
        # because a 2D semimetal (graphene) cannot converge on the integer
        # occupations this unknown profile then selects. Naming the
        # limitation is the CLAUDE.md section 7 discipline applied to the
        # profiler itself. The knobs are deliberately unchanged: finite-T
        # smearing is not yet implemented on the slab GDF route that 2D
        # systems take, so a smeared default here would only fail closed.
        reasons.append(
            f"no low-dimensional branch (dim={dim}, "
            f"{v_per_atom:.0f} bohr^3/atom) -- the covalent/ionic/metallic "
            "signals are 3D-only, so this cell is unclassified by "
            "construction, not judged inconclusive; a semimetallic or "
            "small-gap sheet needs an explicit smearing choice"
        )
        profile = "unknown"
    else:
        reasons.append(
            f"no decisive signal (dim={dim}, {v_per_atom:.0f} bohr^3/atom)"
            " -- keeping conservative plain defaults"
        )
        profile = "unknown"

    return SystemClassification(
        profile=profile,
        reasons=reasons,
        open_shell=open_shell,
        dim=dim,
        volume_per_atom_bohr3=v_per_atom,
        en_spread=en_spread,
    )



# The BIPOLE KS drivers apply a built-in FMIXING 30% default whenever
# the functional is a DFT functional, fock_mixing is left at 0 AND
# DIIS is off (since the 2026-07-13 Gap-B validation: under DIIS the
# mixing is redundant damping -- see pbc_bipole_rks.py). The printed
# strategy must match what the driver actually does, so the KS auto
# table never reports less than that floor; the caller passes the
# floor flag only for BIPOLE KS runs with DIIS disabled.
_KS_DRIVER_FOCK_MIXING_FLOOR = 0.30


def _apply_ks_fock_mixing_floor(
    out: Dict[str, KnobResolution],
    *,
    is_ks: bool,
) -> Dict[str, KnobResolution]:
    if is_ks and out["fock_mixing"].value < _KS_DRIVER_FOCK_MIXING_FLOOR:
        out = dict(out)
        out["fock_mixing"] = KnobResolution(
            _KS_DRIVER_FOCK_MIXING_FLOOR,
            "auto",
            "BIPOLE KS drivers apply FMIXING 30% for DFT functionals by "
            "default; reported here so the log matches the driver",
        )
    return out


def _auto_knobs_for_profile(
    cls: SystemClassification,
    *,
    is_ks: bool,
    ks_floor: bool = True,
) -> Dict[str, KnobResolution]:
    """The strategy table: profile -> knob values with reasons."""
    zero = {
        "smearing_temperature": KnobResolution(
            0.0, "auto", "no smearing for this profile"
        ),
        "fock_mixing": KnobResolution(0.0, "auto", "plain DIIS suffices"),
        "level_shift": KnobResolution(0.0, "auto", "not needed"),
        "damping": KnobResolution(0.0, "auto", "not needed"),
    }
    if cls.profile in ("molecular-limit", "covalent-insulator"):
        note = (
            "molecular-limit boxes and covalent crystals converge in a "
            "few plain-DIIS iterations; aids would only slow them down"
        )
        out = dict(zero)
        out["fock_mixing"] = KnobResolution(0.0, "auto", note)
        return _apply_ks_fock_mixing_floor(out, is_ks=is_ks and ks_floor)
    if cls.profile == "ionic-insulator":
        out = dict(zero)
        out["fock_mixing"] = KnobResolution(
            0.30,
            "auto",
            "ionic-insulator profile: CRYSTAL-style FMIXING 30% "
            "(MgO-class cells measured at ~20 iterations with this aid)",
        )
        out["smearing_temperature"] = KnobResolution(
            0.0,
            "auto",
            "ionic insulators require integer occupations; finite-temperature "
            "smearing can metallize the gap and converge a wrong-energy basin; "
            "FMIXING carries the stabilisation",
        )
        return _apply_ks_fock_mixing_floor(out, is_ks=is_ks and ks_floor)
    if cls.profile == "metallic-candidate":
        out = dict(zero)
        out["fock_mixing"] = KnobResolution(
            0.50, "auto", "metallic candidate: heavy FMIXING (CRYSTAL "
            "practice for conductors)"
        )
        if is_ks:
            out["smearing_temperature"] = KnobResolution(
                0.005,
                "auto",
                'metallic composition: the shared "metal" smearing preset',
            )
        else:
            out["level_shift"] = KnobResolution(
                0.2,
                "auto",
                "HF on a metallic candidate: level shift guards against "
                "occupation flipping (HF cannot smear)",
            )
        return _apply_ks_fock_mixing_floor(out, is_ks=is_ks and ks_floor)
    # unknown
    out = dict(zero)
    for name in CONVERGENCE_KNOBS:
        out[name] = KnobResolution(
            out[name].value, "auto", "conservative default (unknown profile)"
        )
    return _apply_ks_fock_mixing_floor(out, is_ks=is_ks and ks_floor)


def resolve_convergence_strategy(
    system,
    *,
    method: str,
    convergence: Optional[str],
    explicit: Optional[Dict[str, Optional[float]]] = None,
    ks_driver_fock_mixing_floor: bool = False,
) -> ConvergenceStrategy:
    """Resolve the full convergence strategy for one periodic SCF.

    Parameters
    ----------
    system
        The PeriodicSystem.
    method
        "RHF" | "UHF" | "RKS" | "UKS" -- HF methods never receive
        smearing (their drivers reject finite temperature).
    convergence
        ``None`` -> auto (mode "auto-default") unless any explicit knob
        is given, in which case "manual";
        ``"auto"`` -> auto fill of the *unset* knobs (mode
        "auto-requested"; explicit knobs still win per-knob);
        ``"off"``/``"none"`` -> plain defaults (mode "off").
    explicit
        Mapping knob-name -> value for knobs the user explicitly set,
        or ``None``/missing for unset. Unknown keys are rejected.
    ks_driver_fock_mixing_floor
        Whether the route's KS drivers apply a built-in FMIXING 30%
        default for DFT functionals in this run (the BIPOLE drivers do
        when DIIS is off -- the reported strategy then floors
        fock_mixing so the log matches the driver). Since the
        2026-07-13 Gap-B validation the BIPOLE auto default does NOT
        fire under DIIS (the common case), so the flag defaults to
        ``False``; callers pass ``True`` only for BIPOLE KS runs with
        DIIS disabled. Always ``False`` for routes without in-driver
        FMIXING behaviour (e.g. GDF).
    """
    explicit = dict(explicit or {})
    for key in explicit:
        if key not in CONVERGENCE_KNOBS:
            raise ValueError(
                f"resolve_convergence_strategy: unknown knob {key!r} "
                f"(managed knobs: {', '.join(CONVERGENCE_KNOBS)})"
            )
    explicit_set = {k: v for k, v in explicit.items() if v is not None}
    is_ks = method.upper() in ("RKS", "UKS")

    requested = (convergence or "").strip().lower() or None
    if requested not in (None, "auto", "off", "none"):
        raise ValueError(
            f'convergence must be "auto", "off"/"none" or omitted; '
            f"got {convergence!r}"
        )

    if requested in ("off", "none"):
        mode = "off"
        do_auto = False
    elif requested == "auto":
        mode = "auto-requested"
        do_auto = True
    elif explicit_set:
        mode = "manual"
        do_auto = False
    else:
        mode = "auto-default"
        do_auto = True

    classification = classify_periodic_system(system) if do_auto else None

    knobs: Dict[str, KnobResolution] = {}
    auto_table = (
        _auto_knobs_for_profile(
            classification,
            is_ks=is_ks,
            ks_floor=ks_driver_fock_mixing_floor,
        )
        if do_auto
        else {}
    )
    for name in CONVERGENCE_KNOBS:
        if name in explicit_set:
            knobs[name] = KnobResolution(
                float(explicit_set[name]), "explicit", "set by the user"
            )
        elif do_auto:
            knobs[name] = auto_table[name]
        else:
            knobs[name] = KnobResolution(
                0.0, "default", "plain default (auto selection off)"
            )

    return ConvergenceStrategy(
        mode=mode,
        classification=classification,
        knobs=knobs,
    )


def insulator_smearing_warning(
    system,
    temperature_hartree: float,
    *,
    band_gap_hartree: Optional[float] = None,
    metallic: Optional[bool] = None,
) -> Optional[str]:
    """Warn before applying smearing to a detectably insulating system.

    An explicit positive gap is the strongest signal.  Otherwise the same
    cheap pre-SCF classifier used by the automatic convergence strategy
    identifies ionic/covalent insulators.  ``metallic=True`` suppresses only
    that heuristic branch; it cannot override an explicitly supplied gap.
    """
    temperature = float(temperature_hartree)
    if temperature <= 0.0:
        return None

    gap = None if band_gap_hartree is None else float(band_gap_hartree)
    if gap is not None and gap > _CONDUCTING_GAP_HA:
        signal = f"a supplied band gap of {gap:.6g} Ha"
    elif metallic is True:
        return None
    else:
        profile = classify_periodic_system(system).profile
        if profile not in ("ionic-insulator", "covalent-insulator"):
            return None
        signal = f"the pre-SCF {profile} profile"

    return (
        f"Finite-temperature smearing (k_B T = {temperature:.6g} Ha) was "
        f"requested despite {signal}. Smearing can metallize a gapped system "
        "and converge a wrong-energy basin; use integer occupations unless a "
        "trusted conducting-state reference justifies smearing."
    )


# ---------------------------------------------------------------------------
# Strategy v2: post-SCF re-classification check
# ---------------------------------------------------------------------------
# The pre-SCF classification is a guess; the converged spectrum is the
# truth. Generalizes the GDF route's conducting-state warning: when an
# auto-mode run converges with a spectrum that contradicts the assumed
# profile, say so loudly instead of leaving a silently sub-optimal (or
# physically suspect) aid configuration.

# Gap below this -> effectively conducting (matches the GDF route's
# POSSIBLY CONDUCTING STATE threshold, ~27 meV).
_CONDUCTING_GAP_HA = 1e-3
# Gap above this -> comfortably insulating (the smearing auto-resolver's
# "comfortably insulating" threshold is 0.05; be a bit stricter here).
_INSULATING_GAP_HA = 0.1


def converged_gap_hartree(
    result,
    *,
    n_alpha: int,
    n_beta: Optional[int] = None,
) -> Optional[float]:
    """Indirect HOMO-LUMO gap across the k-mesh from a converged result.

    Works for closed-shell results (``mo_energies``) and open-shell
    results (``mo_energies_alpha`` / ``mo_energies_beta`` -- the gap is
    the minimum over both spin channels). Returns ``None`` when the
    eigenvalues or band indices are not derivable (empty/full bands).
    """

    def _channel_gap(eps_per_k, n_occ: int) -> Optional[float]:
        if eps_per_k is None or n_occ <= 0:
            return None
        if len(eps_per_k) == 0:
            return None
        # Γ-only results may carry a single flat eigenvalue array
        # instead of a per-k list -- normalise to one "k-point".
        first = np.asarray(eps_per_k[0])
        if first.ndim == 0:
            eps_per_k = [np.asarray(eps_per_k)]
        vbm = -np.inf
        cbm = np.inf
        for eps in eps_per_k:
            eps = np.real(np.asarray(eps))
            if eps.ndim == 1:
                eps = np.sort(eps)
            if eps.ndim == 0 or n_occ > eps.shape[0] - 1:
                return None  # no virtuals in this channel

            # Some periodic solvers preserve an exactly degenerate band as
            # one row with a trailing manifold axis.  The band axis is still
            # ordered, but indexing it produces an array rather than a scalar.
            # Use the physical edges of the two adjacent manifolds: the
            # highest member below the occupation boundary and the lowest
            # member above it.
            valence = np.asarray(eps[n_occ - 1], dtype=float).reshape(-1)
            conduction = np.asarray(eps[n_occ], dtype=float).reshape(-1)
            if valence.size == 0 or conduction.size == 0:
                return None
            if not (
                np.all(np.isfinite(valence))
                and np.all(np.isfinite(conduction))
            ):
                return None
            vbm = max(vbm, float(np.max(valence)))
            cbm = min(cbm, float(np.min(conduction)))
        if not (np.isfinite(vbm) and np.isfinite(cbm)):
            return None
        return cbm - vbm

    eps_closed = getattr(result, "mo_energies", None)
    eps_alpha = getattr(result, "mo_energies_alpha", None)
    if eps_alpha is not None:
        gaps = []
        g_a = _channel_gap(eps_alpha, int(n_alpha))
        if g_a is not None:
            gaps.append(g_a)
        eps_beta = getattr(result, "mo_energies_beta", None)
        if eps_beta is not None and n_beta:
            g_b = _channel_gap(eps_beta, int(n_beta))
            if g_b is not None:
                gaps.append(g_b)
        return min(gaps) if gaps else None
    if eps_closed is not None:
        return _channel_gap(eps_closed, int(n_alpha))
    return None


def post_scf_profile_check(
    strategy: ConvergenceStrategy,
    gap_ha: Optional[float],
) -> Optional[Tuple[str, str]]:
    """Compare the converged gap against the auto-assumed profile.

    Returns ``None`` when there is nothing to say, otherwise a
    ``(level, message)`` pair with level ``"warning"`` (assumed an
    insulator, converged conducting -- the aid configuration may be
    inadequate and the occupations physically suspect) or ``"note"``
    (assumed metallic, converged insulating -- harmless, smearing
    entropy is ~0). Only fires on auto-mode strategies: a manual user
    choice is the user's statement of intent.
    """
    if gap_ha is None or strategy.classification is None:
        return None
    if not strategy.mode.startswith("auto"):
        return None
    profile = strategy.classification.profile
    if profile != "metallic-candidate" and gap_ha < _CONDUCTING_GAP_HA:
        return (
            "warning",
            f"converged HOMO-LUMO gap {gap_ha:.2e} Ha is effectively zero "
            f"but the auto profile assumed '{profile}' -- possibly a "
            "conducting state with inadequate aids (integer occupations / "
            "no or gentle smearing). Re-run with "
            'convergence="off" plus explicit smearing_temperature= (e.g. '
            "0.005-0.02 Ha) or smearing_metallic=True.",
        )
    if profile == "metallic-candidate" and gap_ha > _INSULATING_GAP_HA:
        return (
            "note",
            f"converged HOMO-LUMO gap {gap_ha:.3f} Ha is comfortably "
            "insulating although the composition suggested a metal; the "
            "applied smearing carries ~zero entropy here and the result "
            "is unaffected. Pass convergence=\"off\" to silence the aids.",
        )
    return None
