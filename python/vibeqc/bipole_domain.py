"""Immutable records of a resolved ab initio BIPOLE finite operator.

This module performs no integrals and imports no native core. It records the
existing kernels' settings; it is not an invariance or convergence certificate.
The four ab initio BIPOLE drivers select physical support by default.
``historical_index_ball`` explicitly names the retained reproduction policy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math


# Match the default Ewald radial tolerance across Coulomb channels.
# This controls the smeared erfc factor, not the total lattice energy error.
DEFAULT_BIPOLE_SR_IMAGE_PRECISION = 1e-8


LATTICE_FIELDS = (
    "cutoff_bohr", "nuclear_cutoff_bohr", "eri_interaction_cutoff_bohr",
    "becke_image_radius_bohr", "slab_ewald_alpha",
    "screening_overlap_threshold", "screening_exchange_threshold",
    "schwarz_threshold", "schwarz_threshold_forces", "sr_range_screening",
    "sr_sparse_traversal", "pair_complete_1e", "gamma_only_0", "coulomb_method",
)
GRID_FIELDS = (
    "atomic_grid_profile", "n_radial", "angular", "lebedev_order",
    "n_theta", "n_phi", "partition", "becke_k", "angular_pruning",
    "orca_angular_points", "vv10_grid_factor",
)


def resolve_bipole_pair_policy(historical_domain=False, *, active=True):
    """Select the BIPOLE policy without changing shared native defaults."""
    if not isinstance(historical_domain, bool):
        raise TypeError("BIPOLE historical_domain must be True or False")
    if historical_domain and not active:
        raise ValueError("bipole_historical_domain requires ab initio BIPOLE RHF/UHF/RKS/UKS")
    return not historical_domain


def resolve_bipole_xc_density_reach(lattice_options):
    """Resolve the native density-image radius, independently of grid weights.

    Nonpositive values retain the native unset convention; positive requests
    are capped by the total AO lattice support. No caller option is mutated.
    """
    cutoff = float(lattice_options.cutoff_bohr)
    requested = float(lattice_options.becke_image_radius_bohr)
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("BIPOLE XC cutoff_bohr must be finite and > 0")
    if not math.isfinite(requested):
        raise ValueError("BIPOLE XC density image radius must be finite")
    # Becke, JCP 88, 2547 (1988), Eqs. (3)-(7), DOI: 10.1063/1.454033:
    # partition unity weights the integrand; it does not set its image support.
    return min(requested, cutoff) if requested > 0.0 else cutoff


def _value(value):
    # pybind enums are immutable but not portable JSON values. Preserve both
    # enum type and member so restoration never guesses from an integer.
    if isinstance(value, (list, tuple)):
        return ("sequence", tuple(_value(item) for item in value))
    if hasattr(value, "name"):
        return ("enum", type(value).__name__, value.name)
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError("BIPOLE domain fields must be finite scalars or named enums")


@dataclass(frozen=True)
class OptionSnapshot:
    values: tuple

    @classmethod
    def capture(cls, options, fields):
        return cls(tuple((name, _value(getattr(options, name))) for name in fields))

    def restore(self, factory):
        options = factory()
        for name, value in self.values:
            if isinstance(value, tuple) and value[0] == "sequence":
                value = list(value[1])
            elif isinstance(value, tuple):
                current = getattr(options, name)
                if type(current).__name__ != value[1]:
                    raise ValueError(f"BIPOLE domain enum type changed for {name}")
                value = getattr(type(current), value[2])
            setattr(options, name, value)
        return options

    def require_match(self, options, *, context, ignore=()):
        for name, expected in self.values:
            if name not in ignore and _value(getattr(options, name)) != expected:
                raise ValueError(f"{context}: {name} differs from the resolved BIPOLE domain")


@dataclass(frozen=True)
class BipoleResolvedDomain:
    """Snapshot shared by an SCF result and its exact/incremental context.

    Separate energy/force screening values are intentional. SAP's all-space
    quadrature is not the per-cell XC grid recorded here. The digest identifies
    settings, not the geometry, basis, compiled kernel or retained tensor.
    """
    lattice_1e: OptionSnapshot
    lattice_2e: OptionSnapshot
    ewald_alpha_bohr_inv: float | None
    reciprocal_cutoff_bohr_inv: float | None
    ewald_real_cutoff_bohr: float | None
    ewald_precision: float
    sr_image_extent_bohr: float | None
    use_ewald_j_split: bool
    exchange_ewald_split: bool
    exchange_exxdiv: str | None
    exact_zone_bohr: float | None
    xc_grid: OptionSnapshot | None
    use_periodic_becke: bool
    becke_image_radius_bohr: float | None
    nuclear_erfc: OptionSnapshot | None = None
    v_ne_grid: OptionSnapshot | None = None
    xc_density_domain: object = None
    exact_j_ke_cutoff: float | None = None
    requested_lattice: OptionSnapshot | None = None
    # ``BipoleSRTruncationEstimate.as_record()`` pairs (#531), or None when the
    # route built no padded erfc short-range term.
    sr_truncation_estimate: tuple | None = None
    schema: str = "vibeqc.bipole.resolved-domain.v1"

    def __post_init__(self):
        one, two = dict(self.lattice_1e.values), dict(self.lattice_2e.values)
        for name in ("pair_complete_1e", "cutoff_bohr"):
            if one[name] != two[name]:
                raise ValueError(f"BIPOLE 1e/2e domain mismatch: {name}")
        # Validate all recorded scalar settings without running a kernel.
        json.dumps(self.as_dict(), allow_nan=False)

    @property
    def policy(self):
        return ("physical_pairs" if dict(self.lattice_2e.values)["pair_complete_1e"]
                else "historical_index_ball")

    def as_dict(self):
        return dict(asdict(self), policy=self.policy)

    @property
    def settings_sha256(self):
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode()
        return hashlib.sha256(encoded).hexdigest()

    def require_ewald_options(self, options, *, context):
        """Require the complete nuclear Ewald state, including the real reach."""
        if options is None:
            if self.ewald_alpha_bohr_inv is not None:
                raise ValueError(f"{context}: missing resolved Ewald options")
            return
        for field, actual in (
            ("ewald_alpha_bohr_inv", options.alpha),
            ("ewald_real_cutoff_bohr", options.real_cutoff_bohr),
            ("reciprocal_cutoff_bohr_inv", options.recip_cutoff_bohr_inv),
            ("ewald_precision", options.tolerance),
        ):
            if actual != getattr(self, field):
                raise ValueError(f"{context}: {field} differs from the resolved BIPOLE domain")

    def require_fock_context(self, ctx):
        self.lattice_2e.require_match(ctx.lat_opts_2e, context="BIPOLE Fock build")
        exact_j_cutoff = ctx.exact_j_ke_cutoff if ctx.exact_j_for_pure_rks else None
        if exact_j_cutoff != self.exact_j_ke_cutoff:
            raise ValueError("BIPOLE Fock build: exact_j_ke_cutoff changed after domain resolution")
        for field, actual in (
            ("ewald_alpha_bohr_inv", ctx.omega_used),
            ("ewald_precision", ctx.ewald_precision),
            ("sr_image_extent_bohr", ctx.sr_image_extent),
            ("use_ewald_j_split", ctx.use_ewald_j_split),
            ("exchange_ewald_split", ctx.exchange_split_active),
            ("exact_zone_bohr", ctx.exact_zone_bohr),
        ):
            if actual != getattr(self, field):
                raise ValueError(f"BIPOLE Fock build: {field} changed after domain resolution")


def capture_bipole_domain(*, lattice_1e, lattice_2e, ewald_options,
                          alpha, reciprocal_cutoff, ewald_precision,
                          sr_image_extent, use_ewald_j_split,
                          exchange_ewald_split, exchange_exxdiv, exact_zone,
                          ks_options=None, v_ne_grid_options=None,
                          xc_density_domain=None, exact_j_ke_cutoff=None,
                          nuclear_erfc_options=None, requested_lattice=None,
                          sr_truncation_estimate=None):
    """Freeze already resolved settings, without choosing another cutoff."""
    return BipoleResolvedDomain(
        sr_truncation_estimate=(None if sr_truncation_estimate is None
                                else tuple(sr_truncation_estimate.as_record())),
        requested_lattice=(None if requested_lattice is None else
                           OptionSnapshot.capture(requested_lattice, LATTICE_FIELDS)),
        nuclear_erfc=(None if nuclear_erfc_options is None else
                      OptionSnapshot.capture(nuclear_erfc_options, LATTICE_FIELDS)),
        v_ne_grid=(None if v_ne_grid_options is None else
                   OptionSnapshot.capture(v_ne_grid_options, GRID_FIELDS)),
        xc_density_domain=(None if xc_density_domain is None else _value(xc_density_domain)),
        exact_j_ke_cutoff=exact_j_ke_cutoff,
        lattice_1e=OptionSnapshot.capture(lattice_1e, LATTICE_FIELDS),
        lattice_2e=OptionSnapshot.capture(lattice_2e, LATTICE_FIELDS),
        ewald_alpha_bohr_inv=alpha,
        reciprocal_cutoff_bohr_inv=reciprocal_cutoff,
        ewald_real_cutoff_bohr=(None if ewald_options is None
                               else float(ewald_options.real_cutoff_bohr)),
        ewald_precision=float(ewald_precision),
        sr_image_extent_bohr=sr_image_extent,
        use_ewald_j_split=bool(use_ewald_j_split),
        exchange_ewald_split=bool(exchange_ewald_split),
        exchange_exxdiv=exchange_exxdiv,
        exact_zone_bohr=exact_zone,
        xc_grid=(None if ks_options is None
                 else OptionSnapshot.capture(ks_options.grid, GRID_FIELDS)),
        use_periodic_becke=(False if ks_options is None
                            else bool(ks_options.use_periodic_becke)),
        becke_image_radius_bohr=(None if ks_options is None
                                else float(ks_options.becke_image_radius_bohr)),
    )
