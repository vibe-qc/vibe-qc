"""Explicit molecular integral-engine selection, independent J/K algorithms."""
from __future__ import annotations

from dataclasses import dataclass, field

from . import _vibeqc_core as _core


@dataclass(frozen=True)
class IntegralEngineOptions:
    """Opt-in integral provider and explicit resource limits.

    Omission at the ordinary method entry points preserves their legacy route.
    The molecular route is RHF/UHF/RKS/UKS with explicit HCORE or
    supplied-density READ, exact/RI/BUPO/RI-BUPO J and exact or streamed RI K. There is no implicit
    provider or algorithm fallback. Its budget covers integral builds, known
    Hamiltonian/confirmation buffers and declared other_live_bytes; the native
    molecular SCF solver retains its own resource policy. Selected periodic
    jobs support neutral 3D RHF on full regular meshes with explicit
    PeriodicIntegralOptions and exact reciprocal J/K. Their budget covers
    domain generation, complete native SCF and managed output reservations.
    """

    backend: str = "libint"
    fallback: str = "error"
    channel_batch: int = 1
    budget: object = field(default_factory=lambda: _core._manta.ResourceBudget())


@dataclass(frozen=True)
class CoulombBuildOptions:
    """Coulomb algorithm and shared RI source when either J or K uses RI."""

    algorithm: str = "exact"
    auxiliary_basis: object = None
    metric: object = field(default_factory=lambda: _core._manta.RIOptions())
    bupo: object = field(default_factory=lambda: _core._manta.BupoOptions())
    ri_bupo: object = field(default_factory=lambda: _core._manta.BupoFittingOptions())


@dataclass(frozen=True)
class ExchangeBuildOptions:
    """Exact-exchange algorithm, chosen independently of Coulomb."""

    algorithm: str = "exact"


_PROVENANCE_FIELDS = (
    "method", "provider_version", "channel_batch", "functional", "xc_provider", "xc_source_identity",
    "range_exchange_source_identity", "xc_grid_points", "vv10_grid_points",
    "range_exchange_shell_calls", "range_exchange_build_calls", "exchange_active",
    "full_exchange_fraction", "range_exchange_fraction", "range_omega",
    "orbital_basis_identity", "molecule_identity", "requested_backend",
    "one_electron_provider", "coulomb_provider", "exchange_provider",
    "coulomb_algorithm", "exchange_algorithm", "one_electron_source_identity",
    "confirmation_coulomb_algorithm", "confirmation_coulomb_source_identity",
    "confirmation_coulomb_build_calls", "bupo_partition_identity", "bupo_near_provider",
    "bupo_build_calls", "bupo_order", "bupo_near_shell_calls", "bupo_far_blocks", "bupo_moment_panels",
    "bupo_moment_primitive_pairs",
    "bupo_moment_primitive_products",
    "ri_bupo_metric_rank",
    "ri_bupo_forward_source_identity",
    "ri_bupo_adjoint_source_identity",
    "ri_bupo_metric_factor_identity",
    "ri_bupo_last_energy_bound",
    "ri_bupo_last_forward_bound",
    "ri_bupo_last_adjoint_bound",
    "ri_bupo_last_metric_residual",
    "ri_bupo_metric_inverse_norm_bound",
    "bupo_integral_tolerance", "bupo_maximum_integral_error", "bupo_last_j_bound", "bupo_setup_wall_seconds",
    "coulomb_source_identity", "exchange_source_identity",
    "terminal_density_alpha_identity", "terminal_density_beta_identity", "fallback_reason",
    "one_electron_shell_calls", "jk_shell_calls", "jk_build_calls",
    "primitive_integrals_evaluated", "integral_modeled_peak_bytes",
    "primitive_evaluations_measured", "full_build_confirmed", "stationarity_confirmed",
    "terminal_energy_delta", "terminal_fock_delta", "terminal_commutator_norm",
    "one_electron_wall_seconds", "jk_wall_seconds",
)


def molecular_engine_provenance(result) -> dict | None:
    """JSON-compatible copy of executed molecular engine evidence, if present."""
    receipt = getattr(result, "engine_provenance", None)
    if receipt is None:
        return None
    record = {name: getattr(receipt, name) for name in _PROVENANCE_FIELDS}
    record["schema_version"] = 4
    record["budget_scope"] = "integral-builds-and-hamiltonian"
    record["screening"] = "none"
    record["incremental"] = False
    record["citation_methods"] = ["manta_md"] if "manta" in (
        receipt.one_electron_provider, receipt.coulomb_provider, receipt.exchange_provider
    ) else []
    if receipt.coulomb_algorithm in ("bupo", "ri_bupo"):
        record["citation_methods"].append("manta_bupo")
    return record


def record_selected_hf_gradient(native_result):
    """Emit the verified source after the native selected HF gradient succeeds."""
    receipt = getattr(native_result, "engine_provenance", None)
    if receipt is None:
        return
    from .output import record

    fields = ("method", "one_electron_provider", "coulomb_provider", "exchange_provider",
              "coulomb_algorithm", "exchange_algorithm", "molecule_identity", "orbital_basis_identity",
              "one_electron_source_identity", "coulomb_source_identity", "exchange_source_identity",
              "terminal_density_alpha_identity", "terminal_density_beta_identity")
    provenance = {name: getattr(receipt, name) for name in fields}
    fitted = "ri" in (receipt.coulomb_algorithm, receipt.exchange_algorithm)
    provenance["schema_version"] = 1
    provenance["citation_methods"] = (["manta_ri_hf_gradient" if fitted else "manta_hf_gradient"]
                                      if receipt.one_electron_provider == "manta" else [])
    record("molecular_hf_gradient",
           f"  HF gradient: {receipt.one_electron_provider}; "
           f"J={receipt.coulomb_algorithm}; K={receipt.exchange_algorithm}; "
           "confirmed SCF source.\n",
           gradient_provenance=provenance)


def run_selected_scf(method, molecule, basis, options, *, integral_engine, coulomb, exchange,
                    read_from=None, fragments=None, dft_plus_u=None, grid_level="orca-defgrid3"):
    from . import _attach_inline_ecp_from_basis_sidecar, _coerce_molecular_basis
    from .density_fitting import check_aux_basis_coverage
    from .output import record

    integral_engine = IntegralEngineOptions() if integral_engine is None else integral_engine
    coulomb = CoulombBuildOptions() if coulomb is None else coulomb
    exchange = ExchangeBuildOptions() if exchange is None else exchange
    if not isinstance(integral_engine, IntegralEngineOptions):
        raise TypeError("integral_engine must be IntegralEngineOptions")
    if not isinstance(coulomb, CoulombBuildOptions) or not isinstance(exchange, ExchangeBuildOptions):
        raise TypeError("coulomb/exchange must use CoulombBuildOptions/ExchangeBuildOptions")
    if integral_engine.fallback != "error":
        raise NotImplementedError("Selected molecular same-source fallback is not yet qualified; use fallback='error' or select libint explicitly")
    if integral_engine.backend not in ("manta", "libint"):
        raise ValueError("integral_engine.backend must be manta or libint")
    if isinstance(integral_engine.channel_batch, bool) or not isinstance(integral_engine.channel_batch, int):
        raise TypeError("channel_batch must be an integer")
    if coulomb.algorithm not in ("exact", "ri", "bupo", "ri_bupo") or exchange.algorithm not in ("exact", "ri"):
        raise ValueError("Selected molecular SCF requires exact, ri, bupo or ri_bupo J and exact or ri K")
    if coulomb.algorithm == "bupo" and integral_engine.backend != "manta":
        raise ValueError("Selected BUPO uses MANTA moments; select backend='manta' and choose bupo.near_backend explicitly")
    if coulomb.algorithm == "ri_bupo" and integral_engine.backend != "manta":
        raise ValueError("Selected RI-BUPO requires backend='manta'")
    if read_from is not None or fragments is not None or dft_plus_u:
        raise NotImplementedError("Selected molecular SCF does not yet admit READ projection, FRAGMO or DFT+U; supply a same-basis READ density directly")
    cls = {"rhf": _core.RHFOptions, "uhf": _core.UHFOptions,
           "rks": _core.RKSOptions, "uks": _core.UKSOptions}[method]
    if options is None:
        opts = cls()
        opts.initial_guess = _core.InitialGuess.HCORE
        if method in ("rks", "uks"):
            from .runner import _apply_grid_level
            _apply_grid_level(opts.grid, grid_level)
    elif isinstance(options, cls):
        opts = cls(options)
    else:
        raise TypeError(f"{method} options must be {cls.__name__}")
    basis = _coerce_molecular_basis(molecule, basis)
    _attach_inline_ecp_from_basis_sidecar(opts, molecule, basis)
    auxiliary = coulomb.auxiliary_basis
    wants_ri = coulomb.algorithm in ("ri", "ri_bupo") or exchange.algorithm == "ri"
    if wants_ri and auxiliary is None:
        raise ValueError("Selected RI algorithm requires coulomb.auxiliary_basis")
    if not wants_ri and auxiliary is not None:
        raise ValueError("An auxiliary basis requires a selected RI algorithm")
    if isinstance(auxiliary, str):
        auxiliary = _core.BasisSet(molecule, auxiliary)
    if auxiliary is not None:
        if not isinstance(auxiliary, _core.BasisSet):
            raise TypeError("auxiliary_basis must be a basis name or BasisSet")
        check_aux_basis_coverage(auxiliary, molecule=molecule, orbital_basis=basis,
                                 aux_basis_name=auxiliary.name, orbital_basis_name=basis.name,
                                 route=f"selected {method}")
    native = _core._manta._MolecularHFEngine()
    native.backend, native.coulomb, native.exchange = integral_engine.backend, coulomb.algorithm, exchange.algorithm
    native.channel_batch, native.budget, native.ri = integral_engine.channel_batch, integral_engine.budget, coulomb.metric
    native.bupo = coulomb.bupo
    native.ri_bupo = coulomb.ri_bupo
    result = getattr(_core._manta, f"_run_{method}")(molecule, basis, opts, native, auxiliary)
    provenance = molecular_engine_provenance(result)
    exchange_text = (f"{provenance['exchange_algorithm']} ({provenance['exchange_provider']})"
                     if provenance["exchange_active"] else "inactive (pure XC)")
    record("molecular_integral_engine",
           f"  Integral engine: {provenance['one_electron_provider']}; "
           f"J={provenance['coulomb_algorithm']} ({provenance['coulomb_provider']}); "
           f"K={exchange_text}.\n"
           f"  Full-density confirmation: {provenance['full_build_confirmed']}; "
           f"stationarity: {provenance['stationarity_confirmed']}; "
           f"J/K shell calls: {provenance['jk_shell_calls']}.\n",
           engine_provenance=provenance)
    if provenance["coulomb_algorithm"] in ("bupo", "ri_bupo"):
        record("molecular_bupo_source",
               f"  BUPO: order {provenance['bupo_order']}; near provider {provenance['bupo_near_provider']}; "
               f"far blocks {provenance['bupo_far_blocks']}; moment panels {provenance['bupo_moment_panels']}.\n"
               f"  Confirmation J: {provenance['confirmation_coulomb_algorithm']}; "
               f"last analytic J approximation bound: {provenance['bupo_last_j_bound']:.3e}.\n",
               engine_provenance=provenance)
    if provenance["coulomb_algorithm"] == "ri_bupo":
        record("molecular_ri_bupo_source",
               f"  RI-BUPO: exact metric rank {provenance['ri_bupo_metric_rank']}; "
               f"forward coupling bound {provenance['ri_bupo_last_forward_bound']:.3e}; "
               f"adjoint coupling bound {provenance['ri_bupo_last_adjoint_bound']:.3e}.\n"
               f"  RI-BUPO energy approximation bound: {provenance['ri_bupo_last_energy_bound']:.3e}; "
               f"metric residual: {provenance['ri_bupo_last_metric_residual']:.3e}.\n",
               engine_provenance=provenance)
    if provenance["xc_provider"]:
        record("molecular_xc_source",
               f"  XC: {provenance['functional']} ({provenance['xc_provider']}); "
               f"grid points: {provenance['xc_grid_points']}; "
               f"exchange fractions: full={provenance['full_exchange_fraction']}, "
               f"erf={provenance['range_exchange_fraction']} at omega={provenance['range_omega']}.\n",
               engine_provenance=provenance)
    return result
