"""Fit sources of the neutral fitted torus: who builds the per-q fits (#519, RG-7).

The real-Gamma neutral cderi is assembled from per-channel unit-cell
Bloch-pair fits ``L_{ka, ka+q}[P, mu, nu]`` in the canonical auxiliary frame,
one batch per kept momentum transfer ``q``
(:func:`~vibeqc.periodic.ccm.neutral.ccm_neutral_cderi_fold`), and either
folded onto the torus or kept as block storage
(:class:`~vibeqc.periodic.ccm.block_cderi.CCMBlockCderi`). Which code builds
those fits is the **fit source**. This module makes that boundary explicit,
so the fold and the block storage stay identical whatever builds the fits:

* :class:`ClassicRSGDFFitSource` (the default): the multi-k GDF driver's
  classic all-reciprocal RSGDF builder
  (:func:`~vibeqc.aux_basis.build_lpq_bloch_native_fft_shared_q`, or the
  per-pair :func:`~vibeqc.aux_basis.build_lpq_bloch_native_fft`), the libint /
  native FFT path every real-Gamma result has used so far;
* :class:`DriverFitCacheSource`: the same classic fits retained by a
  multi-k GDF run in this process (``RSGDFFitCache``), validated to be the
  fold's operator, rotated to the canonical frame and served without a refit.

A source is declared unbound and bound by the fold to its context
(:class:`FoldContext`: unit cell, orbital and auxiliary bases, mesh, cutoffs,
tail, cell list, screening, worker count). The bound object answers
``channel_fits(qi, bras, pool=None) -> {ai: L}`` for the bras the fold asks
for, says whether the fold may use the exact time-reversal relation inside a
self-paired channel and the space-group star plan on its fits, and gives a
JSON-able ``receipt()`` for ``cderi_info``. The fold keeps its own
orchestration (kept channels, time reversal, star reconstruction, the channel
pool, the fold or the block storage); a source only builds fits.

``engine`` names the fitted **operator** family, not the transport: the
classic fold and a driver cache of the classic fit report the engine the
multi-k driver reports for it (``"native-classic-fft-shared-q"``), so the
neutral-source identity of #519 RG-2 is the same whether the fits were built
here or consumed from the driver. A future source that fits a different
operator (the MANTA periodic GDF tiles, RG-8) reports its own engine and is a
different Hamiltonian by identity.

Cite Sun, Berkelbach, McClain & Chan, J. Chem. Phys. 147, 164119 (2017),
doi:10.1063/1.4998644 for the periodic GDF factorisation the fits realise.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np

__all__ = [
    "CLASSIC_ENGINE",
    "MANTA_ENGINE",
    "ClassicRSGDFFitSource",
    "DriverFitCacheSource",
    "FoldContext",
    "MANTAFitSource",
]

#: Engine identity of the classic RSGDF fit, as the multi-k driver records it
#: (``RSGDFFitSource.engine``), whichever batching the fold uses.
CLASSIC_ENGINE = "native-classic-fft-shared-q"
#: Engine identity of the MANTA periodic GDF tiles (#519, RG-8).
MANTA_ENGINE = "manta-periodic-gdf-tiles"


@dataclass(frozen=True)
class FoldContext:
    """What the fold hands a source when it binds it (one fold, one context)."""

    unit: object
    ubasis: object
    aux_modrho: object
    aux_name: str
    kpts: np.ndarray
    frac: np.ndarray
    nrep: np.ndarray
    b_lat: np.ndarray
    ke_cutoff: float
    tail: object
    lat_opts: object
    omega_screen: float
    n_workers: int


class _Stats:
    """Thread-safe counters a bound source exposes to the fold's log line."""

    def __init__(self):
        self._lock = threading.Lock()
        self.values = {"builds": 0, "batches": 0, "bounded_chunks": 0,
                       "served": 0}

    def add(self, **inc):
        with self._lock:
            for key, value in inc.items():
                self.values[key] += int(value)

    def __getitem__(self, key):
        return self.values[key]


class ClassicRSGDFFitSource:
    """The classic RSGDF builder of the multi-k GDF driver (the default).

    ``fit_batch`` selects the builder: ``"shared-q"`` (default) issues one
    batched call per channel over the requested bras; ``"per-pair"`` is the
    historical single-pair loop, kept as the reference. Both are the fold's
    builds verbatim (moved here unchanged), so a fold through this source is
    bit-identical to the fold before the boundary existed.
    """

    supports_time_reversal = True
    supports_star_plan = True
    engine = CLASSIC_ENGINE

    def __init__(self, fit_batch: str = "shared-q"):
        if fit_batch not in ("shared-q", "per-pair"):
            raise ValueError(
                "ClassicRSGDFFitSource: fit_batch must be 'shared-q' or "
                f"'per-pair', got {fit_batch!r}")
        self.fit_batch = fit_batch

    def bind(self, ctx: FoldContext) -> "_BoundClassic":
        return _BoundClassic(self, ctx)


class _BoundClassic:
    supports_time_reversal = True
    supports_star_plan = True
    engine = CLASSIC_ENGINE
    transport = "built"

    def __init__(self, spec: ClassicRSGDFFitSource, ctx: FoldContext):
        self.fit_batch = spec.fit_batch
        self.ctx = ctx
        self.stats = _Stats()

    def channel_fits(self, qi, bras, pool=None):
        """Canonical-frame ``L_{ka, ka+q}`` for every ``ka = kpts[ai]`` in
        ``bras`` of channel ``q = kpts[qi]``, the ket unfolded."""
        from vibeqc.aux_basis import (
            build_lpq_bloch_native_fft,
            build_lpq_bloch_native_fft_shared_q,
        )
        from .neutral import _SINGLE_PAIR_CHUNK_G, _fold_batch_chunk_g

        ctx = self.ctx
        bras = list(bras)
        fits: dict = {}
        if not bras:
            return fits
        if self.fit_batch == "shared-q":
            chunk_g, chunk_eff = _fold_batch_chunk_g(
                len(bras), int(ctx.ubasis.nbasis), n_workers=ctx.n_workers)
            if chunk_eff < _SINGLE_PAIR_CHUNK_G:
                self.stats.add(bounded_chunks=1)
            lpq_list = build_lpq_bloch_native_fft_shared_q(
                ctx.unit, ctx.ubasis, ctx.aux_modrho, ctx.kpts[bras],
                ctx.kpts[qi],
                ke_cutoff=float(ctx.ke_cutoff), tail_ke_cutoff=ctx.tail,
                tail_chunk_g=chunk_g,
                lat_opts=ctx.lat_opts,
                canonical_auxiliary_basis=True,
                omega_screen=float(ctx.omega_screen))
            for ai, l_ab in zip(bras, lpq_list):
                fits[ai] = np.asarray(l_ab)
            self.stats.add(batches=1, builds=len(bras))
            return fits

        # Historical path: one single-pair build per (ka, ka+q).
        def _build_one(ai, _qi=qi):
            return ai, np.asarray(build_lpq_bloch_native_fft(
                ctx.unit, ctx.ubasis, ctx.aux_modrho, ctx.kpts[ai],
                ctx.kpts[ai] + ctx.kpts[_qi],
                ke_cutoff=float(ctx.ke_cutoff), tail_ke_cutoff=ctx.tail,
                lat_opts=ctx.lat_opts,
                canonical_auxiliary_basis=True,
                omega_screen=float(ctx.omega_screen)))

        if pool is not None and len(bras) > 1:
            for ai, l_ab in pool.map(_build_one, bras):
                fits[ai] = l_ab
        else:
            for ai in bras:
                fits[ai] = _build_one(ai)[1]
        self.stats.add(builds=len(bras))
        return fits

    def receipt(self) -> dict:
        ctx = self.ctx
        return {
            "provider": "classic-rsgdf",
            "engine": self.engine,
            "transport": self.transport,
            "fit_batch": self.fit_batch,
            "auxiliary_basis": str(ctx.aux_name),
            "auxiliary_frame": "canonical-modrho",
            "base_reciprocal_cutoff_ha": float(ctx.ke_cutoff),
            "tail_reciprocal_cutoff_ha": (None if ctx.tail is None
                                          else float(ctx.tail)),
            "lattice_cutoff_bohr": float(ctx.lat_opts.cutoff_bohr),
            "support_policy": "historical_index_ball",
            "omega_screen": float(ctx.omega_screen),
        }


class DriverFitCacheSource:
    """The classic fits retained by a multi-k GDF run (``RSGDFFitCache``).

    Binding validates that the cache was fit at the fold's operator (bases,
    ``ke_cutoff``, tail, cell-list cutoff, metric threshold, fit screen, pair
    support, no screening) and that its k-points cover the BvK mesh, then
    serves each requested pair rotated to the canonical auxiliary frame
    (:class:`~vibeqc.periodic.ccm.neutral._FitCacheSource`). The time-reversal
    and star relations are the builder's, not the cache's, so the fold uses
    neither on cached fits.
    """

    supports_time_reversal = False
    supports_star_plan = False
    engine = CLASSIC_ENGINE

    def __init__(self, cache):
        self.cache = cache

    def bind(self, ctx: FoldContext) -> "_BoundDriverCache":
        return _BoundDriverCache(self, ctx)


class _BoundDriverCache:
    supports_time_reversal = False
    supports_star_plan = False
    engine = CLASSIC_ENGINE
    transport = "driver-fit-cache"

    def __init__(self, spec: DriverFitCacheSource, ctx: FoldContext):
        from .neutral import _FitCacheSource

        self.ctx = ctx
        self.cache = spec.cache
        self._serve = _FitCacheSource(
            spec.cache, unit=ctx.unit, ubasis=ctx.ubasis,
            aux_modrho=ctx.aux_modrho, b_lat=ctx.b_lat, nrep=ctx.nrep,
            frac=ctx.frac, ke_cutoff=float(ctx.ke_cutoff), tail=ctx.tail,
            lat_opts=ctx.lat_opts, omega_screen=float(ctx.omega_screen))
        self.stats = _Stats()

    def channel_fits(self, qi, bras, pool=None):
        fits = {ai: self._serve(qi, ai) for ai in bras}
        self.stats.add(served=len(fits))
        return fits

    def receipt(self) -> dict:
        ctx = self.ctx
        return {
            "provider": "classic-rsgdf",
            "engine": self.engine,
            "transport": self.transport,
            "cache_pairs": int(len(self.cache.pairs)),
            "auxiliary_basis": str(ctx.aux_name),
            "auxiliary_frame": "canonical-modrho",
            "base_reciprocal_cutoff_ha": float(ctx.ke_cutoff),
            "tail_reciprocal_cutoff_ha": (None if ctx.tail is None
                                          else float(ctx.tail)),
            "lattice_cutoff_bohr": float(ctx.lat_opts.cutoff_bohr),
            "support_policy": "historical_index_ball",
            "omega_screen": float(ctx.omega_screen),
        }


class MANTAFitSource:
    """The fitted source of MANTA's periodic RI HF (#519, RG-8).

    ``options`` is a ``vibeqc.manta.PeriodicRIRHFOptions`` (or the UHF
    sibling): the finite source of MANTA's ``run_periodic_ri_rhf_from_cutoffs``
    at ``reciprocal_energy_cutoff`` (pair cutoff, fitting ``omega``, auxiliary
    image radius, integral screen, RI metric truncation). Binding builds that
    method's own domains for the fold's BvK mesh
    (``build_periodic_ri_rhf_domains``); each kept channel ``q`` then gets one
    ``PeriodicGDFTiles`` over every requested bra (transfer ``q``, kets
    ``ka + q`` unfolded, the channel's vector set: the primed Gamma sphere at
    ``q = 0``, else the exchange sphere of the ``(Gamma, q)`` transfer) and one
    ``PeriodicRIQBuilder`` whose fixed factor ``F`` gives the fits
    ``L_{ka, ka+q} = B(ka + q) = F^H T(ka, ka + q)``. One factor per channel
    is the frame the fold and the correlated consumers need; MANTA's
    whole-mesh Fock builds one factor per ``(bra, ket)`` pair instead, which
    is the same finite metric on the same vector set and therefore the same
    J and K (exchange is invariant under the auxiliary frame).

    A MANTA-sourced torus is a **different neutral source** from the classic
    fold (a different fitting metric, auxiliary treatment and finite
    support): its engine is :data:`MANTA_ENGINE`, so the neutral-source
    identity of RG-2 never equals a classic run's, and its Bloch control is
    MANTA's own whole-mesh RI method, not ``run_ccm_rhf_gdf``. MANTA's tiles
    are evaluated on their admitted native path under the given
    ``budget``; the fits are materialised once into the block storage the
    fold returns, never per SCF iteration. The domain factory admits the
    upper bound of a whole RI HF run before it builds the domains, so the
    budget must be sized for that bound (MANTA's default refuses most
    cells); a fitting metric that is not positive definite on the given
    finite source is refused by MANTA, not repaired.

    Measured (``tests/test_ccm_manta_fit_source.py``, H2 cube STO-3G at
    (2,1,1) and (3,1,1)): J and K of the block storage on these fits equal
    MANTA's whole-mesh ``PeriodicRIFockBuilder`` on the same domains to
    1e-16, and the dense fold of the same fits gives the block storage's J
    and K to rounding (relative 1e-11).

    Sun, Berkelbach, McClain & Chan, J. Chem. Phys. 147, 164119 (2017),
    doi:10.1063/1.4998644 (q-dependent periodic Gaussian fitting).
    """

    supports_time_reversal = False
    supports_star_plan = False
    engine = MANTA_ENGINE

    def __init__(self, options, reciprocal_energy_cutoff, *,
                 auxiliary_basis=None, budget=None):
        cutoff = float(reciprocal_energy_cutoff)
        if not np.isfinite(cutoff) or cutoff <= 0.0:
            raise ValueError(
                "MANTAFitSource: reciprocal_energy_cutoff must be finite and "
                f"positive, got {reciprocal_energy_cutoff!r}")
        for name in ("fitting", "ri", "pair_cutoff_bohr"):
            if not hasattr(options, name):
                raise TypeError(
                    "MANTAFitSource: options must be a MANTA periodic RI HF "
                    f"options object (missing {name!r}); got "
                    f"{type(options).__name__}")
        self.options = options
        self.reciprocal_energy_cutoff = cutoff
        self.auxiliary_basis = auxiliary_basis
        self.budget = budget

    def bind(self, ctx: FoldContext) -> "_BoundMANTA":
        return _BoundMANTA(self, ctx)


class _BoundMANTA:
    supports_time_reversal = False
    supports_star_plan = False
    engine = MANTA_ENGINE
    transport = "built"

    def __init__(self, spec: MANTAFitSource, ctx: FoldContext):
        from vibeqc import manta
        from vibeqc.aux_basis import make_aux_basis_set

        if float(ctx.omega_screen) != 0.0:
            raise NotImplementedError(
                "MANTAFitSource: the screened (omega_screen > 0) fold is a "
                "classic-source object; MANTA's physical range kernels are "
                "not wired into the fold")
        self.ctx = ctx
        self.options = spec.options
        self.cutoff = spec.reciprocal_energy_cutoff
        self.budget = spec.budget if spec.budget is not None else manta.ResourceBudget()
        mol = ctx.unit.unit_cell_molecule()
        self.aux_name = str(spec.auxiliary_basis or ctx.aux_name)
        self.orbital = ctx.ubasis
        self.auxiliary = make_aux_basis_set(mol, aux_name=self.aux_name,
                                            drop_eta=0.0)
        nrep = tuple(int(n) for n in ctx.nrep)
        self.mesh = manta.RegularKMesh(nrep, (0, 0, 0))
        self.domains = manta.build_periodic_ri_rhf_domains(
            self.orbital, self.auxiliary, ctx.unit, self.mesh, self.cutoff,
            self.options, self.budget)
        nvec = np.asarray(nrep, dtype=int)
        n_c = int(np.prod(nvec))
        # MANTA's mesh order, by integer label, against the fold's.
        self._manta_index = {}
        for i in range(n_c):
            frac = np.asarray(self.mesh.fractional(self.mesh.address(i)), dtype=float)
            label = tuple(int(x) for x in np.round(frac * nvec).astype(int) % nvec)
            self._manta_index[label] = i
        if len(self._manta_index) != n_c:
            raise ValueError("MANTAFitSource: MANTA mesh labels are not unique")
        self._gamma = self._manta_index[(0, 0, 0)]
        self._offsets = np.asarray(self.domains.transfer_offsets, dtype=np.int64)
        self._exchange = np.asarray(self.domains.pair_vectors, dtype=float)
        self._n_c = n_c
        self.stats = _Stats()
        self.channel_records = []
        self._lock = threading.Lock()

    def _vectors(self, qi):
        label = tuple(int(x) for x in np.asarray(self.ctx.frac[qi]) % np.asarray(self.ctx.nrep))
        if label == (0, 0, 0):
            return np.ascontiguousarray(self.domains.reciprocal_vectors, dtype=float)
        p = self._gamma * self._n_c + self._manta_index[label]
        lo, hi = int(self._offsets[p]), int(self._offsets[p + 1])
        return np.ascontiguousarray(self._exchange[lo:hi], dtype=float)

    def channel_fits(self, qi, bras, pool=None):
        from vibeqc import manta

        ctx = self.ctx
        bras = list(bras)
        if not bras:
            return {}
        q = np.ascontiguousarray(ctx.kpts[qi], dtype=float)
        kets = np.ascontiguousarray(ctx.kpts[bras] + q[None, :], dtype=float)
        vectors = self._vectors(qi)
        tiles = manta.PeriodicGDFTiles(
            self.orbital, self.auxiliary, ctx.unit, q, kets, vectors,
            self.options.fitting, self.budget)
        fit = manta.PeriodicRIQBuilder(tiles, 1, self.options.ri, self.budget)
        factor = np.asarray(fit.factor)
        n_mu = int(self.orbital.nbasis)
        n_aux = int(self.auxiliary.nbasis)
        t = np.zeros((len(bras), n_aux, n_mu, n_mu), dtype=complex)
        n_aux_sh = len(self.auxiliary.shells())
        n_sh = len(self.orbital.shells())
        for p in range(n_aux_sh):
            for m in range(n_sh):
                for n in range(n_sh):
                    tile = tiles.evaluate(p, m, n)
                    values = np.asarray(tile.values)
                    off = tuple(int(x) for x in tile.offsets)[-3:]
                    _, n_p, n_m, n_n = values.shape
                    t[:, off[0]:off[0] + n_p, off[1]:off[1] + n_m,
                      off[2]:off[2] + n_n] = values
        b = np.einsum("pl,kpmn->klmn", factor.conj(), t, optimize=True)
        with self._lock:
            self.channel_records.append({
                "label": [int(x) for x in ctx.frac[qi]],
                "rank": int(factor.shape[1]),
                "vectors": int(len(vectors)),
                "source_identity": str(tiles.source_identity),
                "projector_identity": str(fit.metric_info.projector_identity),
            })
        self.stats.add(builds=len(bras), batches=1)
        return {ai: np.ascontiguousarray(b[i]) for i, ai in enumerate(bras)}

    def receipt(self) -> dict:
        o = self.options
        return {
            "provider": "manta",
            "engine": self.engine,
            "transport": self.transport,
            "auxiliary_basis": self.aux_name,
            "auxiliary_frame": "per-channel-ri-q-eigenmode",
            "reciprocal_energy_cutoff_ha": float(self.cutoff),
            "pair_cutoff_bohr": float(o.pair_cutoff_bohr),
            "fitting": {"omega": float(o.fitting.omega),
                        "auxiliary_cutoff_bohr": float(o.fitting.auxiliary_cutoff_bohr),
                        "pair_cutoff_bohr": float(o.fitting.pair_cutoff_bohr),
                        "integral_screen_error": float(o.fitting.integral_screen_error)},
            "channels": sorted(self.channel_records, key=lambda r: r["label"]),
        }
