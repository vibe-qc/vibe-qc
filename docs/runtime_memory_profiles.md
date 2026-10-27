# Explicit runtime memory profiles

`vibeqc.memory_profile.runtime_memory_profile(profile, identity)` opts an estimator
call into a frozen empirical residual profile. No profile is discovered or enabled
by default. The caller authenticates the actual runtime source, native binary,
hardware class and command harness before supplying their identity. A mismatch,
unsupported route, TD response, missing allocation or out-of-range basis count
keeps the peak unknown. Initial support is single-thread molecular RHF with
conventional or direct integrals, separately profiled.

`fit_runtime_memory_profile(domain, observations)` uses exactly the maximum
positive training residual `max(peak_bytes - allocation_bytes, 0)`. Allocation
bytes exclude the assumed Python runtime floor. Observations are independently
measured fresh calculator processes including accounted descendants; observer
memory is excluded. Preserve raw observations and their digests. Domain basis
bounds must equal observed training bounds; interpolation remains an empirical
assumption, not a proven memory bound. The same authenticated command harness
must be used for training and validation. Reserve unseen inputs before evaluating
a frozen profile; do not tune to held-out failures.

The manifest keeps analytical allocation terms separate from
`empirical_baseline_json`, whose provenance includes runtime/native/harness,
hardware, method, route, worker count, dimension range and training receipts.
`model_kind=analytical_allocations_with_empirical_residual` makes this distinction
explicit. `complete=true` means the predictive model has allocation coverage plus
an applicable residual profile; it does **not** mean validated accuracy or a
physical bound. The residual includes unmodelled live memory and allocator effects,
so it is not a disjoint measured resident object. Legacy display estimates are
unchanged. Program headroom is unchanged and applied separately to the raw
prediction; queue margin remains separate. Unsupported routes remain unknown.
