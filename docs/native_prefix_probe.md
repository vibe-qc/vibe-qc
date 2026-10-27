# FFTW final-prefix observations

`scripts/probe_fftw_prefix.py` is a standalone observation helper, not a Queue
seed adapter. It never admits reuse or attests historical provenance. Invoke
it with an existing Python 3.9+ interpreter and a JSON request on standard input:

```json
{"candidate":"/absolute/candidate/install","donor":"/absolute/donor/source","library":"lib/libfftw3.3.dylib"}
```

For builds with the separate threads library, also supply
`"threads_library":"lib/libfftw3_threads.3.dylib"`. The probe then initializes
FFTW threads, requests two threads for a second plan, executes and checks the
transform, and destroys the plan and thread state. It observes initialization,
thread-count configuration and cleanup symbols in that candidate threads
library. The plan/execute/destroy symbols resolved through the threaded handle
must belong to the candidate core, rejecting a mismatched transitive core.
Both library hashes are recorded and checked before/after isolation. Requested
thread count is reported separately from actual workers, which are not observed
for this small transform. This is not qualification of the OpenMP library.

Paths must exist, be canonical and disjoint. The library is a regular file
relative to the final candidate install, not a symlink alias. The candidate
must also contain `include/fftw3.h`. No files are copied or compiled. The
helper uses Darwin `sandbox-exec` to deny reads below the entire donor source.
It verifies donor readability before and after the child, and actual read
denial inside it. Missing paths are not evidence of isolation. Other platforms
return a hold until an isolation backend is implemented and checked.

The child reads `FFTW_FORWARD` and `FFTW_ESTIMATE` from the candidate header,
uses them for a four-point complex transform, and checks its known result.
`dladdr` must bind each called FFTW entry point to the candidate library.
Observed library and header paths and SHA-256 values accompany the result;
before/after hashes reject byte changes. Header ABI constants are the only
data qualified here: this does not test chemistry basis data, other
dependencies, OpenMP, or general ABI portability. Omitting `threads_library`
retains the original serial-only observation and provides no threaded claim.

Successful output uses `vibeqc.fftw-prefix-probe/1`, `decision: observed`,
`source_build_provenance: not_attested` and `seed_admission: not_evaluated`.
Failures return `decision: hold` with nonzero exit. This is deliberately not
`vq.native-seed-verification/1`: a product adapter must first authenticate a
fresh receipt, resolve all candidate semantic inputs, compare complete install
manifests, and establish the actual final candidate identity under Queue's
locks. The caller must also preserve donor lifecycle protection. Only such an
adapter may incorporate these observations into its qualification decision.
Never convert an observation alone into permission for copying or activation.


## Prospective archive collection

For an already scheduled fresh FFTW build, an operator may explicitly set
`VIBEQC_NATIVE_SOURCE_ARCHIVE` to an existing unpatched upstream archive,
alongside `VIBEQC_NATIVE_BUILD_COLLECTOR` and `VIBEQC_NATIVE_RECEIPT_DIR`.
The hook checks the archive against the build recipe's pinned SHA-256 before
configuration, writes `fftw-spec.json` exclusively in the external receipt
directory, then invokes the external collector using `record --spec`.
The spec binds the archive pin and extraction prefix to the actual source,
recipe, configured build, install and install-manifest paths. The collector
must verify extracted source bytes before and after compilation; passing a
spec is not proof that it did so. A collector failure is propagated without
retrying the build. Existing cache hits do not create retrospective receipts.

This option currently supports unpatched FFTW only. It refuses other dependency
routes rather than misrepresenting patched or multistage source provenance.
Omitting it retains the existing collector invocation, whose source identity
alone does not attest the upstream archive. No default collector is installed
or enabled, no archive is fetched, and no build is scheduled by this option.
A successful receipt still needs semantic comparison and final-prefix native
and data verification before a Queue adapter can qualify reuse.
