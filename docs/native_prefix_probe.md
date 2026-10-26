# FFTW final-prefix observations

`scripts/probe_fftw_prefix.py` is a standalone observation helper, not a Queue
seed adapter. It never admits reuse or attests historical provenance. Invoke
it with an existing Python 3.9+ interpreter and a JSON request on standard input:

```json
{"candidate":"/absolute/candidate/install","donor":"/absolute/donor/source","library":"lib/libfftw3.3.dylib"}
```

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
data qualified here: this does not test chemistry basis data, threaded FFTW,
other dependencies, or general ABI portability.

Successful output uses `vibeqc.fftw-prefix-probe/1`, `decision: observed`,
`source_build_provenance: not_attested` and `seed_admission: not_evaluated`.
Failures return `decision: hold` with nonzero exit. This is deliberately not
`vq.native-seed-verification/1`: a product adapter must first authenticate a
fresh receipt, resolve all candidate semantic inputs, compare complete install
manifests, and establish the actual final candidate identity under Queue's
locks. The caller must also preserve donor lifecycle protection. Only such an
adapter may incorporate these observations into its qualification decision.
Never convert an observation alone into permission for copying or activation.
