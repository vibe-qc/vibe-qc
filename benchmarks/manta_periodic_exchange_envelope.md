# Canonical occupied-space exchange certificate

Run `python benchmarks/manta_periodic_exchange_envelope.py --repeats 3`
with one numerical thread and the native runtime from this checkout. Each
process measures a cold certificate, repeated canonical analysis and typed
envelope reuse. Constructor cost, modeled peak memory and RSS are separate.

The component cases retain the original two-point uniform mesh and signed
contracted s basis. Changing the physical AO-image or reciprocal support
changes the omitted domain. The complete RI RHF case uses fitted J and exact
K; its exchange certificate is evaluated separately on the same finite
canonical space. It does not enable a complete RI exchange target.

The experiment reports component costs and enclosure tightness. It does not
claim a production crossover or automatic whole-method accuracy. The
certificate excludes BvK correction, erfc exchange, fitting/metric error,
roundoff and relaxed-state error. The measurements below record the qualified local queue acceptance.

## Qualified measurements

Three independent sequential processes each performed a cold certificate and
three repeats, with one numerical thread. These timings are medians of the
nine repeated calls. The two-point component source uses two contracted s
functions and uniform weights. The finite RI RHF source has fitted J and
exact K; its certificate remains a separate component operation.

| Source | Maximum K bound | Spin-total energy bound (Eh) | Analysis + bound (ms) | Reuse bound (ms) | Modeled peak (MiB) |
|---|---:|---:|---:|---:|---:|
| AO cube 0, reciprocal cube 2 | 7.595036e+07 | 1.771327e+08 | 0.0055 | 0.0049 | 8.134 |
| AO cube 1, reciprocal cube 2 | 3.204660e+05 | 7.232856e+05 | 0.0054 | 0.0046 | 8.217 |
| AO cube 1, reciprocal cube 8 | 2.144372e+05 | 4.803197e+05 | 0.0055 | 0.0045 | 10.555 |
| Complete finite RI RHF | 4.113411e+03 | 6.652200e+05 | 0.0104 | 0.0081 | 8.176 |
| Compact Gamma source | 1.761519e-11 | 5.935745e-10 | 0.0042 | 0.0042 | 35.859 |

Maximum process RSS was 240.4 MiB. The detailed JSON
also reports source preparation time and both energy bounds. Larger physical
support changes the retained operator; changing the reuse schedule does not.
The diffuse cases and the current finite RI source are far too loose for a
1e-6 target. The compact Gamma source meets 1e-8. This demonstrates an
admitted regime while exposing the need for tighter bounds or broader
physical domains before automatic general-purpose admission. These small
cases establish correctness and cost, not a production speedup.

Local vq acceptance exercised eight cases, 576 random spin-pair occupied
states and four complete HF calculations (three RI and one BUPO), including an empty beta
channel and a capped open-shell state. Independent complex-contour Gaussian
quadrature checked the finite and enlarged exchange sums; the additional
reciprocal shell was below 3e-12. NumPy eigenspaces and 70-digit Decimal
positive contractions independently checked the envelope composition.
The maximum complete RI Fock difference was 4.773959e-15 Eh.
The native Gaussian majorant itself reuses its earlier qualified producer.
No suitable ORCA/CRYSTAL reference matches these custom finite domains.

Scientific source: `b323eadbef34be2c1798a56122f878dff76f1d7c`. Native SHA256:
`38b50df5eefa8c7d52668f3265d4dd6cb671d66efce1201edde09ccd1563643f`. The initial seven-case source was `a751cc625c9d6b320f578be8c9535994598682c8`.
The later source adds a test-only import correction and compact benchmark;
the numerical code and initial payload are unchanged. The full automatic
RI-BUPO accuracy ledger and physical support selection remain unfinished.
