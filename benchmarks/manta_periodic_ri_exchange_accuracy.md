# Complete RI exchange accuracy admission

Run `python benchmarks/manta_periodic_ri_exchange_accuracy.py --repeats 3`
with one numerical thread and this checkout's native runtime. Each process
measures cold and repeated complete jobs with the certificate disabled and
enabled, at identical physical sources. States, energies and Fock source
identities must agree bitwise. This measures whole-job overhead, not speedup.

The ordinary RHF and PBE0 cases retain automatic RI coupling selection and
S/T image admission. Their exchange targets are deliberately loose, exposing
the actual source error bounds. The compact helium RHF case uses exponents
4 and 6, a 6-bohr cubic cell, a 1-bohr pair cutoff, 512-Eh reciprocal cutoff,
12.1-bohr auxiliary support and AO/reciprocal completion radii 2/40. Both
exchange targets are 1e-8. All cases include the fixed BvK correction.

The certificate covers the fixed canonical density family at the original
uniform mesh. It excludes Madelung numerical error, fitting/metric error,
other Hamiltonian components, arithmetic, changed overlap spaces and state
relaxation. It is not a whole-method automatic accuracy claim. Numerical
results and actual source/runtime identities are recorded after acceptance.

## Recorded local acceptance

The 2026-10-18 local queue run pins implementation source
`453933d3413c3be9d058463e4cd1b4b2731c6752` and native SHA256
`80089329d766cb2e334dcd6416b9695839c3f5aade81e6dd5164915d52300690`. It exercises eight complete jobs: strict and fixed-BvK
RHF, empty-beta UHF, capped unequal-spin UHF, restricted/unrestricted PBE0,
and compact strict/BvK RHF. Enabled/disabled runs preserve D, F, H, S,
energy and physical source identities bitwise. Two prior standalone signed
s/p certificates retain their exact arrays and identities; the prior PBE0
state also agrees bitwise. Existing qualifying evidence is reused.

Independent finite Gaussian/Boys/Fourier/libint and SciPy HF references
observe maximum Fock error 6.417089e-13 Eh and energy error 2.601253e-13 Eh.
Independent Gaussian overlap enlargements test the fixed-BvK difference
at the evaluated densities and 512 random spin-pair subspaces. NumPy
canonical spaces and 70-digit Decimal contractions check the density
family, unscaled exchange, method scale, BvK and energy intersection
(424 matrix entries). Source admission and public integration
pass 1,066 affected checks. A validation-harness reconstruction initially
changed the last bits of k and was correctly refused by the identity check;
using actual native momenta resolved it without changing science or gates.
The compact high-cutoff comparison also required the previously qualified
contour-shifted Fourier reference: ordinary real-axis quadrature was
under-resolved. Both failed harness attempts remain in the private evidence;
no source, runtime or acceptance tolerance was changed.

### Measured component limits

| Case | Maximum matrix error bound / Eh | Exchange energy bound / Eh | BvK matrix bound / Eh |
| --- | ---: | ---: | ---: |
| RHF Be strict | 4.703118e+02 | 1.514911e+05 | 0.000000e+00 |
| RHF Be BvK automatic RI | 4.703118e+02 | 1.514911e+05 | 5.095245e-04 |
| UHF H BvK | 1.850702e+01 | 1.083548e+03 | 3.118262e-05 |
| UHF Li BvK automatic RI | 9.406235e+02 | 3.029823e+05 | 1.019049e-03 |
| PBE0 Be BvK automatic RI | 1.175779e+02 | 3.787279e+04 | 1.273811e-04 |
| PBE0 Li strict | 2.351559e+02 | 7.574554e+04 | 0.000000e+00 |
| compact RHF He strict | 8.807426e-12 | 2.967876e-10 | 0.000000e+00 |
| compact RHF He BvK | 8.930984e-12 | 3.009511e-10 | 1.235574e-13 |

The compact source passes both 1e-8 targets with fixed BvK included. The
ordinary source bounds remain loose. No single whole-method error target,
useful diffuse-source accuracy or production speedup is inferred.

### Whole-job cost

Three fresh processes each perform one cold job and three repeat jobs per
profile/control, using one numerical thread and a 4-GiB queue reservation.
Reported cold times are process medians; repeated times are medians over
nine samples. Each profile keeps its physical source fixed.

| Profile | Exchange admission | Cold / s | Repeated / s | Repeated CPU / s | Native plan / MiB |
| --- | --- | ---: | ---: | ---: | ---: |
| ordinary-rhf | disabled | 0.1955 | 0.1446 | 0.1391 | 828.074 |
| ordinary-rhf | enabled | 0.1467 | 0.1475 | 0.1415 | 852.550 |
| global-pbe0 | disabled | 0.1351 | 0.1345 | 0.1288 | 816.609 |
| global-pbe0 | enabled | 0.1343 | 0.1349 | 0.1294 | 841.084 |
| compact-rhf | disabled | 2.3382 | 2.3351 | 2.3281 | 217.227 |
| compact-rhf | enabled | 2.3770 | 2.3881 | 2.3804 | 256.156 |

Maximum process RSS was 252.81 MiB. RSS includes Python,
native libraries, all profiles and allocator retention; it is not the
count-only native method plan. Costs are measured complete-job overhead,
with no speedup claim. The fixed-BvK bound excludes numerical error in xi.
