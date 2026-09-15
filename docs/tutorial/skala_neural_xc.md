---
myst:
  html_meta:
    "description": "Run Microsoft SKALA-1.1 in vibe-qc: install the optional CPU runtime, complete a molecular RKS calculation, add D3(BJ), and preserve model and grid provenance."
    "og:title": "vibe-qc tutorial - Microsoft SKALA-1.1 neural XC"
---

(skala_neural_xc_tutorial)=

# Microsoft SKALA-1.1 neural XC: a reproducible molecular calculation

Microsoft SKALA-1.1 is a learned exchange-correlation functional evaluated
from the complete numerical integration grid. In this tutorial you will run a
small fixed-geometry molecular calculation, add the checkpoint-recommended
D3(BJ) correction, and preserve enough provenance to repeat the calculation.
The [SKALA user guide](../user_guide/skala.md) is the complete reference for
the model protocol, aliases, grid contract, cache, and route matrix.

```{important}
This worked calculation covers fixed-geometry molecular RKS. Molecular UKS
and ROKS single points are also available. Gradients, geometry optimization,
Hessians, response properties, and Newton/TRAH paths are unsupported and fail
closed. Periodic SKALA is experimental and is not used for the numerical
workflow on this page.
```

## Install and preflight

Use a compatible Linux CPU environment with Python 3.11 through 3.13. The
runtime window validated by this adapter is PyTorch 2.12 or 2.13:

```sh
python3.13 -m venv .venv-skala
.venv-skala/bin/pip install -e '.[skala]'
```

In-process evaluation is not currently supported on macOS because the native
core and current PyTorch wheels can load conflicting OpenMP runtimes. Python
3.14 also lies outside the adapter's validated window: a PyTorch 2.13 wheel is
available, but the combined native-core, Torch, and OpenMP runtime has not yet
passed the acceptance oracle. Provenance inspection and dry runs remain safe
on both because neither imports PyTorch.

The first real evaluation downloads the immutable checkpoint if it is absent,
then verifies its SHA-256 digest before TorchScript deserialization. To
prefetch it on a connected login node without loading PyTorch:

```python
from vibeqc import ensure_skala_model

path = ensure_skala_model()
print(path)
```

Copy the complete XDG cache tree to an offline compute node if needed. See
[installation and cache details](../user_guide/skala.md#install-the-optional-runtime)
for paths and integrity behavior.

## Why SKALA is different from a libxc functional

A conventional LDA, GGA, or meta-GGA evaluates an energy density point by
point. SKALA consumes the spin densities, their Cartesian gradients,
kinetic-energy densities, coordinates, weights, and atom packing for the
complete grid. Its neural inference is therefore nonlocal over that grid.

The quadrature enters the model features and is therefore part of a
reproducible inference protocol, not merely a transparent accuracy knob.
Microsoft's implementation permits configurable grids, and the training data
used more than one grid construction. vibe-qc deliberately pins its accepted
profile to PySCF 2.14 level 3, matching Microsoft's default PySCF inference
and benchmark protocol. Do not set a different `grid_level` or edit individual
radial and angular controls for a vibe-qc SKALA run.

## Run the molecular single point

The following H2 calculation uses def2-SVP and coordinates in bohr. The
companion script is
[`examples/molecular/input-h2-rks-skala.py`](../../examples/molecular/input-h2-rks-skala.py).

```python
from pathlib import Path

import vibeqc as vq

h2 = vq.Molecule(
    [
        vq.Atom(1, [0.0, 0.0, -0.7]),
        vq.Atom(1, [0.0, 0.0, 0.7]),
    ]
)

result = vq.run_job(
    h2,
    basis="def2-svp",
    method="rks",
    functional="skala-1.1",
    output=Path("h2-skala"),
    progress=False,
    record_hostname=False,
)

print(f"converged: {result.converged}")
print(f"E(SKALA) = {result.energy:.10f} Ha")
```

A successful run ends with stable lines of this form:

```text
converged: True
E(SKALA) = ... Ha
```

The exact total is intentionally not printed here as reference data. Treat an
energy as a benchmark only when the geometry, basis, checkpoint, grid, and an
independent oracle are all pinned.

## Read the result

`result.converged` tells you whether the SCF met its convergence criteria.
`result.energy` is the bare self-consistent electronic plus nuclear energy for
the SKALA XC functional. Inspect the generated artifact family as well:

- `h2-skala.out` contains the SCF trace and energy decomposition;
- `h2-skala.system` records the immutable checkpoint, grid, runtime, and job
  status;
- `h2-skala.bibtex` and `h2-skala.references` contain the routed citations;
- `h2-skala.molden`, `h2-skala.xyz`, and `h2-skala.qvf` preserve orbitals and
  structure for downstream inspection.

The checkpoint identity in `.system` matters as much as the functional name.
Never replace the model file while keeping an old provenance record.

## Add the recommended D3(BJ) correction

The TorchScript artifact provides XC energy and potential only. Its metadata
records the `b3lyp5` D3(BJ) parameters as the expected molecular dispersion
setting, but vibe-qc does not add them silently. Request the correction
explicitly:

```python
result_d3 = vq.run_job(
    h2,
    basis="def2-svp",
    method="rks",
    functional="skala-1.1",
    dispersion="b3lyp5",
    output=Path("h2-skala-d3"),
    progress=False,
    record_hostname=False,
)

print(f"E(SCF)   = {result_d3.energy:.10f} Ha")
print(f"E(D3-BJ) = {result_d3.e_dispersion:+.10f} Ha")
print(f"E(total) = {result_d3.energy_total:.10f} Ha")
```

The output has this stable meaning:

```text
E(SCF)   = ... Ha
E(D3-BJ) = ... Ha
E(total) = ... Ha
```

`energy` remains the bare SCF value, `e_dispersion` is the additive geometry
correction, and `energy_total` is the corrected molecular total. Record
whether D3(BJ) was requested whenever energies are compared.

## Preserve a reproducible record

The immutable model identity is available without Torch or network access:

```python
from vibeqc import skala_model_provenance

model = skala_model_provenance()
for key in ("functional", "revision", "sha256", "expected_d3_settings"):
    print(f"{key}: {model[key]}")
```

The values are pinned by the installed adapter:

```text
functional: skala-1.1
revision: 99b5ed87e5f69d9216e1f9e30148b922eaea1241
sha256: 7f3e8622e1eb520ccd88a55464c3e359ac4d7e5ccbd1fb77a26afa1e1c20a5cd
expected_d3_settings: b3lyp5
```

Before allocating resources for a larger calculation, create the output plan:

```python
vq.run_job(
    h2,
    basis="def2-svp",
    method="rks",
    functional="skala-1.1",
    output="h2-skala-plan",
    dry_run=True,
    record_hostname=False,
)
```

This writes `h2-skala-plan.system` with dry-run status and SKALA provenance.
It performs no checkpoint download, PyTorch import, or SCF. Memory estimation
can be enabled through the normal dry-run controls described in
[Memory budget](../user_guide/memory.md).

## What to test before trusting the result

A converged SCF proves numerical self-consistency, not chemical accuracy.
Before using SKALA for a scientific claim:

1. converge the Gaussian basis and ordinary SCF thresholds;
2. compare the intended observable with an appropriate independent method or
   experimental reference;
3. verify that conclusions do not depend on adding or omitting D3(BJ);
4. preserve `.system`, `.out`, `.bibtex`, and `.references` together; and
5. treat transition-metal, heavy-element, and all periodic results as
   exploratory unless they have system-specific validation.

Molecular open-shell syntax is documented in
[Open-shell UKS and ROKS](../user_guide/skala.md#open-shell-uks-and-roks).
The separate
[`examples/periodic/input-h2-cell-rks-skala-gdf.py`](../../examples/periodic/input-h2-cell-rks-skala-gdf.py)
is an experimental API smoke calculation, not a periodic accuracy benchmark.

## Where this workflow stops

Use SKALA only for fixed-geometry single points in the current release. Do not
request analytic gradients, geometry or cell optimization, stress, Hessians,
TDDFT, or Newton/TRAH response. Those combinations raise an error instead of
silently changing the functional or dropping a derivative.

Periodic GPW and GAPW are available in a deliberately narrow experimental
envelope: 3D GPW RKS at Gamma or on a complete unreduced multi-k mesh, and 3D
GAPW RKS at Gamma. The neutral real-Gamma route and the four-centre Chi-CCM
backend also accept 3D RKS/UKS. Literal four-centre WSSC RKS/UKS is available
through `run_periodic_job(method="aiccm", variant="four-center")` and the
low-level CCM APIs. Open-shell GPW/GAPW, GAPW multi-k, every 2D or periodic ROKS
route, ECP-bearing GDF/RIJCOSX/GPW/GAPW/AICCM jobs, fitted Chi backends,
positive electronic smearing on every periodic external-XC route, DFT+U AICCM
jobs, and external-provider literal-WSSC or Chi hybrids remain fail-closed.
BIPOLE also requires its corrected Ewald gauge and a complete Monkhorst-Pack
mesh or valid expandable IBZ; the legacy gauge and ad-hoc multi-k lists are
rejected.
The full support and failure matrix is maintained in the
[SKALA user guide](../user_guide/skala.md#unsupported-operations).

## References

The functional is described by
[Luise et al., 2025](https://doi.org/10.48550/arXiv.2506.14665). The
independent molecular implementation analysis is
[Pöschel et al., 2026](https://doi.org/10.48550/arXiv.2608.19033).
vibe-qc's central citation database emits both references automatically for a
SKALA job, together with the grid references required by the calculation.

## Next

- [SKALA user guide](../user_guide/skala.md), complete operational reference.
- [Functionals](../user_guide/functionals.md), libxc and generic external-XC
  interfaces.
- [ROHF and ROKS](../user_guide/rohf.md), choosing an open-shell treatment.
- [Periodic methods](../user_guide/periodic_methods.md), experimental route
  controls and convergence.
- [Automatic citations](auto_citations.md), preserving method
  references.
- [SKALA redistribution audit](../license.md#microsoft-skala-11-optional-neural-xc-functional-skala-extra),
  reviewed licenses, notices, and patent caveat.
