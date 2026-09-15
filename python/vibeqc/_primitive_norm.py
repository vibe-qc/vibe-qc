"""libint's primitive normalization factor -- the one shared definition.

Four vibe-qc modules need the factor by which libint2 scales a stored
contraction coefficient: the QVF writer, the molden writer, the basis
filter, and the BIPOLE external-pole primitive splitter. Each previously
carried its own copy with its own paragraph of explanation; the 2026-08-05
AO-convention arc (handovers/HANDOVER_AO_CONVENTION.md) showed how long a
divergence between such near-copies can survive unnoticed, so the
definition now lives here once.

Two copies outside this package are independent *on purpose*:
``qvf-writer/python/qvf_writer.py`` is a dependency-light standalone
producer reference that must stay vendorable without vibe-qc, and
vibe-view's ``renderers/wavefunction.py`` is an independent QVF consumer
(implementing the spec, not calling vibe-qc, is the point). Both are
pinned against QVF spec Appendix A.1 by
``tests/test_ao_convention_invariants.py``.
"""

from __future__ import annotations

import math


def libint_primitive_norm(alpha: float, l: int) -> float:
    """The axial norm ``N(alpha, l)`` -- QVF spec Appendix A.1's ``N_i``.

    For the *axial* Cartesian component ``(l, 0, 0)`` of a shell,
    ``g(r) = (x - Ax)^l exp(-alpha r^2)``, the unit-normalization factor
    is::

        N = (2 alpha / pi)^(3/4) . (4 alpha)^(l/2) / sqrt((2l - 1)!!)

    libint applies this ONE factor, derived from the total ``l``, to the
    whole shell: its stored contraction coefficients are ``c_i . N_i``,
    multiplying bare monomials. The factor does NOT vary per Cartesian
    component, so a mixed component such as ``d_xy`` is deliberately not
    unit-normalized -- on a single Cartesian d shell ``compute_overlap``
    gives ``<xy|xy> = 1/3`` against ``<xx|xx> = 1``. (A genuinely
    unit-normalized Cartesian Gaussian would divide by
    ``sqrt((2lx-1)!!(2ly-1)!!(2lz-1)!!)`` instead of ``sqrt((2l-1)!!)``.)
    For spherical shells and the axial Cartesian components the two
    conventions coincide. Consumers must not stack a per-component
    correction on top of this factor -- doing so is exactly the defect the
    2026-08-05 arc removed from both AO evaluators and the COSX kernel.

    Dividing libint's stored coefficients by this factor recovers the raw
    basis-set-file values (verified against the published H STO-3G
    ``.g94``: 0.15432897, 0.53532814, 0.44463454 to four decimals), which
    is what the QVF and molden formats require on the way out. The
    convention matches libint2 / libcint / Gaussian / Molpro / Turbomole,
    and is pinned behaviourally against a raw libint engine in
    ``tests/test_ao_convention_invariants.py``.
    """
    radial = (2.0 * alpha / math.pi) ** 0.75
    angular = (4.0 * alpha) ** (l / 2.0)
    df = 1.0
    for k in range(1, 2 * l, 2):
        df *= float(k)
    return radial * angular / math.sqrt(df)
