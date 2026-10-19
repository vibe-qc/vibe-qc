"""Geometries and pinned MSINDO-SECCM values for the semi-empirical article.

One home for the systems the article's equation-of-state and chain studies
run, so the release pins (``test_release_msindo_cyclic_cluster_pins.py``)
and the slower uniqueness checks (``test_msindo_ccm.py``) cannot drift apart.

The fcc clusters are built exactly as the article's builders build them: the
primitive fcc cell replicated ``n x n x n``, with translations ``n`` times the
primitive vectors. Every energy is the audited stable state of the native
kernel at max_iter=400, conv_tol=1e-9 under the stated SCF state policy
(issue #249). Each 2x2x2 entry states its largest net atomic charge. For bare
diamond 2x2x2 the neutral state the reference program converges to is a
saddle of the WS-truncated bare Coulomb model (lowest orbital Hessian
eigenvalue -1.76 Ha): the default "reference" policy returns it flagged, and
the "lowest" policy descends to the model's minimum, which alternates
+/-2.95 e on the two sublattices 2.75 Ha lower. The Madelung-embedded
cluster and every 3x3x3 cluster are stable neutral minima under both
policies.
"""

from __future__ import annotations

from itertools import product

import numpy as np

from vibeqc.molecule import ANGSTROM_TO_BOHR

A_MGO = 4.212
A_DIAMOND = 3.567

# MgO B1 2x2x2 bare: the article's campaign value (vq job 60a32b0a8630,
# vibe-qc 0.15.138), reproduced by the current kernel to 2.5e-12 Ha.
MGO_222_BARE_ARTICLE_ENERGY = -137.58937495890854

# (system, madelung, stability policy, total energy / Ha,
#  max |net charge| / e, stability restarts, flagged internal instability)
ARTICLE_222_PINS = (
    ("MgO", False, "reference", MGO_222_BARE_ARTICLE_ENERGY, 1.5298, 0, False),
    ("MgO", True, "reference", -135.2508239641, 1.0811, 0, False),
    ("diamond", True, "reference", -94.5798397420, 0.0, 0, False),
    ("diamond", False, "reference", -94.5798397420, 0.0, 0, True),
    ("diamond", False, "lowest", -97.3280883020, 2.9538, 1, False),
)


def pin_id(entry):
    name, madelung, stability = entry[:3]
    return f"{name}-{'madelung' if madelung else 'bare'}-{stability}"
# (system, madelung, total energy / Ha, stability restarts)
ARTICLE_333_PINS = (
    ("MgO", False, -454.3606698958, 0),
    ("MgO", True, -455.8564864090, 0),
    ("diamond", False, -318.0543576049, 0),
    ("diamond", True, -318.0543576048, 0),
)

# The #249 verifier's C3 chain: atoms at 0, 1.283 and 2.56 A, T = 3.84 A,
# atom 0 displaced transversely by dy bohr. Before the stability audit ran at
# every Hcore gap, dy = 3e-3 converged 16.6 mHa above this lower branch.
# The old DIIS branch is retained as a separate state-selection control.
# A pi/2 rotation of each atomic p_y/p_z pair followed by explicit-density
# continuation recovers it from the lower branch. It is not the lowest state.
# Under "reference", dy = 3e-3 returns the still higher Hcore/DIIS saddle.
BENT_CHAIN_SWEEP_BOHR = (0.0, 1e-4, 1e-3, 3e-3, 1e-2, 5e-2)
BENT_CHAIN_REFERENCE_3E3 = {False: -17.5358307608, True: -17.5391858924}
BENT_CHAIN_LEGACY_ENERGIES = {
    False: (-17.5524245047576, -17.5524245051400, -17.5524245415673,
            -17.5524248360463, -17.5524281858936, -17.5525166138634),
    True: (-17.5524422640831, -17.5524422644633, -17.5524422994900,
           -17.5524425827499, -17.5524458053179, -17.5525310926542),
}


# Lower branch selected by the variational frontier and orbital continuation.
# Both native/Python paths agree; the old branch remains tested above.
BENT_CHAIN_ENERGIES = {
    False: (-17.5524245047576, -17.5524245052173, -17.5524245593964,
            -17.5524249965500, -17.5524299712417, -17.5525625639409),
    True: (-17.5524422640831, -17.5524422645538, -17.5524423212524,
           -17.5524427786342, -17.5524479842477, -17.5525871942850),
}


def fcc_cluster(a, basis, n):
    """``(Z, coords / A, translations / A)`` of a primitive fcc cell replicated
    ``n x n x n``; ``basis`` lists ``(Z, position / A)`` in the primitive cell."""
    prim = [[0.0, a / 2, a / 2], [a / 2, 0.0, a / 2], [a / 2, a / 2, 0.0]]
    Z, coords = [], []
    for i, j, k in product(range(n), repeat=3):
        offset = [i * prim[0][c] + j * prim[1][c] + k * prim[2][c] for c in range(3)]
        for z, site in basis:
            Z.append(z)
            coords.append([site[c] + offset[c] for c in range(3)])
    return Z, np.array(coords), [[n * v for v in vector] for vector in prim]


def article_cluster(name, n):
    """The article's MgO (B1) or diamond cluster, replicated ``n x n x n``."""
    if name == "MgO":
        return fcc_cluster(A_MGO, [(12, [0.0, 0.0, 0.0]), (8, [A_MGO / 2] * 3)], n)
    if name == "diamond":
        return fcc_cluster(
            A_DIAMOND, [(6, [0.0, 0.0, 0.0]), (6, [A_DIAMOND / 4] * 3)], n
        )
    raise ValueError(f"unknown article system {name!r}")


def bent_c3_chain(dy_bohr, *, noise_angstrom=0.0):
    """``(coords / A, translations / A)`` of the bent C3 chain; a fixed,
    non-symmetric noise pattern of amplitude ``noise_angstrom`` is added."""
    coords = np.array([[0.0, dy_bohr / ANGSTROM_TO_BOHR, 0.0],
                       [1.283, 0.0, 0.0], [2.56, 0.0, 0.0]])
    if noise_angstrom:
        pattern = np.array([[1.0, -2.0, 3.0], [-3.0, 1.0, -1.0], [2.0, 2.0, -2.0]])
        coords = coords + noise_angstrom * pattern / 3.0
    return coords, [[3.84, 0.0, 0.0]]
