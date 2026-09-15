"""S22 reference interaction energies (kcal/mol).

Extracted from the Psi4 ``S22.py`` database file (LGPL v3),
which itself sources the values from:

  * S220 — Jurečka, Šponer, Černý, Hobza, *PCCP* **8**, 1985 (2006).
  * S22A — Takatani, Hohenstein, Malagoli, Marshall, Sherrill,
    *JCP* **132**, 144104 (2010). First revision (CCSD(T)/CBS, frozen-core).
  * S22B — Marshall, Burns, Sherrill, *JCP* **135**, 194102 (2011).
    Second revision (CCSD(T)/CBS with corrections); current standard.
"""

from __future__ import annotations

S220: dict[int, float] = {
     1:   -3.170,
     2:   -5.020,
     3:  -18.610,
     4:  -15.960,
     5:  -20.650,
     6:  -16.710,
     7:  -16.370,
     8:   -0.530,
     9:   -1.510,
    10:   -1.500,
    11:   -2.730,
    12:   -4.420,
    13:  -10.120,
    14:   -5.220,
    15:  -12.230,
    16:   -1.530,
    17:   -3.280,
    18:   -2.350,
    19:   -4.460,
    20:   -2.740,
    21:   -5.730,
    22:   -7.050,
}

S22A: dict[int, float] = {
     1:   -3.150,
     2:   -5.070,
     3:  -18.810,
     4:  -16.110,
     5:  -20.690,
     6:  -17.000,
     7:  -16.740,
     8:   -0.530,
     9:   -1.480,
    10:   -1.450,
    11:   -2.620,
    12:   -4.200,
    13:   -9.740,
    14:   -4.590,
    15:  -11.660,
    16:   -1.500,
    17:   -3.290,
    18:   -2.320,
    19:   -4.550,
    20:   -2.710,
    21:   -5.620,
    22:   -7.090,
}

S22B: dict[int, float] = {
     1:   -3.133,
     2:   -4.989,
     3:  -18.753,
     4:  -16.062,
     5:  -20.641,
     6:  -16.934,
     7:  -16.660,
     8:   -0.527,
     9:   -1.472,
    10:   -1.448,
    11:   -2.654,
    12:   -4.255,
    13:   -9.805,
    14:   -4.524,
    15:  -11.730,
    16:   -1.496,
    17:   -3.275,
    18:   -2.312,
    19:   -4.541,
    20:   -2.717,
    21:   -5.627,
    22:   -7.097,
}
