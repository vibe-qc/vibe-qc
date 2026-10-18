# HCO beta pair-space regression

`dlpno_hco_beta_pair_spaces.npz` contains three dimensionless binary64 beta
PNO coefficient matrices (`pair_0`, `pair_1`, `pair_2`). They were generated
with vibe-qc at NormalPNO, cc-pVDZ with its default RI auxiliary basis,
chemical frozen core, and a UHF doublet reference for neutral HCO. Geometry
in angstrom: C (0, 0, 0), O (0, 0, 1.18), H (0, 0.929, -0.574).

These are the beta rows of pairs (6, 9), (6, 10), (9, 10) in the local
spin-orbital indexing. Each full matrix has 25 zero alpha rows before these
26 beta rows. Their concatenated union has rank 26. The portable Eigen 3.4
divide-and-conquer SVD returned non-finite left singular vectors for the
QR factor of this union; changing column order avoided the failure.

The regression supplies synthetic spin-blocked factors and amplitudes,
checks finite triples energies, and checks invariance under reversing each
pair's columns together with both amplitude indices. Keeping the measured
spaces avoids depending on SCF roundoff to reproduce the failing matrix.

The same matrices are also valid inputs to the closed-shell local triples
kernel when interpreted as spatial pair spaces with synthetic factors and
amplitudes. That kernel reproduced the failure too. Its regression compares
the original spaces, reversed columns, and an explicitly constructed NumPy
union with amplitudes transformed into that common basis; it is an algebra
test, not a closed-shell calculation on HCO.
