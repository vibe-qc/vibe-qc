### Changed

- The opt-in bounded UKS DIIS capture now records the default `EDIIS_DIIS` accelerator: each cycle's EDIIS history, quadratic model, simplex coefficients, interpolated energy and anti-replay erasures alongside the DIIS solve, plus the switch metric and the branch that reached the SCF. Refusals name the violated bound, and the basis limit rises from 64 to 128 functions. Ordinary SCF trajectories are unchanged.
