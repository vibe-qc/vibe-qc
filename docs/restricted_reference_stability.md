# Restricted reference stability and correlated calculations

A real closed-shell RHF or RKS stationary point has distinct internal and
external real orbital-rotation channels. Equal alpha and beta rotations
remain within the restricted singlet space. Opposite rotations break the
spin restriction and probe the real UHF or UKS space. The normalized maps
are `(kappa, +kappa)/sqrt(2)` and `(kappa, -kappa)/sqrt(2)`.

This is the transformation in Seeger and Pople, J. Chem. Phys. 66, 3045
(1977), DOI [10.1063/1.434318](https://doi.org/10.1063/1.434318), Section III,
Eqs. (34)-(36). The KS analogue is Bauernschmitt and Ahlrichs, J. Chem. Phys.
104, 9047 (1996), DOI [10.1063/1.471637](https://doi.org/10.1063/1.471637).
The current implementation tests these real channels only. It does not
certify stability to complex orbital rotations.

Both channels have independent eigenvalue, convergence and stop records.
The default aggregate verdict remains conservative: a detected instability
or a missing convergence verdict makes `usable` false. A real external
instability is labelled as external; it does not establish an internal
restricted saddle or an excited electronic state. Historical archives
without channel records cannot be reclassified automatically.

For a closed-shell CCSD, CCSD(T), DLPNO-CCSD or DLPNO-CCSD(T) single point,
`run_job(reference_contract="real_restricted_cc", ...)` requests a separate
constrained-reference assessment. It requires converged SCF, converged tests
for both real channels, and a stable internal channel. `reference_usable`
then describes admission of that real restricted reference. The external
verdict and the conservative `usable` field remain unchanged. This option
does not follow the unstable mode, change the orbitals, suppress a test or
claim an unrestricted ground-state minimum. It is refused for geometry or
Hessian work and for methods outside the listed closed-shell CC family.

Reference admission does not certify the correlated answer. Check amplitude
convergence, the triples definition, frozen core, basis and local truncation
errors separately. Coupled-cluster energies are not variational HF energies;
an external HF instability is a reason to examine reference dependence and
possible strong correlation, not proof of an internally unstable RHF
solution or an accuracy guarantee for its correlated energy. An independent
matched-method comparison and appropriate reference diagnostics remain
necessary for a quantitative application. Never silently substitute the
constrained-reference assessment for a general SCF stability requirement.
