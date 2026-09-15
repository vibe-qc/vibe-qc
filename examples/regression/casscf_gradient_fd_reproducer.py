"""CASSCF analytic-gradient reproducer — H2/6-31G CASSCF(2e,2o) at R=1.5 bohr.

The full-energy central finite difference is the adjudicator.  Tests all
three ``compute_wz`` settings:
  - default (the analytic gradient -- exact for a variational CASSCF)
  - compute_wz=True (a no-op alias of the analytic gradient since the
    experimental W^z correction was retired, GitLab #516)
  - numerical (full CASSCF energy FD)

All three must match the FD adjudicator within tolerance; the analytic
path is the production gradient.  An earlier revision of this file
described the analytic path as "87 %" and expected a ~74 % overshoot away
from equilibrium -- that was the since-fixed energy-weighted-density
defect, not a property of the analytic gradient.  Exit code 0 when every
path passes.
"""

from __future__ import annotations

import sys

import vibeqc as vq
from vibeqc.runner import _run_single_point
from vibeqc.solvers import CASSCFOptions

R_BOHR = 1.5
FD_STEP_BOHR = 1.0e-3
TOLERANCE_HA_PER_BOHR = 1.0e-4


def _molecule(distance: float) -> vq.Molecule:
    return vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, distance])])


def _run(distance: float, *, compute_wz=False):
    molecule = _molecule(distance)
    basis = vq.BasisSet(molecule, "6-31g")
    return _run_single_point(
        "casscf",
        molecule,
        basis,
        functional=None,
        active_space=(2, 2),
        casscf_options=CASSCFOptions(compute_wz=compute_wz),
    )


def main() -> int:
    default = _run(R_BOHR)
    with_wz = _run(R_BOHR, compute_wz=True)
    numerical = _run(R_BOHR, compute_wz="numerical")
    e_plus = float(_run(R_BOHR + FD_STEP_BOHR).energy)
    e_minus = float(_run(R_BOHR - FD_STEP_BOHR).energy)
    finite_difference = (e_plus - e_minus) / (2.0 * FD_STEP_BOHR)

    gradient_default = float(default.gradient[1, 2])
    gradient_wz = float(with_wz.gradient[1, 2])
    gradient_numerical = float(numerical.gradient[1, 2])
    delta_default = gradient_default - finite_difference
    delta_wz = gradient_wz - finite_difference
    delta_numerical = gradient_numerical - finite_difference

    print("CASSCF(2e,2o) H2/6-31G gradient parity")
    print(f"  energy                 = {float(default.energy):+.12f} Ha")
    print(f"  full-energy FD         = {finite_difference:+.12f} Ha/bohr")
    print(
        f"  analytic default       = {gradient_default:+.12f} Ha/bohr  (delta {delta_default:+.12f})"
    )
    print(
        f"  analytic compute_wz    = {gradient_wz:+.12f} Ha/bohr  (delta {delta_wz:+.12f})"
    )
    print(
        f"  numerical FD           = {gradient_numerical:+.12f} Ha/bohr  (delta {delta_numerical:+.12f})"
    )

    # Every path must match the full-energy FD: the analytic gradient is
    # the complete derivative of the variational CASSCF energy (#516), the
    # compute_wz=True alias returns the same vector, and the numerical
    # path IS a finite difference.
    numerical_ok = abs(delta_numerical) <= TOLERANCE_HA_PER_BOHR
    default_ok = abs(delta_default) <= TOLERANCE_HA_PER_BOHR
    wz_ok = abs(delta_wz) <= TOLERANCE_HA_PER_BOHR
    # A linear molecule on z: the transverse components must vanish on
    # every path (the retired W^z term violated this with p/d functions).
    transverse = max(
        float(abs(g.gradient[:, :2]).max()) for g in (default, with_wz, numerical)
    )
    transverse_ok = transverse <= 1.0e-10

    failures = [
        name
        for name, ok in (
            ("analytic", default_ok),
            ("compute_wz=True", wz_ok),
            ("numerical", numerical_ok),
            ("transverse", transverse_ok),
        )
        if not ok
    ]
    verdict = "PASS" if not failures else "FAIL (" + ", ".join(failures) + ")"
    print(f"  max |g_x|, |g_y|       = {transverse:.2e} Ha/bohr")
    print(f"  verdict                = {verdict}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
