"""Parity tests for the MSINDO STO integral kernel.

Reference values are read off a reference MSINDO build's PRINTOPTS=GMAT,SMAT
dumps for H2 and HF (examples/regression/msindo/).
"""

import math

import pytest
from vibeqc.semiempirical.methods import msindo_integrals as K
from vibeqc.semiempirical.methods.msindo import (
    ANGSTROM_TO_BOHR,
    MSINDO_BOHR_ANGSTROM,
)

A2B = ANGSTROM_TO_BOHR


def test_geometry_conversion_matches_reference_const_f():
    """Geometry uses the value recomputed inside the MSINDO executable."""
    assert MSINDO_BOHR_ANGSTROM == pytest.approx(
        0.5291772575069162, abs=1e-16
    )
    assert ANGSTROM_TO_BOHR == pytest.approx(1.8897259582001786, abs=1e-15)


def test_h2_integrals():
    z, R = 1.1576, 0.741 * A2B
    assert K.s2int(1, 0, 0, z, 1, 0, 0, z, R) == pytest.approx(0.69134, abs=1e-4)
    assert K.c2int(1, 0, 0, z, 1, 0, 0, z, R) == pytest.approx(0.54918, abs=1e-4)
    assert K.v2int(1, 0, 0, z, R) == pytest.approx(0.64096, abs=1e-4)


def test_hf_overlaps_with_rotation_axis():
    R = 0.917 * A2B
    zH, zFs, zFp = 1.1576, 2.4974, 2.3510
    # F 2s with H 1s (oracle Smat 0.47270)
    assert K.s2int(2, 0, 0, zFs, 1, 0, 0, zH, R) == pytest.approx(0.47270, abs=1e-4)
    # F 2p-sigma with H 1s along the bond (oracle Smat 0.37419)
    assert K.s2int(2, 1, 0, zFp, 1, 0, 0, zH, R) == pytest.approx(0.37419, abs=1e-4)


def test_radint_one_center_F():
    # F GPP = F0PP + 4/25 F2PP with zeta = MUPE(F) = 2.2465 (oracle 0.87929)
    z = 2.2465
    f0pp = K.radint(0, 2, 2, 2, 2, z, z, z, z)
    f2pp = K.radint(2, 2, 2, 2, 2, z, z, z, z)
    assert f0pp + 4.0 / 25.0 * f2pp == pytest.approx(0.87929, abs=1e-4)


def test_harmtr_rotation():
    # Local orbital order: 1=s, 2=p_sigma, 3,4=p_pi.  T[global_row, local_col].
    # Bond along +z: local p-sigma (col 1) -> global pz (row 3).
    Tz = K.harmtr(2, [0.0, 0.0, 1.0])
    assert Tz[0, 0] == 1.0
    assert Tz[3, 1] == pytest.approx(1.0)
    # Bond along +x: local p-sigma (col 1) -> global px (row 1).
    Tx = K.harmtr(2, [1.0, 0.0, 0.0])
    assert Tx[1, 1] == pytest.approx(1.0)


def test_cpp_analytic_gradient_matches_fd():
    """C++ analytic nuclear gradient matches FD for H2, CO, and H2O."""
    from pathlib import Path

    import numpy as np
    from vibeqc._vibeqc_core.semiempirical.indo import (
        gradient_analytic,
        load_params_from_json,
    )
    from vibeqc.semiempirical.methods.msindo import msindo_gradient_fd

    json_path = (
        Path(__file__).resolve().parents[1]
        / "python"
        / "vibeqc"
        / "semiempirical"
        / "methods"
        / "msindo_params.json"
    )
    params = load_params_from_json(json_path.read_text())

    for name, Z, C in [
        ("H2", [1, 1], [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
        ("CO", [6, 8], [[0.0, 0.0, 0.0], [1.13, 0.0, 0.0]]),
        ("H2O", [8, 1, 1], [[0, 0, 0], [0.95, 0.0, 0.10], [-0.20, 0.92, 0.0]]),
    ]:
        g_cpp = gradient_analytic(Z, C, params)
        g_fd = msindo_gradient_fd(Z, C, step=1e-4)
        max_err = max(abs(np.array(g_cpp) - g_fd).flatten())
        assert max_err < 1e-7, (
            f"{name} C++ analytic gradient differs from FD: max_err={max_err:.1e}"
        )


def test_cpp_integral_derivatives_match_fd():
    """C++ ds2int, dc2int, dv2int match finite differences of s2int, c2int, v2int."""
    import numpy as np
    from vibeqc._vibeqc_core.semiempirical.indo import (
        c2int,
        dc2int,
        ds2int,
        dv2int,
        s2int,
        v2int,
    )

    h = 1e-5
    for R in [0.74 * A2B, 1.5, 2.5]:
        for z in [1.0, 1.625]:
            # overlap derivative
            fd = (
                s2int(1, 0, 0, z, 1, 0, 0, z, R + h)
                - s2int(1, 0, 0, z, 1, 0, 0, z, R - h)
            ) / (2 * h)
            an = ds2int(1, 0, 0, z, 1, 0, 0, z, R)
            assert abs(fd - an) < 1e-8, f"ds2int R={R} z={z}: {fd} vs {an}"
            # Coulomb derivative
            fd_c = (
                c2int(1, 0, 0, z, 1, 0, 0, z, R + h)
                - c2int(1, 0, 0, z, 1, 0, 0, z, R - h)
            ) / (2 * h)
            an_c = dc2int(1, 0, 0, z, 1, 0, 0, z, R)
            assert abs(fd_c - an_c) < 1e-8, f"dc2int R={R} z={z}: {fd_c} vs {an_c}"
            # nuclear attraction derivative
            fd_v = (v2int(1, 0, 0, z, R + h) - v2int(1, 0, 0, z, R - h)) / (2 * h)
            an_v = dv2int(1, 0, 0, z, R)
            assert abs(fd_v - an_v) < 1e-8, f"dv2int R={R} z={z}: {fd_v} vs {an_v}"
