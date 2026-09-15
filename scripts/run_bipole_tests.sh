#!/bin/sh
# Run the BIPOLE far-field test suite.
#
# Sets OMP/OpenBLAS/MKL thread counts to 1 on macOS to prevent
# nested-threading crashes in the vendored libomp.  The pytest
# conftest.py also does this automatically at configure time;
# this script is an extra safety net for direct test invocation.
#
# Usage:
#   .venv/bin/sh scripts/run_bipole_tests.sh
#   .venv/bin/sh scripts/run_bipole_tests.sh --slow   # include slow tests
#   .venv/bin/sh scripts/run_bipole_tests.sh -k "gradient"  # filter

set -eu

cd "$(dirname "$0")/.."

# macOS thread safety: prevent nested OpenMP crashes.
if [ "$(uname -s)" = "Darwin" ]; then
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
    export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
    export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
fi

# AO-pair FT backend (Python reference path, deterministic).
export VIBEQC_AOPAIR_FT_BACKEND="${VIBEQC_AOPAIR_FT_BACKEND:-python}"

# Core BIPOLE test suite (fast tests).
FAST_TESTS="
tests/test_bipole_contractor_parity.py
tests/test_bipole_gradient_parity.py
tests/test_bipole_cpp_wiring.py
tests/test_bipole_fd_gradient.py
tests/test_bipole_parity.py
tests/test_bipole_far_field_scf.py
tests/test_bipole_multik_tolinteg.py
tests/test_bipole_penetration_dispatch.py
tests/test_bipole_exact_vs_far_field_energy.py
tests/test_bipole_production_guards.py
tests/test_bipole_spherical_moment_buffer.py
tests/test_bipole_symmetry_dispatch.py
tests/test_bipole_symmetry_fock.py
tests/test_bipole_sph_to_cart.py
tests/test_bipole_pair_moments_parity.py
tests/test_bipole_far_field_convergence.py
tests/test_bipole_cell_moments.py
tests/test_bipole_lattice_self_energy.py
"

# Slow production tests (run with --slow flag).
SLOW_TESTS="
tests/test_bipole_scf_parity.py
tests/test_bipole_production_far_field.py
"

if [ $# -gt 0 ] && [ "$1" = "--slow" ]; then
    shift
    exec .venv/bin/python -m pytest $FAST_TESTS $SLOW_TESTS -v --timeout=600 "$@"
else
    exec .venv/bin/python -m pytest $FAST_TESTS -v --timeout=120 "$@"
fi
