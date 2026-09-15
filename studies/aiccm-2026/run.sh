#!/usr/bin/env bash
# vq launcher for the AICCM-2026 benchmark.
#
# `vq submit <host> -d studies/aiccm-2026/ -- python run_case.py ...` runs against the
# host's *default* python, which does not have vibeqc importable. As of
# 2026-08-01, `--branch` can prepend an interpreter to directory commands, but
# this payload deliberately launches `bash run.sh`; prepending Python would
# execute the shell script as Python. This wrapper therefore resolves the
# selected interpreter, preflights that exact deployment, and dispatches either
# the A or B producer. Set VIBEQC_PYTHON in the private remote job environment
# when the deployment is outside the checkout. The producer completes a
# same-process composite attestation. The final shell remains alive long enough
# to remove its per-job preflight on exit.
#
#     vq submit compute-small -d studies/aiccm-2026/ -- bash run.sh <system> <route> [--basis B]
#     vq submit compute-small -d studies/aiccm-2026/ -- bash run.sh --b <system> <B-route>
#     vq submit --host compute-large --job-name d101-chi-3d-ri -d studies/aiccm-2026/ -- \
#       bash run.sh --comparison-b 3d --expected-source-sha <full-sha> \
#       --vq-host compute-large
#     vq submit compute-small -d studies/aiccm-2026/ -- bash run.sh --case <producer.py> <args...>
set -euo pipefail

case_script="run_case.py"
mpi_ranks=""
mpi_validation_mode=""
mpi_selector=""
if [ "${1:-}" = "--d114-mpi" ]; then
    # Narrow, rank-safe validation producer for the restricted 3-D chi RHF
    # four-center output-cell scheduler. Ordinary campaign producers are not
    # safe to launch under MPI because every rank writes their result file.
    case_script="run_d114_mpi.py"
    mpi_ranks=2
    mpi_validation_mode="d114-rhf"
    mpi_selector="--d114-mpi"
    shift
elif [ "${1:-}" = "--d116-rks-mpi" ]; then
    # The same rank-safe producer and launcher contract validates the D116
    # non-screened restricted-RKS extension with distinct result provenance.
    case_script="run_d114_mpi.py"
    mpi_ranks=2
    mpi_validation_mode="d116-rks-pbe"
    mpi_selector="--d116-rks-mpi"
    shift
elif [ "${1:-}" = "--b" ]; then
    case_script="run_case_b.py"
    shift
elif [ "${1:-}" = "--comparison-b" ]; then
    # Dedicated D101 chi campaign producer. Keep this selector narrow: the
    # producer owns the 3-D-first receipt, D77 source binding, D83/D88 audit,
    # executed zero-mixing check, and D93 absolute-energy suppression.
    case_script="run_case_cmp_b.py"
    shift
elif [ "${1:-}" = "--case" ]; then
    # Generic producer dispatch (e.g. run_case_cmp.py, the neutral fitted-torus
    # Bloch/GDF representation-control producer). Same interpreter resolution
    # + attestation preflight as the A/B producers.
    case_script="${2:?run.sh: --case needs a producer script filename}"
    shift 2
fi

# Select an explicit deployment, this checkout's venv, or a PATH interpreter
# that can import vibeqc. Site-specific paths belong in private operator config.
_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=""
if [ "${VIBEQC_PYTHON+x}" = x ]; then
    if [ -z "$VIBEQC_PYTHON" ] || [ ! -x "$VIBEQC_PYTHON" ]; then
        echo "run.sh: VIBEQC_PYTHON must name an executable interpreter" >&2
        exit 2
    fi
    PY="$VIBEQC_PYTHON"
elif [ -x "$_here/../../.venv/bin/python" ]; then
    PY="$_here/../../.venv/bin/python"
elif command -v python >/dev/null 2>&1 && python -c "import vibeqc" 2>/dev/null; then
    PY="$(command -v python)"
fi
: "${PY:?run.sh: select a vibeqc-enabled interpreter with VIBEQC_PYTHON or activate its environment}"
echo "run.sh: interpreter selected" >&2

# Verify the host NUMERICALLY before producing any number. A symbol probe cannot see
# a stale compiled core, which returns wrong answers rather than failing to import:
# the irreproducible 2026-07-08 c-diamond row illustrates why that failure class
# cannot be excluded without attestation; its specific attribution remains an
# inference. This preflight is cached per clean source/core/host/package/probe
# identity, so a 59-job batch pays the ~25 s full probe once, not 59 times. It is
# not sufficient producer evidence by itself. The producer binds the copied
# payload, always runs a cheap shifted-mesh native/Python canary in its own
# process, and reruns the full v2 checks there if the current core-path SHA256
# changed after preflight. It re-reads identity and payload before serializing
# the result. Set AICCM_SKIP_PROBE=1 to bypass, in which case the producer marks
# the result untrusted and curate.py refuses it. Create a private directory
# atomically: a skipped or failed preflight cannot inherit an older pass, collide
# with another job, or follow a predictable symlink.
ATTESTATION_ROOT="${AICCM_PROBE_TMPDIR:-${TMPDIR:-/tmp}}"
ATTESTATION_DIR="$(mktemp -d "$ATTESTATION_ROOT/aiccm-probe-attestation.XXXXXX")"
ATTESTATION="$ATTESTATION_DIR/attestation.json"
export AICCM_PROBE_ATTESTATION="$ATTESTATION"
trap 'rm -rf "$ATTESTATION_DIR"' EXIT
if [ "${AICCM_SKIP_PROBE:-0}" != "1" ]; then
    if ! "$PY" "$_here/probe_host.py" \
            --emit "$ATTESTATION" \
            --cache-dir "${AICCM_PROBE_CACHE:-$HOME/.cache/aiccm-probe}" >&2; then
        echo "run.sh: host probe FAILED -- refusing to compute on this host." >&2
        exit 3
    fi
fi

if [ -n "$mpi_ranks" ]; then
    if ! "$PY" -c "import mpi4py" >/dev/null 2>&1; then
        echo "run.sh: $mpi_selector requires mpi4py in $PY" >&2
        exit 4
    fi
    mpi_launcher="${AICCM_MPI_LAUNCHER:-}"
    if [ -z "$mpi_launcher" ]; then
        for candidate in mpirun mpiexec; do
            if command -v "$candidate" >/dev/null 2>&1; then
                mpi_launcher="$(command -v "$candidate")"
                break
            fi
        done
    fi
    : "${mpi_launcher:?run.sh: $mpi_selector requires mpirun or mpiexec}"
    total_cpus="${VQ_CPUS:-$mpi_ranks}"
    case "$total_cpus" in
        ''|*[!0-9]*)
            echo "run.sh: VQ_CPUS must be a positive integer for $mpi_selector" >&2
            exit 4
            ;;
    esac
    if [ "$total_cpus" -lt 1 ]; then
        echo "run.sh: VQ_CPUS must be positive for $mpi_selector" >&2
        exit 4
    fi
    scheduler_tasks="${VQ_SCHEDULER_TASKS:-}"
    if [ -n "$scheduler_tasks" ]; then
        echo "run.sh: $mpi_selector owns both ranks; omit --scheduler-tasks" >&2
        exit 4
    fi
    if [ $((total_cpus % mpi_ranks)) -ne 0 ]; then
        echo "run.sh: VQ_CPUS must divide evenly across $mpi_ranks MPI ranks" >&2
        exit 4
    fi
    threads_per_rank=$((total_cpus / mpi_ranks))
    resource_semantics="vq-total-cpus-divided-by-ranks"
    export OMP_NUM_THREADS="$threads_per_rank"
    export OMP_DYNAMIC=FALSE
    export OMP_PROC_BIND=spread
    export OMP_PLACES=cores
    export OPENBLAS_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export BLIS_NUM_THREADS=1
    mpi_version="$("$mpi_launcher" --version 2>&1 || true)"
    mpi_args=(-np "$mpi_ranks")
    if [[ "$mpi_version" == *"Intel(R) MPI"* ]]; then
        launcher_kind="intel-mpi"
        binding_policy="intel-domain-omp-compact"
        export I_MPI_PIN=1
        export I_MPI_PIN_DOMAIN=omp
        export I_MPI_PIN_ORDER=compact
        mpi_args+=(-ppn "$mpi_ranks")
    elif [[ "$mpi_version" == *"Open MPI"* || "$mpi_version" == *"OpenRTE"* ]]; then
        launcher_kind="open-mpi"
        binding_policy="openmpi-ppr-node-pe-core"
        mpi_args+=(
            --map-by "ppr:${mpi_ranks}:node:PE=${threads_per_rank}"
            --bind-to core
        )
    else
        echo "run.sh: $mpi_selector cannot prove a hybrid binding policy for this MPI launcher" >&2
        exit 4
    fi
    unset AICCM_D114_MPI_LAUNCHED AICCM_D116_MPI_LAUNCHED
    export AICCM_MPI_VALIDATION_PROFILE="$mpi_validation_mode"
    if [ "$mpi_validation_mode" = "d114-rhf" ]; then
        export AICCM_D114_MPI_LAUNCHED=1
    elif [ "$mpi_validation_mode" = "d116-rks-pbe" ]; then
        export AICCM_D116_MPI_LAUNCHED=1
    else
        echo "run.sh: unsupported chi MPI validation profile" >&2
        exit 4
    fi
    export VIBEQC_MPI_REQUIRED=1
    export VIBEQC_MPI_EXPECTED_SIZE="$mpi_ranks"
    export AICCM_MPI_LAUNCHER_KIND="$launcher_kind"
    export AICCM_MPI_BINDING_POLICY="$binding_policy"
    export AICCM_MPI_THREADS_PER_RANK="$threads_per_rank"
    export AICCM_MPI_DECLARED_CPUS="$total_cpus"
    export AICCM_MPI_RESOURCE_SEMANTICS="$resource_semantics"
    echo "run.sh: launching $mpi_ranks MPI ranks x $threads_per_rank OpenMP threads ($binding_policy)" >&2
    "$mpi_launcher" "${mpi_args[@]}" "$PY" "$_here/$case_script" "$@"
else
    "$PY" "$_here/$case_script" "$@"
fi
