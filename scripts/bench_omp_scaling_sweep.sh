#!/usr/bin/env bash
# Drive bench_omp_scaling.py over the thread grid and emit a TSV.
# Two axes, separated so we can tell OpenMP-scalar from BLAS-bound work:
#   * OMP sweep  (BLAS pinned to 1): OMP_NUM_THREADS in 1 2 4 8 16
#   * BLAS sweep (OMP pinned to 1):  VECLIB_MAXIMUM_THREADS in 1 2 4 8
#   * combined   (OMP == BLAS):      1 2 4 8
# min-of-REPS per point (the fastest run is the least contended on a shared box).
set -u
cd "$(dirname "$0")/.."
PY=./.venv-audit-753e3c2a/bin/python
REPS="${REPS:-2}"
OUT="${OUT:-/tmp/vibeqc_omp_scaling.tsv}"
echo -e "method\tsystem\taxis\tomp\tblas\ttime_s" > "$OUT"

run() {  # method system axis omp blas
  local m=$1 s=$2 axis=$3 omp=$4 blas=$5 best="" t
  for _ in $(seq "$REPS"); do
    t=$(OMP_NUM_THREADS=$omp VECLIB_MAXIMUM_THREADS=$blas OPENBLAS_NUM_THREADS=$blas \
        MKL_NUM_THREADS=$blas nice -n 5 "$PY" scripts/bench_omp_scaling.py "$m" "$s" 2>/dev/null \
        | sed -n 's/^TIME=//p')
    [ -z "$t" ] && t="NA"
    if [ "$t" != "NA" ] && { [ -z "$best" ] || awk "BEGIN{exit !($t<$best)}"; }; then best=$t; fi
  done
  [ -z "$best" ] && best="NA"
  echo -e "${m}\t${s}\t${axis}\t${omp}\t${blas}\t${best}" | tee -a "$OUT"
}

# (method system) pairs to sweep on the OMP axis (BLAS=1)
PAIRS=("$@")
[ ${#PAIRS[@]} -eq 0 ] && PAIRS=("rks glycine" "ccsd benzene" "uccsd ch3" "mp2 benzene")

for pair in "${PAIRS[@]}"; do
  set -- $pair; m=$1; s=$2
  for omp in 1 2 4 8 16; do run "$m" "$s" omp "$omp" 1; done
done

# BLAS axis + combined, only for the contraction-heavy closed-shell methods
for pair in "ccsd h2o3" "mp2 h2o3"; do
  set -- $pair; m=$1; s=$2
  for blas in 2 4 8; do run "$m" "$s" blas 1 "$blas"; done
  for n in 2 4 8;    do run "$m" "$s" both "$n" "$n"; done
done

echo "=== done -> $OUT ==="
