#!/usr/bin/env bash
# CRYSTAL23 single-point runner for an AICCM-2026 reference .d12.
#
# Generates the external CRYSTAL23 reference energy for one test-set system at the
# *test-set geometry* (single point — any OPTGEOM block is stripped so the
# reference matches the AICCM geometry exactly). Designed for vq:
#
#     vq submit <host> -d <payload>/ -- bash crun.sh INPUT.d12 <system>
#
# where <payload>/ holds the bundled INPUT.d12 + this script (see
# make_crystal_jobs.py). Writes <system>__crystal23.json (total energy / atom)
# and keeps the full CRYSTAL output as job.out for the audit trail.
set -euo pipefail

D12="${1:?usage: crun.sh INPUT.d12 [system]}"
SYS="${2:-$(basename "$D12" .d12)}"
OUT="${VQ_WORKDIR:-.}"

# Locate a CRYSTAL23 binary (serial or parallel).
CRY=""
for c in Pcrystal crystal23 crystal Pcrystal23 MPPcrystal; do
    if command -v "$c" >/dev/null 2>&1; then CRY="$c"; break; fi
done
if [ -z "$CRY" ]; then
    echo "{\"system\":\"$SYS\",\"error\":\"no CRYSTAL23 binary on PATH\"}" \
        | tee "$OUT/${SYS}__crystal23.json"
    exit 3
fi

# Single point at the test-set geometry: drop OPTGEOM…ENDOPT.
sed '/^[[:space:]]*OPTGEOM/,/^[[:space:]]*ENDOPT/d' "$D12" > job.d12
# CRYSTAL reads a file named INPUT in the run dir (required for the parallel
# Pcrystal binary, which does NOT read stdin — `Pcrystal < deck` gives
# "END OF DATA IN INPUT DECK"). Serial crystal also accepts the INPUT file.
cp -f job.d12 INPUT
"$CRY" > job.out 2>&1 || true
cp -f job.out "$OUT/${SYS}__crystal23.out" 2>/dev/null || true

# CRYSTAL prints the converged total on the SCF-ENDED line as
#   "== SCF ENDED - CONVERGENCE ON ENERGY  E(AU) -7.96...E+01 CYCLES 7"
# Extract the value right after E(AU) (the trailing "CYCLES n" / "tester ..."
# numbers must NOT be picked — that was the energy_ha=3.3E-13 bug).
E=$(grep 'SCF ENDED' job.out | sed -E 's/.*E\(AU\)[[:space:]]*([-+0-9.EeDd]+).*/\1/' | tail -1 || true)
conv=$(grep -cE 'SCF ENDED.*CONVERGENCE' job.out || true)
printf '{"system":"%s","crystal_binary":"%s","energy_ha":%s,"converged":%s}\n' \
    "$SYS" "$CRY" "${E:-null}" "$([ "${conv:-0}" -ge 1 ] && echo true || echo false)" \
    | tee "$OUT/${SYS}__crystal23.json"
