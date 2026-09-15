#!/usr/bin/env bash
#
# build_oracle.sh — reproducibly build a local MSINDO binary for vibe-qc parity
# validation (CLAUDE.md §10 external-validation pattern).
#
# MSINDO (Bredow/Geudtner/Jug; (C) Mulliken Center for Theoretical Chemistry,
# University of Bonn) is NOT part of vibe-qc's runtime — this builds it
# out-of-process purely as a reference oracle. The pristine MSINDO source is
# copied to a temp dir and three minimal, documented patches are applied to the
# COPY (the reference checkout is never modified):
#
#   1. delimiter.h  MV=5000 -> MV (default 200). MSINDO statically allocates
#      COMMON arrays dimensioned O(MV^2) (e.g. MADKONST(MV,2*MV)); at MV=5000
#      the __common segment is ~2.85 GB and collides with the macOS arm64 dyld
#      shared-cache region ("map cache into shared region failed"). Reducing MV
#      only lowers the max system size — the physics is unchanged. 200 atoms is
#      ample for the molecular + small-CCM validation set.
#   2. einles.f  add 'LOGICAL :: EWALDMESH' — a set-only keyword flag that is
#      undeclared upstream (only assigned, never read), so IMPLICIT NONE rejects
#      it. Declaring it locally is inert.
#   3. vderf_shim.f — provide MKL-VML vderf/vderfc (vectorized erf/erfc) via the
#      standard ERF/ERFC intrinsics, since we link reference LAPACK/BLAS, not MKL.
#
# Usage:
#   MSINDO_SRC=/path/to/msindo/2025e ./build_oracle.sh [output_binary]
# Env:
#   MSINDO_SRC   MSINDO 2025 tree containing source/ and include/ (required)
#   MSINDO_MV    MV override (default 200)
#   LAPACK_PREFIX  homebrew lapack prefix (default /opt/homebrew/opt/lapack)
#
set -euo pipefail

MSINDO_SRC="${MSINDO_SRC:-}"
OUT="${1:-$(pwd)/msindo_oracle}"
MV="${MSINDO_MV:-200}"
LAPACK_PREFIX="${LAPACK_PREFIX:-/opt/homebrew/opt/lapack}"

[ -n "$MSINDO_SRC" ] || { echo "ERROR: set MSINDO_SRC=<path to msindo 2025 tree with source/ and include/>"; exit 1; }
[ -d "$MSINDO_SRC/source" ] || { echo "ERROR: $MSINDO_SRC/source not found (set MSINDO_SRC)"; exit 1; }
command -v gfortran >/dev/null || { echo "ERROR: gfortran not on PATH"; exit 1; }

BUILD="$(mktemp -d)"
trap 'rm -rf "$BUILD"' EXIT
echo "[build_oracle] staging pristine MSINDO source -> $BUILD"
cp -R "$MSINDO_SRC/source" "$MSINDO_SRC/include" "$BUILD/"

echo "[build_oracle] patch 1: MV=5000 -> MV=$MV"
perl -pi -e "s/MV=5000/MV=$MV/" "$BUILD/include/delimiter.h"

echo "[build_oracle] patch 2: declare set-only EWALDMESH in einles.f"
if ! grep -q "LOGICAL *:: *EWALDMESH" "$BUILD/source/einles.f"; then
  awk '{print}
       /LOGICAL[[:space:]]*::[[:space:]]*CRYSYS/{print "      LOGICAL              :: EWALDMESH"}' \
    "$BUILD/source/einles.f" > "$BUILD/source/einles.f.tmp"
  mv "$BUILD/source/einles.f.tmp" "$BUILD/source/einles.f"
fi

echo "[build_oracle] patch 3: vderf/vderfc erf shim"
cat > "$BUILD/source/vderf_shim.f" <<'FEOF'
!     vibe-qc oracle build shim: MKL-VML vderf/vderfc via ERF/ERFC intrinsics.
      SUBROUTINE VDERF(N, A, Y)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N
      REAL, INTENT(IN)    :: A(N)
      REAL, INTENT(OUT)   :: Y(N)
      INTEGER :: I
      DO I = 1, N
        Y(I) = ERF(A(I))
      END DO
      END SUBROUTINE VDERF

      SUBROUTINE VDERFC(N, A, Y)
      IMPLICIT NONE
      INTEGER, INTENT(IN) :: N
      REAL, INTENT(IN)    :: A(N)
      REAL, INTENT(OUT)   :: Y(N)
      INTEGER :: I
      DO I = 1, N
        Y(I) = ERFC(A(I))
      END DO
      END SUBROUTINE VDERFC
FEOF

FLAGS="-fdefault-real-8 -std=legacy -fallow-argument-mismatch -fno-align-commons -w"
mkdir -p "$BUILD/obj"
echo "[build_oracle] compiling $(ls "$BUILD"/source/*.f | wc -l | tr -d ' ') files ..."
for f in "$BUILD"/source/*.f; do
  gfortran -c $FLAGS -I"$BUILD/include" "$f" -o "$BUILD/obj/$(basename "$f" .f).o"
done

echo "[build_oracle] linking -> $OUT"
gfortran "$BUILD"/obj/*.o -L"$LAPACK_PREFIX/lib" -llapack -lblas -o "$OUT"
echo "[build_oracle] done: $OUT"
