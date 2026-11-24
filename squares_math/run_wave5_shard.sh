#!/usr/bin/env bash
set -euo pipefail

SHARD="${SHARD_ID:-0}"
MINS="${MINUTES_TOTAL:-42}"
OUT="artifacts/shard_${SHARD}"
DS="/tmp/sq_w5_ds"
mkdir -p "$OUT" "$DS" /tmp/sq_bins

echo "=== Wave 5 Shard ${SHARD} starting on $(nproc) cores (budget ${MINS}m) ==="

# Build needed binaries cleanly per shard role
if [ "$SHARD" -le 9 ]; then
  (cd squares_math/fq_crate && cargo build --release --bin fq)
  ln -sf "$(pwd)/squares_math/fq_crate/target/release/fq" /tmp/sq_bins/fq
elif [ "$SHARD" -le 15 ]; then
  (cd squares_math/fq_crate && cargo build --release --bin fq --bin anneal --bin harden)
  ln -sf "$(pwd)/squares_math/fq_crate/target/release/fq" /tmp/sq_bins/fq
  ln -sf "$(pwd)/squares_math/fq_crate/target/release/anneal" /tmp/sq_bins/anneal
  ln -sf "$(pwd)/squares_math/fq_crate/target/release/harden" /tmp/sq_bins/harden
else
  (cd squares_math/fq_crate && cargo build --release --bin fq)
  ln -sf "$(pwd)/squares_math/fq_crate/target/release/fq" /tmp/sq_bins/fq
  g++ -O3 -std=c++20 squares_math/src/rect_quench.cpp -o /tmp/sq_bins/rect_quench
fi

ln -sf "$(pwd)/squares_math/seeds/pub_base.npz" "$DS/nov24_union.npz"
ln -sf "$(pwd)/squares_math/seeds/lag_targets_v2.json" "$DS/lag_targets.json"
ln -sf "$(pwd)/squares_math/seeds/gaps.json" "$DS/gaps.json"
ln -sf "$(pwd)/squares_math/tools/"*.py "$DS/"

export SQ_DS_DIR="$DS"
export FQ_BIN="/tmp/sq_bins/fq"
export ANNEAL_BIN="/tmp/sq_bins/anneal"
export HARDEN_BIN="/tmp/sq_bins/harden"
export RQ_BIN="/tmp/sq_bins/rect_quench"

case "$SHARD" in
  0|1|2|3|4|5)
    python3 -u squares_math/workers/w5_lband_bandgen.py \
      --union-npz "$DS/nov24_union.npz" \
      --lags-json "$DS/lag_targets.json" \
      --gaps-json "$DS/gaps.json" \
      --shard "$SHARD" --num-shards 6 --max-targets 240 \
      --minutes-total "$MINS" --procs 4 \
      --out-npz "${OUT}/w5_lband_s${SHARD}.npz"
    ;;
  6|7|8|9)
    SUB=$((SHARD - 6))
    python3 -u squares_math/workers/w5_stretch_wave.py \
      --union-npz "$DS/nov24_union.npz" \
      --lags-json "$DS/lag_targets.json" \
      --shard "$SUB" --num-shards 4 \
      --minutes-total "$MINS" --procs 4 \
      --out-npz "${OUT}/w5_stretch_s${SUB}.npz"
    ;;
  10|11|12|13|14|15)
    SUB=$((SHARD - 10))
    python3 -u squares_math/workers/w4_super_engine.py \
      --shard "$SUB" --num-shards 6 --max-targets 240 \
      --minutes-total "$MINS" --procs 4 \
      --out-npz "${OUT}/w5_super_s${SUB}.npz"
    ;;
  16|17|18|19)
    SUB=$((SHARD - 16))
    python3 -u squares_math/workers/w4_quenched_faultdp.py \
      --shard "$SUB" --num-shards 4 --procs 4 --rounds 3 \
      --out-npz "${OUT}/w5_fdp_s${SUB}.npz"
    ;;
  *)
    echo "Unknown shard $SHARD"
    exit 1
    ;;
esac

ls -lh "$OUT"
