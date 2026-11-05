#!/usr/bin/env bash
# Usage: run_baseline.sh <out-dir>   (env BENCH_ARGS: extra args appended to each bench call)
set -euo pipefail
[ $# -eq 1 ] || { echo "usage: $0 <out-dir>" >&2; exit 2; }
out=$1
root=$(cd "$(dirname "$0")/../.." && pwd)
mkdir -p "$out"
gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)
echo "CPU governor: $gov"
[ "$gov" = performance ] || echo "WARNING: governor is not 'performance'; timings will be noisy" >&2
# shellcheck disable=SC2206
extra=(${BENCH_ARGS:-})
for cfg in native portable; do
	bin=$root/build/$cfg/apps/bench
	[ -x "$bin" ] || { echo "missing $bin (build build/$cfg first)" >&2; exit 1; }
	for n in 1 2 4 8; do
		f=$out/$cfg-pinned-t$n.csv
		echo "== $f"
		NN_NUM_THREADS=$n taskset -c 0-$((n - 1)) "$bin" --out "$f" --label pinned "${extra[@]}"
	done
	f=$out/$cfg-unpinned-t12.csv
	echo "== $f"
	NN_NUM_THREADS=12 "$bin" --out "$f" --label unpinned "${extra[@]}"
done
