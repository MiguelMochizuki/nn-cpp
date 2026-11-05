#!/usr/bin/env bash
# Usage: profile.sh <build-dir> <workload> <out-dir>
set -euo pipefail
[ $# -eq 3 ] || { echo "usage: $0 <build-dir> <workload> <out-dir>" >&2; exit 2; }
bd=$1 wl=$2 out=$3
mkdir -p "$out"
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
if ! perf record -g -o "$tmp/perf.data" -- "$bd/apps/bench" --out "$tmp/p.csv" --suite workloads --workload "$wl" \
	>"$tmp/rec.log" 2>&1; then
	cat "$tmp/rec.log" >&2
	echo "perf record failed; perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)" >&2
	exit 1
fi
perf report -i "$tmp/perf.data" --stdio --no-children 2>/dev/null | grep -v '^#' | grep -v '^$' | grep '%' | head -25 >"$out/$wl.txt"
echo "wrote $out/$wl.txt"
