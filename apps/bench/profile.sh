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
	if grep -q '^bench: ' "$tmp/rec.log"; then
		echo "bench failed (see above)" >&2
	else
		echo "perf record failed; perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)" >&2
	fi
	exit 1
fi
perf report -i "$tmp/perf.data" --stdio --no-children -g none >"$tmp/report.txt"
# symbol rows only (lines starting with a percentage), first 25; no early-closing pipe
awk '/^ *[0-9.]+%/ { if (++n <= 25) print }' "$tmp/report.txt" >"$tmp/top.txt"
[ -s "$tmp/top.txt" ] || { echo "perf report produced no symbol rows" >&2; exit 1; }
mv "$tmp/top.txt" "$out/$wl.txt"
echo "wrote $out/$wl.txt"
