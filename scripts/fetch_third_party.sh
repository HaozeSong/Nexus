#!/usr/bin/env bash
# Clone the vendored benchmark and baseline repositories at the commits this artifact was developed against.
# Usage: scripts/fetch_third_party.sh [all|benchmarks|baselines|traces]
#   all         benchmarks + baselines (everything the protocols, benchmarks and tests need)
#   traces      the public TraceLab traces, read only by scripts/profile_workloads.py (not part of "all")
set -e
cd "$(dirname "$0")/../third_party"
what=${1:-all}
case "$what" in all|benchmarks|baselines|traces) ;; *) echo "usage: $0 [all|benchmarks|baselines|traces]" >&2; exit 2 ;; esac
clone() { d=$1; url=$2; rev=$3; if [ -d "$d/.git" ]; then echo "exists: $d"; else git clone -q "$url" "$d" && git -C "$d" checkout -q "$rev" && echo "cloned: $d @ $rev"; fi; }
if [ "$what" = all ] || [ "$what" = benchmarks ]; then
  clone tau2-bench https://github.com/sierra-research/tau2-bench.git b7ea907
  clone WorkBench https://github.com/olly-styles/WorkBench.git 49c7dfd
  echo "export TAU2_DATA_DIR=$PWD/tau2-bench/data   # optional: this is the default the tau2 benchmark uses"
fi
if [ "$what" = all ] || [ "$what" = baselines ]; then
  clone atomix https://github.com/mpi-dsg/atomix.git 61e411b
  clone MEMTX_ https://github.com/lxy1134/MEMTX_.git 4e1124c
fi
if [ "$what" = traces ]; then
  clone TraceLab https://github.com/uw-syfi/TraceLab.git 11b8b14
  # the trace database is a release asset of TraceLab, not a file of its repository
  db=TraceLab/trace/syfi_coding_trace.duckdb
  if [ -f "$db" ]; then echo "exists: $db"; else
    mkdir -p TraceLab/trace
    curl -L --fail -o "$db" https://github.com/uw-syfi/TraceLab/releases/latest/download/syfi_coding_trace.duckdb && echo "downloaded: $db"
  fi
fi
