#!/usr/bin/env bash
# Simulate incoherent e+e- pair production at a muon collider with GuineaPig and
# reduce each bunch crossing to the two numbers the rate figure needs: how many
# pair leptons are produced above a p_T threshold, and the luminosity of that
# crossing.
#
#   ./run_pairs.sh <accelerator> <parameter-set> <n_events> <out-dir> [n_chains]
#
# SKIP_BASE offsets the per-chain seed advance. Chains are decorrelated by
# advancing the random state (chain index - 1) times, so a second batch started
# with the default would replay the first one exactly; set SKIP_BASE past the
# chain count of every earlier batch to accumulate independent statistics.
#
# e.g.  ./run_pairs.sh mumu10tev pairs10tev 16 out/10tev 2
#
# PT_MINS lists the p_T thresholds (GeV) to count at -- every threshold is
# evaluated on the same crossings in one pass, so the resulting curves are
# statistically consistent with each other.
#
# Accelerators and parameter sets come from acc.dat next to this script, which
# is mounted over the one baked into the image. Everything runs inside
# ghcr.io/lawrenceleejr/guineapig_mumu, so the only host requirement is Docker.
#
# Each chain is a separate container running its events back to back, carrying
# the GuineaPig random state forward in rndm.save so successive crossings are
# statistically independent. Chains are seeded apart from each other by advancing
# that state a different number of times first, then run in parallel.
#
# Output: <out-dir>/summary.txt, one line per bunch crossing, with a
# "# columns:" header naming the p_T threshold behind each count:
#   chain, event, n_stored, lumi_m2, n_pt_<threshold>, ...
set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/lawrenceleejr/guineapig_mumu:latest}"
# 0.015 GeV: pair leptons that get out of the beam pipe at all.
# 1.4   GeV: the minimum p_T for a particle to reach the ECAL surface.
PT_MINS="${PT_MINS:-0.015 1.4}"   # GeV

ACCELERATOR="${1:?usage: run_pairs.sh <accelerator> <params> <n_events> <out-dir> [n_chains]}"
PARAMS="${2:?missing parameter set}"
N_EVENTS="${3:?missing number of events}"
OUT_DIR="${4:?missing output directory}"
N_CHAINS="${5:-2}"
SKIP_BASE="${SKIP_BASE:-0}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$OUT_DIR"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"
cp "$HERE/acc.dat" "$OUT_DIR/acc.dat"

# Split the requested crossings as evenly as possible over the chains.
per_chain=$(( N_EVENTS / N_CHAINS ))
remainder=$(( N_EVENTS % N_CHAINS ))

COLUMNS="chain, event, n_stored, lumi_m2"
for pt in $PT_MINS; do COLUMNS="$COLUMNS, n_pt_$pt"; done

echo "GuineaPig: $N_EVENTS crossing(s) of '$PARAMS' on '$ACCELERATOR'"
echo "           $N_CHAINS parallel chain(s), p_T > {$PT_MINS} GeV, output in $OUT_DIR"

pids=()
for (( k = 1; k <= N_CHAINS; k++ )); do
    events=$per_chain
    if (( k <= remainder )); then events=$(( events + 1 )); fi
    (( events > 0 )) || continue

    docker run --rm -v "$OUT_DIR":/work -w /work \
        -e CHAIN="$k" -e SKIP="$(( SKIP_BASE + k - 1 ))" -e EVENTS="$events" \
        -e ACCELERATOR="$ACCELERATOR" -e PARAMS="$PARAMS" -e PT_MINS="$PT_MINS" \
        --entrypoint /bin/bash "$IMAGE" -c '
            set -euo pipefail
            mkdir -p "chain$CHAIN" && cd "chain$CHAIN"
            cp /app/guinea_nofftw . && cp ../acc.dat .

            # Decorrelate the chains: advance the random state SKIP times with a
            # cheap luminosity-only run before the physics starts.
            for (( s = 0; s < SKIP; s++ )); do
                ./guinea_nofftw "$ACCELERATOR" seed seed.out > /dev/null 2>&1
            done

            : > "../summary.chain$CHAIN.txt"
            for (( i = 1; i <= EVENTS; i++ )); do
                ./guinea_nofftw "$ACCELERATOR" "$PARAMS" crossing.out > "log$i.txt" 2>&1
                lumi=$(sed -n "s/^lumi_ee=\(.*\);/\1/p" crossing.out | head -1)
                awk -v chain="$CHAIN" -v ev="$i" -v lumi="$lumi" -v ptmins="$PT_MINS" '"'"'
                    BEGIN { nthr = split(ptmins, thr, /[ ,]+/) }
                    { e = ($1 < 0) ? -$1 : $1
                      pt = e * sqrt($2*$2 + $3*$3)
                      n++
                      for (t = 1; t <= nthr; t++) if (pt > thr[t] + 0) count[t]++ }
                    END { line = chain " " ev " " n+0 " " lumi
                          for (t = 1; t <= nthr; t++) line = line " " count[t]+0
                          print line }
                '"'"' pairs0.dat >> "../summary.chain$CHAIN.txt"
                echo "chain $CHAIN: crossing $i/$EVENTS done ($(tail -1 "../summary.chain$CHAIN.txt"))"
                rm -f pairs0.dat
            done
        ' > "$OUT_DIR/chain$k.log" 2>&1 &
    pids+=($!)
done

if [ ${#pids[@]} -eq 0 ]; then
    echo "nothing to run: n_events=$N_EVENTS over $N_CHAINS chain(s)" >&2
    exit 1
fi

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done

{
    echo "# accelerator: $ACCELERATOR, parameters: $PARAMS, skip_base: $SKIP_BASE"
    echo "# columns: $COLUMNS"
    cat "$OUT_DIR"/summary.chain*.txt
} > "$OUT_DIR/summary.txt"
echo "$(grep -vc '^#' "$OUT_DIR/summary.txt") crossing(s) written to $OUT_DIR/summary.txt"
exit $status
