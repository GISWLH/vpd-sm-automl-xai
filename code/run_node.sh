#!/bin/bash
# start 48 worker processes on this node: run_node.sh <first_rank>; all state stays in node-local /tmp/vpd_sm
cd /tmp/vpd_sm
export HOME=/tmp/vpd_sm/home TMPDIR=/tmp/vpd_sm/tmp MPLCONFIGDIR=/tmp/vpd_sm/tmp/mpl PYTHONWARNINGS=ignore
for r in $(seq $1 $(($1 + 47))); do
  setsid nohup venv/bin/python code/04_worker.py proc out $r 144 8 > logs/$r.log 2>&1 < /dev/null &
done
