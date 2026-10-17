#!/usr/bin/env bash
# Stop pipeline services started by scripts/run_all.sh (PIDs in runs/.pids).
PIDFILE="$(dirname "$0")/../runs/.pids"
[ -f "$PIDFILE" ] || exit 0
while read -r pid; do
  kill "$pid" 2>/dev/null || true
done < "$PIDFILE"
rm -f "$PIDFILE"
