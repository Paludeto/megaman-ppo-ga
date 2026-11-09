#!/usr/bin/env bash
# Stops the fair-comparison scheduler and the jobs it is running (they are re-run on the next start).
cd "$(dirname "$0")/../.."
read -r pid _ < campaign/fair/scheduler.pid
kill -TERM "$pid" && echo "sent SIGTERM to scheduler $pid"
