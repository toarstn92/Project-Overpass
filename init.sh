#!/usr/bin/env sh
# Project Overpass launcher - run from anywhere: ./init.sh
cd "$(dirname "$0")" || exit 1
if command -v python3 >/dev/null 2>&1; then
  exec python3 init.py "$@"
fi
exec python init.py "$@"
