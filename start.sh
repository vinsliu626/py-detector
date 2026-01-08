#!/usr/bin/env bash
set -e

# Render provides $PORT
: "${PORT:=8000}"

# start uvicorn
uvicorn server:app --host 0.0.0.0 --port "$PORT"
