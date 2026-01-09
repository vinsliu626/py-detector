#!/usr/bin/env bash
set -e

: "${PORT:=7860}"

uvicorn server:app --host 0.0.0.0 --port "$PORT"
