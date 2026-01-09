#!/usr/bin/env bash
set -e

: "${PORT:=8000}"

# 绑定端口 + 单 worker（内存更稳）
uvicorn server:app --host 0.0.0.0 --port "$PORT" --workers 1
