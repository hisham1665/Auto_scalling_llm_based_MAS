#!/usr/bin/env bash
set -euo pipefail

python3 main.py experiment --all --scenario "${SCENARIO:-medical}"
