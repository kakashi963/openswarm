#!/bin/bash
# Run the backend pytest suite. Creates backend/.venv if missing, installs
# requirements.txt + requirements-dev.txt the first time (or whenever
# pytest isn't importable), then invokes pytest.
#
# Usage:
#   bash scripts/test.sh                    # run everything (~30s; 500 stress iters)
#   bash scripts/test.sh --quick            # cap DISCONNECT_STRESS_N at 20 for fast iteration
#   bash scripts/test.sh -k disconnect      # forward any pytest args
#   bash scripts/test.sh backend/tests/test_analytics.py -v
#
# Env:
#   DISCONNECT_STRESS_N   Override the stress-iteration count for
#                         test_disconnect_resilience.py (default 500;
#                         --quick sets it to 20).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_ROOT/backend"
VENV_DIR="$BACKEND_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"

# --- Parse our one wrapper flag, then forward the rest to pytest ----------
QUICK=0
PYTEST_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --quick) QUICK=1 ;;
        *) PYTEST_ARGS+=("$arg") ;;
    esac
done

# --- Ensure venv exists ---------------------------------------------------
if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "==> Creating venv at $VENV_DIR"
    python3 -m venv "$VENV_DIR"
fi

# --- Ensure pytest is installed (idempotent: only re-installs if missing) -
if ! "$PYTHON_BIN" -c "import pytest, pytest_asyncio" >/dev/null 2>&1; then
    echo "==> Installing backend deps + dev deps into $VENV_DIR"
    "$PYTHON_BIN" -m pip install --upgrade pip >/dev/null
    "$PYTHON_BIN" -m pip install \
        -r "$BACKEND_DIR/requirements.txt" \
        -r "$BACKEND_DIR/requirements-dev.txt"
fi

# --- Default target = backend/tests/ if caller didn't pass any path -------
if [[ ${#PYTEST_ARGS[@]} -eq 0 ]]; then
    PYTEST_ARGS=(backend/tests/ -v)
fi

# --- --quick: dial the disconnect stress loop way down --------------------
if [[ "$QUICK" -eq 1 ]]; then
    export DISCONNECT_STRESS_N="${DISCONNECT_STRESS_N:-20}"
    echo "==> --quick: DISCONNECT_STRESS_N=$DISCONNECT_STRESS_N"
fi

# Run from project root so `from backend.apps...` imports resolve.
cd "$PROJECT_ROOT"
echo "==> Running: pytest ${PYTEST_ARGS[*]}"
exec "$PYTHON_BIN" -m pytest "${PYTEST_ARGS[@]}"
