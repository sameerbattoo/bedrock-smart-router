#!/bin/bash
# Run the full test suite with all dependencies installed.
#
# Usage:
#   ./scripts/run_tests.sh          # Run all tests
#   ./scripts/run_tests.sh -v       # Verbose output
#   ./scripts/run_tests.sh -k chat  # Run only tests matching "chat"

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Detect virtual environment
if [ -d ".venv" ]; then
    PYTHON=".venv/bin/python"
    PIP=".venv/bin/pip"
elif command -v python3 &>/dev/null; then
    PYTHON="python3"
    PIP="pip3"
else
    echo "ERROR: No Python found. Create a venv: python3 -m venv .venv && source .venv/bin/activate"
    exit 1
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Bedrock Smart Router — Test Suite"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Install test dependencies
echo ""
echo "Installing test dependencies..."
$PIP install -e ".[dev,ml,strands]" --quiet 2>&1 | grep -v "already satisfied" || true
echo "✓ Dependencies installed"

# Run tests
echo ""
echo "Running tests..."
echo ""
$PYTHON -m pytest tests/ "$@" --tb=short -q

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Done"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
