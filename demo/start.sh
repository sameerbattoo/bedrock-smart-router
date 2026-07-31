#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Bedrock Smart Router Demo — One-Command Setup & Launch
# ═══════════════════════════════════════════════════════════════════════
#
# This script handles everything a new user needs:
#   1. Checks prerequisites (Python, Node.js, npm, AWS credentials)
#   2. Installs the bedrock-smart-router Python package (editable)
#   3. Installs backend Python dependencies
#   4. Runs prerequisites (database + guardrail)
#   5. Installs frontend npm packages
#   6. Builds the frontend
#   7. Kills any existing processes on the required ports
#   8. Starts backend + frontend
#   9. Runs health checks
#
# Usage:
#   bash demo/start.sh          (from project root)
#   bash start.sh               (from demo/ directory)
#   ./start.sh                  (if executable)
#
set -e

# ── Resolve paths ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEMO_DIR="$SCRIPT_DIR"
PROJECT_ROOT="$(cd "$DEMO_DIR/.." && pwd)"
BACKEND_DIR="$DEMO_DIR/backend"
FRONTEND_DIR="$DEMO_DIR/frontend"

BACKEND_PORT=8000
FRONTEND_PORT=5173

# ── Colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

step() { echo -e "\n${YELLOW}[$1/$TOTAL_STEPS]${NC} ${BOLD}$2${NC}"; }
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; exit 1; }

TOTAL_STEPS=9

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Bedrock Smart Router Demo — Setup & Launch${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

# ── Step 1: Check prerequisites ─────────────────────────────────────
step 1 "Checking prerequisites..."

# Python
if command -v python3 &>/dev/null; then
  PYTHON=python3
elif command -v python &>/dev/null; then
  PYTHON=python
else
  fail "Python 3.9+ is required but not found. Install from https://python.org"
fi

PY_VERSION=$($PYTHON --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
ok "Python $PY_VERSION ($($PYTHON --version 2>&1))"

# Node.js
if ! command -v node &>/dev/null; then
  fail "Node.js 18+ is required but not found. Install from https://nodejs.org"
fi
NODE_VERSION=$(node --version)
ok "Node.js $NODE_VERSION"

# npm
if ! command -v npm &>/dev/null; then
  fail "npm is required but not found. It should come with Node.js."
fi
ok "npm $(npm --version)"

# AWS credentials
if ! $PYTHON -c "import boto3; boto3.Session().client('sts').get_caller_identity()" &>/dev/null; then
  warn "AWS credentials not configured or expired. The demo requires valid AWS credentials with Bedrock access."
  warn "Run 'aws configure' or set AWS_PROFILE before starting."
else
  AWS_ACCOUNT=$($PYTHON -c "import boto3; print(boto3.Session().client('sts').get_caller_identity()['Account'])" 2>/dev/null)
  ok "AWS credentials valid (account: $AWS_ACCOUNT)"
fi

# uvx (required for MCP servers in Strands agents)
if ! command -v uvx &>/dev/null; then
  # Check if uvx exists in Python's bin directory but isn't on PATH
  UVX_PATH=$($PYTHON -c "import shutil; p = shutil.which('uvx'); print(p or '')" 2>/dev/null)
  if [ -z "$UVX_PATH" ]; then
    echo -e "  Installing uv (provides uvx for MCP servers)..."
    $PYTHON -m pip install uv --quiet 2>&1 | tail -2
    UVX_PATH=$($PYTHON -c "import shutil; p = shutil.which('uvx'); print(p or '')" 2>/dev/null)
  fi
  if [ -n "$UVX_PATH" ]; then
    UVX_DIR=$(dirname "$UVX_PATH")
    export PATH="$UVX_DIR:$PATH"
    ok "uvx found at $UVX_PATH (added to PATH)"
  else
    warn "uvx not found. Strands Agents (use-case 3) will not work without it."
    warn "Install with: pip install uv"
  fi
else
  ok "uvx $(uvx --version 2>/dev/null || echo 'available')"
fi

# Graphviz (required for the `diagrams` Python package to render PNGs)
if ! command -v dot &>/dev/null; then
  echo -e "  Installing graphviz (required for diagram generation)..."
  if command -v brew &>/dev/null; then
    brew install graphviz --quiet 2>&1 | tail -2
  elif command -v apt-get &>/dev/null; then
    sudo apt-get install -y graphviz --quiet 2>&1 | tail -2
  elif command -v yum &>/dev/null; then
    sudo yum install -y graphviz --quiet 2>&1 | tail -2
  else
    warn "graphviz not found and no package manager detected. Diagram generation will not work."
    warn "Install manually: https://graphviz.org/download/"
  fi
  if command -v dot &>/dev/null; then
    ok "graphviz installed ($(dot -V 2>&1 | head -1))"
  fi
else
  ok "graphviz $(dot -V 2>&1 | head -1)"
fi

# ── Step 2: Install bedrock-smart-router package ────────────────────
step 2 "Installing bedrock-smart-router package (editable)..."

if $PYTHON -c "import bedrock_smart_router" &>/dev/null; then
  ok "bedrock-smart-router already installed"
else
  (cd "$PROJECT_ROOT" && $PYTHON -m pip install -e . --quiet 2>&1 | tail -3)
  if $PYTHON -c "import bedrock_smart_router" &>/dev/null; then
    ok "bedrock-smart-router installed"
  else
    fail "Failed to install bedrock-smart-router. Run: pip install -e . from project root"
  fi
fi

# ── Step 3: Install backend Python dependencies ─────────────────────
step 3 "Installing backend Python dependencies..."

if [ -f "$BACKEND_DIR/requirements.txt" ]; then
  $PYTHON -m pip install -r "$BACKEND_DIR/requirements.txt" --quiet 2>&1 | tail -3
  ok "Backend dependencies installed"
else
  warn "No requirements.txt found in backend/"
fi

# Verify critical imports
if ! $PYTHON -c "import fastapi, uvicorn, boto3, strands" &>/dev/null; then
  fail "Critical backend packages missing. Check: pip install fastapi uvicorn boto3 strands-agents"
fi
ok "Backend imports verified (fastapi, uvicorn, boto3, strands)"

# ── Step 4: Run prerequisites (database + guardrail) ────────────────
step 4 "Running prerequisites (database + guardrail)..."
$PYTHON "$DEMO_DIR/prerequisite/setup_all.py" || warn "Prerequisites partially failed (non-critical). Continuing..."
ok "Prerequisites ready"

# Pre-warm MCP server packages (uvx downloads on first run, which can timeout)
if command -v uvx &>/dev/null; then
  echo -e "  Pre-warming MCP server packages (first run downloads ~60s)..."
  uv tool install --force awslabs.aws-documentation-mcp-server --quiet 2>/dev/null || true
  ok "MCP server packages cached"
fi

# ── Step 5: Install frontend npm packages ───────────────────────────
step 5 "Installing frontend npm packages..."

if [ ! -d "$FRONTEND_DIR/node_modules" ] || [ "$FRONTEND_DIR/package.json" -nt "$FRONTEND_DIR/node_modules/.package-lock.json" ]; then
  (cd "$FRONTEND_DIR" && npm install --silent 2>&1 | tail -3)
  ok "npm packages installed"
else
  ok "npm packages up to date"
fi

# ── Step 6: Build frontend ──────────────────────────────────────────
step 6 "Building frontend..."

(cd "$FRONTEND_DIR" && npm run build 2>&1 | grep -E "(built|error)" | head -3)
if [ -f "$FRONTEND_DIR/dist/index.html" ]; then
  ok "Frontend built → dist/"
else
  fail "Frontend build failed. Run 'npm run build' in demo/frontend/ for details."
fi

# ── Step 7: Kill existing processes ─────────────────────────────────
step 7 "Killing existing processes on ports $BACKEND_PORT, $FRONTEND_PORT..."

KILLED=0
for PORT in $BACKEND_PORT $FRONTEND_PORT; do
  PIDS=$(lsof -ti:$PORT 2>/dev/null || true)
  if [ -n "$PIDS" ]; then
    echo "$PIDS" | xargs kill -9 2>/dev/null || true
    KILLED=$((KILLED + 1))
  fi
done

if [ $KILLED -gt 0 ]; then
  ok "Killed processes on $KILLED port(s)"
  sleep 1
else
  ok "Ports are free"
fi

# ── Step 8: Start servers ───────────────────────────────────────────
step 8 "Starting servers..."

# Backend
echo -e "  Starting backend (FastAPI) on port $BACKEND_PORT..."
(cd "$BACKEND_DIR" && $PYTHON -m uvicorn app:app --host 0.0.0.0 --port $BACKEND_PORT --log-level warning) &
BACKEND_PID=$!

# Frontend
echo -e "  Starting frontend (Vite) on port $FRONTEND_PORT..."
(cd "$FRONTEND_DIR" && npm run dev -- --port $FRONTEND_PORT --host 2>/dev/null) &
FRONTEND_PID=$!

ok "Backend PID: $BACKEND_PID | Frontend PID: $FRONTEND_PID"

# ── Step 9: Health checks ───────────────────────────────────────────
step 9 "Running health checks..."

# Backend health
RETRIES=0
MAX_RETRIES=20
while [ $RETRIES -lt $MAX_RETRIES ]; do
  if curl -sf http://localhost:$BACKEND_PORT/api/health | grep -q '"status":"ok"' 2>/dev/null; then
    HEALTH=$(curl -s http://localhost:$BACKEND_PORT/api/health)
    ok "Backend healthy: $HEALTH"
    break
  fi
  RETRIES=$((RETRIES + 1))
  sleep 1
done

if [ $RETRIES -eq $MAX_RETRIES ]; then
  fail "Backend failed to start after ${MAX_RETRIES}s. Check logs above."
fi

# Frontend health
RETRIES=0
MAX_RETRIES=10
while [ $RETRIES -lt $MAX_RETRIES ]; do
  if curl -sf http://localhost:$FRONTEND_PORT | grep -q "<!DOCTYPE html>" 2>/dev/null; then
    ok "Frontend serving HTML"
    break
  fi
  RETRIES=$((RETRIES + 1))
  sleep 1
done

if [ $RETRIES -eq $MAX_RETRIES ]; then
  warn "Frontend may still be starting — check http://localhost:$FRONTEND_PORT in a moment"
fi

# ── Done ────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Demo is live!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${BOLD}Frontend:${NC}  ${CYAN}http://localhost:$FRONTEND_PORT${NC}"
echo -e "  ${BOLD}Backend:${NC}   ${CYAN}http://localhost:$BACKEND_PORT${NC}"
echo -e "  ${BOLD}API docs:${NC}  ${CYAN}http://localhost:$BACKEND_PORT/docs${NC}"
echo ""
echo -e "  Press ${RED}Ctrl+C${NC} to stop both servers."
echo ""

# Open browser
if command -v open &>/dev/null; then
  open "http://localhost:$FRONTEND_PORT"
elif command -v xdg-open &>/dev/null; then
  xdg-open "http://localhost:$FRONTEND_PORT"
fi

# ── Cleanup on exit ─────────────────────────────────────────────────
cleanup() {
  echo ""
  echo -e "${YELLOW}Shutting down...${NC}"
  kill $BACKEND_PID 2>/dev/null || true
  kill $FRONTEND_PID 2>/dev/null || true
  # Clean up any orphans
  lsof -ti:$BACKEND_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
  lsof -ti:$FRONTEND_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
  echo -e "${GREEN}Stopped.${NC}"
  exit 0
}

trap cleanup INT TERM
wait
