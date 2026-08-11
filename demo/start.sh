#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════
# Bedrock Smart Router Demo — One-Command Setup & Launch
# ═══════════════════════════════════════════════════════════════════════
#
# This script handles everything a new user needs:
#   1. Checks system prerequisites (Python, Node.js, npm, graphviz)
#   2. Validates AWS credentials and Bedrock access
#   3. Installs the bedrock-smart-router Python package (editable)
#   4. Installs backend Python dependencies and verifies imports
#   5. Runs prerequisites (SQLite databases + Bedrock Guardrail)
#   6. Prepares agent tools (MCP servers, diagram tool, etc.)
#   7. Installs and builds the frontend
#   8. Kills any existing processes on the required ports
#   9. Starts backend + frontend
#  10. Runs health checks and prints capability summary
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
DIM='\033[2m'
NC='\033[0m'

step() { echo -e "\n${YELLOW}[$1/$TOTAL_STEPS]${NC} ${BOLD}$2${NC}"; }
ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
warn() { echo -e "  ${YELLOW}⚠${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; exit 1; }
info() { echo -e "  ${DIM}$1${NC}"; }

TOTAL_STEPS=10

# Track capabilities for summary
CAP_AWS_CREDS=false
CAP_BEDROCK=false
CAP_GUARDRAILS=false
CAP_UVX=false
CAP_MCP_DOCS=false
CAP_DIAGRAMS=false
CAP_TEXT2SQL=false
CAP_BUDGET_DB=false

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Bedrock Smart Router Demo — Setup & Launch${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"

# ═══════════════════════════════════════════════════════════════════════
# Step 1: System Prerequisites
# ═══════════════════════════════════════════════════════════════════════
step 1 "Checking system prerequisites..."

# Python 3.9+
if command -v python3 &>/dev/null; then
  PYTHON=python3
elif command -v python &>/dev/null; then
  PYTHON=python
else
  fail "Python 3.9+ is required but not found. Install from https://python.org"
fi

PY_VERSION=$($PYTHON --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)
if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 9 ]); then
  fail "Python 3.9+ required, found $PY_VERSION"
fi
ok "Python $PY_VERSION ($($PYTHON --version 2>&1))"

# Node.js 18+
if ! command -v node &>/dev/null; then
  fail "Node.js 18+ is required but not found. Install from https://nodejs.org"
fi
NODE_VERSION=$(node --version | grep -oE '[0-9]+' | head -1)
if [ "$NODE_VERSION" -lt 18 ]; then
  fail "Node.js 18+ required, found $(node --version)"
fi
ok "Node.js $(node --version)"

# npm
if ! command -v npm &>/dev/null; then
  fail "npm is required but not found. It should come with Node.js."
fi
ok "npm $(npm --version)"

# Graphviz (required for diagrams Python package to render PNGs)
if ! command -v dot &>/dev/null; then
  echo -e "  Installing graphviz (required for diagram generation)..."
  if command -v brew &>/dev/null; then
    brew install graphviz --quiet 2>&1 | tail -2
  elif command -v apt-get &>/dev/null; then
    sudo apt-get install -y graphviz --quiet 2>&1 | tail -2
  elif command -v yum &>/dev/null; then
    sudo yum install -y graphviz --quiet 2>&1 | tail -2
  else
    warn "graphviz not found and no package manager detected."
    warn "Diagram generation (use-case 3) will not work."
    warn "Install manually: https://graphviz.org/download/"
  fi
fi
if command -v dot &>/dev/null; then
  ok "graphviz $(dot -V 2>&1 | head -1)"
  CAP_DIAGRAMS=true
fi

# uvx (required for MCP servers)
if ! command -v uvx &>/dev/null; then
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
    CAP_UVX=true
  else
    warn "uvx not found. AWS Docs MCP tool (use-case 3) will not work."
    warn "Install with: pip install uv"
  fi
else
  ok "uvx $(uvx --version 2>/dev/null || echo 'available')"
  CAP_UVX=true
fi

# curl (needed for health checks)
if ! command -v curl &>/dev/null; then
  warn "curl not found — health checks will be skipped"
fi

# ═══════════════════════════════════════════════════════════════════════
# Step 2: Validate AWS Credentials & Bedrock Access
# ═══════════════════════════════════════════════════════════════════════
step 2 "Validating AWS credentials & Bedrock access..."

if $PYTHON -c "import boto3; boto3.Session().client('sts').get_caller_identity()" &>/dev/null; then
  AWS_ACCOUNT=$($PYTHON -c "import boto3; print(boto3.Session().client('sts').get_caller_identity()['Account'])" 2>/dev/null)
  AWS_REGION=$($PYTHON -c "import boto3; print(boto3.Session().region_name or 'us-west-2')" 2>/dev/null)
  ok "AWS credentials valid (account: $AWS_ACCOUNT, region: $AWS_REGION)"
  CAP_AWS_CREDS=true

  # Verify Bedrock access (try listing models)
  if $PYTHON -c "
import boto3
client = boto3.Session(region_name='us-west-2').client('bedrock')
resp = client.list_foundation_models(byOutputModality='TEXT')
assert len(resp.get('modelSummaries', [])) > 0
" &>/dev/null; then
    ok "Bedrock access confirmed (can list models)"
    CAP_BEDROCK=true
  else
    warn "Cannot access Bedrock API. Check IAM permissions: bedrock:ListFoundationModels, bedrock:InvokeModel"
  fi
else
  warn "AWS credentials not configured or expired."
  warn "Run 'aws configure' or set AWS_PROFILE before starting."
  warn "All use-cases require valid AWS credentials with Bedrock access."
fi

# ═══════════════════════════════════════════════════════════════════════
# Step 3: Install bedrock-smart-router package
# ═══════════════════════════════════════════════════════════════════════
step 3 "Installing bedrock-smart-router package (editable)..."

if $PYTHON -c "import bedrock_smart_router" &>/dev/null; then
  BSR_VERSION=$($PYTHON -c "import bedrock_smart_router; print(getattr(bedrock_smart_router, '__version__', 'dev'))" 2>/dev/null)
  ok "bedrock-smart-router already installed ($BSR_VERSION)"
else
  (cd "$PROJECT_ROOT" && $PYTHON -m pip install -e . --quiet 2>&1 | tail -3)
  if $PYTHON -c "import bedrock_smart_router" &>/dev/null; then
    ok "bedrock-smart-router installed"
  else
    fail "Failed to install bedrock-smart-router. Run: pip install -e . from project root"
  fi
fi

# ═══════════════════════════════════════════════════════════════════════
# Step 4: Install & verify backend Python dependencies
# ═══════════════════════════════════════════════════════════════════════
step 4 "Installing backend Python dependencies..."

if [ -f "$BACKEND_DIR/requirements.txt" ]; then
  $PYTHON -m pip install -r "$BACKEND_DIR/requirements.txt" --quiet 2>&1 | tail -3
  ok "Backend dependencies installed"
else
  fail "No requirements.txt found at $BACKEND_DIR/requirements.txt"
fi

# Verify critical imports (grouped by use-case)
MISSING_PKGS=""

# Core web framework
if ! $PYTHON -c "import fastapi, uvicorn, pydantic" &>/dev/null; then
  MISSING_PKGS="$MISSING_PKGS fastapi/uvicorn/pydantic"
fi

# AWS SDK
if ! $PYTHON -c "import boto3" &>/dev/null; then
  MISSING_PKGS="$MISSING_PKGS boto3"
fi

# Strands agents
if ! $PYTHON -c "import strands" &>/dev/null; then
  MISSING_PKGS="$MISSING_PKGS strands-agents"
fi

# MCP protocol
if ! $PYTHON -c "import mcp" &>/dev/null; then
  MISSING_PKGS="$MISSING_PKGS mcp"
fi

# Diagram generation
if ! $PYTHON -c "import diagrams" &>/dev/null; then
  MISSING_PKGS="$MISSING_PKGS diagrams"
  CAP_DIAGRAMS=false
fi

# Vector search (semantic cache)
if ! $PYTHON -c "import faiss" &>/dev/null; then
  MISSING_PKGS="$MISSING_PKGS faiss-cpu"
fi

# Charting
if ! $PYTHON -c "import matplotlib, pandas" &>/dev/null; then
  MISSING_PKGS="$MISSING_PKGS matplotlib/pandas"
fi

if [ -n "$MISSING_PKGS" ]; then
  fail "Missing Python packages:$MISSING_PKGS. Run: pip install -r $BACKEND_DIR/requirements.txt"
fi
ok "All Python imports verified (fastapi, boto3, strands, mcp, diagrams, faiss, matplotlib)"

# ═══════════════════════════════════════════════════════════════════════
# Step 5: Run prerequisites (databases + guardrail)
# ═══════════════════════════════════════════════════════════════════════
step 5 "Running prerequisites (databases + guardrail)..."

# Run the unified setup script
if $PYTHON "$DEMO_DIR/prerequisite/setup_all.py" 2>&1 | tail -5; then
  ok "Prerequisite setup completed"
else
  warn "Prerequisites partially failed (non-critical). Some use-cases may be degraded."
fi

# Verify Text2SQL database
TEXT2SQL_DB="$BACKEND_DIR/text2sql/demo.db"
if [ -f "$TEXT2SQL_DB" ]; then
  ok "Text2SQL database ready ($TEXT2SQL_DB)"
  CAP_TEXT2SQL=true
else
  # Check alternate locations
  ALT_DB=$($PYTHON -c "
import sys; sys.path.insert(0, '$BACKEND_DIR')
try:
    from text2sql.db import DB_PATH
    print(DB_PATH)
except: print('')
" 2>/dev/null)
  if [ -n "$ALT_DB" ] && [ -f "$ALT_DB" ]; then
    ok "Text2SQL database ready ($ALT_DB)"
    CAP_TEXT2SQL=true
  else
    warn "Text2SQL database not found. Use-case 5 (Text-to-SQL) may not work."
  fi
fi

# Verify Budget tracking database
BUDGET_DB="/tmp/bsr_budget.db"
if [ -f "$BUDGET_DB" ]; then
  ok "Budget tracking database ready ($BUDGET_DB)"
  CAP_BUDGET_DB=true
else
  warn "Budget database not found at $BUDGET_DB (will be created on first use)"
  CAP_BUDGET_DB=true  # Auto-created by the backend
fi

# Verify Guardrail config
GUARDRAIL_CONFIG="$DEMO_DIR/prerequisite/.guardrail_config.json"
if [ -f "$GUARDRAIL_CONFIG" ]; then
  GUARDRAIL_ID=$($PYTHON -c "import json; print(json.load(open('$GUARDRAIL_CONFIG')).get('guardrailIdentifier', 'unknown'))" 2>/dev/null)
  ok "Bedrock Guardrail configured (ID: $GUARDRAIL_ID)"
  CAP_GUARDRAILS=true
else
  warn "Guardrail config not found. Use-case 6 (Guardrails) will not work."
  warn "Re-run: python $DEMO_DIR/prerequisite/setup_guardrail.py"
fi

# ═══════════════════════════════════════════════════════════════════════
# Step 6: Prepare agent tools (MCP servers, diagram tool)
# ═══════════════════════════════════════════════════════════════════════
step 6 "Preparing agent tools..."

# Pre-warm MCP documentation server (uvx downloads ~60s on first run)
if [ "$CAP_UVX" = true ]; then
  echo -e "  Pre-warming AWS Documentation MCP server..."
  if uv tool install --force awslabs.aws-documentation-mcp-server --quiet 2>/dev/null; then
    ok "AWS Documentation MCP server cached"
    CAP_MCP_DOCS=true
  else
    warn "Failed to pre-warm AWS Documentation MCP server."
    warn "First use of Strands agent (use-case 3) may be slow."
    CAP_MCP_DOCS=true  # Will still try at runtime
  fi
else
  warn "Skipping MCP server setup (uvx not available)"
fi

# Verify native diagram tool
if [ "$CAP_DIAGRAMS" = true ]; then
  if $PYTHON -c "
from diagrams import Diagram
from diagrams.aws.compute import EC2
import tempfile, os
outfile = '/tmp/generated-diagrams/_preflight_test'
os.makedirs('/tmp/generated-diagrams', exist_ok=True)
with Diagram('test', filename=outfile, show=False):
    EC2('t')
assert os.path.exists(outfile + '.png')
os.remove(outfile + '.png')
" &>/dev/null; then
    ok "Native diagram tool verified (diagrams + graphviz working)"
  else
    warn "Diagram tool pre-flight check failed. Diagrams may not render."
    CAP_DIAGRAMS=false
  fi
fi

# Verify Strands agent can be constructed
if $PYTHON -c "
from strands import Agent
from strands.models import BedrockModel
" &>/dev/null; then
  ok "Strands Agent framework ready"
else
  warn "Strands Agent import failed. Use-case 3 will not work."
fi

# ═══════════════════════════════════════════════════════════════════════
# Step 7: Install and build frontend
# ═══════════════════════════════════════════════════════════════════════
step 7 "Installing and building frontend..."

# npm install
if [ ! -d "$FRONTEND_DIR/node_modules" ] || [ "$FRONTEND_DIR/package.json" -nt "$FRONTEND_DIR/node_modules/.package-lock.json" ]; then
  (cd "$FRONTEND_DIR" && npm install --silent 2>&1 | tail -3)
  ok "npm packages installed"
else
  ok "npm packages up to date"
fi

# Build
(cd "$FRONTEND_DIR" && npm run build 2>&1 | grep -E "(built|error)" | head -3)
if [ -f "$FRONTEND_DIR/dist/index.html" ]; then
  ok "Frontend built → dist/"
else
  fail "Frontend build failed. Run 'npm run build' in demo/frontend/ for details."
fi

# ═══════════════════════════════════════════════════════════════════════
# Step 8: Kill existing processes on required ports
# ═══════════════════════════════════════════════════════════════════════
step 8 "Killing existing processes on ports $BACKEND_PORT, $FRONTEND_PORT..."

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

# ═══════════════════════════════════════════════════════════════════════
# Step 9: Start servers
# ═══════════════════════════════════════════════════════════════════════
step 9 "Starting servers..."

# Backend
echo -e "  Starting backend (FastAPI) on port $BACKEND_PORT..."
(cd "$BACKEND_DIR" && $PYTHON -m uvicorn app:app --host 0.0.0.0 --port $BACKEND_PORT --log-level warning) &
BACKEND_PID=$!

# Frontend (dev server for HMR)
echo -e "  Starting frontend (Vite) on port $FRONTEND_PORT..."
(cd "$FRONTEND_DIR" && npm run dev -- --port $FRONTEND_PORT --host 2>/dev/null) &
FRONTEND_PID=$!

ok "Backend PID: $BACKEND_PID | Frontend PID: $FRONTEND_PID"

# ═══════════════════════════════════════════════════════════════════════
# Step 10: Health checks & capability summary
# ═══════════════════════════════════════════════════════════════════════
step 10 "Running health checks..."

# Backend health
RETRIES=0
MAX_RETRIES=25
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
MAX_RETRIES=15
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

# ── Capability Summary ──────────────────────────────────────────────
echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Capability Summary${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

cap_icon() { [ "$1" = true ] && echo -e "${GREEN}✓${NC}" || echo -e "${RED}✗${NC}"; }

echo -e "  $(cap_icon $CAP_AWS_CREDS) AWS Credentials          — Required for all use-cases"
echo -e "  $(cap_icon $CAP_BEDROCK) Bedrock Access            — Model invocation (InvokeModel)"
echo -e "  $(cap_icon $CAP_GUARDRAILS) Bedrock Guardrails        — Content safety (use-case 6)"
echo -e "  $(cap_icon $CAP_MCP_DOCS) AWS Docs MCP Server       — Documentation search (use-case 3)"
echo -e "  $(cap_icon $CAP_DIAGRAMS) Diagram Generation        — Architecture diagrams (use-case 3)"
echo -e "  $(cap_icon $CAP_TEXT2SQL) Text2SQL Database         — SQL generation (use-case 5)"
echo -e "  $(cap_icon $CAP_BUDGET_DB) Budget Tracking DB        — Cost enforcement (use-case 7)"
echo -e "  $(cap_icon $CAP_UVX) uvx (MCP launcher)        — Agent tool server management"

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
  # Clean up any orphans on our ports
  lsof -ti:$BACKEND_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
  lsof -ti:$FRONTEND_PORT 2>/dev/null | xargs kill -9 2>/dev/null || true
  echo -e "${GREEN}Stopped.${NC}"
  exit 0
}

trap cleanup INT TERM
wait
