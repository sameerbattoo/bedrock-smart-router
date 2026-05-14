#!/bin/bash
# Start both backend and frontend for the demo

echo "Starting Bedrock Smart Router Demo..."
echo ""

# Start backend
echo "→ Starting backend (FastAPI) on port 8000..."
cd "$(dirname "$0")"
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
echo "  Backend PID: $BACKEND_PID"

# Wait for backend to be ready
sleep 2

# Start frontend
echo "→ Starting frontend (Vite) on port 5173..."
cd frontend
npm run dev &
FRONTEND_PID=$!
echo "  Frontend PID: $FRONTEND_PID"

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Demo running!"
echo "  Frontend: http://localhost:5173"
echo "  Backend:  http://localhost:8000"
echo "  API docs: http://localhost:8000/docs"
echo "═══════════════════════════════════════════════════"
echo ""
echo "Press Ctrl+C to stop both servers."

# Trap Ctrl+C to kill both
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait
