#!/bin/bash

# Exit immediately on error
set -e

# Source environment variables
source .env

# Activate virtual environment
source venv/bin/activate

# Ensure required environment variables are set
echo "Checking NEBIUS_API_KEY..."
if [ -z "$NEBIUS_API_KEY" ]; then
  echo "Error: NEBIUS_API_KEY is not set. Please add it to .env or export it before running this script." >&2
  exit 1
fi

# Start FastAPI backend
uvicorn app:api --host 0.0.0.0 --port 8000 &

# Wait a bit to ensure backend is up
sleep 2

# Start Gradio UI
python gradio_ui.py
