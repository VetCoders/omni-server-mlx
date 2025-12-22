#!/bin/bash
# Start MLX Omni Server on port 8100 for LibraxisAI integration
#
# Created by M&K (c)2025 The LibraxisAI Team

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Load .env if exists (using set -a/+a for robust parsing)
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Default values
PORT=${MLX_OMNI_PORT:-8100}
HOST=${MLX_OMNI_HOST:-0.0.0.0}
LOG_LEVEL=${MLX_OMNI_LOG_LEVEL:-info}
CORS=${MLX_OMNI_CORS:-*}

echo "Starting MLX Omni Server on http://${HOST}:${PORT}"
echo "  CORS: ${CORS}"
echo "  Log level: ${LOG_LEVEL}"

# Run with uv
exec uv run python -m mlx_omni_server.main \
    --host "$HOST" \
    --port "$PORT" \
    --log-level "$LOG_LEVEL" \
    --cors-allow-origins "$CORS"
