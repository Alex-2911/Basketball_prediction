#!/bin/bash
# Docker entrypoint script for NBA Prediction System

set -e

echo "🏀 NBA Prediction System - Docker Container Starting"

# Create necessary directories
mkdir -p /app/2026/data /app/2026/output /app/logs

# Check if .env file exists
if [ ! -f /app/.env ]; then
    echo "⚠️  Warning: .env file not found. Creating from .env.example..."
    if [ -f /app/.env.example ]; then
        cp /app/.env.example /app/.env
        echo "✅ Created .env from .env.example"
    else
        echo "⚠️  No .env.example found. You may need to set environment variables manually."
    fi
fi

# Wait for database if USE_DATABASE is true
if [ "${USE_DATABASE}" = "true" ]; then
    echo "⏳ Waiting for database to be ready..."

    # Extract database host from DATABASE_URL or use default
    DB_HOST=${DATABASE_HOST:-postgres}
    DB_PORT=${DATABASE_PORT:-5432}

    # Wait for PostgreSQL to be ready
    until nc -z "$DB_HOST" "$DB_PORT" 2>/dev/null; do
        echo "   Database is unavailable - sleeping"
        sleep 2
    done

    echo "✅ Database is ready!"
fi

# Display environment info
echo "📊 Environment Configuration:"
echo "   USE_DATABASE: ${USE_DATABASE:-false}"
echo "   ODDS_API_KEY: ${ODDS_API_KEY:+***SET***}"
echo "   Python Version: $(python --version)"
echo ""

# Execute the command passed to docker run
echo "🚀 Executing: $@"
exec "$@"
