# 🐳 Docker Setup Guide

Complete guide for running the NBA Prediction System in Docker containers.

## 📋 Prerequisites

- **Docker**: Version 20.10+ ([Install Docker](https://docs.docker.com/get-docker/))
- **Docker Compose**: Version 2.0+ (included with Docker Desktop)
- **2GB+ RAM**: Recommended for Chrome/Selenium
- **5GB+ Disk**: For images and data

## 🚀 Quick Start

### 1. Clone and Configure

```bash
# Clone repository
git clone https://github.com/razi90/Basketball_prediction.git
cd Basketball_prediction

# Copy environment template
cp .env.example .env

# Edit .env with your API keys
nano .env  # or vim, code, etc.
```

**Required in `.env`:**
```bash
ODDS_API_KEY=your_odds_api_key_here
USE_DATABASE=false  # or true for PostgreSQL
```

### 2. Run Dashboard Only (Simplest)

```bash
# Build and run just the dashboard
docker-compose up dashboard

# Access at: http://localhost:8501
```

### 3. Run Full Stack (Dashboard + Database)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f dashboard
```

### 4. Run Prediction Scripts

```bash
# Run a specific script
docker-compose run --rm app python 2026/src/3_predict_games_hybrid_2026.py

# Or enter interactive shell
docker-compose run --rm app bash
```

## 📦 Service Architecture

```
┌─────────────────────────────────────────────────────┐
│  docker-compose.yml                                 │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │
│  │  postgres    │  │  dashboard   │  │   app    │ │
│  │  (optional)  │  │  (Streamlit) │  │ (scripts)│ │
│  │  Port: 5432  │  │  Port: 8501  │  │          │ │
│  └──────────────┘  └──────────────┘  └──────────┘ │
│         │                  │                │       │
│         └──────────────────┴────────────────┘       │
│                   nba_network                       │
└─────────────────────────────────────────────────────┘
```

## 🎯 Common Use Cases

### Case 1: Development (Hot Reload)

```bash
# Run dashboard with live code reload
docker-compose up dashboard

# Edit files locally - changes auto-reload
```

### Case 2: Daily Predictions (Automated)

```bash
# Run full prediction pipeline
docker-compose run --rm app bash -c "
  python 2026/src/1_get_data_previous_game_day_2026.py &&
  python 2026/src/2_get_data_next_game_day_2026.py &&
  python 2026/src/3_predict_games_hybrid_2026.py &&
  python 2026/src/5_enrich_with_betting_recs_2026.py
"
```

### Case 3: Production (All Services)

```bash
# Start everything in background
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f

# Stop all
docker-compose down
```

### Case 4: Database Setup

```bash
# Start database only
docker-compose up -d postgres

# Wait for it to be healthy
docker-compose ps postgres

# Run migration scripts
docker-compose run --rm app python database/scripts/migrate_game_statistics.py
```

## 🔧 Docker Commands Reference

### Building

```bash
# Build all images
docker-compose build

# Build specific service
docker-compose build dashboard

# Force rebuild (no cache)
docker-compose build --no-cache

# Build specific stage
docker build --target dashboard -t nba-dashboard .
```

### Running

```bash
# Start all services
docker-compose up

# Start in background (detached)
docker-compose up -d

# Start specific service
docker-compose up dashboard

# Run one-off command
docker-compose run --rm app python script.py

# Interactive shell
docker-compose run --rm app bash
```

### Debugging

```bash
# View logs (all services)
docker-compose logs

# Follow logs (live)
docker-compose logs -f dashboard

# View logs for specific service
docker-compose logs app

# Inspect running container
docker-compose exec dashboard bash

# Check resource usage
docker stats
```

### Cleanup

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (data loss!)
docker-compose down -v

# Remove all images
docker-compose down --rmi all

# Clean up everything
docker system prune -a
```

## 🗄️ Data Persistence

### Volume Mounts

Data is persisted using Docker volumes:

```yaml
volumes:
  - ./2026/data:/app/2026/data       # Game data (CSVs)
  - ./2026/output:/app/2026/output   # Predictions
  - ./logs:/app/logs                 # Application logs
  - postgres_data:/var/lib/postgresql/data  # Database
```

**This means:**
- ✅ Data survives container restarts
- ✅ You can edit CSVs from host machine
- ✅ Dashboard sees real-time file updates
- ⚠️ Use `docker-compose down -v` cautiously (deletes data!)

### Backup Data

```bash
# Backup prediction data
tar -czf backup-$(date +%Y%m%d).tar.gz 2026/data 2026/output

# Backup database
docker-compose exec postgres pg_dump -U nba_user nba_predictions > backup.sql

# Restore database
docker-compose exec -T postgres psql -U nba_user nba_predictions < backup.sql
```

## 🌐 Environment Variables

### Required

```bash
ODDS_API_KEY=your_api_key_here
```

### Optional

```bash
# Database
USE_DATABASE=false
DATABASE_URL=postgresql://user:pass@postgres:5432/nba_predictions
POSTGRES_PASSWORD=changeme123

# Application
LOG_LEVEL=INFO
ROLLING_WINDOW_SIZE=9

# Dashboard
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=0.0.0.0
```

### Override in docker-compose

```yaml
services:
  dashboard:
    environment:
      - USE_DATABASE=true
      - LOG_LEVEL=DEBUG
```

## 🔐 Security Best Practices

### 1. Never Commit Secrets

```bash
# .env file is in .gitignore
git status  # Should not show .env
```

### 2. Use Strong Database Password

```bash
# Generate random password
openssl rand -base64 32 > .db_password
export POSTGRES_PASSWORD=$(cat .db_password)
```

### 3. Network Isolation

```yaml
# Services only accessible within nba_network
# Dashboard exposed on localhost only
ports:
  - "127.0.0.1:8501:8501"  # Localhost only
  # - "8501:8501"  # ❌ Exposed to network
```

### 4. Read-Only Filesystem (Production)

```yaml
services:
  dashboard:
    read_only: true
    tmpfs:
      - /tmp
      - /app/.streamlit
```

## 🚀 Deployment

### Deploy to Cloud

#### Docker Hub

```bash
# Build and tag
docker build -t username/nba-dashboard:latest .

# Push to Docker Hub
docker push username/nba-dashboard:latest

# Pull on server
docker pull username/nba-dashboard:latest
```

#### AWS ECS

```bash
# Install AWS CLI
aws configure

# Create ECR repository
aws ecr create-repository --repository-name nba-prediction

# Build and push
docker build -t nba-prediction .
docker tag nba-prediction:latest 123456789.dkr.ecr.us-east-1.amazonaws.com/nba-prediction:latest
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/nba-prediction:latest
```

#### Google Cloud Run

```bash
# Build and deploy
gcloud run deploy nba-dashboard \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Scheduled Runs (Cron)

```bash
# Add to crontab
0 9 * * * cd /path/to/project && docker-compose run --rm app python 2026/src/3_predict_games_hybrid_2026.py
```

## 🐞 Troubleshooting

### Issue: Chrome/Selenium Not Working

```bash
# Check Chrome installation
docker-compose run --rm app google-chrome --version

# Check ChromeDriver
docker-compose run --rm app chromedriver --version

# Run with visible browser (for debugging)
docker-compose run --rm -e DISPLAY=$DISPLAY app python script.py
```

### Issue: Database Connection Failed

```bash
# Check database is running
docker-compose ps postgres

# Check health
docker-compose exec postgres pg_isready -U nba_user

# View database logs
docker-compose logs postgres

# Connect manually
docker-compose exec postgres psql -U nba_user -d nba_predictions
```

### Issue: Dashboard Won't Start

```bash
# Check logs
docker-compose logs dashboard

# Rebuild image
docker-compose build --no-cache dashboard

# Test locally first
pip install streamlit
streamlit run dashboard/app.py
```

### Issue: Out of Memory

```bash
# Check resource usage
docker stats

# Increase Docker memory limit (Docker Desktop)
# Settings > Resources > Memory > 4GB+

# Or limit specific service
docker-compose.yml:
  services:
    app:
      mem_limit: 2g
```

### Issue: Port Already in Use

```bash
# Find process using port 8501
lsof -i :8501

# Kill process
kill -9 <PID>

# Or use different port
docker-compose.yml:
  ports:
    - "8502:8501"  # Use 8502 instead
```

## 📊 Performance Optimization

### Multi-Stage Builds

The Dockerfile uses multi-stage builds for smaller images:

```dockerfile
FROM python:3.11-slim as base      # Base dependencies
FROM base as development           # Development tools
FROM base as dashboard             # Dashboard only
FROM base as production            # Optimized runtime
```

Benefits:
- **Smaller images**: 500MB vs 2GB
- **Faster builds**: Cached layers
- **Security**: Fewer attack surfaces

### Build Cache

```bash
# Use BuildKit for faster builds
export DOCKER_BUILDKIT=1
docker-compose build

# Share cache between builds
docker build --cache-from nba-dashboard:latest .
```

## 🧪 Testing

```bash
# Run tests in container
docker-compose run --rm app pytest tests/ -v

# With coverage
docker-compose run --rm app pytest tests/ --cov=2026/src --cov-report=html

# Run specific test
docker-compose run --rm app pytest tests/test_betting_utils.py -v
```

## 📚 Additional Resources

- **Docker Docs**: https://docs.docker.com
- **Docker Compose**: https://docs.docker.com/compose/
- **Streamlit Docker**: https://docs.streamlit.io/knowledge-base/tutorials/deploy/docker
- **PostgreSQL Docker**: https://hub.docker.com/_/postgres

## 🤝 Contributing

When adding Docker features:
1. Test locally with `docker-compose up`
2. Document in this file
3. Update .dockerignore if needed
4. Keep images small (use multi-stage builds)

## 📝 License

Same license as main project (MIT)
