# Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Use official Python 3.11 slim image
# WHY 3.11 not 3.13: better library compatibility, widely tested in production
# WHY slim: removes unnecessary OS packages — smaller image, faster deploy
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# ─────────────────────────────────────────────────────────────────────────────
# Copy requirements FIRST — before copying code
# WHY: Docker layer caching. If requirements.txt hasn't changed,
# this layer is cached. Only code changes trigger a rebuild from here.
# Without this order: every code change reinstalls all packages (slow).
# ─────────────────────────────────────────────────────────────────────────────
COPY requirements.txt .

# Install dependencies
# WHY --no-cache-dir: don't store pip cache in the image — smaller size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Create data directories
# WHY: these directories must exist for fetcher and trainer to write files
RUN mkdir -p data/raw data/processed

# ─────────────────────────────────────────────────────────────────────────────
# Expose port 8000
# WHY: documents which port the app listens on — Render reads this
# ─────────────────────────────────────────────────────────────────────────────
EXPOSE 8000

# ─────────────────────────────────────────────────────────────────────────────
# Start the FastAPI server
# WHY host 0.0.0.0: listens on all network interfaces
#   Without this, the server only accepts localhost connections
#   Inside a container, "localhost" is isolated — external traffic
#   comes through the network interface, not loopback
# WHY no --reload: reload is for development only
#   In production, reloading on file change is unnecessary and wastes resources
# ─────────────────────────────────────────────────────────────────────────────
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]