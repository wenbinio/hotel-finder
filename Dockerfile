# Build the React frontend in a Node stage, then copy the bundle into the
# Python image so a single container ships both the API and the SPA.
FROM node:22-slim AS frontend
WORKDIR /frontend
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund || npm install --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py parsing.py ta_keys.json ./
# Vite outputs the bundle to ../static (relative to /frontend), which means
# /static at the repo root in the Node stage.
COPY --from=frontend /static ./static

# Cloud Run injects $PORT (8080) and ignores EXPOSE. Shell form lets gunicorn
# pick it up; locally `docker run -p 8080:8080` works without setting PORT.
EXPOSE 8080
CMD gunicorn --bind 0.0.0.0:${PORT:-8080} --workers 2 --timeout 120 app:app
