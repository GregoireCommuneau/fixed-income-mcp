# syntax=docker/dockerfile:1

# ─────────────────────────────────────────────────────────────
# Stage 1 — builder: install the package + deps into a venv
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Copy only what's needed to build/install the package.
COPY pyproject.toml ./
COPY macro_quant ./macro_quant

# Isolated venv with the runtime dependencies only (no dev tools).
RUN python -m venv /venv \
    && /venv/bin/pip install --no-cache-dir --upgrade pip \
    && /venv/bin/pip install --no-cache-dir .

# ─────────────────────────────────────────────────────────────
# Stage 2 — runtime: slim final image (just the venv + code)
# ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Run as a non-root user (security best practice).
RUN useradd --create-home --uid 1000 appuser

WORKDIR /app

# Bring the ready-made venv from the builder and the application code.
COPY --from=builder /venv /venv
COPY --chown=appuser:appuser macro_quant ./macro_quant

ENV PATH="/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg

USER appuser

# Default: start the MCP server. It needs a FRED API key at runtime, e.g.:
#   docker run -e FRED_API_KEY=xxxx macroquant
CMD ["python", "-m", "macro_quant.main", "--serve"]
