# ─── Base image ───────────────────────────────────────────────────────────
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    # suppress TF AVX / CUDA spam (optional)
    TF_CPP_MIN_LOG_LEVEL=3 \
    # FastAPI/Torch/HF cache (writable)
    TRANSFORMERS_CACHE=/tmp/hf_cache

WORKDIR /code

# ─── Install Python deps ──────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ─── Copy the rest (app.py, zip, README, etc.) ────────────────────────────
COPY . .

# ─── Expose default port & launch ─────────────────────────────────────────
EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
