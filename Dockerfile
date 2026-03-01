FROM python:3.12-slim

WORKDIR /app

# Heavy/stable deps — cached layer unless requirements-heavy.txt changes
COPY requirements-heavy.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    --mount=type=cache,target=/tmp/wheels \
    pip install --find-links /tmp/wheels -r requirements-heavy.txt \
    && pip wheel -r requirements-heavy.txt -w /tmp/wheels 2>/dev/null || true

# Light/frequently changing deps
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY . .

EXPOSE 8000

ENV PYTHONPATH=.

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
