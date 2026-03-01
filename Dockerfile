FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY . .

EXPOSE 8000

ENV PYTHONPATH=.

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
