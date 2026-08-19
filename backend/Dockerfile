FROM python:3.11-slim

WORKDIR /app

# backend-only deps (avoid the heavy CV/tensorflow stack)
COPY _deps_check.txt /tmp/deps.txt
RUN pip install --no-cache-dir -r /tmp/deps.txt

# copy backend source
COPY . /app/

# install backend as importable
ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
