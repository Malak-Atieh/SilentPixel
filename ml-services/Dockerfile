#building stage
FROM python:3.9-bullseye AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirments.txt

#runtime stage
FROM python:3.9-bullseye-slim

WORKDIR /app

COPY --from=builder /usr/local/lib/python3.9 /usr/local/lib/python3.9
COPY --from=builder /usr/local/bin /usr/local/bin
RUN apt-get update && apt-get install -y libgl1 && rm -rf /var/lib/apt/lists/*
COPY . . 

EXPOSE 6000

CMD ["python", "app.py"]
