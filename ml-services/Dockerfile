FROM python:3.13-bullseye

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirments.txt

COPY . . 

RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*
    
EXPOSE 6000

CMD ["python", "app.py"]
