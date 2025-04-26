FROM node:22-bullseye-slim

WORKDIR /app

COPY  package*.json ./
RUN npm install 

COPY . .

RUN apt-get update && apt-get install -y \
    python3 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

EXPOSE 5000

CMD ["npm", "start"]