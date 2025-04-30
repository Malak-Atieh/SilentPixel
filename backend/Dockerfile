#building stage
FROM node:20-bullseye-slim AS builder

WORKDIR /app

COPY  package*.json ./
RUN npm install ci --omit=dev

#runtime stage
FROM node:20-bullseye-slim

WORKDIR /app
COPY --from=builder /app/node_modules ./node_modules

COPY . .

EXPOSE 5000

CMD ["npm", "start"]