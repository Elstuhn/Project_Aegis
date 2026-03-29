FROM node:22-alpine

WORKDIR /app

COPY backend/package*.json ./backend/
RUN cd backend && npm install --omit=dev

COPY frontend/package*.json ./frontend/
RUN cd frontend && npm install

COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY engine/ ./engine/

RUN cd engine && rm -f ai_engine && gcc -o ai_engine ai_engine.c -lm
RUN chmod +x /app/engine/ai_engine
RUN cd frontend && npm run build

EXPOSE 5173

WORKDIR /app/frontend
CMD ["npm", "run", "prod"]