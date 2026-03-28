FROM node:22-alpine

WORKDIR /app

# Copy backend and install its dependencies
COPY backend/package*.json ./backend/
RUN cd backend && npm install --omit=dev

# Copy frontend and install its dependencies
COPY frontend/package*.json ./frontend/
RUN cd frontend && npm install

# Copy the rest of the source
COPY backend/ ./backend/
COPY frontend/ ./frontend/
COPY engine/ ./engine/

# Build the frontend
RUN cd frontend && npm run build

EXPOSE 5173

# Run from frontend so the prod script can reach ../backend
WORKDIR /app/frontend
CMD ["npm", "run", "prod"]