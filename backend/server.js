const { WebSocketServer } = require('ws');
const { spawn } = require('child_process');
const path = require('path');
const readline = require('readline');

const isProd = process.env.NODE_ENV === 'production';
const frontendUrl = process.env.FRONTEND_URL;

if (isProd && !frontendUrl) {
  console.error('FRONTEND_URL must be set in production.');
  process.exit(1);
}

const verifyClient = isProd
  ? ({ origin }) => origin === frontendUrl
  : undefined;

const wss = new WebSocketServer({ host: '0.0.0.0', port: 8000, verifyClient });
console.log(`Aegis BMD Backend running on ws://localhost:8000 [${isProd ? 'production' : 'local'}]`);

const enginePath = path.join(__dirname, '../engine/ai_engine');
let engine;

try {
  engine = spawn(enginePath, [], { stdio: ['pipe', 'pipe', 'inherit'] });
  console.log('AI Engine process spawned.');
} catch(e) {
  console.error('Failed to spawn AI Engine. Did you compile it?', e);
  process.exit(1);
}

const rl = readline.createInterface({ input: engine.stdout });

rl.on('line', (line) => {
  if (!line.trim()) return;
  wss.clients.forEach((client) => {
    if (client.readyState === 1) {
      client.send(line);
    }
  });
});

engine.on('close', (code) => {
  console.log(`AI Engine process exited with code ${code}`);
});

wss.on('connection', (ws) => {
  console.log('Client connected to Aegis BMD WebSocket Console.');

  ws.on('message', (message) => {
    const data = message.toString();
    if (engine && !engine.killed) {
      engine.stdin.write(data + '\n');
    }
  });

  ws.on('close', () => {
    console.log('Client disconnected.');
  });
});