# AEGIS BMD
Intercontinental ballistic missile interception simulation platform

### Interceptor Missile Training Methods
1. GRPO
2. Evolution Algorithm (OpenAI Evolution Strategies)
3. PPO

Optimized C++ runtime for training (146,000+ training episodes in 5 seconds from ES)

### How to run:
1. Compile engine:
```
cd engine
rm -f ai_engine
g++ -O3 ai_engine.cpp -o ai_engine
```
2. Start backend server
```
cd ../backend
node server.js
```
3. Start frontend server
```
cd frontend
npm run install
npm run dev
```

