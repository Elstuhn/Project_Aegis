# AEGIS BMD
Intercontinental ballistic missile interception simulation platform

### Interceptor Missile Training Methods
1. GRPO
2. Evolution Algorithm (OpenAI Evolution Strategies)
3. PPO

Optimized C++ runtime for training (7k training episodes in 10s)

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

### Stats

| Method | Episodes | PK |
| -------- | -------- | -------- | 
| GRPO  | ~500m  | 25% | 
| PPO  | ~70k  | 94% |
| ES  | ~120k  | 68% |

All using default parameters (not reflecting highest PK gotten, just rough estimates)

<img width="1861" height="963" alt="image" src="https://github.com/user-attachments/assets/98b46cfb-9259-4d77-90ea-429acc602025" />


