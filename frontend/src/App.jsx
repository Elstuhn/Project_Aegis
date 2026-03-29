import { useState, useEffect, useRef, useCallback } from "react";

const W = 880, H = 480;
const TWO_PI = Math.PI * 2;

function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

export default function AegisBMD() {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);
  const scanRef = useRef(0);
  const wsRef = useRef(null);
  const isUpdatingRef = useRef(false);

  const [icbmSpeed, setIcbmSpeed] = useState(7);
  const [maxAlt, setMaxAlt] = useState(360);
  const [launchX, setLaunchX] = useState(60);
  const [targetX, setTargetX] = useState(780);
  const [radarRange, setRadarRange] = useState(400);
  const [interceptSpeed, setInterceptSpeed] = useState(14);
  const [launchDelay, setLaunchDelay] = useState(4);
  const [groupSize, setGroupSize] = useState(8);
  const [clipRatio, setClipRatio] = useState(0.2);
  const [klCoeff, setKlCoeff] = useState(0.04);
  const [grpoLr, setGrpoLr] = useState(0.003);
  const [algo, setAlgo] = useState("GRPO");
  const [randomize, setRandomize] = useState(true);

  const [training, setTraining] = useState(false);
  const [stats, setStats] = useState({ episodes: 0, hits: 0, avgReward: 0, rewardHistory: [], hitRateHistory: [] });
  const [playing, setPlaying] = useState(false);
  const [playFrame, setPlayFrame] = useState(0);
  const [playData, setPlayData] = useState(null);
  const [lastHit, setLastHit] = useState(null);
  const [panel, setPanel] = useState("threat");
  const [wsConnected, setWsConnected] = useState(false);

  const icbmP = { launchX, targetX, maxAlt, speed: icbmSpeed };
  const defP = { radarRange, interceptSpeed, launchDelay };
  const optParams = { algo, randomize, groupSize, clipRatio, klCoeff, lr: grpoLr, sigma: clipRatio };

  useEffect(() => {
    const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = import.meta.env.PROD
      ? `${wsProtocol}://${window.location.hostname}:8000`
      : "ws://localhost:8000";

    const ws = new WebSocket(wsUrl);
    //const ws = new WebSocket("ws://localhost:8000");
    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setWsConnected(false);
    ws.onmessage = (e) => {
      try {
        const msg = JSON.parse(e.data);
        if (msg.type === "update") {
          setStats(msg.stats);
          isUpdatingRef.current = false;
        } else if (msg.type === "rollout") {
          setPlayData(msg);
          setPlayFrame(0);
          setPlaying(true);
          setLastHit(msg.hit);
        } else if (msg.type === "reset") {
          setStats({ episodes: 0, hits: 0, avgReward: 0, rewardHistory: [], hitRateHistory: [] });
          setPlayData(null);
          setLastHit(null);
        }
      } catch (err) {
        console.error("Parse err:", err);
      }
    };
    wsRef.current = ws;
    return () => ws.close();
  }, []);

  useEffect(() => {
    if (!training || !wsConnected) return;
    const id = setInterval(() => {
      if (isUpdatingRef.current) return;
      isUpdatingRef.current = true;
      wsRef.current.send(JSON.stringify({ cmd: "update", icbmP, defP, opt: optParams }));
    }, 10);
    return () => clearInterval(id);
  }, [training, wsConnected, icbmP, defP, optParams]);

  const playEpisode = useCallback(() => {
    if (!wsConnected || training) return;
    wsRef.current.send(JSON.stringify({ cmd: "rollout", icbmP, defP, opt: optParams }));
  }, [wsConnected, training, icbmP, defP, optParams]);

  const resetAgent = () => {
    if (!wsConnected || training) return;
    wsRef.current.send(JSON.stringify({ cmd: "reset" }));
  };

  useEffect(() => {
    if (!playing || !playData) return;
    const mx = Math.max(playData.traj.length, playData.trail.length);
    const id = setInterval(() => {
      setPlayFrame(f => { if (f >= mx - 1) { setPlaying(false); return f; } return f + 1; });
    }, 18);
    return () => clearInterval(id);
  }, [playing, playData]);

  useEffect(() => {
    const cvs = canvasRef.current;
    if (!cvs) return;
    const ctx = cvs.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    cvs.width = W * dpr; cvs.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const now = Date.now();

    ctx.fillStyle = "#040a05";
    ctx.fillRect(0, 0, W, H);
    ctx.globalAlpha = 0.025;
    for (let y = 0; y < H; y += 2) { ctx.fillStyle = y % 4 === 0 ? "#081208" : "#030803"; ctx.fillRect(0, y, W, 1); }
    ctx.globalAlpha = 1;

    ctx.strokeStyle = "rgba(0,70,15,0.1)";
    ctx.lineWidth = 0.5;
    for (let x = 0; x < W; x += 40) { ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke(); }
    for (let y = 0; y < H; y += 40) { ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke(); }

    ctx.font = "bold 9px 'Courier New', monospace";
    ctx.fillStyle = "rgba(0,180,45,0.4)";
    for (let x = 80; x < W; x += 80) ctx.fillText(x, x - 6, H - 4);

    const groundY = H - 30;
    const toS = (p) => ({ x: p.x, y: groundY - p.y * ((groundY - 18) / (maxAlt + 40)) });

    ctx.beginPath();
    ctx.strokeStyle = "rgba(0,160,35,0.25)";
    ctx.lineWidth = 0.8;
    ctx.moveTo(0, groundY);
    for (let x = 0; x < W; x += 3) {
      ctx.lineTo(x, groundY + Math.sin(x * 0.019) * 2.5 + Math.sin(x * 0.006) * 4);
    }
    ctx.stroke();
    const gG = ctx.createLinearGradient(0, groundY, 0, H);
    gG.addColorStop(0, "rgba(0,35,8,0.5)"); gG.addColorStop(1, "rgba(0,12,3,0.8)");
    ctx.fillStyle = gG; ctx.fillRect(0, groundY, W, 30);

    const rc = toS({ x: targetX, y: 0 });
    const rr = radarRange * ((groundY - 18) / (maxAlt + 40)) * 0.65;
    for (let i = 1; i <= 3; i++) {
      ctx.beginPath(); ctx.arc(rc.x, rc.y, rr * i / 3, -Math.PI, 0);
      ctx.strokeStyle = `rgba(0,${80 + i * 25},20,${0.06 + i * 0.025})`;
      ctx.lineWidth = 0.6; ctx.setLineDash([2, 5]); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.font = "bold 9px 'Courier New', monospace";
    for (let i = 1; i <= 3; i++) {
      ctx.fillStyle = "rgba(0,255,80,0.4)";
      ctx.fillText(`${Math.round(radarRange * i / 3)}`, rc.x + rr * i / 3 + 2, rc.y - 4);
    }

    const sa = scanRef.current;
    ctx.save();
    ctx.beginPath(); ctx.moveTo(rc.x, rc.y);
    ctx.arc(rc.x, rc.y, rr, -Math.PI + sa, -Math.PI + sa + 0.3);
    ctx.closePath();
    const sG = ctx.createRadialGradient(rc.x, rc.y, 0, rc.x, rc.y, rr);
    sG.addColorStop(0, "rgba(0,255,50,0.12)"); sG.addColorStop(1, "rgba(0,255,50,0)");
    ctx.fillStyle = sG; ctx.fill();
    ctx.restore();

    const ls = toS({ x: launchX, y: 0 });
    ctx.strokeStyle = "rgba(255,35,25,0.5)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(ls.x, groundY - 14); ctx.lineTo(ls.x - 4, groundY); ctx.lineTo(ls.x + 4, groundY); ctx.closePath(); ctx.stroke();
    ctx.fillStyle = "rgba(255,35,25,0.1)"; ctx.fill();
    ctx.font = "bold 9px 'Courier New', monospace"; ctx.fillStyle = "rgba(255,55,35,0.8)";
    ctx.fillText("HOSTILE ORIGIN", ls.x - 24, groundY + 16);

    ctx.strokeStyle = "rgba(30,180,100,0.5)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(rc.x, groundY - 14); ctx.lineTo(rc.x - 5, groundY); ctx.lineTo(rc.x + 5, groundY); ctx.closePath(); ctx.stroke();
    ctx.fillStyle = "rgba(30,180,100,0.1)"; ctx.fill();
    ctx.fillStyle = "rgba(0,255,80,0.8)";
    ctx.font = "bold 9px 'Courier New', monospace";
    ctx.fillText("EI Radar HQ", rc.x - 24, groundY + 16);

    if (playData) {
      const { traj, trail, startIdx } = playData;
      const fT = Math.min(playFrame, traj.length - 1);
      const fI = Math.max(0, Math.min(playFrame - startIdx, trail.length - 1));

      ctx.beginPath(); ctx.strokeStyle = "rgba(255,35,25,0.05)"; ctx.lineWidth = 0.7; ctx.setLineDash([2, 4]);
      for (let i = 0; i < traj.length; i++) { const s = toS(traj[i]); i === 0 ? ctx.moveTo(s.x, s.y) : ctx.lineTo(s.x, s.y); }
      ctx.stroke(); ctx.setLineDash([]);

      const tLen = 25;
      ctx.beginPath(); ctx.strokeStyle = "rgba(255,70,25,0.5)"; ctx.lineWidth = 1.6;
      for (let i = Math.max(0, fT - tLen); i <= fT; i++) { const s = toS(traj[i]); i === Math.max(0, fT - tLen) ? ctx.moveTo(s.x, s.y) : ctx.lineTo(s.x, s.y); }
      ctx.stroke();

      if (fT < traj.length) {
        const wp = toS(traj[fT]);
        const blink = Math.sin(now * 0.012) > 0;
        if (blink) { ctx.beginPath(); ctx.arc(wp.x, wp.y, 6, 0, TWO_PI); ctx.strokeStyle = "rgba(255,45,25,0.7)"; ctx.lineWidth = 1.2; ctx.stroke(); }
        ctx.beginPath(); ctx.arc(wp.x, wp.y, 2.2, 0, TWO_PI); ctx.fillStyle = "#ff2818"; ctx.fill();
        ctx.strokeStyle = "rgba(255,80,50,0.4)"; ctx.lineWidth = 0.5;
        ctx.strokeRect(wp.x - 8, wp.y - 8, 16, 16);
        ctx.font = "bold 9px 'Courier New', monospace"; ctx.fillStyle = "rgba(255,100,60,0.95)";
        ctx.fillText(`TGT-001`, wp.x + 14, wp.y - 12);
        ctx.fillStyle = "rgba(255,100,60,0.8)";
        ctx.fillText(`ALT ${traj[fT].y.toFixed(0)}  SPD ${icbmSpeed}`, wp.x + 14, wp.y);
        ctx.fillText(`RNG ${dist(traj[fT], { x: targetX, y: 0 }).toFixed(0)}`, wp.x + 14, wp.y + 10);
      }

      if (fI > 0) {
        ctx.beginPath(); ctx.strokeStyle = "rgba(35,210,245,0.4)"; ctx.lineWidth = 1.4;
        for (let i = Math.max(0, fI - tLen); i <= Math.min(fI, trail.length - 1); i++) {
          const s = toS(trail[i]); i === Math.max(0, fI - tLen) ? ctx.moveTo(s.x, s.y) : ctx.lineTo(s.x, s.y);
        }
        ctx.stroke();

        if (fI < trail.length) {
          const ip = toS(trail[fI]);
          ctx.beginPath(); ctx.arc(ip.x, ip.y, 5, 0, TWO_PI); ctx.strokeStyle = "rgba(35,210,245,0.6)"; ctx.lineWidth = 0.8; ctx.stroke();
          ctx.beginPath(); ctx.arc(ip.x, ip.y, 1.8, 0, TWO_PI); ctx.fillStyle = "#22daf5"; ctx.fill();
          ctx.strokeStyle = "rgba(35,210,245,0.3)"; ctx.strokeRect(ip.x - 7, ip.y - 7, 14, 14);
          ctx.font = "bold 9px 'Courier New', monospace"; ctx.fillStyle = "rgba(35,220,255,0.95)";
          ctx.fillText("INT-001 ENGAG", ip.x + 12, ip.y - 8);
        }
      }

      if (!playing && playData) {
        if (lastHit) {
          const hp = toS(trail[trail.length - 1]);
          for (let r = 0; r < 4; r++) {
            ctx.beginPath(); ctx.arc(hp.x, hp.y, 6 + r * 9, 0, TWO_PI);
            ctx.strokeStyle = `rgba(255,${190 - r * 35},30,${0.45 - r * 0.1})`; ctx.lineWidth = 1.8 - r * 0.3; ctx.stroke();
          }
          ctx.beginPath(); ctx.arc(hp.x, hp.y, 4, 0, TWO_PI); ctx.fillStyle = "#ffbb00"; ctx.fill();
        } else {
          const tp = toS({ x: targetX, y: 0 });
          for (let r = 0; r < 3; r++) {
            ctx.beginPath(); ctx.arc(tp.x, tp.y - 3, 5 + r * 7, 0, TWO_PI);
            ctx.strokeStyle = `rgba(255,35,25,${0.45 - r * 0.12})`; ctx.lineWidth = 1.5; ctx.stroke();
          }
        }
      }
    }

    ctx.fillStyle = "rgba(0,12,4,0.88)"; ctx.fillRect(0, 0, W, 20);
    ctx.strokeStyle = "rgba(0,160,35,0.12)"; ctx.lineWidth = 0.5;
    ctx.beginPath(); ctx.moveTo(0, 20); ctx.lineTo(W, 20); ctx.stroke();
    ctx.font = "bold 10px 'Courier New', monospace";
    const dot = training ? "●" : "○";
    ctx.fillStyle = training ? "rgba(0,255,50,0.95)" : "rgba(140,140,140,0.5)";
    ctx.fillText(dot, 10, 14);
    ctx.fillStyle = "rgba(0,255,80,0.8)";
    ctx.fillText("ELSTON INDUSTRIES   AEGIS BMD   ICBM ENGAGEMENT SYSTEM", 22, 14);
    ctx.fillStyle = "rgba(0,255,80,0.6)";
    ctx.textAlign = "right";
    ctx.fillText(wsConnected ? "DATALINK ACTIVE" : "DATALINK OFFLINE", W - 180, 14);
    ctx.fillText(`EP:${stats.episodes}  HITS:${stats.hits}  Pk:${stats.episodes > 0 ? (stats.hits / stats.episodes * 100).toFixed(1) : 0}%`, W - 12, 14);
    ctx.textAlign = "left";

    if (!playing && playData) {
      ctx.fillStyle = lastHit ? "rgba(0,35,8,0.88)" : "rgba(35,4,4,0.88)";
      ctx.fillRect(W / 2 - 120, 25, 240, 30);
      ctx.strokeStyle = lastHit ? "rgba(0,255,80,0.6)" : "rgba(255,50,40,0.6)";
      ctx.lineWidth = 1.5; ctx.strokeRect(W / 2 - 120, 25, 240, 30);
      ctx.font = "bold 13px 'Courier New', monospace"; ctx.fillStyle = lastHit ? "#00ff44" : "#ff3322";
      ctx.textAlign = "center";
      ctx.fillText(lastHit ? "■ THREAT NEUTRALIZED ■" : "■ INTERCEPT FAILURE ■", W / 2, 45);
      ctx.textAlign = "left";
    }

    ctx.fillStyle = "rgba(200,180,40,0.4)"; ctx.font = "bold 9px 'Courier New', monospace";
    ctx.fillText("ELSTON INDUSTRIES // BMD PLATFORM // SIMULATION ONLY", W / 2 - 180, H - 6);
    ctx.fillStyle = "rgba(0,255,80,0.3)"; ctx.font = "bold 11px 'Courier New', monospace";
    ctx.textAlign = "right";
    ctx.fillText("ELSTON INDUSTRIES", W - 12, H - 20);
    ctx.font = "bold 9px 'Courier New', monospace";
    ctx.fillText("AEGIS v1.1", W - 12, H - 8);
    ctx.textAlign = "left";
  }, [playData, playFrame, playing, lastHit, maxAlt, targetX, launchX, radarRange, stats, training, icbmSpeed, wsConnected]);

  useEffect(() => {
    let run = true;
    const loop = () => {
      if (!run) return;
      scanRef.current = (scanRef.current + 0.03) % TWO_PI;
      const cvs = canvasRef.current;
      if (cvs) cvs.dispatchEvent(new Event("tick"));
      rafRef.current = requestAnimationFrame(loop);
    };
    loop();
    return () => { run = false; cancelAnimationFrame(rafRef.current); };
  }, []);

  const renderChart = (data, color, label, h = 52) => {
    if (data.length < 2) return <div style={S.awaitData}>AWAITING TELEMETRY...</div>;
    const last = data.slice(-80);
    const mn = Math.min(...last), mx = Math.max(...last), rng = mx - mn || 1;
    const cw = 214;
    const pts = last.map((v, i) => `${(i / (last.length - 1)) * cw},${h - ((v - mn) / rng) * h}`).join(" ");
    return (
      <svg width={cw} height={h + 24} style={{ display: "block", marginBottom: 12 }}>
        <defs><linearGradient id={`c-${label.replace(/ /g, '')}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor={color} stopOpacity="0.12" /><stop offset="100%" stopColor={color} stopOpacity="0" /></linearGradient></defs>
        <polygon points={pts + ` ${cw},${h} 0,${h}`} fill={`url(#c-${label.replace(/ /g, '')})`} />
        <polyline points={pts} fill="none" stroke={color} strokeWidth="1" opacity="0.65" />
        <text x="0" y={h + 18} fill="rgba(0,255,80,0.9)" fontSize="10" fontWeight="bold" fontFamily="'Courier New', monospace">{label}</text>
        <text x={cw - 30} y={h + 18} fill={color} fontSize="10" fontWeight="bold" fontFamily="'Courier New', monospace">{last[last.length - 1]?.toFixed(1)}</text>
      </svg>
    );
  };

  const sl = (label, val, set, mn, mx, step = 1, clr = "#00ff44") => (
    <div style={{ marginBottom: 8 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11, fontWeight: "bold", color: "rgba(0,255,80,0.9)", fontFamily: "'Courier New', monospace", letterSpacing: 0.5, marginBottom: 3 }}>
        <span>{label}</span><span style={{ color: clr }}>{Number.isInteger(step) ? val : val.toFixed(3)}</span>
      </div>
      <input type="range" min={mn} max={mx} step={step} value={val}
        onChange={e => set(Number(e.target.value))}
        disabled={training}
        style={{ width: "100%", accentColor: clr, height: 2, opacity: training ? 0.25 : 0.7 }}
      />
    </div>
  );

  const hitRate = stats.episodes > 0 ? (stats.hits / stats.episodes * 100).toFixed(1) : "0.0";

  return (
    <div style={S.root}>
      <div style={{ ...S.topBar, padding: "12px 18px" }}>
        <div style={{ ...S.statusDot, background: training ? "#00ff44" : "#333", boxShadow: training ? "0 0 8px #00ff44" : "none", marginLeft: 16 }} />
        <span style={{ fontSize: 13, color: "rgba(0,255,80,0.9)", letterSpacing: 2, fontWeight: 800 }}>
          AEGIS ICBM INTERCEPTOR SIMULATION PLATFORM
        </span>
        <div style={{ ...S.topRight, fontSize: 12 }}>
          <span style={{ ...S.topStat, fontSize: 12, color: wsConnected ? "#00ff44" : "#ff3322" }}>SYS: {wsConnected ? "NOMINAL" : "DATALINK ERROR"}</span>
        </div>
      </div>

      <div style={{ display: "flex", flex: 1, overflow: "hidden", minHeight: 0 }}>
        <div style={S.leftPanel}>
          <div style={S.panelTabs}>
            {[{ id: "threat", l: "THREAT" }, { id: "defense", l: "DEF" }, { id: "grpo", l: "AI CONF" }, { id: "intel", l: "INTEL" }].map(t => (
              <button key={t.id} onClick={() => setPanel(t.id)}
                style={{ ...S.tabBtn, background: panel === t.id ? "rgba(0,200,50,0.2)" : "transparent", color: panel === t.id ? "#00ff44" : "rgba(0,255,80,0.7)", borderBottom: panel === t.id ? "3px solid rgba(0,255,50,0.8)" : "3px solid transparent" }}>
                {t.l}
              </button>
            ))}
          </div>

          <div style={S.panelBody}>
            {panel === "threat" && <>
              <div style={S.sectionHead("#ff4030")}>◆ THREAT CONFIG</div>
              {sl("ICBM VELOCITY", icbmSpeed, setIcbmSpeed, 2, 20, 0.5, "#ff4030")}
              {sl("APOGEE ALT", maxAlt, setMaxAlt, 100, 460, 10, "#ff4030")}
              {sl("LAUNCH ORIGIN", launchX, setLaunchX, 20, 250, 5, "#ff4030")}
              {sl("IMPACT TGT", targetX, setTargetX, 550, 860, 5, "#ff4030")}
            </>}
            {panel === "defense" && <>
              <div style={S.sectionHead("#00b830")}>◆ DEFENSE CONFIG</div>
              {sl("EI-4700 RANGE", radarRange, setRadarRange, 100, 600, 10)}
              {sl("INTERCEPT SPD", interceptSpeed, setInterceptSpeed, 4, 30, 0.5)}
              {sl("LAUNCH DLY", launchDelay, setLaunchDelay, 0, 30, 1)}
            </>}
            {panel === "grpo" && <>
              <div style={S.sectionHead("#30a8f0")}>◆ ALGORITHM & TRAINING</div>
              <div style={{ display: "flex", gap: 6, marginBottom: 12 }}>
                <button onClick={() => setAlgo("GRPO")} style={{ flex: 1, padding: "8px 0", fontSize: 11, background: algo === "GRPO" ? "rgba(0,255,80,0.25)" : "transparent", color: algo === "GRPO" ? "#00ff44" : "rgba(0,180,40,0.6)", border: algo === "GRPO" ? "1px solid #00ff44" : "1px solid rgba(0,255,80,0.3)", cursor: "pointer", fontWeight: "bold" }}>GRPO</button>
                <button onClick={() => setAlgo("ES")} style={{ flex: 1, padding: "8px 0", fontSize: 11, background: algo === "ES" ? "rgba(0,255,80,0.25)" : "transparent", color: algo === "ES" ? "#00ff44" : "rgba(0,180,40,0.6)", border: algo === "ES" ? "1px solid #00ff44" : "1px solid rgba(0,255,80,0.3)", cursor: "pointer", fontWeight: "bold" }}>ES (EVOL.)</button>
                <button onClick={() => setAlgo("PPO")} style={{ flex: 1, padding: "8px 0", fontSize: 11, background: algo === "PPO" ? "rgba(0,255,80,0.25)" : "transparent", color: algo === "PPO" ? "#00ff44" : "rgba(0,180,40,0.6)", border: algo === "PPO" ? "1px solid #00ff44" : "1px solid rgba(0,255,80,0.3)", cursor: "pointer", fontWeight: "bold" }}>PPO</button>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 16, cursor: "pointer" }} onClick={() => setRandomize(!randomize)}>
                <div style={{ width: 14, height: 14, border: "2px solid #00ff44", background: randomize ? "#00ff44" : "transparent" }} />
                <span style={{ fontSize: 11, color: "rgba(0,255,80,0.9)", fontWeight: "bold" }}>DOMAIN RANDOMIZATION</span>
              </div>
              {algo === "GRPO" && <>
                {sl("GROUP SIZE", groupSize, setGroupSize, 4, 32, 1, "#30a8f0")}
                {sl("CLIP RATIO", clipRatio, setClipRatio, 0.05, 0.5, 0.01, "#30a8f0")}
                {sl("KL COEFF", klCoeff, setKlCoeff, 0.0, 0.2, 0.005, "#30a8f0")}
                {sl("LR", grpoLr, setGrpoLr, 0.001, 0.02, 0.001, "#30a8f0")}
              </>}
              {algo === "ES" && <>
                {sl("POP SIZE (G)", groupSize, setGroupSize, 4, 64, 2, "#30a8f0")}
                {sl("NOISE (SIGMA)", clipRatio, setClipRatio, 0.01, 0.5, 0.01, "#30a8f0")}
                {sl("LR", grpoLr, setGrpoLr, 0.005, 0.1, 0.005, "#30a8f0")}
              </>}
              {algo === "PPO" && <>
                {sl("BATCH SIZE", groupSize, setGroupSize, 8, 128, 8, "#30a8f0")}
                {sl("CLIP RATIO", clipRatio, setClipRatio, 0.05, 0.3, 0.01, "#30a8f0")}
                {sl("LR", grpoLr, setGrpoLr, 0.0001, 0.01, 0.0001, "#30a8f0")}
              </>}
            </>}
            {panel === "intel" && <>
              <div style={S.sectionHead("#c8a020")}>◆ Training TELEMETRY</div>
              <div style={S.statGrid}>
                <div style={S.statItem}>EPISODES<br /><span style={{ color: "#00ff44", fontSize: 15 }}>{stats.episodes}</span></div>
                <div style={S.statItem}>HITS<br /><span style={{ color: "#00ff88", fontSize: 15 }}>{stats.hits}</span></div>
                <div style={S.statItem}>Pk<br /><span style={{ color: parseFloat(hitRate) > 50 ? "#00ff88" : "#ffcc44", fontSize: 15 }}>{hitRate}%</span></div>
                <div style={S.statItem}>AVG REW<br /><span style={{ color: stats.avgReward > 0 ? "#00ff88" : "#ff6050", fontSize: 15 }}>{stats.avgReward.toFixed(1)}</span></div>
              </div>
              {renderChart(stats.rewardHistory, "#00b830", "AVG REWARD")}
              {renderChart(stats.hitRateHistory, "#30a8f0", "HIT RATE")}
            </>}
          </div>

          <div style={S.actionBar}>
            <button onClick={() => setTraining(!training)} style={{ ...S.actionBtn, background: training ? "rgba(255,35,25,0.06)" : "rgba(0,255,50,0.05)", color: training ? "#ff2818" : "#00c830" }}>
              {training ? "■ CEASE" : "▶ COMMENCE"}
            </button>
            <button onClick={playEpisode} disabled={training} style={{ ...S.actionBtn, background: "rgba(255,35,25,0.06)", color: training ? "rgba(35,210,245,0.12)" : "#22c8e8" }}>
              ▶ ENGAGE
            </button>
            <button onClick={resetAgent} style={{ ...S.actionBtn, background: "rgba(255,35,25,0.06)", color: "rgba(255,170,50,0.3)" }}>
              ↻ RESET
            </button>
          </div>
        </div>

        <div style={S.mainArea}>
          <canvas ref={canvasRef} style={S.canvas} />
          <div style={S.statusStrip}>
            {[
              { l: "MODE", v: training ? "TRAIN" : playing ? "ENGAGED" : "STBY", c: training ? "#00ff44" : playing ? "#22c8e8" : "#444" },
              { l: "EPISODES", v: stats.episodes, c: "#00b830" },
              { l: "Pk", v: `${hitRate}%`, c: "#00ff44" },
              { l: "ALGO", v: `${algo} (C++)`, c: "#00b830" }
            ].map((item, i) => (
              <div key={i} style={S.stripCell}>
                <span style={{ color: "rgba(0,255,80,0.9)", fontSize: 10, fontWeight: "bold" }}>{item.l}</span><br />
                <span style={{ color: item.c, fontSize: 13, fontWeight: 800 }}>{item.v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

const S = {
  root: { background: "#030704", height: "100vh", overflow: "hidden", color: "#b0f0b0", fontFamily: "'Courier New', 'Lucida Console', monospace", fontSize: 14, display: "flex", flexDirection: "column" },
  topBar: { background: "linear-gradient(180deg, #081208, #040a05)", borderBottom: "1px solid rgba(0,160,35,0.2)", padding: "10px 14px", display: "flex", alignItems: "center", gap: 10 },
  statusDot: { width: 10, height: 10, borderRadius: "50%" },
  topRight: { marginLeft: "auto", display: "flex", gap: 16 },
  topStat: { fontSize: 12, color: "rgba(0,255,80,0.85)", letterSpacing: 1, fontWeight: 800 },
  leftPanel: { width: 320, minWidth: 320, background: "rgba(4,10,5,0.98)", borderRight: "1px solid rgba(0,160,35,0.2)", display: "flex", flexDirection: "column" },
  panelTabs: { display: "flex", borderBottom: "1px solid rgba(0,160,35,0.2)", flexShrink: 0 },
  tabBtn: { flex: 1, padding: "12px 0", fontSize: 12, letterSpacing: 1.5, border: "none", cursor: "pointer", fontFamily: "inherit", fontWeight: 800 },
  panelBody: { padding: "16px 18px", flex: 1, overflowY: "auto", minHeight: 0 },
  sectionHead: (c) => ({ fontSize: 11, letterSpacing: 2, color: c, opacity: 1, marginBottom: 14, fontWeight: 800, borderBottom: `2px solid ${c}66`, paddingBottom: 6 }),
  statGrid: { display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 16 },
  statItem: { padding: "10px 10px", background: "rgba(0,160,35,0.1)", border: "1px solid rgba(0,255,80,0.25)", fontSize: 11, color: "rgba(0,255,80,0.95)", lineHeight: 1.8, fontWeight: 700 },
  awaitData: { color: "rgba(0,255,80,0.7)", fontSize: 11, fontWeight: "bold", fontFamily: "'Courier New', monospace", padding: "12px 0" },
  actionBar: { padding: "12px 18px", borderTop: "1px solid rgba(0,160,35,0.2)", background: "rgba(4,8,5,0.95)", flexShrink: 0 },
  actionBtn: { width: "100%", padding: "12px 0", fontSize: 13, letterSpacing: 2.5, fontWeight: 800, border: "1px solid", cursor: "pointer", fontFamily: "inherit", marginBottom: 6, display: "block" },
  mainArea: { flex: 1, padding: 16, display: "flex", flexDirection: "column", background: "#030805", justifyContent: "center" },
  canvas: { width: W, height: H, border: "2px solid rgba(0,255,80,0.2)", boxShadow: "0 0 20px rgba(0,255,80,0.05), inset 0 0 50px rgba(0,18,4,0.6)", margin: "0 auto", display: "block" },
  statusStrip: { display: "flex", marginTop: 12, border: "1px solid rgba(0,160,35,0.25)", background: "rgba(4,10,5,0.95)", letterSpacing: 1.2, width: W, margin: "12px auto 0" },
  stripCell: { flex: 1, padding: "8px 12px", borderRight: "1px solid rgba(0,160,35,0.2)" },
};
