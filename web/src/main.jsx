import React, { useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import './styles.css';

const api = async (path, options = {}) => {
  const response = await fetch(path, {
    headers: options.body ? { 'Content-Type': 'application/json' } : undefined,
    ...options,
  });
  if (!response.ok) throw new Error(await response.text());
  return response.json();
};

const emptyBoard = (size) =>
  Array.from({ length: size }, () =>
    Array.from({ length: size }, () => Array.from({ length: size }, () => 0)),
  );

const moveIndex = (x, y, z, size) => z * size * size + y * size + x;
const moveCoord = (move, size) => ({
  z: Math.floor(move / (size * size)),
  y: Math.floor((move % (size * size)) / size),
  x: move % size,
});

function Header({ active }) {
  return (
    <header className="topbar">
      <div>
        <h1>Qubic Lab</h1>
        <p>{active === 'play' ? 'Play and inspect selected agents.' : 'Review runs, datasets, artifacts, and evaluations.'}</p>
      </div>
      <nav>
        <a className={active === 'runs' ? 'active' : ''} href="/runs">Run viewer</a>
        <a className={active === 'play' ? 'active' : ''} href="/play">Play app</a>
      </nav>
    </header>
  );
}

function Home() {
  return (
    <>
      <Header active="home" />
      <main className="home">
        <a href="/runs">
          <strong>Run viewer</strong>
          <span>Start training, inspect artifacts, compare models, and generate self-play datasets.</span>
        </a>
        <a href="/play">
          <strong>Play app</strong>
          <span>Play against a chosen model with a 3D board, heatmap, and explicit legal move controls.</span>
        </a>
      </main>
    </>
  );
}

function RunViewerApp() {
  const [runs, setRuns] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedRun, setSelectedRun] = useState('');
  const [selectedModel, setSelectedModel] = useState('random');
  const [runData, setRunData] = useState({ latest: null, history: [] });
  const [liveState, setLiveState] = useState(null);
  const [tournament, setTournament] = useState(null);
  const [dataset, setDataset] = useState(null);
  const [busy, setBusy] = useState('');
  const [error, setError] = useState('');

  const loadRuns = async () => {
    const data = await api('/api/runs');
    setRuns(data.runs || []);
    if (!selectedRun && data.runs?.[0]) setSelectedRun(data.runs[0].run_dir);
  };

  const loadModels = async () => {
    const data = await api('/api/models');
    setModels(data.models || []);
    const preferred = data.models?.find((m) => m.kind === 'neural') || data.models?.[0];
    if (preferred && selectedModel === 'random') setSelectedModel(preferred.id);
  };

  const loadRun = async (runDir = selectedRun) => {
    if (!runDir) return;
    setError('');
    const data = await api(`/api/run?run_dir=${encodeURIComponent(runDir)}`);
    setRunData(data);
  };

  useEffect(() => {
    loadRuns().catch((err) => setError(String(err)));
    loadModels().catch((err) => setError(String(err)));
    const timer = setInterval(async () => {
      const state = await api('/api/state');
      setLiveState(state);
      if (state.latest) setRunData({ latest: state.latest, history: state.history || [] });
    }, 1500);
    return () => clearInterval(timer);
  }, []);

  const startRun = async (method) => {
    setBusy(method);
    setError('');
    try {
      await api('/api/start', {
        method: 'POST',
        body: JSON.stringify({ method, size: 3, episodes: 5000, batch_episodes: 64, log_every: 100 }),
      });
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy('');
    }
  };

  const stopRun = async () => {
    await api('/api/stop', { method: 'POST' });
  };

  const runTournament = async () => {
    setBusy('tournament');
    setError('');
    try {
      const ids = models.slice(0, 6).map((m) => m.id);
      const data = await api('/api/eval/tournament', {
        method: 'POST',
        body: JSON.stringify({ model_ids: ids, size: 3, games: 12 }),
      });
      setTournament(data);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy('');
    }
  };

  const generateDataset = async () => {
    setBusy('dataset');
    setError('');
    try {
      const data = await api('/api/selfplay/generate', {
        method: 'POST',
        body: JSON.stringify({ model_id: selectedModel, size: 3, games: 100, greedy: false }),
      });
      setDataset(data);
      await loadRuns();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy('');
    }
  };

  return (
    <>
      <Header active="runs" />
      <main className="layout">
        <aside>
          <label>Saved run<select value={selectedRun} onChange={(e) => setSelectedRun(e.target.value)}>
            {runs.map((r) => <option key={r.run_dir} value={r.run_dir}>{r.method} · {r.run_id || r.run_dir}</option>)}
          </select></label>
          <button onClick={() => loadRun()}>Load selected run</button>
          <button onClick={() => startRun('ppo')} disabled={Boolean(busy)}>Start PPO 5k</button>
          <button onClick={() => startRun('grpo')} disabled={Boolean(busy)}>Start GRPO 5k</button>
          <button onClick={stopRun}>Stop active run</button>
          <label>Dataset model<select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
            {models.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
          </select></label>
          <button onClick={generateDataset} disabled={Boolean(busy)}>Generate 100-game dataset</button>
          <button onClick={runTournament} disabled={Boolean(busy)}>Run tournament</button>
          <div className="status">
            <span>{liveState?.running ? 'training active' : 'idle'}</span>
            <span>{busy ? `working: ${busy}` : runData.latest?.method || 'no run loaded'}</span>
            {error && <b>{error}</b>}
          </div>
        </aside>
        <section className="panel">
          <RunSummary latest={runData.latest} />
          <MetricsChart history={runData.history || []} />
          {runData.latest?.heatmap && <Board3D analysis={runData.latest} compact />}
          <Artifacts latest={runData.latest} />
          <RunList runs={runs} onLoad={(runDir) => { setSelectedRun(runDir); loadRun(runDir); }} />
          {dataset && <JsonBlock title="Latest self-play dataset" value={dataset} />}
          {tournament && <JsonBlock title="Tournament" value={tournament.leaderboard} />}
        </section>
      </main>
    </>
  );
}

function PlayApp() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('random');
  const [humanPlayer, setHumanPlayer] = useState(1);
  const [game, setGame] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [error, setError] = useState('');

  useEffect(() => {
    api('/api/models').then((data) => {
      setModels(data.models || []);
      const preferred = data.models?.find((m) => m.kind === 'neural') || data.models?.[0];
      if (preferred) setSelectedModel(preferred.id);
    }).catch((err) => setError(String(err)));
  }, []);

  const selected = models.find((m) => m.id === selectedModel);
  const size = selected?.size || 3;

  const analyzeOpening = async () => {
    setError('');
    const data = await api('/api/analyze/position', {
      method: 'POST',
      body: JSON.stringify({ model_id: selectedModel, board: emptyBoard(size), player: 1 }),
    });
    setGame(null);
    setAnalysis(data);
  };

  const newGame = async () => {
    setError('');
    const data = await api('/api/play/new', {
      method: 'POST',
      body: JSON.stringify({ model_id: selectedModel, size, human_player: humanPlayer }),
    });
    setGame({ ...data, human_moves: [] });
    setAnalysis(data.state);
  };

  const playMove = async (move) => {
    if (!game || game.done) return;
    setError('');
    const moves = [...(game.human_moves || []), move];
    try {
      const data = await api('/api/play/move', {
        method: 'POST',
        body: JSON.stringify({ model_id: selectedModel, size, human_player: humanPlayer, moves }),
      });
      setGame({ ...data, human_moves: moves });
      setAnalysis(data.state);
    } catch (err) {
      setError(String(err));
    }
  };

  return (
    <>
      <Header active="play" />
      <main className="play-layout">
        <section className="panel play-main">
          <div className="toolbar">
            <label>Model<select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
              {models.map((m) => <option key={m.id} value={m.id}>{m.label}</option>)}
            </select></label>
            <label>Side<select value={humanPlayer} onChange={(e) => setHumanPlayer(Number(e.target.value))}>
              <option value={1}>Human X</option>
              <option value={-1}>Human O</option>
            </select></label>
            <button onClick={newGame}>New game</button>
            <button onClick={analyzeOpening}>Analyze opening</button>
          </div>
          {error && <div className="error">{error}</div>}
          <Summary analysis={analysis} game={game} />
          <Board3D analysis={analysis} onMove={game && !game.done ? playMove : null} />
        </section>
        <aside className="play-side">
          <MoveButtons analysis={analysis} onMove={game && !game.done ? playMove : null} />
          <MoveList game={game} />
        </aside>
      </main>
    </>
  );
}

function Board3D({ analysis, onMove, compact = false }) {
  const canvasRef = useRef(null);
  const viewRef = useRef({ yaw: -0.72, pitch: 0.72, dragging: false, moved: false, last: null });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const draw = () => drawBoard(ctx, canvas, analysis, viewRef.current, Boolean(onMove));
    draw();

    const down = (event) => {
      viewRef.current.dragging = true;
      viewRef.current.moved = false;
      viewRef.current.last = { x: event.clientX, y: event.clientY };
      canvas.setPointerCapture(event.pointerId);
    };
    const move = (event) => {
      if (!viewRef.current.dragging || !viewRef.current.last) return;
      const dx = event.clientX - viewRef.current.last.x;
      const dy = event.clientY - viewRef.current.last.y;
      if (Math.hypot(dx, dy) > 2) viewRef.current.moved = true;
      viewRef.current.yaw += dx * 0.01;
      viewRef.current.pitch = Math.max(0.15, Math.min(1.35, viewRef.current.pitch + dy * 0.01));
      viewRef.current.last = { x: event.clientX, y: event.clientY };
      draw();
    };
    const up = () => {
      viewRef.current.dragging = false;
      viewRef.current.last = null;
    };
    const click = (event) => {
      if (!onMove || !analysis?.heatmap || viewRef.current.moved) return;
      const rect = canvas.getBoundingClientRect();
      const sx = ((event.clientX - rect.left) / rect.width) * canvas.width;
      const sy = ((event.clientY - rect.top) / rect.height) * canvas.height;
      const hit = nearestPoint(analysis, viewRef.current, canvas, sx, sy);
      const idx = hit ? moveIndex(hit.x, hit.y, hit.z, analysis.heatmap.length) : null;
      if (hit && (analysis.legal_moves || []).includes(idx)) onMove(idx);
    };
    canvas.addEventListener('pointerdown', down);
    canvas.addEventListener('pointermove', move);
    canvas.addEventListener('pointerup', up);
    canvas.addEventListener('click', click);
    return () => {
      canvas.removeEventListener('pointerdown', down);
      canvas.removeEventListener('pointermove', move);
      canvas.removeEventListener('pointerup', up);
      canvas.removeEventListener('click', click);
    };
  }, [analysis, onMove]);

  return <canvas ref={canvasRef} className={`board3d ${compact ? 'compact' : ''}`} width="1200" height="720" />;
}

function flattenHeatmap(heatmap) {
  const points = [];
  if (!heatmap) return points;
  heatmap.forEach((layer, z) =>
    layer.forEach((row, y) => row.forEach((value, x) => points.push({ x, y, z, value: Number(value) || 0 }))),
  );
  return points;
}

function project(p, size, view, scale, cx, cy) {
  const ox = p.x - (size - 1) / 2;
  const oy = p.y - (size - 1) / 2;
  const oz = p.z - (size - 1) / 2;
  const cyaw = Math.cos(view.yaw);
  const syaw = Math.sin(view.yaw);
  const cp = Math.cos(view.pitch);
  const sp = Math.sin(view.pitch);
  const x1 = ox * cyaw - oz * syaw;
  const z1 = ox * syaw + oz * cyaw;
  const y1 = oy * cp - z1 * sp;
  const z2 = oy * sp + z1 * cp;
  return { x: cx + x1 * scale, y: cy + y1 * scale, depth: z2 };
}

function nearestPoint(analysis, view, canvas, sx, sy) {
  const size = analysis.heatmap.length;
  const scale = Math.min(canvas.width, canvas.height) / (size <= 3 ? 4.4 : 5.4);
  const cx = canvas.width * 0.53;
  const cy = canvas.height * 0.55;
  let best = null;
  for (const p of flattenHeatmap(analysis.heatmap)) {
    const s = project(p, size, view, scale, cx, cy);
    const d = Math.hypot(s.x - sx, s.y - sy);
    if (d < 42 && (!best || d < best.d)) best = { ...p, d };
  }
  return best;
}

function drawBoard(ctx, canvas, analysis, view, playable) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#101418';
  ctx.fillRect(0, 0, w, h);
  if (!analysis?.heatmap) {
    ctx.fillStyle = '#9badb4';
    ctx.font = '22px Inter, system-ui';
    ctx.fillText('Choose a model, load a run, or start a game.', 36, 52);
    return;
  }

  const size = analysis.heatmap.length;
  const points = flattenHeatmap(analysis.heatmap);
  const maxAbs = Math.max(1e-4, ...points.map((p) => Math.abs(p.value)));
  const scale = Math.min(w, h) / (size <= 3 ? 4.4 : 5.4);
  const cx = w * 0.53;
  const cy = h * 0.55;

  const line = (a, b, color, width = 1) => {
    const pa = project(a, size, view, scale, cx, cy);
    const pb = project(b, size, view, scale, cx, cy);
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(pa.x, pa.y);
    ctx.lineTo(pb.x, pb.y);
    ctx.stroke();
  };

  for (let y = 0; y < size; y += 1) {
    for (let z = 0; z < size; z += 1) line({ x: 0, y, z }, { x: size - 1, y, z }, 'rgba(130,150,158,.25)');
    for (let x = 0; x < size; x += 1) line({ x, y, z: 0 }, { x, y, z: size - 1 }, 'rgba(130,150,158,.25)');
  }
  for (let x = 0; x < size; x += 1) {
    for (let z = 0; z < size; z += 1) line({ x, y: 0, z }, { x, y: size - 1, z }, 'rgba(130,150,158,.16)');
  }

  drawArrows(ctx, analysis, view, scale, cx, cy);
  const legal = new Set(analysis.legal_moves || []);
  const occupied = new Map();
  analysis.board?.forEach((layer, z) => layer.forEach((row, y) => row.forEach((v, x) => {
    if (v) occupied.set(moveIndex(x, y, z, size), v);
  })));

  const sorted = points.map((p) => ({ ...p, screen: project(p, size, view, scale, cx, cy) })).sort((a, b) => a.screen.depth - b.screen.depth);
  const bestMove = analysis.top_moves?.[0]?.move;
  for (const p of sorted) {
    const idx = moveIndex(p.x, p.y, p.z, size);
    const occ = occupied.get(idx);
    const norm = p.value / maxAbs;
    const radius = occ ? 25 : 10 + 26 * Math.abs(norm);
    ctx.beginPath();
    ctx.arc(p.screen.x, p.screen.y, radius, 0, Math.PI * 2);
    ctx.fillStyle = occ === 1 ? '#67d2a7' : occ === -1 ? '#f07c6b' : valueColor(norm, legal.has(idx) ? 0.92 : 0.22);
    ctx.fill();
    ctx.strokeStyle = idx === bestMove ? '#f5c15d' : playable && legal.has(idx) ? '#dce7e9' : 'rgba(235,242,244,.38)';
    ctx.lineWidth = idx === bestMove ? 5 : playable && legal.has(idx) ? 2 : 1;
    ctx.stroke();
    if (occ) {
      ctx.fillStyle = '#07100d';
      ctx.font = 'bold 24px Inter, system-ui';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(occ === 1 ? 'X' : 'O', p.screen.x, p.screen.y + 1);
    }
  }

  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = '#e6ecef';
  ctx.font = '19px Inter, system-ui';
  const best = analysis.top_moves?.[0];
  ctx.fillText(best ? `best (${best.x},${best.y},${best.z}) p=${best.prob.toFixed(3)} value=${Number(analysis.value || 0).toFixed(3)}` : 'terminal position', 32, 38);
  ctx.fillStyle = '#9badb4';
  ctx.font = '14px Inter, system-ui';
  ctx.fillText('green: X  red: O  amber arrows: top policy moves  drag: rotate', 32, h - 28);
}

function drawArrows(ctx, analysis, view, scale, cx, cy) {
  const size = analysis.heatmap.length;
  const top = (analysis.top_moves || []).slice(0, 8);
  if (!top.length) return;
  const center = { x: (size - 1) / 2, y: (size - 1) / 2, z: (size - 1) / 2 };
  for (const move of top) {
    const target = { x: move.x, y: move.y, z: move.z };
    const a = project(center, size, view, scale, cx, cy);
    const b = project(target, size, view, scale, cx, cy);
    const end = { x: a.x + (b.x - a.x) * 0.82, y: a.y + (b.y - a.y) * 0.82 };
    const alpha = Math.max(0.18, Math.min(0.9, move.prob * 8));
    const color = `rgba(245,193,93,${alpha})`;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 + 5 * move.prob;
    ctx.beginPath();
    ctx.moveTo(a.x, a.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
    const angle = Math.atan2(end.y - a.y, end.x - a.x);
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(end.x, end.y);
    ctx.lineTo(end.x - 14 * Math.cos(angle - 0.45), end.y - 14 * Math.sin(angle - 0.45));
    ctx.lineTo(end.x - 14 * Math.cos(angle + 0.45), end.y - 14 * Math.sin(angle + 0.45));
    ctx.closePath();
    ctx.fill();
  }
}

function valueColor(t, alpha) {
  if (t >= 0) return `rgba(${Math.round(100 + 80 * t)},${Math.round(165 + 70 * t)},${Math.round(145 + 20 * t)},${alpha})`;
  const u = -t;
  return `rgba(${Math.round(230 + 15 * u)},${Math.round(140 - 40 * u)},${Math.round(110 - 30 * u)},${alpha})`;
}

function MetricsChart({ history }) {
  const canvasRef = useRef(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    drawMetrics(canvas.getContext('2d'), canvas, history || []);
  }, [history]);
  return <canvas ref={canvasRef} className="chart" width="1200" height="640" />;
}

function drawMetrics(ctx, canvas, history) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#101418';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#e6ecef';
  ctx.font = '22px Inter, system-ui';
  ctx.fillText('Training diagnostics', 56, 34);
  if (history.length < 2) {
    ctx.fillStyle = '#9badb4';
    ctx.font = '17px Inter, system-ui';
    ctx.fillText('Load or start a run to draw live plots.', 56, 70);
    return;
  }
  const xs = history.map((d) => d.episode);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const panels = [
    { x: 78, y: 60, w: 1080, h: 150, title: 'Outcome rates', ymin: 0, ymax: 1 },
    { x: 78, y: 260, w: 1080, h: 130, title: 'Optimization', ymin: 0, ymax: Math.max(0.05, ...history.map((d) => Math.abs(Number(d.mean_abs_update || 0)))) },
    { x: 78, y: 450, w: 1080, h: 130, title: 'Policy statistics', ymin: 0, ymax: Math.max(0.05, ...history.map((d) => Number(d.entropy || 0)), ...history.map((d) => Number(d.approx_kl || 0))) },
  ];
  for (const p of panels) axes(ctx, p);
  series(ctx, panels[0], history, minX, maxX, (d) => d.recent?.x_win_rate, '#67d2a7');
  series(ctx, panels[0], history, minX, maxX, (d) => d.recent?.o_win_rate, '#f07c6b');
  series(ctx, panels[0], history, minX, maxX, (d) => d.recent?.draw_rate, '#7aa7ff');
  series(ctx, panels[1], history, minX, maxX, (d) => Math.abs(Number(d.mean_abs_update || 0)), '#f5c15d');
  series(ctx, panels[2], history, minX, maxX, (d) => Number(d.entropy || 0), '#b58cff');
  series(ctx, panels[2], history, minX, maxX, (d) => Number(d.approx_kl || 0), '#5cc8ff');
}

function axes(ctx, p) {
  ctx.fillStyle = '#151c21';
  ctx.fillRect(p.x, p.y, p.w, p.h);
  ctx.strokeStyle = '#31434a';
  ctx.strokeRect(p.x, p.y, p.w, p.h);
  ctx.fillStyle = '#dce7e9';
  ctx.font = '16px Inter, system-ui';
  ctx.fillText(p.title, p.x + 12, p.y + 22);
  ctx.fillStyle = '#8fa2a9';
  ctx.font = '12px Inter, system-ui';
  for (let i = 0; i <= 4; i += 1) {
    const frac = i / 4;
    const y = p.y + p.h - frac * p.h;
    const v = p.ymin + frac * (p.ymax - p.ymin);
    ctx.strokeStyle = '#26343a';
    ctx.beginPath();
    ctx.moveTo(p.x, y);
    ctx.lineTo(p.x + p.w, y);
    ctx.stroke();
    ctx.fillText(v.toFixed(2), p.x - 44, y + 4);
  }
}

function series(ctx, p, history, minX, maxX, getY, color) {
  const values = history.map((d) => ({ x: d.episode, y: Number(getY(d)) })).filter((d) => Number.isFinite(d.y));
  if (values.length < 2) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.6;
  ctx.beginPath();
  values.forEach((d, i) => {
    const x = p.x + ((d.x - minX) / Math.max(1, maxX - minX)) * p.w;
    const y = p.y + p.h - ((d.y - p.ymin) / Math.max(1e-9, p.ymax - p.ymin)) * p.h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
}

function RunSummary({ latest }) {
  const recent = latest?.recent || {};
  return (
    <div className="summary">
      <div><span>Method</span><strong>{latest?.method || 'none'}</strong></div>
      <div><span>Episode</span><strong>{latest?.episode || 0}</strong></div>
      <div><span>X win</span><strong>{Math.round((recent.x_win_rate || 0) * 100)}%</strong></div>
      <div><span>Draw</span><strong>{Math.round((recent.draw_rate || 0) * 100)}%</strong></div>
    </div>
  );
}

function Summary({ analysis, game }) {
  const top = analysis?.top_moves?.slice(0, 5) || [];
  const status = game?.done ? (game.winner === 0 ? 'draw' : `${game.winner === 1 ? 'X' : 'O'} wins`) : 'active';
  return (
    <div className="summary">
      <div><span>Value</span><strong>{Number(analysis?.value || 0).toFixed(3)}</strong></div>
      <div><span>To move</span><strong>{analysis?.player === -1 ? 'O' : 'X'}</strong></div>
      <div><span>Status</span><strong>{status}</strong></div>
      <div className="topmoves"><span>Top moves</span>{top.map((m) => <b key={m.move}>({m.x},{m.y},{m.z}) {m.prob.toFixed(2)}</b>)}</div>
    </div>
  );
}

function Artifacts({ latest }) {
  if (!latest?.run_dir) return null;
  const files = ['analysis.md', 'analysis.json', 'curves.png', 'first_move_heatmap.png', 'first_move_policy.json', 'model.pt'];
  return (
    <div className="artifact-row">
      {files.map((file) => (
        <a key={file} href={`/api/artifact?run_dir=${encodeURIComponent(latest.run_dir)}&file=${file}`} target="_blank" rel="noreferrer">{file}</a>
      ))}
    </div>
  );
}

function RunList({ runs, onLoad }) {
  return (
    <div>
      <h2>Saved runs</h2>
      <div className="rungrid">
        {runs.slice(0, 24).map((r) => (
          <article key={r.run_dir}>
            <h3>{r.method || 'run'} · {r.run_id || r.run_dir}</h3>
            <p>{r.run_dir}</p>
            <p>X {Math.round((r.recent?.x_win_rate || 0) * 100)}% · O {Math.round((r.recent?.o_win_rate || 0) * 100)}% · ep {r.episode || 0}</p>
            <button onClick={() => onLoad(r.run_dir)}>Open</button>
          </article>
        ))}
      </div>
    </div>
  );
}

function MoveButtons({ analysis, onMove }) {
  const moves = useMemo(() => {
    const size = analysis?.heatmap?.length || 3;
    return (analysis?.legal_moves || []).map((move) => ({ move, ...moveCoord(move, size) }));
  }, [analysis]);
  return (
    <div className="move-panel">
      <h2>Legal moves</h2>
      <div className="move-grid">
        {moves.map((m) => (
          <button key={m.move} disabled={!onMove} onClick={() => onMove?.(m.move)}>
            {m.x},{m.y},{m.z}
          </button>
        ))}
      </div>
    </div>
  );
}

function MoveList({ game }) {
  return (
    <div className="move-panel">
      <h2>Game history</h2>
      <pre className="json">{game ? JSON.stringify({ done: game.done, winner: game.winner, history: game.history }, null, 2) : 'Start a game to make moves.'}</pre>
    </div>
  );
}

function JsonBlock({ title, value }) {
  return (
    <div>
      <h2>{title}</h2>
      <pre className="json">{JSON.stringify(value, null, 2)}</pre>
    </div>
  );
}

const path = window.location.pathname;
const root = createRoot(document.getElementById('root'));
if (path.startsWith('/play')) root.render(<PlayApp />);
else if (path.startsWith('/runs')) root.render(<RunViewerApp />);
else root.render(<Home />);
