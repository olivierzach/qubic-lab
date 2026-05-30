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

const apiMaybe = async (path, fallback = null, options = {}) => {
  try {
    return await api(path, options);
  } catch {
    return fallback;
  }
};

const defaults = {
  method: 'ppo',
  size: 3,
  episodes: 5000,
  batch_episodes: 64,
  update_epochs: 4,
  hidden: 128,
  log_every: 100,
  lr: 0.0003,
  gamma: 0.99,
  clip_eps: 0.2,
  entropy_coef: 0.02,
  value_coef: 0.5,
  alpha: 0.25,
  epsilon: 0.35,
  epsilon_min: 0.03,
  epsilon_decay: 0.9995,
  seed: 0,
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

const pct = (value) => `${Math.round((Number(value) || 0) * 100)}%`;
const fixed = (value, places = 3) => Number(value || 0).toFixed(places);
const methodLabel = (method) => String(method || 'run').toUpperCase();

function Header({ active }) {
  return (
    <header className="topbar">
      <a className="brand" href="/lab">
        <span>Qubic Lab</span>
        <small>{active === 'play' ? 'standalone play' : 'research notebook'}</small>
      </a>
      <nav aria-label="Primary">
        <a className={active === 'lab' ? 'active' : ''} href="/lab">Lab</a>
        <a className={active === 'play' ? 'active' : ''} href="/play">Play</a>
      </nav>
    </header>
  );
}

function Home() {
  return (
    <>
      <Header active="lab" />
      <main className="home paper-width">
        <a href="/lab">
          <strong>Research lab</strong>
          <span>Configure training runs, inspect live diagnostics, compare saved agents, and browse heatmap snapshots.</span>
        </a>
        <a href="/play">
          <strong>Play app</strong>
          <span>Choose a model and side, start a clean game, use legal coordinate moves, and review history.</span>
        </a>
      </main>
    </>
  );
}

function LabApp() {
  const [runs, setRuns] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedRun, setSelectedRun] = useState('');
  const [selectedModel, setSelectedModel] = useState('random');
  const [runData, setRunData] = useState({ latest: null, history: [] });
  const [liveState, setLiveState] = useState(null);
  const [runConfig, setRunConfig] = useState(defaults);
  const [runDefaults, setRunDefaults] = useState({});
  const [timeline, setTimeline] = useState([]);
  const [snapshotIndex, setSnapshotIndex] = useState(0);
  const [analysis, setAnalysis] = useState(null);
  const [sourceMode, setSourceMode] = useState('live');
  const [busy, setBusy] = useState('');
  const [notice, setNotice] = useState('');
  const [error, setError] = useState('');

  const selected = models.find((m) => m.id === selectedModel);
  const snapshots = timeline.length ? timeline : (runData.history || []).filter((d) => d.heatmap);
  const activeSnapshot = snapshots[snapshotIndex] || runData.latest || analysis;

  const loadRuns = async () => {
    const data = await api('/api/runs');
    const nextRuns = data.runs || [];
    setRuns(nextRuns);
    if (!selectedRun && nextRuns[0]) setSelectedRun(nextRuns[0].run_dir);
  };

  const loadModels = async () => {
    const data = await api('/api/models');
    const nextModels = data.models || [];
    setModels(nextModels);
    const preferred = nextModels.find((m) => m.kind === 'neural') || nextModels[0];
    if (preferred && selectedModel === 'random') setSelectedModel(preferred.id);
  };

  const loadDefaults = async () => {
    const data = await apiMaybe('/api/run/defaults', null);
    const values = data?.defaults || {};
    setRunDefaults(values);
    if (values.ppo) setRunConfig((current) => ({ ...current, ...values.ppo }));
  };

  const loadRun = async (runDir = selectedRun) => {
    if (!runDir) return;
    setError('');
    const data = await api(`/api/run?run_dir=${encodeURIComponent(runDir)}`);
    setRunData({ latest: data.latest || data, history: data.history || [] });
    setSourceMode('saved');
    setSnapshotIndex(0);
  };

  const loadTimeline = async (modelId = selectedModel) => {
    if (!modelId) return;
    const data = await apiMaybe(`/api/model/timeline?model_id=${encodeURIComponent(modelId)}`, { timeline: [] });
    setTimeline(Array.isArray(data) ? data : data.timeline || data.snapshots || []);
    setSnapshotIndex(0);
  };

  useEffect(() => {
    loadDefaults().catch(() => {});
    loadRuns().catch((err) => setError(String(err)));
    loadModels().catch((err) => setError(String(err)));
    const timer = setInterval(async () => {
      try {
        const state = await api('/api/state');
        setLiveState(state);
        if (state.latest && sourceMode === 'live') setRunData({ latest: state.latest, history: state.history || [] });
      } catch {
        setLiveState((state) => state);
      }
    }, 1500);
    return () => clearInterval(timer);
  }, [sourceMode]);

  useEffect(() => {
    loadTimeline(selectedModel).catch(() => setTimeline([]));
  }, [selectedModel]);

  const updateConfig = (key, value) => {
    setRunConfig((current) => {
      if (key !== 'method') return { ...current, [key]: value };
      return { ...current, ...(runDefaults[value] || {}), method: value };
    });
  };

  const showLive = () => {
    setSourceMode('live');
    if (liveState?.latest) setRunData({ latest: liveState.latest, history: liveState.history || [] });
  };

  const startRun = async () => {
    setBusy('run');
    setNotice('');
    setError('');
    const body = JSON.stringify(runConfig);
    try {
      const data = await api('/api/run', { method: 'POST', body });
      setSourceMode('live');
      setNotice(data?.run_dir ? `Started ${data.run_dir}` : 'Run submitted');
      await loadRuns();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy('');
    }
  };

  const stopRun = async () => {
    setBusy('stop');
    setError('');
    try {
      await api('/api/stop', { method: 'POST' });
      setNotice('Stop requested');
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy('');
    }
  };

  const analyzeModel = async () => {
    setBusy('analyze');
    setError('');
    try {
      const size = selected?.size || runConfig.size || 3;
      const data = await api('/api/analyze/position', {
        method: 'POST',
        body: JSON.stringify({ model_id: selectedModel, board: emptyBoard(size), player: 1 }),
      });
      setAnalysis(data);
      setSnapshotIndex(0);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy('');
    }
  };

  return (
    <>
      <Header active="lab" />
      <main className="lab-page paper-width">
        <section className="paper-section title-row">
          <div>
            <p className="kicker">Experiment record</p>
            <h1>Research Lab</h1>
            <p className="lede">Configure Qubic training runs, track live outcomes, and inspect saved model policy surfaces over time.</p>
          </div>
          <div className="run-state">
            <span>{liveState?.running ? 'running' : 'idle'}</span>
            <span>{sourceMode === 'live' ? 'live source' : 'saved source'}</span>
            <b>{runData.latest?.method ? methodLabel(runData.latest.method) : 'no active run'}</b>
          </div>
        </section>

        {(error || notice) && <div className={error ? 'message error' : 'message'}>{error || notice}</div>}

        <section className="lab-grid">
          <aside className="control-rail paper-section">
            <h2>Run Parameters</h2>
            <RunControls config={runConfig} onChange={updateConfig} busy={busy} onStart={startRun} onStop={stopRun} />

            <h2>Saved Sources</h2>
            <button onClick={showLive} disabled={!liveState?.latest}>Show live state</button>
            <label>Saved run
              <select value={selectedRun} onChange={(e) => setSelectedRun(e.target.value)}>
                <option value="">Select run</option>
                {runs.map((r) => <option key={r.run_dir} value={r.run_dir}>{methodLabel(r.method)} · {r.run_id || r.run_dir}</option>)}
              </select>
            </label>
            <button onClick={() => loadRun()} disabled={!selectedRun}>Load run</button>

            <label>Model
              <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                {models.map((m) => <option key={m.id} value={m.id}>{m.label || m.id}</option>)}
              </select>
            </label>
            <button onClick={analyzeModel} disabled={!selectedModel || Boolean(busy)}>Analyze opening</button>
          </aside>

          <section className="paper-stack">
            <SystemInputs config={runConfig} selected={selected} latest={runData.latest} />
            <RunSummary latest={runData.latest} analysis={analysis} />
            <MetricsChart history={runData.history || []} />
            <SnapshotViewer
              snapshot={activeSnapshot}
              snapshots={snapshots}
              snapshotIndex={snapshotIndex}
              onIndex={setSnapshotIndex}
              analysis={analysis}
            />
            <div className="two-column">
              <ModelCatalog models={models} selectedModel={selectedModel} onSelect={setSelectedModel} />
              <RunList runs={runs} onLoad={(runDir) => { setSelectedRun(runDir); loadRun(runDir); }} />
            </div>
            <Artifacts latest={runData.latest} />
          </section>
        </section>
      </main>
    </>
  );
}

function RunControls({ config, onChange, busy, onStart, onStop }) {
  const number = (key) => (event) => onChange(key, Number(event.target.value));
  const isPolicyGradient = ['ppo', 'grpo'].includes(config.method);
  return (
    <div className="form-grid">
      <label>Method
        <select value={config.method} onChange={(e) => onChange('method', e.target.value)}>
          <option value="ppo">PPO</option>
          <option value="grpo">GRPO</option>
          <option value="q_learning">Q-learning</option>
          <option value="sarsa">SARSA</option>
          <option value="expected_sarsa">Expected SARSA</option>
          <option value="monte_carlo">Monte Carlo</option>
        </select>
      </label>
      <label>Board size<input type="number" min="2" max={isPolicyGradient ? 5 : 4} value={config.size} onChange={number('size')} /></label>
      <label>Episodes<input type="number" min="1" step="500" value={config.episodes} onChange={number('episodes')} /></label>
      <label>Log every<input type="number" min="1" step="25" value={config.log_every} onChange={number('log_every')} /></label>
      <label>Gamma<input type="number" min="0" max="1" step="0.01" value={config.gamma} onChange={number('gamma')} /></label>
      <label>Seed<input type="number" step="1" value={config.seed} onChange={number('seed')} /></label>
      {isPolicyGradient ? (
        <>
          <label>Batch episodes<input type="number" min="1" step="8" value={config.batch_episodes || 32} onChange={number('batch_episodes')} /></label>
          <label>Update epochs<input type="number" min="1" step="1" value={config.update_epochs || 4} onChange={number('update_epochs')} /></label>
          <label>Hidden width<input type="number" min="8" step="16" value={config.hidden || 128} onChange={number('hidden')} /></label>
          <label>Learning rate<input type="number" min="0" step="0.0001" value={config.lr || 0.0003} onChange={number('lr')} /></label>
          <label>Clip epsilon<input type="number" min="0.01" max="1" step="0.01" value={config.clip_eps || 0.2} onChange={number('clip_eps')} /></label>
          <label>Entropy coef<input type="number" min="0" step="0.005" value={config.entropy_coef || 0} onChange={number('entropy_coef')} /></label>
          <label>Value coef<input type="number" min="0" step="0.05" value={config.value_coef || 0} onChange={number('value_coef')} /></label>
        </>
      ) : (
        <>
          <label>Alpha<input type="number" min="0" max="1" step="0.01" value={config.alpha || 0.25} onChange={number('alpha')} /></label>
          <label>Epsilon<input type="number" min="0" max="1" step="0.01" value={config.epsilon || 0.35} onChange={number('epsilon')} /></label>
          <label>Epsilon min<input type="number" min="0" max="1" step="0.01" value={config.epsilon_min || 0.03} onChange={number('epsilon_min')} /></label>
          <label>Epsilon decay<input type="number" min="0" max="1" step="0.0001" value={config.epsilon_decay || 0.9995} onChange={number('epsilon_decay')} /></label>
        </>
      )}
      <div className="button-row span-2">
        <button onClick={onStart} disabled={Boolean(busy)}>Start run</button>
        <button onClick={onStop} disabled={busy === 'stop'}>Stop</button>
      </div>
    </div>
  );
}

function SystemInputs({ config, selected, latest }) {
  const rows = [
    ['method', methodLabel(config.method)],
    ['episodes', config.episodes],
    ['batch', config.batch_episodes || 'n/a'],
    ['log every', config.log_every],
    ['lr / alpha', config.lr || config.alpha || 'n/a'],
    ['model', selected?.label || selected?.id || 'none'],
    ['model kind', selected?.kind || 'unknown'],
    ['run dir', latest?.run_dir || 'none'],
    ['latest episode', latest?.episode || 0],
  ];
  return (
    <section className="paper-section">
      <h2>System Inputs</h2>
      <dl className="input-table">
        {rows.map(([key, value]) => (
          <React.Fragment key={key}>
            <dt>{key}</dt>
            <dd>{value}</dd>
          </React.Fragment>
        ))}
      </dl>
    </section>
  );
}

function SnapshotViewer({ snapshot, snapshots, snapshotIndex, onIndex, analysis }) {
  const max = Math.max(0, snapshots.length - 1);
  const source = snapshot || analysis;
  return (
    <section className="paper-section">
      <div className="section-head">
        <div>
          <h2>Policy Heatmap</h2>
          <p>{snapshots.length ? `Snapshot ${snapshotIndex + 1} of ${snapshots.length}` : 'Current or analyzed position'}</p>
        </div>
        <label className="range-label">time
          <input
            type="range"
            min="0"
            max={max}
            value={Math.min(snapshotIndex, max)}
            onChange={(e) => onIndex(Number(e.target.value))}
            disabled={!snapshots.length}
          />
        </label>
      </div>
      <Board3D analysis={source} compact />
      <PolicyTable analysis={source} />
    </section>
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
      const nextModels = data.models || [];
      setModels(nextModels);
      const preferred = nextModels.find((m) => m.kind === 'neural') || nextModels[0];
      if (preferred) setSelectedModel(preferred.id);
    }).catch((err) => setError(String(err)));
  }, []);

  const selected = models.find((m) => m.id === selectedModel);
  const size = selected?.size || 3;

  const newGame = async () => {
    setError('');
    try {
      const data = await api('/api/play/new', {
        method: 'POST',
        body: JSON.stringify({ model_id: selectedModel, size, human_player: humanPlayer }),
      });
      setGame({ ...data, human_moves: [] });
      setAnalysis(data.state);
    } catch (err) {
      setError(String(err));
    }
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
      <main className="play-page paper-width">
        <section className="paper-section title-row">
          <div>
            <p className="kicker">Standalone board</p>
            <h1>Play Qubic</h1>
          </div>
          <div className="toolbar compact-toolbar">
            <label>Model
              <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                {models.map((m) => <option key={m.id} value={m.id}>{m.label || m.id}</option>)}
              </select>
            </label>
            <label>Side
              <select value={humanPlayer} onChange={(e) => setHumanPlayer(Number(e.target.value))}>
                <option value={1}>Human X</option>
                <option value={-1}>Human O</option>
              </select>
            </label>
            <button onClick={newGame}>New game</button>
          </div>
        </section>
        {error && <div className="message error">{error}</div>}
        <section className="play-grid">
          <div className="paper-section play-board">
            <Summary analysis={analysis} game={game} />
            <Board3D analysis={analysis} onMove={game && !game.done ? playMove : null} />
          </div>
          <aside className="paper-section play-tools">
            <MoveButtons analysis={analysis} onMove={game && !game.done ? playMove : null} />
            <MoveList game={game} />
          </aside>
        </section>
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
  ctx.fillStyle = '#fbfaf5';
  ctx.fillRect(0, 0, w, h);
  if (!analysis?.heatmap) {
    ctx.fillStyle = '#6f6659';
    ctx.font = '22px Georgia, serif';
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
    for (let z = 0; z < size; z += 1) line({ x: 0, y, z }, { x: size - 1, y, z }, 'rgba(58,50,42,.24)');
    for (let x = 0; x < size; x += 1) line({ x, y, z: 0 }, { x, y, z: size - 1 }, 'rgba(58,50,42,.24)');
  }
  for (let x = 0; x < size; x += 1) {
    for (let z = 0; z < size; z += 1) line({ x, y: 0, z }, { x, y: size - 1, z }, 'rgba(58,50,42,.14)');
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
    ctx.fillStyle = occ === 1 ? '#2f6f59' : occ === -1 ? '#a7503f' : valueColor(norm, legal.has(idx) ? 0.82 : 0.2);
    ctx.fill();
    ctx.strokeStyle = idx === bestMove ? '#9c6b18' : playable && legal.has(idx) ? '#221f1a' : 'rgba(52,45,37,.42)';
    ctx.lineWidth = idx === bestMove ? 5 : playable && legal.has(idx) ? 2 : 1;
    ctx.stroke();
    if (occ) {
      ctx.fillStyle = '#fbfaf5';
      ctx.font = 'bold 24px Georgia, serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(occ === 1 ? 'X' : 'O', p.screen.x, p.screen.y + 1);
    }
  }

  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = '#221f1a';
  ctx.font = '19px Georgia, serif';
  const best = analysis.top_moves?.[0];
  ctx.fillText(best ? `best (${best.x},${best.y},${best.z})  p=${fixed(best.prob)}  value=${fixed(analysis.value)}` : 'terminal position', 32, 38);
  ctx.fillStyle = '#6f6659';
  ctx.font = '14px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillText('X: green  O: red  arrows: top policy  drag: rotate', 32, h - 28);
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
    const color = `rgba(156,107,24,${alpha})`;
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
  if (t >= 0) return `rgba(${Math.round(72 + 55 * t)},${Math.round(132 + 65 * t)},${Math.round(108 + 25 * t)},${alpha})`;
  const u = -t;
  return `rgba(${Math.round(170 + 25 * u)},${Math.round(92 - 22 * u)},${Math.round(72 - 18 * u)},${alpha})`;
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
  ctx.fillStyle = '#fbfaf5';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#221f1a';
  ctx.font = '22px Georgia, serif';
  ctx.fillText('Training diagnostics', 56, 34);
  if (history.length < 2) {
    ctx.fillStyle = '#6f6659';
    ctx.font = '17px Georgia, serif';
    ctx.fillText('Load or start a run to draw live plots.', 56, 70);
    return;
  }
  const xs = history.map((d) => d.episode || d.step || 0);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const panels = [
    { x: 78, y: 60, w: 1080, h: 150, title: 'Outcome rates', ymin: 0, ymax: 1 },
    { x: 78, y: 260, w: 1080, h: 130, title: 'Value and update magnitude', ymin: -1, ymax: 1 },
    { x: 78, y: 450, w: 1080, h: 130, title: 'Policy statistics', ymin: 0, ymax: Math.max(0.05, ...history.map((d) => Number(d.entropy || 0)), ...history.map((d) => Number(d.approx_kl || 0))) },
  ];
  for (const p of panels) axes(ctx, p);
  series(ctx, panels[0], history, minX, maxX, (d) => d.recent?.x_win_rate, '#2f6f59');
  series(ctx, panels[0], history, minX, maxX, (d) => d.recent?.o_win_rate, '#a7503f');
  series(ctx, panels[0], history, minX, maxX, (d) => d.recent?.draw_rate, '#526f9e');
  series(ctx, panels[1], history, minX, maxX, (d) => d.value ?? d.mean_value, '#2b5f88');
  series(ctx, panels[1], history, minX, maxX, (d) => Math.abs(Number(d.mean_abs_update || 0)), '#9c6b18');
  series(ctx, panels[2], history, minX, maxX, (d) => Number(d.entropy || 0), '#6b5b95');
  series(ctx, panels[2], history, minX, maxX, (d) => Number(d.approx_kl || 0), '#477486');
}

function axes(ctx, p) {
  ctx.fillStyle = '#f7f3ea';
  ctx.fillRect(p.x, p.y, p.w, p.h);
  ctx.strokeStyle = '#a99c88';
  ctx.strokeRect(p.x, p.y, p.w, p.h);
  ctx.fillStyle = '#221f1a';
  ctx.font = '16px Georgia, serif';
  ctx.fillText(p.title, p.x + 12, p.y + 22);
  ctx.fillStyle = '#746b5d';
  ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
  for (let i = 0; i <= 4; i += 1) {
    const frac = i / 4;
    const y = p.y + p.h - frac * p.h;
    const v = p.ymin + frac * (p.ymax - p.ymin);
    ctx.strokeStyle = '#ded5c7';
    ctx.beginPath();
    ctx.moveTo(p.x, y);
    ctx.lineTo(p.x + p.w, y);
    ctx.stroke();
    ctx.fillText(v.toFixed(2), p.x - 48, y + 4);
  }
}

function series(ctx, p, history, minX, maxX, getY, color) {
  const values = history
    .map((d) => ({ x: d.episode || d.step || 0, y: Number(getY(d)) }))
    .filter((d) => Number.isFinite(d.y));
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

function RunSummary({ latest, analysis }) {
  const recent = latest?.recent || {};
  const top = (latest?.top_moves || analysis?.top_moves || []).slice(0, 4);
  return (
    <section className="summary-grid">
      <Metric label="Method" value={latest?.method ? methodLabel(latest.method) : 'none'} />
      <Metric label="Episode" value={latest?.episode || 0} />
      <Metric label="X win" value={pct(recent.x_win_rate)} />
      <Metric label="O win" value={pct(recent.o_win_rate)} />
      <Metric label="Draw" value={pct(recent.draw_rate)} />
      <Metric label="Value" value={fixed(latest?.value ?? analysis?.value)} />
      <div className="metric policy-cell">
        <span>Top policy moves</span>
        <div>{top.length ? top.map((m) => <b key={m.move}>({m.x},{m.y},{m.z}) {fixed(m.prob, 2)}</b>) : 'none'}</div>
      </div>
    </section>
  );
}

function Metric({ label, value }) {
  return <div className="metric"><span>{label}</span><strong>{value}</strong></div>;
}

function Summary({ analysis, game }) {
  const top = analysis?.top_moves?.slice(0, 4) || [];
  const status = game?.done ? (game.winner === 0 ? 'draw' : `${game.winner === 1 ? 'X' : 'O'} wins`) : 'active';
  return (
    <div className="summary-grid play-summary">
      <Metric label="Value" value={fixed(analysis?.value)} />
      <Metric label="To move" value={analysis?.player === -1 ? 'O' : 'X'} />
      <Metric label="Status" value={status} />
      <div className="metric policy-cell">
        <span>Top moves</span>
        <div>{top.length ? top.map((m) => <b key={m.move}>({m.x},{m.y},{m.z}) {fixed(m.prob, 2)}</b>) : 'new game pending'}</div>
      </div>
    </div>
  );
}

function PolicyTable({ analysis }) {
  const top = analysis?.top_moves?.slice(0, 8) || [];
  return (
    <table className="policy-table">
      <thead><tr><th>rank</th><th>move</th><th>probability</th><th>value</th></tr></thead>
      <tbody>
        {top.map((m, index) => (
          <tr key={m.move}>
            <td>{index + 1}</td>
            <td>({m.x}, {m.y}, {m.z})</td>
            <td>{fixed(m.prob, 4)}</td>
            <td>{fixed(m.value ?? analysis.value)}</td>
          </tr>
        ))}
        {!top.length && <tr><td colSpan="4">No policy distribution available.</td></tr>}
      </tbody>
    </table>
  );
}

function ModelCatalog({ models, selectedModel, onSelect }) {
  return (
    <section className="paper-section compact-section">
      <h2>Model Catalog</h2>
      <div className="list-stack">
        {models.slice(0, 12).map((model) => (
          <button
            className={model.id === selectedModel ? 'row-button active' : 'row-button'}
            key={model.id}
            onClick={() => onSelect(model.id)}
          >
            <span>{model.label || model.id}</span>
            <small>{model.kind || 'model'} · size {model.size || 3}</small>
          </button>
        ))}
      </div>
    </section>
  );
}

function Artifacts({ latest }) {
  if (!latest?.run_dir) return null;
  const files = ['analysis.md', 'analysis.json', 'curves.png', 'first_move_heatmap.png', 'first_move_policy.json', 'model.pt'];
  return (
    <section className="paper-section compact-section">
      <h2>Artifacts</h2>
      <div className="artifact-row">
        {files.map((file) => (
          <a key={file} href={`/api/artifact?run_dir=${encodeURIComponent(latest.run_dir)}&file=${file}`} target="_blank" rel="noreferrer">{file}</a>
        ))}
      </div>
    </section>
  );
}

function RunList({ runs, onLoad }) {
  return (
    <section className="paper-section compact-section">
      <h2>Saved Runs</h2>
      <div className="list-stack">
        {runs.slice(0, 12).map((r) => (
          <button className="row-button" key={r.run_dir} onClick={() => onLoad(r.run_dir)}>
            <span>{methodLabel(r.method)} · {r.run_id || r.run_dir}</span>
            <small>X {pct(r.recent?.x_win_rate)} · O {pct(r.recent?.o_win_rate)} · ep {r.episode || 0}</small>
          </button>
        ))}
      </div>
    </section>
  );
}

function MoveButtons({ analysis, onMove }) {
  const moves = useMemo(() => {
    const size = analysis?.heatmap?.length || 3;
    return (analysis?.legal_moves || []).map((move) => ({ move, ...moveCoord(move, size) }));
  }, [analysis]);
  return (
    <div className="move-panel">
      <h2>Legal Coordinates</h2>
      <div className="move-grid">
        {moves.map((m) => (
          <button key={m.move} disabled={!onMove} onClick={() => onMove?.(m.move)}>
            {m.x},{m.y},{m.z}
          </button>
        ))}
        {!moves.length && <p>No legal moves loaded.</p>}
      </div>
    </div>
  );
}

function MoveList({ game }) {
  const history = game?.history || game?.moves || [];
  return (
    <div className="move-panel">
      <h2>History</h2>
      <ol className="history-list">
        {history.map((move, index) => (
          <li key={`${index}-${JSON.stringify(move)}`}>{typeof move === 'number' ? move : JSON.stringify(move)}</li>
        ))}
        {!history.length && <li>Start a game to make moves.</li>}
      </ol>
    </div>
  );
}

const path = window.location.pathname;
const root = createRoot(document.getElementById('root'));
if (path.startsWith('/play')) root.render(<PlayApp />);
else if (path.startsWith('/lab') || path.startsWith('/runs')) root.render(<LabApp />);
else root.render(<Home />);
