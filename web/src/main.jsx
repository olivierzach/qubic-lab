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
  opponent_mix: 'self:0.4,tactical:0.4,random:0.2',
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

const pct = (value) => {
  const number = Number(value);
  return Number.isFinite(number) ? `${Math.round(number * 100)}%` : '--';
};
const fixed = (value, places = 3) => Number(value || 0).toFixed(places);
const methodLabel = (method) => String(method || 'run').toUpperCase();
const shortMix = (mix) => {
  const value = String(mix || '');
  if (!value) return '';
  if (value.includes('tactical:1')) return 'tactical';
  return value.replaceAll(':0.', ':.').replace(',random:0', '');
};
const runLabel = (run) => {
  const mix = shortMix(run?.config?.opponent_mix);
  return `${methodLabel(run?.method || run?.config?.method)} · ${run?.run_id || run?.run_dir} · ep ${run?.episode || 0}${mix ? ` · ${mix}` : ''}`;
};
const modelMeta = (model) => {
  const parts = [model?.kind || 'model', `size ${model?.size || 3}`];
  if (model?.episode) parts.push(`ep ${model.episode}`);
  if (model?.opponent_mix) parts.push(shortMix(model.opponent_mix));
  if (model?.recent_x != null) parts.push(`X ${pct(model.recent_x)}`);
  return parts.filter(Boolean).join(' · ');
};
const formatBytes = (bytes = 0) => {
  if (!bytes) return '';
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
};

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
  const [artifacts, setArtifacts] = useState([]);
  const [snapshotIndex, setSnapshotIndex] = useState(0);
  const [analysis, setAnalysis] = useState(null);
  const [sourceMode, setSourceMode] = useState('live');
  const [busy, setBusy] = useState('');
  const [notice, setNotice] = useState('');
  const [error, setError] = useState('');
  const [evalGames, setEvalGames] = useState(200);
  const [evalResult, setEvalResult] = useState(null);
  const [stateSpaceGames, setStateSpaceGames] = useState(80);
  const [stateSpace, setStateSpace] = useState(null);

  const selected = models.find((m) => m.id === selectedModel);
  const snapshots = timeline.length ? timeline : (runData.history || []).filter((d) => d.heatmap);
  const activeSnapshot = snapshots[snapshotIndex] || runData.latest || analysis;
  const previousSnapshot = snapshots[Math.max(0, snapshotIndex - 1)] || null;

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
    setArtifacts(data.artifacts || data.latest?.artifacts || []);
    setSourceMode('saved');
    setSnapshotIndex(0);
    if ((data.artifacts || []).some((item) => ['model.pt', 'q_table.npz'].includes(item.file))) {
      setSelectedModel(data.latest?.run_dir || runDir);
    }
  };

  const loadTimeline = async (modelId = selectedModel) => {
    if (!modelId) return;
    const data = await apiMaybe(`/api/model/timeline?model_id=${encodeURIComponent(modelId)}`, { timeline: [] });
    setTimeline(Array.isArray(data) ? data : data.timeline || data.snapshots || []);
    if (data?.artifacts) setArtifacts(data.artifacts);
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

  useEffect(() => {
    const runDir = runData.latest?.run_dir;
    if (!runDir) {
      setArtifacts([]);
      return;
    }
    apiMaybe(`/api/artifacts?run_dir=${encodeURIComponent(runDir)}`, { artifacts: [] })
      .then((data) => setArtifacts(data?.artifacts || runData.latest?.artifacts || []));
  }, [runData.latest?.run_dir]);

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
      await api('/api/run', { method: 'POST', body });
      setSourceMode('live');
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
      setNotice('');
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy('');
    }
  };

  const stepRun = async () => {
    setBusy('step');
    setNotice('');
    setError('');
    try {
      const data = await api('/api/step', { method: 'POST', body: JSON.stringify(runConfig) });
      setSourceMode('live');
      setRunData({ latest: data.latest, history: data.history || [] });
      setArtifacts(data.artifacts || []);
      await loadRuns();
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy('');
    }
  };

  const resetStepRun = async () => {
    setBusy('reset-step');
    setError('');
    try {
      await api('/api/step/reset', { method: 'POST' });
      setRunData({ latest: null, history: [] });
      setArtifacts([]);
      setNotice('');
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

  const evaluateSelected = async () => {
    if (!selectedModel || selectedModel === 'random') return;
    setBusy('eval');
    setError('');
    try {
      const size = selected?.size || runConfig.size || 3;
      const modelIds = selectedModel === 'tactical' ? [selectedModel] : [selectedModel, 'tactical'];
      const data = await api('/api/eval/tournament', {
        method: 'POST',
        body: JSON.stringify({ model_ids: modelIds, size, games: evalGames, seed: runConfig.seed || 0 }),
      });
      setEvalResult(data);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy('');
    }
  };

  const sampleStateSpace = async () => {
    setBusy('state-space');
    setError('');
    try {
      const size = selected?.size || runConfig.size || 3;
      const data = await api('/api/state-space/sample', {
        method: 'POST',
        body: JSON.stringify({
          model_id: selectedModel,
          size,
          games: stateSpaceGames,
          seed: runConfig.seed || 0,
          greedy: false,
        }),
      });
      setStateSpace(data);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy('');
    }
  };

  return (
    <>
      <main className="lab-page paper-width compact-lab">
        {(error || notice) && <div className={error ? 'message error' : 'message'}>{error || notice}</div>}

        <section className="lab-grid">
          <aside className="control-rail paper-section">
            <RunControls
              config={runConfig}
              onChange={updateConfig}
              busy={busy}
              onStart={startRun}
              onStop={stopRun}
              onStep={stepRun}
              onResetStep={resetStepRun}
            />

            <button onClick={showLive} disabled={!liveState?.latest}>Show live state</button>
            <label>Saved run
              <select value={selectedRun} onChange={(e) => setSelectedRun(e.target.value)}>
                <option value="">Select run</option>
                {runs.map((r) => <option key={r.run_dir} value={r.run_dir}>{runLabel(r)}</option>)}
              </select>
            </label>
            <button onClick={() => loadRun()} disabled={!selectedRun}>Load run</button>

            <label>Model
              <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}>
                {models.map((m) => <option key={m.id} value={m.id}>{m.label || m.id}</option>)}
              </select>
            </label>
            <button onClick={analyzeModel} disabled={!selectedModel || Boolean(busy)}>Analyze opening</button>
            <label>Eval games<input type="number" min="2" step="20" value={evalGames} onChange={(e) => setEvalGames(Number(e.target.value))} /></label>
            <div className="button-row">
              <button onClick={evaluateSelected} disabled={!selectedModel || selectedModel === 'random' || Boolean(busy)}>Evaluate</button>
              <a className="button-link" href={`/play?model=${encodeURIComponent(selectedModel)}`}>Play selected</a>
            </div>
            <label>State samples<input type="number" min="4" max="500" step="20" value={stateSpaceGames} onChange={(e) => setStateSpaceGames(Number(e.target.value))} /></label>
            <button onClick={sampleStateSpace} disabled={!selectedModel || Boolean(busy)}>Sample state map</button>

            <SystemInputs config={runConfig} selected={selected} latest={runData.latest} compact />
          </aside>

          <section className="paper-stack">
            <RunSummary latest={runData.latest} analysis={analysis} />
            <EvalSummary result={evalResult} selectedModel={selectedModel} />
            <div className="lab-focus-grid">
              <section className="paper-section plot-window">
                <MetricsChart history={runData.history || []} latest={runData.latest} />
              </section>
              <SnapshotViewer
                snapshot={activeSnapshot}
                snapshots={snapshots}
                snapshotIndex={snapshotIndex}
                previousSnapshot={previousSnapshot}
                onIndex={setSnapshotIndex}
                analysis={analysis}
                stateSpace={stateSpace}
              />
            </div>
            <div className="two-column">
              <ModelCatalog models={models} selectedModel={selectedModel} onSelect={setSelectedModel} />
              <RunList runs={runs} onLoad={(runDir) => { setSelectedRun(runDir); loadRun(runDir); }} />
            </div>
            <Artifacts latest={runData.latest} artifacts={artifacts} />
          </section>
        </section>
      </main>
    </>
  );
}

function RunControls({ config, onChange, busy, onStart, onStop, onStep, onResetStep }) {
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
          <label className="span-2">Opponent mix<input value={config.opponent_mix || 'self'} onChange={(e) => onChange('opponent_mix', e.target.value)} /></label>
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
      <div className="button-row span-2">
        <button onClick={onStep} disabled={Boolean(busy)}>{config.method === 'ppo' || config.method === 'grpo' ? 'Step batch' : 'Step episode'}</button>
        <button onClick={onResetStep} disabled={busy === 'reset-step'}>Reset step</button>
      </div>
    </div>
  );
}

function SystemInputs({ config, selected, latest, compact = false }) {
  const rows = [
    ['method', methodLabel(config.method)],
    ['episodes', config.episodes],
    ['batch', config.batch_episodes || 'n/a'],
    ['opponents', config.opponent_mix || 'self'],
    ['log every', config.log_every],
    ['lr / alpha', config.lr || config.alpha || 'n/a'],
    ['model', selected?.label || selected?.id || 'none'],
    ['model kind', selected?.kind || 'unknown'],
    ['run dir', latest?.run_dir || 'none'],
    ['latest episode', latest?.episode || 0],
  ];
  return (
    <section className={compact ? 'sidebar-block' : 'paper-section'}>
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

function SnapshotViewer({ snapshot, snapshots, snapshotIndex, previousSnapshot, onIndex, analysis, stateSpace }) {
  const max = Math.max(0, snapshots.length - 1);
  const source = snapshot || analysis;
  return (
    <>
      <section className="paper-section board-window">
        <LabBoard3D snapshot={source} />
      </section>

      <section className="paper-section heatmap-window">
        <div className="snapshot-toolbar">
          <span>{snapshots.length ? `${snapshotIndex + 1}/${snapshots.length}` : 'current'}</span>
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
        <ValueHeatmap snapshot={source} previous={previousSnapshot} />
      </section>

      <section className="paper-section policy-window">
        <PolicyTable analysis={source} />
      </section>

      <section className="paper-section state-window">
        <StateSpaceDiagram data={stateSpace} />
      </section>
    </>
  );
}

function heatmapStats(heatmap) {
  const values = flattenHeatmap(heatmap).map((p) => p.value).filter((v) => Number.isFinite(v));
  if (!values.length) return { min: -1, max: 1, maxAbs: 1 };
  const min = Math.min(...values);
  const max = Math.max(...values);
  return { min, max, maxAbs: Math.max(1e-6, Math.abs(min), Math.abs(max)), span: Math.max(1e-6, max - min) };
}

function heatmapAt(heatmap, x, y, z) {
  return Number(heatmap?.[z]?.[y]?.[x] || 0);
}

function heatmapTopMoves(heatmap, limit = 8) {
  const size = heatmap?.length || 0;
  return flattenHeatmap(heatmap)
    .map((p) => ({ ...p, move: moveIndex(p.x, p.y, p.z, size), prob: p.value }))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, limit);
}

function policyRankMap(snapshot, limit = 12) {
  const top = (snapshot?.top_moves || heatmapTopMoves(snapshot?.heatmap, limit)).slice(0, limit);
  return new Map(top.map((m, index) => [`${m.x}-${m.y}-${m.z}`, { ...m, rank: index + 1 }]));
}

function mixRgb(a, b, t) {
  const u = Math.max(0, Math.min(1, t));
  return a.map((value, index) => Math.round(value + (b[index] - value) * u));
}

function rgb(values) {
  return `rgb(${values[0]},${values[1]},${values[2]})`;
}

function cellTone(value, stats) {
  const neutral = [21, 24, 32];
  if (!Number.isFinite(value)) return rgb(neutral);
  if (stats.min < 0 && stats.max > 0) {
    const t = Math.min(1, Math.abs(value) / Math.max(stats.maxAbs, 1e-6));
    if (t < 0.03) return 'rgb(24,28,36)';
    const color = value >= 0 ? [98, 176, 141] : [208, 106, 88];
    return rgb(mixRgb(neutral, color, 0.22 + 0.78 * Math.sqrt(t)));
  }
  const raw = (value - stats.min) / Math.max(stats.span, 1e-6);
  const t = Math.pow(Math.max(0, Math.min(1, raw)), 0.55);
  if (t < 0.04) return 'rgb(22,25,33)';
  if (t < 0.5) return rgb(mixRgb(neutral, [76, 118, 158], t / 0.5));
  if (t < 0.86) return rgb(mixRgb([76, 118, 158], [98, 176, 141], (t - 0.5) / 0.36));
  return rgb(mixRgb([98, 176, 141], [213, 164, 71], (t - 0.86) / 0.14));
}

function cellIntensity(value, stats) {
  if (stats.min < 0 && stats.max > 0) return Math.min(1, Math.abs(value) / Math.max(stats.maxAbs, 1e-6));
  return Math.max(0, Math.min(1, (value - stats.min) / Math.max(stats.span, 1e-6)));
}

function ValueHeatmap({ snapshot, previous }) {
  const heatmap = snapshot?.heatmap;
  if (!heatmap) {
    return <div className="value-heatmap empty">No heatmap snapshot loaded.</div>;
  }
  const size = heatmap.length;
  const stats = heatmapStats(heatmap);
  const top = heatmapTopMoves(heatmap, 6);
  const policy = policyRankMap(snapshot, 10);
  return (
    <div className="value-heatmap">
      <div className="heatmap-meta">
        <span>episode {snapshot?.episode || 0}</span>
        <span>range {fixed(stats.min, 4)} to {fixed(stats.max, 4)}</span>
        <span>{snapshot?.method ? methodLabel(snapshot.method) : 'position'}</span>
      </div>
      <div className="layer-grid">
        {heatmap.map((layer, z) => (
          <section className="layer-panel" key={`z-${z}`}>
            <header>z = {z}</header>
            <div className="cells" style={{ gridTemplateColumns: `repeat(${size}, minmax(0, 1fr))` }}>
              {layer.map((row, y) =>
                row.map((value, x) => {
                  const current = Number(value || 0);
                  const delta = current - heatmapAt(previous?.heatmap, x, y, z);
                  const showDelta = Boolean(previous?.heatmap) && Math.abs(delta) >= 0.0005;
                  const isTop = top.some((m) => m.x === x && m.y === y && m.z === z);
                  const policyMove = policy.get(`${x}-${y}-${z}`);
                  return (
                    <div
                      className={isTop ? 'heat-cell top-cell' : 'heat-cell'}
                      key={`${x}-${y}-${z}`}
                      title={`(${x}, ${y}, ${z}) value ${fixed(current, 5)}${previous?.heatmap ? ` delta ${delta > 0 ? '+' : ''}${fixed(delta, 5)}` : ''}`}
                      style={{ background: cellTone(current, stats) }}
                    >
                      <b>{fixed(current, Math.abs(current) < 0.01 ? 4 : 3)}</b>
                      {policyMove && <em>#{policyMove.rank} {fixed(policyMove.prob ?? policyMove.value, 2)}</em>}
                      {showDelta && <small>{delta > 0 ? '+' : ''}{fixed(delta, 3)}</small>}
                    </div>
                  );
                }),
              )}
            </div>
          </section>
        ))}
      </div>
      <div className="heatmap-legend">
        <span className="low">low {fixed(stats.min, 3)}</span>
        <span className="mid">mid {fixed((stats.min + stats.max) / 2, 3)}</span>
        <span className="high">high {fixed(stats.max, 3)}</span>
      </div>
    </div>
  );
}

function LayerStackView({ snapshot }) {
  const heatmap = snapshot?.heatmap;
  if (!heatmap) return null;
  const size = heatmap.length;
  const stats = heatmapStats(heatmap);
  const policy = policyRankMap(snapshot, 8);
  return (
    <div className="stack-view" aria-label="Stacked value and policy projection">
      <div className="stack-copy">
        <h3>3D stack projection</h3>
        <p>Each z layer is a square board. Color is value, gold labels are top policy targets.</p>
      </div>
      <div className="stack-layers" style={{ '--layers': size }}>
        {heatmap.map((layer, z) => (
          <div className="stack-layer" style={{ '--z': z, '--size': size }} key={`stack-${z}`}>
            <header>z {z}</header>
            <div className="stack-grid" style={{ gridTemplateColumns: `repeat(${size}, minmax(0, 1fr))` }}>
              {layer.map((row, y) =>
                row.map((value, x) => {
                  const current = Number(value || 0);
                  const ranked = policy.get(`${x}-${y}-${z}`);
                  return (
                    <div
                      className={ranked ? 'stack-cell stack-top' : 'stack-cell'}
                      key={`${x}-${y}-${z}`}
                      title={`(${x}, ${y}, ${z}) value ${fixed(current, 4)}`}
                      style={{ background: cellTone(current, stats) }}
                    >
                      <span>{fixed(current, size <= 3 ? 2 : 1)}</span>
                      {ranked && <b>{ranked.rank}</b>}
                    </div>
                  );
                }),
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function StateSpaceDiagram({ data }) {
  const nodes = data?.nodes || [];
  if (!nodes.length) {
    return (
      <div className="state-map empty">
        <b>sampled state space</b>
        <span>Use Sample state map to roll out the selected model and plot reachable states by ply and value.</span>
      </div>
    );
  }
  const width = 760;
  const height = 210;
  const margin = { left: 34, right: 12, top: 18, bottom: 24 };
  const maxPly = Math.max(1, ...nodes.map((n) => Number(n.ply || 0)));
  const values = nodes.map((n) => Number(n.value || 0)).filter(Number.isFinite);
  const minValue = Math.min(-1, ...values);
  const maxValue = Math.max(1, ...values);
  const maxVisits = Math.max(1, ...nodes.map((n) => Number(n.visits || 1)));
  const xFor = (ply) => margin.left + (Number(ply || 0) / maxPly) * (width - margin.left - margin.right);
  const yFor = (value) => margin.top + (1 - ((Number(value || 0) - minValue) / Math.max(1e-6, maxValue - minValue))) * (height - margin.top - margin.bottom);
  const colorFor = (r) => {
    const value = Math.max(-1, Math.min(1, Number(r || 0)));
    return value >= 0 ? `rgba(98,176,141,${0.42 + 0.38 * value})` : `rgba(208,106,88,${0.42 + 0.38 * Math.abs(value)})`;
  };
  const rows = Array.from({ length: 5 }, (_, i) => minValue + (i / 4) * (maxValue - minValue));
  const cols = Array.from({ length: Math.min(6, maxPly + 1) }, (_, i) => Math.round((i / Math.max(1, Math.min(5, maxPly))) * maxPly));
  return (
    <div className="state-map">
      <div className="state-map-meta">
        <b>sampled state space</b>
        <span>{data.total_nodes} states</span>
        <span>{data.games} games</span>
        <span>D {pct(data.outcomes?.draw_rate)}</span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Sampled state-space values by ply">
        {rows.map((value) => (
          <g key={`row-${value}`}>
            <line x1={margin.left} x2={width - margin.right} y1={yFor(value)} y2={yFor(value)} />
            <text x={margin.left - 7} y={yFor(value) + 4} textAnchor="end">{fixed(value, 1)}</text>
          </g>
        ))}
        {cols.map((ply) => (
          <g key={`col-${ply}`}>
            <line x1={xFor(ply)} x2={xFor(ply)} y1={margin.top} y2={height - margin.bottom} />
            <text x={xFor(ply)} y={height - 6} textAnchor="middle">{ply}</text>
          </g>
        ))}
        <text x={margin.left} y={12}>value by ply, radius = visits, color = rollout return</text>
        {nodes.map((node) => {
          const radius = 2 + 7 * Math.sqrt(Number(node.visits || 1) / maxVisits);
          return (
            <circle
              key={node.id}
              cx={xFor(node.ply)}
              cy={yFor(node.value)}
              r={radius}
              fill={colorFor(node.return)}
            >
              <title>{`ply ${node.ply}  V ${fixed(node.value, 3)}  R ${fixed(node.return, 3)}  visits ${node.visits}`}</title>
            </circle>
          );
        })}
      </svg>
    </div>
  );
}

function LabBoard3D({ snapshot }) {
  const canvasRef = useRef(null);
  const viewRef = useRef({ yaw: -0.74, pitch: 0.86, dragging: false, last: null });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const draw = () => drawLabBoard(ctx, canvas, snapshot, viewRef.current);
    draw();

    const down = (event) => {
      viewRef.current.dragging = true;
      viewRef.current.last = { x: event.clientX, y: event.clientY };
      canvas.setPointerCapture(event.pointerId);
    };
    const move = (event) => {
      if (!viewRef.current.dragging || !viewRef.current.last) return;
      const dx = event.clientX - viewRef.current.last.x;
      const dy = event.clientY - viewRef.current.last.y;
      viewRef.current.yaw += dx * 0.01;
      viewRef.current.pitch = Math.max(0.2, Math.min(1.34, viewRef.current.pitch + dy * 0.01));
      viewRef.current.last = { x: event.clientX, y: event.clientY };
      draw();
    };
    const up = () => {
      viewRef.current.dragging = false;
      viewRef.current.last = null;
    };
    canvas.addEventListener('pointerdown', down);
    canvas.addEventListener('pointermove', move);
    canvas.addEventListener('pointerup', up);
    canvas.addEventListener('pointerleave', up);
    return () => {
      canvas.removeEventListener('pointerdown', down);
      canvas.removeEventListener('pointermove', move);
      canvas.removeEventListener('pointerup', up);
      canvas.removeEventListener('pointerleave', up);
    };
  }, [snapshot]);

  return <canvas ref={canvasRef} className="board3d lab-board3d" width="1200" height="720" />;
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
      const requested = new URLSearchParams(window.location.search).get('model');
      const preferred = nextModels.find((m) => m.id === requested) || nextModels.find((m) => m.kind === 'neural') || nextModels[0];
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

function tileCorners(p) {
  const r = 0.43;
  return [
    { x: p.x - r, y: p.y - r, z: p.z },
    { x: p.x + r, y: p.y - r, z: p.z },
    { x: p.x + r, y: p.y + r, z: p.z },
    { x: p.x - r, y: p.y + r, z: p.z },
  ];
}

function drawProjectedTile(ctx, corners, fill, stroke, lineWidth = 1.2) {
  ctx.beginPath();
  corners.forEach((p, index) => {
    if (index === 0) ctx.moveTo(p.x, p.y);
    else ctx.lineTo(p.x, p.y);
  });
  ctx.closePath();
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = lineWidth;
  ctx.stroke();
}

function drawLabBoard(ctx, canvas, snapshot, view) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#111318';
  ctx.fillRect(0, 0, w, h);

  if (!snapshot?.heatmap) {
    ctx.fillStyle = '#a9a192';
    ctx.font = '22px Georgia, serif';
    ctx.fillText('No value surface loaded.', 34, 46);
    return;
  }

  const size = snapshot.heatmap.length;
  const points = flattenHeatmap(snapshot.heatmap);
  const stats = heatmapStats(snapshot.heatmap);
  const scale = Math.min(w, h) / (size <= 3 ? 4.15 : 5.1);
  const cx = w * 0.5;
  const cy = h * 0.54;
  const board = snapshot.board || [];
  const occupied = new Map();
  board.forEach((layer, z) => layer?.forEach((row, y) => row?.forEach((v, x) => {
    if (v) occupied.set(moveIndex(x, y, z, size), v);
  })));
  const policy = policyRankMap(snapshot, 10);
  const topMove = [...policy.values()][0]?.move;
  const topRanks = new Set([...policy.values()].map((m) => m.move));

  ctx.save();
  ctx.globalAlpha = 0.82;
  drawCubeGrid(ctx, size, view, scale, cx, cy);
  ctx.restore();

  drawArrows(ctx, { ...snapshot, top_moves: [...policy.values()] }, view, scale, cx, cy);

  const tiles = points
    .map((p) => {
      const screen = project(p, size, view, scale, cx, cy);
      const corners = tileCorners(p).map((corner) => project(corner, size, view, scale, cx, cy));
      return { ...p, screen, corners };
    })
    .sort((a, b) => a.screen.depth - b.screen.depth);

  for (const p of tiles) {
    const idx = moveIndex(p.x, p.y, p.z, size);
    const occ = occupied.get(idx);
    const intensity = cellIntensity(p.value, stats);
    const ranked = policy.get(`${p.x}-${p.y}-${p.z}`);
    const fill = occ === 1 ? 'rgba(98,176,141,.92)' : occ === -1 ? 'rgba(208,106,88,.92)' : cellTone(p.value, stats);
    const stroke = idx === topMove ? '#d5a447' : ranked ? 'rgba(213,164,71,.72)' : 'rgba(240,234,220,.32)';
    drawProjectedTile(ctx, p.corners, fill, stroke, idx === topMove ? 3.4 : ranked ? 2.2 : 1.05);

    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    if (occ) {
      ctx.fillStyle = '#111318';
      ctx.font = 'bold 24px Georgia, serif';
      ctx.fillText(occ === 1 ? 'X' : 'O', p.screen.x, p.screen.y);
    } else if (size <= 4) {
      ctx.fillStyle = intensity > 0.72 ? '#111318' : '#f0eadc';
      ctx.font = '700 12px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillText(fixed(p.value, size <= 3 ? 2 : 1), p.screen.x, p.screen.y + (ranked ? 8 : 0));
    }
    if (ranked) {
      ctx.fillStyle = '#d5a447';
      ctx.beginPath();
      ctx.arc(p.screen.x, p.screen.y - 16, 12, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#111318';
      ctx.font = '800 12px ui-monospace, SFMono-Regular, Menlo, monospace';
      ctx.fillText(String(ranked.rank), p.screen.x, p.screen.y - 16);
    }
  }

  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = '#f0eadc';
  ctx.font = '18px Georgia, serif';
  const best = [...policy.values()][0];
  ctx.fillText(best ? `best (${best.x},${best.y},${best.z})  p=${fixed(best.prob ?? best.value, 3)}  value=${fixed(snapshot.value)}` : 'value surface', 30, 36);
  ctx.fillStyle = '#a9a192';
  ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillText(`n=${size}  gold=policy rank  green/red=value`, 30, h - 24);
}

function drawCubeGrid(ctx, size, view, scale, cx, cy) {
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
    for (let z = 0; z < size; z += 1) line({ x: 0, y, z }, { x: size - 1, y, z }, 'rgba(196,184,160,.18)');
    for (let x = 0; x < size; x += 1) line({ x, y, z: 0 }, { x, y, z: size - 1 }, 'rgba(196,184,160,.18)');
  }
  for (let x = 0; x < size; x += 1) {
    for (let z = 0; z < size; z += 1) line({ x, y: 0, z }, { x, y: size - 1, z }, 'rgba(196,184,160,.1)');
  }
}

function drawBoard(ctx, canvas, analysis, view, playable) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#111318';
  ctx.fillRect(0, 0, w, h);
  if (!analysis?.heatmap) {
    ctx.fillStyle = '#a9a192';
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
    for (let z = 0; z < size; z += 1) line({ x: 0, y, z }, { x: size - 1, y, z }, 'rgba(196,184,160,.24)');
    for (let x = 0; x < size; x += 1) line({ x, y, z: 0 }, { x, y, z: size - 1 }, 'rgba(196,184,160,.24)');
  }
  for (let x = 0; x < size; x += 1) {
    for (let z = 0; z < size; z += 1) line({ x, y: 0, z }, { x, y: size - 1, z }, 'rgba(196,184,160,.14)');
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
    ctx.fillStyle = occ === 1 ? '#62b08d' : occ === -1 ? '#d06a58' : valueColor(norm, legal.has(idx) ? 0.82 : 0.2);
    ctx.fill();
    ctx.strokeStyle = idx === bestMove ? '#d5a447' : playable && legal.has(idx) ? '#f0eadc' : 'rgba(220,210,190,.42)';
    ctx.lineWidth = idx === bestMove ? 5 : playable && legal.has(idx) ? 2 : 1;
    ctx.stroke();
    if (occ) {
      ctx.fillStyle = '#101216';
      ctx.font = 'bold 24px Georgia, serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(occ === 1 ? 'X' : 'O', p.screen.x, p.screen.y + 1);
    }
  }

  ctx.textAlign = 'left';
  ctx.textBaseline = 'alphabetic';
  ctx.fillStyle = '#f0eadc';
  ctx.font = '19px Georgia, serif';
  const best = analysis.top_moves?.[0];
  ctx.fillText(best ? `best (${best.x},${best.y},${best.z})  p=${fixed(best.prob)}  value=${fixed(analysis.value)}` : 'terminal position', 32, 38);
  ctx.fillStyle = '#a9a192';
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
    const weight = Math.min(1, Math.abs(Number(move.prob ?? move.value ?? 0)));
    const alpha = Math.max(0.18, Math.min(0.9, weight * 8));
    const color = `rgba(213,164,71,${alpha})`;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 + 5 * weight;
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
  if (t >= 0) return `rgba(${Math.round(70 + 58 * t)},${Math.round(142 + 72 * t)},${Math.round(116 + 38 * t)},${alpha})`;
  const u = -t;
  return `rgba(${Math.round(184 + 38 * u)},${Math.round(92 - 16 * u)},${Math.round(78 - 10 * u)},${alpha})`;
}

function MetricsChart({ history, latest }) {
  const canvasRef = useRef(null);
  const [hover, setHover] = useState(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const render = () => {
      const rect = canvas.getBoundingClientRect();
      const width = Math.max(640, Math.round(rect.width || 960));
      const height = Math.max(360, Math.round(rect.height || 512));
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      drawMetrics(ctx, { width, height }, history || [], latest || null, hover);
    };
    render();
    const observer = new ResizeObserver(render);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [history, latest?.episode, latest?.episodes, latest?.run_dir, hover]);
  const onMove = (event) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(640, Math.round(rect.width || 960));
    const height = Math.max(360, Math.round(rect.height || 512));
    const scaleX = width / Math.max(1, rect.width);
    const scaleY = height / Math.max(1, rect.height);
    const next = metricHoverFromPointer(
      history || [],
      latest || null,
      width,
      height,
      (event.clientX - rect.left) * scaleX,
      (event.clientY - rect.top) * scaleY,
    );
    setHover(next ? { ...next, left: next.left / scaleX, top: next.top / scaleY } : null);
  };
  return (
    <div className="chart-wrap" onMouseMove={onMove} onMouseLeave={() => setHover(null)}>
      <canvas ref={canvasRef} className="chart" />
      {hover && (
        <div className="chart-tooltip" style={{ left: hover.left, top: hover.top }}>
          <b>episode {axisLabel(hover.episode)}</b>
          {hover.items.map((item) => <span key={item.label} style={{ color: item.color }}>{item.label} {item.text}</span>)}
        </div>
      )}
    </div>
  );
}

function getMetricModel(history, latest, w, h) {
  const xs = history.map((d) => d.episode || d.step || 0);
  const minX = 0;
  const maxX = Math.max(1, Number(latest?.episodes || 0), Number(latest?.episode || 0), ...xs);
  const values = (getY) => history.map((d) => Number(getY(d))).filter((v) => Number.isFinite(v));
  const range = (items, fallbackMin, fallbackMax) => {
    if (!items.length) return [fallbackMin, fallbackMax];
    const min = Math.min(...items, fallbackMin);
    const max = Math.max(...items, fallbackMax);
    const pad = Math.max(0.02, (max - min) * 0.12);
    return [min - pad, max + pad];
  };
  const valueItems = [...values((d) => d.value ?? d.mean_value), ...values((d) => Math.abs(Number(d.mean_abs_update || 0)))];
  const entropyItems = values((d) => d.entropy);
  const klItems = values((d) => d.approx_kl).filter((v) => v > 0);
  const hasModelMetrics = history.some((d) => Number(d.recent_model?.window || 0) > 0) || Number(latest?.recent_model?.window || 0) > 0;
  const margin = { left: 64, right: 22, top: 34, bottom: 34 };
  const gap = 30;
  const panelH = (h - margin.top - margin.bottom - gap * 2) / 3;
  const panelW = w - margin.left - margin.right;
  const [valueMin, valueMax] = range(valueItems, -0.05, 0.05);
  const [entropyMin, entropyMax] = range(entropyItems, 0, 0.05);
  const klMin = Math.max(1e-7, klItems.length ? Math.min(...klItems) * 0.5 : 1e-6);
  const klMax = Math.max(klMin * 10, klItems.length ? Math.max(...klItems) * 2 : 1e-3);
  const panels = [
    { x: margin.left, y: margin.top, w: panelW, h: panelH, title: 'outcomes', ymin: 0, ymax: 1 },
    { x: margin.left, y: margin.top + panelH + gap, w: panelW, h: panelH, title: 'value/update', ymin: valueMin, ymax: valueMax },
    {
      x: margin.left,
      y: margin.top + (panelH + gap) * 2,
      w: panelW,
      h: panelH,
      title: 'policy stats',
      ymin: entropyMin,
      ymax: entropyMax,
      right: { ymin: klMin, ymax: klMax },
    },
  ];
  const defs = [
    hasModelMetrics ? [
      { label: 'M', color: '#62b08d', get: (d) => d.recent_model?.win_rate },
      { label: 'Opp', color: '#d06a58', get: (d) => d.recent_model?.loss_rate },
      { label: 'D', color: '#76a8cf', get: (d) => d.recent_model?.draw_rate },
      { label: 'Xside', color: '#d5a447', get: (d) => d.recent_model?.as_x_win_rate },
      { label: 'Oside', color: '#b79adb', get: (d) => d.recent_model?.as_o_win_rate },
    ] : [
      { label: 'X', color: '#62b08d', get: (d) => d.recent?.x_win_rate },
      { label: 'O', color: '#d06a58', get: (d) => d.recent?.o_win_rate },
      { label: 'D', color: '#76a8cf', get: (d) => d.recent?.draw_rate },
    ],
    [
      { label: 'V', color: '#7db5d7', get: (d) => d.value ?? d.mean_value },
      { label: '|dQ|', color: '#d5a447', get: (d) => Math.abs(Number(d.mean_abs_update || 0)) },
    ],
    [
      { label: 'H', color: '#b79adb', get: (d) => Number(d.entropy || 0) },
      { label: 'KL', color: '#8cc8d4', get: (d) => Number(d.approx_kl || 0), transform: 'log', ymin: klMin, ymax: klMax },
    ],
  ];
  return { minX, maxX, panels, defs };
}

function metricHoverFromPointer(history, latest, width, height, pointerX, pointerY) {
  if (history.length < 2) return null;
  const model = getMetricModel(history, latest, Math.max(640, width), Math.max(360, height));
  const activePanel = model.panels.find((p) => pointerY >= p.y && pointerY <= p.y + p.h);
  const plotPanel = activePanel || model.panels[0];
  if (pointerX < plotPanel.x || pointerX > plotPanel.x + plotPanel.w) return null;
  const episode = model.minX + ((pointerX - plotPanel.x) / Math.max(1, plotPanel.w)) * (model.maxX - model.minX);
  const nearest = history.reduce((best, item) => {
    const itemEpisode = item.episode || item.step || 0;
    const bestEpisode = best.episode || best.step || 0;
    return Math.abs(itemEpisode - episode) < Math.abs(bestEpisode - episode) ? item : best;
  }, history[0]);
  const nearestEpisode = nearest.episode || nearest.step || 0;
  const x = plotPanel.x + ((nearestEpisode - model.minX) / Math.max(1, model.maxX - model.minX)) * plotPanel.w;
  const items = model.defs.flatMap((group) =>
    group.map((item) => ({ label: item.label, color: item.color, value: Number(item.get(nearest)) }))
      .filter((item) => Number.isFinite(item.value))
      .map((item) => ({ ...item, text: metricLabel(item.value) })),
  );
  return {
    episode: nearestEpisode,
    row: nearest,
    left: Math.max(10, Math.min(width - 174, x + 12)),
    top: Math.max(8, Math.min(height - 150, pointerY + 12)),
    items,
  };
}

function drawMetrics(ctx, canvas, history, latest, hover) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#111318';
  ctx.fillRect(0, 0, w, h);
  if (history.length < 2) {
    ctx.fillStyle = '#a9a192';
    ctx.font = '17px Georgia, serif';
    ctx.fillText('Load, start, or step a run to draw live plots.', 44, 44);
    return;
  }
  const { minX, maxX, panels, defs } = getMetricModel(history, latest, w, h);
  panels.forEach((panel, index) => {
    axes(ctx, panel, minX, maxX);
    defs[index].forEach((item) => series(ctx, panel, history, minX, maxX, item));
    legend(ctx, panel, history, defs[index]);
  });
  if (hover?.row) {
    const episode = hover.row.episode || hover.row.step || 0;
    panels.forEach((panel, index) => {
      const x = panel.x + ((episode - minX) / Math.max(1, maxX - minX)) * panel.w;
      ctx.strokeStyle = 'rgba(240,234,220,.34)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, panel.y);
      ctx.lineTo(x, panel.y + panel.h);
      ctx.stroke();
      defs[index].forEach((item) => point(ctx, panel, episode, Number(item.get(hover.row)), minX, maxX, item));
    });
  }
}

function axisLabel(value) {
  if (!Number.isFinite(value)) return '0';
  if (Math.abs(value) >= 1000000) return `${(value / 1000000).toFixed(1)}M`;
  if (Math.abs(value) >= 1000) return `${Math.round(value / 1000)}k`;
  return String(Math.round(value));
}

function metricLabel(value) {
  if (!Number.isFinite(value)) return '--';
  if (Math.abs(value) > 0 && Math.abs(value) < 0.001) return value.toExponential(0);
  return value.toFixed(3);
}

function yNorm(value, min, max, transform) {
  if (transform === 'log') {
    if (value <= 0 || min <= 0 || max <= 0) return null;
    const lo = Math.log10(min);
    const hi = Math.log10(max);
    return (Math.log10(value) - lo) / Math.max(1e-9, hi - lo);
  }
  return (value - min) / Math.max(1e-9, max - min);
}

function axes(ctx, p, minX, maxX) {
  ctx.fillStyle = '#a9a192';
  ctx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
  ctx.fillText(p.title.toUpperCase(), p.x, p.y - 8);
  ctx.fillStyle = '#151820';
  ctx.fillRect(p.x, p.y, p.w, p.h);
  ctx.strokeStyle = '#3b424d';
  ctx.lineWidth = 1;
  ctx.strokeRect(p.x, p.y, p.w, p.h);
  ctx.fillStyle = '#a9a192';
  ctx.font = '10px ui-monospace, SFMono-Regular, Menlo, monospace';
  for (let i = 0; i <= 2; i += 1) {
    const frac = i / 2;
    const y = p.y + p.h - frac * p.h;
    const v = p.ymin + frac * (p.ymax - p.ymin);
    ctx.strokeStyle = i === 0 ? '#3b424d' : '#272d36';
    ctx.beginPath();
    ctx.moveTo(p.x, y);
    ctx.lineTo(p.x + p.w, y);
    ctx.stroke();
    ctx.fillText(v.toFixed(2), p.x - 52, y + 4);
  }
  ctx.fillStyle = '#756f66';
  ctx.fillText(axisLabel(minX), p.x, p.y + p.h + 16);
  ctx.textAlign = 'right';
  ctx.fillText(axisLabel(maxX), p.x + p.w, p.y + p.h + 16);
  if (p.right) {
    ctx.fillStyle = '#8cc8d4';
    for (let i = 0; i <= 2; i += 1) {
      const frac = i / 2;
      const y = p.y + p.h - frac * p.h;
      const lo = Math.log10(p.right.ymin);
      const hi = Math.log10(p.right.ymax);
      const value = 10 ** (lo + frac * (hi - lo));
      ctx.fillText(metricLabel(value), p.x + p.w - 5, y + 4);
    }
  }
  ctx.textAlign = 'left';
}

function series(ctx, p, history, minX, maxX, item) {
  const ymin = item.ymin ?? p.ymin;
  const ymax = item.ymax ?? p.ymax;
  const values = history
    .map((d) => ({ x: d.episode || d.step || 0, y: Number(item.get(d)) }))
    .filter((d) => Number.isFinite(d.y) && (item.transform !== 'log' || d.y > 0));
  if (values.length < 2) return;
  ctx.strokeStyle = item.color;
  ctx.lineWidth = 2.2;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  ctx.beginPath();
  values.forEach((d, i) => {
    const x = p.x + ((d.x - minX) / Math.max(1, maxX - minX)) * p.w;
    const norm = yNorm(d.y, ymin, ymax, item.transform);
    if (norm == null) return;
    const y = p.y + p.h - Math.max(0, Math.min(1, norm)) * p.h;
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  const last = values[values.length - 1];
  const x = p.x + ((last.x - minX) / Math.max(1, maxX - minX)) * p.w;
  const norm = yNorm(last.y, ymin, ymax, item.transform) ?? 0;
  const y = p.y + p.h - Math.max(0, Math.min(1, norm)) * p.h;
  ctx.fillStyle = item.color;
  ctx.beginPath();
  ctx.arc(x, y, 3.2, 0, Math.PI * 2);
  ctx.fill();
}

function point(ctx, p, episode, value, minX, maxX, item) {
  if (!Number.isFinite(value) || (item.transform === 'log' && value <= 0)) return;
  const ymin = item.ymin ?? p.ymin;
  const ymax = item.ymax ?? p.ymax;
  const norm = yNorm(value, ymin, ymax, item.transform);
  if (norm == null) return;
  const x = p.x + ((episode - minX) / Math.max(1, maxX - minX)) * p.w;
  const y = p.y + p.h - Math.max(0, Math.min(1, norm)) * p.h;
  ctx.fillStyle = item.color;
  ctx.strokeStyle = '#111318';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(x, y, 4, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
}

function legend(ctx, p, history, items) {
  let x = p.x + p.w - 8;
  ctx.textAlign = 'right';
  ctx.font = '700 10px ui-monospace, SFMono-Regular, Menlo, monospace';
  items.slice().reverse().forEach((item) => {
    const values = history.map((d) => Number(item.get(d))).filter((v) => Number.isFinite(v));
    const text = `${item.label} ${values.length ? fixed(values[values.length - 1], 3) : '--'}`;
    ctx.fillStyle = item.color;
    ctx.fillText(text, x, p.y - 8);
    x -= ctx.measureText(text).width + 14;
  });
  ctx.textAlign = 'left';
}

function RunSummary({ latest, analysis }) {
  const recent = latest?.recent || {};
  const modelRecent = latest?.recent_model || {};
  const hasModelMetrics = Number(modelRecent.window || 0) > 0;
  const top = (latest?.top_moves || analysis?.top_moves || []).slice(0, 4);
  return (
    <section className="summary-grid">
      <Metric label="Method" value={latest?.method ? methodLabel(latest.method) : 'none'} />
      <Metric label="Episode" value={latest?.episode || 0} />
      {hasModelMetrics ? (
        <>
          <Metric label="Model win" value={pct(modelRecent.win_rate)} />
          <Metric label="Opp win" value={pct(modelRecent.loss_rate)} />
          <Metric label="Draw" value={pct(modelRecent.draw_rate)} />
          <Metric label="As X" value={pct(modelRecent.as_x_win_rate)} />
          <Metric label="As O" value={pct(modelRecent.as_o_win_rate)} />
        </>
      ) : (
        <>
          <Metric label="X win" value={pct(recent.x_win_rate)} />
          <Metric label="O win" value={pct(recent.o_win_rate)} />
          <Metric label="Draw" value={pct(recent.draw_rate)} />
        </>
      )}
      <Metric label="Value" value={fixed(latest?.value ?? analysis?.value)} />
      <div className="metric policy-cell">
        <span>Top policy moves</span>
        <div>{top.length ? top.map((m) => <b key={m.move}>({m.x},{m.y},{m.z}) {fixed(m.prob, 2)}</b>) : 'none'}</div>
      </div>
    </section>
  );
}

function EvalSummary({ result, selectedModel }) {
  const matches = result?.matches || [];
  const opponents = ['random', 'tactical'].filter((id) => id !== selectedModel);
  const rows = opponents
    .map((opponent) => {
      const match = matches.find((item) =>
        (item.a === selectedModel && item.b === opponent) || (item.b === selectedModel && item.a === opponent),
      );
      if (!match) return null;
      const wins = match.wins || {};
      const modelWins = wins[selectedModel] || 0;
      const opponentWins = wins[opponent] || 0;
      const draws = wins.draw || 0;
      const games = match.games || result.games || Math.max(1, modelWins + opponentWins + draws);
      const score = (modelWins + 0.5 * draws) / Math.max(1, games);
      const asX = (match.records || []).filter((record) => record.x_model === selectedModel);
      const asO = (match.records || []).filter((record) => record.o_model === selectedModel);
      const asXWins = asX.filter((record) => record.winner === selectedModel).length;
      const asOWins = asO.filter((record) => record.winner === selectedModel).length;
      return {
        opponent,
        modelWins,
        opponentWins,
        draws,
        games,
        score,
        asXRate: asX.length ? asXWins / asX.length : null,
        asORate: asO.length ? asOWins / asO.length : null,
      };
    })
    .filter(Boolean);
  if (!rows.length) return null;
  return (
    <section className="eval-strip">
      <span>fixed eval</span>
      {rows.map((row) => (
        <React.Fragment key={row.opponent}>
          <b>{pct(row.score)}</b>
          <span>vs {row.opponent}</span>
          <span>W {row.modelWins}</span>
          <span>D {row.draws}</span>
          <span>L {row.opponentWins}</span>
          <span>X {pct(row.asXRate)}</span>
          <span>O {pct(row.asORate)}</span>
        </React.Fragment>
      ))}
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
  const top = analysis?.top_moves?.slice(0, 8) || heatmapTopMoves(analysis?.heatmap, 8);
  return (
    <table className="policy-table">
      <thead><tr><th>rank</th><th>move</th><th>weight</th><th>value</th></tr></thead>
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
      <div className="list-stack">
        {models.slice(0, 12).map((model) => (
          <button
            className={model.id === selectedModel ? 'row-button active' : 'row-button'}
            key={model.id}
            onClick={() => onSelect(model.id)}
          >
            <span>{model.label || model.id}</span>
            <small>{modelMeta(model)}</small>
          </button>
        ))}
      </div>
    </section>
  );
}

function Artifacts({ latest, artifacts }) {
  if (!latest?.run_dir) return null;
  const files = artifacts || latest.artifacts || [];
  return (
    <section className="paper-section compact-section">
      <div className="artifact-row">
        {files.map((item) => (
          <a key={item.file} href={item.url || `/api/artifact?run_dir=${encodeURIComponent(latest.run_dir)}&file=${item.file}`} target="_blank" rel="noreferrer">
            {item.label || item.file}
            <small>{formatBytes(item.bytes)}</small>
          </a>
        ))}
        {!files.length && <p>No completed artifacts yet. They appear after a run writes analysis and plots.</p>}
      </div>
    </section>
  );
}

function RunList({ runs, onLoad }) {
  return (
    <section className="paper-section compact-section">
      <div className="list-stack">
        {runs.slice(0, 12).map((r) => (
          <button className="row-button" key={r.run_dir} onClick={() => onLoad(r.run_dir)}>
            <span>{runLabel(r)}</span>
            <small>X {pct(r.recent?.x_win_rate)} · O {pct(r.recent?.o_win_rate)} · D {pct(r.recent?.draw_rate)}</small>
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
