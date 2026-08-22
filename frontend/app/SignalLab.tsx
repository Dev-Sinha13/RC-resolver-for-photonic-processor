"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  DEFAULT_CONFIG,
  MODEL_META,
  formatPercent,
  runLab,
  type LabResult,
  type LinkConfig,
  type ModelKey,
} from "./simulation";

function SignalChart({ result, selected }: { result: LabResult; selected: ModelKey }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const box = canvas.getBoundingClientRect();
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(box.width * ratio));
    canvas.height = Math.max(1, Math.round(box.height * ratio));
    const context = canvas.getContext("2d");
    if (!context) return;
    context.scale(ratio, ratio);
    const width = box.width;
    const height = box.height;
    context.clearRect(0, 0, width, height);
    const css = getComputedStyle(document.documentElement);
    const grid = css.getPropertyValue("--line") || "#26314b";
    const muted = css.getPropertyValue("--muted") || "#8f9bb7";
    const ink = css.getPropertyValue("--ink") || "#eef5ff";
    const accent = MODEL_META[selected].color;
    const left = 48;
    const right = width - 18;
    const top = 28;
    const signalBottom = height * 0.61;
    const predictionTop = height * 0.72;
    const bottom = height - 24;

    context.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
    context.fillStyle = muted;
    context.fillText("DETECTOR WAVEFORM", left, 16);
    context.fillText("TARGET / PREDICTION", left, predictionTop - 12);
    context.strokeStyle = grid;
    context.lineWidth = 1;
    for (let line = 0; line <= 4; line += 1) {
      const y = top + (signalBottom - top) * line / 4;
      context.beginPath();
      context.moveTo(left, y);
      context.lineTo(right, y);
      context.stroke();
    }

    const displayedSymbols = Math.min(42, result.targetBits.length);
    const values = result.waveform.slice(
      result.waveformStart,
      result.waveformStart + displayedSymbols * result.samplesPerSymbol,
    );
    const min = Math.min(...values);
    const max = Math.max(...values);
    context.strokeStyle = "#40d9ff";
    context.lineWidth = 1.8;
    context.beginPath();
    values.forEach((value, index) => {
      const x = left + (right - left) * index / Math.max(1, values.length - 1);
      const y = signalBottom - (signalBottom - top) * (value - min) / Math.max(1e-6, max - min);
      if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
    });
    context.stroke();

    const drawLane = (laneValues: number[], color: string, offset: number) => {
      context.strokeStyle = color;
      context.lineWidth = offset === 0 ? 2 : 2.5;
      context.beginPath();
      laneValues.slice(0, displayedSymbols).forEach((value, index) => {
        const x0 = left + (right - left) * index / displayedSymbols;
        const x1 = left + (right - left) * (index + 1) / displayedSymbols;
        const y = bottom - value * (bottom - predictionTop) + offset;
        if (index === 0) context.moveTo(x0, y); else context.lineTo(x0, y);
        context.lineTo(x1, y);
      });
      context.stroke();
    };
    drawLane(result.targetBits, muted, -3);
    drawLane(result.models[selected].predictions, accent, 3);
    context.fillStyle = ink;
    context.fillText("0", 28, bottom + 3);
    context.fillText("1", 28, predictionTop + 3);
  }, [result, selected]);

  return (
    <canvas
      ref={canvasRef}
      className="signal-canvas"
      role="img"
      aria-label={`Received detector waveform with target bits and ${MODEL_META[selected].label} predictions`}
    />
  );
}

function RangeControl({
  label,
  value,
  min,
  max,
  step,
  unit,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  unit: string;
  onChange: (value: number) => void;
}) {
  return (
    <label className="range-control">
      <span><b>{label}</b><output>{value.toFixed(step < 1 ? 2 : 0)} {unit}</output></span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

const explanations: Record<ModelKey, string> = {
  raw: "The receiver decides from one calibrated sample. It has no temporal model to undo inter-symbol interference.",
  ffe: "A learned linear combination of the current and previous 16 waveform samples reverses channel memory.",
  esn: "A fixed recurrent tanh network converts recent samples into a high-dimensional memory state.",
  photonic: "A masked detector sample drives virtual nodes in a nonlinear delay loop. Only the final linear readout is trained.",
};

export default function SignalLab() {
  const [config, setConfig] = useState(DEFAULT_CONFIG);
  const [selected, setSelected] = useState<ModelKey>("photonic");
  const [seed, setSeed] = useState(731);
  const result = useMemo(() => runLab(config, seed), [config, seed]);
  const selectedModel = result.models[selected];
  const update = (key: keyof LinkConfig, value: number) => {
    setConfig((current) => ({ ...current, [key]: value }));
  };

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Photonic Signal Lab home">
          <span className="brand-mark" aria-hidden="true"><i /><i /><i /></span>
          <span>PHOTONIC SIGNAL LAB</span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#lab">Live lab</a>
          <a href="#how">How it works</a>
          <a href="#results">Verified results</a>
        </nav>
        <a className="source-link" href="https://github.com/Dev-Sinha13/RC-resolver-for-photonic-processor" target="_blank" rel="noreferrer">
          View source <span aria-hidden="true">↗</span>
        </a>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <p className="eyebrow"><span /> Interactive optical receiver</p>
          <h1>Watch a damaged light signal <em>become readable again.</em></h1>
          <p className="hero-lede">
            Send bits through a simulated fibre link, add real optical impairments,
            and compare how linear, digital recurrent, and photonic reservoir
            receivers reconstruct the message—causally, one symbol at a time.
          </p>
          <div className="hero-actions">
            <a className="primary-button" href="#lab">Open the live experiment <span>↓</span></a>
            <span className="run-note"><b>Runs in your browser</b><small>No setup or dataset download</small></span>
          </div>
        </div>
        <div className="hero-signal" aria-hidden="true">
          <div className="beam beam-a" />
          <div className="beam beam-b" />
          <div className="signal-card">
            <span>PHOTONIC RECOVERY</span>
            <strong>74.2%</strong>
            <small>fewer bit errors in the verified run</small>
            <div className="mini-wave"><i /><i /><i /><i /><i /><i /><i /><i /></div>
          </div>
        </div>
      </section>

      <section className="lab-section" id="lab">
        <div className="section-heading">
          <div><p className="eyebrow"><span /> Live experiment</p><h2>Stress the fibre. Test the receiver.</h2></div>
          <p>Every control reruns a browser-scale experiment with a new detector waveform, recurrent state history, trained readout, and held-out BER.</p>
        </div>

        <div className="lab-shell">
          <aside className="controls-panel">
            <div className="panel-title"><span>01</span><div><b>Channel conditions</b><small>Change the physics</small></div></div>
            <RangeControl label="Fibre length" value={config.distance} min={0} max={50} step={1} unit="km" onChange={(value) => update("distance", value)} />
            <RangeControl label="Launch power" value={config.power} min={0} max={17} step={1} unit="dBm" onChange={(value) => update("power", value)} />
            <RangeControl label="Detector SNR" value={config.snr} min={8} max={30} step={1} unit="dB" onChange={(value) => update("snr", value)} />
            <RangeControl label="TX/RX bandwidth" value={config.bandwidth} min={3} max={10} step={0.5} unit="GHz" onChange={(value) => update("bandwidth", value)} />
            <RangeControl label="Timing jitter" value={config.jitter} min={0} max={0.1} step={0.01} unit="UI" onChange={(value) => update("jitter", value)} />
            <div className="control-actions">
              <button onClick={() => setSeed((current) => current + 101)}>Generate new bits</button>
              <button className="quiet-button" onClick={() => { setConfig(DEFAULT_CONFIG); setSeed(731); }}>Reset</button>
            </div>
            <div className="channel-readout">
              <span><small>Fibre loss</small><b>{result.lossDb.toFixed(1)} dB</b></span>
              <span><small>Estimated memory</small><b>{result.memoryUi} UI</b></span>
            </div>
          </aside>

          <div className="visual-panel">
            <div className="panel-toolbar">
              <div><span className="live-dot" /> LIVE SIGNAL</div>
              <div className="legend"><span><i className="cyan" />Detector</span><span><i className="muted-dot" />Target</span><span><i style={{ background: MODEL_META[selected].color }} />Prediction</span></div>
            </div>
            <SignalChart result={result} selected={selected} />
            <div className="model-tabs" role="tablist" aria-label="Recovery model">
              {(Object.keys(MODEL_META) as ModelKey[]).map((key) => (
                <button key={key} className={selected === key ? "active" : ""} onClick={() => setSelected(key)} role="tab" aria-selected={selected === key}>
                  <span style={{ background: MODEL_META[key].color }} />
                  <b>{MODEL_META[key].label}</b>
                  <small>{formatPercent(result.models[key].ber)} BER</small>
                </button>
              ))}
            </div>
          </div>

          <aside className="prediction-panel">
            <div className="panel-title"><span>02</span><div><b>Prediction</b><small>Held-out symbols</small></div></div>
            <div className="score-ring" style={{ "--score": `${Math.max(0, 100 - selectedModel.ber * 100) * 3.6}deg`, "--score-color": MODEL_META[selected].color } as React.CSSProperties}>
              <div><strong>{(100 - selectedModel.ber * 100).toFixed(1)}%</strong><small>symbol accuracy</small></div>
            </div>
            <div className="prediction-stats">
              <span><small>Bit-error rate</small><b>{formatPercent(selectedModel.ber)}</b></span>
              <span><small>Errors</small><b>{selectedModel.errors} / {result.targetBits.length}</b></span>
              <span><small>Architecture</small><b>{selectedModel.kind}</b></span>
            </div>
            <div className="bit-preview">
              <small>Target → prediction</small>
              {result.targetBits.slice(0, 14).map((bit, index) => (
                <span key={index} className={selectedModel.predictions[index] === bit ? "correct" : "error"}>{bit}→{selectedModel.predictions[index]}</span>
              ))}
            </div>
            <p className="model-explainer">{explanations[selected]}</p>
          </aside>
        </div>
        <p className="simulation-note"><b>About this live lab:</b> it is a fast educational approximation designed for instant interaction. The verified results below come from the repository’s higher-fidelity split-step Fourier simulation.</p>
      </section>

      <section className="how-section" id="how">
        <div className="section-heading compact">
          <div><p className="eyebrow"><span /> Under the hood</p><h2>From message to recovered bits.</h2></div>
          <p>The system stays causal: a decision may use the detector waveform up to the current receiver time, never future information.</p>
        </div>
        <div className="pipeline">
          {[
            ["01", "Encode", "A known binary message becomes an on–off optical intensity waveform."],
            ["02", "Propagate", "Fibre loss, chromatic dispersion, and Kerr nonlinearity reshape the light."],
            ["03", "Detect", "A bandwidth-limited photodetector adds electrical noise and timing jitter."],
            ["04", "Remember", "Recurrent reservoir states preserve recent waveform context as fading memory."],
            ["05", "Resolve", "A ridge-regression readout maps the fixed states back to transmitted symbols."],
          ].map(([number, title, copy], index) => (
            <article key={number}>
              <div className="pipeline-icon"><span>{number}</span></div><h3>{title}</h3><p>{copy}</p>
              {index < 4 && <i className="connector" aria-hidden="true">→</i>}
            </article>
          ))}
        </div>
        <div className="science-grid">
          <article><span>FIBRE MODEL</span><h3>Split-step Fourier propagation</h3><p>The research code alternates linear dispersion and attenuation in frequency with nonlinear Kerr phase rotation in time.</p><code>A(z + Δz) ≈ L½ · N · L½ · A(z)</code></article>
          <article><span>RESERVOIR STATE</span><h3>Fixed dynamics, simple training</h3><p>The optical loop is never backpropagated through. Its nonlinear states form a temporal feature map; only a regularized linear output is fitted.</p><code>ŷ(t) = Wout · x(t) + b</code></article>
          <article><span>PHOTONIC NONLINEARITY</span><h3>Mach–Zehnder-like response</h3><p>A sin² transfer curve approximates intensity modulation while delayed feedback gives virtual nodes recent context.</p><code>xᵢ(t) = sin²(αxᵢ(t−1) + βmᵢu(t) + φ)</code></article>
        </div>
      </section>

      <section className="results-section" id="results">
        <div className="section-heading compact">
          <div><p className="eyebrow"><span /> Python-verified benchmark</p><h2>Promising, without hiding the controls.</h2></div>
          <p>3,800 held-out decisions at 10 Gbaud over 25 km. Lower BER is better. The photonic reservoir helps substantially, while simpler baselines remain important.</p>
        </div>
        <div className="results-table" role="table" aria-label="Verified optical equalization results">
          <div className="result-row result-head" role="row"><span>Receiver</span><span>Bit errors</span><span>BER</span><span>Relative scale</span></div>
          {[
            ["Raw receiver", 128, 0.033684, "raw"],
            ["17-tap FFE", 1, 0.000263, "ffe"],
            ["Digital ESN", 1, 0.000263, "esn"],
            ["Photonic delay reservoir", 33, 0.008684, "photonic"],
          ].map(([name, errors, ber, key]) => (
            <div className={`result-row ${key === "photonic" ? "highlight" : ""}`} role="row" key={String(key)}>
              <span><i style={{ background: MODEL_META[key as ModelKey].color }} />{String(name)}{key === "photonic" && <small>FOCUS MODEL</small>}</span>
              <span>{String(errors)} / 3,800</span><span>{Number(ber).toFixed(6)}</span>
              <span><b style={{ width: `${Math.max(2, Number(ber) / 0.033684 * 100)}%`, background: MODEL_META[key as ModelKey].color }} /></span>
            </div>
          ))}
        </div>
        <div className="result-callout"><strong>74.2%</strong><p>fewer bit errors than the calibrated raw receiver</p><span>The photonic model removed 95 of 128 errors. The FFE and ESN were stronger on this particular channel, preventing an exaggerated superiority claim.</span></div>
      </section>

      <footer>
        <div><span className="brand-mark" aria-hidden="true"><i /><i /><i /></span><b>Photonic Signal Lab</b></div>
        <p>Built to make reservoir computing and optical signal recovery inspectable—not magical.</p>
        <a href="https://github.com/Dev-Sinha13/RC-resolver-for-photonic-processor" target="_blank" rel="noreferrer">Explore the research code ↗</a>
      </footer>
    </main>
  );
}
