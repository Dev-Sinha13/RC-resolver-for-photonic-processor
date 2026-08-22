export type ModelKey = "raw" | "ffe" | "esn" | "photonic";

export type LinkConfig = {
  distance: number;
  power: number;
  snr: number;
  bandwidth: number;
  jitter: number;
};

export type ModelResult = {
  key: ModelKey;
  label: string;
  kind: string;
  ber: number;
  errors: number;
  predictions: number[];
};

export type LabResult = {
  waveform: number[];
  waveformStart: number;
  samplesPerSymbol: number;
  targetBits: number[];
  models: Record<ModelKey, ModelResult>;
  lossDb: number;
  memoryUi: number;
};

export const DEFAULT_CONFIG: LinkConfig = {
  distance: 25,
  power: 10,
  snr: 18,
  bandwidth: 7.5,
  jitter: 0.02,
};

export const MODEL_META: Record<ModelKey, { label: string; kind: string; color: string }> = {
  raw: { label: "Raw receiver", kind: "Current sample only", color: "#8a92a6" },
  ffe: { label: "17-tap FFE", kind: "Linear memory", color: "#b678ff" },
  esn: { label: "Digital ESN", kind: "Recurrent tanh states", color: "#40d9ff" },
  photonic: { label: "Photonic reservoir", kind: "sin² delay dynamics", color: "#ffbd59" },
};

function randomGenerator(seed: number) {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function gaussian(random: () => number) {
  const u = Math.max(random(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
}

function interpolate(values: number[], position: number) {
  const bounded = Math.max(0, Math.min(values.length - 1, position));
  const left = Math.floor(bounded);
  const right = Math.min(values.length - 1, left + 1);
  const fraction = bounded - left;
  return values[left] * (1 - fraction) + values[right] * fraction;
}

function solveLinear(matrix: number[][], vector: number[]) {
  const size = vector.length;
  const augmented = matrix.map((row, index) => [...row, vector[index]]);
  for (let column = 0; column < size; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < size; row += 1) {
      if (Math.abs(augmented[row][column]) > Math.abs(augmented[pivot][column])) pivot = row;
    }
    [augmented[column], augmented[pivot]] = [augmented[pivot], augmented[column]];
    const diagonal = Math.abs(augmented[column][column]) < 1e-10 ? 1e-10 : augmented[column][column];
    for (let value = column; value <= size; value += 1) augmented[column][value] /= diagonal;
    for (let row = 0; row < size; row += 1) {
      if (row === column) continue;
      const factor = augmented[row][column];
      for (let value = column; value <= size; value += 1) {
        augmented[row][value] -= factor * augmented[column][value];
      }
    }
  }
  return augmented.map((row) => row[size]);
}

function fitRidge(features: number[][], targets: number[], penalty = 1e-3) {
  const width = features[0].length + 1;
  const gram = Array.from({ length: width }, () => Array(width).fill(0));
  const rhs = Array(width).fill(0);
  features.forEach((feature, row) => {
    const values = [1, ...feature];
    for (let i = 0; i < width; i += 1) {
      rhs[i] += values[i] * targets[row];
      for (let j = 0; j < width; j += 1) gram[i][j] += values[i] * values[j];
    }
  });
  for (let index = 1; index < width; index += 1) gram[index][index] += penalty;
  return solveLinear(gram, rhs);
}

function predict(weights: number[], features: number[][]) {
  return features.map((feature) =>
    weights[0] + feature.reduce((sum, value, index) => sum + value * weights[index + 1], 0),
  );
}

function bestThreshold(targets: number[], scores: number[]) {
  const unique = [...scores].sort((a, b) => a - b);
  let best = unique[0] - 1e-6;
  let fewest = Number.POSITIVE_INFINITY;
  for (const threshold of unique) {
    const errors = scores.reduce(
      (total, score, index) => total + (Number(score > threshold) !== targets[index] ? 1 : 0),
      0,
    );
    if (errors < fewest) {
      fewest = errors;
      best = threshold;
    }
  }
  return best;
}

function evaluateModel(
  key: ModelKey,
  trainFeatures: number[][],
  trainTargets: number[],
  testFeatures: number[][],
  testTargets: number[],
) {
  const weights = fitRidge(trainFeatures, trainTargets);
  const trainScores = predict(weights, trainFeatures);
  const scores = predict(weights, testFeatures);
  const threshold = bestThreshold(trainTargets, trainScores);
  const predictions = scores.map((score) => Number(score > threshold));
  const errors = predictions.reduce(
    (total, value, index) => total + (value !== testTargets[index] ? 1 : 0),
    0,
  );
  return {
    key,
    label: MODEL_META[key].label,
    kind: MODEL_META[key].kind,
    ber: errors / testTargets.length,
    errors,
    predictions,
  };
}

export function runLab(config: LinkConfig, seed: number): LabResult {
  const random = randomGenerator(seed);
  const samplesPerSymbol = 10;
  const bitCount = 180;
  const bits = Array.from({ length: bitCount }, () => Number(random() > 0.5));
  let waveform = bits.flatMap((bit) => Array(samplesPerSymbol).fill(bit));

  const filterStrength = Math.max(0.08, Math.min(0.92, config.bandwidth / 11));
  for (let index = 1; index < waveform.length; index += 1) {
    waveform[index] = waveform[index - 1] + filterStrength * (waveform[index] - waveform[index - 1]);
  }
  const blurPasses = Math.max(0, Math.round(config.distance / 12));
  for (let pass = 0; pass < blurPasses; pass += 1) {
    waveform = waveform.map((value, index, source) =>
      0.14 * source[Math.max(0, index - 2)]
      + 0.22 * source[Math.max(0, index - 1)]
      + 0.28 * value
      + 0.22 * source[Math.min(source.length - 1, index + 1)]
      + 0.14 * source[Math.min(source.length - 1, index + 2)],
    );
  }

  const lossDb = 0.2 * config.distance;
  const attenuation = 10 ** (-lossDb / 10);
  const nonlinearDrive = Math.max(0, config.power - 7) * config.distance / 650;
  const noiseStd = 0.52 * 10 ** (-config.snr / 20);
  waveform = waveform.map((value, index, source) => {
    const previousBit = source[Math.max(0, index - samplesPerSymbol)];
    const nonlinear = nonlinearDrive * value * previousBit * (1 - 0.35 * value);
    return attenuation * (value + nonlinear) + noiseStd * gaussian(random);
  });

  const mean = waveform.reduce((sum, value) => sum + value, 0) / waveform.length;
  const variance = waveform.reduce((sum, value) => sum + (value - mean) ** 2, 0) / waveform.length;
  const scale = Math.sqrt(variance) || 1;
  const normalized = waveform.map((value) => (value - mean) / scale);
  const decisionPositions = bits.map((_, index) =>
    index * samplesPerSymbol
    + 0.5 * (samplesPerSymbol - 1)
    + gaussian(random) * config.jitter * samplesPerSymbol,
  );
  const rawSamples = decisionPositions.map((position) => interpolate(normalized, position));

  const esnNodes = 28;
  const photonicNodes = 36;
  const esnRandom = randomGenerator(4103);
  const photonicRandom = randomGenerator(9001);
  const esnInput = Array.from({ length: esnNodes }, () => 2 * esnRandom() - 1);
  const esnBias = Array.from({ length: esnNodes }, () => 0.12 * (2 * esnRandom() - 1));
  const photonicMask = Array.from({ length: photonicNodes }, () => photonicRandom() > 0.5 ? 1 : -1);
  let esnState = Array(esnNodes).fill(0);
  let photonicState = Array(photonicNodes).fill(0);
  const esnTimeline: number[][] = [];
  const photonicTimeline: number[][] = [];

  normalized.forEach((sample) => {
    const previous = esnState;
    esnState = previous.map((state, node) => {
      const candidate = Math.tanh(
        0.5 * esnInput[node] * sample
        + 0.92 * previous[(node + esnNodes - 1) % esnNodes]
        + esnBias[node],
      );
      return 0.78 * state + 0.22 * candidate;
    });
    const delayed = photonicState;
    const updated = Array(photonicNodes).fill(0);
    let serial = delayed[photonicNodes - 1];
    for (let node = 0; node < photonicNodes; node += 1) {
      const drive = 0.5 * delayed[node] + 0.25 * photonicMask[node] * sample + Math.PI / 4;
      serial = 0.5 * serial + 0.5 * Math.sin(drive) ** 2;
      updated[node] = serial;
    }
    photonicState = updated;
    esnTimeline.push([...esnState]);
    photonicTimeline.push([...photonicState]);
  });

  const trainEnd = 105;
  const testStart = 120;
  const trainRows: Record<ModelKey, number[][]> = { raw: [], ffe: [], esn: [], photonic: [] };
  const testRows: Record<ModelKey, number[][]> = { raw: [], ffe: [], esn: [], photonic: [] };
  const trainTargets: number[] = [];
  const testTargets: number[] = [];

  for (let decision = 8; decision < bits.length; decision += 1) {
    const targetIndex = decision - 1;
    const sampleIndex = Math.max(0, Math.min(normalized.length - 1, Math.round(decisionPositions[decision])));
    const rows: Record<ModelKey, number[]> = {
      raw: [rawSamples[targetIndex]],
      ffe: Array.from({ length: 17 }, (_, tap) => normalized[Math.max(0, sampleIndex - tap)]),
      esn: esnTimeline[sampleIndex],
      photonic: photonicTimeline[sampleIndex],
    };
    if (targetIndex < trainEnd) {
      trainTargets.push(bits[targetIndex]);
      (Object.keys(rows) as ModelKey[]).forEach((key) => trainRows[key].push(rows[key]));
    } else if (targetIndex >= testStart) {
      testTargets.push(bits[targetIndex]);
      (Object.keys(rows) as ModelKey[]).forEach((key) => testRows[key].push(rows[key]));
    }
  }

  const models = Object.fromEntries(
    (Object.keys(MODEL_META) as ModelKey[]).map((key) => [
      key,
      evaluateModel(key, trainRows[key], trainTargets, testRows[key], testTargets),
    ]),
  ) as Record<ModelKey, ModelResult>;

  return {
    waveform: normalized,
    waveformStart: testStart * samplesPerSymbol,
    samplesPerSymbol,
    targetBits: testTargets,
    models,
    lossDb,
    memoryUi: Math.max(1, Math.round(1 + config.distance / 14 + (10 - config.bandwidth) / 2.5)),
  };
}

export function formatPercent(value: number) {
  return `${(100 * value).toFixed(2)}%`;
}
