// polymarket-nvidia-ceo-forecast-01 — quantile forecast for the "Next CEO of NVIDIA?"
// event. One case per run: read every *.in file of the input directory and write the
// matching *.out file.
//
// The forecast is persistence around the market's checkpoint price, with two pieces:
// the market's own empirical tail shape (the distribution of its h-step price changes,
// recentered on its median) and the standard normal scaled by its live volatility.
// The median forecast is therefore always the checkpoint price.

type CaseMarket = {
  id: string;
  prices: number[];
};

type CaseFile = {
  case: string;
  horizon_minutes: number;
  markets: CaseMarket[];
};

const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
const NORMAL_QUANTILES = [-1.644854, -0.67449, 0, 0.67449, 1.644854];

const VOLATILITY_WINDOW = 120;
const SHAPE_LOOKBACK = 1440;
const SHAPE_STRIDE = 2;
const SHAPE_MIN_SAMPLES = 30;
const SPREAD_SCALE = 0.45;
const SHAPE_WEIGHT = 0.35;
const NORMAL_WEIGHT = 0.55;
const MIN_VOLATILITY = 0.0005;
const FALLBACK_VOLATILITY = 0.0025;
const DEFAULT_HORIZON = 240;

const fs = process.getBuiltinModule("node:fs");
const path = process.getBuiltinModule("node:path");

function readCaseFile(filePath: string): CaseFile {
  const parsed = JSON.parse(fs.readFileSync(filePath, "utf8"));
  return parsed as CaseFile;
}

// population standard deviation of the last minute-to-minute moves
function liveVolatility(prices: number[]): number {
  const last = prices.length - 1;
  const first = Math.max(1, last + 1 - VOLATILITY_WINDOW);
  const moves = last + 1 - first;
  if (moves < 3) return FALLBACK_VOLATILITY;

  let sum = 0;
  for (let i = first; i <= last; i++) sum += prices[i] - prices[i - 1];
  const mean = sum / moves;

  let spread = 0;
  for (let i = first; i <= last; i++) {
    const deviation = prices[i] - prices[i - 1] - mean;
    spread += deviation * deviation;
  }
  const volatility = Math.sqrt(spread / moves);
  return volatility < MIN_VOLATILITY ? MIN_VOLATILITY : volatility;
}

// every second price of the last day of history can seed an h-step change
function originStops(historyLength: number): number[] {
  const last = historyLength - 1;
  const earliest = Math.max(0, last - SHAPE_LOOKBACK);
  const stops: number[] = [];
  for (let i = earliest; i <= last; i += SHAPE_STRIDE) stops.push(i);
  return stops;
}

// linear interpolation between order statistics, at position level * (count - 1)
function sampleQuantile(sorted: Float64Array, count: number, level: number): number {
  const position = level * (count - 1);
  const lower = Math.floor(position);
  const fraction = position - lower;
  if (lower + 1 >= count) return sorted[lower];
  return sorted[lower] + fraction * (sorted[lower + 1] - sorted[lower]);
}

// each level's distance from the median of the h-step changes, or null when the
// history is too short to describe the shape
function tailOffsets(
  prices: number[],
  stops: number[],
  step: number,
  scratch: Float64Array,
): number[] | null {
  const last = prices.length - 1;
  let count = 0;
  for (let s = 0; s < stops.length; s++) {
    const origin = stops[s];
    const future = origin + step;
    if (future > last) break;
    scratch[count] = prices[future] - prices[origin];
    count += 1;
  }
  if (count < SHAPE_MIN_SAMPLES) return null;

  const sorted = scratch.subarray(0, count);
  sorted.sort();
  const middle = sampleQuantile(sorted, count, 0.5);
  const offsets: number[] = [];
  for (const level of LEVELS) offsets.push(sampleQuantile(sorted, count, level) - middle);
  return offsets;
}

function sixDecimals(value: number): number {
  if (!Number.isFinite(value)) return 0.5;
  let rounded = Math.round(value * 1e6) / 1e6;
  if (rounded < 0) rounded = 0;
  else if (rounded > 1) rounded = 1;
  return rounded === 0 ? 0 : rounded;
}

// one row of five quantiles for a given minute ahead
function quantileRow(spot: number, offsets: number[] | null, spread: number): number[] {
  const row: number[] = [];
  let previous = 0;
  for (let level = 0; level < LEVELS.length; level++) {
    const normal = NORMAL_QUANTILES[level] * spread;
    const value = offsets
      ? spot + SPREAD_SCALE * SHAPE_WEIGHT * offsets[level] + NORMAL_WEIGHT * normal
      : spot + normal;
    let number = sixDecimals(value);
    if (number < previous) number = previous;
    previous = number;
    row.push(number);
  }
  return row;
}

function forecastRows(prices: number[], horizon: number): number[][] {
  const spot = prices[prices.length - 1];
  const volatility = liveVolatility(prices);
  const stops = originStops(prices.length);
  const scratch = new Float64Array(stops.length);
  const rows: number[][] = [];

  for (let step = 1; step <= horizon; step++) {
    const spread = SPREAD_SCALE * volatility * Math.sqrt(step);
    const offsets = tailOffsets(prices, stops, step, scratch);
    rows.push(quantileRow(spot, offsets, spread));
  }
  return rows;
}

function writeForecast(outputDir: string, name: string, caseFile: CaseFile): void {
  const horizon = caseFile.horizon_minutes || DEFAULT_HORIZON;
  const forecasts = caseFile.markets.map((market) => ({
    id: market.id,
    q: forecastRows(market.prices, horizon),
  }));
  const payload = JSON.stringify({ forecasts }) + "\n";
  fs.writeFileSync(path.join(outputDir, name.slice(0, -3) + ".out"), payload);
}

function run(): void {
  const inputDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outputDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  fs.mkdirSync(outputDir, { recursive: true });

  const names = fs
    .readdirSync(inputDir)
    .filter((name) => name.endsWith(".in"))
    .sort();

  for (const name of names) {
    writeForecast(outputDir, name, readCaseFile(path.join(inputDir, name)));
  }
}

run();
