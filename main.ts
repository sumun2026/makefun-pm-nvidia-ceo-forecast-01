// polymarket-nvidia-ceo-forecast-01 entry (Node 22, standard library only, no network).
//
// The forecast blends two shapes around the market's last price x (persistence, the centre): the
// market's own EMPIRICAL h-step distribution, which keeps its fat tails and its jump sizes, with the
// live-volatility NORMAL shape x + z_tau * K * sigma * sqrt(h), which keeps the market's current
// regime that an old window can no longer see. The empirical part alone is blind to a fresh change
// of volatility; the normal part alone understates the tail risk of a jumpy market, so each covers
// the other's blind spot and the blend is weighted to score best on the pinball loss.
//
// The forecast for one market at horizon h is a blend of two shapes around its last price x:
//
//   (a) the EMPIRICAL shape: the market's own distribution of h-step price changes over the last
//       WINDOW minutes of its history (every STRIDE minutes), centred on its median and scaled by
//       SCALE. It carries the market's own jump sizes and its fat tails.
//
//   (b) the NORMAL shape: x + z_tau * K * sigma * sqrt(h), where sigma is the market's live
//       volatility, the standard deviation of its last VOL_WINDOW one-minute changes. It carries the
//       market's current regime, which the window of (a) can be too old to see.
//
//   q_tau(h) = x + LAMBDA * SCALE * (emp_tau(h) - emp_median(h)) + (1 - LAMBDA) * z_tau * K * sigma * sqrt(h)
//
// The median stays at x (persistence): a leave-one-case-out check on the practice showed every drift
// term generalising worse than persistence. The scale of the spread (K, LAMBDA, SCALE) was fitted on
// the practice cases only; the shape of the optimum is flat, so the numbers are not knife-edge.
//
// The programme reads every <id>.in in the input directory, in name order, and writes one <id>.out per
// case: {"forecasts":[{"id","q"}]}, every market of the case exactly once, q exactly horizon_minutes
// rows of five non-decreasing numbers in [0,1], rounded to 6 decimals. Two runs write the same bytes:
// there is no randomness anywhere in it.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level, to 6 decimals. */
const Z = [-1.644854, -0.67449, 0, 0.67449, 1.644854];

/** Minutes of history the empirical h-step distribution is built from. */
const WINDOW = 1440;
/** Minutes between the origins sampled for that distribution. */
const STRIDE = 2;
/** The fewest samples the empirical distribution needs before it is used. */
const MIN_SAMPLES = 30;
/** Minutes of recent one-minute changes that estimate the market's live volatility. */
const VOL_WINDOW = 120;
/** The live volatility's scale in the normal part. */
const K = 0.45;
/** The weight of the empirical shape in the blend. */
const LAMBDA = 0.45;
/** The empirical part's scale. */
const SCALE = 0.35;
/** The smallest live volatility a market is given, in price units. */
const VOL_FLOOR = 0.0005;
/** The house's own spread, for a market with too little history to estimate any volatility. */
const FALLBACK_SIGMA = 0.0025;

interface CaseMarket {
  id: string;
  prices: number[];
}

interface Case {
  horizon_minutes: number;
  markets: CaseMarket[];
}

/** A price clamped to [0,1] and rounded to 6 decimals (0, never -0). */
const price6 = (x: number): number => Math.round(Math.min(1, Math.max(0, x)) * 1e6) / 1e6 + 0;

/** The quantile of a sorted array, by linear interpolation between the two neighbours of the level. */
function quantile(sorted: number[], p: number): number {
  const n = sorted.length;
  if (n === 0) return 0;
  const pos = p * (n - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) return sorted[lo] as number;
  return (sorted[lo] as number) + ((sorted[hi] as number) - (sorted[lo] as number)) * (pos - lo);
}

/** The standard deviation of the last `window` one-minute changes of a price series. */
function liveSigma(prices: number[], window: number): number {
  const n = prices.length;
  if (n < 4) return FALLBACK_SIGMA;
  const from = Math.max(1, n - window);
  let sum = 0;
  let sumSq = 0;
  let count = 0;
  for (let i = from; i < n; i++) {
    const d = (prices[i] as number) - (prices[i - 1] as number);
    sum += d;
    sumSq += d * d;
    count++;
  }
  const mean = sum / count;
  const variance = Math.max(0, sumSq / count - mean * mean);
  return Math.max(VOL_FLOOR, Math.sqrt(variance));
}

/**
 * The market's own h-step change distribution, for every horizon: the offsets of each level from the
 * distribution's median. Null for a horizon with too few samples to estimate.
 */
function empiricalOffsets(prices: number[], horizon: number): (number[] | null)[] {
  const n = prices.length;
  const from = Math.max(0, n - WINDOW);
  const offsets: (number[] | null)[] = [];
  const samples: number[] = [];
  for (let h = 1; h <= horizon; h++) {
    const last = n - 1 - h;
    samples.length = 0;
    for (let t = from; t <= last; t += STRIDE) {
      samples.push((prices[t + h] as number) - (prices[t] as number));
    }
    if (samples.length < MIN_SAMPLES) {
      offsets.push(null);
      continue;
    }
    samples.sort((a, b) => a - b);
    const median = quantile(samples, 0.5);
    offsets.push(QUANTILES.map((tau) => quantile(samples, tau) - median));
  }
  return offsets;
}

/** One market's rows: the five quantiles for every minute after the checkpoint. */
function forecast(m: CaseMarket, horizon: number): number[][] {
  const prices = m.prices;
  if (prices.length === 0) {
    return Array.from({ length: horizon }, () => QUANTILES.map(() => 0.5));
  }
  const x = prices[prices.length - 1] as number;
  const sigma = liveSigma(prices, VOL_WINDOW);
  const offsets = empiricalOffsets(prices, horizon);
  const rows: number[][] = [];
  for (let h = 1; h <= horizon; h++) {
    const spread = K * sigma * Math.sqrt(h);
    const emp = offsets[h - 1];
    const row: number[] = [];
    for (let j = 0; j < QUANTILES.length; j++) {
      const normal = (Z[j] as number) * spread;
      const value = emp ? x + LAMBDA * SCALE * (emp[j] as number) + (1 - LAMBDA) * normal : x + normal;
      row.push(price6(value));
    }
    rows.push(row);
  }
  return rows;
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const forecasts = c.markets.map((m) => ({ id: m.id, q: forecast(m, c.horizon_minutes) }));
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ forecasts })}\n`);
}
