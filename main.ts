// gta6-trophy-rarity-forecast-01 — forecast Exophase GTA VI PS5 trophy earn rates at the case's target.
//
// First principles: a trophy's earn rate is earners / players-tracked. Both move. New players
// joining Exophase dilute every rate (they have earned nothing yet); existing players progressing
// through the game raise the earners. The rate itself follows a completion curve: it rises fast
// while the first wave of players reaches the trophy, then flattens as the rest are the slow
// ones — so the forecast fits a saturating curve, rate(t) = Rinf / (1 + (t0/t)^k), to the
// case's own history by grid search, and shrinks it toward a class prior (story chapter, named
// mission, side strand, grind/collectible, easy activity) built from how Rockstar trophies
// progress in week one. A straight line in log time would keep the early slope forever and
// overshoot by tens of points; the saturating form cannot. Spreads are honest: a prediction
// interval from the fit's residuals plus terms for refresh staleness and dilution, widening
// with the horizon. Deterministic, standard library only, no network, no clock, no randomness.
//
// Entry: node /bot/main.ts (Node 22), reads every <id>.in in $MAKEFUN_INPUT_DIR (/input), writes
// <id>.out to $MAKEFUN_OUTPUT_DIR (/output): {"trophies":[{"id":n,"q":[q05,q25,q50,q75,q95]}]}.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}
interface Read {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: Array<number | null>;
}
interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: { name: string; platform: string; release_date: string; source: string };
  trophies: Trophy[];
  history: Read[];
  notes: string;
}

const clamp = (x: number, lo: number, hi: number): number => Math.min(hi, Math.max(lo, x));
/** Earn rate in percent, clamped the way the scorer clamps before the logit. */
const clampRate = (x: number): number => clamp(x, 0.01, 99.99);
const logit = (pct: number): number => {
  const p = clampRate(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-clamp(l, -9.21, 9.21)));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
/** Hours between two ISO instants. */
const hoursBetween = (a: string, b: string): number =>
  (Date.parse(b) - Date.parse(a)) / 3_600_000;

/**
 * A saturating curve fit: rate(t) = Rinf / (1 + (t0 / t)^k).
 *
 * A trophy's earn rate is a completion CDF: it rises fast while the first wave of players
 * reaches it, then flattens as the remaining players are the slow ones. A straight line in
 * (log t, logit rate) space — the house trend baseline — keeps rising at the early slope
 * forever and overshoots the target by tens of points. The saturating form cannot do that:
 * it bends over. Fit by grid search over (t0, k) with Rinf from weighted least squares.
 */
function saturatingFit(
  ts: number[],
  rates: number[],
  ws: number[],
): { rInf: number; t0: number; k: number; residSd: number; r2: number; sse: number } {
  const T0_GRID = [2, 3, 5, 8, 12, 20, 30, 45, 70, 110, 170, 260, 400, 600, 900, 1400];
  const K_GRID = [0.8, 1.0, 1.2, 1.5, 1.8, 2.2];
  const mean = rates.reduce((s, r, i) => s + ws[i]! * r, 0) / ws.reduce((s, w) => s + w, 0);
  const sst = rates.reduce((s, r, i) => s + ws[i]! * (r - mean) * (r - mean), 0);
  let best: { rInf: number; t0: number; k: number; sse: number } | null = null;
  for (const t0 of T0_GRID) {
    for (const k of K_GRID) {
      let num = 0;
      let den = 0;
      const basis = ts.map((t) => 1 / (1 + Math.pow(t0 / Math.max(t, 0.05), k)));
      for (let i = 0; i < ts.length; i++) {
        num += ws[i]! * basis[i]! * rates[i]!;
        den += ws[i]! * basis[i]! * basis[i]!;
      }
      if (den <= 1e-12) continue;
      const rInf = clamp(num / den, 0, 100);
      let sse = 0;
      for (let i = 0; i < ts.length; i++) {
        const r = rates[i]! - rInf * basis[i]!;
        sse += ws[i]! * r * r;
      }
      if (best === null || sse < best.sse) best = { rInf, t0, k, sse };
    }
  }
  if (best === null) return { rInf: mean, t0: 30, k: 1.5, residSd: 0, r2: 0, sse: 0 };
  const sw = ws.reduce((s, w) => s + w, 0);
  const residSd = Math.sqrt(best.sse / Math.max(sw, 1e-9));
  return { ...best, residSd, r2: sst > 1e-12 ? clamp(1 - best.sse / sst, 0, 1) : 0 };
}

// ---------------------------------------------------------------------------
// Trophy classes and their week-one priors.
//
// A trophy's rate at game-age t hours follows a saturating curve in log time:
//   prior(t) = Rinf / (1 + (t0 / t)^k)
// Rinf is the rate the trophy settles toward once the early rush is over, t0 the game-age at
// which it reaches half of Rinf. Values are Rockstar week-one orders of magnitude: story
// chapters are finished by a large share of players in days; 100% completion and the platinum
// need hundreds of hours, so almost nobody holds them in week one; grinds sit in between.
// These are shrinkage targets, not answers: a strong fit in the case's own history overrides
// them, a flat or noisy history leans on them.
// ---------------------------------------------------------------------------
type TrophyClass =
  | 'platinum'
  | 'completion'
  | 'final'
  | 'chapter'
  | 'story'
  | 'side'
  | 'grind'
  | 'easy';

interface Prior {
  cls: TrophyClass;
  /** Ultimate (settled) rate, percent. */
  rInf: number;
  /** Game-age in hours at half of Rinf. */
  t0: number;
  /** Steepness of the rise in log time. */
  k: number;
  /** Hard sanity cap on the forecast median, percent. */
  cap: number;
}

const CHAPTER_RINF = [55, 45, 36, 28];
const CHAPTER_T0 = [30, 45, 60, 80];

function classify(t: Trophy): Prior {
  const name = `${t.name} ${t.description}`.toLowerCase();
  if (t.tier === 'platinum') return { cls: 'platinum', rInf: 0.15, t0: 500, k: 1.2, cap: 1.0 };
  if (/100% completion/.test(name)) return { cls: 'completion', rInf: 0.4, t0: 400, k: 1.2, cap: 2.0 };
  if (/final mission/.test(name) && !/for /.test(name)) {
    return { cls: 'final', rInf: 18, t0: 90, k: 1.5, cap: 60 };
  }
  const ch = /complete chapter (\d)/.exec(name);
  if (ch) {
    const n = clamp(parseInt(ch[1]!, 10), 1, 4);
    return { cls: 'chapter', rInf: CHAPTER_RINF[n - 1]!, t0: CHAPTER_T0[n - 1]!, k: 1.5, cap: 92 };
  }
  if (/final mission for /.test(name)) return { cls: 'side', rInf: 14, t0: 90, k: 1.3, cap: 55 };
  if (
    /250 stunts|gold medal|100 locations|card sets|50 challenges|every spot|unique animal|8 mission strands|bank vaults|explore all of|garage properties|laundering|discover 100 vehicles|earn a gold/.test(
      name,
    )
  ) {
    return { cls: 'grind', rInf: 2.5, t0: 150, k: 1.2, cap: 15 };
  }
  if (/complete (the |[a-z]+ )?(heist|run|play|takedown|operation|child)/.test(name)) {
    return { cls: 'story', rInf: 20, t0: 60, k: 1.4, cap: 65 };
  }
  return { cls: 'easy', rInf: 20, t0: 35, k: 1.4, cap: 80 };
}

/** The class prior's rate at game-age t hours. */
const priorRate = (p: Prior, tHours: number): number =>
  (p.rInf / (1 + Math.pow(p.t0 / Math.max(tHours, 0.5), p.k)));

// ---------------------------------------------------------------------------
// The forecast.
// ---------------------------------------------------------------------------

/**
 * Forecast one trophy's five quantiles.
 *
 * @param series change-point reads for this trophy: {tHours since release, rate, players}
 * @param nSeries the players-tracked series (all reads): {tHours, players}
 * @param targetHours game-age at the target, hours since release
 * @param horizonDays horizon in days
 * @param checkpointHours game-age at the checkpoint
 * @param prior the trophy's class prior
 */
function forecastTrophy(
  series: Array<{ t: number; rate: number; players: number }>,
  nSeries: Array<{ t: number; players: number }>,
  targetHours: number,
  horizonDays: number,
  checkpointHours: number,
  prior: Prior,
): number[] {
  const last = series[series.length - 1]!;
  const lastRate = last.rate;

  // A trophy every tracked player holds stays held: its rate cannot fall by progression,
  // only by dilution, and launch night showed it sticky at 100% through a 6x influx.
  if (lastRate >= 99.5) {
    const l = logit(99.9);
    return Z.map((z) => round4(clampRate(invLogit(l + z * 0.5))));
  }

  const ts = series.map((s) => s.t);
  const rates = series.map((s) => s.rate);
  // Recency weights: a read a day old counts half as much as the latest one.
  const ws = series.map((s) => Math.pow(0.5, (last.t - s.t) / 24));
  // A refresh that jumps the tracked count dilutes every rate in one step; the point right
  // after such a jump is a level shift, not signal about the curve — downweight it.
  for (let i = 1; i < series.length; i++) {
    const prevN = series[i - 1]!.players;
    if (prevN > 0) {
      const jump = Math.abs(Math.log(series[i]!.players / prevN));
      if (jump > 0.1) ws[i] = ws[i]! / (1 + 3 * jump);
    }
  }

  const flat =
    series.length < 3 || Math.max(...rates) - Math.min(...rates) < 0.05;
  const fit = flat
    ? { rInf: lastRate, t0: 30, k: 1.5, residSd: 0, r2: 0, sse: 0 }
    : saturatingFit(ts, rates, ws);
  const fitRate = clampRate(fit.rInf / (1 + Math.pow(fit.t0 / Math.max(targetHours, 0.5), fit.k)));

  // Blend the fit with the class prior in logit space. A strong fit (many change points,
  // high R2) earns most of the weight; a flat or noisy history leans on the prior. The fit
  // on early data cannot pin down the curve's ceiling — many ceilings fit the same steep
  // rise — so a fit that claims the rate will multiply by the target gets shrunk harder.
  const priorT = priorRate(prior, targetHours);
  let wFit = flat ? 0.35 : 0.65;
  if (!flat && series.length >= 8 && fit.r2 >= 0.6) wFit = 0.75;
  else if (!flat && fit.r2 < 0.3) wFit = 0.5;
  if (fitRate > 2.5 * Math.max(lastRate, 0.05)) wFit *= 0.8;
  let median = invLogit(wFit * logit(fitRate) + (1 - wFit) * logit(priorT));

  // Launch-night dilution: when the page has not refreshed at all since release (flat
  // players-tracked, flat rates) and the target is hours away, the next refresh brings the
  // tracked count up to the players who have actually started — launch night's first refresh
  // took it 638 -> 3,868 — and every rate falls with it. Earners grow only as fast as the
  // earliest players play, so predict the dilution, not the flat page.
  const nFlat = nSeries.length > 0 && nSeries.every((s) => s.players === nSeries[0]!.players);
  if (nFlat && checkpointHours < 36 && horizonDays * 24 < 12) {
    const hoursAhead = targetHours - checkpointHours;
    const nMult = Math.pow(1 + hoursAhead / 1.5, 1.0);
    const eMult = 1 + hoursAhead / 8;
    const diluted = (lastRate * eMult) / nMult;
    // The dilution forecast replaces the fits here: with a flat history there is no trend
    // to fit, and the prior curve cannot know the refresh is coming.
    median = clampRate(0.5 * median + 0.5 * diluted);
  }

  // Sanity caps from the class prior: no median above what the class can reach in week one.
  median = clamp(median, 0.01, prior.cap);

  // ---- Spread: honest, in logit units. ----
  // Base random-walk term: a week of launch is not a random walk, it is a rush with unknown
  // refresh timing, but the predictive error of the median is only ~0.3-0.5 logit units, so
  // the base stays close to that — a spread far wider pays at every quantile.
  let spread = 0.32 * Math.sqrt(horizonDays);
  // Fit uncertainty: the residuals' scale in rate points, converted to logit units at the
  // median (a residual of r points is ~r / (median * (1 - median/100)) logit units).
  const residLogit = fit.residSd / Math.max(median * (1 - median / 100), 0.5);
  spread = Math.sqrt(spread * spread + Math.pow(0.7 * residLogit, 2));
  // Rare trophies and grinds move in jumps; give them a little room.
  if (prior.cls === 'grind' || prior.cls === 'platinum' || prior.cls === 'completion') spread += 0.05;
  if (lastRate < 1) spread += 0.05;
  // A refresh can land anywhere in the target hour and the page it prints is as of its own
  // last refresh, so the answer is staler than the target by hours: widen for that.
  spread += 0.05;
  // The flat-page launch case is a guess at a refresh, not a measurement.
  if (nFlat && checkpointHours < 36 && horizonDays * 24 < 12) spread += 0.35;
  spread = clamp(spread, 0.3, 5);

  const l = logit(median);
  const qs = Z.map((z) => round4(clampRate(invLogit(l + z * spread))));
  // Keep the row non-decreasing (rounding can tie, never invert).
  for (let i = 1; i < qs.length; i++) qs[i] = Math.max(qs[i]!, qs[i - 1]!);
  return qs;
}

function forecastCase(c: Case): { id: number; q: number[] }[] {
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) {
    throw new Error(`${c.id} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  }
  if (c.history.length === 0) throw new Error(`${c.id} has no history`);
  const release = c.game?.release_date
    ? `${c.game.release_date}T00:00:00Z`
    : c.history[0]!.slot;
  const targetHours = Math.max(hoursBetween(release, c.target), 0.5);
  const checkpointHours = Math.max(hoursBetween(release, c.at), 0);
  const horizonDays = c.horizon_minutes / 1440;

  // Players-tracked series, all reads (refresh jumps included: the N fit needs them).
  const nSeries = c.history.map((r) => ({
    t: Math.max(hoursBetween(release, r.at), 0),
    players: r.players_tracked,
  }));

  return c.trophies.map((t, i) => {
    // Change points for this trophy: keep a read when its rate or the tracked count moved.
    const series: Array<{ t: number; rate: number; players: number }> = [];
    let prevRate: number | null = null;
    let prevPlayers = -1;
    for (const r of c.history) {
      const rate = r.rates[i];
      if (rate === null || rate === undefined) continue;
      const th = Math.max(hoursBetween(release, r.at), 0);
      if (rate !== prevRate || r.players_tracked !== prevPlayers) {
        series.push({ t: th, rate, players: r.players_tracked });
      }
      prevRate = rate;
      prevPlayers = r.players_tracked;
    }
    if (series.length === 0) {
      // The trophy never appeared in a read: say ~nothing, wide.
      const l = logit(0.05);
      return { id: t.id, q: Z.map((z) => round4(clampRate(invLogit(l + z * 1.5)))) };
    }
    // Always end the series on the last read, even if it duplicates the previous point.
    const lastRead = c.history[c.history.length - 1]!;
    const lastRate = lastRead.rates[i];
    if (lastRate !== null && lastRate !== undefined) {
      const lastPoint = series[series.length - 1]!;
      if (lastPoint.rate !== lastRate || lastPoint.players !== lastRead.players_tracked) {
        series.push({
          t: Math.max(hoursBetween(release, lastRead.at), 0),
          rate: lastRate,
          players: lastRead.players_tracked,
        });
      }
    }
    return {
      id: t.id,
      q: forecastTrophy(series, nSeries, targetHours, horizonDays, checkpointHours, classify(t)),
    };
  });
}

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
