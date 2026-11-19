"""GTA VI trophy rarity forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the
percent of Exophase-tracked players who will have earned it at the case's
target, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy, scored by
pinball loss on the logit scale.

Model, in one paragraph: a trophy's earn rate is a ratio, holders over
players tracked, and the two sides move for different reasons, so they are
projected separately and divided again at the target. Holders grow as the
incumbent players progress through the game; players tracked grows as new
profiles sync, and each sync dilutes every rate toward the newcomers'
much lower earn rate. Per trophy, log holders and log players tracked are
each fitted by least squares against log(1 + hours since release) over the
last 24 hours of reads (slopes shrunk toward zero and clipped), then read
at the target; the ratio of the two projections is the median. On top of
that sits a launch-week dilution prior: Exophase's page refreshes in
jumps, not continuously (launch night it sat unchanged for hours, then
jumped from 638 to 3,868 players tracked and every non-saturated rate
fell to roughly a fifth), so when the page has gone stale -- no rate has
changed for a while -- the median is shifted down by an amount that grows
with the staleness and with the tracked-player growth rate, a growth rate
whose prior is large on launch night and decays over the first days. The
spread combines the house random walk (0.5 logit per root-day), the two
fits' residuals, and a staleness term that is large when no refresh has
ever been observed and small once refreshes are visible in the history;
the lower quantiles are stretched a little wider than the upper ones,
because a refresh can only dilute rates sharply downward. A trophy
already at 100% is left there with a hairline spread: newcomers earn it
too. Everything is deterministic, standard library only, no network.

How the grader runs it: `python3 -P -s /bot/main.py` (Python 3.13,
standard library only). It reads every <id>.in in the input directory
($MAKEFUN_INPUT_DIR, else /input) and writes <id>.out to the output
directory ($MAKEFUN_OUTPUT_DIR, else /output): {"trophies": [{"id": n,
"q": [q05, q25, q50, q75, q95]}, ...]}, one row per trophy, each q five
finite numbers from 0 to 100, non-decreasing, at most 65,536 bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports. (This entry is one file, but keep the
# line so the layout stays valid if a helper is ever added.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# The spread of a rate's move, in logit units per square-root day (house).
SIGMA_RW = 0.5
# Launch-week dilution prior: logit shift per log unit of catch-up growth.
GAMMA = 2.0
# Prior tracked-player growth rate on launch night, per hour, decaying
# with a 24 h time constant as the launch flood passes.
GROWTH0 = 0.40
GROWTH_DECAY_HOURS = 24.0
# When refreshes are visible in the history, the fitted trends already
# carry the dilution, so the prior shift is damped to this fraction.
PRIOR_DAMP = 0.4
# Extra spread (logit) for a stale page whose refresh has never been seen.
SIGMA_STALE = 0.9
# When refreshes are visible, the staleness spread is damped to this.
STALE_DAMP = 0.3
# Lower quantiles stretch wider than upper ones (refreshes dilute down).
DOWN_STRETCH = 1.2
UP_STRETCH = 1.0
# Hard caps, in logit units, on how far a median may move from the last
# read, and on the total spread: a forecast must stay sane when a fit
# extrapolates wildly on a short or jumpy history.
MAX_MEDIAN_MOVE = 2.5
MAX_SIGMA = 1.8
# A trophy this saturated is earned by everyone, newcomers included.
SATURATED = 99.0
# Fallback rate (percent) when a trophy has no usable read at all.
DEFAULT_RATE = 0.5


def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100.0 - clamp(pct)))


def inv_logit(lg: float) -> float:
    lg = max(-30.0, min(30.0, lg))
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does.

    A whole number comes back as an int, so json.dumps writes 100 where
    JavaScript writes 100 (not 100.0).
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(text: str) -> datetime:
    return datetime.fromisoformat(str(text).replace("Z", "+00:00"))


def ols(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Least-squares intercept and slope of y on x (slope 0 if degenerate)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    return my - slope * mx, slope


def residual_std(xs: list[float], ys: list[float], a: float, b: float) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    ss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    return math.sqrt(ss / max(1, n - 2))


def shrink(value: float, n: int, lo: float, hi: float) -> float:
    """Shrink a fitted slope toward zero (n reads: factor n / (n + 4)),
    then clip it into [lo, hi]."""
    return max(lo, min(hi, value * n / (n + 4)))


def forecast_case(case: dict) -> dict:
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_days = float(case["horizon_minutes"]) / 1440.0

    release = case.get("game", {}).get("release_date") or "2026-11-19"
    epoch = parse_time(release + "T00:00:00Z")
    at = parse_time(case["at"])
    target = parse_time(case["target"])

    if not history:
        return {"trophies": [_flat_row(t["id"]) for t in trophies]}

    times = [(parse_time(h["at"]) - epoch).total_seconds() / 3600.0 for h in history]
    t_last = times[-1]
    x = [math.log1p(t) for t in times]
    # A target at or before the last read is a zero horizon, never a
    # backward extrapolation: the forecast is the last read, widened only
    # by the staleness terms below.
    dx = max(0.0, math.log1p((target - epoch).total_seconds() / 3600.0) - x[-1])
    players = [max(1, int(h.get("players_tracked") or 1)) for h in history]

    # Staleness: hours from the last read where any rate changed to `at`.
    # If no read ever changed, the page has been stale since the first read.
    last_change = 0
    for j in range(1, len(history)):
        if history[j]["rates"] != history[j - 1]["rates"]:
            last_change = j
    staleness = max(
        0.0, (at - parse_time(history[last_change]["at"])).total_seconds() / 3600.0
    )
    ever_changed = any(
        history[j]["rates"] != history[j - 1]["rates"] for j in range(1, len(history))
    )

    # Reads in the last 24 hours (all of them when the history is shorter).
    window = [j for j, t in enumerate(times) if t >= t_last - 24.0] or list(range(len(history)))
    wx = [x[j] for j in window]

    # Tracked-player growth: log N on log-time, and its residual spread.
    log_n = [math.log(p) for p in players]
    if len(window) >= 3:
        a_n, b_n = ols(wx, [log_n[j] for j in window])
        resid_n = residual_std(wx, [log_n[j] for j in window], a_n, b_n)
        b_n = shrink(b_n, len(window), 0.0, 3.0)
    else:
        b_n, resid_n = 0.0, 0.0

    # Growth rate per hour implied by the fit, floored by the launch-week
    # prior, which is large on launch night and decays over the first days.
    growth_fit = b_n / (1.0 + t_last) if b_n > 0 else 0.0
    growth_prior = GROWTH0 * math.exp(-t_last / GROWTH_DECAY_HOURS)
    growth = max(growth_fit, growth_prior)

    rows = []
    for i, trophy in enumerate(trophies):
        rates = [h["rates"][i] if i < len(h["rates"]) else None for h in history]
        last_rate = next((r for r in reversed(rates) if r is not None), None)
        if last_rate is None:
            last_rate = DEFAULT_RATE
        last_logit = logit(last_rate)

        # Holders per read: rate * players tracked (rate floored so a 0%
        # read still gives a positive holder count for the log fit).
        holders = [
            max(0.01, r if r is not None else 0.01) / 100.0 * players[j]
            for j, r in enumerate(rates)
        ]
        log_h = [math.log(max(h, 1e-9)) for h in holders]
        pts = [(x[j], log_h[j]) for j in window]
        if len(pts) >= 3:
            a_h, b_h = ols([p[0] for p in pts], [p[1] for p in pts])
            resid_h = residual_std([p[0] for p in pts], [p[1] for p in pts], a_h, b_h)
            b_h = shrink(b_h, len(pts), -0.2, 3.0)
        else:
            b_h, resid_h = 0.0, 0.0

        # Median: project holders and players separately, divide at target.
        p_target = math.exp((log_h[-1] + b_h * dx) - (math.log(players[-1]) + b_n * dx))
        p_target = max(0.0001, min(0.9999, p_target))
        median = math.log(p_target / (1.0 - p_target))
        median = max(last_logit - MAX_MEDIAN_MOVE, min(last_logit + MAX_MEDIAN_MOVE, median))

        # Launch-week dilution prior: a stale page's next refresh dilutes
        # every non-saturated rate toward the newcomers' lower earn rate.
        if last_rate < SATURATED:
            dilution = -GAMMA * math.log1p(growth * staleness)
            median += (PRIOR_DAMP if ever_changed else 1.0) * dilution
            median = max(
                last_logit - MAX_MEDIAN_MOVE, min(last_logit + MAX_MEDIAN_MOVE, median)
            )

        # Spread: random walk + both fits' residuals + staleness term.
        stale_w = STALE_DAMP if ever_changed else 1.0
        sigma = math.sqrt(
            (SIGMA_RW * math.sqrt(horizon_days)) ** 2
            + resid_h ** 2
            + resid_n ** 2
            + (stale_w * SIGMA_STALE * math.sqrt(min(staleness, 48.0) / 24.0)) ** 2
        )
        sigma = min(sigma, MAX_SIGMA)
        if last_rate >= SATURATED:
            sigma = min(sigma, 0.05)

        qs: list[float | int] = []
        for z in Z:
            stretch = DOWN_STRETCH if z < 0 else UP_STRETCH if z > 0 else 1.0
            qs.append(round4(clamp(inv_logit(median + z * sigma * stretch))))
        # Rounding must not break the non-decreasing order within a row.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def _flat_row(trophy_id: int) -> dict:
    """Fallback row when a case carries no history at all."""
    spread = SIGMA_RW * math.sqrt(1.0)
    median = logit(DEFAULT_RATE)
    qs: list[float | int] = []
    for z in Z:
        qs.append(round4(clamp(inv_logit(median + z * spread))))
    return {"id": trophy_id, "q": qs}


def _persistence_row(trophy_id: int, last_rate: float, horizon_days: float) -> dict:
    """House-persistence row, used only if the main path fails on a case."""
    spread = SIGMA_RW * math.sqrt(max(horizon_days, 0.0))
    median = logit(last_rate)
    qs: list[float | int] = []
    for z in Z:
        qs.append(round4(clamp(inv_logit(median + z * spread))))
    for k in range(1, 5):
        if qs[k] < qs[k - 1]:
            qs[k] = qs[k - 1]
    return {"id": trophy_id, "q": qs}


def fallback_forecast(case: dict) -> dict:
    """Never leave a case without an output: persistence on the last read."""
    try:
        horizon_days = float(case.get("horizon_minutes", 0)) / 1440.0
        history = case.get("history") or []
        last = history[-1] if history else None
        rows = []
        for i, trophy in enumerate(case["trophies"]):
            rate = None
            if last is not None and i < len(last["rates"]):
                rate = last["rates"][i]
            rows.append(
                _persistence_row(
                    trophy["id"],
                    rate if rate is not None else DEFAULT_RATE,
                    horizon_days,
                )
            )
        return {"trophies": rows}
    except Exception:
        return {"trophies": [_flat_row(t["id"]) for t in case.get("trophies", [])]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        try:
            if case.get("levels") != LEVELS:
                raise ValueError(f"{path.name} asks for levels {case.get('levels')}, not {LEVELS}")
            forecast = forecast_case(case)
        except Exception:
            forecast = fallback_forecast(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
