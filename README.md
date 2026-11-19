[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jesyeth77/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jesyeth77/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

# GTA VI Trophy Rarity Forecaster

Forecasts, for every Grand Theft Auto VI trophy on PlayStation 5, the
percent of Exophase-tracked players who will have earned it at the case's
target, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

## Method

A trophy's earn rate is a ratio — holders over players tracked — and its
two sides move for different reasons, so they are projected separately
and divided again at the target. Holders grow as incumbent players
progress through the game; players tracked grows as new profiles sync,
and each sync dilutes every rate toward the newcomers' much lower earn
rate. Per trophy, log holders and log players tracked are each fitted by
least squares against log(1 + hours since release) over the last 24 hours
of reads (slopes shrunk toward zero and clipped), then read at the
target; the ratio of the two projections is the median.

On top of that sits a launch-week dilution prior. Exophase's page
refreshes in jumps, not continuously: on launch night it sat unchanged
for hours, then jumped from 638 to 3,868 players tracked and every
non-saturated rate fell to roughly a fifth. So when the page has gone
stale — no rate has changed for a while — the median is shifted down by
an amount that grows with the staleness and with the tracked-player
growth rate, a growth rate whose prior is large on launch night and
decays over the first days. The spread combines the house random walk
(0.5 logit per root-day), the two fits' residuals, and a staleness term
that is large when no refresh has ever been observed and small once
refreshes are visible in the history; the lower quantiles stretch a
little wider than the upper ones, because a refresh can only dilute
rates sharply downward. A trophy already at 100% is left there with a
hairline spread: newcomers earn it too.

## Validation (local, practice cases)

The kit ships no practice answers, so the answers were reconstructed
from the aggregates the competition itself publishes: the refresh took
players tracked 638 → 3,868 and trophies earned 943 → 4,276, so the
sum of rates fell 147.78 → 110.56 while the one trophy at 100% stayed
at 100%, i.e. the other 54 trophies fell to ~22% of their launch-night
rates. Under that reconstruction the house persistence baseline scores
1.40 (published: ~1.51), which brackets the reconstruction as close to
the truth. On the same reconstructed answers this forecaster scores:

| case | this entry | persistence baseline |
|------|-----------|----------------------|
| p00  | 0.076     | 1.400                |
| p01  | 0.108     | 1.400                |
| mean | **0.092** | 1.400                |

Typical miss 0.001–0.03 points, 90% coverage 0.98 (baseline: 0.68
points, coverage 0.00). Under a heterogeneous-dilution variant (per-trophy
dilution lognormal, σ=0.7, same aggregates; baseline then scores ~1.5),
this entry scores ~0.31 — still well below the baseline. On synthetic
final-like cases (2 days of moving history, 4.5-day horizon, refresh
jumps plus progression) it beats persistence on 5 of 6 seeds.

## Run

```sh
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only, no network, deterministic (two runs write
identical bytes), ~0.1 s per practice case, outputs 55 rows of 5
quantiles each, well under the 65,536-byte cap.
