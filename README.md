# Binance Market Microstructure

Empirical measurement of order-flow memory, price impact, and Hawkes self-excitation on Binance
USDT-M perpetual futures tick data — classical microstructure results replicated on crypto and
benchmarked against the published equities literature, then carried to a 121-symbol
cross-section, a 41-symbol endogeneity panel, an execution-cost replay, and a six-regime
robustness check running from 2023-06 to 2026-07.

[![CI](https://github.com/Varkot-dev/crypto-microstructure/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Varkot-dev/crypto-microstructure/actions/workflows/ci.yml)
[![Live results](https://img.shields.io/badge/live%20results-varkot--dev.github.io-blue)](https://varkot-dev.github.io/crypto-microstructure/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Live results site:** <https://varkot-dev.github.io/crypto-microstructure/> — interactive
cross-section and kernel explorers, plus how the numbers were verified.

## In 60 seconds

- **Order flow has long memory, but the activity-dependence of its sign flips fades over three
  years.** The sign-flip-probability law (`p_flip` vs. log-activity) measures **slope +0.1114,
  R² = 0.2632** in the 2023-06 baseline (121 symbols) and passes out of sample one month later
  (2023-07: slope +0.0920, R² = 0.2328). It weakens to about +0.045 in 2024-07 and 2025-07 (each
  roughly 2.3-2.4 standard errors from zero) and is absent in 2026-07: slope +0.0230 (stderr
  0.0325) on the fixed 2023 universe, and **slope +0.0006 (stderr 0.0126), R² = 0.000, n = 231**
  on the 2026 market's own universe. The native-universe run is free of the fixed-2023-panel
  survivorship, so the disappearance is not explained by which 2023 symbols survived (it still
  applies the 1M-event floor, and the universe differs in composition; see the report). Refit
  by cohort, the flip slope on the 39 contracts present in both periods fell from +0.0999
  (t +4.0) to +0.0307 (t +1.1), indistinguishable from zero, and the 192 contracts listed since
  show none (t -0.2).
  → [`results/q4_cross_section.md`](results/q4_cross_section.md),
  [`results/q8_regimes.md`](results/q8_regimes.md)
- **The order-flow-memory exponent γ̂ is liquidity-invariant in 2023; the later break is
  confined to the 2023-listed contracts.** Regressed against log-activity across the 121-symbol
  cross-section: **slope −0.0112, R² = 0.0003** — essentially flat, and still flat in 2023-07
  and 2024-07 (R² 0.0086, 0.0013). On the same fixed universe R² then jumps to 0.2505 in 2025-07
  and 0.2441 in 2026-07, but on the 2026 market's own 231-symbol universe it is small again
  (R² = 0.0196). Splitting that universe by cohort: on the same 39 contracts γ̂ went from no
  activity dependence (2023-06 slope +0.0335, t +1.0) to a clear one (2026-07 slope +0.1896,
  t +3.6, R² = 0.256), while the 192 contracts listed since show little (slope +0.0422, t +1.3,
  R² = 0.009). The break is a within-cohort change among 2023-listed contracts, diluted
  market-wide by newer listings; it is not a survivorship artifact.
  → [`results/q4_cross_section.md`](results/q4_cross_section.md),
  [`results/q8_regimes.md`](results/q8_regimes.md)
- **~70% of trades are reactions to other trades in 2023, drifting down moderately after.** The
  Hawkes branching ratio across a 41-symbol panel has **median α̂ = 0.7070** (range
  0.3699–0.8790) under an exponential kernel — a documented lower bound. Raw medians in later
  months are not comparable at face value: the share of fits that lock onto the fast component of
  a multi-timescale kernel (and so understate α̂ by construction) rises from 0.12 to 0.31–0.49.
  On kernel-mode-robust measures (slow-mode α̂, kernel-free count-variance n̂, and symbols paired
  across months) the drift is downward by roughly 0.02–0.17 in α̂ and 0.01–0.06 in n̂, and not
  monotonic; the two largest paired α̂ shifts (2025-07, 2026-07) have bootstrap intervals that
  include zero, so the count-variance decline is the firmer evidence of direction. α̂ shows no activity dependence beyond 2 standard errors in any regime.
  → [`results/q6_endogeneity.md`](results/q6_endogeneity.md),
  [`results/q8_regimes.md`](results/q8_regimes.md)
- **Front-loading execution trades a deterministic cost for a stochastic one.** Against replayed
  flow on 6 panel symbols, front-loaded schedules pay a small positive mean shortfall
  (+0.1306) with standard deviation **0.3404**, versus ≈5.2 for TWAP and the reactive schedule —
  roughly a **15× variance reduction**, holding for every symbol in the panel.
  → [`results/q7_execution.md`](results/q7_execution.md)
- **The response function rises 5.39× before plateauing, exactly where flow-memory theory
  predicts.** `R(1) = 0.0104` rises to a plateau near 0.056 (ℓ ≈ 300–500); the independently
  predicted band from Q1's exponent is 3.5–6.9×, and the measured 5.39× falls inside it.
  → [`results/q2_results.md`](results/q2_results.md)

## What's here

| Question | Finding | Details |
|---|---|---|
| Q0 | Raw-print γ̂ is inflated +0.29 to +0.50 vs. correctly aggregated γ̂ | [md](results/q0_aggregation_effect.md) |
| Q1 | Order flow has long memory (BTC γ̂ = 0.3803, ETH γ̂ = 0.2380) | [md](results/q1_results.md) · [png](results/q1_acf_loglog.png) |
| Q1b | The short-lag zigzag in Q1's ACF is real structure, not a tie-break artifact | [md](results/q1b_zigzag.md) |
| Q2 | Response function rises 5.39×, matching flow-memory theory's predicted band | [md](results/q2_results.md) · [png](results/q2_response.png) |
| Q3 | Price change is linear in OFI (R² = 0.40), below equities' 65–70% | [md](results/q3_results.md) · [png](results/q3_ofi_scatter.png) |
| Q4 | In 2023-06, γ̂ is liquidity-invariant (R² = 0.0003); p_flip is not (R² = 0.2632) | [md](results/q4_cross_section.md) · [png](results/q4_gamma_vs_activity.png) |
| Q4b | Activity dominates the p_flip law; tick size is a real but minor contributor | [md](results/q4b_tick_confound.md) |
| Q5 | Critical balance β = (1−γ)/2 holds for 12 of 16 kernel-panel symbols | [md](results/q5_kernel_panel.md) · [png](results/q5_kernel_panel.png) |
| Q6 | Median branching ratio 0.707 — ~70% of trades are endogenous reactions | [md](results/q6_endogeneity.md) · [png](results/q6_endogeneity.png) |
| Q7 | Front-loaded execution: 15× variance reduction vs. TWAP/reactive | [md](results/q7_execution.md) · [png](results/q7_execution.png) |
| Q8 | Flip law fades to zero by 2026 on both fixed and native universes; γ̂ break confined to the 2023-listed contracts, diluted by newer listings; endogeneity drifts down modestly | [md](results/q8_regimes.md) · [png](results/q8_regimes.png) |

```
docs/        report.md (full write-up), research/, plans/, specs/ — see docs/README.md
src/         pipeline (data/) → signals/ → estimators/ → analyses/ (Q0–Q8) → execution/
results/     figures + per-question .md write-ups + .json/.parquet artifacts (committed)
site/        static results site; build_data.py derives site/data/*.json from results/
LEARNING.md  concepts, judgment calls, and a 21-question interview drill
```

## Quickstart

```bash
git clone git@github.com:Varkot-dev/crypto-microstructure.git
cd crypto-microstructure
uv sync

# Verify the code before trusting it: 171 tests against synthetic ground truth
uv run pytest -m "not network" -q
uv run ruff check src/ tests/

# Sync one month of one symbol (checksum-verified download + Parquet ingest)
uv run python -c "
from pathlib import Path
from microstructure.data.catalog import sync
sync(Path('data'), 'BTCUSDT', 'aggTrades', '2023-06', '2023-06')
"

# Run one analysis (writes results/q1_*.png/.md/.json)
uv run python -m microstructure.analyses.q1_orderflow_memory \
    --root data --out results --symbols BTCUSDT --periods 2023-06 --max-lag 1000
```

Full reproduction steps for every question (Q1 through the Phase-4 regime comparator) are in
[`docs/report.md`](docs/report.md#reproducing-from-a-fresh-clone).

## How it was built and verified

- **Every estimator is validated against synthetic series with analytically known answers**
  before touching real data — i.i.d. signs (ACF exactly 0), a Markov chain (ACF = (2p−1)^k),
  FARIMA noise (γ = 1 − 2d), a known impact kernel the response estimator must recover, and a
  simulated Hawkes process with a planted branching ratio the MLE must recover.
- **Two tests exist specifically to catch spurious near-criticality**: a regime-switching
  Poisson process with no self-excitation at all, which both branching-ratio estimators
  misreport as near-critical (n̂ = 0.934, α̂ = 0.976) unless business-time rescaling is applied.
- **Headline numbers are reproduced independently where possible**: Q2's response ratio was
  predicted from Q1's exponent on a different dataset before being measured directly, and Q8
  recomputes every regime regression from source records with its own OLS rather than trusting
  upstream JSON.
- **Self-corrections are documented, not hidden.** See
  [LEARNING.md §3](LEARNING.md#3-price-impact-and-the-response-function) (an early misreading of
  the response function) and [LEARNING.md §7.4](LEARNING.md#74-the-thinning-bias-episode-what-a-rejected-fix-looks-like)
  (a rejected fix and why it was rejected).
- CI (`.github/workflows/ci.yml`) runs `ruff check` and the non-network test suite on every push;
  the Pages workflow (`.github/workflows/pages.yml`) regenerates `site/data/*.json` from
  `results/` and fails the build if the committed slices are stale.
- Test coverage: 89% of src/, measured 2026-09-26 with pytest-cov.

## Further reading

- **[LEARNING.md](LEARNING.md)** — the concepts and the interview-prep version of this project:
  the order book, autocorrelation, impact, OFI, Hawkes self-excitation, temporal robustness, the
  statistics behind every number above, and a 21-question drill.
- **[docs/report.md](docs/report.md)** — the full research report: every question in detail,
  methodology, numbers, and caveats, plus the complete reproduction guide and known limitations.
- **[docs/README.md](docs/README.md)** — index of `docs/research/`, `docs/plans/`, and
  `docs/specs/`.
