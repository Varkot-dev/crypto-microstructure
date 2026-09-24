# Phase 4: Temporal Robustness of the Cross-Sectional Laws

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or executing-plans. Checkbox steps.

**Goal:** The project's weakest point is single-regime sampling (June 2023). Phase 4 tests whether the headline cross-sectional findings survive across three regimes — **June 2023** (baseline), **July 2023** (adjacent month; Phase 1.5 showed within-symbol regime drift), and **July 2026** (three years later, post-ETF, matured market):
1. The flip-probability law (p_flip rises with activity; Q4) — same sign and comparable slope in every regime?
2. γ liquidity-invariance (Q4) — still flat?
3. Endogeneity level and its liquidity-invariance (Q6) — does median α drift toward or away from criticality as the market matures?
4. Symbol-level stability — are individual symbols' p_flip / γ / α persistent across months (rank correlation), or is the cross-sectional law stable while symbols reshuffle?
Plus honest survivorship accounting: which June-2023 symbols still exist in July 2026 (delistings are data, not noise).

**Data:** 207-symbol universe aggTrades for 2023-07 and 2026-07 (syncing; 2026 will have missing/delisted symbols — the sync logs record them). Existing analyses are reused unchanged via their CLIs with `--out results/regimes/<period>/` and `--period/--month`.

**Honesty doctrine:** every cross-regime comparison reports n (overlap), and every "law holds" statement carries per-regime slope, R², and the symbol overlap. No pooling across regimes.

---

### Task 1: Regime runs (data-dependent; controller-triggered after sync)
- `uv run python -m microstructure.analyses.q4_cross_section --root data --out results/regimes/2023-07 --symbols-file results/universe_2023-06.txt --period 2023-07 --min-events 1000000`
- same for `--period 2026-07 --out results/regimes/2026-07` (missing symbols land in failures — expected; that list IS the survivorship record).
- Q6 for both months on the same top-40+panel symbol file (`results/q6_symbols_2023-06.txt`): `--out results/regimes/<period> --month <period>` (each ~1h compute; run sequentially in background).
- Commit `feat: Phase-4 regime runs (Q4/Q6 for 2023-07 and 2026-07)`.

### Task 2: Regime comparator (`analyses/q8_regimes.py`) — build on synthetic now, real later
- `run_q8(root, out_dir, baseline_dir: Path, regime_dirs: dict[str, Path]) -> dict`: loads q4_cross_section.json (+ q6_endogeneity.json when present) from baseline and each regime dir.
- Per regime: n_success, flip-law slope/R²/stderr (recomputed from per-symbol records with own OLS, not trusted from the json), γ slope/R², γ median/IQR, p_flip median, count anti-persistent, α median/IQR (if q6 present).
- Cross-regime: symbol overlap sets (baseline∩each regime; survivorship: |2023-06 ∩ 2026-07| and the list of non-survivors); Spearman rank correlation of p_flip, γ (and α) across each regime pair on the overlap; the "law stability" table: same sign of flip slope in all regimes? slope ratio vs baseline; γ invariance (all R² < 0.05?).
- Outputs: results/q8_regimes.{md,json,png} — png: 3 panels (flip law per regime overlaid; γ vs activity per regime overlaid; symbol-level p_flip 2023-06 vs 2026-07 scatter with y=x). md: methodology, regime table, law-stability verdicts phrased from data (either outcome is the finding), survivorship section, caveats (min-events filter shifts membership across regimes; 2026 universe is the 2023 list — new 2026 listings deliberately excluded to keep the panel fixed; state this).
- Synthetic test: three fake regime dirs with planted jsons (constructed records: regime A/B with a planted positive flip slope, regime C with the sign flipped and half the symbols missing) → the comparator reports sign agreement False for C, correct overlap counts, correct survivorship list, rank correlations near planted values.
- Commit `feat: Q8 regime comparator (implementation)`; after Task 1 completes, REAL RUN → `feat: Q8 regime-robustness real results` with the law-stability table in the body.

### Task 3: Synthesis
LEARNING.md Phase-4 section (what temporal robustness means for a cross-sectional law; our verdicts with numbers; survivorship as a confound; what changed 2023→2026 in the market and whether the laws noticed), README gallery + reproduction, drill += 3 ("does your law hold out of sample in time?", "how does survivorship bias your 2026 comparison?", "what changed in three years?"). All numbers artifact-verified.

**Then:** verification workflow (independent recomputation of the law-stability table from the regime jsons + symbol-level spot-checks; whole-branch review; fix wave) → PR → merge.
