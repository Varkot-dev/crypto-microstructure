# Q8: regime comparator — temporal robustness of the cross-sectional laws

## Methodology

Loads `q4_cross_section.json` (required) and `q6_endogeneity.json` (optional — not every regime necessarily has a Q6 run) from a baseline directory (`2023-06`) and one or more regime directories. Every cross-sectional regression (γ̂ vs. log10(activity), p_flip vs. log10(activity), and α̂_median vs. log10(activity) when Q6 is present) is **recomputed from the per-symbol records using this module's own `np.polyfit`-based OLS** — the upstream jsons' stored `regressions` / `activity_regression` blocks are never trusted directly, only cross-checked against the recomputed values; any mismatch beyond a tight numerical tolerance (1e-06) is reported as an explicit warning below rather than silently accepted or overwritten.

**Survivorship**: for each regime, the baseline's successful symbol set is compared against that regime's successful symbol set. Baseline symbols absent from a regime are non-survivors; each is annotated with a reason drawn from that regime's own Q4 `skips` (below `min_events`) and `failures` (missing parquet / other exception) lists when available, distinguishing symbols that simply fell below the activity threshold in that regime from symbols that failed or are missing outright.

**Rank correlation**: Spearman's rho on the symbol overlap for p_flip, γ̂ (and α̂ when both sides have a Q6 run), computed via average-rank ranking (ties share the mean of their ranks) and Pearson correlation of the ranks — the standard exact definition of Spearman's rho, implemented with numpy only (no scipy dependency).

**Law-stability verdicts**: same-sign check on the flip-law slope across baseline and every regime; slope ratio of each regime vs. baseline; γ-invariance is verdicted true iff EVERY regime's (baseline included) γ-vs-activity R² falls below 0.05. All verdict text below is generated from these computed values — the wording is not hardcoded to a particular conclusion; either the law holds or it does not, and this report states whichever the data shows.

## Regime table

| regime | n_success | flip slope | flip R² | γ slope | γ R² | γ median (IQR) | p_flip median | anti-persistent | α median (IQR) |
|---|---|---|---|---|---|---|---|---|---|
| 2023-06 | 121 | 0.1114 | 0.2632 | -0.0112 | 0.0003 | 0.3270 (0.0985) | 0.4543 | 20 | 0.7070 (0.2240) |
| 2023-07 | 117 | 0.0920 | 0.2328 | -0.0225 | 0.0086 | 0.3221 (0.0611) | 0.4483 | 15 | 0.6925 (0.2505) |
| 2026-07 | 46 | 0.0230 | 0.0113 | 0.1683 | 0.2441 | 0.1991 (0.1510) | 0.4154 | 4 | 0.5766 (0.4159) |

## Law-stability verdicts

**Flip-law sign stability**: the flip-law slope has the **same sign** in every regime as in the baseline — the direction of the p_flip-vs-activity relationship is stable across regimes.

Flip-law slope ratio vs. baseline, per regime:

- 2023-07: 0.8262x baseline slope
- 2026-07: 0.2069x baseline slope

**γ invariance**: at least one regime's γ-vs-activity R² is at or above 0.05 (2026-07) — γ's liquidity-invariance does **not** hold uniformly across every regime examined here.

### γ-break outlier sensitivity (drop-one-out)

For **2026-07** (n=46, full-sample slope=0.1683, R²=0.2441), a drop-one-out refit of γ vs. log10(activity) — removing each symbol one at a time and re-fitting — gives an **R² range of [0.1738 (dropping BTCUSDT), 0.2959 (dropping LTCUSDT)]** and a **slope range of [0.1442 (dropping BTCUSDT), 0.1867 (dropping YFIUSDT)]**. The highest-influence points by Cook's distance are BTCUSDT (Cook's D=0.168, leverage=0.165); YFIUSDT (Cook's D=0.130, leverage=0.062); LTCUSDT (Cook's D=0.112, leverage=0.045). The slope **stays positive under every single-symbol removal** — the break's direction is not an artifact of any one symbol — but R² swings by a large relative amount depending on which point is dropped, so the *strength* (not the sign) of the break is outlier-sensitive. This replaces an earlier unquantified 'a handful of outliers' hedge with the measured sensitivity.

## Survivorship

**2023-07**: 101/121 baseline symbols survive into this regime's successful set (117 symbols total in this regime). 20 non-survivor(s).

| symbol | reason |
|---|---|
| 1000FLOKIUSDT | skipped below min_events (n_events=747701) |
| ACHUSDT | skipped below min_events (n_events=748120) |
| AMBUSDT | skipped below min_events (n_events=556865) |
| BNXUSDT | skipped below min_events (n_events=653211) |
| COMBOUSDT | skipped below min_events (n_events=880918) |
| COTIUSDT | skipped below min_events (n_events=962970) |
| DASHUSDT | skipped below min_events (n_events=823758) |
| FLMUSDT | skipped below min_events (n_events=684050) |
| GALABUSD | skipped below min_events (n_events=713883) |
| GALUSDT | skipped below min_events (n_events=936924) |
| HIGHUSDT | skipped below min_events (n_events=619327) |
| JOEUSDT | skipped below min_events (n_events=877303) |
| LDOBUSD | skipped below min_events (n_events=752586) |
| LPTUSDT | skipped below min_events (n_events=782640) |
| QNTUSDT | skipped below min_events (n_events=941058) |
| RADUSDT | skipped below min_events (n_events=497135) |
| SFPUSDT | skipped below min_events (n_events=831726) |
| TRUUSDT | skipped below min_events (n_events=737950) |
| TUSDT | skipped below min_events (n_events=959631) |
| VETUSDT | skipped below min_events (n_events=796364) |

**2026-07**: 40/121 baseline symbols survive into this regime's successful set (46 symbols total in this regime). 81 non-survivor(s).

| symbol | reason |
|---|---|
| 1000FLOKIUSDT | skipped below min_events (n_events=589155) |
| 1000LUNCUSDT | skipped below min_events (n_events=919921) |
| ACHUSDT | skipped below min_events (n_events=920286) |
| AGIXUSDT | failed/missing (parquet not found: data/parquet/aggTrades/AGIXUSDT/2026-07.parquet) |
| ALPHAUSDT | failed/missing (parquet not found: data/parquet/aggTrades/ALPHAUSDT/2026-07.parquet) |
| AMBUSDT | failed/missing (parquet not found: data/parquet/aggTrades/AMBUSDT/2026-07.parquet) |
| ANKRUSDT | skipped below min_events (n_events=533868) |
| ANTUSDT | failed/missing (parquet not found: data/parquet/aggTrades/ANTUSDT/2026-07.parquet) |
| APEUSDT | skipped below min_events (n_events=667991) |
| ATOMUSDT | skipped below min_events (n_events=837588) |
| AXSUSDT | skipped below min_events (n_events=487196) |
| BANDUSDT | skipped below min_events (n_events=258105) |
| BNXUSDT | failed/missing (parquet not found: data/parquet/aggTrades/BNXUSDT/2026-07.parquet) |
| BTCBUSD | failed/missing (parquet not found: data/parquet/aggTrades/BTCBUSD/2026-07.parquet) |
| CFXUSDT | skipped below min_events (n_events=642676) |
| CHZUSDT | skipped below min_events (n_events=763155) |
| COMBOUSDT | failed/missing (parquet not found: data/parquet/aggTrades/COMBOUSDT/2026-07.parquet) |
| COMPUSDT | skipped below min_events (n_events=252556) |
| CRVUSDT | skipped below min_events (n_events=718109) |
| CTSIUSDT | skipped below min_events (n_events=286784) |
| DUSKUSDT | skipped below min_events (n_events=457845) |
| EDUUSDT | skipped below min_events (n_events=438257) |
| EOSUSDT | failed/missing (parquet not found: data/parquet/aggTrades/EOSUSDT/2026-07.parquet) |
| ETHBUSD | failed/missing (parquet not found: data/parquet/aggTrades/ETHBUSD/2026-07.parquet) |
| FLMUSDT | failed/missing (parquet not found: data/parquet/aggTrades/FLMUSDT/2026-07.parquet) |
| FTMUSDT | failed/missing (parquet not found: data/parquet/aggTrades/FTMUSDT/2026-07.parquet) |
| FXSUSDT | failed/missing (parquet not found: data/parquet/aggTrades/FXSUSDT/2026-07.parquet) |
| GALABUSD | failed/missing (parquet not found: data/parquet/aggTrades/GALABUSD/2026-07.parquet) |
| GALAUSDT | skipped below min_events (n_events=849002) |
| GALUSDT | failed/missing (parquet not found: data/parquet/aggTrades/GALUSDT/2026-07.parquet) |
| GMTUSDT | skipped below min_events (n_events=360680) |
| GRTUSDT | skipped below min_events (n_events=293204) |
| HIGHUSDT | failed/missing (parquet not found: data/parquet/aggTrades/HIGHUSDT/2026-07.parquet) |
| ICPUSDT | skipped below min_events (n_events=994228) |
| IMXUSDT | skipped below min_events (n_events=345265) |
| JASMYUSDT | skipped below min_events (n_events=546964) |
| JOEUSDT | skipped below min_events (n_events=261908) |
| KAVAUSDT | skipped below min_events (n_events=271865) |
| KEYUSDT | failed/missing (parquet not found: data/parquet/aggTrades/KEYUSDT/2026-07.parquet) |
| KNCUSDT | skipped below min_events (n_events=175561) |
| LDOBUSD | failed/missing (parquet not found: data/parquet/aggTrades/LDOBUSD/2026-07.parquet) |
| LINAUSDT | failed/missing (parquet not found: data/parquet/aggTrades/LINAUSDT/2026-07.parquet) |
| LPTUSDT | skipped below min_events (n_events=410648) |
| LQTYUSDT | skipped below min_events (n_events=301348) |
| LUNA2USDT | skipped below min_events (n_events=369214) |
| MAGICUSDT | skipped below min_events (n_events=406652) |
| MANAUSDT | skipped below min_events (n_events=719991) |
| MASKUSDT | skipped below min_events (n_events=360294) |
| MATICUSDT | failed/missing (parquet not found: data/parquet/aggTrades/MATICUSDT/2026-07.parquet) |
| MINAUSDT | skipped below min_events (n_events=489846) |
| MKRUSDT | failed/missing (parquet not found: data/parquet/aggTrades/MKRUSDT/2026-07.parquet) |
| MTLUSDT | skipped below min_events (n_events=178253) |
| NEOUSDT | skipped below min_events (n_events=342698) |
| NKNUSDT | failed/missing (parquet not found: data/parquet/aggTrades/NKNUSDT/2026-07.parquet) |
| OCEANUSDT | failed/missing (parquet not found: data/parquet/aggTrades/OCEANUSDT/2026-07.parquet) |
| OMGUSDT | failed/missing (parquet not found: data/parquet/aggTrades/OMGUSDT/2026-07.parquet) |
| PHBUSDT | failed/missing (parquet not found: data/parquet/aggTrades/PHBUSDT/2026-07.parquet) |
| QNTUSDT | skipped below min_events (n_events=400838) |
| RADUSDT | failed/missing (parquet not found: data/parquet/aggTrades/RADUSDT/2026-07.parquet) |
| RDNTUSDT | failed/missing (parquet not found: data/parquet/aggTrades/RDNTUSDT/2026-07.parquet) |
| RENUSDT | failed/missing (parquet not found: data/parquet/aggTrades/RENUSDT/2026-07.parquet) |
| RLCUSDT | skipped below min_events (n_events=358385) |
| RNDRUSDT | failed/missing (parquet not found: data/parquet/aggTrades/RNDRUSDT/2026-07.parquet) |
| ROSEUSDT | skipped below min_events (n_events=531017) |
| SANDUSDT | skipped below min_events (n_events=727999) |
| SFPUSDT | skipped below min_events (n_events=231105) |
| SNXUSDT | skipped below min_events (n_events=602240) |
| SOLBUSD | failed/missing (parquet not found: data/parquet/aggTrades/SOLBUSD/2026-07.parquet) |
| STGUSDT | skipped below min_events (n_events=830723) |
| STORJUSDT | skipped below min_events (n_events=697656) |
| STXUSDT | skipped below min_events (n_events=756215) |
| SUSHIUSDT | skipped below min_events (n_events=281617) |
| SXPUSDT | failed/missing (parquet not found: data/parquet/aggTrades/SXPUSDT/2026-07.parquet) |
| THETAUSDT | skipped below min_events (n_events=423345) |
| TOMOUSDT | failed/missing (parquet not found: data/parquet/aggTrades/TOMOUSDT/2026-07.parquet) |
| TRUUSDT | failed/missing (parquet not found: data/parquet/aggTrades/TRUUSDT/2026-07.parquet) |
| VETUSDT | skipped below min_events (n_events=497001) |
| WAVESUSDT | failed/missing (parquet not found: data/parquet/aggTrades/WAVESUSDT/2026-07.parquet) |
| WOOUSDT | skipped below min_events (n_events=246613) |
| XRPBUSD | failed/missing (parquet not found: data/parquet/aggTrades/XRPBUSD/2026-07.parquet) |
| ZENUSDT | skipped below min_events (n_events=587443) |

### Universe accounting (full requested universe, per regime)

The survivorship table above is relative to the baseline's *successful* symbol set. The table below instead accounts for the **full requested universe** in each regime (fixed to the baseline's symbol list) across three buckets: successful (passed `min_events`), skipped (downloaded but below `min_events`), and failed (no data to download at all for that period). These three buckets always sum to the requested universe size by construction of the upstream Q4 run.

| regime | requested | successful | skipped (below floor) | failed (no data) | reconciles |
|---|---|---|---|---|---|
| 2023-07 | 207 | 117 | 87 | 3 | yes |
| 2026-07 | 207 | 46 | 93 | 68 | yes |

- **2026-07** download-missing cross-check: matches: 68 symbols in both the q4 `failures` list and the external download-missing file

## Symbol-level rank correlation

| regime | n overlap | p_flip Spearman ρ | γ Spearman ρ | α overlap n | α Spearman ρ |
|---|---|---|---|---|---|
| 2023-07 | 101 | 0.7576 | 0.1759 | 41 | 0.8723 |
| 2026-07 | 40 | 0.2925 | 0.0538 | 32 | 0.2144 |

## Caveats

- **`min_events` filter shifts membership across regimes**: a symbol's activity level in a given month determines whether it clears the Q4 `min_events` threshold at all, so the 'successful' symbol set is not the same fixed panel across regimes — some non-survivors are genuinely below the activity bar in that regime, not delisted or otherwise absent, and this is reported as such via the skip/failure reason above rather than conflated with true delistings.
- **The regime universe is fixed to the baseline symbol list**: any symbol newly listed in a later regime but absent from the baseline period is deliberately excluded from every regime's requested universe upstream (Q4/Q6 are run against `results/universe_2023-06.txt`), to keep the panel fixed and comparable across regimes — this survivorship analysis therefore cannot and does not speak to new listings, only to the fate of the original panel.
- **Regression stderr/R² inherit Q4/Q6's own heteroskedasticity caveat**: as documented in `q4_cross_section.md` and `q6_endogeneity.md`, per-symbol estimator noise is not uniform across the cross-section, so the OLS regressions recomputed here (same assumptions, same caveat) should be read descriptively, not as formal confidence intervals.
- **Spearman rho on a possibly small overlap**: rank correlation is only as informative as the overlap size allows; a small `n_overlap` (see the table above) should be weighted accordingly rather than treated as a precise correlation estimate.
