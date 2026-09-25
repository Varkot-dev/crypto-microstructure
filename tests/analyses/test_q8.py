"""Tests for Q8: regime comparator (Phase 4, Task 2).

Contract under test (see docs/superpowers/plans/2026-09-24-phase4-regimes.md,
Task 2, and its binding elaboration):

1. Three fake regime dirs with planted `q4_cross_section.json` (+ q6 for
   two of them): baseline "2023-06", regime A/B with a planted positive
   flip-law slope (same sign as baseline), regime C with the flip-law
   sign flipped and half the baseline symbols missing (to exercise
   survivorship). `run_q8` must:
   - report `flip_law_same_sign_all_regimes = False` (because of C)
   - report correct symbol-overlap counts
   - report the correct non-survivor list for C
   - recover Spearman rank correlations near the planted values
   - produce q8_regimes.{md,json,png}
2. A regime cross-check mismatch (stored regression doesn't match what
   `run_q8` recomputes from records) must surface as a warning string in
   the output json, not be silently ignored or crash the run.
3. A regime dir with NO q6_endogeneity.json must be handled gracefully
   (q6-derived fields are None, no exception).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from microstructure.analyses.q8_regimes import (
    _average_ranks,
    _reg_mismatch,
    run_q8,
    spearman_corr,
)

N_EVENTS_BASE = 1_000_000


def _q4_record(symbol: str, n_events: int, gamma: float, p_flip: float) -> dict:
    return {
        "symbol": symbol,
        "n_events": n_events,
        "gamma": gamma,
        "stderr": 0.01,
        "acf1": 2 * (1.0 - p_flip) - 1.0,
        "p_flip": p_flip,
        "zigzag_amplitude": 0.05,
        "total_qty": float(n_events) * 0.5,
    }


def _ols(x: np.ndarray, y: np.ndarray) -> dict:
    (slope, intercept), cov = np.polyfit(x, y, 1, cov=True)
    yhat = slope * x + intercept
    resid = y - yhat
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return {
        "slope": float(slope), "intercept": float(intercept),
        "stderr": float(np.sqrt(cov[0, 0])), "r2": r2, "n": int(x.size),
    }


def _write_q4_json(
    out_dir: Path, period: str, records: list[dict], skips: list[dict] | None = None,
    failures: list[dict] | None = None, corrupt_regressions: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_n = np.array([np.log10(r["n_events"]) for r in records])
    gamma = np.array([r["gamma"] for r in records])
    p_flip = np.array([r["p_flip"] for r in records])
    regressions = {
        "gamma_vs_activity": _ols(log_n, gamma) if len(records) >= 3 else None,
        "p_flip_vs_activity": _ols(log_n, p_flip) if len(records) >= 3 else None,
    }
    if corrupt_regressions and regressions["p_flip_vs_activity"] is not None:
        regressions["p_flip_vs_activity"]["slope"] += 5.0  # deliberately wrong

    payload = {
        "period": period,
        "min_events": 1_000_000,
        "max_lag": 1000,
        "n_symbols_requested": len(records) + len(skips or []) + len(failures or []),
        "n_symbols_successful": len(records),
        "n_symbols_skipped": len(skips or []),
        "n_symbols_failed": len(failures or []),
        "symbols": records,
        "skips": skips or [],
        "failures": failures or [],
        "regressions": regressions,
    }
    (out_dir / "q4_cross_section.json").write_text(json.dumps(payload, indent=2))


def _write_q6_json(out_dir: Path, month: str, alpha_by_symbol: dict[str, float], n_events_by_symbol: dict[str, int]) -> None:
    records = [
        {
            "symbol": sym,
            "n_events": n_events_by_symbol[sym],
            "alpha_median": alpha,
            "alpha_iqr": 0.05,
            "alphas": [alpha] * 6,
            "n_converged": 6,
            "alpha_cv": alpha + 0.01,
            "raw_delta": 0.001,
            "raw_alpha_window1": alpha,
            "median_beta": 2.0,
            "median_mu": 1.0,
            "count_variance_window_bt": 200.0,
        }
        for sym, alpha in alpha_by_symbol.items()
    ]
    log_n = np.array([np.log10(r["n_events"]) for r in records])
    alpha_med = np.array([r["alpha_median"] for r in records])
    activity_regression = _ols(log_n, alpha_med) if len(records) >= 3 else None
    payload = {
        "month": month,
        "windows": 6,
        "top_n": 40,
        "n_symbols_requested": len(records),
        "n_symbols_successful": len(records),
        "n_symbols_failed": 0,
        "records": records,
        "failures": [],
        "activity_regression": activity_regression,
        "agreement": {"median_abs_diff": 0.01, "correlation": 0.9, "n": len(records)},
    }
    (out_dir / "q6_endogeneity.json").write_text(json.dumps(payload, indent=2))


# ---------------------------------------------------------------------------
# Planted fixture: 10 baseline symbols with n_events spanning several
# decades of log10 so log-activity has real variance, and p_flip/gamma
# constructed as an EXACT linear function of log10(n_events) so the
# planted OLS slope is recoverable near-exactly and, crucially, the
# per-symbol RANKS of p_flip/gamma are perfectly monotonic in n_events —
# this lets regime A/B (same symbols, same monotonic order, different
# noise/intercept) have a Spearman rho near +1.0 against baseline, and
# regime C's SIGN-FLIPPED p_flip law invert that rank order, giving a
# Spearman rho near -1.0 against baseline on the overlap.
# ---------------------------------------------------------------------------

SYMBOLS = [f"SYM{i}USDT" for i in range(10)]
N_EVENTS = {sym: N_EVENTS_BASE * (i + 1) for i, sym in enumerate(SYMBOLS)}
LOG_N = {sym: np.log10(N_EVENTS[sym]) for sym in SYMBOLS}

# Baseline: p_flip rises with log-activity (positive planted slope), gamma flat.
FLIP_SLOPE_BASE = 0.05
FLIP_INTERCEPT_BASE = 0.3
GAMMA_BASE = 0.4  # constant -> R^2 ~ 0 (gamma-invariant)


def _baseline_records() -> list[dict]:
    return [
        _q4_record(
            sym, N_EVENTS[sym], gamma=GAMMA_BASE,
            p_flip=FLIP_INTERCEPT_BASE + FLIP_SLOPE_BASE * LOG_N[sym],
        )
        for sym in SYMBOLS
    ]


def _regime_records(flip_slope: float, flip_intercept: float, symbols: list[str]) -> list[dict]:
    return [
        _q4_record(
            sym, N_EVENTS[sym], gamma=GAMMA_BASE,
            p_flip=flip_intercept + flip_slope * LOG_N[sym],
        )
        for sym in symbols
    ]


@pytest.fixture
def three_regime_dirs(tmp_path: Path) -> dict:
    baseline_dir = tmp_path / "baseline"
    a_dir = tmp_path / "regime_a"
    b_dir = tmp_path / "regime_b"
    c_dir = tmp_path / "regime_c"

    baseline_records = _baseline_records()
    _write_q4_json(baseline_dir, "2023-06", baseline_records)
    _write_q6_json(
        baseline_dir, "2023-06",
        alpha_by_symbol={sym: 0.5 + 0.02 * LOG_N[sym] for sym in SYMBOLS},
        n_events_by_symbol=N_EVENTS,
    )

    # Regime A/B: same sign positive slope as baseline, full symbol set.
    a_records = _regime_records(flip_slope=0.06, flip_intercept=0.28, symbols=SYMBOLS)
    _write_q4_json(a_dir, "2023-07", a_records)
    _write_q6_json(
        a_dir, "2023-07",
        alpha_by_symbol={sym: 0.55 + 0.02 * LOG_N[sym] for sym in SYMBOLS},
        n_events_by_symbol=N_EVENTS,
    )

    b_records = _regime_records(flip_slope=0.04, flip_intercept=0.32, symbols=SYMBOLS)
    _write_q4_json(b_dir, "2024-07", b_records)
    # Regime B deliberately has NO q6 -> must be handled gracefully.

    # Regime C: sign-flipped slope (negative), half the symbols missing.
    # 5 survive, 5 do not: 2 explicitly skipped (below min_events), 3
    # explicitly failed/missing, to exercise both non-survivor reasons.
    c_survivor_symbols = SYMBOLS[:5]
    c_missing_symbols = SYMBOLS[5:]
    c_records = _regime_records(flip_slope=-0.07, flip_intercept=0.9, symbols=c_survivor_symbols)
    c_skips = [
        {"symbol": sym, "reason": "below min_events", "n_events": 500}
        for sym in c_missing_symbols[:2]
    ]
    c_failures = [
        {"symbol": sym, "reason": "FileNotFoundError: parquet not found"}
        for sym in c_missing_symbols[2:]
    ]
    _write_q4_json(c_dir, "2026-07", c_records, skips=c_skips, failures=c_failures)

    return {
        "baseline_dir": baseline_dir,
        "regime_dirs": {"2023-07": a_dir, "2024-07": b_dir, "2026-07": c_dir},
        "baseline_records": baseline_records,
        "a_records": a_records,
        "b_records": b_records,
        "c_records": c_records,
        "c_survivor_symbols": c_survivor_symbols,
        "c_missing_symbols": c_missing_symbols,
        "c_skips": c_skips,
        "c_failures": c_failures,
    }


# ---------------------------------------------------------------------------
# Spearman correlation unit tests (own numpy implementation, no scipy)
# ---------------------------------------------------------------------------


def test_average_ranks_handles_ties_with_mean_rank():
    ranks = _average_ranks(np.array([10.0, 20.0, 20.0, 30.0]))
    np.testing.assert_allclose(ranks, [1.0, 2.5, 2.5, 4.0])


def test_spearman_corr_perfect_monotonic_is_one():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    assert spearman_corr(x, y) == pytest.approx(1.0)


def test_spearman_corr_perfect_inverse_is_minus_one():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([50.0, 40.0, 30.0, 20.0, 10.0])
    assert spearman_corr(x, y) == pytest.approx(-1.0)


def test_spearman_corr_constant_array_is_none():
    x = np.array([1.0, 1.0, 1.0])
    y = np.array([1.0, 2.0, 3.0])
    assert spearman_corr(x, y) is None


def test_spearman_corr_too_few_points_is_none():
    assert spearman_corr(np.array([1.0]), np.array([2.0])) is None


# ---------------------------------------------------------------------------
# run_q8 integration tests on planted regime dirs
# ---------------------------------------------------------------------------


def test_run_q8_sign_agreement_and_law_stability(three_regime_dirs: dict, tmp_path: Path):
    out_dir = tmp_path / "results"
    result = run_q8(
        out_dir,
        baseline_dir=three_regime_dirs["baseline_dir"],
        regime_dirs=three_regime_dirs["regime_dirs"],
        baseline_label="2023-06",
    )

    stability = result["law_stability"]
    # Regime C has a sign-flipped flip-law slope -> overall same-sign must be False.
    assert stability["flip_law_same_sign_all_regimes"] is False

    # A and B individually agree in sign with baseline (positive).
    slopes = stability["flip_law_slope_by_label"]
    assert slopes["2023-07"] > 0
    assert slopes["2024-07"] > 0
    assert slopes["2026-07"] < 0
    assert slopes["__baseline__"] > 0

    # Gamma is constant across every regime by construction -> R^2 ~ 0 everywhere.
    assert stability["gamma_invariant_all_regimes"] is True
    for r2 in stability["gamma_r2_by_label"].values():
        assert r2 < 0.05


def test_run_q8_overlap_and_survivorship_counts(three_regime_dirs: dict, tmp_path: Path):
    out_dir = tmp_path / "results"
    result = run_q8(
        out_dir,
        baseline_dir=three_regime_dirs["baseline_dir"],
        regime_dirs=three_regime_dirs["regime_dirs"],
        baseline_label="2023-06",
    )

    surv_a = result["survivorship"]["2023-07"]
    assert surv_a["n_baseline"] == 10
    assert surv_a["n_regime"] == 10
    assert surv_a["n_survivors"] == 10
    assert surv_a["non_survivors"] == []

    surv_c = result["survivorship"]["2026-07"]
    assert surv_c["n_baseline"] == 10
    assert surv_c["n_regime"] == 5
    assert surv_c["n_survivors"] == 5
    assert sorted(surv_c["survivors"]) == sorted(three_regime_dirs["c_survivor_symbols"])

    non_survivor_symbols = {ns["symbol"] for ns in surv_c["non_survivors"]}
    assert non_survivor_symbols == set(three_regime_dirs["c_missing_symbols"])

    # Reasons distinguish skipped-below-min-events from failed/missing.
    reasons_by_symbol = {ns["symbol"]: ns["reason"] for ns in surv_c["non_survivors"]}
    for sym in [s["symbol"] for s in three_regime_dirs["c_skips"]]:
        assert "skipped below min_events" in reasons_by_symbol[sym]
    for sym in [f["symbol"] for f in three_regime_dirs["c_failures"]]:
        assert "failed/missing" in reasons_by_symbol[sym]

    rank_corr_a = result["rank_correlations"]["2023-07"]
    assert rank_corr_a["n_overlap"] == 10

    rank_corr_c = result["rank_correlations"]["2026-07"]
    assert rank_corr_c["n_overlap"] == 5


def test_run_q8_spearman_near_planted_values(three_regime_dirs: dict, tmp_path: Path):
    out_dir = tmp_path / "results"
    result = run_q8(
        out_dir,
        baseline_dir=three_regime_dirs["baseline_dir"],
        regime_dirs=three_regime_dirs["regime_dirs"],
        baseline_label="2023-06",
    )

    # Regime A: p_flip is a positive linear function of log_n on the SAME
    # symbol set as baseline (also positive-linear in log_n) -> both series
    # are monotonically increasing in the same symbol order -> rank
    # correlation must be exactly 1.0 (no ties, perfectly monotonic).
    rc_a = result["rank_correlations"]["2023-07"]
    assert rc_a["p_flip_spearman"] == pytest.approx(1.0)
    # Gamma is constant in both baseline and regime A -> undefined (None).
    assert rc_a["gamma_spearman"] is None
    # Q6 present on both sides -> alpha overlap should be full and rho = 1.0
    # (alpha_median is also monotonically increasing in log_n on both sides).
    assert rc_a["alpha_n_overlap"] == 10
    assert rc_a["alpha_spearman"] == pytest.approx(1.0)

    # Regime B has no q6 -> alpha fields must be gracefully None/0, not raise.
    rc_b = result["rank_correlations"]["2024-07"]
    assert rc_b["alpha_n_overlap"] == 0
    assert rc_b["alpha_spearman"] is None

    # Regime C: p_flip is a NEGATIVE linear function of log_n on the
    # overlap (5 symbols) while baseline's p_flip is positive-linear in
    # log_n on those same 5 symbols -> perfectly anti-monotonic -> rho = -1.0.
    rc_c = result["rank_correlations"]["2026-07"]
    assert rc_c["p_flip_spearman"] == pytest.approx(-1.0)


def test_run_q8_recomputes_not_trusts_stored_regressions(tmp_path: Path):
    """A stored regression block that doesn't match the records must surface a warning."""
    baseline_dir = tmp_path / "baseline"
    regime_dir = tmp_path / "regime"
    baseline_records = _baseline_records()
    _write_q4_json(baseline_dir, "2023-06", baseline_records)
    regime_records = _regime_records(flip_slope=0.06, flip_intercept=0.28, symbols=SYMBOLS)
    _write_q4_json(regime_dir, "2023-07", regime_records, corrupt_regressions=True)

    out_dir = tmp_path / "results"
    result = run_q8(
        out_dir, baseline_dir=baseline_dir, regime_dirs={"2023-07": regime_dir},
        baseline_label="2023-06",
    )

    regime_summary = result["regime_summaries"]["2023-07"]
    assert regime_summary["flip_law_mismatch_warning"] is not None
    assert "disagrees" in regime_summary["flip_law_mismatch_warning"]
    # The recomputed slope (trustworthy) must NOT equal the corrupted stored one.
    assert regime_summary["flip_law"]["slope"] != pytest.approx(0.06 + 5.0, abs=0.5)


def test_run_q8_same_sign_all_regimes_true_when_no_flip(tmp_path: Path):
    """All regimes sharing the baseline's flip-slope sign -> same_sign must be True.

    Guards against a hardcoded `False` in `_law_stability`: this fixture has
    NO sign-flipped regime at all, so the only way this assertion passes is
    if the sign comparison is actually performed.
    """
    baseline_dir = tmp_path / "baseline"
    a_dir = tmp_path / "regime_a"
    b_dir = tmp_path / "regime_b"

    baseline_records = _baseline_records()
    _write_q4_json(baseline_dir, "2023-06", baseline_records)

    # Both regimes keep the same positive sign as baseline (+0.05).
    a_records = _regime_records(flip_slope=0.06, flip_intercept=0.28, symbols=SYMBOLS)
    _write_q4_json(a_dir, "2023-07", a_records)
    b_records = _regime_records(flip_slope=0.04, flip_intercept=0.32, symbols=SYMBOLS)
    _write_q4_json(b_dir, "2024-07", b_records)

    out_dir = tmp_path / "results"
    result = run_q8(
        out_dir,
        baseline_dir=baseline_dir,
        regime_dirs={"2023-07": a_dir, "2024-07": b_dir},
        baseline_label="2023-06",
    )

    stability = result["law_stability"]
    assert stability["flip_law_same_sign_all_regimes"] is True
    for label, slope in stability["flip_law_slope_by_label"].items():
        assert slope > 0, f"{label} slope unexpectedly non-positive: {slope}"


def test_reg_mismatch_flags_n_only_difference():
    """`_reg_mismatch` must flag a mismatch when only `n` differs (slope/intercept/r2 equal)."""
    stored = {"slope": 0.05, "intercept": 0.3, "stderr": 0.01, "r2": 0.5, "n": 10}
    recomputed = {"slope": 0.05, "intercept": 0.3, "stderr": 0.01, "r2": 0.5, "n": 9}
    warning = _reg_mismatch(recomputed, stored)
    assert warning is not None
    assert "n differs" in warning


def test_run_q8_output_files_exist(three_regime_dirs: dict, tmp_path: Path):
    out_dir = tmp_path / "results"
    run_q8(
        out_dir,
        baseline_dir=three_regime_dirs["baseline_dir"],
        regime_dirs=three_regime_dirs["regime_dirs"],
        baseline_label="2023-06",
    )
    assert (out_dir / "q8_regimes.json").exists()
    assert (out_dir / "q8_regimes.md").exists()
    assert (out_dir / "q8_regimes.png").exists()

    # json round-trips
    payload = json.loads((out_dir / "q8_regimes.json").read_text())
    assert payload["baseline_label"] == "2023-06"
    assert set(payload["regime_labels"]) == {"2023-07", "2024-07", "2026-07"}


def test_run_q8_single_regime_no_q6_either_side(tmp_path: Path):
    """Baseline and single regime both lacking q6 must not raise and must report None alpha fields."""
    baseline_dir = tmp_path / "baseline"
    regime_dir = tmp_path / "regime"
    _write_q4_json(baseline_dir, "2023-06", _baseline_records())
    _write_q4_json(regime_dir, "2023-07", _regime_records(0.05, 0.3, SYMBOLS))

    out_dir = tmp_path / "results"
    result = run_q8(
        out_dir, baseline_dir=baseline_dir, regime_dirs={"2023-07": regime_dir},
        baseline_label="2023-06",
    )
    assert result["baseline_summary"]["alpha_median"] is None
    assert result["regime_summaries"]["2023-07"]["alpha_median"] is None
    rc = result["rank_correlations"]["2023-07"]
    assert rc["alpha_spearman"] is None
    assert rc["alpha_n_overlap"] == 0
