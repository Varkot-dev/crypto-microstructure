"""Tests for Q6b: kernel-K sensitivity panel.

Contract under test (see the q6b_kernel_sensitivity.py module docstring):
1. A planted TWO-timescale-kernel symbol (alpha=(0.25,0.35), beta=(5,0.2),
   true n=0.6) must show a large K=1->K=2 rise (Delta21 > 0.15) and n_hat_2
   within +-0.1 of 0.6 -- the headline "K=1 underestimates a genuinely
   long-memory kernel" finding.
2. A planted SINGLE-exponential-kernel symbol (n=0.4, well-specified at
   K=1) must show a small Delta21 (|Delta21| < 0.08) -- the negative
   control: no spurious rise when the true kernel really is K=1.
3. A missing symbol must land in `failures`, never abort the run.
4. Output files (.json/.md/.parquet/.png) must exist, and the parquet row
   count must equal the number of successful symbols.
5. The drift-suspect flagging logic (1/beta_slow at K=2 vs. the
   deseasonalization bin width) is unit-tested directly against
   `_is_drift_suspect`, independent of a full pipeline run.

Runtime: windows=2 and ks=(1,2) are used throughout (rather than the
production defaults windows=6, ks=(1,2,3)) to keep this file's wall clock
well under the ~90s synchronous budget -- fit_hawkes_multiexp's K=2 fit
dominates cost (~1.5-2s per ~30k-event window measured locally), so cutting
windows from 6 to 2 and dropping K=3 entirely is a ~6x-9x runtime reduction
relative to full production settings, at the cost of using only 2
(not 6) window-level samples for the per-symbol median -- acceptable here
since these tests assert Delta21 direction/magnitude with generous
tolerances, not exact production-panel point estimates. Each planted
fixture is sized to land in the ~35-45k event range (see the per-fixture
comments below), keeping each symbol's total fit cost (2 windows x 2 Ks)
in the single-digit seconds.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from microstructure.analyses.q6b_kernel_sensitivity import (
    DESEASON_BIN_WIDTH_S,
    DRIFT_SUSPECT_MULTIPLIER,
    _is_drift_suspect,
    _symbol_record,
    run_q6b,
)
from microstructure.data.catalog import parquet_path
from microstructure.estimators.hawkes import simulate_hawkes_exp, simulate_hawkes_multiexp

WINDOWS = 2
KS = (1, 2)

# Two-timescale planted kernel: true n = 0.25 + 0.35 = 0.6, timescales 25x
# apart (beta=5 -> fast, beta=0.2 -> slow). t_end=24_000s lands ~30-40k
# events at this (mu, alphas, betas) combination (measured via a one-off
# probe at seed=42: ~30.5k events) -- large enough for 2 sub-windows to each
# carry a statistically meaningful sample for both K=1 and K=2 fits, small
# enough to keep the K=2 fit (the dominant per-window cost) in the
# low-single-digit-seconds range per window.
TWOEXP_MU = 0.5
TWOEXP_ALPHAS = np.array([0.25, 0.35])
TWOEXP_BETAS = np.array([5.0, 0.2])
TWOEXP_N_TRUE = float(TWOEXP_ALPHAS.sum())  # 0.6
TWOEXP_T_END = 24_000.0

# Single-exponential planted kernel: true n = 0.4, well-specified at K=1 --
# the negative control. t_end tuned so this also lands in the ~35-45k event
# range at mu=1.0, beta=2.0 (matches the scale test_hawkes.py and
# test_q6.py's own single-exponential fixtures use).
ONEEXP_MU, ONEEXP_BETA = 1.0, 2.0
ONEEXP_ALPHA = 0.4
ONEEXP_T_END = 20_000.0

DELTA21_TOL_TWOEXP = 0.15  # Delta21 must exceed this on the two-timescale symbol
N2_TOL_TWOEXP = 0.1  # n_hat_2 must land within this of the true n=0.6
DELTA21_TOL_ONEEXP = 0.08  # |Delta21| must stay below this on the single-exp symbol


def _write_event_times_fixture(
    root: Path, symbol: str, times_s: np.ndarray, month: str = "2023-06",
) -> int:
    """Write a raw array of event times (float seconds from an arbitrary
    origin) as an aggTrades-schema parquet: int64 epoch-ms `ts`, alternating
    +-1 aggressor sign via `is_buyer_maker`, positive `qty`. Same
    dedup/nudge convention as test_q6.py's fixture writers (ms collisions
    are nudged forward by 1ms so event COUNT is preserved rather than
    merged away by `to_aggressor_events`). Returns the final event count
    actually written (after dedup nudging).
    """
    t0 = datetime(2023, 6, 1, 0, 0, 0, tzinfo=UTC)
    ts_ms = (times_s * 1000.0).astype(np.int64)
    ts_ms = np.maximum.accumulate(ts_ms)
    for i in range(1, ts_ms.size):
        if ts_ms[i] <= ts_ms[i - 1]:
            ts_ms[i] = ts_ms[i - 1] + 1
    n = ts_ms.size
    event_ts = [t0 + timedelta(milliseconds=int(ms)) for ms in ts_ms]
    signs = np.where(np.arange(n) % 2 == 0, 1, -1)
    qty = np.full(n, 1.0) + 0.01 * (np.arange(n) % 5)

    agg = pl.DataFrame(
        {
            "agg_trade_id": np.arange(n),
            "price": np.full(n, 100.0),
            "qty": qty,
            "first_trade_id": np.arange(n),
            "last_trade_id": np.arange(n),
            "ts": event_ts,
            "is_buyer_maker": signs < 0,
        },
        schema_overrides={"ts": pl.Datetime("ms", "UTC")},
    )
    agg_path = parquet_path(root, symbol, "aggTrades", month)
    agg_path.parent.mkdir(parents=True, exist_ok=True)
    agg.write_parquet(agg_path)
    return n


@pytest.fixture(scope="module")
def planted_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("q6b_planted")

    twoexp_times = simulate_hawkes_multiexp(
        TWOEXP_MU, TWOEXP_ALPHAS, TWOEXP_BETAS, TWOEXP_T_END, seed=42
    )
    assert twoexp_times.size > 2 * WINDOWS * 10, "need enough events for a stable 2-window panel"
    _write_event_times_fixture(root, "TWOEXPUSDT", twoexp_times)

    oneexp_times = simulate_hawkes_exp(ONEEXP_MU, ONEEXP_ALPHA, ONEEXP_BETA, ONEEXP_T_END, seed=7)
    assert oneexp_times.size > 2 * WINDOWS * 10, "need enough events for a stable 2-window panel"
    _write_event_times_fixture(root, "ONEEXPUSDT", oneexp_times)

    # MISSINGUSDT deliberately has no parquet on disk -> must land in failures.
    return root


def test_run_q6b_two_timescale_symbol_shows_large_delta21(planted_root: Path):
    out_dir = planted_root / "results_two"
    result = run_q6b(
        planted_root, out_dir, symbols=["TWOEXPUSDT"], month="2023-06",
        windows=WINDOWS, ks=KS,
    )
    assert result["n_symbols_successful"] == 1, result["failures"]
    rec = result["records"][0]

    assert rec["delta21"] > DELTA21_TOL_TWOEXP, (
        f"expected Delta21 > {DELTA21_TOL_TWOEXP} on the two-timescale symbol, "
        f"got {rec['delta21']} (n_hat_1={rec['n_median_by_k'].get(1)}, "
        f"n_hat_2={rec['n_median_by_k'].get(2)})"
    )
    n2 = rec["n_median_by_k"][2]
    assert abs(n2 - TWOEXP_N_TRUE) < N2_TOL_TWOEXP, (
        f"expected n_hat_2 within {N2_TOL_TWOEXP} of true n={TWOEXP_N_TRUE}, got {n2}"
    )


def test_run_q6b_single_exp_symbol_shows_small_delta21(planted_root: Path):
    out_dir = planted_root / "results_one"
    result = run_q6b(
        planted_root, out_dir, symbols=["ONEEXPUSDT"], month="2023-06",
        windows=WINDOWS, ks=KS,
    )
    assert result["n_symbols_successful"] == 1, result["failures"]
    rec = result["records"][0]

    assert abs(rec["delta21"]) < DELTA21_TOL_ONEEXP, (
        f"expected |Delta21| < {DELTA21_TOL_ONEEXP} on the well-specified K=1 symbol, "
        f"got {rec['delta21']} (n_hat_1={rec['n_median_by_k'].get(1)}, "
        f"n_hat_2={rec['n_median_by_k'].get(2)})"
    )


def test_run_q6b_missing_symbol_lands_in_failures(planted_root: Path):
    out_dir = planted_root / "results_missing"
    result = run_q6b(
        planted_root, out_dir,
        symbols=["TWOEXPUSDT", "MISSINGUSDT"], month="2023-06",
        windows=WINDOWS, ks=KS,
    )
    by_symbol = {r["symbol"]: r for r in result["records"]}
    assert "TWOEXPUSDT" in by_symbol
    assert "MISSINGUSDT" not in by_symbol

    failures = {f["symbol"]: f for f in result["failures"]}
    assert "MISSINGUSDT" in failures
    assert failures["MISSINGUSDT"]["reason"]
    assert result["n_symbols_successful"] == 1
    assert result["n_symbols_failed"] == 1


def test_run_q6b_never_aborts_on_all_failures(tmp_path: Path):
    out_dir = tmp_path / "results"
    result = run_q6b(
        tmp_path, out_dir, symbols=["GHOSTUSDT"], month="2023-06",
        windows=WINDOWS, ks=KS,
    )
    assert result["records"] == []
    assert len(result["failures"]) == 1
    assert result["failures"][0]["symbol"] == "GHOSTUSDT"
    assert (out_dir / "q6b_kernel_sensitivity.md").exists()
    assert (out_dir / "q6b_kernel_sensitivity.json").exists()
    assert (out_dir / "q6b_kernel_sensitivity.parquet").exists()
    assert (out_dir / "q6b_kernel_sensitivity.png").exists()


def test_run_q6b_outputs_exist_and_parquet_row_count_matches_successes(planted_root: Path):
    out_dir = planted_root / "results_outputs"
    result = run_q6b(
        planted_root, out_dir,
        symbols=["TWOEXPUSDT", "ONEEXPUSDT", "MISSINGUSDT"], month="2023-06",
        windows=WINDOWS, ks=KS,
    )
    assert (out_dir / "q6b_kernel_sensitivity.json").exists()
    assert (out_dir / "q6b_kernel_sensitivity.md").exists()
    assert (out_dir / "q6b_kernel_sensitivity.parquet").exists()
    assert (out_dir / "q6b_kernel_sensitivity.png").exists()

    df = pl.read_parquet(out_dir / "q6b_kernel_sensitivity.parquet")
    assert df.height == result["n_symbols_successful"] == 2
    assert set(df.columns) >= {
        "symbol", "n_events", "n_hat_k1", "n_hat_k2", "n_hat_k3", "delta21",
        "median_inv_beta_slow_k2_s", "ratio_inv_beta_slow_to_bin_width",
        "ratio_inv_beta_slow_to_window_length", "drift_suspect",
    }

    # Per-symbol JSON records carry every field the module docstring promises.
    for rec in result["records"]:
        assert "n_median_by_k" in rec
        assert "n_converged_by_k" in rec
        assert "delta21" in rec
        assert "median_inv_beta_slow_k2_s" in rec
        assert "ratio_inv_beta_slow_to_bin_width" in rec
        assert "ratio_inv_beta_slow_to_window_length" in rec
        assert "drift_suspect" in rec
        assert isinstance(rec["drift_suspect"], bool)


def test_run_q6b_cross_section_reports_delta21_distribution_and_frac_near_critical(
    planted_root: Path,
):
    out_dir = planted_root / "results_cross"
    result = run_q6b(
        planted_root, out_dir,
        symbols=["TWOEXPUSDT", "ONEEXPUSDT"], month="2023-06",
        windows=WINDOWS, ks=KS,
    )
    cross = result["cross_section"]
    assert cross["delta21_distribution"] is not None
    assert cross["delta21_distribution"]["n"] == 2
    assert cross["frac_n2_at_least_0_9"] is not None
    # Neither planted symbol has n_hat_2 >= 0.9 (0.6 and ~0.4 respectively).
    assert cross["frac_n2_at_least_0_9"] == 0.0


def test_run_q6b_correlates_delta21_with_q6_count_variance_gap(planted_root: Path, tmp_path: Path):
    """Optional --q6-json input: when present, the cross-section correlates
    Delta21 with (alpha_cv - alpha_median) per symbol, keyed by symbol name.
    This is a wiring test (the correlation itself needs no particular
    sign/magnitude here) -- it only asserts the field is populated when the
    JSON overlaps the run's symbols, and cleanly absent when it does not.
    """
    q6_json_path = tmp_path / "fake_q6.json"
    q6_json_path.write_text(
        '{"records": ['
        '{"symbol": "TWOEXPUSDT", "alpha_median": 0.5, "alpha_cv": 0.7},'
        '{"symbol": "ONEEXPUSDT", "alpha_median": 0.3, "alpha_cv": 0.32}'
        "]}"
    )

    out_dir = planted_root / "results_q6json"
    result = run_q6b(
        planted_root, out_dir,
        symbols=["TWOEXPUSDT", "ONEEXPUSDT"], month="2023-06",
        windows=WINDOWS, ks=KS, q6_json=q6_json_path,
    )
    cv_corr = result["cross_section"]["cv_gap_correlation"]
    assert cv_corr is not None
    assert cv_corr["n"] == 2

    # Without --q6-json, the field must be present but inert (None / no entry).
    out_dir_no_q6 = planted_root / "results_no_q6json"
    result_no_q6 = run_q6b(
        planted_root, out_dir_no_q6,
        symbols=["TWOEXPUSDT", "ONEEXPUSDT"], month="2023-06",
        windows=WINDOWS, ks=KS,
    )
    assert result_no_q6["cross_section"]["cv_gap_correlation"] is None
    assert result_no_q6["q6_json_used"] is None


# --- drift-suspect flag logic (unit-level, no pipeline run needed) ---------


def test_drift_suspect_flags_slow_timescale_far_above_bin_width():
    slow_timescale = DRIFT_SUSPECT_MULTIPLIER * DESEASON_BIN_WIDTH_S * 2.0
    assert _is_drift_suspect(slow_timescale) is True


def test_drift_suspect_does_not_flag_timescale_well_below_threshold():
    fast_timescale = DESEASON_BIN_WIDTH_S * 0.5
    assert _is_drift_suspect(fast_timescale) is False


def test_drift_suspect_boundary_is_strictly_greater_than():
    boundary = DRIFT_SUSPECT_MULTIPLIER * DESEASON_BIN_WIDTH_S
    assert _is_drift_suspect(boundary) is False
    assert _is_drift_suspect(boundary + 1e-6) is True


def test_drift_suspect_flags_non_finite_timescale():
    assert _is_drift_suspect(float("inf")) is True
    assert _is_drift_suspect(float("nan")) is True


# --- JSON serializability of the per-symbol record -------------------------

# Every scalar value that can appear inside a `_symbol_record` return value
# (directly or nested inside `n_median_by_k`/`n_converged_by_k`/`per_window`)
# must be one of these Python-native types. numpy scalars (np.bool_,
# np.float64, np.int64, ...) are NOT included here on purpose: this is the
# regression this test guards against (json.dumps raises `TypeError: Object
# of type bool/float64/... is not JSON serializable` on a numpy scalar, even
# though `isinstance(np.bool_(True), bool)` etc. can be True/False depending
# on the numpy version -- the type() check below is deliberately exact, not
# isinstance-based, so a numpy subclass cannot slip through).
_JSON_NATIVE_SCALAR_TYPES = (str, int, float, bool, type(None))


def _assert_only_native_scalars(value: object, path: str = "$") -> None:
    """Recursively assert every leaf in a JSON-able structure is a Python-
    native scalar (str/int/float/bool/None), not a numpy scalar subclass.

    `type(value) in _JSON_NATIVE_SCALAR_TYPES` (not `isinstance`) is
    intentional: `numpy.bool_`/`numpy.float64` register as subclasses of
    `bool`/`float` on some numpy versions, which would let an isinstance
    check silently pass on exactly the regression this test exists to catch.
    """
    if isinstance(value, dict):
        for k, v in value.items():
            _assert_only_native_scalars(v, f"{path}.{k}")
        return
    if isinstance(value, list):
        for i, v in enumerate(value):
            _assert_only_native_scalars(v, f"{path}[{i}]")
        return
    assert type(value) in _JSON_NATIVE_SCALAR_TYPES, (
        f"{path}: expected a Python-native scalar, got {type(value).__name__} ({value!r}) "
        "-- numpy scalars (e.g. numpy.bool_, numpy.float64) are not JSON serializable "
        "by json.dumps and must be coerced with bool()/float()/int() at the record-"
        "building site before being placed in the record dict"
    )


def test_symbol_record_round_trips_through_json_dumps(planted_root: Path):
    """The per-symbol record dict `_symbol_record` returns must contain only
    Python-native scalar types and must round-trip through `json.dumps`
    (allow_nan=True, the default -- Delta21/ratios can be NaN when a K value
    is absent or a slow component's beta rounds to zero) without raising.

    This directly targets the hotfix regression: `fit.converged` (from
    `fit_hawkes_multiexp`) and the `_is_drift_suspect` comparison result can
    both arrive as numpy scalar types rather than Python `bool`, and
    `json.dumps` has no default encoder for those -- `TypeError: Object of
    type bool is not JSON serializable` (the numpy bool's __class__.__name__
    prints as "bool", which is what made this regression confusing).
    """
    rec = _symbol_record(planted_root, "ONEEXPUSDT", "2023-06", WINDOWS, KS)

    encoded = json.dumps(rec)  # must not raise; allow_nan=True is the default
    decoded = json.loads(encoded)
    assert decoded["symbol"] == "ONEEXPUSDT"

    _assert_only_native_scalars(rec)
