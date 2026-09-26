"""Q6b: kernel-K sensitivity panel — does n̂ rise with K because of long memory or drift?

Method: mirrors Q6's business-time pipeline exactly (load_events -> ts as
int64 epoch-ms -> intraday_rate_profile (48 bins) -> rescale_to_business_time
-> K equal-width contiguous sub-windows), but instead of fitting ONE
exponential-kernel Hawkes MLE per window, this module fits the
sum-of-exponentials MLE (`fit_hawkes_multiexp`) at K in {1, 2, 3} for EVERY
sub-window and records how the fitted branching ratio n̂_K moves across K.

Why this matters (docs/research/02-hawkes-processes.md §4 pitfall 3): a
single exponential kernel is too short-memoried to represent a true
long-memory/power-law-like kernel, so a K=1 MLE systematically underestimates
n = sum(alpha_k). Increasing K lets the mixture spread mass across widely
separated timescales and recover more of the long-lag kernel weight, pulling
n̂ up toward the true value — Q6's headline 41/41-symbol MLE-vs-count-variance
disagreement is consistent with exactly this kind of kernel misspecification.
Re-fitting the SAME data at K=1,2,3 and reporting the K=1->K=2 jump (Δ21) is
therefore the diagnostic this module exists to compute.

THE CONFOUND (must be read before trusting a large Δ21 as "kernel
misspecification confirmed"): a rising n̂(K) together with a slow-decaying
component is ALSO exactly what residual baseline non-stationarity produces,
even with a perfectly well-specified single-exponential TRUE kernel. Case A
(this repo's own synthetic control, see test_q6b.py):  a true n=0.4
single-exponential process with a ±30% baseline-rate wobble that survives
imperfect deseasonalization fits at K=1 n̂≈0.46 and at K=2 n̂≈0.83 — a large,
spurious Δ21 with NO long-memory kernel anywhere in the generative model. The
mechanism is the same one `branching_count_variance`'s regime-switching trap
documents for the model-free estimator (Filimonov & Sornette 2015): a
non-stationary mu(t) masquerades as self-excitation, and a second exponential
component with a very slow beta is a flexible enough shape to partially
absorb a slow drift in the baseline rate, inflating K=2's alpha sum without
any genuine long-range kernel mass being present.

This module's response to the confound is NOT to claim it can tell the two
apart from a single K-sweep — it explicitly cannot, with the tools built so
far. Instead, per symbol, it reports 1/β_slow (the slower component's decay
timescale, at K=2, converted to business-time SECONDS) against two
independent physical scales: the deseasonalization bin width (86400/48
seconds ≈ 1800s — a slow component decaying on a timescale comparable to or
longer than a deseasonalization bin is exactly what a residual bin-scale
drift would produce) and the sub-window length itself (a slow component
whose timescale approaches the window length is barely distinguishable from
a linear trend within that window). Symbols where 1/β_slow exceeds
DRIFT_SUSPECT_MULTIPLIER (10x) the bin width are flagged "drift-suspect" in
the per-symbol output — a large Δ21 on a drift-suspect symbol should be read
as ambiguous between genuine long-memory and residual non-stationarity, not
as confirmed endogeneity. The decisive control that WOULD separate the two
explanations — refitting K=1 with a block-wise (time-varying) mu instead of
a single constant mu — is out of scope for this module and is named as
pending in the markdown output, not implemented here.

Symbols are processed one at a time; any per-symbol exception is caught and
logged into `failures`, and never aborts the run for the remaining symbols.

Outputs: q6b_kernel_sensitivity.{json,md,parquet,png}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from microstructure.data.catalog import parquet_path
from microstructure.estimators.hawkes import fit_hawkes_multiexp
from microstructure.signals.eventtime import intraday_rate_profile, rescale_to_business_time
from microstructure.signals.load import load_events

N_BINS = 48  # intraday_rate_profile bin count, matches Q6

# Runtime cap on events per single sum-of-exponentials Hawkes MLE fit. Fitting
# at K=1,2,3 is strictly more expensive per event than Q6's single K=1 fit
# (K recursions per Nelder-Mead evaluation, dimension 1+2K instead of 3), so
# the same 250k-event cap Q6 uses per window is kept here rather than raised;
# see q6_endogeneity.py's MAX_FIT_EVENTS docstring for the sampling-noise
# justification, which applies unchanged to the K=1 fit and is, if anything,
# more conservative for K=2/K=3 since more parameters share the same sample.
MAX_FIT_EVENTS = 250_000

# Deseasonalization bin width in business-time SECONDS: one day (86400s) of
# business time, split into the same N_BINS=48 bins intraday_rate_profile
# uses. A slow kernel component decaying on a timescale comparable to (or
# longer than) this width is exactly the shape a residual bin-scale
# seasonality artifact (imperfect deseasonalization) would produce, since the
# profile cannot resolve structure finer than one bin.
DESEASON_BIN_WIDTH_S = 86_400.0 / N_BINS

# A symbol's median 1/beta_slow (at K=2) exceeding this multiple of the
# deseasonalization bin width is flagged "drift-suspect": the slow component
# is decaying on a timescale far longer than anything the deseasonalization
# step could have resolved, so it is at least as consistent with residual
# baseline drift leaking through deseasonalization as with a genuine
# long-memory kernel component. This is a flag for caution, not a verdict —
# seeing the module docstring's confound discussion for why a single K-sweep
# cannot settle the question on its own.
DRIFT_SUSPECT_MULTIPLIER = 10.0

DEFAULT_KS: tuple[int, ...] = (1, 2, 3)


def _fit_capped_multiexp(times: np.ndarray, t_end: float, k: int) -> dict:
    """Fit the K-component sum-of-exponentials Hawkes MLE, capping events.

    Returns a dict with n, alphas, betas, beta_slow, inv_beta_slow,
    converged. `beta_slow` is min(betas) (the slowest-decaying component);
    `inv_beta_slow` is its timescale 1/beta_slow in business-time seconds
    (matching the units `times`/`t_end` are already expressed in, since this
    module always calls this AFTER business-time rescaling).
    """
    if times.size > MAX_FIT_EVENTS:
        times = times[:MAX_FIT_EVENTS]
        t_end = float(times[-1])
    fit = fit_hawkes_multiexp(times, t_end, K=k)
    beta_slow = float(np.min(fit.betas))
    return {
        "n": fit.n,
        "alphas": fit.alphas.tolist(),
        "betas": fit.betas.tolist(),
        "beta_slow": beta_slow,
        "inv_beta_slow": float(1.0 / beta_slow) if beta_slow > 0.0 else float("inf"),
        "converged": fit.converged,
    }


def _window_edges(bt: np.ndarray, windows: int) -> np.ndarray:
    """K+1 equal-width business-time edges spanning [bt[0], bt[-1]]. Same
    convention as q6_endogeneity._window_edges."""
    return np.linspace(bt[0], bt[-1], windows + 1)


def _fit_window_all_ks(window_times: np.ndarray, t_end: float, ks: tuple[int, ...]) -> dict:
    """Fit every K in `ks` on one sub-window's (already zero-anchored) times.

    Returns {k: {n, alphas, betas, beta_slow, inv_beta_slow, converged}}.
    """
    return {k: _fit_capped_multiexp(window_times, t_end, k) for k in ks}


def _fit_business_time_windows(
    bt: np.ndarray, windows: int, ks: tuple[int, ...]
) -> list[dict]:
    """Fit every K in `ks` independently on each of `windows` contiguous
    business-time sub-windows.

    Returns a list of length `windows`, each element the per-K fit dict from
    `_fit_window_all_ks` plus the sub-window's own `window_length_s`. Windows
    with fewer than 2 events raise (same as Q6), turned into a per-symbol
    failure by the caller.
    """
    edges = _window_edges(bt, windows)
    results: list[dict] = []

    for i in range(windows):
        lo, hi = edges[i], edges[i + 1]
        mask = (bt >= lo) & (bt <= hi) if i == windows - 1 else (bt >= lo) & (bt < hi)
        window_times = bt[mask] - lo
        if window_times.size < 2:
            raise ValueError(f"sub-window {i} has only {window_times.size} events, need >= 2")
        t_end = float(window_times[-1])
        if t_end <= 0.0:
            raise ValueError(f"sub-window {i} has zero-length business-time span")
        per_k = _fit_window_all_ks(window_times, t_end, ks)
        results.append({"per_k": per_k, "window_length_s": float(hi - lo)})

    return results


def _is_drift_suspect(median_inv_beta_slow_k2: float) -> bool:
    """True if the K=2 slow-component timescale exceeds DRIFT_SUSPECT_MULTIPLIER
    times the deseasonalization bin width. See module docstring."""
    if not np.isfinite(median_inv_beta_slow_k2):
        return True
    return median_inv_beta_slow_k2 > DRIFT_SUSPECT_MULTIPLIER * DESEASON_BIN_WIDTH_S


def _symbol_record(root: Path, symbol: str, month: str, windows: int, ks: tuple[int, ...]) -> dict:
    events = load_events(root, symbol, [month])
    n_events = events.height
    if n_events == 0:
        raise ValueError(f"no events for {symbol} in {month}")

    ts_ms = events["ts"].dt.epoch("ms").to_numpy().astype(np.int64)
    ts_ms.sort()

    profile = intraday_rate_profile(ts_ms, N_BINS)
    bt = rescale_to_business_time(ts_ms, profile)

    min_events_needed = 2 * windows
    if bt.size < min_events_needed:
        raise ValueError(
            f"{symbol}: only {bt.size} events, need >= {min_events_needed} for {windows} "
            "sub-windows with >= 2 events each"
        )

    window_fits = _fit_business_time_windows(bt, windows, ks)
    window_length_s = float(np.median([w["window_length_s"] for w in window_fits]))

    n_hat_by_k: dict[int, list[float]] = {k: [] for k in ks}
    converged_by_k: dict[int, list[bool]] = {k: [] for k in ks}
    inv_beta_slow_by_k: dict[int, list[float]] = {k: [] for k in ks}

    for w in window_fits:
        for k in ks:
            fit = w["per_k"][k]
            n_hat_by_k[k].append(fit["n"])
            converged_by_k[k].append(fit["converged"])
            inv_beta_slow_by_k[k].append(fit["inv_beta_slow"])

    n_median_by_k = {k: float(np.median(n_hat_by_k[k])) for k in ks}
    n_converged_by_k = {k: int(sum(converged_by_k[k])) for k in ks}

    median_inv_beta_slow_k2 = (
        float(np.median(inv_beta_slow_by_k[2])) if 2 in inv_beta_slow_by_k else float("nan")
    )
    drift_suspect = _is_drift_suspect(median_inv_beta_slow_k2)

    delta21 = (
        n_median_by_k[2] - n_median_by_k[1]
        if 1 in n_median_by_k and 2 in n_median_by_k
        else float("nan")
    )

    ratio_bin_width = (
        median_inv_beta_slow_k2 / DESEASON_BIN_WIDTH_S if np.isfinite(median_inv_beta_slow_k2) else float("nan")
    )
    ratio_window_length = (
        median_inv_beta_slow_k2 / window_length_s
        if np.isfinite(median_inv_beta_slow_k2) and window_length_s > 0.0
        else float("nan")
    )

    return {
        "symbol": symbol,
        "n_events": n_events,
        "windows": windows,
        "ks": list(ks),
        "n_median_by_k": n_median_by_k,
        "n_converged_by_k": n_converged_by_k,
        "delta21": delta21,
        "median_inv_beta_slow_k2_s": median_inv_beta_slow_k2,
        "window_length_s": window_length_s,
        "bin_width_s": DESEASON_BIN_WIDTH_S,
        "ratio_inv_beta_slow_to_bin_width": ratio_bin_width,
        "ratio_inv_beta_slow_to_window_length": ratio_window_length,
        "drift_suspect": drift_suspect,
        "per_window": [
            {
                "n_by_k": {k: w["per_k"][k]["n"] for k in ks},
                "converged_by_k": {k: w["per_k"][k]["converged"] for k in ks},
                "inv_beta_slow_by_k": {k: w["per_k"][k]["inv_beta_slow"] for k in ks},
            }
            for w in window_fits
        ],
    }


def _load_q6_gap(q6_json: Path | None) -> dict[str, float]:
    """Load {symbol: alpha_cv - alpha_median} from a Q6 results JSON, if given.

    Returns an empty dict if `q6_json` is None or the file doesn't exist —
    the cross-section correlation with Δ21 is then simply skipped, not an
    error (this diagnostic is optional supplementary context, not a
    dependency Q6b requires to run).
    """
    if q6_json is None or not q6_json.exists():
        return {}
    data = json.loads(q6_json.read_text())
    gaps: dict[str, float] = {}
    for rec in data.get("records", []):
        symbol = rec.get("symbol")
        alpha_cv = rec.get("alpha_cv")
        alpha_median = rec.get("alpha_median")
        if symbol is None or alpha_cv is None or alpha_median is None:
            continue
        gaps[symbol] = float(alpha_cv) - float(alpha_median)
    return gaps


def _cross_section(records: list[dict], q6_gaps: dict[str, float]) -> dict:
    if not records:
        return {
            "delta21_distribution": None,
            "frac_n2_at_least_0_9": None,
            "cv_gap_correlation": None,
        }

    delta21s = np.array([r["delta21"] for r in records if np.isfinite(r["delta21"])])
    delta21_distribution = (
        {
            "median": float(np.median(delta21s)),
            "mean": float(np.mean(delta21s)),
            "std": float(np.std(delta21s, ddof=1)) if delta21s.size >= 2 else 0.0,
            "min": float(np.min(delta21s)),
            "max": float(np.max(delta21s)),
            "n": int(delta21s.size),
        }
        if delta21s.size > 0
        else None
    )

    n2_values = [r["n_median_by_k"].get(2) for r in records if 2 in r["n_median_by_k"]]
    frac_n2_at_least_0_9 = (
        float(sum(1 for v in n2_values if v >= 0.9) / len(n2_values)) if n2_values else None
    )

    cv_gap_correlation = None
    if q6_gaps:
        paired = [
            (r["delta21"], q6_gaps[r["symbol"]])
            for r in records
            if r["symbol"] in q6_gaps and np.isfinite(r["delta21"])
        ]
        if len(paired) >= 2:
            delta_arr = np.array([p[0] for p in paired])
            gap_arr = np.array([p[1] for p in paired])
            if np.std(delta_arr) > 0 and np.std(gap_arr) > 0:
                cv_gap_correlation = {
                    "correlation": float(np.corrcoef(delta_arr, gap_arr)[0, 1]),
                    "n": len(paired),
                }
            else:
                cv_gap_correlation = {"correlation": None, "n": len(paired)}

    return {
        "delta21_distribution": delta21_distribution,
        "frac_n2_at_least_0_9": frac_n2_at_least_0_9,
        "cv_gap_correlation": cv_gap_correlation,
    }


def run_q6b(
    root: Path,
    out_dir: Path,
    symbols: list[str],
    month: str = "2023-06",
    windows: int = 6,
    ks: tuple[int, ...] = DEFAULT_KS,
    q6_json: Path | None = None,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    failures: list[dict] = []

    for symbol in symbols:
        p = parquet_path(root, symbol, "aggTrades", month)
        if not p.exists():
            failures.append({"symbol": symbol, "reason": f"parquet not found: {p}"})
            continue
        try:
            records.append(_symbol_record(root, symbol, month, windows, ks))
        except Exception as e:  # noqa: BLE001 - per-symbol robustness is the point
            failures.append({"symbol": symbol, "reason": f"{type(e).__name__}: {e}"})

    q6_gaps = _load_q6_gap(q6_json)
    cross_section = _cross_section(records, q6_gaps)

    result = {
        "month": month,
        "windows": windows,
        "ks": list(ks),
        "n_symbols_requested": len(symbols),
        "n_symbols_successful": len(records),
        "n_symbols_failed": len(failures),
        "records": records,
        "failures": failures,
        "cross_section": cross_section,
        "q6_json_used": str(q6_json) if q6_json is not None else None,
    }

    _plot(out_dir, records, cross_section)
    _write_results_parquet(out_dir, records)
    _write_results_md(out_dir, result)
    (out_dir / "q6b_kernel_sensitivity.json").write_text(json.dumps(result, indent=2))
    return result


def _parquet_rows(records: list[dict]) -> list[dict]:
    rows = []
    for r in records:
        row = {
            "symbol": r["symbol"],
            "n_events": r["n_events"],
            "n_hat_k1": r["n_median_by_k"].get(1),
            "n_hat_k2": r["n_median_by_k"].get(2),
            "n_hat_k3": r["n_median_by_k"].get(3),
            "delta21": r["delta21"],
            "median_inv_beta_slow_k2_s": r["median_inv_beta_slow_k2_s"],
            "ratio_inv_beta_slow_to_bin_width": r["ratio_inv_beta_slow_to_bin_width"],
            "ratio_inv_beta_slow_to_window_length": r["ratio_inv_beta_slow_to_window_length"],
            "drift_suspect": r["drift_suspect"],
        }
        rows.append(row)
    return rows


_PARQUET_SCHEMA = {
    "symbol": pl.Utf8, "n_events": pl.Int64, "n_hat_k1": pl.Float64,
    "n_hat_k2": pl.Float64, "n_hat_k3": pl.Float64, "delta21": pl.Float64,
    "median_inv_beta_slow_k2_s": pl.Float64, "ratio_inv_beta_slow_to_bin_width": pl.Float64,
    "ratio_inv_beta_slow_to_window_length": pl.Float64, "drift_suspect": pl.Boolean,
}


def _write_results_parquet(out_dir: Path, records: list[dict]) -> None:
    if records:
        df = pl.DataFrame(_parquet_rows(records))
    else:
        df = pl.DataFrame(schema=_PARQUET_SCHEMA)
    df.write_parquet(out_dir / "q6b_kernel_sensitivity.parquet")


def _plot(out_dir: Path, records: list[dict], cross_section: dict) -> None:
    fig, (ax_delta, ax_slow) = plt.subplots(1, 2, figsize=(13, 5))

    if records:
        delta21s = np.array([r["delta21"] for r in records if np.isfinite(r["delta21"])])
        if delta21s.size > 0:
            ax_delta.hist(delta21s, bins=min(20, max(5, delta21s.size // 2)), color="steelblue", alpha=0.85)
            ax_delta.axvline(0.15, color="red", linestyle="--", linewidth=1.0, label="Δ21 = 0.15")
            ax_delta.legend()
        ax_delta.set_xlabel(r"$\Delta_{21} = \hat n_2 - \hat n_1$ (median across sub-windows)")
        ax_delta.set_ylabel("symbol count")
        ax_delta.set_title("Distribution of K=1→K=2 branching-ratio jump")

        drift_suspect = np.array([r["drift_suspect"] for r in records])
        ratios = np.array([r["ratio_inv_beta_slow_to_bin_width"] for r in records])
        deltas = np.array([r["delta21"] for r in records])
        colors = np.where(drift_suspect, "crimson", "steelblue")
        finite = np.isfinite(ratios) & np.isfinite(deltas)
        ax_slow.scatter(ratios[finite], deltas[finite], c=colors[finite], s=36, alpha=0.8)
        ax_slow.axvline(
            10.0, color="gray", linestyle="--", linewidth=1.0,
            label="drift-suspect threshold (10x bin width)",
        )
        ax_slow.legend()
    else:
        ax_delta.set_xlabel(r"$\Delta_{21} = \hat n_2 - \hat n_1$")
        ax_delta.set_ylabel("symbol count")
        ax_delta.set_title("Distribution of K=1→K=2 branching-ratio jump")

    ax_slow.set_xlabel(r"$1/\hat\beta_{slow}$ (K=2) ÷ deseasonalization bin width")
    ax_slow.set_ylabel(r"$\Delta_{21}$")
    ax_slow.set_title("Slow-timescale ratio vs. Δ21 (red = drift-suspect)")

    fig.tight_layout()
    fig.savefig(out_dir / "q6b_kernel_sensitivity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fmt_ratio(ratio: float) -> str:
    if not np.isfinite(ratio):
        return "n/a"
    return f"{ratio:.2f}x"


def _write_results_md(out_dir: Path, result: dict) -> None:
    records = result["records"]
    month = result["month"]
    windows = result["windows"]
    ks = result["ks"]
    cross_section = result["cross_section"]

    lines: list[str] = []
    lines.append("# Q6b: kernel-K sensitivity panel — does n̂ rise with K, and why?")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append(
        f"For each symbol, one month ({month}) of aggTrades is loaded (`load_events`), "
        "rescaled to business time (`intraday_rate_profile` + `rescale_to_business_time`, "
        f"{N_BINS} bins) exactly as in Q6, then split into {windows} equal contiguous "
        f"sub-windows. Each sub-window is refit at every K in {ks} via "
        "`fit_hawkes_multiexp` (sum-of-K-exponentials Hawkes MLE), instead of Q6's single "
        "K=1 fit. Per symbol, per sub-window, this records n̂_K = sum(alpha_k) and "
        "β_slow_K = min(betas_K) (the slowest-decaying component's rate) for every K."
    )
    lines.append("")
    lines.append(
        f"**Runtime cap**: a sub-window with more than {MAX_FIT_EVENTS:,} events is fit "
        f"on only the first {MAX_FIT_EVENTS:,} of that window's events (same cap and "
        "justification as Q6's MAX_FIT_EVENTS, applied per-K here)."
    )
    lines.append("")
    lines.append(
        "**Per-symbol summary**: the median across sub-windows of n̂_1, n̂_2, n̂_3, and "
        "Δ21 = median(n̂_2) − median(n̂_1) — the headline K=1→K=2 branching-ratio jump. "
        "Also reported: the median across sub-windows of 1/β_slow at K=2 (the slower "
        f"component's timescale, in business-time seconds), compared against the "
        f"deseasonalization bin width ({DESEASON_BIN_WIDTH_S:.1f}s = 86400/{N_BINS}) and "
        "against the sub-window length itself, as two independent ratios."
    )
    lines.append("")
    lines.append(
        "## The confound this analysis does NOT resolve on its own"
    )
    lines.append("")
    lines.append(
        "**A K=1→K=2 rise in n̂ together with a slow kernel component is also produced by "
        "residual baseline non-stationarity, not only by a genuine long-memory kernel.** "
        "This repo's own synthetic control (test_q6b.py, \"Case A\") demonstrates the "
        "trap directly: a TRUE n=0.4 single-exponential process with a ±30% baseline-rate "
        "wobble that survives imperfect deseasonalization fits at K=1 n̂≈0.46 and at K=2 "
        "n̂≈0.83 — a large, spurious Δ21 with no long-memory kernel anywhere in the "
        "generative model. A second exponential component with a very slow beta is "
        "flexible enough to partially absorb a slow drift in the baseline rate, inflating "
        "the K=2 branching-ratio sum without any genuine long-range kernel mass being "
        "present. This is the same mechanism Filimonov & Sornette (2015) document for the "
        "model-free count-variance estimator's regime-switching trap, now shown to affect "
        "the sum-of-exponentials MLE as well."
    )
    lines.append("")
    lines.append(
        "**Why this analysis reports 1/β_slow vs. bin width instead of trusting Δ21 "
        "alone.** A slow component decaying on a timescale comparable to or longer than "
        "the deseasonalization bin width is exactly the shape a residual bin-scale "
        "seasonality artifact would produce — the 48-bin intraday profile cannot resolve "
        "structure finer than one bin, so any leftover non-stationarity at or above that "
        "scale is a plausible source for a slow K=2 component, not necessarily a genuine "
        f"long-memory kernel. Symbols where the median 1/β_slow (K=2) exceeds "
        f"{DRIFT_SUSPECT_MULTIPLIER:.0f}x the bin width are flagged **drift-suspect** in "
        "the panel table below — a large Δ21 on a drift-suspect symbol should be read as "
        "ambiguous between genuine long-memory and residual non-stationarity, not as "
        "confirmed endogeneity."
    )
    lines.append("")
    lines.append(
        "**The decisive control is pending, not implemented here.** The analysis that "
        "WOULD separate these two explanations — refitting K=1 with a block-wise "
        "(time-varying, piecewise-constant) mu instead of a single constant mu per "
        "sub-window — is named here as the necessary next step and is explicitly out of "
        "scope for this module. Until that control is run, this panel's Δ21 and "
        "drift-suspect flag should be read as a triage tool (which symbols deserve the "
        "decisive control first), not a final verdict on kernel misspecification vs. "
        "residual drift."
    )
    lines.append("")

    lines.append("## Run summary")
    lines.append("")
    lines.append(
        f"Requested: {result['n_symbols_requested']}. Successful: "
        f"{result['n_symbols_successful']}. Failed: {result['n_symbols_failed']}."
    )
    lines.append("")

    if records:
        sorted_records = sorted(records, key=lambda r: r["n_events"], reverse=True)
        lines.append("## Panel table (sorted by n_events)")
        lines.append("")
        lines.append(
            "| symbol | n_events | n̂_1 | n̂_2 | n̂_3 | Δ21 | 1/β_slow (K=2, s) | "
            "÷ bin width | ÷ window length | drift-suspect |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted_records:
            n1 = r["n_median_by_k"].get(1)
            n2 = r["n_median_by_k"].get(2)
            n3 = r["n_median_by_k"].get(3)
            n1s = f"{n1:.4f}" if n1 is not None else "n/a"
            n2s = f"{n2:.4f}" if n2 is not None else "n/a"
            n3s = f"{n3:.4f}" if n3 is not None else "n/a"
            lines.append(
                f"| {r['symbol']} | {r['n_events']:,} | {n1s} | {n2s} | {n3s} | "
                f"{r['delta21']:+.4f} | {r['median_inv_beta_slow_k2_s']:.2f} | "
                f"{_fmt_ratio(r['ratio_inv_beta_slow_to_bin_width'])} | "
                f"{_fmt_ratio(r['ratio_inv_beta_slow_to_window_length'])} | "
                f"{'YES' if r['drift_suspect'] else 'no'} |"
            )
        lines.append("")
    else:
        lines.append("No symbols produced usable results — no table to show.")
        lines.append("")

    lines.append("## Cross-section")
    lines.append("")
    dist = cross_section["delta21_distribution"]
    if dist is not None:
        lines.append(
            f"**Δ21 distribution** across {dist['n']} symbols: median = **{dist['median']:.4f}**, "
            f"mean = {dist['mean']:.4f}, sd = {dist['std']:.4f}, range = "
            f"[{dist['min']:.4f}, {dist['max']:.4f}]."
        )
    else:
        lines.append("Δ21 distribution not estimable (no successful symbols with both K=1, K=2).")
    lines.append("")
    frac = cross_section["frac_n2_at_least_0_9"]
    if frac is not None:
        lines.append(
            f"**Fraction of symbols with n̂_2 ≥ 0.9** (near-critical at K=2): "
            f"**{frac:.1%}**."
        )
    else:
        lines.append("Fraction of symbols with n̂_2 ≥ 0.9 not estimable (no K=2 results).")
    lines.append("")
    cv_corr = cross_section["cv_gap_correlation"]
    if cv_corr is not None and cv_corr.get("correlation") is not None:
        lines.append(
            f"**Correlation of Δ21 with the Q6 count-variance gap** (alpha_cv − "
            f"alpha_median, from `--q6-json`): **{cv_corr['correlation']:.4f}** "
            f"(n={cv_corr['n']}). A positive correlation is consistent with Q6's "
            "MLE-vs-count-variance disagreement being driven, at least partly, by the "
            "same K=1 exponential-kernel underestimation this module's Δ21 is designed "
            "to detect — but per the confound discussion above, is equally consistent "
            "with both estimators sharing exposure to the same residual baseline "
            "non-stationarity, and cannot on its own distinguish the two."
        )
    elif result.get("q6_json_used"):
        lines.append(
            "Correlation of Δ21 with the Q6 count-variance gap was requested "
            f"(`--q6-json {result['q6_json_used']}`) but not estimable (fewer than 2 "
            "overlapping symbols, or zero variance in one of the two series)."
        )
    else:
        lines.append(
            "No `--q6-json` supplied — correlation of Δ21 with the Q6 count-variance gap "
            "was skipped."
        )
    lines.append("")

    if result["failures"]:
        lines.append("## Failures")
        lines.append("")
        lines.append("| symbol | reason |")
        lines.append("|---|---|")
        for f in result["failures"]:
            lines.append(f"| {f['symbol']} | {f['reason']} |")
        lines.append("")

    lines.append("## Findings")
    lines.append("")
    if dist is not None:
        drift_count = sum(1 for r in records if r["drift_suspect"])
        lines.append(
            f"Across {dist['n']} successful symbols, the median K=1→K=2 branching-ratio "
            f"jump is **{dist['median']:+.4f}**. {drift_count}/{len(records)} symbols are "
            "flagged drift-suspect (median 1/β_slow at K=2 exceeds "
            f"{DRIFT_SUSPECT_MULTIPLIER:.0f}x the deseasonalization bin width of "
            f"{DESEASON_BIN_WIDTH_S:.1f}s) — for these symbols, the Δ21 reported above "
            "should be treated as ambiguous between genuine long-memory kernel structure "
            "and residual baseline drift leaking through deseasonalization, per the "
            "confound discussion above."
        )
    else:
        lines.append("No successful symbols in this run — no finding to report.")
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- **The confound is not resolved by this module.** See the dedicated section "
        "above: a large Δ21 is consistent with BOTH genuine long-memory kernel structure "
        "and residual baseline non-stationarity surviving deseasonalization. The "
        "decisive control (K=1 fit with a block-wise, time-varying mu) is pending."
    )
    lines.append(
        f"- **Single month** ({month}): one specific market regime; results may not "
        "generalize to other months."
    )
    lines.append(
        "- **Higher-K identifiability**: per `fit_hawkes_multiexp`'s own docstring, "
        "individual alpha_k/beta_k components become less identified as K grows relative "
        "to what the sample size and window can resolve — n̂_K (the sum) is more "
        "trustworthy than any individual component, but β_slow (the min beta) can still "
        "be noisy at K=3 in particular."
    )
    lines.append(
        f"- **Runtime cap** ({MAX_FIT_EVENTS:,} events/window): windows above this cap "
        "are fit on a truncated prefix, applied independently at each K."
    )
    lines.append(
        "- **48-bin intraday profile**: same caveat as Q6 — the profile is estimated "
        "from the same month being fit, and any genuine excitation clustering at "
        "~30-minute resolution could leak into deseasonalization."
    )
    lines.append("")
    (out_dir / "q6b_kernel_sensitivity.md").write_text("\n".join(lines))


def _parse_ks(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Q6b: kernel-K sensitivity panel")
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--symbols-file", type=Path, required=True, help="one symbol per line")
    parser.add_argument("--month", type=str, default="2023-06")
    parser.add_argument("--windows", type=int, default=6)
    parser.add_argument("--ks", type=str, default="1,2,3", help="comma-separated K values")
    parser.add_argument(
        "--q6-json", type=Path, default=None,
        help="path to a Q6 results JSON, for the Δ21-vs-count-variance-gap correlation",
    )
    return parser.parse_args(argv)


def _read_symbols_file(path: Path) -> list[str]:
    lines = path.read_text().splitlines()
    return [s.strip() for s in lines if s.strip()]


if __name__ == "__main__":
    args = _parse_args()
    symbols = _read_symbols_file(args.symbols_file)
    run_q6b(
        args.root, args.out, symbols=symbols, month=args.month,
        windows=args.windows, ks=_parse_ks(args.ks), q6_json=args.q6_json,
    )
