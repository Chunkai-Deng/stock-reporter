#!/usr/bin/env python3
"""
Limit-Up Backtest CLI — 历史涨停股技术面回溯分析。

Usage:
    # Full backfill
    python limit_up_backtest.py --start 2026-01-01 --end 2026-06-26

    # Single month
    python limit_up_backtest.py --start 2026-06-01

    # Incremental (process new days since last run)
    python limit_up_backtest.py --incremental

    # Re-process existing events
    python limit_up_backtest.py --reprocess

    # Query mode
    python limit_up_backtest.py --query "industry=白酒" --limit 10

    # Show statistics
    python limit_up_backtest.py --stats

    # Show help
    python limit_up_backtest.py --help
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cloud_stock_reporter import score_stock

from lib.limit_up_backtest.schema import (
    ensure_schema,
    get_connection,
    close_write_connection,
    set_last_processed_date,
    get_last_processed_date,
)
from lib.limit_up_backtest.fetcher import (
    fetch_all_data_for_dates,
    generate_date_range,
    get_incremental_dates,
    clear_kline_cache,
)
from lib.limit_up_backtest.indicators import (
    compute_indicator_series,
    last_snapshot,
)
from lib.limit_up_backtest.patterns import (
    classify_patterns,
    classify_ma_alignment,
)
from lib.limit_up_backtest.queries import query_events
from lib.limit_up_backtest.stats import get_summary_stats, print_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("limit_up_backtest")


# ── Pipeline ───────────────────────────────────────────────────────────

def process_date_range(
    start: str,
    end: str,
    lookback: int = 90,
    max_workers: int = 4,
    rate_delay: float = 0.5,
) -> dict:
    """Run the full pipeline for a date range.

    Returns: {dates_processed: int, events_saved: int, errors: list}
    """
    # Generate date list
    dates = generate_date_range(start, end)
    if not dates:
        logger.info("No trading dates in range %s – %s", start, end)
        return {"dates_processed": 0, "events_saved": 0, "errors": []}

    logger.info("Processing %d trading dates: %s – %s", len(dates), dates[0], dates[-1])

    # Step 1+2+3+4: Fetch all data
    t0 = time.time()
    events_by_date, errors = fetch_all_data_for_dates(
        dates,
        lookback_days=lookback,
        max_workers=max_workers,
        rate_delay=rate_delay,
        progress_callback=lambda msg: logger.info("  %s", msg),
    )
    logger.info("Data fetch completed in %.1fs", time.time() - t0)

    # Step 5+6+7: Compute indicators, classify patterns, store
    conn = get_connection(readonly=False)
    ensure_schema(conn)

    total_saved = 0
    for d in sorted(events_by_date.keys()):
        day_events = events_by_date[d]
        logger.info("Processing %s: %d events", d, len(day_events))

        saved_in_day = 0
        for event in day_events:
            kline_df = event.pop("_kline_df", None)
            code = event["code"]

            if kline_df is None or len(kline_df) < 26:
                error_msg = "No K-line data available"
                if kline_df is not None:
                    error_msg = f"Only {len(kline_df)} K-line bars (need 26+)"
                errors.append({"date": d, "code": code, "error": error_msg})
                continue

            # Compute indicator time series
            ind_df = compute_indicator_series(kline_df)
            if ind_df is None:
                errors.append({
                    "date": d, "code": code, "error": "Indicator computation failed",
                })
                continue

            # Compute pre-limit-up snapshot (D-1 = second-to-last row)
            d_minus_1_snapshot = None
            pre_score = None
            pre_ma_alignment = ""
            if len(ind_df) >= 2:
                d_minus_1_row = ind_df.iloc[-2]  # D-1 (last row is D=0, the limit-up day)
                d_minus_1_snapshot = {
                    "ma5": _safe_float(d_minus_1_row.get("ma5")),
                    "ma10": _safe_float(d_minus_1_row.get("ma10")),
                    "ma20": _safe_float(d_minus_1_row.get("ma20")),
                    "macd": _safe_float(d_minus_1_row.get("macd")),
                    "macd_signal": _safe_float(d_minus_1_row.get("macd_signal")),
                    "macd_hist": _safe_float(d_minus_1_row.get("macd_hist")),
                    "rsi": _safe_float(d_minus_1_row.get("rsi")),
                    "bb_upper": _safe_float(d_minus_1_row.get("bb_upper")),
                    "bb_middle": _safe_float(d_minus_1_row.get("bb_middle")),
                    "bb_lower": _safe_float(d_minus_1_row.get("bb_lower")),
                    "bb_width_pct": _safe_float(d_minus_1_row.get("bb_width_pct")),
                    "k": _safe_float(d_minus_1_row.get("k")),
                    "d": _safe_float(d_minus_1_row.get("d")),
                    "j": _safe_float(d_minus_1_row.get("j")),
                    "vol_ratio": _safe_float(d_minus_1_row.get("vol_ratio")),
                    "vol_trend": d_minus_1_row.get("vol_trend") or "",
                    "adx": _safe_float(d_minus_1_row.get("adx")),
                    "plus_di": _safe_float(d_minus_1_row.get("plus_di")),
                    "minus_di": _safe_float(d_minus_1_row.get("minus_di")),
                    "divergence": d_minus_1_row.get("divergence") or "",
                    "macd_cross": d_minus_1_row.get("macd_cross") or "",
                }
                # Score D-1 snapshot
                try:
                    pre_score = score_stock(
                        _safe_float(d_minus_1_row.get("close")) or 0,
                        0,  # change_pct on D-1 (we don't have it readily, pass 0)
                        d_minus_1_snapshot,
                        {},  # no weekly for this
                    )
                except Exception:
                    pre_score = None

            pre_ma_alignment = classify_ma_alignment(ind_df)

            # Classify patterns
            patterns = classify_patterns(ind_df)

            # ── Store event ─────────────────────────────────────────
            try:
                cur = conn.execute(
                    """
                    INSERT OR REPLACE INTO limit_up_events
                        (trade_date, code, name, change_pct, close_price,
                         turnover, float_market_cap, total_market_cap,
                         turnover_rate, board_lock_fund, first_lock_time,
                         last_lock_time, blow_count, limit_up_count,
                         consecutive, industry, pre_ma_alignment, pre_score)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        d,
                        code,
                        event.get("name"),
                        event.get("change_pct"),
                        event.get("close_price"),
                        event.get("turnover"),
                        event.get("float_market_cap"),
                        event.get("total_market_cap"),
                        event.get("turnover_rate"),
                        event.get("board_lock_fund"),
                        event.get("first_lock_time"),
                        event.get("last_lock_time"),
                        event.get("blow_count"),
                        event.get("limit_up_count"),
                        event.get("consecutive"),
                        event.get("industry"),
                        pre_ma_alignment,
                        pre_score,
                    ),
                )
                event_id = cur.lastrowid

                # ── Store daily indicators ──────────────────────────
                # Map days_before: last row = D0 (limit-up day), second-to-last = D-1, etc.
                n_rows = len(ind_df)
                for i in range(n_rows):
                    days_before = -(n_rows - 1 - i)  # last row=0, first=-(n-1)
                    row = ind_df.iloc[i]
                    actual_date = str(row.name) if hasattr(row, "name") else kline_df.iloc[i]["date"]

                    conn.execute(
                        """
                        INSERT INTO daily_indicators
                            (event_id, days_before, actual_date,
                             open, high, low, close, volume,
                             ma5, ma10, ma20,
                             macd, macd_signal, macd_hist,
                             rsi,
                             bb_upper, bb_middle, bb_lower, bb_width_pct,
                             k, d, j,
                             vol_ratio, vol_trend,
                             adx, plus_di, minus_di,
                             divergence, macd_cross)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            event_id, days_before, str(actual_date),
                            _safe_float(row.get("open")),
                            _safe_float(row.get("high")),
                            _safe_float(row.get("low")),
                            _safe_float(row.get("close")),
                            _safe_float(row.get("volume")),
                            _safe_float(row.get("ma5")),
                            _safe_float(row.get("ma10")),
                            _safe_float(row.get("ma20")),
                            _safe_float(row.get("macd")),
                            _safe_float(row.get("macd_signal")),
                            _safe_float(row.get("macd_hist")),
                            _safe_float(row.get("rsi")),
                            _safe_float(row.get("bb_upper")),
                            _safe_float(row.get("bb_middle")),
                            _safe_float(row.get("bb_lower")),
                            _safe_float(row.get("bb_width_pct")),
                            _safe_float(row.get("k")),
                            _safe_float(row.get("d")),
                            _safe_float(row.get("j")),
                            _safe_float(row.get("vol_ratio")),
                            row.get("vol_trend") or "",
                            _safe_float(row.get("adx")),
                            _safe_float(row.get("plus_di")),
                            _safe_float(row.get("minus_di")),
                            row.get("divergence") or "",
                            row.get("macd_cross") or "",
                        ),
                    )

                # ── Store patterns ───────────────────────────────────
                for p in patterns:
                    conn.execute(
                        """
                        INSERT INTO event_patterns (event_id, pattern_type, confidence, detail)
                        VALUES (?,?,?,?)
                        """,
                        (event_id, p["pattern_type"], p["confidence"], p.get("detail", "{}")),
                    )

                saved_in_day += 1
            except Exception as e:
                logger.error("Failed to store event %s/%s: %s", d, code, e)
                errors.append({"date": d, "code": code, "error": f"DB write: {e}"})

        # Commit per date and update tracker
        conn.commit()
        set_last_processed_date(conn, d)
        conn.commit()
        total_saved += saved_in_day
        logger.info("  %s: %d/%d events saved", d, saved_in_day, len(day_events))

    return {
        "dates_processed": len(events_by_date),
        "events_saved": total_saved,
        "errors": errors,
    }


def _safe_float(val) -> float | None:
    """Convert a value to float, returning None if NaN."""
    if val is None:
        return None
    try:
        import numpy as np
        if isinstance(val, (np.floating,)):
            f = float(val)
        else:
            f = float(val)
        return f if not np.isnan(f) else None
    except (ValueError, TypeError):
        return None


# ── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="历史涨停股技术面回溯分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --start 2026-01-01 --end 2026-06-26    Full backfill
  %(prog)s --start 2026-06-01                      Single month
  %(prog)s --incremental                           Process new days
  %(prog)s --stats                                 Show summary
  %(prog)s --query "consecutive>=3" --limit 10     Query events
        """,
    )

    parser.add_argument("--start", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="End date (YYYY-MM-DD, default=today)")
    parser.add_argument("--incremental", action="store_true", help="Process new days since last run")
    parser.add_argument("--reprocess", action="store_true", help="Re-process ALL existing events")
    parser.add_argument("--query", help="Query mode: filter string (e.g. 'industry=白酒,consecutive>=2')")
    parser.add_argument("--limit", type=int, default=20, help="Query result limit (default 20)")
    parser.add_argument("--stats", action="store_true", help="Show summary statistics")
    parser.add_argument("--optimize-weights", action="store_true", help="Run weight optimization from backtest data")
    parser.add_argument("--method", default="blended", choices=["lift_only", "logistic_only", "blended", "nn"],
                        help="Weight optimization method (default: blended)")
    parser.add_argument("--nn", action="store_true", help="Enable neural network models in weight optimization")
    parser.add_argument("--control-ratio", type=int, default=5, help="Control:positive sample ratio (default 5)")
    parser.add_argument("--full", action="store_true", help="Full optimization with extra rigor (bootstrap, attention viz)")
    parser.add_argument("--lookback", type=int, default=None, help=f"Override lookback days (default from config/env)")
    parser.add_argument("--workers", type=int, default=None, help="Override max parallel workers")
    parser.add_argument("--delay", type=float, default=0.5, help="Rate limit delay between K-line calls (default 0.5s)")

    args = parser.parse_args()

    # Resolve config overrides
    lookback = args.lookback or int(
        os.environ.get("LIMIT_UP_BACKTEST_LOOKBACK", "90")
    )
    max_workers = args.workers or int(
        os.environ.get("LIMIT_UP_BACKTEST_MAX_WORKERS", "4")
    )

    # ── Initialize DB ──────────────────────────────────────────────
    conn = get_connection(readonly=False)
    ensure_schema(conn)
    conn.close()

    # ── Stats mode ─────────────────────────────────────────────────
    if args.stats:
        stats = get_summary_stats()
        print_summary(stats)
        return

    # ── Query mode ─────────────────────────────────────────────────
    if args.query:
        filters = _parse_query_filter(args.query)
        events = query_events(**filters, limit=args.limit)
        if not events:
            print("No matching events found.")
            return
        print(f"Found {len(events)} events:")
        for e in events:
            print(
                f"  {e['trade_date']} {e['code']} {e['name']} "
                f"连板:{e['consecutive'] or 1} 行业:{e.get('industry','')}"
            )
            if e.get("pre_ma_alignment"):
                print(f"    MA: {e['pre_ma_alignment']}  Score: {e.get('pre_score','')}")
        return

    # ── Weight optimization mode ───────────────────────────────────
    if args.optimize_weights:
        _cmd_optimize_weights(args)
        return

    # ── Reprocess mode ─────────────────────────────────────────────
    if args.reprocess:
        conn = get_connection(readonly=False)
        cur = conn.execute("SELECT COUNT(*) as cnt FROM limit_up_events")
        total = cur.fetchone()["cnt"]
        conn.close()
        if total == 0:
            logger.info("No events to reprocess. Run a backfill first.")
            return
        logger.info("Reprocessing %d existing events...", total)
        # Reprocess by date range covering all existing events
        conn = get_connection(readonly=False)
        cur = conn.execute("SELECT MIN(trade_date) as mn, MAX(trade_date) as mx FROM limit_up_events")
        row = cur.fetchone()
        conn.close()
        start = row["mn"]
        end = row["mx"] or datetime.now().strftime("%Y-%m-%d")
        result = process_date_range(start, end, lookback, max_workers, args.delay)
        _report_result(result)
        return

    # ── Determine date range ───────────────────────────────────────
    if args.incremental:
        conn = get_connection(readonly=False)
        dates = get_incremental_dates(conn)
        conn.close()
        if not dates:
            logger.info("No new dates to process.")
            return
        start = dates[0]
        end = dates[-1]
        logger.info("Incremental mode: %s – %s (%d dates)", start, end, len(dates))
    elif args.start:
        start = args.start
        end = args.end or datetime.now().strftime("%Y-%m-%d")
    else:
        parser.print_help()
        print("\nSpecify --start, --incremental, --stats, --query, or --reprocess")
        return

    result = process_date_range(start, end, lookback, max_workers, args.delay)
    _report_result(result)

    # Print summary
    stats = get_summary_stats()
    print_summary(stats)

    # Cleanup
    clear_kline_cache()
    close_write_connection()


def _cmd_optimize_weights(args):
    """Run the weight optimization pipeline."""
    from lib.limit_up_backtest.control_group import build_training_dataset
    from lib.limit_up_backtest.weights import optimize_weights
    from lib.limit_up_backtest.nn_predictor import train_all_models
    from lib.limit_up_backtest.weight_report import generate_report, save_weight_config, save_report

    lookback = args.lookback or int(os.environ.get("LIMIT_UP_BACKTEST_LOOKBACK", "90"))
    max_workers = args.workers or int(os.environ.get("LIMIT_UP_BACKTEST_MAX_WORKERS", "4"))

    logger.info("=" * 50)
    logger.info("Weight Optimization Pipeline")
    logger.info("  Method: %s", args.method)
    logger.info("  Neural networks: %s", "enabled" if args.nn else "disabled")
    logger.info("  Control ratio: %d:1", args.control_ratio)
    logger.info("=" * 50)

    # Step 1: Build training dataset
    dataset = build_training_dataset(
        control_ratio=args.control_ratio,
        max_workers=max_workers,
        rate_delay=args.delay,
        progress_callback=lambda msg: logger.info("  %s", msg),
    )

    X = dataset["feature_matrix"]
    y = dataset["labels"]
    ts_list = dataset["time_series"]

    logger.info(
        "Training data: %d samples, feature matrix %s",
        len(y), X.shape,
    )

    # Step 2: Train neural networks (if enabled)
    nn_importance = None
    nn_results = None
    if args.nn or args.method == "nn":
        logger.info("Training neural network models...")
        nn_results = train_all_models(
            X_binary=X,
            y=y,
            ts_list=ts_list,
            use_torch=args.nn,
        )
        nn_importance = nn_results.get("ensemble_importance")
        if nn_importance:
            logger.info(
                "NN ensemble importance: %s",
                [f"{v:.3f}" for v in nn_importance],
            )

    # Step 3: Optimize weights
    method = args.method if args.method != "nn" else "blended"
    weight_config = optimize_weights(
        X=X,
        y=y,
        nn_importance=nn_importance,
        method=method,
    )

    # Update sample info
    weight_config["sample_info"]["date_range"] = (
        f"{dataset['binary_samples'][0]['date'] if dataset['binary_samples'] else 'N/A'}"
    )

    # Step 4: Save config
    path = save_weight_config(weight_config)
    logger.info("Weight config saved to %s", path)

    # Step 5: Generate report
    report = generate_report(
        weight_config=weight_config,
        nn_results=nn_results,
    )
    report_path = save_report(report)
    logger.info("Report saved to %s", report_path)

    # Print summary
    print("\n" + report)


def _parse_query_filter(query_str: str) -> dict:
    """Parse a simple filter string like 'industry=白酒,consecutive>=2' into kwargs for query_events."""
    kwargs = {}
    for part in query_str.split(","):
        part = part.strip()
        if ">=" in part:
            k, v = part.split(">=", 1)
            kwargs[k.strip()] = int(v.strip()) if v.strip().isdigit() else v.strip()
        elif "<=" in part:
            k, v = part.split("<=", 1)
            kwargs[k.strip()] = int(v.strip()) if v.strip().isdigit() else v.strip()
        elif "=" in part:
            k, v = part.split("=", 1)
            kwargs[k.strip()] = int(v.strip()) if v.strip().isdigit() else v.strip()
    # Map alias names to query_events param names
    mapping = {
        "industry": "industry",
        "consecutive": "min_consecutive",
        "code": "codes",
        "pattern": "pattern_type",
    }
    result = {}
    for k, v in kwargs.items():
        mapped = mapping.get(k, k)
        if mapped == "codes":
            result[mapped] = [v] if not isinstance(v, list) else v
        elif mapped == "min_consecutive":
            result[mapped] = v
        else:
            result[mapped] = v
    return result


def _report_result(result: dict):
    """Print pipeline result summary."""
    logger.info(
        "Pipeline complete: %d dates, %d events saved, %d errors",
        result["dates_processed"],
        result["events_saved"],
        len(result["errors"]),
    )
    if result["errors"]:
        # Print first 10 errors
        logger.warning("Errors (first 10):")
        for err in result["errors"][:10]:
            logger.warning(
                "  %s %s: %s",
                err.get("date", ""),
                err.get("code", ""),
                err.get("error", ""),
            )


if __name__ == "__main__":
    main()
