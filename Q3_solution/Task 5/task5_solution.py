"""
Nomura Quant Challenge 5 - Question 3, Task 5
Dynamic Quoting Under Inventory Pressure (LP / Market Maker perspective).

Submission functions:
    quote(inventory, sigma, alpha, eta) -> (delta_bid, delta_ask)
    validate_quote(...) -> None

Important convention:
    quote() returns RELATIVE half-spreads, i.e. fractions of mid price.
    Example: delta = 0.0003 means 3 bps of mid.

LP side convention from problem statement:
    Side = +1  -> LP buys  -> client sells to us -> our BID is hit
    Side = -1  -> LP sells -> client buys from us -> our ASK is hit

Inventory convention:
    inventory > 0 -> LP is long  -> widen bid, tighten ask
    inventory < 0 -> LP is short -> tighten bid, widen ask
"""

import math
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Strategy constants
# -----------------------------------------------------------------------------
C_BASE = 1.00
K_ALPHA = 1.00
INV_SCALE = 400.0
K_SKEW = 0.60
K_ETA = 2.00
INVENTORY_DEADZONE = 75.0

CMIN = 0.5
DELTA_MAX = 0.005       # 50 bps of mid, relative units
SIGMA_FLOOR = 1e-6

MID_COLS = ["M5", "M10", "M15", "M20", "M25", "M30"]
HORIZONS = [5, 10, 15, 20, 25, 30]


# -----------------------------------------------------------------------------
# Core quoting function required by Task 5
# -----------------------------------------------------------------------------
def quote(inventory: float, sigma: float, alpha: float, eta: float) -> tuple[float, float]:
    """
    Returns (delta_bid, delta_ask) as RELATIVE half-spreads.

    Inputs:
        inventory: current inventory. Positive = long, negative = short.
        sigma: realized volatility in relative return units.
        alpha: adversity score in [0, 1]. Higher = more adverse for LP.
        eta: elapsed fraction of day in [0, 1].

    Output:
        delta_bid: distance of bid below mid, relative to mid.
        delta_ask: distance of ask above mid, relative to mid.

    The function is standalone: it reads no files and loads no models.
    """
    # Defensive input cleaning for hidden evaluator robustness.
    inventory = float(np.nan_to_num(inventory, nan=0.0, posinf=0.0, neginf=0.0))
    sigma = float(np.nan_to_num(sigma, nan=SIGMA_FLOOR, posinf=SIGMA_FLOOR, neginf=SIGMA_FLOOR))
    alpha = float(np.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0))
    eta = float(np.nan_to_num(eta, nan=0.0, posinf=1.0, neginf=0.0))

    sigma_eff = max(sigma, SIGMA_FLOOR)
    alpha = min(max(alpha, 0.0), 1.0)
    eta = min(max(eta, 0.0), 1.0)

    # Volatility-scaled base quote.
    base = C_BASE * sigma_eff

    # Smooth one-sided adversity widening.
    # Alpha below 0.5 is treated as normal/safe flow; alpha above 0.5 widens quotes.
    alpha_excess = max(alpha - 0.5, 0.0)
    risk_mult = 1.0 + K_ALPHA * alpha_excess
    half_spread = base * risk_mult

    # Inventory deadzone: do not overreact to tiny/noisy inventory.
    # Once outside the deadzone, use tanh so skew remains bounded.
    abs_inventory = abs(inventory)
    if abs_inventory <= INVENTORY_DEADZONE:
        inventory_pressure = 0.0
    else:
        adjusted_inventory = math.copysign(abs_inventory - INVENTORY_DEADZONE, inventory)
        inventory_pressure = math.tanh(adjusted_inventory / INV_SCALE)

    # Time pressure: skew grows mainly near close, because end-of-day penalty matters then.
    time_pressure = 1.0 + K_ETA * (eta ** 2)
    skew = K_SKEW * sigma_eff * inventory_pressure * time_pressure

    # Correct skew direction:
    # Long inventory -> skew positive -> delta_bid up, delta_ask down.
    # Short inventory -> skew negative -> delta_bid down, delta_ask up.
    delta_bid_raw = half_spread + skew
    delta_ask_raw = half_spread - skew

    # Constraints. Floor first, cap last. If 0.5*sigma exceeds cap in extreme
    # volatility, cap wins as the final contest-safe hard maximum.
    lower = CMIN * sigma_eff
    upper = DELTA_MAX

    delta_bid = min(max(delta_bid_raw, lower), upper)
    delta_ask = min(max(delta_ask_raw, lower), upper)

    # Final safety guard.
    delta_bid = float(np.nan_to_num(delta_bid, nan=upper, posinf=upper, neginf=0.0))
    delta_ask = float(np.nan_to_num(delta_ask, nan=upper, posinf=upper, neginf=0.0))

    return max(delta_bid, 0.0), max(delta_ask, 0.0)


# -----------------------------------------------------------------------------
# Baselines for validation only
# -----------------------------------------------------------------------------
def quote_fixed(inventory: float, sigma: float, alpha: float, eta: float) -> tuple[float, float]:
    """Fixed-spread baseline: 3 bps relative half-spread, clipped."""
    sigma_eff = max(float(np.nan_to_num(sigma, nan=SIGMA_FLOOR)), SIGMA_FLOOR)
    lower = CMIN * sigma_eff
    d = min(max(0.0003, lower), DELTA_MAX)
    return float(d), float(d)


def quote_vol_alpha(inventory: float, sigma: float, alpha: float, eta: float) -> tuple[float, float]:
    """Symmetric volatility + alpha baseline, without inventory skew."""
    sigma_eff = max(float(np.nan_to_num(sigma, nan=SIGMA_FLOOR)), SIGMA_FLOOR)
    alpha = float(np.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0))
    alpha = min(max(alpha, 0.0), 1.0)

    half = C_BASE * sigma_eff * (1.0 + K_ALPHA * max(alpha - 0.5, 0.0))
    d = min(max(half, CMIN * sigma_eff), DELTA_MAX)
    return float(d), float(d)


# -----------------------------------------------------------------------------
# Data preparation and alpha source for validation
# -----------------------------------------------------------------------------
def _prepare_trade_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    required_cols = ["Date", "time", "Name", "Side", "Volume", "Trade Price", "M0", "Spread"] + MID_COLS
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in trade data: {missing}")

    # Robust timestamp parsing. Works for ISO dates like 2025-03-03 and
    # day-first dates like 03-03-2025.
    df["timestamp"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["time"].astype(str),
        format="mixed",
        dayfirst=True,
        errors="raise",
    )

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["_secs"] = (
        df["timestamp"].dt.hour * 3600
        + df["timestamp"].dt.minute * 60
        + df["timestamp"].dt.second
    )

    return df


def _chronological_split(df: pd.DataFrame, split: str = "test") -> pd.DataFrame:
    """
    Split by unique chronological dates.

    train: first 60% dates
    validation: next 20% dates
    test: latest 20% dates
    all: full data
    """
    split = split.lower()
    if split == "all":
        return df.copy()

    unique_dates = sorted(df["Date"].unique())
    n_dates = len(unique_dates)
    train_end = int(0.60 * n_dates)
    val_end = int(0.80 * n_dates)

    if split == "train":
        chosen = unique_dates[:train_end]
    elif split in {"validation", "val"}:
        chosen = unique_dates[train_end:val_end]
    elif split == "test":
        chosen = unique_dates[val_end:]
    else:
        raise ValueError("split must be one of: train, validation, test, all")

    return df[df["Date"].isin(chosen)].copy().reset_index(drop=True)


def _make_alpha_fn(use_task3_alpha: bool = False) -> Callable[[dict], float]:
    """
    Validation alpha source.

    Default is a lightweight proxy, because Task 5's hidden evaluator passes alpha
    directly into quote(). validate_quote() is only a proxy backtest.

    If use_task3_alpha=True and task3_solution.py is available, we attempt to call
    predict_adversity with a full row dictionary. If it fails, fallback proxy is used.
    """

    def proxy_alpha(row: dict) -> float:
        m0 = float(row.get("M0", 0.0))
        spread = float(row.get("Spread", 0.0))
        rel_spread = spread / m0 if m0 > 0 else 0.0

        # Typical full spread is around a few bps. Wider book is treated as a
        # weak proxy for more adverse/uncertain flow. This uses only current data.
        score = rel_spread * 1e4 - 3.0
        alpha = 1.0 / (1.0 + math.exp(-score))
        return float(min(max(alpha, 0.0), 1.0))

    if not use_task3_alpha:
        print("[alpha] Using lightweight spread-based proxy alpha.")
        return proxy_alpha

    try:
        from task3_solution import predict_adversity  # type: ignore

        def alpha_fn(row: dict) -> float:
            try:
                row_for_model = dict(row)
                row_for_model["tau"] = 20
                a = float(predict_adversity(row_for_model))
                if not np.isfinite(a):
                    return proxy_alpha(row)
                return float(min(max(a, 0.0), 1.0))
            except Exception:
                return proxy_alpha(row)

        print("[alpha] Using Task 3 predict_adversity() with fallback proxy.")
        return alpha_fn

    except Exception:
        print("[alpha] Task 3 unavailable. Using lightweight spread-based proxy alpha.")
        return proxy_alpha


# -----------------------------------------------------------------------------
# Backtest helpers
# -----------------------------------------------------------------------------
def _compute_sigma_from_history(mids: np.ndarray, i: int) -> float:
    """Realized sigma using only previous M0 values within the same day."""
    if i >= 21:
        window = mids[i - 21:i]  # 21 prices -> 20 returns
    elif i >= 2:
        window = mids[:i]
    else:
        return SIGMA_FLOOR

    if len(window) < 2:
        return SIGMA_FLOOR

    prev = window[:-1]
    nxt = window[1:]
    safe_prev = np.where(prev == 0.0, np.nan, prev)
    returns = (nxt - prev) / safe_prev
    returns = returns[np.isfinite(returns)]

    if len(returns) == 0:
        return SIGMA_FLOOR

    return float(max(np.sqrt(np.mean(returns ** 2)), SIGMA_FLOOR))


def _max_drawdown(cumulative_pnl: np.ndarray) -> float:
    if len(cumulative_pnl) == 0:
        return 0.0
    running_peak = np.maximum.accumulate(cumulative_pnl)
    return float(np.max(running_peak - cumulative_pnl))


def _run_strategy(
    df: pd.DataFrame,
    quote_fn: Callable[[float, float, float, float], tuple[float, float]],
    alpha_fn: Callable[[dict], float],
    lam: float,
    gamma: float,
    phi: float,
    seed: int = 42,
) -> Dict[str, float]:
    """Run one strategy under one hidden-parameter scenario."""
    rng = np.random.default_rng(seed)

    daily_gross = []
    daily_penalty = []
    daily_net = []
    end_inv_abs = []
    all_abs_inventory = []

    total_fills = 0
    total_trades = 0
    sum_delta_bid = 0.0
    sum_delta_ask = 0.0
    quote_count = 0

    for _, day in df.groupby("Date", sort=False):
        day = day.reset_index(drop=True)

        mids = day["M0"].to_numpy(dtype=float)
        sides = day["Side"].to_numpy(dtype=int)
        volumes = day["Volume"].to_numpy(dtype=float)
        future_mids = day[MID_COLS].to_numpy(dtype=float).mean(axis=1)
        seconds = day["_secs"].to_numpy(dtype=float)

        row_dicts = day.to_dict("records")

        t_open = float(seconds[0])
        t_close = float(seconds[-1])
        span = max(t_close - t_open, 1.0)

        inventory = 0.0
        gross = 0.0

        for i in range(len(day)):
            side = int(sides[i])
            volume = float(volumes[i])
            m0 = float(mids[i])

            sigma = _compute_sigma_from_history(mids, i)
            eta = float(min(max((seconds[i] - t_open) / span, 0.0), 1.0))
            alpha = alpha_fn(row_dicts[i])

            delta_bid, delta_ask = quote_fn(inventory, sigma, alpha, eta)

            sum_delta_bid += delta_bid
            sum_delta_ask += delta_ask
            quote_count += 1

            # Side=+1: LP buys -> bid hit. Side=-1: LP sells -> ask hit.
            delta_side = delta_bid if side == 1 else delta_ask
            delta_abs = delta_side * m0

            # Fill model. Use relative delta/sigma because quote() returns relative delta.
            p_fill = lam * math.exp(-gamma * delta_side / max(sigma, SIGMA_FLOOR))
            p_fill = float(min(max(p_fill, 0.0), 1.0))

            total_trades += 1
            if rng.random() < p_fill:
                total_fills += 1
                trade_price = m0 - side * delta_abs
                pnl = side * volume * (future_mids[i] - trade_price)
                gross += pnl
                inventory += side * volume

            all_abs_inventory.append(abs(inventory))

        # Daily inventory penalty.
        if len(mids) > 1:
            day_returns = np.diff(mids) / mids[:-1]
            day_returns = day_returns[np.isfinite(day_returns)]
            sigma_day = float(np.sqrt(np.mean(day_returns ** 2))) if len(day_returns) else SIGMA_FLOOR
        else:
            sigma_day = SIGMA_FLOOR

        penalty = phi * (inventory ** 2) * sigma_day
        net = gross - penalty

        daily_gross.append(gross)
        daily_penalty.append(penalty)
        daily_net.append(net)
        end_inv_abs.append(abs(inventory))

    daily_net_arr = np.array(daily_net, dtype=float)
    cumulative = np.cumsum(daily_net_arr)

    daily_std = float(np.std(daily_net_arr)) if len(daily_net_arr) else 0.0
    total_net = float(np.sum(daily_net_arr))

    return {
        "total_gross_pnl": float(np.sum(daily_gross)),
        "total_penalty": float(np.sum(daily_penalty)),
        "total_net_pnl": total_net,
        "daily_mean_pnl": float(np.mean(daily_net_arr)) if len(daily_net_arr) else 0.0,
        "daily_std_pnl": daily_std,
        "sharpe_like_score": float(total_net / max(daily_std, 1.0)),
        "max_drawdown": _max_drawdown(cumulative),
        "fill_rate": float(total_fills / total_trades) if total_trades else 0.0,
        "average_abs_inventory": float(np.mean(all_abs_inventory)) if all_abs_inventory else 0.0,
        "average_end_inventory_abs": float(np.mean(end_inv_abs)) if end_inv_abs else 0.0,
        "average_delta_bid": float(sum_delta_bid / quote_count) if quote_count else 0.0,
        "average_delta_ask": float(sum_delta_ask / quote_count) if quote_count else 0.0,
    }


# -----------------------------------------------------------------------------
# Required validation function
# -----------------------------------------------------------------------------
def validate_quote(
    data_path: str = "trade_data.csv",
    split: str = "test",
    use_task3_alpha: bool = False,
) -> None:
    """
    Runs proxy validation/backtesting of the quoting strategy.

    Default split='test' uses the latest 20% dates as a held-out proxy test.
    Saved output: task5_validation_summary.csv
    """
    df_all = _prepare_trade_data(data_path)
    df = _chronological_split(df_all, split=split)

    if df.empty:
        raise ValueError(f"No rows available for split={split}")

    alpha_fn = _make_alpha_fn(use_task3_alpha=use_task3_alpha)

    # Hidden parameters are not given, so validate robustness across scenarios.
    scenarios = [
        (0.6, 3.0, 1e-5),
        (0.3, 6.0, 1e-4),
        (0.9, 1.0, 1e-6),
    ]

    strategies = {
        "fixed": quote_fixed,
        "vol_alpha": quote_vol_alpha,
        "inventory_skew": quote,
    }

    rows = []
    for strategy_name, strategy_fn in strategies.items():
        scenario_results = []
        for scenario_index, (lam, gamma, phi) in enumerate(scenarios):
            result = _run_strategy(
                df=df,
                quote_fn=strategy_fn,
                alpha_fn=alpha_fn,
                lam=lam,
                gamma=gamma,
                phi=phi,
                seed=42 + scenario_index,
            )
            scenario_results.append(result)

        averaged = {
            key: float(np.mean([r[key] for r in scenario_results]))
            for key in scenario_results[0]
        }
        averaged["strategy"] = strategy_name
        averaged["split"] = split
        rows.append(averaged)

        print(f"\n=== {strategy_name} | split={split} ===")
        for key in [
            "total_net_pnl",
            "sharpe_like_score",
            "max_drawdown",
            "fill_rate",
            "average_abs_inventory",
            "average_end_inventory_abs",
        ]:
            print(f"{key:30s}: {averaged[key]:.6f}")

    summary = pd.DataFrame(rows)
    ordered_cols = [
        "strategy",
        "split",
        "total_gross_pnl",
        "total_penalty",
        "total_net_pnl",
        "daily_mean_pnl",
        "daily_std_pnl",
        "sharpe_like_score",
        "max_drawdown",
        "fill_rate",
        "average_abs_inventory",
        "average_end_inventory_abs",
        "average_delta_bid",
        "average_delta_ask",
    ]
    summary = summary[ordered_cols]
    summary.to_csv("task5_validation_summary.csv", index=False)

    print("\nSaved: task5_validation_summary.csv")


if __name__ == "__main__":
    validate_quote()
