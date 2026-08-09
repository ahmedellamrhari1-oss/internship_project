import json
import math
import platform
import time
import urllib.parse
import urllib.request
from io import StringIO
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULT_DIR = Path("results") / "merton_aapl_empirical"
DATA_DIR = RESULT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
FIG_DIR = RESULT_DIR / "figures"

START_DATE = "2016-01-01"
END_DATE = "2025-12-31"
FRED_FETCH_START = "2015-12-01"
WINDOW = 504
TRADING_DAYS = 252
GAMMAS = [1.5, 3.0, 5.0]


def write_json(path: Path, data):
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def ensure_dirs():
    for path in (RESULT_DIR, DATA_DIR, RAW_DIR, FIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def download_text(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_error = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except Exception as exc:
            last_error = exc
            time.sleep(1.0 + attempt)
    raise last_error


def yahoo_period(date_str):
    return int(pd.Timestamp(date_str, tz="UTC").timestamp())


def download_aapl_raw(start=START_DATE, end=END_DATE, force=False):
    raw_path = RAW_DIR / "aapl_yahoo_chart_raw.json"
    if raw_path.exists() and not force:
        return raw_path

    query = urllib.parse.urlencode({
        "period1": yahoo_period(start),
        "period2": yahoo_period(str(pd.Timestamp(end) + pd.Timedelta(days=1))[:10]),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    })
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/AAPL?{query}"
    raw_path.write_text(download_text(url), encoding="utf-8")
    return raw_path


def parse_yahoo_chart(path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    result = raw["chart"]["result"][0]
    timestamps = result["timestamp"]
    adj_close = result["indicators"]["adjclose"][0]["adjclose"]
    parsed = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None).normalize(),
            "adj_close": adj_close,
        }
    )
    return parsed.dropna().sort_values("date")


def download_fred_raw(start=FRED_FETCH_START, end=END_DATE, force=False):
    raw_path = RAW_DIR / "federal_reserve_h15_treasury_constant_maturities_raw.csv"
    if raw_path.exists() and not force:
        return raw_path

    url = (
        "https://www.federalreserve.gov/datadownload/Output.aspx?"
        + urllib.parse.urlencode(
            {
                "rel": "H15",
                "series": "bf17364827e38702b42a58cf8eaa3f78",
                "lastObs": "",
                "from": "",
                "to": "",
                "filetype": "csv",
                "label": "include",
                "layout": "seriescolumn",
                "type": "package",
            }
        )
    )
    raw_path.write_text(download_text(url), encoding="utf-8")
    return raw_path


def load_market_data(force_download=False):
    ensure_dirs()
    aapl_path = download_aapl_raw(force=force_download)
    fred_path = download_fred_raw(force=force_download)

    aapl = parse_yahoo_chart(aapl_path)
    aapl = aapl[(aapl["date"] >= START_DATE) & (aapl["date"] <= END_DATE)].copy()
    aapl["stock_simple_return"] = aapl["adj_close"].pct_change()
    aapl["stock_log_return"] = np.log(aapl["adj_close"]).diff()

    fred_lines = fred_path.read_text(encoding="utf-8").splitlines()
    header_idx = next(i for i, line in enumerate(fred_lines) if line.startswith('"Time Period"') or line.startswith("Time Period"))
    fred = pd.read_csv(StringIO("\n".join(fred_lines[header_idx:])), parse_dates=["Time Period"], na_values=["", "ND", "."])
    fred = fred.rename(columns={"Time Period": "date", "RIFLGFCM03_N.B": "dgs3mo_percent"})
    fred = fred[["date", "dgs3mo_percent"]]
    fred = fred.sort_values("date")
    fred["rf_annual"] = fred["dgs3mo_percent"] / 100.0
    fred[(fred["date"] >= FRED_FETCH_START) & (fred["date"] <= END_DATE)].to_csv(
        DATA_DIR / "risk_free_dgs3mo_daily.csv",
        index=False,
    )
    rf_daily = fred.set_index("date")["rf_annual"].reindex(aapl["date"]).ffill()

    data = aapl.copy()
    data["rf_annual"] = rf_daily.to_numpy()
    data["rf_daily_return"] = np.exp(data["rf_annual"] / TRADING_DAYS) - 1.0
    data = data.dropna(subset=["stock_simple_return", "stock_log_return", "rf_annual", "rf_daily_return"]).reset_index(drop=True)
    data.to_csv(DATA_DIR / "backtest_daily_data.csv", index=False)
    return data, {"aapl_raw": str(aapl_path), "fred_raw": str(fred_path)}


def first_trading_day_each_month(dates):
    frame = pd.DataFrame({"date": pd.to_datetime(dates)})
    frame["month"] = frame["date"].dt.to_period("M")
    return frame.groupby("month", sort=True)["date"].first().to_list()


def build_monthly_estimates(data, window=WINDOW, gammas=GAMMAS):
    data = data.sort_values("date").reset_index(drop=True).copy()
    decision_dates = first_trading_day_each_month(data["date"])
    rows = []
    for decision_date in decision_dates:
        decision_idx = int(data.index[data["date"].eq(decision_date)][0])
        if decision_idx < window:
            continue

        window_returns = data.loc[decision_idx - window:decision_idx - 1, "stock_log_return"]
        sigma_hat = float(window_returns.std(ddof=1) * math.sqrt(TRADING_DAYS))
        mu_hat = float(TRADING_DAYS * window_returns.mean() + 0.5 * sigma_hat ** 2)
        rf_annual = float(data.loc[decision_idx, "rf_annual"])
        common = {
            "decision_date": decision_date,
            "decision_idx": decision_idx,
            "estimation_start": data.loc[decision_idx - window, "date"],
            "estimation_end": data.loc[decision_idx - 1, "date"],
            "window_observations": int(window_returns.shape[0]),
            "mu_hat": mu_hat,
            "sigma_hat": sigma_hat,
            "rf_annual": rf_annual,
        }
        for gamma in gammas:
            w_star = (mu_hat - rf_annual) / (gamma * sigma_hat ** 2)
            rows.append({**common, "gamma": gamma, "w_star": float(w_star), "w_long_only": float(np.clip(w_star, 0.0, 1.0))})
    return pd.DataFrame(rows)


def strategy_specs(estimates):
    specs = {}
    for gamma in GAMMAS:
        gamma_est = estimates[estimates["gamma"].eq(gamma)].copy()
        specs[f"merton_unconstrained_gamma_{gamma:g}"] = gamma_est[["decision_date", "w_star"]].rename(columns={"w_star": "target_weight"})
        specs[f"merton_long_only_gamma_{gamma:g}"] = gamma_est[["decision_date", "w_long_only"]].rename(columns={"w_long_only": "target_weight"})

    decisions = estimates[["decision_date"]].drop_duplicates().sort_values("decision_date")
    specs["benchmark_aapl_100"] = decisions.assign(target_weight=1.0)
    specs["benchmark_50_50_monthly"] = decisions.assign(target_weight=0.5)
    specs["benchmark_cash_100"] = decisions.assign(target_weight=0.0)
    return specs


def simulate_strategy(data, weights, strategy_name, x0=1.0):
    data = data.sort_values("date").reset_index(drop=True).copy()
    weights = weights.sort_values("decision_date").reset_index(drop=True).copy()
    decision_map = {pd.Timestamp(row.decision_date): float(row.target_weight) for row in weights.itertuples(index=False)}

    wealth = x0
    stock_value = 0.0
    cash_value = x0
    target_weight = 0.0
    rows = []
    turnover_rows = []

    for row in data.itertuples(index=False):
        date = pd.Timestamp(row.date)
        if date in decision_map:
            pre_stock_weight = stock_value / wealth if wealth != 0.0 else np.nan
            target_weight = decision_map[date]
            new_stock_value = target_weight * wealth
            new_cash_value = (1.0 - target_weight) * wealth
            turnover = abs(new_stock_value - stock_value) / abs(wealth) if wealth != 0.0 else np.nan
            stock_value = new_stock_value
            cash_value = new_cash_value
            turnover_rows.append(
                {
                    "strategy": strategy_name,
                    "decision_date": date,
                    "target_weight": target_weight,
                    "pre_rebalance_stock_weight": pre_stock_weight,
                    "turnover": turnover,
                }
            )

        prev_wealth = wealth
        stock_value *= 1.0 + float(row.stock_simple_return)
        cash_value *= 1.0 + float(row.rf_daily_return)
        wealth = stock_value + cash_value
        daily_return = wealth / prev_wealth - 1.0 if prev_wealth != 0.0 else np.nan
        exposure = stock_value / wealth if wealth != 0.0 else np.nan
        rows.append(
            {
                "date": date,
                "strategy": strategy_name,
                "wealth": wealth,
                "daily_return": daily_return,
                "target_weight": target_weight,
                "stock_exposure": exposure,
                "rf_annual": float(row.rf_annual),
            }
        )

    return pd.DataFrame(rows), pd.DataFrame(turnover_rows)


def run_backtest(data, estimates):
    first_idx = int(estimates["decision_idx"].min())
    traded = data.loc[first_idx:].reset_index(drop=True).copy()
    all_daily = []
    all_turnover = []
    for name, weights in strategy_specs(estimates).items():
        daily, turnover = simulate_strategy(traded, weights, name)
        all_daily.append(daily)
        all_turnover.append(turnover)
    return pd.concat(all_daily, ignore_index=True), pd.concat(all_turnover, ignore_index=True)


def max_drawdown(wealth):
    wealth = pd.Series(wealth, dtype=float)
    return float((wealth / wealth.cummax() - 1.0).min())


def crra_utility_and_ce(monthly_returns, gamma):
    gross = 1.0 + pd.Series(monthly_returns, dtype=float).dropna()
    if gross.empty or (gross <= 0.0).any():
        return np.nan, np.nan, np.nan
    mean_power = float(np.mean(gross ** (1.0 - gamma)))
    utility = mean_power / (1.0 - gamma)
    ce_monthly_return = mean_power ** (1.0 / (1.0 - gamma)) - 1.0
    ce_annual_return = (1.0 + ce_monthly_return) ** 12 - 1.0
    return float(utility), float(ce_monthly_return), float(ce_annual_return)


def compute_metrics(daily, turnover):
    rows = []
    for strategy, df in daily.groupby("strategy", sort=False):
        df = df.sort_values("date").copy()
        returns = df["daily_return"]
        wealth = df["wealth"]
        n_days = len(df)
        final_wealth = float(wealth.iloc[-1])
        annual_return = final_wealth ** (TRADING_DAYS / n_days) - 1.0 if final_wealth > 0 else np.nan
        annual_vol = float(returns.std(ddof=1) * math.sqrt(TRADING_DAYS))
        rf_daily_mean = float((np.exp(df["rf_annual"] / TRADING_DAYS) - 1.0).mean())
        excess = returns - rf_daily_mean
        sharpe = float(excess.mean() / returns.std(ddof=1) * math.sqrt(TRADING_DAYS)) if returns.std(ddof=1) > 0 else np.nan
        monthly_wealth = wealth.groupby(df["date"].dt.to_period("M")).last()
        monthly_returns = monthly_wealth.pct_change().dropna()
        t = turnover[turnover["strategy"].eq(strategy)].copy()
        turnover_annual = float(t.groupby(t["decision_date"].dt.year)["turnover"].sum().mean()) if not t.empty else 0.0

        row = {
            "strategy": strategy,
            "final_wealth": final_wealth,
            "annualized_return": float(annual_return),
            "annualized_volatility": annual_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown(wealth),
            "mean_monthly_return": float(monthly_returns.mean()),
            "worst_month": float(monthly_returns.min()),
            "average_annual_turnover": turnover_annual,
            "mean_aapl_exposure": float(df["stock_exposure"].mean()),
            "median_aapl_exposure": float(df["stock_exposure"].median()),
            "min_aapl_exposure": float(df["stock_exposure"].min()),
            "max_aapl_exposure": float(df["stock_exposure"].max()),
        }
        for gamma in GAMMAS:
            utility, ce_monthly, ce_annual = crra_utility_and_ce(monthly_returns, gamma)
            row[f"crra_monthly_utility_gamma_{gamma:g}"] = utility
            row[f"certainty_equivalent_monthly_return_gamma_{gamma:g}"] = ce_monthly
            row[f"certainty_equivalent_annual_return_gamma_{gamma:g}"] = ce_annual
        rows.append(row)
    return pd.DataFrame(rows)


def save_weights(estimates):
    rows = []
    for gamma in GAMMAS:
        gamma_est = estimates[estimates["gamma"].eq(gamma)]
        for variant, column in (("merton_unconstrained", "w_star"), ("merton_long_only", "w_long_only")):
            tmp = gamma_est[["decision_date", "gamma", column]].copy()
            tmp = tmp.rename(columns={column: "target_weight"})
            tmp["variant"] = variant
            rows.append(tmp)
    weights = pd.concat(rows, ignore_index=True)
    weights.to_csv(RESULT_DIR / "monthly_weights.csv", index=False)
    return weights


def plot_results(daily, estimates, metrics):
    for path in FIG_DIR.glob("*.png"):
        path.unlink()

    wealth = daily.pivot(index="date", columns="strategy", values="wealth")
    wealth.plot(figsize=(11, 6))
    plt.ylabel("Wealth, X0=1")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "wealth_all_strategies.png", dpi=160)
    plt.close()

    plt.figure(figsize=(11, 5))
    for gamma, df in estimates.groupby("gamma"):
        plt.plot(df["decision_date"], df["w_star"], label=f"gamma={gamma:g}")
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    plt.ylabel("Unconstrained Merton weight")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "merton_unconstrained_weights.png", dpi=160)
    plt.close()

    est_single = estimates[estimates["gamma"].eq(GAMMAS[0])].copy()
    plt.figure(figsize=(11, 5))
    plt.plot(est_single["decision_date"], est_single["mu_hat"], label="mu_hat")
    plt.plot(est_single["decision_date"], est_single["sigma_hat"], label="sigma_hat")
    plt.plot(est_single["decision_date"], est_single["rf_annual"], label="r")
    plt.ylabel("Annualized estimate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "rolling_estimates_mu_sigma_r.png", dpi=160)
    plt.close()

    drawdowns = wealth / wealth.cummax() - 1.0
    drawdowns.plot(figsize=(11, 6))
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "drawdowns.png", dpi=160)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.scatter(metrics["annualized_volatility"], metrics["annualized_return"])
    for row in metrics.itertuples(index=False):
        plt.annotate(row.strategy, (row.annualized_volatility, row.annualized_return), fontsize=7)
    plt.xlabel("Annualized volatility")
    plt.ylabel("Annualized return")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "return_volatility_scatter.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    for gamma, df in estimates.groupby("gamma"):
        plt.hist(df["w_star"], bins=25, alpha=0.45, label=f"gamma={gamma:g}")
    plt.xlabel("Unconstrained Merton weight")
    plt.ylabel("Count")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "merton_weight_distribution.png", dpi=160)
    plt.close()


def make_summary(data, estimates, daily, metrics, raw_paths, runtime):
    unconstrained = estimates.copy()
    out_of_bounds = unconstrained.groupby("gamma")["w_star"].agg(
        fraction_w_lt_0=lambda s: float(np.mean(s < 0.0)),
        fraction_w_gt_1=lambda s: float(np.mean(s > 1.0)),
    )
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": runtime,
        "raw_data": raw_paths,
        "backtest_start": str(daily["date"].min().date()),
        "backtest_end": str(daily["date"].max().date()),
        "n_trading_days": int(data.shape[0]),
        "n_backtest_days": int(daily["date"].nunique()),
        "n_monthly_decisions": int(estimates["decision_date"].nunique()),
        "first_decision_window_observations": int(estimates["window_observations"].iloc[0]),
        "unconstrained_out_of_bounds_by_gamma": out_of_bounds.reset_index().to_dict(orient="records"),
        "best_final_wealth_strategy": metrics.loc[metrics["final_wealth"].idxmax(), "strategy"],
        "best_crra_ce_annual_return_by_gamma": {
            f"gamma_{gamma:g}": metrics.loc[
                metrics[f"certainty_equivalent_annual_return_gamma_{gamma:g}"].idxmax(),
                ["strategy", f"certainty_equivalent_annual_return_gamma_{gamma:g}"],
            ].to_dict()
            for gamma in GAMMAS
        },
    }
    return summary


def main(force_download=False):
    ensure_dirs()
    started = time.perf_counter()
    data, raw_paths = load_market_data(force_download=force_download)
    estimates = build_monthly_estimates(data)
    daily, turnover = run_backtest(data, estimates)
    metrics = compute_metrics(daily, turnover)
    weights = save_weights(estimates)

    data.to_csv(DATA_DIR / "backtest_daily_data.csv", index=False)
    estimates.to_csv(RESULT_DIR / "rolling_estimates.csv", index=False)
    daily.to_csv(RESULT_DIR / "backtest_daily.csv", index=False)
    turnover.to_csv(RESULT_DIR / "turnover_monthly.csv", index=False)
    metrics.to_csv(RESULT_DIR / "metrics.csv", index=False)
    plot_results(daily, estimates, metrics)

    config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).name,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "data": {
            "ticker": "AAPL",
            "price": "Yahoo Finance adjusted close",
            "risk_free": (
                "DGS3MO equivalent from Federal Reserve H.15/RIFLGFCM03_N.B "
                "converted from annual percent to annual decimal and daily simple return exp(r/252)-1"
            ),
            "start_date": START_DATE,
            "end_date": END_DATE,
            "raw_paths": raw_paths,
        },
        "protocol": {
            "log_return_window": WINDOW,
            "annualization_days": TRADING_DAYS,
            "rebalance": "monthly, first trading day of each month",
            "estimation": "strictly previous 504 daily log returns",
            "gammas": GAMMAS,
            "variants": ["merton_unconstrained", "merton_long_only"],
            "benchmarks": ["benchmark_aapl_100", "benchmark_50_50_monthly", "benchmark_cash_100"],
        },
    }
    write_json(RESULT_DIR / "config.json", config)
    summary = make_summary(data, estimates, daily, metrics, raw_paths, time.perf_counter() - started)
    write_json(RESULT_DIR / "summary.json", summary)
    print(json.dumps(summary, indent=2))
    return data, estimates, daily, turnover, weights, metrics, summary


if __name__ == "__main__":
    main()
