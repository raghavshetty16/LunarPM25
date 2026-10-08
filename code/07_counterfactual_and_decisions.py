"""
Beijing PM2.5 forecasting project - Group 04
Step 7: what the forecasts are actually worth.

Run 01 to 06 first. Two separate pieces of work here.

Part A: policy counterfactuals
------------------------------
Beijing ran two natural experiments inside our data. Every year the Spring
Festival shuts much of the city down for about two weeks. In November 2014,
ahead of the APEC summit, the city went much further: factories and power plants
halted, construction stopped, odd-even number plate restrictions came in, the
winter heating season was delayed by a week, and a public holiday was declared.
Controls ran from 3 to 12 November.

So we can ask what the pollution would have been without the controls. Train the
model on everything up to the day before, forecast through the window, and
compare with what actually happened.

The obvious objection is weather. If that fortnight simply happened to be windy,
the drop proves nothing. So we do it twice: once with a model that knows only the
history of the series, and once with a model that also sees the weather that
actually occurred. The second number is the part that cannot be explained by
meteorology, and it is the one worth quoting.

Part B: evaluating forecasts as decisions
-----------------------------------------
RMSE treats every day equally. The client does not. They issue an advisory or
they do not, and the two mistakes cost different amounts: a missed episode harms
public health, a false alarm costs credibility and some disruption. So we score
every model a second way, as a yes-or-no call on whether tomorrow crosses an
action threshold, and attach a cost to each kind of error.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")

DATA = "final_dataset_daily.csv"
ORDER = (1, 0, 1)
SEASONAL = (1, 0, 1, 7)
WEATHER = ["temp", "dewp", "pres", "wind_mean", "wind_max", "rain_hours", "snow_hours"]

APEC_START, APEC_END = "2014-11-03", "2014-11-12"
SPRING_FESTIVALS = {2011: "2011-02-03", 2012: "2012-01-23",
                    2013: "2013-02-10", 2014: "2014-01-31"}

# The action threshold. 150 µg/m³ of PM2.5 is the point at which Chinese AQI
# moves into the unhealthy band and advisories become reasonable.
THRESHOLD = 150.0
# A missed episode is assumed to cost five times a false alarm. This is a
# judgement, not a measurement, so the report shows how the answer changes if
# the ratio is 2 or 10 instead.
COST_RATIOS = [2, 5, 10]

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


# ------------------------------------------------------------------ Part A

def counterfactual(y, exog, start, end, label, use_weather):
    """Train on everything before `start`, then forecast the window. When
    use_weather is on, the forecast is fed the weather that actually occurred,
    so anything left over is not a weather story."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    train = y[y.index < start]
    window = pd.date_range(start, end, freq="D")

    ex_tr = exog.loc[train.index] if use_weather else None
    ex_fc = exog.loc[window] if use_weather else None

    res = SARIMAX(train, exog=ex_tr, order=ORDER, seasonal_order=SEASONAL,
                  trend="c", enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False)

    fc = res.get_forecast(steps=len(window), exog=ex_fc)
    expected = fc.predicted_mean
    expected.index = window
    ci = fc.conf_int(alpha=0.20)
    ci.index = window

    observed = y.reindex(window)
    gap = observed.mean() - expected.mean()

    kind = "weather-adjusted" if use_weather else "history only"
    print(f"\n  {label} ({kind})")
    print(f"    expected without the controls: {expected.mean():7.1f} µg/m³")
    print(f"    actually observed:             {observed.mean():7.1f} µg/m³")
    print(f"    difference:                    {gap:+7.1f} µg/m³ "
          f"({gap / expected.mean():+.1%})")
    below = (observed < ci.iloc[:, 0]).sum()
    print(f"    days below the 80% lower bound: {below} of {len(window)}")

    return pd.DataFrame({"observed": observed, "expected": expected,
                         "lo80": ci.iloc[:, 0], "hi80": ci.iloc[:, 1]})


def part_a(df):
    print("=" * 68)
    print("Part A: what did the shutdowns actually buy?")
    print("=" * 68)

    y = df["pm25"].interpolate(method="time", limit_direction="both")
    y.index.freq = "D"
    exog = df[WEATHER].astype(float).interpolate(limit_direction="both")

    print("\nAPEC, November 2014")
    apec_plain = counterfactual(y, exog, APEC_START, APEC_END, "APEC window", False)
    apec_adj = counterfactual(y, exog, APEC_START, APEC_END, "APEC window", True)

    apec_plain.to_csv("tables/counterfactual_apec_history_only.csv")
    apec_adj.to_csv("tables/counterfactual_apec_weather_adjusted.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    w = apec_adj.index
    ax.fill_between(w, apec_adj["lo80"], apec_adj["hi80"], color="#7f7f7f",
                    alpha=0.2, label="80% interval, no-controls forecast")
    ax.plot(w, apec_adj["expected"], color="#7f7f7f", lw=1.2,
            label="expected without controls")
    ax.plot(w, apec_adj["observed"], color="#c00000", lw=1.6, marker="o", ms=3,
            label="observed")
    ax.set_title("APEC window, 3 to 12 November 2014 (weather-adjusted)")
    ax.set_ylabel("µg/m³")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig("figures/11_apec_counterfactual.png")
    plt.close(fig)

    print("\nSpring Festival windows, day 0 to day 14")
    rows = []
    for year, date in SPRING_FESTIVALS.items():
        start = pd.Timestamp(date)
        end = start + pd.Timedelta(days=14)
        res = counterfactual(y, exog, start, end, f"Spring Festival {year}", True)
        rows.append({"year": year,
                     "expected": res["expected"].mean(),
                     "observed": res["observed"].mean(),
                     "gap": res["observed"].mean() - res["expected"].mean()})
    fest = pd.DataFrame(rows)
    fest.to_csv("tables/counterfactual_spring_festivals.csv", index=False)

    print("\n  Across the four festivals")
    print(f"    mean gap: {fest['gap'].mean():+.1f} µg/m³")
    print(f"    range:    {fest['gap'].min():+.1f} to {fest['gap'].max():+.1f}")
    neg = (fest["gap"] < 0).sum()
    print(f"    cleaner than expected in {neg} of {len(fest)} years")
    if neg < len(fest):
        print("    Not every year goes the same way, which is worth saying plainly "
              "rather than averaging away.")


# ------------------------------------------------------------------ Part B

def confusion(actual, forecast, threshold):
    a = actual >= threshold
    f = forecast >= threshold
    ok = actual.notna() & forecast.notna()
    a, f = a[ok], f[ok]
    return {"hits": int((a & f).sum()),
            "misses": int((a & ~f).sum()),
            "false_alarms": int((~a & f).sum()),
            "correct_quiet": int((~a & ~f).sum()),
            "episodes": int(a.sum()), "days": int(len(a))}


def part_b(df):
    print("\n" + "=" * 68)
    print("Part B: the same models judged as decisions, not as numbers")
    print("=" * 68)

    actual = df["pm25"]

    arima = pd.read_csv("tables/forecasts_arima.csv", parse_dates=["date"],
                        index_col="date")
    reg = pd.read_csv("tables/forecasts_regression.csv", parse_dates=[0], index_col=0)
    base = pd.read_csv("tables/forecasts_baselines_smoothing.csv",
                       parse_dates=[0], index_col=0)

    test = arima.index
    truth = actual.reindex(test)
    episodes = int((truth >= THRESHOLD).sum())
    print(f"\nAction threshold: {THRESHOLD:.0f} µg/m³")
    print(f"Days in 2014 above it: {episodes} of {truth.notna().sum()} "
          f"({episodes / truth.notna().sum():.1%})")

    candidates = {
        "Naive (h=1)": base["Naive (h=1)"],
        "ARIMA (h=1)": arima["arima_h1"],
        "ARIMA (h=7)": arima["arima_h7"],
        "Calendar regression": reg["model_B"],
    }

    # A rule that uses the interval rather than the point forecast: issue an
    # advisory when the upper bound crosses the threshold. It will cry wolf more
    # often, and catch more real episodes.
    for lvl in ["80", "95"]:
        for h in ["1", "7"]:
            col = f"arima_h{h}_hi{lvl}"
            if col in arima.columns:
                candidates[f"ARIMA h={h}, upper {lvl}% rule"] = arima[col]

    rows = []
    for name, fc in candidates.items():
        fc = fc.reindex(test)
        cm = confusion(truth, fc, THRESHOLD)
        rmse = np.sqrt(np.nanmean((truth - fc) ** 2))
        row = {"model": name, "RMSE": rmse, **cm}
        row["evaluable_days"] = cm["days"]
        for ratio in COST_RATIOS:
            row[f"cost_{ratio}to1"] = cm["misses"] * ratio + cm["false_alarms"]
        rows.append(row)

    table = pd.DataFrame(rows)
    table["hit_rate"] = table["hits"] / table["episodes"]
    table["false_alarm_rate"] = table["false_alarms"] / (
        table["days"] - table["episodes"])
    table.to_csv("tables/decision_evaluation.csv", index=False)

    print("\nHow each model performs as an alert system")
    print(f"  {'model':<28}{'RMSE':>8}{'hits':>7}{'miss':>7}{'false':>7}"
          f"{'hit rate':>10}")
    for _, r in table.iterrows():
        print(f"  {r['model']:<28}{r['RMSE']:>8.1f}{r['hits']:>7}{r['misses']:>7}"
              f"{r['false_alarms']:>7}{r['hit_rate']:>9.0%}")

    print("\nExpected cost, under three views of how bad a miss is")
    print(f"  {'model':<28}{'2:1':>8}{'5:1':>8}{'10:1':>8}")
    for _, r in table.iterrows():
        print(f"  {r['model']:<28}{r['cost_2to1']:>8.0f}"
              f"{r['cost_5to1']:>8.0f}{r['cost_10to1']:>8.0f}")

    print("\nDo the two ways of judging agree?")
    by_rmse = table.sort_values("RMSE")["model"].tolist()
    for ratio in COST_RATIOS:
        by_cost = table.sort_values(f"cost_{ratio}to1")["model"].tolist()
        agree = "yes" if by_rmse[0] == by_cost[0] else "no"
        print(f"  at {ratio}:1, best by cost is {by_cost[0]!r}; "
              f"best by RMSE is {by_rmse[0]!r} -> agree: {agree}")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(table["false_alarm_rate"], table["hit_rate"], color="#1f3864")
    for _, r in table.iterrows():
        ax.annotate(r["model"], (r["false_alarm_rate"], r["hit_rate"]),
                    fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.set_xlabel("false alarm rate")
    ax.set_ylabel("hit rate")
    ax.set_title(f"Catching episodes above {THRESHOLD:.0f} µg/m³ in 2014")
    ax.set_xlim(-0.02, max(0.5, table["false_alarm_rate"].max() * 1.3))
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig("figures/12_decision_tradeoff.png")
    plt.close(fig)


def main():
    df = pd.read_csv(DATA, parse_dates=["date"], index_col="date")
    part_a(df)
    part_b(df)
    print("\nWritten counterfactual and decision tables, plus figures 11 and 12.")


if __name__ == "__main__":
    main()
