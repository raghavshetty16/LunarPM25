"""
Beijing PM2.5 forecasting project - Group 04
Step 3: naive baselines and the exponential smoothing family.

Run 01 and 02 first.

How we evaluate
---------------
Train on 2010 to 2013, test on 2014. The test year holds a full annual cycle,
which matters because a model that only looks good in summer is no use.

The client needs a forecast one to seven days ahead, so that is what we measure.
Every model produces a 1-day-ahead and a 7-day-ahead forecast for each day of
2014, rolling forward one day at a time. Smoothing parameters are estimated once
on the training data and then held fixed, so the model never sees the test year
when choosing them.

A note on the missing days
--------------------------
79 days have no PM2.5 value. The smoothing recursions cannot step over a hole,
so for fitting we interpolate the training series. We never score a forecast
against an interpolated value: the accuracy tables only use days where a real
reading exists.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing

DATA = "final_dataset_daily.csv"
TRAIN_END = "2013-12-31"
HORIZONS = [1, 7]
WEEKLY_PERIOD = 7

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


# ---------------------------------------------------------------- error measures

def error_measures(actual, forecast):
    """The five measures from the course, computed only where both exist."""
    a = np.asarray(actual, dtype=float)
    f = np.asarray(forecast, dtype=float)
    ok = ~np.isnan(a) & ~np.isnan(f)
    a, f = a[ok], f[ok]
    if len(a) == 0:
        return {m: np.nan for m in ["n", "MAD", "MSE", "RMSE", "MAPE", "MPE"]}

    err = a - f
    # Guard the percentage measures against readings at or near zero.
    nonzero = a > 1e-9
    return {
        "n": len(a),
        "ME": np.mean(err),          # plain bias, in µg/m³
        "MAD": np.mean(np.abs(err)),
        "MSE": np.mean(err ** 2),
        "RMSE": np.sqrt(np.mean(err ** 2)),
        "MAPE": np.mean(np.abs(err[nonzero] / a[nonzero])) * 100,
        "MPE": np.mean(err[nonzero] / a[nonzero]) * 100,
    }


# ---------------------------------------------------------------- forecasters
# Each forecaster takes the history available up to and including day t, and
# returns a forecast h steps ahead. Keeping them to one shape makes the rolling
# loop below identical for every model.

def f_naive(history, h, params):
    """Tomorrow looks like today. The floor every other model has to clear."""
    return history[-1]


def f_seasonal_naive(history, h, params):
    """Use the value from the same weekday one week back."""
    period = WEEKLY_PERIOD
    # Step back whole weeks until we land inside the history we actually have.
    offset = period - ((h - 1) % period)
    if offset > len(history):
        return history[-1]
    return history[-offset]


def f_mean(history, h, params):
    """Average of everything so far. Rarely good, useful as a sanity check."""
    return np.mean(history)


def f_ses(history, h, params):
    m = SimpleExpSmoothing(history, initialization_method="estimated")
    r = m.fit(smoothing_level=params["alpha"], optimized=False)
    return r.forecast(h)[-1]


def f_holt(history, h, params):
    m = Holt(history, initialization_method="estimated")
    r = m.fit(smoothing_level=params["alpha"], smoothing_trend=params["beta"],
              optimized=False)
    return r.forecast(h)[-1]


def f_winters(history, h, params):
    m = ExponentialSmoothing(history, trend="add", seasonal="add",
                             seasonal_periods=WEEKLY_PERIOD,
                             initialization_method="estimated")
    r = m.fit(smoothing_level=params["alpha"], smoothing_trend=params["beta"],
              smoothing_seasonal=params["gamma"], optimized=False)
    return r.forecast(h)[-1]


# ---------------------------------------------------------------- fitting

def estimate_parameters(train_values):
    """Let the training data choose the smoothing constants, then freeze them."""
    params = {}

    ses = SimpleExpSmoothing(train_values, initialization_method="estimated").fit()
    params["ses"] = {"alpha": ses.params["smoothing_level"]}

    holt = Holt(train_values, initialization_method="estimated").fit()
    params["holt"] = {"alpha": holt.params["smoothing_level"],
                      "beta": holt.params["smoothing_trend"]}

    hw = ExponentialSmoothing(train_values, trend="add", seasonal="add",
                              seasonal_periods=WEEKLY_PERIOD,
                              initialization_method="estimated").fit()
    params["winters"] = {"alpha": hw.params["smoothing_level"],
                         "beta": hw.params["smoothing_trend"],
                         "gamma": hw.params["smoothing_seasonal"]}

    print("\nSmoothing constants chosen on the training data")
    print(f"  SES      alpha = {params['ses']['alpha']:.3f}")
    print(f"  Holt     alpha = {params['holt']['alpha']:.3f}, "
          f"beta = {params['holt']['beta']:.3f}")
    print(f"  Winters  alpha = {params['winters']['alpha']:.3f}, "
          f"beta = {params['winters']['beta']:.3f}, "
          f"gamma = {params['winters']['gamma']:.3f}")
    print("  A high alpha means the model leans on recent observations and has a "
          "short memory, which is what we would expect from a series this noisy.")
    return params


def rolling_forecast(series_filled, test_index, fn, h, params):
    """Walk through the test period one day at a time, forecasting h ahead."""
    values = series_filled.values
    dates = series_filled.index
    out = pd.Series(index=test_index, dtype=float)

    for target_date in test_index:
        pos = dates.get_loc(target_date)
        origin = pos - h               # the last day we are allowed to know about
        if origin < 10:
            continue
        history = values[:origin + 1]
        try:
            out[target_date] = fn(history, h, params)
        except Exception:
            out[target_date] = np.nan
    return out


# ---------------------------------------------------------------- main

def main():
    df = pd.read_csv(DATA, parse_dates=["date"], index_col="date")
    actual = df["pm25"]

    # Interpolated copy, used only for driving the recursions.
    filled = actual.interpolate(method="time", limit_direction="both")

    train_mask = df.index <= TRAIN_END
    test_index = df.index[~train_mask]
    train_values = filled[train_mask].values

    print("=== Step 3: baselines and smoothing ===")
    print(f"Training: {df.index[train_mask].min().date()} to "
          f"{df.index[train_mask].max().date()} ({train_mask.sum()} days)")
    print(f"Testing:  {test_index.min().date()} to {test_index.max().date()} "
          f"({len(test_index)} days, {actual[test_index].notna().sum()} with a real reading)")

    params = estimate_parameters(train_values)

    models = {
        "Naive": (f_naive, {}),
        "Seasonal naive (weekly)": (f_seasonal_naive, {}),
        "Overall mean": (f_mean, {}),
        "SES": (f_ses, params["ses"]),
        "Holt": (f_holt, params["holt"]),
        "Winters (weekly)": (f_winters, params["winters"]),
    }

    all_forecasts = {}
    rows = []

    for h in HORIZONS:
        print(f"\n--- {h}-day-ahead forecasts for 2014 ---")
        for name, (fn, p) in models.items():
            fc = rolling_forecast(filled, test_index, fn, h, p)
            all_forecasts[(name, h)] = fc
            m = error_measures(actual[test_index], fc)
            m["model"] = name
            m["horizon"] = h
            rows.append(m)
            print(f"  {name:<26} RMSE {m['RMSE']:7.2f}   MAD {m['MAD']:7.2f}   "
                  f"MAPE {m['MAPE']:6.1f}%   MPE {m['MPE']:+6.1f}%")

    results = pd.DataFrame(rows)[["model", "horizon", "n", "ME", "MAD", "MSE",
                                  "RMSE", "MAPE", "MPE"]]
    results.to_csv("tables/accuracy_baselines_smoothing.csv", index=False)

    fc_frame = pd.DataFrame({f"{n} (h={h})": s for (n, h), s in all_forecasts.items()})
    fc_frame.insert(0, "actual", actual[test_index])
    fc_frame.to_csv("tables/forecasts_baselines_smoothing.csv")

    # Which model wins, and by how much over the naive floor?
    print("\nComparison against the naive baseline")
    for h in HORIZONS:
        sub = results[results.horizon == h].set_index("model")
        floor = sub.loc["Naive", "RMSE"]
        best = sub["RMSE"].idxmin()
        print(f"  h={h}: best is {best} at RMSE {sub.loc[best, 'RMSE']:.2f}, "
              f"naive is {floor:.2f} "
              f"({(floor - sub.loc[best, 'RMSE']) / floor:+.1%} change)")

    print("\nBias check")
    print("  MPE is reported because the course asks for it, but it is not "
          "trustworthy on this series: dividing by a reading of 3 µg/m³ turns a")
    print("  small absolute error into a huge percentage, so every model looks "
          "badly biased. Mean error, in the original units, is the honest check.")
    for h in HORIZONS:
        sub = results[results.horizon == h].set_index("model")
        for name in sub.index:
            me = sub.loc[name, "ME"]
            mpe = sub.loc[name, "MPE"]
            verdict = ("under-forecasting" if me > 3 else
                       "over-forecasting" if me < -3 else "roughly unbiased")
            print(f"  h={h}  {name:<26} ME {me:+7.2f} µg/m³  "
                  f"(MPE {mpe:+7.1f}%)  {verdict}")

    # Plot the test year with the two most interesting models.
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(test_index, actual[test_index], lw=0.7, color="#333333", label="actual")
    ax.plot(test_index, all_forecasts[("Winters (weekly)", 7)], lw=0.8,
            color="#c00000", label="Winters, 7 days ahead")
    ax.set_title("2014 test year: actual against the 7-day-ahead Winters forecast")
    ax.set_ylabel("µg/m³")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("figures/07_test_year_forecasts.png")
    plt.close(fig)

    print("\nWritten tables/accuracy_baselines_smoothing.csv, "
          "tables/forecasts_baselines_smoothing.csv, figures/07_test_year_forecasts.png")


if __name__ == "__main__":
    main()
