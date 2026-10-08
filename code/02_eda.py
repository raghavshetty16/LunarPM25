"""
Beijing PM2.5 forecasting project - Group 04
Step 2: exploratory analysis.

Run 01_prepare_data.py first. This reads final_dataset_daily.csv and produces the
figures and tables the EDA section of the report is built on.

The part that matters most is the seasonal index table. Instead of computing one
set of monthly indices for the whole series, we compute them separately for each
year. If a month's index is stable from year to year, whatever drives it is tied
to the solar calendar. If it jumps around, something is moving in and out of that
month. That is the test our whole argument rests on.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

DATA = "final_dataset_daily.csv"
FIG_DIR = "figures"
TAB_DIR = "tables"

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def setup():
    import os
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    df = pd.read_csv(DATA, parse_dates=["date"], index_col="date")
    return df


def plot_series(df):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    axes[0].plot(df.index, df["pm25"], lw=0.4, color="#333333")
    axes[0].set_title("Daily mean PM2.5, Beijing, 2010 to 2014")
    axes[0].set_ylabel("µg/m³")

    monthly = df["pm25"].resample("ME").mean()
    axes[1].plot(monthly.index, monthly.values, marker="o", ms=3, color="#1f3864")
    axes[1].set_title("Monthly mean PM2.5")
    axes[1].set_ylabel("µg/m³")

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/01_series_overview.png")
    plt.close(fig)
    return monthly


def plot_monthly_spread(df):
    """How much does each month vary from year to year? This is the first visual
    hint of the thing the seasonal index table tests properly."""
    data = [df.loc[df.month == m, "pm25"].dropna().values for m in range(1, 13)]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot(data, showfliers=False)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_NAMES)
    ax.set_title("Distribution of daily PM2.5 by month, all years pooled")
    ax.set_ylabel("µg/m³")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/02_monthly_spread.png")
    plt.close(fig)


def seasonal_indices_by_year(monthly):
    """Ratio-to-moving-average, the method from the decomposition session, but
    kept separate by year instead of averaged across all of them."""
    s = monthly.copy()
    # Centred 12-month moving average: average of two 12-month averages, which is
    # how you centre an even-order MA.
    ma = s.rolling(12, center=True).mean().rolling(2, center=True).mean().shift(-1)
    ratio = s / ma

    tab = pd.DataFrame({
        "year": ratio.index.year,
        "month": ratio.index.month,
        "ratio": ratio.values,
    }).dropna()

    pivot = tab.pivot(index="year", columns="month", values="ratio")
    pivot.columns = [MONTH_NAMES[m - 1] for m in pivot.columns]

    summary = pd.DataFrame({
        "mean_index": pivot.mean(),
        "std_across_years": pivot.std(),
        "min": pivot.min(),
        "max": pivot.max(),
    })
    summary["range"] = summary["max"] - summary["min"]

    pivot.round(3).to_csv(f"{TAB_DIR}/seasonal_index_by_year.csv")
    summary.round(3).to_csv(f"{TAB_DIR}/seasonal_index_stability.csv")

    print("\nSeasonal index by year (ratio to centred moving average)")
    print(pivot.round(2).to_string())
    print("\nStability of each month's index across years")
    print(summary.round(3).to_string())

    ranked = summary["std_across_years"].sort_values(ascending=False)
    print("\nMonths ranked by how unstable their index is:")
    for m, v in ranked.items():
        print(f"  {m}: sd = {v:.3f}")

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(summary.index, summary["std_across_years"], color="#1f3864")
    ax.set_title("Year-to-year variability of each month's seasonal index")
    ax.set_ylabel("standard deviation across years")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/03_seasonal_index_stability.png")
    plt.close(fig)

    return pivot, summary


def festival_profile(df):
    """Average PM2.5 by distance from the Spring Festival. If the festival has an
    effect, it shows up here as a clear shape rather than a flat line."""
    window = df[df["lunar_day"].between(-30, 45)]
    profile = window.groupby("lunar_day")["pm25"].agg(["mean", "count"])
    profile.to_csv(f"{TAB_DIR}/festival_profile.csv")

    baseline = df["pm25"].mean()

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(profile.index, profile["mean"], color="#333333", lw=1.2)
    ax.axvline(0, color="#c00000", lw=1, ls="--")
    ax.axhline(baseline, color="#7f7f7f", lw=0.8, ls=":")
    ax.annotate("Festival day", xy=(0, ax.get_ylim()[1] * 0.95),
                xytext=(3, ax.get_ylim()[1] * 0.95), fontsize=8, color="#c00000")
    ax.set_xlabel("days from the first day of the Spring Festival")
    ax.set_ylabel("mean PM2.5 (µg/m³)")
    ax.set_title("PM2.5 in lunar-day coordinates, averaged over 2010 to 2014")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/04_festival_profile.png")
    plt.close(fig)

    print("\nFestival window against the overall mean")
    print(f"  Overall daily mean: {baseline:.1f}")
    for label, lo, hi in [("Week before (-7 to -2)", -7, -2),
                          ("Eve and festival day (-1 to 0)", -1, 0),
                          ("Festival window (0 to 14)", 0, 14),
                          ("After the window (15 to 30)", 15, 30)]:
        sel = df.loc[df["lunar_day"].between(lo, hi), "pm25"].mean()
        print(f"  {label}: {sel:.1f} ({sel - baseline:+.1f} vs overall)")

    return profile


def stationarity_and_autocorrelation(df):
    s = df["pm25"].dropna()

    adf = adfuller(s, autolag="AIC")
    print("\nAugmented Dickey-Fuller test on the daily series")
    print(f"  statistic: {adf[0]:.3f}")
    print(f"  p-value:   {adf[1]:.4f}")
    for k, v in adf[4].items():
        print(f"  critical value {k}: {v:.3f}")
    print("  reading: p below 0.05 means we can reject a unit root")

    lb = acorr_ljungbox(s, lags=[7, 14, 30], return_df=True)
    print("\nLjung-Box test (is there structure, or is this noise?)")
    print(lb.round(4).to_string())

    a = acf(s, nlags=60)
    p = pacf(s, nlags=60)
    conf = 1.96 / np.sqrt(len(s))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, vals, name in [(axes[0], a, "ACF"), (axes[1], p, "PACF")]:
        ax.bar(range(len(vals)), vals, width=0.6, color="#1f3864")
        ax.axhline(conf, color="#c00000", lw=0.7, ls="--")
        ax.axhline(-conf, color="#c00000", lw=0.7, ls="--")
        ax.set_title(f"{name}, daily PM2.5")
        ax.set_xlabel("lag (days)")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/05_acf_pacf.png")
    plt.close(fig)

    pd.DataFrame({"lag": range(len(a)), "acf": a, "pacf": p}).round(4).to_csv(
        f"{TAB_DIR}/acf_pacf.csv", index=False)


def distribution_check(df):
    s = df["pm25"].dropna()
    print("\nShape of the distribution")
    print(f"  mean {s.mean():.1f}, median {s.median():.1f}, sd {s.std():.1f}")
    print(f"  skew {s.skew():.2f}, max {s.max():.0f}")
    if s.skew() > 0.5:
        print("  positive skew and a mean above the median point to a long right "
              "tail, so a log transform is worth testing before differencing")
    else:
        print("  skew is mild here, so a transform may not be needed - check the "
              "histogram before deciding")

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].hist(s, bins=60, color="#1f3864")
    axes[0].set_title("Daily PM2.5")
    axes[1].hist(np.log(s[s > 0]), bins=60, color="#1f3864")
    axes[1].set_title("Log of daily PM2.5")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/06_distribution.png")
    plt.close(fig)


def main():
    df = setup()
    print("=== Step 2: exploratory analysis ===")
    print(f"Loaded {len(df)} daily rows, {df.index.min().date()} to {df.index.max().date()}")

    monthly = plot_series(df)
    plot_monthly_spread(df)
    seasonal_indices_by_year(monthly)
    festival_profile(df)
    stationarity_and_autocorrelation(df)
    distribution_check(df)

    print(f"\nFigures written to {FIG_DIR}/, tables to {TAB_DIR}/")


if __name__ == "__main__":
    main()
