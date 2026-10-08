"""
Beijing PM2.5 forecasting project - Group 04
Step 1: turn the raw hourly file into a clean daily series with calendar features.

Run this first. It expects the raw UCI file in the same folder:
    PRSA_data_2010.1.1-2014.12.31.csv

It writes two things:
    final_dataset_daily.csv   the processed data everything else uses
    data_quality_log.txt      what we found and what we did about it
"""

import pandas as pd
import numpy as np
from pathlib import Path

RAW_FILE = "PRSA_data_2010.1.1-2014.12.31.csv"
OUT_DATA = "final_dataset_daily.csv"
OUT_LOG = "data_quality_log.txt"

# A day needs at least this many valid hourly readings before we trust its average.
# 18 of 24 is a judgement call: strict enough that a day built from a handful of
# night-time readings is thrown out, loose enough that we do not lose whole weeks.
MIN_HOURS_PER_DAY = 18

# Gaps up to this many hours get interpolated. Anything longer is left missing,
# because filling a two-day hole is inventing data rather than repairing it.
MAX_GAP_TO_FILL = 6

# First day of the Spring Festival. These are fixed historical dates, so we list
# them rather than compute them. 2009 is included because the first days of our
# series fall after the 2009 festival and we need the "days since" distance.
SPRING_FESTIVAL = {
    2009: "2009-01-26",
    2010: "2010-02-14",
    2011: "2011-02-03",
    2012: "2012-01-23",
    2013: "2013-02-10",
    2014: "2014-01-31",
    2015: "2015-02-19",
}

# Emission controls around the APEC summit. Restrictions ran 3-12 November 2014;
# the summit itself was 5-11 November.
APEC_START = pd.Timestamp("2014-11-03")
APEC_END = pd.Timestamp("2014-11-12")

log_lines = []


def log(msg):
    print(msg)
    log_lines.append(msg)


def load_raw(path):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def describe_missing(df):
    total = len(df)
    missing = df["pm2.5"].isna().sum()
    log(f"Raw file: {total} rows, {df.datetime.min()} to {df.datetime.max()}")
    log(f"Missing pm2.5 readings: {missing} ({missing / total:.1%})")

    # Find how the gaps are distributed. A thousand scattered single hours is a
    # very different problem from a single gap of a thousand hours.
    is_na = df["pm2.5"].isna().values
    runs, start = [], None
    for i, flag in enumerate(is_na):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - start))
            start = None
    if start is not None:
        runs.append((start, len(is_na) - start))

    if runs:
        lengths = [n for _, n in runs]
        longest_at, longest = max(runs, key=lambda r: r[1])
        log(f"Number of separate gaps: {len(runs)}")
        log(f"Median gap length: {int(np.median(lengths))} hours")
        log(f"Longest gap: {longest} hours, starting {df.datetime.iloc[longest_at]}")
        short = sum(1 for n in lengths if n <= MAX_GAP_TO_FILL)
        log(f"Gaps of {MAX_GAP_TO_FILL} hours or less: {short} of {len(runs)}")
    return df


def trim_burn_in(df):
    """The station started reporting after the weather record began, so the file
    opens with a block of NA. Drop everything before the first real reading."""
    first_valid = df["pm2.5"].first_valid_index()
    if first_valid is None:
        raise ValueError("No pm2.5 readings found at all - check the file.")
    dropped = first_valid
    if dropped:
        log(f"Trimmed {dropped} leading rows with no pm2.5 reading "
            f"(series now starts {df.datetime.iloc[first_valid]})")
    return df.iloc[first_valid:].reset_index(drop=True)


def fill_short_gaps(df):
    """Interpolate only across short gaps. Longer ones stay missing."""
    s = df["pm2.5"]
    filled = s.interpolate(method="linear", limit=MAX_GAP_TO_FILL, limit_area="inside")
    n_filled = filled.notna().sum() - s.notna().sum()
    log(f"Interpolated {n_filled} hourly readings across gaps of "
        f"{MAX_GAP_TO_FILL} hours or less")
    log(f"Left missing after interpolation: {filled.isna().sum()} hours")
    df["pm25_filled"] = filled
    return df


def to_daily(df):
    """Collapse to one row per day. Wind speed gets both the mean and the max,
    because a day with one strong clearing gust behaves differently from a day
    of steady moderate wind, and the mean hides that."""
    df["date"] = df["datetime"].dt.normalize()

    grouped = df.groupby("date")
    daily = pd.DataFrame({
        "pm25": grouped["pm25_filled"].mean(),
        "hours_valid": grouped["pm25_filled"].count(),
        "temp": grouped["TEMP"].mean(),
        "dewp": grouped["DEWP"].mean(),
        "pres": grouped["PRES"].mean(),
        "wind_mean": grouped["Iws"].mean(),
        "wind_max": grouped["Iws"].max(),
        "snow_hours": grouped["Is"].sum(),
        "rain_hours": grouped["Ir"].sum(),
    })

    # Most common wind direction of the day.
    daily["wind_dir"] = grouped["cbwd"].agg(
        lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    )

    thin = daily["hours_valid"] < MIN_HOURS_PER_DAY
    log(f"Days built from fewer than {MIN_HOURS_PER_DAY} valid hours: {thin.sum()} "
        f"of {len(daily)} - their pm25 is set to missing")
    daily.loc[thin, "pm25"] = np.nan

    # Make sure every calendar day is present, even if it has no reading, so the
    # series stays evenly spaced. Every method we use depends on that.
    full_index = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_index)
    daily.index.name = "date"
    log(f"Daily series: {len(daily)} days, {daily.index.min().date()} to "
        f"{daily.index.max().date()}, {daily.pm25.isna().sum()} days without a value")
    return daily


def add_calendar_features(daily):
    idx = daily.index

    daily["t"] = np.arange(1, len(daily) + 1)
    daily["year"] = idx.year
    daily["month"] = idx.month
    daily["day_of_week"] = idx.dayofweek          # Monday = 0
    daily["is_weekend"] = (idx.dayofweek >= 5).astype(int)

    # Beijing's official heating season runs 15 November to 15 March. Coal-fired
    # heating is the main reason winter pollution is worse, so this deserves its
    # own flag rather than being left for the month dummies to pick up.
    daily["heating_season"] = (
        ((idx.month == 11) & (idx.day >= 15))
        | idx.month.isin([12, 1, 2])
        | ((idx.month == 3) & (idx.day <= 15))
    ).astype(int)

    # --- the lunar coordinate ---
    festivals = pd.to_datetime(pd.Series(list(SPRING_FESTIVAL.values())))

    # For each day, how far is it from the nearest Spring Festival? Negative means
    # the festival is still ahead, positive means it has passed. This is the second
    # time axis the whole project is built on.
    days_to = []
    for d in idx:
        diffs = (d - festivals).dt.days
        nearest = diffs.iloc[diffs.abs().idxmin()]
        days_to.append(nearest)
    daily["lunar_day"] = days_to

    daily["cny_eve"] = (daily["lunar_day"] == -1).astype(int)
    daily["cny_day"] = (daily["lunar_day"] == 0).astype(int)
    daily["pre_festival"] = daily["lunar_day"].between(-7, -2).astype(int)
    daily["festival_window"] = daily["lunar_day"].between(0, 14).astype(int)
    daily["lantern"] = (daily["lunar_day"] == 14).astype(int)

    # APEC emission controls, used later as a natural experiment.
    daily["apec"] = ((idx >= APEC_START) & (idx <= APEC_END)).astype(int)

    log(f"Spring Festival days flagged: {int(daily.cny_day.sum())}")
    log(f"Days inside a festival window: {int(daily.festival_window.sum())}")
    log(f"Days under APEC controls: {int(daily.apec.sum())}")
    return daily


def main():
    if not Path(RAW_FILE).exists():
        raise SystemExit(
            f"Could not find {RAW_FILE}. Download it from "
            "https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data "
            "and put it in this folder."
        )

    log("=== Step 1: preparing the data ===\n")
    df = load_raw(RAW_FILE)
    df = describe_missing(df)
    df = trim_burn_in(df)
    df = fill_short_gaps(df)

    daily = to_daily(df)
    daily = add_calendar_features(daily)

    daily.to_csv(OUT_DATA)
    log(f"\nWritten {OUT_DATA} with {len(daily)} rows and {daily.shape[1]} columns")

    Path(OUT_LOG).write_text("\n".join(log_lines), encoding="utf-8")
    print(f"Written {OUT_LOG}")


if __name__ == "__main__":
    main()
