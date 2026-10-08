"""
Beijing PM2.5 forecasting project - Group 04
Step 4: regression, and the test the whole project rests on.

Run 01 to 03 first.

The question
------------
Does describing the calendar in lunar terms as well as Gregorian terms actually
improve the forecast, or does it just add variables?

We fit the same regression twice. Model A knows only the Gregorian calendar:
trend, month, day of week, heating season. Model B adds the Spring Festival
variables. Everything else is held identical, so any difference in accuracy on
the 2014 holdout belongs to the lunar block and nothing else.

Model C adds the weather variables. It is reported separately because it is an
explanatory model, not a forecasting one: using today's wind speed to explain
today's pollution is fair, but using it to forecast seven days ahead would mean
assuming we already know next week's weather. We say so rather than quietly
taking the credit for it.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "final_dataset_daily.csv"
TRAIN_END = "2013-12-31"

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})


def error_measures(actual, forecast):
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    ok = ~np.isnan(a) & ~np.isnan(f)
    a, f = a[ok], f[ok]
    err = a - f
    nz = a > 1e-9
    return {"n": len(a), "ME": err.mean(), "MAD": np.abs(err).mean(),
            "MSE": (err ** 2).mean(), "RMSE": np.sqrt((err ** 2).mean()),
            "MAPE": np.abs(err[nz] / a[nz]).mean() * 100,
            "MPE": (err[nz] / a[nz]).mean() * 100}


def build_design(df):
    """Assemble the predictor blocks. Each block is kept separate so the models
    below can be built by picking blocks rather than rewriting formulas."""
    X = pd.DataFrame(index=df.index)

    # --- solar block ---
    X["t"] = df["t"] / 1000.0                      # scaled so the coefficient reads sensibly
    month_d = pd.get_dummies(df["month"], prefix="m", drop_first=True).astype(float)
    dow_d = pd.get_dummies(df["day_of_week"], prefix="dow", drop_first=True).astype(float)
    solar = pd.concat([X, month_d, dow_d], axis=1)
    solar["heating"] = df["heating_season"].astype(float)

    # --- lunar block ---
    lunar = pd.DataFrame(index=df.index)
    lunar["cny_eve"] = df["cny_eve"].astype(float)
    lunar["cny_day"] = df["cny_day"].astype(float)
    lunar["pre_festival"] = df["pre_festival"].astype(float)
    lunar["festival_window"] = df["festival_window"].astype(float)
    lunar["lantern"] = df["lantern"].astype(float)

    # --- weather block ---
    weather = df[["temp", "dewp", "pres", "wind_mean", "wind_max",
                  "rain_hours", "snow_hours"]].astype(float)

    return solar, lunar, weather


def fit_and_report(name, X_train, y_train, X_test, y_test, note=""):
    Xc = sm.add_constant(X_train, has_constant="add")
    model = sm.OLS(y_train, Xc, missing="drop").fit()

    dw = durbin_watson(model.resid)
    pred = model.predict(sm.add_constant(X_test, has_constant="add"))
    acc = error_measures(y_test, pred)

    print(f"\n{'=' * 64}")
    print(f"{name}")
    if note:
        print(f"  {note}")
    print(f"{'=' * 64}")
    print(f"  Observations used: {int(model.nobs)}   predictors: {len(X_train.columns)}")
    print(f"  R-squared: {model.rsquared:.3f}   adjusted: {model.rsquared_adj:.3f}")
    print(f"  F-statistic: {model.fvalue:.1f} (p = {model.f_pvalue:.3g})")
    print(f"  Durbin-Watson: {dw:.3f}")
    if dw < 1.5:
        print("    below 1.5, so the residuals are positively autocorrelated. "
              "Today's error predicts tomorrow's.")
        print("    That is a real limitation of a pure regression here and it is "
              "what the ARIMA stage in step 5 is for.")
    elif dw > 2.5:
        print("    above 2.5, which points to negative autocorrelation.")
    else:
        print("    close to 2, so no strong autocorrelation left in the residuals.")
    print(f"  Holdout RMSE on 2014: {acc['RMSE']:.2f}   MAD {acc['MAD']:.2f}   "
          f"ME {acc['ME']:+.2f}")
    return model, pred, acc, dw


def main():
    df = pd.read_csv(DATA, parse_dates=["date"], index_col="date")
    solar, lunar, weather = build_design(df)

    y = df["pm25"]
    train = df.index <= TRAIN_END
    test = ~train
    y_tr, y_te = y[train], y[test]

    print("=== Step 4: regression ===")
    print(f"Training rows: {train.sum()}, test rows: {test.sum()}")
    print(f"Training rows with a real reading: {y_tr.notna().sum()}")

    # Model A: Gregorian calendar only
    A = solar
    mA, pA, aA, dwA = fit_and_report(
        "Model A - Gregorian calendar only",
        A[train], y_tr, A[test], y_te,
        "trend, month dummies, day-of-week dummies, heating season")

    # Model B: add the lunar block
    B = pd.concat([solar, lunar], axis=1)
    mB, pB, aB, dwB = fit_and_report(
        "Model B - Gregorian calendar plus Spring Festival variables",
        B[train], y_tr, B[test], y_te,
        "Model A plus five lunar-calendar variables")

    # Model C: explanatory only
    C = pd.concat([solar, lunar, weather], axis=1)
    mC, pC, aC, dwC = fit_and_report(
        "Model C - calendar plus weather (explanatory, not a forecast)",
        C[train], y_tr, C[test], y_te,
        "assumes next week's weather is already known, so not usable operationally")

    # --- the test that matters -------------------------------------------------
    print(f"\n{'=' * 64}")
    print("Do the lunar variables earn their place?")
    print(f"{'=' * 64}")

    print("\nCoefficients on the lunar block in Model B")
    for v in lunar.columns:
        if v in mB.params.index:
            coef, p, t = mB.params[v], mB.pvalues[v], mB.tvalues[v]
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"  {v:<18} {coef:+8.2f} µg/m³   t = {t:+6.2f}   p = {p:.4f} {stars}")

    # Joint F-test: are all five lunar coefficients zero together?
    restrictions = [f"{v} = 0" for v in lunar.columns if v in mB.params.index]
    ftest = mB.f_test(", ".join(restrictions))
    print(f"\nJoint F-test, all lunar coefficients equal zero:")
    print(f"  F = {float(np.squeeze(ftest.fvalue)):.3f}, "
          f"p = {float(np.squeeze(ftest.pvalue)):.4f}")
    if float(np.squeeze(ftest.pvalue)) < 0.05:
        print("  Rejected: taken together the festival variables carry real "
              "information the Gregorian calendar does not.")
    else:
        print("  Not rejected: taken together the festival variables add nothing "
              "the Gregorian calendar did not already have.")
        print("  This is a negative result and it goes in the report as one.")

    print("\nHoldout accuracy on 2014, A against B")
    print(f"  Model A  RMSE {aA['RMSE']:.2f}   MAD {aA['MAD']:.2f}   "
          f"adj R2 {mA.rsquared_adj:.3f}")
    print(f"  Model B  RMSE {aB['RMSE']:.2f}   MAD {aB['MAD']:.2f}   "
          f"adj R2 {mB.rsquared_adj:.3f}")
    change = (aA["RMSE"] - aB["RMSE"]) / aA["RMSE"]
    print(f"  Change in RMSE from adding the lunar block: {change:+.2%}")
    print(f"  Model C  RMSE {aC['RMSE']:.2f} (with weather, explanatory only, "
          f"adj R2 {mC.rsquared_adj:.3f})")

    # --- save everything -------------------------------------------------------
    summary = pd.DataFrame([
        {"model": "A: solar calendar", **aA, "adj_r2": mA.rsquared_adj, "DW": dwA},
        {"model": "B: solar + lunar", **aB, "adj_r2": mB.rsquared_adj, "DW": dwB},
        {"model": "C: + weather (explanatory)", **aC, "adj_r2": mC.rsquared_adj, "DW": dwC},
    ])
    summary.to_csv("tables/accuracy_regression.csv", index=False)

    pd.DataFrame({"actual": y_te, "model_A": pA, "model_B": pB, "model_C": pC}
                 ).to_csv("tables/forecasts_regression.csv")

    with open("tables/regression_summaries.txt", "w") as fh:
        for nm, m in [("Model A", mA), ("Model B", mB), ("Model C", mC)]:
            fh.write(f"\n\n{'=' * 70}\n{nm}\n{'=' * 70}\n")
            fh.write(str(m.summary()))

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y_te.index, y_te, lw=0.7, color="#333333", label="actual")
    ax.plot(y_te.index, pA, lw=1.0, color="#7f7f7f", label="Model A, solar only")
    ax.plot(y_te.index, pB, lw=1.0, color="#c00000", label="Model B, with festival")
    ax.axvline(pd.Timestamp("2014-01-31"), color="#1f3864", ls="--", lw=0.8)
    ax.set_title("2014 holdout: the two calendar models, with the Spring Festival marked")
    ax.set_ylabel("µg/m³")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("figures/08_regression_holdout.png")
    plt.close(fig)

    print("\nWritten tables/accuracy_regression.csv, tables/forecasts_regression.csv, "
          "tables/regression_summaries.txt, figures/08_regression_holdout.png")


if __name__ == "__main__":
    main()
