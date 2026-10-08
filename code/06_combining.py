"""
Beijing PM2.5 forecasting project - Group 04
Step 6: combining forecasts.

Run 01 to 05 first.

This follows the procedure from the combining forecasts session, in the same
order and with the same tests:

  1. check each forecast is unbiased, because combining does not remove bias
  2. correlate the squared errors, to see whether the two models actually know
     different things
  3. sweep the weight from 0.05 to 0.95 and plot what RMSE does
  4. regress actuals on the two forecasts WITH an intercept, and test that the
     intercept is zero
  5. if that passes, regress again WITHOUT an intercept; those coefficients are
     the weights, and they should sum to roughly one and both be positive

The two forecasts we combine are chosen deliberately rather than by convenience.
The ARIMA model knows about persistence: what pollution did yesterday and the day
before. The calendar regression knows about the time of year and the festival,
and crucially its predictions do not degrade with horizon, because the calendar
seven days from now is already known. Those are different kinds of information,
which is exactly the condition the session says combining needs.

One caveat we state rather than hide: the session estimates the weights on the
same sample it evaluates them on. We report that version for comparability, and
then repeat it properly, estimating weights on the first half of 2014 and
testing on the second half.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

SPLIT_DATE = "2014-06-30"


def error_measures(a, f):
    a, f = np.asarray(a, float), np.asarray(f, float)
    ok = ~np.isnan(a) & ~np.isnan(f)
    a, f = a[ok], f[ok]
    err = a - f
    nz = a > 1e-9
    return {"n": len(a), "ME": err.mean(), "MAD": np.abs(err).mean(),
            "RMSE": np.sqrt((err ** 2).mean()),
            "MAPE": np.abs(err[nz] / a[nz]).mean() * 100,
            "MPE": (err[nz] / a[nz]).mean() * 100}


def load_forecasts(h):
    arima = pd.read_csv("tables/forecasts_arima.csv", parse_dates=["date"],
                        index_col="date")
    reg = pd.read_csv("tables/forecasts_regression.csv", parse_dates=[0],
                      index_col=0)
    reg.index.name = "date"

    df = pd.DataFrame({
        "actual": arima["actual"],
        "F1_arima": arima[f"arima_h{h}"],
        "F2_calendar": reg["model_B"],
    })
    return df.dropna()


def step1_bias(df):
    print("\nStep 1: is either forecast biased?")
    print("  Combining cannot fix a model that consistently runs high or low, so")
    print("  this has to be checked before anything else.")
    ok = True
    for col in ["F1_arima", "F2_calendar"]:
        m = error_measures(df["actual"], df[col])
        err = df["actual"] - df[col]
        t, p = stats.ttest_1samp(err.dropna(), 0)
        verdict = "no evidence of bias" if p > 0.05 else "biased"
        print(f"  {col:<14} mean error {m['ME']:+7.2f} µg/m³   "
              f"t = {t:+5.2f}, p = {p:.3f}   {verdict}")
        if p <= 0.05:
            ok = False
    if ok:
        print("  Both pass, so we can carry on.")
    else:
        print("  At least one forecast is biased. The session says to fix that "
              "first; we report the combination anyway but flag it here.")
    return ok


def step2_information(df):
    """His sheet computes squared errors for each model, correlates them, and
    runs a hypothesis test on that correlation."""
    print("\nStep 2: do the two models know different things?")
    e1 = (df["actual"] - df["F1_arima"]) ** 2
    e2 = (df["actual"] - df["F2_calendar"]) ** 2
    r = np.corrcoef(e1, e2)[0, 1]
    n = len(e1)
    se = np.sqrt((1 - r ** 2) / (n - 2))
    t = r / se
    p = 2 * (1 - stats.t.cdf(abs(t), df=n - 2))

    print(f"  correlation between the squared errors: r = {r:.4f}")
    print(f"  SE(r) = {se:.4f},  t = {t:.3f},  df = {n - 2},  p = {p:.4f}")
    if p > 0.05:
        print("  Not significantly different from zero. The errors are unrelated,")
        print("  so each model is contributing something the other misses. This is")
        print("  the condition the session asks for.")
    else:
        print("  Significantly correlated. The two models fail in the same places,")
        print("  so combining them has less to offer than it would otherwise.")
    return r, t, p


def step3_weight_sweep(df, label):
    print("\nStep 3: what does RMSE do as the weight changes?")
    rows = []
    for w in np.round(np.arange(0.05, 1.0, 0.05), 2):
        comb = w * df["F1_arima"] + (1 - w) * df["F2_calendar"]
        rows.append({"weight_on_arima": w,
                     "RMSE": error_measures(df["actual"], comb)["RMSE"]})
    sweep = pd.DataFrame(rows)
    best = sweep.loc[sweep["RMSE"].idxmin()]

    r1 = error_measures(df["actual"], df["F1_arima"])["RMSE"]
    r2 = error_measures(df["actual"], df["F2_calendar"])["RMSE"]
    print(f"  ARIMA alone:            RMSE {r1:.2f}")
    print(f"  Calendar alone:         RMSE {r2:.2f}")
    print(f"  Best simple weighting:  w = {best['weight_on_arima']:.2f} on ARIMA, "
          f"RMSE {best['RMSE']:.2f}")
    gain = (min(r1, r2) - best["RMSE"]) / min(r1, r2)
    print(f"  Improvement over the better single model: {gain:+.2%}")

    sweep.to_csv(f"tables/combination_weight_sweep_{label}.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(sweep["weight_on_arima"], sweep["RMSE"], marker="o", ms=3,
            color="#1f3864")
    ax.axhline(r1, color="#c00000", ls="--", lw=0.8, label="ARIMA alone")
    ax.axhline(r2, color="#7f7f7f", ls=":", lw=0.8, label="calendar alone")
    ax.set_xlabel("weight on the ARIMA forecast")
    ax.set_ylabel("RMSE")
    ax.set_title(f"Combined forecast RMSE against the weight ({label})")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"figures/10_weight_sweep_{label}.png")
    plt.close(fig)
    return sweep, best


def step4_5_regression_weights(df):
    print("\nStep 4: regression with an intercept, to test for bias in the mix")
    X = df[["F1_arima", "F2_calendar"]]
    with_c = sm.OLS(df["actual"], sm.add_constant(X)).fit()
    c, p_c = with_c.params["const"], with_c.pvalues["const"]
    print(f"  intercept = {c:+.3f}, p = {p_c:.4f}")
    if p_c > 0.05:
        print("  We fail to reject the null that the intercept is zero, which is")
        print("  what we want. Proceed to the regression without an intercept.")
    else:
        print("  The intercept is significantly different from zero, which points")
        print("  to bias in the combination. We report it and carry on, but this")
        print("  is a caveat on the weights below.")

    print("\nStep 5: regression without an intercept gives the weights")
    no_c = sm.OLS(df["actual"], X).fit()
    b1, b2 = no_c.params["F1_arima"], no_c.params["F2_calendar"]
    print(f"  b1 (ARIMA)    = {b1:+.4f}   t = {no_c.tvalues['F1_arima']:+.2f}   "
          f"p = {no_c.pvalues['F1_arima']:.4f}")
    print(f"  b2 (calendar) = {b2:+.4f}   t = {no_c.tvalues['F2_calendar']:+.2f}   "
          f"p = {no_c.pvalues['F2_calendar']:.4f}")
    print(f"  b1 + b2 = {b1 + b2:.4f}")
    if abs(b1 + b2 - 1) < 0.1:
        print("  Close to one, as expected when both inputs are unbiased.")
    else:
        print("  Some distance from one, which usually means one of the inputs is "
              "biased after all.")
    if b1 > 0 and b2 > 0:
        print("  Both weights are positive, so the session's advice is satisfied.")
    else:
        print("  One weight is negative. The session says interpretation becomes")
        print("  tenuous here and advises using the method only when both are")
        print("  positive, so we treat this result with caution.")

    with open("tables/combination_regressions.txt", "w") as fh:
        fh.write("WITH INTERCEPT\n")
        fh.write(str(with_c.summary()))
        fh.write("\n\nWITHOUT INTERCEPT\n")
        fh.write(str(no_c.summary()))

    return with_c, no_c, b1, b2


def main():
    print("=== Step 6: combining forecasts ===")

    for h in [1, 7]:
        print(f"\n{'#' * 68}")
        print(f"# Horizon: {h} day(s) ahead")
        print(f"{'#' * 68}")

        df = load_forecasts(h)
        print(f"Days available for this comparison: {len(df)}")

        step1_bias(df)
        step2_information(df)
        sweep, best = step3_weight_sweep(df, f"h{h}")
        with_c, no_c, b1, b2 = step4_5_regression_weights(df)

        # The combination the regression actually proposes
        comb_reg = b1 * df["F1_arima"] + b2 * df["F2_calendar"]
        comb_simple = (best["weight_on_arima"] * df["F1_arima"]
                       + (1 - best["weight_on_arima"]) * df["F2_calendar"])

        print("\nResults for this horizon, all measured on the same days")
        rows = []
        for name, series in [("ARIMA alone", df["F1_arima"]),
                             ("Calendar regression alone", df["F2_calendar"]),
                             (f"Combined, swept weight {best['weight_on_arima']:.2f}",
                              comb_simple),
                             ("Combined, regression weights", comb_reg)]:
            m = error_measures(df["actual"], series)
            m["model"] = name
            m["horizon"] = h
            rows.append(m)
            print(f"  {name:<40} RMSE {m['RMSE']:7.2f}   MAD {m['MAD']:6.2f}   "
                  f"ME {m['ME']:+6.2f}")

        pd.DataFrame(rows)[["model", "horizon", "n", "ME", "MAD", "RMSE",
                            "MAPE", "MPE"]].to_csv(
            f"tables/accuracy_combination_h{h}.csv", index=False)

        # --- the honest version: estimate the weights out of sample -----------
        print("\nSame thing again, but with the weights estimated out of sample")
        print("  Weights from January to June 2014, tested on July to December.")
        first = df[df.index <= SPLIT_DATE]
        second = df[df.index > SPLIT_DATE]
        if len(first) > 30 and len(second) > 30:
            fit = sm.OLS(first["actual"], first[["F1_arima", "F2_calendar"]]).fit()
            w1, w2 = fit.params["F1_arima"], fit.params["F2_calendar"]
            oos = w1 * second["F1_arima"] + w2 * second["F2_calendar"]
            print(f"  weights from the first half: ARIMA {w1:+.3f}, "
                  f"calendar {w2:+.3f}")
            for name, series in [("ARIMA alone", second["F1_arima"]),
                                 ("Calendar alone", second["F2_calendar"]),
                                 ("Combined", oos)]:
                m = error_measures(second["actual"], series)
                print(f"    {name:<22} RMSE on the second half: {m['RMSE']:.2f}")

    print("\nWritten combination tables and figures.")


if __name__ == "__main__":
    main()
