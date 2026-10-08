"""
Beijing PM2.5 forecasting project - Group 04
Step 5: Box-Jenkins.

Run 01 to 04 first.

Why this step exists
--------------------
Every regression in step 4 came back with a Durbin-Watson around 0.93. The
residuals are strongly positively autocorrelated, which means two things. The
forecasts are leaving information unused, and the t-statistics on those
coefficients are not trustworthy, because ordinary least squares assumes
independent errors and we do not have them.

So we do two things here. First, identify an ARIMA model for the series on its
own. Second, re-run the lunar-variable test as an ARIMAX, where the error
structure is modelled properly and the standard errors mean what they claim to.
That second part is the honest version of the test we ran in step 4.

We also check whether the prediction intervals are any good, by counting how
often the actual value for 2014 fell inside them. An interval that claims 95%
coverage and delivers 70% is worse than useless to a client making decisions.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox

warnings.filterwarnings("ignore")

DATA = "final_dataset_daily.csv"
TRAIN_END = "2013-12-31"
HORIZONS = [1, 7]

plt.rcParams.update({"figure.dpi": 130, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False})

LUNAR_VARS = ["cny_eve", "cny_day", "pre_festival", "festival_window", "lantern"]

# Candidates to compare. Kept small and justified by the ACF and PACF from step 2
# rather than searched blindly over a large grid.
CANDIDATES = [
    ((1, 0, 0), (0, 0, 0, 0), "AR(1)"),
    ((2, 0, 0), (0, 0, 0, 0), "AR(2)"),
    ((0, 0, 1), (0, 0, 0, 0), "MA(1)"),
    ((1, 0, 1), (0, 0, 0, 0), "ARMA(1,1)"),
    ((2, 0, 2), (0, 0, 0, 0), "ARMA(2,2)"),
    ((1, 1, 1), (0, 0, 0, 0), "ARIMA(1,1,1)"),
    ((1, 0, 1), (1, 0, 1, 7), "ARMA(1,1) with weekly seasonal"),
]



def trend_for(order, seasonal):
    """SARIMAX leaves the constant out by default. On an undifferenced series
    with a mean near 99 that is badly wrong: with no constant the model is
    pulled towards zero, and if exogenous variables are present they quietly
    soak up the mean instead of measuring what we asked them to measure.
    So we include a constant whenever the series is not differenced."""
    differenced = order[1] > 0 or (len(seasonal) == 4 and seasonal[1] > 0)
    return "n" if differenced else "c"


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


def identify(y_train):
    """Fit each candidate and rank by AIC. Also check what the residuals look
    like, because a model with the best AIC and autocorrelated residuals has not
    finished the job."""
    rows = []
    fitted = {}
    print("\nCandidate models, fitted on 2010 to 2013")
    print(f"  {'model':<34}{'AIC':>10}{'BIC':>10}{'LB p-value':>13}")
    for order, seasonal, label in CANDIDATES:
        try:
            res = SARIMAX(y_train, order=order, seasonal_order=seasonal,
                          trend=trend_for(order, seasonal),
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)
            lb = acorr_ljungbox(res.resid[2:], lags=[14], return_df=True)
            lb_p = float(lb["lb_pvalue"].iloc[0])
            rows.append({"model": label, "order": str(order),
                         "seasonal": str(seasonal), "AIC": res.aic,
                         "BIC": res.bic, "ljungbox_p": lb_p})
            fitted[label] = res
            print(f"  {label:<34}{res.aic:>10.1f}{res.bic:>10.1f}{lb_p:>13.4f}")
        except Exception as exc:
            print(f"  {label:<34} failed: {exc}")

    table = pd.DataFrame(rows).sort_values("AIC")
    table.to_csv("tables/arima_identification.csv", index=False)
    best_label = table.iloc[0]["model"]
    print(f"\n  Lowest AIC: {best_label}")
    print("  A Ljung-Box p-value above 0.05 means the residuals look like white "
          "noise, which is what we want.")
    return best_label, fitted, table


def rolling_arima(y_full, train_end, order, seasonal, h, exog_full=None):
    """Fit once on the training data, then walk forward one day at a time using
    append, which updates the state without re-estimating the parameters. This
    is both faster and more honest than refitting on data the forecast origin
    should not have seen."""
    train_mask = y_full.index <= train_end
    y_tr = y_full[train_mask]
    exog_tr = exog_full[train_mask] if exog_full is not None else None

    res = SARIMAX(y_tr, exog=exog_tr, order=order, seasonal_order=seasonal,
                  trend=trend_for(order, seasonal),
                  enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False)

    test_index = y_full.index[~train_mask]
    point = pd.Series(index=test_index, dtype=float)
    lo80 = pd.Series(index=test_index, dtype=float)
    hi80 = pd.Series(index=test_index, dtype=float)
    lo95 = pd.Series(index=test_index, dtype=float)
    hi95 = pd.Series(index=test_index, dtype=float)

    state = res
    appended = 0

    for i, target in enumerate(test_index):
        # The forecast origin is h days before the target.
        need = i - h + 1          # how many test observations the origin may use
        while appended < need:
            d = test_index[appended]
            new_y = y_full.loc[[d]]
            new_x = exog_full.loc[[d]] if exog_full is not None else None
            state = state.append(new_y, exog=new_x, refit=False)
            appended += 1
        if need < 0:
            continue

        fx = exog_full.loc[test_index[appended:appended + h]] if exog_full is not None else None
        fc = state.get_forecast(steps=h, exog=fx)
        point[target] = fc.predicted_mean.iloc[-1]
        ci80 = fc.conf_int(alpha=0.20)
        ci95 = fc.conf_int(alpha=0.05)
        lo80[target], hi80[target] = ci80.iloc[-1, 0], ci80.iloc[-1, 1]
        lo95[target], hi95[target] = ci95.iloc[-1, 0], ci95.iloc[-1, 1]

    return res, point, (lo80, hi80, lo95, hi95)


def coverage(actual, lo, hi, label):
    ok = actual.notna() & lo.notna() & hi.notna()
    inside = ((actual[ok] >= lo[ok]) & (actual[ok] <= hi[ok])).mean()
    width = (hi[ok] - lo[ok]).mean()
    print(f"  {label}: {inside:.1%} of actual values fell inside, "
          f"average width {width:.0f} µg/m³")
    return inside, width


def main():
    df = pd.read_csv(DATA, parse_dates=["date"], index_col="date")
    y = df["pm25"].interpolate(method="time", limit_direction="both")
    y.index.freq = "D"
    actual = df["pm25"]

    train_mask = df.index <= TRAIN_END
    test_index = df.index[~train_mask]

    print("=== Step 5: Box-Jenkins ===")
    print(f"Training on {train_mask.sum()} days, testing on {len(test_index)}")
    print("Note: the recursions need a complete series, so the 79 missing days are")
    print("interpolated for fitting. Accuracy is still scored only against real readings.")

    best_label, fitted, table = identify(y[train_mask])
    order, seasonal, _ = next(c for c in CANDIDATES if c[2] == best_label)

    rows = []
    stored = {}

    for h in HORIZONS:
        print(f"\n--- {best_label}, {h}-day-ahead forecasts for 2014 ---")
        res, point, bands = rolling_arima(y, TRAIN_END, order, seasonal, h)
        acc = error_measures(actual[test_index], point)
        acc.update({"model": f"{best_label}", "horizon": h})
        rows.append(acc)
        stored[("arima", h)] = (point, bands)
        print(f"  RMSE {acc['RMSE']:.2f}   MAD {acc['MAD']:.2f}   ME {acc['ME']:+.2f}")

        print("  Prediction interval coverage")
        coverage(actual[test_index], bands[0], bands[1], "    nominal 80%")
        coverage(actual[test_index], bands[2], bands[3], "    nominal 95%")

    # ---------------------------------------------------------------- ARIMAX
    print(f"\n{'=' * 64}")
    print("The lunar test again, this time with the autocorrelation modelled")
    print(f"{'=' * 64}")
    print("In step 4 the Durbin-Watson was 0.93, so those standard errors were")
    print("unreliable. Here the error term is an ARMA process, so the t-statistics")
    print("on the festival variables can be read at face value.")

    exog = df[LUNAR_VARS].astype(float)
    res_x = SARIMAX(y[train_mask], exog=exog[train_mask], order=order,
                    seasonal_order=seasonal, trend=trend_for(order, seasonal),
                    enforce_stationarity=False,
                    enforce_invertibility=False).fit(disp=False)

    print("\nFestival coefficients with ARMA errors")
    for v in LUNAR_VARS:
        c, p, t = res_x.params[v], res_x.pvalues[v], res_x.tvalues[v]
        stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
        print(f"  {v:<18}{c:+8.2f} µg/m³   t = {t:+6.2f}   p = {p:.4f} {stars}")

    print(f"\n  AIC without festival variables: {fitted[best_label].aic:.1f}")
    print(f"  AIC with festival variables:    {res_x.aic:.1f}")
    better = "lower" if res_x.aic < fitted[best_label].aic else "higher"
    print(f"  Adding them makes AIC {better}, which is the cleanest single verdict "
          f"on whether they belong.")

    for h in HORIZONS:
        _, point_x, bands_x = rolling_arima(y, TRAIN_END, order, seasonal, h,
                                            exog_full=exog)
        acc = error_measures(actual[test_index], point_x)
        acc.update({"model": f"{best_label} + festival (ARIMAX)", "horizon": h})
        rows.append(acc)
        stored[("arimax", h)] = (point_x, bands_x)
        base = next(r for r in rows if r["model"] == best_label and r["horizon"] == h)
        change = (base["RMSE"] - acc["RMSE"]) / base["RMSE"]
        print(f"\n  h={h}: ARIMAX RMSE {acc['RMSE']:.2f} against plain "
              f"{base['RMSE']:.2f} ({change:+.2%})")

    results = pd.DataFrame(rows)[["model", "horizon", "n", "ME", "MAD", "MSE",
                                 "RMSE", "MAPE", "MPE"]]
    results.to_csv("tables/accuracy_arima.csv", index=False)

    out = {"actual": actual[test_index]}
    for (tag, h), (pt, bands) in stored.items():
        out[f"{tag}_h{h}"] = pt
        if tag == "arima":
            out[f"{tag}_h{h}_lo80"], out[f"{tag}_h{h}_hi80"] = bands[0], bands[1]
            out[f"{tag}_h{h}_lo95"], out[f"{tag}_h{h}_hi95"] = bands[2], bands[3]
    pd.DataFrame(out).to_csv("tables/forecasts_arima.csv")

    # Plot the 7-day-ahead forecast with its interval.
    pt, bands = stored[("arima", 7)]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(test_index, bands[2], bands[3], color="#c00000", alpha=0.15,
                    label="95% prediction interval")
    ax.plot(test_index, actual[test_index], lw=0.7, color="#333333", label="actual")
    ax.plot(test_index, pt, lw=0.9, color="#c00000", label=f"{best_label}, 7 days ahead")
    ax.set_title("2014 holdout: 7-day-ahead ARIMA forecast with prediction interval")
    ax.set_ylabel("µg/m³")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("figures/09_arima_intervals.png")
    plt.close(fig)

    print("\nWritten tables/arima_identification.csv, tables/accuracy_arima.csv, "
          "tables/forecasts_arima.csv, figures/09_arima_intervals.png")


if __name__ == "__main__":
    main()
