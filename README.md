# Group 04 — Time Series Analysis Project

**Forecasting Air Quality in Beijing**

| S. No. | Name | Roll Number |
|---|---|---|
| 1 | Nishkarsh Khandelwal | 27PGAI0051 |
| 2 | Jagath Ponnanna P M | 27PGAI0072 |
| 3 | Raghavendra Prasad Shetti | 27PGAI0052 |
| 4 | Chirag Parida | 27PGAI0055 |

## What is in this folder

```
04LunarPM25_Report.docx            the report
04LunarPM25_Presentation.pptx      the presentation
04LunarPM25_Dataset_Proposal.docx  the dataset proposal submitted earlier

data/original/
    PRSA_data_2010.1.1-2014.12.31.csv   the dataset exactly as downloaded, untouched
data/processed/
    final_dataset_daily.csv             the daily series used for all analysis
    data_quality_log.txt                what the cleaning found and what it did

code/
    01_prepare_data.py                  cleaning, daily aggregation, calendar variables
    02_eda.py                           exploratory analysis and the seasonal index diagnostic
    03_baseline_smoothing.py            naive baselines and exponential smoothing
    04_regression.py                    regression models and the festival variable test
    05_arima.py                         Box-Jenkins, ARIMAX, prediction interval coverage
    06_combining.py                     combining forecasts
    07_counterfactual_and_decisions.py  policy counterfactuals and decision evaluation

figures/    every figure used in the report
tables/     every numerical result, as CSV
```

## Reproducing the analysis

```
pip install pandas numpy statsmodels matplotlib scipy
cd code
cp ../data/original/PRSA_data_2010.1.1-2014.12.31.csv .
mkdir -p figures tables
python 01_prepare_data.py
python 02_eda.py
python 03_baseline_smoothing.py
python 04_regression.py
python 05_arima.py
python 06_combining.py
python 07_counterfactual_and_decisions.py
```

Run them in order. Script 01 writes `final_dataset_daily.csv`, which every later
script reads. Scripts 03 and 05 take several minutes because they roll the
forecast origin forward one day at a time through the test year.

We ran the whole sequence from an empty folder before submitting and confirmed
that every number in the report comes back identical.

## Dataset

Beijing PM2.5 dataset, UCI Machine Learning Repository.
Chen, S. (2015). Beijing PM2.5 [Dataset]. DOI 10.24432/C5JS49. Licensed CC BY 4.0.
https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data

43,824 hourly observations, 1 January 2010 to 31 December 2014. PM2.5 measured at
the US Embassy station in Beijing; meteorological variables from Beijing Capital
International Airport.

## Headline results

- Best model: seasonal ARMA, SARIMAX(1,0,1) x (1,0,1,7) with a constant.
  One day ahead RMSE 64.88 against the naive benchmark of 74.61.
- Prediction intervals are well calibrated at one day: the 80 per cent interval
  contained 80.8 per cent of actual values in the test year.
- Beyond about three days the models carry little information the long run
  average does not already hold.
- APEC emission controls in November 2014 cut PM2.5 by 32.3 µg/m³, about 36 per
  cent, after adjusting for the weather that actually occurred.
- The Spring Festival made the air worse, not better, in all four years tested,
  by an average of 18.8 µg/m³. We had expected the opposite.
- Accuracy and decision value are not the same thing. The model with the best
  RMSE catches 27 of 63 severe episodes; the naive model catches 33; the seven
  day model catches none at all.

## Choices worth knowing about

Two thresholds in `01_prepare_data.py` are judgement calls and sit at the top of
the file so they are easy to change:

- `MIN_HOURS_PER_DAY = 18`, the number of valid hourly readings a day needs
  before we trust its average.
- `MAX_GAP_TO_FILL = 6`, the longest gap we interpolate across.

Section 12 of the report discusses both.
