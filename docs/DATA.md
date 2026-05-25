# Data Sources and Local File Layout

Raw market data are not redistributed in this repository. The included
precomputed outputs can be inspected directly, and the source checkpoints are
included for re-running the frozen evaluation once the raw data files are
prepared locally.

## Expected Layout

Place the following files under `data/raw/`:

```text
data/raw/
  49_Industry_Portfolios_Daily.csv
  F-F_Research_Data_Factors_daily.csv
  F-F_Research_Data_5_Factors_2x3_daily.csv
  F-F_Momentum_Factor_daily.csv
  VIXCLS.parquet
  VXVCLS.parquet
  master_daily_features_macro_dailyonly_raw_only.csv
```

## Source Notes

- `49_Industry_Portfolios_Daily.csv`: daily Fama-French 49 industry portfolio
  returns from the Kenneth R. French Data Library.
- `F-F_Research_Data_Factors_daily.csv`,
  `F-F_Research_Data_5_Factors_2x3_daily.csv`, and
  `F-F_Momentum_Factor_daily.csv`: daily factor files from the Kenneth R.
  French Data Library.
- `VIXCLS.parquet` and `VXVCLS.parquet`: volatility-index series used as
  state features. The code expects parquet files with date-indexed observations.
- `master_daily_features_macro_dailyonly_raw_only.csv`: adjusted ETF price and
  feature panel used for the external ETF universe.

## ETF Universe

The ETF universe contains 20 ETFs:
EEM, GLD, HYG, IEF, IWM, LQD, QQQ, SPY, TLT, UUP,
XLB, XLE, XLF, XLI, XLK, XLP, XLRE, XLU, XLV, XLY.

The pipeline starts from 2015-10-08 to retain XLRE, drops ETF columns with
missing values after that date, and computes daily percentage returns before
forming rolling 252-day decision states.

The code assumes the column conventions implemented in `src/CON2.py`. If a data
provider changes headers or formats, adapt the local preprocessing step before
running the full pipeline.
