# Project goal
Study Zero-day-to-expiration options to help trading decisions.  

Motivation: 0DTE options went from a niche product to roughly half the market in just a few years. 0DTE options have large trading volume, and the growth has outpaced research coverage.  

Research question: Given the observable options-market state (eg. implied volatility, greeks, open-interest exposures) at an intraday snapshot, can we estimate a probability distribution for the underlying's closing price at end-of-day — and does that distribution provide decision-relevant information for 0DTE option positioning?  

Challenges:  
- Data and market microstructure  
    - Intraday data is extremely noisy  
    - No intraday open interest because OI is only updated once per day  
    - Some greeks become explosive as time to maturity approaches 0  
- Modeling challenge  
    - Hard to model distribution due to non-stationarity of market regime  
    - Multiple intraday snapshot but only one closing price realization per trading day  
- Evaluation  
    - Hard to validate against true probabilities due to only one realized closing price  

# Raw data
This project uses Theta Data https://www.thetadata.net/ Options data. A standard option data plan is required to run this project.  
Python library `thetadata` is used to fetch raw data from Theta Data. Add a `creds.txt` with Theta Data username and id under current directory to use thetadata library.  
Run `python src/preprocess/fetch_raw_data.py` to fetch the following raw data:  
- raw greeks and store in `data/raw/greeks/[symbol]` as parquet files (to preserve data type)
- raw open interest and store in `data/raw/oi/[symbol]` as parquet files  
By default, the raw data of SPY from 2026-05-06 to 2026-05-19 are fetched. Customize a range by setting `symbol`, `end` for end date and `periods` in days:  
`python src/preprocess/fetch_raw_data.py --symbol AAPL --end 2026-05-22 --periods 5`  

To generate a summary report based on raw data, run:  
`python src/eda/generate_report.py`  
This generate a raw data report for each symbol and save as  `/data/visualization/raw_data/reports/[symbol]_report.html`.  

# Early exploration with raw data
`notebooks/` contains some early exploration with raw data including: data filtering, timezone handling verification, sanity check and visualization. Then, the features are designed and tested before implemented in src.  

See `notebooks/00_notebook_overview.md` for a more detailed overview.  

# Feature engineering
Run `python src/preprocess/process_raw_data.py` to read raw data from `data/raw`, extract and calculate features, and generate the following files in `data/processed`:  

## `spy_closing_prices.csv` and `spy_opening_prices.csv`

Closing and opening prices are extracted from raw greeks `underlying_price` column at the closing time 16:00 and the opening time 9:30. Closing and opening prices have simple data type, so csv is chosen for better readability. Uniqueness per expiration is asserted.  

## `spy_processed.parquet`

A very basic filter (implied_vol > 0, iv_error < 1.0, delta.abs().between(0.01, 0.99), bid > 0) is applied before feature engineering to filter out meaningless data. Opening (9:30) and closing (16:00) data is also filtered out to avoid 0 denominator in log-return calculation.  

The following columns are directly from raw data: `symbol`, `expiration`, `strike`, `right`, `timestamp`, `bid`, `ask`, `delta`, `theta`, `vega`, `rho`, `epsilon`, `lambda`, `implied_vol`, `iv_error`, `underlying_timestamp`, `underlying_price`.  

`spy_close` and `spy_open` are the closing and opening price on that day.  

Log-return:  
- `log_return_from_open` = ln(underlying_price / spy_open)  
- `log_return` ln(closing price / price at snapshot) is the target. Log-return is conventional in quant ML   

`ttm_min` is time to maturity (expiry) in minutes counting to 16:00.  

Black-Scholes: `d1` and `gamma` are calculated from Black_scholes.  

Open interest related:  
- `open_interest` is read from raw oi data  
- delta exposure `dex`, gamma exposure `gex` and theta exposure `tex` are calculated from OI and the corresponding greeks  
- `total_oi` is the put + call OI at a (day, strike)  
- `put_oi_fraction` = put OI / total OI at a (day, strike)  
- `max_oi_strike` is the strike price with max OI at a day  
- `oi_concentration_top3` is (top 3 largest put + call OI) / total OI in all strikes on the day  
- `distance_to_max_oi` is underlying price - (max OI strike on that day)  

Bid-ask features: 
- `bid_ask_mid` = (bid + ask) / 2  
- `bid_ask_spread` = ask - bid  
- `bid_ask_spread_norm` = bid ask spread / bid ask mid   

Meaningless data with 0 or NaN OI or ask is filtered.  

## `spy_aggregate.parquet`

`timestamp` is the unique key in this table. `underlying_price` is the price at the timestamp.  

Exposure:  
- `net_dex`, `net_gex` and `net_tex` are aggregated from DEX, GEX and TEX  
- `net_gex_norm`, `net_dex_norm`, `net_tex_norm` are normalized using underlying price  

Implied Volatility related features:  
- `atm_iv` at-the-money IV, the IV where the strike is closest to the underlying price
- `iv_call_25d` and `iv_put_25d` are the call and put IV where delta is closest to ±0.25  
- `iv_skew_25d` = iv_put_25d - iv_call_25d  
-  `iv_smile_curvature_25d` = iv_put_25d + iv_call_25d - 2 * atm_iv

`ttm_min` and `ttm_hours` are time to maturity in minutes and in hours.  

`theta_decay` is calculated based on https://flashalpha.com/concepts/theta-decay   


# Data construction  

Construct the ready for model dataset by running: `python src/preprocess/construct_dataset.py`. Results are saved to `data/model_input/`.  

Price related reatures are normalized by underlying_price, and the target log-return is normalized as: `log_return / (atm_iv * sqrt(ttm_years))`. `(atm_iv * sqrt(ttm_years)` is saved as new column `norm_factor` for later calculation.  

For exploratory data analysis about the normalized log return, run `python src/eda/output_distribution.py --symbol SPY --start-date 2025-01-01 --end-date 2026-06-01` to see a histogram of the normalized log return and a fitted normal distribution in `data/visualization/log_return_norm/`.  

The dataset is mainly constructed from aggregate, and some data from processed are collapsed into scaler and added to the dataset.  

The dataset keeps expiration date for later use to avoid leakage. Metadata (expiration and underlying_price) is kept for later reconstruction, so features columns need to be explicitly specified during training.  

Histogram features may be added in a later version. 

# Output formulation

This project uses a normalized log return as the model's output:  
Log return norm = ln(closing price / current price) / (atm iv * sqrt(ttm in years))   

# Model 1: Black-Scholes Analytic Model

Run this model with symbol, start-date and end-date:  
`python src/model/model1.py --symbol SPY --start-date 2026-03-31 --end-date 2026-06-01`  
This prints the pinball loss on console, saves predictions to `data/predictions`, and saves a visualization to `data/visualization/model1`.  


# Model 2: Quantile Regression Model

Run this model:  
`python src/model/model2.py`  
The following arguments can be set, and here are the default arguments:  
`python src/model/model2.py --symbol SPY --start-date 2026-03-31 --end-date 2026-06-01 --window-mode rolling --train-window-days 30 --val-days 1`  
This saves prediction to `data/predictions`, and saves a comparison report to `data/model_report`.  


# Potential application layer
With a distribution of closing price given by the model, an additional application layer may be designed in this project to help decision making. Some examples: 
- User thinking about buying a call: probability of the underlying closing price stays above some price. 
- User thinking about selling a strangle: probability of underlying closing price stays in some range. 
- User wants 40% win rate with 3:1 payoff: some strike price suggestion.  

The general idea is to give some suggestions based on user's customized risk/reward filter. 