# Project goal
Study 0DTE options to help trading decision.  
Potential formulation: train a model to predict a distribution of closing (end-of-day) price.

# Raw data
This project uses Theta Data https://www.thetadata.net/ Options data. A standard option data plan is required to run this project.  
Python library `thetadata` is used to fetch raw data from Theta Data. Add a `creds.txt` with Theta Data username and id under current directory to use thetadata library.  
Run `python src/fetch_raw_data.py` to fetch the following raw data:  
- raw greeks and store in `data/raw/greeks/[symbol]` as parquet files (to preserve data type)
- raw open interest and store in `data/raw/oi/[symbol]` as parquet files  
By default, the raw data of SPY from 2026-05-06 to 2026-05-19 are fetched. Customize a range by setting `symbol`, `end` for end date and `periods` in days:  
`python src/fetch_raw_data.py --symbol AAPL --end 2026-05-22 --periods 5`  

To generate a summary report based on raw data, run:  
`python src/generate_report.py`  
Generated report is saved as `/data/visualization/spy_report.html`.  

# Early exploration with raw data
`notebooks/` contains some early exploration with raw data including:
- clean meaningless data
- check timezone handling
- sanity check
- visualization  

See `notebooks/00_notebook_overview.md` for a more detailed overview.

# Data construction and feature engineering
Run `python src/process_raw_data.py` to read raw data from `data/raw` and generate the following files in `data/processed`:  

## `spy_closing_prices.csv` and `spy_opening_prices.csv`

Closing and opening prices are extracted from raw greeks `underlying_price` column at the closing time 16:00 and the opening time 9:30. Closing and opening prices have simple data type, so csv is chosen for better readability. Uniqueness per expiration is asserted.  

## `spy_processed.parquet`

A very basic filter (implied_vol > 0, iv_error < 1.0, delta.abs().between(0.01, 0.99), bid > 0) is applied before feature engineering to filter out meaningless data. Opening (9:30) and closing (16:00) data is also filtered out to avoid 0 denominator in log-return calculation.  

The following columns are directly from raw data: `symbol`, `expiration`, `strike`, `right`, `timestamp`, `bid`, `ask`, `delta`, `theta`, `vega`, `rho`, `epsilon`, `lambda`, `implied_vol`, `iv_error`, `underlying_timestamp`, `underlying_price`.  

`spy_close` and `spy_open` are the closing and opening price on that day.  

Log-return:  
- `log_return_from_open` = ln(underlying_price / spy_open)  
- `log_return` ln(closing price / price at snapshot) is used as label because log-return is conventional in quant ML   

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

# Output formulation
May start with quantile regression: 10th (10% chance price end up below here), 25th, 50th, 75th, 90th for the reason of no assumption required, thus skews, fat tails or other unexpected behavior can be naturally captured. Training loss is pinball loss.  
Then later, more mathematical distributions can be chosen based on the result of quantile regression. Some candidates: Student-t, Mixture of Gaussians, Skew-normal... May use maximum log-likelihood in training.  

# Potential application layer (low priority)
With a distribution of closing price given by the model, an additional application layer may be designed in this project to help decision making. Some examples: 
- User thinking about buying a call: probability of the underlying closing price stays above some price. 
- User thinking about selling a strangle: probability of underlying closing price stays in some range. 
- User wants 40% win rate with 3:1 payoff: some strike price suggestion.  

The general idea is to give some suggestions based on user's customized risk/reward filter. 