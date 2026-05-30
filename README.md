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
Extract closing and opening prices from raw greeks `underlying_price` column at the closing time 16:00 and the opening time 9:30, and store in `data/processed/spy_closing_prices.csv` and `data/processed/spy_opening_prices.csv`. Choose csv for better readability.  
Log-return log(closing price / price at snapshot) is used as label because log-return is conventional in quant ML.  

Run `python src/process_raw_data.py` to process raw data through the following steps:  
- Load raw greeks from `data/raw/greeks`.  
- Extract and store closing / opening prices.  
- Apply basic filter (implied_vol > 0, iv_error < 1.0, delta.abs().between(0.01, 0.99), bid > 0) to filter out meaningless data.  
- Filter out opening (9:30) and closing (16:00) data.  
- Read `data/processed/spy_closing_prices.csv` and `data/processed/spy_opening_prices.csv`, merge as `spy_close` and `spy_open`, assert uniqueness per expiration.  
- Add `log_return_from_open` = ln(underlying_price / spy_open). 
- Add `ttm_min` time to maturity (expiry) in minutes counting to 16:00.  
- Add `log_return` = ln(spy_close / underlying_price), this is the label.  
- Add `d1` and `gamma` calculated through Black-Scholes.  
- Read `data/raw/oi` and add `open_interest`.  
- Calculate and add delta exposure `dex`, gamma exposure `gex` and theta exposure `tex` from the greeks and OI.  
- Save to `data/processed/spy_processed.parquet`.  
- Calculate `net_dex`, `net_gex` and `net_tex` from DEX, GEX and TEX.  
- calculate `theta_decay` from `net_tex` and time to expiry.  
- Extract `atm_iv`, `iv_call_25d` and `iv_put_25d`, calculate `iv_skew_25d` and `iv_smile_curvature_25d`.  
- Save to `data/processed/spy_exposure.parquet`.  

`data/processed/spy_processed.parquet` columns: ['symbol', 'expiration', 'strike', 'right', 'timestamp', 'bid', 'ask', 'delta', 'theta', 'vega', 'rho', 'epsilon', 'lambda', 'implied_vol', 'iv_error', 'underlying_timestamp', 'underlying_price', 'spy_close', 'spy_open', 'log_return_from_open', 'ttm_min', 'log_return', 'd1', 'gamma', 'open_interest', 'dex', 'gex', 'tex']  

`data/processed/spy_exposure.parquet` columns:  ['timestamp', 'net_dex', 'net_gex', 'underlying_price', 'net_gex_norm', 'net_dex_norm', 'atm_iv', 'iv_call_25d', 'iv_put_25d', 'iv_skew_25d', 'iv_smile_curvature_25d', 'net_tex', 'net_tex_norm', 'ttm_min', 'ttm_hours', 'theta_decay']  


# Output formulation
May start with quantile regression: 10th (10% chance price end up below here), 25th, 50th, 75th, 90th for the reason of no assumption required, thus skews, fat tails or other unexpected behavior can be naturally captured. Training loss is pinball loss.  
Then later, more mathematical distributions can be chosen based on the result of quantile regression. Some candidates: Student-t, Mixture of Gaussians, Skew-normal... May use maximum log-likelihood in training.  

# Potential application layer (low priority)
With a distribution of closing price given by the model, an additional application layer may be designed in this project to help decision making. Some examples: 
- User thinking about buying a call: probability of the underlying closing price stays above some price. 
- User thinking about selling a strangle: probability of underlying closing price stays in some range. 
- User wants 40% win rate with 3:1 payoff: some strike price suggestion.  

The general idea is to give some suggestions based on user's customized risk/reward filter. 