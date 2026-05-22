# Project goal
Study 0DTE options to help trading decision.  
Potential formulation: train a model to predict a distribution of closing (end-of-day) price.

# Raw data
This project uses Theta Data https://www.thetadata.net/ Options data. A standard option data plan is required to run this project.  
Python library `thetadata` is used to fetch raw data from Theta Data. Add a `creds.txt` with Theta Data username and id under current directory to use thetadata library.  
To fetch SPY greeks raw data and store in `data/raw/greeks/SPY` as parquet files, run:  
`python src/fetch_greeks.py`  
By default, the raw data from 2026-05-06 to 2026-05-19 are fetched. Customize a range by setting `end` and `periods`:  
`python src/fetch_greeks.py --end 2026-05-22 --periods 5`  


# Early exploration with raw data
`notebooks/` contains some early exploration with raw data including:
- clean meaningless data
- check timezone handling
- sanity check
- visualization  

See `notebooks/00_notebook_overview.md` for a more detailed overview.

# Data construction and feature engineering




# Output formulation
May start with quantile regression: 10th (10% chance price end up below here), 25th, 50th, 75th, 90th for the reason of no assumption required, thus skews, fat tails or other unexpected behavior can be naturally captured. Training loss is pinball loss.  
Then later, more mathematical distributions can be chosen based on the result of quantile regression. Some candidates: Student-t, Mixture of Gaussians, Skew-normal... May use maximum log-likelihood in training.  

# Potential application layer (low priority)
With a distribution of closing price given by the model, an additional application layer may be designed in this project to help decision making. Some examples: 
- User thinking about buying a call: probability of the underlying closing price stays above some price. 
- User thinking about selling a strangle: pribability of underlying closing price stays in some range. 
- User wants 40% win rate with 3:1 payoff: some strike price suggestion.  

The general idea is to give some suggestions based on user's customized risk/reward filter. 