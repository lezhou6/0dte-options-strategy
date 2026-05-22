# Project goal
Study 0DTE options to help trading decision.  
Potential formulation: train a model to predict a distribution of closing (end-of-day) price.

# Raw data
This project uses Theta Data Options data. A standard option data plan is required to run this project.  
Python library `thetadata` is used to fetch raw data from Theta Data. Add a `creds.txt` with Theta Data username and id under same directory of source code to use thetadata library.  
Raw data is fetched and then stored in `data/raw` as parquet files to preserve data type.  

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