# 01_fetch_greeks  

Fetch 2026-05-06 to 2026-05-19 SPY 0dte options data from Theta Data and save in data/raw/greeks/SPY.  

I choosed parquet for output file type to preserve data type. Each parquet file corresponds to one day of 0dte options. 


# 02_process_raw_data

Read previously saved SPY data from data/raw/greeks/SPY, filter out some meaningless data.  

Time matters to 0dte research, so I checked the timestamp handling in the data. All timestamp are in America/New_York time.  

For 0DTE SPY options, the closing price used for settlement is the SPY official last price at 4:00PM ET. Data is already in ET, so no additional timezone / daylight saving handling is needed.  

Some sanity check visualization:  

- ATM Call / Put ask price vs time to maturity
- underlying price by day
- IV smile at noon