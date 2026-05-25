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

End-of-day price returned by ThetaData is at 17:15 which is not ideal becasue 0DTE closes at 16:00, so the underlying price at 16:00 is used as the closing (settlement) price. Similarly, the underlying price at 9:30 is used as opening price. These are processed in src/ and stored in data/processed. Closing and opening data are filtered because I don't want them in training. Then, closing and opening price are fetched from data/processed and added to dataframe as new column, and I verified data with same expirations have the same closing and opening price.  

Log-return `ln(closing price / current price)` is chosen to be the label, and log-return from open `ln(current price / opening price)` is calculated to be a potential feature. Since this is a 0DTE project, the closing / current / opening prices are fairly close, causing the absolute value of log-return to be very small. I tested the precision needed for log-return to ensure correctness of the closing price reconstruction, and the minimum precision needed for log-return is 5 decimal places. The calculated column in dataframe has 6 decimal places, which can make the closing price and the reconstruction match.  