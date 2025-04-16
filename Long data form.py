import yfinance as yf
import pandas as pd

# Define the stock tickers and date range
tickers = ["VCB.VN", "MSN.VN", "HPG.VN", "VIC.VN", "VJC.VN"]  # Replace with your tickers
start_date = "2023-01-1"
end_date = "2025-04-15"

# Fetch data for all tickers at once with error handling
try:
    data = yf.download(tickers, start=start_date, end=end_date, group_by="ticker", threads=True, auto_adjust=False)
    if data.empty:
        raise ValueError("No data was downloaded. Check ticker symbols or date range.")
except Exception as e:
    print(f"Error downloading data: {e}")
    exit(1)



# Check if data was successfully downloaded for each ticker
available_tickers = []
for ticker in tickers:
    try:
        # Check if the ticker exists in the DataFrame's columns (handle both multi-index and flat index)
        if isinstance(data.columns, pd.MultiIndex):
            # Multi-index case (group_by="ticker")
            if ticker in data.columns.get_level_values(0):
                ticker_data = data[ticker]
                if 'Close' in ticker_data.columns and not ticker_data['Close'].isna().all():
                    available_tickers.append(ticker)
                else:
                    print(f"No valid data for {ticker}: Missing or all-NaN Close prices")
            else:
                print(f"No data column for {ticker}")
        else:
            # Flat index case (e.g., if group_by="ticker" didn't work as expected)
            if 'Close' in data.columns and not data['Close'].isna().all():
                # Single ticker case or unexpected structure
                available_tickers.append(ticker)
            else:
                print(f"No valid data for {ticker}: Missing or all-NaN Close prices")
    except Exception as e:
        print(f"Error checking data for {ticker}: {e}")

if not available_tickers:
    print("No data available for any ticker. Exiting.")
    exit(1)
else:
    print(f"Successfully downloaded data for: {available_tickers}")

# Save individual files for each ticker
for ticker in available_tickers:
    try:
        ticker_data = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
        ticker_data.to_csv(f"{ticker}_data.csv")
        print(f"Saved individual file for {ticker}")
    except Exception as e:
        print(f"Error saving data for {ticker}: {e}")
