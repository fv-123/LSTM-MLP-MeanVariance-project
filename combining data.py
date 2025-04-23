import pandas as pd

# Define the tickers
tickers = ["VCB.VN", "LPB.VN", "HPG.VN", "VIX.VN", "YEG.VN","HAH.VN","HHS.VN"]
features = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

# Read and combine the individual files
all_data = {}
for ticker in tickers:
    df = pd.read_csv(f"{ticker}_data.csv")
    if 'Date' not in df.columns:
        df = df.reset_index()
    df.set_index("Date", inplace=True)
    # Rename columns to include ticker prefix
    df.columns = [f"{ticker}_{feature}" for feature in df.columns]
    all_data[ticker] = df

# Combine all DataFrames on Date index
combined_data = pd.concat(all_data.values(), axis=1)

# Reset index to make Date a column
combined_data = combined_data.reset_index()

# Save the combined wide-format data to a CSV file
combined_data.to_csv("combined_stock_data_wide.csv", index=False)