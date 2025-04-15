import pandas as pd

def process_daily_stock_data(input_file, output_file):
    """
    Parameters:
    - input_file: str, path to the input CSV file.
    - output_file: str, path to save the processed CSV file.
    """
    # Load the data, parsing 'Date/Time' as datetime
    df = pd.read_csv(input_file, parse_dates=['Date/Time'])

    # Drop 'Open Interest'
    if 'Open Interest' in df.columns:
        df = df.drop(columns=['Open Interest'])

    # Set 'Date/Time' as the index
    df = df.set_index('Date/Time')

    # Resample to daily frequency for each ticker
    df_resampled = df.groupby('Ticker').resample('D').agg({
        'Open': 'first',  # First value of the day
        'High': 'max',  # Maximum value of the day
        'Low': 'min',  # Minimum value of the day
        'Close': 'last',  # Last value of the day
        'Volume': 'sum'  # Sum of the day's volume
    })

    # Pivot the data so each ticker's features are in columns
    df_pivoted = df_resampled.unstack('Ticker')

    # Flatten the column names (e.g., 'FPT_Open', 'FPT_Close')
    df_pivoted.columns = [f'{col[1]}_{col[0]}' for col in df_pivoted.columns]

    # Drop rows with any missing values
    df_pivoted = df_pivoted.dropna()

    # Save the processed data to a CSV file
    df_pivoted.to_csv(output_file)
    print(f"Processed daily data saved to {output_file}")


# Example usage
if __name__ == "__main__":
    input_file = 'data/combined_data.csv'  # Replace with your input file path
    output_file = 'data/processed_daily_data.csv'  # Desired output file path
    process_daily_stock_data(input_file, output_file)