import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import zscore, kurtosis, skew

'''ENGINEERING'''
# Robust clipping using IQR

def clip_outliers(df, iqr_multiplier=3.0):
    q1 = df.quantile(0.25, axis=0)
    q3 = df.quantile(0.75, axis=0)
    iqr = q3 - q1
    lower_bound = q1 - iqr_multiplier * iqr
    upper_bound = q3 + iqr_multiplier * iqr
    return df.clip(lower=lower_bound, upper=upper_bound, axis=1)


# Compute latest rolling correlation matrix for heatmap

def compute_latest_corr_matrix(df, window=60):
    rolling_corrs = df.rolling(window=window).corr()
    latest_date = rolling_corrs.index.get_level_values(0).max()
    return rolling_corrs.loc[latest_date]


# Compute shape statistics (skewness and kurtosis) for a DataFrame

def describe_shape(df):
    print("Skewness:")
    print(df.apply(lambda x: skew(x.dropna()), axis=0).round(3))
    print("\nKurtosis:")
    print(df.apply(lambda x: kurtosis(x.dropna(), fisher=False), axis=0).round(3))

'''Features'''
# Compute rolling portfolio volatility from returns

def compute_rolling_portfolio_volatility(future_returns, window=10):
    portfolio_return = future_returns.mean(axis=1)
    return portfolio_return.rolling(window=window).std()

# Compute momentum
def calculate_momentum(df, momentum_period=10):
    """
    Calculate momentum as the difference in log prices over the period.
    """
    df['log_price'] = np.log(df['Close'])
    df['momentum'] = df['log_price'].diff(momentum_period)
    return df

'''Metrics'''
# Directional Accuracy
def directional_accuracy(y_true, y_pred):
    # Compare signs of predicted and actual returns
    correct_signs = np.sign(y_true) == np.sign(y_pred)
    return np.mean(correct_signs)

# Weighted
