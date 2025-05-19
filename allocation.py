import pickle
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas as pd

# Load data
with open('data/risk_model_results.pkl', 'rb') as f:
    risk_data = pickle.load(f)
with open('data/return_model_results.pkl', 'rb') as f:
    return_data = pickle.load(f)

df = pd.read_csv('data/combined_stock_data_wide.csv', parse_dates=["Date"])
df.set_index("Date", inplace=True)
df.dropna(inplace=True)
close_cols = [c for c in df.columns if c.endswith("_Close") and "Adj" not in c]
close_df = df[close_cols]

results_risk = risk_data['results']
results_return = return_data['h_preds_orig']
n_assets = len(close_cols)

# Extract dates and data
dates_risk = [r['date'] for r in results_risk]
dates_return = [r[1] for r in results_return]
returns_pred_log = np.array([r[0] for r in results_return])  # Predicted 7-day log returns
cov_matrices = np.array([r['cov_pred'] for r in results_risk])
y_pred_std = np.array([r['y_pred_std'] for r in results_risk])

# Convert predicted 7-day log returns to simple returns for MVO
returns_pred = np.exp(returns_pred_log) - 1  # Now 7-day simple returns

# Compute actual 7-day log returns from closing prices
actual_seven_day_log_returns = np.log(close_df.shift(-7) / close_df)
actual_seven_day_log_returns = actual_seven_day_log_returns.iloc[7:-7].dropna()  # Align with 7-day shift
# Convert to 7-day simple returns for backtest consistency
actual_seven_day_simple_returns = np.exp(actual_seven_day_log_returns) - 1
actual_dates = actual_seven_day_log_returns.index

# Align dates using intersection
common_dates = sorted(set(dates_risk).intersection(dates_return).intersection(actual_dates))
assert len(common_dates) > 0, "No common dates between risk, return, and actual data"
print(f"✓ Aligned {len(common_dates)} common dates")

date_to_idx_risk = {d: i for i, d in enumerate(dates_risk)}
date_to_idx_return = {d: i for i, d in enumerate(dates_return)}
date_to_idx_actual = {d: i for i, d in enumerate(actual_dates)}
aligned_indices_risk = [date_to_idx_risk[d] for d in common_dates]
aligned_indices_return = [date_to_idx_return[d] for d in common_dates]
aligned_indices_actual = [date_to_idx_actual[d] for d in common_dates]
returns_pred = returns_pred[aligned_indices_return]
returns_pred_log = returns_pred_log[aligned_indices_return]
cov_matrices = cov_matrices[aligned_indices_risk]
y_pred_std = y_pred_std[aligned_indices_risk]
actual_log_returns = actual_seven_day_log_returns.iloc[aligned_indices_actual].values
actual_simple_returns = actual_seven_day_simple_returns.iloc[aligned_indices_actual].values
dates = common_dates

# Debug: Check ranges
print("Predicted 7-day Log Returns Range:", np.min(returns_pred_log), np.max(returns_pred_log))
print("Predicted 7-day Simple Returns Range:", np.min(returns_pred), np.max(returns_pred))
print("Actual 7-day Log Returns Range:", np.min(actual_log_returns), np.max(actual_log_returns))
print("Actual 7-day Simple Returns Range:", np.min(actual_simple_returns), np.max(actual_simple_returns))
print("Volatility Proxy Range:", np.min([np.sqrt(np.diag(cov_matrix)) for cov_matrix in cov_matrices]),
      np.max([np.sqrt(np.diag(cov_matrix)) for cov_matrix in cov_matrices]))

# Apply uncertainty filter using y_pred_std
random_baseline = 1 / np.sqrt(n_assets)
valid_stocks = np.mean(y_pred_std, axis=0) <= random_baseline
valid_stocks = valid_stocks.ravel()  # Flatten to 1D boolean mask
print("Valid Stocks (Uncertainty Filter):", valid_stocks)
if not valid_stocks.all():
    print("Warning: Some stocks excluded due to high uncertainty.")


# Mean-Variance Optimization functions
def portfolio_return(weights, returns):
    return np.sum(returns.mean(axis=0) * weights)


def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(weights.T @ cov_matrix @ weights)


def neg_sharpe_ratio(weights, returns, cov_matrix):
    p_ret = portfolio_return(weights, returns)
    p_vol = portfolio_volatility(weights, cov_matrix)
    return -p_ret / p_vol


# Optimize weights per step
weights_opt = []
for i in range(len(returns_pred)):
    cov_matrix = cov_matrices[i]
    curr_pred_simple_returns = returns_pred[i][valid_stocks]
    curr_actual_log_returns = actual_log_returns[i][valid_stocks]
    curr_actual_simple_returns = actual_simple_returns[i][valid_stocks]
    curr_cov = cov_matrix[np.ix_(valid_stocks, valid_stocks)]
    curr_n_assets = sum(valid_stocks)

    if curr_n_assets == 0:
        curr_w = np.zeros(n_assets)
    else:
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
        bounds = tuple((0, 1) for _ in range(curr_n_assets))
        init_weights = np.ones(curr_n_assets) / curr_n_assets
        res = minimize(neg_sharpe_ratio, init_weights,
                       args=(curr_pred_simple_returns, curr_cov),
                       method='SLSQP', bounds=bounds, constraints=constraints)
        curr_w = np.zeros(n_assets)
        curr_w[valid_stocks] = res.x if res.success else init_weights
    weights_opt.append(curr_w)

weights_opt = np.array(weights_opt)
print("✓ Optimized weights shape:", weights_opt.shape)

# Backtest portfolio over non-overlapping 7-day periods with ground truth
horizon = 7
initial_capital = 100000
portfolio_value_pred = [initial_capital]
portfolio_value_actual = [initial_capital]
transaction_cost = 0.001
portfolio_dates = [dates[0]]

for t in range(0, len(weights_opt) - horizon, horizon):
    curr_w = weights_opt[t]
    # Predicted performance (from predicted 7-day simple returns)
    weighted_return_pred = np.sum(returns_pred[t] * curr_w)
    value_change_pred = portfolio_value_pred[-1] * (1 + weighted_return_pred)
    # Actual performance (from actual 7-day simple returns)
    weighted_return_actual = np.sum(actual_simple_returns[t] * curr_w)
    value_change_actual = portfolio_value_actual[-1] * (1 + weighted_return_actual)
    if t > 0:
        prev_w = weights_opt[t - horizon]
        cost_pred = portfolio_value_pred[-1] * np.sum(np.abs(curr_w - prev_w)) * transaction_cost
        cost_actual = portfolio_value_actual[-1] * np.sum(np.abs(curr_w - prev_w)) * transaction_cost
        value_change_pred -= cost_pred
        value_change_actual -= cost_actual
    portfolio_value_pred.append(value_change_pred)
    portfolio_value_actual.append(value_change_actual)
    portfolio_dates.append(dates[t + horizon])

# Handle remaining steps
remaining_steps = len(weights_opt) % horizon
if remaining_steps > 0:
    t = len(weights_opt) - remaining_steps
    curr_w = weights_opt[t]
    weighted_return_pred = np.sum(returns_pred[t] * curr_w)
    weighted_return_actual = np.sum(actual_simple_returns[t] * curr_w)
    prorated_return_pred = weighted_return_pred * (remaining_steps / horizon)
    prorated_return_actual = weighted_return_actual * (remaining_steps / horizon)
    value_change_pred = portfolio_value_pred[-1] * (1 + prorated_return_pred)
    value_change_actual = portfolio_value_actual[-1] * (1 + prorated_return_actual)
    prev_w = weights_opt[t - horizon if t >= horizon else 0]
    cost_pred = portfolio_value_pred[-1] * np.sum(np.abs(curr_w - prev_w)) * transaction_cost
    cost_actual = portfolio_value_actual[-1] * np.sum(np.abs(curr_w - prev_w)) * transaction_cost
    value_change_pred -= cost_pred
    value_change_actual -= cost_actual
    portfolio_value_pred.append(value_change_pred)
    portfolio_value_actual.append(value_change_actual)
    portfolio_dates.append(dates[-1])

print("✓ Backtest completed, portfolio values aligned with dates")

# Final results
final_value_pred = portfolio_value_pred[-1]
final_value_actual = portfolio_value_actual[-1]
total_return_pred = (final_value_pred - initial_capital) / initial_capital * 100
total_return_actual = (final_value_actual - initial_capital) / initial_capital * 100
print(f"\nPredicted Final Portfolio Value: ${final_value_pred:.2f}")
print(f"Predicted Total Return: {total_return_pred:.2f}%")
print(f"Actual Final Portfolio Value: ${final_value_actual:.2f}")
print(f"Actual Total Return: {total_return_actual:.2f}%")

# Plot portfolio value
plt.figure(figsize=(10, 5))
plt.plot(portfolio_dates, portfolio_value_pred, label='Predicted Portfolio Value', marker='o')
plt.plot(portfolio_dates, portfolio_value_actual, label='Actual Portfolio Value', marker='o')
plt.title("Portfolio Value Over Time (Non-Overlapping 7-day Periods)")
plt.xlabel("Date")
plt.ylabel("Value ($)")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()