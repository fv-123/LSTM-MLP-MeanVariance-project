# Full Pipeline with MC Dropout Uncertainty Estimation

#%% Imports & Configuration
import os, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score

# Hyperparameters
horizon          = 7      # days ahead for realized vol
sequence_length  = 35      # look-back for LSTM
test_frac        = 0.20    # 20% hold-out for test
n_splits         = 5       # CV folds
batch_size       = 64
num_epochs       = 50
patience         = 6
learning_rate    = 3e-4
hidden_size      = 128
num_layers       = 2
dropout_prob     = 0.2     # dropout for MC sampling
mc_samples       = 90      # MC dropout samples per forecast

# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% 1) Load & initial cleanup
df = pd.read_csv("data/combined_stock_data_wide.csv", parse_dates=["Date"])
df.set_index("Date", inplace=True)
df.dropna(inplace=True)  # drop ~8 incomplete days

close_cols  = [c for c in df.columns if c.endswith("_Close") and "Adj" not in c]
volume_cols = [c for c in df.columns if c.endswith("_Volume")]
asset_count = len(close_cols)

#%% 2) Compute returns & realized vol
close_df    = df[close_cols]
log_returns = np.log(close_df / close_df.shift(1))
realized_vol = (
    log_returns
      .rolling(window=horizon, min_periods=horizon).std()
      .shift(-horizon)
)[close_cols]

#%% 3) Feature engineering (no drop)
past_std_ewm = log_returns.ewm(span=horizon, adjust=False).std().add_suffix("_ewm")
disc_log_ret = log_returns.add_suffix("_logret")
liq = (np.log1p(df[volume_cols]).diff()
       .ewm(span=horizon, adjust=False).mean()
     ).rename(columns=lambda c: c.replace("_Volume","_liq"))

#%% 4) Align by intersection
common_idx = (
    past_std_ewm.index
       .intersection(disc_log_ret.index)
       .intersection(liq.index)
       .intersection(realized_vol.index)
)
past_std_ewm = past_std_ewm.loc[common_idx]
disc_log_ret = disc_log_ret.loc[common_idx]
liq           = liq.loc[common_idx]
realized_vol  = realized_vol.loc[common_idx]

#%% 5) Final dropna & reset
features = pd.concat([past_std_ewm, disc_log_ret, liq], axis=1)
targets  = realized_vol.copy()
combined = pd.concat([features, targets], axis=1).dropna()
features = combined[features.columns].reset_index(drop=True)
targets  = combined[targets.columns].reset_index(drop=True)

#%% 6) Sequence creation
def create_sequences(X_df, Y_df, seq_len):
    Xs, Ys = [], []
    for i in range(len(X_df) - seq_len):
        Xs.append(X_df.iloc[i:i+seq_len].values)
        Ys.append(Y_df.iloc[i+seq_len-1].values)
    return np.array(Xs), np.array(Ys)

X_all, y_all = create_sequences(features, targets, sequence_length)
N = len(X_all)
assert N>0, "No sequences generated"

#%% 7) Train/Test split (80/20)
split_i = int((1 - test_frac) * N)
X_trainval, X_test = X_all[:split_i], X_all[split_i:]
y_trainval, y_test = y_all[:split_i], y_all[split_i:]

print(f"Train/Val samples: {len(X_trainval)}, Test samples: {len(X_test)}")

#%% 8) Model with MC Dropout support
class VolLSTM_MC(nn.Module):
    def __init__(self, in_feats, hid, nlayers, out_feats, drop):
        super().__init__()
        self.lstm = nn.LSTM(in_feats, hid, nlayers,
                            batch_first=True, dropout=drop)
        self.dropout = nn.Dropout(drop)
        self.fc   = nn.Linear(hid, out_feats)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        h = self.dropout(h[-1])  # apply dropout at inference for MC
        return self.fc(h)

#%% 9) CV on train/val (standard)
tscv = TimeSeriesSplit(n_splits=n_splits)
for fold, (tr_idx, vl_idx) in enumerate(tscv.split(X_trainval), 1):
    print(f"\n--- CV Fold {fold}/{n_splits} ---")

    # Prepare scalers and loaders
    sx = StandardScaler().fit(X_trainval[tr_idx].reshape(-1, X_all.shape[2]))
    sy = StandardScaler().fit(y_trainval[tr_idx])
    Xtr = sx.transform(X_trainval[tr_idx].reshape(-1, X_all.shape[2])).reshape(-1, sequence_length, X_all.shape[2])
    Xvl = sx.transform(X_trainval[vl_idx].reshape(-1, X_all.shape[2])).reshape(-1, sequence_length, X_all.shape[2])
    ytr = sy.transform(y_trainval[tr_idx])
    yvl = sy.transform(y_trainval[vl_idx])

    tr_dl = DataLoader(
        TensorDataset(
            torch.tensor(Xtr, dtype=torch.float32),
            torch.tensor(ytr, dtype=torch.float32)
        ),
        batch_size=batch_size,
        shuffle=False
    )
    vl_dl = DataLoader(
        TensorDataset(
            torch.tensor(Xvl, dtype=torch.float32),
            torch.tensor(yvl, dtype=torch.float32)
        ),
        batch_size=batch_size,
        shuffle=False
    )

    # Model & optimizer
    model = VolLSTM_MC(X_all.shape[2], hidden_size, num_layers, asset_count, dropout_prob).to(device)
    opt = optim.Adam(model.parameters(), lr=learning_rate)

    best_val, wait = float('inf'), 0
    train_losses, val_losses = [], []  # <--- loss tracking

    for ep in range(1, num_epochs + 1):
        model.train()
        tl = []
        for xb, yb in tr_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = nn.MSELoss()(model(xb), yb)
            loss.backward()
            opt.step()
            tl.append(loss.item())

        model.eval()
        vl = []
        with torch.no_grad():
            for xb, yb in vl_dl:
                xb, yb = xb.to(device), yb.to(device)
                val_loss = nn.MSELoss()(model(xb), yb).item()
                vl.append(val_loss)

        mean_tr_loss = np.mean(tl)
        mean_val_loss = np.mean(vl)
        train_losses.append(mean_tr_loss)
        val_losses.append(mean_val_loss)

        print(f"Ep {ep}/{num_epochs} tr={mean_tr_loss:.4f} vl={mean_val_loss:.4f}")

        if mean_val_loss < best_val:
            best_val = mean_val_loss
            wait = 0
            best_state = model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    print(f"Fold {fold} best val {best_val:.4f}")

    # Plot loss curves for this fold
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f"Fold {fold} Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.show()


#%% 10) Final train on full trainval
sx_final = StandardScaler().fit(X_trainval.reshape(-1, X_all.shape[2]))
sy_final = StandardScaler().fit(y_trainval)
Xtv = sx_final.transform(X_trainval.reshape(-1, X_all.shape[2])).reshape(-1,sequence_length,X_all.shape[2])
ytv = sy_final.transform(y_trainval)
tv_dl = DataLoader(
    TensorDataset(
        torch.tensor(Xtv, dtype=torch.float32),
        torch.tensor(ytv, dtype=torch.float32)
    ),
    batch_size=batch_size,
    shuffle=False
)

final_model = VolLSTM_MC(X_all.shape[2], hidden_size, num_layers, asset_count, dropout_prob).to(device)
opt = optim.Adam(final_model.parameters(), lr=learning_rate)
for ep in range(num_epochs):
    final_model.train()
    for xb,yb in tv_dl:
        xb,yb=xb.to(device), yb.to(device)
        opt.zero_grad()
        nn.MSELoss()(final_model(xb), yb).backward()
        opt.step()

#%% 11) Test evaluation with MC dropout
# Scale test inputs
Xts = sx_final.transform(X_test.reshape(-1, X_all.shape[2])).reshape(-1,sequence_length,X_all.shape[2])
y_true = y_test

# MC predictions
final_model.train()   # keep dropout active
mc_preds = []
with torch.no_grad():
    Xts_tensor = torch.tensor(Xts, dtype=torch.float32).to(device)
    for _ in range(mc_samples):
        mc_preds.append(final_model(Xts_tensor).cpu().numpy())
mc_preds = np.stack(mc_preds, axis=0)  # shape (mc_samples, test_len, asset_count)

# Compute mean & std
y_pred_mean = sy_final.inverse_transform(mc_preds.mean(axis=0))
y_pred_std  = mc_preds.std(axis=0) * sy_final.scale_  # approximate std in original scale

# Magnitude metrics
mae = mean_absolute_error(y_true, y_pred_mean, multioutput='raw_values')
rmse = np.sqrt(mean_squared_error(y_true, y_pred_mean, multioutput='raw_values'))

# Directional accuracy
dir_acc=[]
for j in range(asset_count):
    tdir = np.sign(y_true[1:,j]-y_true[:-1,j])
    pdirection = np.sign(y_pred_mean[1:,j]-y_pred_mean[:-1,j])
    dir_acc.append((tdir==pdirection).mean())
dir_acc = np.array(dir_acc)

# Print
print("\nTest Evaluation:")
for i,c in enumerate(close_cols):
    print(f"{c:10s} MAE={mae[i]:.4f} RMSE={rmse[i]:.4f} DirAcc={dir_acc[i]:.2%}")

# Uncertainty example: histogram of std for first asset
n_assets = y_pred_std.shape[1]
fig, axes = plt.subplots(n_assets, 1, figsize=(6, 3 * n_assets), sharex=True)

for i in range(n_assets):
    axes[i].hist(y_pred_std[:, i], bins=20)
    axes[i].set_title(f"MC Dropout Uncertainty (Asset {i+1})")

plt.tight_layout()
plt.show()

#%% 12) Covariance reconstruction & portfolio volatility on TEST
# 12.1) Build EWMA‐corr on common dates and reset index
corr_full = (
    log_returns
    .loc[common_idx]
    .ewm(span=horizon, adjust=False)
    .corr()
    .unstack(level=1)
    .reset_index(drop=True)
)
# 12.2) Align to test‐set range
start_idx = split_i + sequence_length
corr_test = corr_full.iloc[start_idx : start_idx + len(X_test)].reset_index(drop=True)

# 12.3) Reconstruct Σ and compute portfolio vols
cov_matrices_test = []
for i in range(len(X_test)):
    σ = y_pred_mean[i]  # your mean volatility forecast for test sample i
    R = corr_test.iloc[i].values.reshape(asset_count, asset_count)
    D = np.diag(σ)
    cov_matrices_test.append(D @ R @ D)

weights = np.ones(asset_count) / asset_count
vol_pred_port_test = [np.sqrt(weights @ C @ weights) for C in cov_matrices_test]
# true port vol from diagonal of realized vol
vol_true_port_test = [
    np.sqrt(weights @ np.diag(v**2) @ weights)
    for v in y_true
]

dates_test = common_idx[sequence_length + split_i :
                        sequence_length + split_i + len(vol_pred_port_test)]

# 12.4) Plot
plot_df_test = pd.DataFrame({
    "True Port Vol": vol_true_port_test,
    "Pred Port Vol": vol_pred_port_test
}, index=dates_test)

plt.figure(figsize=(12,5))
plot_df_test.plot(ax=plt.gca())
plt.title("Test‐Set Portfolio Volatility: True vs Predicted")
plt.xlabel("Date"); plt.ylabel("Volatility")
plt.show()

#%% → Paste here: Full‐set training & enhanced test evaluation

# 1) Final Training on full train‐val (you already have this; ensure `final_model` is trained)

# 2) MC‐Dropout Test Inference & Metrics
final_model.train()  # keep dropout active
# scale & reshape test set
Xts = sx_final.transform(X_test.reshape(-1, X_all.shape[2]))
Xts = Xts.reshape(-1, sequence_length, X_all.shape[2])
Xts_tensor = torch.tensor(Xts, dtype=torch.float32).to(device)

# draw MC samples
mc_preds = []
with torch.no_grad():
    for _ in range(mc_samples):
        mc_preds.append(final_model(Xts_tensor).cpu().numpy())
mc_preds = np.stack(mc_preds, axis=0)  # (mc_samples, N_test, assets)

# compute mean & std in original scale
y_pred_mean = sy_final.inverse_transform(mc_preds.mean(0))
y_pred_std  = mc_preds.std(0) * sy_final.scale_

# compute metrics
mae   = mean_absolute_error(y_test, y_pred_mean, multioutput='raw_values')
rmse  = np.sqrt(mean_squared_error(y_test, y_pred_mean, multioutput='raw_values'))
r2    = r2_score(y_test, y_pred_mean, multioutput='raw_values')
evs   = explained_variance_score(y_test, y_pred_mean, multioutput='raw_values')
dir_acc = []
for j in range(asset_count):
    tdir = np.sign(y_test[1:, j] - y_test[:-1, j])
    pdir = np.sign(y_pred_mean[1:, j] - y_pred_mean[:-1, j])
    dir_acc.append((tdir == pdir).mean())
dir_acc = np.array(dir_acc)

# print per‐asset
print("\n=== Test Metrics per Asset ===")
for i, name in enumerate(close_cols):
    print(f"{name:15s} MAE={mae[i]:.4f}, RMSE={rmse[i]:.4f}, "
          f"R2={r2[i]:.3f}, EVS={evs[i]:.3f}, "
          f"DirAcc={dir_acc[i]:.2%}, AvgUnc={y_pred_std[:,i].mean():.4f}")

# 3) Portfolio‐level volatility
# EWMA correlation aligned to test dates
corr_full = log_returns.loc[common_idx].ewm(span=horizon, adjust=False).corr().unstack()
start = sequence_length + split_i
corr_test = corr_full.iloc[start:start+len(Xts)].reset_index(drop=True)

vol_pred_port = []
for i in range(len(Xts)):
    σ = y_pred_mean[i]
    R = corr_test.iloc[i].values.reshape(asset_count, asset_count)
    C = np.diag(σ) @ R @ np.diag(σ)
    vol_pred_port.append(np.sqrt(weights @ C @ weights))
vol_true_port = [
    np.sqrt(weights @ np.diag(v**2) @ weights)
    for v in y_test
]

dates_test = common_idx[start:start+len(vol_pred_port)]
plot_df = pd.DataFrame({'TrueVol': vol_true_port, 'PredVol': vol_pred_port}, index=dates_test)
plt.figure(figsize=(10,4))
plot_df.plot(ax=plt.gca())
plt.title('Portfolio Volatility: True vs Predicted')
plt.show()

# 4) Filter out poor directional assets (DirAcc ≤50%)
bad = np.where(dir_acc <= 0.5)[0]
good = np.setdiff1d(np.arange(asset_count), bad)
print(f"\nExcluding {len(bad)} assets (DirAcc ≤50%):",
      [close_cols[i] for i in bad])

# recompute portfolio vol without bad assets
w_good = np.ones(len(good)) / len(good)
vol_pred_good, vol_true_good = [], []
for i in range(len(Xts)):
    σg = y_pred_mean[i][good]
    Rg = corr_test.iloc[i].values.reshape(asset_count, asset_count)[np.ix_(good, good)]
    Cg = np.diag(σg) @ Rg @ np.diag(σg)
    vol_pred_good.append(np.sqrt(w_good @ Cg @ w_good))
    vol_true_good.append(np.sqrt(w_good @ np.diag(y_test[i][good]**2) @ w_good))

plot_df2 = pd.DataFrame({'TrueVol': vol_true_good, 'PredVol': vol_pred_good}, index=dates_test)
plt.figure(figsize=(10,4))
plot_df2.plot(ax=plt.gca())
plt.title('Filtered Portfolio Vol (DirAcc>50%)')
plt.show()
