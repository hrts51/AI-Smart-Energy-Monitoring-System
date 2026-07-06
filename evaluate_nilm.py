"""
evaluate_nilm.py
================
Loads saved predictions and computes ON/OFF F1 scores.
Run AFTER test_nilm_ukdale.py has completed.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# ── Load data and model ───────────────────────────────────────
DATA_PATH  = "dataset/ukdale_clean.csv"
MODEL_PATH = "models/nilm_ukdale_v3.keras"
APPLIANCES = ["Fridge", "Washing_Machine", "Dishwasher", "TV", "Kettle"]
WINDOW     = 24

print("Loading data...")
pivot_df = pd.read_csv(DATA_PATH, index_col="Time")
pivot_df.index = pd.to_datetime(pivot_df.index, utc=True).tz_localize(None)
pivot_df = pivot_df.sort_index()

appliances = [a for a in APPLIANCES if a in pivot_df.columns]

if "Aggregate" not in pivot_df.columns:
    pivot_df["Aggregate"] = pivot_df[appliances].sum(axis=1)

pivot_df = pivot_df[appliances + ["Aggregate"]].fillna(0)
pivot_df = pivot_df.resample('H').sum().fillna(0)

# Features
pivot_df["hour"]      = pivot_df.index.hour
pivot_df["day"]       = pivot_df.index.dayofweek
pivot_df["month"]     = pivot_df.index.month
pivot_df["is_weekend"]= (pivot_df["day"] >= 5).astype(int)
pivot_df["prev_agg"]  = pivot_df["Aggregate"].shift(1).fillna(0)
pivot_df["prev_agg2"] = pivot_df["Aggregate"].shift(2).fillna(0)
pivot_df["rolling3"]  = pivot_df["Aggregate"].rolling(3).mean().fillna(0)
pivot_df["rolling6"]  = pivot_df["Aggregate"].rolling(6).mean().fillna(0)
pivot_df["rolling24"] = pivot_df["Aggregate"].rolling(24).mean().fillna(0)
pivot_df["hour_sin"]  = np.sin(2 * np.pi * pivot_df["hour"] / 24)
pivot_df["hour_cos"]  = np.cos(2 * np.pi * pivot_df["hour"] / 24)
pivot_df["day_sin"]   = np.sin(2 * np.pi * pivot_df["day"]  / 7)
pivot_df["day_cos"]   = np.cos(2 * np.pi * pivot_df["day"]  / 7)
pivot_df["mon_sin"]   = np.sin(2 * np.pi * pivot_df["month"] / 12)
pivot_df["mon_cos"]   = np.cos(2 * np.pi * pivot_df["month"] / 12)

features = ["Aggregate","hour_sin","hour_cos","day_sin","day_cos",
            "mon_sin","mon_cos","is_weekend","prev_agg","prev_agg2",
            "rolling3","rolling6","rolling24"]

# Split — same as training
split_idx = int(len(pivot_df) * 0.8)
train_df  = pivot_df.iloc[:split_idx]
test_df   = pivot_df.iloc[split_idx:]

scaler_X = MinMaxScaler()
scaler_X.fit(train_df[features].values)
X_test_s = scaler_X.transform(test_df[features].values)

scalers_y = {}
y_test_list = []
for app in appliances:
    s = MinMaxScaler()
    s.fit(train_df[[app]].values)
    scalers_y[app] = s
    y_test_list.append(s.transform(test_df[[app]].values))

y_test_s = np.hstack(y_test_list)

def create_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)

X_test, y_test = create_sequences(X_test_s, y_test_s, WINDOW)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Running predictions...")
preds_list = model.predict(X_test, verbose=0)

pred_final = np.column_stack([
    scalers_y[app].inverse_transform(
        preds_list[i].reshape(-1,1)
    ).ravel()
    for i, app in enumerate(appliances)
])

y_true = np.column_stack([
    scalers_y[app].inverse_transform(
        y_test[:,i].reshape(-1,1)
    ).ravel()
    for i, app in enumerate(appliances)
])

pred_final = np.clip(pred_final, 0, None)

# ── ON/OFF threshold per appliance ───────────────────────────
# Threshold = 10% of each appliance's mean consumption
# If hourly kWh > threshold → ON, else OFF

print("\n" + "="*70)
print("📊 NILM EVALUATION — REGRESSION + ON/OFF STATE DETECTION")
print("="*70)

print(f"\n{'Appliance':<20} | {'MAE':>8} | {'F1':>6} | {'Precision':>9} | {'Recall':>6} | {'Threshold'}")
print("-"*75)

results = []
for i, app in enumerate(appliances):
    mae       = mean_absolute_error(y_true[:,i], pred_final[:,i])
    avg       = y_true[:,i].mean()

    # Threshold: 10% of mean consumption
    threshold = avg * 0.1
    if threshold == 0:
        threshold = 0.001

    y_on   = (y_true[:,i]   > threshold).astype(int)
    p_on   = (pred_final[:,i] > threshold).astype(int)

    f1   = f1_score(y_on, p_on, zero_division=0)
    prec = precision_score(y_on, p_on, zero_division=0)
    rec  = recall_score(y_on, p_on, zero_division=0)

    on_pct = y_on.mean() * 100

    status = "✅" if f1 > 0.7 else ("⚠️" if f1 > 0.5 else "❌")
    print(f"{app:<20} | {mae:>8.4f} | {f1:>6.3f} | {prec:>9.3f} | {rec:>6.3f} | {threshold:.4f} {status}")

    results.append({
        "Appliance": app,
        "MAE_kWh": round(mae, 4),
        "F1_Score": round(f1, 3),
        "Precision": round(prec, 3),
        "Recall": round(rec, 3),
        "ON_pct": round(on_pct, 1),
        "Threshold_kWh": round(threshold, 4)
    })

print("\n📌 F1 > 0.7 = Good | F1 > 0.5 = Acceptable | F1 < 0.5 = Poor")

# ── Naive baseline comparison ─────────────────────────────────
print("\n" + "="*70)
print("📊 BASELINE COMPARISON — BiLSTM vs Naive Proportional")
print("="*70)
print("Naive baseline: splits aggregate proportionally by historical mean")
print()

agg_test = y_true.sum(axis=1)  # total aggregate

print(f"{'Appliance':<20} | {'BiLSTM MAE':>10} | {'Naive MAE':>9} | {'Improvement'}")
print("-"*65)

for i, app in enumerate(appliances):
    bilstm_mae = mean_absolute_error(y_true[:,i], pred_final[:,i])

    # Naive: assign proportion of aggregate based on training mean
    train_mean  = train_df[app].mean()
    total_train = train_df[appliances].mean().sum()
    proportion  = train_mean / (total_train + 1e-6)
    naive_pred  = agg_test * proportion
    naive_mae   = mean_absolute_error(y_true[:,i], naive_pred)

    improvement = ((naive_mae - bilstm_mae) / naive_mae) * 100
    better = "✅ Better" if improvement > 0 else "❌ Worse"
    print(f"{app:<20} | {bilstm_mae:>10.4f} | {naive_mae:>9.4f} | {improvement:>+.1f}% {better}")

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv("nilm_evaluation_results.csv", index=False)
print(f"\n💾 Results saved to nilm_evaluation_results.csv")
print("\n✅ Evaluation complete.")

# ── Confusion Matrix Visualization ───────────────────────────
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import confusion_matrix
import seaborn as sns

print("\n📊 Generating confusion matrix visualizations...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("ON/OFF State Detection — Confusion Matrices\nNILM BiLSTM Model | UK-DALE Dataset",
             fontsize=14, fontweight='bold')

axes = axes.flatten()

for i, app in enumerate(appliances):
    avg       = y_true[:, i].mean()
    threshold = avg * 0.1 if avg > 0 else 0.001

    y_on = (y_true[:, i]    > threshold).astype(int)
    p_on = (pred_final[:, i] > threshold).astype(int)

    cm = confusion_matrix(y_on, p_on)
    f1 = f1_score(y_on, p_on, zero_division=0)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted OFF', 'Predicted ON'],
                yticklabels=['Actual OFF', 'Actual ON'],
                ax=axes[i], cbar=False)

    color = 'green' if f1 > 0.7 else ('orange' if f1 > 0.5 else 'red')
    axes[i].set_title(f"{app}\nF1 Score: {f1:.3f}",
                      fontsize=11, fontweight='bold', color=color)
    axes[i].set_xlabel("Predicted", fontsize=9)
    axes[i].set_ylabel("Actual", fontsize=9)

# Hide the 6th empty subplot
axes[5].set_visible(False)

plt.tight_layout()
plt.savefig("confusion_matrices_nilm.png", dpi=150, bbox_inches='tight')
print("💾 Confusion matrices saved as confusion_matrices_nilm.png")
plt.show()