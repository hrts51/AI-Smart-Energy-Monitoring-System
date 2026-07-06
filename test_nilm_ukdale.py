"""
test_nilm_ukdale.py  v3
=======================
NILM model trained on REAL UK-DALE data.
Fixed: data leakage, resample sum, deprecated ffill, loss weights
"""
import matplotlib
matplotlib.use('TkAgg')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Bidirectional, LSTM, Dense,
                                     Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
from tensorflow.keras.regularizers import l2

# ============================================================
# 0. REPRODUCIBILITY
# ============================================================
os.environ['PYTHONHASHSEED'] = '42'
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

# ============================================================
# 1. DATA LOADING
# ============================================================
DATA_PATH  = "dataset/ukdale_clean.csv"
APPLIANCES = ["Fridge", "Washing_Machine", "Dishwasher", "TV", "Kettle"]

try:
    pivot_df = pd.read_csv(DATA_PATH, index_col="Time")
    pivot_df.index = pd.to_datetime(pivot_df.index, utc=True).tz_localize(None)
    pivot_df = pivot_df.sort_index()

    keep    = [a for a in APPLIANCES if a in pivot_df.columns]
    missing = [a for a in APPLIANCES if a not in pivot_df.columns]
    if missing:
        print(f"⚠️  Missing (will skip): {missing}")
    if not keep:
        print("❌ No appliance columns found."); exit()

    appliances = keep

    if "Aggregate" not in pivot_df.columns:
        pivot_df["Aggregate"] = pivot_df[appliances].sum(axis=1)

    pivot_df = pivot_df[appliances + ["Aggregate"]].fillna(0)

    print(f"✅ UK-DALE loaded | Shape: {pivot_df.shape}")
    print(f"   Appliances : {appliances}")
    print(f"   Date range : {pivot_df.index.min()} → {pivot_df.index.max()}")

    means  = pivot_df[appliances].mean()
    spread = means.max() - means.min()
    if spread < 0.00001:
        print("❌ Appliances have identical distributions."); exit()
    print(f"\n✅ Mean spread: {spread:.5f} kWh — real data confirmed!")
    print(f"   Appliance means:\n{means.round(6)}")

    # FIX: sum not mean — energy accumulates per hour
    pivot_df = pivot_df.resample('H').sum().fillna(0)
    print(f"\n✅ Resampled to hourly | Shape: {pivot_df.shape}")

except FileNotFoundError:
    print(f"❌ File not found: {DATA_PATH}")
    exit()
except Exception as e:
    print(f"❌ Error: {e}"); import traceback; traceback.print_exc(); exit()

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
pivot_df["hour"]       = pivot_df.index.hour
pivot_df["day"]        = pivot_df.index.dayofweek
pivot_df["month"]      = pivot_df.index.month
pivot_df["is_weekend"] = (pivot_df["day"] >= 5).astype(int)
pivot_df["prev_agg"]   = pivot_df["Aggregate"].shift(1).fillna(0)
pivot_df["prev_agg2"]  = pivot_df["Aggregate"].shift(2).fillna(0)
pivot_df["rolling3"]   = pivot_df["Aggregate"].rolling(3).mean().fillna(0)
pivot_df["rolling6"]   = pivot_df["Aggregate"].rolling(6).mean().fillna(0)
pivot_df["rolling24"]  = pivot_df["Aggregate"].rolling(24).mean().fillna(0)

pivot_df["hour_sin"] = np.sin(2 * np.pi * pivot_df["hour"] / 24)
pivot_df["hour_cos"] = np.cos(2 * np.pi * pivot_df["hour"] / 24)
pivot_df["day_sin"]  = np.sin(2 * np.pi * pivot_df["day"]  / 7)
pivot_df["day_cos"]  = np.cos(2 * np.pi * pivot_df["day"]  / 7)
pivot_df["mon_sin"]  = np.sin(2 * np.pi * pivot_df["month"] / 12)
pivot_df["mon_cos"]  = np.cos(2 * np.pi * pivot_df["month"] / 12)

features = [
    "Aggregate",
    "hour_sin", "hour_cos",
    "day_sin",  "day_cos",
    "mon_sin",  "mon_cos",
    "is_weekend",
    "prev_agg", "prev_agg2",
    "rolling3", "rolling6", "rolling24"
]

# ============================================================
# 3. TRAIN/TEST SPLIT FIRST — THEN FIT SCALERS
# FIX: no data leakage — scalers fit on train only
# ============================================================
WINDOW = 24

# Build sequences before scaling to get correct split point
# We scale after split

split_idx = int(len(pivot_df) * 0.8)
train_df  = pivot_df.iloc[:split_idx]
test_df   = pivot_df.iloc[split_idx:]

# Fit scalers on train only
scaler_X = MinMaxScaler()
scaler_X.fit(train_df[features].values)

X_train_scaled = scaler_X.transform(train_df[features].values)
X_test_scaled  = scaler_X.transform(test_df[features].values)

scalers_y = {}
y_train_list = []
y_test_list  = []

for app in appliances:
    s = MinMaxScaler()
    s.fit(train_df[[app]].values)
    scalers_y[app] = s
    y_train_list.append(s.transform(train_df[[app]].values))
    y_test_list.append(s.transform(test_df[[app]].values))

y_train_scaled = np.hstack(y_train_list)
y_test_scaled  = np.hstack(y_test_list)

print(f"\n✅ Features engineered | Total: {len(features)}")

# ============================================================
# 4. SEQUENCE CREATION
# ============================================================
def create_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)

X_train, y_train = create_sequences(X_train_scaled, y_train_scaled, WINDOW)
X_test,  y_test  = create_sequences(X_test_scaled,  y_test_scaled,  WINDOW)

print(f"✅ Sequences | Train: {X_train.shape} | Test: {X_test.shape}")

# ============================================================
# 5. MODEL
# ============================================================
REG        = l2(0.0001)
safe_names = [a.replace(' ', '_') for a in appliances]

inp = Input(shape=(WINDOW, len(features)))

x = Bidirectional(LSTM(128, return_sequences=True))(inp)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

x = Bidirectional(LSTM(64, return_sequences=False))(x)
x = BatchNormalization()(x)
x = Dropout(0.2)(x)

shared = Dense(64, activation='relu', kernel_regularizer=REG)(x)

# Per-appliance heads
outputs = []
for safe in safe_names:
    h   = Dense(32, activation='relu', kernel_regularizer=REG,
                name=f'{safe}_h1')(shared)
    h   = Dense(16, activation='relu', kernel_regularizer=REG,
                name=f'{safe}_h2')(h)
    out = Dense(1,  activation='sigmoid', name=safe)(h)
    outputs.append(out)

model = Model(inputs=inp, outputs=outputs)

# FIX: loss weights built from actual appliance list — no hardcoding
loss_weights = {safe: 1.0 for safe in safe_names}

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={safe: Huber(delta=1.0) for safe in safe_names},
    loss_weights=loss_weights
)

model.summary()
print(f"\n📐 Total parameters: {model.count_params():,}")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=15,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=6, min_lr=1e-6, verbose=1)
]

y_train_dict = {safe: y_train[:, i] for i, safe in enumerate(safe_names)}
y_test_dict  = {safe: y_test[:, i]  for i, safe in enumerate(safe_names)}

history = model.fit(
    X_train, y_train_dict,
    epochs=200,
    batch_size=64,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# 6. PREDICTIONS
# ============================================================
preds_list = model.predict(X_test)

pred_final = np.column_stack([
    scalers_y[app].inverse_transform(
        preds_list[i].reshape(-1, 1)
    ).ravel()
    for i, app in enumerate(appliances)
])

y_true = np.column_stack([
    scalers_y[app].inverse_transform(
        y_test[:, i].reshape(-1, 1)
    ).ravel()
    for i, app in enumerate(appliances)
])

pred_final = np.clip(pred_final, 0, None)

# Per-appliance metrics
print('\n📊 PER-APPLIANCE ACCURACY:')
print(f'{"Appliance":<20} | {"MAE":>8} | {"RMSE":>8} | {"% of avg":>8} | Result')
print('-' * 70)

pct_maes = []
for i, app in enumerate(appliances):
    mae  = mean_absolute_error(y_true[:, i], pred_final[:, i])
    rmse = np.sqrt(mean_squared_error(y_true[:, i], pred_final[:, i]))
    avg  = y_true[:, i].mean()
    pct  = (mae / avg * 100) if avg > 0 else 0
    pct_maes.append(pct / 100)
    good = '✅ Good' if pct < 20 else ('⚠️  OK' if pct < 40 else ('🔶 Fair' if pct < 80 else '❌ Poor'))
    print(f'{app:<20} | {mae:>8.4f} | {rmse:>8.4f} | {pct:>7.1f}% | {good}')

overall_mae = mean_absolute_error(y_true, pred_final)
print(f'\n🎯 OVERALL MAE: {overall_mae:.4f} kWh')

best_epoch  = np.argmin(history.history['val_loss'])
final_train = history.history['loss'][best_epoch]
final_val   = history.history['val_loss'][best_epoch]
gap         = final_val - final_train
print(f'\n📉 Train/Val gap at best epoch {best_epoch+1}: {gap:.4f}')
if gap < 0.01:
    print('   ✅ Good fit')
elif gap < 0.03:
    print('   ⚠️  Mild overfitting')
else:
    print('   ❌ Overfitting detected')

results_df  = pd.DataFrame(pred_final, columns=appliances,
                            index=pivot_df.index[-len(pred_final):])
daily_usage = results_df.resample('D').sum()

# ============================================================
# 7. REPORT
# ============================================================
print("\n" + "🌟"*25)
print("   SMART ENERGY AI ADVISOR — UK-DALE AUDIT REPORT v3")
print("🌟"*25)

print("\n📅 DAILY CONSUMPTION LOG — last 5 days (kWh):")
print(daily_usage.tail(5).round(4))

print("\n🚨 ANOMALY & DRIFT ANALYSIS (|Z| > 2.0):")
total_daily   = daily_usage.sum(axis=1)
anomaly_found = False

for i in range(7, len(total_daily)):
    prev_win = total_daily.iloc[i-7:i]
    mean_val = prev_win.mean()
    std_val  = prev_win.std() + 1e-6
    z        = (total_daily.iloc[i] - mean_val) / std_val
    if abs(z) > 2.0:
        anomaly_found = True
        direction = "📈 Spike" if z > 0 else "📉 Drop"
        print(f"⚠️  {direction} on {total_daily.index[i].date()} (Z: {z:.2f})")
        diff = (daily_usage.iloc[i] - daily_usage.iloc[i-7:i].mean()).sort_values(ascending=False)
        for app, val in diff.head(3).items():
            print(f"   🔸 {app:<20} | {'+' if val>=0 else ''}{val:.4f} kWh")

if not anomaly_found:
    print("   No significant anomalies detected.")

avg_daily  = total_daily.mean()
proj_units = avg_daily * 30

print(f"\n📐 Average daily usage  : {avg_daily:.4f} kWh (monitored appliances)")
print(f"🔮 Monthly projection   : {proj_units:.3f} kWh")
print(f"ℹ️  Note: UK-DALE measures selected appliances, not whole-home.")

print(f"\n📊 Monthly per-appliance estimate:")
monthly = daily_usage.mean() * 30
for app, kwh in monthly.sort_values(ascending=False).items():
    print(f"   {app:<20} : {kwh:.3f} kWh/month")

# UK electricity rates — pence per kWh (not Indian rupees)
# Average UK rate ~28p/kWh in 2024
uk_rate_pence = 28.0
bill_pence    = proj_units * uk_rate_pence
bill_pounds   = bill_pence / 100

print(f"\n💰 Estimated monthly bill : £{bill_pounds:.2f}")
print(f"   (UK rate: {uk_rate_pence}p/kWh — monitored appliances only)")

# Appliance health
print("\n🛠️  APPLIANCE HEALTH STATUS:")
h_mean = daily_usage.mean()
h_std  = daily_usage.std() + 1e-6
z_h    = (daily_usage.iloc[-1] - h_mean) / h_std
health = (100 - (z_h.abs() * 12)).clip(0, 100)

for app in appliances:
    filled = int(health[app] / 10)
    bar    = "█" * filled + "░" * (10 - filled)
    status = "✅ OK" if health[app] > 80 else ("⚠️  MONITOR" if health[app] > 50 else "❌ CHECK")
    print(f"{app:<20} | [{bar}] {health[app]:>5.1f}% | {status}")

print("\n💡 SMART USAGE TIPS (based on UK-DALE patterns):")
daily_per_app = daily_usage.mean()
top2 = daily_per_app.nlargest(2)
for app, kwh in top2.items():
    share = kwh / daily_per_app.sum() * 100
    print(f"   🔸 {app} uses most energy ({share:.1f}% of monitored total)")

print("\n" + "="*60)
print("☕ Kettle: Use full boils only.")
print("❄️  Fridge: Check door seal if usage spikes.")
print("🧺 Washing Machine: Cold wash saves ~40% energy.")
print("📺 TV: Unplug at night — standby wastes 1-5W.")
print("="*60)

os.makedirs("models", exist_ok=True)
export_df = daily_usage.copy()
export_df["Total_kWh"] = export_df.sum(axis=1)
export_df.to_csv("Final_Energy_Audit_UKDALE_v3.csv")
print("\n💾 Report saved as Final_Energy_Audit_UKDALE_v3.csv")

model.save("models/nilm_ukdale_v3.keras")
print("💾 Model saved as models/nilm_ukdale_v3.keras")

# ============================================================
# 8. VISUALIZATION
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Smart Energy AI Advisor — UK-DALE Dashboard v3 (Fixed)",
             fontsize=15, fontweight='bold')

ax = axes[0, 0]
ax.plot(history.history['loss'],     label='Train', color='royalblue',  linewidth=2)
ax.plot(history.history['val_loss'], label='Val',   color='darkorange', linewidth=2, linestyle='--')
ax.axvline(best_epoch, color='green', linestyle=':', linewidth=1.5,
           label=f'Best epoch {best_epoch+1}')
epochs = range(len(history.history['loss']))
ax.fill_between(epochs,
                history.history['loss'],
                history.history['val_loss'],
                alpha=0.1, color='red', label='Gap')
ax.set_title("Learning Curve")
ax.set_xlabel("Epoch")
ax.set_ylabel("Huber Loss")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

total_daily.plot(ax=axes[0, 1], marker='o', color='seagreen', markersize=2, linewidth=1.2)
mean_line = total_daily.mean()
axes[0, 1].axhline(mean_line, color='red', linestyle=':', linewidth=1.2,
                   label=f'Mean: {mean_line:.3f} kWh')
axes[0, 1].set_title("Daily Energy Trend — UK-DALE (kWh)")
axes[0, 1].set_ylabel("kWh / day")
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

maes   = [mean_absolute_error(y_true[:, i], pred_final[:, i])
          for i in range(len(appliances))]
avgs   = [y_true[:, i].mean() + 1e-6 for i in range(len(appliances))]
pcts   = [maes[i] / avgs[i] for i in range(len(appliances))]
colors = ['green' if p < 0.2 else ('orange' if p < 0.4 else ('gold' if p < 0.8 else 'red'))
          for p in pcts]
bars   = axes[0, 2].bar(appliances, maes, color=colors, edgecolor='white', linewidth=0.5)
for bar, mae, pct in zip(bars, maes, pcts):
    axes[0, 2].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(maes)*0.01,
                    f'{mae:.4f}\n({pct*100:.1f}%)',
                    ha='center', va='bottom', fontsize=7)
axes[0, 2].set_title("MAE per Appliance (kWh)")
axes[0, 2].set_ylabel("MAE (kWh)")
axes[0, 2].tick_params(axis='x', rotation=30)
axes[0, 2].grid(True, alpha=0.3, axis='y')

percent_share = (daily_usage.sum() / daily_usage.sum().sum()) * 100
axes[1, 0].pie(percent_share, labels=appliances, autopct='%1.1f%%',
               colors=plt.cm.tab10.colors[:len(appliances)],
               startangle=90, wedgeprops=dict(edgecolor='white', linewidth=1))
axes[1, 0].set_title("Energy Share per Appliance (UK-DALE)")

best_idx = int(np.argmin(pct_maes))
best_app = appliances[best_idx]
n_plot   = min(200, len(y_true))
axes[1, 1].plot(y_true[-n_plot:, best_idx],
                label=f'Actual {best_app}', alpha=0.85, linewidth=1.5)
axes[1, 1].plot(pred_final[-n_plot:, best_idx],
                label='Predicted', linestyle='--', alpha=0.85, linewidth=1.5)
axes[1, 1].set_title(f"Best: {best_app} ({pct_maes[best_idx]*100:.1f}% MAE)")
axes[1, 1].set_ylabel("kWh")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

worst_idx = int(np.argmax(pct_maes))
worst_app = appliances[worst_idx]
axes[1, 2].plot(y_true[-n_plot:, worst_idx],
                label=f'Actual {worst_app}', alpha=0.85,
                color='crimson', linewidth=1.5)
axes[1, 2].plot(pred_final[-n_plot:, worst_idx],
                label='Predicted', linestyle='--', alpha=0.85, linewidth=1.5)
axes[1, 2].set_title(f"Hardest: {worst_app} ({pct_maes[worst_idx]*100:.1f}% MAE)")
axes[1, 2].set_ylabel("kWh")
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("energy_dashboard_ukdale_v3.png", dpi=150, bbox_inches='tight')
plt.show()
print("📊 Dashboard saved as energy_dashboard_ukdale_v3.png")