import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Bidirectional, LSTM, Dense,
                                     Dropout, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber

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
DATA_PATH = "dataset/clean_nilm_drift_dataset.csv"

try:
    df = pd.read_csv(DATA_PATH)
    df["Time"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.sort_values("Time")
    df.rename(columns={
        "Energy Consumption (kWh)": "Energy",
        "Outdoor Temperature (°C)": "Temperature"
    }, inplace=True)

    pivot_df = df.pivot_table(
        index="Time", columns="Appliance Type",
        values="Energy", aggfunc="mean"
    ).fillna(0)
    pivot_df["Temperature"] = df.groupby("Time")["Temperature"].mean()

    # Resample to hourly
    pivot_df = pivot_df.resample('H').mean().fillna(method='ffill').fillna(0)

    appliances = [c for c in pivot_df.columns if c not in ["Time", "Temperature"]]
    pivot_df["Aggregate"] = pivot_df[appliances].sum(axis=1)

    print(f"✅ Data loaded | Shape: {pivot_df.shape}")
    print(f"   Appliances : {appliances}")
    print(f"   Date range : {pivot_df.index.min()} → {pivot_df.index.max()}")

    # ── Unit sanity check ──────────────────────────────────
    print("\n📊 RAW energy stats (before unit fix):")
    raw_means = pivot_df[appliances].mean()
    print(raw_means.round(4))

    if raw_means.mean() > 10:
        print("\n⚠️  Values look like Wh — dividing by 1000 to convert to kWh...")
        pivot_df[appliances] = pivot_df[appliances] / 1000
        pivot_df["Aggregate"] = pivot_df[appliances].sum(axis=1)
        print("✅ Unit conversion applied.")
    else:
        print("\n✅ Units look correct (mean < 10 kWh/hour).")

    print("\n📊 Hourly energy stats (kWh):")
    print(pivot_df[appliances].describe().round(4))

    assert pivot_df[appliances].mean().max() < 50, \
        "❌ Values still unrealistic — check raw data units."

except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit()

# ============================================================
# 2. FEATURE ENGINEERING
# ============================================================
pivot_df["hour"]       = pivot_df.index.hour
pivot_df["day"]        = pivot_df.index.dayofweek
pivot_df["month"]      = pivot_df.index.month
pivot_df["is_weekend"] = (pivot_df["day"] >= 5).astype(int)
pivot_df["prev_agg"]   = pivot_df["Aggregate"].shift(1).fillna(0)
pivot_df["prev_agg2"]  = pivot_df["Aggregate"].shift(2).fillna(0)
pivot_df["rolling3"]   = pivot_df["Aggregate"].rolling(3).mean().fillna(0)  # NEW
pivot_df["rolling6"]   = pivot_df["Aggregate"].rolling(6).mean().fillna(0)  # NEW

# Cyclical encodings
pivot_df["hour_sin"]  = np.sin(2 * np.pi * pivot_df["hour"] / 24)
pivot_df["hour_cos"]  = np.cos(2 * np.pi * pivot_df["hour"] / 24)
pivot_df["day_sin"]   = np.sin(2 * np.pi * pivot_df["day"] / 7)   # NEW
pivot_df["day_cos"]   = np.cos(2 * np.pi * pivot_df["day"] / 7)   # NEW

features = [
    "Aggregate", "Temperature",
    "hour_sin", "hour_cos",
    "day_sin",  "day_cos",
    "month", "is_weekend",
    "prev_agg", "prev_agg2",
    "rolling3", "rolling6"
]

scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(pivot_df[features].values)

# Separate scaler per appliance
scalers_y     = {}
y_scaled_list = []
for app in appliances:
    s   = MinMaxScaler()
    y_s = s.fit_transform(pivot_df[[app]].values)
    scalers_y[app] = s
    y_scaled_list.append(y_s)

y_scaled = np.hstack(y_scaled_list)
print(f"\n✅ Features engineered | Total features: {len(features)}")

# ============================================================
# 3. SEQUENCE CREATION — 24h window
# ============================================================
WINDOW = 24

def create_sequences(X, y, window):
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:i+window])
        ys.append(y[i+window])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, WINDOW)

# Chronological split — never shuffle time series
split   = int(len(X_seq) * 0.8)
X_train = X_seq[:split];  X_test = X_seq[split:]
y_train = y_seq[:split];  y_test = y_seq[split:]

print(f"✅ Sequences | Train: {X_train.shape} | Test: {X_test.shape}")

# ============================================================
# 4. MODEL — per-appliance output heads
#
# KEY FIX: Instead of one shared Dense(n_appliances) output,
# each appliance gets its own Dense head with 32→16→1 layers.
# This prevents gradient interference between appliances and
# breaks the "mean prediction collapse" causing uniform ~33% MAE.
# ============================================================
inp = Input(shape=(WINDOW, len(features)))

# Shared encoder
x = Bidirectional(LSTM(128, return_sequences=True))(inp)
x = BatchNormalization()(x)
x = Dropout(0.1)(x)

x = Bidirectional(LSTM(64, return_sequences=False))(x)   # Bidirectional on both layers
x = BatchNormalization()(x)
x = Dropout(0.1)(x)

shared = Dense(64, activation='relu')(x)

# Per-appliance specialised heads
outputs = []
for app in appliances:
    safe = app.replace(' ', '_')   # "Washing Machine" → "Washing_Machine"
    h   = Dense(32, activation='relu', name=f'{safe}_h1')(shared)
    h   = Dense(16, activation='relu', name=f'{safe}_h2')(h)
    out = Dense(1,  activation='sigmoid', name=safe)(h)
    outputs.append(out)

model = Model(inputs=inp, outputs=outputs)

# Separate Huber loss per appliance head
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss={app.replace(' ', '_'): Huber(delta=1.0) for app in appliances},
    loss_weights={app.replace(' ', '_'): 1.0 for app in appliances}
)

model.summary()
print(f"\n📐 Total parameters: {model.count_params():,}")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=12,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=5, min_lr=1e-6, verbose=1)
]

# Build per-appliance target dicts for training
y_train_dict = {app.replace(' ', '_'): y_train[:, i] for i, app in enumerate(appliances)}
y_test_dict  = {app.replace(' ', '_'): y_test[:, i]  for i, app in enumerate(appliances)}

history = model.fit(
    X_train, y_train_dict,
    epochs=200,
    batch_size=64,
    validation_split=0.15,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# 5. PREDICTIONS — per-appliance inverse transform
# ============================================================
# model.predict returns a list of arrays (one per head)
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

# ── Per-appliance metrics ──────────────────────────────────
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
    good = '✅ Good' if pct < 20 else ('⚠️  OK' if pct < 40 else '❌ Poor')
    print(f'{app:<20} | {mae:>8.4f} | {rmse:>8.4f} | {pct:>7.1f}% | {good}')

overall_mae = mean_absolute_error(y_true, pred_final)
print(f'\n🎯 OVERALL MAE: {overall_mae:.4f} kWh')

results_df  = pd.DataFrame(pred_final, columns=appliances,
                            index=pivot_df.index[-len(pred_final):])
daily_usage = results_df.resample('D').sum()

# ============================================================
# 6. REPORT
# ============================================================
print("\n" + "🌟"*25)
print("   SMART ENERGY AI ADVISOR - FINAL AUDIT REPORT v4")
print("🌟"*25)

print("\n📅 DAILY CONSUMPTION LOG — last 5 days (kWh):")
print(daily_usage.tail(5).round(3))

# ── Anomaly & drift detection ──────────────────────────────
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
            print(f"   🔸 {app:<18} | {'+' if val>=0 else ''}{val:.3f} kWh")

if not anomaly_found:
    print("   No significant anomalies detected (|Z| > 2.0 threshold)")

# ── Monthly forecast ───────────────────────────────────────
avg_daily  = total_daily.mean()
proj_units = avg_daily * 30

print(f"\n📐 Average daily usage : {avg_daily:.2f} kWh")
print(f"🔮 Monthly projection  : {proj_units:.2f} kWh")

if proj_units > 2000:
    print(f"⚠️  WARNING: {proj_units:.0f} kWh/month is extremely high.")
    print("   Check raw data units — may still be in Wh.")
elif proj_units < 1:
    print("⚠️  WARNING: Projected units < 1 kWh/month. Data may be corrupt.")
else:
    if proj_units <= 100:
        bill = proj_units * 5.0
    elif proj_units <= 300:
        bill = 100*5.0 + (proj_units - 100)*7.0
    elif proj_units <= 500:
        bill = 100*5.0 + 200*7.0 + (proj_units - 300)*9.0
    else:
        bill = 100*5.0 + 200*7.0 + 200*9.0 + (proj_units - 500)*10.0

    print(f"💰 Estimated monthly bill : ₹{bill:.2f}")
    print("   (Slab rates: ₹5/7/9/10 per unit)")

# ── Appliance health — abs(z) fix ─────────────────────────
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

# ── Appliance-specific tips ────────────────────────────────
print("\n💡 SMART USAGE TIPS:")
daily_per_app = daily_usage.mean()
top2 = daily_per_app.nlargest(2)
for app, kwh in top2.items():
    share = kwh / daily_per_app.sum() * 100
    print(f"   🔸 {app} uses the most energy ({share:.1f}% of total, ~{kwh:.2f} kWh/day)")

print("\n" + "="*60)
print("🕒 Shift heavy loads to off-peak (10 PM – 6 AM).")
print("☀️  Use 2 PM – 4 PM solar window for daytime loads.")
print("❄️  Set AC to 24°C — each degree below costs ~6% more.")
print("="*60)

# Export
export_df = daily_usage.copy()
export_df["Total_kWh"] = export_df.sum(axis=1)
export_df.to_csv("Final_Energy_Audit_Report_v4.csv")
print("\n💾 Report saved as Final_Energy_Audit_Report_v4.csv")

# ============================================================
# 7. VISUALIZATION
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Smart Energy AI Advisor — Dashboard v4 (Per-Appliance Heads)",
             fontsize=16, fontweight='bold')

# ── Learning curve ─────────────────────────────────────────
axes[0, 0].plot(history.history['loss'],     label='Train',
                color='royalblue',  linewidth=2)
axes[0, 0].plot(history.history['val_loss'], label='Val',
                color='darkorange', linewidth=2, linestyle='--')
best_epoch = np.argmin(history.history['val_loss'])
axes[0, 0].axvline(best_epoch, color='green', linestyle=':', linewidth=1.2,
                   label=f'Best epoch {best_epoch+1}')
axes[0, 0].set_title("Learning Curve")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Huber Loss")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# ── Daily energy trend ─────────────────────────────────────
total_daily.plot(ax=axes[0, 1], marker='o', color='seagreen',
                 markersize=3, linewidth=1.5)
mean_line = total_daily.mean()
axes[0, 1].axhline(mean_line, color='red', linestyle=':', linewidth=1.2,
                   label=f'Mean: {mean_line:.1f} kWh')
axes[0, 1].set_title("Daily Energy Trend (kWh)")
axes[0, 1].set_ylabel("kWh / day")
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# ── MAE bar — now each bar should differ ──────────────────
maes   = [mean_absolute_error(y_true[:, i], pred_final[:, i])
          for i in range(len(appliances))]
avgs   = [y_true[:, i].mean() + 1e-6 for i in range(len(appliances))]
pcts   = [maes[i] / avgs[i] for i in range(len(appliances))]
colors = ['green' if p < 0.2 else ('orange' if p < 0.4 else 'red') for p in pcts]
bars   = axes[0, 2].bar(appliances, maes, color=colors,
                         edgecolor='white', linewidth=0.5)
for bar, mae, pct in zip(bars, maes, pcts):
    axes[0, 2].text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.001,
                    f'{mae:.4f}\n({pct*100:.1f}%)',
                    ha='center', va='bottom', fontsize=7.5)
axes[0, 2].set_title("MAE per Appliance (kWh)\n(bars should now differ)")
axes[0, 2].set_ylabel("MAE (kWh)")
axes[0, 2].tick_params(axis='x', rotation=30)
axes[0, 2].grid(True, alpha=0.3, axis='y')

# ── Pie chart ──────────────────────────────────────────────
percent_share = (daily_usage.sum() / daily_usage.sum().sum()) * 100
axes[1, 0].pie(percent_share, labels=appliances, autopct='%1.1f%%',
               colors=plt.cm.tab10.colors[:len(appliances)],
               startangle=90,
               wedgeprops=dict(edgecolor='white', linewidth=1))
axes[1, 0].set_title("Energy Share per Appliance")

# ── Best predicted appliance ───────────────────────────────
best_idx = int(np.argmin(pct_maes))
best_app = appliances[best_idx]
n_plot   = min(200, len(y_true))
axes[1, 1].plot(y_true[-n_plot:, best_idx],
                label=f'Actual {best_app}', alpha=0.85, linewidth=1.5)
axes[1, 1].plot(pred_final[-n_plot:, best_idx],
                label='Predicted', linestyle='--', alpha=0.85, linewidth=1.5)
axes[1, 1].set_title(f"Best Prediction: {best_app} ({pct_maes[best_idx]*100:.1f}% MAE)")
axes[1, 1].set_ylabel("kWh")
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# ── Worst predicted appliance ──────────────────────────────
worst_idx = int(np.argmax(pct_maes))
worst_app = appliances[worst_idx]
axes[1, 2].plot(y_true[-n_plot:, worst_idx],
                label=f'Actual {worst_app}', alpha=0.85,
                color='crimson', linewidth=1.5)
axes[1, 2].plot(pred_final[-n_plot:, worst_idx],
                label='Predicted', linestyle='--', alpha=0.85, linewidth=1.5)
axes[1, 2].set_title(f"Worst Prediction: {worst_app} ({pct_maes[worst_idx]*100:.1f}% MAE)")
axes[1, 2].set_ylabel("kWh")
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("energy_dashboard_v4.png", dpi=150, bbox_inches='tight')
plt.show()
print("📊 Dashboard saved as energy_dashboard_v4.png")