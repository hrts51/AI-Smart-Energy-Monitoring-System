"""
dashboard.py
============
Streamlit dashboard for NILM Smart Energy Monitoring System
Run with: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Energy Monitor — NILM",
    page_icon="⚡",
    layout="wide"
)

# ── Constants ─────────────────────────────────────────────────
DATA_PATH  = "dataset/ukdale_clean.csv"
MODEL_PATH = "models/nilm_ukdale_v3.keras"
APPLIANCES = ["Fridge", "Washing_Machine", "Dishwasher", "TV", "Kettle"]
WINDOW     = 24
FEATURES   = ["Aggregate","hour_sin","hour_cos","day_sin","day_cos",
               "mon_sin","mon_cos","is_weekend","prev_agg","prev_agg2",
               "rolling3","rolling6","rolling24"]

# ── Load data and model (cached) ──────────────────────────────
@st.cache_data
def load_and_predict():
    pivot_df = pd.read_csv(DATA_PATH, index_col="Time")
    pivot_df.index = pd.to_datetime(pivot_df.index, utc=True).tz_localize(None)
    pivot_df = pivot_df.sort_index()

    appliances = [a for a in APPLIANCES if a in pivot_df.columns]

    if "Aggregate" not in pivot_df.columns:
        pivot_df["Aggregate"] = pivot_df[appliances].sum(axis=1)

    pivot_df = pivot_df[appliances + ["Aggregate"]].fillna(0)
    pivot_df = pivot_df.resample('H').sum().fillna(0)

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

    split_idx = int(len(pivot_df) * 0.8)
    train_df  = pivot_df.iloc[:split_idx]
    test_df   = pivot_df.iloc[split_idx:]

    scaler_X = MinMaxScaler()
    scaler_X.fit(train_df[FEATURES].values)
    X_test_s = scaler_X.transform(test_df[FEATURES].values)

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

    model = tf.keras.models.load_model(MODEL_PATH)
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

    results_df = pd.DataFrame(
        pred_final, columns=appliances,
        index=test_df.index[-len(pred_final):]
    )
    true_df = pd.DataFrame(
        y_true, columns=appliances,
        index=test_df.index[-len(y_true):]
    )

    return results_df, true_df, appliances, pivot_df

# ── Main app ──────────────────────────────────────────────────
st.title("⚡ Smart Energy Monitoring System")
st.markdown("**NILM-based Appliance Disaggregation — UK-DALE Dataset**")
st.markdown("---")

with st.spinner("Loading model and running predictions..."):
    pred_df, true_df, appliances, pivot_df = load_and_predict()

daily_pred = pred_df.resample('D').sum()
daily_true = true_df.resample('D').sum()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Live Monitor",
    "🔍 Appliance Analysis",
    "🚨 Anomaly Detection",
    "💰 Bill Estimator",
    "🔴 Live Simulation"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITOR
# ══════════════════════════════════════════════════════════════
with tab1:
    st.header("📊 Energy Consumption Overview")

    # Top metrics
    col1, col2, col3, col4 = st.columns(4)
    total_daily = daily_pred.sum(axis=1)

    with col1:
        st.metric("Avg Daily Usage", f"{total_daily.mean():.2f} kWh")
    with col2:
        st.metric("Peak Daily Usage", f"{total_daily.max():.2f} kWh")
    with col3:
        st.metric("Monthly Projection", f"{total_daily.mean()*30:.1f} kWh")
    with col4:
        uk_bill = total_daily.mean() * 30 * 0.28
        st.metric("Est. Monthly Bill", f"£{uk_bill:.2f}")

    st.markdown("---")

    # Daily trend
    st.subheader("Daily Total Energy — Predicted vs Actual")
    fig, ax = plt.subplots(figsize=(14, 4))
    daily_true.sum(axis=1).plot(ax=ax, label='Actual', color='steelblue',
                                 linewidth=1.5, alpha=0.8)
    daily_pred.sum(axis=1).plot(ax=ax, label='Predicted', color='darkorange',
                                 linewidth=1.5, linestyle='--', alpha=0.8)
    ax.set_ylabel("kWh / day")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    # Energy share pie
    st.subheader("Energy Share per Appliance")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 6))
        shares = daily_pred.sum()
        ax.pie(shares, labels=appliances, autopct='%1.1f%%',
               colors=plt.cm.tab10.colors[:len(appliances)],
               startangle=90, wedgeprops=dict(edgecolor='white'))
        ax.set_title("Predicted Energy Share")
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Per-Appliance Daily Average")
        summary = pd.DataFrame({
            "Appliance": appliances,
            "Avg Daily kWh": [daily_pred[a].mean() for a in appliances],
            "Monthly kWh": [daily_pred[a].mean()*30 for a in appliances]
        }).set_index("Appliance")
        st.dataframe(summary.round(3), use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — APPLIANCE ANALYSIS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.header("🔍 Per-Appliance Prediction Analysis")

    selected = st.selectbox("Select Appliance", appliances)

    col1, col2, col3 = st.columns(3)
    mae  = mean_absolute_error(true_df[selected], pred_df[selected])
    avg  = true_df[selected].mean()
    pct  = (mae/avg*100) if avg > 0 else 0

    thr  = avg * 0.1 if avg > 0 else 0.001
    y_on = (true_df[selected].values > thr).astype(int)
    p_on = (pred_df[selected].values  > thr).astype(int)
    f1   = f1_score(y_on, p_on, zero_division=0)

    with col1:
        st.metric("MAE", f"{mae:.4f} kWh")
    with col2:
        st.metric("MAE %", f"{pct:.1f}%")
    with col3:
        st.metric("F1 Score (ON/OFF)", f"{f1:.3f}")

    # Actual vs predicted plot
    st.subheader(f"{selected} — Actual vs Predicted (last 200 hours)")
    n = min(200, len(pred_df))
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(true_df[selected].values[-n:], label='Actual',
            color='steelblue', linewidth=1.5, alpha=0.85)
    ax.plot(pred_df[selected].values[-n:], label='Predicted',
            color='darkorange', linewidth=1.5, linestyle='--', alpha=0.85)
    ax.set_ylabel("kWh")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    # Full F1 table
    st.subheader("All Appliances — F1 Score Summary")
    f1_data = []
    for app in appliances:
        m    = mean_absolute_error(true_df[app], pred_df[app])
        a    = true_df[app].mean()
        p    = (m/a*100) if a > 0 else 0
        thr  = a * 0.1 if a > 0 else 0.001
        yo   = (true_df[app].values > thr).astype(int)
        po   = (pred_df[app].values  > thr).astype(int)
        f    = f1_score(yo, po, zero_division=0)
        pr   = precision_score(yo, po, zero_division=0)
        re   = recall_score(yo, po, zero_division=0)
        f1_data.append({
            "Appliance": app,
            "MAE (kWh)": round(m, 4),
            "MAE %": round(p, 1),
            "F1 Score": round(f, 3),
            "Precision": round(pr, 3),
            "Recall": round(re, 3),
            "Status": "✅ Good" if f > 0.7 else ("⚠️ OK" if f > 0.5 else "❌ Poor")
        })
    st.dataframe(pd.DataFrame(f1_data).set_index("Appliance"),
                 use_container_width=True)
    # Confusion Matrices
    st.subheader("ON/OFF Detection — Confusion Matrices")
    import seaborn as sns

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("ON/OFF State Detection — Confusion Matrices",
                 fontsize=13, fontweight='bold')
    axes = axes.flatten()

    for i, app in enumerate(appliances):
        avg       = true_df[app].mean()
        threshold = avg * 0.1 if avg > 0 else 0.001
        y_on = (true_df[app].values    > threshold).astype(int)
        p_on = (pred_df[app].values > threshold).astype(int)

        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_on, p_on)
        f1 = f1_score(y_on, p_on, zero_division=0)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted OFF', 'Predicted ON'],
                    yticklabels=['Actual OFF', 'Actual ON'],
                    ax=axes[i], cbar=False)

        color = 'green' if f1 > 0.7 else ('orange' if f1 > 0.5 else 'red')
        axes[i].set_title(f"{app}\nF1: {f1:.3f}",
                          fontsize=10, fontweight='bold', color=color)
        axes[i].set_xlabel("Predicted", fontsize=8)
        axes[i].set_ylabel("Actual", fontsize=8)

    axes[5].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
# ══════════════════════════════════════════════════════════════
# TAB 3 — ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════
with tab3:
    st.header("🚨 Anomaly & Drift Detection")

    total_daily = daily_pred.sum(axis=1)
    anomalies = []

    for i in range(7, len(total_daily)):
        prev = total_daily.iloc[i-7:i]
        z    = (total_daily.iloc[i] - prev.mean()) / (prev.std() + 1e-6)
        if abs(z) > 2.0:
            diff = (daily_pred.iloc[i] - daily_pred.iloc[i-7:i].mean())
            top  = diff.abs().nlargest(2)
            anomalies.append({
                "Date": total_daily.index[i].date(),
                "Z-Score": round(z, 2),
                "Type": "📈 Spike" if z > 0 else "📉 Drop",
                "Main Cause": top.index[0] if len(top) > 0 else "Unknown",
                "kWh Diff": round(top.iloc[0], 4) if len(top) > 0 else 0
            })

    if anomalies:
        st.warning(f"Found {len(anomalies)} anomalies in test period")
        st.dataframe(pd.DataFrame(anomalies), use_container_width=True)
    else:
        st.success("No significant anomalies detected (|Z| > 2.0 threshold)")

    # Z-score plot
    st.subheader("Daily Z-Score (rolling 7-day window)")
    z_scores = []
    dates    = []
    for i in range(7, len(total_daily)):
        prev = total_daily.iloc[i-7:i]
        z    = (total_daily.iloc[i] - prev.mean()) / (prev.std() + 1e-6)
        z_scores.append(z)
        dates.append(total_daily.index[i])

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates, z_scores, color='steelblue', linewidth=1.2)
    ax.axhline(2,  color='red',    linestyle='--', linewidth=1, label='Threshold +2')
    ax.axhline(-2, color='orange', linestyle='--', linewidth=1, label='Threshold -2')
    ax.fill_between(dates, z_scores,
                    where=[abs(z) > 2 for z in z_scores],
                    color='red', alpha=0.3, label='Anomaly')
    ax.set_ylabel("Z-Score")
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 4 — BILL ESTIMATOR
# ══════════════════════════════════════════════════════════════
with tab4:
    st.header("💰 Energy Bill Estimator")

    st.subheader("Electricity Rate Settings")

    currency = st.radio("Select Currency", ["🇬🇧 UK (Pence/kWh)", "🇮🇳 India (₹/kWh)"])

    if currency == "🇬🇧 UK (Pence/kWh)":
        rate    = st.slider("Rate (pence per kWh)", 20, 40, 28)
        symbol  = "£"
        divisor = 100
        note    = "UK average rate: 28p/kWh (2024)"
    else:
        rate    = st.slider("Rate (₹ per kWh)", 3, 12, 7)
        symbol  = "₹"
        divisor = 1
        note    = "Indian slab rates: ₹5-10/kWh depending on consumption"

    st.caption(note)

    monthly_kwh   = daily_pred.mean() * 30
    total_monthly = monthly_kwh.sum()
    bill          = total_monthly * rate / divisor

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Monthly kWh", f"{total_monthly:.2f} kWh")
    with col2:
        st.metric("Estimated Monthly Bill", f"{symbol}{bill:.2f}")

    st.subheader("Per-Appliance Monthly Cost Breakdown")
    bill_data = []
    for app in appliances:
        kwh  = monthly_kwh[app]
        cost = kwh * rate / divisor
        bill_data.append({
            "Appliance": app,
            "Monthly kWh": round(kwh, 3),
            f"Monthly Cost ({symbol})": round(cost, 2),
            "Share %": round(kwh/total_monthly*100, 1)
        })

    bill_df = pd.DataFrame(bill_data).set_index("Appliance")
    st.dataframe(bill_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = plt.cm.tab10.colors[:len(appliances)]
    ax.bar(appliances, bill_df[f"Monthly Cost ({symbol})"], color=colors)
    ax.set_ylabel(f"Monthly Cost ({symbol})")
    ax.set_title("Per-Appliance Monthly Cost")
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3, axis='y')
    st.pyplot(fig)
    plt.close()

    st.subheader("💡 Energy Saving Tips")
    st.info("☕ Kettle: Only boil what you need — saves ~20% kettle energy")
    st.info("❄️ Fridge: Check door seal if usage spikes unexpectedly")
    st.info("🧺 Washing Machine: Cold wash (30°C) saves ~40% vs 60°C")
    st.info("📺 TV: Unplug at night — standby uses 1-5W continuously")
# ══════════════════════════════════════════════════════════════
# TAB 5 — LIVE SIMULATION
# ══════════════════════════════════════════════════════════════
with tab5:
    st.header("🔴 Real-Time Energy Monitoring Simulation")
    st.markdown("Replays UK-DALE test data hour by hour — simulating a live monitoring system")

    # Controls
    col1, col2, col3 = st.columns(3)
    with col1:
        speed = st.slider("Simulation Speed (seconds per tick)", 0.2, 2.0, 0.5)
    with col2:
        n_hours = st.slider("Hours to simulate", 24, 168, 72)
    with col3:
        start_sim = st.button("▶ Start Simulation", type="primary")

    st.markdown("---")

    # Placeholders for live updating elements
    status_placeholder    = st.empty()
    clock_placeholder     = st.empty()
    metrics_placeholder   = st.empty()
    chart_placeholder     = st.empty()
    history_placeholder   = st.empty()

    if start_sim:
        import time

        # Use last n_hours of test data
        sim_pred = pred_df.iloc[-n_hours:].reset_index()
        sim_true = true_df.iloc[-n_hours:].reset_index()

        running_total = {app: 0.0 for app in appliances}
        history_data  = []

        for tick in range(len(sim_pred)):
            current_time  = sim_pred.iloc[tick]["index"] if "index" in sim_pred.columns else sim_pred.index[tick]
            current_pred  = sim_pred.iloc[tick][appliances]
            current_true  = sim_true.iloc[tick][appliances]

            # Update running totals
            for app in appliances:
                running_total[app] += float(current_pred[app])

            # Determine ON/OFF status
            on_off = {}
            for app in appliances:
                avg_val   = true_df[app].mean()
                threshold = avg_val * 0.1 if avg_val > 0 else 0.001
                on_off[app] = float(current_pred[app]) > threshold

            # Status bar
            status_placeholder.markdown(
                f"### 🕐 Simulating: Hour {tick+1} of {len(sim_pred)}"
            )

            # Clock
            try:
                clock_placeholder.info(f"📅 Timestamp: {sim_pred.iloc[tick].get('index', tick)}")
            except:
                clock_placeholder.info(f"📅 Hour: {tick+1}")

            # Metrics row — running totals
            cols = metrics_placeholder.columns(len(appliances))
            for i, app in enumerate(appliances):
                status_icon = "🟢" if on_off[app] else "🔴"
                cols[i].metric(
                    label=f"{status_icon} {app}",
                    value=f"{float(current_pred[app]):.3f} kWh",
                    delta=f"Total: {running_total[app]:.2f} kWh"
                )

            # Bar chart — current hour consumption
            fig, ax = plt.subplots(figsize=(10, 4))
            colors = ['green' if on_off[app] else 'lightgray' for app in appliances]
            bars = ax.bar(appliances, [float(current_pred[app]) for app in appliances],
                         color=colors, edgecolor='white', linewidth=0.5)
            ax.set_ylabel("kWh (this hour)")
            ax.set_title(f"Current Hour — Appliance Consumption (🟢 = ON, ⬜ = OFF)")
            ax.tick_params(axis='x', rotation=30)
            ax.grid(True, alpha=0.3, axis='y')
            for bar, app in zip(bars, appliances):
                ax.text(bar.get_x() + bar.get_width()/2,
                       bar.get_height() + 0.001,
                       f"{'ON' if on_off[app] else 'OFF'}",
                       ha='center', va='bottom', fontsize=9, fontweight='bold')
            chart_placeholder.pyplot(fig)
            plt.close()

            # Rolling history table
            history_data.append({
                "Hour": tick + 1,
                **{app: round(float(current_pred[app]), 4) for app in appliances},
                "Total": round(sum(float(current_pred[app]) for app in appliances), 4)
            })

            if len(history_data) > 10:
                history_data = history_data[-10:]

            history_placeholder.dataframe(
                pd.DataFrame(history_data).set_index("Hour"),
                use_container_width=True
            )

            time.sleep(speed)

        # Simulation complete
        status_placeholder.success(f"✅ Simulation complete — {len(sim_pred)} hours replayed")
        total_kwh = sum(running_total.values())
        st.metric("Total Energy Consumed (simulation)", f"{total_kwh:.2f} kWh")

        # Final summary
        st.subheader("Simulation Summary")
        summary_data = [{
            "Appliance": app,
            "Total kWh": round(running_total[app], 3),
            "Share %": round(running_total[app]/total_kwh*100, 1)
        } for app in appliances]
        st.dataframe(pd.DataFrame(summary_data).set_index("Appliance"),
                    use_container_width=True)