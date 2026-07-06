"""
make_ukdale_dataset.py  (fixed - sum not mean)
===============================================================
Run this ONCE to generate dataset/ukdale_clean.csv
Then run test_nilm_ukdale.py to train the model.
"""

import pandas as pd
import numpy as np

H5_PATH       = "dataset/ukdale.h5"
OUT_PATH      = "dataset/ukdale_clean.csv"
RESAMPLE_FREQ = "1min"

AGGREGATE_METER = "/building1/elec/meter1"

APPLIANCE_METERS = {
    "Fridge"          : "/building1/elec/meter5",
    "Washing_Machine" : "/building1/elec/meter6",
    "Dishwasher"      : "/building1/elec/meter7",
    "TV"              : "/building1/elec/meter9",
    "Microwave"       : "/building1/elec/meter11",
    "Kettle"          : "/building1/elec/meter12",
}

def extract_power_series(df, label):
    if isinstance(df, pd.Series):
        s = df.copy()
        s.name = label
        return s
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join([str(c) for c in col]).strip('_')
                      for col in df.columns]
    if 'power_active' in df.columns:
        s = df['power_active']
    elif 'power_apparent' in df.columns:
        s = df['power_apparent']
    else:
        s = df.iloc[:, 0]
    s = s.copy()
    s.name = label
    return s


def load_meter(store, key, label):
    try:
        raw = store[key]
        s   = extract_power_series(raw, label)
        s.index = pd.to_datetime(s.index)
        s       = pd.to_numeric(s, errors='coerce')
        # FIX: sum not mean — captures total energy per minute
        s       = s.resample(RESAMPLE_FREQ).sum()
        s       = s.ffill().fillna(0)
        s       = s.clip(lower=0)
        s.name  = label
        mean_w = s.mean()
        max_w  = s.max()
        print(f"  ✅ {label:<20} | {len(s):>8,} rows | "
              f"mean={mean_w:>7.2f}  max={max_w:>8.2f}")
        return s
    except Exception as e:
        print(f"  ❌ {label:<20} | Error: {e}")
        return None


try:
    store = pd.HDFStore(H5_PATH, mode='r')
    print(f"\n📥 Loading meters from UK-DALE (building1)...\n")

    agg = load_meter(store, AGGREGATE_METER, "Aggregate")

    series_list = []
    for label, key in APPLIANCE_METERS.items():
        s = load_meter(store, key, label)
        if s is not None:
            series_list.append(s)

    store.close()

    if not series_list:
        print("\n❌ No meters loaded successfully.")
        exit()

    print(f"\n✅ Loaded {len(series_list)} appliances successfully.")

    print("\n🔗 Aligning time indices and merging...")
    all_series = ([agg] if agg is not None else []) + series_list
    combined   = pd.concat(all_series, axis=1).sort_index()
    combined   = combined.ffill().fillna(0)

    print(f"   Combined shape before trim: {combined.shape}")

    if agg is not None:
        combined = combined[combined["Aggregate"] > 0]
        print(f"   Combined shape after gap removal: {combined.shape}")

    # W → kWh conversion (1-min intervals)
    kwh_factor = 1 / 60 / 1000
    combined   = combined * kwh_factor

    appliance_cols = [s.name for s in series_list]

    print(f"\n📊 Per-appliance kWh stats (1-minute resolution):")
    print(combined[appliance_cols].describe().round(6))

    means  = combined[appliance_cols].mean()
    spread = means.max() - means.min()
    print(f"\n📊 Appliance mean kWh/min:")
    for app, m in means.items():
        print(f"   {app:<20} : {m:.6f} kWh/min  (~{m*60*1000:.1f} W avg)")

    if spread < 0.000001:
        print("\n⚠️  WARNING: Appliances still have very similar means.")
    else:
        print(f"\n✅ Mean spread = {spread:.6f} kWh — appliances are distinct!")

    combined.index.name = "Time"
    combined.to_csv(OUT_PATH)

    print(f"\n💾 Saved to {OUT_PATH}")
    print(f"   Rows: {len(combined):,} | Cols: {list(combined.columns)}")
    print(f"\n✅ Done! Now run: python test_nilm_ukdale.py")

except FileNotFoundError:
    print(f"❌ File not found: {H5_PATH}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()