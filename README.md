# Smart Energy Monitoring System using NILM

## B.Tech Major Project
Non-Intrusive Load Monitoring (NILM) with BiLSTM Neural Network

---

## Project Overview
This system disaggregates total household energy consumption into 
per-appliance consumption using a Bidirectional LSTM (BiLSTM) model 
trained on the UK-DALE dataset. It detects which appliances are ON/OFF, 
estimates energy bills, and flags consumption anomalies.

---

## Project Structure
NILM_Project/
├── dataset/
│   ├── ukdale.h5                    # Raw UK-DALE data
│   ├── ukdale_clean.csv             # Processed dataset
│   └── make_ukdale_dataset.py       # Data preprocessing script
├── models/
│   └── nilm_ukdale_v3.keras         # Trained BiLSTM model
├── test_nilm_ukdale.py              # Model training script
├── evaluate_nilm.py                 # F1 + baseline evaluation
├── dashboard.py                     # Streamlit monitoring dashboard
├── requirements.txt                 # Dependencies
└── README.md                        # This file
---

## How to Run

### Step 1 — Install dependencies
pip install -r requirements.txt

### Step 2 — Prepare dataset (run once)
python3 dataset/make_ukdale_dataset.py

### Step 3 — Train the model
python3 test_nilm_ukdale.py

### Step 4 — Evaluate (F1 + baseline)
python3 evaluate_nilm.py

### Step 5 — Launch dashboard
streamlit run dashboard.py
Open browser at http://localhost:8501

---

## Dataset
- UK-DALE (UK Domestic Appliance-Level Electricity)
- Source: Jack Kelly, William Knottenbelt, Imperial College London
- Building 1, 5 appliances: Fridge, Washing Machine, Dishwasher, TV, Kettle
- ~2.3 million readings at 1-minute resolution
- Resampled to hourly for training

---

## Model Architecture
- Type: Bidirectional LSTM (BiLSTM)
- Input: 24-hour window of aggregate consumption + temporal features
- Output: Per-appliance energy consumption (5 separate heads)
- Loss: Huber loss per appliance head
- Regularisation: L2 + Dropout (0.2)

---

## Results

| Appliance | MAE (kWh) | F1 Score | vs Naive Baseline |
|-----------|-----------|----------|-------------------|
| Fridge | 0.4813 | 0.138 | -28.7% |
| Washing Machine | 0.4849 | 0.147 | -8.3% |
| Dishwasher | 0.1073 | 0.482 | +41.6% ✅ |
| TV | 0.1598 | 0.824 | +23.0% ✅ |
| Kettle | 0.1463 | 0.996 | +57.4% ✅ |

Note: Fridge and Washing Machine are known hard cases in NILM 
literature due to continuous cycling behaviour.

---

## Dashboard Features
- Live Monitor — Daily energy trends, predicted vs actual
- Appliance Analysis — Per-appliance MAE, F1, prediction plots
- Anomaly Detection — Z-score based drift detection
- Bill Estimator — Interactive UK electricity rate calculator

---

## Key Design Decisions
1. BiLSTM over LSTM — captures temporal patterns in both directions
2. Sum resampling — energy accumulates per hour, not averages
3. Per-appliance heads — prevents gradient interference between appliances
4. Chronological split — no data leakage in time series evaluation
5. Huber loss — robust to energy consumption outliers