import pandas as pd

# Load HDF5 file
file_path = "dataset/ukdale.h5"

try:
    # Check available keys/groups
    store = pd.HDFStore(file_path)

    print("Available Keys:")
    print(store.keys())

    # Example: read first dataset
    key = store.keys()[0]
    print(f"\nReading key: {key}")

    df = store[key]

    # Save as CSV
    output_csv = "dataset/converted_dataset.csv"
    df.to_csv(output_csv, index=False)

    print(f"\n✅ CSV saved as: {output_csv}")

    store.close()

except Exception as e:
    print(f"❌ Error: {e}")