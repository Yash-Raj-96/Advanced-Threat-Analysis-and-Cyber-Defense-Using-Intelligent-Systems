import pandas as pd
import os

input_file = 'data/raw/network_logs/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv'
output_file = 'data/cic_ids2017.csv'

# Load the dataset
df = pd.read_csv(input_file)

# 🔧 Strip whitespace from all column names
df.columns = df.columns.str.strip()

# ✅ Now safely access 'Label' column
df['Label'] = df['Label'].apply(lambda x: 'Attack' if 'PortScan' in str(x) else 'Normal')

# 🧼 Drop missing values and replace infs
df.dropna(inplace=True)
df.replace([float('inf'), -float('inf')], 1e6, inplace=True)

# 💾 Save the processed file
os.makedirs("data", exist_ok=True)
df.to_csv(output_file, index=False)

print(f"✅ Processed CIC-IDS2017 data saved to: {output_file} — Shape: {df.shape}")
