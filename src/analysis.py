import pandas as pd

# Read the entire dataset into a DataFrame
df = pd.read_hdf('results.h5', key='user_equipment_kpis')

# Now you can use standard Pandas tools
print(df.head())

import matplotlib.pyplot as plt

# Filter for rows where packets were actually transmitted
active_data = df[df['throughput'] > 0]

plt.figure(figsize=(10, 6))
plt.scatter(active_data['sinr'], active_data['throughput'], alpha=0.5, c=active_data['assigned_prbs'], cmap='viridis')
plt.colorbar(label='Assigned PRBs')
plt.title("System Performance: SINR vs. Throughput")
plt.xlabel("SINR (dB)")
plt.ylabel("Throughput (kb/ms)")
plt.grid(True)
plt.show()