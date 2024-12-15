import pandas as pd
import os
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the data
file_path = "top_potential_customers_with_labels.csv"  # Replace with your file path
df = pd.read_csv(file_path)

# Select relevant numerical columns
X = df[['Annual Income', 'Spending Score']]  # Using income and purchase_amount

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply OPTICS
optics = OPTICS(min_samples=10, xi=0.045)  # Adjust eps as needed
df['Cluster_OPTICS'] = optics.fit_predict(X_scaled)

# Display cluster counts
print(df['Cluster_OPTICS'].value_counts())

# Create a folder called 'CLUSTER_OPTICS' if it doesn't exist
output_folder = "OPTICS\CLUSTER_Cleaned_AI"
os.makedirs(output_folder, exist_ok=True)

# Save clusters with more than 50 data points into separate files
# for cluster_label in df['Cluster_OPTICS'].unique():
#     cluster_data = df[df['Cluster_OPTICS'] == cluster_label]
#     if len(cluster_data) > 10:  # Save only clusters with more than 50 data points
#         label_name = f"cluster_{cluster_label}" if cluster_label != -1 else "noise"
#         output_file = os.path.join(output_folder, f'{label_name}.csv')
#         cluster_data.to_csv(output_file, index=False)
#         print(f"Saved {label_name} with {len(cluster_data)} points to {output_file}")

# Visualize OPTICS results
plt.figure(figsize=(10, 6))
plt.scatter(X['Annual Income'], X['Spending Score'], c=df['Cluster_OPTICS'], cmap='rainbow', s=50)
plt.title('OPTICS Clustering (Scaled Income vs Spending Score)')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.colorbar(label='Cluster')
plt.grid(True)
plt.show()
