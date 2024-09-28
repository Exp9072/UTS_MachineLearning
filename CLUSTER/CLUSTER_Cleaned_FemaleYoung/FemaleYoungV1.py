import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

# Membaca file CSV Anda
df = pd.read_csv('CLEANED_DATA\\female_young.csv')

# Mengambil kolom yang relevan untuk DBSCAN (Spending Score dan Annual Income)
X = df[['Annual Income', 'Spending Score']]

# Normalisasi data menggunakan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menggunakan DBSCAN dengan parameter eps dan min_samples yang bisa Anda sesuaikan
dbscan = DBSCAN(eps=0.165, min_samples=10)
df['Cluster'] = dbscan.fit_predict(X_scaled)

# Menampilkan jumlah klaster dan outliers
print(df['Cluster'].value_counts())

# Simpan hanya klaster yang memiliki lebih dari 50 anggota ke file CSV terpisah
# Note : Kalo mau dapeting file CSV clusternya uncomment code for loop dibawah ini 
"""

"""
for cluster_label in df['Cluster'].unique():
    cluster_data = df[df['Cluster'] == cluster_label]
    if len(cluster_data) > 50:  # Hanya simpan klaster dengan lebih dari 50 data
        cluster_data.to_csv(f'cluster_{cluster_label}.csv', index=False)


# Visualisasi hasil klaster
plt.figure(figsize=(10, 6))
plt.scatter(df['Annual Income'], df['Spending Score'], c=df['Cluster'], cmap='rainbow', s=50)
plt.title('DBSCAN Clustering of Customers Based on Annual Income and Spending Score')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.colorbar()
plt.show()

# Informasi file CSV yang disimpan untuk setiap kluster
files_saved = [f"cluster_{label}.csv" for label in df['Cluster'].unique()]
files_saved
