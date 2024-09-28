import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Membaca file CSV Anda
df = pd.read_csv('CLEANED_DATA\cleaned_old.csv')  # Shopping Mall Customer Segmentation Data
#df = pd.read_csv('Shopping_Mall_Customer_Segmentation _Data.csv')

# Mengambil kolom yang relevan (Annual Income dan Spending Score)
X = df[['Annual Income', 'Spending Score']]

# Normalisasi data menggunakan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Menggunakan NearestNeighbors untuk menentukan nilai eps terbaik
nearest_neighbors = NearestNeighbors(n_neighbors=10)  # Tetapkan 10 tetangga terdekat
nearest_neighbors.fit(X_scaled)
distances, indices = nearest_neighbors.kneighbors(X_scaled)

# Mengurutkan jarak berdasarkan tetangga ke-10
distances = np.sort(distances[:, 9])  # 9 karena indeks mulai dari 0

# Plot K-Distance
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.title('K-Distance Plot for Optimal Epsilon')
plt.xlabel('Data Points sorted by distance')
plt.ylabel('5th Nearest Neighbor Distance')
plt.grid(True)
plt.show()
