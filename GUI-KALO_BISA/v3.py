import customtkinter as ctk
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tkinter import filedialog
import os

# Variabel global untuk melacak apakah ingin menyimpan klaster atau tidak
save_clusters = False

# Fungsi untuk mendeteksi delimiter CSV
def detect_csv_delimiter(filepath):
    with open(filepath, 'r') as file:
        first_line = file.readline()
        if ';' in first_line:
            return ';'
        else:
            return ','

# Fungsi untuk menghitung nilai Eps
def calculate_eps(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    nearest_neighbors = NearestNeighbors(n_neighbors=10)
    nearest_neighbors.fit(X_scaled)
    distances, indices = nearest_neighbors.kneighbors(X_scaled)
    distances = np.sort(distances[:, 9])  # Mengurutkan berdasarkan tetangga ke-10 terdekat

    # Plot K-Distance
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('K-Distance Plot for Optimal Epsilon')
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel('10th Nearest Neighbor Distance')
    plt.grid(True)
    plt.show()

# Fungsi untuk klasterisasi DBSCAN
def perform_dbscan(df, columns, eps, min_samples, min_cluster_size):
    X = df[columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = dbscan.fit_predict(X_scaled)

    # Memvisualisasikan hasil klaster
    plt.figure(figsize=(10, 6))
    plt.scatter(df[columns[0]], df[columns[1]], c=df['Cluster'], cmap='rainbow', s=50)
    plt.title('DBSCAN Clustering of Customers Based on Selected Columns')
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.colorbar()
    plt.show()

    # Menyimpan klaster dengan ukuran lebih dari yang ditentukan jika save_clusters True
    if save_clusters:
        for cluster_label in df['Cluster'].unique():
            cluster_data = df[df['Cluster'] == cluster_label]
            if len(cluster_data) > min_cluster_size:
                cluster_data.to_csv(f'cluster_{cluster_label}.csv', index=False)
                print(f"Saved cluster_{cluster_label}.csv")

# Fungsi untuk membuka file
def open_file():
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if filepath:
        entry_file_path.delete(0, ctk.END)
        entry_file_path.insert(0, filepath)

# Fungsi untuk memproses nilai Eps
def proceed_eps():
    filepath = entry_file_path.get()
    if not filepath or not os.path.exists(filepath):
        print("Silakan pilih file CSV yang valid.")
        return

    # Mendeteksi delimiter CSV
    delimiter = detect_csv_delimiter(filepath)

    # Membaca CSV dan menampilkan opsi kolom
    df = pd.read_csv(filepath, delimiter=delimiter)
    columns = list(df.columns)
    selected_columns = [column_box_1.get(), column_box_2.get()]

    if selected_columns[0] not in columns or selected_columns[1] not in columns:
        print("Pilihan kolom tidak valid.")
        return

    X = df[selected_columns]
    calculate_eps(X)

# Fungsi untuk memproses klasterisasi
def proceed_cluster():
    filepath = entry_file_path.get()
    if not filepath or not os.path.exists(filepath):
        print("Silakan pilih file CSV yang valid.")
        return

    # Mendeteksi delimiter CSV
    delimiter = detect_csv_delimiter(filepath)

    # Membaca CSV
    df = pd.read_csv(filepath, delimiter=delimiter)
    columns = list(df.columns)
    selected_columns = [column_box_1.get(), column_box_2.get()]

    if selected_columns[0] not in columns or selected_columns[1] not in columns:
        print("Pilihan kolom tidak valid.")
        return

    # Mendapatkan parameter
    eps = float(entry_eps.get())
    min_samples = int(entry_min_samples.get()) if entry_min_samples.get() else 10
    min_cluster_size = int(entry_min_cluster_size.get())

    # Melakukan klasterisasi
    perform_dbscan(df, selected_columns, eps, min_samples, min_cluster_size)

# Fungsi untuk mengubah status simpan klaster
def toggle_save_clusters():
    global save_clusters
    save_clusters = not save_clusters
    if save_clusters:
        button_save_clusters.configure(text="Saving: Enabled")
    else:
        button_save_clusters.configure(text="Saving: Disabled")

# Inisialisasi CustomTkinter
app = ctk.CTk()
app.geometry("500x600")
app.title("DBSCAN Clustering Tool")

# Pemilihan File
label_file = ctk.CTkLabel(app, text="Pilih File CSV:")
label_file.pack(pady=5)

entry_file_path = ctk.CTkEntry(app, width=400)
entry_file_path.pack(pady=5)

button_file = ctk.CTkButton(app, text="Browse", command=open_file)
button_file.pack(pady=5)

# Pemilihan Kolom
label_columns = ctk.CTkLabel(app, text="Pilih Kolom (maks 2):")
label_columns.pack(pady=10)

column_box_1 = ctk.CTkEntry(app, placeholder_text="Kolom 1")
column_box_1.pack(pady=5)

column_box_2 = ctk.CTkEntry(app, placeholder_text="Kolom 2")
column_box_2.pack(pady=5)

# Input Parameter
label_eps = ctk.CTkLabel(app, text="Epsilon (eps):")
label_eps.pack(pady=5)
entry_eps = ctk.CTkEntry(app)
entry_eps.pack(pady=5)

label_min_samples = ctk.CTkLabel(app, text="Min Samples (default 10):")
label_min_samples.pack(pady=5)
entry_min_samples = ctk.CTkEntry(app)
entry_min_samples.pack(pady=5)

label_min_cluster_size = ctk.CTkLabel(app, text="Ukuran Klaster Minimum (untuk disimpan):")
label_min_cluster_size.pack(pady=5)
entry_min_cluster_size = ctk.CTkEntry(app)
entry_min_cluster_size.pack(pady=5)

# Tombol untuk Klasterisasi dan Plot Epsilon
button_eps = ctk.CTkButton(app, text="Hitung Optimal Epsilon", command=proceed_eps)
button_eps.pack(pady=10)

button_cluster = ctk.CTkButton(app, text="Lakukan Klasterisasi", command=proceed_cluster)
button_cluster.pack(pady=10)

# Tombol Toggle Simpan Klaster
button_save_clusters = ctk.CTkButton(app, text="Saving: Disabled", command=toggle_save_clusters)
button_save_clusters.pack(pady=10)

app.mainloop()
