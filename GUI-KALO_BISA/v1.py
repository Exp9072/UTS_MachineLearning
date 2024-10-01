import customtkinter as ctk
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
from tkinter import filedialog
import os

# Function for Eps Calculation
def calculate_eps(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    nearest_neighbors = NearestNeighbors(n_neighbors=10)
    nearest_neighbors.fit(X_scaled)
    distances, indices = nearest_neighbors.kneighbors(X_scaled)
    distances = np.sort(distances[:, 9])  # Sort by the 10th nearest neighbor

    # Plotting K-Distance
    plt.figure(figsize=(10, 6))
    plt.plot(distances)
    plt.title('K-Distance Plot for Optimal Epsilon')
    plt.xlabel('Data Points sorted by distance')
    plt.ylabel('10th Nearest Neighbor Distance')
    plt.grid(True)
    plt.show()

# Function for DBSCAN clustering
def perform_dbscan(df, columns, eps, min_samples, min_cluster_size):
    X = df[columns]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['Cluster'] = dbscan.fit_predict(X_scaled)

    # Plotting the clusters
    plt.figure(figsize=(10, 6))
    plt.scatter(df[columns[0]], df[columns[1]], c=df['Cluster'], cmap='rainbow', s=50)
    plt.title('DBSCAN Clustering of Customers Based on Selected Columns')
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])
    plt.colorbar()
    plt.show()

    # Saving clusters with more than specified size
    for cluster_label in df['Cluster'].unique():
        cluster_data = df[df['Cluster'] == cluster_label]
        if len(cluster_data) > min_cluster_size:
            cluster_data.to_csv(f'cluster_{cluster_label}.csv', index=False)
            print(f"Saved cluster_{cluster_label}.csv")

# GUI Creation
def open_file():
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if filepath:
        entry_file_path.delete(0, ctk.END)
        entry_file_path.insert(0, filepath)

def proceed_eps():
    filepath = entry_file_path.get()
    if not filepath or not os.path.exists(filepath):
        print("Please select a valid CSV file.")
        return

    # Load CSV and display column options
    df = pd.read_csv(filepath)
    columns = list(df.columns)
    selected_columns = [column_box_1.get(), column_box_2.get()]

    if selected_columns[0] not in columns or selected_columns[1] not in columns:
        print("Invalid column selection.")
        return

    X = df[selected_columns]
    calculate_eps(X)

def proceed_cluster():
    filepath = entry_file_path.get()
    if not filepath or not os.path.exists(filepath):
        print("Please select a valid CSV file.")
        return

    # Load CSV
    df = pd.read_csv(filepath)
    columns = list(df.columns)
    selected_columns = [column_box_1.get(), column_box_2.get()]

    if selected_columns[0] not in columns or selected_columns[1] not in columns:
        print("Invalid column selection.")
        return

    # Get Parameters
    eps = float(entry_eps.get())
    min_samples = int(entry_min_samples.get())
    min_cluster_size = int(entry_min_cluster_size.get())

    # Perform clustering
    perform_dbscan(df, selected_columns, eps, min_samples, min_cluster_size)

# Initialize CustomTkinter
app = ctk.CTk()
app.geometry("500x500")
app.title("DBSCAN Clustering Tool")

# File Selection
label_file = ctk.CTkLabel(app, text="Select CSV File:")
label_file.pack(pady=5)

entry_file_path = ctk.CTkEntry(app, width=400)
entry_file_path.pack(pady=5)

button_file = ctk.CTkButton(app, text="Browse", command=open_file)
button_file.pack(pady=5)

# Column Selection
label_columns = ctk.CTkLabel(app, text="Select Columns (max 2):")
label_columns.pack(pady=10)

column_box_1 = ctk.CTkEntry(app, placeholder_text="Column 1")
column_box_1.pack(pady=5)

column_box_2 = ctk.CTkEntry(app, placeholder_text="Column 2")
column_box_2.pack(pady=5)

# Parameters Input
label_eps = ctk.CTkLabel(app, text="Epsilon (eps):")
label_eps.pack(pady=5)
entry_eps = ctk.CTkEntry(app)
entry_eps.pack(pady=5)

label_min_samples = ctk.CTkLabel(app, text="Min Samples (for DBSCAN):")
label_min_samples.pack(pady=5)
entry_min_samples = ctk.CTkEntry(app)
entry_min_samples.pack(pady=5)

label_min_cluster_size = ctk.CTkLabel(app, text="Min Cluster Size (to save):")
label_min_cluster_size.pack(pady=5)
entry_min_cluster_size = ctk.CTkEntry(app)
entry_min_cluster_size.pack(pady=5)

# Buttons for Clustering and Epsilon Plot
button_eps = ctk.CTkButton(app, text="Calculate Optimal Epsilon", command=proceed_eps)
button_eps.pack(pady=10)

button_cluster = ctk.CTkButton(app, text="Perform Clustering", command=proceed_cluster)
button_cluster.pack(pady=10)

app.mainloop()
