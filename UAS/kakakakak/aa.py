import pandas as pd

# Load datasets
df_ai = pd.read_csv('top_potential_customers_with_labels.csv')  # Dataset AI dengan Potensial
df_pca = pd.read_csv('customer_data_pca.csv')  # Dataset PCA

# Cek apakah jumlah baris sama
if len(df_ai) != len(df_pca):
    print(f"Jumlah baris tidak sama! Dataset AI: {len(df_ai)}, Dataset PCA: {len(df_pca)}")
else:
    print("Jumlah baris sama, melanjutkan proses penambahan kolom Potensial.")

    # Tambahkan kolom Potensial dari dataset AI ke dataset PCA
    df_pca['Potensial'] = df_ai['Potensial']

    # Periksa hasil
    print("Kolom Potensial berhasil ditambahkan.")
    print(df_pca.head())

    # Simpan dataset PCA dengan label Potensial
    # df_pca.to_csv('customer_data_pca_with_potential.csv', index=False)
