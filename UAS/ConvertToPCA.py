import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def process_and_apply_pca(file_path, output_file_pca, output_file_processed):
    # Membaca data dari file CSV
    data = pd.read_csv(file_path)

    # Mengubah kolom Gender menjadi numerik
    data['Gender'] = data['Gender'].map({'Male': 0, 'Female': 1})

    # Mengubah kolom Potensial menjadi numerik
    data['Potensial'] = data['Potensial'].map({'No': 0, 'Yes': 1})

    # Menghapus kolom Customer ID karena tidak relevan
    data_processed = data.drop(columns=['Customer ID'])

    # Menyimpan data yang sudah diproses
    data_processed.to_csv(output_file_processed, index=False)
    print(f"Data processed saved to: {output_file_processed}")

    # Normalisasi data sebelum PCA
    scaler = StandardScaler()
    data_normalized = scaler.fit_transform(data_processed)

    # Menginisialisasi PCA tanpa menentukan jumlah komponen
    pca_full = PCA()
    pca_full.fit(data_normalized)

    # Menampilkan explained variance ratio untuk semua komponen
    explained_variance_ratio = pca_full.explained_variance_ratio_ * 100
    print("\nExplained Variance Ratio (Percentage Contribution):")
    for i, ratio in enumerate(explained_variance_ratio):
        print(f"Principal Component {i+1}: {ratio:.2f}% variance explained")

    # Cumulative variance explained
    cumulative_variance = explained_variance_ratio.cumsum()
    print("\nCumulative Variance Explained:")
    print(cumulative_variance)

    # Plot cumulative variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.title('Cumulative Explained Variance by Principal Components')
    plt.grid()
    plt.show()

    # Reduksi ke jumlah komponen utama yang diinginkan
    n_components = 2  # Menentukan jumlah komponen utama
    pca = PCA(n_components=n_components)
    data_reduced = pca.fit_transform(data_normalized)

    # Menyimpan hasil PCA ke dalam DataFrame
    data_reduced_df = pd.DataFrame(data_reduced, columns=[f'PC{i+1}' for i in range(n_components)])

    # Menyimpan DataFrame hasil PCA ke file CSV
    data_reduced_df.to_csv(output_file_pca, index=False)
    print(f"PCA results saved to: {output_file_pca}")

    # Mengembalikan hasil sebagai DataFrame
    return data_processed, data_reduced_df

# File input dan output
file_path = 'top_potential_customers_with_labels.csv'  # Ganti dengan jalur file input Anda
output_file_processed = 'customer_data_processed.csv'
output_file_pca = 'customer_data_pca.csv'

# Menjalankan fungsi
processed_data, pca_result = process_and_apply_pca(file_path, output_file_pca, output_file_processed)

# Menampilkan hasil
print("\nProcessed Data:")
print(processed_data.head())
print("\nPCA Result:")
print(pca_result.head())
