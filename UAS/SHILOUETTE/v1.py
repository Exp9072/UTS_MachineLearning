import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the data
df_ai = pd.read_csv('top_potential_customers_with_labels.csv')  # Dataset Annual Income
df_pca = pd.read_csv('customer_data_pca.csv')  # Dataset PCA

# Features for clustering
X_ai = df_ai[['Annual Income', 'Spending Score']]
X_pca = df_pca[['PC1', 'PC2']]

# StandardScaler for AI data
scaler = StandardScaler()
X_ai_scaled = scaler.fit_transform(X_ai)

# DBSCAN and OPTICS on AI
dbscan_ai = DBSCAN(eps=0.05, min_samples=10).fit(X_ai_scaled)  # Adjust eps for AI
optics_ai = OPTICS(min_samples=10, xi=0.045).fit(X_ai)

# DBSCAN and OPTICS on PCA
dbscan_pca = DBSCAN(eps=0.1, min_samples=10).fit(X_pca)  # Adjust eps for PCA
optics_pca = OPTICS(min_samples=10, xi=0.005).fit(X_pca)

# Add cluster labels to dataframes
df_ai['Cluster_DBSCAN'] = dbscan_ai.labels_
df_ai['Cluster_OPTICS'] = optics_ai.labels_

df_pca['Cluster_DBSCAN'] = dbscan_pca.labels_
df_pca['Cluster_OPTICS'] = optics_pca.labels_

# Silhouette Scores
silhouette_scores = {}

# Annual Income - DBSCAN
if len(set(dbscan_ai.labels_)) > 1:
    silhouette_scores['DBSCAN_AI'] = silhouette_score(X_ai, dbscan_ai.labels_)
else:
    silhouette_scores['DBSCAN_AI'] = None

# Annual Income - OPTICS
if len(set(optics_ai.labels_)) > 1:
    silhouette_scores['OPTICS_AI'] = silhouette_score(X_ai, optics_ai.labels_)
else:
    silhouette_scores['OPTICS_AI'] = None

# PCA - DBSCAN
if len(set(dbscan_pca.labels_)) > 1:
    silhouette_scores['DBSCAN_PCA'] = silhouette_score(X_pca, dbscan_pca.labels_)
else:
    silhouette_scores['DBSCAN_PCA'] = None

# PCA - OPTICS
if len(set(optics_pca.labels_)) > 1:
    silhouette_scores['OPTICS_PCA'] = silhouette_score(X_pca, optics_pca.labels_)
else:
    silhouette_scores['OPTICS_PCA'] = None

# Print Silhouette Scores
print("Silhouette Scores:")
for method, score in silhouette_scores.items():
    print(f"{method}: {score}")



# Visualization Function for Multiple Plots
def plot_clustering_multiple(data1, x1_col, y1_col, cluster1_col, title1,
                              data2, x2_col, y2_col, cluster2_col, title2,
                              data3, x3_col, y3_col, cluster3_col, title3,
                              data4, x4_col, y4_col, cluster4_col, title4):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Clustering Results", fontsize=16)
    
    # Plot 1: DBSCAN on Annual Income
    axes[0, 0].scatter(data1[x1_col], data1[y1_col], c=data1[cluster1_col], cmap='rainbow', s=50)
    axes[0, 0].set_title(title1)
    axes[0, 0].set_xlabel(x1_col)
    axes[0, 0].set_ylabel(y1_col)
    axes[0, 0].grid(True)
    
    # Plot 2: OPTICS on Annual Income
    axes[0, 1].scatter(data2[x2_col], data2[y2_col], c=data2[cluster2_col], cmap='rainbow', s=50)
    axes[0, 1].set_title(title2)
    axes[0, 1].set_xlabel(x2_col)
    axes[0, 1].set_ylabel(y2_col)
    axes[0, 1].grid(True)
    
    # Plot 3: DBSCAN on PCA Components
    axes[1, 0].scatter(data3[x3_col], data3[y3_col], c=data3[cluster3_col], cmap='rainbow', s=50)
    axes[1, 0].set_title(title3)
    axes[1, 0].set_xlabel(x3_col)
    axes[1, 0].set_ylabel(y3_col)
    axes[1, 0].grid(True)
    
    # Plot 4: OPTICS on PCA Components
    axes[1, 1].scatter(data4[x4_col], data4[y4_col], c=data4[cluster4_col], cmap='rainbow', s=50)
    axes[1, 1].set_title(title4)
    axes[1, 1].set_xlabel(x4_col)
    axes[1, 1].set_ylabel(y4_col)
    axes[1, 1].grid(True)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for the suptitle
    plt.show()

# Call the function with 4 datasets and plots
plot_clustering_multiple(
    df_ai, 'Annual Income', 'Spending Score', 'Cluster_DBSCAN', 'DBSCAN Clustering (Annual Income)',
    df_ai, 'Annual Income', 'Spending Score', 'Cluster_OPTICS', 'OPTICS Clustering (Annual Income)',
    df_pca, 'PC1', 'PC2', 'Cluster_DBSCAN', 'DBSCAN Clustering (PCA Components)',
    df_pca, 'PC1', 'PC2', 'Cluster_OPTICS', 'OPTICS Clustering (PCA Components)'
)
