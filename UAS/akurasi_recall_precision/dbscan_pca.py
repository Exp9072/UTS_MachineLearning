import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


# Load the dataset
file_path = "customer_data_with_clusters_pca.csv"  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Ensure the "Cluster" column is present
if "Cluster" not in df.columns:
    raise ValueError("The CSV file must contain a 'Cluster' column.")

# Separate features (X) and target (y)
X = df.drop(columns=["Cluster"])  # Drop the target column
y = df["Cluster"]

# Check for non-numeric columns
non_numeric_columns = X.select_dtypes(include=['object']).columns
print(f"Non-numeric columns in the dataset: {non_numeric_columns}")

# Option 1: Remove non-numeric columns (if they are not useful)
X = X.drop(columns=non_numeric_columns)

# Option 2: Encode non-numeric columns (if they are useful)
# For example, we can use LabelEncoder for categorical columns:
# for col in non_numeric_columns:
#     le = LabelEncoder()
#     X[col] = le.fit_transform(X[col])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Initialize the Naive Bayes classifier
nb = GaussianNB()

# Train the model
nb.fit(X_train, y_train)

# Predict the test set
y_pred = nb.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)


# lllll
import seaborn as sns
import matplotlib.pyplot as plt
f, ax = plt.subplots(figsize=(8,5))
sns.heatmap(conf_matrix, annot=True, fmt=".0f", ax=ax)
plt.xlabel("y_head")
plt.ylabel("y_true")
plt.show()