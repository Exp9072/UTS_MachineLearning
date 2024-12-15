import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.X = None
        self.y = None
        self.model = None
        
    def load_and_preprocess(self):
        # Load data
        self.df = pd.read_csv(self.file_path)
        
        # Check for Cluster column
        if "Cluster" not in self.df.columns:
            raise ValueError("The CSV file must contain a 'Cluster' column.")
            
        # Handle missing values
        self.df = self.df.fillna(self.df.mean())
        
        # Prepare features and target
        self.X = self.df.drop(columns=["Cluster"])
        self.y = self.df["Cluster"]
        
        # Handle non-numeric columns
        non_numeric_columns = self.X.select_dtypes(include=['object']).columns
        self.X = self.X.drop(columns=non_numeric_columns)
        
        # Scale features
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        
    def train_and_evaluate(self):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Train model
        self.model = GaussianNB()
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Evaluate
        print("\nModel Performance:")
        print("-----------------")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(f"Precision: {precision_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
        print(f"Recall: {recall_score(y_test, y_pred, average='weighted', zero_division=0):.2f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X, self.y, cv=5)
        print(f"\nCross-validation mean score: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        
        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        # Confusion Matrix
        self.plot_confusion_matrix(y_test, y_pred)
        
    def plot_confusion_matrix(self, y_true, y_pred):
        conf_matrix = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
    def plot_feature_importance(self):
        if hasattr(self.model, 'var_'):
            importances = pd.DataFrame({
                'features': self.df.drop(columns=["Cluster"]).columns,
                'importance': self.model.var_
            })
            importances = importances.sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='features', data=importances)
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.show()
            
    def run_analysis(self):
        """Run complete analysis pipeline"""
        print("Loading and preprocessing data...")
        self.load_and_preprocess()
        
        print("\nTraining and evaluating model...")
        self.train_and_evaluate()
        
        print("\nPlotting feature importance...")
        self.plot_feature_importance()

# Usage
if __name__ == "__main__":
    evaluator = ModelEvaluator("akurasi_recall_precision\customers_with_optics_clusters_pca.csv")
    evaluator.run_analysis()
