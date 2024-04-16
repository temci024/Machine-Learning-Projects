import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
import matplotlib.pyplot as plt

# Load Dataset
def load_data(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

# Train Model
def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

# Main function
def main():
    # Title and header
    st.title("Breast Cancer Detection using Various Machine Learning Algorithms")
    st.header("Validating Models")

    # Choose Dataset file from Local Directory
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        dataset = load_data(uploaded_file)

        # Display dataset summary
        st.subheader("Dataset Summary")
        st.write("Shape:", dataset.shape)
        st.write("First 5 rows:")
        st.write(dataset.head())

        # Mapping Class String Values to Numbers
        dataset['diagnosis'] = dataset['diagnosis'].map({'B': 0, 'M': 1}).astype(int)

        # Segregate Dataset
        X = dataset.iloc[:, 2:32].values
        Y = dataset.iloc[:,1].values

        # Split Dataset
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

        # Feature Scaling
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train) 
        X_test = sc.transform(X_test)

        # Define evaluation metrics
        metrics = {
            'Accuracy': make_scorer(accuracy_score),
            'Precision': make_scorer(precision_score),
            'Recall': make_scorer(recall_score),
            'F1 Score': make_scorer(f1_score)
        }

        # Validating ML algorithms by multiple metrics - Model Score
        models = [
            ('LR', LogisticRegression(solver='liblinear', multi_class='ovr')),
            ('LDA', LinearDiscriminantAnalysis()),
            ('KNN', KNeighborsClassifier()),
            ('CART', DecisionTreeClassifier()),
            ('NB', GaussianNB()),
            ('SVM', SVC(gamma='auto'))
        ]

        results = {}
        for name, model in models:
            kfold = StratifiedKFold(n_splits=10, random_state=None)
            clf = GridSearchCV(model, {}, cv=kfold, scoring=metrics, refit='Accuracy')
            clf.fit(X_train, y_train)
            results[name] = clf

        # Plotting Accuracy, Precision, Recall, and F1 Score separately
        st.subheader("Performance Metrics")

        plt.figure(figsize=(10,6))

        for metric_name in ['Accuracy', 'Precision', 'Recall', 'F1 Score']:
            metric_values = [clf.cv_results_['mean_test_' + metric_name][clf.best_index_] for clf in results.values()]
            plt.plot(results.keys(), metric_values, marker='o', label=metric_name)

        plt.title('Performance Metrics Comparison')
        plt.xlabel('Algorithm')
        plt.ylabel('Score')
        plt.legend()
        st.pyplot(plt)

        # Print suitable model
        best_model_name = max(results, key=lambda key: results[key].best_score_)
        st.subheader("Best Model")
        st.write("The most suitable model for this project is:", best_model_name)

if __name__ == "__main__":
    main()
