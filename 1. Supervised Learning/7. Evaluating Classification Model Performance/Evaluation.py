import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt

# Load Dataset
def load_data(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

# Train Model
def train_model(X_train, y_train):
    model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)
    return model

# Main function
def main():
    # Title and header
    st.title("Classification Model Evaluation - Sale Prediction from Existing Customer")
    st.header("Validating Logistic Regression Model")

    # Choose Dataset file from Local Directory
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        dataset = load_data(uploaded_file)

        # Display dataset summary
        st.subheader("Dataset Summary")
        st.write("Shape:", dataset.shape)
        st.write("First 5 rows:")
        st.write(dataset.head())

        # Segregate Dataset
        X = dataset.iloc[:, :-1]
        Y = dataset.iloc[:, -1]

        # Split Dataset
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

        # Feature Scaling
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train) 
        X_test = sc.transform(X_test)

        # Training
        model = train_model(X_train, y_train)

        # Prediction for all Test Data
        y_pred = model.predict(X_test)

        # Evaluating Model
        st.subheader("Evaluating Model")

        # Confusion Matrix
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        st.write(cm)

        # Accuracy Score
        st.write("Accuracy: {:.2f}%".format(accuracy_score(y_test, y_pred) * 100))

        # Receiver Operating Curve (ROC Curve)
        st.write("Receiver Operating Characteristic (ROC) Curve:")
        fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:,1])
        roc_auc = roc_auc_score(y_test, y_pred)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        st.pyplot(plt)

        # Cross Validation Score
        st.write("Cross Validation Score: {:.2f}%".format(np.mean(cross_val_score(model, X, Y, cv=10)) * 100))

        # Stratified K-fold Cross Validation
        st.write("Stratified K-Fold Cross Validation Score: {:.2f}%".format(np.mean(cross_val_score(model, X, Y, cv=StratifiedKFold(n_splits=3))) * 100))

if __name__ == "__main__":
    main()
