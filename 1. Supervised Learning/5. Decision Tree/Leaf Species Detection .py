import streamlit as st
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data():
    dataset = load_iris()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    Y = dataset.target
    return X, Y

def train_model(X_train, y_train, max_depth):
    model = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth, random_state=0)
    model.fit(X_train, y_train)
    return model

def main():
    st.title("Leaf Species Detection with Decision Trees")

    # Load Dataset
    st.header("Dataset Summary")
    X, Y = load_data()
    st.write("Number of samples:", len(X))
    st.write("Number of features:", len(X.columns))
    st.write("Classes:", np.unique(Y))

    # Split Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    # Parameter Tuning
    st.header("Parameter Tuning")
    max_depth = st.slider("Select max_depth for Decision Tree", min_value=1, max_value=10, value=3)
    accuracy = []
    for i in range(1, 11):
        model = train_model(X_train, y_train, i)
        pred = model.predict(X_test)
        score = accuracy_score(y_test, pred)
        accuracy.append((i, score))

    accuracy_df = pd.DataFrame(accuracy, columns=["Max Depth", "Accuracy"])

    st.line_chart(accuracy_df.set_index("Max Depth"))

    # Train Model
    st.header("Training")
    model = train_model(X_train, y_train, max_depth)
    st.write("Model trained with max_depth:", max_depth)

    # Prediction
    st.header("Prediction")
    y_pred = model.predict(X_test)
    result_df = pd.DataFrame({'Predicted': y_pred, 'Actual': y_test})
    st.write(result_df)

    # Model Evaluation
    st.header("Model Evaluation")
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Accuracy of the Model: {:.2f}%".format(accuracy * 100))

    # Test the Model
    st.header("Test the Model")
    st.write("Use the slider bars to input feature values for testing:")
    input_data = {}
    for feature in X.columns:
        input_data[feature] = st.slider(f"Enter {feature}", float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    st.write("Predicted Class:", prediction[0])

if __name__ == "__main__":
    main()
