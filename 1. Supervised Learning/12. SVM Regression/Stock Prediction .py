import streamlit as st
import pandas as pd
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import numpy as np

# Load Dataset
def load_data(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

# Train SVR Model
def train_model(X_train, y_train):
    model = SVR()
    model.fit(X_train, y_train)
    return model

# Main function
def main():
    # Title and header
    st.title("Stock Prediction using Support Vector Regression")
    st.header("Upload Dataset")

    # Choose Dataset file from Local Directory
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        dataset = load_data(uploaded_file)

        # Display dataset summary
        st.subheader("Dataset Summary")
        st.write("Shape:", dataset.shape)
        st.write("First 5 rows:")
        st.write(dataset.head())

        # Segregate Dataset into Input X & Output Y
        X = dataset.iloc[:, :-1].values
        Y = dataset.iloc[:, -1].values

        # Splitting Dataset for Testing our Model
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

        # Training Dataset using Support Vector Regression
        model = train_model(x_train, y_train)

        # Prediction
        y_pred = model.predict(x_test)

        # Evaluation metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = round(np.sqrt(mse), 2)
        r2score = round(r2_score(y_test, y_pred) * 100, 2)

        st.subheader("Model Evaluation")
        st.write("Root Mean Square Error (RMSE):", rmse)
        st.write("R2 Score:", r2score)

        # Visualizing the results
        st.subheader("Actual vs Predicted")
        plt.figure(figsize=(10, 6))
        plt.scatter(range(len(y_test)), y_test, color='red', label='Actual')
        plt.scatter(range(len(y_pred)), y_pred, color='blue', label='Predicted')
        plt.title('Actual vs Predicted')
        plt.xlabel('Index')
        plt.ylabel('Stock Price')
        plt.legend()
        st.pyplot(plt)

        # User input for prediction
        st.subheader("Test the Model")
        st.write("Enter values for features to predict the stock price:")
        input_features = {}
        for column in dataset.columns[:-1]:
            input_features[column] = st.slider(f"Enter {column}:", min_value=dataset[column].min(), max_value=dataset[column].max(), value=dataset[column].mean())

        input_data = np.array([[input_features[column] for column in dataset.columns[:-1]]])
        prediction = model.predict(input_data)
        st.write("Predicted Stock Price:", prediction[0])

if __name__ == "__main__":
    main()
