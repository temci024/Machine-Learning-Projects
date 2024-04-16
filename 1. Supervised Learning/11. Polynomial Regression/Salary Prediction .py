import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Load Dataset
def load_data(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

# Train Polynomial Regression Model
def train_model(X, Y, degree):
    # Transform X to Polynomial Format
    poly_features = PolynomialFeatures(degree=degree)
    X_poly = poly_features.fit_transform(X)
    
    # Train Linear Regression with X-Polynomial
    model = LinearRegression()
    model.fit(X_poly, Y)
    return model, poly_features

# Main function
def main():
    # Title and header
    st.title("Salary Prediction using Polynomial Regression")
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

        # Cast min_value, max_value, and value to float
        min_value = float(X.min())
        max_value = float(X.max())
        mean_value = float(X.mean())

        # Degree of Polynomial
        degree = st.slider("Select the degree of Polynomial Features:", min_value=int(min_value), max_value=int(max_value), value=int(mean_value))

        # Training Dataset using Polynomial Regression
        model, poly_features = train_model(X, Y, degree)

        # Visualization
        st.subheader("Visualization")
        plt.scatter(X, Y, color="red")
        plt.plot(X, model.predict(poly_features.transform(X)), color="blue")
        plt.title("Polynomial Regression")
        plt.xlabel("Level")
        plt.ylabel("Salary")
        st.pyplot(plt)

        # Prediction
        st.subheader("Salary Prediction")
        level = st.slider("Enter the Level:", min_value=min_value, max_value=max_value, value=mean_value)
        salary_pred = model.predict(poly_features.transform([[level]]))
        st.write("Predicted Salary:", salary_pred[0])

if __name__ == "__main__":
    main()
