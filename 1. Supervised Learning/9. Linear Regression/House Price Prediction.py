import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load Dataset
def load_data(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

# Train Linear Regression Model
def train_model(X, Y):
    model = LinearRegression()
    model.fit(X, Y)
    return model

# Main function
def main():
    # Title and header
    st.title("House Price Prediction using Linear Regression")
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

        # Visualize dataset
        st.subheader("Visualize Dataset")
        plt.scatter(dataset['area'], dataset['price'], color='red')
        plt.xlabel('Area')
        plt.ylabel('Price')
        plt.title('House Price vs Area')
        st.pyplot(plt)

        # Segregate Dataset into Input X & Output Y
        X = dataset[['area']]
        Y = dataset['price']

        # Training Dataset using Linear Regression
        model = train_model(X, Y)

        # Prediction
        st.subheader("House Price Prediction")
        area_input = st.number_input("Enter the area in square feet:")
        predicted_price = model.predict([[area_input]])
        st.write("Predicted Price:", predicted_price[0])

if __name__ == "__main__":
    main()
