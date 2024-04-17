import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load Dataset
@st.cache_data
def load_data():
    dataset = pd.read_csv("C:/Users/USER/Documents/My GitHub Folder/Machine Learning Project/Machine-Learning-Projects/1. Supervised Learning/15. Evaluating Regression Model/Dataset.csv")
    return dataset

def main():
    st.title("Evaluating Regression Model Using R-Squared & Adjusted R-Squared")

    # Load dataset
    dataset = load_data()

    # Display dataset summary
    st.subheader("Dataset Summary")
    st.write(dataset.head())

    # Visualize Dataset
    st.subheader("Visualize Dataset")
    plt.xlabel('Area')
    plt.ylabel('Price')
    plt.scatter(dataset.area, dataset.price, color='red', marker='*')
    st.pyplot(plt.gcf())

    # Segregate Dataset into Input X & Output Y
    X = dataset.drop('price', axis='columns')
    Y = dataset.price

    # Splitting Dataset for Testing our Model
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

    # Training Dataset using Linear Regression
    model = LinearRegression()
    model.fit(x_train, y_train)

    # Visualizing Linear Regression results
    st.subheader("Visualizing Linear Regression results")
    plt.scatter(X, Y, color="red", marker='*')
    plt.plot(X, model.predict(X))
    plt.title("Linear Regression")
    plt.xlabel("Area")
    plt.ylabel("Price")
    st.pyplot(plt.gcf())

    # R-Squared Score
    rsquared = model.score(x_test, y_test)
    st.subheader("R-Squared Score")
    st.write(rsquared)

    # Adjusted R-Squared of the Model
    n = len(dataset)  # Length of Total dataset
    p = len(dataset.columns) - 1  # Length of Features
    adjr = 1 - (1 - rsquared) * (n - 1) / (n - p - 1)
    st.subheader("Adjusted R-Squared of the Model")
    st.write(adjr)

    # User Input for Prediction
    st.subheader("Prediction")
    x_input = st.number_input("Enter area in sq ft for prediction:", format="%f")
    land_area = [[x_input]]
    predicted_price = model.predict(land_area)
    st.write("Predicted Price:", predicted_price[0])

if __name__ == "__main__":
    main()
