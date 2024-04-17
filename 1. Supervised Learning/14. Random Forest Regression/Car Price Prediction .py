import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import scale

# Load Dataset
@st.cache_data
def load_data():
    dataset = pd.read_csv('C:/Users/USER/Documents/My GitHub Folder/Machine Learning Project/Machine-Learning-Projects/1. Supervised Learning/14. Random Forest Regression/Dataset.csv')
    dataset = dataset.drop(['car_ID'], axis=1)
    return dataset

def main():
    st.title("Car Price Prediction using Random Forest")

    # Load dataset
    dataset = load_data()

    # Display dataset summary
    st.subheader("Dataset Summary")
    st.write(dataset.head())

    # Splitting Dataset into X & Y
    X_data = dataset.drop('price', axis='columns')
    numerical_cols = X_data.select_dtypes(exclude=['object']).columns
    X = X_data[numerical_cols]
    Y = dataset['price']

    # Scaling the Independent Variables (Features)
    X_scaled = scale(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    # Splitting Dataset into Train & Test
    x_train, x_test, y_train, y_test = train_test_split(X_scaled, Y, test_size=0.20, random_state=0)

    # Training using Random Forest
    model = RandomForestRegressor()
    model.fit(x_train, y_train)

    # Evaluating Model
    y_pred = model.predict(x_test)
    r2score = r2_score(y_test, y_pred)

    # Display evaluation result
    st.subheader("Model Evaluation")
    st.write("R2 Score:", r2score * 100)

    # User Input for Prediction
    st.subheader("Test Your Own Data")
    st.write("Enter the following features for car price prediction:")
    user_input = {}
    for col in numerical_cols:
        user_input[col] = st.number_input(col, format="%f")
    user_input_scaled = scale(pd.DataFrame(user_input, index=[0]))
    predicted_price = model.predict(user_input_scaled)
    st.write("Predicted Price:", predicted_price[0])

if __name__ == "__main__":
    main()
