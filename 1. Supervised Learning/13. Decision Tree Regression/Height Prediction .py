import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Load Dataset
@st.cache_data
def load_data():
    dataset = pd.read_csv('C:/Users/USER/Documents/My GitHub Folder/Machine Learning Project/Machine-Learning-Projects/1. Supervised Learning/13. Decision Tree Regression/Dataset.csv')
    return dataset

def main():
    st.title("Height Prediction using Decision Tree")

    # Load dataset
    dataset = load_data()

    # Display dataset summary
    st.subheader("Dataset Summary")
    st.write(dataset.head())

    # Segregate dataset into Input X & Output Y
    X = dataset.iloc[:, :-1].values
    Y = dataset.iloc[:, -1].values

    # Splitting Dataset for Testing our Model
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)

    # Training Dataset using Decision Tree
    model = DecisionTreeRegressor()
    model.fit(x_train, y_train)

    # Visualizing Graph
    st.subheader("Visualization")
    st.write("Height prediction using Decision Tree")

    X_val = np.arange(min(x_train.flatten()), max(x_train.flatten()), 0.01)
    X_val = X_val.reshape((len(X_val), 1))
    plt.scatter(x_train, y_train, color='green')
    plt.plot(X_val, model.predict(X_val), color='red')
    plt.title('Height prediction using Decision Tree')
    plt.xlabel('Age')
    plt.ylabel('Height')
    st.pyplot(plt.gcf())

    # Model Evaluation
    st.subheader("Model Evaluation")
    y_pred = model.predict(x_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2score = r2_score(y_test, y_pred)
    st.write("Root Mean Square Error:", rmse)
    st.write("R2 Score:", r2score * 100)

    # User Input for Prediction
    st.subheader("Test Your Own Data")
    age = st.slider("Enter Age:", min_value=int(min(X)[0]), max_value=int(max(X)[0]), step=1)
    predicted_height = model.predict([[age]])
    st.write("Predicted Height:", predicted_height[0])

if __name__ == "__main__":
    main()
