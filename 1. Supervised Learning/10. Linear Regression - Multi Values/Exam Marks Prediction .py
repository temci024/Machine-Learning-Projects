import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Load Dataset
def load_data(file_path):
    dataset = pd.read_csv(file_path)
    return dataset

# Preprocess Dataset
def preprocess_data(dataset):
    # Check for missing values
    if dataset.isna().any().any():
        st.warning("Dataset contains missing values. Preprocessing...")
        # Impute missing values with mean
        imputer = SimpleImputer(strategy="mean")
        dataset = pd.DataFrame(imputer.fit_transform(dataset), columns=dataset.columns)
        st.info("Missing values have been imputed with mean.")
    return dataset

# Train Linear Regression Model
def train_model(X, Y):
    model = LinearRegression()
    model.fit(X, Y)
    return model

# Main function
def main():
    # Title and header
    st.title("Exam Mark Prediction using Linear Regression")
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

        # Preprocess Dataset
        dataset = preprocess_data(dataset)

        # Segregate Dataset into Input X & Output Y
        X = dataset.iloc[:, :-1]
        Y = dataset.iloc[:, -1]

        # Training Dataset using Linear Regression
        model = train_model(X, Y)

        # Prediction
        st.subheader("Exam Mark Prediction")
        st.write("Enter the features for prediction:")

        input_features = {}
        for column in X.columns:
            input_features[column] = st.slider(f"Enter {column}:", min_value=X[column].min(), max_value=X[column].max(), value=X[column].mean())

        input_data = [[input_features[column] for column in X.columns]]
        predicted_mark = model.predict(input_data)
        st.write("Predicted Exam Mark:", predicted_mark[0])

if __name__ == "__main__":
    main()
