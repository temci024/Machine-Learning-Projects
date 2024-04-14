import streamlit as st
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Load Dataset
def load_data():
    dataset = load_digits()
    return dataset

# Train Model
def train_model(X_train, y_train, kernel, C_value, gamma_value):
    model = svm.SVC(kernel=kernel, C=C_value, gamma=gamma_value)
    model.fit(X_train, y_train)
    return model

# Predict Digit
def predict_digit(model, image):
    prediction = model.predict(image.reshape(1, -1))
    return prediction

# Main function
def main():
    st.title('Handwritten Digit Recognition - SVM')
    
    # Load dataset
    dataset = load_data()
    
    # Segregate dataset into X and Y
    X = dataset.images.reshape(len(dataset.images), -1)
    Y = dataset.target
    
    # Split dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)
    
    # Tuning parameters
    st.sidebar.header('Tuning Parameters')
    kernel = st.sidebar.selectbox('Select Kernel', ('linear', 'rbf', 'poly'))
    C_value = st.sidebar.slider('C (Regularization Parameter)', min_value=0.01, max_value=10.0, value=1.0, step=0.01)
    gamma_value = st.sidebar.slider('Gamma', min_value=0.0001, max_value=1.0, value=0.01, step=0.0001)
    
    # Train model
    model = train_model(X_train, y_train, kernel, C_value, gamma_value)
    
    # Evaluate model - Accuracy Score
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.subheader('Model Evaluation')
    st.write(f'Accuracy of the Model: {accuracy * 100:.2f}%')
    
    # Test the Model
    st.sidebar.header('Test the Model')
    n = st.sidebar.slider('Choose a sample image to predict', 0, len(X_test) - 1)
    scaled_image = X_test[n].reshape(8, 8) / 16.0  # Scale pixel values to [0.0, 1.0]
    prediction = predict_digit(model, X_test[n])
    st.image(scaled_image, width=150, caption=f'Predicted Digit: {prediction[0]}')


if __name__ == '__main__':
    main()
