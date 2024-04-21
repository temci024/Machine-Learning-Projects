import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
  
# Load Dataset
@st.cache_data
def load_data():
    dataset = pd.read_csv("C:/Users/USER/Documents/My GitHub Folder/Machine Learning Project/Machine-Learning-Projects/2. Unsupervised Learning/1. K-Means Clustering/Dataset.csv")
    return dataset

def main():
    st.title("Customer Spent Analysis using K-Means Clustering")

    # Load dataset
    dataset = load_data()

    # Display dataset summary
    st.subheader("Dataset Summary")
    st.write(dataset.describe())
    st.write(dataset.head())

    # Segregate & Zipping Dataset
    Income = dataset['INCOME'].values
    Spend = dataset['SPEND'].values
    X = np.array(list(zip(Income, Spend)))

    # Finding the Optimized K Value
    @st.cache_data
    def find_optimal_k(X):
        wcss = []
        for i in range(1, 11):
            km = KMeans(n_clusters=i, random_state=0)
            km.fit(X)
            wcss.append(km.inertia_)
        return wcss

    wcss = find_optimal_k(X)
    fig, ax1 = plt.subplots()
    ax1.plot(range(1, 11), wcss, color="red", marker="8")
    ax1.set_title('Optimal K Value')
    ax1.set_xlabel('Number of clusters')
    ax1.set_ylabel('WCSS')
    st.pyplot(fig)

    # Fitting the k-means to the dataset with k=4
    @st.cache_data
    def fit_kmeans(X):
        model = KMeans(n_clusters=4, random_state=0)
        y_means = model.fit_predict(X)
        return model, y_means

    model, y_means = fit_kmeans(X)

    # Visualizing the clusters for k=4
    fig, ax2 = plt.subplots()
    ax2.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s=50, c='red', label='Cluster 1')
    ax2.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s=50, c='blue', label='Cluster 2')
    ax2.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s=50, c='green', label='Cluster 3')
    ax2.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s=50, c='purple', label='Cluster 4')
    ax2.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s=300, marker='s', c='red', edgecolors='black', linewidth=2, label='Centroids')
    ax2.set_title('Income Spent Analysis')
    ax2.set_xlabel('Income')
    ax2.set_ylabel('Spent')
    ax2.legend()
    st.pyplot(fig)

    # User Input for Testing
    st.subheader("Test Your Own Data")
    test_income = st.number_input("Enter Income:")
    test_spend = st.number_input("Enter Spend:")
    test_result = model.predict([[test_income, test_spend]])
    st.write("Predicted Cluster:", test_result[0]+1)

if __name__ == "__main__":
    main()
