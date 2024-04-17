import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as clus
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import LabelEncoder

# Load Dataset
@st.cache_data
def load_data():
    dataset = pd.read_csv("C:/Users/USER/Documents/My GitHub Folder/Machine Learning Project/Machine-Learning-Projects/2. Unsupervised Learning/2. Hierarchical Clustering/Dataset.csv")
    return dataset

def main():
    st.title("Clustering Income Spent Using Hierarchical Clustering")

    # Load dataset
    dataset = load_data()

    # Display dataset summary
    st.subheader("Dataset Summary")
    st.write(dataset.describe())
    st.write(dataset.head())

    # Label Encoding
    label_encoder = LabelEncoder()
    dataset['Gender'] = label_encoder.fit_transform(dataset['Gender'])

    # Dendrogram Data visualization
    st.subheader("Dendrogram Tree Graph")
    fig, ax = plt.subplots(figsize=(16, 8))
    dendrogram = clus.dendrogram(clus.linkage(dataset, method="ward"))
    ax.set_xlabel('Customers')
    ax.set_ylabel('Distances')
    st.pyplot(fig)

    # Fitting the Hierarchical clustering to the dataset with n=5
    model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='average')
    y_means = model.fit_predict(dataset)

    # Visualizing the number of clusters n=5
    st.subheader("Income Spent Analysis - Hierarchical Clustering")
    fig, ax = plt.subplots()
    X = dataset.iloc[:, [3, 4]].values
    ax.scatter(X[y_means == 0, 0], X[y_means == 0, 1], s=50, c='purple', label='Cluster 1')
    ax.scatter(X[y_means == 1, 0], X[y_means == 1, 1], s=50, c='orange', label='Cluster 2')
    ax.scatter(X[y_means == 2, 0], X[y_means == 2, 1], s=50, c='red', label='Cluster 3')
    ax.scatter(X[y_means == 3, 0], X[y_means == 3, 1], s=50, c='green', label='Cluster 4')
    ax.scatter(X[y_means == 4, 0], X[y_means == 4, 1], s=50, c='blue', label='Cluster 5')
    ax.set_title('Income Spent Analysis - Hierarchical Clustering')
    ax.set_xlabel('Income')
    ax.set_ylabel('Spent')
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
