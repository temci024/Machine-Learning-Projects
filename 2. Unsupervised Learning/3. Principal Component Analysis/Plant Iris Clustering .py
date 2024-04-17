import streamlit as st
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load Dataset
@st.cache_data
def load_data():
    dataset = datasets.load_iris()
    return dataset

def main():
    st.title("Clustering Plant Iris Using Principal Component Analysis")

    # Load dataset
    dataset = load_data()

    # Dataset Segregation
    X = dataset.data
    y = dataset.target
    names = dataset.target_names

    # Fitting the PCA clustering to the dataset with n=2
    model = PCA(n_components=2)
    y_means = model.fit(X).transform(X)

    # Variance Percentage
    st.subheader("IRIS Clustering")
    fig, ax = plt.subplots()
    colors = ['red', 'green', 'orange']

    for color, i, target_name in zip(colors, [0, 1, 2], names):
        ax.scatter(y_means[y == i, 0], y_means[y == i, 1], color=color, lw=2, label=target_name)

    ax.set_title('IRIS Clustering')
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
