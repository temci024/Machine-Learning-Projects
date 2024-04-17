import streamlit as st
import numpy as np
import pandas as pd

# Load Dataset
@st.cache_data
def load_data():
    ratingData = pd.io.parsers.read_csv('C:/Users/USER/Documents/My GitHub Folder/Machine Learning Project/Machine-Learning-Projects/2. Unsupervised Learning/4. Singular Value Decomposition/ratings.dat', 
        names=['user_id', 'movie_id', 'rating', 'time'],
        engine='python', delimiter='::', encoding='latin-1')
    movieData = pd.io.parsers.read_csv('C:/Users/USER/Documents/My GitHub Folder/Machine Learning Project/Machine-Learning-Projects/2. Unsupervised Learning/4. Singular Value Decomposition/movies.dat',
        names=['movie_id', 'title', 'genre'],
        engine='python', delimiter='::', encoding='latin-1')
    return ratingData, movieData

# Compute Similarity
def similar(ratingData, movie_id, top_n):
    index = movie_id - 1
    movie_row = ratingData[index, :]
    magnitude = np.sqrt(np.einsum('ij, ij -> i', ratingData, ratingData))
    similarity = np.dot(movie_row, ratingData.T) / (magnitude[index] * magnitude)
    sort_indexes = np.argsort(-similarity)
    return sort_indexes[:top_n]

def train_model():
    # Load dataset
    ratingData, movieData = load_data()

    # Create the ratings matrix
    ratingMatrix = np.ndarray(
        shape=(np.max(ratingData.movie_id.values), np.max(ratingData.user_id.values)),
        dtype=np.uint8)
    ratingMatrix[ratingData.movie_id.values-1, ratingData.user_id.values-1] = ratingData.rating.values

    # Subtract Mean off - Normalization
    normalizedMatrix = ratingMatrix - np.asarray([(np.mean(ratingMatrix, 1))]).T

    # Computing SVD
    A = normalizedMatrix.T / np.sqrt(ratingMatrix.shape[0] - 1)
    U, S, V = np.linalg.svd(A)

    return V

def main():
    st.title("Movie Recommendation System Using SVD")

    # Check if model is trained
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = train_model()

    # Display data summary
    ratingData, movieData = load_data()

    st.subheader("Rating Data Summary")
    st.write(ratingData.head())

    st.subheader("Movie Data Summary")
    st.write(movieData.head())

    # User Input for Movie Name
    movie_name = st.text_input("Enter Movie Name:")

    # Find movie ID corresponding to input movie name
    movie_id = None
    if movie_name:
        movie_matches = movieData[movieData['title'].str.contains(movie_name, case=False)]
        if not movie_matches.empty:
            movie_id = movie_matches.iloc[0]['movie_id']

    # Calculate and Display Recommendations
    if movie_id is not None:
        st.subheader("Top Recommendations")
        indexes = similar(st.session_state.model_trained, movie_id, top_n=5)
        for idx in indexes:
            st.write(movieData.loc[idx, 'title'])
    elif movie_name:
        st.write("Movie not found. Please enter a valid movie name.")

if __name__ == "__main__":
    main()
