import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.preprocessing import MoviePreprocessor

class RecommendationSystem:
    def __init__(self):
        self.preprocessor = MoviePreprocessor()
        self.preprocessor = MoviePreprocessor()
        self.preprocessor.prepare_data() 
        self.features_matrix = self.preprocessor.get_features_matrix()
        self.movie_titles = self.preprocessor.movies_df['title'].tolist()
        self.similar_movies = []

    def get_recommendations(self, movie_title, k=10):
        movie_row = self.preprocessor.movies_df[self.preprocessor.movies_df['title'].str.lower() == movie_title.lower()]
        print("Movie row",movie_row)
        input_index = movie_row.index[0]
        print("Input index",input_index)
        cosine_sim = cosine_similarity([self.features_matrix[input_index]], self.features_matrix)
        similar_indices = cosine_sim[0].argsort()[-k-1:-1][::-1]
        for idx in similar_indices:
            self.similar_movies.append(self.preprocessor.movies_df.iloc[idx]['title'])
        return self.similar_movies
