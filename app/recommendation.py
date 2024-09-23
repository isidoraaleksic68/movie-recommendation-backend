import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from app.preprocessing import MoviePreprocessor
from app.models import MovieDataset

class RecommendationSystem:
    def __init__(self):
        self.preprocessor = MoviePreprocessor()
        self.movie_data_set = MovieDataset()
        self.preprocessor.prepare_data() 
        self.features_matrix = self.preprocessor.get_features_matrix()
        self.movie_titles = self.preprocessor.movies_df['title'].tolist()
        self.similar_movies = []

    
    def get_recommendations(self, movie_title, k=10):
        movie_row = self.preprocessor.movies_df[self.preprocessor.movies_df['title'].str.lower() == movie_title.lower()]
        input_index = movie_row.index[0]
        cosine_sim = cosine_similarity([self.features_matrix[input_index]], self.features_matrix)
        similar_indices = cosine_sim[0].argsort()[-k-1:-1][::-1]
        self.similar_movies = []
        for idx in similar_indices:
            movie_data = self.preprocessor.movies_df.iloc[idx]
            movie_dict = movie_data.to_dict()
            movie_dict.pop('embedding', None)
            self.similar_movies.append(movie_dict)

        return self.movie_data_set.get_similar_movies_objects(self.similar_movies)
