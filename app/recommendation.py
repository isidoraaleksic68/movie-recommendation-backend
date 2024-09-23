from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from app.config import Config

class RecommendationSystem:
    def __init__(self, movies):
        self.movies = movies
        self.tfidf_matrix = self.compute_tfidf_matrix()

    def compute_tfidf_matrix(self):
        df = pd.read_csv(Config.DATA_PATH)
        tfidf = TfidfVectorizer(stop_words='english')
        df['overview'] = df['overview'].fillna('')  # Handle missing overviews
        return tfidf.fit_transform(df['overview'])

    def get_recommendations(self, title):
        df = pd.read_csv(Config.DATA_PATH)
        idx = df.index[df['title'] == title].tolist()
        print(idx)
        if not idx:
            return []

        idx = idx[0]
        cosine_sim = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = sim_scores[1:11]  # Top 10 similar movies
        movie_indices = [i[0] for i in sim_scores]
        return df['title'].iloc[movie_indices].tolist()
