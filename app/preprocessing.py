import ast
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
from app.models import MovieDataset
from app.config import Config

class MoviePreprocessor:
    def __init__(self, file_path=Config.DATA_PATH):
        self.movies_df_old = pd.read_csv(file_path)
        self.movies_df = pd.read_csv(file_path)
        self.sentence_model = None
        self.features_matrix = None
        self.embeddings_cache_file = "embeddings_cache.pkl"

    def parse_list_column(self, column_name):
        parsed_list = []
        for i in range(len(self.movies_df)):
            items = ast.literal_eval(self.movies_df[column_name].iloc[i])
            parsed_list.append(' '.join([item['name'] for item in items]))
        return parsed_list

    def prepare_data(self):
        self.movies_df['genres'] = self.parse_list_column('genres')
        self.movies_df['keywords'] = self.parse_list_column('keywords')
        self.movies_df['overview'] = self.movies_df['overview'].fillna('')
        
        if os.path.exists(self.embeddings_cache_file):
            print("[INFO] Učitavanje embeddings iz keša...")
            with open(self.embeddings_cache_file, 'rb') as f:
                cached_embeddings = pickle.load(f)
            self.movies_df['embedding'] = cached_embeddings
            print("[INFO] Embeddings učitani iz keša.")
        else:
            print("[INFO] Učitavanje Sentence-BERT modela...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[INFO] Model učitan.")
            
            print("[INFO] Generisanje embeddings za filmove...")
            combined_texts = [f"{row['title']}. {row['overview']}" 
                            for _, row in self.movies_df.iterrows()]
            
            embeddings = self.sentence_model.encode(combined_texts, 
                                                   batch_size=32, 
                                                   show_progress_bar=True)
            
            self.movies_df['embedding'] = list(embeddings)
            print("[INFO] Embeddings generisani.")
            
            print("[INFO] Čuvanje embeddings u keš...")
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump(list(embeddings), f)
            print("[INFO] Embeddings sačuvani u keš.")

        self.create_features_matrix()

    def get_sentence_embedding(self, title, overview):
        if self.sentence_model is None:
            return np.zeros(384)
        
        combined_text = f"{title}. {overview}"
        embedding = self.sentence_model.encode(combined_text, show_progress_bar=False)
        return embedding

    def create_features_matrix(self):
        numerical_features = ['popularity', 'vote_average', 'vote_count', 'revenue']
        scaler = MinMaxScaler()
        self.movies_df[numerical_features] = scaler.fit_transform(self.movies_df[numerical_features])

        count_vectorizer_genres = CountVectorizer(stop_words='english', max_features=50)
        genres_matrix = count_vectorizer_genres.fit_transform(self.movies_df['genres'])
        
        count_vectorizer_keywords = CountVectorizer(stop_words='english', max_features=50)
        keywords_matrix = count_vectorizer_keywords.fit_transform(self.movies_df['keywords'])
        
        count_vectorizer_overview = CountVectorizer(stop_words='english', max_features=50)
        overview_matrix = count_vectorizer_overview.fit_transform(self.movies_df['overview'])

        embedding_matrix = np.vstack(self.movies_df['embedding'].values) * 3
        
        self.features_matrix = np.hstack((
            genres_matrix.toarray(),
            keywords_matrix.toarray(),
            overview_matrix.toarray(),
            embedding_matrix,
            self.movies_df[numerical_features].values
        ))

    def get_features_matrix(self):
        return self.features_matrix


if __name__ == "__main__":
    preprocessor = MoviePreprocessor()
    preprocessor.prepare_data()
    features_matrix = preprocessor.get_features_matrix()
