import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from gensim.models import Word2Vec, KeyedVectors
from gensim.test.utils import common_texts
from app.models import MovieDataset
from app.config import Config

class MoviePreprocessor:
    def __init__(self, file_path=Config.DATA_PATH):
        self.movies_df_old = pd.read_csv(file_path)
        self.movies_df = pd.read_csv(file_path)
        self.word_vectors = None
        self.features_matrix = None

    def parse_list_column(self, column_name):
        """Parse list columns in the DataFrame."""
        parsed_list = []
        for i in range(len(self.movies_df)):
            items = ast.literal_eval(self.movies_df[column_name].iloc[i])
            parsed_list.append(' '.join([item['name'] for item in items]))
        return parsed_list

    def prepare_data(self):
        """Prepare the movie dataset by parsing columns and creating embeddings."""
        self.movies_df['genres'] = self.parse_list_column('genres')
        self.movies_df['keywords'] = self.parse_list_column('keywords')
        
        self.word_vectors = Word2Vec(sentences=common_texts, vector_size=100, window=5, min_count=1, workers=4)
        self.word_vectors.wv.save("word2vec.wordvectors")
        self.word_vectors = KeyedVectors.load("word2vec.wordvectors", mmap='r')

        self.movies_df['overview'] = self.movies_df['overview'].fillna('')
        self.movies_df['embedding'] = self.movies_df.apply(
            lambda row: self.get_sentence_embedding(row['title'], row['overview']),
            axis=1
        )

        self.create_features_matrix()

    def get_sentence_embedding(self, title, overview, overview_weight=2):
        """Generate sentence embeddings for title and overview."""
        title_words = title.split()
        overview_words = overview.split()

        title_embeddings = [self.word_vectors[word] for word in title_words if word in self.word_vectors]
        overview_embeddings = [self.word_vectors[word] for word in overview_words if word in self.word_vectors]

        if title_embeddings or overview_embeddings:
            combined_embeddings = title_embeddings + overview_embeddings * overview_weight
            return np.mean(combined_embeddings, axis=0)
        else:
            return np.zeros(self.word_vectors.vector_size)

    def create_features_matrix(self):
        """Create a feature matrix combining various features."""
        numerical_features = ['popularity', 'vote_average', 'vote_count', 'revenue']
        scaler = MinMaxScaler()
        self.movies_df[numerical_features] = scaler.fit_transform(self.movies_df[numerical_features])

        # Combine features
        count_vectorizer = CountVectorizer(stop_words='english')
        count_matrix = count_vectorizer.fit_transform(self.movies_df['genres'] + ' ' + self.movies_df['keywords'] + ' ' + self.movies_df['overview'])

        embedding_matrix = np.vstack(self.movies_df['embedding'].values)
        self.features_matrix = np.hstack((count_matrix.toarray(), embedding_matrix, self.movies_df[numerical_features].values))

    def get_features_matrix(self):
        """Return the features matrix for use in the recommendation system."""
        return self.features_matrix

# Usage
if __name__ == "__main__":
    preprocessor = MoviePreprocessor()
    preprocessor.prepare_data()
    features_matrix = preprocessor.get_features_matrix()
