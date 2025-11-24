import ast
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from app.models import MovieDataset
from app.config import Config

class MoviePreprocessor:
    def __init__(self, file_path=Config.DATA_PATH, use_train_split=True):
        self.movies_df_full = pd.read_csv(file_path)  # Ceo dataset
        self.movies_df = None  # Train set (ili full ako use_train_split=False)
        self.test_df = None
        self.val_df = None
        self.sentence_model = None
        self.features_matrix = None
        self.embeddings_cache_file = "embeddings_cache.pkl"
        self.split_cache_file = "dataset_split.pkl"
        self.use_train_split = use_train_split
        
        # Fitted transformers (učeni SAMO na train setu)
        self.scaler = None
        self.count_vectorizer_genres = None
        self.count_vectorizer_keywords = None
        self.count_vectorizer_overview = None
        
        # Učitaj ili kreiraj split
        if use_train_split:
            self._load_or_create_split()
        else:
            self.movies_df = self.movies_df_full.copy()
    
    def _load_or_create_split(self, train_size=0.6, test_size=0.2, val_size=0.2, random_state=42):
        """
        Učitava postojeći split ili kreira novi i čuva ga.
        Train set se koristi za aplikaciju, test i val samo za evaluaciju.
        """
        if os.path.exists(self.split_cache_file):
            print("[INFO] Učitavanje train/test/val split iz keša...")
            with open(self.split_cache_file, 'rb') as f:
                split_data = pickle.load(f)
            
            train_indices = split_data['train_indices']
            test_indices = split_data['test_indices']
            val_indices = split_data['val_indices']
            
            self.movies_df = self.movies_df_full.iloc[train_indices].reset_index(drop=True)
            self.test_df = self.movies_df_full.iloc[test_indices].reset_index(drop=True)
            self.val_df = self.movies_df_full.iloc[val_indices].reset_index(drop=True)
            
            print(f"[INFO] Split učitan: Train={len(train_indices)}, Test={len(test_indices)}, Val={len(val_indices)}")
        else:
            print("[INFO] Kreiranje novog train/test/val split-a...")
            indices = np.arange(len(self.movies_df_full))
            
            # Train + temp split
            train_indices, temp_indices = train_test_split(
                indices, train_size=train_size, random_state=random_state
            )
            
            # Test i validation split
            relative_test_size = test_size / (test_size + val_size)
            test_indices, val_indices = train_test_split(
                temp_indices, train_size=relative_test_size, random_state=random_state
            )
            
            self.movies_df = self.movies_df_full.iloc[train_indices].reset_index(drop=True)
            self.test_df = self.movies_df_full.iloc[test_indices].reset_index(drop=True)
            self.val_df = self.movies_df_full.iloc[val_indices].reset_index(drop=True)
            
            # Sačuvaj split
            split_data = {
                'train_indices': train_indices,
                'test_indices': test_indices,
                'val_indices': val_indices
            }
            with open(self.split_cache_file, 'wb') as f:
                pickle.dump(split_data, f)
            
            print(f"[INFO] Split kreiran i sačuvan:")
            print(f"       Train: {len(train_indices)} filmova ({len(train_indices)/len(self.movies_df_full)*100:.1f}%)")
            print(f"       Test:  {len(test_indices)} filmova ({len(test_indices)/len(self.movies_df_full)*100:.1f}%)")
            print(f"       Val:   {len(val_indices)} filmova ({len(val_indices)/len(self.movies_df_full)*100:.1f}%)")
    
    def get_train_data(self):
        """Vraća train DataFrame"""
        return self.movies_df
    
    def get_test_data(self):
        """Vraća test DataFrame"""
        return self.test_df
    
    def get_val_data(self):
        """Vraća validation DataFrame"""
        return self.val_df
    
    def get_full_data(self):
        """Vraća ceo dataset (za evaluaciju)"""
        return self.movies_df_full

    def parse_list_column(self, column_name):
        parsed_list = []
        for i in range(len(self.movies_df)):
            items = ast.literal_eval(self.movies_df[column_name].iloc[i])
            parsed_list.append(' '.join([item['name'] for item in items]))
        return parsed_list

    def prepare_data(self):
        """
        Priprema podatke:
        1. Parsiranje i cleaning - za SVE setove (train/test/val)
        2. Embeddings - generiše se za ceo dataset jednom
        3. FIT transformera - SAMO na train setu
        4. Feature matrica - kreira se samo za train set
        """
        # 1. PARSIRANJE - za ceo dataset (train će se preprocesirati, test/val su već pripremljeni)
        print("[INFO] Parsiranje podataka za train set...")
        self.movies_df['genres'] = self.parse_list_column('genres')
        self.movies_df['keywords'] = self.parse_list_column('keywords')
        self.movies_df['overview'] = self.movies_df['overview'].fillna('')
        
        # Ako koristimo split, parsiranje za test i val
        if self.use_train_split and self.test_df is not None:
            print("[INFO] Parsiranje podataka za test set...")
            self.test_df['genres'] = self._parse_list_for_df(self.test_df, 'genres')
            self.test_df['keywords'] = self._parse_list_for_df(self.test_df, 'keywords')
            self.test_df['overview'] = self.test_df['overview'].fillna('')
            
            print("[INFO] Parsiranje podataka za validation set...")
            self.val_df['genres'] = self._parse_list_for_df(self.val_df, 'genres')
            self.val_df['keywords'] = self._parse_list_for_df(self.val_df, 'keywords')
            self.val_df['overview'] = self.val_df['overview'].fillna('')
        
        # 2. EMBEDDINGS - generiši ili učitaj za ceo dataset
        if os.path.exists(self.embeddings_cache_file):
            print("[INFO] Učitavanje embeddings iz keša...")
            with open(self.embeddings_cache_file, 'rb') as f:
                all_embeddings = pickle.load(f)
            
            # Dodeli embeddings na osnovu indeksa
            if self.use_train_split:
                self._assign_embeddings_from_cache(all_embeddings)
            else:
                self.movies_df['embedding'] = all_embeddings
            print("[INFO] Embeddings učitani iz keša.")
        else:
            print("[INFO] Učitavanje Sentence-BERT modela...")
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("[INFO] Model učitan.")
            
            print("[INFO] Generisanje embeddings za CEO DATASET...")
            combined_texts = [f"{row['title']}. {row['overview']}" 
                            for _, row in self.movies_df_full.iterrows()]
            
            embeddings = self.sentence_model.encode(combined_texts, 
                                                   batch_size=32, 
                                                   show_progress_bar=True)
            
            # Sačuvaj embeddings za ceo dataset
            print("[INFO] Čuvanje embeddings u keš...")
            with open(self.embeddings_cache_file, 'wb') as f:
                pickle.dump(list(embeddings), f)
            print("[INFO] Embeddings sačuvani u keš.")
            
            # Dodeli embeddings
            if self.use_train_split:
                self._assign_embeddings_from_cache(list(embeddings))
            else:
                self.movies_df['embedding'] = list(embeddings)

        # 3. FIT transformera i kreiraj feature matricu - SAMO za train set
        self.create_features_matrix()
    
    def _parse_list_for_df(self, df, column_name):
        """Helper za parsiranje liste u proizvoljan DataFrame"""
        parsed_list = []
        for i in range(len(df)):
            items = ast.literal_eval(df[column_name].iloc[i])
            parsed_list.append(' '.join([item['name'] for item in items]))
        return parsed_list
    
    def _assign_embeddings_from_cache(self, all_embeddings):
        """Dodeljuje embeddings iz cache-a na train/test/val setove"""
        # Učitaj split indekse
        with open(self.split_cache_file, 'rb') as f:
            split_data = pickle.load(f)
        
        train_indices = split_data['train_indices']
        test_indices = split_data['test_indices']
        val_indices = split_data['val_indices']
        
        # Dodeli embeddings
        self.movies_df['embedding'] = [all_embeddings[i] for i in train_indices]
        if self.test_df is not None:
            self.test_df['embedding'] = [all_embeddings[i] for i in test_indices]
        if self.val_df is not None:
            self.val_df['embedding'] = [all_embeddings[i] for i in val_indices]

    def get_sentence_embedding(self, title, overview):
        if self.sentence_model is None:
            return np.zeros(384)
        
        combined_text = f"{title}. {overview}"
        embedding = self.sentence_model.encode(combined_text, show_progress_bar=False)
        return embedding

    def create_features_matrix(self):
        """
        Kreira feature matricu SAMO za train set (ili full ako use_train_split=False).
        FIT transformere SAMO na train setu!
        """
        print(f"[INFO] Kreiranje feature matrice za {len(self.movies_df)} filmova (train set)...")
        
        numerical_features = ['popularity', 'vote_average', 'vote_count', 'revenue']
        
        # FIT scaler SAMO na train setu
        self.scaler = MinMaxScaler()
        self.movies_df[numerical_features] = self.scaler.fit_transform(self.movies_df[numerical_features])

        # FIT vectorizers SAMO na train setu
        self.count_vectorizer_genres = CountVectorizer(stop_words='english', max_features=50)
        genres_matrix = self.count_vectorizer_genres.fit_transform(self.movies_df['genres'])
        
        self.count_vectorizer_keywords = CountVectorizer(stop_words='english', max_features=50)
        keywords_matrix = self.count_vectorizer_keywords.fit_transform(self.movies_df['keywords'])
        
        self.count_vectorizer_overview = CountVectorizer(stop_words='english', max_features=50)
        overview_matrix = self.count_vectorizer_overview.fit_transform(self.movies_df['overview'])

        embedding_matrix = np.vstack(self.movies_df['embedding'].values) * 3
        
        self.features_matrix = np.hstack((
            genres_matrix.toarray(),
            keywords_matrix.toarray(),
            overview_matrix.toarray(),
            embedding_matrix,
            self.movies_df[numerical_features].values
        ))
        
        print(f"[INFO] Feature matrica kreirana: {self.features_matrix.shape}")
        print(f"[INFO] Transformeri FIT-ovani na train setu i sačuvani za test/val transformaciju.")
    
    def create_features_matrix_for_evaluation(self, df_subset):
        """
        Kreira feature matricu za test ili validation set - za evaluaciju.
        KORISTI već fit-ovane transformere sa train seta (TRANSFORM SAMO, bez FIT)!
        """
        if self.scaler is None or self.count_vectorizer_genres is None:
            raise ValueError("Transformeri nisu fit-ovani! Pozovi create_features_matrix() prvo.")
        
        numerical_features = ['popularity', 'vote_average', 'vote_count', 'revenue']
        
        # TRANSFORM (bez fit!) - koristi parametre naučene sa train seta
        df_subset_scaled = df_subset.copy()
        df_subset_scaled[numerical_features] = self.scaler.transform(df_subset[numerical_features])

        # TRANSFORM vectorizers (bez fit!)
        genres_matrix = self.count_vectorizer_genres.transform(df_subset['genres'])
        keywords_matrix = self.count_vectorizer_keywords.transform(df_subset['keywords'])
        overview_matrix = self.count_vectorizer_overview.transform(df_subset['overview'])

        embedding_matrix = np.vstack(df_subset['embedding'].values) * 3
        
        features_matrix = np.hstack((
            genres_matrix.toarray(),
            keywords_matrix.toarray(),
            overview_matrix.toarray(),
            embedding_matrix,
            df_subset_scaled[numerical_features].values
        ))
        
        return features_matrix

    def get_features_matrix(self):
        return self.features_matrix


if __name__ == "__main__":
    preprocessor = MoviePreprocessor()
    preprocessor.prepare_data()
    features_matrix = preprocessor.get_features_matrix()
