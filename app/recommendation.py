import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, linear_kernel

class RecommendationSystem:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.preprocessor.prepare_data()
        self.features_matrix = self.preprocessor.get_features_matrix()
        self.movie_titles = self.preprocessor.movies_df['title'].tolist()
        
        # metrika se bira kasnije (lenja inicijalizacija)
        self.best_metric = None  

    def choose_best_metric(self):
        """
        Odaberi najbolju metricu jednostavnijim kriterijumom:
        probaj cosine, euclidean i dot,
        izaberi onu koja daje najveću prosečnu razliku u skorovima
        (što znači bolju separaciju između filmova).
        """
        metrics = ["cosine", "euclidean", "dot"]
        avg_scores = {}

        for metric in metrics:
            try:
                if metric == "cosine":
                    sims = cosine_similarity(self.features_matrix)
                elif metric == "euclidean":
                    sims = -euclidean_distances(self.features_matrix)
                elif metric == "dot":
                    sims = linear_kernel(self.features_matrix)
                else:
                    continue

                # jednostavan kriterijum: veći raspon = bolja separacija
                score = float(np.std(sims))  
                avg_scores[metric] = score
                print(f"[EVAL] {metric} std = {score:.4f}")
            except Exception as e:
                print(f"[WARN] Evaluacija nije uspela za {metric}: {e}")
                avg_scores[metric] = -1

        self.best_metric = max(avg_scores, key=avg_scores.get)
        print(f"[INFO] Odabrana metrika: {self.best_metric}")

    def safe_convert(self, value):
        if isinstance(value, (np.int64, np.int32)):
            return int(value)
        if isinstance(value, (np.float64, np.float32, np.float16)):
            return float(value)
        if isinstance(value, (np.ndarray,)):
            return value.tolist()
        return value

    def get_recommendations(self, movie_title, k=10):
        # Ako prvi put pozivamo, tek sada biramo metricu
        if self.best_metric is None:
            self.choose_best_metric()

        movie_row = self.preprocessor.movies_df[
            self.preprocessor.movies_df['title'].str.lower() == movie_title.lower()
        ]
        
        if movie_row.empty:
            return []

        input_index = movie_row.index[0]

        if self.best_metric == "cosine":
            sims = cosine_similarity([self.features_matrix[input_index]], self.features_matrix)[0]
        elif self.best_metric == "euclidean":
            sims = -euclidean_distances([self.features_matrix[input_index]], self.features_matrix)[0]
        elif self.best_metric == "dot":
            sims = linear_kernel([self.features_matrix[input_index]], self.features_matrix)[0]
        else:
            sims = cosine_similarity([self.features_matrix[input_index]], self.features_matrix)[0]

        similar_indices = sims.argsort()[-k-1:-1][::-1]

        results = []
        for idx in similar_indices:
            movie_data = self.preprocessor.movies_df.iloc[idx].to_dict()
            movie_data = {k: self.safe_convert(v) for k, v in movie_data.items()}
            movie_data["similarity_score"] = float(sims[idx])
            results.append(movie_data)

        return results
