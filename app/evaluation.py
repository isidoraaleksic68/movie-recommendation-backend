from sklearn.metrics import silhouette_score

class Evaluator:
    def __init__(self, features_matrix, labels):
        self.features = features_matrix
        self.labels = labels

    def clustering_quality(self):
        try:
            score = silhouette_score(self.features, self.labels)
            return score
        except Exception:
            return None

    def diversity(self, recommendations):
        genres = [g for movie in recommendations for g in movie["genres"]]
        unique_genres = set([genre["name"].lower() for genre in genres])
        return len(unique_genres) / len(genres) if genres else 0
