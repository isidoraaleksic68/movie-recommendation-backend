from sklearn.metrics import silhouette_score
import numpy as np
from collections import defaultdict

class Evaluator:
    def __init__(self, features_matrix=None, labels=None):
        self.features = features_matrix
        self.labels = labels

    def clustering_quality(self):
        try:
            score = silhouette_score(self.features, self.labels)
            return score
        except Exception:
            return None

    def diversity_distance(self, recommendations, features_matrix, movie_indices):
        from sklearn.metrics.pairwise import cosine_similarity
        
        if len(movie_indices) < 2:
            return 0.0
        
        rec_features = features_matrix[movie_indices]
        
        similarities = cosine_similarity(rec_features)
        
        n = len(movie_indices)
        total_distance = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total_distance += (1 - similarities[i][j])
                count += 1
        
        return total_distance / count if count > 0 else 0.0

    def precision_at_k(self, recommended_ids, relevant_ids, k):
        if k == 0:
            return 0.0
        
        recommended_k = recommended_ids[:k]
        relevant_set = set(relevant_ids)
        
        hits = sum(1 for movie_id in recommended_k if movie_id in relevant_set)
        return hits / k

    def recall_at_k(self, recommended_ids, relevant_ids, k):
        if len(relevant_ids) == 0:
            return 0.0
        
        recommended_k = recommended_ids[:k]
        relevant_set = set(relevant_ids)
        
        hits = sum(1 for movie_id in recommended_k if movie_id in relevant_set)
        return hits / len(relevant_ids)

    def ndcg_at_k(self, recommended_ids, relevant_ids, k):
        if k == 0 or len(relevant_ids) == 0:
            return 0.0
        
        recommended_k = recommended_ids[:k]
        relevant_set = set(relevant_ids)
        
        dcg = 0.0
        for i, movie_id in enumerate(recommended_k):
            if movie_id in relevant_set:
                dcg += 1.0 / np.log2(i + 2)
        
        idcg = 0.0
        for i in range(min(len(relevant_ids), k)):
            idcg += 1.0 / np.log2(i + 2)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg

    def coverage(self, all_recommendations, total_items):
        if total_items == 0:
            return 0.0
        
        unique_items = set()
        for recs in all_recommendations:
            for movie in recs:
                unique_items.add(movie.get('id'))
        
        return len(unique_items) / total_items

    def novelty(self, recommendations, popularity_scores):
        if not recommendations:
            return 0.0
        
        novelty_sum = 0.0
        count = 0
        
        for movie in recommendations:
            movie_id = movie.get('id')
            if movie_id in popularity_scores:
                popularity = popularity_scores[movie_id]
                if popularity > 0:
                    novelty_sum += -np.log2(popularity)
                    count += 1
        
        return novelty_sum / count if count > 0 else 0.0

    def mean_average_precision(self, all_recommended_ids, all_relevant_ids):
        if len(all_recommended_ids) == 0:
            return 0.0
        
        ap_sum = 0.0
        
        for recommended, relevant in zip(all_recommended_ids, all_relevant_ids):
            if len(relevant) == 0:
                continue
            
            relevant_set = set(relevant)
            precision_sum = 0.0
            hits = 0
            
            for i, movie_id in enumerate(recommended):
                if movie_id in relevant_set:
                    hits += 1
                    precision_sum += hits / (i + 1)
            
            if hits > 0:
                ap_sum += precision_sum / len(relevant)
        
        return ap_sum / len(all_recommended_ids)

    def evaluate_recommendations(self, recommendations, relevant_items=None, all_items_count=None, k=10):
        results = {
            'total_recommendations': len(recommendations)
        }
        
        if relevant_items and len(relevant_items) > 0:
            recommended_ids = [movie.get('id') for movie in recommendations]
            results['precision@k'] = self.precision_at_k(recommended_ids, relevant_items, k)
            results['recall@k'] = self.recall_at_k(recommended_ids, relevant_items, k)
            results['ndcg@k'] = self.ndcg_at_k(recommended_ids, relevant_items, k)
        
        return results
