import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

class MLEvaluator:
    
    def __init__(self, features_matrix, movie_ids, movie_titles):
        self.features_matrix = features_matrix
        self.movie_ids = np.array(movie_ids)
        self.movie_titles = np.array(movie_titles)
        self.n_movies = len(movie_ids)
        
        self.train_indices = None
        self.test_indices = None
        self.val_indices = None
    
    def evaluate_recommendation_quality(self, metric='cosine', k=10, use_test=False):
        if self.train_indices is None or self.val_indices is None:
            raise ValueError("Train i validation indeksi moraju biti postavljeni!")
        
        eval_indices = self.test_indices if use_test else self.val_indices
        eval_set_name = "TEST" if use_test else "VALIDATION"
        
        if eval_indices is None:
            raise ValueError(f"{eval_set_name} indeksi moraju biti postavljeni!")
        
        eval_features = self.features_matrix[eval_indices]
        train_features = self.features_matrix[self.train_indices]
        
        asq_scores = []
        ils_scores = []
        
        for i in range(len(eval_features)):
            eval_movie_features = eval_features[i].reshape(1, -1)
            
            # Izračunaj similarity sa TRAIN setom
            if metric == 'cosine':
                sims = cosine_similarity(eval_movie_features, train_features)[0]
                sims = np.clip(sims, 0, 1)
            elif metric == 'euclidean':
                distances = euclidean_distances(eval_movie_features, train_features)[0]
                max_dist = np.max(distances)
                if max_dist > 0:
                    sims = 1 - (distances / max_dist)
                else:
                    sims = np.ones_like(distances)
            elif metric == 'dot':
                sims_raw = np.dot(train_features, eval_movie_features.T).flatten()
                min_sim = np.min(sims_raw)
                max_sim = np.max(sims_raw)
                if max_sim - min_sim > 0:
                    sims = (sims_raw - min_sim) / (max_sim - min_sim)
                else:
                    sims = np.zeros_like(sims_raw)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            top_k_indices = np.argsort(sims)[-k:][::-1]
            top_k_sims = sims[top_k_indices]
            
            asq = np.mean(top_k_sims)
            asq_scores.append(asq)
            
            top_k_features = train_features[top_k_indices]
            
            if k > 1:
                if metric == 'cosine':
                    intra_sim_matrix = cosine_similarity(top_k_features)
                    intra_sim_matrix = np.clip(intra_sim_matrix, 0, 1)
                elif metric == 'euclidean':
                    intra_dist_matrix = euclidean_distances(top_k_features)
                    max_dist = np.max(intra_dist_matrix)
                    if max_dist > 0:
                        intra_sim_matrix = 1 - (intra_dist_matrix / max_dist)
                    else:
                        intra_sim_matrix = np.ones_like(intra_dist_matrix)
                elif metric == 'dot':
                    intra_sim_raw = np.dot(top_k_features, top_k_features.T)
                    min_sim = np.min(intra_sim_raw)
                    max_sim = np.max(intra_sim_raw)
                    if max_sim - min_sim > 0:
                        intra_sim_matrix = (intra_sim_raw - min_sim) / (max_sim - min_sim)
                    else:
                        intra_sim_matrix = np.zeros_like(intra_sim_raw)
                
                ils = np.mean(intra_sim_matrix[np.triu_indices_from(intra_sim_matrix, k=1)])
                ils_scores.append(ils)
            else:
                ils_scores.append(0.0)
        
        asq_mean = np.mean(asq_scores)
        asq_std = np.std(asq_scores)
        ils_mean = np.mean(ils_scores)
        ils_std = np.std(ils_scores)
        
        quality_scores = [asq * (1 - ils) for asq, ils in zip(asq_scores, ils_scores)]
        quality_mean = np.mean(quality_scores)
        quality_std = np.std(quality_scores)
        
        return {
            'metric': metric,
            'asq': float(asq_mean),
            'asq_std': float(asq_std),
            'ils': float(ils_mean),
            'ils_std': float(ils_std),
            'quality_score': float(quality_mean),
            'quality_std': float(quality_std),
            'k': k,
            'n_samples': len(eval_features),
            'eval_set': eval_set_name,
            'interpretation': {
                'asq': 'Viša ASQ = bolja relevantnost (0.0-1.0)',
                'ils': 'Niža ILS = bolja diverznost (0.0-1.0)',
                'quality': 'Quality = ASQ × (1-ILS), balansira obe metrike (0.0-1.0)'
            }
        }
    
    def cross_validate_recommendations(self, k=10):
        results = []
        for metric in ['cosine', 'euclidean', 'dot']:
            result = self.evaluate_recommendation_quality(metric=metric, k=k, use_test=False)
            results.append(result)
            
            print(f"\n{metric.upper()} similarity (k={k}):")
            print(f"  ASQ (relevantnost): {result['asq']:.4f} ± {result['asq_std']:.4f}")
            print(f"  ILS (redundansa):   {result['ils']:.4f} ± {result['ils_std']:.4f}")
            print(f"  Quality Score:      {result['quality_score']:.4f} ± {result['quality_std']:.4f}")
            print(f"  {result['interpretation']['quality']}")
        
        return results
    
    def final_test_evaluation(self, best_metric, k=10):
        print("\n" + "#"*70)
        print("#" + " "*68 + "#")
        print("#" + "  FINALNA EVALUACIJA NA TEST SETU - NIKAD VIĐENI PODACI".center(68) + "#")
        print("#" + " "*68 + "#")
        print("#"*70)
        print(f"\nVažno: Ovo je JEDINI put da se koristi TEST set!")
        print(f"Metric: {best_metric.upper()}")
        print(f"Top-k: {k}\n")
        
        result = self.evaluate_recommendation_quality(metric=best_metric, k=k, use_test=True)
        
        print("\n" + "="*70)
        print("                 FINALNI REZULTATI")
        print("="*70)
        print(f"\nDataset info:")
        print(f"  - Evaluation samples: {result['n_samples']} test filmova")
        print(f"  - Features: {self.features_matrix.shape[1]} dimenzija")
        print(f"  - Metric: {best_metric.upper()} similarity")
        
        print(f"\nMETRIKE (INTRINSIC - bez user feedback):")
        print(f"\n  1. ASQ (Average Similarity to Query) - RELEVANTNOST")
        print(f"     Score: {result['asq']:.4f} ± {result['asq_std']:.4f}")
        print(f"     {result['interpretation']['asq']}")
        
        print(f"\n  2. ILS (Intra-List Similarity) - DIVERZNOST")
        print(f"     Score: {result['ils']:.4f} ± {result['ils_std']:.4f}")
        print(f"     {result['interpretation']['ils']}")
        
        print(f"\n  3. Combined Quality Score")
        print(f"     Score: {result['quality_score']:.4f} ± {result['quality_std']:.4f}")
        print(f"     {result['interpretation']['quality']}")
        
        print("\n" + "="*70)
        print(f"INTERPRETACIJA:")
        if result['quality_score'] >= 0.7:
            kvalitet = "ODLIČAN"
        elif result['quality_score'] >= 0.5:
            kvalitet = "DOBAR"
        elif result['quality_score'] >= 0.3:
            kvalitet = "PROSEČAN"
        else:
            kvalitet = "SLAB"
        
        print(f"  Ukupan kvalitet: {kvalitet} ({result['quality_score']:.4f})")
        print(f"  Relevantnost (ASQ): {'Visoka' if result['asq'] >= 0.6 else 'Srednja' if result['asq'] >= 0.4 else 'Niska'}")
        print(f"  Diverznost (1-ILS): {'Visoka' if result['ils'] <= 0.4 else 'Srednja' if result['ils'] <= 0.6 else 'Niska'}")
        print("="*70 + "\n")
        
        return result
    
    def compare_all_recommendation_metrics(self, k=10):
        print("\n" + "="*60)
        print("      EVALUACIJA KVALITETA PREPORUKA - VALIDATION SET")
        print("      Standardne INTRINSIC metrike (bez user feedback)")
        print("="*60)
        print(f"Dataset: {len(self.train_indices)} train / {len(self.val_indices)} val / {len(self.test_indices)} test")
        print(f"Features: {self.features_matrix.shape[1]} dimenzija")
        print(f"Top-k: {k} preporuka po query filmu\n")
        
        results = self.cross_validate_recommendations(k=k)
        
        print("\n" + "="*60)
        print("            UPOREDNA ANALIZA SVIH METRIKA")
        print("="*60)
        print(f"{'Metrika':<12} {'ASQ (rel.)':<15} {'ILS (red.)':<15} {'Quality':<12}")
        print("-" * 60)
        
        for result in results:
            metric_name = result['metric'].upper()
            asq_str = f"{result['asq']:.4f}±{result['asq_std']:.3f}"
            ils_str = f"{result['ils']:.4f}±{result['ils_std']:.3f}"
            quality_str = f"{result['quality_score']:.4f}±{result['quality_std']:.3f}"
            print(f"{metric_name:<12} {asq_str:<15} {ils_str:<15} {quality_str:<12}")
        
        best_result = max(results, key=lambda x: x['quality_score'])
        best_metric = best_result['metric'].upper()
        
        print("\n" + "="*60)
        print(f"NAJBOLJA METRIKA: {best_metric}")
        print(f"  ASQ (relevantnost): {best_result['asq']:.4f} - VIŠE je BOLJE")
        print(f"  ILS (redundansa):   {best_result['ils']:.4f} - NIŽE je BOLJE")
        print(f"  Quality Score:      {best_result['quality_score']:.4f}")
        print("="*60 + "\n")
        
        return results
