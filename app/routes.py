from flask import Blueprint, request, jsonify
from app.models import Movie, MovieDataset
from app.recommendation import RecommendationSystem
from app.config import Config
import numpy as np
import json
from app.searching import SearchingSystem
from app.filtering import FilteringSystem
from app.sorting import SortingSystem
from app.media_loading import MediaAndTrailers
from app.evaluation import MLEvaluator
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, linear_kernel

main = Blueprint('main', __name__)


from app.preprocessing import MoviePreprocessor
from app.recommendation import RecommendationSystem
import os
import datetime

# VAŽNO: Koristi train split iz config-a!
USE_TRAIN_SPLIT = Config.USE_TRAIN_SPLIT

movie_class = Movie(movie_data={})
movies = MovieDataset(use_train_split=USE_TRAIN_SPLIT)
preprocessor = MoviePreprocessor(use_train_split=USE_TRAIN_SPLIT)

preprocessor.prepare_data()

ml_evaluator = None
evaluation_results = None
best_algorithm = 'cosine'

if USE_TRAIN_SPLIT:
    import pickle
    with open('dataset_split.pkl', 'rb') as f:
        split_data = pickle.load(f)
    
    train_indices = split_data['train_indices']
    test_indices = split_data['test_indices']
    val_indices = split_data['val_indices']
    
    test_features = preprocessor.create_features_matrix_for_evaluation(preprocessor.test_df)
    val_features = preprocessor.create_features_matrix_for_evaluation(preprocessor.val_df)
    
    full_features = np.zeros((len(preprocessor.movies_df_full), preprocessor.features_matrix.shape[1]))
    full_features[train_indices] = preprocessor.features_matrix
    full_features[test_indices] = test_features
    full_features[val_indices] = val_features
    
    ml_evaluator = MLEvaluator(
        features_matrix=full_features,
        movie_ids=preprocessor.movies_df_full['id'].tolist(),
        movie_titles=preprocessor.movies_df_full['title'].tolist()
    )
    
    ml_evaluator.train_indices = train_indices
    ml_evaluator.test_indices = test_indices
    ml_evaluator.val_indices = val_indices
    
    try:
        start_time = datetime.datetime.now()
        
        comparison = ml_evaluator.compare_all_recommendation_metrics(k=10)
        
        best_result = max(comparison, key=lambda x: x['quality_score'])
        best_algorithm = best_result['metric']
        best_quality_score = best_result['quality_score']
        
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Pripremi detaljan evaluacioni izveštaj
        evaluation_results = {
            'timestamp': start_time.isoformat(),
            'duration_seconds': round(duration, 2),
            'dataset_info': {
                'train_size': len(preprocessor.movies_df),
                'test_size': len(preprocessor.test_df),
                'val_size': len(preprocessor.val_df),
                'total_size': len(preprocessor.movies_df_full),
                'features_dim': preprocessor.features_matrix.shape[1]
            },
            'evaluation_method': {
                'type': 'Train on TRAIN set, evaluate on VALIDATION set',
                'metrics': 'INTRINSIC metrics (ASQ, ILS) - standardne RecSys metrike bez user feedback',
                'metric_description': {
                    'ASQ': 'Average Similarity to Query - relevantnost (0-1, više=bolje)',
                    'ILS': 'Intra-List Similarity - redundansa (0-1, niže=bolje)',
                    'Quality': 'ASQ × (1-ILS) - kombinovano (0-1, više=bolje)'
                },
                'selection_criterion': 'Quality score sa top-10 preporuka na VALIDATION setu',
                'tested_algorithms': ['cosine', 'euclidean', 'dot'],
                'note': 'Test set je rezervisan i nije korišćen tokom optimizacije'
            },
            'best_algorithm': {
                'name': best_algorithm,
                'validation_asq': round(best_result['asq'], 6),
                'validation_ils': round(best_result['ils'], 6),
                'validation_quality_score': round(best_quality_score, 6),
                'quality_std': round(best_result['quality_std'], 6),
                'score_range': '0.0 - 1.0'
            },
            'all_results': [
                {
                    'metric': r['metric'],
                    'asq': round(r['asq'], 6),
                    'asq_std': round(r['asq_std'], 6),
                    'ils': round(r['ils'], 6),
                    'ils_std': round(r['ils_std'], 6),
                    'quality_score': round(r['quality_score'], 6),
                    'quality_std': round(r['quality_std'], 6)
                }
                for r in comparison
            ]
        }
        
        results_file = 'evaluation_results.json'
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        evaluation_results = {
            'error': str(e),
            'fallback_algorithm': 'cosine',
            'timestamp': datetime.datetime.now().isoformat()
        }
        with open('evaluation_results.json', 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, indent=2, ensure_ascii=False)

recommendation_system = RecommendationSystem(preprocessor, best_metric=best_algorithm)
recommendation_cache = {}
recommended_paginated_movies = []
searching_system = SearchingSystem(movies)
filtering_system = FilteringSystem()
sorting_system = SortingSystem()
media_and_trailers = MediaAndTrailers()


def sanitize_movie_data(movies_list):
    sanitized_movies = []
    for movie in movies_list:
        sanitized_movie = {}
        for key, value in movie.items():
            if isinstance(value, float) and (value != value):
                sanitized_movie[key] = None
            else:
                sanitized_movie[key] = value
        sanitized_movies.append(sanitized_movie)
    return sanitized_movies

@main.route('/movies/recommend', methods=['POST'])
def recommend():
    """Vraća preporuke koristeći automatski odabrani najbolji algoritam"""
    data = request.get_json()
    movie_title = data.get('movieTitle', '')
    if movie_title not in recommendation_cache:
        recommendations = recommendation_system.get_recommendations(movie_title, k=100)
        sanitized_recommendations = sanitize_movie_data(recommendations)
        recommendation_cache[movie_title] = sanitized_recommendations
    else:
        sanitized_recommendations = recommendation_cache[movie_title]

    page = int(request.args.get('page', 1))
    per_page = 12
    start = (page - 1) * per_page
    end = start + per_page
    paginated_recommendations = sanitized_recommendations[start:end]
    total_pages = (len(sanitized_recommendations) + per_page - 1) // per_page

    return jsonify({
        'recommendations': paginated_recommendations,
        'total_pages': total_pages,
        'algorithm_used': recommendation_system.best_metric if USE_TRAIN_SPLIT else 'cosine',
        'ml_optimized': USE_TRAIN_SPLIT
    }), 200, {'Content-Type': 'application/json'}


@main.route('/movies/search', methods=['POST'])
def search():
    data = request.get_json()
    movie_title = data.get('query', '')
    
    search_results = searching_system.get_results(movie_title)
    results_dict = [movie.to_dict() for movie in search_results]

    sanitized_search_results = sanitize_movie_data(results_dict)

    page = int(request.args.get('page', 1))
    per_page = 12
    start = (page - 1) * per_page
    end = start + per_page
    paginated_results = sanitized_search_results[start:end]
    total_pages = (len(sanitized_search_results) + per_page - 1) // per_page

    return jsonify({
        'searching results': paginated_results,
        'total_pages': total_pages
    }), 200, {'Content-Type': 'application/json'}

@main.route('/movies/topRated', methods=['GET'])
def get_top_rated():
    movies_list = sorted([movie.to_dict() for movie in movies.get_movies()], key=lambda x: x.get('vote_average', 0), reverse=True)
    sanitized_movies_list = sanitize_movie_data(movies_list)
    page = int(request.args.get('page', 1))
    per_page = 12
    start = (page - 1) * per_page
    end = start + per_page
    paginated_movies = sanitized_movies_list[start:end]
    total_pages = (len(sanitized_movies_list) + per_page - 1) // per_page

    return jsonify({
        'movies': paginated_movies,
        'total_pages': total_pages
    }), 200, {'Content-Type': 'application/json'}


@main.route('/movies/<int:movie_id>', methods=['GET'])
def get_movie_details(movie_id):
    movie = movies.get_movie_by_id(movie_id)

    if movie is None:
        return jsonify({'error': 'Movie not found'}), 404
    
    movie_dict = movie.to_dict()
    sanitized_movie = sanitize_movie_data([movie_dict])[0]

    return jsonify(sanitized_movie), 200, {'Content-Type': 'application/json'}


@main.route('/movies/filter', methods=['POST'])
def filter_movies():
    data = request.get_json()
    genre = data.get('genre')
    language = data.get('language')
    movie_title = data.get('movie_title')
    sanitized_recommendations = recommendation_cache.get(movie_title, [])
    filtering_system = FilteringSystem()
    filtered_movies = filtering_system.filter_movies(genre, language, sanitized_recommendations)
    filtered_movies_dicts = [movie.to_dict() for movie in filtered_movies]
    page = int(request.args.get('page', 1))
    per_page = 12
    start = (page - 1) * per_page
    end = start + per_page
    paginated_filtered_movies = filtered_movies_dicts[start:end]
    total_pages = (len(filtered_movies_dicts) + per_page - 1) // per_page

    return jsonify({
        'filtered_movies': paginated_filtered_movies,
        'total_pages': total_pages
    }), 200, {'Content-Type': 'application/json'}


@main.route('/movies/filtering/metadata', methods=['GET'])
def get_metadata():
    genres = set()
    languages = set()

    for movie in movies.get_movies():
        for genre in movie.genres:
            genre_name = genre.get('name', '').strip().lower()
            if genre_name:
                genres.add(genre_name)

        for spoken_lang in movie.spoken_languages:
            language_name = spoken_lang.get('name', '').strip().lower()
            if language_name and language_name.isalpha():
                languages.add(language_name)

    return jsonify({
        'genres': sorted(genres),
        'spoken_languages': sorted(languages)
    }), 200, {'Content-Type': 'application/json'}

@main.route('/movies/sort', methods=['POST'])
def sort_movies():
    data = request.get_json()
    sort_criteria = data.get('sort_criteria', [])
    movie_title = data.get('movie_title')

    sanitized_recommendations = recommendation_cache.get(movie_title, [])

    movie_objects_recommendation = [Movie(movie) for movie in sanitized_recommendations]

    if not sort_criteria:
        return jsonify({'error': 'No sorting criteria provided'}), 400

    sorted_movies = sorting_system.sort_movies(movie_objects_recommendation, sort_criteria)

    sanitized_sorted_movies = sanitize_movie_data([movie.to_dict() for movie in sorted_movies])

    page = int(request.args.get('page', 1)) 
    per_page = 12
    start = (page - 1) * per_page
    end = start + per_page
    paginated_sorted_movies = sanitized_sorted_movies[start:end]
    total_pages = (len(sanitized_sorted_movies) + per_page - 1) // per_page
   
    return jsonify({
        'sorted_movies': paginated_sorted_movies,
        'total_pages': total_pages
    }), 200


@main.route('/movies/<int:movie_id>/poster', methods=['GET'])
def get_movie_poster(movie_id):
    movie_poster = media_and_trailers.fetch_movie_poster(movie_id)
    if movie_poster:
        return jsonify({"movie_poster": movie_poster}), 200
    else:
        return jsonify({"error": "Movie not found"}), 404
    

@main.route('/movies/<int:movie_id>/trailers', methods=['GET'])
def get_movie_trailers(movie_id):
    trailers = media_and_trailers.fetch_trailers(movie_id)
    if trailers:
        return jsonify({"trailers": trailers}), 200
    else:
        return jsonify({"error": "Movie not found or no trailers available"}), 404

