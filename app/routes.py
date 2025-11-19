from flask import Blueprint, request, jsonify
from app.models import Movie, MovieDataset
from app.recommendation import RecommendationSystem
import numpy as np
import json
from app.searching import SearchingSystem
from app.filtering import FilteringSystem
from app.sorting import SortingSystem
from app.media_loading import MediaAndTrailers
from app.evaluation import Evaluator
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, linear_kernel

main = Blueprint('main', __name__)


from app.preprocessing import MoviePreprocessor
from app.recommendation import RecommendationSystem

movie_class = Movie(movie_data={})
movies = MovieDataset()
preprocessor = MoviePreprocessor()
recommendation_system = RecommendationSystem(preprocessor)
recommendation_cache = {}
recommended_paginated_movies = []
searching_system = SearchingSystem(movies)
filtering_system = FilteringSystem()
sorting_system = SortingSystem()
media_and_trailers = MediaAndTrailers()
evaluator = Evaluator()


def sanitize_movie_data(movies_list):
    sanitized_movies = []
    for movie in movies_list:
        sanitized_movie = {}
        for key, value in movie.items():
            if isinstance(value, float) and (value != value):
                print(f"NaN value detected in field {key}, setting it to null")
                sanitized_movie[key] = None
            else:
                sanitized_movie[key] = value
        sanitized_movies.append(sanitized_movie)
    return sanitized_movies

@main.route('/movies/recommend', methods=['POST'])
def recommend():
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
        'total_pages': total_pages
    }), 200, {'Content-Type': 'application/json'}


@main.route('/movies/search', methods=['POST'])
def search():
    data = request.get_json()
    print("Data from search:", data)
    movie_title = data.get('query', '')
    print("Title from search:", movie_title)
    
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

    for movie in sorted_movies:
        print(f"{movie.title} - Runtime: {movie.runtime}, Popularity: {movie.popularity}")

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


@main.route('/movies/evaluate', methods=['POST'])
def evaluate_recommendations():
    data = request.get_json()
    movie_title = data.get('movieTitle', '')
    k = data.get('k', 10)
    
    if movie_title not in recommendation_cache:
        recommendations = recommendation_system.get_recommendations(movie_title, k=100)
        sanitized_recommendations = sanitize_movie_data(recommendations)
        recommendation_cache[movie_title] = sanitized_recommendations
    else:
        sanitized_recommendations = recommendation_cache[movie_title]
    
    evaluation_results = evaluator.evaluate_recommendations(
        sanitized_recommendations[:k],
        k=k
    )
    
    try:
        movie_row = preprocessor.movies_df[
            preprocessor.movies_df['title'].str.lower() == movie_title.lower()
        ]
        if not movie_row.empty:
            rec_indices = []
            for rec in sanitized_recommendations[:k]:
                rec_title = rec.get('title', '')
                rec_row = preprocessor.movies_df[
                    preprocessor.movies_df['title'].str.lower() == rec_title.lower()
                ]
                if not rec_row.empty:
                    rec_indices.append(rec_row.index[0])
            
            if len(rec_indices) > 1:
                diversity_dist = evaluator.diversity_distance(
                    sanitized_recommendations[:k],
                    recommendation_system.features_matrix,
                    rec_indices
                )
                evaluation_results['diversity_distance'] = diversity_dist
    except Exception as e:
        print(f"Error calculating diversity distance: {e}")
    
    return jsonify({
        'movie_title': movie_title,
        'k': k,
        'metrics': evaluation_results
    }), 200


@main.route('/movies/compare-metrics', methods=['POST'])
def compare_metrics():
    data = request.get_json()
    movie_title = data.get('movieTitle', '')
    k = data.get('k', 10)
    
    movie_row = preprocessor.movies_df[
        preprocessor.movies_df['title'].str.lower() == movie_title.lower()
    ]
    
    if movie_row.empty:
        return jsonify({'error': 'Movie not found'}), 404
    
    input_index = movie_row.index[0]
    features_matrix = recommendation_system.features_matrix
    
    results = {}
    
    cosine_sims = cosine_similarity([features_matrix[input_index]], features_matrix)[0]
    cosine_indices = cosine_sims.argsort()[-k-1:-1][::-1]
    cosine_recs = [preprocessor.movies_df.iloc[idx]['title'] for idx in cosine_indices]
    
    results['cosine'] = {
        'recommendations': cosine_recs,
        'avg_similarity': float(np.mean(cosine_sims[cosine_indices])),
        'std_similarity': float(np.std(cosine_sims[cosine_indices]))
    }
    
    euclidean_sims = -euclidean_distances([features_matrix[input_index]], features_matrix)[0]
    euclidean_indices = euclidean_sims.argsort()[-k-1:-1][::-1]
    euclidean_recs = [preprocessor.movies_df.iloc[idx]['title'] for idx in euclidean_indices]
    
    results['euclidean'] = {
        'recommendations': euclidean_recs,
        'avg_similarity': float(np.mean(euclidean_sims[euclidean_indices])),
        'std_similarity': float(np.std(euclidean_sims[euclidean_indices]))
    }
    
    dot_sims = linear_kernel([features_matrix[input_index]], features_matrix)[0]
    dot_indices = dot_sims.argsort()[-k-1:-1][::-1]
    dot_recs = [preprocessor.movies_df.iloc[idx]['title'] for idx in dot_indices]
    
    results['dot_product'] = {
        'recommendations': dot_recs,
        'avg_similarity': float(np.mean(dot_sims[dot_indices])),
        'std_similarity': float(np.std(dot_sims[dot_indices]))
    }
    
    return jsonify({
        'movie_title': movie_title,
        'k': k,
        'comparison': results
    }), 200


@main.route('/movies/statistics', methods=['GET'])
def get_dataset_statistics():
    all_movies = movies.get_movies()
    
    total_movies = len(all_movies)
    
    all_genres = set()
    genre_counts = {}
    for movie in all_movies:
        for genre in movie.genres:
            if isinstance(genre, dict) and 'name' in genre:
                genre_name = genre['name']
                all_genres.add(genre_name)
                genre_counts[genre_name] = genre_counts.get(genre_name, 0) + 1
    
    all_languages = set()
    for movie in all_movies:
        all_languages.add(movie.original_language)
    
    ratings = [movie.vote_average for movie in all_movies if movie.vote_average > 0]
    avg_rating = sum(ratings) / len(ratings) if ratings else 0
    
    popularities = [movie.popularity for movie in all_movies if movie.popularity > 0]
    avg_popularity = sum(popularities) / len(popularities) if popularities else 0
    
    budgets = [movie.budget for movie in all_movies if movie.budget > 0]
    revenues = [movie.revenue for movie in all_movies if movie.revenue > 0]
    
    avg_budget = sum(budgets) / len(budgets) if budgets else 0
    avg_revenue = sum(revenues) / len(revenues) if revenues else 0
    
    top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    return jsonify({
        'total_movies': total_movies,
        'total_genres': len(all_genres),
        'total_languages': len(all_languages),
        'rating_statistics': {
            'average': round(avg_rating, 2),
            'min': round(min(ratings), 2) if ratings else 0,
            'max': round(max(ratings), 2) if ratings else 0,
            'total_rated': len(ratings)
        },
        'popularity_statistics': {
            'average': round(avg_popularity, 2),
            'min': round(min(popularities), 2) if popularities else 0,
            'max': round(max(popularities), 2) if popularities else 0
        },
        'budget_statistics': {
            'average': round(avg_budget, 2),
            'total_with_budget': len(budgets)
        },
        'revenue_statistics': {
            'average': round(avg_revenue, 2),
            'total_with_revenue': len(revenues)
        },
        'top_genres': [{'name': name, 'count': count} for name, count in top_genres],
        'all_genres': sorted(list(all_genres))
    }), 200
