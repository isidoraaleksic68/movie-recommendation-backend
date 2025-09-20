from flask import Blueprint, request, jsonify
from app.models import Movie, MovieDataset
from app.recommendation import RecommendationSystem
import numpy as np
import json
from app.searching import SearchingSystem
from app.filtering import FilteringSystem
from app.sorting import SortingSystem
from app.media_loading import MediaAndTrailers

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

    return jsonify({'recommendations': paginated_recommendations}), 200, {'Content-Type': 'application/json'}


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

    return jsonify({'searching results': paginated_results}), 200, {'Content-Type': 'application/json'}

@main.route('/movies/topRated', methods=['GET'])
def get_top_rated():
    movies_list = sorted([movie.to_dict() for movie in movies.get_movies()], key=lambda x: x.get('vote_average', 0), reverse=True)
    sanitized_movies_list = sanitize_movie_data(movies_list)
    page = int(request.args.get('page', 1))
    per_page = 12
    start = (page - 1) * per_page
    end = start + per_page
    paginated_movies = sanitized_movies_list[start:end]

    return jsonify(paginated_movies), 200, {'Content-Type': 'application/json'}


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

    return jsonify({'filtered_movies': paginated_filtered_movies}), 200, {'Content-Type': 'application/json'}


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
   
    return jsonify({'sorted_movies': paginated_sorted_movies}), 200


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

