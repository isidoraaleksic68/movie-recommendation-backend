from flask import Blueprint, request, jsonify
from app.models import Movie, MovieDataset
from app.recommendation import RecommendationSystem
import numpy as np
import json
from app.searching import SearchingSystem
from app.filtering import FilteringSystem

main = Blueprint('main', __name__)

movie_class = Movie(movie_data={})
movies = MovieDataset()
recommendation_cache = {}
recommended_paginated_movies = []
recommendation_system = RecommendationSystem()
searching_system = SearchingSystem(movies)
filtering_system = FilteringSystem()

def sanitize_movie_data(movies_list):
    """Sanitize the movie data to ensure no NaN values are present."""
    sanitized_movies = []
    for movie in movies_list:
        sanitized_movie = {}
        for key, value in movie.items():
            if isinstance(value, float) and (value != value):  # Check for NaN
                print(f"NaN value detected in field {key}, setting it to null")
                sanitized_movie[key] = None  # Replace NaN with None
            else:
                sanitized_movie[key] = value
        sanitized_movies.append(sanitized_movie)
    return sanitized_movies

@main.route('/movies/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    movie_title = data.get('movieTitle', '')

    # Cache recommendations for the movie title to avoid recalculating
    if movie_title not in recommendation_cache:
        recommendations = recommendation_system.get_recommendations(movie_title, k=100)  # Fetch up to 100 recommendations
        # recommendations_dict = [movie.to_dict() for movie in recommendations]
        sanitized_recommendations = sanitize_movie_data(recommendations)
        recommendation_cache[movie_title] = sanitized_recommendations  # Store in cache
    else:
        sanitized_recommendations = recommendation_cache[movie_title]

    # Pagination logic
    page = int(request.args.get('page', 1))  # Default to page 1 if not specified
    per_page = 12  # Number of movies per page
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

    # Sanitize search results
    sanitized_search_results = sanitize_movie_data(results_dict)


    # Implement pagination
    page = int(request.args.get('page', 1))  # Default to page 1 if not specified
    per_page = 12  # Number of movies per page
    start = (page - 1) * per_page
    end = start + per_page
    paginated_results = sanitized_search_results[start:end]

    return jsonify({'searching results': paginated_results}), 200, {'Content-Type': 'application/json'}

@main.route('/movies/topRated', methods=['GET'])
def get_top_rated():
    # Get all movies and sort by vote average (descending)
    movies_list = sorted([movie.to_dict() for movie in movies.get_movies()], key=lambda x: x.get('vote_average', 0), reverse=True)

    # Sanitize the movies by removing NaN values
    sanitized_movies_list = sanitize_movie_data(movies_list)

    # Implement pagination
    page = int(request.args.get('page', 1))  # Default to page 1 if not specified
    per_page = 12  # Number of movies per page
    start = (page - 1) * per_page
    end = start + per_page
    paginated_movies = sanitized_movies_list[start:end]

    # Convert to JSON
    return jsonify(paginated_movies), 200, {'Content-Type': 'application/json'}


@main.route('/movies/<int:movie_id>', methods=['GET'])
def get_movie_details(movie_id):
    movie = movies.get_movie_by_id(movie_id)  # Assuming you have a method to get movie by ID

    if movie is None:
        return jsonify({'error': 'Movie not found'}), 404

    # Convert the movie to dictionary and sanitize it
    movie_dict = movie.to_dict()
    sanitized_movie = sanitize_movie_data([movie_dict])[0]  # Sanitize to ensure no NaN values

    return jsonify(sanitized_movie), 200, {'Content-Type': 'application/json'}


@main.route('/movies/filter', methods=['POST'])
def filter_movies():
    data = request.get_json()
    genre = data.get('genre')
    language = data.get('language')
    movie_title = data.get('movie_title')

    # Fetch recommendations based on the provided movie title
    sanitized_recommendations = recommendation_cache.get(movie_title, [])

    # Filter the movies
    filtering_system = FilteringSystem()
    filtered_movies = filtering_system.filter_movies(genre, language, sanitized_recommendations)

    # Convert filtered Movie objects to dictionaries
    filtered_movies_dicts = [movie.to_dict() for movie in filtered_movies]


    # Pagination logic
    page = int(request.args.get('page', 1))  # Default to page 1 if not specified
    per_page = 12  # Number of movies per page
    start = (page - 1) * per_page
    end = start + per_page
    paginated_filtered_movies = filtered_movies_dicts[start:end]

    return jsonify({'filtered_movies': paginated_filtered_movies}), 200, {'Content-Type': 'application/json'}


@main.route('/movies/filtering/metadata', methods=['GET'])
def get_metadata():
    genres = set()
    languages = set()

    for movie in movies.get_movies():
        # Collect unique genres
        for genre in movie.genres:
            genre_name = genre.get('name', '').strip().lower()
            if genre_name:  # Ensure genre name is not empty
                genres.add(genre_name)

        print("Collected genres: ", genres)
        
        # Collect unique spoken languages, filtering out invalid ones
        for spoken_lang in movie.spoken_languages:
            language_name = spoken_lang.get('name', '').strip().lower()
            if language_name and language_name.isalpha():  # Ensure it's a valid language
                languages.add(language_name)

        print("Collected languages: ", languages)

    # Return the sorted lists of unique genres and languages
    return jsonify({
        'genres': sorted(genres),
        'spoken_languages': sorted(languages)
    }), 200, {'Content-Type': 'application/json'}
