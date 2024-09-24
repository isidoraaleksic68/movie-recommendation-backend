from flask import Blueprint, request, jsonify
from app.models import MovieDataset
from app.recommendation import RecommendationSystem
import numpy as np
import json
from app.searching import SearchingSystem

main = Blueprint('main', __name__)

movies = MovieDataset()
recommendation_system = RecommendationSystem()
searching_system = SearchingSystem(movies)

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

@main.route('/recommend', methods=['POST'])
def recommend():    
    data = request.get_json()
    print("DATA:", data)
    movie_title = data.get('query', '')
    print("TITLE:", movie_title)
    
    recommendations = recommendation_system.get_recommendations(movie_title)
    recommendations_dict = [movie.to_dict() for movie in recommendations]

    # Sanitize recommendations
    sanitized_recommendations = sanitize_movie_data(recommendations_dict)

    # If recommendations return an ndarray, convert to list
    if isinstance(recommendations, np.ndarray):
        recommendations = recommendations.tolist()
    
    # Get top-rated recommendations
    top_rated_recommendations = sorted(sanitized_recommendations, key=lambda x: x.get('vote_average', 0), reverse=True)

    return jsonify({'recommendations': top_rated_recommendations})

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

    # Get top-rated search results
    top_rated_search_results = sorted(sanitized_search_results, key=lambda x: x.get('vote_average', 0), reverse=True)

    return jsonify({'searching results': top_rated_search_results})

@main.route('/movies/topRated', methods=['GET'])
def get_all_movies_top_rated():
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
