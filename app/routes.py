from flask import Blueprint, request, jsonify
from app.models import MovieDataset
from app.recommendation import RecommendationSystem
import numpy as np

main = Blueprint('main', __name__)

movies = MovieDataset()
recommendation_system = RecommendationSystem()

@main.route('/recommend', methods=['POST'])
def recommend():    
    data = request.get_json()
    print("DAATA:" , data)
    movie_title = data.get('query', '')
    print("TITLE:" + movie_title)
    
    recommendations = recommendation_system.get_recommendations(movie_title)
    recommendations_dict = [movie.to_dict() for movie in recommendations]

    if isinstance(recommendations, np.ndarray):
        recommendations = recommendations.tolist()
    return jsonify({'recommendations': recommendations_dict})

import json

import json
from flask import request

@main.route('/movies/topRated', methods=['GET'])
def get_all_movies_top_rated():
    # Get all movies and sort by vote average (descending)
    movies_list = sorted([movie.to_dict() for movie in movies.get_movies()], key=lambda x: x.get('vote_average', 0), reverse=True)

    # Sanitize the movies by removing NaN values
    for movie in movies_list:
        for key, value in movie.items():
            if value != value:  # Check for NaN
                print(f"NaN value detected in field {key}, setting it to null")
                movie[key] = None

    # Implement pagination
    page = int(request.args.get('page'))
    per_page = 12  # Number of movies per page
    start = (page - 1) * per_page
    end = start + per_page
    paginated_movies = movies_list[start:end]

    # Convert to JSON
    return json.dumps(paginated_movies), 200, {'Content-Type': 'application/json'}


@main.route('/movies/<title>', methods=['GET'])
def get_movie_details(title):
    movie = next((movie for movie in movies.get_movies() if movie.title == title), None)
    if movie:
        return jsonify(movie.__dict__)
    return jsonify({'error': 'Movie not found'}), 404

# You can add sorting, filtering routes similarly
