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
    if isinstance(recommendations, np.ndarray):
        recommendations = recommendations.tolist()

    return jsonify({'recommendations': recommendations})


@main.route('/movies', methods=['GET'])
def get_all_movies():
    # Return all movies (could be limited/paginated)
    return jsonify([movie.__dict__ for movie in movies.get_movies()])

@main.route('/movies/<title>', methods=['GET'])
def get_movie_details(title):
    movie = next((movie for movie in movies.get_movies() if movie.title == title), None)
    if movie:
        return jsonify(movie.__dict__)
    return jsonify({'error': 'Movie not found'}), 404

# You can add sorting, filtering routes similarly
