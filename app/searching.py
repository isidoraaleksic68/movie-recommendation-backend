from typing import List
from app.models import Movie, MovieDataset

class SearchingSystem:
    
    def __init__(self, movie_dataset: MovieDataset):
        self.movies = movie_dataset.get_movies()

    def get_results(self, query: str) -> List[Movie]:
        query = query.lower()
        search_results = [movie for movie in self.movies if query in movie.title.lower()]
        return search_results
