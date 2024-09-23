import pandas as pd
import ast
from app.config import Config

class Movie:
    def __init__(self, movie_data):
        self.budget = self.safe_cast(movie_data.get('budget', 0), int)
        self.genres = self.parse_list_field(movie_data.get('genres', '[]'))
        self.homepage = movie_data.get('homepage', '')
        self.id = movie_data.get('id', '')
        self.keywords = self.parse_list_field(movie_data.get('keywords', '[]'))
        self.original_language = movie_data.get('original_language', '')
        self.original_title = movie_data.get('original_title', '')
        self.overview = movie_data.get('overview', '')
        self.popularity = self.safe_cast(movie_data.get('popularity', 0.0), float)
        self.production_companies = self.parse_list_field(movie_data.get('production_companies', '[]'))
        self.production_countries = self.parse_list_field(movie_data.get('production_countries', '[]'))
        self.release_date = movie_data.get('release_date', '')
        self.revenue = self.safe_cast(movie_data.get('revenue', 0), int)
        self.runtime = self.safe_cast(movie_data.get('runtime', 0), int)
        self.spoken_languages = self.parse_list_field(movie_data.get('spoken_languages', '[]'))
        self.status = movie_data.get('status', '')
        self.tagline = movie_data.get('tagline', '')
        self.title = movie_data.get('title', '')
        self.vote_average = self.safe_cast(movie_data.get('vote_average', 0.0), float)
        self.vote_count = self.safe_cast(movie_data.get('vote_count', 0), int)

    def parse_list_field(self, field_value):
        """Helper method to parse stringified JSON list fields."""
        try:
            return ast.literal_eval(field_value)
        except (ValueError, SyntaxError):
            return []

    def safe_cast(self, val, to_type):
        """Helper method to safely cast values to a given type."""
        try:
            return to_type(val)
        except (ValueError, TypeError):
            return to_type()  # return default value of the type

class MovieDataset:
    def __init__(self):
        self.movies = self.load_data()

    def load_data(self):
        # Load the CSV data
        df = pd.read_csv(Config.DATA_PATH)
        return [Movie(row) for _, row in df.iterrows()]

    def get_movies(self):
        return self.movies
