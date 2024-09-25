import pandas as pd
import json  # Use json for parsing JSON strings
from app.config import Config
import math

class Movie:
    def __init__(self, movie_data):
        self.budget = self.safe_cast(movie_data.get('budget', 0), int)
        self.genres = self.parse_list_field(movie_data.get('genres', ''))  # Updated to parse correctly
        self.homepage = self.sanitize_string(movie_data.get('homepage', ''), 'homepage')
        self.id = movie_data.get('id', '')
        self.keywords = self.parse_list_field(movie_data.get('keywords', '[]'))
        self.original_language = movie_data.get('original_language', '')
        self.original_title = movie_data.get('original_title', '')
        self.overview = self.get_overview(movie_data)  # Call the method to get overview
        self.popularity = self.safe_cast(movie_data.get('popularity', 0.0), float)
        self.production_companies = self.parse_list_field(movie_data.get('production_companies', '[]'))
        self.production_countries = self.parse_list_field(movie_data.get('production_countries', '[]'))
        self.release_date = movie_data.get('release_date', '')
        self.revenue = self.safe_cast(movie_data.get('revenue', 0), int)
        self.runtime = self.safe_cast(movie_data.get('runtime', 0), int)
        self.spoken_languages = self.parse_list_field(movie_data.get('spoken_languages', '[]'))
        self.status = movie_data.get('status', '')
        self.tagline = self.sanitize_string(movie_data.get('tagline', ''), 'tagline')
        self.title = movie_data.get('title', '')
        self.vote_average = self.safe_cast(movie_data.get('vote_average', 0.0), float)
        self.vote_count = self.safe_cast(movie_data.get('vote_count', 0), int)

    def get_overview(self, movie_data):
        """Get the overview from movie_data."""
        overview = movie_data.get('overview', '')
        if overview is None or (isinstance(overview, str) and overview.strip() == ''):
            print("Sanitized None or empty value in overview")
            return ""
        return overview

    def to_dict(self):
        print("Genres:", self.genres)  # Debugging output
        return {
            'budget': self.safe_nan(self.budget),
            'genres': [{'id': genre['id'], 'name': genre['name']} for genre in self.genres if isinstance(genre, dict)],  # Access dict elements
            'homepage': self.homepage or '',  # Ensure it is an empty string if None
            'id': self.id,
            'keywords': self.keywords,
            'original_language': self.original_language,
            'original_title': self.original_title,
            'overview': self.overview,
            'popularity': self.safe_nan(self.popularity),
            'production_companies': self.production_companies,
            'production_countries': self.production_countries,
            'release_date': self.release_date,
            'revenue': self.safe_nan(self.revenue),
            'runtime': self.safe_nan(self.runtime),
            'spoken_languages': self.spoken_languages,
            'status': self.status,
            'tagline': self.tagline or '',  # Ensure it is an empty string if None
            'title': self.title,
            'vote_average': self.safe_nan(self.vote_average),
            'vote_count': self.safe_nan(self.vote_count)
        }



    def safe_nan(self, value):
        """Convert NaN values to None to avoid JSON serialization errors."""
        if isinstance(value, float) and math.isnan(value):
            return None  # JSON compatible representation
        return value

    def sanitize_string(self, value, field_name):
        """Sanitize string fields to avoid None values."""
        if value is None or (isinstance(value, str) and value.strip() == ''):
            print(f"Sanitized None or empty value in {field_name}")
            return ""
        return value

    def parse_list_field(self, field_value):
        if isinstance(field_value, str) and field_value.strip() != '':
            try:
                return json.loads(field_value)  # Use json to load the data
            except (ValueError, json.JSONDecodeError):
                print(f"Error decoding JSON for value: {field_value}")  # Debugging output

            # If JSON decoding fails, split by spaces
            return [genre.strip() for genre in field_value.split() if genre.strip()]
        return []  # Return an empty list if the input is not valid

    def safe_cast(self, val, to_type):
        """Safely cast value to the specified type, returning default if it fails."""
        try:
            return to_type(val)
        except (ValueError, TypeError):
            return to_type()  # return default value of the type


class MovieDataset:
    def __init__(self):
        self.movies = self.load_data()

    def load_data(self):
        """Load movie data from CSV file."""
        df = pd.read_csv(Config.DATA_PATH)
        return [Movie(row) for _, row in df.iterrows()]

    def get_movies(self):
        """Return all movies."""
        return self.movies

    # def get_similar_movies_objects(self, similar_movies):
    #     return [Movie(pair) for pair in similar_movies]

    def get_movie_by_id(self, movie_id):
        """Return a Movie object by its ID."""
        for movie in self.movies:
            if movie.id == movie_id:  # Accessing id attribute instead of using dictionary-like access
                return movie
        return None

    def __iter__(self):
        """Make MovieDataset iterable."""
        return iter(self.movies)  # This makes MovieDataset iterable
