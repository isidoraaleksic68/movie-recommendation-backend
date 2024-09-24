import pandas as pd
import ast
from app.config import Config
import math

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

    def to_dict(self):
        return {
            'budget': self.budget,
            'genres': self.genres,
            'homepage': self.homepage,
            'id': self.id,
            'keywords': self.keywords,
            'original_language': self.original_language,
            'original_title': self.original_title,
            'overview': self.overview,
            'popularity': self.popularity,
            'production_companies': self.production_companies,
            'production_countries': self.production_countries,
            'release_date': self.release_date,
            'revenue': self.revenue,
            'runtime': self.runtime,
            'spoken_languages': self.spoken_languages,
            'status': self.status,
            'tagline': self.tagline,
            'title': self.title,
            'vote_average': self.vote_average,
            'vote_count': self.vote_count
        }

    def to_dict1(self):
        # Sanitizing and logging each field
        movie_dict = {
            'budget': self.sanitize_number(self.budget, 'budget'),
            'genres': self.sanitize_list(self.genres, 'genres'),
            'homepage': self.sanitize_string(self.homepage, 'homepage'),
            'id': self.sanitize_string(self.id, 'id'),
            'keywords': self.sanitize_list(self.keywords, 'keywords'),
            'original_language': self.sanitize_string(self.original_language, 'original_language'),
            'original_title': self.sanitize_string(self.original_title, 'original_title'),
            'overview': self.sanitize_string(self.overview, 'overview'),
            'popularity': self.sanitize_number(self.popularity, 'popularity'),
            'production_companies': self.sanitize_list(self.production_companies, 'production_companies'),
            'production_countries': self.sanitize_list(self.production_countries, 'production_countries'),
            'release_date': self.sanitize_string(self.release_date, 'release_date'),
            'revenue': self.sanitize_number(self.revenue, 'revenue'),
            'runtime': self.sanitize_number(self.runtime, 'runtime'),
            'spoken_languages': self.sanitize_list(self.spoken_languages, 'spoken_languages'),
            'status': self.sanitize_string(self.status, 'status'),
            'tagline': self.sanitize_string(self.tagline, 'tagline'),
            'title': self.sanitize_string(self.title, 'title'),
            'vote_average': self.sanitize_number(self.vote_average, 'vote_average'),
            'vote_count': self.sanitize_number(self.vote_count, 'vote_count')
        }
        print(f"Sanitized movie: {movie_dict}")
        return movie_dict

    def sanitize_number(self, value, field_name):
        """Sanitize number fields to ensure no NaN values."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            print(f"Sanitized NaN value in {field_name}")
            return 0
        return value

    def sanitize_string(self, value, field_name):
        """Sanitize string fields to avoid None values."""
        if value is None:
            print(f"Sanitized None value in {field_name}")
            return ""
        return value

    def sanitize_list(self, value, field_name):
        """Sanitize list fields to ensure they are not None."""
        if value is None:
            print(f"Sanitized None value in {field_name}")
            return []
        return value

    def parse_list_field(self, field_value):
        try:
            return ast.literal_eval(field_value)
        except (ValueError, SyntaxError):
            return []

    def safe_cast(self, val, to_type):
        try:
            return to_type(val)
        except (ValueError, TypeError):
            return to_type()  # return default value of the type
        

class MovieDataset:
    def __init__(self):
        self.movies = self.load_data()

    def load_data(self):
        df = pd.read_csv(Config.DATA_PATH)
        return [Movie(row) for _, row in df.iterrows()]

    def get_movies(self):
        return self.movies
    
    def get_similar_movies_objects(self, similar_movies):
        movies = []
        for pair in similar_movies:
            movie = Movie(pair)
            movies.append(movie)
        return movies

        
