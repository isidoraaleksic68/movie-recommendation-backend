import os

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_PATH = os.path.join(BASE_DIR, '../data/tmdb_5000_movies.csv')
