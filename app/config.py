import os

class Config:
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    DATA_PATH = os.path.join(BASE_DIR, '../data/tmdb_5000_movies.csv')
    DATA_WITH_MEDIA = os.path.join(BASE_DIR , '../data/movies_with_media.csv' )
    USE_TRAIN_SPLIT = True
    TRAIN_SIZE = 0.6
    TEST_SIZE = 0.2
    VAL_SIZE = 0.2
    RANDOM_STATE = 42
