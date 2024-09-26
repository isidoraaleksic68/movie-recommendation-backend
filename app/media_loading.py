import csv
from app.config import Config

class MediaAndTrailers:
    def __init__(self):
        self.media_and_trailers_db = Config.DATA_WITH_MEDIA

    def fetch_movie_poster(self, movie_id):
        """Fetch the movie poster link from the CSV file based on movie ID."""
        try:
            with open(self.media_and_trailers_db, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if int(row['id']) == movie_id:  # Compare with integer
                        return row['movie_poster']
        except Exception as e:
            print(f"Error reading CSV: {e}")
        return None

    def fetch_trailers(self, movie_id):
        """Fetch trailer links from the CSV file based on movie ID."""
        try:
            with open(self.media_and_trailers_db, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if int(row['id']) == movie_id:  # Compare with integer
                        trailers = row['trailers']
                        return trailers.split(', ') if trailers else []
        except Exception as e:
            print(f"Error reading CSV: {e}")
        return []
