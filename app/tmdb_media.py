import csv
from tmdbv3api import TMDb, Movie
from config import Config

tmdb = TMDb()
tmdb.api_key = '963c3b43716c73c3ecce096f68c176e5'  # TMDb API key
movie_search = Movie()

def fetch_movie_data(movie_id):
    """Fetch movie poster, media, and trailers using the movie ID."""
    try:
        movie_details = movie_search.details(movie_id)

        movie_poster = f"https://image.tmdb.org/t/p/original/{movie_details.poster_path}" if movie_details.poster_path else None
        
        media = [
            f"https://image.tmdb.org/t/p/w780/{backdrop.file_path}" 
            for backdrop in movie_details.backdrops
        ] if hasattr(movie_details, 'backdrops') else []

        trailers = [
            f"https://www.youtube.com/watch?v={video.key}" 
            for video in movie_details.videos.results 
            if video.type == 'Trailer' and video.site == 'YouTube'
        ] if hasattr(movie_details, 'videos') and hasattr(movie_details.videos, 'results') else []

        return movie_poster, media, trailers
    
    except Exception as e:
        print(f"Error fetching data for movie ID {movie_id}: {e}")
        return None, None, None

def create_csv_with_media(input_path, output_path):
    """Create a new CSV file with movie media information."""
    with open(input_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    fieldnames = ['id', 'movie_poster', 'media', 'trailers']

    with open(output_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            movie_id = row['id']
            movie_poster, media, trailers = fetch_movie_data(movie_id)
            row_output = {
                'id': movie_id,
                'movie_poster': movie_poster,
                'media': ', '.join(media) if media else None,
                'trailers': ', '.join(trailers) if trailers else None
            }
            writer.writerow(row_output)

    print("CSV file created successfully.")

if __name__ == "__main__":
    print("Creating CSV with media...")

    data_input_path = Config.DATA_PATH
    data_output_path = Config.DATA_WITH_MEDIA 

    create_csv_with_media(data_input_path, data_output_path)
