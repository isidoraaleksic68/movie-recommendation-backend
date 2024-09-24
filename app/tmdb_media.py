import csv
import tmdbv3api
from config import Config

tmdb = tmdbv3api.TMDb()
tmdb.api_key = '963c3b43716c73c3ecce096f68c176e5'  # adding my tmdb api key
movie_search = tmdbv3api.Movie()

def fetch_movie_data(movie_title):
    search_results = movie_search.search(movie_title)
    if search_results:
        movie = search_results[0]
        movie_details = movie_search.details(movie.id) 

        #fetching movie poster
        movie_poster = f"https://image.tmdb.org/t/p/original/{movie.poster_path}"
        
        #fethching media
        media = []
        if hasattr(movie_details, 'backdrops'):
            for backdrop in movie_details.backdrops:
                media.append(f"https://image.tmdb.org/t/p/w780/{backdrop.file_path}")

        #fetching trailers
        trailer = None
        if hasattr(movie_details, 'videos') and hasattr(movie_details.videos, 'results'):
            for video in movie_details.videos.results:
                if video.type == 'Trailer' and video.site == 'YouTube':
                    trailer = f"https://www.youtube.com/watch?v={video.key}"
                    break

        return movie_poster, media, trailer
    return None, None, None

if __name__ == "__main__":
    print("Creating CSV with media...")

    data_input_path = Config.DATA_PATH  #path to csv with no media links
    data_output_path = Config.DATA_WITH_MEDIA  #path to new csv with media

    with open(data_input_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    with open(data_output_path, 'w', newline='', encoding='utf-8') as file:
        fieldnames = reader.fieldnames + ['movie_poster', 'media', 'trailer']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            movie_title = row['title']
            movie_poster, media, trailer = fetch_movie_data(movie_title)
            row['movie_poster'] = movie_poster
            row['media'] = ', '.join(media) if media else None
            row['trailer'] = trailer
            writer.writerow(row)

    print("CSV file created successfully.")
