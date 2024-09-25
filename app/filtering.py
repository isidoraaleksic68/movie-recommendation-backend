from app.models import Movie

class FilteringSystem:
    def __init__(self):
        self.movies = []

    def filter_movies(self, genre=None, language=None, recommended_movies=[]):
        # Convert recommended_movies to Movie instances
        self.movies = [Movie(movie_data) for movie_data in recommended_movies]
        
        print(f"Genre filter: {genre}")
        print(f"Language filter: {language}")
        print(f"Total recommended movies: {len(recommended_movies)}")
        
        # Initialize filtered list
        filtered = self.movies
        
        # Filter by genre
        if genre:
            genre_filtered = []
            for movie in filtered:
                for g in movie.genres:
                    if g['name'].lower() == genre.lower():
                        genre_filtered.append(movie)
                        break  # Stop checking other genres for this movie
            filtered = genre_filtered  # Replace the filtered list with genre-filtered list
        
        print(f"Movies after genre filtering: {len(filtered)}")
        
        # Filter by language
        if language:
            language_filtered = []
            for movie in filtered:
                for lang in movie.spoken_languages:
                    if lang['name'].lower() == language.lower():
                        language_filtered.append(movie)
                        break  # Stop checking other languages for this movie
            filtered = language_filtered  # Replace the filtered list with language-filtered list
        
        print(f"Movies after language filtering: {len(filtered)}")
        
        return filtered
