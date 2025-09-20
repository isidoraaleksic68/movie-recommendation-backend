from app.models import Movie

class FilteringSystem:
    def __init__(self):
        self.movies = []

    def filter_movies(self, genre=None, language=None, recommended_movies=[]):
        self.movies = [Movie(movie_data) for movie_data in recommended_movies]
        filtered = self.movies

        if genre:
            genre_lower = genre.lower()
            genre_filtered = []
            for movie in filtered:
                if not movie.genres:
                    continue
                for g in movie.genres:
                    name = g['name'] if isinstance(g, dict) and 'name' in g else g
                    if name and genre_lower in name.lower():
                        genre_filtered.append(movie)
                        break
            filtered = genre_filtered

        if language:
            language_lower = language.lower()
            language_filtered = []
            for movie in filtered:
                if not movie.spoken_languages:
                    continue
                for lang in movie.spoken_languages:
                    name = lang['name'] if isinstance(lang, dict) and 'name' in lang else lang
                    if name and language_lower in name.lower():
                        language_filtered.append(movie)
                        break
            filtered = language_filtered

        return filtered
