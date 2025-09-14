from app.models import MovieDataset

class SortingSystem:
    def __init__(self):
        self.movie_dataset = MovieDataset()

    def sort_movies(self, movies_list, sort_criteria):

        def multi_sort(movie):
            sort_tuple = []
            for criteria in sort_criteria:
                value = getattr(movie, criteria, None)
                sort_tuple.append(value if value is not None else 0)
            return tuple(sort_tuple)

        sorted_movies = sorted(movies_list, key=multi_sort, reverse=True)
        return sorted_movies
