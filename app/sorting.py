from app.models import MovieDataset

class SortingSystem:
    def __init__(self):
        self.movie_dataset = MovieDataset()  # Initialize MovieDataset

    def sort_movies(self, movies_list, sort_criteria):

        # Define a custom sorting function
        def multi_sort(movie):
            sort_tuple = []
            for criteria in sort_criteria:
                # Use getattr directly on the movie object
                value = getattr(movie, criteria, None)  # Get the attribute value
                sort_tuple.append(value if value is not None else 0)  # Default to 0 if None
            return tuple(sort_tuple)  # Return as a tuple for sorting

        # Sort movies based on multiple criteria (descending order)
        sorted_movies = sorted(movies_list, key=multi_sort, reverse=True)
        return sorted_movies
