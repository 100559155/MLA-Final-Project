import csv
import os
import requests
import random

API_KEY = 'e7347b7e9a964195c378251b89c4eb22'  # Replace with your API key

def get_movie_details(movie_name):
    # Search for the movie
    search_url = f'https://api.themoviedb.org/3/search/movie?api_key={API_KEY}&query={movie_name}'
    response = requests.get(search_url)
    data = response.json()

    if 'results' in data and data['results']:  # Check if 'results' key exists and is not empty
        movie = data['results'][0]  # Get the first movie result
        movie_id = movie.get('id')  # Use .get() to avoid KeyError
        rating = movie.get('vote_average', 'N/A')  # Get the movie rating

        if not movie_id:
            return None

        # Get movie reviews
        reviews_url = f'https://api.themoviedb.org/3/movie/{movie_id}/reviews?api_key={API_KEY}'
        reviews_response = requests.get(reviews_url)
        reviews = reviews_response.json()

        review_list = [review.get('content', 'No review available') for review in reviews.get('results', [])[:5]]

        return {"Rating": rating, "Reviews": review_list}  # Return a dictionary

    return None

def fetch_movies(page=1):
    url = f'https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&page={page}'
    response = requests.get(url)
    data = response.json()
    return data.get('results', [])  # Use .get() to avoid KeyError

def get_random_movies(n=1000):
    all_movies = []
    page = 1
    while len(all_movies) < n:
        movies = fetch_movies(page)
        if not movies:
            break
        all_movies.extend(movies)
        page += 1
    if len(all_movies) < n:
        n = len(all_movies)  # Prevent random.sample error if not enough movies
    return random.sample(all_movies, n)

# Example movie
def data(movie_name):
    movie_details = get_movie_details(movie_name)
    if movie_details:
        return [{'Movie Name': movie_name, 'Rating': movie_details['Rating'], 'Reviews': "; ".join(movie_details['Reviews'])}]
    return []

# Write to CSV
random_movies = get_random_movies(2000)

a = 1
csv_file = 'movies.csv'

for movie in random_movies:
    print(f"{a} Scraping reviews for: {movie['title']}")
    a += 1
    with open(csv_file, 'a', newline='', encoding='utf-8') as csvFile:
        column_names = ['Movie Name', 'Rating', 'Reviews']
        writer = csv.DictWriter(csvFile, fieldnames=column_names)

        if os.stat(csv_file).st_size == 0:
            writer.writeheader()

        movie_data = data(movie['title'])  # FIXED: Pass movie title string instead of dictionary
        if movie_data:
            writer.writerows(movie_data)
        else:
            print(f"Could not fetch details for {movie['title']}")  # FIXED: Display correct movie title
