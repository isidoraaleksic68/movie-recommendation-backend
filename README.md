# 🎬 Movie Recommendation System - Backend

Welcome to the backend of the **Movie Recommendation System**, a Python-based RESTful API designed to suggest movies based on **cosine similarity**. It analyzes metadata like genre, keywords, and plot summaries to deliver personalized recommendations.

---

## 🛠️ Tech Stack

- **Python 3** – Backend language  
- **Flask** – for building the API  
- **Pandas & NumPy** – Data manipulation and processing  
- **Scikit-learn** – Vectorization and cosine similarity calculation  
- **TMDB 5000 Movie Dataset** – Source of movie metadata  

---

## 📌 Features

- 🔍 Search for similar movies by title
- 🧠 Recommendations based on cosine similarity of metadata
- 🗂️ Filter and sort results by popularity, rating, vote count, etc.
- 🧾 Cleaned and preprocessed movie data for better accuracy
- 🧑‍💻 RESTful endpoints to serve the frontend

---

## 🔄 How It Works

1. Load and preprocess the movie dataset.
2. Combine relevant metadata fields (like genres, keywords, and overview).
3. Convert metadata into **TF-IDF** or **Count Vectors**.
4. Calculate **cosine similarity** between movie vectors.
5. Expose an API endpoint that returns the top N similar movies.

---
## 🚀 Getting Started
1. Clone the repository:
```bash
git clone https://github.com/your-username/movie-recommendation-backend.git
cd movie-recommendation-backend
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the server:
```bash
python run.py
```
4. You can either launch the frontend or use Postman to test the API endpoints.
