from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

app = Flask(__name__)

# Load KDrama data
kdrama_data = pd.read_csv('kdramalist.csv')

# Fill NaN values with empty strings in 'Genre' and 'Sinopsis'
kdrama_data['Genre'] = kdrama_data['Genre'].fillna('')
kdrama_data['Sinopsis'] = kdrama_data['Sinopsis'].fillna('')

# TF-IDF vectorization for 'Genre'
genre_vectorizer = TfidfVectorizer(stop_words='english')
genre_matrix = genre_vectorizer.fit_transform(kdrama_data['Genre'])

# TF-IDF vectorization for 'Sinopsis'
sinopsis_vectorizer = TfidfVectorizer(stop_words='english')
sinopsis_matrix = sinopsis_vectorizer.fit_transform(kdrama_data['Sinopsis'])

# Combine TF-IDF matrices
combined_matrix = np.concatenate([genre_matrix.toarray(), sinopsis_matrix.toarray()], axis=1)

# Compute cosine similarity matrix
cosine_sim = cosine_similarity(combined_matrix, combined_matrix)

# Function to get recommendations based on Genre, Sinopsis, and descending order of Year
def get_recommendations(title, cosine_sim=cosine_sim):
    kdrama_titles_lower = [t.lower() for t in kdrama_data['Name'].tolist()]
    title_lower = title.lower()

    # Split the input title into words
    title_words = title_lower.split()

    if not title_words:
        return ["KDrama not found. Please enter a valid title."]

    # Define the minimum number of words to match based on the number of words in the title
    if len(title_words) == 1:
        min_words_to_match = 1
    elif len(title_words) == 2:
        min_words_to_match = 1
    else:
        min_words_to_match = 2

    # Check if at least the specified number of words match
    matching_titles = [t for t in kdrama_titles_lower if sum(word in t for word in title_words) >= min_words_to_match]

    if not matching_titles:
        return ["KDrama not found. Please enter a valid title."]

    # Use the first matching title for simplicity (you can customize this logic)
    matching_title = matching_titles[0]

    idx = kdrama_titles_lower.index(matching_title)
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get top 10 recommendations (excluding the input KDrama)
    sim_scores = sim_scores[1:16]
    kdrama_indices = [i[0] for i in sim_scores]

    # Sort recommendations by Year in descending order
    recommendations = kdrama_data.iloc[kdrama_indices].sort_values(by='Year', ascending=False)['Name'].tolist()

    return recommendations

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        kdrama_title = request.form['kdrama_title']
        recommendations = get_recommendations(kdrama_title)
        return render_template('index.html', kdrama_title=kdrama_title, recommendations=recommendations, kdrama_data=kdrama_data)
    return render_template('index.html', kdrama_title=None, recommendations=None, kdrama_data=kdrama_data)

if __name__ == '__main__':
    app.run(debug=True)
