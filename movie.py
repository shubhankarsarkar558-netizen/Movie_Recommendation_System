import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Dataset
df = pd.read_csv("data.csv")

# Fill Missing Values
df['overview'] = df['overview'].fillna('')

# Create Feature Column
df['features'] = df['overview']

# Convert Text to Vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(df['features']).toarray()

# Calculate Similarity
similarity = cosine_similarity(vectors)

# Recommendation Function
def recommend(movie):
    movie = movie.lower()

    if movie not in df['title'].str.lower().values:
        print("Movie not found!")
        return

    index = df[df['title'].str.lower() == movie].index[0]

    distances = list(enumerate(similarity[index]))
    movies_list = sorted(distances, reverse=True, key=lambda x: x[1])[1:6]

    print("\nRecommended Movies:\n")

    for i in movies_list:
        print(df.iloc[i[0]].title)

# User Input
movie_name = input("Enter Movie Name: ")

recommend(movie_name)