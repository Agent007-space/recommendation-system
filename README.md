# recommendation-system
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# Sample user ratings data
data = {
    'User': ['A', 'A', 'A', 'B', 'B', 'C', 'C', 'D'],
    'Movie': ['Avengers', 'Titanic', 'Notebook', 'Avengers', 'Titanic', 'Notebook', 'Avengers', 'Titanic'],
    'Rating': [5, 3, 4, 5, 2, 4, 5, 1]
}

df = pd.DataFrame(data)

# Create a user-item matrix
ratings_matrix = df.pivot_table(index='User', columns='Movie', values='Rating').fillna(0)

# Compute cosine similarity between users
user_similarity = cosine_similarity(ratings_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=ratings_matrix.index, columns=ratings_matrix.index)

# Function to recommend movies for a given user
def recommend_movies(target_user, top_n=2):
    if target_user not in ratings_matrix.index:
        print(f"User {target_user} not found.")
        return

    # Get scores of other users with the target user
    sim_scores = user_similarity_df[target_user]
    similar_users = sim_scores.drop(target_user).sort_values(ascending=False)

    weighted_scores = pd.Series(dtype=float)

    for other_user, similarity in similar_users.items():
        other_user_ratings = ratings_matrix.loc[other_user]
        weighted_scores = weighted_scores.add(other_user_ratings * similarity, fill_value=0)

    # Normalize by similarity sum
    sim_sums = similar_users.sum()
    if sim_sums > 0:
        weighted_scores /= sim_sums

    # Remove already rated movies
    rated_by_user = ratings_matrix.loc[target_user][ratings_matrix.loc[target_user] > 0].index
    recommendations = weighted_scores.drop(rated_by_user).sort_values(ascending=False).head(top_n)

    print(f"\nTop {top_n} recommendations for User '{target_user}':")
    for movie, score in recommendations.items():
        print(f"{movie}: predicted rating {score:.2f}")

# Example usage
recommend_movies('C')
