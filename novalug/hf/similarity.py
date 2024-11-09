import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from termcolor import cprint


def create_similarity_embeddings():

    # Load the model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load the CSV file into a DataFrame
    df = pd.read_csv("data/hitting.csv")

    # Get the number of rows
    num_rows = df.shape[0]

    # Initialize an empty list to store row embeddings
    row_embeddings = []

    # Encode each row and store the embeddings
    for i in range(num_rows):
        row_embedding = model.encode(
            df.iloc[i].astype(str).tolist(), convert_to_tensor=True
        )  # Ensure the row is converted to a list of strings
        # Compute the mean embedding for the row
        mean_embedding = torch.mean(row_embedding, dim=0)
        row_embeddings.append(mean_embedding)

    # Convert the list of row embeddings into a 2D tensor
    row_embeddings_tensor = torch.stack(row_embeddings)

    # Check the shape of the embeddings
    print(f"Shape of row embeddings: {row_embeddings_tensor.shape}")

    cprint(row_embeddings_tensor.shape, "red")
    # Ensure the embeddings are 2D
    if len(row_embeddings_tensor.shape) != 2:
        raise ValueError("Embeddings must be a 2D array")

    # Compute the cosine similarity matrix
    cosine_sim_matrix = util.cos_sim(row_embeddings_tensor, row_embeddings_tensor)

    # Move the tensor to the CPU before converting to a NumPy array
    cosine_sim_matrix_cpu = cosine_sim_matrix.cpu()

    # Convert the cosine similarity matrix to a DataFrame for better readability
    cosine_sim_df = pd.DataFrame(cosine_sim_matrix_cpu.numpy())

    # Print the cosine similarity DataFrame
    print(cosine_sim_df)
    return cosine_sim_df


# Function to recommend the top 5 most similar rows
def recommend_top_5(row_index, cosine_sim_df):
    # Get the similarity scores for the given row
    similarity_scores = cosine_sim_df.iloc[row_index]

    # Sort the scores in descending order and get the indices of the top 5 most similar rows
    top_5_indices = similarity_scores.sort_values(ascending=False).index[1:6]

    return top_5_indices


# cosine_sim_df = create_similarity_embeddings()
# # Save the cosine similarity DataFrame to a CSV file
# cosine_sim_df.to_csv("data/cosine_similarity_matrix.csv", index=False)

# # Load the cosine similarity DataFrame from a CSV file
cosine_sim_df = pd.read_csv("data/cosine_similarity_matrix.csv")

# Example usage
row_index = 0  # Index of the row for which to find similar rows
top_5_similar_rows = recommend_top_5(row_index, cosine_sim_df)
print(f"Top 5 rows similar to row {row_index}: {top_5_similar_rows}")

row_index = 6
top_5_similar_rows = recommend_top_5(row_index, cosine_sim_df)
print(f"Top 5 rows similar to row {row_index}: {top_5_similar_rows}")


row_index = 10
top_5_similar_rows = recommend_top_5(row_index, cosine_sim_df)
print(f"Top 5 rows similar to row {row_index}: {top_5_similar_rows}")

row_index = 21
top_5_similar_rows = recommend_top_5(row_index, cosine_sim_df)
print(f"Top 5 rows similar to row {row_index}: {top_5_similar_rows}")
