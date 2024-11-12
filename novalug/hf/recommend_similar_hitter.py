import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from termcolor import cprint


def save_cosine_similarity_matrix(cosine_sim_df, cosine_similarity_matrix_filename):
    # # Save the cosine similarity DataFrame to a CSV file
    cosine_sim_df.to_csv(cosine_similarity_matrix_filename, index=False)


def load_cosine_similarity_matrix(cosine_similarity_matrix_filename):
    # # Load the cosine similarity DataFrame from a CSV file
    cosine_sim_df = pd.read_csv(cosine_similarity_matrix_filename)
    return cosine_sim_df


def create_similarity_embeddings(hitting_data_filename, hitting_stats_df):

    # Load the model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Get the number of rows
    num_rows = hitting_stats_df.shape[0]

    # Initialize an empty list to store row embeddings
    row_embeddings = []

    # Encode each row and store the embeddings
    for i in range(num_rows):
        row_embedding = model.encode(
            hitting_stats_df.iloc[i].astype(str).tolist(), convert_to_tensor=True
        )  # Ensure the row is converted to a list of strings
        # Compute the mean embedding for the row
        mean_embedding = torch.mean(row_embedding, dim=0)
        row_embeddings.append(mean_embedding)

    # Convert the list of row embeddings into a 2D tensor
    row_embeddings_tensor = torch.stack(row_embeddings)

    # Check the shape of the embeddings
    # print(f"Shape of row embeddings: {row_embeddings_tensor.shape}")

    # cprint(row_embeddings_tensor.shape, "red")
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
    # print(cosine_sim_df)
    return cosine_sim_df


# Function to recommend the top 5 most similar rows
def recommend_top_n(row_index, cosine_sim_df, n):
    # Get the similarity scores for the given row
    similarity_scores = cosine_sim_df.iloc[row_index]

    # Sort the scores in descending order and get the indices of the top n most similar rows
    top_n_indices = similarity_scores.sort_values(ascending=False).index[1 : n + 1]

    return top_n_indices


def find_similar_hitters(
    cosine_sim_df, hitting_stats_df, player_name, number_of_players
):
    # Find the row index of the player in the dataframe
    # Find the row index of the player 'Ryan' in the 'BATTER' column
    row_indices = hitting_stats_df.index[hitting_stats_df["BATTER"] == "Ryan"].tolist()
    row_index = row_indices[0]

    # print(f"Row index of player {player_name}: {row_index}")
    top_n_similar_rows = recommend_top_n(row_index, cosine_sim_df, number_of_players)
    cprint(
        f"Top {number_of_players} rows similar to row {row_index}: {top_n_similar_rows}",
        "yellow",
    )

    # return the most similar hitter
    # convert all the rows in top_n_similar_rows to ints
    return [int(i) for i in top_n_similar_rows]

    # return int(top_n_similar_rows[0])


def main():
    hitting_data_filename = "data/stats/hitting.csv"
    cosine_similarity_matrix_filename = "data/similarity/cosine_similarity_matrix.csv"

    # Load the CSV file into a DataFrame
    hitting_stats_df = pd.read_csv(hitting_data_filename)
    # really we would want to remove the starters here...

    # cosine_sim_df = create_similarity_embeddings(hitting_data_filename, hitting_stats_df)
    # save_cosine_similarity_matrix(cosine_sim_df, cosine_similarity_matrix_filename)

    cosine_sim_df = load_cosine_similarity_matrix(cosine_similarity_matrix_filename)
    player_name = "Ryan"
    number_of_players = 5
    similar_player_indexes = find_similar_hitters(
        cosine_sim_df, hitting_stats_df, player_name, number_of_players
    )

    for match_idx, similar_player_idx in enumerate(similar_player_indexes):
        similar_player = hitting_stats_df.iloc[similar_player_idx]
        similar_player_name = similar_player["BATTER"]
        cprint(
            f"Similar hitter #{match_idx+1} to {player_name}: {similar_player_name}",
            "yellow",
        )
        # to see the full details of the similar player, uncomment the line below
        # cprint(similar_player, 'green')
        # print("")

    print("")
    similar_player = hitting_stats_df.iloc[similar_player_indexes[0]]
    similar_player_name = similar_player["BATTER"]
    cprint(f"The most similar hitter to {player_name} is {similar_player_name}", "blue")
