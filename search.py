import torch
from embedding_model import generate_embeddings
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

def perform_search(query_embedding, course_data, top_n=5):
    """
    Perform a search for the most relevant courses.
    :param query_embedding: Embedding of the user's search query
    :param course_data: DataFrame with course information
    :param top_n: Number of top results to return
    :return: DataFrame of top N results
    """
    course_data['embeddings'] = course_data['Title'].apply(lambda x: generate_embeddings(x))

    similarities = []
    for embedding in course_data['embeddings']:
        similarity = cosine_similarity(query_embedding.unsqueeze(0), embedding.unsqueeze(0))
        similarities.append(similarity.item())

    course_data['similarity'] = similarities

    return course_data.sort_values(by='similarity', ascending=False).head(top_n)