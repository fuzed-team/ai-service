"""Embedding comparison and similarity calculation."""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def calculate_cosine_similarity(
    embedding_a: list[float], embedding_b: list[float]
) -> tuple[float, float]:
    """
    Calculate cosine similarity between two embeddings.

    Args:
        embedding_a: First 512D embedding vector
        embedding_b: Second 512D embedding vector

    Returns:
        Tuple of (similarity, distance)
        - similarity: 0.0-1.0 (higher = more similar)
        - distance: 0.0-1.0 (lower = more similar)

    Raises:
        ValueError: If embeddings are not 512-dimensional
    """
    if len(embedding_a) != 512 or len(embedding_b) != 512:
        raise ValueError("Embeddings must be 512-dimensional")

    emb_a = np.array(embedding_a)
    emb_b = np.array(embedding_b)

    # Cosine similarity = dot(a, b) / (||a|| * ||b||)
    similarity = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
    distance = 1 - similarity

    logger.debug(f"Cosine similarity: {similarity:.4f}, distance: {distance:.4f}")

    return float(similarity), float(distance)


def calculate_euclidean_distance(
    embedding_a: list[float], embedding_b: list[float]
) -> float:
    """
    Calculate Euclidean distance between two embeddings.

    Args:
        embedding_a: First 512D embedding vector
        embedding_b: Second 512D embedding vector

    Returns:
        Euclidean distance

    Raises:
        ValueError: If embeddings are not 512-dimensional
    """
    if len(embedding_a) != 512 or len(embedding_b) != 512:
        raise ValueError("Embeddings must be 512-dimensional")

    emb_a = np.array(embedding_a)
    emb_b = np.array(embedding_b)

    distance = np.linalg.norm(emb_a - emb_b)

    logger.debug(f"Euclidean distance: {distance:.4f}")

    return float(distance)
