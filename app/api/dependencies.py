"""API dependencies for dependency injection."""

from app.workers.ml_worker import MLWorkerPool, ml_worker_pool


async def get_ml_worker() -> MLWorkerPool:
    """
    Get ML worker pool instance for dependency injection.

    Returns:
        Global ML worker pool instance
    """
    return ml_worker_pool
