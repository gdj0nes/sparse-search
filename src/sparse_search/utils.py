from typing import Dict, List


def hybrid_score_norm(dense: List[float], sparse: Dict[int, float], alpha: float):
    """Hybrid score using a convex combination

        alpha * dense + (1 - alpha) * sparse

    Args:
        dense: Array of floats representing
        sparse: sparse array
        alpha: scale between 0 and 1

    Returns:
        dense:
        sparse:
    """
    # TODO: raise warning if the vector isn't normalized

    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    return [v * alpha for v in dense], {k: v * (1 - alpha) for k, v in sparse.items()}
