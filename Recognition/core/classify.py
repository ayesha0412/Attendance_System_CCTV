"""
Cosine Similarity Gallery Matching
====================================
Compares a query embedding against all gallery entries using dot product
of L2-normalised vectors (cosine similarity).

No SVM, no neural classifier — just geometry on a unit hypersphere.
This is the correct distance metric for ArcFace/AdaFace embeddings.
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)


def cosine_classify(embedding, gallery, threshold=0.42, debug=False):
    """
    Classify an embedding by cosine similarity against the gallery.

    Args:
        embedding: 512-dim numpy array (query face).
        gallery:   {name: L2-normalised mean embedding} dict.
        threshold: Minimum similarity to accept (below -> "Unknown").
        debug:     If True, log all pairwise scores.

    Returns:
        (name, score) tuple.  name is "Unknown" if below threshold.
    """
    emb = embedding.astype(np.float32)
    emb = emb / (np.linalg.norm(emb) + 1e-8)

    scores = {name: float(np.dot(emb, ref)) for name, ref in gallery.items()}
    best_name  = max(scores, key=scores.get)
    best_score = scores[best_name]

    if debug:
        score_str = "  ".join(f"{n}: {v:.2f}" for n, v in sorted(scores.items()))
        result    = best_name if best_score >= threshold else "Unknown"
        logger.debug("%s  ->  %s", score_str, result)

    if best_score < threshold:
        best_name = "Unknown"

    return best_name, best_score
