"""
Reciprocal Rank Fusion (RRF) for combining multiple ranked result lists.

RRF score formula: sum(1 / (k + rank)) for each list where the document appears.
Does NOT add raw similarity scores — rank position only.
"""
from app.schemas.common import RetrievalResult

def reciprocal_rank_fusion(
    results_lists: list[list[RetrievalResult]],
    k: int = 60,
    top_k: int = 5
) -> list[RetrievalResult]:
    """
    Combines multiple lists of RetrievalResult using Reciprocal Rank Fusion.
    """
    rrf_scores = {}
    result_objects = {}

    for results in results_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk_id
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0.0
                result_objects[chunk_id] = result
            
            rrf_scores[chunk_id] += 1.0 / (k + rank)

    sorted_chunk_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    top_chunk_ids = sorted_chunk_ids[:top_k]

    final_results = []
    for chunk_id in top_chunk_ids:
        original_res = result_objects[chunk_id]
        new_res = RetrievalResult(
            chunk_id=original_res.chunk_id,
            text=original_res.text,
            score=rrf_scores[chunk_id],
            source=original_res.source,
            page=original_res.page,
            document_id=original_res.document_id
        )
        final_results.append(new_res)

    return final_results
