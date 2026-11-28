"""Rank-based fusion across independent retrieval queries; scores are not comparable."""
from app.schemas import SearchResponse, SearchHitResponse

def fuse_results(runs: list[SearchResponse], top_k: int, rrf_k: int=60):
    if len(runs)==1:
        return list(runs[0].results[:top_k]), {}
    scores={}
    representatives={}
    for run in runs:
        seen=set()
        for position,hit in enumerate(sorted(run.results,key=lambda h:h.rank),1):
            pid=hit.passage_id
            if pid in seen:
                continue
            seen.add(pid)
            scores[pid]=scores.get(pid,0.0)+1.0/(rrf_k+position)
            previous=representatives.get(pid)
            if previous is None or position<previous[0]:
                representatives[pid]=(position,hit)
    ordered=sorted(scores,key=lambda pid:(-scores[pid],representatives[pid][0],pid))[:top_k]
    hits=[representatives[pid][1].model_copy(update={"rank":i})
          for i,pid in enumerate(ordered,1)]
    return hits,{pid:scores[pid] for pid in ordered}
