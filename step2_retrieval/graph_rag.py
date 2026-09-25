"""Graph-RAG: GNN-Ret(arXiv 2406.06572) 논문 3.1절을 그대로 따라 만든 "청크들의 그래프(GoPs)"
기반 검색. (이 프로젝트의 세 번째, 마지막 파이프라인 — no-rag / vanilla / graph-rag 중 하나.)

edge는 논문과 동일하게 두 종류:
  1) 구조적 인접(structure-related) — 같은 문서(PDF) 안에서 순서상 바로 옆에 있는 청크끼리 연결
  2) 키워드 공유(keyword-related)  — build_keyword_cache.py가 LLM으로 뽑아둔 키워드가 겹치는 청크끼리 연결
     (먼저 `python -m step2_retrieval.build_keyword_cache` 를 한 번 돌려서 chroma/keyword_cache.json을 만들어둬야 함)

점수 전파는 논문이 "학습 없이 쓸 때" 쓰는 기본값을 그대로 쓴다: 1-layer, α=0.5,
이웃 중 최댓값(=최소 거리)을 메시지로 사용(논문 Table 2에서 평균보다 항상 더 좋았던 방식).

그래프 이웃이 처음 dense 검색 후보(top-K) 안에 없어도 직접 임베딩을 다시 찾아와 점수를
매긴다(이웃이 후보 풀 밖이라 그냥 버려지는 문제를 피하려고).

사용 범위: build_keyword_cache.py가 강의계획서 청크만 처리해뒀으므로, 키워드 edge는 강의계획서
사이에서만 생긴다. 구조적 인접 edge는 추가 비용이 없어서 매뉴얼 포함 전체 청크에 대해 만든다.
"""
import json
import re
from collections import defaultdict

import config as C
from .core import chat, collection, embedder, retrieve, sanitize
from .pipelines import register, SYSTEM_RAG, RULE_TAIL, _ctx

ALPHA = 0.5
POOL_K = 30

KEYWORD_CACHE_PATH = C.DB_DIR / "keyword_cache.json"

_ID_RE = re.compile(r"^(?P<stem>.+)__p(?P<page>\d+)__(?P<si>\d+)$")


def _parse_id(chunk_id: str):
    m = _ID_RE.match(chunk_id)
    if not m:
        return None
    return m.group("stem"), int(m.group("page")), int(m.group("si"))


def _norm_kw(k: str) -> str:
    return re.sub(r"\s+", "", k.strip().lower())


class Graph:
    """전체 코퍼스에 대해 한 번만 만들어서 메모리에 들고 있는다 (앱 재시작 전까지 재사용)."""

    def __init__(self):
        res = collection().get(include=["metadatas"])
        ids, metas = res["ids"], res["metadatas"]

        # ---- ① 구조적 인접: 같은 source 안에서 (page, split-index) 순으로 정렬해 사슬로 연결 ----
        by_source: dict[str, list[str]] = defaultdict(list)
        for cid in ids:
            parsed = _parse_id(cid)
            if parsed is None:
                continue
            stem, page, si = parsed
            by_source[stem].append((page, si, cid))
        self.struct_neighbors: dict[str, set[str]] = defaultdict(set)
        for stem, items in by_source.items():
            items.sort()
            ordered = [cid for _, _, cid in items]
            for i, cid in enumerate(ordered):
                if i > 0:
                    self.struct_neighbors[cid].add(ordered[i - 1])
                if i < len(ordered) - 1:
                    self.struct_neighbors[cid].add(ordered[i + 1])

        # ---- ② 키워드 공유: build_keyword_cache.py 결과로 역인덱스 구성 ----
        self.keywords: dict[str, list[str]] = {}
        if KEYWORD_CACHE_PATH.exists():
            self.keywords = json.loads(KEYWORD_CACHE_PATH.read_text(encoding="utf-8"))
        kw_index: dict[str, set[str]] = defaultdict(set)
        for cid, kws in self.keywords.items():
            for kw in kws:
                kw_index[_norm_kw(kw)].add(cid)
        self.kw_index = kw_index

        print(f"[graph-rag] 그래프 구성 완료: 청크 {len(ids)}개, "
              f"키워드 태깅된 청크 {len(self.keywords)}개")

    def neighbors(self, chunk_id: str) -> set[str]:
        neigh = set(self.struct_neighbors.get(chunk_id, set()))
        for kw in self.keywords.get(chunk_id, []):
            neigh |= self.kw_index.get(_norm_kw(kw), set())
        neigh.discard(chunk_id)
        return neigh


_graph: Graph | None = None


def get_graph() -> Graph:
    global _graph
    if _graph is None:
        _graph = Graph()
    return _graph


def _fetch_by_ids(ids: list[str], query_vec: list[float]) -> dict[str, dict]:
    """풀 밖에 있던 그래프 이웃들을 id로 직접 가져와, 쿼리와의 코사인 유사도를 계산한다
    (bge-m3 임베딩은 normalize_embeddings=True로 저장돼 있어 내적 = 코사인 유사도)."""
    if not ids:
        return {}
    res = collection().get(ids=ids, include=["documents", "metadatas", "embeddings"])
    out = {}
    for cid, doc, meta, emb in zip(res["ids"], res["documents"], res["metadatas"], res["embeddings"]):
        sim = sum(a * b for a, b in zip(query_vec, emb))
        out[cid] = {"id": cid, "text": doc, "score": round(sim, 4), **meta}
    return out


def graph_rescore(query: str, embed_text: str | None = None) -> list[dict]:
    """dense 후보(POOL_K) + 그 이웃(풀 밖이면 직접 점수 매겨서) 전부에 대해 graph_score를 계산해,
    자르지 않고 전부 돌려준다 (실험실 페이지에서 '풀에는 있지만 최종 top-k 밖'인 것들의
    graph_score도 보려고 분리해둠). 실제 검색은 graph_retrieve()를 쓴다.

    embed_text: 주어지면 query 대신 이 텍스트를 임베딩해서 검색한다(HyDE — retrieval.hyde.generate()가
    만든 가상 문서 등). query 자체는 결과에 영향 없음(현재는 로깅 용도로도 안 쓰임)."""
    graph = get_graph()
    text = embed_text if embed_text is not None else query

    pool = retrieve(text, POOL_K)
    pool_by_id = {h["id"]: h for h in pool}

    # 풀에 없는 그래프 이웃들을 모아서 직접 점수를 매겨 풀에 합친다 (이웃이 top-K 밖이라 버려지는 것 방지)
    frontier_ids = set()
    for h in pool:
        frontier_ids |= graph.neighbors(h["id"])
    frontier_ids -= set(pool_by_id)
    if frontier_ids:
        q_vec = embedder("cpu").encode([sanitize(text)], normalize_embeddings=True).tolist()[0]
        pool_by_id.update(_fetch_by_ids(list(frontier_ids), q_vec))

    all_hits = list(pool_by_id.values())
    for h in all_hits:
        neighbor_ids = graph.neighbors(h["id"]) & pool_by_id.keys()
        neighbor_scores = [pool_by_id[nid]["score"] for nid in neighbor_ids]
        boost = max(neighbor_scores) if neighbor_scores else h["score"]
        h["graph_score"] = ALPHA * h["score"] + (1 - ALPHA) * boost

    all_hits.sort(key=lambda h: h["graph_score"], reverse=True)
    return all_hits


def graph_retrieve(query: str, k: int) -> list[dict]:
    return graph_rescore(query)[:k]


@register("graph-rag", "GNN-Ret 논문 방식: 같은 문서 인접 청크 + LLM이 뽑은 키워드 공유 청크끼리 "
                        "유사도를 전파(학습 없음, α=0.5, 이웃 최댓값). 풀 밖 이웃도 직접 점수를 매겨 포함. "
                        "프롬프트는 vanilla와 동일 — 검색 방식만 비교하기 위함.")
def graph_rag(question, history, params):
    k = int(params.get("top_k", C.TOP_K))
    hits = graph_retrieve(question, k)
    msgs = [{"role": "system", "content": SYSTEM_RAG}, *history,
            {"role": "user", "content": f"[근거]\n{_ctx(hits)}\n\n[질문]\n{question}{RULE_TAIL}"}]
    return {"answer": chat(msgs), "retrieved": hits, "prompt": msgs}
