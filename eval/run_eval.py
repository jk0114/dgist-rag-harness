"""'관심분야 -> 추천과목' 검색 품질 평가: vanilla vs graph-rag.

GT(정답)는 extract_course_tree.py가 만든 eval/data/course_tree_gt.json을 쓴다.
연구실 하나당 질의 하나를 만들어 eval/data/queries.json에 사람이 읽을 수 있게 저장한다:
    질의 = "{주요 연구 분야}에 관심이 있는데 어떤 과목을 들으면 좋을까?"
    정답 = 그 연구실의 추천과목 전체(합집합, 정식 과목명 집합)

각 파이프라인의 retrieve()/graph_rescore()가 돌려주는 "청크 랭킹"을, source 파일명에서
과목명을 뽑아 "과목 랭킹"으로 바꾼다(같은 과목의 여러 청크는 제일 높은 순위 것만 남기고
중복 제거). 그 top-K 과목 랭킹을 정답 집합과 비교해 Precision/Recall/MRR/nDCG를 계산한다.
계산식은 metrics() 함수 그대로가 정의다 — 표준 IR 지표(이진 관련성, k 잘라서 평가).

실행:  python -m eval.run_eval [K]   (K 기본값 10)
출력:  eval/data/queries.json          사람이 볼 수 있는 질의+정답 목록 (채점과 별개로 항상 갱신)
       eval/data/results/latest.json   질의별 상세 결과 + 콘솔에 파이프라인별 macro-average 비교표
"""
import json
import math
import sys
from pathlib import Path

from retrieval.core import retrieve
from retrieval.graph_rag import graph_rescore

from eval.course_names import SOURCE_RE, strip_paren

DATA_DIR = Path(__file__).parent / "data"
GT_PATH = DATA_DIR / "course_tree_gt.json"
QUERIES_PATH = DATA_DIR / "queries.json"
RESULT_PATH = DATA_DIR / "results" / "latest.json"
POOL = 60

PIPELINES = ["vanilla", "graph-rag"]


def build_queries() -> list[dict]:
    """GT에서 질의+정답 목록을 만들어 QUERIES_PATH에 저장하고 돌려준다."""
    raw = json.loads(GT_PATH.read_text(encoding="utf-8"))
    queries = []
    for key, entry in raw.items():
        if not entry or not entry.get("courses"):
            continue
        fields = entry.get("research_fields") or []
        if not fields:
            continue
        relevant = sorted(set(entry["courses"]))
        if not relevant:
            continue
        queries.append({
            "id": key,
            "professor": entry.get("professor", ""),
            "lab": entry.get("lab_ko", ""),
            "query": f"{', '.join(fields)}에 관심이 있는데 어떤 과목을 들으면 좋을까?",
            "relevant": relevant,
        })
    QUERIES_PATH.parent.mkdir(parents=True, exist_ok=True)
    QUERIES_PATH.write_text(json.dumps(queries, ensure_ascii=False, indent=1), encoding="utf-8")
    return queries


def hits_to_courses(hits: list[dict], score_key: str, k: int) -> list[str]:
    ordered = sorted(hits, key=lambda h: h[score_key], reverse=True)
    seen, out = set(), []
    for h in ordered:
        m = SOURCE_RE.match(h.get("source", "") or "")
        if not m:
            continue
        name = strip_paren(m.group("name"))
        if name in seen:
            continue
        seen.add(name)
        out.append(name)
        if len(out) >= k:
            break
    return out


def run_pipeline(name: str, query: str, k: int) -> list[str]:
    if name == "vanilla":
        return hits_to_courses(retrieve(query, POOL), "score", k)
    if name == "graph-rag":
        return hits_to_courses(graph_rescore(query), "graph_score", k)
    raise ValueError(name)


def metrics(retrieved: list[str], relevant: set[str], k: int) -> dict:
    """표준 이진-관련성 IR 지표. retrieved/relevant는 과목명 문자열(랭킹은 retrieved의 순서)."""
    topk = retrieved[:k]
    hit_flags = [1 if c in relevant else 0 for c in topk]
    n_hit = sum(hit_flags)
    precision = n_hit / len(topk) if topk else 0.0
    recall = n_hit / len(relevant) if relevant else 0.0
    rr = 0.0
    for i, f in enumerate(hit_flags, 1):
        if f:
            rr = 1.0 / i
            break
    dcg = sum(f / math.log2(i + 1) for i, f in enumerate(hit_flags, 1))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0
    return {"precision": precision, "recall": recall, "mrr": rr, "ndcg": ndcg, "n_hit": n_hit}


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    queries = build_queries()
    print(f"평가 질의 {len(queries)}개 (연구실별 관심분야 -> 추천과목), k={k}")
    print(f"질의+정답 목록: {QUERIES_PATH}\n")

    detail = []
    agg = {p: [] for p in PIPELINES}

    for i, q in enumerate(queries, 1):
        row = {"id": q["id"], "professor": q["professor"], "lab": q["lab"],
               "query": q["query"], "relevant": q["relevant"], "pipelines": {}}
        for p in PIPELINES:
            retrieved = run_pipeline(p, q["query"], k)
            m = metrics(retrieved, set(q["relevant"]), k)
            row["pipelines"][p] = {"retrieved": retrieved, **m}
            agg[p].append(m)
        detail.append(row)
        print(f"[{i}/{len(queries)}] {q['professor']} ({q['lab']}) - "
              + " / ".join(f"{p}:P{agg[p][-1]['precision']:.2f}" for p in PIPELINES), flush=True)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps({"k": k, "detail": detail}, ensure_ascii=False, indent=1),
                            encoding="utf-8")

    print(f"\n=== macro-average (질의 {len(queries)}개, k={k}) ===")
    header = f"{'pipeline':<14}{'Precision@k':>13}{'Recall@k':>11}{'MRR':>9}{'nDCG@k':>9}"
    print(header)
    print("-" * len(header))
    for p in PIPELINES:
        rows = agg[p]
        n = len(rows)
        avg = {m: sum(r[m] for r in rows) / n for m in ("precision", "recall", "mrr", "ndcg")}
        print(f"{p:<14}{avg['precision']:>13.4f}{avg['recall']:>11.4f}{avg['mrr']:>9.4f}{avg['ndcg']:>9.4f}")

    print(f"\n상세 결과 저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
