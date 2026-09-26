"""'관심분야 -> 추천과목' 검색 품질 평가: vanilla vs graph-rag, 그리고 각각 HyDE 유무.

4개 조합을 비교한다: vanilla, vanilla+hyde, graph-rag, graph-rag+hyde. HyDE(step2_retrieval/
hyde.py)는 "질문을 그대로 임베딩" 대신 "질문에 맞춰 LLM이 지어낸 가상의 강의계획서를 임베딩"해서
검색한다 — 질문 문장과 실제 강의계획서 문체 사이의 격차를 줄여보려는 실험. no-rag는 검색을
안 하므로 이 비교와 무관해서 뺐다.

GT(정답)는 extract_course_tree.py가 만든 step3_eval/data/course_tree_gt.json을 쓴다.
연구실 하나당 질의 하나를 만들어 step3_eval/data/queries.json에 사람이 읽을 수 있게 저장한다:
    질의 = "{주요 연구 분야}에 관심이 있는데 어떤 과목을 들으면 좋을까?"
    정답 = 그 연구실의 추천과목 전체(합집합, 정식 과목명 집합)

각 파이프라인의 retrieve()/graph_rescore()가 돌려주는 "청크 랭킹"을, source 파일명에서
과목명을 뽑아 "과목 랭킹"으로 바꾼다(같은 과목의 여러 청크는 제일 높은 순위 것만 남기고
중복 제거). 그 과목 랭킹을 정답 집합과 비교해 Precision/Recall/MRR/nDCG(top-K 고정)와
MAP(전체 랭킹, K 무관)을 계산한다. 계산식은 metrics()/average_precision() 그대로가 정의다.

Precision@K·Recall@K는 질의마다 정답 개수가 다르면(이 데이터셋은 3~33개로 편차가 큼)
질의 간 비교가 공정하지 않다 — 정답 5개인 질의는 Precision@10이 구조적으로 0.5를 못
넘고, 정답 30개인 질의는 Recall@10이 구조적으로 0.33을 못 넘는다. nDCG@K는 IDCG를
min(|정답|,K)로 정규화해서 이미 이 문제에서 자유롭고, MAP은 랭킹 전체를 보고 정답을
찾을 때마다의 precision을 평균 내므로 K 자체가 필요 없다 — 그래서 이 두 지표가 질의
간 공정 비교에 더 적합하다. Precision/Recall@K는 참고용으로 남겨둔다.

실행:  python -m step3_eval.run_eval [K] [START:END]
       K 기본값 10(Precision/Recall/nDCG@K 계산용, MAP은 K와 무관). START:END(1부터, 양끝 포함,
       예: 30:50)를 주면 질의 전체가 아니라 그 구간만 빠르게 돌려보고, 결과는 latest.json이
       아니라 results/partial_START-END.json에 따로 저장한다(전체 실행 결과를 안 덮어씀).
출력:  step3_eval/data/queries.json          사람이 볼 수 있는 질의+정답 목록 (채점과 별개로 항상 갱신)
       step3_eval/data/hyde_cache.json       질의별로 생성한 HyDE 가상 문서 (중단 후 재개 가능)
       step3_eval/data/results/latest.json   질의별 상세 결과 + 콘솔에 파이프라인별 macro-average 비교표
"""
import json
import math
import sys
from pathlib import Path

from step2_retrieval.core import retrieve
from step2_retrieval.graph_rag import graph_rescore
from step2_retrieval.hyde import generate as hyde_generate

from .course_names import SOURCE_RE, strip_paren

DATA_DIR = Path(__file__).parent / "data"
GT_PATH = DATA_DIR / "course_tree_gt.json"
QUERIES_PATH = DATA_DIR / "queries.json"
HYDE_CACHE_PATH = DATA_DIR / "hyde_cache.json"
RESULT_PATH = DATA_DIR / "results" / "latest.json"
POOL = 60

PIPELINES = ["vanilla", "vanilla+hyde", "graph-rag", "graph-rag+hyde"]


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


def hits_to_courses(hits: list[dict], score_key: str) -> list[str]:
    """자르지 않은 전체 과목 랭킹(중복 과목 제거, 점수 내림차순). metrics()가 K로 자르고,
    average_precision()은 이 전체를 그대로 쓴다 — 그래야 같은 한 번의 검색 결과로 두 종류의
    지표를 다 낼 수 있다(K별로 다시 검색할 필요 없음)."""
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
    return out


def load_hyde_cache() -> dict:
    return json.loads(HYDE_CACHE_PATH.read_text(encoding="utf-8")) if HYDE_CACHE_PATH.exists() else {}


def save_hyde_cache(cache: dict):
    HYDE_CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def run_pipeline(name: str, query: str, hyde_text: str | None = None) -> list[str]:
    if name == "vanilla":
        return hits_to_courses(retrieve(query, POOL), "score")
    if name == "vanilla+hyde":
        return hits_to_courses(retrieve(hyde_text, POOL), "score")
    if name == "graph-rag":
        return hits_to_courses(graph_rescore(query), "graph_score")
    if name == "graph-rag+hyde":
        return hits_to_courses(graph_rescore(query, embed_text=hyde_text), "graph_score")
    raise ValueError(name)


def metrics(retrieved: list[str], relevant: set[str], k: int) -> dict:
    """표준 이진-관련성 IR 지표(top-K 고정). retrieved/relevant는 과목명 문자열."""
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


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """AP: 랭킹 전체(자르지 않음)를 보고, 정답을 찾을 때마다 그 순간까지의 precision을 기록해
    평균낸다. K도 유사도 임계값도 필요 없이 질의마다 정답 개수에 자동으로 맞춰지므로(정답 5개
    질의와 30개 질의를 같은 잣대로 비교 가능), Precision/Recall@K가 안고 있는 "질의별 정답
    개수 편차" 문제에서 자유롭다. 찾은 정답 수가 아니라 전체 정답 개수(len(relevant))로
    나누는 게 표준 정의 — 끝까지 못 찾은 정답은 자동으로 0점 취급된다."""
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, c in enumerate(retrieved, 1):
        if c in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def main():
    """실행:  python -m step3_eval.run_eval [K] [START:END]
    START:END는 1부터 시작하는 인덱스 범위(양끝 포함, 예: 30:50) — 전체를 다 안 돌리고 일부만
    빠르게 확인하고 싶을 때 쓴다. 주면 결과를 results/latest.json이 아니라
    results/partial_START-END.json에 따로 저장해서, 전체 실행 결과(latest.json)를 안 덮어쓴다."""
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    queries = build_queries()
    result_path = RESULT_PATH
    if len(sys.argv) > 2:
        start, end = (int(x) for x in sys.argv[2].split(":"))
        queries = queries[start - 1:end]
        result_path = RESULT_PATH.parent / f"partial_{start}-{end}.json"
        print(f"질의 범위 제한: {start}~{end}번째만 실행 ({len(queries)}개)")
    print(f"평가 질의 {len(queries)}개 (연구실별 관심분야 -> 추천과목), k={k}")
    print(f"질의+정답 목록: {QUERIES_PATH}")

    needs_hyde = any(p.endswith("+hyde") for p in PIPELINES)
    hyde_cache = load_hyde_cache() if needs_hyde else {}
    if needs_hyde:
        print(f"HyDE 캐시 {len(hyde_cache)}개 재사용 ({HYDE_CACHE_PATH})")
    print()

    detail = []
    agg = {p: [] for p in PIPELINES}

    for i, q in enumerate(queries, 1):
        hyde_text = None
        if needs_hyde:
            if q["id"] not in hyde_cache:
                hyde_cache[q["id"]] = hyde_generate(q["query"])
                if i % 10 == 0:
                    save_hyde_cache(hyde_cache)
            hyde_text = hyde_cache[q["id"]]

        row = {"id": q["id"], "professor": q["professor"], "lab": q["lab"],
               "query": q["query"], "relevant": q["relevant"], "hyde": hyde_text, "pipelines": {}}
        relevant_set = set(q["relevant"])
        for p in PIPELINES:
            retrieved = run_pipeline(p, q["query"], hyde_text)  # 자르지 않은 전체 랭킹
            m = metrics(retrieved, relevant_set, k)
            m["ap"] = average_precision(retrieved, relevant_set)
            row["pipelines"][p] = {"retrieved": retrieved[:k], **m}
            agg[p].append(m)
        detail.append(row)
        print(f"[{i}/{len(queries)}] {q['professor']} ({q['lab']}) - "
              + " / ".join(f"{p}:P{agg[p][-1]['precision']:.2f} AP{agg[p][-1]['ap']:.2f}" for p in PIPELINES),
              flush=True)

    if needs_hyde:
        save_hyde_cache(hyde_cache)

    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps({"k": k, "detail": detail}, ensure_ascii=False, indent=1),
                            encoding="utf-8")

    print(f"\n=== macro-average (질의 {len(queries)}개, k={k}) ===")
    print("nDCG@k, MAP은 질의별 정답 개수 차이에 자동 정규화됨 — 가장 신뢰할 수 있는 비교 지표.")
    header = f"{'pipeline':<16}{'Precision@k':>13}{'Recall@k':>11}{'MRR':>9}{'nDCG@k':>9}{'MAP':>9}"
    print(header)
    print("-" * len(header))
    for p in PIPELINES:
        rows = agg[p]
        n = len(rows)
        avg = {m: sum(r[m] for r in rows) / n for m in ("precision", "recall", "mrr", "ndcg", "ap")}
        print(f"{p:<16}{avg['precision']:>13.4f}{avg['recall']:>11.4f}{avg['mrr']:>9.4f}"
              f"{avg['ndcg']:>9.4f}{avg['ap']:>9.4f}")

    print(f"\n상세 결과 저장: {result_path}")


if __name__ == "__main__":
    main()
