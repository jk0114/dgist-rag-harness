"""특정 관심분야 질의 몇 개를 직접 골라서, "이 질문엔 이 과목도 나와야 하지 않을까" 하는 목표
과목들이 각 파이프라인(vanilla/vanilla+hyde/graph-rag/graph-rag+hyde)의 전체 랭킹 중 몇 위에
있는지 확인한다.

run_eval.py는 GT(연구실 추천 과목) 전체를 자동으로 채점하지만, 이 스크립트는 그 반대 방향이다 —
GT에 없는 임의의 관심분야를 질문으로 던지고, 사람이 직접 고른 "나왔으면 하는 과목 목록"과
비교한다. RESULTS.md의 "케이스 스터디" 절이 이 스크립트로 얻은 결과다.

CASES 리스트에 {"query": ..., "targets": [...]}를 추가하면 케이스가 늘어난다.

실행: python -m step3_eval.case_study
"""
import json
from pathlib import Path

from step2_retrieval.core import collection, retrieve
from step2_retrieval.graph_rag import get_graph, graph_rescore, _norm_kw
from step2_retrieval.hyde import generate as hyde_generate

from .course_names import course_name_from_source
from .run_eval import hits_to_courses

RESULT_PATH = Path(__file__).parent / "data" / "results" / "case_study.json"
POOL = 100  # 과목 100개까지 훑어서 "top-10엔 없지만 어딘가에는 있는지"까지 본다 (run_eval.py의 POOL=60보다 넉넉하게)

PIPELINES = ["vanilla", "vanilla+hyde", "graph-rag", "graph-rag+hyde"]

CASES = [
    {
        "query": "LLM에 관심이 있는데 어떤 과목을 들으면 좋을까?",
        "targets": ["선형대수학", "확률과 통계 및 실습", "기계학습개론",
                    "컴퓨터 비전 개론", "딥러닝개론", "강화학습"],
    },
    {
        "query": "로보틱스에 관심이 있는데 어떤 과목을 들으면 좋을까?",
        "targets": ["동역학", "자동제어시스템", "지능형제어시스템",
                    "로봇동역학 및 제어", "선형대수학", "컴퓨터 비전 개론"],
    },
    {
        "query": "양자컴퓨팅에 관심이 있는데 어떤 과목을 들으면 좋을까?",
        "targets": ["양자역학Ⅰ", "양자역학Ⅱ", "양자 컴퓨팅 개론", "선형대수학"],
    },
    {
        "query": "AI로 단백질 디자인을 하는데 관심이 있는데 어떤 과목을 들으면 좋을까?",
        "targets": ["기계학습개론", "강화학습", "컴퓨터 비전 개론", "선형대수학",
            "확률과 통계 및 실습", "생화학Ⅰ", "생화학Ⅱ", "세포생물학", "면역학", "공학수학Ⅱ"],
        },
]


def neighborhood_of(hits: list[dict], course_name: str, graph) -> dict | None:
    """hits(graph_rescore 결과) 안에서 course_name에 속한 청크 중 점수가 가장 높은 것을 찾아,
    그 청크가 키워드 그래프로 어떤 다른 과목들과 연결돼 있는지 정리한다. 즉 "이 과목이 왜
    graph-rag 후보로 딸려왔는지"를 키워드 단위로 보여준다. course_name이 hits 안에 없으면 None."""
    candidates = [h for h in hits if course_name_from_source(h.get("source", "")) == course_name]
    if not candidates:
        return None
    score_key = "graph_score" if "graph_score" in candidates[0] else "score"
    chunk = max(candidates, key=lambda h: h[score_key])
    cid = chunk["id"]
    linked_courses: dict[str, list[str]] = {}
    for kw in graph.keywords.get(cid, []):
        neighbor_ids = graph.kw_index.get(_norm_kw(kw), set()) - {cid}
        if not neighbor_ids:
            continue
        meta = collection().get(ids=list(neighbor_ids), include=["metadatas"])
        courses = {course_name_from_source(m.get("source", "")) for m in meta["metadatas"]}
        courses = {c for c in courses if c and c != course_name}
        if courses:
            linked_courses[kw] = sorted(courses)[:6]
    return {"chunk_id": cid, "keywords": graph.keywords.get(cid, []), "linked_courses": linked_courses}


def run_case(query: str, targets: list[str]) -> dict:
    """네 파이프라인 각각에 대해 과목 랭킹(중복 제거된 전체 순서)을 구하고, 목표 과목들의 순위와
    (graph-rag+hyde 기준) 키워드 그래프 이웃 관계까지 뽑는다."""
    hyde_text = hyde_generate(query)
    raw_hits = {
        "vanilla": retrieve(query, POOL),
        "vanilla+hyde": retrieve(hyde_text, POOL),
        "graph-rag": graph_rescore(query),
        "graph-rag+hyde": graph_rescore(query, embed_text=hyde_text),
    }
    score_key = {"vanilla": "score", "vanilla+hyde": "score",
                 "graph-rag": "graph_score", "graph-rag+hyde": "graph_score"}
    rankings = {p: hits_to_courses(raw_hits[p], score_key[p]) for p in PIPELINES}
    ranks = {t: {p: (rankings[p].index(t) + 1 if t in rankings[p] else None) for p in PIPELINES}
             for t in targets}

    graph = get_graph()
    neighborhoods = {}
    for t in targets:
        info = neighborhood_of(raw_hits["graph-rag+hyde"], t, graph)
        if info:
            neighborhoods[t] = info

    return {"query": query, "hyde": hyde_text, "targets": targets,
            "top10": {p: rankings[p][:10] for p in PIPELINES}, "ranks": ranks,
            "neighborhoods": neighborhoods}


def print_case(result: dict):
    print(f"\n질의: {result['query']}")
    print("HyDE 가상 문서(전체):")
    print(result["hyde"])
    print("\ntop-10:")
    for p in PIPELINES:
        print(f"  {p}: {result['top10'][p]}")
    print(f"\n목표 과목 순위 (전체 풀 {POOL}개 중, 없으면 '미발견'):")
    header = f"{'과목':<22}" + "".join(f"{p:>16}" for p in PIPELINES)
    print(header)
    for t in result["targets"]:
        row = f"{t:<22}"
        for p in PIPELINES:
            r = result["ranks"][t][p]
            row += f"{(str(r) if r else '미발견'):>16}"
        print(row)
    print("\ngraph-rag+hyde 기준, 목표 과목이 키워드로 엮인 다른 과목들:")
    for t in result["targets"]:
        info = result["neighborhoods"].get(t)
        if not info:
            print(f"  {t}: (후보 풀에 없어서 이웃을 볼 수 없음)")
            continue
        if not info["linked_courses"]:
            print(f"  {t}: 키워드 {info['keywords']} - 겹치는 다른 과목 없음")
            continue
        print(f"  {t} (키워드 {info['keywords']}):")
        for kw, courses in info["linked_courses"].items():
            print(f"    - '{kw}' 공유: {courses}")


def main():
    all_results = [run_case(c["query"], c["targets"]) for c in CASES]
    for result in all_results:
        print_case(result)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(all_results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n상세 결과 저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
