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

from step2_retrieval.core import retrieve
from step2_retrieval.graph_rag import graph_rescore
from step2_retrieval.hyde import generate as hyde_generate

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


def run_case(query: str, targets: list[str]) -> dict:
    """네 파이프라인 각각에 대해 과목 랭킹(중복 제거된 전체 순서)을 구하고, 목표 과목들의 순위를 뽑는다."""
    hyde_text = hyde_generate(query)
    rankings = {
        "vanilla": hits_to_courses(retrieve(query, POOL), "score"),
        "vanilla+hyde": hits_to_courses(retrieve(hyde_text, POOL), "score"),
        "graph-rag": hits_to_courses(graph_rescore(query), "graph_score"),
        "graph-rag+hyde": hits_to_courses(graph_rescore(query, embed_text=hyde_text), "graph_score"),
    }
    ranks = {t: {p: (rankings[p].index(t) + 1 if t in rankings[p] else None) for p in PIPELINES}
             for t in targets}
    return {"query": query, "hyde": hyde_text, "targets": targets,
            "top10": {p: rankings[p][:10] for p in PIPELINES}, "ranks": ranks}


def print_case(result: dict):
    print(f"\n질의: {result['query']}")
    print("HyDE 가상 문서(앞 200자):", result["hyde"][:200].replace("\n", " "))
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


def main():
    all_results = [run_case(c["query"], c["targets"]) for c in CASES]
    for result in all_results:
        print_case(result)

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(all_results, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n상세 결과 저장: {RESULT_PATH}")


if __name__ == "__main__":
    main()
