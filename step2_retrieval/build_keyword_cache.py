"""강의계획서 청크에서 핵심 키워드(entity)를 로컬 LLM으로 뽑아 캐시에 저장한다.
GNN-Ret 논문(3.1절 "keyword-related")의 그래프 edge 구성 방식 그대로 — LLM에 프롬프트해서
청크별 키워드를 뽑고, 그 키워드를 공유하는 청크끼리 step2_retrieval/graph_rag.py에서 그래프로 연결한다.

실행:  python -m step2_retrieval.build_keyword_cache
출력:  chroma/keyword_cache.json   {chunk_id: [키워드, ...]}
이미 처리된 chunk_id는 재실행 시 건너뛴다 (중단 후 재개 가능).
"""
import json
import re

import chromadb

import config as C
from .core import chat

CACHE_PATH = C.DB_DIR / "keyword_cache.json"

PROMPT = """다음은 대학 강의계획서의 일부 내용이다. 이 글에서 과목의 핵심 주제를 나타내는
키워드(전공 용어·기술/개념 이름·도구/프로그래밍언어 이름 등)를 최대 6개 뽑아라.
사람 이름, 학교 행정 용어(학점·이수구분·강의실·요일/교시 등)는 제외한다.
다른 말 없이 JSON 배열로만 답하라. 예: ["머신러닝", "신경망", "파이썬"]

텍스트:
{text}"""


def extract(text: str) -> list[str]:
    try:
        raw = chat([{"role": "user", "content": PROMPT.format(text=text[:1500])}])
    except Exception as e:
        print(f"    LLM 호출 실패: {e}")
        return []
    m = re.search(r"\[.*\]", raw, re.S)
    if not m:
        return []
    try:
        kws = json.loads(m.group(0))
        return [str(k).strip() for k in kws if str(k).strip()][:6]
    except Exception:
        return []


def load_cache() -> dict:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict):
    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    client = chromadb.PersistentClient(str(C.DB_DIR))
    col = client.get_collection(C.COLLECTION)
    res = col.get(include=["documents", "metadatas"])
    targets = [(i, d) for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
               if "CMN17" in m.get("source", "")]
    print(f"대상 청크(강의계획서): {len(targets)}개")

    cache = load_cache()
    if cache:
        print(f"기존 캐시 {len(cache)}개 재사용")

    done = 0
    for i, (cid, text) in enumerate(targets, 1):
        if cid in cache:
            continue
        cache[cid] = extract(text)
        done += 1
        if done % 20 == 0:
            print(f"  {i}/{len(targets)}  {cid} -> {cache[cid]}")
            save_cache(cache)

    save_cache(cache)
    n_empty = sum(1 for v in cache.values() if not v)
    print(f"\n완료: {CACHE_PATH} (청크 {len(cache)}개, 키워드 못 뽑은 청크 {n_empty}개)")


if __name__ == "__main__":
    main()
