"""HyDE (Hypothetical Document Embeddings, Gao et al. 2023 ACL, "Precise Zero-Shot Dense
Retrieval without Relevance Labels").

질문을 그대로 임베딩해서 검색하는 대신, LLM한테 "이 질문에 맞는 가상의 문서를 실제 문서와
같은 형식으로 써봐"라고 시킨 뒤 그 결과물을 임베딩해서 검색한다. 질문("~에 관심 있는데 어떤
과목?")과 실제 문서(강의계획서 특유의 문체·구조)는 표현이 많이 달라서 임베딩 공간에서도
멀리 떨어져 있을 수 있는데, "질문에 그럴듯하게 답하는 가짜 문서"는 실제 문서와 표현이 훨씬
비슷해서 검색이 더 잘 맞을 수 있다는 게 논문의 핵심 주장. 가짜 문서의 사실 여부(그런 과목이
실제로 있는지)는 중요하지 않다 — "강의계획서답게 생긴 텍스트"이기만 하면 된다.

few-shot 예시는 실제 corpus(강의계획서 청크)에서 그대로 가져와, LLM이 DGIST 강의계획서의
실제 필드 구조(과목명/학기/담당교수/이수구분/학점/교과개요/교과목표/주차별 강의계획 등)를
따라 쓰게 만든다 — 손으로 예시를 짜 넣지 않고 실제 corpus 형식을 그대로 재사용한다.

vanilla/graph-rag 어느 쪽에도 끼울 수 있다: generate()가 만든 텍스트를 retrieve()나
graph_rescore()의 embed_text로 넘기면 된다(둘 다 "질문 대신 이 텍스트를 임베딩해서 검색"을
지원하도록 되어 있다). no-rag는 애초에 검색을 안 하므로 HyDE와 무관하다.
"""
import random

from .core import chat, collection

_FEWSHOT_CACHE: list[str] | None = None


def _fewshot_examples(n: int = 2) -> list[str]:
    """corpus에서 강의계획서 1페이지 청크(과목명·교과개요 등 헤더 필드가 다 들어있음) 몇 개를
    뽑아 few-shot 예시로 쓴다. 매번 같은 예시를 쓰도록 시드를 고정해 재현 가능하게 한다."""
    global _FEWSHOT_CACHE
    if _FEWSHOT_CACHE is None:
        res = collection().get(include=["documents", "metadatas"])
        candidates = [d for d, m in zip(res["documents"], res["metadatas"])
                      if "CMN17" in m.get("source", "") and m.get("page") == 1 and "교과개요" in d]
        random.Random(0).shuffle(candidates)
        _FEWSHOT_CACHE = candidates[:8]
    return _FEWSHOT_CACHE[:n]


PROMPT = """다음은 DGIST 실제 강의계획서 조각 {n}개다. 형식(과목명/학기/담당교수/이수구분/학점/
강의시간/강의실/소속/교과개요/교과목표/주차별 강의계획 등 필드)을 참고하라.

{examples}

이제 아래 관심 분야를 가진 학생에게 어울릴 법한 가상의 강의계획서 조각을 위와 **같은 형식**으로
하나 작성하라. 실제 존재하는 과목일 필요는 없다 — 그 분야를 다룰 법한 과목명, 교과개요, 교과목표,
주차별 강의계획을 그럴듯하게 지어내면 된다. 다른 설명 없이 강의계획서 본문만 출력하라.

관심 분야: {query}"""


def generate(query: str, n_examples: int = 2) -> str:
    """query에 대한 가상의 강의계획서 조각을 생성한다. LLM 호출이 실패하면(서버 다운 등)
    원래 query를 그대로 돌려줘서(사실상 HyDE 없이 검색) 호출부가 죽지 않게 한다."""
    examples = _fewshot_examples(n_examples)
    prompt = PROMPT.format(n=len(examples), examples="\n\n---\n\n".join(examples), query=query)
    try:
        text = chat([{"role": "user", "content": prompt}])
    except Exception as e:
        print(f"    HyDE 생성 실패({e}) — 원래 질문으로 대체")
        return query
    return text.strip() or query
