"""HyDE (Hypothetical Document Embeddings, Gao et al. 2023 ACL, "Precise Zero-Shot Dense
Retrieval without Relevance Labels").

질문을 그대로 임베딩해서 검색하는 대신, LLM한테 "이 질문에 맞는 가상의 문서를 써봐"라고 시킨 뒤
그 결과물을 임베딩해서 검색한다. 질문("~에 관심 있는데 어떤 과목?")과 실제 문서(강의계획서
특유의 문체·구조)는 표현이 많이 달라서 임베딩 공간에서도 멀리 떨어져 있을 수 있는데, "질문에
그럴듯하게 답하는 가짜 문서"는 실제 문서와 표현이 훨씬 비슷해서 검색이 더 잘 맞을 수 있다는 게
논문의 핵심 주장. 가짜 문서의 사실 여부(그런 과목이 실제로 있는지)는 중요하지 않다 —
"강의계획서답게 생긴 텍스트"이기만 하면 된다.

가상 문서에서 학기/담당교수/이수구분/학점/강의시간/강의실/소속 같은 행정 필드는 뺀다(과목명 +
교과개요 + 교과목표 + 주차별 강의계획만 생성). 처음엔 실제 강의계획서 양식을 그대로 흉내내게
했는데, 실측해보니 문제가 있었다: 이 행정 필드들은 전공과 무관하게 모든 강의계획서에 똑같이
들어있어서, 남겨두면 임베딩 유사도가 "내용이 비슷해서"가 아니라 "강의계획서 양식이 똑같이
생겨서" 오른다 — 실제로 AI 관심사 질문의 HyDE 문서가 AI 강의계획서(+0.09)보다 완전히 무관한
인문/사회 강의계획서(+0.11)에 유사도를 더 많이 올려주는 현상이 확인됐다(행정 매뉴얼 같은
비강의계획서 문서에는 상대적으로 덜 오름, +0.06). 즉 "강의계획서 vs 비강의계획서" 구분은
오히려 좋아지는데, 이 앱에서 정작 중요한 "관련 과목 vs 무관한 과목" 구분은 흐려지는 부작용이
있었다. 행정 필드를 빼서 내용(교과개요/교과목표/주차별 계획)만 남기면 이 오염이 줄어든다.

few-shot 예시는 실제 corpus(강의계획서 청크)에서 가져와 행정 필드를 뗀 뒤 사용한다 — 손으로
예시를 짜 넣지 않고 실제 corpus 내용을 재사용하되, 행정 필드는 프롬프트의 지시뿐 아니라
예시에서도 일관되게 안 보이게 한다(few-shot 예시가 지시문보다 형식에 더 강하게 영향을 준다).

vanilla/graph-rag 어느 쪽에도 끼울 수 있다: generate()가 만든 텍스트를 retrieve()나
graph_rescore()의 embed_text로 넘기면 된다(둘 다 "질문 대신 이 텍스트를 임베딩해서 검색"을
지원하도록 되어 있다). no-rag는 애초에 검색을 안 하므로 HyDE와 무관하다.
"""
import random

from .core import chat, collection

_FEWSHOT_CACHE: list[str] | None = None

# 실제 강의계획서 청크의 필드 라벨(각자 자기 줄에 단독으로 나옴). 이 중 내용이 담긴 것만 남기고
# 나머지(학기/담당교수/이수구분/학점/강의시간/강의실/소속/수업방법/학습윤리/교과정책 등)는 버린다.
_ALL_FIELDS = ["학기", "담당교수", "이수구분", "학점", "강의시간", "강의실", "소속",
               "교과개요", "교과목표", "주차별 강의계획", "수업방법", "학습윤리",
               "교과정책(방침)", "교재", "평가방법", "연락처"]
_KEEP_FIELDS = {"교과개요", "교과목표", "주차별 강의계획"}


def _content_only(text: str) -> str:
    """과목명(첫 줄) + 교과개요/교과목표/주차별 강의계획만 남기고 행정 필드는 지운다."""
    lines = text.split("\n")
    if not lines:
        return text
    out = [lines[0]]
    keep = False
    for line in lines[1:]:
        if line.strip() in _ALL_FIELDS:
            keep = line.strip() in _KEEP_FIELDS
            if keep:
                out.append(line)
            continue
        if keep:
            out.append(line)
    return "\n".join(out)


def _fewshot_examples(n: int = 2) -> list[str]:
    """corpus에서 강의계획서 1페이지 청크 몇 개를 뽑아, 행정 필드를 뗀 내용부만 few-shot
    예시로 쓴다. 매번 같은 예시를 쓰도록 시드를 고정해 재현 가능하게 한다."""
    global _FEWSHOT_CACHE
    if _FEWSHOT_CACHE is None:
        res = collection().get(include=["documents", "metadatas"])
        candidates = [d for d, m in zip(res["documents"], res["metadatas"])
                      if "CMN17" in m.get("source", "") and m.get("page") == 1 and "교과개요" in d]
        random.Random(0).shuffle(candidates)
        _FEWSHOT_CACHE = [_content_only(c) for c in candidates[:8]]
    return _FEWSHOT_CACHE[:n]


PROMPT = """다음은 DGIST 실제 강의계획서에서 학기·담당교수·학점·강의시간 같은 행정 정보를 빼고
내용 부분(과목명 / 교과개요 / 교과목표 / 주차별 강의계획)만 남긴 조각 {n}개다.

{examples}

이제 아래 관심 분야를 가진 학생에게 어울릴 법한 가상의 강의계획서를 위와 **같은 형식**(과목명 +
교과개요 + 교과목표 + 주차별 강의계획)으로 하나 작성하라. 실제 존재하는 과목일 필요는 없다 —
그 분야를 다룰 법한 과목명, 교과개요, 교과목표, 주차별 강의계획을 그럴듯하게 지어내면 된다.
학기/담당교수/이수구분/학점/강의시간/강의실/소속 같은 행정 정보는 절대 넣지 마라 — 모든 과목에
똑같이 들어있는 내용이라 넣으면 오히려 검색에 방해된다. 다른 설명 없이 본문만 출력하라.

관심 분야: {query}"""


def generate(query: str, n_examples: int = 2) -> str:
    """query에 대한 가상의 강의계획서 조각(행정 필드 제외)을 생성한다. LLM 호출이 실패하면
    (서버 다운 등) 원래 query를 그대로 돌려줘서(사실상 HyDE 없이 검색) 호출부가 죽지 않게 한다."""
    examples = _fewshot_examples(n_examples)
    prompt = PROMPT.format(n=len(examples), examples="\n\n---\n\n".join(examples), query=query)
    try:
        text = chat([{"role": "user", "content": prompt}])
    except Exception as e:
        print(f"    HyDE 생성 실패({e}) — 원래 질문으로 대체")
        return query
    return text.strip() or query
