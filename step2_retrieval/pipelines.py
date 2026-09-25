"""RAG 파이프라인 레지스트리.

세 가지만 둔다 — no-rag(대조군), vanilla(순정 RAG), graph-rag(retrieval/graph_rag.py):
새 변형을 추가하려면 함수를 하나 만들고 @register("이름")을 붙이면 UI 드롭다운과 CLI에 자동으로 뜬다.
시그니처:  fn(question: str, history: list[dict], params: dict) -> dict
반환 dict 키:
  answer    : 최종 답변 문자열
  retrieved : [{text, score, source, page, title}]  (검색 안 하면 [])
  prompt    : 실제로 모델에 보낸 messages (디버그/발표용)
history 는 이전 대화 [{"role","content"}] 목록 (retrieved 등은 제외된 순수 대화).
"""
from .core import retrieve, chat
import config as C

PIPELINES: dict[str, dict] = {}

def register(name: str, desc: str = ""):
    def deco(fn):
        PIPELINES[name] = {"fn": fn, "desc": desc}
        return fn
    return deco


SYSTEM_PLAIN = "너는 대학 안내 도우미다. 한국어로 간결하게 답하라."

# 검색은 항상 top-k를 돌려주므로 무관한 질문에도 근거가 붙는다. 그래서 모델이 먼저 질문 유형을 가르게 한다.
# (이전 프롬프트는 "반드시 근거만 사용"이라 "1+2는?"에도 "확인할 수 없습니다"로 답했음)
SYSTEM_RAG = """너는 DGIST 대학 안내 도우미다. 대화 상대는 학생·교직원 같은 일반 사용자이고, 아래 [근거]가 무엇인지 모른다.
[근거]는 학교 매뉴얼에서 검색된 문서 조각인데, 질문과 무관한 조각이 섞여 있을 수 있다.
먼저 사용자의 말이 어느 쪽인지 판단하라.
A. 학교의 제도·절차·시스템·규정에 관한 질문 → [근거]만으로 답하고 각 문장 끝에 근거 번호를 [1], [2] 형식으로 붙인다. [근거]에 답이 없으면 지어내지 말고 "그 내용은 학교 안내 자료에서 찾지 못했습니다."라고만 답한다.
B. 그 외의 일반 질문(계산, 상식 등) → [근거]는 무시하고 네 지식으로 바로 답한다. 근거 번호를 붙이지 않는다. 예: "1+2는?" → "3입니다." 단, 오늘 날씨·뉴스·현재 시각처럼 네가 알 수 없는 실시간 정보는 지어내지 말고 알 수 없다고 말한다.
C. 질문이 아닌 말(인사, 감사, 지시, 불만, 감탄, 욕설 등) → 안내 데스크 직원처럼 한두 문장으로 자연스럽게 대꾸한다. [근거]를 언급하지 않고 번호도 붙이지 않으며, "찾지 못했습니다"라고 답하지 않는다. 감사에는 감사로, 지시에는 알겠다고, 앞 답변이 틀렸다는 불만에는 사과하고 어느 부분이 틀렸는지 되묻는다. 무례한 말에도 차분하게 응대한다.
이전 대화의 답변에 근거 번호가 없었더라도, 이번 질문이 A 유형이면 반드시 각 문장 끝에 번호를 붙인다.
말투: 사용자에게 "근거", "제공된", "검색 결과" 같은 내부 용어를 쓰지 않는다. 자료를 가리킬 때는 "학교 안내 자료"라고만 부른다. 한국어로 간결하게 답한다."""

# 질문 바로 뒤에 붙는 규칙 재확인. 멀티턴에서 앞선 답변들에 인용이 없으면(일반 질문이 이어진 뒤) 9B 모델이 그 스타일을 따라
# 인용을 빼먹는데, 생성 직전에 규칙을 한 번 더 보여주면 막힌다 (LM Studio 실측: 인용 0개 → 문장마다 인용).
# 질문이 아닌 말("고마워", "지어내지 마")까지 A 유형으로 몰아 "찾지 못했습니다"로 답하던 것도 여기서 먼저 거른다.
RULE_TAIL = """

[답변 규칙] 먼저 위 말이 질문인지 판단한다. 질문이 아닌 말(인사·감사·지시·불만·욕설·감탄)이면 [근거]를 완전히 무시하고 안내 직원처럼 한두 문장으로 대꾸한다. "찾지 못했습니다"는 쓰지 않는다. 학교의 제도·절차·시스템에 관한 질문이면 [근거]만 사용하고 모든 문장 끝에 반드시 [번호]를 붙인다 (예: "…입니다. [1]"). 그 밖의 일반 질문이면 번호 없이 답한다."""


def _ctx(hits: list[dict]) -> str:
    return "\n\n".join(f"[{i}] ({h['source']} p.{h['page']} · {h['title']})\n{h['text']}"
                       for i, h in enumerate(hits, 1))


@register("no-rag", "대조군: 문서 없이 모델 자체 지식으로만 답변")
def no_rag(question, history, params):
    msgs = [{"role": "system", "content": SYSTEM_PLAIN}, *history,
            {"role": "user", "content": question}]
    return {"answer": chat(msgs), "retrieved": [], "prompt": msgs}


@register("vanilla", "순정 RAG: dense 검색 top-k를 조건 없이 근거로 삽입 → 인용 답변")
def vanilla(question, history, params):
    k = int(params.get("top_k", C.TOP_K))
    hits = retrieve(question, k)   # 질문이 무엇이든 top-k 전부 넣는다. 거르기(임계값·리랭크·라우팅)는 변형에서 다룬다.
    msgs = [{"role": "system", "content": SYSTEM_RAG}, *history,
            {"role": "user", "content": f"[근거]\n{_ctx(hits)}\n\n[질문]\n{question}{RULE_TAIL}"}]
    return {"answer": chat(msgs), "retrieved": hits, "prompt": msgs}


# ---- 이후 변형은 여기 아래에 추가 ----
# @register("hybrid", "dense + sparse 혼합 검색")
# def hybrid(question, history, params): ...

from . import graph_rag  # noqa: F401  (import 부수효과로 "graph-rag" 파이프라인이 PIPELINES에 등록됨)
