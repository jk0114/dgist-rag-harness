"""공통 로직: 텍스트 정화, 임베딩/검색, LLM 호출. 파이프라인과 UI는 이 모듈만 사용한다."""
import re
import chromadb
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import config as C
from indexing.config import EMBED_MODEL

# 모델 특수 토큰처럼 보이는 문자열(<|im_start|>, <|endoftext|> 등)을 문서 텍스트에서 제거.
# 문서 본문이 채팅 템플릿의 역할 토큰으로 오해되는 것을 막는다.
_SPECIAL = re.compile(r"<\|[^|<>]{1,40}\|>|</?s>|\[INST\]|\[/INST\]|<\|eot_id\|>")

def sanitize(text: str) -> str:
    return _SPECIAL.sub(" ", text)


# ---- 임베딩 / 벡터 DB ----
_emb = {}
_col = None

def embedder(device: str = "cpu") -> SentenceTransformer:
    """indexing/config.py 의 EMBED_MODEL을 쓴다 — 색인 때 쓴 모델과 검색 때 쓰는 모델이
    같아야 유사도 비교가 의미 있다."""
    if device not in _emb:
        _emb[device] = SentenceTransformer(EMBED_MODEL, device=device)
    return _emb[device]

def collection():
    global _col
    if _col is None:
        _col = chromadb.PersistentClient(str(C.DB_DIR)).get_collection(C.COLLECTION)
    return _col

def retrieve(query: str, k: int = C.TOP_K) -> list[dict]:
    """[{id, text, score(cosine sim), source, page, title}] 를 점수 내림차순으로 반환."""
    q = embedder("cpu").encode([sanitize(query)], normalize_embeddings=True).tolist()
    r = collection().query(query_embeddings=q, n_results=k,
                           include=["documents", "metadatas", "distances"])
    return [{"id": i, "text": d, "score": round(1 - dist, 4), **m}
            for i, d, m, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0])]


# ---- LLM ----
_llm = OpenAI(base_url=C.LM_URL, api_key="lm-studio")

NO_ANSWER = ("[답변 없음] 모델이 thinking(추론)에 max_tokens를 전부 써서 본문이 비었습니다. "
             "LM Studio 모델 설정에서 Thinking을 끄거나 config.MAX_TOKENS를 올리세요.")

def chat(messages: list[dict], model: str | None = None) -> str:
    """messages = [{"role": "system"|"user"|"assistant", "content": str}, ...]
    채팅 템플릿(역할 특수 토큰)은 LM Studio가 GGUF 내장 템플릿으로 적용한다. 여기서는 평문만 보낸다.
    Qwen3.5는 기본이 thinking 모드라 답변 전에 추론(reasoning_content)을 쓰는데, 그 토큰이 max_tokens를
    다 먹으면 content가 빈 채로 돌아온다. LM Studio에서 확인된 유일한 스위치는 reasoning_effort="none"
    (chat_template_kwargs·enable_thinking·/no_think 는 무시됨). 그래도 비면 빈 문자열 대신 원인을 돌려준다."""
    extra = {} if C.THINKING else {"reasoning_effort": "none"}
    out = _llm.chat.completions.create(
        model=model or C.LM_MODEL, messages=messages,
        temperature=C.TEMPERATURE, max_tokens=C.MAX_TOKENS, extra_body=extra)
    choice = out.choices[0]
    text = choice.message.content or ""
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S)  # 본문에 섞여 온 think 블록 제거
    text = re.sub(r"<think>.*\Z", "", text, flags=re.S).strip()     # 잘려서 닫히지 않은 think 블록 제거
    if not text and choice.finish_reason == "length":
        return NO_ANSWER
    return text

def check_server() -> tuple[bool, str]:
    try:
        ids = [m.id for m in _llm.models.list().data]
        return True, ", ".join(ids)
    except Exception as e:
        return False, str(e)
