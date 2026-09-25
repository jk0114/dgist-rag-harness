"""전역 설정: 경로 + LLM 백엔드 + 검색 개수.
청킹 크기/중복, 임베딩 모델 선택은 indexing/config.py 에 있다(색인 시점에만 정하고,
검색 때는 retrieval/core.py 가 그 값을 그대로 따라간다 — 임베딩 모델이 색인/검색에서
어긋나면 유사도 자체가 의미 없어지기 때문)."""
from pathlib import Path

# ---- 경로 ----
PDF_DIR = Path(r"E:\DGIST\MS\RAGProject\DB")      # PDF 폴더 (하위 폴더까지 전부 색인됨)
DB_DIR = Path(r"E:\DGIST\MS\RAGProject\chroma")   # 벡터 DB 저장 위치 (indexing/ingest.py가 생성)
COLLECTION = "univ"
OCR_CACHE = DB_DIR / "ocr_cache.json"             # 이미지 페이지 OCR 결과 (ingest.py --ocr 로 생성)

# ---- LLM 백엔드 (LM Studio 서버, OpenAI 호환) ----
LM_URL = "http://localhost:1234/v1"
LM_MODEL = "qwen/qwen3.5-9b"   # LM Studio > Developer 탭의 API identifier와 동일하게
TEMPERATURE = 0.0
MAX_TOKENS = 1024
THINKING = False               # Qwen3.5 thinking 모드. True면 답변 전에 추론 토큰을 쓴다(느리고 max_tokens 소모)

# ---- 검색 ----
TOP_K = 4
