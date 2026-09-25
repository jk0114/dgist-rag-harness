from pathlib import Path

# ---- 경로 ----
PDF_DIR = Path(r"E:\DGIST\MS\RAGProject\DB")      # PDF 폴더 (하위 폴더까지 전부 색인됨)
DB_DIR = Path(r"E:\DGIST\MS\RAGProject\chroma")   # 벡터 DB 저장 위치 (ingest.py가 생성)
COLLECTION = "univ"
OCR_CACHE = DB_DIR / "ocr_cache.json"             # 이미지 페이지 OCR 결과 (ingest.py --ocr 로 생성)

# ---- LLM 백엔드 (LM Studio 서버, OpenAI 호환) ----
LM_URL = "http://localhost:1234/v1"
LM_MODEL = "qwen/qwen3.5-9b"   # LM Studio > Developer 탭의 API identifier와 동일하게
TEMPERATURE = 0.0
MAX_TOKENS = 1024
THINKING = False               # Qwen3.5 thinking 모드. True면 답변 전에 추론 토큰을 쓴다(느리고 max_tokens 소모)

# ---- 임베딩 ----
EMBED_MODEL = "BAAI/bge-m3"

# ---- 청킹 ----
OCR_MAX_CHARS = 250      # 텍스트 레이어가 이보다 짧은 페이지는 이미지 페이지로 보고 OCR 텍스트를 합친다
MIN_PAGE_CHARS = 80      # 머리글·쪽번호를 걷어낸 뒤에도 이보다 짧으면 버림 (skipped_pages.json에 기록)
MAX_CHUNK_CHARS = 1200   # 페이지 텍스트가 이보다 길면 분할
OVERLAP_CHARS = 150

# ---- 검색 ----
TOP_K = 4
