"""색인 실험 노브: 청킹 크기/중복, 임베딩 모델.

이 값들을 바꾸고 `python -m step1_indexing.ingest` 를 다시 돌리면 새 설정으로 색인이 다시
만들어진다(같은 chroma 컬렉션을 지우고 새로 채움 — step1_indexing/ingest.py 참고).
step2_retrieval/core.py 는 검색 시점에 EMBED_MODEL 을 여기서 그대로 읽어온다: 색인 때 쓴
임베딩 모델과 검색 때 쓰는 모델이 다르면 벡터 공간이 달라져서 유사도 비교 자체가
무의미해지기 때문이다.
"""

# ---- 임베딩: 청크/질문을 어떤 사전학습 모델로 벡터화할지 ----
# bge-m3: 다국어(한국어 포함) 지원, 현재 기본값. 교체하려면 sentence-transformers 로 로드
# 가능한 HuggingFace 모델 이름이면 아무거나 가능 (예: "intfloat/multilingual-e5-large").
EMBED_MODEL = "BAAI/bge-m3"

# ---- 청킹: 페이지 텍스트를 얼마나 잘게/얼마나 겹치게 자를지 ----
MAX_CHUNK_CHARS = 1200   # 청크 최대 길이(자). 작게 하면 검색 정밀도↑ 문맥↓, 크게 하면 반대.
OVERLAP_CHARS = 150      # 청크 경계에서 겹치는 길이(자). 0이면 겹침 없음 — 경계에서 문맥이 끊길 수 있음.

# ---- OCR/페이지 필터링 (청킹 전 전처리) ----
OCR_MAX_CHARS = 250      # 텍스트 레이어가 이보다 짧은 페이지는 이미지 페이지로 보고 OCR 텍스트를 합친다
MIN_PAGE_CHARS = 80      # 머리글·쪽번호를 걷어낸 뒤에도 이보다 짧으면 버림 (skipped_pages.json에 기록)
