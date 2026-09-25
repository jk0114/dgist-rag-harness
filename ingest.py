"""PDF -> 페이지 단위 청크 -> bge-m3 임베딩 -> Chroma 저장.
실행:  python ingest.py          # chroma/ocr_cache.json 이 있으면 이미지 페이지에 OCR 텍스트를 합친다
       python ingest.py --ocr    # OCR 캐시를 (재)생성한 뒤 색인 (easyocr 필요, GPU 있으면 자동 사용)

페이지 처리 순서
  1. PyMuPDF 텍스트 레이어 추출 (블록을 위→아래, 왼→오른 순으로 정렬)
  2. 텍스트 레이어가 OCR_MAX_CHARS보다 짧으면 이미지 페이지로 보고 OCR 캐시의 텍스트를 뒤에 붙임
  3. 같은 PDF의 여러 페이지에 반복되는 줄(머리글·바닥글), 쪽번호, 한 글자짜리 줄 제거
  4. 남은 텍스트가 MIN_PAGE_CHARS보다 짧으면 버림 (skipped_pages.json에 기록)
"""
import hashlib, json, re, sys
from collections import Counter
import pymupdf
import torch
import chromadb
from sentence_transformers import SentenceTransformer
from config import *
from core import sanitize

_PAGE_NO = re.compile(r"^\s*[-–—]?\s*\d{1,3}\s*[-–—]?\s*$|^\s*\d{1,3}\s*/\s*\d{1,3}\s*$|^\s*(page|p\.)\s*\d{1,3}\s*$", re.I)


def page_text(page) -> str:
    """블록 순서(위→아래, 왼→오른)로 정렬해 슬라이드의 조각난 텍스트를 이어 붙인다."""
    blocks = page.get_text("blocks")
    blocks.sort(key=lambda b: (round(b[1] / 20), b[0]))  # y를 20pt 단위로 묶어 같은 줄 취급
    return "\n".join(b[4].strip() for b in blocks if b[4].strip())


def clean_lines(lines: list[str], boilerplate: set[str]) -> list[str]:
    out = []
    for ln in lines:
        s = ln.strip()
        if len(s) <= 1 or _PAGE_NO.match(s) or s in boilerplate:
            continue
        out.append(s)
    return out


def split_long(text: str):
    """MAX_CHUNK_CHARS 단위로 자르되, 가능하면 줄바꿈 위치에서 자른다."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    out, i = [], 0
    while i < len(text):
        j = min(i + MAX_CHUNK_CHARS, len(text))
        if j < len(text):
            nl = text.rfind("\n", i + MAX_CHUNK_CHARS // 2, j)
            if nl > 0:
                j = nl
        out.append(text[i:j].strip())
        if j >= len(text):
            break
        i = j - OVERLAP_CHARS
    return [c for c in out if c]


# ---- OCR ----
def ocr_pdf_pages(reader, doc, pages: list[int], width: int = 1600, min_conf: float = 0.3) -> dict[str, str]:
    """페이지들을 렌더링해 easyocr로 읽고 {페이지번호: 텍스트}를 돌려준다. 글상자를 줄 단위로 묶는다."""
    res = {}
    for pno in pages:
        page = doc[pno - 1]
        z = width / page.rect.width
        img = page.get_pixmap(matrix=pymupdf.Matrix(z, z)).tobytes("png")
        boxes = [(b, t) for b, t, c in reader.readtext(img, batch_size=8) if c >= min_conf]
        boxes.sort(key=lambda x: (round(x[0][0][1] / 20), x[0][0][0]))
        rows, cur, cur_y = [], [], None
        for b, t in boxes:
            y = round(b[0][1] / 20)
            if cur_y is not None and y != cur_y:
                rows.append(" ".join(cur)); cur = []
            cur.append(t); cur_y = y
        if cur:
            rows.append(" ".join(cur))
        res[str(pno)] = "\n".join(rows)
    return res


def build_ocr_cache():
    import easyocr
    gpu = torch.cuda.is_available()
    reader = easyocr.Reader(["ko", "en"], gpu=gpu, verbose=False)
    cache = {"_meta": {"engine": f"easyocr {easyocr.__version__} ko+en", "width": 1600, "min_conf": 0.3,
                       "rule": f"text layer < {OCR_MAX_CHARS} chars"}}
    for pdf in sorted(PDF_DIR.rglob("*.pdf")):
        doc = pymupdf.open(pdf)
        pages = [i + 1 for i, p in enumerate(doc) if len(page_text(p)) < OCR_MAX_CHARS]
        if pages:
            print(f"OCR {pdf.name}: {len(pages)}p", flush=True)
            cache[pdf.name] = ocr_pdf_pages(reader, doc, pages)
    DB_DIR.mkdir(parents=True, exist_ok=True)
    OCR_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"OCR 캐시 저장: {OCR_CACHE}")


# ---- 청크 ----
def build_chunks():
    ocr = json.loads(OCR_CACHE.read_text(encoding="utf-8")) if OCR_CACHE.exists() else {}
    if not ocr:
        print("※ OCR 캐시 없음 — 이미지 페이지는 건너뜀. `python ingest.py --ocr` 로 생성 가능.")
    chunks, skipped, seen = [], [], {}
    for pdf in sorted(PDF_DIR.rglob("*.pdf")):
        digest = hashlib.sha1(pdf.read_bytes()).hexdigest()
        if digest in seen:  # 내용이 완전히 같은 파일(이름만 다름)은 한 번만 색인하고 출처에 두 이름을 적는다
            for c in chunks:
                if c["meta"]["source"].startswith(seen[digest]):
                    c["meta"]["source"] = f"{seen[digest]} (= {pdf.name})"
            print(f"{pdf.name}: {seen[digest]} 와 동일 파일 → 건너뜀")
            continue
        seen[digest] = pdf.name

        doc = pymupdf.open(pdf)
        raw = []  # (pno, lines)
        for pno, page in enumerate(doc, start=1):
            text = page_text(page)
            if len(text) < OCR_MAX_CHARS:
                text = (text + "\n" + ocr.get(pdf.name, {}).get(str(pno), "")).strip()
            raw.append((pno, sanitize(text).split("\n")))

        # 같은 PDF 안에서 여러 페이지에 되풀이되는 줄 = 머리글/바닥글
        freq = Counter(ln.strip() for _, lines in raw for ln in set(lines) if ln.strip())
        boilerplate = {ln for ln, n in freq.items() if len(doc) >= 6 and n >= max(3, len(doc) * 0.25)}

        n_ok = 0
        for pno, lines in raw:
            lines = clean_lines(lines, boilerplate)
            text = "\n".join(lines)
            if len(text) < MIN_PAGE_CHARS:
                skipped.append({"file": pdf.name, "page": pno, "chars": len(text)})
                continue
            n_ok += 1
            title = lines[0][:80]
            for si, sub in enumerate(split_long(text)):
                chunks.append({
                    "id": f"{pdf.stem}__p{pno}__{si}",
                    "text": sub,
                    "meta": {"source": pdf.name, "page": pno, "title": title},
                })
        print(f"{pdf.name}: {len(doc)}p 중 {n_ok}p 색인" + (f" (머리글 {len(boilerplate)}종 제거)" if boilerplate else ""))
    return chunks, skipped


def main():
    if "--ocr" in sys.argv:
        build_ocr_cache()
    chunks, skipped = build_chunks()
    print(f"\n청크 {len(chunks)}개, 건너뛴 페이지 {len(skipped)}개")
    DB_DIR.mkdir(parents=True, exist_ok=True)
    (DB_DIR / "skipped_pages.json").write_text(
        json.dumps(skipped, ensure_ascii=False, indent=2), encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"임베딩 모델 로드 ({device})...")
    emb = SentenceTransformer(EMBED_MODEL, device=device)

    client = chromadb.PersistentClient(str(DB_DIR))
    try:
        client.delete_collection(COLLECTION)   # 재실행 시 깨끗이 다시 만든다
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    B = 32
    for i in range(0, len(chunks), B):
        batch = chunks[i:i + B]
        vecs = emb.encode([c["text"] for c in batch], batch_size=B,
                          normalize_embeddings=True, show_progress_bar=False)
        col.add(ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["meta"] for c in batch],
                embeddings=vecs.tolist())
        print(f"  {min(i + B, len(chunks))}/{len(chunks)}", end="\r")
    print(f"\n완료. DB: {DB_DIR}  (건너뛴 페이지 목록: skipped_pages.json)")


if __name__ == "__main__":
    main()
