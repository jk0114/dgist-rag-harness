"""'2026 DGIST 연구실별 추천 코스트리.pdf'에서 교수별 연구분야/추천과목을 직접 파싱해
평가용 GT(정답)를 만든다. (step3_eval/run_eval.py가 이 결과를 읽어 질의를 구성한다.)

pymupdf/pdftotext(poppler) 둘 다 이 PDF에서 한글이 깨진 텍스트를 내놓는데(이전에는 OCR로
우회했었다), 원인을 확인해보니 PDF에 폰트별 ToUnicode CMap 자체는 정상인데 일부 폰트가
중첩된 Form XObject의 로컬 리소스 딕셔너리에 같은 리소스 이름("C0_0")으로 다시 정의돼 있어
(다른 폰트인데 이름이 겹침) get_text() 류 API가 리소스 이름을 전역으로 취급하면서 엉뚱한
CMap을 적용해 벌어진 일이었다. pdfminer.six는 리소스를 올바르게 스코프해서 완전히 정확한
텍스트를 뽑아준다 — 그래서 OCR도 LLM도 필요 없이 규칙 기반으로 바로 파싱한다.

페이지 레이아웃: 교수 1명이 페이지 절반(좌/우 컬럼)을 차지한다(일부는 1명이 전체를 씀).
pdfminer가 돌려주는 텍스트 줄(LTTextLine)은 x좌표를 갖고 있어서, 그 x좌표로 반쪽을 나누면
"추천수강과목" 표의 각 항목도 이미 줄 단위로 잘 분리돼 있다(OCR처럼 여러 열이 한 줄에
섞이는 문제 자체가 없음) — 그래서 학년(열) 구분 없이 표 영역의 불릿(· ) 줄을 모두 모으는
것만으로 과목 전체 목록을 정확히 얻는다.

실행:  python -m step3_eval.extract_course_tree
출력:  step3_eval/data/course_tree_gt.json   {"p{page}_{L|R}": {professor, lab_ko, lab_en,
    research_fields, courses, courses_unmatched} | null}
courses_unmatched는 표에는 있지만 정식 과목명 목록과 매칭 못 한 원문(강의계획서 DB에 없거나
표기가 많이 다른 경우) — GT(courses)에는 안 들어가지만 확인용으로 같이 남긴다.
"""
import json
import re
from pathlib import Path

from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTTextLine

from .course_names import SOURCE_RE, strip_paren, norm

PDF_PATH = "2026 DGIST 연구실별 추천 코스트리.pdf"
GT_PATH = Path(__file__).parent / "data" / "course_tree_gt.json"
SYLLABUS_DIR = Path("DB/Syllabus")

_PHONE_RE = re.compile(r"^T\.\s*0\d{2}")
_EMAIL_RE = re.compile(r"E\.\s*([A-Za-z0-9._%+-]+@dgist\.ac\.kr)")
_BULLET_RE = re.compile(r"^[·•・]\s*")
_GRADE_RE = re.compile(r"^[1234]\s*학년$")

_NUM = r"(?:[IVX]+|[ⅠⅡⅢⅣⅤⅥⅦⅧⅨⅩ])"
_TAIL_RE = re.compile(rf"^(?P<base>.*?)\s*(?P<nums>{_NUM}(?:\s*,\s*{_NUM})*)$")
_TRAILING_NUM_RE = re.compile(rf"{_NUM}\s*$")
_ROMAN = {"I": "Ⅰ", "II": "Ⅱ", "III": "Ⅲ", "IV": "Ⅳ", "V": "Ⅴ", "VI": "Ⅵ"}
_FOOTNOTE_RE = re.compile(r"[†‡*※]+\s*$")  # 표 안의 각주 기호(예: "디지털비전/영상처리†")


# ---- 1) pdfminer로 줄(bbox 포함) 뽑기 ----
def _page_lines(layout) -> list[tuple[float, float, str]]:
    """[(x0, y0, text), ...] — y0는 pdfminer 좌표계(아래가 0)라 위에서부터 읽으려면 내림차순 정렬."""
    out = []

    def walk(obj):
        if isinstance(obj, LTTextLine):
            text = obj.get_text().strip()
            if text:
                out.append((obj.x0, obj.y0, text))
        elif isinstance(obj, LTTextContainer):
            for child in obj:
                walk(child)

    for obj in layout:
        walk(obj)
    return out


# ---- 2) 과목명 정규화/매칭 ----
def canonical_map() -> dict[str, str]:
    names = set()
    for p in SYLLABUS_DIR.glob("*.pdf"):
        m = SOURCE_RE.match(p.name)
        if m:
            names.add(strip_paren(m.group("name")))
    return {norm(n): n for n in names}


def _text_variants(raw: str) -> list[str]:
    """각주 기호·부가 설명 괄호가 붙어 있을 수 있어, 뗀 버전도 후보로 같이 시도한다
    (예: "디지털비전/영상처리†", "회로이론과 계측법 (이론)")."""
    variants = [raw]
    no_foot = _FOOTNOTE_RE.sub("", raw).strip()
    if no_foot != raw:
        variants.append(no_foot)
    no_paren = re.sub(r"[\(（][^)）]*[\)）]\s*$", "", no_foot).strip()
    if no_paren and no_paren not in variants:
        variants.append(no_paren)
    return variants


def expand_course_text(raw: str) -> list[str]:
    """'일반화학 I,Ⅱ' -> ['일반화학I','일반화학Ⅱ','일반화학I'] 식으로, 원래 표기(I)와
    유니코드 로마숫자(Ⅰ) 변환본을 둘 다 후보로 낸다 — DGIST 파일명 자체가 일관되지 않아서
    (대부분 과목은 Ⅰ/Ⅱ를 쓰지만 "일반화학실험"처럼 일부는 그냥 영문자 I를 쓴다) 변환해버리면
    오히려 못 찾는 경우가 있다. 접미사 없으면 [raw] 그대로."""
    m = _TAIL_RE.match(raw)
    if not m or not m.group("base"):
        return [raw]
    base = m.group("base").strip()
    nums = [n.strip() for n in m.group("nums").split(",")]
    out = []
    for n in nums:
        out.append(base + n)
        conv = _ROMAN.get(n)
        if conv and conv != n:
            out.append(base + conv)
    return out


def _base_without_numeral(name: str) -> str:
    return norm(_TRAILING_NUM_RE.sub("", name))


def match_canonical(text: str, canon: dict[str, str]) -> list[str]:
    """정확히 일치하면 그거 하나. 접두어로만 일치하는 후보가 여럿이면(예: 학년 표시 없이
    "전기역학"만 쓴 경우) — 그 후보들이 전부 "로마숫자만 다른 같은 과목"일 때만 전부 인정한다.
    "컴퓨터"처럼 짧은 말이 서로 무관한 여러 과목의 접두어가 되는 경우는(코드가 그 차이를
    구분할 근거가 없으므로) 배제하고 빈 리스트를 돌려준다."""
    key = norm(text)
    if key in canon:
        return [canon[key]]
    hits = sorted({v for k, v in canon.items() if k.startswith(key)})
    if len(hits) <= 1:
        return hits
    bases = {_base_without_numeral(h) for h in hits}
    return hits if len(bases) == 1 else []


def resolve_course(raw_bullet: str, canon: dict[str, str]) -> tuple[list[str], bool]:
    """(매칭된 정식 과목명 목록, 후보를 하나라도 찾았는지). 두 번째 값이 False면 이 과목은
    강의계획서 DB에 없거나(개설 안 함/미크롤링) 표기가 너무 달라서 매칭 실패한 것 —
    parse_half()가 courses_unmatched 에 원문 그대로 남긴다."""
    out, seen, matched_any = [], set(), False
    for text in _text_variants(raw_bullet):
        for cand in expand_course_text(text):
            for name in match_canonical(cand, canon):
                matched_any = True
                if name not in seen:
                    seen.add(name)
                    out.append(name)
    return out, matched_any


# ---- 3) 반쪽(교수 1명) 단위로 파싱 ----
def parse_half(lines: list[tuple[float, float, str]], canon: dict[str, str]) -> dict | None:
    """lines: 이 반쪽에 속하는 (x0, y0, text), 위->아래(y0 내림차순) 정렬된 상태."""
    email_idx = next((i for i, (_, _, t) in enumerate(lines) if _EMAIL_RE.search(t)), None)
    if email_idx is None:
        return None

    # 실제 텍스트 추출 순서는 (전화, 이메일, [웹사이트 W. ... (있으면)], 이름+직함,
    # 연구실명(국문), 연구실명(영문)) — 화면에는 이름이 위쪽 색상 바에 크게 보이지만,
    # content stream 순서는 전화/이메일이 먼저 나온다. 웹사이트 줄은 일부 연구실(주로
    # 전기전자컴퓨터공학과)에만 있어서, 있으면 건너뛴다.
    idx = email_idx + 1
    while idx < len(lines) and (lines[idx][2].startswith("W.") or "http" in lines[idx][2]):
        idx += 1
    professor_raw = lines[idx][2] if idx < len(lines) else ""
    professor = re.sub(r"\s*(교수|부교수|조교수|연구교수|명예교수|석좌교수).*$", "", professor_raw).strip()

    lab_ko, lab_en = "", ""
    if idx + 1 < len(lines):
        lab_ko = lines[idx + 1][2]
    if idx + 2 < len(lines) and lines[idx + 2][2].startswith("("):
        lab_en = lines[idx + 2][2].strip("()")

    texts = [t for _, _, t in lines]
    fields_idx = next((i for i, t in enumerate(texts) if "주요 연구 분야" in t), None)
    fields: list[str] = []
    if fields_idx is not None:
        i = fields_idx + 1
        while i < len(texts) and "주요 업적" not in texts[i] and "추천수강과목" not in texts[i]:
            fields.extend(s.strip() for s in texts[i].split(",") if s.strip())
            i += 1
        fields = [re.sub(r"^[·•・]\s*", "", f) for f in fields]

    courses_idx = next((i for i, t in enumerate(texts) if "추천수강과목" in t), None)
    courses: list[str] = []
    unmatched: list[str] = []
    if courses_idx is not None:
        i = courses_idx + 1
        while i < len(texts) and texts[i] != "MEMO":
            t = texts[i]
            if _BULLET_RE.match(t) and not _GRADE_RE.match(t):
                raw = _BULLET_RE.sub("", t).strip()
                names, matched_any = resolve_course(raw, canon)
                for name in names:
                    if name not in courses:
                        courses.append(name)
                if not matched_any:
                    unmatched.append(raw)
            i += 1

    return {"professor": professor, "lab_ko": lab_ko, "lab_en": lab_en,
            "research_fields": fields, "courses": courses, "courses_unmatched": unmatched}


def main():
    canon = canonical_map()
    print(f"정식 과목명 {len(canon)}개")

    result = {}
    for pno, layout in enumerate(extract_pages(PDF_PATH), 1):
        width = layout.x1
        lines = _page_lines(layout)
        if not lines:
            continue
        lines.sort(key=lambda r: -r[1])
        mid = width / 2
        halves = {
            "L": [ln for ln in lines if ln[0] < mid],
            "R": [ln for ln in lines if ln[0] >= mid],
        }
        for side, hl in halves.items():
            key = f"p{pno}_{side}"
            entry = parse_half(hl, canon)
            result[key] = entry
            if entry:
                print(f"  {key}: {entry['professor']}  과목 {len(entry['courses'])}개", flush=True)

    GT_PATH.parent.mkdir(parents=True, exist_ok=True)
    GT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    n_profs = sum(1 for v in result.values() if v and v.get("courses"))
    print(f"\n완료: {GT_PATH} (교수 {n_profs}명, 전체 키 {len(result)}개)")


if __name__ == "__main__":
    main()
