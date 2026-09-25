"""DGIST 개설과목/강의계획서(Syllabus) 크롤러 + PDF 생성.
대상: https://welcome.dgist.ac.kr/ucs/ucsqProfRespSbjtInq/index.do (로그인 불필요, 공개 조회 화면)

과목 목록(list.do)과 상세 내용(lecInfo.do)을 JSON으로 받아, 과목별 PDF를 직접 만든다.
(사이트의 "출력" 버튼이 여는 리포트 뷰어로 원본 그대로 PDF를 뽑을 수도 있지만, 그 PDF는
텍스트가 전부 벡터 그림으로 그려져 있어 PyMuPDF로 텍스트 추출이 안 된다 — 즉 ingest.py가
색인을 못 한다. 그래서 검색 가능한 진짜 텍스트 레이어를 가진 PDF를 직접 만든다.)

주의: 이 사이트의 JSON API는 requests/curl 같은 일반 HTTP 클라이언트로 요청하면 한글이 깨져서
온다(브라우저가 아닌 클라이언트를 구분해 응답을 다르게 주는 것으로 보임). 그래서 Playwright로
실제 브라우저를 띄운 뒤, 그 브라우저 안에서 fetch()를 실행해 호출한다.

크롤링 범위: 2026-1/2학기는 전부, 2025-1/2학기는 "2026학년도에 없는 과목명"만
  (2026학년도 두 학기에 등장하는 과목명 집합을 먼저 만들고, 2025학년도 과목 중 그 집합에
   없는 이름만 포함시킨다 — 같은 과목이 매 학기 반복 개설되는 경우가 많아 중복을 줄이기 위함)

실행:  python crawl_syllabus.py
     (최초 1회) pip install playwright && playwright install chromium
출력:  SyllabusDB/syllabus.jsonl   -- 과목 1개 = 1줄(JSON, 상세 내용 포함)
       SyllabusDB/excluded.jsonl  -- 제외된 과목(UGRP/URP/Thesis/Internship). 필터가 맞는지 검토용.
       SyllabusDB/pdf/*.pdf       -- 과목별 PDF (다른 매뉴얼들과 동일하게 ingest.py가 색인)

이후: SyllabusDB/pdf 의 PDF를 DB/ 밑으로 복사하고 python ingest.py 를 실행하면 색인된다.
"""
import html
import json
import re
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

INDEX_URL = "https://welcome.dgist.ac.kr/ucs/ucsqProfRespSbjtInq/index.do"
OUT_DIR = Path(__file__).parent / "SyllabusDB"
OUT_PATH = OUT_DIR / "syllabus.jsonl"
PDF_DIR = OUT_DIR / "pdf"

# ---- 크롤링 대상 ----
# 조직분류 코드: 대학원=CMN12.02, 대학=CMN12.03, AI대학=CMN12.12, 전문대학원=CMN12.10, 경영대학원=CMN12.11
ORG_CODE = "CMN12.03"          # 대학(학사과정)
ORG_NAME = "대학"

# (학기코드, 표시이름, 그룹) — 그룹 "2026"의 과목명은 전부 받고, "2025"는 2026에 없는 이름만 받는다.
# 학기코드는 selectProfYearTermCombo.do 응답의 CODE 값(예: "2026CMN17.20" = 2026/2학기).
SEMESTERS = [
    ("2026CMN17.20", "2026-2학기", "2026"),
    ("2026CMN17.10", "2026-1학기", "2026"),
    ("2025CMN17.20", "2025-2학기", "2025"),
    ("2025CMN17.10", "2025-1학기", "2025"),
]
PRIORITY_GROUP = "2026"  # 이 그룹의 과목명은 전부 포함, 나머지 그룹은 이 그룹에 없는 이름만 포함

# ---- 제외 규칙: UGRP / URP / Thesis(논문) / Internship(인턴십) ----
# 실제 데이터로 확인한 결과, 이 과목군은 전부 과목번호가 RP(연구) 또는 INT(인턴십)로 시작한다.
EXCLUDE_SBJT_PREFIXES = ("RP", "INT")
EXCLUDE_KEYWORDS = ("ugrp", "urp", "thesis", "internship", "논문", "인턴십")


def is_excluded(row: dict) -> bool:
    no = row.get("SBJT_NO", "")
    name = row.get("SBJT_NM", "")
    if no.startswith(EXCLUDE_SBJT_PREFIXES):
        return True
    low = name.lower()
    return any(k in low for k in EXCLUDE_KEYWORDS)


def norm_name(name: str) -> str:
    return (name or "").strip()


def safe_filename(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|]', "_", name)
    return name.strip()[:150]


def pdf_filename(r: dict) -> str:
    return safe_filename(f"{r['SHYY']}{r['SHTM_DCD']}_{r['SBJT_NO']}_{r['CLSS_NO']}_{r['SBJT_NM']}") + ".pdf"


JS_FETCH = """
async (opts) => {
    const body = new URLSearchParams(opts.data);
    const resp = await fetch(opts.url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'X-Requested-With': 'XMLHttpRequest'
        },
        body: body.toString()
    });
    return await resp.json();
}
"""


def fetch(page, url: str, data: dict, retries: int = 6) -> dict:
    payload = {"url": url, "data": {k: str(v) for k, v in data.items()}}
    last_err = None
    for attempt in range(retries):
        try:
            return page.evaluate(JS_FETCH, payload)
        except Exception as e:
            last_err = e
            time.sleep(1.0 * (attempt + 1))
    raise last_err


def get_combo(page, path: str, org_code: str) -> list[dict]:
    r = fetch(page, f"/ucs/ucsqProfRespSbjtInq/{path}", {
        "commonMenuId": "", "commonProgramId": "UcsqProfRespSbjtInq",
        "searchOrgnClsfDcd": org_code, "searchLang": "ko",
    })
    return r.get("list", [])


def get_curri_years(page, org_code: str) -> list[dict]:
    return get_combo(page, "selectCuriShyy.do", org_code)


def get_course_list(page, org_code: str, year_term: str, curri_shyy: str) -> list[dict]:
    r = fetch(page, "/ucs/ucsqProfRespSbjtInq/list.do", {
        "pageNum": 1, "pageSize": 99999,
        "commonMenuId": "", "commonProgramId": "UcsqProfRespSbjtInq",
        "searchLang": "ko", "searchOrgnClsfDcd": org_code, "langPssbFlag": "N",
        "searchOrgnClsfDcd1": org_code, "searchOrgnClsfDcd2": org_code,
        "selectYearTerm": year_term, "searchCuriShyy": curri_shyy,
        "searchSust": "", "searchCors": "", "searchCptnDcd": "", "searchSbjtDetaDcd": "",
        "searchSbNo": "", "searchNm": "", "searchProfNm": "",
        "_search": "false", "rows": 99999, "page": 1, "sidx": "", "sord": "asc",
    })
    return r.get("user", [])


def get_semester_included(page, org_code: str, year_term: str, curri_years: list[dict]) -> tuple[list, list, list]:
    """해당 학기의 (포함, 제외, 강의계획서없음) 과목 목록. 입학년도 버킷 전체를 합쳐 중복 제거한다."""
    merged: dict[tuple, dict] = {}
    for cy in curri_years:
        rows = get_course_list(page, org_code, year_term, cy["CODE"])
        for r in rows:
            key = (r["SHYY"], r["SHTM_DCD"], r["SBJT_NO"], r["CLSS_NO"])
            merged[key] = r
    all_rows = list(merged.values())

    included, excluded, no_syllabus = [], [], []
    for r in all_rows:
        if is_excluded(r):
            excluded.append(r)
        elif str(r.get("LECPLN_CNT", 0)) in ("0", ""):
            no_syllabus.append(r)
        else:
            included.append(r)
    return included, excluded, no_syllabus


def get_syllabus_detail(page, row: dict) -> dict:
    return fetch(page, "/ucs/ucseLsnPdocMngt/lecInfo.do", {
        "pageNum": 1, "pageSize": 10,
        "commonMenuId": "", "commonProgramId": "UcsqProfRespSbjtInq",
        "conYear": row["SHYY"], "conTerm": row["SHTM_DCD"],
        "conSust": row["ASGN_SUST_CD"], "conCors": row["ASGN_CORS_DCD"],
        "conOrgn": row["ORGN_CLSF_DCD"], "conSbjtNo": row["SBJT_NO"],
        "conClss": row["CLSS_NO"], "conLang": "ko",
    })


# ---- PDF 생성 (직접 만든 텍스트 PDF — 검색 가능한 진짜 텍스트 레이어를 가짐) ----
TERM_NAME = {"CMN17.10": "1학기", "CMN17.11": "계절학기(여름)", "CMN17.20": "2학기", "CMN17.21": "계절학기(겨울)"}

# 담당교수 개인 연락처(전화번호 등)는 강의계획서 PDF에 넣지 않는다.
PDF_SECTIONS = [
    ("교과개요", "LT_SUMA"),
    ("교과목표", "LT_PURO"),
    ("주차별 강의계획", "SCHETCHUL"),
    ("수업방법", "LSN_MTHD"),
    ("학습윤리", "LRN_ITGT"),
    ("교과정책(방침)", "LT_POLY"),
    ("준비물 및 기타", "ETC"),
]


def render_html(r: dict) -> str:
    title = f"{r['SBJT_NM']} ({r['SBJT_NO']}-{r['CLSS_NO']})"
    term = f"{r['SHYY']}년 {TERM_NAME.get(r['SHTM_DCD'], r['SHTM_DCD'])}"

    meta_rows = [
        ("학기", term), ("담당교수", r.get("PROF_NM", "")),
        ("이수구분", r.get("CPTN_NM", "")), ("학점", r.get("PNT", "")),
        ("강의시간", r.get("TLSN_TIME", "")), ("강의실", r.get("LABRM_NM", "")),
        ("소속", r.get("ASGN_SUST_NM", "")),
    ]
    meta_html = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in meta_rows if v
    )

    body_html = ""
    for label, key in PDF_SECTIONS:
        val = (r.get(key) or "").strip()
        if not val:
            continue
        body_html += f'<h2>{html.escape(label)}</h2><pre>{html.escape(val)}</pre>'

    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<style>
  body {{ font-family: "Malgun Gothic", "맑은 고딕", sans-serif; font-size: 12px; line-height: 1.55; color: #222; margin: 30px; }}
  h1 {{ font-size: 18px; border-bottom: 2px solid #333; padding-bottom: 8px; }}
  h2 {{ font-size: 14px; margin-top: 20px; color: #1a4d8f; }}
  table {{ border-collapse: collapse; margin: 12px 0; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 5px 8px; text-align: left; font-size: 11px; }}
  th {{ background: #f0f4f8; width: 110px; white-space: nowrap; }}
  pre {{ white-space: pre-wrap; font-family: inherit; margin: 4px 0; }}
</style></head>
<body>
  <h1>{html.escape(title)}</h1>
  <table>{meta_html}</table>
  {body_html}
</body></html>"""


def load_existing() -> dict[tuple, dict]:
    """이미 syllabus.jsonl에 상세까지 받아둔 과목은 다시 받지 않고 재사용한다."""
    if not OUT_PATH.exists():
        return {}
    out = {}
    for line in OUT_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        key = (r["SHYY"], r["SHTM_DCD"], r["SBJT_NO"], r["CLSS_NO"])
        out[key] = r
    return out


def main():
    OUT_DIR.mkdir(exist_ok=True)
    PDF_DIR.mkdir(exist_ok=True)
    existing = load_existing()
    print(f"기존에 상세까지 받아둔 과목: {len(existing)}개 (재사용)")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(INDEX_URL, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(1000)

        curri_years = get_curri_years(page, ORG_CODE)
        print(f"조직: {ORG_NAME}({ORG_CODE}) / 입학년도(커리큘럼) {len(curri_years)}개: "
              + ", ".join(f"{c['CODE']}({c['NAME']})" for c in curri_years))

        per_semester = {}
        all_excluded = []
        for code, label, group in SEMESTERS:
            included, excluded, no_syl = get_semester_included(page, ORG_CODE, code, curri_years)
            per_semester[code] = included
            all_excluded.extend(excluded)
            print(f"  {label}: 조회 {len(included)+len(excluded)+len(no_syl)}개 / "
                  f"제외 {len(excluded)}개 / 강의계획서없음 {len(no_syl)}개 / 대상 {len(included)}개")

        priority_names = {
            norm_name(r["SBJT_NM"])
            for code, label, group in SEMESTERS if group == PRIORITY_GROUP
            for r in per_semester[code]
        }
        print(f"\n{PRIORITY_GROUP}학년도 과목명 집합: {len(priority_names)}개")

        final_rows = []
        for code, label, group in SEMESTERS:
            rows = per_semester[code]
            if group == PRIORITY_GROUP:
                keep = rows
            else:
                keep = [r for r in rows if norm_name(r["SBJT_NM"]) not in priority_names]
                print(f"  {label}: {PRIORITY_GROUP}학년도와 과목명 중복 아닌 것만 {len(keep)}/{len(rows)}개 채택")
            final_rows.extend(keep)

        print(f"\n최종 대상: {len(final_rows)}개 과목\n")

        with (OUT_DIR / "excluded.jsonl").open("w", encoding="utf-8") as f:
            for r in all_excluded:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        failed = []
        results = []
        for i, r in enumerate(final_rows, 1):
            key = (r["SHYY"], r["SHTM_DCD"], r["SBJT_NO"], r["CLSS_NO"])
            cached = existing.get(key)
            if cached and cached.get("LT_SUMA") is not None:
                results.append(cached)
                continue
            try:
                detail = get_syllabus_detail(page, r)
            except Exception as e:
                print(f"  실패({e}): {r['SBJT_NO']} {r['SBJT_NM']} — 건너뜀")
                failed.append(r)
                continue
            results.append({**r, **detail})
            if i % 20 == 0 or i == len(final_rows):
                print(f"  {i}/{len(final_rows)}  {r['SBJT_NO']} {r['SBJT_NM']}")
            time.sleep(0.3)  # 서버 부담을 줄이기 위한 최소한의 간격

        with OUT_PATH.open("w", encoding="utf-8") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        if failed:
            print(f"\n끝내 실패한 과목 {len(failed)}개 (스크립트를 다시 돌리면 재시도됩니다):")
            for r in failed:
                print("  ", r["SBJT_NO"], r["SBJT_NM"])

        print(f"\nPDF 생성 중...")
        for i, r in enumerate(results, 1):
            fname = pdf_filename(r)
            page.set_content(render_html(r), wait_until="load")
            page.pdf(path=str(PDF_DIR / fname), format="A4",
                     margin={"top": "15mm", "bottom": "15mm", "left": "15mm", "right": "15mm"})
            if i % 20 == 0 or i == len(results):
                print(f"  {i}/{len(results)}  {fname}")

        browser.close()

    print(f"\n완료: {OUT_PATH} ({len(results)}개), PDF: {PDF_DIR}")
    print(f"제외 목록(검토용): {OUT_DIR / 'excluded.jsonl'}")


if __name__ == "__main__":
    main()
