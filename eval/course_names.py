"""강의계획서 PDF 파일명 <-> 정식 과목명 변환 유틸. eval/ 안의 GT 추출·채점 스크립트가 공유한다."""
import re

# crawl_syllabus.py가 만든 PDF 파일명 패턴: {SHYY}{SHTM_DCD}_{과목번호}_{분반}_{과목명}.pdf
SOURCE_RE = re.compile(r"^\d{4}CMN17\.\d{2}_[^_]+_\d+_(?P<name>.+)\.pdf$", re.I)


def strip_paren(name: str) -> str:
    """맨 끝 괄호 하나만 벗긴다(예: '...(이,공)' 이수구분 태그). 과목명 자체에 괄호가 들어있는
    경우(예: '심화화학실험Ⅰ(유기,분석화학)')는 그 괄호가 마지막이 아니라 유지된다."""
    return re.sub(r"[\(（][^)）]*[\)）]\s*$", "", name).strip()


def norm(name: str) -> str:
    return re.sub(r"\s+", "", name.strip())


def course_name_from_source(source: str) -> str | None:
    """강의계획서 PDF 파일명에서 정식 과목명을 뽑는다. 매뉴얼 등 패턴이 안 맞으면 None."""
    m = SOURCE_RE.match(source or "")
    return strip_paren(m.group("name")) if m else None
