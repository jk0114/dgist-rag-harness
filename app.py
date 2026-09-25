"""브라우저 하네스. 실행: streamlit run app.py
사이드바에서 파이프라인(no-rag / vanilla / …)을 갈아끼우며 SLM과 대화한다.
답변 아래에는 답변이 실제로 인용한 출처만 붙는다. 검색된 top-k 전체와 모델에 보낸 원문 프롬프트는
사이드바의 디버그 토글을 켰을 때만 보인다.
테마(라이트 흰+하늘 / 다크 검+적)는 .streamlit/config.toml, 말풍선·아바타 색은 아래 LIGHT/DARK_ 팔레트.
"""
import html, json, logging, re, time
from urllib.parse import quote
import streamlit as st
import streamlit.components.v1 as components
import config as C
from step2_retrieval.core import check_server
from step2_retrieval.pipelines import PIPELINES

# 출처 카드를 클릭하면 실제 PDF의 해당 페이지로 이동하도록: 파일명 -> DB/ 기준 상대경로 색인.
# (.streamlit/config.toml 의 enableStaticServing=true + static/DB 가 DB/ 의 실제 복사본이어야 서빙됨.
#  DB/ 에 PDF를 추가·교체했으면 static/DB 도 다시 복사해줘야 링크가 최신 PDF를 가리킴.)
_PDF_URL_INDEX = {p.name: p.relative_to(C.PDF_DIR).as_posix() for p in C.PDF_DIR.rglob("*.pdf")}


def pdf_url(source: str, page: int) -> str | None:
    rel = _PDF_URL_INDEX.get(source)
    return f"app/static/DB/{quote(rel)}#page={page}" if rel else None

# Streamlit 파일 감시기가 transformers의 지연 로딩 모듈을 건드려 내는 torchvision 경고만 숨긴다 (동작 무관).
logging.getLogger("streamlit.watcher.local_sources_watcher").setLevel(logging.ERROR)

st.set_page_config(page_title="DGIST 안내 도우미", page_icon="🎓", layout="centered")

# ---- 팔레트 (config.toml 의 테마와 같은 색) ----
# Streamlit 테마는 우상단 메뉴에서 즉시 바뀌고 스크립트는 다시 돌지 않는다. 그래서 색을 파이썬에서 굳히지 않고
# CSS 변수로 두고, 아래 작은 스크립트가 화면 배경색을 보고 <html data-theme> 를 갱신해 말풍선·아바타도 같이 바뀌게 한다.
LIGHT = dict(primary="#2EA8E5", user_bg="#E4F3FC", user_fg="#0F2A3A", user_bd="#CBE6F6",
             bot_bg="#F7FAFC", bot_bd="#E3ECF3", card="#EEF5FA", muted="#7A8B99")
DARK_ = dict(primary="#E63946", user_bg="#2A1518", user_fg="#F5EDED", user_bd="#4A1F24",
             bot_bg="#141416", bot_bd="#26262B", card="#1C1C20", muted="#8E8E96")
try:
    _first = DARK_ if st.context.theme.type == "dark" else LIGHT   # 첫 화면용 초기값 (이후엔 스크립트가 맞춤)
except Exception:
    _first = LIGHT
def _vars(p): return "".join(f"--{k.replace('_', '-')}:{v};" for k, v in p.items())
SPARK = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32" width="60" height="60"><circle cx="16" cy="16" r="16" fill="currentColor"/><path d="M16 7l2.4 6.6L25 16l-6.6 2.4L16 25l-2.4-6.6L7 16l6.6-2.4z" fill="#fff"/></svg>'

st.markdown(f"""<style>
:root {{ {_vars(_first)} }}
html[data-theme="light"] {{ {_vars(LIGHT)} }}
html[data-theme="dark"] {{ {_vars(DARK_)} }}

[data-testid="stAppDeployButton"] {{ display: none; }}
.block-container {{ max-width: 900px; padding-top: 4.5rem; padding-bottom: 6rem; }}
[data-testid="stBottomBlockContainer"] {{ max-width: 900px; }}

/* 말풍선 구분용 마커와 테마 감시 iframe 은 자리 차지 없이 숨김 */
[data-testid="stElementContainer"]:has(.m-user), [data-testid="stElementContainer"]:has(.m-bot),
[data-testid="stElementContainer"]:has(iframe[title="st.iframe"]) {{ display: none; }}

[data-testid="stChatMessage"] {{ background: transparent; padding: .2rem 0; gap: .6rem; align-items: flex-start; }}
[data-testid="stChatMessageContent"] {{ padding: .75rem 1.05rem; border-radius: 18px; line-height: 1.6; }}
[data-testid="stChatMessageContent"] > [data-testid="stVerticalBlock"] {{ gap: .55rem; }}

/* 사용자: 오른쪽 말풍선, 아바타 없음 */
[data-testid="stChatMessage"]:has(.m-user) [data-testid^="stChatMessageAvatar"] {{ display: none; }}
[data-testid="stChatMessage"]:has(.m-user) [data-testid="stChatMessageContent"] {{
  flex: 0 1 auto; margin: 0 0 0 auto; max-width: 78%; background: var(--user-bg); color: var(--user-fg);
  border: 1px solid var(--user-bd); border-bottom-right-radius: 6px; }}

/* 도우미: 왼쪽 카드 + 스파크 아바타 (아이콘 색은 CSS 가 칠하므로 테마 전환 즉시 반영) */
[data-testid="stChatMessage"]:has(.m-bot) [data-testid="stChatMessageContent"] {{
  margin: 0; background: var(--bot-bg); border: 1px solid var(--bot-bd); border-top-left-radius: 6px; }}
[data-testid="stChatMessage"]:has(.m-bot) [data-testid^="stChatMessageAvatar"] {{
  width: 34px; height: 34px; border-radius: 50%; background: var(--primary); color: #fff; box-shadow: none; }}
[data-testid="stChatMessage"]:has(.m-bot) [data-testid^="stChatMessageAvatar"] * {{ color: #fff; font-size: 20px; }}

.meta {{ font-size: .76rem; color: var(--muted); margin-top: .45rem; }}
.src {{ display: flex; gap: .65rem; align-items: flex-start; padding: .55rem .75rem; border-radius: 12px; background: var(--card); margin: .35rem 0; }}
.src-n {{ flex: 0 0 auto; width: 22px; height: 22px; border-radius: 50%; background: var(--primary); color: #fff;
          font-size: .74rem; font-weight: 700; display: flex; align-items: center; justify-content: center; margin-top: .1rem; }}
.src-t {{ font-size: .86rem; font-weight: 600; line-height: 1.35; }}
.src-t a {{ color: var(--primary); text-decoration: none; }}
.src-t a:hover {{ text-decoration: underline; }}
.src-p {{ font-weight: 400; color: var(--muted); }}
.src-x {{ font-size: .8rem; color: var(--muted); margin-top: .2rem; line-height: 1.45; }}

.hero {{ text-align: center; padding: 3.2rem 0 1rem; }}
.hero .spark {{ color: var(--primary); line-height: 0; }}
.hero h1 {{ font-size: 1.55rem; margin: .7rem 0 .25rem; }}
.hero p {{ color: var(--muted); margin: 0 0 1.2rem; font-size: .95rem; }}
.st-key-chips .stButton > button {{ border-radius: 999px; font-size: .86rem; padding: .45rem .9rem; }}

.pill {{ display: inline-flex; align-items: center; gap: .5rem; padding: .32rem .75rem; border-radius: 999px;
         font-size: .8rem; background: var(--card); max-width: 100%; }}
.pill span.t {{ overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.dot {{ flex: 0 0 auto; width: 8px; height: 8px; border-radius: 50%; background: #2ecc71; }}
.dot.off {{ background: #e74c3c; }}
</style>""", unsafe_allow_html=True)

# 테마 감시: 앱 배경 밝기로 light/dark 를 판정해 <html data-theme> 에 반영 (0.5초 간격, 화면 요소 없음)
components.html("""<script>
(function () {
  const doc = window.parent.document;
  function apply() {
    const app = doc.querySelector('.stApp'); if (!app) return;
    const m = getComputedStyle(app).backgroundColor.match(/\d+/g); if (!m) return;
    const t = (0.299 * m[0] + 0.587 * m[1] + 0.114 * m[2]) < 128 ? 'dark' : 'light';
    if (doc.documentElement.getAttribute('data-theme') !== t) doc.documentElement.setAttribute('data-theme', t);
  }
  apply(); setInterval(apply, 500);
})();
</script>""", height=0)

EXAMPLES = ["법인카드 기본 한도는 얼마야?", "연구수당 지급 결과에 이의가 있으면 어떻게 해?",
            "TA 신청은 어떻게 하지?", "연구실 안전교육은 어디서 들어?"]


# ---- 사이드바 ----
with st.sidebar:
    st.markdown("### RAG Harness")
    ok, info = check_server()
    served = [m.strip() for m in info.split(",")] if ok else []
    if ok:
        st.markdown(f'<div class="pill"><span class="dot"></span><span class="t">LM Studio · {html.escape(C.LM_MODEL)}</span></div>',
                    unsafe_allow_html=True)
        if served and C.LM_MODEL not in served:
            st.warning(f"`{C.LM_MODEL}` 이 로드돼 있지 않습니다. 로드된 모델: {', '.join(served)}")
    else:
        st.markdown('<div class="pill"><span class="dot off"></span><span class="t">LM Studio 연결 실패</span></div>',
                    unsafe_allow_html=True)
        st.caption("LM Studio → Developer → Start Server")
    if not (C.DB_DIR / "chroma.sqlite3").exists():
        st.warning("벡터 DB 없음. 먼저 `python ingest.py` 실행.")
    st.divider()
    names = list(PIPELINES)
    pipe = st.selectbox("파이프라인", names, index=names.index("vanilla") if "vanilla" in names else 0)
    st.caption(PIPELINES[pipe]["desc"])
    params = {"top_k": st.slider("top-k", 1, 10, C.TOP_K)}
    debug = st.toggle("디버그 보기", False, help="검색된 문서 전체와 모델에 보낸 프롬프트를 표시")
    st.divider()
    if st.button("새 대화", use_container_width=True, icon=":material/add:"):
        st.session_state.msgs = []
        st.rerun()


# ---- 말풍선 ----
def user_bubble(text: str):
    with st.chat_message("user"):
        st.markdown('<span class="m-user"></span>', unsafe_allow_html=True)
        st.markdown(text)


def bot_bubble(res: dict, debug: bool):
    """사용자에게는 답변 + 답변이 실제 인용한 출처만. 검색 전체·프롬프트는 디버그에서만."""
    hits = res.get("retrieved") or []
    cited = [n for n in sorted({int(x) for x in re.findall(r"\[(\d+)\]", res["answer"] or "")}) if 1 <= n <= len(hits)]
    with st.chat_message("assistant", avatar=":material/auto_awesome:"):
        st.markdown('<span class="m-bot"></span>', unsafe_allow_html=True)
        st.markdown(res["answer"] or "*(빈 응답)*")
        if cited:
            with st.expander(f"출처 {len(cited)}개", expanded=True):
                cards = []
                for n in cited:
                    h = hits[n - 1]
                    name = html.escape(re.sub(r"\.pdf$", "", h["source"], flags=re.I))
                    url = pdf_url(h["source"], h["page"])
                    name_html = f'<a href="{url}" target="_blank" rel="noopener">{name}</a>' if url else name
                    ex = html.escape(h["text"][:220].replace("\n", " ")) + ("…" if len(h["text"]) > 220 else "")
                    cards.append(f'<div class="src"><div class="src-n">{n}</div><div>'
                                 f'<div class="src-t">{name_html} <span class="src-p">· p.{h["page"]}</span></div>'
                                 f'<div class="src-x">{ex}</div></div></div>')
                st.markdown("".join(cards), unsafe_allow_html=True)
        meta = f'{res["pipeline"]} · {res["sec"]}s' + ("" if cited or not hits else " · 인용한 출처 없음")
        st.markdown(f'<div class="meta">{meta}</div>', unsafe_allow_html=True)
        if debug and hits:
            with st.expander(f"디버그 · 검색 결과 {len(hits)}개 (모델에 전부 삽입됨)"):
                for i, h in enumerate(hits, 1):
                    head = f"**[{i}]** 유사도 {h['score']:.3f}" + (" · 인용됨" if i in cited else "")
                    st.markdown(f"{head} — `{h['source']}` p.{h['page']} · {h['title']}")
                    st.text(h["text"][:600] + ("…" if len(h["text"]) > 600 else ""))
        if debug:
            with st.expander("디버그 · 모델에 보낸 프롬프트"):
                st.code(json.dumps(res["prompt"], ensure_ascii=False, indent=1), language="json")


# ---- 본문 ----
st.session_state.setdefault("msgs", [])
typed = st.chat_input("메시지를 입력하세요")          # 본문 위치와 무관하게 화면 하단에 고정됨
q = typed or st.session_state.pop("pending", None)

if not st.session_state.msgs and not q:
    st.markdown(f'<div class="hero"><div class="spark">{SPARK}</div>'
                '<h1>DGIST 안내 도우미</h1><p>학교 매뉴얼을 근거로 답하고, 출처를 함께 보여줍니다.</p></div>',
                unsafe_allow_html=True)
    with st.container(key="chips"):
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES):
            if cols[i % 2].button(ex, use_container_width=True):
                st.session_state.pending = ex
                st.rerun()

for m in st.session_state.msgs:
    if m["role"] == "user":
        user_bubble(m["content"])
    else:
        bot_bubble(m["res"], debug)

if q:
    st.session_state.msgs.append({"role": "user", "content": q})
    user_bubble(q)
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.msgs[:-1]]
    with st.spinner("답변 생성 중…"):
        t0 = time.time()
        try:
            res = PIPELINES[pipe]["fn"](q, history, params)
        except Exception as e:
            res = {"answer": f"오류: {e}", "retrieved": [], "prompt": []}
        res.update(pipeline=pipe, sec=round(time.time() - t0, 1))
    st.session_state.msgs.append({"role": "assistant", "content": res["answer"], "res": res})
    bot_bubble(res, debug)
