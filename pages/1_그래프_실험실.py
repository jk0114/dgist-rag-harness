"""실험용 시각화 페이지 (Streamlit 멀티페이지 — 사이드바에 "1 그래프 실험실"로 자동으로 뜸).

질문 하나를 넣으면:
  1) vanilla / graph-rag / vanilla+hyde / graph-rag+hyde 4개 파이프라인의 Query-Chunk 유사도를
     한 화면에서 비교한다 (막대그래프 + 표)
  2) 그 후보들 사이의 그래프(구조적 인접 / 키워드 공유)를 노드-엣지로 보여준다 (HyDE 적용 전,
     즉 순정 graph-rag 풀 기준 — 구조/키워드 edge 자체는 HyDE로 바뀌는 게 아니라 "어떤 청크가
     후보로 뽑혔는지"만 바뀌므로, 그래프 구조를 보는 이 파트는 하나의 기준으로 고정해둔다)

4개 숫자를 얻는 방법: step2_retrieval.graph_rag.graph_rescore()를 (a) 원래 질문, (b) HyDE가
지어낸 가상 문서로 각각 한 번씩만 돌린다. 결과 dict마다 이미 "score"(dense 유사도, = vanilla
점수)와 "graph_score"(그래프 재점수화 후, = graph-rag 점수)가 같이 들어있어서, 호출 2번으로
vanilla/graph-rag/vanilla+hyde/graph-rag+hyde 4개를 전부 얻는다(따로 4번 검색할 필요 없음).
"""
import html as html_mod
import streamlit as st
import plotly.graph_objects as go
from pyvis.network import Network

from step2_retrieval.core import embedder, sanitize
from step2_retrieval import graph_rag as gr
from step2_retrieval import hyde

html_escape = html_mod.escape

st.set_page_config(page_title="RAG 실험실", page_icon="🔬", layout="wide")
st.title("🔬 RAG 실험실")
st.caption("질문 → vanilla/graph-rag/+hyde 4가지 유사도 비교, 그래프 시각화까지 한 번에 확인합니다.")

query = st.text_input("질문", value="저는 인공지능에 관심이 많은데 어떤 과목을 들으면 좋을까요?")
top_k = st.slider("볼 후보 개수", 5, 40, 15)
run = st.button("검색 실행", type="primary")

VARIANTS = [
    ("vanilla", "score", "plain", "#B9CBD8"),
    ("graph-rag", "graph_score", "plain", "#174E89"),
    ("vanilla+hyde", "score", "hyde", "#F5C6A5"),
    ("graph-rag+hyde", "graph_score", "hyde", "#C0392B"),
]


def _score_to_color(score: float, lo: float = 0.3, hi: float = 0.75) -> str:
    """유사도(magnitude)를 밝은 파랑 -> 진한 파랑 한 가지 색조로만 표현 (sequential)."""
    t = max(0.0, min(1.0, (score - lo) / (hi - lo)))
    c1_, c2_ = (222, 236, 249), (23, 78, 137)  # 밝은 파랑 -> 진한 파랑
    r, g, b = (int(c1_[i] + (c2_[i] - c1_[i]) * t) for i in range(3))
    return f"rgb({r},{g},{b})"


def _name(h: dict) -> str:
    t = (h.get("title") or h["source"]).strip()
    return f"{t} (p.{h['page']})"


def _complete_pool(pool_by_id: dict, want_ids: set[str], embed_text: str):
    """pool_by_id는 graph_rescore()가 돌려준, 자기 검색 풀 안의 청크만 score/graph_score를
    갖고 있다. want_ids(비교 화면에 보여줄 청크 전체) 중 이 풀에 없는 것들을, 직접 코사인
    유사도를 계산해(graph_rag._fetch_by_ids 재사용) 채워 넣는다 — 그래야 "이 파이프라인은
    이 청크를 원래 못 찾았다"가 아니라 "이 청크에 대한 점수는 실제로 이거다"를 4개 다
    비교할 수 있다. graph_score는 이웃(그래프 상 연결된 청크)의 score로 계산하는데, 이웃도
    풀에 없으면 한 단계 더 가져와서 채운다."""
    graph_obj = gr.get_graph()
    missing = [cid for cid in want_ids if cid not in pool_by_id]
    if not missing:
        return
    q_vec = embedder("cpu").encode([sanitize(embed_text)], normalize_embeddings=True).tolist()[0]
    pool_by_id.update(gr._fetch_by_ids(missing, q_vec))
    neighbor_missing = set()
    for cid in missing:
        neighbor_missing |= graph_obj.neighbors(cid) - pool_by_id.keys()
    if neighbor_missing:
        pool_by_id.update(gr._fetch_by_ids(list(neighbor_missing), q_vec))
    for cid in missing:
        h = pool_by_id[cid]
        neighbor_ids = graph_obj.neighbors(cid) & pool_by_id.keys()
        neighbor_scores = [pool_by_id[n]["score"] for n in neighbor_ids]
        boost = max(neighbor_scores) if neighbor_scores else h["score"]
        h["graph_score"] = gr.ALPHA * h["score"] + (1 - gr.ALPHA) * boost


if run and query.strip():
    # 버튼 클릭 시점의 질문만 session_state에 고정한다. 아래 슬라이더를 움직여서 생기는
    # 재실행에서는 run이 다시 False가 되지만, 고정해둔 질문이 있으면 이 블록이 계속 돌아야
    # top_k 위젯이 실제로 반응한다 (안 그러면 버튼을 다시 눌러야만 반영됨).
    st.session_state["exp_query"] = query.strip()

active_query = st.session_state.get("exp_query", "")

if active_query:
    # HyDE 텍스트는 질문이 안 바뀌면 재생성하지 않는다 — top_k 슬라이더만 움직여도 이 블록
    # 전체가 다시 실행되는데, 그때마다 LLM을 다시 부르면 느리고(HyDE는 생성에 몇 초 걸림)
    # 결과도 temperature=0이라 어차피 똑같다.
    if st.session_state.get("exp_hyde_key") != active_query:
        with st.spinner("HyDE 가상 문서 생성 중..."):
            st.session_state["exp_hyde_text"] = hyde.generate(active_query)
            st.session_state["exp_hyde_key"] = active_query
    hyde_text = st.session_state["exp_hyde_text"]
    with st.expander("HyDE가 지어낸 가상 문서 (vanilla+hyde/graph-rag+hyde가 이걸 임베딩해서 검색함)"):
        st.text(hyde_text)

    with st.spinner("검색 중... (원래 질문 1번 + HyDE 문서 1번, 그래프 재점수화까지)"):
        # graph_rescore() 결과 하나에 score(=vanilla류)와 graph_score(=graph-rag류)가 같이
        # 들어있어서, 이 호출 2번으로 4개 파이프라인 점수를 전부 얻는다.
        plain_all = gr.graph_rescore(active_query, embed_text=None)
        hyde_all = gr.graph_rescore(active_query, embed_text=hyde_text)
        plain_by_id = {h["id"]: h for h in plain_all}
        hyde_by_id = {h["id"]: h for h in hyde_all}
        meta_by_id = {**plain_by_id, **hyde_by_id}  # source/page/title/text는 둘 다 동일, 점수만 다름

        pool_by_source = {"plain": plain_by_id, "hyde": hyde_by_id}

        def topk_ids(score_key: str, source: str) -> list[str]:
            pool = pool_by_source[source]
            ranked = sorted(pool.values(), key=lambda h: h[score_key], reverse=True)
            return [h["id"] for h in ranked[:top_k]]

        variant_topk_ordered = {name: topk_ids(score_key, source) for name, score_key, source, _ in VARIANTS}
        variant_topk = {name: set(ids) for name, ids in variant_topk_ordered.items()}

    # ==== 1) 4개 파이프라인 Query-Chunk 유사도 비교 ====
    st.subheader("1. Query ↔ Chunk 유사도 — vanilla / graph-rag / vanilla+hyde / graph-rag+hyde")
    st.caption("4개 중 하나라도 top-k에 뽑은 청크를 전부 모아, 그 청크에 대한 4개 파이프라인의 "
               "실제 점수를 전부 계산해서 나란히 비교합니다(어떤 파이프라인이 원래 못 찾았던 "
               "청크도 직접 유사도를 계산해 채워 넣습니다 — 빈 칸이 남지 않습니다).")

    union_ids = set()
    for ids in variant_topk.values():
        union_ids |= ids

    with st.spinner("빠진 조합 점수 채우는 중..."):
        _complete_pool(plain_by_id, union_ids, active_query)
        _complete_pool(hyde_by_id, union_ids, hyde_text)
        meta_by_id = {**plain_by_id, **hyde_by_id}

    def _score_of(cid: str, score_key: str, source: str):
        h = pool_by_source[source].get(cid)
        return h[score_key] if h else None

    def _best_score(cid: str) -> float:
        vals = [v for _, score_key, source, _ in VARIANTS if (v := _score_of(cid, score_key, source)) is not None]
        return max(vals) if vals else 0.0

    union_ids = sorted(union_ids, key=_best_score, reverse=True)

    def _label(cid):
        h = meta_by_id[cid]
        return f"{(h.get('title') or h['source'])[:24]} (p.{h['page']})"

    labels = [_label(cid) for cid in union_ids]

    fig = go.Figure()
    for name, score_key, source, color in VARIANTS:
        scores = [_score_of(cid, score_key, source) for cid in union_ids]
        fig.add_trace(go.Bar(x=scores[::-1], y=labels[::-1], orientation="h",
                              name=name, marker_color=color,
                              hovertemplate=f"%{{y}}<br>{name} %{{x:.3f}}<extra></extra>"))
    fig.update_layout(barmode="group", height=max(420, 30 * len(union_ids)),
                       margin=dict(l=10, r=10, t=10, b=10), xaxis_title="유사도 / 점수",
                       xaxis_range=[0, 1], legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"**Top-{top_k} 순위표** (각 파이프라인이 몇 번째로 어떤 청크를 뽑았는지 — "
                "같은 줄이라도 파이프라인마다 다른 청크일 수 있습니다)")
    rank_rows = []
    for i in range(top_k):
        row = {"순위": i + 1}
        for name, _, _, _ in VARIANTS:
            ids = variant_topk_ordered[name]
            row[name] = _label(ids[i]) if i < len(ids) else "—"
        rank_rows.append(row)
    st.dataframe(rank_rows, use_container_width=True, hide_index=True)

    st.markdown("**점수표** (마지막 열은 graph-rag+hyde가 vanilla 대비 얼마나 바뀌었는지)")
    rows = []
    for cid in union_ids:
        h = meta_by_id[cid]
        vals = {name: _score_of(cid, score_key, source) for name, score_key, source, _ in VARIANTS}
        row = {"청크": _name(h)}
        for name, _, _, _ in VARIANTS:
            v = vals[name]
            row[name] = round(v, 3) if v is not None else "—"
        v0, v3 = vals["vanilla"], vals["graph-rag+hyde"]
        row["Δ(graph+hyde-vanilla)"] = round(v3 - v0, 3) if v0 is not None and v3 is not None else "—"
        rows.append(row)
    st.dataframe(rows, use_container_width=True, hide_index=True)

    with st.expander("본문 미리보기"):
        for cid in union_ids:
            h = meta_by_id[cid]
            st.markdown(f"**{_best_score(cid):.3f}** — `{h['source']}` p.{h['page']} · {h['title']}")
            st.caption(h["text"][:200].replace("\n", " ") + "…")

    # ==== 2) 그래프(구조적 인접 / 키워드 공유) 시각화 ====
    st.divider()
    st.subheader("2. 그래프 — graph-rag가 실제로 후보로 삼은 청크들 사이의 edge")
    st.caption("HyDE 적용 전(순정 질문) 기준의 graph-rag 후보로 고정해서 보여줍니다 — 구조/키워드 edge 자체는 "
               "HyDE로 바뀌지 않고 '어떤 청크가 후보인지'만 바뀌기 때문입니다. "
               "dense top-k만이 아니라, 그래프로 새로 끌려온 청크(vanilla top-k 밖이었던 것)도 노드로 포함합니다 "
               "— 점선 테두리 + ✨ 표시가 그 청크입니다.")

    vanilla_ids = variant_topk["vanilla"]
    graph_ids = variant_topk["graph-rag"]
    node_ids = graph_ids | vanilla_ids  # graph-rag 후보 + 비교용 vanilla 후보를 합쳐서 보여준다
    graph_obj = gr.get_graph()

    struct_pairs: set[frozenset] = set()
    kw_pairs: dict[frozenset, set[str]] = {}
    for cid in node_ids:
        for nid in graph_obj.struct_neighbors.get(cid, set()):
            if nid in node_ids and nid != cid:
                struct_pairs.add(frozenset((cid, nid)))
        for kw in graph_obj.keywords.get(cid, []):
            for nid in graph_obj.kw_index.get(gr._norm_kw(kw), set()):
                if nid in node_ids and nid != cid:
                    kw_pairs.setdefault(frozenset((cid, nid)), set()).add(kw)

    node_struct_ids = {n for pair in struct_pairs for n in pair}
    node_kw_ids = {n for pair in kw_pairs for n in pair}

    def _node_border(nid: str) -> str:
        in_struct, in_kw = nid in node_struct_ids, nid in node_kw_ids
        if in_struct and in_kw:
            return "#8E44AD"
        if in_struct:
            return "#7A8B99"
        if in_kw:
            return "#E67E22"
        return "#CCCCCC"

    net = Network(height="700px", width="100%", directed=False, notebook=False, cdn_resources="in_line")
    net.barnes_hut()

    for cid in node_ids:
        h = plain_by_id[cid]
        is_frontier = cid not in vanilla_ids  # dense top-k 밖이었는데 그래프 덕분에 들어온 노드
        title_text = (h.get("title") or h["source"]).strip()
        badge = "✨ " if is_frontier else ""
        gscore = h["graph_score"]
        label = f"{badge}{title_text[:22]}\nvanilla {h['score']:.3f} → graph {gscore:.3f}"
        tooltip = (f"<b>{html_escape(title_text)}</b><br>{html_escape(h['source'])} p.{h['page']}"
                   f"<br>vanilla 유사도(그래프 적용 전): {h['score']:.3f}"
                   f"<br>graph 점수(재점수화 후): {gscore:.3f}"
                   + ("<br><b>✨ dense top-k 밖에서 그래프로 새로 끌려옴</b>" if is_frontier else ""))
        net.add_node(cid, label=label, title=tooltip, shape="box",
                     color={"background": _score_to_color(h["score"]), "border": _node_border(cid)},
                     borderWidth=3, borderWidthSelected=5, shapeProperties={"borderDashes": is_frontier},
                     font={"multi": True, "align": "center"})

    # edge에 적는 숫자는 항상 "그래프를 돌리기 전, 순정 dense 검색에서의" 질문-청크 유사도다
    # (graph_score가 아니라 원본 score) — 이래야 "이 두 청크가 원래는 이만큼 유사했는데 연결 덕분에
    # 점수가 어떻게 바뀌었는지"를 노드 라벨(vanilla → graph)과 같이 보고 판단할 수 있다.
    for pair in struct_pairs:
        a, b = tuple(pair)
        sa, sb = plain_by_id[a]["score"], plain_by_id[b]["score"]
        net.add_edge(a, b, color="#7A8B99", width=2,
                     label=f"vanilla {sa:.2f} / {sb:.2f}",
                     title=f"[구조적 인접] 같은 문서(과목) 안 바로 옆 페이지<br>vanilla 유사도(그래프 적용 전) {sa:.3f} / {sb:.3f}")

    for pair, kws in kw_pairs.items():
        a, b = tuple(pair)
        sa, sb = plain_by_id[a]["score"], plain_by_id[b]["score"]
        kw_text = ", ".join(sorted(kws))
        net.add_edge(a, b, color="#E67E22", width=1, dashes=True,
                     label=f"vanilla {sa:.2f} / {sb:.2f}",
                     title=f"[키워드 공유: {html_escape(kw_text)}]<br>vanilla 유사도(그래프 적용 전) {sa:.3f} / {sb:.3f}")

    st.caption(
        f"노드 {len(node_ids)}개 (그중 ✨ 신규 {len(node_ids - vanilla_ids)}개) · "
        f"구조적 edge {len(struct_pairs)}개(━ 회색) · 키워드 edge {len(kw_pairs)}개(┅ 주황) · "
        f"edge 라벨 = 두 청크 각각의 **vanilla(그래프 적용 전) 유사도**. "
        f"노드 테두리색: 회색=구조 연결만 / 주황=키워드 연결만 / 보라=둘 다. "
        f"노드 라벨은 'vanilla 점수 → graph 점수'."
    )
    st.components.v1.html(net.generate_html(notebook=False), height=730, scrolling=True)
