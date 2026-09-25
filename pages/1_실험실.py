"""실험용 시각화 페이지 (Streamlit 멀티페이지 — 사이드바에 "1 실험실"로 자동으로 뜸).

질문 하나를 넣으면:
  1) vanilla(dense만) vs 그래프-RAG 유사도를 나란히 비교하고, 점수가 얼마나 바뀌었는지 표로 보여준다
  2) 그 후보들 사이의 그래프(구조적 인접 / 키워드 공유)를 노드-엣지로 보여준다
     (그래프-RAG가 실제로 쓰는 pipeline_gnn_ret.graph_retrieve()/pipeline_track_graph.graph_retrieve()를
      그대로 호출한다 — dense top-k만 보여주면 "그래프 덕분에 풀 밖에서 새로 끌려온 청크"가 아예 안
      보이는 문제가 있어서, 1·2번 모두 그래프 재점수화까지 끝난 결과를 그대로 쓴다)
"""
import html as html_mod
import streamlit as st
import plotly.graph_objects as go
from pyvis.network import Network

from core import retrieve
import pipeline_gnn_ret as gr
import pipeline_track_graph as tg

html_escape = html_mod.escape

st.set_page_config(page_title="RAG 실험실", page_icon="🔬", layout="wide")
st.title("🔬 RAG 실험실")
st.caption("질문 → vanilla/그래프-RAG 유사도 비교, 그래프 시각화까지 한 번에 확인합니다.")

query = st.text_input("질문", value="저는 인공지능에 관심이 많은데 어떤 과목을 들으면 좋을까요?")
c1, c2 = st.columns(2)
with c1:
    graph_choice = st.radio("그래프 방식", ["gnn_ret (구조+키워드)", "track_graph (트랙)"])
with c2:
    top_k = st.slider("볼 후보 개수", 5, 40, 15)
run = st.button("검색 실행", type="primary")

gmod = gr if graph_choice.startswith("gnn_ret") else tg


def _score_to_color(score: float, lo: float = 0.3, hi: float = 0.75) -> str:
    """유사도(magnitude)를 밝은 파랑 -> 진한 파랑 한 가지 색조로만 표현 (sequential)."""
    t = max(0.0, min(1.0, (score - lo) / (hi - lo)))
    c1_, c2_ = (222, 236, 249), (23, 78, 137)  # 밝은 파랑 -> 진한 파랑
    r, g, b = (int(c1_[i] + (c2_[i] - c1_[i]) * t) for i in range(3))
    return f"rgb({r},{g},{b})"


def _name(h: dict) -> str:
    t = (h.get("title") or h["source"]).strip()
    return f"{t} (p.{h['page']})"


if run and query.strip():
    # 버튼 클릭 시점의 질문만 session_state에 고정한다. 아래 슬라이더/라디오를 움직여서 생기는
    # 재실행에서는 run이 다시 False가 되지만, 고정해둔 질문이 있으면 이 블록이 계속 돌아야
    # top_k/final_k/그래프 선택 같은 위젯들이 실제로 반응한다 (안 그러면 버튼을 다시 눌러야만 반영됨).
    st.session_state["exp_query"] = query.strip()

active_query = st.session_state.get("exp_query", "")

if active_query:
    with st.spinner("검색 중..."):
        # vanilla 순위는 별도로 진짜 top_k dense 검색을 한다 (top_k가 그래프 쪽 POOL_K보다 클 수도 있어서).
        vanilla_pool = retrieve(active_query, top_k)
        vanilla_sorted = sorted(vanilla_pool, key=lambda h: h["score"], reverse=True)
        vanilla_top = vanilla_sorted[:top_k]
        vanilla_ids = {h["id"] for h in vanilla_top}

        # graph_rescore(): 실제 그래프-RAG 파이프라인이 쓰는 재점수화를 '자르지 않고' 전부 돌려준다.
        # dense top-k 밖에서 그래프로 새로 끌려온 청크(예: 기계학습개론)의 vanilla 점수도 여기 이미
        # 들어있고(프런티어 청크는 직접 코사인 유사도를 계산해서 채워둠), top-k 밖으로 밀린 vanilla
        # 후보의 graph_score도 여기서 같이 계산되므로 "top-k 밖이라 점수가 없다"는 경우가 안 생긴다.
        if graph_choice.startswith("gnn_ret"):
            all_scored = gr.graph_rescore(active_query)
        else:
            all_scored = tg.graph_rescore(active_query, max(top_k, tg.POOL_K))
        info_by_id = {h["id"]: h for h in all_scored}

        # vanilla_top 중에 그래프 쪽 후보 풀에도 안 잡힌 애(아주 드묾)가 있으면, 최소한 자기 자신
        # 점수로라도 graph_score를 채워준다 (이 풀 안에서는 그래프 이웃 정보를 아예 못 찾았다는 뜻).
        for h in vanilla_top:
            if h["id"] not in info_by_id:
                h["graph_score"] = h["score"]
                info_by_id[h["id"]] = h

        graph_sorted = sorted(info_by_id.values(), key=lambda h: h["graph_score"], reverse=True)
        graph_hits = graph_sorted[:top_k]
        graph_ids = {h["id"] for h in graph_hits}
        graph_rank = {h["id"]: i + 1 for i, h in enumerate(graph_hits)}

    # ==== 1) vanilla vs 그래프-RAG 유사도 비교 ====
    st.subheader("1. Query ↔ Chunk 유사도 — vanilla vs 그래프-RAG")
    union_ids = list(vanilla_ids | graph_ids)
    union_ids.sort(key=lambda cid: graph_rank.get(cid, 999))

    def _label(cid):
        h = info_by_id[cid]
        return f"{(h.get('title') or h['source'])[:24]} (p.{h['page']})"

    labels = [_label(cid) for cid in union_ids]
    # 이제 info_by_id 전원이 score/graph_score를 다 갖고 있어서, top-k 밖이라고 값이 비는 경우가 없다.
    vanilla_scores = [info_by_id[cid]["score"] for cid in union_ids]
    graph_scores = [info_by_id[cid]["graph_score"] for cid in union_ids]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=vanilla_scores[::-1], y=labels[::-1], orientation="h",
                          name="vanilla (dense score)", marker_color="#B9CBD8",
                          hovertemplate="%{y}<br>vanilla 유사도 %{x:.3f}<extra></extra>"))
    fig.add_trace(go.Bar(x=graph_scores[::-1], y=labels[::-1], orientation="h",
                          name=f"그래프-RAG ({graph_choice.split()[0]})", marker_color="#174E89",
                          hovertemplate="%{y}<br>graph 점수 %{x:.3f}<extra></extra>"))
    fig.update_layout(barmode="group", height=max(360, 30 * len(union_ids)),
                       margin=dict(l=10, r=10, t=10, b=10), xaxis_title="유사도 / 점수",
                       xaxis_range=[0, 1], legend=dict(orientation="h", yanchor="bottom", y=1.02))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**점수 변화표** (vanilla top-k 안이었는지/그래프 top-k 안인지도 같이 표시)")
    delta_rows = []
    for cid in union_ids:
        h = info_by_id[cid]
        vs, gs = h["score"], h["graph_score"]
        tag = "🆕 신규 진입" if cid not in vanilla_ids else ("❌ 이탈" if cid not in graph_ids else "✅ 유지")
        delta_rows.append({
            "상태": tag,
            "청크": _name(h),
            "vanilla 점수": round(vs, 3),
            "graph 점수": round(gs, 3),
            "변화(Δ)": round(gs - vs, 3),
        })
    st.dataframe(delta_rows, use_container_width=True, hide_index=True)

    with st.expander("본문 미리보기"):
        for cid in union_ids:
            h = info_by_id[cid]
            st.markdown(f"**{h['score']:.3f}** — `{h['source']}` p.{h['page']} · {h['title']}")
            st.caption(h["text"][:200].replace("\n", " ") + "…")

    # ==== 2) 그래프(구조적 인접 / 키워드 공유) 시각화 ====
    st.divider()
    st.subheader("2. 그래프 — 그래프-RAG가 실제로 후보로 삼은 청크들 사이의 edge")
    st.caption("dense top-k만이 아니라, 그래프로 새로 끌려온 청크(vanilla top-k 밖이었던 것)도 노드로 포함합니다 "
               "— 점선 테두리 + ✨ 표시가 그 청크입니다.")

    node_ids = graph_ids | vanilla_ids  # 그래프-RAG 후보 + 비교용 vanilla 후보를 합쳐서 보여준다
    graph_obj = gr.get_graph() if graph_choice.startswith("gnn_ret") else None

    struct_pairs: set[frozenset] = set()
    kw_pairs: dict[frozenset, set[str]] = {}
    if graph_obj is not None:
        for cid in node_ids:
            for nid in graph_obj.struct_neighbors.get(cid, set()):
                if nid in node_ids and nid != cid:
                    struct_pairs.add(frozenset((cid, nid)))
            for kw in graph_obj.keywords.get(cid, []):
                for nid in graph_obj.kw_index.get(gr._norm_kw(kw), set()):
                    if nid in node_ids and nid != cid:
                        kw_pairs.setdefault(frozenset((cid, nid)), set()).add(kw)
    else:  # track_graph: "같은 트랙" 또는 "같은 과목"을 edge로 취급
        for cid in node_ids:
            h_c = info_by_id[cid]
            tracks_c = tg.course_tracks(h_c["source"])
            for nid in node_ids:
                if nid == cid:
                    continue
                h_n = info_by_id[nid]
                pair = frozenset((cid, nid))
                if h_n["source"] == h_c["source"]:
                    struct_pairs.add(pair)
                else:
                    shared = tracks_c & tg.course_tracks(h_n["source"])
                    if shared:
                        kw_pairs.setdefault(pair, set()).update(shared)

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
        h = info_by_id[cid]
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
        sa, sb = info_by_id[a]["score"], info_by_id[b]["score"]
        net.add_edge(a, b, color="#7A8B99", width=2,
                     label=f"vanilla {sa:.2f} / {sb:.2f}",
                     title=f"[구조적 인접] 같은 문서(과목) 안 바로 옆 페이지<br>vanilla 유사도(그래프 적용 전) {sa:.3f} / {sb:.3f}")

    for pair, kws in kw_pairs.items():
        a, b = tuple(pair)
        sa, sb = info_by_id[a]["score"], info_by_id[b]["score"]
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
