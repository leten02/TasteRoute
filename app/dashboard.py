"""TasteRoute dashboard: plan a walking day on a map and browse the results.

Usage:  streamlit run app/dashboard.py
Needs:  data/processed/ (data_analysis_preprocessing/preprocess.py) and
        pip install -e ".[app]". Walking routes download the street network
        around the chosen places on first use (about a minute, then cached).

Tabs:
    코스 짜기   pick a start, category, user and ranking method; the day's
                places, the walking route on a map and per-leg distances
    실험 결과   model comparison, distance re-ranking (RQ1), learned distance
                model, route optimization (RQ3), from results/*.json
    데이터      preprocessing funnel and the EDA figures

Yelp data stays on this machine: the app reads data/processed/ locally and is
not meant to be deployed publicly (Yelp terms forbid redistribution).
"""
import datetime as dt
import json
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from tasteroute.candidates import CandidateFilter
from tasteroute.data import get_split, load_item_features, load_items, load_ratings, n_users_items
from tasteroute.geo import haversine, user_anchors
from tasteroute.item_extras import (DAYS, attribute_matrix, combine_features, hours_today, is_open_at,
                                    load_item_extras, open_mask)
from tasteroute.models.bpr import BPR
from tasteroute.models.cbf import CBF
from tasteroute.models.visit import LogisticRegression, VisitFeatures
from tasteroute.pipeline import plan_days
from tasteroute.roads import WALK_KMH, RoadNetwork

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
VIEW = (Path(__file__).parent / "route_view.html").read_text(encoding="utf-8")

ACCENT, ACCENT_TINT, INK, MUTED, BORDER, GREY = "#eb6834", "#fdeee7", "#1f2430", "#6b7280", "#e5e7eb", "#c9ced6"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]  # same fixed order as experiments/figures.py

STARTS = {
    "필라델피아 시청": (39.9526, -75.1636),
    "Reading Terminal Market": (39.9533, -75.1590),
    "Rittenhouse Square": (39.9496, -75.1718),
    "Old City · 독립기념관": (39.9489, -75.1500),
    "University City · 펜실베이니아대": (39.9522, -75.1932),
    "Italian Market": (39.9387, -75.1580),
    "Fishtown": (39.9725, -75.1340),
}
CATEGORIES = ["Bakeries", "Coffee & Tea", "Desserts", "Breakfast & Brunch", "Pizza", "Sandwiches", "Korean",
              "Japanese", "Chinese", "Italian", "Mexican", "Vegan", "Ice Cream & Frozen Yogurt", "Bars"]
CATEGORY_KO = {"Bakeries": "빵집", "Coffee & Tea": "카페", "Desserts": "디저트", "Breakfast & Brunch": "브런치",
               "Pizza": "피자", "Sandwiches": "샌드위치", "Korean": "한식", "Japanese": "일식", "Chinese": "중식",
               "Italian": "이탈리안", "Mexican": "멕시칸", "Vegan": "비건", "Ice Cream & Frozen Yogurt": "아이스크림",
               "Bars": "바", "": "전체 음식점"}
GENERIC = {"Restaurants", "Food", "Nightlife", "Food Delivery Services", "Event Planning & Services", "Caterers"}

st.set_page_config(page_title="TasteRoute", page_icon="🥐", layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@500&display=swap');
  html, body, [class*="css"], .stMarkdown, .stText {{ font-family: Inter, -apple-system, "Apple SD Gothic Neo", sans-serif; }}
  #MainMenu, footer {{ visibility: hidden; }}
  header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
  .block-container {{ padding-top: 2.4rem; padding-bottom: 2rem; max-width: 1480px; }}
  .tr-head {{ display: flex; align-items: baseline; gap: 14px; margin-bottom: 4px; }}
  .tr-logo {{ font-size: 26px; font-weight: 700; letter-spacing: -0.02em; color: {INK}; }}
  .tr-logo span {{ color: {ACCENT}; }}
  .tr-sub {{ color: {MUTED}; font-size: 14px; }}
  .tiles {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; margin: 6px 0 16px; }}
  .tile {{ background: #fff; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 16px;
           box-shadow: 0 1px 2px rgba(16,24,40,.06), 0 4px 12px rgba(16,24,40,.04); }}
  .tile.accent {{ background: {ACCENT_TINT}; border-color: #f6c7b3; }}
  .tile .lab {{ font-size: 11.5px; letter-spacing: .04em; text-transform: uppercase; color: {MUTED}; font-weight: 600; }}
  .tile .val {{ font-size: 28px; font-weight: 600; color: {INK}; font-variant-numeric: tabular-nums; margin-top: 2px; }}
  .tile .val small {{ font-size: 15px; color: {MUTED}; font-weight: 500; margin-left: 2px; }}
  .tile .note {{ font-size: 12px; color: {MUTED}; margin-top: 2px; }}
  .card-title {{ font-weight: 700; font-size: 15px; color: {INK}; margin-bottom: 0; }}
  .card-take {{ color: {MUTED}; font-size: 13px; margin: 2px 0 6px; }}
  table.tr {{ width: 100%; border-collapse: collapse; font-size: 13.5px; background: #fff; }}
  table.tr th {{ text-align: right; color: {MUTED}; font-weight: 600; font-size: 12px; padding: 8px 10px;
                 border-bottom: 1px solid {BORDER}; position: sticky; top: 0; background: #fff; }}
  table.tr th:first-child, table.tr td:first-child {{ text-align: left; }}
  table.tr td {{ text-align: right; padding: 8px 10px; border-bottom: 1px solid #f0f1f3; font-variant-numeric: tabular-nums; }}
  table.tr td.best {{ font-weight: 700; background: {ACCENT_TINT}; color: #b8471c; }}
  section[data-testid="stSidebar"] .stMarkdown h3 {{ font-size: 13px; text-transform: uppercase; letter-spacing: .05em; color: {MUTED}; }}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------- engine (cached)
@st.cache_resource(show_spinner="데이터와 모델을 준비하는 중... (처음 한 번, 약 20초)")
def load_engine():
    """Data, the CBF rating model, the candidate filter, and the optional extras.

    Optional parts appear when their files exist: the learned visit model
    (results/learned_distance.json), BPR (data/processed/bpr_model.npz, from
    experiments/bpr.py) and place attributes / opening hours
    (data/processed/item_extras.pkl, from python -m tasteroute.item_extras).
    """
    r = load_ratings()
    items = load_items()
    n_users, n_items = n_users_items(r)
    X, _ = load_item_features()
    train = get_split(r, "train")
    cbf = CBF(n_users, n_items, X, lam=30).fit(train)
    filt = CandidateFilter(items, train, seen=pd.concat([train, get_split(r, "val")]))
    anchors = user_anchors(train, items, n_users)
    feats = VisitFeatures(items, train, anchors, filt.popularity)
    learned = None
    if (RES / "learned_distance.json").exists():
        m = json.loads((RES / "learned_distance.json").read_text())["model"]
        learned = (LogisticRegression.from_dict(m["params"]), feats, m["columns"])
    bpr_path = ROOT / "data" / "processed" / "bpr_model.npz"
    bpr = BPR.load(bpr_path) if bpr_path.exists() else None
    extras, features = None, X
    if (ROOT / "data" / "processed" / "item_extras.pkl").exists():
        extras = load_item_extras()
        A, _ = attribute_matrix(extras, n_items)
        features = combine_features(X, A, 0.5)  # MMR similarity: categories + attributes
    n_train = train.groupby("u").size()
    # example users: the most active users whose anchor is in the city (plus the RQ3 example)
    active = n_train[n_train >= 40].index.values
    examples = [1458] + [int(u) for u in active[:400:40] if u != 1458]
    pop_rank = filt.popularity[filt.eligible]
    return {"items": items, "n_items": n_items, "cbf": cbf, "filt": filt, "anchors": anchors, "learned": learned,
            "bpr": bpr, "extras": extras, "features": features, "n_train": n_train, "examples": examples,
            "eligible_pop": np.sort(pop_rank)}


@st.cache_data(show_spinner=False)
def run_plan(user, start, category, n, method, tau, walking, days=1, diversity=0.0, when=None, cache_only=False):
    """plan_days for these choices (cached per combination).

    when: (first weekday, start minute) to keep only places open from the start
        for two hours on every day of the trip; None ignores opening hours.
    cache_only: use only street networks already on disk (after a failed download).
    """
    eng = load_engine()
    items, cbf = eng["items"], eng["cbf"]
    stars = np.clip(cbf.bias.mu + cbf.bias.b_i, 1, 5) if user is None else cbf.score_all(user)
    mask = np.ones(eng["n_items"], dtype=bool)
    if category:
        mask &= items.categories.str.contains(category, regex=False, na=False).values
    if when is not None and eng["extras"] is not None:
        first, minute = when
        for d in range(days):
            mask &= open_mask(eng["extras"], eng["n_items"], DAYS[(DAYS.index(first) + d) % 7], minute, 120)

    def streets(lats, lons):
        # one quick try: a busy OpenStreetMap server should not freeze the app
        return RoadNetwork.around(lats, lons, tries=0 if cache_only else 1, timeout=25)

    kw = {"user": user, "mask": mask, "tau": tau, "diversity": diversity, "features": eng["features"]}
    if method == "bpr":
        # BPR scores rank places directly: no popularity blend (gamma = 0), then distance
        pred = eng["bpr"].score_all(user)
        plan = plan_days(pred, eng["filt"], items, start, days, n, roads=streets if walking else None, gamma=0, **kw)
    else:
        visit = eng["learned"] if method == "learned" else None
        plan = plan_days(stars, eng["filt"], items, start, days, n, roads=streets if walking else None,
                         visit=visit, **kw)
    plan["stars"] = {int(i): float(stars[i]) for i in plan["ranked"]}
    return plan


# ---------------------------------------------------------------- header
st.markdown('<div class="tr-head"><div class="tr-logo">Taste<span>Route</span></div>'
            '<div class="tr-sub">취향 기반 장소 추천과 실제 길을 따라가는 하루 동선 · Philadelphia, Yelp Open Dataset</div></div>',
            unsafe_allow_html=True)

eng = load_engine()
tab_plan, tab_results, tab_data = st.tabs(["🗺️ 코스 짜기", "📊 실험 결과", "🔎 데이터"])

# ---------------------------------------------------------------- sidebar controls
METHODS = {"학습 모델 (로지스틱 회귀)": "learned", "BPR 순위 학습 + 거리": "bpr", "고정 규칙 (거리 감쇠 τ)": "fixed"}
DAYS_KO = dict(zip(DAYS, ["월", "화", "수", "목", "금", "토", "일"]))

with st.sidebar:
    st.markdown("### 누구를 위한 코스?")
    user_labels = {"새 방문자 (취향 정보 없음)": None}
    user_labels.update({f"사용자 {u} · 평점 {int(eng['n_train'][u])}개": u for u in eng["examples"]})
    user = user_labels[st.selectbox("사용자", list(user_labels), label_visibility="collapsed")]

    st.markdown("### 어디서 출발?")
    start_opts = list(STARTS)
    if user is not None:
        start_opts = ["이 사용자의 생활권 중심"] + start_opts
    start_opts.append("좌표 직접 입력")
    start_name = st.selectbox("출발지", start_opts, label_visibility="collapsed")
    if start_name == "이 사용자의 생활권 중심":
        start = tuple(float(x) for x in eng["anchors"][user])
    elif start_name == "좌표 직접 입력":
        c1, c2 = st.columns(2)
        # limited to greater Philadelphia: the data covers 15 km around City Hall
        start = (c1.number_input("위도", value=39.9526, min_value=39.80, max_value=40.10, format="%.4f"),
                 c2.number_input("경도", value=-75.1636, min_value=-75.40, max_value=-74.90, format="%.4f"))
        start_name = f"{start[0]:.4f}, {start[1]:.4f}"
    else:
        start = STARTS[start_name]

    st.markdown("### 무엇을?")
    category = st.selectbox("카테고리", CATEGORIES + [""], format_func=lambda c: f"{CATEGORY_KO[c]}" + (f" · {c}" if c else ""),
                            label_visibility="collapsed")
    c1, c2 = st.columns(2)
    n = c1.slider("하루에 갈 곳", 3, 8, 6)
    days = c2.slider("며칠", 1, 3, 1, help="2일 이상이면 추천 장소를 k-means로 동네별로 묶어 하루에 한 동네씩 돌아요.")

    st.markdown("### 언제?")
    has_hours = eng["extras"] is not None
    c1, c2 = st.columns(2)
    weekday = c1.selectbox("요일", DAYS, index=5, format_func=lambda d: DAYS_KO[d] + "요일")
    start_time = c2.time_input("출발 시각", value=dt.time(11, 0), step=900)
    stay = st.slider("가게마다 머무는 시간 (분)", 10, 60, 25, 5)
    use_hours = st.toggle("영업 중인 곳만", value=has_hours, disabled=not has_hours,
                          help="출발 후 2시간 동안 영업하는 곳만 골라요. 영업시간 정보가 없는 곳은 포함해요."
                          if has_hours else "python -m tasteroute.item_extras 를 먼저 실행하면 켤 수 있어요.")

    st.markdown("### 어떻게 고를까?")
    options = [k for k, v in METHODS.items()
               if not (v == "learned" and eng["learned"] is None) and not (v == "bpr" and eng["bpr"] is None)]
    method = METHODS[st.radio("순위 방식", options, label_visibility="collapsed")]
    tau = 2.0
    if method in ("fixed", "bpr"):
        tau = st.slider("τ (km): 클수록 먼 곳도 추천", 0.5, 5.0, 2.0, 0.5)
    diversity = st.slider("다양성", 0.0, 0.8, 0.0, 0.1,
                          help="0이면 점수 순서 그대로. 올릴수록 서로 다른 종류의 가게를 섞어요 (MMR).")
    walking = st.toggle("실제 도보 경로 (OpenStreetMap)", value=True,
                        help="끄면 장소 사이를 직선으로 잇고, 거리도 직선거리로 계산해요.")

# ---------------------------------------------------------------- tab 1: plan
DAY_COLORS = ["#eb6834", "#2a78d6", "#1baf7a"]


def hhmm(minute):
    """'13:05' for a minute count from midnight (wraps past 24 h)."""
    minute = int(round(minute)) % (24 * 60)
    return f"{minute // 60:02d}:{minute % 60:02d}"


with tab_plan:
    # after one failed street download, stay on straight lines until the user retries
    if walking and st.session_state.get("osm_failed"):
        c1, c2 = st.columns([5, 1])
        c1.info("OpenStreetMap 서버가 응답하지 않아, 받아 둔 지역은 도보 경로로, 나머지는 직선 경로로 보여드려요.")
        if c2.button("도보 경로 다시 시도"):
            st.session_state.osm_failed = False
            st.rerun()
    start_min = start_time.hour * 60 + start_time.minute
    when = (weekday, start_min) if use_hours else None
    args = (user, start, category, n, method, tau)
    try:
        with st.spinner("코스를 짜는 중... (새 지역의 도로망은 처음 한 번 받아요)"):
            plan = run_plan(*args, walking, days, diversity, when, cache_only=bool(st.session_state.get("osm_failed")))
    except RuntimeError:  # street download failed, or not cached while the server is down
        if not st.session_state.get("osm_failed"):
            st.session_state.osm_failed = True
            st.warning("OpenStreetMap에서 이 지역 도로망을 받지 못해 직선 경로로 보여드려요.")
        plan = run_plan(*args, False, days, diversity, when)
        walking = False

    items, extras = eng["items"], eng["extras"]
    eligible_pop = eng["eligible_pop"]
    rank_of = {i: k + 1 for k, i in enumerate(plan["ranked"])}
    view_days, closed = [], 0
    for d, day in enumerate(plan["days"]):
        wd = DAYS[(DAYS.index(weekday) + d) % 7]
        t, stops = start_min, []
        for i, leg in zip(day["places"], day["legs_km"]):
            row = items.iloc[i]
            t += leg / WALK_KMH * 60
            arrive = t
            t += stay
            hours = extras.hours.get(i) if extras is not None else None
            status = "unknown"
            if isinstance(hours, dict) and hours:
                # minutes past midnight roll over to the next weekday inside is_open_at
                ok = is_open_at(hours, wd, arrive) and is_open_at(hours, wd, arrive + stay)
                status = "open" if ok else "closed"
                closed += status == "closed"
            d0 = float(haversine(start[0], start[1], row.lat, row.lon))
            top = 100 * (1 - np.searchsorted(eligible_pop, eng["filt"].popularity[i]) / len(eligible_pop))
            tags = [c.strip() for c in str(row.categories).split(",") if c.strip() not in GENERIC][:3]
            stops.append({
                "name": row["name"], "lat": float(row.lat), "lon": float(row.lon), "stars": float(row.stars),
                "reviews": int(row.review_count), "pred": plan["stars"][i], "rank": rank_of[i], "tags": tags,
                "leg_km": leg, "leg_min": int(round(leg / WALK_KMH * 60)), "arrive": hhmm(arrive),
                "hours": hours_today(hours, wd) if hours else None, "status": status,
                "why": f"출발지에서 직선 {d0:.1f}km · 인기 상위 {max(top, 0.1):.1f}%",
            })
        view_days.append({"label": f"{d + 1}일차 · {DAYS_KO[wd]}요일" if len(plan["days"]) > 1 else f"{DAYS_KO[wd]}요일",
                          "color": DAY_COLORS[d % 3], "stops": stops, "path": day["path"], "rec_path": day["rec_path"],
                          "end": hhmm(t), "km": day["route_km"]})

    total_km = sum(x["route_km"] for x in plan["days"])
    rec_km = sum(x["rec_order_km"] for x in plan["days"])
    saved = rec_km - total_km
    walk_min = total_km / WALK_KMH * 60
    n_places = sum(len(x["places"]) for x in plan["days"])
    dist_kind = "도보" if walking else "직선"
    method_note = {"learned": "학습 모델", "bpr": f"BPR + 거리 τ={tau:g}km", "fixed": f"고정 규칙 τ={tau:g}km"}[method]
    hours_note = f"영업 중인 곳만 · {closed}곳 도착 시 닫힘" if use_hours else "영업시간 무시"
    st.markdown(f"""
    <div class="tiles">
      <div class="tile"><div class="lab">총 {dist_kind} 거리</div><div class="val">{total_km:.1f}<small>km</small></div>
        <div class="note">직선으로 재면 {sum(x['straight_km'] for x in plan['days']):.1f}km</div></div>
      <div class="tile"><div class="lab">{"걷는 시간" if walking else "직선 환산 시간"} + 머무는 시간</div><div class="val">{walk_min:.0f}<small>분</small> + {n_places * stay}<small>분</small></div>
        <div class="note">{len(plan['days'])}일 · 시속 {WALK_KMH:g}km · {hours_note}</div></div>
      <div class="tile"><div class="lab">방문 장소</div><div class="val">{n_places}<small>곳</small></div>
        <div class="note">{CATEGORY_KO[category]} · {method_note}{f" · 다양성 {diversity:g}" if diversity else ""}</div></div>
      <div class="tile accent"><div class="lab">추천 순서보다</div><div class="val">▼ {saved:.1f}<small>km</small></div>
        <div class="note">Held-Karp 최적 순서 · {saved / max(rec_km, 1e-9) * 100:.0f}% 절약</div></div>
    </div>""", unsafe_allow_html=True)

    pts = [p for x in view_days for p in x["path"]] + [start]
    lats, lons = [p[0] for p in pts], [p[1] for p in pts]
    data = {
        "title": f"{'오늘의' if days == 1 else f'{days}일'} {CATEGORY_KO[category]} 코스",
        "subtitle": f"{start_name}에서 {hhmm(start_min)} 출발 · 카드를 누르면 지도가 그곳으로 가요",
        "walking": walking, "start": {"name": start_name, "lat": start[0], "lon": start[1]}, "days": view_days,
        "bounds": [[min(lons), min(lats)], [max(lons), max(lats)]],
    }
    # "</" escaped as "<\\/" so a place name can never close the <script> holding the data
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    components.html(VIEW.replace("/*__DATA__*/null", payload), height=680)

    with st.expander("이 코스는 어떻게 만들어졌나요?"):
        rank_text = {
            "learned": "학습된 로지스틱 회귀가 거리, 개인 이동 반경, 예상 별점, 인기도로 방문 가능성을 점수 매겨요.",
            "bpr": f"BPR(간 곳과 안 간 곳의 순위를 직접 학습한 모델)의 점수 × e^(−거리/{tau:g}km)로 골라요.",
            "fixed": f"점수 × e^(−거리/{tau:g}km)로 가까운 곳을 우대해요.",
        }[method]
        st.markdown(f"""
1. **예상 별점**: {"새 방문자라 전체 평균 + 장소 편향(μ + b_i)" if user is None else f"사용자 {user}의 취향(CBF: bias + 카테고리 취향)"}으로 모든 장소의 별점을 예측해요. 카드의 '예상 ★'이에요.
2. **후보**: 평점이 20개 이상이고 이 사람이 아직 안 가 본 {CATEGORY_KO[category]}{"" if not use_hours else f" 중 {DAYS_KO[weekday]}요일 {hhmm(start_min)}부터 2시간 영업하는 곳"}만 남기고 상위 200곳을 후보로 뽑아요.
3. **순위**: {rank_text}{f" 다양성 {diversity:g}: 이미 고른 곳과 비슷한 가게는 점수를 깎아요(MMR)." if diversity else ""} 상위 {n * days}곳을 골라요.
4. **{"동네 묶기" if days > 1 else "동선"}**: {f"k-means로 {days}개 동네로 묶어 하루에 한 동네씩, " if days > 1 else ""}{"OpenStreetMap 보행 도로망에서 Dijkstra로 구한 도보 거리" if walking else "직선거리"}로 Held-Karp가 가장 짧은 방문 순서를 찾아요. 가게마다 {stay}분 머문다고 보고 도착 시각과 영업 상태를 계산해요.
""")

# ---------------------------------------------------------------- tab 2: results
def chart_card(title, takeaway, chart):
    """A bordered card with a title, a one-line takeaway and an Altair chart."""
    with st.container(border=True):
        st.markdown(f'<div class="card-title">{title}</div><div class="card-take">{takeaway}</div>', unsafe_allow_html=True)
        st.altair_chart(chart.properties(height=260).configure_axis(
            labelFont="Inter", titleFont="Inter", labelColor=MUTED, titleColor=MUTED, gridColor="#eef0f3",
            domainColor=BORDER, tickColor=BORDER).configure_view(strokeWidth=0), width="stretch")


def load_json(name):
    """results/<name>.json, or None when that experiment has not been run."""
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


with tab_results:
    mc, rk, rt, ld = (load_json(x) for x in ("model_compare", "ranking", "route", "learned_distance"))
    if not all((mc, rk, rt)):
        st.info("results/가 비어 있어요. `python experiments/run_all.py`로 실험을 먼저 돌려 주세요.")
    else:
        warm = mc["warm"]
        best_rmse = min(warm.items(), key=lambda kv: kv[1]["rmse"])
        tau2 = next(x for x in rk["rerank"] if x["decay"] == "exp" and x["tau"] == 2)
        hyb = rk["models"]["Hybrid"]
        s = rt["summary"]
        st.markdown(f"""
        <div class="tiles">
          <div class="tile"><div class="lab">최저 test RMSE</div><div class="val">{best_rmse[1]['rmse']:.3f}</div>
            <div class="note">{best_rmse[0]} · 전체 평균 {warm['Global mean']['rmse']:.3f}</div></div>
          <div class="tile"><div class="lab">거리 재순위화 (RQ1)</div><div class="val">{hyb['mean_dist_km']:.1f}→{tau2['mean_dist_km']:.1f}<small>km</small></div>
            <div class="note">NDCG@10 {hyb['NDCG@10']:.4f} → {tau2['NDCG@10']:.4f}</div></div>
          <div class="tile"><div class="lab">NN + 2-opt 최적해 (RQ3)</div><div class="val">{s['nn_2opt_optimal_share'] * 100:.0f}<small>%</small></div>
            <div class="note">평균 차이 {s['nn_2opt_mean_gap'] * 100:.1f}%</div></div>
          <div class="tile accent"><div class="lab">추천 순서 그대로면</div><div class="val">+{s['rec_order_median_extra'] * 100:.0f}<small>%</small></div>
            <div class="note">최적 경로보다 더 걷는 거리 (중앙값)</div></div>
        </div>""", unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            df = pd.DataFrame([{"model": k, "rmse": v["rmse"], "lo": v["rmse_ci95"][0], "hi": v["rmse_ci95"][1]}
                               for k, v in warm.items()]).sort_values("rmse")
            df["ours"] = df.model.isin(["Hybrid", "CBF"])
            # differences are tiny, so dots with 95% intervals instead of bars (bars must start at 0)
            base = alt.Chart(df).encode(y=alt.Y("model:N", sort=list(df.model), title=None,
                                                axis=alt.Axis(labelFontSize=12, labelColor=INK)))
            err = base.mark_rule(strokeWidth=2.5, color=GREY).encode(
                x=alt.X("lo:Q", title="test RMSE, 95% 신뢰구간 (낮을수록 좋음)", scale=alt.Scale(zero=False)), x2="hi:Q")
            dots = base.mark_circle(size=110, opacity=1).encode(
                x="rmse:Q", color=alt.condition("datum.ours", alt.value(ACCENT), alt.value("#8b93a1")),
                tooltip=["model", alt.Tooltip("rmse:Q", format=".4f"), alt.Tooltip("lo:Q", format=".4f", title="CI 하한"),
                         alt.Tooltip("hi:Q", format=".4f", title="CI 상한")])
            labels = base.mark_text(align="left", dx=10, fontSize=11, color=MUTED).encode(
                x="hi:Q", text=alt.Text("rmse:Q", format=".3f"))
            chart_card("별점 예측: 모델 비교", "Bias, CBF, Hybrid가 통계적으로 같은 1.051. 평점이 희소해서 bias 외 구조를 배우기 어려워요.",
                       err + dots + labels)
        with c2:
            rr = pd.DataFrame(rk["rerank"])
            rr = rr[rr.decay.isin(["exp", "gaussian", "power", "cutoff"])]
            pts = alt.Chart(rr).mark_line(point=alt.OverlayMarkDef(size=40), strokeWidth=2).encode(
                x=alt.X("mean_dist_km:Q", title="추천 목록 평균 거리 (km)"),
                y=alt.Y("NDCG@10:Q", title="NDCG@10"),
                color=alt.Color("decay:N", scale=alt.Scale(domain=["exp", "gaussian", "cutoff", "power"], range=SERIES),
                                legend=alt.Legend(title="감쇠 모양", orient="top", direction="horizontal")),
                order="tau:Q", tooltip=["decay", "tau", alt.Tooltip("mean_dist_km:Q", format=".1f"),
                                        alt.Tooltip("NDCG@10:Q", format=".4f")])
            nod = alt.Chart(pd.DataFrame([{"d": hyb["mean_dist_km"], "n": hyb["NDCG@10"], "label": "거리 반영 없음"}])).encode(
                x="d:Q", y="n:Q")
            chart_card("RQ1 · 거리를 넣으면 더 가깝고 더 정확해요",
                       "점 하나가 거리 감쇠 τ 하나. 사용자는 실제로 가까운 곳을 다녀서, 거리를 넣어도 정확도가 오히려 올라요.",
                       pts + nod.mark_point(color=INK, size=80, filled=True) + nod.mark_text(dx=8, align="left", color=INK).encode(text="label"))

        c3, c4 = st.columns(2, gap="medium")
        with c3:
            if ld:
                lt = pd.DataFrame([{"method": k, "NDCG@10": v["NDCG@10"], "coverage": v["coverage"] * 100}
                                   for k, v in ld["test"].items()])
                names = {"popularity + fixed distance": "인기순 + 규칙", "blend + fixed distance (top 50)": "규칙 · 후보 50",
                         "learned (no personal radius)": "학습 · 반경 없음", "learned (with personal radius)": "학습 + 개인 반경"}
                lt["label"] = lt.method.map(lambda m: names.get(m, "규칙 · 후보 200"))
                lt["ours"] = lt.method.str.startswith("learned")
                base = alt.Chart(lt).encode(y=alt.Y("label:N", sort=list(lt.label), title=None,
                                                    axis=alt.Axis(labelFontSize=12, labelColor=INK, labelLimit=160)))
                bars = base.mark_bar(cornerRadiusEnd=4, height=18).encode(
                    x=alt.X("NDCG@10:Q", title="NDCG@10 (test 3,000명)", scale=alt.Scale(domain=[0, 0.045])),
                    color=alt.condition("datum.ours", alt.value(ACCENT), alt.value(GREY)),
                    tooltip=["label", alt.Tooltip("NDCG@10:Q", format=".4f"), alt.Tooltip("coverage:Q", format=".2f", title="커버리지 %")])
                vals = base.mark_text(align="left", dx=6, fontSize=11, color=INK).encode(
                    x="NDCG@10:Q", text=alt.Text("NDCG@10:Q", format=".4f"))
                ch = bars + vals
                m = ld["model"]
                chart_card("거리 민감도 학습 (로지스틱 회귀)",
                           f"튜닝한 규칙과 유의한 차이는 없고(p = 0.36) 커버리지는 더 넓어요. 1km 더 멀면 인기도 {m['popularity_factor_per_km_at_1km']:.2f}배여야 상쇄.",
                           ch)
        with c4:
            pu = pd.DataFrame(rt["per_user"])
            pu["extra"] = (pu.rec_order / pu.optimal - 1) * 100
            ch = alt.Chart(pu[pu.extra < 150]).mark_bar(color=ACCENT, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
                x=alt.X("extra:Q", bin=alt.Bin(step=10), title="추천 순서로 돌 때 더 걷는 거리 (%)"),
                y=alt.Y("count():Q", title="사용자 수"), tooltip=[alt.Tooltip("count():Q", title="사용자 수")])
            chart_card("RQ3 · 순서만 바꿔도 덜 걸어요",
                       f"6곳을 추천 순서대로 돌면 최적보다 중앙값 {s['rec_order_median_extra'] * 100:.0f}% 더 걸어요 (test 3,000명).",
                       ch)

        with st.container(border=True):
            st.markdown('<div class="card-title">모델 비교 표</div><div class="card-take">warm test 평점 69,847개 · '
                        '95% 신뢰구간은 사용자 단위 bootstrap · p는 bias 대비 paired t-test · cold-start는 학습 평점 1/3/5개 사용자</div>',
                        unsafe_allow_html=True)
            rows = []
            for k, v in warm.items():
                cold = mc["cold"][k]
                rows.append([k, v["rmse"], v["mae"], v["rmse_ci95"], v.get("vs_bias_p"), cold["1"], cold["3"], cold["5"]])
            best = [min(r[c] for r in rows) for c in (1, 2, 5, 6, 7)]
            html = ["<table class='tr'><tr><th>모델</th><th>RMSE</th><th>MAE</th><th>95% CI</th><th>p vs bias</th>"
                    "<th>cold m=1</th><th>cold m=3</th><th>cold m=5</th></tr>"]
            for r in rows:
                cell = lambda v, b: f"<td class='best'>{v:.4f}</td>" if np.isclose(v, b) else f"<td>{v:.4f}</td>"  # noqa: E731
                p = "-" if r[4] is None else f"{r[4]:.3g}"
                html.append(f"<tr><td>{r[0]}</td>{cell(r[1], best[0])}{cell(r[2], best[1])}"
                            f"<td>[{r[3][0]:.3f}, {r[3][1]:.3f}]</td><td>{p}</td>"
                            f"{cell(r[5], best[2])}{cell(r[6], best[3])}{cell(r[7], best[4])}</tr>")
            st.markdown("".join(html) + "</table>", unsafe_allow_html=True)

        with st.expander("발표용 그림 전체 보기 (figures/)"):
            figs = sorted((ROOT / "figures").glob("*.png"))
            cols = st.columns(2)
            for k, f in enumerate(figs):
                cols[k % 2].image(str(f), caption=f.stem, width="stretch")

# ---------------------------------------------------------------- tab 3: data
EDA = [
    ("P1_preprocessing_pipeline", "전처리 단계별 데이터 규모"), ("E6_spatial_density", "필라델피아 음식점 분포"),
    ("E8_user_activity_radius", "사용자 활동 반경: 대부분 가까운 곳만 다녀요"), ("E4_popularity_lorenz", "인기도 쏠림 (Lorenz)"),
    ("E3_activity_long_tail", "사용자 활동의 긴 꼬리"), ("E1_rating_distribution", "별점 분포"),
    ("E9_category_frequency", "카테고리 빈도"), ("E12_popularity_vs_quality", "인기도와 평균 별점"),
]
with tab_data:
    st.markdown("""
    <div class="tiles">
      <div class="tile"><div class="lab">원본 리뷰</div><div class="val">6.99<small>M</small></div><div class="note">Yelp Open Dataset 전체</div></div>
      <div class="tile"><div class="lab">시청 15km 음식점</div><div class="val">809<small>K</small></div><div class="note">중복 제거 후 평점</div></div>
      <div class="tile"><div class="lab">5-core 작업 데이터</div><div class="val">490,539</div><div class="note">사용자 31,743 · 장소 7,693</div></div>
      <div class="tile accent"><div class="lab">밀도</div><div class="val">0.20<small>%</small></div><div class="note">사용자당 학습 평점 중앙값 6개</div></div>
    </div>""", unsafe_allow_html=True)
    fig_dir = ROOT / "data_analysis_preprocessing" / "figures"
    cols = st.columns(2, gap="medium")
    for k, (stem, caption) in enumerate(EDA):
        f = fig_dir / f"{stem}.png"
        if f.exists():
            with cols[k % 2].container(border=True):
                st.markdown(f'<div class="card-title">{caption}</div>', unsafe_allow_html=True)
                st.image(str(f), width="stretch")
