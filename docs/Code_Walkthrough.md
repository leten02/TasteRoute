# 코드 읽기 안내 (Code Walkthrough)

TasteRoute 코드 전체를 처음 보는 사람이 **어디서부터, 무엇을, 왜** 읽으면 되는지 정리한 문서다. 수식과 코드 위치를 짝지어 두어서 발표 질문에 바로 답할 수 있게 했다.

## 1. 큰 그림

```
data/processed (전처리 결과)
      │  data.py: 평점, 가게, 카테고리 행렬, split
      ▼
① 별점 예측 (models/)            "이 사람이 이 가게에 별점 몇 점?"
   baseline ─┬─ mf (SGD)
             ├─ item_knn (이웃)
             ├─ cbf (내용, ridge) ── hybrid (적응형 섞기)
             └─ content_knn (내용 이웃)
   순위 모델: bpr (간 곳 vs 안 간 곳)
      │  score_all(u) → 모든 가게 점수
      ▼
② 후보와 재순위화
   candidates.py  영업 중·평점 20개 이상·안 가 본 곳, 인기도 blend, 상위 200
   rerank.py      거리 감쇠 e^(−d/τ)  또는  models/visit.py 학습 모델,  다양성 MMR
   geo.py         haversine, 사용자 기준점
      ▼
③ 동선
   route.py       Held-Karp(정확) / 완전탐색 / NN + 2-opt
   roads.py       OpenStreetMap 보행 도로망, Dijkstra 도보 거리
   cluster.py     여러 날 코스: 용량 제한 k-means
      ▼
pipeline.py  recommend → route → plan_day / plan_days
      ▼
app/dashboard.py (Streamlit)   demo/bakery_tour.py (folium)
```

모든 별점 모델은 같은 인터페이스를 쓴다: `fit(train)`, `predict(u, i)`, `score_all(u)`. 그래서 실험 스크립트와 파이프라인은 어떤 모델이든 바꿔 끼울 수 있다. BPR은 순위 모델이라 `fit`과 `score_all`만 있다.

## 2. 데이터 (`data.py`)

| 함수 | 역할 |
| --- | --- |
| `load_ratings()` | 평점 49만 개: 사용자 `u`, 가게 `i`(0부터 번호), 별점, 날짜, `split`(train/val/test/dropped), `cold_m` |
| `load_items()` | 가게 7,693곳: 이름, 좌표, 별점, 리뷰 수, 영업 여부, 카테고리, 가격 |
| `load_item_features()` | 가게 × 카테고리 188개 TF-IDF 희소 행렬 (CBF 특성) |
| `get_split(r, name, warm_only)` | 한 split만, `warm_only`면 cold-start 사용자 제외 |

**분할 규칙**: 사용자마다 평점을 약 70/15/15로 나눈다. validation과 test는 최소 1개씩이다. cold-start 사용자 1,306명은 train에 평점 1·3·5개만 남긴다(`cold_m`).

**지켜야 할 원칙**: 하이퍼파라미터는 **validation**에서만 고르고, **test**는 마지막에 한 번만 잰다. 모든 실험 스크립트가 이 순서를 따른다.

## 3. 별점 예측 모델 (`models/`)

### 3.1 Bias (`baseline.py`), 모든 모델의 바닥

```
r̂_ui = μ + b_u + b_i
```
- μ: 전체 평균. b_u: 이 사람이 남보다 후한 정도. b_i: 이 가게가 남보다 좋은 정도.
- **학습**: 교대 최소제곱이다. b_u를 고정하고 b_i를 닫힌 해로, b_i를 고정하고 b_u를 닫힌 해로 10번 반복한다.
  - `b_i = Σ(r − μ − b_u) / (λ + n_i)`
  - 분모의 λ가 L2 정규화다. 평점이 적은 가게는 0 쪽으로 당겨진다.
- `np.bincount(weights=...)`로 사용자·가게별 합을 한 번에 구한다 (반복문 없음).
- **결과**: test RMSE 1.051. 이 프로젝트에서 가장 좋은 별점 모델과 같은 수준이다.

### 3.2 MF (`mf.py`), 경사하강법

```
r̂_ui = μ + b_u + b_i + p_u · q_i
e = r − r̂
b_u ← b_u + η(e − λ b_u),   p_u ← p_u + η(e q_i − λ p_u),   q_i ← q_i + η(e p_u − λ q_i)
```
- 평점 하나씩 갱신하는 SGD이고, 매 epoch마다 순서를 섞는다.
- `p_u`와 `q_i`는 **갱신 전 값**으로 서로를 갱신한다 (`dp`를 먼저 계산하는 이유).
- **조기 종료**: validation RMSE가 3 epoch 동안 나아지지 않으면 멈추고, 가장 좋았던 epoch의 파라미터로 되돌린다.
- **결과**: f = 10, λ = 0.1, 15 epoch에서 test 1.052로, bias보다 근소하게 나쁘다(p = 0.0015). 사용자당 평점 중앙값이 6개라 잠재 요인을 배울 데이터가 부족하다.

### 3.3 Item-kNN (`item_knn.py`), 메모리 기반 협업 필터링

```
r̂_ui = b_ui + Σ_{j∈N_k(i;u)} s'_ij (r_uj − b_uj) / Σ |s'_ij|,     s'_ij = n_ij/(n_ij+β) · s_ij
```
- 유사도 3종(cosine, adjusted, Pearson)은 중심화 방식만 다르다.
- 가게 × 가게 유사도 7,693²을 희소 행렬 곱 한 번(`X @ X.T`)으로 구한다. float32라 약 237MB다.
- `np.argpartition`으로 상위 k 이웃만 고른다 (전체 정렬 없음).
- 평점이 k개보다 적은 사용자는 가진 평점을 전부 쓴다. 분모가 0이면 bias로 대체한다. 가게 자신은 이웃에서 뺀다.
- **결과**: test 1.121로 bias보다 나쁘다. 이웃 후보가 너무 적기 때문이다.
- **버그 이야기**: 사용자마다 정렬 배열을 새로 만들던 코드가 약 160GB를 잡아 컴퓨터가 멈췄다. 정렬을 한 번만 하도록 고쳤다 (`fix(knn)` 커밋).

### 3.4 CBF (`cbf.py`), 내용 기반 필터링

```
θ_u = (X_uᵀX_u + λI)⁻¹ X_uᵀ y_u,    y_u = r − b_ui (bias가 설명하고 남은 것)
```
- 사용자마다 카테고리 특성 위에서 ridge 회귀를 한다.
- 평점 수 n_u가 특성 수 d보다 적으면 같은 해를 `X_uᵀ(X_uX_uᵀ + λI)⁻¹ y_u`로 구한다. d×d 대신 n_u×n_u 연립방정식이라 훨씬 싸다.
- 잔차를 bias 기준으로 잡는 이유: 사용자 평균 기준이면 평점 1개짜리 사용자는 잔차가 0이 되어 아무것도 배우지 못한다.

### 3.5 Hybrid (`hybrid.py`), 적응형 섞기

```
r̂ = α_u · CF + (1 − α_u) · CBF,     α_u = n_u / (n_u + κ)
```
- 평점이 많은 사용자일수록 협업 필터링(MF) 쪽 비중이 커진다.

### 3.6 내용 이웃 (`content_knn.py`), RQ2 재실험

```
r̂_ui = b_ui + Σ_j s_ij (r_uj − b_uj) / (Σ_j |s_ij| + shrink)
```
- s_ij는 카테고리 + 가게 속성 42개 벡터의 cosine 유사도다.
- 사용자별로 학습할 것이 없어서, 평점 하나로도 닮은 가게에 정보가 전해진다.
- **결과**: validation이 shrink를 범위 끝(200)까지 키우는 쪽을 골랐다. 즉 내용 정보를 끄는 게 최선이었다 (RQ2의 답: 아니다).

### 3.7 BPR (`bpr.py`), 순위를 직접 학습

```
x_ui = b_i + p_u·q_i,    최대화 Σ ln σ(x_ui − x_uj) − (λ/2)‖θ‖²   (i: 간 곳, j: 안 간 곳)
```
- 미니배치 SGD를 쓴다. `np.add.at`은 한 배치에 같은 사용자가 여러 번 나와도 갱신을 합친다.
- 음성 샘플 j는 간 곳이면 다시 뽑는다.
- **결과**: 인기순과 차이가 없다(p = 0.61). 평점 21개 이상 사용자에게서만 앞섰다(보정 전 p = 0.03).

## 4. 평가 (`metrics.py`, `experiments/`)

| 지표 | 식 | 코드 |
| --- | --- | --- |
| RMSE, MAE | `√mean(e²)`, `mean(|e|)` | `rmse`, `mae` (형상이 다르거나 빈 입력이면 오류) |
| P@K, R@K | `|L∩Rel|/K`, `|L∩Rel|/|Rel|` | `precision_at_k`, `recall_at_k` |
| NDCG@K | `Σ rel/log₂(pos+1) / IDCG` | `ndcg_at_k` (같은 가게는 처음 한 번만 셈) |

- **관련 있음**: test에서 별점 4 이상 준 가게다.
- **통계**
  - 사용자 단위 bootstrap 95% 신뢰구간: 평점이 아니라 **사용자를** 다시 뽑는다. 같은 사람의 평점끼리는 독립이 아니기 때문이다.
  - paired t-test (`scipy.stats.ttest_rel`): 같은 사용자의 두 모델 오차를 짝지어 비교한다.

## 5. 후보와 재순위화

- `candidates.CandidateFilter`: 영업 중이고, train 평점이 20개 이상이고(평점 몇 개짜리 가게는 별점 모델이 과대평가함), 사용자가 train·val에서 안 간 곳만 남긴다.
- `blend_popularity`: `r̂ + γ·log(1+인기도)`로, 별점 모델에 "갈 만한가" 신호를 더한다. γ는 validation에서 5가 선택됐다.
- `rerank.distance_weight`: 감쇠 모양 4가지가 있다.
  - exp `e^(−d/τ)` (기본)
  - gaussian `e^(−d²/2τ²)`
  - power `1/(1+d/τ)`
  - cutoff (d ≤ τ면 1, 아니면 0)
- `models/visit.py`: 거리, 개인 이동 반경, 예상 별점, 인기도를 쓰는 **로지스틱 회귀**다. 경사하강법은 직접 구현했고 backtracking line search를 쓴다. 고정 규칙 τ를 데이터로 대체한다.
- `rerank.mmr`: `(1−λ)·관련도 − λ·(이미 고른 곳과의 최대 유사도)`로 하나씩 고른다. λ가 클수록 다양해진다.

## 6. 동선

### 6.1 Held-Karp (`route.py`), 정확한 최단 순서
```
C[S][j] = 출발점에서 시작해 집합 S를 한 번씩 방문하고 j에서 끝나는 최단 길이
C[{j}][j] = D[0][j],   C[S][j] = min_{k∈S, k≠j} C[S−{j}][k] + D[k][j]
```
- S는 **비트마스크**다 (정수의 각 비트가 장소 하나). 예를 들어 6곳이면 2⁶ = 64가지 집합이다.
- 시간은 O(N²·2^N)이다. N = 8에서 1ms이고, 12곳까지만 허용한다(`MAX_PLACES`).
- 마지막에 `parent`를 거꾸로 따라가 순서를 복원한다.
- **비교 대상**
  - 완전탐색: N!가지 순서를 다 본다. 정답 확인용이다.
  - NN + 2-opt: 가장 가까운 곳부터 간 뒤, 구간을 뒤집어 짧아지면 바꾼다. 82%의 경우 최적해와 같다.

### 6.2 실제 도보 거리 (`roads.py`)
- **도로망**: OSMnx로 보행 도로망을 받아 `data/osm/`에 캐시한다. 노드는 교차로와 꺾이는 점, 간선은 길 토막이다.
  - 캐시는 범위가 덮이면 재사용하고, 원자적으로 저장하며, 손상된 파일은 건너뛴다.
- **최단 거리**: `scipy.sparse.csgraph.dijkstra`로 장소 사이 최단 거리를 구한다.
  - `return_predecessors`로 경로 좌표를 복원한다.
  - 가장 가까운 도로 노드는 KD-tree로 찾는다.
- **실패 대응**: 서버가 응답하지 않으면 정해진 시간 안에 포기하고 직선거리로 대체한다.

### 6.3 여러 날 (`cluster.py`)
- k-means: Lloyd 알고리즘(배정 ↔ 중심 갱신)을 쓰고, k-means++로 초기화하며, 여러 번 재시작해 가장 좋은 결과를 고른다.
- **용량 제한**: 모든 (장소, 중심) 쌍을 거리순으로 정렬하고, 자리가 남은 동네에 배정한다. 그래서 하루에 정해진 수만큼만 들어간다 (11/1 대신 6/6).

## 7. 파이프라인 (`pipeline.py`)

```python
recommend(...)  # 별점 → 인기도 blend → 후보 200 → 거리 재순위화(규칙/학습) → (MMR) → 상위 n
route(...)      # Held-Karp로 순서, 도로망이 있으면 도보 거리와 실제 경로 좌표
plan_day(...)   # recommend + route
plan_days(...)  # 날짜 × n곳 추천 → 용량 제한 k-means → 날마다 route
```
반환값에는 방문 순서(`places`), 추천 순서(`ranked`), 경로 길이, 구간 거리, 그릴 좌표가 들어 있다.

## 8. 대시보드 (`app/`)

- `dashboard.py`
  - `load_engine()`: 데이터 로드와 CBF 학습을 한 번만 하고 캐시한다 (`st.cache_resource`).
  - `run_plan()`: 같은 선택 조합의 결과를 캐시한다 (`st.cache_data`).
  - 사이드바 선택 → `plan_days` → 카드와 지도용 JSON → `route_view.html`에 넣어 iframe으로 그린다.
  - JSON 속 `</`는 이스케이프해서 가게 이름이 스크립트를 깨지 못하게 한다.
- `route_view.html`: Leaflet 지도와 일정 카드다. 카드를 누르면 지도가 이동하고, 날짜별 색을 쓰며, 추천 순서 경로를 비교할 수 있다.

## 9. 실험 스크립트 지도

| 스크립트 | 질문 | 결과 파일 | 그림 |
| --- | --- | --- | --- |
| `baseline.py` | bias λ | `baseline.json` | |
| `mf.py` | MF f × λ | `mf.json` | M3, M4 |
| `item_knn.py` | 유사도 × k × β | `item_knn.json` | M1 |
| `cbf_hybrid.py` | CBF λ, Hybrid α·κ | `cbf_hybrid.json` | M5/M6 |
| `compare_models.py` | 모델 비교, 신뢰구간, 검정, cold start | `model_compare.json` | V1, V5 |
| `ranking.py` | RQ1 거리 재순위화 | `ranking.json` | R2 |
| `route.py` | RQ3 동선 알고리즘 | `route.json` | T2, T3 |
| `popularity.py` | 인기도 blend | `popularity.json` | D1 |
| `learned_distance.py` | 거리 민감도 학습 | `learned_distance.json` | L1, L2 |
| `bpr.py` | 순위 학습 | `bpr.json` | B1 |
| `cold_start.py` | RQ2 재실험 | `cold_start.json` | C1 |
| `walking.py` | 도보 거리로 RQ1·RQ3 | `walking.json` | W1, W2 |

`python experiments/run_all.py`가 위를 순서대로 모두 돌리고 그림을 다시 그린다.

## 10. 테스트와 CI

- `tests/` 85개는 모두 **합성 데이터**로 돈다. Yelp 데이터 없이 CI에서 통과한다.
  - 수식 손계산 (RMSE, NDCG)
  - 알려진 답 복원 (로지스틱 회귀 가중치, k-means 군집)
  - 경계 조건 (빈 입력, k = 0, 후보 부족, 자정 넘김)
  - 버그 재현 (자기 자신 이웃, 3일 코스)
- **CI**: PR마다 ruff 정적 검사, Python 3.10·3.12 테스트, 커밋 컨벤션 검사를 돈다. CodeRabbit이 리뷰한다.

## 11. 읽는 순서 추천

1. `pipeline.py`: 전체 흐름이 한 화면에 있다.
2. `models/baseline.py`: 모든 모델의 바닥이다.
3. `models/mf.py`: 경사하강법 수업 내용 그대로다.
4. `candidates.py` → `rerank.py`: 추천 목록이 만들어지는 곳이다.
5. `route.py`: Held-Karp 비트마스크 DP다.
6. `experiments/compare_models.py`: 평가와 통계 검정이다.
7. 나머지 모델과 `roads.py`, `app/`은 관심 있는 순서대로 읽으면 된다.

## 12. 발표 질문 대비

| 예상 질문 | 답 |
| --- | --- |
| 왜 MF가 bias보다 못한가? | 사용자당 평점 중앙값 6개, 밀도 0.20%라 잠재 요인을 배울 데이터가 부족하다. 정규화를 키우면 bias로 수렴한다. |
| 거리를 넣으면 정확도가 왜 오르나? | 사람들은 실제로 가까운 곳을 다닌다 (활동 반경 중앙값 2.3km, 그림 E8). 거리는 "갈 곳"의 강한 신호다. |
| 직선거리로 충분한가? | 도보는 직선의 1.30배지만 거의 고르게 늘어서, 순서와 순위 결론은 같다 (W1, W2). 화면의 시간은 실제 도보로 보여 준다. |
| test로 튜닝하지 않았나? | 모든 선택은 validation에서 하고, test는 한 번만 쟀다. RQ1 τ 곡선은 선택이 아니라 trade-off를 보여 주는 그림이다. |
| 왜 Held-Karp인가? | 5~8곳이면 정확한 최적해가 1ms 안에 나온다. NN + 2-opt는 빠르지만 18%는 최적이 아니다. |
| 개인화가 안 되는 이유는? | 별점(MF, CBF), 순위(BPR), 내용(속성 42개) 모두 시도했지만 이 데이터에서는 인기도와 거리가 신호의 대부분이다. 실패도 결과로 보고한다. |
