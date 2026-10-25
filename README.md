# TasteRoute

취향 기반 장소 추천과 동선 생성 · 가천대학교 머신러닝 Term Project (2026)

사용자가 좋아할 만한 장소 Top-N을 추천하고, 거리를 반영해 실제로 갈 수 있는 곳으로 다시 순위를 매긴 뒤, 그곳들을 도는 하루 동선까지 만든다.

## 시스템 구조

| 단계 | 방식 | 내용 |
| --- | --- | --- |
| 1. 별점 예측 | 머신러닝 | Bias, Item-kNN, MF(SGD + L2), CBF(TF-IDF 카테고리 + ridge), Hybrid(적응형 alpha) |
| 2. 거리 재순위화 | 알고리즘 | 예상 별점 × 거리 감쇠 가중치 (haversine, 기본: 지수 감쇠 tau = 2km) |
| 3. 동선 최적화 | 알고리즘 | 최종 5~8곳의 최단 방문 순서 (Held-Karp DP, 비교: 완전탐색, NN + 2-opt). 데모는 OpenStreetMap 보행 도로망 위 Dijkstra 거리로 실제 길을 따라감 |

## 연구 질문

- **RQ1** 거리를 순위에 넣으면 정확도는 얼마나 잃고 이동거리는 얼마나 줄어드나?
- **RQ2** 평가 수가 적은 사용자(cold start)에게 CBF가 오차를 줄여 주나?
- **RQ3** 빠른 근사 경로(NN + 2-opt)는 정확한 최적해에 얼마나 가깝고, 얼마나 빠른가?

## 데이터

[Yelp Open Dataset](https://www.yelp.com/dataset) (리뷰 6,990,280개, 업체 150,346개). 받는 방법은 [`data/README.md`](data/README.md) 참고.

| 단계 | 평점 | 사용자 | 업체 |
| --- | --- | --- | --- |
| 원본 | 6,990,280 | 1,987,929 | 150,346 |
| Philadelphia 시청 반경 15km 음식점, 중복 제거 | 809,070 | 242,139 | 8,956 |
| 5-core (작업 데이터, 밀도 0.20%) | 490,539 | 31,743 | 7,693 |

사용자별로 평점을 약 70 / 15 / 15로 train, validation, test에 나누고 (validation과 test는 사용자마다 최소 1개라서, 평점 5개인 사용자는 3 / 1 / 1), cold-start 실험용 사용자 1,306명(학습 평점 1/3/5개만 남김)을 따로 둔다.

## 평가

| 단계 | 지표 |
| --- | --- |
| 별점 예측 | RMSE, MAE |
| Top-N 랭킹 | Precision@K, Recall@K, NDCG@K, HitRate@K (K = 5, 10) |
| 추천 품질 | Catalog coverage, intra-list diversity, novelty |
| 거리 재순위화 (RQ1) | Top-K 평균 거리, 줄인 거리(km)당 NDCG 손실 |
| Cold start (RQ2) | 학습 평점 1/3/5개일 때 RMSE |
| 동선 (RQ3) | 경로 길이, 최적해 대비 차이, 실행 시간 |

모델 간 차이는 사용자별 paired t-test와 bootstrap 95% 신뢰구간으로 검정한다.

## 폴더 구조

```
data/                         데이터 (커밋 안 함, data/README.md 참고)
data_analysis_preprocessing/  전체 데이터 탐색, 전처리, EDA·전처리 그림과 결과
src/tasteroute/               패키지
  data.py                       전처리 결과 로드, split 조회, 시간순 분할
  metrics.py                    RMSE, MAE, Precision/Recall/HitRate/NDCG@K
  item_extras.py                원본에서 가게 속성 42개와 영업시간 추출
  models/                       baseline(전체 평균, bias), mf, item_knn, cbf, hybrid, content_knn,
                                visit(거리 민감도 학습용 로지스틱 회귀), bpr(순위 학습)
  candidates.py                 후보 생성, 인기도 blend
  geo.py                        haversine, 사용자 기준점
  roads.py                      OpenStreetMap 보행 도로망, Dijkstra 도보 거리와 경로
  rerank.py                     거리 감쇠 재순위화
  route.py                      완전탐색, Held-Karp, NN, 2-opt
  pipeline.py                   한 사람의 하루·여러 날 일정 (추천 → 재순위화 → MMR → 경로)
  cluster.py                    k-means, 용량 제한 k-means (여러 날 코스)
experiments/                  실험 스크립트 (결과는 results/, 그림은 figures/)
results/                      실험 결과 JSON
figures/                      모델·랭킹·동선 그림
demo/                         필라델피아 빵집투어 지도 데모
app/                          지도 중심 대시보드 (Streamlit, 로컬 전용)
tests/                        단위 테스트 (pytest)
notebooks/                    탐색용 노트북 (선택)
docs/                         제안서, 코드 구현 계획, 구현 가이드, 발표 개요
```

별점 모델은 같은 인터페이스를 가진다: `fit(train)`, `predict(u, i)`, `score_all(u)`. 순위 모델 BPR은 별점을 예측하지 않으므로 `fit`과 `score_all`만 있다.

## 실행

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,demo]"          # tasteroute 패키지 설치 (src layout)

# 1. data/raw/에 Yelp JSON을 넣고 전처리 (약 90초, data/processed/ 생성)
python data_analysis_preprocessing/preprocess.py

# 2. 모든 실험 + 그림 재생성 (약 6분, 대부분 MF 그리드)
python experiments/run_all.py

# 3. 테스트
pytest

# 4. 데모: 시청에서 출발하는 빵집 6곳, 실제 길을 따라가는 도보 경로 → demo/bakery_tour.html
#    첫 실행 때 주변 보행 도로망을 OpenStreetMap에서 받는다 (약 1분, data/osm/에 캐시)
python demo/bakery_tour.py            # 새 방문자
python demo/bakery_tour.py --user 1458  # 데이터셋 사용자 맞춤 (전처리가 매긴 0부터의 내부 번호, Yelp user_id 아님)
python demo/bakery_tour.py --straight   # 다운로드 없이 직선 연결

# 5. 대시보드: 조건을 바꾸며 코스, 실험 결과, 데이터를 한 화면에서 (http://localhost:8501)
pip install -e ".[app]"
python -m tasteroute.roads              # (선택) 도심 보행 도로망을 한 번에 받아 두기
streamlit run app/dashboard.py
```

대시보드 화면 구성과 동작은 [docs/Dashboard.md](docs/Dashboard.md)에 정리했다. 영업시간 반영(도착 시각과 영업 상태), 여러 날 코스(용량 제한 k-means로 동네 나누기), 다양성 조절(MMR), BPR 순위 방식을 지원한다. Yelp 데이터를 읽기 때문에 `localhost`에서만 열리고, 공개 서버에 배포하지 않는다.

실험을 하나씩 돌리려면 `python experiments/<이름>.py` (baseline, mf, item_knn, cbf_hybrid, compare_models, ranking, route, popularity, learned_distance, bpr, cold_start, walking, figures). cold_start는 먼저 `python -m tasteroute.item_extras`, walking은 `python -m tasteroute.roads`가 필요하다.

## 결과 요약

모두 test set 기준, 튜닝은 validation에서만 했다.

**별점 예측** (warm 사용자 평점 69,847개, `results/model_compare.json`)

| 모델 | 설정 | RMSE [95% CI] | MAE |
| --- | --- | --- | --- |
| 전체 평균 | - | 1.190 [1.182, 1.198] | 0.943 |
| Bias | λ = 5 | **1.051** [1.044, 1.059] | 0.818 |
| Item-kNN | cosine, k = 80, β = 0 | 1.121 [1.113, 1.130] | 0.839 |
| MF (SGD) | f = 10, λ = 0.1, 15 epoch | 1.052 [1.044, 1.059] | 0.819 |
| CBF (ridge) | λc = 30 | **1.051** [1.043, 1.059] | 0.818 |
| Hybrid | κ = 100 | **1.051** [1.044, 1.059] | 0.818 |

사용자당 학습 평점 중앙값이 6개라 bias 외의 구조를 배우기 어렵다. MF는 bias보다 근소하게 나쁘고(p = 0.0015), CBF와 Hybrid는 bias와 구분되지 않는다(p = 0.81, 0.93).

**RQ1 거리 재순위화** (test 사용자 3,000명, `results/ranking.json`): 지수 감쇠 τ = 2km에서 Hybrid 목록의 평균 거리 6.3km → 2.1km, NDCG@10 0.0009 → 0.0062. 거리를 넣으면 정확도가 떨어지는 대신 오히려 오른다. 사용자가 실제로 가까운 곳을 다니기 때문이다.

**RQ2 cold start** (`results/model_compare.json`, `results/cold_start.json`, [설명](docs/Cold_Start.md)): 학습 평점 1/3/5개 사용자에서 CBF·Hybrid는 bias와 0.002 이내. Yelp 가게 속성 42개를 더하고 평점 하나로도 일반화하는 내용 기반 이웃 모델을 써도 같았다. 속성을 쓴 두 모델에서 validation은 내용 정보를 최대한 끄는 설정을 골랐다 (모든 모델이 bias 대비 p ≥ 0.19). 답: 이 데이터에서 내용 정보는 신규 사용자의 별점을 설명하지 못한다.

**RQ3 동선** (N = 6, `results/route.json`): 추천 순서대로 돌면 최적 경로보다 중앙값 26% 길다. NN + 2-opt는 82%의 사용자에서 정확한 최적해(평균 차이 0.9%). Held-Karp는 N = 8에서 약 1ms.

**실제 도보 거리로 다시 재기** (`results/walking.json`, [설명](docs/Walking_Eval.md)): 도심 test 사용자 2,230명을 보행 도로망 거리로 다시 쟀다. 실제 도보는 직선보다 중앙값 1.30배 길다. 도보 거리로 재순위화해도 정확도에서 유의한 차이를 찾지 못했고(p = 0.61) 목록 평균 도보 거리는 5% 짧았으며(점 추정값), 직선으로 최적화한 동선은 65%가 도보로도 최적이고 평균 1.2%만 더 걷는다. 직선거리는 실험용 근사로 충분하고, 실제 길은 화면의 경로와 시간에 필요하다.

**랭킹 개선** (`results/popularity.json`): 예측 별점만으로는 인기순(P@10 0.0094)을 이기지 못한다. 별점에 log 인기도를 섞고(γ = 5) 거리 재순위화를 하면 P@10 0.0102지만, 인기순 + 거리(0.0100)와 사실상 같다. 이 데이터에서 랭킹 신호는 대부분 인기도와 거리에서 나온다.

**시간순 분할** (`results/temporal.json`, [설명](docs/Temporal_Split.md)): 기존 분할은 사용자마다 평점을 섞어 나눠서 warm 사용자의 88%가 test보다 나중 평점으로 학습한다. 사용자마다 오래된 평점으로 학습하고 최근 평점으로 평가하면 Bias RMSE는 1.051 → 1.108, 인기순 NDCG@10은 0.0323 → 0.0218로 대부분의 지표가 나빠진다. 그래도 별점 모델 넷은 0.0005 안에서 같고, 거리 재순위화는 Hybrid NDCG@10을 0.0012 → 0.0049로 올리고 목록 거리를 5.0km → 2.3km로 줄이며, 가장 좋은 방법은 여전히 Hybrid blend + 거리(0.0245)다. 결론은 분할 방식에 달려 있지 않다.

**거리 민감도 학습** (`results/learned_distance.json`, [설명](docs/ML_Learned_Distance.md)): 고정 규칙 `e^(-d/2km)` 대신 로지스틱 회귀(직접 구현한 경사하강법 + L2)로 거리, 개인 이동 반경, 예상 별점, 인기도의 가중치를 학습했다. 기존 설정보다 NDCG@10이 유의하게 높고(0.0340 → 0.0365, p = 0.003), 검증으로 튜닝한 고정 규칙과는 유의한 차이를 찾지 못했고(p = 0.36), 커버리지는 더 넓었다(1.01% → 1.26%, 점 추정값). 개인 이동 반경을 빼면 유의하게 나빠진다. 1km 지점에서 1km 더 멀어지는 손해는 인기도 1.56배로 상쇄된다.

**순위 학습 BPR** (`results/bpr.json`, [설명](docs/BPR.md)): 별점 대신 "어디에 갔는가"의 순위를 쌍 비교 손실로 직접 학습했다. 전체로는 인기순보다 낫다는 증거가 없고(NDCG@10 0.0326 vs 0.0323, p = 0.61), 활동량별로는 21개 이상 집단에서만 유의하게 앞섰다(0.0419 vs 0.0381, p = 0.03, 세 집단 보정 전; 6~20개 집단도 평균은 높지만 p = 0.13). 학습 목표를 순위로 바꿔도 이 데이터의 개인화 신호는 약하다.

## 참고 문헌과 코드 출처

괄호 안의 코드 경로는 `src/tasteroute/` 기준이다 (예: `models/mf.py` = `src/tasteroute/models/mf.py`).

- Y. Koren, R. Bell, C. Volinsky, "Matrix Factorization Techniques for Recommender Systems", *IEEE Computer*, 2009 — bias 모델, MF와 SGD 갱신식 (`models/baseline.py`, `models/mf.py`)
- A. Yeung, "Matrix Factorization: A Simple Tutorial and Implementation in Python", https://albertauyeung.github.io/2017/04/23/python-matrix-factorization.html/ — MF SGD 구현 참고 (과제 안내 자료)
- B. Sarwar et al., "Item-based Collaborative Filtering Recommendation Algorithms", *WWW*, 2001 — Item-kNN 유사도 (`models/item_knn.py`)
- Y. Koren, "Factor in the Neighbors: Scalable and Accurate Collaborative Filtering", *ACM TKDD*, 2010 — 유사도 shrinkage, bias 위의 이웃 예측
- K. Järvelin, J. Kekäläinen, "Cumulated Gain-based Evaluation of IR Techniques", *ACM TOIS*, 2002 — NDCG (`metrics.py`)
- M. Held, R. Karp, "A Dynamic Programming Approach to Sequencing Problems", *J. SIAM*, 1962 — Held-Karp (`route.py`)
- G. A. Croes, "A Method for Solving Traveling-Salesman Problems", *Operations Research*, 1958 — 2-opt (`route.py`)
- Yelp Open Dataset, https://www.yelp.com/dataset — 데이터 (재배포 금지, 커밋 안 함)
- OpenStreetMap contributors, https://www.openstreetmap.org — 보행 도로망 (ODbL, 데모 실행 때 받고 커밋 안 함)
- G. Boeing, "OSMnx: New Methods for Acquiring, Constructing, Analyzing, and Visualizing Complex Street Networks", *Computers, Environment and Urban Systems*, 2017 — 도로망 다운로드 (`roads.py`)
- E. W. Dijkstra, "A Note on Two Problems in Connexion with Graphs", *Numerische Mathematik*, 1959 — 최단 경로 (`roads.py`)

수업에서 다루지 않은 라이브러리 함수(`np.bincount`, `scipy.sparse.csr_matrix`, `np.argpartition`, `np.linalg.solve`, `scipy.stats.ttest_rel`, `scipy.sparse.csgraph.dijkstra`, `osmnx`, `folium` 등)는 사용한 위치의 주석에 무엇을 하는지와 파라미터를 적었다.

## 문서

- [제안서](docs/Term_Project_Proposal.docx)
- [코드 구현 계획](docs/TasteRoute_Code_Plan.md)
- [구현 가이드](docs/TasteRoute_Build_Guide.md)
- [실제 도보 길찾기](docs/Walking_Route.md)
- [거리 민감도 학습 (로지스틱 회귀)](docs/ML_Learned_Distance.md)
- [순위 학습 BPR](docs/BPR.md)
- [가게 속성으로 cold start 다시 보기 (RQ2)](docs/Cold_Start.md)
- [실제 도보 거리로 RQ1·RQ3 다시 재기](docs/Walking_Eval.md)
- [시간순 분할로 다시 평가하기](docs/Temporal_Split.md)
- [대시보드](docs/Dashboard.md)
- [코드 읽기 안내 (구조, 수식, 읽는 순서, 발표 예상 질문)](docs/Code_Walkthrough.md)
- [발표 개요](docs/TasteRoute_Slides_Outline.md)
- [데이터 분석과 전처리](data_analysis_preprocessing/README.md)

## 과제 요구사항

- 특정 상품·서비스 도메인의 **추천 시스템**을 만든다 (예: 영화, 음악, 음식, 상점·식당 등).
- 협업 필터링(memory-based, model-based), content-based filtering, hybrid filtering **4가지 중 2가지 이상**을 사용한다.
- 수업에서 배운 머신러닝 모델·기법을 함께 쓰면 좋다 (예: kNN, 결정 트리, 클러스터링, 연관 규칙, 벡터 공간 모델, 경사하강법, 정규화, 손실함수).
- 공개 데이터셋 또는 직접 만든 데이터셋을 쓸 수 있다.

**제안서** (2주 안에 제출): 시스템 목적과 도메인, 사용할 데이터셋, 필터링 방식, (선택) 머신러닝 모델

**최종 제출물**

1. PPT 발표 (팀당 7분 + Q&A 3분)
   - End-to-End 과정: 비즈니스 목적 → 데이터 탐색 → 전처리 → 모델링 → 평가·분석 (성공이든 실패든 결과를 설명)
   - 배운 점: 어려웠던 점과 해결 방법, 다시 한다면 다르게 할 것, 개인별로 배운 것
   - 팀워크: 팀원별 역할과 기여도(%)
   - 오픈소스 SW: 전체 소프트웨어 아키텍처 설명, GitHub URL
2. 주석이 잘 달린 소스 코드
   - 인터넷(블로그, Kaggle, GitHub 등)의 코드를 쓰면 **반드시 출처를 밝힌다** (안 밝히면 표절).
   - 수업에서 다루지 않은 모듈·클래스·라이브러리·함수는 파라미터까지 설명을 단다.
3. 결과물: 플롯, 코드 실행 결과
