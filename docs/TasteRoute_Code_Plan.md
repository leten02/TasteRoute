# TasteRoute 코드 구현 계획서

작성일: 2026-10-22 · 저장소: https://github.com/leten02/TasteRoute

데이터 분석과 전처리는 끝났습니다(`data_analysis_preprocessing/`). 남은 일은 모델 코드를 저장소 구조에 맞게 정리하고, 랭킹 약점을 보완하고, 실험과 데모를 완성하는 것입니다.
발표 구성은 별도 문서 [TasteRoute_Slides_Outline.md](TasteRoute_Slides_Outline.md)에 있습니다.

---

## 1. 현재 상태

| 단계 | 상태 | 위치 |
| --- | --- | --- |
| 제안서 (영문, 그림 36개) | 완료 | Docs 문서, `docs/Term_Project_Proposal.docx` |
| 전체 데이터 탐색, 도시 선정 | 완료 | `data_analysis_preprocessing/eda_full_dataset.py` |
| 전처리 (반경 15km, 5-core, 분할, cold-start 그룹) | 완료 | `data_analysis_preprocessing/preprocess.py` |
| 모델 + 재순위화 + 동선 프로토타입 | 1차 완료 (스크립트 하나에 몰려 있음) | `src/prototype.py` (개인 폴더, 저장소에는 아직 없음) |
| 저장소용 모듈 정리, 실험 노트북, 데모 | 미착수 | |

1차 결과 요약 (예비)
- 별점 예측: Bias, MF, CBF, Hybrid 모두 test RMSE 약 1.05. Item-kNN 1.14, 전체 평균 1.19
- 랭킹: 인기순 P@10 0.0095가 모든 예측 모델(0.0005~0.0007)보다 높음 → 가장 큰 약점
- 거리 재순위화: tau = 2km에서 NDCG@10 3.5배 상승, 목록 평균 거리 6.3km → 2.1km
- 동선: NN + 2-opt가 82% 사용자에게서 최적해, 추천 순서 그대로 돌면 최적보다 중앙값 27% 더 김

---

## 2. 코드 구현 목록

### 2.1 저장소 구조 (목표)

```
TasteRoute/
├── data/                         # 커밋 안 함 (raw/, processed/)
├── data_analysis_preprocessing/  # 완료
├── src/tasteroute/
│   ├── data.py          # 전처리 결과 불러오기, 분할 조회
│   ├── geo.py           # haversine, 거리 행렬, 사용자 기준점(anchor)
│   ├── models/
│   │   ├── baseline.py  # 전체 평균, bias 모델
│   │   ├── item_knn.py  # Item-based kNN (cosine / adjusted / Pearson, shrinkage)
│   │   ├── mf.py        # Matrix Factorization (SGD, L2)
│   │   ├── cbf.py       # 카테고리 ridge 기반 CBF
│   │   └── hybrid.py    # 고정 alpha / 적응형 alpha(n_u)
│   ├── candidates.py    # 후보 생성 (지지도 필터, 인기도 반영)
│   ├── rerank.py        # 거리 감쇠, MMR 다양성, 인기도 보정
│   ├── route.py         # 완전탐색, Held-Karp, NN, 2-opt
│   ├── metrics.py       # RMSE, MAE, P@K, R@K, NDCG, coverage, ILD, novelty
│   └── pipeline.py      # 사용자 1명 → 추천 → 재순위화 → 동선 (데모용)
├── experiments/         # 실험 스크립트 (결과는 results/에 JSON/CSV)
├── notebooks/           # 결과 시각화 노트북 (발표 그림)
├── demo/                # 데모 (CLI 또는 지도 HTML)
└── tests/               # 핵심 함수 단위 테스트
```

### 2.2 구현 목록

우선순위: P0 = 발표에 꼭 필요, P1 = 결과를 개선, P2 = 여유가 있을 때

| # | 작업 | 우선 | 완료 기준 | 담당 |
| --- | --- | --- | --- | --- |
| 1 | `prototype.py`를 2.1 구조의 모듈로 분리 | P0 | 각 모델이 `fit(train)`, `predict(u, i)`, `score_all(u)` 인터페이스를 가짐 | |
| 2 | `data.py`, `geo.py` 작성 | P0 | 전처리 결과를 한 줄로 불러오고, haversine이 테스트를 통과 | |
| 3 | Baseline, Item-kNN, MF, CBF, Hybrid 모듈화 | P0 | 프로토타입과 같은 seed에서 같은 RMSE가 재현됨 (±0.001) | |
| 4 | 인기도 반영 후보 생성 (`candidates.py`) | P0 | 예측 점수와 log 인기도를 섞은 점수 또는 인기 상위 N 후보 중 재정렬. P@10이 인기순 이상 | |
| 5 | 재순위화 모듈 (`rerank.py`) | P0 | 지수, 가우시안, 거듭제곱, 컷오프 감쇠와 tau 그리드 실험 재현 | |
| 6 | 동선 모듈 (`route.py`) | P0 | 완전탐색과 Held-Karp 결과가 N ≤ 8에서 일치 (테스트) | |
| 7 | 평가 모듈 (`metrics.py`) + bootstrap 신뢰구간 | P0 | 모든 지표를 한 함수로 계산, 결과를 `results/*.json`으로 저장 | |
| 8 | 실험 스크립트: 모델 비교, RQ1 거리, RQ2 cold-start, RQ3 동선 | P0 | `experiments/run_all.py` 한 번 실행으로 모든 결과 재생성 | |
| 9 | 하이퍼파라미터 튜닝 확대 (MF 학습률, kNN shrinkage beta, CBF lambda) | P1 | validation 기준 그리드 결과 표와 그림 | |
| 10 | CBF 특징 보강: 가격대, 주요 속성(attributes), 영업시간 | P1 | cold-start(m = 1, 3, 5)에서 bias 모델보다 RMSE가 낮은지 확인 | |
| 11 | 시간순 분할 실험 (robustness) | P1 | 랜덤 분할 결과와 나란히 비교한 표 | |
| 12 | 데모 (`demo/`): 사용자 또는 숙소 위치 입력 → 추천 6곳 + 경로 지도 | P1 | 지도 HTML 한 장 출력 (folium 등). 출발지는 데이터셋의 호텔 업체 좌표 사용 | |
| 13 | 영업시간을 반영한 동선 (time window) | P2 | 영업 중인 시간에만 방문하는 경로 생성 | |
| 14 | MMR 다양성, 인기도 보정 최종 튜닝 | P2 | ILD와 NDCG의 trade-off 그림 | |
| 15 | 단위 테스트 (`tests/`) | P1 | haversine, Held-Karp, metrics 테스트 통과 | |
| 16 | 주석과 출처 정리 | P0 | 모든 함수에 docstring. 수업에서 안 다룬 라이브러리 함수는 파라미터까지 설명. 외부 코드 사용 시 출처 표기 | |
| 17 | `requirements.txt`, README 실행 방법 갱신 | P0 | 새 환경에서 README대로 실행하면 결과가 재현됨 | |

### 2.3 과제 요구사항 체크

| 요구사항 | 대응 |
| --- | --- |
| 필터링 4가지 중 2가지 이상 | Memory-based CF(Item-kNN), Model-based CF(MF), CBF, Hybrid 4가지 모두 사용 |
| 수업 기법 활용 | kNN, 경사하강법(SGD), 정규화(L2, ridge), 손실함수(MSE), 벡터 공간 모델(TF-IDF, cosine) |
| 주석이 잘 달린 코드 | 구현 목록 16번 |
| 외부 코드 출처 | 구현 목록 16번. 참고한 코드는 해당 함수 docstring과 README에 링크 |
| 수업 외 라이브러리 설명 | scipy.sparse, pandas groupby 등은 사용 위치에 파라미터 설명 |
| 결과물(플롯, 실행 결과) | `notebooks/`와 `results/`에 저장 |

---

## 3. 일정 (안)

마감일이 정해지면 날짜로 바꿉니다.

| 주차 | 할 일 | 산출물 |
| --- | --- | --- |
| 1주차 | 구현 목록 1~3, 15 | 모델 모듈, 재현 확인 |
| 2주차 | 구현 목록 4~8 | 전체 실험 결과 JSON, 그림 |
| 3주차 | 구현 목록 9~12 | 개선 결과, 데모 |
| 4주차 | 구현 목록 16, 17, 발표 자료와 리허설 | PPT, 최종 코드 |

---

## 4. 정해야 할 것

- [ ] 메인 평가를 랜덤 분할로 할지, 시간순 분할로 할지
- [ ] 데모 형식 (CLI 출력 vs 지도 HTML)
- [ ] `src/prototype.py`를 저장소에 올릴 위치와 시점
