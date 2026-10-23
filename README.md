# TasteRoute

취향 기반 장소 추천과 동선 생성 · 가천대학교 머신러닝 Term Project (2026)

사용자가 좋아할 만한 장소 Top-N을 추천하고, 거리를 반영해 실제로 갈 수 있는 곳으로 다시 순위를 매긴 뒤, 그곳들을 도는 하루 동선까지 만든다.

## 시스템 구조

| 단계 | 방식 | 내용 |
| --- | --- | --- |
| 1. 별점 예측 | 머신러닝 | Bias, Item-kNN, MF(SGD + L2), CBF(TF-IDF 카테고리 + ridge), Hybrid(적응형 alpha) |
| 2. 거리 재순위화 | 알고리즘 | 예상 별점 × 거리 감쇠 가중치 (haversine, 기본: 지수 감쇠 tau = 2km) |
| 3. 동선 최적화 | 알고리즘 | 최종 5~8곳의 최단 방문 순서 (Held-Karp DP, 비교: 완전탐색, NN + 2-opt) |

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

사용자별 70 / 15 / 15로 train, validation, test를 나누고, cold-start 실험용 사용자 1,306명(학습 평점 1/3/5개만 남김)을 따로 둔다.

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
data_analysis_preprocessing/  전체 데이터 탐색, 전처리, 그림 (완료)
src/                          모델, 재순위화, 동선 모듈
experiments/                  실험 스크립트 (결과는 results/)
notebooks/                    결과 시각화
docs/                         제안서, 코드 구현 계획, 발표 개요
```

## 실행

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# data/raw/에 Yelp JSON을 넣은 뒤
python data_analysis_preprocessing/preprocess.py   # 약 90초, data/processed/ 생성
```

## 문서

- [제안서](docs/Term_Project_Proposal.docx)
- [코드 구현 계획](docs/TasteRoute_Code_Plan.md)
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
