# TasteRoute

취향 기반 장소 추천과 동선 생성 — 머신러닝 Term Project

Yelp 리뷰 데이터로 사용자 취향에 맞는 장소 Top-N을 추천하고, 거리를 반영해 실제로 갈 수 있는 추천으로 다듬은 뒤, 추천된 장소들을 도는 하루 동선을 만든다.

> 가천대학교 머신러닝 수업 Term Project (2026)

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

## 구성

1. **후보 추천** — Item-based kNN (memory-based CF), Matrix Factorization (model-based CF), Content-based Filtering (카테고리), Hybrid
2. **거리 재순위화** — 위경도 haversine 거리로 먼 장소 감점
3. **동선 최적화** (여유 시) — 최종 5~8곳 방문 순서를 완전탐색/DP로 계산

## 평가

| 대상 | 지표 |
|------|------|
| 후보 추천 | RMSE, Precision@K, Recall@K |
| 거리 재순위화 | 추천 목록 평균 이동거리, 카테고리 다양성 |
| 동선 최적화 | 총 이동거리, 계산 시간 |
| Cold start | 평가 수 적은 사용자의 RMSE |

## 폴더 구조

```
src/        모듈 코드
notebooks/  탐색·실험·결과 노트북
data/       데이터 (커밋 안 함, data/README.md 참고)
```

## 데이터

[Yelp Open Dataset](https://www.yelp.com/dataset) — 받는 방법은 [`data/README.md`](data/README.md) 참고.

## 실행

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
