# TasteRoute

취향 기반 장소 추천과 동선 생성 — 머신러닝 Term Project

Yelp 리뷰 데이터로 사용자 취향에 맞는 장소 Top-N을 추천하고, 거리를 반영해 실제로 갈 수 있는 추천으로 다듬은 뒤, 추천된 장소들을 도는 하루 동선을 만든다.

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
