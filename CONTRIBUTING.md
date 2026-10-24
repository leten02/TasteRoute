# 협업 규칙

모든 변경은 **이슈 → 브랜치 → PR** 순서로 올린다. `main`과 `develop`에 직접 push하지 않는다 (보호 규칙으로 막혀 있다).

## 브랜치 (Git Flow)

| 브랜치 | 용도 | 어디서 만들고 | 어디로 머지 |
| --- | --- | --- | --- |
| `main` | 제출·발표용 안정 버전 | - | - |
| `develop` | 개발 통합 (기본 브랜치) | `main` | `main` (release로) |
| `feature/<이슈번호>-<설명>` | 기능, 실험, 문서 | `develop` | `develop` |
| `release/<버전>` | 제출 전 마무리 (숫자 확인, 문서 정리) | `develop` | `main`, `develop` |
| `hotfix/<설명>` | `main`의 긴급 수정 | `main` | `main`, `develop` |

예: `feature/31-temporal-split`, `release/1.0`

## 작업 순서

```bash
# 1. GitHub에서 이슈를 만든다 (템플릿: 작업 / 버그). 번호가 31이라면
git switch develop && git pull
git switch -c feature/31-temporal-split

# 2. 작업하고 로컬에서 확인
ruff check .
pytest

# 3. push하고 develop으로 PR. 본문에 "Closes #31"
git push -u origin feature/31-temporal-split
gh pr create --base develop
```

## PR 머지 조건

- CI 통과: `lint` (ruff), `test (3.10)`, `test (3.12)` (pytest)
- CodeRabbit 리뷰 코멘트를 포함한 모든 대화가 resolve됨
- `main`으로 가는 release PR은 팀원 1명 승인
- 머지는 merge commit으로 한다 (squash, rebase 머지는 꺼 둠). 머지된 브랜치는 자동 삭제된다.

이전 PR에 의존하는 PR은 base를 이전 브랜치로 둔다 (stacked PR). 이전 PR이 머지되면 GitHub가 base를 `develop`으로 자동으로 바꾼다.

## 커밋 메시지

`<종류>: <요약>` 형식. 종류는 `feat`(기능), `fix`(버그), `docs`(문서), `test`(테스트), `refactor`, `chore`(설정).

```
feat: MF (SGD, L2, 조기 종료)와 f x lambda 그리드 실험 (A4)
```

## 데이터

Yelp 원본과 전처리 결과(`data/`)는 약관상 커밋하지 않는다. CI의 단위 테스트는 작은 합성 데이터만 쓰므로 Yelp 데이터 없이 돌아간다.
