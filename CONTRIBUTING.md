# 협업 규칙

모든 변경은 **이슈 → 브랜치 → PR** 순서로 올린다. `main`과 `develop`에 직접 push하지 않는다 (보호 규칙으로 막혀 있다).

## 브랜치 (Git Flow)

| 브랜치 | 용도 | 어디서 만들고 | 어디로 머지 |
| --- | --- | --- | --- |
| `main` | 제출·발표용 안정 버전 (기본 브랜치) | - | - |
| `develop` | 개발 통합 | `main` | `main` (release로) |
| `feature/<이슈번호>-<설명>` | 기능, 실험, 문서 | `develop` | `develop` |
| `release/<버전>` | 제출 전 마무리 (숫자 확인, 문서 정리) | `develop` | `main`, `develop` |
| `hotfix/<설명>` | `main`의 긴급 수정 | `main` | `main`, `develop` |

예: `feature/31-temporal-split`, `release/1.0`

기본 브랜치는 `main`이라서 PR을 만들 때 **base를 `develop`으로 직접 골라야 한다**. `develop`에 머지된 PR의 `Closes #n` 이슈는 워크플로(`close-issues.yml`)가 대신 닫는다.

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

- CI 통과: `lint` (ruff), `test (3.10)`, `test (3.12)` (pytest), `commit-convention` (PR 제목과 커밋 메시지 형식)
- CodeRabbit 리뷰 코멘트를 포함한 모든 대화가 resolve됨
- `main`으로 가는 release PR은 팀원 1명 승인
- 머지는 merge commit으로 한다 (squash, rebase 머지는 꺼 둠). 머지 커밋 메시지는 PR 제목과 본문이 되므로 **PR 제목도 커밋 컨벤션을 따른다**. 머지된 브랜치는 자동 삭제된다.

이전 PR에 의존하는 PR은 base를 이전 브랜치로 둔다 (stacked PR). 이전 PR이 머지되면 GitHub가 base를 `develop`으로 자동으로 바꾼다.

## 커밋 컨벤션

AngularJS 커밋 메시지 규칙을 기반으로 한다.

```
<타입>(<범위>): <제목>

<본문>

<푸터>
```

### 타입

| 타입 | 설명 |
| --- | --- |
| `feat` | 새로운 기능 추가 |
| `fix` | 버그 수정 |
| `hotfix` | 서비스 중 긴급 수정 |
| `docs` | 문서 수정 (README, 주석 등) |
| `style` | 코드 포맷팅 (로직 변경 없음) |
| `refactor` | 코드 리팩토링 (기능 변경 없음) |
| `test` | 테스트 코드 추가 또는 수정 |
| `chore` | 빌드 설정, 패키지 관리, CI 등 기타 작업 |

### 범위 (선택)

변경된 부분. 영문으로 쓴다. 이 프로젝트의 예: `data`, `metrics`, `baseline`, `mf`, `knn`, `cbf`, `hybrid`, `candidates`, `rerank`, `route`, `pipeline`, `eval`, `figures`, `demo`, `guide`, `readme`, `ci`

### 제목

1. 50자 이내 (타입과 범위 제외)
2. 마침표를 쓰지 않는다
3. 명령형으로 쓴다 ("추가", "수정")

### 본문 (선택)

**무엇을**, **왜** 바꿨는지 쓴다 (어떻게보다). 변경 동기와 이전 동작과의 차이를 적고, 72자마다 줄바꿈을 권장한다.

### 푸터 (선택)

```
Closes #123
Related to #100
BREAKING CHANGE: <하위 호환이 깨지는 변경과 마이그레이션 방법>
```

### 예시

```
fix(knn): 사용자별 배열 복사로 인한 메모리 폭주 해결

사용자마다 정렬 배열을 새로 만들고 view로 붙잡아 약 160GB를 쓰던
문제를 고쳤다. 정렬을 한 번만 하도록 바꿔 fit이 0.3초로 줄었다.

Related to #6
```

커밋은 파일 단위가 아니라 **작업 단위**로 나눈다. 의미 있는 변경 하나를 커밋 하나로 묶는다.

`commit-convention` CI가 PR 제목과 PR 안의 커밋 제목을 검사한다 (merge, revert 커밋은 제외). 로컬에서 미리 확인하려면

```bash
PR_TITLE="feat(mf): ..." BASE=origin/develop HEAD=HEAD python3 .github/scripts/check_commits.py
```

## 이슈·PR 라벨 (깃모지)

| 라벨 | 용도 | 라벨 | 용도 |
| --- | --- | --- | --- |
| ✨feat | 기능 구현 | 🐛bug | 버그 수정 |
| 📝docs | 문서 작성 | ♻️refactor | 리팩토링 |
| ✅test | 테스트 | 🔧chore | 설정, CI |
| 🚑hotfix | main 긴급 수정 | 🔖release | release 브랜치 |
| 🍱asset | asset 파일(이미지 등) 추가 | 🚚renamed | 리소스 이동, 이름 변경 |
| 💄update UI | UI·스타일 파일 추가·수정 | 🎨UI | UI 그리기 |
| 🔥delete | 코드·파일 삭제 | 🎉initial setting | 프로젝트 시작 |
| 💬text | 텍스트·리터럴 추가·수정 | 🙈ignore | .gitignore 추가·수정 |
| 💡comment | 주석 추가·수정 | | |

## 데이터

Yelp 원본과 전처리 결과(`data/`)는 약관상 커밋하지 않는다. CI의 단위 테스트는 작은 합성 데이터만 쓰므로 Yelp 데이터 없이 돌아간다.
