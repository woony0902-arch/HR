# 조직구성 시뮬레이션 에이전트

내년 전사 조직 개편을 지원하기 위한 에이전트 프로젝트입니다.

**목표**
1. 전사 기능 중복 조직 진단
2. 조직 운영 효율화(계층·스팬·인건비) 개선안 도출
3. 개편 시나리오의 Before/After 시뮬레이션

**조직 계층**: 사업부 / 본부 › 담당 › 팀

## 문서
| 문서 | 내용 |
|---|---|
| [docs/data-requirements.md](docs/data-requirements.md) | **필요 자료 목록** (P0/P1/P2 우선순위) — 자료 수집 요청 시 이 문서를 기준으로 |
| [docs/agent-design.md](docs/agent-design.md) | 에이전트 처리 단계 및 판정 로직 설계 |

## 데이터 템플릿
`templates/` 하위에 입력 데이터 스키마(헤더 + 예시 1~3행)를 두었습니다.
실제 데이터는 이 형식으로 `data/` 디렉터리에 넣습니다. (`data/`는 git에 커밋하지 않음)

| 파일 | 우선순위 | 설명 |
|---|---|---|
| `org_master.csv` | P0 | 조직 계층 마스터 |
| `members.csv` | P0 | 구성원 상세 (익명 사번) |
| `org_roles.csv` | P0 | 정형화된 조직 R&R |
| `org_history.csv` | P0 | 조직 변경 이력 스냅샷 |
| `function_taxonomy.csv` | P1 | 전사 기능 분류 체계 |
| `org_function_map.csv` | P1 | 조직 ↔ 기능 매핑 (에이전트 생성 → HR 확정) |
| `org_kpi.csv` | P1 | 조직별 KPI |
| `org_cost.csv` | P1 | 조직별 비용 |
| `comp_band.csv` | P1 | 직급·직무별 평균 인건비 (개인 인건비 대체용) |
| `headcount_plan.csv` | P1 | 정원/현원/채용 계획 |
| `constraints.csv` | P2 | 법규·규정상 제약 조건 |

## 개인정보 취급
- 사번은 해시 또는 일련번호로 치환 (이름·연락처 미수집)
- 인건비는 가능하면 직급·직무별 밴드 사용
- 평가등급, 징계 이력 등 민감 정보는 수집하지 않음
- 실제 데이터는 저장소에 커밋하지 않음 (`.gitignore`로 `data/` 제외)
