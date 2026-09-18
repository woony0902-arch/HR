"""분석·시뮬레이션 결과를 마크다운 리포트와 CSV로 출력한다."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

LEVEL_MARK = {"ERROR": "🔴", "WARN": "🟡", "INFO": "⚪"}

# 리포트 표에 쓰이는 한글 헤더
COLUMN_LABELS = {
    "level": "등급", "code": "코드", "count": "건수", "message": "내용",
    "hq_code": "본부코드", "hq_name": "본부", "dept_code": "담당코드", "dept_name": "담당",
    "team_code": "팀코드", "team_name": "팀", "org_name": "조직", "org_code": "조직코드",
    "headcount": "인원", "leaders": "팀장수", "span": "span", "reports": "관리대상",
    "span_flag": "span판정", "size_flag": "규모판정", "level_": "계층",
    "avg_age": "평균연령", "median_age": "중위연령", "avg_tenure": "평균근속", "age_std": "연령편차",
    "dept_count": "담당수", "team_count": "팀수", "hq_count": "본부수",
    "avg_team_size": "평균팀규모", "small_teams": "소규모팀", "large_teams": "과대팀",
    "leader_count": "직책자", "leader_ratio_pct": "직책자비율(%)",
    "first_year": "최초연도", "last_year": "최종연도", "years_present": "관측연수",
    "rename_count": "개명횟수", "transfer_count": "이관횟수",
    "headcount_first": "최초인원", "headcount_last": "현재인원",
    "year_from": "이전연도", "year_to": "연도", "successor": "승계 조직",
    "successor_code": "승계코드", "successor_share": "승계비율",
    "successor_count": "승계처수", "left_company": "이탈",
    "absorbed_orgs": "흡수한 조직", "split_events": "분할횟수",
    "status": "유형", "main_source": "주요 출처", "main_source_share": "출처비율",
    "from_outside": "외부유입", "source_count": "출처수",
    "headcount_volatility": "인원변동성", "instability_score": "불안정점수",
    "team_name_from": "이동 출발", "team_name_to": "이동 도착", "people": "인원",
    "bidirectional": "양방향",
    "avg_headcount": "평균인원", "total_leavers": "누적이탈", "avg_attrition_pct": "평균이탈률(%)",
    "attrition_rate_pct": "이탈률(%)", "change_pct": "변화율(%)",
    "leader_age": "팀장연령", "years_to_retire": "정년까지", "candidate_pool": "후보군",
    "risk": "리스크", "reason": "사유", "retiring": "정년도래", "retiring_pct": "정년도래율(%)",
    "median_age": "중위연령", "under_40_pct": "40세미만(%)", "over_50_pct": "50세이상(%)",
    "retire_within_5y": "5년내정년", "retire_year": "정년도래연도", "total": "인원",
    "function_code": "기능코드", "function_name": "기능",
    "이름키워드": "이름 키워드", "역할에서_확인": "역할에서 확인", "일치율": "일치율",
    "실제_대표키워드": "실제 대표 키워드", "total_headcount": "투입인원",
    "teams": "수행 조직", "dispersion": "분산유형",
    "team_a": "팀 A", "team_b": "팀 B", "hq_a": "본부 A", "hq_b": "본부 B",
    "similarity": "역할유사도", "shared_functions": "공유기능",
    "shared_function_count": "공유기능수", "shared_keywords": "공통키워드",
    "headcount_sum": "합산인원", "score": "중복점수",
    "name_count": "명칭수", "code_count": "코드수", "dept_count_": "담당수",
    "emp_id": "사번", "year": "연도", "years": "연도", "leavers": "이탈",
    "hqs": "본부수", "depts": "담당수", "teams_": "팀수",
    "이동": "이동", "이탈": "이탈", "신규": "신규", "잔류": "잔류",
}


def _kr(df: pd.DataFrame) -> pd.DataFrame:
    """표 헤더를 한글로 바꾼다."""
    return df.rename(columns={c: COLUMN_LABELS.get(c, c) for c in df.columns})


def _md(df: pd.DataFrame, limit: int = 15) -> str:
    if df is None or len(df) == 0:
        return "_해당 없음_\n"
    shown = _kr(df.head(limit)).copy()
    for col in shown.columns:                     # 긴 목록이 표를 망가뜨리지 않게 자른다
        if shown[col].dtype == object:
            shown[col] = shown[col].map(
                lambda v: (v[:90] + f"… (외 {v.count(',') - 2}개)")
                if isinstance(v, str) and len(v) > 100 else v)
    table = shown.to_markdown(index=False)
    if len(df) > limit:
        table += f"\n\n_… 전체 {len(df)}건 중 상위 {limit}건. 전체는 CSV 참조._"
    return table + "\n"


def save_tables(tables: dict[str, pd.DataFrame], out_dir: Path) -> list[str]:
    csv_dir = out_dir / "tables"
    csv_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for name, df in tables.items():
        if df is None or len(df) == 0:
            continue
        path = csv_dir / f"{name}.csv"
        df.to_csv(path, index=False, encoding="utf-8-sig")
        saved.append(str(path))
    return saved


def diagnosis_report(ctx: dict, out_dir: Path) -> Path:
    """1차 진단 리포트."""
    head = ctx["headline"]
    lines: list[str] = []
    add = lines.append

    add(f"# 조직 진단 리포트 ({ctx['base_year']}년 기준)\n")
    add(f"분석 대상: {ctx['years'][0]}~{ctx['years'][-1]}년 구성원 명단 {len(ctx['years'])}개 스냅샷 "
        f"+ {ctx['base_year']}년 팀 역할 정의표\n")

    add("## 0. 데이터 품질 점검\n")
    quality = ctx["quality_summary"]
    if len(quality):
        quality = quality.copy()
        quality["level"] = quality["level"].map(lambda v: f"{LEVEL_MARK.get(v, '')} {v}")
        add(_md(quality, 30))
        add("\n🔴 ERROR 항목은 분석 결과를 왜곡하므로 원자료 수정 후 재실행을 권합니다.\n")
    else:
        add("문제 없음.\n")

    add("\n## 1. 전사 구조 요약\n")
    add(f"| 지표 | 값 |\n|---|---|")
    labels = {"headcount": "총 인원", "hq_count": "본부/사업부", "dept_count": "담당",
              "team_count": "팀", "leader_count": "직책자", "leader_ratio_pct": "직책자 비율(%)",
              "avg_team_size": "평균 팀 규모", "median_span": "중위 span", "max_span": "최대 span",
              "small_teams": "소규모 팀", "large_teams": "과대 팀",
              "teams_without_leader": "팀장 공석 팀", "avg_age": "평균 연령", "avg_tenure": "평균 근속"}
    for key, label in labels.items():
        add(f"| {label} | {head[key]} |")
    add("")

    add("\n### 본부별 비교\n")
    add(_md(ctx["hq_comparison"], 20))

    add("\n### span of control 점검\n")
    spans = ctx["span_table"]
    flagged = spans[spans["span_flag"] != "적정"].sort_values("reports", ascending=False)
    add(f"적정 범위를 벗어난 조직 {len(flagged)}건 / 전체 {len(spans)}건\n")
    add(_md(flagged[["level", "org_name", "reports", "span_flag"]], 20))

    add("\n### 소규모 · 과대 팀\n")
    teams = ctx["team_profile"]
    add(_md(teams[teams["size_flag"] != "적정"][
        ["hq_name", "dept_name", "team_name", "headcount", "size_flag", "avg_age"]], 20))

    add("\n## 2. 3개년 조직 동역학\n")
    add("### 연도별 추이\n")
    add(_md(ctx["yearly"], 10))

    add("\n### 조직 승계 이력 (구성원 이동 기반)\n")
    add("부서 코드가 연도 간 재사용되는 체계에서는 코드 매칭으로 통합·개명을 잡을 수 없습니다.\n"
        "구성원이 실제로 어디로 옮겨 갔는지를 근거로 조직의 연속성을 추정했습니다.\n")
    trans = ctx["transitions"]
    if len(trans):
        add("연도별 조직 변동 유형:\n")
        pivot = (trans.groupby(["year_to", "status"]).size().unstack(fill_value=0).reset_index())
        add(_md(pivot, 10))
        add("\n**통합/개명된 조직** (인원의 절반 이상이 다른 조직으로 승계)\n")
        add(_md(trans[trans["status"] == "통합/개명"][
            ["year_to", "team_name", "headcount", "successor", "successor_share", "left_company"]]
            .sort_values("headcount", ascending=False), 20))
        add("\n**분할된 조직**\n")
        add(_md(trans[trans["status"] == "분할"][
            ["year_to", "team_name", "headcount", "successor_count", "successor"]]
            .sort_values("headcount", ascending=False), 15))

    add("\n### 신설 조직의 출처\n")
    add(_md(ctx["new_orgs"][["year", "team_name", "headcount", "status",
                             "main_source", "main_source_share", "from_outside"]]
            .sort_values("headcount", ascending=False), 20))

    add("\n### 불안정 조직 (변동 이벤트가 잦은 순)\n")
    life = ctx["lifecycle"]
    add(_md(life[life["instability_score"] > 0][
        ["team_name", "headcount", "years_present", "absorbed_orgs", "split_events",
         "headcount_volatility", "instability_score"]], 15))

    add("\n### 조직 간 인원 이동\n")
    add("양방향(bidirectional=True)으로 인원이 오간 조직쌍은 업무 경계가 모호할 수 있습니다.\n")
    add(_md(ctx["flows"][["team_name_from", "team_name_to", "people", "bidirectional"]], 15))

    add("\n### 조직별 이탈률 (상위)\n")
    add("_명단에서 사라진 인원 기준이므로 퇴사와 계열사 전출이 구분되지 않습니다._\n")
    add(_md(ctx["attrition"][["team_name", "avg_headcount", "total_leavers", "avg_attrition_pct"]], 15))

    add("\n## 3. 인력 구조와 지속 가능성\n")
    add("### 전사 연령 구조 추이\n")
    add(_md(ctx["age_structure"], 10))

    add("\n### 정년 도래가 집중된 조직 (향후 5년)\n")
    add("해당 조직 인원의 몇 %가 5년 내 정년에 도달하는지입니다. 개편 이전에 먼저 봐야 할 조직들입니다.\n")
    conc = ctx["retire_concentration"]
    add(_md(conc[conc["retiring_pct"] > 0][
        ["hq_name", "team_name", "headcount", "avg_age", "retiring", "retiring_pct"]], 25))

    add("\n### 승계 리스크\n")
    risk = ctx["succession"]
    add(f"높음 {int((risk['risk'] == '높음').sum())}건 · 중간 {int((risk['risk'] == '중간').sum())}건\n")
    add("_중위 연령이 50세인 조직이라 '40세 이상'은 변별력이 없습니다. "
        "정년까지 10년 이상 남은 팀원 수를 후보군 대리지표로 사용했습니다._\n")
    add(_md(risk[risk["risk"] != "낮음"][
        ["team_name", "headcount", "leader_age", "years_to_retire", "candidate_pool", "risk", "reason"]], 20))

    add("\n### 정년 도래 추이\n")
    add(_md(ctx["retirement"], 10))

    add("\n### 자연감소 시뮬레이션 (채용 0 가정)\n")
    add("정년과 과거 이탈률만 적용했을 때의 인원 추이입니다. 개편 필요성의 하한선으로 보시면 됩니다.\n")
    add(_md(ctx["decline"], 20))

    add("\n## 4. 지역 조직 구조\n")
    add("전국 국사·사옥 구조에서는 같은 기능이 여러 지역에 놓이는 것이 정상입니다.\n"
        "따라서 지역 조직의 검토 단위는 '어느 팀과 어느 팀'이 아니라\n"
        "**이 기능을 전국 몇 개 단위로 운영할 것인가**입니다.\n")
    add("\n### 권역별 현황\n")
    add(_md(ctx["region_profile"], 15))
    add("\n### 기능별 권역 분포\n")
    add(_md(ctx["region_grid"], 20))
    add("\n_조직명에서 추론한 지역입니다. 근무지(사업장·국사) 데이터가 확보되면 실제 값으로 대체해야 합니다._\n")

    add("\n## 5. 조직명과 기능의 정합성\n")
    add("조직명은 그 조직이 무엇을 하는지 알리는 1차 신호입니다.\n"
        "이름이 역할을 드러내지 못하거나, 같은 일을 다른 이름으로 부르면 협업 비용이 늘어납니다.\n")

    add("\n### 명명 규칙 일관성\n")
    add(_md(ctx["naming_conventions"], 10))

    add("\n### 이름과 역할이 어긋난 조직\n")
    add("조직명의 핵심어가 그 조직의 역할 서술에 실제로 등장하는지 본 것입니다.\n"
        "일치율이 낮으면 이름만 남고 업무가 바뀌었거나, 이름이 업무 범위를 담지 못하는 경우입니다.\n")
    match = ctx["name_role_match"]
    add(_md(match[match["일치율"] < 0.5][
        ["hq_name", "team_name", "headcount", "이름키워드", "실제_대표키워드", "일치율"]], 25))

    if len(ctx["vague_names"]):
        add("\n### 기능을 특정할 수 없는 조직명\n")
        add(_md(ctx["vague_names"], 20))

    if len(ctx["name_conflict_동명이의"]):
        add("\n### 같은 이름, 다른 역할\n")
        add(_md(ctx["name_conflict_동명이의"], 15))

    if len(ctx["name_conflict_이명동의"]):
        add("\n### 같은 역할, 다른 이름\n")
        add("역할 서술은 매우 닮았는데 조직명에 공통 기능어가 없는 쌍입니다.\n"
            "같은 기능이 서로 다른 이름으로 불리고 있을 가능성이 있습니다.\n")
        add(_md(ctx["name_conflict_이명동의"], 15))

    add("\n## 6. 기능 중복 진단\n")
    add("### 전사 기능 지도\n")
    add("한 기능을 몇 개 팀이 나눠 맡고 있는지, 총 몇 명이 투입되어 있는지입니다.\n")
    fmap = ctx["function_map"]
    add(_md(fmap[fmap["team_count"] > 1][
        ["function_name", "team_count", "hq_count", "total_headcount", "dispersion", "teams"]], 20))

    add("\n### 병렬 조직군 (지역·채널 분할)\n")
    add("기능명이 같고 앞의 식별자만 다른 조직들입니다. 기능 중복이 아니라 커버리지 분할이므로\n"
        "개별 쌍이 아니라 **군 단위**로 통폐합을 검토해야 합니다.\n")
    fam = ctx["parallel_families"]
    add(_md(fam.drop(columns=["codes"]) if "codes" in fam else fam, 15))

    add("\n### 중복 후보 (팀 쌍)\n")
    add("판정초안은 **참고용**입니다. 진성 중복인지 의도된 분산인지는 반드시 사람이 확정해야 합니다.\n")
    dup = ctx["duplicates"]
    add(_md(dup[["team_a", "team_b", "hq_a", "hq_b", "similarity", "shared_functions",
                 "headcount_sum", "score", "판정초안"]] if len(dup) else dup, 20))

    add("\n### 기능 분류 사전 보정용 — 전사 빈출 용어\n")
    add("현재 사전은 일반적인 기업 기능으로 채운 **시드**입니다. 아래 용어를 참고해 "
        "회사 실정에 맞는 기능 분류를 확정해 주셔야 위 결과의 정확도가 올라갑니다.\n"
        "(팀별 대표 키워드는 `tables/keyword_candidates.csv` 참조)\n")
    add(_md(ctx["corpus_terms"], 30))

    if len(ctx["gaps"]):
        add("\n### 담당 조직이 확인되지 않는 기능\n")
        add("_기능 사전 기준입니다. 사전 자체를 회사 실정에 맞게 확정한 뒤 재판정이 필요합니다._\n")
        add(_md(ctx["gaps"], 30))

    add("\n---\n")
    add("## 이 리포트로 할 수 없는 것\n")
    add("- 인건비 절감액 → 조직별 비용 데이터 필요\n"
        "- 중복 후보의 실증(선언 vs 실제) → 조직별 KPI 필요\n"
        "- 신설·강화 기능 도출 → 내년 전략과제 필요\n\n"
        "현재 자료로는 **개편이 무엇을 바꾸는지**까지 계산되며, "
        "**개편이 좋은지**는 판단 기준(비용·성과·전략)이 들어와야 판정할 수 있습니다.\n")

    path = out_dir / "01_진단리포트.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def simulation_report(results: list, comparison: pd.DataFrame, out_dir: Path) -> Path:
    """시나리오 시뮬레이션 리포트."""
    lines: list[str] = []
    add = lines.append
    add("# 조직 개편 시뮬레이션 리포트\n")

    add("## 시나리오 비교\n")
    add(_md(comparison, 30))

    for item in results:
        result, delta, impact, funcs, violations, profile, cross_region = item
        add(f"\n---\n\n## {result.plan.name}\n")
        if result.plan.description:
            add(f"{result.plan.description}\n")

        add("\n### 적용 액션\n")
        for note in result.changelog:
            extra = []
            if note.get("surplus_leaders"):
                extra.append(f"보임해제 {len(note['surplus_leaders'])}명")
            if note.get("new_leader_needed"):
                extra.append(f"신규 팀장 {note['new_leader_needed']}명 필요")
            if note.get("unassigned"):
                extra.append(f"미배치 {note['unassigned']}명")
            suffix = f" — {', '.join(extra)}" if extra else ""
            add(f"- **{note['op']}** {note['detail']} (대상 {note['affected']}명){suffix}")

        add("\n### 구조 지표 Before → After\n")
        add(_md(delta, 20))

        add("\n### 사람에게 미치는 영향\n")
        for key in ["소속 변경 인원", "본부 이동 인원", "보임 해제 인원", "신규 팀장 필요", "미배치 인원"]:
            add(f"- {key}: **{impact[key]}명**")
        add("")

        add("\n### 개편으로 만들어진 조직의 연령 구조\n")
        add("통합은 사람을 합치는 일입니다. 정년이 몰린 조직끼리 합치면 몇 년 뒤 그 조직이 통째로 빕니다.\n")
        add(_md(profile, 20))

        if len(cross_region):
            add("\n### ⚠️ 지역 간 통합\n")
            add("전국에 국사·사옥이 있는 구조에서는 조직을 합쳐도 **사람은 원래 자리에 남습니다.**\n"
                "아래 통합은 인력 집중이 아니라 관리 단위 통합이며, 남은 조직장이 물리적으로\n"
                "떨어진 인력을 관리하게 됩니다. 근무지 데이터로 실제 거리를 확인해야 합니다.\n")
            add(_md(cross_region, 15))

        add("\n### 기능 커버리지\n")
        lost = funcs["기능 결손"]
        if lost:
            add(f"⚠️ **담당 조직이 사라지는 기능: {', '.join(lost)}**\n")
        else:
            add("기능 결손 없음.\n")
        resolved = funcs.get("해소된 중복", [])
        if resolved:
            add(f"✅ **하나의 조직으로 모인 기능: {', '.join(resolved)}**\n")
        add("\n잔존 중복 (개편 후에도 2개 이상 조직이 수행):\n")
        add("_기능 분류 사전이 아직 시드 상태라 이 목록은 과다 집계되어 있습니다. "
            "사전 확정 후 다시 봐야 의미가 있습니다._\n")
        add(_md(funcs["잔존 중복"], 15))

        add("\n### 제약 위반\n")
        add(_md(violations, 20))

    path = out_dir / "02_시뮬레이션리포트.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
