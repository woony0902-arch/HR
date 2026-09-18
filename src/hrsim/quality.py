"""데이터 무결성 점검. 분석 전에 반드시 통과시켜야 할 항목들."""
from __future__ import annotations

import pandas as pd

from .config import Config, ROLE_TEAM

Finding = dict[str, object]


def _add(out: list[Finding], level: str, code: str, message: str, detail: pd.DataFrame | None = None) -> None:
    out.append({"level": level, "code": code, "message": message,
                "count": 0 if detail is None else len(detail), "detail": detail})


def check(members: pd.DataFrame, roles: pd.DataFrame, cfg: Config) -> list[Finding]:
    out: list[Finding] = []
    _check_keys(members, out)
    _check_hierarchy(members, out)
    _check_leaders(members, cfg, out)
    _check_person_fields(members, out)
    _check_roles_coverage(members, roles, out)
    return out


def _check_keys(members: pd.DataFrame, out: list[Finding]) -> None:
    dup = members[members.duplicated(["year", "emp_id"], keep=False)]
    if len(dup):
        _add(out, "ERROR", "DUP_EMP",
             "같은 연도에 사번이 중복됩니다. 겸직이라면 주소속만 남겨야 합니다.",
             dup[["year", "emp_id", "team_name"]])

    blank = members[members["emp_id"].isna() | (members["emp_id"] == "")]
    if len(blank):
        _add(out, "ERROR", "NULL_EMP", "사번이 비어 있는 행이 있습니다.", blank[["year", "team_name"]])

    # 사번이 연도 간 재사용되면 이동/이탈 추적이 전부 틀어진다
    spans = members.groupby("emp_id")["year"].agg(["min", "max", "nunique"])
    gaps = spans[(spans["max"] - spans["min"] + 1) != spans["nunique"]]
    if len(gaps):
        _add(out, "WARN", "EMP_GAP",
             "중간 연도가 비었다가 다시 나타나는 사번입니다. 휴직 복귀인지 사번 재사용인지 확인이 필요합니다.",
             gaps.reset_index())


def _check_hierarchy(members: pd.DataFrame, out: list[Finding]) -> None:
    for code_col, name_col, label in [("hq_code", "hq_name", "본부"),
                                      ("dept_code", "dept_name", "담당"),
                                      ("team_code", "team_name", "팀")]:
        # 같은 코드에 이름이 여러 개 → 개명. 추적은 가능하지만 리포트에 표기해야 한다
        renamed = (members.groupby(code_col)[name_col].nunique()
                   .loc[lambda s: s > 1].reset_index(name="name_count"))
        if len(renamed):
            _add(out, "INFO", f"RENAME_{code_col.upper()}",
                 f"{label} 코드는 같은데 명칭이 바뀐 사례입니다(개명으로 처리).", renamed)

        # 같은 이름에 코드가 여러 개 → 코드 체계가 흔들린 것. 이쪽이 위험하다
        split = (members.groupby(name_col)[code_col].nunique()
                 .loc[lambda s: s > 1].reset_index(name="code_count"))
        if len(split):
            _add(out, "WARN", f"SPLITCODE_{code_col.upper()}",
                 f"동일 {label}명이 서로 다른 코드를 가집니다. 연도 간 동일 조직으로 볼지 확인이 필요합니다.", split)

    # 한 팀이 여러 담당에 걸쳐 있으면 이관이거나 데이터 오류
    moved = (members.groupby(["year", "team_code"])["dept_code"].nunique()
             .loc[lambda s: s > 1].reset_index(name="dept_count"))
    if len(moved):
        _add(out, "ERROR", "TEAM_MULTI_PARENT",
             "같은 연도에 한 팀이 두 개 이상의 담당에 속해 있습니다.", moved)


def _check_leaders(members: pd.DataFrame, cfg: Config, out: list[Finding]) -> None:
    current = members[members["year"] == members["year"].max()]
    leaders = (current[current["position_role"] == ROLE_TEAM]
               .groupby("team_code").size().rename("leader_count"))
    teams = current[["team_code", "team_name"]].drop_duplicates().set_index("team_code")

    joined = teams.join(leaders).fillna({"leader_count": 0})
    none = joined[joined["leader_count"] == 0]
    if len(none):
        _add(out, "WARN", "TEAM_NO_LEADER",
             "팀장이 식별되지 않는 팀입니다. 공석이거나 직책 표기가 다를 수 있습니다.", none.reset_index())

    many = joined[joined["leader_count"] > 1]
    if len(many):
        _add(out, "WARN", "TEAM_MULTI_LEADER", "팀장이 2명 이상으로 식별된 팀입니다.", many.reset_index())

    unmapped = (current[current["position_role"] == "member"]["position"]
                .dropna().loc[lambda s: s != ""].value_counts())
    if len(unmapped):
        _add(out, "INFO", "POSITION_UNMAPPED",
             "팀원으로 분류된 직책값 목록입니다. 리더 직책이 섞여 있으면 config/columns.yaml 의 positions 를 보완하세요.",
             unmapped.reset_index())


def _check_person_fields(members: pd.DataFrame, out: list[Finding]) -> None:
    if "age" in members and members["age"].notna().any():
        bad = members[(members["age"] < 18) | (members["age"] > 75)]
        if len(bad):
            _add(out, "WARN", "AGE_RANGE", "나이가 비정상 범위입니다.", bad[["year", "emp_id", "age"]])

        # 나이와 입사일이 어긋나면 둘 중 하나가 잘못된 것
        entry_age = members["age"] - (members["year"] - members["hire_date"].dt.year)
        bad_entry = members[entry_age < 15]
        if len(bad_entry):
            _add(out, "WARN", "AGE_HIRE_CONFLICT",
                 "입사 시점 나이가 15세 미만으로 계산됩니다. 나이 또는 입사일 확인이 필요합니다.",
                 bad_entry[["year", "emp_id", "age", "hire_date"]])

    future = members[members["hire_date"] > pd.Timestamp(members["year"].max(), 12, 31)]
    if len(future):
        _add(out, "ERROR", "HIRE_FUTURE", "입사일이 기준연도 이후입니다.", future[["year", "emp_id", "hire_date"]])

    # 연도별 인원이 급변하면 스냅샷 기준일이 다를 가능성
    counts = members.groupby("year").size()
    if len(counts) > 1:
        change = counts.pct_change().abs().dropna()
        if (change > 0.30).any():
            _add(out, "WARN", "HEADCOUNT_JUMP",
                 "연도 간 전사 인원이 30% 이상 변동했습니다. 스냅샷 기준일이나 집계 범위가 동일한지 확인하세요.",
                 counts.reset_index(name="headcount"))


def _check_roles_coverage(members: pd.DataFrame, roles: pd.DataFrame, out: list[Finding]) -> None:
    current = members[members["year"] == members["year"].max()]
    in_members = set(current["team_code"])
    in_roles = set(roles["team_code"])

    missing = sorted(in_members - in_roles)
    if missing:
        detail = current[current["team_code"].isin(missing)][["team_code", "team_name"]].drop_duplicates()
        _add(out, "WARN", "ROLE_MISSING", "명단에는 있으나 역할 정의표에 없는 팀입니다.", detail)

    orphan = sorted(in_roles - in_members)
    if orphan:
        detail = roles[roles["team_code"].isin(orphan)][["team_code", "team_name"]].drop_duplicates()
        _add(out, "INFO", "ROLE_ORPHAN", "역할 정의표에는 있으나 현재 명단에 없는 팀입니다.", detail)

    vague = roles[roles["role_item"].str.fullmatch(r".{0,6}(지원|관리|운영|업무|관련.*)", na=False)]
    if len(vague):
        _add(out, "INFO", "ROLE_VAGUE",
             "역할 서술이 모호합니다('~지원', '~관련 업무' 등). 중복 판정 정확도를 떨어뜨립니다.",
             vague[["team_code", "team_name", "role_item"]])


def summary(findings: list[Finding]) -> pd.DataFrame:
    return pd.DataFrame([{k: f[k] for k in ("level", "code", "count", "message")} for f in findings])


def has_blocking(findings: list[Finding]) -> bool:
    return any(f["level"] == "ERROR" for f in findings)
