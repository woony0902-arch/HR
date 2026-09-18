"""조직 구조 지표: 계층, span of control, 조직 규모 분포, 리더 비율."""
from __future__ import annotations

import pandas as pd

from .config import Config, ROLE_DEPT, ROLE_HQ, ROLE_MEMBER, ROLE_TEAM


def team_profile(snap: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """팀 단위 기본 지표. 이후 모든 분석의 기준 테이블."""
    small = cfg.param("small_team_threshold", 4)
    large = cfg.param("large_team_threshold", 20)

    grouped = snap.groupby(["hq_code", "hq_name", "dept_code", "dept_name", "team_code", "team_name"])
    profile = grouped.agg(
        headcount=("emp_id", "count"),
        leaders=("position_role", lambda s: int((s == ROLE_TEAM).sum())),
        avg_age=("age", "mean"),
        avg_tenure=("tenure_years", "mean"),
    ).reset_index()

    # 팀장을 뺀 인원이 실제 관리 대상
    profile["span"] = profile["headcount"] - profile["leaders"]
    profile["avg_age"] = profile["avg_age"].round(1)
    profile["avg_tenure"] = profile["avg_tenure"].round(1)
    profile["size_flag"] = pd.cut(profile["headcount"],
                                  bins=[-1, small - 1, large, 10**9],
                                  labels=["소규모", "적정", "과대"])
    return profile.sort_values(["hq_name", "dept_name", "team_name"]).reset_index(drop=True)


def span_table(snap: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """계층별 조직장 1인당 관리 대상 수."""
    rows = []
    teams = team_profile(snap, cfg)
    for _, t in teams.iterrows():
        rows.append({"level": "팀", "org_code": t["team_code"], "org_name": t["team_name"],
                     "leader_count": t["leaders"], "reports": t["span"]})

    # 담당장은 산하 팀장을, 본부장은 산하 담당장을 관리한다고 본다
    for level, code_col, name_col, role, child_role, child_code in [
        ("담당", "dept_code", "dept_name", ROLE_DEPT, ROLE_TEAM, "team_code"),
        ("본부", "hq_code", "hq_name", ROLE_HQ, ROLE_DEPT, "dept_code"),
    ]:
        for (code, name), group in snap.groupby([code_col, name_col]):
            leaders = int((group["position_role"] == role).sum())
            reports = int((group["position_role"] == child_role).sum())
            if reports == 0:  # 하위 조직장이 없으면 직할 인원으로 대체
                reports = len(group) - leaders
            rows.append({"level": level, "org_code": code, "org_name": name,
                         "leader_count": leaders, "reports": reports})

    table = pd.DataFrame(rows)
    lo, hi = cfg.param("span_min", 3), cfg.param("span_max", 12)
    table["span_flag"] = table["reports"].apply(
        lambda n: "과소" if n < lo else ("과대" if n > hi else "적정"))
    return table


def headline(snap: pd.DataFrame, cfg: Config) -> dict[str, float | int]:
    """전사 요약 지표 한 묶음. 시뮬레이션 전후 비교에 그대로 쓰인다."""
    teams = team_profile(snap, cfg)
    spans = teams.loc[teams["span"] > 0, "span"]
    leaders = snap["position_role"].isin([ROLE_HQ, ROLE_DEPT, ROLE_TEAM])

    return {
        "headcount": int(len(snap)),
        "hq_count": int(snap["hq_code"].nunique()),
        "dept_count": int(snap["dept_code"].nunique()),
        "team_count": int(snap["team_code"].nunique()),
        "leader_count": int(leaders.sum()),
        "leader_ratio_pct": round(leaders.mean() * 100, 1),
        "avg_team_size": round(teams["headcount"].mean(), 1),
        "median_span": float(spans.median()) if len(spans) else 0.0,
        "max_span": int(spans.max()) if len(spans) else 0,
        "small_teams": int((teams["size_flag"] == "소규모").sum()),
        "large_teams": int((teams["size_flag"] == "과대").sum()),
        "teams_without_leader": int((teams["leaders"] == 0).sum()),
        "avg_age": round(snap["age"].mean(), 1) if snap["age"].notna().any() else 0.0,
        "avg_tenure": round(snap["tenure_years"].mean(), 1),
    }


def hq_comparison(snap: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """본부별 구조 비교. 편차가 큰 본부가 개편 1순위 검토 대상."""
    teams = team_profile(snap, cfg)
    by_hq = teams.groupby(["hq_code", "hq_name"]).agg(
        dept_count=("dept_code", "nunique"),
        team_count=("team_code", "nunique"),
        headcount=("headcount", "sum"),
        avg_team_size=("headcount", "mean"),
        small_teams=("size_flag", lambda s: int((s == "소규모").sum())),
        avg_age=("avg_age", "mean"),
    ).reset_index()

    leaders = (snap[snap["position_role"] != ROLE_MEMBER]
               .groupby("hq_code").size().rename("leader_count"))
    by_hq = by_hq.merge(leaders, on="hq_code", how="left").fillna({"leader_count": 0})
    by_hq["leader_ratio_pct"] = (by_hq["leader_count"] / by_hq["headcount"] * 100).round(1)
    by_hq["avg_team_size"] = by_hq["avg_team_size"].round(1)
    by_hq["avg_age"] = by_hq["avg_age"].round(1)
    return by_hq.sort_values("headcount", ascending=False).reset_index(drop=True)
