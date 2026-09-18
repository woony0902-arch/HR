"""3개년 동역학: 조직 생애주기, 인원 이동, 이탈률, 안정성 점수."""
from __future__ import annotations

import pandas as pd


def org_lifecycle(members: pd.DataFrame) -> pd.DataFrame:
    """팀 코드별 등장/소멸/개명/이관 이력."""
    years = sorted(members["year"].unique())
    first, last = years[0], years[-1]

    rows = []
    for code, group in members.groupby("team_code"):
        present = sorted(group["year"].unique())
        names = group.sort_values("year")["team_name"]
        parents = group.sort_values("year").groupby("year")["dept_code"].first()

        renamed = names.nunique() - 1
        transferred = int((parents != parents.shift()).sum() - 1)
        headcounts = group.groupby("year").size()

        rows.append({
            "team_code": code,
            "team_name": names.iloc[-1],
            "first_year": present[0],
            "last_year": present[-1],
            "years_present": len(present),
            "created": present[0] > first,        # 첫 해에 없었으면 신설
            "abolished": present[-1] < last,      # 마지막 해에 없으면 폐지
            "rename_count": int(max(renamed, 0)),
            "transfer_count": max(transferred, 0),
            "headcount_first": int(headcounts.iloc[0]),
            "headcount_last": int(headcounts.iloc[-1]),
            "headcount_volatility": round(float(headcounts.pct_change().abs().mean() or 0), 3),
        })

    life = pd.DataFrame(rows)
    # 변경 이벤트가 잦을수록 기능 정의가 불안정한 조직으로 본다
    life["instability_score"] = (
        life["rename_count"] * 2
        + life["transfer_count"] * 3
        + life["created"].astype(int) * 2
        + life["abolished"].astype(int) * 3
        + (life["headcount_volatility"] * 5).round(1)
    ).round(1)
    return life.sort_values("instability_score", ascending=False).reset_index(drop=True)


def movement(members: pd.DataFrame) -> pd.DataFrame:
    """연도 간 개인의 소속 변화를 추적해 이동/신규/이탈/잔류로 분류한다."""
    frames = []
    years = sorted(members["year"].unique())
    keys = ["emp_id", "hq_code", "hq_name", "dept_code", "team_code", "team_name"]

    for prev_year, year in zip(years, years[1:]):
        prev = members[members["year"] == prev_year][keys]
        curr = members[members["year"] == year][keys]
        merged = prev.merge(curr, on="emp_id", how="outer", suffixes=("_from", "_to"), indicator=True)

        merged["year_from"], merged["year_to"] = prev_year, year
        merged["status"] = merged["_merge"].astype(str).map({
            "left_only": "이탈", "right_only": "신규", "both": "잔류"})
        moved = (merged["_merge"] == "both") & (merged["team_code_from"] != merged["team_code_to"])
        merged.loc[moved, "status"] = "이동"
        frames.append(merged.drop(columns="_merge"))

    return pd.concat(frames, ignore_index=True)


def attrition(members: pd.DataFrame) -> pd.DataFrame:
    """조직별 이탈률. 명단에서 사라진 것이므로 퇴사/전출을 구분하지 않는다."""
    moves = movement(members)
    left = moves[moves["status"] == "이탈"]

    base = (members.groupby(["year", "team_code", "team_name"]).size()
            .rename("headcount").reset_index())
    leavers = (left.groupby(["year_from", "team_code_from"]).size()
               .rename("leavers").reset_index()
               .rename(columns={"year_from": "year", "team_code_from": "team_code"}))

    table = base.merge(leavers, on=["year", "team_code"], how="left").fillna({"leavers": 0})
    table["attrition_pct"] = (table["leavers"] / table["headcount"] * 100).round(1)
    # 마지막 연도는 다음 해 비교 대상이 없어 이탈률을 계산할 수 없다
    return table[table["year"] < members["year"].max()].reset_index(drop=True)


def attrition_by_team(members: pd.DataFrame) -> pd.DataFrame:
    """팀별 평균 이탈률 (자연감소 시뮬레이션의 입력)."""
    table = attrition(members)
    out = (table.groupby(["team_code", "team_name"])
           .agg(avg_attrition_pct=("attrition_pct", "mean"),
                total_leavers=("leavers", "sum"),
                avg_headcount=("headcount", "mean"))
           .reset_index())
    out["avg_attrition_pct"] = out["avg_attrition_pct"].round(1)
    out["avg_headcount"] = out["avg_headcount"].round(1)
    return out.sort_values("avg_attrition_pct", ascending=False).reset_index(drop=True)


def flow_matrix(members: pd.DataFrame, min_flow: int = 2) -> pd.DataFrame:
    """조직 간 인원 이동 매트릭스. 이동이 잦은 조직쌍은 업무 경계가 모호하다는 신호."""
    moves = movement(members)
    internal = moves[moves["status"] == "이동"]

    flows = (internal.groupby(["team_code_from", "team_name_from", "team_code_to", "team_name_to"])
             .size().rename("people").reset_index())
    flows = flows[flows["people"] >= min_flow]

    # 양방향으로 오간 쌍은 조직 경계 재검토 대상
    pair_key = flows.apply(lambda r: tuple(sorted([r["team_code_from"], r["team_code_to"]])), axis=1)
    flows["pair"] = pair_key
    bidirectional = flows["pair"].duplicated(keep=False)
    flows["bidirectional"] = bidirectional
    return flows.drop(columns="pair").sort_values("people", ascending=False).reset_index(drop=True)


def yearly_summary(members: pd.DataFrame) -> pd.DataFrame:
    """연도별 전사 추이."""
    moves = movement(members)
    counts = moves.groupby(["year_to", "status"]).size().unstack(fill_value=0)

    base = members.groupby("year").agg(
        headcount=("emp_id", "count"),
        teams=("team_code", "nunique"),
        depts=("dept_code", "nunique"),
        hqs=("hq_code", "nunique"),
        avg_age=("age", "mean"),
        avg_tenure=("tenure_years", "mean"),
    ).round(1)
    return base.join(counts, how="left").fillna(0).reset_index()
