"""3개년 동역학: 조직 생애주기, 인원 이동, 이탈률, 안정성 점수."""
from __future__ import annotations

import pandas as pd


def org_transitions(members: pd.DataFrame, min_share: float = 0.20) -> pd.DataFrame:
    """연도 간 조직 승계 관계를 **구성원의 실제 이동**으로 추정한다.

    부서 코드가 조직의 정체성이 아니라 계층 내 위치 슬롯인 경우(코드 재사용·재배치),
    코드 매칭은 통합·개명·분할을 전혀 잡아내지 못한다. 사람의 흐름은 그런 코드 체계와
    무관하게 조직의 연속성을 드러낸다.
    """
    years = sorted(members["year"].unique())
    rows = []

    for prev_year, year in zip(years, years[1:]):
        prev = members[members["year"] == prev_year][["emp_id", "team_code", "team_name"]]
        curr = members[members["year"] == year][["emp_id", "team_code", "team_name"]]
        joined = prev.merge(curr, on="emp_id", how="outer", suffixes=("_from", "_to"))

        for code, group in joined[joined["team_code_from"].notna()].groupby("team_code_from"):
            size = len(group)
            stayed = group[group["team_code_to"].notna()]
            dist = stayed["team_code_to"].value_counts()
            successors = dist[dist / size >= min_share]

            top = dist.index[0] if len(dist) else None
            top_share = float(dist.iloc[0] / size) if len(dist) else 0.0
            name_from = group["team_name_from"].dropna().iloc[0]
            name_to = (stayed[stayed["team_code_to"] == top]["team_name_to"].iloc[0]
                       if top is not None else None)

            rows.append({
                "year_from": prev_year, "year_to": year,
                "team_code": code, "team_name": name_from,
                "headcount": size,
                "successor": name_to, "successor_code": top,
                "successor_share": round(top_share, 2),
                "successor_count": int(len(successors)),
                "left_company": int(size - len(stayed)),
                "status": _forward_status(code, top, top_share, len(successors), size, len(stayed)),
            })

    return pd.DataFrame(rows)


def _forward_status(code, top, share, successor_count, size, stayed) -> str:
    if stayed == 0:
        return "해체"
    if successor_count >= 2 and share < 0.60:
        return "분할"
    if top == code:
        return "유지"
    if share >= 0.50:
        return "통합/개명"
    return "분산흡수"


def new_orgs(members: pd.DataFrame, min_share: float = 0.20) -> pd.DataFrame:
    """해당 연도에 새로 나타난 조직이 어디서 왔는지 역추적한다."""
    years = sorted(members["year"].unique())
    rows = []

    for prev_year, year in zip(years, years[1:]):
        prev = members[members["year"] == prev_year][["emp_id", "team_code"]]
        curr = members[members["year"] == year][["emp_id", "team_code", "team_name"]]
        existing = set(prev["team_code"])
        joined = curr.merge(prev, on="emp_id", how="left", suffixes=("_to", "_from"))

        for code, group in joined[~joined["team_code_to"].isin(existing)].groupby("team_code_to"):
            size = len(group)
            came = group[group["team_code_from"].notna()]
            dist = came["team_code_from"].value_counts()
            top_share = float(dist.iloc[0] / size) if len(dist) else 0.0
            sources = dist[dist / size >= min_share]

            rows.append({
                "year": year, "team_code": code,
                "team_name": group["team_name"].iloc[0], "headcount": size,
                "from_outside": int(size - len(came)),
                "main_source": dist.index[0] if len(dist) else None,
                "main_source_share": round(top_share, 2),
                "source_count": int(len(sources)),
                "status": ("신규채용" if len(came) / size < 0.3
                           else ("분리" if top_share >= 0.5 else "혼합신설")),
            })

    return pd.DataFrame(rows)


def org_lifecycle(members: pd.DataFrame) -> pd.DataFrame:
    """조직별 3개년 변동 이력과 불안정 점수. 구성원 이동 기반 계보를 사용한다."""
    transitions = org_transitions(members)
    latest = int(members["year"].max())
    current = members[members["year"] == latest]

    rows = []
    for code, group in current.groupby("team_code"):
        name = group["team_name"].iloc[0]
        history = transitions[transitions["successor_code"] == code]
        own = transitions[transitions["team_code"] == code]

        absorbed = history[history["team_code"] != code]
        events = {
            "통합흡수": int((absorbed["status"] == "통합/개명").sum()),
            "분할경험": int((own["status"] == "분할").sum()),
            "분산흡수": int((absorbed["status"] == "분산흡수").sum()),
        }
        sizes = members[members["team_code"] == code].groupby("year").size()
        volatility = float(sizes.pct_change().abs().mean() or 0) if len(sizes) > 1 else 0.0

        rows.append({
            "team_code": code, "team_name": name,
            "headcount": len(group),
            "years_present": int(sizes.size),
            "absorbed_orgs": events["통합흡수"] + events["분산흡수"],
            "split_events": events["분할경험"],
            "headcount_volatility": round(volatility, 3),
            "instability_score": round(
                events["통합흡수"] * 3 + events["분산흡수"] * 2 + events["분할경험"] * 3
                + (3 - sizes.size) * 2 + volatility * 5, 1),
        })

    return pd.DataFrame(rows).sort_values("instability_score", ascending=False).reset_index(drop=True)


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
