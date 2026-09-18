"""인력 구조: 연령/근속 분포, 승계 리스크, 자연감소 시뮬레이션."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config, ROLE_TEAM
from .dynamics import attrition_by_team

AGE_BINS = [0, 29, 34, 39, 44, 49, 54, 59, 200]
AGE_LABELS = ["~29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60+"]


def age_profile(snap: pd.DataFrame) -> pd.DataFrame:
    """팀별 연령/근속 구조."""
    snap = snap.copy()
    snap["age_band"] = pd.cut(snap["age"], bins=AGE_BINS, labels=AGE_LABELS)

    profile = snap.groupby(["hq_name", "dept_name", "team_code", "team_name"]).agg(
        headcount=("emp_id", "count"),
        avg_age=("age", "mean"),
        median_age=("age", "median"),
        avg_tenure=("tenure_years", "mean"),
        age_std=("age", "std"),
    ).round(1).reset_index()

    bands = (snap.groupby(["team_code", "age_band"], observed=False).size()
             .unstack(fill_value=0).reset_index())
    return profile.merge(bands, on="team_code", how="left")


def gender_profile(snap: pd.DataFrame) -> pd.DataFrame:
    """조직별 성별 구성. 개편 판단 근거가 아니라 다양성 현황 참고용으로만 사용한다."""
    if "gender" not in snap or snap["gender"].isna().all():
        return pd.DataFrame()

    overall = (snap.groupby(["hq_name", "team_code", "team_name", "gender"]).size()
               .unstack(fill_value=0))
    leaders = (snap[snap["position_role"] == ROLE_TEAM]
               .groupby(["team_code", "gender"]).size().unstack(fill_value=0)
               .add_prefix("leader_"))

    out = overall.join(leaders).fillna(0).reset_index()
    numeric = out.select_dtypes("number")
    total = numeric[[c for c in numeric.columns if not c.startswith("leader_")]].sum(axis=1)
    out["headcount"] = total
    return out


def succession_risk(snap: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """팀장 연령과 차하위 후보군 두께로 승계 리스크를 추정한다.

    주의: 직급 데이터가 없어 '팀 내 40세 이상 팀원'을 후보군 대리지표로 사용한다.
    직급이 확보되면 이 함수만 교체하면 된다.
    """
    retire = cfg.param("retirement_age", 60)
    rows = []

    for (code, name), team in snap.groupby(["team_code", "team_name"]):
        leaders = team[team["position_role"] == ROLE_TEAM]
        members = team[team["position_role"] != ROLE_TEAM]
        leader_age = float(leaders["age"].max()) if len(leaders) and leaders["age"].notna().any() else np.nan
        candidates = int((members["age"] >= 40).sum())
        years_left = retire - leader_age if not np.isnan(leader_age) else np.nan

        if len(leaders) == 0:
            risk, reason = "높음", "팀장 공석"
        elif not np.isnan(years_left) and years_left <= 5 and candidates == 0:
            risk, reason = "높음", f"팀장 정년 {int(years_left)}년 이내 · 후보군 없음"
        elif not np.isnan(years_left) and years_left <= 5:
            risk, reason = "중간", f"팀장 정년 {int(years_left)}년 이내 · 후보군 {candidates}명"
        elif candidates == 0 and len(members) >= 3:
            risk, reason = "중간", "후보군 없음"
        else:
            risk, reason = "낮음", ""

        rows.append({"team_code": code, "team_name": name, "headcount": len(team),
                     "leader_age": leader_age, "years_to_retire": years_left,
                     "candidate_pool": candidates, "risk": risk, "reason": reason})

    out = pd.DataFrame(rows)
    order = {"높음": 0, "중간": 1, "낮음": 2}
    return out.sort_values(["risk", "years_to_retire"], key=lambda s: s.map(order) if s.name == "risk" else s
                           ).reset_index(drop=True)


def natural_decline(members: pd.DataFrame, cfg: Config, horizon: int = 5) -> pd.DataFrame:
    """개편 없이 정년과 과거 이탈률만 적용했을 때의 조직별 인원 추이.

    채용을 0으로 가정한 하한선이다. '손대지 않아도 이렇게 된다'를 보여주는 용도.
    """
    retire = cfg.param("retirement_age", 60)
    base_year = int(members["year"].max())
    snap = members[members["year"] == base_year]
    # 개명된 팀은 team_code 기준으로 하나로 합친다
    rates = (attrition_by_team(members).groupby("team_code")["avg_attrition_pct"].mean()) / 100
    default_rate = float(rates.median()) if len(rates) else 0.05

    rows = []
    for (code, name), team in snap.groupby(["team_code", "team_name"]):
        rate = float(rates.get(code, default_rate))
        ages = team["age"].dropna().to_numpy()
        headcount = float(len(team))
        record = {"team_code": code, "team_name": name,
                  "attrition_rate_pct": round(rate * 100, 1), f"Y{base_year}": int(headcount)}

        for step in range(1, horizon + 1):
            # 정년 도달 인원은 확정 감소, 나머지는 과거 이탈률로 감소
            retiring = int(((ages + step > retire) & (ages + step - 1 <= retire)).sum())
            headcount = max(headcount - retiring, 0) * (1 - rate)
            record[f"Y{base_year + step}"] = round(headcount, 1)

        record["change_pct"] = round(
            (record[f"Y{base_year + horizon}"] / max(len(team), 1) - 1) * 100, 1)
        rows.append(record)

    out = pd.DataFrame(rows)
    return out.sort_values("change_pct").reset_index(drop=True)


def retirement_wave(members: pd.DataFrame, cfg: Config, horizon: int = 5) -> pd.DataFrame:
    """연도별 정년 도달 인원 (리더 여부 구분)."""
    retire = cfg.param("retirement_age", 60)
    base_year = int(members["year"].max())
    snap = members[members["year"] == base_year].copy()
    snap["retire_year"] = base_year + (retire - snap["age"])

    upcoming = snap[snap["retire_year"].between(base_year + 1, base_year + horizon)]
    if upcoming.empty:
        return pd.DataFrame(columns=["retire_year", "total", "leaders"])

    out = upcoming.groupby("retire_year").agg(
        total=("emp_id", "count"),
        leaders=("position_role", lambda s: int((s != "member").sum())),
    ).reset_index()
    out["retire_year"] = out["retire_year"].astype(int)
    return out
