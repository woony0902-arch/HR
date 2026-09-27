"""상위 미션과 하위 R&R의 정합성, R&R의 3개년 궤적, 조직장 교체.

담당·본부급 R&R이 확보되면서 가능해진 분석이다.
본부 미션 → 담당 R&R → 팀 R&R 로 내려가며 끊긴 곳을 찾는다.
  - 담당이 선언했으나 어느 팀도 맡지 않은 항목 = 미션 공백
  - 팀이 수행하지만 담당 R&R 에 없는 항목      = 비공식 업무
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .functions import tokenize

ITEM_SPLIT = re.compile(r"[\n○◦•]|(?:^|\s)[-–]\s")


def _items(text) -> list[str]:
    parts = ITEM_SPLIT.split(str(text or ""))
    return [p.strip(" -–—·\t") for p in parts if len(p.strip(" -–—·\t")) > 3]


def _sim(a: str, b: str) -> float:
    """짧은 문장 둘의 토큰 Dice 유사도."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return 2 * len(ta & tb) / (len(ta) + len(tb))


def mission_alignment(upper: pd.DataFrame, roles: pd.DataFrame,
                      threshold: float = 0.30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """담당별 미션 커버리지와, 커버되지 않은 항목 목록.

    upper : rr_upper_{year}.csv  (상위조직·조직명·미션·주요RR)
    roles : team_roles.csv       (팀명·상위조직·구분·역할)
    """
    team_items = (roles[roles["구분"].isin(["미션", "주요"])]
                  .groupby("상위조직")["역할"].apply(list).to_dict())
    team_count = roles.groupby("상위조직")["팀명"].nunique().to_dict()

    summary, gaps = [], []
    for row in upper.itertuples():
        declared = _items(row.주요RR)
        children = team_items.get(row.조직명, [])
        if not declared:
            continue
        covered = 0
        for item in declared:
            best = max((_sim(item, c) for c in children), default=0.0)
            if best >= threshold:
                covered += 1
            else:
                gaps.append({"본부": row.상위조직, "담당": row.조직명,
                             "담당 R&R 항목": item[:80], "최고유사도": round(best, 2),
                             "하위팀수": team_count.get(row.조직명, 0)})
        summary.append({"본부": row.상위조직, "담당": row.조직명,
                        "선언항목": len(declared), "커버": covered,
                        "커버율(%)": round(covered / len(declared) * 100, 1),
                        "하위팀수": team_count.get(row.조직명, 0)})

    out = pd.DataFrame(summary).sort_values(["커버율(%)", "선언항목"]).reset_index(drop=True)
    return out, pd.DataFrame(gaps)


def orphan_team_work(upper: pd.DataFrame, roles: pd.DataFrame,
                     threshold: float = 0.25) -> pd.DataFrame:
    """팀이 '주요 R&R'로 수행하지만 상위 담당 R&R 어디에도 닿지 않는 항목."""
    declared = {row.조직명: _items(row.주요RR) + _items(row.미션) for row in upper.itertuples()}
    rows = []
    for row in roles[roles["구분"] == "주요"].itertuples():
        parent_items = declared.get(row.상위조직)
        if not parent_items:
            continue
        best = max(_sim(row.역할, p) for p in parent_items)
        if best < threshold:
            rows.append({"담당": row.상위조직, "팀": row.팀명,
                         "팀 R&R 항목": row.역할[:80], "최고유사도": round(best, 2)})
    return pd.DataFrame(rows)


def orphan_summary(upper: pd.DataFrame, roles: pd.DataFrame,
                   threshold: float = 0.25) -> pd.DataFrame:
    """담당별로 팀 주요 R&R 중 상위 R&R 에 닿지 않는 비율. 상위 R&R 이 하위를 얼마나 포괄하는가."""
    orphan = orphan_team_work(upper, roles, threshold)
    total = roles[roles["구분"] == "주요"].groupby("상위조직").size().rename("팀항목")
    if orphan.empty:
        return pd.DataFrame()
    missing = orphan.groupby("담당").size().rename("미포괄")
    out = pd.concat([total, missing], axis=1).fillna(0).reset_index().rename(columns={"index": "담당", "상위조직": "담당"})
    out = out[out["팀항목"] > 0]
    out["미포괄율(%)"] = (out["미포괄"] / out["팀항목"] * 100).round(1)
    out["미포괄"] = out["미포괄"].astype(int)
    return out.sort_values("미포괄율(%)", ascending=False).reset_index(drop=True)


def rr_trajectory(team_files: dict[int, Path]) -> pd.DataFrame:
    """같은 이름의 팀에 대해 연도 간 R&R 변화 정도. 낮을수록 역할이 크게 바뀐 것."""
    frames = {y: pd.read_csv(p, dtype=str).fillna("") for y, p in team_files.items()}
    years = sorted(frames)
    rows = []
    for prev, curr in zip(years, years[1:]):
        a = frames[prev].set_index("조직명")
        b = frames[curr].set_index("조직명")
        common = a.index.intersection(b.index)
        for name in common:
            ta = " ".join([a.loc[name, "미션"], a.loc[name, "주요RR"]]) if isinstance(a.loc[name], pd.Series) else ""
            tb = " ".join([b.loc[name, "미션"], b.loc[name, "주요RR"]]) if isinstance(b.loc[name], pd.Series) else ""
            if not ta or not tb:
                continue
            rows.append({"팀": name, "구간": f"{prev}→{curr}",
                         "R&R유사도": round(_sim(ta, tb), 2),
                         "상위조직(전)": a.loc[name, "상위조직"], "상위조직(후)": b.loc[name, "상위조직"]})
        rows_summary = {"구간": f"{prev}→{curr}", "팀": "(집계)", "R&R유사도": None,
                        "상위조직(전)": f"공통 {len(common)}", "상위조직(후)":
                        f"신규 {len(b.index.difference(a.index))} / 소멸 {len(a.index.difference(b.index))}"}
        rows.append(rows_summary)
    return pd.DataFrame(rows)


def leader_turnover(leaders: pd.DataFrame) -> pd.DataFrame:
    """조직명이 같은데 조직장 사번이 바뀐 경우와, 조직장 겸직."""
    years = sorted(leaders["연도"].unique())
    rows = []
    for prev, curr in zip(years, years[1:]):
        a = leaders[leaders["연도"] == prev].drop_duplicates("조직").set_index("조직")["사번"]
        b = leaders[leaders["연도"] == curr].drop_duplicates("조직").set_index("조직")["사번"]
        common = a.index.intersection(b.index)
        changed = int((a[common] != b[common]).sum())
        gone = len(a.index.difference(b.index))
        new = len(b.index.difference(a.index))
        dual = int(leaders[leaders["연도"] == curr]["사번"].duplicated(keep=False).sum())
        rows.append({"구간": f"{prev}→{curr}", "공통조직": len(common), "조직장교체": changed,
                     "교체율(%)": round(changed / max(len(common), 1) * 100, 1),
                     "소멸조직": gone, "신설조직": new, "겸직보임(후)": dual})
    return pd.DataFrame(rows)
