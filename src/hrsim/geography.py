"""조직의 근무 지역 추론과 지역 구조 분석.

전국에 국사·사옥이 있는 회사에서는 조직 통합이 곧 인력 집중이 아니다.
대구구축팀과 부산구축팀을 합쳐도 사람은 대구와 부산에 그대로 남는다.
따라서 지역이 다른 조직의 통합은 '관리 단위 통합'이지 '효율화'가 아니며,
남은 조직장은 물리적으로 떨어진 인력을 관리하게 된다.

주의: 여기서는 조직명으로 지역을 추론한다. 근무지(사업장/국사) 데이터가
확보되면 infer_region() 을 실제 값 조회로 교체해야 한다.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

UNKNOWN = "미상"


class RegionMap:
    def __init__(self, spec: dict):
        self.zone_of: dict[str, str] = {}
        for zone, tokens in spec.get("권역", {}).items():
            for token in tokens:
                self.zone_of[token] = zone
        self.directional = list(spec.get("방향권역", []))
        self.nationwide = list(spec.get("전국조직", []))
        # 긴 토큰을 먼저 검사해야 '경기'가 '경기남부'를 가로채지 않는다
        self.tokens = sorted(self.zone_of, key=len, reverse=True)

    def infer(self, name: str) -> tuple[str, str]:
        """조직명 → (지역, 권역). 지역을 못 찾으면 (미상, 미상)."""
        text = str(name or "")
        if any(word in text for word in self.nationwide):
            return "전국", "전국"
        for token in self.tokens:
            if token in text:
                return token, self.zone_of[token]
        for token in self.directional:
            if text.startswith(token):
                return token, f"{token}권역"
        return UNKNOWN, UNKNOWN


def load_regions(path: str | Path = "config/regions.yaml") -> RegionMap:
    return RegionMap(yaml.safe_load(Path(path).read_text(encoding="utf-8")))


def annotate(snap: pd.DataFrame, regions: RegionMap) -> pd.DataFrame:
    out = snap.copy()
    inferred = out["team_name"].map(regions.infer)
    out["region"] = [r[0] for r in inferred]
    out["zone"] = [r[1] for r in inferred]
    return out


def region_profile(snap: pd.DataFrame, regions: RegionMap) -> pd.DataFrame:
    """권역별 조직·인원 현황."""
    tagged = annotate(snap, regions)
    local = tagged[tagged["zone"] != UNKNOWN]
    if local.empty:
        return pd.DataFrame()

    out = local.groupby("zone").agg(
        팀수=("team_code", "nunique"),
        인원=("emp_id", "count"),
        평균연령=("age", "mean"),
        팀장수=("position_role", lambda s: int((s == "team_leader").sum())),
    ).round(1).reset_index().rename(columns={"zone": "권역"})
    out["팀당인원"] = (out["인원"] / out["팀수"]).round(1)
    return out.sort_values("인원", ascending=False).reset_index(drop=True)


def region_function_grid(snap: pd.DataFrame, regions: RegionMap,
                         min_teams: int = 2) -> pd.DataFrame:
    """권역 × 기능 격자. 같은 기능을 몇 개 권역에서 몇 개 팀으로 수행하는지.

    지역 조직의 통폐합은 '어느 팀과 어느 팀' 이 아니라
    '이 기능을 전국 몇 개 단위로 운영할 것인가' 의 문제다.
    """
    tagged = annotate(snap, regions)
    local = tagged[tagged["zone"] != UNKNOWN].copy()
    if local.empty:
        return pd.DataFrame()

    # 조직명에서 지역 표기를 떼어 낸 나머지를 기능명으로 본다
    local["function"] = [
        name.replace(region, "") if region != UNKNOWN else name
        for name, region in zip(local["team_name"], local["region"])]

    grid = local.groupby("function").agg(
        권역수=("zone", "nunique"),
        팀수=("team_code", "nunique"),
        인원=("emp_id", "count"),
        평균연령=("age", "mean"),
        권역=("zone", lambda s: ", ".join(sorted(set(s)))),
    ).round(1).reset_index().rename(columns={"function": "기능"})
    grid = grid[grid["팀수"] >= min_teams]
    return grid.sort_values(["팀수", "인원"], ascending=False).reset_index(drop=True)
