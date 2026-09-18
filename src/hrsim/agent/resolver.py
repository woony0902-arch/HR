"""조직명 해석기 — 모델이 조직 이름을 지어내지 못하게 막는다.

"대구랑 부산 구축팀 합쳐줘" 에서 실제 조직으로 가는 길은 결정론적이어야 한다.
모델이 조직명을 생성하면 존재하지 않는 팀을 대상으로 시뮬레이션이 돌거나,
비슷한 이름의 다른 조직이 통합된다. 조직 개편에서 이런 오류는 회복이 어렵다.

그래서 해석은 LLM이 아니라 이 모듈이 한다. 모호하면 답을 고르지 않고 되묻는다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd

OK, AMBIGUOUS, NOT_FOUND = "ok", "ambiguous", "not_found"


@dataclass
class Resolution:
    status: str
    query: str
    matches: list[dict] = field(default_factory=list)
    question: str = ""

    @property
    def code(self) -> str | None:
        return self.matches[0]["team_code"] if self.status == OK else None


def _normalize(text: str) -> str:
    return re.sub(r"[\s\-_/·]", "", str(text or "")).lower()


class OrgResolver:
    """현재 스냅샷의 실제 조직 목록에 대해서만 이름을 해석한다."""

    def __init__(self, snap: pd.DataFrame):
        teams = (snap.groupby(["team_code", "team_name"])
                 .agg(headcount=("emp_id", "count"),
                      hq_name=("hq_name", "first"),
                      dept_name=("dept_name", "first"))
                 .reset_index())
        self.teams = teams
        self._by_norm: dict[str, list[dict]] = {}
        for row in teams.to_dict("records"):
            self._by_norm.setdefault(_normalize(row["team_name"]), []).append(row)

    def resolve(self, query: str, limit: int = 8) -> Resolution:
        """조직명 하나를 해석한다. 확실할 때만 코드를 돌려준다."""
        text = _normalize(query)
        if not text:
            return Resolution(NOT_FOUND, query, question="조직명이 비어 있습니다.")

        exact = self._by_norm.get(text, [])
        if len(exact) == 1:
            return Resolution(OK, query, exact)
        if len(exact) > 1:
            return Resolution(AMBIGUOUS, query, exact,
                              self._ask(query, exact))

        partial = [row for norm, rows in self._by_norm.items() if text in norm
                   for row in rows]
        if not partial:
            # 역방향: 질의가 조직명보다 긴 경우 ("수도권 구축팀" → "수북구축팀")
            partial = [row for norm, rows in self._by_norm.items() if norm in text
                       for row in rows]

        if not partial:
            near = self._suggest(text)
            return Resolution(NOT_FOUND, query, near,
                              f"'{query}' 에 해당하는 조직을 찾지 못했습니다."
                              + (f" 혹시 이 중 하나인가요? {self._names(near)}" if near else ""))
        if len(partial) == 1:
            return Resolution(OK, query, partial)
        return Resolution(AMBIGUOUS, query, partial[:limit], self._ask(query, partial[:limit]))

    def resolve_all(self, queries: list[str]) -> tuple[list[str], list[Resolution]]:
        """여러 조직을 한 번에. 하나라도 해석되지 않으면 진행하지 않는다."""
        codes, problems = [], []
        for query in queries:
            result = self.resolve(query)
            (codes.append(result.code) if result.status == OK else problems.append(result))
        return codes, problems

    def _suggest(self, text: str, limit: int = 5) -> list[dict]:
        """접미사가 같은 조직을 후보로 제시한다 ('구축팀' → 구축팀 전체)."""
        for length in (4, 3, 2):
            if len(text) <= length:
                continue
            tail = text[-length:]
            hits = [row for norm, rows in self._by_norm.items() if norm.endswith(tail)
                    for row in rows]
            if hits:
                return sorted(hits, key=lambda r: -r["headcount"])[:limit]
        return []

    @staticmethod
    def _names(rows: list[dict]) -> str:
        return ", ".join(r["team_name"] for r in rows)

    def _ask(self, query: str, rows: list[dict]) -> str:
        listing = "\n".join(
            f"  - {r['team_name']} ({r['hq_name']} · {r['dept_name']} · {r['headcount']}명)"
            for r in rows)
        return f"'{query}' 에 해당하는 조직이 {len(rows)}개입니다. 어느 것을 말씀하시나요?\n{listing}"
