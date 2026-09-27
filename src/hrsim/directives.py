"""지침 원장 — 그룹 가이드, 경영진 의견, 시장 환경, HR 철학을 분석의 전제로 넣는다.

판정 원장이 '이 조직에 대해 이렇게 판단했다'를 담는다면,
지침 원장은 '모든 판단이 이 전제 위에서 이뤄진다'를 담는다.

세 가지로 나눠 다르게 쓴다.
  규칙  기계가 검사한다.   → 시뮬레이션 제약 검사에 자동 반영
  방향  해설이 참고한다.   → 무엇을 먼저 말할지, 어떤 렌즈로 볼지
  전제  배경으로 밝힌다.   → 리포트 머리에 "이 결과의 전제"
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

RULE, DIRECTION, PREMISE = "규칙", "방향", "전제"


@dataclass
class Directive:
    id: str
    source: str
    kind: str
    force: str
    text: str
    scope: str = "전사"
    rule: dict[str, Any] = field(default_factory=dict)
    priority_functions: list[str] = field(default_factory=list)
    lens: str = ""
    valid_until: str = ""

    def applies_to(self, hq_name: str | None) -> bool:
        return self.scope == "전사" or (hq_name is not None and self.scope == hq_name)


class Directives:
    def __init__(self, items: list[Directive]):
        self.items = items

    @classmethod
    def load(cls, path: str | Path = "config/directives.yaml") -> "Directives":
        path = Path(path)
        if not path.exists():
            return cls([])
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls([Directive(**d) for d in raw.get("directives", [])])

    def rules(self) -> list[Directive]:
        return [d for d in self.items if d.kind == RULE]

    def directions(self) -> list[Directive]:
        return [d for d in self.items if d.kind == DIRECTION]

    def premises(self) -> list[Directive]:
        return [d for d in self.items if d.kind == PREMISE]

    # ------------------------------------------------ 규칙 → 검사 파라미터
    def param_overrides(self) -> dict[str, tuple[Any, str]]:
        """config 기본값을 덮어쓸 파라미터와 그 출처. 강제가 권고를 이긴다."""
        out: dict[str, tuple[Any, str]] = {}
        mapping = {"min_team_size": "small_team_threshold", "max_span": "span_max",
                   "max_depth": "max_depth", "leader_ratio_max": "leader_ratio_max"}
        for d in sorted(self.rules(), key=lambda d: d.force != "강제"):   # 강제 먼저
            for key, param in mapping.items():
                if key in d.rule and param not in out:
                    out[param] = (d.rule[key], d.id)
        return out

    def protected_orgs(self) -> dict[str, str]:
        """통합·폐지 금지 조직 → 지침 id."""
        out = {}
        for d in self.rules():
            for name in d.rule.get("protect", []):
                out.setdefault(name, d.id)
        return out

    def required_functions(self) -> dict[str, str]:
        out = {}
        for d in self.rules():
            for name in d.rule.get("require_function", []):
                out.setdefault(name, d.id)
        return out

    def priority_functions(self) -> dict[str, str]:
        out = {}
        for d in self.directions():
            for name in d.priority_functions:
                out.setdefault(name, d.id)
        return out

    def lenses(self) -> list[tuple[str, str]]:
        return [(d.lens, d.id) for d in self.directions() if d.lens]

    # ------------------------------------------------ 충돌 탐지
    def conflicts(self) -> pd.DataFrame:
        """같은 파라미터에 다른 값을 요구하는 지침. 그룹과 CEO가 다르게 말하면 여기 걸린다."""
        seen: dict[str, list[tuple[Any, Directive]]] = {}
        for d in self.rules():
            for key, value in d.rule.items():
                if key in ("protect", "require_function"):
                    continue
                seen.setdefault(key, []).append((value, d))
        rows = []
        for key, entries in seen.items():
            values = {v for v, _ in entries}
            if len(values) > 1:
                rows.append({"항목": key,
                             "지침": " vs ".join(f"{d.id}({d.source}, {d.force})={v}" for v, d in entries),
                             "처리": "강제 우선, 같은 효력이면 더 엄격한 값"})
        return pd.DataFrame(rows)

    def to_frame(self) -> pd.DataFrame:
        rows = [{"id": d.id, "출처": d.source, "구분": d.kind, "효력": d.force,
                 "범위": d.scope, "내용": d.text,
                 "규칙": ", ".join(f"{k}={v}" for k, v in d.rule.items()) if d.rule else ""}
                for d in self.items]
        return pd.DataFrame(rows)


def check_plan_against(directives: Directives, plan_actions: list[dict],
                       before: pd.DataFrame) -> pd.DataFrame:
    """개편안이 보호 조직을 건드리는지, 실행 전에 확인한다."""
    protected = directives.protected_orgs()
    if not protected:
        return pd.DataFrame()
    names = before.drop_duplicates("team_code").set_index("team_code")["team_name"].to_dict()
    hq_of = before.drop_duplicates("team_code").set_index("team_code")["hq_name"].to_dict()
    rows = []
    for action in plan_actions:
        targets = action.get("targets") or ([action["target"]] if "target" in action else [])
        for code in targets:
            for label in (names.get(code, code), hq_of.get(code, "")):
                if label in protected:
                    rows.append({"액션": action.get("op"), "대상": names.get(code, code),
                                 "보호조직": label, "지침": protected[label],
                                 "내용": "통합·폐지 금지 대상입니다"})
    return pd.DataFrame(rows)
