"""설정 로드 및 조직 계층 상수."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# 조직 계층: 본부(사업부) > 담당 > 팀
LEVELS = ["hq", "dept", "team"]
LEVEL_LABEL = {"hq": "본부/사업부", "dept": "담당", "team": "팀"}
# 직책 역할
ROLE_HQ, ROLE_DEPT, ROLE_TEAM, ROLE_MEMBER = "hq_leader", "dept_leader", "team_leader", "member"
LEADER_ROLES = (ROLE_HQ, ROLE_DEPT, ROLE_TEAM)


@dataclass
class Config:
    raw: dict[str, Any]
    data_dir: Path
    out_dir: Path

    @property
    def members(self) -> dict[str, Any]:
        return self.raw["members"]

    @property
    def roles(self) -> dict[str, Any]:
        return self.raw["roles"]

    @property
    def positions(self) -> dict[str, list[str]]:
        return self.raw.get("positions", {})

    @property
    def params(self) -> dict[str, Any]:
        return self.raw.get("params", {})

    def param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def classify_position(self, position: str | None) -> str:
        """직책 문자열을 계층 역할로 분류한다. 더 상위 역할을 우선 매칭."""
        if not position:
            return ROLE_MEMBER
        text = str(position).strip()
        for role in (ROLE_HQ, ROLE_DEPT, ROLE_TEAM):
            for keyword in self.positions.get(role, []):
                if keyword and keyword in text:
                    return role
        return ROLE_MEMBER


def load_config(path: str | Path = "config/columns.yaml",
                data_dir: str | Path = "data",
                out_dir: str | Path = "output") -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    return Config(raw=raw, data_dir=Path(data_dir), out_dir=out)
