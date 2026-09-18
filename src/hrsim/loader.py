"""3개년 구성원 명단과 팀 역할 정의표를 읽어 표준 스키마로 정규화한다."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import Config, ROLE_MEMBER

MEMBER_FIELDS = ["emp_id", "hq_code", "hq_name", "dept_code", "dept_name",
                 "team_code", "team_name", "hire_date", "gender", "age", "position"]


def _rename(df: pd.DataFrame, mapping: dict[str, str], optional: set[str]) -> pd.DataFrame:
    """설정의 {표준명: 실제헤더} 매핑으로 컬럼명을 표준화한다."""
    reverse = {}
    missing = []
    for std, actual in mapping.items():
        if actual in df.columns:
            reverse[actual] = std
        elif std in df.columns:      # 이미 표준명인 경우
            reverse[std] = std
        elif std not in optional:
            missing.append(f"{std}({actual})")
    if missing:
        raise ValueError(f"필수 컬럼 없음: {', '.join(missing)} / 실제 컬럼: {list(df.columns)}")
    out = df.rename(columns=reverse)
    return out[[c for c in mapping if c in out.columns]]


def load_members(cfg: Config) -> pd.DataFrame:
    """연도별 명단 파일을 모두 읽어 하나의 long 포맷으로 합친다."""
    spec = cfg.members
    files = sorted(cfg.data_dir.glob(spec["file_glob"]))
    if not files:
        raise FileNotFoundError(f"{cfg.data_dir}/{spec['file_glob']} 에 해당하는 파일이 없습니다.")

    optional = set(spec.get("optional_columns", []))
    frames = []
    for path in files:
        df = pd.read_csv(path, dtype=str).pipe(_rename, spec["columns"], optional)
        df["year"] = _year_of(path, df, spec)
        frames.append(df)

    members = pd.concat(frames, ignore_index=True)
    members["year"] = members["year"].astype(int)
    members["age"] = pd.to_numeric(members.get("age"), errors="coerce")
    members["hire_date"] = pd.to_datetime(members.get("hire_date"), errors="coerce")
    for col in MEMBER_FIELDS:
        if col in members.columns and members[col].dtype == object:
            members[col] = members[col].str.strip()

    members["position_role"] = members["position"].map(cfg.classify_position).fillna(ROLE_MEMBER)
    # 근속연수: 기준연도 말일 기준
    ref = pd.to_datetime(members["year"].astype(str) + "-12-31")
    members["tenure_years"] = ((ref - members["hire_date"]).dt.days / 365.25).round(1)
    return members.sort_values(["year", "hq_code", "dept_code", "team_code", "emp_id"]).reset_index(drop=True)


def _year_of(path: Path, df: pd.DataFrame, spec: dict) -> int:
    if spec.get("year_from_filename", True):
        match = re.search(r"(20\d{2})", path.stem)
        if match:
            return int(match.group(1))
    if "year" in df.columns:
        return int(df["year"].iloc[0])
    raise ValueError(f"{path.name} 에서 기준연도를 찾을 수 없습니다. 파일명에 연도를 넣거나 year 컬럼을 추가하세요.")


def load_roles(cfg: Config) -> pd.DataFrame:
    """팀 역할 정의표를 '팀 × 역할항목' 한 줄씩으로 분해한다."""
    spec = cfg.roles
    path = cfg.data_dir / spec["file"]
    if not path.exists():
        raise FileNotFoundError(f"역할 정의표가 없습니다: {path}")

    roles = pd.read_csv(path, dtype=str).pipe(
        _rename, spec["columns"], set(spec.get("optional_columns", [])))
    roles["role_item"] = roles["role_item"].fillna("")

    if spec.get("split_multiline", True):
        roles["role_item"] = roles["role_item"].str.split(r"[\n;·•]|^\s*-\s*", regex=True)
        roles = roles.explode("role_item")

    roles["role_item"] = roles["role_item"].str.strip().str.lstrip("-–—*・ ")
    roles = roles[roles["role_item"].str.len() > 1]
    return roles.reset_index(drop=True)


def latest_year(members: pd.DataFrame) -> int:
    return int(members["year"].max())


def snapshot(members: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    """특정 연도의 구성원 스냅샷."""
    year = year if year is not None else latest_year(members)
    return members[members["year"] == year].copy()
