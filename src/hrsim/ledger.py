"""판정 원장 — 사람이 내린 조직 판단을 기록하고 다음 분석에 반영한다.

기계는 후보를 계산할 수 있지만 그것이 진짜 중복인지, 의도된 설계인지는 판단할 수 없다.
그 판단은 맥락을 아는 사람에게서 나오고, 기록되지 않으면 매년 같은 논쟁이 반복된다.

원장에 개인 정보는 담지 않는다. 조직 단위 판단과 그 사유만 기록한다.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable

import pandas as pd

# 판정 종류
KEEP = "유지"              # 손대지 않는다
MERGE = "통합검토"          # 통합 대상으로 본다
REDEFINE = "R&R재정의"      # 조직은 두되 역할을 다시 긋는다
RENAME = "명칭변경"         # 이름만 바꾼다
DEFER = "보류"             # 판단을 미룬다 (사유 필수)
VERDICTS = (KEEP, MERGE, REDEFINE, RENAME, DEFER)

FIELDS = ["id", "kind", "key", "label", "machine_verdict", "verdict",
          "reason", "decided_by", "decided_on", "valid_until", "status"]


@dataclass
class Decision:
    kind: str                  # 중복후보 / 병렬조직군 / 조직명 / 구조
    key: str                   # 정규화된 대상 키
    label: str                 # 사람이 읽는 대상 표시
    verdict: str               # VERDICTS 중 하나
    reason: str                # 왜 그렇게 판단했는가 (필수)
    decided_by: str
    machine_verdict: str = ""
    decided_on: str = field(default_factory=lambda: date.today().isoformat())
    valid_until: str = ""      # 비우면 다음 개편까지 유효
    status: str = "유효"        # 유효 / 철회
    id: str = ""

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError(f"판정은 {VERDICTS} 중 하나여야 합니다: {self.verdict}")
        if not self.reason.strip():
            raise ValueError("판정에는 사유가 필요합니다. 사유 없는 기록은 다음 해에 쓸모가 없습니다.")
        if not self.id:
            self.id = f"{self.kind}:{self.key}:{self.decided_on}"


def pair_key(a: str, b: str) -> str:
    """조직 쌍의 순서에 무관한 키. 'A↔B'와 'B↔A'를 같은 건으로 본다."""
    return " ↔ ".join(sorted([str(a).strip(), str(b).strip()]))


def group_key(members: Iterable[str]) -> str:
    return " + ".join(sorted(str(m).strip() for m in members))


class Ledger:
    def __init__(self, path: str | Path = "data/ledger/decisions.csv"):
        self.path = Path(path)
        self.records: list[Decision] = []
        if self.path.exists():
            self._load()

    def _load(self) -> None:
        with self.path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                row.pop("id", None)
                self.records.append(Decision(**{k: v for k, v in row.items() if k in FIELDS or k != "id"}))

    def save(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            for record in self.records:
                writer.writerow(asdict(record))
        return self.path

    def record(self, decision: Decision, supersede: bool = True) -> Decision:
        """판정을 기록한다. 같은 대상의 이전 판정은 기본적으로 철회 처리한다."""
        if supersede:
            for existing in self.records:
                if existing.key == decision.key and existing.kind == decision.kind:
                    existing.status = "철회"
        self.records.append(decision)
        self.save()
        return decision

    def lookup(self, key: str, kind: str | None = None) -> Decision | None:
        """해당 대상의 유효한 최신 판정."""
        matches = [r for r in self.records
                   if r.key == key and r.status == "유효"
                   and (kind is None or r.kind == kind)]
        return max(matches, key=lambda r: r.decided_on) if matches else None

    def to_frame(self, only_active: bool = True) -> pd.DataFrame:
        rows = [asdict(r) for r in self.records if not only_active or r.status == "유효"]
        if not rows:
            return pd.DataFrame(columns=FIELDS)
        frame = pd.DataFrame(rows)
        return frame.sort_values("decided_on", ascending=False).reset_index(drop=True)

    # ------------------------------------------------------------ 분석 반영

    def annotate_duplicates(self, duplicates: pd.DataFrame) -> pd.DataFrame:
        """중복 후보에 기존 판정을 붙인다."""
        if duplicates.empty:
            return duplicates
        out = duplicates.copy()
        keys = [pair_key(a, b) for a, b in zip(out["team_a"], out["team_b"])]
        decisions = [self.lookup(k, "중복후보") for k in keys]
        out["판정"] = [d.verdict if d else "" for d in decisions]
        out["판정사유"] = [d.reason if d else "" for d in decisions]
        out["판정일"] = [d.decided_on if d else "" for d in decisions]
        return out

    def pending_duplicates(self, duplicates: pd.DataFrame) -> pd.DataFrame:
        """아직 판정하지 않은 후보만. 논의 시간을 여기에 쓰면 된다."""
        annotated = self.annotate_duplicates(duplicates)
        if annotated.empty:
            return annotated
        return annotated[annotated["판정"] == ""].reset_index(drop=True)

    def summary(self) -> pd.DataFrame:
        frame = self.to_frame()
        if frame.empty:
            return pd.DataFrame(columns=["유형", "판정", "건수"])
        return (frame.groupby(["kind", "verdict"]).size()
                .reset_index(name="건수")
                .rename(columns={"kind": "유형", "verdict": "판정"}))
