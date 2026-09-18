"""도구 계약 — 모든 도구가 지켜야 할 반환 형식과 개인정보 경계.

두 가지를 구조로 강제한다.

1. **숫자만 반환하는 도구는 만들지 않는다.** 모든 결과에 근거와 가정을 함께 담는다.
   탐색 과정에서 분석 가정이 네 번 틀렸고, 그때마다 사람이 알려줘서 고쳤다.
   가정이 보이지 않으면 교정할 기회도 없다.

2. **개인 단위 데이터는 모델에 도달하지 않는다.** 프롬프트로 부탁하는 것이 아니라
   반환 직전에 검사해서 차단한다. 배포 환경(외부 LLM 전송 가능 여부)이 확정되지
   않았으므로, 경계를 코드로 그어 두면 어느 쪽으로 결론이 나도 이 계층은 유지된다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# 개인을 식별하거나 개인 단위 속성을 담은 컬럼. 모델에 전달하지 않는다.
BLOCKED_COLUMNS = {
    "emp_id", "사번", "이름", "성명", "연락처", "이메일",
    "age", "나이", "만나이", "생년월일", "gender", "성별",
    "hire_date", "입사일", "최초입사일", "SKB입사일", "그룹입사일",
    "position", "직책", "BAND", "직군",
}
# 사번 형태 (M278, 1428 등)
EMP_ID_PATTERN = re.compile(r"\b[A-Z]?\d{3,6}\b")


class PrivacyError(RuntimeError):
    """개인 단위 데이터가 모델 경계를 넘으려 할 때."""


@dataclass
class ToolResult:
    """도구의 표준 반환값.

    value       모델에 전달되는 집계 결과 (팀 단위 이상)
    basis       근거 — 원문 R&R 인용, 계산식, 데이터 출처
    assumptions 이 결과가 전제하는 것
    caveats     해석 시 주의할 점
    local_only  개인 단위 산출물. 로컬에만 저장하고 모델에는 건수만 알린다.
    """
    value: Any
    basis: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    local_only: pd.DataFrame | None = None

    def for_model(self, max_rows: int = 40) -> dict[str, Any]:
        """모델에 전달할 형태. 개인정보 게이트를 통과한 것만 나간다."""
        payload: dict[str, Any] = {}

        if isinstance(self.value, pd.DataFrame):
            safe = privacy_gate(self.value)
            payload["결과"] = safe.head(max_rows).to_dict("records")
            if len(safe) > max_rows:
                payload["표시"] = f"전체 {len(safe)}행 중 상위 {max_rows}행"
        elif isinstance(self.value, dict):
            payload["결과"] = {k: v for k, v in self.value.items()
                             if k not in BLOCKED_COLUMNS}
        else:
            payload["결과"] = self.value

        if self.basis:
            payload["근거"] = self.basis
        if self.assumptions:
            payload["가정"] = self.assumptions
        if self.caveats:
            payload["주의"] = self.caveats
        if self.local_only is not None:
            # 개인 명단은 건수만. 실물은 로컬 파일로만 나간다.
            payload["개인단위_산출물"] = {
                "건수": len(self.local_only),
                "안내": "개인 명단은 로컬 파일로만 출력됩니다. 모델은 건수만 확인합니다.",
            }
        return payload


def privacy_gate(frame: pd.DataFrame) -> pd.DataFrame:
    """개인 단위 데이터가 모델 경계를 넘으려 하면 막는다.

    개인 식별 컬럼을 조용히 삭제하지 않고 **거부한다.** 사번과 나이만 지운 행은
    여전히 한 사람의 기록이고, 익명처럼 보이기 때문에 더 위험하다.
    호출자가 집계하거나 ToolResult.local_only 로 넘기도록 강제한다.
    """
    present = [c for c in frame.columns if c in BLOCKED_COLUMNS]
    if present:
        raise PrivacyError(
            f"개인 단위 컬럼이 포함되어 있습니다: {', '.join(present)}. "
            "컬럼을 지우는 것으로는 부족합니다 — 행 자체가 개인 기록입니다. "
            "조직 단위로 집계하거나 ToolResult.local_only 로 넘기세요.")

    # 컬럼명을 우회한 사번이 값에 섞여 있는 경우를 잡는다
    for column in frame.select_dtypes("object").columns:
        sample = frame[column].dropna().astype(str).head(50)
        if len(sample) and sample.map(lambda v: bool(EMP_ID_PATTERN.fullmatch(v))).mean() > 0.5:
            raise PrivacyError(
                f"'{column}' 컬럼이 개인 식별자로 보입니다. "
                "개인 단위 데이터는 모델에 전달할 수 없습니다.")
    return frame


def assert_aggregated(frame: pd.DataFrame, unit: str = "team_code") -> None:
    """집계 단위가 조직 이상인지 확인한다. 행이 조직 수보다 많으면 개인 단위다."""
    if unit in frame.columns and frame[unit].duplicated().any():
        raise PrivacyError(
            f"'{unit}' 가 중복됩니다. 조직 단위로 집계되지 않은 결과입니다.")
