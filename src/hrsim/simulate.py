"""개편 시뮬레이션 엔진.

개편안을 통합/분리/이관/폐지/신설 액션의 조합으로 표현하고,
현재 스냅샷에 적용해 Before → After 델타를 계산한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .config import Config, ROLE_MEMBER, ROLE_TEAM
from . import structure

OPS = {"MERGE", "SPLIT", "MOVE", "ABOLISH", "CREATE", "RENAME"}


@dataclass
class Plan:
    name: str
    actions: list[dict[str, Any]]
    description: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Plan":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(name=raw.get("name", Path(path).stem),
                   actions=raw.get("actions", []),
                   description=raw.get("description", ""))

    def validate(self, snap: pd.DataFrame) -> list[str]:
        known = set(snap["team_code"])
        errors = []
        for i, action in enumerate(self.actions, 1):
            op = action.get("op", "").upper()
            if op not in OPS:
                errors.append(f"[{i}] 알 수 없는 액션: {action.get('op')}")
                continue
            targets = action.get("targets") or ([action["target"]] if "target" in action else [])
            for code in targets:
                if code not in known:
                    errors.append(f"[{i}] {op}: 존재하지 않는 팀코드 {code}")
            if op == "ABOLISH" and action.get("reassign_to") and action["reassign_to"] not in known:
                errors.append(f"[{i}] ABOLISH: 재배치 대상 {action['reassign_to']} 이(가) 없습니다")
        return errors


@dataclass
class Result:
    plan: Plan
    before: pd.DataFrame
    after: pd.DataFrame
    changelog: list[dict[str, Any]] = field(default_factory=list)


def apply_plan(snap: pd.DataFrame, plan: Plan, cfg: Config) -> Result:
    """액션을 순서대로 적용한다. 각 액션은 직전 결과 위에서 수행된다."""
    after = snap.copy()
    changelog: list[dict[str, Any]] = []

    for action in plan.actions:
        op = action["op"].upper()
        handler = {"MERGE": _merge, "SPLIT": _split, "MOVE": _move,
                   "ABOLISH": _abolish, "CREATE": _create, "RENAME": _rename}[op]
        after, note = handler(after, action, cfg)
        changelog.append(note)

    return Result(plan=plan, before=snap, after=after, changelog=changelog)


def _org_attrs(snap: pd.DataFrame, dept_code: str) -> dict[str, str]:
    """담당 코드로부터 상위 본부 정보를 가져온다."""
    row = snap[snap["dept_code"] == dept_code]
    if row.empty:
        return {}
    first = row.iloc[0]
    return {"hq_code": first["hq_code"], "hq_name": first["hq_name"],
            "dept_code": dept_code, "dept_name": first["dept_name"]}


def _merge(snap: pd.DataFrame, action: dict, cfg: Config) -> tuple[pd.DataFrame, dict]:
    targets = action["targets"]
    into = action["into"]
    out = snap.copy()
    mask = out["team_code"].isin(targets)

    parent = into.get("dept_code") or out.loc[mask, "dept_code"].mode().iloc[0]
    attrs = _org_attrs(out, parent)

    # 팀장 중 1명만 유지하고 나머지는 보임 해제 대상으로 표시한다
    leaders = out[mask & (out["position_role"] == ROLE_TEAM)]
    keep_team = action.get("keep_leader")
    keep_id = None
    if len(leaders):
        chosen = leaders[leaders["team_code"] == keep_team] if keep_team else leaders
        chosen = chosen if len(chosen) else leaders
        keep_id = chosen.sort_values("tenure_years", ascending=False)["emp_id"].iloc[0]

    surplus = [e for e in leaders["emp_id"] if e != keep_id]
    out.loc[out["emp_id"].isin(surplus), ["position_role", "position"]] = [ROLE_MEMBER, "(보임해제)"]

    out.loc[mask, ["team_code", "team_name"]] = [into["code"], into["name"]]
    for key, value in attrs.items():
        out.loc[mask, key] = value

    return out, {"op": "MERGE", "detail": f"{', '.join(targets)} → {into['name']}",
                 "affected": int(mask.sum()), "surplus_leaders": surplus}


def _split(snap: pd.DataFrame, action: dict, cfg: Config) -> tuple[pd.DataFrame, dict]:
    target = action["target"]
    parts = action["into"]
    out = snap.copy()
    members = out[out["team_code"] == target].sort_values(["position_role", "tenure_years"],
                                                          ascending=[True, False])
    ids = members["emp_id"].tolist()

    # 명시적 배정이 없으면 ratio 로 순차 배분
    assigned: dict[str, list[str]] = {}
    remaining = list(ids)
    for part in parts:
        if part.get("members"):
            picked = [e for e in part["members"] if e in remaining]
        else:
            count = round(len(ids) * float(part.get("ratio", 1 / len(parts))))
            picked = remaining[:count]
        assigned[part["code"]] = picked
        remaining = [e for e in remaining if e not in picked]
    if remaining:  # 잔여 인원은 첫 번째 조직으로
        assigned[parts[0]["code"]].extend(remaining)

    for part in parts:
        mask = out["emp_id"].isin(assigned[part["code"]]) & (out["team_code"] == target)
        out.loc[mask, ["team_code", "team_name"]] = [part["code"], part["name"]]
        if part.get("dept_code"):
            for key, value in _org_attrs(snap, part["dept_code"]).items():
                out.loc[mask, key] = value

    # 분리된 조직에는 팀장이 새로 필요하다
    needed = sum(1 for p in parts
                 if not (out[(out["team_code"] == p["code"]) &
                             (out["position_role"] == ROLE_TEAM)]).shape[0])
    return out, {"op": "SPLIT",
                 "detail": f"{target} → {', '.join(p['name'] for p in parts)}",
                 "affected": len(ids), "new_leader_needed": needed}


def _move(snap: pd.DataFrame, action: dict, cfg: Config) -> tuple[pd.DataFrame, dict]:
    target, to_dept = action["target"], action["to_dept"]
    out = snap.copy()
    mask = out["team_code"] == target
    attrs = _org_attrs(out, to_dept)
    if not attrs:
        attrs = {"dept_code": to_dept, "dept_name": action.get("to_dept_name", to_dept),
                 "hq_code": action.get("to_hq_code", out.loc[mask, "hq_code"].iloc[0]),
                 "hq_name": action.get("to_hq_name", out.loc[mask, "hq_name"].iloc[0])}
    for key, value in attrs.items():
        out.loc[mask, key] = value
    return out, {"op": "MOVE", "detail": f"{target} → {attrs.get('dept_name', to_dept)} 산하",
                 "affected": int(mask.sum())}


def _abolish(snap: pd.DataFrame, action: dict, cfg: Config) -> tuple[pd.DataFrame, dict]:
    target = action["target"]
    out = snap.copy()
    mask = out["team_code"] == target
    affected = int(mask.sum())
    reassign = action.get("reassign_to")

    if not reassign:  # 인원 재배치처를 지정하지 않으면 미배치 인원으로 남긴다
        out.loc[mask, ["team_code", "team_name"]] = ["UNASSIGNED", "(미배치)"]
        return out, {"op": "ABOLISH", "detail": f"{target} 폐지 (재배치처 미지정)",
                     "affected": affected, "unassigned": affected}

    dest = out[out["team_code"] == reassign].iloc[0]
    leaders = out[mask & (out["position_role"] == ROLE_TEAM)]["emp_id"].tolist()
    out.loc[out["emp_id"].isin(leaders), ["position_role", "position"]] = [ROLE_MEMBER, "(보임해제)"]
    for key in ["hq_code", "hq_name", "dept_code", "dept_name", "team_code", "team_name"]:
        out.loc[mask, key] = dest[key]

    return out, {"op": "ABOLISH", "detail": f"{target} 폐지 → {dest['team_name']} 흡수",
                 "affected": affected, "surplus_leaders": leaders}


def _create(snap: pd.DataFrame, action: dict, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """기존 조직에서 인원을 차출해 신설 조직을 만든다."""
    out = snap.copy()
    attrs = _org_attrs(out, action["dept_code"])
    picked: list[str] = []

    for source in action.get("from", []):
        pool = out[out["team_code"] == source["team_code"]].sort_values("tenure_years", ascending=False)
        if source.get("members"):
            picked += [e for e in source["members"] if e in set(pool["emp_id"])]
        else:
            picked += pool["emp_id"].head(int(source.get("count", 0))).tolist()

    mask = out["emp_id"].isin(picked)
    out.loc[mask, ["team_code", "team_name"]] = [action["code"], action["name"]]
    for key, value in attrs.items():
        out.loc[mask, key] = value

    return out, {"op": "CREATE", "detail": f"{action['name']} 신설 ({len(picked)}명 차출)",
                 "affected": len(picked),
                 "new_leader_needed": int(not (out[mask]["position_role"] == ROLE_TEAM).any())}


def _rename(snap: pd.DataFrame, action: dict, cfg: Config) -> tuple[pd.DataFrame, dict]:
    out = snap.copy()
    mask = out["team_code"] == action["target"]
    out.loc[mask, "team_name"] = action["name"]
    return out, {"op": "RENAME", "detail": f"{action['target']} → {action['name']}",
                 "affected": int(mask.sum())}


# ---------------------------------------------------------------- 영향 분석

def metric_delta(result: Result, cfg: Config) -> pd.DataFrame:
    """전사 구조 지표의 Before → After."""
    before = structure.headline(result.before, cfg)
    after = structure.headline(result.after, cfg)

    labels = {
        "headcount": "총 인원", "hq_count": "본부 수", "dept_count": "담당 수",
        "team_count": "팀 수", "leader_count": "직책자 수", "leader_ratio_pct": "직책자 비율(%)",
        "avg_team_size": "평균 팀 규모", "median_span": "중위 span", "max_span": "최대 span",
        "small_teams": "소규모 팀 수", "large_teams": "과대 팀 수",
        "teams_without_leader": "팀장 공석 팀", "avg_age": "평균 연령", "avg_tenure": "평균 근속",
    }
    rows = [{"지표": labels[k], "Before": before[k], "After": after[k],
             "델타": round(after[k] - before[k], 1)} for k in labels]
    return pd.DataFrame(rows)


def people_impact(result: Result) -> dict[str, Any]:
    """소속 변경·보임 해제 등 사람에게 미치는 영향."""
    before = result.before.set_index("emp_id")
    after = result.after.set_index("emp_id")
    common = before.index.intersection(after.index)

    team_changed = common[before.loc[common, "team_code"] != after.loc[common, "team_code"]]
    hq_changed = common[before.loc[common, "hq_code"] != after.loc[common, "hq_code"]]
    demoted = common[(before.loc[common, "position_role"] == ROLE_TEAM)
                     & (after.loc[common, "position_role"] != ROLE_TEAM)]

    new_leader_needed = sum(int(n.get("new_leader_needed", 0)) for n in result.changelog)
    unassigned = int((result.after["team_code"] == "UNASSIGNED").sum())

    return {
        "소속 변경 인원": len(team_changed),
        "본부 이동 인원": len(hq_changed),
        "보임 해제 인원": len(demoted),
        "보임 해제 명단": after.loc[demoted].reset_index()[["emp_id", "team_name"]]
                        .to_dict("records") if len(demoted) else [],
        "신규 팀장 필요": new_leader_needed,
        "미배치 인원": unassigned,
    }


def team_lineage(result: Result) -> dict[str, set[str]]:
    """개편 전 팀코드 → 개편 후 그 인원이 속한 팀코드 집합.

    통합·분리로 코드가 바뀌어도 기능을 승계 조직으로 따라가게 하기 위한 매핑이다.
    구성원의 실제 이동을 근거로 하므로 액션 종류와 무관하게 동작한다.
    """
    before = result.before[["emp_id", "team_code"]]
    after = result.after[["emp_id", "team_code"]]
    merged = before.merge(after, on="emp_id", suffixes=("_before", "_after"))
    merged = merged[merged["team_code_after"] != "UNASSIGNED"]
    return (merged.groupby("team_code_before")["team_code_after"]
            .apply(set).to_dict())


def function_impact(result: Result, tagged: pd.DataFrame) -> dict[str, Any]:
    """개편 후 기능 결손과 잔존 중복.

    기능은 조직 계보를 따라 승계된다. 승계 조직이 하나도 없을 때만 '결손'으로 본다.
    """
    if tagged.empty:
        return {"기능 결손": [], "잔존 중복": pd.DataFrame()}

    lineage = team_lineage(result)
    before_teams = set(result.before["team_code"])
    scoped = tagged[tagged["team_code"].isin(before_teams)]
    names = dict(zip(tagged["function_code"], tagged["function_name"]))

    rows = []
    for row in scoped.itertuples():
        for successor in lineage.get(row.team_code, set()):
            rows.append({"function_code": row.function_code,
                         "function_name": row.function_name,
                         "team_code": successor})
    carried = pd.DataFrame(rows).drop_duplicates() if rows else pd.DataFrame(
        columns=["function_code", "function_name", "team_code"])

    lost = set(scoped["function_code"]) - set(carried["function_code"])

    after_names = (result.after.drop_duplicates("team_code")
                   .set_index("team_code")["team_name"].to_dict())
    remaining = (carried.groupby(["function_code", "function_name"])
                 .agg(team_count=("team_code", "nunique"),
                      teams=("team_code", lambda s: ", ".join(
                          sorted(after_names.get(c, c) for c in set(s)))))
                 .reset_index())
    remaining = remaining[remaining["team_count"] > 1].sort_values(
        "team_count", ascending=False).reset_index(drop=True)

    resolved = _resolved_duplicates(scoped, carried, names)
    return {"기능 결손": sorted(names.get(f, f) for f in lost),
            "잔존 중복": remaining,
            "해소된 중복": resolved}


def _resolved_duplicates(scoped: pd.DataFrame, carried: pd.DataFrame,
                         names: dict[str, str]) -> list[str]:
    """개편 전에는 2개 이상 조직이 수행했으나 개편 후 하나로 모인 기능."""
    before_counts = scoped.groupby("function_code")["team_code"].nunique()
    after_counts = (carried.groupby("function_code")["team_code"].nunique()
                    if len(carried) else pd.Series(dtype=int))
    resolved = [names.get(f, f) for f, n in before_counts.items()
                if n > 1 and after_counts.get(f, 0) == 1]
    return sorted(resolved)


def _violations_of(snap: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    small = cfg.param("small_team_threshold", 4)
    span_max = cfg.param("span_max", 12)
    active = snap[snap["team_code"] != "UNASSIGNED"]
    rows = []

    for (code, name), n in active.groupby(["team_code", "team_name"]).size().items():
        if n < small:
            rows.append({"유형": "최소 인원 미달", "대상": f"{name}({code})",
                         "내용": f"{n}명 (기준 {small}명)", "key": f"size:{code}"})

    for row in structure.span_table(active, cfg).itertuples():
        if row.reports > span_max:
            rows.append({"유형": "span 초과", "대상": f"{row.org_name}({row.org_code})",
                         "내용": f"{row.reports}명 (기준 {span_max}명)", "key": f"span:{row.org_code}"})

    leaders = active.groupby(["team_code", "team_name"])["position_role"].apply(
        lambda s: int((s == ROLE_TEAM).sum()))
    for (code, name), n in leaders[leaders == 0].items():
        rows.append({"유형": "팀장 공석", "대상": f"{name}({code})",
                     "내용": "팀장 없음", "key": f"leader:{code}"})

    unassigned = int((snap["team_code"] == "UNASSIGNED").sum())
    if unassigned:
        rows.append({"유형": "미배치 인원", "대상": "-", "내용": f"{unassigned}명", "key": "unassigned"})

    return pd.DataFrame(rows, columns=["유형", "대상", "내용", "key"])


def changed_org_profile(result: Result, cfg: Config) -> pd.DataFrame:
    """개편으로 새로 만들어진 조직의 인원·연령 구조.

    통합은 사람을 합치는 일이므로, 합친 결과가 어떤 연령 구조가 되는지가
    조직 수 감소보다 중요할 때가 많다. 정년 도래가 몰린 조직끼리 합치면
    몇 년 뒤 그 조직이 통째로 비게 된다.
    """
    retire = cfg.param("retirement_age", 60)
    before_teams = set(result.before["team_code"])
    after = result.after[result.after["team_code"] != "UNASSIGNED"]

    rows = []
    for (code, name), team in after.groupby(["team_code", "team_name"]):
        if code in before_teams and len(team) == int(
                (result.before["team_code"] == code).sum()):
            continue                              # 변화 없는 조직은 건너뛴다
        ages = team["age"].dropna()
        ages = ages[ages > 0]
        if ages.empty:
            continue
        rows.append({
            "조직": name, "인원": len(team),
            "평균연령": round(float(ages.mean()), 1),
            "50세이상": int((ages >= 50).sum()),
            "5년내정년": int((ages >= retire - 5).sum()),
            "5년내정년율(%)": round(float((ages >= retire - 5).mean() * 100), 1),
            "팀장수": int((team["position_role"] == ROLE_TEAM).sum()),
        })
    return (pd.DataFrame(rows).sort_values("5년내정년율(%)", ascending=False)
            .reset_index(drop=True) if rows else pd.DataFrame())


def constraint_check(result: Result, cfg: Config) -> pd.DataFrame:
    """개편안의 규정·기준 위반. 개편으로 새로 생긴 것과 기존 문제를 구분한다."""
    before = set(_violations_of(result.before, cfg)["key"])
    after = _violations_of(result.after, cfg)
    if after.empty:
        return after.drop(columns="key")

    after["발생"] = after["key"].apply(lambda k: "기존" if k in before else "신규")
    return (after.drop(columns="key")
            .sort_values("발생", ascending=False)   # 신규가 먼저
            .reset_index(drop=True))


def new_violation_count(result: Result, cfg: Config) -> int:
    table = constraint_check(result, cfg)
    return 0 if table.empty else int((table["발생"] == "신규").sum())


def compare_plans(snap: pd.DataFrame, plans: list[Plan], cfg: Config) -> pd.DataFrame:
    """여러 시나리오를 한 표로 비교."""
    base = structure.headline(snap, cfg)
    rows = [{"시나리오": "현행(AS-IS)", **{k: base[k] for k in
             ["team_count", "dept_count", "leader_count", "avg_team_size",
              "median_span", "small_teams"]},
             "소속변경": 0, "보임해제": 0, "신규위반": 0}]

    for plan in plans:
        result = apply_plan(snap, plan, cfg)
        after = structure.headline(result.after, cfg)
        impact = people_impact(result)
        rows.append({
            "시나리오": plan.name,
            **{k: after[k] for k in ["team_count", "dept_count", "leader_count",
                                     "avg_team_size", "median_span", "small_teams"]},
            "소속변경": impact["소속 변경 인원"],
            "보임해제": impact["보임 해제 인원"],
            "신규위반": new_violation_count(result, cfg),
        })

    out = pd.DataFrame(rows)
    return out.rename(columns={"team_count": "팀 수", "dept_count": "담당 수",
                               "leader_count": "직책자", "avg_team_size": "평균팀규모",
                               "median_span": "중위span", "small_teams": "소규모팀"})
