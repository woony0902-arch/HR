#!/usr/bin/env python3
"""조직 설계 에이전트 — 초안 (v0).

  python3 agent.py                 # 대화 시작 (ANTHROPIC_API_KEY 필요)
  python3 agent.py --selftest      # 모델 없이 도구 계층만 끝까지 점검
  python3 agent.py --data data_demo  # 합성 데이터로 시연

구성 (docs/agent-architecture.md)
  ① 해석기 · ④ 해설기  = 모델.  조직명을 지어내지 않고 resolve_org 결과만 쓴다.
  ② 해석·검증기 · ③ 실행기 = 아래 도구 함수. 모델은 숫자를 계산하지 않는다.
  ⑤ 기록기 = propose → 사용자 확인 → record. 모델이 임의로 원장에 쓰지 않는다.
개인 단위 데이터는 ToolResult.for_model() 의 privacy_gate 를 넘지 못한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import pandas as pd

from hrsim import (directives as directives_mod, dynamics, functions, geography,
                   ledger as ledger_mod, loader, mission, simulate, structure, workforce)
from hrsim.agent import OrgResolver, ToolResult
from hrsim.agent.resolver import OK
from hrsim.config import load_config

MODEL = "claude-opus-5"


# ============================================================ 상태
class State:
    def __init__(self, config: str, data: str, ledger: str, directives: str):
        self.cfg = load_config(config, data, "output")
        self.members = loader.load_members(self.cfg)
        self.roles = loader.load_roles(self.cfg)
        self.snap = loader.snapshot(self.members)
        self.year = loader.latest_year(self.members)
        self.resolver = OrgResolver(self.snap)
        self.ledger = ledger_mod.Ledger(ledger)
        self.directives = directives_mod.Directives.load(directives)
        self.regions = geography.load_regions(Path(config).parent / "regions.yaml")
        func_dict = functions.load_function_dict(Path(config).parent / "function_dict.yaml")
        self.tagged = functions.tag_functions(self.roles, func_dict, self.cfg.roles.get("tagging_types"))
        self.families = functions.parallel_families(self.roles, self.snap)
        self.transitions = dynamics.org_transitions(self.members)
        self.scenario: list[dict] = []          # 누적 편집 중인 개편안
        self.local_dir = Path("output/session"); self.local_dir.mkdir(parents=True, exist_ok=True)

    def team_name(self, code: str) -> str:
        row = self.snap[self.snap["team_code"] == code]
        return row["team_name"].iloc[0] if len(row) else code


def _json(result: ToolResult) -> str:
    return json.dumps(result.for_model(), ensure_ascii=False, default=str)


def _resolve_or_ask(state: State, query: str):
    """조직명 → 코드. 확실하지 않으면 되물을 내용을 돌려준다."""
    res = state.resolver.resolve(query)
    if res.status == OK:
        return res.code, None
    return None, ToolResult(value={"status": res.status, "candidates": [
        {"팀": m["team_name"], "본부": m["hq_name"], "담당": m["dept_name"], "인원": m["headcount"]}
        for m in res.matches]}, caveats=[res.question or "조직을 특정할 수 없습니다. 사용자에게 되물으세요."])


# ============================================================ 도구
def build_tools(state: State):
    from anthropic import beta_tool
    cfg, snap = state.cfg, state.snap

    @beta_tool
    def resolve_org(query: str) -> str:
        """사용자가 말한 조직명을 실제 조직으로 해석한다. 조직을 다루는 모든 도구 전에 먼저 호출한다.

        Args:
            query: 사용자가 말한 조직 이름. 예: "대구 구축팀", "DC Eng팀"
        """
        res = state.resolver.resolve(query)
        matches = [{"팀코드": m["team_code"], "팀": m["team_name"], "본부": m["hq_name"],
                    "담당": m["dept_name"], "인원": m["headcount"]} for m in res.matches]
        return _json(ToolResult(value={"status": res.status, "matches": matches},
                                caveats=[res.question] if res.question else [],
                                assumptions=[f"{state.year}년 1월 명단의 조직 목록 기준"]))

    @beta_tool
    def org_overview() -> str:
        """전사 구조 요약: 인원, 본부·담당·팀 수, 직책자 비율, span, 연령, 본부별 비교."""
        head = structure.headline(snap, cfg)
        hq = structure.hq_comparison(snap, cfg)
        return _json(ToolResult(value={"전사": head, "본부별": hq.to_dict("records")},
                                basis=[f"{state.year}-01 구성원 명단 집계"],
                                assumptions=["담당장·본부장이 단독으로 잡힌 직할 슬롯은 팀 수에서 제외",
                                             "span = 팀 인원 − 팀장"]))

    @beta_tool
    def team_profile(team_code: str) -> str:
        """특정 팀의 현황: 인원·연령·span, 역할(R&R), 3개년 승계 이력, 지역, 관련 판정.

        Args:
            team_code: resolve_org 가 돌려준 팀코드
        """
        team = snap[snap["team_code"] == team_code]
        if team.empty:
            return _json(ToolResult(value={"error": "없는 팀코드입니다. resolve_org 를 먼저 호출하세요."}))
        name = team["team_name"].iloc[0]
        ages = team["age"].dropna(); ages = ages[ages > 0]
        rr = state.roles[state.roles["team_code"] == team_code]
        history = state.transitions[(state.transitions["successor_code"] == team_code)
                                    | (state.transitions["team_code"] == team_code)]
        region, zone = state.regions.infer(name)
        decisions = [d for d in state.ledger.records if d.status == "유효" and name in d.key]
        value = {
            "팀": name, "본부": team["hq_name"].iloc[0], "담당": team["dept_name"].iloc[0],
            "인원": len(team), "팀장수": int((team["position_role"] == "team_leader").sum()),
            "평균연령": round(float(ages.mean()), 1) if len(ages) else None,
            "50세이상": int((ages >= 50).sum()), "5년내정년": int((ages >= cfg.param("retirement_age", 60) - 5).sum()),
            "지역(추론)": region, "권역": zone,
            "미션": rr[rr["role_type"] == "미션"]["role_item"].tolist()[:3] if "role_type" in rr else [],
            "주요R&R": rr[rr["role_type"] == "주요"]["role_item"].tolist()[:6] if "role_type" in rr else [],
            "3개년이력": history[["year_from", "year_to", "team_name", "successor", "successor_share", "status"]]
                        .to_dict("records"),
            "관련판정": [{"판정": d.verdict, "사유": d.reason, "일자": d.decided_on} for d in decisions],
        }
        return _json(ToolResult(value=value, basis=["R&R 정의표 원문", "3개년 명단의 구성원 이동"],
                                assumptions=["지역은 조직명에서 추론 (근무지 데이터 미확보)",
                                             "승계 이력은 구성원의 실제 이동으로 추정 (부서코드는 재사용되어 신뢰 불가)"]))

    @beta_tool
    def function_view(function_name: str = "") -> str:
        """기능별로 몇 개 조직이 몇 명으로 수행하는지. 이름을 주면 그 기능만.

        Args:
            function_name: 기능명 일부. 비우면 상위 15개
        """
        fmap = functions.function_map(state.tagged, snap)
        if function_name:
            fmap = fmap[fmap["function_name"].str.contains(function_name, na=False)]
        return _json(ToolResult(value=fmap.head(15),
                                assumptions=["기능 분류 사전은 시드 상태 — 실장님 확정 전이라 과다 집계 가능",
                                             "미션·주요 R&R 만 태깅 (세부 R&R 제외)"]))

    @beta_tool
    def duplicate_candidates(limit: int = 10, hq_name: str = "") -> str:
        """아직 판정되지 않은 기능 중복 후보 (팀 쌍). 지역 병렬 조직은 제외되어 있다.

        Args:
            limit: 최대 건수
            hq_name: 본부명으로 필터 (선택)
        """
        dup = functions.duplicate_candidates(state.roles, state.tagged, snap, cfg, state.families)
        pending = state.ledger.pending_duplicates(dup)
        if hq_name:
            pending = pending[(pending["hq_a"] == hq_name) | (pending["hq_b"] == hq_name)]
        cols = ["team_a", "team_b", "hq_a", "hq_b", "similarity", "shared_functions", "headcount_sum", "판정초안"]
        return _json(ToolResult(value=pending[cols].head(limit),
                                basis=["역할 문서 TF-IDF 유사도 + 공유 기능 코드"],
                                assumptions=["판정초안은 참고용. 진성 중복인지 의도된 분산인지는 사람이 확정"],
                                caveats=[f"전체 후보 {len(dup)}쌍 중 판정 완료 {len(dup) - len(pending)}쌍 제외"]))

    @beta_tool
    def parallel_families() -> str:
        """지역·채널로 나뉘어 같은 기능을 수행하는 병렬 조직군. 쌍이 아니라 군 단위로 검토할 대상."""
        fam = state.families.drop(columns=["codes"]) if "codes" in state.families else state.families
        return _json(ToolResult(value=fam, assumptions=["역할 유사도 0.65 이상 군집 + 공통 접미사"]))

    @beta_tool
    def region_view() -> str:
        """기능별 권역 분포와 권역별 현황. '이 기능을 전국 몇 개 단위로 운영하는가'에 답한다."""
        return _json(ToolResult(value={"기능별권역": geography.region_function_grid(snap, state.regions).to_dict("records"),
                                       "권역별현황": geography.region_profile(snap, state.regions).to_dict("records")},
                                assumptions=["지역은 조직명에서 추론. 근무지 데이터 확보 시 교체"],
                                caveats=["전국 국사 구조: 조직을 합쳐도 사람은 원래 자리에 남는다"]))

    @beta_tool
    def workforce_risk(hq_name: str = "") -> str:
        """연령 구조, 정년 도래 집중 조직, 승계 리스크. 본부명을 주면 그 본부만.

        Args:
            hq_name: 본부명 (선택)
        """
        scope = snap if not hq_name else snap[snap["hq_name"] == hq_name]
        conc = workforce.retirement_concentration(scope, cfg)
        succ = workforce.succession_risk(scope, cfg)
        return _json(ToolResult(value={
            "연령구조추이": workforce.age_structure(state.members, cfg).to_dict("records"),
            "정년집중조직": conc.head(10).to_dict("records"),
            "승계리스크높음": succ[succ["risk"] == "높음"][["team_name", "headcount", "leader_age", "reason"]].to_dict("records"),
            "정년도래추이": workforce.retirement_wave(state.members, cfg).to_dict("records")},
            assumptions=[f"정년 {cfg.param('retirement_age', 60)}세", "승계 후보군 = 정년까지 10년 이상 남은 팀원"]))

    @beta_tool
    def mission_alignment(dept_name: str = "") -> str:
        """담당이 선언한 R&R을 하위 팀이 실제로 맡고 있는지 (미션 커버율). 담당명을 주면 그 담당만.

        Args:
            dept_name: 담당명 (선택)
        """
        upper_path = cfg.data_dir / "rr_upper_2026.csv"
        if not upper_path.exists():
            return _json(ToolResult(value={"error": "담당급 R&R 자료가 없습니다."}))
        upper = pd.read_csv(upper_path, dtype=str).fillna("")
        roles = pd.read_csv(cfg.data_dir / "team_roles.csv", dtype=str).fillna("")
        cov, gaps = mission.mission_alignment(upper, roles)
        if dept_name:
            cov, gaps = cov[cov["담당"].str.contains(dept_name)], gaps[gaps["담당"].str.contains(dept_name)]
        return _json(ToolResult(value={"커버율": cov.to_dict("records"), "미커버항목": gaps.head(15).to_dict("records")},
                                basis=["담당급 R&R (2026-07 F버전) ↔ 팀 미션·주요 R&R"],
                                assumptions=["토큰 유사도 0.30 이상이면 '맡고 있음'으로 판정. 상위 R&R이 추상적이면 낮게 나온다"]))

    @beta_tool
    def search_rr(query: str, year: int = 0, limit: int = 12) -> str:
        """R&R 전문(미션·주요·세부)에서 키워드를 찾는다. "이 업무는 어느 조직 소관인가"에 답한다.

        Args:
            query: 찾을 말. 예: "해지방어", "개인정보", "도매대가"
            year: 2024 | 2025 | 2026. 0이면 최신 연도
            limit: 최대 조직 수
        """
        year = year or state.year
        path = cfg.data_dir / f"rr_team_{year}.csv"
        if not path.exists():
            return _json(ToolResult(value={"error": f"{year}년 R&R 자료가 없습니다."}))
        rr = pd.read_csv(path, dtype=str).fillna("")
        body = rr["미션"] + " " + rr["주요RR"] + " " + rr.get("세부RR", "")
        hit = rr[body.str.contains(query, case=False, regex=False)]
        rows = []
        for r in hit.head(limit).itertuples():
            text = " ".join([r.미션, r.주요RR, getattr(r, "세부RR", "")])
            i = text.lower().find(query.lower())
            rows.append({"조직": r.조직명, "상위조직": r.상위조직, "본부": getattr(r, "_4", ""),
                         "문맥": text[max(0, i - 40): i + 60].replace("\n", " ")})
        return _json(ToolResult(value={"연도": year, "건수": len(hit), "결과": rows},
                                basis=[f"{year}년 R&R 정의표 전문 검색 (부분 문자열)"],
                                caveats=["동의어·영문 표기는 잡히지 않는다. 다른 표현으로 다시 찾아볼 것"]))

    @beta_tool
    def rr_history(team_query: str) -> str:
        """한 조직(이름 기준)의 2024·2025·2026 미션·주요 R&R을 나란히 보여주고 연도 간 변경폭을 계산한다.

        Args:
            team_query: 조직명 (연도에 따라 이름이 바뀐 경우 최신 이름)
        """
        from hrsim.mission import _sim
        out, prev_text = {}, None
        for year in (2024, 2025, 2026):
            path = cfg.data_dir / f"rr_team_{year}.csv"
            if not path.exists():
                continue
            rr = pd.read_csv(path, dtype=str).fillna("")
            row = rr[rr["조직명"].str.replace(" ", "") == team_query.replace(" ", "")]
            if row.empty:
                out[str(year)] = {"상태": "해당 연도 자료에 없음"}
                continue
            r = row.iloc[0]
            text = r["미션"] + " " + r["주요RR"]
            entry = {"상위조직": r["상위조직"], "미션": r["미션"][:300], "주요R&R": r["주요RR"][:600]}
            if prev_text:
                entry["전년대비변경폭(%)"] = round((1 - _sim(prev_text, text)) * 100)
            prev_text = text
            out[str(year)] = entry
        return _json(ToolResult(value=out, basis=["연도별 R&R 정의표 원문"],
                                assumptions=["이름이 같은 조직을 같은 조직으로 본다. 개명된 조직은 lineage_review 로 승계를 먼저 확인"],
                                caveats=["변경폭은 토큰 유사도 기반 — 문구만 다듬은 경우에도 높게 나올 수 있다"]))

    @beta_tool
    def lineage_review(team_query: str = "", only_review: bool = True, limit: int = 15) -> str:
        """조직 승계 판정과 신뢰도(확정/높음/보통/낮음/확인필요). 구성원 이동·조직명·조직장 세 근거를 결합한다.

        Args:
            team_query: 조직명 일부로 필터 (선택)
            only_review: True면 사람 확인이 필요한 '확인필요·낮음'만
            limit: 최대 건수
        """
        leaders_path = cfg.data_dir / "leaders_3y.csv"
        leaders = pd.read_csv(leaders_path, dtype=str) if leaders_path.exists() else None
        lineage = dynamics.lineage_confidence(state.members, leaders)
        total = len(lineage)
        if only_review:
            lineage = lineage[lineage["confidence"].isin(["확인필요", "낮음"])]
        if team_query:
            mask = lineage["team_name"].str.contains(team_query, na=False) | lineage["successor"].fillna("").str.contains(team_query)
            lineage = lineage[mask]
        cols = ["year_from", "year_to", "team_name", "headcount", "status", "successor", "successor_share", "confidence", "evidence"]
        return _json(ToolResult(value=lineage[cols].head(limit),
                                basis=["명단의 구성원 이동", "조직명 유사도", "보임도의 조직장 사번 연속성"],
                                assumptions=["부서코드는 재사용되어 근거로 쓰지 않는다"],
                                caveats=[f"전체 승계 판정 {total}건 중 사람 확인 필요 "
                                         f"{int((dynamics.lineage_confidence(state.members, leaders)['confidence'].isin(['확인필요','낮음'])).sum())}건. "
                                         "확인된 판정은 record_decision(kind=구조)으로 원장에 남긴다"]))

    @beta_tool
    def add_action(op: str, targets: str, into_name: str = "", into_code: str = "",
                   dept_code: str = "", keep_leader: str = "", reassign_to: str = "", to_dept: str = "") -> str:
        """현재 편집 중인 개편안에 액션을 하나 추가한다. 누적된다. 대상은 반드시 resolve_org 로 얻은 팀코드.

        Args:
            op: MERGE | ABOLISH | MOVE | RENAME
            targets: 팀코드들, 쉼표 구분. MERGE 는 2개 이상
            into_name: MERGE 시 새 조직 이름 (예: "영남구축팀")
            into_code: MERGE 시 새 조직 코드. 비우면 이름과 동일
            dept_code: MERGE 시 소속 담당 코드. 비우면 다수 팀의 담당
            keep_leader: MERGE 시 팀장을 유지할 원래 팀코드 (선택)
            reassign_to: ABOLISH 시 인원을 흡수할 팀코드
            to_dept: MOVE 시 이동할 담당 코드
        """
        codes = [t.strip() for t in targets.split(",") if t.strip()]
        known = set(snap["team_code"])
        bad = [c for c in codes if c not in known]
        if bad:
            return _json(ToolResult(value={"error": f"없는 팀코드: {bad}. resolve_org 를 먼저 쓰세요."}))
        # 보호 조직·기존 판정 사전 경고
        warnings = []
        protected = state.directives.protected_orgs()
        for c in codes:
            name, hq = state.team_name(c), snap[snap["team_code"] == c]["hq_name"].iloc[0]
            for label in (name, hq):
                if label in protected:
                    warnings.append(f"{name}: 보호 조직 {label} ({protected[label]}) — 통합·폐지 금지")
            for d in state.ledger.records:
                if d.status == "유효" and d.verdict == "유지" and name in d.key:
                    warnings.append(f"{name}: 판정 원장에 '유지' ({d.decided_on}, 사유: {d.reason})")
        action = {"op": op.upper()}
        if action["op"] == "MERGE":
            action["targets"] = codes
            action["into"] = {"code": into_code or into_name, "name": into_name or into_code}
            if dept_code: action["into"]["dept_code"] = dept_code
            if keep_leader: action["keep_leader"] = keep_leader
        elif action["op"] == "ABOLISH":
            action["target"] = codes[0]
            if reassign_to: action["reassign_to"] = reassign_to
        elif action["op"] == "MOVE":
            action["target"] = codes[0]; action["to_dept"] = to_dept
        elif action["op"] == "RENAME":
            action["target"] = codes[0]; action["name"] = into_name
        state.scenario.append(action)
        return _json(ToolResult(value={"추가됨": action, "현재개편안": state.scenario},
                                caveats=warnings or ["경고 없음"]))

    @beta_tool
    def reset_scenario() -> str:
        """편집 중인 개편안을 비운다."""
        state.scenario.clear()
        return _json(ToolResult(value={"현재개편안": []}))

    @beta_tool
    def run_simulation(name: str = "검토안") -> str:
        """편집 중인 개편안을 실행해 Before→After 를 계산한다. 구조 델타·사람 영향·연령 구조·기능 커버리지·지역 간 통합·제약 위반.

        Args:
            name: 시나리오 이름
        """
        if not state.scenario:
            return _json(ToolResult(value={"error": "개편안이 비어 있습니다. add_action 으로 먼저 추가하세요."}))
        plan = simulate.Plan(name=name, actions=list(state.scenario))
        errors = plan.validate(snap)
        if errors:
            return _json(ToolResult(value={"error": errors}))
        result = simulate.apply_plan(snap, plan, cfg)
        impact = simulate.people_impact(result)
        demoted = pd.DataFrame(impact.pop("보임 해제 명단"))
        if len(demoted):   # 개인 명단은 로컬로만
            demoted.to_csv(state.local_dir / f"{name}_보임해제.csv", index=False, encoding="utf-8-sig")
        funcs = simulate.function_impact(result, state.tagged)
        value = {
            "구조델타": simulate.metric_delta(result, cfg).to_dict("records"),
            "사람영향": impact,
            "만들어진조직의연령구조": simulate.changed_org_profile(result, cfg).to_dict("records"),
            "기능결손": funcs["기능 결손"], "하나로모인기능": funcs.get("해소된 중복", []),
            "지역간통합": simulate.cross_region_merges(result).to_dict("records"),
            "제약위반": simulate.constraint_check(result, cfg, state.directives).to_dict("records"),
        }
        return _json(ToolResult(value=value, local_only=demoted if len(demoted) else None,
                                basis=["결정론적 시뮬레이션 — 같은 개편안에 항상 같은 결과"],
                                assumptions=["근무지는 기존 그대로 유지된다고 가정", "채용·이탈은 반영하지 않음 (구조 효과만)"],
                                caveats=["지역 간 통합은 인력 집중이 아니라 관리 단위 통합이다"]))

    @beta_tool
    def recall_directives() -> str:
        """지침 원장: 그룹·모회사 가이드, 경영진 의견, 시장 환경, HR 철학. 모든 판단의 전제."""
        book = state.directives.active()
        return _json(ToolResult(value=book.to_frame(),
                                caveats=[f"미확인(구두) 지침 {len(book.unconfirmed())}건 — 참고로만"]))

    @beta_tool
    def recall_decisions(query: str = "") -> str:
        """판정 원장: 사람이 이미 내린 조직 판단. 조직명 일부로 검색.

        Args:
            query: 조직명 일부 (선택)
        """
        frame = state.ledger.to_frame()
        if query and len(frame):
            frame = frame[frame["label"].str.contains(query, na=False)]
        return _json(ToolResult(value=frame[["kind", "label", "verdict", "reason", "decided_by", "decided_on"]]
                                if len(frame) else frame))

    @beta_tool
    def record_decision(kind: str, targets: str, verdict: str, reason: str,
                        decided_by: str, confirmed_by_user: bool = False) -> str:
        """사용자가 확인한 판정을 원장에 기록한다. 반드시 먼저 사용자에게 내용을 보여주고 확인을 받은 뒤 confirmed_by_user=True 로 호출한다.

        Args:
            kind: 중복후보 | 병렬조직군 | 조직명 | 구조
            targets: 대상 조직명들, 쉼표 구분 (2개면 쌍, 그 외는 군)
            verdict: 유지 | 통합검토 | R&R재정의 | 명칭변경 | 보류
            reason: 판단 사유 (필수)
            decided_by: 판정자 (역할)
            confirmed_by_user: 사용자가 기록에 동의했는가
        """
        if not confirmed_by_user:
            return _json(ToolResult(value={"status": "미기록", "안내": "사용자에게 기록 내용을 보여주고 확인을 받으세요."}))
        names = [t.strip() for t in targets.split(",") if t.strip()]
        key = ledger_mod.pair_key(*names) if len(names) == 2 else ledger_mod.group_key(names)
        d = state.ledger.record(ledger_mod.Decision(kind=kind, key=key, label=key, verdict=verdict,
                                                    reason=reason, decided_by=decided_by))
        return _json(ToolResult(value={"기록됨": d.id, "판정": d.verdict, "사유": d.reason}))

    @beta_tool
    def record_directive(source: str, kind: str, force: str, text: str, conveyed_by: str,
                         channel: str = "구두", rule: str = "", confirmed_by_user: bool = False) -> str:
        """사용자가 확인한 지침을 원장에 기록한다. 먼저 구조화한 내용을 보여주고 확인을 받은 뒤 confirmed_by_user=True 로 호출한다. 기계가 검사할 수 없는 말은 kind=방향 으로.

        Args:
            source: 그룹 | 모회사 | CEO | 경영진 | HR실장 | 시장환경 | 규제
            kind: 규칙 | 방향 | 전제
            force: 강제 | 권고
            text: 지침 내용
            conveyed_by: 전달자의 역할 (이름 아님)
            channel: 문서 | 구두
            rule: kind=규칙일 때 검사 조건. 예: "min_team_size=5" 또는 "protect=정보보호실,감사실"
            confirmed_by_user: 사용자가 기록에 동의했는가
        """
        if not confirmed_by_user:
            return _json(ToolResult(value={"status": "미기록", "안내": "구조화한 내용을 사용자에게 보여주고 확인을 받으세요."}))
        from datetime import date
        parsed = {}
        if rule:
            k, _, v = rule.partition("=")
            parsed[k] = [x.strip() for x in v.split(",")] if k in ("protect", "require_function") else (float(v) if "." in v else int(v))
        if kind == "규칙" and not parsed:
            return _json(ToolResult(value={"error": "규칙에는 검사 조건(rule)이 필요합니다. 없으면 kind=방향 으로 기록하세요."}))
        stem = {"그룹": "GRP", "모회사": "PAR", "CEO": "CEO", "경영진": "EXE", "HR실장": "HR", "시장환경": "MKT", "규제": "REG"}[source]
        year = state.year + 1
        n = sum(1 for d in state.directives.items if d.id.startswith(f"{stem}-{year}-")) + 1
        d = directives_mod.Directive(id=f"{stem}-{year}-{n:02d}", source=source, kind=kind, force=force, text=text,
                                     rule=parsed, channel=channel, conveyed_by=conveyed_by,
                                     recorded_on=date.today().isoformat(),
                                     status="확인됨" if channel == "문서" else "미확인", year=year)
        state.directives.append(d)
        return _json(ToolResult(value={"기록됨": d.id, "상태": d.status},
                                caveats=["구두 지침은 전달자에게 문구를 확인받아야 '확인됨'이 됩니다"] if d.status == "미확인" else []))

    return [resolve_org, org_overview, team_profile, function_view, duplicate_candidates, parallel_families,
            region_view, workforce_risk, mission_alignment, search_rr, rr_history, lineage_review,
            add_action, reset_scenario, run_simulation,
            recall_directives, recall_decisions, record_decision, record_directive]


# ============================================================ 시스템 프롬프트
def system_prompt(state: State) -> str:
    directives = state.directives.active()
    lines = [f"- [{d.id}·{d.source}·{d.kind}·{d.status}] {d.text}" for d in directives.items]
    return f"""당신은 조직 개편 논의를 돕는 분석 도구입니다. HR 담당자와 조직 책임자가 함께 보며 씁니다.
기준 데이터: {state.year}년 1월 구성원 명단({len(state.snap)}명, 3개년), 팀 R&R 정의표.

## 절대 규칙
1. 조직명을 지어내지 않는다. 조직을 다루기 전에 반드시 resolve_org 를 호출하고, ambiguous/not_found 면 되묻는다.
2. 숫자를 계산하거나 추정하지 않는다. 도구가 돌려준 값만 인용한다. 도구에 없는 수치는 "자료에 없다"고 말한다.
3. 판단하지 않는다. "통합해야 한다", "권고한다", "비효율적이다" 같은 단정 금지.
   대신 "이렇게 하면 무엇이 달라지고 어떤 비용이 생기는지"를 말한다.
4. 개인을 다루지 않는다. 특정인 평가·배치·거명 요청은 범위 밖이라고 답한다.
5. 원장에 기록할 때는 먼저 내용을 보여주고 사용자 확인을 받은 뒤에만 confirmed_by_user=True 로 호출한다.

## 답변 형식 — 세 조각으로 나눈다
**계산** 도구가 돌려준 사실 · **가정** 그 계산의 전제 (도구의 '가정' 필드를 옮긴다) · **판단 필요** 사람이 정해야 할 것
길게 나열하지 말고 질문에 답이 되는 것부터 말한다. 표는 필요할 때만.

## 되물어야 하는 경우
조직명이 모호할 때 · 통합 후 팀장을 누구로 할지 정해지지 않았을 때 · 개편 의도(효율화/기능집중/정년대응/지역재편)가 불명확할 때 — 의도에 따라 먼저 말할 지표가 다르다.

## 전제 — 지침 원장 (모든 판단은 이 위에서)
{chr(10).join(lines) if lines else "- (기록된 지침 없음)"}
미확인 지침은 참고로만 쓰고, 그 사실을 밝힌다.

## 이 회사에 대해 알아 둘 것
- 전국에 국사·사옥이 있다. 지역 조직을 합쳐도 사람은 원래 자리에 남는다. 지역 간 통합은 인력 집중이 아니라 관리 단위 통합이다.
- 부서 코드는 재사용되어 조직의 정체성이 아니다. 3개년 이력은 구성원 이동으로 추정한 것이다.
- 기능 분류 사전은 아직 시드 상태다. 기능 지도 수치는 과다 집계될 수 있다.
- 판정 원장에 {len(state.ledger.to_frame())}건의 판정이 있다. 이미 판정된 건은 다시 논쟁하지 말고 그 판정을 먼저 알린다.
"""


# ============================================================ 대화 루프
def chat(state: State) -> None:
    import anthropic
    client = anthropic.Anthropic()
    tools = build_tools(state)
    system = [{"type": "text", "text": system_prompt(state), "cache_control": {"type": "ephemeral"}}]
    messages: list[dict] = []
    print(f"조직 설계 에이전트 v0 · {state.year}년 기준 {len(state.snap)}명 · 팀 {state.snap['team_code'].nunique()}개")
    print("질문을 입력하세요. (종료: quit)\n")

    while True:
        try:
            user = input("나> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user.lower() in ("quit", "exit", "종료"):
            break
        messages.append({"role": "user", "content": user})

        runner = client.beta.messages.tool_runner(
            model=MODEL, max_tokens=16000, system=system, tools=tools, messages=messages,
            thinking={"type": "adaptive"}, output_config={"effort": "high"})
        final = None
        for message in runner:
            final = message
            messages.append({"role": "assistant", "content": message.content})
            tool_response = runner.generate_tool_call_response()
            if tool_response is not None:
                messages.append(tool_response)
                for block in message.content:
                    if block.type == "tool_use":
                        print(f"   ⚙ {block.name}({json.dumps(block.input, ensure_ascii=False)[:80]})")
        if final is not None:
            text = "".join(b.text for b in final.content if b.type == "text")
            print(f"\n에이전트> {text}\n")


# ============================================================ 도구 자가 점검
def selftest(state: State) -> None:
    tools = {t.name: t for t in build_tools(state)}
    resolver = state.resolver
    first_two = list(resolver.teams.sort_values("headcount", ascending=False)["team_code"][:2])
    cases = [
        ("resolve_org", {"query": "구축팀"}),
        ("org_overview", {}),
        ("team_profile", {"team_code": first_two[0]}),
        ("function_view", {"function_name": ""}),
        ("duplicate_candidates", {"limit": 3}),
        ("parallel_families", {}),
        ("region_view", {}),
        ("workforce_risk", {}),
        ("mission_alignment", {}),
        ("search_rr", {"query": "안전"}),
        ("rr_history", {"team_query": state.team_name(first_two[0])}),
        ("lineage_review", {"limit": 5}),
        ("add_action", {"op": "MERGE", "targets": ",".join(first_two), "into_name": "테스트통합팀"}),
        ("run_simulation", {"name": "selftest"}),
        ("reset_scenario", {}),
        ("recall_directives", {}),
        ("recall_decisions", {}),
        ("record_decision", {"kind": "구조", "targets": "테스트", "verdict": "보류", "reason": "x", "decided_by": "t"}),
    ]
    ok = 0
    for name, kwargs in cases:
        try:
            out = tools[name].call(kwargs)
            payload = json.loads(out)
            flag = "⚠" if "error" in str(payload.get("결과", ""))[:40] else "✓"
            print(f"  {flag} {name:22} {len(out):6}자  가정 {len(payload.get('가정', []))} · 주의 {len(payload.get('주의', []))}")
            ok += flag == "✓"
        except Exception as e:                       # noqa: BLE001
            print(f"  ✗ {name:22} {type(e).__name__}: {str(e)[:90]}")
    print(f"\n{ok}/{len(cases)} 통과 · 개인정보 게이트를 통과한 결과만 모델에 전달됩니다.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="config/columns.yaml")
    p.add_argument("--data", default="data")
    p.add_argument("--ledger", default="data/ledger/decisions.csv")
    p.add_argument("--directives", default="config/directives.yaml")
    p.add_argument("--selftest", action="store_true")
    args = p.parse_args()
    state = State(args.config, args.data, args.ledger, args.directives)
    (selftest if args.selftest else chat)(state)


if __name__ == "__main__":
    main()
