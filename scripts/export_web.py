"""웹 에이전트용 데이터 내보내기 — 팀 단위 집계만. 개인 행은 절대 나가지 않는다.

output/web/data.json 한 파일로, 페이지의 JS 도구 계층이 읽는 모든 것을 담는다.
세부 R&R 은 검색 말뭉치로만 싣고, 원문 표시는 미션·주요 R&R 로 제한한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from hrsim import directives as dmod, dynamics, functions, geography, ledger as lmod, loader, mission, structure, workforce
from hrsim.agent.contracts import BLOCKED_COLUMNS
from hrsim.config import load_config

AGE_BINS = [0, 29, 34, 39, 44, 49, 54, 59, 200]
AGE_LABELS = ["~29", "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60+"]


def records(df: pd.DataFrame, limit: int | None = None) -> list[dict]:
    if df is None or len(df) == 0:
        return []
    out = df.head(limit) if limit else df
    out = out.drop(columns=[c for c in out.columns if c in BLOCKED_COLUMNS], errors="ignore")
    return json.loads(out.to_json(orient="records", force_ascii=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/columns.yaml")
    ap.add_argument("--data", default="data")
    ap.add_argument("--ledger", default="data/ledger/decisions.csv")
    ap.add_argument("--out", default="output/web/data.json")
    args = ap.parse_args()

    cfg = load_config(args.config, args.data, "output")
    members = loader.load_members(cfg)
    roles = loader.load_roles(cfg)
    snap = loader.snapshot(members)
    year = loader.latest_year(members)
    retire = cfg.param("retirement_age", 60)
    regions = geography.load_regions(Path(args.config).parent / "regions.yaml")
    func_dict = functions.load_function_dict(Path(args.config).parent / "function_dict.yaml")
    tagged = functions.tag_functions(roles, func_dict, cfg.roles.get("tagging_types"))
    families = functions.parallel_families(roles, snap)
    book = lmod.Ledger(args.ledger)
    guide = dmod.Directives.load(Path(args.config).parent / "directives.yaml")
    leaders_path = cfg.data_dir / "leaders_3y.csv"
    leaders = pd.read_csv(leaders_path, dtype=str) if leaders_path.exists() else None

    # ---- 팀 단위 집계 (시뮬레이션이 JS 에서 합칠 수 있도록 합산 가능한 필드로)
    leader_only = structure.leader_slots(snap)["team_code"].unique().tolist()
    func_by_team = (tagged[tagged["function_code"] != "F-UNK"].groupby("team_code")["function_name"]
                    .apply(lambda s: sorted(set(s))).to_dict()) if len(tagged) else {}
    rr_by_team = {}
    if "role_type" in roles.columns:
        for code, g in roles.groupby("team_code"):
            rr_by_team[code] = {"mission": g[g["role_type"] == "미션"]["role_item"].tolist()[:4],
                                "major": g[g["role_type"] == "주요"]["role_item"].tolist()[:8]}
    teams = []
    for (code, name), t in snap.groupby(["team_code", "team_name"]):
        ages = t["age"].dropna(); ages = ages[ages > 0]
        bands = pd.cut(ages, bins=AGE_BINS, labels=AGE_LABELS).value_counts().reindex(AGE_LABELS, fill_value=0)
        region, zone = regions.infer(name)
        teams.append({
            "code": code, "name": name, "hq": t["hq_name"].iloc[0], "dept": t["dept_name"].iloc[0],
            "dept_code": t["dept_code"].iloc[0],
            "headcount": int(len(t)),
            "leaders": int((t["position_role"] == "team_leader").sum()),
            "upper_leaders": int(t["position_role"].isin(["dept_leader", "hq_leader"]).sum()),
            "age_sum": float(ages.sum()), "age_n": int(len(ages)),
            "age_bands": {k: int(v) for k, v in bands.items()},
            "over50": int((ages >= 50).sum()), "retire5": int((ages >= retire - 5).sum()),
            "region": region, "zone": zone,
            "functions": func_by_team.get(code, []),
            "rr": rr_by_team.get(code, {"mission": [], "major": []}),
            "leader_only": code in leader_only,
        })

    # ---- 3개년 R&R (미션·주요만)
    rr_years = {}
    for y in (2024, 2025, 2026):
        p = cfg.data_dir / f"rr_team_{y}.csv"
        if p.exists():
            df = pd.read_csv(p, dtype=str).fillna("")
            rr_years[str(y)] = {r["조직명"]: {"parent": r["상위조직"], "hq": r.get("본부(시트)", ""),
                                             "mission": r["미션"][:600], "major": r["주요RR"][:1200],
                                             "detail": r.get("세부RR", "")[:2500]}   # 검색 말뭉치용
                                for r in df.to_dict("records")}

    lineage = dynamics.lineage_confidence(members, leaders)
    upper_p = cfg.data_dir / "rr_upper_2026.csv"
    cov, gaps = (mission.mission_alignment(pd.read_csv(upper_p, dtype=str).fillna(""),
                                           pd.read_csv(cfg.data_dir / "team_roles.csv", dtype=str).fillna(""))
                 if upper_p.exists() else (pd.DataFrame(), pd.DataFrame()))
    dup = functions.duplicate_candidates(roles, tagged, snap, cfg, families)

    data = {
        "meta": {"year": int(year), "people": int(len(snap)), "teams": int(structure.actual_teams(snap)["team_code"].nunique()),
                 "generated": date.today().isoformat(), "retire_age": retire,
                 "params": {"small": cfg.param("small_team_threshold", 4), "span_max": cfg.param("span_max", 12)}},
        "headline": structure.headline(snap, cfg),
        "hqs": records(structure.hq_comparison(snap, cfg)),
        "teams": teams,
        "rr": rr_years,
        "lineage": records(lineage[["year_from", "year_to", "team_name", "headcount", "status", "successor",
                                    "successor_share", "left_company", "confidence", "evidence"]]),
        "new_orgs": records(dynamics.new_orgs(members)),
        "families": [{"suffix": r["공통기능"], "count": int(r["조직수"]), "headcount": int(r["총인원"]),
                      "teams": sorted(r["조직"].split(", "))} for r in families.to_dict("records")] if len(families) else [],
        "duplicates": records(dup[["team_a", "team_b", "hq_a", "hq_b", "similarity", "shared_functions",
                                   "headcount_sum", "판정초안"]]) if len(dup) else [],
        "region_grid": records(geography.region_function_grid(snap, regions)),
        "region_profile": records(geography.region_profile(snap, regions)),
        "retire_concentration": records(workforce.retirement_concentration(snap, cfg), 30),
        "succession_high": records(workforce.succession_risk(snap, cfg).query("risk == '높음'")[
            ["team_name", "headcount", "leader_age", "candidate_pool", "reason"]]),
        "age_structure": records(workforce.age_structure(members, cfg)),
        "retirement_wave": records(workforce.retirement_wave(members, cfg)),
        "attrition": records(dynamics.attrition_by_team(members), 30),
        "mission_coverage": records(cov), "mission_gaps": records(gaps, 60),
        "function_map": records(functions.function_map(tagged, snap), 30),
        "directives": records(guide.to_frame()),
        "decisions_seed": records(book.to_frame()),
        "protected": guide.protected_orgs(),
    }

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    # 마지막 방어선: 개인 식별 컬럼명이 어디에도 남아 있지 않은지
    leaked = [c for c in ("emp_id", "사번", "성명", "hire_date") if f'"{c}"' in text]
    if leaked:
        sys.exit(f"개인 식별 컬럼이 포함되어 있습니다: {leaked}")
    out.write_text(text, encoding="utf-8")
    print(f"{out}  {len(text)/1024:.0f} KB · 팀 {len(teams)} · R&R 연도 {list(rr_years)} · 승계 {len(data['lineage'])} "
          f"· 중복후보 {len(data['duplicates'])} · 지침 {len(data['directives'])} · 판정 {len(data['decisions_seed'])}")


if __name__ == "__main__":
    main()
