#!/usr/bin/env python3
"""조직 진단 · 개편 시뮬레이션 실행기.

  python3 run.py diagnose                    # 1차 진단 리포트
  python3 run.py simulate plans/*.yaml       # 개편 시나리오 시뮬레이션
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from hrsim import (dynamics, functions, geography, loader, naming, quality,
                   report, simulate, structure, workforce)
from hrsim.config import load_config


def _load(args) -> tuple:
    cfg = load_config(args.config, args.data, args.out)
    members = loader.load_members(cfg)
    roles = loader.load_roles(cfg)
    return cfg, members, roles


def diagnose(args) -> None:
    cfg, members, roles = _load(args)
    snap = loader.snapshot(members)
    base_year = loader.latest_year(members)

    findings = quality.check(members, roles, cfg)
    if quality.has_blocking(findings):
        print("⚠️  ERROR 등급 데이터 문제가 있습니다. 리포트는 생성하되 결과 해석에 주의하세요.\n")

    func_dict = functions.load_function_dict(Path(args.config).parent / "function_dict.yaml")
    tagged = functions.tag_functions(roles, func_dict, cfg.roles.get("tagging_types"))
    families = functions.parallel_families(roles, snap)
    regions = geography.load_regions(Path(args.config).parent / "regions.yaml")
    conflicts = naming.name_function_conflicts(
        roles, snap, exclude=functions.parallel_membership(families))

    ctx = {
        "base_year": base_year,
        "years": sorted(members["year"].unique().tolist()),
        "quality_summary": quality.summary(findings),
        "headline": structure.headline(snap, cfg),
        "team_profile": structure.team_profile(snap, cfg),
        "span_table": structure.span_table(snap, cfg),
        "hq_comparison": structure.hq_comparison(snap, cfg),
        "yearly": dynamics.yearly_summary(members),
        "lifecycle": dynamics.org_lifecycle(members),
        "transitions": dynamics.org_transitions(members),
        "new_orgs": dynamics.new_orgs(members),
        "flows": dynamics.flow_matrix(members),
        "attrition": dynamics.attrition_by_team(members),
        "succession": workforce.succession_risk(snap, cfg),
        "retirement": workforce.retirement_wave(members, cfg),
        "age_structure": workforce.age_structure(members, cfg),
        "retire_concentration": workforce.retirement_concentration(snap, cfg),
        "decline": workforce.natural_decline(members, cfg),
        "age_profile": workforce.age_profile(snap),
        "gender_profile": workforce.gender_profile(snap),
        "function_map": functions.function_map(tagged, snap),
        "parallel_families": families,
        "duplicates": functions.duplicate_candidates(roles, tagged, snap, cfg, families),
        "gaps": functions.coverage_gaps(tagged, func_dict),
        "region_profile": geography.region_profile(snap, regions),
        "region_grid": geography.region_function_grid(snap, regions),
        "name_role_match": naming.name_role_match(roles, snap),
        "vague_names": naming.vague_names(snap),
        "naming_conventions": naming.naming_conventions(snap),
        "name_conflict_동명이의": conflicts["동명이의"],
        "name_conflict_이명동의": conflicts["이명동의"],
        "keyword_candidates": functions.keyword_candidates(roles),
        "corpus_terms": functions.corpus_terms(roles),
        "tagged": tagged,
    }

    path = report.diagnosis_report(ctx, cfg.out_dir)
    tables = {k: v for k, v in ctx.items() if hasattr(v, "columns")}
    saved = report.save_tables(tables, cfg.out_dir)

    print(f"✅ 진단 리포트: {path}")
    print(f"   CSV {len(saved)}종: {cfg.out_dir}/tables/")
    _print_highlights(ctx)


def _print_highlights(ctx: dict) -> None:
    head = ctx["headline"]
    print("\n── 요약 ──")
    print(f"  인원 {head['headcount']}명 · 본부 {head['hq_count']} · 담당 {head['dept_count']} "
          f"· 팀 {head['team_count']} · 직책자 비율 {head['leader_ratio_pct']}%")
    print(f"  소규모 팀 {head['small_teams']}개 · 과대 팀 {head['large_teams']}개 "
          f"· 중위 span {head['median_span']} · 최대 span {head['max_span']}")

    dup = ctx["duplicates"]
    if len(dup):
        print(f"  중복 후보 {len(dup)}쌍 (상위 3)")
        for row in dup.head(3).itertuples():
            print(f"    · {row.team_a} ↔ {row.team_b} "
                  f"(유사도 {row.similarity}, {row.headcount_sum}명) — {row.판정초안}")

    risk = ctx["succession"]
    print(f"  승계 리스크 '높음' {int((risk['risk'] == '높음').sum())}개 팀")

    quality_df = ctx["quality_summary"]
    if len(quality_df):
        errors = int((quality_df["level"] == "ERROR").sum())
        warns = int((quality_df["level"] == "WARN").sum())
        print(f"  데이터 점검: ERROR {errors} · WARN {warns}")


def run_simulation(args) -> None:
    cfg, members, roles = _load(args)
    snap = loader.snapshot(members)

    func_dict = functions.load_function_dict(Path(args.config).parent / "function_dict.yaml")
    tagged = functions.tag_functions(roles, func_dict, cfg.roles.get("tagging_types"))

    paths = [Path(p) for pattern in args.plans for p in sorted(Path().glob(pattern))] \
        if any("*" in p for p in args.plans) else [Path(p) for p in args.plans]
    if not paths:
        sys.exit("시나리오 파일을 찾을 수 없습니다.")

    plans, results = [], []
    for path in paths:
        plan = simulate.Plan.from_yaml(path)
        errors = plan.validate(snap)
        if errors:
            print(f"❌ {plan.name}: " + "; ".join(errors))
            continue

        result = simulate.apply_plan(snap, plan, cfg)
        results.append((result,
                        simulate.metric_delta(result, cfg),
                        simulate.people_impact(result),
                        simulate.function_impact(result, tagged),
                        simulate.constraint_check(result, cfg),
                        simulate.changed_org_profile(result, cfg),
                        simulate.cross_region_merges(result)))
        plans.append(plan)

    if not plans:
        sys.exit("적용 가능한 시나리오가 없습니다.")

    comparison = simulate.compare_plans(snap, plans, cfg)
    path = report.simulation_report(results, comparison, cfg.out_dir)
    comparison.to_csv(cfg.out_dir / "tables" / "scenario_comparison.csv",
                      index=False, encoding="utf-8-sig")

    print(f"✅ 시뮬레이션 리포트: {path}\n")
    print(comparison.to_string(index=False))
    for result, _, impact, funcs, violations, _profile, _region in results:
        lost = funcs["기능 결손"]
        print(f"\n[{result.plan.name}] 소속변경 {impact['소속 변경 인원']}명 · "
              f"보임해제 {impact['보임 해제 인원']}명 · "
              f"신규 제약위반 {0 if violations.empty else int((violations['발생'] == '신규').sum())}건"
              + (f" · ⚠️ 기능결손: {', '.join(lost)}" if lost else ""))


def main() -> None:
    parser = argparse.ArgumentParser(description="조직 진단 · 개편 시뮬레이션")
    parser.add_argument("--config", default="config/columns.yaml")
    parser.add_argument("--data", default="data")
    parser.add_argument("--out", default="output")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("diagnose", help="1차 진단 리포트 생성")
    sim = sub.add_parser("simulate", help="개편 시나리오 시뮬레이션")
    sim.add_argument("plans", nargs="+", help="시나리오 YAML 경로 (glob 가능)")

    args = parser.parse_args()
    {"diagnose": diagnose, "simulate": run_simulation}[args.command](args)


if __name__ == "__main__":
    main()
