#!/usr/bin/env python3
"""지침 기록 CLI — 인터뷰 중 임원이 말한 그룹 가이드·경영진 의견을 그 자리에서 남긴다.

지침은 매년 달라지고 문서로도 구두로도 오며, HR 담당자는 모르고 임원만 안다.
그래서 파일을 미리 채우는 것이 아니라 세션에서 건져 올려 기록하고, 확인 상태를 둔다.

  # 구두 지침
  python3 scripts/record_directive.py --source 그룹 --channel 구두 --by 유선사업본부장 \\
      --kind 규칙 --force 강제 --text "올해 팀장 보임은 전년 대비 10% 줄인다" \\
      --rule leader_ratio_max=0.10 --year 2027 --recorder HR담당

  # 방향성 (기계 검사 불가, 해설 우선순위에만)
  python3 scripts/record_directive.py --source CEO --channel 구두 --by CEO \\
      --kind 방향 --force 강제 --text "AI DC는 사람을 늘려도 좋다" --priority DC --year 2027 --recorder HR담당

  python3 scripts/record_directive.py --list
  python3 scripts/record_directive.py --confirm GRP-2027-01     # 전달자에게 문구 확인 후
  python3 scripts/record_directive.py --stale 2027              # 올해 재확인이 필요한 지난해 지침
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from hrsim.directives import Directive, Directives

PATH = Path("config/directives.yaml")
SOURCES = ["그룹", "모회사", "CEO", "경영진", "HR실장", "시장환경", "규제"]
PREFIX = {"그룹": "GRP", "모회사": "PAR", "CEO": "CEO", "경영진": "EXE", "HR실장": "HR",
          "시장환경": "MKT", "규제": "REG"}


def _next_id(book: Directives, source: str, year: int) -> str:
    stem = f"{PREFIX[source]}-{year}-"
    n = sum(1 for d in book.items if d.id.startswith(stem)) + 1
    return f"{stem}{n:02d}"


def _parse_rule(items: list[str]) -> dict:
    out = {}
    for item in items or []:
        key, _, value = item.partition("=")
        if key in ("protect", "require_function"):
            out[key] = [v.strip() for v in value.split(",") if v.strip()]
        else:
            try:
                out[key] = float(value) if "." in value else int(value)
            except ValueError:
                out[key] = value
    return out


def _set_status(directive_id: str, status: str) -> None:
    raw = yaml.safe_load(PATH.read_text(encoding="utf-8")) or {}
    hit = False
    for entry in raw.get("directives", []):
        if entry.get("id") == directive_id:
            entry["status"] = status
            hit = True
    if not hit:
        sys.exit(f"지침 {directive_id} 을 찾지 못했습니다.")
    header = PATH.read_text(encoding="utf-8").split("directives:")[0]
    PATH.write_text(header + yaml.safe_dump({"directives": raw["directives"]}, allow_unicode=True,
                                            sort_keys=False, width=100), encoding="utf-8")
    print(f"✅ {directive_id} → {status}")


def main() -> None:
    p = argparse.ArgumentParser(description="지침 기록")
    p.add_argument("--list", action="store_true")
    p.add_argument("--confirm", metavar="ID", help="전달자에게 문구를 확인받았음")
    p.add_argument("--retire", metavar="ID", help="철회")
    p.add_argument("--stale", type=int, metavar="YEAR", help="올해 재확인이 필요한 지난해 지침")
    p.add_argument("--source", choices=SOURCES)
    p.add_argument("--channel", choices=["문서", "구두"], default="구두")
    p.add_argument("--by", help="전달자 역할 (이름 아님)")
    p.add_argument("--on", default="", help="전달 시점")
    p.add_argument("--kind", choices=["규칙", "방향", "전제"])
    p.add_argument("--force", choices=["강제", "권고"], default="권고")
    p.add_argument("--text")
    p.add_argument("--scope", default="전사")
    p.add_argument("--rule", nargs="*", help="key=value (예: min_team_size=5 protect=정보보호실,감사실)")
    p.add_argument("--priority", nargs="*", default=[])
    p.add_argument("--lens", default="")
    p.add_argument("--year", type=int, default=date.today().year + 1, help="적용 개편 사이클 연도")
    p.add_argument("--recorder", default="")
    args = p.parse_args()

    book = Directives.load(PATH)

    if args.list:
        frame = book.to_frame()
        print(frame.to_string(index=False) if len(frame) else "기록된 지침이 없습니다.")
        return
    if args.confirm:
        return _set_status(args.confirm, "확인됨")
    if args.retire:
        return _set_status(args.retire, "철회")
    if args.stale:
        stale = book.stale(args.stale)
        if not stale:
            print("재확인이 필요한 지침이 없습니다.")
        for d in stale:
            print(f"  {d.id} [{d.source}·{d.year}] {d.text}  ← 올해도 유효합니까?")
        return

    missing = [n for n, v in [("--source", args.source), ("--kind", args.kind),
                              ("--text", args.text), ("--by", args.by)] if not v]
    if missing:
        p.error("필요한 인자: " + ", ".join(missing))

    rule = _parse_rule(args.rule)
    if args.kind == "규칙" and not rule:
        p.error("--kind 규칙 에는 --rule 이 필요합니다. 기계가 검사할 수 없으면 --kind 방향 으로 기록하세요.")

    directive = Directive(
        id=_next_id(book, args.source, args.year), source=args.source, kind=args.kind,
        force=args.force, text=args.text, scope=args.scope, rule=rule,
        priority_functions=args.priority, lens=args.lens,
        channel=args.channel, conveyed_by=args.by, conveyed_on=args.on,
        recorded_by=args.recorder, recorded_on=date.today().isoformat(),
        status="확인됨" if args.channel == "문서" else "미확인", year=args.year)
    book.append(directive, PATH)

    print(f"✅ 기록 → {directive.id}")
    print(f"   [{directive.source}·{directive.channel}·{directive.conveyed_by}] {directive.kind}/{directive.force}")
    print(f"   {directive.text}")
    if directive.status == "미확인":
        print("   ⚠️ 구두 지침입니다. 전달자에게 문구를 확인한 뒤 --confirm 으로 상태를 바꾸세요.")


if __name__ == "__main__":
    main()
