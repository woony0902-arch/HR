#!/usr/bin/env python3
"""판정 기록 CLI. 세션 중에 즉시 판단을 남기기 위한 도구.

  python3 scripts/record_decision.py \
      --kind 중복후보 --target "대구구축팀,부산구축팀" \
      --verdict 유지 --reason "국사가 각 지역에 있어 인력 집중 불가" --by "홍길동"

  python3 scripts/record_decision.py --list
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hrsim.ledger import Decision, Ledger, VERDICTS, group_key, pair_key


def main() -> None:
    parser = argparse.ArgumentParser(description="조직 판정 기록")
    parser.add_argument("--ledger", default="data/ledger/decisions.csv")
    parser.add_argument("--list", action="store_true", help="기록된 판정 조회")
    parser.add_argument("--kind", default="중복후보",
                        choices=["중복후보", "병렬조직군", "조직명", "구조"])
    parser.add_argument("--target", help="대상 조직. 쉼표로 구분 (쌍이면 2개)")
    parser.add_argument("--verdict", choices=VERDICTS)
    parser.add_argument("--reason", help="판단 사유 (필수)")
    parser.add_argument("--by", help="판정자")
    parser.add_argument("--machine", default="", help="기계 판정 초안")
    parser.add_argument("--until", default="", help="유효기간")
    args = parser.parse_args()

    ledger = Ledger(args.ledger)

    if args.list:
        frame = ledger.to_frame()
        if frame.empty:
            print("기록된 판정이 없습니다.")
            return
        print(frame[["kind", "label", "verdict", "reason", "decided_by", "decided_on"]]
              .to_string(index=False))
        print(f"\n총 {len(frame)}건")
        print(ledger.summary().to_string(index=False))
        return

    missing = [n for n, v in [("--target", args.target), ("--verdict", args.verdict),
                              ("--reason", args.reason), ("--by", args.by)] if not v]
    if missing:
        parser.error(f"다음 인자가 필요합니다: {', '.join(missing)}")

    targets = [t.strip() for t in args.target.split(",") if t.strip()]
    key = pair_key(*targets) if len(targets) == 2 else group_key(targets)

    previous = ledger.lookup(key, args.kind)
    if previous:
        print(f"⚠️  기존 판정이 있습니다: {previous.verdict} ({previous.decided_on}, {previous.decided_by})")
        print(f"    사유: {previous.reason}")
        print("    새 판정으로 대체합니다.\n")

    decision = ledger.record(Decision(
        kind=args.kind, key=key, label=key, verdict=args.verdict,
        reason=args.reason, decided_by=args.by,
        machine_verdict=args.machine, valid_until=args.until))

    print(f"✅ 기록 완료 → {ledger.path}")
    print(f"   {decision.kind} | {decision.label}")
    print(f"   {decision.verdict} — {decision.reason}")


if __name__ == "__main__":
    main()
