"""실제 HR 엑셀 2종을 파이프라인 표준 스키마(CSV)로 변환한다.

입력
  - 구성원 명단 xlsx : 기준일별 long 포맷, 담당Ⅲ/Ⅱ/Ⅰ/소속부서 4단 계층
  - 조직 R&R xlsx    : 상위조직 | 팀 | 미션 | 주요 R&R | 세부 R&R (병합셀)

주의
  소속부서 코드는 조직의 정체성이 아니라 계층 내 '위치 슬롯'이다.
  (예: FC000 이 2025년 유선사업본부 → 2026년 미디어사업본부)
  따라서 조직 식별자로 코드가 아닌 **조직명**을 사용한다.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# 계층 매핑: 담당Ⅱ(본부/센터/실) › 담당Ⅰ(담당) › 소속부서(팀)
# 담당Ⅲ는 2026년 기준 CEO/CSPO 뿐이라 별도 컬럼(top_name)으로만 보존한다.
LEVEL_COLS = {"top": "담당Ⅲ명(본부)", "hq": "담당Ⅱ명(그룹/트라이브)",
              "dept": "담당Ⅰ명", "team": "소속부서명"}
TEAM_LEADER_TITLES = {"팀장", "PL", "리더"}


def normalize(name: object) -> str:
    """조직명 표준화. 공백·괄호 표기 차이를 흡수한다."""
    text = re.sub(r"\s+", "", str(name or "")).strip()
    return text


def ingest_members(path: Path, out_dir: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, dtype=str)
    out = pd.DataFrame({
        "사번": raw["사번"].str.strip(),
        "본부코드": raw[LEVEL_COLS["hq"]].map(normalize),
        "본부명": raw[LEVEL_COLS["hq"]].str.strip(),
        "담당코드": raw[LEVEL_COLS["dept"]].map(normalize),
        "담당명": raw[LEVEL_COLS["dept"]].str.strip(),
        "팀코드": raw[LEVEL_COLS["team"]].map(normalize),
        "팀명": raw[LEVEL_COLS["team"]].str.strip(),
        "성별": raw["성별"].str.strip(),
        "나이": pd.to_numeric(raw["만나이"], errors="coerce"),
    })
    # 근속은 회사 입사일 기준. 결측 시 최초입사일로 대체
    hire = pd.to_datetime(raw["SKB입사일"], errors="coerce").fillna(
        pd.to_datetime(raw["최초입사일"], errors="coerce"))
    out["입사일"] = hire.dt.date.astype(str)

    # 직책: 보임여부가 N이면 실제 보직자가 아니므로 팀원으로 처리
    title = raw["소속직책명"].str.strip().fillna("")
    out["직책"] = title.where(raw["소속보임여부"].str.strip().eq("Y"), "")

    # 분석에 유용한 원본 필드 보존
    out["직군"] = raw["직군명"].str.strip()
    out["BAND"] = raw["BAND명"].str.strip()
    out["인사상태"] = raw["인사상태명"].str.strip()
    out["최상위"] = raw[LEVEL_COLS["top"]].str.strip()
    out["원부서코드"] = raw["소속부서"].str.strip()
    out["파견겸직"] = (raw["소속부서"] != raw["근무부서"]).map({True: "Y", False: "N"})

    out["기준일"] = raw["기준일"].str.strip()
    out["연도"] = out["기준일"].str.slice(0, 4).astype(int)
    out = _disambiguate_teams(out)

    out_dir.mkdir(parents=True, exist_ok=True)
    for year, group in out.groupby("연도"):
        target = out_dir / f"members_{year}.csv"
        group.drop(columns=["연도", "기준일"]).to_csv(target, index=False, encoding="utf-8-sig")
        print(f"  {target}  ({len(group)}명)")
    return out


def ingest_roles(path: Path, out_dir: Path, members: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_excel(path, dtype=str, header=0)
    raw.columns = ["_blank", "상위조직", "팀", "미션", "주요RR", "세부RR"][: len(raw.columns)]
    raw = raw.drop(columns=["_blank"])
    raw["상위조직"] = raw["상위조직"].ffill()          # 병합셀 복원
    raw = raw[raw["팀"].notna()].copy()

    # 팀명 중복으로 코드가 분리된 경우를 위해 최신 명단에서 이름 → 코드 매핑을 만든다
    latest_members = members[members["연도"] == members["연도"].max()]
    name_to_code = (latest_members.groupby(["팀명", "팀코드"]).size()
                    .reset_index(name="n").sort_values("n", ascending=False)
                    .drop_duplicates("팀명").set_index("팀명")["팀코드"].to_dict())

    rows = []
    for row in raw.itertuples():
        team = str(row.팀).strip()
        code = name_to_code.get(team, normalize(team))
        # 미션은 조직의 존재 이유, 주요/세부 R&R 은 실제 수행 업무.
        # 중복 탐지는 '수행 업무' 기준이 정확하므로 세부 R&R 을 항목 단위로 분해한다.
        for source, weight in [(row.미션, "미션"), (row.주요RR, "주요"), (row.세부RR, "세부")]:
            for item in _split_items(source):
                rows.append({"팀코드": code, "팀명": team,
                             "상위조직": str(row.상위조직).strip(),
                             "구분": weight, "역할": item})

    roles = pd.DataFrame(rows)
    target = out_dir / "team_roles.csv"
    roles.to_csv(target, index=False, encoding="utf-8-sig")

    latest = members[members["연도"] == members["연도"].max()]
    matched = set(roles["팀코드"]) & set(latest["팀코드"])
    print(f"  {target}  ({len(roles)}개 역할 / {roles['팀코드'].nunique()}개 팀)")
    print(f"  최신연도 부서 {latest['팀코드'].nunique()}개 중 R&R 보유 {len(matched)}개")
    missing = sorted(set(roles['팀코드']) - set(latest['팀코드']))
    if missing:
        print(f"  ⚠️ 명단에 없는 R&R 조직 {len(missing)}개: {', '.join(missing[:5])}")
    return roles


def _disambiguate_teams(out: pd.DataFrame) -> pd.DataFrame:
    """같은 연도에 동일 팀명이 여러 담당에 있으면 담당명을 덧붙여 코드를 분리한다."""
    collisions = (out.groupby(["연도", "팀코드"])["담당코드"].nunique()
                  .loc[lambda s: s > 1].reset_index()["팀코드"].unique())
    if len(collisions):
        mask = out["팀코드"].isin(collisions)
        out.loc[mask, "팀코드"] = out.loc[mask, "담당코드"] + "|" + out.loc[mask, "팀코드"]
        print(f"  ℹ️ 팀명 중복 {len(collisions)}건은 담당명을 붙여 구분: {', '.join(collisions)}")
    return out


def _split_items(text: object) -> list[str]:
    """'○', '-', 줄바꿈으로 나열된 항목을 분해한다."""
    if pd.isna(text):
        return []
    parts = re.split(r"[\n○◦•]|(?:^|\s)[-–]\s", str(text))
    return [p.strip(" -–—·\t") for p in parts if len(p.strip(" -–—·\t")) > 3]


def main() -> None:
    parser = argparse.ArgumentParser(description="실제 HR 엑셀 → 표준 CSV 변환")
    parser.add_argument("members_xlsx")
    parser.add_argument("roles_xlsx")
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    out_dir = Path(args.out)
    print("변환:")
    members = ingest_members(Path(args.members_xlsx), out_dir)
    ingest_roles(Path(args.roles_xlsx), out_dir, members)

    print("\n[계층 깊이 점검]")
    for year, group in members.groupby("연도"):
        print(f"  {year}: 최상위 {group['최상위'].nunique()} · 본부 {group['본부명'].nunique()} "
              f"· 담당 {group['담당명'].nunique()} · 팀 {group['팀명'].nunique()} "
              f"· 인원 {len(group)}")


if __name__ == "__main__":
    main()
