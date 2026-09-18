"""실제 데이터 없이 파이프라인을 검증하기 위한 합성 조직 데이터 생성기.

의도적으로 기능 중복(기획/인사/구매/데이터)과 조직 변동(신설·폐지·이관·통합)을 심어 두었다.
"""
from __future__ import annotations

import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

SEED = 20260918
OUT = Path("data")
YEARS = [2024, 2025, 2026]

# (본부코드, 본부명, 담당코드, 담당명, 팀코드, 팀명, 기본인원, 역할)
BASE_ORG = [
    ("H10", "경영지원본부", "D101", "기획담당", "T1011", "경영기획팀", 9,
     ["중장기 전략 수립", "전사 사업계획 수립 및 관리", "경영실적 분석 및 리포팅", "이사회 안건 관리"]),
    ("H10", "경영지원본부", "D101", "기획담당", "T1012", "경영분석팀", 6,
     ["경영 데이터 분석", "실적 대시보드 운영", "KPI 지표 관리", "손익 분석"]),
    ("H10", "경영지원본부", "D102", "재무담당", "T1021", "회계팀", 8,
     ["결산 및 재무제표 작성", "세무 신고", "전표 처리"]),
    ("H10", "경영지원본부", "D102", "재무담당", "T1022", "자금팀", 5,
     ["자금 조달 및 운용", "외환 관리", "금융기관 대응"]),
    ("H10", "경영지원본부", "D102", "재무담당", "T1023", "원가팀", 4,
     ["원가 산정", "예산 편성 및 통제", "비용 분석"]),
    ("H10", "경영지원본부", "D103", "인사담당", "T1031", "인사기획팀", 7,
     ["인사제도 기획 및 개정", "평가/보상제도 운영", "조직설계 및 조직개편 지원", "인력계획 수립"]),
    ("H10", "경영지원본부", "D103", "인사담당", "T1032", "인재채용팀", 5,
     ["신입 및 경력 채용", "채용 브랜딩", "온보딩 운영"]),
    ("H10", "경영지원본부", "D103", "인사담당", "T1033", "인사운영팀", 6,
     ["급여 및 근태 관리", "복리후생 운영", "노무 및 노사 대응"]),
    ("H10", "경영지원본부", "D104", "지원담당", "T1041", "총무팀", 5,
     ["사옥 및 자산 관리", "차량 및 임대차 관리", "총무 구매"]),
    ("H10", "경영지원본부", "D104", "지원담당", "T1042", "구매팀", 7,
     ["원부자재 구매 및 발주", "협력사 관리", "입찰 및 계약 체결"]),
    ("H10", "경영지원본부", "D104", "지원담당", "T1043", "법무팀", 4,
     ["계약 검토 및 자문", "소송 대응", "준법 및 내부통제 점검"]),

    ("H20", "A사업부", "D201", "사업기획담당", "T2011", "사업기획팀", 8,
     ["사업부 중장기 전략 수립", "사업계획 수립 및 실적 관리", "신규사업 검토", "사업 손익 분석"]),
    ("H20", "A사업부", "D201", "사업기획담당", "T2012", "상품기획팀", 6,
     ["상품 기획 및 개발", "가격 정책 수립", "상품 포트폴리오 관리"]),
    ("H20", "A사업부", "D202", "영업담당", "T2021", "국내영업1팀", 12,
     ["국내 거래처 영업 및 수주", "매출 관리", "채널 관리"]),
    ("H20", "A사업부", "D202", "영업담당", "T2022", "국내영업2팀", 11,
     ["국내 거래처 영업 및 수주", "신규 거래처 개발", "판매 실적 관리"]),
    ("H20", "A사업부", "D202", "영업담당", "T2023", "영업지원팀", 6,
     ["견적 및 계약 관리", "매출 채권 관리", "영업 데이터 분석"]),
    ("H20", "A사업부", "D203", "운영담당", "T2031", "생산관리팀", 10,
     ["생산 계획 수립", "공정 및 설비 관리", "품질 관리"]),
    ("H20", "A사업부", "D203", "운영담당", "T2032", "물류팀", 8,
     ["재고 및 창고 관리", "배송 및 출하 관리", "공급망 운영"]),
    ("H20", "A사업부", "D203", "운영담당", "T2033", "소싱팀", 5,
     ["원자재 소싱 및 발주", "협력사 발굴 및 평가", "구매 단가 협상"]),

    ("H30", "B사업부", "D301", "전략담당", "T3011", "전략기획팀", 7,
     ["사업부 전략 수립", "사업계획 및 경영실적 관리", "신사업 발굴 및 투자 검토"]),
    ("H30", "B사업부", "D301", "전략담당", "T3012", "경영지원팀", 6,
     ["사업부 예산 관리", "인사 운영 및 채용 지원", "총무 및 일반 관리"]),
    ("H30", "B사업부", "D302", "마케팅담당", "T3021", "브랜드마케팅팀", 8,
     ["브랜드 전략 수립", "프로모션 및 캠페인 기획", "광고 집행 관리"]),
    ("H30", "B사업부", "D302", "마케팅담당", "T3022", "디지털마케팅팀", 7,
     ["디지털 광고 운영", "고객 데이터 분석", "퍼포먼스 마케팅"]),
    ("H30", "B사업부", "D303", "고객담당", "T3031", "고객서비스팀", 14,
     ["고객 상담 및 CS 운영", "VOC 접수 및 처리", "민원 대응"]),
    ("H30", "B사업부", "D303", "고객담당", "T3032", "고객경험팀", 5,
     ["고객 여정 분석", "VOC 데이터 분석 및 개선 과제 도출", "고객 만족도 조사"]),

    ("H40", "기술본부", "D401", "IT담당", "T4011", "IT인프라팀", 8,
     ["서버 및 네트워크 운영", "클라우드 인프라 관리", "시스템 운영 및 장애 대응"]),
    ("H40", "기술본부", "D401", "IT담당", "T4012", "시스템개발팀", 11,
     ["사내 시스템 개발 및 구축", "애플리케이션 유지보수", "솔루션 도입"]),
    ("H40", "기술본부", "D401", "IT담당", "T4013", "정보보안팀", 4,
     ["정보보호 정책 수립", "개인정보 보호 관리", "보안 침해 대응"]),
    ("H40", "기술본부", "D402", "데이터담당", "T4021", "데이터플랫폼팀", 7,
     ["데이터 플랫폼 구축 및 운영", "데이터 파이프라인 관리", "BI 리포팅 지원"]),
    ("H40", "기술본부", "D402", "데이터담당", "T4022", "데이터분석팀", 6,
     ["전사 데이터 분석", "분석 대시보드 개발", "지표 체계 관리"]),
]

# 연도별 조직 변경 (합성 데이터에 동역학을 심는다)
PATCHES = {
    2025: [
        ("CREATE", "T4023", "H40", "D402", "AI기술팀", 5,
         ["AI 모델 개발", "데이터 분석 자동화", "기술 검증"]),
        ("ABOLISH", "T1041"),                       # 총무팀 폐지 → 지원담당 잔여로 흡수
        ("RENAME", "T3022", "그로스마케팅팀"),
        ("TRANSFER", "T2033", "D104"),              # 소싱팀을 A사업부 → 경영지원본부 지원담당으로 이관
    ],
    2026: [
        ("MERGE", ["T1012"], "T1011"),              # 경영분석팀 → 경영기획팀 흡수
        ("CREATE", "T3013", "H30", "D301", "신사업개발팀", 6,
         ["신규사업 발굴", "사업 타당성 검토", "제휴 및 투자 검토"]),
        ("TRANSFER", "T3032", "D302"),              # 고객경험팀을 고객담당 → 마케팅담당으로 이관
    ],
}

SURNAMES = list("김이박최정강조윤장임한오서신권황안송류전홍")
LEADER_TITLES = {"H": "본부장", "D": "담당임원", "T": "팀장"}


def build_org(year: int) -> dict[str, dict]:
    """해당 연도의 조직 상태를 반환한다."""
    org = {t[4]: {"hq_code": t[0], "hq_name": t[1], "dept_code": t[2], "dept_name": t[3],
                  "team_name": t[5], "size": t[6], "roles": t[7]} for t in BASE_ORG}
    dept_names = {t[2]: t[3] for t in BASE_ORG}
    hq_names = {t[0]: t[1] for t in BASE_ORG}
    dept_hq = {t[2]: t[0] for t in BASE_ORG}

    for patch_year in YEARS:
        if patch_year > year:
            break
        for patch in PATCHES.get(patch_year, []):
            kind = patch[0]
            if kind == "CREATE":
                _, code, hq, dept, name, size, roles = patch
                org[code] = {"hq_code": hq, "hq_name": hq_names[hq], "dept_code": dept,
                             "dept_name": dept_names[dept], "team_name": name,
                             "size": size, "roles": roles}
            elif kind == "ABOLISH":
                org.pop(patch[1], None)
            elif kind == "RENAME":
                org[patch[1]]["team_name"] = patch[2]
            elif kind == "TRANSFER":
                dest = patch[2]
                org[patch[1]].update(dept_code=dest, dept_name=dept_names[dest],
                                     hq_code=dept_hq[dest], hq_name=hq_names[dept_hq[dest]])
            elif kind == "MERGE":
                sources, dest = patch[1], patch[2]
                for src in sources:
                    if src in org:
                        org[dest]["size"] += org[src]["size"]
                        org[dest]["roles"] = list(dict.fromkeys(org[dest]["roles"] + org[src]["roles"]))
                        org.pop(src)
    return org


def main() -> None:
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    next_id = [1000]
    people: dict[str, dict] = {}
    assignment: dict[str, str] = {}      # emp_id → team_code
    leaders: dict[str, str] = {}         # org_code → emp_id

    def hire(team_code: str, year: int, senior: bool = False) -> str:
        next_id[0] += 1
        emp_id = f"E{next_id[0]}"
        age = rng.randint(42, 56) if senior else rng.choices(
            [rng.randint(25, 29), rng.randint(30, 39), rng.randint(40, 49), rng.randint(50, 59)],
            weights=[25, 40, 25, 10])[0]
        max_tenure = min(age - 24, 25)
        tenure = rng.randint(0, max(max_tenure, 1))
        hire_day = date(year - tenure, rng.randint(1, 12), rng.randint(1, 28))
        people[emp_id] = {"age_at": {year: age}, "hire_date": hire_day,
                          "gender": rng.choices(["남", "여"], weights=[58, 42])[0]}
        assignment[emp_id] = team_code
        return emp_id

    rows = []
    for index, year in enumerate(YEARS):
        org = build_org(year)

        if index == 0:
            for code, spec in org.items():
                leaders[code] = hire(code, year, senior=True)
                for _ in range(spec["size"] - 1):
                    hire(code, year)
            for dept in {s["dept_code"] for s in org.values()}:
                team = next(c for c, s in org.items() if s["dept_code"] == dept)
                leaders[dept] = hire(team, year, senior=True)
            for hq in {s["hq_code"] for s in org.values()}:
                team = next(c for c, s in org.items() if s["hq_code"] == hq)
                leaders[hq] = hire(team, year, senior=True)
        else:
            gone = [e for e in list(assignment) if rng.random() < 0.075]
            for emp_id in gone:
                assignment.pop(emp_id)
                for org_code, leader in list(leaders.items()):
                    if leader == emp_id:
                        pool = [e for e, t in assignment.items() if t == org_code]
                        leaders[org_code] = pool[0] if pool else hire(
                            org_code if org_code in org else next(iter(org)), year, senior=True)

            # 폐지·통합된 팀 인원은 같은 담당의 다른 팀으로 재배치
            for emp_id, team in list(assignment.items()):
                if team not in org:
                    spec = next((s for s in org.values()), None)
                    same_dept = [c for c, s in org.items()
                                 if s["dept_code"] == spec["dept_code"]] if spec else []
                    candidates = [c for c, s in org.items()] 
                    assignment[emp_id] = rng.choice(same_dept or candidates)

            for emp_id in list(assignment):           # 내부 이동
                if rng.random() < 0.05:
                    assignment[emp_id] = rng.choice(list(org))

            for code, spec in org.items():
                current = sum(1 for t in assignment.values() if t == code)
                for _ in range(max(spec["size"] - current, 0)):
                    hire(code, year)
                if code not in leaders or assignment.get(leaders[code]) is None:
                    pool = [e for e, t in assignment.items() if t == code]
                    leaders[code] = pool[0] if pool else hire(code, year, senior=True)

        for emp_id, team_code in assignment.items():
            if team_code not in org:
                continue
            spec = org[team_code]
            person = people[emp_id]
            base_year = min(person["age_at"])
            age = person["age_at"][base_year] + (year - base_year)

            title = ""
            if leaders.get(spec["hq_code"]) == emp_id:
                title = LEADER_TITLES["H"]
            elif leaders.get(spec["dept_code"]) == emp_id:
                title = LEADER_TITLES["D"]
            elif leaders.get(team_code) == emp_id:
                title = LEADER_TITLES["T"]

            rows.append({"기준연도": year, "사번": emp_id,
                         "본부코드": spec["hq_code"], "본부명": spec["hq_name"],
                         "담당코드": spec["dept_code"], "담당명": spec["dept_name"],
                         "팀코드": team_code, "팀명": spec["team_name"],
                         "입사일": person["hire_date"].isoformat(),
                         "성별": person["gender"], "나이": age, "직책": title})

    frame = pd.DataFrame(rows)
    for year in YEARS:
        path = OUT / f"members_{year}.csv"
        frame[frame["기준연도"] == year].drop(columns="기준연도").to_csv(
            path, index=False, encoding="utf-8-sig")
        print(f"  {path}  ({(frame['기준연도'] == year).sum()}명)")

    final_org = build_org(YEARS[-1])
    roles = [{"팀코드": code, "팀명": spec["team_name"], "역할": role}
             for code, spec in final_org.items() for role in spec["roles"]]
    role_path = OUT / "team_roles.csv"
    pd.DataFrame(roles).to_csv(role_path, index=False, encoding="utf-8-sig")
    print(f"  {role_path}  ({len(roles)}개 역할 / {len(final_org)}개 팀)")


if __name__ == "__main__":
    print("합성 데이터 생성:")
    main()
