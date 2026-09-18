"""조직명과 실제 기능의 정합성 진단.

조직명은 대외적으로 그 조직이 무엇을 하는지 알리는 1차 신호다.
이름이 역할을 드러내지 못하거나, 같은 일을 다른 이름으로 부르거나,
비슷한 이름이 다른 일을 가리키면 협업 비용과 중복이 함께 늘어난다.
"""
from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from .functions import _cosine, _tfidf, team_documents

# 단독으로 쓰이면 기능을 특정하지 못하는 일반어
GENERIC = {"지원", "기획", "운영", "관리", "사업", "서비스", "전략", "추진", "혁신",
           "개발", "기술", "솔루션", "플랫폼", "통합", "공통", "신규", "미래"}
SUFFIXES = ("팀", "국", "실", "담당", "센터", "본부", "그룹", "파트")


def _core(name: str) -> str:
    """조직명에서 계층 접미사를 떼어 낸 핵심어."""
    text = str(name or "").strip()
    for suffix in SUFFIXES:
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def _name_tokens(core: str) -> list[str]:
    """조직명 핵심어를 의미 단위로 쪼갠다.

    'Core솔루션' 을 한 덩어리로 보면 역할 서술에 '솔루션' 이 있어도 불일치로 잡힌다.
    문자 종류(한글/영문/숫자)가 바뀌는 지점에서 끊고, 긴 한글 복합어는 앞뒤로 나눠
    부분 일치도 확인할 수 있게 한다.
    """
    parts = re.findall(r"[가-힣]+|[A-Za-z]+|[0-9]+", str(core or ""))
    tokens: list[str] = []
    for part in parts:
        if part.isdigit():
            continue
        token = part.lower()
        if len(token) >= 2 and token not in GENERIC:
            tokens.append(token)
        # 긴 한글 복합어는 인접 2음절로도 확인한다 ('미디어안전보건' → '안전', '보건')
        if re.fullmatch(r"[가-힣]+", part) and len(part) >= 4:
            tokens += [part[i:i + 2] for i in range(len(part) - 1)]
    return list(dict.fromkeys(t for t in tokens if t not in GENERIC))


def _is_english(text: str) -> bool:
    letters = re.findall(r"[A-Za-z가-힣]", text)
    return bool(letters) and sum(c.isascii() for c in letters) / len(letters) > 0.5


def name_role_match(roles: pd.DataFrame, snap: pd.DataFrame,
                    top_n: int = 15) -> pd.DataFrame:
    """조직명의 핵심어가 그 조직의 역할 서술에 실제로 등장하는가.

    '이름 따로 일 따로'인 조직을 찾는다. 개편 후 이름만 남고 업무가 바뀐 경우,
    혹은 이름이 업무 범위를 제대로 담지 못하는 경우가 여기 걸린다.
    """
    docs = team_documents(roles)
    vectors, codes = _tfidf(docs)
    index = {code: i for i, code in enumerate(codes)}
    headcount = snap.groupby("team_code").size()
    hq = snap.groupby("team_code")["hq_name"].first()

    rows = []
    for row in docs.itertuples():
        core = _core(row.team_name)
        name_tokens = _name_tokens(core)
        if not name_tokens:
            continue

        body = str(row.document).lower().replace(" ", "")
        hit = [t for t in name_tokens if t in body]
        vector = vectors[index[row.team_code]]
        top_terms = sorted((t for t in vector if len(t) >= 3),
                           key=lambda t: vector[t], reverse=True)[:top_n]

        rows.append({
            "team_code": row.team_code, "team_name": row.team_name,
            "hq_name": hq.get(row.team_code), "headcount": int(headcount.get(row.team_code, 0)),
            "이름키워드": ", ".join(name_tokens),
            "역할에서_확인": ", ".join(hit) or "-",
            "일치율": round(len(hit) / len(name_tokens), 2),
            "실제_대표키워드": ", ".join(top_terms[:6]),
        })

    out = pd.DataFrame(rows)
    return out.sort_values(["일치율", "headcount"], ascending=[True, False]).reset_index(drop=True)


def vague_names(snap: pd.DataFrame) -> pd.DataFrame:
    """일반어만으로 구성되어 기능을 특정할 수 없는 조직명."""
    teams = snap[["team_code", "team_name", "hq_name", "dept_name"]].drop_duplicates()
    headcount = snap.groupby("team_code").size()

    rows = []
    for row in teams.itertuples():
        core = _core(row.team_name)
        tokens = re.findall(r"[가-힣]+|[A-Za-z]+", str(core or ""))
        tokens = [t.lower() for t in tokens if len(t) >= 2]
        if not tokens:
            continue
        specific = [t for t in tokens if t not in GENERIC]
        if specific:
            continue
        rows.append({"hq_name": row.hq_name, "dept_name": row.dept_name,
                     "team_name": row.team_name,
                     "headcount": int(headcount.get(row.team_code, 0)),
                     "사유": f"일반어만으로 구성({', '.join(tokens)})"})
    return (pd.DataFrame(rows).sort_values("headcount", ascending=False)
            .reset_index(drop=True) if rows else pd.DataFrame())


def naming_conventions(snap: pd.DataFrame) -> pd.DataFrame:
    """계층별 명명 규칙의 일관성. 접미사 혼재와 영문·국문 혼용을 본다."""
    rows = []
    for level, code_col, name_col in [("본부", "hq_code", "hq_name"),
                                      ("담당", "dept_code", "dept_name"),
                                      ("팀", "team_code", "team_name")]:
        names = snap[[code_col, name_col]].drop_duplicates()[name_col].dropna().tolist()
        if not names:
            continue
        suffix_count = Counter(next((s for s in SUFFIXES if n.endswith(s)), "(없음)")
                               for n in names)
        english = sum(_is_english(_core(n)) for n in names)
        rows.append({
            "계층": level, "조직수": len(names),
            "접미사분포": ", ".join(f"{k} {v}" for k, v in suffix_count.most_common()),
            "영문명": english,
            "영문비율(%)": round(english / len(names) * 100, 1),
        })
    return pd.DataFrame(rows)


def name_function_conflicts(roles: pd.DataFrame, snap: pd.DataFrame,
                            same_name_max_sim: float = 0.35,
                            diff_name_min_sim: float = 0.60,
                            exclude: dict[str, str] | None = None) -> dict[str, pd.DataFrame]:
    """이름과 기능이 어긋난 두 방향을 찾는다.

    - 이름은 닮았는데 하는 일이 다름 → 이름이 오해를 부른다
    - 하는 일은 같은데 이름이 다름 → 같은 기능이 여러 이름으로 불린다
    """
    docs = team_documents(roles)
    vectors, codes = _tfidf(docs)
    names = dict(zip(docs["team_code"], docs["team_name"]))
    headcount = snap.groupby("team_code").size()
    hq = snap.groupby("team_code")["hq_name"].first()

    exclude = exclude or {}
    similar_name, similar_role = [], []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            a, b = codes[i], codes[j]
            # 지역 병렬 조직군 내부는 이름이 같은 규칙을 따르는 것이 정상이다
            if a in exclude and exclude[a] == exclude.get(b):
                continue
            name_a, name_b = names[a], names[b]
            core_a, core_b = _core(name_a), _core(name_b)
            sim, shared = _cosine(vectors[i], vectors[j])

            # 핵심어가 같은데 역할이 다르다
            if core_a and core_a == core_b and sim < same_name_max_sim:
                similar_name.append({"팀 A": name_a, "팀 B": name_b,
                                     "본부 A": hq.get(a), "본부 B": hq.get(b),
                                     "역할유사도": round(sim, 3),
                                     "인원": int(headcount.get(a, 0) + headcount.get(b, 0)),
                                     "사유": "동일 명칭, 상이한 역할"})

            # 역할은 매우 닮았는데 이름에 공통 기능어가 없다
            if sim >= diff_name_min_sim:
                tokens_a, tokens_b = set(_name_tokens(core_a)), set(_name_tokens(core_b))
                if not (tokens_a & tokens_b):
                    similar_role.append({"팀 A": name_a, "팀 B": name_b,
                                         "본부 A": hq.get(a), "본부 B": hq.get(b),
                                         "역할유사도": round(sim, 3),
                                         "인원": int(headcount.get(a, 0) + headcount.get(b, 0)),
                                         "공통키워드": ", ".join(shared[:4])})

    return {
        "동명이의": (pd.DataFrame(similar_name).sort_values("인원", ascending=False)
                 .reset_index(drop=True) if similar_name else pd.DataFrame()),
        "이명동의": (pd.DataFrame(similar_role).sort_values("역할유사도", ascending=False)
                 .reset_index(drop=True) if similar_role else pd.DataFrame()),
    }
