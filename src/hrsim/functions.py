"""역할 정의표 → 기능 태깅 → 팀 간 중복 후보 탐지.

한국어 형태소 분석기 없이 동작하도록 규칙 기반 토크나이저 + TF-IDF 코사인 유사도로 구현했다.
정밀도를 더 올리려면 tag_functions() 를 LLM 기반 태거로 교체하면 된다(인터페이스 동일).
"""
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

from .config import Config

# 조사/어미로 끝나는 어절의 꼬리를 떼어 낸다
JOSA = ("으로서", "으로써", "에서의", "에게서", "에서", "에게", "으로", "까지", "부터", "라는",
        "이나", "이며", "하며", "하고", "하여", "한다", "하는", "했던", "이란",
        "을", "를", "이", "가", "은", "는", "의", "에", "와", "과", "로", "도", "및")
STOPWORDS = {"업무", "관련", "기타", "각종", "전반", "수행", "위한", "대한", "통한", "지원",
             "사항", "제반", "활동", "이하", "당사", "전사", "부문", "담당", "관리", "운영"}
MIN_TOKEN_LEN = 2


def tokenize(text: str) -> list[str]:
    """어절 분리 → 조사 제거 → 불용어 제거. 한글 3자 이상은 문자 바이그램도 함께 생성."""
    if not text:
        return []
    tokens: list[str] = []
    for word in re.findall(r"[가-힣A-Za-z0-9]+", str(text)):
        word = word.strip()
        for suffix in JOSA:
            if len(word) > len(suffix) + 1 and word.endswith(suffix):
                word = word[: -len(suffix)]
                break
        if len(word) < MIN_TOKEN_LEN or word in STOPWORDS:
            continue
        tokens.append(word.lower())
        # 복합명사를 부분 일치시키기 위한 바이그램 ('인사제도기획' ↔ '인사제도')
        if len(word) >= 4 and re.fullmatch(r"[가-힣]+", word):
            tokens += [word[i:i + 2] for i in range(len(word) - 1)]
    return tokens


def load_function_dict(path: str | Path = "config/function_dict.yaml") -> pd.DataFrame:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return pd.DataFrame([{"function_code": code, "function_name": spec["name"],
                          "keywords": spec.get("keywords", [])} for code, spec in raw.items()])


def team_documents(roles: pd.DataFrame) -> pd.DataFrame:
    """팀별 역할 텍스트를 하나의 문서로 합친다."""
    docs = (roles.groupby(["team_code", "team_name"])["role_item"]
            .apply(lambda s: " ".join(s)).rename("document").reset_index())
    docs["tokens"] = docs["document"].map(tokenize)
    docs["role_count"] = roles.groupby("team_code").size().reindex(docs["team_code"]).to_numpy()
    return docs


def tag_functions(roles: pd.DataFrame, func_dict: pd.DataFrame) -> pd.DataFrame:
    """역할 항목 하나하나를 기능 코드에 매핑한다(복수 매핑 허용)."""
    lookup = [(row.function_code, row.function_name, [k.lower() for k in row.keywords])
              for row in func_dict.itertuples()]
    rows = []
    for row in roles.itertuples():
        text = str(row.role_item).lower().replace(" ", "")
        for code, name, keywords in lookup:
            hits = [k for k in keywords if k.replace(" ", "") in text]
            if hits:
                rows.append({"team_code": row.team_code, "team_name": row.team_name,
                             "role_item": row.role_item, "function_code": code,
                             "function_name": name, "matched": ", ".join(hits)})
    tagged = pd.DataFrame(rows)
    if tagged.empty:
        return tagged

    untagged = set(roles["team_code"]) - set(tagged["team_code"])
    if untagged:  # 사전에 걸리지 않은 팀은 별도 코드로 남겨 검토 대상에 올린다
        leftover = roles[roles["team_code"].isin(untagged)].copy()
        leftover["function_code"] = "F-UNK"
        leftover["function_name"] = "미분류"
        leftover["matched"] = ""
        tagged = pd.concat([tagged, leftover], ignore_index=True)
    return tagged


def function_map(tagged: pd.DataFrame, snap: pd.DataFrame) -> pd.DataFrame:
    """기능 × 조직 지도. 기능별로 몇 개 조직이 몇 명을 투입하고 있는지."""
    if tagged.empty:
        return pd.DataFrame()

    headcount = snap.groupby("team_code").size().rename("headcount")
    hq = snap.groupby("team_code")["hq_name"].first().rename("hq_name")

    pairs = tagged[["function_code", "function_name", "team_code", "team_name"]].drop_duplicates()
    pairs = pairs.join(headcount, on="team_code").join(hq, on="team_code")
    pairs["headcount"] = pairs["headcount"].fillna(0).astype(int)

    summary = pairs.groupby(["function_code", "function_name"]).agg(
        team_count=("team_code", "nunique"),
        hq_count=("hq_name", "nunique"),
        total_headcount=("headcount", "sum"),
        teams=("team_name", lambda s: ", ".join(sorted(set(s)))),
    ).reset_index()
    summary["dispersion"] = summary.apply(
        lambda r: "단일" if r["team_count"] == 1
        else ("전사분산" if r["hq_count"] > 1 else "본부내중복"), axis=1)
    return summary.sort_values(["team_count", "total_headcount"], ascending=False).reset_index(drop=True)


def _tfidf(docs: pd.DataFrame) -> tuple[list[dict[str, float]], list[str]]:
    corpus = docs["tokens"].tolist()
    n = len(corpus)
    df_counts = Counter(token for tokens in corpus for token in set(tokens))
    vectors = []
    for tokens in corpus:
        counts = Counter(tokens)
        total = sum(counts.values()) or 1
        vector = {t: (c / total) * math.log((1 + n) / (1 + df_counts[t])) + 1e-9
                  for t, c in counts.items()}
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1.0
        vectors.append({t: v / norm for t, v in vector.items()})
    return vectors, docs["team_code"].tolist()


def _cosine(a: dict[str, float], b: dict[str, float]) -> tuple[float, list[str]]:
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    contributions = {t: v * longer[t] for t, v in shorter.items() if t in longer}
    score = sum(contributions.values())
    top = sorted(contributions, key=contributions.get, reverse=True)[:6]
    return score, [t for t in top if len(t) > 2] or top


def duplicate_candidates(roles: pd.DataFrame, tagged: pd.DataFrame,
                         snap: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """팀 쌍별 중복 후보. 역할 유사도와 공유 기능 코드를 결합해 점수화한다."""
    docs = team_documents(roles)
    vectors, codes = _tfidf(docs)
    names = dict(zip(docs["team_code"], docs["team_name"]))

    org = snap.groupby("team_code").agg(headcount=("emp_id", "count"),
                                        hq_name=("hq_name", "first"),
                                        dept_name=("dept_name", "first"))
    by_team = (tagged[tagged["function_code"] != "F-UNK"]
               .groupby("team_code")["function_code"].apply(set).to_dict()) if not tagged.empty else {}
    func_names = dict(zip(tagged["function_code"], tagged["function_name"])) if not tagged.empty else {}

    threshold = cfg.param("duplicate_sim_threshold", 0.30)
    rows = []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            a, b = codes[i], codes[j]
            sim, keywords = _cosine(vectors[i], vectors[j])
            shared = by_team.get(a, set()) & by_team.get(b, set())
            if sim < threshold and not shared:
                continue

            hq_a, hq_b = org["hq_name"].get(a), org["hq_name"].get(b)
            dept_a, dept_b = org["dept_name"].get(a), org["dept_name"].get(b)
            rows.append({
                "team_a": names.get(a, a), "team_b": names.get(b, b),
                "team_code_a": a, "team_code_b": b,
                "hq_a": hq_a, "hq_b": hq_b,
                "same_hq": hq_a == hq_b, "same_dept": dept_a == dept_b,
                "similarity": round(sim, 3),
                "shared_functions": ", ".join(sorted(func_names.get(f, f) for f in shared)),
                "shared_function_count": len(shared),
                "shared_keywords": ", ".join(keywords),
                "headcount_sum": int(org["headcount"].get(a, 0) + org["headcount"].get(b, 0)),
            })

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out["score"] = (out["similarity"] * 100 + out["shared_function_count"] * 15).round(1)
    out["판정초안"] = out.apply(_classify, axis=1)
    return out.sort_values("score", ascending=False).reset_index(drop=True)


def _classify(row: pd.Series) -> str:
    """중복의 성격을 3가지로 나눈 초안. 최종 판정은 반드시 사람이 한다."""
    if row["same_dept"]:
        return "진성중복(동일 담당 내)"
    if row["same_hq"]:
        return "진성중복 후보(동일 본부)"
    if row["shared_function_count"] >= 2 or row["similarity"] >= 0.45:
        return "전사분산 — 의도된 설계인지 확인 필요"
    return "경계모호 — R&R 재정의 검토"


def coverage_gaps(tagged: pd.DataFrame, func_dict: pd.DataFrame) -> pd.DataFrame:
    """사전에 정의된 기능 중 담당 조직이 없는 것."""
    if tagged.empty:
        return func_dict[["function_code", "function_name"]].copy()
    covered = set(tagged["function_code"])
    missing = func_dict[~func_dict["function_code"].isin(covered)]
    return missing[["function_code", "function_name"]].reset_index(drop=True)
