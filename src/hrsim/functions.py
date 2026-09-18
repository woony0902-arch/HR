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


def tag_functions(roles: pd.DataFrame, func_dict: pd.DataFrame,
                  tagging_types: list[str] | None = None) -> pd.DataFrame:
    """역할 항목을 기능 코드에 매핑한다(복수 매핑 허용).

    세부 R&R 에는 '안전 교육 이수' 같은 부수적 언급이 많아 그대로 태깅하면
    모든 조직이 모든 기능을 가진 것처럼 보인다. 선언된 핵심 역할(미션·주요 R&R)만
    태깅 대상으로 삼고, 세부 R&R 은 유사도 계산에만 사용한다.
    """
    scope = roles
    if tagging_types and "role_type" in roles.columns:
        scope = roles[roles["role_type"].isin(tagging_types)]
        if scope.empty:
            scope = roles
    lookup = [(row.function_code, row.function_name, [k.lower() for k in row.keywords])
              for row in func_dict.itertuples()]
    rows = []
    for row in scope.itertuples():
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

    untagged = set(scope["team_code"]) - set(tagged["team_code"])
    if untagged:  # 사전에 걸리지 않은 팀은 별도 코드로 남겨 검토 대상에 올린다
        leftover = scope[scope["team_code"].isin(untagged)].drop_duplicates("team_code").copy()
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


def common_suffix(a: str, b: str) -> str:
    """두 조직명의 최장 공통 접미사. '강남운용팀' vs '동작운용팀' → '운용팀'."""
    limit = min(len(a), len(b))
    i = 0
    while i < limit and a[-(i + 1)] == b[-(i + 1)]:
        i += 1
    return a[len(a) - i:] if i else ""


def _shared_suffix(names: list[str]) -> str:
    """여러 조직명의 공통 접미사."""
    suffix = names[0]
    for name in names[1:]:
        suffix = common_suffix(suffix, name)
        if not suffix:
            break
    return suffix


def parallel_families(roles: pd.DataFrame, snap: pd.DataFrame,
                      min_size: int = 3, min_sim: float = 0.65,
                      min_suffix: int = 2) -> pd.DataFrame:
    """같은 기능을 지역·채널로 나눠 수행하는 병렬 조직군.

    접미사가 같다는 것만으로는 부족하고('인사기획팀'과 '전략기획팀'은 다른 기능이다),
    접미사로 먼저 묶으면 '주영업팀'이 '영업팀'을 가려 버린다.
    그래서 **역할 문서 유사도로 먼저 군집을 만들고**, 그 군집이 공통 접미사를 공유할 때만
    병렬 조직으로 인정한다. 통폐합 검토는 쌍이 아니라 이 군 단위로 해야 한다.
    """
    docs = team_documents(roles)
    vectors, codes = _tfidf(docs)
    names = dict(zip(docs["team_code"], docs["team_name"]))
    headcount = snap.groupby("team_code").size()

    # 유사도 임계 이상으로 연결된 조직들을 연결 요소로 묶는다
    parent = {c: c for c in codes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pair_sims: dict[tuple[str, str], float] = {}
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            sim, _ = _cosine(vectors[i], vectors[j])
            pair_sims[(codes[i], codes[j])] = sim
            if sim >= min_sim:
                parent[find(codes[i])] = find(codes[j])

    clusters: dict[str, list[str]] = {}
    for code in codes:
        clusters.setdefault(find(code), []).append(code)

    rows = []
    for members in clusters.values():
        if len(members) < min_size:
            continue
        labels = [names[c] for c in members]
        suffix = _shared_suffix(labels)
        if len(suffix) < min_suffix:
            continue      # 이름 규칙을 공유하지 않으면 병렬 구조가 아니라 실제 중복일 수 있다
        sims = [v for (a, b), v in pair_sims.items() if a in members and b in members]
        rows.append({"공통기능": suffix, "조직수": len(members),
                     "평균유사도": round(sum(sims) / len(sims), 3) if sims else 0.0,
                     "총인원": int(headcount.reindex(members).fillna(0).sum()),
                     "평균인원": round(float(headcount.reindex(members).fillna(0).mean()), 1),
                     "조직": ", ".join(sorted(labels)),
                     "codes": members})

    if not rows:
        return pd.DataFrame()

    # 같은 접미사의 군집은 하나의 군으로 합치고, 같은 이름 규칙을 쓰는 나머지 조직도 흡수한다.
    # (대전영업팀·천안영업팀처럼 둘만 묶인 군집은 min_size 에 걸려 빠지지만 같은 구조다)
    merged: dict[str, set[str]] = {}
    for row in rows:
        merged.setdefault(row["공통기능"], set()).update(row["codes"])
    def sim_of(a: str, b: str) -> float:
        return pair_sims.get((a, b), pair_sims.get((b, a), 0.0))

    for suffix, core in merged.items():
        # 이름 규칙이 같아도 역할이 다르면 흡수하지 않는다(예: '전략영업팀', 'IP망운용팀')
        candidates = [c for c, n in names.items()
                      if n.endswith(suffix) and len(n) > len(suffix) and c not in core]
        core.update(c for c in candidates
                    if max((sim_of(c, m) for m in core), default=0.0) >= 0.45)

    final = []
    for suffix, members in merged.items():
        listed = sorted(members)
        sims = [v for (a, b), v in pair_sims.items() if a in members and b in members]
        final.append({"공통기능": suffix, "조직수": len(listed),
                      "평균유사도": round(sum(sims) / len(sims), 3) if sims else 0.0,
                      "총인원": int(headcount.reindex(listed).fillna(0).sum()),
                      "평균인원": round(float(headcount.reindex(listed).fillna(0).mean()), 1),
                      "조직": ", ".join(sorted(names[c] for c in listed)),
                      "codes": listed})

    return (pd.DataFrame(final).sort_values(["조직수", "총인원"], ascending=False)
            .reset_index(drop=True))


def parallel_membership(families: pd.DataFrame) -> dict[str, str]:
    """팀코드 → 소속 병렬 조직군 이름."""
    if families.empty or "codes" not in families:
        return {}
    return {code: row.공통기능 for row in families.itertuples() for code in row.codes}


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
                         snap: pd.DataFrame, cfg: Config,
                         families: pd.DataFrame | None = None) -> pd.DataFrame:
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

    family_of = parallel_membership(families if families is not None
                                    else parallel_families(roles, snap))
    threshold = cfg.param("duplicate_sim_threshold", 0.30)
    rows = []
    for i in range(len(codes)):
        for j in range(i + 1, len(codes)):
            a, b = codes[i], codes[j]
            sim, keywords = _cosine(vectors[i], vectors[j])
            shared = by_team.get(a, set()) & by_team.get(b, set())
            # 텍스트가 충분히 겹치거나, 기능 코드가 2개 이상 겹칠 때만 후보로 올린다.
            # 둘 중 하나만 약하게 걸리는 쌍까지 담으면 목록이 수천 건이 되어 쓸 수 없다.
            if not (sim >= threshold or (len(shared) >= 2 and sim >= threshold / 2)):
                continue

            name_a, name_b = names.get(a, a), names.get(b, b)
            family = family_of.get(a)
            if family is not None and family == family_of.get(b):
                continue          # 같은 병렬 조직군 내부 쌍은 군 단위로 따로 본다

            hq_a, hq_b = org["hq_name"].get(a), org["hq_name"].get(b)
            dept_a, dept_b = org["dept_name"].get(a), org["dept_name"].get(b)
            rows.append({
                "team_a": name_a, "team_b": name_b,
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


def keyword_candidates(roles: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """팀별 대표 키워드. 회사 실정에 맞는 기능 분류 사전을 만들 때의 출발점이다.

    시드 사전은 일반적인 기업 기능으로 채운 것이라 회사 고유 용어를 담지 못한다.
    각 팀에서 가장 변별력 있는 용어(TF-IDF 상위)를 뽑아 주면 사전 확정 작업이 빨라진다.
    """
    docs = team_documents(roles)
    vectors, codes = _tfidf(docs)
    names = dict(zip(docs["team_code"], docs["team_name"]))

    rows = []
    for code, vector in zip(codes, vectors):
        words = {t: v for t, v in vector.items() if len(t) >= 3}   # 바이그램 제외
        top = sorted(words, key=words.get, reverse=True)[:top_n]
        rows.append({"team_code": code, "team_name": names.get(code, code),
                     "대표키워드": ", ".join(top)})
    return pd.DataFrame(rows)


def corpus_terms(roles: pd.DataFrame, top_n: int = 60) -> pd.DataFrame:
    """전사에서 가장 자주 쓰이는 역할 용어. 사전에 빠진 기능을 찾는 데 쓴다."""
    counter: Counter[str] = Counter()
    doc_freq: Counter[str] = Counter()
    for _, group in roles.groupby("team_code"):
        tokens = [t for item in group["role_item"] for t in tokenize(item) if len(t) >= 3]
        counter.update(tokens)
        doc_freq.update(set(tokens))
    rows = [{"용어": term, "총출현": counter[term], "보유팀수": doc_freq[term]}
            for term in counter]
    return (pd.DataFrame(rows).sort_values("보유팀수", ascending=False)
            .head(top_n).reset_index(drop=True))


def coverage_gaps(tagged: pd.DataFrame, func_dict: pd.DataFrame) -> pd.DataFrame:
    """사전에 정의된 기능 중 담당 조직이 없는 것."""
    if tagged.empty:
        return func_dict[["function_code", "function_name"]].copy()
    covered = set(tagged["function_code"])
    missing = func_dict[~func_dict["function_code"].isin(covered)]
    return missing[["function_code", "function_name"]].reset_index(drop=True)
