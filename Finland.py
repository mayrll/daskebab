import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import pandas as pd


# Словарь шведских и альтернативных названий финских городов -> каноническое имя
SWEDISH_TO_FINNISH = {
    "helsingfors": "helsinki",
    "abo": "turku",
    "tammerfors": "tampere",
    "uleaborg": "oulu",
    "vasa": "vaasa",
    "borga": "porvoo",
    "bjorneborg": "pori",
    "karleby": "kokkola",
    "jakobstad": "pietarsaari",
    "ekenas": "tammisaari",
    "mariehamn": "maarianhamina",
    "hango": "hanko",
    "lovisa": "loviisa",
    "nystad": "uusikaupunki",
    "raumo": "rauma",
    "tornea": "tornio",
    "kristinestad": "kristiinankaupunki",
    "kasko": "kaskinen",
    "narpes": "narpio",
    "nykarleby": "uusikaarlepyy",
    "sibbo": "sipoo",
    "kyrkslatt": "kirkkonummi",
    "inga": "inkoo",
    "pargas": "parainen",
    "kimitoon": "kemionsaari",
    "brahestad": "raahe",
    "tavastehus": "hameenlinna",
    "villmanstrand": "lappeenranta",
    "nyslott": "savonlinna",
    "st michel": "mikkeli",
    "sankt michel": "mikkeli",
    "kajana": "kajaani",
    "fredrikshamn": "hamina",
    "grankulla": "kauniainen",
    "esbo": "espoo",
    "vanda": "vantaa",
    "lahtis": "lahti",
    "lojo": "lohja",
    "nadendal": "naantali",
    "reso": "raisio",
    "st karins": "kaarina",
    "idensalmi": "iisalmi",
    "lappo": "lapua",
    "virdois": "virrat",
    "vittis": "huittinen",
    "ikalis": "ikaalinen",
    "enare": "inari",
    "hogfors": "karkkila",
    "kemi": "kemi",
}


def ascii_fold(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def normalize_town(raw: str) -> str:
    if pd.isna(raw):
        return ""
    s = str(raw).strip().lower()
    # Убираем содержимое в скобках, например "(FI)" или "(ferry)"
    s = re.sub(r"\([^)]*\)", " ", s)
    # Убираем типовые приставки и хвосты ручного ввода
    s = re.sub(
        r"\b(port of|harbour of|harbor of|city of|town of|port|harbour|harbor|satama|terminal|station|ferry|pier)\b",
        " ",
        s,
    )
    s = re.sub(r"[^a-zäöåéü0-9\s-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip(" -")
    folded = ascii_fold(s)
    return SWEDISH_TO_FINNISH.get(folded, folded)


def build_graph(df: pd.DataFrame, fuzzy_threshold: float = 0.90):
    u_raw = df["from_town"].map(normalize_town)
    v_raw = df["to_town"].map(normalize_town)

    # Склеиваем мелкие опечатки к более частотному варианту написания
    freq = Counter(u_raw) + Counter(v_raw)
    freq.pop("", None)
    sorted_names = [name for name, _ in freq.most_common()]

    canonical_map = {}
    kept = []
    for name in sorted_names:
        matched = None
        if len(name) >= 5:
            for k in kept:
                if name[0] == k[0] and abs(len(name) - len(k)) <= 2:
                    if SequenceMatcher(None, name, k).ratio() >= fuzzy_threshold:
                        matched = k
                        break
        if matched:
            canonical_map[name] = matched
        else:
            canonical_map[name] = name
            kept.append(name)

    edges = set()
    towns = set()
    for a, b in zip(u_raw, v_raw):
        ca = canonical_map.get(a, a)
        cb = canonical_map.get(b, b)
        if not ca or not cb or ca == cb:
            continue
        towns.add(ca)
        towns.add(cb)
        edges.add(tuple(sorted((ca, cb))))

    adj = defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)

    ranking = sorted(
        [(town, len(neighbors)) for town, neighbors in adj.items()],
        key=lambda x: (-x[1], x[0]),
    )
    return towns, edges, ranking, canonical_map


def main():
    df = pd.read_csv("routes.csv")
    print(f"Всего строк в routes.csv: {len(df)}")
    print(f"Офисы: {df['office'].nunique()} -> {sorted(df['office'].dropna().unique())}")
    print(f"Типы маршрутов:\n{df['route_type'].value_counts(dropna=False)}\n")

    subsets = [
        ("ALL route_types", df),
        ("ONLY ferry", df[df["route_type"].str.lower().str.strip() == "ferry"]),
        ("NO bus", df[df["route_type"].str.lower().str.strip() != "bus"]),
    ]

    for label, sub_df in subsets:
        for thresh in [1.0, 0.92, 0.88]:
            towns, edges, ranking, cmap = build_graph(sub_df, fuzzy_threshold=thresh)
            top8_deg = [d for _, d in ranking[:8]]
            strict_top8 = len(set(top8_deg)) == 8
            is_match = (len(towns) == 116 and len(edges) == 448 and strict_top8)

            status = "MATCH!" if is_match else "no"
            print(
                f"[{status:6s}] {label:16s} | fuzzy={thresh:.2f} -> "
                f"towns={len(towns)}, edges={len(edges)}, top8_deg={top8_deg}"
            )

            if is_match:
                print("\n=== ТОП-10 ГОРОДОВ ПО ЧИСЛУ УНИКАЛЬНЫХ СВЯЗЕЙ ===")
                for i, (town, deg) in enumerate(ranking[:10], 1):
                    print(f"{i:2d}. {town.title():20s} -> {deg} connections")

                word = "".join(town[0].upper() for town, _ in ranking[:7])
                print(f"\nFLAG: SIGMOID_{word}\n")
                return

    # Если авто-режим чуть не попал в 116 / 448 — выводим диагностику для быстрой правки
    print("\n=== ДИАГНОСТИКА (если 116/448 не совпали точно) ===")
    towns, edges, ranking, cmap = build_graph(df, fuzzy_threshold=1.0)
    merged = {k: v for k, v in cmap.items() if k != v}
    print(f"Склеенные дубликаты ({len(merged)}): {merged}")
    print("Топ-12 городов (ALL, без fuzzy):")
    for i, (town, deg) in enumerate(ranking[:12], 1):
        print(f"  {i:2d}. {town} ({deg})")


if __name__ == "__main__":
    main()