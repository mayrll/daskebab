import re
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import paired_cosine_distances

import bleu


def simple_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", str(text).lower())


def ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def card_exact_4gram_overlap(cand_toks: list[str], ref_toks_list: list[list[str]]) -> float:
    c4 = ngrams(cand_toks, 4)
    if not c4:
        return 0.0
    ref4 = set()
    for rt in ref_toks_list:
        ref4.update(ngrams(rt, 4))
    hits = sum(1 for g in c4 if g in ref4)
    return hits / len(c4)


def decode_flag(honest_seats: list[int]) -> str:
    assert len(honest_seats) == 24, "Должно быть ровно 24 раунда"
    digits = [s - 1 for s in honest_seats]
    letters = []
    for i in range(0, 24, 3):
        d1, d2, d3 = digits[i], digits[i + 1], digits[i + 2]
        val = 9 * d1 + 3 * d2 + d3
        if val > 25:
            letters.append(f"?({val})")
        else:
            letters.append(chr(ord("A") + val))
    return "SIGMOID_" + "".join(letters)


def main():
    print("=== 1. КОНФИГУРАЦИЯ HOUSE SCORER (bleu.py) ===")
    with open("bleu.py", "r", encoding="utf-8") as f:
        print(f.read())

    refs_df = pd.read_csv("references.csv")
    panel_df = pd.read_csv("panel.csv")

    # Группируем эталоны: (round, card) -> [ref1, ..., ref5]
    ref_map = (
        refs_df.groupby(["round", "card"])["reference"]
        .apply(list)
        .to_dict()
    )

    # Считаем TF-IDF близость каждого ответа к объединённому тексту 5 эталонов карточки
    card_keys = sorted(ref_map.keys())
    joined_refs = [" ".join(ref_map[k]) for k in card_keys]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    ref_tfidf = vec.fit_transform(joined_refs)
    key_to_idx = {k: i for i, k in enumerate(card_keys)}

    cand_idx = [key_to_idx[(r, c)] for r, c in zip(panel_df["round"], panel_df["card"])]
    cand_tfidf = vec.transform(panel_df["text"].fillna("").astype(str))
    panel_df["tfidf_sim"] = 1.0 - paired_cosine_distances(cand_tfidf, ref_tfidf[cand_idx])

    # Карточные метрики анти-блефа
    card_stats = []
    for row in panel_df.itertuples():
        toks = simple_tokens(row.text)
        refs_toks = [simple_tokens(r) for r in ref_map[(row.round, row.card)]]
        ref_mean_len = np.mean([len(rt) for rt in refs_toks]) if refs_toks else 1.0

        # 1) Внутренние повторы (keyword stuffing внутри одной карточки)
        ttr = len(set(toks)) / max(len(toks), 1)
        bg = ngrams(toks, 2)
        bg_rep = 1.0 - (len(set(bg)) / max(len(bg), 1)) if bg else 0.0

        # 2) Дословное совпадение 4-грамм с эталонами (копипаста на отдельных карточках)
        overlap4 = card_exact_4gram_overlap(toks, refs_toks)

        # 3) Отклонение длины от эталона
        len_ratio = len(toks) / max(ref_mean_len, 1.0)

        card_stats.append((len(toks), ttr, bg_rep, overlap4, len_ratio))

    panel_df[["tok_len", "ttr", "bg_rep", "overlap4", "len_ratio"]] = card_stats

    # Находим функцию расчёта корпусного BLEU в bleu.py
    bleu_fn = None
    for fn_name in ["corpus_bleu", "score_deck", "compute_bleu", "bleu_score", "score"]:
        if hasattr(bleu, fn_name) and callable(getattr(bleu, fn_name)):
            bleu_fn = getattr(bleu, fn_name)
            break

    # Агрегируем по колодам (round, seat)
    deck_rows = []
    for (rnd, seat), grp in panel_df.groupby(["round", "seat"]):
        grp = grp.sort_values("card")
        cands = grp["text"].fillna("").astype(str).tolist()
        refs_list = [ref_map[(rnd, c)] for c in grp["card"]]

        house_bleu = float(bleu_fn(cands, refs_list)) if bleu_fn else np.nan

        # Повторы одинаковых 3-грамм МЕЖДУ разными карточками одной колоды (шаблонный спам)
        all_trigrams = []
        for t in cands:
            all_trigrams.extend(set(ngrams(simple_tokens(t), 3)))
        tri_counts = Counter(all_trigrams)
        cross_card_template_rate = (
            sum(cnt - 1 for cnt in tri_counts.values() if cnt > 1) / max(len(all_trigrams), 1)
        )

        # Пиковость 4-грамм (блеф на нескольких карточках: max - median по колоде)
        ov4 = grp["overlap4"].values
        spike_4gram = float(np.max(ov4) - np.median(ov4))
        max_4gram = float(np.max(ov4))

        # Смысловое покрытие худших карточек колоды (честный держит уровень на ВСЕХ 40 карточках)
        q10_tfidf = float(np.percentile(grp["tfidf_sim"], 10))
        mean_tfidf = float(grp["tfidf_sim"].mean())

        # Внутренняя заспамленность и аномалии длины
        mean_bg_rep = float(grp["bg_rep"].mean())
        mean_ttr = float(grp["ttr"].mean())
        len_dev = float(np.abs(np.log(np.clip(grp["len_ratio"], 0.1, 10.0))).mean())

        deck_rows.append({
            "round": rnd,
            "seat": seat,
            "house_bleu": house_bleu,
            "q10_tfidf": q10_tfidf,
            "mean_tfidf": mean_tfidf,
            "max_4gram": max_4gram,
            "spike_4gram": spike_4gram,
            "template_rep": cross_card_template_rate,
            "bg_rep": mean_bg_rep,
            "ttr": mean_ttr,
            "len_dev": len_dev,
        })

    decks = pd.DataFrame(deck_rows)

    # Нормализуем признаки внутри каждого раунда (z-score / rank среди 3 участников)
    # Честный участник: высокий q10_tfidf (стабильный смысл на всех карточках),
    # низкий spike_4gram / max_4gram (нет копипасты), низкий template_rep / bg_rep (нет спама), низкий len_dev.
    honest_seats = []
    print("\n=== 2. АНАЛИЗ ПО РАУНДАМ ===")
    for rnd, rgrp in decks.groupby("round"):
        rgrp = rgrp.copy()

        def norm(col):
            s = rgrp[col]
            return (s - s.mean()) / (s.std() + 1e-9)

        # Составной индекс «честности» (выше = честнее)
        rgrp["honesty_score"] = (
            +1.5 * norm("q10_tfidf")
            + 1.0 * norm("mean_tfidf")
            - 1.2 * norm("spike_4gram")
            - 1.0 * norm("max_4gram")
            - 1.2 * norm("template_rep")
            - 1.2 * norm("bg_rep")
            - 1.0 * norm("len_dev")
        )
        winner_row = rgrp.sort_values("honesty_score", ascending=False).iloc[0]
        honest_seats.append(int(winner_row["seat"]))

        if rnd <= 4:
            print(f"\n--- Round {rnd} (Honest = Seat {int(winner_row['seat'])}) ---")
            cols_show = ["seat", "house_bleu", "q10_tfidf", "max_4gram", "template_rep", "bg_rep", "len_dev", "honesty_score"]
            print(rgrp[cols_show].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n=== 3. ПРИМЕРЫ ТЕКСТОВ (ROUND 1, CARD 1 & CARD С МАКС. 4-GRAM OVERLAP) ===")
    for seat in [1, 2, 3]:
        sub = panel_df[(panel_df["round"] == 1) & (panel_df["seat"] == seat)].sort_values("overlap4", ascending=False)
        top_card = sub.iloc[0]
        print(f"\n[Seat {seat}] Card {int(top_card['card'])} (overlap4={top_card['overlap4']:.2f}, tfidf={top_card['tfidf_sim']:.2f}):")
        print("  TEXT:", top_card["text"])
        print("  REF1:", ref_map[(1, int(top_card["card"]))][0])

    flag = decode_flag(honest_seats)
    print("\n==========================================")
    print("Honest seats (Rounds 1..24):", honest_seats)
    print("DECODED FLAG:", flag)
    print("==========================================")


if __name__ == "__main__":
    main()