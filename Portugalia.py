import numpy as np
import pandas as pd


def check_reversal(df: pd.DataFrame, group_series: pd.Series):
    tmp = pd.DataFrame({
        "group": group_series,
        "treatment": df["treatment"],
        "success": df["harvest_success"],
    }).dropna()

    stats = (
        tmp.groupby(["group", "treatment"], observed=True)["success"]
        .agg(["count", "mean"])
        .unstack("treatment")
    )

    if ("mean", "control") not in stats.columns or ("mean", "treatment") not in stats.columns:
        return False, None

    # Убираем пустые группы, если такие возникли при нарезке
    valid = stats.dropna()
    if len(valid) < 2:
        return False, None

    ctrl_rates = valid[("mean", "control")]
    trt_rates = valid[("mean", "treatment")]

    # Полный парадокс Симпсона: в КАЖДОЙ группе control строго лучше treatment
    is_full_reversal = bool((trt_rates < ctrl_rates).all())
    return is_full_reversal, valid


def format_table(stats_df: pd.DataFrame) -> str:
    rows = []
    for grp, row in stats_df.iterrows():
        c_n = int(row[("count", "control")])
        t_n = int(row[("count", "treatment")])
        c_rate = row[("mean", "control")]
        t_rate = row[("mean", "treatment")]
        diff = t_rate - c_rate
        rows.append(
            f"| {grp} | {c_n} | {c_rate:.2%} | {t_n} | {t_rate:.2%} | {diff:+.2%} |"
        )
    header = (
        "| Group | Control N | Control Success | Treatment N | Treatment Success | Diff (Trt - Ctrl) |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: |"
    )
    return header + "\n" + "\n".join(rows)


def main():
    df = pd.read_csv("douro_vineyard_data.csv")

    overall = df.groupby("treatment")["harvest_success"].agg(["count", "mean"])
    ctrl_overall = overall.loc["control", "mean"]
    trt_overall = overall.loc["treatment", "mean"]
    print("=== ОБЩИЙ РЕЗУЛЬТАТ (POOLED) ===")
    print(f"Control   (n={int(overall.loc['control', 'count'])}): {ctrl_overall:.4%}")
    print(f"Treatment (n={int(overall.loc['treatment', 'count'])}): {trt_overall:.4%}\n")

    ignore_cols = {"plot_id", "treatment", "harvest_success"}
    candidate_cols = [c for c in df.columns if c not in ignore_cols]

    found = []

    for col in candidate_cols:
        s = df[col]

        # 1. Проверка как категориальной / дискретной переменной
        if s.dtype == "object" or s.nunique() <= 20:
            ok, stats = check_reversal(df, s)
            if ok:
                found.append((col, "Exact categorical values", stats))

        # 2. Если колонка числовая — проверяем разбиение на группы (bins / quantiles / thresholds)
        if pd.api.types.is_numeric_dtype(s):
            # Равные квантили (qcut) и равные интервалы (cut) на 2..5 групп
            for k in [2, 3, 4, 5]:
                for method_name, binner in [
                    (f"Quantiles (q={k})", lambda x: pd.qcut(x, q=k, duplicates="drop")),
                    (f"Equal-width bins (k={k})", lambda x: pd.cut(x, bins=k)),
                ]:
                    try:
                        binned = binner(s)
                        ok, stats = check_reversal(df, binned)
                        if ok:
                            found.append((col, method_name, stats))
                    except Exception:
                        pass

            # Поиск точного 1-порогового разбиения (Low vs High) по перцентилям
            qs = np.percentile(s.dropna(), np.arange(10, 95, 5))
            for thr in np.unique(np.round(qs, 2)):
                binned = pd.Series(
                    np.where(s <= thr, f"<= {thr}", f"> {thr}"), index=df.index
                )
                ok, stats = check_reversal(df, binned)
                if ok:
                    found.append((col, f"Threshold split at {thr}", stats))
                    break

    if not found:
        print("Полный разворот не найден стандартными сетками. Скинь первые 20 строк CSV.")
        return

    # Группируем находки по имени колонки
    seen_cols = {}
    for col, split_desc, stats in found:
        if col not in seen_cols:
            seen_cols[col] = (split_desc, stats)

    for col, (split_desc, stats) in seen_cols.items():
        flag = f"SIGMOID_{col.upper()}"
        print(f"=== НАЙДЕН КОНФАУНДЕР: {col} ({split_desc}) ===")
        print(format_table(stats))
        print(f"\nFLAG: {flag}\n")


if __name__ == "__main__":
    main()