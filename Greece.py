import inspect
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import model as model_def


def get_features_and_labels():
    X_df = pd.read_csv("test_dataset.csv")
    y_df = pd.read_csv("test_labels.csv")
    y = y_df.iloc[:, -1].values.astype(int)

    # 1. Ищем список признаков в model.py (если задан явно)
    feature_cols = None
    for attr_name in dir(model_def):
        val = getattr(model_def, attr_name)
        if isinstance(val, (list, tuple)) and len(val) > 0 and all(isinstance(c, str) for c in val):
            if set(val).issubset(set(X_df.columns)):
                feature_cols = list(val)
                break

    # 2. Если есть функция препроцессинга в model.py — используем её
    if hasattr(model_def, "preprocess") and callable(model_def.preprocess):
        X = model_def.preprocess(X_df)
        if isinstance(X, pd.DataFrame):
            X = X.values
    elif feature_cols is not None:
        X = X_df[feature_cols].values
    else:
        drop_cols = [c for c in ["id", "encounter_id", "index"] if c in X_df.columns]
        X_clean = X_df.drop(columns=drop_cols)
        if X_clean.select_dtypes(include=["object", "category"]).shape[1] > 0:
            X_clean = pd.get_dummies(X_clean)
        X = X_clean.values

    return np.asarray(X, dtype=np.float32), y


def build_model(input_dim: int) -> nn.Module:
    # Находим класс нейросети в model.py
    net_classes = [
        obj for _, obj in inspect.getmembers(model_def, inspect.isclass)
        if issubclass(obj, nn.Module) and obj is not nn.Module
    ]
    if not net_classes:
        raise RuntimeError("В model.py не найден класс nn.Module")

    NetClass = net_classes[-1]
    sig = inspect.signature(NetClass.__init__)
    required_params = [
        p for name, p in sig.parameters.items()
        if name != "self" and p.default is inspect.Parameter.empty
        and p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if required_params:
        return NetClass(input_dim)
    return NetClass()


def load_weights(net: nn.Module, path: str) -> nn.Module:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, nn.Module):
        return ckpt
    if isinstance(ckpt, dict):
        state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
        net.load_state_dict(state_dict)
    return net


def main():
    X, y = get_features_and_labels()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    oracles = {
        "ARES": "models/model_ares.pth",
        "ATHENA": "models/model_athena.pth",
        "APOLLO": "models/model_apollo.pth",
    }

    results = []
    for name, weights_path in oracles.items():
        net = build_model(X.shape[1])
        net = load_weights(net, weights_path)
        net.eval()

        has_sigmoid = any(isinstance(m, nn.Sigmoid) for m in net.modules())
        with torch.no_grad():
            raw_out = net(X_tensor).squeeze(-1)
            probs = raw_out if has_sigmoid else torch.sigmoid(raw_out)
            probs = probs.cpu().numpy()

        preds = (probs >= 0.50).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, preds).ravel()

        acc = accuracy_score(y, preds)
        prec = precision_score(y, preds, zero_division=0)
        rec = recall_score(y, preds, zero_division=0)
        f1 = f1_score(y, preds, zero_division=0)
        f2 = fbeta_score(y, preds, beta=2.0, zero_division=0)
        roc_auc = roc_auc_score(y, probs)
        pr_auc = average_precision_score(y, probs)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Recall": rec,
            "Precision": prec,
            "F1": f1,
            "F2": f2,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "TP": tp,
            "FN (Missed!)": fn,
            "FP (False Alarm)": fp,
            "TN": tn,
        })

    res_df = pd.DataFrame(results)
    print("\n=== СРАВНЕНИЕ ОРАКУЛОВ (THRESHOLD = 0.50) ===")
    print(res_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Выбираем модель с максимальным Recall (и минимальным числом пропущенных угроз FN)
    best_row = res_df.sort_values(by=["Recall", "F2", "PR-AUC"], ascending=False).iloc[0]
    best_name = best_row["Model"]
    print(f"\nЛучший выбор для защиты Дельф: {best_name} (Recall = {best_row['Recall']:.4f}, FN = {int(best_row['FN (Missed!)'])})")
    print(f"FLAG: SIGMOID_{best_name}")


if __name__ == "__main__":
    main()