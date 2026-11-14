import os
import subprocess
import sys
import warnings
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


def inspect_evaluator_mode() -> str:
    """Определяет по baseline_solution.py / evaluation_script.py, нужны 0/1 или вероятности."""
    for fname in ["evaluation_script.py", "baseline_solution.py"]:
        if os.path.exists(fname):
            with open(fname, "r", encoding="utf-8", errors="ignore") as f:
                txt = f.read()
            if any(k in txt for k in ["roc_auc_score", "average_precision_score", "log_loss", "brier_score"]):
                return "proba"
            if any(k in txt for k in ["f1_score", "accuracy_score", "matthews_corrcoef", "confusion_matrix", "predict("]):
                return "binary"
    return "binary"


def extract_network_features(X: np.ndarray, nets: list[np.ndarray], prefix: str) -> pd.DataFrame:
    """Физические признаки когерентности 3 сетей и спектральной формы."""
    feats = {}
    eps = 1e-8

    energies = []
    maxs = []
    lows = []
    mids = []
    highs = []
    centroids = []
    entropies = []

    bands = np.arange(1, 17, dtype=np.float64)

    for idx, net in enumerate(nets, 1):
        net_pos = np.maximum(net, 0.0)
        e_sum = np.sum(net_pos, axis=1)
        e_l2 = np.linalg.norm(net_pos, axis=1)
        e_max = np.max(net_pos, axis=1)
        e_std = np.std(net_pos, axis=1)

        low = np.sum(net_pos[:, :5], axis=1)
        mid = np.sum(net_pos[:, 5:11], axis=1)
        high = np.sum(net_pos[:, 11:], axis=1)

        probs = net_pos / (e_sum[:, None] + eps)
        centroid = np.sum(probs * bands[None, :], axis=1)
        entropy = -np.sum(probs * np.log(probs + 1e-12), axis=1)

        feats[f"{prefix}_n{idx}_sum"] = np.log1p(e_sum)
        feats[f"{prefix}_n{idx}_l2"] = np.log1p(e_l2)
        feats[f"{prefix}_n{idx}_max"] = np.log1p(e_max)
        feats[f"{prefix}_n{idx}_crest"] = e_max / (e_sum / 16.0 + eps)
        feats[f"{prefix}_n{idx}_low_high"] = np.log1p(low) - np.log1p(high)
        feats[f"{prefix}_n{idx}_mid_high"] = np.log1p(mid) - np.log1p(high)
        feats[f"{prefix}_n{idx}_low_share"] = low / (e_sum + eps)
        feats[f"{prefix}_n{idx}_high_share"] = high / (e_sum + eps)
        feats[f"{prefix}_n{idx}_centroid"] = centroid
        feats[f"{prefix}_n{idx}_entropy"] = entropy
        feats[f"{prefix}_n{idx}_std_mean"] = e_std / (e_sum / 16.0 + eps)

        energies.append(e_sum)
        maxs.append(e_max)
        lows.append(low)
        mids.append(mid)
        highs.append(high)
        centroids.append(centroid)
        entropies.append(entropy)

    e1, e2, e3 = energies
    e_stack = np.vstack([e1, e2, e3])
    e_min = np.min(e_stack, axis=0)
    e_max_all = np.max(e_stack, axis=0)
    e_mid = e1 + e2 + e3 - e_min - e_max_all

    # Ключевые индикаторы одновременного отклика всех 3 сетей (Vrancea vs локальный шум)
    feats[f"{prefix}_min_net_energy"] = np.log1p(e_min)
    feats[f"{prefix}_mid_net_energy"] = np.log1p(e_mid)
    feats[f"{prefix}_max_net_energy"] = np.log1p(e_max_all)
    feats[f"{prefix}_geom_net_energy"] = (np.log1p(e1) + np.log1p(e2) + np.log1p(e3)) / 3.0
    feats[f"{prefix}_coherence_min_max"] = e_min / (e_max_all + eps)
    feats[f"{prefix}_coherence_mid_max"] = e_mid / (e_max_all + eps)
    feats[f"{prefix}_net_energy_cv"] = np.std(e_stack, axis=0) / (np.mean(e_stack, axis=0) + eps)

    # Попарные корреляции формы спектра между 3 сетями
    for (i, a), (j, b) in [((1, nets[0]), (2, nets[1])), ((1, nets[0]), (3, nets[2])), ((2, nets[1]), (3, nets[2]))]:
        ac = a - np.mean(a, axis=1, keepdims=True)
        bc = b - np.mean(b, axis=1, keepdims=True)
        num = np.sum(ac * bc, axis=1)
        den = np.sqrt(np.sum(ac ** 2, axis=1) * np.sum(bc ** 2, axis=1)) + eps
        feats[f"{prefix}_corr_{i}_{j}"] = num / den

    # Согласованность спектрального центроида и энтропии между сетями
    c_stack = np.vstack(centroids)
    feats[f"{prefix}_centroid_std"] = np.std(c_stack, axis=0)
    feats[f"{prefix}_centroid_mean"] = np.mean(c_stack, axis=0)
    feats[f"{prefix}_entropy_mean"] = np.mean(np.vstack(entropies), axis=0)

    # Минимальная низкочастотная энергия по всем 3 сетям
    l_min = np.min(np.vstack(lows), axis=0)
    h_max = np.max(np.vstack(highs), axis=0)
    feats[f"{prefix}_min_low_vs_max_high"] = np.log1p(l_min) - np.log1p(h_max)

    return pd.DataFrame(feats)


def build_features(df: pd.DataFrame, channel_cols: list[str]) -> pd.DataFrame:
    X = df[channel_cols].values.astype(np.float64)

    # Базовые и логарифмические амплитуды по всем 48 каналам
    X_log = np.sign(X) * np.log1p(np.abs(X))
    base_df = pd.DataFrame(X_log, columns=[f"log_{c}" for c in channel_cols])

    # Глобальная статистика по всем 48 каналам
    X_pos = np.maximum(X, 0.0)
    global_df = pd.DataFrame({
        "all_sum": np.log1p(np.sum(X_pos, axis=1)),
        "all_max": np.log1p(np.max(X_pos, axis=1)),
        "all_min": np.log1p(np.min(X_pos, axis=1)),
        "all_q25": np.log1p(np.percentile(X_pos, 25, axis=1)),
        "all_q50": np.log1p(np.percentile(X_pos, 50, axis=1)),
        "all_q75": np.log1p(np.percentile(X_pos, 75, axis=1)),
        "all_active_cnt": np.sum(X_pos > np.median(X_pos), axis=1),
    })

    # Вариант А раскладки сетей: 3 непрерывных блока по 16 каналов (c01..c16, c17..c32, c33..c48)
    nets_block = [X[:, 0:16], X[:, 16:32], X[:, 32:48]]
    feat_block = extract_network_features(X, nets_block, prefix="blk")

    # Вариант Б раскладки сетей: чередование по 3 сети (c01,c04..., c02,c05..., c03,c06...)
    nets_stride = [X[:, 0::3], X[:, 1::3], X[:, 2::3]]
    feat_stride = extract_network_features(X, nets_stride, prefix="str")

    full_df = pd.concat([base_df, global_df, feat_block, feat_stride], axis=1)
    return full_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def get_candidate_models():
    models = {
        "HistGradientBoosting": lambda seed: HistGradientBoostingClassifier(
            max_iter=350,
            learning_rate=0.04,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            random_state=seed,
        ),
        "ExtraTrees": lambda seed: ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        ),
        "RandomForest": lambda seed: RandomForestClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=seed,
        ),
        "MLP_NeuralNet": lambda seed: make_pipeline(
            QuantileTransformer(n_quantiles=500, output_distribution="normal", random_state=seed),
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                alpha=1e-2,
                batch_size=128,
                learning_rate_init=1e-3,
                max_iter=150,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=seed,
            ),
        ),
        "RBF_SVM": lambda seed: make_pipeline(
            QuantileTransformer(n_quantiles=500, output_distribution="normal", random_state=seed),
            StandardScaler(),
            SVC(C=3.0, kernel="rbf", probability=True, random_state=seed),
        ),
    }

    # Автоматически подключаем LightGBM / XGBoost / CatBoost, если они установлены в окружении
    try:
        import lightgbm as lgb
        models["LightGBM"] = lambda seed: lgb.LGBMClassifier(
            n_estimators=500,
            learning_rate=0.03,
            num_leaves=31,
            subsample_for_bin=5000,
            colsample_bytree=0.75,
            subsample=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            random_state=seed,
            verbose=-1,
            n_jobs=-1,
        )
    except ImportError:
        pass

    try:
        import xgboost as xgb
        models["XGBoost"] = lambda seed: xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=5,
            colsample_bytree=0.75,
            subsample=0.8,
            reg_alpha=0.5,
            reg_lambda=1.0,
            eval_metric="logloss",
            random_state=seed,
            n_jobs=-1,
        )
    except ImportError:
        pass

    try:
        from catboost import CatBoostClassifier
        models["CatBoost"] = lambda seed: CatBoostClassifier(
            iterations=700,
            learning_rate=0.04,
            depth=6,
            random_seed=seed,
            verbose=0,
        )
    except ImportError:
        pass

    return models


def find_best_threshold(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, float, float]:
    best_thr, best_f1, best_acc = 0.5, 0.0, 0.0
    for thr in np.linspace(0.20, 0.80, 121):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(y_true, preds)
        acc = accuracy_score(y_true, preds)
        # Оптимизируем баланс F1 и Accuracy
        if (f1 + acc) > (best_f1 + best_acc):
            best_thr, best_f1, best_acc = float(thr), float(f1), float(acc)
    return best_thr, best_f1, best_acc


def main():
    train_df = pd.read_csv("dataset/train.csv")
    sealed_df = pd.read_csv("dataset/sealed.csv")

    channel_cols = [c for c in train_df.columns if c.startswith("c") and c[1:].isdigit()]
    y = train_df["vrancea"].values.astype(int)

    print(f"Train: {train_df.shape}, Sealed: {sealed_df.shape}, Positive rate: {y.mean():.4f}")

    X_train = build_features(train_df, channel_cols)
    X_sealed = build_features(sealed_df, channel_cols)
    print(f"Сгенерировано признаков: {X_train.shape[1]}\n")

    models = get_candidate_models()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    oof_preds = {name: np.zeros(len(train_df), dtype=np.float64) for name in models}
    test_preds = {name: np.zeros(len(sealed_df), dtype=np.float64) for name in