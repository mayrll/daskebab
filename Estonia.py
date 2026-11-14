import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# 1. Загрузка данных
df = pd.read_csv("applications.csv")
exclude_cols = {"submission_id", "date_of_birth", "citizenship", "label"}
feature_cols = [c for c in df.columns if c not in exclude_cols]

X = df[feature_cols].values
y = df["label"].values

print(f"Строк в датасете: {len(df)}")
print(f"Числовые признаки ({len(feature_cols)}): {feature_cols}\n")

# 2. Проверка утечки №1 (обычно обманка): StandardScaler обучен до кросс-валидации
X_leaky_scaled = StandardScaler().fit_transform(X)
preds_baseline = cross_val_predict(
    KNeighborsClassifier(n_neighbors=1),
    X_leaky_scaled,
    y,
    cv=LeaveOneOut(),
)
acc_baseline = accuracy_score(y, preds_baseline)
print(f"[Baseline] LOO + глобальный StandardScaler: {acc_baseline:.4%}")

# 3. Изолированный фикс обманки: переносим StandardScaler внутрь Pipeline
clean_pipe = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1))
preds_fixed_scaler = cross_val_predict(clean_pipe, X, y, cv=LeaveOneOut())
acc_fixed_scaler = accuracy_score(y, preds_fixed_scaler)
print(f"[Decoy 1 Fix] LOO + StandardScaler в Pipeline: {acc_fixed_scaler:.4%}")

# 4. Проверка главного бага: утечка между заявками одного и того же человека
# В 1-NN + LeaveOneOut повторная заявка того же человека становится ближайшим соседом
groups = df["date_of_birth"].astype(str) + "_" + df["citizenship"].astype(str)
print(f"\nУникальных людей (date_of_birth + citizenship): {groups.nunique()} из {len(df)}")

preds_true = cross_val_predict(
    clean_pipe,
    X,
    y,
    cv=LeaveOneGroupOut(),
    groups=groups,
)
true_acc = accuracy_score(y, preds_true)
print(f"[Real Bug Fix] Честный LeaveOneGroupOut: {true_acc:.4%}")
print(f"\nГотовый флаг: SIGMOID_{int(true_acc * 100)}")