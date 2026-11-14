import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier


def load_xy(path: str):
    df = pd.read_csv(path)

    target_candidates = ["released", "release", "avalanche", "label", "target", "y"]
    target_col = next((c for c in target_candidates if c in df.columns), df.columns[-1])

    id_candidates = {"id", "pit_id", "record_id", "sample_id", "index"}
    feature_cols = [
        c for c in df.columns if c != target_col and c.lower() not in id_candidates
    ]

    X = df[feature_cols].copy()
    cat_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    if cat_cols:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        X[cat_cols] = enc.fit_transform(X[cat_cols])

    return X, df[target_col].values, feature_cols


def extract_word(clf: DecisionTreeClassifier, feature_cols: list[str]) -> str:
    tree = clf.tree_
    # Внутренние узлы дерева в порядке обхода (pre-order)
    internal_nodes = [
        i for i in range(tree.node_count) if tree.children_left[i] != -1
    ]
    return "".join(feature_cols[tree.feature[i]][0].upper() for i in internal_nodes).strip()


def main() -> None:
    X, y, feature_cols = load_xy("test-input.csv")

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X, y)

    word = extract_word(clf, feature_cols)
    with open("test-output.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(f"SIGMOID_{word}\n")


if __name__ == "__main__":
    main()