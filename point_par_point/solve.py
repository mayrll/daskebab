"""Recover the six glyphs from the Point par point dataset."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

REFERENCE = (
    "https://data.insideairbnb.com/france/ile-de-france/paris/"
    "2026-09-16/visualisations/listings.csv"
)
KEYS = ["latitude", "longitude", "number_of_reviews", "availability_365"]
FEATURES = ["longitude", "latitude", "price", "availability_365"]
# Orient the six PCA plots for reading. PCA signs are fixed below.
ANGLES = (135, 270, 90, 225, 0, 0)


def draw_letters(features, labels, path):
    image = Image.new("RGB", (1320, 280), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=16)
    for label, angle in enumerate(ANGLES):
        points = features[labels == label]
        centered = points - points.mean(axis=0)
        left, singular, axes = np.linalg.svd(centered, full_matrices=False)
        scores = left[:, :2] * singular[:2]
        # Match orientation to a deterministic loading sign for each component.
        component_signs = np.sign(axes[np.arange(2), np.abs(axes[:2]).argmax(axis=1)])
        scores *= component_signs
        theta = np.radians(angle)
        rotation = [[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]
        plot = scores @ rotation
        if label == 3:
            plot[:, 1] *= -1
        if label == 5:
            plot[:, 0] *= -1
        low, high = np.quantile(plot, [0.01, 0.99], axis=0)
        plot = (plot - (high + low) / 2) / (high - low) * 155
        offset = 220 * label
        draw.text((offset + 21, 18), f"Letter {label + 1}  |  n={len(points)}",
                  font=font, fill="#26384d")
        for x, y in plot:
            x, y = offset + 110 + x, 142 + y
            draw.ellipse((x - 1.25, y - 1.25, x + 1.25, y + 1.25), fill="#14263b")
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--reference", default=REFERENCE)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()

    listings = pd.read_csv(args.csv, dtype={"listing_id": str})
    reference = pd.read_csv(args.reference, usecols=KEYS)
    reference[KEYS[:2]] = reference[KEYS[:2]].round(5)
    joined = listings.merge(reference.drop_duplicates(), on=KEYS, how="left", indicator=True)
    candidates = joined.loc[joined["_merge"].eq("left_only")].copy()
    features = StandardScaler().fit_transform(candidates[FEATURES])
    labels = DBSCAN(eps=0.45, min_samples=5).fit_predict(features)
    means = candidates.assign(cluster=labels).query("cluster >= 0").groupby("cluster").price.mean()
    order = means.sort_values().index
    if len(order) != len(ANGLES):
        raise ValueError(f"Expected six groups, found {len(order)}")
    ordered = pd.Series(labels).map({old: new for new, old in enumerate(order)})
    labels = ordered.fillna(-1).to_numpy(dtype=int)
    joined["label"] = -1
    joined.loc[candidates.index, "label"] = labels

    args.out.mkdir(parents=True, exist_ok=True)
    joined[["listing_id", "label"]].to_csv(args.out / "labels.csv", index=False)
    draw_letters(features, labels, args.out / "letters_separated.png")
    print(joined.label.value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()


