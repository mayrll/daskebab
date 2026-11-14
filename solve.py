"""Explore Point par point: python solve.py paris_listings.csv --out results."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

FEATURES = ["price", "review_scores_rating", "availability_365"]
LOCATION = ["longitude", "latitude"]


def plot(frame, x, y, path, labels=None, bounds=None):
    width, height = 1000, 650
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((24, 12), f"{x} / {y}  |  {len(frame)} points", fill="black")
    points = frame[[x, y]].to_numpy(dtype=float)
    low, high = bounds if bounds is not None else (points.min(0), points.max(0))
    pixels = (points - low) / np.maximum(high - low, 1e-12)
    palette = ("#165aaa", "#b43933", "#187555", "#843faa", "#b36710")
    colors = ["#6c7076"] * len(frame)
    if labels is not None:
        colors = [palette[(label - 1) % len(palette)] for label in labels]
    radius = 1 if len(frame) > 3000 else 2
    for (u, v), color in zip(pixels, colors):
        xx = 24 + u * (width - 48)
        yy = height - 24 - v * (height - 64)
        draw.ellipse((xx-radius, yy-radius, xx+radius, yy+radius), fill=color)
    image.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--clusters", type=Path, help=(
        "Optional CSV of confirmed bounding boxes with columns "
        "label,lon_min,lon_max,lat_min,lat_max. Applied to the selected grey points."
    ))
    parser.add_argument("--method", choices=("minmax", "percentile"), default="minmax")
    parser.add_argument("--fraction", type=float, default=0.01)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1:
        parser.error("--fraction must be greater than 0 and at most 1")

    data = pd.read_csv(args.csv, dtype={"listing_id": str})
    for field in FEATURES + LOCATION:
        data[field] = pd.to_numeric(data[field], errors="coerce")
    data = data.dropna(subset=FEATURES + LOCATION).copy()
    if data.empty:
        parser.error("No rows with complete coordinates and features")
    args.out.mkdir(parents=True, exist_ok=True)

    values = data[FEATURES]
    channels = {
        "minmax": (values - values.min()) / (values.max() - values.min()).replace(0, 1),
        "percentile": values.rank(pct=True),
    }
    coordinates = data[LOCATION].to_numpy()
    bounds = coordinates.min(0), coordinates.max(0)
    plot(data, *LOCATION, args.out / "all_listings.png", bounds=bounds)

    for method, rgb in channels.items():
        data[method + "_grey_distance"] = rgb.max(axis=1) - rgb.min(axis=1)
        for fraction in sorted({0.01, 0.03, 0.09, args.fraction}):
            distance = data[method + "_grey_distance"]
            candidates = data[distance <= distance.quantile(fraction)]
            plot(candidates, *LOCATION,
                 args.out / f"{method}_{fraction:g}.png", bounds=bounds)

    column = args.method + "_grey_distance"
    selected = data[data[column] <= data[column].quantile(args.fraction)].copy()
    selected[["listing_id", *LOCATION, *FEATURES, column]].to_csv(
        args.out / "grey_candidates.csv", index=False
    )
    if args.clusters is None:
        return

    boxes = pd.read_csv(args.clusters).sort_values("label")
    if boxes.empty or boxes.label.tolist() != list(range(1, len(boxes) + 1)):
        parser.error("Cluster labels must be consecutive integers starting at 1")
    selected["label"] = 0
    for box in boxes.itertuples(index=False):
        inside = (
            selected.longitude.between(box.lon_min, box.lon_max)
            & selected.latitude.between(box.lat_min, box.lat_max)
        )
        if not inside.any() or selected.loc[inside, "label"].ne(0).any():
            parser.error(f"Cluster {box.label} is empty or overlaps another cluster")
        selected.loc[inside, "label"] = box.label
    letters = selected[selected.label > 0]
    letters[["listing_id", "label", *LOCATION]].to_csv(args.out / "labels.csv", index=False)
    plot(letters, *LOCATION, args.out / "figure.png", labels=letters.label)


if __name__ == "__main__":
    main()
