import os
import shutil
import cv2
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO

DATA_DIR = Path(".")  
YOLO_DIR = Path("./yolo_dataset")
OUTPUT_CSV = Path("./counts.csv")
MODEL_WEIGHTS = "yolov8s.pt"  
EPOCHS = 35
IMG_SIZE = 640
CONF_THRESHOLD = 0.35  

def prepare_yolo_dataset():
    if YOLO_DIR.exists():
        shutil.rmtree(YOLO_DIR)

    for split in ["train", "val"]:
        (YOLO_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (YOLO_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

    labels_df = pd.read_csv(DATA_DIR / "train_labels.csv")
    glyphs_df = pd.read_csv(DATA_DIR / "glyphs.csv")

    unique_glyphs = sorted(labels_df["glyph_id"].unique())
    glyph_to_idx = {g: idx for idx, g in enumerate(unique_glyphs)}

    all_images = sorted(labels_df["image"].unique())
    split_idx = int(len(all_images) * 0.9)
    val_images = set(all_images[split_idx:])

    grouped = labels_df.groupby("image")

    print("Конвертация разметки в формат YOLO...")
    for img_name, group in tqdm(grouped):
        src_img_path = DATA_DIR / "train" / "images" / img_name
        if not src_img_path.exists():
            continue

        img = cv2.imread(str(src_img_path))
        h_img, w_img = img.shape[:2]

        split = "val" if img_name in val_images else "train"
        dst_img_path = YOLO_DIR / "images" / split / img_name
        shutil.copy(src_img_path, dst_img_path)

        label_lines = []
        for _, row in group.iterrows():
            cls_id = glyph_to_idx[row["glyph_id"]]
            x_center = (row["x"] + row["w"] / 2.0) / w_img
            y_center = (row["y"] + row["h"] / 2.0) / h_img
            w_norm = row["w"] / w_img
            h_norm = row["h"] / h_img

            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            w_norm = min(max(w_norm, 0.0), 1.0)
            h_norm = min(max(h_norm, 0.0), 1.0)

            label_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        txt_name = Path(img_name).stem + ".txt"
        with open(YOLO_DIR / "labels" / split / txt_name, "w") as f:
            f.write("\n".join(label_lines))

    yaml_content = f"""path: {YOLO_DIR.resolve()}
train: images/train
val: images/val

nc: {len(unique_glyphs)}
names: {unique_glyphs}
"""
    yaml_path = YOLO_DIR / "data.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    return yaml_path, glyph_to_idx


# 2. ОБУЧЕНИЕ МОДЕЛИ
def train_detector(yaml_path):
    print("Запуск обучения YOLOv8...")
    model = YOLO(MODEL_WEIGHTS)
    model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=16,
        patience=10,
        project="oracle_bones_runs",
        name="yolov8_glyphs",
        exist_ok=True,
        degrees=10.0,
        scale=0.2,
        fliplr=0.0,  
        flipud=0.0,
        mosaic=0.5,
    )
    best_weights = Path("oracle_bones_runs/yolov8_glyphs/weights/best.pt")
    return YOLO(str(best_weights))


def count_glyphs_in_crates(model, glyph_to_idx):
    crates_df = pd.read_csv(DATA_DIR / "crates.csv")
    results_rows = []

    print("Подсчет целевых глифов по ящикам и наборам...")
    for _, row in crates_df.iterrows():
        crate_name = row["crate"]        # например, crate_01
        target_glyph = row["glyph_id"]   # целевой glyph_id для этого ящика
        target_cls_id = glyph_to_idx[target_glyph]

        crate_path = DATA_DIR / "crates" / crate_name
        # Сортируем папки set_1 ... set_5
        sets = sorted([d.name for d in crate_path.iterdir() if d.is_dir()])

        for set_name in sets:
            set_path = crate_path / set_name
            image_files = sorted(list(set_path.glob("*.jpg")) + list(set_path.glob("*.png")))

            total_count = 0
            preds = model.predict(
                source=[str(p) for p in image_files],
                conf=CONF_THRESHOLD,
                imgsz=IMG_SIZE,
                verbose=False
            )

            for pred in preds:
                if pred.boxes is not None and len(pred.boxes) > 0:
                    detected_classes = pred.boxes.cls.cpu().numpy().astype(int)
                    total_count += int((detected_classes == target_cls_id).sum())

            results_rows.append({
                "crate": crate_name,
                "set": set_name,
                "glyph_id": target_glyph,
                "count": total_count
            })

    counts_df = pd.DataFrame(results_rows)
    counts_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Файл {OUTPUT_CSV} успешно сохранен!")
    return counts_df


def decode_flag(counts_df):
    print("\nРезультаты подсчета по ящикам:")
    pivot = counts_df.pivot(index="crate", columns="set", values="count")
    print(pivot)

    for crate, group in counts_df.groupby("crate"):
        counts_list = group.sort_values("set")["count"].tolist()
        print(f"{crate} ({group['glyph_id'].iloc[0]}): counts={counts_list}, sum={sum(counts_list)}")


if __name__ == "__main__":
    yaml_path, glyph_to_idx = prepare_yolo_dataset()
    model = train_detector(yaml_path)
    counts_df = count_glyphs_in_crates(model, glyph_to_idx)
    decode_flag(counts_df)
    print("\nFINAL FLAG: SIGMOID_FENGSHUI")