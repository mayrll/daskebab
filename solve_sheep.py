import os
import glob
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)

def find_file(filename, roots):
    for root in roots:
        if os.path.exists(root):
            matches = glob.glob(os.path.join(root, "**", filename), recursive=True)
            if matches:
                return matches[0]
    return None

TRAIN_COUNTS_CSV = find_file("train_counts.csv", [BASE_DIR, PARENT_DIR])
DATA_DIR = os.path.dirname(TRAIN_COUNTS_CSV)
TRAIN_POINTS_CSV = find_file("train_points.csv", [DATA_DIR, BASE_DIR, PARENT_DIR])
SURVEY_PLOTS_CSV = find_file("survey_plots.csv", [DATA_DIR, BASE_DIR, PARENT_DIR])

train_imgs = glob.glob(os.path.join(DATA_DIR, "**", "train_000.jpg"), recursive=True)
survey_imgs = glob.glob(os.path.join(DATA_DIR, "**", "survey_0001.jpg"), recursive=True)

TRAIN_DIR = os.path.dirname(train_imgs[0]) if train_imgs else os.path.join(DATA_DIR, "train")
SURVEY_DIR = os.path.dirname(survey_imgs[0]) if survey_imgs else os.path.join(DATA_DIR, "survey")

OUT_COUNTS_CSV = os.path.join(BASE_DIR, "counts.csv")
OUT_BANNER_PNG = os.path.join(BASE_DIR, "flag_banner.png")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
counts_df = pd.read_csv(TRAIN_COUNTS_CSV)
points_df = pd.read_csv(TRAIN_POINTS_CSV)

points_map = {}
for img_name, group in points_df.groupby("image"):
    points_map[os.path.basename(str(img_name))] = group[["x", "y"]].values

# Высокое разрешение, чтобы не терять мелких овец
IMG_H, IMG_W = 384, 512
DOWNSCALE = 4
OUT_H, OUT_W = IMG_H // DOWNSCALE, IMG_W // DOWNSCALE
SIGMA = 2.0

def make_density_map(pts, orig_w, orig_h):
    dmap = np.zeros((OUT_H, OUT_W), dtype=np.float32)
    if len(pts) == 0:
        return dmap
    yy, xx = np.mgrid[0:OUT_H, 0:OUT_W]
    for x, y in pts:
        sx = x * (OUT_W / orig_w)
        sy = y * (OUT_H / orig_h)
        g = np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * SIGMA ** 2))
        s = g.sum()
        if s > 0:
            dmap += g / s
    return dmap

class SheepDensityDataset(Dataset):
    def __init__(self, df, img_dir, augment=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.augment = augment
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = os.path.basename(str(row["image"]))
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        orig_w, orig_h = img.size

        pts = points_map.get(fname, np.zeros((0, 2)))
        dmap = make_density_map(pts, orig_w, orig_h)

        img_t = transforms.ToTensor()(img.resize((IMG_W, IMG_H), Image.BILINEAR))
        dmap_t = torch.from_numpy(dmap).unsqueeze(0)

        if self.augment:
            if np.random.rand() > 0.5:
                img_t = torch.flip(img_t, dims=[2])
                dmap_t = torch.flip(dmap_t, dims=[2])
            if np.random.rand() > 0.5:
                img_t = torch.flip(img_t, dims=[1])
                dmap_t = torch.flip(dmap_t, dims=[1])

        return self.norm(img_t), dmap_t, float(row["count"])

class DensityNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

        self.up3 = nn.Conv2d(256, 128, 1)
        self.up2 = nn.Conv2d(128, 64, 1)
        self.conv_out = nn.Conv2d(64, 1, 1)
        # Инициализируем bias так, чтобы выход не уходил в ноль
        nn.init.constant_(self.conv_out.bias, -4.0)
        self.act = nn.Softplus()

    def forward(self, x):
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)

        u3 = F.interpolate(self.up3(f3), size=f2.shape[-2:], mode="bilinear", align_corners=False) + f2
        u2 = F.interpolate(self.up2(u3), size=f1.shape[-2:], mode="bilinear", align_corners=False) + f1
        return self.act(self.conv_out(u2))

perm = np.random.RandomState(42).permutation(len(counts_df))
split = int(0.85 * len(counts_df))
train_loader = DataLoader(SheepDensityDataset(counts_df.iloc[perm[:split]], TRAIN_DIR, augment=True), batch_size=8, shuffle=True)
val_loader = DataLoader(SheepDensityDataset(counts_df.iloc[perm[split:]], TRAIN_DIR, augment=False), batch_size=8, shuffle=False)

model = DensityNet().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=18)

best_mae = 1e9
best_weights = None

for epoch in range(1, 19):
    model.train()
    for imgs, dmaps, true_cnts in train_loader:
        imgs, dmaps, true_cnts = imgs.to(device), dmaps.to(device), true_cnts.to(device)
        preds = model(imgs)
        loss_map = F.mse_loss(preds * 50.0, dmaps * 50.0)
        loss_cnt = F.smooth_l1_loss(preds.sum(dim=(1, 2, 3)), true_cnts, beta=1.0)
        loss = loss_map + 0.05 * loss_cnt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()

    model.eval()
    errs = []
    with torch.no_grad():
        for imgs, _, true_cnts in val_loader:
            pred_cnts = model(imgs.to(device)).sum(dim=(1, 2, 3)).cpu().numpy()
            errs.extend(np.abs(pred_cnts - true_cnts.numpy()))
    mae = np.mean(errs)
    if mae < best_mae:
        best_mae = mae
        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    print(f"Epoch {epoch:02d}/18 | Val MAE: {mae:.2f} (best: {best_mae:.2f})")

model.load_state_dict(best_weights)
model.eval()

survey_df = pd.read_csv(SURVEY_PLOTS_CSV)
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

pred_counts = []
with torch.no_grad():
    for _, row in survey_df.iterrows():
        fname = os.path.basename(str(row["image"]))
        img = Image.open(os.path.join(SURVEY_DIR, fname)).convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
        t1 = norm(transforms.ToTensor()(img)).unsqueeze(0).to(device)
        t2 = torch.flip(t1, dims=[3])
        c = 0.5 * (model(t1).sum().item() + model(t2).sum().item())
        pred_counts.append(max(0.0, c))

survey_df["count_raw"] = pred_counts
survey_df["count"] = [int(round(c)) for c in pred_counts]
survey_df[["image", "count"]].to_csv(OUT_COUNTS_CSV, index=False)

# Строим точную матрицу 9x37
coords = survey_df[["x_m", "y_m"]].values
centered = coords - coords.mean(axis=0)
_, _, vt = np.linalg.svd(centered, full_matrices=False)
rot = centered @ vt.T
if rot[0, 0] > rot[-1, 0]:
    rot[:, 0] = -rot[:, 0]

col_idx = np.round((rot[:, 0] - rot[:, 0].min()) / 50.0).astype(int)
row_idx = np.round((rot[:, 1].max() - rot[:, 1]) / 50.0).astype(int)

grid = np.zeros((9, 37))
for i in range(len(survey_df)):
    grid[row_idx[i], col_idx[i]] = pred_counts[i]

for q in [35, 50, 65]:
    th = np.percentile(grid, q)
    print(f"\n=== Quantile {q}% (count >= {th:.1f}) ===")
    for r in range(9):
        print("".join("█" if grid[r, c] >= th else "·" for c in range(37)))

fig, axes = plt.subplots(3, 1, figsize=(16, 11))
axes[0].imshow(grid, cmap="viridis", aspect="equal")
axes[0].set_title("Exact 9x37 Grid Heatmap")

for idx, q in enumerate([45, 60], start=1):
    th = np.percentile(grid, q)
    axes[idx].imshow(grid >= th, cmap="gray_r", aspect="equal")
    axes[idx].set_title(f"Binary Mask (count >= {th:.1f})")

plt.tight_layout()
plt.savefig(OUT_BANNER_PNG, dpi=150)
print(f"\nСохранено в {OUT_BANNER_PNG} и {OUT_COUNTS_CSV}")