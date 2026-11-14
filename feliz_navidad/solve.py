from pathlib import Path

import pandas as pd

folder = Path(__file__).resolve().parent
guide = pd.read_csv(folder / "office_price_guide.csv")
purchases = pd.read_csv(folder / "purchases.csv")

receipts = purchases.merge(guide, on="Product", validate="many_to_one")
unit_price = receipts["Total_paid_eur"] / receipts["Quantity"]
fair = receipts[unit_price.between(receipts["Fair_min_eur"], receipts["Fair_max_eur"])]

board = (
    pd.crosstab(fair["List_no"], fair["Pilot_no"])
    .reindex(index=range(1, 7), columns=range(1, 22), fill_value=0)
    .gt(0)
)

for row in board.to_numpy():
    print("".join("#" if filled else "." for filled in row))

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    from PIL import Image

    image = Image.fromarray((~board.to_numpy()).astype("uint8") * 255)
    image.resize((1050, 300), Image.Resampling.NEAREST).save(folder / "board.png")
else:
    fig, ax = plt.subplots(figsize=(10.5, 3))
    ax.imshow(board, cmap="binary", vmin=0, vmax=1, interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(folder / "board.png", dpi=180, bbox_inches="tight", pad_inches=0.15)
