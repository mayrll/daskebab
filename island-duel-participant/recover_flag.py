"""Extract the embedded chart and reproduce its Vigenere decoding."""
import json
import re
from pathlib import Path

root = Path(__file__).resolve().parent
binary = (root / "samoa-linux-x86_64").read_bytes()
match = next(m for m in re.finditer(rb"[ -~]{8,}", binary)
             if m.group().startswith(b'{"prefix":"SIGMOID_"'))
chart = json.loads(match.group().decode("ascii"))
stars = chart["stars"]
positions = [(100, 180), (40, 88), (96, 28), (160, 76), (132, 130)]
anchors = sorted(
    [star for star in stars if star["size"] >= 7],
    key=lambda star: positions.index((star["x"], star["y"])),
)


def distance(first, second):
    return (first["x"] - second["x"]) ** 2 + (first["y"] - second["y"]) ** 2


ordered = []
for anchor in anchors:
    companions = [
        star for star in stars
        if star["size"] < 7
        and min(anchors, key=lambda candidate: distance(star, candidate)) == anchor
    ]
    ordered.extend([anchor] + sorted(companions, key=lambda star: distance(star, anchor)))
cipher = "".join(star["text"] for star in ordered)
key = "FAAVAEILEATUASAMOA"
plaintext = "".join(chr((ord(letter) - ord(key[index % len(key)])) % 26 + 65)
                    for index, letter in enumerate(cipher))
flag = chart["prefix"] + plaintext
print("Embedded chart offset", hex(match.start()))
print("Ciphertext", cipher)
print("Key", key)
print(flag)
(root / "flag.txt").write_text(flag + "\n", encoding="ascii")
