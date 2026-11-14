import csv
import io

# Таблица обратного маппинга символов Windows-1252 (0x80–0x9F) в исходные байты
CP1252_TO_BYTE = {
    0x20AC: 0x80, 0x201A: 0x82, 0x0192: 0x83, 0x201E: 0x84,
    0x2026: 0x85, 0x2020: 0x86, 0x2021: 0x87, 0x02C6: 0x88,
    0x2030: 0x89, 0x0160: 0x8A, 0x2039: 0x8B, 0x0152: 0x8C,
    0x017D: 0x8E, 0x2018: 0x91, 0x2019: 0x92, 0x201C: 0x93,
    0x201D: 0x94, 0x2022: 0x95, 0x2013: 0x96, 0x2014: 0x97,
    0x02DC: 0x98, 0x2122: 0x99, 0x0161: 0x9A, 0x203A: 0x9B,
    0x0153: 0x9C, 0x017E: 0x9E, 0x0178: 0x9F,
}


def fix_mojibake(text: str) -> str:
    """Восстанавливает UTF-8 строку, ошибочно декодированную как Latin-1 / CP1252."""
    s = text.strip()
    for _ in range(2):  # на случай двойного перекодирования
        raw_bytes = bytearray()
        convertible = True
        for ch in s:
            code = ord(ch)
            if code <= 0xFF:
                raw_bytes.append(code)
            elif code in CP1252_TO_BYTE:
                raw_bytes.append(CP1252_TO_BYTE[code])
            else:
                convertible = False
                break

        if not convertible:
            break

        try:
            decoded = raw_bytes.decode("utf-8")
        except UnicodeDecodeError:
            break

        if decoded == s:
            break
        s = decoded

    return s


def parse_temperature_c(temp_raw: str, unit: str) -> str:
    val = float(temp_raw.strip().replace(",", "."))
    if unit.strip().upper().startswith("F"):
        val = (val - 32.0) * 5.0 / 9.0
    formatted = f"{val:.1f}"
    return "0.0" if formatted == "-0.0" else formatted


def main() -> None:
    with open("test-input.csv", "r", encoding="utf-8-sig") as f:
        # Убираем преамбулу из комментариев (#) и пустых строк
        data_lines = [
            line
            for line in f
            if line.strip() and not line.lstrip().startswith("#")
        ]

    if not data_lines:
        with open("test-output.csv", "w", encoding="utf-8", newline="\n") as out:
            out.write("reading_id,station_name,temperature_c,reading_date\n")
        return

    reader = csv.DictReader(io.StringIO("".join(data_lines)), delimiter=";")

    records = {}
    for row in reader:
        if not row or not row.get("reading_id"):
            continue
        rid = int(row["reading_id"].strip())
        if rid in records:
            continue

        station = fix_mojibake(row["station_name"])
        temp_c = parse_temperature_c(row["temperature_raw"], row["unit"])
        rdate = row["reading_date"].strip()

        records[rid] = (rid, station, temp_c, rdate)

    with open("test-output.csv", "w", encoding="utf-8", newline="\n") as out:
        out.write("reading_id,station_name,temperature_c,reading_date\n")
        for rid in sorted(records):
            _, station, temp_c, rdate = records[rid]
            out.write(f"{rid},{station},{temp_c},{rdate}\n")


if __name__ == "__main__":
    main()