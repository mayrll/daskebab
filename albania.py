import math
import numpy as np


class Detector:
    SENSORS = (
        "temperature_c",
        "humidity_pct",
        "vibration_g",
        "acoustic_db",
        "air_pressure_kpa",
        "co2_ppm",
    )

    def __init__(self):
        self.zones = {}
        self.long_win = 240
        self.short_win = 10
        self.coll_win = 5
        self.warmup = 15

        # Пороги срабатывания (в эквиваленте Z-score).
        # Чем выше значение, тем меньше ложных тревог (выше Precision).
        self.point_thresh = 3.7
        self.drift_thresh = 3.6
        self.coll_thresh = 3.5

    def _init_zone(self, x: np.ndarray) -> dict:
        n_sensors = len(self.SENSORS)
        state = {
            "count": 0,
            "anom_streak": 0,
            "last_vals": x.copy(),
            "long_buf": np.zeros((self.long_win, n_sensors), dtype=np.float64),
            "long_len": 0,
            "long_idx": 0,
            "short_buf": np.zeros((self.short_win, n_sensors), dtype=np.float64),
            "short_len": 0,
            "short_idx": 0,
            "drift_hist": np.zeros((self.long_win, n_sensors), dtype=np.float64),
            "drift_len": 0,
            "drift_idx": 0,
            "coll_hist": np.zeros(self.long_win, dtype=np.float64),
            "coll_len": 0,
            "coll_idx": 0,
        }
        return state

    def _extract_vector(self, reading: dict, last_vals: np.ndarray | None) -> np.ndarray:
        vec = np.empty(len(self.SENSORS), dtype=np.float64)
        for i, key in enumerate(self.SENSORS):
            val = reading.get(key)
            if val is None or not math.isfinite(float(val)):
                vec[i] = last_vals[i] if last_vals is not None else 0.0
            else:
                vec[i] = float(val)
        return vec

    @staticmethod
    def _push_2d(buf: np.ndarray, idx: int, length: int, capacity: int, row: np.ndarray):
        buf[idx] = row
        idx = (idx + 1) % capacity
        length = min(length + 1, capacity)
        return idx, length

    @staticmethod
    def _push_1d(buf: np.ndarray, idx: int, length: int, capacity: int, val: float):
        buf[idx] = val
        idx = (idx + 1) % capacity
        length = min(length + 1, capacity)
        return idx, length

    def update(self, reading: dict) -> float:
        zone = reading.get("bunker_zone", "default")

        if zone not in self.zones:
            x = self._extract_vector(reading, None)
            st = self._init_zone(x)
            self.zones[zone] = st
        else:
            st = self.zones[zone]
            x = self._extract_vector(reading, st["last_vals"])
            st["last_vals"] = x

        # Всегда обновляем короткое окно текущих значений
        st["short_idx"], st["short_len"] = self._push_2d(
            st["short_buf"], st["short_idx"], st["short_len"], self.short_win, x
        )
        st["count"] += 1

        # Период прогрева: собираем первичную статистику по зоне
        if st["count"] <= self.warmup:
            st["long_idx"], st["long_len"] = self._push_2d(
                st["long_buf"], st["long_idx"], st["long_len"], self.long_win, x
            )
            if st["short_len"] >= 3:
                s_mean = np.mean(st["short_buf"][: st["short_len"]], axis=0)
                st["drift_idx"], st["drift_len"] = self._push_2d(
                    st["drift_hist"], st["drift_idx"], st["drift_len"], self.long_win, s_mean
                )
            return 0.0

        # 1. Базовая статистика по длинному окну (медиана + MAD устойчивы к выбросам)
        base = st["long_buf"][: st["long_len"]]
        med = np.median(base, axis=0)
        mad = np.median(np.abs(base - med), axis=0) * 1.4826
        std = np.std(base, axis=0)
        scale = np.where(mad > 1e-9, mad, std)
        scale = np.maximum(scale, np.maximum(np.abs(med) * 0.005, 1e-4))

        # 2. Детектор точечных аномалий (резкие одиночные скачки)
        z_curr = (x - med) / scale
        point_z = float(np.max(np.abs(z_curr)))
        point_ratio = point_z / self.point_thresh

        # 3. Детектор медленного дрейфа (сравнение короткого среднего с историей)
        short_mean = np.mean(st["short_buf"][: st["short_len"]], axis=0)
        if st["drift_len"] >= 15:
            d_base = st["drift_hist"][: st["drift_len"]]
            d_med = np.median(d_base, axis=0)
            d_mad = np.median(np.abs(d_base - d_med), axis=0) * 1.4826
            d_scale = np.maximum(d_mad, scale * 0.45)
            drift_z = float(np.max(np.abs(short_mean - d_med) / d_scale))
        else:
            drift_z = float(np.max(np.abs(short_mean - med) / (scale * 0.65)))
        drift_ratio = drift_z / self.drift_thresh

        # 4. Детектор коллективных аномалий (умеренные сдвиги сразу по нескольким датчикам)
        k = min(st["short_len"], self.coll_win)
        # Берем k последних элементов из кольцевого буфера
        recent_indices = [(st["short_idx"] - 1 - j) % self.short_win for j in range(k)]
        recent_k = st["short_buf"][recent_indices]
        z_recent = (recent_k - med) / scale
        m_k = np.mean(z_recent, axis=0)
        coll_energy = float(np.sqrt(np.sum(m_k ** 2)))

        if st["coll_len"] >= 15:
            c_base = st["coll_hist"][: st["coll_len"]]
            c_med = float(np.median(c_base))
            c_mad = max(float(np.median(np.abs(c_base - c_med))) * 1.4826, 0.35)
            coll_z = (coll_energy - c_med) / c_mad
        else:
            coll_z = (coll_energy - 1.2) / 0.55
        coll_ratio = coll_z / self.coll_thresh

        # Объединяем сигналы и переводим в вероятность [0.0, 1.0] с порогом 0.50 при ratio = 1.0
        max_ratio = max(point_ratio, drift_ratio, coll_ratio)
        logit = max(-20.0, min(20.0, 4.5 * (max_ratio - 1.0)))
        score = 1.0 / (1.0 + math.exp(-logit))

        # Обновляем эталонную базу только на нормальных точках (чтобы выбросы не портили статистику).
        # Если сдвиг держится дольше 45 шагов подряд, плавно адаптируемся к новому режиму.
        if score < 0.50:
            st["anom_streak"] = 0
            should_update_base = True
        else:
            st["anom_streak"] += 1
            should_update_base = st["anom_streak"] > 45 and (st["anom_streak"] % 3 == 0)

        if should_update_base:
            st["long_idx"], st["long_len"] = self._push_2d(
                st["long_buf"], st["long_idx"], st["long_len"], self.long_win, x
            )
            st["drift_idx"], st["drift_len"] = self._push_2d(
                st["drift_hist"], st["drift_idx"], st["drift_len"], self.long_win, short_mean
            )
            st["coll_idx"], st["coll_len"] = self._push_1d(
                st["coll_hist"], st["coll_idx"], st["coll_len"], self.long_win, coll_energy
            )

        if not math.isfinite(score):
            return 0.0
        return float(max(0.0, min(1.0, score)))