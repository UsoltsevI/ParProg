#!/usr/bin/env python3
"""
Сводные графики по всем пунктам лабораторной + финальная версия.
Исправлено: автоопределение колонки 'threads' в CSV.
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------- 1. Чтение и агрегация ----------------

def load_csv(path: Path, group_by: str = None, default_threads: str = None):
    """CSV -> {key: {n: [времена]}}.

    group_by:        имя колонки для группировки (например, 'threads').
    default_threads: если колонки нет, подставить это значение.
    """
    if not path.exists():
        print(f"  [skip] {path} не найден")
        return {}

    # сначала читаем заголовок, чтобы узнать, есть ли нужная колонка
    with path.open() as f:
        header = next(csv.reader(f), [])

    has_key = group_by is not None and group_by in header

    grouped = defaultdict(lambda: defaultdict(list))
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            if group_by is None:
                key = "all"
            elif has_key:
                key = row[group_by]
            elif default_threads is not None:
                key = default_threads
            else:
                raise SystemExit(
                    f"{path}: нет колонки '{group_by}'. "
                    f"Доступные колонки: {header}"
                )
            grouped[key][int(row["n"])].append(float(row["time_sec"]))

    if group_by is not None and not has_key:
        print(f"  [warn] {path}: нет колонки '{group_by}', "
              f"использую threads={default_threads}")

    return grouped


def mean_std(by_n):
    sizes = sorted(by_n)
    means = np.array([statistics.mean(by_n[n]) for n in sizes])
    stds  = np.array([statistics.pstdev(by_n[n]) for n in sizes])
    return np.array(sizes), means, stds


# ---------------- 2. Источники ----------------

# (label, file, group_by, default_threads)
SOURCES = [
    ("п.1 naive",     "measurements_naive.csv",     None, None),
    ("п.2 transpose", "measurements_transpose.csv", None, None),
    ("п.3 blocked",   "measurements_blocked.csv",   None, None),
    ("п.4 Strassen",  "measurements_strassen.csv",  None, None),
    ("п.6 AVX",       "measurements_avx.csv",       None, None),
]

# (prefix, file, default_threads)
MULTITHREADED = [
    ("п.5 OpenMP", "measurements_openmp.csv", None),
    ("Final",      "measurements_final.csv",  "8"),  # <- фикс
]

series = []   # (label, sizes, mean, std)

for label, fname, key, def_threads in SOURCES:
    data = load_csv(Path(fname), key, def_threads)
    if not data:
        continue
    sizes, mean, std = mean_std(data["all"])
    series.append((label, sizes, mean, std))
    n_meas = sum(len(v) for v in data["all"].values())
    print(f"  [ok] {fname}: {len(sizes)} размеров, {n_meas} замеров")

for prefix, fname, def_threads in MULTITHREADED:
    data = load_csv(Path(fname), "threads", def_threads)
    if not data:
        continue
    for threads in sorted(data, key=int):
        sizes, mean, std = mean_std(data[threads])
        label = f"{prefix} ({threads} потоков)"
        series.append((label, sizes, mean, std))
        n_meas = sum(len(v) for v in data[threads].values())
        print(f"  [ok] {fname} (threads={threads}): "
              f"{len(sizes)} размеров, {n_meas} замеров")

if not series:
    raise SystemExit("Нет данных — проверьте имена CSV-файлов")

naive = next((s for s in series if s[0].startswith("п.1")), None)
if naive is None:
    print("Внимание: naive не найден — график ускорения не построить")

# проверка, что финальная появилась
if not any(s[0].startswith("Final") for s in series):
    print("Внимание: финальная версия не загрузилась — проверьте measurements_final.csv")


# ---------------- 3. График 1: линейный ----------------

MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h", "<", ">", "p"]

fig, ax = plt.subplots(figsize=(11, 6))
for i, (label, sizes, mean, std) in enumerate(series):
    ax.errorbar(sizes, mean, yerr=std,
                fmt=MARKERS[i % len(MARKERS)] + "-",
                capsize=3, label=label, linewidth=1.5, markersize=6)

ax.set_xlabel("размер матрицы n")
ax.set_ylabel("время, с")
ax.set_title("Время умножения матриц: все версии")
ax.grid(True, which="both", ls=":")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig("plot_all_linear.png", dpi=150)
plt.close(fig)
print("\n-> plot_all_linear.png")


# ---------------- 4. График 2: лог-лог + O(n^3) ----------------

fig, ax = plt.subplots(figsize=(11, 6))
for i, (label, sizes, mean, std) in enumerate(series):
    ax.errorbar(sizes, mean, yerr=std,
                fmt=MARKERS[i % len(MARKERS)] + "-",
                capsize=3, label=label, linewidth=1.5, markersize=6)

if naive is not None:
    _, sizes0, mean0, _ = naive
    c3 = mean0[-1] / (sizes0[-1] ** 3)
    n_ref = np.linspace(min(sizes0), max(sizes0), 100)
    ax.plot(n_ref, c3 * n_ref**3, "k--", linewidth=1,
            label=r"эталон $O(n^3)$")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("размер матрицы n (log)")
ax.set_ylabel("время, с (log)")
ax.set_title("Лог-лог: все версии и эталон $O(n^3)$")
ax.grid(True, which="both", ls=":")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig("plot_all_loglog.png", dpi=150)
plt.close(fig)
print("-> plot_all_loglog.png")


# ---------------- 5. График 3: ускорение относительно naive ----------------

if naive is not None:
    _, sizes_n, mean_n, _ = naive
    naive_map = dict(zip(sizes_n, mean_n))

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (label, sizes, mean, std) in enumerate(series):
        if label.startswith("п.1"):
            continue
        pairs = [(n, naive_map[n] / t) for n, t in zip(sizes, mean)
                 if n in naive_map]
        if not pairs:
            continue
        sizes_use = np.array([p[0] for p in pairs])
        speedup   = np.array([p[1] for p in pairs])
        ax.plot(sizes_use, speedup,
                MARKERS[i % len(MARKERS)] + "-",
                label=label, linewidth=1.5, markersize=6)

    ax.axhline(1.0, color="gray", ls="--", linewidth=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("размер матрицы n (log)")
    ax.set_ylabel("ускорение (naive / версия)")
    ax.set_title("Ускорение относительно наивного алгоритма")
    ax.grid(True, which="both", ls=":")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig("plot_all_speedup.png", dpi=150)
    plt.close(fig)
    print("-> plot_all_speedup.png")


# ---------------- 6. Сводный CSV ----------------

with open("summary_all.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["version", "n", "mean_sec", "std_sec"])
    for label, sizes, mean, std in series:
        for n, m, s in zip(sizes, mean, std):
            w.writerow([label, int(n), f"{m:.6f}", f"{s:.6f}"])
print("-> summary_all.csv")