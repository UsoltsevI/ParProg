#!/usr/bin/env python3
"""
Анализ замеров наивного умножения матриц:
  - агрегирует 5 запусков в среднее по каждому n;
  - считает t(n)/t(n/2) и наклон лог-лог графика (эмпирический порядок O(n^p));
  - строит графики: линейный и лог-лог + сравнение с O(n^3).
"""
import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV_IN  = Path("naive_measure.csv")
CSV_OUT = Path("summary.csv")
PNG_LIN = Path("plot_linear.png")
PNG_LOG = Path("plot_loglog.png")

# ---------- 1. Чтение и агрегация ----------
by_n = defaultdict(list)          # n -> [времена]
with CSV_IN.open() as f:
    for row in csv.DictReader(f):
        n = int(row["n"])
        t = float(row["time_sec"])
        by_n[n].append(t)

sizes  = sorted(by_n)
mean_t = np.array([statistics.mean(by_n[n]) for n in sizes])
std_t  = np.array([statistics.pstdev(by_n[n]) for n in sizes])  # population std
min_t  = np.array([min(by_n[n]) for n in sizes])
max_t  = np.array([max(by_n[n]) for n in sizes])
runs   = np.array([len(by_n[n]) for n in sizes])

# ---------- 2. t(n)/t(n/2) и наклон лог-лог ----------
ratio = np.full_like(mean_t, np.nan)
for i in range(1, len(sizes)):
    ratio[i] = mean_t[i] / mean_t[i-1]

# Эмпирический порядок p из наклона прямой log(t) = p*log(n) + b
log_n = np.log(sizes)
log_t = np.log(mean_t)
p_slope, b = np.polyfit(log_n, log_t, 1)

# ---------- 3. Сохранение summary.csv ----------
with CSV_OUT.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["n", "runs", "mean_sec", "std_sec", "min_sec",
                "max_sec", "t(n)/t(n/2)"])
    for i, n in enumerate(sizes):
        r = "" if np.isnan(ratio[i]) else f"{ratio[i]:.3f}"
        w.writerow([n, runs[i], f"{mean_t[i]:.6f}", f"{std_t[i]:.6f}",
                    f"{min_t[i]:.6f}", f"{max_t[i]:.6f}", r])

print(f"Средние по {len(sizes)} размерам сохранены в {CSV_OUT}")
print(f"Эмпирический наклон log-log: p = {p_slope:.3f} "
      f"(теория для наивного алгоритма ≈ 3)")

# ---------- 4. Таблица в stdout ----------
print(f"\n{'n':>6} {'runs':>5} {'mean, s':>10} {'std, s':>10} "
      f"{'t(n)/t(n/2)':>12}")
print("-" * 50)
for i, n in enumerate(sizes):
    r = "  -" if np.isnan(ratio[i]) else f"{ratio[i]:.2f}"
    print(f"{n:>6} {runs[i]:>5} {mean_t[i]:>10.6f} "
          f"{std_t[i]:>10.6f} {r:>12}")

# ---------- 5. График 1: линейный масштаб ----------
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(sizes, mean_t, yerr=std_t, fmt="o-", capsize=5,
            label="среднее ± σ (по 5 запускам)")

# Подгонка O(n^3): t ≈ c * n^3
c3 = float(np.exp(b))
n_ref = np.linspace(min(sizes), max(sizes), 100)
ax.plot(n_ref, c3 * n_ref**3, "--", color="gray", label=r"$O(n^3)$ (подгонка)")

ax.set_xlabel("размер матрицы n")
ax.set_ylabel("время, с")
ax.set_title("Наивное умножение матриц: время от размера")
ax.grid(True, which="both", ls=":")
ax.legend()
fig.tight_layout()
fig.savefig(PNG_LIN, dpi=150)
plt.close(fig)

# ---------- 6. График 2: лог-лог ----------
fig, ax = plt.subplots(figsize=(8, 5))
ax.errorbar(sizes, mean_t, yerr=std_t, fmt="o-", capsize=5,
            label="измерения (среднее ± σ)")
ax.plot(n_ref, c3 * n_ref**3, "--", color="gray",
        label=f"$O(n^3)$ (наклон {p_slope:.2f})")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("размер матрицы n (log)")
ax.set_ylabel("время, с (log)")
ax.set_title("Лог-лог: наклон ≈ 3 подтверждает $O(n^3)$")
ax.grid(True, which="both", ls=":")
ax.legend()
fig.tight_layout()
fig.savefig(PNG_LOG, dpi=150)
plt.close(fig)

print(f"\nГрафики сохранены: {PNG_LIN}, {PNG_LOG}")