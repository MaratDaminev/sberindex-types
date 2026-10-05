"""Шаг 8 (проверка): чем вызваны месяцы с массовой сменой типа.

Запуск из корня проекта:  python src/07_jump_diagnostics.py
Использует configs/dynamics.yaml. Результат: reports/jump_diagnostics.md

Проверяются три вещи:
  1. менялись ли в месяц скачка сами признаки сильнее обычного;
  2. были ли сменившие тип МО ближе к границе типов, чем оставшиеся;
  3. сохраняется ли месяц скачка при другом способе разбиения спектрального вложения.
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
CFG = yaml.safe_load(open("configs/dynamics.yaml", encoding="utf-8"))
SEED, K, ALPHA = CFG["seed"], CFG["n_clusters"], CFG["alpha"]


def align(prev, cur):
    C = np.array([[np.sum((prev == p) & (cur == c)) for c in range(K)] for p in range(K)])
    r, c = linear_sum_assignment(-C)
    mp = dict(zip(c, r))
    return np.array([mp[x] for x in cur])


def path(As, assign):
    S, out = None, []
    for t, A in enumerate(As):
        S = A if S is None else (1 - ALPHA) * A + ALPHA * S
        lab = SpectralClustering(K, affinity="precomputed", assign_labels=assign, random_state=SEED).fit_predict(S)
        out.append(lab if t == 0 else align(out[-1], lab))
    return np.column_stack(out)


def main():
    feat = pd.read_parquet(CFG["input"]["features"])
    months = sorted(feat["month"].unique())
    names = [pd.Timestamp(m).strftime("%Y-%m") for m in months]
    ids = np.sort(feat["mo_id"].unique())
    cols = [c for c in feat.columns if c not in ("mo_id", "month")]
    n, T = len(ids), len(months)
    pos = pd.Series(np.arange(n), index=ids)
    Xs = [feat[feat["month"] == m].set_index("mo_id").loc[ids, cols].to_numpy() for m in months]
    E = pd.read_parquet(CFG["input"]["edges"])
    As = []
    for m in months:
        e = E[E["month"] == m]
        i, j, w = pos[e["mo_i"]].to_numpy(), pos[e["mo_j"]].to_numpy(), e["w"].to_numpy()
        As.append(coo_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr())
    L = (pd.read_parquet(Path(CFG["output"]["dir"]) / "final_labels.parquet")
         .pivot(index="mo_id", columns="month", values="type").loc[ids].to_numpy())

    # 1. смены типа и изменение признаков по месяцам
    rows = []
    for t in range(1, T):
        mv = L[:, t] != L[:, t - 1]
        dz = np.abs(Xs[t] - Xs[t - 1]).mean(axis=1)
        rows.append({"месяц": names[t], "сменили тип, %": 100 * mv.mean(),
                     "ARI с прошлым месяцем": adjusted_rand_score(L[:, t - 1], L[:, t]),
                     "среднее изменение признаков, все МО": dz.mean(),
                     "у сменивших тип": dz[mv].mean() if mv.any() else np.nan,
                     "у сохранивших тип": dz[~mv].mean()})
    monthly = pd.DataFrame(rows)
    top = monthly.sort_values("сменили тип, %", ascending=False).head(3)["месяц"].tolist()
    jump = top[0]
    tj = names.index(jump)

    # 2. близость к границе в месяц перед скачком: расстояние до центра нового типа минус до центра старого
    flows = pd.crosstab(pd.Series(L[:, tj - 1], name="из типа"), pd.Series(L[:, tj], name="в тип"))
    X0, l0, l1 = Xs[tj - 1], L[:, tj - 1], L[:, tj]
    cent = {c: X0[l0 == c].mean(axis=0) for c in np.unique(l0)}
    mrows = []
    off = flows.where(~np.eye(len(flows), dtype=bool)).stack().sort_values(ascending=False).head(3)
    for (a, b), cnt in off.items():
        gap = np.linalg.norm(X0 - cent[b], axis=1) - np.linalg.norm(X0 - cent[a], axis=1)
        mrows.append({"переход": f"{a} → {b}", "МО": int(cnt),
                      "отступ от границы у перешедших (медиана)": np.median(gap[(l0 == a) & (l1 == b)]),
                      "у оставшихся в типе": np.median(gap[(l0 == a) & (l1 == a)])})
    margins = pd.DataFrame(mrows)

    # 3. другой способ разбиения спектрального вложения
    base_modal = np.array([np.bincount(r).argmax() for r in L])
    vrows = []
    for assign, title in [("kmeans", "k-средних (основной вариант)"), ("discretize", "дискретизация (Yu, Shi, 2003)"),
                          ("cluster_qr", "QR-разложение (Damle et al., 2019)")]:
        P = L - 1 if assign == "kmeans" else path(As, assign)   # основной вариант уже посчитан на шаге 5
        ch = np.array([(P[:, t] != P[:, t - 1]).mean() for t in range(1, T)])
        modal = np.array([np.bincount(r, minlength=K).argmax() for r in P])
        vrows.append({"способ разбиения": title, "сохраняют тип за месяц, %": 100 * (1 - ch.mean()),
                      "самый большой скачок, %": 100 * ch.max(), "месяц скачка": names[int(ch.argmax()) + 1],
                      f"смен в {jump}, %": 100 * ch[tj - 1],
                      "ARI основных типов МО с основным вариантом": adjusted_rand_score(modal, base_modal)})
        print("готово:", assign, flush=True)
    variants = pd.DataFrame(vrows)

    txt = ["# Диагностика месяцев с массовой сменой типа", "",
           f"Месяцы с наибольшей долей смен: {', '.join(top)}. Подробно разобран {jump}.", "",
           "## 1. Смены типа и изменение признаков по месяцам", "",
           "Изменение признаков – среднее по признакам абсолютное изменение стандартизованного значения за месяц.", "",
           monthly.round(3).to_markdown(index=False), "",
           f"## 2. Кто сменил тип в {jump}", "", "Потоки между типами (строка – тип в прошлом месяце):", "",
           flows.to_markdown(), "",
           "Отступ от границы: расстояние до центра нового типа минус расстояние до центра старого, в месяц перед "
           "скачком. Чем меньше значение, тем ближе МО было к границе.", "", margins.round(3).to_markdown(index=False), "",
           "## 3. Зависит ли месяц скачка от способа разбиения", "",
           "Сеть и сглаживание те же; меняется только последний шаг спектральной кластеризации.", "",
           variants.round(3).to_markdown(index=False), ""]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/jump_diagnostics.md").write_text("\n".join(txt), encoding="utf-8")
    print("\n" + "\n".join(txt))
    print("Готово: reports/jump_diagnostics.md")


if __name__ == "__main__":
    main()
