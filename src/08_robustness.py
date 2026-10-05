"""Шаг 9 (проверка): устойчивость типологии.

Запуск из корня проекта:  python src/08_robustness.py      (около 10 минут)
Конфиг: configs/robustness.yaml. Результат: reports/robustness.md

Проверки:
  1. подвыборки: 80% МО без возвращения, полный пересчёт сети и типов; сравнение с основным результатом;
  2. выбор числа типов: то же на усреднённой по месяцам сети для разных K;
  3. чувствительность к числу соседей, весу истории и начальному значению генератора;
  4. пересчёт без каждого признака;
  5. независимый расчёт на первой и второй половине периода.
Мера согласия разбиений - скорректированный индекс Рэнда (ARI): 1 - полное совпадение, 0 - случайное.
Устойчивость отдельного типа - индекс Жаккара между типом и самым похожим на него кластером в повторе
(Hennig, 2007): >= 0.75 - устойчивый, 0.6-0.75 - есть структура, но состав неточен, < 0.6 - неустойчивый.
"""
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score as ari

warnings.filterwarnings("ignore")
CFG = yaml.safe_load(open("configs/robustness.yaml", encoding="utf-8"))
SEED, K0, KNN, ALPHA = CFG["seed"], CFG["n_clusters"], CFG["k"], CFG["alpha"]


def knn_graph(X, k):
    """Симметричная kNN-сеть по косинусному сходству строк X (как на шаге 2)."""
    U = X / np.where((nr := np.linalg.norm(X, axis=1, keepdims=True)) == 0, 1, nr)
    S = U @ U.T
    np.fill_diagonal(S, -np.inf)
    n = len(X)
    nb = np.argsort(-S, axis=1, kind="stable")[:, :k]
    i, j = np.repeat(np.arange(n), k), nb.ravel()
    w = S[i, j]
    ok = w > 0
    A = coo_matrix((w[ok], (i[ok], j[ok])), shape=(n, n)).tocsr()
    return A.maximum(A.T)


def spectral(S, K, seed):
    return SpectralClustering(K, affinity="precomputed", assign_labels="kmeans", random_state=seed).fit_predict(S)


def align(prev, cur, K):
    C = np.array([[np.sum((prev == p) & (cur == c)) for c in range(K)] for p in range(K)])
    r, c = linear_sum_assignment(-C)
    mp = dict(zip(c, r))
    return np.array([mp[x] for x in cur])


def path(As, K=K0, alpha=ALPHA, seed=SEED):
    S, out = None, []
    for t, A in enumerate(As):
        S = A if S is None else (1 - alpha) * A + alpha * S
        lab = spectral(S, K, seed)
        out.append(lab if t == 0 else align(out[-1], lab, K))
    return np.column_stack(out)


def modal(L):
    """Основной тип: самый частый; при равенстве - тот из равных, что встретился позже (не зависит от нумерации)."""
    out = []
    for r in L:
        cnt = np.bincount(r)
        tied = np.flatnonzero(cnt == cnt.max())
        out.append(tied[0] if len(tied) == 1 else next(v for v in r[::-1] if v in tied))
    return np.array(out)


def jaccard_per_cluster(base, other):
    """Для каждого кластера base - наибольший Жаккар с кластерами other."""
    out = {}
    for c in np.unique(base):
        a = base == c
        out[c] = max(((a & (other == d)).sum() / (a | (other == d)).sum()) for d in np.unique(other))
    return out


def verdict(j):
    return "устойчивый" if j >= 0.75 else ("состав неточен" if j >= 0.6 else "неустойчивый")


def main():
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    feat = pd.read_parquet(CFG["input"]["features"])
    months = sorted(feat["month"].unique())
    ids = np.sort(feat["mo_id"].unique())
    cols = [c for c in feat.columns if c not in ("mo_id", "month")]
    n, T = len(ids), len(months)
    Xs = [feat[feat["month"] == m].set_index("mo_id").loc[ids, cols].to_numpy() for m in months]
    L0 = (pd.read_parquet(CFG["input"]["labels"]).pivot(index="mo_id", columns="month", values="type")
          .loc[ids].to_numpy())
    base = modal(L0)                                   # основной тип МО из шага 5 (номера 1..K)
    types = np.unique(base)
    As = [knn_graph(X, KNN) for X in Xs]
    check = ari(modal(path(As)), base)
    print(f"сверка с шагом 5: ARI = {check:.3f}", flush=True)
    txt = ["# Проверки устойчивости типологии", "",
           f"МО: {n}, месяцев: {T}. Основной вариант: сеть cosine, k = {KNN}, alpha = {ALPHA}, K = {K0}. "
           "Сравнивается основной тип МО (тип, в котором МО провело больше всего месяцев).", "",
           "ARI: 1 - полное совпадение разбиений, 0 - совпадение на уровне случайного.", ""]

    # 1. подвыборки, полный расчёт
    sub, B = CFG["subsample"], CFG["subsample"]["repeats_dynamic"]
    m = int(round(sub["share"] * n))
    aris, jac = [], []
    for b in range(B):
        idx = np.sort(rng.choice(n, m, replace=False))
        Ls = path([knn_graph(X[idx], KNN) for X in Xs])
        ms = modal(Ls)
        aris.append(ari(ms, base[idx]))
        jac.append(jaccard_per_cluster(base[idx], ms))
        print(f"подвыборка {b + 1}/{B}: ARI {aris[-1]:.3f}", flush=True)
    J = pd.DataFrame(jac)
    tab = pd.DataFrame({"тип": types, "МО": [int((base == c).sum()) for c in types],
                        "Жаккар, среднее": [J[c].mean() for c in types],
                        "Жаккар, минимум": [J[c].min() for c in types],
                        "доля повторов с Жаккаром >= 0.75": [(J[c] >= 0.75).mean() for c in types]})
    tab["вывод"] = [verdict(v) for v in tab["Жаккар, среднее"]]
    txt += [f"## 1. Подвыборки: {sub['share']:.0%} МО, {B} повторов, полный пересчёт сети и типов", "",
            f"ARI основного типа с результатом на всех МО: среднее {np.mean(aris):.3f}, "
            f"минимум {np.min(aris):.3f}, максимум {np.max(aris):.3f}.", "", tab.round(3).to_markdown(index=False), ""]

    # 2. выбор K на усреднённой сети
    Abar = sum(As) / T
    rows = []
    for K in sub["k_scan"]:
        full = spectral(Abar, K, SEED)
        a, jm = [], []
        for b in range(sub["repeats_k_scan"]):
            idx = np.sort(rng.choice(n, m, replace=False))
            s = spectral(csr_matrix(Abar[idx][:, idx]), K, SEED)
            a.append(ari(s, full[idx]))
            jm.append(min(jaccard_per_cluster(full[idx], s).values()))
        rows.append({"K": K, "ARI, среднее": np.mean(a), "ARI, минимум": np.min(a),
                     "Жаккар худшего кластера, среднее": np.mean(jm),
                     "мин. размер кластера": int(np.bincount(full).min()),
                     "ARI с основным типом (K из шага 5)": ari(full, base)})
        print(f"K = {K}: ARI {np.mean(a):.3f}", flush=True)
    txt += [f"## 2. Число типов: устойчивость на усреднённой по месяцам сети ({sub['repeats_k_scan']} подвыборок на каждое K)", "",
            "Сеть усреднена по 24 месяцам; в подвыборке берётся подсеть на оставшихся МО.", "",
            pd.DataFrame(rows).round(3).to_markdown(index=False), ""]

    # 3. чувствительность к параметрам
    sens, rows = CFG["sensitivity"], []
    for k in sens["graph_k"]:
        L = path(As if k == KNN else [knn_graph(X, k) for X in Xs])
        rows.append({"что изменено": f"число соседей k = {k}", "ARI основного типа с базой": ari(modal(L), base),
                     "сохраняют тип за месяц": np.mean(L[:, 1:] == L[:, :-1])})
    for a in sens["alpha"]:
        L = path(As, alpha=a)
        rows.append({"что изменено": f"вес истории alpha = {a}", "ARI основного типа с базой": ari(modal(L), base),
                     "сохраняют тип за месяц": np.mean(L[:, 1:] == L[:, :-1])})
    for s in sens["seeds"]:
        L = path(As, seed=s)
        rows.append({"что изменено": f"seed = {s}", "ARI основного типа с базой": ari(modal(L), base),
                     "сохраняют тип за месяц": np.mean(L[:, 1:] == L[:, :-1])})
    print("чувствительность: готово", flush=True)
    txt += ["## 3. Чувствительность к параметрам", "", pd.DataFrame(rows).round(3).to_markdown(index=False), ""]

    # 4. удаление признаков
    if CFG.get("drop_features"):
        rows = []
        for f, name in enumerate(cols):
            keep = [c for c in range(len(cols)) if c != f]
            L = path([knn_graph(X[:, keep], KNN) for X in Xs])
            ms = modal(L)
            j = jaccard_per_cluster(base, ms)
            rows.append({"удалённый признак": name, "ARI основного типа с базой": ari(ms, base),
                         "тип с наименьшим Жаккаром": min(j, key=j.get), "его Жаккар": min(j.values())})
            print(f"без признака {name}: ARI {rows[-1]['ARI основного типа с базой']:.3f}", flush=True)
        txt += ["## 4. Пересчёт без одного признака", "",
                "Чем ниже ARI, тем сильнее типология зависит от этого признака.", "",
                pd.DataFrame(rows).sort_values("ARI основного типа с базой").round(3).to_markdown(index=False), ""]

    # 5. половины периода
    if CFG.get("split_years"):
        h = T // 2
        m1, m2 = modal(path(As[:h])), modal(path(As[h:]))
        txt += ["## 5. Независимый расчёт на двух половинах периода", "",
                f"Первая половина: {pd.Timestamp(months[0]):%Y-%m} - {pd.Timestamp(months[h - 1]):%Y-%m}, "
                f"вторая: {pd.Timestamp(months[h]):%Y-%m} - {pd.Timestamp(months[-1]):%Y-%m}.", "",
                f"- ARI основных типов между половинами: {ari(m1, m2):.3f}",
                f"- ARI первой половины с базой: {ari(m1, base):.3f}",
                f"- ARI второй половины с базой: {ari(m2, base):.3f}", ""]

    Path(CFG["output"]).write_text("\n".join(txt), encoding="utf-8")
    print("\n" + "\n".join(txt))
    print(f"Готово: {CFG['output']} ({(time.time() - t0) / 60:.1f} мин)")


if __name__ == "__main__":
    main()
