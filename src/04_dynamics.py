"""Шаг 5: динамика кластеров - сглаживание сети во времени, сопоставление кластеров, переходы МО.

Запуск из корня проекта:  python src/04_dynamics.py
Конфиг: configs/dynamics.yaml

Подход (эволюционная кластеризация, Chi et al., 2007):
  1. сеть месяца t смешивается с прошлой: S_t = (1 - alpha) * A_t + alpha * S_(t-1);
  2. на S_t запускается спектральная кластеризация;
  3. кластеры месяца t сопоставляются с кластерами месяца t-1 венгерским алгоритмом
     по числу общих МО, после чего номер кластера означает один и тот же тип во все месяцы.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score, silhouette_score

sys.path.insert(0, str(Path(__file__).parent))
from icvi import network_indices  # noqa: E402

warnings.filterwarnings("ignore")
CFG = yaml.safe_load(open("configs/dynamics.yaml", encoding="utf-8"))
SEED, K = CFG["seed"], CFG["n_clusters"]


def align(prev, cur):
    """Перенумеровывает кластеры cur так, чтобы они максимально совпадали с prev."""
    C = np.array([[np.sum((prev == p) & (cur == c)) for c in range(K)] for p in range(K)])
    r, c = linear_sum_assignment(-C)
    mp = dict(zip(c, r))
    return np.array([mp[x] for x in cur])


def cluster_path(As, alpha):
    """Метки n x T для заданного alpha, уже сопоставленные между месяцами."""
    S, out = None, []
    for t, A in enumerate(As):
        S = A if S is None else (1 - alpha) * A + alpha * S
        lab = SpectralClustering(K, affinity="precomputed", assign_labels="kmeans", random_state=SEED).fit_predict(S)
        out.append(lab if t == 0 else align(out[-1], lab))
    return np.column_stack(out)


def main():
    feat = pd.read_parquet(CFG["input"]["features"])
    months = sorted(feat["month"].unique())
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

    # 1. выбор alpha: устойчивость против качества в каждом месяце
    rows, paths = [], {}
    for a in CFG["alpha_grid"]:
        L = cluster_path(As, a)
        paths[a] = L
        rows.append({
            "alpha": a,
            "доля МО, сохранивших тип за месяц": np.mean([(L[:, t] == L[:, t + 1]).mean() for t in range(T - 1)]),
            "ARI соседних месяцев": np.mean([adjusted_rand_score(L[:, t], L[:, t + 1]) for t in range(T - 1)]),
            "SW по признакам месяца": np.mean([silhouette_score(Xs[t], L[:, t]) for t in range(T)]),
            "Q по сети месяца": np.mean([network_indices(As[t], L[:, t])["Q"] for t in range(T)]),
        })
        print("alpha", a, "готово", flush=True)
    grid = pd.DataFrame(rows).round(3)

    # 2. итоговая типология при выбранном alpha
    a = CFG["alpha"]
    L = paths[a] if a in paths else cluster_path(As, a)
    # нумеруем типы по убыванию среднего размера: тип 1 - самый крупный
    order = np.argsort(-np.array([(L == c).mean() for c in range(K)]))
    L = np.argsort(order)[L] + 1

    out = Path(CFG["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    lab = pd.DataFrame(L, index=ids, columns=months).rename_axis("mo_id")
    lab.reset_index().melt("mo_id", var_name="month", value_name="type").to_parquet(out / "final_labels.parquet", index=False)

    types = list(range(1, K + 1))
    sizes = pd.DataFrame({c: (L == c).sum(axis=0) for c in types}, index=[pd.Timestamp(m).strftime("%Y-%m") for m in months])
    sizes.index.name = "месяц"
    changed = pd.Series([(L[:, t] != L[:, t + 1]).mean() for t in range(T - 1)],
                        index=sizes.index[1:], name="доля МО, сменивших тип")

    # матрица переходов: доля МО типа a в месяце t, оказавшихся в типе b в месяце t+1
    P = np.zeros((K, K))
    for t in range(T - 1):
        for x, y in zip(L[:, t], L[:, t + 1]):
            P[x - 1, y - 1] += 1
    P = pd.DataFrame(P / P.sum(axis=1, keepdims=True), index=types, columns=types).rename_axis("из типа / в тип")

    # события: разделение и слияние
    s, ev = CFG["event_share"], []
    for t in range(T - 1):
        C = np.array([[np.sum((L[:, t] == p) & (L[:, t + 1] == c)) for c in types] for p in types])
        for p in range(K):                                   # тип p распался
            dest = [types[c] for c in range(K) if C[p].sum() and C[p, c] / C[p].sum() >= s]
            if len(dest) > 1:
                ev.append({"месяц": sizes.index[t + 1], "событие": "разделение", "тип": types[p], "связанные типы": dest})
        for c in range(K):                                   # тип c собран из нескольких
            src = [types[p] for p in range(K) if C[:, c].sum() and C[p, c] / C[:, c].sum() >= s]
            if len(src) > 1:
                ev.append({"месяц": sizes.index[t + 1], "событие": "слияние", "тип": types[c], "связанные типы": src})
    ev = pd.DataFrame(ev, columns=["месяц", "событие", "тип", "связанные типы"])

    # сводка по каждому МО
    modal = np.array([np.bincount(r, minlength=K + 1).argmax() for r in L])
    mo = pd.DataFrame({"mo_id": ids, "основной тип": modal,
                       "доля месяцев в основном типе": (L == modal[:, None]).mean(axis=1).round(3),
                       "число смен типа": (L[:, 1:] != L[:, :-1]).sum(axis=1),
                       "число разных типов": [len(set(r)) for r in L]})
    mo.to_csv(out / "mo_summary.csv", index=False, encoding="utf-8-sig")
    sizes.to_csv(out / "type_sizes.csv", encoding="utf-8-sig")
    P.to_csv(out / "transitions.csv", encoding="utf-8-sig")
    ev.to_csv(out / "events.csv", index=False, encoding="utf-8-sig")

    core = mo.groupby("основной тип").agg(МО=("mo_id", "size"),
                                          средняя_доля_месяцев_в_типе=("доля месяцев в основном типе", "mean"),
                                          доля_МО_без_смен=("число смен типа", lambda x: (x == 0).mean())).round(3)
    txt = ["# Динамика кластеров", "", f"МО: {n}, месяцев: {T}, типов: {K}, метод: спектральная кластеризация на сглаженной сети.", "",
           "## Выбор alpha: устойчивость против качества", "", grid.to_markdown(index=False), "",
           f"## Итоговая типология (alpha = {a})", "",
           f"МО, ни разу не сменивших тип: {(mo['число смен типа'] == 0).mean():.1%}; "
           f"МО, проведших в основном типе не менее 75% месяцев: {(mo['доля месяцев в основном типе'] >= 0.75).mean():.1%}", "",
           "### Типы: размер и устойчивость состава", "", core.to_markdown(), "",
           "### Матрица переходов за месяц", "", P.round(3).to_markdown(), "",
           "### Число МО в типе по месяцам", "", sizes.to_markdown(), "",
           "### Доля МО, сменивших тип, по месяцам", "", changed.round(3).to_frame().to_markdown(), "",
           f"### События (порог {s:.0%} состава): {len(ev)}", "",
           ev.to_markdown(index=False) if len(ev) else "Разделений и слияний не зафиксировано.", ""]
    Path(CFG["output"]["report"]).write_text("\n".join(txt), encoding="utf-8")
    print("\n" + "\n".join(txt))
    print("Готово:", out, "| отчёт:", CFG["output"]["report"])


if __name__ == "__main__":
    main()
