"""Шаги 3-4: кластеризация МО в каждом месяце несколькими методами и сравнение по ICVI.

Запуск из корня проекта:  python src/03_cluster.py
Конфиг: configs/cluster.yaml
Результат: метки кластеров, помесячные метрики и сводный отчёт reports/icvi_comparison.md
"""
import sys
import warnings
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from scipy.sparse import coo_matrix
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering

sys.path.insert(0, str(Path(__file__).parent))
from icvi import all_indices  # noqa: E402

warnings.filterwarnings("ignore")
CFG = yaml.safe_load(open("configs/cluster.yaml", encoding="utf-8"))
SEED, K = CFG["seed"], CFG["n_clusters"]


def run_method(name, X, A):
    if name == "kmeans":
        return KMeans(K, n_init=10, random_state=SEED).fit_predict(X)
    if name == "louvain":
        G = nx.from_scipy_sparse_array(A)
        comms = nx.community.louvain_communities(G, weight="weight", resolution=CFG["louvain_resolution"], seed=SEED)
        lab = np.empty(A.shape[0], dtype=int)
        for c, nodes in enumerate(comms):
            lab[list(nodes)] = c
        return lab
    if name == "spectral":
        return SpectralClustering(K, affinity="precomputed", assign_labels="kmeans", random_state=SEED).fit_predict(A)
    if name == "ward_net":
        return AgglomerativeClustering(K, linkage="ward", connectivity=A).fit_predict(X)
    raise ValueError(name)


def main():
    feat = pd.read_parquet(CFG["input"]["features"])
    months = sorted(feat["month"].unique())
    ids = np.sort(feat["mo_id"].unique())
    cols = [c for c in feat.columns if c not in ("mo_id", "month")]
    n = len(ids)
    pos = pd.Series(np.arange(n), index=ids)
    Xs = {m: feat[feat["month"] == m].set_index("mo_id").loc[ids, cols].to_numpy() for m in months}

    graphs = {}
    for g in CFG["graphs"]:
        E = pd.read_parquet(Path(CFG["input"]["network_dir"]) / f"edges_{g}.parquet")
        graphs[g] = {}
        for m, e in E.groupby("month"):
            i, j, w = pos[e["mo_i"]].to_numpy(), pos[e["mo_j"]].to_numpy(), e["w"].to_numpy()
            graphs[g][m] = coo_matrix((np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(n, n)).tocsr()

    runs = [("kmeans", CFG["reference_graph"])] if CFG["methods"].get("kmeans") else []
    runs += [(meth, g) for g in CFG["graphs"] for meth in ("louvain", "spectral", "ward_net") if CFG["methods"].get(meth)]

    labels, rows = [], []
    for meth, g in runs:
        name = meth if meth == "kmeans" else f"{meth}@{g}"
        for m in months:
            lab = run_method(meth, Xs[m], graphs[g][m])
            labels.append(pd.DataFrame({"config": name, "month": m, "mo_id": ids, "label": lab}))
            rows.append({"config": name, "метод": meth, "сеть": "-" if meth == "kmeans" else g, "month": m,
                         **all_indices(Xs[m], graphs[g][m], lab)})
        print("готово:", name, flush=True)
    monthly = pd.DataFrame(rows)

    # перебор числа кластеров (kmeans по признакам)
    scan = []
    for k in CFG["scan_k"]:
        for m in months:
            lab = KMeans(k, n_init=10, random_state=SEED).fit_predict(Xs[m])
            r = all_indices(Xs[m], graphs[CFG["reference_graph"]][m], lab)
            scan.append({"K": k, **{x: r[x] for x in ("SW", "CH", "S_Dbw", "AVI", "AVU", "Q")}})
    scan = pd.DataFrame(scan).groupby("K").mean().round(3)

    out = Path(CFG["output"]["labels"])
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(labels, ignore_index=True).to_parquet(out, index=False)
    monthly.to_csv(CFG["output"]["icvi_monthly"], index=False, encoding="utf-8-sig")

    num = ["k", "SW", "CH", "S_Dbw", "AVI", "AVU", "MQ", "Q", "мин. размер", "макс. доля"]
    summ = monthly.groupby(["метод", "сеть"], sort=False)[num].mean()
    summ["k"] = summ["k"].round(1)
    summ["CH"] = summ["CH"].round(0)
    summ["мин. размер"] = summ["мин. размер"].round(0)
    summ = summ.round(3).reset_index()
    txt = ["# Сравнение методов кластеризации по ICVI", "",
           f"Среднее по {len(months)} месяцам, МО: {n}. SW, CH, S_Dbw считаются по признакам; "
           "AVI, AVU, MQ, Q - по сети, на которой работал метод (для kmeans - по сети "
           f"{CFG['reference_graph']}).", "",
           "Больше - лучше: SW, CH, AVI, MQ, Q. Меньше - лучше: S_Dbw, AVU. "
           "MQ растёт с числом кластеров k, поэтому сравнивать его можно только при близких k.", "",
           summ.to_markdown(index=False), "",
           "## Перебор числа кластеров (kmeans по признакам, среднее по месяцам)", "",
           scan.to_markdown(), ""]
    Path(CFG["output"]["report"]).write_text("\n".join(txt), encoding="utf-8")
    print("\n" + "\n".join(txt))
    print("Готово:", out, "| отчёт:", CFG["output"]["report"])


if __name__ == "__main__":
    main()
