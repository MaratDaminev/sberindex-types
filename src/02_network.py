"""Шаг 2: сеть МО для каждого месяца по трём правилам построения рёбер.

Запуск из корня проекта:  python src/02_network.py
Конфиг: configs/network.yaml
Результат: data/processed/network/edges_<правило>_k<k>.parquet и reports/network_summary.md

Правила:
  cosine - ребро к k МО с самым большим косинусным сходством векторов признаков в месяце t
  corr   - ребро к k МО, чьи признаки менялись наиболее синхронно в окне вокруг месяца t
  geo    - ребро к k ближайшим МО по автодороге (одинаково для всех месяцев)
Во всех правилах сеть симметризуется: ребро i-j есть, если j среди соседей i или i среди соседей j.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

CFG = yaml.safe_load(open("configs/network.yaml", encoding="utf-8"))


def knn_edges(S, k):
    """S - матрица сходства n x n. Возвращает (i, j, w), i < j, симметричный kNN-граф."""
    S = S.copy()
    np.fill_diagonal(S, -np.inf)
    n = S.shape[0]
    nb = np.argsort(-S, axis=1, kind="stable")[:, :k]   # устойчивая сортировка: при равном сходстве берётся МО с меньшим номером
    i = np.repeat(np.arange(n), k)
    j = nb.ravel()
    w = S[i, j]
    ok = np.isfinite(w) & (w > 0)          # оставляем только положительное сходство
    a, b = np.minimum(i[ok], j[ok]), np.maximum(i[ok], j[ok])
    e = pd.DataFrame({"i": a, "j": b, "w": w[ok]}).groupby(["i", "j"], as_index=False)["w"].max()
    return e


def unit_rows(X):
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(nrm == 0, 1, nrm)


def sim_cosine(X):
    U = unit_rows(X)
    return U @ U.T


def sim_corr(D):
    """D - матрица n x p (изменения признаков в окне). Корреляция Пирсона между строками."""
    U = unit_rows(D - D.mean(axis=1, keepdims=True))
    return U @ U.T


def main():
    feat = pd.read_parquet(CFG["input"]["features"])
    months = sorted(feat["month"].unique())
    ids = np.sort(feat["mo_id"].unique())
    cols = [c for c in feat.columns if c not in ("mo_id", "month")]
    n, T = len(ids), len(months)
    # куб признаков: месяц x МО x признак
    cube = np.stack([feat[feat["month"] == m].set_index("mo_id").loc[ids, cols].to_numpy() for m in months])
    dyn = [c for c in range(len(cols)) if cube[:, :, c].std(axis=0).max() > 1e-9]  # признаки, меняющиеся во времени
    delta = np.diff(cube[:, :, dyn], axis=0)          # (T-1) x n x p: изменение за месяц

    # матрица расстояний по дорогам
    con = pd.read_parquet(CFG["input"]["connection"])
    con = con[con["type"] == CFG["rules"]["geo"]["type"]]
    pos = pd.Series(np.arange(n), index=ids)
    con = con[con["territory_id_x"].isin(ids) & con["territory_id_y"].isin(ids)]
    dist = np.full((n, n), np.nan)
    a, b = pos[con["territory_id_x"]].to_numpy(), pos[con["territory_id_y"]].to_numpy()
    dist[a, b] = con["distance"].to_numpy()
    dist[b, a] = con["distance"].to_numpy()

    out = Path(CFG["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows, store = [], {}
    for k in (CFG["k"], CFG["k_alt"]):
        for rule, rc in CFG["rules"].items():
            if not rc.get("enabled"):
                continue
            parts = []
            geo_static = None
            for t, m in enumerate(months):
                if rule == "cosine":
                    e = knn_edges(sim_cosine(cube[t]), k)
                elif rule == "corr":
                    W = rc["window"]
                    lo = min(max(0, t - W // 2), (T - 1) - W)      # окно из W изменений вокруг месяца t
                    D = delta[lo:lo + W].transpose(1, 0, 2).reshape(n, -1)
                    e = knn_edges(sim_corr(D), k)
                elif rule == "geo":
                    if geo_static is None:
                        d = np.where(np.isnan(dist), np.inf, dist)
                        scale = np.nanmedian(np.sort(d, axis=1)[:, k])     # типичное расстояние до k-го соседа
                        geo_static = knn_edges(np.exp(-np.round(d, 1) / scale), k)
                    e = geo_static
                e = e.assign(month=m)
                parts.append(e)
            E = pd.concat(parts, ignore_index=True)
            E["mo_i"], E["mo_j"] = ids[E["i"]], ids[E["j"]]
            E[["month", "mo_i", "mo_j", "w"]].to_parquet(out / f"edges_{rule}_k{k}.parquet", index=False)
            store[(rule, k)] = E
            # сводка по сети
            deg, comp, giant, dmed = [], [], [], []
            for m, g in E.groupby("month"):
                A = coo_matrix((np.ones(len(g)), (g["i"], g["j"])), shape=(n, n))
                nc, lab = connected_components(A, directed=False)
                comp.append(nc)
                giant.append(np.bincount(lab).max() / n)
                deg.append(2 * len(g) / n)
                dmed.append(np.nanmedian(dist[g["i"], g["j"]]))
            rows.append({"правило": rule, "k": k, "рёбер в месяц": int(len(E) / T),
                         "средняя степень": round(np.mean(deg), 1),
                         "компонент связности": round(np.mean(comp), 1),
                         "доля МО в крупнейшей компоненте": round(np.mean(giant), 3),
                         "медианная длина ребра, км": round(float(np.mean(dmed)), 0)})
    summ = pd.DataFrame(rows)

    # насколько правила дают одни и те же рёбра (Жаккар) и насколько сеть меняется от месяца к месяцу
    def edge_set(E, m):
        g = E[E["month"] == m]
        return set(zip(g["i"], g["j"]))

    def jac(A, B):
        return len(A & B) / len(A | B)

    k = CFG["k"]
    rules = [r for (r, kk) in store if kk == k]
    over = pd.DataFrame(index=rules, columns=rules, dtype=float)
    for r1 in rules:
        for r2 in rules:
            over.loc[r1, r2] = np.mean([jac(edge_set(store[(r1, k)], m), edge_set(store[(r2, k)], m)) for m in months])
    stab = {r: np.mean([jac(edge_set(store[(r, k)], months[t]), edge_set(store[(r, k)], months[t + 1]))
                        for t in range(T - 1)]) for r in rules}

    txt = ["# Сводка по сети", "", f"МО: {n}, месяцев: {T}, признаков: {len(cols)} ({', '.join(cols)})", "",
           "## Структура сети по правилам", "", summ.to_markdown(index=False), "",
           f"## Доля общих рёбер между правилами (Жаккар, k={k})", "", over.round(3).to_markdown(), "",
           f"## Устойчивость сети во времени (Жаккар рёбер соседних месяцев, k={k})", "",
           pd.Series(stab).round(3).to_frame("Жаккар").to_markdown(), ""]
    rep = Path(CFG["output"]["report"])
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))
    print(f"Готово: {out}, отчёт: {rep}")


if __name__ == "__main__":
    main()
