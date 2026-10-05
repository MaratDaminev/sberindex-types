"""Индексы качества кластеров (ICVI).

По признакам (матрица X):
  SW    - средняя ширина силуэта, больше - лучше
  CH    - индекс Калинского-Харабаша, больше - лучше
  S_Dbw - Halkidi & Vazirgiannis (2001): разброс внутри + плотность между кластерами, меньше - лучше
По сети (взвешенная матрица смежности A):
  AVI   - средняя изолируемость (Biswas & Biswas, 2017), больше - лучше
  AVU   - средняя объединяемость (Biswas & Biswas, 2017), меньше - лучше
  MQ    - Modularization Quality, вариант TurboMQ (Mancoridis et al., 1998), больше - лучше; растёт с числом кластеров
  Q     - модулярность Ньюмана, больше - лучше
"""
import numpy as np
from sklearn.metrics import calinski_harabasz_score, silhouette_score


def s_dbw(X, labels):
    ks = np.unique(labels)
    k = len(ks)
    cent = np.array([X[labels == c].mean(axis=0) for c in ks])
    sig = np.array([np.linalg.norm(X[labels == c].var(axis=0)) for c in ks])
    scat = sig.mean() / np.linalg.norm(X.var(axis=0))
    stdev = np.sqrt(sig.sum()) / k

    def dens(point, mask):
        return int((np.linalg.norm(X[mask] - point, axis=1) <= stdev).sum())

    total = 0.0
    for a in range(k):
        ma = labels == ks[a]
        da = dens(cent[a], ma)
        for b in range(k):
            if a == b:
                continue
            mb = labels == ks[b]
            db = dens(cent[b], mb)
            mid = dens((cent[a] + cent[b]) / 2, ma | mb)
            total += mid / max(da, db) if max(da, db) > 0 else 0.0
    return scat + total / (k * (k - 1))


def block_weights(A, labels):
    """W[a, b] - суммарный вес рёбер между кластерами a и b (на диагонали - внутри кластера, каждое ребро один раз)."""
    ks, inv = np.unique(labels, return_inverse=True)
    k = len(ks)
    H = np.zeros((A.shape[0], k))
    H[np.arange(A.shape[0]), inv] = 1
    W = H.T @ (A @ H)
    W[np.diag_indices(k)] /= 2          # внутри кластера каждое ребро посчитано дважды
    return W


def network_indices(A, labels):
    W = block_weights(A, labels)
    k = W.shape[0]
    w_in = np.diag(W)
    cut = W.sum(axis=1) - w_in                      # вес рёбер наружу
    m = w_in.sum() + cut.sum() / 2                  # суммарный вес всех рёбер
    with np.errstate(divide="ignore", invalid="ignore"):
        isola = np.where(w_in + cut > 0, w_in / (w_in + cut), 0.0)
        avi = isola.mean()
        den = cut[:, None] + cut[None, :] - W       # рёбра наружу у обоих кластеров без общих
        unifi = np.where(den > 0, W / den, 0.0)
        np.fill_diagonal(unifi, 0.0)
        avu = unifi.sum() / k
        cf = np.where(2 * w_in + cut > 0, 2 * w_in / (2 * w_in + cut), 0.0)
        mq = cf.sum()
    deg = 2 * w_in + cut
    q = (w_in / m - (deg / (2 * m)) ** 2).sum()
    return {"AVI": avi, "AVU": avu, "MQ": mq, "Q": q}


def all_indices(X, A, labels):
    k = len(np.unique(labels))
    out = {"k": k}
    if 1 < k < len(labels):
        out["SW"] = silhouette_score(X, labels)
        out["CH"] = calinski_harabasz_score(X, labels)
        out["S_Dbw"] = s_dbw(X, labels)
    out.update(network_indices(A, labels))
    sizes = np.bincount(np.unique(labels, return_inverse=True)[1])
    out["мин. размер"] = int(sizes.min())
    out["макс. доля"] = sizes.max() / len(labels)
    return out
