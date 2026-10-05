"""Шаг 7: сборка интерактивного лендинга site/index.html из результатов шагов 1-6.

Запуск из корня проекта:  python src/06_landing.py
Конфиг: configs/landing.yaml
Страница самодостаточна: все данные встроены в один HTML-файл, интернет для просмотра не нужен.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CFG = yaml.safe_load(open("configs/landing.yaml", encoding="utf-8"))
I = CFG["input"]


def albers(lon, lat, lat1=52.0, lat2=64.0, lat0=56.0, lon0=100.0):
    """Равновеликая коническая проекция Альберса - привычный вид карты России."""
    lon = np.where(lon < 0, lon + 360, lon)          # Чукотка за 180-м меридианом
    r = np.radians
    n = (np.sin(r(lat1)) + np.sin(r(lat2))) / 2
    c = np.cos(r(lat1)) ** 2 + 2 * n * np.sin(r(lat1))
    rho = np.sqrt(c - 2 * n * np.sin(r(lat))) / n
    rho0 = np.sqrt(c - 2 * n * np.sin(r(lat0))) / n
    th = n * r(lon - lon0)
    return rho * np.sin(th), -(rho0 - rho * np.cos(th))


def main():
    icfg = yaml.safe_load(open(I["interpret"], encoding="utf-8"))
    names = {int(k): v for k, v in icfg["type_names"].items()}
    macro = {int(k): v for k, v in (icfg.get("macro_types") or {}).items()}
    alpha = yaml.safe_load(open(I["dynamics"], encoding="utf-8"))["alpha"]
    K = len(names)
    dic = (pd.read_excel(I["dictionary"]).sort_values("year_to")
           .drop_duplicates("territory_id", keep="last").set_index("territory_id"))
    L = pd.read_parquet(I["labels"])
    lab = L.pivot(index="mo_id", columns="month", values="type").sort_index()
    months = [pd.Timestamp(m).strftime("%Y-%m") for m in lab.columns]
    ids = lab.index.to_numpy()
    mo = pd.read_csv(I["mo_summary"]).set_index("mo_id").loc[ids]

    # исходные показатели: доли трат, уровень трат, доступность рынков
    raw = pd.read_parquet(I["consumption"])
    raw = raw[raw["territory_id"].isin(ids)]
    w = raw.pivot_table(index=["territory_id", "date"], columns="category", values="value")
    tot = w.pop(CFG["total_category"])
    sh = w.div(tot, axis=0) * 100
    sh["Прочее"] = 100 - sh.sum(axis=1)
    cats = list(sh.columns)
    sh["Траты на жителя, руб."] = tot
    ma = pd.read_parquet(I["market_access"]).set_index("territory_id")["market_access"]
    feat_names = cats + ["Траты на жителя, руб.", "Доступность рынков"]
    mo_prof = sh.groupby(level=0).mean().loc[ids]
    mo_prof["Доступность рынков"] = ma.reindex(ids).to_numpy()
    country = mo_prof.mean()

    # профили типов по всем парам МО-месяц
    X = sh.reset_index().merge(L.assign(date=pd.to_datetime(L["month"]).dt.strftime("%Y-%m"))[["mo_id", "date", "type"]],
                               left_on=["territory_id", "date"], right_on=["mo_id", "date"])
    X["Доступность рынков"] = X["mo_id"].map(ma)
    tprof = X.groupby("type")[feat_names].mean()
    tall = X[feat_names].mean()

    # двойники: ближайшие МО по среднему вектору стандартизованных признаков из других регионов
    F = pd.read_parquet(I["features"])
    Z = F.drop(columns="month").groupby("mo_id").mean().loc[ids].to_numpy()
    region = dic["region_name"].reindex(ids).fillna("").to_numpy()
    d2 = ((Z[:, None, :] - Z[None, :, :]) ** 2).sum(-1)
    d2[region[:, None] == region[None, :]] = np.inf
    twins = np.argsort(d2, axis=1)[:, :CFG["n_twins"]]

    x, y = albers(dic["municipal_district_center_lon"].reindex(ids).to_numpy(float),
                  dic["municipal_district_center_lat"].reindex(ids).to_numpy(float))
    sx = (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x))
    sy = (y - np.nanmin(y)) / (np.nanmax(x) - np.nanmin(x))        # тот же масштаб, что по x

    mos = []
    for i, mid in enumerate(ids):
        mos.append({"id": int(mid), "n": str(dic["municipal_district_name_short"].get(mid, mid)),
                    "r": str(region[i]), "k": str(dic["municipal_district_type"].get(mid, "")),
                    "x": None if np.isnan(sx[i]) else round(float(sx[i]), 5),
                    "y": None if np.isnan(sy[i]) else round(float(sy[i]), 5),
                    "t": "".join(str(int(v)) for v in lab.iloc[i].to_numpy()),
                    "m": int(mo["основной тип"].iloc[i]), "s": round(float(mo["доля месяцев в основном типе"].iloc[i]), 3),
                    "c": int(mo["число смен типа"].iloc[i]),
                    "p": [round(float(v), 1) for v in mo_prof.iloc[i][feat_names]],
                    "tw": [int(j) for j in twins[i]]})

    types = []
    for t in range(1, K + 1):
        g = mo[mo["основной тип"] == t]
        reg = pd.Series(region[(mo["основной тип"] == t).to_numpy()]).value_counts().head(4)
        types.append({"id": t, "name": names[t], "macro": macro.get(t, ""), "color": CFG["colors"][t], "n": int(len(g)),
                      "stable": round(float((g["число смен типа"] == 0).mean()), 3),
                      "prof": [round(float(v), 1) for v in tprof.loc[t]],
                      "lift": [round(float(v), 2) for v in tprof.loc[t] / tall],
                      "regions": [f"{r} ({c})" for r, c in reg.items()],
                      "sizes": [int((lab[c] == t).sum()) for c in lab.columns]})

    P = pd.read_csv(I["transitions"], index_col=0).to_numpy().round(3).tolist()
    ic = pd.read_csv(I["icvi_monthly"])
    icv = (ic.groupby(["метод", "сеть"], sort=False)[["k", "SW", "CH", "S_Dbw", "AVI", "AVU", "MQ", "Q"]].mean()
           .round(3).reset_index())
    icv["CH"] = icv["CH"].round(0)
    icv["k"] = icv["k"].round(1)
    Lm = lab.to_numpy()
    changed = [None] + [round(float((Lm[:, t] != Lm[:, t - 1]).mean()), 3) for t in range(1, len(months))]

    data = {"title": CFG["title"], "subtitle": CFG["subtitle"], "author": CFG.get("author") or CFG.get("authors") or "",
            "repo": CFG.get("repo_url") or "", "alpha": alpha, "months": months, "features": feat_names,
            "country": [round(float(v), 1) for v in country[feat_names]], "types": types, "mos": mos, "P": P,
            "changed": changed, "icvi": {"cols": list(icv.columns), "rows": icv.to_numpy().tolist()},
            "presets": CFG.get("map_presets") or [],
            "stats": {"n": int(len(ids)), "never": round(float((mo["число смен типа"] == 0).mean()), 3),
                      "s75": round(float((mo["доля месяцев в основном типе"] >= 0.75).mean()), 3)}}
    html = Path(I["template"]).read_text(encoding="utf-8").replace("/*DATA*/null", json.dumps(data, ensure_ascii=False))
    out = Path(CFG["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Готово: {out} ({out.stat().st_size / 1024:.0f} КБ). Открой файл в браузере.")


if __name__ == "__main__":
    main()
