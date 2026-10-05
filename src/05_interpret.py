"""Шаг 6: интерпретация типов - профили, состав, примеры, дерево решений, разбор региона.

Запуск из корня проекта:  python src/05_interpret.py
Конфиг: configs/interpret.yaml
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier, export_text

CFG = yaml.safe_load(open("configs/interpret.yaml", encoding="utf-8"))
NAMES = {int(k): v for k, v in CFG["type_names"].items()}
MACRO = {int(k): v for k, v in (CFG.get("macro_types") or {}).items()}


def main():
    I = CFG["input"]
    dic = (pd.read_excel(I["dictionary"]).sort_values("year_to")
           .drop_duplicates("territory_id", keep="last").set_index("territory_id"))
    L = pd.read_parquet(I["labels"])
    L["date"] = pd.to_datetime(L["month"]).dt.strftime("%Y-%m")
    mo = pd.read_csv(I["mo_summary"]).set_index("mo_id")

    raw = pd.read_parquet(I["consumption"])
    w = raw.pivot_table(index=["territory_id", "date"], columns="category", values="value")
    tot = w.pop(CFG["total_category"])
    sh = w.div(tot, axis=0) * 100
    sh["Прочее"] = 100 - sh.sum(axis=1)
    cats = list(sh.columns)
    sh["Траты, руб."] = tot
    sh["Траты к медиане месяца"] = tot / tot.groupby(level="date").transform("median")
    ma = pd.read_parquet(I["market_access"]).set_index("territory_id")["market_access"]
    X = sh.reset_index().merge(L[["mo_id", "date", "type"]], left_on=["territory_id", "date"], right_on=["mo_id", "date"])
    X["Доступность рынков"] = X["mo_id"].map(ma)
    cols = cats + ["Траты, руб.", "Доступность рынков"]

    # 1. профили типов
    prof = X.groupby("type")[cols].mean()
    allm = X[cols].mean()
    lift = prof / allm
    prof_out = prof.copy()
    prof_out.loc["все МО"] = allm
    prof_out.insert(0, "название", [NAMES.get(t, "") for t in prof.index] + [""])

    # 2. состав типов
    m = mo.join(dic[["municipal_district_name_short", "municipal_district_type", "municipal_district_status", "region_name"]])
    m = m.rename(columns={"municipal_district_name_short": "МО", "region_name": "регион", "municipal_district_type": "вид МО"})
    m["название типа"] = m["основной тип"].map(NAMES)
    m["макротип"] = m["основной тип"].map(MACRO)
    kind = (pd.crosstab(m["основной тип"], m["вид МО"], normalize="index") * 100).round(0)
    caps = m[m["municipal_district_status"] == "административный_центр_субъекта"]["основной тип"].value_counts()

    # 3. типичные МО: ни разу не меняли тип и ближе всего к среднему профилю типа
    Z = (X[cols] - X[cols].mean()) / X[cols].std()
    Z["mo_id"], Z["type"] = X["mo_id"].values, X["type"].values
    zc = Z.groupby("type")[cols].mean()
    zm = Z.groupby("mo_id")[cols].mean()
    m["расстояние до центра типа"] = [np.linalg.norm(zm.loc[i] - zc.loc[t]) for i, t in zip(m.index, m["основной тип"])]

    # 4. дерево решений как объяснение типов (суррогатная модель)
    feats = cats + ["Траты к медиане месяца", "Доступность рынков"]
    tr, te = next(GroupShuffleSplit(1, test_size=0.3, random_state=CFG["seed"]).split(X, groups=X["mo_id"]))
    tree = DecisionTreeClassifier(max_depth=CFG["tree_depth"], min_samples_leaf=200, random_state=CFG["seed"])
    tree.fit(X.iloc[tr][feats], X.iloc[tr]["type"])
    acc_tr, acc_te = tree.score(X.iloc[tr][feats], X.iloc[tr]["type"]), tree.score(X.iloc[te][feats], X.iloc[te]["type"])
    imp = pd.Series(tree.feature_importances_, index=feats).sort_values(ascending=False).round(3)

    # 5. отчёт
    txt = ["# Интерпретация типов локальных экономик", "",
           f"МО: {len(m)}, месяцев: {L['date'].nunique()}. Профили посчитаны по исходным (не стандартизованным) данным.", "",
           "## Профили типов: средние доли трат, %, уровень трат и доступность рынков", "",
           prof_out.round(1).to_markdown(), "",
           "## Отличие от среднего по всем МО (1 = как в среднем)", "", lift.round(2).to_markdown(), "",
           "## Состав типов по виду МО, %", "", kind.to_markdown(), "",
           "Административные центры субъектов по типам: " +
           ", ".join(f"тип {t} - {int(c)}" for t, c in caps.sort_index().items()), ""]
    for t in sorted(NAMES):
        g = m[m["основной тип"] == t]
        top3 = lift.loc[t][lift.loc[t] > 1.02].sort_values(ascending=False).head(3)
        low2 = lift.loc[t][lift.loc[t] < 0.98].sort_values().head(3)
        core = g[g["число смен типа"] == 0].nsmallest(CFG["n_examples"], "расстояние до центра типа")
        edge = g.nsmallest(3, "доля месяцев в основном типе")
        txt += [f"## Тип {t}. {NAMES[t]}", "",
                *([f"- Макротип: {MACRO[t]}"] if t in MACRO else []),
                f"- МО с этим основным типом: {len(g)}; ни разу не меняли тип: {(g['число смен типа'] == 0).mean():.0%}",
                "- Выше среднего: " + ", ".join(f"{k} (x{v:.2f})" for k, v in top3.items()),
                "- Ниже среднего: " + ", ".join(f"{k} (x{v:.2f})" for k, v in low2.items()),
                "- Регионы-лидеры: " + ", ".join(f"{r} ({c})" for r, c in g["регион"].value_counts().head(5).items()),
                "- Типичные МО: " + ", ".join(f"{r.МО} ({r.регион})" for r in core.itertuples()),
                "- Пограничные МО (чаще всего меняли тип): " +
                ", ".join(f"{r.МО} ({r.регион})" for r in edge.itertuples()), ""]
    txt += ["## Дерево решений: какими правилами описываются типы", "",
            f"Дерево глубины {CFG['tree_depth']} обучено воспроизводить типы по исходным признакам. "
            f"Это объясняющая (суррогатная) модель, а не проверка качества кластеров. "
            f"Доля верно воспроизведённых типов: {acc_tr:.1%} на обучающей части, {acc_te:.1%} на отложенных 30% МО.", "",
            "Важность признаков:", "", imp[imp > 0].to_frame("важность").to_markdown(), "",
            "```", export_text(tree, feature_names=feats, decimals=1), "```", ""]
    fr = CFG.get("focus_region")
    if fr:
        b = m[m["регион"].str.contains(fr, na=False)]
        txt += [f"## Регион: {fr}", "", f"МО в выборке: {len(b)}", ""]
        for t, g in b.groupby("основной тип"):
            txt += [f"- Тип {t}. {NAMES.get(t, '')} ({len(g)}): " + ", ".join(sorted(g["МО"])), ""]
        mv = b[b["число смен типа"] > 0].sort_values("доля месяцев в основном типе")
        txt += [f"МО региона, хотя бы раз менявших тип: {len(mv)} из {len(b)}. Наименее устойчивые: " +
                ", ".join(f"{n} ({v:.0%} месяцев в основном типе)"
                          for n, v in zip(mv["МО"].head(10), mv["доля месяцев в основном типе"].head(10))), ""]
    Path(CFG["output"]["report"]).write_text("\n".join(txt), encoding="utf-8")
    out = m.reset_index()[["mo_id", "МО", "регион", "вид МО", "основной тип", "название типа", "макротип",
                           "доля месяцев в основном типе", "число смен типа", "расстояние до центра типа"]]
    out.round(3).to_csv(CFG["output"]["mo_table"], index=False, encoding="utf-8-sig")
    print("\n".join(txt))
    print("Готово:", CFG["output"]["report"], "| таблица МО:", CFG["output"]["mo_table"])


if __name__ == "__main__":
    main()
