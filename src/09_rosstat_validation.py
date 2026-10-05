"""Шаг 10 (проверка): различаются ли типы по данным Росстата, которые не участвовали в кластеризации.

Запуск из корня проекта:  python src/09_rosstat_validation.py
Перед этим: python src/00_download_rosstat.py. Конфиг: configs/rosstat.yaml (блок validation).

Типы построены только по безналичным тратам. Здесь они сопоставляются с зарплатой, населением и структурой
занятости из БДПМО Росстата. Для каждого показателя считается критерий Краскела-Уоллиса и размер эффекта
эпсилон-квадрат = (H - k + 1) / (n - k): доля разброса рангов показателя, объясняемая типом (0 - нет связи, 1 - полная).
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kruskal

CFG = yaml.safe_load(open("configs/rosstat.yaml", encoding="utf-8"))
V = CFG["validation"]
RAW = Path(CFG["output_dir"])
CYR = str.maketrans("АВСЕНКМОРТХ", "ABCEHKMOPTX")          # в названиях разделов часть букв набрана кириллицей
SECTION_NAMES = {"A": "сельское хозяйство", "B": "добыча", "C": "обработка", "D": "энергетика", "G": "торговля",
                 "H": "транспорт", "M": "наука и проф. услуги", "O": "госуправление", "P": "образование",
                 "Q": "здравоохранение", "R": "культура и спорт"}


def load(code):
    d = pd.read_csv(RAW / f"{code}.csv", sep=";", dtype=str)
    d["value"] = pd.to_numeric(d["indicator_value"], errors="coerce")
    d["year"] = d["year"].astype(int)
    return d.dropna(subset=["value"])


def section(s):
    parts = s.split()
    return parts[1].translate(CYR) if parts[0] == "Раздел" else "ALL"


def main():
    year = V["year"]
    # 1. коды ОКТМО каждого МО из справочника СберИндекса (все версии: границы и коды менялись)
    dic = pd.read_excel(V["dictionary"], dtype={"oktmo": str})
    dic["oktmo8"] = dic["oktmo"].str.replace("-", "", regex=False).str[:8]
    mo = pd.read_csv(V["mo_summary"])
    ids = set(mo["mo_id"])
    dic = dic[dic["territory_id"].isin(ids)]
    code2tid = dic.drop_duplicates(["oktmo8", "territory_id"])[["oktmo8", "territory_id"]]
    amb = code2tid["oktmo8"].duplicated(keep=False)
    code2tid = code2tid[~amb]                              # код, относящийся к двум МО выборки, не используем

    def attach(d):
        """Строки Росстата -> territory_id: сначала по коду года, затем по устойчивому коду."""
        a = d.merge(code2tid, left_on="oktmo", right_on="oktmo8", how="left")
        b = d.merge(code2tid, left_on="oktmo_stable", right_on="oktmo8", how="left")
        a["territory_id"] = a["territory_id"].fillna(b["territory_id"])
        return a.dropna(subset=["territory_id"]).astype({"territory_id": int})

    # 2. население
    pop = attach(load("Y48112027"))
    pop = pop[(pop["year"] == year) & (pop["indicator_period"] == "На 1 января")]
    pw = pop.pivot_table(index="territory_id", columns="mest", values="value", aggfunc="sum")
    out = pd.DataFrame(index=sorted(ids))
    out.index.name = "mo_id"
    out["население"] = pw.get("Все население")
    urban = pw.get("Городское население", pd.Series(dtype=float)).reindex(out.index)
    out["доля городского населения, %"] = np.where(out["население"].notna(), 100 * urban.fillna(0) / out["население"], np.nan)

    # 3. численность работников и зарплата по разделам ОКВЭД2
    emp = attach(load("Y48423005"))
    wage = attach(load("Y48423007"))
    for d in (emp, wage):
        d["sec"] = d["okved2"].map(section)
    emp = emp[(emp["year"] == year) & (emp["indicator_period"] == V["annual_period"])]
    wage = wage[(wage["year"] == year) & (wage["indicator_period"] == V["annual_period"])]
    ew = emp.pivot_table(index="territory_id", columns="sec", values="value", aggfunc="sum")
    # зарплата: при нескольких кодах на одно МО - среднее, взвешенное по численности
    wt = wage[wage["sec"] == "ALL"].merge(emp[emp["sec"] == "ALL"][["oktmo", "value"]].rename(columns={"value": "n"}),
                                          on="oktmo", how="left")
    wt["n"] = wt["n"].fillna(1.0)
    wsum = (wt["value"] * wt["n"]).groupby(wt["territory_id"]).sum() / wt["n"].groupby(wt["territory_id"]).sum()
    out["зарплата, руб."] = wsum
    out["работников организаций на 100 жителей"] = 100 * ew.get("ALL") / out["население"]
    for s in V["sections"]:
        if s in ew:
            out[f"занятые: {SECTION_NAMES.get(s, s)}, %"] = 100 * ew[s] / ew["ALL"]

    out = out.join(mo.set_index("mo_id")["основной тип"])
    macro = {int(k): v for k, v in (yaml.safe_load(open(V["interpret"], encoding="utf-8")).get("macro_types") or {}).items()}
    names = {int(k): v for k, v in yaml.safe_load(open(V["interpret"], encoding="utf-8"))["type_names"].items()}
    out["макротип"] = out["основной тип"].map(macro)
    Path(V["out_table"]).parent.mkdir(parents=True, exist_ok=True)
    out.round(2).to_csv(V["out_table"], encoding="utf-8-sig")

    # 4. сравнение типов
    vars_ = [c for c in out.columns if c not in ("основной тип", "макротип")]
    n_all = len(out)

    def kw(group_col):
        rows = []
        for v in vars_:
            d = out[[v, group_col]].dropna()
            cov = len(d) / n_all
            if cov < V["min_coverage"] or d[group_col].nunique() < 2:
                rows.append({"показатель": v, "есть данные, % МО": 100 * cov, "эпсилон-квадрат": np.nan, "p": np.nan})
                continue
            groups = [g[v].to_numpy() for _, g in d.groupby(group_col)]
            H, p = kruskal(*groups)
            k, n = len(groups), len(d)
            rows.append({"показатель": v, "есть данные, % МО": 100 * cov, "эпсилон-квадрат": (H - k + 1) / (n - k), "p": p})
        return pd.DataFrame(rows)

    k6, k4 = kw("основной тип"), kw("макротип")
    eff = k6.rename(columns={"эпсилон-квадрат": "эпсилон-квадрат, 6 типов", "p": "p, 6 типов"}).merge(
        k4[["показатель", "эпсилон-квадрат"]].rename(columns={"эпсилон-квадрат": "эпсилон-квадрат, 4 макротипа"}), on="показатель")
    eff = eff.sort_values("эпсилон-квадрат, 6 типов", ascending=False)
    eff["p, 6 типов"] = eff["p, 6 типов"].map(lambda x: "" if pd.isna(x) else ("< 0.001" if x < 0.001 else f"{x:.3f}"))
    shown = k6[k6["эпсилон-квадрат"].notna()]["показатель"].tolist()      # показатели с достаточным покрытием
    dropped = [v for v in vars_ if v not in shown]
    med = out.groupby("основной тип")[shown].median().T
    med.columns = [f"тип {c}" for c in med.columns]
    med["все МО"] = out[shown].median()
    medm = out.groupby("макротип")[shown].median().T
    multi = int((emp[emp["sec"] == "ALL"].groupby("territory_id")["oktmo"].nunique() > 1).sum())
    strong = eff[eff["эпсилон-квадрат, 6 типов"] >= 0.14]["показатель"].tolist()
    weak = eff[eff["эпсилон-квадрат, 6 типов"] < 0.06]["показатель"].tolist()

    matched = out["зарплата, руб."].notna().mean()
    txt = ["# Внешняя проверка типов данными Росстата", "",
           f"Источник: БДПМО Росстата в обработке проекта «Если быть точным», {year} год. Показатели не участвовали "
           "в построении типов. Зарплата и численность работников относятся к организациям без субъектов малого "
           "предпринимательства.", "",
           f"МО в выборке: {n_all}. Сопоставлено с Росстатом: по населению {out['население'].notna().mean():.1%}, "
           f"по зарплате {matched:.1%}. Кодов ОКТМО, относящихся сразу к двум МО выборки и потому не использованных: "
           f"{int(amb.sum())}. МО, которым в {year} году соответствует несколько кодов Росстата (значения "
           f"просуммированы, зарплата усреднена с весами по численности): {multi}.", "",
           "## Насколько типы различаются по каждому показателю", "",
           "Эпсилон-квадрат - доля разброса рангов показателя, объясняемая типом. Ориентиры: от 0,14 - сильная связь, "
           "0,06-0,14 - средняя, меньше 0,06 - слабая.", "",
           eff.round(3).to_markdown(index=False), "",
           *(["Не сравнивались из-за малого покрытия: " + ", ".join(dropped) + ".", ""] if dropped else []),
           "## Медианы по шести типам", "", med.round(1).to_markdown(), "",
           "## Медианы по четырём макротипам", "", medm.round(1).to_markdown(), "",
           "## Что сходится и что нет", "",
           "- Сильная связь с типом: " + (", ".join(strong) if strong else "нет таких показателей") + ".",
           "- Слабая связь с типом: " + (", ".join(weak) if weak else "нет таких показателей") + ".", "",
           "Типы: " + "; ".join(f"{t} - {n}" for t, n in names.items()) + ".", ""]
    Path(V["report"]).write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))
    print("Готово:", V["report"], "| таблица:", V["out_table"])


if __name__ == "__main__":
    main()
