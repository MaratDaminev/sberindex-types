"""Шаг 1: таблица МО x месяц x признаки.

Запуск из корня проекта:  python src/01_features.py
Конфиг: configs/features.yaml
Результат: data/processed/features.parquet и reports/data_audit.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CFG_PATH = Path("configs/features.yaml")


def load_table(path):
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix in (".xlsx", ".xls"):
        return pd.read_excel(p)
    return pd.read_csv(p, sep=None, engine="python")  # разделитель определяется автоматически


def to_month(s):
    """Приводит колонку месяца к pandas Period[M]."""
    if pd.api.types.is_numeric_dtype(s) and s.dropna().between(190001, 209912).all():
        s = s.astype("Int64").astype(str)
        return pd.to_datetime(s, format="%Y%m").dt.to_period("M")
    return pd.to_datetime(s).dt.to_period("M")


def fail(msg, df=None):
    print("\nОШИБКА:", msg)
    if df is not None:
        print("Колонки в файле:", list(df.columns))
        print(df.head())
    sys.exit(1)


def build_wide(df, cfg):
    c = cfg["columns"]
    fmt = cfg["input"]["format"]
    for key in ("mo_id", "month"):
        if not c.get(key) or c[key] not in df.columns:
            fail(f"в configs/features.yaml не задана или не найдена колонка columns.{key}", df)
    df = df.copy()
    df["_mo"] = df[c["mo_id"]]
    df["_month"] = to_month(df[c["month"]])

    if fmt == "long":
        for key in ("category", "value"):
            if not c.get(key) or c[key] not in df.columns:
                fail(f"в configs/features.yaml не задана или не найдена колонка columns.{key}", df)
        excl = set(cfg.get("exclude_categories") or [])
        df = df[~df[c["category"]].isin(excl)]
        wide = df.pivot_table(index=["_mo", "_month"], columns=c["category"],
                              values=c["value"], aggfunc="sum")
    elif fmt == "wide":
        cats = [x for x in c.get("wide_categories") or [] if x not in (cfg.get("exclude_categories") or [])]
        if not cats or any(x not in df.columns for x in cats):
            fail("columns.wide_categories пуст или содержит несуществующие колонки", df)
        wide = df.groupby(["_mo", "_month"])[cats].sum(min_count=1)
    else:
        fail("input.format должен быть long или wide")
    wide.columns = [str(x) for x in wide.columns]
    return wide


def main():
    cfg = yaml.safe_load(open(CFG_PATH, encoding="utf-8"))
    audit = []

    df = load_table(cfg["input"]["path"])
    audit.append(f"Исходный файл: {cfg['input']['path']}, строк: {len(df)}, колонок: {df.shape[1]}")
    wide = build_wide(df, cfg)

    months = sorted(wide.index.get_level_values("_month").unique())
    n_months = len(months)
    n_mo_all = wide.index.get_level_values("_mo").nunique()
    audit.append(f"МО в данных: {n_mo_all}, месяцев: {n_months} ({months[0]} .. {months[-1]})")
    audit.append(f"Категории трат ({wide.shape[1]}): {', '.join(wide.columns)}")
    audit.append(f"Доля пропусков по категориям: {wide.isna().mean().round(3).to_dict()}")

    # 1. Фильтр по полноте
    cov = wide.groupby(level="_mo").size()
    keep = cov[cov >= cfg["filters"]["min_month_coverage"] * n_months].index
    audit.append(f"Порог полноты {cfg['filters']['min_month_coverage']}: оставлено {len(keep)} из {n_mo_all} МО")
    wide = wide[wide.index.get_level_values("_mo").isin(keep)]

    # 2. Полная сетка МО x месяц, интерполяция коротких пропусков
    grid = pd.MultiIndex.from_product([keep, months], names=["_mo", "_month"])
    wide = wide.reindex(grid)
    wide = wide.where(wide.sum(axis=1, min_count=1) > 0)  # нулевые итоги считаем пропуском
    gap = cfg["filters"]["max_interp_gap"]
    wide = wide.groupby(level="_mo", group_keys=False).apply(
        lambda g: g.interpolate(limit=gap, limit_area="inside"))
    bad = wide.isna().any(axis=1).groupby(level="_mo").any()
    drop_mo = bad[bad].index
    audit.append(f"Исключено МО из-за длинных пропусков после интерполяции: {len(drop_mo)}")
    wide = wide[~wide.index.get_level_values("_mo").isin(drop_mo)]

    # 3. Признаки
    tc = cfg.get("total_category")
    if tc:
        if tc not in wide.columns:
            fail(f"total_category '{tc}' не найдена среди категорий: {list(wide.columns)}")
        total = wide[tc]
        cats = wide.drop(columns=tc)
        shares = cats.div(total, axis=0).add_suffix("_share")
        shares["Прочее_share"] = (1 - shares.sum(axis=1)).clip(lower=0)
        audit.append(f"Итог берётся из категории '{tc}'; перечисленные категории дают в среднем "
                     f"{(cats.sum(axis=1) / total).mean():.1%} итога, остаток записан как 'Прочее_share'")
    else:
        total = wide.sum(axis=1)
        shares = wide.div(total, axis=0).add_suffix("_share")
    feat = shares.copy()
    feat["log_total"] = np.log1p(total)

    # 4. Дополнительные месячные признаки
    for ex in cfg.get("extra") or []:
        e = load_table(ex["path"])
        e["_mo"] = e[ex["mo_id"]]
        e["_month"] = to_month(e[ex["month"]])
        e = e.groupby(["_mo", "_month"])[ex["columns"]].mean()
        feat = feat.join(e, how="left")
        audit.append(f"Добавлены признаки {ex['columns']} из {ex['path']}, "
                     f"пропусков: {feat[ex['columns']].isna().mean().round(3).to_dict()}")
        feat[ex["columns"]] = feat.groupby(level="_mo")[ex["columns"]].transform(
            lambda s: s.interpolate(limit=gap, limit_area="inside"))
    for ex in cfg.get("static") or []:
        e = load_table(ex["path"]).set_index(ex["mo_id"])[ex["columns"]]
        e.index.name = "_mo"
        if ex.get("log"):
            e = np.log(e)
        n_before = feat.index.get_level_values("_mo").nunique()
        feat = feat.join(e, on="_mo", how="left")
        miss = feat[ex["columns"]].isna().any(axis=1).groupby(level="_mo").any().sum()
        audit.append(f"Добавлены постоянные признаки {ex['columns']} из {ex['path']}; "
                     f"МО без этих данных (исключены): {int(miss)} из {n_before}")
    feat = feat.dropna()

    # 5. Стандартизация внутри месяца
    cols = list(feat.columns)
    if cfg.get("standardize") == "per_month":
        g = feat.groupby(level="_month")[cols]
        feat[cols] = (feat[cols] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        feat[cols] = feat[cols].fillna(0.0)

    feat = feat.reset_index().rename(columns={"_mo": "mo_id", "_month": "month"})
    feat["month"] = feat["month"].dt.to_timestamp()

    # 6. Проверки и сохранение
    per_month = feat.groupby("month")["mo_id"].nunique()
    audit.append(f"Итог: {feat['mo_id'].nunique()} МО x {feat['month'].nunique()} месяцев, "
                 f"строк {len(feat)}, признаков {len(cols)}")
    audit.append(f"МО в месяц: min {per_month.min()}, max {per_month.max()}")
    audit.append(f"NaN: {int(feat[cols].isna().sum().sum())}, inf: {int(np.isinf(feat[cols]).sum().sum())}")

    out = Path(cfg["output"]["features"])
    out.parent.mkdir(parents=True, exist_ok=True)
    feat.to_parquet(out, index=False)
    aud = Path(cfg["output"]["audit"])
    aud.parent.mkdir(parents=True, exist_ok=True)
    aud.write_text("# Аудит данных\n\n" + "\n".join(f"- {a}" for a in audit) + "\n", encoding="utf-8")
    print("\n".join(audit))
    print(f"\nГотово: {out}")


if __name__ == "__main__":
    main()
