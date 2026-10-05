# Типы локальных экономик России

Динамическая кластеризация муниципалитетов на атрибутированной сети по данным СберИндекса о безналичных расходах, январь 2023 – декабрь 2024.

Онлайн-конкурс СберИндекса 2026, направление «Кластеризация».

Авторы: Даминев Марат УГНТУ, Атнагулов Тимур УГНТУ

- Интерактивный лендинг: https://maratdaminev.github.io/sberindex-types/ (локально: `site/index.html`)
- Методологический отчёт: [reports/methodology.md](reports/methodology.md)

## Что сделано

1. Для 2004 муниципальных образований (МО) и 24 месяцев собрано 8 признаков: шесть долей безналичных трат, уровень трат и доступность рынков.
2. Для каждого месяца построена сеть МО по четырём правилам: косинусное сходство признаков, синхронность изменений признаков, лаговая корреляция изменений, расстояние по автодорогам.
3. Сравнены четыре метода кластеризации (k-средних, Лувен, спектральная кластеризация, метод Уорда с ограничением по сети) по индексам SW, CH, S_Dbw, AVI, AVU, MQ и модулярности.
4. Изменения кластеров во времени отслеживаются эволюционной спектральной кластеризацией: сеть месяца сглаживается сетью прошлых месяцев, кластеры соседних месяцев сопоставляются венгерским алгоритмом.
5. Получено шесть типов локальных экономик; для каждого посчитаны профиль, состав, типичные и пограничные МО.
6. Типология проверена на устойчивость (подвыборки, параметры, удаление признаков) и сопоставлена с данными Росстата о зарплатах, населении и структуре занятости.
7. Результаты показаны на интерактивном лендинге: карта типов по месяцам, карточки МО с экономическими двойниками, динамика и переходы.

## Данные

Исходные данные в репозиторий не входят. Скачайте их и распакуйте в `data/raw`:

| Что | Откуда | Куда положить |
|---|---|---|
| Набор данных хакатона: `consumption.parquet`, `market_access.parquet`, `connection.parquet` | [архив по ссылке со страницы конкурса](https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip) | `data/raw/hackathonlicence/` |
| Справочник МО: `t_dict_municipal_districts.xlsx` | [страница набора данных СберИндекса](https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities) | `data/raw/` |

Данные СберИндекса распространяются по лицензии CC BY-SA 4.0. Цитирование:

- Потребительские безналичные расходы на уровне муниципальных образований по категориям трат. СберИндекс. Данные доступны по адресу https://sberindex.ru/ru/research/data-sense-opisanie-nabora-dannikh-khakatona-sberindeksa-po-munitsipalnim-dannim (данные скачаны 04.10.2026).
- Индекс доступности рынков на уровне муниципальных образований. СберИндекс. Данные доступны по тому же адресу (данные скачаны 04.10.2026).
- Автодорожные и железнодорожные связи между муниципальными образованиями. СберИндекс. Данные доступны по тому же адресу (данные скачаны 04.10.2026).

Для внешней проверки типов используются три показателя БДПМО Росстата (зарплата, численность работников, население) в обработке проекта «Если быть точным»: «Муниципальная статистика России с 2005 года» // Росстат; обработка: «Если быть точным», https://tochno.st/datasets/bdmo. Шаг 10 читает их из архива на сервере, не скачивая архив целиком (около 60 МБ трафика). Шаги 10 и 11 запускаются отдельно и в `run_all.py` не входят.

Результаты в `reports/` и `site/` получены из этих данных и распространяются на тех же условиях (CC BY-SA 4.0).

## Запуск

Проверено на Python 3.13.

```
pip install -r requirements.txt
python src/run_all.py
```

Полный пересчёт занимает около 20 минут на обычном ноутбуке, из них около 10 минут идут проверки устойчивости (шаг 9). Шаги можно запускать и по отдельности, из корня проекта:

| Шаг | Команда | Конфиг | Результат |
|---|---|---|---|
| 1. Признаки | `python src/01_features.py` | `configs/features.yaml` | `data/processed/features.parquet`, `reports/data_audit.md` |
| 2. Сеть | `python src/02_network.py` | `configs/network.yaml` | `data/processed/network/`, `reports/network_summary.md` |
| 3–4. Кластеризация и ICVI | `python src/03_cluster.py` | `configs/cluster.yaml` | `data/processed/clusters/`, `reports/icvi_comparison.md` |
| 5. Динамика | `python src/04_dynamics.py` | `configs/dynamics.yaml` | `data/processed/dynamics/`, `reports/dynamics.md` |
| 6. Интерпретация | `python src/05_interpret.py` | `configs/interpret.yaml` | `reports/interpretation.md`, `data/processed/dynamics/mo_types.csv` |
| 7. Лендинг | `python src/06_landing.py` | `configs/landing.yaml` | `site/index.html` |
| 8. Диагностика скачков | `python src/07_jump_diagnostics.py` | `configs/dynamics.yaml` | `reports/jump_diagnostics.md` |
| 9. Проверки устойчивости | `python src/08_robustness.py` | `configs/robustness.yaml` | `reports/robustness.md` |
| 10. Загрузка данных Росстата | `python src/00_download_rosstat.py` | `configs/rosstat.yaml` | `data/raw/rosstat/` |
| 11. Внешняя проверка типов | `python src/09_rosstat_validation.py` | `configs/rosstat.yaml` | `reports/rosstat_validation.md` |

Все гиперпараметры вынесены в YAML-файлы в `configs/`: число соседей, правила построения рёбер, число кластеров, вес сглаживания, начальное значение генератора случайных чисел.

## Структура репозитория

```
configs/    настройки шагов (YAML)
src/        скрипты шагов, icvi.py с индексами качества, шаблон лендинга
reports/    отчёты, которые пишут скрипты, и методологический отчёт
site/       собранный лендинг
data/       исходные и промежуточные данные (в репозиторий не входят)
```

## Воспроизводимость

Начальное значение генератора случайных чисел зафиксировано (`seed: 42`). При равном сходстве соседей выбирается МО с меньшим номером, поэтому сети строятся одинаково на разных компьютерах. Числа в отчётах могут незначительно отличаться между версиями библиотек.

## Источники методов

- Chi Y., Song X., Zhou D., Hino K., Tseng B. Evolutionary spectral clustering by incorporating temporal smoothness. KDD, 2007.
- Blondel V. et al. Fast unfolding of communities in large networks. J. Stat. Mech., 2008.
- Halkidi M., Vazirgiannis M. Clustering validity assessment: finding the optimal partitioning of a data set. ICDM, 2001.
- Biswas A., Biswas B. Defining quality metrics for graph clustering evaluation. Expert Systems with Applications, 2017.
- Mancoridis S. et al. Using automatic clustering to produce high-level system organizations of source code. IWPC, 1998.
