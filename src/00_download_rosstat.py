"""Загрузка трёх показателей Росстата без скачивания многогигабайтных архивов.

Запуск из корня проекта:  python src/00_download_rosstat.py
Конфиг: configs/rosstat.yaml. Результат: data/raw/rosstat/<код показателя>.csv

Архив на сервере читается по частям (HTTP-запросы с заголовком Range): сначала оглавление архива,
затем только нужные файлы. Из каждого файла сохраняются строки за годы из конфига по муниципальным
образованиям верхнего уровня, без служебных колонок.
"""
import csv
import io
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import yaml

CFG = yaml.safe_load(open("configs/rosstat.yaml", encoding="utf-8"))
BLOCK = 4 * 1024 * 1024       # размер одного запроса, 4 МБ
UA = {"User-Agent": "Mozilla/5.0 (research data download)"}


class HttpRangeFile(io.RawIOBase):
    """Файл на сервере, из которого можно читать произвольный участок."""

    def __init__(self, url):
        self.url, self.pos, self.cache, self.loaded = url, 0, {}, 0
        req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=60) as r:
            cr = r.headers.get("Content-Range")
            if r.status != 206 or not cr:
                raise OSError("сервер не поддерживает чтение по частям")
            self.size = int(cr.split("/")[-1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            lo, hi = i * BLOCK, min((i + 1) * BLOCK, self.size) - 1
            for attempt in range(4):
                try:
                    req = urllib.request.Request(self.url, headers={**UA, "Range": f"bytes={lo}-{hi}"})
                    with urllib.request.urlopen(req, timeout=120) as r:
                        data = r.read()
                    break
                except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                    if attempt == 3:
                        raise
                    print(f"  сбой сети ({e}), повтор...", flush=True)
            if len(self.cache) >= 6:                       # держим в памяти несколько последних блоков
                self.cache.pop(next(iter(self.cache)))
            self.cache[i] = data
            self.loaded += len(data)
        return self.cache[i]

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        blk = self._block(self.pos // BLOCK)
        start = self.pos % BLOCK
        n = min(len(b), len(blk) - start)
        b[:n] = blk[start:start + n]
        self.pos += n
        return n


def open_archive(urls):
    for url in urls:
        try:
            f = HttpRangeFile(url)
            zf = zipfile.ZipFile(io.BufferedReader(f, buffer_size=1024 * 1024))
            print(f"архив найден: {url}\n  размер {f.size / 1e9:.2f} ГБ, файлов внутри: {len(zf.namelist())}", flush=True)
            return f, zf
        except Exception as e:
            print(f"не подошёл: {url}\n  причина: {e}", flush=True)
    return None, None


def main():
    out = Path(CFG["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    years = {str(y) for y in CFG["years"]}
    ycol = CFG["year_column"]
    lcol, lsub = CFG.get("level_column"), CFG.get("level_contains")
    drop = set(CFG.get("drop_columns") or [])
    by_arch = {}
    for code, arch in CFG["indicators"].items():
        by_arch.setdefault(arch, []).append(code)
    failed = []
    for arch, codes in by_arch.items():
        print(f"\n=== {arch} ===", flush=True)
        f, zf = open_archive(CFG["archives"][arch])
        if zf is None:
            failed += codes
            continue
        for code in codes:
            names = [n for n in zf.namelist() if code in n and n.lower().endswith(".csv")]
            if not names:
                print(f"{code}: файл в архиве не найден. Примеры имён: {zf.namelist()[:5]}", flush=True)
                failed.append(code)
                continue
            info = zf.getinfo(names[0])
            print(f"{code}: {names[0]} (в архиве {info.compress_size / 1e6:.0f} МБ, "
                  f"распакованный {info.file_size / 1e6:.0f} МБ)", flush=True)
            kept = total = 0
            with zf.open(info) as raw, open(out / f"{code}.csv", "w", encoding="utf-8", newline="") as dst:
                rd = csv.reader(io.TextIOWrapper(raw, encoding="utf-8-sig", newline=""), delimiter=";")
                wr = csv.writer(dst, delimiter=";")
                header = next(rd)
                keep = [i for i, c in enumerate(header) if c not in drop]
                wr.writerow([header[i] for i in keep])
                print(f"  колонки: {[header[i] for i in keep]}", flush=True)
                yi = header.index(ycol) if ycol in header else None
                li = header.index(lcol) if lcol in header else None
                if yi is None:
                    print(f"  колонки '{ycol}' нет, отбор по годам не выполняется", flush=True)
                for row in rd:
                    total += 1
                    if (yi is None or row[yi] in years) and (li is None or lsub in row[li]):
                        wr.writerow([row[i] for i in keep])
                        kept += 1
                    if total % 2_000_000 == 0:
                        print(f"  прочитано {total / 1e6:.0f} млн строк, сохранено {kept}", flush=True)
            print(f"  готово: {out / (code + '.csv')}, строк {kept} из {total}", flush=True)
        print(f"загружено из сети по этому архиву: {f.loaded / 1e6:.0f} МБ", flush=True)
    if failed:
        sys.exit(f"\nНе получены показатели: {', '.join(failed)}. Проверь адреса архивов в configs/rosstat.yaml.")
    print("\nВсе показатели сохранены в", out)


if __name__ == "__main__":
    main()
