"""Полный пересчёт проекта одной командой:  python src/run_all.py

Запускает шаги по порядку и останавливается на первом, который завершился с ошибкой.
"""
import subprocess
import sys
import time

STEPS = ["01_features.py", "02_network.py", "03_cluster.py", "04_dynamics.py", "05_interpret.py", "06_landing.py", "07_jump_diagnostics.py", "08_robustness.py"]

for step in STEPS:
    print(f"\n===== {step} =====", flush=True)
    t0 = time.time()
    code = subprocess.run([sys.executable, f"src/{step}"]).returncode
    if code != 0:
        sys.exit(f"Шаг {step} завершился с ошибкой (код {code}). Дальнейшие шаги не запускались.")
    print(f"----- {step}: {time.time() - t0:.0f} с", flush=True)
print("\nВсе шаги выполнены. Лендинг: site/index.html, отчёты: reports/")
