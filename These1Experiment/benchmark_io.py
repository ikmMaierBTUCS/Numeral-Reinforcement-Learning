"""
Benchmark: Wie viel Zeit kostet das Datei-Schreiben wirklich?
"""
import time

# Simuliere das data Dictionary nach mehreren Sprachen
data = {}
for i in range(50):  # 50 Sprachen simuliert
    data[f'Language_{i}'] = [[[j, 0] for j in range(0, 100, 10)]]

print(f"Dictionary size: {len(str(data))} characters")
print(f"Number of languages: {len(data)}")

# Benchmark: Wie lange dauert str(data)?
start = time.time()
for _ in range(100):
    s = str(data)
duration_str = time.time() - start
print(f"\n100x str(data): {duration_str:.3f}s = {duration_str*10:.1f}ms per call")

# Benchmark: Wie lange dauert Datei schreiben?
start = time.time()
for _ in range(100):
    with open(r"benchmark_test.txt", 'w') as f:
        f.write(str(data))
duration_write = time.time() - start
print(f"100x file write: {duration_write:.3f}s = {duration_write*10:.1f}ms per call")

# Hochrechnung für reales Szenario
print(f"\n=== Hochrechnung für 278 Sprachen, 10 Schritte ===")
print(f"Aktuell (2780 Schreiboperationen): {2780 * duration_write/100:.1f}s = {2780 * duration_write/100/60:.1f} Minuten")
print(f"Optimiert (278 Schreiboperationen): {278 * duration_write/100:.1f}s = {278 * duration_write/100/60:.1f} Minuten")
print(f"Ersparnis: {(2780-278) * duration_write/100:.1f}s = {(2780-278) * duration_write/100/60:.1f} Minuten")

# Cleanup
import os
if os.path.exists("benchmark_test.txt"):
    os.remove("benchmark_test.txt")
