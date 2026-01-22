"""
Benchmark: Wie viel Zeit kostet das Printen in learn_lexicon()?
Simuliert die Print-Operationen die bei printb=True passieren
"""
import time
import sys

# Simuliere eine typische SCFunction.present() Ausgabe
def fake_present():
    print("Function _ tunc _     maps {1,2,3}x{10,20,30}    by (x,y)        -> 10x+1y+0")

def fake_learning_step_verbose():
    """Simuliert einen Lernschritt mit printb=True"""
    # Typische Ausgaben in learn_lexicon():
    print("Vereinige die folgenden Funktionen:")
    fake_present()
    fake_present()
    print("mergee lies in affine span of inputrange of merger")
    print("0 False")
    print("1 False")
    print("neue_dimension 2")
    print("affine_basis 1")
    # Insgesamt ca. 10-15 Print-Statements pro Merge-Operation
    # Bei mehreren Merges pro Schritt kann das noch mehr werden

def fake_learning_step_silent():
    """Simuliert einen Lernschritt mit printb=False (nur berechnen)"""
    # Keine Prints, nur dummy-Berechnungen
    _ = "Function _ tunc _     maps {1,2,3}x{10,20,30}    by (x,y)        -> 10x+1y+0"
    _ = 1 + 1

# Benchmark mit Prints
print("=== Benchmark: Mit Prints (printb=True) ===")
start = time.time()
for _ in range(100):  # 100 Lernschritte simuliert
    fake_learning_step_verbose()
duration_verbose = time.time() - start
print(f"\n100 Lernschritte MIT Prints: {duration_verbose:.3f}s = {duration_verbose*10:.1f}ms pro Schritt")

# Benchmark ohne Prints
print("\n=== Benchmark: Ohne Prints (printb=False) ===")
# Unterdrücke stdout
old_stdout = sys.stdout
sys.stdout = open('nul' if sys.platform == 'win32' else '/dev/null', 'w')

start = time.time()
for _ in range(100):
    fake_learning_step_silent()
duration_silent = time.time() - start

sys.stdout = old_stdout  # Stelle stdout wieder her
print(f"100 Lernschritte OHNE Prints: {duration_silent:.3f}s = {duration_silent*10:.1f}ms pro Schritt")

# Hochrechnung
print(f"\n=== Hochrechnung für reales Szenario ===")
print(f"Differenz pro Schritt: {(duration_verbose - duration_silent)*10:.1f}ms")
print(f"\nFür 278 Sprachen × 10 Schritte = 2780 Lernschritte:")
print(f"  MIT printb=True:  {2780 * duration_verbose / 100:.1f}s = {2780 * duration_verbose / 100 / 60:.1f} Minuten")
print(f"  MIT printb=False: {2780 * duration_silent / 100:.1f}s = {2780 * duration_silent / 100 / 60:.1f} Minuten")
print(f"  Ersparnis: {2780 * (duration_verbose - duration_silent) / 100:.1f}s = {2780 * (duration_verbose - duration_silent) / 100 / 60:.1f} Minuten")
print(f"  Speedup: {duration_verbose / duration_silent:.1f}x")
