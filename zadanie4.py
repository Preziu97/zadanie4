import numpy as np
import pandas as pd
from pymcdm.methods import TOPSIS, SPOTIS

# 1. Macierz decyzyjna: 5 alternatyw, 4 kryteria (koszt, zysk, czas, ryzyko)
decision_matrix = np.array([
    [100000, 50000, 24, 0.3],
    [120000, 70000, 36, 0.4],
    [110000, 60000, 30, 0.2],
    [95000,  45000, 20, 0.5],
    [105000, 65000, 28, 0.35]
])

# Nazwy kryteriów
criteria_names = ['Koszt', 'Zysk', 'Czas', 'Ryzyko']

# 2. Wagi kryteriów (suma = 1)
weights = np.array([0.3, 0.4, 0.2, 0.1])

# 3. Typy kryteriów: -1 = minimalizowane, 1 = maksymalizowane
criteria_types = np.array([-1, 1, -1, -1])

# 4. Funkcja do ręcznej min-max normalizacji
def manual_minmax_normalization(matrix, types):
    norm_matrix = np.zeros_like(matrix, dtype=float)
    for j in range(matrix.shape[1]):
        col = matrix[:, j]
        min_val = col.min()
        max_val = col.max()
        if max_val - min_val == 0:
            norm_matrix[:, j] = 0
        elif types[j] == 1:
            norm_matrix[:, j] = (col - min_val) / (max_val - min_val)
        else:
            norm_matrix[:, j] = (max_val - col) / (max_val - min_val)
    return norm_matrix

# 5. Normalizacja
norm_matrix = manual_minmax_normalization(decision_matrix, criteria_types)

# 6. TOPSIS
topsis = TOPSIS()
topsis_scores = topsis(norm_matrix, weights, criteria_types)

# 7. SPOTIS
bounds = np.array([
    [90000, 130000],    # koszt
    [40000, 80000],     # zysk
    [18, 40],           # czas
    [0.1, 0.6]          # ryzyko
])
spotis = SPOTIS(bounds)
spotis_scores = spotis(decision_matrix, weights, criteria_types)

# 8. Tworzenie DataFrame z wynikami
alternatives = ['A1', 'A2', 'A3', 'A4', 'A5']
results = pd.DataFrame({
    'Alternatywa': alternatives,
    'TOPSIS_score': topsis_scores,
    'SPOTIS_score': spotis_scores
})
results['Ranga_TOPSIS'] = results['TOPSIS_score'].rank(ascending=False).astype(int)
results['Ranga_SPOTIS'] = results['SPOTIS_score'].rank(ascending=True).astype(int)

print("=== KRYTERIA ===")
for name, typ, w in zip(criteria_names, criteria_types, weights):
    typ_text = "maksymalizowane" if typ == 1 else "minimalizowane"
    print(f"- {name}: {typ_text}, waga = {w}")
print()

# 9. Wyświetlenie wyników
print("=== WYNIKI ANALIZY MCDM ===\n")
print(results)
