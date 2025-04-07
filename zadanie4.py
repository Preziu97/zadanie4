import numpy as np
import pandas as pd
from pymcdm.methods import TOPSIS, SPOTIS
from pymcdm.helpers import normalize_matrix, rankdata, rrankdata

# Macierz decyzyjna: 5 alternatyw, 4 kryteria (koszt, zysk, czas, ryzyko)
decision_matrix = np.array([
    [100000, 50000, 24, 0.3],
    [120000, 70000, 36, 0.4],
    [110000, 60000, 30, 0.2],
    [95000,  45000, 20, 0.5],
    [105000, 65000, 28, 0.35]
])

# Nazwy kryteriów
criteria_names = ['Koszt', 'Zysk', 'Czas', 'Ryzyko']

# Wagi kryteriów (suma = 1)
weights = np.array([0.3, 0.4, 0.2, 0.1])

# Typy kryteriów: -1 = minimalizowane, 1 = maksymalizowane
criteria_types = np.array([-1, 1, -1, -1])

# Normalizacja danych
normalized = normalize_matrix(decision_matrix, 'minmax', criteria_types)
print("\n=== ZNORMALIZOWANA MACIERZ (MIN-MAX) ===\n")
print(normalized)

# TOPSIS
topsis = TOPSIS()
topsis_scores = topsis(decision_matrix, weights, criteria_types)

# SPOTIS (wymaga granic)
bounds = np.array([
    [90000, 130000],    # koszt
    [40000, 80000],     # zysk
    [18, 40],           # czas
    [0.1, 0.6]          # ryzyko
])
spotis = SPOTIS(bounds)
spotis_scores = spotis(decision_matrix, weights, criteria_types)

# Tworzenie DataFrame z wynikami
alternatives = ['A1', 'A2', 'A3', 'A4', 'A5']
results = pd.DataFrame({
    'Alternatywa': alternatives,
    'TOPSIS_score': topsis_scores,
    'SPOTIS_score': spotis_scores
})

# Tworzenie rankingów
results['Ranga_TOPSIS'] = rrankdata(results['TOPSIS_score']).astype(int)
results['Ranga_SPOTIS'] = rankdata(results['SPOTIS_score']).astype(int)

# Wyświetlenie informacji o kryteriach
print("\n=== KRYTERIA ===\n")
for name, typ, w in zip(criteria_names, criteria_types, weights):
    typ_text = "maksymalizowane" if typ == 1 else "minimalizowane"
    print(f"- {name}: {typ_text}, waga = {w}")
print()

# Wyświetlenie wyników
print("=== WYNIKI ANALIZY MCDM ===\n")
print(results)
