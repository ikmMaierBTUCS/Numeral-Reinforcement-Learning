from funktionen import *
import pandas as pd

df = pd.read_csv('Numeral.csv', encoding='utf_16', sep='\t')
arm = df[df['Language']=='Armenian']

for i in [2, 5, 6, 7]:
    raw = arm.iloc[i,2]
    parts = raw.split(' ')
    armenian = parts[0]
    number = i + 1
    print(f"Creating Vocabulary({number}, {repr(armenian)})")
    voc = Vocabulary(number, armenian)
    print(f"  Result: {repr(voc.word)}")
    print(f"  Length: {len(voc.word)}")
    print()
