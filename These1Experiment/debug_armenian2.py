import pandas as pd

df = pd.read_csv('Numeral.csv', encoding='utf_16', sep='\t')
arm = df[df['Language']=='Armenian']

for i in [2, 5, 6, 7]:
    raw = arm.iloc[i,2]
    parts = raw.split(' ')
    armenian = parts[0]
    print(f"Row {i+1}:")
    print(f"  Full: {repr(raw)}")
    print(f"  Armenian: {repr(armenian)}")
    print(f"  Armenian bytes: {armenian.encode('utf-8').hex()}")
    print(f"  Length: {len(armenian)}")
    print()
