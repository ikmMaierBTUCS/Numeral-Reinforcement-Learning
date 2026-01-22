import pandas as pd
from funktionen import *

df = pd.read_csv('Numeral.csv', encoding='utf_16', sep='\t')
filtered = df[df['Language']=='Armenian']

for i in [2, 5, 6, 7]:
    numeral = filtered.iloc[i,2]
    print(f"Original: {repr(numeral)}")
    
    if numeral[0]==' ':
        numeral=numeral[1:]
    if numeral[-1]==' ':
        numeral=numeral[:-1]
    print(f"After trim: {repr(numeral)}")
    
    # My Armenian fix
    words=numeral.split(' ')
    numeral=words[0]
    print(f"After Armenian split: {repr(numeral)}")
    
    numeral=numeral.replace('%',',')
    print(f"After replace: {repr(numeral)}")
    
    voc=Vocabulary(i+1,numeral)
    print(f"Vocabulary word: {repr(voc.word)}")
    print()
