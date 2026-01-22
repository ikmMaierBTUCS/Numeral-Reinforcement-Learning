import pandas as pd

df = pd.read_csv('Numeral.csv', encoding='utf_16', sep='\t')

# Find languages with spaces in numerals (biscriptual candidates)
biscriptual_langs = set()
for lang in df['Language'].unique():
    lang_df = df[df['Language'] == lang]
    if lang_df.iloc[0, 2] and ' ' in str(lang_df.iloc[0, 2]):
        biscriptual_langs.add(lang)

print(f"Potentially biscriptual languages: {sorted(biscriptual_langs)}\n")

# Check samples from each
for lang in sorted(biscriptual_langs):
    lang_df = df[df['Language'] == lang]
    print(f"{lang}:")
    for i in [0, 1, 19, 20]:
        if i < len(lang_df):
            num = lang_df.iloc[i, 1]
            numeral = lang_df.iloc[i, 2]
            print(f"  {num}: {repr(numeral)}")
    print()
