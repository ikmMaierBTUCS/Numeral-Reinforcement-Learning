from funktionen import *

lex = create_lexicon('Armenian', set_limit=30)
for i, e in enumerate(lex):
    if i < 10:
        print(f'{i+1}: "{e.word}" (length={len(e.word)})')
