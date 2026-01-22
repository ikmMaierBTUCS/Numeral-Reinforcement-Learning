"""
Optimierte Version von neu-these1experiment.py
Hauptoptimierungen:
1. Weniger Datei-I/O (nur nach jeder Sprache, nicht nach jedem Schritt)
2. printb=False für weniger Konsolen-Output
3. Frühzeitiges Abbrechen wenn alle Numerale erkannt
4. PARALLELISIERUNG aktiviert (4 Kerne)
"""
from funktionen import *
from multiprocessing import Pool  # Für Parallelisierung

langnumb=pd.read_csv(r'Numeral.csv', encoding = "utf_16", sep = '\t')
package_languagesandnumbers=set([langnumb.iloc[i,0] for i in range(len(langnumb))])
package_languagesandnumbers.remove('Malecite-Passamaquoddy')
package_languagesandnumbers.remove('Language')
package_languagesandnumbers.remove('Armenian')
package_languagesandnumbers = list(package_languagesandnumbers)
package_num2words=['fr','fi','fr_CH','fr_BE','fr_DZ','he','id','it','ja','kn','ko','lt','lv','no','pl','pt','pt_BR','sl','sr','ro','ru','tr','th','vi','nl','uk','es_CO','es','es_VE','cs','de','ar','da','en_GB','en_IN']

sprachen =package_num2words+ package_languagesandnumbers + ['Birom','Yoruba']
sprachen = sorted(sprachen)

step_width = 10
learning_range = 1000

data = {}

def process_language(language):
    """Prozessiert eine einzelne Sprache - kann parallelisiert werden"""
    total_lexicon = create_lexicon(language, set_limit = learning_range)
    if len(total_lexicon) == 0:
        return language, []
    
    knowledge = {v.mapping[-1]:v.root for v in total_lexicon}
    orakel = Oracle('omniscient', knowledge = list(knowledge.values()))
    language_results = []
    
    for i in range(1):
        run_results = []
        # GET LEXICON
        total_lexicon = create_lexicon(language, set_limit = learning_range)
        total_lexicon = shuffle_lexicon(total_lexicon)
        
        print(f"{language} {i} [starting with {len(total_lexicon)} numerals]")
        learning_progress = []
        recognized_numerals = []
        learner_lex = []
        
        for training_size in range(len(knowledge) + 1):
            if training_size % step_width == 0 or training_size == learning_range - 1:
                # cut lexicon
                lexicon = total_lexicon[training_size - step_width : training_size]
            
                # learn (MIT printb=False für Geschwindigkeit!)
                learner_lex = learn_lexicon(lexicon, orakel, initial_lexicon=learner_lex, printb=False, normalize=True)
                
                # evaluate
                unlearned_numbers = []
                errors = 0
                lex = sorted(learner_lex, key=lambda x:x.root.count('_'))
                
                for n in range(1, len(knowledge) + 1):
                    if n not in recognized_numerals:
                        word = knowledge[n]
                        number = grammar_parse(word, lex)
                        if number != n:
                            if number == -1:
                                unlearned_numbers += [n]
                            else:
                                print(f'Bad Error: Interpreted {word} as {number} instead of {n}.')
                                errors += 1
                        else:
                            recognized_numerals += [n]
                
                learning_progress += [[len(recognized_numerals), errors]]
                run_results += [[len(recognized_numerals), errors]]
                
                # Frühzeitiges Abbrechen wenn alle Numerale erkannt
                if len(recognized_numerals) == len(knowledge):
                    print(f"{language}: All {len(knowledge)} numerals recognized at step {training_size}")
                    break
        
        language_results += [run_results]
    
    return language, language_results

# SEQUENZIELLE Version (auskommentiert - für Debugging)
# for language in sprachen:
#     lang, results = process_language(language)
#     data[lang] = results
#     # Nur EINMAL pro Sprache in Datei schreiben
#     with open(r"lernverlauf.txt", 'w') as f:
#         f.write(str(data))
#     print(f"Completed: {language} - Progress: {len(data)}/{len(sprachen)}")

# PARALLELE Version (AKTIVIERT - 4 Kerne)
if __name__ == '__main__':
    print(f"Starting parallel processing with 4 processes for {len(sprachen)} languages...")
    with Pool(processes=4) as pool:  # 4 parallele Prozesse
        results = pool.map(process_language, sprachen)
    
    # Ergebnisse sammeln
    for lang, lang_results in results:
        data[lang] = lang_results
    
    # Finale Datei schreiben
    with open(r"lernverlauf_parallelisiert.txt", 'w') as f:
        f.write(str(data))
    
    print("\n=== FINISHED ===")
    print(f"Total languages processed: {len(data)}")
