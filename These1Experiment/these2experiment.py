from funktionen import *

langnumb=pd.read_csv(r'Numeral.csv', encoding = "utf_16", sep = '\t')
package_languagesandnumbers=set([langnumb.iloc[i,0] for i in range(len(langnumb))])
#print(package_languagesandnumbers)
package_languagesandnumbers.remove('Malecite-Passamaquoddy')
package_languagesandnumbers.remove('Language')
package_languagesandnumbers.remove('Armenian')
package_languagesandnumbers = list(package_languagesandnumbers)
package_num2words=['fr','fi','fr_CH','fr_BE','fr_DZ','he','id','it','ja','kn','ko','lt','lv','no','pl','pt','pt_BR','sl','sr','ro','ru','tr','th','vi','nl','uk','es_CO','es','es_VE','cs','de','ar','da','en_GB','en_IN']

sprachen =package_num2words+ package_languagesandnumbers + ['Birom','Yoruba']
sprachen = sorted(sprachen)


step_width = 39
learning_range = 40

data = {}
#['Kutenai',,'Purepecha''Tongan-Telephone-Style','Yupik',
for language in ['de']: # sprachen:
    total_lexicon = create_lexicon(language, set_limit = learning_range)
    if not language in list(data.keys()): # and len(total_lexicon) > 98: #True: #len(total_lexicon) > 998 and not language in list(data.keys()):
        knowledge = {v.mapping[-1]:v.root for v in total_lexicon}
        orakel = Oracle('arithmetic')
        data[language] = []
        
        for i in range(1):
            data[language] += [[]]
            # GET LEXICON
            total_lexicon = create_lexicon(language, set_limit = learning_range)
            total_lexicon = shuffle_lexicon(total_lexicon)
            
            # [2, 1, 39, 4, 10, 3, 6, 30, 8, 12, 36, 7, 43, 9, 16, 11, 5, 28, 21, 13, 15, 34, 60, 20, 18, 84, 24, 14, 96, 48, 22, 29, 61, 19, 72, 54, 38, 67, 62, 93, 45, 23, 27, 77, 25, 46, 73, 37, 66, 32, 98, 53, 49, 47, 33, 64, 70, 90, 69, 40, 44, 58, 17, 92, 55, 31, 41, 65, 26, 75, 42, 35, 97, 50, 91, 94, 82, 57, 89, 71, 56, 85, 68, 76, 83, 51, 81, 86, 88, 79, 74, 52, 63, 59, 99, 95, 80, 87, 78]
            #total_lexicon = [Vocabulary(i,num2words(i,lang='he')) for i in [1, 2, 3, 9, 6, 10, 4, 20, 27, 12, 7, 25, 8, 5, 26, 22, 11, 71, 60, 23, 30, 93, 59, 40, 14, 13, 19, 18, 56, 37, 43, 17, 51, 91, 28, 34, 15, 36, 21, 35, 67, 29, 32, 47, 62, 16, 39, 55, 50, 94, 70, 31, 90, 65, 75, 68, 97, 82, 63, 72, 89, 80, 42, 46, 38, 48, 64, 45, 53, 24, 54, 85, 83, 33, 74, 44, 84, 66, 77, 49, 73, 41, 81, 95, 79, 57, 52, 98, 76, 78, 61, 87, 92, 69, 58, 96, 86, 88, 99]]
            
            #print(language,i)
            print(language,i,[e.number for e in total_lexicon])
            learning_progress = []
            recognized_numerals = []
            learner_lex = []
            for training_size in range(len(knowledge) + 1):
                if training_size % step_width == 0 or training_size == learning_range - 1:
                    # print('training size = ', training_size)
                
                    # cut lexicon
                    lexicon = total_lexicon[training_size - step_width : training_size]
                
                    # learn
                    learner_lex = learn_lexicon(lexicon,orakel,initial_lexicon=learner_lex,printb=True,normalize=True)
                    
                    # evaluate
                    unlearned_numbers = []
                    errors = 0
                    lex = sorted(learner_lex,key=lambda x:x.root.count('_'))
                    for n in range(1,len(knowledge) + 1):
                        if n not in recognized_numerals:
                            word = knowledge[n]
                            number = grammar_parse(word,lex)
                            if number != n:
                                if number == -1:
                                    unlearned_numbers += [n]
                                    pass
                                    # print('Numeral ' + word + ' unknown for learned lexicon.')
                                else:
                                    print('Bad Error: Interpreted ' + word + ' as ' + str(number) + ' instead of ' + str(n) + '.')
                                    errors += 1
                            else:
                                recognized_numerals += [n]
                                
                    #print('unlearned_numbers:',unlearned_numbers)
                    #print('training size, recognized numerals = ',training_size,len(recognized_numerals))
                    learning_progress += [[len(recognized_numerals),errors]]
                    data[language][-1] += [[len(recognized_numerals),errors]]
                    with open(r"lernverlauf.txt", 'w') as f:
                        f.write(str(data))
                    if len(recognized_numerals) == learning_range - 1:
                        #print('Latest learnt numeral: ',recognized_numerals[-1])
                        break
            #print(end='\n')
            #data[language] += [learning_progress]
            #points = [(step_width*i,learning_progress[i]) for i in range(len(learning_progress))]
            #print(points)
            #plt.scatter([p[0] for p in points], [p[1] for p in points])
            #plt.xlabel("trained numerals")
            #plt.ylabel("understood numerals")
            #plt.show()
            with open(r"lernverlauf.txt", 'w') as f:
                f.write(str(data))

