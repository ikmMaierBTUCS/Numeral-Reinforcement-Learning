from funktionen import *

def create_lexicon(language,set_limit=10**9,lower_limit=0):
    LEX=[]
    try:
        num2words(1, lang=language)
        for integer in list(range(1,1001))+[1002,1006,1100,1200,1206,7000,7002,7006,7100,7200,7206,10000,17000,17200,17206,20000,27000,27006,27200,27206]:
            if integer < set_limit and lower_limit < integer:
                try:
                    numeral=num2words(integer, lang=language)
                    voc=Vocabulary(integer,numeral)
                    LEX=LEX+[voc]
                except:
                    pass
        return LEX
    except:
        try:
            #lanu=pd.read_csv(r'C:\Users\ikm\OneDrive\Desktop\NumeralParsingPerformance\Languages&NumbersData\Numeral.csv', encoding = "utf_16", sep = '\t')
            lanu=pd.read_csv(r'Numeral.csv', encoding = "utf_16", sep = '\t')
            df=lanu[lanu['Language']==language]
            biscriptual=False
            if ' ' in df.iloc[0,2]:
                biscriptual=True
            for i in range(len(df)):
                if i+1 < set_limit:
                    numeral=df.iloc[i,2]
                    if numeral[0]==' ':
                        numeral=numeral[1:]
                    if numeral[-1]==' ':
                        numeral=numeral[:-1]
                    if language in ['Latin','Persian','Arabic']:
                        words=numeral.split(' ')
                        numeral=' '.join(iter(words[:-1]))
                    if language in ['Chuvash','Adyghe','Belarusian','Ukrainian','Ingush']:
                        words=numeral.split(' ')
                        numeral=' '.join(iter(words[:len(words)//2]))
                    # For biscriptual languages: extract native script (not Latin transliteration)
                    elif biscriptual and not language in ['Latin','Persian','Arabic','Chuvash','Adyghe']:
                        ad = AlphabetDetector()
                        # Check if Latin comes first (like Japanese) or second (like Armenian, Russian)
                        first_is_latin = False
                        for char in numeral:
                            if char != ' ':
                                first_is_latin = ad.is_latin(char)
                                break
                        
                        if first_is_latin:
                            # Latin first, native script second (e.g., Japanese "ichi 一")
                            # Find first non-Latin, non-space character and take from there
                            for point in range(len(numeral)):
                                if numeral[point] != ' ' and not ad.is_latin(numeral[point]):
                                    numeral = numeral[point:].strip()
                                    break
                        else:
                            # Native script first, Latin second (e.g., Armenian "մեկ mek")
                            # Find first Latin character and take everything before it
                            for point in range(len(numeral)+1):
                                char = numeral[point:point+1]
                                if char and ad.is_latin(char) and char != ' ':
                                    numeral = numeral[:point].strip()
                                    break
                        
                        # Fallback to old delatinized approach if extraction didn't work
                        if numeral == df.iloc[i,2].strip() or not numeral:
                            numeral=delatinized(df.iloc[i,2].strip())
                    #print(numeral)
                    numeral=numeral.replace('%',',')
                    #print(numeral)
                    voc=Vocabulary(i+1,numeral)
                    LEX=LEX+[voc]
            return LEX
        except:
            #pass
            if True:
                lanu=pd.read_csv(r'DVNum.csv', encoding = "utf_8", sep = ';')
                #print(lanu)
                #print(lanu.columns)
                df=lanu[lanu['Language']==language]
                for i in range(len(df)):
                    if i+1 < set_limit:
                        numeral=df.iloc[i,2]
                        if numeral[0]==' ':
                            numeral=numeral[1:]
                        if numeral[-1]==' ':
                            numeral=numeral[:-1]
                        voc=Vocabulary(i+1,unicodedata.normalize('NFC',numeral))
                        LEX=LEX+[voc]
                for voc in LEX:
                    #pass
                    voc.printVoc()
                    print(len(voc.word))
                return LEX
            #except:
                #raise NotImplementedError("Language "+language+" is not supported or spelled differently")

lex = create_lexicon('Birom')
for e in lex:
    print(f"{e.number}: {repr(e.word)}")

lanu=pd.read_csv(r'DVNum.csv', encoding = "utf_8", sep = '\t')
all_languages = set(lanu['Language'].unique())
duplicates_found = False

for language in sorted(all_languages):
    try:
        lex = create_lexicon(language, set_limit=1000)
        
        # Create a mapping of word -> list of numbers
        word_to_numbers = {}
        for vocab in lex:
            word = vocab.word
            if word:  # Skip empty words
                if word not in word_to_numbers:
                    word_to_numbers[word] = []
                word_to_numbers[word].append(vocab.number)
        
        # Find words that map to multiple numbers
        duplicates = {word: numbers for word, numbers in word_to_numbers.items() if len(numbers) > 1}
        
        if duplicates:
            duplicates_found = True
            print(f"\n{language}:")
            for word, numbers in sorted(duplicates.items()):
                print(f"  '{word}' -> {sorted(numbers)}")
    except Exception as e:
        pass  # Skip languages that fail to load

if not duplicates_found:
    print("\nNo duplicate words found! ✓ All numbers have unique vocabulary words.")