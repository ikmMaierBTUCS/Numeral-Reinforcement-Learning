# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 15:25:36 2023

@author: ikm
"""

import numpy as np
from num2words import num2words
import itertools
from itertools import combinations
from itertools import permutations
from sympy import Matrix
from diophantine import solve
import pandas as pd
from alphabet_detector import AlphabetDetector

class Vocabulary:
    def __init__(self, nb, nal):
        if type(nb) is list or type(nb) is np.array:
            try:
                nb=nb.item()
            except:
                pass
               # print(type(nb))
                #try:
                    #print(type(nb.item()))
                #except:
                    #pass
                #raise TypeError("Number has to be an integer")
        if not type(nal) is str:
            raise TypeError("Numeral has to be a string")
        self.number = nb
        self.word = nal
        self.root = nal
        self.inputrange = []
        self.mapping = [nb]
    def printVoc(self):
        print(str(self.number)+' '+self.numeral)
    def all_outputs(self):
        return [self]
    def dimension(self):
        return 0
    def actual_dimension(self):
        return 0
    def sample(self):
        return self
class Highlight:
    def __init__(self, voc, start):
        self.number=voc.number
        self.numeral=voc.word
        self.start=start
        self.root=voc.word
        self.mapping=[voc.number]
    def end(self):
        return self.start+len(self.numeral)
    def hlrange(self):
        return range(self.start+1,self.end())
    def voc(self):
        return Vocabulary(self.number,self.numeral)
    
class SCFunction:
    '''
    Root is the exponent of the function where _ mark input slots
    Inputrange is a list of lists. The nth lists all SCFunctions that may enter the nth input slot
    Mapping is a list of coefficients. The nth coefficient is the factor by which the nth input would have to be multiplied
        Exception: the last coefficient is the constant coefficient.
    '''
    def __init__(self,root,i,mapping):
        if not type(root) is str:
            raise TypeError("Root has to be a string")
        if not type(i) is list:
            raise TypeError("Inputrange has to be a list")
        dimension=root.count('_')
        if not len(i) == dimension:
            print(i)
            raise TypeError(str(dimension)+"-dimensional function needs "+str(dimension)+" component domains.")
        for component in i:
            if not type(component) is list:
                print(type(component))
                raise TypeError("All component domains have to be lists")
            for entry in component:
                if not type(entry) is Vocabulary and not type(entry) is SCFunction:
                    print(type(entry))
                    raise TypeError("All entries of all input components have to be Vocabulary or SCFunction")
        if not type(mapping) is list or dimension+1!=len(mapping):
            print(type(mapping))
            try:
                print(len(mapping))
            except:
                pass
            raise TypeError("Mapping has to be a list of "+str(dimension)+"+1 coefficients")
        #for coeff in mapping:
            #if not type(coeff) is int and not type(coeff) is float:
                #print(type(coeff))
                #raise TypeError("All coefficients of the mapping have to be integers or floats")
        self.root=root
        self.inputrange=i
        self.mapping=mapping
    def dimension(self):
        '''
        Dimension = Number of input slots
        '''
        return self.root.count('_')
    def number_inputs(self):
        ni = []
        for comp in self.inputrange:
            ni += [[entr.sample().mapping[-1] for entr in comp]]
        return ni
    def input_numberbase(self):
        build_base=[]
        #print(len(self.inputrange))
        base_complete = False
        for root_inputx in cartesian_product(self.inputrange):
            for final_inputx in cartesian_product([entr.all_outputs() for entr in root_inputx]):
                #print(build_base)
                #print([component.number for component in final_inputx]+[1])
                #if not inputx in span(build_base): # matrank(buildbase+inputx)=len(Buildbase)+1
                if np.linalg.matrix_rank(np.array(build_base+[[component.mapping[-1] for component in final_inputx]+[1]], dtype=np.float64),tol=None)==len(build_base)+1:
                    #print(str(self.insert(inputx).number)+' is linear independent')
                    build_base=build_base+[[component.mapping[-1] for  component in final_inputx]+[1]]
                    #print([self.insert(inputx).number])
                if len(build_base)==self.dimension()+1:
                    #print('base complete')
                    base_complete = True
                    break
            if base_complete:
                break
        return build_base
        
    def actual_dimension(self):
        '''
        Dimension of input range with respect to affine linearity
        '''
        numbers = [[self.inputrange[i][j].all_outputs()[0].mapping[-1] for j in range(len(self.inputrange[i]))] for i in range(self.dimension())]
        return np.linalg.matrix_rank(np.array(cartesian_product(numbers+[[1]]), dtype=np.float64))
    def insert(self,inputx):
        '''
        Requires a dimension-long list of input SCFunctions
        Return a new SCFunction where all input SCFunctions are inserted in their respective slot
        Updates inputrange and mapping with respect to new inputslots originating from the input SCFunctions
        '''
        
        # catch errors
        if not type(inputx) is list:
            print(type(inputx))
            raise TypeError('Input has to be a list')
        if len(self.inputrange) != len(inputx) or len(self.inputrange) == 0:
            raise TypeError("Input does not match dimension or "+self.root+" has no inputslots.")
        for component in inputx:
            if not type(component) is Vocabulary and not type(component) is SCFunction:
                print(type(component))
                raise TypeError('All components of the input have to be a Vocabulary or an SCFunction')
                
        # trouble shoot if input is not in the inputrange
        for entry in range(len(inputx)):
            if inputx[entry].root not in [inputfunction.root for inputfunction in self.inputrange[entry]]:
                #print("Input "+str([inp.root for inp in inputx])+" is not in the input range of "+str(self.root))
                break
                
        # initialize root, inputrange and mapping of composed SCF
        rootparts = self.root.split('_')
        output_root = ''
        output_i = []
        output_mapping = []
        constant_coefficient = self.mapping[-1]
        
        # extend root, inputrange and mapping
        for inp in range(len(inputx)):
            output_root += rootparts[inp] + inputx[inp].root
            output_i += inputx[inp].inputrange
            output_mapping += [self.mapping[inp] * coeff for coeff in inputx[inp].mapping[:-1]]
            constant_coefficient += self.mapping[inp] * inputx[inp].mapping[-1]
            
        # finish root and mapping and return composed SCF
        output_root += rootparts[-1]
        output_mapping += [constant_coefficient]
        return SCFunction(output_root,output_i,output_mapping)
    def sample(self):
        if self.dimension() == 0:
            return self
        else:
            return self.insert([comp[0].sample() for comp in self.inputrange])
    
    def all_outputs(self):
        '''
        return all final SCFunctions (vocabulary) without unsatisfied '_'s left, that are derivable from 
        '''
        #print('alloutputs of '+self.root)
        if self.dimension() == 0:
            return [self]
        else:
            all_output = []
            #print(self.inputrange)
            for inputvector in cartesian_product(self.inputrange):
                new_outputs = self.insert(inputvector).all_outputs()
                all_output += new_outputs
            return all_output
    def all_outputs_as_voc(self):
        ao = self.all_outputs()
        aov = []
        for scf in ao:
            aov += [Vocabulary(scf.mapping[-1],scf.root)]
        return aov
    
    def merge(self,mergee):
        if not type(mergee) is SCFunction:
            print('not scf')
            raise TypeError("Can only merge with other SCFunction") 
        if not self.root==mergee.root:
            print('different roots')
            raise BaseException('Cannot merge with SCFunction with different exponent')
        if any(len(mergee.inputrange[comp])>1 for comp in range(self.dimension())):
            print('mergee is not singleton')
            raise BaseException('Merge of SCFunctions is yet only implemented for mergees directly produced by proto_parse')
        if self.dimension() == 0:
            print('merger is not generalizable')
            raise BaseException('SCFunction of dimension 0 cannot merge')
        #if mergee.insert(mergee.inputrange[0]).number!=self.insert(mergee.inputrange[0]).number:
            #print('EXPERIMENTAL ERROR! The constructed SCFunction is not affine linear')
        #if [component.number for component in mergee.inputrange[0]] in SPAN(self.inputrange): # insert(Mergee)=mergee.number
        new_inputrange = []
        for comp in range(self.dimension()):
            if mergee.inputrange[comp][0].root in [ent.root for ent in self.inputrange[comp]]:
                new_inputrange += [self.inputrange[comp]]
            else:
                new_inputrange += [self.inputrange[comp] + mergee.inputrange[comp]]
        insert=self.insert([mergee.inputrange[comp][0] for comp in range(mergee.dimension())])
        if insert.mapping[-1] == mergee.mapping[-1]:
            #print('Current mapping predicts value of '+insert.root+' correctly')
            return SCFunction(self.root,new_inputrange,self.mapping)
        else:
            # DANN PRÜFE ERST OB MERGEE NICHT IM SPANN DER INPUTRANGE LIEGT. WENN DOCH, DANN BRAUCHT ES EINE SEPARATE FUNKTION
            # MACH PROPOSAL UND DANN PRÜFE OB ES EINE HÖHERE DIMENSION HAT ALS SELF. WENN NICHT KANN ES NICHT MERGEN
            build_base = self.input_numberbase()
            build_image = [np.dot(self.mapping,basevec) for basevec in build_base]
            #print([[component[0].number for component in mergee.inputrange]+[1]] + build_base)
            new_dim = np.linalg.matrix_rank(np.array([[component[0].number for component in mergee.inputrange]+[1]] + build_base, dtype=np.float64))
            #print('newdim determined')
            if new_dim > self.actual_dimension():
                #print('expand base and image')
                build_base = [[component[0].number for component in mergee.inputrange]+[1]] + build_base
                build_image = [mergee.mapping[-1]] + build_image
            else:
                print('No merge')
                print(build_base)
                print([component[0].number for component in mergee.inputrange]+[1])
                raise BaseException('Mergee must have a different mapping')
            '''
            build_base=[[component[0].number for component in mergee.inputrange]+[1]]
            build_image=[mergee.mapping[-1]]
            #print(len(self.inputrange))
            base_complete = False
            for root_inputx in cartesian_product(self.inputrange):
                for final_inputx in cartesian_product([entr.all_outputs() for entr in root_inputx]):
                    #print(build_base)
                    #print([component.number for component in final_inputx]+[1])
                    #if not inputx in span(build_base): # matrank(buildbase+inputx)=len(Buildbase)+1
                    if np.linalg.matrix_rank(np.array(build_base+[[component.mapping[-1] for component in final_inputx]+[1]], dtype=np.float64),tol=None)==len(build_base)+1:
                        #print(str(self.insert(inputx).number)+' is linear independent')
                        build_base=build_base+[[component.mapping[-1] for  component in final_inputx]+[1]]
                        #print([self.insert(inputx).number])
                        build_image=build_image+[self.insert(final_inputx).mapping[-1]]
                    if len(build_base)==self.dimension()+1:
                        #print('base complete')
                        base_complete = True
                        break
                if base_complete:
                    break
            '''
            #print('calculation')
            #print(build_base)
            #print(build_image)
            try:
                coefficients=solve(build_base,build_image)[0]
            except:
                coefficients=np.dot(np.linalg.pinv(np.array(build_base, dtype=np.float64),rcond=1e-15),build_image)
            coefficients=[round(coeff) for coeff in coefficients]
            #print(coefficients)
            #print(len(mergee.inputrange+self.inputrange))
            return SCFunction(self.root,new_inputrange,list(coefficients))
    def present(self):
        if self.dimension() == 0:
            print(self.root+" is "+str(self.mapping[-1]))
        else:
            domainstrs = []
            for comp in self.inputrange:
                component = '{'
                for entry in comp:
                    if isinstance(entry,Vocabulary) or entry.dimension() == 0:
                        component += str(entry.mapping[-1])+','
                    else:
                        component += str(entry.root)+','
                component = component[:-1]+'}'
                domainstrs += [component]
            domainstr = 'x'.join(domainstrs)
            if self.dimension() == 1:
                inpstr = 'x'
                outpstr = str(self.mapping[0]) + '*x+' + str(self.mapping[1])
            else:
                inpstr = '('
                outpstr = ''
                for comp in range(self.dimension()):
                    inpstr += 'x'+str(comp)+','
                    outpstr += str(self.mapping[comp]) + '*x' + str(comp) + '+'
                inpstr = inpstr[:-1] + ')'
                if self.mapping[-1] != 0:
                    outpstr += str(self.mapping[-1])
                else:
                    outpstr = outpstr[:-1]
            retstr = "Function " + self.root + " maps " + domainstr + " by " + inpstr + ' -> ' + outpstr
            print(retstr)
    def reinforce(self,lexicon,supervisor):
        copy = self
        candidates_for_abstraction = []
        upper_limit = sum([max([0,coeff]) for coeff in self.mapping])
        for entry in lexicon:
            if all([fe.mapping[-1] < upper_limit for fe in entry.all_outputs()]):
                candidates_for_abstraction += [entry]
        invariant_slots = []
        for comp in range(copy.dimension()):
            if len(copy.inputrange[comp]) == 1:
                invariant_slots += [comp]
        for combination in cartesian_product((copy.dimension() - len(invariant_slots)) * [candidates_for_abstraction]):
            cand = []
            for comp in range(copy.dimension()):
                if comp in invariant_slots:
                    cand += [copy.inputrange[comp][0]]
                else:
                    cand += [combination[0]]
                    combination = combination[1:]
            word_cand = copy.insert([candc.sample() for candc in cand])
            #print(word_cand.root)
            proposed_inputrange = []
            entry_is_new = False
            for comp in range(copy.dimension()):
                if cand[comp].root in [entr.root for entr in copy.inputrange[comp]]:
                    proposed_inputrange += [copy.inputrange[comp]]
                else:
                    proposed_inputrange += [copy.inputrange[comp]+[cand[comp]]]
                    entry_is_new = True
            if entry_is_new:
                proposal = SCFunction(copy.root,proposed_inputrange,copy.mapping)
                proposal_is_new = True
                for entr in lexicon:
                    for word in entr.all_outputs():
                        if word_cand.root == word.root or word_cand.mapping[-1] == word.mapping[-1]:
                            #print('proposal ' + word_cand.root + ' ' + str(word_cand.mapping[-1]) + ' already covered by ' + word.root + ' ' + str(word.mapping[-1]))
                            proposal_is_new = False
                            break
                    if not proposal_is_new:
                        break
                if proposal_is_new and proposal.actual_dimension() == copy.actual_dimension():
                    #print('Can I also say '+ word_cand.root + '?')
                    for v in supervisor:
                        if word_cand.root == v.word:
                            #print('Supervisor: Yes')
                            cand_parse = advanced_parse(word_cand.mapping[-1],word_cand.root,lexicon,False,False)
                            if cand_parse.root == self.root:                                
                                if v.number != word_cand.mapping[-1]:
                                    print('LEARNING ERROR: ' + v.word + ' is ' + str(v.number) + ' but learner assumes ' + str(word_cand.mapping[-1]))
                                copy = proposal
                                #copy.present()
                            else:
                                pass
                                #print('OK, but I think this is not related')
                            break
                    else:
                        pass
                        #print('Supervisor: No')
        return copy

def cartesian_product(listlist):
    if len(listlist)==0:
        print('ERROR: empty input in product')
        return False
    for liste in listlist:
        if not isinstance(liste,list):
            liste=[liste]
    cp=[[x] for x in listlist[0]]
    for liste in listlist[1:]:
        cp=[a+[b] for a in cp for b in liste]
    return cp

def delatinized(string):
    #print(string)
    ad = AlphabetDetector()
    if not ad.is_latin(string):
        if not ad.is_cyrillic(string):
            if ad.is_cyrillic(string[0]) and not ad.is_cyrillic(string[-1]):
                #print('first part is cyrillic')
                for point in range(len(string)):
                    if not ad.is_cyrillic(string[:point]):
                        return string[:point-2]
            elif not ad.is_cyrillic(string[0]) and ad.is_cyrillic(string[-1]):
                #print('last part is cyrillic')
                for point in reversed(range(len(string))):
                    if not ad.is_cyrillic(string[point:]):
                        return string[point+2:]
            elif ad.is_latin(string[0]):
                #print('first part is latin')
                for point in range(len(string)+1):
                    if not ad.is_latin(string[:point]):
                        return string[point-1:] 
            elif ad.is_latin(string[-1]):
                #print('last part is latin')
                for point in reversed(range(len(string)+1)):
                    if not ad.is_latin(string[point:]):
                        return string[:point+1]
            else:
                return string
        else:
            return string
    else:
        return string

def create_lexicon(language):
    LEX=[]
    try:
        num2words(1, lang=language)
        for integer in list(range(1,1001))+[1002,1006,1100,1200,1206,7000,7002,7006,7100,7200,7206,10000,17000,17206,20000,27000,27006,27200,27206]:
            try:
                numeral=num2words(integer, lang=language)
                voc=Vocabulary(integer,numeral)
                LEX=LEX+[voc]
            except:
                pass
        return LEX
    except:
        try:
            lanu=pd.read_csv(r'C:\Users\ikm\OneDrive\Desktop\NumeralParsingPerformance\Languages&NumbersData\Numeral.csv', encoding = "utf_16", sep = '\t')
            df=lanu[lanu['Language']==language]
            biscriptual=False
            if ' ' in df.iloc[0,2]:
                biscriptual=True
            for i in range(len(df)):
                numeral=df.iloc[i,2]
                if numeral[0]==' ':
                    numeral=numeral[1:]
                if numeral[-1]==' ':
                    numeral=numeral[:-1]
                if language in ['Latin','Persian','Arabic']:
                    words=numeral.split(' ')
                    numeral=' '.join(iter(words[:-1]))
                if language in ['Chuvash','Adyghe']:
                    words=numeral.split(' ')
                    numeral=' '.join(iter(words[:len(words)//2]))
                if biscriptual and not language in ['Chuvash','Adyghe','Latin','Persian','Arabic']:
                    numeral=delatinized(numeral)
                #print(numeral)
                numeral=numeral.replace('%',',')
                #print(numeral)
                voc=Vocabulary(i+1,numeral)
                LEX=LEX+[voc]
            return LEX
        except:
            raise NotImplementedError("Language "+language+" is not supported or spelled differently")


def advanced_parse(number, word, lexicon, print_doc, print_result):
    if print_doc: print('parse '+word+' '+str(number))
    lexicon1 = []
    if len(lexicon) != 0 and isinstance(lexicon[0],Vocabulary):
        lexicon1 = lexicon
    else:
        for entry in lexicon:
            if isinstance(entry,Vocabulary):
                lexicon += [entry]
            else: 
                lexicon1 += entry.all_outputs_as_voc()
    lexicon1 = lexicon1+[Vocabulary(number,word)]    
    checkpoint = 0
    highlights=[]
    mult_found=False
    for end in range(0, len(word)+1):
        startrange=set(range(checkpoint, end))
        for highlight in highlights:
            startrange=startrange-set(highlight.hlrange())
            #print('remove '+str(range(highlight[3]+1,len(highlight[0]))))
        startrange=sorted(list(startrange))
        #print('startrange='+str(startrange))
        for start in startrange:
            subnum_found_at_this_end=False
            substr=word[start:end]
            #if print_doc: print('substring = '+str(substr))
            for entry in lexicon1:
                #if printb: print('lexnum = '+str(numeral[0]))
                if substr == entry.word:
                    subnum_found_at_this_end=True
                    subentry_found = False
                    if 2*entry.number < number or mult_found:
                        for highlight in reversed(highlights):
                            if highlight.start >= start:
                                if print_doc: print("remove "+highlight.numeral)
                                highlights.remove(highlight)
                        highlights=highlights+[Highlight(Vocabulary(entry.number,entry.word),start)]
                        if print_doc: print([highlight.numeral for highlight in highlights])
                    else: 
                        if print_doc: print(substr+' violates HC')
                        mult_found=True
                        checkpoint=end
                        potential_highlight = None
                        earliest_laterstart = start+1
                        for highlight in highlights:
                            if highlight.number**2 < entry.number:
                                earliest_laterstart = min(end,highlight.end()) #so factors remain untouched
                        for laterstart in range(earliest_laterstart,end):
                            if print_doc: print('subnum = '+word[laterstart:end])
                            for subentry in lexicon1:
                                if word[laterstart:end] == subentry.word:
                                    if subentry.number**2 <= entry.number:
                                        if print_doc: print(word[laterstart:end]+" is FAC or SUM. If it would contain mult, its square would be larger than "+entry.word+'.')
                                        subentry_found=True
                                        for highlight in reversed(highlights):
                                            if highlight.end() > laterstart:
                                                if print_doc: print("remove "+highlight[0])
                                                highlights.remove(highlight)
                                        highlights=highlights+[Highlight(Vocabulary(subentry.number,subentry.word),laterstart)]
                                        checkpoint = laterstart
                                        potential_highlight = None
                                        if print_doc: print([highlight.numeral for highlight in highlights])
                                    else:
                                        if entry.number % subentry.number != 0 and 2*subentry.number < number:
                                            if print_doc: print(word[laterstart:end]+" probably contains SUM. As "+subentry.word+' is no divisor of '+entry.word+', '+entry.word+' has to contain SUM. '+subentry.word+' cannot contain FAC*MULT, as it is smaller than half of '+entry.word+': And it cannot be FAC, as its square is larger than '+entry.word+'. So it is composed of SUM and MULT. If it turns out to be irreducible with the present properties, we assume it is SUM')
                                            potential_highlight = Highlight(Vocabulary(subentry.number,subentry.word),laterstart)
                                            potential_checkpoint = laterstart
                                        else:
                                            potential_highlight = None
                            if subentry_found:
                                break  
                        if potential_highlight != None:
                            for highlight in reversed(highlights):
                                if highlight.end() > potential_checkpoint:
                                    if print_doc: print("remove "+highlight.word)
                                    highlights.remove(highlight)
                            highlights = highlights+[potential_highlight]
                            checkpoint = potential_checkpoint
                            if print_doc: print([highlight.word for highlight in highlights])
                    break                    
            if subnum_found_at_this_end:
                break
    if len(highlights) == 2:
        if highlights[0].number + highlights[1].number == number or highlights[0].number * highlights[1].number == number:
            suspected_mult = max(highlights, key=lambda highlight: highlight.number)
            if print_doc: print("remove "+suspected_mult.root+' since it is probably mult.')
            highlights.remove(suspected_mult)
    elif len(highlights) == 3:
        suspected_mult = max(highlights, key=lambda highlight: highlight.number)
        if suspected_mult.number**2 > number:
            other_numbers = [highlight.number for highlight in highlights if highlight != suspected_mult]
            if other_numbers[0] * suspected_mult.number + other_numbers[1] == number or other_numbers[1] * suspected_mult.number + other_numbers[0] == number:
                if print_doc: print("remove "+suspected_mult.root+' since it is probably mult.')
                highlights.remove(suspected_mult)
    elif len(highlights) > 3:
        suspected_mult = max(highlights, key=lambda highlight: highlight.number)
        if suspected_mult.number**2 > number:
            other_numbers = [highlight.number for highlight in highlights if highlight != suspected_mult]
            for suspected_factor in other_numbers:
                suspected_summand = sum(other_numbers)-suspected_factor
                if suspected_factor * suspected_mult.number + suspected_summand == number:
                    if print_doc: print("remove "+suspected_mult.root+' since it is probably mult.')
                    highlights.remove(suspected_mult)
                    break
        
    #print(str(highlights)+wort+' '+str(zahl))
    root=word
    for highlight in reversed(highlights):
        root=root[0:highlight.start]+'_'+root[highlight.end():len(root)]
    decompstr = str(number)+'='+root+'('+','.join([str(highlight.number) for highlight in highlights])+')'
    if print_result: print(decompstr)
    return SCFunction(root,[[Vocabulary(highlight.number,highlight.numeral)] for highlight in highlights],[0 for highlight in highlights]+[number])


def learn_language(language):
    print('Learning '+language)
    supervisor = create_lexicon(language)
    learnerlex = []
    samples = 0
    for voc in supervisor:
        if not any([voc.word in [outp.root for outp in entr.all_outputs()] for entr in learnerlex]):
            samples += 1
            #1 parse new word
            #print('What means '+str(voc.number)+'?')
            #print('Supervisor: '+voc.word)#+' means '+str(voc.number))
            parse = advanced_parse(voc.number, voc.word, learnerlex, False, True)
            # Understood?
            understood = False
            for entry in learnerlex:
                if entry.root == parse.root:
                    #2a Yes, assuming to have understood. Trying to update an entry
                    try:
                        learned = entry.merge(parse)
                        learnerlex.remove(entry)
                        learned.present()
                        #print(str(entry.actual_dimension()) + ' < ' + str(learned.actual_dimension()) + ' ?')
                        if entry.actual_dimension() < learned.actual_dimension():
                            print('Attempting reinforcement')
                            learned = learned.reinforce(learnerlex,supervisor)
                        understood = True
                        break
                    except:
                        print('Not related to '+entry.root)
                        pass
            if not understood:
                #2b No, just remembering
                learned = parse
            learned.present()
            learnerlex += [learned]
    #reorganize all inputranges so that they only contain scfunctions, no vocabulary
    new_learnerlex = []
    for lexentr in learnerlex:
        #lexentr.present()
        new_inputrange = []
        for comp in lexentr.inputrange:
            new_comp = []
            for entr in comp:
                if isinstance(entr,SCFunction):
                    if not entr in new_comp:
                        new_comp += [entr]
                elif isinstance(entr,Vocabulary):
                    parse = advanced_parse(entr.number,entr.word,learnerlex,False,False)
                    for scf in learnerlex:
                        if isinstance(scf,SCFunction) and parse.root == scf.root and entr.root in [op.root for op in scf.all_outputs()]:
                            if not scf in new_comp:
                                new_comp += [scf]
            new_inputrange += [new_comp]
        new_learnerlex += [SCFunction(lexentr.root,new_inputrange,lexentr.mapping)]
    learnerlex = new_learnerlex
            
    print('Learned '+str(len(supervisor))+' words in '+language+' and structured them in '+str(len(learnerlex))+' functions.')
    print('It took '+str(samples)+' samples to learn those.')
    print('Those are:')
    for entry in learnerlex:
        entry.present()
    print('')