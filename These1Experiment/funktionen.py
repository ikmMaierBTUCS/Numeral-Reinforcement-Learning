import numpy as np
from num2words import num2words
import itertools
from itertools import combinations
from itertools import permutations
from sympy import Matrix
from diophantine import solve
import pandas as pd
from alphabet_detector import AlphabetDetector
import types
import matplotlib.pyplot as plt
import random

import unicodedata

class Highlight:
    def __init__(self, voc, start):
        self.number=voc.number
        self.numeral=voc.word
        self.start=start
        self.root=voc.word
        self.mapping=[voc.number]
        self.word=voc.word
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
    def __init__(self,root,i,mapping,c=[]):
        if not type(root) is str:
            raise TypeError("Root has to be a string")
        if not type(i) is list:
            raise TypeError("Inputrange has to be a list")
        dimension=root.count('_')
        if not len(i) == dimension:
            print(root)
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
        if c == []:
            self.confirmed_inputrange=i
        else:
            # Validate confirmed_inputrange format
            if not type(c) is list:
                raise TypeError("Confirmed inputrange has to be a list")
            if not len(c) == dimension:
                raise TypeError(str(dimension)+"-dimensional function needs "+str(dimension)+" confirmed component domains.")
            for component in c:
                if not type(component) is list:
                    print(type(component))
                    raise TypeError("All confirmed component domains have to be lists")
                for entry in component:
                    if not type(entry) is Vocabulary and not type(entry) is SCFunction:
                        print(type(entry))
                        raise TypeError("All entries of confirmed component domains have to be Vocabulary or SCFunction")
            self.confirmed_inputrange=c
    def __eq__(self,other):
        if isinstance(self,other.__class__):
            if self.root != other.root or self.mapping != other.mapping:
                return False
            else:
                for comp in range(self.dimension()):
                    r_m = set((e.root,tuple(e.mapping)) for e in self.inputrange[comp])
                    other_r_m = set((e.root,tuple(e.mapping)) for e in other.inputrange[comp])
                    if r_m != other_r_m:
                        return False
                return True
        else:
            return NotImplemented
            
    def __hash__(self):
        return hash((self.root,tuple(self.mapping)))
        
    def dimension(self):
        '''
        Dimension = Number of input slots
        '''
        return self.root.count('_')
    def number_inputs(self, only_confirmed = False):
        ni = []
        if only_confirmed:
            ir = self.confirmed_inputrange
        else:
            ir = self.inputrange
        for comp in ir:
            ni += [[entr.sample().mapping[-1] for entr in comp]]
        return ni
    def input_numberbase(self):
        build_base=[]
        #print(len(self.inputrange))
        base_complete = False
        number_of_variable_input_slots = len([comp for comp in range(self.dimension()) if len(self.inputrange[comp]) > 1])
        for root_inputx in cartesian_product(self.confirmed_inputrange):
            for final_inputx in cartesian_product([entr.all_outputs() for entr in root_inputx]):
                #print(build_base)
                #print([component.number for component in final_inputx]+[1])
                #if not inputx in span(build_base): # matrank(buildbase+inputx)=len(Buildbase)+1
                if np.linalg.matrix_rank(np.array(build_base+[[component.mapping[-1] for component in final_inputx]+[1]], dtype=np.float64),tol=None)==len(build_base)+1:
                    #print(str(self.insert(inputx).number)+' is linear independent')
                    build_base=build_base+[[component.mapping[-1] for  component in final_inputx]+[1]]
                    #print([self.insert(inputx).number])
                if len(build_base) == number_of_variable_input_slots + 1:
                    #print('base complete')
                    base_complete = True
                    break
            if base_complete:
                break
        return build_base
        
    def actual_dimension(self):
        '''
        1 + Dimension of input range with respect to affine linearity
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
        output_c = []
        output_mapping = []
        constant_coefficient = self.mapping[-1]
        
        # extend root, inputrange and mapping
        for inp in range(len(inputx)):
            output_root += rootparts[inp] + inputx[inp].root
            output_i += inputx[inp].inputrange
            output_c += inputx[inp].confirmed_inputrange
            output_mapping += [self.mapping[inp] * coeff for coeff in inputx[inp].mapping[:-1]]
            constant_coefficient += self.mapping[inp] * inputx[inp].mapping[-1]
            
        # finish root and mapping and return composed SCF
        output_root += rootparts[-1]
        output_mapping += [constant_coefficient]
        return SCFunction(output_root,output_i,output_mapping,output_c)

        
    def sample(self, size = -1, only_confirmed = False):
        '''
        If not size given, returns one output of self
        If size given, returns a random size-length list of outputs of self
        '''
        if self.dimension() == 0:
            return self
        else:
            if only_confirmed:
                ir = self.confirmed_inputrange
            else:
                ir = self.inputrange
            if size == -1:
                return self.insert([comp[0].sample() for comp in ir])
            else:
                inputvectors = cartesian_product(ir)
                size = min([size, len(inputvectors)])
                inputsample = random.sample(inputvectors,size)
                return [self.insert(i) for i in inputsample]
    
    def all_outputs(self, only_confirmed = False):
        '''
        return all final SCFunctions (vocabulary) without unsatisfied '_'s left, that are derivable from 
        '''
        #print('alloutputs of '+self.root)
        if self.dimension() == 0:
            return [SCFunction(self.root,[],[round(self.mapping[0])])]
        else:
            if only_confirmed:
                ir = self.confirmed_inputrange
            else:
                ir = self.inputrange
            all_output = []
            #print(self.inputrange)
            for inputvector in cartesian_product(ir):
                new_outputs = self.insert(inputvector).all_outputs()
                all_output += new_outputs
            return all_output
    def all_outputs_as_voc(self, only_confirmed = False):
        ao = self.all_outputs(only_confirmed)
        aov = []
        for scf in ao:
            aov += [Vocabulary(scf.mapping[-1],scf.root)]
        return aov
    
    def merge(self,mergee,allow_mapping_manipulation=False,printb= True):
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
            return mergee
            #raise BaseException('SCFunction ' + self.root + ' of dimension 0 cannot merge')
        #if mergee.insert(mergee.inputrange[0]).number!=self.insert(mergee.inputrange[0]).number:
            #print('EXPERIMENTAL ERROR! The constructed SCFunction is not affine linear')
        #if [component.number for component in mergee.inputrange[0]] in SPAN(self.inputrange): # insert(Mergee)=mergee.number
        new_inputrange = []
        new_confirmed_inputrange = []
        for comp in range(self.dimension()):
            if mergee.inputrange[comp][0].root in [ent.root for ent in self.inputrange[comp]]:
                new_inputrange += [self.inputrange[comp]]
            else:
                new_inputrange += [self.inputrange[comp] + mergee.inputrange[comp]]
            if mergee.confirmed_inputrange[comp][0].root in [ent.root for ent in self.confirmed_inputrange[comp]]:
                new_confirmed_inputrange += [self.confirmed_inputrange[comp]]
            else:
                new_confirmed_inputrange += [self.confirmed_inputrange[comp] + mergee.confirmed_inputrange[comp]]
        insert=self.insert([mergee.inputrange[comp][0] for comp in range(mergee.dimension())])
        if insert.mapping[-1] == mergee.mapping[-1]:
            #print('Current mapping predicts value of '+insert.root+' correctly')
            return SCFunction(self.root,new_inputrange,self.mapping, new_confirmed_inputrange)
        else:
            # DANN PRÜFE ERST OB MERGEE NICHT IM SPANN DER INPUTRANGE LIEGT. WENN DOCH, DANN BRAUCHT ES EINE SEPARATE FUNKTION
            # MACH PROPOSAL UND DANN PRÜFE OB ES EINE HÖHERE DIMENSION HAT ALS SELF. WENN NICHT KANN ES NICHT MERGEN
            build_base = [b for b in self.input_numberbase()]
            build_image = [round(np.dot(self.mapping,basevec)) for basevec in build_base]
            #print([[component[0].number for component in mergee.inputrange]+[1]] + build_base)
            new_dim = np.linalg.matrix_rank(np.array([[component[0].number for component in mergee.inputrange]+[1]] + build_base, dtype=np.float64))
            #print('newdim determined')
            if new_dim > self.actual_dimension() or allow_mapping_manipulation:
                #print('expand base and image')
                build_base = [[component[0].number for component in mergee.inputrange]+[1]] + build_base
                build_image = [mergee.mapping[-1]] + build_image
            else:
                self.sample().present()
                print('No merge')
                self.present()
                mergee.present()
                #print(build_base)
                #print([component[0].number for component in mergee.inputrange]+[1])
                raise BaseException('Mergee must have a different mapping')
            if printb: print(build_base)
            if printb: print(build_image)
            coefficients=intlinsolve([b[:-1] for b in build_base],build_image)
            #try:
                #coefficients=solve(build_base,build_image)[0]
            #except:
                #coefficients=np.dot(np.linalg.pinv(np.array(build_base, dtype=np.float64),rcond=1e-15),build_image)
            #coefficients=[round(coeff) for coeff in coefficients]
            #print(coefficients)
            #print(len(mergee.inputrange+self.inputrange))
            return SCFunction(self.root,new_inputrange,list(coefficients),new_confirmed_inputrange)


    def vereinige(self,mergee, other_mergees = [],printb= True,trust_affinity=True):
        # Behandle Errore
        for me in [mergee]+other_mergees:
            if type(other_mergees) != list:
                raise TypeError("Function multi_mergable requires a non-empty list of mergees")
            if not type(mergee) is SCFunction:
                print('not scf')
                raise TypeError("Can only merge with other SCFunctions") 
            if any(len(mergee.inputrange[comp])>1 for comp in range(self.dimension())):
                print('Mergee is not singleton')
                mergee.present()
                raise BaseException('Merge of SCFunctions is yet only implemented for mergees directly produced by proto_parse')
            if self.dimension() == 0:
                print('merger is not generalizable')
                raise BaseException('SCFunction ' + self.root + ' of dimension 0 cannot merge')
            
        print('Vereinige die folgenden Funktionen:')
        self.present()
        mergee.present()
        for me in other_mergees:
            me.present()

        mergee_inputvector = [mergee.inputrange[i][0] for i in range(mergee.dimension())] # U=(U_1,...,U_k)

        # Prüfe ob u bereits im affinen Spann des Definitionsbereichs von self liegt
        affine_basis = self.input_numberbase() # affine Basis des Defbereichs von self
        if np.linalg.matrix_rank(np.array(affine_basis + [[m.mapping[-1] for m in mergee_inputvector] + [1]], dtype=float), tol=1e-7) == len(affine_basis):
            print('mergee lies in affine span of inputrange of merger')
            if trust_affinity:
                vereinigte_funktion = SCFunction(self.root,[self.inputrange[comp] + mergee.inputrange[comp] for comp in range(self.dimension())], self.mapping, c = [self.confirmed_inputrange[comp] + mergee.inputrange[comp] for comp in range(self.dimension())])
                return vereinigte_funktion
            else:
                # check if self predicts correct value for mergee
                insert=self.insert([mergee.inputrange[comp][0] for comp in range(mergee.dimension())])
                if insert.mapping[-1] == mergee.mapping[-1]:
                    vereinigte_funktion = SCFunction(self.root,[self.inputrange[comp] + mergee.inputrange[comp] for comp in range(self.dimension())], self.mapping, c = [self.confirmed_inputrange[comp] + mergee.inputrange[comp] for comp in range(self.dimension())])
                    return vereinigte_funktion
                else:
                    if printb:
                        print('Merging these functions would cause wrong value calculations, so the latter function is returned as an atom')
                        self.present()
                        mergee.present()
                    return mergee.insert([mergee.inputrange[comp][0] for comp in range(mergee.dimension())])

        other_mergee_inputvectors = [[me.inputrange[i][0] for i in range(mergee.dimension())] for me in other_mergees] # W_2,W_3,...
        for comp in range(self.dimension()):
            print(comp, set([mergee_inputvector[comp]] + [omi[comp] for omi in other_mergee_inputvectors]).issubset(self.inputrange[comp]))
        for comp in range(self.dimension()):
            for e in self.inputrange[comp]:
                e.present()
            mergee_inputvector[comp].present()
            if set([mergee_inputvector[comp]] + [omi[comp] for omi in other_mergee_inputvectors]).issubset(self.inputrange[comp]):
                print('component '+str(comp)+' is already covered by merger')
            else:
                print('component '+str(comp)+' triggers extension')
        neue_dimension = len([comp for comp in range(self.dimension()) if len(self.inputrange[comp]) > 1 or not set(mergee.inputrange[comp] + [omi[comp] for omi in other_mergee_inputvectors]).issubset(self.inputrange[comp])]) # l
        print('neue_dimension',neue_dimension)
        print('affine_basis',len(affine_basis))

        # Konstruiere ersten Teil von y
        bildwerte = [np.dot(self.mapping, basevec) for basevec in affine_basis]
        # Finde b_1,...,b_l
        other_benotigte_vektoren = neue_dimension - len(affine_basis)
        print(other_benotigte_vektoren)
        for other_indices in itertools.combinations(range(len(other_mergee_inputvectors)), other_benotigte_vektoren):
            other_vektoren = [other_mergee_inputvectors[i] for i in other_indices]
            neue_affine_basis = np.array(affine_basis + [[m.mapping[-1] for m in mergee_inputvector] + [1]] + [[m.mapping[-1] for m in other_inputvector] + [1] for other_inputvector in other_vektoren], dtype=float) # prüfe affin lineare unabhängigkeit
            print('neue_affine_basis',neue_affine_basis)
            rang_neue_affine_basis = np.linalg.matrix_rank(neue_affine_basis, tol=1e-7)
            print('rang:', rang_neue_affine_basis, 'neue_dimension:', neue_dimension, 'rang-1:', rang_neue_affine_basis - 1, 'bedingung erfüllt:', rang_neue_affine_basis - 1 == neue_dimension)
            bildwerte_erweitert = bildwerte + [mergee.mapping[-1]] + [other_mergees[i].mapping[-1] for i in other_indices]
            if rang_neue_affine_basis - 1 == neue_dimension:
                print(bildwerte_erweitert)
                koeffizienten = intlinsolve([b[:-1] for b in neue_affine_basis],bildwerte_erweitert)
                print(koeffizienten)
                neuer_definitionsbereich = [self.inputrange[comp] + mergee.inputrange[comp] + [me[comp] for me in other_vektoren] for comp in range(self.dimension())]
                neuer_bestatigter_definitionsbereich = [self.confirmed_inputrange[comp] + mergee.inputrange[comp] + [me[comp] for me in other_vektoren] for comp in range(self.dimension())]
                return SCFunction(self.root,neuer_definitionsbereich, list(koeffizienten), c = neuer_bestatigter_definitionsbereich)
        return mergee
            
    def multi_merge(self,mergees,printb= True,trust_affinity=True):
        for mergee in mergees:
            if type(mergees) != list or len(mergees)==0:
                raise TypeError("Function multi_mergable requires a non-empty list of mergees")
            if not type(mergee) is SCFunction:
                print('not scf')
                raise TypeError("Can only merge with other SCFunctions") 
            if any(len(mergee.inputrange[comp])>1 for comp in range(self.dimension())):
                print('Mergee is not singleton')
                mergee.present()
                raise BaseException('Merge of SCFunctions is yet only implemented for mergees directly produced by proto_parse')
            if self.dimension() == 0:
                print('merger is not generalizable')
                raise BaseException('SCFunction ' + self.root + ' of dimension 0 cannot merge')
                
        new_inputrange = []
        new_confirmed_inputrange = []
        for comp in range(self.dimension()):
            new_inputrange += [self.inputrange[comp].copy()]
            new_confirmed_inputrange += [self.confirmed_inputrange[comp].copy()]
            for mergee in mergees:
                if mergee.inputrange[comp][0].root in [ent.root for ent in self.inputrange[comp]]:
                    pass
                else:
                    #self.present()
                    new_inputrange[comp] += mergee.inputrange[comp]
                    #self.present()
                if mergee.confirmed_inputrange[comp][0].root in [ent.root for ent in self.confirmed_inputrange[comp]]:
                    pass
                else:
                    #self.present()
                    new_confirmed_inputrange[comp] += mergee.confirmed_inputrange[comp]
                    #self.present()

        
        #if all_predictions_right:
        build_base = self.input_numberbase()
        # Erzeuge die Liste wie bisher
        actual_list = build_base + [[mergee.inputrange[comp][0].mapping[-1] for comp in range(mergee.dimension())] + [1] for mergee in mergees]

        # Erzwinge die Umwandlung aller Elemente zu float
        try:
            safe_array = np.array([[float(element) for element in row] for row in actual_list], dtype=float)
            rank = np.linalg.matrix_rank(safe_array, tol=1e-10)
            #print("Rang:", rank)
        except (ValueError, TypeError) as e:
            print(f"Fehler bei der Umwandlung zu float: {e}")
            # Optional: Debug-Ausgabe, um das problematische Element zu finden
            for i, row in enumerate(actual_list):
                for j, element in enumerate(row):
                    try:
                        float(element)
                    except (ValueError, TypeError):
                        print(f"Problem bei Element [{i}][{j}]: Wert = {element}, Typ = {type(element)}")

        #if np.linalg.matrix_rank(np.array(build_base + [[mergee.inputrange[comp][0].mapping[-1] for comp in range(mergee.dimension())]+[1] for mergee in mergees]), tol=1e-10) == len(build_base):
        if rank == len(build_base):
            if trust_affinity:
                return [SCFunction(self.root,new_inputrange,self.mapping,new_confirmed_inputrange)]
            else:
                # check if self predicts correct values vor all mergees
                all_predictions_right = True
                for mergee in mergees:
                    insert=self.insert([mergee.inputrange[comp][0] for comp in range(mergee.dimension())])
                    if insert.mapping[-1] != mergee.mapping[-1]:
                        all_predictions_right = False
                        break
                if all_predictions_right:
                    return [SCFunction(self.root,new_inputrange,self.mapping,new_confirmed_inputrange)]
                else:
                    if printb:
                        print('Merging these functions would cause wrong value calculations, so the latter function is returned as an atom')
                        self.present()
                        for me in mergees:
                            me.present()
                    return [self,mergees[-1].all_outputs()[0]]
                    
        else:
            # DANN PRÜFE ERST OB MERGEE NICHT IM SPANN DER INPUTRANGE LIEGT. WENN DOCH, DANN BRAUCHT ES EINE SEPARATE FUNKTION
            # MACH PROPOSAL UND DANN PRÜFE OB ES EINE HÖHERE DIMENSION HAT ALS SELF. WENN NICHT KANN ES NICHT MERGEN
            #build_base = self.input_numberbase()
            build_image = [np.dot(self.mapping,basevec) for basevec in build_base]

            for mergee in mergees:
                added_basevecs = 0
                mdim = mergee.actual_dimension()
                for vec in cartesian_product(mergee.number_inputs()):
                    if np.linalg.matrix_rank(np.array(build_base + [vec + [1]], dtype=float), tol=1e-10) == len(build_base) + 1:
                        added_basevecs += 1
                        build_base += [vec + [1]]
                        build_image += [np.dot(mergee.mapping,vec + [1])]
                        if added_basevecs == mdim:
                            break
                
            #print([[component[0].number for component in mergee.inputrange]+[1]] + build_base)
            new_dim = np.linalg.matrix_rank(np.array(build_base, dtype=np.float64))
            #print('newdim determined')
            #print('newdim',new_dim,'olddim',self.actual_dimension())
            if printb: self.present()
            if not new_dim > self.actual_dimension():
                self.sample().present()
                print('No merge')
                self.present()
                mergee.present()
                #print(build_base)
                #print([component[0].number for component in mergee.inputrange]+[1])
                raise BaseException('Mergee must have a different mapping')
            if printb: print(build_base)
            if printb: print(build_image)
            coefficients=intlinsolve([b[:-1] for b in build_base],build_image)
            #try:
                #coefficients=solve(build_base,build_image)[0]
            #except:
                #coefficients=np.dot(np.linalg.pinv(np.array(build_base, dtype=np.float64),rcond=1e-15),build_image)
            #coefficients=[round(coeff) for coeff in coefficients]
            #print(coefficients)
            #print(len(mergee.inputrange+self.inputrange))
            #print(new_inputrange)
            return [SCFunction(self.root,new_inputrange,list(coefficients),new_confirmed_inputrange)]

    def multi_mergable(self,mergeelist):
        #function checks whether self can merge with a number of mergees
        #print('merger: ')
        #self.present()
        #print('mergees :')
        #for mergee in mergeelist:
            #mergee.present()
        for mergee in mergeelist:
            if type(mergeelist) != list or len(mergeelist)==0:
                raise TypeError("Function multi_mergable requires a non-empty list of mergees")
            if not type(mergee) is SCFunction:
                print('not scf')
                raise TypeError("Can only merge with other SCFunctions") 
            if any(len(mergee.inputrange[comp])>1 for comp in range(self.dimension())):
                print('Mergee is not singleton')
                mergee.present()
                return False
                #raise BaseException('Merge of SCFunctions is yet only implemented for mergees directly produced by proto_parse')
            if self.dimension() == 0:
                print('merger is not generalizable')
                raise BaseException('SCFunction ' + self.root + ' of dimension 0 cannot merge')
            
            if not self.root==mergee.root:
                return False
                
        #if len(mergeelist) > self.dimension() - self.actual_dimension() + 1:
            #print('No the mergeelist is too long')
            #return False

        mergeeinputs = []
        for mergee in mergeelist:
            if all([len(comp) == 1 for comp in mergee.inputrange]):
                mergeeinputs += [cartesian_product(mergee.number_inputs())[0]]
            else:
                mergeeinputs += [basevec[:-1] for basevec in mergee.input_numberbase()]
                
        #print('mergeeinputs: ', mergeeinputs)
        #print(self.dimension(), cartesian_product(self.number_inputs()), mergeeinputs)
        for innum in cartesian_product(self.number_inputs()):
            invariant_slots = [i for i in range(self.dimension()) if all([innum[i] == mergeeinput[i] for mergeeinput in mergeeinputs])]
            if len(invariant_slots) >= self.dimension() - len(mergeelist):
                #still check linear independence
                if np.linalg.matrix_rank(np.array([mi + [1] for mi in mergeeinputs] + [innum + [1]], dtype=float), tol=1e-10) == len(mergeeinputs) + 1:
                    #print('The following functions can be merged:')
                    #self.present()
                    #for mergee in mergeelist:
                        #mergee.present()
                    return True
        return False
                
            
    
    def present(self, domain=True, printout=True, only_confirmed = False):
        if self.dimension() == 0:
            if printout: 
                print(self.root+" is "+str(self.mapping[-1]))
            return self.root+" is "+str(self.mapping[-1])
        else:
            domainstrs = []
            if only_confirmed:
                ir = self.confirmed_inputrange
            else:
                ir = self.inputrange
            for comp in ir:
                component = '{'
                if len(comp) < 20:
                    for entry in comp:
                        if isinstance(entry,Vocabulary) or entry.dimension() == 0:
                            component += str(entry.mapping[-1])+','
                        else:
                            component += str(entry.root)+','
                else:
                    for entry in comp[:10]:
                        if isinstance(entry,Vocabulary) or entry.dimension() == 0:
                            component += str(entry.mapping[-1])+','
                        else:
                            component += str(entry.root)+','
                    component += '...,'
                    for entry in comp[-10:]:
                        if isinstance(entry,Vocabulary) or entry.dimension() == 0:
                            component += str(entry.mapping[-1])+','
                        else:
                            component += str(entry.root)+','
                component = component[:-1]+'}'
                domainstrs += [component]
            domainstr = 'x'.join(domainstrs)
            if self.dimension() == 1:
                inpstr = 'x'
                outpstr = str(round(self.mapping[0],2)) + 'x+' + str(round(self.mapping[1],2))
            else:
                variables = ['x','y','z','a','b','c','d','e','f','g','h']
                inpstr = '('
                outpstr = ''
                for comp in range(self.dimension()):
                    inpstr += variables[comp]+','
                    outpstr += str(round(self.mapping[comp],2)) + variables[comp] + '+'
                inpstr = inpstr[:-1] + ')'
                if self.mapping[-1] != 0:
                    outpstr += str(round(self.mapping[-1],2))
                else:
                    outpstr = outpstr[:-1]
            if domain:
                retstr = "Function " + self.root + "\t maps " + domainstr + "\t by " + inpstr + '\t -> ' + outpstr
            else:
                retstr = self.root + "\t maps " + inpstr + '\t -> ' + outpstr
            if printout:
                print(retstr)
            return retstr
        
    def verstarke(self,lexicon,orakel,printb=True,aux_lex=[]):
        if self.dimension() == 0:
            return self
        copy = self
        candidates_for_abstraction = []
        upper_limit = sum([max([0,coeff]) for coeff in self.mapping])
        for entry in lexicon:
            # for dummy supervisor we allow a little more tolerance
            if False: # not type(supervisor) == list and supervisor.style in ['dummy']:
                pass
                #produced_numbers = sorted([fe.mapping[-1] for fe in entry.all_outputs()])
                #if produced_numbers[len(produced_numbers)*3//4] < upper_limit:
                    #candidates_for_abstraction += [entry]
                #if sum(entry.mapping) + 5 < sum(self.mapping) or all([fe.mapping[-1] < upper_limit for fe in entry.all_outputs()]):
                    #candidates_for_abstraction += [entry]
                #if all([fe.mapping[-1] < upper_limit for fe in entry.all_outputs(only_confirmed = True)]):
                    #candidates_for_abstraction += [entry]
            else:
                if all([fe.mapping[-1] < upper_limit for fe in entry.all_outputs(only_confirmed = True)]):
                    candidates_for_abstraction += [entry]
        
        verstarkter_definitionsbereich = [self.inputrange[comp].copy() for comp in range(self.dimension())]
        for comp in range(self.dimension()):
            if len(copy.inputrange[comp]) > 1:
                for candidate in candidates_for_abstraction:
                    if not candidate in copy.inputrange[comp]:
                        eingabekombinationen = []
                        for c in range(self.dimension()):
                            if c == comp:
                                eingabekombinationen += [candidate.all_outputs()]
                            else:
                                kombinationskomponente = []
                                for e in self.inputrange[c]:
                                    kombinationskomponente += e.all_outputs()
                                eingabekombinationen += [kombinationskomponente]
                        entwurfe = []
                        for eingabe in cartesian_product(eingabekombinationen):
                            entwurfe += [self.insert(list(eingabe))]
                        print(entwurfe)
                        if orakel.antwort(entwurfe):
                            verstarkter_definitionsbereich[comp] += [candidate]
                            if printb: print('Input '+candidate.root+' added to input slot '+str(comp)+' of '+self.root)
        verstarkte_funktion = SCFunction(self.root,verstarkter_definitionsbereich,self.mapping,self.confirmed_inputrange)
        return verstarkte_funktion

                            
                        

        invariant_slots = []
        for comp in range(copy.dimension()):
            if len(copy.inputrange[comp]) == 1:
                invariant_slots += [comp]
        if len(invariant_slots) == copy.dimension():
            return copy
        # construct set of all input combinations to try out (under omission of invariant_slots)
        # ADJUST THE FOLLOWING IF SUPERVISOR NOT OMNISCIENT
        if True: #aux_lex == []:
            #input_combinations = cartesian_product((copy.dimension() - len(invariant_slots)) * [candidates_for_abstraction])
        #else:
            input_combinations = []
            for varied_slot in range(copy.dimension()):
                components = []
                for comp in range(copy.dimension()):
                    if comp in invariant_slots:
                        pass
                    elif comp == varied_slot:
                        components += [candidates_for_abstraction]
                    else: 
                        components += [[self.inputrange[comp][1]]]
                        #components += [candidates_for_abstraction + self.inputrange[comp]]
                input_combinations += cartesian_product(components)
        for combination in input_combinations:
            cand = []
            for comp in range(copy.dimension()):
                if comp in invariant_slots:
                    cand += [copy.inputrange[comp][0]]
                else:
                    cand += [combination[0]]
                    combination = combination[1:]
            #word_cand = copy.insert([candc.sample() for candc in cand])
            template_cand = copy.insert(cand)
            #print('word_candidate:',template_cand.root)
            proposed_inputrange = []
            entry_is_new = False
            for comp in range(copy.dimension()):
                if cand[comp].root in [entr.root for entr in copy.inputrange[comp]]:
                    proposed_inputrange += [copy.inputrange[comp]]
                else:
                    proposed_inputrange += [copy.inputrange[comp]+[cand[comp]]]
                    entry_is_new = True
            if entry_is_new:
                proposal = SCFunction(copy.root,proposed_inputrange,copy.mapping,copy.confirmed_inputrange)
                proposal_is_new = True
                word_cand = template_cand.sample()
                for entr in lexicon:
                    for word in entr.all_outputs():
                        if (word_cand.root == word.root) ^ (word_cand.mapping[-1] == word.mapping[-1]):
                            #print('proposal ' + word_cand.root + ' ' + str(word_cand.mapping[-1]) + ' already covered by ' + word.root + ' ' + str(word.mapping[-1]))
                            proposal_is_new = False
                            break
                    if not proposal_is_new:
                        break
                # for certain supervisor types redundant proposals are welcome to overwrite existing ones
                if not type(supervisor) == list and supervisor.style in ['arithmetic','dummy']:
                    proposal_is_new = True
                if proposal_is_new and proposal.actual_dimension() == copy.actual_dimension():
                    #print('Can I also say '+ word_cand.root + '?')
                    if type(supervisor) == list:
                        for v in supervisor:
                            if word_cand.root == v.word:
                                #print('Oracle: Yes')
                                cand_parse = advanced_parse(word_cand.mapping[-1],word_cand.root,lexicon,False,False)
                                if cand_parse.root == self.root:                                
                                    if v.number != word_cand.mapping[-1]:
                                        print('LEARNING ERROR: ' + v.word + ' is ' + str(v.number) + ' but learner assumes ' + str(word_cand.mapping[-1]))
                                    copy = proposal
                                    #copy.present()
                                else:
                                    pass
                                    if printb: print('OK, but I think this is not related')
                                break
                            else:
                                pass
                                #print('Oracle: No')
                    else:
                        if supervisor.answer(template_cand):
                            #if printb: print('Oracle: Yes also '+ word_cand.root)
                            copy = proposal
                        else:
                            pass
                            #if printb: print('Oracle: No not '+ word_cand.root)
        return copy

    def reinforce(self,lexicon,supervisor,printb=True,aux_lex=[]):
        if self.dimension() == 0:
            return self
        copy = self
        candidates_for_abstraction = []
        upper_limit = sum([max([0,coeff]) for coeff in self.mapping])
        for entry in lexicon:
            # for dummy supervisor we allow a little more tolerance
            if False: # not type(supervisor) == list and supervisor.style in ['dummy']:
                pass
                #produced_numbers = sorted([fe.mapping[-1] for fe in entry.all_outputs()])
                #if produced_numbers[len(produced_numbers)*3//4] < upper_limit:
                    #candidates_for_abstraction += [entry]
                #if sum(entry.mapping) + 5 < sum(self.mapping) or all([fe.mapping[-1] < upper_limit for fe in entry.all_outputs()]):
                    #candidates_for_abstraction += [entry]
                #if all([fe.mapping[-1] < upper_limit for fe in entry.all_outputs(only_confirmed = True)]):
                    #candidates_for_abstraction += [entry]
            else:
                if all([fe.mapping[-1] < upper_limit for fe in entry.all_outputs(only_confirmed = True)]):
                    candidates_for_abstraction += [entry]
        if candidates_for_abstraction == []:
            return copy
        invariant_slots = []
        for comp in range(copy.dimension()):
            if len(copy.inputrange[comp]) == 1:
                invariant_slots += [comp]
        if len(invariant_slots) == copy.dimension():
            return copy
        # construct set of all input combinations to try out (under omission of invariant_slots)
        # ADJUST THE FOLLOWING IF SUPERVISOR NOT OMNISCIENT
        if True: #aux_lex == []:
            #input_combinations = cartesian_product((copy.dimension() - len(invariant_slots)) * [candidates_for_abstraction])
        #else:
            input_combinations = []
            for varied_slot in range(copy.dimension()):
                components = []
                for comp in range(copy.dimension()):
                    if comp in invariant_slots:
                        pass
                    elif comp == varied_slot:
                        components += [candidates_for_abstraction]
                    else: 
                        components += [[self.inputrange[comp][1]]]
                        #components += [candidates_for_abstraction + self.inputrange[comp]]
                input_combinations += cartesian_product(components)
        for combination in input_combinations:
            cand = []
            for comp in range(copy.dimension()):
                if comp in invariant_slots:
                    cand += [copy.inputrange[comp][0]]
                else:
                    cand += [combination[0]]
                    combination = combination[1:]
            #word_cand = copy.insert([candc.sample() for candc in cand])
            template_cand = copy.insert(cand)
            #print('word_candidate:',template_cand.root)
            proposed_inputrange = []
            entry_is_new = False
            for comp in range(copy.dimension()):
                if cand[comp].root in [entr.root for entr in copy.inputrange[comp]]:
                    proposed_inputrange += [copy.inputrange[comp]]
                else:
                    proposed_inputrange += [copy.inputrange[comp]+[cand[comp]]]
                    entry_is_new = True
            if entry_is_new:
                proposal = SCFunction(copy.root,proposed_inputrange,copy.mapping,copy.confirmed_inputrange)
                proposal_is_new = True
                word_cand = template_cand.sample()
                for entr in lexicon:
                    for word in entr.all_outputs():
                        if (word_cand.root == word.root) ^ (word_cand.mapping[-1] == word.mapping[-1]):
                            #print('proposal ' + word_cand.root + ' ' + str(word_cand.mapping[-1]) + ' already covered by ' + word.root + ' ' + str(word.mapping[-1]))
                            proposal_is_new = False
                            break
                    if not proposal_is_new:
                        break
                # for certain supervisor types redundant proposals are welcome to overwrite existing ones
                if not type(supervisor) == list and supervisor.style in ['arithmetic','dummy']:
                    proposal_is_new = True
                if proposal_is_new and proposal.actual_dimension() == copy.actual_dimension():
                    #print('Can I also say '+ word_cand.root + '?')
                    if type(supervisor) == list:
                        for v in supervisor:
                            if word_cand.root == v.word:
                                #print('Oracle: Yes')
                                cand_parse = advanced_parse(word_cand.mapping[-1],word_cand.root,lexicon,False,False)
                                if cand_parse.root == self.root:                                
                                    if v.number != word_cand.mapping[-1]:
                                        print('LEARNING ERROR: ' + v.word + ' is ' + str(v.number) + ' but learner assumes ' + str(word_cand.mapping[-1]))
                                    copy = proposal
                                    #copy.present()
                                else:
                                    pass
                                    if printb: print('OK, but I think this is not related')
                                break
                            else:
                                pass
                                #print('Oracle: No')
                    else:
                        if supervisor.answer(template_cand):
                            #if printb: print('Oracle: Yes also '+ word_cand.root)
                            copy = proposal
                        else:
                            pass
                            #if printb: print('Oracle: No not '+ word_cand.root)
        return copy


class Vocabulary(SCFunction):
    def __init__(self, nb, nal):
        if type(nb) is list or type(nb) is np.array:
            try:
                nb=nb.item()
            except:
                pass
        if not type(nal) is str:
            raise TypeError("Numeral has to be a string")
        self.number = nb
        self.word = nal
        self.root = nal
        self.inputrange = []
        self.confirmed_inputrange = []
        self.mapping = [nb]
    def printVoc(self):
        print(str(self.number)+' '+self.word)
    def all_outputs(self):
        return [self]
    def dimension(self):
        return 0
    def actual_dimension(self):
        return 1
    def sample(self):
        return self

class Oracle:
    def __init__(self, style, data={}, search_engine=[], estimation_parameters=[30,0], tolerance_factor = 10, security_factor = 5, knowledge=[], negative_knowledge=[]):
        if style not in ['statistic','manual','omniscient','arithmetic', 'dummy','hybrid']:
            raise TypeError("Style must be either 'statistic', 'manual', 'arithmetic', 'dummy', 'hybrid', or 'omniscient'.")
        if style == 'omniscient' and knowledge == [] or type(knowledge) != list:
            raise ValueError('Omniscient supervisor requires a knowledge list.')
        if style in ['statistic','hybrid'] and not isinstance(search_engine, types.FunctionType):
            raise TypeError('Statistic or hybrid supervisor require a method search_engine that maps input strings to a number of search results.')
        self.style = style
        self.data = data
        self.search_engine = search_engine
        self.estimation_parameters = estimation_parameters
        self.knowledge = knowledge
        self.negative_knowledge = negative_knowledge
        if self.style == 'statistic' and self.data != {}:
            self.update([])
        self.tolerance_factor = tolerance_factor
        self.security_factor = security_factor
        
    def antwort(self,entwurfe, tol = 0,show_plot=True):
        if self.style == 'statistic':
            for entwurf in entwurfe:
                if not self.answer(entwurf,tol,show_plot):
                    return False
                if isinstance(proposal,str):
                    raise TypeError("Statistic supervisor only answers to inputs SCFunction, Vocabulary, or [word, number]")
            l = len(entwurfe)
            if l > 20:
                l = 20
                entwurfe = random.sample(entwurfe,20)
            value = sum([p.mapping[-1] for p in entwurfe]) / l
            search_results = sum([self.search_engine(p.root) for p in entwurfe]) / l
            if tol == 0:
                tol = self.tolerance_factor
            lower_limit = np.exp(self.estimation_parameters[0]) * value ** self.estimation_parameters[1] / tol
            if search_results == 0:
                print("0 < " + str(round(lower_limit,1)))
                return False
            if show_plot:
                x = np.arange(min(self.data.keys()),max(list(self.data.keys())+[value]))
                y = np.exp(self.estimation_parameters[0]) * x ** self.estimation_parameters[1] / tol
                plt.plot(x,y,color='black')
                plt.scatter(self.data.keys(),self.data.values(),color='grey')
                plt.scatter([value],[search_results],color='black',s=30)
                plt.xscale("log")
                plt.yscale("log")
                plt.show()
            if search_results > lower_limit:
                print(str(search_results) + " > " + str(lower_limit))
                return True
            else:
                print(str(search_results) + " < " + str(lower_limit))
                return False
        
        elif self.style == 'manual':
            # Wähle einen zufälligen entwurf aus der liste aus
            entwurf = random.choice(entwurfe)
            if isinstance(entwurf, SCFunction) or isinstance(entwurf, Vocabulary):
                entwurf = [entwurf.root,entwurf.mapping[-1]]
            if isinstance(entwurf, list):
                entwurf = entwurf
            while True:
                antwort = input("Does " + proposal[0] + " mean " + str(proposal[1]) + "? (y/n): ").strip().lower()
                if antwort == "y":
                    return True
                elif antwort == "n":
                    return False
                else:
                    print("Please answer 'y' or 'n'.")
            
        elif self.style == 'omniscient':
            # wähle einen zufälligen entwurf aus der liste aus
            entwurf = random.choice(entwurfe)
            if isinstance(entwurf, SCFunction) or isinstance(entwurf, Vocabulary):
                entwurf = entwurf.sample().root
            if isinstance(entwurf, list):
                entwurf = entwurf[0]
            return entwurf in self.knowledge
        elif self.style == 'arithmetic' or self.style == 'dummy':
            #Since the code only make s arithmetically plausible proposals, everything is excepted
            return True

        elif self.style == 'hybrid':
            copy = Oracle('statistic',search_engine=self.search_engine, estimation_parameters=self.estimation_parameters,tolerance_factor = self.tolerance_factor, security_factor = self.security_factor)
            if copy.antwort(entwurfe,tol = copy.tolerance_factor / copy.security_factor):
                return True
            elif not copy.answer(proposal,tol = copy.tolerance_factor * copy.security_factor):
                return False
            else:
                copy = Oracle('manual')
                return copy.answer(proposal)
            
    def answer(self,proposal, tol = 0,show_plot=True):
        if self.style == 'statistic':
            if isinstance(proposal,str):
                raise TypeError("Statistic supervisor only answers to inputs SCFunction, Vocabulary, or [word, number]")
            if isinstance(proposal,Vocabulary):
                proposal = [proposal.root,proposal.mapping[-1]]
            if isinstance(proposal,list):
                value = proposal[1]
                search_results = self.search_engine(proposal[0])
            elif isinstance(proposal,SCFunction):
                proposals = proposal.all_outputs()
                l = len(proposals)
                if l > 20:
                    l = 20
                    proposals = random.sample(proposals,20)
                value = sum([p.mapping[-1] for p in proposals]) / l
                search_results = sum([self.search_engine(p.root) for p in proposals]) / l
            if tol == 0:
                tol = self.tolerance_factor
            lower_limit = np.exp(self.estimation_parameters[0]) * value ** self.estimation_parameters[1] / tol
            if search_results == 0:
                print("0 < " + str(round(lower_limit,1)))
                return False
            if show_plot:
                x = np.arange(min(self.data.keys()),max(list(self.data.keys())+[value]))
                y = np.exp(self.estimation_parameters[0]) * x ** self.estimation_parameters[1] / tol
                plt.plot(x,y,color='black')
                plt.scatter(self.data.keys(),self.data.values(),color='grey')
                plt.scatter([value],[search_results],color='black',s=30)
                plt.xscale("log")
                plt.yscale("log")
                plt.show()
            if search_results > lower_limit:
                print(str(search_results) + " > " + str(lower_limit))
                return True
            else:
                print(str(search_results) + " < " + str(lower_limit))
                return False
        
        elif self.style == 'manual':
            if isinstance(proposal, SCFunction) or isinstance(proposal, Vocabulary):
                proposal = [proposal.root,proposal.mapping[-1]]
            if isinstance(proposal, list):
                proposal = proposal
            while True:
                antwort = input("Does " + proposal[0] + " mean " + str(proposal[1]) + "? (y/n): ").strip().lower()
                if antwort == "y":
                    return True
                elif antwort == "n":
                    return False
                else:
                    print("Please answer 'y' or 'n'.")
            
        elif self.style == 'omniscient':
            if isinstance(proposal, SCFunction) or isinstance(proposal, Vocabulary):
                proposal = proposal.sample().root
            if isinstance(proposal, list):
                proposal = proposal[0]
            return proposal in self.knowledge
        elif self.style == 'arithmetic' or self.style == 'dummy':
            #Since the code only make s arithmetically plausible proposals, everything is excepted
            return True

        elif self.style == 'hybrid':
            copy = Oracle('statistic',search_engine=self.search_engine, estimation_parameters=self.estimation_parameters,tolerance_factor = self.tolerance_factor, security_factor = self.security_factor)
            if copy.answer(proposal,tol = copy.tolerance_factor / copy.security_factor):
                return True
            elif not copy.answer(proposal,tol = copy.tolerance_factor * copy.security_factor):
                return False
            else:
                copy = Oracle('manual')
                return copy.answer(proposal)
            
    def update(self,new_confirmed_words):
        if type(new_confirmed_words) != list or not all([isinstance(w,SCFunction) or isinstance(w,Vocabulary) for w in new_confirmed_words]):
            raise TypeError('Update requires a list of SCFunctions, Vocabularys, or [word, number]-lists as input.')
        for w in new_confirmed_words:
            if isinstance(w,Vocabulary) or isinstance(w,SCFunction):
                w = [w.root,w.mapping[-1]]
            self.data[w[1]] = self.search_engine(w[0])
        data = [[i,self.data[i]] for i in self.data.keys()]
        self.estimation_parameters = bayesian_regression(data)
        
def bayesian_regression(data):
    # Bestimme Koeffizienten a, b sodass die ZWHäufigkeit Y durch den Zahlwert N mit a*N^b angenähert werden kann.
    # Dabei wird davon ausgegangen dass b ca -1.6503571428571429 mit Varianz 0.5683887566137565 ist
    exp_guess = -1.6503571428571429
    exp_uncert2 = 0.5683887566137565
    freq_unaccuracy2 = np.log(100)**2
    
    A = np.array([[len(data), sum([np.log(dat[0] + 0.01) for dat in data])],
                  [sum([np.log(dat[0] + 0.01) / freq_unaccuracy2 for dat in data]), sum([np.log(dat[0] + 0.01)**2 / freq_unaccuracy2 for dat in data]) + 1/exp_uncert2]])

    c = np.array([sum([np.log(dat[1] + 0.01) for dat in data]), sum([np.log(dat[0] + 0.01) * np.log(dat[1] + 0.01) / freq_unaccuracy2 for dat in data]) + exp_guess/exp_uncert2])
    a,b = np.linalg.solve(A,c)
    return [a,b]
            

def intlinsolve(base,image):
    print('base: ',base)
    print('image: ',image)
    try:
        solution = list(solve(base,image)[0])
        print('solution is ',solution)
        return solution+[0]
    except IndexError:
        print('constant needed')
        erweiterte_basis = [list(b)+[1] for b in base]
        print(erweiterte_basis)
        print(image)
        try:
            return solve(erweiterte_basis,image)[0]
        except NotImplementedError:
            print('unique solution')
            #return [round(i) for i in np.dot(np.linalg.pinv(np.array([b+[1] for b in base], dtype=np.float64),rcond=1e-15),image)]
            return [i for i in np.dot(np.linalg.pinv(np.array(erweiterte_basis, dtype=np.float64)),image)]
        except IndexError:
            print('unique solution')
            #return [round(i) for i in np.dot(np.linalg.pinv(np.array([b+[1] for b in base], dtype=np.float64),rcond=1e-15),image)]
            return [i for i in np.dot(np.linalg.pinv(np.array(erweiterte_basis, dtype=np.float64)),image)]
    except NotImplementedError:
        print('unique solution')
        return [round(i) for i in np.dot(np.linalg.pinv(np.array(base, dtype=np.float64)),image)]+[0]

def cartesian_product(listlist):
    if len(listlist)==0:
        #print('ERROR: empty input in product')
        return [[]]
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

def get_numeral(number,language,lexicon=[]):
    try:
        numeral = num2words(number,lang=language)
        return numeral
    except:
        try:
            lanu=pd.read_csv(r'Numeral.csv', encoding = "utf_16", sep = '\t')
            df=lanu[lanu['Language']==language]
            biscriptual=False
            if ' ' in df.iloc[0,2]:
                biscriptual=True
            numeral=df.iloc[number-1,2]
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
            return numeral
        except:
            lanu=pd.read_csv(r'DVNum.csv', encoding = "utf_8", sep = ';')
            df=lanu[lanu['Language']==language]
            numeral=df.iloc[number-1,2]
            if numeral[0]==' ':
                numeral=numeral[1:]
            if numeral[-1]==' ':
                numeral=numeral[:-1]
            numeral = unicodedata.normalize('NFC',numeral)
            return numeral    

def proto_parse(number,numeral,lexicon,print_documentation,print_result): #parse a (int number,str numeral)-pair using (current) lexicon 'lexicon'. boolean print_documentation toggles documentation printout. boolean print_result toggles result printout
    #print('parse '+numeral)
    if print_documentation: print('parse '+numeral+' '+str(number))
    lex1 = []
    if len(lexicon) != 0 and isinstance(lexicon[0],Vocabulary):
        lex1 = lexicon
    else:
        for entry in lexicon:
            if isinstance(entry,Vocabulary):
                lexicon += [entry]
            else: 
                lex1 += entry.all_outputs_as_voc()
    lex1=lex1+[Vocabulary(number,numeral)] # so the new word itself is found at the end
    checkpoint=0 # point from which parsing is finally performed already
    highlights=[] #list of highlights, initially empty
    for end in range(len(numeral)+1): #set end of the observed substring
        startrange=range(checkpoint,end) #start of observed string may lie between checkpoint and end
        for highlight in highlights:
            startrange=set(startrange)-set(highlight.hlrange()) # observed strings may not start inside present highlights. rather they have to fully contain a highlight or be disjoint with it
        startrange=sorted(list(startrange)) # so ints in startrange are sorted by size
        for start in startrange: # set start of observed substring of numeral
            subnum_found_at_this_end=False # boolean condition to break start-loop
            substring=numeral[start:end] #set observed substring
            if print_documentation: print('substring: ',substring)
            for entry in lex1: #browse current lexicon
                if entry.word==substring: #look if substring appears
                    subnum_found_at_this_end=True
                    if 2*entry.number<number: #highlighting condition
                        if print_documentation: print(substring+' <' + str(entry.number) + '/2')
                        #highlights=[highlight for highlight in highlights if not highlight.start>=start]# if a highlight is contained in new highlight, then remove it from list of highlights
                        for highlight in highlights[:]: #browse through present highlights
                            if highlight.start>=start: # if a highlight is contained in new highlight,...
                                if print_documentation: print("remove "+highlight.numeral)
                                highlights.remove(highlight) #then remove it from list of highlights
                        highlights=highlights+[Highlight(entry,start)] # add new highlight
                        if print_documentation: print('Unpacked: ['+','.join([str(highlight.numeral) for highlight in highlights])+']')
                    else:
                        if print_documentation: print(substring+' ≥' + str(entry.number) + '/2')
                        checkpoint=end
                        if print_documentation: print("Set checkpoint behind "+numeral[:checkpoint])
                    break # out of browsing the lexicon
            if subnum_found_at_this_end:
                break # out of start-loop
    root=numeral
    for highlight in reversed(highlights):
        root=root[0:highlight.start]+'_'+root[highlight.end():len(root)]
    decompstr=str(number)+'='+root+'('
    decompstr=decompstr+','.join([str(highlight.number) for highlight in highlights])
    #for highlight in highlights:
    #    decompstr=decompstr+str(highlight.number)+','
    decompstr=decompstr+')'
    if print_result: print(decompstr)
    return SCFunction(root,[[Vocabulary(highlight.number,highlight.numeral)] for highlight in highlights],[0 for highlight in highlights]+[number])

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
            if print_doc: print('substring = '+str(substr))
            for entry in lexicon1:
                if substr == entry.word:
                    subnum_found_at_this_end=True
                    subentry_found = False
                    if 2*entry.number < number or mult_found:
                        if print_doc: print(substr+' is <' + str(number) + '/2')
                        for highlight in reversed(highlights):
                            if highlight.start >= start:
                                if print_doc: print("remove "+highlight.numeral)
                                highlights.remove(highlight)
                        highlights=highlights+[Highlight(Vocabulary(entry.number,entry.word),start)]
                        if print_doc: print('Unpacked: ',[highlight.numeral for highlight in highlights])
                    else: 
                        if print_doc: print(substr+" is ≥" + str(number) + '/2')
                        mult_found=True
                        checkpoint=end
                        potential_highlight = None
                        earliest_laterstart = start+1
                        for highlight in highlights:
                            if highlight.number**2 < entry.number:
                                earliest_laterstart = min(end,highlight.end()) #so factors remain untouched
                        potential_highlight = None
                        for laterstart in range(earliest_laterstart,end):
                            if print_doc: print('subnum = '+word[laterstart:end])
                            for subentry in lexicon1:
                                if word[laterstart:end] == subentry.word:
                                    if subentry.number**2 <= entry.number:
                                        if print_doc: 
                                            print(word[laterstart:end]+" is <sqrt("+str(entry.number)+")") #print(word[laterstart:end]+" is FAC or SUM. If it would contain mult, its square would be larger than "+entry.word+'.')
                                            if potential_highlight:
                                                print("Ignore " + potential_highlight.word + " because " + word[laterstart:end] + " is its subnumeral.")
                                        subentry_found=True
                                        #for highlight in reversed(highlights):
                                            #if highlight.end() > laterstart:
                                                #if print_doc: print("remove "+highlight[0])
                                                #highlights.remove(highlight)
                                        #highlights=highlights+[Highlight(Vocabulary(subentry.number,subentry.word),laterstart)]
                                        #checkpoint = laterstart
                                        potential_highlight = Highlight(Vocabulary(subentry.number,subentry.word),laterstart)
                                        potential_checkpoint = laterstart
                                        if print_doc: print('Unpacked: ',[highlight.numeral for highlight in highlights])
                                    else:
                                        if entry.number % subentry.number != 0 and 2*subentry.number < number:
                                            if print_doc: 
                                                print(word[laterstart:end]+" is at least <" + str(number) + "/2 and is no divisor of " + entry.word + ".") #print(word[laterstart:end]+" probably contains SUM. As "+subentry.word+' is no divisor of '+entry.word+', '+entry.word+' has to contain SUM. '+subentry.word+' cannot contain FAC*MULT, as it is smaller than half of '+entry.word+': And it cannot be FAC, as its square is larger than '+entry.word+'. So it is composed of SUM and MULT. If it turns out to be irreducible with the present properties, we assume it is SUM')
                                                if potential_highlight:
                                                    print("Ignore " + potential_highlight.word + " because " + word[laterstart:end] + " is its subnumeral.")
                                            potential_highlight = Highlight(Vocabulary(subentry.number,subentry.word),laterstart)
                                            potential_checkpoint = laterstart
                                        elif entry.number % subentry.number == 0:
                                            if print_doc: print(str(subentry.number) + " is divisor of " + str(entry.number) + " and is ≥sqrt(" + str(entry.number) + ").")
                                            potential_highlight = None
                                        elif 2*subentry.number >= number:
                                            if print_doc: print(str(subentry.number) +" is at least <" + str(number) + "/2")
                                            potential_highlight = None
                            if subentry_found:
                                break  
                        if potential_highlight != None:
                            for highlight in reversed(highlights):
                                if highlight.end() > potential_checkpoint:
                                    if print_doc: print("remove "+highlight.root)
                                    highlights.remove(highlight)
                            highlights = highlights+[potential_highlight]
                            checkpoint = potential_checkpoint
                            if print_doc: print('Unpacked: ',[highlight.root for highlight in highlights])
                        if print_doc: print("Set checkpoint behind "+word[:checkpoint])
                    break                    
            if subnum_found_at_this_end:
                break
    #print('Unpacked subnums: ',len(highlights))
    if len(highlights) == 2:
        if highlights[0].number + highlights[1].number == number:
            sohi = sorted(highlights, key=lambda highlight: highlight.number)
            suspected_mult = sohi[-1]
            if print_doc: print("remove "+suspected_mult.root+' because ' + sohi[0].root + ' + ' + sohi[1].root + " = " + word + " and "+ sohi[0].root + ' > ' + sohi[1].root + "so it is probably mult.")
            highlights.remove(suspected_mult)
        elif highlights[0].number * highlights[1].number == number:
            sohi = sorted(highlights, key=lambda highlight: highlight.number)
            suspected_mult = sohi[-1]
            if print_doc: print("remove "+suspected_mult.root+' because ' + sohi[0].root + ' * ' + sohi[1].root + " = " + word + " and "+ sohi[0].root + ' > ' + sohi[1].root + "so it is probably mult.")
            highlights.remove(suspected_mult)
    elif len(highlights) == 3:
        sohi = sorted(highlights, key=lambda highlight: highlight.number)
        suspected_mult = sohi[-1]
        #print('suspected mult: ', suspected_mult.root)
        if suspected_mult.number**2 > number:
            #other_numbers = [highlight.number for highlight in highlights if highlight != suspected_mult]
            other_numbers = [highlight for highlight in highlights if highlight != suspected_mult]
            if other_numbers[0].number * suspected_mult.number + other_numbers[1].number == number:
                if print_doc: print("remove "+suspected_mult.root+ " because " + other_numbers[0].root + " * " + suspected_mult.root + " + " + other_numbers[1].root + " = " + word + " and " + suspected_mult.root + " > " + other_numbers[0].root + " so it is probably mult.")
                highlights.remove(suspected_mult)
            elif other_numbers[1].number * suspected_mult.number + other_numbers[0].number == number:
                if print_doc: print("remove "+suspected_mult.root+ " because " + other_numbers[1].root + " * " + suspected_mult.root + " + " + other_numbers[0].root + " = " + word + " and " + suspected_mult.root + " > " + other_numbers[1].root + " so it is probably mult.")
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

def list_scfunctions(language,printb=True,set_limit=999999):
    if printb: print('Parse numerals in '+language)
    lex=[entry for entry in create_lexicon(language) if entry.number<set_limit]
    limit = len(lex)
    if limit > 1000:
        limit = 999999
    set_of_scfunctions=[]
    irreducibles=[]
    for entry in lex:
        parse=advanced_parse(entry.number,entry.word,lex,False,False)
        if '_' in parse.root:
            function_known=False
            for pos in range(len(set_of_scfunctions)):
                if set_of_scfunctions[pos].root == parse.root:
                    set_of_scfunctions[pos]=set_of_scfunctions[pos].merge(parse)
                    #print(set_of_scfunctions[pos].root+' now has '+str(len(set_of_scfunctions[pos].inputrange))+' inputs.')
                    function_known=True
                    break
            if not function_known:
                set_of_scfunctions=set_of_scfunctions+[parse]
        else:
            irreducibles += [parse]
    if printb: print(language+' has '+str(len(set_of_scfunctions))+' numeral functions and '+str(len(irreducibles))+' irreducible numerals when using the advanced numeral decomposer for numerals till '+ str(limit) +'.')
    if printb: print('The number functions are:')
    printout=f""
    for scf in set_of_scfunctions:
        example=scf.sample()
        printout=printout + scf.present(printout=False,domain=True)+', \t e.g. '+str(example.mapping[-1])+' is '+str(example.root)+'\n'
        #+scf.root+':\t x -> '+str([round(coeff) for coeff in scf.mapping[:-1]])+'*x+'+str(round(scf.mapping[-1]))
    if printb: print(printout)
    if printb: print('The irreducibles are: ' + ', '.join([scf.root+' ('+str(scf.mapping[-1])+')' for scf in irreducibles]))
    if printb: print('')
    
    # show clusters
    #for scf in set_of_scfunctions:
        #print(scf.root)
        #print([o.mapping[-1] for o in scf.all_outputs()])
    return len(set_of_scfunctions) + len(irreducibles)
    
def old_list_scfunctions(language,printb=True,set_limit=999999):
    #print('Trying old parser')
    lex=[entry for entry in create_lexicon(language) if entry.number<set_limit]
    limit = len(lex)
    if limit > 1000:
        limit = 999999
    set_of_scfunctions=[]
    irreducible_count=0
    for entry in lex:
        parse=proto_parse(entry.number,entry.word,lex,False,False)
        if '_' in parse.root:
            function_known=False
            for pos in range(len(set_of_scfunctions)):
                if set_of_scfunctions[pos].root == parse.root:
                    set_of_scfunctions[pos]=set_of_scfunctions[pos].merge(parse)
                    #print(set_of_scfunctions[pos].root+' now has '+str(len(set_of_scfunctions[pos].inputrange))+' inputs.')
                    function_known=True
                    break
            if not function_known:
                set_of_scfunctions=set_of_scfunctions+[parse]
        else:
            irreducible_count=irreducible_count+1
    #print('The old parser had structured '+language+' into '+str(len(set_of_scfunctions))+' number functions and '+str(irreducible_count)+' irreducible numbers.')
    #print('')
    if printb: print(language+' has '+str(len(set_of_scfunctions))+' numeral functions and '+str(irreducible_count)+' irreducible numerals when using the prototype numeral decomposer for numerals till '+ str(limit) +'.')
    if printb: print('The number functions are:')
    printout=f""
    for scf in set_of_scfunctions:
        example=scf.sample()
        printout=printout + scf.present(False,False)+', \t e.g. '+str(example.mapping[-1])+' is '+str(example.mapping[-1])+'\n'
        #scf.root+':\t x -> '+str([round(coeff) for coeff in scf.mapping[:-1]])+'*x+'+str(round(scf.mapping[-1]))
            
    if printb: print(printout)
    return len(set_of_scfunctions) + irreducible_count

def decompose_numeral(number,language,version='new'):
    lex = create_lexicon(language)
    for entr in list(reversed(lex))[-number:]:
        if entr.number == number:
            word = entr.word
            break
    if version == 'new':
        advanced_parse(number,word,create_lexicon(language),True,True)
    elif version == 'old':
        proto_parse(number,word,create_lexicon(language),True,True)
            
def decompose_numeral_custom(number,word,lexicon,version='new'):
    if isinstance(lexicon,str):
        try:
            lex = create_lexicon(lexicon)
        except:
            raise NotImplementedError("Assuming " + language + " is a language, it is not supported. Use a custom lexicon to test the decomposer on it")
    elif isinstance(lexicon,list):
        if len(lexicon)==0:
            lex = lexicon
        elif isinstance(lexicon[0],list):
            try:
                lex = [Vocabulary(l[0],l[1]) for l in lexicon]
            except:
                raise TypeError("Custom lexicon must be a list of Vocabulary objects or of (number,numeral) pairs, e.g. [[1,'one'],[2,'two']], ")
        elif isinstance(lexicon[0],Vocabulary):
            lex = lexicon
        else:
            raise TypeError("Custom lexicon must be a list of Vocabulary objects or of (number,numeral) pairs, e.g. [[1,'one'],[2,'two']], ")
                    
    else:
        raise TypeError("Lexicon must be a string representing a language or a list serving as a custom lexicon")
    if version == 'new':
        advanced_parse(number,word,lex,True,True)
    elif version == 'old':
        proto_parse(number,word,lex,True,True)

def grammar_parse(word,lexicon):
    #start = time.time()
    for scf in lexicon:
        rootwords = scf.root.split('_')
        #print(rootwords)
        if all([w in word for w in rootwords]):
            cut_scf = SCFunction(scf.root, [[f for f in comp if all([w in word for w in f.root.split('_')])] for comp in scf.inputrange],scf.mapping)
            for ou in cut_scf.all_outputs(only_confirmed = True):
                if ou.root == word:
                    #end = time.time()
                    #print('Parsing took ' + str(end-start) + ' seconds')
                    return ou.mapping[-1]
    for scf in lexicon:
        rootwords = scf.root.split('_')
        #print(rootwords)
        if all([w in word for w in rootwords]):
            cut_scf = SCFunction(scf.root, [[f for f in comp if all([w in word for w in f.root.split('_')])] for comp in scf.inputrange],scf.mapping)
            for ou in cut_scf.all_outputs():
                if ou.root == word:
                    #end = time.time()
                    #print('Parsing took ' + str(end-start) + ' seconds')
                    return ou.mapping[-1]
    #end = time.time()
    #print('Parsing took ' + str(end-start) + ' seconds')
    return -1

def grammar_generate(number,lexicon):
    for scf in lexicon:
        if all([c >= 0 for c in scf.mapping]):
            mini = sum(scf.mapping)
            maxi = (sum(scf.mapping) - 1)**2
        else:
            mini = sum(scf.mapping) / 2
            maxi = (sum([max(0,c) for c in scf.mapping]))**2
        if number > mini and number < maxi:
            for ou in scf.all_outputs():
                if ou.mapping[-1] == number:
                    return ou.root
    return ''

#%run numeral_decomposition_advanced.ipynb
def random_choice(my_dict):
    keys = list(my_dict.keys())
    weights = np.array(list(my_dict.values()), dtype=float)
    weights /= weights.sum()   # normalisieren zu Wahrscheinlichkeiten
    chosen = np.random.choice(keys, p=weights)
    return chosen

def shuffle_lexicon(lexicon):
    if not type(lexicon) == list:
        raise TypeError('Input lexicon must be of type list')
    #for entry in lexicon:
        #if not type(entry) in [__main__.Vocabulary, __main__.SCFunction]:
            #raise TypeError('All entries of the lexicon must be of type Vocabulary of SCFunction. The following entry ist not:' + str(entry))
    total_weight = sum([entry.number**(-2) if entry.number!=0 else 1 for entry in lexicon])
    numdict = {entry.number: entry.word for entry in lexicon}
    rounddict = {}
    for entry in lexicon:
        rounddict[entry.number] = entry.number
        length_of_rounded_word = len(entry.word)
        for nb in range(entry.number//2,2*entry.number):
            if nb in numdict.keys() and numdict[nb] in entry.word and len(numdict[nb]) < length_of_rounded_word:
                rounddict[entry.number] = nb
                length_of_rounded_word = len(numdict[nb])
    rounding_probability = 0.61
    freqdict = {
        entry.number: (1 - rounding_probability if entry.number == 0 else (1 - rounding_probability) * entry.number**(-2))
        for entry in lexicon
    }
    for nb in rounddict:
        if nb == 0:
            freqdict[rounddict[nb]] += rounding_probability
        else:
            freqdict[rounddict[nb]] += rounding_probability * nb**(-2)

    order = []
    while len(freqdict) != 0:
        choice = random_choice(freqdict)
        del freqdict[choice]
        order += [int(choice)]

    shuffled_lexicon = []
    for nb in order:
        shuffled_lexicon += [Vocabulary(nb,numdict[nb])]
    #for e in shuffled_lexicon:
        #e.printVoc()
    return shuffled_lexicon

def solve_lexicon_clashes(entry1, entry2):
    #determine all (confirmed) values of entry2
    # values2 icludes all input-output pairs
    values2 = [(inp,sum([inp[i]*entry2.mapping[i] for i in range(entry2.dimension())]) + entry2.mapping[-1]) for inp in cartesian_product(entry2.number_inputs())]
    #confirmed_values2 includes all confirmed outputs
    confirmed_values2 = [sum([inp[i]*entry2.mapping[i] for i in range(entry2.dimension())]) + entry2.mapping[-1] for inp in cartesian_product(entry2.number_inputs(only_confirmed = True))]

    defeated_values2 = []
    
    # rebuild inputrange of entry1
    new_inputrange = []
    for comp in range(entry1.dimension()):
        new_comp =[]
        for entr in entry1.inputrange[comp]:
            # check if entry is confirmed
            if entr in entry1.confirmed_inputrange[comp]:
                new_comp += [entr]
            #otherwise
            else:
                # build all number outputs caused by this entry
                inputs_involving_entr = [i for i in cartesian_product(entry1.number_inputs()) if i[comp] == entr.mapping[-1]]
                io = [(inp,sum([entry1.mapping[i]*inp[i] for i in range(entry1.dimension())]) + entry1.mapping[-1]) for inp in inputs_involving_entr]
                # check if any output clashes with a confirmed output of entry2
                if not any([ou[1] in confirmed_values2 for ou in io]):
                    # check if any output clashes with any output of entry2
                    clash_wins = 0
                    clash_losses = 0
                    clashing_values2 = []
                    for v in values2:
                        break_now = False
                        for ou in io:
                            if v[1] == ou[1]:
                                clashing_values2 += [v[1]]
                                input1 = ou[0]
                                input2 = v[0]
                                unconfirmed_components1 = [input1[i] for i in range(len(input1)) if input1[i] not in entry1.number_inputs(only_confirmed = True)[i]]
                                unconfirmed_components2 = [input2[i] for i in range(len(input2)) if input2[i] not in entry2.number_inputs(only_confirmed = True)[i]]
                                if sum(unconfirmed_components2) < sum(unconfirmed_components1):
                                    clash_losses += 1
                                elif sum(unconfirmed_components2) > sum(unconfirmed_components1):
                                    clash_wins += 1
                                elif sum(unconfirmed_components2) == sum(unconfirmed_components1):
                                    pass           
                                if abs(clash_wins - clash_losses) > 3:
                                    break_now = True
                                    break
                        if break_now:
                            break
                    if clash_wins >= clash_losses:
                        new_comp += [entr]
                        defeated_values2 += clashing_values2
        new_inputrange += [new_comp]   
    #defeated_values2 += [ou.mapping[-1] for ou in entry.all_outputs(only_comfirmed = True)]
    return SCFunction(entry1.root,new_inputrange,entry1.mapping,entry1.confirmed_inputrange)


def learn_lexicon(lexicon,orakel,initial_lexicon=[],version='a',printb=True,normalize=True):
    #if printb: print('Learning '+language)
    #supervisor = lexicon
    learnerlex = initial_lexicon
    samples = 0
    orakel_errors = []
    for voc in lexicon:
        #update statistical orakel based on data of new confirmed word
        if orakel.style in ['statistic','hybrid']:
            orakel.update([voc])

        #check if word is known already
        if (type(orakel) != list and orakel.style in ['dummy','arithmetic']) or not any([(voc.word,voc.number) in [(outp.root,outp.mapping[-1]) for outp in entr.all_outputs()] for entr in learnerlex]):

            #check if another word of voc's value has been learned before --> orakel error
            if orakel.style in ['statistic','hybrid'] and voc.mapping[-1] not in orakel.data.keys():
                for entr in learnerlex:
                    for o in entr.all_outputs:
                        if o.mapping[-1] == voc.mapping[-1]:
                            print("ORAKEL ERROR: " + str(voc.mapping[-1]) + " apparently means " + voc.word + " and not " + o.word)
                            orakel_errors += [voc.mapping[-1]]
                        
            samples += 1
            #1 parse new word
            #if printb: print('What means '+str(voc.number)+'?')
            if printb: print('Orakel: '+voc.word+' means '+str(voc.number))
            if version == 'a':
                parse = advanced_parse(voc.number, voc.word, learnerlex, False, printb) #[ou for e in learnerlex for ou in e.all_outputs()]
            elif version == 'p':
                parse = proto_parse(voc.number, voc.word, learnerlex, False, printb) #[ou for e in learnerlex for ou in e.all_outputs()]
            # Understood?
            understood = False
            functions_with_equal_template = [e for e in learnerlex if e.root == parse.root]
            if functions_with_equal_template != []:
                understood = True
                merger = None
                for entry in functions_with_equal_template:
                    if len(entry.number_inputs()) > 1:
                        merger = entry
                        functions_with_equal_template.remove(entry)
                        break
                if merger == None:
                    merger = functions_with_equal_template[0]
                    functions_with_equal_template = functions_with_equal_template[1:]
                gelernte_funktion = merger.vereinige(parse, functions_with_equal_template, printb=printb, trust_affinity=False) # versuche zu mergen
                if set(merger.all_outputs()).issubset(set(gelernte_funktion.all_outputs())): # wenn mergen effektiv war
                    gelernte_funktion = gelernte_funktion.verstarke(learnerlex,orakel,printb)

            if not understood:
                #2b No, just remembering
                gelernte_funktion = parse
            if printb: gelernte_funktion.present()

            updated_entries = [gelernte_funktion]

            # reinforce learnerlex with gelernte_funktion
            for i in range(len(learnerlex)):
                if sum(learnerlex[i].mapping) > sum(gelernte_funktion.mapping) and learnerlex[i].dimension() > 0: 
                    #if printb: print('Attempting to reinforce ' + learnerlex[i].root + ' with ' + learned.root)
                    new_entry = learnerlex[i].reinforce([gelernte_funktion],orakel,printb,aux_lex=learnerlex)
                else:
                    new_entry = learnerlex[i]
                if any([len(new_entry.inputrange[j]) > len(learnerlex[i].inputrange[j]) for j in range(new_entry.dimension())]):
                    updated_entries += [new_entry]
                #learnerlex[i] = new_entry
                #if printb: entry.present()

            if orakel.style == 'arithmetic':
                for entry in learnerlex:
                    gelernte_funktion = solve_lexicon_clashes(gelernte_funktion,entry)
                for entry in learnerlex:
                    entry = solve_lexicon_clashes(entry,gelernte_funktion)

            # remove redundant entries and add updated entries to learnerlexicon
            for ue in updated_entries:
                learned_outputs = ue.all_outputs()
                for entry in learnerlex:
                    # check if all confirmed outputs of entry are covered by learned 
                    all_covered = True
                    for ou in entry.all_outputs(only_confirmed = True):
                        covered = False
                        for lo in learned_outputs:
                            if [round(i) for i in ou.mapping] == [round(i) for i in lo.mapping] and ou.root == lo.root:
                                covered = True
                        if not covered:
                            all_covered = False
                            break
                    if all_covered:
                        if printb:
                            print('Update entry:')
                            entry.present()
                            print('-->')
                            ue.present()
                        learnerlex.remove(entry)
            
                # add learned to learnerlex
                learnerlex += [ue]                
            
            #print()
            #print('Entries: ')
            #for entry in learnerlex:
                #if printb: entry.present()
            #print()
    #reorganize all inputranges so that they only contain scfunctions, no vocabulary
    if normalize: learnerlex = normalize_scf_lexicon(learnerlex,printb)
    #if printb: print('Learned '+str(len(lexicon))+' words and structured them in '+str(len(learnerlex))+' functions.')
    #if printb: print('It took '+str(samples)+' samples to learn those.')
    if printb and orakel.style == 'statistic': print(str(len(orakel_errors)) + ' orakel errors occurred. They affected the numbers ' + ', '.join(orakel_errors) + '.')
    if printb: print('Those are:')
    for entry in learnerlex:
        if printb: 
            entry.present()
            #print('Confirmed form: ',end='')
            #entry.present(only_confirmed=True)
    if printb: print('')
    return learnerlex

def validate(lexicon, oracle):
    learned_words = []
    for e in lexicon:
        learned_words += [ou.root for ou in e.all_outputs()]
    mistake_found = False
    for ou in learned_words:
        if ou not in oracle:
            mistake_found = True
            print(ou, ' is wrong')
    for ou in oracle:
        if ou not in learned_words:
            mistake_found = True
            print(ou, ' is missing')
    if not mistake_found:
        print('No mistakes')


def normalize_scf_lexicon(learnerlex, printb= True):
    new_learnerlex = []
    for lexentr in learnerlex:
        if printb: lexentr.present()
        new_inputrange = []
        for comp in lexentr.inputrange:
            if len(comp) == 1:
                new_comp = [SCFunction(root=comp[0].root,i=comp[0].inputrange,mapping=comp[0].mapping)]
            else:
                new_comp = []
                for entr in comp:
                    representant_found = False
                    for scf in new_learnerlex + learnerlex:
                        if scf.root == entr.root:
                            new_comp += [scf]
                            representant_found = True
                            break
                    if not representant_found:
                        if printb: print('Need to check outputs')
                        #print(type(entr))
                        #print(isinstance(entr,SCFunction))
                        if isinstance(entr,Vocabulary):
                            pou = set([entr.word])
                            #parse = advanced_parse(entr.number,entr.word,learnerlex,False,False)
                        elif isinstance(entr,SCFunction):
                            pou = set([ou.root for ou in entr.all_outputs()])
                            #parse = entr
                        #print(pou)
                        #print('Compare with')
                        for scf in new_learnerlex + learnerlex:
                            sou = set([ou.root for ou in scf.all_outputs()])
                            #print(sou)
                            if pou.issubset(sou):
                                representant_found = True
                                if printb: print('In domain of ' + lexentr.root + ', ' + entr.root + ' can be covered by ' + scf.root)
                                if not scf in new_comp:
                                    new_comp += [scf]
                                break
                    if not representant_found and printb: print('No representant found for ' + entr.root)
            new_inputrange += [new_comp]
        new_learnerlex += [SCFunction(lexentr.root,new_inputrange,lexentr.mapping,lexentr.confirmed_inputrange)]
    return new_learnerlex
