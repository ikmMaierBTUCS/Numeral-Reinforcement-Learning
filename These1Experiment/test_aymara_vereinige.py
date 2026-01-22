"""
Test file to reproduce the vereinige() problem with Aymara numerals.
This sets up the lexicon in a specific order to trigger the issue.
"""
from funktionen import *

# Exact order that triggers the problem
target_order = [1, 2, 3, 4, 5, 31, 9, 7, 6, 10, 55, 21, 38, 16, 8, 25, 67, 15, 40, 23, 12, 13, 14, 24, 28, 35, 50, 52, 11, 34, 57, 33, 65, 22, 17, 43, 32, 96, 80, 19, 37, 39, 63, 26, 29, 49, 48, 71, 20, 18, 58, 59, 72, 36, 42, 90, 62, 69, 53, 97, 41, 84, 51, 46, 27, 60, 45, 30, 68, 47, 81, 56, 94, 89, 99, 95, 74, 44, 64, 61, 77, 86, 83, 54, 78, 79, 82, 73, 93, 66, 70, 85, 98, 88, 75, 76, 91, 92, 87]

# Load full Aymara lexicon
language = 'Aymara'
full_lexicon = create_lexicon(language, set_limit=100)

# Create a mapping from number to Vocabulary object
number_to_voc = {v.number: v for v in full_lexicon}

# Reorder the lexicon according to target_order
total_lexicon = [number_to_voc[n] for n in target_order if n in number_to_voc]

print(f"Total lexicon size: {len(total_lexicon)}")
print(f"Lexicon order: {[v.number for v in total_lexicon]}")

# Setup Oracle and knowledge
knowledge = {v.mapping[-1]: v.root for v in total_lexicon}
orakel = Oracle('omniscient', knowledge=list(knowledge.values()))

# Initialize learning
learning_range = 100
step_width = 10
learner_lex = []

print(f"\nStarting learning for {language}...")
print(f"Knowledge size: {len(knowledge)}")

# Learning loop that might trigger the vereinige() problem
for training_size in range(len(knowledge) + 1):
    if training_size % step_width == 0 or training_size == learning_range - 1:
        print(f"\n--- Training size: {training_size} ---")
        
        # Cut lexicon
        lexicon = total_lexicon[training_size - step_width : training_size]
        print(f"Training with numerals: {[v.number for v in lexicon]}")
        
        try:
            # Learn
            learner_lex = learn_lexicon(lexicon, orakel, initial_lexicon=learner_lex, printb=False, normalize=True)
            print(f"Learner lexicon size: {len(learner_lex)}")
            
            # Print learned functions
            for func in learner_lex:
                func.present()
                
        except Exception as e:
            print(f"ERROR during learning: {type(e).__name__}: {e}")
            print(f"Current learner_lex size: {len(learner_lex)}")
            import traceback
            traceback.print_exc()
            break

print("\nTest completed.")
