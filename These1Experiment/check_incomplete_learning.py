"""
Analyze lernverlauf.txt to find languages where learning did not complete to [99, 0]
"""
import ast

with open(r"lernverlauf.txt", 'r') as f:
    content = f.read()

data = ast.literal_eval(content)

print("Languages where final learning step is NOT [99, 0]:\n")

incomplete_languages = []
for language in sorted(data.keys()):
    # Get the last element of the last inner list
    last_result = data[language][-1][-1]
    
    if last_result != [99, 0]:
        incomplete_languages.append((language, last_result))
        print(f"{language}: {last_result}")

print(f"\nTotal incomplete: {len(incomplete_languages)}/{len(data)}")
print(f"\nIncomplete languages: {[lang for lang, _ in incomplete_languages]}")
