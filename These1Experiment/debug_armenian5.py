from alphabet_detector import AlphabetDetector

ad = AlphabetDetector()

test_cases = [
    'քսան k\'san',
    'քսան մեկ k\'san mek',
    'քսան երկու k\'san erkow',
]

for numeral in test_cases:
    print(f"Input: {repr(numeral)}")
    for point in range(len(numeral)+1):
        char = numeral[point:point+1]
        if char:
            is_latin = ad.is_latin(char)
            print(f"  point {point}: {repr(char)} is_latin={is_latin}")
            if is_latin:
                result = numeral[:point].strip()
                print(f"  Result: {repr(result)}")
                break
    print()
