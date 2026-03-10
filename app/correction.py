import re
import nltk
from nltk.corpus import words

################################################################################
#  load enlisg words
################################################################################
def get_words():
    try: 
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words')
    
    return set(words.words())


################################################################################
#  get casing of the word
################################################################################
def get_casing(regex_str):
    # input:
    #    regex_str - a string with ambiguous words inserted in format (a|b)
    #                   starts with ^ and ends with $
    # return:
    #    capital_indexes - a set of index that should be capitalized
        
    index = 0
    in_bracket_flag = False
    capital_indexes = set()
    for char in regex_str[1:-1]:
        if char == '(':
            in_bracket_flag = True
        elif char == ')':
            in_bracket_flag = False
            index += 1
        else:
            if char.isupper(): 
                capital_indexes.add(index)
            if not in_bracket_flag: 
                index += 1            
    
    return capital_indexes


################################################################################
#  restore case
################################################################################
def case_restore(word, casing):
    # input:
    #    word - a string witht the word in lowercase
    #    casing - a set of index that should be capitalized
    # return:
    #    formatted_word - word with appropriate casing
    
    formatted_word = ''
    for index, char in enumerate(word):
        if index in casing:
            formatted_word += char.upper()
        else: 
            formatted_word += char
    return formatted_word

################################################################################
#  find all words that match the given pattern
################################################################################
def find_matching_words(regex_str, word_set):
    # input:
    #    regex_str - a string with ambiguous words inserted in format (a|b)
    #                   starts with ^ and ends with $
    #    word_set - a set of all english words
    # return:
    #    matches - a list of matching words
    
    casing = get_casing(regex_str)    
    matches = []
    for word in word_set:
        if re.match(regex_str, word, re.IGNORECASE):
            matches.append(word)
    case_preserved_matches = [case_restore(word, casing) for word in matches]
    return case_preserved_matches

if __name__ == "__main__":
    words = get_words()
    
    pattern1 = "^L(i|e)n(i|e)$"
    result1 = find_matching_words(pattern1, words)
    print(f"Pattern '{pattern1}' → Most likely: {result1[0] if result1 else 'No match'}")
    
    pattern2 = "^(W|R)o(w|r)ld$"
    result2 = find_matching_words(pattern2, words)
    print(f"Pattern '{pattern2}' → Most likely: {result2[0] if result2 else 'No match'}")
    
    print("\nAll matches for each pattern:")
    for pattern in ["^L(i|e)n(i|e)$", "^(W|R)o(w|r)ld$"]:
        matches = find_matching_words(pattern, words)
        print(f"{pattern}: {matches}")