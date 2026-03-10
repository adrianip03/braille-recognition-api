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

SYM_PAIRS = {'i': 'e',
             'e': 'i', 
             'w': 'r',
             'r': 'w',
             'h': 'j',
             'j': 'h'}

################################################################################
#  format word into regex form
################################################################################
def word_to_regex(word):
    # input:
    #    word - a string containing the word
    # return:
    #    regex_str - a string with ambiguous words inserted in format (a|b)
    #                   starts with ^ and ends with $
    word_lower = word.lower()
    regex_str = "^"
    for char in word_lower:
        if char in SYM_PAIRS:
            regex_str += f"({char}|{SYM_PAIRS[char]})"
        else: 
            regex_str += f"{char}"
    regex_str += "$"
    return regex_str


################################################################################
#  get casing of the word
################################################################################
def get_casing(word):
    # input:
    #    word - a string containing the word
    # return:
    #    capital_indexes - a set of index that should be capitalized

    capital_indexes = set()
    for index, char in enumerate(word):
        if char.isupper():
            capital_indexes.add(index)
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
def find_matching_words(raw_word, word_set):
    # input:
    #    raw_word - a string containing the original word
    #    word_set - a set of all english words
    # return:
    #    matches - a list of matching words
    
    regex_str = word_to_regex(raw_word)
    casing = get_casing(raw_word)
    matches = []
    for word in word_set:
        if re.match(regex_str, word, re.IGNORECASE):
            matches.append(word)
    if len(matches) == 0:
        return [raw_word]
    case_preserved_matches = [case_restore(word, casing) for word in matches]
    return case_preserved_matches

if __name__ == "__main__":
    words = get_words()
    
    pattern1 = "Helli"
    result1 = find_matching_words(pattern1, words)
    print(f"Pattern '{pattern1}' → Most likely: {result1[0] if result1 else 'No match'}")
    
    pattern2 = "World"
    result2 = find_matching_words(pattern2, words)
    print(f"Pattern '{pattern2}' → Most likely: {result2[0] if result2 else 'No match'}")
    
    print("\nAll matches for each pattern:")
    for pattern in ["Line", "World"]:
        matches = find_matching_words(pattern, words)
        print(f"{pattern}: {matches}")