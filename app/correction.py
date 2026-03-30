import re
import nltk
from nltk.corpus import words
from spellchecker import SpellChecker

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
#  get all candidate words
################################################################################
def generate_candidates(word):
    # input:
    #    word - a string containing the word
    # return:
    #    candidates - a list of possible words
    
    word_lower = word.lower()
    candidates = ["",]
    
    for char in word_lower:
        if char in SYM_PAIRS:
            fork = [string + SYM_PAIRS[char] for string in candidates]
            candidates = [string + char for string in candidates]
            candidates.extend(fork)
        else: 
            candidates = [string + char for string in candidates]
            
    return candidates


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
            formatted_word += char.lower()
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
    
    spell = SpellChecker()    
    regex_str = word_to_regex(raw_word)
    raw_candidates = generate_candidates(raw_word)
    # print(raw_candidates)
    casing = get_casing(raw_word)
    matches = []
    regex_matches = []
    
    word_len = len(raw_word)
    corr_candidate_list = list()
    for raw_candidate in raw_candidates:
        corr_candidates = spell.candidates(raw_candidate)
        if corr_candidates is None: 
            corr_candidates = [raw_candidate]
        for candidate in corr_candidates:
            if len(candidate) == word_len and candidate not in corr_candidate_list:
                corr_candidate_list.append(candidate)
    

    for word in word_set:
        if re.match(regex_str, word, re.IGNORECASE):
            regex_matches.append(word)
        if word in corr_candidate_list:
            matches.append(word)
        
    if len(matches) == 0:
        return [raw_word]
    
    if len(regex_matches) != 0: 
        case_preserved_matches = [case_restore(word, casing) for word in regex_matches]
    else: 
        case_preserved_matches = [case_restore(word, casing) for word in matches]
    return case_preserved_matches

if __name__ == "__main__":
    words = get_words()
    
    # pattern1 = "Helli"
    # result1 = find_matching_words(pattern1, words)
    # print(f"Pattern '{pattern1}' → Most likely: {result1[0] if result1 else 'No match'}")
    
    # pattern2 = "World"
    # result2 = find_matching_words(pattern2, words)
    # print(f"Pattern '{pattern2}' → Most likely: {result2[0] if result2 else 'No match'}")
    
    print("\nAll matches for each pattern:")
    for pattern in "Hillo Rorld This is a ner leni".split():
        matches = find_matching_words(pattern, words)
        print(f"{pattern}: {matches}")