import re
import nltk
from nltk.corpus import words
import wordfreq
import sys

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
             'd': 'f',
             'f': 'd', 
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
#  find word distance
################################################################################
def calc_word_dist(word1, word2):
    # input:
    #    word1 - a string 
    #    word2 - a string
    # return:
    #    word_dist - custom edit distance between the two words
    
    # if words are of different length, define them as inf. far apart
    if len(word1) != len(word2):
        return sys.maxsize
    
    word_dist = 0
    for char1, char2 in zip(word1, word2):
        if char1 != char2: 
            if char1 in SYM_PAIRS and SYM_PAIRS[char1] == char2:
                word_dist += 0.5
            else:
                word_dist += 1
    return word_dist


################################################################################
#  find all words that match the given pattern
################################################################################
def find_matching_words(raw_word, word_set):
    # input:
    #    raw_word - a string containing the original word
    #    word_set - a set of all english words
    # return:
    #    matches - a list of matching words
    
    if raw_word.isnumeric(): 
        return [raw_word]
    
    regex_str = word_to_regex(raw_word)
    casing = get_casing(raw_word)
    edit_matches = []
    regex_matches = []
    
    min_dist = 2
    for word in word_set:
        # regex matching
        if re.match(regex_str, word, re.IGNORECASE):
            freq = wordfreq.zipf_frequency(word, "en")
            regex_matches.append((freq, word))
        
        # edit distance matching
        dist = calc_word_dist(raw_word.lower(), word)
        if dist < min_dist:
            freq = wordfreq.zipf_frequency(word, "en")
            edit_matches = [(freq, word),]
            min_dist = dist
        elif dist == min_dist:
            freq = wordfreq.zipf_frequency(word, "en")
            edit_matches.append((freq, word))
    
    # sort matches by frequency
    regex_matches.sort(reverse=True)
    edit_matches.sort(reverse=True)
    
    # if no matches in either case, return original word
    if len(edit_matches) == 0 and len(regex_matches) == 0:
        return [raw_word]
    
    # prioritize returning regex matches, then edit distance matches
    if len(regex_matches) != 0: 
        case_preserved_matches = [case_restore(word, casing) for _, word in regex_matches]
    else: 
        case_preserved_matches = [case_restore(word, casing) for _, word in edit_matches]
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
    for pattern in "Hella Rorld This is a ner lixi".split():
        matches = find_matching_words(pattern, words)
        print(f"{pattern}: {matches}")