"""
Word correction module for OCR/Braille recognition.
Handles character substitutions and spelling corrections.
"""

import re
import nltk
from nltk.corpus import words
import wordfreq
import sys

# Download NLTK words dataset if not available
def get_words():
    """Load and return set of English words from NLTK."""
    try: 
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words')
    return set(words.words())

# Common character substitution pairs
# Used to correct common OCR/Braille recognition errors
SYM_PAIRS = {
    'i': 'e', 'e': 'i',   # Common confusion pairs
    'd': 'f', 'f': 'd',
    'w': 'r', 'r': 'w',
    'h': 'j', 'j': 'h'
}

def word_to_regex(word):
    """
    Convert a word to regex pattern with ambiguous character handling.
    
    Args:
        word: Input string containing the word
        
    Returns:
        regex_str: Regex pattern with ^ and $ anchors, and (a|b) for ambiguous chars
    """
    word_lower = word.lower()
    regex_str = "^"
    for char in word_lower:
        if char in SYM_PAIRS:
            regex_str += f"({char}|{SYM_PAIRS[char]})"
        else: 
            regex_str += f"{char}"
    regex_str += "$"
    return regex_str

def generate_candidates(word):
    """
    Generate all possible word variations based on character pairs.
    
    Args:
        word: Input string containing the word
        
    Returns:
        candidates: List of possible word variations
    """
    word_lower = word.lower()
    candidates = ["",]
    
    for char in word_lower:
        if char in SYM_PAIRS:
            # Fork into two possibilities: original char and its substitute
            fork = [string + SYM_PAIRS[char] for string in candidates]
            candidates = [string + char for string in candidates]
            candidates.extend(fork)
        else: 
            candidates = [string + char for string in candidates]
    return candidates

def get_casing(word):
    """
    Extract capitalization pattern from word.
    
    Args:
        word: Input string
        
    Returns:
        capital_indexes: Set of indices that should be capitalized
    """
    capital_indexes = set()
    for index, char in enumerate(word):
        if char.isupper():
            capital_indexes.add(index)
    return capital_indexes

def case_restore(word, casing):
    """
    Restore original capitalization pattern to a lowercase word.
    
    Args:
        word: Lowercase word string
        casing: Set of indices that should be capitalized
        
    Returns:
        formatted_word: Word with restored capitalization
    """
    formatted_word = ''
    for index, char in enumerate(word):
        if index in casing:
            formatted_word += char.upper()
        else: 
            formatted_word += char.lower()
    return formatted_word

def calc_word_dist(word1, word2):
    """
    Calculate custom edit distance between two words.
    Substitutions from SYM_PAIRS count as half (0.5) instead of full (1).
    
    Args:
        word1, word2: Two strings to compare
        
    Returns:
        word_dist: Distance value (infinity if different lengths)
    """
    if len(word1) != len(word2):
        return sys.maxsize
    
    word_dist = 0
    for char1, char2 in zip(word1, word2):
        if char1 != char2: 
            # Check if it's a known substitution pair
            if char1 in SYM_PAIRS and SYM_PAIRS[char1] == char2:
                word_dist += 0.5  # Partial penalty for known confusions
            else:
                word_dist += 1
    return word_dist

def find_matching_words(raw_word, word_set):
    """
    Find best matching English words for a given input word.
    Uses regex matching first, then edit distance with frequency ranking.
    
    Args:
        raw_word: Input word (may contain typos)
        word_set: Set of valid English words
        
    Returns:
        matches: List of best matching words (preserves original casing)
    """
    # Return numeric values as-is
    if raw_word.isnumeric(): 
        return [raw_word]
    
    regex_str = word_to_regex(raw_word)
    casing = get_casing(raw_word)
    edit_matches = []
    regex_matches = []
    
    min_dist = 2
    for word in word_set:
        # Regex matching for direct character substitutions
        if re.match(regex_str, word, re.IGNORECASE):
            freq = wordfreq.zipf_frequency(word, "en")
            regex_matches.append((freq, word))
        
        # Edit distance matching for more complex errors
        dist = calc_word_dist(raw_word.lower(), word)
        if dist < min_dist:
            freq = wordfreq.zipf_frequency(word, "en")
            edit_matches = [(freq, word),]
            min_dist = dist
        elif dist == min_dist:
            freq = wordfreq.zipf_frequency(word, "en")
            edit_matches.append((freq, word))
    
    # Sort by frequency (most common first)
    regex_matches.sort(reverse=True)
    edit_matches.sort(reverse=True)
    
    # No matches found - return original
    if len(edit_matches) == 0 and len(regex_matches) == 0:
        return [raw_word]
    
    # Prefer regex matches over edit distance matches
    if len(regex_matches) != 0: 
        case_preserved_matches = [case_restore(word, casing) for _, word in regex_matches]
    else: 
        case_preserved_matches = [case_restore(word, casing) for _, word in edit_matches]
    return case_preserved_matches