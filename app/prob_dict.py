import re
import requests
import os
import math
from collections import Counter

class WordExistenceEvaluator:
    def __init__(self, corpus_url="https://norvig.com/big.txt"):
        self.corpus_path = "big.txt"
        
        # 1. Load/Download Corpus
        if not os.path.exists(self.corpus_path):
            print("Downloading corpus for training...")
            r = requests.get(corpus_url)
            with open(self.corpus_path, 'w', encoding='utf-8') as f:
                f.write(r.text)
        
        with open(self.corpus_path, 'r', encoding='utf-8') as f:
            text = f.read().lower()
            
        # 2. Build Frequency Dictionary
        self.words_count = Counter(re.findall(r'\w+', text))
        self.total_words = sum(self.words_count.values())
        self.vocab_size = len(self.words_count)

    def get_levenshtein(self, s1, s2):
        if len(s1) < len(s2): return self.get_levenshtein(s2, s1)
        if not s2: return len(s1)
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                curr_row.append(min(prev_row[j+1]+1, curr_row[j]+1, prev_row[j]+(c1!=c2)))
            prev_row = curr_row
        return prev_row[-1]

    def compute_existence_probability(self, input_word, length_constraint=None):
        """
        Returns a probability score based on:
        1. Frequency in corpus (if it exists)
        2. Similarity to the nearest valid word (if it doesn't exist)
        """
        target = input_word.lower()
        
        # Hard Constraint: If length doesn't match, probability is effectively 0
        if length_constraint and len(target) != length_constraint:
            return 0.0

        # --- CASE 1: Word exists in dictionary ---
        if target in self.words_count:
            # Standard smoothed probability
            return (self.words_count[target] + 1) / (self.total_words + self.vocab_size)

        # --- CASE 2: Word does NOT exist ---
        # Find the nearest valid word to see "how close" it is to existing
        # For performance, we limit search to common words or words of similar length
        best_dist = float('inf')
        nearest_word_freq = 0
        
        # We only check words of similar length to speed up the loop
        relevant_words = [w for w in self.words_count if abs(len(w) - len(target)) <= 1]
        
        for word in relevant_words:
            dist = self.get_levenshtein(target, word)
            if dist < best_dist:
                best_dist = dist
                nearest_word_freq = self.words_count[word]
            elif dist == best_dist:
                nearest_word_freq = max(nearest_word_freq, self.words_count[word])

        # Formula for non-existent words:
        # We take the probability of the nearest real word and penalize it heavily by distance
        # Each edit distance reduces probability by a factor of 100 (exponential decay)
        nearest_prob = (nearest_word_freq + 1) / (self.total_words + self.vocab_size)
        existence_score = nearest_prob / (100 ** best_dist)
        
        return existence_score

if __name__ == "__main__": 
    # --- Comparison Test ---
    evaluator = WordExistenceEvaluator()

    word_a = "doon"
    word_b = "down"

    prob_a = evaluator.compute_existence_probability(word_a, length_constraint=len(word_a))
    prob_b = evaluator.compute_existence_probability(word_b, length_constraint=len(word_b))

    print(f"Results for Length 4:")
    print(f"P('{word_a}' exists): {prob_a:.10f}")
    print(f"P('{word_b}' exists): {prob_b:.10f}")

    if prob_a > prob_b:
        print(f"\nSuccess: '{word_a}' is more likely to exist than '{word_b}'.")